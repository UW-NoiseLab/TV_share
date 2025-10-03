from ConfigClasses import Config
from omegaconf import OmegaConf
import os
from Dataset import get_dataset

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from psf_gen import PSF_REGISTRY
from forward_func import FORWARD_FUNC_REGISTRY

from utils import show_batch, merge_histories, ssim_external, psnr_external, save_psf
from tv import reconstruct_tv

import time

def forward_and_recon(cfg, val_imgs=None, val_labs=None):
    '''
    Forward and reconstruct the image to get the metric of TV under current cfg
    Arguments:
        cfg: whole pipline's configuration
        val_imgs: (torch.Tensor) validation images, default is None to let the function get the dataset
        val_imgs: (torch.Tensor) validation labels, default is None to let the function get the dataset
    '''
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if (val_imgs is not None) and (val_labs is not None):
        tensor_ds = torch.utils.data.TensorDataset(val_imgs, val_labs)
        val_dataloader = DataLoader(
            tensor_ds,
            batch_size=cfg.ds_config.batch_size,
            shuffle=True,
            num_workers=0
        )
    else:
        _, val_dataset = get_dataset(cfg.ds_config)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.ds_config.batch_size,
            shuffle=True,
            num_workers=0
        )

    psf = PSF_REGISTRY[cfg.forward_config.psf_gen](**cfg.forward_config.psf_params)
    psf = torch.tensor(psf, dtype=torch.float32)

    save_psf(psf, output_dir=output_dir, cfg=cfg)

    all_histories = []

    for i, (img, lab) in enumerate(val_dataloader):   
        
        if i >= cfg.ds_config.num_batch_used:
            break    
        forward = FORWARD_FUNC_REGISTRY[cfg.forward_config.forward_model]
        y_pred = forward(img, psf, **cfg.forward_config.forward_params)
           
        x_init = F.interpolate(
            y_pred[:, 0:3, :, :], 
            size=img.shape[-2:],
            mode='bilinear', 
            align_corners=False
        )
        
        x_rec, hist = reconstruct_tv(
            x_init=x_init,              
            y_pred=y_pred,
            forward_fun=lambda x, **kw: forward(x, psf, **kw),
            forward_kwargs=cfg.forward_config.forward_params,
            **cfg.tv_config
        )

        x_for_metric = x_rec.clamp(0, 1)   
        gt_for_metric = img.clamp(0, 1)
        with torch.no_grad():
            hist["psnr"] = psnr_external(x_for_metric, gt_for_metric, data_range=1.0).cpu()  # [B]
            hist["ssim"] = ssim_external(x_for_metric, gt_for_metric, data_range=1.0).cpu()  # [B]


        all_histories.append(hist)
        
        if i == 0:

            show_batch(img, nrow=4, save_path=f"{output_dir}/input_batch.png")
            print(f"Train sample {i}: img shape {img.shape}, label {lab}")

            show_batch(y_pred, nrow=4, save_path=f"{output_dir}/observed_batch.png")
            plt.close()
      
            show_batch(x_rec, nrow=4, save_path=f"{output_dir}/reconstructed_batch.png")
            plt.close()

            plt.plot(hist["total"], label="total")
            plt.plot(hist["data"],  label="data")
            plt.plot(hist["tv"] * cfg.tv_config.tv_lambda, label=f"tv_lam{cfg.tv_config.tv_lambda}")
            plt.legend(); plt.xlabel("iter"); plt.ylabel("loss")
            plt.savefig(f"{output_dir}/recon_loss_{i}.png")
            plt.close()
                
    final_history = merge_histories(all_histories) 
    final_history['psnr'] = final_history['psnr'].mean().item()
    final_history['ssim'] = final_history['ssim'].mean().item()

    final_history_list = {}
    for k, v in final_history.items():
        if torch.is_tensor(v):
            final_history_list[k] = v.tolist()
        else:
            final_history_list[k] = v
    final_history_list = OmegaConf.create(final_history_list)   
    OmegaConf.save(final_history_list, f"{output_dir}/final_history.yaml") 

    
    print("Final average PSNR: %.4f, SSIM: %.4f" % (final_history['psnr'], final_history['ssim']))
    with open(f"{output_dir}/final_metrics.txt", "w") as f:
        f.write(f"Final average PSNR: {final_history['psnr']:.4f}\n")
        f.write(f"Final average SSIM: {final_history['ssim']:.4f}\n")
        f.write(f"Config: {OmegaConf.to_yaml(cfg)}\n")

    return {
        "ssim": final_history["ssim"],
        "psnr": final_history["psnr"]
    }

if __name__ == "__main__":

    os.makedirs('logs', exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_dir = os.path.join('logs', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
 
    p = 5

    m_list = np.arange(start=2, stop=10, step=1)

    ssim_list = np.zeros(shape=[3, len(m_list)])
    psnr_list = np.zeros(shape=[3, len(m_list)])

    for num_m, m in enumerate(m_list):

        # Test below 3 methods
        # 1. Binning to create a (N/m, N/m) image and reconstruct
        output_dir_method1 = os.path.join(output_dir, f"method1_m:{m}")
        cfg = OmegaConf.structured(Config())
        cfg.forward_config.psf_params['shape'] = (1, 1)
        cfg.forward_config.forward_params['binning'] = tuple(map(int, (m, m)))
        cfg.output_dir = output_dir_method1
        history = forward_and_recon(cfg)
        ssim_list[0, num_m] = history['ssim']
        psnr_list[0, num_m] = history['psnr']

        # 2. Point by point multiple by a (m, m) random matrix to get a (N/m, N/m) image and reconstruct
        output_dir_method2 = os.path.join(output_dir, f"method2_m:{m}")
        cfg = OmegaConf.structured(Config())
        cfg.forward_config.psf_gen = "random"
        cfg.forward_config.psf_params['shape'] = tuple(map(int, (m, m)))
        cfg.forward_config.psf_params['seed'] = 0

        # cfg.forward_config.forward_params['stride'] = int(m)
        # cfg.forward_config.forward_params['binning'] = (1, 1)
        cfg.forward_config.forward_params['stride'] = int(1)
        cfg.forward_config.forward_params['binning'] = tuple(map(int, (m, m)))
        cfg.output_dir = output_dir_method2
        history = forward_and_recon(cfg)
        ssim_list[1, num_m] = history['ssim']
        psnr_list[1, num_m] = history['psnr']


        # 3. Same as 2, but with p number of (m, m) random matrix to get p number of (N/m, N/m) image and reconstruct
        output_dir_method3 = os.path.join(output_dir, f"method3_m:{m}")
        cfg = OmegaConf.structured(Config())
        cfg.forward_config.psf_gen = "random"
        cfg.forward_config.psf_params['shape'] = tuple(map(int, (p, m, m)))
        cfg.forward_config.psf_params['seed'] = 0
        
        # cfg.forward_config.forward_params['stride'] = int(m)
        # cfg.forward_config.forward_params['binning'] = (1, 1)
        cfg.forward_config.forward_params['stride'] = int(1)
        cfg.forward_config.forward_params['binning'] = tuple(map(int, (m, m)))
        cfg.output_dir = output_dir_method3
        history = forward_and_recon(cfg)
        ssim_list[2, num_m] = history['ssim']
        psnr_list[2, num_m] = history['psnr']
    
    methods = ["Method1: Binning",
           "Method2: Random (1 kernel)",
           "Method3: Random (p kernels)"]
    
    # === Plot SSIM ===
    plt.figure(figsize=(8,6))
    for i in range(3):
        plt.plot(m_list, ssim_list[i], marker="o", label=methods[i])
    plt.xlabel("m")
    plt.ylabel("SSIM")
    plt.title("SSIM vs m")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ssim_plot.png"), dpi=200)
    plt.close()

    # === Plot PSNR ===
    plt.figure(figsize=(8,6))
    for i in range(3):
        plt.plot(m_list, psnr_list[i], marker="o", label=methods[i])
    plt.xlabel("m")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs m")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psnr_plot.png"), dpi=200)
    plt.close()

    
    
    
    