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

from utils import show_batch, merge_histories, ssim_external, psnr_external
from tv import reconstruct_tv


if __name__ == "__main__":
    output_dir = r"./output_batch"
    os.makedirs(output_dir, exist_ok=True)
    
    cfg = OmegaConf.structured(Config())
    
    _, val_dataset = get_dataset(cfg.ds_config)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg.ds_config.batch_size, shuffle=True, num_workers=0)
    
    psf = PSF_REGISTRY[cfg.forward_config.psf_gen](**cfg.forward_config.psf_params)
    
    psf = torch.tensor(psf, dtype=torch.float32)
    
    plt.imshow(psf, cmap='hot')
    plt.colorbar()
    plt.title(f"PSF: {cfg.forward_config.psf_gen} with params {cfg.forward_config.psf_params}")
    plt.savefig(f"{output_dir}/psf_{cfg.forward_config.psf_gen}.png")
    plt.close()

    all_histories = []

    for i, (img, lab) in enumerate(val_dataloader):   
        
        if i >= 4:
            break    
        forward = FORWARD_FUNC_REGISTRY[cfg.forward_config.forward_model]
        y_pred = forward(img, psf, **cfg.forward_config.forward_params)
           
        x_init = F.interpolate(
            y_pred, 
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

            show_batch(img, nrow=4)
            plt.title("A batch of input images")
            plt.savefig(f"{output_dir}/input_batch.png")
            plt.close()
            print(f"Train sample {i}: img shape {img.shape}, label {lab}")

            show_batch(y_pred, nrow=4)
            plt.title("A batch of observed images")
            plt.savefig(f"{output_dir}/observed_batch.png")
            plt.close()
      
            show_batch(x_rec, nrow=4)
            plt.title("A batch of reconstructed images")
            plt.savefig(f"{output_dir}/reconstructed_batch.png")
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
    
    
    
    