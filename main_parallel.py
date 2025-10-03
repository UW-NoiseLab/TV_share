import torch
from Dataset import get_dataset
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from ConfigClasses import Config
from main import forward_and_recon
import os, time
import numpy as np
import matplotlib.pyplot as plt

def _build_shared_val(ds_cfg):
    _, val_dataset = get_dataset(ds_cfg)
    loader = DataLoader(val_dataset, batch_size=ds_cfg.batch_size, shuffle=False, num_workers=0)
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x)
        labs.append(y)
    val_imgs = torch.cat(imgs, dim=0).contiguous()
    val_labs = torch.cat(labs, dim=0).contiguous()
    val_imgs.share_memory_()
    val_labs.share_memory_()
    return val_imgs, val_labs

def _worker_run(cfg_dict, val_imgs, val_labs):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    cfg = OmegaConf.create(cfg_dict)
    return forward_and_recon(cfg, val_imgs=val_imgs, val_labs=val_labs)



if __name__ == "__main__":

    import concurrent.futures
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)


    os.makedirs('logs', exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_dir = os.path.join('logs', timestamp)
    os.makedirs(output_dir, exist_ok=True)


    base_cfg = OmegaConf.structured(Config())
    val_imgs, val_labs = _build_shared_val(base_cfg.ds_config)

    p = 5
    m_list = np.arange(start=2, stop=5, step=1)
    seed_list = [17]

    results_dict = {}  # Save all ssim/psnr in all the 3 methods {(method_idx, m_idx): [(ssim, psnr), (ssim, psnr), ...]}

    num_workers = 20
    future_to_idx = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=mp.get_context("spawn")) as ex:
        for num_m, m in enumerate(m_list):
            # ===== Method 1 =====
            cfg1 = OmegaConf.structured(Config())
            cfg1.forward_config.psf_params['shape'] = (1, 1)
            cfg1.forward_config.forward_params['binning'] = tuple(map(int, (m, m)))
            cfg1.output_dir = os.path.join(output_dir, f"method1_m:{m}")
            fut1 = ex.submit(
                _worker_run,
                OmegaConf.to_container(cfg1, resolve=True),
                val_imgs,
                val_labs
            )
            future_to_idx[fut1] = (0, num_m)

            # ===== Method 2 =====
            # Using multiple different seed to init the 
            for seed in seed_list:
                cfg2 = OmegaConf.structured(Config())
                cfg2.forward_config.psf_gen = "random"
                cfg2.forward_config.psf_params['shape'] = tuple(map(int, (m, m)))
                cfg2.forward_config.psf_params['seed'] = seed

                cfg2.forward_config.forward_params['stride'] = int(m)
                cfg2.forward_config.forward_params['binning'] = (1, 1)
                # cfg2.forward_config.forward_params['stride'] = int(1)
                # cfg2.forward_config.forward_params['binning'] = tuple(map(int, (m, m)))
                cfg2.output_dir = os.path.join(output_dir, f"method2_m:{m}_seed:{seed}")
                fut2 = ex.submit(
                    _worker_run,
                    OmegaConf.to_container(cfg2, resolve=True),
                    val_imgs,
                    val_labs
                )
                future_to_idx[fut2] = (1, num_m)

            # ===== Method 3 =====
            for seed in seed_list:
                cfg3 = OmegaConf.structured(Config())
                cfg3.forward_config.psf_gen = "random"
                cfg3.forward_config.psf_params['shape'] = tuple(map(int, (p, m, m)))
                cfg3.forward_config.psf_params['seed'] = seed

                cfg3.forward_config.forward_params['stride'] = int(m)
                cfg3.forward_config.forward_params['binning'] = (1, 1)
                # cfg3.forward_config.forward_params['stride'] = int(1)
                # cfg3.forward_config.forward_params['binning'] = tuple(map(int, (m, m)))
                cfg3.output_dir = os.path.join(output_dir, f"method3_m:{m}_seed:{seed}")
                fut3 = ex.submit(
                    _worker_run,
                    OmegaConf.to_container(cfg3, resolve=True),
                    val_imgs,
                    val_labs
                )
                future_to_idx[fut3] = (2, num_m)
            
            # ===== Method 4 m Orthogonal Prjector Matrix  =====    
            cfg4 = OmegaConf.structured(Config())
            cfg4.forward_config.psf_gen = "orthogonal"
            cfg4.forward_config.psf_params['shape'] = tuple(map(int, (p, m, m)))
            cfg4.forward_config.psf_params['seed'] = 0

            cfg4.forward_config.forward_params['stride'] = int(m)
            cfg4.forward_config.forward_params['binning'] = (1, 1)
            # cfg4.forward_config.forward_params['stride'] = int(1)
            # cfg4.forward_config.forward_params['binning'] = tuple(map(int, (m, m)))
            cfg4.output_dir = os.path.join(output_dir, f"method4_m:{m}")
            fut4 = ex.submit(
                _worker_run,
                OmegaConf.to_container(cfg4, resolve=True),
                val_imgs,
                val_labs
            )
            future_to_idx[fut4] = (3, num_m)


        for fut in concurrent.futures.as_completed(future_to_idx):
            method_idx, m_idx = future_to_idx[fut]
            res = fut.result()  # {'ssim': float, 'psnr': float}
            key = (method_idx, m_idx)
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append((float(res["ssim"]), float(res["psnr"])))

        num_of_methods = 4
        methods = ["Method1: Binning",
           "Method2: Random (1 kernel)",
           "Method3: Random (p kernels)",
           "Method4: Orthogonal & Negative (p kernels)"]

        # Fill in compression ratio for the 3 methods to plot as x later
        cr_list = np.zeros(shape=[num_of_methods, len(m_list)])
        for num_m, m in enumerate(m_list):
            cr_list[0, num_m] = (1.0/m)**2
            cr_list[1, num_m] = (1.0/m)**2
            cr_list[2, num_m] = (1.0/m)**2*p
            cr_list[3, num_m] = (1.0/m)**2*p

        ssim_list = np.zeros(shape=[num_of_methods, len(m_list)]) 
        psnr_list = np.zeros(shape=[num_of_methods, len(m_list)])

        for (method_idx, m_idx), vals in results_dict.items():
            arr = np.array(vals)  # shape = [num_seeds, 2]
            ssim_mean = arr[:,0].mean()
            psnr_mean = arr[:,1].mean()
            ssim_list[method_idx, m_idx] = ssim_mean
            psnr_list[method_idx, m_idx] = psnr_mean
        
        # === Plot SSIM ===
        plt.figure(figsize=(8,6))
        for i in range(num_of_methods):
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
        for i in range(num_of_methods):
            plt.plot(m_list, psnr_list[i], marker="o", label=methods[i])
        plt.xlabel("m")
        plt.ylabel("PSNR (dB)")
        plt.title("PSNR vs m")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "psnr_plot.png"), dpi=200)
        plt.close()

        # === Plot SSIM ===
        plt.figure(figsize=(8,6))
        for i in range(num_of_methods):
            plt.plot(cr_list[i], ssim_list[i], marker="o", label=methods[i])
        plt.xlabel("CR")
        plt.ylabel("SSIM")
        plt.title("SSIM vs CR")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ssim_plot_vs_cr.png"), dpi=200)
        plt.close()

        # === Plot PSNR ===
        plt.figure(figsize=(8,6))
        for i in range(num_of_methods):
            plt.plot(cr_list[i], psnr_list[i], marker="o", label=methods[i])
        plt.xlabel("CR")
        plt.ylabel("PSNR (dB)")
        plt.title("PSNR vs CR")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "psnr_plot_vs_cr.png"), dpi=200)
        plt.close()