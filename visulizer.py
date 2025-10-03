import re
from omegaconf import OmegaConf
import os
import numpy as np
from utils import get_subfolders

def load_log_and_config(log_path):
    with open(log_path, "r") as f:
        text = f.read()

    psnr_match = re.search(r"Final average PSNR:\s*([\d\.]+)", text)
    ssim_match = re.search(r"Final average SSIM:\s*([\d\.]+)", text)

    final_psnr = float(psnr_match.group(1)) if psnr_match else None
    final_ssim = float(ssim_match.group(1)) if ssim_match else None

    cfg_match = re.search(r"Config:\s*(.*)", text, re.S)
    if cfg_match:
        cfg_str = cfg_match.group(1)
        cfg = OmegaConf.create(cfg_str)
    else:
        cfg = OmegaConf.create({})

    cfg.final_psnr = final_psnr
    cfg.final_ssim = final_ssim

    return cfg

log_folder = 'logs/'
subdirs = get_subfolders(log_folder)

default_selected = -1
log_dir_selected = subdirs[default_selected]
print(f"{log_dir_selected} is selected for visualization")

all_training_instances_logs = get_subfolders(log_dir_selected)

results_dict = {}

m_list = sorted({int(re.search(r"m:(\d+)", f).group(1)) for f in all_training_instances_logs})
p = 5 


subname = os.path.basename(os.path.normpath(log_dir_selected))

output_dir = os.path.join("visulizer_imgs", subname)
os.makedirs(output_dir, exist_ok=True)

print("output_dir =", output_dir)

for instance_folder in all_training_instances_logs:
    basename = os.path.basename(instance_folder)
    match = re.match(r"method(\d+)_m:(\d+)(?:_seed:(\d+))?", basename)

    if not match:
        continue

    method_idx = int(match.group(1)) - 1 
    m_idx = m_list.index(int(match.group(2)))  
    if match.group(3)!= None:
        seed = int(match.group(3))

    metrics = load_log_and_config(os.path.join(instance_folder, "final_metrics.txt"))

    key = (method_idx, m_idx)
    if key not in results_dict:
        results_dict[key] = []
    
    ssim = metrics["final_ssim"]
    psnr = metrics["final_psnr"]  
    if ssim == None:
        ssim = np.inf    
    if psnr == None:
        psnr = np.inf     
    results_dict[key].append((ssim, psnr))

num_of_methods = 4
methods = [
    "Method1: Binning",
    "Method2: Random (1 kernel)",
    "Method3: Random (p kernels)",
    "Method4: Orthogonal & Negative (p kernels)"
]

cr_list = np.zeros(shape=[num_of_methods, len(m_list)])
for num_m, m in enumerate(m_list):
    cr_list[0, num_m] = (1.0/m)**2
    cr_list[1, num_m] = (1.0/m)**2
    cr_list[2, num_m] = (1.0/m)**2 * p
    cr_list[3, num_m] = (1.0/m)**2 * p

ssim_list = np.zeros(shape=[num_of_methods, len(m_list)]) 
psnr_list = np.zeros(shape=[num_of_methods, len(m_list)])

for (method_idx, m_idx), vals in results_dict.items():
    arr = np.array(vals)  # shape = [num_seeds, 2]
    ssim_mean = arr[:,0].mean()
    psnr_mean = arr[:,1].mean()
    ssim_list[method_idx, m_idx] = ssim_mean
    psnr_list[method_idx, m_idx] = psnr_mean

import matplotlib.pyplot as plt



# === Plot SSIM vs m ===
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

# === Plot PSNR vs m ===
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

# === Plot SSIM vs CR ===
plt.figure(figsize=(8,6))
for i in range(num_of_methods):
    plt.plot(cr_list[i], ssim_list[i], marker="o", label=methods[i])
plt.xlabel("Compression Rate (CR)")
plt.ylabel("SSIM")
plt.title("SSIM vs CR")
plt.xlim(0, 1)  
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ssim_plot_vs_cr.png"), dpi=200)
plt.close()

# === Plot PSNR vs CR ===
plt.figure(figsize=(8,6))
for i in range(num_of_methods):
    plt.plot(cr_list[i], psnr_list[i], marker="o", label=methods[i])
plt.xlabel("Compression Rate (CR)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs CR")
plt.xlim(0, 1) 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "psnr_plot_vs_cr.png"), dpi=200)
plt.close()
