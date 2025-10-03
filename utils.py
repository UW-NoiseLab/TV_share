import os
import logging
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import matplotlib.patheffects as pe
import math

from torchmetrics.functional.image.ssim import structural_similarity_index_measure as tm_ssim
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as tm_psnr

def setup_logger(log_folder, log_file_name='training_log.txt'):
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, log_file_name)
    
    logger = logging.getLogger(str(os.getpid())) 
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]  # clear existing
    return logger

def merge_histories(histories):
    """
    histories: List[Dict[str, Tensor or list or scalar]]
    convert to tensor and dim=0 cat。
    """
    out = {}
    keys = histories[0].keys()
    for k in keys:
        vs = []
        for h in histories:
            v = h[k]
            if isinstance(v, list):
                v = torch.tensor(v, dtype=torch.float32)
            elif torch.is_tensor(v):
                v = v.detach()
            else:
                v = torch.tensor([v], dtype=torch.float32)
            if v.ndim == 0:
                v = v.unsqueeze(0)
            vs.append(v.cpu())
        out[k] = torch.cat(vs, dim=0)
    return out

def show_batch(
    images: torch.Tensor,
    nrow: int = 4,
    pad: int = 4,
    pad_value: float = 1.0,  
    annotate_size: bool = True,
    fontsize: int = 13,
    save_path: str | None = None,
):
    '''
    Save a batch of images in a grid with annotations.
    images: BCHW tensor, values in [0, 1]
    nrow: number of images in each row of the grid
    pad: padding between images
    pad_value: padding value
    '''
    assert images.dim() == 4, "images must be BCHW"
    B, C, H, W = images.shape

    assert C < 3 or C % 3 == 0, "if having more than 3 channels, it has to %3 = 0 so can be divided into multiple graphs"
    if C > 3 and C % 3 == 0:
        num_groups = C // 3
        for g in range(num_groups):
            indexes = [g, g+num_groups, g+num_groups*2]
            # imgs_group = images[:, g*3:(g+1)*3, :, :]
            imgs_group = images[:, indexes, :, :]
            group_path = None
            if save_path:
                base, ext = os.path.splitext(save_path)
                group_path = f"{base}_group{g}{ext}"
            show_batch(
                imgs_group, nrow=nrow, pad=pad, pad_value=pad_value,
                annotate_size=annotate_size, fontsize=fontsize,
                save_path=group_path
            )
        return

    grid = make_grid(
        images.cpu(), nrow=nrow, padding=pad,
        pad_value=pad_value, normalize=True, value_range=(0, 1)
    )

    npimg = grid.permute(1, 2, 0).numpy()  
    fig = plt.figure(figsize=(16, 16))
    ax = plt.axes([0, 0, 1, 1])            
    ax.imshow(npimg)
    ax.axis("off")

    if annotate_size:
        xmaps = min(nrow, B)
        ymaps = math.ceil(B / xmaps)

        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= B:
                    break
                x0 = x * (W + pad) + pad
                y0 = y * (H + pad) + pad

                ax.text(
                    x0 + 3, y0 + 12, f"{W}×{H}, range[{images[k].min().item():.2f},{images[k].max().item():.2f}]",
                    fontsize=fontsize,
                    color="white",
                    ha="left", va="top",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")]
                )
                k += 1

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

def save_psf(psf, output_dir, cfg):
    if psf.ndim == 2:
        # Save single PSF
        plt.imshow(psf, cmap="hot")
        plt.colorbar()
        plt.title(f"PSF: {cfg.forward_config.psf_gen} with params {cfg.forward_config.psf_params}")
        plt.savefig(os.path.join(output_dir, f"psf_{cfg.forward_config.psf_gen}.png"))
        plt.close()

    elif psf.ndim == 3:
        # Save multiple PSF [C, H, W] into one graph
        n = psf.shape[0]
        cols = min(n, 4)                     # max 4 psf per row
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if n > 1 else [axes]

        for i in range(n):
            im = axes[i].imshow(psf[i], cmap="hot")
            axes[i].set_title(f"PSF[{i}]")
            fig.colorbar(im, ax=axes[i])
        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"PSFs: {cfg.forward_config.psf_gen} with params {cfg.forward_config.psf_params}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"psf_{cfg.forward_config.psf_gen}_multi.png"))
        plt.close()

    else:
        raise ValueError(f"Unexpected psf shape {psf.shape}, only 2D or 3D supported")

def ssim_external(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    SSIM
    """
    # torchmetrics
    return tm_ssim(x, y, data_range=data_range, reduction='none')

def psnr_external(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    PSNR 
    """
    # torchmetrics

    return tm_psnr(x, y, data_range=data_range, reduction='none')

import re
def natural_sort_key(s):
    """Helper function to split strings into numeric and non-numeric parts
    for natural sorting (e.g., '_70' < '_110')."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def get_subfolders(base_path, sort_mode="vscode"):
    """
    Get a list of subfolders in the given base path.

    Args:
        base_path (str): Path to the directory.
        sort_mode (str): Sorting mode for subfolders.
            - "vscode" (default): Natural sort order (e.g., '_70' < '_110'),
                                  similar to VSCode Explorer.
            - "lex": Lexicographic order (dictionary order),
                     e.g., '_110' < '_70'.
            - None: No sorting, return in the order provided by os.listdir().

    Returns:
        list[str]: A list of full paths to subdirectories.
    """
    subdirs = [os.path.join(base_path, name) 
               for name in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, name))]

    if sort_mode == "vscode":
        # Natural sorting (numeric-aware)
        subdirs = sorted(subdirs, key=lambda path: natural_sort_key(os.path.basename(path)))
    elif sort_mode == "lex":
        # Pure lexicographic (dictionary) sorting
        subdirs = sorted(subdirs, key=lambda path: os.path.basename(path))
    # If sort_mode is None, keep the original order from os.listdir()

    return subdirs

if __name__ == "__main__":
    psnr_external(1,2)