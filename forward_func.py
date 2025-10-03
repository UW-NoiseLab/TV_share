import torch
import torch.nn.functional as F
import numpy as np

FORWARD_FUNC_REGISTRY = {}

def register_forward_func(name):
    """Decorator to register a forward function by name"""
    def decorator(fn):
        FORWARD_FUNC_REGISTRY[name] = fn
        return fn
    return decorator

@register_forward_func("conv2d")
def conv2d_forward(x: torch.Tensor, psf: torch.Tensor, **kwargs):
    """2D Convolution Forward Model
    Aruguments:
        x: input image, [H,W] or [C,H,W] or [B,C,H,W]
        psf: point spread function, [H,W] or [C,H,W]
        kwargs: optional arguments
    """
    device = kwargs.get("device", "cpu")
    dtype = kwargs.get("dtype", torch.float32)
    stride = kwargs.get("stride", 1)
    # Whether to repeat psf in shape[0] even if img_channel can divide it, like if psf: [3, H, W] and img: [3, H, W],
    # true will repeat psf to [9, H, W] and output 9 channels, while false make output to be 3 channels
    dup_psf_channels = kwargs.get("dup_psf_channels", True)

    x = x.to(device=device, dtype=dtype)  
    psf = psf.to(device=device, dtype=dtype)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() == 4:
        x = x
    else:
        raise ValueError("x must be 2D, 3D, or 4D tensor or ndarray.")
    img_channels = x.shape[1]
    # psf -> torch [1,1,H,W]
    if psf.dim() == 2:
        psf = psf.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif psf.dim() == 3:
        psf = psf.unsqueeze(1)      # [C,1,H,W]
    else:
        raise ValueError("psf must be 2D or 3D tensor or ndarray.")
    
    # Duplicate PSF to adapt to method 3
    if psf.shape[0] % img_channels == 0:
        if dup_psf_channels:
            psf = psf.repeat(img_channels,1,1,1)
            # psf = psf.repeat_interleave(img_channels, dim=0)
    else:
        psf = psf.repeat(img_channels,1,1,1)
        # psf = psf.repeat_interleave(img_channels, dim=0)

    y_pred = F.conv2d(x, psf, padding="valid", groups=img_channels, stride=stride)

    if kwargs.get("photons_per_unit", None):
        # Warning: this may cut off gradient, which affects reconstruction
        snr = float(kwargs.get("photons_per_unit", 25.0)) 
        lam = y_pred * snr
        # noise = torch.poisson(lam) / snr - y_nonneg
        # y_pred = y_nonneg + noise
        y_pred = torch.poisson(lam) / snr
    
    binning = kwargs.get("binning", None)  # e.g., (by, bx)
    if binning is not None:
        by, bx = binning
        if not (isinstance(by, int) and isinstance(bx, int) and by > 0 and bx > 0):
            raise ValueError("binning must be a tuple of positive ints, e.g., (2,2).")
        y_pred = F.avg_pool2d(y_pred, kernel_size=(by, bx), stride=(by, bx), ceil_mode=False)
    return y_pred             
