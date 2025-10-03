# psf_gen.py
import numpy as np

PSF_REGISTRY = {}

def register_psf_gen(name):
    """Decorator to register a psf generator by name"""
    def decorator(fn):
        PSF_REGISTRY[name] = fn
        return fn
    return decorator


@register_psf_gen("gaussian")
def gaussian_psf(shape=(5, 5), sigma=1.0, **kwargs):
    """Gaussian PSF"""
    h, w = shape
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    d = np.sqrt(x * x + y * y)
    gaussian = np.exp(-((d**2) / (2.0 * sigma**2)))
    return gaussian / gaussian.sum()


@register_psf_gen("uniform")
def uniform_psf(shape=(5, 5), **kwargs):
    """Uniform PSF"""
    h, w = shape
    psf = np.ones((h, w))
    return psf / psf.sum()


@register_psf_gen("motion")
def motion_psf(shape=(5, 5), angle=0.0, length=5, **kwargs):
    """Motion PSF"""
    h, w = shape
    psf = np.zeros((h, w))
    center = (h // 2, w // 2)

    theta = np.deg2rad(angle)
    dx, dy = np.cos(theta), np.sin(theta)
    for i in range(length):
        x = int(center[0] + dx * i)
        y = int(center[1] + dy * i)
        if 0 <= x < h and 0 <= y < w:
            psf[x, y] = 1.0
    return psf / psf.sum()


@register_psf_gen("random")
def random_psf(shape=(5, 5), **kwargs):
    """Random PSF"""
    seed = kwargs.get("seed", 0)
    low = kwargs.get("low", 0)
    high = kwargs.get("high", 100)
    np.random.seed(seed)
    psf = np.random.uniform(low=low, high=high, size=shape)

    if psf.ndim == 3:
        for i in range(len(psf)):
            psf[i] = psf[i]/psf[i].sum()
    elif psf.ndim == 2:
        psf = psf / psf.sum()
        
    return psf

@register_psf_gen("orthogonal")
def orthogonal_psf(shape=(3, 8, 8), **kwargs):
    """
    Generate PSFs from 2D DCT (type-II) separable bases. Channels are L2-orthonormal by default.
    Optional L1 normalization (sum==1) can be applied per channel (breaks orthogonality).
    If C > H*W, extra channels are filled with random noise (normalized).

    Args:
        shape: (C, H, W)
        seed (int, optional): RNG seed for basis order / random fill.
        dtype (np.dtype, optional): output dtype (default np.float32).
        l1_normalize (bool, optional): if True, enforce per-channel sum==1 (breaks orthogonality).

    Returns:
        psf: np.ndarray of shape (C, H, W)
    """
    import numpy as np

    # --------- parse ---------
    if not (len(shape) == 3):
        raise ValueError("shape must be (C, H, W)")
    C, H, W = shape
    N = H * W

    seed = kwargs.get("seed", 0)
    dtype = kwargs.get("dtype", np.float32)
    l1_normalize = bool(kwargs.get("l1_normalize", False))

    rng = np.random.default_rng(seed)

    # --------- 1D DCT-II orthonormal bases ---------
    def dct_1d_basis(n: int) -> np.ndarray:
        """Return (n, n) orthonormal DCT-II basis; row k is k-th unit-norm basis."""
        i = np.arange(n)[None, :]     # (1, n)
        k = np.arange(n)[:, None]     # (n, 1)
        B = np.sqrt(2.0 / n) * np.cos(np.pi / n * (i + 0.5) * k)
        B[0, :] = np.sqrt(1.0 / n)    # DC row
        return B

    Bh = dct_1d_basis(H)  # (H, H)
    Bw = dct_1d_basis(W)  # (W, W)

    # Enumerate 2D separable bases (kh, kw)
    pairs = []
    for s in range(H + W - 1):
        for kh in range(s, -1, -1):
            for kw in [s - kh]:
                if kh < H and kw < W:
                    pairs.append((kh, kw))
                    
    # rng.shuffle(pairs)  # randomize order for variety

    # How many true DCT bases we can provide
    n_bases = min(C, N)

    # Build the first n_bases channels from DCT outer products (each has L2=1)
    psf = np.empty((C, H, W), dtype=dtype)
    for c, (kh, kw) in enumerate(pairs[:n_bases]):
        psf[c] = np.outer(Bh[kh], Bw[kw]).astype(dtype, copy=False)

    # If C > N: fill remaining channels with random noise (cannot keep orthogonality)
    if C > N:
        print(f"Require number of othogonal psf is too many. Filling the rest with random matrix")
        extra = rng.standard_normal(size=(C - N, H, W)).astype(dtype, copy=False)

        if l1_normalize:
            # L1 normalize to sum==1 (shift if sum~0 to avoid division by zero)
            for i in range(extra.shape[0]):
                s = extra[i].sum()
                if abs(s) < 1e-12:
                    x = extra[i] - extra[i].min() + 1e-12  # make positive-ish
                    s = x.sum()
                    extra[i] = x / (s if s != 0 else (H * W))
                else:
                    extra[i] = extra[i] / s
        else:
            # L2 normalize
            flat = extra.reshape(extra.shape[0], -1)
            norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
            flat /= norms
            extra = flat.reshape(extra.shape[0], H, W).astype(dtype, copy=False)

        psf[n_bases:] = extra

    # Optional L1 normalization for all channels
    if l1_normalize:
        for c in range(n_bases):
            s = psf[c].sum()
            if abs(s) < 1e-12:
                x = psf[c] - psf[c].min() + 1e-12  # shift to avoid zero-sum
                s = x.sum()
                psf[c] = x / (s if s != 0 else (H * W))
            else:
                psf[c] = psf[c] / s

    return psf


if __name__ == "__main__":
    import torch
    from utils import save_psf
    from ConfigClasses import Config
    import os
    cfg = Config()
    output_dir = r'./psf_inspect'
    os.makedirs(output_dir, exist_ok=True)
    psf = orthogonal_psf(shape=(30, 4, 4), seed=0)
    psf = torch.tensor(psf, dtype=torch.float32)

    save_psf(psf, output_dir=output_dir, cfg=cfg)