# configclass.py
from dataclasses import dataclass, field
from typing import Dict, Any
from omegaconf import OmegaConf

@dataclass
class DatasetConfig:
    name: str = "ariamp4"  # "celeba_hq", "ariamp4"
    batch_size: int = 16
    img_size: int = 256  # images will be resized to (img_size,img_size)
    num_batch_used: int = 2

@dataclass
class ForwardConfig:
    psf_gen: str = 'gaussian'  # gaussian, uniform, motion, random
    psf_params: Dict[str, Any] = field(default_factory=lambda: {
        "shape": (5,5), # for all
        "sigma": 1.0,   # for gaussian
    })

    forward_model: str = "conv2d"
    forward_params: Dict[str, Any] = field(default_factory=lambda: {
        # "photons_per_unit": 500.0,  # None or float, e.g., 25.0
        # convolution part
        "stride" : 1,
        # avgpool2d part
        "binning": (2, 2),           # None or (by, bx), e.g., (2,2)
    })
    
    
@dataclass
class TVConfig:
    tv_lambda: float = 0.0
    iters: int = 1000
    lr: float = 1e-1
    tv_eps: float = 1e-3
    verbose: bool = True
    device: str = 'cpu'
    patience: int = 20
    delta_tol: float = 0.05
    bound_method: str = "clamp"  # ["clamp", "sigmoid"]

    
@dataclass
class Config:
    ds_config: DatasetConfig = field(default_factory=DatasetConfig)
    forward_config: ForwardConfig = field(default_factory=ForwardConfig)
    tv_config: TVConfig = field(default_factory=TVConfig)
    output_dir: str = "Dummy_dir"
    
if __name__ == "__main__":
    base = OmegaConf.structured(Config())
    print((base.ds_config))  # DictConfig
 