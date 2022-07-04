import torch
import ssmd
import re

from torch import Tensor
from torch.distributions import Uniform, Poisson
from numbers import Number
from typing import Union, Tuple


def add_gaussian(tensor: Tensor, std_dev: Union[Number, Tuple[Number, Number]], mean: Number = 0, inplace: bool = False):
    if not inplace:
        tensor = tensor.clone()

    if isinstance(std_dev, (list, tuple)):
        if len(std_dev) == 0:
            std_dev = 0
        elif len(std_dev) == 1:
            std_dev = mean*std_dev[0]/1.4141
        else:
            assert len(std_dev) == 2
            (min_std_dev, max_std_dev) = std_dev
            std_dev = torch.rand(1) * (max_std_dev-min_std_dev) + min_std_dev
            std_dev = mean*std_dev/1.4141
        
    cov_factor = 0.0
    tensor += torch.randn(tensor.shape) * std_dev * (1-cov_factor) ** 0.5
    
    #corr_noise = torch.randn(tensor.shape[1:4]).unsqueeze(0)
    #corr_noise = corr_noise.repeat(tensor.shape[0],1,1,1)
    #corr_noise = corr_noise * torch.rand(tensor.shape[0],1,1,1) * 2 * torch.mean(std_dev) * cov_factor ** 0.5
    #tensor += corr_noise

    return tensor

def add_gaussian_inf(tensor: Tensor, std_dev: Union[Number, Tuple[Number, Number]], mean: Number = 0, inplace: bool = False):
    if not inplace:
        tensor = tensor.clone()
    tensor += 0
    return tensor

def add_style(style: str) -> Tuple[Tensor, Union[Number, Tensor]]:
    # Extract noise type
    noise_type = re.findall(r"[a-zA-Z]+", style)[0]
    params = [p for p in style.replace(noise_type, "").split("_")]
    params = [float(x) for x in params if x != ""]
    return noise_type, params

def add_noise(images, mean, noise_type, params, inplace: bool = False):
    if noise_type == "gauss":
        return add_gaussian(images, params, mean = mean, inplace=inplace)
    elif noise_type == "inf":
        return add_gaussian_inf(images, params, mean = 0, inplace=inplace)
    else:
        raise NotImplementedError("Noise type not supported")
    
    






