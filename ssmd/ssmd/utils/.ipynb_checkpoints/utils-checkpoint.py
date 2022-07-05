import numpy as np
import torch
import re
import os
import time
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from typing import Any, List, Dict
from torch import Tensor
from collections import OrderedDict

def calculate_psnr(img: Tensor, ref: Tensor, mask: Tensor):
    #print(img.shape)
    img_abs = img[:,0,...] * mask
    ref_abs = ref[:,0,...] * mask
    #print(torch.mean(img_abs))
    #print(torch.mean(ref_abs))
    #img_abs = torch.sqrt(img[:,1,...] ** 2 + img[:,0,...] ** 2) 
    #ref_abs = torch.sqrt(ref[:,1,...] ** 2 + ref[:,0,...] ** 2) 
    mse = F.mse_loss(img_abs, ref_abs, reduction="none")
   # print(torch.mean(mask))
    mse_shape = mse.shape
    if len(mse_shape) == 5:
        mse = mse.view(mse.shape[0],mse.shape[1], -1).mean(2)
        ref_resize = ref_abs.view(ref_abs.shape[0],ref_abs.shape[1], -1)
        img_max,_ = torch.max(ref_resize,2)

    else:
        mse = mse.view(mse.shape[0], -1).mean(1)
        #print(torch.isnan(mse))
        ref_resize = ref_abs.view(ref_abs.shape[0], -1)
        img_max,_ = torch.max(ref_resize,1)
    return 10 * torch.log10(img_max ** 2 / (mse + 1e-12))

class SSIM_cal(nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):

        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w.to(X.device))
        uy = F.conv2d(Y, self.w.to(X.device))
        uxx = F.conv2d(X * X, self.w.to(X.device))
        uyy = F.conv2d(Y * Y, self.w.to(X.device))
        uxy = F.conv2d(X * Y, self.w.to(X.device))
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return torch.mean(S,[1,2],keepdim=True)
    
ssim_cal = SSIM_cal()

def calculate_ssim(img: Tensor, ref: Tensor, mask: Tensor):
    img_abs = img[:,0,...] * mask
    ref_abs = ref[:,0,...] * mask
    #img_abs = torch.sqrt(img[:,1,...] ** 2 + img[:,0,...] ** 2)#.cpu() 
    #ref_abs = torch.sqrt(ref[:,1,...] ** 2 + ref[:,0,...] ** 2)#.cpu()
    ones = torch.ones(ref_abs.shape[0]).to(ref_abs.device)
    return ssim_cal(img_abs,ref_abs,ones)


def conj_sum(amat: Tensor, bmat: Tensor, broadcast: bool = False, batched: bool = False):
    if batched is False:
        cmat = torch.cat([amat[:,:1,:,:]*bmat[:,:1,:,:] + amat[:,1:,:,:]*bmat[:,1:,:,:],
                          -amat[:,:1,:,:]*bmat[:,1:,:,:] + amat[:,1:,:,:]*bmat[:,:1,:,:]],1)
        if broadcast:
            dmat = torch.sum(cmat,dim=0) * 2
            dmat[1,...] = dmat[0,...]
        else:
            dmat = torch.mean(cmat,dim=0)
        return dmat
    else:
        cmat = torch.cat([amat[:,:,:1,:,:]*bmat[:,:,:1,:,:] + amat[:,:,1:,:,:]*bmat[:,:,1:,:,:],
                          -amat[:,:,:1,:,:]*bmat[:,:,1:,:,:] + amat[:,:,1:,:,:]*bmat[:,:,:1,:,:]],2)
        if broadcast:
            dmat = torch.sum(cmat,dim=1) * 2
            dmat[:,1,...] = dmat[:,0,...]
        else:
            dmat = torch.mean(cmat,dim=1)
        return dmat

def list_constants(clazz: Any, private: bool = False) -> List[Any]:
    variables = [i for i in dir(clazz) if not callable(i)]
    regex = re.compile(r"^{}[A-Z0-9_]*$".format("" if private else "[A-Z]"))
    names = list(filter(regex.match, variables))
    values = [clazz.__dict__[name] for name in names]
    return values


def seconds_to_dhms(seconds: float) -> str:
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // (60 * 60) % 24
    d = seconds // (60 * 60 * 24)
    times = [(d, "d "), (h, "h "), (m, "m "), (s, "s")]
    time_str = ""
    for t, char in times:
        time_str += "{:02}{}".format(int(t), char)
    return time_str


class Metric:
    def __init__(self, batched: bool = True, collapse: bool = True):
        self.reset()
        self.batched = batched
        self.collapse = collapse

    def add(self, value: Tensor):
        n = value.shape[0] if self.batched else 1
        if self.collapse:
            data_start = 1 if self.batched else 0
            mean_dims = list(range(data_start, len(value.shape)))
            if len(mean_dims) > 0:
                value = torch.mean(value, dim=mean_dims)
        if self.batched:
            value = torch.sum(value, dim=0)
        if self.total is None:
            self.total = value
        else:
            self.total += value
        self.n += n

    def __add__(self, value: Tensor):
        self.add(value)
        return self

    def accumulated(self, reset: bool = False):
        if self.n == 0:
            return None
        acc = self.total / self.n
        if reset:
            self.reset()
        return acc

    def reset(self):
        self.total = None
        self.n = 0

    def empty(self) -> bool:
        return self.n == 0


class MetricDict(OrderedDict):
    def __missing__(self, key):
        self[key] = value = Metric()
        return value


def separator(cols=80) -> str:
    return "#" * cols


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
