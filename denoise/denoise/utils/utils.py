"""
# Description:
#  Utilities
#
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
"""

import numpy as np
import re
import os
import time
import random
from random import sample

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

def seconds_to_dhms(seconds: float):
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // (60 * 60) % 1000
    time_str = "{:03}{}".format(int(h), "h ")
    time_str += "{:02}{}".format(int(m), "m ")
    time_str += "{:02}{}".format(int(s), "s")
    return time_str

def separator(cols=80):
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


class SSIM_cal(nn.Module):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):

        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        #X = X.unsqueeze(1)
        #Y = Y.unsqueeze(1)
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

        return torch.mean(S,[2,3],keepdim=False)
    
ssim_cal = SSIM_cal()

def calculate_ssim(img, ref, mask):
    mask = mask.to(img.device)
    
    if img.shape[1] == 2:
        img = torch.sqrt(img[:,:1,...] ** 2 + img[:,1:,...] ** 2) 
        ref = torch.sqrt(ref[:,:1,...] ** 2 + ref[:,1:,...] ** 2) 
    
    img = img * mask
    ref = ref * mask

    ones = torch.ones(ref.shape[0]).to(ref.device)
    ssim = ssim_cal(img,ref * mask,ones)

    return ssim

def calculate_psnr(img, ref, mask):
    mask = mask.to(img.device)

    if img.shape[1] == 2:
        img = torch.sqrt(img[:,:1,...] ** 2 + img[:,1:,...] ** 2) 
        ref = torch.sqrt(ref[:,:1,...] ** 2 + ref[:,1:,...] ** 2) 
    
    img = img * mask
    ref = ref * mask
    mse = F.mse_loss(img, ref, reduction="none")

    if len(mse.shape) == 4:
        mse = mse.view(mse.shape[0], -1).mean(1)
        ref_resize = ref.view(ref.shape[0], -1)
        img_max,_ = torch.max(ref_resize,1)
    else:
        print('not implemented psnr')

    psnr = 10 * torch.log10(img_max ** 2 / (mse + 1e-12))

    return psnr

### Unused ###
class resampler():
    def __init__(self, dataset, len_a, len_b, factor, num_workers, batch_size):
        self.dataset = dataset
        self.len_a = len_a
        self.len_b = len_b
        self.len_ab = np.array([len_a,len_b])
        self.min_len = self.len_ab.min()
        self.arg_min_len = self.len_ab.argmin()
        self.factor = factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.list_a = list(range(0, self.len_a))
        self.list_b = list(range(self.len_a, self.len_a+self.len_b))

    def info(self):
        
        if self.arg_min_len == 0:
            return (self.len_a, self.min_len * self.factor, self.len_a + self.min_len * self.factor) 
        else: 
            return (self.min_len * self.factor, self.len_b, self.len_b + self.min_len * self.factor) 
        
    def resample(self):
        if self.arg_min_len == 0:
            list_sample = sample(self.list_b,self.len_a * self.factor)
            list_tot = self.list_a + list_sample
        else:
            list_sample = sample(self.list_a,self.len_b * self.factor)
            list_tot = list_sample + self.list_b
            
        random.shuffle(list_tot)

        data_loader = DataLoader(
              self.dataset,
              batch_size = self.batch_size,
              shuffle = False,
              sampler = SubsetRandomSampler(list_tot),
              pin_memory = True,
              num_workers = self.num_workers)  
        
        return data_loader
    

class auc_calculator():
    def __init__(self):
        self.reset()

    def reset(self):        
        self.threshs = np.arange(0.,1.006,0.005)
        self.sensitivities = np.zeros_like(np.array(self.threshs))
        self.specificities = np.zeros_like(np.array(self.threshs))
        self.FNs = np.zeros_like(np.array(self.threshs))
        self.FPs = np.zeros_like(np.array(self.threshs))
        self.TPs = np.zeros_like(np.array(self.threshs))
        self.TNs = np.zeros_like(np.array(self.threshs))  
    
    def add(self, predX_val, Y_val):
        score = torch.nn.functional.softmax(predX_val, dim=1)[0,1]
        preds = torch.zeros((len(self.threshs),))

        for j in range(len(self.threshs)):
            if score.to('cpu') >= torch.tensor([self.threshs[j]]): pred_ = torch.tensor([1])
            else: pred_ = torch.tensor([0])

            preds[j] = pred_
        target_mat = Y_val.to('cpu') * torch.ones_like(preds)
        correct_mat = preds.eq(target_mat)

        ### Saving the results for each cases (FN, TN, FP, TP) ###
        if Y_val.to('cpu').item() == 0:
            self.TNs += correct_mat.type(torch.DoubleTensor).numpy()
            self.FPs += 1-correct_mat.type(torch.DoubleTensor).numpy()
        else:
            self.TPs += correct_mat.type(torch.DoubleTensor).numpy()
            self.FNs += 1 - correct_mat.type(torch.DoubleTensor).numpy()
        
    def info(self):
        self.sensitivities = (self.TPs) / (self.TPs + self.FNs)
        self.specificities = (self.TNs) / (self.TNs + self.FPs)
        return (self.sensitivities, self.specificities)
    
    def eval(self):
        self.sensitivities = (self.TPs) / (self.TPs + self.FNs)
        self.specificities = (self.TNs) / (self.TNs + self.FPs)
        avg_sensitivity = (self.sensitivities[:-1]+self.sensitivities[1:])/2
        diff_specificity = np.abs(self.specificities[:-1]-self.specificities[1:])
        
        return np.sum( avg_sensitivity*diff_specificity )  

    
def calculate_acc(pred, label):
    with torch.no_grad():
        __, Pred_label = torch.max(pred, 1)
    acc = (Pred_label == label)
    return acc