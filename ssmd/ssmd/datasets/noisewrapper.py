import h5py
import numpy as np
import torch
import time
from collections import deque
from tqdm import tqdm
from typing import Dict, Tuple
from enum import Enum, auto
import nibabel as nib
from scipy import io
import glob
import pydicom
import os
import torch.fft
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

import ssmd
from ssmd.params import Algorithm
from ssmd.utils.utils import conj_sum



class NoiseWrapper(Dataset):
    INPUT = 0
    REFERENCE = 1
    METADATA = 2
    
    def __init__(
        self,
        file_path: str,
        mask_path: str,
        transform: bool,
        chan_frac: float,
        noise_style: str,
        algorithm: Algorithm,
        training_mode: bool,
        patchsize = None,
        img_count = None,
    ):
        super(NoiseWrapper, self).__init__()
        
        self.file_path = file_path
        self.mask_path = mask_path
        self.transform = transform
        self.chan_frac = chan_frac
        self.noise_style = noise_style
        self.algorithm = algorithm
        self.training_mode = training_mode
        self.patchsize = patchsize 
        
        with h5py.File(self.file_path, "r") as h5file:
            self.shape = h5file["shapes"][0]
            count = h5file["images"].shape[0]    
            
        self.img_count = count if img_count is None else img_count    
        if self.shape[1] == self.patchsize:
            self.transform = False 
        
        if self.algorithm == Algorithm.N2S:
            self.masker = ssmd.utils.n2s.Masker()
            
        if self.algorithm == Algorithm.N2SA:
            self.masker = ssmd.utils.n2sa.Masker()
            
        self.noise_type, self.params = ssmd.utils.noise.add_style(self.noise_style)
        
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Dict]:
           
        with h5py.File(self.file_path, "r") as h5file:
            img = h5file["images"][index]
            img = torch.from_numpy(img).type(torch.float)
            sen = h5file['sens'][index]
            sen = torch.from_numpy(sen).type(torch.float)
        if self.mask_path != None:
            with h5py.File(self.mask_path, "r") as h5file:
                mask = h5file['masks'][index]
                mask = torch.from_numpy(mask).type(torch.float)


        img = img.view(self.shape[0], self.shape[1], -1).permute(2,0,1).view(2,-1,self.shape[0], self.shape[1]).permute(1,0,2,3)
        sen = sen.view(self.shape[0], self.shape[1], -1).permute(2,0,1).view(2,-1,self.shape[0], self.shape[1]).permute(1,0,2,3)
        mean = torch.mean(torch.sqrt(img[:,0,:,:] ** 2 + img[:,1,:,:] **2),[1,2],keepdim=True).unsqueeze(-1)

        if self.transform:
            x_st = np.random.randint(self.shape[0] - self.patchsize)
            y_st = np.random.randint(self.shape[1] - self.patchsize)
            img = self.patcher(img, x_st, y_st)
            sen = self.patcher(sen, x_st, y_st)
        
        if self.mask_path == None:
            mask = torch.zeros([img.shape[2],img.shape[3]])
            #print(mask.shape)
            
        metadata = {}
        metadata[NoiseWrapper.Metadata.MASK] = mask
        return self.prepare_input(img, sen, mean, metadata)

    def image_size(self) -> Tensor:
        return torch.tensor(self.__getitem__(0)[0].shape)

    def __len__(self):
        return self.img_count

    def patcher(self, img, x_st, y_st):
        img = img.clone()
        return img[...,
                   0 + x_st:self.patchsize + x_st,
                   0 + y_st:self.patchsize + y_st]
    
    class Metadata(Enum):
        #IMG_MAX = auto()
        MASK = auto()
        LABEL = auto()
        INSEN = auto()
        LASEN = auto()
        META = auto()
        INP_HAT = auto()
        MASK_COORDS = auto()
        
    def prepare_input(self, 
                      clean: Tensor, 
                      sen: Tensor, 
                      mean: Tensor, 
                      metadata: Dict = {}) -> Tuple[Tensor, Tensor, Dict]:
        
        #np.random.seed(int(time.time() * 1e+6) % 10000)
        #if self.training_mode == True:
        #    scale = np.random.randn()*0.2 + 1
        #    clean = clean * scale
        #    mean = mean * scale
            
        noisy = ssmd.utils.noise.add_noise(clean, mean, self.noise_type, self.params)
        
        
        if self.algorithm == Algorithm.C2C:
            if self.training_mode == True:
                channel = clean.shape[0]
                half_chan = int(channel*self.chan_frac)
                #print(channel)
                np.random.seed(int(time.time() * 1e+6) % 10000)
                arr = np.arange(channel)
                np.random.shuffle(arr)

                in_arr = arr[:half_chan]
                out_arr = arr[half_chan:]

                inimg = clean[in_arr,...]
                innoise, lanoise = noisy[in_arr,...], noisy[out_arr,...]
                insen, lasen = sen[in_arr,...], sen[out_arr,...]

                ref = conj_sum(inimg, insen)
                inp = conj_sum(innoise, insen)
                label = conj_sum(lanoise, lasen)
                insen = conj_sum(insen, insen, broadcast = True)*((1-half_chan/channel)/0.5)
                lasen = conj_sum(lasen, lasen, broadcast = True)*(half_chan/channel/0.5)
                
                #alpha, beta = ssmd.utils.cov_calculator.calculate(noisy,sen,in_arr,out_arr)
                alpha, beta = 0, 1
                #print(alpha,beta)
                
                #metric = ["{}".format(alpha)]
                #metric += ["{}".format(beta)]
                #with open(os.path.join('D:/gate/denoising', "alpha_05.csv"), "a") as f:
                #    f.write(",".join(metric) + "\n")
                    
                iid_label = alpha * inp + beta * label
                iid_lasen = alpha * insen + beta * lasen
                
                inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
                ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)
                insen = torch.sqrt(insen[:1,:,:]**2 + insen[1:,:,:]**2)
                iid_label = torch.sqrt(iid_label[:1,:,:]**2 + iid_label[1:,:,:]**2)
                iid_lasen = torch.sqrt(iid_lasen[:1,:,:]**2 + iid_lasen[1:,:,:]**2)
                
                metadata[NoiseWrapper.Metadata.META] = alpha
                metadata[NoiseWrapper.Metadata.INSEN] = insen                
                metadata[NoiseWrapper.Metadata.LABEL] = iid_label
                metadata[NoiseWrapper.Metadata.LASEN] = iid_lasen
                
            else:
                ref = conj_sum(clean, sen)
                inp = conj_sum(noisy, sen)

                inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
                ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)
                
                #print(inp.shape)
                #metadata[NoiseWrapper.Metadata.INSEN] = clean
                #metadata[NoiseWrapper.Metadata.LABEL] = noisy
                #metadata[NoiseWrapper.Metadata.LASEN] = sen
            
        elif self.algorithm == Algorithm.N2C:
            inp = conj_sum(noisy, sen)
            ref = conj_sum(clean, sen)
            inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
            ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)
            metadata[NoiseWrapper.Metadata.LABEL] = ref
            
        elif self.algorithm == Algorithm.N2N:
            inp = conj_sum(noisy, sen)
            ref = conj_sum(clean, sen)
            noisy2 = ssmd.utils.noise.add_noise(clean, mean, self.noise_type, self.params)
            label = conj_sum(noisy2, sen)
            inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
            ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)
            label = torch.sqrt(label[:1,:,:]**2 + label[1:,:,:]**2)
            metadata[NoiseWrapper.Metadata.LABEL] = label
            
        elif self.algorithm == Algorithm.CS:
            if self.training_mode == True:
                inp = conj_sum(noisy, sen)
                tmp_r = inp[0,:,:] + 1j * inp[1,:,:]
                tmp_k = torch.fft.fftn(tmp_r)
                                            
                
                k_shape = tmp_k.shape
                mask= torch.rand(k_shape)
                mask[mask>0.5]=1
                mask[mask<=0.5]=0
                inp_mask = mask
                ref_mask = (1-mask)
                
                
                AC = int(k_shape[0]*0.1)
                inp_mask[0:AC,0:AC], ref_mask[0:AC,0:AC] = 1, 1
                inp_mask[k_shape[0]-AC+1:k_shape[0],0:AC], ref_mask[k_shape[0]-AC+1:k_shape[0],0:AC] = 1, 1
                inp_mask[0:AC,k_shape[1]-AC+1:k_shape[1]], ref_mask[0:AC,k_shape[1]-AC+1:k_shape[1]] = 1, 1
                inp_mask[k_shape[0]-AC+1:k_shape[0],k_shape[1]-AC+1:k_shape[1]] = 1
                ref_mask[k_shape[0]-AC+1:k_shape[0],k_shape[1]-AC+1:k_shape[1]] = 1
                
                
                inp_k = tmp_k * inp_mask
                ref_k = tmp_k * ref_mask
                
                inp_r = torch.fft.ifftn(inp_k).unsqueeze(0)
                ref_r = torch.fft.ifftn(ref_k).unsqueeze(0)
                
                inp = torch.cat((torch.real(inp_r),torch.imag(inp_r)),0)
                label = torch.cat((torch.real(ref_r),torch.imag(ref_r)),0)
                label = torch.sqrt(label[:1,:,:]**2 + label[1:,:,:]**2)
                metadata[NoiseWrapper.Metadata.LABEL] = label
                
                ref = conj_sum(clean, sen)
            else:
                inp = conj_sum(noisy, sen)
                ref = conj_sum(clean, sen)
            
            inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
            ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)
                
        elif self.algorithm == Algorithm.N2V:
            inp = conj_sum(noisy, sen)
            ref = conj_sum(clean, sen)

            
            if self.training_mode == True:
                label = inp.clone()
                inp, mask_coords = ssmd.utils.n2v_ups.manipulate(inp, 5)
                label = torch.sqrt(label[:1,:,:]**2 + label[1:,:,:]**2)
                metadata[NoiseWrapper.Metadata.LABEL] = label
                metadata[NoiseWrapper.Metadata.MASK_COORDS] = mask_coords
                
            inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
            ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)
            
        elif self.algorithm == Algorithm.N2S:
            inp = conj_sum(noisy, sen)
            ref = conj_sum(clean, sen)  
            
            if self.training_mode == True:
                label = inp
                inp, mask = self.masker.mask(inp)
                label = torch.sqrt(label[:1,:,:]**2 + label[1:,:,:]**2)
                metadata[NoiseWrapper.Metadata.LABEL] = label
                metadata[NoiseWrapper.Metadata.MASK_COORDS] = mask
                
            inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
            ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)
            
        elif self.algorithm == Algorithm.N2SA:
            inp = conj_sum(noisy, sen)
            ref = conj_sum(clean, sen)
            
            if self.training_mode == True:
                label = inp
                _, inp_hat, mask = self.masker.mask(inp)
                label = torch.sqrt(label[:1,:,:]**2 + label[1:,:,:]**2)
                inp_hat = torch.sqrt(inp_hat[:1,:,:]**2 + inp_hat[1:,:,:]**2)
                metadata[NoiseWrapper.Metadata.LABEL] = label  
                metadata[NoiseWrapper.Metadata.INP_HAT] = inp_hat
                metadata[NoiseWrapper.Metadata.MASK_COORDS] = mask
                
            inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
            ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)      
            
        elif self.algorithm == Algorithm.NE2NE:
            inp = conj_sum(noisy, sen)
            ref = conj_sum(clean, sen)
                           
        elif self.algorithm == Algorithm.SSDN:
            inp = self.square_pad(conj_sum(noisy, sen))
            ref = self.square_pad(conj_sum(clean, sen))
            inp = torch.sqrt(inp[:1,:,:]**2 + inp[1:,:,:]**2)
            ref = torch.sqrt(ref[:1,:,:]**2 + ref[1:,:,:]**2)
            
        else:
            raise NotImplementedError("Algorithm not supported")
        
        return (inp, ref, metadata)


    def square_pad(self, image: Tensor) -> Tensor:
        HEIGHT = 1
        WIDTH = 2
        output_size = image.shape
        image_size = list(output_size)
        size = max(image_size[HEIGHT], image_size[WIDTH])
        image_size[HEIGHT] = size
        image_size[WIDTH] = size

        left, top = 0, 0
        right = output_size[WIDTH] - image.shape[WIDTH]
        bottom = output_size[HEIGHT] - image.shape[HEIGHT]
        pad_matrix = [[0, 0]] * 3
        pad_matrix[WIDTH] = [left, right]
        pad_matrix[HEIGHT] = [top, bottom]
        np_padded = np.pad(image, pad_matrix, mode="reflect")
        return torch.tensor(np_padded, device=image.device, requires_grad=image.requires_grad)


    
class NoiseWrapper_dicom(Dataset):
    INPUT = 0
    REFERENCE = 1
    METADATA = 2
    
    def __init__(
        self,
        file_path: str,
        mask_path: str,
        transform: bool,
        chan_frac: float,
        noise_style: str,
        algorithm: Algorithm,
        training_mode: bool,
        patchsize = None,
        img_count = None,
    ):
        super(NoiseWrapper_dicom, self).__init__()
        
        self.file_path = file_path
        self.mask_path = mask_path
        self.transform = transform
        self.chan_frac = chan_frac
        self.noise_style = noise_style
        self.algorithm = algorithm
        self.training_mode = training_mode
        self.patchsize = patchsize 
        
        self.files = glob.glob(self.file_path + '/*.dcm')
        self.img_count = len(self.files)
        

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Dict]:
       
        _dcm = pydicom.read_file(self.files[index])
        _img = _dcm.pixel_array.astype(np.float32)
        img = (_img - _img.min()) / (_img.max()-_img.min())


        img = torch.from_numpy(img).type(torch.float).unsqueeze(0)
        mask = torch.zeros([img.shape[1],img.shape[2]])
        
        metadata = {}
        metadata[NoiseWrapper.Metadata.MASK] = mask
        #metadata[NoiseWrapper.Metadata.IMG_MAX] = self.img_max
        inp = img
        ref = inp
        
        return (inp, ref, metadata)
    

    def image_size(self) -> Tensor:
        return torch.tensor(self.__getitem__(0)[0].shape)

    def __len__(self):
        return self.img_count

    class Metadata(Enum):
        #IMG_MAX = auto()
        MASK = auto()
        LABEL = auto()
        INSEN = auto()
        LASEN = auto()
        META = auto()
        INP_HAT = auto()
        MASK_COORDS = auto()
        

