"""
# Description:
#  Datawrapper
#
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
"""

import os
from scipy import io
import numpy as np
from tqdm import tqdm
import csv
import glob
import time
from PIL import Image
import h5py
import pydicom
import mat73
import torch
import torch.fft
import torch.nn.functional as F
import torchvision.transforms.functional as trans_F
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from utils.noise import noise_adder


class DataWrapper(Dataset):
    Input = 0
    Refer = 1
    Name = 2
    Mask = 3
    Insen = 4
    Label = 5
    Lasen = 6
    
    def __init__(self, 
                 file_path, 
                 mask_path,
                 args, 
                 training_mode):
        
        super(DataWrapper, self).__init__()
        self.file_path = file_path
        self.mask_path = mask_path
        self.training_mode = training_mode
        self.args = args
        self.noise_adder = noise_adder(self.args.noise)
        
        
        if self.args.data_type == '*.h5':
            with h5py.File(self.file_path, "r") as h5file:
                self.shape = h5file["shapes"][0]
                self.img_cnt = h5file["images"].shape[0]    
        else:
        
            self.data_hierarchy = 1
            total_list = []
            for file_path in self.file_path:
                for class_names in ({''}):
                    for dir_names in ({'act_tb/','sta_tb/','abnormal/','others/','sus_tb/','normal/'}):
                        data_path = file_path + class_names + dir_names
                        files = glob.glob(data_path + self.args.data_type)
                        for _file in files:
                            total_list.append(os.path.join(_file))                      

            self.file_list = total_list
        
    def __getitem__(self, idx: int):

        if self.args.data_type == '*.h5':
            with h5py.File(self.file_path, "r") as h5file:
                _img = h5file["images"][idx]
                _img = torch.from_numpy(_img).type(torch.float)
                _sen = h5file['sens'][idx]
                _sen = torch.from_numpy(_sen).type(torch.float)
                _img = _img.view(self.shape[0], self.shape[1], -1).permute(2,0,1)
                _img = _img.view(2,-1,self.shape[0], self.shape[1]).permute(1,0,2,3)
                _sen = _sen.view(self.shape[0], self.shape[1], -1).permute(2,0,1)
                _sen = _sen.view(2,-1,self.shape[0], self.shape[1]).permute(1,0,2,3)
            _name = str(idx) + '.h5'
            
            if self.training_mode == True: 
                np.random.seed(int(time.time() * 1e+6) % 10000)

                scale = np.random.randn()*0.2 + 1
                _img = _img * scale        
            
            ## retrospective processing ##
            #_combine = self.conj_sum(_img, _sen)*_img.shape[0]
            #_ret_img = self.com_mul(_combine.unsqueeze(0), _sen)
            #_img = _ret_img
            
            
            mean = torch.mean(torch.sqrt(_img[:,0,:,:] ** 2 + _img[:,1,:,:] **2),[1,2],keepdim=True).unsqueeze(-1)
            _noisy = self.noise_adder.add(_img,mean)
            
            if self.mask_path != None:
                with h5py.File(self.mask_path, "r") as h5file:
                    _mask = h5file['masks'][idx]
                    _mask = torch.from_numpy(_mask).type(torch.float)
                    _mask = _mask.unsqueeze(0)
            else:
                _mask = torch.zeros([1,_img.shape[2],_img.shape[3]])
            
            
        else:
            _name = self.file_list[idx].split('/')[-(self.data_hierarchy+1):]

            ## data loading ##
            if self.args.data_type == '*.mat':
                _img = np.array(io.loadmat(self.file_list[idx])['img'],dtype=float)
                _img = torch.from_numpy(_img).type(torch.float)

            elif _file[-4:].lower() =='*.png':
                _img = np.array(Image.open(self.file_list[idx]))/255

            elif _file[-4:].lower() == '*.dcm':
                _dcm = pydicom.read_file(self.file_list[idx])
                _img = _dcm.pixel_array.astype(np.float32)
                _img = (_img / (2 ** _dcm.BitsStored))
                if int(_dcm.PhotometricInterpretation[-1]) == 1:
                    _img = 1.0 - _img

            else:
                raise('Data type is not supported')
                
            _img = _img.unsqueeze(0)
            

            
        
        if self.training_mode == True:  
            if self.args.data_type == '*.h5':
                channel = _img.shape[0]
                half_chan = int(channel*0.5)

                np.random.seed(int(time.time() * 1e+6) % 10000)
                arr = np.arange(channel)
                np.random.shuffle(arr)

                in_arr = arr[:half_chan]
                out_arr = arr[half_chan:]

                inimg = _img[in_arr,...]
                innoise, lanoise = _noisy[in_arr,...], _noisy[out_arr,...]
                insen, lasen = _sen[in_arr,...], _sen[out_arr,...]

                _ref = self.conj_sum(inimg, insen)
                _inp = self.conj_sum(innoise, insen)
                _label = self.conj_sum(lanoise, lasen)
                _insen = self.conj_sum(insen, insen, broadcast = True)
                _lasen = self.conj_sum(lasen, lasen, broadcast = True)

                #alpha, beta = 0, 1
                    
                #iid_label = alpha * inp + beta * label
                #iid_lasen = alpha * insen + beta * lasen
                
                ### magnitude ###
                #_inp = torch.sqrt(_inp[:1,:,:]**2 + _inp[1:,:,:]**2)
                #_ref = torch.sqrt(_ref[:1,:,:]**2 + _ref[1:,:,:]**2)
                #_insen = torch.sqrt(_insen[:1,:,:]**2 + _insen[1:,:,:]**2)
                #_label = torch.sqrt(_label[:1,:,:]**2 + _label[1:,:,:]**2)
                #_lasen = torch.sqrt(_lasen[:1,:,:]**2 + _lasen[1:,:,:]**2)

                
            else:
                np.random.seed(int(time.time() * 1e+6) % 10000)
                _img = trans_F.resize(_img.unsqueeze(0),size=(self.args.image_size,self.args.image_size)).squeeze()

                img_shape = int(_img.shape[-1]/2)
                _lab = trans_F.resize((1-mask) * _img,size=(img_shape,img_shape)) * 4/(4-mask_f)
                _inp = trans_F.resize(mask * _img,size=(img_shape,img_shape)) * 4/(mask_f)    
                #print(_img.shape)

            return (_inp,
                    _ref,
                    _name,
                    _mask,
                    _insen,
                    _label,
                    _lasen)  

        
        else:
            if self.args.data_type == '*.h5':
                _inp = self.conj_sum(_noisy.clone(), _sen)
                _lab = self.conj_sum(_img.clone(), _sen)
                
                
                ### magnitude ###
                #_inp = torch.sqrt(_inp[:1,:,:]**2 + _inp[1:,:,:]**2)
                #_lab = torch.sqrt(_lab[:1,:,:]**2 + _lab[1:,:,:]**2)

            else:
                _inp = trans_F.resize(_img,size=(img_shape,img_shape)) 
                _lab = _img
            
            return (_inp,
                    _lab,
                    _name,
                    _mask)  

    def image_size(self):
        return torch.tensor(self.__getitem__(0)[0].shape)

    def __len__(self):
        if self.args.data_type == '*.h5':
            return self.img_cnt
        else:
            return len(self.file_list)
    
    def conj_sum(self, amat, bmat, broadcast = False, batched = False):
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
        
    def com_mul(self, amat, bmat, batched = False):
        if batched is False:
            cmat = torch.cat([amat[:,:1,:,:]*bmat[:,:1,:,:] - amat[:,1:,:,:]*bmat[:,1:,:,:],
                              + amat[:,:1,:,:]*bmat[:,1:,:,:] + amat[:,1:,:,:]*bmat[:,:1,:,:]],1)

            return cmat
        else:
            cmat = torch.cat([amat[:,:,:1,:,:]*bmat[:,:,:1,:,:] - amat[:,:,1:,:,:]*bmat[:,:,1:,:,:],
                              + amat[:,:,:1,:,:]*bmat[:,:,1:,:,:] + amat[:,:,1:,:,:]*bmat[:,:,:1,:,:]],2)

            return cmat
