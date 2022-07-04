from __future__ import annotations

import torch
import torch.nn as nn
import ssmd
import numpy as np
import logging
from torch import Tensor
import time

from enum import Enum, auto
from ssmd.params import ConfigValue, Algorithm, Network
from ssmd.models import DnCNN, BlindNetwork, Unet
from ssmd.datasets import NoiseWrapper
from typing import Dict, List

class Denoiser(nn.Module):
    
    Input = 0
    Cleaned = 1
    Ref = 2
    Loss = 3
    Mask = 4
    
    def __init__(self, cfg: Dict, logger, device: str = None):
        super().__init__()
        if device:
            device = torch.device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.cfg = cfg
        self.logger = logger
        self.models = nn.ModuleDict()
        self._models = nn.ModuleDict()

        self.init_networks()
        self.evalflag = 1
        self.trainflag = 1
       
        if self.cfg[ConfigValue.ALGORITHM] == Algorithm.NE2NE:
            self.ne2nesampler = ssmd.utils.ne2ne.masksamplingv2()
            
    def traindebug(self,string):
        if self.trainflag == 1:
            self.logger.debug(string)
            
    def evaldebug(self,string):
        if self.evalflag == 1:
            self.logger.debug(string)
        
    def init_networks(self):
        
        in_chan = 1
        out_chan = 1
        
        if (self.cfg[ConfigValue.ALGORITHM] == Algorithm.C2C or
                self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2N or
                self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2C or
                self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2S or
                self.cfg[ConfigValue.ALGORITHM] == Algorithm.CS or
                self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2SA or
                self.cfg[ConfigValue.ALGORITHM] == Algorithm.NE2NE or
                self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2V):
            
            if self.cfg[ConfigValue.NET_ARCHITECTURE] == Network.DNCNN: 
                if self.cfg[ConfigValue.PARALLEL] is not True:
                    self.model = DnCNN(channels=in_chan).to(self.device)
                else:
                    self.model = nn.DataParallel(DnCNN(channels=in_chan)).to(self.device)
                
            if self.cfg[ConfigValue.NET_ARCHITECTURE] == Network.UNET: 
                if self.cfg[ConfigValue.PARALLEL] is not True:
                    self.model = Unet(in_chan=out_chan,out_chan=out_chan).to(self.device)
                else:
                    self.model = nn.DataParallel(Unet(in_chan=out_chan,out_chan=out_chan)).to(self.device)

        elif self.cfg[ConfigValue.ALGORITHM] == Algorithm.SSDN:
            self.diagonal = True
            if self.diagonal:
                out_chan = in_chan * 2  
            else:
                out_chan = ( in_chan + (in_chan * (in_chan + 1)) // 2) 
                
            if self.cfg[ConfigValue.NET_ARCHITECTURE] == Network.DNCNN: 
                if self.cfg[ConfigValue.PARALLEL] is not True:
                    self.model = BlindNetwork(in_channels=in_chan,out_channels=out_chan,blindspot=True, align=False).to(self.device)
                    self.sig_estimator = BlindNetwork(in_channels=in_chan, out_channels=1, blindspot=False, zero_output_weights=True).to(self.device)
                else:
                    self.model = nn.DataParallel(BlindNetwork(in_channels=in_chan,out_channels=out_chan,blindspot=True, align=False)).to(self.device)
                    self.sig_estimator = nn.DataParallel(BlindNetwork(in_channels=in_chan, out_channels=1, blindspot=False, zero_output_weights=True)).to(self.device)
        else:
             raise NotImplementedError("Network structure is not specified")
            
        self.logger.info(ssmd.utils.summary.summary(self.model,(1,320,320)))
        
    def state_dict(self):
        state_dict = state_dict = super().state_dict()
        state_dict["cfg"] = self.cfg
        return state_dict

    def from_state_dict(state_dict: Dict,logger):
        denoiser = Denoiser(state_dict["cfg"],logger)
        denoiser.load_state_dict(state_dict, strict=False)
        return denoiser

    def forward(self, data: List, **kwargs):
        if self.cfg[ConfigValue.ALGORITHM] == Algorithm.C2C:
            return self._c2c_pipeline(data, **kwargs)
        
        elif (self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2N or
            self.cfg[ConfigValue.ALGORITHM] == Algorithm.CS or
            self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2C):
            return self._mse_pipeline(data, **kwargs)

        elif self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2V:
            return self._n2v_pipeline(data, **kwargs)
        
        elif self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2S:
            return self._n2s_pipeline(data, **kwargs)
        
        elif self.cfg[ConfigValue.ALGORITHM] == Algorithm.N2SA:
            return self._n2sa_pipeline(data, **kwargs)
        
        elif self.cfg[ConfigValue.ALGORITHM] == Algorithm.NE2NE:
            return self._ne2ne_pipeline(data, **kwargs)
        
        elif self.cfg[ConfigValue.ALGORITHM] == Algorithm.SSDN:
            return self._ssdn_pipeline(data, **kwargs)
        else:
            raise NotImplementedError("Unsupported algorithm")

    def _c2c_pipeline(self, data: List, **kwargs) -> Dict:
        if self.training is True:
            self.model.train()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            label = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.LABEL].to(self.device)
            insen = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.INSEN].to(self.device)
            lasen = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.LASEN].to(self.device)
            inp.requires_grad = True
            label.requires_grad = True
                
            cleaned = self.model(inp)

            loss = nn.MSELoss(reduction="none")(cleaned*lasen, label*insen)
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)
            
            self.traindebug("cleaned shape : {}".format(cleaned.shape))
            self.traindebug("insen shape : {}".format(insen.shape))
            self.traindebug("label shape : {}".format(label.shape))
            self.traindebug("lasen shape : {}".format(lasen.shape))
            self.trainflag -= 1  
            
        else:
            self.model.eval()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            
            cleaned = self.model(inp)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            loss = None
                        
            self.evaldebug("valid cleaned shape : {}".format(cleaned.shape))
            self.evalflag -= 1
            
        return (inp, cleaned, ref, loss, mask)
    
    def _mse_pipeline(self, data: List, **kwargs) -> Dict:
        if self.training is True:
            self.model.train()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            label = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.LABEL].to(self.device)
            inp.requires_grad = True
            label.requires_grad = True
            
            cleaned = self.model(inp)
            loss = nn.MSELoss(reduction="none")(cleaned, label)
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)

            self.traindebug("inp shape : {}".format(inp.shape))
            self.traindebug("cleaned shape : {}".format(cleaned.shape))              
            self.trainflag -= 1
            
        else:
            self.model.eval()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            cleaned = self.model(inp)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            loss = None
                
            self.evaldebug("eval inp shape : {}".format(inp.shape))
            self.evaldebug("cleaned shape : {}".format(cleaned.shape))
            self.evalflag -= 1
            
        return (inp, cleaned, ref, loss, mask)
    
    def _n2v_pipeline(self, data: List, **kwargs) -> Dict:
        if self.training is True:
            self.model.train()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK_COORDS].to(self.device)
            brain_mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            label = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.LABEL].to(self.device)
            inp.requires_grad = True
            ref.requires_grad = True
            cleaned = self.model(inp)          
            
            loss = nn.MSELoss(reduction="none")(cleaned*mask, label*mask)
            loss = loss.view(loss.shape[0], -1).sum(1, keepdim=True)#/torch.sum(mask)

            self.traindebug("inp shape : {}".format(inp.shape))
            self.traindebug("cleaned shape : {}".format(cleaned.shape))               
            self.trainflag -= 1
            
        else:
            self.model.eval()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            cleaned = self.model(inp)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            brain_mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            loss = None

            self.evaldebug("eval inp shape : {}".format(inp.shape))
            self.evaldebug("cleaned shape : {}".format(cleaned.shape))
            self.evalflag -= 1
            
        return (inp, cleaned, ref, loss, brain_mask)
    
    def _n2s_pipeline(self, data: List, **kwargs) -> Dict:
        if self.training is True:
            self.model.train()

            inp = data[NoiseWrapper.INPUT].to(self.device)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK_COORDS]
            mask = torch.unsqueeze(mask,1).to(self.device)
            brain_mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            inp.requires_grad = True
            ref.requires_grad = True
            cleaned = self.model(inp)

            loss = nn.MSELoss(reduction="none")(cleaned*mask, ref*mask)
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=True)
            
            self.traindebug("inp shape : {}".format(inp.shape))
            self.traindebug("cleaned shape : {}".format(cleaned.shape))
            self.trainflag -= 1
            
        else:
            self.model.eval()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            brain_mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            cleaned = self.model(inp)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            loss = None
            
            self.evaldebug("eval inp shape : {}".format(inp.shape))
            self.evaldebug("cleaned shape : {}".format(cleaned.shape))
            self.evalflag -= 1
            
        return (inp, cleaned, ref, loss, brain_mask)
    
    def _n2sa_pipeline(self, data: List,  **kwargs) -> Dict:
        if self.training is True:
            self.model.train()
            
            inp = data[NoiseWrapper.INPUT].to(self.device)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            inp_hat = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.INP_HAT].to(self.device)
            mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK_COORDS].to(self.device)
            brain_mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            inp.requires_grad = True
            ref.requires_grad = True

            cleaned = self.model(inp)
            inp_hat_out = self.model(inp_hat)

            
            loss_rec = nn.MSELoss(reduction="none")(cleaned, ref)
            loss_rec = loss_rec.view(loss_rec.shape[0], -1).mean(1, keepdim=True)

            loss_inv = nn.MSELoss(reduction="none")(cleaned*mask, inp_hat_out*mask)
            loss_inv = loss_inv.view(loss_inv.shape[0], -1).sum(1, keepdim=True)/torch.sum(mask)
            
            loss = loss_rec + 2 * torch.sqrt(loss_inv)
            
            self.traindebug("inp shape : {}".format(inp.shape))
            self.traindebug("cleaned shape : {}".format(cleaned.shape))
            self.trainflag -= 1
            
        else:
            self.model.eval()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            cleaned = self.model(inp)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            brain_mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            loss = None

            self.evaldebug("eval inp shape : {}".format(inp.shape))
            self.evaldebug("cleaned shape : {}".format(cleaned.shape))
            self.evalflag -= 1
            
        return (inp, cleaned, ref, loss, brain_mask)
    
    def _ne2ne_pipeline(self, data: List, **kwargs) -> Dict:
        if self.training is True:
            self.model.train()

            inp = data[NoiseWrapper.INPUT].to(self.device)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            pattern = torch.randint(0, 8, (1, ))
            red, blue = self.ne2nesampler(inp, pattern)
            red_cleaned = self.model(red)
            inp.requires_grad = True
            with torch.no_grad():
                cleaned = self.model(inp)

            red_o, blue_o = self.ne2nesampler(cleaned, pattern)
            loss_rec = nn.MSELoss(reduction="none")(blue, red_cleaned)
            loss_rec = loss_rec.view(loss_rec.shape[0], -1).mean(1, keepdim=True)

            loss_reg = nn.MSELoss(reduction="none")(blue - red_cleaned, blue_o - red_o)
            loss_reg = loss_reg.view(loss_reg.shape[0], -1).mean(1, keepdim=True)

            loss = loss_rec + 2 * loss_reg
            
            self.traindebug("inp shape : {}".format(inp.shape))
            self.traindebug("cleaned shape : {}".format(cleaned.shape))
            self.trainflag -= 1
            
        else:
            self.model.eval()
            inp = data[NoiseWrapper.INPUT].to(self.device)
            cleaned = self.model(inp)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            loss = None

            self.evaldebug("eval inp shape : {}".format(inp.shape))
            self.evaldebug("cleaned shape : {}".format(cleaned.shape))
            self.evalflag -= 1
            
        return (inp, cleaned, ref, loss)
        
    def _ssdn_pipeline(self, data: List, **kwargs) -> Dict:
        def batch_mvmul(m, v):  # Batched (M * v).
            return torch.sum(m * v[..., None, :], dim=-1)
        
        def batch_vtmv(v, m):  # Batched (v^T * M * v).
            return torch.sum(v[..., :, None] * v[..., None, :] * m, dim=[-2, -1])
        
        def batch_vvt(v):  # Batched (v * v^T).
            return v[..., :, None] * v[..., None, :]
        
        metadata = data[NoiseWrapper.METADATA]
        noise_style = self.cfg[ConfigValue.NOISE_STYLE]
        num_channels = 1

        diagonal_covariance = self.diagonal
        num_output_components = (num_channels + (num_channels * (num_channels + 1)) // 2)  # Means, triangular A.
        
        if diagonal_covariance:
            num_output_components = num_channels * 2  # Means, diagonal of A.    
            
            
        if self.training is True:
            self.model.train()
            inp = data[NoiseWrapper.INPUT]
            noisy_in = inp.to(self.device)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            brain_mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            param_est_net_out = self.sig_estimator(noisy_in)
            net_out = self.model(noisy_in)
            noise_est_out = torch.mean(param_est_net_out, dim=(1, 2, 3), keepdim=True)
            noise_std = noise_est_out
            
            mu_x = net_out[:, 0:num_channels, ...]  
            A_c = net_out[:, num_channels:num_output_components, ...] 

            if diagonal_covariance is not True:
                raise NotImplementedError("NonDiagonal Unsuported")
                
            sigma_x = A_c ** 2
            sigma_n = noise_std ** 2
            sigma_y = sigma_x + sigma_n
            pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (sigma_x + sigma_n)

            loss_out = 0.5 * ((noisy_in - mu_x) ** 2)/sigma_y + 0.5 * torch.log(sigma_y)
            loss = loss_out - 0.1 * noise_std  
            loss = loss.view(loss_out.shape[0], -1).mean(1, keepdim=True)
            
            self.traindebug("noisy input shape : {}".format(noisy_in.shape))
            self.traindebug("A_c shape : {}".format(A_c.shape))
            self.traindebug("noisy_in shape : {}".format(noisy_in.shape))
            self.traindebug("mu_x shape : {}".format(mu_x.shape))
            self.traindebug("sigma_n shape : {}".format(sigma_n.shape))
            self.traindebug("sigma_y shape : {}".format(sigma_y.shape))
            self.traindebug("sigma_x shape : {}".format(sigma_x.shape))
            self.traindebug("noise_std shape : {}".format(noise_std.shape))
            self.traindebug("loss shape : {}".format(loss_out.shape))
            self.trainflag -= 1
            
        else:
            inp = data[NoiseWrapper.INPUT].to(self.device)
            ref = data[NoiseWrapper.REFERENCE].to(self.device)
            brain_mask = data[NoiseWrapper.METADATA][NoiseWrapper.Metadata.MASK].to(self.device)
            noisy_in = inp
            loss = None
            param_est_net_out = self.sig_estimator(inp)
            net_out = self.model(inp)
            noise_est_out = torch.mean(param_est_net_out, dim=(1, 2, 3), keepdim=True)
            noise_std = noise_est_out

            mu_x = net_out[:, 0:num_channels, ...] 
            A_c = net_out[:, num_channels:num_output_components, ...]  

            if diagonal_covariance is not True:
                raise NotImplementedError("NonDiagonal Unsuported")
                
            sigma_x = A_c ** 2
            sigma_n = noise_std ** 2
            sigma_y = sigma_x + sigma_n
            #print(sigma_x.shape)
            #print(sigma_n.shape)
            pme_out = (noisy_in * sigma_x + mu_x * sigma_n) / (sigma_x + sigma_n)

            self.evaldebug("cleaned shape : {}".format(pme_out.shape))        
            self.evalflag -= 1
            
        return (inp, pme_out, ref, loss, brain_mask)
        
