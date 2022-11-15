"""
# Description:
#  classifier
#
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import logging
import time

import denoise
from datasets import DataWrapper
from utils import separator, str2bool
from models import (
    DnCNN,
    #VisionTransformer,
    )

class Network(nn.Module):
    Input = 0
    Output = 1
    Label = 2
    Loss = 3

    
    def __init__(self, args, logger, device = None, chan_num = None):
        super().__init__()
        if device:
            device = torch.device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.args = args
        self.logger = logger
        self.models = nn.ModuleDict()
        self._models = nn.ModuleDict()
        self.separator = separator
        
        ### Netowkr Initilization ###

        if self.args.network == 'dncnn': 
            if chan_num:
                self.model = DnCNN(channels = chan_num)
            else:
                self.model = DnCNN(channels = 1) 
            
        elif self.args.network == 'transf':
            self.model = VisionTransformer('ViT-B_16', 512, zero_head=True, num_classes=2, vis = True)
            
            ##vor VIT##
            #backbone_model = EfficientNet.from_pretrained('efficientnet-b6',in_channels=1,num_classes=2)
            #self.model.transformer.embeddings.hybrid_model = backbone_model
        else:
             raise NotImplementedError("Network structure is not specified")

        if self.args.parallel == 'yes':
            self.model = nn.DataParallel(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        ### loss model ###
        self.loss_model = nn.MSELoss(reduction='none')
        
    def forward(self, data):
        if self.training is True:
            self.model.train()
            inp = data[DataWrapper.Input].to(self.device)
            label = data[DataWrapper.Label].to(self.device)
            insen = data[DataWrapper.Insen].to(self.device)
            lasen = data[DataWrapper.Lasen].to(self.device)
            output = self.model(inp)

            loss = self.loss_model(output*lasen, label*insen)
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=False)
            
        else:
            self.model.eval()
            inp = data[DataWrapper.Input].to(self.device)
            label = data[DataWrapper.Refer].to(self.device)
            #label = inp
            output = self.model(inp)
            loss = self.loss_model(output,label)
            loss = loss.view(loss.shape[0], -1).mean(1, keepdim=False)
        return (inp, output, label, loss)


    def state_dict(self):
        state_dict = state_dict = super().state_dict()
        state_dict["args"] = self.args
        return state_dict

    def from_state_dict(state_dict, logger, args = None, chan_num = None):

        if args is not None:
            network = Network(args,logger)
        else:
            network = Network(state_dict["args"],logger,chan_num = chan_num)
        network.load_state_dict(state_dict, strict=False)
        return network