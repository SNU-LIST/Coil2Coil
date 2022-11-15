import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torch import Tensor
from typing import Tuple


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 5
        padding = 2
        features = 64

        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        
        for _ in range(num_of_layers-1):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
                
    def forward(self, x):
        #print(len(x.shape)!= 4)
        #print(x.shape)
        if len(x.shape) != 4:
            raise('Not Implemented Error')
            
        out = x + self.dncnn(x)
        return out




