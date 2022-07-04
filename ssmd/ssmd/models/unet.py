import torch
import ssmd
import torch.nn as nn

from torch import Tensor
from typing import Tuple

class Unet(nn.Module):
    def __init__(self,in_chan: int = 2,out_chan: int = 2): 
        super(Unet, self).__init__()
        filter_size = 3
        padding = 1
        
        def Conv2d(in_c,out_c) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_c,out_c, filter_size, stride=1, padding=padding, bias=False),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(out_c,out_c, filter_size, stride=1, padding=padding, bias=False),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                #nn.Conv2d(out_c,out_c, filter_size, stride=1, padding=padding, bias=False),
                #nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )

        self.encode_block_1 = nn.Sequential(Conv2d(in_chan, 48),nn.MaxPool2d(2))
        self.encode_block_2 = nn.Sequential(Conv2d(48, 48),nn.MaxPool2d(2))
        self.encode_block_3 = nn.Sequential(Conv2d(48, 48),nn.MaxPool2d(2))
        self.encode_block_4 = nn.Sequential(Conv2d(48, 48),nn.MaxPool2d(2))
        self.encode_block_5 = nn.Sequential(Conv2d(48, 48),nn.MaxPool2d(2))
        self.encode_block_6 = nn.Sequential(Conv2d(48, 48))
        
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))
        self.decode_block_5 = nn.Sequential(Conv2d(96, 96),nn.Upsample(scale_factor=2, mode="nearest"),)
        self.decode_block_4 = nn.Sequential(Conv2d(144, 96),nn.Upsample(scale_factor=2, mode="nearest"),)
        self.decode_block_3 = nn.Sequential(Conv2d(144, 96),nn.Upsample(scale_factor=2, mode="nearest"),)
        self.decode_block_2 = nn.Sequential(Conv2d(144, 96),nn.Upsample(scale_factor=2, mode="nearest"),)

       
        self.output_block = nn.Sequential(
            nn.Conv2d(96 + in_chan, 96, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, out_chan, kernel_size=1, bias=False))

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, a=0.1)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                        m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4:
            raise('Not Implemented Error')
        inp = x
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        upsample5 = self.decode_block_6(encoded)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        
        x = self.output_block(concat1)

        return x# + inp
