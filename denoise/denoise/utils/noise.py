"""
# Description:
#  Add noise
#
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
"""



import torch

class noise_adder():
    def __init__(self,noise_property):
        if noise_property == 'inf':
            self.noise_property = 'inf'
            self.noise_level = None
        elif noise_property[:5] == 'gauss':
            self.noise_property = 'gauss'
            self.noise_level = noise_property[5:].split('_')
            
            if len(self.noise_level) == 1:
                self.noise = float(self.noise_level[0])
            elif len(self.noise_level) == 2:
                self.noise = float(self.noise_level[0])
                self.noise = float(self.noise_level[1])
            else:
                raise ('not a float value')
                
    def add(self,image,mean):
        image = image.clone()
        if self.noise_property == 'inf':
            
            image += 0
            return image      
        
        else:

            if len(self.noise_level) == 1:
                std_dev = mean*self.noise/1.4141

            else:
                min_std_dev = self.noise_min 
                max_std_dev = self.noise_max
                std_dev = mean*(torch.rand(1) * (max_std_dev-min_std_dev) + min_std_dev)/1.4141

            image += torch.randn(image.shape) * std_dev 
            return image            
