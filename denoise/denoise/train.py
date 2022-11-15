"""
# Description:
#  Training main code
#
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
"""

import os
import math
import logging
from tqdm import tqdm
import numpy as np
from scipy.io import savemat
import time
import argparse
import warnings

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

#import denoise
from params import parse
from datasets import DataWrapper
from network import Network
from utils import separator, logging_helper, seconds_to_dhms, calculate_psnr, calculate_ssim


warnings.filterwarnings('ignore')
logger = logging.getLogger("denoise")

class Trainer:
    def __init__(self, args):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu 
        
        self.args = args
        
        ## Parameters ##
        self.batch = self.args.batch
        self.init_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.separator = separator
        
        ## Experiment path setup ##
        self.run_dir = os.path.abspath(self.args.run_dir)
        self.next_run_id = self.call_next_id()
        self.run_dir_path = os.path.join(self.run_dir, '{:05d}-train'.format(self.next_run_id))
        
        ### logging define ###
        logging_helper.setup(args = self.args,
                                     log_dir = self.run_dir_path,
                                     printing_mode = self.args.printing_mode)
        

    
        ### Hyperparameter define ###
        if self.args.printing_mode != 'yes':
            logger.info(self.separator())
            logger.info("{:05d}-train".format(self.next_run_id))
        logger.info(self.separator())
        logger.info("ArgPaser Info")
        for key, value in vars(self.args).items():
            logger.info('{:15s}: {}'.format(key,value))
        logger.info(self.separator()) 
                        
        ### Data define ###
        logger.info("Loading training dataset...")
        self.trainloader, self.train_wrapper, self.train_count = self.train_data(self.args.train_dataset, self.args.train_mask)
        logger.info("Traininig dataset length : {}".format(self.train_count))
        self.print_interval = (math.ceil(self.train_count/(self.batch*self.args.check_num))*self.batch)
        
        logger.info("Loading validation dataset...")
        self.validloader, self.valid_wrapper, self.valid_count = self.test_data(self.args.valid_dataset, self.args.valid_mask)     
        logger.info("Validation dataset length : {}".format(self.valid_count))
        logger.info(self.separator())
        
        
        ### Networks define ###
        self.network = Network(self.args,logger,device = self.device)
        self._optimizer = optim.Adam(self.network.parameters(), betas=[0.9, 0.99])
        
    
    def train(self):
        ### initialize ###
        self.epoch = 0
        self.train_state_reset()
            
        ### Start ###
        logger.info("TRAINING START")
        logger.info(self.separator())
        
        for epoch in range(self.args.train_epoch):
            self.epoch = epoch
            self.iteration = 0
                
            ### start training ###
            train_tqdm = self.tqdm_loader(self.trainloader)
            train_tqdm.clear()
            logger.info(self.train_state_str())  
            
            for data in train_tqdm:
                self.network.train()
                image_count = data[DataWrapper.Input].shape[0]

                self.iteration += image_count
                    
                optimizer = self.optimizer()
                optimizer.zero_grad()
                outputs = self.network(data)
                loss = outputs[Network.Loss]
                torch.mean(loss).backward()
                optimizer.step()

                with torch.no_grad():
                    self.train_state["image_cnt"] += image_count
                    self.metric_add(self.train_state,"loss",loss)     

                    
                if ((self.iteration % self.print_interval == 0) or
                    (self.iteration == self.train_count)):
                    train_tqdm.clear()
                    logger.info(self.train_state_str())

            self.train_state_reset()
            ### finish training ###
            
            ### start evaluation ###
            if epoch % self.args.eval_interval == 0:
                self.network.eval()
                idx = 0

                valid_tqdm = self.tqdm_loader(self.validloader)
                
                for data in valid_tqdm:
                    with torch.no_grad():
                        image_count = data[DataWrapper.Input].shape[0]
                        outputs = self.network(data)
                        loss = outputs[Network.Loss]
                        self.eval_state["image_cnt"] += image_count
                        self.metric_add(self.eval_state,"loss",loss)  
                        self.metric_add(self.eval_state,"psnr",calculate_psnr(outputs[Network.Output],
                                                                              outputs[Network.Label],
                                                                              data[DataWrapper.Mask]))      
                        self.metric_add(self.eval_state,"ssim",calculate_ssim(outputs[Network.Output],
                                                                              outputs[Network.Label],
                                                                              data[DataWrapper.Mask])) 
                        
                        if ((self.args.save_val == 'yes') and
                           (self.args.printing_mode != 'yes')):
                            #self.save_output(outputs, 
                            #                 save_dir = ('valid/valid_hist_'+ str(epoch) +
                            #                              '/' + data[DataWrapper.Name][0][0]), 
                            #                 file_name = data[DataWrapper.Name][1][0])
                                
                            self.save_output(outputs, 
                                             save_dir = ('valid/valid_hist_'+ str(epoch)),
                                             file_name = 'data' + str(idx) + '.mat')
                                
                        idx += image_count
                        
                eval_summary = self.eval_state_str()
                logger.info(eval_summary)
                logger.info("")
                
            ### finish evaluation ###

    def tqdm_loader(self, dataloader):
        return tqdm(dataloader, 
                    ncols = 80,
                    ascii = True, 
                    desc = 'Epoch {}'.format(self.epoch))
    
    def save_output(self, outputs, save_dir, file_name):
        out = {}
        out['inp'] = outputs[Network.Input].cpu().detach().numpy()
        out['out'] = outputs[Network.Output].cpu().detach().numpy()
        out['lab'] = outputs[Network.Label].cpu().detach().numpy()

        output_dir = os.path.join(self.run_dir_path, save_dir)
        os.makedirs(output_dir, exist_ok=True)
        savemat(os.path.join(output_dir, file_name),out)
        

    def train_state_str(self):
        if self.iteration == 0: 
            summary = '[{:05d}] {:>5} | '.format(int(self.epoch), 'TRAIN')
            summary += 'Lr: {:0.3e}'.format(self.learning_rate())
        else:
            self.print_num += 1
            summary = '{:10}{}/{} | '.format('',self.print_num, self.args.check_num) 
            
                
            summary += '{} | '.format(seconds_to_dhms(time.time() - self.init_time))
            
            image_cnt = self.train_state['image_cnt']
            for key, metric in self.train_state.items():
                if key != 'image_cnt':
                    summary += '{}: {:0.3e} '.format(key, metric/image_cnt)

                                                             
        return summary

    def eval_state_str(self):
        summary = '{:7} {:>5} | '.format('', 'VALID')       
        summary += '{} | '.format(seconds_to_dhms(time.time() - self.init_time))
        
        image_cnt = self.eval_state['image_cnt']
        for key, metric in self.eval_state.items():
            if key != 'image_cnt':
                summary += '{}: {:0.3e} '.format(key, metric/image_cnt)    
                
        if self.args.printing_mode != 'yes':        
            os.makedirs(os.path.join(self.run_dir_path, 'model_hist'), exist_ok=True)  
            torch.save(self.network.state_dict(), 
                       os.path.join(self.run_dir_path, 'model_hist/{:05d}_model.training'.format(self.epoch)))
            summary += ' model saved'
        
        return summary
                
    def call_next_id(self):
        run_ids = []
        if os.path.exists(self.args.run_dir):
            for run_dir_path, _, _ in os.walk(self.args.run_dir):
                run_dir = run_dir_path.split(os.sep)[-1]
                try:
                    run_ids += [int(run_dir.split("-")[0])]
                except Exception:
                    continue
        return max(run_ids) + 1 if len(run_ids) > 0 else 0
    
        
    def optimizer(self):
        learning_rate = self.learning_rate()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate
        return self._optimizer
    
    def learning_rate(self):
        if self.args.lr_tol < self.epoch:
            factor = self.epoch - self.args.lr_tol
        else:
            factor = 0
        return self.args.lr * (self.args.lr_decay ** factor)

    def train_state_reset(self):
        self.train_state = {"image_cnt": 0}
        self.eval_state = {"image_cnt": 0}
        self.print_num = 0
        
    def metric_add(self,state,key,metric):
        if key in state:
            pass
        else:
            state[key] = 0
        
        try:
            metric = metric.cpu().detach().numpy()
        except:
            pass
        
        state[key] += metric.sum()
        
        
    def train_data(self, path, mask_path):

        dataset = DataWrapper(
            file_path = path,
            mask_path = mask_path,
            args = self.args,
            training_mode = True,
        )  

        _ = dataset[0]

        dataloader = DataLoader(
            dataset,
            batch_size = self.args.batch,
            num_workers = self.args.num_workers,
            pin_memory = True,
            shuffle = True,
        )
        return dataloader, dataset, len(dataset)

        
    def test_data(self, path, mask_path):
        dataset = DataWrapper(
            file_path = path,
            mask_path = mask_path,
            args = self.args,
            training_mode = False,
        )  
        
        _ = dataset[0]
        
        dataloader = DataLoader(
            dataset,
            batch_size = 1,
            num_workers = self.args.num_workers,
            pin_memory = True,
            shuffle = False,
        )
        return dataloader, dataset, len(dataset)


if __name__ == "__main__":
    trainer = Trainer(args = parse())            
    trainer.train()
