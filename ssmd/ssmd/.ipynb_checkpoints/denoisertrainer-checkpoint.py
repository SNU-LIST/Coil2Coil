"""
# Description:
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
"""

from __future__ import annotations
import os
import math
import logging
from tqdm import tqdm
import numpy as np
from scipy.io import savemat
import time
from typing import Dict, Tuple, Callable
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim

import ssmd

from ssmd.datasets import NoiseWrapper
from ssmd.params import ConfigValue,  StateValue
from ssmd.denoiser import Denoiser
from ssmd.utils import Metric, MetricDict, separator

logger = logging.getLogger("ssmd.train")

class DenoiserTrainer:
 
    def __init__(self, cfg: Dict, state: Dict = {}, runs_dir: str = 'log', gpu : str = None, run_dir: str = None):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"]= gpu 
        
        self.runs_dir = os.path.abspath(runs_dir)
        self._run_dir = run_dir
        self.cfg = cfg
        self.state = state
        self._denoiser: Denoiser = None
        self.trainloader, self.train_count = None, None
        self.testloader, self.test_count = None, None
        self.batch = None
        self.print_interval = None
        self.run_dir_path = self.set_run_dir_path()
        self.init_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_run_dir_path(self):
        if self._run_dir is None:
            if self.cfg.get(ConfigValue.TAG, None) is None:
                self.cfg[ConfigValue.TAG] = "none"
            run_dir_name = "{:05d}-train-{}".format(self.next_run_id,self.cfg[ConfigValue.TAG])
            self._run_dir = run_dir_name
        return os.path.join(self.runs_dir, self._run_dir)
    
    @property
    def next_run_id(self):
        run_ids = []
        if os.path.exists(self.runs_dir):
            for run_dir_path, _, _ in os.walk(self.runs_dir):
                run_dir = run_dir_path.split(os.sep)[-1]
                try:
                    run_ids += [int(run_dir.split("-")[0])]
                except Exception:
                    continue
        return max(run_ids) + 1 if len(run_ids) > 0 else 0
    
    @property
    def denoiser(self) -> Denoiser:
        return self._denoiser

    @denoiser.setter
    def denoiser(self, denoiser: Denoiser):
        self._denoiser = denoiser
        self._optimizer = optim.Adam(self.denoiser.parameters(), betas=[0.9, 0.99])
        
    @property
    def optimizer(self) -> Optimizer:
        learning_rate = self.learning_rate
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate
        return self._optimizer

    @property
    def learning_rate(self) -> float:
        if self.cfg[ConfigValue.LR_DOWN] < self.state[StateValue.EPOCH]:
            factor = self.state[StateValue.EPOCH] - self.cfg[ConfigValue.LR_DOWN] 
        else:
            factor = 0
        return self.cfg[ConfigValue.LEARNING_RATE] * (self.cfg[ConfigValue.LR_COEFF] ** factor)

    def init_state(self):
        self.state[StateValue.ITERATION] = 0
        self.state[StateValue.EPOCH] = 0
        self.state[StateValue.PREV_PSNR] = 0
        self.state[StateValue.PREV_ITER] = 0
        self.state[StateValue.TRAIN] = MetricDict()
        self.state[StateValue.EVAL] = MetricDict()
        self.reset_metric_dict(self.state[StateValue.EVAL])
        self.reset_metric_dict(self.state[StateValue.TRAIN])
        
    def train(self):
        self.init_state()
        self.batch = self.cfg[ConfigValue.TRAIN_MINIBATCH_SIZE]
        self.print_interval = self.cfg[ConfigValue.PRINT_INTERVAL]
        
        ssmd.logging_helper.setup(self.cfg,self.run_dir_path, "log.txt")
        logger.info(ssmd.utils.separator())
        logger.info("{:05d}-train".format(self.next_run_id-1))
        logger.info(ssmd.utils.separator())
        logger.info("Loading Training Dataset...")
        self.trainloader, _ = self.train_data()
        logger.info("Traininig dataset length : {}".format(self.train_count))
        logger.info("Loaded Training Dataset.")
        if self.cfg[ConfigValue.TEST_DATA_PATH]:
            logger.info("Loading Validation Dataset...")
            self.testloader, _ = self.test_data()
            logger.info("Validation dataset length : {}".format(self.test_count))
            logger.info("Loaded Validation Dataset.")
        else:
            logger.info("Validation Dataset not exist")
        logger.info(ssmd.utils.separator())
        for key, value in self.cfg.items():
            logger.info('{}:{}'.format(key,value))
        logger.info(ssmd.utils.separator())
        logger.info("TRAINING STARTED")
        logger.info(ssmd.utils.separator())

        self.denoiser = Denoiser(self.cfg,logger,self.device)

        train_history = self.state[StateValue.TRAIN]
        eval_history = self.state[StateValue.EVAL]
        
        for epoch in range(self.cfg[ConfigValue.TRAIN_EPOCH]):
            self.state[StateValue.EPOCH] = epoch
            self.state[StateValue.ITERATION] = 0
            
            if epoch % self.cfg[ConfigValue.EVAL_INTERVAL] == 0:
                self.denoiser.eval()
                idx = 0
                for data in self.testloader:
                    with torch.no_grad():
                        image_count = data[NoiseWrapper.INPUT].shape[0]
                        outputs = self.denoiser(data)
                        eval_history["n"] += image_count
                        for name in ({'psnr', 'ssim'}):
                            eval_history[name] += self.calculate_metric(outputs, name)
                    if idx == 0:
                        self.save_output(outputs, 
                                         save_dir = 'eval_imgs', 
                                         file_name = 'epoch_' + str(epoch) + '.mat')
                    idx += image_count
                logger.eval(self.eval_state_str())
                self.reset_metric_dict(self.state[StateValue.EVAL])
                
            for data in self.trainloader:
                self.denoiser.train()
                image_count = data[NoiseWrapper.INPUT].shape[0]
                
                optimizer = self.optimizer
                optimizer.zero_grad()
                outputs = self.denoiser(data)
                torch.mean(outputs[Denoiser.Loss]).backward()
                optimizer.step()
                
                with torch.no_grad():
                    train_history["loss"] += outputs[Denoiser.Loss]
                    train_history["n"] += image_count
                    for name in ({'psnr'}):
                        train_history[name] += self.calculate_metric(outputs, name)
                        
                if self.state[StateValue.ITERATION] % (math.ceil(self.train_count /(self.batch*self.print_interval)) * self.batch) == 0:
                    logger.info(self.train_state_str())
                    logger.info("{:14}| learning_rate, {:.3e}".format("",self.learning_rate))
                    self.reset_metric_dict(self.state[StateValue.TRAIN])

                self.state[StateValue.ITERATION] += image_count  

        logger.info(separator())
        logger.info("TRAINING FINISHED")
        logger.info(separator())

    def save_output(self, outputs, save_dir = str, file_name = str):
        out = {}
        out['cln'] = self.unpad_mat(outputs[Denoiser.Ref])
        out['inp'] = self.unpad_mat(outputs[Denoiser.Input])
        out['out'] = self.unpad_mat(outputs[Denoiser.Cleaned])
        out['mask'] = self.unpad_mat(outputs[Denoiser.Mask])
        output_dir = os.path.join(self.run_dir_path, save_dir)
        os.makedirs(output_dir, exist_ok=True)
        savemat(os.path.join(output_dir, file_name),out)
        
    def unpad_mat(self, img: Tensor):
        
        if (len(img.shape) != 4 and len(img.shape) != 3):
            raise NotImplementedError("Unsupported image dimension")
        if len(img.shape) == 4:
            img = img.permute(2,3,1,0)
        else:
            img = img.permute(2,1,0)
        img = img.cpu().detach().numpy()
        return img
            
    def reset_metric_dict(self,metric_dict: Dict):
        metric_dict["n"] = 0
        for key, metric in metric_dict.items():
            if isinstance(metric, Metric):
                metric.reset()

    def calculate_metric(self, outputs: Dict, name: str) -> Tensor:
        output = outputs[Denoiser.Cleaned]
        clean = outputs[Denoiser.Ref].to(output.device)
        mask = outputs[Denoiser.Mask].to(output.device)
        #output = output * mask
        #clean = clean * mask
        if name == 'psnr':
            metric = ssmd.utils.calculate_psnr(output, clean, mask)
        elif name == 'ssim':
            metric = ssmd.utils.calculate_ssim(output, clean, mask)
        else:
            raise('Metric Not Implemented')
        return metric

    def train_state_str(self) -> str:
        if self.state[StateValue.ITERATION] == 0: 
            summary = "[{:05d}] {:>5} | ".format(int(self.state[StateValue.EPOCH]), "TRAIN")
        else:
            summary = "{:10}{}/{} | ".format("",int((self.state[StateValue.ITERATION]/(self.train_count)+
                                                     1e-10)//(1/self.print_interval)), self.print_interval)
        
        metric_str_list = []
        train_metrics = self.state[StateValue.TRAIN]    
        
        for key, metric in train_metrics.items():
            if isinstance(metric, Metric) and not metric.empty():
                metric_str_list += ["{} = {:8.3e}".format(key, metric.accumulated())]
            
        summary += ", ".join(metric_str_list)
        if len(metric_str_list) > 0:
            summary += " | "
            
        total_train = time.time() - self.init_time
        summary += "[{}]".format(ssmd.utils.seconds_to_dhms(total_train))
        
        return summary

    def eval_state_str(self):
        prefix = "{:7} {:>5}".format("", "VALID")
        summary = "{} | ".format(prefix)
        
        metric_str_list = []
        eval_metrics = self.state[StateValue.EVAL]
        
        for key, metric in eval_metrics.items():
            if isinstance(metric, Metric) and not metric.empty():
                metric_acc = metric.accumulated()
                metric_str_list += ["{} = {:.3f}".format(key, metric_acc)]
                if key == 'psnr':
                    epoch = int(self.state[StateValue.EPOCH])
        
                    if self.state[StateValue.PREV_PSNR] < metric_acc:
                        summary += 'New Record In Epoch {} => '.format(epoch)
                        self.state[StateValue.PREV_PSNR] = metric_acc
                        self.state[StateValue.PREV_ITER] = epoch
                        torch.save(self.denoiser.state_dict(), 
                                   os.path.join(self.run_dir_path, 
                                                "model.training"))
                    else:
                        summary += 'Best Record In Epoch {} => '.format(self.state[StateValue.PREV_ITER])
                        
                    os.makedirs(os.path.join(self.run_dir_path, "model_history"), exist_ok=True)  
                    torch.save(self.denoiser.state_dict(), 
                               os.path.join(self.run_dir_path, "model_history",
                                            "model_{:05d}.training".format(epoch)))
                            
        summary += ", ".join(metric_str_list)
        return summary   


    def train_data(self) -> Tuple[DataLoader, NoiseWrapper]:
        cfg = self.cfg
        
        dataset = NoiseWrapper(
            file_path = cfg[ConfigValue.TRAIN_DATA_PATH],
            mask_path = cfg[ConfigValue.TRAIN_MASK_PATH],
            transform = True,
            chan_frac = cfg[ConfigValue.CHAN_FRAC],
            noise_style = cfg[ConfigValue.NOISE_STYLE],
            algorithm = cfg[ConfigValue.ALGORITHM],
            training_mode = True,
            patchsize = cfg[ConfigValue.PATCH_SIZE],
        )  

        _ = dataset[0]
        self.train_count = len(dataset)
        
        dataloader = DataLoader(
            dataset,
            batch_size = cfg[ConfigValue.TRAIN_MINIBATCH_SIZE],
            num_workers = cfg[ConfigValue.DATALOADER_WORKERS],
            pin_memory = True,
            shuffle = True,
        )
        

        return dataloader, dataset
            
    def test_data(self) -> Tuple[DataLoader, NoiseWrapper]:
        cfg = self.cfg
        
        dataset = NoiseWrapper(
            file_path = cfg[ConfigValue.TEST_DATA_PATH],
            mask_path = cfg[ConfigValue.TEST_MASK_PATH],
            transform = False,
            chan_frac = cfg[ConfigValue.CHAN_FRAC],
            noise_style = cfg[ConfigValue.NOISE_STYLE],
            algorithm = cfg[ConfigValue.ALGORITHM],
            training_mode = False,
            img_count = cfg[ConfigValue.TEST_LEN],
            )

        _ = dataset[0]
        self.test_count = len(dataset)
        
        if self.cfg[ConfigValue.TEST_LEN] is None:
            self.cfg[ConfigValue.TEST_LEN] = self.test_count
            
        dataloader = DataLoader(
            dataset,
            batch_size = cfg[ConfigValue.TEST_MINIBATCH_SIZE],
            num_workers = cfg[ConfigValue.DATALOADER_WORKERS],
            pin_memory = True,
            shuffle = False,
        )
        return dataloader, dataset

    

