import ssmd
import logging
import os
import argparse
from scipy.io import savemat
from tqdm import tqdm
import numpy as np
import math
import csv

from typing import Dict, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torch import Tensor

import ssmd

from ssmd.datasets import NoiseWrapper, NoiseWrapper_dicom
from ssmd.params import Algorithm, ConfigValue, Network
from ssmd.utils import str2bool, Metric, MetricDict
from ssmd.denoiser import Denoiser

logger = logging.getLogger("ssmd.eval")

def parse():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--gpu", type=str, default = '0')
    parser.add_argument("--model","-m",type = str)
    parser.add_argument("--noise_style","-noise",default='gauss1_1.5')
    parser.add_argument("--dataset","-d")
    parser.add_argument("--mask",type = str)
    parser.add_argument("--runs_dir","-dir",type = str)
    parser.add_argument("--run_dir", type = str)
    parser.add_argument("--batch",type=int,default=1)
    parser.add_argument("--len",type=int, default = None)
    parser.add_argument("--tag", type=str)
    
    args = parser.parse_args()
    return args

class DenoiserEvaluator:
    def __init__(self, args):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu 
        
        state_dict = torch.load(args.model, map_location="cpu")
        self.denoiser = Denoiser.from_state_dict(state_dict,logger)
        self.cfg = self.denoiser.cfg
        self.args = args
        
        self.cfg[ConfigValue.TEST_LEN] = args.len
        if args.noise_style is not None:
            self.cfg[ConfigValue.NOISE_STYLE] = args.noise_style 
        if args.runs_dir is not None:
            self.cfg[ConfigValue.LOG_PATH] = args.runs_dir
        if args.batch is not None:
            self.cfg[ConfigValue.TEST_MINIBATCH_SIZE] = args.batch
        if args.tag is not None:
            self.cfg[ConfigValue.TAG] = args.tag
        if args.dataset is not None:
            self.cfg[ConfigValue.TEST_DATA_PATH] = args.dataset
        if args.mask is not None:
            self.cfg[ConfigValue.TEST_MASK_PATH] = args.mask
            
        self.runs_dir = os.path.abspath(args.runs_dir)
        self._run_dir = args.run_dir
        self.dataset = args.dataset
        self.mask = args.mask
        self.run_dir_path = self.set_run_dir_path()
        self.eval_history = MetricDict()
        self.reset_metric_dict(self.eval_history)
        
    def evaluate(self):
        if self.denoiser is None:
            raise RuntimeError("Denoiser not initialised for evaluation")

        ssmd.logging_helper.setup(self.cfg,self.run_dir_path, "log.txt")
        logger.info(ssmd.utils.separator())
        logger.info("ArgPaser Info")
        for key, value in vars(self.args).items():
            logger.info('{:15s}: {}'.format(key,value))
        logger.info(ssmd.utils.separator())
        logger.info("Loading Test Dataset...")
        self.testloader, self.testset = self.test_data()
        logger.info("Loaded Test Dataset.")
        logger.info(ssmd.utils.separator())
        logger.info("{:05d}-eval".format(self.next_run_id-1))
        logger.info(ssmd.utils.separator())
        for key, value in self.cfg.items():
            logger.info('{}:{}'.format(key,value))


        logger.info(ssmd.utils.separator())
        logger.info("EVALUATION STARTED")
        logger.info(ssmd.utils.separator())

        self.denoiser.eval()
        idx = 0
        for data in tqdm(self.testloader):
            with torch.no_grad():
                image_count = data[NoiseWrapper.INPUT].shape[0]
                outputs = self.denoiser(data)
                self.eval_history["n"] += image_count
                metric = []
                for name in ({'psnr','ssim'}):
                    
                    tmp_metric = self.calculate_metric(outputs, name)
                    
                    self.eval_history[name] += tmp_metric
                    tmp_metric = tmp_metric.cpu().detach().numpy()
                    tmp_metric = np.mean(tmp_metric)
                    metric += ["{}".format(tmp_metric)]
                    
                with open(os.path.join(self.run_dir_path, "valid.csv"), "a") as f:
                    f.write(",".join(metric) + "\n")
                    
                self.save_output(outputs, save_dir = 'eval_imgs', file_name = 'result_' + str(idx) + '.mat')
                idx += image_count
                
        logger.info("EVALUATION RESULT")
        logger.info(self.eval_state_str())
        logger.info(ssmd.utils.separator())
        logger.info("EVALUATION FINISHED")
        logger.info(ssmd.utils.separator())
        
#     def calculate_metric(self, outputs: Dict, name: str) -> Tensor:
#         output = outputs[Denoiser.Cleaned]
#         clean = outputs[Denoiser.Ref].to(output.device)
#         mask = outputs[Denoiser.Mask].to(output.device)
#         #output = output * mask
#         #clean = clean * mask
#         if name == 'psnr':
#             metric = ssmd.utils.calculate_psnr(output, clean, mask)
#         elif name == 'ssim':
#             metric = ssmd.utils.calculate_ssim(output, clean, mask)
#         else:
#             raise('Metric Not Implemented')
#         return metric
    
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
    
    def reset_metric_dict(self,metric_dict: Dict):
        metric_dict["n"] = 0
        for key, metric in metric_dict.items():
            if isinstance(metric, Metric):
                metric.reset()
    @property
    def run_dir(self) -> str:       
        return self._run_dir
    
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
    
    def set_run_dir_path(self):
        if self._run_dir is None:
            if self.cfg.get(ConfigValue.TAG, None) is None:
                self.cfg[ConfigValue.TAG] = "none"
            run_dir_name = "{:05d}-eval-{}".format(self.next_run_id,self.cfg[ConfigValue.TAG])
            self._run_dir = run_dir_name
        return os.path.join(self.runs_dir, self._run_dir)
    
    def load_state_dict(self, state_dict: Union[Dict, str]):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location="cpu")
        self.denoiser = Denoiser.from_state_dict(state_dict["denoiser"],logger)
        
    def eval_state_str(self) -> str:
        prefix = "{:7} {:>5}".format("", "VALID")
        summary = "{} | ".format(prefix)
        
        metric_str_list = []
        eval_metrics =  self.eval_history
        
        for key, metric in eval_metrics.items():
            if isinstance(metric, Metric) and not metric.empty():
                metric_acc = metric.accumulated()
                metric_str_list += ["{} = {:.3f}".format(key, metric_acc)]

        summary += ", ".join(metric_str_list)
        return summary  

    def save_output(self, outputs, save_dir = str, file_name = str):
        out = {}
        out['cln'] = self.unpad_mat(outputs[Denoiser.Ref])
        out['inp'] = self.unpad_mat(outputs[Denoiser.Input])
        out['out'] = self.unpad_mat(outputs[Denoiser.Cleaned])
        output_dir = os.path.join(self.run_dir_path, save_dir)
        os.makedirs(output_dir, exist_ok=True)
        savemat(os.path.join(output_dir, file_name),out)
        
    def unpad_mat(self, img: Tensor):
        if len(img.shape) != 4:
            raise NotImplementedError("Unsupported image dimension")
        img = img.permute(2,3,1,0)
        img = img.cpu().detach().numpy()
        return img    

    def test_data(self) -> Tuple[DataLoader, NoiseWrapper]:
        cfg = self.cfg

        dataset = NoiseWrapper_dicom(
            file_path = self.dataset,
            mask_path = self.mask,
            transform = False,
            chan_frac = cfg[ConfigValue.CHAN_FRAC],
            noise_style = cfg[ConfigValue.NOISE_STYLE],
            algorithm = cfg[ConfigValue.ALGORITHM],
            training_mode = False,
            img_count = self.cfg[ConfigValue.TEST_LEN],
            )

        _ = dataset[0]
        self.test_count = len(dataset)
        
        if self.args.len is None:
            self.cfg[ConfigValue.TEST_LEN] = self.test_count           
            
        dataloader = DataLoader(
            dataset,
            batch_size = cfg[ConfigValue.TEST_MINIBATCH_SIZE],
            num_workers = cfg[ConfigValue.DATALOADER_WORKERS],
            pin_memory = True,
            shuffle = False,
        )
        return dataloader, dataset
    
def start(args):
    
    evaluator = DenoiserEvaluator(args) 
    evaluator.evaluate()

if __name__ == "__main__":
    start(parse())