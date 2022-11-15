"""
# Description:
#  Evaluation main code & running example 
#
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr

conda activate denoise
cd /home/juhyung/gate/denoise/denoise
python test_chan1.py --gpu 1

"""


import time
import logging
import os
import argparse
import mat73
import numpy as np
from tqdm import tqdm
from scipy.io import savemat
from scipy import io
import torch
from torch.utils.data import DataLoader, Dataset

from network import Network
from utils import str2bool, separator, logging_helper, seconds_to_dhms

logger = logging.getLogger("denoise.eval")

def parse():
    parser = argparse.ArgumentParser(description='Train')
    
    ### data & directory ###
    parser.add_argument("--model", default = '/home/juhyung/gate/denoise/denoise/model.training') # model path
    parser.add_argument("--dataset","-d", default = '/home/juhyung/data/fordenoising.mat')  # data path
    parser.add_argument("--run_dir","-dir", default = '/home/juhyung/gate/denoise/denoise/log/') # output path
    parser.add_argument("--mat_key", default = 'sus_recon') # data key
    
    ### parameters ###
    parser.add_argument("--gpu", type=str, default = '0')
    parser.add_argument("--batch",type=int, default=1)
    parser.add_argument("--len",type=int, default = None)
    parser.add_argument("--loglv", type=int, default = 20)
    parser.add_argument("--num_workers", type=int, default = 4)
    parser.add_argument("--save_val", type=str, default ='yes')
    parser.add_argument("--printing_mode", type=str, default ='no')    
    
    args = parser.parse_args()
    return args

class DataWrapper(Dataset):
    Input = 0
    Refer = 1
    Index = 2

    
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
        
        self.scaling = 1
        try:
            self.files_raw = mat73.loadmat(self.file_path)[self.args.mat_key] * self.scaling
        except:
            self.files_raw = io.loadmat(self.file_path)[self.args.mat_key] * self.scaling

        #self.files_raw = abs(self.files_raw)
        self.files_raw = torch.from_numpy(self.files_raw).type(torch.float)
        
        self.file_size = self.files_raw.shape

        if len(self.file_size) == 3:
            self.max_int = self.files_raw.max()
            self.min_int = self.files_raw.min()  ## use if QSM processing
            #self.min_int = 0
            self.files_raw = (self.files_raw - self.min_int) / (self.max_int - self.min_int)
        elif len(self.file_size) == 4:
            files_view = self.files_raw.reshape(-1,self.file_size[-1])
            self.max_int = files_view.max(0).values
            self.min_int = files_view.min(0).values ## use if QSM processing
            #self.min_int = 0
            self.files_raw = (self.files_raw - self.min_int) / (self.max_int - self.min_int)
            self.files_raw = self.files_raw.reshape([self.file_size[0],self.file_size[1],-1])
        else:
            raise('cannot input except 3 or 4 dim')
        
        self.img_count = self.files_raw.shape[2]

        self.files = self.files_raw.unsqueeze(0)


    def __getitem__(self, index: int):           
        _img = self.files[:,:,:,index]        

        _ref = _img.clone() 
        return (_img,
                _ref,
                index) 
    

    def image_size(self):
        return torch.tensor(self.__getitem__(0)[0].shape)

    def __len__(self):
        return self.img_count
    
class Evaluator:
    def __init__(self, args):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu 
        
        self.args = args
        
        state_dict = torch.load(self.args.model, map_location="cpu")
        self.network = Network.from_state_dict(state_dict,logger,chan_num=1)
        
        ## experiment path setup ##
        self.run_dir = self.args.run_dir
        self.next_run_id = self.call_next_id()
        self.run_dir_path = os.path.join(self.run_dir,'{:05d}-eval'.format(self.next_run_id))

        ### state init ###
        self.eval_state = {"image_cnt": 0}   
        self.separator = separator
        self.init_time = time.time()
        
    def evaluate(self):
        if self.network is None:
            raise RuntimeError("Network not initialised for evaluation")
            
        logging_helper.setup(args = self.args,
                                     log_dir = self.run_dir_path,
                                     printing_mode = self.args.printing_mode)
        
        ### Hyperparameter define ###
        if self.args.printing_mode != 'yes':
            logger.info(self.separator())
            logger.info("{:05d}-eval".format(self.next_run_id))  
            
        logger.info(self.separator())
        logger.info("Trianing argpaser Info")
        for key, value in vars(self.network.args).items():
            logger.info('{:15s}: {}'.format(key,value))
            
        logger.info(self.separator())
        logger.info("ArgPaser Info")
        for key, value in vars(self.args).items():
            logger.info('{:15s}: {}'.format(key,value))
            
        logger.info(self.separator())
        logger.info("Loading Test Dataset...")
        self.testloader, self.testdataset, self.test_count= self.test_data(self.args.dataset)
        logger.info("Test dataset shape : {}".format(self.testdataset.file_size))
        logger.info("Test dataset length : {}".format(self.test_count))
        logger.info("Loaded Test Dataset.")
      
        
        logger.info(self.separator())
        logger.info("EVALUATION START")
        
        logger.info(self.separator())
        self.network.eval()
        idx = 0
        
        valid_tqdm = self.tqdm_loader(self.testloader)
        
        result_inp = torch.zeros(self.testdataset.files.shape)
        result_out = torch.zeros(self.testdataset.files.shape)
        
        for data in valid_tqdm:
            with torch.no_grad():
                image_count = data[DataWrapper.Input].shape[0]
                self.eval_state["image_cnt"] += image_count
                outputs = self.network(data)

                
                for i in range(image_count):
                    if ((self.args.save_val == 'yes') and
                        (self.args.printing_mode != 'yes')):     
                        result_inp[:,:,:,data[DataWrapper.Index].item()] = outputs[Network.Input].cpu().detach()
                        result_out[:,:,:,data[DataWrapper.Index].item()] = outputs[Network.Output].cpu().detach()

              
                idx += image_count

        if len(self.testdataset.file_size) == 4:
            result_inp = result_inp.reshape(self.testdataset.file_size)
            result_out = result_out.reshape(self.testdataset.file_size)

        result_inp = result_inp * (self.testdataset.max_int - self.testdataset.min_int) + self.testdataset.min_int
        result_out = result_out * (self.testdataset.max_int - self.testdataset.min_int) + self.testdataset.min_int
        
        file_name = self.args.dataset.split('/')[-1][:-4] + '_result' + '.mat'
        os.makedirs(self.run_dir_path, exist_ok=True)
        out = {}
        out['inp'] = result_inp.numpy()
        out['out'] = result_out.numpy()        
        savemat(os.path.join(self.run_dir_path, file_name),out)

        logger.info(self.eval_state_str())
        logger.info(self.separator())
        logger.info("EVALUATION FINISHED")
        logger.info(self.separator())

    
    def tqdm_loader(self, dataloader):
        return tqdm(dataloader, 
                    ncols = 80,
                    ascii = True, 
                    desc = 'Test ')
    
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
        
    def eval_state_str(self):
        summary = '{:>5} | '.format('EVAL RESULT')       
        summary += '{} | '.format(seconds_to_dhms(time.time() - self.init_time))
        
        image_cnt = self.eval_state['image_cnt']
        for key, metric in self.eval_state.items():
            if key != 'image_cnt':
                summary += '{}: {:0.3e} '.format(key, metric/image_cnt)    
                
        
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
    
    def set_run_dir_path(self):
        if self._run_dir is None:
            run_dir_name = "{:05d}-eval".format(self.next_run_id)
            self._run_dir = run_dir_name
        return os.path.join(self.runs_dir, self._run_dir)

        

    def test_data(self, path, mask_path = None):
        
        dataset = DataWrapper(
            file_path = path, 
            mask_path = mask_path,
            args = self.args,
            training_mode = False
        )
        
        _ = dataset[0]
        
        dataloader = DataLoader(
            dataset, 
            batch_size = self.args.batch,
            num_workers = self.args.num_workers, 
            pin_memory = True,
            shuffle = False
        )
        
        return dataloader, dataset, len(dataset)
    

if __name__ == "__main__":
    evaluator = Evaluator(parse()) 
    evaluator.evaluate()

    
    