"""
# Description:
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
"""

import argparse

import ssmd
from ssmd.params import Algorithm, ConfigValue, Network
from ssmd.denoisertrainer import DenoiserTrainer
from ssmd.utils.utils import str2bool

"""
Command example
python train.py -dir D:/gate/denoising/log/ -a n2c -t D:/gate/denoising/data/braintrain.h5 -v D:/gate/denoising/data/brainval.h5 -noise gauss1


python eval.py -dir D:/gate/denoising/logeval/ -m D:/gate/denoising/model.training -d D:/gate/denoising/data/brainval.h5 -noise inf

"""

def base():
    return {
        ConfigValue.TRAIN_EPOCH: 500,
        ConfigValue.TRAIN_MINIBATCH_SIZE: 12,
        ConfigValue.TEST_MINIBATCH_SIZE: 4,
        ConfigValue.PATCH_SIZE: 320,
        ConfigValue.LEARNING_RATE: 1e-4,
        ConfigValue.LR_DOWN: 5,
        ConfigValue.LR_COEFF: 0.87,
        ConfigValue.EVAL_INTERVAL: 5,
        ConfigValue.PRINT_INTERVAL: 2,
        ConfigValue.DATALOADER_WORKERS: 4,
        ConfigValue.PARALLEL: True,
        ConfigValue.LOG_LEVEL: 20,
        ConfigValue.CHAN_FRAC: 0.5,
        ConfigValue.TEST_LEN: None,
        ConfigValue.TRAIN_DATA_PATH: None,
        ConfigValue.TRAIN_MASK_PATH: None,
        ConfigValue.TEST_DATA_PATH: None,
        ConfigValue.TEST_MASK_PATH: None,
        ConfigValue.LOG_PATH: None,
        ConfigValue.TAG: None,
        ConfigValue.NET_ARCHITECTURE: None,
    }

def parse():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--train_dataset","-t", default = None)
    parser.add_argument("--train_mask","-tm")
    parser.add_argument("--validation_dataset","-v", default = None)
    parser.add_argument("--validation_mask","-vm")
    parser.add_argument("--runs_dir","-dir",type = str)
    parser.add_argument("--algorithm","-a",choices=[c.value for c in ssmd.utils.list_constants(Algorithm)],default='c2c')
    parser.add_argument("--noise_style","-noise",default='gauss1.5')
    parser.add_argument("--network","-net",choices=[c.value for c in ssmd.utils.list_constants(Network)])
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--parallel", type=str)
    parser.add_argument("--loglv", type=int)
    parser.add_argument("--len", type=int)
    parser.add_argument("--gpu", type=str, default = '0')
    args = parser.parse_args()
    return args

def start(args):

    cfg = base()
    if args.algorithm is not None:
        cfg[ConfigValue.ALGORITHM] = Algorithm(args.algorithm) 
    if args.noise_style is not None:
        cfg[ConfigValue.NOISE_STYLE] = args.noise_style 
    if args.network is not None:
        cfg[ConfigValue.NET_ARCHITECTURE] = Network(args.network)
    else:
        cfg[ConfigValue.NET_ARCHITECTURE] = Network("dncnn")  
    if args.runs_dir is not None:
        cfg[ConfigValue.LOG_PATH] = args.runs_dir
    if args.batch is not None:
        cfg[ConfigValue.TRAIN_MINIBATCH_SIZE] = args.batch
    if args.patch_size is not None:
        cfg[ConfigValue.PATCH_SIZE] = args.patch_size
    if args.tag is not None:
        cfg[ConfigValue.TAG] = args.algorithm + "_" + args.tag
    elif args.algorithm is not None:
        cfg[ConfigValue.TAG] = args.algorithm
    if args.parallel is not None:
        cfg[ConfigValue.PARALLEL] = str2bool(args.parallel)
    if args.loglv is not None:
        cfg[ConfigValue.LOG_LEVEL] = args.loglv
    if args.len is not None:
        cfg[ConfigValue.TEST_LEN] = args.len
    if args.train_dataset is not None:
        cfg[ConfigValue.TRAIN_DATA_PATH] = args.train_dataset
    if args.train_mask is not None:
        cfg[ConfigValue.TRAIN_MASK_PATH] = args.train_mask
    if args.validation_dataset is not None:
        cfg[ConfigValue.TEST_DATA_PATH] = args.validation_dataset
    if args.validation_mask is not None:
        cfg[ConfigValue.TEST_MASK_PATH] = args.validation_mask
        
    trainer = DenoiserTrainer(cfg, runs_dir=args.runs_dir, gpu = args.gpu)            
    trainer.train()

if __name__ == "__main__":
    start(parse())
    
