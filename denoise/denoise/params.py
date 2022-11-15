"""
# Description:
#  Hyperparameters for training & running code example
#
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#

conda activate ssmd
cd /home/juhyung/denoise/denoise
python train.py --gpu 1

"""

import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train')
    
    ## dataset && network ##
    #parser.add_argument("--train_dataset","-t", default = {'/home/juhyung/data/viet/pre_hcl/test/'})
    #parser.add_argument("--valid_dataset","-v", default = {'/home/juhyung/data/viet/pre_hcl/valid/'})
    parser.add_argument("--train_dataset","-t", default = '/fast_storage/braintrain_real_v3.h5')
    parser.add_argument("--train_mask",default = None)
    parser.add_argument("--valid_dataset","-v", default = '/home/juhyung/ssmd/ssmd/data/brainval_v2.h5')
    parser.add_argument("--valid_mask",default ='/home/juhyung/ssmd/ssmd/data/brainval_v2_b.h5')
    parser.add_argument("--data_type", type=str, default='*.h5')
    parser.add_argument("--run_dir","-dir",type = str, default = '/home/juhyung/denoise/denoise/log/')
    parser.add_argument("--network","-net", default = 'dncnn')

    ## trainig parameters ##
    parser.add_argument("--noise",type=str, default = 'inf')
    parser.add_argument("--gpu", type=str, default = '0')
    parser.add_argument("--batch",type=int,default= 1)
    parser.add_argument("--lr", type=float, default = 1e-4)
    parser.add_argument("--lr_decay", type=float, default = 0.88)
    parser.add_argument("--lr_tol", type=int, default = 2)
    parser.add_argument("--train_epoch", type=int, default = 100)
    
    ## others ##
    parser.add_argument("--loglv", type=int, default = 10)
    parser.add_argument("--check_num", type=int, default = 4)
    parser.add_argument("--eval_interval", type=int, default = 1)
    parser.add_argument("--num_workers", type=int, default = 4)
    parser.add_argument("--save_val", type=str, default ='yes')
    parser.add_argument("--printing_mode", type=str, default ='no')
    parser.add_argument("--parallel", type=str, default ='no')
    parser.add_argument("--tag", type=str, default='channel1')
    
    
    args = parser.parse_args()
    return args
        