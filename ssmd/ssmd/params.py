from __future__ import annotations

from enum import Enum, auto

class ConfigValue(Enum):
    TRAIN_EPOCH = auto()
    TRAIN_MINIBATCH_SIZE = auto()
    TEST_MINIBATCH_SIZE = auto()
    PATCH_SIZE = auto()
    PATCH_STRIDE = auto()
    LEARNING_RATE = auto()
    LR_DOWN = auto()
    LR_COEFF = auto()
    EVAL_INTERVAL = auto()
    PRINT_INTERVAL = auto()
    DATALOADER_WORKERS = auto()
    PARALLEL = auto()
    LOAD_MEMORY = auto()
    LOG_LEVEL = auto()
    CHAN_FRAC = auto()
    TEST_LEN = auto()
    TRAIN_DATA_PATH = auto()
    TEST_DATA_PATH = auto()
    TRAIN_MASK_PATH = auto()
    TEST_MASK_PATH = auto()
    LOG_PATH = auto()
    TAG = auto()
    NET_ARCHITECTURE = auto()
    ALGORITHM = auto()
    PIPELINE = auto()
    NOISE_STYLE = auto()
    NOISE_VALUE = auto()
    
class Algorithm(Enum):
    SSDN = "ssdn"
    N2N = "n2n"
    N2C = "n2c"
    N2V = "n2v"
    C2C= "c2c"
    N2S= "n2s"
    CS= "cs"
    N2SA = 'n2sa'
    NE2NE = "ne2ne"
    
class Network(Enum):
    UNET = "unet"
    DNCNN = "dncnn"
    
class StateValue(Enum):
    ITERATION = auto()
    EPOCH = auto()
    PREV_PSNR = auto()
    PREV_ITER = auto()
    REFERENCE = auto()
    TRAIN = auto()
    EVAL = auto()

    
