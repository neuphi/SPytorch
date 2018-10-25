import numpy as np
import torch
import argparse
import torch
torch.multiprocessing.set_start_method("spawn")

#SET DEVICE

parser = argparse.ArgumentParser(description='PyTorch Gridsearch')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    devicenmbr = input("Which GPU to use?")
    args.device = torch.device('cuda:'+devicenmbr)
else:
    args.device = torch.device('cpu')

# SET PATHS

PATH_DATABASE = "database/"
PATH_DATA     = "data/"
PATH_PLOTS    = "analysis/plots/"
PATH_LOGS     = "analysis/logs/"

# DECAY IDENTIFICATION

ANALYSIS_ID   = "CMS-PAS-SUS-12-026"
TXNAME        = "T1tttt"
MOTHER_LOW    = 600
MOTHER_UP     = 1100
MOTHER_STEP   = 5
LSP_LOW       = 0
LSP_STEP      = 10

# SPLIT INFORMATION

SPLIT_CHOOSE = 1   #1 for split, 0 for no split
SPLIT = [80, 10, 10]   #train/val/test

LEN_TEST_SET	 	= 0
LEN_TRAINING_SET 	= 0
LEN_VALIDATION_SET 	= 0

############

# PICK RESULT

EXP = "CMS-PAS-SUS-12-026"
TX  = "T1tttt"

############

# CONFIGURE PYTORCH

CUDA                 = torch.cuda.is_available()
MINI_BATCH_SIZE      = 32
DIM_IN               = 2
DIM_HIDDEN_1         = 4
DIM_HIDDEN_2         = 16
DIM_HIDDEN_3         = 4
DIM_OUT              = 1
BATCH_SIZE_VAL       = 59
EPOCH_NUM            = 200
ANALYSIS_SAMPLE_SIZE = 100
HYPERLOSS_FUNCTION   = "lin" #"lin", "exp"

############

# HYPERPARAMETERS

INT_LOSS = 5
INT_LOSS_SQ = 25
HID_LAY_MAX = int(input("HID_LAY_MAX?"))
HID_LAY_MIN = int(input("HID_LAY_MIN?"))
HID_LAY_STEP = 1
HID_LAY = range(HID_LAY_MIN,HID_LAY_MAX+1,HID_LAY_STEP)
NOD_MAX = 20
NOD_MIN = 20
NOD_STEP = 4
NOD = range(NOD_MIN, NOD_MAX+1,NOD_STEP)
LR_MIN = 1e-3
LR_MAX = 1.0001e-1
LR_STEP = 9e-3
LEARN_RATE     = [1e-3]#np.arange(LR_MIN, LR_MAX, LR_STEP)
LOSS_FUNCTIONS = ["MSE"] #["MSE", "L1"]
OPTIMIZERS = ["Adam"] #["Adam", "SGD"]
MINIBATCH_SIZES = [1,8,32,64,128,256,512,1024]
#MINIBATCH_SIZES = []
#MINIBATCH_SIZES.append(int(input("Minibatch Size?")))
ACTIVATION_FUNCTIONS = ["rel"] #["lin","rel", "tah", "sig"]
SHAPES = ["trap"] #["lin","trap","ramp"]
#netdata = {}
