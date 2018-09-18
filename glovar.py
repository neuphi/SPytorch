# SET PATHS

PATH_DATABASE = "database/"
PATH_DATA     = "data/"
PATH_PLOTS    = "analysis/plots/"
PATH_LOGS     = "analysis/logs/"

# DECAY IDENTIFICATION

ANALYSIS_ID   = "CMS-PAS-SUS-12-026"
TXNAME        = "T1tttt"

# SPLIT INFORMATION

SPLIT_CHOOSE = 1   #1 for split, 0 for no split
SPLIT = [80, 0, 20]   #train/val/test

############

# PICK RESULT

EXP = "CMS-PAS-SUS-12-026"
TX  = "T1tttt"

############

# CONFIGURE PYTORCH

MINI_BATCH_SIZE     = 32
DIM_IN              = 2
DIM_HIDDEN_1        = 4
DIM_HIDDEN_2        = 16
DIM_HIDDEN_3        = 4
DIM_OUT             = 1
BATCH_SIZE_VAL      = 59
EPOCH_NUM           = 100
ANALYSIS_SAMPLE_SIZE = 10

############

# HYPERPARAMETERS

INT_LOSS = 5
INT_LOSS_SQ = 25
HID_LAY_MAX = 6
HID_LAY_MIN = 4
HID_LAY_STEP = 2
NOD_MAX = 6
NOD_MIN = 4
NOD_STEP = 2
LEARN_RATE     = 1e-3

netdata = {
        "model": 0,
        "layer": 2,
        "nodes": 2,
        "activ": "lin",
        "shape": "trap",
        "batch": 0
        "lrate": 1e-3
        "lossf": 0,
        "optim": 0,
        "hloss": 1e5,
        "lossv": 1e5,
        "predt": 1e5,
        "plytr": [],
        "plyva": []
}