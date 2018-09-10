# SET PATHS

PATH_DATABASE = "database/"
PATH_DATA     = "data/"
PATH_PLOTS    = "analysis/plots/"
PATH_LOGS     = "analysis/logs/"

# DECAY IDENTIFICATION

ANALYSIS_ID   = "CMS-PAS-SUS-12-026"
TXNAME        = "T1tttt"

# SPLIT INFORMATION

#SPLIT_CHOOSE = 1   #1 for split, 0 for no split
#SPLIT = [60, 20, 20]

############

# PICK RESULT

EXP = "CMS-PAS-SUS-12-026"
TX  = "T1tttt"

############

# CONFIGURE PYTORCH

BATCH_SIZE     = 50
DIM_IN         = 2
DIM_HIDDEN_1   = 4
DIM_HIDDEN_2   = 16
DIM_HIDDEN_3   = 4
DIM_OUT        = 1
BATCH_SIZE_VAL = 59
LEARN_RATE     = 1e-3
EPOCH_NUM      = 100

############

# HYPERPARAMETERS

INT_LOSS_SQ = 25
HID_LAY_MAX = 16
HID_LAY_MIN = 1
NOD_MAX = 32
NOD_MIN = 4
