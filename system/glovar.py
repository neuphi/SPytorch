import torch
import argparse
import torch
torch.multiprocessing.set_start_method("spawn")

# DECAY IDENTIFICATION

ANALYSIS_ID   = "CMS-PAS-SUS-12-026"
TXNAME        = "T1tttt"
MOTHER_LOW    = 600
MOTHER_UP     = 1100
MOTHER_STEP   = 5
LSP_LOW       = 0
LSP_STEP      = 10
EPOCH_NUM            = 50
ANALYSIS_SAMPLE_SIZE = 200
HYPERLOSS_FUNCTION   = "lin" #"lin", "exp"

CUDA                 = torch.cuda.is_available()

