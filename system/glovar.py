import torch
import argparse
import torch


# DECAY IDENTIFICATION

ANALYSIS_ID   = "CMS-PAS-SUS-12-026"#"CMS-PAS-SUS-13-018"#"CMS-PAS-SUS-13-016"
TXNAME        = "T1tttt"#"T2bb"
MOTHER_LOW    = 600
MOTHER_UP     = 1100
MOTHER_STEP   = 5
LSP_LOW       = 0
LSP_STEP      = 10
EPOCH_NUM            = 200
ANALYSIS_SAMPLE_SIZE = 200
HYPERLOSS_FUNCTION   = "lin" #"lin", "exp"

#CUDA                 = torch.cuda.is_available()

