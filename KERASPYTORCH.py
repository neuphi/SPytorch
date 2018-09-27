#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:46:07 2018

@author: felix
"""
################ IMPORTS ###################################

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as DataLoader
#from torch.utils.data import Dataset as Dataset
import time
from system.misc import *
from system.initnet import *

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import callbacks

#################GET SIMULATION DATA #######################

#simulate even denser than before, like 100000 samples or so
dataset = SimulateData()
training_set, validation_set, test_set = SplitData(dataset)

saveData([training_set, validation_set, test_set])

LEN_TRAINING_SET    = len(training_set)
LEN_VALIDATION_SET  = len(validation_set)
LEN_TEST_SET        = len(test_set)


training_set = ModDataTorch(training_set)
validation_set = ModDataTorch(validation_set)
test_set = ModDataTorch(test_set)

print ("DATA SIZE:")
print (training_set.size())
print (validation_set.size())
print (test_set.size())

tr_labels = torch.zeros(len(training_set[:,0]), DIM_OUT)
for i in range(len(training_set[:,0])): tr_labels[i] = training_set[i,2]

val_labels = torch.zeros(len(validation_set[:,0]), DIM_OUT)
for i in range(len(validation_set[:,0])): val_labels[i] = validation_set[i,2]

test_labels = torch.zeros(len(test_set[:,0]), DIM_OUT)
for i in range(len(test_set[:,0])): test_labels[i] = test_set[i,2]

############### PLOT 1: PT/SS ##############################

counter = 1
totalcount = len(LOSS_FUNCTIONS) * len(OPTIMIZERS) * len(MINIBATCH_SIZES) * \
len(LEARN_RATE) * len(ACTIVATION_FUNCTIONS) * len(SHAPES) * len(HID_LAY) * len(NOD)
starttime = time.time()

#train example keras model

#train example pytorch model

#Prediction Time over prediction sample size 

############### PLOT 2: Layers #############################

#loop over several Layer architectures

#PT&Loss over layer number

############### PLOT 3: Nodes ##############################

#loop over several node architectures

#PT&Loss over node number

############### PLOT 4: Activation Func ####################

#loop over several node architectures

#PT&Loss over different activ func
