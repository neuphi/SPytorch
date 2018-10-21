#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 06:46:26 2018

@author: felix
"""
################ IMPORTS ###################################

import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as DataLoader
#from torch.utils.data import Dataset as Dataset
import time
from system.misc import *
from system.initnet import *

from mpl_toolkits.mplot3d import Axes3D

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import callbacks

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import callbacks

import statistics as stat
from scipy import stats

################# PARAMETERS ###############################


#################GET SIMULATION DATA #######################

#simulate even denser than before, like 100000 samples or so
dataset = SimulateData()
training_set, validation_set, test_set = SplitData(dataset)

saveData([training_set, validation_set, test_set])

LEN_TRAINING_SET    = len(training_set)
LEN_VALIDATION_SET  = len(validation_set)
LEN_TEST_SET        = len(test_set)

#model keras data
training_data_list, training_labels_list = ModDataKeras(training_set)
validation_data_list, validation_labels_list = ModDataKeras(validation_set)
test_data_list, test_labels_list = ModDataKeras(test_set)

#model torch data
training_set = ModDataTorch(training_set)
validation_set = ModDataTorch(validation_set)
test_set = ModDataTorch(test_set)

tr_labels = torch.zeros(len(training_set[:,0]), DIM_OUT)
for i in range(len(training_set[:,0])): tr_labels[i] = training_set[i,2]
val_labels = torch.zeros(len(validation_set[:,0]), DIM_OUT)
for i in range(len(validation_set[:,0])): val_labels[i] = validation_set[i,2]
test_labels = torch.zeros(len(test_set[:,0]), DIM_OUT)
for i in range(len(test_set[:,0])): test_labels[i] = test_set[i,2]

if CUDA:
    training_set_var = Variable(training_set[:,:2].cuda())
    tr_labels_var = Variable(tr_labels.cuda())
    test_set_var = Variable(test_set[:,:2].cuda())
    test_labels_var = Variable(test_labels.cuda())
    validation_set_var = Variable(validation_set[:,:2].cuda())
    validation_labels_var = Variable(val_labels.cuda())
else:
    training_set_var = Variable(training_set[:,:2])
    tr_labels_var = Variable(tr_labels)
    test_set_var = Variable(test_set[:,:2])
    test_labels_var = Variable(test_labels)
    validation_set_var = Variable(validation_set[:,:2])
    validation_labels_var = Variable(val_labels)

#verbose
print ("DATA SIZE:")
print (training_set.size())
print (validation_set.size())
print (test_set.size())

############################################################
############### PLOT    ##############################
############################################################

dataset_plot_x1 = []
dataset_plot_x2 = []
dataset_plot_y = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')       
print("start")

for i in range(len(dataset)):
    dataset_plot_x1.append(dataset[i][0][0])        
    dataset_plot_x2.append(dataset[i][0][1])        
    dataset_plot_y.append(dataset[i][1])        
    ax.scatter(dataset_plot_x1, dataset_plot_x2, dataset_plot_y, c='r', marker='o')

print("1")
ax.set_xlabel('mother mass')
print("2")
ax.set_ylabel('LSP mass')
print("3")
ax.set_zlabel('cross section upper limit')
print("4")
fig.savefig(PATH_PLOTS + "scatter3d.eps", format = 'eps')
print("5")
