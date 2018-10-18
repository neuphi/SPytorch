#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 06:46:26 2018

@author: felix
"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader as DataLoader
#from torch.utils.data import Dataset as Dataset
import time
from system.misc import *
from system.initnet import *
from system.printtorch import *
from mpl_toolkits.mplot3d import Axes3D

############## DATA ############################################

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

tr_labels = torch.zeros(len(training_set[:,0]), DIM_OUT, device=args.device)
for i in range(len(training_set[:,0])): tr_labels[i] = training_set[i,2]

val_labels = torch.zeros(len(validation_set[:,0]), DIM_OUT, device=args.device)
for i in range(len(validation_set[:,0])): val_labels[i] = validation_set[i,2]

test_labels = torch.zeros(len(test_set[:,0]), DIM_OUT, device=args.device)
for i in range(len(test_set[:,0])): test_labels[i] = test_set[i,2]

training_set_var = Variable(training_set[:,:2])
tr_labels_var = Variable(tr_labels)
test_set_var = Variable(test_set[:,:2])
test_labels_var = Variable(test_labels)
validation_set_var = Variable(validation_set[:,:2])
validation_labels_var = Variable(val_labels)

if CUDA:
    training_set_var.to(device=args.device)
    tr_labels_var.to(device=args.device)
    test_set_var.to(device=args.device)
    test_labels_var.to(device=args.device)
    validation_set_var.to(device=args.device)
    validation_labels_var.to(device=args.device)


############# PLOTS ######################################

dataset_plot_x1 = []
dataset_plot_x2 = []
dataset_plot_y = []
for i in range(len(dataset)):
    dataset_plot_x1.append(dataset[i][0][0])        
    dataset_plot_x2.append(dataset[i][0][1])        
    dataset_plot_y.append(dataset[i][1])        
    
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')       
ax = Axes3D.scatter(xs=dataset_plot_x1, ys=dataset_plot_x2, zs=dataset_plot_y, zdir='z', 
               s=20, c=None, depthshade=True)
#ax.set(xlabel='Hidden Layers', ylabel='Prediction Time Keras (s)')
#ax.grid()
#ax.legend()
#fig.savefig(PATH_PLOTS + "KerasPyTorch/LayersKeras.eps", format = 'eps')