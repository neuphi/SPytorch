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
t0 = time.time()
dataset = SimulateData()
timesim = time.time() - t0
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
############## INITIALIZE NET ##############################
############################################################

layers = 1
nodes = 20
activ = "rel"
shape = "trap"
loss_fn_i = "MSE"
optimizer_i = "Adam"
minibatch = 8
learning_rate = 1.000e-03

netdata = CreateNet(
        layers,            
        nodes,
        activ,
        shape,
        loss_fn_i,
        optimizer_i,
        minibatch,
        learning_rate
        )  
netdata['model'].to(device=args.device)
loss_fn = initloss(loss_fn_i)
optimizer = initopt(optimizer_i, netdata['model'], learning_rate)
#define trainloader
trainloader = DataLoader(training_set, batch_size=minibatch, shuffle=True, num_workers=0)
#loop over epochs
for epoch in range(EPOCH_NUM):
    print("epoch ", epoch, " nodes ", nodes, " layers ", layers, " shape ", shape)  
    #loop over trainloader
    for i, data in enumerate(trainloader):  
        #make data accassable for model
        inputs, labels = modelinputs(data)  
        #do predictions and calculate loss
        labels_pred = netdata["model"](inputs)
        loss_minibatch = loss_fn(labels_pred, labels)
        #make one step to optimize weights
        optimizer.zero_grad()
        loss_minibatch.backward()
        optimizer.step()
    if loss_fn_i == "MSE":
        #netdata["plytr"].append(np.sqrt(loss_fn(netdata["model"](training_set[:,:2]), training_set[:,2]).detach().numpy()))
        netdata["plytr"].append(np.sqrt(loss_fn(netdata["model"](training_set_var), tr_labels_var).cpu().detach().numpy()))
        netdata["plyte"].append(np.sqrt(loss_fn(netdata["model"](test_set_var), test_labels_var).cpu().detach().numpy()))
    else:
        netdata["plytr"].append(loss_fn(netdata["model"](training_set_var), tr_labels_var).cpu().detach().numpy())
        netdata["plyte"].append(loss_fn(netdata["model"](test_set_var), test_labels_var).cpu().detach().numpy())                      
    vall_dummy = loss_fn(netdata["model"](validation_set_var), validation_labels_var).cpu().detach().numpy()
    if netdata["lossv"] > vall_dummy:
        netdata["lossv"] = vall_dummy

epnum 		= len(netdata["plytr"])
x_axis     	= [i for i in range(epnum)]
y_axis_trn 	= netdata["plytr"]
y_axis_val 	= netdata["plyte"]

y_axis_trn[0] 	= y_axis_trn[1]
y_axis_val[0] 	= y_axis_val[1]

plt.figure(1)
plt.title('Loss Function', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.plot(x_axis, y_axis_trn, label = 'Train Loss')
plt.plot(x_axis, y_axis_val, label = 'Test Loss')
plt.legend()
plt.savefig(PATH_PLOTS + "LossFnWinner.eps", format = 'eps')

############################################################
############### PLOT MAP ###################################
############################################################

data_set_torch = ModDataTorch(dataset)
data_set_var = Variable(data_set_torch[:,:2])
t0 = time.time()
netdata["model"](data_set_var)
timeappr = time.time() - t0

print("Validation Loss:")	
print(netdata["lossv"])
print("Time Simulation")
print(timesim)
print("Time Approximation")
print(timeappr)

mother = list(range(MOTHER_LOW, MOTHER_UP, MOTHER_STEP))
LSP = list(range(LSP_LOW,MOTHER_UP,LSP_STEP))
upperlimits = np.zeros((len(LSP), len(mother)))
approxul = np.zeros((len(LSP), len(mother)))

for i in range(len(dataset)):
    upperlimits[len(LSP)-1-LSP.index(dataset[i][0][1])][mother.index(dataset[i][0][0])] = dataset[i][1]
    approxul[len(LSP)-1-LSP.index(dataset[i][0][1])][mother.index(dataset[i][0][0])] = netdata["model"](data_set_torch[i][0:2])

mother_ticks = []
LSP_ticks = []

for i in range(len(mother)):
    if (i % 10) == 0:
        mother_ticks.append(mother[i])
    else:
        mother_ticks.append(" ")

for i in range(len(LSP)):
    if (i % 10) == 0:
        LSP_ticks.append(LSP[len(LSP) - i -1])
    else:
        LSP_ticks.append(" ")


fig, ax = plt.subplots()
plt.xlabel('mother (GeV)')
plt.ylabel('LSP (GeV)')
im = ax.imshow(upperlimits)
# We want to show all ticks...
ax.set_xticks(np.arange(len(mother_ticks)))
ax.set_yticks(np.arange(len(LSP_ticks)))
# ... and label them with the respective list entries
ax.set_xticklabels(mother_ticks)
ax.set_yticklabels(LSP_ticks)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Cross Section Upper Limits")
fig.tight_layout()
fig.savefig(PATH_PLOTS + "Upperlimits.eps", format = 'eps')



fig2, ax2 = plt.subplots()
plt.xlabel('mother (GeV)')
plt.ylabel('LSP (GeV)')
im2 = ax2.imshow(approxul)
ax2.set_xticks(np.arange(len(mother_ticks)))
ax2.set_yticks(np.arange(len(LSP_ticks)))
ax2.set_xticklabels(mother_ticks)
ax2.set_yticklabels(LSP_ticks)
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax2.set_title("Approximated Cross Section Upper Limits")
fig2.tight_layout()
fig2.savefig(PATH_PLOTS + "Upperlimits_approx.eps", format = 'eps')

############################################################
############### RUNTIME COMP ###############################
############################################################

plt.figure()         
plt.title('Upper Limits vs Approximation', fontsize=20)
plt.text(170, 18, 'analysis ID: ' + str(ANALYSIS_ID) + '\ntopology: ' + str(TXNAME) + '\nmass mother: 900 GeV', style='italic',
        bbox={'facecolor':'gray', 'alpha':0.2, 'pad':10})
plt.xlabel('Mass LSP (GeV)')
plt.ylabel('Upper Limits (UL) (fb)')

plt.plot([i*LSP_STEP for i in range(len(upperlimits[90]))], upperlimits[90], label = 'Cross Section Upper Limit')
plt.plot([i*LSP_STEP for i in range(len(upperlimits[90]))], approxul[90], c = 'r', label = 'Approximation')
plt.plot([i*LSP_STEP for i in range(len(upperlimits[90]))], abs(upperlimits[90] - approxul[90]), c = 'g', label = 'Absolute Error')
plt.legend()
plt.savefig("analysis/plots/heatmapslice.eps", format='eps')

