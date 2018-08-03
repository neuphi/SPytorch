#!/usr/bin/python3

# -*- coding: utf-8 -*-

####################### IMPORT STUFF ############################

import torch
import numpy as np
from misc import *
from glovar import *

import torch.nn as nn
import torch.nn.functional as F

import visdom
import matplotlib.pyplot as plt

###################### DEFINE CLASS ##############################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #super().__init__()
        self.hidden1 = nn.Sequential(nn.Linear(DIM_IN, DIM_HIDDEN_1), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(DIM_HIDDEN_1, DIM_HIDDEN_2), nn.ReLU())
        self.hidden3 = nn.Sequential(nn.Linear(DIM_HIDDEN_2, DIM_HIDDEN_3), nn.ReLU())
        self.out = nn.Linear(DIM_HIDDEN_3, DIM_OUT)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x

######################## LOAD DATA ################################

tr_data,tr_labels,val_data,val_labels = loadData()

X={} #create dictionary
for d,l in zip (tr_data,tr_labels):
    X[Hash(d)]=l

###################### INITIALIZE NN ############################

model = Net()
print("\n",model)

loss_fn = nn.L1Loss(size_average=True, reduce=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

###################### MODIFY DATA FOR NN #########################

tr_data_torch    = torch.zeros(BATCH_SIZE, DIM_IN)
tr_labels_torch  = torch.zeros(BATCH_SIZE, DIM_OUT)
val_data_torch   = torch.zeros(BATCH_SIZE_VAL, DIM_IN)
val_labels_torch = torch.zeros(BATCH_SIZE_VAL, DIM_OUT)

for i in range(BATCH_SIZE):
  tr_data_torch[i]   = torch.from_numpy(tr_data[i])
  tr_labels_torch[i] = tr_labels[i]

for i in range(BATCH_SIZE_VAL):
  val_data_torch[i]   = torch.from_numpy(val_data[i])
  val_labels_torch[i] = val_labels[i]

###################### FIT NN #####################################

print ( "Now fitting ... " )

loss_plot_x = [] #save lists for loss plot
loss_plot_y = []

for t in range(EPOCH_NUM):

    labels_pred = model(tr_data_torch)
    loss = loss_fn(labels_pred, tr_labels_torch)

    loss_plot_x.append(t) #fill loss lists with data 
    loss_plot_y.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print ( "Done fitting" )

print("\nloss:", loss.item())

################# VISUALIZE LOSS ##################################

plt.figure(0)  
plt.title('Loss Function', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Value of Loss Function')
plt.plot(loss_plot_x, loss_plot_y)
plt.savefig("analysis/plots/loss.png")

################# MAKE EXAMPLE PREDICTIONS ########################
mass = torch.tensor( [ [ 600, 200 ], [ 700, 200 ], [ 800, 200 ], [ 900, 200 ], [ 1000, 200 ] ] )
preds=model( mass.type(torch.FloatTensor) )
print ( "Now predict" ) 

for m,p in zip ( mass,preds ):
    print ( "%s -> %s, %s" % ( m,p, X[Hash(m)]) )

################ SAVE MODEL ##################################

torch.save(model.state_dict(),"data/torchmodel.h5")

