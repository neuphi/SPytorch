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

from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset as Dataset

######################## LOAD DATA ################################

tr_data,tr_labels,val_data,val_labels = loadData()

SAMPLE_NMBR    = len(tr_data)
SAMPLE_NMBR_VAL= len(val_data)

X={} #create dictionary
for d,l in zip (tr_data,tr_labels):
    X[Hash(d)]=l

###################### INITIALIZE NN ############################

model = Net()
print("\n",model)

loss_fn = nn.MSELoss(size_average=True, reduce=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

###################### MODIFY DATA FOR NN #########################

tr_data_torch    = torch.zeros(SAMPLE_NMBR, DIM_IN)
tr_labels_torch  = torch.zeros(SAMPLE_NMBR, DIM_OUT)
tr_all           = torch.zeros(SAMPLE_NMBR, DIM_IN + DIM_OUT)
val_data_torch   = torch.zeros(SAMPLE_NMBR_VAL, DIM_IN)
val_labels_torch = torch.zeros(SAMPLE_NMBR_VAL, DIM_OUT)

for i in range(SAMPLE_NMBR):
  tr_data_torch[i]   = torch.from_numpy(tr_data[i])
  tr_labels_torch[i] = tr_labels[i]
  tr_all[i][0] = tr_data_torch[i][0]
  tr_all[i][1] = tr_data_torch[i][1]
  tr_all[i][2] = tr_labels_torch[i][0]

for i in range(SAMPLE_NMBR_VAL):
  val_data_torch[i]   = torch.from_numpy(val_data[i])
  val_labels_torch[i] = val_labels[i]

###################### FIT NN #####################################

print ( "Now fitting ... " )

trainloader = DataLoader(tr_all, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)

loss_plot_x = [] #save lists for loss plot
loss_plot_y = []

for t in range(EPOCH_NUM):

    for i, data in enumerate(trainloader):
      inputs    = torch.zeros(torch.numel(data[:,0]), DIM_IN)
      labels    = torch.zeros(torch.numel(data[:,0]), DIM_OUT)
      for j in range(torch.numel(data[:,0])):
        inputs[j][0] = data[j][0]
        inputs[j][1] = data[j][1]
        labels[j]    = data[j][2]
      labels_pred = model(tr_data_torch)
      loss = loss_fn(labels_pred, tr_labels_torch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    loss_plot_x.append(t) #fill loss lists with data 
    loss_plot_y.append(loss)

print("\nloss:", loss.item())

print ( "Done fitting" )

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

