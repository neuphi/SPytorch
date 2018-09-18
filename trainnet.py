#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:59:21 2018

@author: felix
"""
####################### IMPORT STUFF ############################

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset as Dataset
import time
from misc import *
from hyperloss import *
from initnet import *
from printtorch import *

######################## LOAD DATA ################################

#run obtain numbers

tr_data,tr_labels,val_data,val_labels = loadData()

SAMPLE_NMBR    = len(tr_data)
SAMPLE_NMBR_VAL= len(val_data)
    
###################### MODIFY DATA FOR NN #########################

tr_data_torch, tr_labels_torch, val_data_torch, val_labels_torch = ModDataTorch(tr_data,tr_labels,val_data,val_labels)

##################### SPLIT DATA #################################

#if SPLIT_CHOOSE = 1, split Data into SPLIT
#val data is already simulated

##################### TRAINING ###################################

for loss_fn_i in LOSS_FUNCTIONS:
  for optimizer_i in OPTIMIZERS:
    for minibatch in MINIBATCH_SIZES:
      for learning_rate in range(LR_MIN, LR_MAX, LR_STEP):
        for activ in ACTIVATION_FUNCTIONS:
          for shape in SHAPES:
            for layers in range(HID_LAY_MIN,HID_LAY_MAX+1,HID_LAY_STEP):
              for nodes in range(NOD_MIN, NOD_MAX+1,NOD_STEP):
                #initialize net, lossfunction and optimizer
                netdata = CreateNet(
                        layers,            
                        nodes,
                        activ,
                        shape,
                        loss_fn,
                        optimizer
                        )
                loss_fn = initloss(loss_fn_i)
                optimizer = initopt(optimizer_i, netdata['model'], learning_rate)
                #define trainloader
                trainloader = DataLoader(tr_data_torch, batch_size=minibatch, shuffle=True, num_workers=4)
                #loop over epochs
                for epoch in range(EPOCH_NUM):
                  #loop over trainloader
                  for i, data in enumerate(trainloader):
                    #make data accassable for model
                    inputs, labels = modelinputs(data)  
                    #do predictions and calculate loss
                    labels_pred = model(inputs)
                    loss_minibatch = loss_fn(labels_pred, labels)
                    #make one step to optimize weights
                    optimizer.zero_grad()
                    loss_minibatch.backward()
                    optimizer.step()
                  #fill loss lists with data
                  netdata["plytr"].append(np.sqrt(loss_fn(model(tr_data_torch), tr_labels_torch).detach().numpy()))
                  netdata["plyva"].append(np.sqrt(loss_fn(model(val_data_torch), val_labels_torch).detach().numpy()))
                #make sample predictions (val), measure time (sample size)
                for i in range(ANALYSIS_SAMPLE_SIZE):
                  t0 = time.time()
                  preds = model(val_data_torch)
                  t1 = time.time()
                  predtime = predtime + t1-t0
                netdata["predt"] = predtime / ANALYSIS_SAMPLE_SIZE
                netdata["lossval"] = min(loss_plot_y_val)
                print (predtime)
                print (loss)
                #save hyperloss
                netdata["hloss"] = hyperloss(netdata["predt"],netdata["lossval"],INT_LOSS)
                #check if net in top 10 and save
                if NetIsTopPerformer(netdata):
                    UpdateToplist(netdata)

##################### WRITE TOPLIST ##############################
WriteToplist()