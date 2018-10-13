#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:59:21 2018

@author: felix
"""
####################### IMPORT STUFF ############################

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader as DataLoader
#from torch.utils.data import Dataset as Dataset
import time
from system.misc import *
from system.initnet import *
from system.printtorch import *

#################### INITIALIZE CLOCK ############################

t_simulatedata = 0
t_moddata = 0
t_prepnets = 0
t_training = 0 
t_timeanalysis = 0
t_checktoplist = 0
t_writetoplist = 0
t_total = time.time()

######################## GET DATA ################################

t_simulatedata = time.time()

dataset = SimulateData()
training_set, validation_set, test_set = SplitData(dataset)

saveData([training_set, validation_set, test_set])

LEN_TRAINING_SET    = len(training_set)
LEN_VALIDATION_SET  = len(validation_set)
LEN_TEST_SET        = len(test_set)

t_simulatedata = time.time() - t_simulatedata

################## TURN DATA IN TENSORS ##########################

t_moddata = time.time()

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


t_moddata = time.time() - t_moddata

##################### TRAINING ###################################

print("\n STARTING TRAINING PROCESS")

counter = 1
totalcount = len(LOSS_FUNCTIONS) * len(OPTIMIZERS) * len(MINIBATCH_SIZES) * \
len(LEARN_RATE) * len(ACTIVATION_FUNCTIONS) * len(SHAPES) * len(HID_LAY) * len(NOD)
starttime = time.time()

for loss_fn_i in LOSS_FUNCTIONS:
  for optimizer_i in OPTIMIZERS:
    for minibatch in MINIBATCH_SIZES:
      for learning_rate in LEARN_RATE:
        for activ in ACTIVATION_FUNCTIONS:
          for shape in SHAPES:
            for layers in HID_LAY:
              for nodes in NOD:
                #initialize net, lossfunction and optimizer       
                print("\nNET ", counter, " of ", totalcount)
                t_prepnets_dummy = time.time()
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
                if CUDA:
                    netdata['model'].cuda()
                loss_fn = initloss(loss_fn_i)
                optimizer = initopt(optimizer_i, netdata['model'], learning_rate)
                #define trainloader
                trainloader = DataLoader(training_set, batch_size=minibatch, shuffle=True, num_workers=4)
                t_prepnets = t_prepnets + (time.time() - t_prepnets_dummy)
                #loop over epochs
                t_training_dummy = time.time() 
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
                  #fill loss lists with data
                  netdata["model"].cpu()
                  netdata["model"].cpu()
                  if loss_fn_i == "MSE":
                      #netdata["plytr"].append(np.sqrt(loss_fn(netdata["model"](training_set[:,:2]), training_set[:,2]).detach().numpy()))
                      netdata["plytr"].append(np.sqrt(loss_fn(netdata["model"](training_set_var), tr_labels_var).detach().numpy()))
                      netdata["plyte"].append(np.sqrt(loss_fn(netdata["model"](test_set_var), test_labels_var).detach().numpy()))
                  else:
                      netdata["plytr"].append(loss_fn(netdata["model"](training_set_var), tr_labels_var).cpu().detach().numpy())
                      netdata["plyte"].append(loss_fn(netdata["model"](test_set_var), test_labels_var).cpu().detach().numpy())                      
                  if CUDA:
                      netdata["model"].cuda()
                      netdata["model"].cuda()
                  #save validation
                  netdata["model"].cpu()
                  vall_dummy = loss_fn(netdata["model"](validation_set_var), validation_labels_var).detach().numpy()
                  if CUDA:
                      netdata["model"].cuda()
                  if netdata["lossv"] > vall_dummy:
                      netdata["lossv"] = vall_dummy
                t_training = t_training + (time.time() - t_training_dummy)      
                #make sample predictions (val), measure time (sample size)
                t_timeanalysis_dummy = time.time()
                predtime = 0
                for i in range(ANALYSIS_SAMPLE_SIZE):
                  t0 = time.time()
                  preds = netdata["model"](validation_set_var)
                  t1 = time.time()
                  predtime = predtime + t1-t0
                netdata["predt"] = predtime / ANALYSIS_SAMPLE_SIZE
                #fix validation if squared
                if loss_fn_i == "MSE":
                    netdata["lossv"] = np.sqrt(netdata["lossv"])
                #print some infos
                print ("Prediction Time: ",netdata["predt"])
                print ("Validation Loss: ",netdata["lossv"])
                clock = time.time() - starttime     
                print("Runtime: ", round(clock))
                print("Estimated Left: ", round((clock/counter)*(totalcount-counter)) )
                t_timeanalysis = t_timeanalysis + (time.time() - t_timeanalysis_dummy)
                #save hyperloss
                t_checktoplist_dummy = time.time()
                netdata["hloss"] = hyperloss(HYPERLOSS_FUNCTION, netdata["predt"],netdata["lossv"],INT_LOSS)
                #check if net in top 10 and save
                if NetIsTopPerformer(netdata):
                    UpdateToplist(netdata)   
                counter += 1
                t_checktoplist = t_checktoplist + (time.time() - t_checktoplist_dummy)
                #write toplist
                t_writetoplist_dummy = time.time()
                WriteToplist()
                t_writetoplist = t_writetoplist + (time.time() - t_writetoplist_dummy)

##################### TIME ANALYSIS ###############################

t_total = time.time() - t_total

print ("Time Data Simulation: ", t_simulatedata)
print ("Time Data Modification: ", t_moddata)
print ("Time Prepare Nets: ", t_prepnets)
print ("Time Train Nets: ", t_training) 
print ("Time Prediction Time Analysis:", t_timeanalysis)
print ("Time Check Toplist", t_checktoplist)
print ("Time Write Toplist", t_writetoplist)
print ("Time Total", t_total)
