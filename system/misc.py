import pickle
from system.errorlog import *
from system.glovar import *
from system.clock import *
from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb
import numpy as np
from random import shuffle
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse



def saveData(data):
    file = open(PATH_DATA + "data.pcl","wb")
    pickle.dump(len(data), file)
    for item in data:
        pickle.dump(item, file)
    file.close
    
def loadData():
    print("\nloading data ..")
    file = open(PATH_DATA + "data.pcl","rb")
    lgth = pickle.load(file)
    data = []
    for i in range(lgth):
        data.append(pickle.load(file))
    file.close()
    print("  done")
    return data

def Hash ( A ):
    return int(A[0]*10000.+A[1])

def hyperloss(HYPERLOSS_FUNCTION, time, loss, intloss):
    if HYPERLOSS_FUNCTION == "lin":
        return hyperloss_lin(time, loss, intloss)
    elif HYPERLOSS_FUNCTION == "exp":
        return hyperloss_exp(time, loss, intloss)
    else:
        print("WARNING! Hyperloss function undefined! Using linear.")
        return hyperloss_lin(time, loss, intloss)

def hyperloss_exp(time, loss, intloss):
  a = 1000     #some finetuning is possible with the a,b parameters
  b = 5
  c = 1
  if b*(loss-intloss) > 10:
    return a*time + c*np.exp(10)
  else:
    return a*time + c*np.exp(b*(loss-intloss))

def hyperloss_lin(time, loss, intloss):
  a = 1000     #some finetuning is possible with the a,b parameters
  b = 100
  c = 0.01
  if loss > intloss:
    return a*time + b*loss
  else:
    return a*time + c*loss

def initloss (loss_fn_i):
    if loss_fn_i == "MSE":
        loss_fn = nn.MSELoss(reduction = 'elementwise_mean')
    if loss_fn_i == "L1":
        loss_fn = nn.L1Loss(reduction = 'elementwise_mean')
    return loss_fn

def initopt (optimizer_i, model, learning_rate):
    if optimizer_i == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_i == "SGD":
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
    return optimizer

def modelinputs (data):
  inputs    = torch.zeros(torch.numel(data[:,0]), DIM_IN, device=args.device)
  labels    = torch.zeros(torch.numel(data[:,0]), DIM_OUT, device=args.device)
  for j in range(torch.numel(data[:,0])):
    inputs[j][0] = data[j][0]
    inputs[j][1] = data[j][1]
    labels[j]    = data[j][2]
  inputs = Variable(inputs)
  labels = Variable(labels)
  return inputs, labels




def ModDataKeras(dataset):
    data = []
    labels = []
    for i in range(len(dataset)):
        data.append(dataset[i][0])
        labels.append(dataset[i][1])
    return data, labels
