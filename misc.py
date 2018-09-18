from glovar import *
import pickle
from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb
import numpy as np

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #super().__init__()
        self.hidden1 = nn.Linear(DIM_IN, DIM_HIDDEN_1)
        self.hidden2 = nn.Linear(DIM_HIDDEN_1, DIM_HIDDEN_2)
        self.hidden3 = nn.ReLU()
        self.hidden4 = nn.Linear(DIM_HIDDEN_2, DIM_HIDDEN_3)
        self.hidden5 = nn.ReLU()
        self.out = nn.Linear(DIM_HIDDEN_3, DIM_OUT)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.out(x)
        return x

def getExpRes(exp):
    print("load result ..")
    database=Database(PATH_DATABASE)
    print("  done")
    return database.getExpResults(analysisIDs = [exp])[0]

def saveData(data):
    print("\nsaving data ..")
    file = open(PATH_DATA + "data.pcl","wb")
    pickle.dump(len(data), file)
    for item in data:
        pickle.dump(item, file)
    file.close
    print("  done")
    
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

def hyperloss(time, loss, intloss):
  a = 1000     #some finetuning is possible with the a,b parameters
  b = 5
  c = 1
  if b*(loss-intloss) > 10:
    return a*time + c*np.exp(10)
  else:
    return a*time + c*np.exp(b*(loss-intloss))

def initloss (loss_fn_i):
    if loss_fn_i == "MSE":
        loss_fn = nn.MSELoss(size_average=True, reduce=True)
    return loss_fn

def initopt (optimizer_i, model, learning_rate):
    if optimizer_i == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def ModDataTorch(tr_data,tr_labels,val_data,val_labels):
    #initialize tensors
    tr_data_torch    = torch.zeros(SAMPLE_NMBR, DIM_IN)
    tr_labels_torch  = torch.zeros(SAMPLE_NMBR, DIM_OUT)
    val_data_torch   = torch.zeros(SAMPLE_NMBR_VAL, DIM_IN)
    val_labels_torch = torch.zeros(SAMPLE_NMBR_VAL, DIM_OUT)    
    #fill tensors
    for i in range(SAMPLE_NMBR):
      tr_data_torch[i]   = torch.from_numpy(tr_data[i])
      tr_labels_torch[i] = tr_labels[i]    
    for i in range(SAMPLE_NMBR_VAL):
      val_data_torch[i]   = torch.from_numpy(val_data[i])
      val_labels_torch[i] = val_labels[i]      
    return tr_data_torch, tr_labels_torch, val_data_torch, val_labels_torch

def modelinputs (data):
  inputs    = torch.zeros(torch.numel(data[:,0]), DIM_IN)
  labels    = torch.zeros(torch.numel(data[:,0]), DIM_OUT)
  for j in range(torch.numel(data[:,0])):
    inputs[j][0] = data[j][0]
    inputs[j][1] = data[j][1]
    labels[j]    = data[j][2]
  return inputs, labels
