import pickle
from system.glovar import *
from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb
import numpy as np
from random import shuffle
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse

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
    database=Database(PATH_DATABASE)
    return database.getExpResults(analysisIDs = [exp])[0]

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
        loss_fn = nn.MSELoss(size_average=True, reduce=True)
    return loss_fn

def initopt (optimizer_i, model, learning_rate):
    if optimizer_i == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def modelinputs (data):
  inputs    = torch.zeros(torch.numel(data[:,0]), DIM_IN, device=args.device)
  labels    = torch.zeros(torch.numel(data[:,0]), DIM_OUT, device=args.device)
  for j in range(torch.numel(data[:,0])):
    inputs[j][0] = data[j][0]
    inputs[j][1] = data[j][1]
    labels[j]    = data[j][2]
  if CUDA:
    inputs = Variable(inputs.cuda())
    labels = Variable(labels.cuda())
  else:
    inputs = Variable(inputs)
    labels = Variable(labels)
  return inputs, labels

def SimulateData():
    dataset = []
    expres = getExpRes(EXP)
    for mother in np.arange(MOTHER_LOW, MOTHER_UP, MOTHER_STEP):
        for lsp in np.arange (LSP_LOW, mother, LSP_STEP): 
            masses = [[ mother*GeV, lsp*GeV], [ mother*GeV, lsp*GeV]]
            ul     = expres.getUpperLimitFor(txname=TX, mass=masses)
            if type(ul) == type(None):
                continue
            dataset.append([np.array([mother, lsp]), ul.asNumber(fb)])
    return dataset

def SplitData(dataset):

	shuffle(dataset)

	cut1 = int(len(dataset)*SPLIT[0]*0.01)
	cut2 = int(len(dataset)*(SPLIT[0]+SPLIT[1])*0.01)
	cut3 = len(dataset)

	training_set 	= dataset[0    : cut1]
	validation_set 	= dataset[cut1 : cut2]
	test_set 		= dataset[cut2 : cut3]

	return training_set, validation_set, test_set


def ModDataTorch(dataset):
    dataset_torch = torch.zeros(len(dataset), DIM_IN+DIM_OUT, device=args.device)
    for i in range(len(dataset)):        
        dataset_torch[i][:2] = torch.from_numpy(dataset[i][0][:2])
        dataset_torch[i][2] = dataset[i][1]                
    return dataset_torch
