from glovar import *
import pickle
from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb

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

