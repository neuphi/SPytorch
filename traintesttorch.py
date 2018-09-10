####################### IMPORT STUFF ############################

from misc import *
#import torch.nn.functional as F
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

##################### SPLIT DATA #################################

#if SPLIT_CHOOSE = 1, split Data into SPLIT
#actually, we already have train and val data simulated and as we want to overfit, we don't really need any test data

##################### TRAINING ###################################

#add outer loops: lossfunc, activation, minibatch, slope
#lossf = ['mse','l1','nll']
#activf = ['linear','relu','tanh','sigmoid','linz']
#minibatchsize = [1,4,16,64,256]
#slope = [lin, ramp, trap]

for layers in range(HID_LAY_MIN,HID_LAY_MAX,5)
  for noodes in range(NOD_MIN, NOD_MAX, 4)
    #initialize net
    class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        for hidlay in range(layers + 1)
          self.hidden + hidlay =
          self.

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
              #define trainloader
              #loop over trainloader
                #train net with train data
              #save hyperloss (with val data) to dictionary
              #if net in top 100
                #save h5, text file (hyperl, loss(both with test data), predt, archit)
                #save loss.png

#write out network with smallest hyperloss value
