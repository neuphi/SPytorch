#!/usr/bin/python3

########## IMPORT LIBRARIES ###########################

import matplotlib.pyplot as plt
import numpy as np
import torch
from misc import *
from glovar import *

import torch.nn as nn
import torch.nn.functional as F

###################### DEFINE CLASS ##############################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(DIM_IN, DIM_HIDDEN_1)
        #self.layer2 = nn.Linear(DIM_HIDDEN_1, DIM_HIDDEN_2)
        #self.layer3 = nn.Linear(DIM_HIDDEN_2, DIM_HIDDEN_3)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.ReLU()
        self.out    = nn.Linear(DIM_HIDDEN_3, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.out(x)
        return x

########## LOAD DATABASE ##############################

tr_data,tr_labels,val_data,val_labels = loadData()

X={} #create dictionary
for d,l in zip (tr_data,tr_labels):
    X[Hash(d)]=l

##### CREATE MASS ARRAYS WITH LSP=300, #################
##### DIFFERENT MOTHERS ################################

masses = [] 
for i in range ( 600, 1000, 10 ):
    masses.append ( [ i, 300 ] )

mass = torch.tensor( masses )

##### Create some empty Arrays and stuff ##############

pX, pY, pYm = [], [], []

##### Load the model and make predictions ##############

model = Net()
model.load_state_dict(torch.load(PATH_DATA + "torchmodel.h5"))
preds=model ( mass.type(torch.FloatTensor) )

#### Create a tupel list with mass and predictions #####

i=0
for m,p in zip(mass,preds):
    try:
        pY.append(X[Hash(m)])
        pX.append(m[0].numpy())
        pYm.append(p.item())
    except:
        pass
    i=i+1

#### Test: Print lengths of the pY, pX, pYm #############
#print (len(pY))
#print (len(pX))
#print (len(pYm))
#print (i)

#### Do the actual plot #################################

plt.figure(0)
plt.title(r'Fixed LSP, upperlimit vs mother', fontsize=20)
plt.xlabel('mother')
plt.ylabel('upper limit')
plt.scatter ( pX, pY )
plt.scatter ( pX, pYm, c='r' )
plt.savefig ( PATH_PLOTS + "plot.png" )

#preds=model.predict ( mass )
#print ( "Now predict" )
#for m,p in zip ( mass,preds ):
#    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )

#######CREATE A HEATMAP #################################

### initialize the emphy arrays
Heat_calc = [[0 for i in range(40)] for j in range(40)] 
Heat_pred = [[0 for i in range(40)] for j in range(40)]

### fill the arrays with data
i = 0
for mother in range ( 600, 1000, 10 ):
    Hmasses = []
    Hpreds = []
    for lsp in range (0, 400, 10 ):
        Hmasses.append ( [ mother, lsp ] )
    Hmass  = torch.Tensor( Hmasses )
    Hpreds = model( Hmass )
    j = 0    
    for m in Hmass:
        try:
            Heat_calc[i][j] = X[Hash(m)]    #X[Hash(m)] gives the upperlimit for the masses
            Heat_pred[i][j] = Hpreds[j][0].item()            #p is the predictions
        except:
            Heat_calc[i][j] = 0    #X[Hash(m)] gives the upperlimit for the masses
            Heat_pred[i][j] = 0            #p is the predictions
        j+=1
    i+=1

### calculate the quadratic error
Heat_diff = (np.array(Heat_pred) - np.array(Heat_calc))#**2 

###the three heatmaps
plt.title(r'Calculated Upper Limits', fontsize=20)
plt.xlabel('lsp 0-400')
plt.ylabel('mother 600-1000')
plt.imshow(Heat_calc)       
plt.savefig(PATH_PLOTS + "heatcalc.png")
plt.title(r'Predicted Upper Limits', fontsize=20)
plt.imshow(Heat_pred)
plt.savefig(PATH_PLOTS + "heatpred.png")
plt.title(r'Quadratic Difference Prediction/Calculation', fontsize=20)
plt.imshow(Heat_diff)
plt.savefig(PATH_PLOTS + "heatdiff.png")

###the curves next to each other for one lsp(slice)
plt.figure(1)         
plt.title(r'Slice of the Heatmap', fontsize=20)
plt.xlabel('lsp')
plt.ylabel('upper limit')
plt.plot([i*10 for i in range(40)], Heat_calc[20])
plt.plot([i*10 for i in range(40)], Heat_pred[20], c = 'r')
plt.plot([i*10 for i in range(40)], Heat_diff[20], c = 'g')
plt.savefig(PATH_PLOTS + "slice.png")