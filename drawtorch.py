#!/usr/bin/python3

########## IMPORT LIBRARIES ###########################

import matplotlib.pyplot as plt
import numpy as np
import torch
from misc import *
from glovar import *

import torch.nn as nn
import torch.nn.functional as F

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
model.load_state_dict(torch.load("data/torchmodel.h5"))
preds=model ( mass.type(torch.FloatTensor) )

#### Create a tupel list with mass and predictions #####

i=0
for m,p in zip(mass,preds):
    try:
        pY.append ( X[Hash(m)] )
        pX.append ( m[0].numpy() )
        pYm.append ( p.item() )
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
plt.savefig ( "analysis/plots/plot.png" )

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
    Hmass = torch.Tensor( Hmasses )
    Hpreds=model( Hmass )
    j=0    
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
plt.ylabel('mother 1000-600')
plt.imshow(Heat_calc)       
plt.savefig("analysis/plots/heatcalc.png")
plt.title(r'Predicted Upper Limits', fontsize=20)
plt.imshow(Heat_pred)
plt.savefig("analysis/plots/heatpred.png")
plt.title(r'Quadrativc Difference Prediction/Calculation', fontsize=20)
plt.imshow(Heat_diff)
plt.savefig("analysis/plots/heatdiff.png")

###the curves next to each other for one lsp(slice)
plt.figure(1)         
plt.title(r'Upper Limit vs LSP', fontsize=20)
plt.text(170, 18, 'analysis ID: ' + str(ANALYSIS_ID) + '\ntopology: ' + str(TXNAME) + '\nmass mother: 600 GeV', style='italic',
        bbox={'facecolor':'gray', 'alpha':0.2, 'pad':10})
plt.xlabel('Mass LSP [GeV]')
plt.ylabel('Upper Limits (UL) [fb]')
print(Heat_calc[20])
plt.plot([i*10 for i in range(40)], Heat_calc[20], label = 'calculated UL')
plt.plot([i*10 for i in range(40)], Heat_pred[20], c = 'r', label = 'predicted UL')
plt.plot([i*10 for i in range(40)], Heat_diff[20], c = 'g', label = 'absolute error')
plt.legend()
plt.savefig("analysis/plots/slice.png")

