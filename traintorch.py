#!/usr/bin/python3

# -*- coding: utf-8 -*-

####################### IMPORT STUFF ############################

import torch
import numpy as np

###################### DEFINE CLASS ##############################

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


######################## LOAD DATA ################################

print ( "Now loading data" )
import pickle
f=open("data/data.pcl","rb")
tr_data=pickle.load ( f )
tr_labels=pickle.load ( f )
val_data=pickle.load ( f )
val_labels=pickle.load ( f )
f.close()

def hash ( A ):          #define a hash dictionary
    return int(A[0]*10000.+A[1])

X={}
for d,l in zip (tr_data,tr_labels):
    X[hash(d)]=l

###################### INITIALIZE NN ############################

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out, N_val = len(tr_data), 2, 8, 1, len(val_data)

model = TwoLayerNet(D_in, H, D_out)

#model = torch.nn.Sequential(
#    torch.nn.Linear(D_in, H),
#    torch.nn.ReLU(),
#    torch.nn.Linear(H, D_out),
#)
#loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

###################### MODIFY DATA FOR NN #########################

tr_data_torch      = torch.zeros(N, D_in)
tr_labels_torch    = torch.zeros(N, D_out)
val_data_torch     = torch.zeros(N_val, D_in)
val_labels_torch   = torch.zeros(N_val,  D_out)

for i in range(N):
	tr_data_torch[i]   = torch.from_numpy(tr_data[i])
	tr_labels_torch[i] = tr_labels[i]

for i in range(N_val):
	val_data_torch[i]   = torch.from_numpy(val_data[i])
	val_labels_torch[i] = val_labels[i]

###################### FIT NN #####################################

print ( "Now fitting ... " )
for t in range(500):
    labels_pred = model(tr_data_torch)

    loss = criterion(labels_pred, tr_labels_torch)
    #loss = loss_fn(labels_pred, tr_labels_torch)
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print ( "Done fitting" )

################# MAKE EXAMPLE PREDICTIONS ########################
mass = torch.tensor( [ [ 600, 200 ], [ 700, 200 ], [ 800, 200 ], [ 900, 200 ], [ 1000, 200 ] ] )
preds=model( mass.type(torch.FloatTensor) )
print ( "Now predict" ) 

for m,p in zip ( mass,preds ):
    print ( "%s -> %s, %s" % ( m,p, X[hash(m)]) )

################ SAVE MODEL ##################################
#The second saves and loads the entire model:
#torch.save(the_model, PATH)
#Then later:
#the_model = torch.load(PATH)

torch.save(model,"data/torchmodel.h5")
