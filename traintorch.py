#!/usr/bin/python3
# -*- coding: utf-8 -*-

from glovar import *
import numpy as np
from misc import *
import torch

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        #super(Net, self).__init__()
        super().__init__()
        self.hidden1 = nn.Linear(DIM_IN, DIM_HIDDEN_1)
        self.hidden2 = nn.Linear(DIM_HIDDEN_1, DIM_HIDDEN_2)
        self.hidden3 = nn.Linear(DIM_HIDDEN_2, DIM_HIDDEN_3)
        self.out     = nn.Linear(DIM_HIDDEN_3, DIM_OUT)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x

model = Net()
print("\n",model)

loss_fn = nn.L1Loss(size_average=True, reduce=True, reduction='none')
#loss_fn = nn.MSELoss(size_average=True, reduce=True, reduction='elementwise_mean')
#loss_fn = nn.CrossEntropyLoss(weight=None, size_average=True, reduce=True, reduction='elementwise_mean')
#loss_fn = nn.NLLLoss(weight=None, size_average=True, reduce=True, reduction='elementwise_mean')
#loss_fn = nn.PoissonNLLLoss(log_input=False, full=False, size_average=None, eps=1e-08, reduce=None, reduction='elementwise_mean')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

tr_data,tr_labels,val_data,val_labels = loadData()

tr_data_torch    = torch.zeros(BATCH_SIZE, DIM_IN)
tr_labels_torch  = torch.zeros(BATCH_SIZE, DIM_OUT)
val_data_torch   = torch.zeros(BATCH_SIZE_VAL, DIM_IN)
val_labels_torch = torch.zeros(BATCH_SIZE_VAL, DIM_OUT)

for i in range(BATCH_SIZE):
	tr_data_torch[i]   = torch.from_numpy(tr_data[i])
	tr_labels_torch[i] = tr_labels[i]

for i in range(BATCH_SIZE_VAL):
	val_data_torch[i]   = torch.from_numpy(val_data[i])
	val_labels_torch[i] = val_labels[i]

print("\nfitting ..")
err      = 1e5
loss_old = 1e5
epoch    = 1
#for t in range(EPOCH_NUM):
while err > ERROR_TRESHHOLD and epoch < EPOCH_NUM+1:

    labels_pred = model(tr_data_torch)
    loss = loss_fn(labels_pred, tr_labels_torch)
    #print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    epoch   += 1
    loss     = loss.item()
    err      = abs(loss - loss_old)
    loss_old = loss
print("  done")
if err == EPOCH_NUM:
    print("\nFailed to reach convergence.")
else:
    print("\nConvergence reached after", epoch, "iterations")

#model.state_dict(PATH_DATA + "model.h5", prefix = '', keep_vars = False)
#torch.save(model, PATH_DATA + "model.h5")
#print("\nloss:", loss.item())

print("\nprinting tensors ..")
torch.set_printoptions(threshold=50)
#for item in [tr_data_torch,tr_labels_torch,val_data_torch,val_labels_torch,labels_pred]:
for item in [val_labels_torch,labels_pred]:
    print("\nlen=", len(item), "\n", item)


    



#print ( "Done fitting" )
#mass = np.array( [ [ 600, 200 ], [ 700, 200 ], [ 800, 200 ], [ 900, 200 ], [ 1000, 200 ] ] )
#preds=model.predict ( mass )
#print ( "Now predict" ) 
#for m,p in zip ( mass,preds ):
#    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )


#model.save_weights("data/weights.h5")

