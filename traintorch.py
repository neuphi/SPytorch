#!/usr/bin/python3
# -*- coding: utf-8 -*-

from glovar import *
from misc import *
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

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

model = Net()
print("\n",model)

#loss_fn = nn.L1Loss(size_average=True, reduce=True, reduction='none')
loss_fn = nn.MSELoss(size_average=True, reduce=True, reduction='elementwise_mean')
#loss_fn = nn.CrossEntropyLoss(weight=None, size_average=True, reduce=True, reduction='elementwise_mean')
#loss_fn = nn.NLLLoss(weight=None, size_average=True, reduce=True, reduction='elementwise_mean')
#loss_fn = nn.PoissonNLLLoss(log_input=False, full=False, size_average=None, eps=1e-08, reduce=None, reduction='elementwise_mean')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

tr_data,tr_labels,val_data,val_labels = loadData()

tr_data_torch    = torch.zeros(BATCH_SIZE,     DIM_IN)
tr_labels_torch  = torch.zeros(BATCH_SIZE,     DIM_OUT)
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
epoch    = 0

loss_plot_x = []
loss_plot_y = []

while err > ERROR_TRESHOLD and epoch < EPOCH_NUM:

    labels_pred = model(tr_data_torch)
    loss = loss_fn(labels_pred, tr_labels_torch)
    #print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_plot_x.append(epoch)
    loss_plot_y.append(loss)
    
    epoch   += 1
    loss     = loss.item()
    err      = abs(1. - loss / loss_old)
    loss_old = loss
    
print("  done")
if epoch == EPOCH_NUM:
    print("\nFailed to reach convergence.")
else:
    print("\nConvergence reached after", epoch, "iterations.")
print("  final loss:", loss)

torch.save(model.state_dict(),PATH_DATA + "torchmodel.h5")

print("\nprinting tensors ..")
torch.set_printoptions(threshold=50)
#for item in [tr_data_torch,tr_labels_torch,val_data_torch,val_labels_torch,labels_pred]:
for item in [val_labels_torch,labels_pred]:
    print("\nlen=", len(item), "\n", item)


plt.figure(0)
plt.title('loss function', fontsize=20)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(loss_plot_x,loss_plot_y)
plt.show()



#print ( "Done fitting" )
#mass = np.array( [ [ 600, 200 ], [ 700, 200 ], [ 800, 200 ], [ 900, 200 ], [ 1000, 200 ] ] )
#preds=model.predict ( mass )
#print ( "Now predict" ) 
#for m,p in zip ( mass,preds ):
#    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )


#model.save_weights("data/weights.h5")

