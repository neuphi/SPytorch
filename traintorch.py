#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glovar
import numpy as np
from misc import *
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(DIM_IN, LAYERS_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(LAYERS_HIDDEN, DIM_OUT),
)
loss_fn = torch.nn.MSELoss(size_average=False)

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
for t in range(EPOCH_NUM):

    labels_pred = model(tr_data_torch)
    loss = loss_fn(labels_pred, tr_labels_torch)
    #print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("  done")

print("\nloss:", loss.item())

print("\nprinting tensors ..")
torch.set_printoptions(threshold=50)
for item in [tr_data_torch,tr_labels_torch,val_data_torch,val_labels_torch,labels_pred]:
    print("\nlen=", len(item), "\n", item)


    



#print ( "Done fitting" )
#mass = np.array( [ [ 600, 200 ], [ 700, 200 ], [ 800, 200 ], [ 900, 200 ], [ 1000, 200 ] ] )
#preds=model.predict ( mass )
#print ( "Now predict" ) 
#for m,p in zip ( mass,preds ):
#    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )

#model.save("data/model.h5")
#model.save_weights("data/weights.h5")
