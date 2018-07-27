#!/usr/bin/python3

# -*- coding: utf-8 -*-

import torch
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = 5050, 2, 8, 2

# Create random Tensors to hold inputs and outputs

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
	torch.nn.ReLU(),
	torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



print ( "Now loading data" )
import pickle
f=open("data/data.pcl","rb")
tr_data=pickle.load ( f )
tr_labels=pickle.load ( f )
val_data=pickle.load ( f )
val_labels=pickle.load ( f )
f.close()

len_tr_x = len(tr_data)
len_tr_y = len(tr_data[0])
len_val_x = len(val_data)
len_val_y = len(val_data[0])

tr_data_torch      = torch.zeros(len_tr_x, len_tr_y)
tr_labels_torch    = torch.zeros(len_tr_x, len_tr_y)
val_data_torch     = torch.zeros(len_val_x, len_val_y)
val_labels_torch   = torch.zeros(len_val_x, len_val_y)

for i in range(len_tr_x):
	tr_data_torch[i]   = torch.from_numpy(tr_data[i])
	tr_labels_torch[i] = tr_labels[i]

for i in range(len_val_x):
	val_data_torch[i]   = torch.from_numpy(val_data[i])
	val_labels_torch[i] = val_labels[i]


#def hash ( A ):
#    return int(A[0]*10000.+A[1])

#X={}
#for d,l in zip (tr_data,tr_labels):
#    X[hash(d)]=l


print ( "Now fitting ... " )
for t in range(1000):

    labels_pred = model(tr_data_torch)

    loss = loss_fn(labels_pred, tr_labels_torch)
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#print ( "Done fitting" )
#mass = np.array( [ [ 600, 200 ], [ 700, 200 ], [ 800, 200 ], [ 900, 200 ], [ 1000, 200 ] ] )
#preds=model.predict ( mass )
#print ( "Now predict" ) 
#for m,p in zip ( mass,preds ):
#    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )

#model.save("data/model.h5")
#model.save_weights("data/weights.h5")
