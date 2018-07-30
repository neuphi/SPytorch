#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

N, D_in, H, D_out, N_val = 5050, 2, 8, 1, 59

#import torch

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

import pickle
print("\nloading data ..")
f=open("data/data.pcl","rb")
tr_data=pickle.load ( f )
tr_labels=pickle.load ( f )
val_data=pickle.load ( f )
val_labels=pickle.load ( f )
f.close()
print("  done")

tr_data_torch    = torch.zeros(N, D_in)
tr_labels_torch  = torch.zeros(N, D_out)
val_data_torch   = torch.zeros(N_val, D_in)
val_labels_torch = torch.zeros(N_val, D_out)

for i in range(N):
	tr_data_torch[i]   = torch.from_numpy(tr_data[i])
	tr_labels_torch[i] = tr_labels[i]

for i in range(N_val):
	val_data_torch[i]   = torch.from_numpy(val_data[i])
	val_labels_torch[i] = val_labels[i]

print("\nfitting ..")
for t in range(1000):

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
    print("\nlen=",len(item))
    print(item)


    



#print ( "Done fitting" )
#mass = np.array( [ [ 600, 200 ], [ 700, 200 ], [ 800, 200 ], [ 900, 200 ], [ 1000, 200 ] ] )
#preds=model.predict ( mass )
#print ( "Now predict" ) 
#for m,p in zip ( mass,preds ):
#    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )

#model.save("data/model.h5")
#model.save_weights("data/weights.h5")
