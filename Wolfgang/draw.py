#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import keras.models

print ( "Now loading data" )
import pickle
f=open("data.pcl","rb")
tr_data=pickle.load ( f )
tr_labels=pickle.load ( f )
val_data=pickle.load ( f )
val_labels=pickle.load ( f )
f.close()

def Hash ( A ):
    return int(A[0]*10000.+A[1])
X={}
for d,l in zip (tr_data,tr_labels):
    X[Hash(d)]=l

masses = [] 
for i in range ( 600, 1000, 10 ):
    masses.append ( [ i, 300 ] )

mass = np.array( masses )
pX, pY, pYm = [], [], []

model = keras.models.load_model ( "model.h5")
preds=model.predict ( mass )

for m,p in zip(mass,preds):
    try:
        pY.append ( X[Hash(m)] )
        pX.append ( m[0] )
        pYm.append ( p )
    except:
        pass

plt.scatter ( pX, pY )
plt.scatter ( pX, pYm, c='r' )
plt.savefig ( "plot.png" )

#preds=model.predict ( mass )
#print ( "Now predict" )
#for m,p in zip ( mass,preds ):
#    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )
