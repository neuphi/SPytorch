#!/usr/bin/python3

from glovar import *
from misc import *
import matplotlib.pyplot as plt
import numpy as np
import keras.models

tr_data,tr_labels,val_data,val_labels = loadData()

X={}
for d,l in zip (tr_data,tr_labels):
    X[Hash(d)]=l

masses = [] 
for i in range ( 600, 1000, 10 ):
    masses.append ( [ i, 300 ] )

mass = np.array( masses )
pX, pY, pYm = [], [], []

model = keras.models.load_model ( PATH_DATA + "model.h5")
preds=model.predict ( mass )

for m,p in zip(mass,preds):
    try:
        pY.append ( X[Hash(m)] )
        pX.append ( m[0] )
        pYm.append ( p )
    except:
        pass

plt.figure(0)
plt.scatter ( pX, pY )
plt.scatter ( pX, pYm, c='r' )
plt.savefig ( PATH_PLOTS + "plot.png" )

#preds=model.predict ( mass )
#print ( "Now predict" )
#for m,p in zip ( mass,preds ):
#    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )

# HEATMAP

Heat_calc = [[0 for i in range(40)] for j in range(40)]
Heat_pred = [[0 for i in range(40)] for j in range(40)]

i = 0
for mother in range ( 600, 1000, 10 ):
    Hmasses = []
    Hpreds = []
    for lsp in range (0, 400, 10 ):
        Hmasses.append ( [ mother, lsp ] )
    Hmass = np.array( Hmasses )
    Hpreds=model.predict ( Hmass )
    j=0    
    for m in Hmass:
        try:
            Heat_calc[i][j] = X[Hash(m)]    #X[Hash(m)] gives the upperlimit for the masses
            Heat_pred[i][j] = Hpreds[j][0]            #p is the predictions
        except:
            Heat_calc[i][j] = 0    #X[Hash(m)] gives the upperlimit for the masses
            Heat_pred[i][j] = 0            #p is the predictions
        j+=1
    i+=1

Heat_diff = (np.array(Heat_pred) - np.array(Heat_calc))#**2 #quadratischer Fehler

plt.imshow(Heat_calc)
plt.savefig(PATH_PLOTS + "heatcalc.png")
plt.imshow(Heat_pred)
plt.savefig(PATH_PLOTS + "heatpred.png")
plt.imshow(Heat_diff)
plt.savefig(PATH_PLOTS + "heatdiff.png")

plt.figure(1)
plt.plot([i*10 for i in range(40)], Heat_calc[20])
plt.plot([i*10 for i in range(40)], Heat_pred[20], c = 'r')
plt.plot([i*10 for i in range(40)], Heat_diff[20], c = 'g')
plt.savefig(PATH_PLOTS + "slice.png")