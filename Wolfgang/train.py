#!/usr/bin/python3

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import callbacks
import time
import IPython

#inputs = Input(shape=(1,))
#preds = Dense(1,activation='linear')(inputs)
#model = Model ( inputs=inputs, outputs=preds )
## define the network
model = Sequential()
model.add(Dense(4, activation="linear", input_shape=(2,)))
model.add(Dense(16, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="linear" ))
model.summary()

model.compile ( loss="mean_squared_error", optimizer="adam", metrics=["mse"] )
# model.compile ( loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"] )

print ( "Now loading data" )
import pickle
f=open("data.pcl","rb")
tr_data=pickle.load ( f )
tr_labels=pickle.load ( f )
val_data=pickle.load ( f )
val_labels=pickle.load ( f )
f.close()


def hash ( A ):
    return int(A[0]*10000.+A[1])

X={}
for d,l in zip (tr_data,tr_labels):
    X[hash(d)]=l

#cbk=callbacks.TensorBoard ( "logs/log_test_"+str(time.time()), histogram_freq=1, write_graph=True )
cbk=callbacks.TensorBoard ( "logs/log.me", histogram_freq=1, write_graph=True )
cbk.set_model ( model )
            
print ( "Now fitting ... " )
history=model.fit ( np.array(tr_data), np.array(tr_labels), \
                    validation_data = ( np.array(val_data), np.array(val_labels) ), 
                    batch_size=50, epochs=200, callbacks = [ cbk ] )
print ( "Done fitting" )
mass = np.array( [ [ 600, 200 ], [ 700, 200 ], [ 800, 200 ], [ 900, 200 ], [ 1000, 200 ] ] )
preds=model.predict ( mass )
print ( "Now predict" ) 
for m,p in zip ( mass,preds ):
    print ( "%s -> %s, %s" % ( m,p[0], X[hash(m)] ) )

model.save("model.h5")
model.save_weights("weights.h5")
# IPython.embed()
