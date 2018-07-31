#!/usr/bin/python3

from glovar import *
from misc import *
import numpy as np
import keras.models
import random
import time
from smodels.tools.physicsUnits import GeV, fb

expres = getExpRes(EXP)
model = keras.models.load_model(PATH_DATA + "model.h5")

masses, massesG = [], []

for i in range(10000):
    mother = random.uniform ( 600, 1099 )
    lsp = random.uniform ( 1, mother )
    masses.append ( np.array ( [ mother, lsp ] ) )
    massesG.append ( [ [ mother*GeV, lsp*GeV ],  [ mother*GeV, lsp*GeV ] ] )

print ( "Full ... " )
t0=time.time()
ulF,ulK,ulC=[],[],[]
for m in massesG:
    ul = expres.getUpperLimitFor ( txname=TX, mass=m )
    ulF.append ( ul )
t1=time.time()
print ( "Keras batch ... " )
ulB=model.predict ( np.array ( masses ) )
t2=time.time()
print ( "Keras ... " )
for m in masses:
    pred=model.predict ( np.array ( [ m ] ) )
    ulK.append ( pred )
t3=time.time()
print ( "Keras chunk ... " )
for i in range(100):
    ul=model.predict ( np.array ( masses[i::100] ) )
    for x in ul:
        ulC.append ( x )
t4=time.time()
print ("Limits Full ", ulF[:3] )
print ("Limits Keras", ulK[:3] )
print ("Limits Keras", ulB[:3] )
print ("Limits Keras", ulC[:3] )
print ("Time Full ", t1-t0 )
print ("Time Keras", t2-t1 )
print ("Time Batch", t3-t2 )
print ("Time Chunk", t4-t3 )
