#!/usr/bin/python3

########## IMPORT LIBRARIES ###########################

import numpy as np
import random
import time
import torch
from misc import *
from glovar import *

import torch.nn as nn
import torch.nn.functional as F

########## LOAD DATABASE AND MODEL #########################

database=Database( PATH_DATABASE )
expres = database.getExpResults( analysisIDs=[ ANALYSIS_ID ] )[0]
model = Net()
model.load_state_dict(torch.load(PATH_DATA + "torchmodel.h5"))

print ( "Done loading database" )

####### SIMULATE DATA ######################################

SIM_DATA_SIZE = 10000

masses = torch.zeros(SIM_DATA_SIZE, 2)
massesG = []

for i in range(SIM_DATA_SIZE):
    mother = random.uniform ( 600, 1099 )
    lsp = random.uniform ( 1, mother )
    masses[i,0]    = mother
    masses[i,1]    = lsp
    massesG.append ( [ [ mother*GeV, lsp*GeV ],  [ mother*GeV, lsp*GeV ] ] )

###### PREDICTIONS AND MEASUREMENT ########################

print ( "Full ... " )
t0=time.time()
ulF,ulB,ulC=[],[],[]
for m in massesG:
    ul = expres.getUpperLimitFor ( txname= TXNAME, mass=m )
    ulF.append ( ul )
t1=time.time()
print ( "Pytorch ... " )
ulP=model ( masses.type(torch.FloatTensor) )
t2=time.time()
print ( "Pytorch batch ... " )
for m in masses:
    pred=model ( m.type(torch.FloatTensor) )
    ulB.append ( pred )
t3=time.time()
print ( "Pytorch chunk ... " )
for i in range(100):
    ul=model ( masses[i::100].type(torch.FloatTensor) )
    for x in ul:
        ulC.append ( x )
t4=time.time()
#print ("Limits Full ", ulF[:3] )
#print ("Limits Pytorch", ulP[:3] )
#print ("Limits Batch", ulB[:3] )
#print ("Limits Chunk", ulC[:3] )
print ("Time Full ", t1-t0 )
print ("Time Pytorch", t2-t1 )
print ("Time Batch", t3-t2 )
print ("Time Chunk", t4-t3 )

f = open("analysis/timepytorch.txt","w") 

f.write("SIM_DATA_SIZE " + str(SIM_DATA_SIZE) + "\n \n")
f.write("Time Full    " + str(t1-t0) + "\n") 
f.write("... per Sample: " + str((t1-t0) / SIM_DATA_SIZE) + "\n")
f.write("Time Pytorch " + str(t2-t1) + " (Tensor with Length sample size)\n" )
f.write("... per Sample: " + str((t2-t1) / SIM_DATA_SIZE) + "\n")
f.write("Time Batch   " + str(t3-t2) + " (sample size List with every single element)\n" )
f.write("... per Sample: " + str((t3-t2) / SIM_DATA_SIZE) + "\n")
f.write("Time Chunk   " + str(t4-t3) + " (100 element list with (sample size/100) element tensors)\n" )
f.write("... per Sample: " + str((t3-t2) / SIM_DATA_SIZE) + "\n")
f.close() 

