#!/usr/bin/python3

import numpy as np
import random
import pickle

print ( "Now loading database" )
from smodels.experiment.databaseObj import Database
database=Database("database/")
from smodels.tools.physicsUnits import GeV, fb

expres = database.getExpResults( analysisIDs=["CMS-PAS-SUS-12-026"] )[0]

print ( "Done loading database" )

tr_data, tr_labels = [], []
val_data, val_labels=[], []

print ( "Now get upper limits" )
for mother in np.arange ( 600., 1100., 5. ):
    for lsp in np.arange ( 0., mother, 10. ):
        masses=[[ mother*GeV, lsp*GeV], [ mother*GeV, lsp*GeV] ]
        ul = expres.getUpperLimitFor ( txname="T1tttt", mass=masses )
        if type(ul) == type(None):
            continue
        tr_data.append ( np.array( [ mother, lsp ] ) )
        tr_labels.append ( ul.asNumber(fb) )
        if mother == 700. and lsp==200.:
            print ( "training data", mother, lsp, ul.asNumber(fb) )

for i in range(100):
    mother=random.uniform ( 600. , 1100. )
    lsp = random.uniform ( 0., mother-10. )
    masses=[[ mother*GeV, lsp*GeV], [ mother*GeV, lsp*GeV] ]
    ul = expres.getUpperLimitFor ( txname="T1tttt", mass=masses )
    if type(ul) == type(None):
        continue
    val_data.append ( np.array( [ mother, lsp ] ) )
    val_labels.append ( ul.asNumber(fb) )



f=open("data.pcl","wb")
pickle.dump ( tr_data, f )
pickle.dump ( tr_labels, f )
pickle.dump ( val_data, f )
pickle.dump ( val_labels, f )
f.close()

