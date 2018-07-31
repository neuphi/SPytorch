#!/usr/bin/python3

from glovar import *
import numpy as np
import random
from misc import *
from smodels.tools.physicsUnits import GeV, fb

expres = getExpRes(EXP)

tr_data, tr_labels   = [], []
val_data, val_labels = [], []

for mother in np.arange(600., 1100., 5.):
    for lsp in np.arange (0., mother, 10.):

        masses = [[ mother*GeV, lsp*GeV], [ mother*GeV, lsp*GeV]]
        ul     = expres.getUpperLimitFor(txname=TX, mass=masses)
        if type(ul) == type(None):
            continue
        tr_data.append(np.array([mother, lsp]))
        tr_labels.append ( ul.asNumber(fb) )

for i in range(100):

    mother = random.uniform(600. , 1100.)
    lsp    = random.uniform(0., mother-10.)
    masses = [[ mother*GeV, lsp*GeV], [ mother*GeV, lsp*GeV]]
    ul     = expres.getUpperLimitFor(txname=TX, mass=masses)
    if type(ul) == type(None):
        continue
    val_data.append(np.array( [ mother, lsp ]))
    val_labels.append(ul.asNumber(fb))

saveData([tr_data,tr_labels,val_data,val_labels])

