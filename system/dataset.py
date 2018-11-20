from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb
from torch.autograd import Variable
import numpy as np
from random import shuffle
from system.glovar import *
#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#import argparse
from system.errorlog import *

import system.pathfinder as path



def DataConvert(dataset_list):
	lgth = len(dataset_list)
	dataset_torch = torch.zeros(lgth, 3, device=args.device)
	for i in range(lgth): 
		dataset_torch[i][:2] = torch.from_numpy(dataset_list[i][0][:2])
		dataset_torch[i][2]  = dataset_list[i][1]  
	return dataset_torch


def GetExpRes(exp):

    database = Database(path.database)
    return database.getExpResults(analysisIDs = [exp])[0]

def DataSimulate(exp):

    dataset = []
    expres = GetExpRes(exp)
    for mother in np.arange(MOTHER_LOW, MOTHER_UP, MOTHER_STEP):
        for lsp in np.arange (LSP_LOW, mother, LSP_STEP): 
            masses = [[ mother*GeV, lsp*GeV], [ mother*GeV, lsp*GeV]]
            ul     = expres.getUpperLimitFor(txname=TX, mass=masses)
            if type(ul) == type(None):
                continue
            dataset.append([np.array([mother, lsp]), ul.asNumber(fb)])
    return dataset

def DataSplit(dataset, split1, split2, split3):

	if split1+split2+split3 != 1.:
		ErrorRaise['split']
		return 0, 0, 0

	shuffle(dataset)

	lgth = len(dataset)

	cut1 = int(lgth*split1)
	cut2 = int(lgth*(split1+split2))
	cut3 = lgth

	subset1 = dataset[0    : cut1]
	subset2 = dataset[cut1 : cut2]
	subset3 = dataset[cut2 : cut3]

	return subset1, subset2, subset3


def DataGeneratePackage(exp, split1, split2, split3, cuda):

	dataset = DataSimulate(exp)

	list_training, list_validation, list_test = DataSplit(dataset, split1, split2, split3)
	#saveData([list_training, list_validation, list_test])
	
	tensor_training_set   	 = DataConvert(list_training)
	tensor_validation_set 	 = DataConvert(list_validation)
	tensor_test_set		  	 = DataConvert(list_test)

	l1, l2, l3 = len(tensor_training_set[:,0]), len(tensor_validation_set[:,0]), len(tensor_test_set[:,0])

	tensor_training_labels = torch.zeros(l1, 1, device=args.device)
	for i in range(l1): tensor_training_labels[i] = tensor_training_set[i,2]

	tensor_validation_labels = torch.zeros(l2, 1, device=args.device)
	for i in range(l2): tensor_validation_labels[i] = tensor_validation_set[i,2]

	tensor_test_labels = torch.zeros(l3, 1, device=args.device)
	for i in range(l3): tensor_test_labels[i] = tensor_test_set[i,2]

	var_training_set	  	 = Variable(tensor_training_set[:,:2])
	var_training_labels	  	 = Variable(tensor_training_labels)
	var_test_set 	 	 	 = Variable(tensor_test_set[:,:2])
	var_test_labels 	  	 = Variable(tensor_test_labels)
	var_validation_set	  	 = Variable(tensor_validation_set[:,:2])
	var_validation_labels 	 = Variable(tensor_validation_labels)

	if cuda:
		var_training_set.to(device=args.device)
		var_training_labels.to(device=args.device)
		var_test_set.to(device=args.device)
		var_test_labels.to(device=args.device)
		var_validation_set.to(device=args.device)
		var_validation_labels.to(device=args.device)

	GetDataObj = {}

	GetDataObj['list_training']   		   = list_training
	GetDataObj['list_validation'] 		   = list_validation
	GetDataObj['list_test'] 	  		   = list_test

	GetDataObj['tensor_training_set']	   = tensor_training_set
	GetDataObj['tensor_validation_set']    = tensor_validation_set
	GetDataObj['tensor_test_set']	  	   = tensor_test_set

	GetDataObj['tensor_training_labels']   = tensor_training_labels
	GetDataObj['tensor_validation_labels'] = tensor_validation_labels
	GetDataObj['tensor_test_labels']	   = tensor_test_labels

	GetDataObj['var_training_set']	 	   = var_training_set
	GetDataObj['var_validation_set']  	   = var_validation_set
	GetDataObj['var_test_set']	  	 	   = var_test_set

	GetDataObj['var_training_labels']	   = var_training_labels
	GetDataObj['var_validation_labels']    = var_validation_labels
	GetDataObj['var_test_labels']	  	   = var_test_labels

	return GetDataObj


