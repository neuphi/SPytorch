from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb
from torch.autograd import Variable

from system.glovar import *
from system.errorlog import *

import system.pathfinder as path

def GetExpRes(exp):
	database = Database(path.database)
	return database.getExpResults(analysisIDs = [exp])[0]


def DataSimulate(exp, topo):

    dataset = []
    expres = GetExpRes(exp)
    for mother in range(MOTHER_LOW, MOTHER_UP, MOTHER_STEP):
        for lsp in range (LSP_LOW, mother, LSP_STEP): 
            masses = [[ mother*GeV, lsp*GeV], [ mother*GeV, lsp*GeV]]
            ul     = expres.getUpperLimitFor(txname=topo, mass=masses)
            if type(ul) == type(None):
                continue
            dataset.append([mother, lsp, ul.asNumber(fb)])
    return dataset


def DataSplit(dataset, split):

	if sum(split) != 1. or len(split) != 3:
		ErrorRaise['split']
		return 0, 0, 0

	cut = [int(len(dataset)*sum(split[:i])) for i in range(4)]
	return [dataset[cut[i]:cut[i+1]] for i in range(3)]
	

def DataGeneratePackage(exp, topo, split, device):

	dataset = DataSimulate(exp, topo)

	list_training, list_validation, list_test = DataSplit(dataset, split)
	
	tensor_training 	= torch.tensor(list_training)
	tensor_test			= torch.tensor(list_test)
	tensor_validation	= torch.tensor(list_validation)
	
	var_training_set	  	 = Variable(tensor_training.narrow(1, 0, 2)).to(device)#Variable(tensor_training[:][:2])
	var_training_labels	  	 = Variable(tensor_training.narrow(1, 2, 1)).to(device)#Variable(tensor_training[:][2])
	var_test_set 	 	 	 = Variable(tensor_test.narrow(1, 0, 2)).to(device)#Variable(tensor_test[:][:2])
	var_test_labels 	  	 = Variable(tensor_test.narrow(1, 2, 1)).to(device)#Variable(tensor_test[:][2])
	var_validation_set	  	 = Variable(tensor_validation.narrow(1, 0, 2)).to(device)#Variable(tensor_validation[:][:2])
	var_validation_labels 	 = Variable(tensor_validation.narrow(1, 2, 1)).to(device)#Variable(tensor_validation[:][2])

	#if cuda:
	#	var_training_set.to(device)
	#	var_training_labels.to(device)
	#	var_test_set.to(device)
	#	var_test_labels.to(device)
	#	var_validation_set.to(device)
	#	var_validation_labels.to(device)

	GetDataObj = {}

	GetDataObj['list_training']   		= list_training
	GetDataObj['list_validation'] 		= list_validation
	GetDataObj['list_test'] 	  		= list_test

	GetDataObj['tensor_training']	  	= tensor_training
	GetDataObj['tensor_validation']    	= tensor_validation
	GetDataObj['tensor_test']	  	   	= tensor_test

	GetDataObj['var_training_set']	 	= var_training_set
	GetDataObj['var_validation_set']  	= var_validation_set
	GetDataObj['var_test_set']	  	 	= var_test_set

	GetDataObj['var_training_labels']	= var_training_labels
	GetDataObj['var_validation_labels'] = var_validation_labels
	GetDataObj['var_test_labels']	  	= var_test_labels

	return GetDataObj


if __name__ == "__main__":
	DataGeneratePackage(ANALYSIS_ID, TXNAME, [0.8, 0.1, 0.1], 'cpu')

