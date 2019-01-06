from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
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


class data(Dataset):
    def __init__(self, data, device):
        self.x = torch.tensor(data).narrow(1, 0, 2).to(device)
        self.y = torch.tensor(data).narrow(1, 2, 1).to(device)

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
         return (self.x[idx], self.y[idx])


def DataGeneratePackage(exp, topo, split, device):

	dataset = DataSimulate(exp, topo)

	list_training, list_validation, list_test = DataSplit(dataset, split)

	d = data(list_training, device)
	
	tensor_training 	= torch.tensor(list_training, device=device)
	tensor_test			= torch.tensor(list_test, device=device)
	tensor_validation	= torch.tensor(list_validation, device=device)

	data_trai_set		= tensor_training.narrow(1, 0, 2).to(device)
	data_trai_lab		= tensor_training.narrow(1, 2, 1).to(device)

	data_test_set		= tensor_test.narrow(1, 0, 2).to(device)
	data_test_lab		= tensor_test.narrow(1, 2, 1).to(device)

	data_vali_set		= tensor_validation.narrow(1, 0, 2).to(device)
	data_vali_lab		= tensor_validation.narrow(1, 2, 1).to(device)
	
	var_training_set	  	 = Variable(tensor_training.narrow(1, 0, 2)).to(device)
	var_training_labels	  	 = Variable(tensor_training.narrow(1, 2, 1)).to(device)
	var_test_set 	 	 	 = Variable(tensor_test.narrow(1, 0, 2)).to(device)
	var_test_labels 	  	 = Variable(tensor_test.narrow(1, 2, 1)).to(device)
	var_validation_set	  	 = Variable(tensor_validation.narrow(1, 0, 2)).to(device)
	var_validation_labels 	 = Variable(tensor_validation.narrow(1, 2, 1)).to(device)

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

	GetDataObj['data_trai_set']			= data_trai_set
	GetDataObj['data_trai_lab']			= data_trai_lab

	GetDataObj['data_test_set']			= data_test_set
	GetDataObj['data_test_lab']			= data_test_lab

	GetDataObj['data_vali_set']			= data_vali_set
	GetDataObj['data_vali_lab']			= data_vali_lab

	GetDataObj['d']						= d

	return GetDataObj


if __name__ == "__main__":
	DataGeneratePackage(ANALYSIS_ID, TXNAME, [0.8, 0.1, 0.1], 'cpu')

