from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from system.glovar import *
from system.errorlog import *
from random import shuffle
import system.pathfinder as path

def GetExpRes(exp):

	database = Database(path.database)
	return database.getExpResults(analysisIDs = [exp])[0]


def GetTopoMinMax(expres, topo):
	
	with open(expres.path + '/data/' + topo + '.txt') as f:
		content = f.readlines()

	content = [x.strip() for x in content]

	x = []
	found = False
	for line in content:
		if line[0:11] == 'upperLimits':
			found = True
			s = ''
			for i in range(14, len(line)):
				if line[i] != '[' and line[i] != '*':
					s += line[i]
				elif line[i] == '*':		
					break
			x.append(int(float(s)))

		elif line[0:19] == 'expectedUpperLimits':
			break

		elif found:
			s = ''
			for i in range(0, len(line)):
				if line[i] != '[' and line[i] != '*':
					s += line[i]
				elif line[i] == '*':
					break
			x.append(int(float(s)))

	return min(x), max(x), 0

def DataSimulate(exp, topo):

	dataset = []
	expres = GetExpRes(exp)
	
	motherLow, motherUp, daughterLow = GetTopoMinMax(expres, topo)

	print('data boundaries:', motherLow, motherUp, daughterLow)

	for mother in range(motherLow, motherUp, 10):
		for lsp in range (daughterLow, mother, 10):
			masses = [[ mother*GeV, lsp*GeV], [ mother*GeV, lsp*GeV]]
			ul     = expres.getUpperLimitFor(txname=topo, mass=masses)
			if type(ul) == type(None):
				continue
			dataset.append([mother, lsp, ul.asNumber(fb)])

	shuffle(dataset)
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
	training, validation, test = DataSplit(dataset, split)
	datasetTraining = data(training, device)

	GetDataObj = {}

	GetDataObj['training_set']	 	= torch.tensor(training).narrow(1, 0, 2).to(device)
	GetDataObj['training_labels']	= torch.tensor(training).narrow(1, 2, 1).to(device)
	GetDataObj['test_set']	  	 	= torch.tensor(test).narrow(1, 0, 2).to(device)
	GetDataObj['test_labels']	  	= torch.tensor(test).narrow(1, 2, 1).to(device)
	GetDataObj['validation_set']  	= torch.tensor(validation).narrow(1, 0, 2).to(device)
	GetDataObj['validation_labels'] = torch.tensor(validation).narrow(1, 2, 1).to(device)
	GetDataObj['dataset']			= datasetTraining

	return GetDataObj


if __name__ == "__main__":
	analysisId, txName, split, device = 'CMS-PAS-SUS-12-026', 'T1tttt', [0.8, 0.1, 0.1], 'cpu'
	DataGeneratePackage(analysisId, txName, split, device)
