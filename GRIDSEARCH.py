#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################### IMPORT STUFF ############################

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader as DataLoader
#from torch.utils.data import Dataset as Dataset

from math import sqrt as sqrt

from system.misc import *
from system.initnet import *
from system.printtorch import *
from system.dataset_new import *
from system.parameters import *

import sys

import system.pathfinder as path


parser = argparse.ArgumentParser(description='PyTorch Gridsearch')
parser.add_argument('--disable-cuda', action = 'store_true', help = 'Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
	whichDevice = input("Which GPU to use?")	
	device = torch.device('cuda:' + whichDevice)
	use_cuda = True
else:
	device = torch.device('cpu')
	use_cuda = False


print("\nusing CUDA on {}".format(device) if use_cuda else "\nno CUDA, running solely on CPU")

#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True




######################## GENERATE DATA ################################

TimerZero('all')
TimerInit('total')
TimerInit('gendata')



print("\ngenerate data .. ", end = "", flush = True)

GetDataObj = DataGeneratePackage(ANALYSIS_ID, TXNAME, [0.8, 0.1, 0.1], device)

#sys.exit()

TimerSetSave('gendata')
print("done after {}".format(TimerGetStr('gendata')[0]))

##################### TRAINING ###################################

print("initiating grid search\n")

GridParameter = LoadParameters()

counter    = 1
totalcount = GetNetConfigurationNum(GridParameter)

totalworkload = ( totalcount * EPOCH_NUM / len(GridParameter['minibatch']) ) * (sum([len(GetDataObj['list_training'])/b for b in GridParameter['minibatch']]))
progress      = 0

for loss_fn_i in GridParameter['loss_func']:

	for optimizer_i in GridParameter['optimizer']:

		for minibatch in GridParameter['minibatch']:

			for learning_rate in GridParameter['lera_iter']:

				for activ in GridParameter['acti_func']:

					for shape in GridParameter['nodes_shape']:

						for layers in GridParameter['layer_iter']:

							for nodes in GridParameter['nodes_iter']:       

								TimerInit('prepnet')
								
								netdata = CreateNet(layers, nodes, activ, shape, loss_fn_i, optimizer_i, minibatch, learning_rate)  
								model = netdata['model']
								model.to(device)

								loss_fn = initloss(loss_fn_i)
								optimizer = initopt(optimizer_i, model, learning_rate)

								trainloader = DataLoader(GetDataObj['tensor_training'], batch_size=minibatch, shuffle=True, num_workers=0)
								#len_loader  = len(trainloader)
								TimerAddSave('prepnet')

								TimerInit('train')

								for epoch in range(EPOCH_NUM):
									for i, data in enumerate(trainloader):  

										inputs = Variable(data.narrow(1, 0, 2)).to(device)
										labels = Variable(data.narrow(1, 2, 1)).to(device)
									 
										loss_minibatch = loss_fn(model(inputs), labels)

										optimizer.zero_grad()
										loss_minibatch.backward()
										optimizer.step()

										progress += 1

									loss_trai = loss_fn(model(GetDataObj['var_training_set']), GetDataObj['var_training_labels'])
									loss_test = loss_fn(model(GetDataObj['var_test_set']), GetDataObj['var_test_labels'])										

									if loss_fn_i == 'MSE':
										loss_trai = sqrt(loss_trai)
										loss_test = sqrt(loss_test)

									netdata["plytr"].append(loss_trai)
									netdata["plyte"].append(loss_test)
									
									cycl = 100.*progress/totalworkload
									time = TimerGet('total')[0] * (totalworkload/progress - 1.)
									
									print("\rNet %d of %d - Estimated progress: %d%% (+%ds) - Training epoch %d / %d        " % (counter, totalcount, cycl, time, epoch, EPOCH_NUM), end = '', flush = True)
									
								#vall_dummy = loss_fn(netdata["model"](GetDataObj['var_validation_set']), GetDataObj['var_validation_labels']).cpu().detach().numpy()

								
								TimerAddSave('train')      
								TimerInit('analysis')

								TimerInit('predtime')
								for i in range(ANALYSIS_SAMPLE_SIZE):
									model(GetDataObj['var_validation_set'])
								TimerAddSave('predtime')

								netdata["predt"] = TimerGet('predtime')[1] / ANALYSIS_SAMPLE_SIZE
								
								netdata['lossv'] = loss_fn(model(GetDataObj['var_validation_set']), GetDataObj['var_validation_labels'])
								#if netdata["lossv"] > vall_dummy:
								#	netdata["lossv"] = vall_dummy
								
								if loss_fn_i == "MSE":
									netdata["lossv"] = sqrt(netdata["lossv"])
								print

								TimerAddSave('analysis')

								netdata["hloss"] = hyperloss(HYPERLOSS_FUNCTION, netdata["predt"], netdata["lossv"], GridParameter['maxloss'])

								TimerInit('checktop')
								
								if NetIsTopPerformer(netdata):
									UpdateToplist(netdata)   
								
								TimerAddSave('checktop')
								
								counter += 1

##################### SAVE RESULTS ################################

TimerInit('writetop')
WriteToplist()
TimerAddSave('writetop')

##################### TIME ANALYSIS ###############################

TimerSetSave('total')
total = TimerGet('total')[1]

print("\rGrid search completed after {}. - Analysis and toplist saved in folder '{}'\n".format(TimerGetStr('total')[1], path.topology + TXNAME))

dt = TimerGet('gendata')[1]
dp = 100.*dt/total
print("{:^3.1f}% - data generation   ({:^4.2f}s)".format(dp, dt))

dt = TimerGet('prepnet')[1]
dp = 100.*dt/total
print("{:^3.1f}% - net preparation    ({:^4.2f}s)".format(dp, dt))

dt = TimerGet('train')[1]
dp = 100.*dt/total
print("{:^3.1f}% - net training      ({:^4.2f}s)".format(dp, dt))

dt = TimerGet('analysis')[1]
dp = 100.*dt/total
print("{:^3.1f}% - runtime prediction ({:^4.2f}s)".format(dp, dt))

dt = TimerGet('checktop')[1]
dp = 100.*dt/total
print("{:^3.1f}% - checking toplist   ({:^4.2f}s)".format(dp, dt))

dt = TimerGet('writetop')[1]
dp = 100.*dt/total
print("{:^3.1f}% - updating toplist   ({:^4.2f}s)".format(dp, dt))
