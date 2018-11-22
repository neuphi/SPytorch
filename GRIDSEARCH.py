#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################### IMPORT STUFF ############################

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader as DataLoader
#from torch.utils.data import Dataset as Dataset

from system.misc import *
from system.initnet import *
from system.printtorch import *
from system.dataset import *
from system.parameters import *

import system.pathfinder as path


######################## GENERATE DATA ################################

TimerZero('all')
TimerInit('total')
TimerInit('gendata')


print("\ncuda in use" if CUDA else "\nno cuda")

print("\ngenerate data .. ", end = "", flush = True)

GetDataObj = DataGeneratePackage(EXP, 0.8, 0.1, 0.1, CUDA)

TimerSetSave('gendata')
print("done after {}".format(TimerGetStr('gendata')[0]))

##################### TRAINING ###################################

print("initiating grid search\n")

GridParameter = LoadParameters()

counter    = 1
totalcount = GetNetConfigurationNum(GridParameter)

for loss_fn_i in GridParameter['loss_func']:

	for optimizer_i in GridParameter['optimizer']:

		for minibatch in GridParameter['minibatch']:

			for learning_rate in GridParameter['lera_iter']:

				for activ in GridParameter['acti_func']:

					for shape in GridParameter['nodes_shape']:

						for layers in GridParameter['layer_iter']:

							for nodes in GridParameter['nodes_iter']:
				                #initialize net, lossfunction and optimizer       

								TimerInit('prepnet')
								
								netdata = CreateNet(layers, nodes, activ, shape, loss_fn_i, optimizer_i, minibatch, learning_rate)  
								netdata['model'].to(device=args.device)

								loss_fn = initloss(loss_fn_i)
								optimizer = initopt(optimizer_i, netdata['model'], learning_rate)

								#define trainloader
								trainloader = DataLoader(GetDataObj['tensor_training_set'], batch_size=minibatch, shuffle=True, num_workers=0)
								len_loader  = len(trainloader)
								TimerAddSave('prepnet')

								TimerInit('train')

			                	#loop over trainloader
								for i, data in enumerate(trainloader):  
									#make data accessable for model
									inputs, labels = modelinputs(data)  
									#do predictions and calculate loss
									labels_pred = netdata["model"](inputs)
									loss_minibatch = loss_fn(labels_pred, labels)
									#make one step to optimize weights
									optimizer.zero_grad()
									loss_minibatch.backward()
									optimizer.step()
									#fill loss lists with data
									#netdata["model"].cpu()
									#netdata["model"].cpu()
									#print (next(netdata["model"].parameters()).is_cuda)


									clock  = TimerGet('total')[0]
									left   = (clock/counter)*(totalcount-counter)
									remain = clock/(clock+left) * 100.
									print("\rNet %d of %d - Estimated progress: %d%% - Training epoch %d / %d        " % (counter, totalcount, remain, i+1, len_loader), end = '', flush = True)
									
								if loss_fn_i == "MSE":
									##netdata["plytr"].append(np.sqrt(loss_fn(netdata["model"](var_training_set[:,:2]), var_training_labels[:,2]).detach().numpy()))
									netdata["plytr"].append(np.sqrt(loss_fn(netdata["model"](GetDataObj['var_training_set']), GetDataObj['var_training_labels']).cpu().detach().numpy()))
									netdata["plyte"].append(np.sqrt(loss_fn(netdata["model"](GetDataObj['var_test_set']), GetDataObj['var_test_labels']).cpu().detach().numpy()))

								else:
									netdata["plytr"].append(loss_fn(netdata["model"](GetDataObj['var_training_set']), GetDataObj['var_training_labels']).cpu().detach().numpy())
									netdata["plyte"].append(loss_fn(netdata["model"](GetDataObj['var_test_set']), GetDataObj['var_test_labels']).cpu().detach().numpy())                      

								vall_dummy = loss_fn(netdata["model"](GetDataObj['var_validation_set']), GetDataObj['var_validation_labels']).cpu().detach().numpy()

								if netdata["lossv"] > vall_dummy:
									netdata["lossv"] = vall_dummy

								#print (next(netdata["model"].parameters()).is_cuda)
   
								TimerAddSave('train')      
								#make sample predictions (val), measure time (sample size)
								TimerInit('analysis')

								for i in range(ANALYSIS_SAMPLE_SIZE):
									TimerInit('predtime')
									preds = netdata["model"](GetDataObj['var_validation_set'])
									TimerAddSave('predtime')

								netdata["predt"] = TimerGet('predtime')[1] / ANALYSIS_SAMPLE_SIZE
								#fix validation if squared
								if loss_fn_i == "MSE":
									netdata["lossv"] = np.sqrt(netdata["lossv"])

								TimerAddSave('analysis')

								#save hyperloss
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
