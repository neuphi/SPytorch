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
import system.pathfinder as path

torch.multiprocessing.set_start_method("spawn")

if torch.cuda.is_available():
	devCount = torch.cuda.device_count()
	print('CUDA available: {} GPU{} found'.format(devCount, 's' if devCount > 1 else ''))
	whichDevice = min(int(input('Select Device: (-1 = CPU): ')), devCount - 1)
else:
	whichDevice = -1

if whichDevice >= 0:
	device = torch.device('cuda:' + str(whichDevice))
else:
	device = torch.device('cpu')

print("\nUsing CUDA on {}".format(device) if whichDevice >= 0 else "\nNo CUDA, running on CPU")

######################## GENERATE DATA ################################

TimerZero('all')
TimerInit('total')
TimerInit('gendata')

print("\nGenerate data .. ", end = "", flush = True)

GetDataObj = DataGeneratePackage(ANALYSIS_ID, TXNAME, [0.8, 0.1, 0.1], device)

TimerSetSave('gendata')
print("done after {}".format(TimerGetStr('gendata')[0]))

##################### TRAINING ###################################

print("Initiating grid search\n")

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
								model 	= netdata['model'].to(device)

								#loss_fn = initloss(loss_fn_i)
								#optimizer = initopt(optimizer_i, model, learning_rate)

								loss_fn   = nn.MSELoss(reduction = 'elementwise_mean').to(device)
								optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#.to(device)

								#trainloader = DataLoader(GetDataObj['tensor_training'], batch_size=minibatch, shuffle=True, num_workers=0)
								trainloader = DataLoader(GetDataObj['d'], batch_size = minibatch, shuffle = True, num_workers = 0)

								TimerAddSave('prepnet')
								TimerInit('train')

								for epoch in range(EPOCH_NUM):

									for i, data in enumerate(trainloader):  

										inputs = data[0]#torch.tensor(data.narrow(1, 0, 2)).to(device)
										labels = data[1]#torch.tensor(data.narrow(1, 2, 1)).to(device)

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
