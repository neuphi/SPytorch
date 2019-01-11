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
from system.dataset import *
from system.parameters import *
from system.setdevice import *
import system.pathfinder as path

TimerZero('all')
TimerInit('total')

##################### INITIALIZATION #################################

print("Initiating grid search\n")

analysisId	 = 'CMS-PAS-SUS-12-026'
txName		 = 'T1tttt'

device 			= setDevice()
searchParameter = LoadSearchParameters()
searchRange		= LoadSearchRange()
	
dataSplit	 	= searchParameter['dataSplit']
learningRate 	= searchParameter['learningRate']
whichLossFunc	= searchParameter['lossFunction']
batchSize	 	= searchParameter['batchSize']
epochNum	 	= searchParameter['epochNum']
sampleSize	 	= searchParameter['sampleSize']
maxLoss		 	= searchParameter['maxLoss']
whichOptimizer	= searchParameter['optimizer']

if whichLossFunc == 'MSE':
	lossFunc = nn.MSELoss(reduction = 'elementwise_mean').to(device)

shapeRange 	   = searchRange['shape']
nodesRange 	   = searchRange['nodes']
layerRange    = searchRange['layer']
activFuncRange = searchRange['activFunc']

######################## GENERATE DATA ################################

TimerInit('gendata')
print("\nGenerate data .. ", end = "", flush = True)

GetDataObj = DataGeneratePackage(analysisId, txName, dataSplit, device)

trainingSet    	 = GetDataObj['var_training_set']
trainingLabels 	 = GetDataObj['var_training_labels']
testSet    		 = GetDataObj['var_test_set']
testLabels 		 = GetDataObj['var_test_labels']
validationSet    = GetDataObj['var_validation_set']
validationLabels = GetDataObj['var_validation_labels']

TimerSetSave('gendata')
print("done after {}".format(TimerGetStr('gendata')[0]))

netCurrent   	= 1
netTotal     	= GetNetConfigurationNum(searchRange)
progressCurrent = 0
progressTotal   = netTotal * epochNum * len(GetDataObj['list_training']) / batchSize

##################### TRAINING ###################################

for activFunc in activFuncRange:

	for shape in shapeRange:

		for layer in layerRange:

			for nodes in nodesRange:       

				TimerInit('prepnet')
					
				model = CreateNet(shape, nodes, layer, activFunc).to(device)
				
				if whichOptimizer == 'Adam':
					optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)#.to(device)

				trainloader = DataLoader(GetDataObj['d'], batch_size = batchSize, shuffle = True, num_workers = 0)

				trainingLossPlot = []
				testLossPlot 	 = []

				TimerAddSave('prepnet')
				TimerInit('train')

				for epoch in range(epochNum):

					for i, data in enumerate(trainloader):  

						inputs = data[0]
						labels = data[1]

						loss = lossFunc(model(inputs), labels)

						optimizer.zero_grad()
						loss.backward()
						optimizer.step()

						progressCurrent += 1

					if whichLossFunc == 'MSE':
						trainingLossPlot.append(sqrt(lossFunc(model(trainingSet), trainingLabels)))
						testLossPlot.append(sqrt(lossFunc(model(testSet), testLabels)))
					else:
						trainingLossPlot.append(lossFunc(model(trainingSet), trainingLabels))
						testLossPlot.append(lossFunc(model(testSet), testLabels))
					
					cycl = 100.*progressCurrent/progressTotal
					time = TimerGet('total')[0] * (progressTotal/progressCurrent - 1.)
					
					print("\rNet %d of %d - Estimated progress: %d%% (+%ds) - Training epoch %d / %d        " % (netCurrent, netTotal, cycl, time, epoch, epochNum), end = '', flush = True)
					
				TimerAddSave('train')      
				TimerInit('analysis')
				TimerInit('predtime')

				for i in range(sampleSize):
					model(validationSet)
				TimerAddSave('predtime')

				predictionTime = TimerGet('predtime')[1] / sampleSize
				validationLoss = sqrt(lossFunc(model(validationSet), validationLabels))
				hyperLoss 	   = hyperloss(HYPERLOSS_FUNCTION, predictionTime, validationLoss, maxLoss)
								
				TimerAddSave('analysis')
				TimerInit('checktop')
								
				#if NetIsTopPerformer(hyperloss):

				#	modelData = [shape, nodes, layer, activFunc]
				#	modelPerformance = [trainingLossPlot, testLossPlot, predictionTime, validationLoss, hyperLoss]
					
				#	UpdateTopList(model, modelData, modelPerformance, searchParameter)
					
				TimerAddSave('checktop')
								
				netCurrent += 1

##################### SAVE RESULTS ################################

TimerInit('writetop')
#WriteToplist()
TimerAddSave('writetop')

##################### TIME ANALYSIS ###############################

TimerSetSave('total')
total = TimerGet('total')[1]

print("\rGrid search completed after {}. - Analysis and toplist saved in folder '{}'\n".format(TimerGetStr('total')[1], path.topology + TXNAME))

#timerAnalysis('all')
#timerNew('generateData')
##timerSave('training')
#timerContinue('')
#timerPause('')

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
