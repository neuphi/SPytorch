#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:13:54 2018

@author: philipp
"""

from system.glovar import *
from system.misc import *
import system.pathfinder as path
import os
import matplotlib.pyplot as plt

toplist = []


def UpdateTopList(model, modelData, modelPerformance, searchParameter):

	global toplist

	newEntry = [model, modelData, modelPerformance, searchParameter]

	if len(toplist) < 10:
		toplist.append(newEntry)
	else:
		toplist[9] = newEntry

	toplist = sorted(toplist, key = lambda data: data[2][4])


def NetIsTopPerformer(hyperLoss):

	global toplist

	if len(toplist) < 10 or hyperLoss < toplist[-1][2][4]:
		return True

	return False


def GetTableHeader(desc):

	s = ""
	for i in range(135):
		s += "#"
	s += "\n#\n"
	s += "# " + desc + "\n#\n"
	s += "# LAYER | NODES | NODES TOTAL | ACTIV | SHAPE | BATCH SIZE | LEARNING RATE | LOSS FUNC | OPTIM | HYPER LOSS | LOSS VALUE | PRED TIME\n"
	s += "# ___________________________________________________________________________________________________________________________________\n"
	return s

def GetNetToString(entry):

	#modelData = [shape, nodes, layer, activFunc]
	#modelPerformance = [trainLossPlot, testLossPlot, predictionTime, validationLoss, hyperLoss]
	
	#searchParameter 				= {}
	#searchParameter['maxLoss'] 	 	= 5
	#searchParameter['epochNum']		= 100
	#searchParameter['sampleSize']	= 1000
	#searchParameter['lossFunction']	= 'MSE'
	#searchParameter['optimizer']	= 'Adam' #rmsprop

	#searchParameter['batchSize']	= 32 #1,2,4,8
	#searchParameter['learningRate']	= 1e-3
	#searchParameter['dataSplit']    = [0.8, 0.1, 0.1]




	modelData = entry[1]
	modelPerformance = entry[2]
	searchParameter = entry[3]

	s  = "# " + '{:^5d}'.format(modelData[2]) + " | "
	s += '{:^5d}'.format(modelData[1]) + " | "
	s += '{:^11d}'.format(0) + " | " # +1 to include output node
	s += '{:^5}'.format(modelData[3]) + " | "
	s += '{:^5}'.format(modelData[0]) + " | "
	s += '{:^10d}'.format(searchParameter["batchSize"]) + " | "
	s += '{:^13.3e}'.format(searchParameter['learningRate']) + " | "
	s += '{:^9}'.format(searchParameter["lossFunction"]) + " | "
	s += '{:^5}'.format(searchParameter["optimizer"]) + " | "
	s += '{:^10.2f}'.format(modelPerformance[4]) + " | " 	#hyperloss
	s += '{:^10.2f}'.format(modelPerformance[3]) + " | " 	#lossvalue
	s += '{:^9.3e}'.format(modelPerformance[2]) + " | \n"	#predTime
	return s

def GetTableBottom():
	return "# ___________________________________________________________________________________________________________________________________|\n"





def StoreNetData(anadir, entry):

	global toplist
	
	j = toplist.index(entry) + 1
	mod = entry[0]

	# MAKE DIR

	netdir = anadir + '/{}'.format(j)

	if not os.path.exists(netdir):
		print("makedir " + netdir)
		os.makedirs(netdir)

	# SAVE MODEL

	torch.save(mod.state_dict(), netdir + "/net{}.h5".format(j))

	# CREATE PLOT

	modelPerformance = entry[2]
	
	epnum 		= len(modelPerformance[0])
	x_axis     	= [i for i in range(epnum)]
	y_axis_trn 	= modelPerformance[0]
	y_axis_tst 	= modelPerformance[1]

	plt.clf()
	plt.figure(j)
	plt.title('Loss Function', fontsize=20)
	plt.xlabel('Epochs')
	plt.ylabel('Error')
	plt.plot(x_axis, y_axis_trn, label = 'Train Loss')
	plt.plot(x_axis, y_axis_tst, label = 'Test Loss')
	plt.legend()
	plt.savefig(netdir + "/loss{}.eps".format(j), format = 'eps')
	
	# WRITE INFO

	with open(netdir + '/info{}.txt'.format(j), 'w') as f:
		f.write(GetTableHeader("Rank {} Net:".format(j)))
		f.write(GetNetToString(entry))
		f.write(GetTableBottom())
		f.write("#\n#\n# Raw Loss Plot Data:\n\n")
		for i in range(epnum):
			f.write(str(y_axis_trn[i]) + "," + str(y_axis_tst[i]) + "\n")

	# SAVE ACTUAL MODEL

	if j == 1:
		
		savedir = path.netstorage + ANALYSIS_ID + '/'

		if not os.path.exists(savedir):
			os.makedirs(savedir)

		torch.save(mod, savedir + TXNAME + '.pth')





def WriteToplist():

	global toplist

	# MAKE DIR
	
	#anadir = 'analysis/topology/{}'.format(TXNAME)
	anadir = path.topology + TXNAME

	if not os.path.exists(anadir):
		print("makedir " + anadir)
		os.makedirs(anadir)

	# WRITE TOPLIST

	with open(anadir + '/toplist.txt', 'w') as f:
		f.write(GetTableHeader("Top {} Performing Models:".format(len(toplist))))
		for entry in toplist:
			f.write(GetNetToString(entry))	
		f.write(GetTableBottom())
		f.write("#\n# Global Info:\n#")
		f.write("\n#\t-analysis ID: {}".format(ANALYSIS_ID))
		f.write("\n#\t-topology:    {}".format(TXNAME))
		f.write("\n#\t-hyperloss:   1e3 time + e^(5*(loss-intloss))")
		f.write("\n#\t-pred time:   mean over {} single sigma predictions".format(ANALYSIS_SAMPLE_SIZE))
		f.write("\n#\t-size sets:   test set {}, training set {}, validation set {}".format(0.8, 0.1, 0.1))

	# CREATE SUBFOLDER

	for entry in toplist:
		StoreNetData(anadir, entry)


if __name__ == "__main__":

	print('test')
	#import initnet	
	#import random

	#LEN_TEST_SET, LEN_TRAINING_SET, LEN_VALIDATION_SET = 80, 10, 10
	
	#act = ["lin", "rel"]
	#shp = ["lin", "ramp", "trap"]

	#for i in range(20):

	#	a = act[random.randrange(2)]
	#	s = shp[random.randrange(3)]
	#	l = random.randrange(2,20)
	#	n = random.randrange(l,20)
	#	h = random.random() * 100.

	#	pt = [random.random() for i in range(100)]
	#	pv = [random.random() for i in range(100)]

	#	netdata = initnet.CreateNet(l, n, a, s, "mse", "adam", 16, 1e-3)
	#	netdata["hloss"] = h
	#	netdata["plytr"] = pt
	#	netdata["plyte"] = pv		

	#	if NetIsTopPerformer(netdata):
	#		UpdateToplist(netdata)

	#WriteToplist()
