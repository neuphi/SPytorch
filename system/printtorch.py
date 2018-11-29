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


def NetIsTopPerformer(netdata):

	global toplist

	if len(toplist) < 10:
		return True

	#doesnt need to loop over every entry: if toplist[max_int]["hloss"] > ..

	for entry in toplist:
		if entry["hloss"] > netdata["hloss"]:
			return True

	return False


def UpdateToplist(netdata):

	global toplist
	netcopy = netdata.copy()

	if len(toplist) < 10:
		toplist.append(netcopy)
	else:
		toplist[9] = netcopy

	toplist = sorted(toplist, key = lambda data: data["hloss"])




def GetTableHeader(desc):

	a = ""
	for i in range(135):
		a += "#"
	b = "\n#\n"
	c = "# " + desc + "\n#\n"
	d = "# LAYER | NODES | NODES TOTAL | ACTIV | SHAPE | BATCH SIZE | LEARNING RATE | LOSS FUNC | OPTIM | HYPER LOSS | LOSS VALUE | PRED TIME\n"
	e = "# ___________________________________________________________________________________________________________________________________\n"
	return a+b+c+d+e

def GetNetToString(entry):

	a = "# " + '{:^5d}'.format(entry["layer"]) + " | "
	b = '{:^5d}'.format(entry["nodes"]) + " | "
	c = '{:^11d}'.format(entry["nodto"]+1) + " | " # +1 to include output node
	d = '{:^5}'.format(entry["activ"]) + " | "
	e = '{:^5}'.format(entry["shape"]) + " | "
	f = '{:^10d}'.format(entry["batch"]) + " | "
	g = '{:^13.3e}'.format(entry["lrate"]) + " | "
	h = '{:^9}'.format(entry["lossf"]) + " | "
	i = '{:^5}'.format(entry["optim"]) + " | "
	j = '{:^10.2f}'.format(entry["hloss"]) + " | "
	k = '{:^10.2f}'.format(entry["lossv"]) + " | "
	l = '{:^9.3e}'.format(entry["predt"]) + " | \n"
	return a+b+c+d+e+f+g+h+i+j+k+l

def GetTableBottom():
	return "# ___________________________________________________________________________________________________________________________________|\n"





def StoreNetData(anadir, entry):

	global toplist
	
	j = toplist.index(entry) + 1
	mod = entry['model']

	# MAKE DIR

	netdir = anadir + '/{}'.format(j)

	if not os.path.exists(netdir):
		print("makedir " + netdir)
		os.makedirs(netdir)

	# SAVE MODEL

	torch.save(mod.state_dict(), netdir + "/net{}.h5".format(j))

	# CREATE PLOT
	
	epnum 		= len(entry["plytr"])
	x_axis     	= [i for i in range(epnum)]
	y_axis_trn 	= entry["plytr"]
	y_axis_val 	= entry["plyte"]

	plt.clf()
	plt.figure(j)
	plt.title('Loss Function', fontsize=20)
	plt.xlabel('Epochs')
	plt.ylabel('Error')
	plt.plot(x_axis, y_axis_trn, label = 'Train Loss')
	plt.plot(x_axis, y_axis_val, label = 'Test Loss')
	plt.legend()
	plt.savefig(netdir + "/loss{}.eps".format(j), format = 'eps')
	
	# WRITE INFO

	with open(netdir + '/info{}.txt'.format(j), 'w') as f:
		f.write(GetTableHeader("Rank {} Net:".format(j)))
		f.write(GetNetToString(entry))
		f.write(GetTableBottom())
		f.write("#\n#\n# Raw Loss Plot Data:\n\n")
		for i in range(epnum):
			f.write(str(y_axis_trn[i]) + "," + str(y_axis_val[i]) + "\n")

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
		f.write(GetTableHeader("Top 10 Performing Models:"))
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

	import initnet	
	import random

	LEN_TEST_SET, LEN_TRAINING_SET, LEN_VALIDATION_SET = 80, 10, 10
	
	act = ["lin", "rel"]
	shp = ["lin", "ramp", "trap"]

	for i in range(20):

		a = act[random.randrange(2)]
		s = shp[random.randrange(3)]
		l = random.randrange(2,20)
		n = random.randrange(l,20)
		h = random.random() * 100.

		pt = [random.random() for i in range(100)]
		pv = [random.random() for i in range(100)]

		netdata = initnet.CreateNet(l, n, a, s, "mse", "adam", 16, 1e-3)
		netdata["hloss"] = h
		netdata["plytr"] = pt
		netdata["plyte"] = pv		

		if NetIsTopPerformer(netdata):
			UpdateToplist(netdata)

	WriteToplist()
