from glovar import *
from misc import *
import os
import matplotlib.pyplot as plt


toplist = []


def NetIsTopPerformer(netdata):

	global toplist

	if len(toplist) < 10:
		return True

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
	for i in range(120):
		a += "#"
	b = "\n#\n"
	c = "# " + desc + "\n#\n"
	d = "# LAYER | NODES | ACTIV | SHAPE | BATCH SIZE | LEARNING RATE | LOSS FUNC | OPTIM | HYPER LOSS | LOSS VALUE | PRED TIME\n"
	e = "# _____________________________________________________________________________________________________________________\n"
	return a+b+c+d+e

def GetNetToString(entry):
	a = "# " + '{:^5d}'.format(entry["layer"]) + " | "
	b = '{:^5d}'.format(entry["nodes"]) + " | "
	c = '{:^5}'.format(entry["activ"]) + " | "
	d = '{:^5}'.format(entry["shape"]) + " | "
	e = '{:^10d}'.format(entry["batch"]) + " | "
	f = '{:^13.3e}'.format(entry["lrate"]) + " | "
	g = '{:^9}'.format(entry["lossf"]) + " | "
	h = '{:^5}'.format(entry["optim"]) + " | "
	i = '{:^10.2f}'.format(entry["hloss"]) + " | "
	j = '{:^10.2f}'.format(entry["lossv"]) + " | "
	k = '{:^9.3e}'.format(entry["predt"]) + " | \n"
	return a+b+c+d+e+f+g+h+i+j+k

def GetTableBottom():
	return "# _____________________________________________________________________________________________________________________\n"





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

	plt.figure(j)
	plt.title('Loss Function', fontsize=20)
	plt.xlabel('Epochs')
	plt.ylabel('Error')
	plt.plot(x_axis, y_axis_trn, label = 'Train Loss')
	plt.plot(x_axis, y_axis_val, label = 'Test Loss')
	plt.legend()
	plt.savefig(netdir + "/loss{}.png".format(j))
	
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
	
	anadir = 'analysis/topology/{}'.format(TXNAME)

	if not os.path.exists(anadir):
		print("makedir " + anadir)
		os.makedirs(anadir)

	# WRITE TOPLIST

	with open(anadir + '/toplist.txt', 'w') as f:
		f.write(GetTableHeader("Top 10 Performing Models:"))
		for entry in toplist:
			f.write(GetNetToString(entry))	
		f.write(GetTableBottom())
		f.write("#\n# Global Info:\n#\n")
		f.write("#\t-analysis ID: {}".format(ANALYSIS_ID) + "\n#\t-topology: {}".format(TXNAME))

	# CREATE SUBFOLDER

	for entry in toplist:
		StoreNetData(anadir, entry)



if __name__ == "__main__":

	import initnet	
	import random
	
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

		netdata = initnet.CreateNet(l, n, a, s, "mse", "adam")
		netdata["hloss"] = h
		netdata["plytr"] = pt
		netdata["plyte"] = pv		

		if NetIsTopPerformer(netdata):
			UpdateToplist(netdata)

	WriteToplist()
