from glovar import *
from misc import *

class Net(nn.Module):

	def getNodesPerLayer(self, shp, nnod, lay, nlay):

		n = [0, 0]

		if shp == "lin":

			n[0] = nnod
			n[1] = nnod

		elif shp == "trap":

			k = 2 * nnod / nlay
			m = nlay*0.5
			
			for i in range(2):
			
				cl = float(lay + i)
			
				if cl > m:
					cl = m - (cl%m)
				
				n[i] = round(cl*k)	

		elif shp == "ramp":
			
			k = nnod / nlay
	
			for i in range(2):
	
				cl = float(lay + i - 1)
				n[i] = round(nnod - k * cl)
	
			if lay == 0:
				n[1] = nnod
			elif lay == 1:
				n[0] = nnod				

		if lay == 0:
			n[0] = 2
		if lay == nlay - 1:
			n[1] = 1

		return n


	def __init__(self, netdata):
    	
		super(Net, self).__init__()

		lay = netdata["layer"] + 1
		nod = netdata["nodes"]
		act = netdata["activ"]
		shp = netdata["shape"]

		self.seq = nn.Sequential()		

		for i in range(lay):
			
			nin, nout = self.getNodesPerLayer(shp, nod, i, lay)

			if act == "lin":
				self.seq.add_module('lin{}'.format(i),nn.Linear(nin,nout))

			elif act == "rel":
				self.seq.add_module('lin{}'.format(i),nn.Linear(nin,nout))
				if i < lay - 1:
					self.seq.add_module('rel{}'.format(i),nn.ReLU())

	def forward(self, x):
		
		x = self.seq(x)
		return x


def CreateNet(layer, nodes, activ, shape, lossf, optim):
	
	netdata["layer"] = layer 	
	netdata["nodes"] = nodes
	netdata["activ"] = activ
	netdata["shape"] = shape
	netdata["lossf"] = lossf
	netdata["optim"] = optim

	netdata["hloss"] = 1e5
	netdata["lossv"] = 1e5
	netdata["predt"] = 1e5 

	netdata["model"] = Net(netdata)
	print("\n", netdata["model"])	
	return netdata

if __name__ == "__main__":
	model = CreateNet(8, 23, "lin", "ramp", "mse", "adam")
