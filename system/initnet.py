#from system.glovar import *
#from system.misc import *
import torch
import torch.nn as nn

def getNodesPerLayer(shape, nodes, layer):

	net = []
	nodes_total = 0

	for lay in range(layer):

		n = [0, 0]
		n_count = 0

		if shape == "lin":

			n[0] = nodes
			n[1] = nodes
			n_count += nodes

		elif shape == "trap":

			k = 2 * nodes / layer
			m = layer*0.5
			
			for i in range(2):
			
				cl = float(lay + i)
			
				if cl > m:
					cl = m - (cl%m)
				
				n[i] = round(cl*k)

			n_count += n[i]

		elif shape == "ramp":
			
			k = nodes / layer
	
			for i in range(2):
	
				cl = float(lay + i - 1)
				n[i] = round(nodes - k * cl)
	
			if lay == 0:
				n[1] = nodes
			elif lay == 1:
				n[0] = nodes

			n_count += n[i]				

		if lay == 0:
			n[0] = 2
		if lay == layer - 1:
			n[1] = 1
			n_count = 0

		nodes_total += n_count
		net.append(n)

	return [net, nodes_total]


class Net(nn.Module):

	def __init__(self, netShape, activFunc):
    	
		super(Net, self).__init__()
		self.seq = nn.Sequential()

		lastLayer = len(netShape) - 1

		for i in range(len(netShape)):

			nin, nout = netShape[i][0], netShape[i][1]
			
			if activFunc == "lin":
				self.seq.add_module('lin{}'.format(i), nn.Linear(nin,nout))

			elif activFunc == "rel":
				self.seq.add_module('lin{}'.format(i), nn.Linear(nin,nout))

				if i != lastLayer:
					self.seq.add_module('rel{}'.format(i), nn.ReLU())
                    
			elif activFunc == "tah":
				self.seq.add_module('lin{}'.format(i),nn.Linear(nin,nout))

				if i != lastLayer:
					self.seq.add_module('tah{}'.format(i), nn.Tanh())

			elif activFunc == "sig":
				self.seq.add_module('lin{}'.format(i),nn.Linear(nin,nout))

				if i != lastLayer:
					self.seq.add_module('sig{}'.format(i), nn.Sigmoid())   
    
		#print(self.seq)
		#nn.init.xavier_normal_(self.seq)
                 

	def forward(self, x):
		
		x = self.seq(x)
		return x


def CreateNet(shape, nodes, layer, activFunc):

	netshape, nodesTotal = getNodesPerLayer(shape, nodes, layer)
	model = Net(netshape, activFunc)
	
	return model


def LoadNet(params):

	#return CreateNet()
	return 0


if __name__ == "__main__":
	#data = CreateNet(4, 3, "lin", "trap", "mse", "adam", 8, 1e-3)
	#print(data["model"], "\n", data["nodto"])

	#CreateNet(layer, nodes, shape, activ)
	#model = Net(getNodesPerLayer(layer, nodes, shape)[0], activ)

	print(getNodesPerLayer("trap",5,5))
