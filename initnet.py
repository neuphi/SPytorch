from glovar import *
from misc import *

hyp = {
	"layer": 2,
	"nodes": 2,
	"activ": "lin",
	"shape": "trap"
}

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


	def __init__(self, hyp):
    	
		super(Net, self).__init__()

		lay = hyp["layer"] + 1
		nod = hyp["nodes"]
		act = hyp["activ"]
		shp = hyp["shape"]

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


def CreateNet(hyp):
	model = Net(hyp)
	print("\n", model)	
	return model

#(loss, minibatch, optimizer)
if __name__ == "__main__":

	hyp["layer"] = 8
	hyp["nodes"] = 23
	hyp["activ"] = "lin"
	hyp["shape"] = "ramp"

	model = CreateNet(hyp)
