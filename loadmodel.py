import os
from smodels.experiment.databaseObj import Database
from smodels.tools.physicsUnits import GeV, fb
from system.initnet import *
from system.dataset import *
import system.pathfinder as path

def getUpperLimit(expres, topo, masses, predict):

	netFile = path.netstorage + expres.id() + '/' + topo + '.pth'

	if predict and os.path.isfile(netFile):

		model = torch.load(netFile)
		#model.eval()

		ult = model(torch.tensor(masses))
		return [round(sub.item(),3) for sub in ult][0]
	else:
		return round(expres.getUpperLimitFor(txname=topo, mass=masses).asNumber(fb), 3)

	return 0


if __name__ == "__main__":

	exp    = 'CMS-PAS-SUS-12-026'
	topo   = 'T1tttt'

	expres = GetExpRes(exp)
	masses = [[ 800., 150.], [ 800., 150.]]
	
	ulp = getUpperLimit(expres, topo, masses, True)
	uli = getUpperLimit(expres, topo, masses, False)
	print('predicted:', ulp, '\ninterpol:', uli, '\nerror:', round(abs(1.- ulp/uli)*100.,3), '%')

#if neural net found, automatically use, else interpolate
#expand nn to 4(+4) input parameter for both branches?
#how to handle unum (GeV, fb) -> convert back from *GeV to simple float in getUpperLimit method?

