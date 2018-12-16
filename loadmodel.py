from math import sqrt as sqrt

from system.misc import *
from system.initnet import *
from system.printtorch import *
from system.dataset_new import *
from system.parameters import *

import system.pathfinder as path

exp, topo = "CMS-PAS-SUS-12-026", "T1tttt"#"CMS-PAS-SUS-13-018", "T2bb"
#exp, topo = ANALYSIS_ID, TXNAME

GetDataObj 	= DataGeneratePackage(exp, topo, [0.8,0.1,0.1], 'cpu')

GridParameter = LoadParameters()

loss_fn 	= GridParameter['loss_func'][0]
optimizer	= GridParameter['optimizer'][0]
minibatch 	= GridParameter['minibatch'][0]
learn_rate	= GridParameter['lera_iter'][0]
activ 		= GridParameter['acti_func'][0]
shape 		= GridParameter['nodes_shape'][0]
layers 		= GridParameter['layer_iter'][0]
nodes 		= GridParameter['nodes_iter'][0]

netdata = CreateNet(layers, nodes, activ, shape, loss_fn, optimizer, minibatch, learn_rate)
netdata['model'].load_state_dict(torch.load(path.topology + TXNAME + "/1/net1.h5"))
netdata['model'].eval()



loss_fn = nn.MSELoss(reduction = 'elementwise_mean')#.to(device)
loss = loss_fn(netdata['model'](GetDataObj['var_training_set']), GetDataObj['var_training_labels'])

print(sqrt(loss.item()))



