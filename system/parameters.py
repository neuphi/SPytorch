

def GetNetConfigurationNum(GridParameter):
	return len(GridParameter['loss_func']) * len(GridParameter['optimizer']) * len(GridParameter['minibatch']) * len(GridParameter['lera_iter']) * len(GridParameter['acti_func']) * len(GridParameter['nodes_shape']) * len(GridParameter['layer_iter']) * len(GridParameter['nodes_iter'])

def LoadParameters():

	GridParameter = {}

	GridParameter['maxloss'] 	 = 5
	GridParameter['layer_max']	 = 4
	GridParameter['layer_min']	 = 1
	GridParameter['layer_step']	 = 1
	GridParameter['nodes_max']	 = 20
	GridParameter['nodes_min']	 = 20
	GridParameter['nodes_step']	 = 4
	GridParameter['lera_max']	 = 1.0001e-1
	GridParameter['lera_min']	 = 1e-3
	GridParameter['lera_step']	 = 9e-3
	GridParameter['loss_func']	 = ['MSE']
	GridParameter['optimizer']	 = ['Adam']
	GridParameter['acti_func']	 = ['rel']
	GridParameter['nodes_shape'] = ['trap']
	GridParameter['minibatch']	 = [1,8,32,64,128,256,512,1024]

	GridParameter['layer_iter']	 = range(GridParameter['layer_min'], GridParameter['layer_max']+1, GridParameter['layer_step'])
	GridParameter['nodes_iter']	 = range(GridParameter['nodes_min'], GridParameter['nodes_max']+1, GridParameter['nodes_step'])
	GridParameter['lera_iter']	 = [1e-3]#range(GridParameter['lera_min'], GridParameter['lera_max']+1, GridParameter['lera_step'])


	return GridParameter
