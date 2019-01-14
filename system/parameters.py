
def GetNetConfigurationNum(searchRange):
	return len(searchRange['layer']) * len(searchRange['nodes']) * len(searchRange['shape']) * len(searchRange['activFunc'])

def LoadSearchRange():

	layerMin, layerMax, layerStep  = 2, 3, 1
	nodesMin, nodesMax, nodesStep  = 2, 6, 2
	shape	  					   = ['lin']
	activationFunction 			   = ['rel']

	searchRange 			 = {}
	searchRange['layer'] 	 = range(layerMin, layerMax+1, layerStep)
	searchRange['nodes']  	 = range(nodesMin, nodesMax+1, nodesStep)
	searchRange['shape']  	 = shape
	searchRange['activFunc'] = activationFunction

	return searchRange


def LoadSearchParameters():

	searchParameter 				= {}
	searchParameter['maxLoss'] 	 	= 5
	searchParameter['epochNum']		= 200
	searchParameter['sampleSize']	= 10000
	searchParameter['lossFunction']	= 'MSE'
	searchParameter['optimizer']	= 'Adam' #rmsprop

	searchParameter['batchSize']	= 4 #1,2,4,8
	searchParameter['learningRate']	= 1e-3
	searchParameter['dataSplit']    = [0.8, 0.1, 0.1]

	return searchParameter
