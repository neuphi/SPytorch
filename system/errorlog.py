errorStack = {}

errorStack['key'] 	= ['hyper error, invalid error key', 1]
errorStack['timer'] = ['invalid timer addressed', 1]
errorStack['split'] = ['dataset splits do not add up to 1.', 0]

def ErrorPrefix(which):
	if which == 0:
		return 'ERROR! '
	else:
		return 'WARNING! '

def ErrorMsg(key):
	#add event date, time
	print(ErrorPrefix(errorStack[key][1]) + errorStack[key][0])

def ErrorLog(key):
	#todo, add logfile for event	
	return 0
	
def ErrorRaise(key):
	if key in errorStack.keys():
		ErrorMsg(key)
	else:
		ErrorMsg('key')
	

if __name__ == "__main__":
	ErrorRaise('timer')
	ErrorRaise('blub')
