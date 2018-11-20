from system.errorlog import *
from time import time as time

timerStack = {}
timerStack['total']    = [0,0]
timerStack['gendata'] = [0,0]
timerStack['prepnet']  = [0,0]
timerStack['train']    = [0,0]
timerStack['analysis'] = [0,0]
timerStack['checktop'] = [0,0]
timerStack['writetop'] = [0,0]
timerStack['predtime'] = [0,0]

#timer snapshot  = [0,0]

#TimerSaveAdd
#TimerSaveSet

#def TimerUpdate(key):
#	if key in timerStack.keys():
#		timerStack[key] += time() - timerStack[key]
#	elif key == 'all':
#		for value in timerStack.values():
#			value += time() - value
#	else:
#		RaiseError('timer')

def TimerInit(key):
	if key in timerStack.keys():
		timerStack[key][0] = time()
	elif key == 'all':
		t0 = time()
		for value in timerStack.values():
			value[0] = t0
	else:
		ErrorRaise('timer')

def TimerZero(key):
	if key in timerStack.keys():
		timerStack[key][0] = 0
	elif key == 'all':
		for value in timerStack.values():
			value[0] = 0
	else:
		ErrorRaise('timer')

def TimerGet(key):
	if key in timerStack.keys():
		return time() - timerStack[key][0], timerStack[key][1]
	else:
		ErrorRaise('timer')
	return 0, 0

def TimerGetStr(key):
	t = TimerGet(key)
	return '{:^4.2f}s'.format(t[0]), '{:^4.2f}s'.format(t[1]) 

def TimerSetSave(key):
	if key in timerStack.keys():
		timerStack[key][1] = TimerGet(key)[0]
	elif key == 'all':
		for key, value in timerStack.items():
			value[1] = TimerGet(key)[0]
	else:
		ErrorRaise('timer')

def TimerAddSave(key):
	if key in timerStack.keys():
		timerStack[key][1] += TimerGet(key)[0]
	elif key == 'all':
		for key, value in timerStack.items():
			value[1] += TimerGet(key)[0]
	else:
		ErrorRaise('timer')


if __name__ == "__main__":
	TimerZero('all')
	TimerInit('train')
	TimerSave('all')
	print(TimerGetStr('train', 1))
	print('{:^4.2f}s'.format(TimerGet('train', 1)))
