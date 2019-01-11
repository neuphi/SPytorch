import torch

torch.multiprocessing.set_start_method("spawn")

def setDevice():

	if torch.cuda.is_available():
		devCount = torch.cuda.device_count()
		print('CUDA available: {} GPU{} found'.format(devCount, 's' if devCount > 1 else ''))
		whichDevice = min(int(input('Select Device: (-1 = CPU): ')), devCount - 1)
	else:
		whichDevice = -1

	if whichDevice >= 0:
		device = torch.device('cuda:' + str(whichDevice))
	else:
		device = torch.device('cpu')

	print("\nUsing CUDA on {}".format(device) if whichDevice >= 0 else "\nNo CUDA, running on CPU")

	return device
