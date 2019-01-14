



def loadFile(whichFile):

	with file.open(whichFile, 'r') as f:
		content = f.splitlines() + '!'

	d = {}

	for line in content:
		for i in range(len(line)):
			if line[i:i+1] == '/t'
				if key == None:
					key == line[:i]
					j = i

				i += 2
			elif key != None and line[i] == '#' or line[i] == '!':
				d[key] = line[j:i-1]

	return d
			

	
