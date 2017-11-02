import csv
import os
import matplotlib.pyplot as plt

def read(i,data):
	with open(str(i)+'.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row)

if __name__ == "__main__":
	data = []
	os.chdir(os.path.pardir)	# temporarily change the work directory to father directory
	path = 'Ising Model Data/data'
	os.chdir(path)
	#print(os.getcwd())
	N = os.listdir()			# count the number of generated files
#	for i in range(N):
#		read(i,data)
	read(99,data)
#	read(501,data)
	#print(data)

	x = []
	y = []

	for i in range(10000):
		print(i)
		if data[i][2] == '1':
			x.append(int(data[i][0]))	#Note that
			y.append(int(data[i][1]))
		else:
			continue
	
	#print(x)
	#print(y)
	plt.scatter(x,y)
	plt.xlim((0,100))
	plt.ylim((0,100))

	plt.show()