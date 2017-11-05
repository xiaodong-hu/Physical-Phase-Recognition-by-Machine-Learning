import csv
import os
import matplotlib.pyplot as plt

def read(i,data):
	with open(str(i)+'.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row)

def magnetization(data):
	m = 0
	for i in range(2500):
		m += int(data[i][2])
	return abs(m/2500)


if __name__ == "__main__":
	data = []
	os.chdir(os.path.pardir)	# temporarily change the work directory to father directory
	path = 'Ising Model Data/data'
	os.chdir(path)

	N = os.listdir()			# count the number of generated files

	#i = int(input('Input the data you want to study:\n'))


	x = []
	y = []
	
	for i in range(1000):	
		read(i,data)
		T = 1.0 + i*(4.0-1.0)/1000.0;
		x.append(T)
		y.append(magnetization(data))
		#print('T = {:.3}\t\tM = {:.3}'.format(T,magnetization(data)))
		data = []
	

	plt.scatter(x,y)
	# plt.xlim((0,50))
	# plt.ylim((0,50))

	plt.show()

'''
	x = []
	y = []

	for i in range(2500):
		if data[i][2] == '1':
			x.append(int(data[i][0]))	#Note that
			y.append(int(data[i][1]))
		else:
			continue
	
	#print(x)
	#print(y)
	plt.scatter(x,y)
	plt.xlim((0,50))
	plt.ylim((0,50))

	plt.show()
'''