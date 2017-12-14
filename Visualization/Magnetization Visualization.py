import csv
import os
import matplotlib.pyplot as plt
import re
import shutil
import multiprocessing as mp


def configuration_read(file_name):
	data = []
	with open(file_name) as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(int(row[2])) #append spin configuration
	return data
'''
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
'''
def magnetization(spin_data,lattice_size):
	spin_sum = sum(spin_data)		# spin[i] is a list of 1-d spin configuration
	return abs(spin_sum/(lattice_size*lattice_size))

def visualization(lattice_size, file_number):
	# work only in `training set` directory

	T_max = 4.0
	T_min = 1.0
	T_delta = (T_max-T_min)/file_number
	# The Temperature interval must be manually set !!!

	x = []
	y = []
	i = 0

	file_name_list = os.listdir()
	#list.sort(file_name_list)		# NECESSARY to sort the randomly reading list
	#print(file_name_list)		
	file_name_list = sorted(file_name_list, key=lambda d : int(d.split('.')[0]))
	#print(file_name_list)
	for filename in file_name_list:
		spin_data = configuration_read(filename)
		T = T_min + i*T_delta
		x.append(T)
		y.append(magnetization(spin_data, lattice_size))
		i = i + 1
		plt.plot(x,y,'bo')
		plt.title('Lattice Size '+str(lattice_size))
		#sns.lmplot("x","y",data=pd_res,fit_reg=False,size=5,hue="")
	#print(x)
	#print(y)
	plt.show()

def run(dirname):

	os.chdir(dirname)					# go to /Ising Model Data/data/dirname

	if os.path.isdir('draw'):			# check /draw is existed or not
		shutil.rmtree('draw')

	os.system('mkdir draw')
	os.system('cp -frap training\ set/* draw/')
	os.system('cp -frap test\ set/* draw/')

	draw_data_path = 'draw'
	os.chdir(draw_data_path)		# go to /Ising Model Data/data/dirname/training set
	file_number = len(os.listdir())
	lattice_size = re.sub('\D','',dirname)	# get all number in char
	#print(lattice_size)
	visualization(int(lattice_size), file_number)
	os.chdir(os.path.pardir)			# go back to /Ising Model Data/data/dirname

	if os.path.isdir('draw'):			# check /draw is existed or not
		shutil.rmtree('draw')

	os.chdir(os.path.pardir)			# go back to /Ising Model Data/data


if __name__ == "__main__":

	os.chdir(os.path.pardir)	# temporarily change the work directory to /数据挖掘导论
	data_path = 'Ising Model Data/data/'
	os.chdir(data_path)						# go to /Ising Model Data/data
	lattice_name_list = os.listdir()		# list all files
	
	pool = mp.Pool(processes=8)
	pool.map(run, lattice_name_list)


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
