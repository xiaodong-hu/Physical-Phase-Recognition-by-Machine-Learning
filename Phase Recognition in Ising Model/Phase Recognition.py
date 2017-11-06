import tensorflow as tf 
import numpy as np 
import csv
import os

def configuration_read(i,data):
	with open(str(i)+'.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			data.append([row[0],row[1]]) #append spin configuration

def spin_read(i,data):
	with open(str(i)+'.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row[2]) #append spin configuration

def magnetization(spin_data,lattice_size):
	m = 0
	for i in range(lattice_size):
		m += int(spin_data[i])
	return abs(m/2500)

def training_set_generation(number_of_training_set,tensor_x,tensor_y):
	#lattice_size = os.system('wl -l 0.csv')
	configuration_data = []
	spin_data = []
	lattice_size = len(open('0.csv','rU').readlines())  #each file has the same lattice

	for i in range(number_of_training_set):	
		#configuration_read(i,configuration_data)
		spin_read(i,spin_data)
		tensor_x.append(spin_data)
		if magnetization(spin_data,lattice_size)>0.9:
			tensor_y.append(1)
		else:
			tensor_y.append(0)
		spin_data = []
		configuration_data = []


def add_layer(inputs,in_size,out_size,activation_function=None):
	#Add a layer of linear model; default activation function is Relus	
	weight = tf.Variable(tf.random_normal([in_size,out_size]))
	bias = tf.Variable(tf.zero[1, out_size] + 0.1) 
	# bias is NOT recommended to be default zero
	Wx_plus_b = tf.matmul(inputs, weight) + bias
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs



if __name__ == "__main__":
	
	os.chdir(os.path.pardir)	# temporarily change the work directory to father directory
	path = 'Ising Model Data/data'
	os.chdir(path)
	number_of_training_set = len(os.listdir())	# count the number of generated files
	tensor_x = []
	tensor_y = []
	training_set_generation(number_of_training_set,tensor_x,tensor_y)
	#print(tensor_x[0],'\n',tensor_x[1])
	tensor_x = np.array(tensor_x)
	tensor_y = np.transpose([np.array(tensor_y)])

	with tf.variable_scope('Inputs'):
		spins = tf.placeholder(tf.float32, shape = tensor_x.shape, name = 'x')
		phases = tf.placeholder(tf.float32, shape = tensor_y.shape, name = 'y')

	with tf.variable_scope('Layers'):
		l1 = tf.layers.dense(inputs = spins, units = 3, activation = tf.nn.relu, name = 'layer_one')
		outputs = tf.layers.dense(inputs = l1, units = 1, activation = tf.nn.sigmoid, name = 'Output_layer')
		tf.summary.histogram('hidden_out', l1)
		tf.summary.histogram('prediction', outputs)

	loss = tf.losses.mean_squared_error(tensor_y, outputs, scope='loss')
	train_operation = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
	tf.summary.scalar('loss', loss)					# add loss to scalar summary


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer()) # IMPORTANT
	
		writer = tf.summary.FileWriter('/home/hxd/Documents/数据挖掘导论课题-Phase Recognition by Machine Learning/Phase Recognition in Ising Model/log', sess.graph)		# write to file
		merge_operation = tf.summary.merge_all()		# operation to merge all summary

		print('\n{} training data in all\n\n'.format(number_of_training_set))
		for step in range(100):
    		# train and net output
			_, result = sess.run([train_operation,merge_operation],feed_dict = {spins: tensor_x, phases: tensor_y})
			writer.add_summary(result, step)
			if (step+1)%50 == 0:
				print('{} data has been trained'.format(step))
