import tensorflow as tf 
import numpy as np 
import csv
import os

def configuration_read(file_name,data):
	with open(file_name) as f:
		reader = csv.reader(f)
		for row in reader:
			data.append([row[0],row[1]]) #append spin configuration

def spin_read(file_name,data):
	with open(file_name) as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row[2]) #append spin configuration

def magnetization(spin_data,lattice_size):
	m = 0
	for i in range(lattice_size):
		m += int(spin_data[i])
	return abs(m/lattice_size)

def data_read(list_of_set,tensor_x,tensor_y):
	#lattice_size = os.system('wl -l 0.csv')
	configuration_data = []
	spin_data = []
	lattice_size = len(open(list_of_set[0],'rU').readlines())
	#each file has the same lattice; Here we choose the first file as one example
	for file_name in list_of_set:	
		spin_read(file_name, spin_data)
		tensor_x.append(spin_data)						# input data is configurations of spins
		if magnetization(spin_data,lattice_size)>0.9:	# 0.9 is the parameter defined manually
			tensor_y.append(1)							# input label is list of phases
		else:
			tensor_y.append(0)
		spin_data = []									# Reset to be \varnothing
		configuration_data = []							# Reset to be \varnothing


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
	
	os.chdir(os.path.pardir)	# go to /数据挖掘导论
	path_data_training_set = 'Ising Model Data/data/training set'
	path_data_test_set = 'Ising Model Data/data/test set'
	path_data = 'Ising Model Data/data'

	os.chdir(path_data_training_set)		# go to /Ising Model Data/data
	list_of_set = os.listdir()		# list of names of training set
	tensor_x = []
	tensor_y = []
	data_read(list_of_set,tensor_x,tensor_y)	# read training data
	# Read and Store configurations and phases
	tensor_x = np.array(tensor_x)
	tensor_y = np.transpose([np.array(tensor_y)])	
	# Transpose the raw vector [a,b,c] to column vector [[a],[b],[c]] to concord with type of tensor_x
	
	os.chdir(os.path.pardir)	# go back to /Ising Model Data/data
	os.chdir(os.path.pardir)	# go back to /Ising Model Data
	os.chdir(os.path.pardir)	# go back to 数据挖掘导论课题
	path_phase = 'Phase Recognition in Ising Model'
	os.chdir(path_phase)		# go to /Phase Recognition is Ising Model
	if os.path.exists('log'):
		os.system('rm -r log')	# delete the existed /log

	with tf.variable_scope('Inputs'):
		spins = tf.placeholder(tf.float32, shape = tensor_x.shape, name = 'Spins')
		phases = tf.placeholder(tf.float32, shape = tensor_y.shape, name = 'Phases')

	with tf.variable_scope('Layers'):
		l1 = tf.layers.dense(inputs = spins, units = 3, activation = tf.nn.relu, name = 'layer_one')
		outputs = tf.layers.dense(inputs = l1, units = 1, activation = tf.nn.sigmoid, name = 'Output_layer')
		tf.summary.histogram('hidden_layer1', l1)
		tf.summary.histogram('prediction_output_layer', outputs)

	with tf.variable_scope('Training'):
		loss = tf.losses.mean_squared_error(tensor_y, outputs, scope='loss')
		train_operation = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
		tf.summary.scalar('loss', loss)					# add loss to scalar summary

	with tf.name_scope('Accuracy'):
		# Accuracy
		acc = tf.equal(tf.argmax(outputs, 1), tf.argmax(tensor_y, 1))
		acc = tf.reduce_mean(tf.cast(acc, tf.float32))
		tf.summary.scalar("accuracy", acc)


	config = tf.ConfigProto(device_count={"CPU": 4},	# limit to num_cpu_core CPU usage
		inter_op_parallelism_threads = 8, 
		intra_op_parallelism_threads = 8,
		log_device_placement=False)

	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer()) 	# IMPORTANT
	
		writer = tf.summary.FileWriter('log', sess.graph)		# write to file
		merge_operation = tf.summary.merge_all()		# operation to merge all summary

		print('\n{} training data in all\n'.format(len(list_of_set)))
		training_step = input('Input the steps of training:')
		print('\n')
		# Training cycle
		for step in range(100):
    		# train and net output
			_, result = sess.run([train_operation,merge_operation],feed_dict = {spins: tensor_x, phases: tensor_y})
			writer.add_summary(result, step)

			if (step+1)%50 == 0:
				print('{} steps has been trained'.format(step+1))

		print("Accuracy:", acc.eval({spins: tensor_x, phases: tensor_y}))