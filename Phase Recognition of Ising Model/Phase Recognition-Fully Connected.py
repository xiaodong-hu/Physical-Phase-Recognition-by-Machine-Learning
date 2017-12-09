# python 3.6, tensorflow 1.40rc, tensorboard 1.0
# written by hxd
# Since the code invoke `os package` and are running on Archlinux, 
# you must slightly modify it bofore you run on windows 

import tensorflow as tf 
import numpy as np
import math
import csv
import os

def configuration_read(file_name):
	data = []
	with open(file_name) as f:
		reader = csv.reader(f)
		for row in reader:
			data.append([int(row[0]),int(row[1])]) #append spin configuration
	return data

def spin_read(file_name):
	data = []
	with open(file_name) as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(int(row[2])) #append spin configuration
	return data

def magnetization(spin_data,lattice_size):
	spin_sum = sum(spin_data)		# spin[i] is a list of 1-d spin configuration
	return abs(spin_sum/lattice_size)

def data_read(data_path,tensor_x,tensor_y):

	os.chdir(os.path.pardir)		# go to /数据挖掘导论
	os.chdir(data_path)				# go to /Ising Model Data/data
	list_of_filename = os.listdir()	# LIST of data's filename

	lattice_size = len(open(list_of_filename[0],'rU').readlines())
	# each file has the same lattice; Here we choose the first file as one example
	for file_name in list_of_filename:
		spin_data = spin_read(file_name)				# get the spin data for one file

		# input data is flattened configurations of spins
		tensor_x.append(spin_data)						

		if magnetization(spin_data,lattice_size)>0.9:	# 0.9 is the parameter defined manually
		# input labels are list of phases
			tensor_y.append([1,0])						# [1,0] represent high-T phase					
		else:
			tensor_y.append([0,1])						# [0,1] represent low-T phase

	path_phase = 'Phase Recognition of Ising Model'
	os.chdir(os.path.pardir)	# go back to /Ising Model Data/data
	os.chdir(os.path.pardir)	# go back to /Ising Model Data
	os.chdir(os.path.pardir)	# go back to 数据挖掘导论课题
	os.chdir(path_phase)		# go back to /Phase Recognition of Ising Model

	return len(list_of_filename)

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

	path_data_training_set = 'Ising Model Data/data/training set'
	path_data_test_set = 'Ising Model Data/data/test set'
	path_data = 'Ising Model Data/data'

	lattice_size = 30

	with tf.variable_scope('Inputs'):
		spins = tf.placeholder(tf.float32,[None, lattice_size*lattice_size])	# None indicate the mini-batch
		phases = tf.placeholder(tf.float32,[None, 2])							# None indicate the mini-batch
		# Add physics knowledge: two phases

	with tf.variable_scope('Layers'):
		layer_one = tf.layers.dense(inputs = spins, units = 3, activation = tf.nn.relu, name = 'layer_one')
		outputs = tf.layers.dense(inputs = layer_one, units = 2, activation = tf.nn.sigmoid, name = 'output_layer')
		tf.summary.histogram('layer_one', layer_one)
		tf.summary.histogram('output_layer', outputs)

	with tf.variable_scope('Training'):
		training_loss = tf.losses.mean_squared_error(phases, outputs, scope='traing_loss')
		train_operation = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(training_loss)
		tf.summary.scalar('training_loss', training_loss)						# add loss to scalar summary

	with tf.name_scope('Accuracy'):
		acc = tf.equal(tf.argmax(phases,1), tf.argmax(outputs,1))
		acc = tf.reduce_mean(tf.cast(acc, tf.float32))
		tf.summary.scalar("accuracy", acc)



	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer()) 		# IMPORTANT
		
		# Define the mini-batch of a cycle
		# batchsize = 1 means SGD, quick but easy to run into local minimal
		# Conversely, batchsize = All will take up too much resources and slow down the training process
		
		batchsize = int(input('Input the batch size of your data: '))								
		# Each `batchsize` configurations forms a batch of input data

		if os.path.exists('log'):
			os.system('rm -r log')							# delete the existed /log
		
		merge_operation = tf.summary.merge_all()			# operation to merge all summary
		#writer = tf.summary.FileWriter('./log', sess.graph)	# write to file
		log_dir = './log'
		train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)  
		test_writer = tf.summary.FileWriter(log_dir + '/test')  
		print('\n')

		# Training Cycle
		tensor_x = []	
		tensor_y = []
		training_number = data_read(path_data_training_set,tensor_x,tensor_y)	# read training data
		# Read and Store spin configurations and magnetization for training use
		# No need to manual transform tensor_x and tensor_y inteo np.array() type
		epoches = int(input('Input the training epoch: '))
		epoches = min(epoches,math.floor(training_number/batchsize))
		# allowed integer is chosen to be the epoch
		print('batch size = {}\n'.format(batchsize))
		for step in range(epoches):
			feeds ={spins: tensor_x[batchsize*step:batchsize*(step+1)], phases: tensor_y[batchsize*step:batchsize*(step+1)]}
			loss, results, _ = sess.run([training_loss, merge_operation,train_operation], feed_dict=feeds)
			train_writer.add_summary(results, step)
			if step%20 == 0:
				print('{} batches have been trained, losses {}'.format(step,loss))

		print('\n')
		# Test Cycle
		tensor_x = []
		tensor_y = []
		test_number = data_read(path_data_test_set,tensor_x,tensor_y)
		# Read and Store spin configurations and magnetization for testing
		# No need to manual transform tensor_x and tensor_y inteo np.array() type
		epoches = int(input('Input the training epoch: '))
		epoches = min(epoches,math.floor(test_number/batchsize))
		# allowed integer is chosen to be the epoch
		for step in range(epoches):
			feeds ={spins: tensor_x[batchsize*step:batchsize*(step+1)], phases: tensor_y[batchsize*step:batchsize*(step+1)]}
			# Use the trained parameter to check for test set
			accuracy, results = sess.run([acc,merge_operation], feed_dict=feeds)
			test_writer.add_summary(results, step)
			if step%20 == 0:
				print('{} batches have been tested, accuracy {}'.format(step,accuracy))
