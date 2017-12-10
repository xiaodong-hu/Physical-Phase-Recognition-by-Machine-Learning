# python 3.6, tensorflow 1.40rc, tensorboard 1.0
# written by hxd
# Note: Since the code invoke `os package` and are running on Archlinux, 
# you must slightly modify it bofore you run on windows 

import tensorflow as tf 
import numpy as np
import math
import csv
import os
import argparse
import shutil
import random

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

def data_read(path_data,path_data_training,tensor_x,tensor_y):

	os.chdir(os.path.pardir)		# go back to /数据挖掘导论
	os.chdir(path_data)				# go to /Ising Model Data/data
	os.chdir(path_data_training)	# go to /Ising Model Data/data/lattice size 20/training, for example
	
	list_of_filename = os.listdir()	# list of data's filename
	random.shuffle(list_of_filename)# shuffle the list, IMPORTANT to training as well as testing !!!

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

	path_visualization = 'Phase Recognition of Ising Model'
	os.chdir(os.path.pardir)	# go back to /Ising Model Data/data/lattice size 20
	os.chdir(os.path.pardir)	# go back to /Ising Model Data/data
	os.chdir(os.path.pardir)	# go back to /Ising Model Data
	os.chdir(os.path.pardir)	# go back to 数据挖掘导论课题
	os.chdir(path_visualization)# go back to /Phase Recognition of Ising Model

	return len(list_of_filename)

def get_lattice_size(path_to_data):

	os.chdir(path_to_data)			# go to `/Ising Model Data/data/lattice size 20/training set`, for example
	list_of_filename = os.listdir()	# LIST of filenames
	lattice_size = len(open(list_of_filename[0],'rU').readlines())				
	# Note that each data has the same size, so this code always works

	os.chdir(os.path.pardir)		# go back to `/Ising Model Data/data/lattice size 20`
	os.chdir(os.path.pardir)		# go back to `/Ising Model Data/data`
	return math.sqrt(lattice_size)

def clear_events(graph_dir):
	if os.path.isdir(graph_dir):
		shutil.rmtree(graph_dir)


def get_parameters():
	
	parser = argparse.ArgumentParser(description = 'Fully-Connected Phase Regonition')

	parser.add_argument('-s', '--size', type=int, default=20, help='lattice size, default 20')
	parser.add_argument('-b', '--batch', type=int, default=2, help='mini-batch size, default 2')
	# Some notes about mini-batch of a cycle:
	# batchsize = 1, namely put into data one by one, indicates SGD, quick but easy to run into local minimal
	# Conversely, batchsize = All will take up too much resources and extremely slow down the training process
	parser.add_argument('-e', '--epoch', type=int, default=500, help='epoch size, default 100')
	parser.add_argument('-d', '--delete', type=str, default='no', help='whether remove the /log directory, default no')
	args = parser.parse_args()

	return args.size, args.batch, args.epoch, args.delete

	

'''
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
'''



if __name__ == "__main__":

	lattice_size, batchsize, epoches, answer = get_parameters()

	filename = 'lattice size '+str(lattice_size)
	path_data = 'Ising Model Data/data'
	path_data_training_set = filename+'/training set'
	path_data_test_set = filename+'/test set'

	'''
	# Change to the data directory as the work directory 
	os.chdir(os.path.pardir)		# go to /数据挖掘导论
	path_data = 'Ising Model Data/data'
	os.chdir(path_data)				# go to /Ising Model Data/data

	# Define the mini-batch of a cycle
	# batchsize = 1 means SGD, quick but easy to run into local minimal
	# Conversely, batchsize = All will take up too much resources and extremely slow down the training process
	batchsize = int(input('Input the batch size of your data: '))								
	# Each `batchsize` configurations forms a batch of input data
	epoches = int(input('Input the wanted training epoch: '))

	filename = 'lattice size 20'

	path_data_training_set = filename+'/training set'
	path_data_test_set = filename+'/test set'
	#path_data_training_set = 'Ising Model Data/data/training set'
	#path_data_test_set = 'Ising Model Data/data/test set'
	lattice_size = get_lattice_size(path_data_training_set)
	'''

	with tf.variable_scope('Inputs'):
		spins = tf.placeholder(tf.float32,[None, lattice_size*lattice_size])	# None wait for mini-batch
		phases = tf.placeholder(tf.float32,[None, 2])							# None wait for mini-batch
		# Add physics knowledge: two phases

	with tf.variable_scope('Layers'):
		layer_one = tf.layers.dense(inputs = spins, units = 10, activation = tf.nn.relu, name = 'layer_one')
		layer_two = tf.layers.dense(inputs = layer_one, units = 5, activation = tf.nn.relu, name = 'layer_two')
		outputs = tf.layers.dense(inputs = layer_two, units = 2, activation = tf.nn.sigmoid, name = 'output_layer')
		tf.summary.histogram('layer_one', layer_one)
		tf.summary.histogram('layer_one', layer_two)
		tf.summary.histogram('output_layer', outputs)

	with tf.variable_scope('Training'):
		training_loss = tf.losses.mean_squared_error(phases, outputs, scope='traing_loss')
		train_operation = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(training_loss)
		tf.summary.scalar('training_loss', training_loss)						# add loss to scalar summary

	with tf.name_scope('Accuracy'):
		acc = tf.equal(tf.argmax(phases,1), tf.argmax(outputs,1))
		acc = tf.reduce_mean(tf.cast(acc, tf.float32))
		tf.summary.scalar("accuracy", acc)


	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer()) 		# IMPORTANT
		'''
		os.chdir(os.path.pardir)		# go back to /Ising Model Data
		os.chdir(os.path.pardir)		# go back to /数据挖掘导论
		path_visualization = 'Phase Recognition of Ising Model'
		os.chdir(path_visualization)	# go back to /Phase Recognition of Ising Model
		'''
		if os.path.exists('log'):
			if answer == 'yes':
				os.system('rm -r log')						# delete the existed /log
		
		merge_operation = tf.summary.merge_all()			# operation to merge all summary
		# write data seperatedly to files
		log_dir = './log/'+filename
		graph_dir = './log/graph'
		clear_events(graph_dir)							
		# clear graph_dir every time for avoidance of multi-generation of the graph events file
		graph_writer = tf.summary.FileWriter(graph_dir, sess.graph)
		train_writer = tf.summary.FileWriter(log_dir + '/train')
		test_writer = tf.summary.FileWriter(log_dir + '/test')  

		# data reading
		tensor_x_training = []	
		tensor_y_training = []
		training_number = data_read(path_data, path_data_training_set, tensor_x_training, tensor_y_training)	# read training data
		# Read and Store spin configurations and magnetization for training use
		tensor_x_test = []	
		tensor_y_test = []
		test_number = data_read(path_data, path_data_test_set, tensor_x_test, tensor_y_test)	# read test data
		# Read and Store spin configurations and magnetization for testing use
		# No need to manual transform tensor_x and tensor_y inteo np.array() type

		effective_epoches = min(epoches,
					math.floor(training_number/batchsize),
					math.floor(test_number/batchsize))
		
		print('Training and Testing for file {}\n'.format(filename))

		for step in range(effective_epoches):

			# Training Cycles
			training_feeds = {spins: tensor_x_training[batchsize*step:batchsize*(step+1)], 
							phases: tensor_y_training[batchsize*step:batchsize*(step+1)]}
			loss, results, _ = sess.run([training_loss, merge_operation, train_operation], feed_dict=training_feeds)
			train_writer.add_summary(results, step)
			#print('loss {}'.format(loss))

			# Testing Cycles
			test_feeds = {spins: tensor_x_test[batchsize*step:batchsize*(step+1)], 
						phases: tensor_y_test[batchsize*step:batchsize*(step+1)]}
			results, accuracy = sess.run([merge_operation, acc], feed_dict=test_feeds)
			test_writer.add_summary(results, step)
			#print('test accuracy {}'.format(accuracy))
			
			if step%10 == 0:
				print('step {}, \ttrainging losses {:.3f}\ttesting accuracy {:.3f}'.format(step,loss,accuracy))