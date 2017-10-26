import tensorflow as tf 
import numpy as np 

x = tf.placeholder(tf.float32,[None,100])
y = tf.placeholder(tf.float32,[None,2])

W = tf.Variable(0)
b = tf.Variable(0)

def add_linearlayer(inputs,in_size,out_size,activation_function=None):
	#Add a layer of linear model; default activation function is None
	
	weight = tf.Variable(tf.random_normal([in_size,out_size]))
	bias = tf.Variable(tf.zero[1, out_size] + 0.1) 
	# bias is NOT recommended to be default zero
	Wx_plus_b = tf.matmul(inputs, weight) + bias
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs


loss = tf.

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) # IMPORTANT
	sess.run()