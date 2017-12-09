import tensorflow as tf 
import numpy as np 
import os
import math
import matplotlib.pyplot as plt 



num_house = 160
np.random.seed(42)	# set the random seed
house_size = np.random.randint(low=1000, high=3500, size=num_house) 
# output is a LIST of data, with shape (160,)

np.random.seed(72)	# set the random seed
house_prize = house_size*100 + np.random.randint(low=20000, high=70000, size=num_house)

#print(np.shape(house_size))
'''
plt.plot(house_size, house_prize, 'bo') # blue point (default)
plt.ylabel('Price')
plt.xlabel('Size')
plt.show()

'''

batchsize = 5	# each five data as an input, both use for training and testing

training_num = math.floor(0.7*num_house) # 70% data used to train model
testing_num = num_house - training_num


x = tf.placeholder(tf.int32,shape=None, name='size') # shape of x and y are detemrined by the shape of feed data
y = tf.placeholder(tf.int32,shape=None, name='price')

#W = tf.Variable(1)
#bias = tf.Variable(tf.zero[, out_size] + 0.1) 
with tf.Session() as sess:
	for step in range(math.floor(training_num/batchsize)):
		size, prize = sess.run([x,y],feed_dict={x: house_size[step*batchsize:(step+1)*batchsize], y: house_prize[step*batchsize:(step+1)*batchsize]})
		print('Training data: size list {}, prize list {}'.format(size, prize))

	for step in range(math.floor(testing_num/batchsize)):
		size, prize = sess.run([x,y],feed_dict={x: house_size[step*batchsize:(step+1)*batchsize], y: house_prize[step*batchsize:(step+1)*batchsize]})
		print('Testing data: size list {}, prize list {}'.format(size, prize))

