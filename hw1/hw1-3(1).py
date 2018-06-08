import tensorflow as tf 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
from functools import reduce
from operator import mul

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
np.random.shuffle(y_train)
x_test = mnist.test.images
y_test = mnist.test.labels



w_variable ={'w_fc1' : tf.Variable(tf.random_normal([784,1024],stddev=0.01),name='w1'),
             'w_fc2' : tf.Variable(tf.random_normal([1024,625],stddev=0.01),name='w2'),
             'w_fc3' : tf.Variable(tf.random_normal([625,10],stddev=0.01),name='w3')}

b_variable = {'b_fc1' : tf.Variable(tf.random_normal([1024],stddev=0.01),name='w1'),
			    'b_fc2' : tf.Variable(tf.random_normal([625],stddev=0.01),name='w2'),
			    'b_fc3' : tf.Variable(tf.random_normal([10],stddev=0.01),name='w3')}


x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])
drop_prob = tf.placeholder(tf.float32)

with tf.name_scope('fc_1'):
	fc1 = tf.add(tf.matmul(x,w_variable['w_fc1']),b_variable['b_fc1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1,drop_prob)

with tf.name_scope('fc_2'):
	fc2 = tf.add(tf.matmul(fc1,w_variable['w_fc2']),b_variable['b_fc2'])
	fc2 = tf.nn.relu(fc2)
	fc2 = tf.nn.dropout(fc2,drop_prob)

with tf.name_scope('output_layer'):
	fc3 = tf.add(tf.matmul(fc2,w_variable['w_fc3']),b_variable['b_fc3'])
	output = tf.nn.softmax(fc3)


with tf.name_scope('optimization'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=fc3))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


with tf.name_scope('accuracy'):
	temp_1 = tf.argmax(y,1)
	temp_2 = tf.argmax(output,1)
	correct_pred = tf.equal(temp_1,temp_2)
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


loss_trace = []
test_trace = []
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

init = tf.global_variables_initializer()
sess.run(init)
for i in range(4000):
	epoch_loss = []
	for e , r in get_batches(x_train,y_train,batch_size=128):
		feed= {x:e,y:r,drop_prob:0.5}
		loss,_ = sess.run([cost,optimizer],feed_dict=feed)
		epoch_loss.append(loss)
	loss_trace.append(np.mean(epoch_loss))
	epoch_loss = []
	for e , r in get_batches(x_test,y_test,batch_size=128):
		feed= {x:e,y:r,drop_prob:1}
		loss= sess.run(cost,feed_dict=feed)
		epoch_loss.append(loss)
	loss_t = test_trace.append(np.mean(epoch_loss))

iteration = [i for i in range(4000)]
plt.plot(iteration,loss_trace)
plt.plot(iteration,test_trace)
plt.show()