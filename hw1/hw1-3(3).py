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
#np.random.shuffle(y_train)
x_test = mnist.test.images
y_test = mnist.test.labels

w_variable ={'w_fc1' : tf.Variable(tf.random_normal([784,1024],stddev=0.01),name='w1'),
             'w_fc2' : tf.Variable(tf.random_normal([1024,625],stddev=0.01),name='w2'),
             'w_fc3' : tf.Variable(tf.random_normal([625,10],stddev=0.01),name='w3')}

b_variable = {'b_fc1' : tf.Variable(tf.random_normal([1024],stddev=0.01),name='w1'),
			    'b_fc2' : tf.Variable(tf.random_normal([625],stddev=0.01),name='w2'),
			    'b_fc3' : tf.Variable(tf.random_normal([10],stddev=0.01),name='w3')}
l = [i for i in range(1024)]
s = ['w_fc1','w_fc2','w_fc3']
d = ['b_fc1','b_fc2','b_fc3']


x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])
drop_prob = tf.placeholder(tf.float32)

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def main(w_variable,b_variable):
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
	para = get_num_params()
	return fc3,output,para




def get_batches(x,y,batch_size=200):
	temp = np.arange(len(x))
	np.random.shuffle(temp)
	x = x[temp]
	y = y[temp]
	n_batches = len(x)//batch_size
	x,y = x[:n_batches*batch_size],y[:n_batches*batch_size]
	for ii in range(0,len(x),batch_size):
		yield x[ii:ii+batch_size], y[ii:ii+batch_size]


name = ['w_fc1','w_fc2','w_fc3','b_fc1','b_fc2','b_fc3']
temp = np.load('w_in.npy')
pp = np.linspace(-1,2,25)


loss_train = []
acc_train = []

loss_test = []
acc_test = []


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
init = tf.global_variables_initializer()
for j in temp : 
	sess.run(init)
	for i in range(6):
		if i <3:
			tf.assign(w_variable[name[i]],j[i])
		else : 
			tf.assign(b_variable[name[i]],j[i])
	epoch_loss = []
	epoch_acc = []
	for e , r in get_batches(x_train,y_train,batch_size=128):
		feed= {x:e,y:r,drop_prob:1}
		loss,acc = sess.run([cost,accuracy],feed_dict=feed)
		epoch_loss.append(loss)
		epoch_acc.append(acc)
	loss_train.append(np.mean(epoch_loss))
	acc_train.append(np.mean(epoch_acc))

	epoch_loss = []
	epoch_acc = []
	for e , r in get_batches(x_test,y_test,batch_size=128):
		feed= {x:e,y:r,drop_prob:1}
		loss,acc = sess.run([cost,accuracy],feed_dict=feed)
		epoch_loss.append(loss)
		epoch_acc.append(acc)

	loss_test.append(np.mean(epoch_loss))
	acc_test.append(np.mean(epoch_acc))


plt.plot(pp,loss_test)
plt.plot(pp,loss_train)
plt.legend(['test','train'],loc='upper right')
plt.show()



plt.plot(pp,acc_test)
plt.plot(pp,acc_train)
plt.legend(['test','train'],loc='upper right')
plt.show()