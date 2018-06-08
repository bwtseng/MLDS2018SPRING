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
	return fc3,output




def get_batches(x,y,batch_size=200):
	temp = np.arange(len(x))
	np.random.shuffle(temp)
	x = x[temp]
	y = y[temp]
	n_batches = len(x)//batch_size
	x,y = x[:n_batches*batch_size],y[:n_batches*batch_size]
	for ii in range(0,len(x),batch_size):
		yield x[ii:ii+batch_size], y[ii:ii+batch_size]

tt = []
para_trace = []

loss_test_record = []
acc_test_record = []

loss_train_record = []
acc_train_record = []

sen_train_record = []
sen_test_record = []

cc = np.linspace(64,1024,50).astype(np.int32)
#l = [i for i in range(1024)]
for i in range(cc):
	if i != 0 :
		temp_3 = random.sample(l,2)
		w_variable['w_fc1'] = tf.Variable(tf.random_normal([784,temp_3[0]],stddev=0.01),name='w1')
		w_variable['w_fc2'] = tf.Variable(tf.random_normal([temp_3[0],temp_3[1]],stddev=0.01),name='w2')
		w_variable['w_fc3'] = tf.Variable(tf.random_normal([temp_3[1],10],stddev=0.01),name='w3')
		b_variable['b_fc1'] = tf.Variable(tf.random_normal([temp_3[0]],stddev=0.01),name='b1')
		b_variable['b_fc2'] = tf.Variable(tf.random_normal([temp_3[1]],stddev=0.01),name='b2')
		b_variable['b_fc3'] = tf.Variable(tf.random_normal([10],stddev=0.01),name='b3')
	fc3 , output= main(w_variable,b_variable)

	with tf.name_scope('optimization'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=fc3))
		optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

	with tf.name_scope('accuracy'):
		temp_1 = tf.argmax(y,1)
		temp_2 = tf.argmax(output,1)
		correct_pred = tf.equal(temp_1,temp_2)
		accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

	with tf.name_scope('grad'):
		grad = tf.gradients(cost,x)
		norm = tf.norm(grad)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	is_train = True 
	loss_trace = []
	acc_trace = []
	if is_train :
		init = tf.global_variables_initializer()
		sess.run(init)
		for j in range(25):
			epoch_loss = []
			for e , r in get_batches(x_train,y_train,batch_size=256):
				feed= {x:e,y:r,drop_prob:0.5}
				loss,_ = sess.run([cost,optimizer],feed_dict=feed)
				epoch_loss.append(loss)
			loss_trace.append(np.mean(epoch_loss))
		epoch_loss = []
		epoch_acc = []
		epoch_sensi = []
		for e , r in get_batches(x_test,y_test,batch_size=256):
			feed= {x:e,y:r,drop_prob:1}
			loss= sess.run(cost,feed_dict=feed)
			acc_1 = sess.run(accuracy,feed_dict=feed)
			sens  = sess.run(norm,feed_dict=feed)
			epoch_sensi.append(sens)
			epoch_loss.append(loss)
			epoch_acc.append(acc_1)
		sen_test_record.append(np.mean(epoch_sensi))
		loss_test_record.append(np.mean(epoch_loss))
		acc_test_record.append(np.mean(epoch_acc))

		epoch_loss = []
		epoch_acc = []
		epoch_sensi = []

		for e , r in get_batches(x_train,y_train,batch_size=256):
			feed= {x:e,y:r,drop_prob:1}
			loss= sess.run(cost,feed_dict=feed)
			acc_1 = sess.run(accuracy,feed_dict=feed)
			sens = sess.run(norm,feed_dict=feed)
			epoch_loss.append(loss)
			epoch_acc.append(acc_1)
			epoch_sensi.append(sens)
		sen_train_record.append(np.mean(epoch_sensi))
		loss_train_record.append(np.mean(epoch_loss))
		acc_train_record.append(np.mean(epoch_acc))

	sess.close()


ax1 = fig.add_subplot(111)
ax1.plot(cc,loss_train_record,color='b',label='train')
ax1.plot(cc,loss_test_record,'k--',color='r',label='test')
legend = ax1.legend(loc='upper right',shadow=True)
ax1.set_ylabel('cross_entropy_loss')

ax2 = ax1.twinx()
ax2.plot(cc,sen_test_record,color='g',label='sensitive')
ax2.set_ylabel('sensitive')
ax2.set_xlabel('batch_size')
legend = ax2.legend(loc='right', shadow=True, fontsize='x-large')
plt.show()


ax1 = fig.add_subplot(111)
ax1.plot(cc,acc_train_record,color='b',label='train')
ax1.plot(cc,acc_test_record,'k--',color='r',label='test')
legend = ax1.legend(loc='upper right',shadow=True)
ax1.set_ylabel('cross_entropy_loss')

ax2 = ax1.twinx()
ax2.plot(cc,sen_test_record,color='g',label='sensitive')
ax2.set_ylabel('sensitive')
ax2.set_xlabel('batch_size')
legend = ax2.legend(loc='right', shadow=True, fontsize='x-large')
plt.show()