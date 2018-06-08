# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
x_train = (mnist.train.images)
y_train = (mnist.train.labels)
x_test = (mnist.test.images)
y_test = (mnist.test.labels)


w_variable = { 'w_fc1':tf.random_normal([784,1024],stddev=0.01),
               'w_fc2':tf.random_normal([1024,625],stddev=0.01),
               'w_fc3':tf.random_normal([625,10],stddev=0.01)}

b_variable = {'b_fc1':tf.random_normal([1024],stddev=0.01),
              'b_fc2':tf.random_normal([625],stddev=0.01),
              'b_fc3':tf.random_normal([10],stddev=0.01)}


x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])
drop_prob = tf.placeholder(tf.float32)


temp_1 = tf.reshape(w_variable['w_fc1'],[1024*784,1])
temp_2 = tf.reshape(w_variable['w_fc2'],[1024*625,1])
temp_3 = tf.reshape(w_variable['w_fc3'],[625*10,1])

temp_4 = tf.reshape(b_variable['b_fc1'],[1024,1])
temp_5 = tf.reshape(b_variable['b_fc2'],[625,1])
temp_6 = tf.reshape(b_variable['b_fc3'],[10,1])


parameters = tf.Variable(tf.concat([temp_1,temp_4,temp_2,temp_5,temp_3,temp_6],0))





with tf.name_scope('fc_1'):
    idx_from = 0
    layer1_w = tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[1024*784,1]),[784,1024])
    idx_from += (1024*784)
    layer1_b = tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[1024,1]),[1024])
    layer_1 = tf.add(tf.matmul(x,layer1_w),layer1_b)
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1,drop_prob)

with tf.name_scope('fc_2'):
    idx_from += (1024)
    layer2_w = tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[1024*625,1]),[1024,625])
    idx_from += (1024*625)
    layer2_b = tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[625,1]),[625])
    layer_2 = tf.add(tf.matmul(layer_1,layer2_w),layer2_b)
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2,drop_prob)
    
with tf.name_scope('output_layer'):
    idx_from += 625
    layer3_w = tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[625*10,1]),[625,10])
    idx_from += (625*10)
    layer3_b = tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[10,1]),[10])
    layer_o = tf.add(tf.matmul(layer_2,layer3_w),layer3_b)
    prediction = tf.nn.softmax(layer_o)


with tf.name_scope('optimization'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=layer_o))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.name_scope('prediction'):
    temp_1 = tf.arg_max(y,1)
    temp_2 = tf.arg_max(prediction,1)
    correct_pred = tf.equal(temp_1,temp_2)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope('gradients'):
    tvars = tf.trainable_variables()
    grad = tf.gradients(cost,tvars)[0]
    nor = tf.norm(grad,2)

    
def get_batches(x, y, batch_size=200):
    temp = np.arange(len(x))
    np.random.shuffle(temp)
    x = x[temp]
    y = y[temp]
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=8)
is_train = True
loss_trace =[]
grad_trace = []
if is_train :
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(700):
        epoch_loss = []
        #epoch_acc = []
        for e,r in get_batches(x_train,y_train,batch_size=200):
            feed = {x:e,y:r,drop_prob:0.75}
            loss , _ = sess.run([cost,optimizer],feed_dict=feed)
            epoch_loss.append(loss)
        #if i %3 == 0 :
            #saver.save(sess, 'checkpoints/model_'+str(i)+'.ckpt')
        nor_1,_ = sess.run([nor,cost],feed_dict={x:x_train,y:y_train,drop_prob:1}) 
        grad_trace.append(nor_1)
        temp = np.mean(epoch_loss)
        loss_trace.append(temp)


iteration = [i for i in range(700)]
ig1, ax1 = plt.subplots()
ax1.plot(iteration,loss_trace)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()


fig1, ax2 = plt.subplots()
ax2.plot(iteration,grad_trace)
ax2.set_xscale('log')
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()
