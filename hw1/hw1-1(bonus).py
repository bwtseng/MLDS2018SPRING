import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import reduce
from operator import mul
import random


#model 1
#####################################################
f = np.linspace(0, 1, 10000) 
v= np.sinc(5*f)
#v = np.sign(np.sin(5*np.pi*f))
f = f.reshape(10000,1)
v = v.reshape(10000,1)


plt.plot(f,v)
plt.xlim(0,1)
plt.ylim(-2,2)
plt.xticks([0,0.2,0.4,0.6,0.8,1])
plt.yticks([-2,-1,0,1,2])
plt.show()

t = np.linspace(0,1,300)
g  = np.sign(np.sin(5*np.pi*t))
t = t.reshape(300,1)
g = g.reshape(300,1)

variable_w = {'layer1':tf.Variable(tf.random_normal([1,5],stddev=1)),
              'layer2':tf.Variable(tf.random_normal([5,10],stddev=1)),
              'layer3':tf.Variable(tf.random_normal([10,10],stddev=1)),
              'layer4':tf.Variable(tf.random_normal([10,10],stddev=1)),\
              'layer5':tf.Variable(tf.random_normal([10,10],stddev=1)),\
              'layer6':tf.Variable(tf.random_normal([10,10],stddev=1)),\
              'layer7':tf.Variable(tf.random_normal([10,5],stddev=1)),\
              'layer8':tf.Variable(tf.random_normal([5,1],stddev=1))}



variable_b = {'layer1':tf.Variable(tf.constant(0.0,shape=[5])),
              'layer2':tf.Variable(tf.constant(0.0,shape=[10])),
              'layer3':tf.Variable(tf.constant(0.0,shape=[10])),
              'layer4':tf.Variable(tf.constant(0.0,shape=[10])),\
              'layer5':tf.Variable(tf.constant(0.0,shape=[10])),\
              'layer6':tf.Variable(tf.constant(0.0,shape=[10])),\
              'layer7':tf.Variable(tf.constant(0.0,shape=[5])),\
              'layer8':tf.Variable(tf.constant(0.0,shape=[1]))}


x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])


with tf.name_scope('Dense_1'):
    layer1 = tf.add(tf.matmul(x,variable_w['layer1']),variable_b['layer1'])
    layer1_a = tf.nn.relu(layer1)

with tf.name_scope('Dense_2'):
    layer2 = tf.add(tf.matmul(layer1_a,variable_w['layer2']),variable_b['layer2'])
    layer2_a = tf.nn.relu(layer2)


with tf.name_scope('Dense_3'):
    layer3 = tf.add(tf.matmul(layer2_a,variable_w['layer3']),variable_b['layer3'])
    layer3_a = tf.nn.relu(layer3)
    
with tf.name_scope('Dense_4'):
    layer4 = tf.add(tf.matmul(layer3_a,variable_w['layer4']),variable_b['layer4'])
    layer4_a = tf.nn.relu(layer4)
    
with tf.name_scope('Dense_5'):
    layer5 = tf.add(tf.matmul(layer4_a,variable_w['layer5']),variable_b['layer5'])
    layer5_a = tf.nn.relu(layer5)

with tf.name_scope('Dense_6'):
    layer6 = tf.add(tf.matmul(layer5_a,variable_w['layer6']),variable_b['layer6'])
    layer6_a = tf.nn.relu(layer6)
    
with tf.name_scope('Dense_7'):
    layer7 = tf.add(tf.matmul(layer6_a,variable_w['layer7']),variable_b['layer7'])
    layer7_a = tf.nn.relu(layer7)
    
with tf.name_scope('output_layer'):
    layer8 = tf.add(tf.matmul(layer7_a,variable_w['layer8']),variable_b['layer8'])

with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(y,layer8)
    tf.summary.histogram('cost',cost)


with tf.name_scope('predictions'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
sess= tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())


def get_batches(x, y, batch_size=200):
    temp = np.arange(len(x))
    np.random.shuffle(temp)
    x = x[temp]
    y = y[temp]
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]





epochs=20000
loss_trace = []
#accuracy = []
for i in range(epochs):
    epoch_loss=[]
    for (e,r) in get_batches(f,v):
        feed = {x:e , y:r}
        loss,_= sess.run([cost,optimizer],feed_dict=feed)
        epoch_loss.append(loss)
    #loss = sess.run(cost,feed_dict={x:f,y:v})
    #if i %1000 == 0 : 
        #feed = {x:t,y:g}
        #loss,_= sess.run([cost,optimizer],feed_dict=feed)
        #print('loss of {}th epochs is {}'.format(i,loss))
    temp = np.mean(epoch_loss)
    loss_trace.append(loss)

feed = {x:f}
out= sess.run(layer8,feed_dict=feed)


iteration = [i for i in range(20000)]

fig1, ax1 = plt.subplots()
ax1.plot(iteration,loss_trace)
ax1.set_yscale('log')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()


fig1, ax2 = plt.subplots()
ax2.plot(f,out)
ax2.set_xscale('log')
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()



#model 2
#####################################################
'''
f = np.linspace(0, 1, 10000) 
v= np.sinc(5*f)
#v = np.sign(np.sin(5*np.pi*f))
f = f.reshape(10000,1)
v = v.reshape(10000,1)


plt.plot(f,v)
plt.xlim(0,1)
plt.ylim(-2,2)
plt.xticks([0,0.2,0.4,0.6,0.8,1])
plt.yticks([-2,-1,0,1,2])
plt.show()

t = np.linspace(0,1,300)
g  = np.sign(np.sin(5*np.pi*t))
t = t.reshape(300,1)
g = g.reshape(300,1)

variable_w = {'layer1':tf.Variable(tf.random_normal([1,10],stddev=1)),
              'layer2':tf.Variable(tf.random_normal([10,18],stddev=1)),
              'layer3':tf.Variable(tf.random_normal([18,15],stddev=1)),
              'layer4':tf.Variable(tf.random_normal([15,4],stddev=1)),\
              'layer5':tf.Variable(tf.random_normal([4,1],stddev=1))}





variable_b = {'layer1':tf.Variable(tf.constant(0.0,shape=[10])),
              'layer2':tf.Variable(tf.constant(0.0,shape=[18])),
              'layer3':tf.Variable(tf.constant(0.0,shape=[15])),
              'layer4':tf.Variable(tf.constant(0.0,shape=[4])),\
              'layer5':tf.Variable(tf.constant(0.0,shape=[1]))}



x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])


with tf.name_scope('Dense_1'):
    layer1 = tf.add(tf.matmul(x,variable_w['layer1']),variable_b['layer1'])
    layer1_a = tf.nn.relu(layer1)

with tf.name_scope('Dense_2'):
    layer2 = tf.add(tf.matmul(layer1_a,variable_w['layer2']),variable_b['layer2'])
    layer2_a = tf.nn.relu(layer2)


with tf.name_scope('Dense_3'):
    layer3 = tf.add(tf.matmul(layer2_a,variable_w['layer3']),variable_b['layer3'])
    layer3_a = tf.nn.relu(layer3)
    
with tf.name_scope('Dense_4'):
    layer4 = tf.add(tf.matmul(layer3_a,variable_w['layer4']),variable_b['layer4'])
    layer4_a = tf.nn.relu(layer4)
    
with tf.name_scope('Dense_5'):
    layer5 = tf.add(tf.matmul(layer4_a,variable_w['layer5']),variable_b['layer5'])


with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(y,layer5)

with tf.name_scope('predictions'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


def get_batches(x, y, batch_size=200):
    temp = np.arange(len(x))
    np.random.shuffle(temp)
    x = x[temp]
    y = y[temp]
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
sess= tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())



epochs=20000
loss_trace = []
#accuracy = []
for i in range(epochs):
    epoch_loss=[]
    for (e,r) in get_batches(f,v):
        feed = {x:e , y:r}
        loss,_= sess.run([cost,optimizer],feed_dict=feed)
        epoch_loss.append(loss)
    #loss = sess.run(cost,feed_dict={x:f,y:v})
    #if i %1000 == 0 : 
        #feed = {x:t,y:g}
        #loss,_= sess.run([cost,optimizer],feed_dict=feed)
        #print('loss of {}th epochs is {}'.format(i,loss))
    temp = np.mean(epoch_loss)
    loss_trace.append(loss)

feed = {x:f}
out= sess.run(layer8,feed_dict=feed)


iteration = [i for i in range(20000)]

fig1, ax1 = plt.subplots()
ax1.plot(iteration,loss_trace)
ax1.set_yscale('log')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()


fig1, ax2 = plt.subplots()
ax2.plot(f,out)
ax2.set_xscale('log')
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()



'''

