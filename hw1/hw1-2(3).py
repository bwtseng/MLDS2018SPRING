import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import reduce
from operator import mul
import random

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


variable_w = {'layer1':tf.truncated_normal([1,16],stddev=1),
              'layer2':tf.truncated_normal([16,1],stddev=1)}


variable_b = {'layer1':tf.constant(0.0,shape=[16]),
              'layer2':tf.constant(0.0,shape=[1])}

temp_1 = tf.reshape(variable_w['layer1'],[16,1])
temp_2 = tf.reshape(variable_w['layer2'],[16,1])
temp_3 = tf.reshape(variable_b['layer1'],[16,1])
temp_4 = tf.reshape(variable_b['layer2'],[1,1])
parameters = tf.Variable(tf.concat([temp_1,temp_3,temp_2,temp_4],0))

x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])


with tf.name_scope('Dense_1'):
    idx_from = 0
    layer1_w = tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[16,1]),[1,16])
    idx_from +=16
    layer1 = tf.add(tf.matmul(x,layer1_w),tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[16,1]),[16]))
    layer1_a = tf.nn.relu(layer1)

with tf.name_scope('Dense_2'):
    idx_from += 16
    layer2_w = tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[16,1]),[16,1])
    idx_from += 16
    layer2 = tf.add(tf.matmul(layer1_a,layer2_w),tf.reshape(tf.slice(parameters,begin=[idx_from,0],size=[1,1]),[1]))
    #layer2_a = tf.nn.relu(layer2)

with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(y,layer2)

with tf.name_scope('hessian'):
    tvars = tf.trainable_variables()
    dloss_dw = tf.gradients(cost,tvars )[0]
    dim,_ = dloss_dw.get_shape()
    hess = []
    for i in range(dim):
        dfx_i = tf.slice(dloss_dw, begin=[i,0] , size=[1,1])
        ddfx_i = tf.gradients(dfx_i, parameters)[0]
        hess.append(ddfx_i)
    hess = tf.squeeze(hess)


with tf.name_scope('gradients'):
    nor=tf.norm(dloss_dw,2)
    optimizer_1 = tf.train.GradientDescentOptimizer(0.01).minimize(nor)


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

loss_trace = []
mini_ration = []
          
for i in range(50):
    count=0
    feed_no = {x:f,y:v}
    epoch_loss = []
    for (e,r) in get_batches(f,v):
        feed = {x:e , y:r}
        loss,_= sess.run([cost,optimizer],feed_dict=feed)
        epoch_loss.append(loss) 

    he= sess.run(hess,feed_dict=feed_no)
    c_eig = np.linalg.eigvals(he)
    for i in c_eig :
        if i > 0:
            count+=1
    mini_ratio = count/49
    mini_ration.append(mini_ratio)    
    temp_3 = np.mean(epoch_loss)
    loss_trace.append(temp_3)

epoch=50
gradient_loss = []
for i in range(epoch):
    feed_no = {x:f,y:v}
    count = 0
    loss_t = []
    loss_g = []
    for (e,r) in get_batches(f,v):
        feed = {x:e , y:r}
        loss_1,_= sess.run([nor,optimizer_1],feed_dict=feed)
        loss= sess.run(cost,feed_dict=feed)
        loss_g.append(loss_1)
        loss_t.append(loss)

    he= sess.run(hess,feed_dict=feed_no)
    c_eig = np.linalg.eigvals(he)
    for i in c_eig :
        if i > 0:
            count+=1
    mini_ratio = count/49
    mini_ration.append(mini_ratio)
    temp_1 = np.mean(loss_t)
    temp_2 = np.mean(loss_g)
    loss_trace.append(temp_1)
    gradient_loss.append(temp_2)

###因為個我的initializer隨機性的關係，有時候這樣跑完的結果都不一定會相同###
plt.scatter(mini_ration,loss_trace,lw=5,s=5)
