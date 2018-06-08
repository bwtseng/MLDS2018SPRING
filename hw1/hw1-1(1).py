# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from keras.datasets import cifar10
from time import time

chose = 0
if chose == 1: 
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print('===================================================')
    print('train', mnist.train.num_examples,
          ', validation', mnist.validation.num_examples, 
          ', test', mnist.test.num_examples)
    print('===================================================')
    print('train images     :', mnist.train.images.shape,
          'labels:'          , mnist.train.labels.shape)
    print('validation images: ', mnist.validation.images.shape,
          'labels: '          , mnist.validation.labels.shape)
    print('test images      :', mnist.test.images.shape,
          'labels:'          , mnist.test.labels.shape)
    print('===================================================')
    print(len(mnist.train.images[0]))
    print('===================================================')
else:
     data = input_data.read_data_sets("cifar10_data/", one_hot=True)


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')

def bias(shape):v
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], 
                          strides=[1,2,2,1], 
                          padding='SAME')

#shallow model################################################################  
    
# input layer
with tf.name_scope('Input_Layer'):
    x = tf.placeholder("float", shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])

# convolutional layer
with tf.name_scope('C1_Conv'):
    W1 = weight([5,5,1,16])
    b1 = bias([16])
    Conv1=conv2d(x_image, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)
    
#pooling
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)    

#Flatten
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C1_Pool, [-1, 3136])
    
#hidden layer
with tf.name_scope('D_Hidden_layer'):
    W3 = weight([3136, 128])
    b3 = bias([128])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3)+b3)
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, keep_prob=0.8)
    
#output layer
with tf.name_scope('Output_Layer'):
    W4 = weight([128, 10])
    b4 = bias([10])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4) +b4)
    
    
#training 
with tf.name_scope("optimizer"):
    y_label = tf.placeholder("float", shape=[None, 10], name="y_label")
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)
    
#accuracy
with tf.name_scope("evaluate_model"):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), 
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
trainEpochs = 100
batchSize = 300
totalBatchs = int(data.train.num_examples/batchSize)

loss_list1 = []; epoch_list1 = []; accuracy_list1 = []

startTime = time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        #batch_x, batch_y = mnist.train.next_batch(batchSize)
        batch_x, batch_y = data.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict={x: batch_x, 
                                       y_label: batch_y})
    loss, acc = sess.run([loss_function, accuracy], 
                         feed_dict={x: data.validation.images, 
                                    y_label: data.validation.labels})
    epoch_list1.append(epoch)
    loss_list1.append(loss)
    accuracy_list1.append(acc)
    print("Train Epoch:", '%02d' % (epoch+1), "Loss=", 
          "{:.9f}".format(loss), " Accuracy=", acc)
duration = time() - startTime
print("Train Finished takes:", duration)
        

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/CNN', sess.graph)

#deep model #################################################################################
# input layer
with tf.name_scope('Input_Layer'):
    x = tf.placeholder("float", shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])

# convolutional layer
with tf.name_scope('C1_Conv'):
    W1 = weight([5,5,1,16])
    b1 = bias([16])
    Conv1=conv2d(x_image, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)
    
#pooling
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)    

# convolutional layer
with tf.name_scope('C2_Conv'):
    W2 = weight([5,5,16,36])
    b2 = bias([36])
    Conv2=conv2d(C1_Pool, W2) + b2
    C2_Conv = tf.nn.relu(Conv2)

#pooling
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv)
    
    
#Flatten
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, [-1, 1764])
    
#hidden layer
with tf.name_scope('D_Hidden_layer'):
    W3 = weight([1764, 128])
    b3 = bias([128])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3)+b3)
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, keep_prob=0.8)
    
#output layer
with tf.name_scope('Output_Layer'):
    W4 = weight([128, 10])
    b4 = bias([10])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4) +b4)
    
    
#training 
with tf.name_scope("optimizer"):
    y_label = tf.placeholder("float", shape=[None, 10], name="y_label")
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)
    
#accuracy
with tf.name_scope("evaluate_model"):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), 
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
trainEpochs = 100
batchSize = 300
totalBatchs = int(data.train.num_examples/batchSize)
loss_list2 = []; epoch_list2 = []; accuracy_list2 = []

startTime = time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = data.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict={x: batch_x, 
                                       y_label: batch_y})
    loss, acc = sess.run([loss_function, accuracy], 
                         feed_dict={x: data.validation.images, 
                                    y_label: data.validation.labels})
    epoch_list2.append(epoch)
    loss_list2.append(loss)
    accuracy_list2.append(acc)
    print("Train Epoch:", '%02d' % (epoch+1), "Loss=", 
          "{:.9f}".format(loss), " Accuracy=", acc)
duration = time() - startTime
print("Train Finished takes:", duration)
        

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/CNN', sess.graph)

#plot acc & Loss #################################################################################
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(8,4)
plt.plot(epoch_list1, loss_list1, label = 'shallow')
plt.plot(epoch_list2, loss_list2, label = 'deep')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(8,4)
plt.plot(epoch_list1, accuracy_list1, label = 'shallow')
plt.plot(epoch_list2, accuracy_list2, label = 'deep')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
