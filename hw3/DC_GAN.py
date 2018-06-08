import numpy as np
import tensorflow as tf
import os 
import tensorflow.contrib.layers as ly
import matplotlib.pyplot as plt 
import time
from skimage import io 
import random
from scipy.misc import imread, imresize
import pickle 


#np.random.seed(0)
#tf.set_random_seed(0)

#file_path = 'faces'


def load_data(file_path):
    temp = os.listdir(file_path)
    image_list = []
    for i in temp : 
        kk = os.path.join(file_path,i)
        te = imread(kk)
        te = (te /127.5)-1
        te = cv2.resize(te, (64, 64), interpolation=cv2.INTER_CUBIC)
        image_list.append(te)
    return image_list

#image_list  = load_data(file_path)

#extra_path ='extra_data/images'

#image_list = image_list + load_data(extra_path)

#image_list = image_list +load_data('fa')

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
def plot(x):
    x = x - np.min(x)
    x /= np.max(x)
    x *= 255 
    x= x.astype(np.uint8)
    x = x.reshape(64,64,3)
    return x    


batch_size = 64

tf.reset_default_graph()

real_image = tf.placeholder(tf.float32,shape=(None,64,64,3))

noise = tf.placeholder(tf.float32,shape=(None,100))

channel = 3 
def generator_conv(z):
    train = ly.fully_connected(
        z, 4* 4 *512 , activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, (-1, 4, 4,512))
    train = ly.conv2d_transpose(train, 256, 5, stride=2,normalizer_fn=ly.batch_norm,
                                activation_fn=tf.nn.leaky_relu, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 128, 5, stride=2,normalizer_fn=ly.batch_norm,
                                activation_fn=tf.nn.leaky_relu, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 64, 5, stride=2,normalizer_fn=ly.batch_norm,
                                activation_fn=tf.nn.leaky_relu, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 32 , 5, stride=2,normalizer_fn=ly.batch_norm,
                                activation_fn=tf.nn.leaky_relu, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, channel, 5, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    print(train.name)
    return train


#tf.layers.dense

def dis_conv(img, reuse=False):
    with tf.variable_scope('dis_conv') as scope:
        if reuse:
            scope.reuse_variables()
        size = 64
        img = ly.conv2d(img, num_outputs=size, kernel_size=5,padding='SAME', normalizer_fn=ly.batch_norm,
                        stride=2, activation_fn=tf.nn.leaky_relu)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm)
        source_logit = ly.fully_connected(tf.reshape(
            img, [batch_size, 4*4*512]), 1, activation_fn=None)

    return source_logit

with tf.variable_scope('generator_conv'):
    sythetic_image = generator_conv(noise)

tf.summary.image('sythetic_image',sythetic_image)

logits_fake = dis_conv(sythetic_image,reuse=False)

logits_real = dis_conv(real_image,reuse=True)

#fake_loss = tf.nn.sigmoid_cross_entropy(logits=logits_real,labels=tf.zeros([batch_size]))


fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_fake),logits=logits_fake))

#acc,acc_op = tf.equal(labels=tf.argmax(labels, 0), predictions=tf.argmax(logits,0))

'''accuracy can append to this placeholder '''
real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real),logits=logits_real))

d_loss = fake_loss + real_loss

tf.summary.scalar('d_loss',d_loss)

g_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_fake),logits=logits_fake))

tf.summary.scalar('g_loss',g_loss)


theta_g = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_conv')
    
theta_c = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis_conv')


counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

g_opt = tf.train.AdamOptimizer(0.0001).minimize(g_loss,var_list=theta_g)
#g_opt = tf.train.GradientDescentOptimizer(0.0001).minimize(g_loss,var_list=theta_g)

counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

d_opt = tf.train.AdamOptimizer(0.0001).minimize(d_loss,var_list=theta_c)
#d_opt = tf.train.GradientDescentOptimizer(0.0001).minimize(g_loss,var_list=theta_g)


def next_batch(input_image , batch_size=64):
    le = len(input_image)
    epo = le//batch_size
    np.random.shuffle(input_image)
    for i in range(0,epo*batch_size,64):
        yield np.array(input_image[i:i+64])



#tensorboard_dir = 'DCGAN/'   
#if not os.path.exists(tensorboard_dir):
#    os.makedirs(tensorboard_dir)


sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

saver = tf.train.Saver()

#writer = tf.summary.FileWriter(tensorboard_dir)     

#writer.add_graph(sess.graph)


#merged = tf.summary.merge_all()

'''

no = np.random.normal(size=(64,100))
rs= sess.run(sythetic_image,feed_dict={noise:no})
overall = []
for i in range(8):
    temp = []
    for j in range(8):
        temp.append(plot(rs[i * 8 + j]))

    overall.append(np.concatenate(temp, axis=1))
res = np.concatenate(overall, axis=0)
res = np.squeeze(res)
#res = (res+1)/2
plt.figure(figsize=[8, 8])
plt.imshow(res)
plt.show()    


loss = []
for i in range(50000):
    if i < 25 or i%500 == 0:
        citers = 100
    else : 
        citers = 5 
    epoch_loss = []
    count = 0
    start  = time.time()
    for j in next_batch(image_list):
        if count <citers :
            feed_dict = {noise:np.random.normal(size=[batch_size,100]),real_image:j}
            _,loss = sess.run([d_opt,d_loss],feed_dict=feed_dict)
            epoch_loss.append(loss)
            count+=1
        else : 
            break
    merge,_,loss = sess.run([merged,g_opt,g_loss],feed_dict={noise:np.random.normal(size=[batch_size,100]),real_image:j})
    writer.add_summary(merge,i)
    end = time.time()
    print('{}/10000 epochs , cost {} sec'.format(i+1,end-start))
loss = []
for i in range(30):
    epoch_loss = []
    count = 0
    start  = time.time()
    for j in next_batch(image_list):
        batch_z = np.random.normal(size=[batch_size,100])
        feed_dict = {noise:batch_z,real_image:j}
        _,loss = sess.run([d_opt,d_loss],feed_dict=feed_dict)
        epoch_loss.append(loss)
        
        for _ in range(2): 
            _,loss = sess.run([g_opt,g_loss],feed_dict={noise:batch_z})
    merge  = sess.run(merged,feed_dict={noise:batch_z,real_image:j})
    writer.add_summary(merge,i)
    end = time.time()
    print('{}/300 epochs , cost {} sec'.format(i+1,end-start))

'''
saver.restore(sess,'dc_23')

with open('dc_no_20.pickle', 'rb') as handle:
    no = pickle.load(handle)



## relu 
#no = np.random.normal(0, 1, (25, 100))
def save_imgs(no):
    r, c = 5, 5
    #no = np.random.normal(0, 1, (r * c, 100))

    # gen_imgs should be shape (25, 64, 64, 3)
    #gen_imgs = generator.predict(noise)
    gen_imgs = sess.run(sythetic_image,feed_dict={noise:no})

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(plot(gen_imgs[cnt, :,:,:]))
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("samples/gan.png")
    plt.close()

#no  = np.random.normal(0,1,size=(25,100))

save_imgs(no)