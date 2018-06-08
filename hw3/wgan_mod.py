import numpy as np
import pandas as pd 
import tensorflow as tf
import os 
import tensorflow.contrib.layers as ly
import matplotlib.pyplot as plt 
import time
from skimage import io 
import random
from scipy.misc import imread, imresize , imrotate
import cv2
import sys
import pickle

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]='0'


file_path = 'faces'
lam = 10.

def load_data(file_path):
    temp = len(os.listdir(file_path))
    image_list = []
    for i in range(temp) : 
        kk = os.path.join(file_path,str(i)+'.jpg')
        te = imread(kk)
        te = (te /127.5)-1
        #te = te /255 
        te = cv2.resize(te, (64, 64), interpolation=cv2.INTER_CUBIC)
        image_list.append(te)
    return image_list

#image_list = load_data(file_path)[1:]

'''
def load_save(file_path):
    temp = os.listdir(file_path)
    image_list = []
    for i in temp :
        kk = os.path.join(file_path,i)
        te = Image.open(kk)
        te_1= img.rotate(10)
        te_2 = img.rotate(-10)
'''        

hair_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
                 'green hair', 'red hair', 'purple hair', 'pink hair',
                 'blue hair', 'black hair', 'brown hair', 'blonde hair']
eye_dict = ['gray eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']   
count = 0
hair_d = {}
for i in hair_dict:
    hair_d[i] = count
    count += 1

count = 0
eye_d = {}
for i in eye_dict:
    eye_d[i] = count
    count += 1

emb_h = np.zeros((len(hair_d),len(hair_d)))
for i in range(len(hair_d)):
    emb_h[i][i] = 1 


emb_e = np.zeros((len(eye_d),len(eye_d)))
for i in range(len(eye_d)):
    emb_e[i][i] = 1 


tt = []
a = open('testing_tags.txt','r')
for line in a: 
    pair = line.strip('\n').split(',')[1]
    feature_list = pair.split(' ') 
    tt.append(feature_list[0]+' '+feature_list[1]+','+ feature_list[2] +' '+ feature_list[3])

f_h = []
f_e = []

for i in tt : 
    f_h.append(hair_d[i.split(',')[0]])
    f_e.append(eye_d[i.split(',')[1]])


def encoding_matrix():
    hair_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
                 'green hair', 'red hair', 'purple hair', 'pink hair',
                 'blue hair', 'black hair', 'brown hair', 'blonde hair']
    eye_dict = ['gray eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']    
    hair_d = {}
    count = 0
    for i in hair_dict:
        hair_d[i] = count
        count += 1
    count = 0
    eye_d = {}
    for i in eye_dict:
        eye_d[i] = count
        count += 1
    
    gg = pd.read_csv('tags_clean.csv').values
    index = []
    caption = []
    #w_caption = []
    count = 0
    for i in gg : 
        feature = []
	    #feature_1 = []
        temp = i[1].split('\t')
        for j in temp : 
	        qq = j.split(':')[0]
	        if qq in hair_dict:
	            feature.append(qq)
	        if qq in eye_dict:
	            feature.append(qq)
        if len(feature) == 2:
            if feature[0] in hair_dict and feature[1] in hair_dict : 
                    count+=1
                    continue
            if feature[0] in eye_dict and feature[1] in eye_dict : 
                    count+=1
                    continue
            if feature[0].split(' ')[1] == 'hair':
                caption.append([feature[0] +','+feature[1]])
                index.append(count)
                count+=1
            else :
                caption.append([feature[1] +','+feature[0]])
                index.append(count)       
                count+=1
        else:
            count+=1
    image_l = []
    for i in index : 
        image_l.append(image_list[i])
    
    #image_w = []
    #for i in range(len(image_list)):
    #    if i not in index : 
    #        image_w.append(image_list[i])
    
    feature_h = []
    feature_e = []
    count =0
    for i in caption : 
        feature_h.append(hair_d[i[0].split(',')[0]])
        feature_e.append(eye_d[i[0].split(',')[1]])
        count +=1
    
    return feature_h , feature_e , hair_d ,eye_d , image_l 

#feature_h , feature_e , hair_d ,eye_d ,image_list  = encoding_matrix()



#flip_list = [] 
#for i in image_list : 
#    flip_list.append(i[:,::-1])
#image_list = image_list + flip_list
#tes_1 = feature_h.copy()
#tes_2 = feature_e.copy()

#feature_h = feature_h + tes_1
#feature_e = feature_e + tes_2

### copy there images to approximate the real image !!! 







def make_wrong(feature_h , feature_e):
    w_h = []
    le = max(feature_h)+1
    for i in feature_h : 
        qq = [j for j in range(le)]
        qq.remove(i)
        cc = random.sample(qq,1)[0]
        w_h.append(cc)
    w_e = []
    le = max(feature_e)+1
    for i in feature_e:
        qq = [j for j in range(le)]
        qq.remove(i)
        cc = random.sample(qq,1)[0]
        w_e.append(cc)		
    return w_h ,w_e

#w_h , w_e = make_wrong(feature_h,feature_e)


#extra_path ='extra_data/images'

#ex_image_list = load_data(extra_path)

#extra_tag = 'extra_data/tags.csv'


#cc[0].split(' ')[0]+' '+cc[0].split(' ')[1]+','+cc[0].split(' ')[2]+' '+cc[0].split(' ')[3]

def all_tags(file_path,hair_d,hair_h):
    ex_tags = []
    temp = pd.read_csv(file_path)
    cc = []
    cc.append(list(temp.columns)[1])
    cc[0] =  cc[0].split(' ')[0]+' '+cc[0].split(' ')[1]+','+cc[0].split(' ')[2]+' '+cc[0].split(' ')[3]
    ex_tags.append(cc)
    
    for i in temp.values:
        cc = []
        cc.append(i[1])
        cc[0] =  cc[0].split(' ')[0]+' '+cc[0].split(' ')[1]+','+cc[0].split(' ')[2]+' '+cc[0].split(' ')[3]
        ex_tags.append(cc)
        #word = cc[0].split(' ')[0]+' ' + cc[0].split(' ')[1]
        
        
    #w_caption  = []        
    #for i in ex_tags :
    #    w_c = []
    #    temp_1 = i[0].split(',')[0]
    #    temp_2 = i[0].split(',')[1]
    #    if temp_1 in hair_dict:
            #word = cc[0].split(' ')[0]+' ' + cc[0].split(' ')[1]
    #        copy_h = hair_dict.copy()
    #        copy_h.remove(temp_1)
    #        w_c.append(random.sample(copy_h,1)[0])
                
    #    if temp_2 in eye_dict :
    #        copy_h = eye_dict.copy()
    #        copy_h.remove(temp_2)
    #        w_c.append(random.sample(copy_h,1)[0])
    #    w_caption.append(w_c)
        
    #w_caption.append(w_caption[0])

    ex_h = [] 
    ex_e = []
    for i in ex_tags : 
    	word_1 = i[0].split(',')[0]
    	word_2 = i[0].split(',')[1]
    	ex_h.append(hair_d[word_1])
    	ex_e.append(eye_d[word_2])

    #w_h = [] 
    #w_e = []
    #for i in w_caption: 
    	#word_1 = i.split(' ')[0]+' ' + i.split(' ')[1]
    	#word_2 = i.split(' ')[2]+' ' + i.split(' ')[3]
    #	w_h.append(hair_d[i[0]])
    #	w_e.append(eye_d[i[1]])

    #ex_w_h = ex_h + w_h 
    #ex_w_e = ex_e + w_e 
    
    return ex_h , ex_e 

#ex_h , ex_e = all_tags(extra_tag,hair_d,eye_d)




#image_list = image_list + ex_image_list 

#w_h_caption = w_h + w_h_1 

#w_e_caption = w_e + w_e_1





#real_h_caption = feature_h + ex_h

#real_e_caption = feature_e + ex_e




#temp_1 = pd.read_csv('la_v2.csv')
#eye_c = temp_1[['eye']].values
#hair_c = temp_1[['hair']].values

def preprocess_own():
    e_fe = []
    h_fe = []
    p_ind = []
    for i in range(len(eye_c)):
        if eye_c[i][0] != 'None':
            if hair_c[i][0] != 'None':
                e_fe.append(eye_d[eye_c[i][0]])
                h_fe.append(hair_d[hair_c[i][0]])
                p_ind.append(i)
    #w_h = []
    #le = max(e_fe)+1
    #for i in e_fe : 
    #    qq = [j for j in range(le)]
    #    qq.remove(i)
    #    cc = random.sample(qq,1)[0]
    #    w_h.append(cc)
    #w_e = []
    #le = max(h_fe)+1
    #for i in h_fe:
    #    qq = [j for j in range(le)]
    #    qq.remove(i)
    #    cc = random.sample(qq,1)[0]
    #    w_e.append(cc)	
    img = []
    temp_1 = load_data('fa')
    temp = os.listdir('fa')
    for i in range(len(temp)):
        if i in p_ind : 
            img.append(temp_1[i])
    
    return e_fe , h_fe ,img 

#my_1 ,my_2,my_img= preprocess_own()


#w_h_caption = w_h_caption + my_3
#w_e_caption = w_e_caption + my_4 

#real_e_caption = real_e_caption + my_1
#real_h_caption = real_h_caption + my_2


#image_list = image_list + my_img

#wrong_img = image_list.copy()
#np.random.shuffle(wrong_img)

tf.reset_default_graph()

lam = 10.

noise = tf.placeholder(tf.float64, shape=[None , 100])
real_image = tf.placeholder(tf.float32, shape=[None , 64, 64, 3 ])
wrong_image = tf.placeholder(tf.float32,shape= [None , 64 , 64 , 3])

real_hair = tf.placeholder(tf.int32, shape=[None])
real_eye = tf.placeholder(tf.int32, shape=[None])

real_h = tf.nn.embedding_lookup(emb_h,real_hair)
real_e = tf.nn.embedding_lookup(emb_e,real_eye)

r_text_v = tf.concat([real_h,real_e],1)

batch_size = 64
channel = 3 
def generator_conv(z,captio):
    text_embedding = ly.fully_connected(captio,256,activation_fn=tf.nn.relu)
    train = tf.concat([z,text_embedding],1)
    train = tf.cast(train,tf.float32)
    train = ly.fully_connected(
        train, 4* 4 *512 , activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, (-1, 4, 4,512))
    train = ly.conv2d_transpose(train, 256, 5, stride=2,normalizer_fn=ly.batch_norm,
                                activation_fn=tf.nn.leaky_relu, padding='SAME')
    train = ly.conv2d_transpose(train, 128, 5, stride=2,
                                activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm, padding='SAME')
    train = ly.conv2d_transpose(train, 64, 5 , stride=2,
                                activation_fn=tf.nn.leaky_relu, normalizer_fn=ly.batch_norm, padding='SAME')
    train = ly.conv2d_transpose(train, 32 , 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
    train = ly.conv2d_transpose(train, channel, 5, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME')
    print(train.name)
    return train

with tf.variable_scope('generator_conv') as scope : 
    img = generator_conv(noise,r_text_v)

tf.summary.image('synthetic_image',img)
#tf.layers.dense

def dis_conv(img,caption, reuse=False):
    with tf.variable_scope('dis_conv') as scope:
        if reuse:
            scope.reuse_variables()
        size = 64
        img = ly.conv2d(img, num_outputs=size, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu)
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu)
        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=5,padding='SAME',
                        stride=2, activation_fn=tf.nn.leaky_relu)
        ### text embedding
        caption = tf.cast(caption,tf.float32)

        reduced_text_embeddings = ly.fully_connected(caption,256,activation_fn=tf.nn.relu)
        
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
        
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
        
        tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1])
        
        conca = tf.concat([img,tiled_embeddings],3)
        
        fin = ly.conv2d(conca, num_outputs=size * 8, kernel_size=1,padding='SAME',
                        stride=1, activation_fn=tf.nn.leaky_relu)     
        #fin = ly.conv2d(conca, 1, kernel_size=8,padding='valid',stride=1, activation_fn=None)     
         
        #fin = tf.squeeze(fin,[1,2,3])  ##原本是256哦
        source_logit = ly.fully_connected(tf.reshape(
            fin, [batch_size,4*4*512]), 1, activation_fn=None)
    return source_logit

real_image_logits = dis_conv(real_image, r_text_v)

#wrong_image_logits = dis_conv(real_image, w_text_v, reuse = True)

fake_image_logits = dis_conv(img, r_text_v, reuse = True)

wrong_caption_logits = dis_conv(wrong_image , r_text_v , reuse = True)

#tf.summary.scalar('r_logits',real_image_logits)

d_loss = (tf.reduce_mean(fake_image_logits) + tf.reduce_mean(wrong_caption_logits))/2 - tf.reduce_mean(real_image_logits)

#d_loss = tf.reduce_mean(fake_image_logits+wrong_caption_logits-real_image_logits)

tf.summary.scalar('d_loss',d_loss)


####也可以換成WGAN的方式!!!!
#g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_image_logits,labels= tf.ones_like(fake_image_logits)))

g_loss = tf.reduce_mean(-fake_image_logits)

tf.summary.scalar('g_loss',g_loss)

#d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = real_image_logits, labels = tf.ones_like(real_image_logits)))
#d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = wrong_image_logits, labels = tf.zeros_like(wrong_image_logits)))
#d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_image_logits,labels =  tf.zeros_like(fake_image_logits)))

#d_loss = d_loss1 + d_loss2 + d_loss3



#clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

#maybe this part have the problem , change to wegihts clippgin 


mode='gp'
if mode is 'gp':
    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
    alpha = alpha_dist.sample((64, 1, 1, 1))
    interpolated = real_image + alpha*(img - real_image)
    inte_logit = dis_conv(interpolated, r_text_v , reuse=True)
    gradients = tf.gradients(inte_logit, [interpolated,])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
    d_loss += lam*gradient_penalty




theta_g = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_conv')
    
theta_c = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis_conv')


g_opt = tf.train.AdamOptimizer(0.0005).minimize(g_loss,var_list=theta_g)

d_opt = tf.train.AdamOptimizer(0.0005).minimize(d_loss,var_list=theta_c)


def next_batch(real_image,wrong_image,real_hair,real_eye,batch_size=64):
    le = len(real_image)
    epo = le//batch_size
    #temp = np.array(real_image)
    #temp_1 = np.array(caption)
    #temp_2 = np.array(wrong_image)
    #c = np.arange(le)
    #np.random.shuffle(wrong_image)
    #temp = real_image[c]
    #temp_1 = caption[c]
    #temp_2 = wrong_image[c]
    #np.random.shuffle(input_image)
    c = list(zip(real_image,wrong_image,real_hair,real_eye))
    random.shuffle(c)
    temp , temp_1 , temp_2 , temp_3 = zip(*c)
    for i in range(0,epo*batch_size,64):
        #yield np.array(real_image[i:i+64]),np.array(wrong_image[i:i+64]),np.array(real_hair[i:i+64]),np.array(real_eye[i:i+64]),np.array(wrong_hair[i:i+64]),np.array(wrong_eye[i:i+64])
        yield np.array(temp[i:i+64]),np.array(temp_1[i:i+64]),np.array(temp_2[i:i+64]), np.array(temp_3[i:i+64])
        
#tensorboard_dir = 'conditional_WGAN/'   
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
for i in range(7000):
    if i<5 or i%500 == 0 :
        citers = 100
    else : 
        citers = 10
    epoch_loss = []
    start  = time.time()
    count = 0 
    for j , k , l ,m  in next_batch(image_list,w_image,real_h_caption,real_e_caption):#next_batch(image_list,en_tag,w_image):
        if count < citers : 
            batch_z = np.random.normal(size=[batch_size,100])
            #feed_dict = {noise:batch_z,real_image:j,wrong_image:k,real_hair:l,real_eye:m,wrong_hair:n,wrong_eye:b}
            feed_dict = {noise:batch_z,real_image:j,wrong_image:k,real_hair:l,real_eye:m}
            _,loss = sess.run([d_opt,d_loss],feed_dict=feed_dict)
            count+=1 
        else : 
            break 
    #epoch_loss.append(loss)
    _,loss = sess.run([g_opt,g_loss],feed_dict=feed_dict)
    merge  = sess.run(merged,feed_dict=feed_dict)
    writer.add_summary(merge,i)
    end = time.time()
    print('{}/7000 epochs , cost {} sec'.format(i+1,end-start))
'''

'''
rs = sess.run(img,feed_dict=feed_dict)
overall = []
for i in range(8):
    temp = []
    for j in range(8):
        temp.append(rs[i * 8 + j])

    overall.append(np.concatenate(temp, axis=1))
res = np.concatenate(overall, axis=0)
res = np.squeeze(res)
res = (res+1)/2
plt.figure(figsize=[8, 8])
plt.imshow(res)
plt.show()
'''

saver.restore(sess,'v3')

with open('cwgan_no_TA.pickle', 'rb') as handle:
    no = pickle.load(handle)


#hhh = [i for i in range(12)] + [i for i in range(12)]+ [0]
#eee = [i for i in range(11)] + [i for i in range(11)]+[i for i in range(3)]

def plot(x):
    x = x - np.min(x)
    x /= np.max(x)
    x *= 255 
    x= x.astype(np.uint8)
    x = x.reshape(64,64,3)
    return x    


def save_imgs(no):
    import matplotlib.pyplot as plt

    r, c = 5, 5
    #no = np.random.normal(0, 1, (r * c, 100))

    # gen_imgs should be shape (25, 64, 64, 3)
    #gen_imgs = generator.predict(noise)
    gen_imgs = sess.run(img,feed_dict={noise:no,real_hair:f_h , real_eye:f_e})

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(plot(gen_imgs[cnt, :,:,:]))
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("samples/cgan.png")
    plt.close()


#no  = np.random.normal(0,1,size=(25,100))
save_imgs(no)