# -*- coding: utf-8 -*-
from keras.layers import Conv2D,MaxPooling2D,Flatten , Dense
from keras.models import Sequential
from keras.layers.core import Dropout
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import os 
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#data collection
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
x_train = (mnist.train.images)
#x_train  = x_train.reshape(55000,28,28,1)
y_train = (mnist.train.labels)
x_test = (mnist.test.images)
#x_test = x_test.reshape(10000,28,28,1)
y_test = (mnist.test.labels)

#define model 
model = Sequential()
model.add(Dense(1024,activation='relu',input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(625,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics = ['accuracy'])

#手動執行8次，(手動改 filepath(weights_0, weights_1...., weights_7)、(acc_1.npy, acc_2.npy, ..., acc_8.npy)
filepath="checkpont_k/weights_7.{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,save_weights_only =True, mode='max')
callbacks_list = [checkpoint]
history=model.fit(x_train, y_train, validation_split=0.33,epochs=24 ,batch_size=200, callbacks=callbacks_list)
c = np.round(history.history['acc'],4)
np.save('acc_8.npy',c)


# load weight and acc
def plt_1():
    bb = list()
    cc = list()
    for i in range(8):
        for j in range(1,25,3):
            model = Sequential()
            model.add(Dense(1024,activation='relu',input_shape=(784,)))
            model.add(Dropout(0.5))
            model.add(Dense(625,activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10,activation='softmax'))
            if i==0:
                if j<10:
                    temp = os.path.join('checkpont_k/','weights.0'+str(j)+'.hdf5')
                else : 
                    temp = os.path.join('checkpont_k/','weights.'+str(j)+'.hdf5')
            else : 
                if j <10:
                    temp = os.path.join('checkpont_k/','weights_'+str(i)+'.0'+str(j)+'.hdf5')
                else : 
                    temp = os.path.join('checkpont_k/','weights_'+str(i)+'.'+str(j)+'.hdf5')
            model.load_weights(temp)
            temp_1 = model.get_weights()
            for k in range(0,6,2):
                if k ==0:
                    array_1 = temp_1[k].reshape(-1)
                    continue
                if k!=0:
                    temp_2 = temp_1[k].reshape(-1)
                array_1 = np.concatenate((array_1,temp_2),axis=0)
            bb.append(array_1)
        q = np.round(np.load('loss_'+str(i+1)+'.npy'),4)
        cc.append(list(q))
    return np.array(bb) ,cc 

#PCA
def pca(cc):
    x_emb = PCA(n_components=2,svd_solver='full',whiten=True,copy=True)     
    dd = x_emb.fit_transform(cc)
    return dd 

tt,yy = plt_1()
dd = pca(tt)

#plot picture
j=0
color = ['black','green','red','blue','orange','yellow','cyan','magenta']
for i in range(0,72,8):
    pp = dd[i:i+8]
    x = pp[:,0]
    y = pp[:,1]
    c = yy[j]
    col = color[j]
    plt.scatter(x,y,lw=0,s=0)
    texts = [plt.text(X,Y,Text,color=col) for X,Y,Text in zip(x,y,c)]
    j+=1
    