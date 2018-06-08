import string
import os 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import pandas 
import json
from tensorflow.python.layers.core import Dense
from tensorflow.python.layers import core as layer_core
import random 
import time
import pickle 
import sys


ac = string.ascii_lowercase
ab= string.ascii_uppercase 
ad = ac + ab
tt = []
for i in ad : 
    tt.append(str(i))

exp = ['&','^','$','*','%']

num = [str(i) for i in range(10)]+['.','%']
def seg(te):
    bi = []
    count = 0
    another = 0
    for k in te :
        if another < count : 
            another+=1
            continue
        if k != ' ':
            if k in exp : 
                bi.append(k)
                count+=1
                another+=1
                continue
            if k in num:
                q=te[count]
                s = count
                z=1
                while (s+z) < len(te):
                    if te[s+z] == ' ':
                        break
                    if te[s+z] in num : 
                        q+=te[s+z]
                        z+=1
                        continue
                    if te[s+z] in tt : 
                        q+=te[s+z]
                        z+=1
                bi.append(q)
                another+=1
                count+=len(q)
                if k == te[-1] :
                    if count == (len(te)-1):
                        bi[-1] = bi[-1]+te[s+z]
                continue
            if k in tt : 
                q=te[count]
                s = count
                z=1
                while (s+z)<len(te):
                    if te[s+z]==' ':
                        break
                    if te[s+z] in tt:
                        q+=te[s+z]
                        z+=1
                        continue
                    if te[s+z] not in num and te[s+z] not in tt:
                        break   
                    if te[s+z] in num:
                        q+te[s+z]
                        z+=1
                bi.append(q)
                another +=1
                count +=len(q)
                continue
            if k not in num :
                if k not in tt :
                    bi.append(k)
                    another+=1
                    count+=1
        if k==' ':
            count+=1
            another+=1
            
    return bi



with open('dic.pickle', 'rb') as handle:
    vac = pickle.load(handle)



def embedding_ma(dict):
    c = np.zeros((len(dict),len(dict)))
    count = 0 
    for i in range(len(dict)) :
        if count == 0 :
            c[i][i] =0
            count +=1
            continue
        c[i][i]= 1
    return c

def reve_dict(dict):
    reve = {}
    key = list(vac.keys())
    ll = [str(i) for i in range(-1,len(vac))]
    cc = [i for i in range(len(vac))]
    for i in cc : 
        reve[ll[i]]= key[i]
    return reve

embedding_matrix = embedding_ma(vac)

reve = reve_dict(vac)

def encoding(data,vac,max_len):
    lf = []
    for i in range(len(data)) :
        temp = data[i]
        gg = []
        for j in temp : 
            if j in vac:
                gg.append(vac[j])
            else : 
                gg.append(vac['UNK'])
        if len(gg)<58:
            rest= max_len-len(gg)
            for i in range(rest):
                gg.append(vac['PAD'])
        lf.append(gg)
    return lf 



def lstm_cell():
    lstm = tf.contrib.rnn.BasicLSTMCell(256, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.5)


def construct_graph(mode,dictionary,embedding_matrix):
    dim = 512
    batch_size = 512
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    encoder_input = tf.placeholder( tf.int32 , shape=[None,11] )
    decoder_inputs = tf.placeholder( tf.int32 , shape=[None,11] )
    decoder_length = tf.placeholder( tf.int32 , shape=[None] )
    target_label = tf.placeholder(tf.int32,shape=[None,11])

    Inp = (encoder_input,decoder_inputs,decoder_length,target_label,keep_prob)
    init_emb = tf.contrib.layers.xavier_initializer()
    emb_mat = tf.get_variable('emb_mat',shape=[len(dictionary),300],initializer=init_emb)
    emb_x = tf.nn.embedding_lookup(emb_mat,encoder_input)
    emb_x_y = tf.nn.embedding_lookup( embedding_matrix , decoder_inputs )
    emb_x_y = tf.cast(emb_x_y,tf.float32)

    with tf.name_scope("Encoder"):
        cell_fw = tf.contrib.rnn.BasicLSTMCell(512,reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(512,reuse=tf.get_variable_scope().reuse)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

        enc_rnn_out , enc_rnn_state = tf.nn.bidirectional_dynamic_rnn( cell_fw , cell_bw , emb_x,dtype=tf.float32,swap_memory=True)
        enc_rnn_out = tf.concat(enc_rnn_out, 2)

        c = tf.concat([enc_rnn_state[0][0],enc_rnn_state[1][0]],axis=1)
        h = tf.concat([enc_rnn_state[0][1],enc_rnn_state[1][1]],axis=1)

        enc_rnn_state = tf.contrib.rnn.LSTMStateTuple(c,h)



## less the time_major !!!!

    with tf.variable_scope("Decoder") as decoder_scope:
        projection_layer = layer_core.Dense(len(dictionary),use_bias=False)
        mem_units = 2*dim
        #out_layer = Dense( 2449)
        #batch_size = tf.shape(enc_rnn_out)[0]
        beam_width = 3

    
        num_units = 2*dim
        memory = enc_rnn_out

        if mode == "infer":

            memory = tf.contrib.seq2seq.tile_batch( memory, multiplier=beam_width )
            decoder_length = tf.contrib.seq2seq.tile_batch( decoder_length, multiplier=beam_width)
            enc_rnn_state = tf.contrib.seq2seq.tile_batch( enc_rnn_state, multiplier=beam_width )
            batch_size = batch_size * beam_width

        else:
            batch_size = batch_size

        #attention_states = tf.transpose(enc_rnn_out,[1,0,2])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention( num_units,memory,memory_sequence_length=decoder_length,normalize=True)

        cell = tf.contrib.rnn.BasicLSTMCell( 2*dim )
        cell= tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)        
        #lstm = tf.contrib.rnn.BasicLSTMCell(256, reuse=tf.get_variable_scope().reuse)
        #lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        #cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(2)])

        cell = tf.contrib.seq2seq.AttentionWrapper( cell,
                                                attention_mechanism,
                                                attention_layer_size=num_units,
                                                name="attention")

        decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone( cell_state=enc_rnn_state)



        if mode == "train":

            helper = tf.contrib.seq2seq.TrainingHelper( inputs = emb_x_y , sequence_length = decoder_length )
            decoder = tf.contrib.seq2seq.BasicDecoder( cell = cell, helper = helper, initial_state = decoder_initial_state,output_layer=projection_layer) 
            outputs, final_state, final_sequence_lengths= tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                                           scope=decoder_scope)

            logits = outputs.rnn_output
            sample_ids = outputs.sample_id

        else:
            
            emb = tf.cast(embedding_matrix,tf.float32)
            #de_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state,multiplier=9) 
            #start_tokens = tf.tile(tf.constant([dictionary['BOS']], dtype=tf.int32), [ batch_size ] )
            #end_token = 0

            my_decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = cell,
                                                               embedding = emb,
                                                               start_tokens = tf.fill([512], dictionary['BOS']),
                                                               end_token = dictionary['EOS'],
                                                               initial_state = decoder_initial_state,
                                                               beam_width = beam_width,
                                                               output_layer = projection_layer )

            outputs, t1 , t2 = tf.contrib.seq2seq.dynamic_decode(  my_decoder,
                                                                   maximum_iterations=18,scope=decoder_scope )

            logits = tf.no_op()
            #sample_ids = outputs.rnn_outputs
            sample_ids = outputs.predicted_ids

    
    if mode == "train":
        non_zero = tf.cast(tf.not_equal(target_label,0),tf.float32)
        
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_label,logits=logits)
        la = tf.nn.embedding_lookup( embedding_matrix , target_label )
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=la,logits=logits)
        loss = tf.reduce_sum(loss*non_zero)/tf.reduce_sum(non_zero)

        globel_step = tf.Variable(0,name='globel_step',trainable=False)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss,params)
        clipped_gradient ,_ = tf.clip_by_global_norm(gradients,5)
        optimizer = tf.train.AdamOptimizer(0.001)

        #optimizer = tf.train.GradientDescentOptimizer(1)
        ##clipped gradients is this way to use !!!!!
        train_op = optimizer.apply_gradients(zip(clipped_gradient,params),global_step=globel_step)
        #train_op = optimizer.minimize(loss)
        #or sparse soft_max
        #output_vocab_size = len(dictionary)

        #loss = tf.losses.softmax_cross_entropy(  tf.one_hot( target_label,output_vocab_size ) , logits )
        #train_op = tf.train.AdamOptimizer().minimize(loss)
        ### clip gradient 
        correct = tf.reduce_sum( tf.cast( tf.equal( sample_ids , target_label ) , dtype=tf.float32 ) ) / 42
         #sample_ids = tf.transpose( sample_ids , [2,0,1] )[0]
    else : 
        correct = None
         #correct = tf.reduce_sum( tf.cast( tf.equal( sample_ids , Y ) , dtype=tf.float32 ) ) / maxlen
        loss = None
        train_op = None
        
    return train_op , loss , correct , sample_ids , logits , Inp 


tf.reset_default_graph()
infer_graph = tf.Graph()

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
with infer_graph.as_default():

    _ , _ , correct_test , pred_ids,_,Inp = construct_graph("infer",vac,embedding_matrix)

    infer_saver = tf.train.Saver()

infer_sess = tf.Session(config=tf.ConfigProto(),graph=infer_graph)

model_file= 'model.ckpt-14'

infer_saver.restore(infer_sess, model_file)

a = open(sys.argv[1],'r',encoding='utf-8')

test_in =  [] 

for i in a : 
    test_in.append(i[:-1])

seg_test = []

for i in test_in :
    seg_test.append(seg(i))


le = len(test_in)

test_x = []

for i in seg_test:
    if len(i)<12:
        test_x.append(i)
    else : 
        test_x.append(i[0:11])

test_e = encoding(test_x,vac,max_len=11)
#test_e = encoding(test_x,vac,max_len=11)


if le < 512 : 
    another = 512 - le 
    if another > le : 
        sup = [] 
        inte = another//le
        for i in range(inte):
            sup += test_e
        sup +=test_e[:(another-(le*inte))]
    #test_e = test_e+test_e[:another]
    test_e = test_e + sup
else : 
    another = 512 - (le%512) 
    test_e =test_e + test_e[:another]


#test_e = test_e+ test_e[:240]

question = []

for i in range(0,len(test_e),512) : 
    question.append(test_e[i:i+512])

batch_size = 512
c =  np.ones([batch_size])*11 

ans = []
for i in question :
    feed_dict = {Inp[0]:np.array(i),Inp[2]:c,Inp[4]:1}
    q = infer_sess.run( pred_ids,feed_dict=feed_dict)
    qq = q.transpose(2,0,1)
    ans.append(qq)

def recover_sentence(answer,reverse_dict):
    sentence = [] 
    for i in answer : 
        temp = reverse_dict[str(i)]
        if temp == 'EOS' : 
            sentence.append(' ')
            break 
        else : 
            sentence.append(temp)
    return sentence 


f = open(sys.argv[2], 'w',encoding='utf-8')
co = 0
count=0
total = 0
for i in ans: 
    if co == (len(ans)-1) : 
        tem = i[0][:le-total]
    else :
        tem = i[0]
    for j in range(len(tem)):
        temp = recover_sentence(tem[j],reve)
        se = ''
        count = 0 
        for g in temp : 
            count+=1
            if count==len(temp):
                se = se+g+'\n'
            else : 
                se = se+g
        se =se 
        f.write(se)
        total+=1
    co+=1
f.close()