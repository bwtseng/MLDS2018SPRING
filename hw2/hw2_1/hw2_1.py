import tensorflow as tf 
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas 
import json
from tensorflow.python.layers.core import Dense
from tensorflow.python.layers import core as layer_core
import random 
import pickle 
import sys


with open('vac.pickle', 'rb') as handle:
    vac = pickle.load(handle)

def load_testing_id(filepath):
    #temp = 'MLDS_hw2_1_data/'
    #c = open('testing_label.json')
    d = os.listdir(filepath)
    for i in d:
        if i == 'id.txt':
            d=i
        count+=1

    #d = os.listdir(filepath)[1]
    ff = [] 
    cc = open(os.path.join(filepath,d))
    for i in cc : 
        ff.append(i[:-1])
    return ff

te_id = load_testing_id(sys.argv[1])

def preprocess_test(filepath,ID):
    file = os.listdir(filepath)
    count=0
    for i in file :
        if i =='feat':
            file=i
        count +=1
    file = os.path.join(filepath,file)
    temp = os.listdir(file)
    test_list = []
    for i in range(len(temp)):
        count=0
        while True : 
            if temp[count][:-4] == ID[i]:
                test_list.append(np.load(os.path.join(file,temp[count])))
                break
            else : 
                count +=1 
    return np.array(test_list)


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

def reverse_dict(dictionary):
    b = dict()
    key = list(dictionary.keys())
    number = [str(i) for i in range(len(dictionary))]
    count = 0
    for i in number : 
        b[i] = key[count]
        count +=1

    return b

embeddings_matrix = embedding_ma(vac)
reve = reverse_dict(vac)

def lstm_cell():
    lstm = tf.contrib.rnn.BasicLSTMCell(256, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.5)



def construct_graph(mode,dictionary,embedding_matrix):
    dim = 256
    batch_size = 145
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    encoder_input = tf.placeholder( tf.float32 , shape=[None,80,4096] )
    decoder_inputs = tf.placeholder( tf.int32 , shape=[None,42] )
    decoder_length = tf.placeholder( tf.int32 , shape=[None] )
    target_label = tf.placeholder(tf.int32,shape=[None,42])

    Inp = (encoder_input,decoder_inputs,decoder_length,target_label,keep_prob)

    emb_x_y = tf.nn.embedding_lookup( embedding_matrix , decoder_inputs )
    emb_x_y = tf.cast(emb_x_y,tf.float32)

    with tf.name_scope("Encoder"):
        cell_fw = tf.contrib.rnn.BasicLSTMCell(256,reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(256,reuse=tf.get_variable_scope().reuse)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

        enc_rnn_out , enc_rnn_state = tf.nn.bidirectional_dynamic_rnn( cell_fw , cell_bw , encoder_input,dtype=tf.float32,swap_memory=True)
        enc_rnn_out = tf.concat(enc_rnn_out, 2)

        c = tf.concat([enc_rnn_state[0][0],enc_rnn_state[1][0]],axis=1)
        h = tf.concat([enc_rnn_state[0][1],enc_rnn_state[1][1]],axis=1)

        enc_rnn_state = tf.contrib.rnn.LSTMStateTuple(c,h)
        #mgru_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(4)])#,state_is_tuple=True)
        #init_state = mgru_cell.zero_state(145, dtype=tf.float32)
        #enc_rnn_out, enc_rnn_state = tf.nn.dynamic_rnn(mgru_cell, inputs=encoder_input, initial_state=init_state, time_major=False)



## less the time_major !!!!

    with tf.variable_scope("Decoder") as decoder_scope:
        projection_layer = layer_core.Dense(len(dictionary),use_bias=False)
        mem_units = 2*dim
        #mem_units = dim
        #out_layer = Dense( 2449)
        #batch_size = tf.shape(enc_rnn_out)[0]
        beam_width = 3

    
        num_units = 2*dim
        #num_units = dim
        memory = enc_rnn_out

        if mode == "infer":

            memory = tf.contrib.seq2seq.tile_batch( memory, multiplier=beam_width )
            decoder_length = tf.contrib.seq2seq.tile_batch( decoder_length, multiplier=beam_width)
            enc_rnn_state = tf.contrib.seq2seq.tile_batch( enc_rnn_state, multiplier=beam_width )
            batch_size = batch_size * beam_width

        else:
            batch_size = batch_size

        #attention_states = tf.transpose(enc_rnn_out,[1,0,2])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention( num_units,memory,memory_sequence_length=decoder_length)#,normalize=True)

        cell = tf.contrib.rnn.BasicLSTMCell( 2*dim )
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        #cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(4)])#,state_is_tuple=True)
 

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
            
            emb = tf.cast(embeddings_matrix,tf.float32)
            #de_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state,multiplier=9) 
            #start_tokens = tf.tile(tf.constant([dictionary['BOS']], dtype=tf.int32), [ batch_size ] )
            #end_token = 0

            my_decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = cell,
                                                               embedding = emb,
                                                               start_tokens = tf.fill([145], dictionary['BOS']),
                                                               end_token = dictionary['EOS'],
                                                               initial_state = decoder_initial_state,
                                                               beam_width = beam_width,
                                                               output_layer = projection_layer )

            outputs, t1 , t2 = tf.contrib.seq2seq.dynamic_decode(  my_decoder,
                                                                   maximum_iterations=80,scope=decoder_scope )

            logits = tf.no_op()
            #sample_ids = outputs.rnn_outputs
            sample_ids = outputs.predicted_ids

    
    if mode == "train":
        non_zero = tf.cast(tf.not_equal(target_label,0),tf.float32)
        
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_label,logits=logits)
        la = tf.nn.embedding_lookup( embedding_matrix , target_label )
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=la,logits=logits)
        loss = (tf.reduce_sum(loss*non_zero)/tf.reduce_sum(non_zero))

        globel_step = tf.Variable(0,name='globel_step',trainable=False)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss,params)
        clipped_gradient ,_ = tf.clip_by_global_norm(gradients,5)

        #optimizer = tf.train.AdamOptimizer(0.001)
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        ##clipped gradients is this way to use !!!!!
        train_op = optimizer.apply_gradients(zip(clipped_gradient,params),global_step=globel_step)

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

with infer_graph.as_default():

    _ , _ , correct_test , pred_ids,_,Inp = construct_graph("infer",vac,embeddings_matrix)

    infer_saver = tf.train.Saver()

infer_sess = tf.Session(graph=infer_graph)


#first training then saving  
#model_file=tf.train.latest_checkpoint('ckpt') ##補上去

model_file = 'model.ckpt-290'
infer_saver.restore(infer_sess, model_file)

en_in_t = preprocess_test(sys.argv[1],te_id)

le = len(en_in_t)

if le < 145 : 
    qqq = en_in_t
    another = 145 - le 
    if another > le : 
        sup = []
        inte = another//le
        for i in range(inte):
            en_in_t = np.concatenate((en_in_t,qqq,axis=0)
    else:
        inte = 0 
        en_in_t = np.concatenate((en_in_t,qqq[:(another-(le*inte))]),axis=0))
    #test_e = test_e+test_e[:another]
else : 
    another = le%145
    en_in_t = np.concatenate((en_in_t,en_in_t[:another]),axis=0)
    #test_e =test_e + test_e[:another]

question = []
for i in range(0,le,145):
    question.append(en_in_t[i:i+145])



ans = []
count=0
for i in question : 
    feed_dict = {Inp[0]:i,Inp[2]:np.ones([145])*42,Inp[4]:1}
    q = infer_sess.run( pred_ids,feed_dict=feed_dict)
    q = q.transpose(0,2,1)
    ans.append(q)
    #if count == (len(ans)-1):
    #    ans.append(q[0:another])
    #else : 
    #    ans.appen(q)
#qq =q.transpose(0,2,1)[0:100]

'''
def read_ID():
    temp = open('MLDS_hw2_1_data/testing_id.txt')
    qq = []
    for i in range(100):
        c = temp.readline()[:-1]
        qq.append(c)
    return qq

ID = read_ID()
'''

##剩下改這邊了!!!!
f = open(sys.argv[2],'w',encoding='utf-8')
cou=0
total = 0
ind = 0
for i in ans:
    if cou ==(len(ans)-1):
        tem = i[:(le-total)]
    else :
        tem = i[0]
    for j in range(len(tem)) :
        na = te_id[ind]
        temp = recover_sentence(tem[j][0],reve)
        se = ''
        count = 0 
        for g in temp : 
            count+=1
            if count==len(temp):
                se = se+g+'\n'
            else : 
                se = se+g+' '
        se = na+','+se 
        total+=1
        ind+=1
        f.write(se)
    f.close()
