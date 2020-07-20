# coding=utf-8
'''
基本思想就是利用两个RNN，一个RNN作为encoder，另一个RNN作为decoder
'''
import tensorflow as tf
import numpy as np

tf.reset_default_graph()
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz男人黑色国王女孩上下们']
num_dic = {n: i for i, n in enumerate(char_arr)}

seq_data = [['man', '男人'], ['black', '黑色'], ['king', '国王'], ['girls', '女孩们'], ['up', '上'], ['high', '下']]

# Seq2Seq Parameter
n_step = 5
n_hidden = 128
n_class = len(num_dic) # num_dic 长度 29

def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))  #最大长度为5，不足补空值
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]
        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])

        target_batch.append(target)
    return input_batch, output_batch, target_batch

#model
enc_input = tf.placeholder(tf.float32,[None,None,n_class]) # [batch_size,encoder_step(max_len),n_class]
dec_input = tf.placeholder(tf.float32,[None,None,n_class]) # [batch_size,decoder_step(max_len+1)基于S,E,n_class]
targets = tf.placeholder(tf.int64,[None,None]) # [batch_size,max_len+1]

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell,output_keep_prob=0.5)
    _,enc_states = tf.nn.dynamic_rnn(enc_cell,enc_input,dtype=tf.float32) # [batch_size, n_hidden(=128)]

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    outputs, _ = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)
    # outputs : [batch_size, max_len+1, n_hidden(=128)]

model=tf.layers.dense(outputs,n_class,activation=None) # model : [batch_size, max_len+1, n_class]
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#trainning
sess=tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(5000):
    _,loss=sess.run([optimizer,cost],feed_dict={enc_input: input_batch, dec_input: output_batch, targets: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

model_=sess.run(model,feed_dict={enc_input: input_batch, dec_input: output_batch})
print('model',model)
def translate(word):
    seq_data = [word, 'P' * len(word)]
    print(seq_data)
    input_batch, output_batch, _ = make_batch([seq_data])
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction, feed_dict={enc_input: input_batch, dec_input: output_batch})
    print('re',result)
    decoded = [char_arr[i] for i in result[0]]
    print('de',decoded)
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated.replace('P','')

print('test')
# print('man ->', translate('man'))
print('mans ->', translate('mans'))
# print('king ->', translate('king'))
# print('black ->', translate('black'))
print('upp ->', translate('upp'))
