'''
根据目标词去预测其上下文词。
利用第一个隐含层的W作为查表词典,根据各个词的one_hot向量对应查找词典中对应词向量。


'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.reset_default_graph()
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence =" ".join(sentences).split(' ')
word_list = " ".join(sentences).split(' ')
word_list = list(set(word_list))
word_dict = {w:i for i,w in enumerate(word_list)} #词袋
print(len(word_list))

batch_size = 20
embedding_size = 2  #编码后词向量长度
voc_size = len(word_list) # 词典的size为不同词的个数 为了对应one_hot

def random_batch(data,size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)),size,replace=False)
    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])
        random_labels.append(np.eye(voc_size)[data[i][1]])
    return random_inputs,random_labels

# skipgram窗口大小
skip_grams=[]
for i in range(1,len(word_sequence)-1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i-1]],word_dict[word_sequence[i+1]]]
    for w in context:
        skip_grams.append([target,w])

# model
inputs=tf.placeholder(tf.float32,shape=[None,voc_size])
labels=tf.placeholder(tf.float32,shape=[None,voc_size])

# 两个embedding
w1=tf.Variable(tf.random_uniform([voc_size,embedding_size],-1.0,1.0))
w2=tf.Variable(tf.random_uniform([embedding_size,voc_size],-1.0,1.0))

hidden_layer=tf.matmul(inputs,w1)
output_layer=tf.matmul(hidden_layer,w2)  #s输出维度与输入维度一致
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,labels=labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(10000):
            batch_inputs,batch_labels=random_batch(skip_grams,batch_size)
            _,loss = sess.run([optimizer,cost],feed_dict={inputs:batch_inputs,labels:batch_labels})

            if (epoch+1)%1000==0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            trained_embeddings=w1.eval()
for i ,label in enumerate(word_list):
    print(i,label)
    x,y=trained_embeddings[i]
    print(x,y)
    plt.scatter(x,y)
    plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')

plt.show()
