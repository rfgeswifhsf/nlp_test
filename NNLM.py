'''
输入：前n-1个词的one-hot表示

将输入与一个V*M的矩阵相乘，V表示词汇表的大小，M表示映射后的维度

得到的n-1个词的映射后向量，每个都是1*M维，共n-1个，组合成一维的向量(n-1)*M，表示前n-1个词组成的词向量

接一个softmax计算得到的是预测词的概率，输出是1*V维的，表示在词汇表在最大概率出现的词，即为预测的下一个词

这里最重要的是得到了一个V*M的映射矩阵，对每个词来说都是乘以这个矩阵计算概率，因此这个矩阵就是词汇表中V个词的表示，每一行表示一个词对应的向量
'''

import tensorflow as tf
import  numpy as  np
import warnings
warnings.filterwarnings("ignore")

sentences = [ "i like dog Luckid", "i love coffee mi", "i hate milk hello"]

word_list = " ".join(sentences).split(' ')
word_list = list(set(word_list))
word_dict={w:i for i,w in enumerate(word_list)}

number_dict={i:w for i,w in enumerate(word_list)}
n_class=len(word_dict)  # number of Vocabulary
print('n_class',n_class)

n_step=3 # number of steps ['i like', 'i love', 'i hate']
n_hidden=100 # number of hidden units

def make_batch(sentences):
    input_batch = []
    target_batch = []
    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]] #获取前n个词
        target = word_dict[word[-1]] #获取最后一个要预测的词
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])
    return input_batch,target_batch
a,b=make_batch(sentences)

X=tf.placeholder(tf.float32,[None,n_step,n_class]) # [batch_size, number of steps, number of Vocabulary]
Y=tf.placeholder(tf.float32,[None,n_class])

input=tf.reshape(X,shape=[-1,n_step*n_class])
H=tf.Variable(tf.random_normal([n_step*n_class,n_hidden]))
d=tf.Variable(tf.random_normal([n_hidden]))
U=tf.Variable(tf.random_normal([n_hidden,n_class]))
b=tf.Variable(tf.random_normal([n_class]))

tanh=tf.nn.tanh(d+tf.matmul(input,H)) # [batch_size, n_hidden]
model=tf.matmul(tanh,U)+b  # [batch_size, n_class]
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer=tf.train.AdamOptimizer(0.01).minimize(cost)
prediction=tf.argmax(model,1)

#training
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

input_batch,target_batch=make_batch(sentences)

for epoch in range(10000):
    _,loss=sess.run([optimizer,cost],feed_dict={X:input_batch,Y:target_batch})

    if(epoch + 1)%1000==0:
        print('Epoch : ', '%04d'%(epoch+1),'cost = ','{:.6f}'.format(loss))

model1=sess.run(model,feed_dict={X:input_batch})
print(model1)
# predict
predict=sess.run(prediction,feed_dict={X:input_batch})
print('word_dict : ',word_dict)
print('pre',predict)

# print('pre0',predict[0])
# Test
input=[sen.split()[:n_step] for sen in sentences]
print(input,'-->',[number_dict[n] for n in predict])
