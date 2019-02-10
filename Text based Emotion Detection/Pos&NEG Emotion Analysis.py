import numpy as np
import re
import glob
import tensorflow as tf

batch_size = 128;embedding_dimension = 10;num_classes = 2
hidden_layer_size = 32;times_steps = 53;element_size = 1

print('Following is a program output for text-based Negative & Positive Emotion Analysis.\nI have used RNN Algorithm for the purpose.\nThank You')
#Getting NEGATIVE Data
neg_reviews=[]
for i in glob.glob(path to negative text data):
    list1=[]
    with open(i) as f:
        list1=f.readlines()

    list2=[]
    for i in list1:
        list2.append(re.sub('[\W0-9]+',' ',i))
        
    with open('new.txt','w+') as f1:
        f1.writelines(["%s" % item  for item in list2])
        
    f2=open('new.txt')
    neg_reviews.append(f2.read())
    
#Getting POSITIVE Data
pos_reviews=[]
for j in glob.glob(path to positive text data):
    list3=[]
    with open(j) as f3:
        list3=f3.readlines()

    list4=[]
    for k in list3:
        list4.append(re.sub('[\W0-9]+',' ',k))
        
    with open('new1.txt','w+') as f4:
        f4.writelines(["%s" % item  for item in list4])
        
    f5=open('new1.txt')
    pos_reviews.append(f5.read())

#Calculating Length of each data
length1=[]
length2=[]
for i in range(1000):
    length1.append(len(neg_reviews[i]))
    length2.append(len(pos_reviews[i]))

#Preprossing data in terms of length variations
new_neg_reviews=[]
new_pos_reviews=[]
for i in range(1000):
    if len(neg_reviews[i])>=200:
        new_neg_reviews.append(neg_reviews[i])
for j in range(1000):
    if len(pos_reviews[j])>=200:
        new_pos_reviews.append(pos_reviews[j])

#Calculating NEW Length of each data
length11=[]
length22=[]
for i in range(len(new_neg_reviews)):
    length11.append(len(new_neg_reviews[i]))

for k in range(len(new_pos_reviews)):    
    length22.append(len(new_pos_reviews[k]))

#Finally geeting the clean and set data
final_neg_reviews=[]
final_pos_reviews=[]
for i in range(999):
    final_neg_reviews.append(new_neg_reviews[i][:min(length11)])
    final_pos_reviews.append(new_pos_reviews[i][:min(length11)])

#Label formation
neg=[]
pos=[]
for i in range(999):
    neg.append(0)
for j in range(999):
    pos.append(1)

#Function to produve one hot encodings
def one_hot(vec,val=2):
    n=len(vec)
    out=np.zeros((n,val))
    out[range(n),vec]=1
    return out

#Getting one_hot_encodings
neglb=one_hot(neg)
poslb=one_hot(pos)
labels=np.concatenate((neglb,poslb))


data=final_neg_reviews+final_pos_reviews
wordcount=[]
for i in range(len(data)):
    wordcount.append(len(data[i].split()))    

#Fixing words per sentence in data
data_final=[]
for i in range(len(data)):
    lt=[]
    lt=data[i].split()[:min(wordcount)]
    lt1=[]
    lt1=' '.join(lt)
    data_final.append(lt1)
    
# Map from words to indices
word2index_map ={}
index=0
for sent in data_final:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index+=1
# Inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)

#Data Shuffling
data_indices = list(range(len(data_final)))
np.random.shuffle(data_indices)
data_final = np.array(data_final)[data_indices]

labels = np.array(labels)[data_indices]
train_x = data_final[:1500]
train_y = labels[:1500]
test_x = data_final[1500:]
test_y = labels[1500:]

#Function to produce batches
def get_sentence_batch(data_x,data_y):
    batch_size=128
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].lower().split()]
    for i in batch]
    y = [data_y[i] for i in batch]
    return x,y

#Initializing variables and operations
_inputs = tf.placeholder(tf.int32, shape=[None,None])
_labels = tf.placeholder(tf.float32, shape=[None, num_classes])


with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,
                    embedding_dimension],-1.0, 1.0),name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed,dtype=tf.float32)

weights = {'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size,num_classes],mean=0,stddev=.01))}
biases = {'linear_layer':tf.Variable(tf.truncated_normal([num_classes],mean=0,stddev=.01))}

# Extract the last relevant output and use in a linear layer
final_output = tf.matmul(states[1],weights["linear_layer"]) + biases["linear_layer"]
softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits = final_output,labels = _labels)
cross_entropy = tf.reduce_mean(softmax)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels,1),tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction,tf.float32)))*100


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        x_batch, y_batch = get_sentence_batch(train_x,train_y)
        sess.run(train_step,feed_dict={_inputs:x_batch, _labels:y_batch})
        opt=sess.run(final_output,feed_dict={_inputs:x_batch, _labels:y_batch})
        #print("Outputs:",opt.shape)
        if step % 100 == 0:
            acc = sess.run(accuracy,feed_dict={_inputs:x_batch,
            _labels:y_batch})
            print("Accuracy at %d: %.5f" % (step, acc))
            
    print('\n')       
    for test_batch in range(5):
        x_test, y_test = get_sentence_batch(test_x,test_y)
        batch_pred,batch_acc = sess.run([tf.argmax(final_output,1),accuracy],
                                feed_dict={_inputs:x_test,_labels:y_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))
        
     #Testing   
    samp1=' it was a very bad movie '
    xx1=[[word2index_map[word] for word in samp1.lower().split()]]
    yy1=[[1,0.]]
    qq=sess.run(final_output,feed_dict={_inputs:xx1,_labels:yy1})
    print("Sample 1 Sentence:",samp1)
    if np.argmax(qq,1)==0:
        print('Negative Comment')
    else:
        print('Positive Comment')
    
    samp2=' Happy to see it '
    xx2=[[word2index_map[word] for word in samp2.lower().split()]]
    yy2=[[0.,1]]
    qq2=sess.run(final_output,feed_dict={_inputs:xx2,_labels:yy2})
    print("Sample 2 Sentence:",samp2)
    if np.argmax(qq2,1)==0:
        print('Negative Comment')
    else:
        print('Positive Comment')
