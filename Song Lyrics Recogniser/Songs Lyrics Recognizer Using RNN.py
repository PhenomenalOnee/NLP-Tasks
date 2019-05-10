import numpy as np
import tensorflow as tf
import glob

def processing(text):
    new_text=[]
    for i in text:
        new_text.append(i.lower().replace('...','').replace(',',''))
    #Fixing Number of words per line(columns) for Matrix Formulation latter
    length=len(new_text)
    data=[]
    for i in range(0,length,6):
        data.append(new_text[i:i+6 ])
    #Fixing Number of rowsfor Matrix Formulation latter    
    data=data[0:50]
    return data



with open(r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\nlp\Lyrics\AjLyr.txt") as f:
    ajwrd=[word for line in f for word in line.split()]
    aj11=processing(ajwrd)
f.close()    



with open(r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\nlp\Lyrics\MizLyr.txt") as f1:
    mizwrd=[word for line in f1 for word in line.split()]
    miz11=processing(mizwrd)
f1.close()


with open(r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\nlp\Lyrics\HHH2Lyr.txt") as f2:
    hhh2wrd=[word for line in f2 for word in line.split()]
    hhh211=processing(hhh2wrd)
f2.close()


with open(r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\nlp\Lyrics\HHHLyr.txt") as f3:
    hhhwrd=[word for line in f3 for word in line.split()]
    hhh11=processing(hhhwrd)
f3.close()

#Merging all lyrics
x_input=aj11+miz11+hhh211+hhh11

#Making labels
y1=[[0,0,0,1]]*50
y2=[[0,0,1,0]]*50
y3=[[0,1,0,0]]*50
y4=[[1,0,0,0]]*50
label=y1+y2+y3+y4

batch_size = 25;embedding_dimension = 64;num_classes = 4
hidden_layer_size = 32;times_steps = 6;element_size = 1

# Map from words to indices
word2index_map ={}
index=0
for sent in x_input:
    for word in sent:
        if word not in word2index_map:
            word2index_map[word] = index
            index+=1
# Inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)

data_indices = list(range(len(x_input)))
np.random.shuffle(data_indices)
data = np.array(x_input)[data_indices]

labels = np.array(label)[data_indices]
train_x = data[:100]
train_y = labels[:100]
test_x = data[100:]
test_y = labels[100:]

def get_sentence_batch(batch_size,data_x,data_y):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i]]
    for i in batch]
    y = [data_y[i] for i in batch]
    return x,y

_inputs = tf.placeholder(tf.int32, shape=[batch_size,times_steps])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,
                    embedding_dimension],-1.0, 1.0),name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)
    
with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell,embed,dtype=tf.float32)
    
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
    for step in range(150):
        x_batch, y_batch = get_sentence_batch(batch_size,train_x,train_y)
        sess.run(train_step,feed_dict={_inputs:x_batch, _labels:y_batch})
        if step % 5 == 0:
            acc = sess.run(accuracy,feed_dict={_inputs:x_batch,_labels:y_batch})
            print("Accuracy at %d: %.5f" % (step, acc))
            
    for test_batch in range(10):
        x_test, y_test = get_sentence_batch(batch_size,test_x,test_y)
        batch_pred,batch_acc = sess.run([tf.argmax(final_output,1),accuracy], feed_dict={_inputs:x_test, _labels:y_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))



