import tensorflow as tf #Tensorflow frmaework
import numpy as np 
import glob
import re #For preprocessing text data

from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords


print('Program to classify agreements and amendmenmts documnets\n')
#Function to remove stop words
stop_words=set(stopwords.words('english'))
def remove_stop_words(text):
        sents=[]
        for w in text:
            wrds=[]
            wrds=w.split()
            clean_wrds=[]
            for ww in wrds:
                if ww not in stop_words:
                    clean_wrds.append(ww)
            sent=[]
            sent=' '.join(i for i in clean_wrds)
            sents.append(sent)
        return sents

    
#Getting agreements data
data_=[]
path=r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\Employmnt NLP\document-analytics-master\agreements\*.txt"
for i in glob.glob(path):
    with open(i, encoding='utf8') as ff:

        q=ff.read()
        sent=nltk.sent_tokenize(q)
        
        for i in range(len(sent)):
            sent[i]=re.sub('[\W0-9]+',' ',sent[i])
            sent[i]=sent[i].replace('_',' ')

        sent=remove_stop_words(sent)
                    
        data_.append(sent)

#Getting 300 words per pragraph from documents      
one=[]
for i in range(len(data_)):
    dt=[]    
    st=[]
    st=data_[i]
    text=''
    for i in st:
        cc=i
        text+=cc
        
    for i in range(0,len(text.lower().split()),300):
        tr=text.lower().split()[i:i+300]
        dt.append(' '.join(tr))
    one.append(dt)

#Padding last paragraphs 
for i in range(len(one)):
	    if len(one[i][-1].split())<300:
		    one[i][-1]=one[i][-1]+ ' '+' PAD'*(300-len(one[i][-1].split()))

#Getting amendments data		    
data__=[]
path=r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\Employmnt NLP\document-analytics-master\amendments\*.txt"
for i in glob.glob(path):
    with open(i, encoding='utf8') as ff:
        q=ff.read()
        sent=nltk.sent_tokenize(q)
        
        for i in range(len(sent)):
            sent[i]=re.sub('[\W]+',' ',sent[i])
            sent[i]=sent[i].replace('_',' ')

        sent=remove_stop_words(sent)
        data__.append(sent)

##Getting 300 words per pragraph from documents 
two=[]
for i in range(len(data__)):
    dt=[]    
    st=[]
    st=data__[i]
    text=''
    for i in st:
        cc=i
        text+=' '+cc
        
    for i in range(0,len(text.lower().split()),300):
        tr=text.lower().split()[i:i+300]
        dt.append(' '.join(tr))
        
    two.append(dt)

#Padding last paragraphs  
for i in range(len(two)):
	    if len(two[i][-1].split())<300:
		    two[i][-1]=two[i][-1]+ ' '+' PAD'*(300-len(two[i][-1].split()))
		    
#getting all documents paragraphs
agre=[]
amen=[]
for i in one:
        for j in i:
                agre.append(j)
for i in two:
        for j in i:
                amen.append(j)
                

#Labels list
lab1=[]
lab2=[]
for i in range(len(agre)):
    lab1.append(0)
for i in range(len(amen)):
    lab2.append(1)

#network parameters
embedding_dimension = 10;num_classes = 2
hidden_layer_size = 64

#path to saving model
path=r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\Employmnt NLP\documemnt"

#joining both paragraph data into one list
# and labels
final=agre+amen
labels=lab1+lab2


#Function to create hot_encoding
def one_hot(vec,val=2):
    n=len(vec)
    out=np.zeros((n,val))
    out[range(n),vec]=1
    return out

#One_hot Vector encoding for labels
labels=one_hot(labels)

# Map from words to indices
word2index_map ={}
index=0
for sent in final:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index+=1
                
# Inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)
print("Size of Vocabulary:",vocabulary_size)

#Shuffling data and labels
data_indices = list(range(len(final)))
np.random.shuffle(data_indices)
data = np.array(final)[data_indices]
print("Shape of data:",data.shape)
labels = np.array(labels)[data_indices]
print("Shape of Labels:",labels.shape,'\n')

#Dividing data in training and testing sets
train_x = data[:]
train_y = labels[:]
test_x = data[150:]
test_y = labels[150:]

graph1 = tf.Graph()


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

with graph1.as_default():
        #Initializing variables and operations
        _inputs = tf.placeholder(tf.int32, shape=[None,None])
        _labels = tf.placeholder(tf.float32, shape=[None, num_classes])


        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_dimension],-1.0, 1.0),name='embedding')
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


        with tf.Session(graph=graph1) as sess:
            sess.run(tf.global_variables_initializer())
            print('\nInitializing Training......')
            for step in range(1000):
                x_batch, y_batch = get_sentence_batch(train_x,train_y)
                
                sess.run(train_step,feed_dict={_inputs:x_batch,_labels:y_batch})
                opt=sess.run(final_output,feed_dict={_inputs:x_batch, _labels:y_batch})
                #print("Outputs:",opt.shape)
                if step % 100 == 0:
                    acc = sess.run(accuracy,feed_dict={_inputs:x_batch,_labels:y_batch})
                    print("Accuracy at %d: %.5f" % (step, acc))
                    
            print('\n')       
            for test_batch in range(5):
                x_test, y_test = get_sentence_batch(test_x,test_y)
                batch_pred,batch_acc = sess.run([tf.argmax(final_output,1),accuracy],
                                        feed_dict={_inputs:x_test,_labels:y_test})
                print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))


            #Saving Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, path)
            print("\nModel saved in path: %s" % save_path)
            
            print('\nVerification')
            for samp in glob.glob(r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\Employmnt NLP\document-analytics-master\test\*.txt"):
            #samp=r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\Employmnt NLP\document-analytics-master\amendments\22872_2000-01-14_ADDENDUM TO EMPLOYMENT AGREEMENT.txt"
                with open(samp, encoding='utf8') as ff:
                    q=ff.read()
                    sent=nltk.sent_tokenize(q)
                    
                    for i in range(len(sent)):
                        sent[i]=re.sub('[\W0-9]+',' ',sent[i])
                        sent[i]=sent[i].replace('_',' ')


                    sent=remove_stop_words(sent)
                    docu=sent
                    txt=''
                    for i in docu:
                            cc=i
                            txt+= ' '+cc
                            
                    dxt=[]
                    for i in range(0,len(txt.lower().split()),300):
                            tr=txt.lower().split()[i:i+300]
                            dxt.append(' '.join(tr))

                
                    for i in range(len(dxt)):
                            if len(dxt[-1].split())<300:
                                    dxt[-1]=dxt[-1]+ ' '+' PAD'*(300-len(dxt[-1].split()))
                                    
                    
                    
                    
                    ag=0
                    am=0
                    print("DCOMT NAME:"   ,samp)
                    for i in range(len(dxt)):
                            xx=[[word2index_map[word] for word in dxt[i].lower().split()]]
                            opt=sess.run(final_output,feed_dict={_inputs:xx})
                            print(opt)
                            
                           
                            if np.argmax(opt)==1:
                                    am+=1
                            
                            else:
                                    
                                    ag+=1
                    print("RESULTS:")     
                    if am>ag:
                            print("ITS AN AMENDMENT DOCUMENT")
                    else:
                            print("ITS AN AGGREMENT DOCUMENT")

        

