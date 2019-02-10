import tensorflow as tf #Tensorflow frmaework
import numpy as np 
import glob
import re #For preprocessing text data
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from tensorflow.python.saved_model import tag_constants


raw=[]
##Order of files loaded:
#1.angry.txt
#2.Disgusted.txt
#3.happy.txt
#4.nervous.txt
#5.sad.txt
for i in glob.glob('/path to text data'):
    f=open(i)
    raw.append(f.readlines())
    
raw[2]=raw[2][:210] # Setting size of text data
raw[4]=raw[4][:210]

#Preprocessing data & removal of special characters
data=[]
for i in range(len(raw)):
    for j in range(len(raw[i])):
        data.append(re.sub('[\W0-9]+',' ',raw[i][j]))

#Producing labels for different classes
labl0=[]
labl1=[]
labl2=[]
labl3=[]
labl4=[]
for i in range(len(raw[0])):
    labl0.append(0)
for i in range(len(raw[1])):
    labl1.append(1)
for i in range(len(raw[2])):
    labl2.append(2)
for i in range(len(raw[3])):
    labl3.append(3)
for i in range(len(raw[4])):
    labl4.append(4)
    
#Function to remove stop words from text  
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
        
#Removing stop words from text
stop_words=set(stopwords.words('english'))
sentences=remove_stop_words(data)

#Getting number of words per sentence in data
words_length_2=[]
for i in sentences:
    words_length_2.append(len(i.split()))
            
#padding sentences for Array Usage
paded_data_2=[]
for i in sentences:
    paded_data_2.append(i+' '+'PAD '*(max(words_length_2)-len(i.split())))


#RNN Approach
def rnn():
    print("\n \\\\RNN APPROACH//// \n")
    #Path for saving model trained
    path=r""
    
    batch_size = 128;embedding_dimension = 10;num_classes = 5
    hidden_layer_size = 64;times_steps = 95; element_size = 1
    
    graph1 = tf.Graph()

    #Function to create hot_encoding
    def one_hot(vec,val=5):
        n=len(vec)
        out=np.zeros((n,val))
        out[range(n),vec]=1
        return out

    #One_hot Vector encoding for labels
    lab0=one_hot(labl0)
    lab1=one_hot(labl1)
    lab2=one_hot(labl2)
    lab3=one_hot(labl3)
    lab4=one_hot(labl4)

    labels=np.concatenate((lab0,lab1,lab2,lab3,lab4))
   
    
    # Map from words to indices
    word2index_map ={}
    index=0
    for sent in paded_data_2:
        for word in sent.lower().split():
            if word not in word2index_map:
                word2index_map[word] = index
                index+=1
                
    # Inverse map
    index2word_map = {index: word for word, index in word2index_map.items()}
    vocabulary_size = len(index2word_map)
    print("Size of Vocabulary:",vocabulary_size)

    #Shuffling data and labels
    data_indices = list(range(len(paded_data_2)))
    np.random.shuffle(data_indices)
    data = np.array(paded_data_2)[data_indices]
    print("Shape of data:",data.shape)
    labels = np.array(labels)[data_indices]
    print("Shape of Labels:",labels.shape,'\n')

    #Dividing data in training and testing sets
    train_x = data[:]
    train_y = labels[:]
    test_x = data[950:]
    test_y = labels[950:]

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
        #Initializing variables and operations...
        
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
        

    
        
        with tf.Session(graph=graph1) as sess:
            
            sess.run(tf.global_variables_initializer())
            for step in range(1000):
                x_batch, y_batch = get_sentence_batch(train_x,train_y)
                sess.run(train_step,feed_dict={_inputs:x_batch, _labels:y_batch})
                opt=sess.run(final_output,feed_dict={_inputs:x_batch, _labels:y_batch})
                
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

            #Saving Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, path)
            print("\nModel saved in path: %s" % save_path)
                
            input_data=['You have no love of fatherland in you',"This vexes me more than anything",'There arose a very terrible storm',
                        'It was music to my ears','It is not so with the human heart']

            sents=remove_stop_words(input_data)
            
            #Labelling for testing
            y0=np.array([[1,0.,0.,0.,0.]])
            y1=np.array([[0.,1,0.,0.,0.]])
            y2=np.array([[0.,0.,1,0.,0.]])
            y3=np.array([[0.,0.,0.,1,0.]])
            y4=np.array([[0.,0.,0.,0.,1]])
            yy=np.concatenate((y1,y0,y3,y2,y4))
            c=0
            for i in sents:
                xx=[[word2index_map[word] for word in i.lower().split()]]
                yt=yy[c]
                qq=sess.run(final_output,feed_dict={_inputs:xx,_labels:yt.reshape(1,5)})
                
                
                print("\nSentence:\n",input_data[c])
                #Printing Original Emotion
                if np.argmax(yt)==0:
                    print('Original Emotion: Angry')
                if np.argmax(yt)==1:
                    print('Original Emotion: Disgusted')
                if np.argmax(yt)==2:
                    print('Original Emotion: Happy')
                if np.argmax(yt)==3:
                    print('Original Emotion: Nervous')
                if np.argmax(yt)==4:
                    print('Original Emotion: Sad')

                #Printing Predicted Emotion 
                if np.argmax(qq,1)==4:
                    print('Emotion Predicted: Sad')
                elif np.argmax(qq,1)==0:
                    print('Emotion Predicted: Angry')
                elif np.argmax(qq,1)==1:
                    print('Emotion Predicted: Disgusted')
                elif np.argmax(qq,1)==2:
                    print('Emotion Predicted: Happy')
                else:
                    print('Emotion Predicted: Nervous')

                c+=1
        
def tfidf():
    print("\n \\\\TFIDF APPROACH//// \n")
    lb_names=['Angry','Disgusted','Happy','Nervous','Sad']
    #Label vector
    lablvectr=labl0+labl1+labl2+labl3+labl4
    
    #Initializing CountVectorizer()
    vectorizer_count = CountVectorizer()
    
    #Fitting data
    train_tc = vectorizer_count.fit_transform(sentences)
    print("\nDimensions of training data:", train_tc.shape)
    print('WORDS MAPPING Length-',len(vectorizer_count.vocabulary_))

    train_tcc = vectorizer_count.fit_transform(sentences).todense()
    print("Vectorized form of Sentences:\n",train_tcc)

    #Initializing TFIDF Vectorization
    tfidf = TfidfTransformer()
    train_tfidf = tfidf.fit_transform(train_tc)
    print("Shape of TDIDF Vectors:",train_tfidf.shape)

    input_data=['You have no love of fatherland in you',"This vexes me more than anything",'There arose a very terrible storm',
                        'It was music to my ears','It is not so with the human heart']
    
    sentss=remove_stop_words(input_data)
    

    #Fitting data in Naive Bayes Classifier
    classifier = MultinomialNB().fit(train_tfidf, lablvectr)
    input_tc = vectorizer_count.transform(sentss)
    input_tfidf = tfidf.transform(input_tc)
    predictions = classifier.predict(input_tfidf)

    print("Predictions for each sentence are:",predictions)
    acc=accuracy_score([1,0,3,2,4], predictions)
    print("Accuracy of Classifier:",acc*100)
    
    for sent, category in zip(input_data, predictions):
       print('\nInput Data:', sent, '\n Emotion:', lb_names[category])
      
ans='Y'

print('Following is a program output for text-based Sentiment Analysis.\nFive Sentiments are Used for classification:\nSadness, Anger, Happiness, Nervousness, Disgust\n')
print("Note: The dataset I have Used does not have much variations and \ndiversity in terms of words used.\nHence Accuracy is not state-of-the-art.\nThank You")

while ans=='Y':
    
    print("      ****Menu For Algorithm Selection****")
    print('1.RNN Method')
    print('2.Tfidf Vector Method')
    
    ch=int(input('Enter Choice:'))

    if ch==1:
        rnn()
    elif ch==2:
        tfidf()
    else:
        print('Invalid Choice')

    ans=input("Do You Want To Continue:(Y/N):")
    ans=ans.upper()
