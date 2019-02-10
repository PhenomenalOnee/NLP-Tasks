from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import glob
from sklearn.metrics import accuracy_score

data=[]
for i in glob.glob('path to lyrics text files'):
    f=open(i)
    q=f.read()
    data.append(q)

labels=[0,1,2,3,4,5]
lb_names=['']#Names of lables/songs

#Printing labels 
for i,j in zip(labels,lb_names):
    print(i,j)

#Making lyrics vectors and words vocabulary
vectorizer_count = CountVectorizer()
train_tc = vectorizer_count.fit_transform(data)
print("\nDimensions of training data:", train_tc.shape)
print('WORDS MAPPING Length-',len(vectorizer_count.vocabulary_))

train_tcc = vectorizer_count.fit_transform(data).todense()

#Tfidf vectorization
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
print(train_tfidf.shape)

#input sample
input_data=['']

classifier = MultinomialNB().fit(train_tfidf, labels)
input_tc = vectorizer_count.transform(input_data)
input_tfidf = tfidf.transform(input_tc)
predictions = classifier.predict(input_tfidf)

print(predictions)
acc=accuracy_score(['input sample lables'], predictions)
print(acc*100)
for sent, category in zip(input_data, predictions):
   print('\nInput Data:', sent, '\n song NAme:', lb_names[category])
      
