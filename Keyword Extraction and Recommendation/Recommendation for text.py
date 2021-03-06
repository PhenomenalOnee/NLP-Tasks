import re
import numpy as np
import nltk
import itertools, string
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pandas
import heapq

path=r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\Sonnets.txt"
raw=[]
f=open(path)
raw=f.readlines()
f.close()

#Preproccessing Raw Data
remove=['\n',' ']
data=[]
for i in range(len(raw)):
    if raw[i] not in remove:
        data.append(raw[i])


new_data=[]
for i in range(len(data)):
    s=[]
    s=data[i].split()
    a=''
    a=' '.join(w for w in s)
    new_data.append(a)
    
#Preproccessing Raw Data
for i in range(len(new_data)):
    new_data[i]=new_data[i].replace(',','').replace("'",'').replace(';','').replace(':','').replace('.','').replace('`','')

new_data.append('155')

#Making Sonnets Texts
a=''
sonnets=[]
for i in range(3,len(new_data)):
    if new_data[i].isnumeric()==False:
        a+=new_data[i]+' '
    else:
        sonnets.append(a)
        a=''

#Removing common and stop words
lem = WordNetLemmatizer()
stem = PorterStemmer()
##Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","thy","thee","in","or","why"
             "when","where","how","in","on","thou","yet","st","make","see",'st',"came","made","took","take","ever","never"]

stop_words = stop_words.union(new_words)

#Removing common and stop words and Lemmatizing words
corpus=[]
for i in range(len(sonnets)):
    text=sonnets[i]
    text=text.lower()
    text = text.split()
    ##Stemming
    ps=PorterStemmer()
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  stop_words] 
    text = " ".join(text)
    corpus.append(text)

#Making Sonnets Vectors and Words Vocabulary
cv=CountVectorizer(max_df=0.8,stop_words=stop_words,ngram_range=(1,1))
X=cv.fit_transform(corpus)
indexmap = {index: word for word, index in cv.vocabulary_.items()}
vocabulary_size = len(indexmap)


#Counting Most Frequent Words In Each Sonnet
aray=np.array(X.todense())
keywords=[]
tagger=[]
for i in range(len(aray)):
    a=np.array(aray[i])
    mlt=heapq.nlargest(3,a)
    v=[]
    d=[]
    for i in mlt:
        if i not in d:
            d.append(i )
    for i in range(len(d)):
        q=np.where(a==d[i])
        v.append(q)
    mn=[]
    for i in v:
        a=i
        n=i[0]
        for i in range(len(n)):
            if n[i] not in mn:
                mn.append(n[i])
    index=mn
    keys=[]

    for j in range(len(index)):
        n=''
        n=indexmap[index[j]]
        keys.append(n)
        
    tagger.append(keys)
   
    a=', '.join(i for i in keys)
    keywords.append(a)
    


            
ans='Y'
while ans=='Y':
    sont1=int(input("Select the Sonnet Number For Recommendation(1-154):"))
    if sont1>0:
        print("Sonnet Selected is:\n",sonnets[sont1-1])
        keys=tagger[sont1-1]
        sontList=[]
        for ii in range(len(tagger)):
            count=0
            if ii != sont1-1:
                a=tagger[ii]
                for ij in keys:
                    for j in a:
                        if ij==j:
                            count+=1
                if count>=2:
                    sontList.append(sonnets[ii])

        print("\nRECOMMENDED SONNETS ARE:\n")
        for i in sontList:
            print(i)
            print('\n')
     
    else:
        print('Invalid Sonnet Number')
        
    ans=input("Do You Want To Continue(Y/N):")
    ans=ans.upper()
