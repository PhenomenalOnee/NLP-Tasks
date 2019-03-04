import numpy as np 
import glob
import re #For preprocessing text data
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords



print('Program to extract named entity from documents\n')
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
data=[]
path=r"C:\Users\LAXMI\AppData\Local\Programs\Python\Python35\Employmnt NLP\document-analytics-master\agreements\*.txt"
for i in glob.glob(path):
    with open(i, encoding='utf8') as ff:

        q=ff.read()
        sent=nltk.sent_tokenize(q)
        
        for i in range(len(sent)):
            sent[i]=re.sub('[\W]+',' ',sent[i])
            sent[i]=sent[i].replace('_',' ')
        

        sent=remove_stop_words(sent)
        for i in range(len(sent)):
                wrds=sent[i].split()
                
                                
                a=' '.join(ii.capitalize() for ii in wrds)
                sent[i]= a
                
        data.append(sent)



documents=[]
for i in range(len(data)):
        doc=''
        for j in data[i]:
                c=j
                doc+=' '+c
        documents.append(doc)

        
#Setting grameer to search for prases                
grammar = ('''
    Name: {<NN>?<NNP>?<NN>?<NNP>} # NP
    ''')
grammar2 = ('''
    Date: {<NNP>?<NN>?<CD>+<CD>?<CD>?<CD>+<NNP>?} # NP
    ''')
chunkParser = nltk.RegexpParser(grammar)
chunkParser1=nltk.RegexpParser(grammar2)


#Getting eword tagging POS and chunkinging them
tags=[]
noun_chunks=[]
nos_chunks=[]
for i in documents:
        tags.append(nltk.pos_tag(nltk.word_tokenize(i)))

for i in tags:
        ss=''
        name = chunkParser.parse((i))
        dat=chunkParser1.parse((i))
        for j in name.subtrees(filter=lambda t: t.label()=='Name'):
                ss+=' '+str(j)+','
        noun_chunks.append((ss))
        for j in dat.subtrees(filter=lambda t: t.label()=='Date'):
                sd=' '+str(j)+','
        nos_chunks.append((sd))
                


#Procesing noun phrases obtained
noun_phrases=[]
for i in range(len(noun_chunks)):
        s1=noun_chunks[i]
        s2=s1.replace('(','').replace(')','').replace('/',' ')
        s3=s2.split(',')
        for j in range(len(s3)):
                s3[j]=s3[j].replace('NNP','').replace('NN','').replace('Name','')

        for k in range(len(s3)):
                w=s3[k].split()
                d=''
                for j in w:
                        if len(j)>2:
                                d+=' '+j
                s3[k]=d
        noun_phrases.append(s3)
        
#Procesing numeric phrases obtained        
nos_phrases=[]
for i in range(len(nos_chunks)):
        s1=nos_chunks[i]
        s2=s1.replace('(','').replace(')','').replace('/',' ')
        s3=s2.split(',')
        for j in range(len(s3)):
                s3[j]=s3[j].replace('NNP','').replace('NN','').replace('Date','').replace('CD','')

        for k in range(len(s3)):
                w=s3[k].split()
                d=''
                for j in w:
                        if j.isnumeric()==True:
                                d+=' '+j
                        elif len(j)>2:
                                d+=' '+j
                s3[k]=d
        nos_phrases.append(s3)

for i in noun_phrases:
        for j in i:
                if len(j)<=1:
                        i.remove(j)

for i in nos_phrases:
        for j in i:
                if len(j)<=1:
                        i.remove(j)


print('\nFollowing Noun Phrases for documnets are obtained....')
for i in range(5):
        
        print('For {} document:'.format(i+1))
        print('Top 10 Noun Phrases Are:')
        print(noun_phrases[i][:10])
        print('\n')
        
print('\nFollowing Numeric Phrases for documnets are obtained....')
for i in range(5):
        
        print('For {} document:'.format(i+1))
        print('Numeric Phrases Are:')
        print(nos_phrases[i])
        print('\n')
        

np.savetxt("Output(Noun_Phrases).csv", noun_phrases, delimiter=",", fmt='%s')

np.savetxt("Output(Numeric_Phrases).csv", nos_phrases, delimiter=",", fmt='%s')
