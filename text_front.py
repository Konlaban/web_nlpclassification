import streamlit as st
import pandas as pd
import numpy as np

from datasets import load_dataset
#https://huggingface.co/datasets choose dataset from here. 
dataset_name = "prachathai67k" 
# dataset = load_dataset(dataset_name) 
train,test = load_dataset(dataset_name, split=['train[:5%]', 'test[:5%]'] ) 

import pandas as pd

df = pd.DataFrame(train)
#df.head()
df2 = pd.DataFrame(test)
st.cache_data
st.title('nlp_thai_1')
st.header('train_data')
st.write(df)

import re
import string

def clean_msg(msg):
    
    
    # ลบ text ที่อยู่ในวงเล็บ <> ทั้งหมด
    msg = re.sub(r'<.*?>','', msg)
    
    # ลบ hashtag
    msg = re.sub(r'#','',msg)
    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c),'',msg)
    
    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())
    
    return msg

import pythainlp
import nltk
#nltk.download('omw-1.4')
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words


th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()

def split_word(text):
    
    tokens = word_tokenize(text,engine='newmm')
    
    # Remove stop words ภาษาไทย และภาษาอังกฤษ
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]
    
    # หารากศัพท์ภาษาไทย และภาษาอังกฤษ
    # English
    tokens = [p_stemmer.stem(i) for i in tokens]
    
    # Thai
    tokens_temp=[]
    for i in tokens:
        w_syn = wordnet.synsets(i)
        if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)
    
    tokens = tokens_temp
    
    # ลบตัวเลข
    tokens = [i for i in tokens if not i.isnumeric()]
    
    # ลบช่องว่าง
    tokens = [i for i in tokens if not ' ' in i]

    return tokens
def cleaner(row):
    cleantext = clean_msg(row['body_text'] )
    return cleantext
df['clean_text'] = df.apply(cleaner, axis =1 )
tokens_list = [split_word(txt) for txt in df['clean_text']]

df2['clean_text'] = df2.apply(cleaner, axis =1 )
tokens_list2 = [split_word(txt) for txt in df2['clean_text']]
def mergetoken(i):
    sentence = ' '.join(tokens_list[i])
    return sentence 
def mergetoken2(i):
    sentence = ' '.join(tokens_list[i])
    return sentence 

df['sentence'] = df.index.map(mergetoken)
df2['sentence'] = df2.index.map(mergetoken2)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(df['sentence'])
vectorizer.fit(df2['sentence'])

st.header('clean data sample')
st.write(df['clean_text'])


