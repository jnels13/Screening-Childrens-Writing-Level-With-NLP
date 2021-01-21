
import pandas as pd
import numpy as np
import string
import io
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def stopped_ (data):
    stopwords_list = stopwords.words('english') #removing stopwords
    stopwords_list += list(string.punctuation)  #removing punctuation

def lemma_(data):
    lemmatizer = WordNetLemmatizer()
    data2 = [lemmatizer.lemmatize(w) for w in data]     
    return(data2)

def preprocess_gen(text):
    text1 = word_tokenize(text)
    text2 = stopped_(text1)
    text3 = lemma_(text2)
    text4 = [' '.join(text3)]
    return text4