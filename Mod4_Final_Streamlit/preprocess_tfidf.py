import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_tfidf(text):
    pickle_in = open('../Tfidf_vect.pkl', 'rb') 
    Tfidf_vect = pickle.load(pickle_in)
    text5 = Tfidf_vect.transform(text4)
    return(text5)