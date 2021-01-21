import pandas as pd 
import numpy as np 
import pickle 
from preprocess_tfidf import preprocess_tfidf
from preprocess_w2v import preprocess_w2v
from preprocess_gen import preprocess_gen
from sklearn.preprocessing import LabelEncoder
import streamlit as st 
from PIL import Image 
  
# loading in the model to predict on the data 
pickle_in = open('clf_xgb_tfidf.pkl', 'rb') 
clf_xgb_tfidf = pickle.load(pickle_in) 
pickle_in2 = open('clf_rf_w2v.pkl', 'rb') 
clf_rf_w2v = pickle.load(pickle_in2) 
pickle_in3 = open('Encoder.pkl', 'rb')
Encoder = pickle.load(pickle_in3)
  
def welcome(): 
    return 'welcome all'
  
# defining the function which will make the prediction using  
# the data which the user inputs 
def prediction(text):   
    text2 = preprocess_gen(text)
    predict_me_tfidf = preprocess_tfidf(text2)
    predict_me_w2v = preprocess_w2v(text2)

    prediction_tfidf = Encoder.inverse_transform(clf_xgb_tfidf.predict(predict_me_tfidf))
    prediction_w2v = Encoder.inverse_transform(clf_rf_w2v.predict(predict_me_w2v))

    return prediction_tfidf, prediction_w2v
  
# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    st.title("Grade-Level Prediction") 
      
    # here we define some of the front end elements of the web page like  
    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color: #ABBAEA;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Student Grade Level Classifier ML App </h1> 
    </div> 
    """
      
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # the following lines create text boxes in which the user can enter  
    # the data required to make the prediction 
    text = st.text_input("STUDENT'S TEXT:", "Type or Paste Here") 

    result ="" 
      
    # the below line ensures that when the button called 'Predict' is clicked,  
    # the prediction function defined above is called to make the prediction  
    # and store it in the variable result 
    if st.button("Predict"): 
        result = prediction(text) 
    st.success('The predicted grade level is: \n{} using tf-idf weighting and \n{} using Word2Vec vectorising'.format(result[0], result[1])) 
     
if __name__=='__main__': 
    main() 