import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_wordindex={value: key for key,value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')

#helper function
def decoded_review(encoded_review):
  return ' '.join([reverse_wordindex.get(i-3,'?')for i in encoded_review])

#function preprocess user input
def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word,2) + 3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review

#predict function
def predict_sentiment(review):
  preprocess_input=preprocess_text(review)
  prediction=model.predict(preprocess_input)
  sentiment='Positve' if prediction[0][0]>0.5 else 'Negative'
  return sentiment , prediction[0][0]

#streamlit app
import streamlit as st
st.title('IMDB Movies Review Sentiment')
st.write('Enter Movie Review to classify its postive or negative')

user_input=st.text_area('Movie Review')
if st.button('Classify'):
  preprocess_input=preprocess_text(user_input)
  prediction=model.predict(preprocess_input)
  sentiment='Positve' if prediction[0][0]>0.5 else 'Negative'

  st.write(f'Sentiment: {Sentiment}')
  st.write(f'Prediction: {prediction[0][0]}')

else:
  st.write('Please Enter Movie Review')