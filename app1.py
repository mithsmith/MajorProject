import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import MultinomialNB
from PIL import Image

SAVED_MODEL_FILE = "SavedModel.sav"
 
#df = pd.read_csv('senti_review_file.csv')
#x = df.iloc[:,0].values # Review column as input
#y = df.iloc[:,1].values # Sentiment column as output
st.title("Sentiment Analysis On Reviews")
st.subheader('TF-IDF Vectorizer')
st.write('This project is based on Naive Bayes Classifier.')
loaded_model = joblib.load(SAVED_MODEL_FILE)
#text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
#text_model.fit(x,y)
message = st.text_area("Enter your text below:", "", "type here...")
out_text = loaded_model.predict([message])
if st.button("Analyze Sentiment"):
  st.title(out_text)
