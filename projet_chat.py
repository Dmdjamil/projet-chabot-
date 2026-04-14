#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')  # 🔥 très important (nouvelle version NLTK)
nltk.download('stopwords')
# -----------------------------
# Nettoyage du texte
# -----------------------------
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    words = [word for word in tokens if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english')]

    return " ".join(words)

# -----------------------------
# Dataset simple
# -----------------------------
data = {
    "text": [
        "j aime",
        "incroyable",
        "j aime pas",
        "mauvais",
        "content",
        "triste",
        "bon",
        
    ],
    "label": [1, 1, 0, 0, 1, 0,1]
}

df = pd.DataFrame(data)
df["text"] = df["text"].apply(preprocess)

# -----------------------------
# Vectorisation + Modèle
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("💬 Chatbot de Sentiments")

user_input = st.text_input("Écris un message :")

if st.button("Analyser"):
    clean_text = preprocess(user_input)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("😊 Sentiment Positif")
    elif prediction == 0:
        st.error("😡 Sentiment Négatif")
    else :
        st.texte("Bonjour")

