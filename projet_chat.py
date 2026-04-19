import streamlit as st
import nltk
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="🎬 NLP Movies App", layout="centered")

# -----------------------------
# NLTK setup
# -----------------------------
@st.cache_resource
def load_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

load_nltk()

# -----------------------------
# NLP init
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# -----------------------------
# Model training (utilise cache_data car on manipule des données)
# -----------------------------
@st.cache_data
def train_model():
    try:
        df = pd.read_csv("data.csv")
        df["clean_text"] = df["text"].apply(preprocess)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["clean_text"])
        y = df["label"]

        model = MultinomialNB()
        model.fit(X, y)

        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Fichier **data.csv** introuvable. Veuillez le placer dans le dossier de l'app.")
        st.stop()

model, vectorizer = train_model()

# -----------------------------
# Predict function
# -----------------------------
def predict(text):
    clean = preprocess(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]
    return ("😊 Positif" if pred == 1 else "😡 Négatif"), clean

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Analyse de Films avec NLP")

# Load movies
try:
    movies_df = pd.read_csv("movies.csv")
except FileNotFoundError:
    st.error("❌ Fichier **movies.csv** introuvable.")
    st.stop()

st.subheader("🎬 Choisissez un film")
movie_selected = st.selectbox("Liste des films", movies_df["title"].unique())

description = movies_df[movies_df["title"] == movie_selected]["description"].values[0]
st.write("📖 **Description :**", description)

# -----------------------------
# User review
# -----------------------------
st.subheader("💬 Donnez votre avis")
user_review = st.text_area("Votre impression sur le film", height=150)

if st.button("🔍 Analyser mon avis", type="primary"):
    if not user_review.strip():
        st.warning("Veuillez écrire un avis avant d'analyser.")
    else:
        result, clean_text = predict(user_review)

        st.success(f"**Sentiment détecté :** {result}")
        with st.expander("Voir le texte nettoyé (préprocessing)"):
            st.code(clean_text, language="text")

        # Save review
        new_data = pd.DataFrame({
            "film": [movie_selected],
            "review": [user_review],
            "sentiment": [result]
        })

        file_path = "reviews.csv"
        if not os.path.exists(file_path):
            new_data.to_csv(file_path, index=False)
        else:
            new_data.to_csv(file_path, mode='a', header=False, index=False)

        st.success("✅ Avis sauvegardé avec succès !")
        st.rerun()   # Rafraîchit
