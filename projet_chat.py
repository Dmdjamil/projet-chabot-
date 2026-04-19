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
# CONFIG (TOUJOURS EN PREMIER)
# -----------------------------
st.set_page_config(page_title="NLP App", layout="centered")

# -----------------------------
# Télécharger ressources NLTK
# -----------------------------
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

load_nltk()

# -----------------------------
# Initialisation NLP
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# -----------------------------
# Prétraitement
# -----------------------------
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in stop_words]

    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

# -----------------------------
# Modèle
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("data.csv")

    df["clean_text"] = df["text"].apply(preprocess)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = train_model()

# -----------------------------
# Prédiction
# -----------------------------
def predict(text):
    clean = preprocess(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]

    return ("😊 Positif" if pred == 1 else "😡 Négatif"), clean

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Analyse de films avec NLP")

# Charger films
movies_df = pd.read_csv("movies.csv")

st.subheader("🎬 Choisissez un film")
movie_selected = st.selectbox("Liste des films", movies_df["title"])

description = movies_df[movies_df["title"] == movie_selected]["description"].values[0]
st.write("📖 Description :", description)

# Avis
st.subheader("💬 Donnez votre avis")
user_review = st.text_area("Votre impression sur le film")

# Bouton analyse
if st.button("Analyser mon avis"):
    if user_review.strip() == "":
        st.warning("Veuillez écrire un avis.")
    else:
        result, clean_text = predict(user_review)

        st.success(f"Sentiment : {result}")
        st.info(f"Texte nettoyé : {clean_text}")

        # Sauvegarde sécurisée
        new_data = pd.DataFrame({
            "film": [movie_selected],
            "review": [user_review],
            "sentiment": [result]
        })

        if not os.path.exists("reviews.csv"):
            new_data.to_csv("reviews.csv", index=False)
        else:
            new_data.to_csv("reviews.csv", mode='a', header=False, index=False)

        st.success("✅ Avis sauvegardé !")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📘 À propos")
st.sidebar.write("""
- NLP (NLTK)
- TF-IDF
- Naive Bayes
""")
