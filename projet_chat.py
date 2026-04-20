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
# NLP Tools
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
# Train Model
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
        st.error("❌ Fichier **data.csv** introuvable dans le dossier de l'application.")
        st.stop()

model, vectorizer = train_model()

# -----------------------------
# Predict Function
# -----------------------------
def predict(text):
    clean = preprocess(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]
    return ("😊 Positif" if pred == 1 else "😡 Négatif"), clean

# -----------------------------
# Main UI
# -----------------------------
st.title("🎬 Analyse de Films avec NLP")

# Load movies
try:
    movies_df = pd.read_csv("movies.csv")
    movie_list = movies_df["title"].unique().tolist()
except FileNotFoundError:
    st.error("❌ Fichier **movies.csv** introuvable.")
    st.stop()

st.subheader("🎬 Choisissez un film")
movie_selected = st.selectbox("Liste des films", movie_list)

description = movies_df[movies_df["title"] == movie_selected]["description"].values[0]
st.write("📖 **Description :**", description)

# -----------------------------
# User Review
# -----------------------------
st.subheader("💬 Donnez votre avis")
user_review = st.text_area("Votre impression sur le film", height=150)

if st.button("🔍 Analyser mon avis", type="primary"):
    if not user_review.strip():
        st.warning("Veuillez écrire un avis avant d'analyser.")
    else:
        result, clean_text = predict(user_review)

        st.success(f"**Sentiment détecté :** {result}")
        with st.expander("Voir le texte nettoyé"):
            st.code(clean_text, language=None)

        # Save review
        new_data = pd.DataFrame({
            "film": [movie_selected],
            "review": [user_review],
            "sentiment": [result]
        })

        file_path = "reviews.csv"
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            new_data.to_csv(file_path, index=False)
        else:
            new_data.to_csv(file_path, mode='a', header=False, index=False)

        st.success("✅ Avis sauvegardé avec succès !")
        st.rerun()

# -----------------------------
# REVIEWS SECTION (Corrigée)
# -----------------------------
st.subheader("📊 Avis sauvegardés")

def load_reviews():
    file_path = "reviews.csv"
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame(columns=["film", "review", "sentiment"])
    
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(how='all').reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["film", "review", "sentiment"])


# ←←← Cette ligne est très importante
df_reviews = load_reviews()

if df_reviews.empty:
    st.info("Aucun avis sauvegardé pour le moment.")
else:
    # Filtre par film
    unique_films = sorted(df_reviews["film"].dropna().unique().tolist())
    film_filter = st.selectbox(
        "Filtrer par film",
        ["Tous"] + unique_films
    )

    if film_filter != "Tous":
        filtered_reviews = df_reviews[df_reviews["film"] == film_filter].copy()
    else:
        filtered_reviews = df_reviews.copy()

    st.dataframe(filtered_reviews, use_container_width=True)

    # Statistiques
    st.subheader("📈 Statistiques")
    total = len(filtered_reviews)
    positives = (filtered_reviews["sentiment"].astype(str).str.contains("Positif", na=False)).sum()
    negatives = total - positives

    col1, col2, col3 = st.columns(3)
    col1.metric("Total avis", total)
    col2.metric("😊 Positifs", positives)
    col3.metric("😡 Négatifs", negatives)

    if total > 0:
        st.progress(positives / total, text=f"Pourcentage positif : {positives/total:.1%}")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📘 À propos")
st.sidebar.info("""
App NLP pour analyser les avis de films.

Technologies utilisées :
- Streamlit
- NLTK (prétraitement)
- TF-IDF + Naive Bayes
""")

# Bouton utile en développement
if st.sidebar.button("🗑️ Effacer tous les avis"):
    if os.path.exists("reviews.csv"):
        os.remove("reviews.csv")
        st.success("Tous les avis ont été supprimés.")
        st.rerun()
