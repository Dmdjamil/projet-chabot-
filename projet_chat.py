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
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

load_nltk()

# -----------------------------
# NLP init
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# -----------------------------
# Preprocess
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
# Model training
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
# Predict function
# -----------------------------
def predict(text):
    clean = preprocess(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]

    return ("😊 Positif" if pred == 1 else "😡 Négatif"), clean

# -----------------------------
# UI TITLE
# -----------------------------
st.title("🎬 Analyse de Films avec NLP")

# -----------------------------
# LOAD MOVIES
# -----------------------------
movies_df = pd.read_csv("movies.csv")

st.subheader("🎬 Choisissez un film")
movie_selected = st.selectbox("Liste des films", movies_df["title"])

description = movies_df[movies_df["title"] == movie_selected]["description"].values[0]
st.write("📖 Description :", description)

# -----------------------------
# USER REVIEW
# -----------------------------
st.subheader("💬 Donnez votre avis")
user_review = st.text_area("Votre impression sur le film")

# -----------------------------
# ANALYSIS BUTTON
# -----------------------------
if st.button("Analyser mon avis"):
    if user_review.strip() == "":
        st.warning("Veuillez écrire un avis.")
    else:
        result, clean_text = predict(user_review)

        st.success(f"Sentiment : {result}")
        st.info(f"Texte nettoyé : {clean_text}")

        # Save review
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
# SHOW REVIEWS
# -----------------------------
st.subheader("📊 Avis sauvegardés")
if os.path.exists("reviews.csv") and os.path.getsize("reviews.csv") > 0:
    df_reviews = pd.read_csv("reviews.csv")
else:
    df_reviews = pd.DataFrame(columns=["film", "review", "sentiment"])

if os.path.exists("reviews.csv"):
    df_reviews = pd.read_csv("reviews.csv")

    # filter
    film_filter = st.selectbox(
        "Filtrer par film",
        ["Tous"] + list(df_reviews["film"].unique())
    )

    if film_filter != "Tous":
        df_reviews = df_reviews[df_reviews["film"] == film_filter]

    st.dataframe(df_reviews)

    # stats
    st.subheader("📈 Statistiques")

    st.write("Total avis :", len(df_reviews))
    st.write("Positifs :", (df_reviews["sentiment"].str.contains("Positif")).sum())
    st.write("Négatifs :", (df_reviews["sentiment"].str.contains("Négatif")).sum())

else:
    st.info("Aucun avis sauvegardé pour le moment.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📘 À propos")
st.sidebar.write("""
App NLP pour analyser les avis de films.

Technologies :
- Streamlit
- NLTK
- TF-IDF
- Naive Bayes
""")
