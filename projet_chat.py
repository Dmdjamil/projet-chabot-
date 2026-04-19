import streamlit as st
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Télécharger ressources NLTK
# -----------------------------

nltk.download('punkt')
nltk.download('punkt_tab')   # 🔥 AJOUT IMPORTANT
nltk.download('stopwords')
nltk.download('wordnet')
# -----------------------------
# Initialisation NLP
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

#fichier movie

movies_df = pd.read_csv("movies.csv")

#Selection du film
st.subheader("🎬 Choisissez un film")
movie_selected = st.selectbox(
    "Liste des films",
    movies_df["title"]
)

# Afficher description
description = movies_df[movies_df["title"] == movie_selected]["description"].values[0]
st.write("📖 Description :", description)

#Avis utilisateur
st.subheader("💬 Donnez votre avis")

user_review = st.text_area("Votre impression sur le film")



# -----------------------------
# Fonction de prétraitement
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
# Entraînement modèle
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("data.csv")

    # Nettoyage
    df["clean_text"] = df["text"].apply(preprocess)

    # Vectorisation
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    # Modèle
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

    return "😊 Positif" if pred == 1 else "😡 Négatif", clean

#Analyse
if st.button("Analyser mon avis"):
    if user_review.strip() == "":
        st.warning("Veuillez écrire un avis.")
    else:
        result, clean_text = predict(user_review)

        st.success(f"Sentiment : {result}")
        st.info(f"Texte nettoyé : {clean_text}")

# -----------------------------
# Interface Streamlit
# -----------------------------
st.set_page_config(page_title="NLP App", layout="centered")

st.title("🧠 Analyse de Sentiment (NLP)")
st.write("Tape un texte pour analyser son sentiment")

# Input utilisateur
user_input = st.text_area("✍️ Votre texte ici :")

if st.button("Analyser"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        result, clean_text = predict(user_input)

        st.success(f"Résultat : {result}")
        st.info(f"Texte nettoyé : {clean_text}")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📘 À propos")
st.sidebar.write("""
Cette application utilise :

- NLP (NLTK)
- TF-IDF
- Naive Bayes

Elle permet de classifier un texte en positif ou négatif.
""")

# -----------------------------
# Bonus : choix du modèle (optionnel)
# -----------------------------
st.sidebar.title("⚙️ Options futures")
st.sidebar.write("Tu peux ajouter :")
st.sidebar.write("- Decision Tree")
st.sidebar.write("- Upload CSV")
st.sidebar.write("- Deep Learning")

#sauvegarder les avis
new_data = pd.DataFrame({
    "film": [movie_selected],
    "review": [user_review],
    "sentiment": [result]
})

new_data.to_csv("reviews.csv", mode='a', header=False, index=False)
