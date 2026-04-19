import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Télécharger ressources NLTK
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# Initialisation NLP
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

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
    data = {
        "text": [
            "I love this movie",
            "This film was terrible",
            "Amazing acting and great story",
            "I hate this movie",
            "Best movie ever",
            "Worst film I have seen"
        ],
        "label": [1, 0, 1, 0, 1, 0]
    }

    texts = [preprocess(t) for t in data["text"]]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, data["label"])

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
