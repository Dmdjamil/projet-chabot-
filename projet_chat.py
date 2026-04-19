st.subheader("📊 Avis sauvegardés")

def load_reviews():
    if os.path.exists("reviews.csv") and os.path.getsize("reviews.csv") > 0:
        return pd.read_csv("reviews.csv")
    return pd.DataFrame(columns=["film", "review", "sentiment"])

df_reviews = load_reviews()

if df_reviews.empty:
    st.info("Aucun avis sauvegardé pour le moment.")
else:

    # filtre
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
