import streamlit as st
from recommender import recommend_similar, recommend_for_user, hybrid_recommend, movies_clean

st.title("🎬 Mini Recommender System")

# Content-Based
st.header("Recommandation par film")
film = st.selectbox("Choisis un film :", movies_clean['title'].tolist(), key="film_selectbox")
if st.button("Recommander par film", key="btn_film"):
    recs = recommend_similar(film)
    for r in recs:
        st.write("-", r)

# Collaborative Filtering
st.header("Recommandation par utilisateur")
user_id = st.number_input("Entrez l'ID utilisateur :", min_value=1, max_value=943, key="user_input")
if st.button("Recommander par utilisateur", key="btn_user"):
    recs = recommend_for_user(user_id)
    for r in recs:
        st.write("-", r)

# Hybrid
st.header("Recommandation hybride")
if st.button("Recommandation hybride", key="btn_hybrid"):
    recs = hybrid_recommend(user_id)
    for r in recs:
        st.write("-", r)
