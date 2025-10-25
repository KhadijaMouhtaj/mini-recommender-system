import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
# --- Load dataset ---
data = pd.read_csv("ml-100k/u.data", sep='\t', names=['user_id','movie_id','rating','timestamp'])
movies = pd.read_csv("ml-100k/u.item", sep='|', encoding='latin-1', usecols=[0,1], names=['movie_id','title'])
data = data.merge(movies, on='movie_id')
movies_clean = movies.copy()

# --- Content-Based Filtering ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_clean['title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_similar(title, top_n=5):
    idx = movies_clean.index[movies_clean['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_clean['title'].iloc[movie_indices].tolist()

# --- Collaborative Filtering maison avec SVD ---
# Créer la matrice user x movie
user_movie_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0).values

# SVD
U, sigma, Vt = np.linalg.svd(user_movie_matrix, full_matrices=False)
sigma_matrix = np.diag(sigma)

# Reconstruction avec réduction de dimension (top k)
k = 20
predicted_ratings = np.dot(np.dot(U[:,:k], sigma_matrix[:k,:k]), Vt[:k,:])

def recommend_for_user(user_id, top_n=5):
    user_idx = user_id - 1  # User IDs commencent à 1
    user_ratings = predicted_ratings[user_idx]
    already_rated = data[data['user_id']==user_id]['movie_id'].tolist()
    recommendations = [(i, r) for i, r in enumerate(user_ratings) if i+1 not in already_rated]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    return [movies_clean[movies_clean['movie_id']==i+1]['title'].values[0] for i, _ in recommendations]

# --- Hybrid Recommender ---
def hybrid_recommend(user_id, top_n=5):
    cf = recommend_for_user(user_id, top_n*2)
    cb = recommend_similar(cf[0], top_n*2)
    hybrid = list(dict.fromkeys(cf + cb))
    return hybrid[:top_n]




# --- RMSE ---
def compute_rmse():
    # Créer la matrice vraie
    true_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0).values
    # RMSE sur tous les éléments
    rmse = np.sqrt(mean_squared_error(true_matrix, predicted_ratings))
    return rmse

# --- Precision@K ---
def precision_at_k(k=5, threshold=4):
    precisions = []
    n_users = predicted_ratings.shape[0]
    for user_idx in range(n_users):
        user_pred = predicted_ratings[user_idx]
        user_true = user_movie_matrix[user_idx]
        
        # Indices des top-K prédits
        top_k_idx = np.argsort(user_pred)[::-1][:k]
        
        # Comptage des recommandations correctes
        relevant = sum([1 for i in top_k_idx if user_true[i] >= threshold])
        precisions.append(relevant / k)
    return np.mean(precisions)
