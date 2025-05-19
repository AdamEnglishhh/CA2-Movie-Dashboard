#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Loading the x3 datasets:
movies = pd.read_csv("movies-2.csv", encoding='ISO-8859-1')
ratings = pd.read_csv("rating.csv", encoding='ISO-8859-1')
tags = pd.read_csv("tags.csv", encoding='ISO-8859-1')


# In[2]:


movies.head()


# In[3]:


ratings.head()


# In[4]:


tags.head()


# In[5]:


# One-hot encoding (Movie genres):

genre_dummies = movies['genres'].str.get_dummies(sep='|')

# Computing the average rating per movie:

avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'avg_rating']

# Counting the number of tags per movie:
tag_counts = tags.groupby('movieId').size().reset_index(name='tag_count')


# In[6]:


# Merging all the features together:

movies_features = movies.merge(avg_ratings, on='movieId', how='left')
movies_features = movies_features.merge(tag_counts, on='movieId', how='left')
movies_features = pd.concat([movies_features, genre_dummies], axis=1)


# In[7]:


# Filling the missing tag counts with 0

movies_features['tag_count'] = movies_features['tag_count'].fillna(0)


# In[8]:


# Selecting the features for clustering:

features_for_clustering = movies_features[['avg_rating', 'tag_count'] + list(genre_dummies.columns)]

features_for_clustering.head()


# ## K-Means

# In[9]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# In[10]:


# Standardising the features for clustering:

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_for_clustering)


# In[11]:


# Determining the optimal number of clusters using both, Elbow Method and Silhouette Score:

inertia = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))


# In[12]:


# Plotting the Elbow and Silhouette Score
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(k_range, sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.show()


# In[13]:


# Fit KMeans with optimal k = 4
kmeans_final = KMeans(n_clusters=4, random_state=42)
movies_features['cluster_kmeans'] = kmeans_final.fit_predict(X_scaled)

# Analyze cluster centroids
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans_final.cluster_centers_), 
                               columns=features_for_clustering.columns)

# Compute number of movies in each cluster
cluster_counts = movies_features['cluster_kmeans'].value_counts().sort_index()

cluster_centers, cluster_counts


# ## DBSCAN

# In[14]:


from sklearn.cluster import DBSCAN


# In[15]:


# Preparing features:

genre_dummies = movies['genres'].str.get_dummies(sep='|')
avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'avg_rating']
tag_counts = tags.groupby('movieId').size().reset_index(name='tag_count')

movies_features = movies.merge(avg_ratings, on='movieId', how='left')
movies_features = movies_features.merge(tag_counts, on='movieId', how='left')
movies_features = pd.concat([movies_features, genre_dummies], axis=1)
movies_features['tag_count'] = movies_features['tag_count'].fillna(0)

features_for_clustering = movies_features[['avg_rating', 'tag_count'] + list(genre_dummies.columns)]


# In[16]:


# Standardising:

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_for_clustering)


# In[17]:


# Applying DBSCAN:

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
movies_features['cluster_dbscan'] = dbscan_labels


# In[18]:


# Analysing the clustering output:

n_noise = list(dbscan_labels).count(-1)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels) if n_clusters_dbscan > 1 else np.nan

n_clusters_dbscan, n_noise, dbscan_silhouette


# ## Collaborative Filtering

# In[19]:


from sklearn.metrics.pairwise import cosine_similarity


# In[21]:


# Creating user-item ratings matrix:

user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

user_item_filled = user_item_matrix.fillna(0)

# Computing cosine similarity between users and items:

user_similarity = cosine_similarity(user_item_filled)
item_similarity = cosine_similarity(user_item_filled.T)

# Converting similarities to DF:

user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

user_similarity_df.iloc[:5, :5], item_similarity_df.iloc[:5, :5]


# In[22]:


# Choosing a sample user:
target_user = 316

# Having the movies rated by the target user:

user_ratings = user_item_matrix.loc[target_user]
rated_movies = user_ratings.dropna().index.tolist()


# In[23]:


# ----- User-Based Filtering -----

# Getting similar users:

similar_users = user_similarity_df[target_user].sort_values(ascending=False)[1:6]  # Top 5 Neighbours

# Weighting the average of ratings from similar users for all movies:

weighted_ratings_user_based = user_item_matrix.loc[similar_users.index].T.dot(similar_users)
recommendations_user = weighted_ratings_user_based / similar_users.sum()

# Filtering out already rated movies:

recommendations_user = recommendations_user[~recommendations_user.index.isin(rated_movies)]
top_user_recs = recommendations_user.sort_values(ascending=False).head(5)


# In[24]:


# ----- Item-Based Filtering -----

# Aggregating the similarity scores for items similar to those the user liked:

movie_scores = pd.Series(dtype=float)
for movie_id in rated_movies:
    similar_scores = item_similarity_df[movie_id] * user_ratings[movie_id]
    movie_scores = movie_scores.add(similar_scores, fill_value=0)

# Normalising by similarity sums:

movie_scores = movie_scores / item_similarity_df[rated_movies].sum(axis=1)
movie_scores = movie_scores.drop(labels=rated_movies, errors='ignore')
top_item_recs = movie_scores.sort_values(ascending=False).head(5)


# In[25]:


top_user_recs, top_item_recs


# In[27]:


# Merge basic rating info with movies for easier EDA
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).reset_index()

movies_eda = pd.merge(movies, movie_stats, on='movieId', how='left')
movies_eda = pd.merge(movies_eda, tags.groupby('movieId').size().reset_index(name='tag_count'), on='movieId', how='left')
movies_eda['tag_count'] = movies_eda['tag_count'].fillna(0)


# In[28]:


movies_eda.head()


# In[31]:


pip install streamlit


# In[32]:


import streamlit as st
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv("movies-2.csv", encoding="ISO-8859-1")
    ratings = pd.read_csv("rating.csv", encoding="ISO-8859-1")
    tags = pd.read_csv("tags.csv", encoding="ISO-8859-1")

    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    movies = pd.merge(movies, movie_stats, on='movieId', how='left')
    movies = pd.merge(movies, tags.groupby('movieId').size().reset_index(name='tag_count'), on='movieId', how='left')
    movies['tag_count'] = movies['tag_count'].fillna(0)
    return movies

# Load the data
movies = load_data()

# Set up page
st.set_page_config(page_title="Movie Explorer Dashboard", layout="wide")
st.title("🎥 Movie Explorer Dashboard for 18-35 Year Olds")

# Sidebar filters
st.sidebar.header("Filter Options")
selected_genre = st.sidebar.multiselect("Select Genre(s):", options=sorted(set('|'.join(movies['genres'].dropna()).split('|'))))
min_rating = st.sidebar.slider("Minimum Average Rating:", 0.0, 5.0, 3.0, 0.1)
min_votes = st.sidebar.slider("Minimum Rating Count:", 0, int(movies['rating_count'].max()), 100)

# Filter based on user input
def genre_filter(row):
    if not selected_genre:
        return True
    genres = row['genres'].split('|') if pd.notna(row['genres']) else []
    return any(g in genres for g in selected_genre)

filtered_movies = movies[movies.apply(genre_filter, axis=1)]
filtered_movies = filtered_movies[
    (filtered_movies['avg_rating'] >= min_rating) &
    (filtered_movies['rating_count'] >= min_votes)
]

# Show top movies
st.subheader("Top Rated Movies (Filtered)")
st.dataframe(filtered_movies.sort_values(by='avg_rating', ascending=False).head(10)[[
    'title', 'genres', 'avg_rating', 'rating_count', 'tag_count']])

# Visualization 1: Ratings distribution
st.subheader("Distribution of Average Ratings")
fig, ax = plt.subplots()
movies['avg_rating'].dropna().hist(bins=20, ax=ax, color='skyblue', edgecolor='black')
ax.set_xlabel("Average Rating")
ax.set_ylabel("Number of Movies")
ax.set_title("Distribution of Ratings")
st.pyplot(fig)

# Visualization 2: Popular Genres
st.subheader("Top Genres by Volume")
genre_counts = pd.Series('|'.join(movies['genres'].dropna()).split('|')).value_counts().head(10)
st.bar_chart(genre_counts)

# Suitability explanation
st.markdown("""
### 📊 Why is this Dataset Suitable for Machine Learning?
- **High Volume**: Thousands of user ratings provide a rich interaction dataset.
- **Structured**: Each movie has metadata (genres, tags) and quantitative metrics (ratings).
- **Personalization Ready**: Ratings and tags allow for recommender system development.
- **Cold Start Coverage**: Content-based features (genres, tags) support new item handling.

This makes it ideal for building models like collaborative filtering, clustering, and hybrid recommenders in an online retail environment.
""")


# In[ ]:




