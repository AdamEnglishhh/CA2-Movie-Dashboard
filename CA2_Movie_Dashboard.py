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


# Merge basic rating info with movies for easier EDA
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).reset_index()

movies_eda = pd.merge(movies, movie_stats, on='movieId', how='left')
movies_eda = pd.merge(movies_eda, tags.groupby('movieId').size().reset_index(name='tag_count'), on='movieId', how='left')
movies_eda['tag_count'] = movies_eda['tag_count'].fillna(0)


# In[3]:


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

