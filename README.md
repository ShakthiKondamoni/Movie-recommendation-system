# 🎬 Movie Recommendation System

A modern movie recommendation web application built using **Python, Machine Learning, and Streamlit**. The system uses content-based filtering to recommend similar movies based on genres, keywords, cast, directors, and movie overviews. The application provides an interactive web interface where users can search for a movie and instantly receive recommendations along with movie posters and details.

---

## 🚀 Features

### 🎥 Smart Movie Recommendations

* Content-based recommendation engine
* Recommends top similar movies based on metadata similarity
* Uses genres, keywords, cast, crew, and movie descriptions

### 🌐 Interactive Web Interface

* Built with Streamlit
* User-friendly movie selection
* Instant recommendation generation
* Responsive and clean design

### 🖼️ Movie Posters & Details

* Displays movie posters using TMDB API
* Shows:

  * Movie Title
  * Rating
  * Release Date
  * Runtime
  * Genres
  * Director
  * Main Cast

### 📊 Machine Learning Pipeline

* Text preprocessing and cleaning
* Stemming using NLTK
* Feature extraction using CountVectorizer
* Similarity computation using Cosine Similarity

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### Machine Learning & NLP

* Pandas
* NumPy
* Scikit-learn
* NLTK

### APIs

* TMDB API (for movie posters and metadata)

### Visualization

* Streamlit Components

---

## 🧠 Recommendation Algorithm

The system follows a content-based filtering approach:

1. Load movie and credits datasets.
2. Merge datasets based on movie titles.
3. Extract important features:

   * Genres
   * Keywords
   * Cast
   * Director
   * Overview
4. Preprocess text and apply stemming.
5. Convert text into numerical vectors using CountVectorizer.
6. Calculate similarity using Cosine Similarity.
7. Recommend the most similar movies.

---

## 📂 Project Structure

```text
movie-recommendation/
│
├── app.py
├── movie-recommendation.ipynb
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

This project uses the TMDB 5000 Movie Dataset:

* tmdb_5000_movies.csv
* tmdb_5000_credits.csv

Dataset includes:

* Movie Titles
* Genres
* Keywords
* Cast Information
* Crew Information
* Ratings
* Release Dates
* Overviews

---

