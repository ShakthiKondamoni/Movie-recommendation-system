import ast
import requests
import pandas as pd
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "YOUR_TMDB_API_KEY"  # optional, for posters

st.set_page_config(page_title="Movie Recommender", layout="wide")

ps = PorterStemmer()


def convert(text):
    result = []
    for item in ast.literal_eval(text):
        result.append(item["name"])
    return result


def convert_cast(text):
    result = []
    count = 0
    for item in ast.literal_eval(text):
        if count < 3:
            result.append(item["name"])
            count += 1
    return result


def fetch_director(text):
    result = []
    for item in ast.literal_eval(text):
        if item["job"] == "Director":
            result.append(item["name"])
    return result


def collapse(words):
    return [word.replace(" ", "") for word in words]


def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])


@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "vote_average",
            "release_date",
            "runtime",
        ]
    ]

    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["genres"] = movies["genres"].apply(collapse)
    movies["keywords"] = movies["keywords"].apply(collapse)
    movies["cast"] = movies["cast"].apply(collapse)
    movies["crew"] = movies["crew"].apply(collapse)

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[
        [
            "movie_id",
            "title",
            "tags",
            "overview",
            "genres",
            "cast",
            "crew",
            "vote_average",
            "release_date",
            "runtime",
        ]
    ]

    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())
    new_df["tags"] = new_df["tags"].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return new_df.reset_index(drop=True), similarity


def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()

        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path

    except:
        pass

    return "https://via.placeholder.com/300x450?text=No+Poster"


def recommend(movie_title, movies, similarity):
    movie_index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:
        row = movies.iloc[i[0]]
        recommendations.append(row)

    return recommendations


st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name and get similar movie recommendations with posters and details.")

movies, similarity = load_data()

movie_name = st.selectbox(
    "Choose a movie",
    movies["title"].values
)

if st.button("Recommend"):
    results = recommend(movie_name, movies, similarity)

    st.subheader("Recommended Movies")

    cols = st.columns(5)

    for idx, movie in enumerate(results):
        with cols[idx]:
            st.image(fetch_poster(movie["movie_id"]))
            st.markdown(f"### {movie['title']}")
            st.write(f"⭐ Rating: {movie['vote_average']}")
            st.write(f"📅 Release: {movie['release_date']}")
            st.write(f"⏱ Runtime: {movie['runtime']} min")

            genres = ", ".join(movie["genres"])
            cast = ", ".join(movie["cast"])
            director = ", ".join(movie["crew"])

            st.write(f"🎭 Genre: {genres}")
            st.write(f"🎬 Director: {director}")
            st.write(f"👥 Cast: {cast}")