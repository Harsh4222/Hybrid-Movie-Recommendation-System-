# 🎬 Hybrid Movie Recommendation System

A personalized movie recommendation system that combines **Content-Based Filtering**, **Collaborative Filtering (SVD)**, and a **Hybrid Strategy** — with live movie poster display via the TMDB API.

---

##  What It Does

| System | Input | Method |
|---|---|---|
| Content-Based | Movie title | Cosine similarity on movie tags |
| Collaborative Filtering | User ID | SVD matrix factorization |
| Hybrid | User ID + Movie title | Weighted combination of both |

---

##  Project Structure

```
hybrid-movie-recommendation/
│
├── HYBRID_MOVIE_RECOMMENDATION_PROJECT.ipynb
├── README.md
├── tmdb_6000_movie_dataset.csv
├── tmdb_6000_movie_credits.csv
└── tmdb_6000_movie_ratings.csv        ← Download separately (too large for GitHub)
```

---

##  Dataset

| File | Description | Size |
|---|---|---|
| `tmdb_6000_movie_dataset.csv` | Movie metadata | ~5MB |
| `tmdb_6000_movie_credits.csv` | Cast and crew | ~5MB |
| `tmdb_6000_movie_ratings.csv` | 24M+ user ratings | ~500MB |

> Download all files from https://www.kaggle.com/datasets/aayushsoni4/tmdb-6000-movie-dataset-with-ratings and place them in the project folder.

---

##  Installation

```bash
pip install pandas numpy scikit-learn nltk scikit-surprise matplotlib Pillow requests
```

Get a free TMDB API key at [themoviedb.org](https://www.themoviedb.org/settings/api) and set it in `fetch_poster()`:

```python
api_key = os.getenv('TMDB_API_KEY', 'YOUR_API_KEY_HERE')
```

---

##  How to Run

```bash
git clone https://github.com/YOUR_USERNAME/hybrid-movie-recommendation.git
cd hybrid-movie-recommendation
jupyter notebook
```

Open `HYBRID_MOVIE_RECOMMENDATION_PROJECT.ipynb` and run all cells in order.

---

##  How It Works

### 1. Content-Based Filtering
- Extracts genres, keywords, top 3 cast, director, and overview for each movie
- Combines them into a single `tags` column
- Applies lowercasing and Porter Stemming
- Vectorizes with CountVectorizer (5,000 features)
- Recommends top 5 most similar movies using Cosine Similarity

```python
recommend("Avatar")
```

---

### 2. Collaborative Filtering
- Samples 40,000 ratings from the 24M+ ratings dataset
- Trains SVD, KNNBasic, and KNNWithMeans using 3-fold cross-validation
- SVD selected as the best model
- Predicts unwatched movie ratings for a given user

```python
recommend_cf(user_id=46183, svd_model=svd, ratings_df=ratings_sample, movies_df=new_df)
```

---

### 3. Hybrid System
Combines both scores using a weighted formula:

```
Hybrid Score = α × Content_Similarity + (1 - α) × CF_Score
```

- Default `α = 0.4` → 40% content-based, 60% collaborative
- Excludes already-watched movies and the seed movie itself

```python
hybrid_recommend(
    user_id=46183,
    movie_title="Avatar",
    svd_model=svd,
    new_df=new_df,
    ratings_df=ratings_sample,
    similarity_matrix=similarity,
    top_n=5,
    alpha=0.4
)
```

---

##  Model Evaluation

| Model | RMSE | MAE |
|---|---|---|
| **SVD** | **0.993** | **0.767** |
| KNNBasic | 1.055 | 0.837 |
| KNNWithMeans | 1.106 | 0.863 |

SVD achieves the lowest error and fastest training time.

---

##  Limitations

- Cold start problem for new users or movies with no rating history
- Only 40,000 of 24M+ ratings used due to memory constraints
- Posters require a valid TMDB API key and internet connection

---

##  Future Work

- Train on full dataset using Apache Spark or Dask
- Deploy as a web app with Streamlit or Flask
- Add deep learning models (Neural Collaborative Filtering)
- Add real-time user feedback loop

---

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `NLTK` `Scikit-Surprise` `Matplotlib` `Pillow` `Requests` `Jupyter Notebook`

---

## License

MIT License — free to use and modify.

---

## Acknowledgements

- [TMDB](https://www.themoviedb.org/) for the movie database and poster API
- [Kaggle](https://www.kaggle.com/) for the dataset
- [Surprise](http://surpriselib.com/) for collaborative filtering tools
