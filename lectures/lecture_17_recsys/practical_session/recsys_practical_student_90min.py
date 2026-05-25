# Auto-generated companion script for recsys_practical_student_90min.ipynb
# Keep the notebook as the source of truth.

# %% [markdown]
# # Recommender Systems Foundations - STUDENT VERSION (90 minutes)
#
# **Goal:** follow the instructor-led recommender systems demo and complete small
# implementation checkpoints.
#
# You will work with MovieLens Latest Small and move through:
#
# - popularity baselines;
# - implicit likes;
# - content-based recommendation with genres;
# - item-item collaborative filtering;
# - matrix factorization by hand;
# - leave-last-out evaluation.

# %% [markdown]
# ## Setup

# %%
import importlib.util
import os
import subprocess
import sys

IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ
required_imports = ["numpy", "pandas", "matplotlib", "seaborn", "sklearn"]
pip_names = {"sklearn": "scikit-learn"}
missing = [pkg for pkg in required_imports if importlib.util.find_spec(pkg) is None]

if missing and IN_COLAB:
    to_install = [pip_names.get(pkg, pkg) for pkg in missing]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
elif missing:
    print("Missing packages:", ", ".join(missing))
    print("Install locally with: uv sync")
else:
    print("All required packages are available.")

# %%
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 80)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"
MOVIELENS_DIR = DATA_DIR / "ml-latest-small"

# %% [markdown]
# ## 1. Download and Load MovieLens

# %%
DATA_DIR.mkdir(exist_ok=True)

if not MOVIELENS_DIR.exists():
    if not ZIP_PATH.exists():
        print("Downloading MovieLens Latest Small...")
        urlretrieve(MOVIELENS_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

ratings = pd.read_csv(MOVIELENS_DIR / "ratings.csv")
movies = pd.read_csv(MOVIELENS_DIR / "movies.csv")
tags = pd.read_csv(MOVIELENS_DIR / "tags.csv")

ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
tags["datetime"] = pd.to_datetime(tags["timestamp"], unit="s")

ratings.head()

# %%
movies.head()

# %% [markdown]
# ## 2. Inspect the RecSys Data
#
# TODO: calculate:
#
# - number of users;
# - number of movies with at least one rating;
# - number of ratings;
# - sparsity of the user-item matrix.
#
# Formula:
#
# $$sparsity = 1 - \frac{\text{observed ratings}}{\text{users} \times \text{movies}}$$
#
# It tells us what share of possible user-movie pairs is missing.

# %%
n_users = ratings["userId"].nunique()
n_movies = ratings["movieId"].nunique()
n_ratings = len(ratings)
sparsity = 1 - n_ratings / (n_users * n_movies)

pd.DataFrame(
    {
        "metric": ["users", "movies_with_ratings", "ratings", "sparsity"],
        "value": [n_users, n_movies, n_ratings, round(sparsity, 4)],
    }
)

# %%
ratings["rating"].value_counts().sort_index()

# %%
most_rated = (
    ratings.groupby("movieId")
    .size()
    .reset_index(name="num_ratings")
    .merge(movies, on="movieId")
    .sort_values("num_ratings", ascending=False)
)

most_rated.head(10)

# %% [markdown]
# ## 3. Popularity Baseline
#
# TODO: explain why average rating alone can be unstable.
#
# `groupby("movieId").agg(...)` calculates one row per movie:
#
# - `num_ratings`: how many ratings the movie received;
# - `mean_rating`: the movie's average star rating.

# %%
popular_by_count = (
    ratings.groupby("movieId")
    .agg(num_ratings=("rating", "count"), mean_rating=("rating", "mean"))
    .reset_index()
    .merge(movies, on="movieId")
)

popular_by_count.sort_values("num_ratings", ascending=False).head(10)

# %%
popular_by_count.sort_values("mean_rating", ascending=False).head(10)[
    ["movieId", "title", "num_ratings", "mean_rating"]
]

# %% [markdown]
# Bayesian score:
#
# $$score_i =
# \frac{n_i}{n_i + m} \bar{r}_i +
# \frac{m}{n_i + m} C$$
#
# If a movie has many ratings, its own mean matters more. If it has few ratings,
# the global mean matters more.

# %% [markdown]
# Before using the recommender function, compare a few high-count and low-count
# movies. This is the main teaching point: low-count movies can have very high
# average ratings, but Bayesian scoring does not fully trust tiny samples.

# %%
bayesian_demo = popular_by_count.copy()
global_mean = ratings["rating"].mean()
m = 50
bayesian_demo["bayesian_score"] = (
    bayesian_demo["num_ratings"] / (bayesian_demo["num_ratings"] + m) * bayesian_demo["mean_rating"]
    + m / (bayesian_demo["num_ratings"] + m) * global_mean
)
bayesian_demo["own_mean_weight"] = bayesian_demo["num_ratings"] / (
    bayesian_demo["num_ratings"] + m
)
bayesian_demo["global_mean_weight"] = m / (bayesian_demo["num_ratings"] + m)

many_rating_examples = bayesian_demo.sort_values("num_ratings", ascending=False).head(3)
few_rating_examples = (
    bayesian_demo[bayesian_demo["num_ratings"] <= 2]
    .sort_values(["mean_rating", "bayesian_score"], ascending=False)
    .head(3)
)

bayesian_comparison = pd.concat(
    {
        "many ratings": many_rating_examples,
        "few ratings": few_rating_examples,
    },
    names=["example_type"],
).reset_index(level=0)

bayesian_comparison[
    [
        "example_type",
        "title",
        "num_ratings",
        "mean_rating",
        "own_mean_weight",
        "global_mean_weight",
        "bayesian_score",
    ]
].sort_values(["example_type", "mean_rating"], ascending=[True, False])

# %%
def recommend_popular(interactions=ratings, top_k=10):
    table = (
        interactions.groupby("movieId")
        .agg(num_ratings=("rating", "count"), mean_rating=("rating", "mean"))
        .reset_index()
        .merge(movies, on="movieId")
    )

    global_mean = interactions["rating"].mean()
    m = 50
    table["bayesian_score"] = (
        table["num_ratings"] / (table["num_ratings"] + m) * table["mean_rating"]
        + m / (table["num_ratings"] + m) * global_mean
    )

    return table.sort_values("bayesian_score", ascending=False).head(top_k)[
        ["movieId", "title", "genres", "num_ratings", "mean_rating", "bayesian_score"]
    ]


recommend_popular(top_k=10)

# %% [markdown]
# ## 4. Convert Ratings to Implicit Likes
#
# TODO: change the threshold from `4.0` to `3.5` later and compare the number of likes.
#
# The next line means:
#
# ```text
# rating >= 4.0  -> liked = 1
# rating < 4.0   -> liked = 0
# ```

# %%
ratings["liked"] = (ratings["rating"] >= 4.0).astype(int)
ratings["liked"].value_counts(normalize=True).rename("share").to_frame()

# %% [markdown]
# ## 5. Content-Based Recommendation with Genres
#
# First we turn genre strings into a movie-by-genre matrix.
#
# Example:
#
# ```text
# movieId   Action   Comedy   Drama   Sci-Fi
# 1            0        1       0        0
# 32           0        0       0        1
# 260          1        0       0        1
# ```

# %%
movies = movies.copy()
movies["genres_list"] = movies["genres"].str.split("|")
movies["genres_list"] = movies["genres_list"].apply(
    lambda values: [] if values == ["(no genres listed)"] else values
)

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies["genres_list"])

genre_features = pd.DataFrame(
    genre_matrix,
    columns=mlb.classes_,
    index=movies["movieId"],
)

genre_features.head()

# %% [markdown]
# What this line inside `build_user_profile` means:
#
# ```python
# genre_features.loc[liked_movie_ids].mean(axis=0)
# ```
#
# - `.loc[liked_movie_ids]` selects the movies liked by this user;
# - `mean(axis=0)` averages down the rows, one genre column at a time;
# - the result is the user's genre profile.
#
# Example:
#
# ```text
# liked movies      Action   Comedy   Drama   Sci-Fi
# movie 1              0        1       0        0
# movie 32             0        0       0        1
# movie 260            1        0       0        1
#
# user profile      0.33     0.33    0.00     0.67
# ```

# %%
def build_user_profile(user_id, interactions=ratings, like_threshold=4.0):
    user_likes = interactions[
        (interactions["userId"] == user_id) & (interactions["rating"] >= like_threshold)
    ]
    liked_movie_ids = [
        movie_id for movie_id in user_likes["movieId"].values if movie_id in genre_features.index
    ]

    if not liked_movie_ids:
        return None

    return genre_features.loc[liked_movie_ids].mean(axis=0)


build_user_profile(user_id=1).sort_values(ascending=False).head(10)

# %% [markdown]
# `cosine_similarity` compares each movie's genre vector with the user's genre
# profile:
#
# $$cosine(a,b) = \frac{a \cdot b}{||a||\,||b||}$$
#
# Higher score means the movie's genres match the user's past liked genres better.

# %%
def recommend_content_based(user_id, interactions=ratings, top_k=10):
    profile = build_user_profile(user_id, interactions=interactions)

    if profile is None:
        return recommend_popular(interactions=interactions, top_k=top_k)

    scores = cosine_similarity(genre_features.values, profile.values.reshape(1, -1)).ravel()
    recs = movies.copy()
    recs["content_score"] = scores

    seen_movies = set(interactions[interactions["userId"] == user_id]["movieId"])
    recs = recs[~recs["movieId"].isin(seen_movies)]

    return recs.sort_values("content_score", ascending=False).head(top_k)[
        ["movieId", "title", "genres", "content_score"]
    ]


recommend_content_based(user_id=1, top_k=10)

# %% [markdown]
# ## 6. Item-Item Collaborative Filtering
#
# `pivot_table` turns the long ratings table into a user-item matrix:
#
# ```text
# rows = users
# columns = movies
# values = ratings
# ```
#
# Missing values are filled with `0` only so cosine similarity can be computed.
# The `0` means missing/unknown, not dislike.

# %%
user_item = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
).fillna(0)

user_item.shape

# %% [markdown]
# To compare movies with movies:
#
# - transpose the matrix so each row is a movie;
# - calculate cosine similarity between movie rows;
# - store the result as a movie-by-movie similarity table.

# %%
item_user = user_item.T
item_similarity = cosine_similarity(item_user)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_user.index,
    columns=item_user.index,
)

# %% [markdown]
# `similar_movies(movie_id=1)` sorts the similarity scores for movie `1`.
# The movie itself is dropped because its similarity to itself is always 1.

# %%
def similar_movies(movie_id, similarity_df, top_k=10):
    if movie_id not in similarity_df.index:
        return pd.DataFrame()

    similar_ids = (
        similarity_df[movie_id].sort_values(ascending=False).drop(movie_id).head(top_k).index
    )
    result = movies[movies["movieId"].isin(similar_ids)][["movieId", "title", "genres"]].copy()
    result = result.merge(
        similarity_df.loc[similar_ids, movie_id].rename("similarity"),
        left_on="movieId",
        right_index=True,
    )
    return result.sort_values("similarity", ascending=False)


similar_movies(movie_id=1, similarity_df=item_similarity_df, top_k=10)

# %% [markdown]
# Main calculation inside item-item CF:
#
# ```python
# scores = similarity_df[liked_movie_ids].sum(axis=1)
# ```
#
# This means:
#
# ```text
# score(candidate movie)
# = similarity(candidate, liked movie 1)
# + similarity(candidate, liked movie 2)
# + ...
# ```
#
# Here `axis=1` means summing across the selected liked-movie columns for each
# candidate movie row.

# %%
def recommend_item_item_cf(user_id, interactions, similarity_df, top_k=10, like_threshold=4.0):
    user_likes = interactions[
        (interactions["userId"] == user_id) & (interactions["rating"] >= like_threshold)
    ]
    liked_movie_ids = [
        movie_id for movie_id in user_likes["movieId"].values if movie_id in similarity_df.columns
    ]

    if not liked_movie_ids:
        return recommend_popular(interactions=interactions, top_k=top_k)

    scores = similarity_df[liked_movie_ids].sum(axis=1)
    seen_movies = set(interactions[interactions["userId"] == user_id]["movieId"])
    scores = scores.drop(index=list(seen_movies), errors="ignore")

    rec_ids = scores.sort_values(ascending=False).head(top_k).index
    recs = movies[movies["movieId"].isin(rec_ids)].copy()
    recs = recs.merge(scores.rename("cf_score"), left_on="movieId", right_index=True)
    return recs.sort_values("cf_score", ascending=False)[
        ["movieId", "title", "genres", "cf_score"]
    ]


recommend_item_item_cf(1, interactions=ratings, similarity_df=item_similarity_df, top_k=10)

# %% [markdown]
# ## 7. Matrix Factorization by Hand
#
# Formula:
#
# $$R \approx P Q^T$$
#
# $$\hat{r}_{ui} = p_u \cdot q_i = \sum_f p_{uf}q_{if}$$

# %% [markdown]
# ### Matrix Shapes, Rank, and SVD Connection
#
# Suppose:
#
# - `m` = number of users;
# - `n` = number of items;
# - `k` = number of latent factors.
#
# Then the shapes are:
#
# $$R \in \mathbb{R}^{m \times n}$$
# $$P \in \mathbb{R}^{m \times k}$$
# $$Q \in \mathbb{R}^{n \times k}$$
# $$Q^T \in \mathbb{R}^{k \times n}$$
# $$P Q^T \in \mathbb{R}^{m \times n}$$
#
# Shape check:
#
# ```text
# (m x k) @ (k x n) -> (m x n)
# ```
#
# Rank intuition:
#
# $$rank(P Q^T) \le k$$
#
# Any fully observed real matrix has an SVD. The constraint here is about
# whether a chosen small `k` can reconstruct the matrix exactly.
#
# So an exact reconstruction of a fully observed matrix,
#
# $$R = P Q^T$$
#
# requires:
#
# $$k \ge rank(R)$$
#
# In recommender systems, we usually choose `k` much smaller than the full matrix
# size, so we learn a low-rank approximation:
#
# $$R \approx P Q^T$$
#
# Important caveat:
#
# We do not treat missing ratings as zeros. We learn the factors from observed
# ratings only.
#
# Connection to eigenvalues/eigenvectors:
#
# - Eigenvalue decomposition is for square matrices.
# - User-item matrices are usually rectangular.
# - Any fully observed real matrix has the related classical decomposition SVD:
#
# $$R = U \Sigma V^T$$
#
# - `U` and `V` are related to eigenvectors of `R R^T` and `R^T R`.
# - Singular values are square roots of eigenvalues of those square matrices.
#
# For this practical, the key operation remains the dot product:
#
# $$\hat{r}_{ui} = p_u \cdot q_i$$

# %%
user_factors = pd.DataFrame(
    {"factor_1": [0.9, 0.2], "factor_2": [0.1, 0.8]},
    index=["Alice", "Bob"],
)

item_factors = pd.DataFrame(
    {"factor_1": [5.0, 1.0, 3.0], "factor_2": [1.0, 5.0, 3.0]},
    index=["Action movie", "Romance movie", "Balanced movie"],
)

display(user_factors)
display(item_factors)

# %% [markdown]
# ### Matrix Multiplication View
#
# The matrix multiplication picture is:
#
# ```text
# user factors      item factors transposed      predicted ratings
#     P        @             Q^T            =          R_hat
#  (2 x 2)                 (2 x 3)                    (2 x 3)
# ```
#
# Use the tables below to connect the formulas to the numbers.

# %%
q_transpose = item_factors.T
preview_r_hat = user_factors @ q_transpose

print("P: user factors, shape", user_factors.shape)
display(user_factors)

print("Q^T: item factors transposed, shape", q_transpose.shape)
display(q_transpose)

print("R_hat = P @ Q^T, shape", preview_r_hat.shape)
display(preview_r_hat)

# %% [markdown]
# One cell in `R_hat` is one dot product. This table shows the contribution of
# each latent factor to Alice's predicted score for the Action movie.

# %%
alice_action_components = pd.DataFrame(
    {
        "Alice factor value": user_factors.loc["Alice"],
        "Action item value": item_factors.loc["Action movie"],
    }
)
alice_action_components["component product"] = (
    alice_action_components["Alice factor value"]
    * alice_action_components["Action item value"]
)

display(alice_action_components)
print("Dot product:", alice_action_components["component product"].sum())

# %% [markdown]
# Read the two tables as matrices:
#
# $$P =
# \begin{bmatrix}
# 0.9 & 0.1 \\
# 0.2 & 0.8
# \end{bmatrix}$$
#
# $$Q^T =
# \begin{bmatrix}
# 5.0 & 1.0 & 3.0 \\
# 1.0 & 5.0 & 3.0
# \end{bmatrix}$$
#
# Matrix factorization predicts:
#
# $$\hat{R} = P Q^T$$
#
# Every cell in $\hat{R}$ is one dot product between one user vector and one item
# vector.
#
# Teacher demo:
#
# $$\hat{r}_{Alice,Action} = 0.9 \cdot 5.0 + 0.1 \cdot 1.0 = 4.6$$
#
# TODO by hand:
#
# Fill in the remaining cells:
#
# $$\hat{r}_{Alice,Romance} = 0.9 \cdot 1.0 + 0.1 \cdot 5.0 = ?$$
# $$\hat{r}_{Alice,Balanced} = 0.9 \cdot 3.0 + 0.1 \cdot 3.0 = ?$$
# $$\hat{r}_{Bob,Action} = 0.2 \cdot 5.0 + 0.8 \cdot 1.0 = ?$$
# $$\hat{r}_{Bob,Romance} = 0.2 \cdot 1.0 + 0.8 \cdot 5.0 = ?$$
# $$\hat{r}_{Bob,Balanced} = 0.2 \cdot 3.0 + 0.8 \cdot 3.0 = ?$$
#
# Expected final matrix:
#
# $$\hat{R} =
# \begin{bmatrix}
# 4.6 & 1.4 & 3.0 \\
# 1.8 & 4.2 & 3.0
# \end{bmatrix}$$

# %%
manual_r_hat = pd.DataFrame(
    [
        [4.6, 1.4, 3.0],
        [1.8, 4.2, 3.0],
    ],
    index=user_factors.index,
    columns=item_factors.index,
)

manual_r_hat

# %% [markdown]
# TODO by hand:
#
# Calculate:
#
# - Alice x Action movie;
# - Alice x Romance movie;
# - Bob x Action movie;
# - Bob x Romance movie.

# %%
predicted_ratings = user_factors @ item_factors.T
predicted_ratings

# %%
np.allclose(manual_r_hat, predicted_ratings)

# %% [markdown]
# ### How Matrix Factorization Makes a Recommendation
#
# Matrix factorization is useful because it fills in plausible scores for missing
# user-item pairs.
#
# Suppose we observed only:
#
# ```text
#                Action movie   Romance movie   Balanced movie
# Alice              5.0              ?                ?
# Bob                 ?              5.0               ?
# ```
#
# `R_hat = P @ Q^T` predicts every cell:
#
# ```text
#                Action movie   Romance movie   Balanced movie
# Alice              4.6             1.4              3.0
# Bob                1.8             4.2              3.0
# ```
#
# For Alice, remove the movie she already saw (`Action movie`) and rank the
# remaining movies by predicted score.

# %%
observed_toy_ratings = pd.DataFrame(
    [
        [5.0, np.nan, np.nan],
        [np.nan, 5.0, np.nan],
    ],
    index=["Alice", "Bob"],
    columns=["Action movie", "Romance movie", "Balanced movie"],
)

display(observed_toy_ratings)
display(predicted_ratings)

alice_seen = observed_toy_ratings.loc["Alice"].dropna().index
alice_unseen_scores = predicted_ratings.loc["Alice"].drop(index=alice_seen)

display(alice_unseen_scores.sort_values(ascending=False).to_frame("predicted_score"))
print("Recommendation for Alice:", alice_unseen_scores.idxmax())

# %% [markdown]
# TODO:
#
# Charlie has vector `[0.6, 0.4]`.
#
# Calculate:
#
# - Charlie x Action movie;
# - Charlie x Romance movie;
# - Charlie x Balanced movie.
#
# Which item should be ranked first?

# %%
charlie = np.array([0.6, 0.4])
pd.Series(
    item_factors.to_numpy() @ charlie,
    index=item_factors.index,
    name="Charlie predicted rating",
).sort_values(ascending=False)

# %% [markdown]
# ## 8. Leave-Last-Out Evaluation
#
# Split idea:
#
# - for each user, hide the last liked movie as `test`;
# - use only earlier interactions as `train`;
# - check whether the recommender can recover the hidden movie.

# %%
ratings_sorted = ratings.sort_values("timestamp")
train_parts = []
test_parts = []

for _, user_data in ratings_sorted.groupby("userId"):
    user_data = user_data.sort_values("timestamp")
    liked_data = user_data[user_data["liked"] == 1]

    if len(liked_data) < 2:
        train_parts.append(user_data)
        continue

    test_item = liked_data.tail(1)
    test_timestamp = test_item["timestamp"].iloc[0]
    train_user = user_data[user_data["timestamp"] < test_timestamp]

    if len(train_user) == 0:
        train_parts.append(user_data)
        continue

    train_parts.append(train_user)
    test_parts.append(test_item)

train = pd.concat(train_parts).reset_index(drop=True)
test = pd.concat(test_parts).reset_index(drop=True)

train.shape, test.shape

# %% [markdown]
# Rebuild the user-item matrix and item similarities on `train` only.
#
# This avoids using the hidden test movie when building the recommender.

# %%
train_user_item = train.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
).fillna(0)

train_item_user = train_user_item.T
train_item_similarity = cosine_similarity(train_item_user)

train_item_similarity_df = pd.DataFrame(
    train_item_similarity,
    index=train_item_user.index,
    columns=train_item_user.index,
)

# %% [markdown]
# Hit@10 for one user:
#
# ```text
# 1 if the hidden movie is in the first 10 recommendations
# 0 otherwise
# ```
#
# In code:
#
# ```python
# int(example_test_movie_id in recommended_ids[:10])
# ```

example_user_id = test.iloc[0]["userId"]
example_test_movie_id = test.iloc[0]["movieId"]

recs = recommend_item_item_cf(
    user_id=example_user_id,
    interactions=train,
    similarity_df=train_item_similarity_df,
    top_k=10,
)
recommended_ids = recs["movieId"].tolist()

print("Hidden test movie:", movies.loc[movies["movieId"] == example_test_movie_id, "title"].values)
print("Hit@10:", int(example_test_movie_id in recommended_ids[:10]))
recs

# %% [markdown]
# This quick loop repeats the same 0/1 check for the first 100 test users and
# averages the hits.
#
# It is only a classroom sanity check. The final RecSys metrics below are
# computed on the full test set.

# %%
hits = []

for _, row in test.head(100).iterrows():
    recs = recommend_item_item_cf(
        user_id=row["userId"],
        interactions=train,
        similarity_df=train_item_similarity_df,
        top_k=10,
    )
    hits.append(int(row["movieId"] in recs["movieId"].tolist()[:10]))

print("HitRate@10:", np.mean(hits))

# %% [markdown]
# ## 9. Worked Example: One User, Three Approaches
#
# We now compare three recommendation lists for one user:
#
# - popularity;
# - content-based;
# - item-item collaborative filtering.
#
# The recommender sees only `train`. The hidden movie from `test` is used only
# for metric calculation.

# %%
demo_user_id = test.iloc[0]["userId"]
demo_test_movie_id = test.iloc[0]["movieId"]
demo_test_title = movies.loc[movies["movieId"] == demo_test_movie_id, "title"].iloc[0]

print("Demo user:", demo_user_id)
print("Hidden test movie:", demo_test_title)

demo_user_history = (
    train[train["userId"] == demo_user_id]
    .sort_values("timestamp")
    .tail(10)
    .merge(movies, on="movieId")
)

demo_user_history[["datetime", "title", "genres", "rating", "liked"]]

# %% [markdown]
# Popularity recommendations for this user:

# %%
demo_seen_movies = set(train[train["userId"] == demo_user_id]["movieId"])

demo_popularity_recs = recommend_popular(interactions=train, top_k=200)
demo_popularity_recs = demo_popularity_recs[
    ~demo_popularity_recs["movieId"].isin(demo_seen_movies)
].head(10)

demo_popularity_recs

# %% [markdown]
# Content-based recommendations for this user:

# %%
demo_content_recs = recommend_content_based(
    user_id=demo_user_id,
    interactions=train,
    top_k=10,
)

demo_content_recs

# %% [markdown]
# Item-item CF recommendations for this user:

# %%
demo_cf_recs = recommend_item_item_cf(
    user_id=demo_user_id,
    interactions=train,
    similarity_df=train_item_similarity_df,
    top_k=10,
)

demo_cf_recs

# %% [markdown]
# Metrics for this one user:
#
# - `Hit@10 = 1` if the hidden movie appears in the top 10, else `0`;
# - `Rank@10` is the position of the hidden movie if found;
# - `RR@10 = 1 / rank` if found, else `0`.
# - `Precision@10 = Hit@10 / 10` because there is one hidden relevant item.
# - `Recall@10 = Hit@10` because each test row has one hidden relevant item.
# - `NDCG@10 = 1 / log2(rank + 1)` if found, else `0`.

# %%
metric_rows = []

for approach, recs in [
    ("Popularity", demo_popularity_recs),
    ("Content-based", demo_content_recs),
    ("Item-item CF", demo_cf_recs),
]:
    recommended_ids = recs["movieId"].tolist()
    hit = int(demo_test_movie_id in recommended_ids[:10])
    rank = recommended_ids.index(demo_test_movie_id) + 1 if hit else np.nan
    reciprocal_rank = 1 / rank if hit else 0.0
    ndcg = 1 / np.log2(rank + 1) if hit else 0.0

    metric_rows.append(
        {
            "approach": approach,
            "hidden_movie": demo_test_title,
            "hit@10": hit,
            "precision@10": hit / 10,
            "recall@10": hit,
            "rank@10": rank,
            "rr@10": reciprocal_rank,
            "ndcg@10": ndcg,
        }
    )

pd.DataFrame(metric_rows)

# %% [markdown]
# Full test set metrics:
#
# Now calculate RecSys metrics on the full leave-last-out `test` set.
#
# - `HitRate@10`: average of 0/1 hits;
# - `Precision@10`: average of `hit / 10`;
# - `Recall@10`: same as HitRate here because each user has one hidden item;
# - `MRR@10`: average reciprocal rank;
# - `NDCG@10`: gives more credit when the hidden item is closer to rank 1.

# %%
K = 10
full_metric_rows = []

train_seen_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()
train_liked_by_user = (
    train[train["rating"] >= 4.0]
    .groupby("userId")["movieId"]
    .apply(list)
    .to_dict()
)

full_popularity_recs = recommend_popular(interactions=train, top_k=len(movies))

# %% [markdown]
# Popularity metrics on the full test set.

# %%
hits = []
precisions = []
recalls = []
reciprocal_ranks = []
ndcgs = []

for _, row in test.iterrows():
    user_id = row["userId"]
    test_movie_id = row["movieId"]

    seen_movies = train_seen_by_user.get(user_id, set())
    recs = full_popularity_recs[~full_popularity_recs["movieId"].isin(seen_movies)].head(K)
    recommended_ids = recs["movieId"].tolist()

    hit = int(test_movie_id in recommended_ids)
    rank = recommended_ids.index(test_movie_id) + 1 if hit else None

    hits.append(hit)
    precisions.append(hit / K)
    recalls.append(hit)
    reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
    ndcgs.append(1 / np.log2(rank + 1) if rank is not None else 0.0)

full_metric_rows.append(
    {
        "approach": "Popularity",
        "users_evaluated": len(hits),
        "HitRate@10": np.mean(hits),
        "Precision@10": np.mean(precisions),
        "Recall@10": np.mean(recalls),
        "MRR@10": np.mean(reciprocal_ranks),
        "NDCG@10": np.mean(ndcgs),
    }
)

pd.DataFrame([full_metric_rows[-1]])

# %% [markdown]
# Content-based metrics on the full test set.
#
# This reuses the same cosine-similarity idea as `recommend_content_based`, but
# keeps the full test set quick by reusing precomputed arrays.

# %%
genre_values = genre_features.values
genre_norms = np.linalg.norm(genre_values, axis=1)
genre_movie_ids = genre_features.index.to_numpy()
genre_movie_id_to_position = {
    movie_id: position for position, movie_id in enumerate(genre_movie_ids)
}

# %%
hits = []
precisions = []
recalls = []
reciprocal_ranks = []
ndcgs = []

for _, row in test.iterrows():
    user_id = row["userId"]
    test_movie_id = row["movieId"]

    seen_movies = train_seen_by_user.get(user_id, set())
    liked_positions = [
        genre_movie_id_to_position[movie_id]
        for movie_id in train_liked_by_user.get(user_id, [])
        if movie_id in genre_movie_id_to_position
    ]

    if liked_positions:
        profile = genre_values[liked_positions].mean(axis=0)
        profile_norm = np.linalg.norm(profile)
        scores = genre_values @ profile / (genre_norms * profile_norm + 1e-12)
        scores = scores.copy()
        seen_positions = [
            genre_movie_id_to_position[movie_id]
            for movie_id in seen_movies
            if movie_id in genre_movie_id_to_position
        ]
        scores[seen_positions] = -np.inf
        top_positions = np.argsort(scores)[::-1][:K]
        recommended_ids = genre_movie_ids[top_positions].tolist()
    else:
        recommended_ids = [
            movie_id
            for movie_id in full_popularity_recs["movieId"].tolist()
            if movie_id not in seen_movies
        ][:K]

    hit = int(test_movie_id in recommended_ids)
    rank = recommended_ids.index(test_movie_id) + 1 if hit else None

    hits.append(hit)
    precisions.append(hit / K)
    recalls.append(hit)
    reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
    ndcgs.append(1 / np.log2(rank + 1) if rank is not None else 0.0)

full_metric_rows.append(
    {
        "approach": "Content-based",
        "users_evaluated": len(hits),
        "HitRate@10": np.mean(hits),
        "Precision@10": np.mean(precisions),
        "Recall@10": np.mean(recalls),
        "MRR@10": np.mean(reciprocal_ranks),
        "NDCG@10": np.mean(ndcgs),
    }
)

pd.DataFrame([full_metric_rows[-1]])

# %% [markdown]
# Item-item collaborative filtering metrics on the full test set.
#
# This is the same scoring idea as `recommend_item_item_cf`:
#
# ```python
# score(candidate) = sum(similarity(candidate, each_liked_movie))
# ```
#
# NumPy arrays make the full test set evaluation fast enough for a live demo.

# %%
train_similarity_values = train_item_similarity_df.values
train_similarity_movie_ids = train_item_similarity_df.index.to_numpy()
train_similarity_movie_id_to_position = {
    movie_id: position for position, movie_id in enumerate(train_similarity_movie_ids)
}

# %%
hits = []
precisions = []
recalls = []
reciprocal_ranks = []
ndcgs = []

for _, row in test.iterrows():
    user_id = row["userId"]
    test_movie_id = row["movieId"]

    seen_movies = train_seen_by_user.get(user_id, set())
    liked_positions = [
        train_similarity_movie_id_to_position[movie_id]
        for movie_id in train_liked_by_user.get(user_id, [])
        if movie_id in train_similarity_movie_id_to_position
    ]

    if liked_positions:
        scores = train_similarity_values[:, liked_positions].sum(axis=1)
        scores = scores.copy()
        seen_positions = [
            train_similarity_movie_id_to_position[movie_id]
            for movie_id in seen_movies
            if movie_id in train_similarity_movie_id_to_position
        ]
        scores[seen_positions] = -np.inf
        top_positions = np.argsort(scores)[::-1][:K]
        recommended_ids = train_similarity_movie_ids[top_positions].tolist()
    else:
        recommended_ids = [
            movie_id
            for movie_id in full_popularity_recs["movieId"].tolist()
            if movie_id not in seen_movies
        ][:K]

    hit = int(test_movie_id in recommended_ids)
    rank = recommended_ids.index(test_movie_id) + 1 if hit else None

    hits.append(hit)
    precisions.append(hit / K)
    recalls.append(hit)
    reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
    ndcgs.append(1 / np.log2(rank + 1) if rank is not None else 0.0)

full_metric_rows.append(
    {
        "approach": "Item-item CF",
        "users_evaluated": len(hits),
        "HitRate@10": np.mean(hits),
        "Precision@10": np.mean(precisions),
        "Recall@10": np.mean(recalls),
        "MRR@10": np.mean(reciprocal_ranks),
        "NDCG@10": np.mean(ndcgs),
    }
)

pd.DataFrame([full_metric_rows[-1]])

# %% [markdown]
# Final comparison table for the full test set.

# %%
pd.DataFrame(full_metric_rows).sort_values("HitRate@10", ascending=False)

# %% [markdown]
# ## Summary
#
# - Popularity recommends what many users rated or liked.
# - Content-based recommendation uses item metadata.
# - Collaborative filtering uses user behavior.
# - Matrix factorization scores users and items through dot products.
# - Leave-last-out checks whether a hidden future interaction appears in top-K.
