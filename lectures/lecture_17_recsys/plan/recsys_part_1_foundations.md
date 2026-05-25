# Recommender Systems — Part 1

## From Popularity Baselines to Collaborative Filtering

**Format:** 90-minute instructor-led demo with short discussion checkpoints  
**Environment:** Google Colab  
**Dataset:** MovieLens Latest Small  
**Student level:** completed introductory classical ML course and basic deep learning  
**Main goal:** students understand the core logic of recommender systems by following a teacher-led implementation of several simple recommenders from scratch.

---

# 1. Learning Objectives

By the end of this session, students should be able to explain and follow the implementation of:

1. What a recommender system is.
2. How RecSys differs from standard supervised ML.
3. What users, items, interactions, explicit feedback, and implicit feedback are.
4. Why missing interactions are not necessarily negative labels.
5. How to build a popularity-based recommender.
6. How to build a content-based recommender using item metadata.
7. How to build an item-item collaborative filtering recommender.
8. What matrix factorization means in recommender systems.
9. How to manually calculate a simple latent-factor prediction.
10. How to perform a simple time-based / leave-last-out evaluation.
11. How to calculate simple top-K recommendation metrics such as HitRate@K.

---

# 2. Core Teaching Story

The session should follow a simple progression:

```text
1. Recommend the same popular items to everyone.
2. Use item metadata to recommend similar items.
3. Use behavior of other users to recommend items.
4. Compress user-item behavior into latent factors.
5. Compare these approaches with simple top-K evaluation.
```

This gives students a clear conceptual map:

```text
Popularity baseline
    ↓
Content-based recommendation
    ↓
Collaborative filtering
    ↓
Matrix factorization intuition
    ↓
Top-K evaluation
```

---

# 3. Why MovieLens Latest Small

Use **MovieLens Latest Small** because it is ideal for a first RecSys practice.

It contains:

- approximately 100,000 ratings;
- approximately 9,000 movies;
- approximately 600 users;
- movie genres;
- timestamps;
- optional tags;
- a clean CSV structure.

It works well in Google Colab because:

- it is small;
- it downloads quickly;
- it does not need Kaggle credentials;
- it does not need GPU;
- it is clean enough for teaching;
- it supports both explicit and implicit feedback examples.

Download URL:

```text
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
```

---

# 4. 90-Minute Session Plan

## 0–10 min — Motivation and Problem Framing

Explain with familiar examples:

- Netflix: which movie should we recommend?
- YouTube: which video should be shown next?
- Spotify: which song should play next?
- Amazon/eBay: which products should appear on the page?
- LinkedIn: which people or jobs should be recommended?

Key point:

```text
In classical ML, we often predict one target y for one object.
In RecSys, we usually rank many candidate items for a specific user and context.
```

Basic formulation:

```text
Given:
- user u
- items i1, i2, ..., in
- optional context c

Goal:
Rank items by expected relevance or utility for the user.
```

Important statement:

```text
A recommender system is not only a prediction model.
It is a decision system that decides what limited set of items the user will see.
```

---

## 10–20 min — Dataset and RecSys Data Structure

Explain the main entities:

```text
Users
Items
Interactions
Context
```

For MovieLens:

```text
userId   -> user
movieId  -> item
rating   -> explicit feedback
timestamp -> interaction time
genres   -> item metadata
```

Important distinction:

## Explicit feedback

Examples:

- rating 1–5;
- like/dislike;
- thumbs up/down;
- review score.

## Implicit feedback

Examples:

- click;
- view;
- watch;
- add to cart;
- purchase;
- skip;
- hide;
- dwell time.

Important teaching point:

```text
No interaction does not mean dislike.
```

A user may not interact with an item because:

- they never saw it;
- it was shown too low in the list;
- they saw it but were not interested;
- they were interested but did not have time;
- they already consumed it elsewhere;
- the price, timing, or context was wrong.

---

## 20–35 min — Popularity Baseline

Goal:

```text
Build the simplest recommender: recommend popular movies to everyone.
```

Teaching point:

```text
Always start RecSys work with simple baselines.
A good baseline is not embarrassing. It is necessary.
```

Discuss why popularity is useful:

- it is simple;
- it is fast;
- it works for new users;
- it is often surprisingly strong.

Discuss limitations:

- no personalization;
- popularity bias;
- weak long-tail discovery;
- everyone receives almost the same list.

---

## 35–55 min — Content-Based Recommendation

Goal:

```text
Recommend items similar to what the user liked before.
```

For MovieLens, use genres as item metadata.

Example:

```text
If a user liked many Action and Sci-Fi movies,
recommend more movies with Action and Sci-Fi genres.
```

Conceptual method:

```text
1. Convert movie genres into vectors.
2. Build a user profile as the average vector of movies the user liked.
3. Score unseen movies by cosine similarity to the user profile.
4. Recommend highest-scoring unseen movies.
```

Formula:

```text
user_profile = average(vectors of liked items)
score(user, item) = cosine_similarity(user_profile, item_vector)
```

Pros:

- works for new items if metadata exists;
- explainable;
- does not need other users;
- useful in cold-start item scenarios.

Cons:

- can over-specialize;
- depends on metadata quality;
- may fail to discover surprising but relevant items.

---

## 55–70 min — Item-Item Collaborative Filtering

Goal:

```text
Recommend movies based on behavior patterns of users.
```

Basic intuition:

```text
Movies are similar if they are liked or rated by similar users.
```

Example:

```text
Users who liked Movie A also liked Movie B.
Therefore, if a new user likes Movie A, recommend Movie B.
```

Important contrast:

```text
Content-based recommendation uses item metadata.
Collaborative filtering uses user behavior.
```

For MovieLens:

```text
Build a user-item rating matrix.
Transpose it into an item-user matrix.
Calculate cosine similarity between item vectors.
```

---

## 70–80 min — Matrix Factorization Intuition

Goal:

```text
Show how recommender systems can represent users and items with a small number
of hidden factors and use dot products to predict missing ratings.
```

Core idea:

```text
The original user-item matrix is large and sparse.
Matrix factorization approximates it with two smaller matrices:

R ≈ P Q^T
```

Where:

```text
R = user-item rating matrix
P = user-factor matrix
Q = item-factor matrix
k = number of latent factors
```

Prediction formula:

```text
r_hat(u, i) = p_u · q_i
            = p_u1 q_i1 + p_u2 q_i2 + ... + p_uk q_ik
```

Matrix shape constraints:

```text
If:
R is m x n
P is m x k
Q is n x k

Then:
Q^T is k x n
P Q^T is (m x k) @ (k x n) = m x n
```

Rank intuition:

```text
rank(P Q^T) <= k

A fully observed real matrix always has an SVD.
The constraint is whether a chosen small k can reconstruct the matrix exactly.

For exact reconstruction of a fully observed matrix:
R = P Q^T requires k >= rank(R)

In RecSys we usually choose k much smaller than min(m, n),
so this is a low-rank approximation:
R ≈ P Q^T
```

Connection to eigenvalues and eigenvectors:

```text
Eigenvalue decomposition applies directly to square matrices.
User-item matrices are usually rectangular.
The related classical decomposition is SVD:

R = U Σ V^T

U and V are related to eigenvectors of R R^T and R^T R.
Singular values in Σ are square roots of eigenvalues of those square matrices.
```

RecSys caveat:

```text
We do not factorize a full matrix where missing ratings are treated as real zeros.
We learn user and item factors from observed ratings only.
```

Simple hand-calculation example with two latent factors:

```text
User factors:
Alice = [0.9, 0.1]
Bob   = [0.2, 0.8]

Item factors:
Action movie  = [5, 1]
Romance movie = [1, 5]
Balanced movie = [3, 3]
```

Manual predictions:

```text
r_hat(Alice, Action)  = 0.9*5 + 0.1*1 = 4.6
r_hat(Alice, Romance) = 0.9*1 + 0.1*5 = 1.4
r_hat(Alice, Balanced) = 0.9*3 + 0.1*3 = 3.0

r_hat(Bob, Action)  = 0.2*5 + 0.8*1 = 1.8
r_hat(Bob, Romance) = 0.2*1 + 0.8*5 = 4.2
r_hat(Bob, Balanced) = 0.2*3 + 0.8*3 = 3.0
```

Recommended matrix-table demo:

```text
Show three plain tables side by side:

P       @       Q^T       =       R_hat
2 x 2           2 x 3             2 x 3

Then show one cell as a dot-product contribution table:
r_hat(Alice, Action) = 0.9*5.0 + 0.1*1.0 = 4.6
```

Teaching point:

```text
The factor names are illustrative.
In real matrix factorization, the latent dimensions are learned from observed ratings
and may not have clean human-readable names.
```

Optional objective formula, without deriving the optimizer:

```text
minimize over P, Q:

Σ observed (r_ui - p_u · q_i)^2 + λ(||p_u||^2 + ||q_i||^2)
```

Explain:

```text
The model only tries to match observed ratings.
Then it uses the learned user and item factors to fill in plausible scores for missing pairs.
```

---

## 80–88 min — Simple Evaluation

Goal:

```text
Evaluate whether the recommender can recover a hidden future interaction.
```

Use a simple leave-last-out split:

```text
For each user:
- train on earlier interactions;
- hide the last liked interaction as test;
- generate top-K recommendations;
- check whether the hidden item appears in top-K.
```

Metric:

```text
HitRate@K = 1 if hidden item is in top-K recommendations, else 0
```

Teaching point:

```text
In RecSys, evaluation is usually about ranking quality, not just point prediction accuracy.
```

---

## 88–90 min — Summary

Students should leave with this mental model:

```text
Popularity:
Recommend what many people liked.

Content-based:
Recommend items similar to what this user liked.

Collaborative filtering:
Recommend items liked by users with similar behavior patterns.

Matrix factorization:
Represent users and items with short latent vectors and score them with a dot product.

Evaluation:
Hide known future interactions and test whether the recommender recovers them.
```

---

# 5. Google Colab Notebook Structure

Recommended notebook name:

```text
recsys_part_1_foundations_movielens.ipynb
```

Notebook sections:

```text
1. Setup and download dataset
2. Load and inspect data
3. Basic RecSys data analysis
4. Popularity baseline
5. Convert explicit ratings to implicit likes
6. Content-based recommendation using genres
7. User-item matrix
8. Item-item collaborative filtering
9. Matrix factorization by hand
10. Simple leave-last-out evaluation
11. Discussion and exercises
```

---

# 6. Notebook Step-by-Step

## Step 1 — Setup and Download Dataset

```python
!wget -q https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip -q ml-latest-small.zip
```

```python
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
```

---

## Step 2 — Load Data

```python
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")
tags = pd.read_csv("ml-latest-small/tags.csv")
```

```python
ratings.head()
```

```python
movies.head()
```

```python
tags.head()
```

Expected columns:

```text
ratings.csv:
- userId
- movieId
- rating
- timestamp

movies.csv:
- movieId
- title
- genres

tags.csv:
- userId
- movieId
- tag
- timestamp
```

---

## Step 3 — Inspect Data

```python
n_users = ratings["userId"].nunique()
n_movies = ratings["movieId"].nunique()
n_ratings = len(ratings)

print("Users:", n_users)
print("Movies with ratings:", n_movies)
print("Ratings:", n_ratings)
```

Calculate sparsity:

```python
sparsity = 1 - n_ratings / (n_users * n_movies)
print("Sparsity:", round(sparsity, 4))
```

Explain:

```text
Sparsity tells us how much of the user-item matrix is missing.
Formula:
sparsity = 1 - observed_ratings / (n_users * n_movies)
In recommender systems, high sparsity is normal.
```

Rating distribution:

```python
ratings["rating"].value_counts().sort_index()
```

Most active users:

```python
ratings.groupby("userId").size().sort_values(ascending=False).head(10)
```

Most rated movies:

```python
most_rated = (
    ratings
    .groupby("movieId")
    .size()
    .reset_index(name="num_ratings")
    .merge(movies, on="movieId")
    .sort_values("num_ratings", ascending=False)
)

most_rated.head(10)
```

---

## Step 4 — Popularity Baseline

Simple popularity by number of ratings:

```python
popular_by_count = (
    ratings
    .groupby("movieId")
    .agg(
        num_ratings=("rating", "count"),
        mean_rating=("rating", "mean")
    )
    .reset_index()
    .merge(movies, on="movieId")
    .sort_values("num_ratings", ascending=False)
)

popular_by_count.head(10)
```

Discuss:

```text
This recommends movies that many users rated.
It is robust but not personalized.
```

Simple popularity by average rating:

```python
popular_by_mean = popular_by_count.sort_values("mean_rating", ascending=False)
popular_by_mean.head(10)
```

Discuss the issue:

```text
A movie with one 5-star rating may appear better than a movie with thousands of ratings.
This is unstable.
```

Add a minimum count filter:

```python
popular_filtered = popular_by_count[popular_by_count["num_ratings"] >= 50]
popular_filtered.sort_values("mean_rating", ascending=False).head(10)
```

Optional Bayesian score:

```python
C = ratings["rating"].mean()
m = 50

popular_by_count["bayesian_score"] = (
    (popular_by_count["num_ratings"] / (popular_by_count["num_ratings"] + m))
    * popular_by_count["mean_rating"]
    +
    (m / (popular_by_count["num_ratings"] + m))
    * C
)

popular_by_count.sort_values("bayesian_score", ascending=False).head(10)
```

Explain:

```text
Bayesian score shrinks movies with few ratings toward the global average.
This avoids over-trusting tiny samples.
```

Show both many-rating and few-rating examples:

```python
popular_by_count["own_mean_weight"] = (
    popular_by_count["num_ratings"] / (popular_by_count["num_ratings"] + m)
)
popular_by_count["global_mean_weight"] = m / (popular_by_count["num_ratings"] + m)

many_rating_examples = popular_by_count.sort_values("num_ratings", ascending=False).head(3)
few_rating_examples = (
    popular_by_count[popular_by_count["num_ratings"] <= 2]
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
]
```

Teaching point:

```text
The few-rating examples can have very high mean ratings.
Bayesian score does not fully trust them because there is too little evidence.
This makes the baseline more stable than sorting by mean_rating alone.
```

Function:

```python
def recommend_popular(top_k=10):
    return (
        popular_by_count
        .sort_values("bayesian_score", ascending=False)
        .head(top_k)[["movieId", "title", "genres", "num_ratings", "mean_rating", "bayesian_score"]]
    )

recommend_popular(10)
```

---

## Step 5 — Convert Ratings to Implicit Feedback

Create a binary label:

```python
ratings["liked"] = (ratings["rating"] >= 4.0).astype(int)
ratings.head()
```

Explain:

```text
For top-K recommendation, we often care whether an item is relevant.
Here we define rating >= 4.0 as relevant.
```

Important caveat:

```text
This threshold is arbitrary.
In real projects, the definition of relevance depends on the product goal.
```

---

## Step 6 — Content-Based Recommendation Using Genres

Prepare genre features:

```python
movies["genres_list"] = movies["genres"].str.split("|")

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies["genres_list"])

genre_features = pd.DataFrame(
    genre_matrix,
    columns=mlb.classes_,
    index=movies["movieId"]
)

genre_features.head()
```

Explain:

```text
genre_features is a movie-by-genre matrix.
Rows are movies.
Columns are genres.
Values are 0/1 indicators.

Example:
movieId   Action   Comedy   Drama   Sci-Fi
1            0        1       0        0
32           0        0       0        1
260          1        0       0        1
```

Build user profile:

```python
def build_user_profile(user_id):
    user_likes = ratings[
        (ratings["userId"] == user_id) &
        (ratings["rating"] >= 4.0)
    ]

    liked_movie_ids = user_likes["movieId"].values
    liked_movie_ids = [m for m in liked_movie_ids if m in genre_features.index]

    if len(liked_movie_ids) == 0:
        return None

    profile = genre_features.loc[liked_movie_ids].mean(axis=0)
    return profile
```

Explain:

```text
genre_features.loc[liked_movie_ids]
selects only movies liked by the user.

mean(axis=0)
averages down the rows for every genre column.

The result is the user's genre profile:
how often each genre appears among liked movies.
```

Tiny example:

```text
liked movies      Action   Comedy   Drama   Sci-Fi
movie 1              0        1       0        0
movie 32             0        0       0        1
movie 260            1        0       0        1

user profile      0.33     0.33    0.00     0.67
```

Inspect one user profile:

```python
user_id = 1
profile = build_user_profile(user_id)
profile.sort_values(ascending=False).head(10)
```

Recommend content-based movies:

```python
def recommend_content_based(user_id, top_k=10):
    profile = build_user_profile(user_id)

    if profile is None:
        return recommend_popular(top_k)

    scores = cosine_similarity(
        genre_features.values,
        profile.values.reshape(1, -1)
    ).ravel()

    recs = movies.copy()
    recs["content_score"] = scores

    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"])
    recs = recs[~recs["movieId"].isin(seen_movies)]

    return recs.sort_values("content_score", ascending=False).head(top_k)[
        ["movieId", "title", "genres", "content_score"]
    ]
```

Test:

```python
recommend_content_based(user_id=1, top_k=10)
```

Discuss:

```text
This method recommends movies by genre similarity to the user's past likes.
It does not know whether other users liked those movies.

cosine_similarity compares the user's genre profile vector with each movie's genre vector:
cosine(a, b) = (a · b) / (||a|| ||b||)
```

---

## Step 7 — User-Item Matrix

Create user-item matrix:

```python
user_item = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

user_item.shape
```

Explain:

```text
Rows are users.
Columns are movies.
Values are ratings.
Missing ratings are filled with 0 only for computation.
This does not mean the user disliked the movie.
```

Inspect:

```python
user_item.head()
```

---

## Step 8 — Item-Item Collaborative Filtering

Transpose matrix:

```python
item_user = user_item.T
item_user.shape
```

Calculate item similarity:

```python
item_similarity = cosine_similarity(item_user)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_user.index,
    columns=item_user.index
)
```

Explain:

```text
item_user is a movie-by-user matrix.
Each movie is represented by the ratings it received from users.

cosine_similarity(item_user)
compares every movie vector with every other movie vector.
The result is a movie-by-movie similarity matrix.
```

Find similar movies:

```python
def similar_movies(movie_id, similarity_df, top_k=10):
    if movie_id not in similarity_df.index:
        return pd.DataFrame()

    similar_ids = (
        similarity_df[movie_id]
        .sort_values(ascending=False)
        .drop(movie_id)
        .head(top_k)
        .index
    )

    return movies[movies["movieId"].isin(similar_ids)][["movieId", "title", "genres"]]
```

Test with a known movie:

```python
movies[movies["title"].str.contains("Toy Story", case=False, na=False)].head()
```

```python
similar_movies(movie_id=1, similarity_df=item_similarity_df, top_k=10)
```

Explain:

```text
These movies are similar because users rated them in similar patterns.
This does not use genre metadata directly.
```

Build item-item CF recommender:

```python
def recommend_item_item_cf(user_id, interactions, similarity_df, top_k=10):
    user_likes = interactions[
        (interactions["userId"] == user_id) &
        (interactions["rating"] >= 4.0)
    ]

    liked_movie_ids = [
        movie_id for movie_id in user_likes["movieId"].values
        if movie_id in similarity_df.columns
    ]

    if len(liked_movie_ids) == 0:
        return recommend_popular(top_k)

    scores = similarity_df[liked_movie_ids].sum(axis=1)

    seen_movies = set(interactions[interactions["userId"] == user_id]["movieId"])
    scores = scores.drop(index=list(seen_movies), errors="ignore")

    rec_ids = scores.sort_values(ascending=False).head(top_k).index

    recs = movies[movies["movieId"].isin(rec_ids)].copy()
    recs = recs.merge(
        scores.rename("cf_score"),
        left_on="movieId",
        right_index=True
    )

    return recs.sort_values("cf_score", ascending=False)[
        ["movieId", "title", "genres", "cf_score"]
    ]
```

Explain:

```text
similarity_df[liked_movie_ids]
selects the similarity columns for movies the user liked.

sum(axis=1)
sums across those columns for each candidate movie.

score(candidate) =
    similarity(candidate, liked_movie_1)
  + similarity(candidate, liked_movie_2)
  + ...

Then we remove movies the user has already seen.
```

Test:

```python
recommend_item_item_cf(
    user_id=1,
    interactions=ratings,
    similarity_df=item_similarity_df,
    top_k=10
)
```

---

## Step 9 — Matrix Factorization by Hand

Goal:

```text
Show the core idea of matrix factorization with numbers small enough to
recalculate by hand.
```

Start with the idea:

```text
Instead of storing only direct user-item ratings, we represent:

- each user as a short vector of latent preferences;
- each item as a short vector of latent properties.

The predicted rating is the dot product between these two vectors.
```

Formula:

```text
R ≈ P Q^T

r_hat(u, i) = p_u · q_i
            = Σ_f p_uf q_if
```

Where:

```text
R = original user-item matrix
P = user-factor matrix
Q = item-factor matrix
f = latent factor index
```

Use a tiny two-factor example:

```python
user_factors = pd.DataFrame(
    {
        "factor_1": [0.9, 0.2],
        "factor_2": [0.1, 0.8],
    },
    index=["Alice", "Bob"]
)

item_factors = pd.DataFrame(
    {
        "factor_1": [5.0, 1.0, 3.0],
        "factor_2": [1.0, 5.0, 3.0],
    },
    index=["Action movie", "Romance movie", "Balanced movie"]
)

display(user_factors)
display(item_factors)
```

Manual calculation:

```text
r_hat(Alice, Action movie)
= Alice · Action movie
= 0.9*5.0 + 0.1*1.0
= 4.6
```

In code:

```python
alice_action = 0.9 * 5.0 + 0.1 * 1.0
alice_action
```

All predicted ratings:

```python
predicted_ratings = user_factors @ item_factors.T
predicted_ratings
```

Expected result:

```text
Alice:
- Action movie: 4.6
- Romance movie: 1.4
- Balanced movie: 3.0

Bob:
- Action movie: 1.8
- Romance movie: 4.2
- Balanced movie: 3.0
```

Show the matrix multiplication:

```python
q_transpose = item_factors.T
preview_r_hat = user_factors @ q_transpose

print("P: user factors, shape", user_factors.shape)
display(user_factors)

print("Q^T: item factors transposed, shape", q_transpose.shape)
display(q_transpose)

print("R_hat = P @ Q^T, shape", preview_r_hat.shape)
display(preview_r_hat)
```

Show one dot product:

```python
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
```

Explain:

```text
Alice has a high value on factor_1, so items with high factor_1 receive high scores.
Bob has a high value on factor_2, so items with high factor_2 receive high scores.
The balanced item receives the same score for both users.
```

Show the connection to missing values:

```text
Observed ratings might contain only a few entries:

                Action movie   Romance movie   Balanced movie
Alice                5              ?                ?
Bob                  ?              5                ?

Matrix factorization learns user and item factors from observed entries,
then uses dot products to estimate the missing entries.
```

Show how predicted missing entries become recommendations:

```text
Predicted scores from R_hat = P @ Q^T:

                Action movie   Romance movie   Balanced movie
Alice               4.6             1.4              3.0
Bob                 1.8             4.2              3.0

For Alice:
- Action movie is already observed, so remove it from candidates.
- Remaining candidates:
  - Romance movie: 1.4
  - Balanced movie: 3.0
- Recommend Balanced movie first.
```

In code:

```python
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

alice_unseen_scores.sort_values(ascending=False).to_frame("predicted_score")
```

Optimization objective, shown for intuition only:

```text
minimize over P, Q:

Σ_(u,i observed) (r_ui - p_u · q_i)^2 + λ(||p_u||^2 + ||q_i||^2)
```

Teaching point:

```text
For this first lecture, students do not need to implement the optimizer.
They should understand the representation and be able to manually compute a prediction.
```

---

## Step 10 — Simple Leave-Last-Out Evaluation

Sort by timestamp:

```python
ratings_sorted = ratings.sort_values("timestamp")
```

Split:

```python
train_parts = []
test_parts = []

for user_id, user_data in ratings_sorted.groupby("userId"):
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
```

Explain:

```text
We use older interactions for training.
We hide the last liked movie.
Then we check whether the recommender can recover it.
```

Rebuild the recommender using only training interactions:

```python
train_user_item = train.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

train_item_user = train_user_item.T
train_item_similarity = cosine_similarity(train_item_user)

train_item_similarity_df = pd.DataFrame(
    train_item_similarity,
    index=train_item_user.index,
    columns=train_item_user.index
)
```

Explain:

```text
This avoids evaluation leakage.
The hidden test item must not be used to build item similarities or user histories.
```

For a simple demo, evaluate one user manually:

```python
user_id = test.iloc[0]["userId"]
test_movie_id = test.iloc[0]["movieId"]

recs = recommend_item_item_cf(
    user_id=user_id,
    interactions=train,
    similarity_df=train_item_similarity_df,
    top_k=10
)
recommended_ids = recs["movieId"].tolist()

print("Test movie:", movies[movies["movieId"] == test_movie_id]["title"].values)
print("Hit@10:", int(test_movie_id in recommended_ids[:10]))
recs
```

Explain:

```text
recommended_ids[:10] takes the top 10 recommended movie IDs.
int(test_movie_id in recommended_ids[:10]) returns:
- 1 if the hidden movie is in top 10;
- 0 otherwise.
```

Small sanity-check loop over a few test users:

```python
hits = []

for _, row in test.head(100).iterrows():
    user_id = row["userId"]
    test_movie_id = row["movieId"]

    recs = recommend_item_item_cf(
        user_id=user_id,
        interactions=train,
        similarity_df=train_item_similarity_df,
        top_k=10
    )
    recommended_ids = recs["movieId"].tolist()

    hits.append(int(test_movie_id in recommended_ids[:10]))

print("HitRate@10:", np.mean(hits))
```

Explain:

```text
This is only a quick classroom check, not the final evaluation.
Each hit is 0 or 1.
HitRate@10 is the average of those hits across the sampled users.
The final RecSys metrics below are computed on the full test set.
```

---

## Step 11 — Compare Approaches for One User

Choose one user from the leave-last-out test set:

```python
demo_user_id = test.iloc[0]["userId"]
demo_test_movie_id = test.iloc[0]["movieId"]
demo_test_title = movies.loc[movies["movieId"] == demo_test_movie_id, "title"].iloc[0]

print("Demo user:", demo_user_id)
print("Hidden test movie:", demo_test_title)
```

Show recent training history:

```python
demo_user_history = (
    train[train["userId"] == demo_user_id]
    .sort_values("timestamp")
    .tail(10)
    .merge(movies, on="movieId")
)

demo_user_history[["datetime", "title", "genres", "rating", "liked"]]
```

Popularity recommendations:

```python
demo_seen_movies = set(train[train["userId"] == demo_user_id]["movieId"])

demo_popularity_recs = recommend_popular(interactions=train, top_k=200)
demo_popularity_recs = demo_popularity_recs[
    ~demo_popularity_recs["movieId"].isin(demo_seen_movies)
].head(10)

demo_popularity_recs
```

Content-based recommendations:

```python
demo_content_recs = recommend_content_based(
    user_id=demo_user_id,
    interactions=train,
    top_k=10,
)

demo_content_recs
```

Item-item CF recommendations:

```python
demo_cf_recs = recommend_item_item_cf(
    user_id=demo_user_id,
    interactions=train,
    similarity_df=train_item_similarity_df,
    top_k=10,
)

demo_cf_recs
```

Calculate metrics for this one user:

```python
metric_rows = []

for approach, recs in [
    ("Popularity", demo_popularity_recs),
    ("Content-based", demo_content_recs),
    ("Item-item CF", demo_cf_recs),
]:
    recommended_ids = recs["movieId"].tolist()
    hit = int(demo_test_movie_id in recommended_ids[:10])
    rank = recommended_ids.index(demo_test_movie_id) + 1 if hit else None

    metric_rows.append(
        {
            "approach": approach,
            "hidden_movie": demo_test_title,
            "hit@10": hit,
            "precision@10": hit / 10,
            "recall@10": hit,
            "rank@10": rank,
            "rr@10": 1 / rank if rank is not None else 0.0,
            "ndcg@10": 1 / np.log2(rank + 1) if rank is not None else 0.0,
        }
    )

pd.DataFrame(metric_rows)
```

Explain:

```text
Hit@10 checks whether the hidden item is in the top 10.
Precision@10 is hit / 10 because there are 10 recommended items.
Recall@10 is hit / 1 because there is one hidden relevant movie per user.
Rank@10 shows its position if it was found.
RR@10 is 1/rank if found, otherwise 0.
NDCG@10 gives more credit when the hidden movie is near rank 1.
```

Full test set metrics:

```python
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
```

Popularity baseline on the full test set:

```python
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
```

Content-based model on the full test set:

```python
genre_values = genre_features.values
genre_norms = np.linalg.norm(genre_values, axis=1)
genre_movie_ids = genre_features.index.to_numpy()
genre_movie_id_to_position = {
    movie_id: position for position, movie_id in enumerate(genre_movie_ids)
}
```

Then evaluate:

```python
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
```

Item-item collaborative filtering on the full test set:

```python
train_similarity_values = train_item_similarity_df.values
train_similarity_movie_ids = train_item_similarity_df.index.to_numpy()
train_similarity_movie_id_to_position = {
    movie_id: position for position, movie_id in enumerate(train_similarity_movie_ids)
}
```

Then evaluate:

```python
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
```

Final comparison:

```python
pd.DataFrame(full_metric_rows).sort_values("HitRate@10", ascending=False)
```

Important caveat:

```text
This is still a simplified offline evaluation.
It now avoids using the hidden test items during model building.
Unlike the quick sanity-check loop above, this block evaluates every row in the test set.
A serious production evaluation would also use richer candidate sets, temporal splits,
coverage/diversity metrics, and online experiments.
```

---

# 7. Suggested Exercises

## Exercise 1 — Popularity by Recent Ratings

Modify the popularity recommender to use only recent ratings.

Questions:

```text
Does the recommendation list change?
What is the difference between all-time popularity and recent popularity?
```

---

## Exercise 2 — Change the Like Threshold

Change:

```python
ratings["liked"] = (ratings["rating"] >= 4.0).astype(int)
```

to:

```python
ratings["liked"] = (ratings["rating"] >= 3.5).astype(int)
```

Questions:

```text
How does the number of positive examples change?
How might this affect recommendations?
```

---

## Exercise 3 — Compare Content-Based and CF Recommendations

For the same user, compare:

```python
recommend_content_based(user_id=1, top_k=10)
recommend_item_item_cf(
    user_id=1,
    interactions=ratings,
    similarity_df=item_similarity_df,
    top_k=10
)
```

Questions:

```text
Which list is more genre-consistent?
Which list seems more behavior-driven?
Which one is more surprising?
```

---

## Exercise 4 — Evaluate Popularity Baseline

Implement train-only evaluation for popularity recommendations.

Questions:

```text
Does popularity perform well?
For which users is popularity likely to work?
For which users is it likely to fail?
```

---

## Exercise 5 — Matrix Factorization by Hand

Given:

```text
User vector:
Charlie = [0.6, 0.4]

Item vectors:
Movie A = [5, 1]
Movie B = [1, 5]
Movie C = [3, 3]
```

Tasks:

```text
1. Calculate r_hat(Charlie, Movie A).
2. Calculate r_hat(Charlie, Movie B).
3. Calculate r_hat(Charlie, Movie C).
4. Which movie should be recommended first?
```

Expected arithmetic:

```text
r_hat(Charlie, Movie A) = 0.6*5 + 0.4*1 = 3.4
r_hat(Charlie, Movie B) = 0.6*1 + 0.4*5 = 2.6
r_hat(Charlie, Movie C) = 0.6*3 + 0.4*3 = 3.0
```

---

# 8. Concepts to Avoid in Part 1

Do not go deep into:

- production architecture;
- two-tower neural models;
- ANN indexes;
- SGD / ALS derivations for matrix factorization;
- regularization details beyond the simple objective formula;
- counterfactual evaluation;
- reinforcement learning;
- feature stores;
- online A/B testing.

These topics belong to Part 2 or later advanced material.

---

# 9. Instructor Notes

## Most important conceptual warnings

1. Missing interaction is not necessarily negative feedback.
2. Popularity baseline is a serious baseline.
3. Predicting ratings and recommending top-K items are different tasks.
4. Collaborative filtering uses behavior, not item metadata.
5. Matrix factorization predicts with dot products between user and item vectors.
6. Latent factors are learned representations, not guaranteed human-readable genres.
7. Offline evaluation is only an approximation of real user satisfaction.
8. Random splits can be misleading in RecSys.

## Recommended whiteboard diagram

```text
Users ─────┐
           ├── Interactions ──> User-Item Matrix ──> Collaborative Filtering
Items ─────┘

User-Item Matrix ──> User Factors × Item Factors ──> Matrix Factorization

Items ──> Metadata / Genres ──> Item Vectors ──> Content-Based Recommendation

All interactions ──> Popularity ──> Non-personalized Baseline
```

---

# 10. Final Summary for Students

A recommender system answers this question:

```text
Which items should we show to this user, and in what order?
```

The first basic approaches are:

```text
Popularity:
Show what is generally popular.

Content-based:
Show items similar to the user's liked items.

Collaborative filtering:
Show items liked by users with similar behavior patterns.

Matrix factorization:
Represent users and items as short vectors and score a pair with a dot product.
```

The simplest useful evaluation is:

```text
Hide a known future interaction and check whether the recommender finds it in top-K.
```
