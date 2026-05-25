# Recommender Systems — Part 2

## From Simple Recommendations to a Production-Like Pipeline

**Format:** 90-minute lecture-practice  
**Environment:** Google Colab  
**Dataset:** MovieLens Latest Small  
**Student level:** completed Part 1 on popularity, content-based recommendation, collaborative filtering, and simple top-K evaluation  
**Main goal:** students understand how simple recommenders can be combined into a realistic multi-stage recommendation pipeline.

---

# 1. Learning Objectives

By the end of this session, students should be able to explain and implement:

1. Why real recommender systems are often multi-stage systems.
2. What candidate generation means.
3. Why candidate generation optimizes recall, not final precision.
4. What ranking means in a recommender system.
5. How ranking can be formulated as a supervised ML problem.
6. How to build a ranking dataset with positive examples and sampled negatives.
7. How to engineer simple user, item, and user-item features.
8. How to train a simple ranker with classical ML.
9. What reranking means and why final recommendation lists are not based only on model score.
10. How to evaluate candidate recall, HitRate@K, and NDCG@K.
11. What production RecSys adds beyond a Colab notebook.

---

# 2. Core Teaching Story

Part 1 answered:

```text
How can we recommend items?
```

Part 2 answers:

```text
How can we build a more realistic recommendation pipeline?
```

The main architecture:

```text
Candidate Generation → Ranking → Reranking → Evaluation
```

Key conceptual split:

```text
Candidate generation:
Find many potentially relevant items quickly.
Optimize recall.

Ranking:
Sort candidates more accurately.
Optimize top positions.

Reranking:
Apply product, business, diversity, safety, or UX constraints.
Optimize final user experience.
```

---

# 3. Why Keep MovieLens for Part 2

Use the same dataset as Part 1: **MovieLens Latest Small**.

Reason:

```text
The goal of Part 2 is not data cleaning.
The goal is understanding system structure.
```

Keeping the same dataset allows students to focus on:

- candidate generation;
- ranking dataset construction;
- feature engineering;
- reranking;
- evaluation;
- production reasoning.

The same MovieLens data can support:

```text
ratings.csv -> interactions
movies.csv  -> item metadata
timestamp   -> time-based split
genres      -> content features
```

---

# 4. 90-Minute Session Plan

## 0–10 min — Recap from Part 1

Recall the three simple recommenders:

```text
1. Popularity baseline
2. Content-based recommendation
3. Item-item collaborative filtering
```

Ask:

```text
Can we directly use these as a production recommender?
```

Answer:

```text
Not really. Real systems usually need multiple stages.
```

---

## 10–25 min — Production-Like RecSys Architecture

Explain the standard pipeline:

```text
Candidate Generation → Ranking → Reranking → Serving → Feedback
```

For this practice, focus on:

```text
Candidate Generation → Ranking → Reranking → Evaluation
```

Explain why this exists:

```text
In small MovieLens, we can score every movie for every user.
In real systems, there may be millions of users and millions or billions of items.
Scoring everything with a heavy model is too slow.
```

Typical production constraints:

- latency;
- millions of items;
- fresh inventory;
- cold start;
- business constraints;
- policy filters;
- diversity;
- exploration;
- monitoring;
- feedback loops.

---

## 25–45 min — Candidate Generation

Implement several candidate sources:

```text
1. Popular candidates
2. Content-based candidates
3. Item-item CF candidates
```

Explain:

```text
A candidate generator does not need to perfectly sort items.
It needs to avoid missing good items.
```

Important statement:

```text
If the relevant item is not in the candidate pool, the ranker cannot recover it.
```

---

## 45–65 min — Ranking Model

Turn recommendation into supervised ML.

Each row:

```text
userId, movieId, features, label
```

Label:

```text
1 = user liked the movie
0 = sampled unknown item
```

Important caveat:

```text
Sampled negatives are not necessarily true dislikes.
They are unknown items treated as negative for training.
```

Train a simple ranker:

```text
Logistic Regression
```

Why logistic regression:

- students know it;
- fast in Colab;
- interpretable;
- good bridge from classical ML to RecSys ranking.

---

## 65–75 min — Reranking

Explain:

```text
The highest model scores do not always produce the best final list.
```

Examples:

- too many movies of the same genre;
- duplicates or near-duplicates;
- all recommendations from the same source;
- lack of novelty;
- lack of diversity;
- business constraints;
- policy constraints.

Implement a simple genre-diversity reranker.

---

## 75–85 min — Evaluation

Evaluate three levels:

```text
1. Candidate Recall
2. HitRate@K
3. NDCG@K
```

Explain:

```text
Candidate Recall checks whether candidate generation found the hidden relevant item.
HitRate@K checks whether the final top-K list contains the hidden relevant item.
NDCG@K gives more credit if the hidden item appears higher in the list.
```

---

## 85–90 min — Production Discussion

Discuss what is missing compared to real production systems:

- online A/B testing;
- real-time features;
- approximate nearest neighbor search;
- feature stores;
- monitoring;
- feedback loops;
- exploration/exploitation;
- long-term user satisfaction;
- trust and safety;
- fairness and exposure constraints.

---

# 5. Google Colab Notebook Structure

Recommended notebook name:

```text
recsys_part_2_mini_production_pipeline_movielens.ipynb
```

Notebook sections:

```text
1. Setup and load data
2. Time-based train/test split
3. Prepare genre features
4. Build candidate generators
5. Merge candidates from multiple sources
6. Build ranking training dataset
7. Engineer user and item features
8. Train a simple ranking model
9. Generate ranked recommendations
10. Apply diversity reranking
11. Evaluate candidate recall, HitRate@K, NDCG@K
12. Discuss production extensions
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
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
```

```python
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")
```

```python
ratings["liked"] = (ratings["rating"] >= 4.0).astype(int)
ratings.head()
```

---

## Step 2 — Time-Based Train/Test Split

Sort by time:

```python
ratings_sorted = ratings.sort_values("timestamp")
```

For each user, hide the last liked movie:

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

print(train.shape)
print(test.shape)
```

Explain:

```text
We train on earlier interactions and test on a future liked movie.
This is closer to the real recommendation setting than random splitting.
```

Caveat:

```text
This is still simplified.
In real systems, evaluation design is more complex and depends on exposure logs.
```

---

## Step 3 — Prepare Genre Features

```python
movies["genres_list"] = movies["genres"].str.split("|")

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies["genres_list"])

genre_features = pd.DataFrame(
    genre_matrix,
    columns=mlb.classes_,
    index=movies["movieId"]
)

genre_one_hot = movies[["movieId"]].join(
    movies["genres"].str.get_dummies(sep="|")
)
```

Explain:

```text
Genres give us simple content metadata.
We will use them both for candidate generation and ranking features.
```

---

## Step 4 — Candidate Generator 1: Popular Items

Build popularity from train only:

```python
popular_candidates = (
    train[train["liked"] == 1]
    .groupby("movieId")
    .size()
    .sort_values(ascending=False)
)
```

Function:

```python
def get_popular_candidates(user_id, n=100):
    seen = set(train[train["userId"] == user_id]["movieId"])

    candidates = [
        movie_id for movie_id in popular_candidates.index
        if movie_id not in seen
    ]

    return candidates[:n]
```

Explain:

```text
Popular candidates are useful as fallback and for new users.
They are not personalized.
```

---

## Step 5 — Candidate Generator 2: Content-Based Candidates

Build user profile from liked movies in train:

```python
def build_user_genre_profile(user_id):
    user_likes = train[
        (train["userId"] == user_id) &
        (train["liked"] == 1)
    ]

    liked_ids = [
        movie_id for movie_id in user_likes["movieId"].values
        if movie_id in genre_features.index
    ]

    if len(liked_ids) == 0:
        return None

    return genre_features.loc[liked_ids].mean(axis=0)
```

Content candidates:

```python
def get_content_candidates(user_id, n=100):
    profile = build_user_genre_profile(user_id)

    if profile is None:
        return get_popular_candidates(user_id, n=n)

    scores = cosine_similarity(
        genre_features.values,
        profile.values.reshape(1, -1)
    ).ravel()

    candidate_df = pd.DataFrame({
        "movieId": genre_features.index,
        "content_score": scores
    })

    seen = set(train[train["userId"] == user_id]["movieId"])
    candidate_df = candidate_df[~candidate_df["movieId"].isin(seen)]

    return (
        candidate_df
        .sort_values("content_score", ascending=False)
        .head(n)["movieId"]
        .tolist()
    )
```

Explain:

```text
Content candidates are personalized based on the user's genre history.
This is simple but useful.
```

---

## Step 6 — Candidate Generator 3: Item-Item Collaborative Filtering

Create user-item matrix from train only:

```python
user_item_train = train.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

item_user_train = user_item_train.T
```

Calculate item-item similarity:

```python
item_similarity = cosine_similarity(item_user_train)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_user_train.index,
    columns=item_user_train.index
)
```

Candidate function:

```python
def get_item_item_candidates(user_id, n=100):
    user_likes = train[
        (train["userId"] == user_id) &
        (train["liked"] == 1)
    ]

    liked_ids = [
        movie_id for movie_id in user_likes["movieId"].values
        if movie_id in item_similarity_df.columns
    ]

    if len(liked_ids) == 0:
        return get_popular_candidates(user_id, n=n)

    scores = item_similarity_df[liked_ids].sum(axis=1)

    seen = set(train[train["userId"] == user_id]["movieId"])
    scores = scores.drop(index=list(seen), errors="ignore")

    return scores.sort_values(ascending=False).head(n).index.tolist()
```

Explain:

```text
This candidate generator uses behavior patterns.
It finds movies similar to movies the user liked.
```

---

## Step 7 — Merge Candidate Sources

Function:

```python
def generate_candidates(user_id, n_per_source=100):
    rows = []

    generators = [
        ("popular", get_popular_candidates),
        ("content", get_content_candidates),
        ("item_item_cf", get_item_item_candidates),
    ]

    for source_name, generator in generators:
        movie_ids = generator(user_id, n=n_per_source)

        for rank, movie_id in enumerate(movie_ids, start=1):
            rows.append({
                "userId": user_id,
                "movieId": movie_id,
                "source": source_name,
                "source_rank": rank
            })

    candidate_df = pd.DataFrame(rows)

    if len(candidate_df) == 0:
        return candidate_df

    # Aggregate source-level information.
    source_features = (
        candidate_df
        .groupby(["userId", "movieId"])
        .agg(
            num_sources=("source", "nunique"),
            best_source_rank=("source_rank", "min"),
            sources=("source", lambda x: "|".join(sorted(set(x))))
        )
        .reset_index()
    )

    return source_features
```

Test:

```python
candidate_example = generate_candidates(user_id=1, n_per_source=20)
candidate_example.head()
```

Explain:

```text
The same item can be generated by multiple candidate sources.
This is often a useful signal.
```

---

## Step 8 — Candidate Recall

Before training a ranker, check whether candidate generation finds the hidden test item.

```python
def candidate_recall_for_user(user_id, test_movie_id, n_per_source=100):
    candidates = generate_candidates(user_id, n_per_source=n_per_source)

    if len(candidates) == 0:
        return 0

    return int(test_movie_id in set(candidates["movieId"]))
```

Evaluate on a subset:

```python
candidate_hits = []

for _, row in test.iterrows():
    user_id = row["userId"]
    test_movie_id = row["movieId"]

    candidate_hits.append(
        candidate_recall_for_user(user_id, test_movie_id, n_per_source=100)
    )

print("Candidate Recall:", np.mean(candidate_hits))
```

Explain:

```text
This is not final recommendation quality.
It only checks whether the relevant item was found by candidate generation.
```

Critical teaching point:

```text
Candidate generation is an upper bound for the ranker.
If candidate recall is poor, ranking cannot fix it.
```

---

## Step 9 — Build Ranking Training Dataset

Positive examples:

```python
positive_train = train[train["liked"] == 1][["userId", "movieId"]].copy()
positive_train["label"] = 1
```

Negative sampling:

```python
rng = np.random.default_rng(seed=42)
all_movie_ids = ratings["movieId"].unique()

negative_rows = []

for user_id in positive_train["userId"].unique():
    seen = set(train[train["userId"] == user_id]["movieId"])
    possible_negatives = np.array(list(set(all_movie_ids) - seen))

    user_positive_count = len(positive_train[positive_train["userId"] == user_id])
    sample_size = min(user_positive_count * 3, len(possible_negatives))

    if sample_size == 0:
        continue

    sampled_negatives = rng.choice(
        possible_negatives,
        size=sample_size,
        replace=False
    )

    for movie_id in sampled_negatives:
        negative_rows.append({
            "userId": user_id,
            "movieId": movie_id,
            "label": 0
        })

negative_train = pd.DataFrame(negative_rows)

ranking_train = pd.concat(
    [positive_train, negative_train],
    ignore_index=True
)

ranking_train["label"].value_counts()
```

Explain:

```text
We create positive examples from liked movies.
We create negative examples by sampling movies the user did not interact with.
These negatives are not true dislikes. They are unknown items used as negative training examples.
```

---

## Step 10 — User and Item Features

User features from train:

```python
user_features = (
    train
    .groupby("userId")
    .agg(
        user_num_ratings=("rating", "count"),
        user_avg_rating=("rating", "mean"),
        user_num_likes=("liked", "sum")
    )
    .reset_index()
)
```

Item features from train:

```python
item_features = (
    train
    .groupby("movieId")
    .agg(
        item_num_ratings=("rating", "count"),
        item_avg_rating=("rating", "mean"),
        item_like_rate=("liked", "mean")
    )
    .reset_index()
)
```

Merge features:

```python
ranking_train = (
    ranking_train
    .merge(user_features, on="userId", how="left")
    .merge(item_features, on="movieId", how="left")
    .merge(genre_one_hot, on="movieId", how="left")
)
```

Feature columns:

```python
basic_feature_cols = [
    "user_num_ratings",
    "user_avg_rating",
    "user_num_likes",
    "item_num_ratings",
    "item_avg_rating",
    "item_like_rate",
]

genre_feature_cols = [
    col for col in genre_one_hot.columns
    if col != "movieId"
]

feature_cols = basic_feature_cols + genre_feature_cols
```

Explain:

```text
The ranker receives user features, item features, and metadata features.
This is similar to a standard supervised ML table.
```

---

## Step 11 — Train a Simple Ranking Model

```python
X = ranking_train[feature_cols]
y = ranking_train["label"]

ranker = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

ranker.fit(X, y)
```

Check AUC on training data only as a sanity check:

```python
ranking_train["pred"] = ranker.predict_proba(X)[:, 1]
print("Train AUC:", roc_auc_score(y, ranking_train["pred"]))
```

Explain:

```text
AUC is only a sanity check here.
The final recommender must be evaluated with top-K metrics.
```

Important warning:

```text
High AUC does not guarantee good recommendations.
A ranking model can look good globally but still fail in the top positions.
```

---

## Step 12 — Feature Function for Candidate Data

```python
def add_features(candidate_df):
    df = (
        candidate_df
        .merge(user_features, on="userId", how="left")
        .merge(item_features, on="movieId", how="left")
        .merge(genre_one_hot, on="movieId", how="left")
    )

    return df
```

---

## Step 13 — Generate Ranked Recommendations

```python
def recommend_with_ranker(user_id, top_k=10, n_per_source=100):
    candidates = generate_candidates(user_id, n_per_source=n_per_source)

    if len(candidates) == 0:
        return pd.DataFrame()

    candidates = add_features(candidates)

    candidates["rank_score"] = ranker.predict_proba(
        candidates[feature_cols]
    )[:, 1]

    recommendations = (
        candidates
        .sort_values("rank_score", ascending=False)
        .head(top_k)
        .merge(movies, on="movieId", how="left")
    )

    return recommendations[[
        "userId",
        "movieId",
        "title",
        "genres",
        "rank_score",
        "num_sources",
        "best_source_rank",
        "sources"
    ]]
```

Test:

```python
recommend_with_ranker(user_id=1, top_k=10, n_per_source=100)
```

Explain:

```text
Now we have a two-stage system:
1. Generate candidates.
2. Rank candidates with a supervised ML model.
```

---

## Step 14 — Diversity Reranking

Problem:

```text
The top-ranked list may contain too many movies from the same genre.
```

Simple diversity reranker:

```python
def diversity_rerank(recommendations, top_k=10, max_per_genre=3):
    final_rows = []
    genre_counts = {}

    for _, row in recommendations.iterrows():
        genres = row["genres"].split("|")

        allowed = True
        for genre in genres:
            if genre_counts.get(genre, 0) >= max_per_genre:
                allowed = False
                break

        if allowed:
            final_rows.append(row)
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        if len(final_rows) == top_k:
            break

    return pd.DataFrame(final_rows)
```

Use it:

```python
raw_recs = recommend_with_ranker(user_id=1, top_k=50, n_per_source=100)
diverse_recs = diversity_rerank(raw_recs, top_k=10, max_per_genre=3)
diverse_recs
```

Explain:

```text
Reranking modifies the final list after model scoring.
This can improve diversity, user experience, business value, or safety.
```

Important product point:

```text
The highest scoring list is not always the best list.
```

---

## Step 15 — Top-K Metrics

HitRate@K:

```python
def hit_rate_at_k(recommended_ids, test_item_id, k=10):
    return int(test_item_id in recommended_ids[:k])
```

NDCG@K for one hidden relevant item:

```python
def ndcg_at_k(recommended_ids, test_item_id, k=10):
    recommended_ids = recommended_ids[:k]

    if test_item_id not in recommended_ids:
        return 0.0

    rank = recommended_ids.index(test_item_id) + 1
    return 1 / np.log2(rank + 1)
```

Explain:

```text
HitRate@K only checks whether the item is in the list.
NDCG@K rewards placing the relevant item higher.
```

---

## Step 16 — Evaluate the Ranked Recommender

```python
results = []

for _, row in test.iterrows():
    user_id = row["userId"]
    test_movie_id = row["movieId"]

    recs = recommend_with_ranker(user_id, top_k=10, n_per_source=100)

    if len(recs) == 0:
        rec_ids = []
    else:
        rec_ids = recs["movieId"].tolist()

    results.append({
        "userId": user_id,
        "test_movieId": test_movie_id,
        "hit@10": hit_rate_at_k(rec_ids, test_movie_id, k=10),
        "ndcg@10": ndcg_at_k(rec_ids, test_movie_id, k=10)
    })

eval_df = pd.DataFrame(results)

eval_df[["hit@10", "ndcg@10"]].mean()
```

Explain:

```text
This measures whether the final ranked top-10 list recovers the user's hidden future liked movie.
```

---

## Step 17 — Evaluate Reranked Recommendations

```python
rerank_results = []

for _, row in test.iterrows():
    user_id = row["userId"]
    test_movie_id = row["movieId"]

    raw_recs = recommend_with_ranker(user_id, top_k=50, n_per_source=100)

    if len(raw_recs) == 0:
        rec_ids = []
    else:
        reranked = diversity_rerank(raw_recs, top_k=10, max_per_genre=3)
        rec_ids = reranked["movieId"].tolist()

    rerank_results.append({
        "userId": user_id,
        "test_movieId": test_movie_id,
        "hit@10": hit_rate_at_k(rec_ids, test_movie_id, k=10),
        "ndcg@10": ndcg_at_k(rec_ids, test_movie_id, k=10)
    })

rerank_eval_df = pd.DataFrame(rerank_results)
rerank_eval_df[["hit@10", "ndcg@10"]].mean()
```

Discuss:

```text
Reranking may improve diversity but can reduce pure relevance metrics.
This is a real product trade-off.
```

---

# 7. Discussion: What This Pipeline Represents

The notebook simulates a real-world structure:

```text
Candidate Generation:
- popular candidates
- content-based candidates
- item-item CF candidates

Ranking:
- supervised ML model
- user/item/genre features

Reranking:
- diversity constraint

Evaluation:
- candidate recall
- HitRate@K
- NDCG@K
```

This is not production-grade, but it teaches the main architecture.

---

# 8. What Real Production RecSys Adds

## 8.1 Candidate Generation at Scale

Real systems often use:

- approximate nearest neighbor search;
- two-tower models;
- embedding retrieval;
- graph-based retrieval;
- co-visitation matrices;
- query-based retrieval;
- real-time trending items;
- sponsored or business-driven candidates.

---

## 8.2 Ranking at Scale

Real rankers may use:

- gradient boosting;
- deep neural networks;
- sequence models;
- transformers;
- learning-to-rank objectives;
- multi-task learning;
- calibrated probabilities;
- multiple objectives.

Possible prediction targets:

```text
P(click)
P(like)
P(watch)
P(add_to_cart)
P(purchase)
expected revenue
expected satisfaction
long-term retention
```

---

## 8.3 Reranking and Constraints

Real reranking may include:

- diversity;
- novelty;
- freshness;
- fairness of exposure;
- business rules;
- policy filtering;
- safety constraints;
- deduplication;
- seller/category balance;
- exploration slots.

---

## 8.4 Online Evaluation

Offline metrics are not enough.

Online metrics may include:

- CTR;
- conversion rate;
- watch time;
- revenue per user;
- retention;
- satisfaction;
- hide/report rate;
- return rate;
- long-term engagement.

A/B testing is usually required because:

```text
Offline improvement does not always translate into online improvement.
```

---

## 8.5 Feedback Loops and Biases

Important RecSys-specific risks:

```text
Popularity bias:
Popular items get more exposure and become even more popular.

Position bias:
Items at the top get more clicks because they are at the top.

Feedback loop:
The system learns from its own previous recommendations.

Cold start:
New users or new items have little interaction history.

Filter bubble:
Users may be repeatedly shown similar content.
```

---

# 9. Suggested Exercises

## Exercise 1 — Candidate Source Comparison

Evaluate candidate recall separately for:

```text
popular candidates only
content candidates only
item-item CF candidates only
combined candidates
```

Questions:

```text
Which source has the best candidate recall?
Do sources complement each other?
```

---

## Exercise 2 — Change Candidate Pool Size

Try:

```python
n_per_source = 10
n_per_source = 50
n_per_source = 100
n_per_source = 200
```

Questions:

```text
How does candidate recall change?
How does runtime change?
What is the trade-off?
```

---

## Exercise 3 — Add Candidate Source Features to the Ranker

Currently the ranker uses user/item/genre features.

Add:

```text
num_sources
best_source_rank
```

Questions:

```text
Does this improve top-K metrics?
Why might source information be useful?
```

Implementation hint:

```python
feature_cols = basic_feature_cols + genre_feature_cols + ["num_sources", "best_source_rank"]
```

---

## Exercise 4 — Replace Logistic Regression

Try:

```text
RandomForestClassifier
GradientBoostingClassifier
HistGradientBoostingClassifier
```

Questions:

```text
Does a stronger model improve HitRate@10 or NDCG@10?
Does it improve AUC only?
What does this tell us?
```

---

## Exercise 5 — Reranking Trade-Off

Try different values:

```python
max_per_genre = 1
max_per_genre = 2
max_per_genre = 3
max_per_genre = 5
```

Questions:

```text
How does diversity change?
How do HitRate@10 and NDCG@10 change?
What would be the right product decision?
```

---

# 10. Concepts to Avoid Going Too Deep Into

For this second introductory session, do not deeply cover:

- full two-tower model training;
- approximate nearest neighbor internals;
- reinforcement learning;
- counterfactual evaluation;
- causal inference;
- graph neural networks;
- transformer-based sequential recommendation;
- feature store implementation;
- streaming architecture;
- large-scale distributed serving.

Mention them only as advanced topics.

---

# 11. Instructor Notes

## Most Important Conceptual Points

1. Real RecSys is usually multi-stage.
2. Candidate generation and ranking solve different problems.
3. Candidate generation is about recall.
4. Ranking is usually a supervised ML problem over user-item rows.
5. Negative sampling is a practical approximation, not perfect truth.
6. Reranking represents product constraints beyond pure model score.
7. Offline evaluation is useful but incomplete.
8. Production RecSys requires online testing and monitoring.

---

## Recommended Whiteboard Diagram

```text
                  ┌─────────────────────┐
                  │ User + Context       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Candidate Generation │
                  │                     │
                  │ - Popular            │
                  │ - Content-based      │
                  │ - Item-item CF       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Candidate Pool       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Ranking Model        │
                  │ P(user likes item)   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Reranking            │
                  │ diversity/constraints│
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Final Top-K List     │
                  └─────────────────────┘
```

---

# 12. Final Summary for Students

A realistic recommender system is not just one algorithm.

It is usually a pipeline:

```text
Candidate Generation:
Find many potentially relevant items.

Ranking:
Sort those candidates using a stronger model.

Reranking:
Adjust the final list using product and user-experience constraints.

Evaluation:
Measure whether the final top-K list contains relevant future interactions.
```

The key idea:

```text
Recommendation is not only prediction.
Recommendation is ranking plus decision-making under product constraints.
```

