# Recommender Systems — Part 2

## From Simple Recommendations to a Production-Like Pipeline

**Format:** 90-minute lecture-practice  
**Environment:** Google Colab with PyTorch
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
10. Why matrix factorization in RecSys is usually an optimization problem on observed interactions, not a classical exact decomposition of a fully known matrix.
11. What a two-tower retrieval model is, why it is useful in production, and how to train a small PyTorch version.
12. Which metrics belong to candidate generation, ranking, reranking, serving, and online product evaluation.
13. Why HitRate@K alone is not enough for RecSys evaluation.
14. How latency, storage, and recomputation costs shape production RecSys architecture.
15. What production RecSys adds beyond a Colab notebook.

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
Candidate Generation → Ranking → Reranking → Serving → Logging → Evaluation
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

Serving:
Return a top-K list within latency and availability constraints.

Logging:
Record what was shown and how the user responded.

Evaluation:
Measure retrieval quality, ranking quality, list quality, system health,
and online product impact.
```

The stages are separated for a reason:

```text
Candidate generation can be fast and approximate.
Ranking can be slower and more accurate because it sees fewer items.
Reranking can encode product constraints that are not pure relevance.
Serving has latency and reliability constraints.
Logging makes future training and evaluation possible.
```

Important teaching bridge from Part 1:

```text
Classical matrix decomposition:
Start with a fully observed matrix R and decompose it.

RecSys matrix factorization:
Observe only some user-item interactions and learn user/item factors that
fit those observed interactions.

Two-tower retrieval:
Learn functions that produce user and item embeddings, then retrieve items
with a fast vector search.
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

Explain why stages are not merged into one model:

```text
One huge model that scores every item is usually too slow.
Candidate generation narrows millions of items to hundreds or thousands.
Ranking can then spend more computation on a much smaller set.
Reranking can change the final list for diversity, policy, business, or UX.
Different stages have different metrics and failure modes.
```

Add the production cost framing early:

```text
Full scoring all users against all items:
O(num_users * num_items * embedding_dim)

Dense item-item similarity:
O(num_items^2 * num_users) compute
O(num_items^2) storage

Embedding retrieval:
O(num_items * embedding_dim) storage for item vectors
plus a vector index for fast approximate nearest-neighbor search
```

Important production message:

```text
The question is not only "which model is accurate?"
It is also "what must be recomputed, how often, and can serving stay within latency?"
```

---

## 25–40 min — Candidate Generation

Implement several candidate sources:

```text
1. Popular candidates
2. Content-based candidates
3. Item-item CF candidates
4. Simple embedding retrieval / two-tower-style candidates
5. Small PyTorch two-tower retrieval candidates
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

## 40–60 min — Two-Tower Retrieval with PyTorch

Introduce a production-style retrieval model:

```text
User tower: user features + user history -> user embedding
Item tower: item features/content/id -> item embedding
Score: dot product or cosine similarity
```

Formula:

```text
u = f_user(user_features, user_history)
v = f_item(item_features)
score(user, item) = u · v
```

Explain why it matters:

```text
Item embeddings can be precomputed.
At request time, compute one user embedding.
Retrieve nearest item embeddings with ANN/vector search.
This makes personalized retrieval possible at large catalog scale.
```

First show the vector idea without deep learning:

```text
user embedding = average embedding of liked movies
item embedding = genre/content embedding
candidate score = user embedding · item embedding
```

Then train a small PyTorch two-tower model in Colab:

```text
positive pair: user liked movie
negative pair: user did not interact with sampled movie
model score: user_tower(user_id) · item_tower(movie_id)
loss: binary classification loss over positive and sampled negative pairs
```

Then explicitly connect it to Part 1:

```text
Matrix factorization learns one vector per user and one vector per item.
Two-tower models learn functions that produce those vectors from features.
```

---

## 60–72 min — Ranking Model

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

## 72–80 min — Reranking

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

## 80–90 min — Evaluation and Production Metrics

Evaluate different stages with different metrics:

```text
Candidate generation:
- Candidate Recall@N
- source overlap
- runtime / latency
- PyTorch two-tower training loss as a debugging signal, not final quality

Ranking:
- AUC as a sanity check
- HitRate@K
- Precision@K
- Recall@K
- MRR@K
- NDCG@K

Reranking / final list:
- diversity
- novelty
- catalog coverage
- freshness
- constraint violation rate

Serving:
- latency p50/p95/p99
- error rate
- fallback rate

Online product evaluation:
- CTR
- like rate
- watch time
- conversion rate
- retention
- hide/report rate
```

Explain:

```text
Candidate Recall checks whether candidate generation found the hidden relevant item.
HitRate@K checks whether the final top-K list contains at least one relevant item.
Precision@K asks how much of the shown list is relevant.
Recall@K asks how much of the relevant set was recovered.
MRR@K rewards putting the first relevant item high.
NDCG@K rewards putting relevant items higher, with graded relevance if available.
Coverage and diversity ask whether the system serves a healthy catalog and list mix.
Latency and error metrics decide whether the model can actually run in production.
Online metrics decide whether the change helps users and the product.
```

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
3. Explain production constraints: latency, storage, and recomputation costs
4. Prepare genre features
5. Build candidate generators
6. Build a simple two-tower-style retrieval demo
7. Train a small PyTorch two-tower retrieval model
8. Optional T4 demo: pretrained text embeddings for item retrieval
9. Merge candidates from multiple sources
10. Build ranking training dataset
11. Engineer user and item features
12. Train a simple ranking model
13. Generate ranked recommendations
14. Apply diversity reranking
15. Evaluate stage-specific metrics
16. Discuss production extensions
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

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
```

Optional Colab/T4 dependency:

```python
# Installed only in the optional pretrained text embedding section.
# !pip install -q sentence-transformers
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

## Step 3 — Production Constraints: Speed, Storage, and Recomputations

Before building models, make the production constraint explicit.

```text
Recommendation quality is not enough.
The system must also fit latency, storage, and refresh constraints.
```

Approximate cost table:

```python
scale_examples = pd.DataFrame([
    {
        "scenario": "MovieLens small",
        "users": 610,
        "items": movies["movieId"].nunique(),
        "embedding_dim": 32,
    },
    {
        "scenario": "medium product",
        "users": 1_000_000,
        "items": 1_000_000,
        "embedding_dim": 64,
    },
    {
        "scenario": "large product",
        "users": 50_000_000,
        "items": 10_000_000,
        "embedding_dim": 128,
    },
])

scale_examples["all_user_item_scores"] = (
    scale_examples["users"] * scale_examples["items"]
)
scale_examples["dot_product_multiply_adds"] = (
    scale_examples["all_user_item_scores"] * scale_examples["embedding_dim"]
)
scale_examples["item_item_similarity_cells"] = scale_examples["items"] ** 2
scale_examples["item_item_similarity_storage_gb_float32"] = (
    scale_examples["item_item_similarity_cells"] * 4 / 1e9
)
scale_examples["item_embedding_storage_gb_float32"] = (
    scale_examples["items"] * scale_examples["embedding_dim"] * 4 / 1e9
)

scale_examples
```

Explain:

```text
Full scoring all users against all items grows as users * items.
Item-item similarity storage grows as items^2.
Embedding storage grows as items * embedding_dim.

This is why production systems avoid recomputing dense matrices on request.
They precompute item embeddings, build vector indexes, update fresh features,
and retrain heavy models on schedules.
```

Typical refresh pattern:

```text
On every request:
compute current user/context features, retrieve candidates, rank, rerank.

Frequent batch / streaming:
user histories, trending items, counters, availability, freshness features.

Scheduled batch:
item embeddings, vector index, item-item tables if small enough.

Slower retraining:
two-tower model, ranker model, calibration, business logic validation.
```

---

## Step 4 — Prepare Genre Features

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

## Step 5 — Candidate Generator 1: Popular Items

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

## Step 6 — Candidate Generator 2: Content-Based Candidates

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

## Step 7 — Candidate Generator 3: Item-Item Collaborative Filtering

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

## Step 8 — Two-Tower-Style Retrieval Demo

In production, many candidate generators are embedding retrieval systems.
A common architecture is a **two-tower model**.

Core idea:

```text
User tower:
user features + user history -> user embedding

Item tower:
item features/content/id -> item embedding

Retrieval score:
score(user, item) = user_embedding · item_embedding
```

Why this is production-friendly:

```text
The system can precompute item embeddings offline.
At request time it computes one user embedding.
Then it searches nearest item embeddings quickly with ANN/vector search.
```

Start with the same idea using simple vectors:

```text
item embedding = genre one-hot vector
user embedding = average genre vector of liked movies
score = dot product between user embedding and item embedding
```

This is a warm-up before the PyTorch version. It teaches the serving shape
without training a neural network yet.

```python
item_embedding_df = genre_features.copy()
item_embedding_values = item_embedding_df.values
item_embedding_movie_ids = item_embedding_df.index.to_numpy()

item_id_to_embedding_position = {
    movie_id: position
    for position, movie_id in enumerate(item_embedding_movie_ids)
}
```

User embedding:

```python
def build_simple_user_embedding(user_id):
    liked_ids = train[
        (train["userId"] == user_id) &
        (train["liked"] == 1)
    ]["movieId"].tolist()

    liked_positions = [
        item_id_to_embedding_position[movie_id]
        for movie_id in liked_ids
        if movie_id in item_id_to_embedding_position
    ]

    if len(liked_positions) == 0:
        return None

    return item_embedding_values[liked_positions].mean(axis=0)
```

Two-tower-style candidates:

```python
def get_two_tower_style_candidates(user_id, n=100):
    user_embedding = build_simple_user_embedding(user_id)

    if user_embedding is None:
        return get_popular_candidates(user_id, n=n)

    scores = item_embedding_values @ user_embedding

    candidate_df = pd.DataFrame({
        "movieId": item_embedding_movie_ids,
        "two_tower_score": scores
    })

    seen = set(train[train["userId"] == user_id]["movieId"])
    candidate_df = candidate_df[~candidate_df["movieId"].isin(seen)]

    return (
        candidate_df
        .sort_values("two_tower_score", ascending=False)
        .head(n)["movieId"]
        .tolist()
    )
```

Explain:

```text
This mirrors the two-tower retrieval pattern:
1. create a user vector;
2. create item vectors;
3. retrieve items with the largest vector score.

A real two-tower model learns the user and item towers from data.
This demo uses genre vectors so students can see the idea without deep learning.
```

Important distinction:

```text
Matrix factorization learns a separate vector for each known user and item.
Two-tower models learn functions that can produce vectors from features.

This helps with production use cases such as new items, rich metadata,
and scalable vector retrieval.
```

---

## Step 9 — Train a Small PyTorch Two-Tower Model

Now train a small neural two-tower model in Colab.

Teaching goal:

```text
Students should see that the model learns embeddings from user-item examples.
It does not need a fully filled user-item matrix.
It trains on observed positive interactions and sampled unknown negatives.
```

Prepare contiguous IDs for PyTorch:

```python
user_ids = sorted(train["userId"].unique())
movie_ids = sorted(movies["movieId"].unique())

user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}
```

Create positive and sampled negative pairs:

```python
rng = np.random.default_rng(seed=42)

positive_pairs = (
    train[train["liked"] == 1][["userId", "movieId"]]
    .drop_duplicates()
    .copy()
)
positive_pairs["label"] = 1.0

seen_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()
all_movie_ids = np.array(movie_ids)

negative_rows = []

for user_id, user_positive_data in positive_pairs.groupby("userId"):
    seen = seen_by_user.get(user_id, set())
    possible_negatives = np.array([
        movie_id for movie_id in all_movie_ids
        if movie_id not in seen
    ])

    sample_size = min(2 * len(user_positive_data), len(possible_negatives))

    if sample_size == 0:
        continue

    sampled_movie_ids = rng.choice(
        possible_negatives,
        size=sample_size,
        replace=False
    )

    for movie_id in sampled_movie_ids:
        negative_rows.append({
            "userId": user_id,
            "movieId": movie_id,
            "label": 0.0
        })

negative_pairs = pd.DataFrame(negative_rows)

two_tower_train = pd.concat(
    [positive_pairs, negative_pairs],
    ignore_index=True
)

two_tower_train["user_idx"] = two_tower_train["userId"].map(user_to_idx)
two_tower_train["movie_idx"] = two_tower_train["movieId"].map(movie_to_idx)

two_tower_train["label"].value_counts()
```

Explain:

```text
Positive examples are real liked interactions from train.
Negative examples are sampled unknown movies.
They are not guaranteed dislikes; they are training approximations.
```

Build tensors:

```python
user_tensor = torch.tensor(two_tower_train["user_idx"].values, dtype=torch.long)
movie_tensor = torch.tensor(two_tower_train["movie_idx"].values, dtype=torch.long)
label_tensor = torch.tensor(two_tower_train["label"].values, dtype=torch.float32)

dataset = TensorDataset(user_tensor, movie_tensor, label_tensor)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
```

Define a small two-tower model:

```python
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.movie_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode_user(self, user_idx):
        return self.user_tower(self.user_embedding(user_idx))

    def encode_movie(self, movie_idx):
        return self.movie_tower(self.movie_embedding(movie_idx))

    def forward(self, user_idx, movie_idx):
        user_vector = self.encode_user(user_idx)
        movie_vector = self.encode_movie(movie_idx)
        return (user_vector * movie_vector).sum(dim=1)
```

Train for a few epochs:

```python
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

two_tower_model = TwoTowerModel(
    num_users=len(user_to_idx),
    num_movies=len(movie_to_idx),
    embedding_dim=32
).to(device)

optimizer = torch.optim.Adam(two_tower_model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(5):
    total_loss = 0.0

    for batch_users, batch_movies, batch_labels in dataloader:
        batch_users = batch_users.to(device)
        batch_movies = batch_movies.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = two_tower_model(batch_users, batch_movies)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_labels)

    avg_loss = total_loss / len(dataset)
    print(f"epoch={epoch + 1}, loss={avg_loss:.4f}")
```

Explain:

```text
The model learns user and movie embeddings so that liked pairs get higher
dot-product scores than sampled unknown pairs.

This is a retrieval model, not the final ranker.
Its job is to produce a candidate pool.
```

Retrieve candidates from the PyTorch two-tower:

```python
all_movie_indices = torch.arange(len(movie_to_idx), dtype=torch.long).to(device)

with torch.no_grad():
    all_movie_vectors = two_tower_model.encode_movie(all_movie_indices)

def get_torch_two_tower_candidates(user_id, n=100):
    if user_id not in user_to_idx:
        return get_popular_candidates(user_id, n=n)

    user_idx = torch.tensor([user_to_idx[user_id]], dtype=torch.long).to(device)

    with torch.no_grad():
        user_vector = two_tower_model.encode_user(user_idx)
        scores = (all_movie_vectors @ user_vector.squeeze(0)).cpu().numpy()

    candidate_df = pd.DataFrame({
        "movieId": [idx_to_movie[idx] for idx in range(len(idx_to_movie))],
        "torch_two_tower_score": scores
    })

    seen = seen_by_user.get(user_id, set())
    candidate_df = candidate_df[~candidate_df["movieId"].isin(seen)]

    return (
        candidate_df
        .sort_values("torch_two_tower_score", ascending=False)
        .head(n)["movieId"]
        .tolist()
    )
```

Demo for one user:

```python
torch_candidates = get_torch_two_tower_candidates(user_id=1, n=10)

movies[movies["movieId"].isin(torch_candidates)][["movieId", "title", "genres"]]
```

Important production note:

```text
In a real production system, item vectors would usually be computed offline
and stored in a vector index. Online serving would compute the user vector and
retrieve nearest item vectors.
```

---

## Step 10 — Optional T4 Demo: Pretrained Text Embeddings

This optional section uses a ready-made text embedding model to encode movie
titles and genres.

Why show it:

```text
It demonstrates how a production RecSys can use pretrained content models.
It is useful for cold-start items where interaction history is weak.
It gives a visible GPU/T4 demo without replacing the interaction-trained model.
```

The pretrained text section is enabled by default.
It uses CUDA/T4 when available and falls back to CPU otherwise.
For a live demo, Colab T4 is recommended:

```python
RUN_PRETRAINED_TEXT_MODEL = True
```

Install and load:

```python
if RUN_PRETRAINED_TEXT_MODEL:
    !pip install -q sentence-transformers

    from sentence_transformers import SentenceTransformer

    text_device = "cuda" if torch.cuda.is_available() else "cpu"

    text_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=text_device,
    )
```

Create movie texts:

```python
movie_texts = (
    movies["title"].fillna("")
    + " genres: "
    + movies["genres"].fillna("").str.replace("|", ", ", regex=False)
).tolist()
```

Encode item embeddings:

```python
pretrained_item_vectors = text_model.encode(
    movie_texts,
    batch_size=128,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

User embedding and retrieval:

```python
pretrained_movie_ids = movies["movieId"].to_numpy()
movie_id_to_pretrained_position = {
    movie_id: position
    for position, movie_id in enumerate(pretrained_movie_ids)
}

def get_pretrained_text_candidates(user_id, n=100):
    liked_positions = [
        movie_id_to_pretrained_position[movie_id]
        for movie_id in liked_by_user.get(user_id, [])
        if movie_id in movie_id_to_pretrained_position
    ]

    if not liked_positions:
        return get_popular_candidates(user_id, n=n)

    user_vector = pretrained_item_vectors[liked_positions].mean(axis=0)
    user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-12)

    scores = pretrained_item_vectors @ user_vector
    scores = scores.copy()

    seen_positions = [
        movie_id_to_pretrained_position[movie_id]
        for movie_id in seen_by_user.get(user_id, set())
        if movie_id in movie_id_to_pretrained_position
    ]
    scores[seen_positions] = -np.inf

    top_positions = np.argsort(scores)[::-1][:n]
    return pretrained_movie_ids[top_positions].tolist()
```

Explain:

```text
This is not trained on MovieLens interactions.
It uses pretrained language understanding of titles and genre text.
The PyTorch two-tower learns from interactions.
The pretrained text model helps demonstrate content-based retrieval and cold-start intuition.
```

---

## Step 11 — Merge Candidate Sources

Function:

```python
def generate_candidates(user_id, n_per_source=100):
    rows = []

    generators = [
        ("popular", get_popular_candidates),
        ("content", get_content_candidates),
        ("item_item_cf", get_item_item_candidates),
        ("two_tower_style", get_two_tower_style_candidates),
        ("torch_two_tower", get_torch_two_tower_candidates),
    ]

    if RUN_PRETRAINED_TEXT_MODEL:
        generators.append(("pretrained_text", get_pretrained_text_candidates))

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

## Step 12 — Candidate Recall

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

This is why candidate generation and ranking are separated:
candidate generation tries not to miss relevant items, while ranking decides
the final ordering among the retrieved candidates.
```

---

## Step 13 — Build Ranking Training Dataset

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

## Step 14 — User and Item Features

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

## Step 15 — Train a Simple Ranking Model

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

## Step 16 — Feature Function for Candidate Data

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

## Step 17 — Generate Ranked Recommendations

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

## Step 18 — Diversity Reranking

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

## Step 19 — Top-K Metrics

In this notebook, each test row has one hidden relevant movie.
That makes the formulas simple enough to calculate by hand.

Hit@K:

```python
def hit_rate_at_k(recommended_ids, test_item_id, k=10):
    return int(test_item_id in recommended_ids[:k])
```

Precision@K:

```python
def precision_at_k(recommended_ids, test_item_id, k=10):
    hit = hit_rate_at_k(recommended_ids, test_item_id, k=k)
    return hit / k
```

Recall@K:

```python
def recall_at_k(recommended_ids, test_item_id, k=10):
    return hit_rate_at_k(recommended_ids, test_item_id, k=k)
```

Mean Reciprocal Rank contribution:

```python
def reciprocal_rank_at_k(recommended_ids, test_item_id, k=10):
    recommended_ids = recommended_ids[:k]

    if test_item_id not in recommended_ids:
        return 0.0

    rank = recommended_ids.index(test_item_id) + 1
    return 1 / rank
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
Precision@K asks what share of the shown K items is relevant.
Recall@K asks whether the hidden relevant item was recovered.
MRR@K rewards putting the first relevant item near the top.
NDCG@K rewards placing the relevant item higher.

With one hidden item per user, HitRate@K and Recall@K are numerically the same.
In real datasets with many relevant items per user, they are different.
```

Final-list health metrics:

```python
def catalog_coverage(recommendation_lists):
    recommended_items = set()

    for rec_ids in recommendation_lists:
        recommended_items.update(rec_ids)

    return len(recommended_items) / movies["movieId"].nunique()
```

```python
def genre_diversity_for_list(recommendations):
    genres = []

    for genre_string in recommendations["genres"]:
        genres.extend(genre_string.split("|"))

    return len(set(genres))
```

Explain:

```text
Coverage asks whether the recommender uses a broad part of the catalog.
Diversity asks whether the final list contains variety.
These are not replacements for relevance metrics. They measure a different
product requirement.
```

---

## Step 20 — Evaluate the Ranked Recommender

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
        "precision@10": precision_at_k(rec_ids, test_movie_id, k=10),
        "recall@10": recall_at_k(rec_ids, test_movie_id, k=10),
        "mrr@10": reciprocal_rank_at_k(rec_ids, test_movie_id, k=10),
        "ndcg@10": ndcg_at_k(rec_ids, test_movie_id, k=10)
    })

eval_df = pd.DataFrame(results)

eval_df[["hit@10", "precision@10", "recall@10", "mrr@10", "ndcg@10"]].mean()
```

Explain:

```text
This measures whether the final ranked top-10 list recovers the user's hidden future liked movie.
This is a ranking-stage metric, not a candidate-generation metric.
```

---

## Step 21 — Evaluate Reranked Recommendations

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
        "precision@10": precision_at_k(rec_ids, test_movie_id, k=10),
        "recall@10": recall_at_k(rec_ids, test_movie_id, k=10),
        "mrr@10": reciprocal_rank_at_k(rec_ids, test_movie_id, k=10),
        "ndcg@10": ndcg_at_k(rec_ids, test_movie_id, k=10)
    })

rerank_eval_df = pd.DataFrame(rerank_results)
rerank_eval_df[["hit@10", "precision@10", "recall@10", "mrr@10", "ndcg@10"]].mean()
```

Discuss:

```text
Reranking may improve diversity but can reduce pure relevance metrics.
This is a real product trade-off.
```

Also compare list health:

```python
ranked_lists = []
reranked_lists = []

for _, row in test.iterrows():
    user_id = row["userId"]

    raw_recs = recommend_with_ranker(user_id, top_k=50, n_per_source=100)

    if len(raw_recs) == 0:
        ranked_lists.append([])
        reranked_lists.append([])
        continue

    ranked_lists.append(raw_recs.head(10)["movieId"].tolist())

    reranked = diversity_rerank(raw_recs, top_k=10, max_per_genre=3)
    reranked_lists.append(reranked["movieId"].tolist())

print("Ranked coverage:", catalog_coverage(ranked_lists))
print("Reranked coverage:", catalog_coverage(reranked_lists))
```

---

# 7. Discussion: What This Pipeline Represents

The notebook simulates a real-world structure:

```text
Candidate Generation:
- popular candidates
- content-based candidates
- item-item CF candidates
- two-tower-style embedding retrieval
- PyTorch two-tower retrieval

Ranking:
- supervised ML model
- user/item/genre features

Reranking:
- diversity constraint

Evaluation:
- candidate recall
- HitRate@K
- Precision@K
- Recall@K
- MRR@K
- NDCG@K
- coverage
- diversity
```

This is not production-grade, but it teaches the main architecture.

It also makes the cost problem explicit:

```text
Dense item-item matrices are easy in MovieLens but can become impossible
when item count grows.

Full user-item scoring is easy in a notebook but too expensive when both
users and items are large.

Production systems separate offline recomputation from online request-time
serving to stay inside latency budgets.
```

Key reason the stages are separated:

```text
Candidate generation optimizes "do not miss relevant items".
Ranking optimizes "put the best candidates first".
Reranking optimizes "make the final list useful and acceptable as a product".
Serving optimizes "return the list reliably and quickly".
Logging/evaluation optimizes "learn whether the system actually helped".
```

---

# 8. What Real Production RecSys Adds

## 8.1 Candidate Generation at Scale

Real systems often use:

- approximate nearest neighbor search;
- two-tower retrieval models;
- embedding retrieval;
- graph-based retrieval;
- co-visitation matrices;
- query-based retrieval;
- real-time trending items;
- sponsored or business-driven candidates.

Two-tower production shape:

```text
Offline:
train towers -> compute item embeddings -> build vector index

Online:
compute user embedding -> ANN search -> candidate item IDs
```

Recomputation costs:

```text
Item-item similarity:
recompute when interaction history changes enough;
cost grows roughly with items^2 and can become storage-heavy.

Item embeddings:
compute offline for all items;
storage grows with items * embedding_dim;
can be refreshed daily/hourly or incrementally for new items.

User embeddings/features:
often updated more frequently because user intent changes quickly.

Vector index:
rebuilt or incrementally updated after item embeddings change.
```

The PyTorch model in this notebook is a toy version of this pattern:

```text
nn.Embedding(user_id) -> user tower -> user vector
nn.Embedding(movie_id) -> item tower -> item vector
dot product -> relevance logit
```

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

Offline metrics by stage:

```text
Candidate generation:
Candidate Recall@N, source contribution, source overlap, latency.
For the PyTorch two-tower, training loss is only a debugging signal.
The retrieval source still needs Candidate Recall@N.

Ranking:
HitRate@K, Precision@K, Recall@K, MRR@K, NDCG@K.

Reranking:
diversity, novelty, coverage, freshness, constraint violations.

Serving:
p50/p95/p99 latency, error rate, fallback rate.
```

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
two-tower-style candidates only
PyTorch two-tower candidates only
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
Does it improve Precision@10, MRR@10, or NDCG@10?
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
How do HitRate@10, Precision@10, MRR@10, and NDCG@10 change?
What would be the right product decision?
```

---

# 10. Concepts to Avoid Going Too Deep Into

For this second introductory session, do not deeply cover:

- neural two-tower training internals;
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
4. Two-tower retrieval is a scalable way to generate personalized candidates.
5. Ranking is usually a supervised ML problem over user-item rows.
6. Negative sampling is a practical approximation, not perfect truth.
7. Reranking represents product constraints beyond pure model score.
8. Different stages require different metrics.
9. Offline evaluation is useful but incomplete.
10. Production RecSys requires online testing and monitoring.

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
                  │ - Two-tower retrieval│
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
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Logging + Metrics    │
                  │ offline + online     │
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
Measure candidate quality, ranking quality, list quality, serving health,
and online product impact.
```

The key idea:

```text
Recommendation is not only prediction.
Recommendation is retrieval, ranking, decision-making, serving, and learning
under product constraints.
```

