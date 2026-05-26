# Lecture 17 Notes: Recommender Systems

Recommender systems choose and rank items for users. Unlike standard supervised learning, the goal is usually not to predict one label for one row, but to produce a useful top-K list under product and system constraints.

## Core Ideas

- **Users, items, and interactions** form the basic data model.
- **Explicit feedback** includes ratings or reviews; **implicit feedback** includes clicks, views, purchases, plays, skips, or dwell time.
- **Missing interactions are unknown**, not automatically negative labels.
- **Popularity baselines** are simple, strong, and important to beat.
- **Content-based recommenders** use item metadata such as genres, tags, text, or embeddings.
- **Collaborative filtering** uses patterns in user-item behavior.
- **Matrix factorization** represents users and items with latent vectors and scores a pair with a dot product.
- **Leave-last-out evaluation** hides a user's later interaction and checks whether the recommender can recover it.
- **Top-K metrics** such as HitRate, Precision, Recall, MRR, and NDCG evaluate ranked recommendation lists.

## Production Pipeline

Production recommender systems are usually multi-stage:

```text
Candidate generation -> Ranking -> Reranking -> Serving -> Logging -> Evaluation
```

Candidate generation finds many plausible items quickly and is usually optimized for recall. Ranking orders those candidates with richer features or models. Reranking applies product constraints such as diversity, freshness, business rules, safety, or user experience limits.

## Practical Focus

The practical uses MovieLens Latest Small to build:

1. a popularity recommender;
2. a genre-based content recommender;
3. an item-item collaborative filtering recommender;
4. a matrix factorization intuition demo;
5. a leave-last-out evaluation workflow;
6. a production-style extension with candidate sources, retrieval, ranking, reranking, and stage-specific metrics.

## Key Takeaways

- Recommenders are ranking systems, not only prediction models.
- A good offline evaluation split must avoid leaking hidden future interactions into the model.
- Retrieval, ranking, and reranking optimize different parts of the user experience.
- Training loss and AUC are not enough for RecSys; final quality needs top-K and product metrics.
- Production RecSys design is constrained by latency, storage, recomputation cost, monitoring, and feedback loops.
