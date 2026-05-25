# Lecture 17 Practical Session

This directory contains the recommender systems foundations practical.

## Files

| File | Description |
| --- | --- |
| `recsys_practical_student_90min.ipynb` | Student notebook with guided TODO cells and by-hand exercises |
| `recsys_practical_student_90min.py` | Student companion script for review and diffing |

## Structure

The practical follows the first recommender-systems storyline:

| Part | Topic | Dataset / assets |
| --- | --- | --- |
| 1 | Dataset and RecSys framing | MovieLens Latest Small |
| 2 | Popularity baseline | Ratings, movie titles, Bayesian score |
| 3 | Implicit feedback | `rating >= 4.0` as a classroom relevance definition |
| 4 | Content-based recommendation | Movie genres and cosine similarity |
| 5 | Item-item collaborative filtering | User-item matrix and item cosine similarity |
| 6 | Matrix factorization intuition | Tiny two-factor hand calculation, shape/rank/SVD notes, and plain matrix tables |
| 7 | Leave-last-out evaluation | Train-only item similarity and HitRate@10 |

## Environment

Google Colab: the notebook downloads MovieLens Latest Small directly from GroupLens.

Local:

```bash
uv sync
uv run jupyter lab
```

No GPU is needed for the foundations practical. The production-pipeline plan also includes optional PyTorch and Colab/T4 discussion for instructor demos.

## Notes

- The student notebook is the public practical notebook.
- MovieLens Latest Small is downloaded from `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`.
- The evaluation section intentionally rebuilds item similarities on `train` only to avoid leaking hidden test interactions.
- Matrix factorization is included as a by-hand numeric demo.
- The Part 2 plan covers production-style RecSys: candidate generation, two-tower retrieval, ranking, reranking, metrics, latency, storage, and recomputation costs.
