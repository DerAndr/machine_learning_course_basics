# Lecture 17 Practical Session

This directory contains the recommender systems foundations practical for an instructor-led demo.

## Files

| File | Description |
| --- | --- |
| `recsys_practical_student_90min.ipynb` | Student notebook with guided TODO cells and by-hand exercises |
| `recsys_practical_student_90min.py` | Student companion script for review and diffing |
| `recsys_practical_teacher_90min.ipynb` | Teacher notebook with full demo code and solutions |
| `recsys_practical_teacher_90min.py` | Teacher companion script for review and diffing |
| `teacher_cheat_sheet.md` | Teaching quick-reference |

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

No GPU is needed. The practical uses pandas, NumPy, scikit-learn, Matplotlib, and Seaborn.

## Notes

- The teacher notebook is the canonical demo notebook.
- The student notebook keeps the same narrative but replaces selected cells with TODOs.
- MovieLens Latest Small is downloaded from `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`.
- The evaluation section intentionally rebuilds item similarities on `train` only to avoid leaking hidden test interactions.
- Matrix factorization is included as a by-hand numeric demo. SGD, ALS, and production-scale retrieval are left for later material.
