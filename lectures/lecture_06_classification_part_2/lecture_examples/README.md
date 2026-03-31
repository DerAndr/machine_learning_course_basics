# Lecture Examples

This directory contains the lecture example notebooks for Lecture 06: Classification Part 2.

## Files

- `example_01.ipynb` and `example_01.py`
  Title: Imbalanced Binary Classification
  Summary: Imbalanced binary classification with resampling and threshold-aware metrics.
- `example_02.ipynb` and `example_02.py`
  Title: Multiclass Classification with Logistic Regression using One-vs-Rest and One-vs-One Strategies
  Summary: Multiclass classification with native, OvR, and OvO logistic strategies.
- `example_03.ipynb` and `example_03.py`
  Title: Multilabel Classification with Logistic Regression and Evaluation Metrics
  Summary: Multilabel classification and multilabel metric interpretation.
- `example_04.ipynb` and `example_04.py`
  Title: Ordinal Classification with Ordinal Logistic Regression
  Summary: Ordinal classification with ordered targets and ordinal regression.

## Notes

- These notebooks are examples that accompany the lecture.
- Each notebook has a paired `.py` script generated for local reading and version control.
- Some notebooks still contain Colab install cells in the `.ipynb` source; after `uv sync`, those cells can usually be skipped locally. For lecture-specific extras such as Prophet or PyTorch, install the matching optional `uv` group first.
- These examples are not the same thing as separate classroom practical sessions.
- When a lecture also has a dedicated practical session, it lives in `practical_session/`.
