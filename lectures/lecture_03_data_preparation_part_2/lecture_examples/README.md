# Lecture Examples

This directory contains the lecture example notebooks for Lecture 03: Data Preparation Part 2.

## Files

- `example_01.ipynb` and `example_01.py`
  Title: Feature Selection
  Summary: Feature selection with filter, wrapper, and embedded methods.
- `example_02.ipynb` and `example_02.py`
  Title: Feature Generation
  Summary: Feature generation patterns and practical feature engineering.
- `example_03.ipynb` and `example_03.py`
  Title: Dimentionality Reduction
  Summary: Dimensionality reduction with PCA and manifold-learning methods.
- `example_04.ipynb` and `example_04.py`
  Title: Data Splitting & Cross-Validation
  Summary: Data splitting, stratification, resampling, and cross-validation setup.
- `example_05.ipynb` and `example_05.py`
  Title: Data Leakage Examples
  Summary: Leakage scenarios and why they invalidate evaluation.
- `example_06.ipynb` and `example_06.py`
  Title: Pipelines
  Summary: Pipelines for safe preprocessing and modeling workflows.

## Notes

- These notebooks are examples that accompany the lecture.
- Each notebook has a paired `.py` script generated for local reading and version control.
- Some notebooks still contain Colab install cells in the `.ipynb` source; after `uv sync`, those cells can usually be skipped locally. For lecture-specific extras such as Prophet or PyTorch, install the matching optional `uv` group first.
- These examples are not the same thing as separate classroom practical sessions.
- When a lecture also has a dedicated practical session, it lives in `practical_session/`.
