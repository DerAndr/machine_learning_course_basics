# Lecture Examples

This directory contains the lecture example notebooks for Lecture 07: Ensembles.

## Files

- `example_01.ipynb` and `example_01.py`
  Title: Ensemble Models and Techniques with Hyperparameter Tuning, PR Curves, and Class Separation Diagrams
  Summary: Bagging, boosting, and tree ensembles with tuning and comparison.
  Optional setup note: this example uses CatBoost, LightGBM, and XGBoost. The baseline environment does not install them by default. Install the lecture-specific extras with `uv sync --group ensembles`.

## Notes

- These notebooks are examples that accompany the lecture.
- Each notebook has a paired `.py` script generated for local reading and version control.
- Some notebooks still contain Colab install cells in the `.ipynb` source; after `uv sync`, those cells can usually be skipped locally. For lecture-specific extras such as Prophet or PyTorch, install the matching optional `uv` group first.
- These examples are not the same thing as separate classroom practical sessions.
- When a lecture also has a dedicated practical session, it lives in `practical_session/`.
