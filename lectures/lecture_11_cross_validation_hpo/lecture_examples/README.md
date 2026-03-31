# Lecture Examples

This directory contains the lecture example notebooks for Lecture 11: Cross-Validation and Hyperparameter Optimization.

## Files

- `example_01.ipynb` and `example_01.py`
  Title: SECTION 0: SETUP - Installations and Imports
  Summary: Cross-validation, pipelines, and hyperparameter search on classification and regression demos.
  Optional setup note: this example uses H2O AutoML. The baseline environment does not install it by default. Install the lecture-specific extras with `uv sync --group hpo_automl`.
- `example_02.ipynb` and `example_02.py`
  Title: Example 02
  Summary: Alternative hyperparameter-optimization libraries such as Optuna, Hyperopt, and scikit-optimize.
  Optional setup note: this example uses Optuna, Hyperopt, and scikit-optimize. The baseline environment does not install them by default. Install the lecture-specific extras with `uv sync --group hpo_automl`.

## Notes

- These notebooks are examples that accompany the lecture.
- Each notebook has a paired `.py` script generated for local reading and version control.
- Some notebooks still contain Colab install cells in the `.ipynb` source; after `uv sync`, those cells can usually be skipped locally. For lecture-specific extras such as Prophet or PyTorch, install the matching optional `uv` group first.
- These examples are not the same thing as separate classroom practical sessions.
- When a lecture also has a dedicated practical session, it lives in `practical_session/`.
