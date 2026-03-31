# Lecture 11: Cross-Validation and Hyperparameter Optimization

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: SECTION 0: SETUP - Installations and Imports. Cross-validation, pipelines, and hyperparameter search on classification and regression demos.
  Optional setup note: this example uses H2O AutoML. The baseline environment does not install it by default. Install the lecture-specific extras with `uv sync --group hpo_automl`.
- `lecture_examples/example_02.ipynb` and `lecture_examples/example_02.py`: Example 02. Alternative hyperparameter-optimization libraries such as Optuna, Hyperopt, and scikit-optimize.
  Optional setup note: this example uses Optuna, Hyperopt, and scikit-optimize. The baseline environment does not install them by default. Install the lecture-specific extras with `uv sync --group hpo_automl`.
