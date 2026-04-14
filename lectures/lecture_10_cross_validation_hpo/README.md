# Lecture 10: Cross-Validation and Hyperparameter Optimization

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `practical_session/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: SECTION 0: SETUP - Installations and Imports. Cross-validation, pipelines, and hyperparameter search on classification and regression demos.
  Optional setup note: this example uses H2O AutoML. The baseline environment does not install it by default. Install the lecture-specific extras with `uv sync --group hpo_automl`.
- `lecture_examples/example_02.ipynb` and `lecture_examples/example_02.py`: Example 02. Alternative hyperparameter-optimization libraries such as Optuna, Hyperopt, and scikit-optimize.
  Optional setup note: this example uses Optuna, Hyperopt, and scikit-optimize. The baseline environment does not install them by default. Install the lecture-specific extras with `uv sync --group hpo_automl`.

## Practical Session

- `practical_session/crossval_hpo_practical_student_90min.ipynb`: public student practical on phishing-site detection, cross-validation pitfalls, multiple CV designs, randomized search, Optuna, pipelines, nested CV, and optional modern AutoML extensions
- `practical_session/crossval_hpo_practical_student_90min.py`: Python companion script for the practical notebook
- `practical_session/README.md`: practical overview, scope, runtime notes, and teaching intent

The instructor notebook and cheat sheet are maintained separately and are not part of the current public student release.

---

[← Previous](../lecture_09_clustering/README.md) | [All Lectures](../README.md) | [Next →](../lecture_11_explainability_interpretability/README.md)
