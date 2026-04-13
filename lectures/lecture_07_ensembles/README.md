# Lecture 07: Ensembles

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `practical_session/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: Ensemble Models and Techniques with Hyperparameter Tuning, PR Curves, and Class Separation Diagrams. Bagging, boosting, and tree ensembles with tuning and comparison.
  Optional setup note: this example uses CatBoost, LightGBM, and XGBoost. The baseline environment does not install them by default. Install the lecture-specific extras with `uv sync --group ensembles`.

## Practical Session

- `practical_session/ensembles_practical_student_90min.ipynb`: student version with targeted TODO cells in the main model-building sections
- `practical_session/README.md`: practical overview, scope, and runtime notes

The practical covers:

- Random Forest as a bagging baseline
- gradient boosting with XGBoost, LightGBM, and CatBoost
- a two-group classroom split across classification and regression tasks
- a final comparison of stability, metrics, and interpretability trade-offs

The practical session is intentionally separate from `lecture_examples/`.

---

[← Previous](../lecture_06_classification_part_2/README.md) | [All Lectures](../README.md) | [Next →](../lecture_08_time_series/README.md)
