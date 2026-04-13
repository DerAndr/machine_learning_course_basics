# Lecture 04: Regression

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `practical_session/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: Regression Demo. Regression workflow with fitting, diagnostics, and evaluation.

## Practical Session

- `practical_session/regression_practical_student_90min.ipynb`: student practical with targeted TODO cells across visual audit, baseline linear regression, regularization, tree-based models, and residual diagnostics
- `practical_session/regression_practical_student_90min.py`: Python companion script for the practical notebook
- `practical_session/README.md`: practical overview, scope, and runtime notes

The practical covers:

- visual regression diagnostics on the Auto MPG dataset
- ratio-based feature engineering and multicollinearity checks with correlations and VIF
- a leakage-safe preprocessing + linear regression pipeline
- Ridge, Lasso, and tree-based regression comparisons
- residual diagnostics for the best held-out model

The practical session is intentionally separate from `lecture_examples/`.

---

[← Previous](../lecture_03_data_preparation_part_2/README.md) | [All Lectures](../README.md) | [Next →](../lecture_05_classification_part_1/README.md)
