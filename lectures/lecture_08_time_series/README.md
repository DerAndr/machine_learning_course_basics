# Lecture 08: Time Series

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `practical_session/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: Data Preparation Function. Forecasting pipeline with SARIMA, Random Forest, Prophet, and XGBoost.
  Optional setup note: this example uses Prophet and XGBoost. The baseline environment does not install them by default. Install the lecture-specific extras with `uv sync --group time_series`.

## Practical Session

- `practical_session/time_series_practical_student_90min.ipynb`: student version with targeted TODO cells
- `practical_session/time_series_practical_student_90min.py`: generated companion script for the student notebook
- `practical_session/README.md`: practical overview, scope, and runtime notes

The practical covers:

- shared diagnostics with STL, ACF, and PACF
- a short `KFold` vs `TimeSeriesSplit` comparison
- ARIMAX-style model, SARIMAX, Random Forest, CatBoost, and Prophet
- an optional `tsfresh` section at the end

The practical session is intentionally separate from `lecture_examples/`.

---

[← Previous](../lecture_07_ensembles/README.md) | [All Lectures](../README.md) | [Next →](../lecture_09_clustering/README.md)
