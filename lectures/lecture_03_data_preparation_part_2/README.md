# Lecture 03: Data Preparation Part 2

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `practical_session/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: Feature Selection. Feature selection with filter, wrapper, and embedded methods.
- `lecture_examples/example_02.ipynb` and `lecture_examples/example_02.py`: Feature Generation. Feature generation patterns and practical feature engineering.
- `lecture_examples/example_03.ipynb` and `lecture_examples/example_03.py`: Dimensionality Reduction. Dimensionality reduction with PCA and manifold-learning methods.
- `lecture_examples/example_04.ipynb` and `lecture_examples/example_04.py`: Data Splitting & Cross-Validation. Data splitting, stratification, resampling, and cross-validation setup.
- `lecture_examples/example_05.ipynb` and `lecture_examples/example_05.py`: Data Leakage Examples. Leakage scenarios and why they invalidate evaluation.
- `lecture_examples/example_06.ipynb` and `lecture_examples/example_06.py`: Pipelines. Pipelines for safe preprocessing and modeling workflows.

## Practical Session

- `practical_session/data_preparation_part2_practical_student_90min.ipynb`: student practical with targeted TODO cells across feature generation, selection, dimensionality reduction, and pipelines
- `practical_session/data_preparation_part2_practical_student_90min.py`: Python companion script for the practical notebook
- `practical_session/README.md`: practical overview, scope, and runtime notes

The practical covers:

- repairing key missing values and building interpretable engineered features on the same 2930-row Ames variant used across the course
- comparing filter, wrapper, and embedded feature-selection strategies
- PCA and UMAP as complementary dimensionality-reduction tools
- an explicit leakage contrast plus leakage-safe preprocessing with a `ColumnTransformer`, `Pipeline`, and cross-validation

The practical session is intentionally separate from `lecture_examples/`.

---

[← Previous](../lecture_02_data_preparation_part_1/README.md) | [All Lectures](../README.md) | [Next →](../lecture_04_regression/README.md)
