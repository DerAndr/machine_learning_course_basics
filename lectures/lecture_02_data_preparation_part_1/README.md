# Lecture 02: Data Preparation Part 1

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `practical_session/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: Data Preparation - Missing Values. Missing-value inspection and practical imputation strategies.
- `lecture_examples/example_02.ipynb` and `lecture_examples/example_02.py`: Managing Outliers. Outlier detection, interpretation, and treatment choices.
- `lecture_examples/example_03.ipynb` and `lecture_examples/example_03.py`: Category Encoding Techniques Demonstration on Toy Dataset. Categorical encoding methods on toy and tabular data.

## Practical Session

- `practical_session/data_preparation_part1_practical_student_90min.ipynb`: student version with targeted TODO cells and short answer placeholders
- `practical_session/data_preparation_part1_practical_student_90min.py`: Python companion script for the practical notebook
- `practical_session/README.md`: practical overview, scope, and runtime notes

The practical covers:

- missing-value inspection, a quick missingness scan, and a short discussion of likely missingness mechanism
- the distinction between structurally absent fields and genuinely missing numeric measurements
- median imputation and visual comparison of the original and imputed distribution
- IQR-based outlier thresholds, skewness, log transforms, and Isolation Forest outlier detection
- ordinal encoding, one-hot reasoning, and quantile binning
- feature engineering, robust scaling, and a final preprocessing pipeline with `Ridge`

The practical session is intentionally separate from `lecture_examples/`.

---

[← Previous](../lecture_01_eda/README.md) | [All Lectures](../README.md) | [Next →](../lecture_03_data_preparation_part_2/README.md)
