# Data Preparation Part 1 Practical Session

This directory contains a 90-minute classroom practical for Lecture 02.

## Files

- `data_preparation_part1_practical_student_90min.ipynb`
- `data_preparation_part1_practical_student_90min.py`
- `README.md`

## Format

- The student notebook contains targeted TODO placeholders and short answer cells instead of full solutions.
- The Python companion script mirrors the notebook structure for lighter review and diffing.
- The practical uses the Ames Housing dataset as a realistic tabular example with missing values, mixed feature types, and strong outliers.
- The notebook is organized into four classroom blocks:
  - missing-value inspection, quick missingness scanning, and imputation
  - outliers and transformations
  - encodings and binning
  - feature engineering, scaling, and a final preprocessing pipeline with `Ridge`

## Teaching Intent

- Turn preprocessing topics into concrete quantitative exercises instead of abstract API demos.
- Make students compute the effect of preprocessing choices, not only apply them mechanically.
- Reinforce that some Ames Housing missing values mean "feature not present", while others are genuinely missing measurements that require a different treatment.
- Connect Lecture 02 topics to a simple end-to-end modeling pipeline without turning the session into a full modeling lecture.
