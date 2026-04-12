# Data Preparation Part 1 Practical Session

This directory contains a 90-minute classroom practical for Lecture 02.

## Files

- `data_preparation_part1_practical_student_90min.ipynb`
- `README.md`

## Format

- The student notebook contains targeted TODO placeholders and short answer cells instead of full solutions.
- The practical uses the Ames Housing dataset as a realistic tabular example with missing values, mixed feature types, and strong outliers.
- The notebook is organized as one continuous 90-minute flow:
  - setup and raw data inspection
  - missing-value analysis and median imputation
  - outlier thresholds, skewness, and log transforms
  - ordinal encoding, one-hot reasoning, and quantile binning
  - feature engineering, robust scaling, and a final preprocessing pipeline with `Ridge`

## Teaching Intent

- Turn preprocessing topics into concrete quantitative exercises instead of abstract API demos.
- Make students compute the effect of preprocessing choices, not only apply them mechanically.
- Connect Lecture 02 topics to a simple end-to-end modeling pipeline without turning the session into a full modeling lecture.
