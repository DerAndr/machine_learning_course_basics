# Data Preparation Part 2 Practical Session

This directory contains a 90-minute classroom practical for Lecture 03.

## Files

- `data_preparation_part2_practical_student_90min.ipynb`
- `data_preparation_part2_practical_student_90min.py`
- `README.md`

## Format

- The student notebook contains targeted TODO placeholders and short answer cells instead of full solutions.
- The Python companion script mirrors the notebook structure for lighter review and diffing.
- The practical uses the Ames Housing dataset from **OpenML dataset `41211`** as one shared tabular case study across feature generation, feature selection, dimensionality reduction, and leakage-safe validation.
- The notebook renames the OpenML columns into the course's space-separated Ames style so it stays aligned with Lecture 02 and the lecture examples.
- The notebook is organized into four classroom blocks:
  - feature generation and audit
  - feature selection
  - dimensionality reduction
  - validation and pipelines

## Teaching Intent

- Move students from basic cleaning toward more strategic preprocessing decisions.
- Show that feature engineering and feature selection are useful only when they improve generalization rather than produce arbitrary complexity.
- Reinforce that dimensionality reduction solves a different problem than feature selection.
- Make leakage prevention concrete with a short pre-filled leakage contrast before the final reusable sklearn pipeline and cross-validation workflow.
