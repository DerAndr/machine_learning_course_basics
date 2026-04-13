# Classification Part 1 Practical Session

This directory contains a 90-minute classroom practical for Lecture 05.

## Files

- `classification_part1_practical_student_90min.ipynb`
- `classification_part1_practical_student_90min.py`
- `README.md`

## Format

- The student notebook contains targeted TODO placeholders in the main implementation sections rather than a fully blank workflow.
- The Python companion script mirrors the notebook structure for lighter review and diffing.
- The practical keeps the Optuna section optional, so the main classification workflow still runs in the baseline repository environment.
- The notebook can be split across two student groups:
  - Group A: `KNN` + `DecisionTree` + validation curves
  - Group B: `LogisticRegression` + threshold tuning + interpretation

## Teaching Intent

- Show why classification evaluation is not the same as raw accuracy on an imbalanced dataset.
- Compare three baseline classification families with clearly different inductive biases.
- Make threshold choice a first-class part of the workflow rather than a postscript.
- End with interpretable diagnostics: coefficients, permutation importance, and probability distributions.
