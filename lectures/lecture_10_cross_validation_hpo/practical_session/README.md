# Lecture 10 Practical Session

This directory contains a 90-minute classroom practical for Lecture 10.

## Files

- `crossval_hpo_practical_student_90min.ipynb`
- `crossval_hpo_practical_student_90min.py`
- `README.md`

## Format

- The student notebook contains targeted TODO placeholders in the main evaluation and tuning cells.
- The Python companion script mirrors the notebook structure for lighter review and diffing.
- The optional `Optuna`, `H2O AutoML`, and `FLAML` blocks extend the main teaching path, which otherwise uses only the core `scikit-learn` stack.

## Teaching Intent

- Show why poor validation design can produce misleadingly optimistic metrics.
- Contrast accuracy, balanced accuracy, and minority-class recall on an imbalanced classification problem.
- Demonstrate the effect of leakage, then compare `KFold` with `StratifiedKFold`.
- Add two more CV designs that matter in the lecture: `RepeatedStratifiedKFold` and a small-scope `LOOCV` demonstration.
- Use a validation curve, randomized search, Optuna-based adaptive search, and nested CV in one consistent workflow.
- End with a pipeline block that shows why preprocessing must live inside the cross-validation loop.
- Refresh the AutoML discussion with both `H2O AutoML` and `FLAML`, without turning the practical into a tool benchmark.
- Keep AutoML visible as an optional extension, not a substitute for evaluation discipline.
