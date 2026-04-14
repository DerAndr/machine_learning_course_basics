# Explainability and Interpretability Practical Session

This directory contains a 90-minute classroom practical for Lecture 11.

## Files

- `xai_practical_student_90min.ipynb`
- `xai_practical_student_90min.py`
- `README.md`

An internal teacher notebook and cheat sheet exist in the private materials layer, but they are not part of the public student release.

## Format

- The student notebook contains targeted TODO placeholders in the model-training, explanation, and interpretation cells.
- The Python companion script mirrors the notebook structure for lighter review and diffing.
- The practical uses one consistent binary-classification dataset so students can compare explanation methods without re-learning a new problem each time.
- The practical explicitly teaches students how to read each required plot instead of only generating figures.
- The main tooling path uses `scikit-learn`, `alibi`, `lime`, `shap`, and `interpret`.
- `eli5` is mentioned as another ecosystem tool for model inspection, but it is not required in this notebook.

## Teaching Intent

- Contrast an intrinsically interpretable baseline with a stronger but more opaque tree ensemble.
- Use Permutation Feature Importance as the first global post-hoc explanation because it is simple, visual, and methodologically important.
- Compare PDP and ALE on correlated features so students see why different global-effect plots can disagree.
- Keep LIME strictly local and SHAP explicitly split into global (`beeswarm`) and local (`waterfall`) use cases.
- Include small case-by-case local explanation galleries so students can inspect several individual predictions separately, not only one example.
- Add a lightweight SHAP interaction view so the practical also touches the interaction-effects layer without overloading students.
- Introduce `InterpretML` through an Explainable Boosting Machine so students see that modern glass-box models can still be competitive.
- End with interpretation questions, not only code execution, so students practice reading explanation outputs critically.

## Environment

Run this practical with the baseline repository environment:

- `uv sync`

If you work in Google Colab and the runtime is missing packages, install:

- `shap`
- `lime`
- `alibi`
- `interpret`

## Recommended Reading

- Christoph Molnar, *Interpretable Machine Learning*: [online book](https://christophm.github.io/interpretable-ml-book)
