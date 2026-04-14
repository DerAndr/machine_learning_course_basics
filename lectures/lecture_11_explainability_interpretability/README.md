# Lecture 11: Explainability and Interpretability

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `practical_session/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: Interpretability and Explainability. Model explainability workflow centered on PiML.
  Optional setup note: this example uses PiML. Install the lecture-specific extras with `uv sync --group xai_piml` if you are using a compatible Python version. The default Python 3.12 environment does not install PiML because compatible wheels are not available.
- `lecture_examples/example_02.ipynb` and `lecture_examples/example_02.py`: Key Components of the ALE Plot. Interpretability methods with ALE, SHAP, LIME, Alibi, and Interpret.

## Practical Session

- `practical_session/xai_practical_student_90min.ipynb`: public student practical on white-box vs black-box explanations, PFI, PDP, ALE, LIME, SHAP, a lightweight SHAP interaction view, and InterpretML
- `practical_session/xai_practical_student_90min.py`: Python companion script for the practical notebook
- `practical_session/README.md`: practical overview, scope, runtime notes, and teaching intent

The internal teacher notebook and cheat sheet are maintained separately and are not part of the current public student release.

## Recommended Reading

- Christoph Molnar, *Interpretable Machine Learning*: [online book](https://christophm.github.io/interpretable-ml-book)

---

[← Previous](../lecture_10_cross_validation_hpo/README.md) | [All Lectures](../README.md) | [Next →](../lecture_12_intro_neural_networks/README.md)
