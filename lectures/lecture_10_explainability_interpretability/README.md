# Lecture 10: Explainability and Interpretability

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: Interpretability and Explainability. Model explainability workflow centered on PiML.
  Optional setup note: this example uses PiML. Install the lecture-specific extras with `uv sync --group xai_piml` if you are using a compatible Python version. The default Python 3.12 environment does not install PiML because compatible wheels are not available.
- `lecture_examples/example_02.ipynb` and `lecture_examples/example_02.py`: Key Components of the ALE Plot. Interpretability methods with ALE, SHAP, LIME, Alibi, and Interpret.
