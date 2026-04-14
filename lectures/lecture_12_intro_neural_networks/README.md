# Lecture 12: Introduction to Neural Networks

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: MNIST dataset. Neural-network introduction on MNIST with PyTorch.
  Optional setup note: this example uses PyTorch. The baseline environment does not install it by default. Install the lecture-specific extras with `uv sync --group neural_networks`.

## Practical Session

- `practical_session/nn_practical_student_90min.ipynb` and `practical_session/nn_practical_student_90min.py`
- Three parts: NumPy from-scratch neurons, PyTorch regression (California Housing), PyTorch classification (FashionMNIST).
- Bonus CPU vs GPU speed comparison.
- Auto-detects CUDA / MPS / CPU and works on Google Colab without manual setup.
- See `practical_session/README.md` for details.

---

[← Previous](../lecture_11_explainability_interpretability/README.md) | [All Lectures](../README.md) | [Next →](../lecture_13_responsible_ai/README.md)
