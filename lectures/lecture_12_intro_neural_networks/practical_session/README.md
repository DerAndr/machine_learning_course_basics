# Lecture 12 Practical Session

This directory contains a 90-minute classroom practical for Lecture 12.

## Files

| File | Description |
| --- | --- |
| `nn_practical_student_90min.ipynb` | Student notebook with 12 TODO placeholders |
| `nn_practical_student_90min.py` | Generated companion script for review and diffing |
| `README.md` | This file |

## Structure

The practical has three self-contained parts that let teams work in parallel and regroup for the debrief:

| Part | Topic | Framework | Dataset |
| --- | --- | --- | --- |
| 1 | Forward pass and manual back-propagation | NumPy | Synthetic |
| 2 | Regression with PyTorch | PyTorch | California Housing |
| 3 | Classification with PyTorch | PyTorch | FashionMNIST |

**Bonus section:** CPU vs GPU speed comparison (runs automatically when a GPU is available).

## Format

- The student notebook contains targeted TODO placeholders in the main implementation cells.
- The Python companion script mirrors the notebook structure for lighter review and diffing.
- The notebook auto-detects the compute device (CUDA / MPS / CPU) and moves all models and tensors to it, so it runs correctly on Colab GPU, Apple Silicon, and plain CPU.
- On Google Colab the notebook auto-installs missing packages on first run.

## Teaching Intent

- Build intuition by implementing a forward pass and manual back-propagation from scratch in pure NumPy before touching a framework.
- Contrast manual gradient computation with PyTorch autograd to motivate why frameworks exist.
- Cover both regression (California Housing) and classification (FashionMNIST) tasks in PyTorch.
- Introduce regularisation techniques (Dropout, BatchNorm) in practice with minimal boilerplate.
- Use `torchinfo` to display model architecture summaries.
- Use real datasets rather than toy data so that evaluation metrics are meaningful.
- Emphasise evaluation (RMSE, accuracy) and error analysis (visualising misclassified images).
- Scale naturally from a single neuron to a three-layer network to a PyTorch `nn.Module`.
- Demonstrate the GPU speed advantage with a built-in CPU vs GPU benchmark.

## Environment

Local:

```bash
uv sync --group neural_networks
```

Google Colab: no manual setup needed — the notebook auto-installs dependencies.

## Recommended Reading

- [PyTorch — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/)
