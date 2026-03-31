# Lecture Examples

This directory contains the lecture example notebooks for Lecture 12: Introduction to Neural Networks.

## Files

- `example_01.ipynb` and `example_01.py`
  Title: MNIST dataset
  Summary: Neural-network introduction on MNIST with PyTorch.
  Optional setup note: this example uses PyTorch. The baseline environment does not install it by default. Install the lecture-specific extras with `uv sync --group neural_networks`.

## Notes

- These notebooks are examples that accompany the lecture.
- Each notebook has a paired `.py` script generated for local reading and version control.
- Some notebooks still contain Colab install cells in the `.ipynb` source; after `uv sync`, those cells can usually be skipped locally. For lecture-specific extras such as Prophet or PyTorch, install the matching optional `uv` group first.
- These examples are not the same thing as separate classroom practical sessions.
- When a lecture also has a dedicated practical session, it lives in `practical_session/`.
