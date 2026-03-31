# Lecture Examples

This directory contains the lecture example notebooks for Lecture 10: Explainability and Interpretability.

## Files

- `example_01.ipynb` and `example_01.py`
  Title: Interpretability and Explainability
  Summary: Model explainability workflow centered on PiML.
  Optional setup note: this example uses PiML. Install the lecture-specific extras with `uv sync --group xai_piml` if you are using a compatible Python version. The default Python 3.12 environment does not install PiML because compatible wheels are not available.
- `example_02.ipynb` and `example_02.py`
  Title: Key Components of the ALE Plot
  Summary: Interpretability methods with ALE, SHAP, LIME, Alibi, and Interpret.

## Notes

- These notebooks are examples that accompany the lecture.
- Each notebook has a paired `.py` script generated for local reading and version control.
- Some notebooks still contain Colab install cells in the `.ipynb` source; after `uv sync`, those cells can usually be skipped locally. For lecture-specific extras such as Prophet or PyTorch, install the matching optional `uv` group first.
- These examples are not the same thing as separate classroom practical sessions.
- When a lecture also has a dedicated practical session, it lives in `practical_session/`.
