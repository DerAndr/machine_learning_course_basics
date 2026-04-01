# Student Quickstart

This repository can be used in two ways:

- as a lecture-material library with PDFs, notes, and example notebooks
- as a local notebook workspace with `uv`

## 0. If you only want to read the materials

You do not need to install anything to:

- read `lecture_notes.md`
- open lecture PDFs
- browse the repository on GitHub

Local setup is only needed when you want to run notebooks on your machine.

## 1. Install the baseline environment

If you only want the standard shared packages used in most lectures, run:

```bash
uv sync
uv run python tools/check_notebook_environment.py
```

This is the recommended starting point.

## 2. Open the notebooks

```bash
uv run jupyter lab
```

Then open the lecture you need:

- `lectures/<lecture_slug>/lecture_notes.md`
- `lectures/<lecture_slug>/slides/lecture.pdf`
- `lectures/<lecture_slug>/lecture_examples/`

## 3. Install extra packages only for specific lectures

Some lectures use heavier libraries that are not installed by default.

Rule:

- start with the baseline environment
- then add only the group needed for the lecture you are working on

### Lecture 07: Ensembles

```bash
uv sync --group ensembles
uv run python tools/check_notebook_environment.py --group ensembles
```

Use this when the lecture example needs:

- `catboost`
- `lightgbm`
- `xgboost`

### Lecture 08: Time Series

```bash
uv sync --group time_series
uv run python tools/check_notebook_environment.py --group time_series
```

Use this when the lecture example needs:

- `prophet`
- `xgboost`

### Lecture 11: Cross-Validation and Hyperparameter Optimization

```bash
uv sync --group hpo_automl
uv run python tools/check_notebook_environment.py --group hpo_automl
```

Use this when the lecture example needs:

- `h2o`
- `optuna`
- `hyperopt`
- `scikit-optimize`

### Lecture 12: Neural Networks

```bash
uv sync --group neural_networks
uv run python tools/check_notebook_environment.py --group neural_networks
```

Use this when the lecture example needs:

- `torch`
- `torchvision`

### Lecture 10: Optional PiML Example

```bash
uv sync --group xai_piml
```

This is optional and only relevant for the PiML-based explainability example.
On Python 3.12, PiML may be unavailable.

## 4. Daily workflow

Typical workflow:

```bash
uv sync
uv run jupyter lab
```

If a lecture README says that extra setup is needed, run the matching `uv sync --group ...` command first.

For most lectures, the baseline environment is enough.

## 5. What to do if a notebook still has `!pip install ...`

Some notebooks come from Colab and still contain install cells.

Local rule:

- first run `uv sync`
- then install any needed optional group
- then skip the Colab install cell if the package is already available locally

## 6. Quick reference

Baseline:

```bash
uv sync
uv run python tools/check_notebook_environment.py
uv run jupyter lab
```

Optional lecture groups:

```bash
uv sync --group ensembles
uv sync --group time_series
uv sync --group hpo_automl
uv sync --group neural_networks
uv sync --group xai_piml
```
