# Local Setup

## Requirements

- Python 3.12
- `uv`

## Initial Setup

```bash
uv sync
uv run python tools/check_notebook_environment.py
```

This installs the baseline course environment from `pyproject.toml` and verifies that the shared notebook runtime packages are available.

There is no separate `requirements.txt`.

- `pyproject.toml` is the source of truth for dependencies.
- `uv.lock` is the reproducible lockfile.
- Heavy lecture-specific packages are installed only when needed.

## Optional Lecture Groups

Install one of these only for the relevant lecture:

```bash
uv sync --group ensembles
uv sync --group time_series
uv sync --group hpo_automl
uv sync --group neural_networks
uv sync --group xai_piml
```

Current usage:

- `ensembles`: Lecture 07 extras such as `catboost`, `lightgbm`, and `xgboost`
- `time_series`: Lecture 08 extras such as `prophet`
- `hpo_automl`: Lecture 11 extras such as `h2o`, `optuna`, `hyperopt`, and `scikit-optimize`
- `neural_networks`: Lecture 12 extras such as `torch` and `torchvision`
- `xai_piml`: the optional PiML example in Lecture 10 on compatible Python versions

To validate one of these environments explicitly:

```bash
uv run python tools/check_notebook_environment.py --group ensembles
uv run python tools/check_notebook_environment.py --group time_series
uv run python tools/check_notebook_environment.py --group hpo_automl
uv run python tools/check_notebook_environment.py --group neural_networks
```

## Common Commands

Start JupyterLab:

```bash
uv run jupyter lab
```

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

Run type checks:

```bash
uv run ty check src tests
```

Run tests:

```bash
uv run pytest
```

Refresh the generated `.py` companions for lecture notebooks:

```bash
uv run python tools/sync_lecture_examples.py
```

## Working Assumption

Use `uv sync` as the baseline environment for:

- editing notebooks
- regenerating notebook companion scripts
- formatting and linting repository code
- validating repository structure

For day-to-day command usage, see `docs/tooling-guide.md`.
