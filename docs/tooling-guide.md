# Tooling Guide

This repository uses a small, explicit Python toolchain:

- `uv` for environment and command execution
- `ruff` for formatting and linting
- `ty` for type checking shared repository code
- `pytest` for lightweight repository checks
- a small repo-local sync script for notebook companion `.py` files

## Why This Stack

The goal is to keep the course repository:

- easy to bootstrap on a new machine
- strict enough to teach good engineering habits
- simple enough for notebook-based work

## `uv`

Use `uv` as the default entry point for local work.

### Install dependencies

```bash
uv sync
```

This creates or updates the local virtual environment and installs the baseline course runtime plus the authoring toolchain from `pyproject.toml` and `uv.lock`.

Validate the environment after syncing:

```bash
uv run python tools/check_notebook_environment.py
```

### Install lecture-specific extras only when needed

```bash
uv sync --group ensembles
uv sync --group time_series
uv sync --group hpo_automl
uv sync --group neural_networks
uv sync --group xai_piml
```

Current lecture-specific groups:

- `ensembles`: adds `catboost`, `lightgbm`, and `xgboost` for Lecture 07
- `time_series`: adds `prophet` for Lecture 08
- `hpo_automl`: adds `h2o`, `optuna`, `hyperopt`, and `scikit-optimize` for Lecture 11
- `neural_networks`: adds `torch` and `torchvision` for Lecture 12
- `xai_piml`: adds `piml` for the optional PiML example in Lecture 10 on compatible Python versions

Validate an optional lecture environment explicitly:

```bash
uv run python tools/check_notebook_environment.py --group ensembles
uv run python tools/check_notebook_environment.py --group time_series
uv run python tools/check_notebook_environment.py --group hpo_automl
uv run python tools/check_notebook_environment.py --group neural_networks
```

### Run a command inside the project environment

```bash
uv run jupyter lab
uv run pytest
uv run ruff check .
```

Rule:

- prefer `uv run ...` over activating `.venv` manually

### Update the lock file

If you intentionally change dependencies in `pyproject.toml`, refresh the lock file:

```bash
uv lock
uv sync
```

## `ruff`

`ruff` is used for both formatting and linting.

### Format repository code

```bash
uv run ruff format .
```

### Lint repository code

```bash
uv run ruff check .
```

Current scope:

- shared Python code in `src/`
- tests
- documentation-adjacent Python files that are included by the repo config

Current non-goal:

- raw lecture notebook pairs are temporarily excluded from linting while the migrated course materials are still being normalized

## `ty`

`ty` is used for type checking the shared repository code.

### Run type checks

```bash
uv run ty check src tests
```

Current scope:

- `src/`
- `tests/`

Current non-goal:

- full type checking of notebook-derived lecture scripts

This is intentional. The engineering standard is enforced first on reusable code, not on every teaching cell.

## `pytest`

Use `pytest` for repository smoke checks.

### Run tests

```bash
uv run pytest
```

At the moment, tests validate repository-level assumptions rather than lecture notebook behavior.

## Notebook Companion Scripts

Lecture example notebooks live in `lectures/<lecture_slug>/lecture_examples/`.

- `.ipynb` is the source notebook used for execution.
- `.py` is a generated companion script for local reading, diffs, and lightweight execution.

Regenerate the companion scripts and lecture-example READMEs with:

```bash
uv run python tools/sync_lecture_examples.py
```

The sync script also comments out shell magics such as `!pip install ...` in the generated `.py` files so that the script companions stay valid Python.

## Recommended Daily Workflow

1. Sync the environment.

```bash
uv sync
```

If you are preparing a specific lecture with optional heavy dependencies, add the matching group:

```bash
uv sync --group ensembles
uv sync --group time_series
uv sync --group hpo_automl
uv sync --group neural_networks
```

2. Open notebooks.

```bash
uv run jupyter lab
```

3. Regenerate notebook companion scripts after edits.

```bash
uv run python tools/sync_lecture_examples.py
```

4. Validate the repository before committing.

```bash
uv run ruff format .
uv run ruff check .
uv run ty check src tests
uv run pytest
```
