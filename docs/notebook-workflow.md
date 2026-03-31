# Notebook Workflow

## Source of Truth

Tracked lecture examples use:

- `.ipynb` as the source notebook for execution
- `.py` as a generated companion script for readable diffs and local inspection

The `.py` files are regenerated from notebooks with a repository tool.

## Authoring Rules

- Keep reusable logic in `src/mlcourse/`.
- Keep notebooks focused on explanations, experiments, and orchestration.
- Use English for notebook text, comments, and metadata.
- Do not store datasets in git.

## Canonical Lecture Example Layout

Each lecture example directory follows this file pattern:

- `lecture_examples/README.md`
- `lecture_examples/example_01.ipynb`
- `lecture_examples/example_01.py`
- additional `example_XX.ipynb` and `example_XX.py` files as needed

## Recommended Workflow

1. Create or edit a notebook in JupyterLab.
2. If the notebook depends on a lecture-specific heavy package, install the matching optional `uv` group first.
3. Regenerate the companion `.py` script after notebook edits.
4. Move repeated code into `src/mlcourse/` when it becomes reusable.
5. Run formatting, linting, and type checks before committing.

## Sync Command

```bash
uv run python tools/sync_lecture_examples.py
```

This command refreshes:

- `lecture_examples/example_XX.py`
- `lecture_examples/README.md`
- lecture-level `README.md` example listings

Optional lecture groups currently used in the repository:

- `uv sync --group ensembles` for Lecture 07 extras such as `catboost`, `lightgbm`, and `xgboost`
- `uv sync --group time_series` for Lecture 08 extras such as `prophet`
- `uv sync --group hpo_automl` for Lecture 11 extras such as `h2o`, `optuna`, `hyperopt`, and `scikit-optimize`
- `uv sync --group neural_networks` for Lecture 12 extras such as `torch`
