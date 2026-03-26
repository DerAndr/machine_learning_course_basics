# Notebook Workflow

## Source of Truth

Tracked notebooks use Jupytext pairs:

- `.ipynb` for execution and distribution
- `.py` in percent format for readable diffs and review

Both files must stay in sync.

## Authoring Rules

- Keep reusable logic in `src/mlcourse/`.
- Keep notebooks focused on explanations, experiments, and orchestration.
- Use English for notebook text, comments, and metadata.
- Do not store datasets in git.

## Recommended Workflow

1. Create or edit a notebook in JupyterLab.
2. Pair it with a percent-format Python file via Jupytext.
3. Move repeated code into `src/mlcourse/` when it becomes reusable.
4. Run formatting, linting, and type checks before committing.
