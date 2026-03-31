# Machine Learning Course

Lecture notes, slide decks, and example notebooks for an introductory machine learning course.

The repository is organized lecture by lecture and is meant to work both as:

- a course library for students who want to read notes and browse examples
- a local notebook workspace for students who want to run materials with `uv`

## At a Glance

- 14 lecture topics across core machine learning subjects
- lecture notes for revision and recap
- canonical lecture PDFs
- example notebooks for each lecture
- optional local setup with lecture-specific dependency groups

## Start Here

If you are a student:

1. Open `lectures/README.md`
2. Choose a lecture directory
3. Read `lecture_notes.md`
4. Open `slides/lecture.pdf`
5. Run notebooks from `lecture_examples/` if needed

For local setup, start with:

```bash
uv sync
uv run python tools/check_notebook_environment.py
uv run jupyter lab
```

Student-oriented setup instructions live in `docs/student-quickstart.md`.

## Included in This Commit Layer

- `AGENTS.md`
- `lectures/README.md`
- `lectures/lecture_xx_slug/README.md`
- `lectures/lecture_xx_slug/lecture_notes.md`
- `lectures/lecture_xx_slug/links.yaml`
- `lectures/lecture_xx_slug/slides/lecture.pdf`
- `lectures/lecture_xx_slug/lecture_examples/README.md`

## What Is Not Included Yet

- classroom practical sessions, except where explicitly added later
- reusable Python helpers
- publishing manifests
- raw imports and migration history
- extra source formats such as `pptx`

## Directory Layout

```text
lectures/
├── README.md
├── lecture_01_eda/
│   ├── README.md
│   ├── lecture_notes.md
│   ├── links.yaml
│   ├── slides/
│   │   └── lecture.pdf
│   └── lecture_examples/
│       ├── README.md
│       ├── example_01.ipynb
│       ├── example_01.py
│       └── ...
└── ...
```

## Local Setup

Use `uv` as the single environment manager for this repository.

1. `uv sync`
2. `uv run python tools/check_notebook_environment.py`
3. `uv run jupyter lab`

There is no separate `requirements.txt` on purpose.

- `pyproject.toml` is the source of truth for dependencies.
- `uv.lock` is the reproducible lockfile.
- The default environment installs the shared baseline used across the course.
- Heavy or lecture-specific packages can be added only when needed:
  - `uv sync --group ensembles` for Lecture 07 extras such as `catboost`, `lightgbm`, and `xgboost`
  - `uv sync --group time_series` for Lecture 08 extras such as `prophet`
  - `uv sync --group hpo_automl` for Lecture 11 extras such as `h2o`, `optuna`, `hyperopt`, and `scikit-optimize`
  - `uv sync --group neural_networks` for Lecture 12 extras such as `torch`
  - `uv sync --group xai_piml` for the optional PiML example in Lecture 10 on compatible Python versions
## Working With Lecture Examples

- Each lecture example notebook in `lecture_examples/` has a paired `.py` file.
- The `.py` files are generated companions for local reading, diffing, and lightweight execution.
- If a notebook contains Colab-only install cells such as `!pip install ...`, those cells can usually be skipped locally after `uv sync`.
- When a lecture uses optional heavy dependencies, install the matching group first and then re-run the environment check:
  - `uv run python tools/check_notebook_environment.py --group ensembles`
  - `uv run python tools/check_notebook_environment.py --group time_series`
  - `uv run python tools/check_notebook_environment.py --group hpo_automl`
  - `uv run python tools/check_notebook_environment.py --group neural_networks`
- To regenerate the `.py` companions and example READMEs after changing notebooks, run:
  - `uv run python tools/sync_lecture_examples.py`

## How to Read the Repository

For humans:

1. `README.md`
2. `lectures/README.md`
3. `lectures/<lecture_slug>/README.md`
4. `lectures/<lecture_slug>/lecture_notes.md`
5. `lectures/<lecture_slug>/links.yaml`
6. `lectures/<lecture_slug>/slides/lecture.pdf`
7. `lectures/<lecture_slug>/lecture_examples/README.md`
8. `lectures/<lecture_slug>/lecture_examples/example_XX.ipynb`
9. `lectures/<lecture_slug>/lecture_examples/example_XX.py`

For agents:

1. `AGENTS.md`
2. `lectures/README.md`
3. `lectures/<lecture_slug>/README.md`
4. `lectures/<lecture_slug>/lecture_notes.md`
5. `lectures/<lecture_slug>/links.yaml`
6. `lectures/<lecture_slug>/lecture_examples/README.md`
