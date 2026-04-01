# Agent Guide

This repository is a public machine learning course repository.

Its primary purpose is to distribute lecture materials in a clean, stable structure for students and for agent-based navigation.

## Canonical Course Content

The canonical course materials live in `lectures/`.

For each lecture, the main files are:

- `README.md`
- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `lecture_examples/example_XX.ipynb`
- `lecture_examples/example_XX.py`

Use `lectures/index.yaml` as the machine-readable course index.

## Canonical Traversal Order

For general repository navigation:

1. `README.md`
2. `lectures/README.md`
3. `lectures/index.yaml`

For a specific lecture:

1. `lectures/<lecture_slug>/README.md`
2. `lectures/<lecture_slug>/lecture_notes.md`
3. `lectures/<lecture_slug>/links.yaml`
4. `lectures/<lecture_slug>/slides/lecture.pdf`
5. `lectures/<lecture_slug>/lecture_examples/README.md`
6. `lectures/<lecture_slug>/lecture_examples/example_XX.ipynb`
7. `lectures/<lecture_slug>/lecture_examples/example_XX.py`

## Lecture Layers

Within a lecture directory:

- `lecture_notes.md`: student-facing recap and revision notes
- `slides/lecture.pdf`: canonical lecture deck
- `lecture_examples/`: notebooks that accompany the lecture
- `practical_session/`: separate classroom practical materials when present

Important distinction:

- `lecture_examples/` are lecture companion materials
- `practical_session/` is a separate practice layer and should not be merged conceptually into lecture examples

Do not assume that every lecture has `practical_session/`.

## Notebook Contract

- `.ipynb` is the source notebook
- `.py` is a generated companion script
- generated lecture-example scripts should stay in sync with the notebooks

After editing notebooks in `lecture_examples/`, run:

```bash
uv run python tools/sync_lecture_examples.py
```

## Local Tooling

Use `uv` as the default environment manager.

Baseline environment:

```bash
uv sync
uv run python tools/check_notebook_environment.py
```

Optional lecture-specific dependency groups:

- `ensembles`
- `time_series`
- `hpo_automl`
- `neural_networks`
- `xai_piml`

Examples:

```bash
uv sync --group ensembles
uv sync --group time_series
uv sync --group hpo_automl
uv sync --group neural_networks
```

## Non-Canonical Sources

Do not treat these as canonical course content unless explicitly requested:

- `incoming_materials/`
- `legacy_import/`

These directories exist for source collection, migration history, and provenance.

## Editing Guidance

When refining the repository:

- prefer English-only repository text
- keep lecture slugs stable
- avoid re-introducing raw duplication into `lectures/`
- treat `lectures/` as the public course layer
- keep setup instructions aligned with `pyproject.toml` and `uv.lock`
- do not move lecture examples back into ad hoc notebook folders

## What To Ignore By Default

Unless the task explicitly requires them, do not depend on:

- `incoming_materials/`
- `legacy_import/`
- unpublished cleanup notes
- temporary migration leftovers

