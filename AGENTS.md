# Agent Guide

This repository is an introductory machine learning course and the public student source for the interactive textbook.
It contains lecture notes, slide decks, example notebooks, classroom practicals, OKF knowledge modules, and browser-based textbook assets covering 18 lecture blocks from exploratory data analysis to LLM overview.

Use this file to understand what the repository contains and where to find things.

## Course Topics

| # | Lecture | Directory |
|---|--------|-----------|
| 01 | Exploratory Data Analysis | `lecture_01_eda` |
| 02 | Data Preparation Part 1 | `lecture_02_data_preparation_part_1` |
| 03 | Data Preparation Part 2 | `lecture_03_data_preparation_part_2` |
| 04 | Regression | `lecture_04_regression` |
| 05 | Classification Part 1 | `lecture_05_classification_part_1` |
| 06 | Classification Part 2 | `lecture_06_classification_part_2` |
| 07 | Ensembles | `lecture_07_ensembles` |
| 08 | Time Series | `lecture_08_time_series` |
| 09 | Clustering | `lecture_09_clustering` |
| 10 | Cross-Validation and HPO | `lecture_10_cross_validation_hpo` |
| 11 | Explainability and Interpretability | `lecture_11_explainability_interpretability` |
| 12 | Introduction to Neural Networks | `lecture_12_intro_neural_networks` |
| 13 | Responsible AI | `lecture_13_responsible_ai` |
| 14 | ML in Production | `lecture_14_ml_in_production` |
| 15 | Computer Vision | `lecture_15_computer_vision` |
| 16 | Natural Language Processing | `lecture_16_nlp_overview` |
| 17 | Recommender Systems | `lecture_17_recsys` |
| 18 | LLM Overview | `lecture_18_llm_overview` |

## Where Things Live

All course materials are in `lectures/`. Fully packaged lecture directories have:

- `README.md` — lecture overview with links to all files
- `lecture_notes.md` — student-facing recap and revision notes
- `links.yaml` — compact lecture metadata
- `slides/lecture.pdf` — canonical lecture deck
- `lecture_examples/` — example notebooks with paired `.py` scripts
- `practical_session/` — 90-minute classroom practical with TODO cells for students

Lectures `15` to `18` are practical-first drafts. They have lecture READMEs, notes/metadata, and practical sessions, but do not yet have packaged slide decks or separate `lecture_examples/`.
Lecture 18 currently publishes full practical notebooks rather than a separate student TODO version.

Use `lectures/index.yaml` as the machine-readable course index.

## Interactive Textbook Layer

The interactive textbook is published at:

<https://derandr.github.io/machine_learning_course_basics/>

The source of truth is `okf/`. Keep durable pedagogy, concept metadata, prerequisites, learning objectives, lab descriptions, and source links there. The renderer, JavaScript, CSS, and public-safe lab data live outside `okf/`:

- `tools/build_textbook_preview.py` — static preview renderer
- `site/assets/` — committed CSS and browser-lab JavaScript
- `site/data/` — committed public-safe data for browser labs
- `site/_build/` — generated output, not committed

Standalone interactive lecture reviews are sourced separately from OKF:

- `lecture_experiences/content/` — grounded JSON payloads
- `lecture_experiences/<lecture_slug>/index.html` — portable generated reviews
- `site/_build/demos/<lecture_slug>/index.html` — Pages copies created at build time

The committed standalone HTML is the source for both offline use and Pages.
The builder copies it byte-for-byte; do not maintain a second demo under
`site/`.

For agent-facing discovery, the generated `okf-manifest.json` must preserve the same descriptions and learning objectives as the OKF source. Do not maintain a separate hand-written skills list; agent-facing `skills` are generated from `learning_objectives`.

## How to Navigate

To explore the full course:

1. `README.md` — repository overview
2. `lectures/README.md` — lecture map with links
3. `lectures/index.yaml` — machine-readable index (all paths, all lectures)

To improve the interactive textbook:

1. `.agents/skills/ml-course-textbook-contributor/SKILL.md` — agent workflow
2. `docs/contributing-to-textbook.md` — student and maintainer workflow
3. `docs/okf-authoring-guide.md` — OKF authoring rules
4. `okf/` — source knowledge bundle
5. `tools/build_textbook_preview.py` — renderer for the current preview
6. `tests/test_textbook_preview.py` and `tests/test_okf_validation.py` — regression checks

When improving textbook content, keep the student page, OKF metadata, index card description, rendered page, and manifest in sync.

To create or revise an interactive lecture review:

1. `.agents/skills/interactive-learning-experience-builder/SKILL.md` — portable content contract, deterministic generator, and offline validator
2. `.agents/skills/ml-course-interactive-learning-assistant/SKILL.md` — ML-course source, safety, output, and publishing adapter
3. `lecture_experiences/content/<lecture_slug>.json` — grounded content payload
4. `lecture_experiences/<lecture_slug>/index.html` — generated offline review
5. `docs/interactive-lecture-learning-assistant.md` — learner and maintainer guide
6. `.agents/skills/interactive-learning-experience-builder/scripts/generate_learning_experience.py` — deterministic generator
7. `.agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py` — offline/accessibility validator

Create a context profile before authoring: learner and goal, named sources,
excluded/private material, requested accessibility defaults, output path, and
available validation or publishing commands. Use the portable core directly
for one-off work; add an adapter only for recurring local constraints. The
repository core is canonical; synchronize any global installation from it and
compare the skill, template, scripts, and references to detect drift.

Lecture 01 provides the reference structure under
`lecture_experiences/content/lecture_01_eda.json` and
`lecture_experiences/lecture_01_eda/`.

To explore a specific lecture:

1. `lectures/<slug>/README.md` — what's in this lecture
2. `lectures/<slug>/lecture_notes.md` — summary and key concepts
3. `lectures/<slug>/links.yaml` — metadata
4. `lectures/<slug>/slides/lecture.pdf` — slide deck
5. `lectures/<slug>/lecture_examples/README.md` — list of example notebooks
6. `lectures/<slug>/lecture_examples/example_XX.ipynb` — example notebook
7. `lectures/<slug>/practical_session/README.md` — practical overview
8. `lectures/<slug>/practical_session/*_student_90min.ipynb` — student practical notebook

## Practical Sessions

Packaged lectures and most practical-first drafts have a `practical_session/` containing:

- A **student notebook** (`*_practical_student_90min.ipynb`) with TODO cells where students write code.
- A paired **Python script** (`*_practical_student_90min.py`) for lightweight reading and diffing.
- A **README.md** with teaching intent, scope, and environment notes.

The practicals are designed for 90 minutes and use real datasets from OpenML, library datasets, public downloads, or small generated examples.
Lecture 18 is currently an exception: it is a dual full-notebook draft without a separate student TODO notebook.

## Lecture Examples vs Practical Sessions

These are separate layers — do not mix them:

- `lecture_examples/` — demos shown during the lecture to illustrate concepts
- `practical_session/` — hands-on exercises students complete during class

## Notebook Files

- `.ipynb` is the source notebook
- `.py` is an auto-generated companion script for reading and diffing
- After editing notebooks in `lecture_examples/`, regenerate scripts: `uv run python tools/sync_lecture_examples.py`

## Local Setup

Use `uv` as the environment manager:

```bash
uv sync
uv run python tools/check_notebook_environment.py
uv run jupyter lab
```

Some lectures need extra dependencies:

| Group | Lecture | Key packages |
|-------|---------|-------------|
| `ensembles` | 07 | catboost, lightgbm, xgboost |
| `time_series` | 08 | prophet |
| `hpo_automl` | 10 | h2o, optuna, hyperopt, scikit-optimize |
| `neural_networks` | 12 | torch, torchinfo |
| `nlp` | 16 | transformers, sentence-transformers, datasets, gensim |
| `llm` | 18 | transformers, accelerate, peft, bitsandbytes, torchao |
| `xai_piml` | 11 | piml (compatible Python versions only) |
| `ml_in_production` | 14 | evidently, mlflow |

Install with: `uv sync --group <group_name>`

Detailed setup: `docs/student-quickstart.md`

## Supporting Directories

- `docs/` — student-facing setup guides and workflow documentation
- `docs/deep-learning-colab-guide.md` — instructions for running heavy deep learning models locally or on Colab
- `okf/` — concise textbook concepts, learning paths, lab descriptions, and contribution modules
- `.agents/skills/ml-course-textbook-contributor/` — agent workflow for safe textbook contributions
- `.agents/skills/ml-course-student-navigator/` — agent workflow for helping students navigate the course and set up their environment
- `.agents/skills/ml-course-interactive-learning-assistant/` — agent workflow, template, generator, and validator for standalone lecture reviews
- `lecture_experiences/content/` — public-source-grounded lecture review payloads
- `lecture_experiences/lecture_01_eda/` — reference offline EDA review; other lectures follow the same structure
- `site/assets/` and `site/data/` — committed assets and public-safe data used by the textbook preview
- `src/mlcourse/` — shared Python helpers (paths, data utilities)
- `tools/` — repository maintenance scripts, notebook sync, environment checks, OKF validation, and textbook rendering
- `publish/lectures.yaml` — assignment and publication metadata

## Textbook Change Checks

Before committing textbook or OKF changes, run:

```bash
uv run ruff check src/mlcourse/okf_validation.py tools/validate_okf.py tools/build_textbook_preview.py tests/test_okf_validation.py tests/test_textbook_preview.py tests/test_smoke.py
uv run pytest
uv run python tools/validate_okf.py okf/ --strict-warnings
uv run python tools/build_textbook_preview.py
```

For a standalone lecture review, also run:

```bash
uv run python .agents/skills/interactive-learning-experience-builder/scripts/validate_learning_experience.py lecture_experiences/lecture_01_eda/index.html
```

After merging to `main`, verify GitHub Actions and the deployed GitHub Pages URL.

