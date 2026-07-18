# Machine Learning Course

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Content-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE-CONTENT)

Lecture notes, slide decks, and example notebooks for an introductory machine learning course.

The repository is organized lecture by lecture and is meant to work both as:

- a course library for students who want to read notes and browse examples
- a local notebook workspace for students who want to run materials with `uv`

## At a Glance

| # | Lecture | Notes | Practical |
|---|--------|-------|-----------|
| 01 | [Exploratory Data Analysis](lectures/lecture_01_eda/README.md) | [notes](lectures/lecture_01_eda/lecture_notes.md) | [practical](lectures/lecture_01_eda/practical_session/README.md) |
| 02 | [Data Preparation Part 1](lectures/lecture_02_data_preparation_part_1/README.md) | [notes](lectures/lecture_02_data_preparation_part_1/lecture_notes.md) | [practical](lectures/lecture_02_data_preparation_part_1/practical_session/README.md) |
| 03 | [Data Preparation Part 2](lectures/lecture_03_data_preparation_part_2/README.md) | [notes](lectures/lecture_03_data_preparation_part_2/lecture_notes.md) | [practical](lectures/lecture_03_data_preparation_part_2/practical_session/README.md) |
| 04 | [Regression](lectures/lecture_04_regression/README.md) | [notes](lectures/lecture_04_regression/lecture_notes.md) | [practical](lectures/lecture_04_regression/practical_session/README.md) |
| 05 | [Classification Part 1](lectures/lecture_05_classification_part_1/README.md) | [notes](lectures/lecture_05_classification_part_1/lecture_notes.md) | [practical](lectures/lecture_05_classification_part_1/practical_session/README.md) |
| 06 | [Classification Part 2](lectures/lecture_06_classification_part_2/README.md) | [notes](lectures/lecture_06_classification_part_2/lecture_notes.md) | [practical](lectures/lecture_06_classification_part_2/practical_session/README.md) |
| 07 | [Ensembles](lectures/lecture_07_ensembles/README.md) | [notes](lectures/lecture_07_ensembles/lecture_notes.md) | [practical](lectures/lecture_07_ensembles/practical_session/README.md) |
| 08 | [Time Series](lectures/lecture_08_time_series/README.md) | [notes](lectures/lecture_08_time_series/lecture_notes.md) | [practical](lectures/lecture_08_time_series/practical_session/README.md) |
| 09 | [Clustering](lectures/lecture_09_clustering/README.md) | [notes](lectures/lecture_09_clustering/lecture_notes.md) | [practical](lectures/lecture_09_clustering/practical_session/README.md) |
| 10 | [Cross-Validation and HPO](lectures/lecture_10_cross_validation_hpo/README.md) | [notes](lectures/lecture_10_cross_validation_hpo/lecture_notes.md) | [practical](lectures/lecture_10_cross_validation_hpo/practical_session/README.md) |
| 11 | [Explainability and Interpretability](lectures/lecture_11_explainability_interpretability/README.md) | [notes](lectures/lecture_11_explainability_interpretability/lecture_notes.md) | [practical](lectures/lecture_11_explainability_interpretability/practical_session/README.md) |
| 12 | [Introduction to Neural Networks](lectures/lecture_12_intro_neural_networks/README.md) | [notes](lectures/lecture_12_intro_neural_networks/lecture_notes.md) | [practical](lectures/lecture_12_intro_neural_networks/practical_session/README.md) |
| 13 | [Responsible AI](lectures/lecture_13_responsible_ai/README.md) | [notes](lectures/lecture_13_responsible_ai/lecture_notes.md) | [practical](lectures/lecture_13_responsible_ai/practical_session/README.md) |
| 14 | [ML in Production](lectures/lecture_14_ml_in_production/README.md) | [notes](lectures/lecture_14_ml_in_production/lecture_notes.md) | [practical](lectures/lecture_14_ml_in_production/practical_session/README.md) |
| 15 | [Computer Vision](lectures/lecture_15_computer_vision/README.md) | [notes](lectures/lecture_15_computer_vision/lecture_notes.md) | [practical](lectures/lecture_15_computer_vision/practical_session/README.md) |
| 16 | [Natural Language Processing](lectures/lecture_16_nlp_overview/README.md) | [notes](lectures/lecture_16_nlp_overview/lecture_notes.md) | [practical](lectures/lecture_16_nlp_overview/practical_session/README.md) |
| 17 | [Recommender Systems](lectures/lecture_17_recsys/README.md) | [notes](lectures/lecture_17_recsys/lecture_notes.md) | [practical](lectures/lecture_17_recsys/practical_session/README.md) |
| 18 | [LLM Overview](lectures/lecture_18_llm_overview/README.md) | [notes](lectures/lecture_18_llm_overview/lecture_notes.md) | [practical](lectures/lecture_18_llm_overview/practical_session/README.md) |
| 19 | [Course Overview](lectures/lecture_19_course_overview/README.md) | [notes](lectures/lecture_19_course_overview/lecture_notes.md) | - |

Lectures 01-14 are fully packaged with slide decks (`slides/lecture.pdf`) and example notebooks (`lecture_examples/`). Lectures 15-19 are practical-first drafts and the final overview.

## Interactive Textbook

The interactive textbook preview is published on GitHub Pages:

<https://derandr.github.io/machine_learning_course_basics/>

The textbook is built from the Open Knowledge Format bundle in `okf/`. The same source powers:

- readable concept pages for students;
- learning paths and browser-based labs;
- `okf-manifest.json` for agents and tools;
- generated textbook pages under `site/_build/` during preview builds.

The current pilot focuses on Classification Part 1: classification, metrics, threshold choice, one learning path, and one browser-only threshold lab.

### Interactive lecture reviews

Standalone lecture reviews complement the OKF textbook with focused,
offline-capable explanations, graphs, accessibility controls, and knowledge
checks.

- [Live EDA interactive review](https://derandr.github.io/machine_learning_course_basics/demos/lecture_01_eda/)
- [Offline EDA review](lecture_experiences/lecture_01_eda/index.html)
- [Learning-assistant guide](docs/interactive-lecture-learning-assistant.md)

## Assignments & Exams

- **[Mini-Project: NYC Airbnb Price Prediction](mini_projects/airbnb_nyc/README.md)**
- **[Midterm Exam: Example Questions](docs/midterm_examples.md)**
## Start Here

If you are a student:

1. Open lectures/README.md
2. Choose a lecture directory
3. Read lecture_notes.md
4. Open slides/lecture.pdf when the lecture has a packaged slide deck
5. Run notebooks from lecture_examples/ if needed

For local setup, start with:

`ash
uv sync
uv run python tools/check_notebook_environment.py
uv run jupyter lab
`

Student-oriented setup instructions live in docs/student-quickstart.md. For heavy deep learning workloads, read docs/deep-learning-colab-guide.md.

To improve the interactive textbook, start with docs/contributing-to-textbook.md.

## Repository Contents

- `lectures/README.md` for top-level lecture navigation
- `lectures/<lecture_slug>/README.md` for lecture-specific navigation
- `lectures/<lecture_slug>/lecture_notes.md` for student revision and recap
- `lectures/<lecture_slug>/links.yaml` for compact lecture metadata
- `lectures/<lecture_slug>/slides/lecture.pdf` for the canonical lecture deck
- `lectures/<lecture_slug>/lecture_examples/` for example notebooks and paired scripts
- `lectures/<lecture_slug>/practical_session/` for separate classroom practicals
- `lectures/<lecture_slug>/plan/` for draft planning notes when a practical-first lecture is still being packaged
- `okf/` for concise textbook concepts, learning paths, labs, and metadata
- `docs/okf-authoring-guide.md` for OKF authoring rules
- `docs/contributing-to-textbook.md` for student, maintainer, and agent contribution workflow
- `docs/student-quickstart.md` for local setup instructions
- `docs/deep-learning-colab-guide.md` for running heavy deep learning models locally or in Colab
- `site/assets/` and `site/data/` for committed browser-lab assets and public-safe data
- `site/_build/` for generated textbook preview output; this directory is not committed

## Current Scope

- The course covers 14 fully packaged lectures, plus practical-first drafts for Lecture 15: Computer Vision, Lecture 16: Natural Language Processing, Lecture 17: Recommender Systems, and Lecture 18: LLM Overview.
- Every packaged lecture and practical-first draft includes a practical_session/ with student-facing materials.
- Raw imports, migration history, and source collection remain outside the public course layer.
- The interactive textbook layer is a public-safe pilot built from `okf/` and published through GitHub Pages.

## Directory Layout

```text
.
├── okf/
│   ├── index.md
│   ├── supervised-learning/
│   ├── learning-paths/
│   ├── labs/
│   └── contributing/
├── lectures/
│   ├── README.md
│   ├── lecture_01_eda/
│   │   ├── README.md
│   │   ├── lecture_notes.md
│   │   ├── links.yaml
│   │   ├── slides/
│   │   │   └── lecture.pdf
│   │   ├── lecture_examples/
│   │   │   ├── README.md
│   │   │   ├── example_01.ipynb
│   │   │   ├── example_01.py
│   │   │   └── ...
│   │   └── practical_session/
│   │       ├── README.md
│   │       ├── <name>_practical_student_90min.ipynb
│   │       ├── <name>_practical_student_90min.py
│       └── ...
│   └── ...
├── docs/
│   ├── publishing-model.md
│   └── repository-overview.md
├── tools/
│   └── convert_quiz_dumps.py
├── site/
│   ├── assets/
│   ├── data/
│   └── _build/
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
  - `uv sync --group hpo_automl` for Lecture 10 extras such as `h2o`, `optuna`, `hyperopt`, and `scikit-optimize`
  - `uv sync --group neural_networks` for Lecture 12 extras such as `torch` and `torchinfo`
  - `uv sync --group nlp` for Lecture 16 extras such as `transformers`, `sentence-transformers`, `datasets`, and `gensim`
  - `uv sync --group llm` for Lecture 18 extras such as `accelerate`, `peft`, `outlines`, and current `transformers`
  - `uv sync --group xai_piml` for the optional PiML example in Lecture 11 on compatible Python versions
  - `uv sync --group ml_in_production` for Lecture 14 extras such as `evidently` and `mlflow`

## Working With Lecture Examples

- Each lecture example notebook in `lecture_examples/` has a paired `.py` file.
- The `.py` files are generated companions for local reading, diffing, and lightweight execution.
- If a notebook contains Colab-only install cells such as `!pip install ...`, those cells can usually be skipped locally after `uv sync`.
- When a lecture uses optional heavy dependencies, install the matching group first and then re-run the environment check:
  - `uv run python tools/check_notebook_environment.py --group ensembles`
  - `uv run python tools/check_notebook_environment.py --group time_series`
  - `uv run python tools/check_notebook_environment.py --group hpo_automl`
  - `uv run python tools/check_notebook_environment.py --group neural_networks`
  - `uv run python tools/check_notebook_environment.py --group nlp`
  - `uv run python tools/check_notebook_environment.py --group llm`
  - `uv run python tools/check_notebook_environment.py --group ml_in_production`
- To regenerate the `.py` companions and example READMEs after changing notebooks, run:
  - `uv run python tools/sync_lecture_examples.py`

## How to Read the Repository

For humans:

1. README.md
2. [Interactive textbook](https://derandr.github.io/machine_learning_course_basics/)
3. lectures/README.md
4. lectures/<lecture_slug>/README.md
5. lectures/<lecture_slug>/lecture_notes.md
6. lectures/<lecture_slug>/links.yaml
7. lectures/<lecture_slug>/slides/lecture.pdf
8. lectures/<lecture_slug>/lecture_examples/README.md
9. lectures/<lecture_slug>/lecture_examples/example_XX.ipynb
10. lectures/<lecture_slug>/lecture_examples/example_XX.py
11. lectures/<lecture_slug>/practical_session/README.md

For agents:

1. AGENTS.md
2. .agents/skills/ml-course-student-navigator/SKILL.md for helping students navigate materials and environment setup
3. .agents/skills/ml-course-textbook-contributor/SKILL.md for textbook work
4. docs/contributing-to-textbook.md
5. docs/okf-authoring-guide.md
6. okf/index.md
7. lectures/README.md
8. lectures/index.yaml
9. lectures/<lecture_slug>/README.md
10. lectures/<lecture_slug>/lecture_notes.md
11. lectures/<lecture_slug>/links.yaml
12. lectures/<lecture_slug>/lecture_examples/README.md
13. lectures/<lecture_slug>/practical_session/README.md

## License

Unless stated otherwise:

- Source code in src/, 	ests/, 	ools/, and standalone .py files is licensed under the [MIT License](LICENSE).
- Lecture notes, slide decks, PDFs, images, notebooks, and other course content are licensed under [CC BY-NC-SA 4.0](LICENSE-CONTENT).
