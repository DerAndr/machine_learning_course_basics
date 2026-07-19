# Student Quickstart

This repository can be used in two ways:

- as a lecture-material library with PDFs, notes, and example notebooks
- as a local notebook workspace with `uv`
- as a source for the interactive textbook preview

## 0. If you only want to read the materials

You do not need to install anything to:

- read the interactive textbook at <https://derandr.github.io/machine_learning_course_basics/>
- read `lecture_notes.md`
- open lecture PDFs
- browse the repository on GitHub

Local setup is only needed when you want to run notebooks on your machine.

The interactive textbook is the guided concept layer. The lecture folders remain the full course layer with notes, slides, examples, and practical notebooks.

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

If you are reading the textbook source locally, start with:

- `okf/index.md`
- `okf/supervised-learning/classification/index.md`
- `docs/contributing-to-textbook.md`

To rebuild the local static preview after changing textbook pages:

```bash
uv run python tools/build_textbook_preview.py
```

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

### Lecture 10: Cross-Validation and Hyperparameter Optimization

```bash
uv sync --group hpo_automl
uv run python tools/check_notebook_environment.py --group hpo_automl
```

Use this when the lecture example needs:

- `flaml`
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
- `torchinfo`

### Lecture 11: Explainability and Interpretability

```bash
uv sync --group xai_piml
```

Use this only when you want to run the optional PiML-based explainability example.
The main Lecture 11 materials, including the public practical session, run in the baseline environment after `uv sync`.
On Python 3.12, PiML may be unavailable.

### Lecture 14: ML in Production

```bash
uv sync --group ml_in_production
uv run python tools/check_notebook_environment.py --group ml_in_production
```

Use this when the practical session needs:

- `evidently`
- `mlflow`

### Lecture 16: Natural Language Processing

```bash
uv sync --group nlp
uv run python tools/check_notebook_environment.py --group nlp
```

Use this when the practical session needs:

- `transformers`
- `sentence-transformers`
- `datasets`
- `gensim`

### Lecture 17: Recommender Systems

The Part 1 foundations practical runs in the baseline environment:

```bash
uv sync
uv run python tools/check_notebook_environment.py
```

If your instructor asks you to run optional embedding or PyTorch retrieval demos from the production-pipeline material, use the relevant optional group they specify.

### Lecture 18: LLM Overview

Lecture 18 is currently a practical-first draft with full notebooks that are primarily designed for Google Colab GPU runtimes.

For local exploration of surrounding materials, use the baseline setup:

```bash
uv sync
uv run python tools/check_notebook_environment.py
```

For transformer, LoRA, and multimodal package coverage when trying selected local cells, add the LLM group:

```bash
uv sync --group llm
uv run python tools/check_notebook_environment.py --group llm
```

Large-model inference and LoRA cells may exceed local hardware limits and are best run in Colab. Please read the [Deep Learning and Colab Guide](deep-learning-colab-guide.md) for detailed setup instructions and hardware limits.

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

For full details on using Google Colab vs local execution, see the [Deep Learning and Colab Guide](deep-learning-colab-guide.md).

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
uv sync --group nlp
uv sync --group llm
uv sync --group xai_piml
uv sync --group ml_in_production
```

Textbook contribution checks:

```bash
uv run pytest
uv run python tools/validate_okf.py okf/ --strict-warnings
uv run python tools/build_textbook_preview.py
```

## 7. Codex assistance

Codex can help you choose materials, study through short questions, or generate
a focused interactive review from approved course sources.

Open Codex at the repository root so it can discover the repository skills
under `.agents/skills/`. Start with the
[student learning-companion quickstart](student-learning-companion-quickstart.md)
for copy-ready prompts, examples for this course and other repositories, and
instructions for adding the generic skill to your personal Codex.

For course navigation, invoke `$ml-course-student-navigator`, located at
`.agents/skills/ml-course-student-navigator/SKILL.md`.

**What the agent CAN do for you:**
- Help you locate specific lectures, notes, or slides based on topics you want to learn.
- Guide you through the interactive textbook (`okf/`) for conceptual questions.
- Create a short grounded review using the learning-companion skills.
- Troubleshoot local environment setup and `uv` dependencies.
- Explain the difference between `lecture_examples/` and `practical_session/` notebooks.

**What the agent WILL NOT do:**
- It will not solve the `practical_session` notebooks for you. It acts as a tutor and a guide, not a solver.
