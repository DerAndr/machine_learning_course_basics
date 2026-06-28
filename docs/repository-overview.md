# Repository Overview

This repository stores the public student version of the machine learning course and the source for the interactive textbook preview.

## Responsibilities

- Publish student-facing lecture notes, slides, examples, and practical notebooks.
- Keep reusable code outside notebooks.
- Maintain the OKF knowledge bundle used by the interactive textbook.
- Keep agent-readable metadata synchronized with student-facing explanations.
- Keep generated preview output out of Git.

## Main Areas

- `lectures/` contains one directory per lecture.
- `okf/` contains concise textbook concepts, learning paths, labs, and contribution modules.
- `docs/` contains setup, repository, OKF, and textbook contribution guides.
- `.codex/skills/ml-course-textbook-contributor/` contains the agent workflow for textbook improvements.
- `site/assets/` and `site/data/` contain committed browser-lab assets and public-safe data.
- `site/_build/` contains generated textbook preview output and is not committed.
- `src/mlcourse/` contains shared helper code.
- `tools/` contains maintenance, validation, and preview-build scripts.
- `publish/lectures.yaml` tracks assignment and publication state.

## Interactive Textbook

The current public preview is:

<https://derandr.github.io/machine_learning_course_basics/>

The textbook is generated from `okf/`. A concept should remain useful as Markdown in GitHub and as a rendered HTML page. The generated `okf-manifest.json` is also an agent-facing map of the course knowledge.

When changing textbook content, update related indexes and guides at the same time, then run:

```bash
uv run pytest
uv run python tools/validate_okf.py okf/ --strict-warnings
uv run python tools/build_textbook_preview.py
```

## Current State

Lectures `01` to `14` are in place with lecture notes, slide decks, example notebooks, and practical sessions.
Lectures `15` to `18` are practical-first drafts covering Computer Vision, Natural Language Processing, Recommender Systems, and LLM Overview.
The first interactive textbook pilot covers Classification Part 1 and includes concept pages, a learning path, a browser-based threshold lab, and an agent manifest.
Lecture-specific dependency groups are available for heavier libraries (ensembles, time series, HPO/AutoML, neural networks, NLP, LLM, XAI/PiML, ML in production).
