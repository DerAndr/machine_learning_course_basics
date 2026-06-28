---
name: ml-course-student-navigator
description: Helps a student navigate the ML course materials. Use when a student asks to find a lecture, set up their local environment, find example notebooks, or understand the interactive textbook. Guides the student to the correct commands (like `uv sync`) and paths without giving away practical answers.
---

# ML Course Student Navigator

Use this skill when helping a student navigate the Machine Learning Course repository. The student might need help finding lecture notes, slides, examples, practical assignments, or setting up their Python environment to run code.

## Core Directives

- **Be a Guide, Not a Solver:** Your primary role is to help the student find materials and set up their environment. Do not give them direct answers to their `practical_session/` notebooks unless they explicitly ask for conceptual hints.
- **Reference Official Docs:** Point students to `docs/student-quickstart.md` for environment troubleshooting.
- **Explain the Dual Structure:** Help students understand that `okf/` powers the interactive textbook (for concise concept summaries) while `lectures/` contains the full course material (for deep dives and examples).

## Finding Course Materials

When a student is looking for materials for a specific topic, navigate the `lectures/` directory:

1. **Top-Level Navigation:** Reference `lectures/README.md` or `lectures/index.yaml` to find the correct lecture slug (e.g., `lecture_05_classification_part_1`).
2. **Inside a Lecture:**
   - Notes: `lecture_notes.md` (Summary and recap)
   - Slides: `slides/lecture.pdf` (Canonical deck)
   - Examples: `lecture_examples/` (Demo notebooks and `.py` versions)
   - Assignments: `practical_session/` (Hands-on work for the student)

When a student wants a quick conceptual summary or a browser lab, point them to the interactive textbook structure in `okf/` or the live site at `https://derandr.github.io/machine_learning_course_basics/`.

## Environment Setup Guidance

The course uses `uv` for dependency management.

- **Baseline Setup:** Tell the student to run `uv sync` followed by `uv run jupyter lab`. This is sufficient for most lectures (e.g., lectures 01-06, 09).
- **Lecture-Specific Setup:** If a student is starting a heavy lecture, check `docs/student-quickstart.md` and tell them to sync the specific group. For example:
  - Lecture 07 (Ensembles): `uv sync --group ensembles`
  - Lecture 08 (Time Series): `uv sync --group time_series`
  - Lecture 12 (Neural Networks): `uv sync --group neural_networks`
  - Lecture 16 (NLP): `uv sync --group nlp`
  - Lecture 18 (LLMs): `uv sync --group llm`

## Common Student Scenarios

- **"I have an import error for XGBoost/Torch/Transformers"**
  Check the lecture they are on and remind them to run the corresponding `uv sync --group <name>` command.
- **"Where are the answers to the practical?"**
  Gently remind them that practicals (`*_student_90min.ipynb`) are meant for them to solve. Offer to explain the concept they are stuck on using the materials from `okf/` or `lecture_examples/`.
- **"I found a `!pip install` cell in a notebook"**
  Tell them to skip it if they are running locally, and ensure they have synced the correct `uv` group instead.
- **"How do I run heavy deep learning notebooks or fix OOM errors?"**
  Point the student to `docs/deep-learning-colab-guide.md` for guidance on using Google Colab vs. local execution and how to enable GPUs.
