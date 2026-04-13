# Exploratory Data Analysis Practical Session

This directory contains a 90-minute classroom practical for Lecture 01.

## Files

- `eda_practical_student_90min.ipynb`
- `eda_practical_student_90min.py`
- `README.md`

## Format

- The student notebook contains targeted TODO placeholders in the main interpretation cells.
- The student notebook also has a generated Python companion script for easier diffing and review.
- The practical uses the Palmer Penguins dataset from OpenML `42585`.
- The original dataset reference is `allisonhorst.github.io/palmerpenguins`, and the practical maps OpenML column aliases back to the canonical Palmer Penguins names.
- The session stays focused on manual EDA before optional automation.

## Teaching Intent

- Start with dataset structure before plotting.
- Keep the visual layer simple and readable for a first EDA lecture.
- Show how categorical summaries, numerical summaries, and pairwise plots answer different questions.
- Reinforce that EDA is interpretation, not just chart production.

## Environment

The baseline repository environment is enough for this practical.

If you run in Google Colab, install only if needed:

- `openml`
- `pandas`
- `seaborn`
- `matplotlib`

Optional automated-EDA step:

- `ydata-profiling`
