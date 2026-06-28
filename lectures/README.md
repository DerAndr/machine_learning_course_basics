# Lectures

This directory contains the canonical lecture layer for the course.

The interactive textbook lives beside this lecture layer, not inside it. Use `okf/` for concise textbook concepts, learning paths, and lab descriptions; use `lectures/` as the full course source and provenance layer.

## Per-Lecture Structure

Fully packaged lecture directories contain:

- `README.md`
- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `lecture_examples/example_XX.ipynb`
- `lecture_examples/example_XX.py`
- `practical_session/README.md`
- `practical_session/<slug>_practical_student_90min.ipynb`
- `practical_session/<slug>_practical_student_90min.py`

The public release of each practical session contains only the student-facing notebook and README; instructor materials stay unpublished.
Lectures 15-18 are currently practical-first drafts, so they do not yet include slide decks or separate lecture example notebooks.
Lecture 17 also includes draft planning notes under `plan/` for the two recommender-systems parts.
Lecture 18 is currently a dual teacher-notebook draft without a separate student TODO notebook.

## Relationship to the Interactive Textbook

Textbook pages should link back to lecture material for provenance and deeper study. Lecture notes, examples, and practical sessions stay canonical here; textbook pages in `okf/` should summarize and connect concepts rather than duplicate full lectures.

When a lecture explanation changes in a way that affects a textbook concept, review the related OKF page, index entry, and rendered preview.

## Lecture Map

1. [Lecture 01: Exploratory Data Analysis](lecture_01_eda/README.md)
2. [Lecture 02: Data Preparation Part 1](lecture_02_data_preparation_part_1/README.md)
3. [Lecture 03: Data Preparation Part 2](lecture_03_data_preparation_part_2/README.md)
4. [Lecture 04: Regression](lecture_04_regression/README.md)
5. [Lecture 05: Classification Part 1](lecture_05_classification_part_1/README.md)
6. [Lecture 06: Classification Part 2](lecture_06_classification_part_2/README.md)
7. [Lecture 07: Ensembles](lecture_07_ensembles/README.md)
8. [Lecture 08: Time Series](lecture_08_time_series/README.md)
9. [Lecture 09: Clustering](lecture_09_clustering/README.md)
10. [Lecture 10: Cross-Validation and Hyperparameter Optimization](lecture_10_cross_validation_hpo/README.md)
11. [Lecture 11: Explainability and Interpretability](lecture_11_explainability_interpretability/README.md)
12. [Lecture 12: Introduction to Neural Networks](lecture_12_intro_neural_networks/README.md)
13. [Lecture 13: Responsible AI](lecture_13_responsible_ai/README.md)
14. [Lecture 14: ML in Production](lecture_14_ml_in_production/README.md)
15. [Lecture 15: Computer Vision](lecture_15_computer_vision/README.md)
16. [Lecture 16: Natural Language Processing](lecture_16_nlp_overview/README.md)
17. [Lecture 17: Recommender Systems](lecture_17_recsys/README.md)
18. [Lecture 18: LLM Overview](lecture_18_llm_overview/README.md)
19. [Lecture 19: Course Overview](lecture_19_course_overview/README.md)
