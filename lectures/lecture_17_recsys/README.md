# Lecture 17: Recommender Systems

This directory currently contains a two-part practical-first draft for recommender systems.

The material focuses on the concepts students need to understand recommendation pipelines:

- users, items, interactions, explicit feedback, and implicit feedback;
- why missing interactions are not automatically negative examples;
- popularity baselines and Bayesian shrinkage;
- content-based recommendation with item metadata;
- item-item collaborative filtering from user-item behavior;
- matrix factorization intuition with a small by-hand calculation;
- leave-last-out evaluation and top-K metrics;
- candidate generation, ranking, reranking, serving, logging, and evaluation;
- two-tower retrieval as a bridge to production-style RecSys;
- latency, storage, recomputation, coverage, diversity, and online metrics.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `plan/recsys_part_1_foundations.md`
- `plan/recsys_part_2_production_pipeline.md`

## Practical Session

- `practical_session/`: classroom practical materials for MovieLens-based recommender systems.
- `practical_session/recsys_practical_student_90min.ipynb`: student notebook for Part 1 foundations.
- `practical_session/recsys_practical_student_90min.py`: generated companion script for review and diffing.

Part 1 is student-facing and CPU-friendly. The Part 2 plan extends the story into a production-like pipeline with candidate generation, ranking, reranking, and stage-specific metrics.

## Draft Status

This lecture does not yet have a slide deck or separate lecture example notebooks in the repository.
The practical notebook and planning notes are the canonical working artifacts for now.

---

[<- Previous](../lecture_16_nlp_overview/README.md) | [All Lectures](../README.md)
