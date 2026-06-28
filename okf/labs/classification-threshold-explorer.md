---
type: Interactive Lab
title: Classification Threshold Explorer
description: Move a threshold and inspect how binary classification metrics change.
tags: [classification, interactive, supervised-learning]
timestamp: 2026-06-28T00:00:00Z
status: draft
learning_objectives:
  - Connect threshold movement to confusion-matrix counts and derived metrics.
difficulty: introductory
estimated_reading_minutes: 5
prerequisites:
  - /supervised-learning/classification/classification-threshold.md
related_concepts:
  - /supervised-learning/classification/classification.md
  - /supervised-learning/classification/classification-metrics.md
  - /supervised-learning/classification/classification-threshold.md
source_materials:
  - /lectures/lecture_05_classification_part_1/README.md
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
  - /lectures/lecture_05_classification_part_1/slides/lecture.pdf
  - /lectures/lecture_05_classification_part_1/practical_session/README.md
  - /lectures/lecture_05_classification_part_1/practical_session/classification_part1_practical_student_90min.ipynb
---

# Classification Threshold Explorer

## Lab goal

Move a binary classification threshold and observe how the same model scores produce different predicted labels.

## Inputs

- A fixed set of example labels and model scores.
- A threshold slider from 0.00 to 1.00.

## Outputs

- Confusion-matrix counts: true positives, false positives, true negatives, and false negatives.
- Precision, recall, and F1 score.
- A small table showing how each example changes when the threshold moves.

## Fallback

If JavaScript is unavailable, use the precomputed threshold table on the rendered textbook page. It shows the same examples at thresholds 0.30, 0.50, and 0.70.

## Non-goals

This lab does not train a model, upload student data, or claim that one threshold is universally optimal.
