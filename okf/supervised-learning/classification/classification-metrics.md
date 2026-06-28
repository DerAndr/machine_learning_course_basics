---
type: Metric
title: Classification Metrics
description: Confusion-matrix metrics that describe different classification mistakes.
tags: [classification, supervised-learning, evaluation, metric]
timestamp: 2026-06-28T00:00:00Z
status: draft
learning_objectives:
  - Choose between precision, recall, and F1 from the cost of mistakes.
difficulty: introductory
estimated_reading_minutes: 5
prerequisites:
  - /supervised-learning/classification/classification.md
related_concepts:
  - /supervised-learning/classification/classification-threshold.md
related_labs:
  - /labs/classification-threshold-explorer.md
source_materials:
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
---

# Classification Metrics

## Core idea

Classification metrics summarize which decisions were right and which mistakes happened. The confusion matrix is the starting point: true positives and true negatives are correct decisions, while false positives and false negatives are different kinds of errors.

## Why it matters

Accuracy can hide the mistake that matters most. In some settings, a false negative is costly because the system misses a positive case. In others, a false positive is costly because the system raises too many alarms. Precision, recall, and F1 make those trade-offs visible.

## Key mechanics

- Precision asks: of the examples predicted positive, how many were actually positive?
- Recall asks: of the actual positive examples, how many did the model find?
- F1 combines precision and recall into one score when both matter.

## Go deeper

Next, study [classification thresholds](classification-threshold.md). The same model scores can produce different metrics when the threshold changes.
