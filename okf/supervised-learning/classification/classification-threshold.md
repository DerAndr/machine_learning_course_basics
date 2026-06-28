---
type: Concept
title: Classification Threshold
description: A decision cutoff that converts classifier scores into predicted labels.
tags: [classification, supervised-learning]
timestamp: 2026-06-28T00:00:00Z
status: draft
learning_objectives:
  - Explain why changing a threshold changes false positives and false negatives.
difficulty: introductory
estimated_reading_minutes: 4
prerequisites:
  - /supervised-learning/classification/classification.md
related_labs:
  - /labs/classification-threshold-explorer.md
source_materials:
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
---

# Classification Threshold

## Core idea

Many binary classifiers produce a score before they produce a final label. A classification threshold is the cutoff that turns that score into a decision: scores at or above the threshold become the positive class, while scores below it become the negative class.

## Why it matters

The threshold controls the balance between different mistakes. Lowering the threshold usually finds more positive cases, but it can also create more false positives. Raising the threshold usually makes positive predictions more selective, but it can miss more true positive cases.

## Example

If a model estimates that a customer has a 0.62 probability of responding to an offer, a threshold of 0.50 predicts positive. A threshold of 0.70 predicts negative. The model score stayed the same; the decision rule changed.

## Try it

Use the [classification threshold explorer](../../labs/classification-threshold-explorer.md) to move the threshold and watch the confusion matrix, precision, recall, and F1 score change.
