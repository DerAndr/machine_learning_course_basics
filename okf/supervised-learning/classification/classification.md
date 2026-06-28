---
type: Concept
title: Classification
description: Predicting discrete labels from input features.
tags: [classification, supervised-learning]
timestamp: 2026-06-28T00:00:00Z
status: draft
learning_objectives:
  - Distinguish classification targets from regression targets.
difficulty: introductory
estimated_reading_minutes: 3
related_concepts:
  - /supervised-learning/classification/classification-threshold.md
source_materials:
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
---

# Classification

## Core idea

Classification is a supervised learning task where the target is a discrete label. Instead of predicting a continuous number, the model chooses among named outcomes such as spam or not spam, approved or rejected, and class A or class B.

## Why it matters

Many real applications ask for a decision, not only a numeric estimate. A classifier can support that decision, but the model output and the final action are not always the same thing. This is especially visible in binary classification, where a model may produce a score and a separate threshold turns that score into a label.

## Go deeper

Continue with [classification thresholds](classification-threshold.md) or follow the [Classification Part 1 learning path](../../learning-paths/classification-part-1.md).
