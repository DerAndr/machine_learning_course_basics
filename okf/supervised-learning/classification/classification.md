---
type: Concept
title: Classification
description: Predicting discrete labels from input features.
tags: [classification, supervised-learning]
timestamp: 2026-06-28T00:00:00Z
status: draft
learning_objectives:
  - Formalize classification as learning a map from feature vectors to a finite label set.
  - Distinguish hard labels, scores, and estimated probabilities.
difficulty: introductory
estimated_reading_minutes: 8
related_concepts:
  - /supervised-learning/classification/classification-metrics.md
  - /supervised-learning/classification/classification-threshold.md
source_materials:
  - /lectures/lecture_05_classification_part_1/README.md
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
  - /lectures/lecture_05_classification_part_1/slides/lecture.pdf
  - /lectures/lecture_05_classification_part_1/practical_session/README.md
  - /lectures/lecture_05_classification_part_1/practical_session/classification_part1_practical_student_90min.ipynb
---

# Classification

## Core idea

Classification is a supervised learning task where the target is a discrete label. Instead of predicting a continuous number, the model chooses among named outcomes such as spam or not spam, approved or rejected, and class A or class B.

The basic supervised dataset is

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n},
$$

where each $x_i \in \mathcal{X}$ is a feature vector and each label $y_i$ belongs to a finite class set

$$
\mathcal{Y} = \{1, 2, \ldots, K\}.
$$

A classifier learns a prediction rule

$$
\hat{f}: \mathcal{X} \rightarrow \mathcal{Y}
$$

so that a new feature vector $x$ can be assigned a class label $\hat{y} = \hat{f}(x)$.

## Why it matters

Many real applications ask for a decision, not only a numeric estimate: detect fraud, route a support ticket, identify an image category, or decide whether a medical screening result should be escalated. A classifier can support that decision, but the model output and the final action are not always the same thing.

That distinction is crucial. Some classifiers output only a hard label. Others output a score, margin, logit, or estimated probability first. A separate decision rule then turns that continuous value into a class. The threshold can be changed without retraining the model, which is why classification cannot be understood from accuracy alone.

## Task types

Classification tasks differ in the structure of $\mathcal{Y}$:

- Binary classification has two classes, commonly encoded as $y \in \{0, 1\}$.
- Multiclass classification chooses exactly one label from $K > 2$ classes.
- Multilabel classification allows several labels to be true at the same time.
- Ordinal classification uses ordered labels such as low, medium, and high, but the distances between levels are not necessarily equal.

The binary case is the cleanest place to study thresholds and confusion-matrix metrics, so it is the first focus of this pilot.

## Scores, probabilities, and labels

For binary classification, many models first compute a score $s(x)$ and then convert it into a label. If the score is a calibrated probability, we can write

$$
p(x) = P(Y = 1 \mid X = x).
$$

A default label rule is often

$$
\hat{y} =
\begin{cases}
1, & p(x) \ge 0.5,\\
0, & p(x) < 0.5.
\end{cases}
$$

But $0.5$ is not a law of nature. It is only a decision convention. If false negatives and false positives have different costs, the threshold should usually move.

## Model families in this lecture

Lecture 05 contrasts several classifier families:

- K-nearest neighbors: a local vote among nearby training examples.
- Decision trees: recursive feature-threshold rules that split the feature space.
- Logistic regression: a linear score transformed into a probability.
- Support vector classifiers: margin-based separators.
- Naive Bayes: probabilistic classification with a conditional-independence assumption.

These families differ in geometry, assumptions, interpretability, and probability behavior. The common evaluation question is the same: what errors does the classifier make, and are those the errors the application can tolerate?

## Go deeper

Continue with [classification metrics](classification-metrics.md), then study [classification thresholds](classification-threshold.md), or follow the [Classification Part 1 learning path](../../learning-paths/classification-part-1.md).
