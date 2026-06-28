---
type: Concept
title: Classification Threshold
description: A decision cutoff that converts classifier scores into predicted labels.
tags: [classification, supervised-learning]
timestamp: 2026-06-28T00:00:00Z
status: draft
learning_objectives:
  - Write the threshold decision rule that converts a score into a binary label.
  - Explain why threshold movement changes precision, recall, and expected cost.
difficulty: introductory
estimated_reading_minutes: 8
prerequisites:
  - /supervised-learning/classification/classification.md
  - /supervised-learning/classification/classification-metrics.md
related_concepts:
  - /supervised-learning/classification/classification-metrics.md
related_labs:
  - /labs/classification-threshold-explorer.md
source_materials:
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
---

# Classification Threshold

## Core idea

Many binary classifiers produce a score before they produce a final label. A classification threshold is the cutoff that turns that score into a decision: scores at or above the threshold become the positive class, while scores below it become the negative class.

If the model score is $s(x)$ and the threshold is $\tau$, the binary decision rule is

$$
\hat{y}_\tau(x) =
\begin{cases}
1, & s(x) \ge \tau,\\
0, & s(x) < \tau.
\end{cases}
$$

For logistic regression, the score is often an estimated probability:

$$
P(Y=1 \mid X=x) = \sigma(w^\top x + b)
= \frac{1}{1 + e^{-(w^\top x + b)}}.
$$

The default $\tau = 0.5$ is common, but it is not automatically optimal.

## Why it matters

The threshold controls the balance between different mistakes. Lowering the threshold usually finds more positive cases, but it can also create more false positives. Raising the threshold usually makes positive predictions more selective, but it can miss more true positive cases. Use [classification metrics](classification-metrics.md) to describe that trade-off precisely.

At threshold $\tau$, each example contributes to the confusion matrix:

$$
TP(\tau),\ FP(\tau),\ TN(\tau),\ FN(\tau).
$$

So precision, recall, F1, and false positive rate are all functions of $\tau$. Moving the threshold changes the metrics even when the fitted model and its scores stay fixed.

## Example

If a model estimates that a customer has a 0.62 probability of responding to an offer, a threshold of 0.50 predicts positive. A threshold of 0.70 predicts negative. The model score stayed the same; the decision rule changed.

That is the point students often miss: threshold tuning is not retraining. It is choosing the operating point of a scoring model.

## Cost-based threshold intuition

Suppose $p(x) = P(Y=1 \mid X=x)$ is a calibrated probability. Let $C_{FP}$ be the cost of a false positive and $C_{FN}$ be the cost of a false negative. Predicting positive has expected error cost

$$
C_{FP}(1 - p(x)),
$$

while predicting negative has expected error cost

$$
C_{FN}p(x).
$$

Choose the positive class when

$$
C_{FP}(1 - p(x)) \le C_{FN}p(x).
$$

After rearranging:

$$
p(x) \ge \frac{C_{FP}}{C_{FP} + C_{FN}}.
$$

This formula is not a universal recipe because real deployments may include capacity limits, calibration error, fairness constraints, or downstream human review. But it shows why threshold choice belongs to the decision problem, not only to the model.

## Practical threshold search

In practice, threshold selection usually uses a validation set:

- Fit the model on training data.
- Produce validation scores.
- Sweep candidate thresholds.
- Compute the chosen metric or cost at each threshold.
- Pick the threshold before touching the final test set.

Common choices include maximizing $F_1$, maximizing $F_\beta$, controlling false positive rate, or choosing a threshold that matches an application-specific cost.

## Try it

Use the [classification threshold explorer](../../labs/classification-threshold-explorer.md) to move the threshold and watch the confusion matrix, precision, recall, and F1 score change.
