---
type: Metric
title: Classification Metrics
description: Confusion-matrix metrics that describe different classification mistakes.
tags: [classification, supervised-learning, evaluation, metric]
timestamp: 2026-06-28T00:00:00Z
status: draft
learning_objectives:
  - Compute accuracy, precision, recall, F1, and F-beta from confusion-matrix counts.
  - Choose a metric from the cost of false positives and false negatives.
difficulty: introductory
estimated_reading_minutes: 10
prerequisites:
  - /supervised-learning/classification/classification.md
related_concepts:
  - /supervised-learning/classification/classification-threshold.md
related_labs:
  - /labs/classification-threshold-explorer.md
source_materials:
  - /lectures/lecture_05_classification_part_1/README.md
  - /lectures/lecture_05_classification_part_1/lecture_notes.md
  - /lectures/lecture_05_classification_part_1/slides/lecture.pdf
  - /lectures/lecture_05_classification_part_1/practical_session/README.md
  - /lectures/lecture_05_classification_part_1/practical_session/classification_part1_practical_student_90min.ipynb
---

# Classification Metrics

## Core idea

Classification metrics summarize which decisions were right and which mistakes happened. The confusion matrix is the starting point: true positives and true negatives are correct decisions, while false positives and false negatives are different kinds of errors.

## Why it matters

Accuracy can hide the mistake that matters most. In some settings, a false negative is costly because the system misses a positive case. In others, a false positive is costly because the system raises too many alarms. Precision, recall, and F1 make those trade-offs visible.

## Confusion matrix

For binary classification, every prediction falls into one of four cells:

- Actually positive and predicted positive: true positive $TP$.
- Actually positive and predicted negative: false negative $FN$.
- Actually negative and predicted positive: false positive $FP$.
- Actually negative and predicted negative: true negative $TN$.

The names are not decorative. They tell you which mistake happened:

- $FP$: the model raised a positive decision for a negative case.
- $FN$: the model missed a positive case.

Most binary classification metrics are functions of these four counts.

## Accuracy

Accuracy measures the fraction of correct predictions:

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}.
$$

It is useful when classes are balanced and error costs are roughly symmetric. It is dangerous when the positive class is rare. If only 2% of cases are positive, a classifier that always predicts negative can reach 98% accuracy while being useless for finding positives.

## Precision and recall

Precision answers: among predicted positives, how many were actually positive?

$$
\mathrm{Precision} = \frac{TP}{TP + FP}.
$$

High precision matters when false positives are expensive: for example, when every positive prediction triggers a manual investigation.

Recall answers: among actual positives, how many did the model catch?

$$
\mathrm{Recall} = \frac{TP}{TP + FN}.
$$

High recall matters when false negatives are expensive: for example, when missing a positive case is risky.

## F1 and F-beta

F1 is the harmonic mean of precision and recall:

$$
F_1 = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}.
$$

The more general $F_\beta$ score is

$$
F_\beta = (1 + \beta^2) \cdot
\frac{\mathrm{Precision} \cdot \mathrm{Recall}}
{\beta^2 \cdot \mathrm{Precision} + \mathrm{Recall}}.
$$

Use $\beta > 1$ when recall should matter more. Use $\beta < 1$ when precision should matter more.

## Rates used by ROC curves

The true positive rate is the same quantity as recall:

$$
TPR = \frac{TP}{TP + FN}.
$$

The false positive rate is

$$
FPR = \frac{FP}{FP + TN}.
$$

An ROC curve plots $TPR$ against $FPR$ across many thresholds. This makes it a threshold-sweep diagnostic rather than a single fixed-threshold score. AUC summarizes ranking quality: a random ranking is near $0.5$, while a perfect ranking is $1.0$.

## Probability metrics

Hard-label metrics ignore how confident the model was. Log loss evaluates predicted probabilities directly. For binary labels $y_i \in \{0, 1\}$ and predicted probabilities $\hat{p}_i$, the average log loss is

$$
-\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)
\right].
$$

This loss punishes confident wrong predictions strongly. It is important when the probability itself is used downstream, not just the final label.

## Choosing a metric

Do not choose a metric by habit. Start with the decision cost:

- If false positives are costly, inspect precision and $F_\beta$ with $\beta < 1$.
- If false negatives are costly, inspect recall and $F_\beta$ with $\beta > 1$.
- If you need ranking behavior across thresholds, inspect ROC-AUC and precision-recall curves.
- If calibrated probabilities matter, inspect log loss and calibration diagnostics.

## Go deeper

Next, study [classification thresholds](classification-threshold.md). The same model scores can produce different metrics when the threshold changes.
