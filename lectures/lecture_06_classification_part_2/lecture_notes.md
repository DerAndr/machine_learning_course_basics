# Lecture 06 Recap: Classification Part 2

> Lecture number: 06
> Lecture slug: `lecture_06_classification_part_2`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What This Lecture Adds to Basic Classification

The first classification lecture focused on:

- basic binary classification setup
- standard classifiers
- confusion matrix and core metrics
- threshold tuning for binary tasks

This lecture moves into more difficult real-world settings:

- binary class imbalance
- multiclass classification
- multilabel classification
- ordinal classification
- ranking-oriented metrics

The central idea is that once we leave the clean balanced binary case, the whole evaluation logic changes.

In advanced classification tasks, the main difficulty is often not only the model itself, but:

- choosing the right metric
- handling skewed label distributions
- preserving class structure
- respecting label dependencies or ordering

## 2. Binary Imbalanced Classification

Binary imbalance occurs when one class is much rarer than the other.

Typical examples from the slides:

- fraud detection
- rare disease diagnosis
- loan approval risk problems

The lecture emphasizes the main practical effect:

- many models become biased toward the majority class
- minority-class performance becomes poor

This is one of the most common sources of misleading "good" results in applied ML.

## 3. Why Accuracy Fails Under Imbalance

The slides start with a critical warning:

- high accuracy can be achieved by predicting only the majority class

This is exactly right.

Suppose fraud cases are only a tiny fraction of the dataset. A classifier that always predicts `"not fraud"` may have excellent accuracy while being useless in practice.

That is why, under imbalance, evaluation must shift from:

- overall correctness

to:

- how well the minority class is detected
- what types of errors the model makes

## 4. Better Metrics for Imbalanced Binary Problems

The lecture highlights:

- precision
- recall
- F1-score
- ROC-AUC
- precision-recall curve

These are the correct metrics to emphasize because they separate different kinds of success and failure.

### 4.1 Precision

`Precision = TP / (TP + FP)`

Precision answers:

- when the model predicts the positive class, how often is it correct?

This matters when false positives are costly.

Examples:

- flagging legitimate transactions as fraud
- sending too many false alarms to a doctor or analyst

### 4.2 Recall

`Recall = TP / (TP + FN)`

Recall answers:

- among all truly positive cases, how many did the model catch?

This matters when false negatives are costly.

Examples:

- missing fraud
- missing a serious disease

### 4.3 F1-Score

F1 is the harmonic mean of precision and recall.

Why harmonic mean?

- it punishes imbalance between the two
- a model cannot get a high F1 by being excellent on one and poor on the other

This makes F1 useful when:

- both precision and recall matter
- there is no simple single-cost interpretation

## 5. ROC-AUC Under Imbalance

The lecture gives a stronger theoretical interpretation of ROC-AUC:

`ROC AUC = P(score(X_pos) > score(X_neg))`

This means:

- ROC-AUC is the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative one

This is a ranking interpretation, not just a geometric one.

### 5.1 Why ROC-AUC Is Useful

The slides correctly emphasize:

- threshold independence
- class-distribution independence
- ranking ability

ROC-AUC is useful when we care about the classifier’s global ability to separate positives from negatives across many thresholds.

### 5.2 Limitation of ROC-AUC

The lecture also gives an important caveat:

- under extreme imbalance, ROC-AUC can look overly optimistic

Why:

- false positive rate uses `FP / (FP + TN)`
- if `TN` is huge, `FPR` can remain numerically small even when the model produces many false positives in practice

This is why precision-recall analysis often becomes more informative for rare-event detection.

## 6. Precision-Recall Curve and Average Precision

The lecture says the PR curve is often better for imbalanced data, and that is correct.

### 6.1 What the PR Curve Shows

It plots:

- precision
- against recall

across many thresholds.

This directly focuses on:

- minority-class detection quality

instead of mixing in the very large number of true negatives.

### 6.2 Average Precision (AP)

The slides define AP as a weighted average of precisions over recall increments.

Interpretation:

- high AP means the classifier keeps precision relatively high while increasing recall
- low AP means the model cannot recover many positives without also producing many false positives

This is a very useful scalar summary when rare positive detection matters.

## 7. Techniques to Address Imbalance

The lecture divides solutions into:

- data-level methods
- algorithm-level methods

This is the right conceptual split.

## 8. Oversampling and Undersampling

### 8.1 Oversampling

Oversampling increases representation of the minority class.

Two types mentioned in the lecture:

- random oversampling
- SMOTE

#### Random Oversampling

Duplicates minority samples.

Pros:

- simple
- often improves minority recall

Cons:

- can overfit because the model repeatedly sees identical minority examples

#### SMOTE

SMOTE generates synthetic minority samples by interpolating between nearby minority points.

The lecture gives the core formula:

`x_new = x_i + lambda (x_j - x_i)`

where `lambda` is between `0` and `1`.

This is important because SMOTE is not simple duplication. It creates new points along minority-class line segments.

Pros:

- reduces the exact-duplicate problem
- often improves generalization compared with random oversampling

Cons:

- can create unrealistic synthetic points
- may blur class boundaries if applied carelessly
- should still be applied only on training data

### 8.2 Undersampling

Undersampling reduces the majority class.

Methods mentioned:

- random undersampling
- cluster-based undersampling

Pros:

- faster training
- less majority bias

Cons:

- can discard useful information
- may underfit if too much structure is removed

### 8.3 When to Use Which

The lecture gives a practical rule:

- use oversampling when the dataset is small and the minority class needs more representation
- use undersampling when the dataset is large and losing some majority samples is acceptable

That is a good starting heuristic.

## 9. Cost-Sensitive Learning

The lecture then introduces algorithm-level correction.

Instead of changing the data, cost-sensitive learning changes the optimization objective.

Main idea:

- misclassifying a minority or costly class should be penalized more heavily

The slides mention:

- class weights
- cost matrices
- weighted loss

This is often preferable when:

- we want to preserve the original data distribution
- model classes support weighted training well
- business costs are asymmetric

Examples from the lecture:

- fraud detection
- medical diagnosis
- loan approval

## 10. Model-Specific Strategies Under Imbalance

The lecture gives a useful model-by-model table.

### KNN

Challenges:

- majority neighbors dominate local voting

Solutions:

- weighted KNN
- experiment with distance metrics

### Decision Trees

Challenges:

- can overfit majority patterns

Solutions:

- class weighting
- pruning

### Logistic Regression

Challenges:

- decision boundary may favor majority class

Solutions:

- class weighting
- threshold tuning

### Naive Bayes

Challenges:

- rare-class events may be weakly represented

Solutions:

- adjust class priors

### SVM

Challenges:

- majority class can dominate the margin

Solutions:

- class weighting
- one-class variants for anomaly-style setups

This table is practical because it reminds students that imbalance is not solved by one universal trick.

## 11. Practical Notebook: Imbalanced Binary Classification

`Classification2 - 1.ipynb` is the main binary-imbalance notebook.

Dataset:

- credit card fraud detection

This is a very appropriate example because fraud is a classic rare-event setting.

The notebook covers:

- class distribution analysis
- baseline logistic regression
- stratified split
- scaling of `Time` and `Amount`
- random oversampling
- weighted logistic regression
- confusion matrices
- ROC comparison
- cross-validated ROC-AUC
- cross-validated precision-recall curves
- average precision

This is especially valuable because it demonstrates a very common real-world pattern:

1. baseline model looks weak on the minority class
2. resampling or weighting improves recall
3. evaluation must move beyond accuracy

## 12. Multiclass Classification

The lecture then moves from binary classification to multiclass problems.

Definition:

- each instance belongs to exactly one of three or more classes

Examples:

- digit recognition
- quality score prediction if treated as nominal classes
- general category assignment

The main change from binary classification is not only the number of classes. The whole error structure becomes more complex.

## 13. Main Multiclass Strategies

The lecture presents three approaches:

- One-vs-Rest (OvR)
- One-vs-One (OvO)
- multinomial models

### 13.1 One-vs-Rest

Build `K` binary classifiers:

- one classifier per class
- each class is treated as positive against all others

Pros:

- simple
- scalable
- natural extension for many binary models

Cons:

- each classifier sees an imbalanced subproblem
- probability calibration across classes may be imperfect

### 13.2 One-vs-One

Build one classifier for every pair of classes.

Number of classifiers:

- `K(K - 1) / 2`

Pros:

- each classifier solves a simpler pairwise problem

Cons:

- grows quickly with the number of classes
- can be computationally heavier

### 13.3 Multinomial Models

These extend the model directly to multiple classes rather than decomposing into binary tasks.

This is often the cleaner probabilistic formulation when the underlying algorithm supports it.

## 14. Multiclass Evaluation Metrics

The lecture focuses on:

- confusion matrix extension
- per-class precision
- per-class recall
- macro-averaging
- micro-averaging

This is exactly the right evaluation layer.

### 14.1 Per-Class Metrics

For each class `i`, compute:

- `TP_i`
- `FP_i`
- `FN_i`

Then define:

- `Precision_i = TP_i / (TP_i + FP_i)`
- `Recall_i = TP_i / (TP_i + FN_i)`

This matters because a multiclass model can perform well on one class and poorly on another.

### 14.2 Macro-Averaging

Macro-averaging gives equal weight to every class.

Meaning:

- rare classes matter as much as common classes

Use macro metrics when:

- all classes are equally important
- you care about balanced performance across the label space

### 14.3 Micro-Averaging

Micro-averaging pools counts across classes before computing the metric.

Meaning:

- common classes contribute more
- overall instance-level performance is emphasized

Use micro metrics when:

- overall performance is the main goal
- class frequency should influence the average

The lecture explicitly says micro averaging is often preferable under class imbalance, which is a useful rule.

## 15. Practical Notebook: Multiclass Classification

`Classification2 -2.ipynb` covers multiclass classification with logistic regression.

Dataset:

- Wine Quality

Main topics:

- imbalanced target distribution
- stratified split
- scaling
- multiclass SMOTE
- OvR logistic regression
- OvO logistic regression
- confusion matrices
- classification reports
- per-class F1 comparison
- cross-validation with SMOTE inside each fold

This notebook is important because it shows a correct evaluation pattern:

- resampling is applied inside the fold logic
- model strategies are compared directly
- per-class behavior is inspected instead of relying only on overall accuracy

## 16. Multilabel Classification

Multilabel classification is different from multiclass classification.

Main rule:

- one instance can belong to several labels at the same time

Examples from the lecture:

- image tagging
- article tagging
- recommendation-like annotation problems

This changes evaluation substantially because "partially correct" predictions are common and meaningful.

## 17. Example-Based vs Label-Based Metrics

The lecture divides multilabel metrics into two families.

### 17.1 Example-Based Metrics

These evaluate each instance as a full set of labels.

Useful when:

- you care about label sets per sample

### 17.2 Label-Based Metrics

These evaluate each label across the dataset.

Useful when:

- you care about how well each label is modeled individually

This distinction is important because multilabel tasks can be judged either by:

- how good the full label bundle is per sample
- or how good the model is at each tag separately

## 18. Multilabel Example-Based Metrics

The lecture covers:

- Hamming Loss
- Exact Match Ratio (EMR)
- 1/0 Loss
- precision
- recall
- F1 score

### 18.1 Hamming Loss

Measures the fraction of misclassified labels across all instances and labels.

Interpretation:

- lower is better

Why it is useful:

- it is less harsh than exact-match metrics
- each label contributes separately

### 18.2 Exact Match Ratio

Measures the fraction of instances whose predicted label set matches the true label set exactly.

This is a very strict metric.

Even one missing or extra label makes the whole instance incorrect.

### 18.3 1/0 Loss

Counts whether an instance has at least one label error.

This is also strict, though conceptually different from exact-match framing.

### 18.4 Example-Based Precision, Recall, and F1

These adapt the familiar binary concepts to multilabel sets.

For one instance:

- precision measures how many predicted labels are truly relevant
- recall measures how many true labels were recovered
- F1 balances the two

The slides also include a concrete worked example with label sets such as `{cat, dog, tree}` and `{bird, fish}`, which is helpful because multilabel metrics can otherwise feel abstract.

## 19. Label-Based Metrics and LRAP

The lecture highlights:

- micro-averaged metrics
- macro-averaged metrics
- Label Ranking Average Precision (LRAP)

### 19.1 Micro-Averaged Metrics

Weighted by overall label frequency.

Best when:

- overall label prediction quality matters
- common labels should influence the result more

### 19.2 Macro-Averaged Metrics

Treat every label equally.

Best when:

- each label is equally important
- rare labels must not be drowned out by common ones

### 19.3 LRAP

This is one of the most important new metrics in the lecture.

LRAP evaluates how well the model ranks true labels above false labels for each instance.

Interpretation:

- high LRAP means relevant labels are consistently ranked higher
- low LRAP means irrelevant labels often outrank relevant ones

Why LRAP matters:

- in many multilabel systems, ranking is more important than a hard cut-off
- recommendation and tagging systems often care about order, not only final label sets

This connects multilabel classification to ranking problems more generally.

## 20. Multilabel Modeling Approaches

The lecture briefly introduces several multilabel model families.

### Problem Transformation Methods

- Binary Relevance
- Classifier Chains
- Label Powerset

#### Binary Relevance

Train one binary classifier per label.

Pros:

- simple
- flexible

Cons:

- ignores label dependencies

#### Classifier Chains

Predict labels sequentially, using previous label predictions as features.

Pros:

- can capture label dependence

Cons:

- sensitive to label order
- computationally more demanding

#### Label Powerset

Treat each unique label combination as one class.

Pros:

- naturally captures dependence

Cons:

- scales poorly when many combinations exist
- can overfit sparse label combinations

### Algorithm Adaptation and Ensembles

The lecture also mentions:

- ML-kNN
- multilabel decision trees
- multilabel SVMs
- RAkEL
- ensemble classifier chains

The student does not need to memorize all implementations, but should understand the broader idea:

- multilabel tasks often require methods that either model label dependence explicitly or scale robustly despite it

## 21. Practical Notebook: Multilabel Classification

`Classification2 -3.ipynb` uses a synthetic multilabel dataset.

Main topics:

- generating multilabel data
- label distribution inspection
- scaling
- One-vs-Rest logistic regression
- Hamming loss
- subset accuracy
- classification report
- LRAP
- cross-validated Hamming loss
- cross-validated subset accuracy
- cross-validated LRAP

This notebook is useful because it gives students direct exposure to metrics that almost never appear in standard binary-classification tutorials.

## 22. Ranking Problems and Ordinal Classification

The final theoretical block of the lecture introduces ranking and ordinal structure.

### 22.1 Ranking Problems

In ranking tasks, order matters more than hard class assignment.

Examples from the slides:

- search results
- recommendation systems
- question answering

This is why ranking-oriented metrics appear in the lecture: some classification-like tasks are really about getting the ordering right.

### 22.2 Ordinal Classification

Ordinal classification is a special case where classes are ordered:

- low < medium < high

This is different from nominal multiclass classification because:

- predicting `"high"` instead of `"medium"` is not as bad as predicting `"high"` instead of `"low"` in many applications

So ordinary class metrics can miss the structure of the problem.

## 23. Metrics for Ordinal Classification

The slides mention:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Quadratic Weighted Kappa (QWK)
- Spearman rank correlation

### 23.1 MAE

Measures average absolute difference between predicted and true class values.

This makes sense for ordinal problems because label distances carry meaning.

### 23.2 MSE

Squares the discrepancy, so larger mistakes are penalized more strongly.

This is useful when far-off ordinal errors are especially undesirable.

### 23.3 Quadratic Weighted Kappa

QWK measures agreement while penalizing larger disagreements more heavily.

Why it is valuable:

- respects order structure
- widely used in rating and scoring tasks

### 23.4 Spearman Rank Correlation

Useful when:

- relative ordering matters more than exact label identity

This is a natural bridge between ordinal classification and ranking evaluation.

## 24. Techniques for Ordinal Classification

The lecture introduces:

- ordinal logistic regression
- threshold-based regression
- pairwise comparison / ordinal ensembles

### 24.1 Ordinal Logistic Regression

This extends logistic regression to ordered classes.

Instead of modeling one nominal class against another, it models cumulative ordered boundaries.

This is usually a better fit than ordinary multiclass classification when order matters.

### 24.2 Threshold-Based Regression

Train a regression model to predict a score, then map ranges of that score back to ordered classes.

Pros:

- flexible

Cons:

- thresholds must be chosen carefully

### 24.3 Pairwise or Ensemble Methods

These compare relative order or combine several ordered subproblems.

Pros:

- can capture richer ordinal structure

Cons:

- often computationally heavier

## 25. Practical Notebook: Ordinal Classification

`Classification2 -4.ipynb` demonstrates ordinal logistic regression.

Dataset:

- Wine Quality

Main topics:

- ordered target interpretation
- stratified split
- scaling
- ordinal logistic regression with `mord`
- MAE
- QWK
- confusion matrix
- classification report

This notebook is especially valuable because it teaches a subtle but important modeling lesson:

- not every integer-coded target should be treated as ordinary multiclass classification

If the class order carries meaning, the model and the metric should reflect that.

## 26. Key Takeaways

- Class imbalance changes both training and evaluation.
- Accuracy is often misleading for rare-event detection.
- ROC-AUC is useful, but PR-based metrics are often more informative under strong imbalance.
- Oversampling, undersampling, SMOTE, and class weighting solve different imbalance problems.
- Multiclass classification requires per-class analysis and careful averaging choices.
- Macro metrics treat classes equally; micro metrics weight by frequency.
- Multilabel classification needs its own metric family because partial correctness matters.
- LRAP is important when label ranking matters.
- Ordinal classification is not the same as ordinary multiclass classification.
- Ordered targets should often be evaluated with order-aware metrics such as MAE or QWK.

## 27. Quick Revision Questions

1. Why can a model with very high accuracy still be useless on an imbalanced fraud dataset?
2. When is PR analysis more informative than ROC-AUC?
3. What is the main difference between random oversampling and SMOTE?
4. Why must resampling be applied only on training data or within cross-validation folds?
5. What is the conceptual difference between OvR and OvO in multiclass classification?
6. When would macro-averaged metrics be preferable to micro-averaged metrics?
7. Why is multilabel classification different from multiclass classification?
8. What is the difference between Hamming Loss and Exact Match Ratio?
9. What does LRAP evaluate that ordinary multilabel accuracy does not?
10. Why is QWK often better than plain accuracy for ordinal targets?
