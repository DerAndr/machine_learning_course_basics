# Lecture 05 Recap: Classification Part 1

> Lecture number: 05
> Lecture slug: `lecture_05_classification_part_1`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What Classification Solves

Classification is used when the target is **categorical** rather than continuous.

The lecture defines classification as the task of assigning labels to new observations based on patterns learned from historical data.

Typical examples from the slides:

- spam detection
- image recognition
- disease diagnosis
- antifraud systems

Mathematically, the lecture writes the setup as:

- each observation has a feature vector `X_i`
- each observation has a class label `Y_i`
- the label belongs to a set of classes `C = {1, 2, ..., k}`

The goal is to approximate a function:

`Y = f(X)`

that can classify new instances correctly.

The key difference from regression is:

- regression predicts a number
- classification predicts a category or a probability over categories

## 2. Main Types of Classification

The slides distinguish several common task types.

### 2.1 Binary Classification

Two possible outcomes.

Examples:

- fraud / not fraud
- disease / no disease
- spam / not spam

This is the main focus of the practical notebooks in this lecture.

### 2.2 Multiclass Classification

More than two classes.

Example:

- cat / dog / bird

### 2.3 Multilabel Classification

An observation can have several labels at once.

Example:

- an image tagged as both `"indoor"` and `"nature"`

### 2.4 Ordinal Classification

Classes have a natural order.

Example:

- low / medium / high

Important subtlety:

- ordinal classes are ordered
- but the distances between levels are not necessarily equal

That makes them different from both nominal classification and regression.

## 3. Basic Classification Models in This Lecture

The lecture introduces five baseline model families:

- K-Nearest Neighbors
- Decision Tree Classifier
- Logistic Regression
- Support Vector Classifier
- Naive Bayes

This is a useful set because the models rely on very different principles:

- geometry and distance
- recursive splitting
- probabilistic linear boundary
- maximum-margin classification
- probabilistic generative modeling

Understanding these differences helps students choose models more intelligently later.

## 4. K-Nearest Neighbors (KNN)

KNN is one of the simplest classification algorithms.

It is **non-parametric** and **instance-based**:

- it does not learn a compact parametric model in the usual sense
- it stores the training data
- prediction is based on the neighborhood of a new point

The lecture defines the rule as:

- find the `k` nearest neighbors of the new point
- assign the majority class among those neighbors

So KNN works by local voting.

### 4.1 Why KNN Is Intuitive

KNN assumes that similar points tend to have similar labels.

This assumption is often reasonable in low- or medium-dimensional feature spaces where:

- distances are meaningful
- classes cluster locally

### 4.2 Choice of `k`

The lecture correctly notes that `k` should be selected with:

- cross-validation
- grid search
- or a heuristic such as elbow-style inspection

Why `k` matters:

- small `k` gives low bias but high variance
- large `k` gives smoother decision regions but can oversmooth class boundaries

So `k` controls the bias-variance tradeoff.

### 4.3 Weighted KNN

The notes PDF explicitly mentions weighted KNN.

Instead of giving each neighbor the same vote, weighted KNN gives larger influence to closer neighbors.

Why this helps:

- nearby points are often more informative than distant points
- it can reduce the effect of irrelevant or weakly related neighbors

This appears directly in the practical notebook.

## 5. Distance Metrics in KNN

The lecture emphasizes that KNN depends heavily on the distance metric.

This is one of the most important practical facts about the model.

### 5.1 Euclidean Distance

The standard geometric distance in continuous spaces.

Best for:

- continuous numerical features
- situations where straight-line geometry makes sense

### 5.2 Manhattan Distance

Uses absolute differences instead of squared differences.

Best for:

- grid-like structures
- cases where axis-wise movement is more natural

It can also be more robust than Euclidean distance in some settings because it does not square deviations.

### 5.3 Minkowski Distance

A general family that includes:

- Manhattan distance when `p = 1`
- Euclidean distance when `p = 2`

This reminds students that Euclidean and Manhattan are special cases of a broader metric family.

### 5.4 Cosine Distance

Measures angular difference rather than raw Euclidean geometry.

Best for:

- high-dimensional sparse data
- text-like vector spaces

Why:

- direction matters more than magnitude in many text and embedding problems

## 6. Complexity and Limits of KNN

The lecture makes an important computational point:

- training complexity is essentially `O(1)` in the simple view because KNN stores data
- prediction complexity grows with the number of training samples and features

In practice, this means:

- KNN is cheap to fit
- but expensive to query on large datasets

The slides also mention:

- KD-Trees
- approximate nearest neighbors

These are practical speed-up techniques.

### 6.1 Curse of Dimensionality

This is one of the main weaknesses of KNN.

As dimensionality grows:

- data becomes sparse
- nearest and farthest points start to look similarly distant
- local neighborhoods become less informative

The slides mention **distance concentration**, which is the technical idea that in very high dimensions, distances can collapse into a narrow range.

That is why KNN often performs worse when:

- there are many irrelevant features
- dimensionality is high
- features are poorly scaled

### 6.2 Practical Limitations

The lecture lists:

- sensitivity to noise
- curse of dimensionality
- computational inefficiency

All three are highly practical concerns, not just theoretical footnotes.

## 7. Decision Tree Classifier

Decision trees classify by recursively splitting the feature space into regions.

Each internal node applies a decision rule such as:

- `feature <= threshold`

Each leaf node corresponds to a prediction.

This makes decision trees one of the most interpretable classification models.

## 8. Purity, Gini, and Entropy

The lecture explains tree construction through node purity.

The main intuition is:

- good splits create child nodes with more homogeneous class labels

### 8.1 Gini Impurity

The slide gives:

`Gini(D) = 1 - sum_c p(c)^2`

Interpretation:

- if a node contains mostly one class, impurity is low
- if classes are mixed, impurity is higher

Gini can be interpreted as the probability of misclassification if a label is assigned randomly according to the node class distribution.

### 8.2 Entropy

Entropy measures disorder:

`H(D) = - sum_c p(c) log p(c)`

High entropy means:

- classes are mixed

Low entropy means:

- the node is more pure

### 8.3 Information Gain

Information gain measures the reduction in entropy after a split.

So a split is good when it reduces uncertainty strongly.

This is one of the key theoretical foundations of decision trees.

## 9. Geometry of Decision Trees

The lecture notes that decision-tree boundaries are **axis-aligned**.

That means the model splits one feature at a time, producing rectangular regions in feature space.

Why this matters:

- very good for tabular data with threshold-like logic
- less natural for diagonal, circular, or curved decision boundaries

This is one of the main reasons why trees can struggle on data with non-rectangular class separation unless ensembles or deeper trees are used.

The `Classification - prep.ipynb` notebook includes simple synthetic visual illustrations of this idea.

## 10. Complexity and Overfitting in Trees

The lecture states that trees are prone to overfitting, especially when they grow deep and start fitting noise.

This is a central practical issue.

### 10.1 Why Trees Overfit

Because each split is chosen greedily to improve purity on the training data, a deep tree can:

- keep splitting until tiny training patterns are memorized
- create fragile rules that do not generalize

### 10.2 Control Methods

The slides mention:

- pruning
- maximum depth
- minimum samples per leaf
- minimum samples per split

These all reduce effective model complexity.

The practical notebook goes further and demonstrates:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- cost-complexity pruning via `ccp_alpha`

This is a strong practical set because students can see how structure control changes the model, not just the score.

## 11. Why Trees Are Interpretable

The lecture highlights four reasons:

- step-by-step decision paths
- readable if-then rules
- direct visualizations
- feature importance scores

This makes trees attractive in domains where explanation matters, such as:

- medical decision support
- credit scoring
- regulated business settings

Important caveat:

- interpretability is strongest for shallow trees
- a very deep tree may still be technically transparent, but not cognitively easy to understand

## 12. Logistic Regression

Logistic regression is a linear probabilistic classifier for binary outcomes.

The lecture defines it through the sigmoid function:

`P(Y = 1 | X) = sigma(wX + b) = 1 / (1 + exp(-(wX + b)))`

This is a crucial difference from ordinary linear regression:

- linear regression predicts an unrestricted numeric value
- logistic regression transforms a linear score into a probability in `[0, 1]`

## 13. Decision Rule and Threshold

The lecture writes the prediction rule using a threshold `tau`, usually `0.5`.

So:

- if predicted probability is above the threshold, classify as positive
- otherwise classify as negative

This is important because logistic regression is not only a classifier. It is also a **probability estimator**.

That means the threshold can be changed depending on the application:

- higher threshold if false positives are expensive
- lower threshold if false negatives are expensive

This threshold-tuning idea is used directly in the practical notebooks.

## 14. Logistic Loss

The lecture gives the binary cross-entropy or log-loss objective:

`J(w, b) = -(1/n) sum [ y_i log(y_hat_i) + (1 - y_i) log(1 - y_hat_i) ]`

Why this loss is important:

- it fits naturally to probabilistic outputs
- it penalizes confident wrong predictions strongly

This makes logistic regression fundamentally different from simply fitting a line and thresholding it.

## 15. Assumptions and Requirements of Logistic Regression

The slides list several practical conditions:

- linearly separable or nearly linearly separable data works best
- independent observations
- low multicollinearity
- limited outlier influence
- feature scaling recommended

Important nuance:

- exact perfect linear separability is not required
- but logistic regression is still a linear-boundary model in the feature space

So if the true class boundary is strongly non-linear, logistic regression may underfit unless features are transformed.

## 16. Explainability of Logistic Regression

One of logistic regression’s main strengths is interpretability.

The lecture explains coefficients through **log-odds**.

Meaning:

- each coefficient changes the log-odds of the positive class

If a coefficient is:

- positive, it increases the log-odds of the positive class
- negative, it decreases them

Exponentiating a coefficient gives an **odds ratio**.

That is an important practical tool in fields like:

- medicine
- economics
- credit risk

Because it turns model parameters into interpretable multiplicative effects on odds.

## 17. Limits of Logistic Regression

The lecture mentions:

- linear boundary bias
- sensitivity to outliers
- multicollinearity issues
- imbalance sensitivity
- multiclass needs extensions

This is all correct and practically important.

Students should remember:

- logistic regression is strong when the boundary is simple and interpretability matters
- it is weak when the data geometry is highly non-linear or heavily imbalanced without adjustment

## 18. Support Vector Classifier (SVC)

The lecture gives a high-level overview of SVC.

The core idea is:

- find a hyperplane that separates classes
- maximize the margin between classes

The margin is the distance between the decision boundary and the closest training points from each class.

Those closest points are the **support vectors**.

Why margin matters:

- a larger margin usually means a more robust separator
- the model focuses on the most informative boundary points

## 19. Why SVC Is Powerful

The slides list several strengths:

- effective in high dimensions
- robust classification boundary
- can handle non-linear data with kernels

The kernel idea is important:

- instead of explicitly transforming data to a higher-dimensional space
- SVC can compute inner products in that space implicitly

This allows curved decision boundaries in the original feature space.

## 20. Limits of SVC

The slides mention:

- high computational cost
- parameter sensitivity
- no native probability estimates in the basic form

These are important tradeoffs.

SVC can be very strong on medium-sized, structured datasets, but:

- tuning `C`, kernel choice, and kernel parameters matters a lot
- training can become expensive
- calibration may be needed when well-behaved probabilities are required

## 21. Naive Bayes

Naive Bayes is introduced as a probabilistic classifier based on Bayes' theorem.

Its key assumption is:

- features are conditionally independent given the class

This assumption is often unrealistic, but the method still works surprisingly well in many applications.

The lecture gives the posterior form:

`P(C_k | X) = P(X | C_k) P(C_k) / P(X)`

And under the independence assumption:

`P(C_k | X) proportional to P(C_k) product_i P(x_i | C_k)`

So Naive Bayes combines:

- prior class probability
- per-feature likelihoods

and predicts the class with the highest posterior probability.

## 22. Types of Naive Bayes

The lecture lists:

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

### Gaussian

Used when continuous features are modeled as approximately normal inside each class.

### Multinomial

Used for count data, especially:

- text classification
- bag-of-words models

### Bernoulli

Used for binary presence/absence features.

This is a good reminder that "Naive Bayes" is a family, not a single one-size-fits-all classifier.

## 23. Evaluation: Why Accuracy Is Not Enough

The lecture spends a lot of time on evaluation, and that is appropriate.

A classifier can look good under one metric and bad under another.

This is especially true when:

- classes are imbalanced
- false positives and false negatives have different costs
- probabilities matter, not only labels

## 24. Confusion Matrix

The confusion matrix is the basic diagnostic tool.

It contains:

- TP: true positives
- FP: false positives
- TN: true negatives
- FN: false negatives

From this matrix, many other metrics are derived.

The lecture also gives:

- false positive rate: `FP / (FP + TN)`
- false negative rate: `FN / (FN + TP)`

These rates matter because they capture different error types.

## 25. Basic Metrics

### 25.1 Accuracy

`(TP + TN) / (TP + FP + FN + TN)`

Accuracy measures the proportion of correct predictions overall.

Main limitation:

- it can be highly misleading on imbalanced datasets

If the positive class is rare, a model can achieve high accuracy by mostly predicting the majority class.

### 25.2 Precision

`TP / (TP + FP)`

Precision answers:

- when the model predicts positive, how often is it right?

This is important when false positives are costly.

### 25.3 Recall

`TP / (TP + FN)`

Recall answers:

- among all actual positives, how many did the model catch?

This matters when missing positives is costly.

### 25.4 F-beta Score

The lecture writes the general `F_beta` formula.

This is a weighted harmonic mean of precision and recall.

Important intuition:

- `beta = 1` gives `F1`, balancing precision and recall equally
- `beta > 1` gives more weight to recall
- `beta < 1` gives more weight to precision

This is very useful when application costs are asymmetric.

## 26. ROC, AUC, and Probability-Based Evaluation

The lecture introduces ROC and AUC as advanced metrics.

### 26.1 ROC Curve

ROC plots:

- true positive rate
- against false positive rate

across many thresholds.

Why it matters:

- it shows threshold behavior, not just one fixed threshold result

### 26.2 AUC

Area Under the ROC Curve summarizes ranking quality.

Interpretation:

- closer to `1` is better
- around `0.5` means near-random ranking

The lecture also mentions the **Gini coefficient**:

`Gini = 2 * AUC - 1`

This is common in some business scoring contexts.

## 27. Log Loss and Probability Density Plots

The slides also include:

- log loss
- class probability density plots

### 27.1 Log Loss

Log loss evaluates the predicted probabilities directly.

It penalizes:

- overconfident wrong predictions very heavily

This makes it a strong metric when probability calibration matters.

### 27.2 Class Score Distribution

Probability-density or score-distribution plots show how well the model separates classes in terms of predicted confidence.

If the positive and negative score distributions overlap strongly:

- threshold selection becomes harder
- the classifier is less confident and less separable

## 28. Practical Notebooks and What They Add

### 28.1 `Classification - prep.ipynb`

This is a visual helper notebook.

It contains synthetic examples for:

- distance calculations
- decision boundaries
- logistic regression intuition
- SVM intuition

It is mainly there to build geometric intuition rather than to serve as a full production workflow.

### 28.2 `Classification-1.ipynb`

Focus:

- KNN classifier

Dataset:

- Breast Cancer Wisconsin dataset

Key practical topics:

- train-test split with stratification
- feature scaling with `StandardScaler`
- default KNN
- grid search over `n_neighbors`
- cross-validation
- weighted KNN
- distance metric comparison
- confusion matrix
- classification report
- `predict_proba`
- ROC curve and AUC
- precision-recall curve
- average precision
- threshold selection with `F1`
- Youden's J

This notebook is stronger than the slides because it turns abstract evaluation ideas into concrete plots and threshold decisions.

### 28.3 `Classification-2.ipynb`

Focus:

- Decision Tree classifier

Key practical topics:

- default tree
- hyperparameter tuning
- cross-validation
- feature importance
- permutation importance
- tree visualization
- cost-complexity pruning
- pre-pruning with depth and minimum-sample constraints
- probability outputs
- ROC/PR evaluation
- threshold selection

This notebook is valuable because it shows both:

- how to tune the model
- how to explain it

### 28.4 `Classification-3.ipynb`

Focus:

- Logistic Regression classifier

Key practical topics:

- correlation inspection
- removing highly correlated features
- scaling
- binary logistic regression
- grid search for `C`, penalty, and solver
- `L1`, `L2`, and Elastic Net variants
- multinomial extension note
- ROC/PR evaluation
- average precision
- threshold tuning
- optional `statsmodels` logistic regression
- coefficient-based importance
- permutation importance
- p-value-based interpretation through `statsmodels`

This is especially useful because it links:

- ML-style logistic regression in `scikit-learn`
- statistical interpretation in `statsmodels`

## 29. Threshold Tuning

One of the best practical themes in the notebooks is that classification is not only about taking the default threshold `0.5`.

The notebooks tune thresholds using:

- `F1`
- Youden's J statistic
- balance between sensitivity and specificity

This is a very important professional habit.

Different applications require different tradeoffs:

- in medicine, recall may matter more
- in fraud filtering, precision may matter more
- in screening systems, threshold choice can be a business decision

## 30. Main Challenges Highlighted by the Lecture

The slides list several common challenges.

### 30.1 Class Imbalance

When one class dominates, many standard metrics become misleading.

Typical solutions mentioned:

- resampling
- SMOTE
- weighted loss functions

### 30.2 Multiclass and Multilabel Complexity

More labels mean:

- more complex decision structure
- more complex evaluation
- more ways to make partial mistakes

### 30.3 Ordinal Classes

Ordered labels require methods that respect order without pretending the spacing is numeric in a simple linear sense.

### 30.4 High-Dimensional Data

Too many features can:

- slow down learning
- worsen overfitting
- hurt distance-based methods

The lecture suggests dimensionality reduction as one solution.

### 30.5 Interpretability

More complex models can be more accurate, but harder to explain.

That is why model choice is often a tradeoff between:

- predictive power
- transparency

## 31. Key Takeaways

- Classification predicts labels rather than continuous values.
- Different task types include binary, multiclass, multilabel, and ordinal classification.
- KNN is simple and intuitive but sensitive to scaling, noise, and dimensionality.
- Decision trees are interpretable and flexible but prone to overfitting.
- Logistic regression is a linear probabilistic classifier with highly interpretable coefficients.
- SVC maximizes the margin and can handle non-linear boundaries through kernels.
- Naive Bayes is fast and probabilistic, but relies on strong independence assumptions.
- Accuracy alone is rarely enough for serious evaluation.
- Precision, recall, `F_beta`, ROC-AUC, PR curves, and log loss each capture different aspects of performance.
- Probability estimation and threshold tuning are central parts of practical classification.

## 32. Quick Revision Questions

1. What is the difference between binary, multiclass, multilabel, and ordinal classification?
2. Why is feature scaling especially important for KNN and often useful for logistic regression?
3. What does changing `k` do in KNN?
4. Why are decision-tree boundaries axis-aligned?
5. What is the difference between Gini impurity and entropy?
6. Why does logistic regression output probabilities instead of raw class labels directly?
7. What is the practical meaning of a positive logistic-regression coefficient?
8. Why can ROC-AUC and accuracy tell different stories about the same classifier?
9. When is precision more important than recall, and when is recall more important than precision?
10. Why can the default threshold `0.5` be a poor choice in real applications?
