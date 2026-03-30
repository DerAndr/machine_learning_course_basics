# Lecture 11 Notes: Cross-Validation, Hyperparameter Tuning, Pipelines, and AutoML

> Lecture number: 11
> Lecture slug: `lecture_11_cross_validation_hpo`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What This Lecture Is About

This lecture is about how to build machine learning workflows that are not only accurate on the training data, but also reliable on unseen data.

A model is not useful if it only looks good on the sample used to train it. That is why this lecture focuses on:

- cross-validation,
- hyperparameter optimization,
- pipelines,
- AutoML.

These topics are closely connected. Cross-validation estimates generalization. Hyperparameter search improves model settings. Pipelines make preprocessing and modeling consistent. AutoML automates part of that process.

Together, these tools help turn ad hoc experimentation into a reproducible modeling workflow.

## 2. Where This Fits in CRISP-DM

In CRISP-DM, this lecture sits mainly in the modeling and evaluation stages.

This is where we ask:

- which model family should we use,
- which hyperparameters should we choose,
- whether the observed score is trustworthy,
- whether preprocessing and model fitting are applied correctly,
- whether the whole workflow can be rerun consistently.

This part is critical because many modeling mistakes do not come from a bad algorithm. They come from:

- bad validation design,
- data leakage,
- tuning on the wrong split,
- comparing models unfairly,
- reporting optimistic metrics.

## 3. Why Validation and Tuning Matter

The lecture starts with three main reasons.

### 1. Overfitting

Without proper validation, a model may memorize training data instead of learning patterns that generalize.

This is especially dangerous when:

- the dataset is small,
- the model is flexible,
- the feature space is large,
- the hyperparameter search is wide.

### 2. Generalization

The real objective of supervised learning is not low training error. It is good performance on new data.

Validation strategies are therefore not a formality. They are how we estimate whether the model has learned something stable.

### 3. Efficiency

A bad validation protocol wastes time and compute:

- we may tune the wrong model,
- compare models unfairly,
- select parameters that look good only by chance,
- or deploy a system that later fails.

## 4. Cross-Validation: The Core Idea

Cross-validation is a resampling strategy for estimating model performance more reliably than a single train/validation split.

Instead of training once and validating once, we repeat the process across multiple splits of the data.

If the dataset is split into \(K\) folds, then in K-Fold CV:

1. use \(K-1\) folds for training,
2. use the remaining fold for validation,
3. repeat until every fold has served as validation once,
4. average the scores.

The CV estimate is typically:

\[
\text{CV score} = \frac{1}{K} \sum_{i=1}^{K} s_i
\]

where \(s_i\) is the validation score on fold \(i\).

This gives a more stable estimate than one arbitrary split.

## 5. Why CV Is Better Than a Single Split

A single split can be misleading because the result depends too much on one partition of the data.

Cross-validation reduces that dependence:

- more observations are used for training across runs,
- more observations are used for validation across runs,
- performance estimates are averaged,
- variability becomes easier to detect.

However, CV is still an estimate, not the true future performance.

## 6. K-Fold Cross-Validation

K-Fold is the default strategy for many i.i.d. datasets.

Typical choices:

- \(K = 5\),
- \(K = 10\).

### Strengths

- simple and widely supported,
- efficient use of limited data,
- easy comparison across models,
- standard default in many libraries.

### Limitations

- assumes the data is exchangeable enough to be randomly partitioned,
- not suitable for time-ordered data,
- may produce unstable results if class imbalance is severe and folds are naive,
- still has variance depending on the split.

### Practical interpretation

If K-Fold mean accuracy is 0.87 but fold scores range from 0.70 to 0.94, that instability is informative. The mean alone is not enough. Students should always pay attention to score spread, not just the mean.

## 7. Stratified K-Fold

For classification, especially imbalanced classification, naive K-Fold can produce folds with distorted class proportions.

Stratified K-Fold tries to preserve class distribution in each fold.

If the overall class ratio is:

- class A: 50%
- class B: 30%
- class C: 20%

then each fold approximately preserves that ratio.

### Why this matters

If some folds contain too few positive examples, the validation metric becomes unstable or misleading.

Stratification makes the folds more comparable.

### Important note

Stratified K-Fold is natural for classification, but not directly for regression. Regression needs a different workaround, such as binning the target before stratification.

## 8. Leave-One-Out Cross-Validation (LOOCV)

LOOCV is the extreme case where:

- each validation fold contains exactly one sample,
- training uses all remaining samples.

If there are \(n\) samples, the model is trained \(n\) times.

### Advantages

- maximal training data usage in each run,
- conceptually simple,
- sometimes useful for very small datasets.

### Disadvantages

- computationally expensive,
- high variance in the validation estimate,
- often not worth the cost for larger datasets.

Students often think LOOCV must be best because it uses nearly all the data for training. In practice, that is not automatically true. It can be expensive and unstable.

## 9. Repeated K-Fold

Repeated K-Fold runs K-Fold multiple times with different random splits and averages the results.

This is useful when:

- the dataset is small,
- a single K-Fold partition may be unrepresentative,
- you want a lower-variance estimate.

### Trade-off

You gain robustness, but pay more computationally:

\[
\text{total model fits} = K \times R
\]

where \(R\) is the number of repeats.

## 10. Time Series Cross-Validation

Time series data cannot be validated like i.i.d. data because temporal order matters.

If future observations influence training for past validation points, you get leakage.

### Correct principle

Training must always occur on earlier time points, and validation must occur on later time points.

The lecture highlights two common patterns:

- expanding window,
- sliding window.

### Expanding window

The training set grows over time:

- train on the first chunk,
- validate on the next chunk,
- then extend training further forward.

This is useful when older data remains relevant.

### Sliding window

The training window moves forward with fixed or limited size.

This is useful when:

- concept drift exists,
- recent history matters more,
- old data may be obsolete.

### Why this matters

Time series CV is not just a technical variation. It enforces the same information constraints that the model will face in production.

## 11. Nested Cross-Validation

Nested CV is one of the most important ideas in this lecture.

### The problem it solves

If you tune hyperparameters and evaluate the tuned model on the same validation process, the estimate becomes optimistic.

Why?

Because model selection itself adapts to that validation signal.

### Structure

Nested CV uses two loops:

- outer loop: estimates final performance,
- inner loop: performs model or hyperparameter selection.

The inner loop chooses the best configuration using only the outer-training portion. The outer test fold remains untouched until the final evaluation for that split.

### Why nested CV matters

- it separates tuning from final evaluation,
- it reduces selection bias,
- it gives a fairer estimate for model comparison.

### Cost

It is expensive, because tuning is repeated inside each outer fold.

If the outer loop has \(K_{outer}\) folds and the inner loop has \(K_{inner}\) folds, then search cost can grow quickly:

\[
\text{fits} \approx K_{outer} \times K_{inner} \times \text{number of parameter settings}
\]

This is why nested CV is theoretically strong but operationally expensive.

## 12. Cross-Validation for Non-standard Cases

The lecture also covers cases where standard CV is not enough.

### Multiclass classification

Problem:

- class imbalance across multiple classes.

Recommended strategy:

- Stratified K-Fold.

This keeps class proportions more stable across folds and avoids missing minority classes in some validation splits.

If the data also contains natural groups, such as repeated measurements from the same patient, customer, device, or experiment, then group-aware splitting becomes important as well. Otherwise the model can see nearly duplicated information in both training and validation folds and produce overly optimistic scores.

### Multilabel classification

Problem:

- each sample can belong to multiple labels,
- standard stratification is not directly applicable.

The lecture mentions:

- iterative stratification,
- binary relevance style decomposition.

This is important because preserving label balance in multilabel settings is harder than in ordinary multiclass classification.

### Regression

Standard K-Fold is often used for i.i.d. regression, but the lecture notes that it may be sensitive to:

- outliers,
- skewed target distributions.

One workaround is target binning, followed by stratification on those bins.

Students should treat this as a practical heuristic, not a perfect theoretical solution.

The key caveat is that the bins are artificial. They help balance folds, but they do not remove the continuous nature of the regression problem.

## 13. Choosing Metrics Inside CV

Cross-validation is only as meaningful as the metric you evaluate.

The lecture reminds students to match metrics to the task:

- classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC depending on the problem,
- regression: MSE, RMSE, MAE, R-squared,
- multilabel: Hamming Loss, micro/macro F1, and related metrics.

### Key idea

You are not cross-validating “the model” in the abstract. You are cross-validating a model under a specific metric. Change the metric, and the ranking of hyperparameters may change too.

This matters a lot in:

- imbalanced classification,
- asymmetric business costs,
- noisy regression.

## 14. Hyperparameters vs. Learned Parameters

The lecture then shifts to hyperparameter optimization.

### Learned parameters

These are estimated from data during training:

- regression coefficients,
- tree split thresholds,
- neural network weights.

### Hyperparameters

These are configuration choices that govern the training process or model structure:

- tree depth,
- number of trees,
- regularization strength,
- learning rate,
- number of neighbors,
- kernel parameters,
- feature selection settings.

Hyperparameters are not learned automatically by ordinary training. They must be chosen externally.

## 15. Why Hyperparameter Tuning Matters

Good hyperparameters improve:

- bias-variance balance,
- generalization,
- optimization behavior,
- stability,
- sometimes training speed.

Bad hyperparameters can make a strong algorithm perform poorly.

For example:

- a tree that is too deep may overfit,
- a learning rate that is too high may destabilize boosting,
- too few trees may underfit,
- too much regularization may suppress useful signal.

## 16. Grid Search

Grid Search tries every combination in a predefined parameter grid.

If you specify:

- 3 values for depth,
- 4 values for `n_estimators`,
- 5 values for `max_features`,

then Grid Search evaluates:

\[
3 \times 4 \times 5 = 60
\]

parameter combinations.

### Strengths

- simple,
- deterministic,
- easy to explain,
- good when the search space is small and well chosen.

### Weaknesses

- computationally expensive,
- scales poorly with more parameters,
- wastes effort on dimensions that may matter little,
- may miss good regions if the grid is too coarse.

This is the “curse of dimensionality” of brute-force search.

## 17. Randomized Search

Randomized Search samples parameter combinations from specified distributions instead of enumerating every combination.

### Strengths

- often more efficient in large spaces,
- good when only a few hyperparameters matter strongly,
- lets you search broader ranges with limited compute.

### Weaknesses

- does not guarantee evaluation of any specific region,
- results depend on budget and random sampling,
- still requires reasonable search distributions.

In practice, Randomized Search is often a better first step than Grid Search for medium or large spaces.

It is especially attractive for scale-sensitive parameters such as regularization strengths or learning rates, where logarithmic search ranges are often more meaningful than evenly spaced linear grids.

## 18. Bayesian Optimization

The lecture introduces Bayesian optimization as a more sample-efficient search strategy.

### Core idea

Instead of blindly trying points, Bayesian optimization builds a surrogate model of the objective function:

- what performance is expected for a hyperparameter setting,
- where uncertainty remains high,
- which point should be evaluated next.

This makes the search adaptive.

### Why it can be powerful

- fewer evaluations may be needed,
- promising regions are explored more intelligently,
- useful when each training run is expensive.

### Trade-off

- more complex than Grid Search or Randomized Search,
- more tooling and tuning overhead,
- sometimes harder to interpret operationally.

Conceptually, Bayesian optimization balances exploration and exploitation:

- exploration asks where the surrogate model is still uncertain,
- exploitation asks where strong performance is already expected.

The lecture mentions libraries such as:

- Optuna,
- Hyperopt,
- scikit-optimize.

The hyperparameter demo notebook uses several of these in one place, which is helpful for comparing them directly.

## 19. Practical Tips for Tuning

The slides include a good set of pragmatic recommendations.

### Focus on the important hyperparameters

Not every hyperparameter matters equally. Often a small subset drives most of the performance variation.

Examples:

- tree depth,
- learning rate,
- regularization strength,
- number of estimators,
- feature subset size.

### Tune progressively

Start broad, then narrow the range.

This is usually better than starting with a huge fine-grained grid.

### Use proper validation

Hyperparameter search must be nested inside a valid resampling protocol. Otherwise tuning leaks information from validation into model selection.

### Keep an independent test set

Even after CV and tuning, a final untouched test set is still valuable. It is the last sanity check before deployment or reporting.

### Track experiments

Once many tuning runs are involved, experiment tracking becomes essential. The lecture mentions tools such as MLflow.

This is not only for convenience. It is part of reproducibility and auditability.

### Mind your budget

Hyperparameter search can become computationally expensive very quickly. Search strategy should reflect:

- model cost,
- hardware budget,
- deadline,
- dataset size.

## 20. Pipelines

Pipelines are one of the most important engineering topics in this lecture.

### What a pipeline is

A machine learning pipeline is a structured sequence of steps such as:

1. preprocessing,
2. feature transformation,
3. feature selection,
4. model fitting,
5. evaluation or deployment.

In scikit-learn, a `Pipeline` object combines steps into one executable unit.

### Why pipelines matter

The slides emphasize:

- reproducibility,
- automation,
- modularity,
- scalability,
- consistency.

All five are important, but in day-to-day ML work the biggest win is often consistency.

### Consistency between train and inference

If scaling, imputation, encoding, or feature selection is done manually, it is easy to apply slightly different logic to:

- training data,
- validation data,
- test data,
- production data.

Pipelines reduce this risk by applying the exact same transformation sequence.

## 21. Pipelines and Leakage

This lecture should make students very careful about leakage.

### Common leakage mistake

Suppose you standardize the full dataset before CV. Then information from the validation fold influences the mean and standard deviation used for scaling.

This is leakage.

The same issue occurs with:

- imputation,
- target encoding,
- feature selection,
- dimensionality reduction,
- outlier filtering,
- any transformation fitted before the split.

### Correct principle

Every transformation that learns from data must be fit only on the training portion of each fold.

This is exactly why pipelines are so important: when combined with CV tools, they ensure that each fold recomputes preprocessing correctly inside the training split.

Canonical leakage mistakes include:

- scaling the full dataset before cross-validation,
- selecting features on the full dataset before splitting,
- imputing values using statistics computed on validation rows,
- target encoding with access to all labels.

### Parameter naming in pipelines

When tuning a pipeline in scikit-learn, hyperparameters are referenced with the step name:

- `clf__max_depth`
- `select__k`
- `regr__n_estimators`

This matters because pipeline tuning is not just model tuning. It can also include preprocessing and feature selection choices.

## 22. AutoML

AutoML stands for Automated Machine Learning.

The main idea is to automate part of the ML workflow, often including:

- preprocessing,
- model selection,
- hyperparameter tuning,
- sometimes feature engineering,
- sometimes ensembling.

### Why AutoML is attractive

- saves time,
- gives strong baselines quickly,
- useful for non-experts,
- explores more configurations systematically.

### Why AutoML should still be used carefully

- it can be computationally heavy,
- resulting pipelines may be harder to interpret,
- it does not remove the need for problem understanding,
- it can automate bad validation if used carelessly,
- model governance still matters.

The lecture mentions:

- H2O AutoML,
- auto-sklearn,
- TPOT,
- MLJAR AutoML,
- LightAutoML,
- cloud offerings such as SageMaker Autopilot and Azure AutoML.

Students should understand that AutoML is not magic. It is a search and orchestration layer built on top of familiar modeling components.

## 23. Best Practices from the Lecture

The best-practice slides summarize several important habits:

### Keep an independent test set

Use CV and tuning on the training side, but preserve a final untouched set for realistic performance confirmation.

### Track experiments

Use tools such as MLflow or similar systems for:

- parameters,
- metrics,
- artifacts,
- run comparison.

### Respect compute constraints

Search strategy is part of engineering design. A theoretically attractive method may be inappropriate if the budget is too small.

### Maintain interpretability awareness

Complex pipelines and AutoML outputs may perform well, but they can become harder to explain and maintain.

## 24. Practical Notebook Map

This lecture has three main practical notebooks, each emphasizing a different part of the workflow.

### 1. `crossval.ipynb`

This notebook is more conceptual and tool-oriented. It includes:

- SVG illustrations of CV strategies,
- a visual comparison of Grid Search vs Random Search,
- MLflow demo setup,
- examples of AutoML-style tooling such as H2O.

This notebook is useful for building intuition around workflow structure, rather than for one single end-to-end model.

### 2. `Hyperparameter_search.ipynb`

This is the direct tuning notebook. It uses a Random Forest Regressor on the Diabetes dataset and compares:

- Grid Search,
- Randomized Search,
- Hyperopt,
- Optuna,
- Bayesian search via scikit-optimize.

This notebook is important because it moves beyond theory and lets students compare optimization strategies on the same predictive task.

It reinforces the idea that:

- different search methods explore the space differently,
- search budget matters,
- “best method” depends on cost, search space, and practical constraints.

### 3. `CrossValidation_and_pipelines.ipynb`

This is the most workflow-oriented notebook. It demonstrates:

- classification on the Wine dataset,
- regression on the California Housing dataset,
- train/test splitting,
- cross-validation,
- pipelines,
- feature selection,
- GridSearchCV,
- H2O AutoML.

This notebook is especially valuable because it shows how cross-validation, preprocessing, tuning, and model execution fit together in one reproducible structure.

## 25. What Students Should Understand Technically

After this lecture, students should be able to explain the following.

### Why is cross-validation necessary?

Because one split can be misleading, and CV gives a more stable estimate of generalization.

### When should Stratified K-Fold be preferred over K-Fold?

When classification classes are imbalanced and you want folds with representative class proportions.

### Why can standard CV not be used for time series?

Because random splitting breaks temporal order and leaks future information into training.

### Why is nested CV more honest than ordinary tuning plus CV reporting?

Because it separates hyperparameter selection from final performance estimation.

### What is the difference between Grid Search and Randomized Search?

Grid Search tests all specified combinations. Randomized Search samples combinations from distributions.

### Why are pipelines important?

Because they make preprocessing and modeling reproducible and prevent leakage by fitting transformations inside each training fold.

### What is the risk of tuning without a proper validation protocol?

You can overfit to the validation process itself and report an overly optimistic score.

### What is AutoML best used for?

For strong baselines, workflow acceleration, and systematic search, not as a replacement for understanding the problem.

## 26. Key Takeaways

- Cross-validation estimates generalization more reliably than a single split.
- The correct CV strategy depends on the data structure: i.i.d., imbalanced, temporal, grouped, multilabel, or regression.
- Hyperparameter tuning must be integrated with proper validation.
- Pipelines are essential for reproducibility and leakage prevention.
- AutoML can accelerate experimentation, but it does not remove the need for careful evaluation and interpretation.

## 27. Quick Revision Questions

1. Why is a high CV score not automatically trustworthy if preprocessing was done before the split?
2. Why might Randomized Search beat Grid Search in a large hyperparameter space with limited budget?
3. What problem does nested CV solve that ordinary cross-validation does not?
4. Why is Time Series CV fundamentally different from standard K-Fold?
5. Why should an AutoML result still be checked with independent validation and human judgment?
