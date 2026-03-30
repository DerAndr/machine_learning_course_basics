# Lecture 07 Recap: Ensemble Methods

> Lecture number: 07
> Lecture slug: `lecture_07_ensembles`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. Why Ensemble Methods Matter

Ensemble methods combine multiple models instead of relying on a single predictor.

The key idea of the lecture is simple:

- one model can be wrong in a specific way
- many models can make different errors
- combining them can improve robustness and accuracy

The lecture connects this idea to the classic "wisdom of the crowd" story:

- many individual estimates may be noisy
- the aggregate estimate can still be very good

In machine learning, this works when the individual models contribute:

- diversity
- complementary strengths
- partially uncorrelated errors

This is why ensembles are so widely used in:

- industry
- tabular ML
- Kaggle-style competitions
- noisy real-world datasets

## 2. Ensembles and the Bias-Variance Tradeoff

The lecture starts from the bias-variance framework.

Recall:

- **bias** is error from overly simple assumptions
- **variance** is error from excessive sensitivity to training data

A good model balances both.

The slides make the ensemble connection explicit:

- **Bagging** mainly reduces variance
- **Boosting** mainly reduces bias
- **Stacking** tries to leverage different models to improve both

This is the right high-level way to think about ensembles.

## 3. Revisiting Bias and Variance

The slides repeat several practical tools for controlling the tradeoff:

- feature selection and dimensionality reduction reduce variance
- more data reduces variance
- regularization reduces variance while increasing bias
- adding features can reduce bias but may increase variance
- deeper trees reduce bias but can increase variance

This is important because ensemble methods are not isolated tricks. They are part of a larger model-design strategy.

## 4. Three Main Ensemble Families

The lecture organizes the topic into:

- stacking
- bagging
- boosting

These are not small variations of one idea. They differ in:

- how models are trained
- whether base learners are the same or different
- whether learners are independent or sequential
- how final predictions are combined

## 5. Stacking

Stacking combines predictions from multiple models by training a **meta-model** on top of them.

The lecture describes the process as:

1. train several different base models on the same data
2. collect their predictions
3. use those predictions as inputs to a meta-model
4. let the meta-model produce the final prediction

This is a strong idea because different models often capture different structures:

- linear models capture simple global trends
- trees capture thresholds and interactions
- distance-based models capture local geometry
- margin-based models can capture clean separating boundaries

The meta-model tries to learn when each base model should be trusted more.

### 5.1 Why Stacking Can Work

Stacking can reduce both bias and variance when:

- base models are diverse
- their errors are not identical
- the meta-model is trained carefully

### 5.2 Practical Risk in Stacking

The lecture hints at complexity, but students should make one more connection:

- stacking is powerful
- but very easy to leak if meta-features are generated incorrectly

In proper stacking, the meta-model should ideally see **out-of-fold predictions** from base models, not predictions made on the same data those base models were trained on.

Otherwise, the meta-model learns from over-optimistic base outputs.

## 6. Bagging

Bagging stands for **Bootstrap Aggregating**.

The lecture describes it as:

- create multiple subsets from the training data
- train the same type of model on each subset
- aggregate the predictions

Typical aggregation:

- average for regression
- majority vote for classification

The main goal is variance reduction.

If each base learner has variance \(\sigma^2\) and the average pairwise correlation between learners is \(\rho\), then the variance of the ensemble average behaves roughly like:

\[
\mathrm{Var}(\bar{f}) \approx \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2
\]

where \(M\) is the number of learners.

This is the technical reason decorrelation matters so much in ensemble design. If all learners make almost the same errors, averaging helps much less.

### 6.1 Why Bagging Reduces Variance

High-variance models, especially decision trees, can change a lot if the training sample changes slightly.

If we train many such models on bootstrap samples and average them:

- individual fluctuations cancel out
- predictions become more stable
- overfitting risk drops

This is why bagging works especially well with:

- deep trees
- unstable learners

### 6.2 Bootstrap Sampling

Bootstrap means:

- sample with replacement from the training data

As a result:

- some observations appear multiple times in one bootstrap sample
- some observations are left out

This creates diversity among learners even when the model type stays the same.

## 7. Random Forest as the Canonical Bagging Model

The lecture uses Random Forest as the main bagging example.

Random Forest adds two sources of randomness:

- bootstrap sampling of rows
- random subset selection of features at each split

This matters because plain bagged trees can still be too correlated.

Random feature subsetting reduces correlation between trees, which makes averaging more effective.

That is one of the reasons Random Forest is so strong out of the box.

Random Forest also gives a useful built-in diagnostic through out-of-bag (OOB) estimation. Because each tree is trained on a bootstrap sample, some rows are left out for that tree and can be used to estimate performance without a separate validation set. OOB is not a replacement for every evaluation protocol, but it is a valuable practical signal.

## 8. Boosting

Boosting trains models **sequentially**, not independently.

The lecture says:

- each subsequent model tries to reduce the error of the previous one

This is the core idea.

Where bagging averages many independent unstable models, boosting builds a sequence of weak learners that gradually improve the overall predictor.

### 8.1 Why Boosting Reduces Bias

Each learner focuses on what earlier learners handled poorly.

As the sequence grows:

- the ensemble can approximate more complex functions
- underfitting is reduced
- weak learners combine into a strong model

### 8.2 Weighted Combination

The slides show final prediction as a weighted sum of learner outputs.

This is important:

- in boosting, not all learners contribute equally
- later or more effective learners can receive different weights

## 9. AdaBoost vs Gradient Boosting Intuition

The slides mention both AdaBoost and Gradient Boosting.

The practical interpretation students should keep in mind is:

- **AdaBoost** emphasizes misclassified observations by increasing their weight
- **Gradient Boosting** fits new learners to residual errors or negative gradients of the loss

This difference matters because modern boosting libraries such as XGBoost, LightGBM, and CatBoost are closer in spirit to advanced gradient boosting systems than to basic AdaBoost.

## 10. Comparing Stacking, Bagging, and Boosting

The lecture includes a useful comparison table. The main distinctions are worth memorizing.

### Stacking

- combines different model types
- uses a meta-model
- aims to exploit diversity
- can help with both bias and variance

### Bagging

- combines many instances of the same model family
- learners are independent
- aggregation is via averaging or voting
- mainly reduces variance

### Boosting

- combines sequential weak learners
- later models depend on earlier errors
- final prediction is weighted
- mainly reduces bias

### Practical Tradeoff

- stacking is flexible but more complex
- bagging is robust and often easy to use
- boosting is often highly accurate but sensitive to tuning and overfitting

## 11. Strengths and Weaknesses of Each Family

The notes PDF expands the comparison well.

### Stacking

Strengths:

- high flexibility
- leverages diverse model strengths

Weaknesses:

- computationally expensive
- can overfit if the meta-model is too strong
- requires careful validation design

### Bagging

Strengths:

- reduces overfitting in unstable learners
- robust on noisy data

Weaknesses:

- does not reduce bias much
- often needs enough data to create useful bootstrap diversity

### Boosting

Strengths:

- high predictive accuracy
- strong on complex tabular patterns

Weaknesses:

- can overfit without proper control
- training is sequential, so it is slower than bagging
- tuning matters more

## 12. Random Forest: Key Parameters

The lecture gives a practical tuning table for Random Forest.

Students should understand what each major parameter controls.

### `n_estimators`

Number of trees.

Effect:

- more trees usually improve stability
- but increase compute time

### `max_depth`

Maximum tree depth.

Effect:

- deeper trees lower bias
- but raise variance

### `min_samples_split`

Minimum samples required to split a node.

Effect:

- higher values make the model more conservative

### `min_samples_leaf`

Minimum samples required at a leaf.

Effect:

- larger values smooth predictions
- help reduce overfitting

### `max_features`

Number of features considered at each split.

Effect:

- lower values decorrelate trees more
- usually reduce variance
- may add some bias

### `bootstrap`

Whether bootstrap sampling is used.

Usually:

- `True` for standard Random Forest behavior

### `class_weight`

Important for imbalanced classification problems.

This is a useful reminder that ensemble tuning is not only about depth and number of trees. Class imbalance can also be addressed inside the model.

## 13. Suggested Random Forest Tuning Order

The lecture suggests a sensible sequence:

1. `n_estimators`
2. `max_depth`
3. `min_samples_split` and `min_samples_leaf`
4. `max_features`
5. `bootstrap` and `class_weight`

This is a practical engineering pattern:

- start with capacity
- then control complexity
- then refine randomness and class handling

## 14. XGBoost: Core Tuning Parameters

The lecture also gives a solid table for XGBoost.

Important parameters:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `gamma`
- `scale_pos_weight`

### 14.1 `learning_rate` and `n_estimators`

This pair is one of the most important boosting tradeoffs.

General rule:

- smaller `learning_rate` needs more boosting rounds
- larger `learning_rate` needs fewer rounds

Why:

- low learning rate makes each learner contribute more cautiously
- this often improves generalization if enough rounds are used

In additive notation, boosting updates the model as

\[
F_m(x) = F_{m-1}(x) + \eta h_m(x)
\]

where \(h_m(x)\) is the new weak learner and \(\eta\) is the learning rate. This is why `learning_rate` directly controls how aggressively each new tree changes the ensemble.

### 14.2 `max_depth`

Controls tree complexity.

Deeper trees:

- capture richer interactions
- but can overfit

### 14.3 `min_child_weight`

Makes the model more conservative when increased.

This can help prevent small noisy patterns from being turned into splits.

### 14.4 `subsample` and `colsample_bytree`

These introduce randomness in rows and columns.

Why useful:

- reduce overfitting
- improve robustness
- reduce correlation between boosting steps

Controlled randomness in boosting plays a role similar to decorrelation in bagging: it helps prevent the ensemble from fitting the same noisy structure too confidently.

### 14.5 `gamma`

Minimum loss reduction needed for a split.

Higher values:

- make splitting harder
- regularize the model

Conceptually, `gamma` says that not every small local improvement deserves a new branch. That makes the tree less willing to chase weak or noisy structure in the training data.

### 14.6 `scale_pos_weight`

Used for class imbalance.

The lecture suggests setting it according to the class ratio, which is a standard and useful practical heuristic.

## 15. CatBoost: Why It Matters

The lecture treats CatBoost as a distinct modern boosting system, and that is justified.

Its main practical advantage is:

- native handling of categorical features

This can remove a lot of preprocessing burden on tabular problems.

Important CatBoost parameters mentioned:

- `iterations`
- `learning_rate`
- `depth`
- `l2_leaf_reg`
- `border_count`
- `bagging_temperature`
- `scale_pos_weight`
- `one_hot_max_size`

## 16. CatBoost Tuning Intuition

### `iterations` and `learning_rate`

Same broad tradeoff as other boosting systems:

- lower learning rate needs more rounds

### `depth`

Controls tree complexity.

### `l2_leaf_reg`

Acts as regularization to reduce overfitting.

### `border_count`

Controls numeric-feature binning.

This is a more implementation-specific parameter but useful in practice when numeric features are discretized internally.

### `bagging_temperature`

Controls randomness in row sampling.

### `one_hot_max_size`

Controls which categorical features are one-hot encoded internally.

## 17. Model Comparison: Random Forest vs XGBoost vs CatBoost

The lecture includes two comparison tables that are actually very useful.

### Random Forest

Best for:

- simpler tabular problems
- robust baseline
- less tuning pressure

Strength:

- strong out-of-the-box stability

Weakness:

- may not match tuned boosting performance on harder structured datasets

### XGBoost

Best for:

- complex tabular data
- highly optimized predictive tasks

Strengths:

- strong accuracy
- many control knobs
- good support for imbalance and regularization

Weakness:

- high tuning complexity
- categorical features need preprocessing

### CatBoost

Best for:

- tabular data with categorical features
- situations where minimal preprocessing is desired

Strengths:

- strong defaults
- native categorical handling
- good large-scale performance

Weakness:

- still requires compute and tuning for the best results

## 18. Tips and Tricks from the Lecture

The final slides give practical advice worth preserving.

### Use Diverse Models

This is especially important for:

- stacking
- voting systems

Diversity matters because identical models tend to make similar errors.

### Use Cross-Validation

This is essential for:

- parameter tuning
- overfitting detection
- robust comparison across ensemble types

### Monitor Boosting Iterations

Boosting can overfit if the number of rounds becomes too large.

So:

- start smaller
- validate carefully
- do not assume more rounds are always better

### Tune Gradually

The lecture recommends adjusting one key parameter at a time.

This is good advice because ensemble models already have many interacting hyperparameters.

### Use Feature Importance Carefully

The lecture suggests using importance scores to identify weak features.

This is helpful, but students should remember:

- importance is model-specific
- different importance definitions behave differently
- importance is not the same as causality

## 19. Practical Notebook 1: `Ensembles.ipynb`

This notebook goes beyond the slide deck and gives a modern tabular-ML workflow.

Dataset:

- Telco Customer Churn

Models covered:

- Random Forest
- XGBoost
- CatBoost
- LightGBM
- Stacking Classifier

Key practical steps:

- identify categorical and numeric columns
- encode and scale as needed
- compare baseline and tuned models
- evaluate with classification metrics and probability-based curves
- run cross-validation
- compare base learners against the stack

This is valuable because it shows that ensemble work in practice is not only theory about bagging and boosting. It is usually a model-comparison and engineering workflow.

## 20. Practical Notebook 2: `Ensembles_part2 - Full.ipynb`

This notebook is more advanced and more "competition-style".

Dataset:

- credit score prediction

Key topics:

- preprocessing pipeline
- UMAP visualization
- CatBoost tuning with Optuna
- XGBoost baseline
- stacking model
- micro vs macro AUC explanation
- interpretability
- permutation importance for the stack

This notebook extends the lecture in a useful direction:

- modern ensembles are rarely used in isolation
- they are often embedded in a broader pipeline including tuning, dimensionality reduction, and interpretation

## 21. Student Notebook Variant

`Ensembles_part2 -STUDENTS.ipynb` appears to be the guided student-facing version of the advanced practice.

It keeps the same broad structure as the full version but leaves more room for student work.

This is helpful because students can see:

- the target workflow
- but still implement parts themselves

## 22. Interpreting Ensemble Outputs

A practical theme that runs through the notebooks is interpretability.

For ensemble methods, interpretation is usually harder than for plain linear models or shallow trees.

Common tools seen or implied here:

- feature importance
- permutation importance
- model comparison via cross-validation

Students should understand:

- ensembles can be highly predictive
- but the price is often weaker direct interpretability

That is why interpretation methods become essential.

## 23. Micro vs Macro AUC in the Advanced Practice

The advanced notebook explicitly explains micro and macro AUC for multiclass settings.

This is a useful extension beyond the core lecture.

### Macro-AUC

- compute AUC for each class in a one-vs-rest manner
- average them equally

Best when:

- each class matters equally

### Micro-AUC

- aggregate predictions across all classes
- weight according to instance counts

Best when:

- overall discrimination is the focus

This connects well to earlier lectures on multiclass evaluation.

## 24. Key Takeaways

- Ensemble methods combine multiple predictors to improve robustness and accuracy.
- Bagging mainly reduces variance.
- Boosting mainly reduces bias.
- Stacking combines diverse models and can improve both bias and variance.
- Random Forest is a strong bagging-based baseline for tabular data.
- Modern boosting systems such as XGBoost and CatBoost are highly powerful but require thoughtful tuning.
- CatBoost is especially attractive for categorical tabular data.
- Cross-validation is essential when tuning ensembles.
- Feature importance and permutation importance help interpretation, but should be used carefully.
- In practice, ensembles are often part of a larger workflow including preprocessing, tuning, and validation.

## 25. Quick Revision Questions

1. Why does bagging reduce variance rather than bias in most cases?
2. Why does boosting often improve weak learners so effectively?
3. What makes stacking different from both bagging and boosting?
4. Why does Random Forest use both row sampling and feature sampling?
5. What is the main tradeoff between `learning_rate` and `n_estimators` in boosting?
6. When would CatBoost be preferable to XGBoost?
7. Why can boosting overfit if not tuned carefully?
8. Why is model diversity important in stacking?
9. What is the difference between feature importance and permutation importance?
10. Why are ensemble methods so strong on real-world tabular datasets?
