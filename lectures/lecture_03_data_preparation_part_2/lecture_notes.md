# Lecture 03 Recap: Data Preparation Part 2

> Lecture number: 03
> Lecture slug: `lecture_03_data_preparation_part_2`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. Why This Lecture Goes Beyond Basic Cleaning

The previous data preparation lecture focused on:

- missing values
- duplicates
- outliers
- scaling
- encoding

This lecture moves to the next level.

The main message is:

- clean data is necessary
- but clean data alone is not enough

To build a strong model, we also need to decide:

- which features should be kept
- which new features should be created
- how to reduce dimensionality
- how to split the data correctly
- how to validate model quality without cheating
- how to automate preprocessing safely

In other words, this lecture is about making preprocessing more strategic.

## 2. Advanced Data Preparation in the ML Pipeline

The slides place this topic inside the broader sequence of:

1. data collection
2. data cleaning
3. transformation and encoding
4. feature selection and feature generation
5. dimensionality reduction
6. data splitting
7. cross-validation
8. data leakage prevention
9. building data pipelines

So this lecture is the bridge between simple preprocessing and robust model development.

It is also strongly connected to overfitting.

The lecture stresses three practical risks:

- too many features can add noise
- poor train-test splitting gives misleading evaluation
- weak preprocessing design can create leakage

That is why this topic matters so much in real machine learning.

## 3. Feature Selection

Feature selection means choosing the most useful input variables and removing those that are:

- irrelevant
- redundant
- noisy

The goal is not only to make the dataset smaller. The deeper goal is to improve generalization.

Too many features can cause:

- overfitting
- slower training
- harder interpretation
- unstable models
- multicollinearity in linear models

The lecture divides feature selection into three families:

- filter methods
- wrapper methods
- embedded methods

## 4. Filter Methods

Filter methods evaluate features using statistical criteria before model training or with minimal model dependence.

Their main strengths are:

- speed
- simplicity
- model-agnostic behavior

Their main weakness is that they often ignore interactions between features.

### 4.1 Correlation Matrix

The slides describe the correlation matrix as a tool for measuring linear relationships between numerical features.

Important facts:

- correlation coefficients range from `-1` to `1`
- values near `1` indicate strong positive linear relationship
- values near `-1` indicate strong negative linear relationship
- values near `0` indicate weak linear relationship

Why this matters for feature selection:

- if two features are almost duplicates of each other, keeping both may not add much information
- in linear models, high collinearity can make coefficients unstable

The notebook `Coding_3.1.ipynb` goes deeper than the slides and compares:

- Pearson correlation
- Spearman correlation
- Kendall correlation
- point-biserial correlation
- Cramer's V for categorical variables

This is a strong practical addition because it teaches that "correlation" is not one universal quantity.

#### Pearson Correlation

Best for:

- linear relationships between numerical variables

Weakness:

- misses non-linear relationships
- sensitive to outliers

#### Spearman Correlation

Best for:

- monotonic relationships

It uses ranks rather than raw values, so it is more robust when the relationship is not perfectly linear.

#### Kendall Correlation

Also measures ordinal or monotonic association, often with a stricter pairwise interpretation.

#### Point-Biserial Correlation

Useful when:

- one variable is numerical
- the target is binary

This makes it especially relevant for binary classification tasks.

#### Cramer's V

Useful when:

- both variables are categorical

This is important because plain Pearson correlation is not appropriate for nominal categorical variables.

### 4.2 Chi-Square Test

The slides describe the chi-square test as a way to evaluate whether a categorical feature is independent of the target.

Main idea:

- compare observed frequencies with expected frequencies under independence

If the difference is large enough, the feature and target are likely related.

The lecture emphasizes:

- low `p-value` suggests dependence
- high `p-value` means we fail to reject independence

Technical point:

- chi-square is based on contingency tables
- the expected count in each cell is computed from row totals, column totals, and grand total

This is useful for categorical feature screening in classification.

But students should remember the limitations:

- it does not measure effect size directly
- it can be sensitive to sample size
- it assumes counts rather than arbitrary numeric values

In `Coding_3.1.ipynb`, chi-square is implemented directly and also through `SelectKBest(chi2)`.

### 4.3 Weight of Evidence (WoE) and Information Value (IV)

The lecture introduces WoE as a way to encode categories based on how strongly they separate target classes.

The slide formula captures the idea:

- compare the proportion of one target class and the other within each category
- take the logarithm of their ratio

Interpretation:

- positive WoE suggests stronger association with one class
- negative WoE suggests stronger association with the other class

Information Value aggregates this separation across bins or categories and gives a rough measure of predictive usefulness.

This method is common in:

- credit scoring
- risk modeling

The notebook implements WoE and IV manually, which is useful because it helps students see that these are not mysterious library features. They are based on class proportions.

### 4.4 Mutual Information and ANOVA

The notebook adds two more useful filter-style tools:

- mutual information
- ANOVA F-statistic

#### Mutual Information

Mutual information measures how much knowing a feature reduces uncertainty about the target.

Why it matters:

- it can capture non-linear relationships
- it is often more flexible than simple correlation

#### ANOVA F-statistic

ANOVA asks whether a numerical feature has different means across target groups.

High F-statistic and low `p-value` suggest the feature helps distinguish the classes.

This is a good reminder that feature screening can be approached from different statistical angles, not only correlation.

## 5. Wrapper Methods

Wrapper methods search for feature subsets by repeatedly training a model and evaluating performance.

Compared with filter methods, they are:

- usually more accurate
- more sensitive to feature interactions
- much more expensive computationally

The lecture lists:

- forward selection
- backward elimination
- recursive feature elimination
- exhaustive feature search

### 5.1 Forward Selection

Start with no features and add them one by one.

At each step:

- try candidate additions
- keep the feature that improves model quality the most

Strength:

- intuitive
- often cheaper than testing all subsets

Weakness:

- greedy
- may miss combinations that only become useful together

### 5.2 Backward Elimination

Start with all features and remove the least useful ones step by step.

Strength:

- good when you begin with a manageable number of variables

Weakness:

- still expensive
- not practical for very large feature spaces

### 5.3 Recursive Feature Elimination (RFE)

RFE repeatedly fits a model and removes the weakest features.

This is a more systematic way to shrink the feature set.

It depends on having a model that can provide some importance or coefficient-based ranking.

### 5.4 Exhaustive Feature Search

This is the most brute-force method.

It tries many or all possible subsets and picks the best.

Main issue:

- combinatorial explosion

If the number of features is large, the number of subsets becomes enormous very quickly.

The notebook explicitly demonstrates this with combination counts. That is a useful practical warning: exhaustive search sounds attractive, but it often becomes unrealistic.

## 6. Embedded Methods

Embedded methods perform feature selection inside model training.

The lecture highlights:

- Lasso
- Ridge
- decision tree feature importance

These methods are attractive because they combine:

- model fitting
- regularization or importance scoring
- some form of automatic selection

### 6.1 Lasso (L1)

Lasso adds an absolute-value penalty to the loss function.

Main consequence:

- some coefficients can shrink exactly to zero

That makes Lasso useful for feature selection because features with zero coefficients are effectively removed from the model.

Why it is valuable:

- encourages sparse solutions
- useful when many features exist but only some are expected to matter

### 6.2 Ridge (L2)

Ridge adds a squared penalty to the loss function.

Main consequence:

- coefficients are shrunk toward zero
- but usually not all the way to zero

This means Ridge is more about stabilization than hard feature elimination.

It is useful when:

- many features contribute a little
- multicollinearity exists

### 6.3 Tree-Based Feature Importance

Decision trees can rank features according to how much they reduce impurity when splitting.

Why this is useful:

- can capture non-linear structure
- handles interactions better than simple filter methods

But students should remember:

- feature importance is model-specific
- different tree settings can change the ranking
- importance is not the same as causal meaning

## 7. When to Use Filter, Wrapper, or Embedded Methods

The slides compare these three families directly.

### Filter Methods

Use when:

- speed matters
- dataset is large
- you need a quick first screening

Main risk:

- may miss important feature interactions

### Wrapper Methods

Use when:

- model performance matters more than speed
- dataset is small or medium sized
- you can afford computation cost

Main risk:

- expensive
- can overfit when the sample is small

### Embedded Methods

Use when:

- you want feature selection integrated into training
- you already know which model family you will use

Main tradeoff:

- efficient
- but tied to a specific model class

## 8. Feature Generation

Feature generation means creating new variables from existing ones.

This is one of the most creative parts of machine learning.

The lecture lists several approaches:

- transforming existing features
- creating interaction features
- domain-specific features
- time-based features
- aggregation features
- binning and categorization

The key message is important:

- not every new feature is useful
- a good generated feature captures structure
- a bad generated feature only adds noise

This idea is demonstrated very clearly in `Coding_3.2.ipynb`, which shows both good and bad examples.

## 9. Transforming Existing Features

The notebook begins by transforming `Lot Area`.

### Good Example: Log Transformation

`Log_LotArea = log1p(Lot Area)`

Why it is useful:

- compresses large values
- reduces right skew
- often makes a feature easier for linear models to use

### Bad Example: Squaring Lot Area

`LotArea_Squared = Lot Area ** 2`

Why this can be bad:

- can exaggerate scale differences
- can worsen skewness
- may produce a feature that is mathematically valid but practically unhelpful

This is a very good lesson: feature engineering is not about producing more columns. It is about producing more meaningful signal.

## 10. Interaction Features

Interaction features combine variables so that the model can capture joint effects.

### Good Example: Lot Coverage Ratio

The notebook builds:

- `Gr Liv Area / Lot Area`

This reflects how much of the lot is covered by living area.

Why it can be meaningful:

- expresses land-use intensity
- encodes a relationship between house size and lot size

### Bad Example: Lot Area times Year Built

This multiplication is shown as a weak example because the product has no obvious interpretation.

That is a strong teaching point:

- an interaction should represent a plausible relationship
- otherwise it may just create noise

## 11. Domain-Specific Features

Domain-specific features use subject matter understanding.

### Good Example: Price per Lot Area

This feature describes how much price is paid per unit of land.

Why it helps:

- normalizes price by size
- creates a more interpretable economic quantity

### Bad Example: SalePrice per Month Sold

Dividing `SalePrice` by `Mo Sold` is mathematically possible, but it is not a meaningful real-world metric.

This is exactly the kind of mistake students should avoid:

- mathematically legal does not mean practically useful

## 12. Time-Based Features

The lecture highlights time-based feature engineering as another major family.

The notebook creates:

- `House_Age = Yr Sold - Year Built`

This is a strong example because raw year values are often less informative than elapsed time.

Why it works:

- age of a house is easier to interpret than a build year alone
- it often relates directly to condition, depreciation, and renovation patterns

The broader lesson:

- time fields often become more useful after converting them into durations, lags, recency measures, or seasonal components

## 13. Aggregation Features

Aggregation means summarizing information at the group level.

The notebook computes:

- average `Lot Frontage` by `Neighborhood`

This creates a contextual feature.

Why this matters:

- individual observations often depend on the group they belong to
- a neighborhood average can capture local structure that a single row does not show directly

This is an important step toward more relational thinking in feature engineering.

## 14. Binning and Categorization

The lecture includes binning again, but here it is presented as part of feature generation, not only preprocessing.

The notebook shows:

- meaningful bins for `Lot Area`
- arbitrary bins for `SalePrice`

The contrast is valuable.

### Good Binning

When bins reflect meaningful size ranges, the result may:

- improve interpretability
- support rule-based reasoning
- align better with domain language

### Bad Binning

When bins are arbitrary:

- useful detail is lost
- thresholds may be misleading
- categories may not match real structure in the data

## 15. Generating Features from Categorical Data

The notebook also shows that feature engineering is not only for numerical columns.

Examples include:

- one-hot encoding of `MS Zoning`
- label encoding of `Lot Shape`
- frequency encoding of `Neighborhood`
- combining `House Style` and `Overall Qual`

This is important because categorical features often gain power when:

- levels are combined
- frequencies are used
- interactions between categories and quality signals are represented

The notebook then checks how these new features relate to `SalePrice` and compares models before and after the new features are added.

This is exactly the right workflow:

1. design the feature
2. visualize or inspect it
3. test whether it helps the model

## 16. Benefits and Risks of Feature Generation

The lecture lists two major benefits:

- improved model performance
- reduced complexity when aggregation or transformation creates more useful structure

But it also warns about:

- overfitting
- high computational cost

Students should remember:

- every generated feature is a hypothesis
- more features can help
- but they can also make the model memorize noise

## 17. Dimensionality Reduction

The lecture then moves to the curse of dimensionality.

Main idea:

- too many features increase noise
- models become slower
- overfitting becomes more likely

Dimensionality reduction tries to keep the most important information while using fewer dimensions.

The slides distinguish two broad strategies:

- feature selection
- feature extraction

Feature selection keeps original variables.
Feature extraction creates new lower-dimensional representations.

## 18. PCA

Principal Component Analysis is presented as the main linear dimensionality reduction method.

The slides explain the logic:

- identify directions of maximum variance
- project the data onto those directions
- keep fewer components while preserving most information

Important properties:

- PCA creates new variables called principal components
- these components are linear combinations of the original features
- the components are uncorrelated

The slides also mention the covariance matrix and eigenvectors. Students do not need to derive PCA from scratch, but they should understand the intuition:

- PCA rotates the feature space
- the first component captures the largest variance
- the second component captures the next largest variance subject to being orthogonal to the first

The notebook `Coding_3.3.ipynb` uses a mobile device usage dataset, standardizes numeric features, and then applies:

- `PCA(n_components=0.9)`

This means the model keeps enough principal components to preserve about 90 percent of the variance.

This is a useful practical rule because it connects theory to a real parameter choice.

### PCA Strengths

- fast
- mathematically clean
- good preprocessing step
- useful before some models

### PCA Weaknesses

- only linear
- components can be hard to interpret
- variance preservation is not always the same as predictive usefulness

## 19. t-SNE

t-SNE is introduced as a non-linear visualization method.

Main idea:

- preserve local neighborhood structure
- place similar points close together in a low-dimensional space

Why students like it:

- it often reveals clusters beautifully in 2D or 3D

But the lecture correctly warns:

- it is computationally expensive
- it is sensitive to hyperparameters such as perplexity
- it is often non-deterministic
- it emphasizes local structure more than global geometry

So t-SNE is mainly a visualization tool, not a generic feature-reduction method for every downstream model.

## 20. UMAP

UMAP is another non-linear dimensionality reduction method.

The lecture presents it as:

- preserving both local and some global structure
- more efficient than t-SNE
- suitable for larger datasets

Why it matters:

- often faster than t-SNE
- popular for embeddings and exploratory visualization

Main caution:

- still parameter-sensitive
- still harder to interpret than PCA

## 21. Comparing PCA, t-SNE, and UMAP

The slides compare them directly.

### PCA

Best when:

- data is approximately linear
- speed matters
- dimensionality reduction is needed as preprocessing

### t-SNE

Best when:

- the goal is 2D or 3D visualization
- local neighborhoods matter

### UMAP

Best when:

- dataset is larger
- you want both visualization and compact embeddings
- you want a method more scalable than t-SNE

The key lesson is that dimensionality reduction is not one tool. The right technique depends on the objective.

## 22. Data Splitting

The lecture then shifts from feature engineering to evaluation design.

It defines three roles:

- training set
- validation set
- test set

Typical proportions in the slides:

- training: about 80%
- validation: about 10-15%
- test: about 5-10%

The exact split is flexible, but the principle is strict:

- train on one part
- tune on another part
- evaluate finally on untouched data

This is how we estimate generalization rather than memorization.

## 23. Splitting Strategies

The lecture lists several splitting methods.

### 23.1 Random Split

Best for:

- balanced i.i.d. data

The notebook `Coding_3.4.ipynb` contrasts:

- random split with fixed seed
- random split without fixed seed

This is useful because reproducibility matters. If the split changes every run, your evaluation changes too.

### 23.2 Stratified Split

Best for:

- imbalanced classification

Why:

- preserves class proportions across train and test

This matters a lot in fraud or bankruptcy datasets, where the minority class may be very rare.

Without stratification, one split can accidentally contain too few positive cases and make evaluation unstable or misleading.

### 23.3 Time-Based Split

Best for:

- time series
- any data with temporal order

Main rule:

- train on the past
- evaluate on the future

If future rows influence training, the evaluation is unrealistic.

The notebook shows both:

- a correct chronological split
- an incorrect random split for temporal data

### 23.4 Hierarchical or Group Split

The lecture mentions hierarchical data such as:

- users
- schools
- locations

The notebook gives a school/classroom example and uses group-aware splitting.

Why this matters:

- if the same group appears in both train and test, the model can exploit group-specific patterns
- the score may look much better than real deployment performance

## 24. Imbalanced Data and SMOTE

The lecture also connects splitting with class imbalance.

It mentions:

- oversampling
- undersampling
- stratified splitting
- SMOTE

SMOTE generates synthetic minority examples.

Important practical rule, which the notebook shows correctly:

- SMOTE must be applied only to the training set

Why:

- if synthetic balancing is done before the split, information leaks into validation or test data

That would inflate performance estimates.

## 25. Cross-Validation

Cross-validation repeats the train-validation process across several folds.

The slides highlight:

- K-Fold cross-validation
- Leave-One-Out cross-validation (LOOCV)

### 25.1 K-Fold

Process:

1. split data into `k` folds
2. train on `k - 1` folds
3. validate on the remaining fold
4. repeat until every fold has served as validation
5. average the results

Common values:

- `k = 5`
- `k = 10`

Why it is useful:

- gives a more stable estimate than one single split
- uses the dataset more efficiently

### 25.2 LOOCV

Leave-One-Out uses every observation as the validation set once.

Strength:

- almost no data is wasted

Weakness:

- expensive
- can have high variance in the estimate

### 25.3 Stratified K-Fold and TimeSeriesSplit

The notebook goes beyond the slide summary and includes:

- `StratifiedKFold`
- `TimeSeriesSplit`

This is a good reminder that cross-validation must match the data structure.

## 26. Bias-Variance Tradeoff

The lecture includes bias-variance because preprocessing decisions directly affect model complexity.

Definitions:

- **bias**: error from an overly simple model or wrong assumptions
- **variance**: error from a model that changes too much across samples

The classic formula shown in the slides is:

`MSE = Bias^2 + Variance + Irreducible Error`

This matters for data preparation because:

- adding features can reduce bias
- but it can also increase variance
- dimensionality reduction and regularization often reduce variance
- deeper trees may reduce bias but increase variance

The slides also connect this to:

- regularization
- larger training sets
- ensembles
- cross-validation

So bias-variance is not only a modeling topic. It is also a preprocessing design topic.

## 27. Data Leakage

This is one of the most important sections of the lecture.

Data leakage happens when the model uses information that would not really be available at prediction time.

The lecture gives two big categories:

- leakage from future data
- leakage from target-related variables

The central problem is always the same:

- evaluation becomes too optimistic
- deployment performance collapses

## 28. Types of Leakage

### 28.1 Train-Test Contamination

This happens when information from the test set is used during training or preprocessing.

Classic example from the slides:

- compute normalization statistics on the full dataset instead of training data only

Why this is wrong:

- the scaler "sees" the test distribution
- the model benefits indirectly from data it should not know

`Coding_3.5.ipynb` demonstrates exactly this with scaling and imputation.

### 28.2 Target Leakage

Target leakage happens when a feature contains information about the target that would not be available in practice.

Example from the slides:

- using medication prescribed after diagnosis to predict the diagnosis

The notebook gives a simpler version by incorrectly including `SalePrice` itself among the predictors when predicting `SalePrice`.

This creates a model that looks excellent on paper but is meaningless.

### 28.3 Time-Series Leakage

This happens when future information is used to predict the present.

In temporal problems, this is one of the most common mistakes.

The notebook demonstrates:

- a wrong setup using future information
- a correct setup where training precedes testing chronologically

### 28.4 Leakage from Derived Features

Feature engineering can leak information too.

The notebook shows an incorrect derived feature:

- `Price_per_SqFt = SalePrice / Gr Liv Area`

If `SalePrice` is the target, then the feature directly contains target information.

This is a very useful example because leakage often enters not through raw columns, but through "clever" engineered features.

## 29. Preventing Leakage

The lecture gives several rules:

- keep train and test sets strictly separate
- avoid future or target-related features
- use time-aware methods for time series
- do preprocessing inside each cross-validation fold

This last rule is especially important.

For example:

- fit imputer on training fold only
- fit scaler on training fold only
- transform validation fold using the fitted training transformation

If this order is violated, the evaluation is contaminated.

## 30. Pipelines

The final block of the lecture explains why pipelines are so important.

Pipelines combine multiple steps into a single reproducible workflow.

Typical steps include:

- imputation
- encoding
- scaling
- feature engineering
- dimensionality reduction
- model fitting

The main idea is not convenience alone. The real value is correctness.

## 31. Why Pipelines Matter

The slides list several benefits:

- consistency
- efficiency
- reduced human error
- leakage prevention
- better generalization

Why pipelines prevent leakage:

- each transformation is fit only on training data inside the pipeline
- the same learned transformation is then applied to validation or test data

This is much safer than manually preprocessing the whole dataset first.

## 32. Practical Pipeline Example

The notebook `Coding_3.6.ipynb` introduces a practical sklearn-style pipeline using:

- `ColumnTransformer`
- `Pipeline`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- polynomial features

It also creates engineered columns such as:

- `Age`
- `TotalArea`
- `IsNew`
- `AreaPerYear`

This is a strong final example because it shows that feature engineering and preprocessing do not need to be ad hoc notebook code. They can be formalized into a reusable pipeline.

## 33. Practical Notebook Map

### Notebook 1: `Coding_3.1.ipynb`

Focus:

- feature selection

Key practical topics:

- correlation methods
- chi-square
- WoE and IV
- mutual information
- ANOVA
- forward selection
- backward elimination
- RFE
- exhaustive search
- Lasso, Ridge, tree importance

Dataset:

- Adult Census Income

### Notebook 2: `Coding_3.2.ipynb`

Focus:

- feature generation

Key practical topics:

- good vs bad transformations
- interaction features
- domain-specific features
- time-based features
- aggregation
- binning
- categorical feature engineering
- model comparison before and after new features

Dataset:

- Ames Housing

### Notebook 3: `Coding_3.3.ipynb`

Focus:

- dimensionality reduction

Key practical topics:

- standardization before PCA
- PCA with explained variance
- t-SNE visualization
- UMAP visualization

Dataset:

- mobile device usage

### Notebook 4: `Coding_3.4.ipynb`

Focus:

- splitting and cross-validation

Key practical topics:

- random split
- stratified split
- SMOTE
- time-based split
- group-aware split
- K-Fold
- LOOCV
- bias-variance examples

Datasets:

- Iris
- fraud dataset
- bankruptcy dataset
- air quality data

### Notebook 5: `Coding_3.5.ipynb`

Focus:

- leakage

Key practical topics:

- train-test contamination
- target leakage
- time-series leakage
- leakage from derived features
- leakage through imputation

Dataset:

- Ames Housing

### Notebook 6: `Coding_3.6.ipynb`

Focus:

- pipelines

Key practical topics:

- column-wise preprocessing
- numerical and categorical branches
- automated transformations
- reusable model-ready preprocessing flow

## 34. Key Takeaways

- Feature selection removes noise and redundancy.
- Filter, wrapper, and embedded methods solve different problems.
- Feature generation should add meaning, not just more columns.
- Good engineered features often rely on domain logic.
- Dimensionality reduction helps with overfitting, speed, and visualization.
- PCA is linear; t-SNE and UMAP are mostly for non-linear structure and visualization.
- Splitting strategy must match the data structure.
- Cross-validation gives more reliable estimates than a single split.
- Leakage can make a weak model look excellent.
- Pipelines are one of the safest ways to standardize preprocessing.

## 35. Quick Revision Questions

1. Why can two highly correlated features be a problem in a linear model?
2. What is the difference between filter, wrapper, and embedded feature selection?
3. Why is a domain-specific feature often better than a random mathematical combination?
4. What is the difference between feature selection and feature extraction?
5. Why is PCA usually applied after scaling?
6. When would stratified split be preferable to random split?
7. Why must SMOTE be applied only to the training set?
8. What is the difference between train-test contamination and target leakage?
9. Why can a derived feature also create leakage?
10. How do pipelines help prevent preprocessing mistakes?
