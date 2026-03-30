# Lecture 02 Recap: Data Preparation Part 1

> Lecture number: 02
> Lecture slug: `lecture_02_data_preparation_part_1`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. Why Data Preparation Matters

This lecture makes a very practical point: good models start with good data.

Data preparation is the stage where we turn raw data into something a model can actually use. In real projects, this stage often takes more time than model training itself. The reason is simple:

- real data contains missing values
- records may be duplicated
- some values are inconsistent or extreme
- numerical and categorical features usually need different preprocessing

The main warning from the lecture is the classic rule:

**Garbage in -> garbage out**

Even a strong algorithm will perform poorly if the data is noisy, biased, badly encoded, or incorrectly cleaned.

In the lecture, data preparation is presented as part of the **CRISP-DM** pipeline. It sits after initial data understanding and before serious modeling. That means it is not a cosmetic step. It is part of the core machine learning workflow.

## 2. Main Stages of Data Preparation

The lecture lists several stages:

1. data collection
2. data cleaning
3. transformation and encoding
4. feature selection and feature generation
5. dimensionality reduction
6. data splitting
7. cross-validation
8. data leakage prevention
9. building data pipelines

This lecture focuses mostly on stages 2 and 3:

- cleaning the data
- handling missing values
- handling duplicates
- detecting and treating outliers
- scaling and transforming numerical variables
- encoding categorical variables

So the main goal is to prepare a reliable input for later modeling.

## 3. Data Cleaning: The Core Idea

Data cleaning means removing or correcting problems that make the dataset unreliable.

According to the slides, the main tasks are:

- handling missing values
- removing duplicates
- correcting inconsistencies

Why does this matter?

- missing values can break models or bias results
- duplicates can give too much weight to repeated observations
- inconsistent values can silently corrupt analysis
- outliers can distort statistics and some algorithms

In other words, data cleaning is not only about making the dataset look tidy. It is about making later decisions statistically safer.

## 4. Missing Values

Missing values are one of the main topics of this lecture and the first practical notebook.

The lecture explains that missing values can appear because of:

- data entry errors
- system failures
- different data collection rules
- people refusing to answer
- values that are not applicable in some cases

This last case is especially important. In practice, a missing value is not always a problem. Sometimes missingness itself carries information.

In the Ames Housing examples, some missing values mean that a feature does not exist. For example, a missing `Pool QC` value often means the house has no pool. That should often be interpreted as a category such as `"None"`, not as a random absence.

## 5. Types of Missingness

The lecture introduces three standard missing-data mechanisms.

### 5.1 MCAR: Missing Completely at Random

MCAR means the probability of missingness is independent of both:

- observed variables
- unobserved missing value itself

This is the cleanest case statistically.

Example from the lecture:

- some respondents randomly skip a survey question

In the notebook, MCAR is illustrated by randomly removing about 5 percent of values from `Gr Liv Area`.

Why this matters:

- MCAR is the least dangerous form of missingness
- many simple methods behave reasonably under MCAR
- if the proportion is small, deletion may sometimes be acceptable

### 5.2 MAR: Missing at Random

MAR means missingness depends on some other observed feature, but not on the missing value itself once those observed variables are taken into account.

Example from the lecture:

- older people are less likely to report income

The missingness depends on age, which is observed.

In the notebook, MAR is demonstrated by removing values based on another observed variable. This is important because it shows that missingness can be systematic even when it is not directly caused by the hidden value itself.

Why this matters:

- MAR is common in real data
- simple deletion can create bias
- model-based or conditional imputation becomes more attractive

### 5.3 MNAR: Missing Not at Random

MNAR means the probability of missingness depends on the missing value itself.

Example from the lecture:

- people with higher incomes are less likely to report income

In the notebook, MNAR is illustrated by making larger values more likely to be removed.

This is the hardest case because the missingness mechanism is tied to information we do not fully observe.

Why this matters:

- naive imputation may produce misleading distributions
- deletion can remove exactly the most informative part of the data
- domain knowledge becomes especially important

## 6. How to Analyze Missing Values in Practice

The first notebook does not jump directly into imputation. It first treats missingness as something to analyze.

The workflow is:

1. count missing values per column
2. compute the percentage of missing values
3. visualize missingness with a heatmap
4. inspect the meaning of each column
5. decide whether missing values mean "unknown", "not applicable", or "missing because of a process"

This is good practice.

A missing-value heatmap helps answer questions like:

- are missing values concentrated in a few columns?
- do missing values appear in related groups?
- is there a structural pattern rather than random noise?

The notebook also checks how missingness relates to variables such as `SalePrice`, which is exactly the kind of reasoning students should learn:

- not all missingness is equally important
- some missing values are strongly associated with the target
- sometimes the fact that a value is missing is predictive by itself

## 7. Missing-Value Handling Strategies

The lecture groups handling methods into **deletion** and **imputation**.

### 7.1 Deletion

The slides mention:

- listwise deletion
- pairwise deletion
- dropping columns

#### Listwise Deletion

Listwise deletion removes all rows with missing values in the selected analysis.

Pros:

- simple
- easy to implement

Cons:

- can remove a lot of data
- can change the sample distribution
- can introduce bias if missingness is not MCAR

#### Pairwise Deletion

Pairwise deletion uses all rows that are available for each specific pairwise computation.

Pros:

- keeps more data than listwise deletion

Cons:

- different statistics may be computed on different subsets
- results can become harder to compare consistently

#### Dropping a Column

Dropping a feature may be reasonable if:

- the missing rate is extremely high
- the column has weak practical value
- the signal can be replaced by better features

But this should not be automatic. Some sparse columns are still meaningful.

### 7.2 Simple Imputation

The lecture and notebook cover classical imputation methods:

- mean
- median
- mode
- constant value

These are easy to implement with `SimpleImputer`.

#### Mean Imputation

Use case:

- numerical variable
- roughly symmetric distribution
- limited missingness

Weakness:

- sensitive to outliers
- reduces variance
- can artificially pull values toward the center

#### Median Imputation

Use case:

- skewed numerical variable
- presence of outliers

Why it is often safer:

- median is robust to extreme values

#### Mode Imputation

Use case:

- categorical variables
- repeated dominant category

Weakness:

- can over-strengthen the most frequent class

#### Constant Imputation

Use case:

- representing "missing" explicitly
- creating a separate category such as `"None"`
- setting a numeric placeholder when that has domain meaning

Weakness:

- may create artificial patterns if the constant has no real interpretation

## 8. Model-Based Imputation

The notebook goes beyond the slide-level summary and shows more advanced approaches:

- linear regression imputation
- KNN imputation
- iterative imputation with Bayesian Ridge

This is important because simple imputers are not always enough.

### Linear Regression Imputation

Idea:

- predict the missing numerical value using other available variables

Strength:

- uses relationships between features

Weakness:

- assumes the relationship is predictable enough
- can underestimate uncertainty

### KNN Imputation

Idea:

- fill a missing value using similar observations

Strength:

- more flexible than mean imputation
- can capture local structure

Weakness:

- sensitive to feature scaling
- can become slow on large datasets
- "nearest" can become unreliable in high dimensions

### Iterative Imputation

Idea:

- repeatedly model each feature with missing values as a function of the others

Strength:

- more sophisticated and often more realistic

Weakness:

- more complex
- slower
- depends on modeling assumptions

The notebook also compares descriptive statistics before and after imputation and even uses significance tests. That is a good habit: preprocessing should not be treated as magic. You should check how much it changes the data.

## 9. Duplicates

The lecture defines duplicates as records that appear more than once.

Why duplicates are dangerous:

- repeated rows can overweight some observations
- summary statistics become biased
- models may learn repeated patterns too strongly
- train/test leakage can become worse if duplicates are split across sets

The lecture mentions two common ways to remove duplicates:

- using a unique identifier
- using exact matches across features

In practice, students should remember that duplicates are not always exact copies. Sometimes they are near-duplicates caused by:

- different formatting
- different capitalization
- small timestamp changes
- rounding differences

So duplicate handling often begins with exact matches but may later require more careful record linkage.

## 10. Outliers

Outliers are observations that differ strongly from the rest of the data.

The lecture gives an important warning: outliers are not automatically "bad." Some are:

- data errors
- unusual but valid cases
- rare events that are exactly what we want to detect

This distinction matters. Removing an outlier without thinking can damage the dataset.

Examples from the lecture:

- CEO salaries compared with ordinary employees
- anomalous spikes in sensors or financial data

In the Ames dataset, extreme values in house size or price are a good way to discuss this problem.

## 11. Why Outliers Matter Statistically

Outliers can affect:

- the mean
- the standard deviation
- regression coefficients
- distance-based algorithms
- visual interpretations

The lecture specifically mentions that methods such as **linear regression** and **KNN** are sensitive to scale and can be strongly affected by extreme points.

This is because:

- regression tries to fit a line that minimizes residual error
- KNN uses distances directly
- standard scaling uses mean and standard deviation, both of which are sensitive to outliers

So outlier handling is not only about cleaning. It changes model behavior.

## 12. Traditional Outlier Detection Methods

The second notebook demonstrates several classical methods.

### 12.1 Scatter Plot

Scatter plots are the first visual screening tool.

In practice, the notebook plots:

- `Gr Liv Area` on one axis
- `SalePrice` on the other

This helps identify observations that sit far away from the general cloud of points.

Why it is useful:

- intuitive
- preserves geometric structure
- shows whether a point is extreme in one dimension or in the relationship between two variables

Limitation:

- purely visual
- less effective in high dimensions

### 12.2 Boxplot and IQR Rule

The notebook then uses a boxplot and the **interquartile range (IQR)** method.

Recall:

- `Q1` is the 25th percentile
- `Q3` is the 75th percentile
- `IQR = Q3 - Q1`

Common rule:

- lower bound = `Q1 - 1.5 * IQR`
- upper bound = `Q3 + 1.5 * IQR`

Points outside those bounds are flagged as outliers.

Why this method is popular:

- simple
- robust compared with mean-based rules
- works well for many univariate checks

Limitation:

- it is still a rule of thumb
- a skewed but valid distribution can produce many flagged points

### 12.3 Z-Score

The lecture gives the standard formula:

`z = (x - mu) / sigma`

Interpretation:

- the z-score tells how many standard deviations a point is from the mean

A common threshold is:

- `|z| > 3`

Why it is useful:

- simple
- easy to explain

Main limitation:

- it assumes mean and standard deviation are meaningful summaries
- it works best when the distribution is not heavily skewed
- the outliers themselves can distort the mean and standard deviation

### 12.4 Cook's Distance

Cook's Distance appears in both the slides and the notebook.

It is not just an outlier detector in the simple sense. It measures how much a point influences a regression model.

That means a point can be influential because:

- it is extreme in the predictor space
- it has a large residual
- or both

Why this is useful:

- some points may not look very extreme in a univariate plot
- but they can still strongly change the regression fit

This is especially important for linear regression diagnostics.

## 13. Advanced Outlier Detection Methods

The lecture then moves beyond classical one-variable rules.

### 13.1 DBSCAN

The slides describe DBSCAN as a density-based clustering method.

Main idea:

- dense groups form clusters
- points in low-density regions may be treated as noise or outliers

Why it is useful:

- does not assume spherical clusters
- can naturally label isolated points as anomalies

Practical limitation:

- strongly depends on hyperparameters such as `eps` and `min_samples`

### 13.2 Isolation Forest

The lecture states the key principle correctly:

- outliers are few
- outliers are far from the rest

Isolation Forest randomly partitions the data. Points that can be isolated in fewer splits are more likely to be anomalous.

Why it is useful:

- works well in many practical cases
- scalable
- does not require a fully parametric distributional assumption

### 13.3 Local Outlier Factor (LOF)

LOF compares the local density of a point to the density of its neighbors.

Interpretation:

- if a point is much less dense than its neighborhood, it may be an outlier

Why this matters:

- some points are not globally extreme
- but they are unusual relative to their local region

### 13.4 One-Class SVM

The notebook adds One-Class SVM as another anomaly-detection method.

Idea:

- learn the boundary of the normal data region
- label observations outside that region as anomalies

Practical limitation:

- sensitive to parameter choices
- can be expensive on larger datasets

## 14. Outliers in Time Series and Multivariate Settings

The lecture briefly broadens the discussion beyond simple tabular cases.

It mentions:

- ARIMA
- spectral residual methods
- PCA and Mahalanobis distance
- autoencoders
- LSTM-based approaches for time series

The main lesson is that the definition of "outlier" depends on the structure of the data.

For example:

- in a time series, an outlier may be a sudden deviation from temporal pattern
- in multivariate data, a point may look normal in each individual feature but abnormal in the joint feature space

This is why multivariate anomaly detection is often harder than univariate screening.

## 15. What to Do After Detecting Outliers

The lecture gives several options:

- remove them
- impute them
- winsorize them
- transform the variable
- keep them if they are informative

This is one of the most important judgment points in preprocessing.

### Remove

Best when:

- the point is clearly an error
- the record is irrelevant
- the measurement is corrupted

### Impute

Sometimes the lecture treats outliers similarly to missing values:

- replace them with mean, median, mode
- use model-based imputation
- use domain-specific imputation

This can be reasonable if the value is judged invalid.

### Winsorize

Winsorization caps extreme values instead of deleting them.

Why it can help:

- keeps dataset size unchanged
- reduces the influence of extreme tails

### Transform

Transformations such as logs can compress large values and reduce skew.

### Keep

Keep the outlier if it is meaningful.

Example:

- fraud detection
- rare but real medical conditions
- extreme but valid transactions

A model designed to detect rare events should not blindly delete the rare events.

## 16. Transformation and Scaling

The third notebook starts with numerical transformations.

The lecture explains that transformation is used to make features:

- more consistent
- easier for models to interpret
- more comparable in scale

This matters especially for algorithms that depend on distances, margins, or optimization geometry.

## 17. Scaling Methods

### 17.1 Min-Max Scaling

Definition:

- rescales a variable into a fixed interval, usually `[0, 1]`

Formula idea:

- subtract the minimum
- divide by the range

Why it is useful:

- preserves ordering
- useful for distance-based algorithms such as KNN
- often helpful for SVM and gradient-based methods

Weakness:

- very sensitive to extreme values

If one outlier is extremely large, most normal values get compressed into a narrow interval.

### 17.2 Standardization

Definition:

- transforms a feature to have mean `0` and standard deviation `1`

Why it is useful:

- often a good default for many linear models
- widely used before PCA
- convenient when features are measured on different scales

Weakness:

- still sensitive to outliers because it uses the mean and standard deviation

### 17.3 Robust Scaling

Definition:

- scales data using median and IQR instead of mean and standard deviation

Why it matters:

- more stable when outliers are present

This connects directly to the previous section of the lecture. If outliers are real and should stay, robust scaling may be a better choice than standard scaling.

## 18. Transformations

### 18.1 Log Transformation

The lecture says log transformation is useful for:

- large positive values
- right-skewed distributions
- variance stabilization

This is a common choice for financial, count-like, and heavy-tailed variables.

Practical caution:

- ordinary logarithm requires strictly positive values
- in practice, `log1p(x)` is often used to handle zeros safely

The notebook uses `np.log1p`, which is a good practical version.

### 18.2 Box-Cox Transformation

The slide describes Box-Cox as a power transformation that can make data more normal.

The practical detail students should remember is:

- standard Box-Cox requires strictly positive data

That is why the notebook adds `+1` before applying it.

Why use it:

- reduce skewness
- make distributions closer to normal
- sometimes improve model assumptions

### 18.3 Square Root Transformation

Square root transformation is a milder compression than log transformation.

It is often useful for:

- counts
- small positive values
- moderately right-skewed variables

Compared with log transformation:

- it changes the scale less aggressively
- but still reduces the influence of large values

## 19. Binning

The lecture presents binning as the conversion of continuous variables into discrete intervals.

This can be useful for:

- simpler interpretation
- rule-based reasoning
- feature engineering
- reducing sensitivity to noise

But binning always trades precision for simplicity.

### 19.1 Equal-Width Binning

Divides the range into intervals of equal size.

Strength:

- simple

Weakness:

- if the distribution is skewed, some bins may have very few points

### 19.2 Equal-Frequency or Quantile Binning

Creates bins containing roughly equal numbers of observations.

Strength:

- balances counts across bins

Weakness:

- intervals may have very different widths

### 19.3 Custom Binning

Uses domain knowledge to define intervals.

This is often the most interpretable option.

In the notebook, `SalePrice` is converted into user-defined bins such as:

- very low
- low
- medium
- high
- very high

This is a good example of how preprocessing can support business interpretation, not only modeling.

## 20. Why Encoding Is Necessary

Most machine learning models expect numerical inputs. Categorical variables must therefore be converted into numerical form.

The lecture emphasizes that the chosen encoding method depends on:

- whether the category has an order
- how many distinct levels exist
- what model will be used
- whether overfitting is a concern

This is a very important practical idea. There is no single universally best encoding.

## 21. Basic Encoding Methods

### 21.1 One-Hot Encoding

Creates one binary column per category.

Strengths:

- simple
- interpretable
- safe for nominal categories with no order

Weakness:

- can create many columns for high-cardinality variables

Use it when:

- the number of categories is small
- interpretability matters

### 21.2 Label Encoding

Assigns an integer to each category.

Strength:

- compact
- memory efficient

Weakness:

- creates an artificial order

For example, encoding `Cat`, `Dog`, `Bird` as `0`, `1`, `2` may mislead some models into treating the labels as ordered.

### 21.3 Ordinal Encoding

This is appropriate only when categories really have an inherent order.

Example:

- low
- medium
- high

In that case, the numeric code reflects meaning.

But if the categories are nominal, ordinal encoding injects false structure.

## 22. Advanced Encoding Methods

### 22.1 Target Encoding

Target encoding replaces each category with a statistic of the target, usually:

- mean
- median
- sometimes mode

Why it is useful:

- handles high-cardinality features well
- avoids the dimensional explosion of one-hot encoding

Main danger:

- leakage and overfitting

If the encoding uses the full target information too directly, it may memorize noise. The notebook therefore includes a cross-validated custom target-encoding example, which is a much better practical approach than naive encoding on the full dataset.

### 22.2 Frequency Encoding

Replaces a category with how often it appears.

Strength:

- compact
- easy to compute

Weakness:

- frequency does not necessarily represent meaning
- if frequency correlates with the target, it can still behave in ways that are hard to interpret

### 22.3 Binary Encoding

Binary encoding first maps categories to integers and then represents those integers in binary across several columns.

Why it matters:

- reduces dimensionality compared with one-hot
- useful for high-cardinality variables

Tradeoff:

- less interpretable than one-hot encoding

## 23. Additional Encoding Ideas Mentioned in the Slides

The lecture also briefly mentions:

- BaseN encoding
- hashing
- leave-one-out encoding
- Weight of Evidence
- backward difference encoding
- Helmert encoding

Students do not need to memorize all of them at this stage, but they should remember the broader lesson:

- preprocessing design becomes more specialized as the feature space becomes more complex

## 24. Practical Notebook Map

### Notebook 1: `DataPrep_coding1.ipynb`

Main focus:

- missing-value analysis
- missingness types
- deletion strategies
- simple and advanced imputation

Concrete practical ideas:

- count and visualize missing values
- distinguish real missingness from "not applicable"
- simulate MCAR, MAR, and MNAR
- compare imputation methods statistically

### Notebook 2: `DataPrep_coding2.ipynb`

Main focus:

- outlier detection

Concrete practical ideas:

- scatter plot screening
- boxplot and IQR rule
- z-score screening
- Cook's Distance for regression influence
- DBSCAN
- Isolation Forest
- LOF
- One-Class SVM

This notebook is useful because it shows that different methods find different sets of unusual points.

### Notebook 3: `DataPrep_coding3.ipynb`

Main focus:

- encoding categorical features
- scaling and transformation of numerical features
- binning

Concrete practical ideas:

- toy examples for encoding
- one-hot, label, ordinal, binary, frequency, and target encoding
- min-max scaling, standardization, robust scaling
- log, Box-Cox, and square-root transformations
- equal-width, equal-frequency, and custom binning

This notebook is especially useful because it shows both the mathematical idea and the visual effect of preprocessing.

## 25. Common Real-World Challenges

The final part of the lecture reminds students that preprocessing has broader consequences.

Important themes:

- bias and fairness
- transparency of data decisions
- mixed data types
- large datasets and scalability

This is important because preprocessing choices are not neutral. They shape the final model and can introduce unintended bias.

For example:

- target encoding can leak information
- deleting too many rows can distort the population
- scaling decisions can affect which patterns dominate the model
- handling missing values differently across groups can change fairness

## 26. Key Takeaways

- Data preparation is a central part of machine learning, not a side step.
- Missing values must be analyzed before they are filled.
- MCAR, MAR, and MNAR require different levels of caution.
- Duplicates can silently bias a dataset.
- Outliers should be investigated, not automatically removed.
- Different scaling methods are appropriate for different data shapes.
- Transformations help when distributions are skewed or heavy-tailed.
- Encoding must match both the feature type and the model.
- Binning improves interpretability but loses information.
- Good preprocessing decisions make later modeling more reliable.

## 27. Quick Revision Questions

1. Why can mean imputation be misleading for skewed numerical data?
2. What is the difference between MCAR, MAR, and MNAR?
3. When would robust scaling be safer than standardization?
4. Why can target encoding cause leakage?
5. What is the difference between a statistical outlier and an influential point?
6. Why is one-hot encoding often a poor choice for very high-cardinality features?
7. When is custom binning better than equal-width binning?
