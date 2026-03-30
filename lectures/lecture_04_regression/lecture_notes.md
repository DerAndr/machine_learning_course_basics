# Lecture 04 Recap: Regression

> Lecture number: 04
> Lecture slug: `lecture_04_regression`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What Regression Solves

Regression is used when the target variable is **continuous**.

That means the model predicts a number rather than a class label.

Typical examples from the lecture:

- house prices
- stock trends
- sales forecasts
- disease progression

Formally, the lecture writes the problem as:

- each observation has features `X_i`
- each observation has a target `Y_i`
- the goal is to approximate the relationship

`Y = f(X) + epsilon`

Here:

- `f(X)` is the systematic part the model tries to learn
- `epsilon` is the random noise or error term

This is a very important idea. A regression model is not expected to explain everything perfectly. Some variation is due to unobserved factors, randomness, measurement error, or incomplete features.

## 2. Main Types of Regression in the Lecture

The slides mention three basic forms:

- simple regression
- multiple regression
- polynomial regression

### 2.1 Simple Regression

Uses one feature to predict the target.

Example:

- predict house price from living area only

This is useful for intuition, but real datasets are usually too complex for a single predictor.

### 2.2 Multiple Regression

Uses many features.

Example:

- predict house price from area, neighborhood, year built, quality, and other variables

This is the standard practical setup in tabular regression.

### 2.3 Polynomial Regression

Used when the relationship is not linear in the original feature space.

Important nuance:

- polynomial regression is still linear in the coefficients
- it becomes non-linear in the input features because we add transformed terms such as `x^2`, `x^3`, and interactions

This distinction matters because many linear-model tools still apply after polynomial feature expansion.

## 3. Basic Regression Models in the Lecture

The lecture introduces two core model families:

- linear regression
- decision tree regression

This is a useful contrast because they learn very different types of relationships.

### 3.1 Linear Regression

Linear regression assumes the expected target is a linear combination of the features.

The slide writes the model as:

`y_i = beta_0 + beta_1 x_i1 + ... + beta_p x_ip + epsilon_i`

In matrix form:

`Y = X beta + epsilon`

Where:

- `Y` is the vector of target values
- `X` is the design matrix of predictors
- `beta` is the vector of coefficients
- `epsilon` is the vector of errors

Interpretation:

- `beta_0` is the intercept
- each `beta_j` measures how the target changes when the corresponding feature changes, holding the others fixed

This "holding others fixed" part is crucial. In multiple regression, coefficients are conditional effects, not isolated one-variable summaries.

### 3.2 Decision Tree Regressor

Decision tree regression does not fit one global formula. Instead, it splits the feature space into regions and predicts a value inside each region.

Why it is useful:

- can capture non-linear structure
- naturally handles interactions
- does not require the same linear assumptions as OLS

Main weakness:

- can overfit badly if not controlled
- piecewise-constant predictions can become unstable

In practice, this makes trees a good comparison point to linear regression:

- linear regression is simpler and more interpretable
- trees are more flexible but need stronger regularization

## 4. Ordinary Least Squares (OLS)

OLS is the central estimation method in the lecture.

The goal of OLS is to choose coefficients that minimize the **sum of squared residuals**.

Residual:

- residual = actual value - predicted value

If the prediction for an observation is `y_hat_i`, then the residual is:

- `e_i = y_i - y_hat_i`

The OLS objective is:

- minimize the total squared error across all observations

The slides write the loss as:

`S(beta) = (y - X beta)^T (y - X beta)`

This quantity is also tied to:

- RSS: residual sum of squares
- SSR in some slide conventions

Why squared residuals?

- keeps the objective differentiable
- penalizes large errors more strongly than small ones
- gives a closed-form solution under standard assumptions

## 5. Closed-Form OLS Solution

The lecture shows the exact matrix solution:

`beta_hat = (X^T X)^(-1) X^T y`

Students do not need to derive this by hand in most practical settings, but they should understand what it means.

The formula tells us:

- OLS depends on the geometry of the predictors through `X^T X`
- if the predictors are badly conditioned or highly collinear, the inverse becomes unstable
- that instability directly affects the coefficient estimates

This is why multicollinearity becomes a major topic later in the lecture.

## 6. OLS Assumptions and What They Mean

The lecture lists Gauss-Markov assumptions and explains that under these conditions OLS is **BLUE**:

- Best Linear Unbiased Estimator

That phrase is important. It does **not** mean OLS is always the best model overall. It means that among linear unbiased estimators, OLS has the smallest variance when the assumptions hold.

### 6.1 Linearity

The expected target is linear in the coefficients.

This does not mean every real-world relationship must look like a straight line in raw coordinates. It means the model must be representable as a linear combination of chosen predictors.

### 6.2 Independence

Observations should be independent.

Why it matters:

- if observations are dependent, standard errors and inference can be wrong

This becomes especially important in time series or grouped data.

### 6.3 Homoscedasticity

The error variance should be constant across observations.

Meaning:

- residual spread should not systematically grow or shrink with the prediction level

If residual variance changes with fitted values, we have heteroscedasticity.

Why this matters:

- coefficient estimates may still exist
- but inference and uncertainty estimates become less reliable

### 6.4 No Perfect Multicollinearity

Predictors should not be perfectly correlated.

Why:

- if one predictor can be written exactly as a combination of others, `X^T X` is not invertible

This breaks the standard OLS solution.

### 6.5 Zero Mean Errors

The errors should have expected value zero.

This means the model is not systematically biased upward or downward on average.

### 6.6 No Autocorrelation

Residuals should not be correlated across observations.

This is especially important in ordered data such as:

- time series
- panel data
- repeated measurements

### 6.7 Normality of Errors

The lecture correctly notes that normality is mainly needed for exact inference:

- confidence intervals
- hypothesis tests

It is not strictly required to compute OLS coefficients, but it is useful when making formal statistical claims.

## 7. OLS vs MLE

The lecture also compares OLS and Maximum Likelihood Estimation.

This is a more theoretical part, but it is worth understanding.

### 7.1 Maximum Likelihood Idea

MLE chooses parameter values that maximize the probability of observing the data under the assumed model.

For linear regression with independent Gaussian errors, the likelihood leads to a log-likelihood objective that depends on squared residuals.

That is why:

- under the normal-error setup
- OLS and MLE produce the same coefficient estimate for `beta`

So both methods give:

`beta_hat = (X^T X)^(-1) X^T y`

### 7.2 Key Difference

The lecture emphasizes that the variance estimator differs.

Under the standard OLS approach:

- the variance estimate uses a degrees-of-freedom correction
- this gives an unbiased estimator

Under MLE:

- the variance estimate divides by `n`
- this is biased in finite samples

This is a good technical detail for students because it shows:

- two methods can agree on coefficients
- but still differ on uncertainty estimation

### 7.3 Practical Meaning

In many practical ML settings:

- we care mainly about prediction quality

In classical statistics:

- we also care about inference, uncertainty, and significance

That is one reason why libraries like `statsmodels` are often preferred for statistical interpretation, while `scikit-learn` is common for predictive workflows.

## 8. Multicollinearity

Multicollinearity occurs when predictors are highly correlated with each other.

This is a core practical problem in multiple regression.

The lecture describes the consequences clearly:

1. `X^T X` becomes nearly singular
2. the inverse becomes unstable
3. coefficient estimates become highly variable
4. small changes in data can cause large coefficient changes

This is one of the most important reasons why students should not trust regression coefficients blindly.

A model can have decent predictive performance and still have unstable or hard-to-interpret coefficients.

## 9. Detecting Multicollinearity

The slides mention:

- correlation matrix analysis
- Variance Inflation Factor (VIF)

### 9.1 Correlation Matrix

Helps detect strongly related feature pairs.

It is a useful first pass, but it has limits:

- it only sees pairwise relationships
- multicollinearity can involve more than two features

So correlation heatmaps are helpful but not sufficient.

### 9.2 Variance Inflation Factor (VIF)

The lecture gives the formula:

`VIF(X_i) = 1 / (1 - R_i^2)`

Where `R_i^2` is obtained by regressing feature `X_i` on all the other predictors.

Interpretation:

- if `R_i^2` is close to `1`, the feature is highly explained by the other features
- then `VIF` becomes large

Rule of thumb from the slides:

- `VIF > 5` or `VIF > 10` often indicates problematic multicollinearity

This does not mean those thresholds are universal laws, but they are useful warning signals.

## 10. Regularization

The lecture presents regularization as a major solution to overfitting and multicollinearity.

The three main methods are:

- Ridge
- Lasso
- ElasticNet

The common idea is:

- modify the loss function by adding a penalty on coefficient size

This discourages overly large coefficients and makes the model more stable.

## 11. Ridge Regression

Ridge uses an `L2` penalty.

The loss is:

- squared error
- plus lambda times the sum of squared coefficients

Main effect:

- coefficients are shrunk toward zero
- but usually not exactly to zero

Why it helps:

- reduces variance
- handles multicollinearity well
- stabilizes the solution when predictors are strongly correlated

When to use Ridge:

- many predictors have some signal
- you do not want to eliminate features completely
- multicollinearity is present

## 12. Lasso Regression

Lasso uses an `L1` penalty.

Main effect:

- some coefficients can become exactly zero

This makes Lasso useful for:

- shrinkage
- automatic feature selection

When to use Lasso:

- many features are irrelevant
- you want a sparser model

Important caution:

- if predictors are strongly correlated, Lasso may select one and suppress others somewhat arbitrarily

## 13. ElasticNet

ElasticNet combines:

- `L1` penalty from Lasso
- `L2` penalty from Ridge

Why it matters:

- balances feature selection and coefficient shrinkage
- often behaves better than plain Lasso when predictors are correlated

The lecture highlights that ElasticNet is useful when:

- the number of predictors is large
- multicollinearity is present
- you want both sparsity and stability

## 14. Why Scaling Matters for Regularized Regression

The lecture explicitly notes that scaling is important for Ridge and Lasso.

This is a technical point students should remember.

Without scaling:

- features on larger numeric ranges get penalized differently
- the penalty becomes unfair across coefficients

That is why standardized features are usually preferred before regularized linear models.

This also appears in the practical notebook, where preprocessing is handled in a pipeline with scaling.

## 15. Regression Metrics

The lecture includes several core evaluation metrics.

Students should understand not only the formulas, but what each metric emphasizes.

### 15.1 MSE

Mean Squared Error:

`MSE = (1/n) sum (y_i - y_hat_i)^2`

Why it matters:

- standard regression loss
- strongly penalizes large errors

Weakness:

- expressed in squared target units, which can be harder to interpret

### 15.2 RMSE

Root Mean Squared Error:

- square root of MSE

Why it is useful:

- same units as the target
- easier to interpret than MSE

RMSE is often a very practical metric when the target has a business meaning, such as dollars or sales volume.

### 15.3 MAE

Mean Absolute Error:

- average absolute residual

Why it is useful:

- more robust to outliers than MSE/RMSE

Tradeoff:

- does not penalize very large errors as strongly as MSE

### 15.4 R-squared

`R^2` measures the proportion of target variance explained by the model.

Interpretation:

- `R^2 = 1` means perfect fit
- `R^2 = 0` means the model is no better than predicting the mean
- negative `R^2` can happen on test data if the model performs worse than the mean baseline

Students often misuse `R^2`. It does **not** mean:

- the model is correct
- the relationship is causal
- predictions are necessarily accurate in absolute terms

### 15.5 Adjusted R-squared

Adjusted `R^2` corrects for the number of predictors.

Why this matters:

- ordinary `R^2` never decreases when you add features
- adjusted `R^2` penalizes adding predictors that do not improve the model enough

This makes it more meaningful when comparing models of different sizes.

## 16. Residual Analysis

The lecture and notebook both emphasize that regression evaluation is not only about one metric.

Residual analysis is central because residuals tell us where the model is failing.

The practical notebook checks:

- residual sum of squares
- residual scatter plot
- residual histogram
- Q-Q plot
- residual mean, standard deviation, skewness, kurtosis

### 16.1 Residuals vs Fitted or True Values

If residuals show a clear pattern, that suggests the model misses structure.

For example:

- curvature may suggest non-linearity
- funnel shape may suggest heteroscedasticity

### 16.2 Residual Distribution

If residuals are strongly skewed or heavy-tailed, then:

- normality assumptions may be weak
- some inference results may be unreliable

### 16.3 Q-Q Plot

The Q-Q plot compares residual quantiles to normal quantiles.

If the points strongly deviate from the straight line, residual normality is questionable.

This is especially useful when students want to check whether OLS inference assumptions are approximately reasonable.

## 17. Data Preparation for Regression

The lecture repeats several important preprocessing points for regression:

- handle missing data
- scale features when needed
- encode categorical variables

This is not just housekeeping. It directly affects model quality.

### Missing Values

If missing values are left untreated:

- many regression implementations will fail
- or the training sample may shrink too much after deletion

### Categorical Variables

Categorical variables must usually be encoded.

The lecture mentions:

- one-hot encoding
- label encoding

Important caution:

- one-hot encoding can introduce multicollinearity if all dummy columns plus an intercept are kept

This is the well-known dummy-variable trap.

In practice, many libraries handle this automatically or allow dropping one reference category.

## 18. Train-Test Split and Cross-Validation for Regression

The lecture recommends ordinary train-test splitting and K-Fold cross-validation.

It also mentions a regression-specific nuance:

- folds should ideally preserve similar target distributions

This is sometimes approximated in practice by stratifying on target bins or percentiles.

Why this matters:

- if one fold contains mostly low target values and another mostly high target values, estimates can become unstable

The notebook uses:

- train-test split
- 10-fold cross-validation

This is a solid baseline workflow.

## 19. Overfitting and Underfitting in Regression

The slides define:

- overfitting: high train performance, low test performance
- underfitting: poor train and test performance

Typical fixes for overfitting:

- regularization
- cross-validation
- simplifying the model

Typical fixes for underfitting:

- add meaningful features
- increase model flexibility
- try non-linear models

This is a good reminder that model quality depends on balancing complexity.

## 20. Interpretability

The lecture contrasts interpretable linear models with more complex black-box models.

### Linear Regression

Main advantage:

- coefficients can often be interpreted directly

But that interpretation is only safe when:

- the model is reasonably specified
- multicollinearity is not extreme
- preprocessing has been done carefully

### More Complex Regressors

The slides mention models like:

- random forests
- gradient boosting

These can capture non-linear effects better, but interpretation becomes harder.

That is why tools such as:

- SHAP
- LIME

are mentioned for feature-contribution analysis.

## 21. Libraries Mentioned in the Lecture

The lecture highlights three tool families.

### Scikit-learn

Best for:

- predictive pipelines
- standard regression models
- preprocessing and model selection

Main strength:

- clean ML workflow

Main limitation:

- less statistical detail for inference

### Statsmodels

Best for:

- OLS summaries
- p-values
- confidence intervals
- diagnostics

This is why it appears in the practical notebook after the sklearn workflow.

### Deep Learning Libraries

The slides also mention:

- TensorFlow
- Keras
- PyTorch

These are useful for complex non-linear regression, but they are not the main focus of this lecture.

## 22. Practical Notebook: `Regression_demo.ipynb`

This notebook is the main practical anchor for the lecture.

It uses the Ames Housing dataset and follows a realistic regression workflow:

1. dataset loading
2. EDA and feature typing
3. missing-value inspection
4. correlation heatmap
5. preprocessing pipeline
6. linear regression training
7. train/test prediction
8. metric evaluation
9. residual diagnostics
10. VIF analysis
11. cross-validation
12. Ridge and Lasso tuning
13. coefficient inspection
14. predicted vs actual plot
15. statsmodels OLS example

This is a strong practical sequence because it shows that regression is not only about calling `.fit()`. A serious regression workflow includes:

- preprocessing
- diagnostics
- validation
- multicollinearity checks
- regularization comparison

### Pipeline Design in the Notebook

The notebook builds:

- a numerical preprocessing branch
- a categorical preprocessing branch
- a combined `ColumnTransformer`
- a final `Pipeline`

This is good engineering practice because it prevents inconsistent preprocessing between training and test data.

### VIF in Practice

The notebook computes VIF after preprocessing and one-hot encoding.

This is important because multicollinearity can become worse after encoding.

### Ridge and Lasso Tuning

The notebook uses `GridSearchCV` to tune `alpha`.

Students should understand:

- larger `alpha` means stronger regularization
- the right value is data-dependent
- tuning should be done with validation or cross-validation, not by looking at the test set

### Coefficient Analysis

The notebook also extracts coefficients from the tuned Lasso model.

That is useful because students can see:

- which features remain important
- which coefficients shrink toward zero
- how regularization changes interpretability

## 23. About `Regression Draft.ipynb`

The second notebook in the source materials is a rough illustration notebook with synthetic curves and tree-regression sketches.

It is useful mainly as a visual intuition aid:

- what noisy regression data can look like
- how a tree regressor behaves on a non-linear pattern

But the main practical content of the lecture is clearly in `Regression_demo.ipynb`.

## 24. Advanced Topics Mentioned at the End

The lecture briefly names several additional models:

- logistic regression
- quantile regression
- polynomial regression
- Poisson regression
- support vector regression
- generalized linear models

The key point is not to memorize every method now. The main takeaway is:

- regression is a large family of models
- ordinary linear regression is only the starting point
- different target structures and assumptions require different tools

## 25. Key Takeaways

- Regression predicts continuous targets.
- Linear regression models the target as a linear combination of predictors plus noise.
- OLS minimizes the sum of squared residuals.
- Under Gauss-Markov assumptions, OLS is BLUE.
- MLE and OLS agree on coefficients under normal linear regression, but differ in variance estimation.
- Multicollinearity makes coefficients unstable and can be diagnosed with `VIF`.
- Ridge shrinks coefficients; Lasso shrinks and selects; ElasticNet balances both.
- Scaling matters for regularized regression.
- `MSE`, `RMSE`, `MAE`, `R^2`, and adjusted `R^2` answer different evaluation questions.
- Residual analysis is essential for checking whether the model assumptions make sense.
- A full regression workflow includes preprocessing, diagnostics, validation, and regularization.

## 26. Quick Revision Questions

1. What is the difference between simple, multiple, and polynomial regression?
2. Why does OLS minimize squared residuals instead of raw residuals?
3. What does `beta_hat = (X^T X)^(-1) X^T y` tell us about the effect of correlated predictors?
4. Which Gauss-Markov assumptions matter most for coefficient interpretation and inference?
5. Why can two models with similar `R^2` still differ substantially in practical usefulness?
6. What does a large `VIF` mean?
7. When is Ridge preferable to Lasso?
8. Why is feature scaling especially important for regularized regression?
9. What can a Q-Q plot reveal about a regression model?
10. Why is cross-validation important even when a train-test split already exists?
