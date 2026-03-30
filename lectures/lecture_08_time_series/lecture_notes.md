# Lecture 08 Recap: Time Series

> Lecture number: 08
> Lecture slug: `lecture_08_time_series`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What Makes Time Series Different

Time series data is a sequence of observations collected over time.

Examples from the lecture:

- daily stock prices
- hourly temperatures
- monthly sales
- heartbeat intervals

The most important property is:

- **time order matters**

This changes the problem fundamentally.

In ordinary tabular regression, rows are usually treated as independent observations. In time series, that assumption often fails:

- today depends on yesterday
- last month may influence this month
- repeating seasonal structure may appear every week, month, or year

That is why time series needs its own modeling and evaluation logic.

## 2. Core Components of a Time Series

The lecture breaks a series into four parts:

- trend
- seasonality
- cyclicity
- irregularity or noise

This decomposition is one of the most useful mental models in forecasting.

### 2.1 Trend

Trend is the long-term direction of the series:

- increasing
- decreasing
- slowly changing over time

The slides mention methods like:

- moving averages
- polynomial fitting

These help reveal the underlying long-run pattern beneath local noise.

### 2.2 Seasonality

Seasonality is a repeating structure with a fixed or regular calendar period.

Examples:

- weekly demand peaks
- yearly retail surges
- monthly billing cycles

The lecture mentions:

- classical decomposition
- STL decomposition

This is important because seasonality is one of the most common sources of predictable structure in real time series.

### 2.3 Cyclicity

Cyclicality refers to recurring but not strictly fixed-period movements.

Example:

- macroeconomic upturns and downturns

This is different from seasonality because:

- cycles do not need a stable calendar period

The lecture points toward:

- spectral analysis
- Fourier-type reasoning

for detecting such components.

### 2.4 Irregularity or Noise

This is the part left after removing more systematic structure.

Noise matters because:

- no model should be expected to predict random fluctuations perfectly
- residual diagnostics often ask whether what remains is close to noise

## 3. Time Series vs Ordinary Regression

The lecture explicitly contrasts time series with regression.

### Time Series

- observations are dependent
- temporal order must be preserved
- autocorrelation is often present

### Regression

- observations are often assumed independent
- random splitting is usually acceptable
- the goal is usually feature-target mapping rather than explicit temporal dependence

This is one of the key conceptual messages of the lecture.

You can still use regression-style models in time series, but only after careful feature engineering and time-aware validation.

## 4. Why Plain Regression Is Not Enough

The lecture gives several reasons:

- regression ignores autocorrelation unless you create lag features manually
- regression does not naturally model seasonality or trend
- random splitting breaks temporal logic
- standard prediction intervals assume independence

This is a good warning for students:

- a time series is not "just another dataset with dates"

If future data leaks into training, evaluation becomes unrealistic.

## 5. Stationarity

Stationarity is one of the most important theoretical concepts in classical time series.

The lecture defines stationary series as those whose statistical properties remain constant over time.

It distinguishes:

- strict stationarity
- weak or second-order stationarity

### 5.1 Weak Stationarity

For weak stationarity:

- mean is constant
- variance is constant
- autocovariance depends only on lag, not on calendar time

This is usually the most practical notion in forecasting.

### 5.2 Why Stationarity Matters

Many classical models, especially ARIMA-family models, work best when the series is stationary after necessary transformations.

If the series keeps changing its mean, variance, or dependence structure over time, then:

- parameter interpretation becomes harder
- forecasts can become unstable

## 6. Testing for Stationarity

The lecture highlights two standard tests:

- Augmented Dickey-Fuller (ADF)
- KPSS

### 6.1 ADF Test

Null hypothesis:

- the series has a unit root
- so it is non-stationary

Interpretation:

- low `p-value` suggests rejecting non-stationarity

### 6.2 KPSS Test

Null hypothesis:

- the series is stationary

Interpretation:

- low `p-value` suggests rejecting stationarity

This pair is especially useful because the null hypotheses go in opposite directions. In practice, using both can give a more nuanced picture.

## 7. Autocorrelation and Partial Autocorrelation

The lecture next introduces:

- ACF
- PACF

These are foundational diagnostic tools in classical time series modeling.

### 7.1 ACF

The autocorrelation function measures the correlation between the series and its lagged versions.

Interpretation:

- strong spikes at certain lags indicate temporal dependence
- slow decay often suggests persistent structure or non-stationarity

### 7.2 PACF

The partial autocorrelation function measures the direct effect of a lag after removing the influence of intermediate lags.

Why this matters:

- ACF captures both direct and indirect lag relationships
- PACF tries to isolate direct lag influence

### 7.3 Practical Rule from the Lecture

The slides give the standard heuristic:

- ACF helps identify MA order
- PACF helps identify AR order

This is not a magic law, but it is a very useful starting rule for ARIMA modeling.

## 8. ARIMA Fundamentals

ARIMA stands for:

- AutoRegressive
- Integrated
- Moving Average

The lecture explains the three parts clearly.

### 8.1 AR Part

The autoregressive component models the current value as a function of previous values.

So the series depends directly on its own history.

### 8.2 I Part

The integrated component means differencing.

Differencing is used to:

- remove trend
- help achieve stationarity

### 8.3 MA Part

The moving-average component models the current value using current and past error terms.

So ARIMA combines:

- dependence on past observations
- dependence on past shocks or residual effects

## 9. Interpreting `p`, `d`, and `q`

The lecture provides the standard interpretation:

- `p`: AR order
- `d`: differencing order
- `q`: MA order

The practical heuristics are:

- `p` often guided by PACF
- `d` guided by trend and stationarity tests
- `q` often guided by ACF

This is one of the most common starting workflows in classical forecasting.

## 10. SARIMA

SARIMA extends ARIMA with explicit seasonal structure.

The full parameter set is:

- non-seasonal: `(p, d, q)`
- seasonal: `(P, D, Q, m)`

Where:

- `P` is seasonal AR order
- `D` is seasonal differencing order
- `Q` is seasonal MA order
- `m` is the seasonal period length

Example:

- monthly data with yearly seasonality often uses `m = 12`

This makes SARIMA a natural extension when the series contains repeating seasonal behavior.

## 11. Strengths and Limits of ARIMA/SARIMA

The slides summarize them well.

### Strengths

- interpretable
- well understood
- good for linear temporal structure
- strong when stationarity assumptions are approximately satisfied

### Assumptions

- linear relationships
- stationarity after transformation
- residuals close to white noise
- often approximate normality for classical inference

### Limitations

- weaker on strongly non-linear patterns
- parameter tuning can be delicate
- not naturally designed for exogenous variables unless extended

This last point matters a lot. In many business problems, outside factors such as promotions, holidays, or weather influence the target strongly.

## 12. Machine Learning Approaches to Time Series

The lecture then transitions from classical methods to ML-style forecasting.

Main advantages mentioned:

- handle non-linearity
- scale to larger datasets
- incorporate diverse features

This is one of the most important practical takeaways of the lecture:

- classical models are elegant and interpretable
- ML models are often more flexible once the feature engineering is done well

## 13. Feature Engineering for ML Time Series Models

The slides are very explicit here, and this section is critical.

Common time-series features:

- lag features
- rolling statistics
- calendar features
- cyclic encodings

### 13.1 Lag Features

Examples:

- `Y_(t-1)`
- `Y_(t-2)`
- ...

This converts temporal dependence into ordinary supervised-learning features.

### 13.2 Rolling Statistics

Examples:

- moving average
- moving standard deviation

These summarize recent local history and help models capture smoothed context.

### 13.3 Time-Based Features

Examples:

- hour
- day of week
- month
- quarter
- holidays

### 13.4 Cyclic Encoding

The lecture also includes sine and cosine encoding for cyclical variables like month.

This is important because:

- December and January are close in a cycle
- but integer encoding `12` and `1` does not reflect that closeness

Cyclic encoding fixes that geometric problem.

## 14. Tree-Based Models for Time Series

The lecture lists:

- decision trees
- random forests
- gradient boosting

These models are useful after feature engineering because they can:

- capture non-linear interactions
- work with heterogeneous predictors
- provide feature-importance information

But students should remember:

- they do not "understand time" automatically
- the temporal structure must be encoded through features and validation design

## 15. Prophet

Prophet is presented as a business-friendly forecasting tool.

The lecture gives the additive formulation:

`y(t) = g(t) + s(t) + h(t) + epsilon_t`

Where:

- `g(t)` is trend
- `s(t)` is seasonality
- `h(t)` is holidays or events
- `epsilon_t` is noise

This is a very useful conceptual model because it separates the main sources of structure.

### 15.1 Why Prophet Is Popular

According to the lecture:

- automatic changepoint detection
- built-in seasonality and holidays
- robustness to missing data and outliers
- user-friendly interface

This makes Prophet especially attractive for:

- business forecasting
- demand or sales planning
- fast prototyping

## 16. Changepoints

A useful Prophet-specific theme in the lecture is changepoint detection.

Changepoints are times where:

- trend slope changes
- trend direction changes
- the generating process shifts

This matters because many real-world series are not stable over long periods. A model that cannot react to structural change may fail badly.

## 17. Hybrid Models and Ensembles

The lecture mentions hybrid approaches such as:

- ARIMA plus machine learning

The intuition is good:

- use classical models for structured linear temporal patterns
- use ML models for residual non-linearity or rich feature interactions

This is a pragmatic approach rather than a purely ideological one.

The lecture also connects this to ensemble thinking:

- bagging
- boosting
- stacking

for time-series forecasting as well.

## 18. Other Advanced Methods

The lecture briefly surveys:

- state space models
- Kalman filters
- Gaussian processes
- transformer-based models

Students do not need to master all of them now, but they should understand the landscape:

- time series is a large field
- ARIMA is not the final answer
- modern forecasting often uses richer probabilistic or deep learning systems

## 19. Preprocessing in Time Series

This is one of the strongest practical sections of the lecture.

### 19.1 Missing Values

The slides mention:

- forward fill
- interpolation

This is a highly context-dependent choice.

Forward fill is often sensible when:

- the last observed state is still a meaningful approximation

Interpolation is more reasonable when:

- the series evolves smoothly between neighboring points

### 19.2 Differencing

The lecture emphasizes:

- first-order differencing
- seasonal differencing

These are key tools for removing:

- trend
- seasonal non-stationarity

### 19.3 Variance-Stabilizing Transformations

Examples:

- log transform
- square root transform

These help when:

- variance grows with level
- extreme values dominate the scale

### 19.4 Scaling

The lecture notes that scaling matters especially for ML models.

This is correct because:

- distance-based methods
- neural networks
- some optimization procedures

can be sensitive to feature scale.

## 20. Temporal Train-Test Splitting

The lecture strongly emphasizes preserving time order.

This is one of the most important practical rules in forecasting.

### Wrong Approach

- random train-test split

Why wrong:

- future information can leak into training
- evaluation becomes unrealistically optimistic

### Better Approaches

- simple time-based split
- rolling window validation
- sliding window validation

These reflect the real forecasting situation:

- train on the past
- predict the future

## 21. Rolling vs Sliding Window

The lecture names both approaches.

### Rolling Window

- training window expands over time

Use when:

- more historical data is expected to help

### Sliding Window

- training window size stays fixed while moving forward

Use when:

- older data may become stale
- recent behavior is more relevant

This is a very practical modeling choice in non-stationary environments.

## 22. Evaluation Metrics for Forecasting

The lecture highlights:

- MAE
- MSE
- RMSE
- MAPE

Each answers a slightly different question.

### 22.1 MAE

Average absolute error.

Why useful:

- easy to interpret
- less sensitive to extreme errors than MSE

### 22.2 MSE

Squares errors, so larger misses matter more.

Useful when:

- large forecasting mistakes are especially costly

### 22.3 RMSE

Square root of MSE.

Why useful:

- same units as the target
- still penalizes large errors strongly

### 22.4 MAPE

Measures relative error in percentage terms.

Main caveat from the lecture:

- not robust when actual values are near zero

This caveat is important and often forgotten.

## 23. Residual Diagnostics

The lecture also stresses that a good time-series model should leave residuals that look close to noise.

Checks mentioned:

- residual ACF
- Ljung-Box test
- residual normality
- Q-Q plots

This is valuable because forecasting is not just about one scalar metric. The residual structure tells us whether important temporal dependence remains unmodeled.

## 24. AIC and BIC

The lecture includes:

- Akaike Information Criterion
- Bayesian Information Criterion

These are model-selection criteria that trade off:

- goodness of fit
- model complexity

Lower values indicate a more favorable balance.

They are especially common in ARIMA/SARIMA model comparison.

## 25. Confidence and Prediction Intervals

The slides dedicate a separate section to forecast uncertainty.

This is excellent, because a point forecast alone is often not enough.

### Classical Models

ARIMA/SARIMA often provide prediction intervals from their own residual variance structure.

### Tree-Based Models

The lecture mentions:

- quantile regression forests
- bootstrapping

This is important because many ML models do not naturally output calibrated uncertainty intervals.

### Prophet

Prophet provides built-in prediction intervals.

Students should remember:

- wider intervals mean higher uncertainty
- interval calibration matters
- uncertainty estimation is part of model quality, not an optional extra

## 26. Advanced Practical Topics

The lecture touches several advanced but useful directions:

- complex seasonality handling
- multivariate time series
- anomaly detection
- changepoint detection
- SVD-style decomposition

### 26.1 Multivariate Time Series

When several related series interact:

- one variable may help forecast another

Examples:

- multiple economic indicators
- multiple sensors

This is more complex than univariate forecasting because cross-variable structure matters.

### 26.2 Time Series Anomaly Detection

The lecture mentions:

- statistical methods such as z-scores
- moving-average thresholds
- Isolation Forest
- autoencoders
- One-Class SVM

This is useful because not all time-series tasks are forecasting. Many practical tasks focus on unusual events.

### 26.3 Changepoint Detection

The lecture also gives techniques like:

- CUSUM
- Bayesian changepoint methods
- PELT

This is an important extension of the earlier Prophet discussion: time-series systems often change regime, and detecting that is itself a task.

### 26.4 SVD for Decomposition

The appendix discusses SVD for decomposition and dimensionality reduction.

This is more advanced, but the key idea is:

- decompose structured multivariate time-series data into lower-rank components
- reveal latent shared patterns
- reduce noise and complexity

## 27. Practical Notebook Map

### `Coding_TimSeries.ipynb`

This is the main practical benchmark notebook.

It compares:

- SARIMA
- Random Forest
- Prophet
- XGBoost

Dataset:

- Airline Passengers

Main topics:

- common data preparation
- time-based train-test split
- lag features for ML models
- multi-model comparison
- MAE / RMSE / MAPE
- residual diagnostics
- time-series cross-validation with confidence intervals

This notebook is especially useful because it places classical and ML models side by side.

### `TimeSeries_visualizations.ipynb`

This notebook is more of a visual laboratory.

It contains:

- synthetic series generation
- ACF/PACF illustrations
- ARIMA and SARIMA examples
- Prophet examples
- Bitcoin-style forecasting demos
- residual analysis examples

This is helpful for intuition building rather than only final benchmarking.

### `class_timeseries.ipynb`

This notebook appears to be a more exploratory or class-style project workflow using:

- Prophet
- ARIMA
- Random Forest
- tsfresh feature extraction

It is useful as an example of a broader applied workflow rather than a single isolated model demo.

### `no___TimeSeries Coding.ipynb`

This notebook is another forecasting comparison workflow using:

- ARIMA / SARIMA
- Prophet
- Random Forest
- XGBoost

Its main value is in showing an alternative practical setup on market-like data with lag features and model comparison.

## 28. Best Practices from the Lecture

The final best-practice section is worth keeping almost as a checklist.

- validate with time-aware schemes
- preserve temporal order
- check model assumptions
- incorporate domain knowledge
- monitor performance over time
- retrain when the process changes
- document preprocessing and model settings carefully

This is exactly the right operational mindset for forecasting work.

## 29. Key Takeaways

- Time series data is temporally ordered and dependent.
- Trend, seasonality, cyclicity, and noise are the core structural components.
- Stationarity is central for classical models like ARIMA.
- ADF and KPSS help diagnose stationarity from different null hypotheses.
- ACF and PACF are diagnostic tools for temporal dependence and AR/MA order selection.
- ARIMA and SARIMA are strong classical tools for linear, structured forecasting.
- ML models become powerful for time series once lagged and calendar features are engineered properly.
- Prophet is a practical additive forecasting model with trend, seasonality, holidays, and changepoints.
- Temporal train-test splitting is mandatory to avoid leakage.
- Forecasting quality includes both point accuracy and uncertainty calibration.

## 30. Quick Revision Questions

1. Why is random train-test splitting usually wrong for time series?
2. What is the difference between trend, seasonality, and cyclicity?
3. What is weak stationarity, and why does ARIMA care about it?
4. How do ADF and KPSS differ in their null hypotheses?
5. What roles do ACF and PACF play in ARIMA model identification?
6. Why can lag features turn a forecasting problem into a supervised-learning problem?
7. When would Prophet be more convenient than ARIMA?
8. Why is MAPE dangerous when values are near zero?
9. What does it mean if residuals still show autocorrelation after fitting a model?
10. Why are confidence intervals important in forecasting, not just point predictions?
