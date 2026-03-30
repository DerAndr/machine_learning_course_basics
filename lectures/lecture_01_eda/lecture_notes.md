# Lecture 01 Recap: Exploratory Data Analysis

> Lecture number: 01
> Lecture slug: `lecture_01_eda`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What EDA Is

Exploratory Data Analysis (EDA) is the process of examining a dataset before formal modeling or hypothesis testing.

The lecture defines EDA as the stage where we:

- summarize the main characteristics of the data
- use visualizations to understand patterns
- detect anomalies and unusual values
- study distributions
- inspect relationships between variables

The key idea is that EDA comes **before** machine learning, not after it.

You do EDA first because you need to understand:

- what your variables mean
- which variables are numerical or categorical
- whether there are missing values
- whether the target has extreme values or skewed behavior
- which features seem useful

## 2. Why EDA Matters

The lecture presents EDA as the bridge between **raw data** and **conclusions**.

According to the slides, EDA helps with:

- descriptive analysis
- adjustment of variable types
- detection and treatment of missing data
- identification of atypical data
- correlation analysis

That means EDA is not just “making plots.” It is the stage where you reduce uncertainty before building a model.

Without EDA, you risk:

- trusting bad columns
- ignoring missing values
- failing to notice outliers
- choosing the wrong preprocessing pipeline
- building a model on weak assumptions

## 3. EDA in the Data Analytics Pipeline

The lecture connects EDA to **CRISP-DM**.

EDA belongs to the **Data Understanding** stage.

This matters because machine learning is not just:

1. load data
2. fit model
3. evaluate score

Instead, a good project starts with understanding:

- the business or domain context
- the dataset structure
- the quality of the data
- the relationships that may matter later

So EDA is part of the professional workflow, not an optional academic step.

## 4. Main Goals of EDA

The slides list several core objectives. These are worth remembering because they explain why we use so many different statistics and plots.

### Understand Data Structure

This means:

- how many rows and columns exist
- which columns are numerical
- which columns are categorical
- how the target variable is represented

In practice, this is why the notebooks begin with:

- loading the Ames Housing dataset
- reading the field descriptions
- using `df.info()`
- printing `df.describe()`

### Spot Patterns and Relationships

EDA should reveal:

- trends
- correlations
- clusters
- suspicious feature interactions

This becomes visible later in the notebooks through:

- scatter plots
- heatmaps
- pair plots
- grouped distribution plots

### Identify Outliers and Anomalies

Outliers matter because they can:

- distort averages
- dominate scale-sensitive plots
- weaken model stability

In practice, boxplots and violin plots make this visible quickly.

### Assess Data Distribution

The lecture emphasizes that a variable may be:

- approximately normal
- skewed
- multi-modal

This matters because distribution shape affects:

- summary statistics
- transformations
- model assumptions

### Guide Feature Engineering

EDA is already the beginning of feature engineering.

For example:

- if a variable is heavily skewed, you may later transform it
- if a category is too sparse, you may group levels
- if variables are strongly related, you may select or combine them more carefully

### Handle Missing Data

The lecture repeatedly treats missing data as a first-class problem.

This is correct because missingness is not just a cleaning inconvenience. It can change model behavior and interpretation.

### Reduce Errors and Support Model Choice

EDA reduces wrong assumptions.

If you understand the structure of the data, you are more likely to:

- choose sensible features
- avoid leakage
- pick suitable models
- interpret performance correctly

## 5. Types of Data

The lecture highlights the importance of recognizing data type before choosing any analysis tool.

The main types are:

- numerical
- categorical
- ordinal
- time-series

This is a basic but critical point.

If you confuse data types, your EDA becomes misleading. For example:

- using mean on pure categories is usually meaningless
- using counts for continuous numerical variables hides distribution shape
- treating ordered categories like unordered labels loses information

## 6. Numerical Data: Core Statistics

The lecture organizes numerical summaries into three groups:

- central tendency
- spread
- distribution shape

### 6.1 Central Tendency

The main measures are:

- **Mean**: the average value
- **Median**: the middle value after sorting
- **Mode**: the most frequent value

These measures answer slightly different questions:

- the **mean** uses all values and is sensitive to extreme observations
- the **median** is robust and is often more stable for skewed variables
- the **mode** is useful when repeated values or dominant categories matter

So, when a distribution is skewed, the median is often a better summary of the “typical” value than the mean.

Why this matters:

- the mean is sensitive to extreme values
- the median is more robust to outliers
- the mode is useful when repeated values matter

The lecture also stresses the ideal case of **zero skewness**, where:

- mean = median = mode

That is useful as a reference point. Real data often departs from this pattern.

### 6.2 Spread

The main measures are:

- **Variance**
- **Standard deviation**

These describe how far values are spread around the center.

Technically:

- **variance** is the average squared deviation from the mean
- **standard deviation** is the square root of variance, so it is easier to interpret because it is expressed in the original units

If the standard deviation is large, values vary widely around the center. If it is small, the variable is more concentrated.

If spread is large, the average alone tells only part of the story.

In the practical notebooks, this idea shows up when students inspect `SalePrice` and other housing variables. A central value is not enough; students also need to see how widely values vary.

### 6.3 Distribution Shape

The lecture focuses on:

- **Skewness**
- **Kurtosis**

#### Skewness

Skewness tells you whether a distribution is symmetric or tilted.

- **Positive skewness**: long right tail
- **Negative skewness**: long left tail
- **Zero skewness**: ideally symmetric

The slides describe:

- positive skewness as `Mode < Median < Mean`
- negative skewness as `Mean < Median < Mode`

This is a very useful memory rule.

Practical interpretation:

- strong positive skew often appears when a variable has many moderate values and a few very large ones
- strong negative skew is less common in many business datasets but can appear when there is a hard upper bound and a long lower tail

For many ML tasks, strong skewness suggests that a transformation may later be useful.

#### Kurtosis

Kurtosis measures tail heaviness.

The lecture introduces:

- **Mesokurtic**: close to normal
- **Leptokurtic**: sharper peak, heavier tails
- **Platykurtic**: flatter peak, lighter tails

The lecture also explains why **heavy tails** matter:

- they mean extreme values are more likely
- they often imply more outlier risk

So kurtosis is not only about the “peak” of the distribution. In practice, students should think of it as a warning sign for unusual extremes.

This is especially important when students later work with unstable or noisy real-world data.

### 6.4 Calculation Details Matter

One of the strongest theoretical points in the lecture is the warning:

- pay attention to how statistics are calculated

Two specific examples appear in the slides:

- `bias=True`
- `fisher=True`

The lecture explains that:

- bias correction may change the result
- Fisher’s kurtosis subtracts `3` so that a normal distribution has kurtosis `0`

This is important because students often assume that every library computes the exact same thing. That is not always true.

## 7. Categorical Data: Core Statistics

For categorical variables, the lecture switches to a different set of summaries:

- frequency count
- proportion / percentage
- mode
- unique count
- missing-value count

These are the correct questions for category-based features.

Each of them tells something different:

- **frequency count** tells you the raw size of each category
- **proportion** tells you how dominant a category is relative to the whole dataset
- **mode** gives the most common category
- **unique count** tells you how many distinct values the column has
- **missing-value count** tells you whether category information is incomplete

In practice, the notebooks inspect object-type columns using descriptive summaries and unique-value inspection. This helps students see:

- which categories dominate
- how many unique groups exist
- whether some columns have missing values

## 8. Practical Dataset: Ames Housing

The practical notebooks are centered on **Ames Housing**.

This is a strong dataset for EDA because it contains:

- mixed data types
- many explanatory features
- a meaningful target variable: `SalePrice`
- real variation in quality, size, and neighborhood information

The notebooks begin with basic inspection:

- load the dataset
- inspect the description file
- display sample rows
- convert one row to dictionary form
- run `df.info()`
- run `df.describe()` on different column types

This first block is not just setup. It is already EDA.

Students should notice that the practical flow matches the theory exactly:

- inspect structure first
- do not jump straight into plots

## 9. Univariate Analysis

The lecture defines univariate analysis as studying **one variable at a time**.

Its goals are:

- understand distribution
- understand center and spread
- detect anomalies
- support cleaning decisions

In the practice notebooks, the main univariate variable is `SalePrice`.

### 9.1 Quantile Analysis

Theory:

- quantiles divide the distribution into percent-based sections
- they help identify median, lower quartiles, upper quartiles, and thresholds

Important technical idea:

- Q1 is the 25th percentile
- Q2 is the median, or 50th percentile
- Q3 is the 75th percentile

The range between Q1 and Q3 is the **interquartile range (IQR)**.

Practice:

- the notebook computes quantiles for `SalePrice`
- this helps students understand how house prices are distributed across the lower, middle, and upper parts of the market

Why it matters:

- quantiles are a direct way to reason about expensive vs inexpensive observations
- they prepare students for boxplots and outlier logic

This is especially important in housing data, where a few very expensive houses can distort averages.

### 9.2 Histogram

Theory:

- a histogram shows frequency by bins
- it is one of the most basic tools for seeing shape

The slide says students should look for:

- bell-shaped patterns
- skewness
- multiple peaks

Practice:

- the notebook builds histograms using both Matplotlib and Seaborn
- this lets students see the same variable in slightly different plotting styles

Main lesson:

- a histogram is often your first serious look at a numerical variable

Students should also remember that histogram interpretation depends on the chosen bin size:

- too few bins can hide structure
- too many bins can create noisy-looking patterns

### 9.3 KDE

Theory:

- KDE is a non-parametric estimate of the probability density function
- it gives a smooth curve instead of discrete bins

Practice:

- the notebook computes KDE with SciPy and also uses Seaborn’s built-in density support
- students can compare the smoothed density with the histogram

Why this matters:

- a histogram depends on chosen bin width
- KDE gives another perspective on shape

But KDE should still be interpreted carefully:

- it is an estimate
- smoothing can hide local structure or make noise look like signal

### 9.4 ECDF

Theory:

- ECDF shows cumulative probability
- it is useful for percentiles and rank understanding

Practice:

- the notebook computes an ECDF-like cumulative view of `SalePrice`

Why this matters:

- ECDF helps answer questions like “what proportion of houses cost less than X?”

Compared to a histogram, ECDF is often better when you care about thresholds and percentiles rather than visual shape only.

### 9.5 Boxplot

Theory:

- a boxplot shows min, Q1, median, Q3, and max
- it highlights possible outliers

Practice:

- the notebook draws boxplots for `SalePrice`
- students see which values sit far outside the interquartile range

Why this matters:

- boxplots are compact and useful for fast anomaly inspection

Technical detail:

- many boxplots flag outliers using the rule `Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR`

This does not automatically mean those points are “bad data.” It means they deserve inspection.

### 9.6 Violin Plot

Theory:

- violin plots combine distribution density with summary structure

Practice:

- the notebook plots `SalePrice` with violin plots
- one version also marks mean, median, and mode

Why this is useful:

- students can compare the “shape” story from KDE with the summary story from boxplots

## 10. Bivariate Analysis

Bivariate analysis studies **two variables together**.

The lecture says this stage is about:

- correlation
- dependency
- trend
- interaction

It is especially important for predictive modeling because many models depend on relationships between variables, not just individual variable summaries.

### 10.1 Scatter Plot

Theory:

- scatter plots are used for two numerical variables
- they reveal positive trend, negative trend, clusters, and outliers

Practice:

- the notebook studies `SalePrice` versus `Gr Liv Area`

This is a very good example because it is interpretable:

- larger living area often means higher price
- but the relationship is not perfect
- unusual houses stand out visually

### 10.2 Regression Line and Correlation

Theory:

- a regression line helps summarize the direction of a relationship
- correlation quantifies its linear strength

Practice:

- the notebook uses `sns.regplot`
- it also computes Pearson correlation and R-squared

This is an important step for students because it shows:

- a plot can suggest a relationship
- correlation gives a numerical measure
- regression adds a simple trend explanation

Technical reminder:

- **Pearson correlation** measures linear association
- a high correlation does not prove causation
- a low Pearson correlation does not mean there is no relationship; the relationship may simply be non-linear

### 10.3 Correlation Matrix / Heatmap

Theory:

- a correlation matrix compares many pairs of numerical variables at once
- values range from `-1` to `1`

Practice:

- the notebook computes a heatmap over selected numerical Ames variables
- it also uses a masked triangular version

Why this matters:

- students can quickly identify which features move together
- they can look for features strongly related to `SalePrice`

But students should also remember:

- correlation does not prove causation
- correlation only captures some types of relationships, especially linear ones

This is why correlation heatmaps are useful for screening, but not sufficient for final interpretation on their own.

### 10.4 Cross-Tabulation

Theory:

- cross-tabulation is for categorical variables
- it shows counts of category combinations

Practice:

- the notebook builds a cross-tab between variables such as `Overall Qual` and `Neighborhood`
- the result is visualized as a heatmap

This teaches students that not all relationship analysis is numeric-to-numeric.

It also introduces an important habit:

- counts should be interpreted in context
- large cells may reflect real association, but they may also reflect overall class imbalance

### 10.5 Joint Plot

Theory:

- a joint plot combines the relation between two variables with their marginal distributions

Practice:

- the notebook uses joint plots for `SalePrice` and `Gr Liv Area`

This is useful because students see:

- the central relationship
- the individual distribution of each variable

### 10.6 Hexbin Plot

Theory:

- hexbin is useful when scatter plots become crowded

Practice:

- the notebook uses hexbin-style plots for the same relationship

Why this matters:

- dense datasets can hide patterns through overplotting
- hexbin solves that by showing density instead of raw point overlap

### 10.7 Bar Plot

Theory:

- bar plots are useful when comparing categories against a numerical measure

Practice:

- the notebook uses grouped comparisons involving housing-related categories

This helps students learn when a categorical view is more appropriate than a scatter plot.

### 10.8 Pair Plot

Theory:

- pair plots show scatter plots for all pairs of selected numerical variables

Practice:

- the notebook builds pair plots for selected Ames features

Why this matters:

- pair plots are a fast way to inspect many possible relationships at once
- they are especially good for early exploration

Their main limitation is scale:

- they become visually heavy when too many variables are included
- they work best on a selected subset of features

## 11. Multivariate Analysis

The lecture defines multivariate analysis as studying **more than two variables at once**.

This is where students move from simple relationships to more realistic structure.

The lecture emphasizes that:

- complex patterns may be invisible in simpler plots
- multivariate thinking supports better modeling and better interpretation

### 11.1 Combined Numerical and Categorical Views

Practice notebooks include grouped summaries where students see how numerical behavior changes across categories.

This is important because many real insights are conditional:

- a feature may matter differently in different groups

### 11.2 3D Scatter Plot

Theory:

- a 3D scatter plot extends the relationship view to three variables

Practice:

- the notebook uses Plotly to create interactive 3D visualizations

This is useful because students see that multivariate EDA often benefits from interactivity.

### 11.3 Violin Plot for Multivariate Cases

Theory:

- violin plots can be adapted to compare grouped distributions

Practice:

- the notebook uses violin-style grouped plots to show richer comparisons across categories

## 12. Automation in EDA

The second notebook block moves into automated EDA.

The lecture does not present automation as magic. Instead, it explains both the value and the limitations.

### 12.1 Why Automate EDA

The slides list three benefits:

- saves time and effort
- gives quick insight
- makes reporting more consistent

This is especially helpful when:

- datasets are large
- students need a first overview quickly
- multiple datasets must be compared

### 12.2 `ydata-profiling`

Theory:

- creates automated descriptive reports
- includes data types, missing values, quantiles, distributions, and correlation summaries

Practice:

- the notebook installs and runs `ydata-profiling`
- a sample of the Ames dataset is used to generate a report

This helps students see how quickly a broad overview can be generated.

From a workflow perspective, this is useful when you need an initial audit of:

- variable types
- missingness
- suspicious columns
- obvious correlations

### 12.3 `Sweetviz`

Theory:

- useful for comparison-oriented reports
- supports target analysis

Practice:

- the notebook runs Sweetviz on a sampled version of the dataset

The main learning point is:

- automated tools can speed up inspection, especially when comparing datasets or subsets

This becomes especially valuable later when students compare training and validation data.

### 12.4 `D-Tale`

Theory:

- interactive exploration interface for pandas DataFrames

Practice:

- the notebook launches D-Tale in notebook mode

This shows students that EDA can also be interactive and browsing-oriented, not only plot-oriented.

That is useful when students need to inspect rows, categories, filters, and local patterns manually.

### 12.5 Why Automation Is Not Enough

The lecture gives a strong warning:

- automated tools lack context
- they may miss subtle problems
- they can encourage passive analysis

This is one of the most important takeaways from the whole lecture.

Correct mindset:

- use automated tools as a starting point
- always follow up with manual inspection

## 13. Visualization Tools and Resources

The lecture also broadens the discussion beyond classic Matplotlib charts.

### Why Visualization Matters

The slides emphasize that visualization:

- improves understanding
- improves communication
- supports storytelling
- helps decision-making

This is why EDA is so visual. A good chart can reveal structure faster than a table of numbers.

But the lecture’s implicit standard is also important:

- a plot should answer a question
- visual clarity matters more than decoration

### Storytelling With Matplotlib

The lecture highlights the importance of:

- titles
- labels
- legends
- annotations
- color choices

This matters because a plot is not only for the analyst. It is also for communication.

### Accessibility

The lecture explicitly mentions color-blind-friendly color schemes.

This is a strong practical point:

- a good visualization should be readable, not just pretty

### Advanced Visualization Libraries

The lecture introduces:

- **Seaborn** for statistical graphics
- **Plotly** for interactive and web-based visualizations
- **Altair** for declarative interactive plotting
- **Bokeh** for large-scale and real-time interactive visualizations
- **ggplot for Python** for layered grammar-style plotting

The practical notebook demonstrates several of these tools directly, so students see that the slide content is not just theoretical.

Students do not need to master all libraries immediately. The main point is to understand that tool choice depends on the task:

- quick statistical EDA
- interactive exploration
- dashboards
- presentation-quality visuals

## 14. Common EDA Challenges

The lecture ends with common problems that EDA must reveal early:

- missing data
- inconsistent data formats
- outliers
- high dimensionality
- data imbalance

Each of these can damage later modeling if ignored.

For example:

- missing data can create biased summaries
- inconsistent units can make columns incomparable
- outliers can dominate scaling and regression
- high dimensionality can make interpretation difficult
- imbalance can distort evaluation and learning behavior

The slides also point forward to later lectures, especially Data Preparation, where these issues will be handled more deeply.

## 15. How the Practical Notebooks Fit the Theory

### `EDA_1-4.ipynb`

This notebook is the manual EDA core.

It shows students how to:

- inspect the Ames Housing dataset
- compute basic descriptive statistics
- analyze one variable at a time
- study relationships between pairs of variables
- extend analysis to multiple variables

### `EDA_5-6.ipynb`

This notebook expands the lecture into:

- automated EDA reports
- alternative visualization libraries
- interactive EDA tools

Together, the two notebooks follow the lecture structure very closely:

1. basic understanding of data
2. univariate analysis
3. bivariate analysis
4. multivariate analysis
5. automation
6. advanced visualization tools

## 16. Final Takeaways

The lecture recap slide summarizes the core message well:

- EDA helps you understand data before modeling
- EDA helps you make better decisions
- EDA prevents analytical mistakes
- EDA is iterative, not one-time

Students should leave this lecture with the mindset that EDA is an ongoing analytical habit.

You do not perform EDA once and forget it. You return to it whenever:

- new variables appear
- preprocessing changes the data
- model results look suspicious
- you need to explain patterns more clearly

That is why EDA is both a technical skill and a reasoning habit.

## 17. Quick Revision Questions

After reviewing this lecture, you should be able to explain:

- What is EDA and why does it happen before modeling?
- Why do data types matter so much in EDA?
- What is the difference between center, spread, and shape?
- How do skewness and kurtosis help describe numerical data?
- When should you use histogram, KDE, ECDF, boxplot, or violin plot?
- What is the difference between scatter plots, heatmaps, cross-tabs, and pair plots?
- Why is automated EDA useful, and why is it not enough on its own?
- How do the Ames Housing notebooks illustrate the lecture concepts in practice?
