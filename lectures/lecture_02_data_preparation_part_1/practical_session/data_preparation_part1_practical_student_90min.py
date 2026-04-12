# /// script
# source-notebook = "data_preparation_part1_practical_student_90min.ipynb"
# generated-by = "Codex notebook export"
# ///

# %% [markdown]
# # Data Preparation Part 1: Practical Session - STUDENT VERSION (90 minutes)
# 
# **Learning Objectives:**
# - diagnose missingness and argue about plausible missing-data mechanisms
# - measure how imputation and log transforms change a distribution
# - compare univariate and multivariate outlier-detection ideas
# - practice ordinal reasoning, one-hot encoding intuition, quantile binning, and robust scaling
# - assemble a small preprocessing pipeline before fitting a regression model
# 
# This notebook uses targeted TODO placeholders while keeping one shared classroom flow across the main preprocessing topics from Lecture 02.

# %% [markdown]
# ## Setup

# %% [markdown]
# ## Setup Note
# 
# ```python
# # If needed:
# # pip install -U numpy pandas matplotlib seaborn scikit-learn liac-arff
# ```
# 
# This practical uses the Ames Housing dataset from **OpenML dataset `41211`**. We immediately rename the OpenML columns into the course's space-separated Ames style so the practical stays aligned with the rest of the course.

# %%
# NOTE: notebook magic commented for local script use: !pip install -U numpy pandas matplotlib seaborn scikit-learn liac-arff

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("✓ Libraries loaded successfully!")

# %% [markdown]
# ## Shared Helper Functions
# 
# These helper utilities keep the practical focused on interpretation and preprocessing decisions instead of repeating the same plotting boilerplate in every block.

# %%
def plot_distribution(series, title, xlabel, bins=40, kde=True, color='steelblue'):
    """Show a histogram and boxplot together for a single feature."""
    clean = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios': [4, 1]})

    sns.histplot(clean, bins=bins, kde=kde, color=color, ax=axes[0])
    axes[0].axvline(clean.median(), color='crimson', linestyle='--', linewidth=2, label=f"Median = {clean.median():.1f}")
    axes[0].axvline(clean.mean(), color='black', linestyle=':', linewidth=2, label=f"Mean = {clean.mean():.1f}")
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    sns.boxplot(x=clean, color=color, ax=axes[1])
    axes[1].set_title('Boxplot')
    axes[1].set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()


def plot_imputation_impact(original, imputed, left_label, right_label, title, xlabel, bins=40):
    """Show imputation impact with separate histograms, boxplots, and missing counts."""
    missing_before = int(original.isna().sum())
    missing_after = int(imputed.isna().sum())

    if missing_before == 0:
        print(
            "The original series already has no missing values. "
            "For this comparison, keep the raw column unchanged and create a separate imputed copy."
        )

    original_clean = original.dropna()
    imputed_clean = imputed.dropna()
    combined = pd.concat([original_clean, imputed_clean], ignore_index=True)
    shared_bins = np.histogram_bin_edges(combined, bins=bins)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    sns.histplot(original_clean, bins=shared_bins, kde=True, color='steelblue', ax=axes[0, 0])
    axes[0, 0].axvline(original_clean.median(), color='crimson', linestyle='--', linewidth=2)
    axes[0, 0].set_title(f'{left_label} distribution')
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel('Frequency')

    sns.histplot(imputed_clean, bins=shared_bins, kde=True, color='darkorange', ax=axes[0, 1])
    axes[0, 1].axvline(imputed_clean.median(), color='crimson', linestyle='--', linewidth=2)
    axes[0, 1].set_title(f'{right_label} distribution')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel('Frequency')

    sns.boxplot(
        data=pd.concat([original_clean.rename(left_label), imputed_clean.rename(right_label)], axis=1),
        orient='h',
        palette=['steelblue', 'darkorange'],
        ax=axes[1, 0],
    )
    axes[1, 0].set_title('Side-by-side boxplots')
    axes[1, 0].set_xlabel(xlabel)

    missing_counts = pd.Series(
        {
            f'{left_label}\nmissing': missing_before,
            f'{right_label}\nmissing': missing_after,
        }
    )
    sns.barplot(x=missing_counts.index, y=missing_counts.values, palette=['steelblue', 'darkorange'], ax=axes[1, 1])
    axes[1, 1].set_title('Missing values before vs after')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xlabel('')

    plt.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_before_after_transform(original, transformed, original_label, transformed_label, xlabel_original, xlabel_transformed, bins=40):
    """Compare a feature before and after a transformation on separate scales."""
    original_clean = original.dropna()
    transformed_clean = transformed.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    sns.histplot(original_clean, bins=bins, kde=True, color='steelblue', ax=axes[0, 0])
    axes[0, 0].set_title(original_label)
    axes[0, 0].set_xlabel(xlabel_original)
    axes[0, 0].set_ylabel('Frequency')

    sns.histplot(transformed_clean, bins=bins, kde=True, color='darkorange', ax=axes[0, 1])
    axes[0, 1].set_title(transformed_label)
    axes[0, 1].set_xlabel(xlabel_transformed)
    axes[0, 1].set_ylabel('Frequency')

    sns.boxplot(x=original_clean, color='steelblue', ax=axes[1, 0])
    axes[1, 0].set_title(f'{original_label} boxplot')
    axes[1, 0].set_xlabel(xlabel_original)

    sns.boxplot(x=transformed_clean, color='darkorange', ax=axes[1, 1])
    axes[1, 1].set_title(f'{transformed_label} boxplot')
    axes[1, 1].set_xlabel(xlabel_transformed)

    plt.tight_layout()
    plt.show()


def plot_top_missing_counts(missing_summary, top_n=10):
    """Plot the top missing-value counts for quick inspection."""
    top_missing = missing_summary.head(top_n).sort_values(ascending=True)
    if top_missing.empty:
        print('No missing values detected after parsing the dataset.')
        return
    plt.figure(figsize=(8, 5))
    top_missing.plot(kind='barh', color='indianred')
    plt.title('Top columns by missing-value count')
    plt.xlabel('Missing values')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def load_ames_housing_openml():
    """Load OpenML 41211 and rename columns into the course's Ames style."""
    ames = fetch_openml(data_id=41211, as_frame=True, parser='auto')

    if getattr(ames, 'frame', None) is not None:
        df = ames.frame.copy()
    else:
        target = ames.target.rename('Sale_Price') if hasattr(ames.target, 'rename') else pd.Series(ames.target, name='Sale_Price')
        df = pd.concat([ames.data.copy(), target], axis=1)

    rename_map = {column: column.replace('_', ' ') for column in df.columns}
    rename_map.update(
        {
            'Sale_Price': 'SalePrice',
            'Year_Sold': 'Yr Sold',
            'First_Flr_SF': '1st Flr SF',
            'Second_Flr_SF': '2nd Flr SF',
            'Three_season_porch': '3Ssn Porch',
        }
    )
    df = df.rename(columns=rename_map)

    # OpenML can surface missing categorical values as literal sentinel strings.
    # Normalize them up front so the rest of the practical sees true NaNs.
    df = df.replace(
        {
            '?': np.nan,
            'NA': np.nan,
            'NaN': np.nan,
            'nan': np.nan,
            'None': np.nan,
        }
    )

    for column in df.select_dtypes(include=['object', 'string']).columns:
        df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)

    return df

# %% [markdown]
# ## How To Work In Teams
# 
# 1. **Group A** works on **Section 1**: Missing Values and Imputation.
# 2. **Group B** works on **Sections 2 and 3**: Outliers, Transformations, Encodings, and Binning.
# 3. **Group C** works on **Section 4**: Feature Engineering, Scaling, and Pipelines.
# 4. At the end, each group reports the main numeric answers from its block and one preprocessing lesson that mattered for modeling.
# 
# **Important:**
# - You do **not** need to finish the whole notebook during class.
# - Focus on the TODO cells inside your assigned block first.
# - Keep intermediate objects like `df_work`, `preprocess`, and engineered features because later tasks reuse them.

# %% [markdown]
# ## 1. Missing Values and Imputation (⏱️ ~20 min)
# 
# **Scenario:** You are preparing the Ames Housing dataset for a downstream price-prediction workflow.
# 
# This first block focuses on data loading, missingness reasoning, and the effect of a simple but defensible imputation strategy.

# %% [markdown]
# ### 1.1 Load and Inspect ✏️ TODO (⏱️ ~5 min)
# 
# **Instructions:**
# Load the Ames Housing dataset into a DataFrame named `df`.
# - Source: OpenML dataset `41211`
# - Use the helper `load_ames_housing_openml()`
# - Keep the course-style column names with spaces such as `Lot Frontage`, `1st Flr SF`, and `Yr Sold`
# 
# **Calculation 1:** What is the exact number of rows and columns in the raw dataset?

# %%
# TODO:
# 1. Load the dataset with load_ames_housing_openml().
# 2. Save it to a DataFrame named df.
# 3. Print the shape and display the first rows.

# df = ...

# %% [markdown]
# **Answer 1:** [Student provides rows/columns].

# %% [markdown]
# ### 1.2 Quick Missingness Scan (Pre-filled)
# 
# Before deciding how to treat `Lot Frontage`, take a quick look at where missing values are concentrated. In Ames Housing, some high-missing columns are often better interpreted as **feature not present** rather than **bad data collection**. That distinction matters: a categorical field like `Pool QC` may be structurally absent, while a numeric field like `Lot Frontage` is usually a genuinely missing measurement.

# %%
missing_summary = (
    df.isna().sum()
    .sort_values(ascending=False)
)
missing_summary = missing_summary[missing_summary > 0]

print('Top columns by missing-value count:')
print(missing_summary.head(10))
plot_top_missing_counts(missing_summary)

top_missing_cols = missing_summary.head(8).index.tolist()
if top_missing_cols:
    plt.figure(figsize=(10, 4))
    sns.heatmap(
        df[top_missing_cols].isna(),
        cbar=False,
        cmap=sns.color_palette('gray_r', as_cmap=True),
        yticklabels=False,
    )
    plt.title('Missingness Pattern for the Top Missing Columns')
    plt.xlabel('Columns')
    plt.tight_layout()
    plt.show()
else:
    print('No missing values found in this dataset.')

lot_frontage_missing_rate = (
    df.assign(lot_frontage_missing=df['Lot Frontage'].isna())
    .groupby('Neighborhood')['lot_frontage_missing']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

if lot_frontage_missing_rate.sum() > 0:
    plt.figure(figsize=(8, 5))
    lot_frontage_missing_rate.sort_values().plot(kind='barh', color='slateblue')
    plt.title('Neighborhoods with the highest Lot Frontage missing rate')
    plt.xlabel('Missing-rate share')
    plt.ylabel('Neighborhood')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()
else:
    print('Lot Frontage has no missing values after parsing, so the neighborhood missing-rate plot is skipped.')

# %% [markdown]
# ### 1.3 Quantify Missingness ✏️ TODO (⏱️ ~6 min)
# **Calculation 2:** What is the exact number of missing values in the `Lot Frontage` feature? Based on the quick scan above, which missingness mechanism (MCAR, MAR, MNAR) seems most plausible here, and why is this different from a clearly "not applicable" field such as `Pool QC`?

# %%
# TODO:
# 1. Count missing values in df['Lot Frontage'].
# 2. Print the count.
# 3. In the answer cell, argue which missingness mechanism seems most plausible.

# %% [markdown]
# **Answer 2:** [Number] missing values. Likely mechanism: [MCAR / MAR / MNAR + one-sentence justification]. Also note why this is different from a structurally absent field such as `Pool QC`.

# %% [markdown]
# ### 1.4 The Impact of Imputation ✏️ TODO (⏱️ ~9 min)
# If you apply Median Imputation to `Lot Frontage`, it changes the distribution. Let's quantify how much.
# 
# **Calculation 3:** Create a copy of the dataframe `df_work = df.copy()`. Impute `Lot Frontage` using the **column median computed on the full dataframe used in this practical**. Store the result in a **new column** called `Lot Frontage_imputed` and keep the original `Lot Frontage` unchanged for comparison. What is the exact new overall mean of `Lot Frontage_imputed` across all 2930 rows after imputation (rounded to 2 decimal places)?

# %%
# TODO:
# 1. Create df_work = df.copy().
# 2. Compute the median of Lot Frontage.
# 3. Create Lot Frontage_imputed with fillna(median) and do not overwrite the original Lot Frontage column.
# 4. Compute and print the new overall mean rounded to 2 decimals.

# %% [markdown]
# **Answer 3:** New mean after imputation is [Value].

# %% [markdown]
# #### Real-World Note
# 
# For this classroom calculation, we use the full dataframe so everyone gets the same numeric answer. In a real ML workflow, imputation statistics such as the median must be fit on the training split only and then reused on validation/test data to avoid leakage.

# %%
plot_imputation_impact(
    df['Lot Frontage'],
    df_work['Lot Frontage_imputed'],
    left_label='Original',
    right_label='Median-imputed',
    title='Lot Frontage: Original vs Median-Imputed',
    xlabel='Lot Frontage (feet)'
)

# %% [markdown]
# ## 2. Outliers and Transformations (⏱️ ~25 min)
# 
# This block turns skewed numeric features and unusual observations into measurable preprocessing choices rather than vague intuition.

# %% [markdown]
# ### 2.1 IQR Thresholds ✏️ TODO (⏱️ ~6 min)
# The `SalePrice` feature in real estate is famously right-skewed.
# 
# **Calculation 4:** Calculate the Interquartile Range (IQR) for `SalePrice`. Based on the Tukey rule ($Q3 + 1.5 \times IQR$), what is the exact mathematical maximum boundary for `SalePrice` before a house is deemed an outlier? How many houses in the dataset exceed this limit?

# %%
# TODO:
# 1. Compute Q1, Q3, and the IQR for SalePrice.
# 2. Compute the Tukey upper bound: Q3 + 1.5 * IQR.
# 3. Count how many houses exceed that boundary.
# 4. Print the boundary and the count.

# %% [markdown]
# **Answer 4:** Upper boundary is [Amount]. Outlier count is [Count].

# %% [markdown]
# ### 2.2 Stabilizing Skewness ✏️ TODO (⏱️ ~8 min)
# Because of these massive multi-million dollar outliers, we want to apply a log transformation.
# 
# **Calculation 5:** Apply the transformation `np.log1p()` to the `SalePrice` column. After applying this transformation, what is the exact maximum value of this column rounded to 2 decimal places?

# %%
plot_distribution(
    df_work['SalePrice'],
    title='Distribution of SalePrice',
    xlabel='SalePrice'
)

saleprice_scatter = df_work[['Gr Liv Area', 'SalePrice']].dropna()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=saleprice_scatter.sample(n=min(len(saleprice_scatter), 1200), random_state=RANDOM_STATE),
    x='Gr Liv Area',
    y='SalePrice',
    alpha=0.45,
    s=24,
)
plt.title('SalePrice vs Gr Liv Area')
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.tight_layout()
plt.show()

skewness = df_work['SalePrice'].skew()
kurtosis = df_work['SalePrice'].kurt()
print(f"Skewness of SalePrice: {skewness:.2f}")
print(f"Kurtosis of SalePrice: {kurtosis:.2f}")

# %% [markdown]
# #### Pre-filled Illustration: Mean vs Median vs Mode on `SalePrice`
# 
# On a right-skewed variable, these three statistics do different jobs:
# - `mean` gets pulled toward the long right tail
# - `median` stays closer to the center of the bulk of the data
# - `mode` marks the most common local value
# 
# That is one reason median-based preprocessing is often more robust than mean-based preprocessing when strong outliers are present.

# %%
saleprice_mode = df_work['SalePrice'].mode().iloc[0]
saleprice_mean = df_work['SalePrice'].mean()
saleprice_median = df_work['SalePrice'].median()

stats_table = pd.DataFrame(
    {
        'statistic': ['mean', 'median', 'mode'],
        'value': [saleprice_mean, saleprice_median, saleprice_mode],
    }
)
display(stats_table)

plt.figure(figsize=(10, 5))
sns.histplot(df_work['SalePrice'].dropna(), bins=45, kde=True, color='steelblue')
plt.axvline(saleprice_mean, color='black', linestyle=':', linewidth=2, label=f"Mean = {saleprice_mean:,.0f}")
plt.axvline(saleprice_median, color='crimson', linestyle='--', linewidth=2, label=f"Median = {saleprice_median:,.0f}")
plt.axvline(saleprice_mode, color='darkgreen', linestyle='-.', linewidth=2, label=f"Mode = {saleprice_mode:,.0f}")
plt.title('SalePrice: mean vs median vs mode')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Interpretation Note
# 
# A strongly right-skewed target often benefits from a log transform because it compresses the long upper tail and makes large-value differences less dominant.

# %%
# TODO:
# 1. Create a log-transformed SalePrice column with np.log1p.
# 2. Compute the maximum transformed value.
# 3. Print it rounded to 2 decimals.

# df_work['SalePrice_Log'] = ...

# %% [markdown]
# **Answer 5:** Maximum logged value is [Value].

# %%
if 'SalePrice_Log' in df_work.columns:
    plot_before_after_transform(
        df_work['SalePrice'],
        df_work['SalePrice_Log'],
        original_label='Original SalePrice',
        transformed_label='Log-transformed SalePrice',
        xlabel_original='SalePrice',
        xlabel_transformed='log1p(SalePrice)',
    )
else:
    print("Run Task 5 first to visualize the log-transformed SalePrice distribution.")

# %% [markdown]
# #### Pre-filled Illustration: Metrics After Log-Transforming the Target
# 
# If a model is trained on `log1p(SalePrice)`, its predictions also live on the **log scale**.
# 
# That means:
# - metrics computed directly on the log scale describe error in log-units
# - if you want an interpretable metric in the original price units, you must apply `np.expm1()` to both predictions and targets first
# 
# Forgetting this does **not** necessarily crash the code, but it gives you a metric on the wrong scale and can make the model look misleadingly good.

# %%
if 'SalePrice_Log' in df_work.columns:
    example_slice = df_work[['SalePrice', 'SalePrice_Log']].dropna().head(400).copy()

    # Mock log-scale predictions: close, but not perfect.
    y_true_log = example_slice['SalePrice_Log']
    y_pred_log = y_true_log - 0.12

    rmse_on_log_scale = root_mean_squared_error(y_true_log, y_pred_log)

    y_true_original = np.expm1(y_true_log)
    y_pred_original = np.expm1(y_pred_log)
    rmse_after_inverse_transform = root_mean_squared_error(y_true_original, y_pred_original)

    metric_comparison = pd.DataFrame(
        {
            'metric': ['RMSE on log scale', 'RMSE after inverse transform'],
            'value': [rmse_on_log_scale, rmse_after_inverse_transform],
            'unit': ['log-units', 'price units'],
        }
    )
    display(metric_comparison)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=metric_comparison, x='metric', y='value', hue='unit', dodge=False, palette=['slateblue', 'darkorange'])
    plt.title('Why inverse-transforming matters for target-based metrics')
    plt.xlabel('')
    plt.ylabel('RMSE value')
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()
else:
    print("Run Task 5 first to see the metric comparison for log-transformed targets.")

# %% [markdown]
# ### 2.3 Multivariate Outliers (Isolation Forest) ✏️ TODO (⏱️ ~7 min)
# Sometimes univariate IQR isn't enough. Let's look for outliers in a multi-dimensional space.
# 
# **Calculation 6:** Train an `IsolationForest` (with `contamination=0.01` and `random_state=42`) exclusively on a subset containing only `Lot Area` and `SalePrice`. Drop any nulls from this subset before fitting. How many anomalies (indicated by -1) does the forest detect?

# %%
# TODO:
# 1. Fit IsolationForest(contamination=0.01, random_state=RANDOM_STATE) on ['Lot Area', 'SalePrice'].
# 2. Drop missing rows before fitting.
# 3. Count how many predictions equal -1.
# 4. Print the anomaly count.

# %% [markdown]
# **Answer 6:** Isolation Forest detected [Value] anomalies.

# %% [markdown]
# ## 3. Encodings and Binning (⏱️ ~20 min)
# 
# This block focuses on turning categorical and temporal features into representations a model can actually use.

# %% [markdown]
# ### 3.1 Ordinal Encoding ✏️ TODO (⏱️ ~6 min)
# The feature `Exter Qual` contains ordinal text data: 'Ex' (Excellent), 'Gd' (Good), 'TA' (Typical/Average), 'Fa' (Fair), 'Po' (Poor).
# 
# **Calculation 7:** Map these string categories to integers dictionary-style: {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}. Apply this to the `Exter Qual` column. What is the newly calculated mean of this numerical column (rounded to 2 decimal places)?

# %%
# TODO:
# 1. Map Exter Qual with {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}.
# 2. Store the encoded values in a new column.
# 3. Compute and print the mean rounded to 2 decimals.

# %% [markdown]
# **Answer 7:** Mean encoded quality is [Value].

# %% [markdown]
# ### 3.2 Data Binning (Discretization) ✏️ TODO (⏱️ ~7 min)
# Continuous features like `Year Built` can sometimes be too noisy. We can group them.
# 
# **Calculation 8:** Discretize `Year Built` into exactly 5 equal-sized (quantile) bins using `pd.qcut()`. Look at the value counts of the resulting bins. How many houses fall into the absolute newest age bin?

# %%
# TODO:
# 1. Use pd.qcut on Year Built with q=5.
# 2. Inspect the value counts of the bins.
# 3. Find how many houses fall into the newest bin.
# 4. Print the counts and the final number.

# %% [markdown]
# **Answer 8:** [Count] houses fall into the newest bin.

# %%
year_built_quantiles = df_work['Year Built'].quantile(np.linspace(0, 1, 6))

plt.figure(figsize=(10, 5))
sns.histplot(df_work['Year Built'].dropna(), bins=30, color='teal')
for boundary in year_built_quantiles.iloc[1:-1]:
    plt.axvline(boundary, color='crimson', linestyle='--', linewidth=1.5)
plt.title('Year Built distribution with qcut boundaries')
plt.xlabel('Year Built')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 One-Hot Categorical Count ✏️ TODO (⏱️ ~7 min)
# Look at the nominal `Neighborhood` column.
# 
# **Calculation 9:** How many unique neighborhoods are represented in the dataset? If you were to pass `Neighborhood` directly into a `OneHotEncoder(drop='first', sparse_output=False)`, exactly how many NEW boolean/numeric columns would be added to your dataset representing this single feature?

# %%
# TODO:
# 1. Count the number of unique Neighborhood values.
# 2. Recall that OneHotEncoder(drop='first') removes one dummy column.
# 3. Compute how many new columns would be added.
# 4. Print both numbers.

# %% [markdown]
# **Answer 9:** [X] unique neighborhoods and [Y] added one-hot columns.

# %% [markdown]
# ### 3.4 Correlation Checks (Pre-filled)
# 
# Before building the final pipeline, it helps to quantify which numeric features move most strongly with `SalePrice`.
# 
# We look at:
# - **Pearson correlation** for linear association
# - **Spearman correlation** for monotonic association
# 
# Neither one proves causality, but both help you spot useful predictors and redundant features.

# %%
correlation_features = ['SalePrice', 'Gr Liv Area', 'Total Bsmt SF', 'Garage Area', 'Lot Frontage', 'Year Built']
correlation_frame = df_work[correlation_features].dropna()

pearson_to_target = correlation_frame.corr(numeric_only=True)['SalePrice'].drop('SalePrice').sort_values(ascending=False)
spearman_to_target = correlation_frame.corr(method='spearman', numeric_only=True)['SalePrice'].drop('SalePrice').sort_values(ascending=False)

correlation_summary = pd.DataFrame(
    {
        'pearson_with_saleprice': pearson_to_target,
        'spearman_with_saleprice': spearman_to_target.reindex(pearson_to_target.index),
    }
)
display(correlation_summary.round(3))

plt.figure(figsize=(8, 4))
correlation_summary['pearson_with_saleprice'].sort_values().plot(kind='barh', color='mediumpurple')
plt.title('Pearson correlation with SalePrice')
plt.xlabel('Correlation coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

corr_matrix = correlation_frame.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.set_theme(style="white", font_scale=1.0)

fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
ax.set_facecolor("white")
ax.grid(False)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=1,
    linecolor="white",
    cbar_kws={"shrink": 0.85, "label": "Correlation"},
    ax=ax,
)

ax.set_title("Triangular correlation heatmap for selected numeric features", pad=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Feature Engineering, Scaling, and Pipelines (⏱️ ~25 min)
# 
# This final block combines the earlier preprocessing ideas into a small end-to-end regression pipeline.

# %% [markdown]
# ### 4.1 Total Square Footage Engineering ✏️ TODO (⏱️ ~5 min)
# House price is highly dependent on total size.
# 
# **Calculation 10:** Create a new feature `Total_Square_Footage` by summing `Total Bsmt SF`, `1st Flr SF`, and `2nd Flr SF` together (handle NA values safely by filling with 0 before summing). What is the absolute highest total combined square footage of any house in this dataset?

# %%
# TODO:
# 1. Create Total_Square_Footage from basement, first-floor, and second-floor area.
# 2. Fill missing values with 0 before summing.
# 3. Compute the maximum total square footage.
# 4. Print the largest value.

# %% [markdown]
# **Answer 10:** Largest total square footage is [Value] sqft.

# %% [markdown]
# ### 4.2 Robust Scaling the Engineered Feature ✏️ TODO (⏱️ ~5 min)
# If we are using distance-based algorithms, scaling `Total_Square_Footage` is mandatory.
# 
# **Calculation 11:** Apply the `RobustScaler` (which uses IQR) to your `Total_Square_Footage` column. After scaling, what is the newly scaled value of that massive outlier from Task 10 (the largest house)? Round to 2 decimal places.

# %%
# TODO:
# 1. Fit RobustScaler on Total_Square_Footage.
# 2. Store the scaled values in a new column.
# 3. Find the scaled value of the largest house from Task 10.
# 4. Print it rounded to 2 decimals.

# %% [markdown]
# **Answer 11:** Scaled value of the largest house is [Value].

# %%
scaled_candidates = ['Total_Square_Footage_RobustScaled', 'Total_Square_Footage_scaled', 'Total_Square_Footage_robust_scaled']
available_scaled = next((column for column in scaled_candidates if column in df_work.columns), None)

if 'Total_Square_Footage' in df_work.columns and available_scaled is not None:
    plot_before_after_transform(
        df_work['Total_Square_Footage'],
        df_work[available_scaled],
        original_label='Original total square footage',
        transformed_label='Robust-scaled total square footage',
        xlabel_original='Square footage',
        xlabel_transformed='Scaled value',
    )
else:
    print("Run Task 11 first to visualize the effect of RobustScaler on Total_Square_Footage.")

# %% [markdown]
# ### 4.3 The Final Pipeline Blueprint ✏️ TODO (⏱️ ~7 min)
# Put it all together by mapping preprocessing pipelines into a `ColumnTransformer`.
# 
# 1. Select these 4 numeric features: `['Lot Frontage', 'Total Bsmt SF', '1st Flr SF', 'Gr Liv Area']`
# 2. Apply: Median Imputation -> StandardScaler
# 3. Select this 1 categorical feature: `['Neighborhood']`
# 4. Apply: Most Frequent Imputer -> OneHotEncoder(drop='first', sparse_output=False)
# 
# **Calculation 12:** For this task, we are only checking the transformed feature space, not doing model evaluation yet. If you fit/transform this exact `ColumnTransformer` (named `preprocess`) on the selected practical-session subset using **all rows currently in the notebook**, what is the exact `shape` (rows, columns) of the resulting numpy array?

# %%
# TODO:
# 1. Define numeric and categorical feature lists exactly as requested above.
# 2. Build the numeric and categorical pipelines.
# 3. Combine them in a ColumnTransformer named preprocess.
# 4. Fit-transform the selected subset and print the resulting shape.

# %% [markdown]
# **Answer 12:** Resulting transformed shape is (rows, [cols]).

# %% [markdown]
# #### Real-World Note
# 
# Here we fit-transform the full practical-session subset only to inspect the resulting feature-space shape. In a real workflow, the `ColumnTransformer` must be fit on the training split only, and the fitted transformer should then be applied to validation/test data without refitting.

# %% [markdown]
# ### 4.4 Model Evaluation Challenge ✏️ TODO (⏱️ ~8 min)
# Finally, append a `Ridge(alpha=10.0)` model to your `ColumnTransformer` from Task 12 inside a final Pipeline. We're going to predict `SalePrice`.
# 
# **Calculation 13:**
# 1. `train_test_split` your data (80% train, 20% test, `random_state=42`).
# 2. Train the pipeline on `X_train`, `y_train`.
# 3. Predict on `X_test`.
# What is the Root Mean Squared Error (RMSE) on the test set, rounded to the nearest integer?

# %%
# TODO:
# 1. Split X and y with train_test_split(test_size=0.2, random_state=RANDOM_STATE).
# 2. Build a Pipeline with preprocess and Ridge(alpha=10.0).
# 3. Fit on the training data and predict on the test data.
# 4. Compute RMSE and R2, then print both.

# %% [markdown]
# **Answer 13:** Test RMSE is [Integer].
