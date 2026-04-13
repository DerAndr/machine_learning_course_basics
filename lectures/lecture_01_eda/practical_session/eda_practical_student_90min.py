# %% [markdown]
# # Exploratory Data Analysis Practical Session — Student Version
#
# **Dataset:** Palmer Penguins ([OpenML `42585`](https://www.openml.org/search?type=data&sort=runs&id=42585&status=active))
#
# **Dataset reference:** [allisonhorst.github.io/palmerpenguins](https://allisonhorst.github.io/palmerpenguins/)
#
# **Learning goals**
# - inspect dataset structure before plotting
# - compare categorical and numerical summaries
# - visualise center, spread, skew, and missingness
# - study relationships across species, islands, and sex
# - connect univariate, bivariate, and multivariate EDA in one compact workflow
#
# **Classroom framing**
# - this practical is intentionally simple and visual
# - the goal is not modeling, but building good analytical habits before modeling starts
# - keep short written observations as you go; EDA is interpretation, not just plotting

# %% [markdown]
# ## Setup
#
# This notebook runs with the baseline repository environment.
#
# If you work in Google Colab, the standard scientific stack is usually already available. If needed, install:
#
# ```python
# !pip install openml pandas seaborn matplotlib -qqq
# ```

# %%
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.datasets import fetch_openml

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_theme(style='whitegrid', font_scale=0.95)

# %% [markdown]
# ## Shared Helper Functions
#
# These helpers keep the main EDA cells short and make the plots more consistent.

# %%
DATASET_ID = 42585
DATASET_URL = 'https://www.openml.org/search?type=data&sort=runs&id=42585&status=active'
NUMERIC_COLS = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
CATEGORICAL_COLS = ['species', 'island', 'sex']
EXPECTED_COLUMNS = CATEGORICAL_COLS + NUMERIC_COLS
COLUMN_ALIASES = {
    'bill_length_mm': {'bill_length_mm', 'culmen_length_mm', 'bill length mm', 'culmen length mm'},
    'bill_depth_mm': {'bill_depth_mm', 'culmen_depth_mm', 'bill depth mm', 'culmen depth mm'},
    'flipper_length_mm': {'flipper_length_mm', 'flipper length mm'},
    'body_mass_g': {'body_mass_g', 'body mass g', 'body_mass (g)', 'body mass (g)'},
    'species': {'species'},
    'island': {'island'},
    'sex': {'sex'},
}


def _normalize_column_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace('__', '_')
        .replace('-', '_')
        .replace('/', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('.', '_')
    )


def load_penguins(dataset_id=DATASET_ID):
    dataset = fetch_openml(data_id=dataset_id, as_frame=True)
    df = dataset.frame.copy()

    rename_map = {}
    normalized_lookup = {_normalize_column_name(col): col for col in df.columns}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_column_name(alias)
            if normalized_alias in normalized_lookup:
                rename_map[normalized_lookup[normalized_alias]] = canonical_name
                break

    df = df.rename(columns=rename_map)

    missing_required = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(
            f'OpenML dataset {dataset_id} is missing expected Palmer Penguins columns: {missing_required}. '            f'Available columns: {df.columns.tolist()}'
        )

    df = df[EXPECTED_COLUMNS].copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def info_summary(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def plot_missing_counts(missing_counts, title='Missing values by column'):
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if missing_counts.empty:
        print('No missing values found.')
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(missing_counts.index, missing_counts.values, color=sns.color_palette('crest', n_colors=len(missing_counts)))
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Missing value count')
    ax.set_ylabel('Column')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_missing_overview(df, title='Missingness overview'):
    missing_mask = df.isna().astype(int).T
    if missing_mask.values.sum() == 0:
        print('No missing values found.')
        return
    plt.figure(figsize=(12, 3.5))
    sns.heatmap(
        missing_mask,
        cmap=sns.color_palette(['#F1FAEE', '#E63946'], as_cmap=True),
        cbar=False,
        xticklabels=False,
        yticklabels=missing_mask.index,
        linewidths=0,
    )
    plt.title(title, fontweight='bold')
    plt.xlabel('Row index')
    plt.ylabel('Column')
    plt.tight_layout()
    plt.show()


def plot_mean_median_mode(series, title, xlabel):
    clean = series.dropna()
    mode_value = clean.mode().iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios': [2.2, 1]})

    sns.histplot(clean, bins=24, kde=True, color='#5B8E7D', edgecolor='white', ax=axes[0])
    axes[0].axvline(clean.mean(), color='#D62828', linestyle='--', linewidth=2, label=f"Mean: {clean.mean():.1f}")
    axes[0].axvline(clean.median(), color='#1D3557', linestyle='-.', linewidth=2, label=f"Median: {clean.median():.1f}")
    axes[0].axvline(mode_value, color='#F4A261', linestyle=':', linewidth=3, label=f"Mode: {mode_value:.1f}")
    axes[0].set_title(title, fontweight='bold')
    axes[0].set_xlabel(xlabel)
    axes[0].legend()

    sns.boxplot(x=clean, color='#A8DADC', ax=axes[1])
    axes[1].set_title('Boxplot')
    axes[1].set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()


def plot_numeric_distribution(series, title, xlabel, color='#6BAED6'):
    clean = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios': [2.2, 1]})
    sns.histplot(clean, bins=24, kde=True, color=color, edgecolor='white', ax=axes[0])
    axes[0].set_title(title, fontweight='bold')
    axes[0].set_xlabel(xlabel)
    sns.boxplot(x=clean, color=color, ax=axes[1])
    axes[1].set_title('Boxplot')
    axes[1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df, cols):
    corr = df[cols].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        vmin=-1,
        vmax=1,
        cmap='coolwarm',
        square=True,
        linewidths=0,
        cbar_kws={'shrink': 0.8},
    )
    plt.title('Numeric correlation heatmap', fontweight='bold')
    plt.tight_layout()
    plt.show()
    return corr

# %% [markdown]
# ## How To Work In Teams
#
# A simple classroom split for this practical:
# - Group A: dataset structure, missingness, and univariate analysis
# - Group B: grouped comparisons and cross-tabs
# - Group C: correlations, pair plots, and final synthesis

# %% [markdown]
# ## 1. Load and Inspect the Dataset
#
# Start with structure before interpretation. The goal is to answer:
# - how many rows and columns are there?
# - which variables are numeric vs categorical?
# - where are the missing values?
# - what already looks useful for later plots?

# %%
df = load_penguins()

print(f'Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns')
print('\nColumns:')
print(df.columns.tolist())

display(df.head())

# %%
# TODO:
# 1. Display the last 5 rows.
# 2. Print df.info().
# 3. Compute missing values per column.
# 4. Identify which columns are numerical and which are categorical.
# 5. Write down 2-3 high-level observations before moving on.

# %% [markdown]
# ## 2. Quick Structural Audit
#
# This pre-filled block makes the first pass more visual. Use it to connect `df.info()` with what you see in plots.

# %%
missing_counts = df.isna().sum().sort_values(ascending=False)
plot_missing_counts(missing_counts)
plot_missing_overview(df)

summary_frame = pd.DataFrame(
    {
        'dtype': df.dtypes.astype(str),
        'missing_count': df.isna().sum(),
        'missing_pct': (df.isna().mean() * 100).round(1),
        'n_unique': df.nunique(dropna=True),
    }
).sort_values(['missing_count', 'n_unique'], ascending=[False, False])

display(summary_frame)

# %% [markdown]
# ## 3. Univariate Analysis
#
# In this section, focus on one variable at a time. Keep asking:
# - what is typical?
# - how spread out is the variable?
# - is the distribution skewed?
# - do different visual summaries tell the same story?

# %%
species_counts = df['species'].value_counts()
species_pct = (df['species'].value_counts(normalize=True) * 100).round(1)

display(pd.DataFrame({'count': species_counts, 'pct': species_pct}))

plt.figure(figsize=(8, 4))
plt.bar(species_counts.index, species_counts.values, color=sns.color_palette('Set2', n_colors=len(species_counts)), edgecolor='white')
plt.title('Penguins by species', fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %%
# TODO:
# 1. Compute count and percentage distribution for island.
# 2. Display a small summary table.
# 3. Plot a bar chart for island counts.
# 4. Write one sentence about which island is most common.

# %%
body_mass = df['body_mass_g']

summary_stats = pd.Series(
    {
        'mean': body_mass.mean(),
        'median': body_mass.median(),
        'mode': body_mass.mode().iloc[0],
        'std': body_mass.std(),
        'min': body_mass.min(),
        'max': body_mass.max(),
    }
)

display(summary_stats.to_frame('body_mass_g').round(2))
plot_mean_median_mode(body_mass, 'Body mass: mean vs median vs mode', 'Body mass (g)')

# %% [markdown]
# ### Quantiles and ECDF
#
# This short pre-filled block connects the lecture's percentile language to a concrete variable before students move on.

# %%
body_mass_clean = df['body_mass_g'].dropna()
quantiles = body_mass_clean.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).rename('body_mass_g')
display(quantiles.to_frame().round(1))

x_sorted = np.sort(body_mass_clean)
y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios': [1.2, 1.8]})
sns.boxplot(x=body_mass_clean, color='#BDE0FE', ax=axes[0])
axes[0].set_title('Body mass boxplot')
axes[0].set_xlabel('Body mass (g)')
for q, color in zip([0.25, 0.5, 0.75], ['#577590', '#D62828', '#577590']):
    axes[0].axvline(body_mass_clean.quantile(q), color=color, linestyle='--', linewidth=1.8)

axes[1].step(x_sorted, y_ecdf, where='post', color='#1D3557', linewidth=2)
axes[1].set_title('Body mass ECDF', fontweight='bold')
axes[1].set_xlabel('Body mass (g)')
axes[1].set_ylabel('Cumulative proportion')
axes[1].set_ylim(0, 1.02)
plt.tight_layout()
plt.show()

# %%
# TODO:
# 1. Compute mean, median, std, min, and max for flipper_length_mm.
# 2. Plot a histogram with KDE.
# 3. Add a boxplot for flipper_length_mm.
# 4. Compare the shape of flipper_length_mm with body_mass_g.

# %% [markdown]
# ## 4. Bivariate Analysis
#
# Now compare variables across groups and ask whether the same numeric story changes by category.

# %%
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='species', y='body_mass_g', hue='species', palette='Set2', legend=False)
plt.title('Body mass across species', fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Body mass (g)')
plt.tight_layout()
plt.show()

display(df.groupby('species', observed=False)['body_mass_g'].agg(['mean', 'median', 'std', 'count']).round(1))

# %%
# TODO:
# 1. Group by island and compute mean, median, std, and count for flipper_length_mm.
# 2. Plot a boxplot of flipper_length_mm by island.
# 3. Decide whether the island differences look large or modest.

# %%
# TODO:
# 1. Drop rows with missing sex.
# 2. Create a split violin plot for flipper_length_mm by species and sex.
# 3. Explain whether sex differences look similar across all species.

# %%
# TODO:
# 1. Build a cross-tabulation of species vs island.
# 2. Plot it as a heatmap.
# 3. Explain which combinations dominate the dataset.

# %% [markdown]
# ## 5. Relationships and Multivariate Views
#
# Here the goal is to connect numeric relationships with species-level structure.

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='flipper_length_mm',
    y='body_mass_g',
    hue='species',
    style='sex',
    s=90,
)
plt.title('Flipper length vs body mass', fontweight='bold')
plt.xlabel('Flipper length (mm)')
plt.ylabel('Body mass (g)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Regression Line and Correlation
#
# The lecture stresses that a plot suggests a relationship, while correlation and a fitted line summarize it numerically.

# %%
plot_df = df.dropna(subset=['flipper_length_mm', 'body_mass_g'])
pearson_corr = plot_df['flipper_length_mm'].corr(plot_df['body_mass_g'])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.8, 1]})
sns.regplot(
    data=plot_df,
    x='flipper_length_mm',
    y='body_mass_g',
    scatter_kws={'alpha': 0.65, 's': 45, 'color': '#457B9D'},
    line_kws={'color': '#D62828', 'linewidth': 2},
    ax=axes[0],
)
axes[0].set_title(f'Flipper length vs body mass (r = {pearson_corr:.2f})', fontweight='bold')
axes[0].set_xlabel('Flipper length (mm)')
axes[0].set_ylabel('Body mass (g)')

species_corr = (
    plot_df.groupby('species', observed=False)
    .apply(lambda g: g['flipper_length_mm'].corr(g['body_mass_g']))
    .rename('pearson_r')
    .sort_values(ascending=False)
)
axes[1].barh(species_corr.index, species_corr.values, color=['#A8DADC', '#F4A261', '#90BE6D'])
axes[1].set_title('Within-species Pearson correlation')
axes[1].set_xlabel('Pearson r')
axes[1].set_xlim(0, 1)
plt.tight_layout()
plt.show()

# %%
# TODO:
# 1. Plot bill_length_mm vs bill_depth_mm, colored by species.
# 2. Compute the global Pearson correlation.
# 3. Compute the same correlation within each species.
# 4. Comment on why the global relationship can differ from the within-species pattern.

# %%
correlation_matrix = plot_corr_heatmap(df, NUMERIC_COLS)

top_pairs = (
    correlation_matrix.where(~np.eye(len(correlation_matrix), dtype=bool))
    .stack()
    .sort_values(key=lambda s: s.abs(), ascending=False)
)
print('Strongest numeric correlations:')
print(top_pairs.head(6).round(3))

# %%
# TODO:
# 1. Build a pairplot for NUMERIC_COLS plus species.
# 2. Drop missing rows first.
# 3. Use species as hue.
# 4. Decide which variable pair separates species most clearly.

# %% [markdown]
# ## 6. Optional: Automated EDA
#
# Manual EDA should come first. Automated reports are useful later when you want a quick broad scan.

# %%
# Optional:
# 1. Install ydata-profiling if your environment does not have it.
# 2. Generate a compact profile report.
# 3. Compare what the report notices first with what you already found manually.

# %% [markdown]
# ### Optional Install For Automated EDA
#
# If you want to run the profiling step in Colab or another clean environment, install the optional package first.

# %%
# Notebook-only install cell:
# !pip install ydata-profiling -qqq

# %% [markdown]
# ## Final Mini-Report Prompts
#
# Write a short EDA summary in plain language.
#
# Suggested structure:
# 1. Dataset overview: size, variable types, missingness
# 2. Univariate findings: one categorical and one numerical takeaway
# 3. Bivariate findings: one grouped comparison and one numeric relationship
# 4. Multivariate synthesis: what seems most useful to remember before modeling
# 5. Open questions: what would you inspect next if this were a real project?

