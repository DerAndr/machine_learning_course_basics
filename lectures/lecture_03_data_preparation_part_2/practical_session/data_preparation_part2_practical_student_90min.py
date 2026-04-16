# %% [markdown]
# # Data Preparation Part 2: Practical Session - STUDENT VERSION (90 minutes)
# 
# **Learning Objectives:**
# - engineer useful features from raw tabular columns instead of creating arbitrary noise
# - compare filter, wrapper, and embedded feature-selection strategies on the same regression target
# - practice dimensionality reduction with PCA and non-linear visualization with UMAP
# - connect feature work to leakage-safe validation and reusable preprocessing pipelines
# - explain why train/test splitting and pipelines matter for honest model evaluation
# 
# This notebook uses targeted TODO placeholders while keeping one shared classroom flow across the main strategic preprocessing topics from Lecture 03.

# %% [markdown]
# ## Setup

# %% [markdown]
# ## Setup Note
# 
# ```python
# # If needed:
# # pip install -U umap-learn scikit-learn liac-arff
# ```
# 
# This practical uses the Ames Housing dataset from **OpenML dataset `41211`**. We immediately rename the columns into the course's space-separated style so they stay aligned with Lecture 02 and the lecture examples.

# %% [code]
# NOTE: notebook magic commented for local script use: !pip install -U umap-learn scikit-learn liac-arff

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print('✓ Libraries loaded successfully!')

# %% [markdown]
# ## Shared Helper Functions
# 
# These helper utilities keep the practical focused on interpretation and preprocessing choices instead of repeating plotting boilerplate.

# %%
def plot_ranked_series(series, title, xlabel, top_n=10, color='steelblue'):
    """Plot the top-ranked values of a pandas Series."""
    top = series.head(top_n).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    top.plot(kind='barh', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def plot_hist_and_box(series, title, xlabel, bins=40, color='steelblue'):
    """Show a distribution with a histogram and a boxplot."""
    clean = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios': [4, 1]})

    sns.histplot(clean, bins=bins, kde=True, color=color, ax=axes[0])
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


def plot_cumulative_variance(cumulative_variance):
    """Visualize cumulative explained variance for PCA."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(0.95, color='crimson', linestyle='--', label='95% variance threshold')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('PCA cumulative explained variance')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_missing_percentages(missing_pct, top_n=10):
    """Visualize the most-missing columns."""
    top_missing = missing_pct.head(top_n).sort_values(ascending=True)
    if top_missing.empty:
        print('No missing values detected after parsing the dataset.')
        return

    plt.figure(figsize=(8, 5))
    top_missing.plot(kind='barh', color='indianred')
    plt.title('Top columns by missing percentage')
    plt.xlabel('Missing percentage')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_scatter_with_target(frame, x_col, y_col='SalePrice', title=None, xlabel=None, ylabel=None, sample_n=1200):
    """Plot a target relationship with sampled points for readability."""
    clean = frame[[x_col, y_col]].dropna()
    if clean.empty:
        print(f'No non-missing rows available for {x_col} vs {y_col}.')
        return

    sample = clean.sample(n=min(len(clean), sample_n), random_state=RANDOM_STATE)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x=x_col, y=y_col, alpha=0.45, s=28, color='teal')
    plt.title(title or f'{y_col} vs {x_col}')
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.tight_layout()
    plt.show()


def plot_binary_target_boxplot(frame, binary_col, target_col='SalePrice'):
    """Compare a numeric target across a binary feature."""
    clean = frame[[binary_col, target_col]].dropna().copy()
    if clean.empty:
        print(f'No non-missing rows available for {binary_col} vs {target_col}.')
        return

    clean[binary_col] = clean[binary_col].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=clean, x=binary_col, y=target_col, palette='Set2', ax=axes[0])
    axes[0].set_title(f'{target_col} by {binary_col}')
    axes[0].set_xlabel(binary_col)
    axes[0].set_ylabel(target_col)

    avg_target = clean.groupby(binary_col)[target_col].mean().rename('mean_saleprice')
    avg_target.plot(kind='bar', color=['steelblue', 'darkorange'], ax=axes[1])
    axes[1].set_title(f'Mean {target_col} by {binary_col}')
    axes[1].set_xlabel(binary_col)
    axes[1].set_ylabel(f'Mean {target_col}')
    axes[1].tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.show()


def plot_selection_overlap(method_sets, top_n=15):
    """Visualize which features are shared across selection methods."""
    feature_union = sorted(set().union(*method_sets.values()))
    if not feature_union:
        print('No feature-selection results are available yet.')
        return

    overlap = pd.DataFrame(
        {method: [int(feature in selected) for feature in feature_union] for method, selected in method_sets.items()},
        index=feature_union,
    )
    overlap['method_count'] = overlap.sum(axis=1)
    overlap = overlap.sort_values('method_count', ascending=False).head(top_n)

    plt.figure(figsize=(8, max(4, 0.35 * len(overlap))))
    sns.heatmap(
        overlap.drop(columns='method_count'),
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        cbar=False,
        linewidths=0.5,
        linecolor='white',
    )
    plt.title('Selection-method overlap for top consensus features')
    plt.xlabel('Method')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_pca_loadings(pca, feature_names, n_components=3):
    """Show how strongly original features contribute to the first principal components."""
    n_show = min(n_components, pca.components_.shape[0])
    loading_frame = pd.DataFrame(
        pca.components_[:n_show],
        index=[f'PC{i}' for i in range(1, n_show + 1)],
        columns=feature_names,
    )

    plt.figure(figsize=(9, 4 + 0.5 * n_show))
    sns.heatmap(
        loading_frame,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        linecolor='white',
    )
    plt.title('PCA component loadings')
    plt.xlabel('Original feature')
    plt.ylabel('Principal component')
    plt.tight_layout()
    plt.show()


def plot_cv_score_comparison(safe_scores, leaky_scores):
    """Compare fold-level scores for a safe and a leaky validation setup."""
    comparison = pd.DataFrame(
        {
            'Safe baseline': safe_scores,
            'Leaky feature set': leaky_scores,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    comparison.plot(kind='bar', ax=axes[0], color=['steelblue', 'darkorange'])
    axes[0].set_title('Fold-by-fold CV R² scores')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('R²')
    axes[0].tick_params(axis='x', rotation=0)

    sns.boxplot(data=comparison, orient='h', palette=['steelblue', 'darkorange'], ax=axes[1])
    axes[1].set_title('Score distribution by setup')
    axes[1].set_xlabel('R²')

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
    df = df.replace({'?': np.nan, 'NA': np.nan, 'NaN': np.nan, 'nan': np.nan, 'None': np.nan})
    object_columns = df.select_dtypes(include=['object', 'string']).columns
    if len(object_columns) > 0:
        df[object_columns] = df[object_columns].replace(r'^\s*$', np.nan, regex=True)

    return df

# %% [markdown]
# ## How To Work In Teams
# 
# 1. **Group A** works on **Section 1**: Feature Generation and Encoded Analysis Frame.
# 2. **Group B** works on **Section 2**: Feature Selection.
# 3. **Group C** works on **Sections 3 and 4**: Dimensionality Reduction, Validation, and Pipelines.
# 4. At the end, each group reports one preprocessing decision that improved signal and one methodological risk that could create leakage.
# 
# **Important:**
# - You do **not** need to finish the whole notebook during class.
# - Focus on the TODO cells inside your group's block first.
# - Keep intermediate objects like `df_work`, `X_encoded`, `preprocessor`, and `model_pipeline` because later tasks reuse them.

# %% [markdown]
# ## 1. Feature Generation and Audit (⏱️ ~25 min)
# 
# **Scenario:** You are extending the Ames Housing workflow from Lecture 02. This practical uses the same **2930-row Ames variant**, but now it is loaded from OpenML dataset `41211` instead of a direct CSV source. The goal is not only to clean the data, but to decide which features are worth keeping, which new ones to create, and how to package the whole process for safe evaluation.

# %% [markdown]
# ### 1.1 Load the Ames Housing Dataset (Pre-filled)

# %%
df = load_ames_housing_openml()

print(f'Loaded Ames Housing with {df.shape[0]} rows and {df.shape[1]} columns.')
display(df.head())

# %% [markdown]
# ### 1.2 Quick Audit (Pre-filled)

# %%
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

print('Top columns by missing percentage:')
display(missing_pct.head(10).to_frame(name='missing_pct'))
plot_missing_percentages(missing_pct)

print('Numeric summary preview:')
display(df.describe(include=[np.number]).T.head(10))

plot_hist_and_box(
    df['SalePrice'],
    title='SalePrice baseline distribution',
    xlabel='SalePrice',
    bins=45,
    color='slateblue',
)

plot_scatter_with_target(
    df,
    x_col='Gr Liv Area',
    y_col='SalePrice',
    title='SalePrice vs Gr Liv Area before feature engineering',
    xlabel='Gr Liv Area',
    ylabel='SalePrice',
)

# %% [markdown]
# ### 1.3 Build a Working Copy and Repair `Lot Frontage` ✏️ TODO (⏱️ ~8 min)
# 
# `Lot Frontage` still has missing values. Fill it using the **median within each `Neighborhood`**. If a neighborhood-specific median is unavailable, fall back to the overall column median.
# 
# **Calculation 1:** After this two-step fill strategy, how many missing values remain in `Lot Frontage`?

# %%
# TODO:
# 1. Create df_work = df.copy(deep=True).
# 2. Fill Lot Frontage by Neighborhood median using groupby + transform.
# 3. Apply a fallback fill with the overall Lot Frontage median.
# 4. Print the remaining missing-value count.

# df_work = ...

# %% [markdown]
# **Answer 1:** Remaining missing values in `Lot Frontage` = [Value].

# %% [markdown]
# ### 1.4 Engineer Core Features ✏️ TODO (⏱️ ~8 min)
# 
# Create three simple but interpretable engineered features:
# 
# - `HouseAge = Yr Sold - Year Built`
# - `TotalArea = Gr Liv Area + Total Bsmt SF` (fill basement missing values with 0 before summing)
# - `HasPool = 1 if Pool Area > 0 else 0`
# 
# **Calculation 2:** What is the maximum value of `TotalArea` after feature engineering?

# %%
# TODO:
# 1. Create HouseAge.
# 2. Create TotalArea safely.
# 3. Create HasPool as a binary column.
# 4. Print the max of TotalArea.

# df_work['HouseAge'] = ...

# %% [markdown]
# **Answer 2:** Maximum `TotalArea` = [Value].

# %% [markdown]
# ### 1.4.1 Why These Engineered Features Matter (Pre-filled)
# 
# These engineered columns are useful because they compress raw inputs into signals that are easier for a model to learn and easier for us to interpret:
# - `HouseAge` turns two calendar columns into one age feature
# - `TotalArea` combines two strong size measurements into one broader footprint
# - `HasPool` simplifies a sparse numeric column into a yes/no flag

# %%
required_engineered = {'HouseAge', 'TotalArea', 'HasPool', 'SalePrice'}

if required_engineered.issubset(df_work.columns):
    plot_hist_and_box(
        df_work['HouseAge'],
        title='HouseAge distribution',
        xlabel='HouseAge (years)',
        bins=35,
        color='teal',
    )

    plot_hist_and_box(
        df_work['TotalArea'],
        title='TotalArea distribution',
        xlabel='TotalArea (sq ft)',
        bins=40,
        color='darkorange',
    )

    plot_scatter_with_target(
        df_work,
        x_col='TotalArea',
        y_col='SalePrice',
        title='SalePrice vs TotalArea',
        xlabel='TotalArea (sq ft)',
        ylabel='SalePrice',
    )

    plot_binary_target_boxplot(df_work, binary_col='HasPool', target_col='SalePrice')
else:
    print('Run Task 2 first to visualize the engineered features.')

# %% [markdown]
# ### 1.5 Build a Quick Encoded Analysis Frame ✏️ TODO (⏱️ ~9 min)
# 
# For fast feature scoring, create a quick numeric analysis frame `X_encoded`:
# 
# 1. start from `df_work`
# 2. drop `SalePrice`
# 3. drop columns with more than 40% missing values
# 4. convert remaining object/category columns to category codes
# 5. fill the remaining missing values with `0`
# 
# **Calculation 3:** What is the final shape of `X_encoded`?

# %%
# TODO:
# 1. Build X_encoded from df_work.
# 2. Drop SalePrice.
# 3. Remove columns with >40% missingness.
# 4. Encode remaining categorical columns as category codes.
# 5. Fill residual missing values with 0.
# 6. Print the final shape.

# X_encoded = ...

# %% [markdown]
# **Answer 3:** `X_encoded.shape` = ([rows], [cols]).

# %% [markdown]
# #### Method Note
# 
# `X_encoded` is a **quick classroom analysis frame** for ranking and comparing features. The category codes are acceptable for this exploratory block, but they are **not** the final modeling representation. In the pipeline section later, we go back to proper imputers, encoders, and train-only fitting.

# %% [markdown]
# ### 1.6 Correlation Check for Engineered Features (Pre-filled)
# 
# Before formal feature selection begins, it helps to verify that the engineered features move with `SalePrice` in ways that make sense.

# %%
engineered_corr_features = ['SalePrice', 'Gr Liv Area', 'Total Bsmt SF', 'HouseAge', 'TotalArea', 'HasPool']

if set(engineered_corr_features).issubset(df_work.columns):
    engineered_corr = df_work[engineered_corr_features].dropna()
    engineered_to_target = (
        engineered_corr.corr(numeric_only=True)['SalePrice']
        .drop('SalePrice')
        .sort_values(ascending=False)
    )
    display(engineered_to_target.to_frame(name='pearson_with_saleprice').round(3))

    plot_ranked_series(
        engineered_to_target,
        title='Engineered-feature correlation with SalePrice',
        xlabel='Pearson correlation',
        top_n=len(engineered_to_target),
        color='mediumpurple',
    )

    corr_matrix = engineered_corr.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor='white',
    )
    plt.title('Triangular correlation heatmap for engineered features')
    plt.tight_layout()
    plt.show()
else:
    print('Run Tasks 1 and 2 first to inspect engineered-feature correlations.')

# %% [markdown]
# ## 2. Feature Selection (⏱️ ~25 min)
# 
# This block compares multiple ways to answer the same practical question: which features seem most useful for predicting `SalePrice`?

# %% [markdown]
# ### 2.1 Define the Target (Pre-filled)

# %%
y = df_work['SalePrice'].copy()
print(f'Target mean: {y.mean():.2f}')

# %% [markdown]
# ### 2.2 Filter Method: Mutual Information ✏️ TODO (⏱️ ~6 min)
# 
# Use `mutual_info_regression` on `X_encoded` against `y`.
# 
# **Calculation 4:** Which feature has the highest mutual-information score?

# %%
# TODO:
# 1. Compute mutual_info_regression(X_encoded, y).
# 2. Wrap the scores in a pandas Series indexed by feature name.
# 3. Sort descending and inspect the top 10 features.
# 4. Plot the top 10 scores.

# mi_scores = ...

# %% [markdown]
# **Answer 4:** Top MI feature = [Feature Name].

# %% [markdown]
# ### 2.3 Filter Method: SelectKBest ✏️ TODO (⏱️ ~5 min)
# 
# Use `SelectKBest(score_func=f_regression, k=15)`.
# 
# **Calculation 5:** How many features are selected, and which one appears first in your selected-feature list?

# %%
# TODO:
# 1. Fit SelectKBest with f_regression and k=15.
# 2. Extract the selected feature names.
# 3. Display them as a DataFrame or Series.

# selected_features = ...

# %% [markdown]
# **Answer 5:** Number selected = [Value]. First selected feature = [Feature Name].

# %% [markdown]
# ### 2.4 Wrapper Method: Sequential Feature Selection ✏️ TODO (⏱️ ~7 min)
# 
# To keep runtime manageable and the linear solver numerically stable, run forward selection only on the **top 20 mutual-information features** using `Ridge(alpha=1.0)` and keep **8 final features**.
# 
# **Calculation 6:** Which 8 features survive sequential feature selection?

# %%
# TODO:
# 1. Take the top 20 features from mi_scores.
# 2. Create a SequentialFeatureSelector with Ridge(alpha=1.0).
# 3. Select 8 features using forward selection and cv=3.
# 4. Print the final selected feature names.

# top20_features = ...
# sfs_selected = ...

# %% [markdown]
# **Answer 6:** Sequentially selected features = [List of 8 features].

# %% [markdown]
# ### 2.5 Embedded Methods and Consensus ✏️ TODO (⏱️ ~7 min)
# 
# Use two embedded strategies:
# 
# - `Lasso(alpha=500, max_iter=2000)` on a scaled version of `X_encoded`
# - `RandomForestRegressor` feature importances on `X_encoded`
# 
# Then compare the top features across MI, SelectKBest, Sequential Feature Selection, Lasso, and Random Forest.
# 
# **Calculation 7:** Which features are selected by at least 3 of the 5 methods?

# %%
# TODO:
# 1. Scale X_encoded for Lasso.
# 2. Fit Lasso and collect the top non-zero coefficients by absolute value.
# 3. Fit RandomForestRegressor and collect the top feature importances.
# 4. Store the top embedded-method outputs in variables named lasso_top and rf_top.
# 5. Build a small consensus count across MI, SelectKBest, SFS, Lasso, and RF.
# 6. Optionally collect the method-level sets in a dictionary named method_sets for later visualization.
# 7. Print the features selected by at least 3 methods.

# consensus_features = ...

# %% [markdown]
# **Answer 7:** Consensus features = [Feature Names].

# %% [markdown]
# ### 2.6 Visual Comparison of Selection Methods (Pre-filled)
# 
# Feature-selection results are easier to interpret when you compare overlap across methods rather than reading each list separately.

# %%
selection_objects_ready = all(
    name in locals()
    for name in ['mi_scores', 'selected_features', 'sfs_selected', 'lasso_top', 'rf_top']
)

if selection_objects_ready:
    if 'method_sets' not in locals():
        method_sets = {
            'mi': set(mi_scores.head(15).index),
            'kbest': set(selected_features),
            'sfs': set(sfs_selected),
            'lasso': set(lasso_top.index),
            'rf': set(rf_top.index),
        }

    plot_selection_overlap(method_sets, top_n=15)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lasso_top.head(10).sort_values().plot(kind='barh', color='darkgreen', ax=axes[0])
    axes[0].set_title('Top absolute Lasso coefficients')
    axes[0].set_xlabel('|Coefficient|')
    axes[0].set_ylabel('Feature')

    rf_top.head(10).sort_values().plot(kind='barh', color='darkorange', ax=axes[1])
    axes[1].set_title('Top Random Forest importances')
    axes[1].set_xlabel('Importance')
    axes[1].set_ylabel('Feature')

    plt.tight_layout()
    plt.show()
else:
    print('Run the feature-selection tasks first to compare methods visually.')

# %% [markdown]
# #### Real-World Note
# 
# For this classroom comparison, the selectors run on the shared analysis frame so everyone sees the same rankings. In a real project, feature selection should be fit **inside** the training workflow or cross-validation loop rather than on the full dataset.

# %% [markdown]
# ## 3. Dimensionality Reduction (⏱️ ~20 min)
# 
# Feature selection keeps original columns. Dimensionality reduction instead builds a smaller representation of the data. The question is no longer “which columns survive?” but “how much structure can we keep in fewer dimensions?”

# %% [markdown]
# ### 3.1 Prepare a Scaled Numeric Matrix ✏️ TODO (⏱️ ~5 min)
# 
# Use these columns:
# 
# `['Gr Liv Area', 'Total Bsmt SF', 'Garage Area', '1st Flr SF', 'Lot Area', 'TotalArea']`
# 
# Fill missing values with `0`, scale them, and store the result in `X_pca_scaled`.
# 
# **Calculation 8:** What is the shape of `X_pca_scaled`?

# %%
# TODO:
# 1. Select the requested numeric columns.
# 2. Fill missing values with 0.
# 3. Fit a StandardScaler and transform the matrix.
# 4. Print the resulting shape.

# X_pca_scaled = ...

# %% [markdown]
# **Answer 8:** `X_pca_scaled.shape` = ([rows], [cols]).

# %% [markdown]
# ### 3.2 PCA at 95% Variance ✏️ TODO (⏱️ ~7 min)
# 
# Fit `PCA(n_components=0.95, random_state=RANDOM_STATE)` on `X_pca_scaled`.
# 
# **Calculation 9:** How many principal components are needed to retain at least 95% of the variance?

# %%
# TODO:
# 1. Fit PCA with n_components=0.95.
# 2. Transform X_pca_scaled.
# 3. Print the number of retained components and the explained-variance ratios.
# 4. Plot the cumulative explained variance.

# pca = ...

# %% [markdown]
# **Answer 9:** PCA retained [Value] components for 95% variance.

# %% [markdown]
# ### 3.2.1 PCA Structure View (Pre-filled)
# 
# PCA becomes much more interpretable when you inspect both the variance profile and the component loadings.

# %%
if 'pca' in locals():
    variance_by_component = pd.Series(
        pca.explained_variance_ratio_,
        index=[f'PC{i}' for i in range(1, len(pca.explained_variance_ratio_) + 1)],
        name='explained_variance_ratio',
    )
    display(variance_by_component.round(3).to_frame())

    plt.figure(figsize=(8, 4))
    variance_by_component.plot(kind='bar', color='cornflowerblue')
    plt.title('Explained variance ratio by principal component')
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.tight_layout()
    plt.show()

    plot_pca_loadings(pca, pca_features, n_components=min(3, pca.n_components_))

    if 'X_pca_reduced' in locals() and X_pca_reduced.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_pca_reduced[:, 0],
            X_pca_reduced[:, 1],
            c=y,
            cmap='viridis',
            s=14,
            alpha=0.55,
        )
        plt.colorbar(scatter, label='SalePrice')
        plt.title('First two PCA components colored by SalePrice')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.show()
else:
    print('Run Task 9 first to inspect PCA structure.')

# %% [markdown]
# ### 3.3 UMAP Visualization ✏️ TODO (⏱️ ~8 min)
# 
# Use UMAP to create a 2D visualization of `X_pca_scaled` and color the plot by `SalePrice`.
# 
# **Interpretation Prompt:** Does the 2D view suggest a few sharply separated groups, or mostly a continuous structure?

# %%
# TODO:
# 1. Fit umap.UMAP(n_components=2, random_state=RANDOM_STATE) on X_pca_scaled.
# 2. Plot the two UMAP coordinates with points colored by y.
# 3. Use a small point size and alpha for readability.

# X_umap = ...

# %% [markdown]
# **Answer 10:** The UMAP view suggests [continuous structure / several clear clusters + one-sentence justification].

# %% [markdown]
# ## 4. Validation and Pipelines (⏱️ ~20 min)
# 
# This final block turns your feature decisions into a leakage-safe sklearn workflow.

# %% [markdown]
# ### 4.1 Quick Leakage Contrast (Pre-filled)
# 
# Before building the safe pipeline, compare a normal baseline with an intentionally broken feature set. The second model cheats by receiving a direct copy of the target itself. This is an extreme example, but it makes the leakage warning obvious: impossible validation scores usually mean the model saw information it should never have had at prediction time.

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

X_safe = df_work[['Lot Area', 'Gr Liv Area']].fillna(0).copy()
safe_scores = cross_val_score(
    LinearRegression(),
    X_safe,
    y,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
)

X_leaky = X_safe.copy()
X_leaky['SalePrice_copy'] = y
leaky_scores = cross_val_score(
    LinearRegression(),
    X_leaky,
    y,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
)

print(f"Safe baseline mean CV R²: {safe_scores.mean():.3f}")
print(f"Leaky feature-set mean CV R²: {leaky_scores.mean():.3f}")

plot_cv_score_comparison(safe_scores, leaky_scores)

# %% [markdown]
# #### Leakage Note
# 
# This pre-filled example is intentionally absurd. Real leakage is usually subtler: a target-derived feature, a global target encoding, or preprocessing fit on the full dataset can still inflate scores without looking as obviously wrong as `SalePrice_copy`.

# %% [markdown]
# ### 4.2 Define a Leakage-Safe Preprocessor ✏️ TODO (⏱️ ~8 min)
# 
# Use these features:
# 
# - numeric: `['TotalArea', 'HouseAge', 'Garage Area', 'Lot Frontage']`
# - categorical: `['Neighborhood', 'Bldg Type', 'Kitchen Qual']`
# 
# Build:
# 
# - numeric pipeline: `SimpleImputer(strategy='median') -> StandardScaler()`
# - categorical pipeline: `SimpleImputer(strategy='most_frequent') -> OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`
# - final `ColumnTransformer` named `preprocessor`
# 
# **Calculation 11:** What are the numeric and categorical feature counts in this preprocessor design?

# %%
# TODO:
# 1. Define num_features and cat_features exactly as listed.
# 2. Build num_pipe and cat_pipe.
# 3. Combine them in a ColumnTransformer called preprocessor.
# 4. Print the number of numeric and categorical source features.

# preprocessor = ...

# %% [markdown]
# **Answer 11:** Numeric feature count = [Value]. Categorical feature count = [Value].

# %% [markdown]
# ### 4.3 Build a Full Pipeline and Cross-Validate ✏️ TODO (⏱️ ~8 min)
# 
# Attach `LinearRegression()` after the preprocessor and evaluate with `KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)`.
# 
# **Calculation 12:** What is the mean cross-validated `R²` score rounded to 3 decimals?

# %%
# TODO:
# 1. Build a Pipeline with preprocessor and LinearRegression().
# 2. Define X_final and y_final.
# 3. Run 5-fold CV with scoring='r2'.
# 4. Print the individual scores and the mean score.

# model_pipeline = ...

# %% [markdown]
# **Answer 12:** Mean CV `R²` = [Value].

# %% [markdown]
# #### Real-World Note
# 
# This final pipeline is the lecture's main workflow lesson. In a real ML project, imputation, scaling, encoding, feature selection, and dimensionality reduction should all be fit only inside the training workflow. Pipelines help enforce that rule and prevent train/test contamination.

# %% [markdown]
# ## Final Checklist
# 
# - [ ] I can explain the difference between feature selection and dimensionality reduction.
# - [ ] I can name one filter, one wrapper, and one embedded selection method.
# - [ ] I understand why feature engineering can help or hurt.
# - [ ] I can explain why preprocessing should be fit inside a pipeline instead of on the full dataset.
# - [ ] I can explain one realistic source of leakage from this lecture.
