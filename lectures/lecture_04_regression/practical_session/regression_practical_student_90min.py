# /// script
# source-notebook = "regression_practical_student_90min.ipynb"
# generated-by = "Codex notebook export"
# ///

# %% [markdown]
# # Regression Practical Session - STUDENT VERSION (90 minutes)
# 
# **Learning Objectives:**
# - inspect a regression dataset with clear visual diagnostics instead of relying only on summary tables
# - engineer a few interpretable features and judge whether they add useful signal
# - build a leakage-safe preprocessing + regression pipeline
# - compare plain linear regression with Ridge, Lasso, and tree-based regressors
# - diagnose multicollinearity with correlations and VIF
# - use residual plots to judge whether the model assumptions look reasonable
# 
# This notebook keeps the strong visual flow of the original regression lab while converting it into a structured classroom practical with targeted TODO cells.

# %% [markdown]
# ## Setup

# %% [markdown]
# ## Setup Note
# 
# ```python
# # If needed:
# # pip install -U pandas seaborn scikit-learn statsmodels
# ```
# 
# This practical uses the classic **Auto MPG** regression dataset via a lightweight CSV source. The target is `mpg`, and the exercise focuses on predictive regression workflow rather than domain-specific automotive knowledge.

# %%
# NOTE: notebook magic commented for local script use: !pip install -U pandas seaborn scikit-learn statsmodels

# %%
import warnings
from io import StringIO
import ssl
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("✓ Libraries loaded successfully!")

# %% [markdown]
# ## Shared Helper Functions
# 
# These helper utilities keep the practical focused on model reasoning and diagnostics instead of repeating plotting boilerplate in every section.

# %%
def load_auto_mpg():
    """Load the Auto MPG dataset with a small SSL-tolerant fallback."""
    try:
        df = pd.read_csv(DATA_URL)
    except Exception:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(DATA_URL, context=context) as response:
            df = pd.read_csv(StringIO(response.read().decode("utf-8")))
    df["origin"] = df["origin"].astype("category")
    return df


def plot_distribution(series, title, xlabel, bins=30, color="steelblue"):
    """Show a histogram and a boxplot together."""
    clean = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [4, 1]})

    sns.histplot(clean, bins=bins, kde=True, color=color, ax=axes[0])
    axes[0].axvline(clean.mean(), color="black", linestyle=":", linewidth=2, label=f"Mean = {clean.mean():.2f}")
    axes[0].axvline(clean.median(), color="crimson", linestyle="--", linewidth=2, label=f"Median = {clean.median():.2f}")
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    sns.boxplot(x=clean, color=color, ax=axes[1])
    axes[1].set_title("Boxplot")
    axes[1].set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()


def plot_missing_counts(missing_counts, top_n=10):
    """Plot the top missing-value counts."""
    top = missing_counts[missing_counts > 0].head(top_n).sort_values(ascending=True)
    if top.empty:
        print("No missing values detected.")
        return

    plt.figure(figsize=(8, 4))
    top.plot(kind="barh", color="indianred")
    plt.title("Top missing-value counts")
    plt.xlabel("Missing values")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def safe_divide(numerator, denominator):
    """Safely divide two series while protecting against zero denominators."""
    denom_safe = denominator.replace(0, np.nan)
    return numerator / denom_safe


def evaluate_regression_model(name, y_train, y_train_pred, y_test, y_test_pred):
    """Return a compact metric dictionary for consistent model comparison."""
    return {
        "Model": name,
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
        "Test RMSE": root_mean_squared_error(y_test, y_test_pred),
    }


def plot_ranked_series(series, title, xlabel, top_n=10, color="steelblue"):
    """Plot a ranked pandas Series."""
    top = series.head(top_n).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    top.plot(kind="barh", color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df):
    """Compare models by error and generalization gap."""
    ordered = results_df.sort_values("Test RMSE").copy()
    ordered["Overfit Gap"] = ordered["Train R2"] - ordered["Test R2"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = ["steelblue" if "Linear" in m or "Ridge" in m or "Lasso" in m else "darkorange" for m in ordered["Model"]]

    axes[0].barh(ordered["Model"], ordered["Test RMSE"], color=colors, alpha=0.8)
    axes[0].set_title("Test RMSE comparison")
    axes[0].set_xlabel("Test RMSE")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].scatter(ordered["Train R2"], ordered["Test R2"], s=180, c=colors, alpha=0.8, edgecolor="black")
    min_r2 = min(ordered["Train R2"].min(), ordered["Test R2"].min()) - 0.03
    max_r2 = max(ordered["Train R2"].max(), ordered["Test R2"].max()) + 0.03
    axes[1].plot([min_r2, max_r2], [min_r2, max_r2], "k--", alpha=0.5)
    axes[1].set_xlim(min_r2, max_r2)
    axes[1].set_ylim(min_r2, max_r2)
    axes[1].set_title("Train vs test R²")
    axes[1].set_xlabel("Train R²")
    axes[1].set_ylabel("Test R²")
    axes[1].grid(alpha=0.3)

    for _, row in ordered.iterrows():
        axes[1].annotate(
            row["Model"],
            (row["Train R2"], row["Test R2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted_panels(y_true, prediction_map):
    """Show actual-vs-predicted scatter plots for a few fitted models."""
    if not prediction_map:
        print("No prediction panels to show.")
        return

    model_items = list(prediction_map.items())
    fig, axes = plt.subplots(1, len(model_items), figsize=(6 * len(model_items), 5), squeeze=False)

    for ax, (model_name, y_pred) in zip(axes[0], model_items):
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="black", linewidth=0.5)
        diagonal_min = min(y_true.min(), y_pred.min())
        diagonal_max = max(y_true.max(), y_pred.max())
        ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "crimson", linestyle="--", linewidth=2)
        ax.set_title(model_name)
        ax.set_xlabel("Actual MPG")
        ax.set_ylabel("Predicted MPG")
        ax.grid(alpha=0.3)

    plt.suptitle("Actual vs predicted on the test set", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_residual_diagnostics(y_true, y_pred, title_prefix):
    """Show the standard four-panel residual diagnostic view."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(y_pred, residuals, alpha=0.65, edgecolor="black", linewidth=0.5)
    axes[0, 0].axhline(0, color="crimson", linestyle="--", linewidth=2)
    axes[0, 0].set_title(f"{title_prefix}: Residuals vs Predicted")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].grid(alpha=0.3)

    sns.histplot(residuals, bins=20, kde=True, color="darkorange", ax=axes[0, 1])
    axes[0, 1].axvline(0, color="crimson", linestyle="--", linewidth=2)
    axes[0, 1].set_title(f"{title_prefix}: Residual Distribution")
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")

    sm.qqplot(residuals, line="45", ax=axes[1, 0], markerfacecolor="steelblue", markeredgecolor="black", alpha=0.5)
    axes[1, 0].set_title(f"{title_prefix}: Q-Q Plot")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].scatter(y_true, y_pred, alpha=0.65, edgecolor="black", linewidth=0.5)
    diagonal_min = min(y_true.min(), y_pred.min())
    diagonal_max = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "crimson", linestyle="--", linewidth=2)
    axes[1, 1].set_title(f"{title_prefix}: Actual vs Predicted")
    axes[1, 1].set_xlabel("Actual target")
    axes[1, 1].set_ylabel("Predicted target")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def coefficient_frame_from_pipeline(fitted_pipeline, numeric_features, categorical_features):
    """Return a tidy coefficient DataFrame for a fitted linear pipeline."""
    preprocess = fitted_pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    coefficients = fitted_pipeline.named_steps["model"].coef_
    coef_frame = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_frame["abs_coefficient"] = coef_frame["coefficient"].abs()
    return coef_frame.sort_values("abs_coefficient", ascending=False)


def find_model_label(results_df, keyword):
    """Find the first model label containing a keyword."""
    matches = results_df.loc[results_df["Model"].str.contains(keyword, case=False, regex=False), "Model"]
    return matches.iloc[0] if not matches.empty else None

# %% [markdown]
# ## How To Work In Teams
# 
# 1. **Group A** works on **Section 1**: Audit, visuals, feature engineering, and multicollinearity.
# 2. **Group B** works on **Section 2**: Baseline linear pipeline and coefficient interpretation.
# 3. **Group C** works on **Sections 3 and 4**: regularization, tree-based models, and residual diagnostics.
# 4. At the end, each group reports one modeling decision that improved generalization and one warning sign that could make a regression model misleading.
# 
# **Important:**
# - You do **not** need to finish the entire notebook during class.
# - Keep intermediate objects like `df_work`, `X_train`, `baseline_pipeline`, and `results_df`, because later sections reuse them.
# - When a pre-filled visualization says “Run Task X first”, that means the TODO cell is supposed to create the required variables.

# %% [markdown]
# ## 1. Audit, Visuals, and Feature Engineering (⏱️ ~25 min)
# 
# **Scenario:** You are building a regression workflow for predicting `mpg` (miles per gallon). The target is continuous, so this is a proper regression problem rather than classification.

# %% [markdown]
# ### 1.1 Load and Inspect the Dataset (Pre-filled)

# %%
df = load_auto_mpg()

print(f"Dataset shape: {df.shape}")
display(df.head())

print("\nTarget summary:")
display(df["mpg"].describe().to_frame(name="mpg"))

# %% [markdown]
# ### 1.2 Missingness and Visual Diagnostics ✏️ TODO (⏱️ ~8 min)
# 
# The dataset is small enough that you can inspect it visually before modeling.
# 
# **Calculation 1:** How many missing values does the `horsepower` column contain?

# %%
# TODO:
# 1. Compute missing-value counts for all columns.
# 2. Print the result for horsepower.
# 3. Optionally store the full sorted counts in a Series named missing_counts.

# missing_counts = ...

# %% [markdown]
# **Answer 1:** Missing `horsepower` values = [Value].

# %%
if "missing_counts" in locals():
    plot_missing_counts(missing_counts)
else:
    print("Run Task 1 first to visualize missing-value counts.")

plot_distribution(
    df["mpg"],
    title="Distribution of MPG",
    xlabel="MPG",
    bins=25,
    color="slateblue",
)

plt.figure(figsize=(9, 5))
sns.regplot(data=df, x="horsepower", y="mpg", scatter_kws={"alpha": 0.55}, line_kws={"color": "crimson"})
plt.title("MPG vs Horsepower")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="weight", y="acceleration", hue="origin", palette="Set2", alpha=0.75)
plt.title("Weight vs Acceleration by Origin")
plt.xlabel("Weight")
plt.ylabel("Acceleration")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="origin", y="mpg", palette="Set3", ax=axes[0])
axes[0].set_title("MPG by Origin")
axes[0].set_xlabel("Origin")
axes[0].set_ylabel("MPG")

sns.boxplot(data=df, x="cylinders", y="mpg", palette="crest", ax=axes[1])
axes[1].set_title("MPG by Cylinders")
axes[1].set_xlabel("Cylinders")
axes[1].set_ylabel("MPG")
plt.tight_layout()
plt.show()

mpg_by_year = df.groupby("model_year", as_index=False)["mpg"].mean()
plt.figure(figsize=(9, 5))
sns.lineplot(data=mpg_by_year, x="model_year", y="mpg", marker="o", color="darkgreen")
plt.title("Average MPG by Model Year")
plt.xlabel("Model Year")
plt.ylabel("Average MPG")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="displacement",
    y="mpg",
    hue="cylinders",
    size="weight",
    sizes=(30, 180),
    alpha=0.7,
    palette="viridis",
)
plt.title("MPG vs Displacement by Cylinder Count")
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.3 Engineer Ratio Features ✏️ TODO (⏱️ ~8 min)
# 
# The raw columns are already useful, but ratio features can reveal interpretable structure.
# 
# Create:
# - `power_to_weight = horsepower / weight`
# - `displacement_per_cylinder = displacement / cylinders`
# - `weight_per_cylinder = weight / cylinders`
# - `car_age = max(model_year) - model_year`
# 
# **Calculation 2:** After feature engineering, what is the maximum value of `power_to_weight` rounded to 4 decimals?

# %%
# TODO:
# 1. Create df_work = df.copy().
# 2. Use safe_divide for the ratio features.
# 3. Create car_age from model_year.
# 4. Print the max power_to_weight rounded to 4 decimals.

# df_work = ...

# %% [markdown]
# **Answer 2:** Max `power_to_weight` = [Value].

# %%
required_engineered = {"power_to_weight", "displacement_per_cylinder", "weight_per_cylinder", "car_age"}

if "df_work" in locals() and required_engineered.issubset(df_work.columns):
    plot_distribution(
        df_work["power_to_weight"],
        title="Power-to-weight distribution",
        xlabel="power_to_weight",
        bins=25,
        color="darkorange",
    )

    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df_work, x="power_to_weight", y="mpg", hue="origin", alpha=0.7, palette="Set2")
    plt.title("MPG vs Power-to-weight")
    plt.xlabel("power_to_weight")
    plt.ylabel("MPG")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df_work, x="car_age", y="mpg", hue="origin", alpha=0.7, palette="viridis")
    plt.title("MPG vs Car Age")
    plt.xlabel("car_age")
    plt.ylabel("MPG")
    plt.tight_layout()
    plt.show()
else:
    print("Run Task 2 first to visualize the engineered features.")

# %% [markdown]
# ### 1.4 Correlations and Multicollinearity ✏️ TODO (⏱️ ~9 min)
# 
# Lecture 04 makes an important distinction:
# - strong pairwise correlations can signal redundancy
# - VIF helps detect multicollinearity more systematically
# 
# **Calculation 3:** Using the numeric columns in `df_work`, which feature has the strongest absolute Pearson correlation with `mpg` (excluding `mpg` itself), and which feature has the highest VIF?

# %%
# TODO:
# 1. Select numeric columns from df_work.
# 2. Compute the numeric correlation matrix.
# 3. Build a Series of absolute correlations with mpg (excluding mpg).
# 4. Build a VIF table for numeric predictors only.
# 5. Print the top correlation feature and the highest-VIF feature.

# corr_matrix = ...
# vif_data = ...

# %% [markdown]
# **Answer 3:** Strongest absolute correlation with `mpg` = [Feature]. Highest VIF = [Feature].

# %%
if "corr_matrix" in locals():
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    plt.figure(figsize=(11, 8))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Triangular correlation heatmap (numeric features)")
    plt.tight_layout()
    plt.show()
else:
    print("Run Task 3 first to visualize the correlation matrix.")

if "vif_data" in locals():
    vif_sorted = vif_data.sort_values("VIF", ascending=False)
    display(vif_sorted)
    plot_ranked_series(
        vif_sorted.set_index("Feature")["VIF"],
        title="Variance Inflation Factor by feature",
        xlabel="VIF",
        top_n=min(10, len(vif_sorted)),
        color="indianred",
    )
else:
    print("Run Task 3 first to visualize the VIF rankings.")

# %% [markdown]
# #### Multicollinearity Note
# 
# `car_age` and `model_year` carry almost the same information because one is a direct linear transform of the other.
# 
# That is why the VIF spike here is not a mysterious statistical accident. It is telling you that the design matrix contains a redundant temporal signal.
# 
# In the modeling section, keep `car_age` and drop `model_year`.

# %% [markdown]
# ## 2. Baseline Linear Regression Workflow (⏱️ ~20 min)
# 
# This block builds the clean predictive baseline: split once, preprocess safely, fit a linear model, and evaluate honestly on held-out data.

# %% [markdown]
# ### 2.1 Define Features and Train/Test Split ✏️ TODO (⏱️ ~5 min)
# 
# Use:
# - target: `mpg`
# - drop identifier-like text column: `name`
# - drop redundant temporal feature: `model_year` (keep `car_age` instead)
# - keep `origin` as categorical
# - all remaining numeric columns as numeric predictors
# 
# **Calculation 4:** After `train_test_split(test_size=0.2, random_state=RANDOM_STATE)`, how many rows are in the test set?

# %%
# TODO:
# 1. Start from df_work if available, otherwise df.
# 2. Drop rows where mpg is missing.
# 3. Drop name and model_year from the predictor frame.
# 4. Create numeric_features and categorical_features.
# 5. Run train_test_split and print the test-set size.

# X_train = ...

# %% [markdown]
# **Answer 4:** Test-set rows = [Value].

# %% [markdown]
# ### 2.2 Build the Preprocessing + Linear Regression Pipeline ✏️ TODO (⏱️ ~6 min)
# 
# Build:
# - numeric pipeline: `SimpleImputer(strategy='median') -> StandardScaler()`
# - categorical pipeline: `SimpleImputer(strategy='most_frequent') -> OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)`
# - final baseline pipeline: `preprocess -> LinearRegression()`
# 
# **Calculation 5:** How many numeric and categorical source features are entering the preprocessor?

# %%
# TODO:
# 1. Build numeric_transformer and categorical_transformer.
# 2. Combine them into preprocessor.
# 3. Create baseline_pipeline with LinearRegression.
# 4. Print the numeric and categorical feature counts.

# baseline_pipeline = ...

# %% [markdown]
# **Answer 5:** Numeric source features = [Value]. Categorical source features = [Value].

# %% [markdown]
# ### 2.3 Fit and Evaluate the Baseline ✏️ TODO (⏱️ ~9 min)
# 
# **Calculation 6:** Fit `baseline_pipeline` on the training data. What is the exact test RMSE rounded to 3 decimals?

# %%
# TODO:
# 1. Fit baseline_pipeline on X_train and y_train.
# 2. Predict on train and test.
# 3. Build a baseline_metrics dictionary with Train R2, Test R2, and Test RMSE.
#    Use the model label `Baseline Linear Regression`.
# 4. Store results_df = pd.DataFrame([baseline_metrics]).
# 5. Print the test RMSE rounded to 3 decimals.

# results_df = ...

# %% [markdown]
# **Answer 6:** Baseline test RMSE = [Value].

# %%
if all(name in locals() for name in ["baseline_pipeline", "X_train", "X_test", "y_train", "y_test"]):
    baseline_coef = coefficient_frame_from_pipeline(baseline_pipeline, numeric_features, categorical_features)
    display(baseline_coef.head(12))

    plot_ranked_series(
        baseline_coef.set_index("feature")["abs_coefficient"],
        title="Top absolute coefficients from baseline linear regression",
        xlabel="|Coefficient|",
        top_n=min(12, len(baseline_coef)),
        color="mediumseagreen",
    )
else:
    print("Run Task 6 first to inspect the fitted baseline coefficients.")

# %% [markdown]
# ### 2.4 Optional Inference View with statsmodels (Pre-filled)
# 
# `scikit-learn` is convenient for predictive pipelines, but Lecture 04 also discusses statistical interpretation.
# 
# This short block shows the same fitted training design through a `statsmodels` OLS lens so you can inspect coefficient significance more directly.

# %%
if "baseline_pipeline" in locals():
    X_train_transformed = baseline_pipeline.named_steps["preprocess"].transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    X_train_sm = sm.add_constant(X_train_transformed)
    sm_ols = sm.OLS(y_train, X_train_sm).fit()

    feature_names = baseline_pipeline.named_steps["preprocess"].get_feature_names_out()
    ols_summary_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": sm_ols.params[1:],
            "p_value": sm_ols.pvalues[1:],
        }
    ).sort_values("p_value")
    display(ols_summary_frame.head(12))
else:
    print("Run Task 6 first to inspect the statsmodels significance view.")

# %% [markdown]
# #### Real-World Note
# 
# This baseline is deliberately plain. It gives you a reference point before regularization or tree-based flexibility. If a more complex model does not beat it honestly on held-out data, that complexity may not be worth it.

# %% [markdown]
# ## 3. Regularization and Tree-Based Models (⏱️ ~25 min)
# 
# Lecture 04 is not just about fitting one line. It is about comparing model families and understanding what each one fixes or sacrifices.

# %% [markdown]
# ### 3.1 Ridge Regression with Cross-Validation ✏️ TODO (⏱️ ~6 min)
# 
# Use `GridSearchCV` with this alpha grid:
# 
# `np.logspace(-2, 2, 9)`
# 
# and scoring = `neg_root_mean_squared_error`.
# 
# **Calculation 7:** What alpha gives the best cross-validated Ridge model?

# %%
# TODO:
# 1. Build a ridge_pipeline with preprocess + Ridge().
# 2. Build a GridSearchCV over model__alpha = np.logspace(-2, 2, 9).
# 3. Fit on X_train, y_train.
# 4. Store best_ridge_pipeline and ridge_metrics.
#    Use the model label `Ridge Regression`.
# 5. Print the best alpha.

# ridge_search = ...

# %% [markdown]
# **Answer 7:** Best Ridge alpha = [Value].

# %% [markdown]
# ### 3.2 Lasso Regression and Sparsity ✏️ TODO (⏱️ ~6 min)
# 
# Use `GridSearchCV` with this alpha grid:
# 
# `np.logspace(-3, 1, 9)`
# 
# and `Lasso(max_iter=10000)`.
# 
# **Calculation 8:** After fitting the best Lasso model, how many non-zero coefficients remain?

# %%
# TODO:
# 1. Build a lasso_pipeline with preprocess + Lasso(max_iter=10000).
# 2. Grid-search over model__alpha = np.logspace(-3, 1, 9).
# 3. Fit on X_train, y_train.
# 4. Store best_lasso_pipeline and lasso_metrics.
#    Use the model label `Lasso Regression`.
# 5. Count the non-zero coefficients and print the count.

# lasso_search = ...

# %% [markdown]
# **Answer 8:** Non-zero Lasso coefficients = [Value].

# %%
if all(name in locals() for name in ["best_ridge_pipeline", "best_lasso_pipeline"]):
    ridge_coef = coefficient_frame_from_pipeline(best_ridge_pipeline, numeric_features, categorical_features).rename(
        columns={"coefficient": "ridge_coef", "abs_coefficient": "ridge_abs"}
    )
    lasso_coef = coefficient_frame_from_pipeline(best_lasso_pipeline, numeric_features, categorical_features).rename(
        columns={"coefficient": "lasso_coef", "abs_coefficient": "lasso_abs"}
    )

    coef_compare = (
        ridge_coef[["feature", "ridge_coef"]]
        .merge(lasso_coef[["feature", "lasso_coef"]], on="feature", how="outer")
        .fillna(0)
    )
    coef_compare["ridge_abs"] = coef_compare["ridge_coef"].abs()
    coef_compare["lasso_abs"] = coef_compare["lasso_coef"].abs()
    coef_compare["max_abs"] = coef_compare[["ridge_abs", "lasso_abs"]].max(axis=1)
    coef_compare = coef_compare.sort_values("max_abs", ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    coef_compare.sort_values("ridge_abs")["ridge_coef"].plot(kind="barh", color="steelblue", ax=axes[0])
    axes[0].set_title("Top Ridge coefficients")
    axes[0].set_xlabel("Coefficient")
    axes[0].set_ylabel("Feature")

    coef_compare.sort_values("lasso_abs")["lasso_coef"].plot(kind="barh", color="darkorange", ax=axes[1])
    axes[1].set_title("Top Lasso coefficients")
    axes[1].set_xlabel("Coefficient")
    axes[1].set_ylabel("Feature")

    plt.tight_layout()
    plt.show()
else:
    print("Run Tasks 7 and 8 first to compare Ridge and Lasso coefficients.")

# %% [markdown]
# ### 3.3 Tree-Based Regression ✏️ TODO (⏱️ ~8 min)
# 
# Fit two tree-based models with small grids:
# 
# - `DecisionTreeRegressor(random_state=RANDOM_STATE)` with:
#   - `max_depth`: `[3, 5, 8, None]`
#   - `min_samples_leaf`: `[1, 3, 5]`
# - `RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200)` with:
#   - `max_depth`: `[4, 8, None]`
#   - `min_samples_leaf`: `[1, 3, 5]`
# 
# **Calculation 9:** Which of the two tree-based models gets the lower test RMSE?

# %%
# TODO:
# 1. Build dt_pipeline and rf_pipeline with the shared preprocessor.
# 2. Grid-search the requested hyperparameters.
# 3. Fit both on X_train, y_train.
# 4. Store dt_metrics and rf_metrics.
#    Use the model labels `Decision Tree Regression` and `Random Forest Regression`.
# 5. Append them to results_df.
# 6. Print which tree model has the lower test RMSE.

# dt_search = ...

# %% [markdown]
# **Answer 9:** Better tree-based model by test RMSE = [Model Name].

# %%
if "results_df" in locals() and len(results_df) >= 3:
    display(results_df.sort_values("Test RMSE"))
    plot_model_comparison(results_df)
else:
    print("Run Tasks 6 to 9 first to compare model families visually.")

if "results_df" in locals() and "X_test" in locals() and "y_test" in locals():
    fitted_predictions = {}
    label_to_var = [
        ("Baseline Linear Regression", "baseline_pipeline"),
        ("Ridge", "best_ridge_pipeline"),
        ("Lasso", "best_lasso_pipeline"),
        ("Decision Tree", "best_dt_pipeline"),
        ("Random Forest", "best_rf_pipeline"),
    ]

    for keyword, var_name in label_to_var:
        if var_name in locals():
            matched_label = find_model_label(results_df, keyword)
            if matched_label is not None:
                fitted_predictions[matched_label] = locals()[var_name].predict(X_test)

    best_labels = [
        label
        for label in results_df.sort_values("Test RMSE")["Model"].tolist()
        if label in fitted_predictions
    ][:3]

    if best_labels:
        plot_actual_vs_predicted_panels(
            y_test,
            {label: fitted_predictions[label] for label in best_labels},
        )
    else:
        print("Run Tasks 6 to 9 first to compare actual vs predicted across models.")
else:
    print("Run Tasks 6 to 9 first to compare actual vs predicted across models.")

# %% [markdown]
# #### Model-Family Note
# 
# If a tree-based model wins on test RMSE, that does **not** mean the linear model was useless.
# 
# In Lecture 04 terms:
# - linear models are still the clearest tool for coefficient interpretation and classical OLS-style reasoning
# - tree models are useful when the signal is more non-linear or interaction-heavy
# - the honest comparison is always based on held-out performance, not on how flexible a model looks in training

# %% [markdown]
# ## 4. Residual Diagnostics and Wrap-up (⏱️ ~20 min)
# 
# Good regression work does not end at one metric. Residual plots often show what summary scores hide.

# %% [markdown]
# ### 4.1 Residual Diagnostics for the Best Model ✏️ TODO (⏱️ ~8 min)
# 
# Choose the best-performing model in `results_df` by **lowest test RMSE**.
# 
# **Calculation 10:** What is the exact mean residual on the test set for that best model, rounded to 4 decimals?

# %%
# TODO:
# 1. Identify the best model from results_df.
# 2. Select its fitted pipeline and test predictions.
# 3. Compute residuals = y_test - y_pred_best.
# 4. Print the mean residual rounded to 4 decimals.

# best_model_name = ...

# %% [markdown]
# **Answer 10:** Mean residual for the best model = [Value].

# %%
model_registry = {}
for name in ["baseline_pipeline", "best_ridge_pipeline", "best_lasso_pipeline", "best_dt_pipeline", "best_rf_pipeline"]:
    if name in locals():
        model_registry[name] = locals()[name]

if "results_df" in locals() and model_registry:
    best_row = results_df.sort_values("Test RMSE").iloc[0]
    best_model_label = best_row["Model"]

    if "best_model_name" in locals():
        best_model_label = best_model_name

    label_to_var = {
        find_model_label(results_df, "Baseline Linear Regression"): "baseline_pipeline",
        find_model_label(results_df, "Ridge"): "best_ridge_pipeline",
        find_model_label(results_df, "Lasso"): "best_lasso_pipeline",
        find_model_label(results_df, "Decision Tree"): "best_dt_pipeline",
        find_model_label(results_df, "Random Forest"): "best_rf_pipeline",
    }

    best_var_name = label_to_var.get(best_model_label)

    if best_var_name in model_registry:
        best_pipeline = model_registry[best_var_name]
        best_predictions = best_pipeline.predict(X_test)
        plot_residual_diagnostics(y_test, best_predictions, title_prefix=best_model_label)

        residuals = y_test - best_predictions
        print(f"Best model: {best_model_label}")
        print(f"Mean residual: {residuals.mean():.4f}")
        print(f"Std residual: {residuals.std():.4f}")
        print(f"Min residual: {residuals.min():.4f}")
        print(f"Max residual: {residuals.max():.4f}")
    else:
        print("Run Task 10 and make sure best_model_name matches one of the fitted model labels.")
else:
    print("Run Tasks 6 to 10 first to inspect residual diagnostics.")

# %% [markdown]
# #### Interpretation Note
# 
# In residual diagnostics, look for:
# - random scatter around zero in residuals vs predicted
# - no obvious funnel shape
# - a roughly symmetric residual histogram
# - a Q-Q plot that is not wildly curved
# 
# None of these need to be perfect in classroom data, but strong violations tell you where a regression model is struggling.

# %% [markdown]
# ## Final Reflection
# 
# Answer briefly in Markdown:
# 
# 1. Which features looked most useful for predicting `mpg`?
# 2. Did regularization or tree-based models improve generalization over the baseline?
# 3. What would you change next if this model were going into a real project?
