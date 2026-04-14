# %%
# /// script
# source-notebook = "xai_practical_student_90min.ipynb"
# generated-by = "Codex notebook export"
# ///

# %% [markdown]
# # Explainability and Interpretability: Practical Session - STUDENT VERSION (90 minutes)
#
# **Learning objectives:**
# - compare an intrinsically interpretable baseline with a stronger black-box model on the same task;
# - work with `PFI`, `PDP`, `ALE`, `LIME`, `SHAP`, and `InterpretML` in one notebook;
# - practice reading explanation plots, not only generating them;
# - distinguish clearly between global and local explanations.

# %% [markdown]
# ## Setup
#
# For local work in this repository, prefer:
#
# ```bash
# uv sync
# uv run python tools/check_notebook_environment.py
# ```
#
# If you run this notebook in Google Colab and a package is missing, install:
#
# ```python
# %pip install -q shap lime alibi interpret
# ```
#
# Tooling note:
#
# - This notebook uses `scikit-learn`, `alibi`, `lime`, `shap`, and `interpret`.
# - `eli5` is another well-known model-inspection tool in the ecosystem, but it is not required here.

# %% [markdown]
# ## Imports

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import HTML, display
from alibi.explainers import ALE, plot_ale
from interpret.glassbox import ExplainableBoostingClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shap

warnings.filterwarnings("ignore")
sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "axes.facecolor": "#FBFDFF",
        "figure.facecolor": "white",
        "grid.color": "#DCEAF2",
        "axes.edgecolor": "#149ECA",
        "axes.labelcolor": "#17324D",
        "xtick.color": "#17324D",
        "ytick.color": "#17324D",
    },
)

RANDOM_STATE = 42
FOCUS_FEATURE = "mean radius"
INTERACTION_FEATURE = "mean perimeter"
TOP_FEATURES = [FOCUS_FEATURE, INTERACTION_FEATURE]
PLOT_COLORS = {
    "teal": "#22B8BD",
    "blue": "#149ECA",
    "orange": "#F28E2B",
    "rose": "#D1495B",
    "ink": "#17324D",
    "grid": "#DCEAF2",
    "panel": "#FBFDFF",
}

# %% [markdown]
# ## Shared Helper Functions

# %%
def load_breast_cancer_frame():
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series((dataset.target == 0).astype(int), name="malignant")
    class_mapping = {0: "benign", 1: "malignant"}
    return X, y, class_mapping


def evaluate_binary_model(model, X_train, X_test, y_train, y_test, model_name):
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "train_accuracy": accuracy_score(y_train, train_pred),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "train_roc_auc": roc_auc_score(y_train, train_proba),
                "test_roc_auc": roc_auc_score(y_test, test_proba),
            }
        ]
    )
    return summary.round(3)


def style_axis(ax, title=None, xlabel=None, ylabel=None, border_color=None):
    ax.set_facecolor(PLOT_COLORS["panel"])
    ax.grid(True, color=PLOT_COLORS["grid"], linewidth=0.8, alpha=0.9)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color or PLOT_COLORS["blue"])
        spine.set_linewidth(1.4)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", color=PLOT_COLORS["ink"], pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=PLOT_COLORS["ink"])
    if ylabel:
        ax.set_ylabel(ylabel, color=PLOT_COLORS["ink"])


def plot_logistic_coefficients(fitted_pipeline, feature_names, top_n=12):
    coefs = pd.Series(
        fitted_pipeline.named_steps["model"].coef_[0],
        index=feature_names,
        name="scaled_weight",
    ).sort_values()

    selected = pd.concat([coefs.head(top_n // 2), coefs.tail(top_n // 2)])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PLOT_COLORS["rose"] if value < 0 else PLOT_COLORS["blue"] for value in selected.values]
    ax.barh(selected.index, selected.values, color=colors, edgecolor="white", linewidth=1)
    ax.axvline(0, color=PLOT_COLORS["ink"], linewidth=1.2)
    style_axis(
        ax,
        title="Scaled Logistic Regression Coefficients",
        xlabel="Coefficient on standardized features",
        ylabel="Feature",
        border_color=PLOT_COLORS["blue"],
    )
    plt.tight_layout()
    plt.show()

    return coefs.sort_values(key=np.abs, ascending=False)


def plot_importance_table(importance_df, x_col, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        importance_df["feature"],
        importance_df[x_col],
        xerr=importance_df.get("std"),
        color=PLOT_COLORS["blue"],
        alpha=0.9,
        edgecolor="white",
        linewidth=1,
    )
    ax.invert_yaxis()
    style_axis(
        ax,
        title=title,
        xlabel=x_col.replace("_", " ").title(),
        ylabel="Feature",
        border_color=PLOT_COLORS["teal"],
    )
    plt.tight_layout()
    plt.show()


def make_lime_explainer(X_train):
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["benign", "malignant"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )


def build_case_review_table(model, X_test, y_test, class_mapping, threshold=0.5):
    malignant_probability = model.predict_proba(X_test)[:, 1]
    predicted_label = (malignant_probability >= threshold).astype(int)

    review_df = pd.DataFrame(
        {
            "true_label": y_test.map(class_mapping),
            "predicted_label": pd.Series(predicted_label, index=X_test.index).map(class_mapping),
            "malignant_probability": malignant_probability,
            "confidence_gap": np.abs(malignant_probability - threshold),
        },
        index=X_test.index,
    )

    review_df["case_note"] = "general"

    malignant_mask = y_test == 1
    benign_mask = y_test == 0

    if malignant_mask.any():
        high_risk_idx = review_df.loc[malignant_mask, "malignant_probability"].idxmax()
        review_df.loc[high_risk_idx, "case_note"] = "high-risk malignant"

    borderline_idx = review_df["confidence_gap"].idxmin()
    review_df.loc[borderline_idx, "case_note"] = "borderline case"

    if benign_mask.any():
        confident_benign_idx = review_df.loc[benign_mask, "malignant_probability"].idxmin()
        review_df.loc[confident_benign_idx, "case_note"] = "confident benign"

    return review_df.sort_values(["case_note", "malignant_probability"], ascending=[True, False])

# %% [markdown]
# ## How To Work In Teams
#
# If you are working in pairs or small groups, a clean split is:
#
# - Group A: Sections 1 to 4
# - Group B: Section 5
# - Group C: Sections 6 to 8
#
# Then regroup and compare which methods are global, which are local, and which plots were easiest to explain.

# %% [markdown]
# ## 1. Dataset and Problem Setup
#
# We use the Breast Cancer Wisconsin dataset from `scikit-learn`.
#
# The target is recoded so:
#
# - `1 = malignant`
# - `0 = benign`
#
# We keep the dataset fixed throughout the notebook so the main comparison is between explanation methods, not between different datasets.

# %%
X, y, class_mapping = load_breast_cancer_frame()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("Class mapping:", class_mapping)
print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print("\nClass balance:")
display(y.value_counts(normalize=True).rename(index=class_mapping).to_frame("share").round(3))

correlation_snapshot = X[TOP_FEATURES].corr().round(3)
print("\nCorrelation snapshot for the PDP/ALE comparison:")
display(correlation_snapshot)

# %% [markdown]
# ### Quick Audit Note
#
# Before moving on:
#
# - confirm that the class balance is not extremely skewed;
# - notice that the two chosen `TOP_FEATURES` are strongly correlated;
# - remember that this correlation is exactly why `PDP` and `ALE` may tell slightly different stories later.

# %% [markdown]
# ## 2. White-Box Baseline: Logistic Regression
#
# We start with a standardized logistic-regression pipeline.
#
# The goal is not to claim that coefficients answer every question. The goal is to have a readable baseline before we move to more opaque models.

# %% [markdown]
# ### How To Read The Coefficient Plot
#
# - The **y-axis** lists the features.
# - The **x-axis** shows the standardized coefficient.
# - A **positive** coefficient means higher values push the prediction toward `malignant`.
# - A **negative** coefficient means higher values push the prediction toward `benign`.
# - Larger absolute values mean the feature matters more **inside this linear model**.

# %%
logistic_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=4000, random_state=RANDOM_STATE)),
    ]
)

# TODO:
# 1. Fit the logistic pipeline on the training split.
# 2. Evaluate it with evaluate_binary_model(...).
# 3. Plot the standardized coefficients.
#
# Hint:
# evaluate_binary_model(logistic_pipeline, X_train, X_test, y_train, y_test, "LogisticRegression")

logistic_pipeline.fit(...)
logistic_summary = evaluate_binary_model(...)
display(logistic_summary)

top_logistic_features = plot_logistic_coefficients(logistic_pipeline, X_train.columns)
display(top_logistic_features.head(10).to_frame("abs_scaled_weight"))

# %% [markdown]
# ## 3. Black-Box Model: Random Forest
#
# The Random Forest gives us a stronger but much less transparent comparison point.

# %%
rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
)
rf_clf.fit(X_train, y_train)

rf_summary = evaluate_binary_model(rf_clf, X_train, X_test, y_train, y_test, "RandomForest")
display(pd.concat([logistic_summary, rf_summary], ignore_index=True))

# %% [markdown]
# ### Quick Comparison Prompt
#
# In 1 to 2 sentences:
#
# - Which model looks stronger on the test split?
# - Does the gain in predictive performance look large enough to justify moving away from the more interpretable baseline?

# %% [markdown]
# ## 4. Global Explainability with Permutation Feature Importance
#
# `PFI` asks how much the model gets worse when a feature is shuffled.

# %% [markdown]
# ### How To Read The PFI Plot
#
# - The **y-axis** lists the features.
# - The **x-axis** shows the average drop in the chosen metric after shuffling that feature.
# - Larger values mean the model relied more on that feature.
# - Error bars show variability across repeated shuffles.
#
# Remember:
#
# - `PFI` is global;
# - `PFI` is not directional;
# - correlated features can split or hide importance.

# %%
# TODO:
# 1. Compute permutation_importance(...) on the test split.
# 2. Use ROC AUC as the scoring metric.
# 3. Build a small ranked DataFrame and plot it.
#
# Hint:
# permutation_importance(rf_clf, X_test, y_test, ..., scoring="roc_auc")

pfi = permutation_importance(
    ...,
    ...,
    ...,
    n_repeats=...,
    random_state=RANDOM_STATE,
    scoring=...,
)

pfi_df = (
    pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_importance": pfi.importances_mean,
            "std": pfi.importances_std,
        }
    )
    .sort_values("mean_importance", ascending=False)
    .head(12)
)

display(pfi_df.round(4))
plot_importance_table(pfi_df.iloc[::-1], "mean_importance", "Permutation Feature Importance (ROC AUC drop)")

# %% [markdown]
# ## 5. PDP vs ALE on Correlated Features
#
# We focus on one feature pair throughout the rest of the notebook:
#
# - main effect feature: `mean radius`
# - comparison / interaction feature: `mean perimeter`
#
# This keeps the visual story tighter and makes the `PDP` vs `ALE` comparison easier to read.

# %% [markdown]
# ### How To Read PDP And ALE
#
# For **both** plots:
#
# - the **x-axis** is the feature value;
# - the **y-axis** is the estimated effect or response;
# - the curve shape shows whether the relationship is flat, smooth, or nonlinear.
#
# Key difference:
#
# - `PDP` averages over synthetic substitutions;
# - `ALE` accumulates local changes where the data actually exists.
# - In this notebook we compare them on the **same feature** so the visual difference is easier to see.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
PartialDependenceDisplay.from_estimator(
    rf_clf,
    X_train,
    features=[FOCUS_FEATURE],
    kind="average",
    grid_resolution=30,
    line_kw={"color": PLOT_COLORS["blue"], "linewidth": 3},
    ax=axes[0],
)
ale_predictor = lambda values: rf_clf.predict_proba(
    pd.DataFrame(values, columns=X_train.columns)
)[:, 1]
ale_explainer = ALE(ale_predictor, feature_names=X_train.columns.tolist())
ale_exp = ale_explainer.explain(X_train.values)

plot_ale(ale_exp, features=[FOCUS_FEATURE], ax=axes[1])
if axes[1].lines:
    axes[1].lines[0].set_color(PLOT_COLORS["teal"])
    axes[1].lines[0].set_linewidth(3)

style_axis(
    axes[0],
    title=f"PDP: {FOCUS_FEATURE}",
    xlabel=FOCUS_FEATURE,
    ylabel="Predicted malignant probability",
    border_color=PLOT_COLORS["blue"],
)
style_axis(
    axes[1],
    title=f"ALE: {FOCUS_FEATURE}",
    xlabel=FOCUS_FEATURE,
    ylabel="Accumulated local effect",
    border_color=PLOT_COLORS["teal"],
)
fig.suptitle(
    f"Comparing Total vs Main Effects for {FOCUS_FEATURE}",
    fontsize=18,
    fontweight="bold",
    color=PLOT_COLORS["ink"],
    y=1.04,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visual Takeaway Prompt
#
# Use the paired figure above and write 2 to 3 sentences:
#
# - Do `PDP` and `ALE` agree on the overall direction of the trend?
# - Which curve looks smoother or more stable?
# - Where might correlation or unrealistic substitutions explain a mismatch?

# %% [markdown]
# **Short-answer task:**
#
# Compare the `PDP` and `ALE` stories with the earlier `PFI` ranking.
#
# - Does a feature being globally important automatically mean its effect curve is simple?
# - What does this section add that `PFI` alone could not tell you?

# %% [markdown]
# ## 6. Local Explainability with LIME
#
# LIME fits a simple local surrogate around one selected case.

# %% [markdown]
# ### How To Read The LIME Plot
#
# - Each bar is one local feature contribution.
# - Bars pushing right or left support different classes.
# - The rules are local threshold statements for this case only.
#
# This is a **local** explanation, not a global summary.

# %%
candidate_case_table = build_case_review_table(rf_clf, X_test, y_test, class_mapping)
candidate_cases = candidate_case_table.loc[
    candidate_case_table["case_note"] != "general",
    ["true_label", "predicted_label", "malignant_probability", "case_note"],
].round(3)

print("Candidate cases for local explanations:")
display(candidate_cases)

# %% [markdown]
# ### Case-Level Prediction Explanations
#
# Use the table above to choose a few contrasting predictions.
#
# Good classroom choices are:
#
# - one confident malignant case,
# - one borderline case,
# - one confident benign case.
#
# The goal is not only to ask **what** the model predicted, but also **why** the model behaved differently across cases.

# %%
lime_explainer = make_lime_explainer(X_train)

# TODO:
# 1. Choose one interesting test case.
# 2. Explain that case with LimeTabularExplainer.
# 3. Plot the explanation for the malignant class.
#
# Hint:
# a good first choice is a high-risk case, for example
# np.argmax(rf_clf.predict_proba(X_test)[:, 1])

lime_index = ...
print("Explaining test-row index:", lime_index)
print("True label:", class_mapping[int(y_test.iloc[lime_index])])
print("Predicted malignant probability:", round(rf_clf.predict_proba(X_test.iloc[[lime_index]])[0, 1], 3))

lime_exp = lime_explainer.explain_instance(
    X_test.iloc[lime_index].values,
    rf_clf.predict_proba,
    num_features=8,
)

fig = lime_exp.as_pyplot_figure(label=1)
fig.set_size_inches(10, 5)
plt.title("LIME Explanation for One Test Case")
plt.tight_layout()
plt.show()

display(pd.DataFrame(lime_exp.as_list(label=1), columns=["rule", "weight"]))

# %% [markdown]
# ### Optional Notebook LIME View
#
# If your notebook frontend supports it, inspect one case in the richer notebook-native LIME format.
# Keep this as a separate case view rather than mixing several cases together.

# %%
# TODO:
# Pick one case index from candidate_cases for a notebook-native LIME explanation.
lime_notebook_case = candidate_cases.index[0]

print(f"Notebook LIME case: {lime_notebook_case}")
lime_notebook_exp = lime_explainer.explain_instance(
    X_test.loc[lime_notebook_case].values,
    rf_clf.predict_proba,
    num_features=8,
)
display(HTML(lime_notebook_exp.as_html(labels=(1,))))

# %% [markdown]
# ### Optional LIME Case Gallery
#
# Choose 2 or 3 indices from `candidate_cases` and run the next cell.
# Keep the plots separate so you can compare them one by one.

# %%
# TODO:
# Replace the example list with 2 or 3 indices from candidate_cases.
lime_case_gallery = candidate_cases.index[:2].tolist()

for case_idx in lime_case_gallery:
    case_probability = rf_clf.predict_proba(X_test.loc[[case_idx]])[0, 1]
    case_label = class_mapping[int(y_test.loc[case_idx])]

    print(f"Case {case_idx} | true={case_label} | malignant_probability={case_probability:.3f}")
    case_exp = lime_explainer.explain_instance(
        X_test.loc[case_idx].values,
        rf_clf.predict_proba,
        num_features=8,
    )

    fig = case_exp.as_pyplot_figure(label=1)
    fig.set_size_inches(10, 4.8)
    plt.title(f"LIME local explanation for case {case_idx}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Global + Local Explainability with SHAP
#
# Here we use two SHAP plots on purpose:
#
# - `beeswarm` = global
# - `waterfall` = local

# %% [markdown]
# ### How To Read SHAP Beeswarm And Waterfall
#
# For the **beeswarm**:
#
# - each dot is one row;
# - the **x-axis** is the SHAP contribution;
# - the **y-axis** is the ranked feature list;
# - color shows whether the feature value is high or low.
#
# For the **waterfall**:
#
# - start from the baseline prediction;
# - each bar moves the prediction up or down for one case;
# - the final value is the model output for that case.

# %%
shap_background = X_train.sample(n=min(120, len(X_train)), random_state=RANDOM_STATE)
shap_sample = X_test.sample(n=min(80, len(X_test)), random_state=RANDOM_STATE)

# TODO:
# 1. Build a SHAP explainer for the random forest.
# 2. Compute SHAP values on shap_sample.
# 3. Draw a beeswarm plot for the malignant class.
#
# Hint:
# the malignant class is class index `1` in the SHAP output here.

shap_explainer = shap.Explainer(...)
shap_values = shap_explainer(...)

plt.figure()
shap.plots.beeswarm(shap_values[..., 1], max_display=12, show=False)
plt.title("SHAP Beeswarm for the Malignant Class")
plt.tight_layout()
plt.show()

# %%
# TODO:
# 1. Choose one row from shap_sample.
# 2. Draw a SHAP waterfall plot for that case.
# 3. Write 2 to 3 sentences explaining which features pushed the prediction most.

waterfall_index = ...
plt.figure()
shap.plots.waterfall(shap_values[waterfall_index, ..., 1], max_display=10, show=False)
plt.title("SHAP Waterfall for One Test Case")
plt.tight_layout()
plt.show()

print("Explained SHAP sample row index:", waterfall_index)
display(shap_sample.iloc[[waterfall_index]])

# %% [markdown]
# ### Optional Notebook SHAP Force View
#
# This is another single-case explanation, but in the notebook-native SHAP force style.
# Use it for one chosen case at a time.

# %%
shap.initjs()

# TODO:
# Pick one position from shap_local_cases after you build the local SHAP frame below,
# or choose a single test case directly and explain it here.
single_force_case = candidate_cases.index[0]
single_force_frame = X_test.loc[[single_force_case]]
single_force_values = shap_explainer(single_force_frame)

print(f"Force-plot case: {single_force_case}")
print("True label:", class_mapping[int(y_test.loc[single_force_case])])
print("Predicted malignant probability:", round(rf_clf.predict_proba(single_force_frame)[0, 1], 3))

shap.plots.force(single_force_values[0, :, 1])

# %% [markdown]
# ### Optional SHAP Waterfall Gallery
#
# Build a few separate waterfall plots for contrasting predictions.
# Do **not** merge them into one large chart. The goal is to inspect each case on its own.

# %%
shap_local_cases = candidate_cases.index[:3].tolist()
shap_local_frame = X_test.loc[shap_local_cases]
shap_local_values = shap_explainer(shap_local_frame)

# TODO:
# Replace the example list with 2 or 3 positions from the local SHAP case list below.
display(pd.DataFrame({"position": range(len(shap_local_cases)), "case_index": shap_local_cases}))

waterfall_gallery_positions = [0, 1]

for position in waterfall_gallery_positions:
    case_idx = shap_local_cases[position]
    case_probability = rf_clf.predict_proba(X_test.loc[[case_idx]])[0, 1]
    case_label = class_mapping[int(y_test.loc[case_idx])]

    print(f"Case {case_idx} | true={case_label} | malignant_probability={case_probability:.3f}")
    plt.figure()
    shap.plots.waterfall(shap_local_values[position, ..., 1], max_display=10, show=False)
    plt.title(f"SHAP waterfall for case {case_idx}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Optional Interaction Effects with SHAP
#
# The earlier sections covered:
#
# - global feature ranking,
# - total effects,
# - main effects,
# - local explanations for one case.
#
# A lightweight next step is to ask whether the contribution of one feature changes depending on another feature.

# %% [markdown]
# ### How To Read The SHAP Interaction Plot
#
# - The **x-axis** is the actual value of the focus feature.
# - The **y-axis** is the SHAP contribution of that same feature.
# - The **color** shows the value of the second feature.
#
# What to look for:
#
# - if the color pattern changes systematically along the curve, the second feature may be modifying the first feature's contribution;
# - if points with similar `x` values still spread out vertically, that can also hint at interactions or subgroup effects.

# %%
plt.figure(figsize=(10, 6))
shap.dependence_plot(
    FOCUS_FEATURE,
    shap_values[..., 1].values,
    shap_sample,
    interaction_index=INTERACTION_FEATURE,
    show=False,
)
style_axis(
    plt.gca(),
    title=f"SHAP Dependence: {FOCUS_FEATURE} colored by {INTERACTION_FEATURE}",
    xlabel=FOCUS_FEATURE,
    ylabel=f"SHAP value for {FOCUS_FEATURE}",
    border_color=PLOT_COLORS["orange"],
)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Optional interpretation prompt:**
#
# In 2 to 3 sentences:
#
# - does the effect of `mean radius` look constant across all values of `mean perimeter`?
# - does this plot suggest a real interaction pattern, or mostly strong correlation / shared structure?

# %% [markdown]
# ## 9. InterpretML Extension
#
# We finish with an Explainable Boosting Machine (`InterpretML`).
#
# This gives us a modern glass-box comparison point: often more flexible than a linear model, but still designed to remain inspectable.

# %% [markdown]
# ### How To Read The InterpretML Plot
#
# - The **y-axis** lists model terms.
# - The **x-axis** shows their global importance inside the EBM.
# - Larger values mean the EBM relied more on that term overall.

# %%
ebm = ExplainableBoostingClassifier(
    interactions=4,
    max_bins=64,
    random_state=RANDOM_STATE,
)
ebm.fit(X_train, y_train)

ebm_summary = evaluate_binary_model(ebm, X_train, X_test, y_train, y_test, "EBM")
display(pd.concat([logistic_summary, rf_summary, ebm_summary], ignore_index=True))

ebm_importance = (
    pd.Series(ebm.term_importances(), index=ebm.term_names_, name="importance")
    .sort_values(ascending=False)
    .head(12)
    .reset_index()
    .rename(columns={"index": "feature"})
)

plot_importance_table(ebm_importance.iloc[::-1], "importance", "InterpretML EBM Term Importance")
display(ebm_importance.head(10).round(4))

# %% [markdown]
# ## 10. Debrief
#
# Answer briefly:
#
# 1. Which plot would you use to explain **one prediction** to a stakeholder?
# 2. Which plot would you use to inspect the model **globally**?
# 3. Why is `PFI` different from coefficient size?
# 4. Why can `ALE` be safer than `PDP` when features are correlated?
# 5. What did the SHAP interaction plot add beyond `PDP`, `ALE`, and the local waterfall?
# 6. Why is "important for the model" not the same as "causes the outcome"?
