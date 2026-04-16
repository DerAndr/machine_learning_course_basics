# %% [markdown]
# # Responsible AI: Practical Session — STUDENT VERSION (90 minutes)
# 
# **Learning objectives:**
# - assess model fairness with Fairlearn’s `MetricFrame` and scalar disparity metrics;
# - compare multiple fairness definitions: demographic parity, equalized odds, equal opportunity;
# - apply two mitigation strategies: post-processing (`ThresholdOptimizer`) and in-processing (`ExponentiatedGradient`);
# - analyze fairness across multiple sensitive features (sex **and** race);
# - build a lightweight Model Card to document model behavior.
# 
# **Dataset:** [UCI Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) via `sklearn.datasets.fetch_openml`.
# 
# **Tools:**
# - [Fairlearn](https://fairlearn.org/) — fairness assessment and mitigation (mandatory)
# - [VerifyML](https://github.com/cylynx/verifyml) — model card generation toolkit (mentioned for reference)
# - [What-If Tool](https://pair-code.github.io/what-if-tool/) — interactive model exploration (mentioned for reference)

# %% [markdown]
# ## Setup
# 
# For local work in this repository, prefer:
# 
# ```bash
# uv sync
# uv run python tools/check_notebook_environment.py
# ```

# %%
# Auto-install fairlearn if running outside the managed environment (e.g. Google Colab)
try:
    import fairlearn
except ImportError:
    # NOTE: notebook magic commented for local script use: %pip install -q fairlearn

# %% [markdown]
# ## Imports

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
)
from sklearn.model_selection import train_test_split

from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    mean_prediction,
    plot_model_comparison as fairlearn_plot_model_comparison,
    selection_rate,
    selection_rate_difference,
    selection_rate_ratio,
    true_negative_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    EqualizedOdds,
    ErrorRate,
    ExponentiatedGradient,
)

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# %% [markdown]
# ## Fairness Metric Glossary
# 
# Before diving in, here is a quick reference for the fairness metrics we will use throughout this practical.
# 
# | Metric | Formula / Meaning | Perfect Value | When to Use |
# |---|---|---|---|
# | **Selection Rate** | $\frac{\text{\# predicted positive}}{\text{\# total}}$ | Equal across groups | When the **rate of positive outcomes** matters (e.g., loan approvals). |
# | **Demographic Parity Difference (DPD)** | $\max_{g} SR_g - \min_{g} SR_g$ | 0 | When you want **equal selection rates** regardless of ground-truth labels. |
# | **Demographic Parity Ratio (DPR)** | $\frac{\min_{g} SR_g}{\max_{g} SR_g}$ | 1 (≥ 0.8 = “four-fifths rule”) | Industry rule-of-thumb for adverse impact. |
# | **True Positive Rate (TPR) / Recall** | $\frac{TP}{TP + FN}$ | Equal across groups | When **missing a positive case** is costly (e.g., disease screening). |
# | **False Positive Rate (FPR)** | $\frac{FP}{FP + TN}$ | Equal across groups | When **false alarms** disproportionately harm a group (e.g., criminal justice). |
# | **False Negative Rate (FNR)** | $\frac{FN}{FN + TP} = 1 - TPR$ | Equal across groups | Complement of TPR. |
# | **True Negative Rate (TNR)** | $\frac{TN}{TN + FP} = 1 - FPR$ | Equal across groups | Complement of FPR. |
# | **Equal Opportunity Difference** | $\max_{g} TPR_g - \min_{g} TPR_g$ | 0 | When you care about **equal recall** for the positive class. |
# | **Equalized Odds Difference (EOD)** | $\max(\Delta TPR, \Delta FPR)$ across groups | 0 | Strictest pairwise criterion — requires **both TPR and FPR** to be equal. |
# | **Equalized Odds Ratio** | $\min(TPR_{ratio}, FPR_{ratio})$ | 1 | Ratio variant of EOD; easier to interpret as a “how close to parity” number. |
# | **Mean Prediction** | Average predicted label | Close across groups | Quick sanity check — is the model predicting positive equally often? |
# 
# > **Key insight:** These metrics can **conflict**. Satisfying demographic parity does **not** guarantee equalized odds, and vice versa (Chouldechova, 2017; Kleinberg et al., 2016). Choosing the right metric depends on the specific harm you want to prevent.
# >
# > **References:**
# > - Fairlearn common fairness metrics: [fairlearn.org/main/user_guide/assessment/common_fairness_metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html)
# > - Barocas, Hardt, Narayanan, *Fairness and Machine Learning*: [fairmlbook.org](http://www.fairmlbook.org/), chapters 1–2
# > - Verma & Rubin (2018), “Fairness definitions explained”: [DOI:10.1145/3194770.3194776](https://doi.org/10.1145/3194770.3194776)

# %% [markdown]
# ## Shared Helper Functions

# %%
def load_adult_dataset():
    """Load and prepare the Adult Census Income dataset."""
    print("Loading Adult Census dataset from OpenML...")
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = data.frame

    features = ["age", "education-num", "hours-per-week", "sex", "race", "class"]
    df = df[features].dropna()

    df["target"] = df["class"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    df = df.drop(columns=["class"])

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
    return df


def build_metric_frame(y_true, y_pred, sensitive_features, label=""):
    """Build a MetricFrame with a comprehensive set of fairness-relevant metrics."""
    metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        "recall (TPR)": true_positive_rate,
        "FPR": false_positive_rate,
        "FNR": false_negative_rate,
        "TNR": true_negative_rate,
        "selection_rate": selection_rate,
        "mean_prediction": mean_prediction,
        "count": count,
    }
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print("\nOverall metrics:")
    display(mf.overall.to_frame("value").T)
    print("\nMetrics by group:")
    display(mf.by_group)
    print("\nMax difference across groups (per metric):")
    display(mf.difference().to_frame("max_diff").T)
    return mf


def scalar_fairness_summary(y_true, y_pred, sensitive_features, label=""):
    """Print a comprehensive set of scalar fairness metrics."""
    results = {}
    results["sel_rate_diff"] = selection_rate_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    results["sel_rate_ratio"] = selection_rate_ratio(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    results["dpd"] = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    results["dpr"] = demographic_parity_ratio(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    results["eod"] = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    results["eor"] = equalized_odds_ratio(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    results["eo_diff"] = equal_opportunity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    results["eo_ratio"] = equal_opportunity_ratio(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    if label:
        print(f"\n--- {label} ---")
    print(f"  Selection Rate Difference      : {results['sel_rate_diff']:+.4f}  (0 = parity)")
    print(f"  Selection Rate Ratio           : {results['sel_rate_ratio']:.4f}   (1 = parity, >=0.8 = four-fifths rule)")
    print(f"  Demographic Parity Difference  : {results['dpd']:.4f}   (0 = parity)")
    print(f"  Demographic Parity Ratio       : {results['dpr']:.4f}   (1 = parity)")
    print(f"  Equal Opportunity Difference   : {results['eo_diff']:.4f}   (0 = parity, measures TPR gap)")
    print(f"  Equal Opportunity Ratio        : {results['eo_ratio']:.4f}   (1 = parity)")
    print(f"  Equalized Odds Difference      : {results['eod']:.4f}   (0 = parity, max of TPR & FPR gap)")
    print(f"  Equalized Odds Ratio           : {results['eor']:.4f}   (1 = parity)")
    return results


def plot_fairness_dashboard(mf, title="Fairness Dashboard"):
    """Multi-panel dashboard: selection rate, TPR/FNR, FPR/TNR, accuracy by group."""
    by_group = mf.by_group
    groups = by_group.index.astype(str)
    x = np.arange(len(groups))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # Panel 1: Selection Rate
    ax = axes[0, 0]
    bars = ax.bar(x, by_group["selection_rate"], color="steelblue", alpha=0.8)
    ax.axhline(mf.overall["selection_rate"], color="red", ls="--", lw=1.2, label="overall")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title("Selection Rate (P(y_hat=1))"); ax.set_ylim(0, None); ax.legend(fontsize=8)
    for bar, val in zip(bars, by_group["selection_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Panel 2: TPR (Recall)
    ax = axes[0, 1]
    bars = ax.bar(x, by_group["recall (TPR)"], color="forestgreen", alpha=0.8)
    ax.axhline(mf.overall["recall (TPR)"], color="red", ls="--", lw=1.2, label="overall")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title("True Positive Rate (Recall)"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    for bar, val in zip(bars, by_group["recall (TPR)"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Panel 3: FPR
    ax = axes[0, 2]
    bars = ax.bar(x, by_group["FPR"], color="tomato", alpha=0.8)
    ax.axhline(mf.overall["FPR"], color="red", ls="--", lw=1.2, label="overall")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title("False Positive Rate"); ax.set_ylim(0, None); ax.legend(fontsize=8)
    for bar, val in zip(bars, by_group["FPR"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Panel 4: FNR
    ax = axes[1, 0]
    bars = ax.bar(x, by_group["FNR"], color="darkorange", alpha=0.8)
    ax.axhline(mf.overall["FNR"], color="red", ls="--", lw=1.2, label="overall")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title("False Negative Rate (1 - TPR)"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    for bar, val in zip(bars, by_group["FNR"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Panel 5: Accuracy
    ax = axes[1, 1]
    bars = ax.bar(x, by_group["accuracy"], color="mediumpurple", alpha=0.8)
    ax.axhline(mf.overall["accuracy"], color="red", ls="--", lw=1.2, label="overall")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title("Accuracy"); ax.set_ylim(0.5, 1); ax.legend(fontsize=8)
    for bar, val in zip(bars, by_group["accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Panel 6: Mean Prediction
    ax = axes[1, 2]
    bars = ax.bar(x, by_group["mean_prediction"], color="teal", alpha=0.8)
    ax.axhline(mf.overall["mean_prediction"], color="red", ls="--", lw=1.2, label="overall")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title("Mean Prediction (avg y_hat)"); ax.set_ylim(0, None); ax.legend(fontsize=8)
    for bar, val in zip(bars, by_group["mean_prediction"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_metric_comparison(mf_before, mf_after, title="Before vs After Mitigation"):
    """Side-by-side bar chart comparing group metrics before and after mitigation."""
    plot_metrics = ["accuracy", "recall (TPR)", "selection_rate", "FPR", "FNR"]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(18, 4.5), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for ax, metric in zip(axes, plot_metrics):
        before_vals = mf_before.by_group[metric]
        after_vals = mf_after.by_group[metric]
        groups = before_vals.index.astype(str)
        x = np.arange(len(groups))
        w = 0.35

        ax.bar(x - w / 2, before_vals.values, w, label="Before", alpha=0.8, color="steelblue")
        ax.bar(x + w / 2, after_vals.values, w, label="After", alpha=0.8, color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_title(metric)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_intersectional_heatmap(y_true, y_pred, sf1, sf2, metric_fn, metric_name, sf1_name, sf2_name):
    """Heatmap of a metric for all intersectional subgroups."""
    df_tmp = pd.DataFrame({
        "y_true": y_true.values,
        "y_pred": y_pred,
        sf1_name: sf1.values,
        sf2_name: sf2.values,
    })
    pivot = df_tmp.groupby([sf1_name, sf2_name])[["y_true", "y_pred"]].apply(
        lambda g: metric_fn(g["y_true"], g["y_pred"]) if len(g) > 10 else np.nan
    ).unstack()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, linewidths=0.5, vmin=0, vmax=1)
    ax.set_title(f"{metric_name} by {sf1_name} x {sf2_name}", fontweight="bold")
    plt.tight_layout()
    plt.show()
    return pivot

# %% [markdown]
# ---
# 
# ## 1. Dataset and Baseline Model

# %%
df = load_adult_dataset()
display(df.head())

# %%
# Sensitive features: keep as strings for clarity in plots
sensitive_sex = df["sex"]
sensitive_race = df["race"]

# Encode sex for model input
df["sex_encoded"] = df["sex"].map({"Male": 1, "Female": 0})

X = df[["age", "education-num", "hours-per-week", "sex_encoded"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

# Preserve sensitive features aligned with train/test splits
sex_train = sensitive_sex.loc[X_train.index]
sex_test = sensitive_sex.loc[X_test.index]
race_test = sensitive_race.loc[X_test.index]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# %%
# Train a baseline model
baseline_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=RANDOM_STATE
)
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
print(f"Baseline Accuracy:          {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Baseline Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_baseline):.4f}")

# %% [markdown]
# ---
# 
# ## 2. Fairness Assessment: Sex as Sensitive Feature ✏️ TODO
# 
# Use `build_metric_frame` and `scalar_fairness_summary` to assess the baseline model’s fairness with respect to **sex**.
# 
# **What to look for:**
# - Which group has a higher **selection rate**? (A selection rate gap means the model disproportionately predicts positive outcomes for one group.)
# - Which group has a higher **recall (TPR)**? (A TPR gap means the model misses more true positives in one group — this is what **equal opportunity** measures.)
# - What does the **demographic parity difference** tell us? (It measures the absolute gap in selection rates; values > 0.1 are often considered concerning.)
# - Does the **four-fifths rule** hold? (Demographic Parity Ratio ≥ 0.8 is a commonly used legal threshold in US employment law.)

# %%
# TODO:
# 1. Call build_metric_frame with y_test, y_pred_baseline, and sex_test.
# 2. Call scalar_fairness_summary with the same arguments.
#
# Hint: build_metric_frame(y_test, y_pred_baseline, sex_test, label="Baseline - Sex")

mf_sex_baseline = build_metric_frame(...)

scalar_sex_baseline = scalar_fairness_summary(...)

# %% [markdown]
# ### Fairness Dashboard: Visual Overview
# 
# The dashboard below shows six key metrics broken down by group. The **red dashed line** marks the overall (pooled) value. Disparities appear as unequal bar heights.
# 
# > **How to read it:**
# > - **Selection Rate**: if bars differ, the model predicts “high income” more often for one group → potential demographic parity violation.
# > - **TPR (Recall)**: if bars differ, the model catches actual high-earners better in one group → equal opportunity violation.
# > - **FPR**: if bars differ, the model produces more false alarms in one group → important for equalized odds.
# > - **FNR = 1 − TPR**: mirrors TPR; a high FNR means many real positives are being missed.
# > - **Accuracy**: overall correctness per group; can mask disparity if class balance differs.
# > - **Mean Prediction**: average predicted label; another angle on selection rate.

# %%
plot_fairness_dashboard(mf_sex_baseline, title="Baseline Fairness Dashboard — Sex")

# %% [markdown]
# ### All Metrics by Group (Detail View)

# %%
mf_sex_baseline.by_group.plot(
    kind="bar", subplots=True, layout=(2, 5), figsize=(20, 8),
    title="Baseline — All Metrics by Sex", legend=False,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation Guide
# 
# **Questions to discuss:**
# 1. Which group has a higher selection rate? What does this mean in the context of income prediction?
# 2. Look at the **Demographic Parity Ratio** in the scalar summary. Does it pass the four-fifths rule (≥ 0.8)?
# 3. Compare the **TPR gap** (Equal Opportunity Difference) with the **FPR gap**. Which kind of error is more unequally distributed?
# 4. The lecture mentions that **no single fairness metric solves every problem** (Chouldechova, 2017). Which metric would you prioritize here, and why?
# 
# > **Reference:** Fairlearn docs explain each metric with mathematical formulas and worked examples:
# > [fairlearn.org/main/user_guide/assessment/common_fairness_metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html)

# %% [markdown]
# ---
# 
# ## 3. Fairness Assessment: Race as Sensitive Feature ✏️ TODO
# 
# Repeat the same assessment, but now using **race** as the sensitive feature.
# 
# The Adult dataset has five race categories. With more groups, the maximum pairwise difference tends to be **larger** — even small per-group variations compound. This is important to keep in mind when setting fairness thresholds.

# %%
# TODO:
# 1. Call build_metric_frame with y_test, y_pred_baseline, and race_test.
# 2. Call scalar_fairness_summary with the same arguments.
#
# Note: with more groups, the differences can be larger.

mf_race_baseline = build_metric_frame(...)

scalar_race_baseline = scalar_fairness_summary(...)

# %%
plot_fairness_dashboard(mf_race_baseline, title="Baseline Fairness Dashboard — Race")

# %%
mf_race_baseline.by_group.plot(
    kind="bar", subplots=True, layout=(2, 5), figsize=(20, 8),
    title="Baseline — All Metrics by Race", legend=False,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation Guide
# 
# **Questions to discuss:**
# 1. Which racial groups have the highest and lowest selection rates?
# 2. Are the disparities in recall and FPR consistent across groups, or are some groups affected differently?
# 3. How does the equalized odds difference compare between the sex and race analyses?
# 4. With five groups, the `count` column matters — groups with very few samples produce noisy metric estimates. Which groups should you be cautious about?
# 
# > **Reference:** Verma & Rubin (2018) catalog 20+ fairness definitions and show how they relate to each other:
# > [DOI:10.1145/3194770.3194776](https://doi.org/10.1145/3194770.3194776)

# %% [markdown]
# ---
# 
# ## 4. Mitigation Strategy 1: Post-Processing with ThresholdOptimizer ✏️ TODO
# 
# **ThresholdOptimizer** adjusts decision thresholds per group **after** the model is trained.
# 
# It finds optimal group-specific thresholds to satisfy a fairness constraint (e.g., equalized odds or demographic parity).
# 
# **How it works:**
# 1. The original model provides probability scores via `predict_proba`.
# 2. For each group, ThresholdOptimizer searches for a threshold that best satisfies the constraint.
# 3. At prediction time, each sample’s group determines which threshold is applied.
# 
# **Key trade-off:** post-processing can improve fairness metrics, but it may reduce overall accuracy because it deliberately adjusts predictions for some groups.
# 
# > **Reference:** Hardt, Price, Srebro (2016), “Equality of Opportunity in Supervised Learning”: [arxiv.org/abs/1610.02413](https://arxiv.org/abs/1610.02413)

# %%
# TODO:
# 1. Create a ThresholdOptimizer with:
#    - estimator=baseline_model
#    - constraints="equalized_odds"
#    - prefit=True
#    - predict_method='predict_proba'
# 2. Fit it on X_train, y_train with sensitive_features=sex_train.
# 3. Predict on X_test with sensitive_features=sex_test.
#
# Hint: The optimizer wraps the existing model; it does NOT retrain it.

threshold_optimizer = ThresholdOptimizer(
    ...
)

threshold_optimizer.fit(...)

y_pred_threshold = threshold_optimizer.predict(...)

print("ThresholdOptimizer fitting complete.")
print(f"Mitigated Accuracy:          {accuracy_score(y_test, y_pred_threshold):.4f}")
print(f"Mitigated Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_threshold):.4f}")

# %% [markdown]
# ### Evaluate ThresholdOptimizer Fairness

# %%
mf_sex_threshold = build_metric_frame(
    y_test, y_pred_threshold, sex_test, label="ThresholdOptimizer - Sex"
)

scalar_sex_threshold = scalar_fairness_summary(
    y_test, y_pred_threshold, sex_test, label="ThresholdOptimizer - Sex"
)

# %%
plot_fairness_dashboard(mf_sex_threshold, title="ThresholdOptimizer Fairness Dashboard — Sex")

# %%
plot_metric_comparison(
    mf_sex_baseline, mf_sex_threshold,
    title="Baseline vs ThresholdOptimizer (Sex)"
)

# %% [markdown]
# ### Interpretation Guide
# 
# **Questions to discuss:**
# 1. Did the equalized odds difference decrease after mitigation? By how much?
# 2. What happened to overall accuracy? Is the trade-off acceptable?
# 3. Look at the **Equal Opportunity Difference** (TPR gap) before and after. Did it improve?
# 4. ThresholdOptimizer is a **post-processing** method. Advantages: no retraining needed, model-agnostic. Limitations: requires group membership at prediction time, may not generalize to unseen groups.
# 
# > **When to use:** ThresholdOptimizer is ideal when you have a strong existing model and want to adjust fairness without retraining. It is widely used in industry because it is fast and predictable.

# %% [markdown]
# ---
# 
# ## 5. Mitigation Strategy 2: In-Processing with ExponentiatedGradient ✏️ TODO
# 
# **ExponentiatedGradient** (Agarwal et al., 2018) is an in-processing technique: it retrains the model with a fairness constraint baked into the optimization.
# 
# **How it works:**
# 1. It solves a min-max game: minimize classification error subject to a fairness constraint.
# 2. The algorithm iteratively reweights training samples to push the model toward satisfying the constraint.
# 3. The result is a **randomized classifier** — an ensemble of models with associated weights.
# 
# Unlike ThresholdOptimizer, it produces a **new** model that is inherently fairness-aware.
# 
# We will use `EqualizedOdds` as the constraint with a small `difference_bound`.
# 
# > **Reference:** Agarwal et al. (2018), “A Reductions Approach to Fair Classification”: [arxiv.org/abs/1803.02453](https://arxiv.org/abs/1803.02453)

# %%
# TODO:
# 1. Create an ExponentiatedGradient mitigator with:
#    - estimator = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
#    - constraints = EqualizedOdds(difference_bound=0.01)
#    - objective = ErrorRate(costs={'fp': 0.1, 'fn': 0.9})
# 2. Fit it on X_train, y_train with sensitive_features=sex_train.
# 3. Predict on X_test.
#
# Note: this takes longer than ThresholdOptimizer because it retrains models.

constraint = EqualizedOdds(...)
objective = ErrorRate(...)

mitigator = ExponentiatedGradient(
    ...
)

print("Training ExponentiatedGradient (this may take a minute)...")
mitigator.fit(...)

y_pred_eg = mitigator.predict(X_test)

print(f"\nExponentiatedGradient Accuracy:          {accuracy_score(y_test, y_pred_eg):.4f}")
print(f"ExponentiatedGradient Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_eg):.4f}")

# %% [markdown]
# ### Evaluate ExponentiatedGradient Fairness

# %%
mf_sex_eg = build_metric_frame(
    y_test, y_pred_eg, sex_test, label="ExponentiatedGradient - Sex"
)

scalar_sex_eg = scalar_fairness_summary(
    y_test, y_pred_eg, sex_test, label="ExponentiatedGradient - Sex"
)

# %%
plot_fairness_dashboard(mf_sex_eg, title="ExponentiatedGradient Fairness Dashboard — Sex")

# %%
plot_metric_comparison(
    mf_sex_baseline, mf_sex_eg,
    title="Baseline vs ExponentiatedGradient (Sex)"
)

# %% [markdown]
# ---
# 
# ## 6. Comparison: All Three Approaches ✏️ TODO
# 
# Build a summary table and a scatter plot comparing the baseline, ThresholdOptimizer, and ExponentiatedGradient.

# %%
# TODO:
# Build a summary DataFrame comparing all three approaches.
# Include: accuracy, balanced_accuracy, and fairness metrics for each.
#
# Hint: You already have scalar_sex_baseline, scalar_sex_threshold, scalar_sex_eg.

summary = pd.DataFrame({
    "Baseline": {
        "Accuracy": ...,
        "Balanced Accuracy": ...,
        "Selection Rate Diff": ...,
        "Demographic Parity Diff": ...,
        "Demographic Parity Ratio": ...,
        "Equal Opportunity Diff": ...,
        "Equalized Odds Diff": ...,
        "Equalized Odds Ratio": ...,
    },
    "ThresholdOptimizer": {
        "Accuracy": ...,
        "Balanced Accuracy": ...,
        "Selection Rate Diff": ...,
        "Demographic Parity Diff": ...,
        "Demographic Parity Ratio": ...,
        "Equal Opportunity Diff": ...,
        "Equalized Odds Diff": ...,
        "Equalized Odds Ratio": ...,
    },
    "ExponentiatedGradient": {
        "Accuracy": ...,
        "Balanced Accuracy": ...,
        "Selection Rate Diff": ...,
        "Demographic Parity Diff": ...,
        "Demographic Parity Ratio": ...,
        "Equal Opportunity Diff": ...,
        "Equalized Odds Diff": ...,
        "Equalized Odds Ratio": ...,
    },
})

display(summary.round(4))

# %% [markdown]
# ### Model Comparison Scatter Plot
# 
# Fairlearn’s `plot_model_comparison` places each model on a 2D plane:
# - **x-axis**: performance metric (balanced accuracy — higher is better)
# - **y-axis**: fairness metric (equalized odds difference — lower is better)
# 
# The ideal model sits in the **bottom-right corner** (high performance, low disparity).

# %%
fig, ax = plt.subplots(figsize=(8, 6))
fairlearn_plot_model_comparison(
    y_preds={"Baseline": y_pred_baseline, "ThresholdOpt": y_pred_threshold, "ExpGrad": y_pred_eg},
    y_true=y_test,
    sensitive_features=sex_test,
    x_axis_metric=balanced_accuracy_score,
    y_axis_metric=equalized_odds_difference,
    show_plot=False,
    point_labels=True,
    point_labels_position=(0.002, 0.002),
    ax=ax,
)
ax.set_xlabel("Balanced Accuracy (higher is better)")
ax.set_ylabel("Equalized Odds Difference (lower is better)")
ax.set_title("Performance vs Fairness Trade-Off", fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation Guide
# 
# **Questions to discuss:**
# 1. Which method provides the best fairness-accuracy trade-off in the scatter plot?
# 2. Is there a clear “winner”? Why or why not?
# 3. Why might you prefer **post-processing** over **in-processing** in a real deployment?
#    - Post-processing: faster, no retraining, keeps existing model; but needs group labels at prediction time.
#    - In-processing: builds fairness into the model itself; but slower, needs retraining, may not converge.
# 4. The lecture mentions that **no single fairness metric solves every problem**. How does this table illustrate that point?
# 5. Look at the **Equalized Odds Ratio** column. Values closer to 1 mean the TPR and FPR ratios between groups are closer to parity. Which method gets closest?

# %% [markdown]
# ---
# 
# ## 7. Intersectional Fairness Analysis
# 
# Real-world systems affect people along **multiple** dimensions simultaneously. Analyzing sex and race **separately** can miss disparities that only appear at intersections (e.g., “Female / Black” vs. “Male / White”).
# 
# > **Reference:** Crenshaw (1989) introduced the concept of intersectionality. In ML fairness, Buolamwini & Gebru (2018) showed that facial analysis systems had higher error rates for darker-skinned women than for any single demographic group analyzed alone.
# > [proceedings.mlr.press/v81/buolamwini18a.html](http://proceedings.mlr.press/v81/buolamwini18a.html)

# %%
# Combine the two sensitive features into one intersectional feature
intersectional_test = sex_test.astype(str) + " / " + race_test.astype(str)

mf_intersect = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "recall (TPR)": true_positive_rate,
        "FPR": false_positive_rate,
        "count": count,
    },
    y_true=y_test,
    y_pred=y_pred_baseline,
    sensitive_features=intersectional_test,
)

print("Intersectional analysis (Sex x Race):")
display(mf_intersect.by_group.sort_values("selection_rate", ascending=False))

# %% [markdown]
# ### Intersectional Heatmaps
# 
# Heatmaps make it easy to spot which **combinations** of sex and race have the highest or lowest metric values. Look for cells that are noticeably darker or lighter than their row/column neighbors — these indicate intersectional effects that single-axis analysis misses.

# %%
_ = plot_intersectional_heatmap(
    y_test, y_pred_baseline, sex_test, race_test,
    metric_fn=selection_rate, metric_name="Selection Rate",
    sf1_name="Sex", sf2_name="Race",
)

# %%
_ = plot_intersectional_heatmap(
    y_test, y_pred_baseline, sex_test, race_test,
    metric_fn=true_positive_rate, metric_name="True Positive Rate (Recall)",
    sf1_name="Sex", sf2_name="Race",
)

# %%
_ = plot_intersectional_heatmap(
    y_test, y_pred_baseline, sex_test, race_test,
    metric_fn=false_positive_rate, metric_name="False Positive Rate",
    sf1_name="Sex", sf2_name="Race",
)

# %% [markdown]
# ### Interpretation Guide
# 
# **Questions to discuss:**
# 1. Which intersectional group has the highest and lowest selection rate?
# 2. Does analyzing sex and race **separately** miss disparities that appear in the intersectional heatmaps?
# 3. Are there cells with very low **count**? What does that mean for the reliability of the metric estimates?
# 4. What challenges arise when mitigating bias for many intersectional groups? (Hint: sample size, computational cost, conflicting constraints.)

# %% [markdown]
# ---
# 
# ## 8. Lightweight Model Card ✏️ TODO
# 
# The lecture emphasizes that **documentation is part of technical quality** (see lecture notes, section 8).
# 
# A Model Card records the intended use, training data context, evaluation results, known limitations, and fairness considerations of a model. It serves as a contract between the model developer and the downstream users or auditors.
# 
# **What goes into a Model Card:**
# - **Model details**: name, version, type, training algorithm
# - **Intended use**: what the model is designed for (and what it should NOT be used for)
# - **Training data**: source, size, known biases, preprocessing
# - **Evaluation**: performance and fairness metrics, broken down by relevant subgroups
# - **Limitations**: known failure modes, out-of-distribution risks
# - **Ethical considerations**: potential harms, recommended safeguards
# 
# > **Reference:** Mitchell et al. (2019), “Model Cards for Model Reporting”: [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)

# %%
# TODO:
# Fill in the model card fields.
# Think about what matters for someone who will use or audit this model.

model_card = {
    "model_name": "...",
    "version": "...",
    "description": "...",
    "intended_use": "...",
    "training_data": {
        "name": "...",
        "size": f"{len(X_train)} training samples, {len(X_test)} test samples",
        "features_used": list(X.columns),
        "sensitive_features_analyzed": ["sex", "race"],
        "known_data_issues": [
            # List at least 2 known data issues.
        ],
    },
    "performance": {
        "accuracy": round(accuracy_score(y_test, y_pred_baseline), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred_baseline), 4),
    },
    "fairness_analysis": {
        "sex": {
            "demographic_parity_diff": round(scalar_sex_baseline["dpd"], 4),
            "equalized_odds_diff": round(scalar_sex_baseline["eod"], 4),
        },
        "race": {
            "demographic_parity_diff": round(scalar_race_baseline["dpd"], 4),
            "equalized_odds_diff": round(scalar_race_baseline["eod"], 4),
        },
    },
    "limitations": [
        # List at least 2 model limitations.
    ],
    "ethical_considerations": [
        # List at least 2 ethical considerations.
    ],
}

import json
print(json.dumps(model_card, indent=2))

# %% [markdown]
# ### Further Resources on Model Documentation
# 
# - **VerifyML** ([github.com/cylynx/verifyml](https://github.com/cylynx/verifyml)): open-source toolkit for automated model card generation with built-in fairness and explainability tests. Adapted from Google’s Model Card Toolkit. Useful for teams that want a code-first approach to model documentation with automatic test reporting.
# - **What-If Tool** ([pair-code.github.io/what-if-tool](https://pair-code.github.io/what-if-tool/get-started/)): Google’s interactive visual tool for exploring model behavior, fairness, and performance across subgroups directly in Jupyter notebooks. Useful for manual exploration and stakeholder presentations.
# - **Google Model Card Toolkit** ([github.com/tensorflow/model-card-toolkit](https://github.com/tensorflow/model-card-toolkit)): framework for generating structured model cards as JSON and HTML.

# %% [markdown]
# ---
# 
# ## 9. Debrief
# 
# **Key takeaways from this practical:**
# 
# 1. **Fairness metrics are not interchangeable.** Demographic parity, equalized odds, and equal opportunity answer different questions. Satisfying one does **not** guarantee the others (Chouldechova, 2017; Kleinberg et al., 2016). Choose based on the specific harm you want to prevent.
# 
# 2. **The four-fifths rule** (DPR ≥ 0.8) is a widely used legal threshold, but it is a minimum bar, not a guarantee of fairness.
# 
# 3. **Post-processing vs in-processing** are complementary mitigation strategies, each with trade-offs:
#    - `ThresholdOptimizer`: fast, model-agnostic, works on existing models, but may reduce accuracy and requires group labels at prediction time.
#    - `ExponentiatedGradient`: produces a new fairness-aware model, but costs more training time and may not always converge.
# 
# 4. **Intersectional analysis** often reveals disparities that single-feature analysis misses. Always check combinations of sensitive features.
# 
# 5. **Documentation (Model Cards)** is part of responsible AI practice, not an afterthought. It creates accountability and helps downstream users understand what a model can and cannot do.
# 
# 6. **Fairness is context-dependent.** There is no universal threshold or metric. The right choice depends on the domain, the stakeholders, and the specific harms at stake.
# 
# 
# **What the lecture covers but this practical does not (time constraints):**
# - Pre-processing mitigation: data rebalancing (oversampling / undersampling) — lecture section 17.
# - Interpretability: SHAP, LIME, saliency maps — covered in depth in Lecture 11.
# - Privacy: differential privacy, federated learning — lecture section 18.
# - Adversarial robustness: evasion and poisoning attacks — lecture section 19.
# - Causal ML and conformal prediction — lecture sections 20 and 22.
# 
# These topics are important for a complete Responsible AI workflow. Students are encouraged to revisit the lecture notes and slides for details.
# ---
# 
# **Recommended further reading:**
# - Fairlearn User Guide: [fairlearn.org/main/user_guide](https://fairlearn.org/main/user_guide/index.html)
# - Fairlearn common fairness metrics: [fairlearn.org/main/user_guide/assessment/common_fairness_metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html)
# - Barocas, Hardt, Narayanan, *Fairness and Machine Learning*: [fairmlbook.org](http://www.fairmlbook.org/)
# - Verma & Rubin (2018), “Fairness definitions explained”: [DOI:10.1145/3194770.3194776](https://doi.org/10.1145/3194770.3194776)
# - Mitchell et al. (2019), “Model Cards for Model Reporting”: [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)
# - Hardt, Price, Srebro (2016), “Equality of Opportunity in Supervised Learning”: [arXiv:1610.02413](https://arxiv.org/abs/1610.02413)
# - Agarwal et al. (2018), “A Reductions Approach to Fair Classification”: [arXiv:1803.02453](https://arxiv.org/abs/1803.02453)
