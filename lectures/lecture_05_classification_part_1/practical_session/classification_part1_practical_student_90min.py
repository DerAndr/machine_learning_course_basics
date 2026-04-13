# %% [markdown]
# # Classification Part 1 Practical Session
#
# **Dataset:** Adult Census Income ([OpenML `1590`](https://www.openml.org/search?type=data&status=active&id=1590))
#
# **Learning Goals**
# - compare `KNN`, `DecisionTreeClassifier`, and `LogisticRegression` on one realistic binary task
# - build leakage-safe sklearn `Pipeline`s for mixed tabular data
# - evaluate classification beyond accuracy using confusion matrices, ROC, PR, and calibration
# - tune and interpret probability thresholds instead of accepting `0.5` blindly
# - connect model behavior back to the lecture ideas: geometry, trees, and probabilistic linear models

# %% [markdown]
# ## 1. Setup
#
# Run this notebook with the baseline repository environment:
#
# - `uv sync`
#
# Optional local environment for the Optuna section:
#
# - `uv sync --group hpo_automl`
#
# In Google Colab, install:
#
# - `openml`
# - `optuna` (optional; only for the Optuna section)
#
# This notebook keeps Optuna **optional**. If it is unavailable, the rest of the practical still runs.

# %%
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import gaussian_kde
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    }
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
nprng = np.random.default_rng(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
print("Imports complete")

# %% [markdown]
# ## 2. Load the Adult Census Dataset
#
# We use the **Adult Census Income** dataset from OpenML.
#
# Why it works well here:
# - binary target with real class imbalance
# - mixed numeric and categorical features
# - enough rows to make learning curves and KNN runtime trade-offs visible
# - interpretable business framing: predicting whether annual income is above `50K`

# %%
raw = fetch_openml(data_id=1590, as_frame=True, parser="auto")

df = raw.frame.copy()
df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]

df["target"] = (df["class"].astype(str).str.strip().str.startswith(">")).astype(int)
df.drop(columns=["class"], inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
for col in df.select_dtypes(include="category").columns:
    df[col] = df[col].cat.rename_categories({v: str(v).strip() for v in df[col].cat.categories})

target_col = "target"
class_balance = df[target_col].value_counts(normalize=True).sort_index()

print(f"Shape: {df.shape}")
print("\nClass balance:")
print(class_balance.rename("proportion").round(3))
print(f"\nNaive majority-class accuracy: {class_balance.max():.3f}")
display(df.head(5))

# %% [markdown]
# ## 3. Quick Data Exploration
#
# Before modeling, inspect:
# - numeric feature distributions by class
# - categorical groups with especially high or low `>50K` rate
# - the scale of the imbalance, because it affects metric choice and thresholding

# %%
# Identify numeric vs categorical columns (excluding target)
feature_cols = [c for c in df.columns if c != target_col]

numeric_cols     = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df[feature_cols].select_dtypes(
                       exclude=["number"]).columns.tolist()

print(f"Features    : {len(feature_cols)}  "
      f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
print(f"\nNumeric     : {numeric_cols}")
print(f"Categorical : {categorical_cols}")
print(f"\nMissing values:")
missing = df[feature_cols].isna().sum()
print(missing[missing > 0] if missing.any() else "  None")
df.describe(include="all").T

# %%
# ── EDA visualisations ──────────────────────────────────────
fig = plt.figure(figsize=(18, 10))

# Row 1: numeric feature distributions by class (4 panels)
num_show = numeric_cols[:4]
for idx, feat in enumerate(num_show):
    ax = fig.add_subplot(3, 4, idx + 1)
    for cls, color, label in [(0, "#2196F3", "≤50K"), (1, "#F44336", ">50K")]:
        vals = df.loc[df[target_col] == cls, feat].dropna()
        ax.hist(vals, bins=30, alpha=0.55, color=color, density=True, label=label)
    ax.set_title(feat, fontweight="bold")
    ax.set_yticks([])
    if idx == 0:
        ax.legend(fontsize=9)

# Row 2: remaining numeric features
num_show2 = numeric_cols[4:]
for idx, feat in enumerate(num_show2):
    ax = fig.add_subplot(3, 4, idx + 5)
    for cls, color in [(0, "#2196F3"), (1, "#F44336")]:
        vals = df.loc[df[target_col] == cls, feat].dropna()
        ax.hist(vals, bins=30, alpha=0.55, color=color, density=True)
    ax.set_title(feat, fontweight="bold")
    ax.set_yticks([])

# Row 3: top categorical features (rate of >50K per category)
cat_show = categorical_cols[:4]
for idx, feat in enumerate(cat_show):
    ax = fig.add_subplot(3, 4, idx + 9)
    rate = (df.groupby(df[feat].astype(str))[target_col]
              .mean()
              .sort_values(ascending=False)
              .head(10))
    ax.barh(range(len(rate)), rate.values, color="#7986CB", alpha=0.85)
    ax.set_yticks(range(len(rate)))
    ax.set_yticklabels(rate.index, fontsize=8)
    ax.set_xlim(0, 1)
    ax.axvline(df[target_col].mean(), color="red", linestyle="--",
               linewidth=1, label="overall avg")
    ax.set_title(f"{feat}\n(rate >50K)", fontweight="bold", fontsize=10)
    ax.invert_yaxis()
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle("Adult Census Income — EDA Overview", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Preprocessing and Split
#
# We make the train/test split **before** fitting any preprocessing step.
#
# Preprocessing choice in this practical:
# - numeric features: median imputation + standard scaling
# - categorical features: most-frequent imputation + one-hot encoding
#
# Why `OneHotEncoder` here?
# - it avoids inventing an arbitrary order between categories
# - that is especially important for `LogisticRegression`
# - it is also safer for distance-based models than raw ordinal codes
#
# This is a deliberate classroom simplification: one shared preprocessing recipe across all three baseline models.

# %%
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col].copy()

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape}")
print(f"Test : {X_test.shape}")
print("\nClass balance (train):")
print(y_train.value_counts(normalize=True).sort_index().round(3))

numeric_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocess = ColumnTransformer(
    [
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop",
)

print("\nPreprocessing pipeline defined")

# %% [markdown]
# ## 5. Shared Evaluation Utilities
#
# These helpers keep the later sections compact and make the metric logic explicit.

# %%
def proba_1(pipe, X):
    return np.asarray(pipe.predict_proba(X)[:, 1])


def evaluate_classifier(y_true, y_score, threshold=0.5):
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }, y_pred


def summarize_metrics(metrics_dict):
    rows = [{"model": name, **metrics} for name, metrics in metrics_dict.items()]
    return pd.DataFrame(rows).set_index("model").sort_values("f1", ascending=False).round(4)


def find_best_threshold(y_true, y_score, metric="f1", num=201):
    y_score = np.asarray(y_score)
    thresholds = np.linspace(0, 1, num)
    best_t, best_v, hist = 0.5, -1, []
    for t in thresholds:
        metrics, _ = evaluate_classifier(y_true, y_score, threshold=t)
        value = metrics[metric]
        hist.append((t, value))
        if value > best_v:
            best_t, best_v = t, value
    return best_t, best_v, pd.DataFrame(hist, columns=["threshold", metric])


def _compute_kde(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    xs = np.linspace(0, 1, 500)

    def safe_kde(values):
        if len(values) < 2 or np.allclose(np.std(values), 0):
            return np.zeros_like(xs)
        return gaussian_kde(values, bw_method="scott")(xs)

    return xs, neg, pos, safe_kde(neg), safe_kde(pos)


def plot_confusion_matrix(cm, class_names=("≤50K", ">50K")):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontweight="bold")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for (i, j), value in np.ndenumerate(cm):
        color = "white" if value > cm.max() / 2 else "black"
        ax.text(j, i, f"{labels[i][j]}\n{value:,}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_probability_distribution(
    y_true,
    y_score,
    threshold=0.5,
    label0="≤50K (0)",
    label1=">50K (1)",
    title="Predicted Probability Distribution",
    ax=None,
):
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9, 5))

    xs, neg, pos, d_neg, d_pos = _compute_kde(y_true, y_score)
    ax.fill_between(xs, d_neg, alpha=0.30, color="#2196F3")
    ax.fill_between(xs, d_pos, alpha=0.30, color="#F44336")
    ax.plot(xs, d_neg, color="#1565C0", linewidth=2.5, label=label0)
    ax.plot(xs, d_pos, color="#B71C1C", linewidth=2.5, label=label1)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("P(income >50K)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    if standalone:
        plt.tight_layout()
        plt.show()


def baseline_diagnostic(name, clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    y_score = clf.predict_proba(X_te)[:, 1]
    metrics, y_pred = evaluate_classifier(y_te, y_score)
    cm = confusion_matrix(y_te, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        f"{name} | F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  Acc={metrics['accuracy']:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["≤50K", ">50K"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["≤50K", ">50K"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for (i, j), value in np.ndenumerate(cm):
        color = "white" if value > cm.max() / 2 else "black"
        ax.text(j, i, f"{labels[i][j]}\n{value:,}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    RocCurveDisplay.from_predictions(y_te, y_score, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve")
    ax.legend(fontsize=9)

    ax = axes[2]
    y_arr = np.asarray(y_te)
    neg = y_score[y_arr == 0]
    pos = y_score[y_arr == 1]
    bins = np.linspace(0, 1, 35)
    ax.hist(neg, bins=bins, alpha=0.55, color="#2196F3", density=True, label="≤50K (class 0)")
    ax.hist(pos, bins=bins, alpha=0.55, color="#F44336", density=True, label=">50K (class 1)")
    xs = np.linspace(0, 1, 400)
    if len(neg) > 1 and not np.allclose(np.std(neg), 0):
        ax.plot(xs, gaussian_kde(neg)(xs), "#1565C0", linewidth=2)
    if len(pos) > 1 and not np.allclose(np.std(pos), 0):
        ax.plot(xs, gaussian_kde(pos)(xs), "#B71C1C", linewidth=2)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="threshold = 0.5")
    ax.set_xlabel("P(income >50K)")
    ax.set_ylabel("Density")
    ax.set_title("Score Probability Histogram")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    return clf, metrics


print("Utility functions loaded")

# %% [markdown]
# ## How To Work In Teams
#
# This notebook can run in pairs or small groups.
#
# Suggested split:
# - **Group A**: focus on `KNN` and `DecisionTree` baseline pipelines, validation curves, and model comparison
# - **Group B**: focus on `LogisticRegression`, threshold tuning, and interpretation
#
# Shared checkpoints:
# 1. agree on why accuracy is not enough for this dataset
# 2. compare the best-performing model at threshold `0.5`
# 3. discuss whether the threshold should move for a loan / tax / screening use case

# %% [markdown]
# ## 6. Quick Baselines Without Pipelines
#
# This section is intentionally procedural.
#
# Why keep it?
# - students can see the transformed feature matrix explicitly
# - the classifier APIs are easier to read before everything is hidden inside sklearn `Pipeline`s
# - it makes later `Pipeline` sections feel like a refactor, not magic

# %%
preprocess.fit(X_train, y_train)
X_tr = preprocess.transform(X_train)
X_te = preprocess.transform(X_test)
feature_names = preprocess.get_feature_names_out().tolist()

print(f"Transformed shapes → train: {X_tr.shape} | test: {X_te.shape}")
print(f"Expanded feature count after one-hot encoding: {len(feature_names)}")

# %%
# ── Step 2: helper that draws the 3-panel diagnostic for one model ──────────

def baseline_diagnostic(name, clf, X_tr, y_tr, X_te, y_te):
    """Fit clf, then draw Confusion Matrix | ROC Curve | Probability Histogram."""
    clf.fit(X_tr, y_tr)
    y_score = clf.predict_proba(X_te)[:, 1]
    m, y_pred = evaluate_classifier(y_te, y_score)
    cm = confusion_matrix(y_te, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{name}   |   F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}  "
                 f"Acc={m['accuracy']:.3f}",
                 fontsize=13, fontweight="bold")

    # ── panel 1: confusion matrix ────────────────────────────
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["≤50K", ">50K"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["≤50K", ">50K"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > cm.max() / 2 else "black"
        ax.text(j, i, f"{labels[i][j]}\n{v:,}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ── panel 2: ROC curve ───────────────────────────────────
    ax = axes[1]
    RocCurveDisplay.from_predictions(y_te, y_score, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve")
    ax.legend(fontsize=9)

    # ── panel 3: probability score histogram ──────────────────
    ax = axes[2]
    y_arr = np.asarray(y_te)
    neg = y_score[y_arr == 0]
    pos = y_score[y_arr == 1]

    bins = np.linspace(0, 1, 35)
    ax.hist(neg, bins=bins, alpha=0.55, color="#2196F3",
            density=True, label="≤50K (class 0)")
    ax.hist(pos, bins=bins, alpha=0.55, color="#F44336",
            density=True, label=">50K (class 1)")

    # Overlay KDE curves
    xs = np.linspace(0, 1, 400)
    if len(neg) > 1:
        ax.plot(xs, gaussian_kde(neg)(xs), "#1565C0", linewidth=2)
    if len(pos) > 1:
        ax.plot(xs, gaussian_kde(pos)(xs), "#B71C1C", linewidth=2)

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5,
               alpha=0.7, label="threshold = 0.5")
    ax.set_xlabel("P(income >50K)")
    ax.set_ylabel("Density")
    ax.set_title("Score Probability Histogram")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return clf, m

print("baseline_diagnostic() defined")

# %% [markdown]
# ### 6a) K-Nearest Neighbours
#
# **Key hyperparameters:**
#
# | Parameter | Typical values | Effect |
# |-----------|---------------|--------|
# | `n_neighbors` | 3, 5, 10, 20 | Smaller → more complex boundary; larger → smoother |
# | `weights` | `'uniform'`, `'distance'` | Distance weights give closer neighbours more influence |
# | `metric` | `'euclidean'`, `'manhattan'` | Distance function used |
# | `algorithm` | `'ball_tree'`, `'kd_tree'`, `'brute'` | Index structure; affects speed, not accuracy |

# %%
knn = KNeighborsClassifier(
    n_neighbors=10,        # number of nearest neighbours
    weights="distance",    # closer neighbours vote more
    metric="euclidean",    # standard L2 distance
    algorithm="auto",      # sklearn picks the fastest structure
    n_jobs=-1,             # parallelise distance computation
)

knn_fitted, knn_metrics = baseline_diagnostic(
    "KNN  (n_neighbors=10, weights='distance')",
    knn, X_tr, y_train, X_te, y_test,
)

# %% [markdown]
# ### 6b) Decision Tree
#
# **Key hyperparameters:**
#
# | Parameter | Typical values | Effect |
# |-----------|---------------|--------|
# | `max_depth` | 3–15, `None` | Max tree depth; `None` = unlimited (overfits) |
# | `min_samples_split` | 2–50 | Min samples to split a node |
# | `min_samples_leaf` | 1–50 | Min samples in a leaf; higher → smoother boundary |
# | `max_features` | `'sqrt'`, `'log2'`, float | Features considered per split |
# | `criterion` | `'gini'`, `'entropy'`, `'log_loss'` | Split quality measure |

# %%
dt = DecisionTreeClassifier(
    max_depth=8,             # limit tree depth to avoid severe overfitting
    min_samples_split=20,    # need at least 20 samples to attempt a split
    min_samples_leaf=10,     # each leaf must cover at least 10 samples
    criterion="gini",        # Gini impurity (or 'entropy' / 'log_loss')
    max_features=None,       # consider all features at each split
    random_state=RANDOM_STATE,
)

dt_fitted, dt_metrics = baseline_diagnostic(
    "Decision Tree  (max_depth=8, min_samples_leaf=10)",
    dt, X_tr, y_train, X_te, y_test,
)

# %% [markdown]
# ### 6c) Logistic Regression
#
# **Key hyperparameters:**
#
# | Parameter | Typical values | Effect |
# |-----------|---------------|--------|
# | `C` | 1e-3 … 1e3 | Inverse regularisation strength; smaller C → more regularisation |
# | `penalty` | `'l1'`, `'l2'`, `'elasticnet'`, `None` | Regularisation type |
# | `solver` | `'lbfgs'`, `'saga'`, `'liblinear'` | Optimiser; must match `penalty` |
# | `max_iter` | 100–5000 | Max optimiser iterations; increase if convergence warning |
# | `class_weight` | `None`, `'balanced'` | Upweights minority class if `'balanced'` |

# %%
lr = LogisticRegression(
    C=1.0,                  # default; try 0.01 (more reg.) or 100 (less reg.)
    penalty="l2",           # ridge regularisation (most common default)
    solver="lbfgs",         # good for l2; use 'saga' for l1 / elasticnet
    max_iter=1000,          # increase if you see ConvergenceWarning
    class_weight=None,      # try 'balanced' on imbalanced datasets
    random_state=RANDOM_STATE,
)

lr_fitted, lr_metrics = baseline_diagnostic(
    "Logistic Regression  (C=1.0, penalty='l2')",
    lr, X_tr, y_train, X_te, y_test,
)

# %%
# ── Summary table for this section ────────────────────────────
raw_summary = summarize_metrics({
    "KNN (no pipeline)": knn_metrics,
    "DecisionTree (no pipeline)": dt_metrics,
    "LogisticRegression (no pipeline)": lr_metrics,
})
print("Quick baseline comparison (no pipelines, manual preprocessing):\n")
display(raw_summary)

print("\nKey takeaways:")
print("  • All three models beat the naive 76%-accuracy baseline")
print("  • Accuracy is still inflated — focus on F1 and ROC-AUC")
print("  • Probability histograms show how well classes separate")
print("  • The next sections wrap this into Pipelines for cleaner reuse")

# %% [markdown]
# ## 7. Build Baseline Pipeline Models ✏️ TODO
#
# **Your task:** create three leakage-safe classification pipelines and compare them at threshold `0.5`.
#
# Requirements:
# 1. Create a `pipelines` dictionary with:
#    - `KNN`: start with `n_neighbors=5`
#    - `DecisionTree`: start with `max_depth=5`
#    - `LogisticRegression`: start with `max_iter=1000`
# 2. Each pipeline should contain the shared `preprocess` step plus the classifier.
# 3. Fit every pipeline on `X_train, y_train`.
# 4. Store fitted models in `fitted_baseline_pipes`.
# 5. Compute `scores_at_05`, `metrics_at_05`, and `summary_05` on the test split.
#
# Questions:
# - Which model performs best by `F1`?
# - Why is accuracy still a weak metric here?
# - Which model gives the cleanest probability separation?

# %%
# TODO:
# 1. Build the three baseline pipelines.
# 2. Fit each one on the training split.
# 3. Store the fitted models in fitted_baseline_pipes.
# 4. Compute scores_at_05, metrics_at_05, and summary_05 on the test split.

pipelines = {}
metrics_at_05, scores_at_05, fitted_baseline_pipes = {}, {}, {}

print("TODO: build the baseline pipelines, fit them, and create summary_05.")

# %% [markdown]
# ## 8. Hyperparameter Exploration
#
# ### Part A — Validation Curves
#
# Use validation curves to connect each model to the lecture themes:
# - `KNN`: bias-variance via `n_neighbors`
# - `DecisionTree`: depth control and overfitting
# - `LogisticRegression`: regularisation strength via `C`

# %%
# Pre-filled validation curves
print("Computing validation curves (this may take a minute)...\n")

# KNN: limited range to avoid very long compute
param_range_knn = np.array([1, 3, 5, 7, 10, 15, 20])
tr_knn, te_knn = validation_curve(
    Pipeline([("prep", preprocess), ("clf", KNeighborsClassifier())]),
    X_train, y_train,
    param_name="clf__n_neighbors", param_range=param_range_knn,
    cv=3, scoring="f1", n_jobs=-1)          # cv=3 for speed with large data
print(f"KNN  — best n_neighbors : {param_range_knn[te_knn.mean(1).argmax()]}")

param_range_dt = np.arange(1, 16)
tr_dt, te_dt = validation_curve(
    Pipeline([("prep", preprocess),
              ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
    X_train, y_train,
    param_name="clf__max_depth", param_range=param_range_dt,
    cv=5, scoring="f1", n_jobs=-1)
print(f"DT   — best max_depth   : {param_range_dt[te_dt.mean(1).argmax()]}")

param_range_lr = np.logspace(-3, 3, 7)
tr_lr, te_lr = validation_curve(
    Pipeline([("prep", preprocess),
              ("clf", LogisticRegression(max_iter=1000,
                                        random_state=RANDOM_STATE))]),
    X_train, y_train,
    param_name="clf__C", param_range=param_range_lr,
    cv=5, scoring="f1", n_jobs=-1)
print(f"LR   — best C           : {param_range_lr[te_lr.mean(1).argmax()]:.4f}")

# %%
# Plot validation curves — all three models side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

configs = [
    (tr_knn, te_knn, param_range_knn, "n_neighbors",        False, "KNN"),
    (tr_dt,  te_dt,  param_range_dt,  "max_depth",          False, "Decision Tree"),
    (tr_lr,  te_lr,  param_range_lr,  "C (regularization)", True,  "Logistic Regression"),
]

for ax, (train_s, test_s, xs, xlabel, use_log, title) in zip(axes, configs):
    if use_log:
        ax.set_xscale("log")
    ax.plot(xs, test_s.mean(1),  "o-",  label="Validation")
    ax.fill_between(xs, test_s.mean(1)  - test_s.std(1),
                        test_s.mean(1)  + test_s.std(1), alpha=0.15)
    ax.plot(xs, train_s.mean(1), "o--", label="Training")
    ax.fill_between(xs, train_s.mean(1) - train_s.std(1),
                        train_s.mean(1) + train_s.std(1), alpha=0.15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title(f"{title}\nValidation Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

plt.suptitle("Validation Curves — F1 Score vs Hyperparameter",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Part B — Learning Curves
#
# This block is pre-filled so you can focus on interpretation.
#
# Questions:
# - Which model looks most data-limited?
# - Which model has the biggest train/validation gap?
# - Does `KNN` look like a good scaling choice for much larger tables?

# %%
if not pipelines:
    print("Complete Section 7 first, then compute learning curves.")
else:
    print("Computing learning curves...\n")

    MAX_TRAIN = 10_000
    lc_results = {}

    for name, pipe in pipelines.items():
        max_frac = MAX_TRAIN / len(X_train) if name == "KNN" else 1.0
        upper = min(max_frac, 1.0)
        sizes = np.linspace(0.05, upper, 8)
        sz, tr_s, te_s = learning_curve(
            pipe,
            X_train,
            y_train,
            cv=3,
            scoring="f1",
            train_sizes=sizes,
            n_jobs=-1,
        )
        lc_results[name] = (sz, tr_s, te_s)
        print(f"{name}: final val F1 = {te_s.mean(1)[-1]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, (sz, tr_s, te_s)) in zip(axes, lc_results.items()):
        ax.plot(sz, tr_s.mean(1), "o--", label="Training")
        ax.fill_between(sz, tr_s.mean(1) - tr_s.std(1), tr_s.mean(1) + tr_s.std(1), alpha=0.15)
        ax.plot(sz, te_s.mean(1), "o-", label="Validation")
        ax.fill_between(sz, te_s.mean(1) - te_s.std(1), te_s.mean(1) + te_s.std(1), alpha=0.15)
        ax.set_title(f"{name}\nLearning Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("F1 Score")
        ax.legend(fontsize=9)

    plt.suptitle("Learning Curves — Performance vs Training Data Size", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. Optional: Hyperparameter Tuning with Optuna
#
# This is an **optional** advanced section.
#
# Use it if:
# - `optuna` is available in your environment
# - you want to compare smarter search to the manual validation-curve step
#
# If you skip it, the notebook will continue with the baseline pipeline models from Section 7.

# %%
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    print(f"Optuna {optuna.__version__}")
except ModuleNotFoundError:
    optuna = None
    OPTUNA_AVAILABLE = False
    print("Optuna is not installed. Skip this optional section or install it with `uv sync --group hpo_automl`.")

# %%
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_knn = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def make_objective(model_name):
    def objective(trial):
        if model_name == "KNN":
            clf = Pipeline([
                ("prep", preprocess),
                ("clf", KNeighborsClassifier(
                    n_neighbors=trial.suggest_int("n_neighbors", 3, 30),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                    metric=trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
                    n_jobs=-1,
                )),
            ])
            cv_use = cv_knn
        elif model_name == "DecisionTree":
            clf = Pipeline([
                ("prep", preprocess),
                ("clf", DecisionTreeClassifier(
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 50),
                    criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                    random_state=RANDOM_STATE,
                )),
            ])
            cv_use = cv_inner
        else:
            clf = Pipeline([
                ("prep", preprocess),
                ("clf", LogisticRegression(
                    C=trial.suggest_float("C", 1e-4, 1e3, log=True),
                    solver=trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                )),
            ])
            cv_use = cv_inner

        scores = cross_val_score(clf, X_train, y_train, cv=cv_use, scoring="f1", n_jobs=-1)
        return scores.mean()

    return objective


print("Objective functions ready")

# %%
best_pipes = {}
studies = {}
summary_optuna = None

if not OPTUNA_AVAILABLE:
    print("Skip: Optuna is unavailable in this environment.")
else:
    N_TRIALS = {"KNN": 20, "DecisionTree": 40, "LogisticRegression": 40}
    for name, n_trials in N_TRIALS.items():
        study = optuna.create_study(direction="maximize", study_name=name)
        study.optimize(make_objective(name), n_trials=n_trials, show_progress_bar=False)
        studies[name] = study
        print(f"{name}: best CV F1 = {study.best_value:.4f}")

    rows = []
    for name, study in studies.items():
        params = study.best_params
        if name == "KNN":
            pipe = Pipeline([
                ("prep", preprocess),
                ("clf", KNeighborsClassifier(
                    n_neighbors=params["n_neighbors"],
                    weights=params["weights"],
                    metric=params["metric"],
                    n_jobs=-1,
                )),
            ])
        elif name == "DecisionTree":
            pipe = Pipeline([
                ("prep", preprocess),
                ("clf", DecisionTreeClassifier(
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    min_samples_split=params["min_samples_split"],
                    criterion=params["criterion"],
                    random_state=RANDOM_STATE,
                )),
            ])
        else:
            pipe = Pipeline([
                ("prep", preprocess),
                ("clf", LogisticRegression(
                    C=params["C"],
                    solver=params["solver"],
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                )),
            ])
        pipe.fit(X_train, y_train)
        best_pipes[name] = pipe
        row = {"model": name, "best_f1": round(study.best_value, 4), "n_trials": len(study.trials)}
        row.update({f"param_{k}": v for k, v in params.items()})
        rows.append(row)

    summary_optuna = pd.DataFrame(rows).set_index("model")
    print("\nBest hyperparameters found by Optuna:\n")
    display(summary_optuna)

# %%
if not studies:
    print("No Optuna studies to visualize.")
else:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, study) in zip(axes, studies.items()):
        trials_df = study.trials_dataframe()
        values = trials_df["value"].tolist()
        best_so_far = [max(values[: i + 1]) for i in range(len(values))]
        ax.scatter(range(len(values)), values, alpha=0.45, s=25, color="#7986CB", label="Trial F1")
        ax.plot(range(len(best_so_far)), best_so_far, color="#F44336", linewidth=2.5, label="Best so far")
        ax.set_title(f"{name}\nBest F1 = {study.best_value:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trial #")
        ax.set_ylabel("CV F1 Score")
        ax.legend(fontsize=9)
    plt.suptitle("Optuna Optimisation History — F1 Score per Trial", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 10. Model Selection and Diagnostics ✏️ TODO
#
# Use the currently available candidate models:
# - if you ran Optuna, compare the tuned models
# - otherwise compare the baseline pipelines from Section 7
#
# Your task:
# 1. build `candidate_pipes`
# 2. evaluate them at threshold `0.5`
# 3. create `summary_selected`
# 4. pick `best_name`
#
# Then use the next diagnostic cells to inspect that best model.

# %%
# TODO:
# 1. If best_pipes is non-empty, use it; otherwise fall back to fitted_baseline_pipes.
# 2. Compute scores_selected and metrics_selected.
# 3. Create summary_selected and identify best_name.

candidate_pipes = best_pipes if best_pipes else fitted_baseline_pipes
candidate_label = "Optuna-tuned" if best_pipes else "baseline"

if not candidate_pipes:
    print("Complete Section 7 first, or run the optional Optuna section.")
else:
    print("TODO: evaluate candidate_pipes, create summary_selected, and set best_name.")

# %% [markdown]
# ### Diagnostic Plots
#
# These cells are pre-filled, but they depend on `best_name`, `scores_selected`, and `summary_selected` from the previous TODO block.

# %%
if "best_name" not in globals() or "scores_selected" not in globals():
    print("Complete the model-selection TODO first, then run diagnostics here.")
else:
    y_score_best = scores_selected[best_name]
    m_best, y_pred_best = evaluate_classifier(y_test, y_score_best, threshold=0.5)
    cm = confusion_matrix(y_test, y_pred_best)

    print(f"Best model: {best_name}")
    print("-" * 40)
    for key, value in m_best.items():
        print(f"  {key:12s}: {value:.4f}")
    print(f"\nConfusion Matrix:\n  TN={cm[0,0]:,}  FP={cm[0,1]:,}\n  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# %%
if "cm" not in globals() or "y_score_best" not in globals():
    print("Run the previous diagnostic cell first.")
else:
    plot_confusion_matrix(cm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, y_score_best, ax=ax1, name=best_name)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax1.legend(); ax1.set_title("ROC Curve", fontweight="bold")
    PrecisionRecallDisplay.from_predictions(y_test, y_score_best, ax=ax2, name=best_name)
    baseline_pr = y_test.mean()
    ax2.axhline(baseline_pr, color="gray", linestyle="--", alpha=0.6, label=f"No-skill baseline ({baseline_pr:.2f})")
    ax2.legend(fontsize=9)
    ax2.set_title("Precision-Recall Curve", fontweight="bold")
    plt.suptitle(f"Diagnostic Curves — {best_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()

    plot_probability_distribution(
        y_test, y_score_best, threshold=0.5,
        label0="≤50K income (0)", label1=">50K income (1)",
        title=f"Predicted Probability Distribution — {best_name}"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y_test, y_score_best, n_bins=10, ax=ax, name=best_name)
    ax.set_title("Calibration Curve\n(perfect calibration = diagonal)", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 11. Threshold Optimisation ✏️ TODO
#
# **Your task:** find a better classification threshold for the current best model.
#
# Requirements:
# 1. use `find_best_threshold(..., metric="f1")`
# 2. compare the new threshold against `0.5`
# 3. re-evaluate **all** candidate models at the new threshold
#
# Real-world question:
# - if false positives and false negatives have different business costs, would you still optimise only `F1`?

# %%
# TODO:
# 1. Use find_best_threshold on y_test and y_score_best.
# 2. Store best_t, best_v, hist, and m_opt.
# 3. Print the threshold comparison.

if "y_score_best" not in globals():
    print("Run the diagnostics section first, so y_score_best is available.")
else:
    print("TODO: compute the threshold sweep and identify best_t.")

# %%
if not all(name in globals() for name in ["best_t", "hist", "m_best", "best_name", "y_score_best"]):
    print("Complete the threshold-optimisation TODO first.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(hist["threshold"], hist["f1"], linewidth=2.5, color="#3F51B5", label="F1 Score")
    ax.axvline(best_t, color="#F44336", linestyle="--", linewidth=2, label=f"Optimal: {best_t:.3f}")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Default: 0.5", alpha=0.7)
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Threshold Sweep — {best_name}", fontsize=13, fontweight="bold")
    ax.legend()

    plot_probability_distribution(
        y_test,
        y_score_best,
        threshold=best_t,
        label0="≤50K (0)",
        label1=">50K (1)",
        title=f"Distributions at Optimal Threshold ({best_t:.2f})",
        ax=axes[1],
    )

    plt.suptitle("Threshold Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()

# %%
print(f"\nAll candidate models at threshold = {best_t:.4f}:\n")
metrics_opt_all = {}
for name, scores in scores_selected.items():
    metrics_opt_all[name], _ = evaluate_classifier(y_test, scores, threshold=best_t)
    print(f"  {name:22s} F1={metrics_opt_all[name]['f1']:.4f} Prec={metrics_opt_all[name]['precision']:.4f} Rec={metrics_opt_all[name]['recall']:.4f}")

summary_opt = summarize_metrics(metrics_opt_all)
print()
display(summary_opt)

# %% [markdown]
# ## 12. Model Interpretation ✏️ TODO
#
# Use this section to connect performance back to the lecture models.
#
# Suggested tasks:
# - inspect the strongest `LogisticRegression` coefficients
# - inspect `DecisionTree` permutation importance
# - compare the two importance stories qualitatively
#
# Questions:
# - which features look economically plausible?
# - where do linear and tree-based importance stories diverge?
# - which differences come from using different feature spaces?

# %%
# TODO:
# 1. Pull feature names from the fitted preprocessor.
# 2. Extract logistic-regression coefficients.
# 3. Build coef_df and visualize the strongest effects.

if "candidate_pipes" not in globals() or not candidate_pipes:
    print("Complete model selection first.")
else:
    all_feature_names = candidate_pipes[best_name].named_steps["prep"].get_feature_names_out().tolist()
    lr_pipe = candidate_pipes.get("LogisticRegression")
    coef_df = None

    if lr_pipe:
        lr_feature_names = lr_pipe.named_steps["prep"].get_feature_names_out().tolist()
        coefs = lr_pipe.named_steps["clf"].coef_.ravel()
        coef_df = (
            pd.DataFrame({"feature": lr_feature_names, "coefficient": coefs, "abs_coef": np.abs(coefs)})
            .sort_values("abs_coef", ascending=False)
            .reset_index(drop=True)
        )
        display(coef_df.head(15))
        print("TODO: turn coef_df into a signed coefficient plot.")
    else:
        print("LogisticRegression is not available in candidate_pipes.")

# %%
# TODO:
# 1. Compute permutation importance for the fitted DecisionTree pipeline.
# 2. Use the original input column names, because permutation_importance runs on the whole pipeline.
# 3. Build pi_df and visualize the strongest features.
# 4. If coef_df exists, compare the results qualitatively, keeping in mind that the logistic model uses one-hot-expanded features while the tree importance is reported on original input columns.

if "candidate_pipes" not in globals() or not candidate_pipes:
    print("Complete model selection first.")
else:
    dt_pipe = candidate_pipes.get("DecisionTree")
    if dt_pipe:
        idx_sample = nprng.choice(len(X_test), size=min(3000, len(X_test)), replace=False)
        X_test_sub = X_test.iloc[idx_sample]
        y_test_sub = y_test.iloc[idx_sample]
        print("TODO: compute permutation importance on the subsampled test set using the original input columns as feature names.")
    else:
        print("DecisionTree is not available in candidate_pipes.")

# %%
dt_pipe = candidate_pipes.get("DecisionTree")
if dt_pipe:
    dt_clf = dt_pipe.named_steps["clf"]
    dt_feature_names = dt_pipe.named_steps["prep"].get_feature_names_out().tolist()
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(
        dt_clf,
        feature_names=dt_feature_names,
        class_names=["≤50K", ">50K"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree — first 3 levels", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
else:
    print("DecisionTree is not available in candidate_pipes.")

# %% [markdown]
# ## 13. All-Model Probability Distribution Comparison
#
# This final visual wraps the lecture together:
# - probability estimation
# - class separation
# - threshold friendliness
# - differences between model families, not just one best score

# %%
if "scores_selected" not in globals() or not scores_selected:
    print("Complete model selection first, then compare the probability distributions here.")
else:
    fig, axes = plt.subplots(1, len(scores_selected), figsize=(5.5 * len(scores_selected), 5))
    if len(scores_selected) == 1:
        axes = [axes]
    y_true_arr = np.asarray(y_test)

    for ax, (name, scores) in zip(axes, scores_selected.items()):
        xs, neg, pos, d_neg, d_pos = _compute_kde(y_true_arr, scores)
        ax.fill_between(xs, d_neg, alpha=0.40, color="#2196F3", label="≤50K income")
        ax.fill_between(xs, d_pos, alpha=0.40, color="#F44336", label=">50K income")
        ax.plot(xs, d_neg, "#1565C0", linewidth=2.5)
        ax.plot(xs, d_pos, "#B71C1C", linewidth=2.5)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
        roc = metrics_selected[name]["roc_auc"]
        f1 = metrics_selected[name]["f1"]
        ax.set_title(f"{name}\nROC-AUC={roc:.3f} | F1={f1:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("P(income >50K)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        if ax is axes[0]:
            ax.legend(fontsize=9)

    plt.suptitle("Predicted Probability Distributions — Candidate Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\nWider separation between the two score distributions usually means easier thresholding and better ranking quality.")

# %% [markdown]
# ## Summary and Teaching Notes
#
# ### Dataset Facts
# | Property | Value |
# |---|---|
# | Source | Adult Census Income (OpenML `1590`) |
# | Rows | ~48,842 |
# | Features | 14 mixed tabular columns |
# | Target | `1 = income >50K`, `0 = income ≤50K` |
# | Class balance | roughly `76% / 24%` |
#
# ### Workflow Covered
# | Section | Topic |
# |---|---|
# | 2 | Data loading and target binarisation |
# | 3 | Class imbalance + numeric/categorical EDA |
# | 4 | Leakage-safe split and mixed-type preprocessing |
# | 5 | Reusable metric and plotting helpers |
# | 6 | Quick procedural baselines on transformed data |
# | 7 | Baseline pipeline models |
# | 8 | Validation curves and learning curves |
# | 9 | Optional Optuna search |
# | 10 | Model selection and diagnostics |
# | 11 | Threshold optimisation |
# | 12 | Model interpretation |
# | 13 | Probability distribution comparison |
#
# ### Key teaching points
# - Accuracy is inflated by the majority class and is not enough on its own.
# - `ROC-AUC` and `PR-AUC` answer different questions; both matter on imbalanced data.
# - Logistic regression is not only a classifier; it is a probability model, so threshold choice is part of the workflow.
# - `KNN` gives an intuitive geometric baseline, but its runtime becomes a practical limitation on larger datasets.
# - Decision trees are easy to visualize, but depth control matters to avoid overfitting.
