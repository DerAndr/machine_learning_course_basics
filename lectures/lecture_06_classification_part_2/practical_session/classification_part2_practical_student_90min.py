# /// script
# source-notebook = "classification_part2_practical_student_90min.ipynb"
# generated-by = "Codex notebook export"
# ///

# %% [markdown]
# # Classification Part 2: Practical Session - STUDENT VERSION (90 minutes)
#
# **Learning Objectives:**
# - Handle imbalanced binary classification with oversampling, class weights, and threshold tuning
# - Compare native multiclass, One-vs-Rest, and One-vs-One strategies on a real multiclass dataset
# - Build multilabel classifiers with metrics that match multilabel behavior
# - Apply cross-validation without leakage when scaling or resampling are involved
#
# This notebook uses targeted TODO placeholders in the main coding cells while keeping one shared classroom flow across all three group tasks.

# %% [markdown]
# ## Setup

# %% [markdown]
# ## Setup Note
#
# ```python
# # If needed:
# # pip install imbalanced-learn kagglehub openpyxl liac-arff
# ```
#
# In Google Colab, run the install cell once before continuing. `Spambase` and `emotions` are loaded from OpenML, while `Dry Bean` is downloaded with `kagglehub`. `openpyxl` is included because the Kaggle package may contain an Excel file.

# %%
# !pip install -U imbalanced-learn kagglehub openpyxl liac-arff

# %%
import warnings
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, hamming_loss, accuracy_score,
    label_ranking_average_precision_score, make_scorer,
    precision_score, recall_score, average_precision_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    jaccard_score, precision_recall_fscore_support, coverage_error,
    label_ranking_loss, log_loss, top_k_accuracy_score,
    precision_recall_curve, roc_curve, auc
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("✓ All libraries imported")

# %% [markdown]
# ## Shared Helper Functions
#
# These helper utilities keep the practical focused on modeling and interpretation instead of repeating the same plotting and evaluation boilerplate in every block.

# %%
def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """Calculate and visualize a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_heatmap(cm, labels=labels, title=title)

def plot_confusion_matrix_heatmap(cm, labels=None, title='Confusion Matrix'):
    """Visualize confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y, title='Class Distribution'):
    """Visualize class distribution"""
    plt.figure(figsize=(8, 5))
    if isinstance(y, pd.Series):
        y_counts = y.value_counts().sort_index()
    else:
        y_counts = pd.Series(y).value_counts().sort_index()

    plt.bar(y_counts.index, y_counts.values, alpha=0.7, edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)

    total = y_counts.sum()
    for idx in y_counts.index:
        value = y_counts[idx]
        plt.text(idx, value + total * 0.01, f'{value}\n({100 * value / total:.1f}%)',
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def compare_models_metrics(metrics_dict):
    """Create comparison table"""
    return pd.DataFrame(metrics_dict).T.round(3)

def evaluate_multiclass(y_true, y_pred, labels, y_proba=None):
    """Return a richer multiclass metrics dictionary."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }

    if y_proba is not None:
        y_true_bin = label_binarize(y_true, classes=labels)
        metrics['log_loss'] = log_loss(y_true, y_proba, labels=labels)
        metrics['roc_auc_ovr_macro'] = roc_auc_score(
            y_true_bin, y_proba, multi_class='ovr', average='macro'
        )
        metrics['roc_auc_ovr_weighted'] = roc_auc_score(
            y_true_bin, y_proba, multi_class='ovr', average='weighted'
        )
        metrics['top_2_accuracy'] = top_k_accuracy_score(y_true, y_proba, k=2, labels=labels)

    return metrics

def evaluate_multilabel(y_true, y_pred, y_score=None):
    """Return a richer multilabel metrics dictionary."""
    metrics = {
        'hamming_loss': hamming_loss(y_true, y_pred),
        'subset_accuracy': accuracy_score(y_true, y_pred),
        'micro_precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'micro_recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'samples_f1': f1_score(y_true, y_pred, average='samples', zero_division=0),
        'jaccard_micro': jaccard_score(y_true, y_pred, average='micro', zero_division=0),
        'jaccard_macro': jaccard_score(y_true, y_pred, average='macro', zero_division=0),
        'jaccard_samples': jaccard_score(y_true, y_pred, average='samples', zero_division=0),
    }

    if y_score is not None:
        metrics['lrap'] = label_ranking_average_precision_score(y_true, y_score)
        metrics['ranking_loss'] = label_ranking_loss(y_true, y_score)
        metrics['coverage_error'] = coverage_error(y_true, y_score)

    return metrics

print("✓ Utility functions loaded")

def plot_multiclass_roc_pr_curves(y_true, y_score, class_names, title_prefix='Multiclass'):
    """Plot one-vs-rest ROC and PR curves for each class."""
    y_true_bin = label_binarize(y_true, classes=class_names)

    plt.figure(figsize=(9, 7))
    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title_prefix}: One-vs-Rest ROC Curves')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 7))
    for idx, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_score[:, idx])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f'{class_name} (AUC={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title_prefix}: One-vs-Rest PR Curves')
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()

def multilabel_cardinality_density(y):
    """Compute label cardinality and label density for a multilabel target matrix."""
    if isinstance(y, pd.DataFrame):
        y_values = y.values
    else:
        y_values = np.asarray(y)

    label_counts_per_sample = y_values.sum(axis=1)
    n_labels = y_values.shape[1]

    return {
        'label_cardinality': label_counts_per_sample.mean(),
        'label_density': label_counts_per_sample.mean() / n_labels,
        'min_labels_per_sample': label_counts_per_sample.min(),
        'max_labels_per_sample': label_counts_per_sample.max(),
    }

def plot_multilabel_cardinality_distribution(y, title='Multilabel: labels per sample'):
    """Plot how many labels each sample has."""
    if isinstance(y, pd.DataFrame):
        label_counts_per_sample = y.sum(axis=1)
    else:
        label_counts_per_sample = np.asarray(y).sum(axis=1)

    counts = pd.Series(label_counts_per_sample).value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, edgecolor='black', alpha=0.8)
    plt.xlabel('Number of active labels in a sample')
    plt.ylabel('Number of samples')
    plt.title(title)
    for idx, value in counts.items():
        plt.text(idx, value, str(value), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

print("✓ Extended plotting and multilabel summary helpers loaded")


def plot_multilabel_roc_pr_curves(y_true, y_score, label_names, title_prefix='Multilabel'):
    """Plot ROC and PR curves for each multilabel target."""
    if isinstance(y_true, pd.DataFrame):
        y_true_values = y_true.values
    else:
        y_true_values = np.asarray(y_true)

    plt.figure(figsize=(9, 7))
    for idx, label_name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_true_values[:, idx], y_score[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{label_name} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title_prefix}: ROC Curves by Label')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 7))
    for idx, label_name in enumerate(label_names):
        precision, recall, _ = precision_recall_curve(y_true_values[:, idx], y_score[:, idx])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f'{label_name} (AUC={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title_prefix}: PR Curves by Label')
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## How To Work In Teams
#
# 1. **Group A** works on **Section 1**: Imbalanced Binary Classification
# 2. **Group B** works on **Section 2**: Multiclass Classification
# 3. **Group C** works on **Section 3**: Multilabel Classification
# 4. At the end, each group presents:
#    - the model(s) it ran
#    - the main metric trade-offs
#    - one methodological lesson from the task
#
# **Important:**
# - You do **not** need to complete the whole notebook during class.
# - Focus on the TODO placeholders inside your group's block.
# - The remaining pre-filled cells are there to keep the session moving and to support the final discussion.

# %% [markdown]
# ## 1. Imbalanced Binary Classification (⏱️ ~40 min)
#
# **Scenario:** Build a spam detector for an email service using the real UCI Spambase dataset.
#
# Each observation is an email represented by numeric summaries of its content, such as word frequencies, character frequencies, and capital-letter statistics. The target is binary: `0 = ham`, `1 = spam`.
#
# **Business Context:**
# - False Positive (ham → spam): expensive because an important email may be hidden (`$10` cost)
# - False Negative (spam → ham): annoying but cheaper because the user can delete the spam manually (`$1` cost)
#
# **You'll Learn:**
# - Why accuracy can be misleading when the positive class matters
# - Why weighted averages can still hide weak spam detection
# - Random Oversampling vs Class Weights vs SMOTE
# - Why precision-recall analysis is often more informative than accuracy
# - How threshold tuning changes business cost
# - How to avoid leakage when resampling inside cross-validation
#
#
# **Group A:** complete the TODO placeholders in this block. Prioritize the baseline, model comparison, threshold tuning, and leakage-safe CV.

# %% [markdown]
# ### 1.1 Load & Explore Dataset (Pre-filled)

# %%
spambase = fetch_openml(data_id=44, as_frame=True)

X_spam = spambase.data.copy()
y_spam = spambase.target.astype(int).copy()

print(f"Dataset shape: {X_spam.shape}")
print("\nClass distribution (counts):")
print(y_spam.value_counts().sort_index())
print("\nClass distribution (proportions):")
print(y_spam.value_counts(normalize=True).sort_index().round(3))
print("\nFeature preview:")
display(X_spam.head())

plot_class_distribution(y_spam, title='Spambase: Ham (0) vs Spam (1)')

# %% [markdown]
# ### 1.2 Prepare Data (Pre-filled)

# %%
# Train-test split
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(
    X_spam, y_spam, test_size=0.25, stratify=y_spam, random_state=RANDOM_STATE
)

print(f"Training set: {X_train_spam.shape}")
print(f"Test set: {X_test_spam.shape}")
print("\nTrain class distribution:")
print(y_train_spam.value_counts().sort_index())
print("\nScaling will happen inside each modeling pipeline to keep preprocessing and validation clean.")

# %% [markdown]
# ### 1.3 Baseline Model ✏️ TODO (⏱️ ~5 min)
#
# **Your Task:**
# 1. Train `LogisticRegression` on the original training data
# 2. Predict on the test set
# 3. Calculate precision, recall, F1, PR-AUC, and ROC-AUC
# 4. Print a classification report and confusion matrix
#
# **Key Question:** Why can accuracy still look decent even when spam detection is weak?
#
#
# **Group A hint:** only replace the placeholder lines, then run the cell and interpret why accuracy is not enough here.

# %%
# Baseline logistic regression on the original training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

baseline_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('model', ...),  # TODO (Group A): use LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
])

f1_scores_baseline = cross_val_score(
    baseline_pipeline,
    X_train_spam,
    y_train_spam,
    cv=cv,
    scoring=...,  # TODO (Group A): use the F1 metric for binary spam detection
    n_jobs=-1
)

baseline_pipeline.fit(X_train_spam, y_train_spam)
y_pred_baseline = baseline_pipeline.predict(X_test_spam)
y_score_baseline = baseline_pipeline.predict_proba(X_test_spam)[:, 1]

baseline_metrics = {
    'accuracy': accuracy_score(y_test_spam, y_pred_baseline),
    'precision': precision_score(y_test_spam, y_pred_baseline, zero_division=0),
    'recall': recall_score(y_test_spam, y_pred_baseline, zero_division=0),
    'f1': f1_score(y_test_spam, y_pred_baseline, zero_division=0),
    'average_precision': average_precision_score(y_test_spam, y_score_baseline),
    'roc_auc': roc_auc_score(y_test_spam, y_score_baseline),
}

print(f"Baseline CV F1 scores: {np.round(f1_scores_baseline, 3)}")
print(f"Mean CV F1: {f1_scores_baseline.mean():.3f} (+/- {f1_scores_baseline.std():.3f})")
print("\nTest metrics:")
for metric_name, metric_value in baseline_metrics.items():
    print(f"{metric_name}: {metric_value:.3f}")

print("\nClassification report:")
print(classification_report(y_test_spam, y_pred_baseline, target_names=['Ham', 'Spam'], zero_division=0))

plot_confusion_matrix(
    y_test_spam,
    y_pred_baseline,
    title='Baseline Logistic Regression',
    labels=['Ham', 'Spam']
)

print("Method note: accuracy can stay high even when recall for the spam class is still unsatisfactory.")

# %% [markdown]
# ### 1.4 Random Oversampling ✏️ TODO (⏱️ ~6 min)
#
# **Your Task:**
# 1. Use `RandomOverSampler` to rebalance only the training data
# 2. Train the model on resampled data
# 3. Evaluate on the untouched test set
# 4. Compare recall and PR-oriented metrics with the baseline
#
# **Key Point:** Oversampling duplicates minority samples. The test set must stay untouched.

# %%
ros_pipeline = ImbPipeline([
    ('ros', RandomOverSampler(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
])

f1_scores_ros = cross_val_score(
    ros_pipeline,
    X_train_spam,
    y_train_spam,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

ros_pipeline.fit(X_train_spam, y_train_spam)
y_pred_ros = ros_pipeline.predict(X_test_spam)
y_score_ros = ros_pipeline.predict_proba(X_test_spam)[:, 1]

ros_metrics = {
    'accuracy': accuracy_score(y_test_spam, y_pred_ros),
    'precision': precision_score(y_test_spam, y_pred_ros, zero_division=0),
    'recall': recall_score(y_test_spam, y_pred_ros, zero_division=0),
    'f1': f1_score(y_test_spam, y_pred_ros, zero_division=0),
    'average_precision': average_precision_score(y_test_spam, y_score_ros),
    'roc_auc': roc_auc_score(y_test_spam, y_score_ros),
}

print(f"ROS CV F1 scores: {np.round(f1_scores_ros, 3)}")
print(f"Mean CV F1: {f1_scores_ros.mean():.3f} (+/- {f1_scores_ros.std():.3f})")
print("\nTest metrics:")
for metric_name, metric_value in ros_metrics.items():
    print(f"{metric_name}: {metric_value:.3f}")

print("\nClassification report:")
print(classification_report(y_test_spam, y_pred_ros, target_names=['Ham', 'Spam'], zero_division=0))

plot_confusion_matrix(
    y_test_spam,
    y_pred_ros,
    title='Random Oversampling + Logistic Regression',
    labels=['Ham', 'Spam']
)

# %% [markdown]
# ### 1.5 SMOTE ✏️ TODO (⏱️ ~6 min)
#
# **Your Task:**
# 1. Apply `SMOTE` inside the training pipeline
# 2. Train the model and evaluate it on the original test set
# 3. Compare the trade-off with Random Oversampling
#
# **Key Difference:** SMOTE creates synthetic minority examples instead of duplicating existing ones.

# %%
smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
])

f1_scores_smote = cross_val_score(
    smote_pipeline,
    X_train_spam,
    y_train_spam,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

smote_pipeline.fit(X_train_spam, y_train_spam)
y_pred_smote = smote_pipeline.predict(X_test_spam)
y_score_smote = smote_pipeline.predict_proba(X_test_spam)[:, 1]

smote_metrics = {
    'accuracy': accuracy_score(y_test_spam, y_pred_smote),
    'precision': precision_score(y_test_spam, y_pred_smote, zero_division=0),
    'recall': recall_score(y_test_spam, y_pred_smote, zero_division=0),
    'f1': f1_score(y_test_spam, y_pred_smote, zero_division=0),
    'average_precision': average_precision_score(y_test_spam, y_score_smote),
    'roc_auc': roc_auc_score(y_test_spam, y_score_smote),
}

print(f"SMOTE CV F1 scores: {np.round(f1_scores_smote, 3)}")
print(f"Mean CV F1: {f1_scores_smote.mean():.3f} (+/- {f1_scores_smote.std():.3f})")
print("\nTest metrics:")
for metric_name, metric_value in smote_metrics.items():
    print(f"{metric_name}: {metric_value:.3f}")

print("\nClassification report:")
print(classification_report(y_test_spam, y_pred_smote, target_names=['Ham', 'Spam'], zero_division=0))

plot_confusion_matrix(
    y_test_spam,
    y_pred_smote,
    title='SMOTE + Logistic Regression',
    labels=['Ham', 'Spam']
)

# %% [markdown]
# ### 1.6 Class Weights ✏️ TODO (⏱️ ~5 min)
#
# **Your Task:**
# 1. Train `LogisticRegression` with `class_weight='balanced'`
# 2. Compare this loss-based approach with explicit resampling
# 3. Check whether recall improves enough to justify any precision drop
#
# **Advantage:** No duplicated or synthetic data is created.

# %%
weighted_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ))
])

f1_scores_weighted = cross_val_score(
    weighted_pipeline,
    X_train_spam,
    y_train_spam,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

weighted_pipeline.fit(X_train_spam, y_train_spam)
y_pred_weighted = weighted_pipeline.predict(X_test_spam)
y_score_weighted = weighted_pipeline.predict_proba(X_test_spam)[:, 1]

weighted_metrics = {
    'accuracy': accuracy_score(y_test_spam, y_pred_weighted),
    'precision': precision_score(y_test_spam, y_pred_weighted, zero_division=0),
    'recall': recall_score(y_test_spam, y_pred_weighted, zero_division=0),
    'f1': f1_score(y_test_spam, y_pred_weighted, zero_division=0),
    'average_precision': average_precision_score(y_test_spam, y_score_weighted),
    'roc_auc': roc_auc_score(y_test_spam, y_score_weighted),
}

print(f"Class-weighted CV F1 scores: {np.round(f1_scores_weighted, 3)}")
print(f"Mean CV F1: {f1_scores_weighted.mean():.3f} (+/- {f1_scores_weighted.std():.3f})")
print("\nTest metrics:")
for metric_name, metric_value in weighted_metrics.items():
    print(f"{metric_name}: {metric_value:.3f}")

print("\nClassification report:")
print(classification_report(y_test_spam, y_pred_weighted, target_names=['Ham', 'Spam'], zero_division=0))

plot_confusion_matrix(
    y_test_spam,
    y_pred_weighted,
    title='Class-Weighted Logistic Regression',
    labels=['Ham', 'Spam']
)

# %% [markdown]
# ### 1.7 Compare All Approaches ✏️ TODO (⏱️ ~5 min)
#
# **Your Task:**
# 1. Build a comparison table for all four approaches
# 2. Focus on recall, F1, and PR-AUC rather than accuracy alone
# 3. Decide which model is the best candidate for threshold tuning
#
# **Method note:** For spam detection, precision-recall trade-offs usually matter more than weighted averages.

# %%
task1_model_objects = {
    'Baseline': baseline_pipeline,
    'RandomOverSampler': ros_pipeline,
    'SMOTE': smote_pipeline,
    'ClassWeight': weighted_pipeline,
}

models_comparison = pd.DataFrame([
    {
        'Model': 'Baseline',
        'cv_f1_mean': f1_scores_baseline.mean(),
        'test_precision': baseline_metrics['precision'],
        'test_recall': baseline_metrics['recall'],
        'test_f1': baseline_metrics['f1'],
        'test_average_precision': baseline_metrics['average_precision'],
        'test_roc_auc': baseline_metrics['roc_auc'],
    },
    {
        'Model': 'RandomOverSampler',
        'cv_f1_mean': f1_scores_ros.mean(),
        'test_precision': ros_metrics['precision'],
        'test_recall': ros_metrics['recall'],
        'test_f1': ros_metrics['f1'],
        'test_average_precision': ros_metrics['average_precision'],
        'test_roc_auc': ros_metrics['roc_auc'],
    },
    {
        'Model': 'SMOTE',
        'cv_f1_mean': f1_scores_smote.mean(),
        'test_precision': smote_metrics['precision'],
        'test_recall': smote_metrics['recall'],
        'test_f1': smote_metrics['f1'],
        'test_average_precision': smote_metrics['average_precision'],
        'test_roc_auc': smote_metrics['roc_auc'],
    },
    {
        'Model': 'ClassWeight',
        'cv_f1_mean': f1_scores_weighted.mean(),
        'test_precision': weighted_metrics['precision'],
        'test_recall': weighted_metrics['recall'],
        'test_f1': weighted_metrics['f1'],
        'test_average_precision': weighted_metrics['average_precision'],
        'test_roc_auc': weighted_metrics['roc_auc'],
    },
]).sort_values(by=..., ascending=False)  # TODO (Group A): sort by the main CV selection metric

print("Model comparison (sorted by cross-validated F1):")
display(models_comparison.round(3))

best_task1_model_name = models_comparison.iloc[0]['Model']
best_task1_model = task1_model_objects[best_task1_model_name]

print(f"\nBest candidate for threshold tuning: {best_task1_model_name}")
print("Method note: the candidate model is chosen from CV, while the test metrics stay reserved for final comparison.")

plt.figure(figsize=(10, 5))
comparison_plot = models_comparison.melt(
    id_vars='Model',
    value_vars=['test_precision', 'test_recall', 'test_f1', 'test_average_precision'],
    var_name='metric',
    value_name='score'
)
sns.barplot(data=comparison_plot, x='Model', y='score', hue='metric')
plt.title('Task 1 Model Comparison on Test Metrics')
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.8 Threshold Optimization ✏️ TODO (⏱️ ~8 min)
#
# **Business Goal:** Minimize cost = `FP × $10 + FN × $1`
#
# **Your Task:**
# 1. Choose the strongest candidate from the comparison table
# 2. Split the training set again into train/validation
# 3. Search thresholds from `0.10` to `0.90`
# 4. Choose the validation threshold with the lowest business cost
# 5. Compare the default `0.50` threshold with the tuned threshold on the held-out test set
#
# **Key Idea:** The best probability threshold is often not `0.50`.
#
#
# **Group A hint:** the key coding step is the business-cost formula; the important discussion step is why the best threshold is rarely exactly 0.50.

# %%
X_train_threshold, X_val_threshold, y_train_threshold, y_val_threshold = train_test_split(
    X_train_spam,
    y_train_spam,
    test_size=0.20,
    stratify=y_train_spam,
    random_state=RANDOM_STATE
)

best_task1_model.fit(X_train_threshold, y_train_threshold)
val_scores = best_task1_model.predict_proba(X_val_threshold)[:, 1]

threshold_rows = []
for threshold in np.arange(0.10, 0.91, 0.05):
    val_pred = (val_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val_threshold, val_pred).ravel()
    threshold_rows.append({
        'threshold': threshold,
        'precision': precision_score(y_val_threshold, val_pred, zero_division=0),
        'recall': recall_score(y_val_threshold, val_pred, zero_division=0),
        'f1': f1_score(y_val_threshold, val_pred, zero_division=0),
        'cost': ...,  # TODO (Group A): implement business cost = FP * 10 + FN * 1
    })

threshold_df = pd.DataFrame(threshold_rows)
best_threshold = threshold_df.sort_values(['cost', 'f1'], ascending=[True, False]).iloc[0]['threshold']

print(f"Validation-selected threshold for {best_task1_model_name}: {best_threshold:.2f}")
display(threshold_df.round(3))

best_task1_model.fit(X_train_spam, y_train_spam)
test_scores = best_task1_model.predict_proba(X_test_spam)[:, 1]

y_pred_default_threshold = (test_scores >= 0.50).astype(int)
y_pred_tuned_threshold = (test_scores >= best_threshold).astype(int)

for threshold_name, predictions in [('Default 0.50', y_pred_default_threshold), (f'Tuned {best_threshold:.2f}', y_pred_tuned_threshold)]:
    tn, fp, fn, tp = confusion_matrix(y_test_spam, predictions).ravel()
    cost = ...  # TODO (Group A): use the same business-cost formula here
    print(f"\n{threshold_name}")
    print(f"precision: {precision_score(y_test_spam, predictions, zero_division=0):.3f}")
    print(f"recall: {recall_score(y_test_spam, predictions, zero_division=0):.3f}")
    print(f"f1: {f1_score(y_test_spam, predictions, zero_division=0):.3f}")
    print(f"business cost: {cost}")
    plot_confusion_matrix(
        y_test_spam,
        predictions,
        title=f'{best_task1_model_name} at threshold {threshold_name}',
        labels=['Ham', 'Spam']
    )

plt.figure(figsize=(8, 4))
sns.lineplot(data=threshold_df, x='threshold', y='cost', marker='o')
plt.title('Validation Cost by Threshold')
plt.ylabel('Cost = FP × 10 + FN × 1')
plt.tight_layout()
plt.show()

print("Method note: this uses a validation split for threshold selection so the final test set remains a cleaner check.")

# %% [markdown]
# ### 1.9 Cross-Validation with SMOTE ✏️ TODO (⏱️ ~5 min)
#
# **Critical:** Apply SMOTE *inside* each CV fold to avoid leakage.
#
# **Your Task:**
# 1. Use a pipeline with `SMOTE`, `StandardScaler`, and `LogisticRegression`
# 2. Run 5-fold stratified CV
# 3. Report a PR-oriented metric such as `average_precision`

# %%
cv_smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
])

cv_scores_smote_pr = cross_val_score(
    cv_smote_pipeline,
    X_train_spam,
    y_train_spam,
    cv=cv,
    scoring=...,  # TODO (Group A): use a PR-oriented metric
    n_jobs=-1
)

print(f"SMOTE pipeline PR-AUC scores: {np.round(cv_scores_smote_pr, 3)}")
print(f"Mean PR-AUC: {cv_scores_smote_pr.mean():.3f} (+/- {cv_scores_smote_pr.std():.3f})")
print("Method note: SMOTE is fitted inside each fold, so validation data never leaks into the resampling step.")

# %% [markdown]
# ## 2. Multiclass Classification (⏱️ ~40 min)
#
# **Scenario:** Predict the bean variety for each sample in the real UCI Dry Bean dataset.
#
# Each observation is described by numeric shape and geometry features extracted from bean images. The target has seven classes.
#
# **You'll Learn:**
# - How native multiclass logistic regression differs from OvR and OvO wrappers
# - Why macro F1 and weighted F1 tell different stories
# - How to inspect class-level confusion on a real multiclass problem
# - How to keep optional SMOTE usage methodologically clean
#
#
# **Group B:** complete the TODO placeholders in this block. Prioritize the baseline model, OvR vs OvO comparison, and macro-F1-based model selection.

# %% [markdown]
# ### 2.1 Load Dry Bean Dataset (Pre-filled)

# %%
dry_bean_dir = Path(kagglehub.dataset_download('sansuthi/dry-bean-dataset'))
dry_bean_candidates = sorted(dry_bean_dir.rglob('*.csv')) + sorted(dry_bean_dir.rglob('*.xlsx'))
if not dry_bean_candidates:
    raise FileNotFoundError(f'No CSV or XLSX file found in {dry_bean_dir}')

dry_bean_path = dry_bean_candidates[0]
if dry_bean_path.suffix.lower() == '.csv':
    dry_bean_frame = pd.read_csv(dry_bean_path)
else:
    dry_bean_frame = pd.read_excel(dry_bean_path)

if 'Class' not in dry_bean_frame.columns:
    raise ValueError(f"Expected a 'Class' column in {dry_bean_path.name}, found: {list(dry_bean_frame.columns)}")

X_multi = dry_bean_frame.drop(columns='Class').copy()
y_multi = dry_bean_frame['Class'].astype(str).copy()
bean_labels = sorted(y_multi.unique())
print(f'Loaded Dry Bean data from: {dry_bean_path}')

print(f"Dataset shape: {X_multi.shape}")
print("\nClass distribution:")
print(y_multi.value_counts().sort_index())
print("\nFeature preview:")
display(X_multi.head())

plot_class_distribution(y_multi, title='Dry Bean Class Distribution')

# %% [markdown]
# ### 2.2 Prepare Data (Pre-filled)

# %%
# Train-test split
X_train_bean, X_test_bean, y_train_bean, y_test_bean = train_test_split(
    X_multi, y_multi, test_size=0.25, stratify=y_multi, random_state=RANDOM_STATE
)

scaler_bean = StandardScaler()
X_train_bean_scaled = scaler_bean.fit_transform(X_train_bean)
X_test_bean_scaled = scaler_bean.transform(X_test_bean)

print(f"Training set: {X_train_bean.shape}")
print(f"Test set: {X_test_bean.shape}")
print("\nTrain class distribution:")
print(y_train_bean.value_counts().sort_index())

# %% [markdown]
# ### 2.3 Baseline Multiclass Model ✏️ TODO (⏱️ ~5 min)
#
# **Your Task:**
# 1. Train a native multiclass `LogisticRegression`
# 2. Evaluate with accuracy, macro F1, and weighted F1
# 3. Print the classification report and confusion matrix
#
# **Key Metric Contrast:**
# - **Macro F1** gives each bean class equal weight
# - **Weighted F1** gives larger classes more influence
#
#
# **Group B hint:** pay attention to the difference between native multiclass logistic regression and explicit wrappers like OvR and OvO.

# %%
baseline_bean = ...  # TODO (Group B): create a native multiclass LogisticRegression model
baseline_bean.fit(X_train_bean_scaled, y_train_bean)

y_pred_bean_base = baseline_bean.predict(X_test_bean_scaled)
y_proba_bean_base = baseline_bean.predict_proba(X_test_bean_scaled)

bean_base_metrics = evaluate_multiclass(
    y_test_bean,
    y_pred_bean_base,
    labels=bean_labels,
    y_proba=y_proba_bean_base
)

print("Baseline multiclass logistic regression:")
display(pd.Series(bean_base_metrics).round(3).to_frame('value'))

print("\nClassification report:")
print(classification_report(y_test_bean, y_pred_bean_base, zero_division=0))

cm = confusion_matrix(y_test_bean, y_pred_bean_base, labels=bean_labels)
plot_confusion_matrix_heatmap(cm, labels=bean_labels, title='Dry Bean Baseline Confusion Matrix')

# %% [markdown]
# ### 2.4 Optional SMOTE Extension ✏️ TODO (⏱️ ~7 min)
#
# **Your Task:**
# 1. Apply `SMOTE` only to the training data
# 2. Fit the same model on the balanced version
# 3. Compare macro F1 and weighted F1 with the original baseline
#
# **Note:** The real Dry Bean dataset is kept as-is. This SMOTE step is an instructional extension, not part of the original data source.

# %%
smote_bean = SMOTE(random_state=RANDOM_STATE)
X_train_bean_smote, y_train_bean_smote = smote_bean.fit_resample(X_train_bean, y_train_bean)

print("After SMOTE (training split only):")
print(pd.Series(y_train_bean_smote).value_counts().sort_index())

smote_bean_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
])
smote_bean_model.fit(X_train_bean_smote, y_train_bean_smote)

y_pred_bean_smote = smote_bean_model.predict(X_test_bean)
y_proba_bean_smote = smote_bean_model.predict_proba(X_test_bean)

bean_smote_metrics = evaluate_multiclass(
    y_test_bean,
    y_pred_bean_smote,
    labels=bean_labels,
    y_proba=y_proba_bean_smote
)

print("\nSMOTE extension:")
display(pd.Series(bean_smote_metrics).round(3).to_frame('value'))

print("\nClassification report:")
print(classification_report(y_test_bean, y_pred_bean_smote, zero_division=0))

cm = confusion_matrix(y_test_bean, y_pred_bean_smote, labels=bean_labels)
plot_confusion_matrix_heatmap(cm, labels=bean_labels, title='Dry Bean SMOTE Confusion Matrix')

# %% [markdown]
# ### 2.5 One-vs-Rest (OvR) Strategy ✏️ TODO (⏱️ ~6 min)
#
# **Your Task:**
# 1. Explicitly wrap logistic regression in `OneVsRestClassifier`
# 2. Train on the original class distribution
# 3. Evaluate and compare with the native multiclass baseline
#
# **Concept:** OvR trains one binary classifier per class: "Is this sample class X or not?"

# %%
ovr_bean = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ...),  # TODO (Group B): wrap LogisticRegression in OneVsRestClassifier
])
ovr_bean.fit(X_train_bean, y_train_bean)

y_pred_ovr = ovr_bean.predict(X_test_bean)
y_proba_ovr = ovr_bean.predict_proba(X_test_bean)

bean_ovr_metrics = evaluate_multiclass(
    y_test_bean,
    y_pred_ovr,
    labels=bean_labels,
    y_proba=y_proba_ovr
)

print("One-vs-Rest:")
display(pd.Series(bean_ovr_metrics).round(3).to_frame('value'))

print("\nClassification report:")
print(classification_report(y_test_bean, y_pred_ovr, zero_division=0))

# %% [markdown]
# ### 2.6 One-vs-One (OvO) Strategy ✏️ TODO (⏱️ ~6 min)
#
# **Your Task:**
# 1. Use `OneVsOneClassifier`
# 2. Train on the original class distribution
# 3. Evaluate and compare with OvR
#
# **Concept:** OvO trains one classifier for each pair of classes. With 7 bean classes, that means 21 binary models.

# %%
ovo_bean = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ...),  # TODO (Group B): wrap LogisticRegression in OneVsOneClassifier
])
ovo_bean.fit(X_train_bean, y_train_bean)

y_pred_ovo = ovo_bean.predict(X_test_bean)

bean_ovo_metrics = evaluate_multiclass(
    y_test_bean,
    y_pred_ovo,
    labels=bean_labels,
    y_proba=None   # OvO does not expose class probabilities in this setup
)

print("One-vs-One:")
display(pd.Series(bean_ovo_metrics).round(3).to_frame('value'))

print("\nClassification report:")
print(classification_report(y_test_bean, y_pred_ovo, zero_division=0))

print(f"\nNumber of OvO estimators trained: {len(ovo_bean.named_steps['clf'].estimators_)}")

# %% [markdown]
# ### 2.7 Compare Strategies ✏️ TODO (⏱️ ~4 min)
#
# **Your Task:**
# 1. Create one comparison table for all approaches
# 2. Identify which model wins on macro F1
# 3. Discuss whether the gain is worth the extra complexity
#
#
# **Group B hint:** macro F1 is the main decision metric here because it treats all bean classes equally.

# %%
multiclass_comparison = {
    'Native Logistic': bean_base_metrics,
    'SMOTE Extension': bean_smote_metrics,
    'OvR': bean_ovr_metrics,
    'OvO': bean_ovo_metrics,
}

comparison_df = compare_models_metrics(multiclass_comparison)

sort_metric = ...  # TODO (Group B): choose the main multiclass comparison metric
comparison_df = comparison_df.sort_values(sort_metric, ascending=False)

print("Multiclass strategy comparison:")
display(comparison_df)

multiclass_predictions = {
    'Native Logistic': y_pred_bean_base,
    'SMOTE Extension': y_pred_bean_smote,
    'OvR': y_pred_ovr,
    'OvO': y_pred_ovo,
}
multiclass_probabilities = {
    'Native Logistic': y_proba_bean_base,
    'SMOTE Extension': y_proba_bean_smote,
    'OvR': y_proba_ovr,
}
best_multiclass_model = comparison_df.index[0]

print(f"Best model by {sort_metric}: {best_multiclass_model}")

comparison_df[['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'mcc', 'cohen_kappa']].plot(
    kind='bar', figsize=(11, 6), alpha=0.85
)
plt.title('Dry Bean Strategy Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.8 Per-Class Analysis ✏️ TODO (⏱️ ~6 min)
#
# **Your Task:**
# 1. Extract class-level F1 scores for the strongest strategy
# 2. Visualize which bean classes are hardest to classify
# 3. Identify the most common confusions

# %%
best_multiclass_predictions = multiclass_predictions[best_multiclass_model]
class_report = classification_report(
    y_test_bean,
    best_multiclass_predictions,
    output_dict=True,
    zero_division=0
)

per_class_df = pd.DataFrame({
    'precision': {label: class_report[label]['precision'] for label in bean_labels},
    'recall': {label: class_report[label]['recall'] for label in bean_labels},
    'f1': {label: class_report[label]['f1-score'] for label in bean_labels},
    'support': {label: class_report[label]['support'] for label in bean_labels},
}).sort_values('f1')

display(per_class_df.round(3))

plt.figure(figsize=(10, 6))
plt.bar(per_class_df.index, per_class_df['f1'].values, alpha=0.8, edgecolor='black')
plt.xlabel('Bean Class')
plt.ylabel('F1-Score')
plt.title(f'Per-Class F1-Scores ({best_multiclass_model})')
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.grid(axis='y', alpha=0.3)

for i, value in enumerate(per_class_df['f1'].values):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

worst_classes = per_class_df.head(3)
print("Worst-performing bean classes:")
display(worst_classes.round(3))

cm_df = pd.DataFrame(
    confusion_matrix(y_test_bean, best_multiclass_predictions, labels=bean_labels),
    index=bean_labels,
    columns=bean_labels
)
hardest_class = worst_classes.index[0]
print(f"\nMost common predictions for true class '{hardest_class}':")
print(cm_df.loc[hardest_class].sort_values(ascending=False).head(4))

if best_multiclass_model in multiclass_probabilities:
    best_multiclass_proba = multiclass_probabilities[best_multiclass_model]
    top2_hits = top_k_accuracy_score(y_test_bean, best_multiclass_proba, k=2, labels=bean_labels)
    print(f"\nTop-2 accuracy for {best_multiclass_model}: {top2_hits:.3f}")

# %% [markdown]
# ### 2.9 ROC and PR Curves by Class ✏️ TODO (⏱️ ~5 min)
#
# **Your Task:**
# 1. Use the probability outputs of the strongest probability-based multiclass model
# 2. Plot **one-vs-rest ROC curves** for all bean classes
# 3. Plot **one-vs-rest PR curves** for all bean classes
#
# **Why this matters:** accuracy and F1 summarize the final class decision, but ROC/PR curves show how separable each class is across thresholds.

# %%
multiclass_proba_candidates = {
    'Native Logistic': y_proba_bean_base,
    'SMOTE Extension': y_proba_bean_smote,
    'OvR': y_proba_ovr,
}

best_multiclass_proba_model = comparison_df.loc[
    comparison_df.index.intersection(multiclass_proba_candidates.keys()),
    'macro_f1'
].idxmax()

print(f"Using probability outputs from: {best_multiclass_proba_model}")

plot_multiclass_roc_pr_curves(
    y_test_bean,
    multiclass_proba_candidates[best_multiclass_proba_model],
    bean_labels,
    title_prefix=f'Dry Bean ({best_multiclass_proba_model})'
)

# %% [markdown]
# ### 2.10 Cross-Validation ✏️ TODO (⏱️ ~6 min)
#
# **Your Task:**
# 1. Perform 5-fold stratified CV
# 2. Keep `SMOTE` inside the pipeline so each fold is clean
# 3. Report mean macro F1

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

bean_cv_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
])

bean_cv_scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'macro_f1': 'f1_macro',
    'weighted_f1': 'f1_weighted',
}

cv_results_bean = cross_validate(
    bean_cv_pipeline,
    X_multi,
    y_multi,
    cv=skf,
    scoring=bean_cv_scoring,
    n_jobs=-1
)

bean_cv_summary = pd.DataFrame({
    metric: cv_results_bean[f'test_{metric}']
    for metric in bean_cv_scoring
})

print("Cross-validation metrics across folds:")
display(bean_cv_summary.round(3))

print("CV mean ± std:")
display(pd.DataFrame({
    'mean': bean_cv_summary.mean().round(3),
    'std': bean_cv_summary.std().round(3)
}))

print("Method note: SMOTE is fit separately inside each training fold, which avoids leakage.")

# %% [markdown]
# ## 3. Multilabel Classification (⏱️ ~35 min)
#
# **Scenario:** Work with the real OpenML `emotions` dataset.
#
# This is a multilabel problem: each sample can belong to several emotional categories at the same time, so the target is a binary label matrix rather than a single class vector.
#
# **You'll Learn:**
# - The difference between multiclass and multilabel prediction
# - Why subset accuracy is strict in multilabel settings
# - Why micro F1 and Hamming loss are often easier to interpret
# - How LRAP evaluates ranking quality across labels
#
#
# **Group C:** complete the TODO placeholders in this block. Prioritize the multilabel model, the main metric summary, and ranking/co-occurrence interpretation.

# %% [markdown]
# ### 3.1 Load emotions Dataset (Pre-filled)

# %%
emotions = fetch_openml(name='emotions', version=4, as_frame=True)

X_ml = emotions.data.copy()
y_raw = emotions.target.copy()

if isinstance(y_raw, pd.DataFrame):
    y_ml = y_raw.copy()
elif isinstance(y_raw, pd.Series):
    first_value = y_raw.dropna().iloc[0]
    if isinstance(first_value, str) and first_value.strip().startswith('['):
        parsed = y_raw.apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)
        y_ml = pd.DataFrame(parsed.tolist())
    elif isinstance(first_value, str) and ',' in first_value:
        y_ml = y_raw.str.get_dummies(sep=',')
    else:
        y_ml = pd.DataFrame(y_raw)
else:
    y_ml = pd.DataFrame(y_raw)

y_ml.columns = [str(col) for col in y_ml.columns]

# Fix: Explicitly replace 'TRUE'/'FALSE' strings with 1/0 before general numeric conversion.
y_ml = y_ml.astype(str).apply(lambda col: col.str.upper().eq('TRUE')).astype(int)

y_ml = y_ml.apply(pd.to_numeric, errors='coerce')

if y_ml.isna().any().any():
    raise ValueError("Could not safely convert the multilabel target to numeric 0/1 indicators.")

y_ml = y_ml.astype(int)
emotion_labels = list(y_ml.columns)

print(f"Feature matrix shape: {X_ml.shape}")
print(f"Target matrix shape: {y_ml.shape}")
print("\nFeature preview:")
display(X_ml.head())
print("\nTarget preview:")
display(y_ml.head())
print("\nLabel frequencies:")
print(y_ml.sum(axis=0).sort_values(ascending=False))
print(f"\nAverage labels per sample: {y_ml.sum(axis=1).mean():.2f}")

# %% [markdown]
# ### 3.2 Prepare Data (Pre-filled)

# %%
# Train-test split
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_ml, y_ml, test_size=0.25, random_state=RANDOM_STATE
)

print(f"Training set: {X_train_ml.shape}")
print(f"Test set: {X_test_ml.shape}")
print("\nTraining label frequencies:")
print(y_train_ml.sum(axis=0).sort_values(ascending=False))

# %% [markdown]
# ### 3.3 Label Cardinality and Label Density ✏️ TODO (⏱️ ~4 min)
#
# **Your Task:**
# 1. Compute **label cardinality** = average number of active labels per sample
# 2. Compute **label density** = cardinality divided by the total number of labels
# 3. Visualize how many labels each sample has
#
# **Interpretation:** cardinality tells you how many labels are typically active; density normalizes that value by the label space size.
#
# **Group C hint:** cardinality is an average count of active labels; density normalizes that count by the total number of labels.

# %%
train_label_stats = ...  # TODO (Group C): compute label cardinality and density on the training split
test_label_stats = ...  # TODO (Group C): compute label cardinality and density on the test split

print("Training multilabel structure:")
display(pd.Series(train_label_stats).round(3).to_frame('value'))

print("Test multilabel structure:")
display(pd.Series(test_label_stats).round(3).to_frame('value'))

plot_multilabel_cardinality_distribution(
    y_train_ml,
    title='Emotions training set: labels per sample'
)

# %% [markdown]
# ### 3.4 Build Multilabel Classifier ✏️ TODO (⏱️ ~6 min)
#
# **Your Task:**
# 1. Use `OneVsRestClassifier` with `LogisticRegression`
# 2. Fit on the training data
# 3. Predict a full label vector for each test sample
#
# **Contrast with multiclass:** multiclass means exactly one class is true; multilabel means several labels can be `1` at the same time.

# %%
multilabel_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ...),  # TODO (Group C): wrap LogisticRegression in OneVsRestClassifier
])

multilabel_pipeline.fit(X_train_ml, y_train_ml)

Y_pred_ml = multilabel_pipeline.predict(X_test_ml)
Y_pred_proba_ml = multilabel_pipeline.predict_proba(X_test_ml)

print("Sample true labels (first 5 rows):")
display(y_test_ml.head())
print("\nSample predicted labels (first 5 rows):")
display(pd.DataFrame(Y_pred_ml[:5], columns=emotion_labels, index=y_test_ml.index[:5]))

# %% [markdown]
# ### 3.5 Multilabel Metrics ✏️ TODO (⏱️ ~8 min)
#
# **Your Task:** Calculate multilabel-specific metrics:
#
# 1. **Hamming Loss**: fraction of incorrect label decisions (lower is better)
# 2. **Subset Accuracy**: exact match across all labels (strict)
# 3. **Macro F1**: average F1 per label
# 4. **Micro F1**: aggregates all label decisions together
#
# **Question:** Why is subset accuracy usually much lower than micro F1?
#
#
# **Group C hint:** subset accuracy is strict because every label for a sample must match exactly.

# %%
multilabel_metrics = ...  # TODO (Group C): evaluate the multilabel predictions with probabilities

print("Multilabel metrics:")
display(pd.Series(multilabel_metrics).round(3).to_frame('value'))

print("\nPer-label classification report:")
print(classification_report(y_test_ml, Y_pred_ml, target_names=emotion_labels, zero_division=0))

per_label_precision, per_label_recall, per_label_f1, per_label_support = precision_recall_fscore_support(
    y_test_ml, Y_pred_ml, zero_division=0
)

per_label_df = pd.DataFrame({
    'precision': per_label_precision,
    'recall': per_label_recall,
    'f1': per_label_f1,
    'support': per_label_support,
    'positive_rate_true': y_test_ml.mean(axis=0).values,
    'positive_rate_pred': pd.DataFrame(Y_pred_ml, columns=emotion_labels).mean(axis=0).values,
}, index=emotion_labels).sort_values('f1')

display(per_label_df.round(3))

plt.figure(figsize=(10, 5))
plt.bar(per_label_df.index, per_label_df['f1'].values, alpha=0.8, edgecolor='black')
plt.xlabel('Emotion Label')
plt.ylabel('F1-Score')
plt.title('Per-Label F1-Scores')
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.grid(axis='y', alpha=0.3)

for i, value in enumerate(per_label_df['f1'].values):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\nMethod note:")
print("- Subset accuracy is strict: every label for a sample must match exactly.")
print("- Micro metrics emphasize the frequent labels.")
print("- Macro metrics expose weak performance on rare labels.")
print("- Samples F1 summarizes quality per observation rather than per label.")

# %% [markdown]
# ### 3.6 Label Ranking / Coverage Metrics ✏️ TODO (⏱️ ~5 min)
#
# **Your Task:**
# 1. Calculate LRAP using predicted probabilities
# 2. Interpret what ranking quality means in a multilabel setting
#
# **LRAP:** Measures whether true labels tend to receive higher scores than false labels.

# %%
lrap = ...  # TODO (Group C): compute LRAP
ranking_loss_ml = ...  # TODO (Group C): compute ranking loss
coverage_error_ml = ...  # TODO (Group C): compute coverage error

print(f"Label Ranking Average Precision (LRAP): {lrap:.3f}")
print(f"Ranking Loss: {ranking_loss_ml:.3f} (lower is better)")
print(f"Coverage Error: {coverage_error_ml:.3f} (lower is better)")

print("\nInterpretation:")
print("- LRAP checks whether true labels are ranked above irrelevant labels.")
print("- Ranking loss penalizes incorrect orderings in the score ranking.")
print("- Coverage error tells us how far down the ranked labels we must go, on average, to cover all true labels.")

# %% [markdown]
# ### 3.7 ROC and PR Curves by Label ✏️ TODO (⏱️ ~5 min)
#
# **Your Task:**
# 1. Use the predicted probabilities for each emotion label
# 2. Plot **ROC curves** for all labels
# 3. Plot **PR curves** for all labels
#
# **Why this matters:** in multilabel problems, each label is effectively a separate binary problem, so per-label threshold behavior can differ a lot.

# %%
plot_multilabel_roc_pr_curves(
    y_test_ml,
    Y_pred_proba_ml,
    emotion_labels,
    title_prefix='Emotions'
)

# %% [markdown]
# ### 3.8 Cross-Validation ✏️ TODO (⏱️ ~5 min)
#
# **Your Task:**
# 1. Perform 5-fold CV for the multilabel pipeline
# 2. Report mean Hamming Loss, micro F1, and subset accuracy
# 3. Keep scaling inside the pipeline

# %%
hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)
micro_f1_scorer = make_scorer(f1_score, average='micro', zero_division=0)
macro_f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)
samples_f1_scorer = make_scorer(f1_score, average='samples', zero_division=0)
jaccard_samples_scorer = make_scorer(jaccard_score, average='samples', zero_division=0)

multilabel_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

multilabel_cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', OneVsRestClassifier(LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)))
])

multilabel_cv_scoring = {
    'hamming_loss': hamming_scorer,
    'subset_accuracy': 'accuracy',
    'micro_f1': micro_f1_scorer,
    'macro_f1': macro_f1_scorer,
    'samples_f1': samples_f1_scorer,
    'jaccard_samples': jaccard_samples_scorer,
}

cv_results_ml = cross_validate(
    multilabel_cv_pipeline,
    X_ml,
    y_ml,
    cv=multilabel_cv,
    scoring=multilabel_cv_scoring,
    n_jobs=-1
)

multilabel_cv_summary = pd.DataFrame({
    metric: (-cv_results_ml[f'test_{metric}'] if metric == 'hamming_loss' else cv_results_ml[f'test_{metric}'])
    for metric in multilabel_cv_scoring
})

print("Cross-validation results (5-fold):")
display(multilabel_cv_summary.round(3))

print("CV mean ± std:")
display(pd.DataFrame({
    'mean': multilabel_cv_summary.mean().round(3),
    'std': multilabel_cv_summary.std().round(3)
}))

# %% [markdown]
# ### 3.9 Analyze Label Co-occurrence ✏️ TODO (⏱️ ~6 min)
#
# **Your Task:**
# 1. Calculate how often emotion-label pairs appear together
# 2. Visualize the co-occurrence matrix
# 3. Identify the strongest label overlaps
#
# **Reading tip:** the co-occurrence matrix is symmetric, so the heatmap intentionally shows only one triangle.

# %%
cooccurrence = y_ml.T.dot(y_ml)
np.fill_diagonal(cooccurrence.values, 0)
# The matrix is symmetric, so we hide one triangle in the heatmap.
mask = np.triu(np.ones_like(cooccurrence, dtype=bool))

plt.figure(figsize=(8, 6))
sns.heatmap(cooccurrence, mask=mask, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Emotion Label Co-occurrence Matrix')
plt.tight_layout()
plt.show()

print("Most common emotion-label pairs:")
pair_counts = []
for i, left_label in enumerate(emotion_labels):
    for j, right_label in enumerate(emotion_labels):
        if j <= i:
            continue
        pair_counts.append((left_label, right_label, int(cooccurrence.iloc[i, j])))

pair_counts = sorted(pair_counts, key=lambda row: row[2], reverse=True)
for left_label, right_label, count in pair_counts[:10]:
    print(f"{left_label} + {right_label}: {count}")

# %% [markdown]
# ## Summary & Reflection
#
# ### Key Takeaways
#
# #### Task 1: Imbalanced Binary Classification
# - Accuracy can hide weak minority-class performance.
# - Precision, recall, F1, and PR-AUC are more informative for spam detection.
# - Resampling must stay inside the training pipeline or CV folds.
# - Threshold tuning should be done on validation logic, not directly on the test set.
#
# #### Task 2: Multiclass Classification
# - Native multiclass logistic regression, OvR, and OvO can behave differently on the same real dataset.
# - Macro F1 highlights small-class weakness that weighted F1 may smooth over.
# - Class-level reports and confusion matrices show which bean varieties are most confusable.
#
# #### Task 3: Multilabel Classification
# - Multilabel targets are matrices, not single-label vectors.
# - Subset accuracy is strict because every label must be correct at once.
# - Micro F1 and Hamming loss are usually easier to interpret than plain accuracy alone.
# - LRAP adds a ranking-based view of prediction quality.
#
# ### Why These Datasets Were Chosen
# - Replaced the synthetic binary dataset with UCI Spambase.
# - Replaced the multiclass toy setup with the real UCI Dry Bean dataset.
# - Replaced the synthetic multilabel dataset with the OpenML emotions dataset.
# - Updated metrics to better match binary imbalance, multiclass, and multilabel settings.
# - Preserved the original teaching structure while cleaning up leakage-prone patterns.

# %% [markdown]
# ## Bonus Challenges (⏱️ ~5 min)
#
# ### Bonus 1: Cost-Sensitive Threshold Analysis
# Compare threshold tuning with class weights. Which approach gives the lowest validation cost and the best test-set trade-off?
#
# ### Bonus 2: Feature Importance
# Inspect `coef_` values to identify the most influential features for spam detection or bean classification.
#
# ### Bonus 3: Ensemble Voting
# Combine Task 1 predictions from the baseline, ROS, SMOTE, and class-weighted models.
#
# ### Bonus 4: Classifier Chains
# Replace `OneVsRestClassifier` with `ClassifierChain` on the multilabel task and compare micro F1.
#
# ### Bonus 5: Another Real Dataset
# Apply the same workflow to a different real dataset and decide which metrics best match the task type.
