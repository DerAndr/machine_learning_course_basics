# /// script
# source-notebook = "ensembles_practical_student_90min.ipynb"
# generated-by = "Codex notebook export"
# ///

# %% [markdown]
# # Ensembles: Practical Session - STUDENT VERSION (90 minutes, 2-group format)
# 
# **Learning objectives:**
# - Compare a single decision tree against ensemble methods on both classification and regression tasks.
# - Show why bagging can stabilize tree-based models.
# - Get a first practical orientation to XGBoost, LightGBM, and CatBoost.
# - Discuss the main advantages and disadvantages of ensembles.
# - Keep stacking visible conceptually without building a leakage-prone implementation during class.

# %% [markdown]
# ## Setup
# 
# For local work in this repository, prefer:
# 
# ```bash
# uv sync --group ensembles
# uv run python tools/check_notebook_environment.py --group ensembles
# ```
# 
# If you run the notebook in Colab or in a fresh environment, first check the next cell. If packages are missing, install them before running the import cell.

# %%
import importlib.util

optional_packages = ['openml', 'xgboost', 'lightgbm', 'catboost']
package_status = {pkg: importlib.util.find_spec(pkg) is not None for pkg in optional_packages}
missing_packages = [pkg for pkg, is_available in package_status.items() if not is_available]

if missing_packages:
    print('Missing packages:', ', '.join(missing_packages))
    print(f"Install if needed: !pip install -q {' '.join(missing_packages)}")
else:
    print('All optional packages are available.')

print('\nBoosting libraries:')
for pkg in ['xgboost', 'lightgbm', 'catboost']:
    status = 'available' if package_status[pkg] else 'missing'
    print(f'- {pkg}: {status}')

# %% [markdown]
# ## Imports

# %%
from time import perf_counter
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sns.set_theme(style='whitegrid')
RANDOM_STATE = 42

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

# %% [markdown]
# ## Shared Plotting Functions
# 
# The next cell keeps only the reusable plotting blocks. Model training and evaluation are written directly in the notebook cells below.

# %%
def plot_confusion(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_density(y_true, y_score, model_name='Model', negative_label='Class 0', positive_label='Class 1'):
    y_score = np.asarray(y_score)
    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_score[np.asarray(y_true) == 0], label=negative_label, fill=True, color='steelblue', alpha=0.30)
    sns.kdeplot(y_score[np.asarray(y_true) == 1], label=positive_label, fill=True, color='indianred', alpha=0.30)
    plt.axvline(0.50, linestyle='--', color='gray', linewidth=1, label='Threshold = 0.50')
    plt.xlim(0, 1)
    plt.xlabel('Predicted probability / score')
    plt.ylabel('Density')
    plt.title(f'Density plot of predicted scores: {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_threshold_sweep(y_true, y_score, model_name='Model'):
    thresholds = np.linspace(0.05, 0.95, 19)
    precision_values = []
    recall_values = []
    f1_values = []

    for threshold in thresholds:
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
        precision_values.append(precision_score(y_true, y_pred, zero_division=0))
        recall_values.append(recall_score(y_true, y_pred, zero_division=0))
        f1_values.append(f1_score(y_true, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1_values))
    best_threshold = thresholds[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision_values, marker='o', label='Precision')
    plt.plot(thresholds, recall_values, marker='o', label='Recall')
    plt.plot(thresholds, f1_values, marker='o', label='F1')
    plt.axvline(0.50, linestyle='--', color='gray', linewidth=1, label='Threshold = 0.50')
    plt.axvline(best_threshold, linestyle='--', color='darkred', linewidth=1.5,
                label=f'Best F1 threshold = {best_threshold:.2f}')
    plt.xlabel('Decision threshold')
    plt.ylabel('Metric value')
    plt.title(f'Threshold sweep: {model_name}')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves(y_true, score_map, title_prefix='Classification comparison'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    prevalence = float(np.mean(y_true))

    for model_name, y_score in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y_true, y_score)

        axes[0].plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={roc_auc:.3f})')
        axes[1].plot(recall, precision, lw=2, label=f'{model_name} (AP={ap:.3f})')

    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1, label='Random baseline')
    axes[0].set_title(f'{title_prefix}: ROC curves')
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].legend()

    axes[1].axhline(prevalence, linestyle='--', color='gray', lw=1,
                    label=f'Positive prevalence = {prevalence:.3f}')
    axes[1].set_title(f'{title_prefix}: PR curves')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def _take_rows(X, indices):
    return X.iloc[indices] if hasattr(X, 'iloc') else X[indices]


def plot_cv_roc_curves(model, X, y, cv=None, model_name='Model'):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    plt.figure(figsize=(8, 5))
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model_fold = clone(model)
        X_train_fold = _take_rows(X, train_idx)
        X_test_fold = _take_rows(X, test_idx)
        y_train_fold = _take_rows(y, train_idx)
        y_test_fold = _take_rows(y, test_idx)

        model_fold.fit(X_train_fold, y_train_fold)
        y_score_fold = model_fold.predict_proba(X_test_fold)[:, 1]

        fpr, tpr, _ = roc_curve(y_test_fold, y_score_fold)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, lw=1, alpha=0.25, label=f'Fold {fold_idx} AUC={fold_auc:.3f}')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='navy', lw=2.5,
             label=f'Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})')
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color='navy',
        alpha=0.15,
        label='± 1 std. dev.',
    )
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'Cross-validated ROC curves: {model_name}')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_top_feature_importance(model, feature_names, top_n=15, title='Feature Importance'):
    if not hasattr(model, 'feature_importances_'):
        print('This model does not expose feature_importances_.')
        return

    importance = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    importance.plot(kind='barh', color='steelblue')
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


def plot_regression_diagnostics(y_true, y_pred, title_prefix='Model', target_label='Target'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    if len(residuals) >= 20:
        zoom_low, zoom_high = np.quantile(residuals, [0.01, 0.99])
        central_mask = (residuals >= zoom_low) & (residuals <= zoom_high)
    else:
        zoom_low, zoom_high = residuals.min(), residuals.max()
        central_mask = np.ones_like(residuals, dtype=bool)

    central_pred = y_pred[central_mask]
    central_residuals = residuals[central_mask]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix}: Residual Diagnostics', fontsize=14, y=1.02)

    axes[0, 0].scatter(central_pred, central_residuals, alpha=0.6, edgecolor='k', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel(f'Predicted {target_label}')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted (central 98%)')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(central_residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residual Distribution (central 98%)')
    axes[0, 1].grid(alpha=0.3)

    stats.probplot(residuals, dist='norm', plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].scatter(y_true, y_pred, alpha=0.6, edgecolor='k', linewidth=0.5)
    diagonal_min = min(np.min(y_true), np.min(y_pred))
    diagonal_max = max(np.max(y_true), np.max(y_pred))
    axes[1, 1].plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max],
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel(f'Actual {target_label}')
    axes[1, 1].set_ylabel(f'Predicted {target_label}')
    axes[1, 1].set_title('Actual vs Predicted')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    q01, q05, q50, q95, q99 = np.quantile(residuals, [0.01, 0.05, 0.50, 0.95, 0.99])
    outlier_share = 100 * (1 - central_mask.mean())

    print('Note: the top row is zoomed to the 1st-99th residual percentiles for readability.')
    print(f'Zoom range used in the top row: [{zoom_low:,.2f}, {zoom_high:,.2f}]')
    print(f'Observations outside that central range: {outlier_share:.2f}%')
    print('\n' + '=' * 60)
    print('RESIDUAL DIAGNOSTICS')
    print('=' * 60)
    print(f'Mean residual:      {residuals.mean():.4f} (should be close to 0)')
    print(f'Std dev residuals:  {residuals.std():.4f}')
    print(f'Min residual:       {residuals.min():.4f}')
    print(f'1% residual:        {q01:.4f}')
    print(f'5% residual:        {q05:.4f}')
    print(f'Median residual:    {q50:.4f}')
    print(f'95% residual:       {q95:.4f}')
    print(f'99% residual:       {q99:.4f}')
    print(f'Max residual:       {residuals.max():.4f}')
    print(f'Abs mean error:     {np.abs(residuals).mean():.4f}')
    print('=' * 60)


def plot_residuals(y_true, y_pred, title_prefix='Model', target_label='Target'):
    plot_regression_diagnostics(y_true, y_pred, title_prefix=title_prefix, target_label=target_label)

# %% [markdown]
# ## Shared Orientation: Ensemble Families and Boosting Libraries
# 
# This notebook keeps the core student work focused on two questions:
# 
# 1. What changes when we move from one decision tree to an ensemble?
# 2. How should we think about different boosting libraries in practice?
# 
# We will use a single encoded feature matrix for all tree-based models so that the comparison stays simple and consistent.
# In real projects, CatBoost can often work directly with raw categorical columns, which is one of its practical advantages.

# %%
ensemble_guide = pd.DataFrame([
    {
        'Method / Library': 'Decision Tree',
        'Family': 'Single model',
        'Why show it': 'Strong local rules, but unstable across data splits.',
        'Typical risk': 'High variance and easy overfitting.',
    },
    {
        'Method / Library': 'Random Forest',
        'Family': 'Bagging',
        'Why show it': 'Stabilizes trees by averaging many decorrelated trees.',
        'Typical risk': 'Less interpretable and slower than one tree.',
    },
    {
        'Method / Library': 'XGBoost',
        'Family': 'Boosting',
        'Why show it': 'Common strong baseline with rich regularization controls.',
        'Typical risk': 'Can be slower and more tuning-heavy than LightGBM.',
    },
    {
        'Method / Library': 'LightGBM',
        'Family': 'Boosting',
        'Why show it': 'Fast histogram-based boosting, especially on larger tabular data.',
        'Typical risk': 'Can overfit small noisy data if left unchecked.',
    },
    {
        'Method / Library': 'CatBoost',
        'Family': 'Boosting',
        'Why show it': 'Often a strong default on categorical tabular data.',
        'Typical risk': 'Still less transparent than simpler baselines.',
    },
    {
        'Method / Library': 'Stacking',
        'Family': 'Meta-ensemble',
        'Why show it': 'Combines different model families through a meta-model.',
        'Typical risk': 'Easy to leak if meta-features are not built out-of-fold.',
    },
])

display(ensemble_guide)
print('Installed boosting libraries:', {name: flag for name, flag in {
    'xgboost': XGBClassifier is not None,
    'lightgbm': LGBMClassifier is not None,
    'catboost': CatBoostClassifier is not None,
}.items()})

# %% [markdown]
# ---
# # Task 1: Classification Track
# 
# **Scenario:** Predict whether a bank client will subscribe to a term deposit using a real OpenML classification dataset.
# 
# **Your goal:** show the difference between one tree, bagging, and boosting on the same task.

# %% [markdown]
# ## 1.1 Load and Explore the Classification Dataset (Pre-filled)
# 
# The loader uses the stable OpenML dataset id for `credit-g`.
# A classroom-size sample is used when the full dataset is large, so the notebook stays runnable during the session.

# %%
classification_data = fetch_openml(data_id=31, as_frame=True, parser='auto', cache=False)
classification_dataset_label = 'credit-g (OpenML data_id=31)'
classification_target_column = 'class'

X_clf = classification_data.frame.drop(columns=[classification_target_column]).copy()
y_clf_raw = classification_data.frame[classification_target_column].astype(str)

if len(X_clf) > 12000:
    sampled_index = X_clf.sample(n=12000, random_state=RANDOM_STATE).index
    X_clf = X_clf.loc[sampled_index].copy()
    y_clf_raw = y_clf_raw.loc[sampled_index].copy()

class_counts = y_clf_raw.value_counts().sort_values(ascending=False)
positive_label = class_counts.idxmin()
negative_label = [label for label in class_counts.index if label != positive_label][0]
y_clf = (y_clf_raw == positive_label).astype(int)

print(f'Classification dataset source: {classification_dataset_label}')
print(f'Shape after optional sampling: {X_clf.shape}')
print(f'Positive class encoded as 1: {positive_label}')
display(class_counts.rename('count').to_frame())
display(X_clf.head())

# %% [markdown]
# ## 1.2 Split and Preprocess the Classification Data
# 
# Fill in the split directly in the notebook and read the preprocessing pipeline carefully.

# %%
X_train_clf_raw, X_test_clf_raw, y_train_clf, y_test_clf = train_test_split(
    X_clf,
    y_clf,
    test_size=...,          # TODO (Group A): choose the test size
    stratify=...,           # TODO (Group A): keep the class balance in the split
    random_state=RANDOM_STATE,
)

clf_categorical_cols = X_train_clf_raw.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
clf_numeric_cols = [col for col in X_train_clf_raw.columns if col not in clf_categorical_cols]

try:
    clf_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    clf_onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

clf_transformers = []
if clf_numeric_cols:
    clf_transformers.append(
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), clf_numeric_cols)
    )
if clf_categorical_cols:
    clf_transformers.append(
        (
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', clf_onehot),
            ]),
            clf_categorical_cols,
        )
    )

clf_preprocessor = ColumnTransformer(
    transformers=clf_transformers,
    remainder='drop',
    verbose_feature_names_out=False,
)

X_train_clf = clf_preprocessor.fit_transform(X_train_clf_raw)
X_test_clf = clf_preprocessor.transform(X_test_clf_raw)
clf_feature_names = np.asarray(clf_preprocessor.get_feature_names_out(), dtype=object)

cleaned_clf_feature_names = []
seen_feature_names = {}
for raw_name in clf_feature_names:
    clean_name = str(raw_name)
    clean_name = clean_name.replace('[', '(').replace(']', ')').replace('<', 'lt').replace('>', 'gt')
    clean_name = clean_name.replace('{', '(').replace('}', ')').replace(':', '_').replace(',', '_')
    clean_name = clean_name.replace(' ', '_')
    if clean_name in seen_feature_names:
        seen_feature_names[clean_name] += 1
        clean_name = f'{clean_name}__{seen_feature_names[clean_name]}'
    else:
        seen_feature_names[clean_name] = 0
    cleaned_clf_feature_names.append(clean_name)

clf_feature_names = np.asarray(cleaned_clf_feature_names, dtype=object)
X_train_clf = pd.DataFrame(X_train_clf, columns=clf_feature_names, index=X_train_clf_raw.index)
X_test_clf = pd.DataFrame(X_test_clf, columns=clf_feature_names, index=X_test_clf_raw.index)

print('Train shape after encoding:', X_train_clf.shape)
print('Test shape after encoding:', X_test_clf.shape)
print('Categorical columns:', len(clf_categorical_cols))
print('Numeric columns:', len(clf_numeric_cols))

# %% [markdown]
# ## 1.3 Single Decision Tree Baseline
# 
# Write the key training steps yourself: define the model, fit it, predict, and compute the metrics.

# %%
clf_tree = DecisionTreeClassifier(
    max_depth=...,          # TODO (Group A): choose a tree depth
    min_samples_leaf=...,   # TODO (Group A): choose a leaf size
    random_state=RANDOM_STATE,
)

start = perf_counter()
clf_tree.fit(..., ...)      # TODO (Group A): fit the model on the training data
clf_tree_fit_time = perf_counter() - start

y_pred_tree = ...           # TODO (Group A): predict on the test data
y_score_tree = ...          # TODO (Group A): get probabilities for class 1. Hint: use predict_proba(... )[:, 1]

clf_tree_result = {
    'Model': 'Decision Tree',
    'Family': 'Single tree',
    'Accuracy': ...,        # TODO calculate accuracy
    'F1': ...,              # TODO calculate F1
    'ROC-AUC': ...,         # TODO calculate ROC-AUC
    'PR-AUC': ...,          # TODO calculate PR-AUC
    'Fit Time (s)': round(clf_tree_fit_time, 3),
}

classification_model_objects = {'Decision Tree': clf_tree}
classification_score_map = {'Decision Tree': y_score_tree}
classification_result_rows = [clf_tree_result]

display(pd.DataFrame(classification_result_rows))
print(classification_report(y_test_clf, y_pred_tree, digits=3))
plot_confusion(
    y_test_clf,
    y_pred_tree,
    labels=[negative_label, positive_label],
    title='Decision Tree confusion matrix',
)

# %% [markdown]
# ## 1.4 Instability Demo: One Tree vs Random Forest (Pre-filled)
# 
# Run this block after you finish the decision tree baseline.
# The point is to show that one good split does not mean one stable model.

# %%
classification_instability_rows = []
for split_seed in range(1, 7):
    X_train_seed_raw, X_test_seed_raw, y_train_seed, y_test_seed = train_test_split(
        X_clf,
        y_clf,
        test_size=0.25,
        stratify=y_clf,
        random_state=split_seed,
    )

    seed_categorical_cols = X_train_seed_raw.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    seed_numeric_cols = [col for col in X_train_seed_raw.columns if col not in seed_categorical_cols]

    try:
        seed_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        seed_onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

    seed_transformers = []
    if seed_numeric_cols:
        seed_transformers.append(
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), seed_numeric_cols)
        )
    if seed_categorical_cols:
        seed_transformers.append(
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', seed_onehot),
                ]),
                seed_categorical_cols,
            )
        )

    seed_preprocessor = ColumnTransformer(
        transformers=seed_transformers,
        remainder='drop',
        verbose_feature_names_out=False,
    )

    X_train_seed = seed_preprocessor.fit_transform(X_train_seed_raw)
    X_test_seed = seed_preprocessor.transform(X_test_seed_raw)
    seed_feature_names = np.asarray(seed_preprocessor.get_feature_names_out(), dtype=object)

    cleaned_seed_feature_names = []
    seen_seed_feature_names = {}
    for raw_name in seed_feature_names:
        clean_name = str(raw_name)
        clean_name = clean_name.replace('[', '(').replace(']', ')').replace('<', 'lt').replace('>', 'gt')
        clean_name = clean_name.replace('{', '(').replace('}', ')').replace(':', '_').replace(',', '_')
        clean_name = clean_name.replace(' ', '_')
        if clean_name in seen_seed_feature_names:
            seen_seed_feature_names[clean_name] += 1
            clean_name = f'{clean_name}__{seen_seed_feature_names[clean_name]}'
        else:
            seen_seed_feature_names[clean_name] = 0
        cleaned_seed_feature_names.append(clean_name)

    seed_feature_names = np.asarray(cleaned_seed_feature_names, dtype=object)
    X_train_seed = pd.DataFrame(X_train_seed, columns=seed_feature_names, index=X_train_seed_raw.index)
    X_test_seed = pd.DataFrame(X_test_seed, columns=seed_feature_names, index=X_test_seed_raw.index)

    seed_models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=split_seed),
        'Random Forest': RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=20,
            random_state=split_seed,
            n_jobs=-1,
        ),
    }

    for model_name, model in seed_models.items():
        model.fit(X_train_seed, y_train_seed)
        y_score_seed = model.predict_proba(X_test_seed)[:, 1]
        classification_instability_rows.append({
            'Split': split_seed,
            'Model': model_name,
            'ROC-AUC': ...,   # TODO calculate ROC-AUC for this split
            'PR-AUC': ...,    # TODO calculate PR-AUC for this split
        })

classification_instability_df = pd.DataFrame(classification_instability_rows)
display(
    classification_instability_df.groupby('Model')[['ROC-AUC', 'PR-AUC']]
    .agg(['mean', 'std'])
    .round(4)
)

plt.figure(figsize=(8, 4))
sns.boxplot(data=classification_instability_df, x='Model', y='ROC-AUC')
sns.stripplot(data=classification_instability_df, x='Model', y='ROC-AUC', color='black', alpha=0.6)
plt.title('Classification stability across repeated splits')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.5 Bagging with Random Forest
# 
# Again, write the main modeling steps directly in code instead of only changing hyperparameters.

# %%
clf_rf = RandomForestClassifier(
    n_estimators=...,       # TODO (Group A): choose the forest size
    min_samples_leaf=...,   # TODO (Group A): choose a leaf size
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

start = perf_counter()
clf_rf.fit(..., ...)        # TODO (Group A): fit the model on the training data
clf_rf_fit_time = perf_counter() - start

y_pred_rf = ...             # TODO (Group A): predict on the test data
y_score_rf = ...            # TODO (Group A): get probabilities for class 1. Hint: use predict_proba(... )[:, 1]

clf_rf_result = {
    'Model': 'Random Forest',
    'Family': 'Bagging',
    'Accuracy': ...,        # TODO calculate accuracy
    'F1': ...,              # TODO calculate F1
    'ROC-AUC': ...,         # TODO calculate ROC-AUC
    'PR-AUC': ...,          # TODO calculate PR-AUC
    'Fit Time (s)': round(clf_rf_fit_time, 3),
}

classification_model_objects['Random Forest'] = clf_rf
classification_score_map['Random Forest'] = y_score_rf
classification_result_rows.append(clf_rf_result)

display(pd.DataFrame([clf_rf_result]))
plot_confusion(
    y_test_clf,
    y_pred_rf,
    labels=[negative_label, positive_label],
    title='Random Forest confusion matrix',
)

# %% [markdown]
# ## 1.6 Boosting Libraries
# 
# Train the available boosting libraries directly in code and compare them with the tree and the forest.

# %%
classification_boosting_results = []
classification_boosting_models = {}

if XGBClassifier is not None:
    classification_boosting_models['XGBoost'] = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method='hist',
    )
if LGBMClassifier is not None:
    classification_boosting_models['LightGBM'] = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
if CatBoostClassifier is not None:
    classification_boosting_models['CatBoost'] = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_seed=RANDOM_STATE,
    )

for model_name, model in classification_boosting_models.items():
    start = perf_counter()
    model.fit(..., ...)     # TODO (Group A): fit each boosting model on the training data
    fit_time = perf_counter() - start

    y_pred = ...            # TODO (Group A): predict on the test data
    y_score = ...           # TODO (Group A): get probabilities for class 1. Hint: use predict_proba(... )[:, 1]

    result = {
        'Model': model_name,
        'Family': 'Boosting',
        'Accuracy': ...,    # TODO calculate accuracy
        'F1': ...,          # TODO calculate F1
        'ROC-AUC': ...,     # TODO calculate ROC-AUC
        'PR-AUC': ...,      # TODO calculate PR-AUC
        'Fit Time (s)': round(fit_time, 3),
    }

    classification_boosting_results.append(result)
    classification_model_objects[model_name] = model
    classification_score_map[model_name] = y_score

classification_boosting_df = pd.DataFrame(classification_boosting_results)
if classification_boosting_df.empty:
    print('No third-party boosting libraries are installed. Run `uv sync --group ensembles` or install the packages in Colab.')
else:
    display(classification_boosting_df.sort_values(['ROC-AUC', 'PR-AUC'], ascending=False))

# %% [markdown]
# ## 1.7 Compare All Classification Models
# 
# **Your task:**
# 1. Build one comparison table.
# 2. Identify the best model by ROC-AUC and PR-AUC.
# 3. Write one sentence about the main trade-off of the winning model.

# %%
classification_summary = pd.DataFrame(classification_result_rows + classification_boosting_results)
classification_summary = classification_summary.sort_values(['ROC-AUC', 'PR-AUC', 'F1'], ascending=False).reset_index(drop=True)
display(classification_summary)

# %% [markdown]
# ## 1.8 Probability Diagnostics: Density, ROC, PR, and Threshold Sweep
# 
# Use the strongest probability-based models to inspect class separation and threshold behavior.
# The density plot is especially useful for discussing where a threshold may be too conservative or too aggressive.
# 
# Interpretation guide:
# - `ROC-AUC`: `0.50` is random, `0.70-0.80` is usable, `0.80-0.90` is strong, and `>0.90` deserves a leakage check.
# - `PR-AUC`: always compare it with the positive-class prevalence shown on the PR plot. A model can have a decent ROC-AUC but only a small gain over prevalence in PR space.
# - Density plot: strong overlap means threshold choice is genuinely hard; clean separation means many thresholds will work.
# - Threshold sweep: if precision and recall move sharply, the model is sensitive to threshold tuning and business assumptions matter.

# %%
selected_probability_models = []
for preferred_model in ['Decision Tree', 'Random Forest']:
    if preferred_model in classification_score_map:
        selected_probability_models.append(preferred_model)

best_boosting_name = next(
    (name for name in classification_summary['Model']
     if name not in {'Decision Tree', 'Random Forest'} and name in classification_score_map),
    None,
)
if best_boosting_name is not None:
    selected_probability_models.append(best_boosting_name)

selected_score_map = {name: classification_score_map[name] for name in selected_probability_models}
print('Probability-based models selected for visual comparison:', selected_probability_models)

best_probability_model_name = next(name for name in classification_summary['Model'] if name in classification_score_map)
plot_density(
    y_test_clf,
    classification_score_map[best_probability_model_name],
    model_name=best_probability_model_name,
    negative_label=str(negative_label),
    positive_label=str(positive_label),
)
plot_roc_pr_curves(y_test_clf, selected_score_map, title_prefix='Classification model comparison')
plot_threshold_sweep(
    y_test_clf,
    classification_score_map[best_probability_model_name],
    model_name=best_probability_model_name,
)

# %% [markdown]
# ## 1.9 Cross-Validated ROC Curve
# 
# This block shows the mean ROC curve and the fold-to-fold spread for one selected model.
# It re-fits the preprocessing + model pipeline inside each fold so the ROC estimate stays leakage-safe.

# %%
cv_model_name = 'Random Forest' if 'Random Forest' in classification_model_objects else best_probability_model_name
print('Cross-validated ROC model:', cv_model_name)
cv_pipeline = Pipeline([
    ('preprocess', clone(clf_preprocessor)),
    ('model', clone(classification_model_objects[cv_model_name])),
])
plot_cv_roc_curves(
    cv_pipeline,
    X_train_clf_raw,
    y_train_clf,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    model_name=cv_model_name,
)

# %% [markdown]
# ## 1.10 Threshold Selection by F1 and Business Cost
# 
# This block makes the threshold discussion explicit.
# Compare two choices for the same model:
# - the threshold that maximizes `F1`
# - the threshold that minimizes a simple business cost function
# 
# Your task:
# 1. Pick one business story for the marketing campaign or invent your own.
# 2. Fill in `SCENARIO_NAME`, `BUSINESS_RATIONALE`, `FP_COST`, and `FN_COST`.
# 3. Before you run the cell, predict whether the business-cost threshold should be higher, lower, or similar to the `F1` threshold.
# 4. After you run the cell, compare the two thresholds and explain whether the result matched your expectation.
# 5. Be ready to justify your choice in class.
# 
# For bank marketing, a false positive can mean an unnecessary outreach attempt, while a false negative can mean missing a potentially responsive customer.
# The exact numbers are adjustable; the point is to show that threshold choice depends on business priorities.

# %%
analysis_model_name = best_probability_model_name
analysis_scores = classification_score_map[analysis_model_name]

# TODO (Group A): choose or invent one business scenario.
# Example scenario ideas:
# - low-cost outreach campaign
# - balanced campaign with limited contact budget
# - high-value campaign where missing a positive customer is very expensive
SCENARIO_NAME = '...'
BUSINESS_RATIONALE = 'Explain in one or two sentences why these costs make sense.'

# TODO (Group A): choose business costs that fit your story.
# FP = outreach cost to a non-converting customer
# FN = missed opportunity on a potentially converting customer
FP_COST = ...
FN_COST = ...

# TODO (Group A): before running the cell, write your prediction.
# Use one of: 'higher', 'lower', 'similar'
PREDICTED_COST_THRESHOLD_DIRECTION = '...'

threshold_grid = np.linspace(0.05, 0.95, 37)
threshold_rows = []

for threshold in threshold_grid:
    y_pred_threshold = (np.asarray(analysis_scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_clf, y_pred_threshold).ravel()
    threshold_rows.append({
        'Threshold': round(threshold, 2),
        'Precision': ...,      # TODO calculate precision
        'Recall': ...,         # TODO calculate recall
        'F1': ...,             # TODO calculate F1
        'False Positives': int(fp),
        'False Negatives': int(fn),
        'Business Cost': int(fp * FP_COST + fn * FN_COST),
    })

threshold_df = pd.DataFrame(threshold_rows)
best_f1_idx = threshold_df['F1'].idxmax()
best_cost_idx = threshold_df['Business Cost'].idxmin()

best_f1_threshold = float(threshold_df.loc[best_f1_idx, 'Threshold'])
best_cost_threshold = float(threshold_df.loc[best_cost_idx, 'Threshold'])
threshold_gap = best_cost_threshold - best_f1_threshold
if threshold_gap > 0.03:
    observed_direction = 'higher'
elif threshold_gap < -0.03:
    observed_direction = 'lower'
else:
    observed_direction = 'similar'

print(f'Analysis model: {analysis_model_name}')
print(f'Scenario: {SCENARIO_NAME}')
print(f'Business rationale: {BUSINESS_RATIONALE}')
print(f'Business costs used: FP={FP_COST}, FN={FN_COST}')
print(f'Your prediction: the business-cost threshold should be {PREDICTED_COST_THRESHOLD_DIRECTION} than the F1 threshold.')
print(f'Observed result: the business-cost threshold is {observed_direction} than the F1 threshold.')
print() 
print('Best threshold by F1:')
display(threshold_df.loc[[best_f1_idx]])
print('Best threshold by business cost:')
display(threshold_df.loc[[best_cost_idx]])

positive_rate = float(np.mean(y_test_clf))
print(f'Positive class prevalence on the evaluation set: {positive_rate:.3f}')

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

ax1.plot(threshold_df['Threshold'], threshold_df['F1'], marker='o', color='navy', label='F1')
ax2.plot(threshold_df['Threshold'], threshold_df['Business Cost'], marker='o', color='darkred', label='Business Cost')

ax1.axvline(best_f1_threshold, linestyle='--', color='navy', linewidth=1.5,
            label=f'Best F1 threshold = {best_f1_threshold:.2f}')
ax1.axvline(best_cost_threshold, linestyle='--', color='darkred', linewidth=1.5,
            label=f'Best cost threshold = {best_cost_threshold:.2f}')
ax1.set_xlabel('Decision threshold')
ax1.set_ylabel('F1', color='navy')
ax2.set_ylabel('Business Cost', color='darkred')
ax1.set_title(f'Threshold selection: {analysis_model_name}')

handles_1, labels_1 = ax1.get_legend_handles_labels()
handles_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc='best')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.11 Feature Importance and Interpretation
# 
# **Your task:**
# 1. Inspect the strongest tree-based classifier.
# 2. Name the most influential features.
# 3. Explain what became harder to interpret relative to one decision tree.

# %%
best_classification_model_name = classification_summary.iloc[0]['Model']
best_classification_model = classification_model_objects[best_classification_model_name]
print('Best classification model:', best_classification_model_name)
plot_top_feature_importance(
    best_classification_model,
    clf_feature_names,
    top_n=15,
    title=f'Classification feature importance: {best_classification_model_name}',
)

# %% [markdown]
# ## 1.12 Group A Debrief Prompts
# 
# Group A should be ready to answer:
# - How did the single decision tree behave relative to Random Forest?
# - Did bagging mainly improve average quality, stability, or both?
# - Which boosting library would you shortlist for a future classification project, and why?
# - Which `FP_COST` and `FN_COST` values did you choose, and what business story do they represent?
# - Did the business-cost threshold end up higher, lower, or similar to the `F1` threshold?
# - What did you lose when you moved from one tree to an ensemble?

# %% [markdown]
# ---
# # Task 2: Regression Track
# 
# **Scenario:** Predict a continuous target using a real OpenML regression dataset.
# 
# **Your goal:** show the difference between one tree, bagging, and boosting on the same task, then inspect residual behavior carefully.

# %% [markdown]
# ## 2.1 Load and Explore the Regression Dataset (Pre-filled)
# 
# The loader uses a fixed OpenML regression dataset so the classroom practical stays consistent.
# A classroom-size sample is used when the dataset is large.

# %%
regression_data = fetch_openml(data_id=42092, as_frame=True, parser='auto', cache=False)
regression_dataset_label = 'house_sales (OpenML data_id=42092)'
regression_target_label = 'target'

X_reg = regression_data.data.copy()
y_reg = pd.to_numeric(pd.Series(regression_data.target, name=regression_target_label), errors='coerce')

valid_rows = y_reg.notna()
X_reg = X_reg.loc[valid_rows].copy()
y_reg = y_reg.loc[valid_rows].astype(float)

if len(X_reg) > 12000:
    sampled_index = X_reg.sample(n=12000, random_state=RANDOM_STATE).index
    X_reg = X_reg.loc[sampled_index].copy()
    y_reg = y_reg.loc[sampled_index].copy()

print(f'Regression dataset source: {regression_dataset_label}')
print(f'Target label: {regression_target_label}')
print(f'Shape after optional sampling: {X_reg.shape}')
print('Target summary:')
display(y_reg.describe().to_frame(name='target'))
display(X_reg.head())

# %% [markdown]
# ## 2.2 Split and Preprocess the Regression Data
# 
# Fill in the split directly in the notebook and read the preprocessing pipeline carefully.

# %%
X_train_reg_raw, X_test_reg_raw, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=...,          # TODO (Group B): choose the test size
    random_state=RANDOM_STATE,
)

reg_categorical_cols = X_train_reg_raw.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
reg_numeric_cols = [col for col in X_train_reg_raw.columns if col not in reg_categorical_cols]

try:
    reg_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    reg_onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

reg_transformers = []
if reg_numeric_cols:
    reg_transformers.append(
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), reg_numeric_cols)
    )
if reg_categorical_cols:
    reg_transformers.append(
        (
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', reg_onehot),
            ]),
            reg_categorical_cols,
        )
    )

reg_preprocessor = ColumnTransformer(
    transformers=reg_transformers,
    remainder='drop',
    verbose_feature_names_out=False,
)

X_train_reg = reg_preprocessor.fit_transform(X_train_reg_raw)
X_test_reg = reg_preprocessor.transform(X_test_reg_raw)
reg_feature_names = np.asarray(reg_preprocessor.get_feature_names_out(), dtype=object)

cleaned_reg_feature_names = []
seen_feature_names = {}
for raw_name in reg_feature_names:
    clean_name = str(raw_name)
    clean_name = clean_name.replace('[', '(').replace(']', ')').replace('<', 'lt').replace('>', 'gt')
    clean_name = clean_name.replace('{', '(').replace('}', ')').replace(':', '_').replace(',', '_')
    clean_name = clean_name.replace(' ', '_')
    if clean_name in seen_feature_names:
        seen_feature_names[clean_name] += 1
        clean_name = f'{clean_name}__{seen_feature_names[clean_name]}'
    else:
        seen_feature_names[clean_name] = 0
    cleaned_reg_feature_names.append(clean_name)

reg_feature_names = np.asarray(cleaned_reg_feature_names, dtype=object)
X_train_reg = pd.DataFrame(X_train_reg, columns=reg_feature_names, index=X_train_reg_raw.index)
X_test_reg = pd.DataFrame(X_test_reg, columns=reg_feature_names, index=X_test_reg_raw.index)

print('Train shape after encoding:', X_train_reg.shape)
print('Test shape after encoding:', X_test_reg.shape)
print('Categorical columns:', len(reg_categorical_cols))
print('Numeric columns:', len(reg_numeric_cols))

# %% [markdown]
# ## 2.3 Single Decision Tree Baseline
# 
# Write the key training steps yourself: define the model, fit it, predict, and compute the metrics.

# %%
reg_tree = DecisionTreeRegressor(
    max_depth=...,          # TODO (Group B): choose a tree depth
    min_samples_leaf=...,   # TODO (Group B): choose a leaf size
    random_state=RANDOM_STATE,
)

start = perf_counter()
reg_tree.fit(..., ...)      # TODO (Group B): fit the model on the training data
reg_tree_fit_time = perf_counter() - start

y_pred_reg_tree = ...      # TODO (Group B): predict on the test data
reg_tree_rmse = ...        # TODO calculate RMSE
if np.any(np.isclose(np.asarray(y_test_reg), 0.0)):
    reg_tree_mape = np.nan
else:
    reg_tree_mape = ...    # TODO calculate MAPE

reg_tree_result = {
    'Model': 'Decision Tree',
    'Family': 'Single tree',
    'RMSE': round(reg_tree_rmse, 4),
    'MAE': ...,            # TODO calculate MAE
    'MedianAE': ...,       # TODO calculate MedianAE
    'R2': ...,             # TODO calculate R2
    'Explained Var': ...,  # TODO calculate explained variance
    'MAPE': round(reg_tree_mape, 4) if np.isfinite(reg_tree_mape) else np.nan,
    'Fit Time (s)': round(reg_tree_fit_time, 3),
}

regression_model_objects = {'Decision Tree': reg_tree}
regression_result_rows = [reg_tree_result]

display(pd.DataFrame(regression_result_rows))
plot_residuals(y_test_reg, y_pred_reg_tree, title_prefix='Decision Tree', target_label=regression_target_label)

# %% [markdown]
# ## 2.4 Instability Demo: One Tree vs Random Forest (Pre-filled)
# 
# Run this block after you finish the decision tree baseline.
# The point is to show that one good split does not mean one stable regressor.

# %%
regression_instability_rows = []
for split_seed in range(1, 7):
    X_train_seed_raw, X_test_seed_raw, y_train_seed, y_test_seed = train_test_split(
        X_reg,
        y_reg,
        test_size=0.25,
        random_state=split_seed,
    )

    seed_categorical_cols = X_train_seed_raw.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    seed_numeric_cols = [col for col in X_train_seed_raw.columns if col not in seed_categorical_cols]

    try:
        seed_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        seed_onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

    seed_transformers = []
    if seed_numeric_cols:
        seed_transformers.append(
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), seed_numeric_cols)
        )
    if seed_categorical_cols:
        seed_transformers.append(
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', seed_onehot),
                ]),
                seed_categorical_cols,
            )
        )

    seed_preprocessor = ColumnTransformer(
        transformers=seed_transformers,
        remainder='drop',
        verbose_feature_names_out=False,
    )

    X_train_seed = seed_preprocessor.fit_transform(X_train_seed_raw)
    X_test_seed = seed_preprocessor.transform(X_test_seed_raw)
    seed_feature_names = np.asarray(seed_preprocessor.get_feature_names_out(), dtype=object)

    cleaned_seed_feature_names = []
    seen_seed_feature_names = {}
    for raw_name in seed_feature_names:
        clean_name = str(raw_name)
        clean_name = clean_name.replace('[', '(').replace(']', ')').replace('<', 'lt').replace('>', 'gt')
        clean_name = clean_name.replace('{', '(').replace('}', ')').replace(':', '_').replace(',', '_')
        clean_name = clean_name.replace(' ', '_')
        if clean_name in seen_seed_feature_names:
            seen_seed_feature_names[clean_name] += 1
            clean_name = f'{clean_name}__{seen_seed_feature_names[clean_name]}'
        else:
            seen_seed_feature_names[clean_name] = 0
        cleaned_seed_feature_names.append(clean_name)

    seed_feature_names = np.asarray(cleaned_seed_feature_names, dtype=object)
    X_train_seed = pd.DataFrame(X_train_seed, columns=seed_feature_names, index=X_train_seed_raw.index)
    X_test_seed = pd.DataFrame(X_test_seed, columns=seed_feature_names, index=X_test_seed_raw.index)

    seed_models = {
        'Decision Tree': DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=split_seed),
        'Random Forest': RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=5,
            random_state=split_seed,
            n_jobs=-1,
        ),
    }

    for model_name, model in seed_models.items():
        model.fit(X_train_seed, y_train_seed)
        y_pred_seed = model.predict(X_test_seed)
        regression_instability_rows.append({
            'Split': split_seed,
            'Model': model_name,
            'RMSE': np.sqrt(mean_squared_error(y_test_seed, y_pred_seed)),
        })

regression_instability_df = pd.DataFrame(regression_instability_rows)
display(
    regression_instability_df.groupby('Model')[['RMSE']]
    .agg(['mean', 'std'])
    .round(4)
)

plt.figure(figsize=(8, 4))
sns.boxplot(data=regression_instability_df, x='Model', y='RMSE')
sns.stripplot(data=regression_instability_df, x='Model', y='RMSE', color='black', alpha=0.6)
plt.title('Regression stability across repeated splits')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.5 Bagging with Random Forest
# 
# Again, write the main modeling steps directly in code instead of only changing hyperparameters.

# %%
reg_rf = RandomForestRegressor(
    n_estimators=...,       # TODO (Group B): choose the forest size
    min_samples_leaf=...,   # TODO (Group B): choose a leaf size
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

start = perf_counter()
reg_rf.fit(..., ...)        # TODO (Group B): fit the model on the training data
reg_rf_fit_time = perf_counter() - start

y_pred_reg_rf = ...        # TODO (Group B): predict on the test data
reg_rf_rmse = ...          # TODO calculate RMSE
if np.any(np.isclose(np.asarray(y_test_reg), 0.0)):
    reg_rf_mape = np.nan
else:
    reg_rf_mape = ...      # TODO calculate MAPE

reg_rf_result = {
    'Model': 'Random Forest',
    'Family': 'Bagging',
    'RMSE': round(reg_rf_rmse, 4),
    'MAE': ...,            # TODO calculate MAE
    'MedianAE': ...,       # TODO calculate MedianAE
    'R2': ...,             # TODO calculate R2
    'Explained Var': ...,  # TODO calculate explained variance
    'MAPE': round(reg_rf_mape, 4) if np.isfinite(reg_rf_mape) else np.nan,
    'Fit Time (s)': round(reg_rf_fit_time, 3),
}

regression_model_objects['Random Forest'] = reg_rf
regression_result_rows.append(reg_rf_result)

display(pd.DataFrame([reg_rf_result]))
plot_residuals(y_test_reg, y_pred_reg_rf, title_prefix='Random Forest', target_label=regression_target_label)

# %% [markdown]
# ## 2.6 Boosting Libraries
# 
# Train the available boosting libraries directly in code and compare them with the tree and the forest.

# %%
regression_boosting_results = []
regression_boosting_models = {}

if XGBRegressor is not None:
    regression_boosting_models['XGBoost'] = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method='hist',
    )
if LGBMRegressor is not None:
    regression_boosting_models['LightGBM'] = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
if CatBoostRegressor is not None:
    regression_boosting_models['CatBoost'] = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_seed=RANDOM_STATE,
    )

for model_name, model in regression_boosting_models.items():
    start = perf_counter()
    model.fit(..., ...)     # TODO (Group B): fit each boosting model on the training data
    fit_time = perf_counter() - start

    y_pred = ...            # TODO (Group B): predict on the test data
    rmse = ...              # TODO calculate RMSE
    if np.any(np.isclose(np.asarray(y_test_reg), 0.0)):
        test_mape = np.nan
    else:
        test_mape = ...     # TODO calculate MAPE

    result = {
        'Model': model_name,
        'Family': 'Boosting',
        'RMSE': round(rmse, 4),
        'MAE': ...,            # TODO calculate MAE
        'MedianAE': ...,       # TODO calculate MedianAE
        'R2': ...,             # TODO calculate R2
        'Explained Var': ...,  # TODO calculate explained variance
        'MAPE': round(test_mape, 4) if np.isfinite(test_mape) else np.nan,
        'Fit Time (s)': round(fit_time, 3),
    }

    regression_boosting_results.append(result)
    regression_model_objects[model_name] = model

regression_boosting_df = pd.DataFrame(regression_boosting_results)
if regression_boosting_df.empty:
    print('No third-party boosting libraries are installed. Run `uv sync --group ensembles` or install the packages in Colab.')
else:
    display(regression_boosting_df.sort_values(['RMSE', 'MAE', 'R2'], ascending=[True, True, False]))

# %% [markdown]
# ## 2.7 Compare All Regression Models
# 
# **Your task:**
# 1. Build one comparison table.
# 2. Identify the best model by RMSE.
# 3. Write one sentence about the main trade-off of the winning model.

# %%
regression_summary = pd.DataFrame(regression_result_rows + regression_boosting_results)
regression_summary = regression_summary.sort_values(['RMSE', 'MAE', 'R2'], ascending=[True, True, False]).reset_index(drop=True)
display(regression_summary)

# %% [markdown]
# ## 2.8 Best Model Diagnostics: Feature Importance, Residuals, Q-Q Plot, and Actual vs Predicted
# 
# This block follows the same logic as a practical regression review: select the best model by test RMSE, then inspect residual behavior in more detail.
# 
# Interpretation guide:
# - `RMSE` vs `MAE`: if `RMSE` is much larger than `MAE`, a small number of large misses are dominating the error.
# - Mean residual near zero is necessary but not sufficient: it can still hide very asymmetric tails.
# - Q-Q plot: a straight middle with bent tails usually means outliers or heavy-tailed errors.
# - Actual vs predicted: systematic points below the diagonal at the high end suggest underprediction of expensive cases.
# - The top row is zoomed to the central residual range for readability; the summary statistics below still use all observations.

# %%
best_model_idx = regression_summary['RMSE'].idxmin()
best_regression_model_name = regression_summary.loc[best_model_idx, 'Model']
best_regression_model = regression_model_objects[best_regression_model_name]

print(f'Best performing model: {best_regression_model_name}')
print(f"Test RMSE: {regression_summary.loc[best_model_idx, 'RMSE']:.4f}")
print(f"Test R2: {regression_summary.loc[best_model_idx, 'R2']:.4f}")
print(f"Test MAE: {regression_summary.loc[best_model_idx, 'MAE']:.4f}")
print(f"MedianAE: {regression_summary.loc[best_model_idx, 'MedianAE']:.4f}")
print(f"Explained Var: {regression_summary.loc[best_model_idx, 'Explained Var']:.4f}")
print(f"MAPE: {regression_summary.loc[best_model_idx, 'MAPE']:.4f}" if np.isfinite(regression_summary.loc[best_model_idx, 'MAPE']) else 'MAPE: undefined because the target contains zeros.')

plot_top_feature_importance(
    best_regression_model,
    reg_feature_names,
    top_n=15,
    title=f'Regression feature importance: {best_regression_model_name}',
)

y_pred_best_reg = best_regression_model.predict(X_test_reg)
plot_regression_diagnostics(
    y_test_reg,
    y_pred_best_reg,
    title_prefix=best_regression_model_name,
    target_label=regression_target_label,
)

# %% [markdown]
# ## 2.9 Group B Debrief Prompts
# 
# Group B should be ready to answer:
# - How unstable was one tree across repeated splits?
# - Did Random Forest mainly improve accuracy, stability, or both?
# - Which boosting library would you shortlist for a future regression project, and why?
# - Which regression metric changed the most across models?
# - Did the residual plots support the same conclusion as RMSE alone?

# %% [markdown]
# ---
# # Whole-Class Debrief
# 
# Use the final discussion to compare what changed across task types.

# %% [markdown]
# ## Debrief Questions
# 
# - Did bagging help in both tracks for the same reason?
# - Did the best boosting library stay the same across classification and regression?
# - Where did ensembles give the clearest benefit over one decision tree?
# - Where did ensembles become harder to explain or harder to maintain?
# - If you had to ship one model tomorrow, what extra validation would you want first?

# %% [markdown]
# ## Stacking Takeaway
# 
# Stacking is part of the ensemble landscape, but it is not part of the core coding exercise today.
# The main reason is methodological: proper stacking needs out-of-fold meta-features, otherwise the meta-model can leak training information.
# 
# That makes stacking a good topic for a teacher-led discussion or a follow-up notebook, not a first 90-minute classroom implementation.
