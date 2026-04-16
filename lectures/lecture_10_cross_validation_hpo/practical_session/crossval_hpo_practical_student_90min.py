# %% [markdown]
# # Cross-Validation and Hyperparameter Optimization: Practical Session - STUDENT VERSION (90 minutes)
# 
# **Learning objectives:**
# - show why naive evaluation can look better than the real out-of-sample performance;
# - compare `KFold` and `StratifiedKFold` under class imbalance;
# - use validation curves, randomized search, and adaptive tuning to tune a model more systematically;
# - see why nested CV is a stricter estimate than reporting the best inner-CV score;
# - keep AutoML visible as an optional extension, not a replacement for understanding validation design.

# %% [markdown]
# ## Setup
# 
# For local work in this repository, prefer:
# 
# ```bash
# uv sync --group hpo_automl
# uv run python tools/check_notebook_environment.py --group hpo_automl
# ```
# 
# In Colab, the setup cell below will install missing optional packages automatically.
# Locally, it only prints the matching `uv sync --group hpo_automl` guidance.
# 
# The main practical path uses only `scikit-learn` plus the baseline stack.
# The heavier tuning and AutoML blocks are optional and use `optuna`, `h2o`, and `flaml`.

# %%
import importlib.util
import os

IN_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ

optional_packages = ['optuna', 'h2o', 'flaml']
package_status = {pkg: importlib.util.find_spec(pkg) is not None for pkg in optional_packages}

if IN_COLAB:
    missing = [pkg for pkg, available in package_status.items() if not available]
    if missing:
        print(f'Colab detected: installing {", ".join(missing)}')
        import subprocess
        import sys

        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
        package_status = {pkg: importlib.util.find_spec(pkg) is not None for pkg in optional_packages}
    else:
        print('Colab detected: all optional packages are already available.')
else:
    for pkg in optional_packages:
        status = 'available' if package_status[pkg] else 'missing'
        print(f'Optional package {pkg}: {status}')

    if not package_status['h2o']:
        print('Install H2O only if you want the H2O AutoML extension:')
        print('  uv sync --group hpo_automl')

    if not package_status['optuna']:
        print('Install Optuna only if you want the Optuna tuning extension:')
        print('  uv sync --group hpo_automl')

    if not package_status['flaml']:
        print('Install FLAML only if you want the FLAML AutoML extension:')
        print('  uv sync --group hpo_automl')

# %% [markdown]
# ## Imports

# %%
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from scipy.stats import randint
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
RANDOM_STATE = 42

# %% [markdown]
# ## Shared Helper Functions

# %%
def load_phishing_websites():
    dataset = fetch_openml(data_id=4534, as_frame=True, parser='auto')
    df = dataset.frame.copy()

    target_col = 'Result' if 'Result' in df.columns else df.columns[-1]
    y_raw = df[target_col].astype(str)

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_raw), name='target', index=df.index)
    class_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    X = df.drop(columns=[target_col]).apply(pd.to_numeric, errors='coerce')
    return X, y, class_mapping


def plot_class_balance(y, class_mapping, title='Class distribution'):
    counts = y.value_counts().sort_index()
    labels = [class_mapping.get(idx, str(idx)) for idx in counts.index]

    plt.figure(figsize=(7, 4))
    colors = sns.color_palette('Set2', n_colors=len(counts))
    plt.bar(labels, counts.values, color=colors, edgecolor='black', linewidth=0.8)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_missing_summary(X):
    missing_counts = X.isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print('No missing values were detected in this dataset.')
        return

    plt.figure(figsize=(8, 4))
    plt.bar(missing_counts.index, missing_counts.values, color='steelblue')
    plt.title('Missing values by feature')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_top_correlation_heatmap(X, top_n=12):
    variance_ranking = X.var().sort_values(ascending=False)
    top_cols = variance_ranking.head(top_n).index.tolist()
    corr = X[top_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0,
        cbar_kws={'shrink': 0.85},
    )
    plt.title(f'Triangular correlation heatmap for the {top_n} highest-variance features')
    plt.tight_layout()
    plt.show()
    return corr


def plot_confusion(cm, labels, title='Confusion matrix'):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_score_distribution(score_map, ylabel='Score', title='Cross-validation score comparison'):
    records = []
    for label, scores in score_map.items():
        for fold_idx, score in enumerate(scores, start=1):
            records.append({'model': label, 'fold': fold_idx, 'score': score})

    score_df = pd.DataFrame(records)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=score_df, x='model', y='score', color='white')
    sns.stripplot(data=score_df, x='model', y='score', hue='model', dodge=False, size=7, alpha=0.75)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('')
    if plt.gca().legend_ is not None:
        plt.gca().legend_.remove()
    plt.tight_layout()
    plt.show()


def plot_fold_lines(score_map, ylabel='Score', title='Fold-by-fold comparison'):
    plt.figure(figsize=(8, 5))
    for label, scores in score_map.items():
        plt.plot(range(1, len(scores) + 1), scores, marker='o', linewidth=2, label=label)
    plt.xticks(range(1, len(next(iter(score_map.values()))) + 1))
    plt.xlabel('Fold')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_validation_curve_summary(param_range, train_scores, test_scores, param_name='Parameter'):
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_mean, marker='o', color='darkorange', label='Train score')
    plt.plot(param_range, test_mean, marker='s', color='navy', label='CV score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color='darkorange', alpha=0.15)
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color='navy', alpha=0.15)
    plt.xlabel(param_name)
    plt.ylabel('ROC AUC')
    plt.title(f'Validation curve: {param_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_random_search_results(cv_results, top_n=12):
    results = pd.DataFrame(cv_results).sort_values('rank_test_score').head(top_n).copy()
    results['label'] = [f"rank {rank}" for rank in results['rank_test_score']]

    plt.figure(figsize=(9, 5))
    plt.barh(results['label'], results['mean_test_score'], xerr=results['std_test_score'], color='slateblue', alpha=0.85)
    plt.gca().invert_yaxis()
    plt.xlabel('Mean CV ROC AUC')
    plt.title(f'Top {top_n} search configurations')
    plt.tight_layout()
    plt.show()


def plot_optuna_history(trials_df):
    complete = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    if complete.empty:
        print('No completed Optuna trials to visualize.')
        return

    complete = complete.sort_values('number').reset_index(drop=True)
    complete['best_so_far'] = complete['value'].cummax()

    plt.figure(figsize=(8, 5))
    plt.plot(complete['number'], complete['value'], marker='o', linewidth=1.5, alpha=0.8, label='trial score')
    plt.plot(complete['number'], complete['best_so_far'], marker='s', linewidth=2.5, label='best so far')
    plt.xlabel('Trial')
    plt.ylabel('Mean CV ROC AUC')
    plt.title('Optuna optimization history')
    plt.legend()
    plt.tight_layout()
    plt.show()

    param_cols = [col for col in ['params_n_estimators', 'params_max_depth'] if col in complete.columns]
    if len(param_cols) == 2:
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(
            complete['params_max_depth'],
            complete['params_n_estimators'],
            c=complete['value'],
            cmap='viridis',
            s=140,
            edgecolor='black',
        )
        plt.xlabel('max_depth')
        plt.ylabel('n_estimators')
        plt.title('Optuna search path')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Mean CV ROC AUC')
        plt.tight_layout()
        plt.show()


def plot_nested_vs_non_nested(non_nested_scores, nested_scores):
    comparison_df = pd.DataFrame(
        {
            'seed': [7, 21, 42],
            'non_nested_cv': non_nested_scores,
            'nested_cv': nested_scores,
        }
    )
    long_df = comparison_df.melt(id_vars='seed', var_name='estimate', value_name='roc_auc')

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=long_df, x='estimate', y='roc_auc', color='white')
    sns.stripplot(data=long_df, x='estimate', y='roc_auc', hue='estimate', dodge=False, size=8)
    if plt.gca().legend_ is not None:
        plt.gca().legend_.remove()
    plt.title('Nested vs non-nested CV estimates')
    plt.xlabel('')
    plt.ylabel('ROC AUC')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for _, row in comparison_df.iterrows():
        plt.plot(['non_nested_cv', 'nested_cv'], [row['non_nested_cv'], row['nested_cv']], marker='o', linewidth=2)
    plt.ylabel('ROC AUC')
    plt.title('Optimism gap by random seed')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## How To Work In Teams
# 
# If you are working in pairs or small groups, a clean split is:
# 
# - Group A: Sections 1 and 2
# - Group B: Sections 3 and 4
# - Group C: Sections 5 and 6
# 
# Then compare:
# 
# - which evaluation mistake looked most convincing at first;
# - which metric changed the story the most;
# - how much optimism remained after tuning.

# %% [markdown]
# ## 1. Dataset: Phishing Websites
# 
# We will work with the **Phishing Websites** classification dataset from OpenML.
# It is a good fit for this lecture because:
# 
# - the target is binary and easy to evaluate;
# - the feature matrix is tabular and fast enough for repeated CV;
# - we can demonstrate class imbalance, leakage, and tuning without heavy preprocessing.
# 
# This practical keeps preprocessing deliberately light so that the main focus stays on validation design.

# %%
X, y, class_mapping = load_phishing_websites()

print(f'Dataset shape: {X.shape}')
print('Class mapping:', class_mapping)
print('\nClass distribution (counts):')
display(y.value_counts().sort_index().rename(index=class_mapping).to_frame('count'))
print('\nFeature preview:')
display(X.head())

plot_class_balance(y, class_mapping, title='Phishing Websites: class distribution')
plot_missing_summary(X)
correlation_matrix = plot_top_correlation_heatmap(X, top_n=12)

# %% [markdown]
# ### 1.1 Quick Audit Note
# 
# Most features are already encoded into a small numeric space.
# That is why this notebook can focus on **evaluation design** instead of spending most of the time on preprocessing.
# In a real project, the same CV logic would need to wrap any feature engineering and imputation steps as well.

# %% [markdown]
# ## 2. Pitfall 1: The Accuracy Paradox
# 
# A classifier can look good on **accuracy** while still being weak on the class you actually care about.
# 
# To make that visible, we will create a much more imbalanced subset and compare:
# 
# - accuracy,
# - balanced accuracy,
# - minority-class recall.

# %%
class_counts = y.value_counts()
majority_class = class_counts.idxmax()
minority_class = class_counts.idxmin()

# TODO:
# Keep all majority-class rows, but only a small fraction of minority-class rows.
minority_fraction = ...
minority_index = y[y == minority_class].sample(frac=minority_fraction, random_state=RANDOM_STATE).index
selected_index = y[y == majority_class].index.union(minority_index)

X_imbal = X.loc[selected_index].copy()
y_imbal = y.loc[selected_index].copy()

X_train_imbal, X_test_imbal, y_train_imbal, y_test_imbal = train_test_split(
    X_imbal,
    y_imbal,
    test_size=0.3,
    stratify=y_imbal,
    random_state=RANDOM_STATE,
)

rf_imbal = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE)
rf_imbal.fit(X_train_imbal, y_train_imbal)
y_pred_imbal = rf_imbal.predict(X_test_imbal)

# TODO:
# Compute three metrics that tell different stories on imbalanced data.
accuracy_imbal = ...
balanced_accuracy_imbal = ...
minority_recall_imbal = ...

print('Imbalanced class distribution:')
display(y_imbal.value_counts(normalize=True).sort_index().rename(index=class_mapping).to_frame('share'))

print('\nMetrics on the imbalanced subset:')
print(f'Accuracy: {accuracy_imbal:.3f}')
print(f'Balanced accuracy: {balanced_accuracy_imbal:.3f}')
print(f'Minority-class recall: {minority_recall_imbal:.3f}')

# %%
imbalance_labels = [class_mapping.get(idx, str(idx)) for idx in y_imbal.value_counts().sort_index().index]
plot_class_balance(y_imbal, class_mapping, title='Imbalanced subset: class distribution')

cm_imbal = confusion_matrix(y_test_imbal, y_pred_imbal)
plot_confusion(cm_imbal, imbalance_labels, title='Imbalanced subset: confusion matrix')

metric_table = pd.DataFrame(
    {
        'metric': ['accuracy', 'balanced_accuracy', 'minority_recall'],
        'value': [accuracy_imbal, balanced_accuracy_imbal, minority_recall_imbal],
    }
)

plt.figure(figsize=(7, 4))
plt.bar(metric_table['metric'], metric_table['value'], color=['steelblue', 'darkorange', 'firebrick'])
plt.ylim(0, 1.05)
plt.title('Accuracy can hide what happens to the minority class')
plt.ylabel('Metric value')
plt.tight_layout()
plt.show()

display(metric_table.round(3))

# %% [markdown]
# ### Real-World Note
# 
# If the minority class is the costly case, a high overall accuracy may still be unacceptable.
# That is why evaluation should always match the business risk, not just the easiest aggregate metric.

# %% [markdown]
# ## 3. Pitfall 2: Data Leakage
# 
# Leakage means the model gets access to information that would not be available at prediction time.
# 
# To make the effect obvious, we add one obviously suspicious feature and compare CV scores:
# 
# - the clean feature matrix;
# - the same matrix plus a synthetic **golden feature** that is almost the target itself.

# %%
cv_clean = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

X_leaky = X.copy()
noise = np.random.normal(0, 0.05, size=len(y))

# TODO:
# Build a suspicious feature that is almost the target itself.
X_leaky['golden_feature'] = ...

clean_scores = cross_val_score(
    RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE),
    X,
    y,
    cv=cv_clean,
    scoring='roc_auc',
    n_jobs=-1,
)

leaky_scores = cross_val_score(
    RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE),
    X_leaky,
    y,
    cv=cv_clean,
    scoring='roc_auc',
    n_jobs=-1,
)

print(f'Clean CV ROC AUC: {clean_scores.mean():.3f} ± {clean_scores.std():.3f}')
print(f'Leaky CV ROC AUC: {leaky_scores.mean():.3f} ± {leaky_scores.std():.3f}')

# %%
plot_score_distribution(
    {'Clean features': clean_scores, 'With golden feature': leaky_scores},
    ylabel='ROC AUC',
    title='Leakage can make cross-validation look unrealistically strong',
)
plot_fold_lines(
    {'Clean features': clean_scores, 'With golden feature': leaky_scores},
    ylabel='ROC AUC',
    title='Fold-by-fold view: clean vs leaky feature space',
)

summary_table = pd.DataFrame(
    {
        'setting': ['clean', 'leaky'],
        'mean_roc_auc': [clean_scores.mean(), leaky_scores.mean()],
        'std_roc_auc': [clean_scores.std(), leaky_scores.std()],
    }
)
display(summary_table.round(3))

# %% [markdown]
# ### Teaching Note
# 
# This is an intentionally exaggerated example.
# Real leakage is often harder to see because it hides inside feature engineering, aggregation windows, preprocessing fitted on the full dataset, or target-aware imputations.

# %% [markdown]
# ## 4. Robust Evaluation: K-Fold vs Stratified K-Fold
# 
# For binary classification, `StratifiedKFold` is usually the safer default because each fold keeps roughly the same class ratio.
# 
# We will compare the two strategies on the imbalanced subset using **balanced accuracy**.

# %%
base_model = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE)

# TODO:
# Compare a naive fold strategy with a stratified one.
kfold = ...
stratified_kfold = ...

kfold_scores = cross_val_score(
    base_model,
    X_imbal,
    y_imbal,
    cv=kfold,
    scoring='balanced_accuracy',
    n_jobs=-1,
)

stratified_scores = cross_val_score(
    base_model,
    X_imbal,
    y_imbal,
    cv=stratified_kfold,
    scoring='balanced_accuracy',
    n_jobs=-1,
)

print(f'KFold balanced accuracy: {kfold_scores.mean():.3f} ± {kfold_scores.std():.3f}')
print(f'StratifiedKFold balanced accuracy: {stratified_scores.mean():.3f} ± {stratified_scores.std():.3f}')

# %%
plot_score_distribution(
    {'KFold': kfold_scores, 'StratifiedKFold': stratified_scores},
    ylabel='Balanced accuracy',
    title='Balanced accuracy across folds',
)
plot_fold_lines(
    {'KFold': kfold_scores, 'StratifiedKFold': stratified_scores},
    ylabel='Balanced accuracy',
    title='Fold-by-fold stability on the imbalanced subset',
)

# %% [markdown]
# ### 4.1 Validation Curve
# 
# A validation curve helps us see where a hyperparameter begins to overfit.
# Here we inspect `max_depth` for a random forest on the full dataset.

# %%
print('Runtime note: a validation curve repeats model fitting over many parameter values and folds.')

# %%
depth_grid = np.arange(2, 21, 2)
validation_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

t0 = perf_counter()
train_scores, test_scores = validation_curve(
    RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE),
    X,
    y,
    param_name='max_depth',
    param_range=depth_grid,
    cv=validation_cv,
    scoring='roc_auc',
    n_jobs=-1,
)
validation_elapsed = perf_counter() - t0

plot_validation_curve_summary(depth_grid, train_scores, test_scores, param_name='max_depth')
print(f'Validation-curve runtime: {validation_elapsed:.1f}s')

# %% [markdown]
# ### 4.2 Two More Cross-Validation Designs
# 
# The lecture also discusses strategies beyond one ordinary `KFold` run.
# Two useful additions are:
# 
# - `RepeatedStratifiedKFold`, which gives a more stable estimate by averaging over multiple random fold assignments;
# - `LeaveOneOut`, which is conceptually simple but quickly becomes expensive.

# %%
print('Runtime note: the next repeated-CV block fits the model many more times than one ordinary 5-fold run.')

# %%
repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
t0 = perf_counter()
repeated_scores = cross_val_score(
    base_model,
    X_imbal,
    y_imbal,
    cv=repeated_cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
)
repeated_elapsed = perf_counter() - t0

print(f'RepeatedStratifiedKFold balanced accuracy: {repeated_scores.mean():.3f} ± {repeated_scores.std():.3f}')
print(f'Number of validation runs: {len(repeated_scores)}')
print(f'Runtime: {repeated_elapsed:.1f}s')

plot_score_distribution(
    {
        'StratifiedKFold': stratified_scores,
        'RepeatedStratifiedKFold': repeated_scores,
    },
    ylabel='Balanced accuracy',
    title='Repeated stratification gives a more stable view than one single split set',
)

# %% [markdown]
# `LeaveOneOut` is easier to understand than to use efficiently.
# To keep runtime reasonable, we only demonstrate it on a small subset and with a simpler model.
# We also switch to **accuracy** here: with LOOCV, each validation fold contains exactly one sample,
# so ROC AUC is not defined fold-by-fold.

# %%
print('Runtime note: LOOCV scales with the number of rows, so we keep this demo on a smaller subset.')

# %%
loocv_sample_idx = y.groupby(y, group_keys=False).apply(
    lambda part: part.sample(n=min(len(part), 80), random_state=RANDOM_STATE)
).index
X_loocv = X.loc[loocv_sample_idx].copy()
y_loocv = y.loc[loocv_sample_idx].copy()

loocv_model = LogisticRegression(max_iter=2000)
loocv = LeaveOneOut()
t0 = perf_counter()
loocv_scores = cross_val_score(
    loocv_model,
    X_loocv,
    y_loocv,
    cv=loocv,
    scoring='accuracy',
    n_jobs=-1,
)
loocv_elapsed = perf_counter() - t0

loocv_comparison_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
t0 = perf_counter()
loocv_baseline_scores = cross_val_score(
    loocv_model,
    X_loocv,
    y_loocv,
    cv=loocv_comparison_cv,
    scoring='accuracy',
    n_jobs=-1,
)
baseline_elapsed = perf_counter() - t0

comparison_small = pd.DataFrame(
    {
        'strategy': ['LOOCV', 'StratifiedKFold (5 folds)'],
        'mean_accuracy': [loocv_scores.mean(), loocv_baseline_scores.mean()],
        'std_accuracy': [loocv_scores.std(), loocv_baseline_scores.std()],
        'num_fits': [len(loocv_scores), len(loocv_baseline_scores)],
    }
)
display(comparison_small.round(3))
print(f'LOOCV runtime: {loocv_elapsed:.1f}s')
print(f'5-fold stratified runtime on the same subset: {baseline_elapsed:.1f}s')

plt.figure(figsize=(8, 4))
plt.bar(comparison_small['strategy'], comparison_small['num_fits'], color=['firebrick', 'steelblue'])
plt.ylabel('Number of model fits')
plt.title('LOOCV is much more expensive even on a small subset')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Scope Note
# 
# This practical stays in the binary-classification setting on purpose.
# 
# - For regression, you would keep the same train/test discipline but use a different split strategy than `StratifiedKFold`.
# - For time series, random folds are not valid and the split must respect temporal order.
# 
# Those cases belong to the same lecture family, but not to this classroom workflow.

# %% [markdown]
# ## 5. Hyperparameter Optimization: Randomized Search and Optuna
# 
# We will now tune a random forest on a proper train split and keep the final test split untouched until the end.
# 
# Even though this dataset does not require much preprocessing, we still wrap the model in a `Pipeline`.
# That way the same evaluation pattern would remain safe if we later inserted preprocessing steps.

# %%
print('Runtime note: randomized search is one of the slower blocks because it repeats CV across many settings.')

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f'Train shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')

# %%
rf_pipeline = Pipeline(
    [
        ('model', RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)

param_dist = {
    'model__n_estimators': randint(60, 220),
    'model__max_depth': randint(3, 24),
    'model__min_samples_split': randint(2, 12),
    'model__min_samples_leaf': randint(1, 6),
    'model__max_features': ['sqrt', 'log2', None],
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# TODO:
# Configure a randomized search that maximizes ROC AUC.
random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_dist,
    n_iter=...,
    scoring='...',
    cv=...,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)

t0 = perf_counter()
random_search.fit(X_train, y_train)
search_elapsed = perf_counter() - t0

print('Best parameters:')
display(random_search.best_params_)
print(f"Best inner-CV ROC AUC: {random_search.best_score_:.3f}")
print(f'Randomized-search runtime: {search_elapsed:.1f}s')

# %%
plot_random_search_results(random_search.cv_results_, top_n=10)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

test_metrics = pd.DataFrame(
    {
        'metric': ['accuracy', 'balanced_accuracy', 'roc_auc'],
        'value': [
            accuracy_score(y_test, y_pred),
            balanced_accuracy_score(y_test, y_pred),
            roc_auc_score(y_test, y_proba),
        ],
    }
)

display(test_metrics.round(3))
plot_confusion(
    confusion_matrix(y_test, y_pred),
    [class_mapping[idx] for idx in sorted(class_mapping)],
    title='Hold-out confusion matrix for the tuned model',
)

plt.figure(figsize=(6, 4))
plt.bar(['Best inner CV ROC AUC', 'Hold-out ROC AUC'], [random_search.best_score_, roc_auc_score(y_test, y_proba)], color=['slateblue', 'darkgreen'])
plt.ylim(0, 1.05)
plt.title('Inner-CV winner vs untouched hold-out score')
plt.ylabel('ROC AUC')
plt.tight_layout()
plt.show()

print('Classification report on the hold-out set:')
print(classification_report(y_test, y_pred, target_names=[class_mapping[idx] for idx in sorted(class_mapping)]))

# %% [markdown]
# ### 5.1 What Randomized Search Gives You
# 
# The score inside `RandomizedSearchCV` is still an **inner-loop estimate**.
# It is good for model selection, but it is not the most conservative final estimate of generalization.
# It is still a very useful baseline because it gives you:
# 
# - a broad search over the space with a fixed compute budget;
# - one clean `scikit-learn` API;
# - a straightforward path to the best pipeline object.

# %% [markdown]
# ### 5.2 Optional Extension: Optuna
# 
# `RandomizedSearchCV` is a strong baseline, but the lecture also covers more adaptive search methods.
# `Optuna` is a good example because it can steer later trials toward more promising regions instead of sampling every trial independently.
# 
# To keep the comparison fair, we optimize the **same random-forest family** on the **same train split** and with the **same inner CV metric**.

# %%
print('Runtime note: Optuna is usually lighter than nested CV, but each trial still reruns cross-validation.')

# %%
try:
    import optuna
except ImportError:
    print('Optional dependency not installed: optuna')
    print('Install only if you want this extension: !pip install -q optuna')
else:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optuna_objective(trial):
        rf_trial = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 60, 220),
            max_depth=trial.suggest_int('max_depth', 3, 24),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 12),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 6),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        trial_pipeline = Pipeline([('model', rf_trial)])
        return cross_val_score(
            trial_pipeline,
            X_train,
            y_train,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
        ).mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    t0 = perf_counter()
    study.optimize(optuna_objective, n_trials=18, show_progress_bar=False)
    optuna_elapsed = perf_counter() - t0

    print(f'Best Optuna CV ROC AUC: {study.best_value:.3f}')
    print('Best Optuna parameters:')
    display(pd.Series(study.best_trial.params, name='value').to_frame())
    print(f'Optuna runtime: {optuna_elapsed:.1f}s')

    plot_optuna_history(study.trials_dataframe())

    optuna_best_model = Pipeline(
        [
            (
                'model',
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **study.best_trial.params,
                ),
            )
        ]
    )
    optuna_best_model.fit(X_train, y_train)
    optuna_pred = optuna_best_model.predict(X_test)
    optuna_proba = optuna_best_model.predict_proba(X_test)[:, 1]
    optuna_auc = roc_auc_score(y_test, optuna_proba)
    optuna_balanced_accuracy = balanced_accuracy_score(y_test, optuna_pred)

    hpo_comparison = pd.DataFrame(
        {
            'workflow': ['RandomizedSearchCV', 'Optuna'],
            'best_inner_cv_roc_auc': [random_search.best_score_, study.best_value],
            'holdout_roc_auc': [roc_auc_score(y_test, y_proba), optuna_auc],
            'holdout_balanced_accuracy': [
                balanced_accuracy_score(y_test, y_pred),
                optuna_balanced_accuracy,
            ],
        }
    )
    display(hpo_comparison.round(3))

    plt.figure(figsize=(7, 4))
    x = np.arange(len(hpo_comparison))
    width = 0.36
    plt.bar(x - width / 2, hpo_comparison['best_inner_cv_roc_auc'], width=width, label='inner CV')
    plt.bar(x + width / 2, hpo_comparison['holdout_roc_auc'], width=width, label='hold-out')
    plt.xticks(x, hpo_comparison['workflow'])
    plt.ylim(0, 1.05)
    plt.ylabel('ROC AUC')
    plt.title('Randomized search vs Optuna on the same tuning task')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Real-World Note
# 
# Randomized search is simple and dependable.
# Optuna adds a more adaptive search strategy on top of the same validation discipline.
# 
# Neither one replaces the need for:
# 
# - a correct split strategy,
# - a held-out test set,
# - and a stricter final estimate when you need one.
# 
# That is exactly why nested CV exists.

# %% [markdown]
# ## 6. Nested Cross-Validation and Optional AutoML
# 
# Nested CV separates:
# 
# - the **inner loop** used for tuning;
# - the **outer loop** used for evaluation.
# 
# This usually gives a slightly lower, but more honest, performance estimate than reporting only the best inner-CV score.

# %%
print('Runtime note: nested CV is intentionally expensive because each outer fold runs its own inner search.')

# %%
seed_values = [7, 21, 42]
non_nested_scores = []
nested_scores = []

t0 = perf_counter()
for seed in seed_values:
    inner_cv_loop = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    outer_cv_loop = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 100)

    search_loop = RandomizedSearchCV(
        estimator=Pipeline([('model', RandomForestClassifier(random_state=RANDOM_STATE))]),
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=inner_cv_loop,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )

    search_loop.fit(X, y)
    non_nested_scores.append(search_loop.best_score_)

    nested_score = cross_val_score(
        search_loop,
        X,
        y,
        cv=outer_cv_loop,
        scoring='roc_auc',
        n_jobs=-1,
    ).mean()
    nested_scores.append(nested_score)
nesting_elapsed = perf_counter() - t0

comparison_table = pd.DataFrame(
    {
        'seed': seed_values,
        'non_nested_cv': non_nested_scores,
        'nested_cv': nested_scores,
        'optimism_gap': np.array(non_nested_scores) - np.array(nested_scores),
    }
)

display(comparison_table.round(3))
plot_nested_vs_non_nested(non_nested_scores, nested_scores)
print(f'Nested-CV runtime: {nesting_elapsed:.1f}s')

# %% [markdown]
# ### 6.1 Optional Extension: H2O AutoML
# 
# This cell is optional.
# Use it only if `h2o` is available in your environment.
# 
# The goal is not to treat AutoML as magic.
# The point is to compare automated model search against the validation discipline you have already built manually.

# %%
print('Runtime note: H2O has startup overhead because it launches a separate runtime before the search begins.')

# %%
try:
    import h2o
    from h2o.automl import H2OAutoML
except ImportError:
    print('Optional dependency not installed: h2o')
    print('Install only if you want this extension: !pip install -q h2o')
else:
    t0 = perf_counter()
    h2o.init(max_mem_size='2G', nthreads=-1)

    train_frame = X_train.copy()
    train_frame['target'] = y_train.to_numpy()
    test_frame = X_test.copy()
    test_frame['target'] = y_test.to_numpy()

    train_h2o = h2o.H2OFrame(train_frame)
    test_h2o = h2o.H2OFrame(test_frame)
    train_h2o['target'] = train_h2o['target'].asfactor()
    test_h2o['target'] = test_h2o['target'].asfactor()

    aml = H2OAutoML(
        max_models=6,
        seed=RANDOM_STATE,
        max_runtime_secs=90,
        verbosity='warn',
    )
    aml.train(x=X_train.columns.tolist(), y='target', training_frame=train_h2o)

    leaderboard = aml.leaderboard.as_data_frame()
    display(leaderboard.head(10))

    perf = aml.leader.model_performance(test_h2o)
    print('Hold-out ROC AUC from the H2O leader:')
    print(round(perf.auc(), 3))
    print(f'H2O AutoML runtime: {perf_counter() - t0:.1f}s')

    h2o.shutdown(prompt=False)

# %% [markdown]
# ### 6.2 Optional Extension: FLAML
# 
# FLAML is a lightweight AutoML and tuning library from Microsoft.
# It is useful here because it gives a compact, time-budgeted search API while still fitting naturally into the same evaluation mindset.
# In this classroom version, we keep the estimator list to `scikit-learn` models so the extension stays lightweight and does not require extra booster packages.

# %%
print('Runtime note: FLAML is usually lighter to start than H2O, but the search still scales with the time budget.')

# %%
try:
    from flaml import AutoML
except ImportError:
    print('Optional dependency not installed: flaml')
    print('Install only if you want this extension: !pip install -q flaml')
else:
    flaml_settings = {
        'time_budget': 60,
        'metric': 'roc_auc',
        'task': 'classification',
        'estimator_list': ['lrl1', 'lrl2', 'rf', 'extra_tree'],
        'eval_method': 'cv',
        'n_splits': 5,
        'split_type': 'stratified',
        'log_file_name': 'flaml_lecture10.log',
        'verbose': 0,
    }

    t0 = perf_counter()
    flaml_automl = AutoML()
    flaml_automl.fit(X_train=X_train, y_train=y_train, **flaml_settings)
    flaml_elapsed = perf_counter() - t0

    print('Best FLAML estimator:', flaml_automl.best_estimator)
    print('Best FLAML config:')
    display(pd.Series(flaml_automl.best_config, name='value').to_frame())

    flaml_proba = flaml_automl.predict_proba(X_test)[:, 1]
    flaml_auc = roc_auc_score(y_test, flaml_proba)
    print(f'Hold-out ROC AUC from the FLAML leader: {flaml_auc:.3f}')
    print(f'FLAML runtime: {flaml_elapsed:.1f}s')

# %% [markdown]
# ## 7. Pipelines as Part of the Training Workflow
# 
# Cross-validation should not be thought of as something that happens **after** preprocessing.
# In a proper workflow, CV repeatedly trains the **whole pipeline**:
# 
# - feature selection,
# - scaling,
# - model fitting.
# 
# If we fit preprocessing once on all training rows and only then cross-validate the model, we have already leaked information across folds.

# %%
print('Runtime note: this final comparison is lighter than nested CV, but it still refits the workflow multiple times.')

# %%
pipeline_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

leaky_selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = leaky_selector.fit_transform(X_train, y_train)

leaky_scaler = StandardScaler()
X_train_selected_scaled = leaky_scaler.fit_transform(X_train_selected)

t0 = perf_counter()
leaky_pipeline_scores = cross_val_score(
    LogisticRegression(max_iter=2000),
    X_train_selected_scaled,
    y_train,
    cv=pipeline_cv,
    scoring='roc_auc',
    n_jobs=-1,
)

safe_pipeline = Pipeline(
    [
        ('select', SelectKBest(score_func=f_classif, k=10)),
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000)),
    ]
)
safe_pipeline_scores = cross_val_score(
    safe_pipeline,
    X_train,
    y_train,
    cv=pipeline_cv,
    scoring='roc_auc',
    n_jobs=-1,
)
pipeline_elapsed = perf_counter() - t0

pipeline_comparison = pd.DataFrame(
    {
        'workflow': ['Leaky prefit preprocessing', 'Safe pipeline inside CV'],
        'mean_roc_auc': [leaky_pipeline_scores.mean(), safe_pipeline_scores.mean()],
        'std_roc_auc': [leaky_pipeline_scores.std(), safe_pipeline_scores.std()],
    }
)
display(pipeline_comparison.round(3))

plot_score_distribution(
    {
        'Leaky preprocessing': leaky_pipeline_scores,
        'Pipeline inside CV': safe_pipeline_scores,
    },
    ylabel='ROC AUC',
    title='Cross-validation must wrap the full training pipeline',
)

safe_pipeline.fit(X_train, y_train)
pipeline_holdout_auc = roc_auc_score(y_test, safe_pipeline.predict_proba(X_test)[:, 1])
print(f'Safe pipeline hold-out ROC AUC: {pipeline_holdout_auc:.3f}')
print(f'Pipeline comparison runtime: {pipeline_elapsed:.1f}s')

# %% [markdown]
# ## 8. Wrap-Up
# 
# Before you move on, make sure you can explain:
# 
# 1. why accuracy looked misleading on the imbalanced subset;
# 2. why the golden feature made the CV score look suspiciously strong;
# 3. why `StratifiedKFold` is the safer default for this classification task;
# 4. why the best inner-CV score is not the same thing as a final unbiased estimate;
# 5. what AutoML can automate, and what it still cannot replace.
