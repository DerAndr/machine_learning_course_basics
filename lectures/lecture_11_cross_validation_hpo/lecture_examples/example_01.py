# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # SECTION 0: SETUP - Installations and Imports

# %%
#############################################################
# SECTION 0: SETUP - Installations and Imports
#############################################################

# Install necessary libraries
# NOTE: notebook magic commented for local script use: !pip install scikit-learn==1.0.2 --quiet
# NOTE: notebook magic commented for local script use: !pip install h2o --quiet

# Import standard libraries
import numpy as np
import pandas as pd

# Import scikit-learn modules
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    TimeSeriesSplit,
    RepeatedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif

# Import h2o for AutoML
import h2o
from h2o.automl import H2OAutoML

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # SECTION 1: CLASSIFICATION DEMO (Wine dataset)

# %%
#############################################################
# SECTION 1: CLASSIFICATION DEMO (Wine dataset)
#############################################################

print("============================================================")
print("SECTION 1: CLASSIFICATION DEMO (Wine Dataset)")
print("============================================================\n")

from sklearn.datasets import load_wine

# 1.1 Load the Wine dataset
wine_data = load_wine()
X_wine = wine_data.data
y_wine = wine_data.target
print("Wine dataset shape:", X_wine.shape)
print("Class distribution:", np.bincount(y_wine))

# %%
# 1.2 Split Data into Training and Test Sets
print("\n--- DATA SPLITTING ---")

# Stratified splitting to maintain class distribution
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
)
print("Training set shape:", X_train_wine.shape)
print("Test set shape:", X_test_wine.shape)
print("Training set class distribution:", np.bincount(y_train_wine))
print("Test set class distribution:", np.bincount(y_test_wine))

# %%
from time import time
# 1.3 Cross-Validation on Training Set
print("\n--- CROSS-VALIDATION DEMO (Classification) ---")

# 1.3.1 K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
clf_kf = RandomForestClassifier(random_state=42)
t0 = time()
scores_kfold = cross_val_score(clf_kf, X_train_wine, y_train_wine, cv=kf, scoring='accuracy')
t1 = time()
print("K-Fold CV (5 splits) accuracy scores:", scores_kfold)
print("Mean K-Fold CV accuracy:", np.mean(scores_kfold), "Time taken:", t1 - t0)

# 1.3.2 Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_skf = RandomForestClassifier(random_state=42)
t0 = time()
scores_skfold = cross_val_score(clf_skf, X_train_wine, y_train_wine, cv=skf, scoring='accuracy')
t1 = time()
print("\nStratified K-Fold CV (5 splits) accuracy scores:", scores_skfold)
print("Mean Stratified K-Fold CV accuracy:", np.mean(scores_skfold), "Time taken:", t1 - t0)

# 1.3.3 Leave-One-Out Cross-Validation
loocv = LeaveOneOut()
clf_loocv = RandomForestClassifier(random_state=42)
t0 = time()
scores_loocv = cross_val_score(clf_loocv, X_train_wine, y_train_wine, cv=loocv, scoring='accuracy')
t1 = time()
print("\nLeave-One-Out CV: Number of evaluations =", len(scores_loocv))
print("Mean Leave-One-Out CV accuracy:", np.mean(scores_loocv), "Time taken:", t1 - t0)

# 1.3.4 Time Series Split (For Demonstration)
tscv = TimeSeriesSplit(n_splits=3)
clf_tscv = RandomForestClassifier(random_state=42)
t0 = time()
scores_tscv = cross_val_score(clf_tscv, X_train_wine, y_train_wine, cv=tscv, scoring='accuracy')
t1 = time()
print("\nTimeSeriesSplit (3 splits) accuracy scores:", scores_tscv)
print("Mean TimeSeriesSplit CV accuracy:", np.mean(scores_tscv), "Time taken:", t1 - t0)

# 1.3.5 Repeated K-Fold Cross-Validation
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
clf_rkf = RandomForestClassifier(random_state=42)
t0 = time()
scores_rkf = cross_val_score(clf_rkf, X_train_wine, y_train_wine, cv=rkf, scoring='accuracy')
t1 = time()
print("\nRepeated K-Fold (5 splits x2 repeats) accuracy scores:", scores_rkf)
print("Mean Repeated K-Fold CV accuracy:", np.mean(scores_rkf), f"Time taken:", t1 - t0)

print("""
[COMMENT] For classification tasks like the Wine dataset, StratifiedKFold is preferred
because it preserves the class distribution in each fold. RepeatedKFold offers more
robust estimates but requires more computational resources.
""")

# %%
# 1.4 Pipelines and Hyperparameter Tuning
print("\n--- PIPELINES DEMO (Classification) ---")

# Construct a machine learning pipeline
pipeline_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=8)),
    ('clf', RandomForestClassifier(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid_clf = {
    'select__k': [5, 8, 10],
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 5]
}

# Initialize GridSearchCV with StratifiedKFold
grid_cv_clf = GridSearchCV(
    estimator=pipeline_clf,
    param_grid=param_grid_clf,
    scoring='accuracy',
    cv=skf,  # Using StratifiedKFold defined earlier
    n_jobs=-1
)

# Perform grid search on the training data
t0 = time()
grid_cv_clf.fit(X_train_wine, y_train_wine)
t1 = time()
print("Best parameters (Pipeline, Wine):", grid_cv_clf.best_params_)
print("Best cross-validated accuracy on Training Set:", grid_cv_clf.best_score_, "Time taken:", t1 - t0)

# Evaluate the best pipeline on the test set
best_pipeline_clf = grid_cv_clf.best_estimator_
best_pipeline_clf.fit(X_train_wine, y_train_wine)
y_pred_clf = best_pipeline_clf.predict(X_test_wine)
accuracy_clf = accuracy_score(y_test_wine, y_pred_clf)

print("Pipeline final hold-out accuracy on Test Set:", accuracy_clf)

# %% [markdown]
# # SECTION 2: REGRESSION DEMO (California Housing dataset)

# %%
#############################################################
# SECTION 2: REGRESSION DEMO (California Housing dataset)
#############################################################

print("\n\n============================================================")
print("SECTION 2: REGRESSION DEMO (California Housing)")
print("============================================================\n")

from sklearn.datasets import fetch_california_housing

# 2.1 Load the California Housing dataset
cal_data = fetch_california_housing()
X_cal = cal_data.data
y_cal = cal_data.target

print("California Housing dataset shape:", X_cal.shape)
print("Target (Median House Value) range:", (y_cal.min(), y_cal.max()))

# %%
# 2.2 Split Data into Training and Test Sets
print("\n--- DATA SPLITTING ---")

X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(
    X_cal, y_cal, test_size=0.3, random_state=42
)
print("Training set shape:", X_train_cal.shape)
print("Test set shape:", X_test_cal.shape)
print("Training target range:", (y_train_cal.min(), y_train_cal.max()))
print("Test target range:", (y_test_cal.min(), y_test_cal.max()))

# %%
# 2.3 Cross-Validation on Training Set
print("\n--- CROSS-VALIDATION DEMO (Regression) ---")

# 2.3.1 K-Fold Cross-Validation
kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
regr_kf = RandomForestRegressor(random_state=42)
t0 = time()
scores_kfold_reg = cross_val_score(
    regr_kf, X_train_cal, y_train_cal, cv=kf_reg,
    scoring='neg_mean_squared_error'
)
rmse_kfold_reg = np.sqrt(-scores_kfold_reg)
t1 = time()
print("K-Fold CV (5 splits) RMSE scores:", rmse_kfold_reg)
print("Mean K-Fold CV RMSE:", np.mean(rmse_kfold_reg), "Time taken:", t1 - t0)

# 2.3.2 Repeated K-Fold Cross-Validation
rkf_reg = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
regr_rkf = RandomForestRegressor(random_state=42)
t0 = time()
scores_rkf_reg = cross_val_score(
    regr_rkf, X_train_cal, y_train_cal, cv=rkf_reg,
    scoring='neg_mean_squared_error'
)

rmse_rkf_reg = np.sqrt(-scores_rkf_reg)
t1 = time()
print("\nRepeated K-Fold (5 splits x2 repeats) RMSE scores:", rmse_rkf_reg)
print("Mean Repeated K-Fold CV RMSE:", np.mean(rmse_rkf_reg), "Time taken:", t1 - t0)

print("""
[COMMENT] For regression tasks, K-Fold and Repeated K-Fold are commonly used.
TimeSeriesSplit is reserved for temporal data to maintain chronological order.
""")

# %%
# 2.4 Pipelines and Hyperparameter Tuningё
print("\n--- PIPELINES DEMO (Regression) ---")

# Construct a machine learning pipeline
pipeline_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('regr', RandomForestRegressor(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid_reg = {
    'regr__n_estimators': [50, 100],
    'regr__max_depth': [None, 5],
}

# Initialize GridSearchCV with K-Fold Cross-Validation
grid_cv_reg = GridSearchCV(
    estimator=pipeline_reg,
    param_grid=param_grid_reg,
    scoring='neg_mean_squared_error',  # Negative MSE for compatibility
    cv=kf_reg,
    n_jobs=-1
)

# Perform grid search on the training data
t0 = time()
grid_cv_reg.fit(X_train_cal, y_train_cal)
t1 = time()
print("Best parameters (Pipeline, Regression):", grid_cv_reg.best_params_)
print("Best cross-validated negative MSE on Training Set:", grid_cv_reg.best_score_, "Time taken:", t1 - t0)

# Evaluate the best pipeline on the test set
best_pipeline_reg = grid_cv_reg.best_estimator_
best_pipeline_reg.fit(X_train_cal, y_train_cal)
y_pred_reg = best_pipeline_reg.predict(X_test_cal)
rmse_reg = np.sqrt(mean_squared_error(y_test_cal, y_pred_reg))
print("Pipeline final hold-out RMSE on Test Set:", rmse_reg)

# %%
# 2.5 AutoML with h2o
print("\n--- AUTO ML DEMO (Regression) ---")

# Initialize h2o
h2o.init()

# Prepare training and test data for h2o
df_train_cal = pd.DataFrame(X_train_cal, columns=cal_data.feature_names)
df_train_cal['target'] = y_train_cal

df_test_cal = pd.DataFrame(X_test_cal, columns=cal_data.feature_names)
df_test_cal['target'] = y_test_cal

train_h2o_cal = h2o.H2OFrame(df_train_cal)
test_h2o_cal = h2o.H2OFrame(df_test_cal)

# Define predictors and response
predictors_cal = list(cal_data.feature_names)
target_cal = 'target'

# Initialize and run AutoML
aml_reg = H2OAutoML(
    max_runtime_secs=60,  # Adjust as needed
    seed=42,
    sort_metric='rmse',   # Optimize for RMSE
    project_name='CaliforniaHousing_AutoML'
)
t0 = time()
aml_reg.train(x=predictors_cal, y=target_cal, training_frame=train_h2o_cal)
t1 = time()
print("Time taken:", t1 - t0)
# Display the AutoML leaderboard
lb_reg = aml_reg.leaderboard
print("\nAutoML Leaderboard (California Housing Regression):")
print(lb_reg.head(rows=lb_reg.nrows))

# Evaluate the best model on the test set
leader_reg = aml_reg.leader
perf_reg = leader_reg.model_performance(test_data=test_h2o_cal)
print("\nBest Model Test Performance (California Housing Regression):")
print(perf_reg)

# Shutdown h2o to free resources
h2o.shutdown(prompt=False)
