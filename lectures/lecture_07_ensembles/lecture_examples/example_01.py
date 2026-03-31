# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Ensemble Models and Techniques with Hyperparameter Tuning, PR Curves, and Class Separation Diagrams

# %% [markdown]
# This notebook demonstrates various ensemble models on a real-world classification dataset with baseline and tuned versions.
# 
# We'll evaluate **Random Forest**, **XGBoost**, **CatBoost**, **LightGBM**, and a **Stacking Classifier** combining K-Nearest Neighbors, Ridge Classifier, and SVC.
# 
# Each model will be visualized.

# %% [markdown]
# ## Install Required Libraries

# %%
# NOTE: notebook magic commented for local script use: !pip install xgboost catboost lightgbm scikit-learn matplotlib -q

# %% [markdown]
# # Import Libraries

# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# %%
# connect to google drive
# NOTE: Colab-only import commented for local script use: from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# # Load the dataset

# %%
# We will use Telco Customer Churn Dataset https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Load the Telco Customer Churn dataset
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(data_path)

# Drop customer ID column as it's not relevant for prediction
df = df.drop(columns=['customerID'])

# Convert target column 'Churn' to binary (1 for Yes, 0 for No)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

df.describe(include='all')

# %%
print(df.info())

# %%
df.nunique()

# %%
df['Churn'].value_counts(normalize=True)

# %%
# Identify categorical columns based on unique value count (10 or fewer unique values)
unique_counts = df.nunique()
categorical_cols = unique_counts[unique_counts <= 5].index.tolist()
categorical_cols.remove('Churn')  # Remove target from categorical columns

# Identify numeric columns (those not identified as categorical)
numeric_cols = [col for col in df.columns if col not in categorical_cols and col != 'Churn']

# Convert columns that should be numeric to numeric values, setting errors='coerce' to replace non-numeric entries with NaN
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Encode categorical columns (not required for CatBoost, so we'll keep separate encodings)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# %%
categorical_cols

# %%
numeric_cols

# %%
# Split features and target for encoded dataset and non-encoded for CatBoost
X = df_encoded.drop('Churn', axis=1)
y = df['Churn']

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# Standardize numeric features only for models that need scaling
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# %%
# Display identified categorical and numeric columns
print("Categorical Columns (5 or fewer unique values):", categorical_cols)
print("Numeric Columns:", numeric_cols)

# %%
# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')  # For binary classification
    recall = recall_score(y_test, y_pred, average='binary')  # For binary classification
    f1 = f1_score(y_test, y_pred, average='binary')  # For binary classification
    average_precision = average_precision_score(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    plot_pr_curve(model, X_test, y_test, name)
    plot_roc_curve(model, X_test, y_test, name)
    #plot_class_separation(model, X_test, y_test, name)
    plot_density(y_test, y_probs, name)
    return {
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': roc_auc,
        'Average Precision': average_precision
    }

# Plot PR Curve
def plot_pr_curve(model, X_test, y_test, model_name="Model"):
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    average_precision = average_precision_score(y_test, y_probs)
    plt.plot(recall, precision, marker='.', label=f'{model_name} (AP={average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve for {model_name}')
    plt.legend()
    plt.show()

# Plot ROC Curve
def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='b', label=f'{model_name} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()
    plt.show()

# Plot Class Separation using PCA
def plot_class_separation(model, X_test, y_test, model_name="Model"):
    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test)
    y_pred = model.predict(X_test)
    plt.scatter(X_test_2d[y_test == 0][:, 0], X_test_2d[y_test == 0][:, 1], alpha=0.6, label='Class 0')
    plt.scatter(X_test_2d[y_test == 1][:, 0], X_test_2d[y_test == 1][:, 1], alpha=0.6, label='Class 1')
    plt.scatter(X_test_2d[y_pred != y_test][:, 0], X_test_2d[y_pred != y_test][:, 1], color='red', marker='x', label='Misclassified')
    plt.legend()
    plt.title(f'Class Separation Plot for {model_name}')
    plt.show()

# Plot density of predicted probabilities
def plot_density(y_test, y_probs, model_name="Model"):
    sns.kdeplot(y_probs[y_test == 0], label='Class 0', fill=True)
    sns.kdeplot(y_probs[y_test == 1], label='Class 1', fill=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'Density Plot for Predicted Probabilities - {model_name}')
    plt.legend()
    plt.show()

# Function to collect metrics for comparison
def collect_metrics(models, X_test, y_test):
    metrics = []
    for model in models:
        metrics.append(evaluate_model(model['model'], X_test, y_test, name=model['name']))
    return pd.DataFrame(metrics)

# %% [markdown]
# # 1. Random Forest - Baseline vs Tuned

# %%
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
evaluate_model(rf_clf, X_test, y_test, name="Random Forest (Baseline)")

# %%
# Hyperparameter tuning for Random Forest
rf_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1')
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
evaluate_model(rf_best, X_test, y_test, name="Random Forest (Tuned)")

# %%
rf_grid.best_params_

# %% [markdown]
# # 2. XGBoost - Baseline vs Tuned

# %%
xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
evaluate_model(xgb_clf, X_test, y_test, name="XGBoost (Baseline)")

# %%
# Hyperparameter tuning for XGBoost
xgb_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=5, scoring='f1')
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
evaluate_model(xgb_best, X_test, y_test, name="XGBoost (Tuned)")

# %% [markdown]
# # 3. Stacking Classifier - Baseline

# %%
base_learners = [
    ('knn', KNeighborsClassifier()),
    ('ridge', RidgeClassifier()),
    ('svc', SVC(probability=True))
]
final_estimator = LogisticRegression(solver='saga', C=0.1, max_iter=100)
stack_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=final_estimator,
    cv=5
)
stack_clf.fit(X_train, y_train)
evaluate_model(stack_clf, X_test, y_test, name="Stacking Classifier (Baseline)")

# %% [markdown]
# # 4. CatBoost - Baseline

# %%
X1 = df.drop('Churn', axis=1)
y1 = df['Churn']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)

cat_clf = CatBoostClassifier(cat_features=categorical_cols, iterations=100, verbose=0, random_state=42)
cat_clf.fit(X_train1, y_train1)
evaluate_model(cat_clf, X_test1, y_test1, name="CatBoost (Baseline)")

# %% [markdown]
# # Perform cross-validation and collect metrics for all models

# %%
from sklearn.model_selection import cross_validate

models = [
    {'model': rf_best, 'name': 'Random Forest (Tuned)'},
    {'model': xgb_best, 'name': 'XGBoost (Tuned)'},
    {'model': stack_clf, 'name': 'Stacking Classifier (Baseline)'}
]

cv_results = []
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']

for model in models:
    scores = cross_validate(model['model'], X, y, cv=5, scoring=scoring, return_train_score=False)
    model_results = {'Model': model['name']}
    for metric in scoring:
        model_results[f'CV {metric.capitalize()} Mean'] = scores[f'test_{metric}'].mean()
        model_results[f'CV {metric.capitalize()} Std'] = scores[f'test_{metric}'].std()
    cv_results.append(model_results)

# Cross-validation for CatBoost separately
cbt_scores = cross_validate(cat_clf, X1, y1, cv=5, scoring=scoring, return_train_score=False)
cbt_results = {'Model': 'CatBoost Classifier'}
for metric in scoring:
    cbt_results[f'CV {metric.capitalize()} Mean'] = cbt_scores[f'test_{metric}'].mean()
    cbt_results[f'CV {metric.capitalize()} Std'] = cbt_scores[f'test_{metric}'].std()
cv_results.append(cbt_results)

# Convert results to DataFrame
cv_results_df = pd.DataFrame(cv_results)
print("\nCross-Validation Results:\n", cv_results_df)

# %%
display("Cross-Validation Results:", cv_results_df)

# %%
from sklearn.tree import DecisionTreeClassifier
# Cross-validation for base learners separately
base_learners = [
    ('tree', DecisionTreeClassifier(class_weight='balanced')),
    ('Ridge Classifier', RidgeClassifier()),
    ('svc', SVC(C=2.0, kernel='sigmoid', probability=True)),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
]
cv_results_base = []
for name, model in base_learners:
    scores = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=False)
    model_results = {'Model': name}
    for metric in scoring:
        model_results[f'CV {metric.capitalize()} Mean'] = scores[f'test_{metric}'].mean()
        model_results[f'CV {metric.capitalize()} Std'] = scores[f'test_{metric}'].std()
    cv_results_base.append(model_results)

# %%
cv_results_base_df = pd.DataFrame(cv_results_base)
cv_results_base_df

# %%
pass
