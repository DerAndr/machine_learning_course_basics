# /// script
# source-notebook = "example_02.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Multiclass Classification with Logistic Regression using One-vs-Rest and One-vs-One Strategies

# %% [markdown]
# In this notebook, we'll:
# 
# * Use the Wine Quality dataset for multiclass classification.
# * Implement Logistic Regression using One-vs-Rest (OvR) and One-vs-One (OvO) strategies.
# * Compare the performance of both approaches.
# * Utilize appropriate evaluation metrics.

# %% [markdown]
# Multiclass classification can be approached using strategies that decompose the problem into multiple binary classification tasks:
# 
# **One-vs-Rest (OvR):** Fits one classifier per class, with the samples of that class as positive samples and all other samples as negatives.
# 
# **One-vs-One (OvO):** Fits one classifier per pair of classes.
# 
# **Objective:** Build Logistic Regression classifiers using OvR and OvO strategies to predict wine quality ratings, and compare their performance.

# %% [markdown]
# # Import Libraries

# %%
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score,
    make_scorer
)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Import SMOTE for handling imbalance
from imblearn.over_sampling import SMOTE

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Load and Explore the Data

# %% [markdown]
# We'll use the **Wine Quality** dataset from the UCI Machine Learning Repository.

# %% [markdown]
# ## Load the Dataset

# %%
# Load the dataset directly from the URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

# %% [markdown]
# ## Explore the Dataset

# %%
# Display the first five rows
display(data.head())

# %% [markdown]
# ## Check for Missing Values

# %%
# Check for null values
data.isnull().sum()

# %% [markdown]
# **Observation:** No missing values in the dataset.

# %% [markdown]
# ## Target Variable Distribution

# %%
# Distribution of the 'quality' variable
quality_counts = data['quality'].value_counts().sort_index()
print(quality_counts)

# %%
# Plot the distribution
plt.figure(figsize=(8,6))
sns.countplot(x='quality', data=data)
plt.title('Wine Quality Distribution')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# **Observation:** The dataset is imbalanced with some quality scores being underrepresented.

# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# ## Feature Matrix and Target Vector

# %%
X = data.drop('quality', axis=1)
y = data['quality']

# %% [markdown]
# ## Split into Training and Testing Sets

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% [markdown]
# ## Feature Scaling

# %%
# Initialize the scaler
scaler = StandardScaler()

# Fit on training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# # Handling Class Imbalance

# %% [markdown]
# ## Check Class Distribution

# %%
# Class distribution in y_train
class_counts = pd.Series(y_train).value_counts().sort_index()
print(class_counts)

# %% [markdown]
# **Observation:** Classes like '3' and '8' have very few samples.

# %% [markdown]
# ## Handling Class Imbalance with SMOTE

# %%
# Class distribution in y_train
class_counts = pd.Series(y_train).value_counts().sort_index()
print("Original class distribution:\n", class_counts)

# %% [markdown]
# ### Apply SMOTE

# %% [markdown]
# **SMOTE** generates synthetic samples for minority classes by interpolating between existing minority instances.

# %%
# Initialize SMOTE
smote = SMOTE(random_state=42)

# Resample the training data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# %% [markdown]
# **Note:** For multiclass problems, SMOTE automatically applies to all minority classes.

# %%
# New class distribution
resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
print("Resampled class distribution with SMOTE:\n", resampled_counts)

# %% [markdown]
# **Observation:** All classes now have the same number of samples, achieving balance.

# %% [markdown]
# # Modeling with Logistic Regression

# %% [markdown]
# We will implement both **One-vs-Rest (OvR)** and **One-vs-One (OvO)** strategies using Logistic Regression.

# %% [markdown]
# ## One-vs-Rest (OvR) Strategy

# %% [markdown]
# Initialize and Train the OvR Model

# %%
# Initialize the One-vs-Rest Logistic Regression model
lr_ovr = OneVsRestClassifier(
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
)

# Train the model on the resampled data
lr_ovr.fit(X_resampled, y_resampled)

# %%
# Predictions
y_pred_ovr = lr_ovr.predict(X_test)

# %% [markdown]
# ## One-vs-One (OvO) Strategy

# %% [markdown]
# Initialize and Train the OvO Model

# %%
# Initialize the One-vs-One Logistic Regression model
lr_ovo = OneVsOneClassifier(
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
)

# Train the model on the resampled data
lr_ovo.fit(X_resampled, y_resampled)

# %%
# Predictions
y_pred_ovo = lr_ovo.predict(X_test)

# %% [markdown]
# # Model Evaluation

# %% [markdown]
# ## Confusion Matrix

# %% [markdown]
# ### OvR Confusion Matrix

# %%
from sklearn.metrics import ConfusionMatrixDisplay

cm_ovr = confusion_matrix(y_test, y_pred_ovr, labels=sorted(y.unique()))
disp_ovr = ConfusionMatrixDisplay(confusion_matrix=cm_ovr, display_labels=sorted(y.unique()))
disp_ovr.plot(cmap='Blues')
plt.title('Confusion Matrix - One-vs-Rest')
plt.show()

# %% [markdown]
# ### OvO Confusion Matrix

# %%
cm_ovo = confusion_matrix(y_test, y_pred_ovo, labels=sorted(y.unique()))
disp_ovo = ConfusionMatrixDisplay(confusion_matrix=cm_ovo, display_labels=sorted(y.unique()))
disp_ovo.plot(cmap='Greens')
plt.title('Confusion Matrix - One-vs-One')
plt.show()

# %% [markdown]
# ## Classification Report

# %% [markdown]
# ### OvR Classification Report

# %%
print('Classification Report - One-vs-Rest:\n')
print(classification_report(y_test, y_pred_ovr))

# %% [markdown]
# ### OvO Classification Report

# %%
print('Classification Report - One-vs-One:\n')
print(classification_report(y_test, y_pred_ovo))

# %% [markdown]
# ## Visualize F1-Scores

# %% [markdown]
# ### OvR F1-Scores

# %%
# Calculate F1-score for each class
f1_scores_ovr = f1_score(y_test, y_pred_ovr, average=None, labels=sorted(y.unique()))

# Plot F1-scores
plt.figure(figsize=(8,6))
sns.barplot(x=sorted(y.unique()), y=f1_scores_ovr)
plt.title('F1-Score per Class - One-vs-Rest')
plt.xlabel('Quality Score')
plt.ylabel('F1-Score')
plt.show()

# %% [markdown]
# ### OvO F1-Scores

# %%
# Calculate F1-score for each class
f1_scores_ovo = f1_score(y_test, y_pred_ovo, average=None, labels=sorted(y.unique()))

# Plot F1-scores
plt.figure(figsize=(8,6))
sns.barplot(x=sorted(y.unique()), y=f1_scores_ovo)
plt.title('F1-Score per Class - One-vs-One')
plt.xlabel('Quality Score')
plt.ylabel('F1-Score')
plt.show()

# %% [markdown]
# ## Cross-Validated Metrics

# %% [markdown]
# We will perform cross-validation to compare the performance of both strategies more robustly, applying SMOTE within each fold.

# %% [markdown]
# ### Cross-Validation Setup

# %%
# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% [markdown]
# ### Cross-Validated F1-Score Function with SMOTE

# %%
# Define a function to perform cross-validation with SMOTE
def cross_val_score_smote(model, X, y, cv, scoring):
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        # Apply SMOTE
        X_res_cv, y_res_cv = smote.fit_resample(X_train_cv, y_train_cv)

        # Fit model
        model.fit(X_res_cv, y_res_cv)

        # Predict on test set
        y_pred_cv = model.predict(X_test_cv)

        # Calculate score
        score = f1_score(y_test_cv, y_pred_cv, average=scoring)
        scores.append(score)
    return scores

# %% [markdown]
# ### Cross-Validated F1-Score for OvR

# %%
# Initialize the One-vs-Rest Logistic Regression model
lr_ovr_cv = OneVsRestClassifier(
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
)

# Perform cross-validation
f1_macro_scores_ovr = cross_val_score_smote(
    lr_ovr_cv, X_train, y_train, skf, scoring='macro'
)
print('Cross-Validated Macro F1-Score (OvR):', f1_macro_scores_ovr)
print('Mean Macro F1-Score (OvR):', np.mean(f1_macro_scores_ovr))

# %% [markdown]
# ### Cross-Validated F1-Score for OvO

# %%
# Initialize the One-vs-One Logistic Regression model
lr_ovo_cv = OneVsOneClassifier(
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
)

# Perform cross-validation
f1_macro_scores_ovo = cross_val_score_smote(
    lr_ovo_cv, X_train, y_train, skf, scoring='macro'
)
print('Cross-Validated Macro F1-Score (OvO):', f1_macro_scores_ovo)
print('Mean Macro F1-Score (OvO):', np.mean(f1_macro_scores_ovo))

# %% [markdown]
# **Note:** Cross-validation with OvO may take longer due to the increased number of classifiers.

# %% [markdown]
# # Conclusion
# 
# - **Class Imbalance Handling with SMOTE**:
#   - SMOTE effectively balanced the dataset by generating synthetic samples for minority classes.
#   - Improved model performance across all classes.
# - **One-vs-Rest (OvR) vs. One-vs-One (OvO)**:
#   - **OvR**:
#     - Simpler and faster to train.
#     - Suitable when you have a large number of classes.
#   - **OvO**:
#     - Requires training more classifiers (*n(n - 1)/2*).
#     - Can be more accurate when classifiers are strong in binary discrimination.
# - **Evaluation Metrics**:
#   - **F1-Score**: Used to assess the balance between precision and recall.
#   - **Macro-Averaging**: Provided an unweighted mean metric, treating all classes equally.
# - **Cross-Validation with SMOTE**:
#   - Applying SMOTE within each fold ensures that synthetic samples are generated only from the training data, avoiding data leakage.
#   - Cross-validation provides a robust estimate of model performance.
# 
# **Key Takeaways**:
# 
# - **SMOTE** can be effectively combined with both OvR and OvO strategies to handle class imbalance in multiclass classification.
# - The choice between **OvR** and **OvO** depends on the specific problem and computational resources.
# - **Evaluation metrics** and cross-validation are crucial for comparing models and strategies.

# %%
pass
