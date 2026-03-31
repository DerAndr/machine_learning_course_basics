# /// script
# source-notebook = "example_03.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Multilabel Classification with Logistic Regression and Evaluation Metrics

# %% [markdown]
# In this notebook, we'll:
# 
# Use a multilabel dataset.
# * Implement Logistic Regression for multilabel classification.
# * Address any class imbalance if necessary.
# * Utilize appropriate evaluation metrics for multilabel classification.
# * Perform cross-validation and compute cross-validated metrics.

# %% [markdown]
# # Introduction

# %% [markdown]
# 
# **Multilabel classification** involves assigning each instance to one or more classes. This is different from multiclass classification, where each instance is assigned to one and only one class.
# 
# **Objective:** Build a Logistic Regression classifier to predict multiple labels for each instance and evaluate the model using appropriate multilabel metrics.

# %% [markdown]
# # Import Libraries

# %%
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    classification_report,
    label_ranking_average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Generate and Explore the Data

# %% [markdown]
# 
# Since multilabel datasets can be hard to find and load, we'll use scikit-learn's `make_multilabel_classification` to generate a synthetic dataset suitable for multilabel classification.

# %% [markdown]
# ### Generate the Dataset

# %%
# Generate a synthetic multilabel dataset
X, Y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5,
                                      n_labels=2, allow_unlabeled=False, random_state=42)

# %% [markdown]
# - **n_samples**: Number of samples.
# - **n_features**: Number of features.
# - **n_classes**: Number of possible labels.
# - **n_labels**: Average number of labels per instance.
# - **allow_unlabeled**: If `False`, all samples have at least one label.

# %% [markdown]
# ### Explore the Dataset

# %%
# Convert to DataFrame for easier exploration
X = pd.DataFrame(X)
Y = pd.DataFrame(Y, columns=[f'Label_{i}' for i in range(1, 6)])

# Display the first five rows of features
X.head()

# %%
# Display the first five rows of labels
display(Y.head())

# %% [markdown]
# ### Check Label Distribution

# %%
# Sum of each label
label_counts = Y.sum()
print('Label Counts:\n', label_counts)

# %%
# Plot the label distribution
plt.figure(figsize=(8,6))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title('Label Distribution')
plt.xlabel('Labels')
plt.ylabel('Number of Instances')
plt.show()

# %% [markdown]
# **Observation**: The labels may not be perfectly balanced, but for this synthetic dataset, the imbalance is typically not severe.

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Split into Training and Testing Sets

# %%
# Split features and labels into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# %% [markdown]
# ### Feature Scaling

# %%
# Initialize the scaler
scaler = StandardScaler()

# Fit on training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ## Modeling with Logistic Regression

# %% [markdown]
# In multilabel classification, we can use **One-vs-Rest (OvR)** strategy, where a binary classifier is fit for each label.

# %% [markdown]
# ### Initialize and Train the Model

# %%
# Initialize the Logistic Regression model
model = OneVsRestClassifier(
    LogisticRegression(solver='lbfgs', max_iter=100, random_state=42)
)

# Train the model
model.fit(X_train, Y_train)

# %% [markdown]
# ### Predictions on the Test Set

# %%
# Predictions
Y_pred = model.predict(X_test)

# %% [markdown]
# ## Model Evaluation

# %% [markdown]
# Evaluating multilabel classifiers requires specialized metrics.

# %% [markdown]
# ### Hamming Loss

# %% [markdown]
# The **Hamming Loss** is the fraction of labels that are incorrectly predicted.

# %%
from sklearn.metrics import hamming_loss

# Calculate Hamming Loss
hl = hamming_loss(Y_test, Y_pred)
print('Hamming Loss:', hl)

# %% [markdown]
# **Interpretation**: Lower Hamming Loss indicates better performance.

# %% [markdown]
# ### Subset Accuracy

# %% [markdown]
# The **Subset Accuracy** is the fraction of samples that have all their labels correctly predicted.

# %%
from sklearn.metrics import accuracy_score

# Calculate Subset Accuracy
subset_acc = accuracy_score(Y_test, Y_pred)
print('Subset Accuracy:', subset_acc)

# %% [markdown]
# **Interpretation**: This is a strict metric; the prediction for a sample is considered correct only if all its predicted labels match the true labels.

# %% [markdown]
# ### Classification Report

# %% [markdown]
# We can generate a classification report for each label.

# %%
print('Classification Report:\n')
print(classification_report(Y_test, Y_pred, target_names=Y.columns))

# %% [markdown]
# ### Label Ranking Average Precision (LRAP)

# %% [markdown]
# **LRAP** measures how well the classifier ranks the true labels.

# %%
from sklearn.metrics import label_ranking_average_precision_score

# Predict probabilities
Y_pred_proba = model.predict_proba(X_test)

# Calculate LRAP
lrap = label_ranking_average_precision_score(Y_test.values, Y_pred_proba)
print('Label Ranking Average Precision (LRAP):', lrap)

# %% [markdown]
# **Interpretation**: LRAP ranges from 0 to 1, with higher scores indicating better performance.

# %% [markdown]
# ## Cross-Validated Metrics

# %% [markdown]
# We'll perform cross-validation to get a more robust evaluation.

# %% [markdown]
# ### Cross-Validation Setup

# %%
# Initialize KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %% [markdown]
# ### Cross-Validated Hamming Loss

# %%
from sklearn.model_selection import cross_val_score

# Cross-validated Hamming Loss
from sklearn.metrics import make_scorer

hl_scores = cross_val_score(
    model, X, Y, cv=kf, scoring=make_scorer(hamming_loss,greater_is_better=False), n_jobs=-1
)
print('Cross-Validated Hamming Loss:', -hl_scores)
print('Mean Hamming Loss:', -hl_scores.mean())

# %% [markdown]
# ### Cross-Validated Subset Accuracy

# %%
# Cross-validated Subset Accuracy
subset_acc_scores = cross_val_score(
    model, X, Y, cv=kf, scoring='accuracy', n_jobs=-1
)
print('Cross-Validated Subset Accuracy:', subset_acc_scores)
print('Mean Subset Accuracy:', subset_acc_scores.mean())

# %% [markdown]
# ### Cross-Validated LRAP

# %%
# Define a custom scorer for LRAP
from sklearn.metrics import make_scorer

lrap_scorer = make_scorer(label_ranking_average_precision_score)

# Cross-validated LRAP
lrap_scores = cross_val_score(
    model, X, Y, cv=kf, scoring=lrap_scorer, n_jobs=-1
)
print('Cross-Validated LRAP:', lrap_scores)
print('Mean LRAP:', lrap_scores.mean())

# %% [markdown]
# ## Conclusion

# %% [markdown]
# 
# 
# - **Multilabel Classification**: Successfully built a Logistic Regression model using One-vs-Rest strategy to handle multilabel classification.
# - **Evaluation Metrics**:
#     - **Hamming Loss**: Provided an average misclassification rate over all labels.
#     - **Subset Accuracy**: Offered a strict metric where only exact matches are considered correct.
#     - **Label Ranking Average Precision (LRAP)**: Measured the ability of the classifier to rank true labels higher than false ones.
# - **Cross-Validation**: Provided robust estimates of model performance across different splits of the data.
# 
# **Key Takeaways**:
# 
# - Multilabel classification requires specialized models and evaluation metrics.
# - Logistic Regression with One-vs-Rest strategy is effective for multilabel problems.
# - Evaluation metrics like Hamming Loss, Subset Accuracy, and LRAP provide insights into different aspects of model performance.
# - Cross-validation enhances the reliability of performance estimates.
# 
# ---
# 
# **Encouragement**:
# 
# By exploring multilabel classification, you've broadened your understanding of machine learning tasks that involve more complex label structures. Keep experimenting with different datasets and models to continue enhancing your machine learning skills!

# %%
pass
