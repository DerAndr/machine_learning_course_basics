# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # K Nearest Neighbours Classifier
# 
# 
# 1.  [Imports](#scrollTo=Izuws63frWoq&line=1&uniqifier=1)
# 2.  [Data Praparation](#scrollTo=THLfjwqqsK63&line=1&uniqifier=1)
# 3.  [KNN Model - Training, Evaluation](#scrollTo=iA4wh_HWwAou&line=1&uniqifier=1)
# 4.  [Hyperparamenter Tuning](#scrollTo=VWGhX0SJBVlv&line=1&uniqifier=1)
# 5.  [Advanced KNN Variantions](#scrollTo=ynnQPDnwDCTv&line=1&uniqifier=1)
# 6. [Model Evaluation](#scrollTo=wMY-kPXSxYLv&line=1&uniqifier=1)
# 7. [Conclusion](#scrollTo=AojSUwoJj0QP)
# 8. [Refecences](#scrollTo=m1yGl81ljon5)
# 

# %% [markdown]
# # 1. Imports

# %%
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# %% [markdown]
# # 2. Dataset Preparation

# %% [markdown]
# ## 2.1. Loading Dataset

# %%
# Load the Breast Cancer dataset
data = load_breast_cancer()

# Create a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display the first five rows
df.head()

# %% [markdown]
# ## 2.2 Dataset Overview

# %%
# Basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFeature Names:", data.feature_names)
print("\nTarget Classes:", data.target_names)

# %% [markdown]
# ## Check Class Distribution

# %%
df['target'].value_counts(normalize=True)

# %% [markdown]
# The dataset is relatively balanced (67/37), making it suitable for binary classification without imbalance concerns.

# %% [markdown]
# ## 2.3 Data Preprocessing

# %% [markdown]
# **Proper preprocessing is crucial for KNN as it relies on distance metrics.**

# %%
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# %%
# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# %% [markdown]
# **KNN is sensitive to the scale of the data. We'll standardize the features using StandardScaler.**

# %%
# Initialize the scaler
scaler = StandardScaler()

# Fit on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # 3. KNN Model - Training, Evaluation

# %% [markdown]
# ## 3.1 Basic KNN with Default Parameters

# %%
# Initialize KNN with default parameters (n_neighbors=5)
knn = KNeighborsClassifier()

# Fit the model
knn.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = knn.predict(X_test_scaled)

# %%
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# %%
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
# Classification Report
print(classification_report(y_test, y_pred, target_names=data.target_names))

# %% [markdown]
# # Hyperparameter Tuning

# %% [markdown]
# ## GridSearch for optimal K

# %%
# Define the parameter grid
param_grid = {'n_neighbors': np.arange(1, 31)}

# Initialize GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')

# Fit the grid search
grid.fit(X_train_scaled, y_train)

# Best parameters
print(f"Best number of neighbors: {grid.best_params_['n_neighbors']}")

# Best cross-validation accuracy
print(f"Best cross-validation accuracy: {grid.best_score_:.2f}")

# %% [markdown]
# ## Retrain with Optimal K

# %%
# Get the best K
best_k = grid.best_params_['n_neighbors']

# Initialize KNN with best K
knn_best = KNeighborsClassifier(n_neighbors=best_k)

# Fit the model
knn_best.fit(X_train_scaled, y_train)

# Predict on test data
y_pred_best = knn_best.predict(X_test_scaled)

# Evaluate
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test Accuracy with K={best_k}: {accuracy_best:.2f}")

# %% [markdown]
# ## Cross-Validation Scores

# %%
# Cross-validation scores
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_skf = cross_val_score(knn_best, X_train_scaled, y_train, cv=skf, scoring='accuracy')
#OR
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_kf = cross_val_score(knn_best, X_train_scaled, y_train, cv=kf, scoring='accuracy')
# print(f"Cross-validation scores (StratifiedKFold): {cv_scores_skf}")
print(f"Mean CV Accuracy (StratifiedKFold): {cv_scores_skf.mean():.2f}")

# print(f"Cross-validation scores: {cv_scores_kf}")
print(f"Mean CV Accuracy: {cv_scores_kf.mean():.2f}")

# %% [markdown]
# # Advanced KNN Variations

# %% [markdown]
# ## Weighted KNN

# %% [markdown]
# Weighted KNN assigns different weights to neighbors based on their distance. Closer neighbors have higher weights.

# %%
# Initialize Weighted KNN
knn_weighted = KNeighborsClassifier(n_neighbors=best_k, weights='distance')

# Fit the model
knn_weighted.fit(X_train_scaled, y_train)

# Predict
y_pred_weighted = knn_weighted.predict(X_test_scaled)

# Evaluate
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
print(f"Weighted KNN Accuracy: {accuracy_weighted:.2f}")

# %% [markdown]
# ## Using Different Distance Metrics

# %% [markdown]
# KNN can use various distance metrics. Let's compare Euclidean and Manhattan distances.

# %%
# Initialize KNN with Manhattan distance
knn_manhattan = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')

# Fit the model
knn_manhattan.fit(X_train_scaled, y_train)

# Predict
y_pred_manhattan = knn_manhattan.predict(X_test_scaled)

# Evaluate
accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)
print(f"Manhattan KNN Accuracy: {accuracy_manhattan:.2f}")

# %%
# Summary of accuracies
variants = {
    'Standard KNN': accuracy_best,
    'Weighted KNN': accuracy_weighted,
    'Manhattan KNN': accuracy_manhattan
}

# Create a DataFrame for visualization
variant_df = pd.DataFrame(list(variants.items()), columns=['Variant', 'Accuracy'])

# Plot
sns.barplot(x='Variant', y='Accuracy', data=variant_df)
plt.ylim(0.90, 1.00)
plt.title('KNN Variants Comparison')
plt.show()

# %%
pass

# %% [markdown]
# 
# 
# ---
# 

# %% [markdown]
# # Model Evaluation

# %% [markdown]
# ### Accuracy score

# %%
accuracy = accuracy_score(y_test, y_pred_best)
print(f"Accuracy: {accuracy:.2f}")

# %% [markdown]
# ### Confusion Matrix

# %%
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# ### Classification report

# %%
# Classification Report
print(classification_report(y_test, y_pred_best, target_names=data.target_names))

# %% [markdown]
# **Support** is the number of actual occurrences of the class in the specified dataset.

# %% [markdown]
# ### Estimate Probabilities
# Since KNN return only labels, we need to estimate the probabilities. Luckily there is such a function in sklearn inmplementation of KNN Classifier.
# 
# Once the nearest neighbors are identified, `predict_proba` calculates the probability of each class by considering the proportion of neighbors belonging to each class.

# %%
# Predict probabilities for ROC
y_proba = knn_best.predict_proba(X_test_scaled)[:,1]

# %% [markdown]
# ### Receiver Operator Characteristic curve, ROC curve

# %%
# Compute ROC curve and AUC
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'KNN (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')  # Diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# %%
plot_roc_curve(y_test, y_proba)

# %% [markdown]
# ### Interpretation of ROC Curve and AUC
# 
# Mathematically, the **ROC (Receiver Operating Characteristic) Curve** plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold settings. The **Area Under the Curve (AUC)** represents the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance.
# 
# ### Key Components:
# - **True Positive Rate (TPR)**, also known as **Recall**:
#   
#   $TPR = \frac{TP}{TP + FN}$
#   
#   TPR indicates the proportion of actual positives correctly identified by the classifier.
# 
# - **False Positive Rate (FPR)**:
#   
#   $FPR = \frac{FP}{FP + TN}$
#   
#   FPR represents the proportion of actual negatives incorrectly labeled as positives.
# 
# ### ROC Curve Plot:
# The ROC curve plots **FPR** on the x-axis and **TPR** on the y-axis, illustrating the trade-off between sensitivity and specificity across different classification thresholds. A classifier with no discriminative power would follow the diagonal line (random guess), while a good classifier will have a curve that rises steeply towards the top left corner.
# 
# ### Area Under the ROC Curve (AUC):
# - **AUC-ROC** is a single scalar value summarizing the ROC Curve, representing the overall performance of the classifier across all thresholds.
#   
#   - **AUC = 1**: Perfect classifier that completely distinguishes between positive and negative classes.
#   - **AUC = 0.5**: Random classifier with no discriminative ability (the diagonal line).
#   - **AUC < 0.5**: Worse than random; the classifier inversely discriminates the classes, which is generally problematic.
# 
# ### Higher AUC-ROC values:
# Higher AUC values indicate that the classifier can achieve a high true positive rate (recall) without a corresponding high false positive rate, meaning it can better separate the positive class from the negative class.
# 
# ### Lower AUC-ROC values:
# Lower AUC values suggest that the classifier struggles to distinguish between the classes, often performing only marginally better than random guessing.
# 
# ### When to Use AUC-ROC:
# - **Balanced Classes**: AUC-ROC is particularly informative when the positive and negative classes are roughly balanced.
# - **Comparing Classifiers**: It is commonly used to compare the overall discriminatory power of different classifiers.
# 
# ---
# 
# ### Practical Use Cases:
# 1. **Medical Diagnostics**: In situations where it's crucial to correctly identify true positives (e.g., diagnosing a disease), AUC-ROC helps assess the classifier's ability to capture true cases without generating too many false alarms.
# 2. **Fraud Detection**: In fraud detection, AUC-ROC helps determine the model's effectiveness in identifying fraudulent transactions while minimizing the misclassification of legitimate transactions as fraud.
# 
# In summary, the ROC Curve provides insight into the balance between recall and false positive rate at different thresholds, while AUC-ROC offers a single metric that quantifies the model's overall ability to distinguish between positive and negative classes across all thresholds.

# %% [markdown]
# ### Class score distribution

# %%
def plot_class_score_distribution(y_true, y_prob):
    """
    Plots the class score distribution for two classes based on two arrays.

    Parameters:
    - y_true: array-like, ground truth binary labels (0 or 1)
    - y_prob: array-like, predicted scores or probabilities for the positive class
    """
    plt.figure(figsize=(8, 6))

    # Plot KDE for each class
    for class_value, color, linestyle in zip(np.unique(y_true), ['darkorange', 'navy'], ['-', '--']):
        subset_scores = y_prob[np.array(y_true) == class_value]
        sns.kdeplot(
            subset_scores,
            color=color,
            linestyle=linestyle,
            label=f'Z={class_value}',
            fill=True,
            alpha=0.3
        )

    # Add legend, labels, and title
    plt.legend(title="Class")
    plt.xlabel('Estimated Propensity Score')
    plt.ylabel('Density')
    plt.title('Class Score Distribution')
    plt.show()

# %%
plot_class_score_distribution(y_test, y_proba)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve(y_true, y_prob, pos_label=1):
    """
    Plots the Precision-Recall curve for a binary classifier.

    Parameters:
    - y_true: array-like of shape (n_samples,), ground truth binary labels (0 or 1)
    - y_scores: array-like of shape (n_samples,), predicted scores or probabilities for the positive class
    - pos_label: int, the label of the positive class (default is 1)
    """
    # Calculate precision-recall pairs for different probability thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
    avg_precision = average_precision_score(y_true, y_prob)

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', linewidth=2, label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()

# %%
plot_precision_recall_curve(y_test, y_proba)

# %% [markdown]
# **TL;DR:** Average Precision (AP) is a single-number summary of the Precision-Recall (PR) curve, representing the area under the PR curve. It captures how well a binary classifier distinguishes positive from negative examples, especially in imbalanced datasets.

# %% [markdown]
# ### **Interpretation of Average Precision (AP)**
# 
# 
# Mathematically, AP is calculated by summing up the precision values at different recall thresholds, weighted by the change in recall:
# $AP = \sum_{i}(R_{i} - R_{i-1})\cdot P_{i}, $
# 
# where $R_{i}$ and $P_{i}$ are the recall and precision values at threshold $i$.
# 
# 
# **Higher AP values** (closer to 1) indicate that the classifier maintains a high level of precision even as recall increases, meaning it can capture positive instances without introducing too many false positives.
# 
# 
# **Lower AP values** suggest that as recall increases, precision drops, indicating that the model struggles to balance capturing positives without mistakenly labeling negatives as positives.
# Relationship with AUC-PR:
# 
# AP is closely related to the Area Under the Precision-Recall Curve (AUC-PR). In fact, AP is often considered a version of AUC-PR where precision is averaged over all recall levels.
# Unlike traditional ROC-AUC (which considers both classes equally), **AP and AUC-PR emphasize the model’s performance on the positive class,** which is crucial in imbalanced datasets.

# %% [markdown]
# **When to Use AP:**
# 
# **For Imbalanced Classes:** AP is particularly useful in domains like medical diagnostics, fraud detection, and anomaly detection, where the positive class (e.g., disease, fraud) is rare but important to identify.
# 
# 
# **Evaluating High Recall Systems:** If capturing all positives is critical, AP can help balance the need for high recall without drastically sacrificing precision.

# %% [markdown]
# ### Estimating Threshold

# %% [markdown]
# #### Optimal threshold based on F1 score

# %%
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

def find_optimal_threshold(y_true, y_prob):
    """
    Finds the optimal threshold based on F1 score, and provides insights using PR curve.

    Parameters:
    - y_true: array-like, ground truth binary labels (0 or 1)
    - y_prob: array-like, predicted scores or probabilities for the positive class

    Returns:
    - optimal_thresholds: dict with optimal thresholds for F1 score and PR curve insights
    """
    # Calculate precision, recall, thresholds for PR Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Method 1: Using F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_idx]

    # Method 2: Midpoint of precision and recall (if you want balanced precision and recall)
    pr_diff = np.abs(precision - recall)
    optimal_idx_pr = np.argmin(pr_diff)
    optimal_threshold_pr_balance = thresholds[optimal_idx_pr]

    # Collect results
    optimal_thresholds = {
        "F1_score": optimal_threshold_f1,
        "Precision-Recall_Balance": optimal_threshold_pr_balance
    }

    # Plot Precision-Recall curve with marked optimal thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='PR Curve', color='blue')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', label=f'Optimal F1 Threshold = {optimal_threshold_f1:.2f}')
    plt.scatter(recall[optimal_idx_pr], precision[optimal_idx_pr], color='green', label=f'Balanced PR Threshold = {optimal_threshold_pr_balance:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Optimal Thresholds')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

    return optimal_thresholds

# %%
find_optimal_threshold(y_test, y_proba)

# %% [markdown]
# Youden's J Statistic (Optimal ROC Threshold)

# %%
from sklearn.metrics import roc_curve

def optimal_threshold_youden(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

# %%
optimal_threshold_youden(y_test, y_proba)

# %%
y_hat_adj = np.where(y_proba > 0.4, 1, 0)
# classification report
print(classification_report(y_test, y_hat_adj))

# %%
print(classification_report(y_test, y_pred_best))

# %% [markdown]
# #### Maximizing Specificity and Sensitivity Balance

# %%
def optimal_threshold_sensitivity_specificity(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    balance = np.abs(specificity - tpr)
    optimal_idx = np.argmin(balance)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

# %%
optimal_threshold_sensitivity_specificity(y_test, y_proba)

# %% [markdown]
# # Conclusion

# %% [markdown]
# 
# 
# In this guide, we explored the **K-Nearest Neighbors (KNN) classifier** using the Breast Cancer Wisconsin dataset. The key steps we covered include:
# 
# - **Data Loading and Exploration**: Understanding the dataset and its class distribution.
# - **Data Preprocessing**: Scaling the features, which is crucial for KNN due to its reliance on distance metrics.
# - **Model Implementation**: Building and evaluating a basic KNN model with default parameters.
# - **Hyperparameter Tuning**: Using Grid Search to find the optimal number of neighbors (K).
# - **Advanced Variations**: Implementing **Weighted KNN** and experimenting with different distance metrics (Euclidean vs. Manhattan) to observe their impact on performance.

# %% [markdown]
# ### Key Takeaways:
# 

# %% [markdown]
# - **Feature Scaling**: Always scale your features when using KNN, as the algorithm relies on distance calculations which are affected by feature magnitude.
# - **Choosing K**: A smaller value for K can lead to overfitting, while a larger K may cause underfitting. **Cross-validation** is an effective way to find the optimal value of K that balances bias and variance.
# - **Weighted KNN and Distance Metrics**: Assigning weights based on distance can improve the model, especially if some neighbors are closer and therefore more relevant. Trying different distance metrics, such as Manhattan distance, can also provide better results depending on the data characteristics.
# 
# KNN is a straightforward yet powerful algorithm, making it an excellent choice for introducing machine learning concepts. It also provides a good foundation for understanding more complex algorithms and concepts in machine learning.

# %% [markdown]
# # References

# %% [markdown]
# 
# 
# - [Scikit-learn K-Nearest Neighbors Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
# - [Grid Search and Cross-Validation in Scikit-Learn](https://scikit-learn.org/stable/modules/grid_search.html)

# %%
pass

# %%
pass
