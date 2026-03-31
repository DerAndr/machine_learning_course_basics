# /// script
# source-notebook = "example_04.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Ordinal Classification with Ordinal Logistic Regression

# %% [markdown]
# In this notebook, we'll:
# 
# - Introduce **ordinal classification** and its importance.
# - Use the **Wine Quality** dataset as an example.
# - Implement **Ordinal Logistic Regression** using the `mord` library.
# - Evaluate the model using appropriate metrics like **Mean Absolute Error (MAE)** and **Quadratic Weighted Kappa (QWK)**.
# - Visualize and interpret the results.

# %% [markdown]
# ## Table of Contents
# 
# 1. [Introduction](#scrollTo=wgyxz23kveT2)
# 2. [Import Libraries](#scrollTo=ajSc4WREvqqx)
# 3. [Load and Explore the Data](#scrollTo=tVe6yxu-vy-V)
# 4. [Data Preprocessing](#scrollTo=neKyOMkywbKQ)
# 5. [Ordinal Logistic Regression](#scrollTo=saiB-CF-woiJ)
# 6. [Model Evaluation](#scrollTo=wD64kDqrw3K4)
#     - [Mean Absolute Error (MAE)](#scrollTo=ZQxAgsG9w4Ta)
#     - [Quadratic Weighted Kappa (QWK)](#scrollTo=WB2-9LgVxAGZ)
# 7. [Conclusion](#scrollTo=htKAGeWbxUfs)

# %% [markdown]
# ## Introduction

# %% [markdown]
# ### What is Ordinal Classification?

# %% [markdown]
# **Ordinal classification** (or ordinal regression) deals with predicting categories that have a natural, ordered relationship. Unlike nominal classification (where classes are unordered) and regression (where outputs are continuous), ordinal classification predicts labels that are discrete and ordered.

# %% [markdown]
# **Examples of ordinal variables:**
# 
# - Movie ratings (e.g., 1 star to 5 stars)
# - Customer satisfaction levels (e.g., "Very Unsatisfied" to "Very Satisfied")
# - Wine quality ratings (e.g., scores from 0 to 10)

# %% [markdown]
# ### Objective

# %% [markdown]
# We aim to predict the quality of wine based on its physicochemical properties using ordinal classification techniques.
# 
# 
# ---
# 

# %% [markdown]
# ## Import Libraries

# %%
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report

# Import Ordinal Logistic Regression from the mord package
# NOTE: notebook magic commented for local script use: !pip install mord
from mord import LogisticIT

# For Quadratic Weighted Kappa
from sklearn.metrics import cohen_kappa_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Load and Explore the Data

# %% [markdown]
# We'll use the **Wine Quality** dataset from the UCI Machine Learning Repository.

# %% [markdown]
# ### Load the Dataset

# %%
# Load the dataset directly from the URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

# %% [markdown]
# ### Explore the Dataset

# %%
# Display the first five rows
display(data.head())

# %%
# Summary statistics
data.describe()

# %% [markdown]
# ### Check for Missing Values

# %%
# Check for null values
data.isnull().sum()

# %% [markdown]
# **Observation**: No missing values in the dataset.

# %% [markdown]
# ### Target Variable Distribution

# %%
# Distribution of the 'quality' variable
quality_counts = data['quality'].value_counts().sort_index()
print("Quality Counts:\n", quality_counts)

# %%
# Plot the distribution
plt.figure(figsize=(8,6))
sns.countplot(x='quality', data=data)
plt.title('Wine Quality Distribution')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# **Observation**: The quality scores range from 3 to 8, with most wines rated between 5 and 6.

# %% [markdown]
# ---
# 
# ## Data Preprocessing

# %% [markdown]
# 
# ### Feature Matrix and Target Vector

# %% [markdown]

# %%
X = data.drop('quality', axis=1)
y = data['quality']

# %% [markdown]
# ### Split into Training and Testing Sets

# %% [markdown]
# We'll use stratified splitting to maintain the distribution of quality scores in both training and testing sets.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
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
# ---
# 
# ## Ordinal Logistic Regression

# %% [markdown]
# ### Introduction to Ordinal Logistic Regression

# %% [markdown]
# **Ordinal Logistic Regression** is a type of regression model used for predicting an ordinal variable. It extends the logistic regression model to handle ordered categorical variables by modeling the cumulative probabilities of the categories.
# 
# We'll use the `LogisticIT` model from the `mord` library, which implements an ordinal logistic regression with proportional odds.

# %% [markdown]
# ### Train the Model

# %%
# Initialize the Ordinal Logistic Regression model
model = LogisticIT()

# Train the model
model.fit(X_train, y_train)

# %% [markdown]
# ### Predictions on the Test Set

# %%
# Predictions
y_pred = model.predict(X_test)

# %% [markdown]
# ## Model Evaluation

# %% [markdown]
# ### Mean Absolute Error (MAE)

# %% [markdown]
# **Mean Absolute Error (MAE)** measures the average magnitude of errors between predicted and actual values.

# %%
from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error (MAE):', mae)

# %% [markdown]
# **Interpretation**: A lower MAE indicates better model performance.

# %% [markdown]
# ### Quadratic Weighted Kappa (QWK)

# %% [markdown]
# **Quadratic Weighted Kappa (QWK)** measures the agreement between two ratings. It ranges from -1 (complete disagreement) to 1 (complete agreement). It penalizes larger discrepancies more than smaller ones.

# %%
from sklearn.metrics import cohen_kappa_score

# Calculate Quadratic Weighted Kappa
qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
print('Quadratic Weighted Kappa (QWK):', qwk)

# %% [markdown]
# **Interpretation**: A higher QWK indicates better agreement between predicted and actual quality scores.

# %% [markdown]
# ### Confusion Matrix

# %%
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
disp = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% [markdown]
# **Observation**: The confusion matrix shows that most predictions are close to the actual values, indicating the model respects the ordinal nature of the data.

# %% [markdown]
# ### Classification Report

# %%
print('Classification Report:\n')
print(classification_report(y_test, y_pred))

# %% [markdown]
# **Note**: While precision, recall, and F1-score are not ideal for ordinal data, they provide insight into the model's performance for each class.

# %% [markdown]
# ---
# 
# ## Conclusion
# 
# - **Ordinal Classification**: Successfully built an ordinal logistic regression model to predict wine quality ratings.
# - **Evaluation Metrics**:
#     - **Mean Absolute Error (MAE)**: Provided an average magnitude of prediction errors.
#     - **Quadratic Weighted Kappa (QWK)**: Measured the agreement between predicted and actual ratings, accounting for the ordinal nature.
# - **Model Performance**:
#     - The model achieved a relatively low MAE and a reasonable QWK score, indicating good performance in predicting ordered categories.
#     - The confusion matrix showed that most predictions are close to the actual ratings, respecting the order.
# 
# **Key Takeaways**:
# 
# - **Ordinal Logistic Regression** is suitable for predicting ordered categorical variables.
# - Using appropriate evaluation metrics is crucial for ordinal classification tasks.
# - The `mord` library provides easy-to-use implementations of ordinal regression models.
# 
# ---
# 
# **Encouragement**:
# 
# Exploring ordinal classification allows you to handle problems where the target variable has a natural order. By understanding and applying ordinal regression techniques, you're enhancing your ability to build models that respect the inherent structure of your data. Keep experimenting with different datasets and models to deepen your machine learning expertise!
# 
# ---

# %%
pass
