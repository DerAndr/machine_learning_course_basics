# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Responsible AI - Guided Practice
# 
# Welcome to the practice session on **Responsible AI**.
# 
# In this notebook, we will move beyond simply training a model with high accuracy. We will explore:
# 1.  **Responsible ML**: Checking for bias and fairness in our models using `fairlearn`.
# 2.  **Mitigation**: Applying techniques to reduce model bias.
# 
# ---

# %% [markdown]
# ## Part 1: Responsible ML - Fairness and Bias
# 
# We will use the **Census Income (Adult)** dataset. The goal is to predict whether an individual earns more than $50K/year.
# 
# **Objective**: Train a model and check if it treats different groups (e.g., Male vs. Female) fairly.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# 1. Load Data
# We use a subset or the full version from OpenML
print("Loading dataset...")
data = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
df = data.frame

# Simplify for demonstration: select a few features
features = ['age', 'education-num', 'hours-per-week', 'sex', 'race', 'class']
df = df[features].dropna()

# Convert target to binary (1 if >50K, 0 otherwise)
df['target'] = df['class'].apply(lambda x: 1 if '>50K' in str(x) else 0)
df = df.drop(columns=['class'])

print("Dataset shape:", df.shape)
df.head()

# %% [markdown]
# ### Data Preprocessing
# We need to encode categorical variables. For fairness analysis, we will keep `sex` as a **sensitive attribute**.

# %%
# Encode 'sex' (Male=1, Female=0)
df['sex_encoded'] = df['sex'].map({'Male': 1, 'Female': 0})

X = df[['age', 'education-num', 'hours-per-week', 'sex_encoded']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# ### Train a Baseline Model

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# %% [markdown]
# ### Fairness Analysis with Fairlearn
# 
# We utilize `MetricFrame` to analyze model performance across different groups (Sensitive Features).

# %%
try:
    from fairlearn.metrics import MetricFrame, selection_rate, count
    from sklearn.metrics import accuracy_score, recall_score
except ImportError:
    # !pip install fairlearn
    from fairlearn.metrics import MetricFrame, selection_rate, count
    from sklearn.metrics import accuracy_score, recall_score

# Define metrics we want to check
metrics = {
    'accuracy': accuracy_score,
    'recall': recall_score,
    'selection_rate': selection_rate,  # How often model predicts positive class (>50k)
    'count': count
}

# Create MetricFrame
# A_test contains the sensitive sensitive feature (sex) for the test set
sex_test = X_test['sex_encoded'].map({1: 'Male', 0: 'Female'})

mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sex_test
)

# Print overall and group-level metrics
print("Overall Metrics:")
print(mf.overall)
print("\nGroup-level Metrics:")
print(mf.by_group)

# %% [markdown]
# **Visualization**: Plotting metrics by group helps identify disparities quickly.

# %%
mf.by_group.plot(kind='bar', subplots=True, layout=(2, 2), figsize=(10, 8), title="Metrics by Sex")
plt.show()

# %% [markdown]
# ### Bias Mitigation: Post-processing
# 
# We observed a disparity. Let's use `ThresholdOptimizer` to find optimal thresholds for each group to achieve **Equalized Odds** (or another parity criteria), ensuring comparable TPR and FPR across groups.

# %%
from fairlearn.postprocessing import ThresholdOptimizer

# Initialize Optimizer
# constraints="equalized_odds" tries to equalize True Positive Rate and False Positive Rate
optimizer = ThresholdOptimizer(
    estimator=model, 
    constraints="equalized_odds", 
    prefit=True,
    predict_method='predict_proba'
)

# Fit the optimizer
# Note: We need sensitive features for the training part of optimization
sex_train = X_train['sex_encoded'].map({1: 'Male', 0: 'Female'})
optimizer.fit(X_train, y_train, sensitive_features=sex_train)

# Predict using the fairness-aware optimizer
y_pred_fair = optimizer.predict(X_test, sensitive_features=sex_test)

print("Optimization Complete.")

# %% [markdown]
# #### Checking Results After Mitigation

# %%
mf_fair = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_fair,
    sensitive_features=sex_test
)

print("Original Recall Difference:", mf.difference()['recall'])
print("Mitigated Recall Difference:", mf_fair.difference()['recall'])

print("\nMitigated Group Metrics:")
print(mf_fair.by_group)

# %% [markdown]
# ### Conclusion
# You have now:
# 1.  Analyzed model bias using **Fairlearn MetricFrames**.
# 2.  Mitigated bias using **ThresholdOptimizer**.
# 3.  Compared group-level outcomes before and after mitigation.
