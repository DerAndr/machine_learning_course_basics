# /// script
# source-notebook = "example_04.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Data Splitting & Cross-Validation

# %% [markdown]
# * [Random split](#scrollTo=JFCrcH-MdxL5)
# * [Stratified split](#scrollTo=FzX_o-23d0LH)
# * [SMOTE](#scrollTo=COCyr9wbeUDR)
# * [Time-based split](#scrollTo=Xabtutjod3tD)
# * [Hierarchical split](#scrollTo=XhSxZG1od8sQ)
# * [KFold CV](#scrollTo=GaIYbwKLeD_J)
# * [Leave-One-Out Cross-Validation (LOOCV)](#scrollTo=OZqq9shhhjgW)
# * [Bias - Variance](#scrollTo=ULd6dA9KeK8z)

# %% [markdown]
# # Import Libraries and Load Real Datasets

# %%
# connect to google drive
# NOTE: Colab-only import commented for local script use: from google.colab import drive
drive.mount('/content/drive')

# %%
import numpy as np
import pandas as pd
import requests
import zipfile
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, cross_val_score, TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path

# Load Iris dataset for illustration (Balanced dataset)
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = pd.Series(iris.target, name='target')

import seaborn as sns
ames_data = sns.load_dataset('diamonds')  # Using 'diamonds' dataset for real data demonstration (similar in structure)

# %%
# paths
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/AirQuality/') # replace with your path to the dataset!
dataset_path = data_path/'AirQualityUCI.csv'


# Get the dataset
air_quality = pd.read_csv(dataset_path, sep=';', decimal=',', na_values=-200)
air_quality['Datetime'] = pd.to_datetime(air_quality['Date'] + ' ' + air_quality['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
air_quality = air_quality.drop(columns=['Date', 'Time']).dropna(subset=['Datetime'])
air_quality = air_quality.drop(columns=['Unnamed: 15', 'Unnamed: 16'])

air_quality = air_quality.dropna(subset=['CO(GT)', 'PT08.S1(CO)'], how='any')
air_quality = air_quality.fillna(-200)

# Filling or removing rows with too many missing values
threshold = len(air_quality.columns) * 0.5  # Allow rows with less than 50% missing data
air_quality = air_quality.dropna(thresh=threshold)

# Display the cleaned dataset
display(air_quality.head(10))
air_quality.shape

# %%
# paths
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Diabetes Health Indicators Dataset/') # replace with your path to the dataset!
dataset_path = data_path/'diabetes_012_health_indicators_BRFSS2015.csv'


# Get the dataset
diabetes = pd.read_csv(dataset_path)
display(diabetes.head())
diabetes['Diabetes_012'].value_counts(normalize=True)

# %%
# Load Iris dataset for illustration (Balanced dataset)
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = pd.Series(iris.target, name='target')

# %%
# Load Ames Housing dataset (using seaborn for demonstration)
import seaborn as sns
ames_data = sns.load_dataset('diamonds')  # Using 'diamonds' dataset for real data demonstration (similar in structure)

# %% [markdown]
# 
# 
# ---
# 

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# %% [markdown]
# # Random split

# %%
# DO: Random Split with fixed seed
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Fit a simple model
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Split - Good Example Accuracy: {accuracy:.2f}")

# Plot the data split
plt.figure(figsize=(10, 5))
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, s=20., cmap='cividis', alpha=0.6, marker='o', label='Training Set')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, s=80., cmap='cividis', alpha=0.8, marker='x', label='Test Set')
plt.title('Random Split - Good Example (Fixed Seed)')
plt.legend()
plt.show()

# %%
# DON'T: Random Split without fixed seed
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2)

# Fit a simple model
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Split - Good Example Accuracy: {accuracy:.2f}")

# Plot the data split
plt.figure(figsize=(10, 5))
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, s=20., cmap='cividis', alpha=0.6, marker='o', label='Training Set')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, s=80., cmap='cividis', alpha=0.8, marker='x', label='Test Set')
plt.title('Random Split - Good Example (Fixed Seed)')
plt.legend()
plt.show()

# %% [markdown]
# # Stratified split

# %%
from sklearn.datasets import fetch_openml
imbalanced = fetch_openml("Credit_Card_Fraud_")
print(imbalanced.DESCR)

imbalanced = imbalanced.frame

# %%
imbalanced.head()

# %%
pd.Series(imbalanced['fraud']).value_counts(normalize=True)

# %%
 a = imbalanced.sample(frac=0.01, random_state=42)

# %%
# DO: Stratified Split

# Create a dataset with imbalanced classes
X, y = a, a['fraud']
X = X.drop(columns=['fraud'])
# make dataset even mode imbalanced = make 50% of 1s zeros

# get numeric features
numeric_features = X.select_dtypes(include=['number']).columns
print(y.value_counts(normalize=True))

# Perform a stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=17)

# scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

# Fit a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(y_test.value_counts(normalize=True))
print(y_train.value_counts(normalize=True))

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Stratified Split -: Acc: {accuracy:.2f}, Pr: {precision:2f}, Rc: {recall:2f}, f1: {f1:2f}")

# %%
# Create a dataset with imbalanced classes
X, y = a, a['fraud']
X = X.drop(columns=['fraud'])

numeric_features = X.select_dtypes(include=['number']).columns
print(y.value_counts(normalize=True))

# Perform a stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=None,
                                                    shuffle=True,
                                                    random_state=17)

# scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

# Fit a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(y_test.value_counts(normalize=True))
print(y_train.value_counts(normalize=True))

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Not Stratified Split -: Acc: {accuracy:.2f}, Pr: {precision:2f}, Rc: {recall:2f}, f1: {f1:2f}")

# %%
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Taiwanese Bankruptcy/') # replace with your path to the dataset!
dataset_path = data_path/'data.csv'


# Get the dataset
bancruptcy = pd.read_csv(dataset_path)
display(bancruptcy.head())
print(bancruptcy.shape)
bancruptcy['Bankrupt?'].value_counts(normalize=True)

# %%
# DO: Stratified Split
target_name = 'Bankrupt?'
a = bancruptcy.copy()
# Create a dataset with imbalanced classes
X, y = a, a[target_name]
X = X.drop(columns=[target_name])
# make dataset even mode imbalanced = make 50% of 1s zeros

# get numeric features
numeric_features = X.select_dtypes(include=['number']).columns
print(y.value_counts(normalize=True))

# Perform a stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=17)

# scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

# Fit a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(y_test.value_counts(normalize=True))
print(y_train.value_counts(normalize=True))

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Stratified Split -: Acc: {accuracy:.2f}, Pr: {precision:2f}, Rc: {recall:2f}, f1: {f1:2f}")

# %%
# DON'T: not Stratified Split
target_name = 'Bankrupt?'
a = bancruptcy.copy()
# Create a dataset with imbalanced classes
X, y = a, a[target_name]
X = X.drop(columns=[target_name])
# make dataset even mode imbalanced = make 50% of 1s zeros

# get numeric features
numeric_features = X.select_dtypes(include=['number']).columns
print(y.value_counts(normalize=True))

# Perform a stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    #stratify=y,
                                                    random_state=17)

# scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

# Fit a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(y_test.value_counts(normalize=True))
print(y_train.value_counts(normalize=True))

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Stratified Split -: Acc: {accuracy:.2f}, Pr: {precision:2f}, Rc: {recall:2f}, f1: {f1:2f}")

# %% [markdown]
# # SMOTE

# %% [markdown]
# Please review! https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

# %%
# Good Example: SMOTE applied on Training Set Only
smote = SMOTE(random_state=42)
a = bancruptcy.copy()
target_name = 'Bankrupt?'
X, y = a, a[target_name]
X = X.drop(columns=[target_name])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=17)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_scaled = scaler.transform(X_test[numeric_features])

X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)


# Fit a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Stratified Split -: Acc: {accuracy:.2f}, Pr: {precision:2f}, Rc: {recall:2f}, f1: {f1:2f}")

# %% [markdown]
# ## SMOTE

# %% [markdown]
# ### for Balancing Data

# %%
# Oversample with SMOTE and random undersample for imbalanced dataset
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
 n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# summarize class distribution
counter = Counter(y)
print('class distribution',counter)

# define pipeline
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X, y = pipeline.fit_resample(X, y)

# summarize the new class distribution
counter = Counter(y)
print('new class distribution', counter)

# scatter plot of examples by class label
for label, _ in counter.items():
 row_ix = where(y == label)[0]
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

# %% [markdown]
# ### SMOTE oversampling for imbalanced classification

# %%
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
 n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# values to evaluate
k_values = [1, 2, 3, 4, 5, 6, 7]

for k in k_values:
 # define pipeline
 model = DecisionTreeClassifier()
 over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
 under = RandomUnderSampler(sampling_strategy=0.5)
 steps = [('over', over), ('under', under), ('model', model)]
 pipeline = Pipeline(steps=steps)

 # evaluate pipeline
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
 score = mean(scores)
 print('> k=%d, Mean ROC AUC: %.3f' % (k, score))

# %% [markdown]
# # Time-based split

# %%
pass

# %%
a = air_quality.copy()
a = a.fillna(-200)
train_size = int(len(a) * 0.8)
train_data = a[:train_size]
test_data = a[train_size:]

# Fit a simple model (Linear Regression on CO_GT)
model = LinearRegression()
model.fit(train_data[['PT08.S1(CO)']], train_data['CO(GT)'])
y_pred = model.predict(test_data[['PT08.S1(CO)']])

# Evaluate performance
mse = mean_squared_error(test_data['CO(GT)'], y_pred)
print(f"Time-Based Split - Good Example Mean Squared Error: {mse:.2f}")

# Plot the time-based data split
plt.figure(figsize=(10, 5))
plt.plot(train_data['Datetime'], train_data['CO(GT)'], label='Training Set', color='blue', alpha=0.6)
plt.plot(test_data['Datetime'], test_data['CO(GT)'], label='Test Set', color='red', alpha=0.6)
plt.title('Time-Based Split - Good Example (Chronological Order Maintained)')
plt.legend()
plt.show()

# %%
X_train, X_test, y_train, y_test = train_test_split(a[['PT08.S1(CO)']], a['CO(GT)'], test_size=0.2, random_state=42)

# Fit a simple model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Time-Based Split - Bad Example Mean Squared Error (Random Split): {mse:.2f}")

# Plotting the data (scatter to show randomness)
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Set')
plt.scatter(X_test, y_test, color='red', alpha=0.6, marker='x', label='Test Set')
plt.title('Time-Based Split - Bad Example (Random Split)')
plt.legend()
plt.show()

# %%
# 9. TimeSeriesSplit
# Good Example: TimeSeriesSplit
X_air_quality = a[['PT08.S1(CO)', 'NO2(GT)', 'T', 'RH']]
y_air_quality = a['CO(GT)']

tscv = TimeSeriesSplit(n_splits=5)
model = LinearRegression()
mse_values = []

for train_index, test_index in tscv.split(X_air_quality):
    X_train, X_test = X_air_quality.iloc[train_index], X_air_quality.iloc[test_index]
    y_train, y_test = y_air_quality.iloc[train_index], y_air_quality.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)
    print(f"TimeSeriesSplit - Mean Squared Error: {mse:.2f}")

overall_mse = np.mean(mse_values)
print(f"Overall Mean Squared Error from TimeSeriesSplit: {overall_mse:.2f}")

# %% [markdown]
# # Hierarchical split

# %% [markdown]
# In this more complex hierarchical split, we need to ensure that all data from the same school is either in the training or the test set. This means that if we put a school in the training set, all classrooms and students from that school must also be in the training set.
# 
# This hierarchical approach avoids leakage of school-level characteristics between training and testing.

# %%
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Sample dataset
data = {
    'District_ID': ['D1', 'D1', 'D1', 'D1', 'D2', 'D3'],
    'School_ID': ['S1', 'S1', 'S1', 'S2', 'S3', 'S4'],
    'Grade': ['G1', 'G1', 'G2', 'G1', 'G3', 'G2'],
    'Classroom_ID': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
    'Student_ID': ['ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6'],
    'District_Funding': ['High', 'High', 'High', 'High', 'Medium', 'Low'],
    'School_Quality': ['Excellent', 'Excellent', 'Excellent', 'Good', 'Average', 'Poor'],
    'Teacher_Experience': ['12 Years', '8 Years', '10 Years', '7 Years', '5 Years', '3 Years'],
    'Student_Age': [10, 10, 11, 10, 12, 11],
    'Student_Gender': ['F', 'M', 'F', 'M', 'F', 'M'],
    'Exam_Score': [92, 88, 85, 80, 75, 60]
}

df = pd.DataFrame(data)
display('Original df:', df)

# Setting up the GroupShuffleSplit for cluster splitting
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

# Split based on the 'Store_ID' to ensure all customers from the same store are in the same split
for train_idx, test_idx in gss.split(df, groups=df['School_ID']):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

# Displaying the results
print()
display("Training Set:", train_df)
print()
display("Testing Set:", test_df)

# %% [markdown]
# # KFold CV

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

target_name = 'Bankrupt?'
a = bancruptcy.copy()

# Create a dataset with imbalanced classes
X, y = a, a[target_name]
X = X.drop(columns=[target_name])

model = DecisionTreeClassifier()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=1),
    'recall': make_scorer(recall_score, zero_division=1),
    'f1': make_scorer(f1_score, zero_division=1)
}

# Perform cross-validation with multiple scoring metrics
cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring, verbose=2)

# Calculate and print average scores for each metric
print(f"Stratified Split - Acc: {cv_results['test_accuracy'].mean():.2f}, "
      f"Pr: {cv_results['test_precision'].mean():.2f}, "
      f"Rc: {cv_results['test_recall'].mean():.2f}, "
      f"f1: {cv_results['test_f1'].mean():.2f}")

# %% [markdown]
# # Leave-One-Out Cross-Validation (LOOCV)

# %%
from sklearn.model_selection import LeaveOneOut

model = LogisticRegression(max_iter=200)
loo = LeaveOneOut()
cv_scores = cross_val_score(model, X_iris, y_iris, cv=loo)
print(f"Leave-One-Out Cross-Validation - Good Example Mean LOOCV Score: {cv_scores.mean():.2f}")

# %% [markdown]
# # Bias - Variance

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load Ames Housing dataset
# paths
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Ames housing/') # replace with your path to the dataset!
fields_description_path = data_path/'data_description.txt'
dataset_path = data_path/'AmesHousing.csv'

# get fields description
with open(fields_description_path, 'r') as f:
    fields_description = f.read()

# get the dataset
df = pd.read_csv(dataset_path)
# Data Cleaning: Fill missing numeric values with zero for simplicity
df.fillna(0, inplace=True)

# Selecting features and the target
features = ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF']
X = df[features].values
y = df['SalePrice'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear regression model (High Bias)
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
y_pred_linear = linear_reg.predict(X_test_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)
mape_linear = mean_absolute_percentage_error(y_test, y_pred_linear)

# Decision Tree regression model (High Variance)
dt_reg = DecisionTreeRegressor(max_depth=15, random_state=42)
dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mape_dt = mean_absolute_percentage_error(y_test, y_pred_dt)

# Random Forest regression model (Ensemble to reduce Variance)
rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

# Plotting the Bias-Variance Tradeoff
plt.figure(figsize=(18, 6))

# Plot predictions for Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], y_test, s=20, alpha=0.5, color='blue', label='True Values')
plt.scatter(X_test[:, 0], y_pred_linear, s=20, alpha=0.5, color='darkorange', label='Linear Model Prediction')
plt.title('High Bias (Underfitting) - Linear Regression')
plt.xlabel('Living Area (sq ft)')
plt.ylabel('Sale Price')
plt.legend()

# Plot predictions for Decision Tree Regression
plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], y_test, s=20, alpha=0.5, color='blue', label='True Values')
plt.scatter(X_test[:, 0], y_pred_dt, s=20, alpha=0.5, color='darkorange', label='Decision Tree Prediction')
plt.title('High Variance - Decision Tree Regression')
plt.xlabel('Living Area (sq ft)')
plt.ylabel('Sale Price')
plt.legend()

# Plot predictions for Random Forest Regression
plt.subplot(1, 3, 3)
plt.scatter(X_test[:, 0], y_test, s=20, alpha=0.5, color='blue', label='True Values')
plt.scatter(X_test[:, 0], y_pred_rf, s=20, alpha=0.5, color='darkorange', label='Random Forest Prediction')
plt.title('Reduced Variance - Random Forest Regression')
plt.xlabel('Living Area (sq ft)')
plt.ylabel('Sale Price')
plt.legend()

plt.tight_layout()
plt.show()

print(f"MSE {mse_linear} , MAPE {mape_linear} for Linear Regression: ")
print(f"MSE {mse_dt} , MAPE {mape_dt} for Decision Tree Regression: ")
print(f"MSE {mse_rf} , MAPE {mape_rf} for Random Forest Regression: ")

# %%
pass

# %%
pass

# %%
pass
