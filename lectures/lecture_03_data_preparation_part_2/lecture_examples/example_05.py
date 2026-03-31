# /// script
# source-notebook = "example_05.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Data Leakage Examples

# %%
# connect to google drive
# NOTE: Colab-only import commented for local script use: from google.colab import drive
drive.mount('/content/drive')

# %%
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Ames housing/')
dataset_path = data_path/'AmesHousing.csv'
df = pd.read_csv(dataset_path)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from math import sqrt

# %% [markdown]
# # Example 1: Train-Test Contamination (Leaking future information during normalization)

# %%
# Splitting dataset into training and test sets
df = pd.read_csv(dataset_path)

# Incorrect: Using entire dataset to fit the scaler (including test data)
scaler = StandardScaler()
scaler.fit(df[['Lot Frontage', 'SalePrice']])  # Leaking information from test set

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Correct: Fit scaler only on training data
scaler.fit(train_df[['Lot Frontage', 'SalePrice']])

# Normalize features using only training data
X_train_scaled = scaler.transform(train_df[['Lot Frontage', 'SalePrice']])
X_test_scaled = scaler.transform(test_df[['Lot Frontage', 'SalePrice']])

# %% [markdown]
# # Example 2: Target Leakage (Feature containing information about the target)

# %%
# Including 'SalePrice' in the feature set, which is also the target variable
df = pd.read_csv(dataset_path)
X = df[['Lot Frontage', 'SalePrice', 'MS SubClass']]  # Incorrect, 'SalePrice' is the target
y = df['SalePrice']

# Correct: Remove target variable from features
X_correct = df[['Lot Frontage', 'MS SubClass']]

# %% [markdown]
# # Example 3: Time-Series Leakage (Using future information)

# %% [markdown]
# ## Assume we have a time-series dataset where 'YearBuilt' could influence the prices in future years
# ## Incorrect: Including 'YearBuilt' for predicting future housing prices without adjusting for the timeline

# %%
# Adding an example with a hypothetical time-series split
df = pd.read_csv(dataset_path)
df['SaleYear'] = df['Yr Sold']  # Assume 'YrSold' represents the year of sale

# Incorrect: Using future information for prediction
# Here, we are using 'SaleYear' and 'YearBuilt' without respecting the temporal order
future_df = df.sort_values(by='SaleYear')
X_future = future_df[['Year Built', 'Lot Frontage', 'MS SubClass']]  # Incorrect, contains future information
y_future = future_df['SalePrice']

# Splitting without respecting time-series nature
train_df_future = future_df.iloc[:int(0.8 * len(future_df))]
test_df_future = future_df.iloc[int(0.8 * len(future_df)):]  # Data from the future is leaking into training

# Features and target
X_train_time = train_df_future[['Year Built', 'Lot Frontage', 'MS SubClass']]
y_train_time = train_df_future['SalePrice']
X_test_time = test_df_future[['Year Built', 'Lot Frontage', 'MS SubClass']]
y_test_time = test_df_future['SalePrice']

# Train model without time-series leakage
model_time = RandomForestRegressor(random_state=42)
model_time.fit(X_train_time, y_train_time)

# Predicting and evaluating
predictions_time = model_time.predict(X_test_time)
mae_time = mean_absolute_error(y_test_time, predictions_time)
mape_time = mean_absolute_percentage_error(y_test_time, predictions_time)
rmse_time = sqrt(mean_squared_error(y_test_time, predictions_time))
print(f"Mean Absolute Error (with Time-Series Leakage): {mae_time}")
print(f"Mean Absolute Percentage Error (with Time-Series Leakage): {mape_time}")
print(f"Root Mean Squared Error (with Time-Series Leakage): {rmse_time}")

# %%
# Correct: Ensure training data is prior to test data
train_df_time = future_df[future_df['SaleYear'] < 2010]  # Assume data before 2010 is used for training
test_df_time = future_df[future_df['SaleYear'] >= 2010]

# Features and target
X_train_time = train_df_time[['Year Built', 'Lot Frontage', 'MS SubClass']]
y_train_time = train_df_time['SalePrice']
X_test_time = test_df_time[['Year Built', 'Lot Frontage', 'MS SubClass']]
y_test_time = test_df_time['SalePrice']

# Train model without time-series leakage
model_time = RandomForestRegressor(random_state=42)
model_time.fit(X_train_time, y_train_time)

# Predicting and evaluating
predictions_time = model_time.predict(X_test_time)
mae_time = mean_absolute_error(y_test_time, predictions_time)
mape_time = mean_absolute_percentage_error(y_test_time, predictions_time)
rmse_time = sqrt(mean_squared_error(y_test_time, predictions_time))
print(f"Mean Absolute Error (without Time-Series Leakage): {mae_time}")
print(f"Mean Absolute Percentage Error (without Time-Series Leakage): {mape_time}")
print(f"Root Mean Squared Error (without Time-Series Leakage): {rmse_time}")

# %% [markdown]
# # Example 4: Leakage from Derived Features (Improper feature engineering)

# %% [markdown]
# ## Creating a feature that directly uses target information

# %% [markdown]
# ## Incorrect: Creating a new feature that is derived from the target variable

# %%
train_df['Price_per_SqFt'] = train_df['SalePrice'] / train_df['Gr Liv Area']  # Leakage, as 'SalePrice' is the target

# %% [markdown]
# ## Correct: Ensure that derived features do not use target variable information

# %%
# Creating a feature based on non-target columns
train_df['LotFrontage_per_MSSubClass'] = train_df['Lot Frontage'] / (train_df['MS SubClass'] + 1)

# %% [markdown]
# # Example 5: Data Leakage from Data Imputation

# %% [markdown]
# ## Incorrect: Imputing missing values using the entire dataset (including test set)

# %%
imputer = SimpleImputer(strategy='mean')
X_transformed = imputer.fit(df[['Lot Frontage']])  # Leaking test set information during imputation

# %% [markdown]
# ## Correct: Fit imputer only on training data

# %%
# Correct: Fit imputer only on training data
imputer.fit(train_df[['Lot Frontage']])

# Model Training and Evaluation (with Corrected Dataset)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Feature and target separation
X_train = train_df[['Lot Frontage', 'MS SubClass']]
y_train = train_df['SalePrice']
X_test = test_df[['Lot Frontage', 'MS SubClass']]
y_test = test_df['SalePrice']

# Create an imputer for missing values, fitting it only on the training data
imputer = SimpleImputer(strategy='mean')

# Fit imputer on the training data and transform both training and test data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Normalizing features using only training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Training the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predicting and evaluating
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
rmse = sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Percentage Error: {mape}")
print(f"Root Mean Squared Error: {rmse}")

# %%
pass
