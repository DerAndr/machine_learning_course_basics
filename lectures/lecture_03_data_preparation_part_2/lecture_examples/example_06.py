# /// script
# source-notebook = "example_06.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Pipelines

# %%
# connect to google drive
# NOTE: Colab-only import commented for local script use: from google.colab import drive
drive.mount('/content/drive')

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Ames housing/')
dataset_path = data_path/'AmesHousing.csv'
df = pd.read_csv(dataset_path)

# %%
# Feature engineering
df['Age'] = 2024 - df['Year Built']
df['TotalArea'] = df['Lot Frontage'].fillna(0) * df['Lot Area']
df['IsNew'] = np.where(df['Year Built'] > 2010, 1, 0)
df['AreaPerYear'] = df['TotalArea'] / (df['Age'] + 1)
df['FrontageToAreaRatio'] = df['Lot Frontage'].fillna(0) / (df['Lot Area'] + 1)

# Features and target
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the preprocessing steps
numeric_features = ['Lot Frontage', 'Lot Area', 'Year Built', 'Age', 'TotalArea', 'AreaPerYear', 'FrontageToAreaRatio']
categorical_features = ['MS SubClass', 'MS Zoning']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('feature_selection', SelectKBest(score_func=f_regression, k='all'))
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for both numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Function to calculate regression metrics
def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }

# Linear Regression Model
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the Linear Regression model
linear_model.fit(X_train, y_train)

# Predict and calculate metrics for Linear Regression
y_pred_linear = linear_model.predict(X_test)
linear_metrics = calculate_regression_metrics(y_test, y_pred_linear)

print("Linear Regression Metrics:")
for metric, value in linear_metrics.items():
    print(f'{metric}: {value:.2f}')

# Save the Linear Regression model
joblib.dump(linear_model, 'linear_regression_model.pkl')

# Decision Tree Regressor Model with GridSearchCV
decision_tree_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

param_grid_dt = {
    'regressor__max_depth': [3, 5, 7, None],
    'regressor__min_samples_split': [2, 5, 10]
}

grid_search_dt = GridSearchCV(decision_tree_model, param_grid_dt, cv=5, scoring='r2', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

# Predict and calculate metrics for Decision Tree
y_pred_dt = grid_search_dt.best_estimator_.predict(X_test)
decision_tree_metrics = calculate_regression_metrics(y_test, y_pred_dt)

print("Decision Tree Metrics:")
for metric, value in decision_tree_metrics.items():
    print(f'{metric}: {value:.2f}')

# Save the best Decision Tree model
joblib.dump(grid_search_dt.best_estimator_, 'best_decision_tree_model.pkl')

# %%
pass
