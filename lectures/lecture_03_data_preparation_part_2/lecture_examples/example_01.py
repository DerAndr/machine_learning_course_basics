# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Feature Selection
# 
# ## 1. [FILTER METHODS](#scrollTo=BLVzn55ihVaf)
# Filter methods use statistical techniques to assess the relationship between input features and the target variable, selecting those with the strongest associations.
# - **Examples**: Correlation Matrix, Chi-square test, Weight of Evidence (WoE), ANOVA, Mutual information
# 
# ## 2. [WRAPPING METHODS](#scrollTo=2W0-KlE09dTC&line=1&uniqifier=1)
# Wrapping methods evaluate multiple feature subsets by training and testing a machine learning model, selecting features based on model performance.
# - **Examples**: Forward Selection, Backward Elimination, Recursive Feature Elimination (RFE), Exhaustive Feature Searchn
# 
# ## 3. [EMBEDDED METHODS](#scrollTo=NbyyGujD9kK4&line=1&uniqifier=1)
# Embedded methods perform feature selection during the process of model training, using the inherent properties of specific algorithms.
# - **Examples**: LASSO (L1 regularization), Ridge (L2 regularization), Decision Trees (feature importance)

# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# %% [markdown]
# # Dataset Loading

# %% [markdown]
# This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). The prediction task is to determine whether a person makes over $50K a year.

# %%
from sklearn.datasets import fetch_openml

# Load the "Adult (Census Income)" dataset
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame

# Handle categorical encoding for categorical features
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})

target_name = "class"
df[target_name] = df[target_name].map({'<=50K': 0, '>50K': 1}).astype('int')
target = df[target_name]

# %%
df.head()

# %%
print(df.shape)

# %% [markdown]
# 1. **age**: (continuous, positive integer) The age of the individual.
# 2. **workclass**: (categorical, 9 distinct values) Simplified employment status of an individual
# 3. **fnlwgt**: (continuous, positive integer) Final weight of the record. Basically interpret as the number of
# people represented by this row.
# 4. **education-num**: (categorical, 13 distinct values) The education level, in ascending positive integer
# value.
# 5. **education**: (categorical, 13 distinct values) The education level. Note that for simplicity, we will
# ignore this column because of the existence of education-num column.
# 6. **marital-status**: (categorical, 7 distinct values) Marital status of a person.
# 7. **occupatioin**: (categorical, 15 distinct values) Rough category of the occupation.
# 8. **relationship**: (categorical, 6 distinct values) Relationship in terms of the family. Note that we ignore
# this column since the semantic is somewhat covered by marital-status and gender.
# 9. **race**: (categorical, 5 distinct values) Race of the person.
# 10. **sex**: (boolean) gender at-birth.
# 11. **capital-gain**: (continuous) Dollar gain of capital.
# 12. **capital-loss**: (continuous) Dollar loss of capital.
# 13. **hours-per-week**: (continous positive integer) Working hours per week.
# 14. **native-country**: (categorical, 41 distinct values) Country at birth.
# 15. **class**: ≥ 50K or < 50K (per year).

# %%
df.isnull().sum()

# %%
df.describe(include='all')

# %%
# Define Categorical and Numerical columns
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Step 2: Handle missing values in categorical columns by filling with 'no data'
for col in categorical_cols:
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.add_categories(['no data'])
df[categorical_cols] = df[categorical_cols].fillna('no data')

# %%
df.isnull().sum()

# %% [markdown]
# # Create Highly Correlated Features for Illustration

# %%
# Create a highly correlated feature based on 'education-num'
df['education-num-high-corr'] = df['education-num'] * 1.05 + np.random.normal(0, 0.5, size=len(df))

# Create another highly correlated feature based on 'age'
df['age-high-corr'] = df['age'] * 1.1 + np.random.normal(0, 1, size=len(df))

# Create a highly correlated feature with the target
df['high-corr-target'] = df['class'] * 10 + np.random.normal(0, 1, size=len(df))

# Create highly dependent categorical features
df['workclass_high_dep'] = df['workclass'].apply(lambda x: 'Private' if x == 'Private' else 'Other')

# Create a non-linear relationship between 'education-num' and a new feature
df['education-num-nonlinear'] = np.sin(df['education-num'] * 0.5) + df['education-num'] * 0.3 + np.random.normal(0, 0.2, size=len(df))

# %%
numerical_cols

# %%
numerical_cols.extend(['education-num-high-corr', 'age-high-corr', 'high-corr-target', 'education-num-nonlinear'])
categorical_cols.extend(['workclass_high_dep'])

# %% [markdown]
# # FILTER METHODS

# %% [markdown]
# ## Correlation coefficients, Correlation Matrix

# %% [markdown]
# A correlation matrix shows the relationship (linear dependence) between numerical features. Features with high correlation (near 1 or -1) are considered strongly related, and one may be excluded to reduce multicollinearity.

# %% [markdown]
# We'll use [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) by default (which assumes a linear relationship between features). The correlation coefficient varies between -1 and 1:
# *   1 means a perfect positive correlation.
# *   -1 means a perfect negative correlation
# *   0 means no linear correlation.

# %% [markdown]
# ### Numerical Pearson correlation

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import time

start_t = time.time()
# Calculate correlation matrix for numerical features
corr_matrix = df[numerical_cols].corr(method='pearson')
corr_matrix = corr_matrix.mask(np.triu(np.ones_like(corr_matrix, dtype=bool)))
end_t = time.time()
display(corr_matrix)

print(f"Time taken: {end_t - start_t:.2f} seconds")
# Plot correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1) # Pay attention to min and max!
plt.title('Pearson Correlation Matrix for Numerical Features')
plt.show()

# %% [markdown]
# 
# ### Numerical [Spearman Correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
# Measures monotonic relationships (captures both linear and non-linear monotonic relationships).

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import time

start_t = time.time()
# Calculate correlation matrix for numerical features
corr_matrix = df[numerical_cols].corr(method='spearman')
corr_matrix = corr_matrix.mask(np.triu(np.ones_like(corr_matrix, dtype=bool)))
end_t = time.time()
print(f"Time taken: {end_t - start_t:.2f} seconds")
display(corr_matrix)

# Plot correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1) # Pay attention to min and max!
plt.title('Spearman Correlation Matrix for Numerical Features')
plt.show()

# %% [markdown]
# ### Numerical [Kendall Correlation](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)
# Measures ordinal association between two variables, capturing monotonic relationships.

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import time

start_t = time.time()
# Calculate correlation matrix for numerical features
corr_matrix = df[numerical_cols].corr(method='kendall')
corr_matrix = corr_matrix.mask(np.triu(np.ones_like(corr_matrix, dtype=bool)))
end_t = time.time()
print(f"Time taken: {end_t - start_t:.2f} seconds")
display(corr_matrix)

# Plot correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1) # Pay attention to min and max!
plt.title('Kendall Correlation Matrix for Numerical Features')
plt.show()

# %% [markdown]
# ### Numerical - Binary Point-biserial correlation

# %% [markdown]
# To calculate the correlation between numerical features and a binary target (like <=50K or >50K in the Adult dataset), let us use [Point-Biserial Correlation] (https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient)

# %%
from scipy.stats import pointbiserialr

threshold = 0.7
# Calculate Point-Biserial Correlation between each numerical feature and the binary target
start_t = time.time()
for col in numerical_cols:
    prefix = ''
    corr, p_value = pointbiserialr(df[col], df['class'])
    if abs(corr) > threshold:
        prefix = '!!! ATTENTION !!!'
    print(f"{prefix} Correlation between {col} and target: {corr:.4f} (p-value: {p_value:.4f})")

# %% [markdown]
# For each numerical feature, this code outputs the Point-Biserial Correlation coefficient (between -1 and 1) and the p-value. The closer the correlation is to -1 or 1, the stronger the relationship between the feature and the binary target.

# %% [markdown]
# How to Interpret:
# *  **A positive correlation** means that as the numerical feature increases, the likelihood of the positive class (>50K) increases.
# *  **A negative correlation** means that as the numerical feature increases, the likelihood of the positive class decreases.
# *  **p-value:** If the p-value is small (typically < 0.05), the correlation is statistically significant.

# %% [markdown]
# ### Categorical - Categorican Cramér's V
# Use [Cramér's V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) to assess correlations between **categorical features**.

# %%
from scipy.stats import chi2_contingency

# Function to calculate Cramér's V
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# Example: Calculate Cramér's V
for i, col1 in enumerate(categorical_cols):
    for col2 in categorical_cols[i+1:]:
        cramers_v_val = cramers_v(df[col1], df[col2])
        print(f"Cramér's V between {col1} and {col2}: {cramers_v_val:.4f}")

# %% [markdown]
# ## [Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)

# %% [markdown]
# Mutual Information quantifies the amount of information gained about the target from knowing the feature values. It works well for both numerical and categorical features and is useful for non-linear relationships.
# 
# Higher MI values mean the feature provides more information about the target, which suggests it could be more important for predictive tasks. However, **MI is not bounded between -1 and 1** like correlation and works well for detecting non-linear relationships.

# %%
from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information between numerical features and the target
mutual_info = mutual_info_classif(df[numerical_cols], df['class'])

# Print mutual information scores
for i, col in enumerate(numerical_cols):
    print(f"Mutual Information between {col} and target: {mutual_info[i]:.4f}")

# %% [markdown]
# ## [Weight of Evidence, WoE](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)

# %% [markdown]
# **Interpretation of WoE:**
# * Positive WoE indicates that a category is associated with a higher likelihood of the target event (in this case, income > 50K).
# * Negative WoE suggests that a category is associated with a lower likelihood of the target event.
# * A WoE value of 0 means that the category does not provide any predictive power.
# 
# **Interpretation of IV:**
# * IV < 0.02: Not Predictive
# * 0.02 - 0.1: Weak Predictor
# * 0.1 - 0.3: Medium Predictor
# * 0.3 - 0.5: Strong Predictor
# * IV > 0.5: Very Strong Predictor (possible overfitting or data leakage)

# %%
# Function to calculate WoE and Information Value (IV) using numpy
def calc_woe_iv(df, feature, target):
    # Calculate the cross-tabulation of the feature against the target
    crosstab = pd.crosstab(df[feature], df[target], dropna=False).astype(float)

    # Add total counts
    crosstab['total'] = crosstab[0] + crosstab[1]

    # Calculate event and non-event rates
    crosstab['event_rate'] = crosstab[1] / crosstab[1].sum()
    crosstab['non_event_rate'] = crosstab[0] / crosstab[0].sum()

    # Handle division by zero and calculate WoE
    crosstab['woe'] = np.log((crosstab['event_rate'] + 1e-10) / (crosstab['non_event_rate'] + 1e-10))

    # Calculate Information Value (IV)
    crosstab['iv'] = (crosstab['event_rate'] - crosstab['non_event_rate']) * crosstab['woe']
    iv = crosstab['iv'].sum()

    return crosstab[['event_rate', 'non_event_rate', 'woe', 'iv']], iv

# Example: Calculate WoE and Information Value (IV) for 'education'
woe_education, iv_education = calc_woe_iv(df, 'education', 'class')
print("WoE for education:", woe_education)
print("Information Value (IV) for education: {:.4f}".format(iv_education))

# %% [markdown]
# ## [ANOVA - Analysis of Variance](https://en.wikipedia.org/wiki/Analysis_of_variance)

# %% [markdown]
# The ANOVA F-statistic tests whether the means of a numerical feature significantly differ between two groups (in this case, income levels <=50K vs >50K). Here’s how to interpret the results:
# 
# * **F-statistic:**
# 
# A high F-statistic indicates that there is a significant difference in the mean values of the feature across the groups, implying that the feature may have predictive power in determining the target variable.
# A low F-statistic suggests no significant difference in means, implying the feature might not be useful for prediction.
# * **p-value:**
# 
# If the p-value is less than 0.05, it means the feature has a significant effect on the target, i.e., the means of the groups are significantly different.
# A high p-value (e.g., > 0.05) means there is no significant difference between the means for the groups, implying the feature might not be relevant for predicting the target.

# %%
from sklearn.feature_selection import f_classif

# Calculate ANOVA F-statistic for numerical features
f_stat, p_values = f_classif(df[numerical_cols], df['class'])
for i, col in enumerate(numerical_cols):
    print(f"ANOVA F-statistic for {col}: {f_stat[i]:.4f}, p-value: {p_values[i]:.4f}")

# Create a structured numpy array with feature names and F-statistic scores
dtype = [('feature', 'U50'), ('score', 'f8')]  # Define data types for the structured array
features_scores = np.array(list(zip(numerical_cols, f_stat)), dtype=dtype)

# Sort features by F-statistic scores in descending order
features_scores = np.sort(features_scores, order='score')[::-1]

# Print the sorted features with their scores in a table format
print("\nSorted features by ANOVA F-statistic:")
print(f"{'Feature Name':<30}{'F-Statistic Score':>20}")

# Print each feature with fixed-length formatting
for row in features_scores:
    print(f"{row['feature']:<30}{row['score']:>20.4f}")

# %% [markdown]
# ## [Chi-Square Test](https://en.wikipedia.org/wiki/Chi-squared_test)

# %% [markdown]
# The Chi-Square test measures the statistical significance of the association between each categorical feature and the target variable.
# 
# **The higher the Chi-Square score, the stronger the association between the feature and the target.**
# 
# However, there's no absolute threshold for the Chi-Square score itself because the meaningfulness of the score depends on the number of observations and categories in the feature.

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
import numpy as np

# Encode categorical features
df_encoded = df[categorical_cols].apply(LabelEncoder().fit_transform)

# Chi-square test for association with the target
chi_scores, p_values = chi2(df_encoded, df['class'])
for i, col in enumerate(categorical_cols):
    print(f"Chi-square score for {col}: {chi_scores[i]:.4f}, p-value: {p_values[i]:.4f}")

# Create a structured numpy array with feature names and Chi-square scores
dtype = [('feature', 'U50'), ('score', 'f8')]  # Define data types for the structured array
features_scores = np.array(list(zip(categorical_cols, chi_scores)), dtype=dtype)

# Sort features by Chi-square scores in descending order
features_scores = np.sort(features_scores, order='score')[::-1]

# Print the sorted features with their scores in a table format
print("\nSorted features by Chi-square score:")
print(f"{'Feature Name':<30}{'Chi-square Score':>20}")

# Print each feature with fixed-length formatting
for row in features_scores:
    print(f"{row['feature']:<30}{row['score']:>20.4f}")

# %% [markdown]
# The same happeins if we use SelectKBest with score_func=chi2

# %%
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
df_encoded = df[categorical_cols].apply(LabelEncoder().fit_transform)

# Apply SelectKBest to select top k features
k = 5  # You can adjust k to select the desired number of top features
selector = SelectKBest(score_func=chi2, k=k)
selector.fit(df_encoded, df['class'])

# Get the selected features
selected_features = [col for col, is_selected in zip(categorical_cols, selector.get_support()) if is_selected]
print("Top", k, "selected categorical features based on Chi-Square scores:", selected_features)

# %%
pass

# %%
pass

# %%
pass

# %% [markdown]
# # WRAPPING METHODS

# %% [markdown]
# ## [Forward Selection](https://scikit-learn.org/dev/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
import time

# Prepare feature and target sets
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Define the model
model = DecisionTreeClassifier()

# Forward selection
# Start the timer
start_time = time.time()
sfs = SequentialFeatureSelector(estimator=model, n_features_to_select='auto', direction='forward', cv=3)
sfs = sfs.fit(X, y) # You should use train split to find best features, not the whole dataset!

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Selected features
selected_features_forward = list(X.columns[sfs.get_support()])
print("Selected features by Forward Selection:", selected_features_forward)

# %% [markdown]
# ## [Backward Elimination](https://scikit-learn.org/dev/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
import time

# Prepare feature and target sets
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Define the model
model = DecisionTreeClassifier()

# Forward selection
# Start the timer
start_time = time.time()
sfs = SequentialFeatureSelector(estimator=model, n_features_to_select='auto', direction='backward', cv=3)
sfs = sfs.fit(X, y) # You should use train split to find best features, not the whole dataset!

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Selected features
selected_features_backward = list(X.columns[sfs.get_support()])
print("Selected features by Backward Selection:", selected_features_backward)

# %% [markdown]
# ## [Recursive Feature Elimination, RFE](https://scikit-learn.org/dev/modules/generated/sklearn.feature_selection.RFE.html)

# %%
from sklearn.feature_selection import RFE

# Prepare feature and target sets
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Define the model
model = DecisionTreeClassifier()

start_time = time.time()
# Recursive Feature Elimination
rfe = RFE(estimator=model, n_features_to_select=10)
rfe = rfe.fit(X, y) # You should use train split to find best features, not the whole dataset!
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Selected features
selected_features_rfe = X.columns[rfe.support_].tolist()
print("Selected features by Recursive Feature Elimination (RFE):", selected_features_rfe)

# %% [markdown]
# ## [Exhaustive Feature Search](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/)

# %%
import math
subset_size = 5
full_feature_set = 10
print(f'The number of possible subsets of size 4 for a set of {full_feature_set} features is')
math.comb(full_feature_set, subset_size)

# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from mlxtend.feature_selection import ExhaustiveFeatureSelector

# Prepare feature and target sets
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Define the model
model = DecisionTreeClassifier(random_state=42, max_depth=10)

start_time = time.time()
# Exhaustive Feature Search
features = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'hours-per-week']
# for illustration! Full dataset takes math.comb(full_feature_set, subset_size) * CV iterations!
# math.comb(n, k) = n! / (k! * (n - k)!)
# For big datasets it can date DAYS!
efs = ExhaustiveFeatureSelector(model, min_features=5, max_features=5, scoring='accuracy', print_progress=True, cv=3, n_jobs=-1)
efs = efs.fit(X[features], y) # You should use train split to find best features, not the whole dataset!
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Selected features
selected_features_exhaustive = list(efs.best_feature_names_)
print("Selected features by Exhaustive Feature Search:", selected_features_exhaustive)

# %% [markdown]
# # EMBEDDED METHODS

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Prepare feature and target sets
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# Select only the selected features
X = X[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LASSO (L1 regularization)
start_time = time.time()
lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
lasso.fit(X_scaled, y)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Extracting the feature importance
lasso_coef = np.array(lasso.coef_).flatten()
selected_features_lasso = [feature for feature, coef in zip(X.columns, lasso_coef) if coef != 0]

print("Selected features by LASSO (L1 Regularization):")
for feature, coef in zip(X.columns, lasso_coef):
    print(f"{feature}: {coef:.4f}")

# Create a DataFrame for easy sorting and visualization
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_coef
})

# Filter out zero coefficients and sort by absolute values
feature_importance = feature_importance[feature_importance['Coefficient'] != 0]
feature_importance = feature_importance.reindex(feature_importance['Coefficient'].abs().sort_values(ascending=False).index)

# Plotting the sorted features by importance
plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='navy')
plt.xlabel('LASSO Coefficient')
plt.title('Feature Importance by LASSO (L1 Regularization)')
plt.gca().invert_yaxis()  # To have the highest coefficient at the top

# Add values to the bars
for bar in bars:
    plt.text(
        bar.get_width(),  # x-coordinate of the text (end of the bar)
        bar.get_y() + bar.get_height() / 2,  # y-coordinate of the text (middle of the bar)
        f'{bar.get_width():.4f}',  # Text value with 4 decimal places
        va='center',  # Vertical alignment of text
        ha='left' if bar.get_width() > 0 else 'right'  # Horizontal alignment
    )

plt.show()

# %%
# Prepare feature and target sets
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# Select only the selected features
X = X[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge (L2 regularization)
start_time = time.time()
ridge = LogisticRegression(penalty='l2', solver='saga', max_iter=1000, random_state=42)
ridge.fit(X_scaled, y)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Extracting the feature importance
ridge_coef = np.array(ridge.coef_).flatten()

print("\nFeature importance by Ridge (L2 Regularization):")
for feature, coef in zip(X.columns, ridge_coef):
    print(f"{feature}: {coef:.4f}")

# Create a DataFrame for easy sorting and visualization
feature_importance_ridge = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': ridge_coef
})

# Sort by absolute values of the coefficients
feature_importance_ridge = feature_importance_ridge.reindex(feature_importance_ridge['Coefficient'].abs().sort_values(ascending=False).index)

# Plotting the sorted features by importance
plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance_ridge['Feature'], feature_importance_ridge['Coefficient'], color='navy')
plt.xlabel('Ridge Coefficient')
plt.title('Feature Importance by Ridge (L2 Regularization)')
plt.gca().invert_yaxis()  # To have the highest coefficient at the top

# Add values to the bars
for bar in bars:
    plt.text(
        bar.get_width(),  # x-coordinate of the text (end of the bar)
        bar.get_y() + bar.get_height() / 2,  # y-coordinate of the text (middle of the bar)
        f'{bar.get_width():.4f}',  # Text value with 4 decimal places
        va='center',  # Vertical alignment of text
        ha='left' if bar.get_width() > 0 else 'right'  # Horizontal alignment
    )

plt.show()

# %%
from sklearn.tree import DecisionTreeClassifier

# Prepare feature and target sets
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
X = X[features]


# Train Decision Tree model
start_time = time.time()
tree_model = DecisionTreeClassifier(random_state=42, max_depth=10)
tree_model.fit(X, y)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Extract feature importances
feature_importances = tree_model.feature_importances_

print("\nFeature importance by Decision Tree:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")


# Create a DataFrame for easy sorting and visualization
feature_importance_tree_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort by feature importance
feature_importance_tree_df = feature_importance_tree_df.sort_values(by='Importance', ascending=False)

# Plotting the sorted features by importance
plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance_tree_df['Feature'], feature_importance_tree_df['Importance'], color='navy')
plt.xlabel('Feature Importance')
plt.title('Feature Importance by DecisionTreeClassifier')
plt.gca().invert_yaxis()  # To have the highest importance at the top

# Add values to the bars
for bar in bars:
    plt.text(
        bar.get_width(),  # x-coordinate of the text (end of the bar)
        bar.get_y() + bar.get_height() / 2,  # y-coordinate of the text (middle of the bar)
        f'{bar.get_width():.4f}',  # Text value with 4 decimal places
        va='center',  # Vertical alignment of text
        ha='left' if bar.get_width() > 0 else 'right'  # Horizontal alignment
    )

plt.show()

# %%
pass
