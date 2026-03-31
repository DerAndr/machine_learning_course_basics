# /// script
# source-notebook = "example_03.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %%
# connect to google drive
# NOTE: Colab-only import commented for local script use: from google.colab import drive
drive.mount('/content/drive')

# %%
import pandas as pd
import numpy as np
import scipy
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

# %%
# paths
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Ames housing/') # replace with your path to the dataset!
fields_description_path = data_path/'data_description.txt'
dataset_path = data_path/'AmesHousing.csv'

# get fields description
with open(fields_description_path, 'r') as f:
    fields_description = f.read()

# get the dataset
df = pd.read_csv(dataset_path, na_values='NA', index_col='Order')
df.head(10)

# %% [markdown]
# # Category Encoding Techniques Demonstration on Toy Dataset

# %%
# NOTE: notebook magic commented for local script use: ! pip install category_encoders

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set colorblind-friendly palette
sns.set_palette("colorblind")

# %% [markdown]
# ## Toy dataset for category encoding techniques

# %%
toy_data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'A', 'C', 'A', 'A', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B']
})
print("Toy Dataset for Category Encoding:")
target_data = pd.Series([5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3], name='Target')
toy_data

# %%
toy_data.value_counts()

# %% [markdown]
# ## One-Hot Encoding

# %%
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded_data = one_hot_encoder.fit_transform(toy_data[['Category']])
df_one_hot_encoded = pd.DataFrame(one_hot_encoded_data, columns=one_hot_encoder.get_feature_names_out(['Category']))
print("One-Hot Encoded values:")
df_one_hot_encoded['original'] = toy_data['Category']
df_one_hot_encoded.head(10)

# %% [markdown]
# ### Plot One-Hot Encoding

# %%
plt.figure(figsize=(10, 4))
sns.heatmap(df_one_hot_encoded[['Category_A', 'Category_B',	'Category_C']], annot=True, cbar=False)
plt.title('One-Hot Encoding Heatmap')
plt.show()

# %% [markdown]
# ## Label Encoding

# %%
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_label_encoded = toy_data.copy()
df_label_encoded['Category'] = label_encoder.fit_transform(toy_data['Category'])
print("Label Encoded values:")
df_label_encoded

# combine with original to show mapping
df_label_encoded['Original'] = toy_data['Category']
df_label_encoded.head(5)

# %% [markdown]
# ## Ordinal Encoding

# %%
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder(categories=[['A', 'B', 'C']])
df_ordinal_encoded = toy_data.copy()
df_ordinal_encoded['Category'] = ordinal_encoder.fit_transform(toy_data[['Category']])
print("Ordinal Encoded values:")
df_ordinal_encoded.head(5)

# %% [markdown]
# ## Binary Encoding

# %%
import category_encoders as ce

binary_encoder = ce.BinaryEncoder(cols=['Category'])
df_binary_encoded = binary_encoder.fit_transform(toy_data)
print("Binary Encoded values:")

# show original
df_binary_encoded['Original'] = toy_data['Category']
df_binary_encoded.head(5)

# %% [markdown]
# ## Frequency Encoding

# %%
frequency_encoded_data = toy_data.copy()
frequency_map = toy_data['Category'].value_counts().to_dict()
frequency_encoded_data['Category'] = frequency_encoded_data['Category'].map(frequency_map)
print("Frequency Encoded values:")

frequency_encoded_data['Original'] = toy_data['Category']
frequency_encoded_data.head(10)

# %%
toy_data.value_counts()

# %% [markdown]
# ## Target Encoding

# %%
from category_encoders import TargetEncoder

target_encoder = TargetEncoder(cols=['Category'])
df_target_encoded = target_encoder.fit_transform(toy_data, target_data)
print("Target Encoded values:")
df_target_encoded['Original'] = toy_data['Category']
df_target_encoded.head(10)

# %% [markdown]
# ## Target Encoding - Custom implementation

# %%
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

def cross_validated_target_encode(df, target, column, n_splits=5):
    # Add target to the DataFrame for grouping operations
    df_with_target = df.copy()
    df_with_target['Target'] = target.values
    encoded_values = pd.Series(index=df.index, dtype=np.float64)
    global_mean = target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(df):
        train_fold, val_fold = df_with_target.iloc[train_index], df_with_target.iloc[val_index]

        # Compute target means from the train fold
        means = train_fold.groupby(column)['Target'].mean()

        # Apply to the validation fold
        encoded_values.iloc[val_index] = val_fold[column].map(means).fillna(global_mean)

    return encoded_values

# Apply cross-validated target encoding
toy_data['Category_Encoded_CV'] = cross_validated_target_encode(toy_data, target_data, 'Category')

print("Toy Dataset with Cross-Validated Target Encoded values:")
print(toy_data)

print("\nUnique encoded values for each category:")
toy_data.groupby('Category')['Category_Encoded_CV'].mean()

# %%
# Worth taking a look: https://contrib.scikit-learn.org/category_encoders/

# %% [markdown]
# ------------

# %% [markdown]
# # Transformation and Encoding

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set colorblind-friendly palette
sns.set_palette("colorblind")

# %%
categorical_column = 'Garage Type'
numeric_column = 'Lot Frontage'

# %% [markdown]
# ## Scaling

# %% [markdown]
# ### Min-Max Scaling (Normalization)

# %%
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

min_max_scaler = MinMaxScaler()
df_min_max_scaled = df.copy()
df_min_max_scaled[numeric_column] = min_max_scaler.fit_transform(df[[numeric_column]])
print("Min-Max Scaled values:")

df_min_max_scaled[numeric_column+'_original'] = df[numeric_column]
df_min_max_scaled[[numeric_column, numeric_column+'_original']].head(10)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Min-Max Scaling: Original vs Scaled on vertically stacked subplots, each with its own x-axis
fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # Removed sharex=True

# Original data
sns.histplot(df[numeric_column], color='navy', label='Original', kde=True, alpha=0.5, ax=axes[0])
axes[0].set_title('Original Data - Min-Max Scaling')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Min-Max Scaled data
sns.histplot(df_min_max_scaled[numeric_column], color='darkorange', label='Min-Max Scaled', kde=True, alpha=0.5, ax=axes[1])
axes[1].set_title('Min-Max Scaled Data')
axes[1].set_xlabel(numeric_column)
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Standardization

# %%
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
df_standard_scaled = df.copy()
df_standard_scaled[numeric_column] = standard_scaler.fit_transform(df[[numeric_column]])
print("Standardized values:")
df_standard_scaled[numeric_column+'_original'] = df[numeric_column]
df_standard_scaled[[numeric_column, numeric_column+'_original']].head(10)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Standard Scaling: Original vs Scaled on vertically stacked subplots, each with its own x-axis
fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # Removed sharex=True

# Original data
sns.histplot(df[numeric_column], color='navy', label='Original', kde=True, alpha=0.5, ax=axes[0])
axes[0].set_title('Original Data - Standard Scaling')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Standard Scaled data
sns.histplot(df_standard_scaled[numeric_column], color='darkorange', label='Standard Scaled', kde=True, alpha=0.5, ax=axes[1])
axes[1].set_title('Standard Scaled Data')
axes[1].set_xlabel(numeric_column)
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Robust Scaling

# %%
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()
df_robust_scaled = df.copy()
df_robust_scaled[numeric_column] = robust_scaler.fit_transform(df[[numeric_column]])
print("Robust Scaled values:")
df_robust_scaled[numeric_column+'_original'] = df[numeric_column]
df_robust_scaled[[numeric_column, numeric_column+'_original']].head(10)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Robust Scaling: Original vs Scaled on vertically stacked subplots, each with its own x-axis
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Original data
sns.histplot(df[numeric_column], color='navy', label='Original', kde=True, alpha=0.5, ax=axes[0])
axes[0].set_title('Original Data - Robust Scaling')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Standard Scaled data
sns.histplot(df_robust_scaled[numeric_column], color='darkorange', label='Standard Scaled', kde=True, alpha=0.5, ax=axes[1])
axes[1].set_title('Robust Scaled Data')
axes[1].set_xlabel(numeric_column)
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Transformation

# %% [markdown]
# ### Log Transformation
# ... with NaN handling

# %%
import numpy as np
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log1p, validate=True)
df_log_transformed = df.copy()

# Fill NaN values with the median for numerical columns
df_log_transformed[numeric_column] = df_log_transformed[numeric_column].fillna(df_log_transformed[numeric_column].median())

# Apply log transformation
df_log_transformed[numeric_column] = log_transformer.transform(df_log_transformed[[numeric_column]])
print("Log Transformed values:")

df_log_transformed[numeric_column+'_original'] = df[numeric_column]
df_log_transformed[[numeric_column, numeric_column+'_original']].head(10)

# %%
# Plot Log Transformation
plt.figure(figsize=(8, 4))
sns.histplot(df[numeric_column], color='navy', label='Original', kde=True, alpha=0.5)
plt.title('Original Data - Log Transformation')
plt.ylabel('Frequency')
plt.xlabel(numeric_column)
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df_log_transformed[numeric_column], color='darkorange', label='Log Transformed', kde=True, alpha=0.5)
plt.title('Log Transformed Data')
plt.ylabel('Frequency')
plt.xlabel(numeric_column)
plt.legend()
plt.show()

# %% [markdown]
# ### Box-Cox Transformation

# %%
from scipy.stats import boxcox

df_boxcox_transformed = df.copy()
df_boxcox_transformed[numeric_column] = df_boxcox_transformed[numeric_column] + 1  # Adding 1 to avoid zero or negative values
positive_values = df_boxcox_transformed[numeric_column] > 0
transformed_data, _ = boxcox(df_boxcox_transformed.loc[positive_values, numeric_column])
df_boxcox_transformed.loc[positive_values, numeric_column] = transformed_data
print("Box-Cox Transformed values:")

df_boxcox_transformed[numeric_column+'_original'] = df[numeric_column]
df_boxcox_transformed[[numeric_column, numeric_column+'_original']].head(10)

# %%
# Plot Box-Cox Transformation
plt.figure(figsize=(8, 4))
sns.histplot(df[numeric_column], color='navy', label='Original', kde=True, alpha=0.5)
plt.title('Original Data - Box-Cox Transformation')
plt.ylabel('Frequency')
plt.xlabel(numeric_column)
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df_boxcox_transformed[numeric_column], color='darkorange', label='Box-Cox Transformed', kde=True, alpha=0.5)
plt.title('Box-Cox Transformed Data')
plt.ylabel('Frequency')
plt.xlabel(numeric_column)
plt.legend()
plt.show()

# %% [markdown]
# ### Square Root Transformation

# %%
df_sqrt_transformed = df.copy()
df_sqrt_transformed[numeric_column] = np.sqrt(df[[numeric_column]].fillna(df[numeric_column].median()))
print("Square Root Transformed values:")

df_sqrt_transformed[numeric_column+'_original'] = df[numeric_column]
df_sqrt_transformed[[numeric_column, numeric_column+'_original']].head(10)

# %%
# Plot Square Root Transformation
plt.figure(figsize=(8, 4))
sns.histplot(df[numeric_column], color='navy', label='Original', kde=True, alpha=0.5)
plt.title('Original Data - Square Root Transformation')
plt.ylabel('Frequency')
plt.xlabel(numeric_column)
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df_sqrt_transformed[numeric_column], color='darkorange', label='Square Root Transformed', kde=True, alpha=0.5)
plt.title('Square Root Transformed Data')
plt.ylabel('Frequency')
plt.xlabel(numeric_column)
plt.legend()
plt.show()

# %% [markdown]
# ## Encoding

# %% [markdown]
# ### One Hot Encoding

# %%
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_data = one_hot_encoder.fit_transform(df[[categorical_column]].fillna('None'))
df_one_hot_encoded = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out([categorical_column]))
df_one_hot_encoded.index = df.index  # Align index with original dataframe
print("One-Hot Encoded values:")
df_one_hot_encoded['original'] = df[categorical_column]
df_one_hot_encoded.sample(10)

# %% [markdown]
# ### Label Encoding

# %%
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
# Create a copy of the DataFrame
df_label_encoded = df.copy()
# Apply label encoding to the categorical column, handling NaN values by replacing them with 'None'
df_label_encoded[categorical_column] = label_encoder.fit_transform(df[categorical_column].fillna('None'))
df_label_encoded[categorical_column+'_original'] = df[categorical_column]
print("Label Encoded values:")
df_label_encoded[[categorical_column, categorical_column+'_original']].sample(10)

# %% [markdown]
# # Binning

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from pathlib import Path

# %%
# paths
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Ames housing/') # replace with your path to the dataset!
fields_description_path = data_path/'data_description.txt'
dataset_path = data_path/'AmesHousing.csv'

# get fields description
with open(fields_description_path, 'r') as f:
    fields_description = f.read()

# get the dataset
df = pd.read_csv(dataset_path, na_values='NA', index_col='Order')
df.head(10)

# %%
# Selecting specific column for binning examples
numeric_column = 'SalePrice'

# %%
# Scatter plot of original SalePrice data
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df[numeric_column], color='blue', alpha=0.5)
plt.title('Scatter Plot of Original SalePrice Data')
plt.xlabel('Index')
plt.ylabel('SalePrice')
plt.show()

# %% [markdown]
# ## Step 1: Equal-Width Binning

# %%
# Divides the range of values into bins of equal size
equal_width_binning = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
df_equal_width_binned = df.copy()
df_equal_width_binned['SalePrice Binned'] = equal_width_binning.fit_transform(df[[numeric_column]].fillna(df[numeric_column].median()))
print("Equal-Width Binning:")
df_equal_width_binned[['SalePrice Binned']].head()

# %%
# Plotting Equal-Width Binning
plt.figure(figsize=(10, 6))
plt.hist(df_equal_width_binned['SalePrice Binned'], bins=5, color='skyblue', edgecolor='black')
plt.title('Equal-Width Binning of SalePrice')
plt.xlabel('Binned SalePrice')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ## Step 2: Equal-Frequency Binning

# %%
# Each bin has approximately the same number of data points
equal_frequency_binning = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df_equal_freq_binned = df.copy()
df_equal_freq_binned['SalePrice Binned'] = equal_frequency_binning.fit_transform(df[[numeric_column]].fillna(df[numeric_column].median()))
print("Equal-Frequency Binning:")
print(df_equal_freq_binned[['SalePrice Binned']].head())

# %%
# Plotting Equal-Frequency Binning
plt.figure(figsize=(10, 6))
plt.hist(df_equal_freq_binned['SalePrice Binned'], bins=5, color='lightgreen', edgecolor='black')
plt.title('Equal-Frequency Binning of SalePrice')
plt.xlabel('Binned SalePrice')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ## Step 3: Custom Binning

# %%
# User-defined bins based on domain knowledge
custom_bins = [0, 50000, 100000, 150000, 200000, np.inf]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
df_custom_binned = df.copy()
df_custom_binned['SalePrice Binned'] = pd.cut(df[numeric_column].fillna(df[numeric_column].median()), bins=custom_bins, labels=labels)
print("Custom Binning:")
df_custom_binned[['SalePrice Binned']].head()

# %%
# Plotting Custom Binning
plt.figure(figsize=(10, 6))
df_custom_binned['SalePrice Binned'].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Custom Binning of SalePrice')
plt.xlabel('Binned SalePrice')
plt.ylabel('Frequency')
plt.show()

# %%
# plot histogram for SalePrice
df['SalePrice'].plot(kind='hist', bins=5, title='SalePrice Histogram')
plt.show()

# %%
pass
