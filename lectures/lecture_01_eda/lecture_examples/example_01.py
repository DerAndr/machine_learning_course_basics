# /// script
# source-notebook = "example_01.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Exploratory Data Analysis - Demos [1-4]

# %% [markdown]
# ## Connect to google drive and load the dataset

# %%
# connect to google drive
# NOTE: Colab-only import commented for local script use: from google.colab import drive
drive.mount('/content/drive')

# %%
import pandas as pd
import numpy as np
import scipy
from collections import Counter
from pathlib import Path

# %%
# paths
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Ames housing/')
fields_description_path = data_path/'data_description.txt'
dataset_path = data_path/'AmesHousing.csv'

# get fields description
with open(fields_description_path, 'r') as f:
    fields_description = f.read()

# %%
print(fields_description)

# %%
# get the dataset
df = pd.read_csv(dataset_path, na_values='NA', index_col='Order')

# %%
df.head(10)

# %%
# show elements of the first row as ditionary
df.iloc[0].to_dict()

# %% [markdown]
# # Dataset Overview

# %%
# get the basic info about the dataset
df.info()

# %%
# get statistics
df.describe()

# %%
# show statistics for one field
df.describe()['Overall Qual']

# %%
# show statistics for field of type == object
df.describe(include='object')

# %%
# get the number of unique values
df.describe(include='object').T['unique']

# %%
# descriptive statistics of numerical value
df.SalePrice.describe()

# %%
df['SalePrice'].describe()

# %%
# descriptive statistics of numerical values
df.describe(include=[np.number])

# %%
# exclude numerical values for statistics
df.describe(exclude=[np.number])

# %% [markdown]
# ## Data Types of columns

# %%
df.dtypes

# %%
# df total size in bytes
size_bytes_by_columns = df.memory_usage()
# Get total memory usage in bytes (including index)
total_size_bytes = df.memory_usage(deep=True).sum()
print(f'Total memory usage: {total_size_bytes / (1024 **2)} MiB')

# %%
def detect_column_types(df: pd.DataFrame,
                        max_unique_categories: int = 10,
                        category_threshold: float = 0.1) -> dict:
    """
    Detect column types in a DataFrame, ensuring that categorical columns aren't too granular.

    Parameters:
    - df: pandas DataFrame to analyze.
    - max_unique_categories: Maximum allowed unique values for a column to be considered categorical.
    - category_threshold: Proportion threshold for categorizing columns based on uniqueness (unique_values / total_values).

    Returns:
    - column_types: A dictionary mapping each column to its detected type.
    """
    column_types = {}

    for column in df.columns:
        # Get unique values count including NaN (dropna=False)
        unique_values = df[column].nunique(dropna=False)
        total_values = len(df[column])

        if pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = _detect_numeric_type(df[column], unique_values)
        elif pd.api.types.is_object_dtype(df[column]):
            column_types[column] = _detect_object_type(df[column], unique_values, total_values, max_unique_categories, category_threshold)
        elif pd.api.types.is_bool_dtype(df[column]):
            column_types[column] = 'bool'
        else:
            column_types[column] = df[column].dtype  # Default fallback for other types

    return column_types

def _detect_numeric_type(series: pd.Series, unique_values: int) -> str:
    """
    Detects numeric column types, including binary detection.

    Parameters:
    - series: pandas Series representing the column.
    - unique_values: Number of unique values in the column.

    Returns:
    - A string representing the inferred type ('bool', 'float64', 'int64').
    """
    if unique_values == 2 and series.isin([0, 1, None]).all():
        return 'bool'  # Binary column (0/1) including NaN
    return 'float64'  # Default to float64 for numeric columns (adjust if necessary for int)

def _detect_object_type(series: pd.Series,
                        unique_values: int,
                        total_values: int,
                        max_unique_categories: int,
                        category_threshold: float) -> str:
    """
    Detects object (string) column types, considering whether to classify as categorical.

    Parameters:
    - series: pandas Series representing the column.
    - unique_values: Number of unique values in the column (including NaN).
    - total_values: Total number of values in the column.
    - max_unique_categories: Maximum number of unique values allowed for categorical columns.
    - category_threshold: Ratio threshold for unique values relative to total values to classify as categorical.

    Returns:
    - A string representing the inferred type ('category', 'object').
    """
    if (unique_values <= max_unique_categories) or (unique_values / total_values < category_threshold):
        return 'category'  # Column can be treated as categorical
    return 'object'  # Too many unique values, treat as general object (string)

# %%
# Automatically detect column types, with granularity control
column_types = detect_column_types(df, max_unique_categories=40, category_threshold=0.5)

# Apply the detected types
df1 = df.astype(column_types)
# df total size in bytes
size_bytes_by_columns_df1 = df1.memory_usage()
# Get total memory usage in bytes (including index)
total_size_bytes_df1 = df1.memory_usage(deep=True).sum()
print(f'Total memory usage: {total_size_bytes_df1} bytes')

# %%
saved_memory = (total_size_bytes - total_size_bytes_df1) / (1024**2)
print(f'==Saved memory==')
print(f'Before types adjustment: {total_size_bytes/(1024**2):.2f} MiB')
print(f'After types adjustment: {total_size_bytes_df1/(1024**2):.2f} MiB')
# calsulate saved MiB and percentage
saved_memory = (total_size_bytes - total_size_bytes_df1) / (1024**2)
percentage_saved = (1 - total_size_bytes_df1 / total_size_bytes) * 100
print(f'Saved {saved_memory:.2f} MiB ({percentage_saved:.2f}%)')

# %% [markdown]
# ## Statistical Measurements: Numerical Data

# %% [markdown]
# ### Measures of Central Tendency: Mean, Median, Mode

# %%
# calculate mean, median, mode with core python functions
def _mean(data):
    return sum(data) / len(data)

def _median(data):
    sorted_values = sorted(data)
    mid_index = len(sorted_values) // 2
    return (sorted_values[mid_index] + sorted_values[-mid_index - 1]) / 2

def _mode(data):
    counter = Counter(data)
    return counter.most_common(1)[0][0]

mean = _mean(df.SalePrice)
median = _median(df.SalePrice)
mode = _mode(df.SalePrice)

print(f'Mean: {mean}, Median: {median} Mode: {mode}')

# %%
# calculate mean, median, mode with numpy
mean = np.mean(df.SalePrice)
median = np.median(df.SalePrice)
np.unique(df.SalePrice, return_counts=True)
mode = Counter(df.SalePrice).most_common(1)[0][0]
print(f'Mean: {mean}, Median: {median}, Mode: {mode}')
# OR
mode = scipy.stats.mode(df.SalePrice)
print(f'Mean: {mean}, Median: {median}, Mode: {mode}')

# %%
# calculate mean, median, mode with pandas
mean = df.SalePrice.mean()
median = df.SalePrice.median()
mode = df.SalePrice.mode()
print(f'Mean: {mean}, Median: {median}, Mode: {mode}')

# %% [markdown]
# ### Measures of Data Spread: Standard Deviation, Variance

# %%
# calculate Standard Deviation, Variance with core python functions

def _std(data):
    mean = _mean(data)
    return (sum((x - mean) ** 2 for x in data) / (len(data) - 1)) ** 0.5

def _var(data):
    mean = _mean(data)
    return sum((x - mean) ** 2 for x in data) / (len(data) - 1)

std = _std(df.SalePrice)
var = _var(df.SalePrice)
print(f'Standard Deviation: {std}, Variance: {var}')

# %%
# calculate Standard Deviation, Variance with numpy
std = np.std(df.SalePrice)
var = np.var(df.SalePrice)
print(f'Standard Deviation: {std}, Variance: {var}')

# %%
# calculate Standard Deviation, Variance with Pandas
std = df.SalePrice.std()
var = df.SalePrice.var()
print(f'Standard Deviation: {std}, Variance: {var}')

# %% [markdown]
# By default, NumPy's std and var functions calculate the population standard deviation and variance, while Pandas' corresponding methods calculate the sample standard deviation and variance. This difference in the denominator used in the calculations leads to slightly different results.

# %% [markdown]
# To make Pandas' calculations match those of NumPy's default behavior, you can set the ddof parameter to 0 when calling the corresponding methods:

# %%
# calculate Standard Deviation, Variance with Pandas
std = df.SalePrice.std(ddof=0)
var = df.SalePrice.var(ddof=0)
print(f'Standard Deviation: {std}, Variance: {var}')

# %% [markdown]
# ### Indicators of Distribution Shape: Skewness, Kurtosis

# %%
# calculate Skewness, Kurtosis with core python functions

def _skewness(data):
    mean = _mean(data)
    std = _std(data)
    return sum((x - mean) ** 3 for x in data) / (len(data) * std ** 3)

def _kurtosis(data):
    mean = _mean(data)
    std = _std(data)
    return sum((x - mean) ** 4 for x in data) / (len(data) * std ** 4) -3

skewness = _skewness(df.SalePrice)
kurtosis = _kurtosis(df.SalePrice)
print(f'Skewness: {skewness}, Kurtosis: {kurtosis}')

# %%
# calculate Skewness, Kurtosis with scipy functions

skewness = scipy.stats.skew(df.SalePrice)
kurtosis = scipy.stats.kurtosis(df.SalePrice)
print(f'Skewness: {skewness}, Kurtosis: {kurtosis}')

# %%
# calculate Skewness, Kurtosis with pandas functions

skewness = df.SalePrice.skew()
kurtosis = df.SalePrice.kurtosis()
print(f'Skewness: {skewness}, Kurtosis: {kurtosis}')

# %%
# calculate Skewness, Kurtosis with scipy functions with bias=False

skewness = scipy.stats.skew(df.SalePrice, bias=False)
kurtosis = scipy.stats.kurtosis(df.SalePrice, bias=False)
print(f'Skewness: {skewness}, Kurtosis: {kurtosis}')

# %% [markdown]
# ## Statistical Measurements: Categorical Data

# %% [markdown]
# ### Frequency Count, Proportion, Mode, Unique Count, Missing Values

# %%
df.Fence.value_counts(dropna=False)

# %%
df['Neighborhood'].value_counts(normalize=True)

# %%
def analyze_categorical_column(df, column_name):
    """
    Analyze a categorical column from a DataFrame, computing:
    - Frequency count
    - Proportion/Percentage of each category
    - Mode (most frequent category)
    - Unique count (number of distinct categories)
    - Missing values (NaN count)
    """
    column = df[column_name]

    # Frequency count
    freq_count = column.value_counts(dropna=False)

    # Proportion or Percentage
    proportion = column.value_counts(normalize=True, dropna=False) * 100

    # Mode (most frequent category)
    mode_value = column.mode(dropna=False).iloc[0]

    # Unique count
    unique_count = column.nunique(dropna=False)

    # Missing values
    missing_values = column.isnull().sum()

    # Display results
    print(f"Analysis of '{column_name}' column:")
    print(f"Frequency Count:\n{freq_count}")
    print(f"\nProportion (%):\n{proportion}")
    print(f"\nMode (most frequent category): {mode_value}")
    print(f"Unique Count: {unique_count}")
    print(f"Missing Values: {missing_values}")

# Analyze the 'Class' column (or any other categorical column)
analyze_categorical_column(df, 'Fence')

# %% [markdown]
# # **DEMO Univariate analysis**
# 
# ---

# %% [markdown]
# # Univariate Analysis

# %% [markdown]
# ## Quantile Analysis

# %%
quantile_analysis = df['SalePrice'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
quantile_analysis

# %% [markdown]
# ### Histogram

# %%
# calculate histogram and draw it with with matplotlib
import matplotlib.pyplot as plt

plt.hist(df.SalePrice, bins=50, edgecolor='black', color='royalblue')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Histogram of Sale Price')
plt.show()

# %%
# calculate histogram and draw it with with seaborn
import seaborn as sns

sns.histplot(df.SalePrice, bins=50, color='royalblue')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Histogram of Sale Price')
plt.show()

# %% [markdown]
# ## KDE (Kernel Density Estimation)

# %%
# calculate KDE and draw it with with matplotlib

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# Calculate the KDE using scipy's gaussian_kde
kde = stats.gaussian_kde(df['SalePrice'])

# Create an array of x values for which to calculate the KDE
x_values = np.linspace(min(df['SalePrice']), max(df['SalePrice']), 1000)

# Evaluate the KDE on the x-values
kde_values = kde(x_values)

# Plot the KDE
plt.plot(x_values, kde_values, label='KDE')

# Customize the plot
plt.title('KDE Plot of Values')
plt.xlabel('Values')
plt.ylabel('Density')

# Optional: fill the area under the KDE curve
plt.fill_between(x_values, kde_values, alpha=0.5)

plt.legend()
plt.show()

# %%
import seaborn as sns

sns.histplot(df.SalePrice, bins=50, color='royalblue' , kde=True)
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Histogram of Sale Price')
plt.show()
#

# %%
ax = sns.histplot(data=df, x="SalePrice", kde=False, stat='density', color='grey')
# thick line
sns.kdeplot(data=df, x="SalePrice", color='darkorange', ax=ax, linewidth=3)
plt.xlabel('Sale Price')

# %% [markdown]
# ## ECDF (Empirical Cumulative Distribution Function)

# %%
# Calculate with pandas
ecdf = df.SalePrice.value_counts(normalize=True).sort_index().cumsum()
ecdf.plot(marker='.', linestyle='none')
plt.xlabel('Sale Price')
plt.ylabel('ECDF')
plt.title('ECDF Plot of Sale Price')
plt.show()

# %% [markdown]
# ## Boxplot

# %%
# Calculate boxplot and draw it

plt.boxplot(df.SalePrice)
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Boxplot of Sale Price')
plt.show()

# %%
# Boxplot for SalePrice
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['SalePrice'])
plt.title('Boxplot of SalePrice')
plt.xlabel('Sale Price')
plt.show()

# %%
Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['SalePrice'] < (Q1 - 1.5 * IQR)) | (df['SalePrice'] > (Q3 + 1.5 * IQR))]
outliers

# %% [markdown]
# ## Violin Plot

# %%
plt.figure(figsize=(10, 6))
sns.violinplot(x=df['SalePrice'])
plt.title('Violin Plot of SalePrice')
plt.xlabel('Sale Price')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.violinplot(y=df['SalePrice'], color='lightgreen')
plt.title('Violin Plot of SalePrice')
plt.xlabel('Sale Price')
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Calculating statistics
mean = df['SalePrice'].mean()
median = df['SalePrice'].median()
mode = stats.mode(df['SalePrice']).mode

# Creating the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=df['SalePrice'], color='lightblue')

# Adding lines for mode, median, and mean
plt.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(median, color='red', linestyle='-', label=f'Median: {median:.2f}')
plt.axvline(mode, color='orange', linestyle='-', label=f'Mode: {mode:.2f}')

# Adding titles and labels
plt.title('Violin Plot of SalePrice')
plt.xlabel('Sale Price')
plt.legend()

# Displaying the plot
plt.show()

# %% [markdown]
# # **DEMO Bivariate analysis**
# 
# ---

# %% [markdown]
# # Bivariate Analysis

# %% [markdown]
# ## Scatter plot

# %%
# Scatter plot of SalePrice vs. Gr Liv Area
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)
plt.title('Scatter Plot: SalePrice vs. Gr Liv Area')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.show()

# %%
# scatter plot with regression line and correlation coefficient
sns.regplot(x='Gr Liv Area', y='SalePrice', data=df,
            line_kws={'color': 'red'},  # Color of the regression line
            scatter_kws={'alpha': 0.5})  # Transparency of the scatter points (0=transparent, 1=opaque)

plt.title('Scatter Plot: SalePrice vs. Gr Liv Area')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# Performing linear regression using statsmodels
X = sm.add_constant(df['Gr Liv Area'])  # Adds a constant term (intercept) to the predictor
model = sm.OLS(df['SalePrice'], X).fit()  # Fitting the model
intercept, slope = model.params  # Extracting intercept and slope
r_squared = model.rsquared  # Extracting R-squared
# Calculating Pearson correlation coefficient
pearson_corr, _ = stats.pearsonr(df['Gr Liv Area'], df['SalePrice'])  # Pearson correlation coefficient

# Creating the regplot with a custom line color and point transparency
plt.figure(figsize=(8, 6))
sns.regplot(x='Gr Liv Area', y='SalePrice', data=df,
            line_kws={'color': 'red'},  # Color of the regression line
            scatter_kws={'alpha': 0.5})  # Transparency of the scatter points

# Adding the regression equation, R-squared, and Pearson correlation coefficient to the plot
plt.text(0.05, 0.95, f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.2f}\nPearson r = {pearson_corr:.2f}',
         transform=plt.gca().transAxes,  # Ensure text is positioned relative to plot axes
         fontsize=12, verticalalignment='top')

# Adding titles and labels
plt.title('Regression Plot with Equation and R-squared')
plt.xlabel('Gr Liv Area')
plt.ylabel('Sale Price')

# Displaying the plot
plt.show()

# %% [markdown]
# ## Correlation Matrix (Heatmap)

# %%
correlation_matrix = df[['SalePrice', 'Gr Liv Area', 'Garage Cars', 'Lot Area', 'Overall Qual']].corr()
correlation_matrix['SalePrice'].sort_values(ascending=False)

# %%
# Correlation matrix using a heatmap
plt.figure(figsize=(10, 6))
corr_matrix = df[['SalePrice', 'Gr Liv Area', 'Garage Cars', 'Lot Area', 'Overall Qual']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate the correlation matrix
corr_matrix = df[['SalePrice', 'Gr Liv Area', 'Garage Cars', 'Lot Area', 'Overall Qual']].corr()

# Create a mask for the upper triangle (values not needed)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the figure
plt.figure(figsize=(10, 6))

# Plot the heatmap with the mask
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5, square=True)

# Add title and show the plot
plt.title('Correlation Matrix Heatmap (Lower Triangle)')
plt.show()

# %% [markdown]
# ## Cross-Tabulation (Contingency Table)

# %%
crosstab = pd.crosstab(df['Overall Qual'], df['Neighborhood'])

# Heatmap visualization with improved aesthetics
plt.figure(figsize=(14, 8))

sns.heatmap(crosstab, annot=True, cmap='binary', linewidths=0.5, cbar_kws={'label': 'Count'}, fmt='g')

# Adding more descriptive titles and labels
plt.title('Distribution of Overall Quality Across Neighborhoods', fontsize=16)
plt.xlabel('Neighborhood', fontsize=12)
plt.ylabel('Overall Quality', fontsize=12)

# Rotating the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to avoid clipping of the labels
plt.tight_layout()

# Show the plot
plt.show()

# %% [markdown]
# ## Joint Plot

# %%
# Joint plot showing SalePrice vs. GrLivArea with histograms
sns.jointplot(x='Gr Liv Area', y='SalePrice', data=df, kind='scatter', height=8)
plt.suptitle('Joint Plot: SalePrice vs. GrLivArea', y=1.02)
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats


# Performing linear regression using statsmodels
X = sm.add_constant(df['Gr Liv Area'])  # Adds a constant term (intercept) to the predictor
model = sm.OLS(df['SalePrice'], X).fit()  # Fitting the model
intercept, slope = model.params  # Extracting intercept and slope
r_squared = model.rsquared  # Extracting R-squared

# Calculating Pearson correlation coefficient
pearson_corr, _ = stats.pearsonr(df['Gr Liv Area'], df['SalePrice'])  # Pearson correlation coefficient

# Creating the jointplot with a regression line and custom KDE line color
g = sns.jointplot(x='Gr Liv Area', y='SalePrice', data=df, kind='reg', height=8,
                  scatter_kws={'alpha': 0.5, 's':10}, line_kws={'color': 'red'},
                  marginal_kws={'fill': True, 'color': 'darkorange'})  # Change KDE line color

# Adding the regression equation, R-squared, and Pearson correlation coefficient
g.ax_joint.text(0.05, 0.95, f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.2f}\nPearson r = {pearson_corr:.2f}',
                transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top')

# Calculating statistics for marginal distributions (mean, median, std)
x_mean, x_median, x_std = df['Gr Liv Area'].mean(), df['Gr Liv Area'].median(), df['Gr Liv Area'].std()
y_mean, y_median, y_std = df['SalePrice'].mean(), df['SalePrice'].median(), df['SalePrice'].std()

# Adding statistics for the X-axis (bottom marginal distribution)
g.ax_marg_x.text(0.05, 0.85, f'Mean: {x_mean:.2f}\nMedian: {x_median:.2f}\nStd: {x_std:.2f}',
                 transform=g.ax_marg_x.transAxes, fontsize=10)

# Adding statistics for the Y-axis (left marginal distribution)
g.ax_marg_y.text(0.05, 0.85, f'Mean: {y_mean:.2f}\nMedian: {y_median:.2f}\nStd: {y_std:.2f}',
                 transform=g.ax_marg_y.transAxes, fontsize=10, rotation=0)

# Moving the title to the bottom of the plot
plt.subplots_adjust(top=0.9, bottom=0.15)  # Adjust plot layout to make space for the title at the bottom
plt.suptitle('Jointplot of Gr Liv Area vs SalePrice', y=0.02)

# Displaying the plot
plt.show()

# %% [markdown]
# ## Hexbin Plot

# %%
# Hexbin plot for large datasets
plt.figure(figsize=(10, 6))
plt.hexbin(df['Gr Liv Area'], df['SalePrice'], gridsize=20, cmap='binary')
plt.colorbar(label='count in bin')
plt.title('Hexbin Plot: SalePrice vs. GrLivArea')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.show()

# %%
sns.jointplot(x='Gr Liv Area', y='SalePrice', kind="hex", data=df, height=8, joint_kws={'gridsize': 20}, color='black')
plt.suptitle('Joint Plot: SalePrice vs. GrLivArea', y=1.02)
plt.show()

# %% [markdown]
# ## Bar plot

# %%
# Bar plot to compare SalePrice by the number of GarageCars
plt.figure(figsize=(10, 6))
sns.barplot(x='Garage Cars', y='SalePrice', data=df)
plt.title('Bar Plot: SalePrice by Number of Garage Cars')
plt.xlabel('Number of Garage Cars')
plt.ylabel('Average Sale Price ($)')
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Calculate statistics (standard deviation) by 'Garage Cars'
stats_df = df.groupby('Garage Cars').agg(std_saleprice=('SalePrice', 'std')).reset_index()

# Setting a color palette
colors = sns.color_palette("Blues_d", n_colors=df['Garage Cars'].nunique())

# Create the bar plot
plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='Garage Cars', y='SalePrice', data=df, errorbar=None, palette=colors, hue='Garage Cars', dodge=False, legend=False)

# Adding exact values on top of the bars (mean of each group)
for p in barplot.patches:
    height = p.get_height()
    barplot.annotate(f'${height:,.0f}',
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='baseline', fontsize=11, color='black',
                     xytext=(0, 5), textcoords='offset points')

# Adding only the standard deviation as annotations
for idx, row in stats_df.iterrows():
    if pd.notna(row["std_saleprice"]):
        # Add standard deviation annotation if it's not NaN
        barplot.annotate(f'Std: ${row["std_saleprice"]:,.0f}',
                         (idx, row["std_saleprice"]),
                         ha='center', va='top', fontsize=10, color='darkblue')
    else:
        # Annotate "N/A" when standard deviation is not defined
        barplot.annotate('Std: N/A',
                         (idx, 0),
                         ha='center', va='top', fontsize=10, color='darkblue')

# Adding gridlines for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Customize title and labels
plt.title('Average Sale Price by Number of Garage Cars', fontsize=16, weight='bold')
plt.xlabel('Number of Garage Cars', fontsize=12)
plt.ylabel('Average Sale Price ($)', fontsize=12)

# Adding useful context information below the title
total_cars = df['Garage Cars'].count()
overall_std = df['SalePrice'].std()
plt.suptitle(f'Total Data Points: {total_cars} | Overall Std: ${overall_std:,.0f}',
             fontsize=10, color='gray')

# Display the plot
plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Setting a color palette
colors = sns.color_palette("Blues_d", n_colors=df['Garage Cars'].nunique())

# Create the boxplot
plt.figure(figsize=(12, 8))
boxplot = sns.boxplot(x='Garage Cars', y='SalePrice', data=df, palette=colors)

# Adding exact values for median of each boxplot
medians = df.groupby('Garage Cars')['SalePrice'].median()
for index, median in enumerate(medians):
    boxplot.annotate(f'Median: ${median:,.0f}',
                     (index, median),
                     ha='center', va='top', fontsize=11, color='black',
                     xytext=(0, -12), textcoords='offset points')

# Adding gridlines for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Customize title and labels
plt.title('Distribution of Sale Price by Number of Garage Cars (Boxplot)', fontsize=16, weight='bold')
plt.xlabel('Number of Garage Cars', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)

# Adding useful context information below the title
total_cars = df['Garage Cars'].count()
overall_std = df['SalePrice'].std()
plt.suptitle(f'Total Data Points: {total_cars} | Overall Std: ${overall_std:,.0f}',
             fontsize=10, color='gray')

# Display the plot
plt.tight_layout()
plt.show()

# %%
df['Garage Cars'].value_counts()

# %%
# calculate mean and std by groupby
df[['SalePrice', 'Garage Cars']].groupby('Garage Cars').agg(['mean', 'std'])

# %%
# Group by 'Neighborhood' and calculate the mean and median sale price
neighborhood_stats = df.groupby('Neighborhood')['SalePrice'].agg(['mean', 'median', 'std', 'count']).sort_values(by='mean', ascending=False)
neighborhood_stats

# %%
pivot = pd.pivot_table(df, values='SalePrice', index=['Neighborhood'], aggfunc=[np.mean, np.median, np.std])
pivot

# %% [markdown]
# ## Pair Plot

# %%
# Pair plot showing relationships between multiple variables
sns.pairplot(df[['SalePrice', 'Gr Liv Area', 'Lot Area', 'Overall Qual']])
plt.suptitle('Pair Plot: Relationships between variables', y=1.02)
plt.show()

# %% [markdown]
# # **DEMO Multivariate analysis**
# 
# 
# ---

# %% [markdown]
# # Multivariate analysis

# %% [markdown]
# ## Categorical and Numeric Statistics Together

# %%
df.groupby('Garage Cars').agg({
    'SalePrice': ['mean', 'median', 'std', 'count'],
    'Lot Area': ['mean', 'median'],
})

# %% [markdown]
# ## 3D Scatter Plot

# %%
import pandas as pd
import plotly.express as px

# Load your dataset (Replace this with your actual dataset loading code)
# df = pd.read_csv('your_dataset.csv')  # Example to load dataset from a CSV

# 3D Scatter plot using Plotly for your dataset (Assuming columns 'GrLivArea', 'LotArea', and 'SalePrice')
fig = px.scatter_3d(df, x='Gr Liv Area', y='Lot Area', z='SalePrice',
                    color='Overall Qual',  # Color the points based on 'OverallQual'
                    title='3D Scatter Plot: GrLivArea, LotArea, and SalePrice',
                    labels={'GrLivArea':'Ground Living Area', 'LotArea':'Lot Area', 'SalePrice':'Sale Price'})

# Set marker size smaller
fig.update_traces(marker=dict(size=3))  # Adjust 'size' for smaller markers

# Show the plot
fig.show()

# %% [markdown]
# ## Violin plot for multivariate case

# %%
# Assuming 'Neighborhood' and 'OverallQual' are available categorical variables
# Limit the number of Neighborhoods by selecting the top 5 most frequent ones
top_bldgtypes = df['Neighborhood'].value_counts().index[:5]

# Filter the dataset to include only the top 5 Neighborhoods
df_filtered = df[df['Neighborhood'].isin(top_bldgtypes)]

# Violin plot for SalePrice grouped by the top 5 Neighborhood categories and OverallQual
plt.figure(figsize=(16, 8))

# Create the violin plot with hue for multivariate analysis
sns.violinplot(x='Neighborhood', y='SalePrice', hue='Overall Qual', data=df_filtered, split=True, inner='quart', palette='Set2')

# Add titles and labels
plt.title('Violin Plot: SalePrice by Neighborhood and Overall Quality', fontsize=16)
plt.xlabel('Neighborhood', fontsize=12)
plt.ylabel('SalePrice', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

# %%
pass
