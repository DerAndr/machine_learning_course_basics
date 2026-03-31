# /// script
# source-notebook = "example_03.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Dimentionality Reduction

# %% [markdown]
# 
# [Principal Component Analysis (PCA)](#scrollTo=lHWy51NEbMU_&line=1&uniqifier=1):
# * Identifies the axes of maximum variance in the data and projects the data onto a lower-dimensional subspace.
# * Linear method.
# 
# [t-SNE (t-Distributed Stochastic Neighbor Embedding)](#scrollTo=CjOE2Yp_bF9a&line=1&uniqifier=1)
# * Non-linear technique for visualizing high-dimensional data by preserving the local structure.
# * Often used for 2D or 3D visualization of data clusters.
# 
# [UMAP (Uniform Manifold Approximation and Projection)](#scrollTo=RRV1h6era_pl&line=1&uniqifier=1):
# * Similar to t-SNE but computationally more efficient.
# * Preserves both local and global structures of data, making it useful for visualizations and embeddings.

# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path

# %% [markdown]
# # Dataset Loading

# %%
# connect to google drive
# NOTE: Colab-only import commented for local script use: from google.colab import drive
drive.mount('/content/drive')

# %%
# paths
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/MobileDeviceUsage/') # replace with your path to the dataset!
fields_description_path = data_path/'info.txt'
dataset_path = data_path/'user_behavior_dataset.csv'

# get fields description
with open(fields_description_path, 'r') as f:
    fields_description = f.read()

# get the dataset
df = pd.read_csv(dataset_path)
# Data Cleaning: Fill missing numeric values with zero for simplicity
df.fillna(0, inplace=True)
df.head(10)

# %% [markdown]
# 
# 
# ---
# 

# %%
print(fields_description)

# %%
numeric_features = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)',
                    'Number of Apps Installed', 'Data Usage (MB/day)', 'Age']
X = df[numeric_features].fillna(0)
target_name = 'User Behavior Class'
y = df[target_name]

X_scaled = StandardScaler().fit_transform(X)

# %%
X_scaled

# %% [markdown]
# # Principal Component Analysis (PCA)

# %%
# Step 3: Principal Component Analysis (PCA)
import time
start_time = time.time()
pca = PCA(n_components=0.9)  # Keep 90% of variance
X_pca = pca.fit_transform(X_scaled)
end_time = time.time()
print(f'Time taken for PCA: {end_time - start_time:.2f} seconds')

# Plot PCA results
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap('jet', np.max(y) - np.min(y) + 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.7, s=5, cmap=cmap, vmin=min(y.unique()), vmax=max(y.unique()))
cbar = plt.colorbar(scatter, ticks=sorted(y.unique()))  # Set colorbar ticks to unique values of y
cbar.set_label(target_name)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Dataset (90% Variance)')
plt.xticks(range(int(X_pca[:, 0].min()) - 1, int(X_pca[:, 0].max()) + 2))  # Set x-axis ticks to integer values
plt.yticks(range(int(X_pca[:, 1].min()) - 1, int(X_pca[:, 1].max()) + 2))  # Set y-axis ticks to integer values
plt.show()


# %%
# show the variance explained with 2 components:
print(f'Variance explained by the first component: {pca.explained_variance_ratio_[0]:.2f}')
print(f'Variance explained by the second component: {pca.explained_variance_ratio_[1]:.2f}')
# number of components to explain 90% variance
print(f'Number of components to explain 90% variance: {pca.n_components_}')

# %% [markdown]
# # t-SNE (t-Distributed Stochastic Neighbor Embedding)

# %%
import time
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42, perplexity=50.0, early_exaggeration=15.0)
X_tsne = tsne.fit_transform(X_scaled)
end_time = time.time()
print(f'Time taken for t-SNE: {end_time - start_time:.2f} seconds')

# %%
# Plot t-SNE results
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap('jet', np.max(y) - np.min(y) + 1)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, alpha=0.7, s=5, cmap=cmap, vmin=min(y.unique()), vmax=max(y.unique()))
cbar = plt.colorbar(scatter, ticks=sorted(y.unique()))  # Set colorbar ticks to unique values of y
cbar.set_label(target_name)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Dataset')
plt.show()

# %% [markdown]
# # UMAP (Uniform Manifold Approximation and Projection)

# %%
# NOTE: notebook magic commented for local script use: ! pip install umap-learn

# %%
import umap
import time

start_time = time.time()
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10)
X_umap = umap_reducer.fit_transform(X_scaled)
end_time = time.time()
print(f'Time taken for UMAP: {end_time - start_time:.2f} seconds')

# %%
# Plot UMAP results
plt.figure(figsize=(8, 6))
cmap = plt.get_cmap('jet', np.max(y) - np.min(y) + 1)
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, alpha=0.7, s=5, cmap=cmap, vmin=min(y.unique()), vmax=max(y.unique()))
cbar = plt.colorbar(scatter, ticks=sorted(y.unique()))  # Set colorbar ticks to unique values of y
cbar.set_label(target_name)

plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP of Ames Housing Dataset')
plt.show()

# %%
pass
