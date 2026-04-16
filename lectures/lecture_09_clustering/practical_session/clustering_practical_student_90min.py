# %% [markdown]
# # Clustering: Practical Session - STUDENT VERSION (90 minutes)
# 
# This practical uses **Fashion-MNIST** from OpenML: 70,000 grayscale `28 x 28` clothing images across 10 clothing categories.
# 
# To keep the notebook responsive in Google Colab, we work with a fixed random subset of `5,000` images. That is large enough to feel realistic, but still manageable in class.

# %% [markdown]
# ## Setup
# 
# If you run this in Google Colab and an import is missing, install the lightweight extras first:
# 
# ```python
# %pip install -q openml plotly ipywidgets umap-learn
# ```
# 
# The practical assumes these packages are available in the notebook runtime.

# %%
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from IPython.display import clear_output, display
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.datasets import fetch_openml, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="talk")
RANDOM_STATE = 42
DATA_ID = 40996
SAMPLE_SIZE = 5000
PCA_COMPONENTS = 30
UMAP_SAMPLE_SIZE = 2500
INTERACTIVE_SAMPLE_SIZE = 1500

np.random.seed(RANDOM_STATE)

print("Environment ready.")
print("This practical uses OpenML Fashion-MNIST (data_id=40996).")

# %% [markdown]
# ## Shared Helper Functions

# %%
FASHION_LABELS = {
    "0": "T-shirt/top",
    "1": "Trouser",
    "2": "Pullover",
    "3": "Dress",
    "4": "Coat",
    "5": "Sandal",
    "6": "Shirt",
    "7": "Sneaker",
    "8": "Bag",
    "9": "Ankle boot",
}


def load_clustering_dataset(sample_size=SAMPLE_SIZE):
    dataset = fetch_openml(data_id=DATA_ID, as_frame=False, parser="auto")

    X_pixels = dataset.data.astype(np.float32)
    y_digits = np.asarray(dataset.target).astype(str)
    y_reference_labels = pd.Series([FASHION_LABELS[digit] for digit in y_digits], name="fashion_item")

    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(sample_size, len(X_pixels))
    sample_indices = np.sort(rng.choice(len(X_pixels), size=sample_size, replace=False))

    X_pixels = X_pixels[sample_indices]
    y_reference_labels = y_reference_labels.iloc[sample_indices].reset_index(drop=True)
    y_reference = pd.factorize(y_reference_labels, sort=True)[0]

    dataset_info = {
        "dataset_name": "Fashion-MNIST",
        "source": "OpenML",
        "openml_data_id": DATA_ID,
        "description": "Grayscale 28x28 images of 10 clothing categories.",
        "full_rows": int(len(dataset.data)),
        "sample_rows": int(len(X_pixels)),
        "n_features": int(X_pixels.shape[1]),
        "image_shape": (28, 28),
        "n_classes": int(y_reference_labels.nunique()),
        "target_values": sorted(y_reference_labels.unique().tolist()),
    }

    return X_pixels, y_reference_labels, y_reference, sample_indices, dataset_info


def preprocess_features(X_pixels, n_components=PCA_COMPONENTS):
    # The preprocessing stays intentionally simple and explicit:
    # 1. normalize pixel intensities to [0, 1]
    # 2. compress 784 pixels into a smaller PCA representation
    # 3. standardize the reduced coordinates before distance-based clustering
    X_normalized = X_pixels / 255.0
    pca_model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )
    X_pca = pca_model.fit_transform(X_normalized)
    reducer_scaler = StandardScaler()
    X_cluster = reducer_scaler.fit_transform(X_pca)
    return X_normalized, X_pca, X_cluster, pca_model, reducer_scaler


def evaluate_labels(X, labels, y_true=None):
    labels = np.asarray(labels)
    noise_mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    result = {
        "n_clusters": n_clusters,
        "noise_fraction": float(np.mean(labels == -1)) if -1 in labels else 0.0,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "ari": np.nan,
        "nmi": np.nan,
    }

    if n_clusters >= 2 and noise_mask.sum() > n_clusters:
        X_eval = X[noise_mask] if -1 in labels else X
        labels_eval = labels[noise_mask] if -1 in labels else labels

        if len(np.unique(labels_eval)) >= 2:
            result["silhouette"] = silhouette_score(X_eval, labels_eval)
            result["davies_bouldin"] = davies_bouldin_score(X_eval, labels_eval)
            result["calinski_harabasz"] = calinski_harabasz_score(X_eval, labels_eval)

    if y_true is not None and n_clusters >= 1:
        result["ari"] = adjusted_rand_score(y_true, labels)
        result["nmi"] = normalized_mutual_info_score(y_true, labels)

    return result


def safe_silhouette_for_metric(X, labels, metric):
    labels = np.asarray(labels)
    noise_mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2 or noise_mask.sum() <= n_clusters:
        return np.nan

    X_eval = X[noise_mask] if -1 in labels else X
    labels_eval = labels[noise_mask] if -1 in labels else labels

    if len(np.unique(labels_eval)) < 2:
        return np.nan

    return silhouette_score(X_eval, labels_eval, metric=metric)


def _sort_labels(values):
    def key(value):
        value = str(value)
        if value == "-1":
            return (1, -1)
        if value.lstrip("-").isdigit():
            return (0, int(value))
        return (0, value)

    return sorted(values, key=key)


def plot_projection(ax, embedding, labels, title, xlabel="Component 1", ylabel="Component 2", legend=False):
    label_series = pd.Series(labels, dtype="object").astype(str)
    unique_labels = _sort_labels(label_series.unique())
    colors = sns.color_palette("tab10", n_colors=max(3, len(unique_labels)))
    palette = {label: colors[idx % len(colors)] for idx, label in enumerate(unique_labels)}

    if "-1" in palette:
        palette["-1"] = "#222222"

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=label_series,
        palette=palette,
        s=28,
        alpha=0.75,
        ax=ax,
        legend=legend,
        linewidth=0,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_silhouette(ax, X, labels, title):
    labels = np.asarray(labels)
    mask = labels != -1

    if mask.sum() == 0 or len(set(labels[mask])) < 2:
        ax.text(0.5, 0.5, "Silhouette is undefined", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlim(-0.2, 1.0)
        return

    X_eval = X[mask]
    labels_eval = labels[mask]
    sample_values = silhouette_samples(X_eval, labels_eval)
    y_lower = 10

    for cluster_id in sorted(np.unique(labels_eval)):
        cluster_values = np.sort(sample_values[labels_eval == cluster_id])
        size = cluster_values.shape[0]
        y_upper = y_lower + size
        color = sns.color_palette("tab10", n_colors=10)[int(cluster_id) % 10]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            facecolor=color,
            alpha=0.75,
        )
        ax.text(-0.05, y_lower + 0.5 * size, str(cluster_id))
        y_lower = y_upper + 10

    ax.axvline(sample_values.mean(), color="red", linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_xlim(-0.2, 1.0)


def show_image_grid(images, titles=None, n_cols=8, suptitle=None, cmap="binary"):
    images = np.asarray(images)

    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)

    n_images = len(images)
    if n_images == 0:
        print("No images to display.")
        return

    n_cols = min(n_cols, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < n_images:
            ax.imshow(images[idx], cmap=cmap)
            if titles is not None:
                ax.set_title(str(titles[idx]), fontsize=9)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        plt.tight_layout()
    plt.show()


def plot_cluster_mean_images(images, labels, suptitle):
    images = np.asarray(images)
    labels = np.asarray(labels)
    cluster_ids = [cluster_id for cluster_id in sorted(np.unique(labels)) if cluster_id != -1]

    if not cluster_ids:
        print("No non-noise clusters available for mean-image visualization.")
        return

    n_cols = min(5, len(cluster_ids))
    n_rows = int(np.ceil(len(cluster_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, cluster_id in zip(axes, cluster_ids):
        mean_image = images[labels == cluster_id].mean(axis=0)
        ax.imshow(mean_image, cmap="binary")
        ax.set_title(f"Cluster {cluster_id}\n(n={int((labels == cluster_id).sum())})")
        ax.axis("off")

    for ax in axes[len(cluster_ids):]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def representative_indices(model_name, model, X_repr, labels, cluster_id, n_examples=16, sort_mode="prototype"):
    labels = np.asarray(labels)
    cluster_indices = np.where(labels == cluster_id)[0]

    if len(cluster_indices) == 0:
        return np.array([], dtype=int)

    n_examples = min(n_examples, len(cluster_indices))

    if sort_mode == "random":
        rng = np.random.default_rng(RANDOM_STATE + int(cluster_id) + len(cluster_indices))
        return np.sort(rng.choice(cluster_indices, size=n_examples, replace=False))

    if model_name in {"KMeans", "MiniBatchKMeans"} and cluster_id != -1:
        reference = model.cluster_centers_[cluster_id]
    else:
        reference = X_repr[cluster_indices].mean(axis=0)

    distances = np.linalg.norm(X_repr[cluster_indices] - reference, axis=1)
    order = np.argsort(distances)
    return cluster_indices[order[:n_examples]]


def build_cluster_summary(labels, y_reference_labels, X_repr):
    labels = np.asarray(labels)
    y_reference_labels = pd.Series(y_reference_labels).reset_index(drop=True)
    summary_rows = []

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_targets = y_reference_labels[mask]
        target_shares = cluster_targets.value_counts(normalize=True)
        dominant_label = target_shares.index[0]
        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "size": int(mask.sum()),
                "dominant_hidden_label": dominant_label,
                "dominant_share": round(float(target_shares.iloc[0]), 3),
                "top_hidden_labels": ", ".join(
                    f"{label} ({share:.0%})"
                    for label, share in target_shares.head(3).items()
                ),
                "mean_distance_to_center": round(
                    float(
                        np.linalg.norm(
                            X_repr[mask] - X_repr[mask].mean(axis=0),
                            axis=1,
                        ).mean()
                    ),
                    3,
                ),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("cluster").reset_index(drop=True)


def show_cluster_gallery(
    model_name,
    cluster_id,
    fitted_models,
    X_repr,
    images,
    y_reference_labels=None,
    n_examples=24,
    sort_mode="prototype",
):
    model_bundle = fitted_models[model_name]
    labels = model_bundle["labels"]
    chosen_indices = representative_indices(
        model_name=model_name,
        model=model_bundle["model"],
        X_repr=X_repr,
        labels=labels,
        cluster_id=cluster_id,
        n_examples=n_examples,
        sort_mode=sort_mode,
    )

    if len(chosen_indices) == 0:
        print(f"{model_name}: cluster {cluster_id} is empty.")
        return

    if y_reference_labels is None:
        print(f"{model_name} | cluster {cluster_id} | size={(labels == cluster_id).sum()}")
        titles = [f"idx={idx}" for idx in chosen_indices]
    else:
        target_counts = (
            pd.Series(y_reference_labels).iloc[chosen_indices].value_counts(normalize=True).head(3)
        )
        print(
            f"{model_name} | cluster {cluster_id} | size={(labels == cluster_id).sum()} | "
            f"top hidden labels: "
            + ", ".join(f"{label} ({share:.0%})" for label, share in target_counts.items())
        )
        titles = [
            f"{pd.Series(y_reference_labels).iloc[idx]}\nidx={idx}"
            for idx in chosen_indices
        ]
    show_image_grid(
        images[chosen_indices],
        titles=titles,
        n_cols=6,
        suptitle=f"{model_name}: representative images from cluster {cluster_id}",
    )


def plot_interactive_3d(embedding_3d, labels, title, hover_labels=None):
    if px is None:
        print("Plotly is not available. Install `plotly` to get the interactive 3D plot.")
        return None

    labels_array = np.asarray(labels, dtype=object).astype(str)

    plot_df = pd.DataFrame(
        {
            "PC1": np.asarray(embedding_3d[:, 0]),
            "PC2": np.asarray(embedding_3d[:, 1]),
            "PC3": np.asarray(embedding_3d[:, 2]),
            "label": labels_array,
        }
    )
    hover_fields = None
    if hover_labels is not None:
        hover_array = np.asarray(hover_labels, dtype=object).astype(str)
        plot_df["hidden_label"] = hover_array
        hover_fields = ["hidden_label"]

    fig = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="label",
        hover_data=hover_fields,
        title=title,
        opacity=0.72,
        width=900,
        height=700,
    )
    fig.update_traces(marker={"size": 3})
    fig.show()
    return fig


def show_widget_cluster_browser(fitted_models, X_repr, images, y_reference_labels=None):
    if widgets is None:
        print("ipywidgets is not available. Use show_cluster_gallery(...) manually.")
        return None

    cluster_options = []
    for model_name, bundle in fitted_models.items():
        for cluster_id in sorted(np.unique(bundle["labels"])):
            label = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
            cluster_options.append((f"{model_name} | {label}", (model_name, int(cluster_id))))

    selector = widgets.Dropdown(
        options=cluster_options,
        description="Cluster",
        layout=widgets.Layout(width="420px"),
    )
    n_examples_slider = widgets.IntSlider(
        value=18,
        min=6,
        max=36,
        step=6,
        description="Images",
        continuous_update=False,
    )
    sort_mode = widgets.Dropdown(
        options=["prototype", "random"],
        value="prototype",
        description="Order",
    )
    out = widgets.Output()

    def refresh(*_):
        with out:
            clear_output(wait=True)
            model_name, cluster_id = selector.value
            show_cluster_gallery(
                model_name=model_name,
                cluster_id=cluster_id,
                fitted_models=fitted_models,
                X_repr=X_repr,
                images=images,
                y_reference_labels=y_reference_labels,
                n_examples=n_examples_slider.value,
                sort_mode=sort_mode.value,
            )

    selector.observe(refresh, names="value")
    n_examples_slider.observe(refresh, names="value")
    sort_mode.observe(refresh, names="value")

    display(widgets.VBox([widgets.HBox([selector, sort_mode]), n_examples_slider, out]))
    refresh()
    return selector

# %% [markdown]
# ## 1. Load the OpenML Dataset
# 
# We use **Fashion-MNIST** from OpenML. Each row is one grayscale clothing image flattened into 784 pixel columns.
# 
# Important teaching point:
# 
# - the full dataset has `70,000` images
# - the notebook uses a fixed subset of `5,000`
# - the reference labels are used only for optional benchmarking and interpretation

# %%
X_pixels, y_reference_labels, y_reference, sample_indices, dataset_info = load_clustering_dataset()
X_images_uint8 = X_pixels.reshape(-1, 28, 28)

print(f"Dataset: {dataset_info['dataset_name']} ({dataset_info['source']})")
print(dataset_info["description"])
print(
    f"Full rows: {dataset_info['full_rows']} | Working subset: {dataset_info['sample_rows']} | "
    f"Features per image: {dataset_info['n_features']}"
)
print("The hidden reference labels are kept for later optional benchmarking and interpretation only.")

display(
    pd.DataFrame(
        {
            "openml_data_id": [dataset_info["openml_data_id"]],
            "full_rows": [dataset_info["full_rows"]],
            "working_subset": [dataset_info["sample_rows"]],
            "pixel_features": [dataset_info["n_features"]],
            "image_shape": [dataset_info["image_shape"]],
        }
    )
)

gallery_rng = np.random.default_rng(RANDOM_STATE)
gallery_idx = np.sort(gallery_rng.choice(len(X_images_uint8), size=24, replace=False))
show_image_grid(
    X_images_uint8[gallery_idx],
    titles=[f"sample idx={idx}" for idx in gallery_idx],
    n_cols=6,
    suptitle="Random images from the working subset",
)

# %% [markdown]
# ## 2. Simple Preprocessing
# 
# Preprocessing stays intentionally understandable:
# 
# 1. pixel intensities are divided by `255`, so values move to `[0, 1]`
# 2. the `784` raw pixel columns are compressed to `30` PCA components
# 3. the reduced coordinates are standardized before clustering
# 
# Why reduce first?
# 
# - raw Euclidean distance in 784 dimensions is noisy
# - PCA keeps the strongest variation patterns
# - the clustering algorithms then work on a cleaner, faster representation
# 
# Important limitation:
# 
# - PCA is a **linear** transformation of the original pixels
# - keeping only `30` components is also a **lossy compression** step
# - this can remove local structure that matters for density-based methods like `DBSCAN`
# - we still use it here because it makes the classroom example much faster and easier to inspect

# %%
X_normalized, X_pca_full, X_cluster, pca_model, reducer_scaler = preprocess_features(X_pixels)
X_images = X_normalized.reshape(-1, 28, 28)

explained_variance = np.cumsum(pca_model.explained_variance_ratio_)

summary_df = pd.DataFrame(
    {
        "rows_used_for_clustering": [len(X_cluster)],
        "raw_pixel_features": [X_pixels.shape[1]],
        "pca_components": [PCA_COMPONENTS],
        "cumulative_explained_variance": [round(float(explained_variance[-1]), 3)],
    }
)

display(summary_df)

plt.figure(figsize=(9, 4))
plt.plot(np.arange(1, PCA_COMPONENTS + 1), explained_variance, marker="o")
plt.axhline(0.8, linestyle="--", color="grey", linewidth=1)
plt.title("Cumulative explained variance of the retained PCA components")
plt.xlabel("Number of components kept")
plt.ylabel("Explained variance")
plt.tight_layout()
plt.show()

print(
    f"Cumulative explained variance after {PCA_COMPONENTS} components: "
    f"{explained_variance[-1]:.1%}"
)
print("Clustering will use the PCA representation after scaling the retained coordinates.")
print(
    "Teaching note: PCA is a linear, lossy compression of the original pixels. "
    "That helps runtime and visualization, but it can also weaken local density structure, "
    "which is one reason DBSCAN may underperform on this simplified classroom representation."
)

# %% [markdown]
# ## 3. Why Encoding Categorical Features Matters
# 
# Fashion-MNIST itself does **not** need encoding because pixel intensities are already numeric.
# 
# Still, students often ask what to do when a clustering table contains nominal categories. The short answer is:
# 
# - arbitrary label encoding like `red=0`, `blue=1`, `green=2` is usually bad for distance-based clustering
# - one-hot encoding is usually safer because it removes the fake order
# - but many one-hot columns can still distort Euclidean geometry, so encoding alone does not solve everything

# %%
category_labels = ["smooth", "rough", "striped"]

bad_encoded = np.array([0, 1, 2], dtype=float).reshape(-1, 1)
bad_distances = np.abs(bad_encoded - bad_encoded.T)

distance_table = pd.DataFrame(
    bad_distances,
    index=category_labels,
    columns=category_labels,
)

print("Bad idea: label encoding creates fake numeric distances between nominal categories.")
display(distance_table)

# %%
mixed_example = pd.DataFrame(
    {
        "mean_intensity": [0.22, 0.25, 0.71, 0.68],
        "texture": ["soft", "soft", "rigid", "matte"],
    }
)

encoded_example = pd.get_dummies(mixed_example, columns=["texture"], dtype=int)

display(mixed_example)
display(encoded_example)

print("One-hot encoding is safer than arbitrary label encoding for nominal categories.")
print(
    "But if many dummy columns appear, Euclidean distance may overreact to category mismatches, "
    "and KMeans may stop being a good default."
)

# %% [markdown]
# ## 4. Shared Diagnostics
# 
# Before fitting any clustering algorithm, inspect the geometry of the dataset.
# 
# For image data, one projection is never enough. We use:
# 
# - a PCA projection for a fast global view
# - a UMAP projection for a more neighborhood-focused non-linear view
# - later, after fitting models, cluster-based visual diagnostics and galleries
# 
# Important warning for students:
# 
# - `UMAP` is useful for visual intuition
# - but we still should not choose a clustering model from one pretty 2D picture alone
# - hidden labels are kept for later optional benchmarking, not for the first model-choice pass

# %%
X_pca_2d = X_pca_full[:, :2]
X_pca_3d = X_pca_full[:, :3]

umap_rng = np.random.default_rng(RANDOM_STATE)
umap_idx = np.sort(umap_rng.choice(len(X_cluster), size=min(UMAP_SAMPLE_SIZE, len(X_cluster)), replace=False))

umap_model = umap.UMAP(
    n_neighbors=20,
    min_dist=0.08,
    metric="euclidean",
    random_state=RANDOM_STATE,
)
X_umap = umap_model.fit_transform(X_cluster[umap_idx])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(
    x=X_pca_2d[:, 0],
    y=X_pca_2d[:, 1],
    s=24,
    alpha=0.55,
    ax=axes[0],
    color="steelblue",
)
axes[0].set_title(
    f"First two PCA components ({pca_model.explained_variance_ratio_[:2].sum():.1%} variance)"
)
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

sns.scatterplot(
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    s=24,
    alpha=0.55,
    ax=axes[1],
    color="darkorange",
)
axes[1].set_title("UMAP on the PCA representation")
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")

plt.tight_layout()
plt.show()

print(
    "Use these unlabeled projections to discuss geometry, overlap, and local neighborhoods before looking at any hidden-label benchmark."
)

# %% [markdown]
# ## 5. Choose the Number of Clusters
# 
# For centroid-based models we need a defensible `k`.
# 
# Fashion-MNIST is deliberately harder than the old wheat-kernel dataset, so the curves will not look perfect. That is useful: students should see that clustering often involves compromises rather than one obvious answer.

# %%
# TODO:
# 1. Loop over k from about 6 to 14.
# 2. Fit KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20) on X_cluster.
# 3. Store inertia, silhouette, Davies-Bouldin, and Calinski-Harabasz.
# 4. Plot at least the elbow curve and the silhouette curve.
# 5. Pick one working value of k and save it into selected_k.

selected_k = 10
print(
    "TODO: replace this fallback with your own choice after plotting the k-search diagnostics. "
    f"Current temporary value: k={selected_k}"
)

# %% [markdown]
# <details>
# <summary>Hint: one simple pattern for choosing <code>k</code></summary>
# 
# Use the same structure as in many sklearn experiments:
# 
# - create an empty list, for example `k_rows = []`
# - loop over `k` from about `6` to `14`
# - inside the loop:
#   - fit `KMeans(...)`
#   - get `labels`
#   - call `evaluate_labels(X_cluster, labels)`
#   - append one small dictionary to the list
# - convert the list to `pd.DataFrame`
# - plot `inertia` and `silhouette`
# 
# Minimal skeleton:
# 
# ```python
# k_rows = []
# for k in range(6, 15):
#     model = KMeans(...)
#     labels = model.fit_predict(X_cluster)
#     metrics = evaluate_labels(X_cluster, labels)
#     k_rows.append({
#         "k": k,
#         "inertia": model.inertia_,
#         "silhouette": metrics["silhouette"],
#     })
# ```
# 
# How to decide:
# 
# - do **not** hunt for one magical number
# - prefer a region where inertia starts flattening
# - among those candidates, pick a `k` with a reasonable silhouette and a cluster picture you can explain
# - treat hidden-label benchmark metrics, if used at all, as a later sanity check rather than the main selection rule
# </details>

# %% [markdown]
# ## How To Work In Teams
# 
# Suggested classroom split:
# 
# - **Team A** works on `KMeans` and `AgglomerativeClustering`
# - **Team B** works on `MiniBatchKMeans` and `DBSCAN`
# 
# Shared steps that everyone does together first:
# 
# 1. load the data
# 2. preprocess it
# 3. inspect PCA / UMAP
# 4. choose one common value of `k`
# 
# Then split the work:
# 
# - **Team A** chooses `best_linkage`, fits `KMeans` and `AgglomerativeClustering`, and compares their cluster pictures
# - **Team B** chooses `best_dbscan_params`, fits `MiniBatchKMeans` and `DBSCAN`, and compares their cluster pictures
# 
# Then reunite:
# 
# 1. build one shared comparison table
# 2. compare metrics and visualizations
# 3. inspect representative images from the most interesting clusters
# 
# The goal is not to prove one team is "correct". The goal is to see how different clustering ideas tell slightly different stories about the same data.

# %% [markdown]
# ## 6. Tune DBSCAN and Agglomerative Clustering
# 
# To keep the practical focused, we stop at four core model families:
# 
# - `KMeans`
# - `MiniBatchKMeans`
# - `DBSCAN`
# - `AgglomerativeClustering`
# 
# This practical works well in a **two-team format**:
# 
# - **Team A**: `KMeans` + `AgglomerativeClustering`
# - **Team B**: `MiniBatchKMeans` + `DBSCAN`
# 
# The shared tuning block here prepares both teams:
# 
# - everyone uses the same `selected_k`
# - Team A needs `best_linkage`
# - Team B needs `best_dbscan_params`
# 
# ### Team B: DBSCAN presets
# 
# Start with these four presets:
# 
# - `strict`: `eps=3.2`, `min_samples=12`
# - `balanced`: `eps=3.8`, `min_samples=8`
# - `looser`: `eps=4.2`, `min_samples=8`
# - `very loose`: `eps=4.8`, `min_samples=5`
# 
# What the parameters mean:
# 
# - `eps`: radius of the local neighborhood
# - `min_samples`: how many nearby points are needed before DBSCAN calls a region dense
# 
# How to think about them:
# 
# - smaller `eps` means stricter local neighborhoods, usually more noise and more fragmented clusters
# - larger `eps` means easier merging, usually fewer noise points but a higher risk of lumping different groups together
# - larger `min_samples` makes DBSCAN stricter
# 
# Simple tuning rule:
# 
# 1. first change `eps`
# 2. then, if the result still looks too noisy or too loose, adjust `min_samples`
# 3. prefer a setting with a readable picture, a sensible number of clusters, and not-too-extreme noise
# 
# ### Team A: Agglomerative presets
# 
# Start with these three presets:
# 
# - `ward`
# - `complete`
# - `average`
# 
# What the main parameter means:
# 
# - `linkage` tells the algorithm how to measure the distance between two existing groups before merging them
# 
# How to think about it:
# 
# - `ward`: prefers compact, variance-minimizing clusters; often the best first choice after scaling
# - `complete`: conservative merging based on the farthest pair of points; can separate groups more aggressively
# - `average`: softer compromise based on average between-cluster distance
# 
# Simple tuning rule:
# 
# 1. keep `n_clusters=selected_k`
# 2. compare a few linkage options
# 3. choose the one that gives the clearest picture and reasonable metrics, not just the top score in one column

# %%
# TODO:
# Team B can start with 3-4 manual DBSCAN settings, then edit them and rerun this cell.
# Do not over-automate this step. The point is to see how eps and min_samples change the geometry.

dbscan_try_params = [
    {"label": "strict", "eps": 3.2, "min_samples": 12},
    {"label": "balanced", "eps": 3.8, "min_samples": 8},
    {"label": "looser", "eps": 4.2, "min_samples": 8},
    {"label": "very loose", "eps": 4.8, "min_samples": 5},
]

dbscan_rows = []
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.ravel()

for ax, params in zip(axes, dbscan_try_params):
    labels = DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit_predict(X_cluster)
    metrics = evaluate_labels(X_cluster, labels)
    dbscan_rows.append(
        {
            "preset": params["label"],
            "eps": params["eps"],
            "min_samples": params["min_samples"],
            **metrics,
        }
    )
    plot_projection(
        ax,
        X_pca_2d,
        labels,
        (
            f"{params['label']}\n"
            f"eps={params['eps']}, min={params['min_samples']}\n"
            f"clusters={metrics['n_clusters']}, noise={metrics['noise_fraction']:.0%}"
        ),
        xlabel="PC1",
        ylabel="PC2",
        legend=False,
    )

plt.tight_layout()
plt.show()

dbscan_preview_df = pd.DataFrame(dbscan_rows).sort_values(
    ["silhouette", "noise_fraction"],
    ascending=[False, True],
)
display(dbscan_preview_df.round(3))

# TODO:
# After looking at the plots, choose one setting manually and save it into best_dbscan_params.
best_dbscan_params = {"eps": 3.8, "min_samples": 8}
print(
    "TODO: replace this fallback after your visual DBSCAN comparison. "
    f"Current temporary parameters: {best_dbscan_params}"
)

# %%
# TODO:
# Team A can start with 3 simple linkage options, then decide manually which one looks best.
# Do not overcomplicate this step: compare a few pictures, compare a few internal metrics, and pick one.

agglo_try_configs = [
    {"label": "ward", "linkage": "ward"},
    {"label": "complete", "linkage": "complete", "metric": "euclidean"},
    {"label": "average", "linkage": "average", "metric": "euclidean"},
]

agglo_rows = []
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, config in zip(axes, agglo_try_configs):
    labels = AgglomerativeClustering(
        n_clusters=selected_k,
        linkage=config["linkage"],
        **({"metric": config["metric"]} if "metric" in config else {}),
    ).fit_predict(X_cluster)
    metrics = evaluate_labels(X_cluster, labels)
    agglo_rows.append(
        {
            "linkage": config["label"],
            **metrics,
        }
    )
    plot_projection(
        ax,
        X_pca_2d,
        labels,
        (
            f"{config['label']} linkage\n"
            f"clusters={metrics['n_clusters']}, silhouette={metrics['silhouette']:.3f}"
        ),
        xlabel="PC1",
        ylabel="PC2",
        legend=False,
    )

plt.tight_layout()
plt.show()

linkage_df = pd.DataFrame(agglo_rows).sort_values(
    ["silhouette", "davies_bouldin"],
    ascending=[False, True],
)
display(linkage_df.round(3))

# TODO:
# After looking at the plots and the internal-metric table, choose one linkage manually.
best_linkage = "ward"
print(
    "TODO: replace this fallback after your manual agglomerative comparison. "
    f"Current temporary choice: {best_linkage}"
)

# %% [markdown]
# <details>
# <summary>Hint: how to tune DBSCAN and Agglomerative without overthinking it</summary>
# 
# For **DBSCAN**:
# 
# - start from the provided presets
# - rerun the cell after changing only one thing at a time
# - usually change `eps` first
# - only then adjust `min_samples`
# 
# What to watch:
# 
# - too much noise means the setting is too strict
# - very few big blobs usually means the setting is too loose
# - a good classroom choice is often the setting with a readable picture, moderate noise, and reasonable silhouette
# 
# For **Agglomerative**:
# 
# - compare `ward`, `complete`, and `average`
# - keep `n_clusters=selected_k`
# - choose the linkage that gives a clean picture and sensible metrics
# 
# Quick heuristic:
# 
# - `ward` is the safest first choice after scaling
# - `complete` may separate groups more aggressively
# - `average` is often a softer compromise
# </details>

# %% [markdown]
# ## 7. Fit the Core Clustering Models
# 
# We keep four core models that are easier to compare and explain in class:
# 
# - `KMeans`
# - `MiniBatchKMeans`
# - `DBSCAN`
# - `AgglomerativeClustering`
# 
# Suggested classroom flow:
# 
# 1. complete the shared setup, preprocessing, and model-selection steps together
# 2. **Team A** focuses on `KMeans` and `AgglomerativeClustering`
# 3. **Team B** focuses on `MiniBatchKMeans` and `DBSCAN`
# 4. reunite for the metric table, visual comparison, and cluster interpretation
# 
# This split is useful because the teams do **not** get two almost identical models:
# 
# - Team A compares centroid-based clustering against hierarchical clustering
# - Team B compares a fast centroid approximation against density-based clustering

# %%
model_results = []
fitted_models = {}

# TODO:
# Fit KMeans on X_cluster with your selected_k value.
# Save:
# - the fitted model into fitted_models["KMeans"]["model"]
# - the labels into fitted_models["KMeans"]["labels"]
# - one result row into model_results

print("TODO: fit KMeans and store the result in model_results and fitted_models.")

# %% [markdown]
# <details>
# <summary>Hint: reuse one template for all four model-fitting cells</summary>
# 
# The four fitting cells can all follow the same pattern:
# 
# 1. create the model
# 2. get labels with `fit_predict(...)`
# 3. evaluate with `evaluate_labels(...)`
# 4. build one result row
# 5. append it to `model_results`
# 6. save the model and labels into `fitted_models`
# 
# Minimal pattern:
# 
# ```python
# start = time.perf_counter()
# model = SomeClusteringModel(...)
# labels = model.fit_predict(X_cluster)
# row = {
#     "algorithm": "...",
#     **evaluate_labels(X_cluster, labels),
#     "fit_seconds": time.perf_counter() - start,
# }
# model_results.append(row)
# fitted_models["..."] = {"model": model, "labels": labels}
# ```
# 
# Only a few details change:
# 
# - for `KMeans` and `MiniBatchKMeans`, also store `inertia`
# - for `DBSCAN`, use `best_dbscan_params`
# - for `Agglomerative`, build `agg_kwargs` first depending on `best_linkage`
# 
# For the comparison table, the shortest safe version is:
# 
# ```python
# comparison_df = pd.DataFrame(model_results).set_index("algorithm")
# ```
# 
# Then sort it by `silhouette` and `davies_bouldin` if you want a cleaner internal-metric ranking.
# </details>

# %%
# TODO:
# Fit MiniBatchKMeans on X_cluster.
# Suggested starting point:
# MiniBatchKMeans(n_clusters=selected_k, random_state=RANDOM_STATE, batch_size=256, n_init=10)

print("TODO: fit MiniBatchKMeans and append its metrics to model_results.")

# %%
# TODO:
# Fit DBSCAN with the parameters saved in best_dbscan_params.
# Append a result row to model_results and store the fitted object in fitted_models["DBSCAN"].

print("TODO: fit DBSCAN and save its labels.")

# %%
# TODO:
# Fit AgglomerativeClustering with n_clusters=selected_k and linkage=best_linkage.
# If best_linkage is not "ward", pass metric="euclidean".
# Save the result into fitted_models["Agglomerative"] and append the metrics to model_results.

print("TODO: fit AgglomerativeClustering and save its labels.")

# %%
# TODO:
# Turn model_results into a comparison table sorted by internal quality metrics.
# Start with silhouette descending and Davies-Bouldin ascending.
# Save it into comparison_df.

if model_results:
    comparison_df = (
        pd.DataFrame(model_results)
        .sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
        .set_index("algorithm")
    )
    display(comparison_df.round(3))
else:
    print("Complete the model-fitting cells above, then build comparison_df here.")

# %% [markdown]
# ## 8. How to Read Clustering Quality Metrics
# 
# These metrics answer slightly different questions:
# 
# - `silhouette`: higher is better; it rewards compact clusters that are far apart
# - `Davies-Bouldin`: lower is better; it penalizes clusters that overlap too much
# - `Calinski-Harabasz`: higher is better; it grows when between-cluster separation is strong
# - `ARI`: optional benchmark metric; compares the clustering to the reference labels and ignores label permutations
# - `NMI`: optional benchmark metric; measures agreement with the reference labels in an information-theoretic way
# 
# No single metric is a magic answer, especially on an image dataset where class boundaries overlap.

# %%
metric_guide_df = pd.DataFrame(
    [
        {
            "metric": "silhouette",
            "direction": "higher is better",
            "how_to_read_it": "Balances cluster compactness and separation.",
        },
        {
            "metric": "davies_bouldin",
            "direction": "lower is better",
            "how_to_read_it": "Penalizes clusters that overlap or spread too much.",
        },
        {
            "metric": "calinski_harabasz",
            "direction": "higher is better",
            "how_to_read_it": "Rewards strong between-cluster separation.",
        },
        {
            "metric": "ari",
            "direction": "higher is better",
            "how_to_read_it": "Teacher-only comparison to hidden labels; label permutation safe.",
        },
        {
            "metric": "nmi",
            "direction": "higher is better",
            "how_to_read_it": "Teacher-only agreement score based on shared information.",
        },
    ]
)

display(metric_guide_df)

# %%
if not fitted_models:
    print("Complete Section 7 first, then compare the cluster projections here.")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], s=12, alpha=0.45, color="#4c72b0")
    axes[0].set_title("Shared PCA view (unlabeled baseline)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    for ax, name in zip(
        axes[1:5],
        ["KMeans", "MiniBatchKMeans", "DBSCAN", "Agglomerative"],
    ):
        if name in fitted_models:
            plot_projection(ax, X_pca_2d, fitted_models[name]["labels"], name, xlabel="PC1", ylabel="PC2", legend=False)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{name} not fitted yet", ha="center", va="center")

    axes[5].axis("off")
    axes[5].text(0.5, 0.5, "Use this panel to summarize which model looked most convincing.", ha="center", va="center")

    plt.tight_layout()
    plt.show()

# %%
if not fitted_models:
    print("Complete Section 7 first, then draw silhouette plots here.")
else:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes, ["KMeans", "DBSCAN", "Agglomerative"]):
        if name in fitted_models:
            plot_silhouette(ax, X_cluster, fitted_models[name]["labels"], f"{name} silhouette plot")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{name} not fitted yet", ha="center", va="center")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. Prototype Images and Fast-vs-Batch KMeans
# 
# At this point students should already see one important idea:
# 
# - `MiniBatchKMeans` is mainly about speed
# - the cluster interpretation still comes from looking at prototype images and representative examples
# 
# For a teaching notebook, these image summaries are often more useful than another abstract metric.

# %%
if not fitted_models:
    print("Fit the models first, then compare timings and prototype images here.")
else:
    timing_rows = []
    for name, model in [
        ("KMeans", KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=20)),
        (
            "MiniBatchKMeans",
            MiniBatchKMeans(
                n_clusters=selected_k,
                random_state=RANDOM_STATE,
                batch_size=256,
                n_init=10,
            ),
        ),
    ]:
        start = time.perf_counter()
        model.fit(X_cluster)
        timing_rows.append({"algorithm": name, "seconds": time.perf_counter() - start})

    display(pd.DataFrame(timing_rows).round(3))

    if "KMeans" in fitted_models:
        plot_cluster_mean_images(X_images, fitted_models["KMeans"]["labels"], "KMeans mean image per cluster")

# %% [markdown]
# ## 10. Interactive Cluster Explorer
# 
# This practical includes two image-centric exploration tools:
# 
# 1. a rotatable `3D` cluster map in PCA space
# 2. a widget-based cluster gallery that shows representative images from any fitted model
# 
# In Google Colab this gives students an easy way to ask: "What does cluster 4 actually look like?"
# 
# At this stage it is useful to compare the fitted clusters against the hidden labels as a post-hoc benchmark.
# The key rule is: use this block to audit and interpret the fitted models, not to replace the earlier internal-metric selection process.

# %%
if not fitted_models:
    print("Complete Section 7 first, then use the interactive cluster explorer here.")
else:
    interactive_rng = np.random.default_rng(RANDOM_STATE)
    interactive_idx = np.sort(
        interactive_rng.choice(len(X_pca_3d), size=min(INTERACTIVE_SAMPLE_SIZE, len(X_pca_3d)), replace=False)
    )

    base_model_name = "KMeans" if "KMeans" in fitted_models else next(iter(fitted_models))
    plot_interactive_3d(
        X_pca_3d[interactive_idx],
        fitted_models[base_model_name]["labels"][interactive_idx],
        f"{base_model_name} clusters in 3D PCA space (rotate this plot)",
        hover_labels=y_reference_labels.iloc[interactive_idx],
    )

    show_widget_cluster_browser(
        fitted_models=fitted_models,
        X_repr=X_cluster,
        images=X_images,
        y_reference_labels=y_reference_labels,
    )

# %% [markdown]
# ## 11. OPTIONAL: Stability Across Random Restarts
# 
# Students often ask whether the clustering would jump around if we reran the notebook.
# 
# This block is **optional** for the main classroom flow. Use it if you want to discuss reproducibility and whether one lucky random initialization can change the story.
# 
# A simple classroom answer is to repeat the algorithms that use random initialization and compare the labelings with `ARI`. That works because `ARI` is invariant to label permutations.

# %%
# TODO:
# Repeat at least KMeans and MiniBatchKMeans with different random seeds.
# Compare each rerun against the first run using adjusted_rand_score.
# Save the result in stability_df and summarize it.

print("TODO: build a restart-stability experiment here.")

# %% [markdown]
# <details>
# <summary>Hint: a compact way to do the optional stability check</summary>
# 
# You do not need a complicated experiment here.
# 
# Use only two algorithms:
# 
# - `KMeans`
# - `MiniBatchKMeans`
# 
# Simple pattern:
# 
# - create a small factory dictionary
# - for each algorithm, rerun it with seeds `0, 1, 2, ..., 9`
# - compare each run with the first run using `adjusted_rand_score`
# - also store the silhouette of each run
# 
# Skeleton:
# 
# ```python
# stability_rows = []
# for algorithm_name, factory in stability_factories.items():
#     reference_labels = None
#     for seed in range(10):
#         model = factory(seed)
#         labels = model.fit_predict(X_cluster)
#         ...
# ```
# 
# How to read it:
# 
# - if `ARI` stays close to `1`, the algorithm is stable
# - if it jumps around a lot, be careful about overinterpreting one lucky run
# </details>

# %% [markdown]
# ## 12. OPTIONAL: Representation and Distance Choices
# 
# For image clustering, representation matters almost as much as algorithm choice.
# 
# This block is **optional** for the core practical. Keep it if you want students to compare what happens when we change the feature representation or the distance metric.
# 
# Two quick benchmark checks:
#
# - compare agglomerative clustering under different distance metrics
# - compare KMeans on raw normalized pixels vs on the PCA-based clustering representation
#
# Important note:
#
# - if the clustering model uses `manhattan` or `cosine`, the silhouette should be recomputed with the same metric
# - otherwise we would be fitting in one geometry and scoring in another

# %%
if "selected_k" not in globals():
    print("Choose k first, then compare distance metrics here.")
else:
    distance_rows = []
    distance_labels = {}

    for metric_name in ["euclidean", "manhattan", "cosine"]:
        model = AgglomerativeClustering(n_clusters=selected_k, linkage="average", metric=metric_name)
        labels = model.fit_predict(X_cluster)
        distance_labels[metric_name] = labels
        benchmark_metrics = evaluate_labels(X_cluster, labels, y_reference)
        distance_rows.append(
            {
                "metric": metric_name,
                "n_clusters": benchmark_metrics["n_clusters"],
                "noise_fraction": benchmark_metrics["noise_fraction"],
                "silhouette_fit_metric": safe_silhouette_for_metric(X_cluster, labels, metric_name),
                "ari": benchmark_metrics["ari"],
                "nmi": benchmark_metrics["nmi"],
            }
        )

    distance_df = pd.DataFrame(distance_rows).sort_values("silhouette_fit_metric", ascending=False)
    display(distance_df.round(3))

# %%
if "selected_k" not in globals():
    print("Choose k first, then compare raw pixels vs PCA here.")
else:
    labels_raw = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=20).fit_predict(X_normalized)
    labels_reduced = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=20).fit_predict(X_cluster)

    dr_comparison_df = pd.DataFrame(
        [
            {
                "representation": "Raw normalized pixels",
                **evaluate_labels(X_normalized, labels_raw),
            },
            {
                "representation": "PCA + standardized coordinates",
                **evaluate_labels(X_cluster, labels_reduced),
            },
        ]
    )
    display(dr_comparison_df.round(3))

# %% [markdown]
# ## 13. Interpret the Clusters
# 
# For image clustering, a practical cluster description should combine:
# 
# 1. cluster size
# 2. a prototype or mean image
# 3. representative examples close to the cluster center
# 4. only optionally, a reference-label mix as a post-hoc benchmarking sanity check
# 
# That is a more useful description than a raw list of 784 pixel averages.

# %% [markdown]
# <details>
# <summary>Hint: how to turn clusters into a plain-language explanation</summary>
# 
# A good cluster description here is not a giant table of numbers. It is:
# 
# 1. cluster size
# 2. visual prototype / mean image
# 3. representative examples
# 4. only after that, an optional hidden-label mix if you want a benchmark sanity check
# 
# You already have a helper for the optional summary table:
# 
# ```python
# cluster_summary_df = build_cluster_summary(
#     interpretation_labels,
#     y_reference_labels,
#     X_cluster,
# )
# ```
# 
# Then inspect images:
# 
# ```python
# show_cluster_gallery(
#     interpretation_model_name,
#     cluster_id=...,
#     fitted_models=fitted_models,
#     X_repr=X_cluster,
#     images=X_images,
#     y_reference_labels=y_reference_labels,
# )
# ```
# 
# A good final sentence sounds like:
# 
# - "This cluster is compact, visually consistent, and dominated by similar shoe-like images."
# - "This cluster mixes several upper-body clothing types, so the boundary is visually ambiguous even before we check the hidden-label mix."
# </details>

# %%
# TODO:
# Pick one fitted model for interpretation, for example KMeans.
# Build a cluster_summary_df table with:
# - cluster id
# - cluster size
# - average distance to the cluster center
# - optionally, add the hidden-label mix as a post-hoc benchmark summary

if "KMeans" in fitted_models:
    interpretation_model_name = "KMeans"
    interpretation_labels = fitted_models[interpretation_model_name]["labels"]
    cluster_summary_df = build_cluster_summary(interpretation_labels, y_reference_labels, X_cluster)
    display(cluster_summary_df)
else:
    print("Fit at least one model first, then build cluster_summary_df here.")

# %%
if "KMeans" in fitted_models:
    interpretation_model_name = "KMeans"
    interpretation_labels = fitted_models[interpretation_model_name]["labels"]
    plot_cluster_mean_images(X_images, interpretation_labels, "KMeans mean image per cluster")

    # TODO:
    # Choose one pure-looking cluster and one mixed cluster.
    # Use show_cluster_gallery(...) to inspect representative images from both.
else:
    print("Fit at least one model first, then inspect prototype images here.")

# %% [markdown]
# ## 14. Common Pitfalls
# 
# Even after moving to Fashion-MNIST, the old pitfalls still matter.
# 
# Below are four common visual mistakes students should recognize immediately:
# 
# - **too few clusters**: different groups get merged into one cluster
# - **too many clusters**: one natural group gets split into artificial pieces
# - **scaling trap**: one large-scale feature dominates Euclidean distance
# - **shape trap**: `KMeans` prefers roughly spherical groups and struggles on curved structures
# 
# On real image tasks there is also a fifth pitfall: poor feature representation. That is why this notebook clusters in PCA space rather than on raw pixels only.

# %%
X_blobs, _ = make_blobs(
    n_samples=600,
    centers=[(-4, -1), (0, 3), (4, -1)],
    cluster_std=[0.9, 0.8, 1.0],
    random_state=RANDOM_STATE,
)
labels_too_few = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20).fit_predict(X_blobs)
labels_too_many = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=20).fit_predict(X_blobs)

X_unscaled, _ = make_blobs(
    n_samples=500,
    centers=3,
    cluster_std=1.0,
    random_state=RANDOM_STATE,
)
X_unscaled[:, 0] = X_unscaled[:, 0] * 100
X_unscaled_fixed = StandardScaler().fit_transform(X_unscaled)

X_moons, _ = make_moons(n_samples=500, noise=0.06, random_state=RANDOM_STATE)

labels_bad_scale = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit_predict(X_unscaled)
labels_good_scale = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit_predict(X_unscaled_fixed)
labels_bad_shape = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20).fit_predict(X_moons)
labels_good_shape = DBSCAN(eps=0.18, min_samples=5).fit_predict(X_moons)

fig, axes = plt.subplots(2, 3, figsize=(19, 12))

plot_projection(
    axes[0, 0],
    X_blobs,
    labels_too_few,
    "Too few clusters: KMeans with k=2",
    xlabel="Feature 1",
    ylabel="Feature 2",
    legend=False,
)

plot_projection(
    axes[0, 1],
    X_blobs,
    labels_too_many,
    "Too many clusters: KMeans with k=5",
    xlabel="Feature 1",
    ylabel="Feature 2",
    legend=False,
)

sns.scatterplot(
    x=X_unscaled[:, 0],
    y=X_unscaled[:, 1],
    hue=labels_bad_scale,
    palette="tab10",
    s=35,
    ax=axes[0, 2],
    legend=False,
)
axes[0, 2].set_title("Scaling trap: KMeans on unscaled data")
axes[0, 2].set_xlabel("Feature with huge scale")
axes[0, 2].set_ylabel("Small-scale feature")

sns.scatterplot(
    x=X_unscaled_fixed[:, 0],
    y=X_unscaled_fixed[:, 1],
    hue=labels_good_scale,
    palette="tab10",
    s=35,
    ax=axes[1, 0],
    legend=False,
)
axes[1, 0].set_title("Scaling fix: standardize first")
axes[1, 0].set_xlabel("Scaled feature 1")
axes[1, 0].set_ylabel("Scaled feature 2")

sns.scatterplot(
    x=X_moons[:, 0],
    y=X_moons[:, 1],
    hue=labels_bad_shape,
    palette="tab10",
    s=35,
    ax=axes[1, 1],
    legend=False,
)
axes[1, 1].set_title("Shape trap: KMeans on moons")
axes[1, 1].set_xlabel("Feature 1")
axes[1, 1].set_ylabel("Feature 2")

plot_projection(
    axes[1, 2],
    X_moons,
    labels_good_shape,
    "DBSCAN handles curved shapes better",
    xlabel="Feature 1",
    ylabel="Feature 2",
    legend=False,
)

plt.tight_layout()
plt.show()

print(
    "Visual rule of thumb: wrong k merges or splits groups, no scaling bends distance-based models, "
    "and KMeans struggles when the true structure is curved rather than roughly spherical."
)

# %% [markdown]
# ## 15. Debrief Prompts
# 
# Use these final prompts:
# 
# - Why is Fashion-MNIST much harder to cluster cleanly than `seeds`?
# - Which model gave the most convincing balance of metrics, visual structure, and interpretation?
# - Did **Team A** and **Team B** reach the same conclusion? If not, where did the stories diverge?
# - What did the manual DBSCAN parameter search teach us about density-based clustering?
# - What does the restart-stability analysis tell us about trusting one run?
# - Why did PCA help before clustering, even though PCA itself is not a clustering algorithm?
# - How would you explain one cluster to a non-technical stakeholder using images rather than feature names?
