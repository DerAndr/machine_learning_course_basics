# Clustering Practical Session

This directory contains a 90-minute classroom practical for Lecture 09.

## Files

- `clustering_practical_student_90min.ipynb`
- `clustering_practical_student_90min.py`
- `README.md`

An internal instructor notebook and cheat sheet exist, but they are not part of the current public student release.

## Format

- The student notebook contains targeted TODO placeholders in the model-selection, algorithm-fitting, comparison, stability, and interpretation cells.
- The Python companion script mirrors the notebook structure for lighter review and diffing.
- The student notebook follows the same classroom flow as the internal instructor version, including a short `How To Work In Teams` section.
- The practical uses the OpenML dataset `Fashion-MNIST` (`data_id=40996`): 70,000 grayscale `28 x 28` clothing images across 10 classes.
- To keep the practical responsive in Google Colab, the notebook works with a fixed random subset of `5,000` images.
- The shared opening section covers:
  - explicit dataset loading
  - simple preprocessing: pixel normalization, PCA to a lower-dimensional image representation, and scaling of the reduced coordinates
  - a short example showing why label encoding is dangerous for nominal categories, and why one-hot is safer but not a universal fix for Euclidean clustering
  - PCA and UMAP diagnostics on image embeddings
  - a rotatable interactive 3D cluster map
  - choosing `k` with elbow and silhouette
  - short manual preset exploration for both `DBSCAN` and agglomerative clustering, with parameter explanations for students
- The session covers these clustering model families:
  - `KMeans`
  - `MiniBatchKMeans`
  - `DBSCAN`
  - `AgglomerativeClustering`
- The practical works well in a two-team classroom mode:
  - Team A: `KMeans` + `AgglomerativeClustering`
  - Team B: `MiniBatchKMeans` + `DBSCAN`
- The final sections cover interactive image galleries, cluster interpretation with prototype images, visually obvious clustering mistakes, clustering pitfalls, and a short debrief discussion.
- The practical also includes:
  - a short metric guide for silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, and NMI
  - an **optional** stability check across repeated random restarts
  - an **optional** block on representation and distance choices
  - a compact workflow for turning image clusters into plain-language descriptions using size, mean image, representative examples, and optional benchmark-label mix

## Teaching Intent

- Move from a tiny, almost toy-sized dataset to a realistic image-clustering task without making the notebook too heavy for class.
- Show that clustering quality should be judged with metrics and diagnostics, not only by a 2D scatter plot.
- Contrast centroid-based, density-based, and hierarchical clustering in one coherent notebook.
- Make students explain what the clusters mean with images and prototypes, not only produce labels.
- End with common failure modes so students leave with realistic caution about scaling, geometry, and parameter choice.

## Environment

Run this practical with the baseline repository environment:

- `uv sync`

If you run in Google Colab, install:

- `openml`
- `plotly`
- `ipywidgets`
- `umap-learn`
