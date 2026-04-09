# Lecture 09: Clustering

This directory contains the lecture materials for this topic.

## Core Files

- `lecture_notes.md`
- `links.yaml`
- `slides/lecture.pdf`
- `lecture_examples/README.md`
- `practical_session/README.md`

## Lecture Examples

- `lecture_examples/example_01.ipynb` and `lecture_examples/example_01.py`: Import Libs. Clustering workflow with internal metrics, visualization, and multiple algorithms.

## Practical Session

- `practical_session/clustering_practical_student_90min.ipynb`: public student version with targeted TODO cells and built-in hints
- `practical_session/README.md`: practical overview, scope, and runtime notes

The instructor notebook and cheat sheet are maintained separately and are not part of the current public student release.

The practical covers:

- preprocessing and clustering on an OpenML Fashion-MNIST image subset
- PCA, UMAP, and interactive 3D diagnostics
- `KMeans`, `MiniBatchKMeans`, `DBSCAN`, and `AgglomerativeClustering`
- internal clustering metrics with optional benchmark-label checks
- a two-team classroom flow: Team A works with `KMeans` and `AgglomerativeClustering`, Team B works with `MiniBatchKMeans` and `DBSCAN`
- image-based cluster interpretation, interactive cluster galleries, common clustering pitfalls, and optional extensions on stability and representation choices

The practical session is intentionally separate from `lecture_examples/`.
