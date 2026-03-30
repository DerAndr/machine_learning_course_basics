# Lecture 09 Recap: Clustering

> Lecture number: 09
> Lecture slug: `lecture_09_clustering`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What Clustering Solves

Clustering is an **unsupervised learning** task.

Unlike classification:

- there are no target labels
- the goal is not to predict a known class
- the goal is to discover structure in unlabeled data

The lecture defines clustering as grouping data points based on similarity.

Typical use cases from the slides:

- customer segmentation
- image segmentation
- document grouping
- anomaly detection
- grouping words, sentences, or documents

This is an important mindset shift for students:

- in clustering, the structure is inferred rather than supervised
- there may be multiple plausible clusterings depending on the metric, algorithm, and business goal

## 2. Clustering in the Broader ML Landscape

The slides contrast:

- supervised learning
- unsupervised learning
- semi-supervised learning

Clustering belongs to the unsupervised family because:

- the dataset has no label telling us the "correct" cluster
- the algorithm must organize the data by internal structure

This also means evaluation is harder than in supervised learning.

## 3. What Makes a Good Cluster?

The lecture frames clustering around a simple intuition:

- points within the same cluster should be similar
- points in different clusters should be dissimilar

But "similar" can mean different things:

- close in Euclidean distance
- high density neighborhood
- high cosine similarity
- same statistical distribution
- overlapping membership with probabilities
- domain-specific heuristic relationship

This is one of the main conceptual lessons of the lecture:

- clustering is not one problem with one universal definition
- the algorithm should match the geometry and meaning of the data

## 4. Main Clustering Principles

The slides identify several clustering logics:

- distance-based
- density-based
- similarity-based
- probability-based
- domain-specific rule-based

This is an excellent overview because students often think only of K-Means.

## 5. Distance and Similarity Measures

Distance choice is foundational in clustering.

The lecture lists several measures:

- Euclidean distance
- Manhattan distance
- Minkowski distance
- Jaccard distance
- Cosine similarity
- Hamming distance
- Mahalanobis distance

Students should not memorize every formula, but should understand when each family makes sense.

### 5.1 Euclidean Distance

Best for:

- continuous numerical data
- roughly spherical geometry

Important caution:

- scale matters a lot

If one feature has a much larger numeric range than another, Euclidean distance becomes dominated by that feature.

### 5.2 Manhattan Distance

Uses absolute coordinate differences instead of squared ones.

Often useful for:

- grid-like geometry
- sparse settings
- cases where large individual coordinate differences should not be over-penalized as strongly as in Euclidean distance

### 5.3 Minkowski Distance

A general family:

- Euclidean is a special case
- Manhattan is another special case

### 5.4 Cosine Similarity

Best for:

- high-dimensional sparse data
- text vectors
- cases where vector direction matters more than magnitude

This is very common in NLP-style clustering.

### 5.5 Jaccard and Hamming

Useful for:

- binary or categorical structures

Jaccard focuses on overlap between sets.
Hamming counts coordinate mismatches.

### 5.6 Mahalanobis Distance

Important when:

- features are correlated

Unlike Euclidean distance, Mahalanobis adjusts for covariance structure.

That means it can treat a point as "less unusual" if it lies along a correlated direction of natural variation.

## 6. Major Families of Clustering Algorithms

The lecture provides a broad taxonomy:

- centroid-based
- hierarchical
- density-based
- graph-based
- distribution-based
- fuzzy or soft clustering
- grid-based
- semi-supervised
- constraint-based

This is useful because it shows that clustering is a family of design choices, not just one algorithm.

## 7. Hard vs Soft Clustering

The slides explicitly distinguish **hard** and **soft** clustering.

### Hard Clustering

Each point belongs to exactly one cluster.

Examples:

- K-Means
- hierarchical clustering
- DBSCAN

### Soft Clustering

A point can belong to multiple clusters with different degrees or probabilities.

Examples:

- Gaussian Mixture Models
- Fuzzy C-Means

This distinction matters when cluster boundaries are:

- overlapping
- ambiguous
- naturally probabilistic

## 8. K-Means Clustering

K-Means is one of the most important algorithms in the lecture and one of the most widely used clustering methods overall.

Its goal is to partition the data into `k` non-overlapping clusters by minimizing within-cluster variance.

The slides express this through **WCSS**:

- Within-Cluster Sum of Squares

This objective measures cluster compactness around centroids.

### 8.1 Objective Intuition

Lower WCSS means:

- points are closer to their assigned centroids
- clusters are tighter and more compact

K-Means is therefore fundamentally a centroid-based optimization method.

### 8.2 K-Means Algorithm Steps

The lecture gives the standard loop:

1. choose initial centroids
2. assign each point to the nearest centroid
3. recompute centroids as cluster means
4. repeat until convergence

This is a very important pattern for students:

- assignment step
- update step

K-Means alternates between them until centroid movement becomes small enough.

### 8.3 Strengths of K-Means

- simple
- fast
- scalable
- works well when clusters are compact and roughly spherical

### 8.4 Limitations of K-Means

The lecture emphasizes:

- sensitive to initialization
- requires choosing `k`
- struggles with overlapping or non-spherical clusters
- sensitive to outliers

This is exactly right.

K-Means also tends to prefer clusters with:

- similar size
- similar variance
- roughly convex shape

So it is a strong baseline, but not a universal solution.

## 9. Computational Complexity of K-Means

The slides give total complexity approximately as:

- `O(t * n * d * k)`

where:

- `n` is number of points
- `d` is dimensionality
- `k` is number of clusters
- `t` is number of iterations

The main takeaway is practical:

- K-Means is usually efficient on moderate tabular datasets
- but costs increase with more clusters, more features, and worse initialization

## 10. Hierarchical Clustering

Hierarchical clustering builds a tree-like structure called a **dendrogram**.

The lecture explains two broad directions:

- agglomerative: start with single points and merge
- divisive: start with one cluster and split

The practical focus is mainly on agglomerative clustering.

## 11. Linkage Criteria

The lecture carefully lists several linkage methods.

### 11.1 Single Linkage

Distance between two clusters is the minimum distance between any pair of points across the clusters.

Effect:

- can produce elongated or chain-like clusters

### 11.2 Complete Linkage

Distance between clusters is the maximum pairwise distance.

Effect:

- tends to produce tighter, more compact clusters

### 11.3 Average Linkage

Uses the average of all pairwise distances across the two clusters.

Effect:

- balances flexibility and compactness

### 11.4 Ward's Method

Merges clusters in a way that minimizes increase in within-cluster variance.

Effect:

- often works well for compact, spherical clusters

This linkage discussion is important because "hierarchical clustering" is not one single behavior. Linkage choice strongly changes the result.

## 12. Strengths and Limits of Hierarchical Clustering

### Strengths

- no need to predefine `k`
- dendrogram provides visual structure
- captures nested cluster relationships
- can handle nontrivial shapes depending on linkage

### Limitations

- high computational cost
- sensitive to outliers
- merges are irreversible
- difficult in very high dimensions
- dendrogram cutting can be subjective

The lecture’s emphasis on irreversibility is especially important:

- a bad early merge cannot be corrected later

## 13. DBSCAN

DBSCAN is one of the most conceptually important alternatives to K-Means.

Instead of centroids, it uses **density**.

The lecture defines three point types:

- core points
- border points
- noise points

And two main parameters:

- `epsilon`
- `MinPts`

### 13.1 Core Idea

Clusters are dense connected regions.

This has a major practical advantage:

- DBSCAN can find arbitrarily shaped clusters
- it can identify outliers directly

That makes it very different from K-Means.

### 13.2 Core, Border, and Noise Points

### Core Points

Points with at least `MinPts` neighbors within radius `epsilon`.

### Border Points

Points that are not core points themselves, but lie in the neighborhood of a core point.

### Noise Points

Points that are neither core nor border points.

These are treated as outliers.

### 13.3 Strengths of DBSCAN

- no need to choose `k`
- handles arbitrary cluster shapes
- naturally separates noise
- useful for anomaly-heavy datasets

### 13.4 Limitations of DBSCAN

The lecture highlights:

- difficult parameter tuning
- struggles with varying density
- high-dimensional issues
- border-point ambiguity

This is a very accurate practical summary.

DBSCAN works best when:

- density scale is reasonably consistent
- the geometry of clusters is not spherical
- noise handling is valuable

## 14. Fuzzy C-Means

Fuzzy C-Means is the main soft-clustering algorithm in the lecture.

Instead of assigning each point to exactly one cluster, it gives each point a **membership degree** for each cluster.

So the output is not only:

- cluster centers

but also:

- a membership matrix

This is useful when cluster boundaries overlap and a hard assignment would be too rigid.

### 14.1 Fuzziness Parameter `m`

The lecture highlights parameter `m`.

Interpretation:

- larger `m` gives softer clustering with more overlap
- lower `m` makes the solution closer to hard clustering

This is a very important parameter because it controls how ambiguous membership is allowed to be.

### 14.2 Strengths

- handles overlapping clusters
- flexible
- useful in image segmentation and pattern recognition

### 14.3 Limitations

- computationally heavier
- sensitive to initialization
- sensitive to `m` and number of clusters
- affected by noise

So fuzzy clustering is conceptually powerful, but also more delicate.

## 15. Comparative View of Algorithms

The lecture’s comparison table is useful as a decision guide.

### K-Means

Best for:

- large numerical datasets
- fast baseline partitioning

### Hierarchical

Best for:

- nested structure
- exploratory analysis
- cases where dendrogram insight matters

### DBSCAN

Best for:

- irregular shapes
- outlier-heavy data
- cases where `k` is unknown

### Gaussian Mixture Models

Best for:

- probabilistic cluster assignments
- roughly Gaussian-shaped components

### Fuzzy C-Means

Best for:

- overlapping cluster membership
- ambiguous boundaries

### Spectral and Mean Shift

These appear in the comparison table as more advanced options for non-convex structure or automatic mode-finding behavior.

Students do not need to master them now, but should know they exist when K-Means and DBSCAN fail.

## 16. Internal Clustering Metrics

Because clustering usually has no labels, internal metrics are crucial.

The lecture lists:

- WCSS
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

### 16.1 WCSS

Measures compactness around centroids.

Useful especially for:

- K-Means
- elbow-method inspection

Lower is better, but:

- WCSS almost always drops as `k` increases

So it cannot be used alone without model-selection logic.

### 16.2 Silhouette Score

The lecture gives the standard form:

- compare cohesion within the assigned cluster
- against separation from the nearest alternative cluster

Range:

- from `-1` to `1`

Interpretation:

- higher is better
- near zero suggests overlapping clusters
- negative values suggest poor assignments

This is one of the most useful all-purpose internal metrics.

### 16.3 Davies-Bouldin Index

Measures similarity between each cluster and its most similar neighbor, balancing dispersion and separation.

Interpretation:

- lower is better

### 16.4 Calinski-Harabasz Index

Measures between-cluster dispersion relative to within-cluster dispersion.

Interpretation:

- higher is better

This metric rewards:

- compact clusters
- good separation

## 17. External Clustering Metrics

When ground-truth labels exist for evaluation, the lecture lists:

- Rand Index
- Adjusted Rand Index
- Normalized Mutual Information
- V-Measure

This is useful in educational settings and benchmark datasets such as Iris, where labels exist but are not used during clustering.

### 17.1 Rand Index

Measures pairwise agreement between clustering assignments and ground truth.

### 17.2 Adjusted Rand Index

Corrects Rand Index for chance agreement.

This makes ARI much more meaningful than plain RI in many practical comparisons.

### 17.3 NMI

Measures shared information between clustering assignments and true labels.

Range:

- `0` to `1`

Higher means stronger correspondence.

### 17.4 V-Measure

Balances:

- homogeneity
- completeness

This provides another principled external score when ground-truth labels are available.

## 18. Choosing the Number of Clusters

This is one of the most practical parts of clustering work.

The lecture suggests several tools:

- elbow method
- domain knowledge
- silhouette score
- Davies-Bouldin / Calinski-Harabasz
- dendrogram cutting

This is exactly the right answer:

- there is rarely one universally correct `k`
- the choice should combine metric evidence and domain meaning

## 19. High-Dimensional Data

The lecture emphasizes the curse of dimensionality.

In high dimensions:

- distance becomes less meaningful
- computation becomes harder
- cluster structure may be harder to recover

Suggested solutions:

- PCA
- t-SNE
- UMAP
- feature selection

Important nuance:

- t-SNE and UMAP are often more useful for visualization than for the actual clustering objective
- PCA can help denoise and compress while preserving more global structure

## 20. Feature Scaling

The lecture explicitly reminds students that scaling matters in clustering.

This is critical.

Distance-based clustering can be badly distorted if one feature dominates numerically.

Common methods:

- standardization
- min-max scaling

This is often a mandatory preprocessing step before:

- K-Means
- hierarchical clustering with distance metrics
- DBSCAN

## 21. Outliers and Noise

The lecture notes that outliers can:

- distort centroids
- damage cluster shape
- degrade metric quality

Suggested responses:

- outlier detection beforehand
- robust algorithms such as DBSCAN
- use of K-Medoids rather than centroid means in some settings

This is especially important for K-Means, which is quite sensitive to extreme points.

## 22. Initialization Sensitivity

The lecture correctly highlights initialization as a major issue, especially for K-Means and fuzzy methods.

Bad initialization can lead to:

- worse local optima
- slower convergence
- unstable results across runs

Recommended techniques:

- smarter initialization
- multiple restarts

This is why methods like `k-means++` are so useful in practice.

## 23. Scalability and Efficiency

The slides also address computational efficiency.

Strategies mentioned:

- sampling
- incremental clustering
- parallel processing

This is important because some clustering methods are excellent conceptually but too expensive on large datasets.

Students should learn to ask not only:

- does the algorithm fit the geometry?

but also:

- does it fit the scale of the data?

## 24. Advanced and Practical Extensions

The lecture’s advanced section is quite good.

It mentions:

- Gower distance for mixed data
- K-Prototypes
- clustering as preprocessing
- domain knowledge integration
- semi-supervised clustering
- constrained clustering
- hyperparameter tuning
- cluster profiling
- PCA / t-SNE / UMAP visualizations

The big idea is:

- clustering is rarely the end of the workflow
- it is often part of a broader analytical or feature-engineering pipeline

## 25. Clustering as Preprocessing

The lecture explicitly mentions uses such as:

- anomaly detection
- data compression
- feature engineering through cluster memberships

This is a very good practical note.

Clusters can be used as:

- segmentation outputs
- synthetic categorical features
- exploratory summaries

## 26. Practical Notebook Map

### `Clustering Demo.ipynb`

This is the main structured practice notebook.

It covers:

- loading and scaling the Iris dataset
- running several clustering methods
- computing internal metrics
- using true Iris labels only for comparison and visualization
- PCA-based visualization of clustering outcomes

This is useful because it connects theory directly to:

- metric calculation
- algorithm comparison
- 2D visual interpretation

### `Clustering-illustrations.ipynb`

This notebook is more visual and intuition-driven.

It contains:

- synthetic cluster examples
- K-Means illustrations
- Gaussian Mixture illustrations
- hierarchical clustering visuals and dendrogram ideas
- DBSCAN neighborhood and density illustrations
- fuzzy-clustering demos

Its main value is conceptual:

- students can see how cluster shape, initialization, and algorithm assumptions affect the result

## 27. Best Practices

The final slide gives a very solid checklist.

- understand feature types and data geometry first
- preprocess carefully
- scale where appropriate
- reduce dimensionality when needed
- choose the algorithm to match the data, not by habit
- tune hyperparameters systematically
- validate with suitable metrics
- interpret clusters in domain context

This is exactly the right operational attitude for clustering.

## 28. Key Takeaways

- Clustering is an unsupervised task for organizing unlabeled data into meaningful groups.
- Distance or similarity choice strongly affects the result.
- K-Means is fast and useful, but assumes compact centroid-based structure.
- Hierarchical clustering gives a dendrogram and does not require `k` upfront.
- DBSCAN is powerful for irregular shapes and noise handling.
- Fuzzy C-Means is useful when cluster membership is ambiguous or overlapping.
- Internal metrics evaluate compactness and separation without labels.
- External metrics are useful only when reference labels exist for evaluation.
- Scaling, dimensionality reduction, and outlier handling are often essential preprocessing steps.
- There is no single best clustering algorithm for all datasets.

## 29. Quick Revision Questions

1. Why is clustering considered unsupervised learning?
2. When would cosine similarity be preferable to Euclidean distance?
3. Why does K-Means struggle with non-spherical clusters?
4. What does a dendrogram provide that K-Means does not?
5. Why can DBSCAN detect outliers naturally?
6. What is the difference between hard and soft clustering?
7. Why is silhouette score often more informative than WCSS alone?
8. When should ARI be preferred over raw Rand Index?
9. Why is feature scaling so important for clustering?
10. Why is there often no single objectively correct number of clusters?
