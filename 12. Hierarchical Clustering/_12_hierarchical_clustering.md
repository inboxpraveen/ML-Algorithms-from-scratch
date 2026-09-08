# Hierarchical Clustering from Scratch: A Comprehensive Guide

Welcome to the world of Hierarchical Clustering! 🎯 In this comprehensive guide, we'll explore one of the most intuitive and powerful unsupervised learning algorithms. Think of it as the "family tree" of data clustering!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is Hierarchical Clustering?](#what-is-hierarchical-clustering)
3. [How Hierarchical Clustering Works](#how-hierarchical-clustering-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Advantages and Limitations](#advantages-and-limitations)
11. [Simplifications vs. Canonical Hierarchical Clustering](#simplifications-vs-canonical-hierarchical-clustering)
12. [Complete Usage Example](#complete-usage-example)
13. [Key Concepts to Remember](#key-concepts-to-remember)
14. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Hierarchical Clustering from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _12_hierarchical_clustering.py  (the __main__ block runs this)
# Or copy the HierarchicalClustering class from _12_hierarchical_clustering.py
# and paste it above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the HierarchicalClustering class here ----
# class HierarchicalClustering: ...

np.random.seed(42)

def cluster_purity(true_labels, pred_labels):
    """Fraction of points sitting in a cluster dominated by their own class."""
    correct = 0
    for cluster in np.unique(pred_labels):
        members = true_labels[pred_labels == cluster]
        correct += np.bincount(members).max()
    return correct / len(true_labels)

# ------ Three planted Gaussian blobs ------
blob_centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X = np.vstack([np.random.randn(30, 2) * 0.6 + c for c in blob_centers])
y = np.repeat([0, 1, 2], 30)

# Shuffle BEFORE splitting so train and test both cover all three blobs
order = np.random.permutation(len(X))
X, y = X[order], y[order]
X_train, X_test = X[:60], X[60:]
y_train, y_test = y[:60], y[60:]

# Which linkage recovers the blobs?
for linkage_name in ['single', 'complete', 'average', 'ward']:
    model = HierarchicalClustering(n_clusters=3, linkage=linkage_name)
    labels = model.fit_predict(X_train)
    heights = model.linkage_matrix_[-3:, 2]
    print(f"{linkage_name:<10} purity={cluster_purity(y_train, labels):.3f}  "
          f"last 3 merge heights = [{heights[0]:.2f}, {heights[1]:.2f}, {heights[2]:.2f}]")

# Fit once, then assign the held-out points by nearest cluster centroid
model = HierarchicalClustering(n_clusters=3, linkage='ward').fit(X_train)
test_labels = model.predict(X_test)
print(f"\nTrain purity : {cluster_purity(y_train, model.labels_):.4f}")
print(f"Test  purity : {cluster_purity(y_test, test_labels):.4f}")
for i in range(3):
    print(f"  ({X_test[i,0]:6.2f}, {X_test[i,1]:6.2f})  true blob={y_test[i]}  "
          f"predicted cluster={test_labels[i]}")

# ------ Build once, cut many times ------
cities = np.array([
    [37.77, -122.42], [34.05, -118.24], [47.61, -122.33],   # SF, LA, Seattle
    [40.71,  -74.01], [42.36,  -71.06], [38.91,  -77.04],   # NYC, Boston, DC
    [41.88,  -87.63], [44.98,  -93.27],                     # Chicago, Minneapolis
])
city_names = ['SF', 'LA', 'Seattle', 'NYC', 'Boston', 'DC', 'Chicago', 'Minneapolis']

city_model = HierarchicalClustering(n_clusters=3, linkage='average')
print("\nCity clusters:", city_model.fit_predict(cities))
print("Merge heights:", np.round(city_model.linkage_matrix_[:, 2], 2))

for k in [2, 3, 4]:
    city_model.set_n_clusters(k)          # re-cut, NO refitting
    groups = [[city_names[i] for i in range(8) if city_model.labels_[i] == c]
              for c in range(k)]
    print(f"  k={k}: " + " | ".join("{" + ", ".join(g) + "}" for g in groups))
```

Expected output:
```
single     purity=1.000  last 3 merge heights = [0.78, 3.46, 3.60]
complete   purity=1.000  last 3 merge heights = [2.67, 7.92, 8.23]
average    purity=1.000  last 3 merge heights = [1.46, 5.87, 6.04]
ward       purity=1.000  last 3 merge heights = [3.75, 25.50, 27.46]

Train purity : 1.0000
Test  purity : 1.0000
  (  3.15,   5.21)  true blob=2  predicted cluster=0
  (  3.31,   5.18)  true blob=2  predicted cluster=0
  (  6.06,  -1.19)  true blob=1  predicted cluster=1

City clusters: [0 0 0 1 1 1 2 2]
Merge heights: [ 3.38  5.21  5.6   6.44 12.   16.78 40.95]
  k=2: {SF, LA, Seattle} | {NYC, Boston, DC, Chicago, Minneapolis}
  k=3: {SF, LA, Seattle} | {NYC, Boston, DC} | {Chicago, Minneapolis}
  k=4: {SF, LA} | {Seattle} | {NYC, Boston, DC} | {Chicago, Minneapolis}
```

Two things to read off that output. First, **every** linkage gets purity 1.000 on
well-separated blobs — the linkage choice only starts to matter on harder data.

Second, look at the three merge heights on each row. They are the last three merges:
4->3 clusters, then 3->2, then 2->1. In every row the jump happens between the
**first and second** number (ward: 3.75 then 25.50), not between the second and third
(25.50 then 27.46, barely a step). Merging the last three blobs down to two is what
costs a lot, so you cut below that jump — and are left with 3 clusters. That is the
dendrogram saying "stop at 3".

---

## What is Hierarchical Clustering?

Hierarchical Clustering is an **unsupervised learning algorithm** that builds a hierarchy of clusters, creating a tree-like structure (called a dendrogram) that shows how data points are grouped at different levels of similarity.

**Real-world analogy**: 
Imagine organizing a library. You start with individual books, group similar books into topics, then group topics into categories, then categories into sections. This creates a hierarchy: Books → Topics → Categories → Sections. That's exactly how hierarchical clustering works!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Unsupervised, Hierarchical Clustering |
| **Learning Style** | Bottom-up (Agglomerative) or Top-down (Divisive) |
| **Tasks** | Clustering, Taxonomy Creation, Pattern Discovery |
| **Output** | Dendrogram (tree structure) + cluster assignments |
| **Key Advantage** | No need to specify number of clusters beforehand |

### The Core Idea

```
"Build a hierarchy of clusters by iteratively merging 
(or splitting) the most similar groups"
```

Unlike k-Means, you **don't need to specify k in advance**! You can decide how many clusters you want by "cutting" the tree at different heights.

---

## How Hierarchical Clustering Works

### Two Approaches

#### 1. Agglomerative (Bottom-Up) - Most Common

```
Start: Each point is its own cluster
     ●  ●  ●  ●  ●  ●
     ↓
Step 1: Merge two closest clusters
     (●●)  ●  ●  ●  ●
     ↓
Step 2: Continue merging
     (●●)  (●●)  ●  ●
     ↓
Step 3: Keep going
     ((●●)(●●))  (●●)
     ↓
End: All points in one cluster
     ((((●●)(●●))(●●)))
```

#### 2. Divisive (Top-Down) - Less Common

```
Start: All points in one cluster
     ((●●●●●●))
     ↓
Split into sub-clusters recursively
     (●●●)  (●●●)
     ↓
Continue until each point is alone
     ●  ●  ●  ●  ●  ●
```

**We'll focus on Agglomerative** as it's more popular and intuitive — and it is the
only one `HierarchicalClustering` implements. The divisive sketch above is background
only; there is no code behind it (see
[Simplifications vs. Canonical Hierarchical Clustering](#simplifications-vs-canonical-hierarchical-clustering)).

### The Agglomerative Algorithm in 4 Steps

```
Step 1: Initialization
        Start with n clusters (each point is a cluster)
         ↓
Step 2: Find Closest Pair
        Calculate distances between all cluster pairs
        Find the two closest clusters
         ↓
Step 3: Merge
        Combine the two closest clusters into one
         ↓
Step 4: Repeat
        Repeat Steps 2-3 until only one cluster remains
```

### Visual Example

This is the real merge trace produced by `HierarchicalClustering(linkage='average')`
(average is the class default) on these six points. Every distance below is what the
code actually prints in `linkage_matrix_[:, 2]`, and it matches
`scipy.cluster.hierarchy.linkage(X, 'average')` exactly.

```
Data Points:
    A: [1, 1]
    B: [2, 1]
    C: [1, 2]
    D: [8, 8]
    E: [9, 8]
    F: [8, 9]

Initial Clusters:
    {A}, {B}, {C}, {D}, {E}, {F}

Step 1: Merge A and B (closest pair, tied with D-E; A-B wins the tie by scan order)
    {A,B}, {C}, {D}, {E}, {F}
    Distance: 1.00

Step 2: Merge D and E (the other distance-1.0 pair)
    {A,B}, {C}, {D,E}, {F}
    Distance: 1.00

Step 3: Merge {A,B} with C
    {A,B,C}, {D,E}, {F}
    Distance: 1.21     <- mean of d(A,C)=1.00 and d(B,C)=1.41

Step 4: Merge {D,E} with F
    {A,B,C}, {D,E,F}
    Distance: 1.21     <- mean of d(D,F)=1.00 and d(E,F)=1.41

Step 5: Merge both groups
    {A,B,C,D,E,F}
    Distance: 9.93     <- mean of all 3 x 3 = 9 cross distances
```

Note that the heights only ever go **up**: 1.00, 1.00, 1.21, 1.21, 9.93. That is
guaranteed for single, complete, average and Ward linkage, and it is what makes a
dendrogram drawable. If you ever compute a merge height *lower* than the one before
it, something is wrong.

Swap the linkage and only the heights change, never this merge order:

| Linkage | Merge heights |
|---------|---------------|
| single | 1.00, 1.00, 1.00, 1.00, 9.22 |
| average | 1.00, 1.00, 1.21, 1.21, 9.93 |
| complete | 1.00, 1.00, 1.41, 1.41, 10.63 |
| ward | 1.00, 1.00, 1.29, 1.29, 17.15 |

---

## The Mathematical Foundation

### Distance Metrics

First, we need to measure distance between **individual points**:

#### 1. Euclidean Distance (Most Common)

```
d(x, y) = √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]
```

**Example**:
```python
Point A: [1, 2]
Point B: [4, 6]

d = √[(1-4)² + (2-6)²]
d = √[9 + 16]
d = √25 = 5
```

#### 2. Manhattan Distance

```
d(x, y) = |x₁-y₁| + |x₂-y₂| + ... + |xₙ-yₙ|
```

#### 3. Cosine Distance

```
d(x, y) = 1 - (x·y) / (||x|| × ||y||)
```

Cosine distance ignores vector *length* and compares only direction, which is why it
is the default for TF-IDF document vectors: a long article and a short note about the
same topic point the same way. All three of these metrics are available as
`distance_metric='euclidean' | 'manhattan' | 'cosine'`.

### Linkage Methods

Now, how do we measure distance between **clusters** (groups of points)?

#### 1. Single Linkage (Minimum)

Distance between closest points in each cluster:

```
d(C₁, C₂) = min{d(x, y) : x∈C₁, y∈C₂}
```

**Visualization**:
```
Cluster 1: ● ●
           
Cluster 2:     ● ●

Distance = shortest distance between any two points
         = distance from ● (C1) to ● (C2)
```

**Pros**: Can find elongated clusters  
**Cons**: Sensitive to noise ("chaining effect")

#### 2. Complete Linkage (Maximum)

Distance between farthest points in each cluster:

```
d(C₁, C₂) = max{d(x, y) : x∈C₁, y∈C₂}
```

**Visualization**:
```
Cluster 1: ● ●
           
Cluster 2:     ● ●

Distance = longest distance between any two points
         = distance from ● (C1) to ● (C2) (farthest)
```

**Pros**: Creates compact clusters  
**Cons**: Sensitive to outliers

#### 3. Average Linkage (UPGMA)

Average distance between all point pairs:

```
d(C₁, C₂) = (1/|C₁||C₂|) × Σ Σ d(x, y)
                           x∈C₁ y∈C₂
```

**Pros**: Balanced, robust  
**Cons**: Computationally expensive

#### 4. Ward's Method (Minimum Variance)

Minimize within-cluster variance:

```
d(C₁, C₂) = √[(2×|C₁|×|C₂|)/(|C₁|+|C₂|)] × ||μ₁ - μ₂||
```

Where μ₁, μ₂ are cluster centroids.

**Pros**: Creates very compact, balanced clusters  
**Cons**: Assumes spherical clusters

**Example Comparison** (all four numbers verified against the implementation and
against SciPy):
```python
Cluster A: [1,1], [2,2]
Cluster B: [8,8], [9,9]

The four cross distances are:
    d([1,1],[8,8]) = 9.90    d([1,1],[9,9]) = 11.31
    d([2,2],[8,8]) = 8.49    d([2,2],[9,9]) =  9.90

Single Linkage:
    min distance = d([2,2], [8,8]) = 8.49

Complete Linkage:
    max distance = d([1,1], [9,9]) = 11.31

Average Linkage:
    (9.90 + 11.31 + 8.49 + 9.90) / 4 = 9.90

Ward's Method:
    mu_A = [1.5, 1.5],  mu_B = [8.5, 8.5],  ||mu_A - mu_B|| = 9.90
    sqrt((2 * 2 * 2) / (2 + 2)) * 9.90 = sqrt(2) * 9.90 = 14.00
```

Ward's value is *larger* than the largest point-to-point distance here, which
surprises people. That is expected: Ward's height is not a distance between points
at all, it is `sqrt(2 * increase in within-cluster sum of squares)`. The
`sqrt(2*n1*n2/(n1+n2))` factor grows with cluster size, which is exactly how Ward
discourages merging two already-large clusters.

### The Dendrogram

A tree diagram showing the clustering hierarchy. This is the *actual* tree for the
six points A..F from the Visual Example under average linkage — the four heights on
the axis are the four distinct values in `linkage_matrix_[:, 2]`:

```
Height
 9.93 |     ╭───────────────────────╮
      |     │                       │
      |     │                       │
 1.21 |  ╭──┴──╮                 ╭──┴──╮
      |  │     │                 │     │
 1.00 | ╭┴╮    │                ╭┴╮    │
      | │ │    │                │ │    │
    0 | A B    C                D E    F
```

**Reading the Dendrogram**:
- **Vertical lines**: Show clusters being merged
- **Height**: Distance at which merge occurs
- **Horizontal cut**: Determines number of clusters

**Example** — every cut here is `model.set_n_clusters(k)` on the same fitted model:
```
Cut anywhere between 1.21 and 9.93 (say height 5):
    → 2 clusters: {A,B,C} and {D,E,F}          k = 2

Cut anywhere between 1.00 and 1.21 (say height 1.1):
    → 4 clusters: {A,B}, {C}, {D,E}, {F}       k = 4

Cut below 1.00 (say height 0.5):
    → 6 clusters: {A}, {B}, {C}, {D}, {E}, {F} k = 6
```

Notice there is no cut that gives 3 or 5 clusters *cleanly* here: those cuts would
have to slice through a tie (two merges at exactly the same height). The tree still
returns a 3-cluster answer if you ask for one — it just replays one of the two tied
merges and stops.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='average', 
                 distance_metric='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric
```

### Core Methods

1. **`__init__(n_clusters, linkage, distance_metric)`** - Initialize model
   - n_clusters: Number of final clusters (can be changed later)
   - linkage: How to measure cluster distance
   - distance_metric: How to measure point distance

2. **`_calculate_distance(x1, x2)`** - Private helper method
   - Computes distance between two points
   - Supports 'euclidean', 'manhattan' and 'cosine'
   - Returns a single float value

3. **`_compute_distance_matrix(X)`** - Private helper method
   - Builds the full n x n table of point-to-point distances, once
   - This is the O(n^2) memory the theory section talks about
   - Every later distance request becomes a table lookup instead of a recomputation

4. **`_calculate_cluster_distance(X, cluster1_indices, cluster2_indices)`** - Cluster distance
   - Measures distance between two clusters, given their *index lists*
   - Uses the specified linkage method
   - Core of the algorithm

5. **`fit(X)`** - Build the hierarchy
   - Merges all the way down to a single cluster, so the dendrogram is complete
   - Stores every merge in `linkage_matrix_` (SciPy-compatible)
   - Then cuts that hierarchy at `n_clusters` and fills in `labels_`

6. **`_cut_dendrogram(n_clusters)`** - Private helper method
   - Replays the first `n_samples - n_clusters` rows of `linkage_matrix_`
   - Does **not** re-cluster anything
   - Returns the list of index lists making up the partition

7. **`set_n_clusters(k)`** - Re-cut an already fitted tree
   - Changes `n_clusters` and recomputes `labels_` from the stored hierarchy
   - No refit: this is the "build once, cut many times" workflow
   - Returns `self`, so `model.set_n_clusters(5).labels_` works

8. **`predict(X)`** - Assign cluster labels
   - Passing the exact training matrix back returns the fitted `labels_`
   - For genuinely new points: nearest **training-cluster centroid**
   - (sklearn's `AgglomerativeClustering` has no `predict` at all; this is a
     practical extension, not part of the canonical algorithm)

9. **`fit_predict(X)`** - Fit and predict in one step
   - Convenience method
   - Calls `fit(X)` and returns the resulting `labels_`
   - Returns cluster labels

10. **`get_linkage_matrix()`** - Get merge history
    - Returns dendrogram structure
    - Compatible with scipy for visualization
    - Shows which clusters merged and when

---

## Step-by-Step Example

Let's walk through a complete example clustering **cities** based on coordinates:

### The Data

```python
import numpy as np

# City locations: [latitude, longitude] (simplified)
X = np.array([
    # West Coast cities
    [37.77, -122.42],  # San Francisco
    [34.05, -118.24],  # Los Angeles
    [47.61, -122.33],  # Seattle
    
    # East Coast cities
    [40.71, -74.01],   # New York
    [42.36, -71.06],   # Boston
    [38.91, -77.04],   # Washington DC
    
    # Midwest cities
    [41.88, -87.63],   # Chicago
    [44.98, -93.27],   # Minneapolis
])

city_names = ['SF', 'LA', 'Seattle', 'NYC', 'Boston', 'DC', 'Chicago', 'Minneapolis']
```

### Training the Model

```python
# Paste the HierarchicalClustering class from _12_hierarchical_clustering.py above

# Create model
model = HierarchicalClustering(
    n_clusters=3,  # Want 3 regions
    linkage='average',
    distance_metric='euclidean'
)

# Fit and predict
labels = model.fit_predict(X)
```

### What Happens Internally

Every distance below is the real value from `model.get_linkage_matrix()[:, 2]`,
and it matches `scipy.cluster.hierarchy.linkage(X, 'average')` to 15 decimals.

**Iteration 0** - Initial state:
```
Clusters: {SF}, {LA}, {Seattle}, {NYC}, {Boston}, {DC}, {Chicago}, {Minneapolis}
```

**Iteration 1** - Find closest pair:
```
All 28 pair distances calculated...
Closest: NYC and Boston (distance = 3.38)
Merge: {SF}, {LA}, {Seattle}, {NYC, Boston}, {DC}, {Chicago}, {Minneapolis}
```

**Iteration 2** - Next closest:
```
Closest: {NYC, Boston} and DC (distance = 5.21)
    = mean of d(NYC,DC)=3.52 and d(Boston,DC)=6.90
Merge: {SF}, {LA}, {Seattle}, {NYC, Boston, DC}, {Chicago}, {Minneapolis}
```

**Iteration 3**:
```
Closest: SF and LA (distance = 5.60)
Merge: {SF, LA}, {Seattle}, {NYC, Boston, DC}, {Chicago}, {Minneapolis}
```

**Iteration 4**:
```
Closest: Chicago and Minneapolis (distance = 6.44)
Merge: {SF, LA}, {Seattle}, {NYC, Boston, DC}, {Chicago, Minneapolis}
```

**Continue until...**
```
Iteration 5: {SF, LA} + Seattle                             at 12.00
Iteration 6: {NYC, Boston, DC} + {Chicago, Minneapolis}     at 16.78
Iteration 7: everything                                     at 40.95

Full height sequence: 3.38, 5.21, 5.60, 6.44, 12.00, 16.78, 40.95
                                        ^^^^^^^^^^^ the big gaps live here

Cut just below 12.00 -> 4 clusters
Cut just below 16.78 -> 3 clusters:
    Cluster 0: {SF, LA, Seattle}         <- West Coast
    Cluster 1: {NYC, Boston, DC}         <- East Coast
    Cluster 2: {Chicago, Minneapolis}    <- Midwest
Cut just below 40.95 -> 2 clusters (West Coast vs everything else)
```

Note that the heights never decrease. Average linkage (UPGMA) is *monotone*: a merge
can never happen at a lower height than the merge before it. Seeing 2.8 then 4.5 then
3.2 would be a sign of a broken implementation, not of interesting data.

### Results

```python
print("Cluster assignments:", labels)
# Output: [0 0 0 1 1 1 2 2]

for cluster in range(3):
    cities_in_cluster = [city_names[i] for i in range(len(labels)) if labels[i] == cluster]
    print(f"Cluster {cluster}: {', '.join(cities_in_cluster)}")

# Output:
# Cluster 0: SF, LA, Seattle
# Cluster 1: NYC, Boston, DC
# Cluster 2: Chicago, Minneapolis
```

Changed your mind about `k`? The hierarchy is already built, so ask it a different
question — no refitting:

```python
model.set_n_clusters(4)
print(model.labels_)
# Output: [0 0 1 2 2 2 3 3]
#   -> {SF, LA}, {Seattle}, {NYC, Boston, DC}, {Chicago, Minneapolis}

model.set_n_clusters(2)
print(model.labels_)
# Output: [0 0 0 1 1 1 1 1]
#   -> {SF, LA, Seattle}, {everything east of the Rockies}
```

---

## Real-World Applications

### 1. **Document Organization**
Organize documents into topics and subtopics:
- Input: Document text features
- Output: Hierarchical topic structure
- Example: "News → Politics → Elections → Local Elections"

### 2. **Species Classification**
Create biological taxonomy:
- Input: Genetic or morphological features
- Output: Evolutionary tree (phylogenetic tree)
- Example: "Animals → Mammals → Primates → Humans"

### 3. **Social Network Analysis**
Discover community structure:
- Input: User connections and interactions
- Output: Nested communities
- Example: "University → Departments → Research Groups → Teams"

### 4. **Image Segmentation**
Group similar regions in images:
- Input: Pixel colors and positions
- Output: Hierarchical image regions
- Example: "Scene → Objects → Parts → Pixels"

### 5. **Customer Segmentation**
Create detailed market segments:
- Input: Customer behavior, demographics
- Output: Nested customer groups
- Example: "Customers → High Value → Premium → VIP"

### 6. **Gene Expression Analysis**
Group genes with similar functions:
- Input: Gene expression levels
- Output: Gene hierarchy by function
- Example: "Genes → Metabolism → Energy → ATP Production"

### 7. **Product Categorization**
Organize products for e-commerce:
- Input: Product attributes
- Output: Category hierarchy
- Example: "Electronics → Computers → Laptops → Gaming Laptops"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Distance Calculation

```python
def _calculate_distance(self, x1, x2):
    if self.distance_metric == 'euclidean':
        # Euclidean distance: sqrt(sum((x1 - x2)^2))
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif self.distance_metric == 'manhattan':
        # Manhattan distance: sum(|x1 - x2|)
        return np.sum(np.abs(x1 - x2))
    elif self.distance_metric == 'cosine':
        # Cosine distance: 1 - (x1 . x2) / (||x1|| * ||x2||)
        norm_product = np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2))
        if norm_product == 0:
            return 0.0 if np.allclose(x1, x2) else 1.0
        return 1.0 - np.sum(x1 * x2) / norm_product
    else:
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")
```

These three lines up with the three formulas in
[The Mathematical Foundation](#the-mathematical-foundation) one for one.

**How it works**:
```python
# Euclidean example
x1 = [1, 2]
x2 = [4, 6]

diff = x1 - x2 = [-3, -4]
squared = diff² = [9, 16]
sum_squared = 25
distance = √25 = 5
```

### 2. Cluster Distance (Linkage)

Note the signature: clusters are passed as **index lists**, not as arrays of points.
That is what lets each loop *read* `d(x, y)` out of the cached `n x n` table
`self._distance_matrix_` instead of recomputing it on every merge step.

```python
def _calculate_cluster_distance(self, X, cluster1_indices, cluster2_indices):
    cluster1_points = X[cluster1_indices]
    cluster2_points = X[cluster2_indices]

    # distances[i, j] = d(point i, point j), computed once in fit()
    distances = self._distance_matrix_
    if distances is None or distances.shape[0] != X.shape[0]:
        distances = self._compute_distance_matrix(X)

    if self.linkage == 'single':
        # d(C1, C2) = min d(x, y), x in C1, y in C2
        min_distance = float('inf')
        for i1 in cluster1_indices:
            for i2 in cluster2_indices:
                distance = distances[i1, i2]
                min_distance = min(min_distance, distance)
        return min_distance

    elif self.linkage == 'complete':
        # d(C1, C2) = max d(x, y), x in C1, y in C2
        max_distance = 0
        for i1 in cluster1_indices:
            for i2 in cluster2_indices:
                distance = distances[i1, i2]
                max_distance = max(max_distance, distance)
        return max_distance

    elif self.linkage == 'average':
        # d(C1, C2) = (1 / (|C1| * |C2|)) * sum of all cross distances
        total_distance = 0
        count = 0
        for i1 in cluster1_indices:
            for i2 in cluster2_indices:
                distance = distances[i1, i2]
                total_distance += distance
                count += 1
        return total_distance / count if count > 0 else 0

    elif self.linkage == 'ward':
        # d(C1, C2) = sqrt(2*|C1|*|C2| / (|C1|+|C2|)) * ||mu1 - mu2||
        # Ward needs CENTROIDS, so it works from the raw points, not the table.
        centroid1 = np.mean(cluster1_points, axis=0)
        centroid2 = np.mean(cluster2_points, axis=0)

        n1 = len(cluster1_indices)
        n2 = len(cluster2_indices)

        distance = np.sqrt((2.0 * n1 * n2) / (n1 + n2)) * \
                   self._calculate_distance(centroid1, centroid2)
        return distance

    else:
        raise ValueError(f"Unknown linkage method: {self.linkage}")
```

Each of the first three branches is the matching formula from
[Linkage Methods](#linkage-methods) transcribed literally: walk every pair
`(x in C1, y in C2)` and take the min, the max, or the mean. The Ward branch is the
one line of real mathematics in the file - the
`sqrt(2*n1*n2/(n1+n2)) * ||mu1 - mu2||` formula from
[Ward's Method](#4-wards-method-minimum-variance) written out directly.

**Example** (`X = [[1,1],[2,2],[8,8],[9,9]]`, cluster A = indices `[0,1]`,
cluster B = indices `[2,3]`):
```python
Single: min(d([1,1],[8,8]), d([1,1],[9,9]),
            d([2,2],[8,8]), d([2,2],[9,9]))
      = min(9.90, 11.31, 8.49, 9.90) = 8.49

Complete: max(...) = 11.31

Average: (9.90 + 11.31 + 8.49 + 9.90) / 4 = 9.90

Ward: sqrt(2*2*2 / (2+2)) * ||[1.5,1.5] - [8.5,8.5]||
    = sqrt(2) * 9.90 = 14.00
```

### 3. The Main Algorithm Loop

This is the body of `fit`. Two details matter and are easy to get wrong:

- The loop runs **down to one cluster**, not down to `n_clusters`. The whole
  dendrogram gets built regardless of `k`, which is what makes re-cutting free.
- The merged cluster is written back at position `merge_i` (the *lower* of the two
  positions), not appended to the end. Appending would scramble the cluster
  numbering so that the last cluster formed would get the highest label.

```python
# Start with each point as its own cluster
clusters = [[i] for i in range(self.n_samples_)]

# SciPy ID convention: points keep 0..n-1, merge number m creates ID n + m
cluster_ids = list(range(self.n_samples_))
next_cluster_id = self.n_samples_
distance_cache = {}

# Merge until a single cluster remains -> the COMPLETE dendrogram
while len(clusters) > 1:
    # Find closest pair
    min_distance = float('inf')
    merge_i, merge_j = -1, -1

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            pair_key = (cluster_ids[i], cluster_ids[j])
            if pair_key not in distance_cache:
                distance_cache[pair_key] = self._calculate_cluster_distance(
                    X, clusters[i], clusters[j]
                )
            distance = distance_cache[pair_key]

            if distance < min_distance:
                min_distance = distance
                merge_i, merge_j = i, j

    # Record the merge for the dendrogram
    self.linkage_matrix_.append([
        min(cluster_ids[merge_i], cluster_ids[merge_j]),
        max(cluster_ids[merge_i], cluster_ids[merge_j]),
        min_distance,
        len(clusters[merge_i]) + len(clusters[merge_j])
    ])

    # Merge the closest clusters IN PLACE at the lower position
    clusters[merge_i] = clusters[merge_i] + clusters[merge_j]
    cluster_ids[merge_i] = next_cluster_id
    clusters.pop(merge_j)
    cluster_ids.pop(merge_j)
    next_cluster_id += 1
```

**Why the `distance_cache`?** After a merge, only the pairs involving the brand-new
cluster have changed. Every other cluster pair holds the same points it held a moment
ago, so its distance is still valid. Reusing them is pure bookkeeping - the merge
choices are identical either way - but it is the difference between a fit that takes
0.4 seconds on 150 points and one that takes 40.

**Step-by-step**:
```python
# Initial: 6 points
clusters    = [[0], [1], [2], [3], [4], [5]]
cluster_ids = [ 0,   1,   2,   3,   4,   5 ]

# Iteration 1: closest pair is (0, 1) at distance 1.0
clusters    = [[0,1], [2], [3], [4], [5]]
cluster_ids = [  6,    2,   3,   4,   5 ]
linkage_matrix_ = [[0, 1, 1.0, 2]]

# Iteration 2: closest pair is (3, 4) at distance 1.0
clusters    = [[0,1], [2], [3,4], [5]]
cluster_ids = [  6,    2,    7,    5 ]
linkage_matrix_ = [[0, 1, 1.0, 2], [3, 4, 1.0, 2]]

# And so on, until one cluster with ID 10 remains.
```

### 4. Creating Labels from Hierarchy

Once `fit` has the full hierarchy, `_cut_dendrogram(k)` replays the first
`n_samples - k` recorded merges and stops. Nothing is re-clustered:

```python
def _cut_dendrogram(self, n_clusters):
    clusters = [[i] for i in range(self.n_samples_)]
    cluster_ids = list(range(self.n_samples_))
    next_cluster_id = self.n_samples_

    for row in self.linkage_matrix_[:self.n_samples_ - n_clusters]:
        id_a, id_b = int(row[0]), int(row[1])
        pos_a = cluster_ids.index(id_a)
        pos_b = cluster_ids.index(id_b)
        low, high = min(pos_a, pos_b), max(pos_a, pos_b)

        clusters[low] = clusters[low] + clusters[high]
        cluster_ids[low] = next_cluster_id
        clusters.pop(high)
        cluster_ids.pop(high)
        next_cluster_id += 1

    return clusters
```

The resulting index lists become the label array (this lives at the tail of `fit`,
and is re-run by `set_n_clusters`):

```python
def _labels_from_clusters(self, clusters):
    labels = np.zeros(self.n_samples_, dtype=int)

    for cluster_id, cluster_indices in enumerate(clusters):
        for idx in cluster_indices:
            labels[idx] = cluster_id

    return labels
```

`predict(X)` is a **different** thing. It returns `labels_` when handed the training
matrix back, and otherwise assigns each new point to the nearest *training-cluster
centroid*:

```python
cluster_centers = []
for cluster_indices in self._final_clusters:
    # NOTE: self._X_fit, the TRAINING data - _final_clusters indexes into it
    cluster_centers.append(np.mean(self._X_fit[cluster_indices], axis=0))
```

### 5. Building the Linkage Matrix

```python
# Store merge history
# Format: [cluster_i, cluster_j, distance, size]
linkage_matrix = []

for each merge:
    linkage_matrix.append([
        cluster_i,      # First cluster
        cluster_j,      # Second cluster
        merge_distance, # Distance at merge
        new_size        # Size of merged cluster
    ])
```

**The ID rule** is what makes this a valid SciPy linkage matrix, and it is the single
easiest thing to get wrong: the `n` original points are IDs `0 .. n-1`, and the
cluster produced by **row m** is given ID `n + m`. So the two IDs on row `m` are
always strictly less than `n + m`, and no ID may ever appear twice in columns 0-1.

**Linkage Matrix Example** - the real output of
`HierarchicalClustering(linkage='average').fit(X).get_linkage_matrix()` on the
six points A..F from the Visual Example (`n = 6`):
```
[[ 0.  1.  1.0000  2.]    # A + B          -> new cluster ID 6
 [ 3.  4.  1.0000  2.]    # D + E          -> new cluster ID 7
 [ 2.  6.  1.2071  3.]    # C + cluster 6  -> new cluster ID 8
 [ 5.  7.  1.2071  3.]    # F + cluster 7  -> new cluster ID 9
 [ 8.  9.  9.9331  6.]]   # cluster 8 + cluster 9 -> ID 10, the root
```

Because the IDs obey that rule, this matrix can be handed straight to
`scipy.cluster.hierarchy.dendrogram`, `cophenet` and `fcluster`.

---

## Model Evaluation

### 1. Dendrogram Visualization

The best way to evaluate hierarchical clustering!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

X = np.array([
    [37.77, -122.42], [34.05, -118.24], [47.61, -122.33],
    [40.71,  -74.01], [42.36,  -71.06], [38.91,  -77.04],
    [41.88,  -87.63], [44.98,  -93.27],
])
city_names = ['SF', 'LA', 'Seattle', 'NYC', 'Boston', 'DC', 'Chicago', 'Minneapolis']

model = HierarchicalClustering(n_clusters=3, linkage='average').fit(X)

# Get linkage matrix
linkage_matrix = model.get_linkage_matrix()

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=city_names)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cities')
plt.ylabel('Distance')
plt.show()
```

This works because `get_linkage_matrix()` returns a *valid* SciPy linkage matrix
(see the ID rule above) - SciPy will refuse to draw anything else.

**What to look for**:
- **Long vertical lines**: Good natural clusters
- **Short vertical lines**: Similar points, good merge
- **Height of cuts**: Determines number of clusters

### 2. Cophenetic Correlation

Measures how well dendrogram preserves pairwise distances:

```
cophenetic_correlation = correlation(original_distances, dendrogram_distances)
```

**Range**: -1 to +1
- **> 0.8**: Excellent preservation
- **0.7 - 0.8**: Good
- **< 0.7**: Poor fit

**Example** (continuing with the city model above):
```python
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(linkage_matrix, pdist(X))
print(f"Cophenetic correlation: {c:.3f}")

# Output: Cophenetic correlation: 0.920 (Excellent!)
```

For reference, the same measurement on the standardised Iris data with Ward linkage
gives `0.823` - still good, but Ward deliberately distorts distances in exchange for
compact clusters, so it always scores below average linkage on this metric.

### 3. Silhouette Score

Same as k-Means - measures cluster quality:

```
Silhouette = (b - a) / max(a, b)

where:
    a = average distance to points in same cluster
    b = average distance to points in nearest other cluster
```

**Range**: -1 to +1
- **> 0.7**: Excellent
- **0.5 - 0.7**: Good
- **< 0.5**: Poor

### 4. Calinski-Harabasz Index

Ratio of between-cluster to within-cluster dispersion:

**Higher is better**

### 5. Davies-Bouldin Index

Average similarity between each cluster and its most similar cluster:

**Lower is better**

### Choosing Number of Clusters

#### Method 1: Visual Inspection of Dendrogram

```
Look for "big jumps" in merge distances:

Height
  10 |       │         ← Big jump here!
   8 |       │         
   6 |       ├───      ← Small jumps
   5 |   ┌───┤         
   3 |   │   │     
   1 | ┌─┴─┐ │     
   0 | A B C D E   

Cut before the big jump → 2 clusters
```

#### Method 2: Elbow Method

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=120, centers=4, n_features=2,
                  cluster_std=0.8, random_state=42)

# BUILD ONCE - the hierarchy does not depend on k at all
model = HierarchicalClustering(n_clusters=2, linkage='ward').fit(X)

# CUT MANY TIMES - set_n_clusters re-cuts the stored tree, no refitting
k_values = list(range(2, 10))
silhouette_scores = []
for k in k_values:
    labels = model.set_n_clusters(k).labels_
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot and find elbow
optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"Optimal k: {optimal_k}")

# Output: Optimal k: 4
```

Doing it this way is not just tidier, it is 8x less work: one fit instead of eight.

#### Method 3: Domain Knowledge

```
Use business understanding:
- "We have 3 regions: West, Central, East"
- "Products fit into 5 categories"
```

---

## Advantages and Limitations

### Advantages ✅

1. **No Need to Specify k in Advance**
   - Explore different numbers of clusters
   - Can decide after seeing dendrogram
   - More flexible than k-Means

2. **Produces Dendrogram**
   - Visual representation of hierarchy
   - Shows relationships between clusters
   - Interpretable structure

3. **Deterministic**
   - Same input → Same output
   - No random initialization
   - Reproducible results

4. **Works with Any Distance Metric**
   - The algorithm only ever needs `d(x, y)`, so any metric plugs in
   - This implementation ships `'euclidean'`, `'manhattan'` and `'cosine'`;
     adding another means one more branch in `_calculate_distance`
   - Flexible for different data types
   - Caveat: Ward is the exception. Its formula is derived from Euclidean sums of
     squares, so `fit` raises `ValueError` for `linkage='ward'` with any other
     metric (exactly as scikit-learn does)

5. **Captures Hierarchy**
   - Natural for nested structures
   - Shows multi-level relationships
   - Useful for taxonomy

6. **Handles Non-Spherical Clusters**
   - Single linkage can find elongated clusters
   - More flexible than k-Means
   - No shape assumptions

### Limitations ❌

1. **Computationally Expensive**
   - O(n²) space for distance matrix
   - O(n³) time for naive implementation
   - Slow on large datasets (> 10,000 points)

2. **Cannot Undo Merges**
   - Once merged, cannot split
   - Early mistakes propagate
   - Can lead to poor final clusters

3. **Sensitive to Noise and Outliers**
   - Especially with single linkage
   - Outliers can distort structure
   - May need preprocessing

4. **Difficulty Handling Different Sizes**
   - Some linkages prefer balanced clusters
   - May not work well with varied cluster sizes
   - Ward's method assumes similar sizes

5. **Choosing Linkage is Tricky**
   - Different linkages give different results
   - No universal "best" linkage
   - Requires domain knowledge or experimentation

6. **Memory Requirements**
   - Stores entire distance matrix (`self._distance_matrix_`, built once in `fit`)
   - O(n²) memory - 10,000 points is already an 800 MB float64 table
   - Prohibitive for very large datasets

### When to Use Hierarchical Clustering

**Good Use Cases**:
- ✅ Small to medium datasets (< 10,000 points)
- ✅ Need to see hierarchy of relationships
- ✅ Don't know number of clusters in advance
- ✅ Want deterministic results
- ✅ Creating taxonomies or dendrograms
- ✅ Analyzing biological/genealogical data

**Bad Use Cases**:
- ❌ Large datasets (> 50,000 points)
- ❌ Need real-time clustering
- ❌ Flat clustering sufficient
- ❌ Memory constrained environments
- ❌ Need to update clusters incrementally

### Hierarchical vs k-Means

| Aspect | Hierarchical | k-Means |
|--------|-------------|---------|
| **Specify k** | No | Yes |
| **Deterministic** | Yes | No (random init) |
| **Speed** | Slow O(n³) | Fast O(nkdi) |
| **Memory** | High O(n²) | Low O(n) |
| **Hierarchy** | Yes | No |
| **Large data** | ❌ | ✅ |
| **Dendrogram** | ✅ | ❌ |

---

## Simplifications vs. Canonical Hierarchical Clustering

The clustering itself is not simplified: merge heights match
`scipy.cluster.hierarchy.linkage` to within 4e-15 for single, complete, average and
Ward under euclidean, manhattan and cosine, and the partitions match
`sklearn.cluster.AgglomerativeClustering` with an Adjusted Rand Index of 1.000.
What is left out is around the edges:

| Canonical behaviour | Here | Consequence |
|---------------------|------|-------------|
| Divisive (top-down) clustering as an alternative strategy | Not implemented - only agglomerative | The divisive section above is conceptual background, not code you can call |
| `distance_threshold` cut ("cut at height 7", sklearn's option) | Cut by `n_clusters` only | To cut at a height, read `linkage_matrix_[:, 2]` and count how many merges sit below your threshold, then pass that `k` to `set_n_clusters` |
| SciPy's nearest-neighbour-chain algorithm, O(n²) time | Plain scan of all cluster pairs at every step, O(n³) distance lookups | Fine to a few hundred points (n=200 fits in about 1 second); SciPy stays fast into the tens of thousands |
| Centroid and median linkage (which can produce *inversions* - a merge lower than the one before it) | Not implemented | Every hierarchy this class produces is monotone, so it is always drawable as a dendrogram |
| `AgglomerativeClustering` has **no** `predict` | `predict` is provided, using nearest training-cluster centroid | Convenient, but it is an extension. A point predicted into cluster 2 would not necessarily have joined cluster 2 had it been present during `fit` |
| Connectivity constraints (only merge neighbours in a graph) | Not implemented | Cannot be used for the connectivity-constrained image-segmentation variant |

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate sample data
X, true_labels = make_blobs(
    n_samples=150,
    centers=3,
    n_features=2,
    cluster_std=0.8,
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']

print("Comparing Linkage Methods:\n")
print(f"{'Linkage':<15} {'Silhouette':<15} {'Notes':<30}")
print("-" * 60)

for linkage in linkage_methods:
    model = HierarchicalClustering(
        n_clusters=3,
        linkage=linkage,
        distance_metric='euclidean'
    )
    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    
    notes = {
        'single': 'Good for elongated clusters',
        'complete': 'Creates compact clusters',
        'average': 'Balanced approach',
        'ward': 'Minimizes variance'
    }
    
    print(f"{linkage:<15} {silhouette:<15.3f} {notes[linkage]:<30}")

# Use best performing linkage
print("\n" + "="*60)
print("Training final model with Ward linkage...")
print("="*60 + "\n")

model = HierarchicalClustering(
    n_clusters=3,
    linkage='ward',
    distance_metric='euclidean'
)

labels = model.fit_predict(X_scaled)

# Evaluate
silhouette = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette:.3f}")

# Analyze clusters
print("\nCluster Analysis:")
for cluster in range(3):
    cluster_data = X[labels == cluster]
    print(f"\nCluster {cluster}:")
    print(f"  Size: {len(cluster_data)} points")
    print(f"  Center: {np.mean(cluster_data, axis=0)}")
    print(f"  Std Dev: {np.std(cluster_data, axis=0)}")

# Visualize dendrogram
print("\nGenerating dendrogram...")
linkage_matrix = model.get_linkage_matrix()

plt.figure(figsize=(12, 5))

# Plot 1: Dendrogram
plt.subplot(1, 2, 1)
from scipy.cluster.hierarchy import dendrogram
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Plot 2: Clusters
plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green']
for cluster in range(3):
    cluster_points = X[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
               c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
plt.title('Cluster Assignments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

# Try different numbers of clusters
print("\n" + "="*60)
print("Finding optimal number of clusters...")
print("="*60 + "\n")

k_range = list(range(2, 8))
scores = []

print(f"{'k':<5} {'Silhouette':<15}")
print("-" * 20)

# Note: no refitting here. `model` is already fitted, and set_n_clusters just
# re-cuts the hierarchy it already built - the whole point of the algorithm.
for k in k_range:
    labels = model.set_n_clusters(k).labels_
    score = silhouette_score(X_scaled, labels)
    scores.append(score)
    print(f"{k:<5} {score:<15.3f}")

optimal_k = k_range[np.argmax(scores)]
print(f"\nOptimal number of clusters: {optimal_k}")
```

**Expected output** (the plotting call opens a window; everything else prints):
```
Comparing Linkage Methods:

Linkage         Silhouette      Notes                         
------------------------------------------------------------
single          0.877           Good for elongated clusters   
complete        0.877           Creates compact clusters      
average         0.877           Balanced approach             
ward            0.877           Minimizes variance            

============================================================
Training final model with Ward linkage...
============================================================

Silhouette Score: 0.877

Cluster Analysis:

Cluster 0:
  Size: 50 points
  Center: [-2.66301205  8.9251602 ]
  Std Dev: [0.66565758 0.76449609]

Cluster 1:
  Size: 50 points
  Center: [-6.77912214 -6.88758091]
  Std Dev: [0.89912136 0.79441173]

Cluster 2:
  Size: 50 points
  Center: [4.59523718 2.10914946]
  Std Dev: [0.71362662 0.80855562]

Generating dendrogram...

============================================================
Finding optimal number of clusters...
============================================================

k     Silhouette     
--------------------
2     0.701          
3     0.877          
4     0.717          
5     0.551          
6     0.368          
7     0.370          

Optimal number of clusters: 3
```

All four linkages score identically here because the three blobs are so cleanly
separated that every method finds exactly the same partition. That is the *expected*
result on easy data - the linkage choice only starts to matter when clusters touch,
overlap or are non-convex (see USAGE EXAMPLE 3 in the `.py`, where single linkage
recovers two interleaved moons perfectly while scoring the worst silhouette).

---

## Key Concepts to Remember

### 1. **No Need to Specify k**
Unlike k-Means, you can decide the number of clusters after seeing the dendrogram!

### 2. **Linkage Method Matters**
- Single: Elongated clusters, sensitive to noise
- Complete: Compact clusters, sensitive to outliers
- Average: Balanced, good default
- Ward: Compact, assumes similar sizes

### 3. **Build Once, Cut Many Times**
Build the hierarchy once with `fit(X)`, then try different numbers of clusters with
`set_n_clusters(k)`. `fit` always merges all the way down to one cluster, so the
whole tree is stored in `linkage_matrix_` and re-cutting is just a replay of the
recorded merges — no distance is recomputed.

```python
import numpy as np
np.random.seed(0)
X = np.vstack([np.random.randn(10, 2) + c for c in [[0, 0], [6, 0], [3, 5]]])

model = HierarchicalClustering(n_clusters=3, linkage='ward').fit(X)   # one fit
for k in range(2, 8):
    print(k, model.set_n_clusters(k).labels_)                         # six cuts
```

### 4. **Dendrogram is Key**
The dendrogram visualization is the most important tool for understanding and evaluating your clustering.

### 5. **Computational Cost**
O(n²) space, O(n³) time - only practical for datasets with < 10,000 points.

### 6. **Deterministic**
Same data + same parameters = same results (no random initialization).

### 7. **Cannot Undo**
Once two clusters are merged, they cannot be split. Early mistakes propagate!

---

## Conclusion

Hierarchical Clustering is a powerful and intuitive algorithm for discovering nested structures in data! By understanding:
- How the algorithm builds a hierarchy of clusters
- How different linkage methods affect results
- How to read and interpret dendrograms
- How to choose the right number of clusters
- When hierarchical clustering is appropriate

You've gained a valuable tool for exploratory data analysis! 🎯

**When to Use Hierarchical Clustering**:
- ✅ Creating taxonomies or hierarchies
- ✅ Small to medium datasets
- ✅ Unknown number of clusters
- ✅ Need interpretable structure
- ✅ Biological/genealogical analysis

**When to Use Something Else**:
- ❌ Large datasets → Use k-Means, DBSCAN
- ❌ Need speed → Use k-Means, Mini-Batch k-Means
- ❌ Flat clustering sufficient → Use k-Means
- ❌ Memory constrained → Use online algorithms
- ❌ Need probabilistic assignments → Use GMM

**Next Steps**:
- Try hierarchical clustering on your own datasets
- Experiment with different linkage methods
- Compare results with k-Means
- Learn to interpret dendrograms effectively
- Explore advanced methods (BIRCH for large datasets)
- Study cophenetic correlation
- Try different distance metrics

Happy Clustering! 💻🎯

