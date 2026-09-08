# k-Means Clustering from Scratch: A Comprehensive Guide

Welcome to the world of k-Means Clustering! 🎯 In this comprehensive guide, we'll explore one of the most popular unsupervised machine learning algorithms. Think of it as the "find natural groups in your data" algorithm!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is k-Means Clustering?](#what-is-k-means-clustering)
3. [How k-Means Works](#how-k-means-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Choosing the Right k](#choosing-the-right-k)
11. [Feature Scaling: Important for k-Means](#feature-scaling-important-for-k-means)
12. [Advantages and Limitations](#advantages-and-limitations)
13. [Complete Usage Example](#complete-usage-example)
14. [Simplifications vs. Canonical k-Means](#simplifications-vs-canonical-k-means)
15. [Key Concepts to Remember](#key-concepts-to-remember)
16. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# k-Means Clustering from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _10_kmeans_clustering.py  (the __main__ block runs this)
# Or copy the KMeansClustering class from _10_kmeans_clustering.py above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the KMeansClustering class here (from _10_kmeans_clustering.py) ----
# class KMeansClustering: ...

np.random.seed(42)

# ------ 1. RECOVER THREE PLANTED BLOBS ------
# We know the answer in advance, so we can check whether k-Means finds it.
true_centers = np.array([[0.0, 0.0], [7.0, 7.0], [0.0, 7.0]])
X = np.vstack([c + np.random.randn(100, 2) for c in true_centers])

# Shuffle before slicing, so the held-out points come from all three blobs
order = np.random.permutation(len(X))
X = X[order]
X_train, X_test = X[:240], X[240:]

km = KMeansClustering(n_clusters=3, init='kmeans++', n_init=10, random_state=42)
km.fit(X_train)

print(f"Converged in {km.n_iter_} iterations")
print(f"Train inertia          : {km.inertia_:.2f}")
# score() returns NEGATIVE inertia; divide by n to compare train against test
print(f"Train mean sq distance : {-km.score(X_train) / len(X_train):.4f}")
print(f"Test  mean sq distance : {-km.score(X_test) / len(X_test):.4f}")

print("\nRecovered centroids (should match the planted centers):")
print(km.get_cluster_centers().round(2))
print("Cluster sizes:", np.bincount(km.labels_))

# ------ 2. ELBOW METHOD: HOW MANY CLUSTERS? ------
print("\nk    inertia")
for k in range(1, 7):
    m = KMeansClustering(n_clusters=k, init='kmeans++', n_init=10, random_state=42)
    m.fit(X_train)
    print(f"{k}   {m.inertia_:9.2f}")

# ------ 3. COLOR QUANTIZATION ------
reds = np.column_stack([np.random.randint(190, 231, 150),
                        np.random.randint(30, 71, 150),
                        np.random.randint(30, 71, 150)])
greens = np.column_stack([np.random.randint(30, 71, 150),
                          np.random.randint(190, 231, 150),
                          np.random.randint(50, 91, 150)])
blues = np.column_stack([np.random.randint(20, 61, 150),
                         np.random.randint(60, 101, 150),
                         np.random.randint(190, 231, 150)])
pixels = np.vstack([reds, greens, blues]).astype(float)

palette = KMeansClustering(n_clusters=3, init='kmeans++', n_init=10, random_state=42)
labels = palette.fit_predict(pixels)

print("\nDominant colors (RGB):")
for i, c in enumerate(palette.get_cluster_centers().astype(int)):
    print(f"  Color {i}: RGB({c[0]:3d}, {c[1]:3d}, {c[2]:3d})  "
          f"used by {np.sum(labels == i)} pixels")

print("score(X) = %.2f (negative inertia: higher is better)" % palette.score(pixels))

new_pixels = np.array([[205.0, 45.0, 45.0], [45.0, 205.0, 70.0], [35.0, 80.0, 210.0]])
print("Unseen pixels assigned to colors:", palette.predict(new_pixels))
```

Expected output:
```
Converged in 2 iterations
Train inertia          : 444.36
Train mean sq distance : 1.8515
Test  mean sq distance : 1.9754

Recovered centroids (should match the planted centers):
[[ 7.07  6.99]
 [-0.14  6.89]
 [-0.07  0.06]]
Cluster sizes: [75 82 83]

k    inertia
1     5674.27
2     2347.67
3      444.36
4      386.49
5      336.43
6      284.53

Dominant colors (RGB):
  Color 0: RGB(209,  51,  50)  used by 150 pixels
  Color 1: RGB( 40,  80, 210)  used by 150 pixels
  Color 2: RGB( 49, 209,  69)  used by 150 pixels
score(X) = -182838.43 (negative inertia: higher is better)
Unseen pixels assigned to colors: [0 2 1]
```

Two things worth noticing straight away:

- The recovered centroids `(7.07, 6.99)`, `(-0.14, 6.89)`, `(-0.07, 0.06)` are the three planted centers `(7,7)`, `(0,7)`, `(0,0)` to within ~0.15 - that is the known-answer test that proves the implementation works.
- The **cluster IDs are arbitrary**. Cluster 0 happens to be the `(7,7)` blob here; a different seed would number the same geometry differently. Always read the centroid, never trust a specific ID.

---

## What is k-Means Clustering?

k-Means is an **unsupervised learning algorithm** that groups similar data points into k clusters. Unlike supervised learning (where we have labels), k-Means discovers patterns in data without being told what to look for!

**Real-world analogy**: 
Imagine organizing your closet. You naturally group similar items together - shirts with shirts, pants with pants, shoes with shoes. You don't need someone to label each item; you just see the similarities and create groups. That's exactly how k-Means works!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Unsupervised, Partitional Clustering |
| **Learning Style** | Iterative optimization |
| **Tasks** | Clustering, Pattern Discovery, Segmentation |
| **Output** | k cluster assignments + k centroids |
| **Key Parameter** | k (number of clusters) |

### The Core Idea

```
"Group data into k clusters where points in the same cluster 
are similar, and points in different clusters are dissimilar"
```

k-Means finds cluster "centers" (centroids) and assigns each point to the nearest center!

---

## How k-Means Works

### The Algorithm in 4 Steps

```
Step 1: Initialize
        Randomly choose k points as initial cluster centers (centroids)
         ↓
Step 2: Assignment
        Assign each data point to the nearest centroid
         ↓
Step 3: Update
        Move each centroid to the mean of its assigned points
         ↓
Step 4: Repeat
        Repeat Steps 2-3 until centroids stop moving
```

### Visual Example

```
Initial Setup:
    
    Data Points: ●●●  ■■■  ▲▲▲
    Random Centroids: X₁  X₂  X₃
    
    
Iteration 1 - Assignment:
    
    ●●● → X₁    (closest to X₁)
    ■■■ → X₂    (closest to X₂)
    ▲▲▲ → X₃    (closest to X₃)
    

Iteration 1 - Update:
    
    Move X₁ to center of ●●●
    Move X₂ to center of ■■■
    Move X₃ to center of ▲▲▲
    

Iteration 2 - Assignment:
    
    Reassign points to new nearest centroids
    (some points might switch clusters)
    

Continue until convergence...
    
Final Result:
    
    Cluster 1: ●●● with centroid X₁
    Cluster 2: ■■■ with centroid X₂
    Cluster 3: ▲▲▲ with centroid X₃
```

### Why "k-Means"?

- **k**: The number of clusters you want to find
- **Means**: Each cluster center is the mean (average) of points in that cluster

```
k=2: Split data into 2 groups
k=3: Split data into 3 groups
k=5: Split data into 5 groups
```

**Important**: You must specify k before running the algorithm!

---

## The Mathematical Foundation

### Distance Metric

k-Means uses **Euclidean distance** to measure similarity:

```
distance(x, centroid) = √[(x₁-c₁)² + (x₂-c₂)² + ... + (xₙ-cₙ)²]
```

**Example**:
```python
Point: [3, 4]
Centroid: [0, 0]

distance = √[(3-0)² + (4-0)²]
        = √[9 + 16]
        = √25 = 5
```

### Assignment Step

Assign each point to the cluster with the nearest centroid:

```
cluster(x) = argmin distance(x, centroidₖ)
             k=1..K
```

In plain English: "Which centroid is closest to this point?"

**Example**:
```
Point x: [5, 5]
Centroid 1: [2, 2]  → distance = 4.24
Centroid 2: [8, 8]  → distance = 4.24
Centroid 3: [5, 1]  → distance = 4.00

Assign to Cluster 3 (smallest distance)
```

### Update Step

Move each centroid to the mean of its assigned points:

```
centroidₖ = (1/nₖ) × Σ(all points in cluster k)
```

**Example**:
```
Cluster 1 points: [1,1], [2,2], [3,3]

New centroid = ([1+2+3]/3, [1+2+3]/3)
             = (2, 2)
```

#### Why the *mean*? (the one line of calculus that makes k-Means click)

The mean is not an arbitrary "sensible-looking" summary - it is the **exact minimizer** of the objective once the cluster membership is fixed. Hold the assignments still and ask: which point c makes the total squared distance to every member of cluster k as small as possible?

```
f(c) = Σ ||x - c||²        (sum over x in Cₖ)

Differentiate with respect to c and set to zero:

df/dc = Σ -2(x - c) = 0
      → Σ x  =  nₖ · c
      → c    =  (1/nₖ) Σ x      ← the arithmetic mean
```

f is a sum of convex quadratics, so this stationary point is the *global* minimum, not just a local one.

**That is exactly why the update step in the code is `np.mean(cluster_points, axis=0)` and nothing else.** The choice of squared Euclidean distance in the assignment step is what forces the mean in the update step - the two halves of the algorithm are not independent design decisions. Change the distance and the summary changes with it: minimizing the sum of *absolute* distances gives the **median** instead, which is the k-Medians algorithm.

### Objective Function

k-Means minimizes the **within-cluster sum of squares (WCSS)**, also called **inertia**:

```
J = Σ Σ ||x - centroidₖ||²
    k x∈Cₖ
```

Where:
- J = Total inertia (smaller is better)
- Cₖ = Set of points in cluster k
- ||·|| = Euclidean distance

**Interpretation**:
- Lower inertia = tighter, more compact clusters
- Higher inertia = loose, spread-out clusters

**Example**:
```
Cluster 1: [1,1], [2,2]  with centroid [1.5, 1.5]
    Distance 1: √[(1-1.5)² + (1-1.5)²] = 0.707
    Distance 2: √[(2-1.5)² + (2-1.5)²] = 0.707
    
    Cluster inertia = 0.707² + 0.707² = 1.0

Total Inertia = Sum of all cluster inertias
```

### Convergence

The algorithm stops when:

1. **Centroids stop moving** (or move very little)
   ```
   ||new_centroids - old_centroids|| < tolerance
   ```

2. **Maximum iterations reached**
   ```
   iteration_count >= max_iter
   ```

#### Why convergence is *guaranteed*

People often repeat "k-Means always converges" without saying why. The argument is short and worth knowing, because it also tells you exactly what k-Means does *not* guarantee.

```
1. The assignment step cannot increase J.
   Each point moves to its nearest centroid. If it stays put, its
   contribution ||x - cₖ||² is unchanged; if it switches, it switches
   precisely because the new centroid is closer, so its term shrinks.

2. The update step cannot increase J either.
   By the derivation above, the mean is the global minimizer of
   Σ||x - c||² for a fixed membership. Replacing the old centroid with
   the mean can only lower (or tie) each cluster's contribution.

3. So J never increases, and J >= 0 is bounded below.

4. There are only finitely many ways to partition n points into k
   groups (fewer than kⁿ). Each time the partition actually changes,
   J strictly drops - so no partition can ever be revisited. With only
   finitely many partitions available, the changes must stop after
   finitely many steps. Once the partition stops changing the means
   stop changing too, the shift falls to 0, and the loop exits.
```

**What this does NOT promise**: that the J you land on is the *smallest possible* J. Steps 1-4 only say you roll downhill until you can no longer move; which valley you end up in depends entirely on where you started. That is why the implementation has `init='kmeans++'` (start in a good place) and `n_init` (start in several places and keep the best). See [Simplifications vs. Canonical k-Means](#simplifications-vs-canonical-k-means).

**A note on `tol` in this implementation**: the stopping test compares the raw Frobenius norm `||new_centroids - old_centroids||` of the whole stacked centroid matrix against `tol` - an absolute distance in the units of your features. scikit-learn instead compares the *squared* shift against `tol` scaled by the mean feature variance, so the same numeric `tol=1e-4` is a **stricter** test here. In practice it rarely matters: on standardized iris, given the same initial centroids, this implementation and scikit-learn both take 7 iterations and both land on inertia 140.0328. The difference only surfaces when the centroids creep toward their final position instead of snapping into it. See [Simplifications vs. Canonical k-Means](#simplifications-vs-canonical-k-means).

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class KMeansClustering:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, 
                 init='random', random_state=None, n_init=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.n_init = n_init
```

### Core Methods

1. **`__init__(n_clusters, max_iter, tol, init, random_state, n_init)`** - Initialize model
   - n_clusters: Number of clusters (k)
   - max_iter: Maximum iterations
   - tol: Convergence tolerance on centroid movement
   - init: Initialization method ('random' or 'kmeans++')
   - random_state: Seed for a *private* RandomState (your global `np.random` stream is never disturbed)
   - n_init: How many times to restart from a fresh initialization, keeping the lowest-inertia run (default 1; scikit-learn defaults to 10)

2. **`_initialize_centroids(X)`** - Private helper method
   - Initialize k centroids using chosen method
   - Random: Pick k random points
   - k-means++: Smart initialization for faster convergence

3. **`_assign_clusters(X)`** - Assignment step
   - Assign each point to nearest centroid
   - Returns array of cluster labels
   - Core of the algorithm

4. **`_update_centroids(X, labels)`** - Update step
   - Calculate new centroid positions
   - Each centroid = mean of assigned points
   - If a cluster ends up empty, its centroid is relocated to the point currently farthest from its own centroid (same rule scikit-learn uses)
   - Returns new centroid positions

5. **`_calculate_inertia(X, labels)`** - Calculate quality metric
   - Sum of squared distances to centroids
   - Lower = better clustering
   - Used for evaluation, for the elbow method, and to pick the best of the `n_init` restarts - **not** for the stopping rule, which compares centroid movement

6. **`fit(X)`** - Train the model
   - Main algorithm loop
   - Alternates between assignment and update
   - Stops when converged or max_iter reached
   - Runs the whole procedure `n_init` times and keeps the lowest-inertia result
   - Ends with a **final assignment pass** so that `labels_`, `inertia_` and `centroids` always describe the same state

7. **`predict(X)`** - Assign new points to clusters
   - Finds nearest centroid for each point
   - Useful for new data after training
   - Returns cluster labels

8. **`fit_predict(X)`** - Train and predict in one step
   - Convenience method
   - Equivalent to fit(X) then predict(X) - genuinely equivalent, because fit() ends with a final assignment pass
   - Returns cluster labels

9. **`transform(X)`** - Get distances to centroids
   - Returns distances to all k centroids
   - Useful for soft clustering
   - Shape: (n_samples, n_clusters)

10. **`get_cluster_centers()`** - Get final centroids
    - Returns the k centroid positions
    - Useful for interpretation and visualization
    - Raises a clear `ValueError` if the model has not been fitted yet

11. **`fit_transform(X)`** - Train, then return the cluster-distance matrix
    - Equivalent to fit(X) followed by transform(X)
    - Shape: (n_samples, n_clusters)

12. **`score(X)`** - Evaluate a fitted model on any data
    - Returns **negative** inertia, so that higher is better (matching scikit-learn's convention)
    - Unlike accuracy or R², it is unbounded and scale-dependent: compare it only across models fitted on the same data
    - Verified against scikit-learn: on standardized iris, ours reports `-140.0328` where sklearn reports `-139.8205`

### Fitted Attributes

| Attribute | Shape | Meaning |
|-----------|-------|---------|
| `centroids` | (n_clusters, n_features) | Final cluster centers (`None` before `fit`) |
| `labels_` | (n_samples,) | Cluster ID assigned to each training point |
| `inertia_` | float | Final objective value J (within-cluster sum of squares) |
| `n_iter_` | int | Iterations used by the winning run |

---

## Step-by-Step Example

Let's walk through a complete example clustering **customers** based on age and spending:

### The Data

```python
import numpy as np

# Customer data: [age, spending_score (1-100)]
X = np.array([
    # Young customers, low spending
    [25, 30], [28, 35], [23, 28], [26, 32],
    
    # Middle-aged customers, high spending
    [45, 80], [48, 85], [42, 78], [47, 82],
    
    # Senior customers, medium spending
    [65, 50], [62, 55], [68, 52], [63, 48]
])
```

### Training the Model

```python
from _10_kmeans_clustering import KMeansClustering

# Create model with 3 clusters
model = KMeansClustering(n_clusters=3, random_state=42)
labels = model.fit_predict(X)
```

### What Happens Internally

Everything below is the **actual** trace for `random_state=42`, not an idealised one. It is worth following closely, because the seed makes an unlucky choice and the algorithm recovers anyway.

**Initialization** (`init='random'` picks 3 of the 12 customers - here indices 10, 9 and 0):
```
Centroid 0: [68, 52]   <- senior group
Centroid 1: [62, 55]   <- ALSO the senior group!
Centroid 2: [25, 30]   <- young group

Note the problem: two of the three seeds landed in the same group, and
nobody seeded the middle-aged group. This is exactly the weakness that
init='kmeans++' and n_init>1 exist to fix.
```

**Iteration 1 - Assignment** (each customer goes to its nearest centroid):
```
Customer [25, 30] -> Cluster 2   (distances: 48.30, 44.65, 0.00)
Customer [28, 35] -> Cluster 2   (distances: 43.46, 39.45, 5.83)
Customer [23, 28] -> Cluster 2   (distances: 51.00, 47.43, 2.83)
...
Customer [45, 80] -> Cluster 1   (closest of a bad set - centroid 1 is at [62,55])
Customer [62, 55] -> Cluster 1   (distance = 0)
Customer [65, 50] -> Cluster 0   (distance = 3.61)

Resulting labels: [2 2 2 2 1 1 1 1 0 1 0 0]
The four middle-aged customers get lumped in with [62, 55].
```

**Iteration 1 - Update** (each centroid moves to the mean of its members):
```
Cluster 0 points: [65,50], [68,52], [63,48]
New Centroid 0 = mean = [65.33, 50.00]

Cluster 1 points: [45,80], [48,85], [42,78], [47,82], [62,55]
New Centroid 1 = mean = [48.80, 76.00]      <- dragged up toward the spenders

Cluster 2 points: [25,30], [28,35], [23,28], [26,32]
New Centroid 2 = mean = [25.50, 31.25]

Centroid shift = 25.0632, far above tol -> keep going
```

**Iteration 2**: centroid 1 has now moved into the middle-aged group, so `[62, 55]` deserts it and returns to cluster 0. The labels settle into `[2 2 2 2 1 1 1 1 0 0 0 0]` and the centroids become `[64.5, 51.25]`, `[45.5, 81.25]`, `[25.5, 31.25]`. Shift = 6.3804.

**Iteration 3**: nothing moves. Shift = 0.0000 < tol, so the loop stops.

**Final Result**:
```python
print("Cluster assignments:", labels)
# Output: [2 2 2 2 1 1 1 1 0 0 0 0]
#
# The grouping is perfect, but note the IDs: the young customers got ID 2,
# not ID 0. CLUSTER IDs ARE ARBITRARY - they are just the slot each centroid
# happened to occupy at initialization. Never write code that assumes
# "cluster 0 = the young segment"; look at the centroid instead.

print("\nCluster centers:")
print(model.get_cluster_centers())
# Output:
# [[64.5  51.25]   <- Senior, medium spending    (cluster 0)
#  [45.5  81.25]   <- Middle-aged, high spending (cluster 1)
#  [25.5  31.25]]  <- Young, low spending        (cluster 2)

print(f"\nInertia: {model.inertia_:.2f}")
# Output: Inertia: 135.25
#
# Check it by hand from the three group means:
#   young  39.75  +  middle-aged  47.75  +  senior  47.75  =  135.25
# scikit-learn's KMeans reports 135.250 on the same 12 points.

print(f"Converged in: {model.n_iter_} iterations")
# Output: Converged in: 3 iterations
```

### Using the Model for Predictions

```python
# New customers to classify
X_new = np.array([
    [27, 33],   # Young, low spending        → the young cluster
    [46, 81],   # Middle-aged, high spending → the high-spender cluster
    [64, 51]    # Senior, medium spending    → the senior cluster
])

predictions = model.predict(X_new)
print("New customer clusters:", predictions)
# Output: [2 1 0]
#
# Read this against the centroids printed above: cluster 2 IS the young
# segment, cluster 1 the high spenders, cluster 0 the seniors. Each new
# customer landed in the right segment; only the numbering looks surprising.
```

---

## Real-World Applications

### 1. **Customer Segmentation**
Group customers with similar behavior:
- Input: Purchase history, demographics, spending patterns
- Output: Customer segments (e.g., "budget shoppers", "premium buyers")
- Example: "Target marketing campaigns to each segment"

### 2. **Image Compression**
Reduce colors in an image:
- Input: Pixel RGB values
- Output: k dominant colors
- Example: "Reduce 16 million colors to 16 representative colors"
- ⚠️ With *this* teaching implementation, fit the palette on a random **subsample** of a few thousand pixels and then call `predict()` on the full image. The assignment loop here is a readable Python `for` over samples, so fitting a full 256×256 image (65,536 pixels, k=8, capped at 20 iterations) takes about 10 seconds, while a 2,000-pixel subsample finds the same palette in 0.4s:
  ```python
  sub = pixels[np.random.RandomState(0).choice(len(pixels), 2000, replace=False)]
  model = KMeansClustering(n_clusters=8, init='kmeans++', n_init=10, random_state=0)
  model.fit(sub)                       # fast: learns the palette
  quantized = model.get_cluster_centers()[model.predict(pixels)]   # apply to all
  ```

### 3. **Document Clustering**
Group similar documents:
- Input: Document features (word frequencies, topics)
- Output: Document clusters (news articles, research papers)
- Example: "Automatically organize news by topic"

### 4. **Anomaly Detection**
Find unusual data points:
- Input: Normal behavior patterns
- Output: Points far from any cluster = anomalies
- Example: "Detect fraudulent transactions"

### 5. **Market Segmentation**
Identify market niches:
- Input: Product features, pricing, customer preferences
- Output: Market segments
- Example: "Find underserved market opportunities"

### 6. **Image Segmentation**
Partition images into regions:
- Input: Pixel positions and colors
- Output: Distinct image regions
- Example: "Separate foreground from background"

### 7. **Recommendation Systems**
Group similar users or items:
- Input: User preferences, item features
- Output: User/item clusters
- Example: "Recommend items popular in your cluster"

### 8. **Gene Sequence Analysis**
Cluster genes with similar functions:
- Input: Gene expression patterns
- Output: Gene clusters
- Example: "Identify genes with related biological roles"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Random Initialization

```python
def _initialize_centroids(self, X):
    # rng is a PRIVATE np.random.RandomState seeded from random_state.
    # Using np.random.seed() here would silently hijack the caller's
    # global random stream - a real bug, not a style preference.
    if self.init == 'random':
        # Randomly select k data points
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[indices]
```

**How it works**:
```python
Data: [[1,1], [2,2], [3,3], [8,8], [9,9]]
k = 2

Random indices: [1, 4]
Initial centroids: [[2,2], [9,9]]
```

**Why this approach?**
- Simple and fast
- Works well for well-separated clusters
- Can lead to different results on different runs

### 2. k-Means++ Initialization

```python
elif self.init == 'kmeans++':
    # Choose first centroid randomly
    centroids = [random_point]
    
    # Choose remaining centroids far from existing ones
    for _ in range(1, k):
        distances = [min distance to any existing centroid]
        probabilities = distances² / sum(distances²)
        next_centroid = choose with probability ∝ distance²
```

**Why k-means++?**
- Better initial positions
- Faster convergence
- More consistent results
- Recommended for most cases!

**It also comes with a theoretical guarantee.** Arthur & Vassilvitskii (2007) proved that the *expected* objective value of a k-means++ seeding is within a factor of `8(ln k + 2)` of the global optimum, i.e. **O(log k)-competitive**, before Lloyd's iterations even begin. Plain random seeding has no such bound at all - it can be arbitrarily bad. This is why the recommendation is not just folklore.

The implementation samples each new centroid with probability proportional to `D(x)²`, where `D(x)` is the distance from `x` to the nearest already-chosen centroid. Squaring is what makes far-away points dramatically more likely to be picked than merely-distant ones, while still leaving a small chance of skipping an outlier.

One edge case the code has to handle: if every remaining point already sits exactly on a chosen centroid (duplicated rows, say), then `D(x)² = 0` everywhere and `distances / distances.sum()` is `0/0`. The implementation checks the sum and falls back to a uniform pick.

**Example**:
```python
Data: [1], [2], [3], [8], [9]
k = 2

Step 1: Random first centroid = [2]

Step 2: Calculate distances²
    [1]: (1-2)² = 1
    [2]: (2-2)² = 0
    [3]: (3-2)² = 1
    [8]: (8-2)² = 36
    [9]: (9-2)² = 49

Step 3: Choose second centroid with probability ∝ distance²
    Likely to choose [8] or [9] (far from [2])
```

### 3. Assignment Step

```python
def _assign_clusters(self, X):
    # This is the formula  label(x) = argmin_k ||x - c_k||  line for line
    for i, x in enumerate(X):
        # Calculate distance to each centroid
        distances = np.linalg.norm(x - self.centroids, axis=1)
        
        # Assign to nearest
        labels[i] = np.argmin(distances)
```

**Step-by-step**:
```python
Point: [5, 5]
Centroids: [[2,2], [8,8], [5,1]]

distances = [
    √[(5-2)² + (5-2)²] = 4.24,   ← Centroid 0
    √[(5-8)² + (5-8)²] = 4.24,   ← Centroid 1
    √[(5-5)² + (5-1)²] = 4.00    ← Centroid 2 (minimum!)
]

Assign to cluster 2
```

### 4. Update Step

```python
def _update_centroids(self, X, labels):
    for k in range(self.n_clusters):
        # Get all points in this cluster
        cluster_points = X[labels == k]
        
        if len(cluster_points) > 0:
            # New centroid = mean of points
            # (this line IS  c_k = (1/n_k) * sum of x in C_k)
            new_centroids[k] = np.mean(cluster_points, axis=0)
        else:
            # Empty cluster - see below
            new_centroids[k] = X[farthest]
```

**Example**:
```python
Cluster 0 points: [[1,1], [2,2], [3,3]]

Mean calculation:
    x-axis: (1 + 2 + 3) / 3 = 2
    y-axis: (1 + 2 + 3) / 3 = 2
    
New centroid: [2, 2]
```

**The empty-cluster branch**: a cluster can lose every one of its points, and `np.mean([])` is `nan`. The naive fix - "just keep the old centroid" - is worse than it looks: that centroid is now stranded where no data lives, it will never attract anything again, and your k=5 model quietly becomes a k=4 model. That is not hypothetical; before this was fixed, the elbow sweep in USAGE EXAMPLE 3 printed the *identical* inertia 1755.06 for k=4 and k=5, because the k=5 fit had a dead cluster.

The implementation does what scikit-learn does instead: **relocate the dead centroid onto the sample that is currently worst served** - the point with the largest distance to its own centroid. That point is where a new cluster is most needed, and moving there strictly lowers J. If two clusters die at once, the second takes the next-worst point rather than duplicating the first. With this rule the k=5 sweep drops to 185.22 with five genuinely populated clusters.

(One honest caveat: if the data has fewer than k *distinct* points - e.g. 20 copies of `[0,0]` and 20 of `[10,10]` with k=4 - relocation lands on a duplicate of an existing centroid and the cluster stays empty. No algorithm can do better; there is nothing there to cluster.)

### 5. Convergence Check

```python
# Calculate how much centroids moved
centroid_shift = np.linalg.norm(new_centroids - self.centroids)

# Stop if movement is tiny
if centroid_shift < self.tol:
    n_iter = iteration + 1
    break
```

**Example**:
```python
Old centroids: [[2.0, 2.0], [8.0, 8.0]]
New centroids: [[2.1, 2.0], [8.0, 8.1]]

Shift = √[(2.1-2.0)² + (2.0-2.0)² + (8.0-8.0)² + (8.1-8.0)²]
      = √[0.01 + 0 + 0 + 0.01]
      = 0.141

If tolerance = 0.0001 → Keep iterating
If tolerance = 0.2 → Stop! (converged)
```

### 6. Inertia Calculation

```python
def _calculate_inertia(self, X, labels):
    inertia = 0
    for i, x in enumerate(X):
        centroid = self.centroids[labels[i]]
        inertia += np.linalg.norm(x - centroid) ** 2
```

**Example**:
```python
Points: [[1,1], [2,2], [8,8], [9,9]]
Labels: [0, 0, 1, 1]
Centroids: [[1.5, 1.5], [8.5, 8.5]]

For [1,1] in cluster 0:
    distance² = (1-1.5)² + (1-1.5)² = 0.5

For [2,2] in cluster 0:
    distance² = (2-1.5)² + (2-1.5)² = 0.5

For [8,8] in cluster 1:
    distance² = (8-8.5)² + (8-8.5)² = 0.5

For [9,9] in cluster 1:
    distance² = (9-8.5)² + (9-8.5)² = 0.5

Inertia = 0.5 + 0.5 + 0.5 + 0.5 = 2.0
```

### 7. The Final Assignment Pass

Look carefully at the order of operations inside one Lloyd iteration:

```python
labels        = self._assign_clusters(X)          # uses the CURRENT centroids
new_centroids = self._update_centroids(X, labels)
self.centroids = new_centroids                    # centroids have now MOVED
```

When the loop exits, `labels` was computed against the *previous* centroids, but `self.centroids` holds the *new* ones. Storing that `labels` would leave `labels_`, `inertia_` and `centroids` describing three slightly different states. Measured on 180 Gaussian points with `max_iter=1`: `labels_` disagreed with `predict(X)` on 10 of the 180 points, and `inertia_` reported 656.45 where the value consistent with the stored centroids is 439.93 - a 49% overstatement of the model's own error.

So `fit()` ends with one more assignment, exactly as scikit-learn does:

```python
# Final assignment against the FINAL centroids
labels = self._assign_clusters(X)
inertia = self._calculate_inertia(X, labels)
```

This is invisible on a converged run (nothing moved, so nothing changes) which is precisely why the bug could hide. It also makes `fit_predict(X)` genuinely equal to `fit(X)` then `predict(X)`, and it means `fit()` no longer crashes when `max_iter=0`.

### 8. Multiple Restarts (`n_init`)

k-Means converges to a *local* optimum, and which one depends entirely on the initialization. The cheapest, most effective remedy is to run the whole thing several times and keep the best:

```python
for _ in range(self.n_init):
    self.centroids = self._initialize_centroids(X)   # a fresh start each time
    ...Lloyd's loop...
    labels  = self._assign_clusters(X)
    inertia = self._calculate_inertia(X, labels)
    if best_inertia is None or inertia < best_inertia:
        best_inertia, best_centroids, best_labels = inertia, self.centroids, labels
```

Inertia is the objective J itself, so "keep the lowest inertia" is literally "keep the best answer to the problem we posed". This is the single largest fidelity lever in the whole class. On `make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)` with k=4:

| Configuration | Inertia | vs. scikit-learn |
|---|---|---|
| `init='random'`, `n_init=1` (the old default) | 1755.06 | 8.6x worse |
| `init='kmeans++'`, `n_init=1` | 203.89 | matches |
| `init='kmeans++'`, `n_init=10` | **203.89** | **exact match, ARI = 1.000** |
| `sklearn.cluster.KMeans(n_init=10)` | 203.89 | - |

The default here is `n_init=1` so that the documented example outputs stay reproducible; scikit-learn defaults to 10. **For any real use, set `n_init=10`.**

---

## Model Evaluation

### 1. Inertia (Within-Cluster Sum of Squares)

```
Inertia = Σ Σ distance²(point, centroid)
          k points∈k
```

**Interpretation**:
- Lower inertia = tighter clusters = better fit
- BUT: More clusters always give lower inertia!
- Need to balance k with inertia

**Example**:
```python
model.fit(X)
print(f"Inertia: {model.inertia_:.2f}")
# Lower is better, but watch for overfitting!
```

### 2. Elbow Method

Find optimal k by plotting inertia vs k:

```
Inertia
   |
   |\
   | \
   |  \___
   |      \___  ← "Elbow" at k=3
   |          \___
   +---------------> k
     1 2 3 4 5 6
```

**How to use**:
```python
inertias = []
for k in range(2, 11):
    # init='kmeans++' + n_init=10 matters here: a single unlucky 'random'
    # start at one value of k can flatten the elbow out of existence
    model = KMeansClustering(n_clusters=k, init='kmeans++', n_init=10,
                             random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)

# Plot inertias and look for the "elbow"
# The elbow is where inertia starts decreasing more slowly
```

**Interpretation**:
- Before elbow: Each new cluster helps a lot
- At elbow: Diminishing returns begin
- After elbow: New clusters don't help much

### 3. Silhouette Score

Measures how well points fit in their clusters:

```
Silhouette = (b - a) / max(a, b)

where:
    a = average distance to points in same cluster
    b = average distance to points in nearest other cluster
```

**Range**: -1 to +1
- +1: Perfect clustering (far from other clusters)
- 0: On cluster boundary
- -1: Wrong cluster (closer to other cluster)

**Example**:
```python
from sklearn.metrics import silhouette_score

labels = model.fit_predict(X)
score = silhouette_score(X, labels)

if score > 0.7:
    print("Excellent clustering!")
elif score > 0.5:
    print("Good clustering")
elif score > 0.25:
    print("Weak clustering")
else:
    print("Poor clustering - points don't fit clusters well")
```

### 4. Davies-Bouldin Index

Measures average similarity between clusters:

```
DB = (1/k) × Σ max((σᵢ + σⱼ) / d(cᵢ, cⱼ))
              i  j≠i
```

**Lower is better**: Want low within-cluster variance, high between-cluster distance

### 5. Calinski-Harabasz Index (Variance Ratio Criterion)

Ratio of between-cluster to within-cluster dispersion:

```
CH = (trace(Bₖ) / trace(Wₖ)) × ((n - k) / (k - 1))
```

**Higher is better**: Want high between-cluster variance, low within-cluster variance

---

## Choosing the Right k

### Methods to Find Optimal k

#### 1. **Elbow Method**
```
Plot inertia vs k, look for "elbow"

Pros: Simple, visual
Cons: Elbow not always clear
```

#### 2. **Silhouette Analysis**
```
Choose k with highest average silhouette score

Pros: Considers cluster quality
Cons: Computationally expensive
```

#### 3. **Domain Knowledge**
```
Use business understanding

Example: "We need 3 customer tiers: Budget, Standard, Premium"

Pros: Makes business sense
Cons: May not match data structure
```

#### 4. **Gap Statistic**
```
Compare inertia to random data

Optimal k = where real data is most different from random

Pros: Principled statistical approach
Cons: Complex to implement
```

### Example: Finding Optimal k

```python
import numpy as np
from sklearn.metrics import silhouette_score

# Try different k values
k_range = range(2, 11)
inertias = []
silhouettes = []

for k in k_range:
    model = KMeansClustering(n_clusters=k, init='kmeans++', n_init=10,
                             random_state=42)
    labels = model.fit_predict(X)
    
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# Method 1: Elbow in inertia plot
# Look for k where inertia decrease slows down

# Method 2: Maximum silhouette score
best_k = k_range[np.argmax(silhouettes)]
print(f"Optimal k by silhouette: {best_k}")
```

---

## Feature Scaling: Important for k-Means

### Why Scaling Matters

k-Means uses Euclidean distance, so feature scales matter!

**Example without scaling**:
```python
Feature 1: Age (20-80)          → Range = 60
Feature 2: Income ($20k-$200k)  → Range = 180,000

Distance dominated by income!
Age difference of 20 years ≈ Income difference of $20
```

**Example with scaling**:
```python
Feature 1: Age (scaled to 0-1)     → Range = 1
Feature 2: Income (scaled to 0-1)  → Range = 1

Both features contribute equally!
```

### Standardization (Recommended)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now: mean=0, std=1 for each feature
model = KMeansClustering(n_clusters=3)
model.fit(X_scaled)
```

**Formula**:
```
x_scaled = (x - mean) / std_dev
```

---

## Advantages and Limitations

### Advantages ✅

1. **Simple and Fast**
   - Easy to understand and implement
   - Fast on moderate-sized datasets
   - Scales well with number of features

2. **Guaranteed Convergence**
   - Always converges (though maybe to local optimum)
   - Usually converges in few iterations
   - This is provable, not just observed: both steps weakly decrease J, J is bounded below by 0, and there are finitely many partitions - see [Why convergence is guaranteed](#the-mathematical-foundation)

3. **Works Well for Spherical Clusters**
   - Great when clusters are round and well-separated
   - Clear cluster boundaries

4. **Easy to Interpret**
   - Cluster centers have clear meaning
   - Can profile each cluster

5. **Versatile**
   - Works on many types of data
   - Can be adapted for various domains

### Limitations ❌

1. **Must Specify k in Advance**
   - Need to know how many clusters
   - Wrong k = poor results
   - Requires trial and error

2. **Sensitive to Initialization**
   - Different starting points = different results
   - Can get stuck in local optima
   - Solution: `KMeansClustering(..., init='kmeans++', n_init=10)` - smart seeding plus multiple restarts, keeping the lowest-inertia run. This is measurable, not decorative: on 4-blob data it moves inertia from 1755.06 to 203.89

3. **Assumes Spherical Clusters**
   - Struggles with elongated or irregular shapes
   - Assumes clusters have similar sizes
   - Not good for nested clusters

4. **Sensitive to Outliers**
   - Outliers pull centroids away from true centers
   - Can distort cluster shapes
   - Solution: Remove outliers first

5. **Requires Feature Scaling**
   - Large-scale features dominate distance
   - Must scale features appropriately
   - Solution: Standardize before clustering

6. **Hard Clustering Only**
   - Each point belongs to exactly one cluster
   - No "soft" or probabilistic assignments
   - Solution: Use Gaussian Mixture Models for soft clustering

### When to Use k-Means

**Good Use Cases**:
- ✅ Roughly spherical clusters
- ✅ Clusters of similar sizes
- ✅ Clear separation between groups
- ✅ Know approximate number of clusters
- ✅ Need fast, simple clustering

**Bad Use Cases**:
- ❌ Unknown number of clusters
- ❌ Irregular cluster shapes (crescents, nested circles)
- ❌ Very different cluster sizes
- ❌ Many outliers in data
- ❌ Need hierarchical relationships

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate synthetic customer data
X, true_labels = make_blobs(
    n_samples=300, 
    centers=4, 
    n_features=2,
    cluster_std=0.6, 
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using elbow method
print("Finding optimal k...\n")
print(f"{'k':<5} {'Inertia':<15} {'Silhouette':<15}")
print("-" * 35)

for k in range(2, 8):
    model = KMeansClustering(n_clusters=k, init='kmeans++', n_init=10,
                             random_state=42)
    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    
    print(f"{k:<5} {model.inertia_:<15.2f} {silhouette:<15.3f}")

# Train final model with optimal k=4
print("\n" + "="*50)
print("Training final model with k=4...")
print("="*50 + "\n")

model = KMeansClustering(n_clusters=4, init='kmeans++', n_init=10,
                         random_state=42)
labels = model.fit_predict(X_scaled)

# Evaluate
silhouette = silhouette_score(X_scaled, labels)

print(f"Final Results:")
print(f"  Inertia: {model.inertia_:.2f}")
print(f"  Silhouette Score: {silhouette:.3f}")
print(f"  Converged in: {model.n_iter_} iterations")

# Analyze clusters
print("\nCluster Analysis:")
for cluster in range(4):
    cluster_data = X[labels == cluster]
    print(f"\nCluster {cluster}:")
    print(f"  Size: {len(cluster_data)} points")
    # inverse_transform needs a 2-D array, so reshape the single centroid row
    # into (1, n_features) and take row 0 back out
    center_original = scaler.inverse_transform(model.centroids[cluster].reshape(1, -1))[0]
    print(f"  Center (original scale): {center_original}")
    print(f"  Mean feature 1: {np.mean(cluster_data[:, 0]):.2f}")
    print(f"  Mean feature 2: {np.mean(cluster_data[:, 1]):.2f}")

# Predict new data
X_new = np.array([[1.5, 2.0], [-1.0, -1.0]])
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)

print(f"\nNew data predictions:")
for i, pred in enumerate(predictions):
    print(f"  Point {i+1}: Cluster {pred}")
```

---

## Simplifications vs. Canonical k-Means

This implementation is deliberately written for reading, not for racing. The core mathematics is faithful - verified against `sklearn.cluster.KMeans` and against hand computation - but four things are simpler here than in a production library. Each is listed with what canonical k-Means does, why this file does something else, and what it costs you in practice.

### 1. `tol` is an absolute shift, not a variance-scaled squared shift

- **Canonical (scikit-learn)**: declares convergence when the *squared* centroid shift falls below `tol × mean(var(X, axis=0))`, so the tolerance automatically adapts to the scale of your data.
- **Here**: `np.linalg.norm(new_centroids - self.centroids) < tol` - the plain Frobenius norm of the stacked centroid matrix, in raw feature units.
- **Why**: one line, one formula, and it reads exactly like the "centroids stopped moving" sentence in the theory section.
- **Consequence**: with the same numeric `tol=1e-4` this test is stricter, so you may pay a few extra iterations. Measured on standardized iris with *identical* initial centroids, both implementations took 7 iterations and reached the same inertia 140.0328 - the stopping rule made no observable difference at all. If your features are on wildly different scales, scale them (which you should do for k-Means anyway) and the two remain comparable.

### 2. k-means++ is the plain version, not the greedy variant

- **Canonical (scikit-learn)**: for each new centroid it draws `2 + log(k)` candidate points from the D(x)² distribution, evaluates the resulting inertia for each, and keeps the best candidate ("greedy k-means++").
- **Here**: draws exactly one candidate per centroid, which is the sampling rule as stated in the original Arthur & Vassilvitskii paper.
- **Why**: the greedy trial loop roughly triples the length of `_initialize_centroids` and obscures the one idea that matters - `P(x) ∝ D(x)²`.
- **Consequence**: a single k-means++ start here is slightly more variable than scikit-learn's. On standardized iris, `init='kmeans++', n_init=10` reaches inertia 140.033 where scikit-learn reaches 139.821 - a 0.15% gap. **`n_init` is the practical remedy** and it closes the gap entirely on well-separated data (203.8907 vs 203.8907, ARI 1.000 on 4-blob data).

### 3. The distance loops are Python `for` loops, not batched linear algebra

- **Canonical**: computes the full (n_samples × n_clusters) distance matrix with BLAS via the `||x||² - 2x·c + ||c||²` expansion, in chunked, multi-threaded Cython.
- **Here**: `_assign_clusters`, `_calculate_inertia` and `transform` each loop over samples in Python.
- **Why**: this repository's stated rule is clarity over performance, and `for i, x in enumerate(X)` is the algorithm written out loud.
- **Consequence**: measured on this machine, 500×2 points with k=5 fits in 0.06s and 10,000×5 with k=10 in about 5s - fine for learning, but hundreds of times slower than the compiled version, and the gap grows with `n × k × iterations`. A full 256×256 image (65,536 pixels, k=8, capped at 20 iterations) takes 10.2s, where fitting a 2,000-pixel random subsample finds the same palette in 0.41s. Size your data accordingly - and for image quantization, fit on a subsample and `predict()` the rest (see [Real-World Applications](#real-world-applications)).

### 4. Not implemented at all

- **Mini-batch k-Means** (Sculley 2010), which updates centroids from small random batches for datasets that do not fit in memory. Use `sklearn.cluster.MiniBatchKMeans`.
- **Elkan's / Hamerly's triangle-inequality acceleration**, which skips distance computations that cannot possibly change an assignment. It changes nothing about the answer, only the speed, and it would completely obscure the assignment step.
- **`sample_weight`**, which lets each point count more than once in the mean.

None of these change the objective J or the answer k-Means converges to; they are engineering, and they are exactly what you get from scikit-learn in production.

---

## Key Concepts to Remember

### 1. **k-Means is Unsupervised**
No labels needed! Discovers patterns automatically.

### 2. **Must Choose k**
The number of clusters must be specified before training.

### 3. **Initialization Matters**
Use k-means++ for better, more consistent results.

### 4. **Feature Scaling is Critical**
Always standardize features before clustering.

### 5. **Evaluates by Inertia**
Lower inertia = tighter clusters, but watch for overfitting.

### 6. **Assumptions**
Works best with:
- Spherical clusters
- Similar cluster sizes
- Well-separated groups

### 7. **Local Optima**
May converge to a suboptimal solution. Run multiple times with different initializations - that is what the `n_init` parameter does (`n_init=10` is scikit-learn's default and a good habit).

### 8. **Cluster IDs Are Arbitrary**
`labels_` contains 0, 1, 2 ... but those numbers carry no meaning. They are just the slots the centroids happened to occupy at initialization, so the same clustering can come back numbered differently on a different seed. Identify a cluster by its **centroid**, never by its ID - and never compare two runs' labels element-by-element (use the Adjusted Rand Index for that).

---

## Conclusion

k-Means Clustering is a powerful tool for discovering patterns in data! By understanding:
- How the algorithm iteratively refines cluster assignments
- How centroids represent cluster centers
- How to choose k using the elbow method
- How feature scaling affects results
- When k-means works well (and when it doesn't)

You've gained a fundamental tool for unsupervised learning! 🎯

**When to Use k-Means**:
- ✅ Customer segmentation
- ✅ Image compression
- ✅ Document clustering
- ✅ Anomaly detection
- ✅ Market segmentation

**When to Use Something Else**:
- ❌ Don't know number of clusters → Use DBSCAN, hierarchical clustering
- ❌ Non-spherical clusters → Use DBSCAN, spectral clustering
- ❌ Need cluster hierarchy → Use hierarchical clustering
- ❌ Need probabilistic assignments → Use Gaussian Mixture Models

**Next Steps**:
- Try k-means on your own datasets
- Experiment with different k values
- Compare with other clustering algorithms (DBSCAN, Hierarchical)
- Learn about advanced variants (k-means++, mini-batch k-means)
- Explore dimensionality reduction before clustering (PCA)
- Study cluster validation techniques

Happy Clustering! 💻🎯

