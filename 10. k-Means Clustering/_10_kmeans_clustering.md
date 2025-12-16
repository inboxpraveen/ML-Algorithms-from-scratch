# k-Means Clustering from Scratch: A Comprehensive Guide

Welcome to the world of k-Means Clustering! 🎯 In this comprehensive guide, we'll explore one of the most popular unsupervised machine learning algorithms. Think of it as the "find natural groups in your data" algorithm!

## Table of Contents
1. [What is k-Means Clustering?](#what-is-k-means-clustering)
2. [How k-Means Works](#how-k-means-works)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)

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

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class KMeansClustering:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, 
                 init='random', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
```

### Core Methods

1. **`__init__(n_clusters, max_iter, tol, init, random_state)`** - Initialize model
   - n_clusters: Number of clusters (k)
   - max_iter: Maximum iterations
   - tol: Convergence tolerance
   - init: Initialization method ('random' or 'kmeans++')
   - random_state: Random seed for reproducibility

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
   - Returns new centroid positions

5. **`_calculate_inertia(X, labels)`** - Calculate quality metric
   - Sum of squared distances to centroids
   - Lower = better clustering
   - Used for convergence and evaluation

6. **`fit(X)`** - Train the model
   - Main algorithm loop
   - Alternates between assignment and update
   - Stops when converged or max_iter reached

7. **`predict(X)`** - Assign new points to clusters
   - Finds nearest centroid for each point
   - Useful for new data after training
   - Returns cluster labels

8. **`fit_predict(X)`** - Train and predict in one step
   - Convenience method
   - Equivalent to fit(X) then predict(X)
   - Returns cluster labels

9. **`transform(X)`** - Get distances to centroids
   - Returns distances to all k centroids
   - Useful for soft clustering
   - Shape: (n_samples, n_clusters)

10. **`get_cluster_centers()`** - Get final centroids
    - Returns the k centroid positions
    - Useful for interpretation and visualization

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
from kmeans_clustering import KMeansClustering

# Create model with 3 clusters
model = KMeansClustering(n_clusters=3, random_state=42)
labels = model.fit_predict(X)
```

### What Happens Internally

**Initialization** (random):
```
Randomly select 3 customers as initial centroids:
Centroid 1: [25, 30]
Centroid 2: [45, 80]
Centroid 3: [65, 50]
```

**Iteration 1 - Assignment**:
```
Customer [25, 30] → Cluster 1 (distance to centroid 1 = 0)
Customer [28, 35] → Cluster 1 (distance = 5.83)
Customer [23, 28] → Cluster 1 (distance = 2.83)
...
Customer [45, 80] → Cluster 2 (distance = 0)
Customer [48, 85] → Cluster 2 (distance = 5.83)
...
Customer [65, 50] → Cluster 3 (distance = 0)
Customer [62, 55] → Cluster 3 (distance = 5.83)
...
```

**Iteration 1 - Update**:
```
Cluster 1 points: [25,30], [28,35], [23,28], [26,32]
New Centroid 1 = mean = [25.5, 31.25]

Cluster 2 points: [45,80], [48,85], [42,78], [47,82]
New Centroid 2 = mean = [45.5, 81.25]

Cluster 3 points: [65,50], [62,55], [68,52], [63,48]
New Centroid 3 = mean = [64.5, 51.25]
```

**Iteration 2**:
```
Reassign points to new centroids...
Update centroids again...
Continue until centroids stop moving!
```

**Final Result**:
```python
print("Cluster assignments:", labels)
# Output: [0 0 0 0 1 1 1 1 2 2 2 2]

print("\nCluster centers:")
print(model.get_cluster_centers())
# Output:
# [[25.5  31.25]   ← Young, low spending
#  [45.5  81.25]   ← Middle-aged, high spending
#  [64.5  51.25]]  ← Senior, medium spending

print(f"\nInertia: {model.inertia_:.2f}")
# Output: Inertia: 168.00

print(f"Converged in: {model.n_iter_} iterations")
# Output: Converged in: 3 iterations
```

### Using the Model for Predictions

```python
# New customers to classify
X_new = np.array([
    [27, 33],   # Young, low spending → Should be Cluster 0
    [46, 81],   # Middle-aged, high spending → Should be Cluster 1
    [64, 51]    # Senior, medium spending → Should be Cluster 2
])

predictions = model.predict(X_new)
print("New customer clusters:", predictions)
# Output: [0 1 2]
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
    if self.init == 'random':
        # Randomly select k data points
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
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
    for i, x in enumerate(X):
        # Calculate distance to each centroid
        distances = np.linalg.norm(X[i] - self.centroids, axis=1)
        
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
        
        # New centroid = mean of points
        new_centroids[k] = np.mean(cluster_points, axis=0)
```

**Example**:
```python
Cluster 0 points: [[1,1], [2,2], [3,3]]

Mean calculation:
    x-axis: (1 + 2 + 3) / 3 = 2
    y-axis: (1 + 2 + 3) / 3 = 2
    
New centroid: [2, 2]
```

### 5. Convergence Check

```python
# Calculate how much centroids moved
centroid_shift = np.linalg.norm(new_centroids - old_centroids)

# Stop if movement is tiny
if centroid_shift < self.tol:
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
    model = KMeansClustering(n_clusters=k)
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
    model = KMeansClustering(n_clusters=k, random_state=42)
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
   - Solution: Use k-means++ or run multiple times

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
    model = KMeansClustering(n_clusters=k, init='kmeans++', random_state=42)
    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    
    print(f"{k:<5} {model.inertia_:<15.2f} {silhouette:<15.3f}")

# Train final model with optimal k=4
print("\n" + "="*50)
print("Training final model with k=4...")
print("="*50 + "\n")

model = KMeansClustering(n_clusters=4, init='kmeans++', random_state=42)
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
    print(f"  Center (original scale): {scaler.inverse_transform(model.centroids[cluster])}")
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
May converge to suboptimal solution. Run multiple times with different initializations.

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

