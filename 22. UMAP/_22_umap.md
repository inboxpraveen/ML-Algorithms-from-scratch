# UMAP - Uniform Manifold Approximation and Projection

## Overview

**UMAP (Uniform Manifold Approximation and Projection)** is a state-of-the-art dimensionality reduction technique that constructs a high-dimensional graph representation of your data and optimizes a low-dimensional graph to be as structurally similar as possible. It's based on manifold learning and topological data analysis, making it both theoretically principled and practically effective.

### Key Concept

Imagine your high-dimensional data lives on a curved surface (manifold) in high-dimensional space. UMAP:
1. Learns the shape of this manifold by building a graph
2. Projects it down to 2D or 3D while preserving the manifold structure
3. Keeps both local neighborhoods AND global relationships intact

Think of it like creating a map: you want nearby cities to be close on the map (local structure), but you also want continents in the right relative positions (global structure). UMAP does both!

## When to Use UMAP

### Perfect For:
- **Data Visualization**: Visualize high-dimensional data in 2D/3D
- **Feature Engineering**: Reduce dimensions before machine learning
- **Exploratory Analysis**: Discover clusters and patterns
- **Large Datasets**: Handles 100,000+ samples efficiently
- **Biological Data**: Single-cell genomics, protein analysis
- **Text Analysis**: Visualize word embeddings, document spaces

### Advantages Over Other Methods:
- **vs t-SNE**: 10-100x faster, preserves global structure, more general purpose
- **vs PCA**: Captures non-linear relationships, better visualization
- **vs Autoencoders**: No training needed, solid mathematical foundation

## Mathematical Foundation

### 1. The Core Idea

UMAP models data as a **fuzzy topological structure** (a weighted graph) and finds a similar structure in lower dimensions.

**High-level process:**
```
High-D Data → k-NN Graph → Fuzzy Simplicial Set → Optimization → Low-D Embedding
```

### 2. Fuzzy Simplicial Sets

Instead of saying "point A is connected to point B" (binary), UMAP says "point A has a 0.8 probability of being connected to point B" (fuzzy).

**Why fuzzy?** Real data often has ambiguous boundaries and overlapping structures.

### 3. Key Mathematical Components

#### a) Local Connectivity (ρ)

For each point, find the distance to its nearest neighbor. This defines "local" for that point.

```
ρᵢ = distance to nearest neighbor of point i
```

#### b) Smooth Approximation (σ)

Normalize distances so the sum of probabilities equals a target (related to perplexity).

```
Target = log₂(k)  where k = number of neighbors
```

Find σᵢ such that:
```
Σⱼ exp(-(dᵢⱼ - ρᵢ)/σᵢ) = log₂(k)
```

#### c) Membership Strength

Probability that points i and j are connected in the manifold:

```
v(dᵢⱼ) = exp(-(max(0, dᵢⱼ - ρᵢ))/σᵢ)
```

#### d) Fuzzy Union

Combine directional probabilities:

```
w(i,j) = v(dᵢⱼ) + v(dⱼᵢ) - v(dᵢⱼ) × v(dⱼᵢ)
```

This is the **fuzzy set union** formula.

### 4. Low-Dimensional Optimization

In the embedding space, use a simple probability function:

```
P(d) = 1 / (1 + a × d^(2b))
```

Where:
- `d` = distance in low-dimensional space
- `a, b` = parameters controlling curve shape (based on min_dist)

**Objective:** Minimize cross-entropy between high-D and low-D graphs:

```
CE = Σᵢⱼ wᵢⱼ log(wᵢⱼ / qᵢⱼ) + (1 - wᵢⱼ) log((1 - wᵢⱼ) / (1 - qᵢⱼ))
```

Where:
- `wᵢⱼ` = high-D edge weight
- `qᵢⱼ` = low-D edge weight (computed from embedded distances)

## Algorithm Steps

### Step 1: Construct k-NN Graph

For each point, find its k nearest neighbors in the high-dimensional space.

```python
# For each point i
for i in range(n_samples):
    # Find k nearest neighbors
    neighbors_i = k_nearest_neighbors(X[i], k)
    distances_i = distances_to_neighbors(X[i], neighbors_i)
```

### Step 2: Compute Local Metrics (ρ and σ)

Adapt the metric to local density variations.

```python
# For each point i
for i in range(n_samples):
    # ρ = distance to nearest neighbor
    rho[i] = distances_i[0]
    
    # σ = found via binary search
    # Such that: Σⱼ exp(-(dᵢⱼ - ρᵢ)/σᵢ) ≈ log₂(k)
    sigma[i] = binary_search_for_sigma(distances_i, rho[i], target=log2(k))
```

### Step 3: Build Fuzzy Simplicial Set

Compute edge weights for the high-dimensional graph.

```python
for i in range(n_samples):
    for j in neighbors_of_i:
        # Compute membership strength
        v_ij = exp(-(max(0, d_ij - rho[i])) / sigma[i])
        v_ji = exp(-(max(0, d_ji - rho[j])) / sigma[j])
        
        # Fuzzy set union
        w_ij = v_ij + v_ji - v_ij * v_ji
        
        graph[i, j] = w_ij
```

### Step 4: Initialize Embedding

Use spectral embedding or random initialization.

```python
if init == 'spectral':
    # Use graph Laplacian eigenvectors
    embedding = spectral_layout(graph)
else:
    # Random initialization
    embedding = random_normal(n_samples, n_components)
```

### Step 5: Optimize via SGD

Iteratively improve the embedding using stochastic gradient descent.

```python
for epoch in range(n_epochs):
    for (i, j) in edges:
        # Compute distance in embedding
        d_ij_low = distance(embedding[i], embedding[j])
        
        # Attractive force (for connected pairs)
        grad = compute_attractive_gradient(d_ij_low, w_ij)
        
        # Update positions
        embedding[i] += learning_rate * grad
        embedding[j] -= learning_rate * grad
        
        # Negative sampling: repulsive force
        for k in random_samples(n_negative=5):
            grad_repulsive = compute_repulsive_gradient(embedding[i], embedding[k])
            embedding[i] += learning_rate * grad_repulsive
```

## Parameters Explained

### n_neighbors (default=15)

Controls the balance between local and global structure.

**Small values (5-10):**
- Focus on very local structure
- Tight, well-separated clusters
- May miss broader patterns

**Medium values (15-30):**
- Balanced view (recommended default)
- Good for most use cases

**Large values (50-100):**
- Emphasize global structure
- Better capture of overall data topology
- May blur fine details

**Rule of thumb:** Start with 15, increase if you need more global context.

### min_dist (default=0.1)

Minimum distance between points in the embedding.

**Small values (0.0-0.1):**
- Tightly packed clusters
- Good for cluster analysis
- Points can be very close

**Medium values (0.1-0.3):**
- Balanced spacing (recommended)
- Good general-purpose choice

**Large values (0.3-0.99):**
- More evenly distributed points
- Points repel each other more
- Better for understanding relationships

**Tip:** Use 0.0 for clustering tasks, 0.1-0.3 for general visualization.

### n_components (default=2)

Dimensionality of the embedding space.

- **2D**: Best for visualization, plots
- **3D**: Interactive 3D visualization
- **Higher (5-50)**: Dimensionality reduction for ML pipelines

### metric (default='euclidean')

Distance metric for comparing points.

- **euclidean**: Standard choice, works for most data
- **manhattan**: Equal weight to all dimensions
- **cosine**: Text data, normalized vectors, embeddings

### learning_rate (default=1.0)

Step size for optimization.

- **Low (0.1-0.5)**: Slower, more stable
- **Medium (0.5-2.0)**: Good balance (recommended)
- **High (2.0-10.0)**: Faster, may be unstable

### n_epochs (default=200)

Number of optimization iterations.

- **Minimum**: 100 (very fast but may not converge)
- **Recommended**: 200-500 (good quality)
- **High quality**: 500-1000 (best results, slower)

## Code Example

```python
import numpy as np
from _22_umap import UMAP

# Generate high-dimensional data
np.random.seed(42)
X = np.random.randn(500, 50)  # 500 samples, 50 features

# Create three clusters
X[:200] += [5, 0, 0, 0, 0] + [0]*45
X[200:400] += [0, 5, 0, 0, 0] + [0]*45
X[400:] += [0, 0, 5, 0, 0] + [0]*45

# Apply UMAP
umap = UMAP(
    n_components=2,      # Reduce to 2D
    n_neighbors=15,      # Balance local/global
    min_dist=0.1,        # Moderate spacing
    n_epochs=200,        # Good convergence
    random_state=42,     # Reproducibility
    verbose=1            # Show progress
)

# Fit and transform
X_embedded = umap.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Embedded shape: {X_embedded.shape}")

# Now you can visualize X_embedded in 2D!
```

## Practical Use Cases

### 1. Visualizing High-Dimensional Data

```python
# Example: Visualizing MNIST digits (28x28 = 784 dimensions)
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data  # (1797, 64)
y = digits.target

# Apply UMAP
umap = UMAP(n_components=2, n_neighbors=15, random_state=42)
X_embedded = umap.fit_transform(X)

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='Spectral', s=5)
plt.colorbar()
plt.title('UMAP projection of Digits dataset')
plt.show()
```

### 2. Feature Engineering for ML

```python
# Reduce dimensions before classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Original high-dimensional data
X_train = np.random.randn(1000, 100)
y_train = np.random.randint(0, 2, 1000)

# Reduce dimensions
umap = UMAP(n_components=10, n_neighbors=15, random_state=42)
X_reduced = umap.fit_transform(X_train)

# Train classifier on reduced data
clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X_reduced, y_train, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 3. Exploring Different Parameter Settings

```python
# Compare different n_neighbors
for n_neighbors in [5, 15, 50]:
    umap = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=42
    )
    X_embedded = umap.fit_transform(X)
    
    # Plot each result
    plt.figure()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=2)
    plt.title(f'UMAP with n_neighbors={n_neighbors}')
    plt.show()
```

## UMAP vs t-SNE: Detailed Comparison

| Aspect | UMAP | t-SNE |
|--------|------|-------|
| **Speed** | Fast (10-100x faster) | Slow |
| **Global Structure** | ✓ Preserves well | ✗ Often lost |
| **Local Structure** | ✓ Excellent | ✓ Excellent |
| **Scalability** | 100,000+ samples | ~10,000 samples |
| **General Purpose** | ✓ Yes (can use for ML) | ✗ Visualization only |
| **Deterministic** | More stable | More random |
| **Parameters** | Intuitive | Less intuitive (perplexity) |
| **Theory** | Topological | Probabilistic |

**When to use each:**
- **UMAP**: Default choice for most tasks, especially if >1000 samples
- **t-SNE**: When you specifically need very local structure emphasis

## Common Issues and Solutions

### Issue 1: Clusters Overlap Too Much

**Problem:** Points from different clusters blend together

**Solutions:**
- Decrease `n_neighbors` (e.g., 15 → 5) to focus on local structure
- Decrease `min_dist` (e.g., 0.1 → 0.0) for tighter clusters
- Increase `n_epochs` for better convergence

### Issue 2: Points Too Spread Out

**Problem:** Embedding is too uniform, no clear structure

**Solutions:**
- Increase `n_neighbors` (e.g., 15 → 50) for more global context
- Increase `min_dist` (e.g., 0.1 → 0.3) for more separation

### Issue 3: Inconsistent Results

**Problem:** Different runs give different results

**Solutions:**
- Set `random_state` for reproducibility
- Increase `n_epochs` (e.g., 200 → 500) for more stable convergence
- Use `init='spectral'` for more consistent initialization

### Issue 4: Too Slow

**Problem:** Taking too long to run

**Solutions:**
- Decrease `n_neighbors` (e.g., 50 → 15)
- Decrease `n_epochs` (e.g., 500 → 200)
- Use fewer samples if possible
- Consider using GPU-accelerated libraries for production

## Tips for Success

### 1. Start Simple
```python
# Good first attempt
umap = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
```

### 2. Experiment Systematically

Test parameters one at a time:
```python
# Test different n_neighbors
for k in [5, 15, 30, 50]:
    umap = UMAP(n_neighbors=k, random_state=42)
    # ... fit and visualize

# Test different min_dist
for md in [0.0, 0.1, 0.3, 0.5]:
    umap = UMAP(min_dist=md, random_state=42)
    # ... fit and visualize
```

### 3. Validate Your Results

- Check if known clusters are separated
- Compare with domain knowledge
- Try multiple random seeds
- Cross-validate if using for ML

### 4. Understand Your Data

- Normalize features if scales vary widely
- Handle missing values before UMAP
- Consider which metric makes sense (euclidean vs cosine)

## Performance Considerations

### Time Complexity

- **k-NN computation**: O(n² d) for n samples, d dimensions
  - Can be improved to O(n log n d) with spatial data structures
- **Optimization**: O(n × n_epochs)
- **Overall**: O(n² d + n × n_epochs)

### Space Complexity

- **k-NN graph**: O(n × k)
- **Embedding**: O(n × n_components)
- **Overall**: O(n × k)

### Scaling Tips

1. **For large n**: Use approximate k-NN (production libraries do this)
2. **For large d**: Consider PCA preprocessing (reduce to ~50 dims)
3. **For quality**: Increase n_epochs (200 → 500)
4. **For speed**: Decrease n_neighbors (15 → 10)

## Advanced Topics

### 1. Supervised UMAP

You can guide UMAP with labels:
- Modify the graph construction to favor same-class connections
- Useful when you have some labeled data

### 2. Parametric UMAP

Learn a function (neural network) that maps high-D → low-D:
- Can transform new points efficiently
- Requires deep learning framework

### 3. Inverse Transform

Mapping low-D → high-D (not implemented here):
- Useful for generating new samples
- Requires storing training data and learning inverse mapping

## Further Reading

### Papers
- **Original UMAP paper**: McInnes, Healy, Melville (2018)
  - "UMAP: Uniform Manifold Approximation and Projection"
  - Very readable, includes mathematical details

### Comparisons
- "How to Use t-SNE Effectively" (Wattenberg, et al. 2016)
- "Dimensionality Reduction: A Comparative Review" (van der Maaten, et al. 2009)

### Applications
- Single-cell genomics: Visualizing cell populations
- NLP: Exploring word embeddings (Word2Vec, GloVe, BERT)
- Computer vision: Understanding neural network features

## Summary

**UMAP is a powerful, fast, and theoretically grounded dimensionality reduction technique.**

**Key takeaways:**
1. ✓ Preserves both local AND global structure
2. ✓ Much faster than t-SNE
3. ✓ Can be used for both visualization and feature engineering
4. ✓ Intuitive parameters (n_neighbors, min_dist)
5. ✓ Scales to large datasets

**Default settings work well:**
- `n_components=2` for visualization
- `n_neighbors=15` for balanced structure
- `min_dist=0.1` for moderate spacing
- `n_epochs=200` for good convergence

**When in doubt**, start with defaults and adjust based on results!

---

## Implementation Notes

This implementation is educational and prioritizes clarity. For production use:
- Use the official `umap-learn` library (highly optimized)
- Consider GPU acceleration for very large datasets
- Use approximate k-NN for faster computation

**Our implementation demonstrates the core algorithm** so you can understand how UMAP actually works!

---

**Happy embedding!** 🎨📊🔍
