# UMAP - Uniform Manifold Approximation and Projection

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [Overview](#overview)
3. [When to Use UMAP](#when-to-use-umap)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Algorithm Steps](#algorithm-steps)
6. [Step-by-Step Example: Six Points by Hand](#step-by-step-example-six-points-by-hand)
7. [Parameters Explained](#parameters-explained)
8. [Code Example](#code-example)
9. [Practical Use Cases](#practical-use-cases)
10. [UMAP vs t-SNE: Detailed Comparison](#umap-vs-t-sne-detailed-comparison)
11. [Common Issues and Solutions](#common-issues-and-solutions)
12. [Tips for Success](#tips-for-success)
13. [Performance Considerations](#performance-considerations)
14. [Advanced Topics](#advanced-topics)
15. [Simplifications vs. Canonical UMAP](#simplifications-vs-canonical-umap)
16. [Further Reading](#further-reading)
17. [Summary](#summary)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra
dependencies beyond NumPy. It finishes in about two seconds.

```python
# ---------------------------------------------------------------
# UMAP from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _22_umap.py  (the __main__ block runs a fuller version)
# Or copy the UMAP class from _22_umap.py and paste it below.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the UMAP class here (from _22_umap.py) ----
# class UMAP: ...

np.random.seed(42)

# Three well-separated Gaussian clusters in 5 dimensions
n_per_cluster = 40
cluster1 = np.random.randn(n_per_cluster, 5) + [0, 0, 0, 0, 0]
cluster2 = np.random.randn(n_per_cluster, 5) + [5, 5, 5, 5, 5]
cluster3 = np.random.randn(n_per_cluster, 5) + [10, 0, 10, 0, 10]
X = np.vstack([cluster1, cluster2, cluster3])
labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)

# Hold out the first 5 points of each cluster. Train and test indices are
# disjoint by construction - no row appears in both.
holdout = np.concatenate([np.arange(c * n_per_cluster, c * n_per_cluster + 5)
                          for c in range(3)])
is_train = np.ones(len(X), dtype=bool)
is_train[holdout] = False
X_train, y_train = X[is_train], labels[is_train]
X_test, y_test = X[holdout], labels[holdout]

model = UMAP(
    n_components=2,      # Reduce to 2D
    n_neighbors=15,      # Balance local/global
    min_dist=0.1,        # Moderate spacing
    n_epochs=60,         # Enough for 105 points
    random_state=42      # Reproducibility
)
X_embedded = model.fit_transform(X_train)


def knn_purity(embedding, y, k=10):
    """Share of each point's k nearest embedded neighbours with the same label."""
    d = np.sqrt(((embedding[:, None, :] - embedding[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float((y[np.argsort(d, axis=1)[:, :k]] == y[:, None]).mean())


print(f"Original shape : {X_train.shape}")
print(f"Embedded shape : {X_embedded.shape}")
print(f"Graph edges    : {len(model.graph_)}")
print(f"TRAIN 10-NN label purity : {knn_purity(X_embedded, y_train):.4f}")

# transform() places unseen points into the SAME map, without refitting
X_test_embedded = model.transform(X_test)
d_test = np.sqrt(((X_test_embedded[:, None, :] - X_embedded[None, :, :]) ** 2).sum(-1))
nearest = np.argmin(d_test, axis=1)
print(f"TEST  1-NN label agreement: {float((y_train[nearest] == y_test).mean()):.4f}")

for i in range(3):
    print(f"  point {i} (cluster {y_train[i]}) -> "
          f"({X_embedded[i, 0]:6.2f}, {X_embedded[i, 1]:6.2f})")

# a and b are FITTED from min_dist - they are not hard-coded constants
for min_dist in [0.0, 0.1, 0.5]:
    a, b = UMAP(min_dist=min_dist)._find_ab_params(min_dist)
    print(f"  min_dist={min_dist:.2f} -> a={a:.4f}, b={b:.4f}")
```

Expected output:
```
Original shape : (105, 5)
Embedded shape : (105, 2)
Graph edges    : 2066
TRAIN 10-NN label purity : 1.0000
TEST  1-NN label agreement: 1.0000
  point 0 (cluster 0) -> ( -1.27,  -9.36)
  point 1 (cluster 0) -> ( -0.08,  -8.05)
  point 2 (cluster 0) -> ( -0.77,  -7.91)
  min_dist=0.00 -> a=1.9328, b=0.7905
  min_dist=0.10 -> a=1.5769, b=0.8951
  min_dist=0.50 -> a=0.5830, b=1.3342
```

Running `python _22_umap.py` directly executes a larger four-part demo (Swiss roll,
cluster separation with `transform()`, a `min_dist` sweep, and euclidean vs cosine).
It takes about 9 seconds.

---

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
- **Large Datasets**: the `umap-learn` library handles 100,000+ samples efficiently
  (*this* from-scratch version builds a dense n x n distance matrix and is comfortable
  up to roughly 500-1000 samples - see [Simplifications](#simplifications-vs-canonical-umap))
- **Biological Data**: Single-cell genomics, protein analysis
- **Text Analysis**: Visualize word embeddings, document spaces

### Advantages Over Other Methods:
- **vs t-SNE**: 10-100x faster, preserves global structure, more general purpose
- **vs PCA**: Captures non-linear relationships, better visualization
- **vs Autoencoders**: No training needed, solid mathematical foundation

*(The speed advantage belongs to the `umap-learn` library's Numba kernels. This
from-scratch file matches UMAP's **quality** - see
[Simplifications](#simplifications-vs-canonical-umap) for the measured comparison -
but not its speed.)*

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
Σⱼ exp(-(max(0, dᵢⱼ - ρᵢ))/σᵢ) = log₂(k)
```

The `max(0, ·)` matters: the nearest neighbour has `dᵢⱼ = ρᵢ`, so its term is
`exp(0) = 1`. That single guaranteed term is UMAP's **local connectivity**
guarantee - no point is ever left isolated. The sum increases monotonically with
σᵢ, so `_smooth_knn_distances` finds σᵢ by plain bisection on `[0, 1e10]`.

**Why `log₂(k)`?** It is the entropy of a uniform distribution over `k` neighbours.
Asking for that much total connectivity means "widen σᵢ until point i has about
`log₂(k)` neighbours' worth of membership". In a dense region a small σᵢ suffices;
in a sparse region σᵢ grows. That per-point rescaling is precisely the "uniform
distribution on the manifold" assumption UMAP is named after.

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

#### Where `a` and `b` actually come from

`a` and `b` are **not constants**. They are the least-squares fit of `q(d)` to a
piecewise target curve built from `min_dist` and `spread`:

```
psi(d) = 1                              if d <  min_dist
       = exp(-(d - min_dist) / spread)  if d >= min_dist
```

In words: *"anything closer than `min_dist` counts as equally close; beyond that,
similarity decays exponentially."* Fitting `q(d) = 1/(1 + a·d^(2b))` to `psi(d)`
turns that verbal rule into two numbers. This is the **only** place `min_dist`
enters the algorithm, which is why it has to be a real fit - a lookup table of two
magic numbers makes `min_dist` inert.

`_find_ab_params` runs Levenberg-Marquardt on `(log a, log b)` (logs keep both
positive and let one solver span the whole range). Measured against
`scipy.optimize.curve_fit` on the same target curve, the worst relative error in
`a` or `b` is `5.0e-06` at the default `spread=1.0` (sweeping `min_dist` from 0 to
1 in steps of 0.005) and `7.8e-06` over spreads 0.25 to 20, keeping `umap-learn`'s
own precondition `min_dist <= spread`:

| min_dist | a | b | effect |
|---------:|------:|------:|--------|
| 0.00 | 1.9328 | 0.7905 | tightest packing |
| 0.10 | 1.5769 | 0.8951 | default |
| 0.25 | 1.1214 | 1.0575 | roomier |
| 0.50 | 0.5830 | 1.3342 | evenly spread |
| 0.99 | 0.1193 | 1.9164 | maximally spread |

Raising `min_dist` lowers `a` and raises `b`, which flattens `q` near the origin:
two points that are already closer than `min_dist` gain almost nothing by squeezing
closer still, so the attractive gradient there goes to nearly zero.

`spread` is not a constructor argument in this implementation - `UMAP` always fits
at `spread=1.0`, and the swept range above comes from calling the internal
`_find_ab_params(min_dist, spread=...)` directly. (`umap-learn` does expose
`spread` on its estimator.)

### 5. Deriving the SGD Gradients

This is the bridge from the cross-entropy above to the two lines of code in
`_optimize_embedding`. Write `s = ||yᵢ - yⱼ||²` so that `q = 1/(1 + a·s^b)`, and use

```
dq/ds       = -a·b·s^(b-1) / (1 + a·s^b)²
ds/dyᵢ      = 2 (yᵢ - yⱼ)
```

**Attractive term** (the `w log(w/q)` half; only `-w log q` depends on the layout):

```
d/dyᵢ [ -w log q ]  =  -w/q · dq/ds · 2(yᵢ - yⱼ)
                    =  +w · 2ab·s^(b-1) / (1 + a·s^b) · (yᵢ - yⱼ)
```

Gradient **descent** moves against that, so the code adds

```
grad_coef = -2ab·s^(b-1) / (1 + a·s^b)          # _optimize_embedding
```

**Repulsive term** (the `(1-w) log((1-w)/(1-q))` half):

```
d/dyᵢ [ -(1-w) log(1-q) ]  =  (1-w)/(1-q) · dq/ds · 2(yᵢ - yⱼ)
                           =  -(1-w) · 2b / (s·(1 + a·s^b)) · (yᵢ - yⱼ)
```

so descent adds

```
grad_coef = +2b / ((0.001 + s)·(1 + a·s^b))     # _optimize_embedding
```

Two deliberate deviations in that last line, both taken from `umap-learn`:
- `(1 - w)` is dropped. Negative samples are drawn uniformly at random, and for a
  random pair `w ≈ 0`, so `(1 - w) ≈ 1`.
- `0.001` is added to `s` so two coincident points do not divide by zero.

Every gradient component is finally clipped to `[-4, 4]`. Without the clip, a pair
that starts almost coincident produces a coefficient in the thousands and flings a
point out of the layout on the very first epoch.

*(Both were checked against a central finite difference of the cross-entropy. The
**derived** coefficients - the attractive one, and the repulsive one **without**
the `0.001` - agree with the finite difference to `2.7e-09` and `5.7e-08` worst
relative error. The repulsive coefficient **as coded** deliberately differs,
because the `0.001` is a real change to the formula and not a rounding: it costs
`1.0e-03` relative at `s = 1`, `1.0e-02` at `s = 0.1`, and `9.1e-01` at
`s = 1e-04` - which is exactly the point, since that is where the true gradient
blows up.)*

### 6. Negative Sampling: why UMAP is O(n), not O(n²)

The repulsive sum in the cross-entropy runs over **every** non-edge - about `n²`
pairs. t-SNE pays that cost. UMAP does not: it estimates the repulsion from a
handful of uniformly random points per attractive update.

- **Sampling distribution:** uniform over all `n` vertices. Uniform is correct here
  because the term being estimated, `Σ (1-wᵢₖ) log(1-qᵢₖ)`, has essentially the same
  weight `(1-w) ≈ 1` for every non-neighbour.
- **Why 5?** `negative_sample_rate=5` is `umap-learn`'s default: enough samples that
  the repulsive force does not jitter, few enough that repulsion costs only 5x the
  attraction. Raising it spreads clusters further apart; lowering it lets them merge.
- **Only the anchor moves.** A negative sample is a random stand-in for "everything
  else", not a genuine partner, so `embedding[i]` is updated and `embedding[k]` is
  left alone.

The matching trick on the attractive side is **edge sampling**. Rather than visiting
every edge every epoch and scaling by `w`, an edge is visited once every
`w_max / w` epochs, so over the whole run it is attracted in proportion to its
weight. Edges with `w < w_max / n_epochs` would never come up at all and are pruned
up front. Together these two ideas make one epoch cost
`O(n · n_neighbors · (1 + n_negative))` instead of `O(n²)`.

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
    # Such that: Σⱼ exp(-(max(0, dᵢⱼ - ρᵢ))/σᵢ) ≈ log₂(k)
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
    W = dense_weight_matrix(graph)              # n x n
    D = diag(W.sum(axis=1))                     # degree matrix
    L = I - D**-0.5 @ W @ D**-0.5               # symmetric normalised Laplacian
    eigenvalues, eigenvectors = eigh(L)         # ascending order
    embedding = eigenvectors[:, 1:n_components + 1]   # skip the sqrt-degree vector
    embedding *= 10.0 / abs(embedding).max()    # spread to ~10 units
    embedding += normal(scale=1e-4)             # break exact ties
else:
    # Random initialization
    embedding = random_normal(n_samples, n_components)
```

**Why skip column 0?** Because it is fixed by the graph's degrees alone. For the
**symmetric** normalised Laplacian `L_sym = I - D^-1/2 W D^-1/2`, the eigenvector
for eigenvalue 0 on a connected component is `D^(1/2)·1`, i.e. proportional to
`sqrt(degree)` - check it: `L_sym D^(1/2)1 = D^(1/2)1 - D^-1/2 W 1 = D^(1/2)1 -
D^-1/2 d = 0`. (The *constant* vector is the null vector of the **random-walk**
Laplacian `I - D^-1 W`; the two coincide only on a regular graph. Measured on the
connected k-NN graph this class builds from `RandomState(0).randn(60, 5)` with
`n_neighbors=10`, whose degrees range 3.32-9.43: `std(v0) = 1.83e-02`, so `v0` is
*not* constant, while `|corr(v0, sqrt(degree))| = 1.000000000000` and
`std(v0 / sqrt(degree)) = 3.5e-15`. The absolute value is not sloppiness: `eigh`
pins each eigenvector only up to sign, so the correlation itself comes out `+1` or
`-1` depending on the LAPACK build - it is `-1` on the machine these numbers were
measured on.)

Either way it carries no layout information, so it is skipped. Columns 1, 2, ...
are the smoothest *non-trivial* functions on the graph: vertices joined by heavy
edges get similar values. Starting the SGD from that layout instead of from noise
is the main reason UMAP is more reproducible than t-SNE.

`_spectral_initialization` implements exactly this with `np.linalg.eigh`. It is
O(n³), which is fine for the few-hundred-point datasets this file targets and
under a second at n = 500.

### Step 5: Optimize via SGD

Iteratively improve the embedding using stochastic gradient descent.

```python
# Prune edges too weak to be sampled even once, then schedule the rest:
# an edge of weight w comes up every w_max / w epochs.
edges = [e for e in graph if graph[e] >= w_max / n_epochs]
epochs_per_sample   = w_max / weights
epoch_of_next_sample = epochs_per_sample.copy()

a, b = find_ab_params(min_dist)

for epoch in range(n_epochs):
    alpha = learning_rate * (1.0 - epoch / n_epochs)     # linear decay to 0

    for idx in where(epoch_of_next_sample <= epoch):     # this edge's turn
        i, j = edges[idx]
        epoch_of_next_sample[idx] += epochs_per_sample[idx]

        diff = embedding[i] - embedding[j]
        s    = sum(diff ** 2)                            # squared low-D distance

        # Attractive force: -2ab*s^(b-1) / (1 + a*s^b)
        grad_coef = 0.0 if s <= 0 else -2*a*b * s**(b-1) / (1 + a * s**b)
        grad = clip(grad_coef * diff, -4.0, 4.0)

        embedding[i] += alpha * grad
        embedding[j] -= alpha * grad

        # Negative sampling: repulsive force, anchor only
        for k in random_integers(n_samples, size=5):
            diff_ik = embedding[i] - embedding[k]
            s_ik    = sum(diff_ik ** 2)

            # Repulsive force: +2b / ((0.001 + s) * (1 + a*s^b))
            grad_coef = 2*b / ((0.001 + s_ik) * (1 + a * s_ik**b))
            embedding[i] += alpha * clip(grad_coef * diff_ik, -4.0, 4.0)
```

Note that the attractive gradient is **not** multiplied by `w_ij`. The weight is
already expressed in *how often* the edge is sampled, so multiplying again would
count it twice. Note also the two guards - `s <= 0` and the `0.001` - and the
`[-4, 4]` clip: without them a pair of duplicate rows produces `0 ** -0.2 = inf`
and the whole embedding turns into NaN.

## Step-by-Step Example: Six Points by Hand

Every number below was produced by the actual methods in `_22_umap.py`, so you can
reproduce them line by line. Six points in 2-D, `n_neighbors=3`: three in a **tight**
group and three in a **loose** group. Watch how ρ and σ adapt to that density
difference - that adaptation is the whole point of the algorithm.

```python
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.6],      # tight group (0,1,2)
              [8.0, 8.0], [11.0, 8.0], [8.5, 11.0]])   # loose group (3,4,5)

u = UMAP(n_neighbors=3, min_dist=0.1, n_epochs=20, random_state=42)
D        = u._compute_distances(X)
ki, kd   = u._compute_knn_graph(D)
rho, sig = u._smooth_knn_distances(kd)
graph    = u._compute_membership_strengths(ki, kd, rho, sig)
```

**Step 1-2: k-NN and the local metric.** Target is `log₂(3) = 1.585`.

| point | 3 nearest | their distances | ρᵢ | σᵢ |
|---|---|---|---|---|
| 0 | 1, 2, 3 | 1.000, 1.600, 11.314 | 1.0000 | 1.1186 |
| 1 | 0, 2, 3 | 1.000, 1.887, 10.630 | 1.0000 | 1.6391 |
| 2 | 0, 1, 3 | 1.600, 1.887, 10.245 | 1.6000 | 0.5349 |
| 3 | 4, 5, 2 | 3.000, 3.041, 10.245 | 3.0000 | 0.0772 |
| 4 | 3, 5, 2 | 3.000, 3.905, 12.726 | 3.0000 | 1.6721 |
| 5 | 3, 4, 2 | 3.041, 3.905, 12.673 | 3.0414 | 1.5985 |

ρ is simply "distance to my nearest neighbour": 1.0 for the tight group, 3.0 for the
loose one. **The same absolute distance means different things to different points**,
and ρ is what encodes that. Point 3's σ is tiny (0.0772) because its two neighbours
sit at 3.000 and 3.041 - nearly tied, so their two membership terms alone already
exceed the target and σ must shrink hard.

**Step 3: directed membership** `v(i→j) = exp(-max(0, dᵢⱼ - ρᵢ)/σᵢ)`

```
v(0→1) = exp(-max(0, 1.000 - 1.000)/1.1186) = 1.0000    <- nearest neighbour, always 1
v(0→2) = exp(-max(0, 1.600 - 1.000)/1.1186) = 0.5849
v(0→3) = exp(-max(0,11.314 - 1.000)/1.1186) = 0.0001    <- across the gap
v(2→0) = exp(-max(0, 1.600 - 1.600)/0.5349) = 1.0000    <- 0 IS 2's nearest
v(4→5) = exp(-max(0, 3.905 - 3.000)/1.6721) = 0.5820
```

**Step 4: fuzzy union** `w(i,j) = v(i→j) + v(j→i) - v(i→j)·v(j→i)`

| edge | v(i→j) | v(j→i) | w(i,j) |
|---|---|---|---|
| (0,1) | 1.0000 | 1.0000 | **1.0000** |
| (0,2) | 0.5849 | 1.0000 | **1.0000** |
| (1,2) | 0.5821 | 0.5850 | **0.8266** |
| (3,4) | 1.0000 | 1.0000 | **1.0000** |
| (3,5) | 0.5850 | 1.0000 | **1.0000** |
| (4,5) | 0.5820 | 0.5825 | **0.8255** |
| (1,3) | 0.0028 | 0.0000 | **0.0028** |
| (2,4) | 0.0000 | 0.0030 | **0.0030** |

Two things to notice:

1. **Edge (0,2) has weight 1.0 even though `v(0→2)` is only 0.585.** Point 0 is not
   especially sure about 2, but 2 is *certain* about 0 (0 is 2's nearest neighbour).
   The union `A + B - A·B` = "A **or** B" keeps the edge. This is exactly why UMAP
   symmetrises with a t-conorm instead of averaging - averaging would have given
   0.79 and weakened a link one endpoint was certain about.
2. **The two groups are joined only by edges of weight ~0.003.** With
   `n_epochs = 20`, pruning drops everything below `1.0/20 = 0.05`, so those
   cross-group edges are removed entirely before optimisation begins.

**Step 5: the low-D kernel.** `_find_ab_params(0.1)` returns `a = 1.5769`,
`b = 0.8951`, giving

```
q(0.1) = 0.9751     q(0.5) = 0.6868     q(1.0) = 0.3881     q(2.0) = 0.1549
```

**Step 6: the result.** After 20 epochs, `fit_transform` returns pairwise 2-D
distances of roughly

```
within  the tight group : 0.10 - 7.09
within  the loose group : 0.01 - 0.41
between the two groups  : 15.8 - 16.8
```

The 40x gap between "same group" and "different group" is the fuzzy graph made
geometric.

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

**Measured on the demo's three planted clusters** (105 points, 30 epochs), showing
the fitted `(a, b)` and the resulting mean cluster radius:

| min_dist | a | b | mean cluster radius | 10-NN purity |
|---------:|------:|------:|--------------------:|-------------:|
| 0.00 | 1.9328 | 0.7905 | 0.766 | 1.0000 |
| 0.25 | 1.1214 | 1.0575 | 0.966 | 1.0000 |
| 0.50 | 0.5830 | 1.3342 | 1.175 | 1.0000 |

Clusters spread out as `min_dist` rises but stay perfectly separated - `min_dist`
controls *packing*, not *separation*. (Run `python _22_umap.py` to reproduce this
table.)

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

`n_epochs` does double duty: it is also the **edge-pruning threshold**. Edges with
weight below `w_max / n_epochs` are dropped before optimisation, because they would
never be sampled even once. Raising `n_epochs` therefore keeps *more* weak edges as
well as running longer.

### init (default='spectral')

How the embedding is seeded before the SGD starts.

- **'spectral'**: eigenvectors 1..n_components of the symmetric normalised graph
  Laplacian, rescaled to about 10 units across plus `1e-4` noise. Structure-aware,
  so runs with the same `random_state` are highly stable.
- **'random'**: `N(0, 1)` scaled by 10. Useful for checking that a result is not an
  artefact of the initialisation.

On the 5-cluster benchmark used
[below](#umap-vs-t-sne-detailed-comparison) both reach 10-NN purity 1.0000
(trustworthiness 0.9641 spectral, 0.9645 random); spectral is the safer default
because it does not depend on a lucky draw.

### random_state (default=None)

Seed for the model's **private** `np.random.RandomState`.

- An `int` makes `fit()` / `fit_transform()` bit-for-bit reproducible; calling
  `fit()` twice on the same data gives an identical embedding.
- That guarantee is per numeric stack, not universal. The SGD amplifies last-bit
  differences: perturbing the fitted `(a, b)` by `1e-10` relative moves individual
  points by up to 5.6 units on a layout only 16.5 x 10.7 units across (measured on
  the Example 2 global fit - 105 points, `n_neighbors=20`, `n_epochs=40`,
  `random_state=42`), so a different BLAS/LAPACK build can print coordinates
  unlike the ones quoted in this file even at the same seed.
- `None` draws fresh randomness on every fit.
- The model never calls `np.random.seed()`, so your own global NumPy stream is left
  untouched. (An earlier version of this file did reseed it - constructing a `UMAP`
  object silently changed every later `np.random` call in the caller's script.)

### verbose (default=0)

- **0**: silent.
- **1**: progress - k-NN construction, fuzzy set, epoch counter, completion.
- **2**: adds the smallest Laplacian eigenvalues from the spectral init and the
  fitted `q(d) = 1/(1 + a·d^(2b))` kernel. Useful when a layout looks wrong: a run
  of near-zero eigenvalues means the k-NN graph is disconnected.

## Code Example

```python
import numpy as np
from _22_umap import UMAP   # run from inside the "22. UMAP" folder

# Generate high-dimensional data
np.random.seed(42)
X = np.random.randn(180, 20)  # 180 samples, 20 features

# Create three clusters
X[:60] += [5, 0, 0, 0, 0] + [0]*15
X[60:120] += [0, 5, 0, 0, 0] + [0]*15
X[120:] += [0, 0, 5, 0, 0] + [0]*15
y = np.repeat([0, 1, 2], 60)

# Apply UMAP
umap = UMAP(
    n_components=2,      # Reduce to 2D
    n_neighbors=15,      # Balance local/global
    min_dist=0.1,        # Moderate spacing
    n_epochs=60,         # Good convergence at this size
    random_state=42,     # Reproducibility
    verbose=1            # Show progress
)

# Fit and transform
X_embedded = umap.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Embedded shape: {X_embedded.shape}")

# Did it work? Check that the planted clusters stayed together.
d = np.sqrt(((X_embedded[:, None, :] - X_embedded[None, :, :])**2).sum(-1))
np.fill_diagonal(d, np.inf)
purity = (y[np.argsort(d, axis=1)[:, :10]] == y[:, None]).mean()
print(f"10-NN label purity: {purity:.4f}")

# Now you can visualize X_embedded in 2D!
```

Expected output (about 3 seconds):
```
Computing k-NN graph with k=15...
Computing fuzzy simplicial set...
Optimizing embedding in 2D...
Epoch 1/60
Epoch 51/60
Epoch 60/60
UMAP embedding complete!
Original shape: (180, 20)
Embedded shape: (180, 2)
10-NN label purity: 0.9900
```

## Practical Use Cases

### 1. Visualizing High-Dimensional Data

```python
# Example: Visualizing the sklearn digits dataset (8x8 images = 64 dimensions)
import numpy as np
from _22_umap import UMAP
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data[:300]      # (300, 64) - subsample: see the note below
y = digits.target[:300]

# Apply UMAP.  300 rows at n_epochs=60 takes about 6 seconds with this
# from-scratch implementation; the full 1797 rows would take several minutes
# because _compute_distances builds a dense n x n matrix.
umap = UMAP(n_components=2, n_neighbors=15, n_epochs=60, random_state=42)
X_embedded = umap.fit_transform(X)

# Did the digit classes separate?
d = np.sqrt(((X_embedded[:, None, :] - X_embedded[None, :, :])**2).sum(-1))
np.fill_diagonal(d, np.inf)
print(f"10-NN digit purity: {(y[np.argsort(d, 1)[:, :10]] == y[:, None]).mean():.4f}")

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='Spectral', s=5)
plt.colorbar()
plt.title('UMAP projection of Digits dataset')
plt.show()
```

Expected output:
```
10-NN digit purity: 0.9670
```

Ten handwritten-digit classes, 64 raw pixel dimensions, and 97% of every point's
ten nearest neighbours in the 2-D map are the same digit. Nothing told UMAP the
labels - it recovered them from pixel geometry alone.

### 2. Feature Engineering for ML

Always generate labels with **real signal** in them, and always compare against the
raw-feature baseline. (Generating `y` with `np.random.randint` labels pure noise, so
any "accuracy" you print is a coin flip near 0.50 - not a result.)

```python
# Reduce dimensions before classification
import numpy as np
from _22_umap import UMAP
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(random_state=42)

# CASE A: the label is a linear rule over 2 of 30 features.
# UMAP is UNSUPERVISED - it has no way to know those 2 features matter.
np.random.seed(0)
X_a = np.random.randn(150, 30)
y_a = (X_a[:, 0] + X_a[:, 1] > 0).astype(int)
R_a = UMAP(n_components=5, n_neighbors=15, n_epochs=30, random_state=42).fit_transform(X_a)
print(f"A  raw 30-D : {cross_val_score(clf, X_a, y_a, cv=5).mean():.3f}")
print(f"A  UMAP 5-D : {cross_val_score(clf, R_a, y_a, cv=5).mean():.3f}")

# CASE B: the label IS the cluster identity - exactly what UMAP preserves.
np.random.seed(0)
centres = np.random.randn(3, 5) * 6
X_b = np.random.randn(150, 30)
y_b = np.repeat([0, 1, 2], 50)
X_b[:, :5] += centres[y_b]
R_b = UMAP(n_components=5, n_neighbors=15, n_epochs=30, random_state=42).fit_transform(X_b)
print(f"B  raw 30-D : {cross_val_score(clf, X_b, y_b, cv=5).mean():.3f}")
print(f"B  UMAP 5-D : {cross_val_score(clf, R_b, y_b, cv=5).mean():.3f}")
```

Expected output (about 6 seconds):
```
A  raw 30-D : 0.933
A  UMAP 5-D : 0.627
B  raw 30-D : 1.000
B  UMAP 5-D : 1.000
```

**This is the honest picture, and it is the most important thing on this page about
using UMAP as a preprocessor.** UMAP preserves *neighbourhood* structure. In case B
the label is neighbourhood structure, so 30 -> 5 dimensions costs nothing. In case A
the label is a linear rule hidden in 2 of 30 features; UMAP's k-NN graph is dominated
by the other 28 noise dimensions, so the reduction **destroys** the signal and
accuracy falls from 0.933 to 0.627.

Use UMAP before a model when you believe the label lives in the manifold's cluster
structure. If the label is a sparse linear rule, use a supervised feature selector
(or supervised UMAP, which this implementation does not provide) instead.

### 3. Exploring Different Parameter Settings

```python
import numpy as np
import matplotlib.pyplot as plt
from _22_umap import UMAP

# Any dataset will do; here are two planted clusters in 20-D
np.random.seed(42)
X = np.random.randn(120, 20)
X[:60, :5] += 4

# Compare different n_neighbors (three fits, about 5 seconds total)
for n_neighbors in [5, 15, 30]:
    umap = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        n_epochs=40,
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

*The Speed and Scalability rows describe the `umap-learn` library. On a 5-cluster
benchmark (150 points in 10-D: 5 groups of 30 around centres
`RandomState(42).randn(5, 10) * 5`, unit-variance noise, 200 epochs) this
pure-Python teaching version took 9.1 s against 0.2 s for `sklearn.manifold.TSNE`
(best of 3 runs each) - about 40x slower - while scoring trustworthiness 0.964 vs
t-SNE's 0.974, and 10-NN purity 1.000 for both. The trustworthiness figures are
`sklearn.manifold.trustworthiness` at its default `n_neighbors=5`, not the 10 used
for purity; at `n_neighbors=10` they are 0.974 and 0.985. The algorithm is right;
only the implementation is slow.*

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
- **Decrease** `min_dist` (e.g., 0.1 → 0.0) so clusters pack tightly and the gaps
  between them become visible. Raising `min_dist` does the opposite - it is what
  *causes* a uniform, structureless look. (Measured: mean cluster radius 0.766 at
  `min_dist=0.0` versus 1.175 at `min_dist=0.5` on the same data.)

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
- **Spectral initialization**: O(n³) - `np.linalg.eigh` on the dense Laplacian
- **Optimization**: O(n_epochs × n × n_neighbors × (1 + n_negative_samples))
  - The number of directed entries in `graph_` is at most `2·n·n_neighbors`, and
    in practice well under it because many neighbour pairs are mutual and only get
    stored once each way (measured 2,878 at n = 150, k = 15, against the bound of
    4,500). Each attractive event also draws 5 negative samples, so the naive
    `O(n × n_epochs)` understates the real cost by roughly `2·k·6`.
- **Overall**: O(n² d + n³ + n_epochs × n × k × 6)

### Space Complexity

- **Pairwise distance matrix**: O(n²) - `_compute_distances` allocates a dense
  `(n_samples, n_samples)` float64 array unconditionally
- **k-NN graph**: O(n × k)
- **Embedding**: O(n × n_components)
- **Overall**: O(n²) - this implementation materialises the full pairwise distance
  matrix. At n = 10,000 that is already 0.80 GB and at n = 100,000 it would be
  80 GB. Production UMAP uses approximate nearest neighbours and never builds it.

### Scaling Tips

1. **For large n**: Use approximate k-NN (production libraries do this)
2. **For large d**: Consider PCA preprocessing (reduce to ~50 dims)
3. **For quality**: Increase n_epochs (200 → 500)
4. **For speed**: Decrease n_neighbors (15 → 10)

## Advanced Topics

### Transforming New Data (out-of-sample points)

`transform(X_new)` embeds unseen rows into an **already fitted** map, without
refitting and without moving the existing points:

```python
model = UMAP(n_components=2, n_neighbors=15, n_epochs=60, random_state=42)
X_embedded = model.fit_transform(X_train)      # build the map
X_test_embedded = model.transform(X_test)      # place new points into it
```

Three steps, all reusing the fitted model:

1. Find each new point's `k` nearest **training** rows and compute the same
   ρ / σ local metric that `fit()` used, giving membership strengths to them.
2. Place the new point at the strength-weighted average of those neighbours'
   existing coordinates.
3. Run `n_epochs // 4` attractive-only SGD sweeps with the training embedding
   frozen.

Calling `transform()` before `fit()` raises a clear
`ValueError: This UMAP instance is not fitted yet.`

**Limitation:** this is the simplified version - no negative sampling on the new
points, and the attractive gradient is scaled by the edge weight directly rather
than through edge scheduling. New points can therefore sit slightly closer to their
neighbours than a full refit would place them. If you need the most faithful
possible layout of train + test together, run `fit_transform` on the combined data
instead (but note that this is *transductive*: the test rows influence the map).

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

## Simplifications vs. Canonical UMAP

Everything in the Mathematical Foundation above is implemented literally. These are
the places where `_22_umap.py` deliberately differs from `umap-learn`, what the
library does instead, and what it costs you.

| Area | Canonical `umap-learn` | Here | Practical consequence |
|---|---|---|---|
| Nearest neighbours | Approximate NN-descent, O(n^1.14) | Exact dense `n x n` matrix | Exact answers, but O(n²) memory - keep n under ~500-1000 |
| σ convention | `n_neighbors` counts the point itself, so `k-1` true neighbours vs `log₂(k)` | Paper's Algorithm 2: `k` true neighbours vs `log₂(k)` | σ ~2.6% smaller than `umap-learn(n_neighbors=k+1)`; mean edge-weight difference 0.008 over 1544 edges |
| `local_connectivity` | Tunable; interpolates ρ between neighbours | Fixed at 1 (ρ = 1st-neighbour distance) | No control over how aggressively isolated points are connected |
| Disconnected graphs | `multi_component_layout` lays out each component separately, then arranges them | Plain `eigh`; with `c` components the 0-eigenspace is `c`-dimensional, so columns 1..`n_components` are an arbitrary basis of it and each component becomes a blob whose internal spread follows `sqrt(degree)`, not its geometry | Measured on 3 planted components (75 points, 10-unit layout): max point-to-centroid spread 1.4-2.2 at init, one component collapsed to 2e-4. The SGD still separates them (10-NN purity 1.0000 over 3 seeds), but the first few epochs do work the library avoids |
| Negative sample count | Scheduled via `epochs_per_negative_sample` | Fixed 5 per attractive event | Identical in expectation; slightly less variance control |
| `transform()` | Attractive **and** repulsive passes against the frozen training set | Attractive-only | New points sit marginally closer to their neighbours |
| Objective variants | `repulsion_strength` (gamma), `set_op_mix_ratio`, supervised/`target_metric` modes | Not exposed | No supervised UMAP, no fuzzy-intersection blending |
| Inverse transform | `inverse_transform` maps low-D back to high-D | Not implemented | Cannot generate new high-D samples |
| Speed | Numba-compiled kernels | Pure Python/NumPy | 9.1 s vs 0.2 s for `sklearn`'s t-SNE on 150 points, 200 epochs |

**What is NOT simplified**, and was verified numerically. The `umap-learn` package
is not installed here, so nothing below was compared against the library at
runtime: the references are a port of `umap-learn`'s published `smooth_knn_dist`
(for ρ and σ), `scipy.optimize.curve_fit` (for `a` and `b`), central finite
differences of the cross-entropy (for the gradients), and `sklearn`'s t-SNE, PCA
and `trustworthiness` as peers:

- ρ matches `umap-learn` exactly (max absolute difference 0.0).
- σ matches a port of `umap-learn`'s bisection driven to the paper's target to
  1.1e-05 absolute (per-point ratio 0.999989-1.000013) across n/k of 80/15, 60/5,
  200/30, 150/15 and 120/10. The one case where the two drift apart is `k = 2`:
  there the target `log₂(2) = 1` is already reached by the guaranteed
  nearest-neighbour term alone, so σ is only pinned down to the tolerance band and
  any small σ is as good as any other.
- `(a, b)` match `scipy.optimize.curve_fit` to a worst relative error of 5.0e-06
  over `min_dist` swept 0 to 1 at the default `spread=1.0`, and 7.8e-06 over
  spreads 0.25 to 20.
- The symmetrisation is the true probabilistic t-conorm, and the maximum edge weight
  is exactly 1.0, as the local-connectivity guarantee requires.
- Both **derived** SGD gradients agree with a central finite difference of the
  cross-entropy to 2.7e-09 (attractive) and 5.7e-08 (repulsive) worst relative
  error; the repulsive one as coded carries `umap-learn`'s `0.001` guard, which is
  a deliberate 1.0e-03 relative deviation at `s = 1` and larger below that. Both
  are clipped to `[-4, 4]` as in `umap-learn`.
- On the 5-cluster benchmark above the embedding scores trustworthiness 0.964 and
  10-NN purity 1.000, against `sklearn` t-SNE's 0.974 / 1.000, PCA's 0.954 / 0.997
  and the floor set by a random 2-D layout (`RandomState(0).randn(150, 2)`),
  0.502 / 0.159. Every trustworthiness number here is
  `sklearn.manifold.trustworthiness` at its default `n_neighbors=5` while the
  purities are at 10; at `n_neighbors=10` the same four layouts score
  0.974 / 0.985 / 0.970 / 0.500.

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
2. ✓ Much faster than t-SNE (the `umap-learn` library; this teaching version is not)
3. ✓ Can be used for both visualization and feature engineering
4. ✓ Intuitive parameters (n_neighbors, min_dist)
5. ✓ Scales to large datasets (again: the library, not this O(n²) implementation)

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

**Honest performance numbers for this file** (measured, Python 3.13 / NumPy 2.3):

| Task | Time |
|---|---|
| `python _22_umap.py` (the full four-part demo, 8 fits) | 8.6 s |
| 105 points, 5-D, k=15, 60 epochs | 1.9 s |
| 150 points, 10-D, k=15, 200 epochs | 9.1 s (`sklearn` t-SNE: 0.2 s) |
| 300 points, 10-D, k=15, 200 epochs | 20.0 s |

(All four are the best of 3 runs on a quiet machine. These are pure-Python loops
competing for one core, so a busy machine is slower by a large factor - repeated
timings of the demo alone ranged from 8.6 s to 31 s on the same hardware.)

The cost is dominated by the pure-Python SGD loop: the 150-point row fires 188,314
attractive events across its 200 epochs, so roughly 48 microseconds each including
the 5 negative samples that ride along with every one. See
[Simplifications](#simplifications-vs-canonical-umap) for the full list of
differences from `umap-learn`.

**Our implementation demonstrates the core algorithm** so you can understand how UMAP actually works!

---

**Happy embedding!** 🎨📊🔍
