# t-SNE Algorithm from Scratch: A Comprehensive Guide

Welcome to the world of dimensionality reduction and visualization! In this comprehensive guide, we'll explore t-SNE (t-Distributed Stochastic Neighbor Embedding) - one of the most powerful algorithms for visualizing high-dimensional data in 2D or 3D space.

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is t-SNE?](#what-is-t-sne)
3. [How t-SNE Works](#how-t-sne-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Comparing with Other Methods](#comparing-with-other-methods)
11. [Computational Complexity](#computational-complexity)
12. [Simplifications vs. Canonical t-SNE](#simplifications-vs-canonical-t-sne)
13. [Advantages and Limitations](#advantages-and-limitations)
14. [Key Concepts to Remember](#key-concepts-to-remember)
15. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra
dependencies beyond NumPy, and it finishes in about two seconds.

t-SNE is **unsupervised** and has no `predict()`, so there is no train/test split to
report. The honest quality measure is **neighbour preservation**: what fraction of each
point's nearest neighbours in the original space are still its nearest neighbours in the
embedding. The script computes that with plain NumPy, and compares it against a random
2-D layout so the number means something.

```python
# ---------------------------------------------------------------
# t-SNE from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _14_tsne.py   (the __main__ block runs this)
# Or copy the TSNE class from _14_tsne.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the TSNE class here (from _14_tsne.py) ----
# class TSNE: ...

np.random.seed(42)

def make_blobs(n_per_cluster=50, n_features=10, separation=6.0):
    """Three well-separated Gaussian blobs in n_features dimensions."""
    direction = np.random.randn(n_features)
    direction = direction / np.linalg.norm(direction)
    offsets = [0.0, separation, -separation]

    chunks, labels = [], []
    for cluster_id, offset in enumerate(offsets):
        center = offset * direction
        chunks.append(center + np.random.randn(n_per_cluster, n_features))
        labels.extend([cluster_id] * n_per_cluster)
    return np.vstack(chunks), np.array(labels)

def neighbor_preservation(X_high, Y_low, k=10):
    """Share of each point's k nearest neighbours that survive the projection."""
    def nearest(Z):
        sq = np.sum(np.square(Z), axis=1)
        D = sq[:, np.newaxis] + sq[np.newaxis, :] - 2 * np.dot(Z, Z.T)
        np.fill_diagonal(D, np.inf)
        return np.argsort(D, axis=1)[:, :k]

    high_nn, low_nn = nearest(X_high), nearest(Y_low)
    shared = [len(set(high_nn[i]) & set(low_nn[i])) for i in range(len(X_high))]
    return float(np.mean(shared)) / k

X, labels = make_blobs()

# ------ Fit ------
model = TSNE(n_components=2, perplexity=15, learning_rate=200,
             n_iter=500, random_state=42)
Y = model.fit_transform(X)

print(f"Embedding shape   : {Y.shape}")
print(f"Final KL(P||Q)    : {model.kl_divergence_:.4f}")
print(f"kNN(10) preserved : {neighbor_preservation(X, Y, k=10):.3f}")

# ------ Did the planted clusters survive? ------
for cluster_id in np.unique(labels):
    pts = Y[labels == cluster_id]
    centroid = pts.mean(axis=0)
    spread = np.mean(np.sqrt(np.sum((pts - centroid) ** 2, axis=1)))
    print(f"  cluster {cluster_id}: centroid=({centroid[0]:7.2f},{centroid[1]:7.2f})"
          f"  spread={spread:5.2f}")

# ------ What perplexity actually does ------
print("\n  perplexity    final KL    kNN(10) preserved")
for perplexity in [5, 15, 30]:
    sweep = TSNE(n_components=2, perplexity=perplexity, learning_rate=200,
                 n_iter=400, random_state=42)
    Y_sweep = sweep.fit_transform(X)
    print(f"     {perplexity:5.1f}       {sweep.kl_divergence_:7.4f}"
          f"           {neighbor_preservation(X, Y_sweep, k=10):.3f}")
```

Expected output:
```
Embedding shape   : (150, 2)
Final KL(P||Q)    : 0.9288
kNN(10) preserved : 0.463
  cluster 0: centroid=(  12.17,  -3.10)  spread= 8.25
  cluster 1: centroid=( -15.47, -30.82)  spread= 8.94
  cluster 2: centroid=(   3.30,  33.92)  spread=10.28

  perplexity    final KL    kNN(10) preserved
       5.0        2.0721           0.329
      15.0        1.4188           0.368
      30.0        0.5203           0.487
```

**How to read those numbers:**

- The cluster centroids are 38 to 67 units apart while each cluster's internal spread is
  only 8-10 units. Separation swamps spread, so the three planted blobs stayed three
  blobs. That is the property t-SNE promises.
- `kNN(10) preserved = 0.463` sounds low until you see that a *random* 2-D layout of the
  same points scores about `0.055` (the `__main__` block in `_14_tsne.py` prints this
  baseline for you). The blobs are isotropic 10-D Gaussians, so their internal
  neighbour ordering genuinely cannot all fit into two dimensions - no method could reach
  1.0 here. Beating random by 8x is the local structure t-SNE actually kept.
- In the sweep, KL **falls** as perplexity rises. That is not a universal law: on these
  blobs a large perplexity spreads `P` smoothly over the whole cluster, which is an easy
  target for 2-D, while a tiny perplexity insists on the exact ordering of five nearest
  neighbours, which 2-D cannot honour. On data with real fine sub-structure the trade-off
  reverses. **Never compare KL across datasets or across perplexities as if it were an
  accuracy.**

---

## What is t-SNE?

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a **non-linear dimensionality reduction technique** primarily used for **visualizing high-dimensional data**. It excels at preserving local structure - keeping similar data points close together while revealing cluster patterns.

**Real-world analogy**: 
Imagine you have a 3D sculpture that you want to photograph (project to 2D). Instead of just taking a flat projection, t-SNE is like an intelligent artist who arranges objects on a canvas to preserve which objects were near each other in 3D, making clusters and relationships visible even in 2D!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Non-linear Dimensionality Reduction |
| **Learning Style** | Unsupervised Learning |
| **Primary Use** | Data Visualization, Exploratory Analysis |
| **Output** | 2D or 3D embedding coordinates |
| **Key Strength** | Preserves local structure, reveals clusters |

### The Core Idea

```
"Convert high-dimensional distances to probabilities representing similarities,
then find a low-dimensional map that matches these similarity patterns"
```

Unlike PCA (which preserves global structure linearly), t-SNE:
- Focuses on preserving **local neighborhoods**
- Can capture **non-linear relationships**
- Excels at **revealing cluster structure**
- Uses **probability distributions** to model similarities

### When to Use t-SNE

**Perfect for:**
- 📊 Visualizing high-dimensional datasets (100+ dimensions)
- 🔍 Exploring cluster structure in your data
- 🖼️ Visualizing image embeddings, word vectors, or features
- 🧬 Analyzing biological data (gene expression, protein structures)
- 🎨 Understanding neural network representations

**Not ideal for:**
- ❌ Exact distance preservation (use MDS)
- ❌ Interpretable dimensions (use PCA)
- ❌ Very large datasets (>10,000 points) without optimization
- ❌ Finding outliers (t-SNE can hide them)

---

## How t-SNE Works

### The Algorithm in 5 Steps

```
Step 1: Compute pairwise distances in high-dimensional space
         ↓
Step 2: Convert distances to probabilities (Gaussian distribution)
         Adjust variance to achieve target "perplexity"
         ↓
Step 3: Initialize random low-dimensional embedding (2D or 3D)
         ↓
Step 4: Compute low-dimensional probabilities (Student t-distribution)
         ↓
Step 5: Minimize KL divergence using gradient descent
         Move points to match high-D and low-D probabilities
```

### Visual Example

Let's say we have 5 points in 3D space that we want to map to 2D:

```
High-Dimensional Space (3D):
Points: A, B, C, D, E

Compute distances:
  A-B: close    (distance = 1.2)
  A-C: far      (distance = 5.8)
  B-C: medium   (distance = 3.1)
  ...

Convert to probabilities (using Gaussian):
  P(A similar to B) = 0.35  (high - they're close)
  P(A similar to C) = 0.02  (low - they're far)
  ...
```

**Step 1: High-Dimensional Similarities**

```
Gaussian Kernel: 
  Similarity ∝ exp(-distance²/(2σ²))

For point A, compute similarity to all others:
  sim(A→B) = exp(-1.2²/(2σ²)) = 0.68  (after normalization)
  sim(A→C) = exp(-5.8²/(2σ²)) = 0.01
  
Normalize to get probability distribution
```

**Step 2: Choose Variance (σ) Based on Perplexity**

```
Perplexity = 2^(Entropy)

Controls effective number of neighbors:
  Low perplexity (5):  Very local, few neighbors matter
  Med perplexity (30): Balanced approach
  High perplexity (50): More global structure

For each point, binary search to find σ that gives target perplexity
```

**Step 3: Initialize 2D Positions Randomly**

```
Initial Random Placement:
    C
      A    
  E       B
         D
         
(Random positions near origin)
```

**Step 4: Compute Low-D Similarities (Student t-distribution)**

```
Student t-distribution with 1 DOF:
  Q(i,j) = (1 + ||yi - yj||²)⁻¹ / Σ(1 + ||yk - yl||²)⁻¹

This has heavier tails than Gaussian, which helps prevent crowding
```

**Step 5: Optimize via Gradient Descent**

```
Goal: Make Q (low-D probabilities) match P (high-D probabilities)

KL Divergence: 
  KL(P||Q) = Σ P_ij log(P_ij/Q_ij)
  
Gradient tells us how to move each point:
  - Attractive force: Pull together points that should be close (high P_ij)
  - Repulsive force: Push apart points that should be far (low P_ij)

After many iterations:
    C
         
  E   A  B    
         
       D

Clusters emerge! A and B stay close, C moves away, etc.
```

### Why Student t-distribution?

**The Crowding Problem:**

```
High dimensions → Low dimensions
  Many points at distance 5 → Cannot all be at distance 5 in 2D!
  
Solution: Use Student t-distribution
  - Heavier tails than Gaussian
  - Allows moderate distances in high-D to become larger distances in low-D
  - Points can spread out without losing similarity structure
```

**Visual Comparison:**

```
Gaussian (in high-D):
  Quick decay, most probability near center
  |
  |      ___
  |   __/   \__
  |__/         \__
  |________________

Student t (in low-D):
  Slower decay, heavier tails
  |
  |    _____
  |   /     \
  |  /       \____
  |_/____________
  
Heavier tails allow more distance variation in low-D
```

---

## The Mathematical Foundation

### 1. High-Dimensional Similarities (Gaussian)

For each point i, we define conditional probability that i would pick j as neighbor:

```
p(j|i) = exp(-||xi - xj||² / (2σi²)) / Σ(k≠i) exp(-||xi - xk||² / (2σi²))
```

**Components:**
- `||xi - xj||²`: Squared Euclidean distance between points i and j
- `σi`: Bandwidth (variance) for point i, adapted based on perplexity
- Denominator: Normalization to make it a probability distribution

**Example:**
```
Point A at [1, 2, 3], Point B at [1.5, 2.2, 3.1]
Distance² = (1-1.5)² + (2-2.2)² + (3-3.1)² = 0.25 + 0.04 + 0.01 = 0.30

With σA = 1.0:
  p(B|A) = exp(-0.30/(2×1.0²)) / Σk exp(-||xA - xk||²/(2×1.0²))
         = exp(-0.15) / Z
         = 0.86 / Z
         
High probability because B is close to A
```

### 2. Perplexity and Entropy

Perplexity determines the effective number of neighbors:

```
Perplexity(Pi) = 2^(H(Pi))

where H(Pi) = -Σj p(j|i) log₂(p(j|i))  (Shannon entropy)
```

**Interpretation:**
```
Perplexity = 30 means:
  "Consider roughly 30 nearest neighbors for each point"
  
Perplexity controls σi:
  - For dense regions: smaller σ (local focus)
  - For sparse regions: larger σ (look further for neighbors)
```

**Example:**
```
Uniform distribution over k items:
  p(j) = 1/k for all j
  H = -Σ(1/k)×log₂(1/k) = log₂(k)
  Perplexity = 2^(log₂(k)) = k

So perplexity ≈ effective number of neighbors
```

### 3. Symmetrized Probabilities

To make the similarity matrix symmetric:

```
pij = (p(j|i) + p(i|j)) / (2n)
```

where n is the number of points.

**Why symmetrize?**
- Conditional probabilities p(j|i) and p(i|j) may differ
- Joint probabilities are easier to work with
- Ensures consistent similarity relationships

### 4. Low-Dimensional Similarities (Student t)

In the embedded space, we use Student t-distribution with 1 degree of freedom:

```
qij = (1 + ||yi - yj||²)⁻¹ / Σ(k≠l) (1 + ||yk - yl||²)⁻¹
```

**Why Student t?**
- Heavier tails than Gaussian
- Allows dissimilar points to be far apart
- Prevents crowding of moderate distances

**Example:**
```
Points Y1 = [0, 0], Y2 = [2, 1]
Distance² = 2² + 1² = 5

qij = (1 + 5)⁻¹ / Z
    = 0.167 / Z
    
Compare to Gaussian: exp(-5/2) / Z = 0.082 / Z
Student t assigns higher probability (heavier tail)
```

### 5. Cost Function: KL Divergence

The optimization minimizes Kullback-Leibler divergence:

```
KL(P||Q) = Σi Σj pij log(pij/qij)
```

**Interpretation:**
- Measures how different Q is from P
- Asymmetric: KL(P||Q) ≠ KL(Q||P)
- Minimum (0) when P = Q
- Larger values = worse match

**Components:**
```
When pij is large (points should be similar):
  If qij is small (points far apart): large penalty log(pij/qij)
  If qij is large (points close): small penalty
  
When pij is small (points should be dissimilar):
  Contributes little to cost (pij × log(...) ≈ 0)
  
Effect: Focuses on keeping similar points together
```

### 6. Gradient Computation

The gradient with respect to low-dimensional coordinates:

```
∂(KL(P||Q))/∂yi = 4 Σj (pij - qij)(yi - yj)(1 + ||yi - yj||²)⁻¹
```

**Where the 4 and the (1 + d²)⁻¹ come from** (the two "magic" pieces):

Write `d²ij = ||yi - yj||²` and `zij = (1 + d²ij)⁻¹`, so `qij = zij / Z` with
`Z = Σ(k≠l) zkl`. The cost is `C = Σij pij log pij - Σij pij log qij`; only the second
term depends on `Y`.

1. **Chain rule through the distance.** `yi` enters `C` through every `d²ij`, so
   `∂C/∂yi = Σj (∂C/∂d²ij) · (∂d²ij/∂yi)`. The inner derivative is the easy half:
   `∂d²ij/∂yi = 2(yi - yj)`. **That is the first factor of 2.**

2. **Each pair is counted twice.** `yi` appears in the pair `(i, j)` *and* in the pair
   `(j, i)`, and the matrices are symmetric (`pij = pji`, `qij = qji`), so the two
   contributions are identical and add. **That is the second factor of 2.** Two times
   two gives the **4**.

3. **The heavy tail leaves its fingerprint.** Differentiating
   `-Σ pij log(zij / Z)` with respect to `d²ij` gives `-(pij - qij) · (1/zij) · ∂zij/∂d²ij`.
   Since `zij = (1 + d²ij)⁻¹`, we get `∂zij/∂d²ij = -(1 + d²ij)⁻² = -z²ij`, and the ratio
   `z²ij / zij` collapses to exactly `zij = (1 + d²ij)⁻¹`. So
   `∂C/∂d²ij = (pij - qij) · (1 + d²ij)⁻¹`.

Putting the three together reproduces the boxed formula. The surviving
`(1 + d²ij)⁻¹` is the **same Student-t kernel used to build q**: the heavy tail that fixes
crowding in the forward pass also damps the force between far-apart points in the
backward pass, which is why t-SNE does not explode when two points drift far away.

In the code this is one line, and the comment above it repeats the formula verbatim:

```python
gradient = 4 * np.sum((PQ_diff[:, :, np.newaxis] *      # (p_ij - q_ij)
                       Y_diff *                          # (y_i - y_j)
                       inv_distances[:, :, np.newaxis]), # (1 + d^2)^-1
                      axis=1)                            # sum over j
```

**Physical Interpretation:**

```
For each point i, the gradient has two forces:

Attractive Force (pij > qij):
  - Points that should be close but are far
  - Gradient points from yi toward yj
  - Strength ∝ (pij - qij)
  
Repulsive Force (qij > pij):
  - Points that are close but should be far
  - Gradient points from yj away from yi
  - Strength ∝ (qij - pij)
  
The (1 + ||yi - yj||²)⁻¹ term:
  - Moderates force based on current distance
  - Prevents very large updates for distant points
```

**Example:**
```
Point A at [0, 0], Point B at [1, 1]
pAB = 0.3 (should be similar)
qAB = 0.1 (currently far)

Difference: 0.3 - 0.1 = 0.2 (positive → attractive)

Distance² = 2
Force = 4 × 0.2 × ([0,0] - [1,1]) × (1 + 2)⁻¹
      = 0.8 × [-1,-1] × 0.33
      = [-0.27, -0.27]
      
Gradient tells A to move toward B (negative direction toward [1,1])
```

### 7. Optimization with Momentum

To speed up convergence and avoid local minima:

```
Y(t+1) = Y(t) - α × ∂C/∂Y(t) + momentum × (Y(t) - Y(t-1))
```

Note the **minus** sign in front of the gradient: this is gradient *descent*, we move
against the gradient. (The original van der Maaten & Hinton paper prints this update with
a **plus** sign — both where it introduces the momentum term and again in the t-SNE
pseudocode of Algorithm 1, it reads
`Y(t) = Y(t-1) + η ∂C/∂Y + α(t)(Y(t-1) - Y(t-2))` — and it defines no separate
negative-gradient symbol, so as printed it is gradient *ascent*. Do not copy the paper's
sign: van der Maaten's own reference `tsne.py` descends,
`iY = momentum * iY - eta * (gains * dY)`. The code follows the minus-sign convention
above:

```python
Y_velocity = momentum * Y_velocity - self.learning_rate * gradient
Y = Y + Y_velocity
```

which is the same rule written with the velocity accumulated in a variable.)

**Parameters:**
- `α`: Learning rate (how far to move)
- `momentum`: Fraction of previous velocity to keep (0.5 during early exaggeration,
  then 0.8; both phases switch at `early_exaggeration_iter`)

**Effect:**
```
Without momentum:
  - Can get stuck in local minima
  - Oscillates in narrow valleys
  
With momentum:
  - Builds up speed in consistent directions
  - Dampens oscillations
  - Escapes shallow local minima
```

### 8. Early Exaggeration

In initial iterations, multiply P by a factor (typically 12):

```
P_early = early_exaggeration × P
```

**Why?**
```
Creates tight clusters initially:
  - High P values → strong attractive forces
  - Points form dense clusters
  - Easier to separate clusters later
  
Without early exaggeration:
  - Clusters may overlap from start
  - Harder to untangle later
  
Timeline (the switch point is early_exaggeration_iter, default 250):
  Iterations 0-250: P × 12, momentum 0.5 (tight clusters form)
  Iterations 250+:  P × 1,  momentum 0.8 (clusters adjust and spread)
```

Both the exaggeration and the momentum schedule key off the **same**
`early_exaggeration_iter`, so shortening the exaggeration phase shortens the
gentle-momentum phase with it.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, random_state=None, early_exaggeration=12.0,
                 early_exaggeration_iter=250, min_grad_norm=1e-7, verbose=0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        # ... other parameters
```

### Core Methods

1. **`__init__(...)`** - Initialize model
   - Set all hyperparameters
   - Typical defaults work well for most datasets

2. **`_compute_pairwise_distances(X)`** - Distance computation
   - Efficiently compute all pairwise squared Euclidean distances
   - Uses matrix operations for speed

3. **`_compute_joint_probabilities(distances, target_perplexity)`** - High-D similarities
   - For each point, binary search for optimal σ
   - Compute Gaussian similarities
   - Symmetrize the probability matrix

4. **`_compute_low_dim_affinities(Y, return_num=False)`** - Low-D similarities
   - Compute distances in embedded space
   - Apply Student t-distribution
   - Normalize to probabilities
   - With `return_num=True` it also hands back the un-normalized numerator
     `(1 + ||yi - yj||²)⁻¹`, which is exactly what the gradient needs

5. **`_compute_gradient(P, Q, Y, inv_distances=None)`** - Gradient calculation
   - Compute attractive and repulsive forces
   - Return gradient for all points
   - `inv_distances` accepts the numerator from step 4 so the O(n²d) distance
     matrix is built once per iteration rather than twice; leave it `None` and
     the method computes it itself

6. **`_compute_kl_divergence(P, Q)`** - Cost function
   - Measure how well Q matches P
   - Lower is better

7. **`fit_transform(X)`** - Main algorithm
   - Accepts a 2-D array, a 1-D array, or a plain list of lists
   - Validates the request: raises `ValueError` if `perplexity >= n_samples`,
     because the largest entropy reachable with n-1 neighbours is log₂(n-1)
   - Compute high-D probabilities
   - Initialize embedding from a **private** `np.random.RandomState(random_state)`,
     so seeding t-SNE never disturbs your own global NumPy random stream
   - Run gradient descent optimization
   - Return final embedding

8. **`fit(X)`** - Fit interface
   - Calls fit_transform
   - Stores embedding in self.embedding_

### Fitted Attributes

After `fit` or `fit_transform`:

| Attribute | Meaning |
|-----------|---------|
| `embedding_` | The `(n_samples, n_components)` result, same array `fit_transform` returns |
| `kl_divergence_` | KL(P‖Q) recomputed from the **returned** embedding, using the un-exaggerated P |
| `n_iter_` | Iterations actually run (fewer than `n_iter` if the gradient-norm threshold tripped) |

---

## Step-by-Step Example

Let's walk through a complete example of **visualizing handwritten digits**:

### The Data

```python
from sklearn.datasets import load_digits
import numpy as np

# Load digits dataset
digits = load_digits()

# This implementation is the EXACT O(n²) t-SNE - every iteration touches all
# n×n pairs - so all 1797 digits take about 3.2 minutes (190 s, timed end to
# end). Use the first 400 rows, which already contain all ten classes, and it
# runs in 9.6 seconds.
X = digits.data[:400]   # 400 samples, 64 features (8×8 pixel images)
y = digits.target[:400] # Labels (0-9)

# Each sample is an 8×8 grayscale image flattened to 64 features
# Sample: [0, 0, 5, 13, 9, ..., 0, 0] represents pixel intensities
```

### Applying t-SNE

```python
# There is no module named `tsne`. Either run the file directly
# (`python _14_tsne.py`), import it by its real name from this folder, or
# paste the TSNE class from _14_tsne.py above this snippet.
from _14_tsne import TSNE

# Create t-SNE model
tsne = TSNE(
    n_components=2,      # Map to 2D
    perplexity=30,       # Consider ~30 neighbors
    learning_rate=200,   # Step size for optimization
    n_iter=1000,         # 1000 gradient descent iterations
    random_state=42,     # For reproducibility
    verbose=2            # Show progress plus the perplexity actually achieved
)

# Fit and transform
X_embedded = tsne.fit_transform(X)
# Output: (400, 2) - 2D coordinates for each digit
```

**What happens internally** (this is real, captured output - run it and you will see
exactly these numbers, because `random_state=42` seeds a private RNG):

```
[t-SNE] Computing pairwise distances...
[t-SNE] Computing P-values...
[t-SNE] Achieved perplexity: mean=30.000, min=30.000, max=30.000 (requested 30)
[t-SNE] Mean sigma from bandwidth search: 11.4085
[t-SNE] Starting optimization with 1000 iterations...
[t-SNE] Iteration 50/1000, KL divergence: 3.0771, Gradient norm: 0.447133
[t-SNE] Iteration 100/1000, KL divergence: 3.1243, Gradient norm: 0.383796
[t-SNE] Iteration 200/1000, KL divergence: 2.9707, Gradient norm: 0.408276
[t-SNE] Iteration 250/1000, KL divergence: 2.8710, Gradient norm: 0.444089
[t-SNE] Iteration 300/1000, KL divergence: 3.2030, Gradient norm: 0.003457
[t-SNE] Iteration 500/1000, KL divergence: 1.6507, Gradient norm: 0.003205
[t-SNE] Iteration 700/1000, KL divergence: 0.4709, Gradient norm: 0.001904
[t-SNE] Iteration 1000/1000, KL divergence: 0.3584, Gradient norm: 0.000267
[t-SNE] Optimization finished!
[t-SNE] Final KL divergence: 0.3584
```
(abridged - the real run prints a line every 50 iterations)

**Reading the trace - four things worth noticing:**

1. **The perplexity line is a proof, not decoration.** `verbose=2` reports the perplexity
   the bandwidth search actually reached. It reads 30.000 because the target entropy is
   `log₂(30)` and the entropy is measured in bits. If those two used different logarithm
   bases, a request for 30 would quietly become `2^ln(30) = 10.56` and every neighbourhood
   in the map would be three times too small.

2. **KL barely moves - and even rises - for the first 250 iterations.** That is early
   exaggeration doing its job. The optimizer is descending on `12 × P`, but the number
   printed is the KL against the *true* `P`, so the two disagree on purpose. Do not read
   this phase as "not converging".

3. **The gradient norm collapses by two orders of magnitude at iteration 300.** That is
   the instant early exaggeration switches off (iteration 250) and the forces fall back to
   their un-inflated scale. Momentum rises from 0.5 to 0.8 at the same moment.

4. **The real descent happens in the second half**: 2.87 → 1.65 → 0.47 → 0.358. Cutting
   `n_iter` to 500 here would stop at KL 1.65 with the ten digit clusters still tangled.
   "Minimum 250 iterations" is enough to finish exaggeration, not enough to finish the map.

### Visualizing Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE visualization of Handwritten Digits')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

**What you see:**

```
Visual Result:
          
    ⑧ ⑧      ⓪ ⓪
   ⑧ ⑧ ⑧    ⓪ ⓪ ⓪
              
  ⑨ ⑨         ① ①
 ⑨ ⑨ ⑨      ① ① ①
              
   ⑦ ⑦       ② ②
  ⑦ ⑦       ② ② ②
           
    ⑥        ③ ③
   ⑥ ⑥      ③ ③
              
   ⑤ ⑤       ④ ④
  ⑤ ⑤       ④ ④ ④

10 distinct clusters, one for each digit!
```

**Interpretation:**
- **Cluster separation**: Digits 0, 1 are very distinct
- **Overlap regions**: 3 and 5 might be close (similar shapes)
- **Outliers**: Misclassified or ambiguous digits
- **Distance between clusters**: Not meaningful in t-SNE!
  - Only local structure (within clusters) is preserved
  - Inter-cluster distances can be arbitrary

### Analyzing Parameter Effects

**Perplexity Comparison:**

```
perplexities = [5, 30, 50]

Perplexity = 5 (very local):
    ⓪  ①
  ⑧  ②     ③   ④
   ⑨  ⑦  ⑥  ⑤
   
Many small, tight clusters
Over-fragments the data

Perplexity = 30 (balanced):
    ⓪ ⓪    ① ①
   ⑧ ⑧    ② ②
  ⑨ ⑨    ③ ③
   ⑦ ⑦   ④ ④
  ⑥ ⑥   ⑤ ⑤
  
Clear clusters, good separation
BEST CHOICE

Perplexity = 50 (global):
    ⓪ ①
   ⑧ ②
  ⑨ ⑦ ③
   ⑥ ⑤ ④
   
Clusters merge, less separated
More global structure
```

---

## Real-World Applications

### 1. **Image Analysis & Computer Vision**
Visualizing high-dimensional image features:
- Input: Images or CNN features (1000+ dimensions)
- Output: 2D scatter plot revealing image clusters
- Example: Group similar products, detect image duplicates
- **Business Value**: Visual search, quality control, dataset exploration

**Specific Applications:**
```
Fashion E-commerce:
  Extract features from product images
  → Apply t-SNE
  → See which products look similar
  → Improve recommendation system
  
Medical Imaging:
  Tumor scan features (shape, texture, intensity)
  → t-SNE visualization
  → Identify tumor subtypes
  → Aid diagnosis and treatment planning
```

### 2. **Natural Language Processing**
Visualizing word embeddings and document representations:
- Input: Word2Vec, GloVe, BERT embeddings (300+ dimensions)
- Output: Word relationship maps
- Example: "king" - "man" + "woman" ≈ "queen" visualized
- **Business Value**: Understanding semantic relationships, debugging NLP models

**Example:**
```
Word Categories in Embedding Space:

    animals          food
   🐕 🐈 🐎      🍎 🍊 🥖
     🐘             🍕 🍔
     
         countries        sports
        🇺🇸 🇫🇷 🇯🇵      ⚽ 🏀 🎾
         🇬🇧 🇩🇪         ⚾ 🏈

Semantic relationships preserved!
```

### 3. **Single Cell Biology**
Analyzing gene expression data:
- Input: Gene expression profiles (10,000+ genes per cell)
- Output: Cell type clusters
- Example: Identify rare cell populations, disease signatures
- **Business Value**: Drug discovery, disease understanding

**Example:**
```
Cell Types from Expression:

  T-cells          B-cells
   🔴 🔴          🔵 🔵
  🔴 🔴 🔴      🔵 🔵 🔵
  
         Stem cells
          🟢 🟢
         🟢 🟢 🟢
         
    Macrophages    Neurons
     🟡 🟡        🟣 🟣
    🟡 🟡 🟡      🟣 🟣

Different cell types cluster by expression profile
```

### 4. **Recommender Systems**
Understanding user-item relationships:
- Input: User embeddings, item embeddings
- Output: Visual map of preferences
- Example: Netflix - see which movies are similar
- **Business Value**: Better recommendations, content organization

**Applications:**
```
Movie Recommendations:
  
  Action        SciFi
   🎬 🎬        🚀 🚀
  🎬 🎬 🎬    🚀 🚀 🚀
  
       Drama        Comedy
      😢 😢        😂 😂
     😢 😢 😢      😂 😂
     
User A likes action → recommend nearby movies
```

### 5. **Anomaly Detection**
Visualizing normal vs anomalous patterns:
- Input: Transaction features, log features, sensor data
- Output: Outlier detection in 2D
- Example: Fraud detection, equipment failure prediction
- **Business Value**: Early warning systems, security

**Example:**
```
Network Traffic Patterns:

Normal traffic (dense cluster):
  ✓ ✓ ✓ ✓
 ✓ ✓ ✓ ✓ ✓
  ✓ ✓ ✓
  
Anomalies (far from cluster):
        ⚠ 
           ⚠
  ⚠
  
Outliers may indicate attacks or failures
```

### 6. **Neural Network Interpretation**
Understanding what networks learn:
- Input: Activations from hidden layers
- Output: Feature visualization
- Example: See how CNN filters group images
- **Business Value**: Model debugging, interpretability, trust

**Example:**
```
CNN Layer 5 Activations:

  Dog faces    Cat faces
   🐕 🐕        🐈 🐈
  🐕 🐕 🐕    🐈 🐈 🐈
  
      Car fronts    Car sides
       🚗 🚗        🚙 🚙
      🚗 🚗 🚗      🚙 🚙

Network learns meaningful features!
```

### 7. **Customer Segmentation**
Visualizing customer groups:
- Input: Customer features (purchases, demographics, behavior)
- Output: Customer segments
- Example: Identify distinct shopper types
- **Business Value**: Targeted marketing, personalization

**Example:**
```
Customer Segments:

  Budget          Premium
 shoppers        shoppers
   💰 💰          💎 💎
  💰 💰 💰      💎 💎 💎
  
      Seasonal      Loyal
      buyers       customers
       🎁 🎁        ⭐ ⭐
      🎁 🎁 🎁      ⭐ ⭐

Different segments need different strategies
```

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Computing Pairwise Distances

```python
def _compute_pairwise_distances(self, X):
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
    D = np.maximum(D, 0)
    return D
```

**How it works** (worked arithmetic, not runnable code):
```
# Efficient computation of ||xi - xj||²
# Expansion: ||xi - xj||² = ||xi||² + ||xj||² - 2xi·xj

X = [[1, 2],    # Point 1
     [3, 4]]    # Point 2

sum_X = [1² + 2², 3² + 4²] = [5, 25]

# Broadcasting magic:
sum_X[:, newaxis]:     sum_X[newaxis, :]:
  [[5],                [[5, 25]]
   [25]]

# Addition broadcasts to:
  [[5+5,  5+25],
   [25+5, 25+25]]
  
# Subtract 2*dot product:
dot(X, X.T) = [[5,  11],
               [11, 25]]

D = [[10-10,  30-22],    [[0, 8],
     [30-22,  50-50]]  =  [8, 0]]

Distance from point 1 to 2: √8 ≈ 2.83
```

**Why this trick?**
- Avoid explicit loops over all pairs
- Uses optimized BLAS operations
- Same O(n²d) complexity as an explicit double loop, but the BLAS matmul runs it
  orders of magnitude faster

### 2. Computing Joint Probabilities (Perplexity)

```python
def _compute_joint_probabilities(self, distances, target_perplexity):
    n = distances.shape[0]
    P = np.zeros((n, n))

    # Perplexity = 2^H, so the target entropy is log2(perplexity).
    # The entropy below is measured in BITS, so the target must be too.
    target_entropy = np.log2(target_perplexity)

    for i in range(n):
        # Binary search for optimal beta (precision), beta = 1/(2*sigma^2)
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        # Every j EXCEPT i. p(i|i) is not part of the definition, and since
        # distances[i, i] = 0 we would get exp(0) = 1, the largest term in
        # the row - each point would become its own nearest neighbour.
        idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        Di = distances[i, idx]

        for _ in range(50):
            P_i = np.exp(-Di * beta)
            sum_P_i = np.sum(P_i)
            P_i = P_i / sum_P_i

            entropy = -np.sum(P_i * np.log2(np.maximum(P_i, 1e-12)))
            entropy_diff = entropy - target_entropy

            if abs(entropy_diff) < 1e-5:
                break

            # Adjust beta
            if entropy_diff > 0:
                # Too much entropy: sharpen the Gaussian (raise beta)
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                # Too little entropy: widen the Gaussian (lower beta)
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

        P[i, idx] = P_i   # write back to the same off-diagonal slots

    # Symmetrize
    P = (P + P.T) / (2 * n)
    return P
```

> **The single most important line here is `np.log2`.** The loop measures entropy in bits
> with `np.log2`, so the target it is chasing must also be in bits. Writing
> `target_entropy = np.log(target_perplexity)` compiles, runs, produces a perfectly
> plausible-looking embedding - and silently solves for a perplexity of
> `2^ln(p)` instead of `p`. A request for 30 becomes 10.56, for 50 becomes 15.05, for 5
> becomes 3.05. Nothing warns you. Run with `verbose=2` and confirm the printed
> "Achieved perplexity" is the number you asked for.

**Step-by-step example** (worked arithmetic, re-measured and verified):
```
Point i, distances to 4 others: [1.0, 2.0, 3.0, 10.0]
Target perplexity: 3.0
Target entropy: log2(3.0) = 1.585 bits   <- log2, NOT ln (ln 3 = 1.099)

Iteration 1: beta = 1.0
  P_i = exp(-[1, 2, 3, 10] * 1.0) = [0.368, 0.135, 0.050, 0.000]
  Normalize: [0.665, 0.245, 0.090, 0.000]
  Entropy: -sum(p × log2(p)) = 1.202   (perplexity 2.30)
  Too low! Need higher entropy → decrease beta (widen the Gaussian)
  beta_max = 1.0, beta_min still -inf, so halve: beta = 0.5

Iteration 2: beta = 0.5
  P_i = exp(-[1, 2, 3, 10] * 0.5) = [0.607, 0.368, 0.223, 0.007]
  Normalize: [0.504, 0.305, 0.185, 0.006]
  Entropy: 1.513   (perplexity 2.86)
  Still too low → halve again: beta = 0.25

Iteration 3: beta = 0.25
  Normalize: [0.401, 0.313, 0.244, 0.042]
  Entropy: 1.742   (perplexity 3.35)
  OVERSHOT. Now both bounds are finite (beta_min = 0.25, beta_max = 0.5),
  so from here the search bisects instead of halving: beta = 0.375

Iterations 4-8: 0.375 → 0.4375 → 0.40625 → 0.390625 → 0.398438
  Entropies:   1.612 → 1.560  → 1.585   → 1.598    → 1.591

Converges at beta ~= 0.4061  ->  entropy 1.5850 bits  ->  perplexity 3.0000
(sigma = sqrt(1 / (2 * beta)) = 1.11)
```

Notice the shape of the search: the entropy is **monotonically decreasing in beta**
(a sharper Gaussian concentrates probability on fewer neighbours, which is lower
entropy), so plain bisection is guaranteed to converge. The doubling/halving in the
first few steps exists only to *bracket* the answer before a finite `beta_min` and
`beta_max` are both known.

### 3. Computing Low-Dimensional Affinities

```python
def _compute_low_dim_affinities(self, Y, return_num=False):
    distances = self._compute_pairwise_distances(Y)
    num = 1 / (1 + distances)     # the Student-t kernel, un-normalized
    np.fill_diagonal(num, 0)
    sum_Q = np.sum(num)
    Q = num / sum_Q
    Q = np.maximum(Q, 1e-12)

    if return_num:
        return Q, num             # the gradient needs `num` too
    return Q
```

`num` is exactly the `(1 + ||yi - yj||²)⁻¹` matrix that appears in the gradient formula.
Handing it back costs nothing and saves `_compute_gradient` from rebuilding the whole
distance matrix a second time in the same iteration.

**Example:**
```
Y = [[0, 0],     # Point 1
     [1, 1],     # Point 2
     [5, 5]]     # Point 3

Distances²:
  [[0, 2, 50],
   [2, 0, 32],
   [50, 32, 0]]

Student t-distribution:
  Q_unnorm = 1 / (1 + distances²)
           = [[inf, 0.333, 0.020],
              [0.333, inf, 0.031],
              [0.020, 0.031, inf]]
  
  Set diagonal to 0:
           = [[0, 0.333, 0.020],
              [0.333, 0, 0.031],
              [0.020, 0.031, 0]]
  
  Normalize (sum = 0.768):
  Q = [[0, 0.434, 0.026],
       [0.434, 0, 0.040],
       [0.026, 0.040, 0]]

High Q (0.434) for nearby points (1-2)
Low Q (0.026) for distant points (1-3)
```

### 4. Computing Gradient

```python
def _compute_gradient(self, P, Q, Y, inv_distances=None):
    n = Y.shape[0]
    Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

    if inv_distances is None:
        # Only needed when the caller did not already have it
        distances = self._compute_pairwise_distances(Y)
        inv_distances = 1 / (1 + distances)
        np.fill_diagonal(inv_distances, 0)

    PQ_diff = P - Q
    gradient = 4 * np.sum((PQ_diff[:, :, np.newaxis] *      # (p_ij - q_ij)
                           Y_diff *                          # (y_i - y_j)
                           inv_distances[:, :, np.newaxis]), # (1 + d^2)^-1
                          axis=1)                            # sum over j
    return gradient
```

Line for line this is the formula from the Mathematical Foundation:
`∂C/∂yi = 4 Σj (pij - qij)(yi - yj)(1 + ||yi - yj||²)⁻¹`. The `4` is the literal `4`,
the three broadcast factors are the three bracketed terms, and `axis=1` is the `Σj`.

**Example:**
```
3 points in 2D:
Y = [[0, 0],    P = [[0,   0.4, 0.1],    Q = [[0,   0.3, 0.05],
     [1, 0],         [0.4, 0,   0.2],         [0.3, 0,   0.15],
     [0, 1]]         [0.1, 0.2, 0  ]]         [0.05, 0.15, 0   ]]

For point 0:
  PQ_diff[0,:] = [0, 0.1, 0.05]  # Should move closer to 1 and 2
  
  Y_diff[0,:,:] = [[0,0], [0-1,0-0], [0-0,0-1]]
                = [[0,0], [-1,0], [0,-1]]
  
  distances[0,:] = [0, 1, 1]
  inv_dist[0,:] = [0, 0.5, 0.5]
  
  Gradient contribution from point 1:
    4 × 0.1 × [-1,0] × 0.5 = [-0.2, 0]  (pull toward point 1)
  
  Gradient contribution from point 2:
    4 × 0.05 × [0,-1] × 0.5 = [0, -0.1]  (pull toward point 2)
  
  Total gradient[0] = [-0.2, -0.1]
  
Now the actual update the code performs (all three lines):

  1. velocity = momentum × velocity - learning_rate × gradient[0]
     First iteration, velocity starts at [0, 0] and momentum is 0.5.
     With an illustrative learning_rate = 1.0 (see the warning below):
     velocity = 0.5 × [0,0] - 1.0 × [-0.2,-0.1] = [0.2, 0.1]

  2. Y[0] = Y[0] + velocity = [0,0] + [0.2,0.1] = [0.2, 0.1]
     (moves toward points 1 and 2, as the attractive forces asked)

  3. Y = Y - mean(Y, axis=0)
     The whole cloud is recentred on the origin every iteration. KL depends
     only on relative positions, so this drift removal is free.
```

> **Why `learning_rate = 1.0` in that arithmetic and not the default 200?** With
> `learning_rate = 200` this single step would be `200 × [-0.2, -0.1] = [40, 20]` - a
> 40-unit jump from a configuration whose points are 1 unit apart. That is not a bug in
> t-SNE; it is what the *first* steps genuinely look like. The embedding is initialised at
> a scale of `1e-4`, so gradients start enormous relative to the layout and the map
> expands violently before settling. The illustrative small rate above is only so the
> arithmetic stays readable.

### 5. Main Optimization Loop

```python
def fit_transform(self, X):
    # Setup
    X = np.asarray(X, dtype=float)          # accept lists / 1-D input
    if self.perplexity >= n_samples:        # unreachable entropy target
        raise ValueError(...)

    distances = self._compute_pairwise_distances(X)
    P = self._compute_joint_probabilities(distances, self.perplexity)

    rng = np.random.RandomState(self.random_state)   # private, not global
    Y = rng.randn(n_samples, self.n_components) * 1e-4
    Y_velocity = np.zeros_like(Y)

    # Bound before the loop so n_iter=0 returns the initialization
    # instead of raising UnboundLocalError on the n_iter_ line below
    iteration = -1

    # Optimization
    for iteration in range(self.n_iter):
        # Early exaggeration
        if iteration < self.early_exaggeration_iter:
            P_effective = P * self.early_exaggeration
        else:
            P_effective = P

        # Compute Q and gradient (reusing `num` for the gradient)
        Q, num = self._compute_low_dim_affinities(Y, return_num=True)
        gradient = self._compute_gradient(P_effective, Q, Y, inv_distances=num)

        # Check convergence
        if np.linalg.norm(gradient) < self.min_grad_norm:
            break

        # Momentum update - same switch point as early exaggeration
        momentum = 0.5 if iteration < self.early_exaggeration_iter else 0.8
        Y_velocity = momentum * Y_velocity - self.learning_rate * gradient
        Y = Y + Y_velocity

        # Recenter
        Y = Y - np.mean(Y, axis=0)

    # Report the cost of the embedding we are actually returning, against the
    # un-exaggerated P. The Q left in the loop belongs to the previous Y.
    self.embedding_ = Y
    Q = self._compute_low_dim_affinities(Y)
    self.kl_divergence_ = self._compute_kl_divergence(P, Q)
    self.n_iter_ = iteration + 1

    return Y
```

**Optimization trace** (numbers are the real 400-digit run from the Step-by-Step Example):
```
Iteration 0:
  P: High-D similarities (fixed)
  Y: Random [-0.0001, 0.0001] (initialization)
  Q: Almost uniform (random positions)
  Gradient: Large (big mismatch between P and Q)

Iteration 50 (early exaggeration):
  P_effective: P × 12 (exaggerated), momentum 0.5
  Y: Points moving into rough clusters
  Gradient norm: 0.447  KL vs true P: 3.0771

Iteration 250 (exaggeration ends, momentum -> 0.8):
  P_effective: P × 1 (normal)
  Y: Clusters formed, need refinement
  Gradient norm: 0.444  KL vs true P: 2.8710
  (at iteration 300 the gradient norm has already fallen to 0.0035 -
   the forces were 12x inflated before the switch)

Iteration 1000 (final):
  Y: Well-separated clusters
  Q: Close match to P
  Gradient norm: 0.000267 (converged)
  KL divergence: 0.3584
```

---

## Model Evaluation

### Hyperparameter Selection

t-SNE has several important hyperparameters:

#### 1. Perplexity

```
Range: 5 to 50 (typically)

Low Perplexity (5-15):
  ✓ Very local structure
  ✓ Fine-grained clusters
  ✗ May over-fragment
  ✗ Sensitive to noise
  
Medium Perplexity (25-35):
  ✓ Balanced local/global
  ✓ Robust default choice
  ✓ Works for most datasets
  
High Perplexity (40-50):
  ✓ More global structure
  ✓ Broader patterns
  ✗ May merge distinct clusters
  ✗ Slower computation
```

**Rule of thumb:**
```
Perplexity should be less than n_samples / 3

For different dataset sizes:
  100 samples: perplexity = 10-20
  1,000 samples: perplexity = 20-40
  10,000 samples: perplexity = 30-50

When in doubt: Start with 30
```

**Hard limit, enforced by the code:** `perplexity` must be strictly less than
`n_samples`, otherwise `fit_transform` raises `ValueError`. The reason is exact rather
than stylistic: a point has only `n - 1` neighbours to spread probability over, so the
largest achievable entropy is `log₂(n - 1)` bits, i.e. a perplexity of `n - 1`. Asking
for more is asking the binary search to solve an equation with no solution - it would
run its 50 iterations, drive beta toward zero, and hand back a uniform `P` with no
warning. So `perplexity=100` is fine on 400 points and an error on 90.

**The bottom end has no guard, and it needs one more than you would think.** At
`perplexity=1` the target entropy is `log₂(1) = 0` bits, which no point whose two nearest
neighbours are nearly equidistant can ever reach; the search pushes beta up until
`exp(-d²·beta)` underflows to zero for that entire row, and the row of `P` collapses to
zeros. Measured on the Quick Start blobs, `P.sum()` is then `0.88` instead of `1.0`
(18 of 150 rows gone), and nothing warns you. Stay in the documented 5-50 range.

#### 2. Learning Rate

```
Range: 10 to 1000

Low Learning Rate (10-100):
  ✓ Stable optimization
  ✓ Less likely to diverge
  ✗ Slow convergence
  ✗ May get stuck in local minima
  
Medium Learning Rate (150-250):
  ✓ Good balance
  ✓ Reasonable convergence speed
  ✓ Default choice: 200
  
High Learning Rate (500-1000):
  ✓ Fast convergence
  ✗ May overshoot
  ✗ Unstable, points bounce around
  ✗ Worse final quality
```

**Guidelines:**
```
If optimization looks unstable:
  → Decrease learning rate
  
If converging too slowly:
  → Increase learning rate
  
If you see "ball" shape (all points in circle):
  → Learning rate too high OR not enough iterations
```

#### 3. Number of Iterations

```
Minimum: 250 (for early exaggeration)
Typical: 1000
High quality: 2000-5000

Trade-off:
  More iterations → Better convergence, slower
  Fewer iterations → Faster, may not converge
  
Convergence indicators:
  ✓ KL divergence stops decreasing
  ✓ Gradient norm very small
  ✓ Visual appearance stops changing
```

### Quality Metrics

#### 1. KL Divergence

```
KL(P||Q) = Σij Pij log(Pij / Qij)

The ONLY two things KL is good for:

  1. Comparing runs on the SAME data with the SAME perplexity.
     Different random seeds -> pick the run with the lowest KL.

  2. Checking convergence within one run.
     Print it every 50 iterations; when it stops falling, you are done.
```

**What KL is not.** There are no universal "good" and "bad" KL values, and any table that
claims otherwise is wrong. KL scales with the number of points, the intrinsic
dimensionality, and the perplexity. Three measurements from this repository make the point:

| Run | Points | Perplexity | Final KL |
|-----|--------|------------|----------|
| Quick Start blobs | 150 | 5 | 2.0721 |
| Quick Start blobs | 150 | 30 | 0.5203 |
| 400 handwritten digits | 400 | 30 | 0.3584 |

The digits map has the *lowest* KL of the three and is by far the hardest problem. The same
blobs score 2.07 or 0.52 depending only on the perplexity you asked for. **Never read KL as
an accuracy, and never compare it across datasets or across perplexities.** For a number
that does mean something on its own, use neighbour preservation (see below) or a
silhouette score against known labels.

#### 2. Visual Inspection

```
Good t-SNE visualization:
  ✓ Clear cluster separation
  ✓ Clusters have consistent density
  ✓ Similar points (same class) grouped
  ✓ Different classes separated
  
Warning signs:
  ✗ All points in a ball (increase iterations/decrease lr)
  ✗ Severe crowding (try different perplexity)
  ✗ Expected clusters merged (increase perplexity)
  ✗ Over-fragmented (decrease perplexity)
```

#### 3. Cluster Preservation

```
Compare with known labels (if available):

from sklearn.metrics import silhouette_score

# Silhouette score measures cluster quality
score = silhouette_score(X_embedded, y)

Score interpretation:
  0.7-1.0: Strong, well-separated clusters
  0.5-0.7: Reasonable structure
  0.25-0.5: Weak structure, overlapping clusters
  < 0.25:  No meaningful clustering

Note: This only makes sense if data truly has clusters!
```

#### 4. Neighbour Preservation (label-free, NumPy only)

Silhouette needs labels. Neighbour preservation does not, and it asks the only question
t-SNE ever promised to answer: *did local neighbourhoods survive the projection?*

```python
def neighbor_preservation(X_high, Y_low, k=10):
    """Share of each point's k nearest neighbours that survive the projection."""
    def nearest(Z):
        sq = np.sum(np.square(Z), axis=1)
        D = sq[:, np.newaxis] + sq[np.newaxis, :] - 2 * np.dot(Z, Z.T)
        np.fill_diagonal(D, np.inf)          # a point is not its own neighbour
        return np.argsort(D, axis=1)[:, :k]

    high_nn, low_nn = nearest(X_high), nearest(Y_low)
    shared = [len(set(high_nn[i]) & set(low_nn[i])) for i in range(len(X_high))]
    return float(np.mean(shared)) / k
```

This is the same trick as `_compute_pairwise_distances`, applied twice and intersected.
It is essentially a simplified *trustworthiness*, the standard t-SNE quality measure.

**Always report it against a baseline.** On the Quick Start data it returns `0.463`, which
looks poor until you score a *random* 2-D layout of the same points: `0.055`. There is also
a ceiling well below 1.0, because those blobs are isotropic 10-D Gaussians whose internal
neighbour ordering simply cannot fit into two dimensions. A bare `0.463` is
uninterpretable; `0.463 against a floor of 0.055` is a result.

### Common Issues and Solutions

#### Issue 1: Crowding (Dense Ball)

```
Symptom: All points clustered in a tight ball

Causes:
  - Learning rate too high
  - Not enough iterations
  - Perplexity too high
  
Solutions:
  → Reduce learning rate (e.g., 200 → 50)
  → Increase iterations (e.g., 1000 → 2000)
  → Reduce perplexity (e.g., 50 → 30)
```

#### Issue 2: Over-fragmentation

```
Symptom: Too many small clusters, expected groups split

Cause: Perplexity too low

Solution:
  → Increase perplexity (e.g., 5 → 30)
```

#### Issue 3: Merged Clusters

```
Symptom: Distinct groups merged together

Causes:
  - Perplexity too high
  - Not enough iterations
  
Solutions:
  → Decrease perplexity
  → Increase iterations
  → Try different random initialization
```

#### Issue 4: Different Results Each Run

```
Symptom: Results look different every time

Cause: Random initialization

Solutions:
  ✓ Set random_state for reproducibility
  ✓ Run multiple times, choose best (lowest KL divergence)
  ✓ This is normal! t-SNE is stochastic
```

### Best Practices

1. **Preprocessing:**
```python
# Center and normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Or use PCA first (for very high dimensions)
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
X_embedded = tsne.fit_transform(X_reduced)
```

2. **Multiple Runs:**
```python
# Run several times with different random seeds
best_kl = np.inf
best_embedding = None

for seed in range(5):
    tsne = TSNE(random_state=seed, verbose=0)
    embedding = tsne.fit_transform(X)
    
    if tsne.kl_divergence_ < best_kl:
        best_kl = tsne.kl_divergence_
        best_embedding = embedding

# Use best_embedding
```

3. **Parameter Sweep:**
```python
# Try different perplexity values
for perplexity in [10, 20, 30, 40, 50]:
    tsne = TSNE(perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)
    # Visualize and compare
```

---

## Comparing with Other Methods

### t-SNE vs PCA

```
PCA (Principal Component Analysis):
  ✓ Fast (closed-form solution)
  ✓ Interpretable components
  ✓ Preserves global structure
  ✗ Linear, misses non-linear patterns
  ✗ Poor at revealing clusters
  
t-SNE:
  ✓ Reveals non-linear patterns
  ✓ Excellent cluster visualization
  ✓ Preserves local structure
  ✗ Slow (iterative optimization)
  ✗ Components not interpretable
  ✗ Different each run

When to use:
  PCA: Quick exploration, linear relationships
  t-SNE: Cluster visualization, non-linear data
```

### t-SNE vs UMAP

```
UMAP (Uniform Manifold Approximation and Projection):
  ✓ Much faster than t-SNE
  ✓ Preserves more global structure
  ✓ Better for large datasets
  ✗ More hyperparameters
  ✗ Less mature/tested
  
t-SNE:
  ✓ Better local structure preservation
  ✓ More established, well-understood
  ✓ Extensive research on behavior
  ✗ Slower
  ✗ Less global structure

When to use:
  UMAP: Large datasets (>10k), need speed
  t-SNE: Moderate datasets, best cluster separation
```

### t-SNE vs MDS

```
MDS (Multidimensional Scaling):
  ✓ Preserves all pairwise distances
  ✓ Global distance relationships
  ✗ Computationally expensive
  ✗ Less flexible than t-SNE
  
t-SNE:
  ✓ Focuses on important (local) distances
  ✓ Better visual separation
  ✗ Distances between clusters meaningless

When to use:
  MDS: Need accurate distance preservation
  t-SNE: Need cluster visualization
```

---

## Computational Complexity

### Time Complexity

```
Standard Algorithm:
  - Distance computation: O(n² × d)
  - P computation: O(n² × log(perplexity))
  - Each gradient descent iteration: O(n²)
  - Total: O(n² × d + n² × iterations)
  
For n=10,000, d=100, iterations=1000:
  ~10¹⁰ operations ≈ several minutes to hours

Practical limits:
  - Up to ~10,000 points: Feasible
  - 10,000-50,000: Slow but possible
  - >50,000: Need approximations (Barnes-Hut, FFT)
```

### Space Complexity

```
Storage required:
  - Input data: O(n × d)
  - Distance matrix: O(n²)
  - Probability matrices P, Q: O(n²)
  - Embedding: O(n × n_components)
  
Total: O(n² + n×d)

For n=10,000:
  - P matrix: 10,000² × 8 bytes ≈ 800 MB
  - Feasible on modern computers
  
For n=100,000:
  - P matrix: 100,000² × 8 bytes ≈ 80 GB
  - Need sparse approximations
```

### Optimization Strategies

1. **PCA Preprocessing**
```python
# Reduce to ~50 dimensions first
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_reduced)
```

Benefits:
  - Faster distance computation (d drops from e.g. 784 to 50)
  - Removes noise
  - Often improves results

2. **Barnes-Hut Approximation**
```
Not implemented in our version, but:
  - Uses spatial indexing (quadtree/octree)
  - Approximates far-away interactions
  - Reduces complexity to O(n log n)
  - Enables 50,000+ points
```

3. **Mini-batch Approach**
```python
# For very large datasets
# Train on sample, embed rest
n_sample = 5000
sample_idx = np.random.choice(len(X), n_sample, replace=False)
X_sample = X[sample_idx]

tsne = TSNE()
X_sample_embedded = tsne.fit_transform(X_sample)

# Then embed remaining points (out of scope for this implementation)
```

---

## Simplifications vs. Canonical t-SNE

This implementation is faithful to van der Maaten & Hinton (2008) on every point that
defines the *cost function*: the same conditional Gaussians, the same binary search on
`H(P_i) = log₂(perplexity)`, the same symmetrization `(p(j|i) + p(i|j)) / 2n`, the same
Student-t `q_ij`, the same KL cost, the same 4-factor gradient.

The *optimizer schedule*, however, follows **sklearn**, not the paper — worth knowing
before you cite this file:

| schedule constant | this code | vdM & Hinton 2008 | vdM's reference `tsne.py` | sklearn 1.7.2 |
|---|---|---|---|---|
| exaggeration factor | 12 | 4 | 4 | 12 |
| exaggeration ends at iteration | 250 | 50 | 100 | 250 |
| momentum 0.5 → 0.8 at iteration | 250 | 250 | 20 | 250 |

Of the three references, only sklearn ties the two phases to a single constant
(`_EXPLORATION_MAX_ITER = 250` sets both), which is what `early_exaggeration_iter` does
here. The paper flips momentum at 250 but stops exaggerating at 50; `tsne.py` flips at
20 and stops exaggerating at 100. This is a schedule choice, not a change to the model:
the cost being minimized is identical either way.

Verified against `sklearn 1.7.2` on 150 standard normal points in 20-D, the joint `P`
matrix agrees with `sklearn.manifold._utils._binary_search_perplexity` to
`max|ΔP| = 1.0e-08`, and the achieved perplexity is `30.000` for a request of `30.0`.

Four things are deliberately left out. Each is listed with what canonical t-SNE does, why
it is omitted, and what it costs you.

### 1. Adaptive per-parameter gains

**Canonical:** the reference implementation keeps a per-coordinate gain array and nudges
it every step, so coordinates whose gradient keeps flipping sign are slowed down:

```
gains = (gains + 0.2) * ((grad > 0) != (velocity > 0)) +
        (gains * 0.8) * ((grad > 0) == (velocity > 0))
gains = np.maximum(gains, 0.01)          # min_gain
velocity = momentum * velocity - learning_rate * (gains * grad)
```

**Here:** plain momentum, `velocity = momentum * velocity - learning_rate * grad`.

**Why:** it is an optimizer heuristic, not part of the model, and the extra state array
obscures the one line that matters. The repo's rule is clarity over cleverness.

**Consequence:** measured to be none on well-conditioned data. The benchmark, written out
so you can re-run it: `X, y = make_blobs(n_samples=200, centers=4, n_features=10,
cluster_std=1.0, random_state=42)` from `sklearn.datasets`, both solvers at perplexity 30,
learning rate 200, random init, 1000 iterations, `random_state=42`. This implementation
reaches **KL 0.2309 / trustworthiness 0.9718 / silhouette 0.9467**;
`sklearn.manifold.TSNE(method='exact', init='random')` - spelling the iteration count
`max_iter=1000`, because sklearn 1.7.2 no longer accepts `n_iter` - reaches
**KL 0.2363 / 0.9706 / 0.9485**. (Trustworthiness with `n_neighbors=5`, silhouette
of the embedding against the planted labels, and both KLs recomputed from this file's own
`P` so the two are scored identically.) Neither gap is real: over seeds 42, 0 and 7 this
code spans KL 0.2307-0.2362 and sklearn spans 0.2345-0.2384, so the run-to-run spread is
wider than the difference between the solvers. On badly-scaled or very high-dimensional
input, gains would converge in fewer iterations.

### 2. Barnes-Hut / FFT approximation

**Canonical:** modern t-SNE (and sklearn's default `method='barnes_hut'`) approximates the
repulsive forces with a quadtree, reducing each iteration from `O(n²)` to `O(n log n)` and
making 50,000+ points practical.

**Here:** the exact `O(n²)` computation, which is what the mathematics literally says.

**Why:** the quadtree is several hundred lines of tree-building and traversal that teach
data structures rather than t-SNE. This is well past the point where added machinery stops
explaining the algorithm.

**Consequence:** the real one. Every iteration touches all `n²` pairs, and the
`(n, n, n_components)` difference tensor also makes memory grow as `n²`. 9.6 seconds for
400 points × 1000 iterations, and 190 seconds (3.2 minutes) for all 1797 digits - both
complete runs timed end to end on the machine this guide was written on, not
extrapolations from a shorter run. The cost grows as n², so expect your own constant.
Beyond a few thousand points, use `sklearn.manifold.TSNE`. This is why every USAGE EXAMPLE
in `_14_tsne.py` subsamples.

### 3. PCA initialization

**Canonical:** sklearn now defaults to `init='pca'`, which starts the embedding from the
first two principal components. This makes runs reproducible without a seed and preserves
noticeably more global structure.

**Here:** `init` is always random: `rng.randn(n, n_components) * 1e-4`, which is the
scale sklearn uses (`1e-4 * standard_normal(...)`). The paper's Algorithm 1 samples
`Y(0)` from `N(0, 10⁻⁴ I)` — a *variance* of `10⁻⁴`, i.e. a standard deviation of `10⁻²`,
a hundred times wider. Measured on the Quick Start blobs, that hundredfold change moves
the final KL from `0.9288` to `0.8209` and neighbour preservation from `0.463` to
`0.486`: a different random map of the same quality, because early exaggeration blows the
layout up by orders of magnitude within the first few dozen iterations either way.

**Why:** PCA-init would require a full PCA inside a t-SNE file, duplicating algorithm #11.

**Consequence:** results vary with `random_state`, and the *relative arrangement of
clusters* is less stable between runs than sklearn's default would give. Within-cluster
structure is unaffected. Set `random_state` and, for anything important, run 3-5 seeds and
keep the lowest KL (see "Multiple Runs" above).

### 4. No `transform()` for new points

**Canonical:** there is none either. `sklearn.manifold.TSNE` has no `transform` method.
t-SNE optimizes point *positions*, not a mapping function, so there is nothing to apply to
a new sample.

**Consequence:** to embed new data you must refit everything. If you need a reusable
mapping, use PCA, an autoencoder, or parametric t-SNE. This is discussed again under
"Cannot Embed New Points" below.

### What is NOT a simplification

- The distance computation uses the `||x||² + ||y||² - 2x·y` expansion. That is an
  algebraic identity, exact up to floating point, not an approximation.
- `P` is clipped at `1e-12` and the entropy uses `np.log2(np.maximum(P_i, 1e-12))`. Both
  are numerical guards against `log(0)`; they change no result you can measure.
- The gradient-norm early stop (`min_grad_norm`) is an *addition*, not an omission, and it
  comes from sklearn (`min_grad_norm=1e-7`, the same default, checked as
  `if grad_norm <= min_grad_norm: break` inside `_gradient_descent`). Van der Maaten's
  reference `tsne.py` has no `break` at all and always runs the full `max_iter`. At
  `1e-7` the gradient's contribution to the step it cuts short is
  `learning_rate × 1e-7 = 2e-5`, so it cannot change a result you can see; it only saves
  iterations after the embedding has settled.

---

## Advantages and Limitations

### Advantages ✅

1. **Excellent Visualization**
   - Creates beautiful, interpretable plots
   - Reveals cluster structure clearly
   - Non-linear patterns visible

2. **Preserves Local Structure**
   - Similar points stay close together
   - Captures manifold structure
   - Better than PCA for complex data

3. **Flexible**
   - Works with any distance metric
   - No assumptions about data distribution
   - Handles non-linear relationships

4. **Well-Established**
   - Extensive research and validation
   - Well-understood behavior
   - Many successful applications

5. **Unsupervised**
   - No labels needed
   - Exploratory analysis tool
   - Discovers hidden patterns

### Limitations ❌

1. **Computationally Expensive**
   ```
   O(n²) complexity:
     - Slow for large datasets (>10k points)
     - Minutes to hours for moderate datasets
     - Not suitable for real-time applications
   ```

2. **Non-Deterministic**
   ```
   Different runs give different results:
     - Random initialization
     - Non-convex optimization
     - Solution: Set random_state, run multiple times
   ```

3. **Hyperparameter Sensitive**
   ```
   Results depend on parameters:
     - Perplexity significantly affects output
     - Learning rate affects convergence
     - No automatic way to choose
     - Requires experimentation
   ```

4. **Global Structure Not Preserved**
   ```
   Distances between clusters are meaningless:
     - Cannot interpret inter-cluster distances
     - Cannot compare sizes of clusters
     - Only local neighborhoods are meaningful
   ```

5. **Curse of Dimensionality**
   ```
   For very high dimensions:
     - All distances become similar
     - Harder to preserve structure
     - May need PCA preprocessing
   ```

6. **Cannot Embed New Points**
   ```
   No transform() method:
     - Must retrain for new data
     - Expensive for incrementing updates
     - Unlike PCA which has clear projection
   ```

### When to Use t-SNE

**Good Use Cases:**
- ✅ Visualizing high-dimensional data (images, embeddings)
- ✅ Exploring cluster structure
- ✅ Comparing different datasets visually
- ✅ Understanding neural network representations
- ✅ Exploratory data analysis
- ✅ Presentation and communication

**Bad Use Cases:**
- ❌ Feature extraction for downstream tasks → Use PCA or autoencoders
- ❌ Embedding new/unseen data → Use parametric methods
- ❌ Very large datasets (>50k) → Use UMAP or approximations
- ❌ Measuring exact distances → Use MDS
- ❌ Need deterministic results → Use PCA
- ❌ Real-time applications → Use PCA or random projections

---

## Key Concepts to Remember

### 1. **Local vs Global Structure**
t-SNE preserves local structure (nearby points) but not global structure (far points). Cluster positions and distances between clusters are not meaningful.

### 2. **The Perplexity Balance**
Perplexity controls the local/global trade-off:
- Low perplexity: Very local, fine details
- High perplexity: More global, broader patterns
- Default 30 works for most cases

### 3. **Student t-Distribution is Key**
Using Student t in low-D (instead of Gaussian) prevents crowding:
- Heavier tails allow moderate distances to spread out
- Points can separate without losing similarities

### 4. **Optimization is Stochastic**
Results vary each run due to:
- Random initialization
- Non-convex optimization
- Solution: Run multiple times, use random_state

### 5. **Not a Dimension Reduction for ML**
t-SNE is for visualization, not feature extraction:
- Cannot embed new points
- Components not interpretable
- Use PCA/autoencoders for feature extraction

### 6. **Visual Interpretation**

```
What you CAN interpret:
  ✓ Which points cluster together
  ✓ Relative densities within clusters
  ✓ Outliers within clusters
  
What you CANNOT interpret:
  ✗ Distances between clusters
  ✗ Cluster sizes (can be distorted)
  ✗ Orientation/rotation of plot
  ✗ Exact positions (vary each run)
```

---

## Conclusion

t-SNE is a powerful tool for visualizing and exploring high-dimensional data! By understanding:
- How it converts distances to probabilities
- The role of perplexity in balancing local/global structure
- Why Student t-distribution prevents crowding
- How gradient descent optimizes the embedding
- Best practices for hyperparameter selection

You've gained an essential technique for making sense of complex, high-dimensional datasets! 🎨📊

**When to Use t-SNE:**
- ✅ Visualizing complex datasets
- ✅ Exploring cluster structure
- ✅ Understanding embeddings and features
- ✅ Communicating patterns to stakeholders
- ✅ Moderate-sized datasets (<10k points)

**When to Use Something Else:**
- ❌ Need fast results → PCA, random projection
- ❌ Large datasets → UMAP, PCA
- ❌ Feature extraction → PCA, autoencoders
- ❌ Preserve exact distances → MDS
- ❌ Interpretable components → PCA, ICA

**Next Steps:**
- Try t-SNE on your own dataset
- Experiment with different perplexity values
- Compare with PCA to see the difference
- Learn about UMAP for faster alternative
- Study Barnes-Hut t-SNE for large datasets
- Explore parametric t-SNE for new point embedding

Happy visualizing! 💻🎨📈

