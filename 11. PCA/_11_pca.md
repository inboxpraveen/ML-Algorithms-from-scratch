# Principal Component Analysis (PCA) from Scratch: A Comprehensive Guide

Welcome to the world of Principal Component Analysis! 📊 In this comprehensive guide, we'll explore one of the most powerful dimensionality reduction techniques in machine learning. Think of it as finding the "essence" of your data!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is PCA?](#what-is-pca)
3. [How PCA Works](#how-pca-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Choosing Number of Components](#choosing-number-of-components)
11. [Feature Scaling: Critical for PCA](#feature-scaling-critical-for-pca)
12. [Simplifications vs. Canonical PCA](#simplifications-vs-canonical-pca)
13. [Advantages and Limitations](#advantages-and-limitations)
14. [Complete Usage Example](#complete-usage-example)
15. [PCA vs Other Dimensionality Reduction Methods](#pca-vs-other-dimensionality-reduction-methods)
16. [Key Concepts to Remember](#key-concepts-to-remember)
17. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# PCA from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _11_pca.py  (the __main__ block runs a fuller version)
# Or copy the PrincipalComponentAnalysis class from _11_pca.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the PrincipalComponentAnalysis class here (from _11_pca.py) ----
# class PrincipalComponentAnalysis: ...

np.random.seed(42)

# ---- 1. Known-answer test: hide a 2-D plane inside 5-D and find it again ----
basis = np.linalg.qr(np.random.randn(5, 2))[0]        # true 2-D plane
latent = np.random.randn(300, 2) * np.array([3.0, 1.0])
X = latent @ basis.T + 0.1 * np.random.randn(300, 5) + np.array([10., -4., 7., 0., 2.])

# PCA is unsupervised, but it still learns a mean and a basis - so hold rows out.
X_train, X_test = X[:240], X[240:]

pca = PrincipalComponentAnalysis(n_components=2)
Z = pca.fit_transform(X_train)

print("shape %s -> train %s / test %s" % (X.shape, X_train.shape, X_test.shape))
# Slice to n_components_! explained_variance_ratio_ is full length (5 here),
# so summing the whole array would always give 1.0.
ratios = pca.explained_variance_ratio_[:pca.n_components_]
print("explained_variance_ratio_[:2] : [%.4f, %.4f]" % (ratios[0], ratios[1]))
print("variance retained             : %.4f" % np.sum(ratios))

# If PCA found the planted plane, components_ @ basis is orthogonal,
# so both of its singular values are 1.0.
overlap = np.linalg.svd(pca.components_ @ basis, compute_uv=False)
print("planted-subspace overlap      : %.4f, %.4f" % (overlap[0], overlap[1]))
# Train and test should agree closely: the subspace generalizes.
print("train score (-MSE)            : %.5f" % pca.score(X_train))
print("test  score (-MSE)            : %.5f" % pca.score(X_test))
print("noise floor                   : %.5f" % (0.1 ** 2 * 3 / 5))

# ---- 2. Let PCA choose k for you: keep 95% of the variance ----
factors = np.random.randn(400, 3) * np.array([4.0, 3.0, 2.0])   # only 3 real factors
X2 = factors @ np.random.randn(3, 10) + 0.05 * np.random.randn(400, 10)

pca2 = PrincipalComponentAnalysis(n_components=0.95)
pca2.fit_transform(X2)
print("\n10 features built from 3 factors -> PCA kept %d components"
      % pca2.n_components_)
print("variance retained: %.4f"
      % np.sum(pca2.explained_variance_ratio_[:pca2.n_components_]))
cum = np.cumsum(pca2.explained_variance_ratio_)
for i in range(4):
    print("  PC%-2d ratio=%.4f  cumulative=%.4f"
          % (i + 1, pca2.explained_variance_ratio_[i], cum[i]))

# ---- 3. PCA as a denoiser ----
t = np.linspace(0, 10, 200)
X_clean = np.column_stack([np.sin(t), np.cos(t), 2 * np.sin(t),
                           2 * np.cos(t), np.sin(t) + np.cos(t),
                           np.sin(t) - np.cos(t)])
X_noisy = X_clean + np.random.normal(0, 0.1, X_clean.shape)

pca3 = PrincipalComponentAnalysis(n_components=2)
X_denoised = pca3.inverse_transform(pca3.fit_transform(X_noisy))

before = np.mean((X_noisy - X_clean) ** 2)
after = np.mean((X_denoised - X_clean) ** 2)
print("\nMSE(noisy, clean)    = %.6f" % before)
print("MSE(denoised, clean) = %.6f" % after)
print("noise removed        = %.2f%%" % ((1 - after / before) * 100))
```

Expected output:
```
shape (300, 5) -> train (240, 5) / test (60, 5)
explained_variance_ratio_[:2] : [0.9021, 0.0951]
variance retained             : 0.9972
planted-subspace overlap      : 1.0000, 1.0000
train score (-MSE)            : -0.00566
test  score (-MSE)            : -0.00580
noise floor                   : 0.00600

10 features built from 3 factors -> PCA kept 3 components
variance retained: 0.9999
  PC1  ratio=0.7012  cumulative=0.7012
  PC2  ratio=0.1831  cumulative=0.8843
  PC3  ratio=0.1156  cumulative=0.9999
  PC4  ratio=0.0000  cumulative=1.0000

MSE(noisy, clean)    = 0.010764
MSE(denoised, clean) = 0.003650
noise removed        = 66.09%
```

**How to read this output.** Part 1 is a *known-answer test*: we built data that genuinely lies on a 2-D plane, so PCA has a right answer to find. The overlap of `1.0000, 1.0000` says the recovered plane is the planted plane. The train and test scores agree to within 0.0002 -- the subspace was learned from 240 rows and works just as well on 60 rows it never saw -- and both stop at the noise floor (0.00600) rather than at zero, which is exactly right: the noise is the only thing PCA could not explain, so no amount of extra data would drive the error lower. Part 2 shows PCA counting the hidden factors: 10 observed columns, 3 real factors, and `n_components=0.95` discovers 3. Part 3 shows why "throw away the small components" is the same thing as "denoise": the signal is concentrated in 2 directions, while noise is spread evenly over all 6, so dropping 4 directions deletes mostly noise.

---

## What is PCA?

Principal Component Analysis (PCA) is a **dimensionality reduction technique** that transforms high-dimensional data into a lower-dimensional space while preserving as much information (variance) as possible.

**Real-world analogy**: 
Imagine taking a photo of a 3D sculpture. The photo is 2D, but if you position the camera correctly, you can capture most of the important details. PCA does exactly this - it finds the best "angles" (principal components) to view your data!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Unsupervised, Linear Transformation |
| **Purpose** | Dimensionality Reduction, Feature Extraction |
| **Tasks** | Data Compression, Visualization, Noise Reduction |
| **Output** | Lower-dimensional representation of data |
| **Key Benefit** | Reduces complexity while preserving information |

### The Core Idea

```
"Find the directions of maximum variance in your data"
```

PCA works by:
1. **Finding directions** (principal components) where data varies most
2. **Ranking them** by importance (variance explained)
3. **Projecting data** onto the top components
4. **Discarding** less important dimensions

---

## How PCA Works

### The Algorithm in 6 Steps

```
Step 1: Center the data (subtract mean from each feature)
         ↓
Step 2: Compute covariance matrix (how features vary together)
         ↓
Step 3: Calculate eigenvectors & eigenvalues of covariance matrix
         ↓
Step 4: Sort eigenvectors by eigenvalues (descending)
         ↓
Step 5: Select top k eigenvectors (principal components)
         ↓
Step 6: Project data onto principal components
```

### Visual Example: 2D to 1D

```
Original Data (2D):
    y
    4 |     ●
    3 |   ●   ●
    2 | ●   ●
    1 | ●
    0 +--------- x
      0 1 2 3 4

PCA finds direction of maximum spread:

    y
    4 |     ●
    3 |   ● / ●    ← PC1 (diagonal direction)
    2 | ● / ●          Maximum variance here!
    1 | ●
    0 +--------- x
      0 1 2 3 4

After projection to 1D:
    ●-●-●-●-●  ← All points on PC1 line
    (Most information preserved!)
```

### Why Reduce Dimensions?

**Benefits of PCA:**

```
High Dimensions (100 features):
  ❌ Hard to visualize
  ❌ Slow to train models
  ❌ Risk of overfitting
  ❌ Curse of dimensionality
  ❌ Lots of noise

         ↓ PCA ↓

Low Dimensions (10 features):
  ✅ Easy to visualize
  ✅ Fast training
  ✅ Better generalization
  ✅ Reduced complexity
  ✅ Noise filtered out
```

### Variance Preservation

```
Original data: 100 features
PCA with 10 components: 95% variance retained

Information preserved: 95%
Dimensions reduced: 90%
Speed improvement: 10x faster!
```

---

## The Mathematical Foundation

### 1. Data Centering

First, we center the data by subtracting the mean:

```
X_centered = X - mean(X)
```

**Why?** PCA finds directions of variance from the mean. If data isn't centered, the first component might just point toward the mean!

**Example:**
```python
Original data:
X = [[1, 2],
     [2, 4],
     [3, 6]]

Mean: [2, 4]

Centered:
X_centered = [[-1, -2],
              [ 0,  0],
              [ 1,  2]]
```

**Visualization:**
```
Before centering:        After centering:
    y                        y
    6 |   ●                  2 |   ●
    4 | ●                    0 | ●  (mean at origin)
    2 | ●                   -2 | ●
    0 +---- x                0 +---- x
      0 2 4                    -2 0 2
```

### 2. Covariance Matrix

The covariance matrix measures how features vary together:

```
Cov(X) = (X^T × X) / (n - 1)
```

For 2 features:
```
Cov = [ Var(x₁)      Cov(x₁,x₂) ]
      [ Cov(x₂,x₁)   Var(x₂)    ]
```

**Interpretation:**
- **Diagonal elements**: Variance of each feature
- **Off-diagonal elements**: Covariance between features
- **Positive covariance**: Features increase together
- **Negative covariance**: One increases, other decreases

**Example:**
```python
X = [[-1, -2],
     [ 0,  0],
     [ 1,  2]]

Cov = [[ 1.0,  2.0],   # Var(x₁)=1, Cov(x₁,x₂)=2
       [ 2.0,  4.0]]   # Cov(x₂,x₁)=2, Var(x₂)=4
```

**Meaning**: x₂ has more variance (4 vs 1), and they're positively correlated (cov=2)

### 2.5 Why Eigenvectors? The One Derivation That Matters

Every PCA tutorial says "find the directions of maximum variance" and then, without explanation, says "so eigendecompose the covariance matrix." Those two sentences look unrelated. They aren't — and the bridge between them is the entire mathematical content of PCA. Here it is.

**Step A: write down the thing we actually want to maximize.**

Pick a candidate direction, a unit vector **w**. Project the centered data onto it: each sample xᵢ becomes the single number wᵀxᵢ. The variance of those projections is

```
Var(projections) = (1/(n-1)) Σᵢ (wᵀxᵢ)²
                 = (1/(n-1)) Σᵢ wᵀ xᵢ xᵢᵀ w
                 = wᵀ [ (1/(n-1)) Σᵢ xᵢ xᵢᵀ ] w
                 = wᵀ C w                        ← C is the covariance matrix!
```

That is the whole reason the covariance matrix shows up. `wᵀCw` **is** the variance along direction **w**. So PCA's goal becomes a clean optimization problem:

```
maximize    wᵀ C w
subject to  ||w|| = 1        (i.e. wᵀw = 1)
```

The constraint matters: without it you could make the "variance" arbitrarily large by just making **w** longer, which says nothing about direction.

**Step B: solve it with a Lagrange multiplier.**

Form the Lagrangian, with multiplier λ enforcing the unit-length constraint:

```
L(w, λ) = wᵀ C w - λ (wᵀ w - 1)
```

Set the gradient with respect to **w** to zero. Using `∂(wᵀCw)/∂w = 2Cw` (valid because C is symmetric) and `∂(wᵀw)/∂w = 2w`:

```
∂L/∂w = 2 C w - 2 λ w = 0

        =>   C w = λ w
```

**That is the eigenvector equation.** We did not assume it, choose it, or import it — it fell out of "maximize variance subject to unit length." Any direction that is a stationary point of the variance *must* be an eigenvector of C.

**Step C: which eigenvector?**

Left-multiply `Cw = λw` by wᵀ and use wᵀw = 1:

```
wᵀ C w = λ wᵀ w = λ
```

The left side is the variance along **w**. So **the eigenvalue λ is literally the variance captured by its eigenvector**. To maximize variance, take the eigenvector with the largest eigenvalue. That is PC1.

For PC2 you solve the same problem with the extra constraint that **w** be orthogonal to PC1, and the same algebra hands you the eigenvector with the second-largest eigenvalue. And so on down the list. This is why sorting eigenvalues in descending order (Step 4 of the algorithm) is not a convenience — it is the ranking of components by the amount of variance each one explains.

**The three payoffs of this derivation:**

| Result | Where it shows up in the code |
|--------|-------------------------------|
| `Cw = λw` — components are eigenvectors of C | `np.linalg.eigh(covariance_matrix)` |
| λ = variance along that component | `self.explained_variance_ = eigenvalues` |
| Bigger λ = more variance, so sort descending | `idx = eigenvalues.argsort()[::-1]` |

**One more consequence — orthogonality is free.** C is a real symmetric matrix, and a standard theorem of linear algebra (the spectral theorem) says every such matrix has a full set of *real* eigenvalues and *orthonormal* eigenvectors. So the "PCs are perpendicular" property advertised below is not something PCA imposes; it is inherited from the symmetry of the covariance matrix. This is also the concrete reason the implementation calls `np.linalg.eigh` (the symmetric solver) rather than `np.linalg.eig` (the general one) — see [Understanding the Code](#understanding-the-code).

### 3. Eigenvalues and Eigenvectors

We decompose the covariance matrix:

```
Cov × v = λ × v

where:
  v = eigenvector (direction of principal component)
  λ = eigenvalue (variance along that direction)
```

**Intuitive Meaning:**
- **Eigenvector**: Direction in space
- **Eigenvalue**: How much data spreads in that direction

**Example:**
```python
Cov = [[1, 2],
       [2, 4]]

Eigenvalues:  λ₁ = 5.0,  λ₂ = 0.0
Eigenvectors: v₁ = [0.45, 0.89],  v₂ = [-0.89, 0.45]
```

**Visualization:**
```
    y
    4 |     ●
    3 |   ●↗  ●   ← PC1: direction of v₁ (most variance)
    2 | ●↗  ●
    1 | ●          PC2: direction of v₂ (no variance)
    0 +--------- x
      0 1 2 3 4

PC1 (eigenvalue=5.0): Main direction of data spread
PC2 (eigenvalue=0.0): Perpendicular, no spread
```

### 4. Principal Components

Principal components are the eigenvectors sorted by eigenvalues:

```
PC1 = eigenvector with largest eigenvalue  (most important)
PC2 = eigenvector with 2nd largest eigenvalue
...
PCₙ = eigenvector with smallest eigenvalue  (least important)
```

**Properties:**
1. **Orthogonal**: All PCs are perpendicular to each other
2. **Ordered**: PC1 explains most variance, PC2 second-most, etc.
3. **Uncorrelated**: Features in PC space are independent

### 5. Projection Formula

To transform data to principal component space:

```
X_transformed = X_centered × PC^T

where:
  X_centered: centered data (n_samples × n_features)
  PC: principal components matrix (n_components × n_features)
  X_transformed: projected data (n_samples × n_components)
```

**Example:**
```python
X_centered = [[-1, -2],    PC = [[0.45, 0.89]]  (just PC1)
              [ 0,  0],
              [ 1,  2]]

X_transformed = X_centered × PC^T
              = [[-1×0.45 + -2×0.89],   = [[-2.23],
                 [ 0×0.45 +  0×0.89],      [ 0.00],
                 [ 1×0.45 +  2×0.89]]      [ 2.23]]

Reduced from 2D to 1D!
```

### 6. Explained Variance Ratio

How much information each component captures:

```
Explained Variance Ratio = λᵢ / Σλⱼ

where:
  λᵢ = eigenvalue of component i
  Σλⱼ = sum of all eigenvalues
```

**Example:**
```python
Eigenvalues: [5.0, 3.0, 1.0, 0.5]
Total: 9.5

Variance ratios:
  PC1: 5.0/9.5 = 0.526 (52.6%)
  PC2: 3.0/9.5 = 0.316 (31.6%)
  PC3: 1.0/9.5 = 0.105 (10.5%)
  PC4: 0.5/9.5 = 0.053 (5.3%)

Cumulative:
  PC1: 52.6%
  PC1+PC2: 84.2%
  PC1+PC2+PC3: 94.7%  ← Keep 3 components for ~95% variance!
  All: 100%
```

### 7. Reconstruction

To transform back to original space:

```
X_reconstructed = X_transformed × PC + mean

where:
  X_transformed: data in PC space
  PC: principal components
  mean: original data mean
```

**Note**: If we kept all components, reconstruction is perfect. If we dropped some, there's information loss.

### 8. The SVD View: The Same PCA, Computed Differently

Our implementation eigendecomposes the covariance matrix. **Every production PCA — scikit-learn's included — does something that looks completely different: a Singular Value Decomposition of the centered data matrix itself.** They give the same answer. Knowing why is what connects this file to the rest of the world.

**The SVD.** Any matrix factors as

```
X_centered = U Σ Vᵀ

where:
  U: (n_samples × r)   orthonormal columns  (scaled scores)
  Σ: (r × r)           diagonal, entries σ₁ ≥ σ₂ ≥ ... ≥ 0  (singular values)
  V: (n_features × r)  orthonormal columns  (the directions)
  r = min(n_samples, n_features)
```

**Why V is exactly our `components_`.** Substitute the SVD into the covariance formula:

```
C = X_centeredᵀ X_centered / (n-1)
  = (U Σ Vᵀ)ᵀ (U Σ Vᵀ) / (n-1)
  = V Σ Uᵀ U Σ Vᵀ / (n-1)
  = V Σ² Vᵀ / (n-1)               (because UᵀU = I)
  = V [Σ²/(n-1)] Vᵀ
```

That last line *is* an eigendecomposition of C: the eigenvectors are the columns of V, and the eigenvalues are σᵢ²/(n−1). So:

```
components_[i]        = vᵢ            (i-th right singular vector)
explained_variance_[i] = λᵢ = σᵢ²/(n-1)
singular_values_[i]    = σᵢ = sqrt(λᵢ × (n-1))
```

**That last line is exactly what `fit()` computes:**

```python
self.singular_values_ = np.sqrt(eigenvalues[:self.n_components_] * (n_samples - 1))
```

So `singular_values_` is not a leftover from some other algorithm — it is the SVD's σ, reconstructed from our λ. It is the one attribute that lets you check our output against an SVD-based library directly.

**So why does sklearn prefer SVD?** Two reasons, both practical:

1. **Numerical conditioning.** Forming C squares the data's condition number: if X_centered has singular values spanning 10⁸, C's eigenvalues span 10¹⁶, which is past float64's precision. SVD works on X_centered directly and never squares anything.
2. **Cost.** Building C is O(n·d²) and eigendecomposing it is O(d³). For "wide" data — the eigenfaces and 20,000-gene cases mentioned under [Real-World Applications](#real-world-applications), where d ≫ n — that is catastrophic. SVD costs O(n·d·min(n,d)), which for d = 20,000 and n = 200 is thousands of times cheaper.

**Why does this file use the covariance route anyway?** Because "variance → covariance matrix → eigenvectors" is the story that explains *what PCA is*, and section 2.5 derived it from first principles. The SVD route is the same mathematics with the intermediate step optimized away — better engineering, worse teaching. See [Simplifications vs. Canonical PCA](#simplifications-vs-canonical-pca).

### 9. Two Conventions Worth Knowing

**Why `n - 1` and not `n`?** `np.cov` divides by `n - 1` (Bessel's correction), giving the *unbiased* estimate of the population covariance from a sample. Dividing by `n` gives the maximum-likelihood estimate instead. For PCA the choice is almost irrelevant: it scales every eigenvalue by the same constant `n/(n-1)`, so the eigen*vectors* — the principal components — are byte-for-byte identical, and `explained_variance_ratio_` is unchanged because the constant cancels in the ratio. Only the absolute `explained_variance_` values shift, which is why `singular_values_` carries the matching `(n - 1)` factor.

**Component signs are arbitrary.** If **w** is an eigenvector, so is −**w**: both describe the same axis, and both give the same variance. LAPACK may return either one, and the choice can flip between machines or numpy versions. Our implementation therefore adopts scikit-learn's `svd_flip` convention — flip each component so its largest-magnitude entry is positive:

```python
max_abs_rows = np.argmax(np.abs(eigenvectors), axis=0)
signs = np.sign(eigenvectors[max_abs_rows, np.arange(eigenvectors.shape[1])])
eigenvectors = eigenvectors * signs
```

This changes nothing mathematically — it just makes the printed numbers reproducible, and makes `components_` match `sklearn.decomposition.PCA` component-for-component instead of differing by random sign flips. That match is against **scikit-learn 1.5 or newer**: 1.5 is the release where `PCA` began reading each sign off the component itself (`svd_flip(..., u_based_decision=False)`) rather than off the transformed scores. On older scikit-learn the subspace is identical but individual components come back negated about half the time.

**What if n_samples < n_features?** This is the eigenfaces case: 40 photos, 4096 pixels each. The covariance matrix is 4096×4096 but has rank at most 39, so at least 4057 eigenvalues are exactly zero. Two things follow. First, only `min(n_samples - 1, n_features)` components carry any information; asking for more just returns arbitrary directions from the null space. Second, this is precisely the regime where the *symmetric* eigensolver is mandatory: `np.linalg.eig` on such a rank-deficient matrix returns **complex** eigenvalues and eigenvectors (with imaginary parts that are pure floating-point noise), which poisons every downstream computation. `np.linalg.eigh` exploits symmetry and always returns real, orthonormal results. If you write PCA yourself, this is the single easiest way to get silently wrong output.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class PrincipalComponentAnalysis:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None           # Principal components
        self.mean_ = None                 # Data mean
        self.explained_variance_ = None   # Variance per component
        self.explained_variance_ratio_ = None  # Proportion of variance
        self.singular_values_ = None      # sqrt(eigenvalue * (n_samples - 1))
        self.n_features_ = None           # Number of features in original data
        self.n_components_ = None         # Actual number of components kept
        self.noise_variance_ = None       # Mean variance of the DISCARDED components
```

### Fitted Attributes

Everything the model learns. Note carefully which arrays are length `n_components_` and which are length `n_features` — this is the single most common source of wrong numbers when using this class.

| Attribute | Shape / length | Meaning |
|-----------|----------------|---------|
| `components_` | `(n_components_, n_features)` | The principal components (eigenvectors), one per row, sorted by decreasing variance. Also called *loadings*: `components_[i][j]` is how much feature `j` contributes to PCᵢ. |
| `mean_` | `(n_features,)` | Per-feature mean of the training data, subtracted in `transform` and added back in `inverse_transform`. |
| `explained_variance_` | `(n_features,)` — **full length** | The eigenvalues λᵢ: the variance of the data along each component. |
| `explained_variance_ratio_` | `(n_features,)` — **full length** | λᵢ / Σλⱼ: the *fraction* of total variance each component explains. Sums to 1.0 over the whole array. |
| `singular_values_` | `(n_components_,)` | σᵢ = √(λᵢ × (n−1)), the SVD singular values. See [The SVD View](#8-the-svd-view-the-same-pca-computed-differently). |
| `n_components_` | int | How many components were actually kept. Differs from the `n_components` you passed in when you passed a float (e.g. `0.95`) or `None`. |
| `n_features_` | int | Number of features seen during `fit`. |
| `noise_variance_` | float | Mean of the *discarded* eigenvalues, σ² — averaged over the first `min(n_samples, n_features)` only, since eigenvalues past that are structural zeros rather than measurements. That window is scikit-learn's and it is not tight: centering drops the rank to `n_samples − 1`, so when `n_samples ≤ n_features` the last eigenvalue *inside* the window is a structural zero too. Used by `get_covariance()`; exactly `0.0` when all components are kept. |

> **The full-length trap.** `explained_variance_` and `explained_variance_ratio_` deliberately keep **all `n_features` entries**, not just the kept ones, so that you can draw a full scree plot from a fitted model. The consequence is that
> ```python
> sum(pca.explained_variance_ratio_)          # ALWAYS 1.0 - never what you want
> sum(pca.explained_variance_ratio_[:pca.n_components_])   # correct retained variance
> ```
> Always slice. scikit-learn truncates these arrays instead, so this is a deliberate difference — see [Simplifications vs. Canonical PCA](#simplifications-vs-canonical-pca).

### Core Methods

1. **`__init__(n_components)`** - Initialize model
   - n_components: Number of components to keep
   - Can be int (exact number) or float (variance threshold)
   - None = keep all components

2. **`fit(X)`** - Compute principal components
   - Centers the data
   - Computes covariance matrix
   - Finds eigenvectors and eigenvalues (with `np.linalg.eigh`, the *symmetric* solver)
   - Fixes the arbitrary eigenvector signs so output is reproducible
   - Sorts and selects top components
   - Raises `ValueError` on non-2-D input or an invalid `n_components`

3. **`transform(X)`** - Project data to PC space
   - Centers data using training mean
   - Multiplies by principal components
   - Returns lower-dimensional representation, shape `(n_samples, n_components_)`
   - Raises `ValueError` if the model is not fitted yet, or if `X` has the wrong number of features

4. **`fit_transform(X)`** - Convenience method
   - Combines fit() and transform()
   - Returns transformed data directly

5. **`inverse_transform(X_transformed)`** - Reconstruct data
   - Projects back to original space
   - Adds back the mean
   - Returns approximation of original data, shape `(n_samples, n_features)`

6. **`score(X)`** - Evaluate model fit
   - Returns the **negative** mean reconstruction error, `-MSE`
   - **Higher (i.e. closer to zero) is better**; `0.0` means the projection lost nothing
   - Equivalent to `-np.mean((X - pca.inverse_transform(pca.transform(X)))**2)`
   - Note: scikit-learn's `PCA.score` returns an average log-likelihood instead, which is a completely different scale. The two are not comparable — see [Simplifications vs. Canonical PCA](#simplifications-vs-canonical-pca).

7. **`get_covariance()`** - Reconstruct the data covariance from the model
   - Formula: `Cov ≈ Wᵀ diag(λᵢ − σ²) W + σ² I`, where `W = components_`, `λ = explained_variance_`, and `σ² = noise_variance_`
   - Read it as *signal plus noise*: every direction gets a baseline noise variance σ², and the kept components are topped up by the excess variance λᵢ − σ² that makes them stand out
   - When all components are kept, σ² = 0 and this returns the exact sample covariance matrix
   - Matches `sklearn.decomposition.PCA.get_covariance()` to within ~1e-14
   - It does **not** preserve the sample trace when `n_samples < n_features`: σ² is added to all `n_features` directions, but only `min(n_samples, n_features)` of them carry sample variance. On a 30×100 standard-normal matrix (`np.random.RandomState(0)`, `k=5`) the trace is 271.57 against 93.95 for `np.cov` — sklearn's `get_covariance()` returns the same 271.57, so this is a shared convention rather than a divergence

---

## Step-by-Step Example

Let's walk through a complete example with **simple 2D data** to understand every step:

### The Data

```python
import numpy as np

# Original data: 5 samples, 2 features
X = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10]
])

print("Original data shape:", X.shape)  # (5, 2)
```

**Visualization:**
```
    y
   10 |        ●
    8 |      ●
    6 |    ●
    4 |  ●
    2 | ●
    0 +----------- x
      0 1 2 3 4 5

Data lies roughly on a line!
Perfect for PCA to 1D
```

### Step 1: Center the Data

```python
mean = np.mean(X, axis=0)  # [3, 6]
X_centered = X - mean

print("Centered data:")
print(X_centered)
# [[-2, -4],
#  [-1, -2],
#  [ 0,  0],
#  [ 1,  2],
#  [ 2,  4]]
```

**After centering:**
```
    y
    4 |        ●
    2 |      ●
    0 |    ●  (now centered at origin)
   -2 |  ●
   -4 | ●
      +----------- x
     -2 -1 0 1 2
```

### Step 2: Compute Covariance Matrix

```python
cov = np.cov(X_centered.T)

print("Covariance matrix:")
print(cov)
# [[ 2.5,  5.0],
#  [ 5.0, 10.0]]
```

**Interpretation:**
- Var(x) = 2.5: x varies from -2 to 2
- Var(y) = 10.0: y varies from -4 to 4 (more variance!)
- Cov(x,y) = 5.0: strong positive correlation

### Step 3: Compute Eigenvalues & Eigenvectors

```python
# eigh, not eig: the covariance matrix is symmetric, and eigh is the solver
# for symmetric matrices. It guarantees real eigenvalues and orthonormal
# eigenvectors; eig does not.
eigenvalues, eigenvectors = np.linalg.eigh(cov)

print("Eigenvalues:", eigenvalues)
# [ 0.0, 12.5]     ← note the order: ASCENDING, not sorted for us!

print("Eigenvectors:")
print(eigenvectors)
# [[-0.894,  0.447],   ← columns, matching the eigenvalues above
#  [ 0.447,  0.894]]
#    ^ for λ=0.0        ^ for λ=12.5
```

**Analysis:**
- λ = 12.5: one component captures ALL the variance
- λ = 0.0: the other has NO variance
- This makes sense - data is perfectly linear
- **The eigenvalues came back in ascending order.** LAPACK makes no promise about which order you get (`eigh` happens to return ascending; `eig` returns them in no defined order at all). This is exactly why Step 4 exists.

### Step 4: Sort by Eigenvalues, Then Fix the Signs

```python
idx = eigenvalues.argsort()[::-1]  # [1, 0] - the sort really does reorder!
eigenvalues = eigenvalues[idx]     # [12.5, 0.0]
eigenvectors = eigenvectors[:, idx]
# now: [[ 0.447, -0.894],
#       [ 0.894,  0.447]]

# Eigenvector signs are arbitrary (v and -v are the same axis), so we adopt
# sklearn's svd_flip convention: make each column's largest-magnitude entry
# positive. Column 0's biggest entry (0.894) is already positive - keep it.
# Column 1's biggest entry (-0.894) is negative - flip the whole column.
max_abs_rows = np.argmax(np.abs(eigenvectors), axis=0)          # [1, 0]
signs = np.sign(eigenvectors[max_abs_rows, np.arange(2)])       # [1., -1.]
eigenvectors = eigenvectors * signs
# now: [[ 0.447,  0.894],
#       [ 0.894, -0.447]]

explained_variance_ratio = eigenvalues / sum(eigenvalues)
print("Variance explained:", explained_variance_ratio)
# [1.0, 0.0]  ← PC1 explains 100% of variance!
```

### Step 5: Select Principal Components

```python
# Keep just 1 component (reduces 2D → 1D)
n_components = 1
PC = eigenvectors[:, :n_components].T

print("Principal Component:")
print(PC)  # [[0.447, 0.894]]
```

**Visualization:**
```
    y
    4 |        ●
    2 |      ●↗     ← PC1 direction: [0.447, 0.894]
    0 |    ●↗
   -2 |  ●↗
   -4 | ●
      +----------- x
     -2 -1 0 1 2

PC1 points along the data spread!
```

### Step 6: Transform Data

```python
X_transformed = X_centered @ PC.T

print("Transformed data (1D):")
print(X_transformed)
# [[-4.47],
#  [-2.24],
#  [ 0.00],
#  [ 2.24],
#  [ 4.47]]
```

**Result:**
```
Original (2D):          Transformed (1D):
    y                       
    4 |        ●            4.47  ●
    2 |      ●             2.24  ●
    0 |    ●         →     0.00  ●
   -2 |  ●                -2.24  ●
   -4 | ●                 -4.47  ●
      +---- x

Reduced from 2D to 1D!
Information preserved: 100%
```

### Using Our PCA Class

```python
from _11_pca import PrincipalComponentAnalysis

# Create and fit PCA
pca = PrincipalComponentAnalysis(n_components=1)
X_reduced = pca.fit_transform(X)

print("Reduced data shape:", X_reduced.shape)  # (5, 1)
print("Variance explained:", pca.explained_variance_ratio_[0])  # 1.0

# The class reproduces Steps 5 and 6 exactly:
print("components_:", pca.components_)      # [[0.4472136  0.89442719]]
print("X_reduced:", X_reduced.ravel())
# [-4.47213595 -2.23606798  0.          2.23606798  4.47213595]
print("singular_values_:", pca.singular_values_)   # [7.07106781]
# check: sqrt(12.5 * (5-1)) = sqrt(50) = 7.0710678

# Reconstruct
X_reconstructed = pca.inverse_transform(X_reduced)
print("Reconstruction error:", np.mean((X - X_reconstructed)**2))  # 0.0
```

Note the reconstruction error is **exactly** zero, not merely small: the data really did lie on a line, PC1 captured 100% of the variance, and nothing was thrown away.

---

## Real-World Applications

### 1. **Image Compression**
Reduce image file size while preserving quality:
- Input: High-resolution image (millions of pixels)
- Output: Compressed representation
- Example: "Compress 1MB image to 100KB with 95% quality"

### 2. **Data Visualization**
Visualize high-dimensional data in 2D/3D:
- Input: Dataset with 100+ features
- Output: 2D/3D projection for plotting
- Example: "Visualize customer segments in 2D"

### 3. **Noise Reduction**
Remove noise while keeping signal:
- Input: Noisy measurements
- Output: Clean data (keeping top PCs)
- Example: "Clean sensor data by removing noise components"

### 4. **Feature Engineering**
Create better features for ML models:
- Input: Many correlated features
- Output: Fewer, uncorrelated features
- Example: "Transform 50 features → 10 principal components"

### 5. **Face Recognition (Eigenfaces)**
Represent faces compactly:
- Input: Face images (thousands of pixels)
- Output: Compact face representation (50 components)
- Example: "Recognize faces using 50 eigenfaces"

### 6. **Genomics**
Analyze gene expression data:
- Input: Expression levels of 20,000 genes
- Output: Key patterns (principal components)
- Example: "Find main patterns in cancer gene expression"

### 7. **Recommender Systems**
Find latent factors in user preferences:
- Input: User-item ratings matrix
- Output: Latent features (user/item factors)
- Example: "Find movie preference patterns"

### 8. **Finance**
Identify main market factors:
- Input: Returns of 500 stocks
- Output: Main market factors (5-10 components)
- Example: "Find main drivers of stock market"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Centering the Data

```python
self.mean_ = np.mean(X, axis=0)
X_centered = X - self.mean_
```

**What this does:**
```python
# Example
X = [[1, 10],
     [2, 20],
     [3, 30]]

mean = [2, 20]  # average of each column

X_centered = [[-1, -10],  # 1-2=-1, 10-20=-10
              [ 0,   0],  # 2-2=0,  20-20=0
              [ 1,  10]]  # 3-2=1,  30-20=10
```

**Why necessary?** 
- PCA measures variance from the mean
- Without centering, results are wrong!
- The first PC might just point to the data mean

### 2. Computing Covariance

```python
covariance_matrix = np.cov(X_centered.T)
```

**What `np.cov` does:**
```python
# For each pair of features i,j:
cov[i,j] = sum((X[:,i] - mean[i]) * (X[:,j] - mean[j])) / (n-1)

# Result is symmetric matrix:
#     feature1  feature2
# f1 [  var1      cov12  ]
# f2 [  cov21     var2   ]
```

**Interpretation:**
```python
cov = [[2.0, 1.5],
       [1.5, 3.0]]

# Means:
# - Feature 1 has variance 2.0
# - Feature 2 has variance 3.0
# - They're positively correlated (cov=1.5)
```

### 3. Eigendecomposition

```python
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
```

**What this finds:**
```
For covariance matrix C, finds vectors v and values λ such that:
C @ v = λ * v          (λ is a scalar, so this is a scalar multiply)

Results:
- eigenvalues: [λ₁, λ₂, ..., λₙ]  (variance along each PC)
- eigenvectors: [v₁, v₂, ..., vₙ] (direction of each PC, as COLUMNS)
```

This is the equation [section 2.5](#25-why-eigenvectors-the-one-derivation-that-matters) derived from "maximize wᵀCw subject to ‖w‖ = 1". The solver just does the arithmetic.

**Example:**
```python
C = [[2, 1],
     [1, 2]]

eigenvalues = [3, 1]
eigenvectors = [[0.707,  0.707],   # PC1: diagonal direction
                [0.707, -0.707]]   # PC2: other diagonal
```

#### Why `eigh` and not `eig`?

This one-letter difference is the most important correctness detail in the whole file.

| | `np.linalg.eig` | `np.linalg.eigh` |
|---|---|---|
| Assumes | general square matrix | **real symmetric** matrix |
| Eigenvalues | may be complex | always real |
| Eigenvectors | not guaranteed orthogonal | always orthonormal |
| Order returned | undefined | ascending |

A covariance matrix is *always* real and symmetric, so `eigh` is simply the right tool. Using `eig` breaks in two concrete ways:

1. **Complex output on wide data.** When `n_samples <= n_features` the covariance matrix is rank deficient, and `eig`'s general routine returns `complex128` — eigenvalues like `57.635+0.j`, whose imaginary parts are pure round-off. Every downstream array (`components_`, `singular_values_`, and everything `transform()` returns) then becomes complex, and handing that to a classifier fails with `ValueError: Complex data not supported`. This is exactly the eigenfaces regime the [applications](#real-world-applications) section advertises.
2. **Non-orthogonal components.** With repeated or zero eigenvalues, `eig` gives no orthogonality guarantee. On a rank-deficient 60×6 matrix asking for 6 components, `max|components_ @ components_.T − I|` came out as **1.0** — the components were not a basis at all, so the "PCs are orthogonal" property in [Key Concepts](#key-concepts-to-remember) was simply false. With `eigh` the same measurement is **7.8e-16**, i.e. exact to floating-point precision.

Two small guards accompany the solver:

```python
covariance_matrix = np.atleast_2d(np.cov(X_centered.T))  # np.cov on ONE column
                                                         # returns a 0-d scalar
eigenvalues = np.maximum(eigenvalues, 0.0)   # a variance can't be negative;
                                             # -2e-16 is round-off, and would
                                             # make sqrt() below return NaN
```

### 4. Sorting Components

```python
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

**Step by step:**
```python
# Before sorting
eigenvalues = np.array([1.0, 5.0, 3.0])

# argsort gives indices that would sort
idx = eigenvalues.argsort()  # [0, 2, 1]

# [::-1] reverses to get descending order
idx = idx[::-1]  # [1, 2, 0]

# Apply sorting
eigenvalues = eigenvalues[idx]  # [5.0, 3.0, 1.0] ✓
eigenvectors = eigenvectors[:, idx]  # columns reordered
```

**Why this step is not optional:** `eigh` returns eigenvalues in *ascending* order, so without this sort your "first principal component" would be the direction of *least* variance — the exact opposite of what PCA is for.

#### Then: fixing the arbitrary signs

```python
max_abs_rows = np.argmax(np.abs(eigenvectors), axis=0)
signs = np.sign(eigenvectors[max_abs_rows, np.arange(eigenvectors.shape[1])])
signs[signs == 0] = 1.0
eigenvectors = eigenvectors * signs
```

**What this does:** for each component (each *column*), find its largest-magnitude entry and, if that entry is negative, flip the entire column.

**Why:** an eigenvector `v` and its negation `-v` point along the same axis and explain the same variance, so LAPACK is free to hand back either. Without a convention, `components_` and every `transform()` output can flip sign between numpy versions or machines, which makes plots mirror themselves and makes documented example output unreproducible. This is the rule `sklearn.decomposition.PCA` applies from **version 1.5 onward** — `svd_flip(..., u_based_decision=False)`, i.e. the sign is read off the component itself — so adopting it means our `components_` match sklearn's exactly, with no sign mismatches. Measured against scikit-learn 1.7.2: max absolute difference 2.7e-15 on iris `k=2`, 3.9e-15 on iris `k=4`, 9.2e-16 on standardized wine `k=3`, and 1.4e-12 over 300 random datasets (`np.random.RandomState(12345)`, 2009 components in total, zero sign mismatches). Before 1.5, sklearn took the sign from the transformed scores instead; against those versions the subspace is identical but individual components are negated roughly half the time — 993 of those same 2009 components.

**What it does *not* do:** change the mathematics. Variance, `explained_variance_ratio_`, reconstruction error and the spanned subspace are all identical either way.

### 5. Selecting Components

```python
if isinstance(self.n_components, float) and 0 < self.n_components < 1:
    # Keep enough for desired variance
    cumsum = np.cumsum(self.explained_variance_ratio_)
    self.n_components_ = np.argmax(cumsum >= self.n_components) + 1
```

**How this works:**
```python
explained_variance_ratio = np.array([0.5, 0.3, 0.15, 0.05])
cumsum = np.cumsum(explained_variance_ratio)  # [0.5, 0.8, 0.95, 1.0]

# Want 95% variance
n_components = 0.95

# Find first index where cumsum >= 0.95
idx = np.argmax(cumsum >= 0.95)  # returns 2 (0-based index)
n_components_ = idx + 1          # 3 - need 3 components
```

### 6. Projection

```python
X_transformed = np.dot(X_centered, self.components_.T)
```

**Matrix multiplication:**
```python
X_centered: (n_samples × n_features)
components: (n_components × n_features)
components.T: (n_features × n_components)

Result: (n_samples × n_components)

# Example:
X_centered = [[-1, -2, -3],    components = [[0.5, 0.5, 0.7],
              [ 0,  0,  0],                  [0.7, 0.0, 0.7]]
              [ 1,  2,  3]]    

components.T = [[0.5, 0.7],
                [0.5, 0.0],
                [0.7, 0.7]]

X_transformed = X_centered × components.T
              = [[-1×0.5+-2×0.5+-3×0.7, -1×0.7+-2×0.0+-3×0.7],
                 [ 0×0.5+ 0×0.5+ 0×0.7,  0×0.7+ 0×0.0+ 0×0.7],
                 [ 1×0.5+ 2×0.5+ 3×0.7,  1×0.7+ 2×0.0+ 3×0.7]]
              = [[-3.6, -2.8],
                 [ 0.0,  0.0],
                 [ 3.6,  2.8]]

Reduced from 3D to 2D!
```

### 7. Inverse Transform

```python
X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
```

**How it works:**
```python
# Forward: X_centered × PC.T = X_transformed
# Backward: X_transformed × PC = X_centered_reconstructed

X_transformed: (n_samples × n_components)
components: (n_components × n_features)
Result: (n_samples × n_features)

Then add back mean to get back to original scale
```

**Example:**
```python
X_transformed = [[2.0],
                 [0.0],
                 [-2.0]]

components = [[0.7, 0.7]]  # 1 component

X_centered_recon = X_transformed × components
                 = [[2.0×0.7, 2.0×0.7],
                    [0.0×0.7, 0.0×0.7],
                    [-2.0×0.7, -2.0×0.7]]
                 = [[ 1.4,  1.4],
                    [ 0.0,  0.0],
                    [-1.4, -1.4]]

X_reconstructed = X_centered_recon + mean
                = [[ 1.4,  1.4],    + [3, 5]
                   [ 0.0,  0.0],
                   [-1.4, -1.4]]
                = [[4.4, 6.4],
                   [3.0, 5.0],
                   [1.6, 3.6]]
```

---

## Model Evaluation

### 1. Explained Variance

Shows how much information each component captures:

```python
pca = PrincipalComponentAnalysis(n_components=None)
pca.fit(X)

print("Explained variance:")
for i, var in enumerate(pca.explained_variance_):
    print(f"  PC{i+1}: {var:.2f}")

print("\nExplained variance ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
```

**Output example:**
```
Explained variance:
  PC1: 12.50
  PC2: 3.20
  PC3: 0.80
  PC4: 0.20

Explained variance ratio:
  PC1: 0.7485 (74.85%)  ← Most important!
  PC2: 0.1916 (19.16%)
  PC3: 0.0479 (4.79%)
  PC4: 0.0120 (1.20%)   ← Least important
```

(Check the arithmetic yourself: the four eigenvalues sum to 12.50 + 3.20 + 0.80 + 0.20 = 16.70, and 12.50 / 16.70 = 0.7485.)

### 2. Cumulative Variance

Shows total variance retained with k components:

```python
cumulative = np.cumsum(pca.explained_variance_ratio_)

print("Cumulative variance:")
for i, cum in enumerate(cumulative):
    print(f"  First {i+1} components: {cum:.4f} ({cum*100:.2f}%)")
```

**Output:**
```
Cumulative variance:
  First 1 components: 0.7485 (74.85%)
  First 2 components: 0.9401 (94.01%)  ← 2 components for ~94%!
  First 3 components: 0.9880 (98.80%)
  First 4 components: 1.0000 (100.00%)
```

**Decision**: Keep 2 components to retain ~94% of the variance, or 3 to retain almost 99%.

### 3. Reconstruction Error

Measures information loss from dimensionality reduction:

```python
# Original data
X_original = X

# Reduce dimensions
X_reduced = pca.transform(X)

# Reconstruct
X_reconstructed = pca.inverse_transform(X_reduced)

# Calculate error
mse = np.mean((X_original - X_reconstructed) ** 2)
relative_error = mse / np.var(X_original)

print(f"Reconstruction MSE: {mse:.6f}")
print(f"Relative error: {relative_error:.6f}")
```

`score()` is a one-call shortcut for exactly this quantity, negated:

```python
pca.score(X) == -np.mean((X - pca.inverse_transform(pca.transform(X)))**2)
```

so `score()` is **higher-is-better** (0.0 is perfect), while the raw `mse` above is lower-is-better. Do not compare either number to `sklearn.decomposition.PCA.score`, which returns an average log-likelihood on an entirely different scale — see [Simplifications vs. Canonical PCA](#simplifications-vs-canonical-pca).

**Interpretation:**
```
MSE = 0.001: Excellent reconstruction (very little loss)
MSE = 0.1: Good reconstruction (acceptable loss)
MSE = 1.0: Poor reconstruction (significant loss)
```

These thresholds assume standardized features (variance ≈ 1 per column). On unscaled data, compare `mse` to `np.var(X)` instead of to a fixed number — that is what `relative_error` above does.

### 4. Visual Evaluation: Scree Plot

A scree plot shows variance explained by each component:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Plot variance per component
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')

# Plot cumulative variance
plt.subplot(1, 2, 2)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.title('Cumulative Variance')
plt.legend()

plt.tight_layout()
plt.show()
```

**What to look for:**
```
Good scree plot:         Bad scree plot:
    |█                       |█
    |█                       |█
    |▓                       |█
    |▒                       |█
    |░ ░ ░ ░                 |▓
    +-------                 |▒ ░ ░ ░
    ↑                        +-------
"Elbow" here:               No clear elbow
Keep ~3 components          Hard to decide
```

### 5. Component Loadings

Shows how much each original feature contributes to each PC:

```python
# Get loadings
loadings = pca.components_

print("Component loadings:")
print("Feature contributions to each PC:")
for i, pc in enumerate(loadings):
    print(f"\nPC{i+1}:")
    for j, loading in enumerate(pc):
        print(f"  Feature {j+1}: {loading:.4f}")
```

**Interpretation:**
```
PC1: [0.7, 0.7, 0.0, 0.0]
  → PC1 is combination of Features 1 and 2

PC2: [0.0, 0.0, 0.7, -0.7]
  → PC2 is difference between Features 3 and 4
```

---

## Choosing Number of Components

### Method 1: Fixed Number

Use when you know exact dimensionality needed:

```python
# Reduce to 2D for visualization
pca = PrincipalComponentAnalysis(n_components=2)
X_2d = pca.fit_transform(X)
```

**When to use:**
- Visualization (2D or 3D)
- Fixed architecture requirements
- Specific dimensionality reduction goal

### Method 2: Variance Threshold

Keep components until reaching variance threshold:

```python
# Keep 95% of variance
pca = PrincipalComponentAnalysis(n_components=0.95)
X_reduced = pca.fit_transform(X)

print(f"Kept {pca.n_components_} components")
print(f"Variance retained: {sum(pca.explained_variance_ratio_[:pca.n_components_]):.2%}")
```

**Common thresholds:**
- 0.95 (95%): Standard choice, good balance
- 0.99 (99%): High fidelity, minimal loss
- 0.90 (90%): Aggressive reduction, faster
- 0.80 (80%): Very aggressive, major speedup

### Method 3: Elbow Method

Look for "elbow" in scree plot:

```
Variance explained:
    |
0.5 |●               ← PC1: 50%
0.3 |  ●             ← PC2: 30%
0.1 |    ●           ← PC3: 10% (elbow here!)
0.05|      ● ● ●     ← PC4+: small contributions
    +------------
     1  2  3  4  5

Choose 3 components (before elbow flattens)
```

### Method 4: Cross-Validation

Evaluate with downstream task:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

best_score = 0
best_n = 0

for n in [2, 5, 10, 15, 20, 25]:
    pca = PrincipalComponentAnalysis(n_components=n)
    X_reduced = pca.fit_transform(X_train)
    
    clf = LogisticRegression()
    scores = cross_val_score(clf, X_reduced, y_train, cv=5)
    avg_score = np.mean(scores)
    
    print(f"n={n:2d}: {avg_score:.4f}")
    
    if avg_score > best_score:
        best_score = avg_score
        best_n = n

print(f"\nBest n_components: {best_n}")
```

### Decision Framework

```
Use Fixed Number when:
  ✓ Visualization requirement (2D/3D)
  ✓ Hardware constraints
  ✓ Fixed model architecture

Use Variance Threshold when:
  ✓ Want to preserve information
  ✓ Balance speed vs accuracy
  ✓ Standard use case

Use Elbow Method when:
  ✓ Exploratory analysis
  ✓ Want intuitive selection
  ✓ No specific target

Use Cross-Validation when:
  ✓ Have labeled data
  ✓ Downstream task defined
  ✓ Need optimal performance
```

---

## Feature Scaling: Critical for PCA

### Why Scaling Matters

PCA is sensitive to feature scales because it measures variance:

**Without scaling:**
```python
Feature 1: Age (20-80)           → Variance ≈ 400
Feature 2: Income (20k-200k)     → Variance ≈ 3,000,000,000

PCA will be dominated by income!
Age is virtually ignored
```

**With scaling:**
```python
Feature 1: Age (scaled)      → Variance ≈ 1
Feature 2: Income (scaled)   → Variance ≈ 1

Both features contribute fairly!
```

### Standardization (Z-score)

Most common for PCA:

```
x_scaled = (x - mean) / std
```

**Code:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now: mean=0, std=1 for each feature
pca = PrincipalComponentAnalysis(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
```

**After standardization:**
```
All features have:
  - Mean = 0
  - Standard deviation = 1
  - Equal contribution to PCA
```

### Min-Max Scaling

Alternative when you need specific range:

```
x_scaled = (x - min) / (max - min)
```

**Code:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Now: all features in range [0, 1]
pca = PrincipalComponentAnalysis(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
```

### When Scaling is Optional

Don't scale when:
- All features already on same scale
- Features are same units (e.g., all pixels in image)
- Variance differences are meaningful

**Example: Image pixels**
```python
# All pixels are 0-255, same scale
X_images = load_images()  # shape: (n_images, height*width)

# Can apply PCA directly
pca = PrincipalComponentAnalysis(n_components=50)
X_pca = pca.fit_transform(X_images)
```

---

## Simplifications vs. Canonical PCA

The core of this implementation is numerically faithful: on iris (k=2 and k=4), standardized wine (k=3) and random correlated 300×8 data (k=4), it reproduces `sklearn.decomposition.PCA`'s `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, the transformed scores and `get_covariance()` to a maximum absolute difference of **2.3e-13** — floating-point noise — with matching component signs (measured against scikit-learn 1.7.2; the signs match any scikit-learn ≥ 1.5) and identical reconstruction error to 10 decimal places. But four things are deliberately different or deliberately absent. Each is listed here so you are never surprised by a number.

### 1. Covariance + eigendecomposition instead of SVD

**What canonical PCA does.** scikit-learn computes `U, S, Vt = svd(X_centered)` and reads `components_ = Vt`, `explained_variance_ = S**2 / (n-1)`, `singular_values_ = S`.

**What this file does.** Builds `C = X_centeredᵀ X_centered / (n−1)` explicitly and calls `np.linalg.eigh(C)`.

**Why.** As shown in [The SVD View](#8-the-svd-view-the-same-pca-computed-differently), the two are algebraically identical. The covariance route is kept because it makes the derivation in [section 2.5](#25-why-eigenvectors-the-one-derivation-that-matters) visible in the code, and this repository's rule is clarity over performance.

**Practical consequence.** Two, both mild:
- *Conditioning.* Forming `C` squares the condition number. For well-scaled data (which, per [Feature Scaling](#feature-scaling-critical-for-pca), is what you should be feeding PCA anyway) this is invisible. For data spanning many orders of magnitude, the smallest eigenvalues lose precision.
- *Cost.* O(n·d²) to form `C` plus O(d³) to decompose it, versus SVD's O(n·d·min(n,d)). Measured `fit()` times on this machine: 200×5 → 0.001 s; 1000×50 → 0.003 s; 500×400 → 0.04 s; 300×1000 → 0.14 s; 200×2000 → 0.49 s. Note the shape of that growth: the last two rows added no samples but quadrupled `d`, and the time grew ~3.5×, tracking d³ rather than the data size. Fine for teaching-sized data; do not point it at 20,000 genes.

### 2. `explained_variance_` and `explained_variance_ratio_` are full length

**What canonical PCA does.** sklearn truncates both arrays to `n_components_`, so `pca.explained_variance_ratio_.sum()` is the retained variance.

**What this file does.** Keeps all `n_features` entries, so a scree plot can be drawn from any fitted model without refitting with `n_components=None`.

**Practical consequence.** `sum(pca.explained_variance_ratio_)` is **always 1.0** here and tells you nothing. Retained variance is `sum(pca.explained_variance_ratio_[:pca.n_components_])`. Every example in `_11_pca.py` and this document uses the sliced form.

### 3. `score()` returns −MSE, not a log-likelihood

**What canonical PCA does.** `sklearn.decomposition.PCA.score(X)` returns the average Gaussian **log-likelihood** of the samples under the probabilistic-PCA model, using the *same* covariance `get_covariance()` returns — `Wᵀ diag(λᵢ − σ²) W + σ² I`, with the noise floor subtracted from the retained eigenvalues, not left in — and a proper normalizing constant.

**What this file does.** Returns `-np.mean((X - inverse_transform(transform(X)))**2)` — the negative mean reconstruction error.

**Why.** Reconstruction error is what a learner can compute by hand and reason about; the pPCA likelihood needs a matrix determinant and a Gaussian normalizer that add machinery without adding insight.

**Practical consequence.** The two numbers are on completely different scales and **must not be compared**:

| Data | `our score(X)` | `sklearn PCA.score(X)` |
|------|----------------|------------------------|
| iris, k=2 | −0.0253 | −2.6998 |
| iris, k=4 | ≈ −1e−31 | −2.5328 |
| standardized wine, k=3 | −0.3347 | −15.7019 |

Both agree on *direction* — higher is better, and both improve as k grows — but only the magnitudes of our own scores are comparable to each other. If you need the real log-likelihood, compute it separately rather than expecting `score()` to give it.

### 4. Not implemented: whitening, and kernel PCA

**Whitening.** sklearn's `PCA(whiten=True)` divides each transformed component by its standard deviation, so the output has unit variance in every direction:

```
Z_whitened[:, i] = Z[:, i] / sqrt(explained_variance_[i])
```

This is useful when a downstream model (an SVM with an RBF kernel, say) assumes isotropic inputs. It is not implemented here because it is a post-processing step, not part of PCA proper — and you can apply it in one line yourself:

```python
Z = pca.fit_transform(X)
Z_whitened = Z / np.sqrt(pca.explained_variance_[:pca.n_components_])
```

**Kernel PCA.** Named several times in this document as "the non-linear alternative", so here is the one-paragraph version. PCA can only find *linear* subspaces; data curled onto a spiral or a sphere defeats it. Kernel PCA fixes this by imagining a map φ(x) into a much higher-dimensional space where the structure *is* linear, and then running PCA there. The trick is that you never build φ(x): every quantity PCA needs depends only on inner products, so you replace `φ(xᵢ)ᵀφ(xⱼ)` with a kernel function `k(xᵢ, xⱼ)` — commonly the RBF kernel `exp(-γ‖xᵢ - xⱼ‖²)`. Concretely, instead of eigendecomposing the `d × d` covariance matrix you eigendecompose the centered `n × n` kernel matrix `K`, and the projections come from its eigenvectors. The costs: it is O(n²) in memory and O(n³) in time, it has a bandwidth hyperparameter γ to tune, and — because you never form φ — there is no exact `inverse_transform`, so kernel PCA cannot be used for reconstruction or denoising the way ordinary PCA can. Implementing it properly is a separate algorithm, not a flag on this one.

---

## Advantages and Limitations

### Advantages ✅

1. **Dimensionality Reduction**
   - Reduces features while keeping information
   - Makes data manageable
   - Speeds up training

2. **Removes Correlation**
   - Principal components are uncorrelated
   - Better for algorithms sensitive to correlation
   - Cleaner feature space

3. **Noise Reduction**
   - Small components often represent noise
   - Keeping top components filters noise
   - Improves signal-to-noise ratio

4. **Visualization**
   - Reduce to 2D/3D for plotting
   - Understand data structure
   - Identify patterns and clusters

5. **Computational Efficiency**
   - Fewer features = faster algorithms
   - Lower memory requirements
   - Scalable to large datasets

6. **No Labels Needed**
   - Unsupervised method
   - Works without target variable
   - Useful for exploratory analysis

### Limitations ❌

1. **Interpretability Loss**
   - PCs are linear combinations of features
   - Hard to interpret what PCs mean
   - Original features more interpretable

2. **Linear Assumption**
   - Only captures linear relationships
   - Misses non-linear patterns
   - May need kernel PCA for non-linear data ([sketched here](#simplifications-vs-canonical-pca))

3. **Scale Sensitive**
   - MUST scale features appropriately
   - Results change with scaling method
   - Can be misleading if not scaled

4. **Variance ≠ Importance**
   - High variance doesn't always mean important
   - Low variance features might be crucial
   - May lose important information

5. **Outlier Sensitive**
   - Outliers affect mean and variance
   - Can skew principal components
   - May need outlier removal first

6. **Not Sparse**
   - PCs use all features (non-zero coefficients)
   - Cannot remove irrelevant features
   - Consider sparse PCA for feature selection

### When to Use PCA

**Good Use Cases:**
- ✅ Many correlated features
- ✅ Need faster training
- ✅ High-dimensional visualization
- ✅ Remove multicollinearity
- ✅ Compress data
- ✅ Noise reduction

**Bad Use Cases:**
- ❌ Features already independent
- ❌ Need interpretable features
- ❌ Non-linear relationships
- ❌ Very sparse data
- ❌ Few features to begin with
- ❌ Outliers present (clean first)

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load breast cancer dataset (30 features)
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Original dataset: {X.shape}")
print(f"Features: {data.feature_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CRITICAL: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit PCA with all components (for analysis)
pca_full = PrincipalComponentAnalysis(n_components=None)
pca_full.fit(X_train_scaled)

# Analyze variance
print("\n=== Variance Analysis ===")
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
for i, (var, cum) in enumerate(zip(pca_full.explained_variance_ratio_[:10], cumsum[:10])):
    print(f"PC{i+1:2d}: {var:6.4f} ({var*100:5.2f}%)  |  Cumulative: {cum:6.4f} ({cum*100:5.2f}%)")

# Find number of components for 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"\nComponents for 95% variance: {n_components_95}")

# Apply PCA with optimal components
pca = PrincipalComponentAnalysis(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\n=== Dimensionality Reduction ===")
print(f"Original dimensions: {X_train_scaled.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")
print(f"Reduction: {(1 - X_train_pca.shape[1]/X_train_scaled.shape[1])*100:.1f}%")
print(f"Variance retained: {sum(pca.explained_variance_ratio_[:pca.n_components_]):.4f}")

# Train classifier on original data
print("\n=== Model Performance ===")
clf_original = LogisticRegression(max_iter=10000, random_state=42)
clf_original.fit(X_train_scaled, y_train)
y_pred_original = clf_original.predict(X_test_scaled)
acc_original = accuracy_score(y_test, y_pred_original)
print(f"Accuracy (original 30D): {acc_original:.4f}")

# Train classifier on PCA data
clf_pca = LogisticRegression(max_iter=10000, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy (PCA {pca.n_components_}D):   {acc_pca:.4f}")

# Calculate reconstruction error
X_reconstructed = pca.inverse_transform(X_train_pca)
recon_error = np.mean((X_train_scaled - X_reconstructed) ** 2)
print(f"\nReconstruction MSE: {recon_error:.6f}")

# Visualize in 2D
pca_2d = PrincipalComponentAnalysis(n_components=2)
X_train_2d = pca_2d.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 6))
colors = ['red', 'blue']
labels = ['Malignant', 'Benign']

for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y_train == i
    plt.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                c=color, alpha=0.6, label=label, edgecolors='k', s=50)

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Breast Cancer Dataset - PCA Projection (2D)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Slice to :2 - explained_variance_ratio_ holds all 30 ratios, so summing the
# whole array would always print 100.00%.
print(f"\n2D projection retains {sum(pca_2d.explained_variance_ratio_[:2]):.2%} of variance")
```

---

## PCA vs Other Dimensionality Reduction Methods

### PCA vs t-SNE

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| Type | Linear | Non-linear |
| Speed | Fast | Slow |
| Deterministic | Yes | No |
| Global structure | Preserved | Not preserved |
| Local structure | Not emphasized | Preserved |
| Use case | General purpose | Visualization |

### PCA vs LDA

| Aspect | PCA | LDA |
|--------|-----|-----|
| Supervision | Unsupervised | Supervised |
| Goal | Maximum variance | Maximum separation |
| Labels needed | No | Yes |
| Use case | Reduce dimensions | Classification |

### PCA vs Autoencoders

| Aspect | PCA | Autoencoders |
|--------|-----|--------------|
| Complexity | Linear | Non-linear |
| Training | Instant | Time-consuming |
| Interpretability | Better | Worse |
| Flexibility | Limited | High |

(A linear autoencoder with a squared-error loss and a k-unit bottleneck learns *exactly* the PCA subspace — same span, though not necessarily the same rotation inside it. Non-linear activations are what buy autoencoders anything extra.)

### PCA vs Kernel PCA

| Aspect | PCA | Kernel PCA |
|--------|-----|------------|
| Decomposes | `d × d` covariance matrix | `n × n` kernel matrix |
| Structure found | Linear subspaces only | Non-linear manifolds |
| Cost | O(n·d² + d³) | O(n²) memory, O(n³) time |
| Hyperparameters | None (just k) | Kernel choice + bandwidth γ |
| `inverse_transform` | Exact | Only approximate (pre-image problem) |
| Denoising / compression | Yes | Awkward — no exact reconstruction |

See [Simplifications vs. Canonical PCA](#simplifications-vs-canonical-pca) for how the kernel trick works.

---

## Key Concepts to Remember

### 1. **PCA Finds Directions of Maximum Variance**
Not necessarily the most "important" features, but directions where data spreads most.

### 2. **Always Standardize Features**
PCA is extremely sensitive to feature scales. Standardization is almost always necessary.

### 3. **Principal Components are Orthogonal**
All PCs are perpendicular to each other and uncorrelated.

### 4. **Variance Explained is Key**
Use explained variance ratio to decide how many components to keep.

### 5. **PCA is Linear**
Can only capture linear relationships. For non-linear patterns, use kernel PCA or other methods.

### 6. **Information Loss is Inevitable**
Keeping fewer components means losing some information. This is a tradeoff for dimensionality reduction.

---

## Conclusion

Principal Component Analysis is a fundamental technique in data science! By understanding:
- How PCA finds directions of maximum variance
- How eigenvalues and eigenvectors work
- How to choose the right number of components
- How to properly scale features

You've gained a powerful tool for:
- ✅ Reducing dimensionality
- ✅ Visualizing high-dimensional data
- ✅ Removing noise
- ✅ Speeding up ML algorithms
- ✅ Understanding data structure

**When to Use PCA:**
- ✅ Many correlated features
- ✅ Need visualization (2D/3D)
- ✅ Speed up training
- ✅ Remove multicollinearity
- ✅ Compress data

**When to Consider Alternatives:**
- ❌ Need interpretable features → Use feature selection
- ❌ Non-linear relationships → Use kernel PCA or t-SNE
- ❌ Labeled data for classification → Use LDA
- ❌ Very sparse data → Use sparse PCA
- ❌ Complex patterns → Use autoencoders

**Next Steps:**
- Apply PCA to your datasets
- Experiment with different numbers of components
- Combine PCA with classification/regression
- Learn about kernel PCA for non-linear data
- Explore other dimensionality reduction techniques (t-SNE, UMAP)
- Study applications in specific domains (images, genomics, finance)

Happy dimensionality reducing! 📊🎯

