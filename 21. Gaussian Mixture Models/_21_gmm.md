# Gaussian Mixture Models (GMM) from Scratch: A Comprehensive Guide

Welcome to Gaussian Mixture Models! 🚀 In this comprehensive guide, we'll explore GMM - a powerful probabilistic model for soft clustering and density estimation. Think of it as the "sophisticated probabilistic cousin" of K-means clustering!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is GMM?](#what-is-gmm)
3. [How GMM Works](#how-gmm-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Tips and Best Practices](#tips-and-best-practices)
11. [Comparison with Other Algorithms](#comparison-with-other-algorithms)
12. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
13. [Further Reading and Resources](#further-reading-and-resources)
14. [Summary](#summary)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra
dependencies beyond NumPy. It is exactly what `python _21_gmm.py` runs.

```python
# ---------------------------------------------------------------
# Gaussian Mixture Model from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _21_gmm.py  (the __main__ block runs exactly this)
# Or copy the GaussianMixtureModel class from _21_gmm.py and paste below.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the GaussianMixtureModel class here (from _21_gmm.py) ----
# class GaussianMixtureModel: ...

np.random.seed(42)

# --- Build three genuinely elliptical, correlated Gaussian blobs ---
# Elliptical on purpose: this is exactly what separates GMM from k-Means,
# which can only ever draw spherical clusters. The centres are also placed
# close enough that the blobs genuinely overlap, so that some points really
# do belong to two components - otherwise every posterior would be 1.000 and
# the "soft" in soft clustering would never be visible.
mu_list = [np.array([0.0, 0.0]), np.array([3.5, 3.5]), np.array([3.5, 0.0])]
cov_list = [
    np.array([[1.0, 0.7], [0.7, 0.8]]),    # tilted, positively correlated
    np.array([[2.0, -0.9], [-0.9, 1.0]]),  # tilted the other way
    np.array([[0.6, 0.0], [0.0, 1.5]]),    # axis-aligned, tall and thin
]
blobs = [np.random.multivariate_normal(mu_list[k], cov_list[k], 100)
         for k in range(3)]
X = np.vstack(blobs)
y_true = np.array([0] * 100 + [1] * 100 + [2] * 100)

# Shuffle before slicing so train and test cover the same region of space
order = np.random.permutation(len(X))
X, y_true = X[order], y_true[order]
X_train, X_test = X[:225], X[225:]
y_train, y_test = y_true[:225], y_true[225:]

print("=" * 58)
print("DEMO 1 - Soft clustering: 3 elliptical Gaussian blobs")
print("=" * 58)

gmm = GaussianMixtureModel(
    n_components=3,
    covariance_type='full',
    random_state=42,
    max_iter=200
)
gmm.fit(X_train)

train_ll = gmm.score(X_train)
test_ll = gmm.score(X_test)

print(f"Converged : {gmm.converged_} after {gmm.n_iter_} EM iterations")
print(f"Train log-likelihood : {train_ll:10.3f} "
      f"({train_ll / len(X_train):7.3f} per sample)")
print(f"Test  log-likelihood : {test_ll:10.3f} "
      f"({test_ll / len(X_test):7.3f} per sample)")
print("  (score() returns the TOTAL; sklearn returns the per-sample mean)")

print("\nMixing weights pi_k :", np.round(gmm.weights_, 3))
print("Component means mu_k :")
for k, mean in enumerate(gmm.means_):
    print(f"  component {k}: [{mean[0]:6.3f} {mean[1]:6.3f}]")

# The whole point of GMM: some points genuinely belong to two components.
proba = gmm.predict_proba(X_test)
least_sure = np.argsort(proba.max(axis=1))[:5]
print("\n5 least-confident held-out points (this is 'soft' clustering):")
print("       (x, y)         ->  P(comp 0)  P(comp 1)  P(comp 2)  hard")
for i in least_sure:
    print(f"  ({X_test[i, 0]:6.2f}, {X_test[i, 1]:6.2f})  ->  "
          f"{proba[i, 0]:9.3f}  {proba[i, 1]:9.3f}  {proba[i, 2]:9.3f}"
          f"  -> {np.argmax(proba[i])}")

# Cluster purity: cluster labels are arbitrary, so score each cluster by the
# true class that dominates it. 100% means every cluster is pure.
def purity(labels, truth, n_clusters):
    correct = 0
    for k in range(n_clusters):
        members = truth[labels == k]
        if len(members) > 0:
            correct += np.bincount(members).max()
    return correct / len(truth)

print(f"\nTrain cluster purity : {purity(gmm.labels_, y_train, 3):.2%}")
print(f"Test  cluster purity : {purity(gmm.predict(X_test), y_test, 3):.2%}")

# --- Demo 2: how many components? Let BIC decide. ---
print("\n" + "=" * 58)
print("DEMO 2 - Choosing K with BIC / AIC")
print("=" * 58)
print(f"{'K':>3} {'BIC':>12} {'AIC':>12} {'log-lik':>12}")
print("-" * 42)

bic_scores = []
for k in range(1, 6):
    m = GaussianMixtureModel(n_components=k, covariance_type='full',
                             random_state=42, max_iter=200)
    m.fit(X_train)
    bic_scores.append(m.bic(X_train))
    print(f"{k:>3} {m.bic(X_train):>12.2f} {m.aic(X_train):>12.2f} "
          f"{m.score(X_train):>12.2f}")

best_k = int(np.argmin(bic_scores)) + 1
print(f"\nBIC is minimised at K = {best_k} (true K = 3) -> lower BIC is better")

# --- Demo 3: GMM is generative, k-Means is not ---
print("\n" + "=" * 58)
print("DEMO 3 - GMM is generative: sample from the fitted model")
print("=" * 58)

X_gen, comp_gen = gmm.sample(n_samples=300)
shares = np.bincount(comp_gen, minlength=3) / len(comp_gen)

print("Component share of 300 generated points vs the fitted weights:")
print("  generated shares :", np.round(shares, 3))
print("  fitted weights   :", np.round(gmm.weights_, 3))
print("\nFeature statistics, real training data vs generated data:")
print(f"  real  mean {np.round(X_train.mean(axis=0), 2)}   "
      f"std {np.round(X_train.std(axis=0), 2)}")
print(f"  fake  mean {np.round(X_gen.mean(axis=0), 2)}   "
      f"std {np.round(X_gen.std(axis=0), 2)}")
print("\nThe generated cloud matches the real one because GMM learned the "
      "whole density,")
print("not just the cluster centres - that is what k-Means cannot do.")
```

Expected output:
```
==========================================================
DEMO 1 - Soft clustering: 3 elliptical Gaussian blobs
==========================================================
Converged : True after 16 EM iterations
Train log-likelihood :   -803.921 ( -3.573 per sample)
Test  log-likelihood :   -280.655 ( -3.742 per sample)
  (score() returns the TOTAL; sklearn returns the per-sample mean)

Mixing weights pi_k : [0.32  0.307 0.373]
Component means mu_k :
  component 0: [ 3.470  3.412]
  component 1: [ 3.404 -0.422]
  component 2: [ 0.105  0.098]

5 least-confident held-out points (this is 'soft' clustering):
       (x, y)         ->  P(comp 0)  P(comp 1)  P(comp 2)  hard
  (  2.43,   1.99)  ->      0.272      0.325      0.402  -> 2
  (  3.56,   1.84)  ->      0.468      0.528      0.004  -> 1
  (  5.26,   0.77)  ->      0.623      0.377      0.000  -> 0
  (  2.27,   2.45)  ->      0.761      0.092      0.147  -> 0
  (  3.70,   1.43)  ->      0.158      0.842      0.001  -> 1

Train cluster purity : 95.56%
Test  cluster purity : 96.00%

==========================================================
DEMO 2 - Choosing K with BIC / AIC
==========================================================
  K          BIC          AIC      log-lik
------------------------------------------
  1      1881.76      1864.67      -927.34
  2      1797.81      1760.23      -869.12
  3      1699.92      1641.84      -803.92
  4      1722.48      1643.91      -798.96
  5      1749.07      1650.00      -796.00

BIC is minimised at K = 3 (true K = 3) -> lower BIC is better

==========================================================
DEMO 3 - GMM is generative: sample from the fitted model
==========================================================
Component share of 300 generated points vs the fitted weights:
  generated shares : [0.32  0.337 0.343]
  fitted weights   : [0.32  0.307 0.373]

Feature statistics, real training data vs generated data:
  real  mean [2.19 1.  ]   std [1.94 1.93]
  fake  mean [2.34 1.06]   std [1.93 2.01]

The generated cloud matches the real one because GMM learned the whole density,
not just the cluster centres - that is what k-Means cannot do.
```

Three things to notice in that output:

- **The soft assignments are real.** The five least-confident held-out points get
  posteriors like `[0.272, 0.325, 0.402]` - the model is genuinely torn between
  components, and says so. K-means would silently pick one and tell you nothing.
- **BIC finds the right K.** It is minimised at K = 3, which is the number of
  Gaussians the data was actually generated from.
- **The model is generative.** `sample()` draws new points whose mean and standard
  deviation match the training data, because GMM learned the whole density and not
  just three centres.

---

## What is GMM?

Gaussian Mixture Model (GMM) is a **probabilistic model** that assumes data points are generated from a mixture of several Gaussian (normal) distributions. Unlike hard clustering methods like K-means, GMM provides **soft assignments** - each point has a probability of belonging to each cluster.

**Real-world analogy**: 
Imagine you're analyzing customer behavior. Instead of saying "Customer A definitely belongs to Segment 1", GMM says "Customer A has 60% probability of being in Segment 1, 30% in Segment 2, and 10% in Segment 3." This is more realistic because people often exhibit mixed behaviors!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Probabilistic Clustering & Density Estimation |
| **Learning Style** | Unsupervised Learning |
| **Primary Use** | Soft Clustering, Density Estimation, Anomaly Detection |
| **Core Concept** | Data from mixture of Gaussian distributions |
| **Key Innovation** | Soft assignments with probabilities |

### The Core Idea

```
"GMM assumes your data comes from K hidden Gaussian distributions,
 and each point probabilistically belongs to each distribution."
```

**GMM vs K-means:**

| Feature | K-means | GMM |
|---------|---------|-----|
| **Assignment** | Hard (one cluster only) | Soft (probability for each) |
| **Cluster Shape** | Spherical | Elliptical (any shape) |
| **Output** | Cluster labels | Probabilities + labels |
| **Flexibility** | Less flexible | Highly flexible |
| **Use Cases** | Simple clustering | Complex clustering, density estimation, anomaly detection |

### Why Use GMM?

**Advantages:**
- **Soft clustering**: Captures uncertainty in cluster membership
- **Flexible cluster shapes**: Can model elliptical clusters of any orientation
- **Density estimation**: Models the underlying data distribution
- **Anomaly detection**: Identifies low-probability regions
- **Generative**: Can generate new samples from learned distribution
- **Probabilistic**: Provides meaningful uncertainty estimates

**Limitations:**
- **Slower than K-means**: More complex computations
- **Sensitive to initialization**: Can converge to local optima
- **Requires choosing K**: Number of components must be specified
- **Assumes Gaussian**: May not fit non-Gaussian data well
- **More parameters**: Needs more data to estimate reliably

---

## How GMM Works

### The Big Picture

GMM works in two main phases using the **Expectation-Maximization (EM) algorithm**:

```
1. INITIALIZATION
   ├─ Place K Gaussian components over the data
   ├─ Each has: mean (μ), covariance (Σ), and weight (π)
   └─ K-means++ style means, covariances seeded from cov(X)

2. EM ITERATIONS (repeat until convergence)
   │
   ├─ E-STEP (Expectation)
   │  └─ For each point, compute probability of belonging to each component
   │     "Which Gaussian likely generated this point?"
   │
   └─ M-STEP (Maximization)
      └─ Update parameters to maximize likelihood
         "What parameters best explain these assignments?"
```

### The EM Algorithm in Detail

**Initialization:**
```
For K components:
- Weights (π): Start with equal weights (1/K each)
- Means (μ): Use K-means++ to select diverse centers
  (first centre uniform, each later centre drawn with probability
   proportional to its squared distance to the nearest chosen centre)
- Covariances (Σ): Start with the data's own scatter, cov(X), for every
  component - NOT the identity matrix. See "Why not the identity matrix?" below.
```

**Why not the identity matrix?**

Seeding every Σ_k with `I` says "each component starts one unit wide in every
direction". That is only sensible if your features happen to be standardized.
Give the model a feature measured in dollars with a standard deviation of 500 and
those starting Gaussians are ~500x too narrow: on the very first E-step most
points' responsibilities saturate to 0 or 1, EM behaves like hard k-means, and it
settles into a bad local optimum before it takes a single meaningful step.

Seeding with `cov(X)` makes the starting Gaussians as wide as the data, which is
what scikit-learn achieves by deriving its initial covariances from a k-means
responsibility pass. Measured on the customer-segmentation data of USAGE EXAMPLE
3 (four features, the last one on a ~5000 scale), changing *only* this line:

| Initial Σ_k | Total log-likelihood | Recovered cluster sizes (true segments: 100 / 150 / 100) |
|---|---|---|
| `I` (identity) | -6499.62 | 250 / 26 / 74 - two segments merged, the third split in half |
| `cov(X)` | **-6150.60** | **150 / 100 / 100 - every segment recovered exactly** |
| scikit-learn `GaussianMixture` | -6150.60 | 100 / 100 / 150 - every segment recovered exactly |

(Cluster numbering is arbitrary, so only the multiset of sizes is comparable.)

Standardizing the features first fixes the identity version too (see
[Tips: Feature Preprocessing](#5-feature-preprocessing)) - but a good default
should not require the user to know that.

**E-Step (Expectation): Calculate Responsibilities**
```
For each sample x_n and component k:

γ(z_nk) = P(component k | sample n)
        = (π_k × N(x_n | μ_k, Σ_k)) / Σ_j(π_j × N(x_n | μ_j, Σ_j))

Where:
- γ(z_nk) is the "responsibility" (probability that k generated n)
- π_k is the weight of component k
- N(x_n | μ_k, Σ_k) is the Gaussian probability density

Example:
Sample #5 responsibilities: [0.65, 0.30, 0.05]
→ 65% likely from component 0
→ 30% likely from component 1
→ 5% likely from component 2
```

**M-Step (Maximization): Update Parameters**
```
Update each component k:

1. Effective number of points assigned to k:
   N_k = Σ_n γ(z_nk)

2. Update weight:
   π_k = N_k / N

3. Update mean:
   μ_k = (Σ_n γ(z_nk) × x_n) / N_k

4. Update covariance:
   Σ_k = (Σ_n γ(z_nk) × (x_n - μ_k)(x_n - μ_k)^T) / N_k
```

**Convergence:**
```
After each iteration, compute log-likelihood:

log L = Σ_n log(Σ_k π_k × N(x_n | μ_k, Σ_k))

Stop when:
- Change in log-likelihood < tolerance
- OR reached max iterations

Typical convergence: 5-30 iterations on well-scaled data
(measured on this implementation: 5 for USAGE EXAMPLE 8, 11 for USAGE EXAMPLE 1,
 16 for the Quick Start demo, 20 for the Step-by-Step Example below, and 8 for
 the EXAMPLE 3 data once standardized - but 24 for that same data left
 unstandardized)
More iterations are needed when features are on very different scales.

Note: score() and the tol check use the TOTAL log-likelihood summed over all
samples, not the per-sample mean that scikit-learn reports.
```

### Visual Example

```
Imagine data from 3 overlapping groups:

Initial State (iteration 0):
  🔴  🔵  🟢
  Random initialization

After 5 iterations:
  🔴🔵  🔵  🟢🔵
  Components start separating

After 20 iterations (converged):
  🔴🔴🔴  🔵🔵🔵  🟢🟢🟢
  Clear separation with soft boundaries

Key point: Points near boundaries belong to multiple clusters!
Point at boundary: 40% red, 35% blue, 25% green
```

---

## The Mathematical Foundation

### 1. The Gaussian Distribution

**Single Gaussian (Normal Distribution):**

```
N(x | μ, Σ) = (1 / √((2π)^d |Σ|)) × exp(-0.5 × (x-μ)^T Σ^(-1) (x-μ))

Where:
- x: data point (d-dimensional)
- μ: mean vector (center)
- Σ: covariance matrix (shape/orientation)
- |Σ|: determinant of covariance
- d: number of dimensions
```

**Components:**
- **Mean (μ)**: Center of the distribution
- **Covariance (Σ)**: Shape and orientation
  - Diagonal: Independent features
  - Off-diagonal: Feature correlations

### 2. Mixture Model

**Multiple Gaussians combined:**

```
P(x) = Σ_{k=1}^K π_k × N(x | μ_k, Σ_k)

Where:
- K: number of components
- π_k: mixing coefficient (weight) for component k
- Σ π_k = 1 (weights sum to 1)
- 0 ≤ π_k ≤ 1 (valid probabilities)
```

**Interpretation:**
```
"The probability of observing x is the weighted sum of 
 probabilities from each Gaussian component"

Example with 3 components:
P(x) = 0.5 × N(x|μ_1,Σ_1) + 0.3 × N(x|μ_2,Σ_2) + 0.2 × N(x|μ_3,Σ_3)
       ↑                    ↑                     ↑
    Component 1        Component 2          Component 3
    (50% of data)      (30% of data)        (20% of data)
```

### 3. The EM Algorithm

**E-Step: Posterior Probability (Bayes' Theorem)**

```
γ(z_nk) = P(k | x_n) = P(x_n | k) × P(k) / P(x_n)
                     = π_k × N(x_n | μ_k, Σ_k) / Σ_j π_j × N(x_n | μ_j, Σ_j)

Example calculation for sample x_n:
- P(x_n | k=1) = 0.8, π_1 = 0.5 → numerator = 0.40
- P(x_n | k=2) = 0.3, π_2 = 0.3 → numerator = 0.09
- P(x_n | k=3) = 0.1, π_3 = 0.2 → numerator = 0.02
- Sum = 0.51

γ(z_n1) = 0.40 / 0.51 = 0.784
γ(z_n2) = 0.09 / 0.51 = 0.176
γ(z_n3) = 0.02 / 0.51 = 0.039
```

**M-Step: Maximum Likelihood Estimates**

```
Given responsibilities γ(z_nk), update parameters:

1. Mixing coefficients (weights):
   π_k = N_k / N = (Σ_n γ(z_nk)) / N

2. Means:
   μ_k = (Σ_n γ(z_nk) × x_n) / N_k
       = weighted average of points, weighted by responsibility

3. Covariances:
   Σ_k = (Σ_n γ(z_nk) × (x_n - μ_k)(x_n - μ_k)^T) / N_k
       = weighted covariance of points

Where N_k = Σ_n γ(z_nk) is the effective number of points in component k
```

**Log-Likelihood (Objective Function):**

```
log L(θ) = Σ_{n=1}^N log(Σ_{k=1}^K π_k × N(x_n | μ_k, Σ_k))

This is what we're maximizing!

Each iteration of EM is guaranteed to:
- Increase log-likelihood (or keep it the same)
- Eventually converge to a local maximum
```

### 4. Covariance Types

**Full Covariance:**
```
Σ_k = [σ²_11  σ²_12]  (2x2 example)
      [σ²_21  σ²_22]

- Most flexible
- Can model any elliptical shape and orientation
- K × d × (d+1) / 2 parameters
  (a covariance matrix is symmetric, so only the upper triangle is free)
```

**Diagonal Covariance:**
```
Σ_k = [σ²_1   0  ]
      [0    σ²_2]

- Axis-aligned ellipses
- Features independent within cluster
- K × d parameters
```

**Spherical Covariance:**
```
Σ_k = σ² × I = [σ²   0 ]
               [0   σ²]

- Circular/spherical clusters
- Same variance in all directions
- K parameters
```

**Tied Covariance:**
```
All components share same Σ:
Σ = Σ_1 = Σ_2 = ... = Σ_K

- Reduces parameters
- All clusters have same shape, different locations
- d × (d+1) / 2 parameters (symmetric, and shared by every component)
```

### 5. Model Selection

**Bayesian Information Criterion (BIC):**

```
BIC = -2 × log L + n_parameters × log(N)
      ↑                ↑
   Goodness of fit   Penalty for complexity

Lower BIC = Better model

Number of parameters:
- Means: K × d
- Weights: K - 1 (sum to 1 constraint)
- Covariances: 
  * Full: K × d × (d+1) / 2
  * Diag: K × d
  * Spherical: K
  * Tied: d × (d+1) / 2
```

**Akaike Information Criterion (AIC):**

```
AIC = -2 × log L + 2 × n_parameters

Lower AIC = Better model

AIC penalizes complexity less than BIC
→ AIC often selects more complex models
→ BIC preferred when N is large
```

---

## Implementation Details

### Key Steps in Our Implementation

**1. Initialization**
```python
def _initialize_parameters(self, X):
    # K-means++ style initialization
    # - Select diverse initial means (D^2 sampling)
    # - Equal initial weights (1/K each)
    # - Covariances seeded from np.cov(X), so the starting Gaussians
    #   are as wide as the data instead of one unit wide
```

**2. E-Step**
```python
def _e_step(self, X):
    # Compute log P(x_n | k) for numerical stability
    # Add log weights: log P(x_n | k) + log π_k
    # Normalize to get responsibilities: γ(z_nk)
    # Return responsibilities and log-likelihood
```

**3. M-Step**
```python
def _m_step(self, X, responsibilities):
    # Calculate N_k for each component
    # Update weights: π_k = N_k / N
    # Update means: μ_k = weighted average
    # Update covariances: Σ_k = weighted covariance
```

**4. Numerical Stability Tricks**

```python
# 1. Log-sum-exp for numerical stability
def log_sum_exp(arr):
    max_val = max(arr)
    return max_val + log(sum(exp(arr - max_val)))

# 2. Regularization for covariance
Σ_k = Σ_k + ε × I  (ε = reg_covar = 1e-6)

# 3. Avoid zero probabilities
N_k = sum(γ) + 10 × eps

# 4. Cholesky factorisation instead of an explicit matrix inverse
P = L^-T   where   Σ = L L^T
```

**Why the Cholesky factor?**

The Gaussian log-density needs two things from Σ: the Mahalanobis distance
`(x-μ)ᵀ Σ⁻¹ (x-μ)` and `log|Σ|`. Forming `Σ⁻¹` explicitly with `np.linalg.inv`
squares the condition number of the problem, and `np.linalg.det` of a nearly
singular matrix underflows to 0 and takes `log(0) = -inf` with it.

Instead, factor `Σ = L Lᵀ` (Cholesky, L lower-triangular) and define
`P = L⁻ᵀ`, so that `Σ⁻¹ = P Pᵀ`. Then both quantities fall out cheaply and stably:

```
(x - μ)ᵀ Σ⁻¹ (x - μ) = ‖ (x - μ) @ P ‖²        # no inverse ever formed
log|Σ|               = -2 × sum(log(diag(P)))  # a sum of logs, cannot underflow
```

`_compute_precision_cholesky()` returns `P` (the same convention scikit-learn
stores in `precisions_cholesky_`), and `_estimate_log_gaussian_prob()` uses it for
the `'full'` and `'tied'` types. For `'diag'` and `'spherical'` the covariance is
already diagonal, so there is nothing to factor - those branches use the variances
directly. Verified against `scipy.stats.multivariate_normal.logpdf`: maximum
absolute difference 1.3e-13 across all four covariance types.

### Simplification vs. canonical scikit-learn GaussianMixture

This implementation matches `sklearn.mixture.GaussianMixture` exactly on fitted
weights, means, covariances, total log-likelihood, BIC and AIC for all four
covariance types (differences of 0.0000 in BIC/AIC, ARI 1.0000 between the two
label vectors, on a seeded 600-point 3-cluster set). Three things are deliberately
simpler:

1. **Initialisation.** scikit-learn's default `init_params='kmeans'` runs a full
   k-means to convergence and derives the starting responsibilities from its
   labels. This class does a single k-means++ D² seeding pass for the means and
   uses `cov(X)` for every covariance (see
   [Why not the identity matrix?](#the-em-algorithm-in-detail)). Both usually
   land in the same optimum; scikit-learn's start is slightly better conditioned.

2. **`score()` returns the SUM, not the mean.** `gmm.score(X)` is the total
   log-likelihood over all samples; `GaussianMixture.score(X)` is the per-sample
   mean. They differ by a factor of `len(X)` (see
   [Model Evaluation](#1-log-likelihood)). `bic()`, `aic()` and the `tol`
   convergence check are all consistent with the summed form.

3. **Not implemented at all:** the Bayesian / variational variant
   (`BayesianGaussianMixture`), `warm_start`, and the `means_init` /
   `precisions_init` / `weights_init` hand-seeding parameters. If you need those,
   use scikit-learn - this file exists to show you how EM works, not to replace it.

---

## Step-by-Step Example

Let's cluster 2D data with 3 Gaussian components. **Every number below was produced
by actually running this implementation** - copy the dataset block, run it, and you
will get exactly these values.

### Dataset
```python
import numpy as np

# 150 points from 3 Gaussians
np.random.seed(0)
g1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50)      # Group 1
g2 = np.random.multivariate_normal([5, 5], [[2, 0.5], [0.5, 1]], 50)  # Group 2
g3 = np.random.multivariate_normal([5, 0], [[1, -0.5], [-0.5, 1]], 50)  # Group 3
X = np.vstack([g1, g2, g3])

gmm = GaussianMixtureModel(n_components=3, covariance_type='full',
                           random_state=42, max_iter=100)
gmm.fit(X)
```

### Initialization (Iteration 0)

k-means++ picks three actual data rows as the starting means, and every covariance
starts as the scatter of the whole dataset:

```
Component 0: μ = [ 3.6372, -0.2541], π = 0.3333
Component 1: μ = [ 6.3377,  4.8895], π = 0.3333
Component 2: μ = [ 1.2303,  1.2024], π = 0.3333

Covariances: all three start at cov(X) = [[6.9919, 2.6175],
                                          [2.6175, 6.9117]]
```

Notice how wide that starting covariance is - it spans all three groups at once.
That is deliberate: a wide start keeps every responsibility away from 0 and 1 on
the first pass, which is what lets EM actually move.

### Iteration 1

**E-Step:** Calculate responsibilities

```
Sample #0 at [1.7641, 0.4002]:
  N(x | μ_0, Σ_0) = 0.016470,  π_0 × N = 0.333 × 0.016470 = 0.005490
  N(x | μ_1, Σ_1) = 0.002890,  π_1 × N = 0.333 × 0.002890 = 0.000963
  N(x | μ_2, Σ_2) = 0.022250,  π_2 × N = 0.333 × 0.022250 = 0.007417
  Sum = 0.013870

  γ(z_0,0) = 0.005490 / 0.013870 = 0.3958
  γ(z_0,1) = 0.000963 / 0.013870 = 0.0695
  γ(z_0,2) = 0.007417 / 0.013870 = 0.5347  ← leaning to component 2, but only just

Sample #75 at [4.5098, 6.4483] (inside group 2):
  γ(z_75,0) = 0.0463  ( 4.6%)
  γ(z_75,1) = 0.7750  (77.5%)  ← strongly component 1
  γ(z_75,2) = 0.1787  (17.9%)
```

Those are genuinely *soft* numbers - nothing is 0.99 yet. With `cov(X)` seeding,
**0 of the 150 points** have a responsibility above 0.99 after the first E-step.
Force the covariances back to the identity matrix and that becomes **111 of 150**,
with sample #75 collapsing to a flat `[0.0000, 1.0000, 0.0000]`. EM would then be
shuffling near-hard assignments from the very first step, which is exactly how it
gets stuck. This is the same effect described in
[Why not the identity matrix?](#the-em-algorithm-in-detail).

**M-Step:** Update parameters

```
Component 0:
  N_0 = Σ_n γ(z_n,0) = 53.17
  π_0 = 53.17 / 150 = 0.3545
  μ_0 = (Σ_n γ(z_n,0) × x_n) / 53.17 = [3.9565, 0.2460]
  Σ_0 = weighted covariance = [[5.4302, 0.1253], [0.1253, 2.8684]]

Component 1:
  N_1 = 42.53
  π_1 = 0.2836
  μ_1 = [5.0745, 4.3421]
  Σ_1 = [[2.2751, 1.1366], [1.1366, 5.1733]]

Component 2:
  N_2 = 54.29
  π_2 = 0.3620
  μ_2 = [1.4034, 1.1191]
  Σ_2 = [[5.6084, 2.3335], [2.3335, 4.2310]]
```

**Log-likelihood trace** (the value at the start of each iteration, before that
iteration's M-step):

```
Iteration  1: log L = -713.46
Iteration  2: log L = -672.20
Iteration  3: log L = -647.92
Iteration  4: log L = -630.09
Iteration  5: log L = -622.74
Iteration  6: log L = -618.17
Iteration  7: log L = -612.96
Iteration  8: log L = -605.97
Iteration  9: log L = -597.58
Iteration 10: log L = -593.16
Iteration 11: log L = -592.49
Iteration 12: log L = -592.30
Iteration 13: log L = -592.22
Iteration 14: log L = -592.19
Iteration 15: log L = -592.18
...
Iteration 20: log L = -592.18   ← change < tol = 1e-4, stop
```

It never goes down. That is the EM guarantee from the
[Log-Likelihood section](#3-the-em-algorithm), visible in real numbers.

### Convergence (Iteration 20)

```python
gmm.converged_    # True
gmm.n_iter_       # 20
gmm.lower_bound_  # -592.18
gmm.score(X)      # -592.18  (the TOTAL log-likelihood over all 150 points)
```

The three recovered components against the three the data really came from:

| Component | Recovered π | Recovered μ | Recovered Σ | True generator |
|---|---|---|---|---|
| 2 | 0.3526 | [0.176, 0.112] | [[1.4445, -0.0175], [-0.0175, 0.9772]] | N([0,0], [[1,0],[0,1]]) |
| 1 | 0.3347 | [4.954, 5.146] | [[2.0681, 0.5424], [0.5424, 1.1358]] | N([5,5], [[2,0.5],[0.5,1]]) |
| 0 | 0.3127 | [5.211, -0.122] | [[0.7921, -0.4552], [-0.4552, 0.7669]] | N([5,0], [[1,-0.5],[-0.5,1]]) |

Component numbering is arbitrary (EM has no idea which Gaussian you called "group
1"), but every mean, weight and covariance - including the *signs of the
off-diagonal correlations* - has been recovered. That is the whole algorithm
working.

### Final Results

**Hard Clustering (predict):**
```python
labels = gmm.predict(X)      # same as gmm.labels_ after fit
labels[:10]                  # [2 2 2 2 2 2 2 2 2 2]  - all of group 1 -> component 2
np.bincount(labels)          # [47 50 53]  (true group sizes: 50 / 50 / 50)
```

**Soft Clustering (predict_proba):**
```python
probe = np.array([[2.5, 2.5],    # between the [0,0] and [5,5] blobs
                  [0.1, 0.1],    # dead centre of the [0,0] blob
                  [5.0, 2.5]])   # between the [5,5] and [5,0] blobs
print(np.round(gmm.predict_proba(probe), 4))

# [2.5, 2.5] -> [0.1402, 0.6521, 0.2077]   genuinely split three ways
# [0.1, 0.1] -> [0.0000, 0.0000, 1.0000]   no doubt at all
# [5.0, 2.5] -> [0.1285, 0.8709, 0.0006]   mostly component 1
```

The first row is the reason GMM exists. A hard clusterer would stamp that point
"cluster 1" and throw away the fact that it is only 65% sure.
---

## Real-World Applications

### 1. Customer Segmentation

**Problem:** Group customers with overlapping behaviors

```python
# Customer features: [frequency, recency, monetary_value, engagement]
# Some customers exhibit mixed behaviors

GMM provides:
- Segment labels (VIP, Regular, Occasional, Inactive)
- Membership probabilities (e.g., 60% VIP, 40% Regular)
- Identify "boundary" customers for special campaigns
```

**Why GMM over K-means?**
- Captures uncertainty (customer might belong to multiple segments)
- Handles different cluster shapes (VIP cluster might be elongated)
- Natural probabilistic interpretation

### 2. Image Segmentation

**Problem:** Separate image regions (foreground/background)

```python
# Pixel features: [R, G, B] or [R, G, B, texture, gradient]

GMM learns:
- Distribution of colors in each region
- Soft boundaries (anti-aliasing)
- Handles color gradients naturally
```

**Advantage:**
- Soft segmentation for smooth boundaries
- Can model complex color distributions

### 3. Anomaly Detection

**Problem:** Identify unusual data points

```python
# Train GMM on normal data
gmm.fit(X_normal)

# Low probability = anomaly
log_probs = gmm.score_samples(X_test)
anomalies = X_test[log_probs < threshold]
```

**Why GMM?**
- Models complex normal behavior with multiple modes
- Provides probability (severity of anomaly)
- More flexible than single Gaussian

### 4. Speech Recognition

**Problem:** Model phoneme distributions

```python
# Audio features: MFCCs (Mel-frequency cepstral coefficients)

For each phoneme:
- GMM models distribution of acoustic features
- Diagonal covariance (features approximately independent)
- Multiple components capture variations (speaker, context)
```

**Hidden Markov Models (HMM) with GMM emissions:**
- Each HMM state has GMM for observation probability
- Captures both temporal structure (HMM) and feature distributions (GMM)

### 5. Medical Diagnosis

**Problem:** Identify disease subtypes

```python
# Patient features: [symptoms, lab_results, biomarkers]

GMM discovers:
- Disease subtypes (clusters)
- Patient assignment probabilities
- Uncertainty in diagnosis
```

**Clinical value:**
- Probabilistic diagnosis (not just "yes" or "no")
- Identify borderline cases needing more testing
- Personalized treatment based on subtype

### 6. Finance: Market Regime Detection

**Problem:** Identify market conditions (bull, bear, sideways)

```python
# Features: [returns, volatility, volume, momentum]

GMM identifies:
- Market regimes (3-4 components)
- Transition periods (high uncertainty)
- Regime-specific strategies
```

**Trading applications:**
- Regime-dependent portfolio allocation
- Risk management based on current regime
- Detect regime changes early (increasing uncertainty)

---

## Understanding the Code

### Core Class Structure

```python
class GaussianMixtureModel:
    def __init__(self, n_components=3, covariance_type='full', ...):
        # Configuration
        self.n_components = n_components
        self.covariance_type = covariance_type
        
        # Learned parameters
        self.weights_ = None      # π (mixing coefficients)
        self.means_ = None        # μ (component means)
        self.covariances_ = None  # Σ (component covariances)
        self.labels_ = None       # hard labels of the training data
        self.converged_ = False   # did this fit converge?
        self.n_iter_ = 0          # EM iterations actually run
        self.lower_bound_ = None  # final total log-likelihood
```

### Key Methods

**1. fit(X, y=None): Train the model**
```python
def fit(self, X, y=None):        # y is ignored, present for API consistency
    X = self._check_array(X)     # lists and 1-D input are accepted

    for _ in range(self.n_init):     # independent EM restarts
        # 1. Initialize parameters
        self._initialize_parameters(X)

        # 2. EM iterations
        for iteration in range(max_iter):
            # E-step: compute responsibilities
            responsibilities, log_likelihood = self._e_step(X)

            # M-step: update parameters
            self._m_step(X, responsibilities)

            # Check convergence
            if change < tolerance:
                break

        # 3. Keep this restart only if it beat the previous best
        ...

    self.labels_ = self.predict(X)
    return self
```

**2. predict(X): Hard clustering**
```python
def predict(self, X):
    # Get responsibilities
    responsibilities = self._e_step(X)[0]
    
    # Assign to component with highest probability
    return np.argmax(responsibilities, axis=1)
```

**3. predict_proba(X): Soft clustering**
```python
def predict_proba(self, X):
    # Return full responsibility matrix
    # Each row sums to 1.0
    return self._e_step(X)[0]
```

**4. sample(n): Generate samples**
```python
def sample(self, n_samples):
    # 1. Select component for each sample (based on weights)
    components = np.random.choice(K, size=n, p=self.weights_)
    
    # 2. Sample from selected component's Gaussian
    for i, k in enumerate(components):
        X[i] = np.random.multivariate_normal(
            self.means_[k],
            self.covariances_[k]
        )
    
    return X, components
```

**5. score_samples(X): Per-point log-density**
```python
def score_samples(self, X):
    # log p(x_n) = log( Σ_k π_k × N(x_n | μ_k, Σ_k) ), one value per row
    weighted_log_prob = self._estimate_log_gaussian_prob(X) + np.log(self.weights_)
    return self._log_sum_exp(weighted_log_prob, axis=1)   # shape (n_samples,)
```
`score(X)` is exactly `score_samples(X).sum()`. Use `score_samples` whenever you
need *which* points are unlikely (anomaly detection) rather than how well the
model fits overall.

**6. fit_predict(X) and labels_: the clustering shorthand**
```python
labels = gmm.fit_predict(X)   # identical to gmm.fit(X).labels_
```

**The full public API:** `fit`, `fit_predict`, `predict`, `predict_proba`,
`score`, `score_samples`, `sample`, `bic`, `aic`, plus the fitted attributes
`weights_`, `means_`, `covariances_`, `labels_`, `converged_`, `n_iter_`,
`lower_bound_`.

### Understanding Parameters

**n_components:** How many Gaussians?
```python
Too few: Underfits, can't capture complexity
Just right: Captures true structure
Too many: Overfits, splits natural clusters

Use BIC/AIC to select optimal K
```

**covariance_type:** Cluster shape
```python
'full': Any ellipse orientation (most flexible)
    Use when: Clusters have different shapes/orientations
    Parameters: K × d × (d+1) / 2

'diag': Axis-aligned ellipses
    Use when: Features independent within clusters
    Parameters: K × d
    
'spherical': Circular clusters (like K-means)
    Use when: Clusters are roughly spherical
    Parameters: K
    
'tied': All clusters same shape
    Use when: Clusters have similar shapes, reduces overfitting
    Parameters: d × (d+1) / 2
```

**max_iter & tol:** Convergence control
```python
max_iter=100: Maximum EM iterations
    Typically converges in 5-30 iterations on well-scaled data
    (more when features are on very different scales)

tol=1e-4: Stop when log-likelihood change < tol
    Smaller tol: More precise, slower
    Larger tol: Faster, less precise
    Careful: the change compared is in the TOTAL log-likelihood, not the
    per-sample mean, so tol=1e-4 here behaves like sklearn's tol=1e-4/n
```

**n_init:** How many EM restarts
```python
n_init=1 (default, same as sklearn): a single run - fast, but at the mercy
    of where the k-means++ seeding happened to land
n_init=5..10: run EM that many times from different starts and keep the
    parameters with the highest final log-likelihood

Cost is linear: n_init=10 takes ~10x as long. Worth it whenever the data
has overlapping or unbalanced clusters.
```

---

## Model Evaluation

### 1. Log-Likelihood

```python
log_likelihood = gmm.score(X)          # TOTAL, summed over all samples
per_sample     = gmm.score(X) / len(X) # what sklearn's .score() returns

Interpretation:
- Higher = better fit
- Compare models on SAME data
- Not interpretable in absolute terms

Careful: score() here returns the SUM over samples, while
scikit-learn's GaussianMixture.score returns the per-sample MEAN.
To compare:  gmm.score(X)  ==  sklearn_gmm.score(X) * len(X)
bic(), aic() and the tol convergence check all use the summed form.
Divide by len(X) yourself before comparing across datasets of
different sizes.

Use for:
- Monitoring convergence
- Comparing different n_components
```

### 2. Information Criteria

**BIC (Bayesian Information Criterion):**
```python
bic = gmm.bic(X)

# Lower is better
# Penalizes complexity more than AIC
# Preferred for large datasets

# Use for model selection:
bic_scores = []
for k in range(1, 10):
    gmm = GaussianMixtureModel(n_components=k)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))

best_k = np.argmin(bic_scores) + 1
```

**AIC (Akaike Information Criterion):**
```python
aic = gmm.aic(X)

# Lower is better
# Penalizes complexity less than BIC
# May select more complex models

# Often used in combination with BIC
```

### 3. Silhouette Score (with labels)

```python
from sklearn.metrics import silhouette_score

labels = gmm.predict(X)
score = silhouette_score(X, labels)

# Range: [-1, 1]
# 1: Perfect clustering
# 0: Overlapping clusters
# -1: Wrong clustering
```

### 4. Cluster Quality Metrics

**Within-cluster variance:**
```python
for k in range(n_components):
    cluster_mask = (labels == k)
    cluster_data = X[cluster_mask]
    variance = np.var(cluster_data, axis=0)
    print(f"Cluster {k} variance: {variance}")
```

**Separation between clusters:**
```python
# Distance between means
for i in range(n_components):
    for j in range(i+1, n_components):
        dist = np.linalg.norm(gmm.means_[i] - gmm.means_[j])
        print(f"Distance {i}-{j}: {dist:.2f}")
```

### 5. Visualization

**2D scatter plot with soft colors:**
```python
import matplotlib.pyplot as plt

# Get probabilities
probs = gmm.predict_proba(X)

# Plot with alpha based on confidence
for k in range(n_components):
    # Color intensity = probability
    plt.scatter(X[:, 0], X[:, 1], 
                alpha=probs[:, k], 
                c=f'C{k}',
                label=f'Component {k}')

# Plot means
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
           marker='X', s=200, c='black', 
           edgecolors='white', linewidths=2)

plt.legend()
plt.title('GMM Soft Clustering')
```

**Confidence plot:**
```python
# Show confidence of assignments
max_probs = np.max(probs, axis=1)
plt.hist(max_probs, bins=50)
plt.xlabel('Max Probability (Confidence)')
plt.ylabel('Count')
plt.title('Assignment Confidence Distribution')

# Identify uncertain points
uncertain = X[max_probs < 0.6]
print(f"Uncertain points: {len(uncertain)}")
```

### 6. Cross-Validation Strategy

```python
from sklearn.model_selection import KFold

# For density estimation tasks
kf = KFold(n_splits=5, shuffle=True)
log_likelihoods = []

for train_idx, val_idx in kf.split(X):
    gmm = GaussianMixtureModel(n_components=3)
    gmm.fit(X[train_idx])
    
    # Evaluate on validation set.
    # score() returns the fold TOTAL, so divide by the fold size to get a
    # per-sample figure that is comparable across folds of different lengths.
    val_ll = gmm.score(X[val_idx]) / len(val_idx)
    log_likelihoods.append(val_ll)

print(f"Avg per-sample log-likelihood: {np.mean(log_likelihoods):.4f}")
print(f"Std: {np.std(log_likelihoods):.4f}")
```

---

## Tips and Best Practices

### 1. Choosing Number of Components

```python
import matplotlib.pyplot as plt

# Strategy 1: BIC/AIC curve
bics = []
for k in range(1, 11):
    gmm = GaussianMixtureModel(n_components=k)
    gmm.fit(X)
    bics.append(gmm.bic(X))

# Look for "elbow" in curve
plt.plot(range(1, 11), bics, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC')

# Strategy 2: Domain knowledge
# If modeling customer segments, business might define 3-5 segments

# Strategy 3: Silhouette analysis
# Try different K, compare silhouette scores
```

### 2. Choosing Covariance Type

```python
# Start with 'full' if you have enough data
# n_samples > 10 × n_features × n_components

# Use 'diag' if:
# - Features are approximately independent
# - Limited data
# - Need faster computation

# Use 'spherical' if:
# - Just need simple clustering (like K-means)
# - Very limited data
# - Features have similar scales

# Use 'tied' if:
# - Clusters have similar shapes
# - Want to reduce overfitting
```

### 3. Initialization

```python
# Our implementation uses K-means++ seeding for the means and cov(X) for
# the covariances. EM only ever finds a LOCAL maximum, so the cheapest
# insurance is simply to restart it - that is what n_init does:

gmm = GaussianMixtureModel(n_components=3, n_init=10, random_state=42)
gmm.fit(X)      # runs EM 10 times, keeps the highest-log-likelihood result

# The same thing written out by hand, if you want to inspect each run:
best_gmm = None
best_ll = -np.inf

for seed in range(10):
    gmm = GaussianMixtureModel(n_components=3, random_state=seed)
    gmm.fit(X)
    ll = gmm.score(X)
    
    if ll > best_ll:
        best_ll = ll
        best_gmm = gmm
```

### 4. Handling Convergence Issues

```python
# If not converging:
# 1. Increase max_iter
gmm = GaussianMixtureModel(max_iter=200)

# 2. Increase regularization
gmm = GaussianMixtureModel(reg_covar=1e-5)

# 3. Reduce n_components
# 4. Check for outliers, consider removing
# 5. Standardize features
```

### 5. Feature Preprocessing

```python
# Always standardize features!
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixtureModel(n_components=3)
gmm.fit(X_scaled)

# For new data:
X_new_scaled = scaler.transform(X_new)
predictions = gmm.predict(X_new_scaled)

# Or with numpy only:
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

**Why it matters, concretely.** A full covariance matrix has to describe a feature
with std 500 and a feature with std 2 in the same matrix, so its entries span six
orders of magnitude and it becomes badly conditioned - EM then needs many more
iterations and is far more likely to stall in a poor optimum. On the
customer-segmentation data of USAGE EXAMPLE 3, standardizing cuts convergence
from 24 EM iterations to 8. Seeding the covariances from `cov(X)` (which this
implementation does, see
[Why not the identity matrix?](#the-em-algorithm-in-detail)) removes most of the
damage, but standardizing is still the right habit.

### 6. Anomaly Detection Threshold

```python
# Train on normal data
gmm.fit(X_normal)

# Get one log-likelihood per training point, in a single vectorised call
log_probs = gmm.score_samples(X_normal)     # shape (len(X_normal),)

# Set threshold at percentile - calibrated on NORMAL data only.
# Never compute the percentile over the data you are about to score:
# the anomalies would then help choose the cut-off meant to catch them,
# and the number of detections would be pinned at exactly 5% by construction.
threshold = np.percentile(log_probs, 5)  # Bottom 5%

# Detect anomalies
anomalies = X_test[gmm.score_samples(X_test) < threshold]
```

---

## Comparison with Other Algorithms

### GMM vs K-Means

| Aspect | K-Means | GMM |
|--------|---------|-----|
| **Assignments** | Hard (one cluster) | Soft (probabilities) |
| **Cluster Shape** | Spherical only | Any elliptical shape |
| **Algorithm** | Iterative reassignment | EM algorithm |
| **Output** | Labels only | Labels + probabilities |
| **Speed** | Faster | Slower (more parameters) |
| **Use Case** | Simple, fast clustering | Complex distributions |

### GMM vs Hierarchical Clustering

| Aspect | Hierarchical | GMM |
|--------|-------------|-----|
| **Structure** | Tree/dendrogram | Flat clusters |
| **K Selection** | Choose by cutting tree | Must specify |
| **Probabilistic** | No | Yes |
| **Scalability** | O(n²) or O(n³) | O(n × k × iter) |
| **Soft Clustering** | No | Yes |

### GMM vs DBSCAN

| Aspect | DBSCAN | GMM |
|--------|--------|-----|
| **Cluster Shape** | Arbitrary | Elliptical |
| **Noise Handling** | Explicit noise class | Probabilistic |
| **Parameters** | ε, minPts | K, covariance type |
| **Density** | Density-based | Probabilistic |
| **Non-convex** | Yes | No |

---

## Common Pitfalls and Solutions

### 1. Singular Covariance Matrix

**Problem:** Covariance matrix becomes non-invertible

**Causes:**
- Too many components for data size
- Features are perfectly correlated
- Numerical precision issues

**Solutions:**
```python
# Add regularization (done automatically)
gmm = GaussianMixtureModel(reg_covar=1e-6)

# Reduce n_components
# Remove perfectly correlated features
# Use 'diag' or 'spherical' covariance
```

**The failure mode behind it: component collapse.**

This is not a numerical accident, it is a real property of the objective. Put one
component's mean exactly on a single data point and shrink its covariance toward
zero: `N(x | μ_k, Σ_k)` at that point grows without bound as `|Σ_k| → 0`, so the
likelihood **diverges to +∞**. The maximum-likelihood problem for a Gaussian
mixture with free covariances is genuinely unbounded - there is no "best" answer
to converge to, only a singularity to fall into.

Signs you have hit it:
- A component's weight collapses to roughly `1/N`
- Its covariance has a near-zero eigenvalue (`np.linalg.eigvalsh(gmm.covariances_[k])`)
- `lower_bound_` shoots up implausibly instead of levelling off

What actually stops it here: `reg_covar` puts a floor of `ε` on every eigenvalue
by adding `ε I` to each covariance, which caps how narrow a component can get, and
`N_k = Σγ + 10·eps` in the M-step prevents division by an empty component. If a
component still collapses, raise `reg_covar` to `1e-4`, lower `n_components`, or
switch to `'diag'`/`'spherical'`, which have far fewer degrees of freedom to
collapse along.

### 2. Poor Initialization

**Problem:** Converges to bad local optimum

**Solution:**
```python
# Simplest: let n_init do the restarts for you
gmm = GaussianMixtureModel(n_components=3, n_init=10, random_state=42)
gmm.fit(X)

# Equivalent by hand, if you want to keep every candidate:
best_score = -np.inf
best_gmm = None

for seed in range(10):
    gmm = GaussianMixtureModel(random_state=seed)
    gmm.fit(X)
    score = gmm.score(X)
    
    if score > best_score:
        best_score = score
        best_gmm = gmm
```

Also: standardize first, and remember that this implementation seeds the
covariances from `cov(X)` precisely because the identity matrix was the single
biggest source of bad local optima on unscaled data.

### 3. Wrong Number of Components

**Problem:** Under/overfitting

**Solution:**
```python
# Use BIC/AIC for selection
bic_scores = []
for k in range(1, 11):
    gmm = GaussianMixtureModel(n_components=k)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))

optimal_k = np.argmin(bic_scores) + 1
```

### 4. Non-Gaussian Data

**Problem:** Data doesn't follow Gaussian distributions

**Solution:**
```python
# Transform data (log, Box-Cox)
X_transformed = np.log1p(X)  # For skewed data

# Use more components to approximate distribution
gmm = GaussianMixtureModel(n_components=10)

# Or use different algorithm (DBSCAN, etc.)
```

### 5. Scalability Issues

**Problem:** Large datasets, many features

**Solution:**
```python
# Use 'diag' or 'spherical' covariance
gmm = GaussianMixtureModel(covariance_type='diag')

# Reduce dimensionality first (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)
gmm.fit(X_reduced)

# Sample subset for initial fit
```

---

## Further Reading and Resources

### Academic Papers

1. **Original EM Algorithm:**
   - Dempster, Laird, Rubin (1977): "Maximum Likelihood from Incomplete Data via the EM Algorithm"

2. **GMM Theory:**
   - Bishop (2006): "Pattern Recognition and Machine Learning", Chapter 9

3. **Applications:**
   - Reynolds et al. (2000): "Speaker Verification Using Adapted Gaussian Mixture Models"

### Online Resources

- [Scikit-learn GMM Documentation](https://scikit-learn.org/stable/modules/mixture.html)
- [StatQuest: Gaussian Mixture Models](https://www.youtube.com/user/joshstarmer)
- [Chris Bishop's Book](https://www.microsoft.com/en-us/research/people/cmbishop/)

### Related Algorithms in This Repository

- [K-Means Clustering](../10.%20k-Means%20Clustering/_10_kmeans_clustering.md): Hard clustering baseline
- [Hierarchical Clustering](../12.%20Hierarchical%20Clustering/_12_hierarchical_clustering.md): Alternative clustering
- [PCA](../11.%20PCA/_11_pca.md): Dimensionality reduction before GMM
- [t-SNE](../14.%20t-SNE/_14_tsne.md): Visualization after clustering

---

## Summary

### Key Takeaways

✅ **GMM provides soft clustering** - each point has probabilities for all clusters

✅ **Flexible cluster shapes** - can model elliptical clusters of any orientation

✅ **EM algorithm** - iteratively improves model by E-step (assign) and M-step (update)

✅ **Multiple applications** - clustering, density estimation, anomaly detection, generation

✅ **Model selection** - use BIC/AIC to choose optimal number of components

✅ **Covariance types** - trade-off between flexibility and complexity

### When to Use GMM

**Choose GMM when you need:**
- Soft assignments (probability of belonging to each cluster)
- Non-spherical clusters with different shapes/orientations
- Density estimation of complex distributions
- Probabilistic clustering with uncertainty quantification
- Generative model (can sample new data)

**Choose alternatives when:**
- Need simple, fast hard clustering → K-means
- Need to find arbitrary-shaped clusters → DBSCAN
- Need hierarchical structure → Hierarchical Clustering
- Data is clearly non-Gaussian → Non-parametric methods

---

**Congratulations!** 🎉 You now understand Gaussian Mixture Models from scratch. You've learned:
- How GMM models data as mixture of Gaussians
- The EM algorithm for parameter estimation
- Different covariance types and their trade-offs
- Practical applications and implementation details

**Next steps:**
- Implement GMM on your own dataset
- Try different covariance types and compare
- Use BIC/AIC for model selection
- Combine with dimensionality reduction (PCA)
- Explore advanced variants (Bayesian GMM, Variational Inference)

Happy clustering! 🚀
