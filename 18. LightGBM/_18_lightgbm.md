# LightGBM from Scratch: A Comprehensive Guide

Welcome to LightGBM! 🚀 In this comprehensive guide, we'll explore LightGBM (Light Gradient Boosting Machine) - a fast, distributed, high-performance gradient boosting framework that's become the preferred choice for large-scale machine learning tasks. Think of it as the "speed champion" of gradient boosting!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is LightGBM?](#what-is-lightgbm)
3. [How LightGBM Works](#how-lightgbm-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Simplifications vs. canonical LightGBM](#simplifications-vs-canonical-lightgbm)
11. [LightGBM vs XGBoost vs Gradient Boosting](#lightgbm-vs-xgboost-vs-gradient-boosting)
12. [Summary](#summary)
13. [References and Further Learning](#references-and-further-learning)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy. (Running `python _18_lightgbm.py` executes the same thing from the file's `__main__` block.)

```python
# ---------------------------------------------------------------
# LightGBM from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _18_lightgbm.py  (the __main__ block runs this)
# Or copy the LightGBM class from _18_lightgbm.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the LightGBM class here (from _18_lightgbm.py) ----
# class LightGBM: ...

np.random.seed(42)

# ------ REGRESSION: predict y = x^2 + noise ------
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Shuffle before splitting: trees cannot extrapolate beyond the training range.
# Without shuffling the last 50 x-values would all be above the training maximum.
idx = np.random.permutation(200)
X, y = X[idx], y[idx]

X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

model = LightGBM(
    n_estimators=60,
    learning_rate=0.1,
    num_leaves=15,        # LightGBM's main complexity control
    min_data_in_leaf=10,
    lambda_l2=1.0         # L2 regularization
)
model.fit(X_train, y_train)

print(f"Train R2: {model.score(X_train, y_train):.4f}")
print(f"Test  R2: {model.score(X_test,  y_test):.4f}")

preds = model.predict(X_test)
for i in range(5):
    print(f"  x={X_test[i,0]:5.2f}  true={y_test[i]:5.2f}  pred={preds[i]:5.2f}")

# ------ CLASSIFICATION: two Gaussian blobs ------
X0 = np.random.randn(100, 2) + np.array([-2, -2])
X1 = np.random.randn(100, 2) + np.array([ 2,  2])
X_c = np.vstack([X0, X1])
y_c = np.array([0]*100 + [1]*100)
idx = np.random.permutation(200)
X_c, y_c = X_c[idx], y_c[idx]

cls = LightGBM(
    n_estimators=40,
    learning_rate=0.1,
    num_leaves=15,
    min_data_in_leaf=10,
    objective='binary'
)
cls.fit(X_c[:150], y_c[:150])

print(f"\nClassification accuracy: {cls.score(X_c[150:], y_c[150:]):.2%}")
proba = cls.predict_proba(X_c[150:])
for i in range(3):
    print(f"  true={y_c[150+i]}  P(0)={proba[i,0]:.3f}  P(1)={proba[i,1]:.3f}")

# ------ WHAT MAKES IT LIGHTGBM: histogram bins and the leaf budget ------
def count_leaves(node):
    if node['type'] == 'leaf':
        return 1
    return count_leaves(node['left']) + count_leaves(node['right'])

binned = model._apply_binning(X_train)
print(f"\nBins used by feature 0 (max_bin={model.max_bin}): {len(np.unique(binned[:, 0]))}")
print(f"Leaves in the first 5 trees (num_leaves={model.num_leaves}): "
      f"{[count_leaves(t) for t in model.trees[:5]]}")
```

Expected output:
```
Train R2: 0.9779
Test  R2: 0.9563
  x=-2.88  true= 8.17  pred= 8.16
  x= 0.23  true= 0.14  pred= 0.18
  x= 2.55  true= 6.38  pred= 6.66
  x=-1.43  true= 1.71  pred= 2.22
  x= 2.34  true= 6.19  pred= 5.49

Classification accuracy: 100.00%
  true=1  P(0)=0.010  P(1)=0.990
  true=0  P(0)=0.931  P(1)=0.069
  true=1  P(0)=0.009  P(1)=0.991

Bins used by feature 0 (max_bin=255): 150
Leaves in the first 5 trees (num_leaves=15): [9, 9, 9, 9, 9]
```

The last two lines are the two things that make this LightGBM rather than plain
gradient boosting. `max_bin=255` is a **ceiling**: 150 distinct training values
produce 150 bins, and a binary 0/1 column would produce exactly 2. `num_leaves`
is a **budget**: growth is best-first and stops when the budget (or
`min_data_in_leaf`, whichever binds first) is reached - here at 9 leaves.

---

## What is LightGBM?

LightGBM (Light Gradient Boosting Machine) is a **gradient boosting framework developed by Microsoft** that uses tree-based learning algorithms. It's designed to be distributed and efficient, with significant advantages in training speed, memory usage, and accuracy, especially on large datasets.

**Real-world analogy**: 
If XGBoost is like a meticulous craftsman carefully examining every detail, LightGBM is like a smart engineer who:
- Uses blueprints (histograms) to work faster
- Builds from the most important parts first (leaf-wise growth)
- Focuses on critical cases (gradient-based sampling)
- Bundles similar materials together (feature bundling)

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Ensemble Learning (Advanced Boosting) |
| **Learning Style** | Supervised Learning |
| **Primary Use** | Regression, Classification, Ranking |
| **Base Learners** | Decision trees with leaf-wise growth |
| **Key Innovation** | Histogram-based learning + Leaf-wise growth |

### The Core Idea

```
"LightGBM = Gradient Boosting + Speed Optimizations + Smart Sampling"
```

LightGBM improves upon XGBoost and standard gradient boosting through:
- **Histogram-based learning**: Bins continuous features for faster splits
- **Leaf-wise tree growth**: Grows trees by best leaf, not level
- **GOSS**: Gradient-based One-Side Sampling for efficient training
- **EFB**: Exclusive Feature Bundling to reduce dimensions
- **Optimized for speed**: Parallel learning, GPU support

The first three are implemented in `_18_lightgbm.py` (GOSS behind
`boosting_type='goss'`). EFB is explained here but not implemented - see
[Simplifications vs. canonical LightGBM](#simplifications-vs-canonical-lightgbm).

### Key Differences from XGBoost

**1. Tree Growth Strategy**
```
XGBoost: Level-wise (depth-wise) growth
├── Split all nodes at each level
├── More balanced trees
└── Potentially slower

LightGBM: Leaf-wise (best-first) growth
├── Split only the leaf with maximum gain
├── Deeper, more asymmetric trees
└── Faster convergence

Example:
XGBoost (Level-wise):        LightGBM (Leaf-wise):
       Root                         Root
      /    \                       /    \
     A      B                     A      B
    / \    / \                   / \
   C   D  E  F                  C   D
                                   / \
                                  E   F
```

**2. Histogram-based Learning**
```
Traditional GB: Considers all possible split points
├── For each feature
├── Try every unique value
└── Slow for continuous features

LightGBM: Bins features into histograms
├── Discretize continuous features into bins (e.g., 255 bins)
├── Only try bin boundaries as splits
├── Much faster split finding
└── Lower memory usage

Example: Temperature feature [15°, 18°, 22°, 25°, 28°, 30°, 35°]
Bins: [<20°, 20-25°, 25-30°, >30°]
Only need to check 3 split points instead of 6!
```

**3. Training Speed**
```
Small Dataset (<10K rows):
XGBoost ≈ LightGBM

Large Dataset (>100K rows):
LightGBM can be 10-20x faster!

Why?
- Histogram building is faster
- Leaf-wise growth converges quicker
- Better memory efficiency
```

**4. Memory Usage**
```
XGBoost: Stores all feature values
LightGBM: Stores binned histograms
Result: LightGBM uses ~1/8 memory for the same dataset
```

---

## How LightGBM Works

### The Algorithm in 7 Steps

```
Step 1: Build histogram bins for all features
         - Discretize continuous features
         - Typical: 255 bins per feature
         ↓
Step 2: Initialize predictions (base_score)
         ↓
Step 3: For each boosting iteration:
         a. Calculate gradients and hessians
         b. Row sampling: GOSS (boosting_type='goss') or bagging
         ↓
Step 4: Build tree using LEAF-WISE strategy:
         - Find leaf with maximum gain
         - Split that leaf (not the whole level)
         - Repeat until num_leaves reached
         ↓
Step 5: For each potential split:
         - Use histogram to quickly find best split
         - Gain = G_L²/(H_L+λ) + G_R²/(H_R+λ) - G_P²/(H_P+λ)
           (no factor of ½ - that is XGBoost's convention, not LightGBM's)
         ↓
Step 6: Calculate leaf weights: w* = -ThresholdL1(G, α)/(H+λ)
         where ThresholdL1(G, α) = sign(G)·max(|G| - α, 0), α = lambda_l1
         ↓
Step 7: Update predictions: F(x) = F(x) + η × tree(x)
         ↓
Repeat Steps 3-7 for n_estimators
```

### Visual Example: Regression with LightGBM

Let's predict house prices using LightGBM:

```
Data:
Size (sqft): [1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000]
Price ($k):  [100,  120,  180,  200,  240,  260,  280,  350]
```

**Step 1: Build Histogram Bins**

```
Original size values: [1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000]

If max_bin = 4, create 4 bins:
Bin 0: size ≤ 1350  → [1000, 1200]
Bin 1: 1350 < size ≤ 1900  → [1500, 1800]
Bin 2: 1900 < size ≤ 2350  → [2000, 2200]
Bin 3: size > 2350  → [2500, 3000]

Binned representation: [0, 0, 1, 1, 2, 2, 3, 3]

Advantage: Instead of checking 7 possible splits, only check 3!
```

**Step 2: Initialize**

```
F₀(x) = mean(price) = 216.25
Current predictions: [216.25] × 8
Residuals: [-116.25, -96.25, -36.25, -16.25, 23.75, 43.75, 63.75, 133.75]
```

**Step 3: Build First Tree (Leaf-wise Strategy)**

Traditional Level-wise (XGBoost):
```
1. Split all data at root
2. Split both children
3. Continue level by level

Result: Balanced tree
Depth 2:      Root
            /      \
           L1      R1
          /  \    /  \
         L2  R2  L3  R3
```

LightGBM Leaf-wise:
```
1. Split root → creates L1, R1
2. Find which leaf (L1 or R1) has max gain
3. Split only that leaf
4. Repeat

Result: Asymmetric but optimal tree
         Root (Gain=100)
        /    \
       L1    R1 (Gain=80) ← Split this next!
            /  \
           R2  R3 (Gain=60) ← Then this
              /  \
             R4  R5
```

**Iteration 1: First Split**

```
Calculate gradients and hessians:
For squared loss: g = pred - y, h = 1
(Careful: the gradient is the NEGATIVE of the residual listed in Step 2.
 Cheap houses are over-predicted, so their gradient is POSITIVE.)

  g = 216.25 - y
    = [116.25, 96.25, 36.25, 16.25, -23.75, -43.75, -63.75, -133.75]

Sample data after binning:
Bin | Count | G_sum  | H_sum
----|-------|--------|-------
 0  |   2   |  212.5 |   2
 1  |   2   |   52.5 |   2
 2  |   2   |  -67.5 |   2
 3  |   2   | -197.5 |   2

Try split at Bin ≤ 1 (size ≤ 1900):
  Left:  G_L = 265, H_L = 4
  Right: G_R = -265, H_R = 4

Gain (with λ=1, and no ½ - see the note in Step 5 above):
  = (265)²/(4+1) + (-265)²/(4+1) - 0²/(8+1)
  = 70225/5 + 70225/5 - 0
  = 14045 + 14045
  = 28090

Leaf weights:
  w_left  = -(265)/(4+1) = -53.0
  w_right = -(-265)/(4+1) = 53.0

Update predictions:
  Small houses: 216.25 + 0.1 × (-53.0) = 210.95
  Large houses: 216.25 + 0.1 × 53.0    = 221.55

Sanity check: the small houses really average 150.0 and the large ones 282.5,
so the first tree must push the small group DOWN and the large group UP. It does.
The minus sign in w* = -G/(H+λ) is what makes the tree step against the gradient.
```

**Why Leaf-wise is Faster:**

```
Level-wise (XGBoost):
Iteration 1: Split 1 node
Iteration 2: Split 2 nodes
Iteration 3: Split 4 nodes
Total: 1 + 2 + 4 = 7 splits for depth 3

Leaf-wise (LightGBM):
Always split only 1 node (the best one)
Total: 3 splits for 3 iterations
→ More efficient, better loss reduction
```

---

## The Mathematical Foundation

### 1. Objective Function

LightGBM optimizes the same regularized objective as XGBoost:

```
Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₜ)

Where:
- L(yᵢ, ŷᵢ) = loss function
- Ω(fₜ) = regularization for tree t
- Ω(f) = γT + ½λΣ(w²)
  - γ: penalty for number of leaves
  - λ: L2 regularization on leaf weights
  - T: number of leaves
```

### 2. Taylor Expansion

Second-order approximation of the loss function:

```
L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾ + fₜ(xᵢ)) ≈ L(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢfₜ(xᵢ) + ½hᵢfₜ²(xᵢ)

Where:
- gᵢ = ∂L/∂ŷ⁽ᵗ⁻¹⁾ (first-order gradient)
- hᵢ = ∂²L/∂ŷ⁽ᵗ⁻¹⁾² (second-order gradient, hessian)
```

For squared loss (L2):
```
L = ½(y - ŷ)²
g = ŷ - y
h = 1
```

For log loss (binary classification):
```
L = -[y·log(p) + (1-y)·log(1-p)]
where p = sigmoid(ŷ)
g = p - y
h = p(1 - p)
```

### 3. Split Gain Calculation

For a given split, the gain is:

```
score(G, H) = ThresholdL1(G, α)² / (H + λ)

Gain = score(G_L, H_L) + score(G_R, H_R) - score(G_L+G_R, H_L+H_R) - γ

Where:
- G_L = Σ gᵢ for samples in left child
- H_L = Σ hᵢ for samples in left child
- G_R = Σ gᵢ for samples in right child
- H_R = Σ hᵢ for samples in right child
- λ = L2 regularization (lambda_l2)
- α = L1 regularization (lambda_l1), applied through ThresholdL1 below
- γ = minimum gain to split (min_gain_to_split)
```

**Interpretation:**
- Higher gain = better split
- First two terms: scores of children
- Third term: score of parent
- γ: complexity penalty (discourage unnecessary splits)

**Why no ½?** XGBoost writes this gain with a leading ½, which falls out of the
Taylor expansion. LightGBM's `GetLeafSplitGain` drops it, so **when γ = 0** a
LightGBM gain is exactly twice the XGBoost one for the same split. Both
conventions subtract γ *un-halved* though, so writing `r` for the un-penalized
`score(G_L,H_L) + score(G_R,H_R) - score(G_P,H_P)`, they read `r - γ` and
`r/2 - γ` and the ratio drifts above 2 as soon as γ > 0. Concretely, on the root
split of `np.random.seed(7); X = np.random.randn(300, 4);
y = 3*X[:,0] - 2*X[:,1] + X[:,2] + np.random.randn(300)*0.5` with `lambda_l2=1`,
`num_leaves=31` and `min_data_in_leaf=5`, that un-penalized `r` is 1459.8623, so
`(r - γ) / (r/2 - γ)` measures 2.000000 at γ = 0, 2.001372 at γ = 1 and 2.013890
at γ = 10.

Doubling is monotone, so within one node it never changes *which* candidate wins.
It does change what a given `min_gain_to_split` means, and therefore which
candidates clear the `gain > 0` bar at all: on that same data, at γ = 3 with
`num_leaves=31` and `min_data_in_leaf=5`, the first tree takes 26 splits under the
LightGBM convention against 20 under the halved one.
This implementation follows LightGBM and omits the ½. `_calculate_gain` in
`_18_lightgbm.py` is a line-for-line transcription of the formula above,
including the `- γ`.

**Scale note.** This repo's `17. XGBoost/_17_xgboost.py` uses the halved
convention, so the two penalties live on different scales: `gamma=g` there is the
same pruning *threshold* as `min_gain_to_split=2g` here, since both admit a split
exactly when the un-penalized `r` exceeds `2g`. Verified inside this class -
halving its gain and pruning at `g` reproduces the un-halved gain pruning at `2g`,
with identical split counts and identical predictions. That is a statement about
the two conventions, not about the two files: `_17_xgboost.py` grows level-wise on
exact midpoints of raw values, this one grows leaf-wise on histogram bins, so
their trees differ regardless of how the penalty is scaled.

### 4. Optimal Leaf Weight

The optimal weight for a leaf is:

```
w* = -ThresholdL1(G_j, α) / (H_j + λ)

Where:
- G_j = Σ gᵢ for samples in leaf j
- H_j = Σ hᵢ for samples in leaf j
- λ = L2 regularization (lambda_l2)
- α = L1 regularization (lambda_l1)

This weight minimizes the loss + regularization
```

**L1 soft-thresholding.** L2 shrinks a leaf weight smoothly by growing the
denominator. L1 does something different: it pulls the *numerator* toward zero by
a fixed amount, and once the gradient evidence in a leaf is weaker than α the
weight becomes exactly zero.

```
ThresholdL1(G, α) = sign(G) × max(|G| - α, 0)

                  = G - α   if G >  α
                    G + α   if G < -α
                    0        if |G| ≤ α
```

With α = 0 this collapses back to the familiar `w* = -G/(H+λ)`. The same
`ThresholdL1` is applied inside the gain, which is what keeps the split score and
the leaf weight consistent with each other. In the code this is
`_threshold_l1`, called by both `_calculate_leaf_weight` and `_calculate_gain`.

Worked check (from the `.py`, `lambda_l2=2.0`): a leaf with G = -10, H = 5 gets
`w* = 10/(5+2) = 1.4286`. Raise `lambda_l1` to 4 and it becomes
`w* = (10-4)/7 = 0.8571`. Raise it to 20 and the weight is exactly `0.0`.

### 5. Histogram-based Split Finding

Instead of considering all data points, LightGBM uses histograms:

```
Traditional: O(#data × #features)
Histogram: O(#bins × #features)

For each feature:
1. Create histogram with max_bin buckets
2. Accumulate G and H for each bin:
   H[k] = {G_sum: Σgᵢ, H_sum: Σhᵢ} for samples in bin k

3. Find best split by scanning bins:
   For threshold at bin k:
     G_L = Σ H[i].G_sum for i ≤ k
     H_L = Σ H[i].H_sum for i ≤ k
     G_R = G_total - G_L
     H_R = H_total - H_L
     Calculate Gain

Speedup: O(#data × #features) → O(#bins × #features)
If #bins = 255 and #data = 1,000,000: ~4000x fewer operations!
```

### 6. Leaf-wise vs Level-wise Growth

**Level-wise (XGBoost):**
```
Strategy: Split all leaves at current level
Complexity: O(2^depth) splits per iteration
Advantage: Balanced trees, easier to parallelize by level
Disadvantage: May waste computation on low-gain splits
```

**Leaf-wise (LightGBM):**
```
Strategy: Split only the leaf with maximum gain
Complexity: O(num_leaves) splits total
Advantage: Better loss reduction, faster convergence
Disadvantage: Can grow very deep, risk overfitting

Control overfitting with:
- max_depth: Limit tree depth
- num_leaves: Maximum number of leaves
- min_data_in_leaf: Minimum samples per leaf
```

### 7. Gradient-based One-Side Sampling (GOSS)

GOSS is LightGBM's technique to reduce data for training:

```
Idea: Not all instances are equally important
- Large gradients = poorly fitted → important
- Small gradients = well fitted → less important

Algorithm:
1. Sort instances by |gradient|
2. Keep top a% with large gradients
3. Randomly sample b% from remaining
4. Amplify small gradient samples by (1-a)/b
5. Build tree on this subset

Example:
100K samples → Keep 20K large gradient + 10K random small gradient
Train on 30K samples but approximate full 100K!

Speedup: ~3x with minimal accuracy loss
```

**In this implementation:** GOSS is implemented in `_goss_sample` and is
**off by default**. Turn it on with `boosting_type='goss'`, and tune it with
`top_rate` (the *a* above, default 0.2) and `other_rate` (the *b*, default 0.1).
GOSS and bagging are mutually exclusive, exactly as in the real library: with
`boosting_type='goss'` the `bagging_fraction` / `bagging_freq` parameters are
ignored. On small datasets GOSS usually costs a little accuracy for no speed
benefit - it pays off when there are far more rows than the trees need.

---

## Implementation Details

### Key Components

**1. Histogram Building**
```python
def _build_histogram(self, X):
    # For each feature
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        if len(unique_values) <= self.max_bin:
            # Few distinct values: every value gets its own bin. There are k
            # bins, so only k-1 cut points are needed -> the [:-1].
            thresholds = unique_values[:-1]
        else:
            # Otherwise cut at quantiles. The [1:-1] drops the 0th and 100th
            # percentile, which are the min and max and cannot separate anything.
            percentiles = np.linspace(0, 100, self.max_bin + 1)[1:-1]
            thresholds = np.percentile(feature_values, percentiles)
            thresholds = np.unique(thresholds)

        self.bin_thresholds.append(thresholds)

        # right=True means bin k holds thresholds[k-1] < x <= thresholds[k],
        # i.e. "x <= threshold" - exactly the test the trees use when they
        # predict. With numpy's default (right=False) a value sitting exactly ON
        # a threshold would fall into the NEXT bin, which merges the top two
        # values of every low-cardinality feature: a 0/1 flag would collapse
        # into one constant bin and could never be split on.
        X_binned[:, feature_idx] = np.digitize(feature_values, thresholds, right=True)
```

`max_bin` is a **ceiling, not a quota**: a 4-level ordinal produces 4 bins and a
binary flag produces 2, no matter what `max_bin` is set to.

**2. Leaf-wise Tree Growth (best-first)**

This is the part that makes the algorithm LightGBM rather than "XGBoost with
bins". There is **no recursion into left and then right**. Instead the growing
tree keeps a *frontier* of leaves, each already carrying the best split it could
make and that split's gain, and the loop repeatedly picks the best leaf in the
whole tree:

```python
def _build_tree_leaf_wise(self, X_binned, gradient, hessian, depth=0,
                          feature_indices=None):
    # The root starts life as a single leaf; score its best split
    root = make_leaf(all_rows)
    frontier = []                       # candidate leaves + their best split
    offer(root, all_rows, depth)        # push it if it is splittable

    n_leaves = 1
    while frontier and n_leaves < self.num_leaves:
        # BEST-FIRST: split the highest-gain leaf ANYWHERE in the tree
        candidate = pop_max_gain(frontier)

        left_rows  = rows of the candidate where bin <= candidate.bin
        right_rows = the rest

        # Turn that leaf into a split node IN PLACE, so its parent - which
        # already holds a reference to the dict - sees the new subtree
        candidate.node.update({'type': 'split', 'feature': ..., 'threshold': ...,
                               'gain': ..., 'left': left_leaf, 'right': right_leaf})
        n_leaves += 1                   # one leaf became two

        offer(left_leaf,  left_rows,  candidate.depth + 1)
        offer(right_leaf, right_rows, candidate.depth + 1)

    return root      # whatever is still on the frontier simply stays a leaf
```

`offer` is where the guards live: a leaf is never pushed onto the frontier if it
has fewer than `min_data_in_leaf` rows, if its hessian sum is below
`min_sum_hessian_in_leaf`, if `max_depth` is set and reached, or if its best
gain is not positive (`min_gain_to_split` is already subtracted inside
`_calculate_gain`).

**2b. Finding a split with the histogram**

`offer` calls `_find_best_split`, which is where the paper's speed comes from:

```python
def _find_best_split(self, X_binned, gradient, hessian, indices,
                     feature_indices, n_bins):
    for feature_idx in feature_indices:
        bins = X_binned[indices, feature_idx]

        # ONE pass over the leaf's rows builds the histogram      -> O(#data)
        hist_gradient = np.bincount(bins, weights=g, minlength=n_bins)
        hist_hessian  = np.bincount(bins, weights=h, minlength=n_bins)
        hist_count    = np.bincount(bins,            minlength=n_bins)

        # ONE prefix sum turns it into EVERY "bin <= k" split      -> O(#bins)
        G_left  = np.cumsum(hist_gradient);  G_right = G_total - G_left
        H_left  = np.cumsum(hist_hessian);   H_right = H_total - H_left

        # Score every threshold at once, mask the invalid ones, take the max
        gains = self._calculate_gain(G_left, H_left, G_right, H_right)
```

The right child costs nothing: it is the parent minus the left. The naive
alternative - rebuild a boolean mask and re-sum the gradients for every candidate
threshold - is `O(#data x #bins)` instead of `O(#data + #bins)`, which at
`max_bin=255` is roughly 85x more arithmetic per feature per node.

**3. Prediction with Binned Features**
```python
def predict(self, X):
    # Apply learned binning
    X_binned = apply_binning(X, bin_thresholds)
    
    # Start with base prediction
    predictions = base_score
    
    # Add each tree's contribution
    for tree in trees:
        predictions += learning_rate * predict_tree(tree, X_binned)
    
    # For classification, apply sigmoid
    if objective == 'binary':
        predictions = sigmoid(predictions)
```

---

## Step-by-Step Example

Let's work through a complete example: predicting if a customer will buy (classification).

### Dataset

```
Customer Data:
ID | Age | Income($k) | Website_visits | Previous_purchases | Buy?
---|-----|------------|----------------|--------------------| ----
1  | 25  |    30      |      2         |         0          |  0
2  | 35  |    50      |      5         |         1          |  1
3  | 45  |    70      |      8         |         3          |  1
4  | 28  |    35      |      1         |         0          |  0
5  | 50  |    90      |     12         |         5          |  1
6  | 32  |    45      |      6         |         2          |  1
7  | 22  |    25      |      1         |         0          |  0
8  | 55  |   100      |     15         |         8          |  1
```

### Step 1: Build Histograms

```
Age bins (max_bin=2):
  Bin 0: Age ≤ 35 → [25, 35, 28, 32, 22]
  Bin 1: Age > 35 → [45, 50, 55]

Income bins (max_bin=2):
  Bin 0: Income ≤ 52.5 → [30, 50, 35, 45, 25]
  Bin 1: Income > 52.5 → [70, 90, 100]

Website_visits bins (max_bin=2):
  Bin 0: Visits ≤ 5.5 → [2, 5, 1, 1, 6, 1]
  Bin 1: Visits > 5.5 → [8, 12, 15]

Previous_purchases bins (max_bin=2):
  Bin 0: Purchases ≤ 1.5 → [0, 1, 0, 0, 2, 0]
  Bin 1: Purchases > 1.5 → [3, 5, 8]

Result: Continuous features → Integer bins (0 or 1)
```

### Step 2: Initialize

```
Target: Buy? [0, 1, 1, 0, 1, 1, 0, 1]
Positive rate: p = 5/8 = 0.625

For binary classification:
base_score = log(p / (1-p)) = log(0.625 / 0.375) = log(1.667) = 0.51

Initial predictions (log-odds): [0.51] × 8
Initial probabilities: sigmoid(0.51) = 0.625 for all
```

### Step 3: Calculate Gradients and Hessians

```
For binary log loss:
g = p - y
h = p(1-p)

Sample 1: y=0, p=0.625
  g₁ = 0.625 - 0 = 0.625
  h₁ = 0.625 × 0.375 = 0.234

Sample 2: y=1, p=0.625
  g₂ = 0.625 - 1 = -0.375
  h₂ = 0.625 × 0.375 = 0.234

All gradients: [0.625, -0.375, -0.375, 0.625, -0.375, -0.375, 0.625, -0.375]
All hessians: [0.234, 0.234, 0.234, 0.234, 0.234, 0.234, 0.234, 0.234]
```

### Step 4: Find Best Split (Histogram-based)

```
Try splitting on Website_visits (binned):
  Bin 0 (Visits ≤ 5.5): Samples [1,2,4,6,7] → indices [0,1,3,5,6]
  Bin 1 (Visits > 5.5): Samples [3,5,8] → indices [2,4,7]

Left (Bin 0):
  G_L = 0.625 + (-0.375) + 0.625 + (-0.375) + 0.625 = 1.125
  H_L = 0.234 × 5 = 1.170
  Samples: 5, Buyers: 2 (40%)

Right (Bin 1):
  G_R = -0.375 + (-0.375) + (-0.375) = -1.125
  H_R = 0.234 × 3 = 0.702
  Samples: 3, Buyers: 3 (100%)

Calculate gain (λ=1):
  Score_L = (1.125)² / (1.170 + 1) = 1.266 / 2.170 = 0.583
  Score_R = (-1.125)² / (0.702 + 1) = 1.266 / 1.702 = 0.744
  Score_P = (0)² / (1.872 + 1) = 0 / 2.872 = 0
  
  Gain = 0.583 + 0.744 - 0 = 1.327     (no ½ - LightGBM convention)
```

Try other features similarly and pick best gain.

### Step 5: Create Leaf Weights

```
Assume Website_visits split is best.

Left leaf weight:
  w_left = -G_L / (H_L + λ) = -1.125 / (1.170 + 1) = -0.518

Right leaf weight:
  w_right = -G_R / (H_R + λ) = -(-1.125) / (0.702 + 1) = 0.661

Interpretation:
- Left: Decrease log-odds by 0.518 → lower probability of buying
- Right: Increase log-odds by 0.661 → higher probability of buying
```

### Step 6: Update Predictions

```
Learning rate η = 0.1

Samples in left leaf [1,2,4,6,7]:
  Old log-odds: 0.51
  New log-odds: 0.51 + 0.1 × (-0.518) = 0.51 - 0.052 = 0.458
  New probability: sigmoid(0.458) = 0.613

Samples in right leaf [3,5,8]:
  Old log-odds: 0.51
  New log-odds: 0.51 + 0.1 × 0.661 = 0.51 + 0.066 = 0.576
  New probability: sigmoid(0.576) = 0.640
```

### Step 7: Continue Building Trees

```
Iteration 2: Calculate new gradients based on updated predictions
Iteration 3: Build another tree
...
Iteration 100: Final model

Final prediction for new customer [Age=40, Income=60, Visits=10, Purchases=4]:
  1. Bin features: [1, 1, 1, 1]
  2. Start with base_score: 0.51
  3. Add tree 1: 0.51 + 0.1×tree1 = ...
  4. Add tree 2: ... + 0.1×tree2 = ...
  ...
  100. Final log-odds: 2.34
  101. Convert to probability: sigmoid(2.34) = 0.912 → Predict: Buy!
```

---

## Real-World Applications

### 1. E-commerce: Click-Through Rate (CTR) Prediction

**Problem**: Predict if user will click on an ad

**Why LightGBM?**
- Millions of users, fast prediction needed
- Many features (user profile, ad features, context)
- Need to retrain frequently with new data

**Features**:
```
User: age, gender, location, device, browsing_history
Ad: category, position, format, bid_price
Context: time_of_day, day_of_week, season
Interactions: user_interest × ad_category
```

**Benefits**:
- Fast training: Retrain daily with 100M samples
- Fast prediction: Serve 1000s predictions per second
- High accuracy: 2-3% CTR improvement = millions in revenue

### 2. Finance: Credit Risk Assessment

**Problem**: Predict loan default probability

**Why LightGBM?**
- Handle mixed data types (numerical, categorical)
- Interpret feature importance (regulatory requirement)
- High accuracy needed (cost of false negatives is huge)

**Features**:
```
Demographics: age, income, employment_years
Credit: credit_score, debt_to_income, delinquencies
Loan: amount, term, purpose, interest_rate
```

**Benefits**:
- Better risk estimation → reduce defaults by 15-20%
- Fast enough for real-time approval decisions
- Feature importance helps explain decisions

### 3. Healthcare: Disease Risk Prediction

**Problem**: Predict patient risk for disease

**Why LightGBM?**
- Handle missing values well (common in medical data) - a property of the
  *production library*; the from-scratch class in this folder does not route
  NaN to a learned default side, so impute before calling `fit`
- Good with high-dimensional sparse data
- Provides probability scores, not just binary yes/no

**Features**:
```
Vitals: blood_pressure, heart_rate, BMI, temperature
Labs: glucose, cholesterol, hemoglobin
History: previous_conditions, family_history, medications
Lifestyle: smoking, exercise, diet
```

**Benefits**:
- Early detection → better patient outcomes
- Risk stratification → allocate resources efficiently
- Faster than deep learning, easier to interpret

### 4. Retail: Demand Forecasting

**Problem**: Predict product sales for inventory planning

**Why LightGBM?**
- Time series with many external features
- Need forecasts for thousands of products
- Training speed crucial for daily updates

**Features**:
```
Historical: sales_lag_1, sales_lag_7, sales_lag_30
Calendar: day_of_week, month, holiday, season
Promotion: discount_percent, ad_spend
External: weather, competitor_price, economic_indicators
```

**Benefits**:
- 10-15% improvement in forecast accuracy
- Reduce stockouts and overstock
- Train 1000s of models (one per product) quickly

---

## Understanding the Code

### Core Class Structure

```python
class LightGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, 
                 num_leaves=31, ...):
        # Key parameters
        self.num_leaves = num_leaves  # Max leaves per tree
        self.max_bin = max_bin  # Histogram bins
        self.learning_rate = learning_rate
        # ... more parameters
        
    def fit(self, X, y):
        # 1. Build histograms
        # 2. Initialize predictions
        # 3. Train trees sequentially
        
    def predict(self, X):
        # 1. Apply binning
        # 2. Accumulate tree predictions
        # 3. Convert to probabilities if classification
```

### Key Methods Explained

**1. Histogram Building**
```python
def _build_histogram(self, X):
    """
    Convert continuous features to discrete bins
    
    Why: Dramatically speeds up split finding
    - Original: Try every unique value
    - Histogram: Try only bin boundaries
    
    Example: 
      Feature values: [1.2, 1.5, 1.8, 2.1, 2.4, 2.7]
      With max_bin=3: Bin 0 (<1.7), Bin 1 (1.7-2.3), Bin 2 (>2.3)
      Split candidates: 2 instead of 5
    """
```

**2. Leaf-wise Tree Building**
```python
def _build_tree_leaf_wise(self, X_binned, gradient, hessian, depth=0,
                          feature_indices=None):
    """
    Build tree by splitting the best leaf first (best-first / leaf-wise)

    How: keep a frontier of candidate leaves, each with its best split
    already scored, and repeatedly split the one with the highest gain
    anywhere in the tree. Stop at num_leaves leaves.

    Why: Better than level-wise
    - Focuses computation on high-gain splits
    - Converges faster with fewer leaves

    Danger: Can overfit if not controlled
    - Use max_depth to limit depth
    - Use num_leaves to limit total leaves
    - Use min_data_in_leaf for minimum samples
    """
```

Because the budget is counted in *leaves*, two trees with the same `num_leaves`
can have very different depths. Run `python _18_lightgbm.py` and Demo 3 prints
exactly this: with `num_leaves=15` the trees carry 10-13 leaves but reach depth
4-9, which a balanced level-wise tree of the same size never would.

**3. Gradient and Hessian Calculation**
```python
def _compute_gradient_hessian(self, y_true, y_pred):
    """
    Calculate first and second derivatives of loss
    
    Why use hessian (second derivative)?
    - Better approximation of loss function
    - More accurate optimization direction
    - Faster convergence
    
    For regression (squared loss):
      gradient = pred - y (how far off)
      hessian = 1 (constant curvature)
      
    For classification (log loss):
      gradient = p - y (probability error)
      hessian = p(1-p) (uncertainty)
    """
```

**4. Gain Calculation**
```python
def _calculate_gain(self, gradient_left, hessian_left,
                    gradient_right, hessian_right):
    """
    Calculate improvement from split
    
    Formula: score(G_L,H_L) + score(G_R,H_R) - score(G_P,H_P) - min_gain_to_split
    where    score(G, H) = ThresholdL1(G, lambda_l1)² / (H + lambda_l2)

    Interpretation:
    - First two terms: Quality of children
    - Third term: Quality of parent
    - Difference: Improvement from split
    - λ (lambda_l2): Regularization penalty in the denominator
    - α (lambda_l1): Soft-threshold applied to G before squaring

    No leading ½: that is XGBoost's convention, not LightGBM's.
    The returned value is NET of min_gain_to_split, and it is what gets
    stored in node['gain'] and summed by get_feature_importance('gain').

    Higher gain = better split
    """
```

### Important Parameters

**Tree Structure:**
```python
num_leaves=31           # Max leaves (main complexity control)
max_depth=-1           # Max depth (-1 = unlimited)
min_data_in_leaf=20    # Min samples per leaf
min_sum_hessian_in_leaf=1e-3  # Min sum of hessians per leaf
                       # For regression h=1, so this is "min rows" again;
                       # for binary h=p(1-p), so it also rejects leaves made
                       # only of already-confident rows. XGBoost calls it
                       # min_child_weight.
```

**Task:**
```python
objective='regression'  # 'regression' -> squared loss, score() returns R2
                        # 'binary'     -> log loss, score() returns accuracy,
                        #                 and predict_proba(X) becomes available
```

**Learning:**
```python
learning_rate=0.1       # Shrinkage (lower = more robust)
n_estimators=100       # Number of trees
```

**Speed vs Accuracy:**
```python
max_bin=255            # Histogram bins
                       # Higher = more accurate but slower
                       # 255 is LightGBM default
                       # Try 63 or 127 for speed
```

**Regularization:**
```python
lambda_l1=0.0          # L1 regularization
lambda_l2=0.0          # L2 regularization
min_gain_to_split=0.0  # Min gain to split (like gamma)
```

**Sampling:**
```python
feature_fraction=1.0    # Column subsampling: one feature subset per TREE
bagging_fraction=1.0   # Row subsampling
bagging_freq=0         # Re-draw the row subsample every k iterations
                       # (the same subsample is reused in between)

boosting_type='gbdt'   # 'goss' switches to Gradient-based One-Side Sampling
top_rate=0.2           # GOSS: fraction of largest-|gradient| rows always kept
other_rate=0.1         # GOSS: fraction sampled from the rest, amplified by
                       #       (1 - top_rate) / other_rate
```

**Reproducibility:**
```python
random_state=None       # None -> use numpy's global RNG (np.random.seed still works)
                        # int  -> private RandomState, unaffected by other code
```

---

## Model Evaluation

### Metrics to Use

**Regression:**
```python
# R2 Score (coefficient of determination)
r2 = model.score(X_test, y_test)
print(f"R2: {r2:.4f}")  # 1.0 is perfect, 0.0 is baseline

# Mean Absolute Error
predictions = model.predict(X_test)
mae = np.mean(np.abs(y_test - predictions))
print(f"MAE: {mae:.2f}")

# Root Mean Squared Error
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
print(f"RMSE: {rmse:.2f}")
```

**Classification:**
```python
# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

# Class probabilities: predict() returns P(class=1) for objective='binary',
# and predict_proba() returns both columns, [P(class=0), P(class=1)]
proba = model.predict_proba(X_test)
for i in range(3):
    print(f"true={y_test[i]}  P(0)={proba[i, 0]:.3f}  P(1)={proba[i, 1]:.3f}")

# Confusion Matrix
predictions = model.predict(X_test)          # == proba[:, 1]
predicted_classes = (predictions >= 0.5).astype(int)

# Calculate metrics
TP = sum((predicted_classes == 1) & (y_test == 1))
FP = sum((predicted_classes == 1) & (y_test == 0))
FN = sum((predicted_classes == 0) & (y_test == 1))
TN = sum((predicted_classes == 0) & (y_test == 0))

precision = TP / (TP + FP)  # Of predicted positives, how many correct?
recall = TP / (TP + FN)     # Of actual positives, how many found?
f1 = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1: {f1:.4f}")
```

### Hyperparameter Tuning

**Start with Defaults:**
```python
model = LightGBM(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=20
)
```

**Tune num_leaves (Most Important!):**
```python
# Try: 7, 15, 31, 63, 127
# Smaller: Less overfitting, may underfit
# Larger: More complex, may overfit

np.random.seed(42)
X = np.random.randn(200, 5)
y = 2*X[:, 0] - 3*X[:, 1] + X[:, 2] + np.random.randn(200)*0.5
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

for num_leaves in [7, 15, 31, 63]:
    model = LightGBM(n_estimators=100, learning_rate=0.1,
                     num_leaves=num_leaves, min_data_in_leaf=5)
    model.fit(X_train, y_train)
    train = model.score(X_train, y_train)
    test = model.score(X_test, y_test)
    print(f"num_leaves={num_leaves:3d}: train R2 {train:.4f}  "
          f"test R2 {test:.4f}  gap {train-test:.4f}")
```

Real output:
```
num_leaves=  7: train R2 0.9976  test R2 0.9176  gap 0.0800
num_leaves= 15: train R2 0.9994  test R2 0.8930  gap 0.1064
num_leaves= 31: train R2 0.9991  test R2 0.8714  gap 0.1276
num_leaves= 63: train R2 0.9991  test R2 0.8714  gap 0.1276
```

The train-test gap widens as the budget grows: that is overfitting, and on this
dataset `num_leaves=7` is the right answer. The last two rows are identical
because with 150 training rows and `min_data_in_leaf=5` the trees run out of
splittable leaves at around 30 - once another guard binds first, raising
`num_leaves` does nothing at all.

**Tune learning_rate and n_estimators Together:**
```python
# Lower learning_rate needs more n_estimators
# Common pairs:
#   lr=0.1, n_estimators=100
#   lr=0.05, n_estimators=200
#   lr=0.01, n_estimators=1000

model = LightGBM(learning_rate=0.05, n_estimators=200)
```

**Add Regularization if Overfitting:**
```python
model = LightGBM(
    num_leaves=31,
    min_data_in_leaf=20,      # Increase to 50-100
    lambda_l2=1.0,            # Add L2 regularization
    min_gain_to_split=0.1     # Require minimum gain
)
```

**Use Feature/Data Sampling:**
```python
model = LightGBM(
    feature_fraction=0.8,     # Use 80% features per tree
    bagging_fraction=0.8,     # Use 80% data per iteration
    bagging_freq=5            # Apply every 5 iterations
)
```

### Feature Importance

```python
# A runnable version: visits and purchases drive y, age is irrelevant
np.random.seed(0)
n = 300
age = np.random.uniform(20, 60, n)
income = np.random.uniform(20, 120, n)
visits = np.random.uniform(0, 20, n)
purchases = np.random.uniform(0, 10, n)
X_train = np.column_stack([age, income, visits, purchases])
y_train = 3*visits + 2*purchases + 0.3*income + np.random.randn(n)*0.5

model = LightGBM(n_estimators=100, learning_rate=0.1,
                 num_leaves=31, min_data_in_leaf=10)
model.fit(X_train, y_train)

# Get importance
importance = model.get_feature_importance('gain')

# Display (ASCII bars - a Windows console cannot print block characters)
feature_names = ['age', 'income', 'visits', 'purchases']
for name, imp in sorted(zip(feature_names, importance),
                       key=lambda x: x[1], reverse=True):
    print(f"{name:15s}: {imp:.4f} {'#'*int(imp*50)}")
```

Real output:
```
visits         : 0.7058 ###################################
income         : 0.2223 ###########
purchases      : 0.0713 ###
age            : 0.0006
```

`visits` has the largest coefficient *and* the widest range, so it dominates the
gain. `age` does not enter the formula at all and lands at essentially zero -
which is the check to run whenever you suspect a feature is being memorized.
Note that `'gain'` importances are net of `min_gain_to_split`, so raising that
parameter lowers every stored gain by that amount *and* prunes the low-gain
splits away entirely - the totals move by more than a constant shift.

### Avoiding Overfitting

**Signs of Overfitting:**
```python
train_score = model.score(X_train, y_train)  # 0.95
test_score = model.score(X_test, y_test)     # 0.75
# Large gap = overfitting!
```

**Solutions:**

1. **Reduce Model Complexity:**
```python
# Decrease num_leaves
model = LightGBM(num_leaves=15)  # Was 63

# Limit depth
model = LightGBM(max_depth=5)

# Increase min_data_in_leaf
model = LightGBM(min_data_in_leaf=50)  # Was 20
```

2. **Add Regularization:**
```python
model = LightGBM(
    lambda_l2=1.0,           # L2 penalty
    min_gain_to_split=0.1    # Min gain required
)
```

3. **Use Sampling:**
```python
model = LightGBM(
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5
)
```

4. **Early Stopping:**
```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20
)
```

---

## Simplifications vs. canonical LightGBM

This folder is a teaching implementation. It matches the real library on the
things that define the algorithm - histogram binning, best-first leaf-wise growth
bounded by `num_leaves`, the second-order gain, L1 soft-thresholding, and GOSS -
and it deliberately leaves out three things. They are listed here so that nothing
in this guide reads as a promise the code does not keep.

One caveat on the word *matches*: the correspondences claimed here are to the
published algorithm and to LightGBM's documented formulas, checked by reading
them - nothing in this folder has ever been diffed numerically against the
`lightgbm` package itself, which is not a dependency of this repo.

### 1. EFB: Exclusive Feature Bundling - NOT implemented

**What canonical LightGBM does.** In a sparse, high-dimensional dataset (one-hot
encoded categories, bag-of-words) most feature pairs are *mutually exclusive*:
they are rarely non-zero on the same row. EFB builds a conflict graph whose nodes
are features and whose edge weights count rows where both are non-zero, greedily
colours it so that features in one bundle conflict on at most
`max_conflict_rate` of rows, and then merges each bundle into a single feature by
**offsetting** each member's bins into a disjoint range. A bundle of a 3-bin and
a 4-bin feature becomes one 7-bin feature. Split finding then costs
`O(#bundles)` instead of `O(#features)`.

**Why it is omitted.** EFB touches binning, split finding, tree storage and
prediction all at once - the bin offsets have to be undone whenever a split is
mapped back to a real feature. Implementing it faithfully would roughly double
the size of the file while changing no prediction at all: EFB is a *speed and
memory* optimization, and on the few-hundred-row datasets used here it would
never be triggered.

**Practical consequence.** None for accuracy. On a genuinely sparse dataset with
thousands of one-hot columns this implementation is proportionally slower than
the real library.

### 2. Missing values - NOT handled

**What canonical LightGBM does.** NaN is a bin of its own. At every split the
learner tries sending the missing rows left and sending them right, keeps
whichever gives the higher gain, and stores that as the node's default direction
for unseen NaNs at prediction time.

**Why it is omitted.** It needs an extra branch in binning, in the histogram
scan, in the node schema and in `_predict_tree`.

**Practical consequence.** `fit` does `np.array(X, dtype=float)`, so a NaN
propagates: `np.digitize` places it in the last bin and its gradient poisons that
bin's histogram. **Impute before calling `fit`.**

### 3. Native categorical splits - NOT supported

**What canonical LightGBM does.** A feature declared categorical is not split as
`value <= k`. LightGBM sorts the categories by `G_k / H_k` (the Fisher 1958
trick) and finds the best *subset* split, so `{red, blue} vs {green}` is
reachable even though red and green are not adjacent.

**Why it is omitted.** It is a separate split-finding path with its own
regularization parameters (`cat_smooth`, `cat_l2`, `max_cat_threshold`).

**Practical consequence.** Every column here is treated as ordered. An ordinal
feature (season 1-4, day-of-week 1-7) still works well: it gets its own bin per
level, and a chain of `<=` splits can isolate any level. An *unordered*
categorical encoded as arbitrary integers will be handled worse than the real
library, because the arbitrary integer order constrains which groupings are
reachable. One-hot encode such features before using this implementation.

### What is NOT simplified

For the record, these are faithful:

| Piece | Status |
|-------|--------|
| Histogram binning + prefix-scan split search | Implemented (`_build_histogram`, `_find_best_split`) |
| Best-first leaf-wise growth bounded by `num_leaves` | Implemented (`_build_tree_leaf_wise`) |
| Second-order gain `G²/(H+λ)`, no XGBoost ½ | Implemented (`_calculate_gain`) |
| L1 soft-thresholding `ThresholdL1` | Implemented (`_threshold_l1`) |
| GOSS with the `(1-a)/b` amplification | Implemented, opt-in (`_goss_sample`) |
| Per-tree `feature_fraction`, cached bagging, early stopping | Implemented (`fit`) |

One optimization the paper describes that is also missing is **histogram
subtraction**: a sibling's histogram equals the parent's minus the child's, so
only the smaller child needs to be scanned. This implementation rebuilds both
children's histograms from scratch. It is a constant-factor speed trick with no
effect on the model, and building both keeps `_find_best_split` self-contained.

---

## LightGBM vs XGBoost vs Gradient Boosting

### Speed Comparison

```
Dataset: 1 Million samples, 100 features

Training Time:
├── Gradient Boosting: ~2 hours
├── XGBoost: ~15 minutes
└── LightGBM: ~2 minutes  ← 7-8x faster!

Why LightGBM is faster:
- Histogram-based split finding
- Leaf-wise growth (fewer splits)
- Better memory efficiency
```

### When to Use Each

**Use Gradient Boosting when:**
- Small dataset (<10K samples)
- Need simplicity and transparency
- Learning the fundamentals

**Use XGBoost when:**
- Medium dataset (10K-100K samples)
- Need highest accuracy
- Have time for extensive tuning
- Ecosystem support (wide adoption)

**Use LightGBM when:**
- Large dataset (>100K samples) ← Best choice!
- Speed is critical
- Memory is limited
- Many categorical features (production library only - see Simplifications)
- Need good default parameters

### Accuracy Comparison

```
Generally similar accuracy, but:

Small datasets (<10K):
XGBoost ≈ LightGBM ≈ Gradient Boosting

Large datasets (>100K):
LightGBM ≥ XGBoost > Gradient Boosting

Categorical features:
LightGBM > XGBoost (native categorical support - in the production library;
                    this from-scratch version treats every column as numeric)

Why LightGBM can be better:
- Leaf-wise growth finds better splits
- GOSS focuses on hard examples
- Less likely to underfit large datasets
```

---

## Summary

### Key Takeaways

1. **LightGBM = Speed + Efficiency**
   - Histogram-based learning → Fast split finding
   - Leaf-wise growth → Better accuracy with fewer leaves
   - Low memory usage → Can handle huge datasets

2. **Main Innovations**
   - **Histograms**: Bin continuous features → 10-20x speedup *(implemented)*
   - **Leaf-wise**: Split best leaf first → Better convergence *(implemented)*
   - **GOSS**: Sample based on gradients → Reduce data while keeping accuracy
     *(implemented, opt-in via `boosting_type='goss'`)*
   - **EFB**: Bundle sparse features → Reduce dimensions
     *(explained here, not implemented - see Simplifications)*

3. **Best Practices**
   ```python
   # Start here
   model = LightGBM(
       n_estimators=100,
       learning_rate=0.1,
       num_leaves=31,
       min_data_in_leaf=20
   )
   
   # If overfitting
   model = LightGBM(
       num_leaves=15,          # Reduce
       min_data_in_leaf=50,    # Increase
       lambda_l2=1.0,          # Add regularization
       feature_fraction=0.8    # Add randomness
   )
   
   # If underfitting
   model = LightGBM(
       num_leaves=63,          # Increase
       n_estimators=200,       # More trees
       learning_rate=0.05      # Lower rate, more trees
   )
   ```

4. **When to Use LightGBM**
   - ✅ Large datasets (>100K samples)
   - ✅ Many features (>100 features)
   - ✅ Need fast training
   - ✅ Limited memory
   - ✅ Categorical features (production library; not this teaching version)
   - ❌ Very small datasets (<1K samples) - use simpler models

### Next Steps

1. **Run the examples** in the `.py` file
2. **Try your own dataset** - start with default parameters
3. **Tune num_leaves** first - biggest impact
4. **Add regularization** if overfitting
5. **Compare with XGBoost** to see speed difference
6. **Study feature importance** to understand your data

---

## References and Further Learning

### Official Resources
- **LightGBM Documentation**: https://lightgbm.readthedocs.io/
- **Paper**: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NIPS 2017)
- **GitHub**: https://github.com/microsoft/LightGBM

### Key Concepts to Explore
- Histogram-based learning algorithms
- Leaf-wise vs level-wise tree growth
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- Distributed and parallel learning

### Related Algorithms
- XGBoost (main competitor, level-wise growth)
- CatBoost (handles categorical features differently)
- Gradient Boosting (foundation algorithm)
- Random Forests (alternative ensemble method)

---

**Remember**: LightGBM is "light" in memory and "heavy" in performance! Use it when you need speed without sacrificing accuracy. Happy learning! 🚀

---

*This guide is part of the "ML Algorithms from Scratch" series. For more algorithms, check out the repository!*
