# K-Nearest Neighbors (KNN) from Scratch: A Comprehensive Guide

Welcome to the world of K-Nearest Neighbors! 🎯 In this comprehensive guide, we'll explore one of the simplest yet most powerful machine learning algorithms. Think of it as the "birds of a feather flock together" algorithm!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is K-Nearest Neighbors?](#what-is-k-nearest-neighbors)
3. [How KNN Works](#how-knn-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Choosing the Right k](#choosing-the-right-k)
11. [Feature Scaling: Critical for KNN](#feature-scaling-critical-for-knn)
12. [Advantages and Limitations](#advantages-and-limitations)
13. [Complete Usage Example](#complete-usage-example)
14. [Optimizations and Variants](#optimizations-and-variants)
15. [Key Concepts to Remember](#key-concepts-to-remember)
16. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# KNN from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _5_knn.py  (the __main__ block runs this)
# Or copy the KNearestNeighbors class from _5_knn.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the KNearestNeighbors class here (from _5_knn.py) ----
# class KNearestNeighbors: ...

np.random.seed(42)

# ------ CLASSIFICATION: three overlapping Gaussian blobs ------
centers = np.array([[0.0, 0.0], [4.0, 4.0], [8.0, 0.0]])
n_per_class = 60
spread = 1.8
X = np.vstack([np.random.randn(n_per_class, 2) * spread + c for c in centers])
y = np.repeat([0, 1, 2], n_per_class)

# Shuffle BEFORE slicing, or the test set would be one whole class
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
X_train, X_test = X[:135], X[135:]
y_train, y_test = y[:135], y[135:]

clf = KNearestNeighbors(k=5, distance_metric='euclidean', task='classification')
clf.fit(X_train, y_train)

print(f"Train accuracy : {clf.score(X_train, y_train):.4f}")
print(f"Test  accuracy : {clf.score(X_test, y_test):.4f}")

# Column j of predict_proba corresponds to clf.classes_[j]
probas = clf.predict_proba(X_test)
preds = clf.predict(X_test)
print(f"\nprobability columns map to classes_ = {clf.classes_}")
for i in range(5):
    print(f"  true={y_test[i]}  pred={preds[i]}  "
          f"P0={probas[i,0]:.2f}  P1={probas[i,1]:.2f}  P2={probas[i,2]:.2f}")

# ------ REGRESSION: y = 3*sin(x) + noise ------
X_r = np.random.uniform(-3, 3, size=(200, 1))   # uniform, so a plain slice is safe
y_r = 3 * np.sin(X_r.ravel()) + np.random.randn(200) * 0.3

reg = KNearestNeighbors(k=5, task='regression')
reg.fit(X_r[:150], y_r[:150])

print(f"\nTrain R2 : {reg.score(X_r[:150], y_r[:150]):.4f}")
print(f"Test  R2 : {reg.score(X_r[150:], y_r[150:]):.4f}")

reg_preds = reg.predict(X_r[150:])
for i in range(5):
    print(f"  x={X_r[150+i,0]:5.2f}  true={y_r[150+i]:6.2f}  pred={reg_preds[i]:6.2f}")
```

Expected output:
```
Train accuracy : 0.9481
Test  accuracy : 0.9556

probability columns map to classes_ = [0 1 2]
  true=0  pred=0  P0=1.00  P1=0.00  P2=0.00
  true=1  pred=1  P0=0.00  P1=1.00  P2=0.00
  true=0  pred=1  P0=0.40  P1=0.60  P2=0.00
  true=0  pred=1  P0=0.40  P1=0.60  P2=0.00
  true=2  pred=2  P0=0.00  P1=0.00  P2=1.00

Train R2 : 0.9828
Test  R2 : 0.9806
  x=-0.70  true= -1.94  pred= -1.96
  x= 0.26  true=  1.31  pred=  0.55
  x= 2.44  true=  1.75  pred=  2.02
  x= 0.75  true=  2.58  pred=  2.06
  x=-2.30  true= -2.03  pred= -2.09
```

Notice rows 3 and 4 of the classification output: `P0=0.40, P1=0.60` means 2 of the
5 neighbors were class 0 and 3 were class 1. Those two points sit in the overlap
between the blobs, and the model gets them wrong — but it tells you it is unsure.
That is exactly what `predict_proba` is for.

Running `python _5_knn.py` directly also prints a third demo sweeping `k`, the
distance metric and the weighting scheme side by side.

---

## What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a **simple, intuitive, non-parametric algorithm** used for both classification and regression. It makes predictions based on the principle that similar things are near each other.

**Real-world analogy**: 
Imagine you move to a new neighborhood and want to know if it's safe. You ask your 5 nearest neighbors about crime rates. If 4 out of 5 say it's safe, you'd conclude it's probably a safe neighborhood. That's exactly how KNN works!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Instance-based, Non-parametric |
| **Learning Style** | Lazy learning (no training phase) |
| **Tasks** | Classification and Regression |
| **Decision** | Based on k nearest neighbors |
| **Key Parameter** | k (number of neighbors) |

### The Core Idea

```
"You are the average of the k closest examples to you"
```

KNN doesn't learn a model! Instead, it:
1. **Memorizes** all training data
2. **Finds** the k most similar examples when predicting
3. **Votes** (classification) or **averages** (regression) their labels

---

## How KNN Works

### The Algorithm in 5 Steps

```
Step 1: Store all training data (X_train, y_train)
         ↓
Step 2: For new point x, calculate distance to ALL training points
         ↓
Step 3: Find the k nearest neighbors (smallest distances)
         ↓
Step 4: Classification: Vote for most common class
        Regression: Average the values
         ↓
Step 5: Return prediction
```

### Visual Example

```
Training Data (2D):
    
    Class A: ●  ●  ●
    Class B: ■  ■  ■
    New Point: ?

Step 1: Calculate distances
    
    ●(2.1)  ●(3.5)  ●(5.2)
         ?
    ■(1.8)  ■(4.1)  ■(6.3)

Step 2: Sort by distance, pick k=3 nearest
    
    Nearest: ■(1.8), ●(2.1), ●(3.5)
    
Step 3: Vote (2 Class A, 1 Class B)
    
    Prediction: Class A ●
```

### Why "K-Nearest"?

The "K" in KNN is crucial:

```
k=1: Look at 1 nearest neighbor
     → Very flexible, sensitive to noise
     → High variance, low bias

k=3: Look at 3 nearest neighbors
     → More stable, some noise tolerance
     → Balanced

k=10: Look at 10 nearest neighbors
      → Very stable, robust to noise
      → Low variance, high bias
```

**Visual Comparison**:
```
k=1:  Decision boundary is wiggly, complex
      ●●●●●●●●●●
       ■■■■■■■■■■
      
k=5:  Decision boundary is smoother
      ●●●●●●●●●●
      ----------
      ■■■■■■■■■■
      
k=20: Decision boundary is very smooth
      ●●●●●●●●●●
      ==========
      ■■■■■■■■■■
```

---

## The Mathematical Foundation

### Distance Metrics

KNN relies on measuring "distance" between points. The most common metrics are:

#### 1. Euclidean Distance (L2)

The straight-line distance between two points:

```
d(x, y) = √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]
```

**Example**:
```python
Point A: [1, 2]
Point B: [4, 6]

d = √[(1-4)² + (2-6)²]
d = √[(-3)² + (-4)²]
d = √[9 + 16]
d = √25 = 5
```

**Visualization**:
```
    y
    6 |      B
    5 |     /
    4 |    /  d=5
    3 |   /
    2 | A
    1 |
    0 +----------- x
      0 1 2 3 4
```

**When to use**: Most cases, natural measure of distance

#### 2. Manhattan Distance (L1)

The city-block distance (sum of absolute differences):

```
d(x, y) = |x₁-y₁| + |x₂-y₂| + ... + |xₙ-yₙ|
```

**Example**:
```python
Point A: [1, 2]
Point B: [4, 6]

d = |1-4| + |2-6|
d = 3 + 4
d = 7
```

**Visualization**:
```
    y
    6 |      B
    5 |      ↑
    4 |      ↑  4 blocks up
    3 |      ↑
    2 | A→→→→↑  3 blocks right
    1 |
    0 +----------- x
      0 1 2 3 4
```

**When to use**: High-dimensional data, when features are independent

### Classification Decision Rule

For a new point x, find k nearest neighbors and predict:

```
ŷ = mode(y₁, y₂, ..., yₖ)
```

Where:
- ŷ = predicted class
- y₁, y₂, ..., yₖ = labels of k nearest neighbors
- mode = most frequent value

**Example**:
```
k = 5
Neighbor labels: [A, A, B, A, C]

Count: A=3, B=1, C=1
Prediction: A (majority vote)
```

### Regression Decision Rule

For regression, predict the average:

```
ŷ = (1/k) × Σ(y₁ + y₂ + ... + yₖ)
```

**Example**:
```
k = 3
Neighbor values: [100, 150, 125]

Prediction: (100 + 150 + 125) / 3 = 125
```

### Probability Estimation

KNN can also provide probability estimates:

```
P(class=c|x) = (number of neighbors with class c) / k
```

**Example**:
```
k = 5
Neighbor labels: [A, A, B, A, C]

P(A) = 3/5 = 0.60 (60%)
P(B) = 1/5 = 0.20 (20%)
P(C) = 1/5 = 0.20 (20%)
```

### Why Does Averaging Neighbors Work At All?

The rules above can look like folk wisdom — "similar things are similar, so take
the average". They are not. KNN is a **local, non-parametric estimator of a
conditional expectation**, and that is where its rules come from.

**What we actually want.** The best possible prediction at a point x, under
squared error, is the conditional mean:

```
f(x) = E[y | X = x]
```

For classification the same quantity, applied to the indicator of class c,
is the conditional class probability:

```
P(y = c | X = x) = E[ 1{y = c} | X = x ]
```

**Why we cannot compute it.** That expectation averages over all examples having
*exactly* the value x. In continuous feature space we have exactly zero such
examples — every training point is unique. The quantity we want is unobservable.

**KNN's move.** Relax "exactly at x" to "near x". If we assume f is reasonably
smooth — that points close together have similar targets — then averaging over a
small neighborhood approximates averaging at the point:

```
f̂(x) = (1/k) × Σ y_i  for x_i in N_k(x)     ≈  E[y | X = x]
```

where `N_k(x)` is the set of the k nearest training points. **That is exactly the
regression rule** `prediction = np.mean(k_nearest_labels)`. Apply the same
substitution to the class indicator and you get
`P(class=c|x) = (count of c among neighbors) / k` — **exactly `predict_proba`** —
and taking the most probable class gives the majority vote. All three rules the
implementation uses are the same idea applied to different targets.

**Why k controls bias and variance, precisely.** The approximation makes two
errors, and `k` trades one against the other:

| | Small k | Large k |
|---|---|---|
| **Neighborhood size** | tiny — genuinely local | wide — includes far-off points |
| **Bias** (is the neighborhood really "at x"?) | low | **high**: averages in regions where `f` differs |
| **Variance** (how noisy is the average?) | **high**: averaging k values cuts noise by only 1/√k | low |

That is the bias-variance story of KNN, derived rather than asserted. At `k=1`
the neighborhood is as local as possible (minimum bias) but the estimate is a
single noisy observation (maximum variance) — which is precisely why `k=1` scores
1.0000 on training data and worse on test data.

**Why it converges.** This is the theoretical result that justifies the whole
algorithm:

```
as n → ∞,  if  k → ∞  and  k/n → 0,  then  f̂(x) → E[y | X = x]
```

The two conditions are the two errors above. `k → ∞` kills the variance (we
average more points); `k/n → 0` keeps the neighborhood shrinking relative to the
data, so the bias vanishes too. Note they must hold *together* — k growing, but
more slowly than n. Satisfy both and KNN is **universally consistent** — it converges to the optimal
predictor for *any* underlying function, with no assumptions about its form. That
is a remarkably strong guarantee for such a simple rule, and it is what
"non-parametric" buys you.

The catch is the rate, not the guarantee: reaching a given accuracy needs a
number of samples that grows exponentially in the number of features. That is the
[curse of dimensionality](#seeing-the-curse-of-dimensionality), and it is the
price of assuming nothing.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class KNearestNeighbors:
    def __init__(self, k=5, distance_metric='euclidean', task='classification',
                 weights='uniform'):
        # ... validation of k, distance_metric, task, weights ...
        self.k = k
        self.distance_metric = distance_metric
        self.task = task
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None   # set by fit() for classification
```

### Core Methods

1. **`__init__(k, distance_metric, task, weights)`** - Initialize model
   - k: Number of neighbors to consider (must be >= 1)
   - distance_metric: 'euclidean' or 'manhattan'
   - task: 'classification' or 'regression'
   - weights: 'uniform' (every neighbor equal) or 'distance' (w = 1/d)
   - Invalid values are rejected **here**, not later inside `predict`

2. **`_calculate_distance(x1, x2)`** - Private helper method
   - Computes distance between two points
   - Supports multiple distance metrics
   - Returns a single float value

3. **`fit(X, y)`** - "Train" the model
   - Stores the training data as a **float copy** (see note below)
   - No actual learning happens (lazy learning)
   - O(1) time complexity
   - Sets `classes_` and rejects `k > n_samples`

4. **`_get_neighbors(x)`** - Private helper: the neighbor search
   - Distances from `x` to every training row, then the k smallest
   - Returns `(k_indices, k_distances)`
   - Shared by `_predict_single` and `predict_proba`, so the two can never disagree

5. **`_neighbor_weights(k_distances)`** - Private helper: voting weights
   - `'uniform'` -> all 1.0; `'distance'` -> 1/d, with d=0 taking all the weight

6. **`_predict_single(x)`** - Predict for one sample
   - Calls `_get_neighbors`, then votes (classification) or averages (regression)
   - Returns single prediction

7. **`predict(X)`** - Predict for multiple samples
   - Calls _predict_single for each sample
   - Returns array of predictions
   - Main prediction interface

8. **`predict_proba(X)`** - Get class probabilities
   - Only for classification tasks
   - Returns probability distribution over classes
   - Based on neighbor label frequencies
   - **Column `j` corresponds to `model.classes_[j]`** (the sorted unique labels).
     Never guess this ordering from context - read it off `classes_`.

9. **`score(X, y)`** - Calculate performance
   - Accuracy for classification
   - R² score for regression
   - Accuracy is in [0, 1]; R² is at most 1 and **can be negative** when the model
     predicts worse than the mean of `y`

### Why `fit` copies into a float array

`fit` does `self.X_train = np.array(X, dtype=float)` rather than storing `X`
directly. Both halves of that line fix a real bug:

- **`dtype=float`**: integer inputs would wrap around during `(x1 - x2)`. With
  `uint8` pixel data (the digit-recognition use case!), `255 - 240` is fine but
  `240 - 255` wraps to `241` instead of `-15`, so a far-away image looks like a
  near neighbor and the prediction is silently wrong.
- **`np.array` rather than `np.asarray`**: `np.array` always copies. Without the
  copy, editing your own array after calling `fit` would quietly change the
  model's predictions, because the model would be pointing at your memory.

---

## Step-by-Step Example

Let's walk through a complete example predicting **fruit type** based on weight and sweetness:

### The Data

```python
import numpy as np

# Features: [weight (grams), sweetness (1-10)]
X_train = np.array([
    [150, 8],   # Apple
    [170, 9],   # Apple
    [140, 7],   # Apple
    [350, 4],   # Orange
    [380, 5],   # Orange
    [340, 3],   # Orange
    [200, 9],   # Strawberry
    [180, 10],  # Strawberry
    [190, 8]    # Strawberry
])

# Labels: 0=Apple, 1=Orange, 2=Strawberry
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
```

### Training the Model

```python
model = KNearestNeighbors(k=3, task='classification')
model.fit(X_train, y_train)
```

**What happens internally**:
- The model simply stores X_train and y_train
- No coefficients are learned (unlike regression)
- Training is instantaneous!

### Making Predictions

```python
# New fruit to classify
X_test = np.array([[160, 8]])  # 160g, sweetness 8

# Step 1: Calculate distances to all training points
distances = [
    dist([160,8], [150,8]) = 10.0,    # Apple 1
    dist([160,8], [170,9]) = 10.05,   # Apple 2
    dist([160,8], [140,7]) = 20.02,   # Apple 3
    dist([160,8], [350,4]) = 190.04,  # Orange 1
    ...
]

# Step 2: Find k=3 nearest
Nearest: [10.0 (Apple), 10.05 (Apple), 20.02 (Apple)]

# Step 3: Vote
Classes: [0, 0, 0]
Prediction: 0 (Apple) ✓
```

### Complete Prediction Code

```python
# Predict for multiple samples
X_test = np.array([
    [160, 8],   # Should be Apple
    [360, 4],   # Should be Orange
    [185, 9]    # Should be Strawberry
])

predictions = model.predict(X_test)
print("Predicted classes:", predictions)
# Output: [0, 1, 2] (Apple, Orange, Strawberry)

# Get probabilities
probabilities = model.predict_proba(X_test)
print("\nProbabilities:")
for i, probs in enumerate(probabilities):
    print(f"Sample {i+1}: Apple={probs[0]:.2f}, Orange={probs[1]:.2f}, Strawberry={probs[2]:.2f}")
# Output:
# Sample 1: Apple=1.00, Orange=0.00, Strawberry=0.00
# Sample 2: Apple=0.00, Orange=1.00, Strawberry=0.00
# Sample 3: Apple=0.33, Orange=0.00, Strawberry=0.67
```

**Why isn't Sample 3 a clean 1.00?** This is a distance tie, and it is worth
tracing by hand. For the query `[185, 9]` the distances are:

| Training point | Label | Distance |
|---|---|---|
| `[180, 10]` | Strawberry | 5.0990 |
| `[190, 8]` | Strawberry | 5.0990 |
| `[170, 9]` | **Apple** | **15.0000** |
| `[200, 9]` | **Strawberry** | **15.0000** |

The first two neighbors are unambiguous, but the third slot is a **tie at exactly
15.0000** between an Apple and a Strawberry. Only one of them can be the third
neighbor, so the answer is 2 Strawberries + 1 Apple = `0.33 / 0.00 / 0.67`.

Which one wins? `np.argsort` is a **stable** sort, so it keeps the tied points in
their original array order, and `[170, 9]` (index 1) comes before `[200, 9]`
(index 6). The Apple takes the slot.

This is not a defect — it is what every KNN implementation must do, and the rule
has to be *deterministic* so that repeated runs agree. It is worth knowing that
even scikit-learn's own backends split on this exact query: `algorithm='kd_tree'`
and `'ball_tree'` return `0.33 / 0.00 / 0.67` (matching ours), while
`algorithm='brute'` returns `0.00 / 0.00 / 1.00`. Both backends compute the tied
distance as exactly `15.0` — the difference is not rounding. It is that brute's
k-smallest selection is not a *stable* sort, so it simply makes no promise about
which of two exactly-tied points it keeps. Ours does, because `argsort` is stable.

**The real lesson**: when a query is equidistant from two classes, the prediction
rests on a tie-break rather than on evidence. An even `k`, or duplicated points,
makes this more likely. Prefer odd `k` for binary problems, and treat a
`predict_proba` row that is nearly split as "the model does not know".

---

## Real-World Applications

### 1. **Recommender Systems**
Recommend products based on similar users:
- Input: User preferences, purchase history
- Output: Recommended products
- Example: "Customers like you also bought..."

### 2. **Image Recognition**
Classify images based on similar images:
- Input: Image features (pixels, edges, colors)
- Output: Object class (cat, dog, car, etc.)
- Example: "This image looks most like a cat"

### 3. **Medical Diagnosis**
Diagnose diseases based on similar patient profiles:
- Input: Symptoms, test results, medical history
- Output: Disease diagnosis
- Example: "Patient profile matches diabetes cases"

### 4. **Credit Risk Assessment**
Assess loan risk based on similar applicants:
- Input: Income, credit score, employment, debt
- Output: Risk level (low, medium, high)
- Example: "Similar profiles have 15% default rate"

### 5. **Handwriting Recognition**
Recognize handwritten digits:
- Input: Pixel intensities of handwritten digit
- Output: Digit (0-9)
- Example: "This handwriting looks like a '7'"

### 6. **Anomaly Detection**
Detect unusual patterns:
- Input: Transaction features, user behavior
- Output: Normal or anomalous
- Example: "This transaction differs from normal patterns"

### 7. **Real Estate Price Prediction**
Predict house prices based on similar properties:
- Input: Size, location, age, bedrooms
- Output: Estimated price
- Example: "Similar houses sold for $350k-$400k"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Distance Calculation

```python
def _calculate_distance(self, x1, x2):
    if self.distance_metric == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif self.distance_metric == 'manhattan':
        return np.sum(np.abs(x1 - x2))
```

**How it works**:
```python
# Euclidean example
x1 = [1, 2, 3]
x2 = [4, 5, 6]

diff = x1 - x2 = [-3, -3, -3]
squared = diff² = [9, 9, 9]
sum_squared = 27
distance = √27 ≈ 5.196

# Manhattan example
absolute_diff = |x1 - x2| = [3, 3, 3]
distance = sum([3, 3, 3]) = 9
```

**Why these metrics?**
- **Euclidean**: Natural measure, like using a ruler
- **Manhattan**: Useful in high dimensions, less affected by outliers

### 2. Finding Nearest Neighbors

This is the body of `_get_neighbors(x)`. Both `_predict_single` and
`predict_proba` call it, so there is exactly one neighbor-search implementation
and the two paths cannot drift apart:

```python
# Calculate all distances
distances = []
for x_train in self.X_train:
    distance = self._calculate_distance(x, x_train)
    distances.append(distance)

distances = np.array(distances)

# Sort and get k smallest
k_indices = np.argsort(distances)[:self.k]

return k_indices, distances[k_indices]
```

**Step-by-step**:
```python
# Example with k=3
distances = [5.2, 2.1, 8.3, 1.5, 4.7, 3.2]

# argsort returns indices that would sort the array
sorted_indices = [3, 1, 5, 4, 0, 2]
                  ↓  ↓  ↓
# Take first k=3
k_indices = [3, 1, 5]  # Points with distances 1.5, 2.1, 3.2
```

**A note on cost and on ties**: `np.argsort` sorts *all* n distances, which is
O(n log n), even though we only need the k smallest. `np.argpartition` would do
it in O(n), but `argsort` reads more clearly and this repo prefers clarity over
speed, so the code keeps `argsort` with a comment saying so. The sort being
**stable** is not incidental, though: it is what makes a distance tie break
toward the earlier training row, deterministically, every run.

### 3. Making Classification Predictions

```python
# Get labels of k nearest neighbors
k_nearest_labels = self.y_train[k_indices]

# Find most common class
unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
prediction = unique_labels[np.argmax(counts)]
```

**Example**:
```python
k_nearest_labels = [0, 1, 0]  # Classes of 3 nearest neighbors

# Count occurrences
unique_labels = [0, 1]
counts = [2, 1]

# Most common
argmax(counts) = 0  # Index of maximum count
prediction = unique_labels[0] = 0
```

**What happens on a vote tie?** With `k=4` and labels `[0, 0, 1, 1]` both counts
are 2. `np.unique` returns `unique_labels` in **sorted** order and `np.argmax`
returns the **first** maximum, so the tie goes to the smallest class label — here,
class 0. That is the same rule scikit-learn uses, and it is why odd `k` is
recommended for binary classification: it makes the tie impossible.

### 4. Making Regression Predictions

```python
# Get values of k nearest neighbors
# (same variable as the classification branch - it holds values, not labels, here)
k_nearest_labels = self.y_train[k_indices]

# w comes from self._neighbor_weights(k_distances):
#   weights='uniform'  -> all 1.0
#   weights='distance' -> 1/d
w = self._neighbor_weights(k_distances)

# Calculate mean. With uniform weights np.average IS np.mean;
# with weights='distance' it becomes sum(w_j * y_j) / sum(w_j).
prediction = np.average(k_nearest_labels, weights=w)
```

**Example**:
```python
k_nearest_labels = [100, 150, 125]

prediction = (100 + 150 + 125) / 3 = 125
```

### 5. Computing Probabilities

```python
# For each class, sum the weight sitting on that class
# (with weights='uniform' every w is 1.0, so this is just a count)
total_weight = np.sum(w)
for c in classes:
    prob = np.sum(w[k_nearest_labels == c]) / total_weight
    class_probs.append(prob)
```

Note the divisor is `total_weight` — the weight actually collected — and **not**
`self.k`. Under uniform weights the two are the same number, but dividing by the
real total is what guarantees every row sums to exactly 1.0 in all cases.

**Example**:
```python
k = 5
k_nearest_labels = [0, 0, 1, 0, 2]
classes = [0, 1, 2]

# Count for each class
count_0 = 3  → prob_0 = 3/5 = 0.6
count_1 = 1  → prob_1 = 1/5 = 0.2
count_2 = 1  → prob_2 = 1/5 = 0.2

probabilities = [0.6, 0.2, 0.2]
```

---

## Model Evaluation

### For Classification

#### 1. Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Example**:
```python
y_true = [0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1]

correct = 4
total = 5
accuracy = 4/5 = 0.8 (80%)
```

#### 2. Confusion Matrix

```
                Predicted
              0       1
Actual   0   [TN]    [FP]
         1   [FN]    [TP]
```

#### 3. Precision and Recall

```
Precision = TP / (TP + FP)  # Of predicted positives, how many correct?
Recall = TP / (TP + FN)     # Of actual positives, how many found?
```

### For Regression

#### R² Score (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)

where:
SS_res = Σ(y_true - y_pred)²  # Residual sum of squares
SS_tot = Σ(y_true - y_mean)²  # Total sum of squares
```

**Interpretation**:
- R² = 1.0: Perfect predictions
- R² = 0.8: 80% of variance explained
- R² = 0.0: No better than predicting mean
- R² < 0.0: Worse than predicting mean

**Example**:
```python
y_true = [100, 200, 150, 250]
y_pred = [110, 190, 160, 240]
y_mean = 175

SS_res = (100-110)² + (200-190)² + (150-160)² + (250-240)²
       = 100 + 100 + 100 + 100 = 400

SS_tot = (100-175)² + (200-175)² + (150-175)² + (250-175)²
       = 5625 + 625 + 625 + 5625 = 12500

R² = 1 - (400/12500) = 1 - 0.032 = 0.968 (96.8% of variance explained)
```

---

## Choosing the Right k

### The k-Value Trade-off

```
Small k (1-3):
  Pros: Flexible, captures local patterns
  Cons: Sensitive to noise, overfitting
  
Medium k (5-9):
  Pros: Balanced, good generalization
  Cons: May miss some local patterns
  
Large k (15+):
  Pros: Robust to noise, smooth boundaries
  Cons: May miss important patterns, underfitting
```

### Visual Comparison

```
k=1: Very Complex Decision Boundary
    ●●●■●●
    ●■■■●●
    ■■■●●●
    
k=5: Moderate Complexity
    ●●●●●●
    ------
    ■■■■■■
    
k=20: Very Simple Boundary
    ●●●●●●
    ======
    ■■■■■■
```

### Rule of Thumb

1. **Start with k = √n**
   - n = number of training samples
   - Example: 100 samples → k ≈ 10

2. **Use odd k for binary classification**
   - Avoids ties in voting
   - Example: k = 3, 5, 7, not 2, 4, 6

3. **Cross-validation**
   - Try multiple k values
   - Choose k with best validation performance

### Example: Finding Optimal k

Here is the full, runnable version. Note the **three-way split**: `k` is a
hyperparameter, so it must be chosen on a validation set. Tuning `k` on the test
set and then reporting that same test score is a classic way to fool yourself.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)

# Three-way split: train / validation / test.
# k is chosen on the VALIDATION set, never on the test set.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

k_values = range(1, 31)
scores = []

for k in k_values:
    model = KNearestNeighbors(k=k, task='classification')
    model.fit(X_train, y_train)               # fit() must be called for every k
    scores.append(model.score(X_val, y_val))  # score on validation, not test

best_k = list(k_values)[int(np.argmax(scores))]
print(f"Best k on validation: {best_k}  (val accuracy = {max(scores):.4f})")

# Only now, with k fixed, touch the test set once for an unbiased estimate
final = KNearestNeighbors(k=best_k, task='classification')
final.fit(X_train, y_train)
print(f"Test accuracy with k={best_k}: {final.score(X_test, y_test):.4f}")
```

Output:
```
Best k on validation: 1  (val accuracy = 0.9667)
Test accuracy with k=1: 0.9333
```

**Read that result sceptically.** It says `k=1` — but eleven different values
(`k = 1, 3, 4, 6, 7, 9, 10, 11, 12, 14, 16`) *all* score exactly 0.9667 on the
validation set, and `np.argmax` simply returned the first of them. The validation
set has only 30 rows, so one sample is worth 3.3 percentage points and it cannot
distinguish these values at all.

Two lessons, and they matter more than the number itself:
1. With small data, prefer **k-fold cross-validation** over a single validation
   split; averaging over folds gives a far less noisy estimate.
2. When several `k` values tie, do not take `argmax` at its word — prefer the
   **largest** `k` among the tied values, since larger `k` means a smoother
   boundary and less variance for the same measured accuracy.

---

## Feature Scaling: Critical for KNN

### Why Scaling Matters

KNN uses distances, so feature scales matter greatly!

**Example without scaling**:
```python
Feature 1: Age (20-80)          → Range = 60
Feature 2: Income (20k-200k)    → Range = 180,000

Distance dominated by income!
Age difference of 30 years ≈ Income difference of $30
```

**Example with scaling**:
```python
Feature 1: Age (scaled to 0-1)        → Range = 1
Feature 2: Income (scaled to 0-1)     → Range = 1

Both features contribute equally!
```

### Standardization (Z-score)

Most common approach:

```
x_scaled = (x - mean) / std_dev
```

**Example**:
```python
from sklearn.preprocessing import StandardScaler

# Original data
X = [[20, 30000],
     [40, 60000],
     [60, 90000]]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Now: mean=0, std=1 for each feature
```

### Min-Max Scaling

Alternative approach:

```
x_scaled = (x - min) / (max - min)
```

**When to use each**:
- **Standardization**: When features have different distributions
- **Min-Max**: When you need values in specific range (0-1)

---

## Advantages and Limitations

### Advantages ✅

1. **Simple and Intuitive**
   - Easy to understand and explain
   - No complex math required

2. **No Training Phase**
   - Instant "training" (just storing data)
   - Can add new data easily

3. **Naturally Handles Multi-class**
   - No modification needed for multiple classes
   - Works with any number of classes

4. **No Assumptions**
   - Non-parametric (no assumptions about data distribution)
   - Flexible decision boundaries

5. **Good for Small to Medium Datasets**
   - Often performs well with limited data
   - Can capture complex patterns

### Limitations ❌

1. **Computationally Expensive**
   - O(n×d) for each prediction
   - n = training samples, d = dimensions
   - Slow on large datasets

2. **Memory Intensive**
   - Must store all training data
   - Large datasets require lots of memory

3. **Sensitive to Feature Scaling**
   - MUST scale features appropriately
   - Otherwise dominated by large-scale features

4. **Curse of Dimensionality**
   - Performance degrades in high dimensions
   - Distances become meaningless when d is large
   - See [the demonstration below](#seeing-the-curse-of-dimensionality) — this is
     usually asserted and rarely shown, but you can measure it in five lines

5. **Sensitive to Irrelevant Features**
   - Noisy features affect all distance calculations
   - Feature selection is crucial

6. **Imbalanced Data Issues**
   - Majority class can dominate predictions
   - May need weighted KNN

### When to Use KNN

**Good Use Cases**:
- ✅ Small to medium datasets (< 100k samples)
- ✅ Low to moderate dimensions (< 20 features)
- ✅ No time constraints for prediction
- ✅ Need interpretable results
- ✅ Complex, non-linear decision boundaries

**Bad Use Cases**:
- ❌ Large datasets (millions of samples)
- ❌ High-dimensional data (100+ features)
- ❌ Real-time predictions required
- ❌ Features have very different scales (unless scaled)
- ❌ Many irrelevant features

### Seeing the Curse of Dimensionality

"Distances become meaningless in high dimensions" is repeated everywhere and
demonstrated almost nowhere. It takes five lines. Scatter 500 random points in a
unit cube of `d` dimensions, pick a query point, and compare the **nearest**
distance to the **farthest** distance:

```python
import numpy as np

np.random.seed(42)
n = 500
print(f"{'dims':<8}{'nearest':<12}{'farthest':<12}{'ratio near/far':<16}")
print("-" * 48)
for d in [2, 5, 20, 100, 500]:
    X = np.random.rand(n, d)          # n random points in the unit cube
    q = np.random.rand(d)             # one query point
    dist = np.sqrt(((X - q) ** 2).sum(axis=1))
    print(f"{d:<8}{dist.min():<12.3f}{dist.max():<12.3f}{dist.min()/dist.max():<16.3f}")
```

Output:
```
dims    nearest     farthest    ratio near/far  
------------------------------------------------
2       0.022       0.906       0.024           
5       0.190       1.745       0.109           
20      1.120       2.371       0.472           
100     3.342       4.891       0.683           
500     8.155       9.717       0.839           
```

Read the last column. In 2-D the nearest point is **40x closer** than the farthest
one — "nearest neighbor" is a meaningful distinction. By 500-D the nearest point
is 84% of the distance to the farthest one: **every point is roughly the same
distance away**. The ratio is climbing toward 1.

That is the whole curse in one number. KNN's entire premise is that "near" means
"similar". When every point is equidistant, the k nearest neighbors are barely
different from k points picked at random, and the algorithm degrades to guessing
the majority class — while still costing you a full scan of the training set for
every prediction.

**What to do about it**: reduce dimensionality first (PCA, feature selection), or
use a metric less prone to concentration. This is also the real reason Manhattan
(L1) is often recommended over Euclidean (L2) in high dimensions — lower-order
norms concentrate more slowly.

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load iris dataset
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CRITICAL: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = KNearestNeighbors(k=5, distance_metric='euclidean', task='classification')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Evaluate model
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Show predictions with probabilities
print("\nSample Predictions:")
for i in range(5):
    print(f"True: {data.target_names[y_test[i]]}")
    print(f"Predicted: {data.target_names[y_pred[i]]}")
    print(f"Probabilities: Setosa={y_proba[i][0]:.2f}, "
          f"Versicolor={y_proba[i][1]:.2f}, Virginica={y_proba[i][2]:.2f}\n")

# Try different k values
print("Testing different k values:")
for k in [1, 3, 5, 7, 9, 11]:
    model = KNearestNeighbors(k=k, task='classification')
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    print(f"k={k:2d}: Accuracy={score:.4f}")
```

---

## Optimizations and Variants

### 1. Weighted KNN

Give more weight to closer neighbors:

```
weight_i = 1 / distance_i

Prediction = Σ(weight_i × vote_i) / Σ(weight_i)
```

**Benefit**: Closer neighbors have more influence

**This one is implemented** — pass `weights='distance'`:

```python
model = KNearestNeighbors(k=5, task='classification', weights='distance')
model.fit(X_train, y_train)
```

The default is `weights='uniform'`, which is the canonical KNN rule (and
scikit-learn's default): every one of the k neighbors counts the same regardless
of how far away it is.

**The edge case that matters**: what if a neighbor is at distance 0, i.e. the
query point is itself a training point? Then `1 / 0` is a division by zero. The
implementation follows scikit-learn's rule: if any neighbor is at distance 0,
those exact-match neighbors take *all* the weight and the rest get zero. That is
both numerically safe and the right answer — the query *is* that training point.

One consequence is worth knowing before it surprises you: with
`weights='distance'`, **training accuracy is always 1.0000 for any k**, because
every training point is at distance 0 from itself and therefore out-votes all
its neighbors. A perfect training score means nothing here; only the test column
tells you anything. You can see this directly in the `DEMO 3` table printed by
`python _5_knn.py`.

### 2. Ball Tree / KD Tree

Data structures to speed up neighbor search:

```
Naive approach: O(n×d) per prediction
With tree: O(d×log(n)) per prediction
```

**Benefit**: Much faster for moderate dimensions

### 3. Approximate Nearest Neighbors

Trade accuracy for speed:

```
Find "approximately" nearest neighbors
Speedup: 10-100x faster
Accuracy loss: 1-5%
```

**Benefit**: Enables use on large datasets

---

## Key Concepts to Remember

### 1. **KNN is a Lazy Learner**
No training phase! Just stores data and computes at prediction time.

### 2. **Feature Scaling is CRITICAL**
Always standardize or normalize features before using KNN.

### 3. **k is the Most Important Hyperparameter**
- Too small → overfitting, noise sensitivity
- Too large → underfitting, over-smoothing
- Use cross-validation to find optimal k

### 4. **Distance Metric Matters**
- Euclidean: Most common, natural measure
- Manhattan: Better in high dimensions
- Choose based on your data and domain

### 5. **Computational Cost**
- Training: O(1) - instant
- Prediction: O(n×d) - expensive
- Memory: O(n×d) - stores all data

### 6. **Curse of Dimensionality**
Performance degrades as dimensions increase:
- Distances become similar in high dimensions
- "Nearest" neighbors aren't actually near
- Solution: Feature selection, dimensionality reduction

---

## Conclusion

K-Nearest Neighbors is a simple yet powerful algorithm! By understanding:
- How distance metrics measure similarity
- How voting/averaging produces predictions
- How k controls model complexity
- How feature scaling affects performance

You've gained a fundamental tool in your machine learning toolkit! 🎯

**When to Use KNN**:
- ✅ Small to medium datasets
- ✅ Non-linear decision boundaries
- ✅ Need interpretable predictions
- ✅ Multi-class classification
- ✅ Both classification and regression

**When to Use Something Else**:
- ❌ Large datasets → Use decision trees, random forests
- ❌ High dimensions → Use dimensionality reduction first
- ❌ Need fast predictions → Use logistic regression, SVM
- ❌ Many irrelevant features → Use regularized models

**Next Steps**:
- Try KNN on your own datasets
- Experiment with different k values and distance metrics
- Compare with other algorithms (Logistic Regression, Decision Trees)
- Learn about weighted KNN and approximate methods
- Explore KNN for anomaly detection
- Study curse of dimensionality in depth

Happy coding! 💻🎯

