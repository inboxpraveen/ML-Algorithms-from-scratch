# Support Vector Machine (SVM) from Scratch: A Comprehensive Guide

Welcome to the world of Support Vector Machines! 🎯 In this comprehensive guide, we'll explore one of the most powerful and elegant machine learning algorithms. Think of it as finding the "widest street" that separates two neighborhoods!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is Support Vector Machine?](#what-is-support-vector-machine)
3. [How SVM Works](#how-svm-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Choosing Hyperparameters](#choosing-hyperparameters)
11. [Feature Scaling: Critical for SVM](#feature-scaling-critical-for-svm)
12. [Advantages and Limitations](#advantages-and-limitations)
13. [Simplification vs. Canonical SVM](#simplification-vs-canonical-svm)
14. [Complete Usage Example](#complete-usage-example)
15. [SVM vs Other Algorithms](#svm-vs-other-algorithms)
16. [Key Concepts to Remember](#key-concepts-to-remember)
17. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra
dependencies beyond NumPy. (It uses the same model, seed and data as DEMO 1 of the
`__main__` block at the bottom of `_8_svm.py`, so the numbers below are exactly the
ones `python _8_svm.py` produces - that demo just lays them out differently and adds
two more sections.)

```python
# ---------------------------------------------------------------
# SVM from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _8_svm.py   (the __main__ block runs this)
# Or copy the SupportVectorMachine class from _8_svm.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the SupportVectorMachine class here (from _8_svm.py) ----
# class SupportVectorMachine: ...

np.random.seed(42)

# ------ Two Gaussian blobs, one per class ------
X_neg = np.random.randn(100, 2) + np.array([-1.5, -1.5])   # class -1
X_pos = np.random.randn(100, 2) + np.array([ 1.5,  1.5])   # class +1
X_all = np.vstack([X_neg, X_pos])
y_all = np.array([-1] * 100 + [1] * 100)

# Shuffle BEFORE slicing - otherwise the "test set" would be one class only
shuffle_idx = np.random.permutation(200)
X_all, y_all = X_all[shuffle_idx], y_all[shuffle_idx]

X_train_raw, X_test_raw = X_all[:140], X_all[140:]
y_train, y_test = y_all[:140], y_all[140:]

# Standardize with TRAIN statistics only (never peek at the test set)
mu, sigma = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
X_train = (X_train_raw - mu) / sigma
X_test = (X_test_raw - mu) / sigma

model = SupportVectorMachine(
    learning_rate=0.001,   # step size for each sub-gradient update
    lambda_param=0.01,     # regularization: bigger -> wider margin
    iterations=500         # epochs (full passes over the training set)
)
model.fit(X_train, y_train)

print(f"Train Accuracy : {model.score(X_train, y_train):.4f}")
print(f"Test  Accuracy : {model.score(X_test,  y_test):.4f}")

params = model.get_params()
print(f"w = {params['weights']}   b = {params['bias']:+.4f}")
print(f"Margin 2/||w|| : {2.0 / params['norm_w']:.4f}")
print(f"Loss           : {model.losses[0]:.4f} -> {model.losses[-1]:.4f}")
print(f"Support vectors: {len(model.support_vector_indices_)} of {len(X_train)}")

# predict() always returns -1/+1; decision_function() gives signed distance
distances = model.decision_function(X_test)
predictions = model.predict(X_test)
for i in range(5):
    print(f"  true={y_test[i]:+d}  pred={int(predictions[i]):+d}  "
          f"f(x)={distances[i]:+.3f}")
```

Expected output:
```
Train Accuracy : 0.9857
Test  Accuracy : 0.9833
w = [1.27384031 1.05834527]   b = -0.1350
Margin 2/||w|| : 1.2076
Loss           : 1.0000 -> 0.0798
Support vectors: 15 of 140
  true=+1  pred=+1  f(x)=+0.741
  true=+1  pred=+1  f(x)=+2.731
  true=+1  pred=+1  f(x)=+1.503
  true=+1  pred=+1  f(x)=+1.769
  true=-1  pred=-1  f(x)=-2.862
```

Running `python _8_svm.py` prints these same numbers (in a slightly wider layout, with
an extra confidence column) plus two more demos: a lambda sweep showing the margin
`2/||w||` widening from 0.9981 to 1.9406 as lambda goes from 0.001 to 0.1, and a
scaling demo showing that an unscaled feature leaves the loss at 5.5030 where the
standardized fit reaches 0.0800.

---

## What is Support Vector Machine?

Support Vector Machine (SVM) is a **powerful supervised learning algorithm** used primarily for classification. It works by finding the optimal boundary (hyperplane) that best separates different classes in your data.

**Real-world analogy**: 
Imagine you're a city planner trying to build a road that separates two neighborhoods. You don't just want any road – you want the **widest possible road** that keeps maximum distance from both neighborhoods. That's exactly what SVM does with data!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Discriminative, Supervised |
| **Tasks** | Binary Classification (can extend to multi-class) |
| **Decision Boundary** | Linear hyperplane (in this implementation) |
| **Optimization Goal** | Maximum margin separation |
| **Key Parameters** | Learning rate, Regularization (λ) |

### The Core Idea

```
"Find the decision boundary that maximizes the distance 
to the nearest data points from both classes"
```

SVM's unique approach:
1. **Finds** a decision boundary (hyperplane)
2. **Maximizes** the margin (distance to nearest points)
3. **Minimizes** classification errors

The points closest to the boundary are called **Support Vectors** – they literally "support" the decision boundary! You can pull them out of a fitted model here with `model.support_vector_indices_`; the [Support Vectors](#support-vectors-which-points-actually-matter) subsection shows why they are the only points that matter.

---

## How SVM Works

### The Algorithm in 5 Steps

```
Step 1: Initialize weights (w) to all zeros and bias (b) to 0
        (no randomness - so every run gives identical results)
         ↓
Step 2: For each training sample, compute: y_i * (w·x_i + b)
         ↓
Step 3: If value < 1: Point is misclassified or within margin
        → Update w and b to push boundary away
         ↓
Step 4: If value >= 1: Point is correctly classified outside margin
        → Only apply regularization (keep margin wide)
         ↓
Step 5: Repeat for `iterations` epochs
        (a fixed budget - there is no convergence test, see the
         loss-curve notes under Model Evaluation)
```

### Visual Example: Finding the Best Boundary

```
Poor Boundary (too close to one class):
    ●●●●|
    ●●● |  ■■■■
    ●●●●|  ■■■■
           ■■■■

Better Boundary (centered but no margin):
    ●●●●
    ●●●● |
    ●●●●|■■■■
        | ■■■■

BEST Boundary (maximum margin):
    ●●●●
    ●●●●  :  :
    ●●●●  :| :  ■■■■
          :| :  ■■■■
          :  :  ■■■■
    
    : = margin boundaries
    | = decision boundary
```

### The Margin Concept

```
    Support Vectors (points on margin)
           ↓         ↓
    ●●●●  ●●         ■■  ■■■■
    ●●●●  ●●         ■■  ■■■■
    ●●●●  ●●         ■■  ■■■■
          
          ←margin→
          
    Margin Width = 2 / ||w||
    
    Goal: Maximize margin = Minimize ||w||
```

**Key Insight**: The margin width is inversely proportional to the magnitude of the weight vector. So minimizing ||w|| maximizes the margin!

#### Where does 2/||w|| come from?

This is worth deriving rather than memorising: it is the one fact that turns "find the widest street" into an equation a computer can minimise.

The two kerbs of the street are the places where the decision function equals +1 and -1:

```
w·x + b = +1      ← the +1 margin line
w·x + b = -1      ← the -1 margin line
```

Stand on any point x₊ of the first line and walk straight across to the second one. "Straight across" means perpendicular to the boundary, and the perpendicular direction is w/||w|| (that is exactly what "w is normal to the boundary" means). So after walking a distance d you land at:

```
x₋ = x₊ - d * (w / ||w||)
```

Now plug that landing point into the second line's equation:

```
w·(x₊ - d * w/||w||) + b = -1

(w·x₊ + b)  -  d * (w·w)/||w||  = -1
 └────┬───┘
   this is +1, because x₊ sits on the +1 line

Since w·w = ||w||²,  (w·w)/||w|| = ||w||, so:

    1 - d*||w|| = -1
       d*||w||  =  2
            d   =  2 / ||w||
```

That distance d **is** the street width. A larger ||w|| gives a narrower street, so:

```
maximize the margin  2/||w||   ⟺   minimize  ||w||²
```

And ||w||² is precisely the term `self.lambda_param * np.dot(self.weights, self.weights)` in `_compute_loss` - the regularizer is not a bolted-on afterthought, it *is* the margin-widening objective.

You can read the street width straight off a fitted model:

```python
params = model.get_params()
print(2.0 / params['norm_w'])   # street width, in feature units
```

### Why Maximum Margin?

```
Small Margin:
    ●●●●|■■■■
    ●●● |■■■■
    
    Problem: New points near boundary 
             easily misclassified
             
Large Margin:
    ●●●●    :  :    ■■■■
    ●●●●    :| :    ■■■■
    
    Benefit: More robust, better generalization
             More confident predictions
```

---

## The Mathematical Foundation

### The Decision Function

For a point x, the decision function is:

```
f(x) = w·x + b
```

Where:
- **w** = weight vector (perpendicular to decision boundary)
- **x** = feature vector
- **b** = bias term (shifts boundary position)

**Classification Rule**:
```
If f(x) ≥ 0: Predict class +1
If f(x) < 0: Predict class -1
```

**Example**:
```python
w = [2, 3]
b = -5
x = [1, 2]

f(x) = [2, 3]·[1, 2] + (-5)
     = 2*1 + 3*2 - 5
     = 2 + 6 - 5
     = 3

Since 3 > 0: Predict class +1 ✓
```

### The Margin

For a point to be correctly classified **outside the margin**:

```
y_i * (w·x_i + b) ≥ 1
```

Where:
- y_i ∈ {-1, +1} is the true label
- w·x_i + b is the decision function

**Three cases**:

```
Case 1: y_i * (w·x_i + b) ≥ 1
        → Correctly classified, outside margin ✓
        
Case 2: 0 < y_i * (w·x_i + b) < 1
        → Correctly classified, but inside margin ⚠️
        
Case 3: y_i * (w·x_i + b) ≤ 0
        → Misclassified ✗
```

**Visual Representation**:

```
Class +1:  ●●●●
           ●●●●  ●●     margin    ▲ y=+1: want w·x + b ≥ +1
                  :  :  boundary  | y=-1: want w·x + b ≤ -1
                  :| :            ▼
                  :  :     ■■
           ■■■■         ■■■■
           ■■■■
Class -1:
```

### The Hinge Loss

SVM uses **hinge loss** to penalize points within or on the wrong side of the margin:

```
Hinge Loss = max(0, 1 - y_i * (w·x_i + b))
```

**Behavior**:

```
y * f(x)  |  Loss      |  Interpretation
----------|------------|----------------------------------
≥ 1       |  0         |  Correct, outside margin ✓
0 to 1    |  > 0       |  Correct, but within margin ⚠️
≤ 0       |  ≥ 1       |  Misclassified ✗
```

**Graph**:

```
Loss
  |
2 |              /
  |            /
1 |          /
  |        /
0 |______/__________________ y*f(x)
  -1     0      1      2
         
  Penalty increases as point moves
  further from correct side of margin
```

### The Complete Objective Function

SVM minimizes:

```
L(w, b) = λ||w||² + (1/n) Σ max(0, 1 - y_i * (w·x_i + b))
          ↑                ↑
     Regularization    Hinge Loss
     (maximize margin) (minimize errors)
```

**Two competing goals**:

1. **Minimize ||w||²**: Make margin as wide as possible
2. **Minimize hinge loss**: Correctly classify all points

**The parameter λ balances these goals**:

```
Large λ:  Prioritize wide margin
          → More tolerance for misclassification
          → Simpler model (less overfitting)
          
Small λ:  Prioritize correct classification
          → Less tolerance for misclassification
          → More complex model (may overfit)
```

### Gradients for Optimization

To minimize the loss, we compute gradients:

**When y_i * (w·x_i + b) < 1** (within or wrong side of margin):
```
∂L/∂w = 2λw - y_i * x_i
∂L/∂b = -y_i
```

**When y_i * (w·x_i + b) ≥ 1** (correct and outside margin):
```
∂L/∂w = 2λw
∂L/∂b = 0
```

**Gradient Descent Updates**:
```
w ← w - learning_rate * ∂L/∂w
b ← b - learning_rate * ∂L/∂b
```

### Support Vectors: Which Points Actually Matter

Look carefully at the two gradient cases above. When `y_i * (w·x_i + b) ≥ 1`, the point contributes **only** the regularization term `2λw` – a term that does not mention `x_i` at all. That point could be deleted from the dataset and the update would be identical.

So the training points fall into two groups:

```
y_i * (w·x_i + b) >  1   →  hinge loss 0, no data gradient
                            "spectator" - delete it, nothing changes

y_i * (w·x_i + b) <= 1   →  hinge loss active, gradient includes -y_i * x_i
                            SUPPORT VECTOR - it is holding the boundary up
```

That inequality is the whole definition. Because sub-gradient descent never lands
exactly on `y*f(x) == 1`, the implementation uses a small tolerance:

```python
margins = y_labels * (X @ self.weights + self.bias)
self.support_vector_indices_ = np.where(margins <= 1 + 1e-3)[0]
```

`fit()` stores those indices in `self.support_vector_indices_`, and
`get_support_vectors(X, y, tol=1e-3)` computes them for any dataset:

```python
model.fit(X_train, y_train)
sv = model.support_vector_indices_          # indices into X_train
print(f"{len(sv)} of {len(X_train)} points support the boundary")
print(X_train[sv])                           # the points themselves
```

**Sanity check you can run yourself**: refit the model on *only* the support vectors and you should land on nearly the same boundary. Refit it on only the spectators and the boundary moves a lot. That is the concrete meaning of "only support vectors matter".

Note the direction of the effect: a **larger λ** widens the margin, which sweeps more points inside it, so **more** points become support vectors. In the demo's lambda sweep, `#SV` climbs 10 → 15 → 46 as lambda goes 0.001 → 0.01 → 0.1.

### Beyond Linear: The Kernel Trick

This implementation solves the **primal** problem directly in terms of `w` and `b`, which is why it can only ever draw a straight boundary. The other route – the one that makes SVM famous – is to solve the **dual** problem instead.

Every training point gets a weight `a_i ≥ 0` (a Lagrange multiplier), and the optimal `w` turns out to be a weighted sum of the training points themselves:

```
w = Σ_i a_i * y_i * x_i
```

Substituting that back into `f(x) = w·x + b` gives a decision function written purely in terms of **dot products between points**:

```
f(x) = Σ_i a_i * y_i * (x_i · x) + b
```

Two consequences follow:

1. **Only support vectors have `a_i > 0`.** Every spectator gets `a_i = 0` and drops out of the sum entirely. This is the dual's version of the same fact the primal shows through the hinge gradient – and it is why a trained kernel SVM stores only its support vectors.

2. **The features `x` only ever appear inside a dot product.** So you can replace `x_i · x` with any function `K(x_i, x)` that behaves like a dot product in some higher-dimensional space, *without ever computing coordinates in that space*. That substitution is the **kernel trick**:

```
Linear:      K(a, b) = a·b
Polynomial:  K(a, b) = (a·b + c)^d
RBF:         K(a, b) = exp(-gamma * ||a - b||²)
```

An RBF kernel corresponds to an infinite-dimensional feature space, yet costs one exponential per pair of points. The boundary is still a straight line *in that space*, which is a curved, flexible boundary back in the original one.

```
Original 2-D space          Lifted space (via kernel)
  ●●● ■■■ ●●●                     ■■■■■■
   ■■■■■■■■■         →       ───────────────  ← linear boundary here
  ●●● ■■■ ●●●                 ●●●●      ●●●●
  not linearly separable      separable by a plane
```

**What that means for this file**: the primal sub-gradient code below cannot do any of this – it has no `a_i` and never stores training points. Solving the dual needs a constrained quadratic program (in practice, the SMO algorithm), which is a substantially larger implementation. See [Simplification vs. Canonical SVM](#simplification-vs-canonical-svm) for exactly what is and is not here, and use `sklearn.svm.SVC(kernel='rbf')` when you need non-linear boundaries.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class SupportVectorMachine:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []
        self.classes_ = None                 # original labels seen by fit()
        self.support_vector_indices_ = None  # filled in at the end of fit()
```

### Core Methods

1. **`__init__(learning_rate, lambda_param, iterations)`** - Initialize model
   - learning_rate: Step size for gradient descent (0.0001 to 0.01)
   - lambda_param: Regularization strength (0.001 to 1.0)
   - iterations: Number of training iterations

2. **`_compute_loss(X, y)`** - Private helper method
   - Computes total loss (regularization + hinge loss)
   - Used for tracking training progress
   - Returns single float value

3. **`fit(X, y)`** - Train the model
   - Converts labels to -1 and +1 if needed (and remembers the original encoding in `classes_`)
   - Initializes weights and bias to zeros
   - Performs sub-gradient descent optimization
   - Updates weights based on margin violations
   - Records `self.support_vector_indices_` at the end
   - Resets `self.losses`, which ends up with `iterations + 1` entries: `losses[0]` is the loss before any update, `losses[-1]` the loss after the last epoch

4. **`predict(X)`** - Predict class labels
   - Computes decision function: w·x + b
   - Returns +1 or -1 based on sign
   - **Always** returns -1/+1, even if `fit()` was given 0/1 labels - use `score()`, or convert with `np.where(pred == -1, 0, 1)`
   - Main prediction interface

5. **`predict_proba(X)`** - Heuristic confidence scores
   - Squashes the decision function through a sigmoid: `1 / (1 + exp(-f(x)))`
   - Returns shape `(n_samples, 2)`: column 0 for class -1, column 1 for class +1
   - **Not calibrated probabilities** - a real SVM needs Platt scaling for that

6. **`decision_function(X)`** - Get decision values
   - Returns signed distances from boundary
   - Positive = class +1, Negative = class -1
   - Magnitude indicates confidence
   - Raises a clear `"Model is not fitted yet"` error if called before `fit()`

7. **`get_support_vectors(X, y, tol=1e-3)`** - Find the boundary-defining points
   - Returns the indices where `y_i * (w·x_i + b) <= 1 + tol`
   - These are the only points with an active hinge term
   - `fit()` stores the same thing for the training set in `support_vector_indices_`

8. **`score(X, y)`** - Calculate accuracy (this is a classifier, so it is accuracy, not R²)
   - Returns proportion of correct predictions
   - Handles both -1/+1 and 0/1 labels
   - Used for model evaluation

9. **`get_params()`** - Get model parameters
   - Returns weights, bias, and weight norm
   - Useful for interpretation
   - Weight norm indicates margin width (`2 / norm_w`)

---

## Step-by-Step Example

Let's walk through a complete example classifying **fruits** based on weight and sweetness:

### The Data

```python
import numpy as np

# Features: [weight (grams), sweetness (1-10)]
X_train = np.array([
    [150, 8],   # Apple
    [170, 9],   # Apple
    [140, 7],   # Apple
    [160, 8],   # Apple
    [350, 4],   # Orange
    [380, 5],   # Orange
    [340, 3],   # Orange
    [360, 4]    # Orange
])

# Labels: +1 = Apple, -1 = Orange
y_train = np.array([1, 1, 1, 1, -1, -1, -1, -1])
```

### Visualizing the Data

```
Sweetness
   10|
    9|  ●
    8|  ●    ●
    7|  ●
    6|
    5|              ■
    4|              ■   ■
    3|              ■
    2|
    1|
    0+----------------------- Weight
      0  100 200 300 400
      
  ● = Apple (+1)
  ■ = Orange (-1)
```

### Training the Model

```python
model = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, iterations=1000)
model.fit(X_train, y_train)
```

**What happens internally**:

**Iteration 1**:
```
Initial: w = [0, 0], b = 0
First point: x = [150, 8], y = +1

Check: y * (w·x + b) = 1 * (0*150 + 0*8 + 0) = 0
Since 0 < 1: Point is within margin!

Gradients:
  ∂L/∂w = 2*0.01*[0,0] - 1*[150,8] = [-150, -8]
  ∂L/∂b = -1

Update:
  w = [0,0] - 0.001*[-150,-8] = [0.15, 0.008]
  b = 0 - 0.001*(-1) = 0.001
```

**After many iterations**:
```
Final: w ≈ [-0.017460, 0.564212], b ≈ 0.052000
```

These are the real fitted values – run the code above and print
`model.get_params()` to confirm them. Because the weights start at exactly zero
and nothing in `fit()` is random, this fit is **deterministic**: you get these
numbers every single run.

Read the signs, they tell the story:

```
w[0] = -0.0175  (weight)     → NEGATIVE: heavier fruit is less Apple-like
w[1] = +0.5642  (sweetness)  → POSITIVE: sweeter fruit is more Apple-like
b    = +0.0520               → tiny shift of the boundary
```

Note also how small `w[0]` is compared to `w[1]`. That is **not** because weight
in grams is unimportant – it is because the raw feature is 40x larger in
magnitude, so a small coefficient already produces a big contribution. This is
exactly the distortion the [Feature Scaling](#feature-scaling-critical-for-svm)
section warns about; this example is left unscaled only because 8 well-separated
points still work out.

Sanity check on the training set – every fruit ends up on the correct side with
`y * f(x) > 1`, i.e. outside the margin:

```
Apples  (y=+1): f = [+1.947, +2.162, +1.557, +1.772]   all ≥ +1  ✓
Oranges (y=-1): f = [-3.802, -3.762, -4.192, -3.977]   all ≤ -1  ✓

Training accuracy = 1.00
||w|| = 0.5645  →  margin width 2/||w|| = 3.54
```

### Making Predictions

```python
X_test = np.array([
    [155, 8],   # Similar to apples
    [360, 4],   # Similar to oranges
    [250, 6]    # Boundary case
])

predictions = model.predict(X_test)
distances = model.decision_function(X_test)
```

**Prediction process for first test point**:

```python
x = [155, 8]
w = [-0.017460, 0.564212]
b = 0.052

f(x) = w·x + b
     = (-0.017460)*155 + 0.564212*8 + 0.052
     = -2.7063 + 4.5137 + 0.052
     = 1.8594

Since 1.8594 > 0: Predict +1 (Apple) ✓
Confidence: |1.8594| = 1.86, and it is > 1, so this point sits
outside the margin entirely - the model is not just right, it is safe.
```

### Complete Prediction Results

```python
print("Test Results:")
for i, x in enumerate(X_test):
    pred = predictions[i]
    dist = distances[i]
    label = "Apple" if pred == 1 else "Orange"
    print(f"  Point {x}: {label} (distance={dist:.2f})")

# Output:
#   Point [155   8]: Apple (distance=1.86)      ← outside the margin, confident
#   Point [360   4]: Orange (distance=-3.98)    ← deep in Orange territory
#   Point [250   6]: Orange (distance=-0.93)    ← INSIDE the margin (|f| < 1),
#                                                 a genuine borderline call
```

The third point is the interesting one. At 250 g it is heavier than any apple in
the training set, but at sweetness 6 it is sweeter than any orange. The model
puts it on the Orange side, but with `|f(x)| = 0.93 < 1` it lands *inside the
street* – exactly the region where SVM says "I am not confident". A point like
this would have been a support vector had it been in the training set.

Run the numbers yourself:

```python
model.predict(X_test)            # -> [ 1. -1. -1.]
model.decision_function(X_test)  # -> [ 1.85937417 -3.97680952 -0.92776644]
```

---

## Real-World Applications

### 1. **Image Classification**
Classify images into categories:
- Input: Image features (pixels, edges, textures)
- Output: Object class (cat, dog, car, etc.)
- Example: "Is this image a cat or dog?"

### 2. **Text Classification**
Categorize text documents:
- Input: Text features (word frequencies, TF-IDF)
- Output: Category (spam/not spam, positive/negative)
- Example: Email spam detection

### 3. **Medical Diagnosis**
Diagnose diseases from patient data:
- Input: Medical test results, symptoms, patient history
- Output: Diagnosis (disease present or not)
- Example: "Does this patient have diabetes?"

### 4. **Face Recognition**
Identify or verify faces:
- Input: Facial features (distances, angles, landmarks)
- Output: Person identity or verification result
- Example: Unlock phone with face

### 5. **Handwriting Recognition**
Recognize handwritten characters:
- Input: Pixel values of handwritten character
- Output: Character class (0-9, A-Z)
- Example: Check processing, postal code recognition

### 6. **Credit Scoring**
Assess creditworthiness:
- Input: Income, credit history, debt, employment
- Output: Approved or denied
- Example: "Should we approve this loan?"

### 7. **Bioinformatics**
Classify biological data:
- Input: Gene expression levels, protein sequences
- Output: Disease classification, gene function
- Example: Cancer type classification

### 8. **Quality Control**
Detect defective products:
- Input: Sensor readings, measurements, images
- Output: Defective or acceptable
- Example: Manufacturing defect detection

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. The Hinge Loss Computation

```python
def _compute_loss(self, X, y):
    distances = 1 - y * (X @ self.weights + self.bias)
    hinge_loss = np.maximum(0, distances)
    loss = self.lambda_param * np.dot(self.weights, self.weights) + np.mean(hinge_loss)
    return loss
```

**Step-by-step breakdown**:

```python
# Example with one point
y = 1  # True label
w = [0.5, 0.3]
x = [2, 4]
b = -1

# Step 1: Calculate decision function
decision = w·x + b = 0.5*2 + 0.3*4 - 1 = 1 + 1.2 - 1 = 1.2

# Step 2: Calculate margin distance
distance = 1 - y * decision = 1 - 1*1.2 = -0.2

# Step 3: Apply max(0, distance)
hinge = max(0, -0.2) = 0  # No penalty (outside margin)

# Step 4: Add regularization
regularization = λ * ||w||² = 0.01 * (0.5² + 0.3²) = 0.01 * 0.34 = 0.0034

# Step 5: Total loss
total_loss = regularization + hinge = 0.0034 + 0 = 0.0034
```

**Why this works**:
- When correctly classified outside margin: hinge = 0
- When within margin or misclassified: hinge > 0
- Regularization term keeps weights small (wide margin)

### 2. The Training Loop

```python
# Start a fresh loss history; entry 0 is the loss before any update
self.losses = [self._compute_loss(X, y_labels)]

for iteration in range(self.iterations):
    for idx, x_i in enumerate(X):
        condition = y_labels[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1

        if condition:
            dw = 2 * self.lambda_param * self.weights
            db = 0
        else:
            dw = 2 * self.lambda_param * self.weights - y_labels[idx] * x_i
            db = -y_labels[idx]

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    # Record the loss AFTER this epoch's updates
    self.losses.append(self._compute_loss(X, y_labels))
```

**Why the loss is recorded after the inner loop**: if you append the loss at the
*top* of the epoch, `losses[-1]` describes the model as it was one full epoch ago
and the final epoch's improvement never gets recorded at all. Appending after the
updates makes `losses[-1]` the loss of the model you actually keep. The list is
also reset at the start of every `fit()`, so re-fitting the same object gives a
clean curve instead of two runs glued together.

The list ends up with `iterations + 1` entries:

```
losses[0]   loss before any update  (always exactly 1.0, since w = 0 and b = 0
                                     make every hinge term max(0, 1-0) = 1)
losses[1]   loss after epoch 1
...
losses[-1]  loss after the final epoch = the model you are holding
```

**Example iteration**:

```python
# Current state
w = [0.1, 0.2]
b = 0.05
learning_rate = 0.001
λ = 0.01

# Point: x = [150, 8], y = +1
decision = 1 * (0.1*150 + 0.2*8 + 0.05) = 16.65

# Check condition
16.65 >= 1? YES → Point is correctly classified outside margin

# Compute gradients (only regularization)
dw = 2 * 0.01 * [0.1, 0.2] = [0.002, 0.004]
db = 0

# Update weights
w = [0.1, 0.2] - 0.001 * [0.002, 0.004]
  = [0.099998, 0.199996]
b = 0.05 - 0.001 * 0 = 0.05

# Point: x = [250, 6], y = +1 (suppose this is within margin)
decision = 1 * (0.1*250 + 0.2*6 + 0.05) = 26.25
26.25 >= 1? YES, but let's say it's close to boundary

# If it were within margin (decision < 1):
# dw = 2*0.01*w - 1*x = [0.002, 0.004] - [250, 6]
#    = [-249.998, -5.996]
# db = -1
# Updates would be much larger to push boundary away!
```

### 3. The Prediction Function

```python
def predict(self, X):
    # np.where also settles the exact-zero case: f(x) == 0 -> +1
    return np.where(self.decision_function(X) >= 0, 1.0, -1.0)
```

**How it works** (using the real fitted fruit model, `w = [-0.017460, 0.564212]`, `b = 0.052`):

```python
# Example with 3 test points
X_test = [[100, 9],   # Should be +1  (light and very sweet)
          [400, 3],   # Should be -1  (heavy and not sweet)
          [250, 6]]   # Near boundary

# Calculate decision values
linear_output = X_test @ w + b
# = [100*(-0.017460) + 9*0.564212 + 0.052,
#    400*(-0.017460) + 3*0.564212 + 0.052,
#    250*(-0.017460) + 6*0.564212 + 0.052]
# = [-1.746 + 5.078 + 0.052,
#    -6.984 + 1.693 + 0.052,
#    -4.365 + 3.385 + 0.052]
# = [3.3839, -5.2394, -0.9278]

# Take the sign
predictions = [sign(3.3839), sign(-5.2394), sign(-0.9278)]
            = [+1, -1, -1]         ← matches the comments above ✓
```

Notice the third value: `|-0.9278| < 1`, so that point sits **inside the margin**. It is classified as -1, but it is the kind of point the model is least sure about.

**Why `np.where` and not `np.sign`?** `np.sign(0)` returns `0`, which is not a valid class label, so the older form needed a follow-up fix-up line (`predictions[predictions == 0] = 1`) – and that line crashed when `X` was a single 1-D sample, because the result was then a bare scalar with no item assignment. `np.where(f >= 0, 1.0, -1.0)` handles the tie and the shape in one step.

### 4. The Decision Function

```python
def decision_function(self, X):
    if self.weights is None:
        raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

    # atleast_2d lets a single 1-D sample through as one row
    X = np.atleast_2d(np.asarray(X, dtype=float))

    return X @ self.weights + self.bias
```

Everything else routes through this method – `predict`, `predict_proba`, `score` and `get_support_vectors` all call it – so the "not fitted" guard and the shape handling only need to exist in one place.

**Interpretation of output**:

```python
# Decision values for 3 points
decision_values = [2.5, -1.8, 0.05]

# Point 1: 2.5
#   → Strongly class +1 (far from boundary)
#   → High confidence

# Point 2: -1.8
#   → Strongly class -1 (far from boundary)
#   → High confidence

# Point 3: 0.05
#   → Weakly class +1 (near boundary)
#   → Low confidence, could go either way
```

**Visual representation**:

```
              decision_function(x)
        -3    -2    -1    0    +1   +2   +3
Class -1 ←----------------------→ Class +1
         ■■■      :       :      ●●●
    Strong -1  Weak -1  Weak +1  Strong +1
```

---

## Model Evaluation

### For Binary Classification

#### 1. Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Example**:
```python
y_true = [ 1,  1, -1, -1,  1]
y_pred = [ 1, -1, -1, -1,  1]

correct = 4  # Indices 0, 2, 3, 4
total = 5
accuracy = 4/5 = 0.8 (80%)
```

#### 2. Confusion Matrix

For SVM with classes -1 and +1:

```
                Predicted
              -1      +1
Actual   -1   [TN]    [FP]
          +1  [FN]    [TP]

TN = True Negatives  (correctly predicted -1)
TP = True Positives  (correctly predicted +1)
FN = False Negatives (predicted -1, actually +1)
FP = False Positives (predicted +1, actually -1)
```

**Example**:
```python
y_true = [-1, -1, -1, +1, +1, +1, +1, +1]
y_pred = [-1, -1, +1, +1, +1, +1, -1, +1]

Confusion Matrix:
              Predicted
              -1    +1
Actual   -1  [ 2     1 ]  ← 2 TN, 1 FP
          +1 [ 1     4 ]  ← 1 FN, 4 TP
```

#### 3. Precision, Recall, F1-Score

```
Precision = TP / (TP + FP)  # Of predicted +1, how many correct?
Recall    = TP / (TP + FN)  # Of actual +1, how many found?
F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
```

**Example** (using confusion matrix above):
```
Precision = 4 / (4 + 1) = 4/5 = 0.80 (80%)
Recall    = 4 / (4 + 1) = 4/5 = 0.80 (80%)
F1-Score  = 2 * (0.80 * 0.80) / (0.80 + 0.80) = 0.80 (80%)
```

#### 4. Decision Function Analysis

```python
# Analyze prediction confidence
# (assumes `model`, `X_test_scaled` and `y_test` from the Complete Usage Example
#  below, with y_test encoded as -1/+1. Note X_test_scaled, NOT the raw X_test -
#  feeding unscaled data to a model trained on scaled data gives nonsense.)
distances = model.decision_function(X_test_scaled)

for i, (dist, true_label) in enumerate(zip(distances, y_test)):
    pred_label = +1 if dist >= 0 else -1
    confidence = abs(dist)
    # ASCII markers only - a checkmark glyph crashes a cp1252 Windows console
    correct = "OK" if pred_label == true_label else "XX"

    print(f"{correct} True:{true_label:+2d}, Pred:{pred_label:+2d}, "
          f"Confidence:{confidence:.2f}")
```

**Real output** – the first 10 rows on the breast-cancer test set from the
[Complete Usage Example](#complete-usage-example):
```
OK True:+1, Pred:+1, Confidence:1.02
OK True:-1, Pred:-1, Confidence:4.51
OK True:-1, Pred:-1, Confidence:2.25
OK True:+1, Pred:+1, Confidence:3.04
OK True:+1, Pred:+1, Confidence:4.22
OK True:-1, Pred:-1, Confidence:9.84
OK True:-1, Pred:-1, Confidence:8.32
OK True:-1, Pred:-1, Confidence:1.23
OK True:+1, Pred:+1, Confidence:0.35
OK True:+1, Pred:+1, Confidence:2.95
```

Read the confidence column, not just the OK/XX column:

- **Confidence > 1** – the point is *outside* the margin. Rows 2-8 above are safe.
- **Confidence < 1** – the point is **inside the margin**, in the middle of the
  street. Row 9 (`0.35`) is a correct prediction the model is barely committing to.
- The only two mistakes in this whole 114-row test set are
  `XX True:-1, Pred:+1, Confidence:1.06` and `XX True:-1, Pred:+1, Confidence:0.22`
  – both very close to the boundary, exactly where you would expect errors to be.

That last point is the payoff: SVM does not just tell you the class, it tells you how
far into the street the point sits, so you can route low-confidence cases to a human.

### Training Progress Evaluation

#### Loss Curve Analysis

```python
import matplotlib.pyplot as plt

plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('SVM Training Loss')
plt.grid(True)
plt.show()
```

**What to look for**:

```
Good Training:
Loss
 |╲
 | ╲╷╷╴╴╴╴╴╴╴╴╴╴╴╴╴  ← Steep drop, then a jittery plateau
 |                    
 +------------------ Iterations

Problem: Not Converging
Loss
 |╲  ╱╲  ╱
 | ╲╱  ╲╱    ← Oscillating, not decreasing
 |                    
 +------------------ Iterations
 → Try smaller learning rate

Problem: Slow Convergence
Loss
 |╲
 | ╲
 |  ╲        ← Still decreasing
 |   ╲       
 +------------------ Iterations
 → Try more iterations or larger learning rate
```

**A caveat about "smooth"**: this is *sub-gradient* descent on a hinge loss with a
**fixed** step size, and it does not converge to a point – it settles into a band
around the optimum and rattles inside it. So expect the real curve to drop fast and
then stay jagged, and do not read small late-stage wiggles as a bug:

Measured on the Quick Start model (500 epochs on the standardized blobs):

```python
model.losses[0]     # 1.0000  (before any update)
model.losses[1]     # 0.7982
model.losses[10]    # 0.1338
model.losses[100]   # 0.0807
model.losses[300]   # 0.0800
model.losses[-1]    # 0.0798
```

Looks like a clean decrease at 4 decimal places – but 129 of the last 450 epochs
actually made the loss go **up** slightly, and the true minimum was hit at epoch
442, four microunits below the final value (0.07983030 vs 0.07983432). That is
normal, not a failure. Canonical Pegasos avoids this by shrinking the step size as
`eta_t = 1/(lambda * t)`; see [Simplification vs. Canonical SVM](#simplification-vs-canonical-svm).
On an **unscaled** dataset the effect is much louder – the loss can climb far above
its starting value for the first few dozen epochs before it comes down.

---

## Choosing Hyperparameters

### Learning Rate

The learning rate controls how fast the model learns:

```
Too Small (0.00001):
  Pros: Stable, guaranteed convergence
  Cons: Very slow, may not reach optimum
  
Good Range (0.0001 - 0.01):
  Pros: Balanced speed and stability
  Cons: May need tuning
  
Too Large (0.1+):
  Pros: Fast initial progress
  Cons: May oscillate, miss optimum
```

**Visual comparison**:

```
Small LR:    ╲           Reaches bottom slowly
              ╲_________
              
Medium LR:    ╲         Reaches bottom efficiently
               ╲____
               
Large LR:     ╲ ╱╲ ╱    Bounces around, never settles
              ╲╱  ╲╱
```

### Lambda (Regularization Parameter)

Lambda controls the margin-error tradeoff:

```
Small λ (0.0001 - 0.001):
  Effect: Narrow margin, focus on classification
  Result: May overfit, good training accuracy
  Use when: Data is clean, need precise boundary
  
Medium λ (0.01 - 0.1):
  Effect: Balanced margin and classification
  Result: Good generalization
  Use when: Standard case, balanced priorities
  
Large λ (1.0+):
  Effect: Wide margin, tolerates errors
  Result: May underfit, prioritizes simplicity
  Use when: Noisy data, want robustness
```

**Visual comparison**:

```
Small λ (narrow margin):
    ●●●|■■■
    ●●●|■■■
    Problem: Sensitive to noise
    
Medium λ (balanced):
    ●●●  :| :  ■■■
    ●●●  :| :  ■■■
    Good: Robust and accurate
    
Large λ (wide margin):
    ●●●    :|    ■■■
    ●●● ●  :|  ■ ■■■
    Note: Tolerates misclassification
```

### Number of Iterations

```
Too Few (< 100):
  Problem: Model hasn't converged
  Sign: Loss still decreasing rapidly
  
Good Range (500-2000):
  Result: Model converged
  Sign: Loss plateaued
  
Too Many (> 5000):
  Problem: Wasted computation
  Sign: Loss unchanged for long time
  Note: Consider early stopping
```

---

## Feature Scaling: Critical for SVM

### Why Scaling Matters

SVM is **extremely sensitive** to feature scales because it uses distances!

**Example without scaling**:
```python
Feature 1: Age (20-80)          → Range = 60
Feature 2: Income ($20k-$200k)  → Range = 180,000

Distance calculation dominated by income!
Age difference of 30 years ≈ Income difference of $30
```

**Example with scaling**:
```python
Feature 1: Age (scaled to 0-1)      → Range = 1
Feature 2: Income (scaled to 0-1)   → Range = 1

Both features contribute equally!
```

### Standardization (Z-score Normalization)

Most common approach for SVM:

```
x_scaled = (x - mean) / std_dev
```

**Effect**:
- Mean = 0
- Standard deviation = 1
- Preserves outliers

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
# Result: Each feature has mean≈0, std≈1

# CRITICAL: Use same scaler for test data!
X_test_scaled = scaler.transform(X_test)
```

### Impact on SVM

```
Without Scaling:
    Decision boundary may be dominated by large-scale features
    Convergence is slow
    Weights have wildly different magnitudes
    
With Scaling:
    All features contribute proportionally
    Faster convergence
    Weights are interpretable
    Better generalization
```

---

## Advantages and Limitations

### Advantages ✅

1. **Effective in High Dimensions**
   - Works well with many features
   - Even when features > samples

2. **Memory Efficient**
   - Only support vectors matter
   - Can discard other training points (in theory)
   - *Caveat about this implementation*: storing only the support vectors is a
     property of the **dual/kernel** formulation. This primal implementation is
     even leaner – once trained it keeps just `w` and `b` and no training points
     at all. Use `get_support_vectors(X, y)` to see which points were doing the
     work.

3. **Maximum Margin**
   - Optimal separation boundary
   - Better generalization than many algorithms

4. **Robust to Outliers**
   - Focus on support vectors
   - Points far from boundary don't affect it

5. **Versatile**
   - Can use different kernels (linear, RBF, polynomial)
   - This implementation: linear kernel

6. **Clear Geometric Interpretation**
   - Easy to visualize and understand
   - Decision boundary has clear meaning

### Limitations ❌

1. **Binary Classification Only** (this implementation)
   - Need modifications for multi-class
   - Can use one-vs-rest or one-vs-one approaches

2. **Sensitive to Feature Scaling**
   - MUST scale features properly
   - Otherwise completely unreliable

3. **Choice of Hyperparameters**
   - Performance depends on λ and learning rate
   - May need cross-validation to tune

4. **No Probability Estimates** (native)
   - Only gives class labels and distances
   - Unlike logistic regression with built-in probabilities
   - `predict_proba(X)` here squashes the decision function through a sigmoid,
     `1 / (1 + exp(-f(x)))`, so you get a usable confidence ranking – but those
     numbers are **not calibrated**. Real SVM probabilities need Platt scaling
     (fitting a logistic regression on `f(x)` with cross-validation), which is
     not implemented here.

5. **Training Can Be Slow**
   - O(n² to n³) for large datasets
   - This implementation: O(n × iterations)

6. **Black Box Decision**
   - Hard to interpret feature importance
   - Unlike decision trees or linear regression

### When to Use SVM

**Good Use Cases**:
- ✅ Binary classification problems
- ✅ High-dimensional data (many features)
- ✅ Clear margin of separation exists
- ✅ Small to medium datasets (< 10k samples)
- ✅ Want maximum-margin solution
- ✅ Data is standardized

**Bad Use Cases**:
- ❌ Very large datasets (millions of samples)
- ❌ Need probability estimates (use logistic regression)
- ❌ Multi-class with many classes (complex)
- ❌ Need interpretable feature importance
- ❌ Overlapping classes with no clear separation
- ❌ Real-time predictions needed (KNN might be slow too)

---

## Simplification vs. Canonical SVM

This file implements the **primal linear soft-margin SVM trained by sub-gradient
descent**. That is a real SVM – it optimizes the real objective and lands on
essentially the same boundary a library would find – but it is not the whole
algorithm. Here is exactly what is missing and what it costs you.

### What IS implemented

The objective in `_compute_loss` / `fit` is the standard soft-margin primal:

```
L(w, b) = λ||w||² + (1/n) Σ_i max(0, 1 - y_i * (w·x_i + b))
```

Compared against `sklearn.svm.LinearSVC(loss='hinge')` and
`sklearn.svm.SVC(kernel='linear')` on the same standardized data – matching the
regularization with `C = 1 / (2 * λ * n)`, which is the exact conversion between
sklearn's `0.5||w||² + C Σ hinge` and the objective above – this implementation
agrees closely:

| Dataset | cosine(w_ours, w_sklearn) | objective ours / sklearn | test accuracy ours / sklearn |
|---------|---------------------------|--------------------------|------------------------------|
| Breast cancer (30 features, standardized) | 0.9977 vs LinearSVC, 0.9998 vs SVC | 0.078830 / 0.078808 | 0.9825 / 0.9825 |
| `make_classification` (5 features) | 0.9999 | 0.755208 / 0.755007 | 0.6875 / 0.7000 |

The weight vectors point the same way to 3-4 decimal places and the objective
value is within 0.03% of sklearn's optimum. **The math here is not a toy.**

### 1. No kernels, no dual formulation

**Canonical**: solve the dual problem
`max_a Σ a_i - ½ ΣΣ a_i a_j y_i y_j K(x_i, x_j)` subject to `0 ≤ a_i ≤ C` and
`Σ a_i y_i = 0`, then predict with `f(x) = Σ_i a_i y_i K(x_i, x) + b`. Swapping
`K` for an RBF or polynomial kernel buys non-linear boundaries for free – see
[Beyond Linear: The Kernel Trick](#beyond-linear-the-kernel-trick).

**Here**: only `K(a, b) = a·b`, expressed directly as `w·x + b`.

**Why omitted**: the dual is a constrained quadratic program. Doing it properly
means implementing SMO (working-set selection, the two-variable analytic solve,
KKT-violation bookkeeping, a kernel cache) – several hundred lines, and the
resulting code teaches SMO rather than SVM.

**Consequence**: this model can only draw straight boundaries. On data that is not
linearly separable in its raw features (concentric rings, XOR), accuracy will be
poor no matter how you tune `lambda_param`. Use `sklearn.svm.SVC(kernel='rbf')`,
or engineer non-linear features by hand before calling `fit`.

### 2. Fixed learning rate instead of the Pegasos schedule

**Canonical**: Pegasos (Shalev-Shwartz et al., 2007) uses a decaying step size
`eta_t = 1 / (λ * t)` at update `t`, which gives a proven convergence rate.

**Here**: `learning_rate` is a constant.

**Consequence**: the iterate does not converge to a point; it settles into a band
around the optimum and jitters there, so the loss curve is jagged at the tail and
`losses[-1]` is occasionally a hair above `min(losses)`. Measured on the demo's
blob data, both schedules land on the same answer (final loss 0.0798 either way,
identical train/test accuracy), so this costs accuracy only when the learning rate
is badly mismatched to the feature scale – which is another reason to standardize.

### 3. Binary classification only

**Canonical**: libraries wrap the binary solver in one-vs-rest or one-vs-one to
handle K classes.

**Here**: `fit` maps `y` to exactly two labels and raises a `ValueError` if only
one class is present.

**Consequence**: for multi-class you would train K models (one per class, "this
class vs everything else") and predict with `argmax` over their
`decision_function` values. That is a genuinely small amount of code to add on
top of this class if you want the exercise.

### 4. No calibrated probabilities

**Canonical**: `SVC(probability=True)` runs Platt scaling – it fits a logistic
regression `p = 1/(1 + exp(A*f(x) + B))` on cross-validated decision values.

**Here**: `predict_proba` uses `A = -1, B = 0` with no fitting at all, i.e. a
plain sigmoid of `f(x)`.

**Consequence**: the ordering of the confidences is meaningful, the absolute
values are not. Do not threshold them at 0.5 expecting calibrated behaviour, and
do not feed them into expected-value calculations.

### 5. No bias regularization difference, and no shrinking heuristics

The bias `b` is updated by plain sub-gradient steps and is not regularized (which
matches the standard formulation). Production solvers add shrinking heuristics,
caching and a duality-gap stopping test for speed on large data; none of that
changes the answer, so none of it is here.

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert labels to -1 and +1
y = np.where(y == 0, -1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CRITICAL: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM
model = SupportVectorMachine(
    learning_rate=0.001,
    lambda_param=0.01,
    iterations=1000
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_distances = model.decision_function(X_test_scaled)

# Evaluate model
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
# Note: Need to convert -1/+1 back to 0/1 for sklearn metrics
y_test_01 = np.where(y_test == -1, 0, 1)
y_pred_01 = np.where(y_pred == -1, 0, 1)
print(classification_report(y_test_01, y_pred_01, 
                           target_names=data.target_names))

# Analyze predictions
print("\nSample Predictions with Confidence:")
for i in range(10):
    true_label = data.target_names[y_test_01[i]]
    pred_label = data.target_names[y_pred_01[i]]
    confidence = abs(y_distances[i])
    # ASCII markers only - a checkmark glyph raises UnicodeEncodeError
    # on a default Windows (cp1252) console and kills the script here
    status = "OK" if y_pred[i] == y_test[i] else "XX"
    print(f"  {status} True: {true_label:12s} | Pred: {pred_label:12s} | "
          f"Confidence: {confidence:.4f}")

# Model parameters
params = model.get_params()
print(f"\nModel Parameters:")
print(f"  Weight vector norm: {params['norm_w']:.4f}")
print(f"  Approximate margin width: {2/params['norm_w']:.4f}")
print(f"  Bias: {params['bias']:.4f}")

# Which training points define the boundary?
sv = model.support_vector_indices_
print(f"  Support vectors: {len(sv)} of {len(X_train_scaled)} training points")

# Training progress
print(f"\nTraining Progress:")
print(f"  Initial loss (before any update): {model.losses[0]:.4f}")
print(f"  Final loss: {model.losses[-1]:.4f}")
print(f"  Improvement: {model.losses[0] - model.losses[-1]:.4f}")
```

**Expected output** (deterministic - `fit` starts from zeros and has no randomness):

```
Train Accuracy: 0.9824
Test Accuracy: 0.9825

Confusion Matrix:
[[41  2]
 [ 0 71]]

Classification Report:
              precision    recall  f1-score   support

   malignant       1.00      0.95      0.98        43
      benign       0.97      1.00      0.99        71

    accuracy                           0.98       114
   macro avg       0.99      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114


Sample Predictions with Confidence:
  OK True: benign       | Pred: benign       | Confidence: 1.0163
  OK True: malignant    | Pred: malignant    | Confidence: 4.5056
  OK True: malignant    | Pred: malignant    | Confidence: 2.2482
  OK True: benign       | Pred: benign       | Confidence: 3.0391
  OK True: benign       | Pred: benign       | Confidence: 4.2172
  OK True: malignant    | Pred: malignant    | Confidence: 9.8412
  OK True: malignant    | Pred: malignant    | Confidence: 8.3236
  OK True: malignant    | Pred: malignant    | Confidence: 1.2256
  OK True: benign       | Pred: benign       | Confidence: 0.3500
  OK True: benign       | Pred: benign       | Confidence: 2.9503

Model Parameters:
  Weight vector norm: 1.4697
  Approximate margin width: 1.3608
  Bias: 0.3120
  Support vectors: 43 of 455 training points

Training Progress:
  Initial loss (before any update): 1.0000
  Final loss: 0.0788
  Improvement: 0.9212
```

Note the last block: 43 of 455 points (about 9%) are support vectors. The other
412 could be deleted from the training set without moving the boundary. For
comparison, `sklearn.svm.SVC(kernel='linear', C=1/(2*0.01*455))` reports 52
support vectors on the same data - the small difference comes from sub-gradient
descent stopping in a band around the optimum rather than exactly on it.

---

## SVM vs Other Algorithms

### SVM vs Logistic Regression

| Aspect | SVM | Logistic Regression |
|--------|-----|---------------------|
| Goal | Maximum margin | Maximum likelihood |
| Loss | Hinge loss | Log loss |
| Output | Class + distance | Class + probability |
| Sensitivity | Robust to outliers | Affected by outliers |
| Interpretation | Geometric boundary | Probabilistic |
| Speed | Slower training | Faster training |

**When to choose**:
- SVM: Want maximum margin, clear separation
- Logistic: Need probabilities, faster training

### SVM vs KNN

| Aspect | SVM | KNN |
|--------|-----|-----|
| Training | Learns model | Just stores data |
| Prediction | Fast (linear) | Slow (distance to all points) |
| Memory | Small (weights only) | Large (all training data) |
| Decision | Global boundary | Local neighborhoods |
| Scaling | Required | Required |

**When to choose**:
- SVM: Need fast predictions, interpretable boundary
- KNN: Simple baseline, non-linear patterns

### SVM vs Decision Trees

| Aspect | SVM | Decision Trees |
|--------|-----|----------------|
| Boundary | Linear (this impl) | Axis-aligned splits |
| Interpretability | Hard | Easy |
| Feature scaling | Required | Not required |
| Overfitting | Regularization (λ) | Pruning |
| Multi-class | Complex | Natural |

**When to choose**:
- SVM: High-dimensional, clear margin
- Trees: Need interpretability, mixed feature types

---

## Key Concepts to Remember

### 1. **Maximum Margin Principle**
SVM finds the boundary with the largest margin to the nearest points.

### 2. **Support Vectors Are Key**
Only points near the boundary (support vectors) affect the decision boundary.
Concretely: they are the points with `y_i * (w·x_i + b) <= 1`, the only ones whose
hinge term is still active. Get them from a fitted model with
`model.support_vector_indices_` or `model.get_support_vectors(X, y)`.

### 3. **Hinge Loss Penalizes Margin Violations**
Points within margin or on wrong side incur loss.

### 4. **Lambda Controls Margin-Error Tradeoff**
- Large λ → wide margin, tolerates errors
- Small λ → narrow margin, fewer errors

### 5. **Feature Scaling is MANDATORY**
Always standardize features before training SVM.

### 6. **Decision Function Gives Confidence**
Magnitude of decision function indicates prediction confidence.

### 7. **Binary Classification** (this implementation)
For multi-class, use one-vs-rest or one-vs-one strategies.

---

## Conclusion

Support Vector Machine is a powerful and elegant algorithm! By understanding:
- How maximum margin provides better generalization
- How hinge loss encourages correct classification
- How regularization controls model complexity
- How feature scaling affects performance

You've gained a fundamental tool in your machine learning toolkit! 🎯

**When to Use SVM**:
- ✅ Binary classification with clear separation
- ✅ High-dimensional data
- ✅ Want maximum-margin solution
- ✅ Small to medium datasets
- ✅ Can standardize features

**When to Use Something Else**:
- ❌ Very large datasets → Use logistic regression, neural networks
- ❌ Need probabilities → Use logistic regression
- ❌ Multi-class with many classes → Use decision trees, neural networks
- ❌ Need interpretability → Use decision trees, linear regression
- ❌ Can't scale features → Use tree-based methods

**Next Steps**:
- Try SVM on your own datasets
- Experiment with different λ and learning rates
- Compare with logistic regression and KNN
- Learn about kernel SVM (non-linear boundaries) – start with the
  [kernel trick](#beyond-linear-the-kernel-trick) section above, then read
  [Simplification vs. Canonical SVM](#simplification-vs-canonical-svm) for what
  this implementation deliberately leaves out
- Study multi-class SVM extensions
- Explore support vector regression (SVR)

Happy coding! 💻🎯

