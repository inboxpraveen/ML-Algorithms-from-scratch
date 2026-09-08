# Gradient Boosting from Scratch: A Comprehensive Guide

Welcome to the world of Gradient Boosting! 🚀 In this comprehensive guide, we'll explore one of the most powerful machine learning algorithms - Gradient Boosting. Think of it as training a team of specialists where each new member focuses on correcting the mistakes of the previous team!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is Gradient Boosting?](#what-is-gradient-boosting)
3. [How Gradient Boosting Works](#how-gradient-boosting-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Computational Complexity](#computational-complexity)
11. [Simplifications vs. Canonical Gradient Boosting](#simplifications-vs-canonical-gradient-boosting)
12. [Advantages and Limitations](#advantages-and-limitations)
13. [Comparing with Alternatives](#comparing-with-alternatives)
14. [Key Concepts to Remember](#key-concepts-to-remember)
15. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Gradient Boosting from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _16_gradient_boosting.py  (the __main__ block runs this)
# Or copy the GradientBoosting class from _16_gradient_boosting.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the GradientBoosting class here (from _16_gradient_boosting.py) ----
# class GradientBoosting: ...

np.random.seed(42)

# ------ REGRESSION: predict y = x^2 + noise ------
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Shuffle before splitting: trees cannot extrapolate beyond the training range.
# Without shuffling the last 50 x-values would all be above the training max.
idx = np.random.permutation(200)
X, y = X[idx], y[idx]

X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

print(f"Train R2: {model.score(X_train, y_train):.4f}")
print(f"Test  R2: {model.score(X_test,  y_test):.4f}")
print(f"Training MSE: {model.train_loss_[0]:.4f} (1 tree) -> {model.train_loss_[-1]:.4f} (100 trees)")

preds = model.predict(X_test)
for i in range(3):
    print(f"  x={X_test[i,0]:5.2f}  true={y_test[i]:5.2f}  pred={preds[i]:5.2f}")

# ------ ROBUST REGRESSION: 'mae' leaves hold the MEDIAN residual ------
y_dirty = y_train.copy()
y_dirty[:8] += 60.0                      # 8 corrupted labels
for loss_name in ['mse', 'mae']:
    m = GradientBoosting(n_estimators=100, learning_rate=0.1,
                         max_depth=3, loss=loss_name)
    m.fit(X_train, y_dirty)
    print(f"\nloss='{loss_name}' trained on corrupted labels -> clean Test R2: "
          f"{m.score(X_test, y_test):.4f}")

# ------ CLASSIFICATION: two Gaussian blobs ------
X0 = np.random.randn(100, 2) + np.array([-2, -2])
X1 = np.random.randn(100, 2) + np.array([ 2,  2])
X_c = np.vstack([X0, X1])
y_c = np.array([0]*100 + [1]*100)
idx = np.random.permutation(200)
X_c, y_c = X_c[idx], y_c[idx]

cls = GradientBoosting(n_estimators=20, learning_rate=0.3,
                       max_depth=3, loss='log_loss')
cls.fit(X_c[:150], y_c[:150])

print(f"\nClassification accuracy: {cls.score(X_c[150:], y_c[150:]):.2%}")
proba = cls.predict_proba(X_c[150:])
for i in range(3):
    print(f"  true={y_c[150+i]}  P(0)={proba[i,0]:.4f}  P(1)={proba[i,1]:.4f}")
```

Expected output:
```
Train R2: 0.9917
Test  R2: 0.9519
Training MSE: 6.7362 (1 tree) -> 0.0676 (100 trees)
  x=-2.88  true= 8.17  pred= 9.10
  x= 0.23  true= 0.14  pred= 0.43
  x= 2.55  true= 6.38  pred= 6.87

loss='mse' trained on corrupted labels -> clean Test R2: -20.4280

loss='mae' trained on corrupted labels -> clean Test R2: 0.9559

Classification accuracy: 100.00%
  true=1  P(0)=0.0011  P(1)=0.9989
  true=0  P(0)=0.9989  P(1)=0.0011
  true=1  P(0)=0.0011  P(1)=0.9989
```

Three things to notice:
- **The shuffle is not optional.** `np.linspace` produces sorted x-values; slicing them directly hands the model a test set that lies entirely outside the range it was trained on, and a tree can only ever repeat the leaf value it learned at the edge.
- **`loss='mae'` survives the corrupted labels and `loss='mse'` does not.** That is the terminal-region line search at work: an MAE leaf stores the *median* residual of its samples, so eight wild values cannot move it, while an MSE leaf stores the *mean* and gets dragged along.
- **The probabilities are decisive (0.0011 / 0.9989), not timid.** That comes from the Newton leaf update for log loss. Both are explained in [The Mathematical Foundation](#the-mathematical-foundation).

---

## What is Gradient Boosting?

Gradient Boosting is an **ensemble learning algorithm** that builds models sequentially, where each new model corrects the errors made by previous models. Unlike AdaBoost (which adjusts sample weights), Gradient Boosting fits new models to the residual errors (negative gradients) of the combined ensemble.

**Real-world analogy**: 
Imagine you're learning to shoot basketball free throws. After your first attempt, you notice you're shooting too short. Your second attempt corrects this by shooting a bit farther. After that, you notice you're slightly to the left, so your third attempt adjusts right. Each attempt corrects the specific errors from before. Gradient Boosting works the same way - each model corrects what previous models got wrong!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Ensemble Learning (Boosting) |
| **Learning Style** | Supervised Learning |
| **Primary Use** | Regression and Classification |
| **Base Learners** | Decision trees (typically shallow) |
| **Key Principle** | Fit models to negative gradients (residuals) |

### The Core Idea

```
"Each new model learns to predict the errors of the previous models"
```

This powerful principle works through:
- **Sequential learning**: Models are trained one after another
- **Error correction**: Each model focuses on what previous models missed
- **Gradient descent in function space**: Minimizes loss function step by step
- **Flexible**: Works with any differentiable loss function

### Key Concepts

**1. Loss Function**: Measures how far predictions are from truth
```
Regression: MSE = mean((y_true - y_pred)²)
            (the gradient derivation later uses the halved form ½(y - F)²;
             the ½ only cancels the 2 in the derivative and does not move
             the minimum - see The Mathematical Foundation)
Classification: Log Loss = -mean(y·log(p) + (1-y)·log(1-p))
                           where p = sigmoid(F(x))
```

**2. Gradient (Residual)**: Direction to improve predictions
```
For MSE: gradient = y_pred - y_true
         (how much to reduce each prediction)

For each step, fit a model to: -gradient
```

**3. Sequential Updates**: Each model improves the ensemble
```
F₀(x) = initial prediction (mean for regression)
F₁(x) = F₀(x) + learning_rate × tree₁(x)
F₂(x) = F₁(x) + learning_rate × tree₂(x)
...
Fₘ(x) = Fₘ₋₁(x) + learning_rate × treeₘ(x)
```

**4. Learning Rate**: Controls contribution of each tree
```
learning_rate = 0.1:  Conservative, needs more trees
learning_rate = 0.3:  Aggressive, faster convergence
learning_rate = 0.01: Very conservative, best generalization
```

---

## How Gradient Boosting Works

### The Algorithm in 5 Steps

```
Step 1: Initialize predictions (mean for regression, log-odds for classification)
         ↓
Step 2: Calculate negative gradient (residuals/errors)
         ↓
Step 3: Fit a decision tree to the negative gradient
         ↓
Step 4: Update predictions: F(x) = F(x) + learning_rate × tree(x)
         ↓
Step 5: Repeat Steps 2-4 for M iterations
         ↓
Final: F(x) = F₀(x) + learning_rate × Σ treeᵢ(x)
```

### Visual Example: Regression

Let's predict house prices with 5 data points:

```
Data:
x (size):  [1000, 1500, 2000, 2500, 3000] sqft
y (price): [100,  180,  240,  280,  350]  k$

Goal: Predict price from size
```

**Iteration 0: Initialize**

```
F₀(x) = mean(y) = (100 + 180 + 240 + 280 + 350) / 5 = 230

Current predictions: [230, 230, 230, 230, 230]
Residuals (errors): [-130, -50, 10, 50, 120]
                    (y_true - y_pred)
```

**Iteration 1: Fit first tree to residuals**

```
Residuals: [-130, -50, 10, 50, 120]

Train tree₁ to predict these residuals:
  Best split: x ≤ 1750
    Left (x ≤ 1750):  mean residual = (-130 - 50) / 2 = -90
    Right (x > 1750): mean residual = (10 + 50 + 120) / 3 = 60

Tree₁ predictions: [-90, -90, 60, 60, 60]

Update predictions (learning_rate = 0.1):
  F₁(x) = F₀(x) + 0.1 × tree₁(x)
  F₁(x) = [230, 230, 230, 230, 230] + 0.1 × [-90, -90, 60, 60, 60]
        = [221, 221, 236, 236, 236]

New residuals: [100-221, 180-221, 240-236, 280-236, 350-236]
             = [-121, -41, 4, 44, 114]
```

**Iteration 2: Fit second tree to new residuals**

```
Residuals: [-121, -41, 4, 44, 114]

Train tree₂:
  Split: x ≤ 1250
    Left:  mean = (-121 - 41) / 2 = -81
    Right: mean = (4 + 44 + 114) / 3 = 54

Tree₂ predictions: [-81, -81, 54, 54, 54]

Update:
  F₂(x) = F₁(x) + 0.1 × tree₂(x)
        = [221, 221, 236, 236, 236] + 0.1 × [-81, -81, 54, 54, 54]
        = [212.9, 212.9, 241.4, 241.4, 241.4]

New residuals: [-112.9, -32.9, -1.4, 38.6, 108.6]
```

**Continue for M iterations...**

After many iterations:
```
Final predictions approach true values:
F₁₀₀(x) ≈ [100, 180, 240, 280, 350]

Each tree corrects residual errors!
```

### Why Sequential Error Correction Works

**Traditional approach (single model)**:
```
Single complex tree:
  Tries to learn everything at once
  May overfit or underfit
  Hard to generalize
  
Result: ~80% accuracy
```

**Gradient Boosting approach**:
```
Tree 1: Learns main trends (simple patterns)
        → Residuals still large
        
Tree 2: Learns what Tree 1 missed (medium patterns)
        → Residuals getting smaller
        
Tree 3: Learns remaining errors (fine patterns)
        → Residuals very small
        
Trees 4-100: Continue refining
        
Result: ~95% accuracy!
```

**The Magic**: Each tree specializes in correcting different types of errors, creating a comprehensive solution!

---

## The Mathematical Foundation

### 1. Loss Functions

The loss function measures prediction quality:

**Mean Squared Error (Regression)**:
```
L(y, F(x)) = ½(y - F(x))²

Gradient: ∂L/∂F = F(x) - y
Negative gradient: y - F(x) = residuals

→ Fit trees to residuals!
```

**Mind the ½ - there are two scales in play, and they are both correct.**
The ½ is bookkeeping: it cancels the 2 that the derivative produces, which is the
only reason `∂L/∂F` comes out as the clean `F(x) - y`. Halving a loss cannot move
its minimiser, so the trees, the leaf values and every prediction are identical
with or without it. But the *number you read* does change, so keep them straight:

| Quantity | Formula | Where |
|---|---|---|
| the loss that is differentiated | `½(y - F)²` | `_mse_gradient` returns its `F - y` |
| the loss that is reported | `mean((y - F)²)` | `_mse_loss`, stored in `train_loss_`, printed as "Training MSE" |

The reported one is exactly **twice** `mean(½(y - F)²)`. Same minimiser, same
argmin in every leaf, different constant - so do not be surprised when
`train_loss_[-1]` is double what you get by evaluating `½(y - F)²` by hand.

**Log Loss (Binary Classification)**:
```
L(y, F(x)) = -y·log(p) - (1-y)·log(1-p)

where p = sigmoid(F(x)) = 1/(1 + e^(-F(x)))

Gradient: ∂L/∂F = p - y
Negative gradient: y - p

→ Fit trees to (y - p)!
```

### 2. Gradient Descent in Function Space

Regular gradient descent optimizes parameters:
```
θₜ₊₁ = θₜ - learning_rate × ∇L(θₜ)
```

Gradient Boosting optimizes functions:
```
Fₜ₊₁(x) = Fₜ(x) - learning_rate × ∇L(Fₜ(x))
        = Fₜ(x) + learning_rate × hₜ(x)

where hₜ(x) is fitted to -∇L(Fₜ(x))
```

**Example**:
```
Current predictions: F(x) = [5, 10, 15]
True values:         y    = [3, 12, 14]
Loss: MSE

Gradients: ∇L = F(x) - y = [2, -2, 1]
Negative gradients: -∇L = [-2, 2, -1]

Fit tree to [-2, 2, -1]:
  Tree predicts: h(x) = [-1.8, 1.9, -0.9]

Update (learning_rate = 0.1):
  F_new(x) = [5, 10, 15] + 0.1 × [-1.8, 1.9, -0.9]
           = [4.82, 10.19, 14.91]

Closer to true values!
```

### 3. The Gradient Boosting Algorithm (Formal)

**Input**: 
- Training data: {(xᵢ, yᵢ)}ⁿᵢ₌₁
- Loss function: L(y, F(x))
- Number of iterations: M
- Learning rate: η

**Algorithm**:
```
1. Initialize model with constant:
   F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)
   
   For squared error:   F₀(x) = mean(y)
   For absolute error:  F₀(x) = median(y)
   For classification:  F₀(x) = log(p/(1-p))

2. For m = 1 to M:
   
   a. Compute negative gradient (pseudo-residuals):
      rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₘ₋₁}
   
   b. Fit regression tree to {(xᵢ, rᵢₘ)}ⁿᵢ₌₁, giving terminal
      regions (leaves) R₁ₘ, R₂ₘ, ..., R_Jm
   
   c. Terminal-region line search - for each leaf j, choose the
      constant that minimises the ORIGINAL loss inside that leaf:
      γⱼₘ = argmin_γ Σ_{xᵢ ∈ Rⱼₘ} L(yᵢ, Fₘ₋₁(xᵢ) + γ)
   
   d. Update model:
      Fₘ(x) = Fₘ₋₁(x) + η · Σⱼ γⱼₘ · 1(x ∈ Rⱼₘ)

3. Output: F_M(x)
```

**Step 2c is the step people skip - and it matters.**

The tree in step 2b decides *where* the boundaries go. Step 2c decides *how far to step* inside each region. Fitting a tree to the gradient and then just using the tree's own leaf averages is only correct when the loss is squared error, because that is the one case where "average of the negative gradient" and "argmin of the loss" are the same number.

Solving the argmin in closed form for each of our three losses:

```
MSE      L = ½(y - F)²
         d/dγ Σ ½(yᵢ - Fᵢ - γ)² = 0
         γⱼₘ = mean(rᵢ)                  where rᵢ = yᵢ - Fₘ₋₁(xᵢ)
         (the tree already stores this, so no correction is needed)

MAE      L = |y - F|
         Σ |rᵢ - γ| is minimised by the MIDDLE residual
         γⱼₘ = median(rᵢ)
         (NOT mean(sign(rᵢ)), which is what the tree's own leaf holds
          and which is trapped in [-1, 1] no matter how large the errors)

         With an even count, every value between the two central residuals
         is an equally good minimiser - np.median's average of the two
         included. The code takes the LOWER central residual,
         sorted(r)[(n-1)//2], as a tie-break rather than as a better argmin:
         it is an actually observed value, and it is what scikit-learn's
         weighted 50th percentile returns, which is why the uniform-X 'mae'
         reference check below matches sklearn to 0.0e+00.

Log Loss L = -y·log(p) - (1-y)·log(1-p),  p = sigmoid(F)
         no closed form, so take one Newton step from γ = 0:
         γⱼₘ = -Σ(∂L/∂F) / Σ(∂²L/∂F²) = Σ rᵢ / Σ pᵢ(1 - pᵢ)
         with rᵢ = yᵢ - pᵢ and pᵢ = sigmoid(Fₘ₋₁(xᵢ))
```

Read the log-loss formula as a confidence dial. The numerator `Σ rᵢ` is bounded by the leaf's sample count, but the denominator `Σ pᵢ(1 - pᵢ)` *shrinks* as the model grows confident (p near 0 or 1 makes p(1-p) tiny). So the step size grows exactly where the model is already sure, which is what drives log loss towards zero. Using the raw leaf mean `mean(y - p)` instead caps every step at 1 and leaves the model permanently under-confident.

`_update_leaf_values()` in `_16_gradient_boosting.py` implements exactly these three lines, and `fit()` calls it right after `_create_decision_tree()` returns.

**Why does a leaf hold a single constant at all?**

Because a tree is not a function of `x` in any smooth sense - it is a *partition*. It
carves the input space into J disjoint boxes and then has nothing left to say about
where inside a box a point sits. So the only freedom left is one number per box, and
step 2c picks that number optimally. This is also why boosting composes so well: one
depth-3 tree can only produce 8 distinct values, but a hundred of them, each shifted
by η, add up to a finely graded surface.

It also explains the algorithm's blind spot. Every leaf constant was learned from
training points that fell inside that box, so a test point outside the training range
lands in an edge box and receives that box's constant - forever. Gradient boosting
cannot extrapolate, which is precisely why every example in these files shuffles
before splitting.

**What the log-odds initialization buys you**

For classification the model works in log-odds space, not probability space:
`F(x)` is unbounded and `p = sigmoid(F(x))` squashes it into (0, 1). Setting
`F₀ = log(p̄ / (1 - p̄))`, where `p̄` is the training base rate, means the model starts
out predicting exactly the class balance and every subsequent tree spends its capacity
on *deviations* from that base rate rather than rediscovering it. On a 5%-positive
fraud dataset this is the difference between starting at `p = 0.05` and starting at
`p = 0.5`, i.e. several wasted boosting rounds. It also keeps the additive update
legal: adding a tree to a probability could push it past 1, but adding a tree to a
log-odds score never leaves the valid range.

```
Base rate p̄ = 0.05  ->  F₀ = log(0.05 / 0.95) = -2.944  ->  sigmoid(F₀) = 0.05 ✓
Base rate p̄ = 0.50  ->  F₀ = log(0.50 / 0.50) =  0.000  ->  sigmoid(F₀) = 0.50 ✓
```

**Example Calculation**:
```
Iteration 1:
  Current: F₀(x) = 5.0
  True: y = 8.0
  Loss: MSE = ½(y - F)²
  
  Gradient: ∂L/∂F = F - y = 5.0 - 8.0 = -3.0
  Negative gradient: -(-3.0) = 3.0
  
  Fit tree to 3.0: h₁(x) = 2.8
  
  Update: F₁(x) = 5.0 + 0.1 × 2.8 = 5.28
  
  New error: 8.0 - 5.28 = 2.72 (smaller!)
```

### 4. Learning Rate and Number of Trees

**Trade-off**:
```
learning_rate × n_trees ≈ constant performance

Examples:
  η = 1.0,   M = 50   → Fast, may overfit
  η = 0.1,   M = 500  → Balanced
  η = 0.01,  M = 5000 → Slow, best generalization
```

**Why shrinkage (low learning rate) helps**:
```
High learning rate (η = 1.0):
  Each tree makes large corrections
  May overshoot optimal solution
  Risk of overfitting early

Low learning rate (η = 0.1):
  Each tree makes small corrections
  More trees needed, but smoother path
  Better generalization
  Less likely to overfit
```

**Mathematical intuition**:
```
Without shrinkage:
  F_M(x) = F₀(x) + Σₘ hₘ(x)
  
  Early trees dominate (fitted to large residuals)
  Later trees less important

With shrinkage:
  F_M(x) = F₀(x) + η·Σₘ hₘ(x)
  
  All trees contribute more equally
  Ensemble is more robust
```

### 5. Tree Depth and Complexity

**Shallow trees (depth 3-5)** - Recommended:
```
Advantages:
  ✓ Faster training
  ✓ Less overfitting
  ✓ Good interaction modeling (2-5 features)
  ✓ Ensemble of many simple models

Each tree captures simple patterns:
  Tree 1: if x₁ > 5 and x₂ ≤ 10 → predict +2
  Tree 2: if x₁ ≤ 3 → predict -1
  Tree 3: if x₃ > 7 and x₁ > 5 → predict +0.5
  
Combined: Complex decision boundary!
```

**Deep trees (depth 8+)**:
```
Disadvantages:
  ✗ Slower training
  ✗ Risk of overfitting
  ✗ High variance
  
Use only when:
  - Very complex patterns
  - Large amounts of data
  - Heavy regularization
```

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, 
                 max_depth=3, min_samples_split=2, 
                 loss='mse', subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.init_prediction = None
        self.train_loss_ = []
```

### Core Methods

1. **`__init__(...)`** - Initialize model
   - n_estimators: Number of trees
   - learning_rate: Shrinkage parameter
   - max_depth: Tree complexity
   - loss: 'mse', 'mae', or 'log_loss'
   - subsample: Row fraction per tree (stochastic gradient boosting)
   - random_state: Seed for the subsampling draw, so `subsample < 1.0` is reproducible

2. **`_get_gradient(y_true, y_pred)`** - Calculate gradients
   - Returns ∂L/∂F, the gradient of the loss with respect to the current
     prediction (e.g. `y_pred - y_true` for MSE)
   - `fit()` negates it to get the pseudo-residuals the tree is fitted to
   - Different for each loss function

3. **`_create_decision_tree(X, y, depth)`** - Build regression tree
   - Recursively splits to minimize variance
   - Candidate thresholds are midpoints between consecutive distinct values
   - Returns tree structure (dict), each split node recording the gain it achieved

4. **`_update_leaf_values(tree, X, y_true, current_predictions)`** - Line search
   - Friedman's step 2c: replaces each leaf constant with the value that
     minimises the original loss for the samples in that leaf
   - median(r) for 'mae', Σr / Σp(1-p) for 'log_loss', no-op for 'mse'

5. **`_predict_tree(tree, X)`** - Predict with single tree
   - Traverses tree structure
   - Returns predictions for all samples

6. **`fit(X, y)`** - Train the ensemble
   - Initialize with mean / median / log-odds depending on the loss
   - Sequentially fit trees to gradients, then run the leaf line search
   - Update predictions each iteration and record the loss in `train_loss_`

7. **`predict(X)`** - Make predictions
   - Sum all tree predictions
   - Apply sigmoid for classification

8. **`predict_proba(X)`** - Predict probabilities
   - For classification only
   - Returns P(class=0), P(class=1)

9. **`score(X, y)`** - Evaluate performance
   - R² for regression
   - Accuracy for classification

10. **`staged_predict(X)` / `staged_score(X, y)`** - Learning curves
    - Predictions/scores after each tree
    - Useful for finding optimal n_estimators

11. **`get_feature_importance()`** - Feature importance
    - Based on summed variance reduction (gain), not split counts
    - Normalized to sum to 1

---

## Step-by-Step Example

Let's walk through a complete example of **regression**:

### The Data

```python
import numpy as np

# Create synthetic data: y = x² + noise
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(100) * 0.5

# Shuffle first! np.linspace returns sorted x-values, so slicing straight
# away would give a test set that lies entirely outside the training range.
indices = np.random.permutation(100)
X, y = X[indices], y[indices]

# Split train/test - the two slices must not overlap
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]
```

### Training the Model

```python
# Paste the GradientBoosting class from _16_gradient_boosting.py above,
# or import it if you have added the folder to your path.

# Create model
model = GradientBoosting(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3
)

# Train
model.fit(X_train, y_train)
```

**What happens internally - Iteration 0**:

All the numbers below were read out of an actual run of the snippet above
(`model.init_prediction`, `y_train[:5]`, `model.trees[0]`).

```
Initialize:
  F₀(x) = mean(y_train) = 2.9504

Current predictions: all samples = 2.9504
Residuals: y - F₀(x)

Example for first 5 shuffled training samples:
  x:         [-1.18, 0.39, -2.21, -0.52, -2.58]
  y:         [ 1.10, -0.26,  3.94,  0.35,  7.02]
  F₀(x):     [ 2.95,  2.95,  2.95,  2.95,  2.95]
  Residuals: [-1.85, -3.21,  0.99, -2.60,  4.07]

(The shuffle is why the y-values jump around instead of descending
 smoothly - x is no longer in sorted order.)
```

**Iteration 1**: Fit first tree

```
Gradients (MSE): ∂L/∂F = F₀(x) - y = -residuals
Negative gradients: residuals = [-1.85, -3.21, 0.99, -2.60, 4.07, ...]

Fit tree₁ to these residuals. Forcing max_depth=1 for readability, the
single best split is:
  Best split found: x ≤ -2.2424
    Left branch  (x ≤ -2.2424): mean residual =  4.2091
    Right branch (x > -2.2424): mean residual = -0.5336

Tree₁(x) = 4.2091 if x ≤ -2.2424, else -0.5336

Update predictions (learning_rate = 0.1):
  F₁(x) = F₀(x) + 0.1 × tree₁(x)
  
  For x = -2.5: F₁ = 2.9504 + 0.1 × ( 4.2091) = 3.3713
  For x =  0.4: F₁ = 2.9504 + 0.1 × (-0.5336) = 2.8970

(The real model uses max_depth=3, so tree₁ has 8 leaves and splits at
 x ≤ -2.2424 first - the same root, then six more splits below it.)
```

**Iteration 2**: Fit second tree

```
Residuals shrink slightly everywhere, and the next tree finds a
different root split because the largest remaining error has moved.

Update:
  F₂(x) = F₁(x) + 0.1 × tree₂(x)
```

**After 50 iterations**:

```
Final model:
  F₅₀(x) = 2.9504 + 0.1 × [tree₁(x) + tree₂(x) + ... + tree₅₀(x)]

Training MSE fell from 5.8929 (after tree 1) to 0.0785 (after tree 50);
model.train_loss_ holds the whole sequence.

Predictions now closely follow y = x²!
```

### Making Predictions

```python
# Predict on test data
predictions = model.predict(X_test)

# Evaluate
test_score = model.score(X_test, y_test)
print(f"Test R2: {test_score:.4f}")

# Sample predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"x: {X_test[i, 0]:5.2f}, "
          f"True: {y_test[i]:5.2f}, "
          f"Predicted: {predictions[i]:5.2f}")
```

**Output** (captured from an actual run, not hand-written):
```
Test R2: 0.9713

Sample Predictions:
x:  0.33, True:  0.58, Predicted:  0.25
x: -2.88, True:  8.61, Predicted:  8.62
x: -2.94, True:  8.57, Predicted:  9.07
x: -0.70, True: -0.18, Predicted:  0.22
x: -2.33, True:  5.21, Predicted:  5.45
```

### Visualizing Learning Progress

```python
# Get scores after each tree
train_scores = model.staged_score(X_train, y_train)
test_scores = model.staged_score(X_test, y_test)

# Find optimal number of trees
optimal_n = np.argmax(test_scores) + 1
print(f"Optimal trees: {optimal_n}")
print(f"Best test R2: {test_scores[optimal_n-1]:.4f}")
```

Measured on this dataset:
```
Optimal trees: 39
Best test R2: 0.9715

           train R2   test R2
after  1     0.1694    0.1608
after 10     0.8206    0.8159
after 30     0.9808    0.9700
after 50     0.9889    0.9713   (plateaued)
```

Notice how small the train/test gap stays (0.9889 vs 0.9713). That is depth-3 trees
plus a 0.1 learning rate doing their job - see [Detecting Overfitting](#detecting-overfitting)
for what the same curves look like when they go wrong.

---

## Real-World Applications

### 1. **Ranking and Search Engines**
The #1 use case for Gradient Boosting!
- Input: Query-document pairs with features (relevance scores)
- Output: Ranking score
- Example: Google Search, Bing use gradient boosting variants
- **Business Value**: Better search results, higher user satisfaction

**How it works**:
```
Features for query "machine learning":
  Document A: [title_match: 1.0, content_match: 0.8, pagerank: 0.9]
  Document B: [title_match: 0.5, content_match: 0.9, pagerank: 0.6]
  
Gradient Boosting learns optimal ranking:
  Tree 1: Heavily weights title_match
  Tree 2: Balances with content quality
  Tree 3: Adds pagerank consideration
  ...
  
Result: Documents ranked by learned relevance
```

### 2. **House Price Prediction**
One of the most accurate methods for real estate:
- Input: Size, location, age, features
- Output: Predicted price
- Example: Zillow's Zestimate uses gradient boosting
- **Business Value**: Accurate property valuations

**Applications**:
```
Features: [size, bedrooms, bathrooms, age, distance_to_city, school_rating]

Gradient Boosting captures:
  - Non-linear size effects (price/sqft varies with size)
  - Feature interactions (size × location)
  - Market segments (luxury vs. standard)
  
Achieves R² > 0.9 on many markets!
```

### 3. **Click-Through Rate (CTR) Prediction**
Essential for online advertising:
- Input: User features, ad features, context
- Output: Probability of click
- Example: Facebook Ads, Google AdWords
- **Business Value**: Billions in advertising revenue

**Example**:
```
Features:
  User: [age: 28, interests: sports, location: NYC]
  Ad: [category: shoes, price: high, brand: Nike]
  Context: [time: evening, device: mobile]

Model predicts: P(click) = 0.034 (3.4%)

Used for:
  - Ad ranking (show highest CTR ads)
  - Bid optimization (bid based on expected clicks)
  - Budget allocation
```

### 4. **Customer Churn Prediction**
Identify customers likely to leave:
- Input: Usage patterns, demographics, support interactions
- Output: Churn probability
- Example: Telecom, SaaS, subscription services
- **Business Value**: Proactive retention saves revenue

**Example**:
```
Features: [tenure_months, monthly_usage, support_calls, 
           competitor_contact, payment_delays]

Gradient Boosting identifies patterns:
  Tree 1: Declining usage is red flag
  Tree 2: Combined with support calls → high risk
  Tree 3: Recent competitor contact → urgent
  
Action: Target high-risk customers with retention offers
Success rate: 60% churn prevention with top 10% highest risk
```

### 5. **Credit Scoring**
Assess loan default risk:
- Input: Credit history, income, debt, employment
- Output: Default probability
- Example: FICO alternatives, P2P lending
- **Business Value**: Better risk management, fewer defaults

**Example**:
```
Features: [credit_score, income, debt_to_income, employment_years,
           recent_inquiries, delinquencies]

Model learns complex risk patterns:
  - Income matters more for low credit scores
  - Employment stability crucial for high debt
  - Recent inquiries worse for short credit history
  
Result: More accurate than linear scoring
Reduces default rate by 15-20%
```

### 6. **Medical Diagnosis and Prognosis**
Predict disease risk or outcomes:
- Input: Symptoms, test results, patient history
- Output: Disease probability or survival time
- Example: Cancer prognosis, diabetes risk, ICU mortality
- **Business Value**: Better patient care, resource allocation

**Example**:
```
Features: [age, biomarkers, genetic factors, lifestyle, medical_history]

Gradient Boosting for diabetes risk:
  Tree 1: High glucose + high BMI
  Tree 2: Family history + age
  Tree 3: Lifestyle factors
  
Accuracy: 85%+ for 5-year diabetes prediction
Allows early intervention!
```

**Note**: For educational purposes only - medical decisions require professional evaluation!

### 7. **Fraud Detection**
Identify fraudulent transactions:
- Input: Transaction features, user behavior, patterns
- Output: Fraud probability
- Example: Credit card fraud, insurance fraud
- **Business Value**: Prevented fraud losses

**Example**:
```
Features: [amount, merchant_category, location, time, 
           user_history, device_id, velocity]

Gradient Boosting detects subtle patterns:
  - Unusual amount for user
  - New location + high amount
  - Multiple transactions in short time
  - Device mismatch
  
Real-time scoring: < 50ms per transaction
Catch rate: 85% of fraud with 1% false positive rate
```

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Calculating Gradients

```python
def _mse_gradient(self, y_true, y_pred):
    """Gradient of L = (1/2)(y - F)^2: dL/dF = F - y, the negative residuals"""
    return y_pred - y_true

def _mae_gradient(self, y_true, y_pred):
    """Gradient of MAE: sign of residuals"""
    return np.sign(y_pred - y_true)

def _get_gradient(self, y_true, y_pred):
    """Calculate gradient based on loss function"""
    if self.loss == 'mse':
        return self._mse_gradient(y_true, y_pred)
    elif self.loss == 'mae':
        return self._mae_gradient(y_true, y_pred)
    elif self.loss == 'log_loss':
        return self._log_loss_gradient(y_true, y_pred)
```

**How it works**:
```python
# Example for MSE
y_true = np.array([10, 20, 30])
y_pred = np.array([12, 18, 32])

gradient = y_pred - y_true
         = [12-10, 18-20, 32-30]
         = [2, -2, 2]

# This tells us:
# Sample 0: predicted too high by 2 → need to decrease
# Sample 1: predicted too low by 2 → need to increase
# Sample 2: predicted too high by 2 → need to decrease

# We fit tree to NEGATIVE gradient:
negative_gradient = -[2, -2, 2] = [-2, 2, -2]

# Tree learns to predict these corrections!
```

### 2. Building Decision Trees

```python
def _create_decision_tree(self, X, y, depth=0):
    """Create regression tree to predict gradients"""
    
    # Stopping criteria
    if depth >= self.max_depth or n_samples < self.min_samples_split:
        return {'type': 'leaf', 'value': np.mean(y)}
    
    # Find best split
    for feature_idx in range(n_features):
        # Candidates are MIDPOINTS between consecutive distinct values,
        # so the boundary never sits exactly on a training point
        distinct_values = np.unique(X[:, feature_idx])
        thresholds = (distinct_values[:-1] + distinct_values[1:]) / 2

        for threshold in thresholds:
            # Calculate variance reduction
            left_mask = X[:, feature_idx] <= threshold
            gain = current_var - (left_var + right_var)
            
            if gain > best_gain:
                best_gain = gain
                # Store best split
```

**Step-by-step example** (every number below is `np.var`'s actual output):
```python
# Data to fit (gradients)
X = [[1], [2], [3], [4], [5], [6]]
y = [-2, -1, -1, 1, 2, 3]  # gradients to fit

# Current (weighted) variance at this node
np.var(y) = 3.2222
current_variance = np.var(y) × 6 = 19.3333

# Try split at the midpoint x ≤ 3.5  (= (3 + 4) / 2):
left_y  = [-2, -1, -1]      # x ≤ 3.5, i.e. x = 1, 2, 3
right_y = [1, 2, 3]         # x > 3.5, i.e. x = 4, 5, 6

left_var  = np.var(left_y)  × 3 = 0.2222 × 3 = 0.6667
right_var = np.var(right_y) × 3 = 0.6667 × 3 = 2.0000

total_var = 0.6667 + 2.0000 = 2.6667
gain = 19.3333 - 2.6667 = 16.6667 (good split!)

# The loop scores every midpoint and keeps the best. The next candidate,
# x ≤ 4.5, puts [-2,-1,-1,1] on the left and [2,3] on the right and scores
#   19.3333 - (1.1875 × 4 + 0.25 × 2) = 19.3333 - 5.25 = 14.0833
# which is worse, so 3.5 wins. The whole sweep, for thresholds
# 1.5 / 2.5 / 3.5 / 4.5 / 5.5, is 6.5333 / 10.0833 / 16.6667 / 14.0833 / 8.5333.

# Create tree (exactly what _create_decision_tree returns here at max_depth=1):
{
  'type': 'split',
  'feature': 0,
  'threshold': 3.5,
  'gain': 16.6667,                             # used by get_feature_importance()
  'left': {'type': 'leaf', 'value': -1.3333},  # mean of [-2,-1,-1]
  'right': {'type': 'leaf', 'value': 2.0}      # mean of [1,2,3]
}
```

Note that the gain is *weighted* variance (`np.var(...) × n_samples`), not raw
variance. Without the weight a two-sample leaf would count as much as a
two-hundred-sample leaf, and the tree would happily chase tiny pockets of noise.

### 3. Fitting the Ensemble

```python
def fit(self, X, y):
    # Initialize with the loss-minimising constant F_0
    if self.loss == 'log_loss':
        p = np.clip(np.mean(y), 1e-10, 1 - 1e-10)
        self.init_prediction = np.log(p / (1 - p))   # log-odds
    elif self.loss == 'mae':
        self.init_prediction = np.median(y)          # median, not mean
    else:
        self.init_prediction = np.mean(y)
    current_predictions = np.full(n_samples, self.init_prediction)
    
    # Train trees sequentially
    for i in range(self.n_estimators):
        # Calculate negative gradients (pseudo-residuals)
        gradients = -self._get_gradient(y, current_predictions)
        
        # Fit tree to gradients -> this fixes the tree STRUCTURE
        tree = self._create_decision_tree(X, gradients)
        
        # Line search -> this fixes the leaf CONSTANTS (Friedman step 2c)
        if self.loss != 'mse':
            self._update_leaf_values(tree, X, y, current_predictions)
        
        self.trees.append(tree)
        
        # Update predictions
        tree_predictions = self._predict_tree(tree, X)
        current_predictions += self.learning_rate * tree_predictions
        
        # Record the loss so the caller can watch it fall
        self.train_loss_.append(self._compute_loss(y, current_predictions))
```

**The two-phase split is the whole trick.** Phase one asks "where are the errors
clustered?" and answers it with variance reduction on the gradient. Phase two asks
"how big a correction does each cluster need?" and answers it with the closed-form
argmin of the real loss. Because the structure search only ever sees the gradient,
the same tree-growing code serves regression, robust regression and classification;
only `_update_leaf_values` changes.

**Detailed execution trace**:
```python
# Initial state
X = [[1], [2], [3], [4], [5]]
y = [1, 4, 9, 16, 25]  # y = x²

# Iteration 0: Initialize
init_prediction = mean(y) = (1+4+9+16+25)/5 = 11
current_predictions = [11, 11, 11, 11, 11]

# Iteration 1
gradients = y - current_predictions = [-10, -7, -2, 5, 14]
tree₁ fitted to gradients
tree₁ predictions = [-8, -6, -1, 6, 13]  (learned pattern)

current_predictions = [11, 11, 11, 11, 11] + 0.1 × [-8, -6, -1, 6, 13]
                    = [10.2, 10.4, 10.9, 11.6, 12.3]

# Iteration 2
new_gradients = y - [10.2, 10.4, 10.9, 11.6, 12.3]
              = [-9.2, -6.4, -1.9, 4.4, 12.7]

tree₂ fitted to new gradients...

# After 50 iterations
final_predictions ≈ [1, 4, 9, 16, 25]  (very close!)
```

### 4. Making Predictions

```python
def predict(self, X):
    # Start with initial prediction
    predictions = np.full(n_samples, self.init_prediction)
    
    # Add contribution from each tree
    for tree in self.trees:
        tree_predictions = self._predict_tree(tree, X)
        predictions += self.learning_rate * tree_predictions
    
    return predictions
```

**Example**:
```python
# Test sample
X_test = [[3.5]]

# Initial prediction
pred = init_prediction = 11.0

# Add each tree's contribution
pred += 0.1 × tree₁.predict([3.5])  # +0.25
pred += 0.1 × tree₂.predict([3.5])  # +0.18
pred += 0.1 × tree₃.predict([3.5])  # +0.12
...
pred += 0.1 × tree₅₀.predict([3.5]) # +0.01

# Final prediction
pred ≈ 12.25 (true value: 3.5² = 12.25)  ✓
```

### 5. Subsampling (Stochastic Gradient Boosting)

```python
def fit(self, X, y):
    # A private RandomState when random_state is given, otherwise the global
    # RNG (so np.random.seed(...) still controls the run)
    rng = np.random if self.random_state is None else np.random.RandomState(self.random_state)

    for i in range(self.n_estimators):
        gradients = -self._get_gradient(y, current_predictions)
        
        # Subsample data
        if self.subsample < 1.0:
            sample_size = int(n_samples * self.subsample)
            indices = rng.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
            gradients_sample = gradients[indices]
```

**Why subsampling helps**:
```python
# Without subsampling (subsample=1.0):
All samples used for each tree
Each tree sees same data
Risk: Trees become too similar
      Overfitting to training data

# With subsampling (subsample=0.8):
Random 80% of samples per tree
Each tree sees different data
Benefit: More diverse trees
         Better generalization
         Reduced overfitting

Example with 100 samples:
Tree 1: trained on samples [3, 7, 12, 15, ..., 97] (80 samples)
Tree 2: trained on samples [1, 5, 8, 19, ..., 99] (80 different samples)
Tree 3: trained on samples [2, 4, 11, 13, ..., 95] (80 different samples)

Each tree learns slightly different patterns → robust ensemble!
```

---

## Model Evaluation

### Choosing Parameters

#### Number of Estimators (n_estimators)

```
Small (10-50):
  ✓ Very fast training
  ✗ May underfit
  ✗ Not enough error correction
  
Medium (100-300):
  ✓ Good balance
  ✓ Usually sufficient
  ✓ Reasonable training time
  
Large (500-2000):
  ✓ Best performance on large datasets
  ✗ Longer training
  ✗ May overfit without regularization
  
Very Large (2000+):
  ✗ Diminishing returns
  ✗ Very slow training
  ✓ Use only with very low learning rate
```

**How to choose**:
```python
# Use learning curves
train_scores = model.staged_score(X_train, y_train)
test_scores = model.staged_score(X_test, y_test)

# Plot and find where test score plateaus
# Choose n_estimators at that point

# Typical patterns:
# - Test score increases then plateaus: good!
# - Test score decreases after peak: overfitting!
#   → Use early stopping at peak
```

#### Learning Rate

```
High (0.3-1.0):
  ✓ Fast convergence
  ✓ Fewer trees needed
  ✗ May overfit
  ✗ Can overshoot optimum
  
Medium (0.1-0.3):
  ✓ Good default
  ✓ Balanced speed and accuracy
  ✓ Works for most problems
  
Low (0.01-0.1):
  ✓ Best generalization
  ✓ Most robust
  ✗ Needs many trees
  ✗ Slower training
  
Very Low (< 0.01):
  ✓ Maximum robustness
  ✗ Needs thousands of trees
  ✗ Very slow
  ✓ Use for critical applications
```

**Interaction with n_estimators**:
```
Rule of thumb: learning_rate × n_estimators ≈ constant

Examples with similar performance:
  lr=0.1, n=500   → Total learning: 50
  lr=0.05, n=1000 → Total learning: 50
  lr=0.01, n=5000 → Total learning: 50

Lower learning rate gives better results but takes longer!
```

#### Max Depth

```
Shallow (1-2):
  ✓ Very fast
  ✓ Strong regularization
  ✗ May underfit complex patterns
  ✓ Good for linear-ish relationships
  
Medium (3-5):
  ✓ Recommended default
  ✓ Captures interactions (3-4 features)
  ✓ Good balance
  ✓ Works for most problems
  
Deep (6-8):
  ✓ Captures complex patterns
  ✗ Slower training
  ✗ Risk of overfitting
  ✓ Use with low learning rate
  
Very Deep (9+):
  ✗ High risk of overfitting
  ✗ Very slow
  ✗ Rarely beneficial
  ✓ Use only with large datasets + heavy regularization
```

**Guideline by dataset size**:
```
Small dataset (< 1000 samples):
  → max_depth = 2-3
  → Focus on regularization

Medium dataset (1K-100K samples):
  → max_depth = 3-5
  → Standard setting

Large dataset (100K-1M+ samples):
  → max_depth = 5-8
  → Can afford complexity
```

#### Subsample Ratio

```
Full (1.0):
  ✓ Uses all data
  ✓ Deterministic
  ✗ May overfit
  
High (0.8-0.9):
  ✓ Slight regularization
  ✓ Still stable
  ✓ Good default
  
Medium (0.5-0.8):
  ✓ Strong regularization
  ✓ More diverse trees
  ✗ Higher variance
  
Low (< 0.5):
  ✗ Too much randomness
  ✗ Unstable
  ✗ Rarely beneficial
```

### Performance Metrics

#### For Regression

**R² Score (Coefficient of Determination)**:
```python
r2 = model.score(X_test, y_test)

Interpretation:
  R² = 1.0:  Perfect predictions
  R² = 0.9:  90% of variance explained (excellent)
  R² = 0.7:  70% of variance explained (good)
  R² = 0.5:  50% of variance explained (acceptable)
  R² < 0.3:  Poor model, need improvement
  R² < 0.0:  Model worse than predicting mean!
```

**Mean Absolute Error (MAE)**:
```python
predictions = model.predict(X_test)
mae = np.mean(np.abs(y_test - predictions))

Interpretation:
  In same units as target variable
  Average prediction error
  Robust to outliers
  
Example: Predicting house prices
  MAE = $25,000 means average error is $25k
```

**Root Mean Squared Error (RMSE)**:
```python
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

Interpretation:
  Penalizes large errors more than MAE
  In same units as target
  
Example: RMSE = $35k vs MAE = $25k
  → Model has some large errors
```

#### For Classification

**Accuracy**:
```python
accuracy = model.score(X_test, y_test)

Interpretation:
  Simple: fraction correct
  Good for balanced classes
  
Caution: Misleading for imbalanced data!
```

**Precision, Recall, F1**:
```python
predictions = (model.predict(X_test) >= 0.5).astype(int)

# Calculate manually
tp = np.sum((predictions == 1) & (y_test == 1))
fp = np.sum((predictions == 1) & (y_test == 0))
fn = np.sum((predictions == 0) & (y_test == 1))

precision = tp / (tp + fp)  # Of predicted positive, how many correct?
recall = tp / (tp + fn)     # Of actual positive, how many found?
f1 = 2 * (precision * recall) / (precision + recall)

Example: Fraud detection
  Precision = 0.80: 80% of flagged transactions are fraud
  Recall = 0.60: 60% of all fraud detected
  F1 = 0.69: Harmonic mean
```

### Detecting Overfitting

**Learning Curves**:
```python
train_scores = model.staged_score(X_train, y_train)
test_scores = model.staged_score(X_test, y_test)

# Plot both curves
import matplotlib.pyplot as plt
plt.plot(train_scores, label='Train')
plt.plot(test_scores, label='Test')
plt.legend()
plt.show()
```

**Patterns to recognize**:
```
Healthy Model:
  Train ────── (high, plateaus)
  Test  ────── (slightly lower, plateaus)
  Gap: Small and stable
  → Good generalization!

Overfitting:
  Train ───────↗ (keeps improving)
  Test  ──────↘ (deteriorates)
  Gap: Growing
  → Stop earlier or increase regularization

Underfitting:
  Train ───↗ (still improving)
  Test  ───↗ (still improving)
  Gap: Small
  → Add more trees or increase max_depth
```

### Feature Importance

```python
importance = model.get_feature_importance()

# Visualize
feature_names = ['feature_0', 'feature_1', ...]
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {imp:.4f}")
```

**Use cases**:
```
1. Feature Selection:
   Drop features with importance < 0.01
   Simplify model, faster training
   
2. Feature Engineering:
   Create interactions of important features
   Example: if 'size' and 'location' important,
            create 'size × location_score'
   
3. Model Interpretation:
   Explain to stakeholders: "Price mainly depends on size and location"
   
4. Domain Validation:
   Check if important features make sense
   Red flag if random features are important!
```

---

## Computational Complexity

### Time Complexity

**Training** (canonical gradient boosting, as implemented by XGBoost/LightGBM/sklearn):
```
O(M × N × F × K × log(N))

where:
  M = number of trees (n_estimators)
  N = number of samples
  F = number of features
  K = max_depth (tree depth)
  log(N) = for sorting features when finding splits
```

**This teaching implementation is a factor of N/log(N) worse:**
```
O(M × N² × F × K)

Why: _create_decision_tree rebuilds a boolean mask and calls np.var twice for
EVERY candidate threshold, and there are O(N) candidates per feature per node.
Optimized libraries pre-sort each feature once and sweep running sums instead.

The slow version is kept deliberately - it puts the variance-reduction formula
literally on screen. The practical consequence is that you must keep the data
small: the USAGE EXAMPLE blocks use 120-400 rows for this reason.

Measured on this machine (Python 3.13, numpy 2.3):
  N=150,  F=1, M=100, K=3   ->   1.2 s
  N=150,  F=6, M=50,  K=4   ->   4.6 s
  N=300,  F=6, M=100, K=4   ->  19.1 s

Doubling N alone roughly quadruples the time, because the split scan is quadratic
in N. For anything larger, use scikit-learn's GradientBoostingRegressor or XGBoost.
```

**Prediction**:
```
O(M × N × K)

where:
  M = number of trees
  N = number of samples
  K = max_depth (tree depth)

Very fast! 
  Typical: 1000 trees, depth 5
  Prediction for 1 sample: < 1ms
  Prediction for 1M samples: < 1 second
```

**Comparison**:
```
Training Time (N samples, F features):
  Gradient Boosting: O(M × N × F × K × log(N))  [sequential, optimized libraries]
  Random Forest: O(M × N × F × log(N))          [parallelizable]
  Neural Network: O(epochs × N × layers × units) [varies greatly]
  
Prediction Time:
  Gradient Boosting: O(M × K)      [fast]
  Random Forest: O(M × K)          [fast]
  Neural Network: O(layers × units) [fast]
```

### Space Complexity

```
O(M × 2^K × F)

where:
  M = number of trees
  2^K = maximum nodes per tree (depth K)
  F = features per split (just index)

Typical storage:
  100 trees, depth 3:
  100 × 2^3 × 8 bytes = ~6 KB (tiny!)
  
Even large models:
  1000 trees, depth 5:
  1000 × 2^5 × 8 bytes = ~250 KB (still small!)

Very memory efficient compared to neural networks!
```

### Parallelization

**Training**:
```
Limited parallelization:
  ✗ Trees must be sequential (each depends on previous)
  ✓ Can parallelize feature search within each tree
  ✓ Can parallelize across data samples (map-reduce style)
  
Libraries like XGBoost and LightGBM parallelize effectively!
```

**Prediction**:
```
Highly parallelizable:
  ✓ Each sample independent
  ✓ Can evaluate on multiple CPUs/GPUs
  ✓ Near-linear speedup with cores
```

---

## Simplifications vs. Canonical Gradient Boosting

This implementation is faithful on the parts that define the algorithm - the
initialization `F₀`, the pseudo-residuals, the variance-reduction split search, the
terminal-region line search and the shrinkage update. On a shared synthetic dataset
it reproduces `sklearn.ensemble.GradientBoostingRegressor`'s predictions to machine
precision (see [Reference check](#reference-check-against-scikit-learn) below).

What it deliberately leaves out, and what that costs you:

| Canonical feature | Here | Consequence |
|---|---|---|
| **Huber / quantile / custom loss** | Only `'mse'`, `'mae'`, `'log_loss'` | No middle ground between squared and absolute error, and no way to plug in your own `L`. The *framework* admits any differentiable loss - `_get_gradient` and `_update_leaf_values` are the only two places that would need a new branch. |
| **Multiclass classification** | Binary only | Canonical multiclass boosting grows K trees per round (one per class) and applies a softmax. Here `loss='log_loss'` expects `y ∈ {0, 1}`. |
| **Early stopping / validation monitoring** | Not present | You must pick `n_estimators` yourself. `staged_score(X_val, y_val)` gives you the curve to pick it from - see [Visualizing Learning Progress](#visualizing-learning-progress). |
| **Warm start / incremental training** | Not present | Every `fit()` call clears `self.trees` and rebuilds from scratch. There is no `partial_fit` and no way to append trees to a trained model. |
| **Missing-value handling** | Not present | `np.nan` in `X` will silently fall to the right of every threshold. Impute before fitting. XGBoost instead learns a default direction per split. |
| **Categorical features** | Not present | Encode as integers and the splits will treat them as ordered, which is usually wrong. CatBoost and LightGBM handle this properly. |
| **L1/L2 leaf regularization, gamma pruning** | Not present | Leaves are unregularized argmins. This is standard Friedman gradient boosting; the regularized objective is what XGBoost adds on top - see `17. XGBoost`. |
| **Pre-sorted O(N log N) split scan** | O(N²) rescan | Training is slow; keep the data under a few hundred rows. See [Computational Complexity](#computational-complexity). |
| **Column subsampling (`colsample_bytree`)** | Row subsampling only | Every tree sees every feature. `subsample < 1.0` gives you Friedman's stochastic gradient boosting, but not Random-Forest-style feature decorrelation. |
| **`min_samples_leaf`, `min_impurity_decrease`** | Only `min_samples_split` | Slightly less control over leaf size; a split is accepted whenever `gain > 0`. |

### Reference check against scikit-learn

Every test below uses `n_estimators=100`, `learning_rate=0.1`, `max_depth=3` on
seeded synthetic data shared with the scikit-learn model. Each names its own dataset.

```
loss='mse'       y = x^2 + noise, 200 rows shuffled, 150/50 split
                 vs GradientBoostingRegressor(criterion='squared_error')
                 max |prediction difference| = 8.9e-16 on train
                 (identical tree structures, identical thresholds, identical leaves)
                 On the test set it is 0.94, not 8.9e-16: 10 of the 50 test
                 x-values sit within 1e-7 of a threshold sklearn rounded to
                 float32 and fall the other way. Snap X to the float32 grid
                 (X.astype(np.float32).astype(float)) and both train and test
                 agree to 8.9e-16 - see the caveat below.

loss='log_loss'  two Gaussian blobs at (-2,-2) and (2,2), 200 rows, 150/50 split
                 vs GradientBoostingClassifier
                 train log loss  0.000022 both sides
                 mean |P_ours - P_sklearn| on the test set = 0.000000

loss='mae'       np.random.seed(42)
                 X = np.random.uniform(-5, 5, (200, 1))     # on the float32 grid
                 y = 100 * X.ravel() + np.random.randn(200) * 5
                 fitted and scored on all 200 rows
                 (a wide y-range is what exposes a broken leaf update: without
                  the line search an 'mae' leaf holds mean(sign(r)) in [-1, 1],
                  so 100 trees at lr 0.1 could move a prediction by at most 10)
                 vs GradientBoostingRegressor(loss='absolute_error')
                 R2   0.999684 both sides
                 MAE  3.6496   both sides
                 max |prediction difference| = 0.0e+00

loss='mae'       same x^2 / 150-50 dataset as the first test
                 test R2   0.9551 (ours) vs 0.9603 (sklearn)
                 train MAE 0.2747 (ours) vs 0.2496 (sklearn)
                 - within 0.006 R2, and the gap splits into two measured causes:
                 - snapping X to the float32 grid lifts ours to 0.9595 while
                   sklearn stays at 0.9603, so threshold rounding (the caveat
                   below) accounts for ~0.0044 of the 0.0052 test-R2 gap
                 - the remaining ~0.0008, and the whole train-MAE gap, starts at
                   round 26 (0-indexed, i.e. the 27th tree): thresholds -2.7889 and
                   2.8342 tie there at exactly equal gain (5929/900), but np.var
                   evaluates them 2.8e-14 apart, so the two libraries take
                   different - equally optimal - splits and never re-converge
```

The one caveat: scikit-learn stores `X` internally as `float32`, so its split
thresholds are rounded to `float32`. Feed it data that is already on the `float32`
grid and the `'mse'` run above agrees exactly; otherwise a handful of test points
sitting within `~1e-7` of a threshold can land on opposite sides. Snapping the grid
does not close the fourth row's `'mae'` gap, which has the separate cause listed
with that row.

---

## Advantages and Limitations

### Advantages ✅

1. **High Predictive Accuracy**
   - Often wins Kaggle competitions
   - State-of-the-art for structured/tabular data
   - Typically outperforms most other algorithms

2. **Handles Complex Patterns**
   - Non-linear relationships
   - Feature interactions automatically captured
   - No need for manual feature engineering

3. **Flexible Loss Functions**
   - The *framework* can optimize any differentiable loss - you only need its
     gradient and the argmin of the loss inside a leaf
   - Works for regression, classification, ranking
   - *This implementation ships three:* `'mse'`, `'mae'`, `'log_loss'`. There is no
     hook for a user-supplied loss - see
     [Simplifications](#simplifications-vs-canonical-gradient-boosting)

4. **Robust to Outliers** (with appropriate loss)
   - `loss='mae'` is robust, and demonstrably so: in the
     [Quick Start](#quick-start-plug-and-play-example), eight corrupted training
     labels drop the `'mse'` model to Test R2 = -20.43 while `'mae'` holds 0.96
   - Doesn't require outlier removal
   - *Huber loss (a squared-error/absolute-error hybrid) is the usual third option
     in production libraries but is not implemented here*

5. **Handles Mixed Data Types**
   - Numerical features: direct use
   - Categorical features: works well (encode as integers)
   - Missing values: can be handled with modifications

6. **Feature Importance**
   - Built-in importance scores
   - Helps model interpretation
   - Useful for feature selection

7. **Incremental Learning** (in production libraries)
   - XGBoost, LightGBM and scikit-learn's `warm_start=True` let you append trees
     to an already-trained model and continue later
   - *Not available here:* every call to `fit()` starts with `self.trees = []` and
     rebuilds the ensemble from scratch. Calling `fit()` twice with
     `n_estimators=5` leaves you with 5 trees, not 10

### Limitations ❌

1. **Sequential Training (Slower)**
   ```
   Cannot train trees in parallel
   Each tree depends on previous trees
   
   Training time:
     Random Forest (parallel): 10 seconds
     Gradient Boosting (sequential): 60 seconds
     
   For very large datasets, this matters!
   
   Solutions:
     - Use XGBoost, LightGBM (optimized implementations)
     - Use stochastic gradient boosting (subsample)
     - Use GPUs (with supporting libraries)
   ```

2. **Hyperparameter Sensitive**
   ```
   Many parameters to tune:
     - n_estimators
     - learning_rate
     - max_depth
     - min_samples_split
     - subsample
     
   Poor settings → poor performance
   
   Typical tuning time:
     Grid search: Hours to days
     Random search: Hours
     
   Solution:
     - Start with good defaults
     - Use learning curves
     - Bayesian optimization
   ```

3. **Prone to Overfitting**
   ```
   With deep trees and many estimators:
     Training accuracy → 100%
     Test accuracy → poor
     
   Example:
     max_depth=10, n_estimators=1000
     → Train R²=0.99, Test R²=0.70 (overfitting!)
     
   Solutions:
     - Reduce max_depth (3-5)
     - Lower learning_rate (0.01-0.1)
     - Use subsample (0.8)
     - Early stopping
   ```

4. **Memory Intensive for Deep Trees**
   ```
   Each tree stores split information
   
   Deep trees (depth 10+):
     2^10 = 1024 nodes per tree
     1000 trees × 1024 nodes = ~1M nodes
     
   For very large models:
     Can use several GB of RAM
     
   Solution:
     - Keep trees shallow (depth 3-5)
     - Use leaf-wise growth (LightGBM)
   ```

5. **Extrapolation Problems**
   ```
   Trees can only predict within training range
   
   Example:
     Training: prices $100k - $500k
     Prediction for $1M house: capped at ~$500k
     
   Cannot extrapolate beyond training data!
   
   Solution:
     - Ensure training data covers prediction range
     - Use linear model for extrapolation
     - Add features to indicate out-of-range
   ```

6. **Less Effective on Very High-Dimensional Sparse Data**
   ```
   Text data with 10,000+ features (mostly zeros):
     Trees struggle to find good splits
     Many features never used
     
   Better algorithms for this:
     - Linear models (Logistic Regression, SVM)
     - Neural networks
     
   Gradient Boosting shines on:
     Dense, structured, tabular data (<1000 features)
   ```

### When to Use Gradient Boosting

**Good Use Cases**:
- ✅ Structured/tabular data (most common)
- ✅ Medium to large datasets (1K-1M+ samples)
- ✅ Regression or classification
- ✅ Need high accuracy
- ✅ Feature importance required
- ✅ Competitions (Kaggle)
- ✅ Moderate number of features (<1000)

**Bad Use Cases**:
- ❌ Very high-dimensional sparse data → Use Linear Models
- ❌ Images, audio, video → Use Neural Networks (CNNs, RNNs)
- ❌ Natural language processing → Use Transformers
- ❌ Need very fast training → Use Random Forest
- ❌ Real-time training required → Use online learning algorithms
- ❌ Extrapolation critical → Use parametric models

---

## Comparing with Alternatives

### Gradient Boosting vs. AdaBoost

```
Gradient Boosting:
  ✓ More general (any loss function)
  ✓ Better performance typically
  ✓ Can do regression and classification
  ✓ More flexible
  ✗ More hyperparameters
  ✗ Slightly more complex
  
AdaBoost:
  ✓ Simpler conceptually
  ✓ Fewer hyperparameters
  ✓ Good for binary classification
  ✗ Only exponential loss
  ✗ Sensitive to outliers
  ✗ Less flexible

When to choose:
  Gradient Boosting: Almost always (more powerful)
  AdaBoost: Educational purposes, very simple problems
```

### Gradient Boosting vs. Random Forest

```
Gradient Boosting:
  ✓ Usually higher accuracy
  ✓ Better feature importance
  ✓ More interpretable (fewer, shallower trees)
  ✗ Slower training (sequential)
  ✗ More prone to overfitting
  ✗ More hyperparameters
  
Random Forest:
  ✓ Faster training (parallel)
  ✓ More robust (harder to overfit)
  ✓ Fewer hyperparameters
  ✗ Lower accuracy typically
  ✗ Needs more trees for same performance
  ✗ Larger model size

When to choose:
  Gradient Boosting: Need best accuracy, have time to tune
  Random Forest: Need speed, robustness, less tuning
```

### Gradient Boosting vs. XGBoost/LightGBM/CatBoost

```
Our Gradient Boosting (Educational):
  ✓ Easy to understand
  ✓ Simple implementation
  ✗ Slower (Python, no optimization)
  ✗ Limited features
  ✗ Not production-ready
  
XGBoost/LightGBM/CatBoost (Production):
  ✓ 10-100x faster (C++, optimized)
  ✓ Built-in regularization
  ✓ Handles missing values
  ✓ Categorical feature support
  ✓ GPU acceleration
  ✓ Early stopping, CV built-in
  ✗ More complex to understand
  
When to choose:
  Our implementation: Learning, understanding internals
  XGBoost/LightGBM: Production, competitions, real work
```

**Feature Comparison**:
```
Feature                    | Ours | XGBoost | LightGBM | CatBoost
---------------------------|------|---------|----------|----------
Speed                      | ⭐    | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐
Regularization             | ❌    | ✅       | ✅       | ✅
Missing value handling     | ❌    | ✅       | ✅       | ✅
Categorical features       | ❌    | ❌       | ✅       | ✅
GPU support                | ❌    | ✅       | ✅       | ✅
Ease of understanding      | ⭐⭐⭐⭐⭐| ⭐⭐⭐   | ⭐⭐     | ⭐⭐
```

### Gradient Boosting vs. Neural Networks

```
Gradient Boosting:
  ✓ Better for tabular data
  ✓ Less data needed
  ✓ Faster training
  ✓ Built-in feature importance
  ✓ No data preprocessing needed
  ✗ Cannot handle images, text directly
  ✗ No transfer learning
  
Neural Networks:
  ✓ Better for images, audio, text
  ✓ Transfer learning available
  ✓ Can learn representations
  ✗ Needs more data
  ✗ Slower training
  ✗ Needs preprocessing/normalization
  ✗ Less interpretable

When to choose:
  Gradient Boosting: Structured/tabular data
  Neural Networks: Images, text, audio, video
```

---

## Key Concepts to Remember

### 1. **Gradient Descent in Function Space**
Not optimizing parameters, but optimizing predictions by adding functions (trees)

```
Traditional: θ ← θ - η∇L(θ)
Gradient Boosting: F(x) ← F(x) + η·h(x)
where h(x) fits -∇L(F(x))
```

### 2. **Sequential Error Correction**
Each tree corrects mistakes of previous trees

```
Tree 1: Learns main patterns
Tree 2: Corrects Tree 1's errors
Tree 3: Corrects remaining errors
...
Tree M: Final refinements

Together: Highly accurate!
```

### 3. **Bias-Variance Trade-off**
```
Shallow trees + many estimators: Low bias, low variance (optimal!)
Deep trees + few estimators: High variance, low bias (overfitting)
Shallow trees + few estimators: High bias, low variance (underfitting)

Sweet spot: depth=3-5, n_estimators=100-500
```

### 4. **Learning Rate is Crucial**
```
High learning rate (0.3+):
  - Fast convergence
  - May overfit
  - Can overshoot

Low learning rate (0.01-0.1):
  - Slow convergence
  - Better generalization
  - More robust

Rule: learning_rate × n_estimators ≈ constant
```

### 5. **Shrinkage (Learning Rate) Prevents Overfitting**
```
Without shrinkage:
  Early trees dominate
  Later trees overfit

With shrinkage:
  All trees contribute equally
  Smoother learning
  Better generalization
```

### 6. **Trees Should Be Shallow**
```
Depth 3-5 is usually optimal:
  - Captures 2-4 way interactions
  - Fast to train
  - Regularization effect
  - Many trees together → complexity

Depth 10+:
  - Individual trees too complex
  - Ensemble overfits
  - Slower
  - Rarely beneficial
```

### 7. **Subsampling Helps**
```
Stochastic Gradient Boosting (subsample < 1.0):
  - Each tree sees different data
  - More diverse ensemble
  - Reduces overfitting
  - Faster training

Typical: subsample = 0.8
```

---

## Conclusion

Gradient Boosting is one of the most powerful machine learning algorithms, especially for structured/tabular data! By understanding:
- How sequential error correction works
- How gradient descent in function space optimizes predictions
- How to choose n_estimators, learning_rate, and max_depth
- When Gradient Boosting excels and when to use alternatives
- The importance of regularization and early stopping

You've gained deep insight into the algorithm that powers many winning Kaggle solutions and production systems! 🚀

**When to Use Gradient Boosting**:
- ✅ Structured/tabular data (CSV files, databases)
- ✅ Need high accuracy (competitions, critical applications)
- ✅ Regression or classification tasks
- ✅ Feature importance required
- ✅ Have time for hyperparameter tuning
- ✅ Medium to large datasets

**When to Use Something Else**:
- ❌ Images, audio, video → Use Neural Networks (CNNs)
- ❌ Natural language → Use Transformers (BERT, GPT)
- ❌ Very high-dimensional sparse data → Use Linear Models
- ❌ Need very fast training → Use Random Forest
- ❌ Simple problem, limited data → Use Logistic Regression
- ❌ Real-time online learning → Use online algorithms

**Next Steps**:
- Try Gradient Boosting on your own datasets
- Compare with Random Forest to see the difference
- Experiment with learning_rate and n_estimators
- Learn about XGBoost, LightGBM for production use
- Study advanced techniques (custom loss functions, early stopping)
- Explore CatBoost for categorical features
- Practice on Kaggle competitions!

**For Production Use**:
Always use optimized libraries:
- **XGBoost**: Most popular, good all-around
- **LightGBM**: Fastest, best for large datasets
- **CatBoost**: Best for categorical features

Happy Boosting! 💻🚀📊

