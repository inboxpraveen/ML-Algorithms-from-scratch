# CatBoost from Scratch: A Comprehensive Guide

Welcome to CatBoost! 🚀 In this comprehensive guide, we'll explore CatBoost (Categorical Boosting) - a powerful gradient boosting framework developed by Yandex that excels at handling categorical features and uses symmetric trees for better generalization. Think of it as the "smart handler" of gradient boosting!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is CatBoost?](#what-is-catboost)
3. [How CatBoost Works](#how-catboost-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [CatBoost vs XGBoost vs LightGBM](#catboost-vs-xgboost-vs-lightgbm)
11. [Advantages & Limitations](#advantages--limitations)
12. [Summary](#summary)
13. [References and Further Learning](#references-and-further-learning)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy. (Running `python _19_catboost.py` directly executes the same three demos from its `__main__` block.)

```python
# ---------------------------------------------------------------
# CatBoost from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _19_catboost.py  (the __main__ block runs this)
# Or copy the CatBoost class from _19_catboost.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the CatBoost class here (from _19_catboost.py) ----
# class CatBoost: ...

np.random.seed(42)

# ------ REGRESSION: predict y = x^2 + noise ------
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Shuffle before splitting: trees cannot extrapolate beyond the training range.
# Without shuffling the last 50 x-values would all be > training max.
idx = np.random.permutation(200)
X, y = X[idx], y[idx]

X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

model = CatBoost(
    n_estimators=100,
    learning_rate=0.1,
    depth=4,             # 2^4 = 16 leaves per symmetric tree
    l2_leaf_reg=3.0,     # L2 on leaf values (CatBoost's default)
    random_seed=42
)
model.fit(X_train, y_train)

# score() returns NEGATIVE RMSE for regression, so negate it to read an error
print(f"Train RMSE: {-model.score(X_train, y_train):.4f}")
print(f"Test  RMSE: {-model.score(X_test,  y_test):.4f}")

preds = model.predict(X_test)
for i in range(3):
    print(f"  x={X_test[i,0]:5.2f}  true={y_test[i]:5.2f}  pred={preds[i]:5.2f}")

# ------ CLASSIFICATION: two Gaussian blobs ------
X0 = np.random.randn(100, 2) + np.array([-2, -2])
X1 = np.random.randn(100, 2) + np.array([ 2,  2])
X_c = np.vstack([X0, X1])
y_c = np.array([0]*100 + [1]*100)
idx = np.random.permutation(200)
X_c, y_c = X_c[idx], y_c[idx]

cls = CatBoost(n_estimators=50, learning_rate=0.3, depth=3,
               objective='binary', random_seed=42)
cls.fit(X_c[:150], y_c[:150])

print(f"\nClassification accuracy: {cls.score(X_c[150:], y_c[150:]):.2%}")
proba = cls.predict_proba(X_c[150:])
for i in range(3):
    print(f"  true={y_c[150+i]}  P(0)={proba[i,0]:.3f}  P(1)={proba[i,1]:.3f}")

# ------ CATEGORICAL: a raw string column, no one-hot encoding ------
plans = np.array(['basic', 'plus', 'pro', 'enterprise'])
value = {'basic': 10.0, 'plus': 25.0, 'pro': 60.0, 'enterprise': 150.0}

plan_col = np.random.choice(plans, 400)
usage_col = np.random.uniform(0, 10, 400)
revenue = (np.array([value[p] for p in plan_col])
           + 3.0 * usage_col + np.random.randn(400) * 5.0)

X_cat = np.empty((400, 2), dtype=object)
X_cat[:, 0] = plan_col     # strings, straight into the model
X_cat[:, 1] = usage_col
idx = np.random.permutation(400)
X_cat, revenue = X_cat[idx], revenue[idx]

cat_model = CatBoost(n_estimators=120, learning_rate=0.1, depth=4,
                     cat_features=[0], random_seed=42)
cat_model.fit(X_cat[:300], revenue[:300])

print(f"\nCategorical test RMSE: {-cat_model.score(X_cat[300:], revenue[300:]):.4f}"
      f"  (std of target = {np.std(revenue[300:]):.2f})")
for plan in plans:
    print(f"  {plan:11s} -> ordered target statistic "
          f"{cat_model._cat_encodings[0]['mapping'][plan]:7.2f}")
```

Expected output:
```
Train RMSE: 0.4129
Test  RMSE: 0.4494
  x=-2.88  true= 8.17  pred= 8.36
  x= 0.23  true= 0.14  pred= 0.16
  x= 2.55  true= 6.38  pred= 6.70

Classification accuracy: 100.00%
  true=1  P(0)=0.006  P(1)=0.994
  true=0  P(0)=0.974  P(1)=0.026
  true=1  P(0)=0.003  P(1)=0.997

Categorical test RMSE: 4.9401  (std of target = 58.92)
  basic       -> ordered target statistic   25.67
  plus        -> ordered target statistic   40.06
  pro         -> ordered target statistic   76.33
  enterprise  -> ordered target statistic  164.18
```

Notice the third block: the `plan` column holds **strings** and is handed to the model as-is. `cat_features=[0]` tells CatBoost to encode it with ordered target statistics, and the recovered statistics (25.67 / 40.06 / 76.33 / 164.18) track the true plan values (10 / 25 / 60 / 150 plus the average usage effect) without any one-hot encoding.

---

## What is CatBoost?

CatBoost (Categorical Boosting) is a **gradient boosting framework developed by Yandex** that handles categorical features naturally and uses symmetric (oblivious) trees. It addresses critical issues like prediction shift through ordered boosting, making it highly robust and accurate.

**Real-world analogy**: 
If XGBoost is a meticulous craftsman and LightGBM is a speed demon, CatBoost is like a wise architect who:
- Builds symmetric, balanced structures (oblivious trees)
- Prevents contamination (ordered boosting avoids target leakage)
- Handles different materials naturally (categorical features)
- Focuses on stability and reliability

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Ensemble Learning (Advanced Boosting) |
| **Learning Style** | Supervised Learning |
| **Primary Use** | Regression, Classification, Ranking |
| **Base Learners** | Symmetric (oblivious) decision trees |
| **Key Innovation** | Ordered boosting + Symmetric trees + Categorical handling |

### The Core Idea

```
"CatBoost = Gradient Boosting + Symmetric Trees + Ordered Boosting + Categorical Intelligence"
```

CatBoost improves upon XGBoost and LightGBM through:
- **Symmetric trees**: All nodes at same level use same split
- **Ordered boosting**: Prevents prediction shift and target leakage
- **Ordered target statistics**: Smart encoding for categorical features
- **Robust defaults**: Works well out-of-the-box
- **Handles categoricals natively**: No need for one-hot encoding

> **How this maps onto `_19_catboost.py`:** symmetric trees, the quantizer and
> the L2-regularized leaf/split formulas are the core of the file and are always
> active. Ordered boosting is available as `boosting_type='Ordered'` (the
> default is `'Plain'`, i.e. classic gradient boosting), and ordered target
> statistics run for the columns you list in `cat_features=[...]`. What is *not*
> reproduced is spelled out in
> [Advantages & Limitations](#advantages--limitations).

### Key Differences from XGBoost and LightGBM

**1. Tree Structure: Symmetric vs Asymmetric**
```
XGBoost (Level-wise):          LightGBM (Leaf-wise):         CatBoost (Symmetric):
      Root                            Root                           Root
     /    \                          /    \                         /    \
    A      B                        A      B                       A      B
   / \    / \                      / \                        [Feature 2]   [Feature 2]
  C   D  E  F                     C   D                          / \           / \
                                     / \                        C   D         E   F
                                    E   F
                                    
Balanced tree              Asymmetric tree                Symmetric tree
All level splits           Best leaf split                Same split at level
different                  different                      SAME for both!
```

**Key difference**: In CatBoost, both A and B split on the SAME feature with SAME threshold!

**2. Symmetric (Oblivious) Trees**
```
Traditional Trees:
- Each node can split on any feature
- Flexible but complex
- Hard to regularize

CatBoost Symmetric Trees:
- All nodes at level L split on same feature & threshold
- Simpler structure
- Natural regularization
- Branch-free prediction: the same O(depth) comparisons any binary tree needs,
  but the whole batch shares ONE comparison per level (see below)

Example with depth=3:
Level 0: ALL split on "Age <= 30"
Level 1: ALL split on "Income <= 50K"  
Level 2: ALL split on "Score <= 700"
Result: 2^3 = 8 leaves

Prediction path: Just check 3 conditions → get leaf index!
```

**3. Ordered Boosting**
```
Problem with traditional boosting:
- When fitting tree T, use predictions from trees 1..T-1
- But trees 1..T-1 were trained on the SAME data
- This causes PREDICTION SHIFT and TARGET LEAKAGE
- Model sees labels during gradient calculation

CatBoost's Solution: Ordered Boosting
- Divide data into random permutations
- For each sample, use predictions from models trained on OTHER samples
- Prevents target leakage
- More robust, less overfitting

Simplified example:
Training samples: [1, 2, 3, 4, 5]
- For sample 3's gradient: use model trained only on {1, 2}
- For sample 5's gradient: use model trained only on {1, 2, 3, 4}
- Never use same sample for both training and gradient calculation!
```

**4. Categorical Feature Handling**
```
XGBoost/LightGBM:
- Need to encode categoricals manually
- One-hot encoding (explodes features)
- Label encoding (loses information)
- Target encoding (risk of target leakage)

CatBoost:
- Handles categoricals NATIVELY
- Uses "Ordered Target Statistics"
- Computes target mean for each category
- But in special order to prevent leakage
- Automatically optimal encoding

Example: Color = ["Red", "Blue", "Green"]
CatBoost internally: Red→0.65, Blue→0.42, Green→0.78
(based on target statistics, not arbitrary numbers!)

In this implementation that is a real code path, not just theory:

    model = CatBoost(cat_features=[0])     # column 0 holds the strings
    model.fit(X, y)                        # X[:, 0] can be "Red"/"Blue"/...
    model._cat_encodings[0]['mapping']     # -> the learned statistic per level

See the Quick Start above for a runnable version.
```

**5. Default Learning Rate**
```
XGBoost: 0.3 (aggressive)
LightGBM: 0.1 (moderate)
CatBoost: 0.03 (conservative)

Why CatBoost uses lower rate?
- Symmetric trees are simpler
- Ordered boosting adds complexity
- Lower rate + more trees = better generalization
```

---

## How CatBoost Works

### The Algorithm in 7 Steps

```
Step 1: Quantize numerical features into discrete bins
         - Similar to LightGBM's histogram
         - Typical: 128 borders per feature
         ↓
Step 2: Handle categorical features (if any)
         - Convert to ordered target statistics
         - Prevents target leakage through ordering
         ↓
Step 3: Initialize predictions (base_score)
         ↓
Step 4: For each boosting iteration:
         a. Calculate gradients
         b. Apply ordered boosting (boosting_type='Ordered'; default 'Plain')
         ↓
Step 5: Build SYMMETRIC tree:
         - For each level (depth):
           * Try all features and thresholds
           * Pick split that ALL nodes at this level will use
           * Split ALL current partitions with this split
         ↓
Step 6: Calculate leaf values with L2 regularization:
         value = -sum(gradients) / (sum(hessians) + l2_leaf_reg)
         (squared loss has h = 1, so sum(hessians) IS the sample count;
          log loss has h = p(1-p), which is much smaller)
         ↓
Step 7: Update predictions: F(x) = F(x) + η × tree(x)
         ↓
Repeat Steps 4-7 for n_estimators
```

### Visual Example: Binary Classification with CatBoost

Let's predict loan default using symmetric trees:

```
Data:
Customer | Income | Debt | Existing_Loans | Default?
---------|--------|------|----------------|----------
   A     |   50   |  20  |       1        |    0
   B     |   80   |  15  |       0        |    0
   C     |   40   |  35  |       2        |    1
   D     |   90   |  10  |       0        |    0
   E     |   35   |  40  |       3        |    1
   F     |   70   |  25  |       1        |    0
   G     |   45   |  38  |       2        |    1
   H     |   95   |   8  |       0        |    0
```

**Step 1: Quantize Features**

```
Income bins (border_count=2):
  Bin 0: Income ≤ 60 → [50, 40, 35, 45]
  Bin 1: Income > 60 → [80, 90, 70, 95]

Debt bins (border_count=2):
  Bin 0: Debt ≤ 22.5 → [20, 15, 10, 8]
  Bin 1: Debt > 22.5 → [35, 40, 25, 38]

Existing_Loans bins (border_count=2):
  Bin 0: Loans ≤ 1.5 → [1, 0, 0, 1, 0]
  Bin 1: Loans > 1.5 → [2, 3, 2]
```

**Step 2: Initialize**

```
Default rate: p = 3/8 = 0.375
base_score = log(0.375 / 0.625) = log(0.6) = -0.51

Initial predictions (log-odds): [-0.51] × 8
Initial probabilities: sigmoid(-0.51) = 0.375 for all
```

**Step 3: Calculate Gradients**

```
For binary log loss:
g = p - y

Customer | y | p=0.375 | gradient
---------|---|---------|----------
   A     | 0 |  0.375  |  0.375
   B     | 0 |  0.375  |  0.375
   C     | 1 |  0.375  | -0.625
   D     | 0 |  0.375  |  0.375
   E     | 1 |  0.375  | -0.625
   F     | 0 |  0.375  |  0.375
   G     | 1 |  0.375  | -0.625
   H     | 0 |  0.375  |  0.375

Gradient array: [0.375, 0.375, -0.625, 0.375, -0.625, 0.375, -0.625, 0.375]

For log loss the Hessian is h = p(1-p), so at this first iteration every
customer has the same one:

h = 0.375 × 0.625 = 0.234375   (all eight)

From here on H means the SUM of these h over a group, never the sample
count - see "2. Gradient and Hessian Calculation" below. A 4-sample group
has H = 0.9375, not 4.
```

**Step 4: Build Symmetric Tree (Depth=2)**

```
LEVEL 0: Choose split for ALL root partitions

Try Income <= Bin 0 (Income ≤ 60):
  Left: [A, C, E, G] gradients = [0.375, -0.625, -0.625, -0.625]
  Right: [B, D, F, H] gradients = [0.375, 0.375, 0.375, 0.375]
  
  Calculate gain (with l2_leaf_reg=3), using Score = G² / (H + λ):
    Left:  G_L = -1.5, H_L = 4 × 0.234375 = 0.9375
           Score_L = (-1.5)² / (0.9375 + 3) = 2.25 / 3.9375 = 0.5714
    Right: G_R = 1.5, H_R = 4 × 0.234375 = 0.9375
           Score_R = (1.5)² / (0.9375 + 3) = 2.25 / 3.9375 = 0.5714
    Parent: G_P = 0, H_P = 8 × 0.234375 = 1.875
           Score_P = 0² / (1.875 + 3) = 0
    
    Gain = 0.5714 + 0.5714 - 0 = 1.1429  ← Best split!

Decision: Level 0 splits on "Income <= 60"

Current state:
├─ Partition 0 (Low income): [A, C, E, G]
└─ Partition 1 (High income): [B, D, F, H]
```

```
LEVEL 1: Choose ONE split for BOTH partitions

Try Debt <= Bin 0 (Debt ≤ 22.5):

For Partition 0 (Low income):
  Left (Low income, Low debt): [A] gradients = [0.375]
  Right (Low income, High debt): [C, E, G] gradients = [-0.625, -0.625, -0.625]

For Partition 1 (High income):
  Left (High income, Low debt): [B, D, H] gradients = [0.375, 0.375, 0.375]
  Right (High income, High debt): [F] gradients = [0.375]

Calculate total gain across BOTH partitions, using Score = G² / (H + λ).
Every h is 0.234375 here, so a 1-sample group has H = 0.234375, a 3-sample
group H = 0.703125 and a 4-sample group H = 0.9375:

  Partition 0: Score_L = 0.375²/(0.234375+3)    = 0.04348
               Score_R = (-1.875)²/(0.703125+3) = 0.94937
               Score_P = (-1.5)²/(0.937500+3)   = 0.57143
               Gain_partition0 = 0.04348 + 0.94937 - 0.57143 = +0.42142

  Partition 1: Score_L = 1.125²/(0.703125+3)    = 0.34177
               Score_R = 0.375²/(0.234375+3)    = 0.04348
               Score_P = 1.5²/(0.937500+3)      = 0.57143
               Gain_partition1 = 0.34177 + 0.04348 - 0.57143 = -0.18618

  Gain_partition0 + Gain_partition1 = 0.421 + (-0.186) = 0.235

Note that partition 1's gain is NEGATIVE. Its four samples (B, D, H, F) all
carry the same gradient +0.375, so that partition was already pure - splitting
it can only shed score. The split still wins overall because partition 0 gains
far more than partition 1 loses. That trade-off is exactly what "one split for
the whole level" costs, and it is the price symmetric trees pay for the extra
regularization they buy.

This is the best split across all features!

Decision: Level 1 splits on "Debt <= 22.5"

Final tree structure (Symmetric!):
                    [Income <= 60]
                   /              \
          [Debt <= 22.5]      [Debt <= 22.5]
            /        \          /        \
         Leaf0     Leaf1     Leaf2     Leaf3
          [A]    [C,E,G]   [B,D,H]      [F]
```

**Step 5: Calculate Leaf Values**

```
w* = -G / (H + λ)

Leaf 0 (Low income, Low debt): [A]
  G = 0.375, H = 0.234375
  value = -0.375 / (0.234375 + 3) = -0.116

Leaf 1 (Low income, High debt): [C, E, G]
  G = -1.875, H = 0.703125
  value = -(-1.875) / (0.703125 + 3) = 0.506

Leaf 2 (High income, Low debt): [B, D, H]
  G = 1.125, H = 0.703125
  value = -1.125 / (0.703125 + 3) = -0.304

Leaf 3 (High income, High debt): [F]
  G = 0.375, H = 0.234375
  value = -0.375 / (0.234375 + 3) = -0.116

Notice how L2 regularization (3.0) shrinks values toward zero!

Every number in this walkthrough is what `_build_symmetric_tree` actually
returns for these eight rows when it is run with `random_strength=0` - it
picks Income at level 0 (gain 1.1429) and Debt at level 1 (gain 0.2352),
with exactly these four leaf values. That setting matters here: Income is
not the unique winner at level 0, because "Debt <= 22.5" scores exactly the
same 1.1428571, and with the jitter off it is `np.argmax` keeping the first
candidate that settles the tie. At the default `random_strength=1.0` the tie
can go either way (12 Income / 17 Debt / 1 Existing_Loans over random_seed
0-29), so keep the jitter off to reproduce the tree printed here. Had we
divided by the sample count instead of by H, leaf 1 would read 0.313 rather
than 0.506. The count (3) is over four times H (0.703); the +3 from
l2_leaf_reg softens that, but the step still comes out 1.6x too small - on
every leaf, on every iteration.
```

**Step 6: Update Predictions**

```
Learning rate η = 0.05

Customer A: -0.51 + 0.05 × (-0.116) = -0.516
Customer C: -0.51 + 0.05 × 0.506 = -0.485
Customer B: -0.51 + 0.05 × (-0.304) = -0.525
...

After 100 trees (measured by actually running CatBoost(n_estimators=100,
learning_rate=0.05, depth=2, l2_leaf_reg=3.0, border_count=2,
random_strength=0, objective='binary') on these eight rows):
High-risk customers (C, E, G) → positive log-odds → p = 0.754
Low-risk customers (A, F)     → negative log-odds → p = 0.268
Low-risk customers (B, D, H)  → negative log-odds → p = 0.171
Every customer ends on the correct side of 0.5. B, D and H all land in leaf 2,
so the model can only ever give the three of them one common probability.
A and F sit in different leaves (0 and 3), but each is alone in its leaf with
the same gradient, so those two leaves hold identical values in all 100 trees
and the customers end up indistinguishable anyway - four leaves cannot say
more than four things.
```

**Why Symmetric Trees Help:**

```
Advantages:
1. Regularization: Simpler structure prevents overfitting
2. Fast prediction: Just check depth conditions
3. Interpretability: Easy to understand decision path
4. Robustness: Less sensitive to noise

Prediction for new customer:
- Income = 55 → Goes left (≤ 60)
- Debt = 30 → Goes right (> 22.5)
- Leaf index: 01 (binary) = 1 → Leaf 1
- Prediction: Add leaf 1 value from each tree!

Traditional tree: O(depth) comparisons, but each sample follows its OWN path
Symmetric tree:   O(depth) comparisons too - the asymptotics are IDENTICAL.

The real win is different: every sample at level L is tested against the SAME
(feature, threshold), so a level is one vectorised array comparison for the
entire batch, the leaf index is assembled by bit arithmetic, and the answer is
a single fancy-index lookup into leaf_values. No per-sample branching, no
pointer chasing. See `goes_right * (2 ** remaining_depth)` in `_leaf_indices`.
```

---

## The Mathematical Foundation

### 1. Objective Function

CatBoost optimizes a regularized objective similar to XGBoost:

```
Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₜ)

Where:
- L(yᵢ, ŷᵢ) = loss function (RMSE for regression, logloss for classification)
- Ω(fₜ) = regularization for tree t
- Ω(f) = γT + λΣ(w²ᵢ)
  - γ: penalty for number of leaves (implicit through depth)
  - λ: L2 regularization on leaf weights (l2_leaf_reg)
  - T: number of leaves = 2^depth
```

### 2. Gradient and Hessian Calculation

Every leaf value and every split score is built from two per-sample
quantities: the gradient g and the Hessian h.

```
g = ∂L/∂ŷ          h = ∂²L/∂ŷ²

For squared loss (L2):
L = ½(y - ŷ)²
g = ŷ - y
h = 1                       <- constant!

For log loss (binary classification):
L = -[y·log(p) + (1-y)·log(1-p)]
where p = sigmoid(ŷ) = 1/(1 + e^(-ŷ))
g = p - y
h = p(1 - p)                <- at most 0.25

Why the Hessian matters:
- Leaf value and split score both divide by (Σh + λ), never by the count.
- For squared loss Σh IS the sample count, so the two are the same number.
  That is why every count-based formula quoted in this guide is correct for
  regression - and it is CatBoost's 'Gradient' leaf estimation method.
- For log loss Σh = Σp(1-p) ≤ 0.25·N, so the count is 4x too large or more.
  Using it would shrink every classification step by that factor and the
  model could never become confident. Real CatBoost defaults to 'Newton'
  leaf estimation for Logloss for exactly this reason, and so does
  `_compute_hessians` in `_19_catboost.py`.

Measured on two overlapping unit-variance Gaussian blobs - the __main__
demo's blobs with centres at (-1, -1) and (+1, +1) instead of (-2, -2) and
(+2, +2): on a fresh np.random.seed(42), draw 100 class-0 points, then 100
class-1 points, then permutation(200), then split 150 train / 50 test. With
40 trees, depth 4, learning_rate 0.1 and random_strength 0 (which makes the
fit deterministic - no model seed needed), switching the denominator from
the count to Σp(1-p) moved test logloss from 0.3621 to 0.2059 and widened
the predicted-probability range from [0.227, 0.779] to [0.040, 0.960]. Both
versions get the same 90% test accuracy - only the confidence changes.
sklearn 1.7.2's GradientBoostingClassifier(random_state=42) on the same
split scores 0.3296 with range [0.001, 0.999], so the count denominator
does not even reach it while the Newton denominator comfortably passes it.
```

### 3. Symmetric Tree Split

For a symmetric tree, all nodes at level L use the same split:

```
At each level, find split (feature, threshold) that maximizes:

Gain = Σ [Score(left_i) + Score(right_i) - Score(parent_i)]
       for all current partitions i

Where for each partition:
Score = G² / (H + λ)          (a similarity score - HIGHER is better)
- G = sum of gradients in the partition
- H = sum of Hessians in the partition. For squared loss h = 1, so H is just
      N, the sample count; for logloss h = p(1-p), so H is well below N.
- λ = l2_leaf_reg

ONE sign convention is used everywhere in this document and in the code:
Score is non-negative, a Gain above 0 means the split helps, and
`_build_symmetric_tree` maximises it with `np.argmax(scored_gains)`.

Process:
1. Start with all data as one partition
2. For each level:
   - Try all features and thresholds
   - Evaluate: if this split is applied to ALL partitions, what's total gain?
   - Pick best overall split
   - Apply it to ALL partitions → double the partitions
3. After depth levels: have 2^depth partitions (leaves)
```

### 4. Leaf Value Calculation

Optimal leaf value with L2 regularization:

```
w* = -G / (H + λ)

Where:
- G = Σ gᵢ for samples in leaf
- H = Σ hᵢ for samples in leaf. For squared loss h = 1, so H is simply N,
      the sample count; for log loss h = p(1-p), so H is well below N.
- λ = l2_leaf_reg (default: 3.0)

Interpretation:
- Without regularization (λ=0): w = -G/H (a Hessian-weighted average)
- With regularization: w is shrunk toward zero
- Leaves with little curvature (small H): more shrinkage
- Leaves with much curvature (large H): less shrinkage

Example (squared loss, where H = N so we can count samples):
Leaf with 10 samples, G = -5.0, λ = 3.0
w = -(-5.0) / (10 + 3) = 5.0 / 13 = 0.385

Same gradient with 2 samples:
w = 5.0 / (2 + 3) = 5.0 / 5 = 1.0
→ Smaller leaf gets more shrinkage!

For log loss you cannot substitute the count: see the eight-customer
walkthrough above, where H = 0.703 for a three-sample leaf and using 3
instead would shrink the step by more than 4x.
```

### 5. Prediction with Symmetric Trees

Fast prediction using binary indexing:

```
For a tree with depth D:
1. Initialize leaf_index = 0
2. For each level l from 0 to D-1:
   a. Check split condition at level l
   b. If condition FALSE (goes right):
      leaf_index += 2^(D-l-1)
3. Return leaf_value[leaf_index]

Example: depth = 3
Level 0: Income <= 60?  → NO  → leaf_index += 4 = 4
Level 1: Debt <= 20?    → YES → leaf_index += 0 = 4
Level 2: Loans <= 1?    → NO  → leaf_index += 1 = 5
→ Leaf index = 5 → return leaf_value[5]

Complexity: O(depth) per sample - the same as a traditional binary tree.
What differs is the constant factor: because the split at a level does not
depend on which node a sample landed in, all n samples are handled by ONE
numpy comparison per level (`_leaf_indices`) instead of n independent walks.
```

### 6. Ordered Boosting (Conceptual)

CatBoost addresses prediction shift:

```
Problem: Traditional Boosting
- Fit tree T using gradients from model M_{T-1}
- But M_{T-1} was trained on the SAME data
- Model has seen the labels during training
- Causes overfitting and prediction shift

Solution: Ordered Boosting
- Use multiple random permutations of data
- For sample i: calculate gradient using model trained only on samples BEFORE i
- Prevents target leakage

Simplified algorithm:
1. Create random permutation σ of training data
2. For sample at position i in σ:
   - Use model M_i trained only on σ[0:i]
   - Calculate gradient g_i using M_i
3. Build tree using these unbiased gradients

Full CatBoost implementation:
- Uses multiple permutations
- Maintains multiple models
- Complex but prevents overfitting

This implementation (`boosting_type='Ordered'`):
- ONE permutation sigma, and log2(n) supporting models M_j
- M_j has only ever been updated from the first 2^(j-1) rows of sigma
- The row at position p in sigma takes its gradient from the largest such
  model whose prefix ends at or before p, so no row is ever scored by a
  model that has seen it
- Those unbiased gradients choose the tree STRUCTURE; the returned model
  then re-fits the leaf values on its own gradients, which keeps its
  training loss monotonically decreasing
- Default is `boosting_type='Plain'` (classic boosting), because on the
  clean numeric synthetics used in this repo Ordered actually LOSES.
  Measured on USAGE EXAMPLE 5's data shape (200 rows x 5 features, test
  RMSE averaged over 10 seeds):
      100 trees, lr 0.05, depth 6:  Plain 1.1693, Ordered 1.2385  (+5.9%)
      150 trees, lr 0.05, depth 4:  Plain 0.9829, Ordered 1.0805  (+9.9%)
  Ordered wins on only 2 of those 10 seeds. Shrink the data and add noise
  and the sign flips, which is the regime ordered boosting was designed
  for: on 40 training rows with noise sigma 2.0 over 12 seeds it wins
  7/12, 2.4527 against Plain's 2.4814.
```

### 7. Ordered Target Statistics (for Categoricals)

Smart categorical encoding to prevent leakage:

```
Problem: Simple target encoding
- For category C: encode as mean(target | category = C)
- But this uses the SAME samples' targets
- Target leakage! Model has seen the answer

CatBoost's Solution: Ordered Target Statistics
1. Create random permutation of data
2. For sample i with category C:
   - Encode C as mean of target for samples with C that appear BEFORE i
   - Add prior (smooth with global mean)

Formula:
OTS(x_i) = (countPrior × prior + Σ y_j) / (countPrior + count)

Where:
- Sum over j: samples with same category BEFORE i in permutation
- prior: global mean target
- countPrior: smoothing parameter (typically 1-10)

Example:
Category "Red" appears at positions: 3, 7, 12, 18
Targets: 1, 0, 1, 1
Prior = 0.5, countPrior = 1

Position 3 (first Red): 
  OTS = (1×0.5 + 0) / (1 + 0) = 0.5  (only prior)
Position 7 (second Red):
  OTS = (1×0.5 + 1) / (1 + 1) = 0.75  (prior + first Red's target)
Position 12 (third Red):
  OTS = (1×0.5 + 1+0) / (1 + 2) = 0.5  (prior + first two Reds)
Position 18 (fourth Red):
  OTS = (1×0.5 + 1+0+1) / (1 + 3) = 0.625  (prior + first three Reds)

No target leakage! Each sample only uses previous samples' targets.
```

---

## Implementation Details

### Key Components

**1. Feature Quantization**
```python
def _quantize_features(self, X):
    # For each feature
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        # Create borders. A border must fall strictly BETWEEN two observed
        # values, never ON one: np.digitize(v, borders) counts borders <= v,
        # so a border sitting exactly on a value merges it with the next one
        # up. Put a border on 0 for a 0/1 feature and BOTH values land in
        # bin 1 - the feature becomes constant and can never be split on.
        if len(unique_values) <= border_count:
            borders = midpoints(unique_values)   # (u[:-1] + u[1:]) / 2
        else:
            percentiles = linspace(0, 100, border_count+1)[1:-1]  # interior only
            borders = percentile(feature_values, percentiles)
        
        # Assign bin indices
        X_quantized[:, feature_idx] = digitize(feature_values, borders)
```

**2. Symmetric Tree Building**
```python
def _build_symmetric_tree(self, X_quantized, gradients, hessians):
    splits = []
    # ONE integer per sample instead of a list of boolean masks:
    # partition p's children are numbered 2p (left) and 2p+1 (right)
    partition_id = zeros(n_samples, dtype=int)
    n_partitions = 1

    for level in range(depth):
        candidates = []
        for feature in features:
            # ONE pass per feature: histogram G, H and counts over every
            # (partition, bin) cell at once
            code = partition_id * n_bins + X_quantized[:, feature]
            g_hist = bincount(code, weights=gradients).reshape(n_partitions, n_bins)
            h_hist = bincount(code, weights=hessians ).reshape(n_partitions, n_bins)
            c_hist = bincount(code                   ).reshape(n_partitions, n_bins)

            # cumsum along bins IS the left child for every threshold at once;
            # the row total minus it is the right child
            g_left = cumsum(g_hist, axis=1);  g_right = g_left[:, -1:] - g_left
            h_left = cumsum(h_hist, axis=1);  h_right = h_left[:, -1:] - h_left

            gain = (score(g_left, h_left) + score(g_right, h_right)
                    - score(g_left[:, -1:], h_left[:, -1:]))   # score = G^2/(H+lambda)
            gain = where(split_is_legal_for_this_partition, gain, 0.0)
            candidates += [(feature, t, gain[:, t].sum()) for t in present_bins]

        if not candidates:          # nothing legal left to split on
            break
        best = argmax(jitter(candidates))   # jitter scaled by random_strength
        splits.append(best)
        partition_id = partition_id * 2 + (X_quantized[:, best.feature] > best.threshold)
        n_partitions *= 2

    leaf_values = [calculate_value(partition_id == p, gradients, hessians)
                   for p in range(n_partitions)]
    return {'splits': splits, 'leaf_values': leaf_values}
```

**3. Fast Symmetric Tree Prediction**
```python
def _predict_tree(self, tree, X_quantized):
    n_samples = len(X_quantized)
    leaf_indices = zeros(n_samples)
    
    # Binary indexing for fast lookup
    for level, split in enumerate(tree['splits']):
        feature_idx = split['feature']
        threshold = split['threshold']
        
        # Samples going right: add to leaf index
        goes_right = X_quantized[:, feature_idx] > threshold
        remaining_depth = tree['depth'] - level - 1
        leaf_indices += goes_right * (2 ** remaining_depth)
    
    # Get predictions from leaf values
    predictions = tree['leaf_values'][leaf_indices]
    return predictions
```

**4. Leaf Value with L2 Regularization**
```python
def _calculate_leaf_value(self, gradients, indices, hessians=None):
    gradient_sum = sum(gradients[indices])
    hessian_sum  = sum(indices) if hessians is None else sum(hessians[indices])

    # CatBoost formula: shrinkage through L2 reg.
    # For squared loss h = 1, so hessian_sum IS the sample count and this is
    # the familiar -G/(N+lambda). For logloss h = p(1-p), and using the count
    # instead would under-step every update by roughly 3-4x.
    value = -gradient_sum / (hessian_sum + l2_leaf_reg)

    return value
```

---

## Step-by-Step Example

Let's work through a complete example: predicting house prices (regression).

### Dataset

```
House Data:
ID | Size(sqft) | Bedrooms | Age(years) | Price($k)
---|------------|----------|------------|----------
1  |   1200     |    2     |     10     |    180
2  |   1800     |    3     |      5     |    280
3  |   1500     |    3     |     15     |    220
4  |   2200     |    4     |      3     |    350
5  |   1000     |    2     |     20     |    150
6  |   2500     |    4     |      2     |    400
7  |   1400     |    2     |     12     |    200
8  |   2000     |    3     |      7     |    300
```

### Step 1: Quantize Features

```
Size bins (border_count=2):
  Bin 0: Size ≤ 1650 → [1200, 1500, 1000, 1400]
  Bin 1: Size > 1650 → [1800, 2200, 2500, 2000]

Bedrooms bins:
  Bin 0: Bedrooms ≤ 2.5 → [2, 2, 2]
  Bin 1: Bedrooms > 2.5 → [3, 3, 4, 4, 3]

Age bins (border_count=2):
  Bin 0: Age ≤ 8.5 → [5, 3, 2, 7]
  Bin 1: Age > 8.5 → [10, 12, 15, 20]
```

### Step 2: Initialize

```
Mean price: (180 + 280 + 220 + 350 + 150 + 400 + 200 + 300) / 8 = 260

base_score = 260
Initial predictions: [260, 260, 260, 260, 260, 260, 260, 260]
```

### Step 3: Calculate Gradients

```
For regression (squared loss):
g = pred - y

ID | y   | pred | gradient
---|-----|------|----------
1  | 180 | 260  |   80
2  | 280 | 260  |  -20
3  | 220 | 260  |   40
4  | 350 | 260  |  -90
5  | 150 | 260  |  110
6  | 400 | 260  | -140
7  | 200 | 260  |   60
8  | 300 | 260  |  -40

Gradients: [80, -20, 40, -90, 110, -140, 60, -40]
```

### Step 4: Build Symmetric Tree (Depth=2)

```
LEVEL 0: Choose split for root

Try Size <= Bin 0 (Size ≤ 1650):
  Left (Small houses): [1, 3, 5, 7]
    Gradients: [80, 40, 110, 60]
    G_L = 290, N_L = 4
    Score_L = (290)² / (4 + 3) = 84100 / 7 = 12014.3

  Right (Large houses): [2, 4, 6, 8]
    Gradients: [-20, -90, -140, -40]
    G_R = -290, N_R = 4
    Score_R = (-290)² / (4 + 3) = 84100 / 7 = 12014.3

  Parent:
    G_P = 0, N_P = 8
    Score_P = 0² / (8 + 3) = 0

  Gain = (Score_L + Score_R) - Score_P = 24028.6 - 0 = +24028.6

(Same convention as everywhere else in this guide: Score = G²/(H+λ), higher is
 better, and a positive Gain means the split helps. This is a REGRESSION
 example, so h = 1 and H is exactly the sample count N - which is why every
 denominator below counts houses. The classification walkthrough earlier
 could not do that.  Here the parent's gradients
 cancel to exactly zero while each child's do not, so the split is very
 valuable - it separates the over-predicted houses from the under-predicted.)

Best split: Size <= 1650

Current partitions:
├─ Partition 0 (Small): [1, 3, 5, 7]
└─ Partition 1 (Large): [2, 4, 6, 8]
```

```
LEVEL 1: Choose ONE split for BOTH partitions

Try Bedrooms <= Bin 0 (Bedrooms ≤ 2.5):

Partition 0 (Small houses):
  Left (Small, ≤2 bed): [1, 5, 7] - All have 2 bedrooms
    G = 80 + 110 + 60 = 250
  Right (Small, >2 bed): [3] - Has 3 bedrooms
    G = 40

Partition 1 (Large houses):
  Left (Large, ≤2 bed): [] - None
  Right (Large, >2 bed): [2, 4, 6, 8] - All have 3-4 bedrooms
    G = -290

Calculate gain for this split...
Best split: Bedrooms <= 2.5

Final tree:
                [Size <= 1650]
               /              \
      [Bedrooms <= 2.5]  [Bedrooms <= 2.5]
         /        \          /        \
     Leaf0      Leaf1    Leaf2      Leaf3
   [1,5,7]      [3]       []      [2,4,6,8]
```

### Step 5: Calculate Leaf Values

```
Leaf 0 (Small, ≤2 bed): [1, 5, 7]
  G = 250, N = 3
  value = -250 / (3 + 3) = -41.67

Leaf 1 (Small, >2 bed): [3]
  G = 40, N = 1
  value = -40 / (1 + 3) = -10.00

Leaf 2 (Large, ≤2 bed): []
  value = 0 (empty leaf)

Leaf 3 (Large, >2 bed): [2, 4, 6, 8]
  G = -290, N = 4
  value = -(-290) / (4 + 3) = 41.43
```

### Step 6: Update Predictions

```
Learning rate η = 0.05

House 1 (Leaf 0): 260 + 0.05 × (-41.67) = 260 - 2.08 = 257.92
House 3 (Leaf 1): 260 + 0.05 × (-10.00) = 260 - 0.50 = 259.50
House 2 (Leaf 3): 260 + 0.05 × 41.43 = 260 + 2.07 = 262.07
...

After this iteration:
- Small houses with ≤2 bed: predictions decrease (were overestimated)
- Large houses with >2 bed: predictions increase (were underestimated)
- Model is learning the pattern!
```

### Step 7: Continue Iterations

```
Iteration 2: Calculate new gradients from updated predictions
Iteration 3: Build another symmetric tree
...
Iteration 100: Final model

Final predictions after 100 trees:
(measured by actually running CatBoost(n_estimators=100, learning_rate=0.05,
 depth=2, l2_leaf_reg=3.0, random_strength=0) on the 8 houses above)

House 1 (Small, 2 bed, old):    Predicted 183 (actual 180) ✓
House 4 (Large, 4 bed, new):    Predicted 347 (actual 350) ✓
House 5 (Smallest, 2 bed, old): Predicted 177 (actual 150) ✗

Why is house 5 off by 27k? A depth-2 tree has only 4 leaves, and houses
1, 5 and 7 all land in the same one (small, ≤2 bedrooms). Every tree must
give them the SAME value, so the model can only predict their average
(~177) and cannot tell the 1000 sqft house apart from the 1400 sqft one.
That is symmetric trees showing their cost: raise depth (or add trees at a
lower learning rate) and the leaf splits finer. It is also why 8 rows is a
teaching example, not a benchmark.

New house [1600 sqft, 3 bed, 8 years]:
1. Size 1600 ≤ 1650? YES → Partition 0
2. Bedrooms 3 > 2.5? YES → Leaf 1
3. Sum contributions from Leaf 1 across all trees
4. Final prediction = 220k (measured)
```

---

## Real-World Applications

### 1. E-commerce: Product Categorization

**Problem**: Automatically categorize products from titles and features

**Why CatBoost?**
- Many categorical features (brand, seller, category)
- Handles text-derived features naturally
- Fast training for millions of products
- Excellent accuracy out-of-the-box

**Features**:
```
Text: product_title_words (categorical)
Categorical: brand, seller_id, existing_category
Numerical: price, weight, dimensions
Derived: brand_price_segment, title_length
```

**Benefits**:
- No need for extensive one-hot encoding
- Natural handling of rare brands/sellers
- Robust to new categories
- 95%+ categorization accuracy

### 2. Finance: Credit Scoring

**Problem**: Predict loan default risk

**Why CatBoost?**
- Handles missing values well
- Categorical features (occupation, location)
- Ordered boosting prevents overfitting on small datasets
- Regulatory compliance (explainable predictions)

**Features**:
```
Categorical: occupation, city, education, marital_status
Numerical: income, debt_to_income, credit_score, age
Derived: income_to_loan_ratio, employment_stability_score
```

**Benefits**:
- Better risk assessment (20-30% improvement over logistic regression)
- Handles rare occupations/locations without overfitting
- Feature importance for regulatory explanation
- Robust with default parameters

### 3. Retail: Customer Lifetime Value (CLV) Prediction

**Problem**: Predict total revenue from each customer

**Why CatBoost?**
- Mix of categorical and numerical features
- Long tail of customer behaviors
- Need for accurate predictions across segments

**Features**:
```
Categorical: acquisition_channel, first_product_category, location_tier
Numerical: days_since_signup, total_orders, avg_order_value
Behavioral: browsing_frequency, email_engagement, support_contacts
```

**Benefits**:
- Accurate CLV predictions enable targeted marketing
- Segment customers effectively
- 15-20% improvement over traditional methods
- Fast retraining with new data

### 4. Healthcare: Disease Diagnosis Support

**Problem**: Assist in disease diagnosis from symptoms and test results

**Why CatBoost?**
- Categorical symptoms (yes/no, severity levels)
- Numerical lab values
- Handles missing test results naturally
- High accuracy requirements

**Features**:
```
Categorical: symptoms (fever, cough, fatigue), medical_history
Numerical: lab_values (blood_pressure, glucose, white_cell_count)
Demographic: age, gender, bmi
```

**Benefits**:
- High diagnostic accuracy (comparable to specialists)
- Probability scores help prioritize cases
- Interpretable feature importance
- Robust to missing lab values

### 5. Web Analytics: User Conversion Prediction

**Problem**: Predict if website visitor will convert

**Why CatBoost?**
- Many categorical features (device, browser, referrer)
- Session-based features
- Need for fast online predictions
- Handle cold-start (new visitors)

**Features**:
```
Categorical: traffic_source, device_type, browser, landing_page
Behavioral: pages_viewed, time_on_site, scroll_depth
Contextual: day_of_week, hour, season
Historical: previous_visits, email_subscriber
```

**Benefits**:
- Real-time conversion probability
- Personalized content recommendations
- 10-15% increase in conversion rate
- Fast prediction (< 1ms per user)

---

## Understanding the Code

### Core Class Structure

```python
class CatBoost:
    def __init__(self, n_estimators=100, learning_rate=0.03,
                 depth=6, l2_leaf_reg=3.0, **other_params):
        # Key parameters
        self.depth = depth  # Tree depth (controls 2^depth leaves)
        self.learning_rate = learning_rate  # Shrinkage
        self.l2_leaf_reg = l2_leaf_reg  # Regularization strength
        ...

    def fit(self, X, y):
        # 1. Encode cat_features with ordered target statistics
        # 2. Quantize features into bins
        # 3. Initialize predictions with base_score
        # 4. Train symmetric trees sequentially
        ...

    def predict(self, X):
        # 1. Apply the stored categorical encoding, then quantization
        # 2. Fast prediction using binary indexing
        # 3. Convert to probabilities if classification
        ...
```

### Key Methods Explained

**1. Feature Quantization**
```python
def _quantize_features(self, X):
    """
    Convert continuous features to discrete bins
    
    Why: Faster split evaluation and more robust
    - Original: Try every unique value
    - Quantized: Try only bin boundaries
    - Adds regularization through discretization
    
    Example: 
      Prices: [100, 150, 180, 220, 250, 300]
      With border_count=3: 
        Bin 0 (≤165), Bin 1 (165-235), Bin 2 (>235)
    """
```

**2. Symmetric Tree Building**
```python
def _build_symmetric_tree(self, X_quantized, gradients):
    """
    Build tree where all nodes at same level use same split
    
    Why symmetric trees?
    - Natural regularization (simpler structure)
    - Faster prediction (binary indexing)
    - Less prone to overfitting
    - Easier to parallelize
    
    Algorithm:
    - For each level (0 to depth-1):
      * Find ONE best split for ALL current partitions
      * Apply it to ALL partitions
      * Double the number of partitions
    - Result: 2^depth leaves with symmetric structure
    """
```

**3. Fast Prediction with Binary Indexing**
```python
def _predict_tree(self, tree, X_quantized):
    """
    Fast prediction using binary representation of tree path
    
    Why fast?
    - Both tree kinds need O(depth) comparisons per sample - the
      asymptotics are the same, and anyone claiming otherwise is wrong.
    - The win is the constant: a traditional tree walks each sample down
      its OWN path (branching, pointer chasing). A symmetric tree uses the
      same (feature, threshold) for every sample at a level, so one numpy
      comparison handles the entire batch, and the leaf index falls out of
      bit arithmetic + a single fancy-index lookup.
    
    Algorithm:
    1. Start with leaf_index = 0
    2. For each level's split:
       - If goes RIGHT: add 2^(remaining_depth)
       - If goes LEFT: add 0
    3. Return leaf_value[leaf_index]
    
    Example (depth=3):
      Path: R-L-R
      Index: 0 + 4 + 0 + 1 = 5
      Return: leaf_value[5]
    """
```

**4. Leaf Value with Strong Regularization**
```python
def _calculate_leaf_value(self, gradients, indices, hessians=None):
    """
    Calculate optimal leaf value with L2 regularization
    
    Formula: value = -sum(gradients) / (sum(hessians) + l2_leaf_reg)
             (hessians=None means "all ones", the squared-loss case, where
              sum(hessians) is just the sample count)
    
    Why L2 in denominator?
    - Shrinks leaf values toward zero
    - More shrinkage for leaves with little curvature (small sum of h)
    - Less shrinkage for leaves with much curvature (large sum of h)
    - Prevents overfitting to small groups
    
    Example (squared loss, so sum(hessians) == the sample count):
      Leaf with 100 samples, sum(g)=-50, λ=3:
        value = 50 / (100 + 3) = 0.485
      
      Leaf with 5 samples, sum(g)=-50, λ=3:
        value = 50 / (5 + 3) = 6.25
        (the unregularized value would be 50/5 = 10.0, so λ=3 cuts it by 37%,
         versus only 3% for the 100-sample leaf - the SMALL leaf is shrunk
         much harder, which is exactly the intent)
    
    CatBoost default λ=3.0 is higher than XGBoost's 1.0!
    """
```

### Important Parameters

**Tree Structure:**
```python
depth=6                # Tree depth (2^6 = 64 leaves)
                       # Controls model complexity
                       # Typical: 4-10
                       
min_data_in_leaf=1     # Min samples per leaf
                       # CatBoost trusts regularization, uses 1
```

**Learning:**
```python
learning_rate=0.03     # Shrinkage (lower than XGBoost/LightGBM)
                       # CatBoost uses conservative default
                       # Typical: 0.01-0.1
                       
n_estimators=100       # Number of trees
                       # More trees with lower learning rate
```

**Regularization:**
```python
l2_leaf_reg=3.0        # L2 regularization strength
                       # Higher than XGBoost default (1.0)
                       # Strong regularization prevents overfitting
                       # Typical: 1-10
```

**Speed vs Accuracy:**
```python
border_count=128       # Number of feature bins
                       # Higher = more accurate, slower
                       # Typical: 32, 64, 128, 254
```

**Randomness:**
```python
random_strength=1.0    # Randomization in split selection
                       # Jitter added to each candidate's gain is
                       #   random_strength * std(this level's gains) * N(0,1)
                       # Scaling by the level's own spread is what makes the
                       # parameter mean the same thing whether your target is
                       # in dollars or thousands of dollars
                       # Typical: 0-2

random_seed=None       # Seed for this model's private RNG
                       # int  -> reproducible regardless of global RNG state
                       # None -> seeded FROM the global RNG, so an outer
                       #         np.random.seed(42) still reproduces the fit
```

**Categorical features and boosting scheme:**
```python
cat_features=None      # Column indices holding categories (strings or codes)
                       # e.g. cat_features=[0, 3]
                       # Those columns are encoded with ordered target
                       # statistics instead of one-hot / label encoding
                       # None = every column is numeric

boosting_type='Plain'  # 'Plain'   = classic gradient boosting (default)
                       # 'Ordered' = CatBoost's unbiased scheme; a sample's
                       #             gradient comes from a supporting model
                       #             that never saw that sample
                       # Try 'Ordered' on small, noisy, categorical data
```

### Parameter Tuning Guidelines

**Start with defaults:**
```python
model = CatBoost(
    n_estimators=100,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0
)
# CatBoost has great defaults! Often works well as-is.
```

**If underfitting (train and test loss both high):**
```python
# Increase model complexity
model = CatBoost(
    n_estimators=200,      # More trees
    depth=8,               # Deeper trees
    learning_rate=0.05,    # Slightly higher rate
    l2_leaf_reg=1.0        # Less regularization
)
```

**If overfitting (train loss low, test loss high):**
```python
# Increase regularization
model = CatBoost(
    n_estimators=100,
    depth=4,               # Shallower trees
    learning_rate=0.03,
    l2_leaf_reg=10.0,      # More regularization
    random_strength=2.0    # More randomness
)
```

---

## Model Evaluation

### Metrics to Use

**Regression:**
```python
# RMSE (Root Mean Squared Error)
rmse = -model.score(X_test, y_test)  # Note: score returns negative RMSE
print(f"RMSE: {rmse:.2f}")

# Mean Absolute Error
predictions = model.predict(X_test)
mae = np.mean(np.abs(y_test - predictions))
print(f"MAE: {mae:.2f}")

# R² Score
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - predictions) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(f"R²: {r2:.4f}")
```

**Classification:**
```python
# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

# Confusion Matrix and Metrics
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int)

TP = np.sum((predicted_classes == 1) & (y_test == 1))
FP = np.sum((predicted_classes == 1) & (y_test == 0))
FN = np.sum((predicted_classes == 0) & (y_test == 1))
TN = np.sum((predicted_classes == 0) & (y_test == 0))

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.4f}")

# ROC-AUC
# (Would need to implement or use sklearn for full ROC curve)
```

### Feature Importance

```python
# Train model
model.fit(X_train, y_train)

# Get importance
importance = model.get_feature_importance('split')

# Display
feature_names = ['size', 'bedrooms', 'age', 'location']
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    bar = '#' * int(imp * 50)     # ASCII only: U+2588 crashes a cp1252 console
    print(f"{name:15s}: {imp:.4f} {bar}")

# Illustrative output shape (your numbers depend on your data):
# size           : 0.4821 ########################
# bedrooms       : 0.3012 ###############
# age            : 0.1567 ########
# location       : 0.0600 ###
```

### Learning Curves

```python
# Train with validation set
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Plot learning curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(model.train_scores, label='Train')
plt.plot(model.val_scores, label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()

# Interpret:
# - Train and val decreasing: Model learning well
# - Val starts increasing: Overfitting, use early stopping
# - Val plateaus: Model converged, can stop early
```

### Cross-Validation

```python
# Manual K-Fold Cross-Validation
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    model = CatBoost(n_estimators=100, learning_rate=0.03, depth=6)
    model.fit(X_train_fold, y_train_fold)
    
    score = model.score(X_val_fold, y_val_fold)
    scores.append(score)
    print(f"Fold {fold+1}: {score:.4f}")

print(f"\nMean CV Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

### Avoiding Overfitting

**Signs of Overfitting:**
```python
# score() returns NEGATIVE RMSE for regression, so negate it to read an error
train_rmse = -model.score(X_train, y_train)  # e.g. 0.0682
test_rmse  = -model.score(X_test,  y_test)   # e.g. 0.9311
# Test RMSE much larger than train RMSE = overfitting!
# (those are USAGE EXAMPLE 7's measured l2_leaf_reg=0.1 numbers; raising
#  l2_leaf_reg to 3.0 narrows the gap to 0.2794 train / 0.8344 test)
```

**Solutions:**

1. **Use Early Stopping:**
```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20  # Stop if no improvement for 20 rounds
)
```

2. **Reduce Tree Depth:**
```python
model = CatBoost(depth=4)  # Was 6, now shallower
```

3. **Increase L2 Regularization:**
```python
model = CatBoost(l2_leaf_reg=10.0)  # Was 3.0, now stronger
```

4. **Lower Learning Rate with More Trees:**
```python
model = CatBoost(
    n_estimators=300,      # More trees
    learning_rate=0.01     # Lower rate
)
```

5. **Add Randomness:**
```python
model = CatBoost(random_strength=2.0)  # More randomization
```

---

## CatBoost vs XGBoost vs LightGBM

### Comparison Table

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Tree Growth** | Level-wise | Leaf-wise | Level-wise (symmetric) |
| **Default LR** | 0.3 | 0.1 | 0.03 |
| **Categorical Handling** | Manual encoding | Manual encoding | Native (ordered target statistics) |
| **Speed (Large Data)** | Medium | **Fastest** | Fast |
| **Overfitting Risk** | Medium | Higher | **Lower** |
| **Default Performance** | Good | Good | **Best** |
| **Tree Structure** | Asymmetric | Asymmetric | **Symmetric** |
| **Best For** | Competitions | Speed & large data | Categoricals & robustness |

### When to Use Each

**Use XGBoost when:**
- Industry standard needed
- Extensive documentation/resources needed
- Medium-sized datasets
- Have time for hyperparameter tuning
- Need ecosystem support (wide community)

**Use LightGBM when:**
- Speed is critical
- Very large datasets (>1M samples)
- Memory is limited
- Numerical features dominate
- Need GPU acceleration

**Use CatBoost when:**
- Many categorical features ← **Best choice!**
- Want great results with default parameters
- Overfitting is a concern
- Need robust, production-ready model
- Limited time for tuning

### Accuracy Comparison

```
Typical benchmark results:

Numerical features only:
LightGBM ≈ XGBoost ≈ CatBoost

Many categorical features:
CatBoost > LightGBM > XGBoost

Small datasets (<10K):
CatBoost ≈ XGBoost > LightGBM

Large datasets (>100K):
LightGBM ≥ CatBoost > XGBoost

Default parameters:
CatBoost > LightGBM > XGBoost
(CatBoost has best defaults!)
```

### Speed Comparison

```
Dataset: 100K samples, 50 features

Training Time:
├── XGBoost: 45 seconds
├── LightGBM: 12 seconds  ← Fastest!
└── CatBoost: 30 seconds

Prediction Time (1000 samples):
├── XGBoost: 15 ms
├── LightGBM: 8 ms
└── CatBoost: 5 ms  ← Fastest! (symmetric trees)

Why CatBoost prediction is fast?
- Symmetric trees → one comparison per level for the WHOLE batch (branch-free),
  not a lower asymptotic cost - a normal binary tree is also O(depth)
- Binary indexing → direct lookup, no pointer chasing
- No need for tree traversal
```

---

## Advantages & Limitations

### Advantages of CatBoost (the algorithm)

| Advantage | Why it matters |
|-----------|----------------|
| **Native categorical handling** | Ordered target statistics encode a category by the targets of *earlier* rows only, so you get the predictive power of target encoding without its leakage, and no feature explosion from one-hot |
| **Ordered boosting** | Removes the prediction shift caused by scoring a sample with a model that already fitted it - the benefit grows as the dataset shrinks |
| **Symmetric (oblivious) trees** | Every level is one split, so the model has far fewer degrees of freedom than a free-form tree of the same depth: strong built-in regularization |
| **Branch-free prediction** | The whole batch shares one comparison per level and one fancy-index lookup, which vectorizes cleanly |
| **Strong defaults** | `learning_rate=0.03`, `l2_leaf_reg=3.0` and `depth=6` are deliberately conservative; CatBoost usually performs well before any tuning |

### Limitations of CatBoost (the algorithm)

- **Slower training than LightGBM** on large numeric datasets - obliviousness costs accuracy per tree, so more trees are needed
- **Symmetric trees underfit** genuinely asymmetric structure: if only one region of feature space needs a deep split, every region gets it
- **Ordered boosting costs memory and time**, maintaining several supporting models
- **Target statistics need enough rows per category** - with a handful of examples the encoding is mostly prior

### Limitations of THIS from-scratch implementation

This file is written to be read, not to be deployed. Concretely:

| Area | What is here | What real CatBoost does |
|------|--------------|-------------------------|
| Split search | Exhaustive over every (feature, bin), histogram-accelerated | Same idea, plus multithreading, GPU kernels and feature bundling |
| Border selection | Uniform quantiles (`border_count` of them) | Several strategies, `GreedyLogSum` by default |
| Ordered boosting | One permutation, log2(n) supporting models, structure only | Several permutations, and its own criterion for picking structure |
| Categorical features | Ordered target statistics averaged over 4 permutations | Same, plus feature *combinations* built greedily during training |
| Objectives | `'regression'` (RMSE) and `'binary'` (Logloss) | Dozens, including multiclass, ranking and custom losses |
| Missing values | Not handled - `NaN` propagates | Dedicated `Min`/`Max` NaN handling per feature |
| Subsampling | None | Bagging, `rsm` column sampling, MVS |

### Simplification vs. canonical CatBoost

Three gaps are worth naming precisely, because they are where this code and
the paper (*"CatBoost: unbiased boosting with categorical features"*,
NeurIPS 2018) genuinely diverge:

1. **Categorical feature combinations.** Real CatBoost greedily builds
   *combinations* of categorical features (e.g. `country × device`) as the
   tree grows, encoding each combination with its own target statistic. That
   is what lets it capture interactions between high-cardinality columns.
   This implementation encodes each categorical column independently. The
   practical consequence: an interaction between two categoricals must be
   discovered through ordinary splits on their separate statistics, which
   needs more depth and more trees.

2. **Multiple permutations for boosting.** The paper samples several
   permutations and alternates between them across trees, which averages out
   the noise a single ordering introduces (a row that lands early sees almost
   no history). Here `boosting_type='Ordered'` uses ONE permutation for the
   whole fit. The ordered *target statistics* do average over 4 permutations,
   which is why they behave well; ordered *boosting* does not, which is part
   of why it is not the default.

3. **Structure-selection criterion.** The paper scores candidate structures
   against the supporting models with a cosine-similarity criterion. Here the
   unbiased gradients feed the ordinary `G²/(H+λ)` gain, which is simpler and
   is the criterion the rest of this guide teaches.

Measured consequence of (2) and (3): on the clean numeric synthetics used
throughout this repo, `boosting_type='Ordered'` is 6-10% WORSE on test RMSE
than `'Plain'` - not a hair behind it. On USAGE EXAMPLE 5's data shape over
10 seeds it scores 1.2385 against Plain's 1.1693 at depth 6, and 1.0805
against 0.9829 at depth 4, winning on only 2 seeds of 10. It does come out
ahead where the theory says it should: on 40 noisy training rows over 12
seeds it wins 7/12 (2.4527 vs 2.4814). Ordered boosting here is implemented
faithfully enough to demonstrate the mechanism, not to reproduce the paper's
benchmark wins.

---

## Summary

### Key Takeaways

1. **CatBoost = Robustness + Categorical Intelligence**
   - Symmetric trees → Natural regularization
   - Ordered boosting → Prevents target leakage
   - Native categorical handling → No manual encoding needed
     (here: `CatBoost(cat_features=[0, 3])`)
   - Great defaults → Works well out-of-the-box

2. **Main Innovations**
   - **Symmetric Trees**: All nodes at level use same split → simpler, faster
   - **Ordered Boosting**: Prevents prediction shift → more robust
   - **Ordered Target Statistics**: Smart categorical encoding → no leakage
   - **Strong Regularization**: High default L2 (3.0) → less overfitting

3. **Best Practices**
   ```python
   # Start here (usually works great!)
   model = CatBoost(
       n_estimators=100,
       learning_rate=0.03,
       depth=6,
       l2_leaf_reg=3.0
   )
   
   # If underfitting
   model = CatBoost(
       n_estimators=200,
       depth=8,
       learning_rate=0.05,
       l2_leaf_reg=1.0
   )
   
   # If overfitting
   model = CatBoost(
       depth=4,
       l2_leaf_reg=10.0,
       random_strength=2.0
   )
   ```

4. **When to Use CatBoost**
   - ✅ Many categorical features (best choice!)
   - ✅ Want good results with minimal tuning
   - ✅ Concerned about overfitting
   - ✅ Need robust production model
   - ✅ Small to medium datasets
   - ❌ Very large datasets where speed is critical (use LightGBM)

### Comparison Summary

```
XGBoost:  "The Industry Standard"
          + Mature, well-documented
          + Good balance of speed and accuracy
          - Needs more tuning
          - Manual categorical handling

LightGBM: "The Speed Champion"
          + Fastest on large datasets
          + Memory efficient
          + Great for numerical features
          - Easier to overfit
          - Needs careful tuning

CatBoost: "The Robust Expert"
          + Best with categorical features
          + Great default parameters
          + Less prone to overfitting
          + Fastest prediction (symmetric trees)
          - Slower training than LightGBM
          - Newer, smaller community
```

### Quick Decision Guide

```
Do you have categorical features?
├─ YES → Use CatBoost
└─ NO → Is speed critical?
    ├─ YES → Use LightGBM
    └─ NO → Use XGBoost or CatBoost
    
Is dataset small (<10K samples)?
├─ YES → Use CatBoost (more robust)
└─ NO → Is it huge (>1M samples)?
    ├─ YES → Use LightGBM (fastest)
    └─ NO → Use CatBoost (best defaults)

Limited time for tuning?
└─ Use CatBoost (best out-of-the-box)
```

### Next Steps

1. **Run the file**: `python _19_catboost.py` runs the three demos from the
   Quick Start (regression, classification, and a raw string column)
2. **Compare with your data** - try defaults first!
3. **Add categorical features** with `cat_features=[...]` - see CatBoost shine
4. **Monitor for overfitting** - use validation set
5. **Compare with XGBoost/LightGBM** - see the differences
6. **Study symmetric trees** - understand the structure

---

## References and Further Learning

### Official Resources
- **CatBoost Documentation**: https://catboost.ai/
- **Paper**: "CatBoost: unbiased boosting with categorical features" (NeurIPS 2018)
- **GitHub**: https://github.com/catboost/catboost
- **Tutorial**: https://catboost.ai/docs/concepts/tutorials.html

### Key Concepts to Explore
- Symmetric (oblivious) decision trees
- Ordered boosting and prediction shift
- Ordered target statistics for categorical features
- Comparison with XGBoost and LightGBM
- Handling of missing values

### Related Algorithms
- XGBoost (main competitor, asymmetric trees)
- LightGBM (main competitor, leaf-wise growth)
- Gradient Boosting (foundation algorithm)
- Random Forests (alternative ensemble method)

### Advanced Topics
- GPU acceleration in CatBoost
- Distributed training
- Custom loss functions
- Text features handling
- Embeddings for categorical features

---

**Remember**: CatBoost is robust and smart! It's especially powerful when you have categorical features and want excellent results without extensive tuning. Happy learning! 🚀

---

*This guide is part of the "ML Algorithms from Scratch" series. For more algorithms, check out the repository!*
