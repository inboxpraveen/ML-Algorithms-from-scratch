# Random Forests from Scratch: A Comprehensive Guide

Welcome to the world of Random Forests! 🌲🌲🌲 In this comprehensive guide, we'll explore one of the most powerful and popular machine learning algorithms. Think of it as a "committee of experts" where many decision trees vote together!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What are Random Forests?](#what-are-random-forests)
3. [How Random Forests Work](#how-random-forests-work)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Advantages and Limitations](#advantages-and-limitations)
11. [Simplifications vs. Canonical Random Forest](#simplifications-vs-canonical-random-forest)
12. [Choosing Hyperparameters](#choosing-hyperparameters)
13. [Complete Usage Example](#complete-usage-example)
14. [Key Concepts to Remember](#key-concepts-to-remember)
15. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Random Forest from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _7_random_forests.py  (the __main__ block runs this,
#           plus a third demo comparing one tree vs bagging vs a forest)
# Or paste the DecisionTree class from _6_decision_trees.py, then the
# _RandomFeatureTree and RandomForest classes from _7_random_forests.py, above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the DecisionTree class here (from _6_decision_trees.py) ----
# class DecisionTree: ...
# ---- Paste _RandomFeatureTree and RandomForest here (from _7_random_forests.py) ----
# class _RandomFeatureTree(DecisionTree): ...
# class RandomForest: ...

np.random.seed(42)

# ------ CLASSIFICATION: two Gaussian blobs in 4 dimensions ------
X0 = np.random.randn(100, 4) - 1.5     # class 0
X1 = np.random.randn(100, 4) + 1.5     # class 1
X = np.vstack([X0, X1])
y = np.array([0] * 100 + [1] * 100)

# Shuffle before splitting: the rows are stacked class-by-class, so an
# unshuffled 150/50 split would put class 0 in train and class 1 in test.
idx = np.random.permutation(200)
X, y = X[idx], y[idx]
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

clf = RandomForest(
    n_estimators=25,
    max_depth=5,
    task='classification',
    max_features='sqrt',   # k = floor(sqrt(4)) = 2 features per split
    oob_score=True,        # free validation on the ~36.8% out-of-bag rows
    random_state=42
)
clf.fit(X_train, y_train)

print(f"Features per split (k of p) : {clf.max_features_} of {clf.n_features_}")
print(f"Train Accuracy              : {clf.score(X_train, y_train):.4f}")
print(f"Test  Accuracy              : {clf.score(X_test,  y_test):.4f}")
print(f"Out-of-bag Accuracy         : {clf.oob_score_:.4f}")

proba = clf.predict_proba(X_test)
for i in range(3):
    print(f"  true={y_test[i]}  P(0)={proba[i, 0]:.2f}  P(1)={proba[i, 1]:.2f}")

# ------ REGRESSION: y = x^2 + noise ------
X_reg = np.linspace(-3, 3, 200).reshape(-1, 1)
y_reg = X_reg.ravel() ** 2 + np.random.randn(200) * 0.5

# Shuffle before splitting: linspace is sorted, and trees cannot extrapolate
# beyond the x-range they were trained on.
idx = np.random.permutation(200)
X_reg, y_reg = X_reg[idx], y_reg[idx]

reg = RandomForest(n_estimators=25, max_depth=6, task='regression',
                   criterion='mse', random_state=42)
reg.fit(X_reg[:150], y_reg[:150])

print(f"\nTrain R2 : {reg.score(X_reg[:150], y_reg[:150]):.4f}")
print(f"Test  R2 : {reg.score(X_reg[150:], y_reg[150:]):.4f}")

preds = reg.predict(X_reg[150:])
for i in range(3):
    print(f"  x={X_reg[150 + i, 0]:5.2f}  true={y_reg[150 + i]:5.2f}  pred={preds[i]:5.2f}")
```

Expected output:
```
Features per split (k of p) : 2 of 4
Train Accuracy              : 1.0000
Test  Accuracy              : 0.9800
Out-of-bag Accuracy         : 0.9867
  true=1  P(0)=0.08  P(1)=0.92
  true=0  P(0)=1.00  P(1)=0.00
  true=1  P(0)=0.00  P(1)=1.00

Train R2 : 0.9806
Test  R2 : 0.9354
  x=-0.83  true= 0.77  pred= 0.43
  x=-1.13  true= 2.26  pred= 1.36
  x=-0.32  true= 0.53  pred= 0.36
```

Runs in under a second. Notice that `P(0)=0.08` means exactly 2 of the 25 trees voted for class 0 - these probabilities are vote fractions, so they always land on a multiple of `1/n_estimators`.

---

## What are Random Forests?

Random Forests are **ensemble learning methods** that combine multiple decision trees to create a more robust and accurate model. Instead of relying on a single decision tree, a random forest uses the "wisdom of the crowd" by combining predictions from many trees.

**Real-world analogy**:
Imagine you're trying to predict tomorrow's weather. Instead of asking one meteorologist, you ask 100 different meteorologists and take a majority vote. Even if some are wrong, the collective wisdom is usually more accurate. That's exactly how a Random Forest works!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Ensemble, Tree-based |
| **Learning Style** | Supervised, Parallel training |
| **Tasks** | Classification and Regression |
| **Output** | Multiple decision trees voting together |
| **Key Advantage** | Reduces overfitting compared to single trees |

### The Core Ideas

```
1. "Many trees are better than one" - Ensemble learning
2. "Bootstrap sampling" - Each tree trained on random data subset (random ROWS)
3. "Random feature subsets" - Each split considers only k random features
                              (random COLUMNS) - this is the "Random" in the name
4. "Majority voting" - Classification combines predictions by voting
5. "Averaging" - Regression combines predictions by averaging
```

A Random Forest:
1. **Creates** many decision trees (typically 50-500)
2. **Trains** each tree on a random subset of training data (with replacement)
3. **Restricts** each split inside each tree to a random subset of the features
4. **Makes** each tree vote on the final prediction
5. **Combines** all votes to make the final prediction

> **Bagging vs. Random Forest.** Take away idea 3 and you have plain *bagging* -
> bootstrap aggregation of ordinary decision trees. Bagging works, but every tree
> tends to find the same dominant feature and split on it first, so the trees stay
> highly correlated and averaging them removes less variance than you would hope.
> Random feature subsets are Breiman's fix: they force different trees to look at
> different evidence. In this repository you can flip between the two with a single
> argument: `max_features=None` gives you bagging, `max_features='sqrt'` gives you
> a Random Forest.

---

## How Random Forests Work

### The Algorithm in 5 Steps

```
Step 1: Randomly sample training data (with replacement) for each tree
         |
         v
Step 2: Build a decision tree on each random sample
        Step 2a: at EVERY node, draw k of the p features at random
        Step 2b: pick the best split among those k features only
         |
         v
Step 3: Repeat Steps 1-2 for n_estimators trees
         |
         v
Step 4: For new data, each tree makes a prediction
         |
         v
Step 5: Final prediction = majority vote (classification) or average (regression)
```

### Random Feature Selection (the "Random" in Random Forest)

At every node the tree normally scans all `p` features to find the best split.
A Random Forest scans only `k` of them, drawn fresh at each node **without
replacement**:

```
Standard decision tree node:   consider features {0, 1, 2, ..., p-1}
Random Forest node:            consider features {3, 7}        (k = 2, redrawn each node)
                               next node:        {0, 5}
                               next node:        {2, 7}
```

**Standard choices for k** (`max_features` in the code):

| Setting | k | Where it comes from |
|---------|---|---------------------|
| `'sqrt'` | floor(sqrt(p)) | Breiman's classification recommendation; scikit-learn's classifier default |
| `1/3` (float) | floor(p/3) | Breiman's regression recommendation |
| `None` | p | All features -> this is plain bagging; scikit-learn's regressor default |
| `'log2'` | floor(log2(p)) | Aggressive decorrelation for very wide data |

`max_features='auto'` (our default) picks `sqrt(p)` for classification and `p`
for regression, matching scikit-learn's two defaults.

> **Read that regression default carefully.** `k = p` means every column is a
> candidate at every node, which is exactly the `max_features=None` row above -
> so **a default regression forest here does no column subsampling at all; it is
> plain bagging**, bit-for-bit identical to passing `max_features=None`. That is
> scikit-learn's default too, and it is a defensible one, but it means the
> mechanism this whole section is about is switched off unless you turn it on.
> For a genuine regression forest pass `max_features=1/3` (Breiman's rule) or
> `max_features='sqrt'`. The classification default *does* subsample.

**Why smaller k helps**: it stops one strong predictor from dominating every
tree. The individual trees get a bit worse (higher bias, since they sometimes
have to split on a mediocre feature), but they get much *less alike*, and the
averaging step then removes far more variance. The mathematics of that trade is
in [Bias-Variance Tradeoff](#bias-variance-tradeoff) below.

### Bootstrap Sampling (Bagging)

**What is bootstrap sampling?**
Random sampling **with replacement** - meaning the same data point can be selected multiple times.

**Visual Example**:
```
Original Data: [A, B, C, D, E, F, G, H]  (8 samples)

Bootstrap Sample 1: [A, A, C, D, F, F, G, H]  # A and F appear twice
Bootstrap Sample 2: [B, C, D, E, E, G, H, H]  # E and H appear twice
Bootstrap Sample 3: [A, B, C, C, D, E, F, G]  # C appears twice

Each tree sees different data = more diversity!
```

**Why it works**:
- Each tree learns from slightly different data
- Reduces overfitting - no single tree memorizes all training data
- Creates diversity among trees
- Averages out individual tree errors

### Visual Example: Forest Prediction

```
Training Data (Loan Approval):
Features: [Age, Income, Credit_Score]
Classes: Approve / Reject

Building Random Forest with 5 trees:

Tree 1: Random 10 rows + random features → Builds tree → Predicts: Approve
Tree 2: Random 10 rows + random features → Builds tree → Predicts: Approve
Tree 3: Random 10 rows + random features → Builds tree → Predicts: Reject
Tree 4: Random 10 rows + random features → Builds tree → Predicts: Approve
Tree 5: Random 10 rows + random features → Builds tree → Predicts: Approve

Final Prediction: Majority Vote = Approve (4 out of 5 trees)
Confidence: 80% (4/5 trees agreed)
```

*(Illustrative numbers - the worked example later in this document uses a real
run and reports what the code actually produces.)*

---

## The Mathematical Foundation

### Bootstrap Sampling Theory

Each bootstrap sample:
- Has the same size as the original dataset
- Contains about 63.2% unique samples (on average)
- The remaining 36.8% are duplicates

**Probability calculation**:
```
For a dataset of n samples:
P(sample selected at least once) = 1 - (1 - 1/n)^n ≈ 1 - 1/e ≈ 0.632
```

### Out-of-Bag (OOB) Scoring - the payoff of that 0.632

The ~36.8% of rows a tree never saw are called that tree's **out-of-bag** rows.
They are, for that tree, genuine unseen data. So we can score the forest for
free, with no held-out split at all:

```
For each training row i:
    collect the trees for which row i was out-of-bag
    let only those trees vote on row i
oob_score_ = accuracy (classification) or R² (regression) of those predictions
```

A row escapes scoring only if it is *in*-bag for every single tree, which happens
with probability 0.632^B. For B = 25 trees that is 0.632^25 = 1.04e-05, about 1 in
96,000 rows; at B = 100 it is 1.2e-20. So in practice every row gets scored. Pass
`oob_score=True` and read `model.oob_score_`:

```python
model = RandomForest(n_estimators=25, oob_score=True, random_state=42)
model.fit(X_train, y_train)
print(model.oob_score_)   # e.g. 0.9867 in the Quick Start above
```

OOB scoring is one of the nicest properties of bagging: cross-validation without
paying for cross-validation.

### The Split Criterion Each Tree Optimises

Feature randomness decides *which* splits are considered; the criterion decides
*which one wins*. Every node picks the split with the largest **information
gain**:

```
Gain = I(parent) - [ (n_L / n) · I(left) + (n_R / n) · I(right) ]
```

where `I` is the impurity measure named by `criterion`:

| criterion | Task | Formula | Meaning |
|-----------|------|---------|---------|
| `'gini'` | classification | I = 1 - Σ p_c² | Chance two random samples in the node have different labels |
| `'entropy'` | classification | I = -Σ p_c · log₂(p_c) | Bits needed to encode the node's labels |
| `'mse'` | regression | I = (1/n) Σ (y_i - ȳ)² | Variance of the values in the node |

`p_c` is the fraction of the node's samples that belong to class `c`.

**Worked example** - a node with 10 samples, 6 of class 1 and 4 of class 0:
```
I_gini(parent) = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48

Split it into left = [6 samples, all class 1] and right = [4 samples, all class 0]:
I(left)  = 1 - 1.0² = 0
I(right) = 1 - 1.0² = 0
Gain     = 0.48 - [ (6/10)·0 + (4/10)·0 ] = 0.48   (a perfect split)
```

This is exactly `DecisionTree._information_gain` in `_6_decision_trees.py`, and
it is also the quantity `RandomForest._accumulate_importances` sums up to build
`feature_importances_`.

### Ensemble Combination

**Classification - Majority Voting**:
```
Final Prediction = mode(predictions from all trees)

Example with 5 trees:
Tree predictions: [1, 1, 0, 1, 1]
Counts: Class 0 = 1, Class 1 = 4
Final: Class 1 (majority)
```

**Regression - Averaging**:
```
Final Prediction = mean(predictions from all trees)

Example with 5 trees:
Tree predictions: [250k, 270k, 240k, 265k, 255k]
Final: (250 + 270 + 240 + 265 + 255) / 5 = 256k
```

### Bias-Variance Tradeoff

**Single Decision Tree**:
- Low bias (can fit complex patterns)
- High variance (sensitive to data changes)
- Prone to overfitting

**Random Forest**:
- Slightly higher bias (due to averaging)
- Much lower variance (due to ensemble)
- Better generalization!

```
Variance Reduction (the idealised version):
If the B trees were INDEPENDENT, each with variance σ²:
Variance of ensemble = σ² / B
```

But bagged trees are **not** independent - they are trained on overlapping
bootstrap samples of the same data, so they make correlated errors. Writing
`ρ` for the average pairwise correlation between two trees' predictions, the
honest formula is:

```
Var(ensemble) = ρ·σ²  +  (1 - ρ)·σ² / B
                 ^              ^
                 |              +-- vanishes as B -> infinity
                 +----------------- the floor: does NOT vanish
```

Read it slowly, because this single line explains the whole algorithm:

- **The second term is what more trees buy you.** It shrinks like 1/B, so going
  from 10 to 100 trees helps, and going from 100 to 1000 barely does. *That* is
  where "diminishing returns" comes from - not from the independent formula,
  which promises unlimited improvement.
- **The first term is the floor.** No number of trees gets below `ρ·σ²`. If your
  trees are 80% correlated you keep 80% of a single tree's variance no matter how
  many you grow.
- **So to do better you must lower ρ, not raise B.** Bootstrap sampling lowers it
  a little (different rows). Random feature subsets lower it a lot (different
  evidence). That is precisely why `max_features` exists.

```
In practice:
- 1 tree:     variance σ²
- 10 trees:   ρ·σ² + 0.10·(1 - ρ)·σ²
- 100 trees:  ρ·σ² + 0.01·(1 - ρ)·σ²   <- already ~at the ρ·σ² floor
- Lower max_features -> lower ρ -> a lower floor
```

DEMO 3 in `_7_random_forests.py` measures ρ indirectly: it reports the average
fraction of test rows on which two trees of the forest disagree. Bagging over 12
features scores 0.2915; the same forest restricted to `k = 3` features scores
0.3714. More disagreement means less correlation means a lower variance floor.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class _RandomFeatureTree(DecisionTree):
    def _best_split(self, X, y):
        # Search only k randomly chosen features (this is what makes it a FOREST)

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, ...):
        # Initialize forest parameters
        
    def _bootstrap_sample(self, X, y):
        # Create random subset of training data
        
    def fit(self, X, y):
        # Build all trees in the forest
        
    def predict(self, X):
        # Combine predictions from all trees
        
    def predict_proba(self, X):
        # Get class probabilities (classification)
        
    def score(self, X, y):
        # Calculate model performance
```

### Core Methods

1. **`__init__(...)`** - Initialize forest. All ten parameters:

   | Parameter | Default | What it does |
   |-----------|---------|--------------|
   | `task` | `'classification'` | **Set this first** - `'classification'` or `'regression'`. Leaving it at the default while passing continuous targets silently builds a classifier. |
   | `n_estimators` | 100 | Number of trees |
   | `max_depth` | `None` | Depth of each tree (`None` = unlimited) |
   | `min_samples_split` | 2 | Minimum samples needed to split a node |
   | `min_samples_leaf` | 1 | Minimum samples required in a leaf |
   | `bootstrap` | `True` | Sample rows with replacement |
   | `criterion` | `'gini'` | `'gini'`/`'entropy'` for classification, `'mse'` for regression (auto-corrected for regression) |
   | `random_state` | `None` | Seed for the model's own private RNG |
   | `max_features` | `'auto'` | Features considered per split: `'auto'`, `'sqrt'`, `'log2'`, `None`, an int or a float |
   | `oob_score` | `False` | Compute `oob_score_` from the out-of-bag rows |

2. **`_bootstrap_sample(X, y, return_indices=False)`** - Private helper method
   - Creates random sample with replacement
   - Returns subset for one tree
   - Size = original dataset size
   - With `return_indices=True` also returns the drawn row indices (used for OOB)

3. **`_RandomFeatureTree._best_split(X, y)`** - Private helper method
   - Draws k random columns, delegates the gain search to `DecisionTree`
   - Maps the winning column back to its index in the full feature matrix
   - This one override is the whole difference between bagging and a forest

4. **`fit(X, y)`** - Train the forest
   - Creates n_estimators bootstrap samples
   - Builds a feature-subsampling decision tree on each sample
   - Stores all trees in the forest
   - Fills in `feature_importances_` and, if requested, `oob_score_`

5. **`predict(X)`** - Make predictions
   - Gets prediction from each tree
   - Classification: Returns majority vote
   - Regression: Returns average

6. **`predict_proba(X)`** - Get probabilities
   - Only for classification
   - Returns proportion of trees predicting each class
   - Column `j` corresponds to `model.classes_[j]`, so any label encoding works
   - Example: 70 of 100 trees predict Class 1 → probability = 0.70

7. **`score(X, y)`** - Evaluate performance
   - Classification: Returns accuracy
   - Regression: Returns R² score

### Attributes Available After `fit()`

| Attribute | Meaning |
|-----------|---------|
| `trees` | The list of fitted trees |
| `classes_` | Sorted unique training labels; column order of `predict_proba` |
| `n_classes_` | Number of classes |
| `n_features_` | Number of features seen during fit |
| `max_features_` | The resolved integer k used at every split |
| `criterion_` | The criterion actually used (after regression auto-correction) |
| `feature_importances_` | Mean decrease in impurity per feature, normalised to sum to 1 |
| `oob_score_` | Out-of-bag score, when `oob_score=True` |

---

## Step-by-Step Example

Let's walk through a complete example predicting **loan approval** based on customer features:

### The Data

```python
import numpy as np

# Features: [Age, Income ($k), Credit Score]
X_train = np.array([
    [25, 45, 650],   # Young, moderate income, fair credit
    [35, 75, 720],   # Middle-aged, good income, good credit
    [45, 95, 780],   # Older, high income, excellent credit
    [30, 50, 600],   # Young, moderate income, poor credit
    [40, 80, 750],   # Middle-aged, good income, good credit
    [50, 120, 800],  # Older, high income, excellent credit
    [28, 40, 580],   # Young, low income, poor credit
    [42, 85, 740],   # Middle-aged, good income, good credit
])

# Labels: 0=Reject, 1=Approve
y_train = np.array([0, 1, 1, 0, 1, 1, 0, 1])
```

### Training the Model

```python
model = RandomForest(n_estimators=5, max_depth=3, task='classification', random_state=42)
model.fit(X_train, y_train)
```

**What happens internally** - these are the *actual* row indices `fit()` draws
with `random_state=42`:

```
Step 1: Create 5 bootstrap samples

Bootstrap Sample 1: Rows [6, 3, 4, 6, 2, 7, 4, 4]   # row 4 three times, rows 0/1/5 unused
  -> Build Tree 1

Bootstrap Sample 2: Rows [2, 6, 2, 2, 7, 4, 3, 7]
  -> Build Tree 2

Bootstrap Sample 3: Rows [4, 1, 7, 3, 5, 5, 1, 7]
  -> Build Tree 3

Bootstrap Sample 4: Rows [3, 1, 5, 4, 3, 0, 0, 2]
  -> Build Tree 4

Bootstrap Sample 5: Rows [1, 7, 3, 3, 7, 6, 5, 5]
  -> Build Tree 5

Step 2: At every node, each tree also picks k = floor(sqrt(3)) = 1 of the 3
        features at random and can only split on that one.

Step 3: Different rows AND different features = each tree learns a different
        pattern. That is the whole point.
```

Only sample 1 is what a fresh `np.random.RandomState(42)` gives you as its first
draw. Samples 2-5 are *not* the next four draws from that generator, because the
forest and its trees share one RNG: between two bootstrap draws, Tree 1 consumes
random numbers picking its `cols` at every node it grows. For the same reason,
calling `model._bootstrap_sample(X_train, y_train, return_indices=True)` yourself
after `fit()` returns different indices again - the generator has moved on.

### Making Predictions

```python
# New customer application
X_test = np.array([[38, 70, 700]])  # Middle-aged, good income, decent credit

# Get prediction
prediction = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Prediction: {'Approved' if prediction[0] == 1 else 'Rejected'}")
print(f"Confidence: {probabilities[0][1]:.2f}")
```

### Internal Prediction Process

Here is the mechanism, sketched with a **hypothetical** 4-1 split so you can see
how voting works when the trees disagree:

```python
# Step 1: Each tree makes its prediction
Tree 1: Input [38, 70, 700] -> Predicts 1 (Approve)
Tree 2: Input [38, 70, 700] -> Predicts 1 (Approve)
Tree 3: Input [38, 70, 700] -> Predicts 0 (Reject)
Tree 4: Input [38, 70, 700] -> Predicts 1 (Approve)
Tree 5: Input [38, 70, 700] -> Predicts 1 (Approve)

# Step 2: Count votes
Approve (1): 4 votes
Reject (0): 1 vote

# Step 3: Final prediction
Final: Approve (majority)
Confidence: 4/5 = 0.80 (80%)
```

**What actually happens on this dataset**: applicant `[38, 70, 700]` gets a
unanimous **5-0** for Approve, i.e. confidence `1.00`, not `0.80`. Our 8 training
rows are perfectly separable (every approved applicant has a higher credit score
than every rejected one), so no tree has any reason to disagree. To see a real
split vote you need a genuinely borderline applicant - see the next section.

### Complete Example

```python
# Multiple test applicants
X_test = np.array([
    [38, 70, 700],   # Good candidate
    [26, 35, 550],   # Risky candidate
    [48, 110, 790],  # Excellent candidate
    [33, 60, 690],   # Borderline - right on the decision boundary
])

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

for i in range(len(predictions)):
    status = "Approved" if predictions[i] == 1 else "Rejected"
    confidence = probabilities[i][1] if predictions[i] == 1 else probabilities[i][0]
    print(f"Applicant {i+1}: {status} (confidence: {confidence:.2f})")
```

Real output:
```
Applicant 1: Approved (confidence: 1.00)
Applicant 2: Rejected (confidence: 1.00)
Applicant 3: Approved (confidence: 1.00)
Applicant 4: Rejected (confidence: 0.80)
```

Only the borderline applicant produces a divided forest: 4 trees vote Reject and
1 votes Approve, giving the 0.80 confidence. That is the forest telling you it is
genuinely unsure - information a single decision tree could never give you.

Note also the granularity: with `n_estimators=5` the only confidences that can
ever appear are 0.2, 0.4, 0.6, 0.8 and 1.0. If you want finer probabilities, grow
more trees.

---

## Real-World Applications

### 1. **Credit Risk Assessment**
Predicting loan defaults:
- Input: Income, credit score, debt ratios
- Output: Risk level
- Benefit: Banks reduce losses from bad loans

### 2. **Medical Diagnosis**
Disease prediction and classification:
- Input: Symptoms, test results, medical history
- Output: Disease probability
- Example: Cancer detection, diabetes prediction

### 3. **Fraud Detection**
Identifying fraudulent transactions:
- Input: Transaction amount, location, time, patterns
- Output: Fraud probability
- Example: Credit card fraud prevention

### 4. **Customer Churn Prediction**
Predicting which customers will leave:
- Input: Usage patterns, demographics, complaints
- Output: Churn probability
- Benefit: Targeted retention campaigns

### 5. **Stock Market Prediction**
Forecasting price movements:
- Input: Historical prices, volume, indicators
- Output: Price direction
- Example: Algorithmic trading systems

### 6. **Recommendation Systems**
Product and content recommendations:
- Input: User behavior, preferences, history
- Output: Recommendation scores
- Example: Netflix, Amazon recommendations

### 7. **Image Classification**
Object recognition in images:
- Input: Pixel values, image features
- Output: Object categories
- Example: Medical image analysis, quality control

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Bootstrap Sampling

```python
def _bootstrap_sample(self, X, y, return_indices=False):
    n_samples = len(X)
    
    if self.bootstrap:
        # Sample with replacement, using the model's PRIVATE generator
        indices = self._rng.choice(n_samples, size=n_samples, replace=True)
    else:
        # Use all samples
        indices = np.arange(n_samples)
    
    if return_indices:
        return X[indices], y[indices], indices
    return X[indices], y[indices]
```

**How it works**:
```python
# Example with 5 samples (inside the class, so self._rng is the model's own RNG)
indices = self._rng.choice(5, size=5, replace=True)
# Could return: [0, 2, 2, 4, 1]
# Sample 2 appears twice, sample 3 is not included (it is out-of-bag)
```

Note `self._rng`, not `np.random`. The model owns a `np.random.RandomState`
seeded from `random_state`, so setting a seed on the forest never disturbs the
random numbers the rest of your program is drawing.

### 2. Random Feature Selection at Each Split

This is the shortest method in the file and the most important one. It is the
only difference between bagging and a Random Forest:

```python
class _RandomFeatureTree(DecisionTree):
    def _best_split(self, X, y):
        n_features = X.shape[1]
        k = min(self.max_features, n_features)

        # Draw k candidate features WITHOUT replacement for THIS node
        cols = self._rng.choice(n_features, size=k, replace=False)

        # Let the parent class do the actual gain computation on the subset
        split = super()._best_split(X[:, cols], y)

        if split is None and k < n_features:
            # No valid split exists among the k sampled columns. Two different
            # situations reach here, and sklearn treats them differently:
            #   ...  (the two cases are spelled out in the table below)
            return super()._best_split(X, y)

        if split is not None:
            # Map the subset column index back to the original feature index
            split['feature_index'] = int(cols[split['feature_index']])

        return split
```

**How it works**:
```python
# 6 features, k = 2. At one node:
cols = [4, 1]                  # only features 4 and 1 are candidates
# super()._best_split sees a 2-column matrix and returns feature_index = 0,
# meaning "the first of my two columns" - which is the ORIGINAL feature 4.
# So we remap:  split['feature_index'] = cols[0]  ->  4
```

Two consequences worth noticing:
- `cols` is redrawn at **every node**, not once per tree, so a single tree still
  gets to use every feature - just never all of them at the same decision.
- Because this randomness is independent of the data, the trees differ even when
  `bootstrap=False`. Feature subsampling alone is enough to build a forest.

**About that fallback.** `DecisionTree._best_split` returns `None` whenever it
finds no usable split, and that happens for two different reasons - which
scikit-learn handles two different ways:

| Why no split was found | scikit-learn | Us |
|------------------------|--------------|-----|
| every sampled column is constant, so there is no threshold to try | keeps drawing further features until it finds a usable one | full scan of all `p` (near-equivalent) |
| a sampled column varies, but every midpoint would leave a child below `min_samples_leaf` | stops and makes this node a leaf | full scan of all `p` (a real, if small, divergence) |

So the comment's opening line - "No valid split exists among the k sampled
columns" - is the accurate trigger: it is *not* only the all-constant case. (The
fence above elides the rest of that nine-line comment; the full text is in
`_7_random_forests.py`.) The second row is a deliberate simplification: one
branch is easier to read than two. It fires rarely, and
usually deep in the tree where few samples remain - but it is not always
harmless. With `max_features=1` and a nearly constant column it can fire at the
**root**: on a 12-row set whose feature 0 is `[0]*11 + [1]` with
`min_samples_leaf=2`, drawing `cols=[0]` blocks the only midpoint, so
scikit-learn would return a single-leaf tree while we widen the search and grow
a normal one on feature 1.

### 3. Forest Building

```python
def fit(self, X, y):
    ...
    self.max_features_ = self._resolve_max_features(n_features)   # k

    self.trees = []
    for i in range(self.n_estimators):
        # Create bootstrap sample
        X_sample, y_sample, indices = self._bootstrap_sample(X, y, return_indices=True)
        
        # Build a tree that subsamples features at every split
        tree = _RandomFeatureTree(max_features=self.max_features_, rng=self._rng,
                                  max_depth=self.max_depth, ...)
        tree.fit(X_sample, y_sample)
        
        # Store tree
        self.trees.append(tree)

        # Accumulate this tree's impurity decrease into feature_importances_
        self._accumulate_importances(tree, X_sample, y_sample)
```

**Key points**:
- Each tree is independent
- Trees can be trained in parallel (not implemented here)
- Each tree sees different rows *and* different features
- `indices` is kept so the rows this tree did **not** see can be used for OOB scoring

### 4. Ensemble Prediction (Classification)

```python
def predict(self, X):
    # Get all tree predictions
    tree_predictions = []
    for tree in self.trees:
        tree_pred = tree.predict(X)
        tree_predictions.append(tree_pred)
    
    tree_predictions = np.array(tree_predictions)
    
    # Majority voting
    predictions = []
    for i in range(len(X)):
        sample_preds = tree_predictions[:, i]
        unique_preds, counts = np.unique(sample_preds, return_counts=True)
        majority_vote = unique_preds[np.argmax(counts)]
        predictions.append(majority_vote)
    
    return np.array(predictions)
```

**Example**:
```python
# 3 trees, 2 samples
tree_predictions = [
    [1, 0],  # Tree 1
    [1, 0],  # Tree 2
    [0, 0],  # Tree 3
]

# For sample 0: votes are [1, 1, 0] → majority is 1
# For sample 1: votes are [0, 0, 0] → majority is 0
# Result: [1, 0]
```

### 5. Probability Estimation

```python
def predict_proba(self, X):
    probabilities = []
    for i in range(len(X)):
        sample_preds = tree_predictions[:, i]
        
        class_probs = []
        for class_label in self.classes_:
            # Proportion of trees predicting this class
            prob = np.mean(sample_preds == class_label)
            class_probs.append(prob)
        
        probabilities.append(class_probs)
    
    return np.array(probabilities)
```

**Example**:
```python
# 5 trees predict: [1, 1, 0, 1, 1]
# Class 0: 1/5 = 0.20
# Class 1: 4/5 = 0.80
# Probabilities: [0.20, 0.80]
```

The loop runs over `self.classes_` - the sorted unique labels stored during
`fit()` - rather than over `range(self.n_classes_)`. That matters: iterating over
`0, 1, ..., n_classes-1` only works when your labels happen to be exactly those
integers. Comparing against the real labels means string labels (`'cat'`,
`'dog'`), `{1, 2}` or `{-1, +1}` all work, and column `j` always means
`classes_[j]`:

```python
model.fit(X, np.array(['cat', 'cat', 'cat', 'dog', 'dog', 'dog']))
model.classes_          # array(['cat', 'dog'], dtype='<U3')
model.predict_proba(X)  # column 0 = P(cat), column 1 = P(dog)
```

### 6. Feature Importance

```python
def _accumulate_importances(self, tree, X_sample, y_sample):
    stack = [(tree.tree, X_sample, y_sample)]
    while stack:
        node, X_node, y_node = stack.pop()
        if node['type'] == 'leaf':
            continue
        ...
        parent_impurity = tree._calculate_impurity(y_node)
        child_impurity = ((n_left / n) * tree._calculate_impurity(y_node[left_mask])
                          + (n_right / n) * tree._calculate_impurity(y_node[right_mask]))
        self.feature_importances_[feature_index] += n * (parent_impurity - child_impurity)
        ...
```

This is the **Mean Decrease in Impurity** (MDI) measure, and it is exactly the
information-gain formula from
[The Split Criterion](#the-split-criterion-each-tree-optimises), weighted by how
many samples reached the node. Sum it over every internal node of every tree,
then normalise so the values add to 1:

```python
model.feature_importances_
# e.g. for the house-price example in USAGE EXAMPLE 3:
# Size = 0.6742, Age = 0.1235, Bedrooms = 0.2023
```

Read these as "how much of the total impurity reduction did this feature
deliver". A feature that never gets chosen scores exactly 0.

---

## Model Evaluation

### For Classification

#### 1. Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Example**:
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

correct = 4
total = 5
accuracy = 4/5 = 0.80 (80%)
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

#### R² Score
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

---

## Advantages and Limitations

### Advantages ✅

1. **High Accuracy**
   - Often outperforms single decision trees
   - Competitive with other top algorithms
   - Good default choice for many problems

2. **Handles Overfitting Well**
   - Bootstrap sampling reduces variance
   - Ensemble averaging smooths predictions
   - Naturally regularized

3. **Works with Many Features**
   - Handles high-dimensional data
   - No need for feature scaling
   - Robust to irrelevant features

4. **Handles Missing Values** *(production libraries only - NOT this implementation)*
   - Real Random Forests deal with NaNs via surrogate splits or by sending
     missing values down whichever branch reduces loss most
   - **This from-scratch version does not.** `NaN <= threshold` evaluates to
     `False` in NumPy, so every missing value is silently routed down the right
     branch and you get a plausible-looking but meaningless prediction. Impute
     before fitting - see [Simplifications](#simplifications-vs-canonical-random-forest)

5. **No Feature Scaling Needed**
   - Works with raw features: splits are thresholds, so units do not matter
   - No need to scale or normalize
   - Handles categorical features (with encoding)
   - (You still need to impute missing values - see point 4)

6. **Provides Feature Importance**
   - `model.feature_importances_` ranks features by mean decrease in impurity
   - Shows which features matter most
   - Helps with feature selection
   - Aids in model interpretation

7. **Free Validation Estimate**
   - `oob_score=True` gives you `model.oob_score_` without a held-out split
   - Useful when data is scarce and you cannot spare a validation set

8. **Versatile**
   - Works for classification and regression
   - Handles multi-class problems
   - Works with various data types

### Limitations ❌

1. **Slower Prediction**
   - Must query every tree
   - Not ideal for real-time applications
   - Tradeoff: accuracy vs speed

2. **Memory Intensive**
   - Stores many trees
   - Each tree can be large
   - Not suitable for very limited memory

3. **Less Interpretable**
   - Hard to visualize 100 trees
   - Cannot easily trace decisions
   - "Black box" nature

4. **Longer Training Time**
   - Must train many trees
   - Slower than single tree
   - But parallelizable!

5. **Diminishing Returns**
   - More trees = more training time
   - Limited accuracy improvement after ~100 trees
   - Need to balance trees vs time

### When to Use Random Forest

**Good Use Cases**:
- ✅ Medium to large datasets
- ✅ Many features (high dimensions)
- ✅ Need high accuracy
- ✅ Have time for training
- ✅ Want robust model with minimal tuning
- ✅ Mixed data types

**Bad Use Cases**:
- ❌ Very small datasets (< 100 samples)
- ❌ Need real-time predictions (milliseconds)
- ❌ Require full interpretability
- ❌ Very limited memory
- ❌ Linear relationships (use regression instead)

---

## Simplifications vs. Canonical Random Forest

This implementation is written for learning, so a few things a production library
does are deliberately left out. Here is the honest list.

### 1. Probabilities are vote fractions, not averaged leaf distributions

**What canonical does**: scikit-learn's `predict_proba` averages each tree's
*leaf class distribution*. If a leaf holds 7 samples of class 1 and 3 of class 0,
that tree contributes `[0.3, 0.7]`, not a hard vote.

**What we do**: each tree casts one hard vote and we report the fraction:

```
P(class = c | x) = (# trees voting c) / B
```

**Why**: this is what Breiman's 2001 paper actually specifies, and it is the
version the voting sections of this document teach.

**Consequence**: probabilities are quantised to multiples of `1/n_estimators`
and are more confident (more 0.00 and 1.00 values) than scikit-learn's. Measured
on iris with 30 trees, the mean absolute difference against
`RandomForestClassifier.predict_proba` is about 0.018, while the hard predictions
agree 100% of the time. Use more trees if you need finer probabilities.

### 2. No missing-value handling

**What canonical does**: surrogate splits (CART) or a learned default direction
(as in XGBoost/LightGBM) route NaNs sensibly.

**What we do**: nothing. `DecisionTree._predict_single` tests
`x[feature_index] <= threshold`, and in NumPy `np.nan <= t` is `False`, so every
missing value silently goes right. Fitting with NaNs raises no error either.

**Consequence**: impute before you fit. A NaN will not crash anything, which is
worse than crashing - it produces a confident, meaningless answer.

### 3. No parallelism

Trees are grown one after another in a Python loop. Real implementations fit them
across cores (`n_jobs`), which is where most of scikit-learn's ~10x speed
advantage on these examples comes from. Random Forests are embarrassingly
parallel by construction; we simply do not exploit it, for readability.

### 4. Exhaustive threshold search

Each node tries **every** unique value of each candidate feature as a threshold.
Production libraries bin features into ~256 histogram buckets first. Ours is
exact but O(n × unique values) per feature - fine for the datasets in this
repository, slow on tens of thousands of rows.

### 5. No class weights, no `min_impurity_decrease`, no cost-complexity pruning

These are all real knobs on `sklearn.ensemble.RandomForestClassifier` that this
implementation does not expose. Depth and leaf-size limits are the only
regularisation available here.

### 6. A dead-end feature draw widens the search instead of ending the node

When no valid split exists among the k sampled columns we rescan all `p`
features. scikit-learn does the same when every drawn column was constant, but
when a column varies and only `min_samples_leaf` blocks its thresholds it stops
and makes a leaf - so we can grow a subtree, and with `max_features=1` even a
whole tree, where scikit-learn would not. The worked example is in
[Understanding the Code](#understanding-the-code), section 2.

---

## Choosing Hyperparameters

### Number of Trees (n_estimators)

```
Small (10-50):
  Pros: Faster training and prediction
  Cons: May underfit, higher variance
  
Medium (50-200):
  Pros: Good balance
  Cons: None - usually optimal
  
Large (200+):
  Pros: Maximum accuracy, lowest variance
  Cons: Slower, diminishing returns
```

**Rule of thumb**: Start with 100, increase if needed

### Tree Depth (max_depth)

```
Shallow (3-5):
  Pros: Fast, less overfitting
  Cons: May underfit
  
Medium (10-15):
  Pros: Good balance
  Cons: None - usually good default
  
Deep (None/unlimited):
  Pros: Maximum flexibility
  Cons: Slower, can overfit
```

**Rule of thumb**: Start with 10-15, adjust based on performance

### Bootstrap Sampling

```
bootstrap=True (default):
  - Standard random forest
  - Each tree sees ~63% unique samples
  - Required for oob_score
  - Recommended for most cases

bootstrap=False:
  - All trees see all rows
  - The ONLY remaining randomness is max_features
  - With bootstrap=False AND max_features=None the forest collapses:
    every tree is byte-identical and you have paid n_estimators times
    the training cost for one decision tree
  - Also disables oob_score (there are no out-of-bag rows)
```

Measured: `RandomForest(n_estimators=20, max_depth=4, bootstrap=False, random_state=3)`
produces 3 distinct root splits among its 20 trees on a 4-feature dataset; add
`max_features=None` and all 20 root splits become identical.

### Features per Split (max_features)

```
max_features='sqrt'  (k = floor(sqrt(p)))
  - Breiman's recommendation for classification; sklearn's classifier default
  - Strongest decorrelation, fastest training
  - Best default for wide data with correlated features

max_features='auto' (our default)
  - sqrt(p) for classification, all p for regression
  - Matches scikit-learn's two defaults exactly
  - On regression that means k = p, i.e. bagging: set 1/3 or 'sqrt' explicitly
    if you want column subsampling on a regression task

max_features=1/3     (k = floor(p/3))
  - Breiman's recommendation for regression
  - Worth trying whenever p is large

max_features=None    (k = p)
  - This is BAGGING, not a Random Forest
  - Use it only as the baseline you compare against
```

**Rule of thumb**: leave it at `'auto'`. If your model is overfitting or your
trees look suspiciously alike, lower k; if it is underfitting badly, raise it.

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset (124 training rows x 13 features - this whole script runs in
# about 2 seconds; a 30-feature dataset like breast_cancer would take minutes,
# because this from-scratch tree tries every unique value as a threshold)
data = load_wine()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {data.target_names}")

# Create and train Random Forest
model = RandomForest(
    n_estimators=20,
    max_depth=6,
    task='classification',
    max_features='sqrt',   # k = floor(sqrt(13)) = 3 features per split
    oob_score=True,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate
print(f"\nTrain Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test  Accuracy: {model.score(X_test, y_test):.4f}")
print(f"OOB   Accuracy: {model.oob_score_:.4f}")

# Detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Show some predictions
print("\nSample Predictions:")
for i in range(5):
    true_label = data.target_names[y_test[i]]
    pred_label = data.target_names[y_pred[i]]
    confidence = y_proba[i][y_pred[i]]
    print(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.2f}")

# Which measurements actually drove the decisions?
print("\nTop 5 features by importance:")
for rank, f_idx in enumerate(np.argsort(model.feature_importances_)[::-1][:5], start=1):
    print(f"  {rank}. {data.feature_names[f_idx]:<25} {model.feature_importances_[f_idx]:.4f}")
```

Expected output:
```
Training samples: 124
Test samples: 54
Features: 13
Classes: ['class_0' 'class_1' 'class_2']

Train Accuracy: 1.0000
Test  Accuracy: 0.9815
OOB   Accuracy: 0.9516

Classification Report:
              precision    recall  f1-score   support

     class_0       1.00      1.00      1.00        19
     class_1       1.00      0.95      0.98        21
     class_2       0.93      1.00      0.97        14

    accuracy                           0.98        54
   macro avg       0.98      0.98      0.98        54
weighted avg       0.98      0.98      0.98        54


Sample Predictions:
  True: class_0, Predicted: class_0, Confidence: 1.00
  True: class_0, Predicted: class_0, Confidence: 1.00
  True: class_2, Predicted: class_2, Confidence: 0.90
  True: class_0, Predicted: class_0, Confidence: 1.00
  True: class_1, Predicted: class_1, Confidence: 0.95

Top 5 features by importance:
  1. color_intensity           0.2127
  2. flavanoids                0.1705
  3. proline                   0.1056
  4. hue                       0.1047
  5. alcohol                   0.0983
```

Two things worth noticing. First, the OOB accuracy (0.9516) is far more
informative than the training accuracy (1.0000, which just says the trees
memorised their bootstrap samples), and it comes for free. It sits slightly
*below* the test accuracy here (0.9815) - that is normal and expected: each row
is judged by only the ~37% of trees that did not see it, so OOB behaves like a
smaller forest and is a mildly pessimistic estimate. Second, the importances rank
`color_intensity` and `flavanoids` at the top, which are exactly the chemical
measurements you would use to tell these three cultivars apart.

---

## Key Concepts to Remember

### 1. **Ensemble Learning**
Random Forest uses the power of many models (trees) to make better predictions than any single model.

### 2. **Bootstrap Sampling**
Each tree trains on a random subset of data (random ROWS), creating diversity and reducing overfitting. About 63.2% of rows land in each sample; the rest are out-of-bag.

### 3. **Random Feature Subsets**
Each split only gets to consider k random features (random COLUMNS). This is the difference between a Random Forest and plain bagging, and it is the reason the trees stop making the same mistakes together.

### 4. **Majority Voting**
Classification combines predictions democratically - each tree gets one vote.

### 5. **Bias-Variance Tradeoff**
Random Forest reduces variance (overfitting) while maintaining low bias (good fit). The variance floor is `rho * sigma^2`, so lowering the between-tree correlation `rho` matters more than adding trees.

### 6. **Hyperparameter Tuning**
- More trees = better (up to a point set by rho)
- Tree depth controls complexity
- Bootstrap and max_features both add diversity

### 7. **Computational Complexity**
- Training: O(n_estimators × tree_cost), and tree_cost scales with k, not p -
  which is why `max_features='sqrt'` also makes training several times faster
- Prediction: O(n_estimators × tree_depth)
- Memory: O(n_estimators × tree_size)

---

## Conclusion

Random Forests are one of the most powerful and practical machine learning algorithms! By understanding:

- How bootstrap sampling creates diverse trees
- How random feature subsets decorrelate them - the "Random" in Random Forest
- How ensemble voting combines predictions
- Why the variance floor `rho * sigma^2` is what really limits an ensemble
- How hyperparameters control performance
- When to use (and not use) random forests

You've gained a versatile tool that works well across many different problems! 🌲🌲🌲

**When to Use Random Forest**:
- ✅ Need high accuracy with minimal tuning
- ✅ Have sufficient training data
- ✅ Can afford slightly longer training time
- ✅ Want robust predictions
- ✅ Don't need real-time predictions

**When to Use Something Else**:
- ❌ Need instant predictions → Use simpler models
- ❌ Need full interpretability → Use single decision tree
- ❌ Very small dataset → Use simpler models
- ❌ Linear relationships → Use linear regression

**Next Steps**:
- Run `python _7_random_forests.py` and read DEMO 3 - it measures the tree
  disagreement that `max_features` buys you
- Try Random Forest on your own datasets
- Experiment with different n_estimators, max_depth and max_features
- Compare with single Decision Trees, and with `max_features=None` (bagging)
- Inspect `model.feature_importances_` to see which inputs carry the signal
- Use `oob_score=True` instead of a validation split when data is scarce
- Explore Gradient Boosting as an alternative
- Study ensemble methods in depth

Happy coding! 🌲🤖
