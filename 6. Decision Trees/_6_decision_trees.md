# Decision Trees from Scratch: A Comprehensive Guide

Welcome to the world of Decision Trees! 🌳 In this comprehensive guide, we'll explore one of the most intuitive and powerful machine learning algorithms. Think of it as a flowchart that makes decisions!

## Table of Contents
0. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
1. [What are Decision Trees?](#what-are-decision-trees)
2. [How Decision Trees Work](#how-decision-trees-work)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Advantages and Limitations](#advantages-and-limitations)
11. [Preventing Overfitting](#preventing-overfitting)
12. [Simplifications vs. Canonical CART](#simplifications-vs-canonical-cart)
13. [Complete Usage Example](#complete-usage-example)
14. [Key Concepts to Remember](#key-concepts-to-remember)
15. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Decision Tree from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _6_decision_trees.py  (the __main__ block runs this)
# Or copy the DecisionTree class from _6_decision_trees.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the DecisionTree class here (from _6_decision_trees.py) ----
# class DecisionTree: ...

np.random.seed(42)

# ------ CLASSIFICATION: three OVERLAPPING Gaussian blobs ------
centers = np.array([[-1.2, -1.2], [0.0, 1.4], [1.2, -1.0]])
X = np.vstack([c + np.random.randn(100, 2) * 1.1 for c in centers])
y = np.repeat([0, 1, 2], 100)

# Shuffle before slicing: the rows above are grouped by class, so an
# unshuffled slice would put whole classes in the test set only.
idx = np.random.permutation(300)
X, y = X[idx], y[idx]
X_train, X_test = X[:220], X[220:]
y_train, y_test = y[:220], y[220:]

clf = DecisionTree(max_depth=3, criterion='gini', task='classification')
clf.fit(X_train, y_train)

print(f"Train accuracy : {clf.score(X_train, y_train):.2%}")
print(f"Test  accuracy : {clf.score(X_test,  y_test):.2%}")
print(f"Depth / leaves : {clf.get_depth()} / {clf.get_n_leaves()}")
print(f"Importances    : {np.round(clf.feature_importances_, 4)}")

preds = clf.predict(X_test)
proba = clf.predict_proba(X_test)
for i in range(3):
    print(f"  true={y_test[i]}  pred={preds[i]}  P(pred)={proba[i, preds[i]]:.2f}")

# ------ REGRESSION: y = x^2 + noise ------
X_r = np.linspace(-3, 3, 200).reshape(-1, 1)
y_r = X_r.ravel() ** 2 + np.random.randn(200) * 0.3

# Shuffle: this data is generated in SORTED x order and a tree cannot
# extrapolate, so an unshuffled split would put every test x out of range.
idx = np.random.permutation(200)
X_r, y_r = X_r[idx], y_r[idx]

reg = DecisionTree(max_depth=4, criterion='mse', task='regression')
reg.fit(X_r[:150], y_r[:150])
print(f"\nTrain R2 : {reg.score(X_r[:150], y_r[:150]):.4f}")
print(f"Test  R2 : {reg.score(X_r[150:], y_r[150:]):.4f}")

# ------ Depth controls overfitting ------
print("\nmax_depth  train    test     leaves")
for depth in [1, 2, 3, 5, 8, None]:
    m = DecisionTree(max_depth=depth).fit(X_train, y_train)
    print(f"{str(depth):>9}  {m.score(X_train, y_train):.4f}  "
          f"{m.score(X_test, y_test):.4f}   {m.get_n_leaves()}")
```

Expected output:
```
Train accuracy : 83.64%
Test  accuracy : 81.25%
Depth / leaves : 3 / 8
Importances    : [0.4763 0.5237]
  true=0  pred=0  P(pred)=0.92
  true=2  pred=2  P(pred)=0.88
  true=0  pred=0  P(pred)=0.92

Train R2 : 0.9613
Test  R2 : 0.9536

max_depth  train    test     leaves
        1  0.5955  0.6000   2
        2  0.8227  0.8375   4
        3  0.8364  0.8125   8
        5  0.8955  0.8125   23
        8  0.9727  0.8000   44
     None  1.0000  0.7625   50
```

Read that last table before anything else - it is the whole story of decision trees in six lines. Training accuracy climbs monotonically to a perfect `1.0000`, while **test** accuracy peaks at `max_depth=2` and then *falls*. Everything past the peak is the tree memorising noise. The rest of this guide explains why that happens and what to do about it.

---

## What are Decision Trees?

Decision Trees are **hierarchical, tree-structured models** that make predictions by learning simple decision rules from data. They split data recursively based on feature values, creating a tree of decisions.

**Real-world analogy**: 
Imagine a doctor diagnosing a patient. They ask: "Do you have a fever?" If yes, ask "Is it above 102°F?". Each answer leads to more questions until reaching a diagnosis. That's exactly how a decision tree works!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Non-parametric, Tree-based |
| **Learning Style** | Recursive partitioning |
| **Tasks** | Classification and Regression |
| **Output** | Tree structure with decision rules |
| **Interpretability** | Highly interpretable (white-box) |

### The Core Idea

```
"Make predictions by asking a series of yes/no questions"
```

A decision tree:
1. **Starts** with all training data at the root
2. **Asks** questions (splits) based on features
3. **Divides** data into subsets at each node
4. **Repeats** until reaching pure or small groups (leaves)
5. **Predicts** based on the majority class/average value in each leaf

---

## How Decision Trees Work

### The Algorithm in 5 Steps

```
Step 1: Start with all training data at root
         ↓
Step 2: Find best feature and threshold to split on
        (Maximize information gain)
         ↓
Step 3: Split data into left and right child nodes
         ↓
Step 4: Recursively repeat Steps 2-3 for each child
        (Until stopping criteria met)
         ↓
Step 5: Assign prediction value to each leaf node
        (Most common class or average value)
```

### Visual Example

```
Training Data:
Age  Income  Buy_Computer
25   30k     No
45   80k     Yes
35   50k     Yes
20   25k     No
50   90k     Yes

Building the Tree:

The algorithm scores every candidate question. Here BOTH "Age <= 30?" and
"Income <= 40k?" split these five rows perfectly (each scores IG = 0.48),
so they tie - and the tie goes to the lower feature index, which is Age.

                   [Root: All Data]
                          |
                     Age <= 30?
                    /           \
                  Yes           No
                  /               \
            [Buy = No]        [Buy = Yes]
             (Leaf)              (Leaf)

Making Prediction for [Age=28, Income=35k]:
  1. Age <= 30? → Yes (go left)
  2. Reached leaf → Predict "No"

When one question is NOT enough, the tree keeps asking. Any child that is
still mixed gets its own split, and the structure grows another level:

                   [Root: All Data]
                          |
                     Age <= 30?
                    /           \
            [Income <= 40k?]  [Buy = Yes]
              /      \           (Leaf)
        [Buy = No] [Buy = Yes]
         (Leaf)      (Leaf)
```

### Why Trees?

**Visual Decision Boundaries**:
```
Linear Model:         Decision Tree:
    ●●●●●●                 ●●●●|●●
    ------                 ----|--
    ■■■■■■                 ■■■■|■■
  (Straight line)      (Rectangle regions)
```

Decision trees create **rectangular decision boundaries** by splitting on feature values, allowing them to capture complex, non-linear patterns.

---

## The Mathematical Foundation

### Impurity Measures

Decision trees split data to **reduce impurity** (make subsets more homogeneous). Three common measures:

#### 1. Gini Impurity (Classification)

Measures the probability of incorrectly classifying a randomly chosen element:

```
Gini = 1 - Σ(p_i²)

where p_i = proportion of class i
```

**Properties**:
- Gini = 0: Pure node (all samples same class)
- Gini = 0.5: Maximum impurity for binary (50-50 split)
- Range: [0, 0.5] for binary, [0, 1-1/n] for n classes

**Example**:
```python
# Node with 10 samples: 7 class A, 3 class B
p_A = 7/10 = 0.7
p_B = 3/10 = 0.3

Gini = 1 - (0.7² + 0.3²)
     = 1 - (0.49 + 0.09)
     = 1 - 0.58
     = 0.42

# Pure node: 10 samples, all class A
p_A = 10/10 = 1.0
Gini = 1 - 1.0² = 0 (perfect!)
```

#### 2. Entropy (Classification)

Measures the average amount of information (in bits) needed to identify the class:

```
Entropy = -Σ(p_i × log₂(p_i))

where p_i = proportion of class i
```

**Properties**:
- Entropy = 0: Pure node
- Entropy = 1: Maximum impurity for binary (50-50 split)
- Range: [0, log₂(n)] for n classes

**Example**:
```python
# Node with 10 samples: 7 class A, 3 class B
p_A = 0.7, p_B = 0.3

Entropy = -(0.7 × log₂(0.7) + 0.3 × log₂(0.3))
        = -(0.7 × -0.515 + 0.3 × -1.737)
        = -(-0.360 + -0.521)
        = 0.881

# Pure node: all class A
Entropy = -(1.0 × log₂(1.0)) = 0 (perfect!)
```

**Why do Gini and Entropy almost always pick the same split?**

Because Gini **is** entropy, to first order. Write entropy in nats and expand
ln(p) around p = 1 using the first-order Taylor approximation ln(p) ≈ p - 1:

```
H = -Σ(p_i × ln(p_i))
  ≈ -Σ(p_i × (p_i - 1))     [substituting ln(p_i) ≈ p_i - 1]
  = Σ(p_i) - Σ(p_i²)
  = 1 - Σ(p_i²)
  = Gini
```

So Gini is the first-order Taylor approximation of entropy measured in nats.
(Note where the expansion happens: p = 1 is the point at which the LOGARITHM is
linearised, not a claim about which class balance the approximation is best at.)
The two curves have the same shape - zero at the pure ends, maximal at the
uniform middle - and differ only in scale and curvature. That is why they rank
candidate splits almost identically, and why `criterion='gini'` is the default:
it gets the same answer without evaluating a logarithm.

"Almost" is doing real work in that sentence. When two splits are nearly tied,
the small curvature difference can flip the winner, and because the tree is
built greedily that one flip changes every node below it. `USAGE EXAMPLE 5` in
the `.py` shows exactly this: on the wine dataset Gini and Entropy both reach
100% training accuracy but land 9 percentage points apart on test accuracy.
That is not a defect in either criterion - it is the **high variance** of a
single deep tree, and it is the reason Random Forests exist.

#### 3. Mean Squared Error (Regression)

Measures the variance of values in a node:

```
MSE = (1/n) × Σ(y_i - ȳ)²

where ȳ = mean of y values
```

**Example**:
```python
# Node with values: [100, 120, 110, 130]
mean = (100 + 120 + 110 + 130) / 4 = 115

MSE = ((100-115)² + (120-115)² + (110-115)² + (130-115)²) / 4
    = (225 + 25 + 25 + 225) / 4
    = 500 / 4
    = 125
```

**Why is MSE the right impurity for regression?**

Look at the formula again: `MSE = (1/n) × Σ(y_i - ȳ)²` is exactly the
**variance** of the node's targets. So "reduce impurity" and "reduce variance"
are the same instruction, and information gain for regression is usually called
**variance reduction**.

This is not an arbitrary choice - it falls out of what a leaf predicts. A leaf
answers with a single number `c`, and the value of `c` that minimises the
squared error inside that leaf is the mean:

```
minimise Σ(y_i - c)²   over c
d/dc:  -2 × Σ(y_i - c) = 0   ->   c = mean(y)
and the resulting error per sample is exactly the variance
```

So MSE measures precisely the error the leaf will still make after doing the
best it can. Choosing the split that minimises the weighted child MSE is
choosing the split that leaves the least unexplained error. Classification uses
Gini or entropy for the same reason - they measure the error a *majority-vote*
leaf will still make.

### Information Gain

The reduction in impurity from a split:

```
Information Gain = Impurity(parent) - Weighted Average(Impurity(children))

IG = I(parent) - [n_left/n × I(left) + n_right/n × I(right)]
```

**Goal**: Choose split that **maximizes information gain** (biggest reduction in impurity).

**Example**:
```python
Parent node: 10 samples (6 A, 4 B)
Gini(parent) = 1 - (0.6² + 0.4²) = 0.48

Split on Feature X <= 5:
  Left:  4 samples (4 A, 0 B) → Gini = 0 (pure!)
  Right: 6 samples (2 A, 4 B) → Gini = 1 - (0.33² + 0.67²) = 0.44

Information Gain = 0.48 - [4/10 × 0 + 6/10 × 0.44]
                 = 0.48 - 0.264
                 = 0.216

Good split! Reduced impurity significantly.
```

### Splitting Algorithm

For each node:

```
1. For each feature:
   a. For each possible threshold:
      - Split data into left (≤ threshold) and right (> threshold)
      - Calculate information gain
   b. Keep track of best split

2. Choose split with highest information gain

3. Create left and right child nodes

4. Recursively apply to children
```

#### Which thresholds are "possible"?

A continuous feature has infinitely many thresholds, so this loop needs a
finite candidate list. The key observation is that the *partition* only changes
when the threshold crosses a data value. If a feature takes the sorted values
`20, 22, 25, 30`, then every threshold in `[25, 30)` produces the identical
left/right grouping. There are therefore only **k - 1** genuinely different
splits for a feature with k distinct values - and a constant feature
(k = 1) offers no split at all.

Which representative do we pick from each interval? CART, scikit-learn, and this
implementation all use the **midpoint** between consecutive unique values:

```
sorted unique values:   20      22      25      30
midpoint candidates:        21     23.5    27.5
```

Using a raw data value (`<= 25`) instead of a midpoint (`<= 27.5`) gives the
**same partition of the training set** but parks the boundary directly on an
observed point. Any future sample landing at 26 or 27 - inside the empty gap the
training data never occupied - then gets pushed to the right child, when the
evidence says the boundary should sit halfway. The midpoint splits the
difference, which is the most defensible guess in a region you have no data for.

This is why the choice is invisible on training scores but visible on test
scores: fitting the same trees either way, training R² is identical to six
decimal places, while individual test predictions can differ substantially.

```
Complexity per node: O(n_features × n × log n)
  - sorting/uniquing each feature column:      O(n log n)
  - evaluating each of its ~n candidate splits: O(n) each
Total for a balanced tree of depth d: O(d × n_features × n log n)
```

That last line explains why this readable implementation takes seconds where
scikit-learn's compiled Cython takes milliseconds: it re-scans and re-partitions
the data with NumPy at every candidate threshold, instead of sorting once and
sweeping the split point incrementally. Clarity is the deliberate trade.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, criterion='gini', task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.task = task
        self.tree = None
        self.n_features = None
        self.n_classes = None
        self.classes_ = None    # sorted unique labels; predict_proba's columns
```

### Core Methods

1. **`__init__(...)`** - Initialize model
   - max_depth: Maximum tree depth (None = unlimited)
   - min_samples_split: Min samples to split a node
   - min_samples_leaf: Min samples in leaf
   - criterion: 'gini', 'entropy', or 'mse'
   - task: 'classification' or 'regression'

2. **`_gini_impurity(y)`** - Calculate Gini impurity
   - Measures node impurity for classification
   - Returns value between 0 (pure) and 1 - 1/n_classes (maximally mixed);
     that ceiling is 0.5 for binary, 0.667 for 3 classes

3. **`_entropy(y)`** - Calculate entropy
   - Alternative impurity measure
   - Returns value between 0 (pure) and log₂(n_classes) (maximally mixed);
     that ceiling is 1.0 for binary, 1.585 for 3 classes

4. **`_mse(y)`** - Calculate mean squared error
   - Impurity measure for regression
   - Returns variance of values

5. **`_information_gain(y, y_left, y_right, parent_impurity=None)`** - Calculate information gain
   - Measures quality of a split
   - Higher is better
   - `parent_impurity` is an optional speed hint: I(parent) is the same for every
     candidate split of a node, so `_best_split` computes it once and passes it in

6. **`_best_split(X, y)`** - Find optimal split
   - Tests all features and all midpoint thresholds
   - Returns `{'feature_index', 'threshold', 'gain'}` for the highest-gain split

7. **`_build_tree(X, y, depth)`** - Recursively build tree
   - Main tree construction algorithm
   - Returns tree structure (nested dictionaries)

8. **`fit(X, y)`** - Train the model
   - Builds the tree from training data
   - Accepts plain Python lists as well as arrays
   - Returns `self`, so calls chain: `DecisionTree().fit(X, y).predict(X_new)`

9. **`predict(X)`** - Make predictions
   - Traverses tree for each sample
   - Returns predicted labels/values

10. **`predict_proba(X)`** - Class probabilities (classification only)
    - Returns shape `(n_samples, n_classes)`, columns ordered by `classes_`
    - Each leaf stores the class histogram of its training samples, so
      `P(c) = class_counts[c] / n_samples_in_leaf`

11. **`score(X, y)`** - Calculate performance
    - Accuracy for classification
    - R² score for regression

12. **`get_depth()`** - Get tree depth
    - Returns the longest root-to-leaf path measured in **edges**, the same
      unit as `max_depth` and the same convention as scikit-learn:
      a single leaf is 0, one split with two leaves is 1

13. **`get_n_leaves()`** - Count leaf nodes
    - Returns number of leaves (decision outcomes)

14. **`feature_importances_`** (property) - Impurity-based importance (MDI)
    - Each split's information gain, weighted by the fraction of data that
      reached it, summed per feature and normalised to sum to 1
    - Same definition scikit-learn uses

---

## Step-by-Step Example

Let's walk through building a decision tree for **customer purchase prediction**:

### The Data

```python
import numpy as np

# Features: [age, income_in_thousands]
X_train = np.array([
    [25, 30],   # Young, low income → No
    [45, 80],   # Middle-aged, high income → Yes
    [35, 50],   # Middle-aged, medium income → Yes
    [20, 25],   # Young, low income → No
    [50, 90],   # Older, high income → Yes
    [30, 35],   # Young, low income → No
    [40, 70],   # Middle-aged, high income → Yes
    [22, 28],   # Young, low income → No
])

# Labels: 0 = No purchase, 1 = Purchase
y_train = np.array([0, 1, 1, 0, 1, 0, 1, 0])
```

### Building the Tree (Step-by-Step)

**Step 1: Root Node**

The root holds all 8 samples, 4 "No" and 4 "Yes":

```
Gini(root) = 1 - (0.5² + 0.5²) = 0.5
```

Now `_best_split` enumerates every candidate. Each feature has 8 distinct
values, so each contributes 7 midpoints. This is the **complete** list the code
evaluates - every number below was produced by running `_information_gain` on
this exact dataset:

```
Age    unique values: 20  22  25  30  35  40  45  50
Age    midpoints    :   21  23.5 27.5 32.5 37.5 42.5 47.5

  Age <= 21.0    (1 left,  7 right)   IG = 0.0714
  Age <= 23.5    (2 left,  6 right)   IG = 0.1667
  Age <= 27.5    (3 left,  5 right)   IG = 0.3000
  Age <= 32.5    (4 left,  4 right)   IG = 0.5000  ← best
  Age <= 37.5    (5 left,  3 right)   IG = 0.3000
  Age <= 42.5    (6 left,  2 right)   IG = 0.1667
  Age <= 47.5    (7 left,  1 right)   IG = 0.0714

Income unique values: 25  28  30  35  50  70  80  90
Income midpoints    :   26.5 29  32.5 42.5  60  75  85

  Income <= 26.5 (1 left,  7 right)   IG = 0.0714
  Income <= 29.0 (2 left,  6 right)   IG = 0.1667
  Income <= 32.5 (3 left,  5 right)   IG = 0.3000
  Income <= 42.5 (4 left,  4 right)   IG = 0.5000  ← ties for best!
  Income <= 60.0 (5 left,  3 right)   IG = 0.3000
  Income <= 75.0 (6 left,  2 right)   IG = 0.1667
  Income <= 85.0 (7 left,  1 right)   IG = 0.0714
```

Two candidates tie at the maximum IG = 0.5, because in this tidy dataset age
and income rise together - every young customer is also a low-income customer,
so either question separates the classes perfectly.

**How is the tie broken?** By nothing more than loop order. `_best_split`
updates its running best with a strict `if gain > best_gain`, so a later
candidate must *beat* the incumbent, not merely match it. Features are scanned
in index order, so `Age` (feature 0) is seen first and keeps the crown:

```
Choose: Age <= 32.5   (IG = 0.5)
```

That is worth internalising: when features are correlated, which one a tree
"selects" can come down to column order. It is a big part of why single-tree
feature importances are unstable, and why `feature_importances_` reports
`[1.0, 0.0]` here even though Income was equally informative.

**Step 2: Left Child (Age ≤ 32.5)**
```
Rows with age 25, 20, 30, 22
Data: 4 samples (4 No, 0 Yes)
Gini = 0 (Pure!)

Create leaf: Predict "No"     class_counts = [4, 0]
```

**Step 3: Right Child (Age > 32.5)**
```
Rows with age 45, 35, 50, 40
Data: 4 samples (0 No, 4 Yes)
Gini = 0 (Pure!)

Create leaf: Predict "Yes"    class_counts = [0, 4]
```

Both children are pure, so `_build_tree`'s "all samples have the same label"
check fires immediately and neither child is split again. `max_depth=3` never
binds - the tree stops at one split because there is no impurity left to remove.

**Final Tree**:
```
                 [Root]
            Age <= 32.5?
              /        \
            Yes        No
            /            \
      [Predict: No]  [Predict: Yes]
```

### Training the Model

```python
model = DecisionTree(max_depth=3, criterion='gini', task='classification')
model.fit(X_train, y_train)

print(f"Tree depth: {model.get_depth()}")
# Output: Tree depth: 1
# (depth is counted in EDGES: one split = one edge, like sklearn's get_depth())

print(f"Number of leaves: {model.get_n_leaves()}")
# Output: Number of leaves: 2

print(f"Feature importances: {model.feature_importances_}")
# Output: Feature importances: [1. 0.]
```

### Making Predictions

```python
# New customers
X_test = np.array([
    [28, 32],   # Young, low income
    [42, 75],   # Middle-aged, high income
    [55, 95]    # Older, high income
])

predictions = model.predict(X_test)
print("Predictions:", predictions)
# Output: [0 1 1]  (No, Yes, Yes)   - numpy prints without commas

# Trace prediction for [28, 32]:
# 1. Age <= 32.5? → 28 <= 32.5 → Yes (go left)
# 2. Reached leaf → Predict "No" ✓

# Probabilities come from the leaf's stored class counts.
# Both leaves are pure here, so every probability is 0.0 or 1.0:
print("Probabilities:\n", model.predict_proba(X_test))
# Output: [[1. 0.]
#          [0. 1.]
#          [0. 1.]]
```

Notice what the midpoint threshold bought us. Had the tree split at the raw
value `Age <= 30` (the largest "No" age in the training data), a 31-year-old
customer would be classified "Yes" on the strength of a single year. The
midpoint `32.5` places the boundary in the middle of the observed gap
(30 -> 35), which is the most defensible position given no data in between.

---

## Real-World Applications

### 1. **Medical Diagnosis**
Diagnose diseases based on symptoms:
- Input: Symptoms, test results, patient history
- Output: Disease diagnosis
- Example: "Fever > 100°F AND Cough → Likely Flu"

### 2. **Credit Approval**
Decide whether to approve loans:
- Input: Income, credit score, debt, employment
- Output: Approve or Deny
- Example: "Income > $50k AND Credit Score > 650 → Approve"

### 3. **Customer Churn Prediction**
Predict if customers will leave:
- Input: Usage patterns, complaints, tenure
- Output: Will churn or stay
- Example: "Support tickets > 5 AND Tenure < 6 months → High risk"

### 4. **Email Spam Detection**
Classify emails as spam:
- Input: Keywords, sender, links
- Output: Spam or Not Spam
- Example: "Contains 'FREE' AND many links → Spam"

### 5. **Fraud Detection**
Identify fraudulent transactions:
- Input: Transaction amount, location, time, history
- Output: Fraudulent or Legitimate
- Example: "Amount > $1000 AND Location = Foreign → Flag for review"

### 6. **Product Recommendations**
Recommend products to customers:
- Input: Purchase history, browsing behavior
- Output: Product categories to recommend
- Example: "Bought electronics AND browsed laptops → Recommend accessories"

### 7. **Employee Attrition**
Predict employee turnover:
- Input: Salary, years at company, satisfaction scores
- Output: Will leave or stay
- Example: "Satisfaction < 3 AND No promotion in 2 years → High risk"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Calculating Gini Impurity

```python
def _gini_impurity(self, y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini
```

**Step-by-step**:
```python
y = [0, 0, 1, 1, 1]

# Count classes
unique: [0, 1]
counts: [2, 3]

# Calculate probabilities
probabilities = [2/5, 3/5] = [0.4, 0.6]

# Gini impurity
gini = 1 - (0.4² + 0.6²)
     = 1 - (0.16 + 0.36)
     = 1 - 0.52
     = 0.48
```

### 2. Finding Best Split

```python
def _best_split(self, X, y):
    n_samples, n_features = X.shape

    # Guard 1: too small to split at all
    if n_samples < self.min_samples_split:
        return None

    best_gain = -1
    best_split = None

    # I(parent) is identical for every candidate, so compute it ONCE
    parent_impurity = self._calculate_impurity(y)

    for feature_index in range(n_features):
        feature_values = X[:, feature_index]
        unique_values = np.unique(feature_values)

        # Candidates are MIDPOINTS between consecutive unique values
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

        for threshold in thresholds:
            # Split data
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            # Guard 2: this is the ONLY place min_samples_leaf does anything.
            # It filters CANDIDATE SPLITS - it is not a post-hoc pruning pass.
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            # Calculate gain
            gain = self._information_gain(y, y[left_mask], y[right_mask],
                                          parent_impurity=parent_impurity)

            # Update best. Strict `>` means the FIRST of any tie wins.
            if gain > best_gain:
                best_gain = gain
                best_split = {'feature_index': feature_index,
                              'threshold': threshold,
                              'gain': gain}

    return best_split
```

**How it works**:
1. Try every feature as a potential split
2. Try every midpoint between consecutive unique values as a threshold
3. Calculate information gain for each split
4. Keep the split with highest gain (ties go to the first candidate seen)

The returned `gain` is not decoration: `_build_tree` stores it on the internal
node, and `feature_importances_` later adds it up.

**Example** (the real numbers from the 8-row dataset in
[Step-by-Step Example](#step-by-step-example)):
```python
# Testing splits on Feature 0 (Age):
Threshold = 27.5: IG = 0.3000
Threshold = 32.5: IG = 0.5000 ← Best for this feature
Threshold = 37.5: IG = 0.3000

# Testing splits on Feature 1 (Income):
Threshold = 32.5: IG = 0.3000
Threshold = 42.5: IG = 0.5000 ← ties with Age <= 32.5
Threshold = 60.0: IG = 0.3000

Choose: Feature 0 (Age), Threshold = 32.5
        - the tie is broken by `gain > best_gain` being strict,
          so the lower feature index wins
```

### 3. Building Tree Recursively

```python
def _build_tree(self, X, y, depth=0):
    # Check stopping criteria
    if self.max_depth is not None and depth >= self.max_depth:
        return self._create_leaf(y)
    
    if len(np.unique(y)) == 1:  # Pure node
        return self._create_leaf(y)
    
    if n_samples < self.min_samples_split:  # Too few samples
        return self._create_leaf(y)
    
    # Find best split
    best_split = self._best_split(X, y)
    
    if best_split is None:  # No valid split
        return self._create_leaf(y)
    
    # Split data
    left_mask = X[:, best_split['feature_index']] <= best_split['threshold']
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[~left_mask], y[~left_mask]
    
    # Recursively build subtrees
    left_subtree = self._build_tree(X_left, y_left, depth + 1)
    right_subtree = self._build_tree(X_right, y_right, depth + 1)
    
    return {'type': 'internal', 
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'gain': best_split['gain'],        # kept for feature_importances_
            'n_samples': n_samples,            # kept for feature_importances_
            'left': left_subtree, 
            'right': right_subtree}
```

**The node schema** (the contract every other method reads):

```
Internal node:
{'type': 'internal', 'feature_index': int, 'threshold': float,
 'gain': float, 'n_samples': int, 'left': node, 'right': node}

Leaf node:
{'type': 'leaf', 'value': prediction, 'n_samples': int,
 'class_counts': array or None}    # counts only for task='classification'
```

**Stopping Criteria**:
1. **Max depth reached**: Prevent tree from growing too deep
2. **Pure node**: All samples have same label
3. **Too few samples**: Can't split reliably
4. **No valid split**: No split improves purity

### 4. Making Predictions

```python
def _predict_single(self, x, node):
    # If leaf, return value
    if node['type'] == 'leaf':
        return node['value']
    
    # Otherwise, go left or right
    if x[node['feature_index']] <= node['threshold']:
        return self._predict_single(x, node['left'])
    else:
        return self._predict_single(x, node['right'])
```

**Traversing the tree**:
```python
# Predict for sample [28, 32]

# Start at root
Node: Age <= 32.5?
Check: 28 <= 32.5? → Yes → Go left

# At left child (leaf)
Node: Leaf with value = 0, class_counts = [4, 0]
Return: 0 (No purchase)
```

### 5. Class Probabilities and Feature Importance

Two quantities fall out of the fitted tree almost for free.

**`predict_proba`** reuses the same traversal, but reads the leaf's stored class
histogram instead of its single `value`:

```python
def predict_proba(self, X):
    probabilities = np.zeros((len(X), self.n_classes))
    for i, x in enumerate(X):
        leaf = self._find_leaf(x, self.tree)      # same walk as _predict_single
        counts = leaf['class_counts']
        probabilities[i] = counts / counts.sum()  # P(c | leaf)
    return probabilities
```

These are empirical leaf frequencies, not calibrated probabilities. A leaf that
saw 7 of class A and 3 of class B reports `0.7 / 0.3`; a *pure* leaf reports
`1.0 / 0.0` no matter how few samples it holds. Deep trees therefore produce
badly overconfident probabilities - one more reason to constrain depth.

**`feature_importances_`** is Mean Decrease in Impurity (MDI). Every internal
node already recorded the gain it achieved and how many samples reached it:

```
importance[f] = Σ over nodes t that split on feature f of
                    (n_t / n_total) × IG(t)

then normalised so the vector sums to 1
```

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = DecisionTree(max_depth=5, criterion='gini', task='classification')
model.fit(X_train, y_train)

for name, imp in zip(data.feature_names, model.feature_importances_):
    print(f"{name:<20} {imp:.4f}")

# sepal length (cm)    0.0170
# sepal width (cm)     0.0000
# petal length (cm)    0.9046
# petal width (cm)     0.0785
```

Weighting by `n_t / n_total` matters: a split near the root that cleanly divides
all 150 samples deserves far more credit than an equally "clean" split that
tidied up the last 4 samples in a corner. This is the same definition
scikit-learn uses, so the two agree on the same fitted tree.

Two caveats worth carrying with you:
- MDI is **biased toward high-cardinality features**. A continuous feature
  offers hundreds of candidate thresholds, so it gets more chances to look good
  than a binary flag does.
- **Correlated features split the credit.** In the Step-by-Step Example, Age and
  Income were equally informative, yet importance came out `[1.0, 0.0]` purely
  because Age won a tie-break. Never read a zero as "this feature is useless".

---

## Model Evaluation

### For Classification

#### Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

#### Confusion Matrix

```
                Predicted
              0       1
Actual   0   [TN]    [FP]
         1   [FN]    [TP]
```

#### Precision, Recall, F1

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### For Regression

#### R² Score

```
R² = 1 - (SS_res / SS_tot)

where:
SS_res = Σ(y_true - y_pred)²
SS_tot = Σ(y_true - y_mean)²
```

### Example Evaluation

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Data (this fence is self-contained so you can run it as-is)
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTree(max_depth=5, criterion='gini', task='classification')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Tree statistics
print(f"\nTree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")
```

---

## Hyperparameter Tuning

### Key Hyperparameters

#### 1. max_depth

Controls maximum tree depth:

```
Small depth (2-5):
  Pros: Simple, interpretable, less overfitting
  Cons: May underfit, miss complex patterns
  
Large depth (10-20):
  Pros: Captures complex patterns
  Cons: Overfitting, hard to interpret
  
None (unlimited):
  Pros: Maximum flexibility
  Cons: Almost always overfits
```

**Visual**:
```
Depth = 2:           Depth = 5:           Depth = None:
   Simple             Moderate              Very Complex
   
    ●●●●               ●●|●●               ●|●|●
    ----               --|--               -|-|-
    ■■■■               ■■|■■               ■|■|■
```

#### 2. min_samples_split

Minimum samples to split a node:

```
min_samples_split = 2 (default):
  - Aggressive splitting
  - Complex tree, may overfit
  
min_samples_split = 20:
  - Conservative splitting
  - Simpler tree, better generalization
```

#### 3. min_samples_leaf

Minimum samples in leaf node:

```
min_samples_leaf = 1 (default):
  - Can create leaves with single sample
  - Risk of overfitting
  
min_samples_leaf = 10:
  - Each leaf has at least 10 samples
  - Smoother predictions, less overfitting
```

#### 4. criterion

Split quality measure:

```
Gini:
  - Faster to compute
  - Tends to isolate most frequent class
  - Default for most implementations
  
Entropy:
  - Information theory based
  - More balanced splits
  - Slightly slower
```

### Finding Optimal Parameters

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Data + a VALIDATION split. Tuning against the test set would leak it, so we
# carve the validation set out of the training data and leave test untouched.
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

# Grid search over parameters
# NOTE: 5 x 3 x 3 = 45 fits. That is fine on iris (120 rows, 4 features), but
# this implementation costs seconds per fit on a 500x30 dataset - size the grid
# to the data, or you will be waiting a long time.
depths = [3, 5, 7, 10, None]
min_splits = [2, 10, 20]
min_leafs = [1, 5, 10]

best_score = 0
best_params = {}

for depth in depths:
    for min_split in min_splits:
        for min_leaf in min_leafs:
            model = DecisionTree(max_depth=depth, 
                               min_samples_split=min_split,
                               min_samples_leaf=min_leaf,
                               criterion='gini',
                               task='classification')
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                best_params = {
                    'max_depth': depth,
                    'min_samples_split': min_split,
                    'min_samples_leaf': min_leaf
                }

print(f"Best parameters: {best_params}")
print(f"Best validation score: {best_score:.4f}")
```

---

## Advantages and Limitations

### Advantages ✅

1. **Highly Interpretable**
   - Easy to visualize and explain
   - Decision rules are human-readable
   - "White-box" model

2. **Handles Non-linear Relationships**
   - Can capture complex patterns
   - No assumption about data distribution
   - Creates flexible decision boundaries

3. **No Feature Scaling Needed**
   - Works with features on different scales
   - No normalization required
   - Split decisions are based on thresholds

4. **Handles Mixed Data Types**
   - Works with numerical and categorical features
   - Can handle missing values (with extensions)

5. **Fast Prediction**
   - O(log n) prediction time with balanced tree
   - Simple tree traversal

6. **Feature Importance**
   - Can easily compute feature importance
   - Shows which features are most useful
   - Implemented here as `feature_importances_` - the split gains the tree
     already recorded, weighted by node size and normalised. See the worked
     example in [Understanding the Code](#understanding-the-code).

### Limitations ❌

1. **Prone to Overfitting**
   - Can create overly complex trees
   - Memorizes training data
   - Solution: Limit depth, pruning, ensemble methods

2. **High Variance**
   - Small changes in data → very different tree
   - Unstable predictions
   - Solution: Use ensemble methods (Random Forests)

3. **Biased Toward Dominant Classes**
   - With imbalanced data, may ignore minority class
   - Solution: Class weights, resampling

4. **Can't Extrapolate**
   - Predictions limited to training data range
   - Won't predict values outside training range

5. **Greedy Algorithm**
   - Locally optimal splits (not globally optimal)
   - May miss better overall tree structure

6. **Sensitive to Outliers**
   - Outliers can create unnecessary splits
   - Solution: Outlier removal, robust splitting

### When to Use Decision Trees

**Good Use Cases**:
- ✅ Need interpretable model
- ✅ Have mixed data types
- ✅ Non-linear relationships
- ✅ Feature interactions important
- ✅ Don't want to scale features

**Bad Use Cases**:
- ❌ Need stable predictions
- ❌ Linear relationships (use regression)
- ❌ Very high dimensional data
- ❌ Need to extrapolate
- ❌ Imbalanced data (without handling)

---

## Preventing Overfitting

### 1. Pre-pruning (Early Stopping)

Stop tree growth early:

```python
# Limit tree depth
model = DecisionTree(max_depth=5)

# Require minimum samples to split
model = DecisionTree(min_samples_split=20)

# Require minimum samples per leaf
model = DecisionTree(min_samples_leaf=10)
```

**Effect**:
```
Before:                    After (max_depth=2):
      [Root]                     [Root]
     /      \                   /      \
   [A]      [B]               [A]      [B]
   / \      / \               / \      (leaf)
  [C][D]  [E][F]            [C][D]
  / \  \                   (leaf)(leaf)
[G][H][I]
(Many levels!)           (Stopped at depth 2)

Depth is counted in EDGES: Root -> A is 1, A -> C is 2. So max_depth=2
is what stops C and D from splitting again.
```

### 2. Cross-Validation

Validate on held-out data:

Our `DecisionTree` is not a scikit-learn estimator (it has no `get_params`, so
`cross_val_score` cannot clone it). In the spirit of this repo, here is k-fold
cross-validation written out with nothing but NumPy - it is about eight lines,
and writing it once makes the idea concrete:

```python
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Split the shuffled row indices into 5 roughly equal folds
rng = np.random.RandomState(42)
folds = np.array_split(rng.permutation(len(X)), 5)

depths = list(range(1, 11))
scores = []

for depth in depths:
    fold_scores = []
    for k in range(5):
        val_idx = folds[k]                                    # hold this out
        train_idx = np.concatenate([folds[j] for j in range(5) if j != k])

        model = DecisionTree(max_depth=depth)
        model.fit(X[train_idx], y[train_idx])
        fold_scores.append(model.score(X[val_idx], y[val_idx]))

    scores.append(np.mean(fold_scores))   # average over the 5 folds
    print(f"  max_depth={depth:<5} cv accuracy = {scores[-1]:.4f}")

best_depth = depths[int(np.argmax(scores))]
print(f"Optimal max_depth: {best_depth}")
```

Every row serves as validation data exactly once, so the score averages over 5
different train/validation partitions instead of trusting a single lucky split -
which matters enormously for a model as high-variance as a decision tree.

### 3. Ensemble Methods

Combine multiple trees:

```
Single Tree:          Random Forest:
   Unstable           Stable (average of many trees)
   High variance      Low variance
   May overfit        Better generalization
```

### 4. Post-pruning (Not Implemented Here)

Pre-pruning has a known weakness: it is **short-sighted**. A split that looks
worthless on its own may be exactly the split that enables two excellent splits
beneath it (think of the XOR pattern, where neither feature helps alone). Stop
early and you never find out.

Post-pruning takes the opposite approach: grow the tree out fully, *then* cut
back the branches that do not pay for themselves. The standard method is
**cost-complexity pruning** (`ccp_alpha` in scikit-learn), which scores a tree by

```
R_alpha(T) = R(T) + alpha × |leaves(T)|

where R(T)         = total impurity of the tree's leaves
      |leaves(T)|  = number of leaves (the complexity penalty)
      alpha        = price charged per leaf
```

Raising `alpha` makes leaves expensive, so subtrees that bought only a small
impurity reduction get collapsed back into a single leaf. Sweeping `alpha` from
0 upward yields a nested sequence of ever-smaller trees, and you pick among them
by cross-validation.

**This implementation does not include post-pruning** - see
[Simplifications vs. Canonical CART](#simplifications-vs-canonical-cart).
Use `max_depth`, `min_samples_split` and `min_samples_leaf` instead, and lean on
cross-validation to choose them.

---

## Simplifications vs. Canonical CART

This is a teaching implementation. It is faithful on the parts that define the
algorithm - and it is measurably so: on all 150 iris rows its predictions agree
with scikit-learn's `DecisionTreeClassifier` on **100% of samples** at
`max_depth` 1-10 for both Gini and Entropy, its leaf counts match exactly on
that data under every `min_samples_leaf` / `min_samples_split` setting tested
(1/2/5/10 crossed with 2/5/10/20, at `max_depth=None`), and its regression
training R² matches `DecisionTreeRegressor` exactly on the diabetes dataset at
`max_depth` 2-5.

`feature_importances_` is the one comparison that needs a caveat. Where both
implementations pick the same splits the two vectors agree to ~1e-16, but
scikit-learn breaks **exact gain ties** using a random feature permutation
seeded from its `random_state`, whereas this code always keeps the lowest
feature index. When a tie falls the other way the partition of the data is
unchanged and only the credit moves: fitting all 150 iris rows at `max_depth=5`
with scikit-learn's `random_state=42`, exactly one of the 17 nodes splits on a
different feature and the two importance vectors differ by 0.013; at
`max_depth=1` with its `random_state=0`, scikit-learn splits on petal width
where this code splits on petal length - the identical partition, but
importances of `[0, 0, 0, 1]` against `[0, 0, 1, 0]`. The same tie-breaking
is why individual regression predictions can differ at `max_depth` >= 4 on
tie-heavy data even when the two trees' training R² is identical.

Four things canonical CART does that this file deliberately leaves out:

### 1. Post-pruning (cost-complexity / `ccp_alpha`)

- **Canonical**: grow fully, then minimise `R_alpha(T) = R(T) + alpha × |leaves(T)|`,
  producing a nested sequence of subtrees to choose between by cross-validation.
- **Here**: pre-pruning only (`max_depth`, `min_samples_split`, `min_samples_leaf`).
- **Consequence**: you may occasionally miss a good tree that requires one
  unpromising split to reach a rewarding one. In practice tuning `max_depth` by
  cross-validation gets you most of the way there.
- **Why omitted**: the pruning-path algorithm (computing the effective alpha of
  every subtree, then collapsing in order) is roughly as much code again as the
  entire tree builder, and it would bury the recursive-partitioning idea that
  this file exists to teach.

### 2. Missing values / surrogate splits

- **Canonical CART**: when a sample's split feature is missing, fall back to a
  *surrogate* split - a different feature that mimics the primary split's
  partition most closely.
- **Here**: `fit` calls `np.asarray(X, dtype=float)` and `NaN` comparisons are
  always `False`, so missing values silently all go right. Impute before fitting.

### 3. Native categorical features

- **Canonical**: some implementations search over subset partitions of a
  categorical feature's levels (`{red, blue}` vs `{green}`).
- **Here**: every feature is treated as numeric and split by `<=`. One-hot
  encode categorical columns first. (Note that scikit-learn shares this
  limitation - it is not unique to this implementation.)

### 4. Optimised split search

- **Canonical (scikit-learn)**: sorts each feature once per node and sweeps the
  split point, updating the class counts incrementally in compiled Cython.
- **Here**: re-partitions the data with NumPy masks at every candidate
  threshold, giving `O(n_features × n × log n)` per node with a large constant.
- **Consequence**: a 455x30 dataset (breast cancer, `random_state=42` split)
  fits in 1-2 seconds here versus ~5 ms in scikit-learn. The trees come out the
  same **shape** - identical leaf counts and depths at `max_depth` 3, 5 and
  None, and identical predictions on all 455 training rows - but they are not
  bit-identical: scikit-learn casts `X` to float32 before splitting, so its
  thresholds land ~1e-8 (relative) away from the float64 midpoints used here,
  and it breaks tied splits differently. With scikit-learn's `random_state=42`
  that moves 1 of the 114 test rows. Per this repo's "clarity over performance"
  rule, the wall clock is the intended trade.

One more practical note: both `_build_tree` and `_predict_single` use plain
Python recursion, because recursion *is* the lesson. On pathological data an
unconstrained tree can therefore hit Python's recursion limit (~1000) where
scikit-learn's explicit stack would not. Setting `max_depth` avoids this.

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = DecisionTree(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    criterion='gini',
    task='classification'
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Tree statistics
print(f"\nTree Statistics:")
print(f"  Depth: {model.get_depth()}")
print(f"  Number of leaves: {model.get_n_leaves()}")

# Which features drove the splits?
print("\nFeature Importances:")
for name, imp in zip(data.feature_names, model.feature_importances_):
    print(f"  {name:<20} {imp:.4f}")

# Show predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"  True: {data.target_names[y_test[i]]}, "
          f"Predicted: {data.target_names[y_pred[i]]}")

# Compare different depths
# NOTE: keep printed output ASCII-only ("->", not an arrow character).
# A Windows console defaults to cp1252 and raises UnicodeEncodeError otherwise.
print("\nComparing Tree Depths:")
for depth in [2, 3, 5, 7, None]:
    model = DecisionTree(max_depth=depth, task='classification')
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    actual_depth = model.get_depth()
    n_leaves = model.get_n_leaves()
    
    print(f"  max_depth={str(depth):>4} -> "
          f"train={train_acc:.3f}, test={test_acc:.3f}, "
          f"depth={actual_depth}, leaves={n_leaves}")
```

---

## Key Concepts to Remember

### 1. **Trees Make Sequential Decisions**
Like a flowchart, they ask questions one at a time until reaching a decision.

### 2. **Greedy Splitting**
At each node, choose the split that gives the biggest immediate improvement (not globally optimal).

Why settle for greedy? Because finding the *optimal* tree is intractable.
Hyafil and Rivest proved in 1976 that constructing a minimal binary decision
tree is NP-complete, and the reason is easy to feel: the choice at the root
changes which splits are even available three levels down, so you cannot
evaluate a root split without exploring every tree that could grow beneath it.
The number of candidate trees explodes combinatorially with depth and features.

Greedy splitting sidesteps this by making each decision locally and never
revisiting it. That is fast and usually good, but it is exactly why a tree can
miss an XOR-style pattern: neither feature alone reduces impurity, so no split
looks worthwhile at the root even though a two-level tree would separate the
data perfectly. It is also why post-pruning exists (grow first, judge later) and
why the tie-break in [Step-by-Step Example](#step-by-step-example) is not a
detail - one arbitrary choice at the root propagates through the whole tree.

### 3. **Overfitting is Common**
Deep trees memorize training data. Always use max_depth or other constraints!

### 4. **No Feature Scaling Needed**
Unlike KNN or Neural Networks, decision trees work fine with unscaled features.

### 5. **High Interpretability**
Can visualize and explain every decision. Great for getting stakeholder buy-in!

### 6. **Recursive Algorithm**
Building a tree is inherently recursive: solve problem by solving smaller subproblems.

---

## Conclusion

Decision Trees are a fundamental and powerful algorithm! By understanding:
- How trees recursively split data
- How impurity measures guide splits
- How to prevent overfitting with constraints
- How to interpret and visualize decisions

You've gained a crucial tool in your machine learning toolkit! 🌳

**When to Use Decision Trees**:
- ✅ Need interpretable model
- ✅ Non-linear patterns
- ✅ Mixed data types
- ✅ Feature interactions matter
- ✅ Classification or regression

**When to Use Something Else**:
- ❌ Need stable predictions → Use ensemble methods
- ❌ High-dimensional sparse data → Use linear models
- ❌ Linear relationships → Use linear/logistic regression
- ❌ Need *calibrated* probabilities → Use logistic regression, or calibrate
  the tree. `predict_proba` here returns raw leaf frequencies, so a pure leaf
  reports 1.0 however few samples it holds - confident, but not honest.

**Next Steps**:
- Try decision trees on your own datasets
- Experiment with different hyperparameters
- Learn about Random Forests (ensemble of trees)
- Study Gradient Boosting (sequential trees)
- Explore feature importance analysis with `feature_importances_`
- Visualize your trees to understand decisions

Happy coding! 💻🌳

