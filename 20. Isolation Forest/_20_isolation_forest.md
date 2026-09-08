# Isolation Forest from Scratch: A Comprehensive Guide

![Isolation Forest](https://img.shields.io/badge/Algorithm-Anomaly%20Detection-red)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow)
![Type](https://img.shields.io/badge/Type-Unsupervised-blue)

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [Introduction](#introduction)
3. [When to Use Isolation Forest](#when-to-use-isolation-forest)
4. [How It Works](#how-it-works)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Step-by-Step Example](#step-by-step-example)
7. [Implementation Details](#implementation-details)
8. [Understanding the Code](#understanding-the-code)
9. [Usage Examples](#usage-examples)
10. [Hyperparameters Explained](#hyperparameters-explained)
11. [Advantages & Limitations](#advantages--limitations)
12. [Comparison with Other Anomaly Detection Methods](#comparison-with-other-anomaly-detection-methods)
13. [Tips & Best Practices](#tips--best-practices)
14. [Performance Characteristics](#performance-characteristics)

---

## Quick Start: Plug-and-Play Example

This mirrors DEMO 1 of the `if __name__ == "__main__":` block at the bottom of
`_20_isolation_forest.py` (same data, same seed, same prints - just unindented
out of the `__main__` guard). Copy it, paste it, run it - NumPy is the only
dependency. Or just run the file directly:

```bash
python _20_isolation_forest.py
```

```python
# ---------------------------------------------------------------
# Isolation Forest from Scratch - Complete Runnable Example
# Requires: numpy only
# This is DEMO 1 of the __main__ block, verbatim.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the IsolationForest class here (from _20_isolation_forest.py) ----
# class IsolationForest: ...

np.random.seed(42)

def _report(name, y_true, y_pred):
    tp = np.sum((y_pred == -1) & (y_true == -1))
    fp = np.sum((y_pred == -1) & (y_true == 1))
    fn = np.sum((y_pred == 1) & (y_true == -1))
    acc = np.mean(y_pred == y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"  {name:5s} accuracy={acc:7.2%}  precision={prec:7.2%}  recall={rec:7.2%}")

print("=" * 55)
print("DEMO 1 - Outliers in a 2-D Gaussian cloud")
print("=" * 55)
print("500 inliers ~ N(0, 0.5^2) plus 40 outliers ~ U(-4, 4).")
print("The model never sees a label - it only measures how few random")
print("splits it takes to isolate each point.")

X_in = np.random.randn(500, 2) * 0.5
X_out = np.random.uniform(-4, 4, (40, 2))
X = np.vstack([X_in, X_out])
y = np.array([1] * 500 + [-1] * 40)          # 1 = normal, -1 = anomaly

# Shuffle BEFORE splitting so both halves hold the same mix of classes,
# and keep the halves disjoint: [:400] and [400:] (not [300:]!).
perm = np.random.permutation(540)
X, y = X[perm], y[perm]
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

model = IsolationForest(
    n_estimators=100,
    max_samples='auto',        # -> psi = min(256, n_samples)
    contamination=40 / 540,    # match the REAL 7.4% outlier rate
    random_state=42
)
model.fit(X_train)             # unsupervised: labels are never used

print(f"\npsi (max_samples_) = {model.max_samples_}   "
      f"c(psi) = {model._calculate_c(model.max_samples_):.4f}   "
      f"height limit = {int(np.ceil(np.log2(max(model.max_samples_, 2))))}")
print("score s(x) = 2^(-E[h(x)] / c(psi)):  ~1 = anomaly, ~0.5 = average, ~0 = normal")

_report("Train", y_train, model.predict(X_train))
_report("Test ", y_test, model.predict(X_test))

train_scores = model.decision_function(X_train)   # higher = more anomalous
print(f"\n  mean score of inliers  ~= {train_scores[y_train == 1].mean():.4f}")
print(f"  mean score of outliers ~= {train_scores[y_train == -1].mean():.4f}")
print(f"  flagging threshold     =  {model.threshold:.4f}")

test_scores = model.decision_function(X_test)
test_pred = model.predict(X_test)
show = np.concatenate([np.where(y_test == 1)[0][:3],
                       np.where(y_test == -1)[0][:3]])
print("\nSample test predictions (true, score, predicted):")
for i in show:
    print(f"  x=({X_test[i, 0]:5.2f},{X_test[i, 1]:5.2f})  "
          f"true={y_test[i]:+d}  score={test_scores[i]:.4f}  pred={test_pred[i]:+d}")
```

Expected output:

```
=======================================================
DEMO 1 - Outliers in a 2-D Gaussian cloud
=======================================================
500 inliers ~ N(0, 0.5^2) plus 40 outliers ~ U(-4, 4).
The model never sees a label - it only measures how few random
splits it takes to isolate each point.

psi (max_samples_) = 256   c(psi) = 10.2448   height limit = 8
score s(x) = 2^(-E[h(x)] / c(psi)):  ~1 = anomaly, ~0.5 = average, ~0 = normal
  Train accuracy= 99.00%  precision= 90.00%  recall= 96.43%
  Test  accuracy= 98.57%  precision=100.00%  recall= 83.33%

  mean score of inliers  ~= 0.4026
  mean score of outliers ~= 0.6646
  flagging threshold     =  0.5454

Sample test predictions (true, score, predicted):
  x=(-0.41,-0.16)  true=+1  score=0.3773  pred=+1
  x=(-0.04, 0.56)  true=+1  score=0.3795  pred=+1
  x=( 0.13, 0.39)  true=+1  score=0.3773  pred=+1
  x=( 3.80, 1.20)  true=-1  score=0.7092  pred=-1
  x=( 2.95, 1.82)  true=-1  score=0.6998  pred=-1
  x=( 0.38, 2.54)  true=-1  score=0.6192  pred=-1
```

Running the file directly also prints DEMO 2, which refits the same kind of
model on 500 manufactured parts (5 sensors, 450 in spec, 50 out of spec) at
three different `contamination` values:

```
  contam   flagged   train prec   train rec   test prec   test rec
-----------------------------------------------------------------
    0.05        20      100.00%      48.78%     100.00%     55.56%
    0.10        40       85.00%      82.93%      80.00%     88.89%
    0.15        60       63.33%      92.68%      66.67%     88.89%
```

Read the two demos together and the whole algorithm is on screen: DEMO 1 shows
that outliers really do get shorter paths (mean score 0.66 vs 0.40), and DEMO 2
shows that `contamination` only slides the cut-off along that same ranking -
raise it and recall climbs while precision falls.

---

## Introduction

**Isolation Forest** is an unsupervised machine learning algorithm designed specifically for anomaly detection. Unlike traditional methods that profile normal data, Isolation Forest explicitly isolates anomalies.

### Key Insight

The algorithm is based on a simple but powerful idea:
> **Anomalies are few and different, therefore they are easier to isolate than normal points.**

Think of it like this: In a crowd of people standing close together, if one person is standing far away from everyone else, it's much easier to "isolate" that person with fewer divisions of space.

### Real-World Analogy

Imagine you're organizing books on a shelf:
- **Normal books**: Most books are similar sizes, clustered together. You need many separators to isolate a specific book
- **Anomalous book**: A very large or tiny book stands out. You need very few separators to isolate it

Isolation Forest works the same way - anomalies require fewer "separations" (splits) to isolate.

---

## When to Use Isolation Forest

### Perfect For:
- **Fraud Detection**: Credit card fraud, insurance fraud
- **Intrusion Detection**: Network security, cybersecurity attacks
- **Manufacturing**: Defect detection, quality control
- **Healthcare**: Disease outbreak detection, unusual patient readings
- **System Monitoring**: Server anomalies, application performance
- **IoT**: Sensor malfunction detection

### When NOT to Use:
- **Small datasets** (< 100 samples): Not enough data for reliable isolation
- **High-dimensional data with no clear anomalies**: May produce too many false positives
- **When all anomalies must be caught**: Isolation Forest may miss some subtle anomalies
- **Clustered anomalies**: If anomalies form their own cluster, they may appear "normal"

---

## How It Works

Isolation Forest builds an ensemble of **Isolation Trees** (similar to Random Forests) but with a completely different goal and construction method.

### Step-by-Step Process

#### 1. Build Multiple Isolation Trees

For each tree:
1. **Randomly subsample** data (typically 256 samples)
2. **Randomly select** a feature
   - more precisely, among the features that still *vary* inside the current
     node - a constant column cannot be cut
3. **Randomly select** a split value between min and max of that feature
4. **Recursively partition** until:
   - Only one sample in node, OR
   - Maximum depth reached, OR
     (the depth cap is `ceil(log2 psi)`, see the Mathematical Foundation)
   - All samples are identical
     (strictly: every *allowed* feature is constant in the node)

#### 2. Calculate Path Lengths

For each sample:
- Pass it through all trees
- Record the **path length** (number of edges from root to leaf)
- Shorter paths → More anomalous
- Longer paths → More normal

#### 3. Compute Anomaly Score

Average path length across all trees, normalized to [0, 1]:
- **Score → 1**: Very likely anomaly
- **Score → 0.5**: Borderline
- **Score → 0**: Very likely normal

#### 4. Apply Threshold

Based on the `contamination` parameter, determine a threshold:
- Samples with score > threshold are marked as anomalies

### Visual Example

The whole point is that the anomaly's leaf is **shallower** than everyone
else's. A tree where every leaf sits at the same depth would prove nothing, so
watch the depth numbers, not the shape:

```
Isolation Tree Example (deepest leaf at depth = 4):

                        [Feature 2 < 5.3]                       depth 0
                       /                 \
              [ANOMALY]                [Feature 0 < 2.1]        depth 1
              h(x) = 1                /                 \
                          [Feature 1 < 8.4]        [Feature 1 < 3.0]   depth 2
                          /            \            /            \
                    [Normal]   [Feature 0 < 0.7]  [Normal]   [Normal]  depth 3
                    h(x) = 3    /            \    h(x) = 3   h(x) = 3
                          [Normal]     [Normal]                        depth 4
                          h(x) = 4     h(x) = 4

The lone far-out point fell on the empty side of the very first split, so it
was isolated after ONE cut:            h(anomaly) = 1
The crowded normal points needed 3-4 cuts before each sat alone:
                                       h(normal)  = 3 or 4
```

Averaged over 100 such random trees, that gap is what the anomaly score
measures. Note that `h(x)` here is a *count of edges walked*: the leaf-size
correction described below only kicks in when a leaf still holds several points
because the height limit stopped the tree early.

---

## Mathematical Foundation

> **Two different counts, two different symbols.** Keep these straight - the
> single most common mistake when reading this formula is to normalise by the
> number of trees.
>
> | Symbol | Meaning | In the code | Typical value |
> |--------|---------|-------------|---------------|
> | *n* | number of trees in the forest | `n_estimators` | 100 |
> | *ψ* (psi) | subsample size **used to grow one tree** | `max_samples_` | 256 |
>
> *n* appears **only** in the averaging step below. Every appearance of *c(·)*
> and of the score *s(·)* takes **ψ**, never *n*, and never the size of the
> full dataset.

### Path Length

For a sample **x** in an isolation tree:
- **h(x)** = number of edges from root to leaf where x lands

### Average Path Length

Over all trees:

```
E[h(x)] = (1/n) × Σ h_i(x)
```

where n is the number of trees (`n_estimators`). This is the only place *n*
appears.

### Normalization Constant

To normalize path lengths, we use the average path length of an unsuccessful search in a Binary Search Tree with **ψ** nodes - where ψ is the *per-tree subsample size*, `max_samples_` (256 by default):

```
c(ψ) = 2H(ψ-1) - 2(ψ-1)/ψ
```

where H(i) ≈ ln(i) + 0.5772 (Euler's constant), so in the form the code uses:

```
c(ψ) = 2(ln(ψ-1) + 0.5772156649) - 2(ψ-1)/ψ
```

with the two special cases `c(ψ) = 0` for ψ ≤ 1 and `c(2) = 1`. This is
`_calculate_c()` in `_20_isolation_forest.py`, and it matches scikit-learn's
internal `_average_path_length` to within 3e-12.

This represents the expected path length for a normal point: it is the yardstick every measured path is compared against.

### Anomaly Score

The final anomaly score is:

```
s(x, ψ) = 2^(-E[h(x)] / c(ψ))
```

**Interpretation:**
- **E[h(x)] ≪ c(ψ)** → s(x) → 1 (very anomalous)
- **E[h(x)] ≈ c(ψ)** → s(x) ≈ 0.5 (borderline)
- **E[h(x)] ≫ c(ψ)** → s(x) → 0 (very normal)

### Why leaves get a path-length bonus

A tree stops growing at the height limit, so many leaves still contain several
points that were *never actually separated*. Charging those points only the
depth they reached would understate their path and make them look anomalous.
So the reference implementation adds back the path they would still have
needed - modelled, again, as a random BST on the points sitting in that leaf:

```
h(x) = depth_reached  +  c(size_of_leaf)
```

This is exactly the return line of `_path_length()`:

```python
if tree['type'] == 'leaf':
    return current_height + self._calculate_c(tree['size'])
```

A leaf holding one point adds `c(1) = 0` (it really was isolated). A leaf
holding 40 points adds `c(40) ≈ 6.53`, pushing that point firmly into
"normal" territory - which it should be, since it is sitting in a crowd.

### Why the height limit is ⌈log₂ ψ⌉

`fit()` caps every tree at

```
height_limit = ceil(log2(max(ψ, 2)))  # 8 when ψ = 256
```

(the `max(ψ, 2)` only exists so a degenerate ψ of 0 or 1 cannot produce
`log2(0) = -inf`; for every ψ ≥ 2 it is exactly `ceil(log2 ψ)`).

Two reasons, and both are about *anomalies*:

1. ⌈log₂ ψ⌉ is the paper's stand-in for the **typical** depth of a random
   binary tree over ψ points. Note it sits a little *below* the average path
   length c(ψ) = 10.24 that the score normalises by - 8 < 10.24 - which is
   exactly why the leaf bonus in the next paragraph is not optional.
   Anomalies are isolated well before either number, so every extra level past
   this limit only subdivides the dense normal region.
2. A branch stopped by the limit is truncated in *memory*, not in *effect*: its
   leaf hands back `c(size_of_leaf)`, the paper's estimate of the path those
   points would still have spent if the tree had kept growing. Growing
   deeper would, on average, only re-derive a number the formula already
   supplies - at real time and memory cost.

Beware a tempting shortcut here: it is **not** true that a long path has
"saturated" the score. s is a plain exponential, so a path of exactly c(ψ)
scores 0.5, one of 2·c(ψ) scores 0.25 and one of 3·c(ψ) scores 0.125 - it keeps
halving, it never flattens. Reason 2, not saturation, is why capping the depth
is free.

Points cut off by the limit are not thrown away: they get the leaf bonus above,
so their score is still well defined.

### Why This Works

The math formalizes the intuition:
1. **Anomalies are isolated faster** → shorter path h(x)
2. **Shorter path relative to c(ψ)** → score approaches 1
3. **Exponential transformation** → emphasizes differences

---

## Step-by-Step Example

Every number below is hand-checkable with a calculator, and the code prints the
same values (`python _20_isolation_forest.py` starts by printing
`psi (max_samples_) = 256   c(psi) = 10.2448   height limit = 8`).

**Setup.** A forest of `n_estimators = 100` trees, each grown on a subsample of
`max_samples = 256` points. So ψ = 256 and n = 100.

**Step 1 - the yardstick c(ψ).**

```
c(256) = 2(ln(255) + 0.5772156649) - 2(255)/256
       = 2(5.5412635 + 0.5772157) - 1.9921875
       = 12.2369584 - 1.9921875
       = 10.2448        <- the expected path of an "average" point
```

**Step 2 - average a point's path over the 100 trees.**
Suppose a suspicious transaction lands at depths
`3, 5, 4, 6, 4, ...` and the mean over all 100 trees comes to

```
E[h(x_anomaly)] = 4.5        (well under 10.24)
```

and a typical normal row averages

```
E[h(x_normal)]  = 11.0       (a bit over 10.24)
```

**Step 3 - convert each to a score.**

```
s(anomaly, 256) = 2^(-4.5 / 10.2448)  = 2^(-0.4392) = 0.738
s(normal,  256) = 2^(-11.0 / 10.2448) = 2^(-1.0737) = 0.475
```

**Step 4 - read the scores.** 0.738 is far above 0.5, so this point was
isolated in roughly *half* the cuts an average point needs: flag it. 0.475 sits
just under 0.5: unremarkable.

**Step 5 - turn scores into labels.** `fit()` scores the training set once and
stores the (1 - contamination) quantile as `self.threshold`. With
`contamination = 0.074` on the Quick Start data that came out at **0.5454**;
`predict()` is then literally

```python
predictions = np.where(scores > self.threshold, -1, 1)
```

so our 0.738 point is labelled **-1** (anomaly) and the 0.475 point **+1**
(normal). Nothing else in the model changes when you move `contamination` -
only where this one cut falls.

**Sanity check you can run.** On pure uniform noise no point is special, so
E[h(x)] should land near c(ψ) and every score near 0.5. Measured on 500 uniform
3-D points with 200 trees, repeated over 20 different data seeds: **mean score
0.504 to 0.514**, averaging 0.509. (No seed is quoted because the point is that
*any* seed lands there; expect your own run to differ in the third decimal.)
That is the formula behaving exactly as designed.

---

## Implementation Details

### Core Algorithm

```python
class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto',
                 contamination=0.1, max_features=1.0, random_state=None):
        # Store parameters, plus a PRIVATE RNG:
        #   self._rng = np.random.RandomState(random_state)
        # so fitting never disturbs the caller's global np.random stream

    def fit(self, X, y=None):          # y is ignored - unsupervised
        # 1. Resolve psi  -> self.max_samples_   (int / float / 'auto')
        # 2. Resolve the column budget -> self.max_features_
        # 3. height_limit = ceil(log2(psi))
        # 4. For each of n_estimators trees:
        #       subsample psi rows, draw this tree's column subset ONCE,
        #       then _build_tree()
        # 5. Score the training set and store the
        #    (1 - contamination) quantile as self.threshold

    def predict(self, X):
        # 1. Calculate average path lengths
        # 2. Convert to anomaly scores
        # 3. Apply threshold to classify (-1 anomaly / +1 normal)
```

Note the trailing underscore convention: `max_samples` / `max_features` are
what *you* passed, `max_samples_` / `max_features_` are what `fit()` resolved
them to for this particular dataset. Refitting on data with a different number
of columns re-resolves them; your constructor arguments are never overwritten.

### Building an Isolation Tree

```python
def _build_tree(self, X, height_limit, current_height=0, features=None):
    # Base case 1: max depth reached, or a single sample left
    if current_height >= height_limit or len(X) <= 1:
        return {'type': 'leaf', 'size': len(X)}

    # `features` are the columns THIS TREE may split on (drawn once in fit()).
    if features is None:
        features = np.arange(X.shape[1])

    # Base case 2: keep only the columns that still VARY inside this node.
    # A constant column cannot be split - but its siblings still can, so we
    # redraw among the splittable ones instead of giving up.
    col_min, col_max = X.min(axis=0), X.max(axis=0)
    candidates = features[col_min[features] < col_max[features]]
    if len(candidates) == 0:                 # every allowed column is constant
        return {'type': 'leaf', 'size': len(X)}

    # Randomly select feature and split value
    feature = candidates[self._rng.randint(len(candidates))]
    min_val, max_val = X[:, feature].min(), X[:, feature].max()
    split_value = self._rng.uniform(min_val, max_val)

    # Partition data
    left_mask = X[:, feature] < split_value
    right_mask = ~left_mask

    # Base case 3: a degenerate draw put everything on one side
    if not np.any(left_mask) or not np.any(right_mask):
        return {'type': 'leaf', 'size': len(X)}

    # Recursively build subtrees (same allowed column subset)
    return {
        'type': 'internal',
        'feature': feature,                  # absolute column index
        'split_value': split_value,
        'left':  self._build_tree(X[left_mask],  height_limit, current_height + 1, features),
        'right': self._build_tree(X[right_mask], height_limit, current_height + 1, features)
    }
```

**Why base case 2 matters more than it looks.** The obvious shortcut - draw one
feature, and if it happens to be constant in this node give up and make a leaf -
quietly flattens the score scale on data with low-cardinality columns
(categorical codes, small counts, one-hot flags). Measured on 440 rows with 1
informative column (400 points from `N(0, 1)` plus 40 from `U(-6, 6)`) and 9
all-zero columns bolted on, `n_estimators=100`; the data for seed *s* comes from
`np.random.RandomState(s)` for *s* = 0, 1, 2, 3, 4, every model is fitted with
`random_state=42`, and each cell is the mean over those 5 data seeds with ± its
standard deviation across them:

| Variant | ROC AUC | score spread (sd) | trees that are a bare root leaf |
|---------|---------|-------------------|---------------------------------|
| Give up on first constant draw | 0.853 ± 0.042 | 0.0020 | 89.6 / 100 |
| Redraw among splittable columns | **0.885 ± 0.029** | **0.0732** | **0 / 100** |
| scikit-learn 1.7.2 | 0.876 ± 0.032 | 0.0749 | - |

Read the AUC column and the spread column as two separate effects, because they
are not the same size. The *ranking* degrades only mildly: redrawing wins by
0.022 AUC on average and on 18 of the 20 data seeds 0-19 (it loses on 2), so it
would overstate the case to call the gap decisive. What collapses is the score
*scale*. Without the redraw ~90 of 100 trees never split at all, every point
collects the same constant path length from those trees, and the whole score
range shrinks to about 0.014 wide - a 27x squeeze against 0.386 with the
redraw. That part held on every seed tried, and an anomaly score you cannot put
a threshold on is the real damage.

(The give-up variant is not shipped code - it is a one-line change to
`_build_tree` - so its exact AUC digit depends on how you write it. Its spread
and bare-root columns follow from the 1-in-10 chance of drawing the informative
column and come out the same however you code it.)

### Key Differences from Random Forest

| Aspect | Isolation Forest | Random Forest |
|--------|------------------|---------------|
| **Goal** | Isolate anomalies | Classify/predict |
| **Splits** | Randomly selected | Optimized (information gain) |
| **Training** | Unsupervised | Supervised |
| **Output** | Anomaly score | Class/value |
| **Depth** | Limited (log n) | Can be deep |

---

## Understanding the Code

Four private helpers carry the whole algorithm. Each one is a single formula
from the Mathematical Foundation above, so you can read them side by side.

| Method | Formula it implements | Where it lives |
|--------|----------------------|----------------|
| `_calculate_c(n)` | c(n) = 2(ln(n-1) + 0.5772) - 2(n-1)/n | the normaliser, and the leaf bonus |
| `_build_tree(...)` | random cut: feature ~ U(splittable), value ~ U(min, max) | grows one isolation tree |
| `_path_length(x, tree)` | h(x) = depth_reached + c(size_of_leaf) | walks one tree for one point |
| `_anomaly_score(h)` | s(x, ψ) = 2^(-E[h(x)] / c(ψ)) | turns paths into 0-1 scores |

**1. `_calculate_c(n)` - the yardstick.**

```python
if n <= 1:    return 0          # already isolated
elif n == 2:  return 1          # two points always take one cut
else:         return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n
```

Line for line the c(ψ) formula, Euler-Mascheroni constant and all.

**2. `_build_tree(...)` - one random cut, then recurse.**
Note what is *absent*: there is no impurity, no gain, no target. The split is
chosen by two coin flips (which column, where in its range), which is why an
isolation tree is so much cheaper than a decision tree.

**3. `_path_length(x, tree)` - the leaf bonus.**

```python
if tree['type'] == 'leaf':
    return current_height + self._calculate_c(tree['size'])
if x[tree['feature']] < tree['split_value']:
    return self._path_length(x, tree['left'], current_height + 1)
else:
    return self._path_length(x, tree['right'], current_height + 1)
```

The first line is `h(x) = depth + c(size)`. The rest is an ordinary binary
descent, counting one edge per level.

**4. `_anomaly_score(path_lengths)` - the exponential squash.**

```python
c = self._calculate_c(max(self.max_samples_, 2))   # c(psi), psi >= 2
scores = np.power(2, -path_lengths / c)            # s = 2^(-E[h]/c(psi))
```

`max(..., 2)` is a guard: `c(1) = 0` would divide by zero and return NaN for
every sample. It is this implementation's own guard, **not** a copy of
scikit-learn's - sklearn leaves the denominator unclamped and guards the
division instead, which makes the two disagree at the single degenerate setting
`max_samples=1`: sklearn scores every point 0.5 there, this code scores every
point 1.0 (both measured). The 2008 paper leaves `c(1)` undefined, so neither
value is the "right" one. For every ψ ≥ 2 - i.e. every fit anyone would
actually run - the clamp is inert and the two agree.

**5. `fit()` - where the parameters become numbers.**
`fit()` resolves ψ and the column budget, computes `height_limit`, grows the
trees, then makes one extra scoring pass over the whole training set to set
`self.threshold`. That last pass is the expensive part of `fit()` in this
pure-Python implementation - it is O(n × n_estimators × depth).

**6. `decision_function()` vs `score_samples()` - watch the sign.**

| Method | Sign convention | Matches sklearn? |
|--------|-----------------|------------------|
| `decision_function(X)` | 0..1, **higher = more anomalous** (the raw paper score) | **No** - sklearn's method of the same name is negative for outliers |
| `score_samples(X)` | negative, more negative = more anomalous | **Yes** - agrees with sklearn's `score_samples` to ~0.005 mean absolute difference |

This is the one place where porting code to scikit-learn will silently invert
your results, so it is worth remembering: here `score > threshold` means
anomaly; in sklearn `decision_function < 0` means anomaly.

### Simplifications vs. canonical Isolation Forest

The mathematics above is the paper's, unmodified - `_calculate_c` matches
scikit-learn's internal `_average_path_length` to 3e-12, and on a 550 x 4
benchmark `score_samples` correlates with sklearn's at r = 0.997 (Spearman
0.988, mean absolute difference 0.0046, 99.6% identical labels). What this
implementation deliberately leaves out:

1. **No `bootstrap` option.** Subsamples are always drawn *without*
   replacement, as in the original paper. scikit-learn offers
   `bootstrap=True` as an alternative; the practical difference is small.
2. **No parallelism (`n_jobs`) and no Cython.** Trees are grown one at a time
   in plain Python, which is what makes it readable and ~50x slower - see
   [Performance Characteristics](#performance-characteristics).
3. **Axis-parallel cuts only.** *Extended* Isolation Forest uses randomly
   oriented hyperplanes instead of single-feature cuts, which removes the
   rectangular "ghost" artefacts a standard iForest leaves in its score map.
   Implementing it means replacing the scalar `split_value` with a random
   normal vector and an offset - a different algorithm, not a tweak.
4. **`decision_function` keeps the paper's sign**, not sklearn's, as above.

---

## Usage Examples

Example 1 below runs as written (verified: it prints exactly the output shown).
Examples 2 and 3 are sketches - they call placeholder helpers
(`generate_normal_transactions`, `generate_fraudulent_transactions`,
`get_latest_metrics`, `alert_admin`) that you would supply yourself;
self-contained, runnable versions of both scenarios are USAGE
EXAMPLE 2 and USAGE EXAMPLE 9 in `_20_isolation_forest.py`.

### Example 1: Basic Anomaly Detection

```python
import numpy as np
from _20_isolation_forest import IsolationForest

# Generate data
np.random.seed(42)
X_normal = np.random.randn(300, 2) * 0.5
X_anomalies = np.random.uniform(-4, 4, (20, 2))
X = np.vstack([X_normal, X_anomalies])
y_true = np.array([1] * 300 + [-1] * 20)      # 1 = normal, -1 = anomaly

# Train model. 20 anomalies out of 320 rows is 6.25%, so pass THAT as
# contamination - leaving it at 0.1 would flag 32 rows and 12 would be
# false alarms. random_state makes the run reproducible.
model = IsolationForest(n_estimators=100, contamination=20 / 320, random_state=42)
model.fit(X)

# Predict
predictions = model.predict(X)
scores = model.decision_function(X)

tp = np.sum((predictions == -1) & (y_true == -1))
fp = np.sum((predictions == -1) & (y_true == 1))
fn = np.sum((predictions == 1) & (y_true == -1))

print(f"Anomalies detected: {np.sum(predictions == -1)}")
print(f"Precision: {tp / (tp + fp):.2%}   Recall: {tp / (tp + fn):.2%}")
print(f"Highest score: {scores.max():.4f}   Threshold: {model.threshold:.4f}")
```

Expected output:
```
Anomalies detected: 20
Precision: 80.00%   Recall: 80.00%
Highest score: 0.7696   Threshold: 0.5410
```

### Example 2: Credit Card Fraud Detection

```python
# Simulate transactions: [amount, time, frequency, ...]
X_normal = generate_normal_transactions(1000)
X_fraud = generate_fraudulent_transactions(50)
X = np.vstack([X_normal, X_fraud])

model = IsolationForest(
    n_estimators=150,
    max_samples=256,
    contamination=0.05  # Expect 5% fraud
)
model.fit(X)

# Flag suspicious transactions
predictions = model.predict(X)
fraud_indices = np.where(predictions == -1)[0]
```

### Example 3: Real-time Monitoring

```python
# Train on historical normal data
model = IsolationForest(contamination=0.01)
model.fit(historical_data)

# fit() has already stored the cut-off implied by contamination:
#   model.threshold = the (1 - contamination) quantile of the training scores
# Use that attribute - there is no bare `threshold` variable to invent.
print(f"Alerting on scores above {model.threshold:.4f}")

# Monitor new data
while True:
    new_sample = get_latest_metrics()
    score = model.decision_function([new_sample])[0]

    if score > model.threshold:
        alert_admin(f"Anomaly detected! Score: {score}")
```

Raising `model.threshold` by hand after fitting is a legitimate way to make a
live system less noisy without retraining: the ranking never changes, only the
alert budget.

---

## Hyperparameters Explained

### n_estimators

**Number of isolation trees to build**

```python
model = IsolationForest(n_estimators=100)
```

- **Higher values**: More stable, accurate, but slower
- **Lower values**: Faster, but less reliable
- **Recommended**: 100-200 for most cases
- **Rule of thumb**: Start with 100, increase if results are unstable

### max_samples

**Number of samples to draw for each tree**

```python
model = IsolationForest(max_samples='auto')  # Uses min(256, n_samples)
model = IsolationForest(max_samples=256)     # Fixed number
model = IsolationForest(max_samples=0.5)     # 50% of data
```

- **'auto'**: Recommended default (256 or less)
- **Smaller values**: Faster training, more randomness
- **Larger values**: More comprehensive but slower
- **Original paper**: Recommends 256 as sweet spot

### contamination

**Expected proportion of anomalies in dataset**

```python
model = IsolationForest(contamination=0.1)  # 10% anomalies
```

- **Higher values**: More samples flagged as anomalies
- **Lower values**: Only the most extreme anomalies flagged
- **Critical**: Should match your domain knowledge
- **Typical range**: 0.01 (1%) to 0.1 (10%)

**How to choose:**
- **Known fraud rate**: Use that rate
- **Unknown**: Start with 0.05-0.1, adjust based on results
- **Imbalanced data**: Use lower values

### max_features

**Number of features to consider for each split**

```python
model = IsolationForest(max_features=1.0)   # Use all features
model = IsolationForest(max_features=0.5)   # Use 50% of features
model = IsolationForest(max_features=3)     # Each tree sees exactly 3 columns
```

- **1.0**: All features (default)
- **< 1.0**: Increases randomness and diversity
- **Use when**: High-dimensional data or correlated features

**Per tree, not per split.** The column subset is drawn once in `fit()` and
then fixed for the whole tree, which is also scikit-learn's behaviour. Drawing
a fresh subset at every node would be marginally identical to using all
columns, and the parameter would do nothing at all. The resolved integer is
stored on `model.max_features_`.

### random_state

**Random seed for reproducibility**

```python
model = IsolationForest(random_state=42)
```

- Set for reproducible results
- Important for debugging and comparison
- Seeds a **private** `np.random.RandomState` stored on the model, so fitting
  never reseeds the global `np.random` stream your own program is using. (A
  common bug in from-scratch estimators is calling `np.random.seed()` inside
  `__init__`, which silently changes every later `np.random.*` call the caller
  makes - and makes "independent" models in a loop identical.)

---

## Advantages & Limitations

### Advantages

1. **Fast Training & Prediction**
   - Linear time complexity: O(n log n)
   - Much faster than distance-based methods

2. **Handles High-Dimensional Data**
   - Works well with many features
   - No distance calculations needed

3. **No Need for Labels**
   - Fully unsupervised
   - No labeled anomalies required

4. **Memory Efficient**
   - Uses subsampling
   - Doesn't store training data

5. **Robust to Normal Data Variations**
   - Focuses on isolating anomalies
   - Not affected by normal data structure

6. **Few Hyperparameters**
   - Mainly need to tune contamination
   - Good default values available

### Limitations

1. **Contamination Parameter Required**
   - Need to estimate proportion of anomalies
   - Wrong estimate affects performance

2. **Struggles with Clustered Anomalies**
   - If anomalies form their own cluster
   - May appear "normal" to the algorithm

3. **Random Behavior**
   - Results can vary between runs
   - Need multiple trees for stability

4. **No Feature Importance**
   - Doesn't directly tell which features indicate anomaly
   - Harder to interpret than some methods
   - More precisely: no *built-in* importance, because splits are random and
     there is no impurity gain to accumulate. Permutation importance works
     fine as a wrapper (see USAGE EXAMPLE 5 in `_20_isolation_forest.py`) -
     measure the score drop **on the flagged rows**, not averaged over the
     whole dataset, or the noise columns will win

5. **Not Ideal for Streaming Data**
   - Need to retrain for concept drift
   - Online learning not straightforward

6. **Boundary Cases**
   - Samples near decision boundary can be unstable
   - May classify differently between runs

---

## Comparison with Other Anomaly Detection Methods

### Isolation Forest vs One-Class SVM

| Aspect | Isolation Forest | One-Class SVM |
|--------|------------------|---------------|
| **Speed** | Fast (linear) | Slow (quadratic) |
| **Scalability** | Excellent | Poor for large datasets |
| **High dimensions** | Good | Struggles |
| **Interpretability** | Medium | Low |
| **Parameters** | Few | Several (kernel, nu, gamma) |

### Isolation Forest vs LOF (Local Outlier Factor)

| Aspect | Isolation Forest | LOF |
|--------|------------------|-----|
| **Approach** | Isolation-based | Density-based |
| **Speed** | Faster | Slower |
| **Memory** | Lower | Higher |
| **Local anomalies** | Good | Excellent |
| **Global anomalies** | Excellent | Good |

### Isolation Forest vs Statistical Methods

| Aspect | Isolation Forest | Statistical |
|--------|------------------|-------------|
| **Assumptions** | Minimal | Strong (distribution) |
| **Multivariate** | Native | Complex |
| **Robustness** | High | Varies |
| **Interpretability** | Medium | High |

### When to Use Each

- **Isolation Forest**: Large datasets, high dimensions, speed critical
- **One-Class SVM**: Small datasets, complex boundaries
- **LOF**: Local anomalies important, have resources
- **Statistical**: Well-understood distributions, need interpretability

---

## Tips & Best Practices

The snippets in this section are fragments to read rather than programs to run:
several of them reference names this numpy-only module does not provide (`plt`,
`pd`, `StandardScaler`, `LabelEncoder`, `ks_test`, `lof`, `flag_as_anomaly`).

### 1. Choosing Contamination

```python
# Strategy 1: Domain knowledge
if fraud_rate_known:
    contamination = fraud_rate

# Strategy 2: Visual inspection
scores = model.decision_function(X)
plt.hist(scores, bins=50)
# Look for natural gap, choose contamination accordingly

# Strategy 3: Cross-validation
for contam in [0.01, 0.05, 0.1, 0.15]:
    model = IsolationForest(contamination=contam)
    # Evaluate on validation set with known labels
```

### 2. Dealing with Class Imbalance

```python
# If anomalies are < 1%
model = IsolationForest(
    contamination=0.001,  # Very low
    n_estimators=200      # More trees for stability
)
```

### 3. Feature Scaling

```python
# Isolation Forest doesn't require scaling
# But it can help in some cases

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest()
model.fit(X_scaled)
```

### 4. Handling Categorical Features

```python
# Option 1: One-hot encoding
X_encoded = pd.get_dummies(X, columns=['category_col'])

# Option 2: Label encoding (preserves memory)
from sklearn.preprocessing import LabelEncoder
X['category_col'] = LabelEncoder().fit_transform(X['category_col'])
```

### 5. Monitoring Performance Over Time

```python
# For production systems
def monitor_drift():
    # Train on baseline
    baseline_scores = model.decision_function(X_baseline)
    
    # Check new data periodically
    new_scores = model.decision_function(X_new)
    
    # Compare distributions
    if ks_test(baseline_scores, new_scores).pvalue < 0.05:
        print("Distribution shift detected! Retrain model.")
```

### 6. Combining with Other Methods

```python
# Ensemble of anomaly detectors
from sklearn.ensemble import VotingClassifier

# Vote: anomaly if both agree
if iforest.predict(x) == -1 and lof.predict(x) == -1:
    flag_as_anomaly(x)
```

### 7. Explaining Anomalies

```python
# Find which features contribute most to anomaly
def explain_anomaly(model, X, sample_idx):
    baseline_score = model.decision_function([X[sample_idx]])[0]
    
    contributions = {}
    for feature_idx in range(X.shape[1]):
        # Replace with mean
        X_modified = X[sample_idx].copy()
        X_modified[feature_idx] = np.mean(X[:, feature_idx])
        
        modified_score = model.decision_function([X_modified])[0]
        contributions[feature_idx] = baseline_score - modified_score
    
    return contributions
```

### 8. Hyperparameter Tuning

This class is a from-scratch implementation, **not** a scikit-learn estimator -
it has no `get_params`/`set_params`, so `GridSearchCV` cannot clone it
(`TypeError: Cannot clone object ...`). Loop over the grid yourself; it is three
lines and you keep full control of the scoring:

```python
import itertools

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_samples': [128, 256],
    'contamination': [0.05, 0.1, 0.15]
}

# Note: Need labeled validation data for this (1 = normal, -1 = anomaly)
best_score, best_params = -1, None
keys = list(param_grid)

for values in itertools.product(*(param_grid[k] for k in keys)):
    params = dict(zip(keys, values))
    model = IsolationForest(random_state=42, **params).fit(X_train)

    # score(X, y) returns accuracy when y is given
    acc = model.score(X_val, y_val)
    if acc > best_score:
        best_score, best_params = acc, params

print(f"Best accuracy {best_score:.2%} with {best_params}")
best_model = IsolationForest(random_state=42, **best_params).fit(X_train)
```

Accuracy is a blunt metric when anomalies are 1% of the data (always predicting
"normal" scores 99%). If you have labels, ranking metrics such as ROC AUC or
precision@k computed from `decision_function()` are usually the better target.

---

## Performance Characteristics

### Time Complexity

- **Training**: O(t × ψ × log ψ × d)
  - t = number of trees
  - ψ = subsample size
  - d = number of features

- **Prediction**: O(t × log ψ × d)

### Space Complexity

- O(t × ψ)
- Much lower than methods that store all training data

### Typical Runtimes

**This pure-Python implementation** (measured on the machine this guide was
written on, Python 3.13 + NumPy 2.3, `n_estimators=100`):

| Dataset Size | Features | Trees | `fit()` time (measured) |
|--------------|----------|-------|-------------------------|
| 1,000 | 10 | 100 | 0.4 sec |
| 5,000 | 10 | 100 | 1.4 sec |
| 10,000 | 50 | 100 | 2.7 sec |
| 20,000 | 20 | 100 | 5.3 sec |
| 100,000 | 100 | 100 | 28 sec |

Note that `fit()` is *not* dominated by growing the trees - each tree only ever
sees ψ = 256 rows, so tree construction barely grows with n. It is dominated by
the extra pass at the end of `fit()` that scores **every** training row to set
`self.threshold`, which costs O(n × n_estimators × depth) Python-level
recursive calls. Scoring 20,000 rows through 100 trees is 2 million
`_path_length` calls, and that single pass is why the times above grow almost
exactly linearly in n while barely noticing the feature count.

**Reference C/Cython implementation** (scikit-learn 1.7.2, same data and
settings) fits that 20,000 x 20 case in **0.11 sec** - roughly 50x faster.
Use it in production; use this one to understand what it is doing.

---

## 🎓 Further Learning

### Original Paper
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
- [Link to paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

### Key Concepts to Understand
1. Binary Search Trees
2. Ensemble Learning
3. Anomaly Detection Fundamentals
4. Subsampling and Bootstrap

### Related Algorithms
- Extended Isolation Forest (improvements on split selection)
- Deep Isolation Forest (neural network version)
- Robust Random Cut Forest (streaming version)

---

## 🔗 Quick Reference

### Import and Basic Usage
```python
from _20_isolation_forest import IsolationForest

model = IsolationForest()
model.fit(X_train)
predictions = model.predict(X_test)
```

### Key Methods
- `fit(X)`: Train the model
  - full signature is `fit(X, y=None)`; `y` is accepted and ignored (unsupervised)
- `predict(X)`: Return -1 for anomaly, 1 for normal
- `decision_function(X)`: Return anomaly scores (0-1)
  - **higher = more anomalous** (the raw paper score s(x, ψ)); sklearn's method
    of the same name uses the opposite sign
- `score_samples(X)`: Return negative anomaly scores
  - **this** is the one that matches sklearn's `score_samples`
- `score(X, y=None)`: accuracy against labels (1 normal / -1 anomaly) when `y`
  is given; the mean anomaly score otherwise

### Key Attributes After Fitting
- `model.trees`: List of isolation trees
- `model.threshold`: Anomaly score threshold
  - specifically the (1 - contamination) quantile of the training scores
- `model.max_samples_`: Actual subsample size used
  - this is ψ, the value fed to c(ψ)
- `model.max_features_`: Actual number of columns each tree may split on

---

## Summary

Isolation Forest is a powerful, efficient anomaly detection algorithm that:
- Works by **isolating anomalies** rather than profiling normal data
- Is **fast and scalable** to large datasets
- Requires **minimal assumptions** about data distribution
- Needs careful tuning of the **contamination parameter**

**Best for**: Large-scale anomaly detection where speed matters and you have a rough idea of anomaly proportion.

---

**Happy Anomaly Hunting!** 🔍🎯
