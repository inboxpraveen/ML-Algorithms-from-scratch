# AdaBoost from Scratch: A Comprehensive Guide

Welcome to the world of Ensemble Learning! 🚀 In this comprehensive guide, we'll explore AdaBoost (Adaptive Boosting) - one of the most powerful and elegant boosting algorithms. Think of it as combining the wisdom of many "weak" experts to make incredibly strong predictions!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is AdaBoost?](#what-is-adaboost)
3. [How AdaBoost Works](#how-adaboost-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Computational Complexity](#computational-complexity)
11. [Advantages and Limitations](#advantages-and-limitations)
12. [Comparing with Alternatives](#comparing-with-alternatives)
13. [Key Concepts to Remember](#key-concepts-to-remember)
14. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# AdaBoost from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _15_adaboost.py  (the __main__ block runs a fuller version)
# Or copy the AdaBoost class from _15_adaboost.py and paste it below.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the AdaBoost class here (from _15_adaboost.py) ----
# class AdaBoost: ...

np.random.seed(42)

# ------ Two Gaussian blobs. Labels MUST be -1 / +1. ------
X0 = np.random.randn(100, 2) + np.array([-2, -2])
X1 = np.random.randn(100, 2) + np.array([ 2,  2])
X = np.vstack([X0, X1])
y = np.array([-1] * 100 + [1] * 100)   # fit() raises ValueError on 0/1 labels

# Shuffle before splitting: the rows were stacked class-by-class, so an
# unshuffled X[:150] would be almost entirely class -1.
idx = np.random.permutation(200)
X, y = X[idx], y[idx]

# Non-overlapping split. X[:150], X[50:] would leak 100 training rows.
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

model = AdaBoost(n_estimators=50, learning_rate=1.0)
model.fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.2%}")
print(f"Test  Accuracy: {model.score(X_test,  y_test):.2%}")

preds = model.predict(X_test)         # -1 / +1
proba = model.predict_proba(X_test)   # P(class = +1), 1-D array
for i in range(3):
    print(f"  true={y_test[i]:+d}  pred={preds[i]:+.0f}  P(+1)={proba[i]:.4f}")

# ------ Does boosting actually beat one stump? Ring vs. disk. ------
Xr = np.random.randn(250, 2) * 2
yr = np.where(Xr[:, 0] ** 2 + Xr[:, 1] ** 2 > 4, 1, -1)
idx = np.random.permutation(250)
Xr, yr = Xr[idx], yr[idx]

one = AdaBoost(n_estimators=1).fit(Xr[:190], yr[:190])
many = AdaBoost(n_estimators=50).fit(Xr[:190], yr[:190])
print(f"\nSingle stump      test accuracy: {one.score(Xr[190:], yr[190:]):.2%}")
print(f"AdaBoost 50 stumps test accuracy: {many.score(Xr[190:], yr[190:]):.2%}")

# Watch the ensemble improve, learner by learner
staged = many.staged_score(Xr[190:], yr[190:])
print("Test accuracy after 1, 5, 10, 25, 50 learners:",
      [f"{staged[k - 1]:.2%}" for k in (1, 5, 10, 25, 50)])

print("Feature importance:", np.round(many.get_feature_importance(), 4))
```

Expected output:
```
Train Accuracy: 100.00%
Test  Accuracy: 100.00%
  true=+1  pred=+1  P(+1)=0.8808
  true=-1  pred=-1  P(+1)=0.1192
  true=+1  pred=+1  P(+1)=0.8808

Single stump      test accuracy: 61.67%
AdaBoost 50 stumps test accuracy: 93.33%
Test accuracy after 1, 5, 10, 25, 50 learners: ['61.67%', '76.67%', '76.67%', '95.00%', '93.33%']
Feature importance: [0.4632 0.5368]
```

Two things worth noticing straight away:

- **`P(+1)` tops out at 0.8808, not 1.0.** That is not a bug. `predict_proba` maps the weighted vote through `1 / (1 + exp(-2 * F(x) / sum|alpha|))`, and the exponent is bounded by +/-2, so the output lives in `[0.1192, 0.8808]`. scikit-learn's SAMME returns exactly the same bound. Read it as a monotone confidence score, not a calibrated probability.
- **One stump gets 61.67%, fifty get 93.33%.** The ring-vs-disk boundary cannot be drawn with one axis-aligned cut, but a *weighted sum* of axis-aligned cuts approximates it well. That is boosting doing its job.

---

## What is AdaBoost?

AdaBoost (Adaptive Boosting) is an **ensemble learning algorithm** that combines multiple weak classifiers to create a strong classifier. It was one of the first successful boosting algorithms and remains widely used today.

**Real-world analogy**: 
Imagine you're trying to diagnose a complex medical case. Instead of relying on one junior doctor, you consult many junior doctors, but you pay more attention to those who have been right before. Each doctor focuses on the cases the previous doctors got wrong. Together, they make better diagnoses than any senior doctor alone!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Ensemble Learning (Boosting) |
| **Learning Style** | Supervised Learning |
| **Primary Use** | Classification (also regression variant exists) |
| **Base Learners** | Weak classifiers (typically decision stumps) |
| **Key Principle** | Sequential learning with focus on mistakes |

### The Core Idea

```
"Focus on mistakes from previous learners and combine weak learners into a strong one"
```

This principle works through:
- **Sequential training**: Each new learner focuses on examples misclassified by previous learners
- **Weighted voting**: Better learners get more say in the final decision
- **Adaptive**: Sample weights adapt based on performance

### Key Concepts

**1. Weak Learner**: A classifier slightly better than random guessing
```
Example: A decision stump (1-level decision tree)
         Just asks one question: "Is feature X > threshold?"
         Accuracy: 51-60% (barely better than 50% random)
```

**2. Sample Weights**: How much attention to pay to each training example
```
Initially: All samples have equal weight (1/N)
After training: Misclassified samples get higher weights
Result: Next learner focuses more on hard examples
```

**3. Learner Weight (Alpha)**: How much to trust each weak learner
```
α = 0.5 × ln((1 - error) / error)

High alpha: Low error → Trust this learner more
Low alpha: High error → Trust this learner less
```

**4. Final Prediction**: Weighted majority vote
```
Final(x) = sign(α₁·h₁(x) + α₂·h₂(x) + ... + αₜ·hₜ(x))
           where hₜ(x) is prediction of weak learner t
```

---

## How AdaBoost Works

### The Algorithm in 5 Steps

```
Step 1: Initialize all sample weights equally
         ↓
Step 2: Train a weak learner on weighted data
         ↓
Step 3: Calculate learner's error and weight (alpha)
         ↓
Step 4: Update sample weights (increase for misclassified)
         ↓
Step 5: Repeat Steps 2-4 for T iterations
         ↓
Final: Combine all learners with weighted voting
```

### Visual Example

Let's classify circles (O) vs. crosses (X):

```
Dataset: 10 samples

O O O X X
O O X X X

Initial weights: all equal (0.1 each)
```

**Round 1: Train first weak learner**

```
Weak Learner 1 finds boundary:
    |
O O | O X X
O O | X X X
    |
    
Mistakes: 2 samples (marked with *)
O O  O* X  X
O O  X* X  X

Error = 2/10 = 0.2 (20%)
Alpha₁ = 0.5 × ln((1-0.2)/0.2) = 0.69
```

**Round 2: Update weights and train second learner**

```
Update weights:
- Correct predictions: weight × e^(-0.69) = weight × 0.5
- Wrong predictions: weight × e^(0.69) = weight × 2.0

New weights (larger circles = higher weight):
o o O● x x
o o X● x x

Weak Learner 2 focuses on mistakes:
  |
o | o O● x x
o | o X● x x
  |

This learner focuses on the previously misclassified samples!
```

**Final: Combine learners**

```
Final Classifier = α₁ × Learner₁ + α₂ × Learner₂ + ...

For new sample at position (2, 1):
  Learner 1 says: X (cross)    weight: 0.69
  Learner 2 says: O (circle)   weight: 0.42
  Learner 3 says: X (cross)    weight: 0.55
  
  Total vote: 0.69 + 0.55 - 0.42 = 0.82 > 0
  → Predict: X (cross)
```

### Why Sequential Learning Works

**Traditional ensemble (Random Forest)**:
```
Train all models independently in parallel
Learner 1: Looks at random subset → 60% accuracy
Learner 2: Looks at random subset → 60% accuracy
Learner 3: Looks at random subset → 60% accuracy
Combined: ~65% accuracy
```

**AdaBoost's sequential approach**:
```
Learner 1: Learns easy patterns → 60% accuracy
           ↓ (finds 40% hard cases)
Learner 2: Specializes in Learner 1's mistakes → 55% accuracy on hard cases
           ↓ (finds even harder cases)
Learner 3: Specializes in remaining mistakes → 52% accuracy on very hard cases
Combined: ~85% accuracy!
```

**The Magic**: Each learner specializes in different types of mistakes, creating complementary expertise!

---

## The Mathematical Foundation

### 1. Sample Weights Initialization

At the start, all samples have equal importance:

```
w₁(i) = 1/N    for all i = 1, 2, ..., N

where:
  - N = number of training samples
  - w₁(i) = initial weight for sample i
```

**Example**:
```
10 training samples
Initial weights: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
Sum = 1.0 (weights are normalized)
```

### 2. Training Weak Learner

For round t, train weak learner hₜ on weighted data:

```
hₜ: X → {-1, +1}

The learner minimizes weighted error:
error_t = Σ wₜ(i) × I[hₜ(xᵢ) ≠ yᵢ]
         i=1

where:
  - I[condition] = 1 if condition true, 0 otherwise
  - yᵢ ∈ {-1, +1} is true label
  - wₜ(i) is current weight of sample i
```

**Example**:
```
Predictions:  [+1, +1, -1, +1, -1, +1, -1, +1, -1, -1]
True labels:  [+1, +1, -1, +1, -1, -1, +1, +1, -1, -1]
Matches:      [ ✓,  ✓,  ✓,  ✓,  ✓,  ✗,  ✗,  ✓,  ✓,  ✓ ]
Weights:      [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]

Weighted error = 0.1 + 0.1 = 0.2 (20%)
```

### 3. Calculate Learner Weight (Alpha)

The weight αₜ represents how much to trust learner t:

```
αₜ = 0.5 × ln((1 - εₜ) / εₜ)

where:
  - εₜ = weighted error of learner t (0 < εₜ < 0.5)
  - ln = natural logarithm
```

**Interpretation**:

```
Error  │  Alpha  │  Interpretation
───────┼─────────┼──────────────────────
0.50   │  0.00   │  Random guessing → no trust
0.40   │  0.20   │  Slightly better → some trust
0.30   │  0.42   │  Decent → moderate trust
0.20   │  0.69   │  Good → high trust
0.10   │  1.10   │  Excellent → very high trust
0.05   │  1.47   │  Near perfect → maximum trust
```

**Why this formula?**

```
As error → 0:   alpha → +∞  (perfect classifier, infinite trust)
As error → 0.5: alpha → 0   (random, no trust)
As error → 1:   alpha → -∞  (opposite classifier, negative trust)
```

**Example**:
```
Learner with 20% error:
α = 0.5 × ln((1 - 0.2) / 0.2)
  = 0.5 × ln(0.8 / 0.2)
  = 0.5 × ln(4)
  = 0.5 × 1.386
  = 0.693
```

### 4. Update Sample Weights

After each round, update weights to focus on mistakes:

```
wₜ₊₁(i) = wₜ(i) × exp(-αₜ × yᵢ × hₜ(xᵢ))

Then normalize: wₜ₊₁(i) = wₜ₊₁(i) / Σⱼ wₜ₊₁(j)

Simplified (because yᵢ and hₜ(xᵢ) are both ±1, their product is +1 when the
learner is right and -1 when it is wrong):
  - If correctly classified: wₜ₊₁(i) = wₜ(i) × e^(-αₜ)  (decrease)
  - If misclassified:       wₜ₊₁(i) = wₜ(i) × e^(αₜ)   (increase)
```

> **This is the one equation to get right.** A common mistake is to write
> `wₜ₊₁(i) = wₜ(i) × exp(αₜ × I[hₜ(xᵢ) ≠ yᵢ])`, which leaves correct samples
> untouched instead of shrinking them. The two rules differ by a square root:
> the correct one changes the wrong-to-right weight ratio by **e^(2αₜ) = (1-εₜ)/εₜ**,
> the mistaken one only by **e^(αₜ) = √((1-εₜ)/εₜ)**. For εₜ = 0.2 that is a
> **4x** re-weighting versus a 2x one.
>
> Why it matters: only the correct rule makes hₜ's weighted error under the
> **new** weights come out to exactly 0.5. That is AdaBoost's defining
> invariant, and it is what stops round t+1 from simply re-selecting the same
> stump. With the mistaken rule the measured errors drift (0.33, 0.39, 0.43, ...
> instead of 0.50) and the ensemble wastes rounds re-picking learners it
> already has.

**Example** (a 4-sample dataset, so εₜ and αₜ must be derived from *these* four
samples — do not reuse the 0.693 from the 10-sample illustration above):
```
Current weights:  [0.25, 0.25, 0.25, 0.25]   (4 samples, uniform, sum = 1.0)
Predictions:      [ ✓,    ✗,    ✓,    ✓   ]

ε = 0.25 (one wrong sample carrying weight 0.25)
α = 0.5 × ln((1 - 0.25) / 0.25) = 0.5 × ln(3) = 0.549
    e^(-0.549) = 0.5774      e^(+0.549) = 1.7321

After update:
  Sample 0: 0.25 × 0.5774 = 0.1443  (correct, reduced)
  Sample 1: 0.25 × 1.7321 = 0.4330  (wrong, increased)
  Sample 2: 0.25 × 0.5774 = 0.1443  (correct, reduced)
  Sample 3: 0.25 × 0.5774 = 0.1443  (correct, reduced)

Before normalization: [0.1443, 0.4330, 0.1443, 0.1443]  Sum = 0.8660
After normalization:  [0.1667, 0.5000, 0.1667, 0.1667]  Sum = 1.0

→ Next learner focuses 50% attention on the misclassified sample!
```

That 50% is not a coincidence — it *is* the invariant from the box above. This
learner's weighted error under the new weights is exactly 0.500, so it is now no
better than a coin flip on the re-weighted data. Whenever you work an example by
hand, computing α from the same ε you use for the mistakes is what makes the
check come out right.

**Why exponential?**

```
Exponential magnifies differences:
- Large alpha (good learner) → Large weight changes
- Small alpha (weak learner) → Small weight changes

This creates strong adaptive focus!
```

### 5. Final Prediction

Combine all learners with weighted voting:

```
H(x) = sign(Σ αₜ × hₜ(x))
            t=1

where:
  - T = number of weak learners
  - αₜ = weight of learner t
  - hₜ(x) ∈ {-1, +1} = prediction of learner t
  - sign(z) = +1 if z ≥ 0, else -1
```

**A note on ties**: `np.sign(0)` is `0`, which is not a class label. `predict()`
therefore uses `np.where(weighted_sum >= 0, 1, -1)`, breaking an exact tie
toward +1 so the output only ever contains valid labels.

**Example**:
```
3 learners making predictions:

For test sample x:
  Learner 1: predicts +1, weight α₁ = 0.693
  Learner 2: predicts -1, weight α₂ = 0.420
  Learner 3: predicts +1, weight α₃ = 0.549

Weighted sum = 0.693×(+1) + 0.420×(-1) + 0.549×(+1)
             = 0.693 - 0.420 + 0.549
             = 0.822

sign(0.822) = +1

Final prediction: +1 (positive class)
```

### 6. Training Error Bound

**Theoretical Guarantee**: AdaBoost's training error decreases exponentially!

```
Training Error ≤ exp(-2 Σ γₜ²)
                      t=1

where γₜ = 0.5 - εₜ is the "margin" by which learner t beats random guessing

Key insight: Even if each weak learner is only slightly better than random,
            their combination can achieve very low error!
```

**Example**:
```
10 weak learners, each with 40% error (60% accuracy):
  γ = 0.5 - 0.4 = 0.1 (10% better than random)

Training error bound:
  ≤ exp(-2 × 10 × 0.1²)
  = exp(-0.2)
  = 0.819

But typically much lower due to adaptation!
Empirical training error often < 1%
```

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alphas = []          # one alpha per fitted weak learner
        self.weak_learners = []   # one stump dict per fitted weak learner
        self.n_features = None    # set by fit(); doubles as the "is fitted?" flag
```

### Core Methods

1. **`__init__(n_estimators, learning_rate)`** - Initialize model
   - n_estimators: Number of weak learners to train
   - learning_rate: Shrinks contribution of each classifier

2. **`_create_decision_stump(X, y, weights)`** - Create weak learner
   - Searches every feature and every candidate threshold
   - Tries **both polarities** at each threshold, so the returned error is always ≤ 0.5
   - Returns `(stump_dict, error)` where `stump_dict` has the keys
     `'feature'`, `'threshold'`, `'left_prediction'`, `'right_prediction'`

3. **`_stump_predict(stump, X)`** - Predict with stump
   - Apply threshold rule to make predictions

4. **`fit(X, y)`** - Train AdaBoost ensemble
   - Main algorithm implementation
   - Iteratively trains weak learners
   - Updates sample weights adaptively
   - Stops early if a stump classifies every training sample correctly

5. **`predict(X)`** - Make predictions
   - Combines all weak learners
   - Returns weighted majority vote (-1 / +1)

6. **`predict_proba(X)`** - Predict probabilities
   - Returns P(class = +1) as a 1-D array, bounded to about [0.12, 0.88]
   - Based on weighted sum of learners

7. **`score(X, y)`** - Calculate accuracy
   - Returns fraction of correct predictions (accuracy, not R²)

8. **`get_feature_importance()`** - Feature importance
   - Which features are most useful
   - Sums `abs(alpha)` per feature, then normalizes to sum to 1

9. **`staged_score(X, y)`** - Learning curve
   - Accuracy after each learner
   - Shows improvement over iterations

10. **`print_learners(max_display=10)`** - Inspect the ensemble
    - Prints a table of the fitted stumps: index, feature, threshold, alpha, and polarity
    - The `L->R` column reads `-1->+1` for "predict -1 at or below the threshold, +1 above"

### Label convention (read this first)

`fit()` **requires labels in {-1, +1}** and rejects anything else:

```python
model.fit(X, np.array([0, 1, 1, 0]))
# ValueError: Labels must be -1 or +1. Got: [0 1]
```

This is the first wall most people hit coming from scikit-learn, which happily
accepts `0/1`. The conversion is one line:

```python
y = np.where(y == 0, -1, 1)
```

The ±1 convention is not arbitrary bookkeeping — the whole derivation depends on
it. The product `yᵢ × hₜ(xᵢ)` is +1 exactly when the learner is right and -1
when it is wrong, which is what lets a single `exp(-αₜ × yᵢ × hₜ(xᵢ))` express
both the "shrink" and the "grow" branch of the weight update.

### Simplification vs. canonical AdaBoost

This implementation is deliberately the classic **binary discrete AdaBoost**
(Freund & Schapire 1997, the SAMME special case for two classes). What a
production library adds, and what it costs you here:

| Canonical feature | Status here | Consequence |
|---|---|---|
| Multi-class via SAMME (`alpha = ln((1-e)/e) + ln(K-1)`) | **Not implemented** | Binary problems only. For K classes use one-vs-rest around this class, or sklearn. |
| SAMME.R (real-valued, uses class probabilities) | **Not implemented** | Converges in fewer rounds in sklearn ≤1.5; removed in newer sklearn, so little is lost. |
| AdaBoost.R2 for regression | **Not implemented** | Classification only. |
| Arbitrary base estimator (`estimator=` in sklearn) | **Not implemented** | Depth-1 stumps only. Fine for teaching; deeper trees help on problems that need feature interactions. |
| Per-sample `sample_weight` passed to `fit` | **Not implemented** | Initial weights are always uniform, `1/N`. |
| `alpha = ln((1-e)/e)` (sklearn SAMME) | Uses `0.5 × ln((1-e)/e)` | Freund-Schapire convention. Every stored alpha is exactly **half** of sklearn's. Predictions are identical, because `predict` only takes the *sign* of the weighted sum and `predict_proba` divides by `Σ|α|`. |
| Per-round sample-weight floor at machine epsilon (sklearn issue #20320) | **Not implemented** | Did not bite in any configuration measured inside the documented `learning_rate` range of 0.1–1.0 (three datasets × `learning_rate` ∈ {0.1, 0.5, 1.0}, `n_estimators=100`: no α ever reached the clip). Above that range the weights collapse until α pins at the value the `1e-10` error clip implies. On `make_classification(n_samples=150, n_features=4, random_state=3)` with `n_estimators=50` the pinning starts at `learning_rate=2.0` (18 of 50 alphas pinned, train accuracy still 100%) and is ruinous at `learning_rate=5.0`: train accuracy **96.67% after the first stump → 4.67% after all 50**, 48 of the 50 alphas pinned at 57.5646. sklearn's `AdaBoostClassifier` with depth-1 stumps at the same `learning_rate` and `n_estimators` scores 96.67%. |

Everything else — the stump search over both polarities, the alpha formula, the
weight update, the early stop on a perfect learner, the SAMME probability
transform — matches the reference algorithm.

---

## Step-by-Step Example

Let's walk through a complete example of **binary classification**:

### The Data

```python
import numpy as np

# Simple 2D dataset: classify red vs blue points
# Feature 1: X-coordinate, Feature 2: Y-coordinate
X = np.array([
    [1, 2], [2, 3], [3, 3], [4, 5],  # Class -1 (blue)
    [5, 1], [6, 2], [7, 2], [8, 1]   # Class +1 (red)
])

y = np.array([-1, -1, -1, -1, +1, +1, +1, +1])

# 8 samples, 2 features
```

### Training the Model

```python
# Paste the AdaBoost class from _15_adaboost.py above this line,
# or run that file directly with `python _15_adaboost.py`.

# Create AdaBoost with 3 weak learners
model = AdaBoost(n_estimators=3)

# Train the model
model.fit(X, y)

print(f"Learners actually fitted: {len(model.alphas)}")   # 1
print(f"Alphas: {[round(a, 4) for a in model.alphas]}")   # [11.5129]
print(f"Stump : {model.weak_learners[0]}")
# {'feature': 0, 'threshold': 4.5, 'left_prediction': -1, 'right_prediction': 1}
```

**What happens internally - Round 1**:

```
Initial weights: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
(all equal, sum = 1.0)

Find best split:
  Feature 0 has distinct values 1,2,3,4,5,6,7,8, so the candidate thresholds
  are the MIDPOINTS 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5.

  Feature 0, threshold 4.5:
    Samples [0,1,2,3] → predict -1 (correct!)
    Samples [4,5,6,7] → predict +1 (correct!)
    
  Weighted error = 0.0 (perfect split!)
  
But wait! Alpha needs ln((1-ε)/ε), and ε = 0 makes that infinite.
The code clips: np.clip(0.0, 1e-10, 1 - 1e-10) = 1e-10

Alpha₁ = 0.5 × ln((1 - 1e-10) / 1e-10) = 11.51

Decision Stump 1:
  if feature_0 ≤ 4.5: predict -1
  else: predict +1
```

```
Update weights:
  All samples classified correctly
  All weights × e^(-11.51) = weights × 1e-5
  After normalization: all still equal [0.125, 0.125, ...]

A perfect stump leaves nothing to boost, so fit() STOPS EARLY:
1 learner is fitted even though n_estimators=3 was requested.
(scikit-learn's AdaBoostClassifier does exactly the same.)
```

**Round 2**: *hypothetical* — this is what the next round would look like if the
data were **not** perfectly separable. The 8-sample dataset above never reaches
it; the numbers below imagine a first learner that got samples 2 and 5 wrong
(2 of 8, so ε = 0.25 and α = 0.549).

```
Suppose learner 1 made mistakes on samples 2 and 5.

Its weighted error is ε = 0.125 + 0.125 = 0.25, so
  α = 0.5 × ln((1 - 0.25) / 0.25) = 0.5 × ln(3) = 0.549
  e^(+0.549) = 1.732      e^(-0.549) = 0.577

Update weights  (w × e^(-α·y·h), i.e. shrink the 6 right, grow the 2 wrong):
  Sample 2: 0.125 × 1.732 = 0.2165  (increased!)
  Sample 5: 0.125 × 1.732 = 0.2165  (increased!)
  Others:   0.125 × 0.577 = 0.0722  (decreased)

Before normalization: [0.0722, 0.0722, 0.2165, 0.0722, 0.0722, 0.2165, 0.0722, 0.0722]
Sum = 6 × 0.0722 + 2 × 0.2165 = 0.866

After normalization: [0.083, 0.083, 0.250, 0.083, 0.083, 0.250, 0.083, 0.083]
Sum = 1.000  (a normalized weight vector must sum to exactly 1 - check it)

THE INVARIANT, visible: learner 1's weighted error under these NEW weights is
  0.250 + 0.250 = 0.500  exactly.
Learner 1 is now no better than a coin flip on the re-weighted data, so
learner 2 has no incentive to copy it. That is the whole trick.

Learner 2 focuses on samples 2 and 5!
Finds different split optimized for these hard cases
```

**Round 3**: More fine-tuning (still hypothetical)

```
Each learner specializes:
  Learner 1: General patterns  (alpha: 0.549, from the eps = 0.25 above)
  Learner 2: Previous mistakes (alpha: 0.470)
  Learner 3: Remaining errors  (alpha: 0.420)

Alphas need not shrink like this - see the worked run at the end of this
section, where they RISE (0.97 -> 1.28 -> 1.61) because each re-weighted
problem turned out easier than the last.
```

### Making Predictions

```python
# Test sample
X_test = np.array([[4, 3]])

# Get prediction
prediction = model.predict(X_test)
print(f"Prediction: {prediction[0]}")   # -1.0

# Get confidence = P(class = +1)
proba = model.predict_proba(X_test)
print(f"P(+1): {proba[0]:.4f}")         # 0.1192
```

**Internal calculation**:

```
For test point [4, 3], with the ONE learner that was actually fitted:

Learner 1 (feature_0 ≤ 4.5 → -1): predicts -1, alpha = 11.5129

Weighted sum F(x) = 11.5129 × (-1) = -11.5129

sign(-11.5129) = -1
Final prediction: -1 (blue class)

P(+1) = 1 / (1 + exp(-2 × F(x) / Σ|α|))
      = 1 / (1 + exp(-2 × (-11.5129) / 11.5129))
      = 1 / (1 + exp(2))
      = 0.1192

Read that as "confidently -1". Because Σ|α| normalizes the exponent, a
single-learner ensemble always lands on the extreme of the achievable range,
[0.1192, 0.8808] - the same bound scikit-learn's SAMME reports.
```

### Model Evaluation

```python
# Check accuracy
train_accuracy = model.score(X, y)
print(f"Training Accuracy: {train_accuracy:.2%}")

# See learning progress
staged_scores = model.staged_score(X, y)
for i, acc in enumerate(staged_scores, 1):
    print(f"After {i} learner(s): {acc:.2%}")

# Output:
# Training Accuracy: 100.00%
# After 1 learner(s): 100.00%
```

Only **one** line, and it is already at 100%. That is the honest result for this
dataset, and it is worth understanding rather than hiding: the 8 points are
perfectly separable by `feature_0 ≤ 4.5`, the very first stump finds that split,
its weighted error is 0, and `fit()` stops early. Requesting `n_estimators=3`
does not force three learners when one is already perfect.

To see a real multi-step learning curve you need data a single stump *cannot*
solve — flip one label and re-run:

```python
y_hard = y.copy()
y_hard[5] = -1          # sample [6, 2] is now a mislabeled point inside the +1 cluster

model = AdaBoost(n_estimators=3)
model.fit(X, y_hard)

print(f"Learners fitted: {len(model.alphas)}")
print([f"{a:.4f}" for a in model.alphas])
for i, acc in enumerate(model.staged_score(X, y_hard), 1):
    print(f"After {i} learner(s): {acc:.2%}")

# Output:
# Learners fitted: 3
# ['0.9730', '1.2825', '1.6094']
# After 1 learner(s): 87.50%
# After 2 learner(s): 87.50%
# After 3 learner(s): 100.00%
```

Now the numbers tell a real story. Learner 1 gets 7 of 8 right, so
ε = 1/8 = 0.125 and α = 0.5 × ln(0.875 / 0.125) = 0.5 × ln(7) = **0.973** —
exactly the first value printed. Learner 2 does not improve accuracy on its own,
but it shifts the weighted vote; learner 3 finally flips the stubborn point and
the ensemble reaches 100%. Note the alphas *rising* (0.97 → 1.28 → 1.61): each
learner is solving an easier re-weighted problem than the last.

---

## Real-World Applications

### 1. **Face Detection (Viola-Jones Framework)**
The most famous application of AdaBoost!
- Input: Image patches
- Output: Face or non-face
- Example: Camera auto-focus, Facebook photo tagging
- **Business Value**: Real-time face detection in consumer devices

**How it works**:
```
Weak Learners: Simple Haar-like features
  - "Is the eye region darker than forehead?"
  - "Is the nose bridge brighter than cheeks?"
  
AdaBoost combines 200+ such simple features:
  Feature 1 (alpha: 1.2): Checks eye region
  Feature 2 (alpha: 0.8): Checks mouth region
  Feature 3 (alpha: 0.6): Checks nose
  ...

Result: Real-time face detection at 30+ FPS!
```

### 2. **Medical Diagnosis**
Combining multiple diagnostic tests:
- Input: Symptoms, test results, patient history
- Output: Disease presence probability
- Example: Cancer detection, heart disease prediction
- **Business Value**: More accurate diagnoses, reduced false positives/negatives

**Example**:
```
Weak Learner 1: "High blood pressure? → Heart disease"
Weak Learner 2: "High cholesterol? → Heart disease"
Weak Learner 3: "Family history + age > 50? → Heart disease"
...

AdaBoost combines these simple rules into sophisticated diagnosis
Better than any single test alone!
```

### 3. **Fraud Detection**
Identifying fraudulent transactions:
- Input: Transaction features (amount, location, time, merchant)
- Output: Fraud or legitimate
- Example: Credit card fraud, insurance claims
- **Business Value**: Reduced financial losses

**Applications**:
```
Each weak learner checks simple patterns:
  - "Amount > $1000 and international? → Suspicious"
  - "Multiple transactions in 1 hour? → Suspicious"
  - "Unusual merchant category? → Suspicious"

AdaBoost learns which combinations matter most
Adapts to new fraud patterns over time
```

### 4. **Customer Churn Prediction**
Predicting which customers will leave:
- Input: Usage patterns, customer service calls, payment history
- Output: Likely to churn or not
- Example: Telecom, subscription services
- **Business Value**: Targeted retention campaigns

**Example**:
```
Weak patterns:
  - Reduced usage last month → Churn
  - Contacted support 3+ times → Churn
  - Competitor offer received → Churn

AdaBoost identifies which patterns matter most
Allows proactive intervention
```

### 5. **Text Classification**
Spam detection, sentiment analysis:
- Input: Email or document text
- Output: Category (spam/ham, positive/negative)
- Example: Email filters, product review analysis
- **Business Value**: Better user experience, insights from text data

**Example**:
```
Weak Learners (simple text rules):
  - Contains "free money"? → Spam
  - Contains "click here"? → Spam
  - Misspellings count > 5? → Spam

AdaBoost weighs importance of each clue
Much better than simple keyword matching!
```

### 6. **Quality Control in Manufacturing**
Defect detection in production:
- Input: Sensor readings, measurements, images
- Output: Defective or acceptable
- Example: PCB inspection, product quality
- **Business Value**: Reduced defects, lower costs

**Example**:
```
Weak Learners check simple criteria:
  - "Temperature > 75°C during process? → Defect"
  - "Pressure variance > 0.5? → Defect"
  - "Visual feature X detected? → Defect"

AdaBoost learns complex failure patterns
Better than manual inspection rules!
```

### 7. **Credit Scoring**
Assessing loan default risk:
- Input: Credit history, income, debt ratio, employment
- Output: Default risk score
- Example: Loan approval decisions
- **Business Value**: Better risk management

```
Weak risk indicators:
  - "Income/debt ratio < 2? → High risk"
  - "Credit inquiries > 3 last month? → High risk"
  - "Previous defaults? → High risk"

AdaBoost creates sophisticated risk model
More accurate than linear scoring
```

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Initializing Sample Weights

```python
def fit(self, X, y):
    n_samples = len(X)
    weights = np.ones(n_samples) / n_samples
```

**How it works**:
```python
n_samples = 8
weights = np.ones(8) / 8
# Result: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]

# Equal importance to all samples initially
# Sum = 1.0 (normalized probability distribution)
```

### 2. Training Weak Learner (Decision Stump)

The real helper is called **`_create_decision_stump`**, and it returns a stump
*dictionary* plus the error — not a `(feature, threshold, error)` triple:

```python
def _create_decision_stump(self, X, y, weights):
    n_samples, n_features = X.shape
    best_error = float('inf')
    best_stump = None
    total_weight = np.sum(weights)

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        # Candidate thresholds are MIDPOINTS between consecutive distinct
        # values, as sklearn's tree splitter does.
        if len(unique_values) == 1:
            thresholds = unique_values          # constant feature: no gap
        else:
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            # ... plus sklearn's midpoint guard, explained under the snippet
            rounded_up = thresholds >= unique_values[1:]
            overflowed = ~np.isfinite(thresholds)
            thresholds = np.where(rounded_up | overflowed,
                                  unique_values[:-1], thresholds)

        for threshold in thresholds:
            # Polarity A: predict -1 at or below the threshold, +1 above
            predictions = np.ones(n_samples)
            predictions[feature_values <= threshold] = -1

            misclassified = (predictions != y).astype(float)
            error = np.sum(weights * misclassified)

            if error < best_error:
                best_error = error
                best_stump = {'feature': feature_idx, 'threshold': threshold,
                              'left_prediction': -1, 'right_prediction': 1}

            # Polarity B is the mirror image. Every sample A got right, B gets
            # wrong, so its error is total_weight - error.
            error_flipped = total_weight - error

            if error_flipped < best_error:
                best_error = error_flipped
                best_stump = {'feature': feature_idx, 'threshold': threshold,
                              'left_prediction': 1, 'right_prediction': -1}

    return best_stump, best_error
```

Three details that are easy to miss:

- **Both polarities are searched.** Many textbook implementations only try
  "≤ threshold → -1". Trying the mirror too guarantees the returned error is
  never worse than half the total weight (0.5 for the normalized weights `fit()`
  passes), so α is never negative and every learner contributes.
- **Polarity B costs nothing.** Polarity B is wrong on exactly the samples
  polarity A got right, so its weighted error is `total_weight - error`.
  Recomputing it from scratch would double the innermost loop for no new
  information. Subtract from `total_weight`, not from a literal `1.0`: the
  weights only sum to 1 up to floating-point rounding, so `1.0 - error` leaves a
  residue of up to `2e-16` for a *flawless* mirrored stump instead of `0.0`, and
  it is simply wrong for any weight vector that is not normalized.
- **The midpoint is guarded.** For two feature values one ULP apart,
  `(v[i] + v[i+1]) / 2` rounds *up* onto `v[i+1]` — that happens for half of all
  adjacent double pairs — and for values near `1e308` the sum overflows to
  `inf`. Since the rule is `col <= threshold`, either case drags the upper value
  to the **left** and the intended split becomes unreachable: a perfectly
  separable two-value feature would score 0.5 error and α = 0. Falling back to
  `v[i]` restores it. sklearn's tree splitter carries the same guard
  (`if threshold == Xf[p]: threshold = Xf[p-1]`), and on ordinary data it never
  fires.

One thing the midpoints *do* change: the old raw-value candidate list also
contained `threshold = max(v)`, which sends every sample left and is therefore a
constant classifier. Midpoints drop it, so the stump family here is one
candidate smaller per feature. That is deliberate — sklearn's depth-1 tree
cannot emit a constant classifier either — but it means the best achievable
weighted error is occasionally a little higher than raw values would allow.

**Step-by-step example**:
```python
# Data
X = [[1], [2], [3], [4], [5], [6]]
y = [-1, -1, -1, +1, +1, +1]
weights = [0.2, 0.2, 0.1, 0.1, 0.2, 0.2]

# Distinct values 1..6 -> candidate thresholds 1.5, 2.5, 3.5, 4.5, 5.5

# Try threshold 3.5
predictions = [X[i] <= 3.5 ? -1 : +1]
            = [-1, -1, -1, +1, +1, +1]  (perfect!)

errors = [False, False, False, False, False, False]
weighted_error = 0.0        # polarity A
error_flipped  = 1.0        # polarity B: total_weight - 0.0, and here
                            # total_weight = sum(weights) = 1.0

# Polarity A at threshold 3.5 is the best split!
```

### 3. Inside fit(): calculating alpha

There is no `_calculate_alpha` method — the computation lives inline in the
`fit()` loop:

```python
# Prevent error from being 0 or 1 (numerical stability)
error = np.clip(error, 1e-10, 1 - 1e-10)

# Calculate learner weight (alpha)
# Higher alpha = lower error = more trust
alpha = 0.5 * np.log((1 - error) / error)
alpha = alpha * self.learning_rate  # Apply learning rate
```

**Why the clipping?**
```python
# Without clipping:
error = 0.0
alpha = 0.5 * np.log((1 - 0) / 0)
      = 0.5 * np.log(1 / 0)
      = 0.5 * np.log(inf)
      = inf  ❌ (numerical issues!)

# With clipping:
error = 0.0
error = np.clip(0.0, 1e-10, 1 - 1e-10) = 1e-10
alpha = 0.5 * np.log((1 - 1e-10) / 1e-10)
      ≈ 11.5  ✓ (large but finite)
```

**Learning rate effect**:
```python
# Without learning rate (learning_rate = 1.0):
error = 0.2
alpha = 0.693

# With learning rate = 0.5:
error = 0.2
alpha = 0.693 * 0.5 = 0.347

# Effect: Smaller alphas → more conservative updates
#         Helps prevent overfitting!
```

Because the learning rate scales α *before* it enters the weight update, the
wrong-to-right re-weighting ratio becomes `e^(2α) = ((1-ε)/ε)^learning_rate` —
identical to scikit-learn's SAMME, which computes `exp(learning_rate × ln((1-ε)/ε))`.

**The early stop**:
```python
# ... right after the stump's predictions are computed:
perfect = bool(np.all(predictions == y))

# ... and after the weights are updated and the learner is stored:
self.weak_learners.append(stump)
self.alphas.append(alpha)

if perfect:
    # Zero training error -- stop early rather than appending
    # n_estimators-1 identical copies of this stump.
    break
```

Why stopping is the right move: a stump that gets *everything* right multiplies
every weight by the same `e^(-α)`, and normalizing then hands back the exact
weight vector the round started with. Round t+1 would re-select the identical
stump, forever. Without the break, a perfectly separable dataset produces
`n_estimators` copies of the *same* stump, each with α ≈ 11.51, and
`get_feature_importance()` reports 100% for whichever feature it used.
scikit-learn's `AdaBoostClassifier` breaks out of the loop the same way.

Note **what** is tested: the predictions, not the weighted error. The two agree
whenever the weights still carry real mass, which is why `ε_t == 0` is the
textbook way to state the condition. But at a large `learning_rate` the easy
samples' weights collapse toward `0` and the misclassified mass collapses with
them — long before any weight is literally `0`, and it never has to get there.
On `make_moons(n_samples=120, noise=0.25, random_state=1)` at
`learning_rate=3.0` the smallest weight is `4.3e-22` by round 5, and there a
tolerance of `error <= 1e-10` fires on a stump
that reports `ε = 1e-13` while getting 19 of 120 samples wrong — ending training
at 84.17% where boosting on to 60 learners reaches 100%. Thirteen rounds later
the reported error is exactly `0.0` on a stump with 79 of 120 wrong (the
ensemble is at 69.17% there): its true misclassified mass, `9.5e-18`, disappears
into the `total_weight - error` subtraction that scores the mirrored polarity.
Testing the predictions cannot be fooled either way.
scikit-learn arrives at the same predicate from the other side: it floors every
sample weight at machine epsilon before each round, so its `error <= 0` test can
only fire on a genuinely flawless learner.

### 4. Inside fit(): updating sample weights

There is no `_update_weights` method either — this is also inline in `fit()`:

```python
# Make predictions with this stump
predictions = self._stump_predict(stump, X)

# Update sample weights: w_i <- w_i * exp(-alpha * y_i * h_t(x_i))
# y * predictions is +1 where the stump is right, -1 where it is wrong:
#   Correct: multiply by e^(-alpha) (decrease weight)
#   Wrong:   multiply by e^(+alpha) (increase weight)
weights = weights * np.exp(-alpha * y * predictions)

# Normalize weights to sum to 1
weights = weights / np.sum(weights)
```

Note the exponent: **`-alpha * y * predictions`**, not `alpha * (predictions != y)`.
See the boxed warning in [The Mathematical Foundation](#the-mathematical-foundation)
for why the difference is not cosmetic.

**Detailed example**:
```python
weights = np.array([1/6] * 6)              # 0.1667 each, summing to 1
y = np.array([-1, -1, -1, +1, +1, +1])
predictions = np.array([-1, -1, +1, +1, +1, +1])
#                        ✓   ✓   ✗   ✓   ✓   ✓

# The learner's weighted error: one wrong sample carrying weight 1/6
eps = 1/6 = 0.1667
alpha = 0.5 * ln((1 - 0.1667) / 0.1667) = 0.5 * ln(5) = 0.8047

# y * predictions = [+1, +1, -1, +1, +1, +1]
updates = np.exp(-0.8047 * y * predictions)
        = [e^-0.8047, e^-0.8047, e^+0.8047, e^-0.8047, e^-0.8047, e^-0.8047]
        = [0.4472, 0.4472, 2.2361, 0.4472, 0.4472, 0.4472]

# Update weights
weights = 0.1667 * [0.4472, 0.4472, 2.2361, 0.4472, 0.4472, 0.4472]
        = [0.0745, 0.0745, 0.3727, 0.0745, 0.0745, 0.0745]

# Normalize
sum = 0.7454
weights = [0.1000, 0.1000, 0.5000, 0.1000, 0.1000, 0.1000]

# The misclassified sample now carries 0.5 of ALL the weight - a 5x jump from
# 0.1667, and exactly the ratio (1-eps)/eps = 5 that the theory predicts.
# Check the invariant: this learner's error under the new weights is 0.5. ✓
```

### 5. Making Final Predictions

```python
def predict(self, X):
    # Calculate weighted sum of all learners
    weighted_sum = np.zeros(len(X))
    
    for alpha, stump in zip(self.alphas, self.weak_learners):
        predictions = self._stump_predict(stump, X)
        weighted_sum += alpha * predictions
    
    # Return sign of weighted sum, with ties broken toward +1
    # (np.sign(0) would return 0, which is not a class label)
    return np.where(weighted_sum >= 0, 1.0, -1.0)
```

**Example**:
```python
# 3 learners, 2 test samples
alphas = [0.693, 0.420, 0.549]

# Predictions for each learner
learner_1_pred = np.array([+1, -1])
learner_2_pred = np.array([+1, +1])
learner_3_pred = np.array([-1, +1])

# Calculate weighted sum
weighted_sum = 0.693 * [+1, -1] + 0.420 * [+1, +1] + 0.549 * [-1, +1]
             = [0.693, -0.693] + [0.420, 0.420] + [-0.549, 0.549]
             = [0.564, 0.276]

# Final predictions
final = np.where([0.564, 0.276] >= 0, 1.0, -1.0)
      = [+1, +1]
```

### 6. Feature Importance

```python
def get_feature_importance(self):
    importance = np.zeros(self.n_features)
    
    for alpha, stump in zip(self.alphas, self.weak_learners):
        feature_idx = stump['feature']
        importance[feature_idx] += abs(alpha)   # abs(): a negative alpha still
                                                # means the feature was USED
    
    # Normalize (guarded: every alpha could be 0 if every stump scored
    # exactly 0.5 error, and 0/0 would be nan)
    if np.sum(importance) > 0:
        importance = importance / np.sum(importance)
    
    return importance
```

**How it works**:
```python
# 3 features, 5 learners
alphas = [0.7, 0.5, 0.6, 0.4, 0.3]
features_used = [0, 1, 0, 2, 0]

# Accumulate importance
importance[0] += 0.7 + 0.6 + 0.3 = 1.6
importance[1] += 0.5 = 0.5
importance[2] += 0.4 = 0.4

# Normalize
total = 1.6 + 0.5 + 0.4 = 2.5
importance = [1.6/2.5, 0.5/2.5, 0.4/2.5]
           = [0.64, 0.20, 0.16]

# Feature 0 is most important (64%)!
```

---

## Model Evaluation

### Choosing Parameters

#### Number of Estimators (n_estimators)

```
Small (10-50):
  ✓ Faster training
  ✓ Less overfitting risk
  ✗ May underfit
  ✗ Not leveraging full boosting power
  
Medium (50-200):
  ✓ Good balance
  ✓ Usually optimal
  ✓ Reasonable training time
  
Large (200-500+):
  ✓ Maximum performance
  ✗ Risk of overfitting
  ✗ Slower training
  ✗ Diminishing returns
```

**How to choose**:

```python
# Hand-rolled k-fold. sklearn's cross_val_score CANNOT be used here:
#   cross_val_score(AdaBoost(), X, y, cv=5)
#   -> TypeError: Cannot clone object '<AdaBoost object>': it does not seem to
#      be a scikit-learn estimator as it does not implement a 'get_params' method
# Writing the loop yourself is 8 lines and shows exactly what a fold is.

def cross_val_accuracy(make_model, X, y, k=5, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    scores = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        model = make_model()                       # a FRESH model per fold
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[test_idx], y[test_idx]))
    return np.array(scores)

for n in [10, 25, 50, 100]:
    s = cross_val_accuracy(lambda n=n: AdaBoost(n_estimators=n), X, y, k=5)
    print(f"n_estimators={n:3d}: {s.mean():.2%} (+/- {s.std() * 2:.2%})")

# Choose where the curve plateaus - more estimators past that point only
# costs training time.
```

#### Learning Rate

```
High (1.0):
  ✓ Faster convergence
  ✓ Fewer estimators needed
  ✗ More prone to overfitting
  
Medium (0.5-0.8):
  ✓ Balanced approach
  ✓ Good default
  
Low (0.1-0.3):
  ✓ Better generalization
  ✓ More robust
  ✗ Needs more estimators
  ✗ Slower training
```

**Interaction with n_estimators**:
```
Rule of thumb:
  learning_rate × n_estimators ≈ constant

Examples:
  learning_rate=1.0, n_estimators=50
  learning_rate=0.5, n_estimators=100  (similar performance)
  learning_rate=0.1, n_estimators=500  (similar performance)

Lower learning rate + more estimators = comparable fit, reached in smaller
steps. USAGE EXAMPLE 5 in the .py runs exactly this trade-off on 200 points
and gets 96.88% train / 90.00% test on all three rows. A low rate is not
automatically better - it just needs proportionally more estimators. The
benefit shows up on noisy data, where smaller steps overfit more slowly.
```

### Performance Metrics

#### 1. Accuracy

```python
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

**Interpretation**:
```
90%+ accuracy: Excellent (for most problems)
80-90% accuracy: Good
70-80% accuracy: Acceptable (depends on problem)
<70% accuracy: May need more data or different approach
```

#### 2. Learning Curves

```python
train_scores = model.staged_score(X_train, y_train)
test_scores = model.staged_score(X_test, y_test)

import matplotlib.pyplot as plt
plt.plot(train_scores, label='Training')
plt.plot(test_scores, label='Testing')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

**What to look for**:
```
Ideal curve:
  ┌───────────────
  │     Test ──── (plateaus)
  │   Train ──── (slightly higher)
  └─────────────>

Overfitting:
  ┌───────────────
  │ Train ───────↗ (keeps increasing)
  │     Test ──── (plateaus or decreases)
  └─────────────>
  Solution: Reduce n_estimators, lower learning_rate

Underfitting:
  ┌───────────────
  │ Train ───↗↗
  │ Test ──↗↗ (both still increasing)
  └─────────────>
  Solution: Increase n_estimators
```

#### 3. Feature Importance

```python
importance = model.get_feature_importance()

for i, imp in enumerate(importance):
    print(f"Feature {i}: {imp:.3f}")

# Visualization
plt.bar(range(len(importance)), importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()
```

**Use cases**:
```
1. Feature Selection:
   - Remove features with near-zero importance
   - Reduce dimensionality
   - Speed up training

2. Feature Engineering:
   - Focus on important features
   - Create derived features from important ones

3. Interpretation:
   - Explain model decisions
   - Validate domain knowledge
```

### Comparing with Base Learner

You do not need sklearn for this — `AdaBoost(n_estimators=1)` *is* a single
decision stump, fitted by exactly the same search:

```python
# Ring vs. disk: 250 points, target y = +1 outside the circle of radius 2
np.random.seed(42)
X = np.random.randn(250, 2) * 2
y = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 > 4, 1, -1)

idx = np.random.permutation(250)
X, y = X[idx], y[idx]
X_train, X_test = X[:190], X[190:]
y_train, y_test = y[:190], y[190:]

# One stump vs. 50 boosted stumps
stump = AdaBoost(n_estimators=1).fit(X_train, y_train)
adaboost = AdaBoost(n_estimators=50).fit(X_train, y_train)

print(f"Single Stump : {stump.score(X_test, y_test):.2%}")
print(f"AdaBoost (50): {adaboost.score(X_test, y_test):.2%}")

# Output:
# Single Stump : 45.00%
# AdaBoost (50): 95.00%
# -> +50.00 percentage points
```

A single axis-aligned cut has essentially no purchase on a radially symmetric
target — on this 60-row test set it lands *below* chance, at 45%. Fifty of them,
weighted, box in the circle well enough for 95%. (The Quick Start at the top of
this page runs the same experiment at a different point in the random stream and
gets 61.67% → 93.33%; the gap is large either way, but the single-stump number
is noisy because it is one arbitrary cut.)

Report the gap in **percentage points**, not with `:.2%`. Formatting the
*difference* of two proportions as a percentage invites the reader to hear
"50% improvement", which is a different quantity — the relative gain here is
95.00/45.00 - 1 = 111%.

### Cross-Validation

Reusing `cross_val_accuracy` from above (again: sklearn's `cross_val_score`
raises `TypeError` on this class, because it is not an sklearn estimator):

```python
# 5-fold cross-validation on the two overlapping blobs from Example 2
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + np.array([-1, -1]),
               np.random.randn(50, 2) + np.array([ 1,  1])])
y = np.array([-1] * 50 + [1] * 50)

scores = cross_val_accuracy(lambda: AdaBoost(n_estimators=50), X, y, k=5)

print(f"Scores: {np.round(scores, 4)}")
print(f"Mean: {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")

# Output:
# Scores: [0.9  0.85 0.9  0.85 1.  ]
# Mean: 90.00% (+/- 10.95%)
```

Running the `n_estimators` sweep from the previous section on this same data
gives:

```
n_estimators= 10: 93.00% (+/- 8.00%)
n_estimators= 25: 91.00% (+/- 9.80%)
n_estimators= 50: 90.00% (+/- 10.95%)
n_estimators=100: 90.00% (+/- 10.95%)
```

The curve is flat-to-slightly-falling: on 100 overlapping points, 10 stumps are
already enough, and adding 90 more only fits noise. This is exactly the
"plateau" you are looking for — and a useful reminder that more estimators is
not automatically better.

---

## Computational Complexity

### Time Complexity

**Training**:
```
O(T × N × M × F)

where:
  T = number of estimators (n_estimators)
  N = number of samples
  M = number of unique values per feature (for finding splits)
  F = number of features

This implementation: O(T × N² × F) for continuous features.
With continuous data every value is distinct, so M = N - 1 midpoints, and
each candidate threshold is scored against all N samples.

Measured wall clock on this code, T = 50:
    200 samples ×  2 features:   0.13 s
    500 samples ×  5 features:   0.93 s
   1000 samples × 10 features:   4.91 s

Doubling N alone (F = 2, T = 10) shows the N² term emerging as NumPy's
per-call overhead stops dominating:
     250 ->  500:  1.84x
     500 -> 1000:  2.59x
    1000 -> 2000:  2.72x
    2000 -> 4000:  3.18x   (approaching the 4.00x that N² predicts)
Doubling F alone is exactly linear: 1.95x, 1.98x.

Library implementations pre-sort each feature once and sweep the split point,
reaching O(T × N × F × log(N)). That is the right complexity to quote for
scikit-learn; it is not what the readable loop in this file does.
```

**Prediction**:
```
O(T × N × 1)  [very fast!]

where:
  T = number of estimators
  N = number of samples to predict
  1 = constant time per stump prediction

Typical: O(T × N)
```

**Comparison with other algorithms**:
```
Training Time (for N samples, F features):
  AdaBoost (this file):  O(T × N² × F)          [exhaustive threshold scan]
  AdaBoost (sklearn):    O(T × N × F × log(N))  [pre-sorted split sweep]
  Random Forest: O(T × N × F × log(N))  [similar to sklearn's AdaBoost]
  Deep Neural Net: O(epochs × N × hidden_units × layers)  [usually slower]
  Linear SVM: O(N² × F) to O(N³ × F)  [slower for large N]

Prediction Time:
  AdaBoost: O(T × N)  [fast]
  Random Forest: O(T × N × tree_depth)  [slower]
  Deep Neural Net: O(N × hidden_units × layers)  [depends on architecture]
  Linear SVM: O(N × F)  [fast]
```

### Space Complexity

```
O(T × F)  [very efficient!]

Store:
  - T decision stumps
  - Each stump: feature_idx, threshold, prediction (constant space)
  - T alpha values

Total: Very compact model!

Example:
  50 estimators, 100 features
  Memory: ~50 × 3 × 8 bytes = 1.2 KB
  (extremely compact compared to neural networks!)
```

### Parallelization

```
Training: ❌ Sequential (cannot parallelize across estimators)
  - Each estimator depends on previous ones
  - Must train one after another

Prediction: ✅ Parallelizable (can parallelize across samples)
  - Each sample independent
  - Can evaluate on multiple CPUs/GPUs

Feature search: ✅ Parallelizable (within each estimator)
  - Can search different features in parallel
  - Helps with high-dimensional data
```

---

## Advantages and Limitations

### Advantages ✅

1. **High Accuracy**
   - Often matches or beats complex models
   - Combines weak learners into strong learner
   - Theoretical guarantees on training error

2. **Simple and Interpretable**
   - Easy to understand boosting principle
   - Feature importance readily available
   - Individual weak learners are interpretable

3. **Versatile** (the algorithm, not this file)
   - Works with various weak learners
   - Can handle binary and multi-class classification (via SAMME)
   - Variant for regression (AdaBoost.R2)
   - **This implementation is binary-only, with decision stumps as the only
     weak learner.** See [Simplification vs. canonical AdaBoost](#simplification-vs-canonical-adaboost).

4. **Few Hyperparameters**
   - Mainly: n_estimators and learning_rate
   - Less tuning than neural networks
   - Good default performance

5. **Resistant to Overfitting (with proper settings)**
   - Learning rate controls fitting speed
   - Can achieve good generalization
   - Early stopping helps prevent overfitting

6. **Handles Imbalanced Data**
   - Automatically focuses on hard examples
   - Minority class often hard to classify
   - AdaBoost naturally pays more attention to it

### Limitations ❌

1. **Sensitive to Noisy Data and Outliers**
   ```
   Problem: Outliers get increasing weight
   
   Example:
     Mislabeled sample: always wrong
     AdaBoost keeps increasing its weight
     Model focuses excessively on this error
   
   Solution:
     - Clean data before training
     - Use robust loss functions
     - Consider Gradient Boosting instead
   ```

2. **Sequential Training (Slow)**
   ```
   Cannot parallelize across estimators:
     Must train estimator t before t+1
     
   For large datasets:
     - Training time can be long
     - Unlike Random Forest (trains in parallel)
   
   Solution:
     - Use Gradient Boosting with histogram-based learning
     - Consider XGBoost for speed
   ```

3. **Risk of Overfitting with Too Many Estimators**
   ```
   Unlike Random Forest:
     - Can overfit with too many trees
     - Training error → 0, test error increases
   
   Solution:
     - Use cross-validation
     - Monitor test error
     - Use early stopping
     - Lower learning rate
   ```

4. **Weak Learners Must Be Better Than Random**
   ```
   If weak learner has 50% error:
     alpha = 0.5 × ln((1-0.5)/0.5) = 0
     No contribution!
   
   For very complex problems:
     - Single stumps may not be sufficient
     - Need deeper weak learners
     - But then loses simplicity advantage
   ```

5. **Binary Classification Focus**
   ```
   Originally designed for binary classification
   
   For multi-class:
     - Need extensions (SAMME, SAMME.R)
     - More complex
     - Slower training
   ```

6. **Less Effective on Very High-Dimensional Data**
   ```
   With thousands of features:
     - Many irrelevant features
     - Weak learners struggle to find good splits
     - Training becomes slow
   
   Solution:
     - Feature selection first
     - Use deep trees instead of stumps
     - Consider other algorithms (Linear models, Neural Nets)
   ```

### When to Use AdaBoost

**Good Use Cases**:
- ✅ Binary classification with clean data
- ✅ Medium-sized datasets (1K-100K samples)
- ✅ Moderate number of features (<100)
- ✅ Need interpretable model
- ✅ Have well-defined weak learner
- ✅ Want feature importance

**Bad Use Cases**:
- ❌ Very noisy data with many outliers → Use robust methods
- ❌ Need fast training and parallel processing → Use Random Forest
- ❌ Very large datasets (millions of samples) → Use XGBoost, LightGBM
- ❌ High-dimensional sparse data → Use Linear models
- ❌ Complex multi-class problems → Use Neural Networks
- ❌ Time series with temporal dependencies → Use RNNs, LSTMs

---

## Comparing with Alternatives

### AdaBoost vs. Random Forest

```
AdaBoost:
  ✓ Often higher accuracy
  ✓ Better with weak learners
  ✓ Smaller model size
  ✗ Sequential (slower)
  ✗ More prone to overfitting
  ✗ Sensitive to outliers
  
Random Forest:
  ✓ Parallelizable (faster)
  ✓ More robust to noise
  ✓ Handles high dimensions better
  ✗ Larger model size
  ✗ Individual trees deeper (less interpretable)
  ✗ May need more trees for same accuracy

When to choose:
  AdaBoost: Clean data, need high accuracy, smaller model
  Random Forest: Noisy data, need speed, very large datasets
```

### AdaBoost vs. Gradient Boosting

```
AdaBoost:
  ✓ Simpler to understand
  ✓ Fewer hyperparameters
  ✗ Locked to exponential loss (that IS the alpha/re-weighting derivation)
  ✗ Less flexible
  ✗ Sample weighting can be extreme
  
Gradient Boosting:
  ✓ More flexible (many loss functions)
  ✓ Better handles outliers
  ✓ Often better performance
  ✗ More hyperparameters to tune
  ✗ More complex conceptually
  ✗ Slower training

When to choose:
  AdaBoost: Starting point, simpler problem, interpretability
  Gradient Boosting: Complex problem, need best performance
```

### AdaBoost vs. XGBoost/LightGBM

```
AdaBoost:
  ✓ Simpler to implement and understand
  ✓ Good for education
  ✗ Slower
  ✗ Less features (no regularization, etc.)
  
XGBoost/LightGBM:
  ✓ Much faster (optimized implementations)
  ✓ Built-in regularization
  ✓ Handles missing values
  ✓ Many advanced features
  ✗ More complex
  ✗ More hyperparameters

When to choose:
  AdaBoost: Learning, simple projects, interpretability
  XGBoost/LightGBM: Production, competitions, best performance
```

---

## Key Concepts to Remember

### 1. **Sequential Learning is Powerful**
Each learner focuses on previous mistakes, creating specialized expertise that complements other learners.

### 2. **Weak Learners + Weighted Voting = Strong Learner**
```
Measured in this repo, on the ring-vs-disk target:
  Single stump:          45% test accuracy
  50 stumps (AdaBoost):  95% test accuracy

The whole is greater than the sum of its parts!
```

### 3. **Adaptive Sample Weights Drive Learning**
```
Round 1: Focus equally on all samples
Round 2: Focus on mistakes from Round 1
Round 3: Focus on mistakes from Round 2
...

Result: Comprehensive coverage of data space
```

### 4. **Alpha Values Encode Learner Quality**
```
High alpha: Good learner → More influence
Low alpha: Weak learner → Less influence

Automatic quality control!
```

### 5. **Balance Between Weak and Strong Learners**
```
Too weak: Each learner contributes little → need many estimators
Too strong: Overfit quickly → lose boosting benefits

Sweet spot: Decision stumps (1-level trees)
```

### 6. **Outliers Are Dangerous**
```
Outlier: Consistently misclassified
AdaBoost: Keeps increasing its weight
Result: Model distorted by few bad samples

Solution: Clean data first!
```

### 7. **Learning Rate Controls Fitting Speed**
```
learning_rate = 1.0: Aggressive learning, fast convergence, risk overfitting
learning_rate = 0.1: Conservative, slower, smaller steps

Lower rate needs proportionally more estimators to reach the same fit
(learning_rate x n_estimators ~= constant). It helps on noisy data, where
small steps overfit more slowly - not on every dataset.
```

---

## Conclusion

AdaBoost is a powerful and elegant algorithm that demonstrates the principle of ensemble learning! By understanding:
- How sequential training focuses on mistakes
- How sample weights adapt to highlight hard examples
- How weak learners combine through weighted voting
- How to choose n_estimators and learning_rate
- When AdaBoost excels and when to use alternatives

You've gained insight into one of the most important algorithms in machine learning! 🚀

**When to Use AdaBoost**:
- ✅ Binary classification with clean data
- ✅ Need interpretable ensemble model
- ✅ Want automatic feature importance
- ✅ Have effective weak learner
- ✅ Medium-sized datasets

**When to Use Something Else**:
- ❌ Very noisy/outlier-heavy data → Random Forest, robust methods
- ❌ Very large datasets → XGBoost, LightGBM
- ❌ Need parallelizable training → Random Forest
- ❌ Complex multi-class problems → Neural Networks, Gradient Boosting
- ❌ High-dimensional sparse data → Linear models, Neural Networks

**Next Steps**:
- Try AdaBoost on your own classification problems
- Compare with single decision tree to see boosting effect
- Experiment with n_estimators and learning_rate
- Learn about Gradient Boosting (generalization of AdaBoost)
- Explore XGBoost and LightGBM for production use
- Study other ensemble methods (Bagging, Stacking)

Happy Boosting! 💻🚀📊

