# Naive Bayes from Scratch: A Comprehensive Guide

Welcome to the world of Naive Bayes! 🎯 In this comprehensive guide, we'll explore one of the most elegant and efficient machine learning algorithms. Think of it as the "assume the best, calculate probabilities" algorithm!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is Naive Bayes?](#what-is-naive-bayes)
3. [How Naive Bayes Works](#how-naive-bayes-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Advantages and Limitations](#advantages-and-limitations)
11. [Variants Comparison](#variants-comparison)
12. [Simplifications vs. Canonical Naive Bayes](#simplifications-vs-canonical-naive-bayes)
13. [Complete Usage Example](#complete-usage-example)
14. [Tips for Better Performance](#tips-for-better-performance)
15. [Key Concepts to Remember](#key-concepts-to-remember)
16. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Naive Bayes from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _9_naive_bayes.py  (the __main__ block runs this)
# Or copy the NaiveBayes class from _9_naive_bayes.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the NaiveBayes class here (from _9_naive_bayes.py) ----
# class NaiveBayes: ...

np.random.seed(42)

# ------ GAUSSIAN NB: three blobs of continuous measurements ------
X0 = np.random.randn(100, 3) + np.array([0, 0, 0])
X1 = np.random.randn(100, 3) + np.array([3, 3, -3])
X2 = np.random.randn(100, 3) + np.array([-3, 2, 2])
X = np.vstack([X0, X1, X2])
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

# Shuffle BEFORE splitting: the rows above are grouped by class, so a raw
# X[:220] split would leave almost all of class 2 out of the training set.
idx = np.random.permutation(300)
X, y = X[idx], y[idx]

X_train, X_test = X[:220], X[220:]
y_train, y_test = y[:220], y[220:]

model = NaiveBayes(variant='gaussian')
model.fit(X_train, y_train)

print("Gaussian NB")
print(f"  Train Accuracy: {model.score(X_train, y_train):.2%}")
print(f"  Test  Accuracy: {model.score(X_test,  y_test):.2%}")
print(f"  Learned priors: {np.round(model.class_priors, 3)}")

proba = model.predict_proba(X_test)
pred = model.predict(X_test)
for i in range(3):
    print(f"    true={y_test[i]}  pred={pred[i]}  "
          f"P(0)={proba[i,0]:.3f}  P(1)={proba[i,1]:.3f}  P(2)={proba[i,2]:.3f}")

# ------ MULTINOMIAL NB: three planted document topics ------
topic_word = np.random.dirichlet(np.ones(40) * 0.3, 3)   # 3 topics x 40 words
doc_topic = np.random.randint(0, 3, 300)
X_docs = np.array([np.random.multinomial(40, topic_word[t]) for t in doc_topic])

idx = np.random.permutation(300)
X_docs, y_docs = X_docs[idx], doc_topic[idx]

text = NaiveBayes(variant='multinomial')
text.fit(X_docs[:220], y_docs[:220])

print("\nMultinomial NB")
print(f"  Train Accuracy: {text.score(X_docs[:220], y_docs[:220]):.2%}")
print(f"  Test  Accuracy: {text.score(X_docs[220:], y_docs[220:]):.2%}")
for i, c in enumerate(text.classes):
    top = np.argsort(text.feature_probs[i])[::-1][:3]
    print(f"    topic {c} top word ids: {[int(j) for j in top]}")
```

Expected output:
```
Gaussian NB
  Train Accuracy: 98.18%
  Test  Accuracy: 97.50%
  Learned priors: [0.336 0.355 0.309]
    true=2  pred=2  P(0)=0.000  P(1)=0.000  P(2)=1.000
    true=0  pred=0  P(0)=1.000  P(1)=0.000  P(2)=0.000
    true=1  pred=1  P(0)=0.000  P(1)=1.000  P(2)=0.000

Multinomial NB
  Train Accuracy: 100.00%
  Test  Accuracy: 100.00%
    topic 0 top word ids: [20, 7, 32]
    topic 1 top word ids: [0, 12, 29]
    topic 2 top word ids: [1, 8, 4]
```

The second demo is a **known-answer test**: the documents really were generated from three
hand-made word distributions, so a correct implementation should recover the topic almost
perfectly. Getting 100% there is evidence the multinomial likelihood table is right, not luck.

---

## What is Naive Bayes?

Naive Bayes is a **probabilistic classification algorithm** based on Bayes' Theorem with a "naive" assumption that all features are independent of each other. Despite this strong assumption, it works surprisingly well in practice!

**Real-world analogy**: 
Imagine diagnosing a disease. A doctor looks at symptoms (fever, cough, fatigue) and calculates: "Given these symptoms, what's the probability it's the flu?" Naive Bayes does exactly this—it calculates the probability of each possible class and picks the most likely one.

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Probabilistic, Generative |
| **Learning Style** | Supervised learning |
| **Tasks** | Classification (primarily) |
| **Decision** | Based on maximum posterior probability |
| **Key Assumption** | Feature independence |

### The Core Idea

```
"Given some features, what's the probability of each class?
 Pick the class with highest probability!"
```

Naive Bayes:
1. **Learns** probability distributions from training data
2. **Calculates** posterior probability for each class
3. **Predicts** the class with highest probability

---

## How Naive Bayes Works

### The Algorithm in 5 Steps

```
Step 1: Learn prior probabilities P(class) from training data
         ↓
Step 2: Learn likelihood probabilities P(features|class)
         ↓
Step 3: For new data, calculate posterior P(class|features)
         using Bayes' Theorem
         ↓
Step 4: Calculate posterior for each possible class
         ↓
Step 5: Predict the class with highest posterior probability
```

### Visual Example

```
Training Phase:
    
    Class A samples: ●●●●● (50 samples)
    Class B samples: ■■■ (30 samples)
    Total: 80 samples
    
    P(A) = 50/80 = 0.625
    P(B) = 30/80 = 0.375
    
    Learn feature distributions for each class

Prediction Phase:
    
    New sample: ? with features [x₁, x₂]
    
    Calculate:
        P(A|features) ∝ P(A) × P(features|A)
        P(B|features) ∝ P(B) × P(features|B)
    
    Compare:
        P(A|features) = 0.72
        P(B|features) = 0.28
    
    Prediction: Class A ●
```

### Why "Naive"?

The "naive" assumption is that all features are **conditionally independent** given the class:

```
Naive assumption:
P(x₁, x₂, x₃|class) = P(x₁|class) × P(x₂|class) × P(x₃|class)

Reality (usually):
Features are often correlated!
But Naive Bayes works well anyway!
```

**Example of the assumption**:
```
Predicting if an email is spam based on:
- Contains "free": yes
- Contains "money": yes
- Contains "urgent": yes

Naive assumption: These words appear independently
Reality: These words often appear together in spam

Despite this, Naive Bayes still works great for spam detection!
```

---

## The Mathematical Foundation

### Bayes' Theorem

The foundation of Naive Bayes is Bayes' Theorem:

```
P(A|B) = P(B|A) × P(A) / P(B)
```

For classification:

```
P(class|features) = P(features|class) × P(class) / P(features)
```

**Breaking it down**:

- **P(class|features)** = Posterior probability
  - What we want: probability of class given the features
  
- **P(features|class)** = Likelihood
  - Probability of seeing these features in this class
  
- **P(class)** = Prior probability
  - Overall probability of this class
  
- **P(features)** = Evidence
  - Overall probability of seeing these features
  - (We can ignore this for classification since it's the same for all classes)

### The Naive Bayes Formula

For classification with features x₁, x₂, ..., xₙ:

```
P(class|x₁,x₂,...,xₙ) ∝ P(class) × P(x₁|class) × P(x₂|class) × ... × P(xₙ|class)
```

**Simplified**:
```
Posterior ∝ Prior × Likelihood₁ × Likelihood₂ × ... × Likelihoodₙ
```

### Example Calculation

```
Problem: Classify email as Spam or Not Spam

Features:
- x₁: Contains "free" (yes = 1)
- x₂: Contains "meeting" (no = 0)

From training data:
- P(Spam) = 0.4, P(Not Spam) = 0.6
- P("free"|Spam) = 0.7, P("free"|Not Spam) = 0.1
- P("meeting"|Spam) = 0.1, P("meeting"|Not Spam) = 0.5

Calculate:

P(Spam|features) ∝ P(Spam) × P("free"=yes|Spam) × P("meeting"=no|Spam)
                 ∝ 0.4 × 0.7 × (1-0.1)
                 ∝ 0.4 × 0.7 × 0.9
                 ∝ 0.252

P(Not Spam|features) ∝ P(Not Spam) × P("free"=yes|Not Spam) × P("meeting"=no|Not Spam)
                     ∝ 0.6 × 0.1 × (1-0.5)
                     ∝ 0.6 × 0.1 × 0.5
                     ∝ 0.030

Prediction: Spam (0.252 > 0.030)
```

> **Note on this hand calculation**: it uses *presence/absence* probabilities — the
> `(1 - 0.1)` term is "the probability that 'meeting' is **absent** from a spam email".
> That is **Bernoulli** Naive Bayes, which is the easiest variant to work by hand but is
> **not** one of the two variants this class implements. The variants you can actually run
> are worked end-to-end in [Step-by-Step Example](#step-by-step-example) (Gaussian) and in
> USAGE EXAMPLE 3 inside `_9_naive_bayes.py` (Multinomial). See
> [Simplifications vs. Canonical Naive Bayes](#simplifications-vs-canonical-naive-bayes).

### Types of Naive Bayes

#### 1. Gaussian Naive Bayes

For **continuous features** (e.g., height, weight, temperature):

Assumes features follow a **Gaussian (normal) distribution**:

```
P(xᵢ|class) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))
```

Where:
- μ = mean of feature i in class
- σ² = variance of feature i in class

**Example**:
```python
Feature: Height (cm)
Class A: mean=170, std=10
Class B: mean=160, std=8

For new sample with height=165:
P(height=165|A) = calculate using Gaussian formula
P(height=165|B) = calculate using Gaussian formula
```

**When to use**: 
- Continuous features (measurements, sensors, financial data)
- Features roughly follow normal distribution

#### 2. Multinomial Naive Bayes

For **discrete features** (e.g., word counts, frequencies):

```
P(xᵢ|class) = (count of feature i in class + α) / (total count in class + α×n_features)
```

Where α is a smoothing parameter (usually 1, called Laplace smoothing).

**In this implementation α is hard-wired to 1**, so `fit` computes exactly
`(feature_counts + 1) / (total_count + n_features)` — the same table as
`sklearn.naive_bayes.MultinomialNB(alpha=1.0)`. There is no `alpha` argument; see
[Simplifications vs. Canonical Naive Bayes](#simplifications-vs-canonical-naive-bayes).

**Example**:
```python
Feature: Word "free" appears 5 times in email
Class Spam: word "free" appeared 100 times in 1000 total words
Class Not Spam: word "free" appeared 10 times in 1000 total words

P("free"=5|Spam) = calculate based on multinomial distribution
```

**When to use**:
- Text classification (spam detection, sentiment analysis)
- Document categorization
- Word counts or frequencies

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class NaiveBayes:
    def __init__(self, variant='gaussian'):
        # Rejects anything other than 'gaussian' / 'multinomial' with a ValueError,
        # so an unsupported variant fails immediately instead of at predict time.
        self.variant = variant
        self.classes = None
        self.class_priors = None
        self.means = None          # For Gaussian
        self.variances = None      # For Gaussian
        self.feature_probs = None  # For Multinomial
        self.n_features = None     # Set by fit; used to validate predict inputs
```

### Core Methods

1. **`__init__(variant)`** - Initialize model
   - variant: 'gaussian' or 'multinomial'
   - Determines which probability distribution to use
   - Raises `ValueError` for any other value

2. **`fit(X, y)`** - Train the model
   - Calculates prior probabilities P(class)
   - For Gaussian: learns mean and variance per feature per class
   - For Multinomial: learns feature probability distributions
   - Returns `self`, so `model.fit(X, y).predict(X)` works
   - Time complexity: O(n×d) where n=samples, d=features

3. **`_as_2d(X)`** - Private helper
   - Coerces plain Python lists into a float numpy array
   - Reshapes a 1-D input: a row of length `n_features` becomes one sample,
     anything else becomes a single-feature column
   - After `fit`, raises `ValueError` if the feature count does not match
     `n_features` — otherwise NumPy would silently broadcast a wrong-width row
     against the stored means and return a confident, meaningless prediction
   - Called at the top of `fit`, `predict` and `predict_proba`

4. **`_check_is_fitted()`** - Private helper
   - Raises a readable `ValueError` if `predict`/`predict_proba` is called before `fit`
   - Without it the failure surfaces as `'NoneType' object is not subscriptable`

5. **`_calculate_gaussian_likelihood(x, class_idx)`** - Private helper
   - Calculates P(features|class) using Gaussian distribution
   - Uses log probabilities to avoid numerical underflow
   - Returns log likelihood

6. **`_calculate_multinomial_likelihood(x, class_idx)`** - Private helper
   - Calculates P(features|class) for multinomial distribution
   - Relies on the Laplace smoothing applied in `fit`, which already guarantees
     every probability is strictly positive - so no epsilon is needed inside the log
   - Returns log likelihood

7. **`_predict_single(x)`** - Predict for one sample
   - Calculates posterior for each class
   - Returns class with highest posterior
   - Uses log probabilities for numerical stability

8. **`predict(X)`** - Predict for multiple samples
   - Calls _predict_single for each sample
   - Returns array of predictions
   - Main prediction interface

9. **`predict_proba(X)`** - Get class probabilities
   - Returns posterior probability for each class
   - Probabilities sum to 1
   - Useful for confidence estimation

10. **`score(X, y)`** - Calculate accuracy
    - This is a classifier, so `score` returns **accuracy**, not R^2
    - Accuracy = correct predictions / total predictions
    - Returns value between 0 and 1

---

## Step-by-Step Example

Let's walk through a complete example predicting **fruit type** based on weight and diameter:

### The Data

```python
import numpy as np

# Features: [weight (grams), diameter (cm)]
X_train = np.array([
    [150, 7],   # Apple
    [170, 8],   # Apple
    [140, 6.5], # Apple
    [160, 7.5], # Apple
    [350, 9],   # Orange
    [380, 9.5], # Orange
    [340, 8.5], # Orange
    [370, 9.2], # Orange
])

# Labels: 0=Apple, 1=Orange
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
```

### Training the Model

```python
model = NaiveBayes(variant='gaussian')
model.fit(X_train, y_train)
```

**What happens internally**:

**Step 1: Calculate Priors**
```
P(Apple) = 4/8 = 0.5
P(Orange) = 4/8 = 0.5
```

**Step 2: Calculate Statistics per Class**

For Apples (class 0):
```
Weight: mean=155g, variance=125
Diameter: mean=7.25cm, variance=0.3125
```

For Oranges (class 1):
```
Weight: mean=360g, variance=250
Diameter: mean=9.05cm, variance=0.1325
```

These are **population** variances (`np.var`, i.e. dividing by n, not n-1) — the
same convention scikit-learn's `GaussianNB` uses. You can check the weight
variance for Apples by hand: `mean((150-155)^2, (170-155)^2, (140-155)^2, (160-155)^2)`
`= mean(25, 225, 225, 25) = 125`.

If you print `model.variances` you will see `125.0000107` rather than exactly `125`.
That extra `1.069e-05` is the **variance-smoothing floor** `fit()` adds:
`epsilon = 1e-9 * max(var(X, axis=0))`. Here the largest feature variance across the
whole training set is the weight variance, `10693.75`, so `epsilon = 1.069e-05`.
Its job is explained in [Understanding the Code](#understanding-the-code).

### Making Predictions

```python
# New fruit to classify
X_test = np.array([[155, 7.2]])  # 155g, 7.2cm diameter

# Calculate posterior for Apple (class 0)
P(Apple) = 0.5
P(weight=155|Apple) = Gaussian(155, mean=155, var=125) = high probability
P(diameter=7.2|Apple) = Gaussian(7.2, mean=7.25, var=0.3125) = high probability

Posterior(Apple) ∝ 0.5 × high × high = VERY HIGH

# Calculate posterior for Orange (class 1)
P(Orange) = 0.5
P(weight=155|Orange) = Gaussian(155, mean=360, var=250) = very low probability
P(diameter=7.2|Orange) = Gaussian(7.2, mean=9.05, var=0.1325) = very low probability

Posterior(Orange) ∝ 0.5 × very_low × very_low = VERY LOW

# Prediction: Apple (class 0) ✓
```

### Complete Prediction Code

```python
# Predict for multiple samples
X_test = np.array([
    [155, 7.2],  # Should be Apple
    [360, 9.1],  # Should be Orange
])

predictions = model.predict(X_test)
print("Predicted classes:", predictions)
# Output: [0 1]   (Apple, Orange)

# Get probabilities
probabilities = model.predict_proba(X_test)
print("\nProbabilities:")
for i, probs in enumerate(probabilities):
    print(f"Sample {i+1}: Apple={probs[0]:.4f}, Orange={probs[1]:.4f}")
# Output:
# Sample 1: Apple=1.0000, Orange=0.0000
# Sample 2: Apple=0.0000, Orange=1.0000
```

Those are not exact 0s and 1s — `.4f` is just rounding. The raw posteriors are

```
[[1.0,          8.4446e-43],
 [3.8485e-76,   1.0       ]]
```

A 155 g fruit is about 13 standard deviations away from the Orange weight mean
(`sqrt(250) ~= 15.8 g`, and `|155 - 360| = 205 g`), and the two features multiply, so
the losing probability collapses to `1e-43`. **This is the whole reason the
implementation works in log space**: if you multiplied the raw densities instead of
adding their logarithms, a handful of extra features would drive the product to exactly
`0.0` in float64 and every class would tie at zero.

---

## Real-World Applications

### 1. **Spam Detection**
Filter spam emails based on word content:
- Input: Email word frequencies ("free", "money", "urgent")
- Output: Spam or Not Spam
- Example: "Email with 'free' and 'money' is 95% likely spam"

### 2. **Sentiment Analysis**
Determine sentiment of text:
- Input: Words and phrases in review/tweet
- Output: Positive, Negative, or Neutral
- Example: "This review is 80% likely positive"

### 3. **Medical Diagnosis**
Diagnose diseases based on symptoms:
- Input: Patient symptoms and test results
- Output: Disease diagnosis
- Example: "Symptoms match flu with 75% probability"

### 4. **Document Classification**
Categorize documents into topics:
- Input: Document word frequencies
- Output: Topic category (sports, politics, technology)
- Example: "Article belongs to 'Technology' category"

### 5. **Weather Prediction**
Predict weather based on conditions:
- Input: Temperature, humidity, pressure, wind
- Output: Sunny, Rainy, Cloudy
- Example: "70% chance of rain given these conditions"

### 6. **Credit Scoring**
Assess loan approval risk:
- Input: Income, credit history, employment status
- Output: Approve or Reject
- Example: "Applicant has 85% chance of approval"

### 7. **Recommendation Systems**
Suggest products based on user behavior:
- Input: User viewing history, ratings
- Output: Product categories to recommend
- Example: "User is 90% likely to prefer electronics"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Calculating Prior Probabilities

```python
# Count samples in each class
for idx, c in enumerate(self.classes):
    self.class_priors[idx] = np.sum(y == c) / n_samples
```

**How it works**:
```python
# Example
y = [0, 0, 1, 0, 1, 1, 1]  # 3 class 0, 4 class 1

P(class=0) = 3/7 = 0.428
P(class=1) = 4/7 = 0.572
```

**Why it matters**: 
- Classes that appear more often have higher prior probability
- Influences final prediction when features are ambiguous

### 2. Learning Gaussian Parameters

```python
# Computed ONCE from the full training matrix, before the per-class loop
epsilon = 1e-9 * np.var(X, axis=0).max()

for idx, c in enumerate(self.classes):
    X_c = X[y == c]  # Get all samples of class c
    self.means[idx, :] = np.mean(X_c, axis=0)
    self.variances[idx, :] = np.var(X_c, axis=0) + epsilon
```

**Step-by-step**:
```python
# Example: Class A samples
X_c = [[150, 7], [170, 8], [140, 6.5], [160, 7.5]]

# Calculate mean for each feature
mean_weight = (150 + 170 + 140 + 160) / 4 = 155
mean_diameter = (7 + 8 + 6.5 + 7.5) / 4 = 7.25

# Calculate variance for each feature
var_weight = mean((150-155)², (170-155)², (140-155)², (160-155)²)
           = mean(25, 225, 225, 25) = 125
```

**Why the `+ epsilon`? (variance smoothing)**

The likelihood divides by the variance. If a feature happens to be **constant inside one
class** — say every patient diagnosed with allergy has `fever_days = 0` — then
`np.var(X_c, axis=0)` is exactly `0` for that feature and the very next step divides by
zero, producing `inf` and then `nan`. The epsilon is a floor that keeps the division
finite.

Why scale it by `np.var(X, axis=0).max()` instead of using a flat `1e-9`?
Because "small" depends on units. A flat `1e-9` is invisible next to a weight measured in
grams (variance ~10000) but is **enormous** next to a feature whose own variance is
`1e-12`. Scaling the floor by the largest feature variance in the dataset makes it
uniformly negligible in whatever units you happen to be using.

This is exactly scikit-learn's rule:

```
GaussianNB.epsilon_ = var_smoothing * max(var(X, axis=0)),   var_smoothing = 1e-9
```

Matching it is not cosmetic. On the raw (unscaled) breast-cancer dataset,
sklearn's `epsilon_` is `3.2154e-04`, roughly **100,000x larger** than a flat `1e-9`,
and it nearly doubles the smallest per-class variance. With a flat `1e-9` this
implementation scored 0.9649 against sklearn's 0.9737 and their `predict_proba` outputs
differed by up to 0.77; with the scaled epsilon the two agree on every prediction and
their probabilities differ by less than `1e-15`. See USAGE EXAMPLE 5 in
`_9_naive_bayes.py`.

### 3. Calculating Gaussian Likelihood

```python
def _calculate_gaussian_likelihood(self, x, class_idx):
    mean = self.means[class_idx]
    variance = self.variances[class_idx]
    
    # Log likelihood to avoid underflow
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance))
    log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / variance)
    
    return log_likelihood
```

**Where do those two lines come from?**

This is the step most readers skip past, so let's do it properly. Start from the naive
independence assumption — the *only* thing that makes the whole algorithm tractable:

```
Step 1 - the naive assumption turns a joint density into a product:

    P(x₁, x₂, ..., x_d | c) = P(x₁|c) × P(x₂|c) × ... × P(x_d|c)
                            = ∏ⱼ P(xⱼ|c)

Step 2 - for Gaussian NB each factor is a normal density:

    P(xⱼ|c) = 1/√(2πσ²_cj) × exp( -(xⱼ - μ_cj)² / (2σ²_cj) )

Step 3 - take logs. A logarithm turns the PRODUCT into a SUM,
         and cancels the exp:

    log P(x|c) = Σⱼ [ -½·log(2πσ²_cj)  -  (xⱼ - μ_cj)² / (2σ²_cj) ]

Step 4 - split that single sum into two independent sums:

    log P(x|c) =  -½ · Σⱼ log(2πσ²_cj)        <- normalizing constant,
                                                 does not depend on x
                  -½ · Σⱼ (xⱼ - μ_cj)²/σ²_cj   <- scaled squared distance
                                                 to the class mean
```

Those two sums are, line for line, the two statements in the code:

```python
log_likelihood  = -0.5 * np.sum(np.log(2 * np.pi * variance))   # first sum
log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / variance)    # second sum
```

Read the second term out loud and Naive Bayes stops being mysterious: it is a
**variance-weighted squared distance from the sample to the class mean**. A class wins
when the sample sits close to its mean *relative to how spread out that class is*. The
first term is the price a class pays for being spread out at all — a wide class has large
`sigma^2`, so `log(2*pi*sigma^2)` is large and gets subtracted.

Note that `variance`, `mean` and `x` here are whole vectors of length `d`, so
`np.sum(...)` *is* the "sum over j". The naive independence assumption is what allows
that single `np.sum` to stand in for a d-dimensional joint density.

**Why log probabilities?**
```python
# Without log: multiply many small probabilities
P(x₁|class) = 0.01
P(x₂|class) = 0.001
P(x₃|class) = 0.0001
Product = 0.01 × 0.001 × 0.0001 = 0.000000001  # Underflow!

# With log: add log probabilities
log(P(x₁|class)) = -4.6
log(P(x₂|class)) = -6.9
log(P(x₃|class)) = -9.2
Sum = -4.6 + (-6.9) + (-9.2) = -20.7  # No underflow!
```

### 4. Handling Multinomial Features

```python
# Calculate feature probabilities with Laplace smoothing
for idx, c in enumerate(self.classes):
    X_c = X[y == c]
    feature_counts = np.sum(X_c, axis=0)
    total_count = np.sum(feature_counts)
    self.feature_probs[idx, :] = (feature_counts + 1) / (total_count + n_features)
```

**Example with Laplace smoothing**:
```python
# Without smoothing (BAD)
feature_count = 0  # Word never appeared in class
total_count = 1000
P(feature|class) = 0/1000 = 0  # Problem: zero probability!

# With smoothing (GOOD)
P(feature|class) = (0 + 1) / (1000 + n_features)  # Non-zero!

# Why it helps: Avoids saying "impossible" for unseen words
```

Smoothing has a second, quieter benefit: because the numerator `(count + 1)` is always at
least `1`, every probability is **strictly positive**, so `_calculate_multinomial_likelihood`
can call `np.log(feature_probs)` directly with no `log(0)` risk and no epsilon fudge:

```python
log_likelihood = np.sum(x * np.log(feature_probs))   # sum_i  x_i * log(p_i)
```

An epsilon inside that log would not be "safe" — it would be a silent bias on every
log-likelihood, and the bias grows as the vocabulary grows and the individual `p_i`
shrink.

### 5. Making Predictions

```python
# Calculate posterior for each class
posteriors = []
for idx, c in enumerate(self.classes):
    prior = np.log(self.class_priors[idx])
    likelihood = self._calculate_gaussian_likelihood(x, idx)
    posterior = prior + likelihood  # log(prior × likelihood)
    posteriors.append(posterior)

# Return class with highest posterior
return self.classes[np.argmax(posteriors)]
```

**Example**:
```python
# Two classes: 0 and 1
log_prior_0 = -0.69    # log(0.5)
log_likelihood_0 = -2.5
posterior_0 = -0.69 + (-2.5) = -3.19

log_prior_1 = -0.69    # log(0.5)
log_likelihood_1 = -8.2
posterior_1 = -0.69 + (-8.2) = -8.89

# Class 0 has higher posterior (-3.19 > -8.89)
Prediction: 0
```

### 6. Converting to Probabilities

**First: where did the evidence term go?**

Bayes' theorem has a denominator that section 5 quietly dropped:

```
                P(x|class) × P(class)
P(class|x)  =  ----------------------
                        P(x)
```

`predict` never computes `P(x)` because **it is the same number for every class**.
Dividing every score by the same positive constant cannot change which score is largest,
and `predict` only needs `argmax`. That is what the `∝` ("proportional to") symbol
throughout this document means.

`predict_proba`, however, *does* need real probabilities that sum to 1 — and it gets
`P(x)` for free. By the law of total probability, `P(x)` is just the sum of the
numerators over all classes:

```
P(x) = Σ_k  P(x|class_k) × P(class_k)
```

So "divide by the evidence" and "normalize the class scores so they sum to 1" are the
*same operation*. That is precisely the last line below.

```python
# Convert log posteriors to probabilities
posteriors = np.array(posteriors)
posteriors = np.exp(posteriors - np.max(posteriors))  # Numerical stability
posteriors = posteriors / np.sum(posteriors)  # Normalize to sum to 1 == divide by P(x)
```

The `- np.max(posteriors)` is the **log-sum-exp trick**. The log posteriors can easily be
-700 or lower (see the fruit example, where a raw posterior was `1e-76`), and
`np.exp(-800)` underflows to `0.0`, which would make every class zero and the division
`0/0`. Subtracting the maximum first guarantees the largest exponent is exactly `exp(0) = 1`,
so nothing overflows and the denominator is never zero. Because the same constant is
subtracted from every class, it cancels in the ratio and the answer is unchanged.

**Example**:
```python
log_posteriors = [-3.19, -8.89]

# Subtract max for stability
adjusted = [-3.19 - (-3.19), -8.89 - (-3.19)]
         = [0, -5.7]

# Exponentiate
exp_values = [exp(0), exp(-5.7)]
           = [1.0, 0.0033]

# Normalize
sum_exp = 1.0 + 0.0033 = 1.0033
probabilities = [1.0/1.0033, 0.0033/1.0033]
              = [0.997, 0.003]  # Sums to 1.0
```

---

## Model Evaluation

### Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Example**:
```python
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 0]
                    ↑
                 wrong

correct = 7
total = 8
accuracy = 7/8 = 0.875 (87.5%)
```

### Confusion Matrix

```
                Predicted
              Class 0  Class 1
Actual    0   [  TN  ] [  FP  ]
          1   [  FN  ] [  TP  ]
```

**Example**:
```
                Predicted
              Not Spam  Spam
Actual  Not S  [  45  ] [  5  ]  → 5 false positives
        Spam   [  3   ] [ 47  ]  → 3 false negatives
```

### Precision, Recall, F1-Score

```
Precision = TP / (TP + FP)  # Of predicted positive, how many correct?
Recall = TP / (TP + FN)     # Of actual positive, how many found?
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example**:
```python
TP = 47  # Correctly predicted spam
FP = 5   # Incorrectly predicted spam
FN = 3   # Missed spam emails

Precision = 47 / (47 + 5) = 0.904 (90.4%)
  → "90.4% of emails we marked as spam were actually spam"

Recall = 47 / (47 + 3) = 0.940 (94.0%)
  → "We caught 94% of all spam emails"

F1 = 2 × (0.904 × 0.940) / (0.904 + 0.940) = 0.922
```

### Cross-Validation

Test model on multiple train/test splits.

Note that `sklearn.model_selection.cross_val_score` **cannot** be used on this class:
it calls `sklearn.base.clone`, which requires a `get_params` method, so passing our
`NaiveBayes` object raises
`TypeError: Cannot clone object ... it does not seem to be a scikit-learn estimator`.
Either subclass `sklearn.base.BaseEstimator`, or just write the K-fold loop — it is
eight lines and uses nothing but the class's own API:

```python
import numpy as np

def cross_val_score_nb(X, y, k=5, variant='gaussian', seed=42):
    """Plain K-fold cross-validation using only NaiveBayes.fit / .score."""
    rng = np.random.RandomState(seed)          # private RNG, not the global one
    order = rng.permutation(len(X))            # shuffle so folds are not class-sorted
    folds = np.array_split(order, k)

    scores = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        model = NaiveBayes(variant=variant)
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[test_idx], y[test_idx]))
    return np.array(scores)

# Example on three well-separated Gaussian blobs
np.random.seed(0)
X = np.vstack([np.random.randn(100, 3),
               np.random.randn(100, 3) + 3,
               np.random.randn(100, 3) - 3])
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

scores = cross_val_score_nb(X, y, k=5)
print("Fold scores:", np.round(scores, 3))
print(f"Accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
# Output:
# Fold scores: [1.    1.    0.983 0.983 1.   ]
# Accuracy: 0.993 (+/- 0.008)
```

Note `np.random.RandomState(seed)` rather than `np.random.seed(seed)`: the folds get their
own private random stream instead of silently resetting the global NumPy RNG for the rest
of your program.

---

## Advantages and Limitations

### Advantages ✅

1. **Fast and Efficient**
   - Training is very fast (just calculate statistics)
   - Prediction is also fast
   - Works well with large datasets

2. **Works with Small Data**
   - Performs well even with limited training samples
   - Doesn't require huge datasets

3. **Handles High Dimensions Well**
   - Works effectively with many features
   - Great for text classification (thousands of words)

4. **Probabilistic Predictions**
   - Provides probability estimates
   - Useful for ranking and confidence scores

5. **Simple and Interpretable**
   - Easy to understand and explain
   - Can see which features influence prediction

6. **Handles Missing Data** *(property of the algorithm, not implemented in this class)*
   - The algorithm can ignore features with missing values: just drop those terms
     from the per-feature sum
   - **This class does not do that.** Pass a `NaN` and it propagates silently through
     `np.sum`, giving `NaN` likelihoods for every class. Impute or drop first.

7. **Online Learning** *(property of the algorithm, not implemented in this class)*
   - The learned statistics are just counts, sums and sums-of-squares, so they can be
     updated incrementally without revisiting old data
   - **This class has no `partial_fit`.** Calling `fit` again discards the previous
     model completely. See
     [Simplifications vs. Canonical Naive Bayes](#simplifications-vs-canonical-naive-bayes).

### Limitations ❌

1. **Independence Assumption**
   - Assumes features are independent (often false)
   - May not capture feature interactions
   - Example: "free" and "money" often appear together

2. **Zero Probability Problem**
   - Unseen feature values get zero probability
   - Solution: Use Laplace smoothing
   - Still can be problematic with very sparse data

3. **Continuous Features Assumption**
   - Gaussian Naive Bayes assumes normal distribution
   - Features may not actually be normally distributed
   - Can hurt performance if assumption is violated

4. **Not the Best for Complex Patterns**
   - Can't learn complex feature interactions
   - Other algorithms may perform better on complex data

5. **Sensitive to Feature Scales (Gaussian)**
   - Different feature scales affect probability calculations
   - May need feature scaling for best results

6. **Calibration Issues**
   - Predicted probabilities may not be well-calibrated
   - 0.8 probability doesn't necessarily mean 80% confidence

### When to Use Naive Bayes

**Good Use Cases**:
- ✅ Text classification (spam, sentiment, topics)
- ✅ Document categorization
- ✅ Real-time prediction (fast inference)
- ✅ High-dimensional data (many features)
- ✅ Small to medium datasets
- ✅ Need probabilistic predictions
- ✅ Features are mostly independent

**Bad Use Cases**:
- ❌ Features are highly correlated
- ❌ Need to capture complex feature interactions
- ❌ Features don't fit assumed distributions
- ❌ Requires best possible accuracy (use ensemble methods)
- ❌ Very imbalanced classes (without adjustments)

---

## Variants Comparison

### Gaussian Naive Bayes

```
Best for: Continuous features
Assumption: Features follow normal distribution
Use cases: 
  - Medical diagnosis (measurements)
  - Weather prediction (temperature, pressure)
  - Financial predictions (prices, volumes)

Example features:
  - Height: 175.5 cm
  - Weight: 72.3 kg
  - Temperature: 38.2°C
```

### Multinomial Naive Bayes

```
Best for: Discrete features (counts)
Assumption: Features follow multinomial distribution
Use cases:
  - Text classification
  - Spam detection
  - Document categorization

Example features:
  - Word "free" appears 3 times
  - Word "money" appears 2 times
  - Word "meeting" appears 0 times
```

### Bernoulli Naive Bayes

> **Not implemented in this class.** It is described here for completeness because it is
> the third standard variant and the easiest one to work by hand (the spam calculation in
> [The Mathematical Foundation](#the-mathematical-foundation) is Bernoulli-style).
> `NaiveBayes(variant='bernoulli')` raises
> `ValueError: variant must be 'gaussian' or 'multinomial', got 'bernoulli'`.
> For binary presence/absence features, use `variant='multinomial'` on 0/1 data as a
> close stand-in, or `sklearn.naive_bayes.BernoulliNB`.

```
Best for: Binary features (present/absent)
Assumption: Features are binary
Use cases:
  - Text classification (word presence)
  - Feature presence detection

Example features:
  - Contains "free": Yes (1)
  - Contains "money": No (0)
  - Contains "urgent": Yes (1)
```

What makes Bernoulli genuinely different from Multinomial (and why 0/1 Multinomial is
only a stand-in): Bernoulli explicitly rewards a feature being **absent**. Its likelihood
is

```
P(x|class) = ∏ⱼ [ pⱼ^xⱼ × (1 - pⱼ)^(1 - xⱼ) ]
```

so a document that lacks the word "free" gets an explicit `(1 - p_free)` factor.
Multinomial simply contributes nothing for a zero count.

---

## Simplifications vs. Canonical Naive Bayes

This implementation is deliberately small so the math stays visible. Everything the
Gaussian and Multinomial variants *do* compute matches scikit-learn's `GaussianNB` and
`MultinomialNB(alpha=1.0)` to floating-point noise (learned means, variances and priors
agree exactly; `predict_proba` agrees to about `1e-15`). What follows is what canonical
implementations offer that this one does not, and what it costs you.

| Canonical feature | Here | Consequence |
|---|---|---|
| **Bernoulli / Complement / Categorical NB** | Only `'gaussian'` and `'multinomial'` | Binary presence/absence data has no dedicated variant; absence is not explicitly rewarded. Use 0/1 counts with `'multinomial'` as an approximation. |
| **Tunable Laplace `alpha`** | Fixed at `alpha = 1` | You cannot weaken smoothing (`alpha=0.01`) on a huge vocabulary or strengthen it on a tiny corpus. sklearn exposes `MultinomialNB(alpha=...)`. |
| **User-supplied class priors** | Always estimated as class frequency | No `priors=` argument. For imbalanced data you must override `model.class_priors` after `fit` (see [Tips](#tips-for-better-performance)). |
| **Tunable `var_smoothing`** | Fixed at `1e-9` (scaled by `max(var(X, axis=0))`, sklearn's rule) | You cannot dial the variance floor up for very noisy features. The default value and the scaling rule are the same as sklearn's. |
| **`partial_fit` / online updates** | Not implemented | Calling `fit` again replaces the model. Re-fit on the full dataset instead. |
| **`sample_weight`** | Not implemented | Cannot weight individual training rows. |
| **Missing-value handling** | Not implemented | `NaN` propagates to `NaN` likelihoods for every class. Impute before fitting. |
| **Vectorized prediction** | One Python loop per sample | Slower than sklearn's matrix form by a large constant factor, but far easier to read. This repository's rule is clarity over performance. |

Each of these is genuinely optional: none of them changes the decision rule
`argmax_c [ log P(c) + log P(x|c) ]`, which is the thing you came here to understand.

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optional: Scale features for Gaussian Naive Bayes
# (Not strictly necessary but can help with numerical stability)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = NaiveBayes(variant='gaussian')
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
print("\nSample Predictions with Confidence:")
for i in range(5):
    true_label = data.target_names[y_test[i]]
    pred_label = data.target_names[y_pred[i]]
    confidence = np.max(y_proba[i])
    
    print(f"Sample {i+1}:")
    print(f"  True: {true_label}")
    print(f"  Predicted: {pred_label}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Probabilities: Malignant={y_proba[i][0]:.3f}, Benign={y_proba[i][1]:.3f}\n")
```

---

## Tips for Better Performance

### 1. Feature Engineering

```python
# Transform features to be more normally distributed
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer()
X_transformed = transformer.fit_transform(X)
```

### 2. Laplace Smoothing

Already implemented! The `+ 1` in multinomial probability calculation:
```python
P(feature|class) = (count + 1) / (total + n_features)
```

### 3. Feature Selection

Remove irrelevant features:
```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=10)  # Keep top 10 features
X_selected = selector.fit_transform(X, y)
```

### 4. Handle Imbalanced Classes

`__init__` has no `priors` argument (unlike sklearn's `GaussianNB(priors=...)`), so
override the attribute on the instance. It must be done **after** `fit`, because `fit`
computes and overwrites `class_priors` from the training class frequencies:

```python
model = NaiveBayes(variant='gaussian')
model.fit(X_train, y_train)

print("Estimated priors:", np.round(model.class_priors, 3))

# Give equal weight to all classes, so a rare class is not drowned out
model.class_priors = np.ones(len(model.classes)) / len(model.classes)

print("Uniform priors  :", np.round(model.class_priors, 3))
print("Accuracy with uniform priors:", model.score(X_test, y_test))
```

On a 90/10 imbalanced two-class problem this typically trades a little overall accuracy
for much better recall on the minority class — exactly the trade you usually want when
the rare class is the one that matters (fraud, disease, defects).

### 5. Log Probabilities

Always use log probabilities (we do this!):
```python
# Prevents numerical underflow
log_prob = log(P1) + log(P2) + log(P3)
# Instead of: prob = P1 × P2 × P3  (can underflow!)
```

---

## Key Concepts to Remember

### 1. **Bayes' Theorem is the Foundation**
P(class|features) = P(features|class) × P(class) / P(features)

### 2. **The "Naive" Assumption**
Features are assumed independent given the class. Often wrong, but works anyway!

### 3. **Prior and Likelihood**
- Prior: P(class) - how common is each class
- Likelihood: P(features|class) - how typical are features for class

### 4. **Two Main Variants**
- Gaussian: For continuous features
- Multinomial: For discrete features (counts)

### 5. **Fast and Efficient**
- Training: O(n×d) - just calculate statistics
- Prediction: O(k×d) - where k is number of classes
- Great for real-time applications

### 6. **Works Great for Text**
Despite independence assumption, excellent for:
- Spam detection
- Sentiment analysis
- Document classification

---

## Conclusion

Naive Bayes is an elegant and powerful algorithm! By understanding:
- How Bayes' Theorem enables probabilistic classification
- Why the "naive" independence assumption still works
- How different variants handle different data types
- How to interpret probability predictions

You've gained a fundamental tool that's still widely used in practice! 🎯

**When to Use Naive Bayes**:
- ✅ Text classification
- ✅ Need fast predictions
- ✅ High-dimensional data
- ✅ Small datasets
- ✅ Need probability estimates

**When to Use Something Else**:
- ❌ Features are highly correlated → Use logistic regression, neural networks
- ❌ Need best accuracy → Use Random Forests, XGBoost
- ❌ Need to capture interactions → Use decision trees, SVM
- ❌ Non-normal continuous features → Transform or use other methods

**Next Steps**:
- Try Naive Bayes on your own datasets
- Compare Gaussian vs Multinomial variants
- Experiment with text classification
- Compare with other algorithms
- Learn about calibration techniques
- Explore semi-supervised Naive Bayes

Happy coding! 💻🎯

