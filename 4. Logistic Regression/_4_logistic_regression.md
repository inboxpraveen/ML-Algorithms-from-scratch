# Logistic Regression from Scratch: A Comprehensive Guide

Welcome to the world of Logistic Regression! 🎯 In this comprehensive guide, we'll explore how to solve binary classification problems. Think of it as the go-to algorithm when you need to answer yes/no questions based on data!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is Logistic Regression?](#what-is-logistic-regression)
3. [Regression vs Classification](#regression-vs-classification)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Hyperparameter Tuning](#hyperparameter-tuning)
11. [Visualizing Logistic Regression](#visualizing-logistic-regression)
12. [Key Concepts to Remember](#key-concepts-to-remember)
13. [Simplifications vs scikit-learn](#simplifications-vs-scikit-learn)
14. [Complete Usage Example](#complete-usage-example)
15. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Logistic Regression from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _4_logistic_regression.py  (the __main__ block runs this)
# Or copy the LogisticRegression class from _4_logistic_regression.py
# and paste it above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the LogisticRegression class here ----
# class LogisticRegression: ...

np.random.seed(42)

# ------ TWO OVERLAPPING GAUSSIAN BLOBS ------
X0 = np.random.randn(100, 2) + np.array([-1, -1])   # class 0 cloud
X1 = np.random.randn(100, 2) + np.array([ 1,  1])   # class 1 cloud
X = np.vstack([X0, X1])
y = np.array([0] * 100 + [1] * 100)

# Shuffle before slicing: the rows are stacked class-0-then-class-1, so an
# unshuffled split would hand the test set every single class-1 point.
idx = np.random.permutation(200)
X, y = X[idx], y[idx]

X_train, X_test = X[:150], X[150:]   # [150:], NOT [50:] - no overlap
y_train, y_test = y[:150], y[150:]

model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train, y_train)

print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test  accuracy: {model.score(X_test,  y_test):.4f}")
print(f"Loss: {model.losses[0]:.4f} -> {model.losses[-1]:.4f}")

proba = model.predict_proba(X_test)
preds = model.predict(X_test)
for i in range(3):
    print(f"  true={y_test[i]}  P(y=1)={proba[i]:.4f}  pred={preds[i]}")

# The threshold is a public argument: move it without refitting.
for t in [0.3, 0.5, 0.7]:
    p = model.predict(X_test, threshold=t)
    print(f"  threshold={t:.1f} -> positives={int(p.sum())}  "
          f"accuracy={np.mean(p == y_test):.4f}")

# ------ STUDENT PASS/FAIL, WITH FEATURE SCALING ------
X_stud = np.array([[1.0, 20], [2.0, 40], [3.0, 60], [4.0, 90], [5.0, 75],
                   [1.5, 30], [2.5, 50], [3.5, 70], [4.5, 90]])
y_stud = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])       # 0 = Fail, 1 = Pass

# Attendance spans 20-100 while study hours span 1-5. Standardize, or the
# attendance column dominates every gradient.
mu, sd = X_stud.mean(axis=0), X_stud.std(axis=0)
stud = LogisticRegression(learning_rate=0.5, iterations=5000)
stud.fit((X_stud - mu) / sd, y_stud)

c = stud.get_coefficients()
print(f"\nStudent model accuracy: {stud.score((X_stud - mu) / sd, y_stud):.4f}")
print(f"Intercept {c['intercept']:.4f}, coefficients {np.round(c['coefficients'], 4)}")
print(f"Odds ratios per std dev: {np.round(np.exp(c['coefficients']), 2)}")

X_new = np.array([[2, 30], [4, 85], [3, 55]])
print(f"P(pass) for three new students: "
      f"{np.round(stud.predict_proba((X_new - mu) / sd), 4)}")
print(f"Predicted outcomes: {stud.predict((X_new - mu) / sd)}")
```

Expected output:
```
Train accuracy: 0.9400
Test  accuracy: 0.9200
Loss: 0.6931 -> 0.1522
  true=1  P(y=1)=0.8839  pred=1
  true=0  P(y=1)=0.0002  pred=0
  true=1  P(y=1)=0.2590  pred=0
  threshold=0.3 -> positives=25  accuracy=0.9000
  threshold=0.5 -> positives=24  accuracy=0.9200
  threshold=0.7 -> positives=20  accuracy=0.9200

Student model accuracy: 0.7778
Intercept 0.5558, coefficients [2.042  1.1856]
Odds ratios per std dev: [7.71 3.27]
P(pass) for three new students: [0.0804 0.9697 0.5962]
Predicted outcomes: [0 1 1]
```

Running `python _4_logistic_regression.py` directly executes a slightly fuller version of this (three demos, about 1.4 seconds).

---

## What is Logistic Regression?

Logistic Regression is a **classification algorithm** (despite its name!) used to predict binary outcomes (0 or 1, Yes or No, True or False). It estimates the probability that an instance belongs to a particular class.

**Real-world analogy**: 
Imagine a doctor diagnosing if a patient has a disease. Instead of predicting a continuous value (like temperature), the doctor predicts a probability: "There's an 85% chance this patient has the disease." If the probability is above 50%, diagnose as "has disease" (1), otherwise "no disease" (0).

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Problem Type** | Binary Classification |
| **Output** | Probability between 0 and 1 |
| **Decision** | Threshold-based (typically 0.5) |
| **Training Method** | Gradient Descent |
| **Loss Function** | Binary Cross-Entropy |

### The Mathematical Equation

The prediction formula uses the **sigmoid function**:

```
p(y=1|x) = 1 / (1 + e^(-z))

where: z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

Where:
- **p(y=1|x)** = probability that y equals 1 given features x
- **e** = Euler's number (approximately 2.718)
- **z** = linear combination of features (like in linear regression)
- **b₀** = intercept (bias term)
- **b₁, b₂, ..., bₙ** = coefficients for each feature

---

## Regression vs Classification

### The Key Difference

| Linear Regression | Logistic Regression |
|------------------|---------------------|
| **Predicts continuous values** | **Predicts probabilities** |
| Output: Any real number | Output: Between 0 and 1 |
| Example: House price ($200,000) | Example: Spam email (85% spam) |
| Loss: Mean Squared Error | Loss: Binary Cross-Entropy |
| Line fitting | S-curve (sigmoid) fitting |

### Why Not Use Linear Regression for Classification?

```
Linear Regression Output:  ...  -1.5  |  0.2  |  0.5  |  0.8  |  1.2  |  2.5  ...
                                  ❌     ✓      ✓      ✓      ❌     ❌
                            (Negative!)              (Over 1!)
                            
Logistic Regression Output: ... 0.05  | 0.20  | 0.50  | 0.80  | 0.95 ...
                                  ✓      ✓      ✓      ✓      ✓
                            (Always between 0 and 1!)
```

**Problems with Linear Regression for Classification**:
1. Can predict values < 0 or > 1 (not valid probabilities!)
2. Sensitive to outliers
3. Assumes linear relationship with class labels
4. Poor decision boundaries

**Why Logistic Regression Works**:
1. Outputs are always valid probabilities (0 to 1)
2. Better handles outliers
3. S-shaped curve fits binary data naturally
4. Clear probabilistic interpretation

---

## The Mathematical Foundation

### The Sigmoid Function

The heart of logistic regression is the **sigmoid function** (also called logistic function):

```
σ(z) = 1 / (1 + e^(-z))
```

**Properties**:
- Maps any real number to (0, 1)
- S-shaped curve
- σ(0) = 0.5 (midpoint)
- σ(∞) → 1
- σ(-∞) → 0

**Visualization**:
```
    1.0 |              ________
        |            /
  p(y)  |           /
        |          /
    0.5 |  _______/________
        |        /
    0.0 |_______/_______________
           -∞    0    ∞
                 z
```

### How It Works

**Step 1**: Compute linear combination
```
z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

**Step 2**: Apply sigmoid function
```
p = 1 / (1 + e^(-z))
```

**Step 3**: Make decision
```
if p >= 0.5: predict 1 (positive class)
else:        predict 0 (negative class)
```

### Example Calculation

```python
# Given: x₁=2, x₂=3, b₀=0.5, b₁=1.2, b₂=0.8

# Step 1: Linear combination
z = 0.5 + (1.2 × 2) + (0.8 × 3)
z = 0.5 + 2.4 + 2.4 = 5.3

# Step 2: Sigmoid
p = 1 / (1 + e^(-5.3))
p = 1 / (1 + 0.005)
p = 0.995

# Step 3: Decision
p >= 0.5 → Predict class 1 (99.5% confidence!)
```

### The Loss Function: Binary Cross-Entropy

Unlike linear regression (which uses Mean Squared Error), logistic regression uses **Binary Cross-Entropy Loss**:

```
Loss = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
```

**Why this loss function?**

For a single example:
- If y = 1 (true class is 1):
  - Loss = -log(p)
  - If p is close to 1 → loss is small ✓
  - If p is close to 0 → loss is large ✗

- If y = 0 (true class is 0):
  - Loss = -log(1-p)
  - If p is close to 0 → loss is small ✓
  - If p is close to 1 → loss is large ✗

**Example**:
```
True Label: 1,  Predicted: 0.9  →  Loss = -log(0.9) = 0.105 (good!)
True Label: 1,  Predicted: 0.1  →  Loss = -log(0.1) = 2.303 (bad!)
True Label: 0,  Predicted: 0.1  →  Loss = -log(0.9) = 0.105 (good!)
True Label: 0,  Predicted: 0.9  →  Loss = -log(0.1) = 2.303 (bad!)
```

### Gradient Descent Optimization

Since there's no closed-form solution (like the Normal Equation for linear regression), we use **Gradient Descent**:

**Algorithm**:
```
1. Initialize coefficients to ZERO
2. For each iteration:
   a. Compute predictions: p = sigmoid(X @ θ)
   b. Compute error: error = p - y
   c. Compute gradients: gradients = (1/n) * X^T @ error
   d. Update coefficients: θ = θ - learning_rate * gradients
3. Repeat until convergence
```

**Why zero and not random?** Neural networks must break symmetry with random
weights, but binary cross-entropy with a *linear* model is a **convex** problem:
there is exactly one minimum and every starting point rolls into it. Starting at
zero makes every run of this file byte-for-byte reproducible, and it is what
scikit-learn and statsmodels do. It also gives a memorable first loss - with all
coefficients zero, every prediction is `sigmoid(0) = 0.5`, so the initial loss is
always `-log(0.5) = log(2) = 0.6931`. You will see that number at the start of
every loss curve in this guide.

**Key Parameters**:
- **Learning Rate (α)**: Step size for updates
  - Too large → Overshooting, unstable
  - Too small → Slow convergence
  - Typical values: 0.001 to 0.1 on raw features; up to 0.5 once the features
    are standardized (scaling is what makes the larger steps safe)

- **Iterations**: Number of update steps
  - More iterations → Better convergence
  - Too many → Wasted computation
  - Typical values: 500 to 10,000

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, fit_intercept=True,
                 reg_lambda=0.0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.reg_lambda = reg_lambda
        self.coefficients = None
        self.intercept = None
        self.feature_coefficients = None
        self.losses = []  # Track training progress
```

### Core Methods

1. **`__init__(learning_rate, iterations, fit_intercept, reg_lambda)`** - Initialize model
   - Set hyperparameters for training
   - `learning_rate`: Controls step size (0.01 on raw features, 0.1-0.5 on standardized ones)
   - `iterations`: Number of training steps (typically 500-5000)
   - `fit_intercept`: Whether to prepend a column of ones so the model learns a
     bias term (default `True`). Set `False` only when the data is already
     centered or a bias column is already in `X`.
   - `reg_lambda`: L2 (ridge) penalty strength on the feature coefficients
     (default `0.0` = no penalty). The intercept is never penalized.
     Equivalent to scikit-learn's `C` through `reg_lambda = 1 / C`.

2. **`_sigmoid(z)`** - Private helper method
   - Applies sigmoid activation function
   - Maps linear output to probabilities
   - Handles numerical stability

3. **`fit(X, y)`** - Train the model
   - Implements gradient descent optimization
   - Minimizes binary cross-entropy loss
   - Updates coefficients iteratively
   - Rejects labels that are not 0/1, accepts plain Python lists and 1-D `X`
   - Returns `self`, so `model = LogisticRegression().fit(X, y)` works
   - Records `len(model.losses) == iterations + 1` values: one per iteration
     plus a final one measured *after* the last update, so `losses[-1]` is the
     loss of the model you actually get back

4. **`predict_proba(X)`** - Get probabilities
   - Returns probabilities for class 1
   - Values between 0 and 1
   - Useful for understanding confidence
   - **Shape note**: this returns a 1-D array of shape `(n_samples,)` holding
     `P(y=1)` only. scikit-learn returns an `(n_samples, 2)` matrix instead;
     here you write `1 - p` when you need `P(y=0)`.

5. **`predict(X, threshold)`** - Get class labels
   - Converts probabilities to class labels
   - Default threshold = 0.5
   - Returns 0 or 1

6. **`score(X, y)`** - Calculate accuracy
   - Measures proportion of correct predictions
   - Returns value between 0 and 1
   - 1.0 = perfect classification

7. **`get_coefficients()`** - Get model parameters
   - Returns intercept and feature coefficients
   - Useful for interpretation

---

## Step-by-Step Example

Let's walk through a complete example predicting **student pass/fail** based on study hours and attendance:

### The Data

```python
import numpy as np

# Features: [study_hours, attendance_percentage]
X_train = np.array([
    [1, 20],    # 1 hour study, 20% attendance → Fail
    [2, 40],    # 2 hours study, 40% attendance → Fail
    [3, 60],    # 3 hours study, 60% attendance → Fail  (the hard case)
    [4, 90],    # 4 hours study, 90% attendance → Pass
    [5, 75],    # 5 hours study, 75% attendance → Pass
    [1.5, 30],  # Low effort → Fail
    [2.5, 50],  # Medium effort → Pass
    [3.5, 70],  # High effort → Pass
    [4.5, 90]   # High effort → Pass
])

# Target: 0 = Fail, 1 = Pass
y_train = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])
```

Two deliberate properties of this tiny dataset:

- **The `[3, 60] → Fail` student is the hard case.** They studied *more* than the
  `[2.5, 50] → Pass` student and *less* than the `[3.5, 70] → Pass` student, so
  **no straight line can label all nine rows correctly**. That is realistic — real
  data is not separable either.
- **Attendance is *correlated with* study hours, not a copy of it.** If attendance
  were exactly `20 × hours` in every row, the design matrix would be rank
  deficient, the two coefficients would not be individually identifiable, and the
  "Interpreting Results" section below would be meaningless. Note that `[4, 90]`
  and `[5, 75]` break that exact relationship.

### Training the Model

```python
# Standardize FIRST: attendance spans 20-100 while study hours span 1-5.
# On the raw scale the attendance column dominates every gradient and the
# default learning_rate=0.01 makes the loss climb instead of fall.
mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
X_train_scaled = (X_train - mu) / sd

model = LogisticRegression(learning_rate=0.5, iterations=5000)
model.fit(X_train_scaled, y_train)

print(f"Training accuracy: {model.score(X_train_scaled, y_train):.4f}")
print(f"Loss: {model.losses[0]:.4f} -> {model.losses[-1]:.4f}")
# Training accuracy: 0.7778
# Loss: 0.6931 -> 0.2776
```

**What happens internally**:
1. Coefficients initialized to zero: θ = [0.0, 0.0, 0.0], so every prediction
   starts at `sigmoid(0) = 0.5` and the first loss is exactly `log(2) = 0.6931`
2. For 5000 iterations:
   - Compute z = X @ θ
   - Apply sigmoid: p = 1/(1+e^(-z))
   - Compute loss and gradients
   - Update coefficients
3. Final coefficients learned: θ = [0.5558, 2.0420, 1.1856] = [intercept, coef₁, coef₂]

The training accuracy of **0.7778 is 7 correct out of 9** — it misses both the
`[3, 60]` and the `[2.5, 50]` student. And here is a subtlety worth pausing on:
a straight line *can* get 8 of 9 on this data (drop the `[3, 60]` row and the
remaining eight are cleanly separable at about 2.25 study hours). Gradient descent
did not find that line because **it is not minimizing accuracy — it is minimizing
log-loss**, and log-loss punishes a *confident* wrong answer far more harshly than
accuracy does. Rather than place the boundary where `[3, 60]` gets a confidently
wrong probability, the optimizer prefers a gentler boundary that hedges on two
points. Minimizing log-loss and maximizing accuracy do not pick the same line.

### Making Predictions

```python
# New students
X_test = np.array([
    [2, 30],   # Low study, low attendance
    [4, 85],   # High study, high attendance
    [3, 55]    # Medium study, medium attendance
])

# Scale new data with the TRAINING mu and sd - never recompute them on test data
X_test_scaled = (X_test - mu) / sd

# Get probabilities
probabilities = model.predict_proba(X_test_scaled)
print("Probabilities of passing:", probabilities)
# Output: [0.08038779 0.96969598 0.59623216]

# Get class predictions
predictions = model.predict(X_test_scaled)
print("Predicted outcomes:", predictions)
# Output: [0 1 1]  (Fail, Pass, Pass)
```

The middle student is a confident Pass (0.97), the first a confident Fail (0.08),
and the third sits at 0.60 — barely over the 0.5 threshold. That last number is
the model honestly telling you it is not sure.

### Interpreting Results

```python
coeffs = model.get_coefficients()
print(f"Intercept: {coeffs['intercept']:.4f}")
print(f"Study Hours Coefficient: {coeffs['coefficients'][0]:.4f}")
print(f"Attendance Coefficient: {coeffs['coefficients'][1]:.4f}")
```

**Interpretation**:
- **Positive coefficients** → Feature increases probability of class 1
- **Negative coefficients** → Feature decreases probability of class 1
- **Larger magnitude** → Stronger influence on prediction

**Actual output for the model trained above** (on standardized features):
```
Intercept: 0.5558
Study Hours Coefficient: 2.0420
Attendance Coefficient: 1.1856
```

### The Log-Odds (Logit) Interpretation

"Larger magnitude means stronger influence" is vague. Logistic regression can do
much better than that, because the coefficients have an *exact* meaning.

Start from the model and solve for z:

```
        p = 1 / (1 + e^(-z))

  =>  1/p = 1 + e^(-z)

  =>  p / (1 - p) = e^z                     (a little algebra)

  =>  log(p / (1 - p)) = z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

The quantity `p / (1 - p)` is the **odds** (a probability of 0.75 is odds of 3, or
"3 to 1"). Its logarithm is the **log-odds**, also called the **logit**. So:

> **Logistic regression is a plain linear model — it is just linear in log-odds
> space instead of in probability space.** The `z = X @ theta` that the code
> computes on the line `linear_model = X_with_bias @ self.coefficients` *is* the
> predicted log-odds. The sigmoid is only the last step that converts it back
> into a probability.

That gives every coefficient a precise reading:

- A one-unit increase in xⱼ **adds bⱼ to the log-odds**.
- Adding to a logarithm multiplies the thing itself, so it **multiplies the odds
  by e^(bⱼ)**. That multiplier is called the **odds ratio**.

For our student model (features are standardized, so "one unit" means "one
standard deviation"):

```python
coeffs = model.get_coefficients()
print("Odds ratios:", np.exp(coeffs['coefficients']))
# Odds ratios: [7.70592527 3.27275116]
```

- Study hours: e^2.0420 = 7.71 → one extra standard deviation of study time
  (about 1.3 hours here) multiplies the odds of passing by roughly **7.7**.
- Attendance: e^1.1856 = 3.27 → one extra standard deviation of attendance
  (about 24 percentage points) multiplies the odds by roughly **3.3**.

Because both features were standardized, their coefficients are also directly
comparable: study hours matters more than attendance in this dataset. On raw,
unscaled features you could **not** make that comparison — a coefficient of 0.04
per attendance-percent and 0.8 per study-hour say nothing about relative
importance until you account for the very different units.

---

## Real-World Applications

### 1. **Medical Diagnosis**
Predict disease presence based on symptoms and tests:
- Input: Blood pressure, cholesterol, age, BMI
- Output: Has disease (1) or Healthy (0)
- Example: "85% probability of diabetes"

### 2. **Email Spam Detection**
Classify emails as spam or not spam:
- Input: Word frequencies, sender info, links
- Output: Spam (1) or Not Spam (0)
- Example: "92% probability of spam"

### 3. **Credit Risk Assessment**
Predict loan default risk:
- Input: Income, credit score, debt, employment
- Output: Will default (1) or Won't default (0)
- Example: "15% probability of default"

### 4. **Customer Churn Prediction**
Predict if customer will leave:
- Input: Usage patterns, support tickets, tenure
- Output: Will churn (1) or Stay (0)
- Example: "68% probability of churning"

### 5. **Fraud Detection**
Identify fraudulent transactions:
- Input: Transaction amount, location, time, history
- Output: Fraudulent (1) or Legitimate (0)
- Example: "3% probability of fraud"

### 6. **Marketing Campaign Response**
Predict if customer will respond to campaign:
- Input: Demographics, past purchases, engagement
- Output: Will respond (1) or Won't respond (0)
- Example: "42% probability of conversion"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. The Sigmoid Function

```python
def _sigmoid(self, z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))
```

**Why clip values?**
- Large negative z → e^(-z) becomes huge. `np.exp` overflows float64 past
  z = -709, returning `inf`, and `1 / (1 + inf)` then poisons the gradients with
  `nan`. Clipping the **lower** end is what actually prevents this.
- Large positive z → nothing overflows, but the sigmoid **saturates**: float64
  runs out of resolution and `_sigmoid(z)` returns exactly `1.0` for any
  z greater than about 37 — long before the clip at +500 would ever bite. That
  is harmless here because `_compute_loss` clips probabilities to
  `[1e-15, 1 - 1e-15]` before taking any logarithm, so `log(0)` never happens.
- So the upper clip is cosmetic symmetry; the lower clip is the one doing work.

**How it transforms data**:
```
z = -10  →  σ(z) = 0.00005  (almost 0)
z = -2   →  σ(z) = 0.12     (low probability)
z = 0    →  σ(z) = 0.50     (uncertain)
z = 2    →  σ(z) = 0.88     (high probability)
z = 10   →  σ(z) = 0.99995  (almost 1)
```

### 2. Computing Loss

```python
def _compute_loss(self, y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
```

**Why add epsilon?**
- log(0) is undefined (negative infinity)
- If prediction is exactly 0 or 1, log breaks
- epsilon (tiny value) prevents this: log(1e-15) ≈ -34.5 (large but finite)

**What the formula does**:
- For class 1 samples: Uses first term `y * log(p)`
- For class 0 samples: Uses second term `(1-y) * log(1-p)`
- Averages across all samples

### 3. Gradient Descent Update

```python
# Forward pass
linear_model = X_with_bias @ self.coefficients
y_pred = self._sigmoid(linear_model)

# Backward pass
error = y_pred - y
gradients = (1 / n_samples) * (X_with_bias.T @ error)

# Update
self.coefficients -= self.learning_rate * gradients
```

**Step-by-step**:
1. **Forward pass**: Compute predictions
   - Linear: z = Xθ
   - Non-linear: p = σ(z)

2. **Backward pass**: Compute gradients
   - Error: e = p - y (difference between predicted and true)
   - Gradient: ∇ = (1/n) X^T e (direction to minimize loss)

3. **Update**: Move in opposite direction of gradient
   - θ_new = θ_old - α∇ (α = learning rate)

**Intuition**:
- If prediction too high (p > y): Gradient is positive → decrease coefficients
- If prediction too low (p < y): Gradient is negative → increase coefficients

#### Where That Gradient Comes From (the derivation)

The formula `(1/n) X^T (p - y)` is usually presented as something to memorize.
It is not — it falls out of the chain rule, and the way it falls out is the most
elegant thing about logistic regression. Take a single training example with
label y, and write p = σ(z) where z = xᵀθ.

**Step 1 — differentiate the loss with respect to the probability p.**
```
L = -[ y·log(p) + (1-y)·log(1-p) ]

dL/dp = -y/p + (1-y)/(1-p)
```

**Step 2 — differentiate the sigmoid.** This is the identity that makes
everything work:
```
σ(z) = 1 / (1 + e^(-z))

σ'(z) = e^(-z) / (1 + e^(-z))²
      = [1 / (1 + e^(-z))] · [e^(-z) / (1 + e^(-z))]
      = σ(z) · (1 - σ(z))
      = p(1 - p)
```
The derivative of the sigmoid is expressible in terms of the sigmoid itself.

**Step 3 — chain them together and watch the cancellation.**
```
dL/dz = (dL/dp) · (dp/dz)

      = [ -y/p + (1-y)/(1-p) ] · p(1-p)

      = -y(1-p) + (1-y)p            <- the p and (1-p) denominators CANCEL

      = -y + yp + p - yp

      = p - y
```
Every trace of the sigmoid has vanished. The derivative of the loss with respect
to the linear output is simply **the error, p - y**.

**Step 4 — go the last step to the coefficients.** Since z = xᵀθ, we have
dz/dθ = x, so for one example dL/dθ = (p - y)·x. Averaging over all n examples
and stacking the rows into the matrix X gives exactly what the code computes:
```
grad = (1/n) · X^T (σ(Xθ) - y)     <-  gradients = (1/n) * X_with_bias.T @ error
```

**Why this matters**: that is the *same shape of formula* as the gradient of
mean squared error in linear regression, `(1/n) X^T (Xθ - y)`. Swapping a
squared-error loss for a cross-entropy loss and a linear output for a sigmoid
output changes nothing about the update rule. This is not a coincidence — it is
the defining property of a *canonical link function*, and it is precisely why
binary cross-entropy is paired with the sigmoid instead of, say, MSE. Had we
used MSE with a sigmoid, the `p(1-p)` factor from Step 2 would have survived,
and it goes to zero whenever the model is confidently wrong (p near 0 or 1),
stalling learning exactly when you most need it.

#### Optional: the L2 regularization term

With `reg_lambda > 0` the objective gains a penalty and the gradient gains one
extra term:
```
objective = BCE(y, p) + (reg_lambda / 2n) · Σⱼ θⱼ²

grad_j    = (1/n) X^T (p - y)_j + (reg_lambda / n) · θⱼ      (j != intercept)
```
which is these two lines in `fit()`:
```python
if self.reg_lambda > 0:
    gradients[penalized] += (
        (self.reg_lambda / n_samples) * self.coefficients[penalized]
    )
```
`penalized` is a slice that skips index 0 when `fit_intercept=True`, because
penalizing the bias would make the model's predictions depend on where you
happened to put the origin.

### 4. Making Predictions

```python
def predict(self, X, threshold=0.5):
    probabilities = self.predict_proba(X)
    predictions = (probabilities >= threshold).astype(int)
    return predictions
```

**Threshold selection**:
- **threshold = 0.5**: Balanced (default)
- **threshold > 0.5**: More conservative (fewer positives)
- **threshold < 0.5**: More liberal (more positives)

**Example scenarios**:

```python
# Medical diagnosis (prefer false positives over false negatives)
predictions = model.predict(X, threshold=0.3)  # Lower threshold

# Fraud detection (prefer false negatives over false positives)
predictions = model.predict(X, threshold=0.7)  # Higher threshold
```

---

## Model Evaluation

### Accuracy

The simplest metric:
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**When it works well**:
- Balanced classes (50/50 split)
- Equal cost of errors

**When it's misleading**:
- Imbalanced classes (e.g., 95% class 0, 5% class 1)
- Example: Predict all as 0 → 95% accuracy but useless!

### Confusion Matrix

A better view of model performance:

```
                    Predicted
                  0         1
Actual    0    [TN]      [FP]
          1    [FN]      [TP]

Where:
- TN (True Negative): Correctly predicted 0
- FP (False Positive): Incorrectly predicted 1 (Type I error)
- FN (False Negative): Incorrectly predicted 0 (Type II error)
- TP (True Positive): Correctly predicted 1
```

**Example**:
```python
from sklearn.metrics import confusion_matrix

y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 1, 0, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[3 1]    3 correct 0s, 1 incorrect
#  [1 3]]   1 incorrect, 3 correct 1s
```

### Precision and Recall

**Precision**: Of all predicted positives, how many are correct?
```
Precision = TP / (TP + FP)
```
- High precision → Few false alarms
- Important when false positives are costly

**Recall** (Sensitivity): Of all actual positives, how many did we find?
```
Recall = TP / (TP + FN)
```
- High recall → Few missed positives
- Important when false negatives are costly

**Trade-off**:
```
High Threshold (0.8):  High Precision, Low Recall
Low Threshold (0.2):   Low Precision, High Recall
```

### F1 Score

Harmonic mean of precision and recall:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

- Balances precision and recall
- Good for imbalanced datasets
- Range: 0 to 1 (1 is best)

### Example: Complete Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Get predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
```

---

## Hyperparameter Tuning

### Learning Rate Selection

**Effects of different learning rates**:

| Learning Rate | Effect | When to Use |
|---------------|--------|-------------|
| **0.001 - 0.01** | Slow, stable convergence | Large datasets, start here |
| **0.01 - 0.1** | Moderate speed | Most cases, good default |
| **0.1 - 1.0** | Fast but may oscillate | Small datasets, scaled features |
| **> 1.0** | May diverge | Rarely useful |

**How to choose**:
1. Start with 0.01
2. Plot loss curve
3. If loss decreases smoothly → good
4. If loss oscillates → decrease learning rate
5. If loss decreases too slowly → increase learning rate

### Iterations Selection

**Guidelines**:
- Plot loss curve during training
- Stop when loss plateaus (no improvement)
- Typical range: 500 - 5000 iterations
- More iterations ≠ better (after convergence)

### Regularization Strength (`reg_lambda`)

| reg_lambda | Effect | When to Use |
|------------|--------|-------------|
| **0.0** | No penalty, plain maximum likelihood | Default. Plenty of data, no separation |
| **0.1 - 1.0** | Mild shrinkage | Most real problems; `1.0` matches scikit-learn's default `C=1.0` |
| **10 - 100** | Heavy shrinkage | Few samples relative to features, or perfectly separable classes |

`reg_lambda` and scikit-learn's `C` are the same knob read from opposite ends:
`reg_lambda = 1 / C`. Larger `reg_lambda` (or smaller `C`) means more
regularization. Measured on 800 standardized samples with 5 features,
`reg_lambda=1.0` reproduces `sklearn.linear_model.LogisticRegression(C=1.0)`'s
weight vector to within 6e-09.

**How to choose**: start at 0.0. If the coefficients come out implausibly large,
or the training accuracy is 1.0000 while the test accuracy is much lower, or the
loss keeps creeping toward zero without ever settling, raise it.

### Example: Finding Optimal Hyperparameters

```python
# Try different combinations
learning_rates = [0.001, 0.01, 0.1]
iterations_list = [500, 1000, 2000]

best_score = 0
best_params = {}

for lr in learning_rates:
    for iters in iterations_list:
        model = LogisticRegression(learning_rate=lr, iterations=iters)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        if score > best_score:
            best_score = score
            best_params = {'lr': lr, 'iterations': iters}

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.4f}")
```

---

## Visualizing Logistic Regression

### 1. Decision Boundary (2D)

```python
import numpy as np
import matplotlib.pyplot as plt

# Train model on 2D data
model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train, y_train)

# Create mesh grid
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict probabilities on mesh
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
plt.colorbar(label='P(y=1)')
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Plot data points
plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], 
            c='blue', label='Class 0', edgecolors='k', s=100)
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], 
            c='red', label='Class 1', edgecolors='k', s=100)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.legend()
plt.show()
```

### 2. Loss Curve

```python
plt.figure(figsize=(10, 6))
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()
```

**What to look for**:
- Smooth decrease → Good convergence
- Oscillations → Learning rate too high
- Flat immediately → Learning rate too low or already converged
- Still decreasing at end → Need more iterations

### 3. Sigmoid Function Visualization

```python
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid, linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision threshold')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5)
plt.xlabel('z (linear output)')
plt.ylabel('σ(z) (probability)')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

---

## Key Concepts to Remember

### 1. **Logistic Regression is for Classification**
Despite the name, it's a classification algorithm, not regression!

### 2. **Outputs are Probabilities**
- Always between 0 and 1
- Can be interpreted as confidence
- Use threshold to convert to class labels

### 3. **Feature Scaling is Important**
Always standardize features for faster, more stable convergence:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Or with plain NumPy, which is all this repo needs — fit the statistics on the
**training** data only, then reuse them for the test data:
```python
mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
X_train_scaled = (X_train - mu) / sd
X_test_scaled  = (X_test  - mu) / sd     # same mu, sd - do NOT refit
```

This is not optional advice you can skip. Gradient descent takes one step size
for *all* coefficients, so a feature measured in the hundreds produces gradients
hundreds of times larger than a feature measured in single digits. The step that
is right for one is catastrophic for the other. On the raw student data of the
Step-by-Step Example (attendance 20-100, hours 1-5) with the default
`learning_rate=0.01`, the loss **rises instead of falling**: it starts at 0.6931,
thrashes between 0.76 and 5.62, ends at 2.5968, and goes *up* on 555 of the 1000
steps. The model settles on predicting Pass for all nine students and scores
0.5556 — exactly the majority-class baseline, no better than always guessing
Pass without looking at the features at all. Standardize first
and the same data converges smoothly and monotonically from 0.6931 to 0.2776 for
an accuracy of 0.7778. That is why every example in this guide scales before it
fits.

### 4. **No Closed-Form Solution**
Unlike linear regression, we must use iterative optimization (gradient descent)

### 5. **Assumptions**
Logistic regression assumes:
- Binary outcome (can be extended to multiclass)
- Linear decision boundary
- **Observations are independent of one another** (this is *not* the Naive Bayes
  assumption that the *features* are independent — logistic regression is
  perfectly happy with correlated features)
- No *perfect* multicollinearity among features. Two features that are exact
  multiples of each other make the design matrix rank deficient and the
  individual coefficients unidentifiable — the fit still works, but you can no
  longer say what each feature contributed
- Large sample size for reliable estimates

### 6. **Limitations**
- Can only draw a **linear** decision boundary — it cannot separate classes that
  need a curved boundary unless you engineer the features yourself (add x², x·y,
  and so on)
- Assumes a linear relationship between features and log-odds
- Sensitive to outliers
- May underperform with highly correlated features (the fit is fine; the
  *interpretation* of individual coefficients becomes unstable)
- **Perfectly separable data is a problem, not a gift.** If some straight line
  splits the classes with no mistakes, the unregularized maximum-likelihood
  solution has no finite optimum: pushing the coefficients larger always makes
  the loss a little smaller, so they grow without bound. On the four-point set
  `X = [[-2], [-1], [1], [2]]`, `y = [0, 0, 1, 1]` the coefficient reaches 8.5
  after 20,000 iterations and is still climbing. This is exactly what the
  `reg_lambda` (L2) option fixes — with `reg_lambda=1.0` the same fit settles at
  1.01 and stops. Run `python _4_logistic_regression.py` to see DEMO 3 do this.

---

## Simplifications vs scikit-learn

This implementation is deliberately small enough to read in one sitting. Here is
exactly what it does and does not do, compared with
`sklearn.linear_model.LogisticRegression`, so nothing on this page overpromises.

**Implemented, and verified to match scikit-learn:**

| Feature | Status | Measured agreement |
|---------|--------|--------------------|
| Unregularized binary MLE | Yes (`reg_lambda=0.0`) | max coefficient difference 1.1e-08 vs `penalty=None` on 800x5 standardized data |
| L2 (ridge) penalty | Yes (`reg_lambda>0`) | max coefficient difference 5.8e-09 vs `penalty='l2', C=1.0` on the same data |
| Intercept excluded from the penalty | Yes | matches scikit-learn's convention |
| Probability output, thresholding, accuracy | Yes | identical predicted labels on every fit run to convergence; 98.25% label agreement on the breast-cancer example, where 2000 iterations stop short of convergence |

**Not implemented here** (each is a deliberate simplification, not an oversight):

1. **L1 (lasso) penalty and elastic net.** Real logistic regression can drive
   coefficients to exactly zero for feature selection, using the
   soft-thresholding operator `shrink(g, alpha) = sign(g)·max(|g| - alpha, 0)`.
   Plain gradient descent cannot do this, because the L1 penalty is not
   differentiable at zero — it needs a proximal or coordinate-descent solver, a
   substantially different optimizer. Consequence: use `reg_lambda` (L2) when you
   want shrinkage; reach for scikit-learn when you want sparsity.
2. **Multinomial / softmax regression.** This class handles two classes only, and
   `fit()` raises a `ValueError` if `y` contains anything but 0 and 1. The
   canonical extension replaces the sigmoid with
   `softmax(z)_k = e^(z_k) / Σⱼ e^(z_j)` and the binary cross-entropy with
   categorical cross-entropy. Consequence: for K classes you would train K
   one-vs-rest copies of this model yourself.
3. **A second-order solver.** scikit-learn defaults to `lbfgs`, and statsmodels
   uses Newton-Raphson / IRLS, both of which use curvature information (the
   Hessian `X^T diag(p(1-p)) X`) and converge in tens of iterations rather than
   thousands. First-order gradient descent is used here because it makes the
   gradient formula visible in the code. Consequence: you must choose
   `learning_rate` and `iterations` yourself, and you must standardize features.
4. **A convergence check.** There is no `tol` and no early stopping — the loop
   always runs exactly `iterations` times. Consequence: inspect `model.losses`
   to see whether you converged; a flat tail means you can lower `iterations`.
5. **Class weights and sample weights.** Every sample counts equally. On heavily
   imbalanced data, scikit-learn's `class_weight='balanced'` would reweight the
   loss; here you would resample the data or move the decision threshold instead
   (see the threshold discussion above).

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# IMPORTANT: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = LogisticRegression(learning_rate=0.1, iterations=2000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Evaluate model
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show some predictions with probabilities
print("\nSample Predictions:")
for i in range(5):
    print(f"True: {y_test[i]}, Predicted: {y_pred[i]}, Probability: {y_proba[i]:.4f}")

# Plot training loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

# Examine coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.4f}")
print(f"Number of features: {len(coeffs['coefficients'])}")

# Find most important features
feature_importance = np.abs(coeffs['coefficients'])
top_features = np.argsort(feature_importance)[-5:]
print("\nTop 5 most important features:")
for idx in top_features[::-1]:
    print(f"  {data.feature_names[idx]}: {coeffs['coefficients'][idx]:.4f}")
```

---

## Conclusion

Logistic Regression is a fundamental and powerful algorithm for binary classification! By understanding:
- How sigmoid transforms linear outputs to probabilities
- Why the model is *linear in log-odds*, so every coefficient is an odds ratio
- How the chain rule collapses the cross-entropy gradient to `(1/n) X^T (p - y)`
- How gradient descent optimizes the model
- How to interpret probabilities and make decisions
- How to evaluate classification performance

You've gained a crucial tool in your machine learning toolkit! 🎯

**When to Use Logistic Regression**:
- ✅ Binary classification problems
- ✅ Need probability estimates
- ✅ Want interpretable model (coefficients are odds ratios)
- ✅ Classes separable by a roughly linear boundary
- ✅ Need fast training and predictions

**When to Use Something Else**:
- ❌ Multi-class with many classes → Use multinomial logistic regression
- ❌ Non-linear decision boundaries → Use kernel methods, trees, or neural networks
- ❌ Very large datasets → Use SGD (stochastic gradient descent)
- ❌ Need feature selection → Use Lasso (L1) regularization

**Next Steps**:
- Try with your own classification data
- Experiment with different thresholds
- Compare with scikit-learn's LogisticRegression
- Experiment with `reg_lambda` — L2 regularization **is** implemented here
  (`reg_lambda = 1 / C`); L1 / lasso is not, and the
  [Simplifications](#simplifications-vs-scikit-learn) section explains why
- Explore ROC curves and AUC scores
- Study multinomial logistic regression for multi-class

Happy coding! 💻🎯


