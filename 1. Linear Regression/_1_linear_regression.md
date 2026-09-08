# Simple Linear Regression from Scratch: A Comprehensive Guide

Welcome to the world of Linear Regression! 📈 In this comprehensive guide, we'll explore how to predict outcomes using a single input feature. Think of it as finding the best-fit line through a scatter plot of points!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is Linear Regression?](#what-is-linear-regression)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Visualizing Linear Regression](#visualizing-linear-regression)
9. [Key Concepts to Remember](#key-concepts-to-remember)
10. [Advantages & Limitations](#advantages--limitations)
11. [Complete Usage Example](#complete-usage-example)
12. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is exactly what the `if __name__ == "__main__":` block in
`_1_linear_regressions.py` runs. You can also just run the file directly:

```bash
python _1_linear_regressions.py
```

No dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Linear Regression from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _1_linear_regressions.py  (the __main__ block runs this)
# Or paste the LinearRegression class from _1_linear_regressions.py above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the LinearRegression class here (from _1_linear_regressions.py) ----
# class LinearRegression: ...

np.random.seed(42)

# --- Demo 1: recover a line we planted exactly ---
print("=" * 55)
print("DEMO 1 - Exact recovery: salary = 25000 + 5000 * years")
print("=" * 55)

X_years = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y_salary = np.array([30000, 35000, 40000, 45000, 50000,
                     55000, 60000, 65000, 70000, 75000])

salary_model = LinearRegression()
salary_model.fit(X_years, y_salary)
coeffs = salary_model.get_coefficients()

print(f"Recovered intercept : {coeffs['intercept']:.2f}   (true 25000.00)")
print(f"Recovered slope     : {coeffs['slope']:.2f}   (true 5000.00)")
print(f"Train R2            : {salary_model.score(X_years, y_salary):.4f}")

# A flat 1-D array works too - fit/predict reshape it for you.
future = salary_model.predict(np.array([11, 12, 15]))
print(f"Predictions for 11, 12, 15 years: {np.round(future, 2)}")

# --- Demo 2: noisy data with a real held-out test split ---
print("\n" + "=" * 55)
print("DEMO 2 - Noisy data: y = 3.5x - 2.0 + noise")
print("=" * 55)

n = 200
X_noisy = np.random.uniform(0, 10, n).reshape(-1, 1)
y_noisy = 3.5 * X_noisy.ravel() - 2.0 + np.random.randn(n) * 1.5

# Shuffle before slicing so train and test cover the same x range,
# and slice at the SAME index so the two sets never overlap.
# X[:150] / X[150:] is disjoint; X[:150] / X[50:] would leak 100 rows.
idx = np.random.permutation(n)
X_noisy, y_noisy = X_noisy[idx], y_noisy[idx]
X_train, X_test = X_noisy[:150], X_noisy[150:]
y_train, y_test = y_noisy[:150], y_noisy[150:]

noisy_model = LinearRegression()
noisy_model.fit(X_train, y_train)
nc = noisy_model.get_coefficients()

print(f"True   b0=-2.00  b1=3.50")
print(f"Fitted b0={nc['intercept']:.4f}  b1={nc['slope']:.4f}")
print(f"Train R2 : {noisy_model.score(X_train, y_train):.4f}")
print(f"Test  R2 : {noisy_model.score(X_test, y_test):.4f}")

preds = noisy_model.predict(X_test)
print("\nSample predictions (x, true, predicted):")
for i in range(5):
    print(f"  x={X_test[i, 0]:5.2f}  true={y_test[i]:7.2f}  pred={preds[i]:7.2f}")

# --- Demo 3: what R2 looks like when there is no signal at all ---
print("\n" + "=" * 55)
print("DEMO 3 - Sanity check: R2 on pure noise")
print("=" * 55)

X_junk = np.random.randn(100, 1)
y_junk = np.random.randn(100)  # independent of X - nothing to learn

junk_model = LinearRegression()
junk_model.fit(X_junk, y_junk)
print(f"Train R2 on pure noise: {junk_model.score(X_junk, y_junk):.4f}"
      "  (near 0 = no signal)")

# --- Demo 4: the closed form matches the matrix solution ---
print("\n" + "=" * 55)
print("DEMO 4 - Closed form b1 = cov(x,y)/var(x) matches the code")
print("=" * 55)

x_flat = X_train.ravel()
b1_closed = (np.sum((x_flat - x_flat.mean()) * (y_train - y_train.mean()))
             / np.sum((x_flat - x_flat.mean()) ** 2))
b0_closed = y_train.mean() - b1_closed * x_flat.mean()

print(f"Closed form      : b0={b0_closed:.6f}  b1={b1_closed:.6f}")
print(f"Normal Equation  : b0={nc['intercept']:.6f}  b1={nc['slope']:.6f}")
print(f"Max difference   : {max(abs(b0_closed - nc['intercept']), abs(b1_closed - nc['slope'])):.2e}")
```

Expected output:
```
=======================================================
DEMO 1 - Exact recovery: salary = 25000 + 5000 * years
=======================================================
Recovered intercept : 25000.00   (true 25000.00)
Recovered slope     : 5000.00   (true 5000.00)
Train R2            : 1.0000
Predictions for 11, 12, 15 years: [ 80000.  85000. 100000.]

=======================================================
DEMO 2 - Noisy data: y = 3.5x - 2.0 + noise
=======================================================
True   b0=-2.00  b1=3.50
Fitted b0=-1.8161  b1=3.4846
Train R2 : 0.9809
Test  R2 : 0.9784

Sample predictions (x, true, predicted):
  x= 8.15  true=  28.17  pred=  26.60
  x= 9.70  true=  30.72  pred=  31.98
  x= 3.01  true=   7.49  pred=   8.67
  x= 7.32  true=  24.06  pred=  23.69
  x= 6.60  true=  20.62  pred=  21.18

=======================================================
DEMO 3 - Sanity check: R2 on pure noise
=======================================================
Train R2 on pure noise: 0.0216  (near 0 = no signal)

=======================================================
DEMO 4 - Closed form b1 = cov(x,y)/var(x) matches the code
=======================================================
Closed form      : b0=-1.816144  b1=3.484627
Normal Equation  : b0=-1.816144  b1=3.484627
Max difference   : 3.55e-15
```

Notice Demo 1: with noise-free data the model recovers the planted
intercept and slope *exactly*, and Demo 3 shows what the opposite looks
like - an R2 near 0 when there is no relationship to find at all.

---

## What is Linear Regression?

Linear Regression is the simplest and most fundamental machine learning algorithm. It finds the **best-fit straight line** through data points to model the relationship between a single input feature and a target variable.

**Real-world analogy**: 
Imagine plotting house prices against their square footage on a graph. Linear regression draws the straight line that best represents this relationship, allowing you to predict prices for houses you haven't seen yet!

### The Mathematical Equation

The formula for simple linear regression is:

```
y = b₀ + b₁x
```

Where:
- **y** = target variable (what we want to predict)
- **x** = input feature (independent variable)
- **b₀** = intercept (where the line crosses the y-axis)
- **b₁** = slope (how steep the line is)

**Example**: If predicting salary from years of experience:
```
Salary = 25000 + 5000 × Years_of_Experience
```
- Intercept (b₀) = $25,000 (starting salary)
- Slope (b₁) = $5,000 (salary increase per year)

---

## The Mathematical Foundation

### What "best fit" actually means: the cost function

Before any formula, we have to say what makes one line better than another.
For a candidate line, the **residual** of point *i* is how far the true value
sits above or below the line:

```
residual_i = y_i - (b₀ + b₁x_i)
```

We square each residual (so that being 3 too high is penalised the same as
being 3 too low, and so big misses hurt disproportionately) and add them up.
That sum is the **cost function**:

```
J(b₀, b₁) = Σ (y_i - b₀ - b₁x_i)²
```

"Fitting the model" means finding the (b₀, b₁) that makes J as small as
possible. This is why the method is called **Ordinary Least Squares** — we
are literally minimising a sum of squares.

### Deriving the closed form

J is a smooth bowl-shaped (convex) function of b₀ and b₁, so its minimum is
wherever both partial derivatives are zero. Differentiate:

```
∂J/∂b₀ = -2 Σ (y_i - b₀ - b₁x_i)      = 0
∂J/∂b₁ = -2 Σ x_i(y_i - b₀ - b₁x_i)   = 0
```

Dropping the -2 and solving the two equations together gives the classic
result for simple linear regression:

```
b₁ = Σ (x_i - x̄)(y_i - ȳ) / Σ (x_i - x̄)²
b₀ = ȳ - b₁x̄
```

Read the slope formula out loud and it is just **cov(x, y) / var(x)**: how
much x and y move together, divided by how much x moves on its own. The
intercept formula says the fitted line always passes through the centre of
mass of the data, the point (x̄, ȳ).

This is worth internalising because it makes simple linear regression
hand-computable, and because it is the same answer the matrix code returns.
Demo 4 of `_1_linear_regressions.py` checks exactly this and prints a
difference of about `3.55e-15` — floating-point noise, i.e. the same number.

You can verify it yourself:

```python
x = X_train.ravel()
b1 = np.sum((x - x.mean()) * (y_train - y_train.mean())) / np.sum((x - x.mean()) ** 2)
b0 = y_train.mean() - b1 * x.mean()

coeffs = model.get_coefficients()
print(f"Closed form     : b0={b0:.4f}  b1={b1:.4f}")
print(f"Normal Equation : b0={coeffs['intercept']:.4f}  b1={coeffs['slope']:.4f}")
```

On the salary data above both lines print
`b0=25000.0000  b1=5000.0000` — the same answer by two different routes.

### Matrix Representation

Even simple linear regression can be expressed using matrices:

```
Y = Xθ
```

Where:
- **Y** is an (n×1) vector of target values
- **X** is an (n×2) matrix (n samples, 1 feature + bias)
- **θ** is a (2×1) vector of coefficients [b₀, b₁]

### The Normal Equation

To find the optimal coefficients that minimize prediction error, we use the **Normal Equation**:

```
θ = (XᵀX)⁻¹Xᵀy
```

This gives us the best slope and intercept in one calculation!

**Where it comes from**: it is the same derivation as above, written in
matrix notation. The cost function becomes

```
J(θ) = (y - Xθ)ᵀ(y - Xθ)
```

Differentiating with respect to the vector θ and setting the result to zero:

```
∂J/∂θ = -2Xᵀ(y - Xθ) = 0
   =>   XᵀXθ = Xᵀy          <- the "normal equations"
   =>   θ = (XᵀX)⁻¹Xᵀy
```

So the matrix formula is not a separate trick to memorise — it is the
two-equation derivation above, generalised to any number of features.

**Breaking it down**:
1. **Xᵀ** = transpose of X matrix
2. **XᵀX** = matrix multiplication
3. **(XᵀX)⁻¹** = inverse of the matrix
4. **Xᵀy** = transpose of X multiplied by y

### When XᵀX cannot be inverted

The formula above quietly assumes XᵀX has an inverse. It does not when two
columns of X are perfectly collinear — the matrix is **singular**. This is
easy to walk into:

- A **constant feature column**. Combined with the bias column of ones, the
  two are perfectly correlated. This is the classic **dummy-variable trap**:
  one-hot encoding a category into *k* columns when you already have an
  intercept gives *k* columns that always sum to 1.
- A **duplicated or rescaled feature** (e.g. a length in metres and the same
  length in feet).
- **More features than samples** (n < p), where XᵀX is always singular.

Here is what that costs you. Take `X = [[1,7],[2,7],[3,7],[4,7],[5,7]]` and
`y = [2,4,6,8,10]` — the second column is constant, and the true relationship
is simply `y = 2x`:

| Solver | Result |
|---|---|
| `np.linalg.inv(XᵀX)` | coefficients `[5.625, 2.0, 0.65625]`, **R² = -12.05** |
| `np.linalg.pinv(X)` | coefficients `[0.0, 2.0, 0.0]`, **R² = 1.0** |

The exact inverse does not raise an error here — the condition number of
XᵀX is about `3.9e17`, so numerically it is "invertible" and returns
confident nonsense. (With a *duplicated* column it fails louder, raising
`LinAlgError: Singular matrix`.)

That is why the implementation uses the **pseudo-inverse**:

```
θ = X⁺y        where X⁺ is the Moore-Penrose pseudo-inverse of X
```

The Moore-Penrose pseudo-inverse returns the **minimum-norm least-squares
solution** — among all the coefficient vectors that fit the data equally
well, it picks the smallest one — instead of failing or returning garbage.
On full-rank, well-conditioned data it lands on the answer `inv` gives to
machine precision, so you lose nothing.

### Why the code never forms XᵀX

Look closely at that formula: it is `pinv(X)`, not `pinv(XᵀX)Xᵀ`. In algebra
those are **the same operator** — `pinv(XᵀX)Xᵀ = pinv(X)` is an identity, and
θ = (XᵀX)⁻¹Xᵀy is still the equation being solved. In floating point they are
not the same at all, and the reason is one of the most useful lessons in
numerical linear algebra.

**Forming XᵀX squares the condition number.** The singular values of XᵀX are
the *squares* of those of X. So any rank cutoff applied to XᵀX is testing
squared numbers, and NumPy's `pinv` — which zeroes singular values smaller
than `1e-15` times the largest — throws away a direction the data genuinely
had as soon as cond(X) exceeds `1/sqrt(1e-15)` ≈ **3×10⁷**.

That threshold is not exotic. A single feature with a large offset and a
modest spread — a meter reading, a price in cents, a Unix timestamp — reaches
it on its own, with one feature and no collinearity anywhere. Take
`x = 100000 + Uniform(0, 100)` and `y = 3(x - 100000) + 50 + N(0, 1)`, with
n = 100. The design matrix is full rank (`np.linalg.matrix_rank` = 2) and
cond(X) = 3.5×10⁸. One draw (`np.random.RandomState(0)`; other seeds move only
the last digits):

| Solve | b₁ (true 3.0) | RSS | R² |
|---|---|---|---|
| `pinv(XᵀX) @ Xᵀ @ y` | 0.0019 | 7.5e5 | 0.0013 |
| `pinv(X) @ y` | 2.9994 | 99.2 | **0.9999** |
| scikit-learn | 2.9994 | 99.2 | **0.9999** |

Same algebra, same identity, completely different answers — because the first
route took the pseudo-inverse of a matrix whose conditioning it had already
destroyed. Keep θ = (XᵀX)⁻¹Xᵀy in your head as the formula; solve it by taking
the SVD of X.

scikit-learn does the same SVD-based least-squares solve on X, and goes one
step further by centring X and y before solving. Keep the example above and
just enlarge the offset, and the two part company: at an offset of 10⁸
(cond(X) = 3.5×10¹⁴) our uncentred `pinv(X) @ y` still scores R² = 0.999867,
but at 10⁹ (cond(X) = 3.5×10¹⁶) it has collapsed to R² = 0.000000, while
scikit-learn's centred solve still scores 0.999867 — and is still scoring
0.999867 at an offset of 10¹³, where cond(X) = 3.5×10²⁴. That is a real
limitation of the code here, and a deliberate one: centring X and y and
reconstructing the intercept afterwards would stop `fit()` being a direct
transcription of θ = (XᵀX)⁻¹Xᵀy, which is the formula this guide teaches.

One subtlety worth knowing: on rank-deficient data our *fitted values* (the
predictions on the training rows) match scikit-learn exactly, but the
individual coefficients — and so the predictions at new points — can differ.
For example, fitting the single point `X = [[3.]]`, `y = [9.]`, both models
predict 9.0 at x = 3, but ours predicts 14.4 at x = 5 where scikit-learn
predicts 9.0. scikit-learn centres the data before solving, so its "smallest"
solution minimises the norm of the slopes only, while ours includes the
intercept. Both are legitimate
least-squares answers — which is really the point: when features are
collinear there is no unique set of coefficients, so you should not read
meaning into any single one of them.

### Why a closed form at all?

Most machine learning algorithms have no formula for their answer; they
search for it with gradient descent, taking small downhill steps:

```
θ := θ - α * ∂J/∂θ      (repeat until it stops improving)
```

Linear regression is one of the rare cases where you can set the derivative
to zero and *solve* for the optimum directly. There is no learning rate, no
number of epochs, no convergence to check — which is why `fit()` in this
implementation is one line of linear algebra and has no loop at all.

The catch is cost: inverting a (p+1)x(p+1) matrix is roughly **O(p³)**, and
the SVD the code actually uses is the same order in p. With
one feature that is nothing. With tens of thousands of features it becomes
slower than gradient descent, which is why large-scale linear models are
usually fitted iteratively instead.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class LinearRegression:
    def __init__(self):
        self.coefficients = None           # [intercept, slope_1, ..., slope_p]
        self.intercept = None              # b0, i.e. coefficients[0]
        self.feature_coefficients = None   # the slopes only, coefficients[1:]
```

All three attributes stay `None` until `fit()` is called. Every other method
checks this and raises a clear `ValueError("Model is not fitted yet. Call
fit(X, y) first.")` rather than failing with a cryptic matrix error.

### Core Methods

1. **`fit(X, y)`** - Train the model
   - Accepts a plain list, a flat `(n_samples,)` array, or `(n_samples, 1)`
   - Adds bias term (column of ones)
   - Calculates coefficients using Normal Equation
   - Stores intercept and slope separately

2. **`predict(X)`** - Make predictions
   - Adds bias term to new data
   - Applies the linear equation: y = b₀ + b₁x
   - Returns predicted values

3. **`get_coefficients()`** - Get model parameters
   - Returns a dict with `'intercept'`, `'slope'` and `'coefficients'`
   - `'slope'` is the *first* feature coefficient; `'coefficients'` (same as
     the `.feature_coefficients` attribute) holds all of them, which matters
     only if you fit more than one feature
   - Useful for understanding the relationship

4. **`score(X, y)`** - Calculate R² score
   - Measures how well the line fits the data
   - Returns a value **≤ 1**: 1 = perfect fit, 0 = no better than predicting
     the mean, and **negative = worse than predicting the mean**. R² has no
     lower bound — see the interpretation ladder below.

---

## Step-by-Step Example

Let's walk through a complete example predicting **salary** based on **years of experience**:

### The Data

```python
import numpy as np

# Years of experience
X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# Corresponding salaries
y_train = np.array([30000, 35000, 40000, 45000, 50000, 
                     55000, 60000, 65000, 70000, 75000])
```

### Training the Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

**What happens internally**:
1. Adds a column of ones to X_train → becomes [[1, 1], [1, 2], [1, 3], ...]
2. Computes the pseudo-inverse X⁺ of that matrix
3. Multiplies it by y
4. Stores the resulting coefficients [intercept, slope]

### Working the numbers by hand

Let's actually multiply the matrices out, using just the first three rows so
the arithmetic stays small. X (with the bias column) and y are:

```
X = [[1, 1],        y = [30000,
     [1, 2],             35000,
     [1, 3]]             40000]
```

**Step 1 — XᵀX** (a 2x2 matrix; the entries are n, Σx, Σx and Σx²):

```
XᵀX = [[3,  6],        3 samples, Σx = 1+2+3 = 6
       [6, 14]]        Σx² = 1+4+9 = 14
```

**Step 2 — invert it.** For a 2x2 matrix the inverse is
`1/det * [[d, -b], [-c, a]]`, and here `det = 3*14 - 6*6 = 6`:

```
(XᵀX)⁻¹ = 1/6 * [[14, -6],   =  [[ 2.3333, -1.0],
                  [-6,  3]]      [-1.0,     0.5]]
```

**Step 3 — Xᵀy**:

```
Xᵀy = [ Σy,   ]  = [105000,
        Σx*y ]      220000]
```
(Σy = 30000+35000+40000 = 105000; Σxy = 1*30000 + 2*35000 + 3*40000 = 220000)

**Step 4 — multiply**:

```
θ = (XᵀX)⁻¹Xᵀy = [ 2.3333*105000 + (-1.0)*220000,
                   -1.0*105000   +   0.5*220000  ]
                = [25000, 5000]
```

There it is: **intercept = 25,000 and slope = 5,000**, the same numbers the
code prints and the same numbers the closed form
`b₁ = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²` gives. The whole algorithm is these four
multiplications.

### Making Predictions

```python
# Predict salaries for 11, 12, and 15 years of experience
X_test = np.array([11, 12, 15]).reshape(-1, 1)
predictions = model.predict(X_test)
print("Predicted salaries:", predictions)
```

### Interpreting Coefficients

```python
coeffs = model.get_coefficients()
print(f"Intercept: ${coeffs['intercept']:.2f}")
print(f"Slope: ${coeffs['slope']:.2f} per year")
```

**What do these mean?**
- **Intercept**: Base salary (when experience = 0)
- **Slope**: Salary increase for each additional year of experience

For example:
- Intercept = $25,000 → Starting salary
- Slope = $5,000 → Each year adds $5,000 to salary

---

## Real-World Applications

### 1. **Sales Forecasting**
Predict sales based on advertising budget:
- Input: Advertising spend
- Output: Sales revenue
- Example: "For every $1000 spent on ads, sales increase by $5000"

### 2. **Real Estate**
Predict house price based on size:
- Input: Square footage
- Output: House price
- Example: "Each additional square foot adds $150 to the price"

### 3. **Medical Research**
Predict disease progression:
- Input: Time since diagnosis
- Output: Disease severity score
- Example: "Disease severity increases by 2.5 points per year"

### 4. **Economics**
Predict GDP growth:
- Input: Investment rate
- Output: GDP growth percentage
- Example: "1% increase in investment → 0.3% GDP growth"

### 5. **Education**
Predict test scores:
- Input: Study hours
- Output: Test score
- Example: "Each hour of study increases score by 5 points"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Adding the Bias Term

```python
X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
```

**Why?** The intercept (b₀) represents the value when x=0. By adding a column of ones, we can include it in our matrix multiplication.

**Example transformation**:
```
Before: [[1],        After: [[1, 1],
         [2],                [1, 2],
         [3]]                [1, 3]]
```

### 2. Normal Equation Implementation

```python
self.coefficients = np.linalg.pinv(X_with_bias) @ y
```

**Step-by-step**:
1. `np.linalg.pinv(X_with_bias)` → the pseudo-inverse X⁺, which NumPy builds
   from the SVD of X
2. `@ y` → multiply it by the target vector
3. Result → [intercept, slope]

This *is* the Normal Equation θ = (XᵀX)⁻¹Xᵀy, not a different method:
`pinv(XᵀX)Xᵀ` and `pinv(X)` are the same operator. The hand calculation above
works the formula out the literal way, which is the right way to understand
it; the code takes the route that is safe in floating point.

**Why `pinv(X)` and not `inv(XᵀX)`?** Two separate reasons, and both matter:

- `inv` breaks on perfectly collinear features: it either raises
  `LinAlgError` or, worse, silently returns nonsense (the constant-column
  example earlier scores R² = -12.05 with `inv` and R² = 1.0 with `pinv`).
- Even using `pinv`, building `XᵀX` first squares the condition number and
  silently drops a real direction of the data above cond(X) ≈ 3×10⁷ — on the
  meter-reading example, R² falls from 0.9999 to 0.0013.

On full-rank, well-conditioned data all of these agree to machine precision,
so taking the SVD route costs nothing and removes two whole classes of silent
failure. See *"When XᵀX cannot be inverted"* and *"Why the code never forms
XᵀX"* under The Mathematical Foundation above.

### 3. Making Predictions

```python
return X_with_bias @ self.coefficients
```

**What it does**: For each x value, calculates: y = b₀×1 + b₁×x

**Example calculation**:
```
For x = 5:
y = b₀×1 + b₁×5 = 25000 + 5000×5 = 50000
```

### 4. R² Score (Model Evaluation)

```python
ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares

# If y is constant, ss_tot is 0 and the ratio would divide by zero.
# scikit-learn's convention: 1.0 for a perfect fit, 0.0 otherwise.
# (We test with a RELATIVE tolerance, scikit-learn tests ss_res == 0
#  exactly, so a merely near-perfect fit scores 1.0 here, 0.0 there.)
if ss_tot == 0:
    return 1.0 if np.allclose(y_pred, y) else 0.0

r2_score = 1 - (ss_res / ss_tot)
```

**Interpretation**:
- **R² = 1.0** → Perfect predictions (all points on the line)
- **R² = 0.9** → Excellent fit (90% of variance explained)
- **R² = 0.7** → Good fit (70% of variance explained)
- **R² = 0.5** → Moderate fit (50% of variance explained)
- **R² = 0.0** → No better than predicting the average
- **R² < 0.0** → Worse than predicting the average

---

## Visualizing Linear Regression

Here's how you can visualize your linear regression model:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create and train model
X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y_train = np.array([30000, 35000, 40000, 45000, 50000, 
                     55000, 60000, 65000, 70000, 75000])

model = LinearRegression()
model.fit(X_train, y_train)

# Create predictions for plotting
X_line = np.linspace(0, 12, 100).reshape(-1, 1)
y_line = model.predict(X_line)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data', s=100)
plt.plot(X_line, y_line, color='red', linewidth=2, label='Best Fit Line')
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)
plt.title('Linear Regression: Salary vs Experience', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Display equation
coeffs = model.get_coefficients()
print(f"Equation: y = {coeffs['intercept']:.2f} + {coeffs['slope']:.2f}x")
```

---

## Key Concepts to Remember

### 1. **Best Fit Line**
The line that minimizes the sum of squared distances from all points to the line.

### 2. **Assumptions**

Linear regression rests on four assumptions. They are worth knowing not as
trivia but because each one, when violated, breaks something specific.

The single most useful diagnostic is a **residual plot** — plot
`y - model.predict(X)` on the vertical axis against the predictions on the
horizontal axis. If the assumptions hold, it should look like a
structureless horizontal band of points around zero:

```python
residuals = y_train - model.predict(X_train)
plt.scatter(model.predict(X_train), residuals)
plt.axhline(0, color='red')
plt.xlabel('Predicted'); plt.ylabel('Residual')
```

| Assumption | What it means | How to check | What breaks if violated |
|---|---|---|---|
| **Linearity** | y really is a straight-line function of x | Residual plot shows a curve (a U or arch) instead of a band | The model is biased everywhere; no amount of data fixes it. Add a polynomial term or use a non-linear model |
| **Normally distributed errors** | The noise around the line is Gaussian | Histogram or Q-Q plot of the residuals | The coefficients are still fine, but p-values and confidence intervals become untrustworthy |
| **Constant variance** (homoscedasticity) | The spread of the noise is the same everywhere | Residual plot fans out into a cone shape | Coefficients stay unbiased but are no longer the most precise estimates, and standard errors mislead. Often fixed by modelling log(y) |
| **Independence** | One observation's error tells you nothing about another's | Residuals plotted in time order show runs or cycles | Standard errors are badly understated, making a weak model look statistically significant. Common with time series |

Note the pattern: violating linearity breaks the *predictions*; violating
the other three mostly breaks your *confidence claims* about the predictions.

### 3. **When to Use**
- You have one input feature
- The relationship appears linear
- You want an interpretable model
- You need quick predictions

---

## Advantages & Limitations

### Advantages

- **Interpretable.** The slope is a plain-English statement: "one more year
  of experience is worth $5,000." Almost no other model gives you that.
- **No hyperparameters to tune.** There is no learning rate, no depth, no
  number of estimators — the Normal Equation has one exact answer.
- **Fast and exact.** Fitting is a single matrix solve, not an iterative
  search, so there is no convergence to babysit.
- **Needs little data.** With one feature you can fit a sensible line from a
  handful of points, where a tree ensemble or neural net would overfit.
- **A serious baseline.** If a complicated model cannot beat linear
  regression, that is important information about your problem.

### Limitations

- **Only works for linear relationships.** It cannot represent a curve; the
  residual plot will show an arch and the model stays biased.
- **Sensitive to outliers.** Because errors are *squared*, one bad point can
  drag the whole line. A point 10 units off contributes 100x the penalty of
  a point 1 unit off. Huber or RANSAC regression are robust alternatives.
- **Cannot capture complex patterns** such as interactions or thresholds
  unless you engineer those features yourself.
- **Correlation, not causation.** A fitted slope does not license a causal
  claim about intervening on x.
- **Extrapolates confidently but blindly.** Predicting salary at 40 years of
  experience from data spanning 1-10 years gives a number with no warning
  attached. (Unlike tree models, it *will* extrapolate — whether you should
  trust it is another matter.)
- **Collinear features break the exact inverse.** Handled here with the
  pseudo-inverse, but the underlying coefficients remain unstable and
  individually uninterpretable.
- **Use Multiple Regression for multiple features**, and Ridge/Lasso when
  features are correlated or numerous.

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load diabetes dataset (we'll use only BMI feature)
data = load_diabetes()
X, y = data.data[:, 2:3], data.target  # BMI column only

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
r2 = model.score(X_test, y_test)
print(f"R^2 Score: {r2:.4f}")

# Examine coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.2f}")
print(f"Slope: {coeffs['slope']:.2f}")

# Interpret
print(f"\nInterpretation:")
print(f"For every 1 unit increase in BMI, disease progression")
print(f"{'increases' if coeffs['slope'] > 0 else 'decreases'} by {abs(coeffs['slope']):.2f} units")
```

Expected output:
```
R^2 Score: 0.2334

Intercept: 152.00
Slope: 998.58

Interpretation:
For every 1 unit increase in BMI, disease progression
increases by 998.58 units
```

An R² of 0.23 is not a mistake — it is an honest result. BMI alone explains
about 23% of the variance in diabetes progression, which is real signal but
far from the whole story. This is what a *useful but incomplete* single
feature looks like, and it is the motivation for Multiple Linear Regression:
using all ten features on this same train/test split raises the test R² from
0.2334 to 0.4526.

---

## Conclusion

Simple Linear Regression is the foundation of machine learning! By understanding:
- How to fit a line through data
- What intercept and slope mean
- How to make predictions
- How to evaluate model quality

You've taken your first step into the world of machine learning! 🎯

**Next Steps**:
- Try with your own data
- Visualize your results
- Compare with scikit-learn's LinearRegression
- Learn about Multiple Linear Regression (when you have multiple features)
- Explore Ridge and Lasso regression (regularized versions)

Happy coding! 💻📈
