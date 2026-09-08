# Multiple Linear Regression from Scratch: A Comprehensive Guide

Welcome to the world of Multiple Linear Regression! 📊 In this detailed guide, we'll explore how to predict outcomes using multiple input features. Think of it as upgrading from drawing a line in 2D to fitting a plane (or hyperplane) in multi-dimensional space!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is Multiple Linear Regression?](#what-is-multiple-linear-regression)
3. [Simple vs Multiple Regression](#simple-vs-multiple-regression)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Key Concepts to Remember](#key-concepts-to-remember)
10. [Complete Usage Example](#complete-usage-example)
11. [Advantages & Limitations](#advantages--limitations)
12. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is exactly the `if __name__ == "__main__":` block at the bottom of
`_2_multiple_regression.py`, so you can simply run `python _2_multiple_regression.py`.
Nothing beyond NumPy is required.

```python
# ---------------------------------------------------------------
# Multiple Linear Regression from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _2_multiple_regression.py  (the __main__ block runs this)
# Or copy the MultipleRegression class from _2_multiple_regression.py below.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the MultipleRegression class here (from _2_multiple_regression.py) ----
# class MultipleRegression: ...

np.random.seed(42)

# --- Demo 1: can the Normal Equation recover coefficients we planted? ---
print("=" * 55)
print("DEMO 1 - Recovering known coefficients from noisy data")
print("=" * 55)

n_samples, n_features = 500, 4
true_intercept = 5.0
true_coefs = np.array([3.0, -2.0, 0.5, 0.0])  # feature 4 is pure noise

X = np.random.randn(n_samples, n_features)
y = true_intercept + X @ true_coefs + np.random.randn(n_samples) * 0.5

# Shuffle BEFORE slicing so train and test come from the same distribution
idx = np.random.permutation(n_samples)
X, y = X[idx], y[idx]
X_train, X_test = X[:400], X[400:]   # disjoint: 400 train, 100 test
y_train, y_test = y[:400], y[400:]

model = MultipleRegression()
model.fit(X_train, y_train)
coeffs = model.get_coefficients()

print("Parameter       True    Recovered")
print(f"  intercept  {true_intercept:8.3f} {coeffs['intercept']:12.3f}")
for i in range(n_features):
    print(f"  b{i + 1}         {true_coefs[i]:8.3f} "
          f"{coeffs['coefficients'][i]:12.3f}")
print("(b4 is planted at 0.0 - the model correctly finds no effect)")

print(f"\nTrain R2 : {model.score(X_train, y_train):.4f}")
print(f"Test  R2 : {model.score(X_test, y_test):.4f}")
print("\nSample predictions (true -> predicted):")
preds = model.predict(X_test)
for i in range(5):
    print(f"  true={y_test[i]:7.3f}  ->  pred={preds[i]:7.3f}")

# --- Demo 2: the house-price story with enough rows to be stable ---
print("\n" + "=" * 55)
print("DEMO 2 - House prices from square feet, bedrooms, age")
print("=" * 55)

n_houses = 200
sqft = np.random.uniform(800, 4000, n_houses)
bedrooms = np.random.randint(2, 6, n_houses)
age = np.random.uniform(0, 50, n_houses)
X_house = np.column_stack((sqft, bedrooms, age))

# True pricing rule, plus noise of about +/- $15k
y_house = (50000 + 150 * sqft + 10000 * bedrooms - 800 * age
           + np.random.randn(n_houses) * 15000)

idx = np.random.permutation(n_houses)
X_house, y_house = X_house[idx], y_house[idx]
X_tr, X_te = X_house[:160], X_house[160:]   # disjoint: 160 train, 40 test
y_tr, y_te = y_house[:160], y_house[160:]

house_model = MultipleRegression()
house_model.fit(X_tr, y_tr)
hc = house_model.get_coefficients()

print(f"  Base price (intercept) : {hc['intercept']:12,.0f}   (true 50,000)")
print(f"  Per square foot        : {hc['coefficients'][0]:12,.2f}   (true 150.00)")
print(f"  Per bedroom            : {hc['coefficients'][1]:12,.0f}   (true 10,000)")
print(f"  Per year of age        : {hc['coefficients'][2]:12,.0f}   (true -800)")
print("The age coefficient is negative -> older houses are worth less,")
print("which the 5-row example in USAGE EXAMPLE 1 is far too small to show.")

print(f"\nTrain R2 : {house_model.score(X_tr, y_tr):.4f}")
print(f"Test  R2 : {house_model.score(X_te, y_te):.4f}")
print("\nSample predictions (actual vs predicted price):")
house_preds = house_model.predict(X_te)
for i in range(3):
    print(f"  {X_te[i, 0]:6.0f} sqft, {int(X_te[i, 1])} bed, {X_te[i, 2]:4.1f} yrs"
          f"  ->  actual {y_te[i]:10,.0f}   predicted {house_preds[i]:10,.0f}")

# --- Demo 3: what happens when two features are identical? ---
print("\n" + "=" * 55)
print("DEMO 3 - Perfectly collinear features (the singular case)")
print("=" * 55)

# Column 5 is an exact copy of column 2 from Demo 1
X_dup = np.hstack((X_train, X_train[:, [1]]))
dup_model = MultipleRegression()
dup_model.fit(X_dup, y_train)
b = dup_model.feature_coefficients

print("Feature 5 is an exact copy of feature 2, so X^T X is singular and")
print("infinitely many coefficient vectors give identical predictions.")
print("The pseudo-inverse picks the minimum-norm one, which splits the")
print("shared effect evenly between the two copies:")
print(f"  b2 = {b[1]:7.3f}    b5 = {b[4]:7.3f}    b2 + b5 = {b[1] + b[4]:7.3f}"
      f"   (true -2.000)")
print(f"  Train R2 : {dup_model.score(X_dup, y_train):.4f}   "
      f"(identical to the {model.score(X_train, y_train):.4f} above)")

# What the textbook (X^T X)^-1 X^T y would have produced on this same data
X_bias = np.hstack((np.ones((len(X_dup), 1)), X_dup))
pinv_sse = np.sum((y_train - dup_model.predict(X_dup)) ** 2)
try:
    naive = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y_train
    naive_sse = np.sum((y_train - X_bias @ naive) ** 2)
    print("\nThe explicit inverse does not even fail loudly here - it returns")
    print("a vector that is simply not the least-squares solution:")
    print(f"  inv(X^T X) X^T y  ->  SSE {naive_sse:10.3f}")
except np.linalg.LinAlgError as err:
    # Whether inv raises or silently returns garbage depends on the LAPACK
    # build; both outcomes are failures, and both are why fit() avoids it.
    print("\nThe explicit inverse fails outright here:")
    print(f"  inv(X^T X) X^T y  ->  LinAlgError: {err}")
print(f"  pinv(X) y         ->  SSE {pinv_sse:10.3f}   (the true minimum)")
print("\nSame fit, unstable coefficients: that is multicollinearity, and")
print("that silent wrongness is why fit() uses the pseudo-inverse.")
```

Expected output:
```
=======================================================
DEMO 1 - Recovering known coefficients from noisy data
=======================================================
Parameter       True    Recovered
  intercept     5.000        5.006
  b1            3.000        2.999
  b2           -2.000       -2.031
  b3            0.500        0.469
  b4            0.000        0.033
(b4 is planted at 0.0 - the model correctly finds no effect)

Train R2 : 0.9821
Test  R2 : 0.9805

Sample predictions (true -> predicted):
  true=  5.463  ->  pred=  6.103
  true=  4.019  ->  pred=  4.803
  true=  3.136  ->  pred=  3.738
  true=  6.546  ->  pred=  6.640
  true=  3.138  ->  pred=  3.105

=======================================================
DEMO 2 - House prices from square feet, bedrooms, age
=======================================================
  Base price (intercept) :       60,043   (true 50,000)
  Per square foot        :       146.28   (true 150.00)
  Per bedroom            :       10,236   (true 10,000)
  Per year of age        :         -857   (true -800)
The age coefficient is negative -> older houses are worth less,
which the 5-row example in USAGE EXAMPLE 1 is far too small to show.

Train R2 : 0.9867
Test  R2 : 0.9879

Sample predictions (actual vs predicted price):
    3772 sqft, 2 bed, 41.2 yrs  ->  actual    590,163   predicted    596,876
    2582 sqft, 4 bed, 36.4 yrs  ->  actual    452,429   predicted    447,409
    3567 sqft, 3 bed,  8.1 yrs  ->  actual    611,733   predicted    605,594

=======================================================
DEMO 3 - Perfectly collinear features (the singular case)
=======================================================
Feature 5 is an exact copy of feature 2, so X^T X is singular and
infinitely many coefficient vectors give identical predictions.
The pseudo-inverse picks the minimum-norm one, which splits the
shared effect evenly between the two copies:
  b2 =  -1.016    b5 =  -1.016    b2 + b5 =  -2.031   (true -2.000)
  Train R2 : 0.9821   (identical to the 0.9821 above)

The explicit inverse does not even fail loudly here - it returns
a vector that is simply not the least-squares solution:
  inv(X^T X) X^T y  ->  SSE    159.948
  pinv(X) y         ->  SSE     89.206   (the true minimum)

Same fit, unstable coefficients: that is multicollinearity, and
that silent wrongness is why fit() uses the pseudo-inverse.
```

---

## What is Multiple Linear Regression?

Multiple Linear Regression is an extension of simple linear regression that allows us to predict a target variable using **multiple features** (independent variables) instead of just one.

**Real-world analogy**: 
- **Simple Linear Regression**: Predicting house price based only on square footage
- **Multiple Linear Regression**: Predicting house price based on square footage, number of bedrooms, number of bathrooms, location, and age

### The Mathematical Equation

The general formula for multiple linear regression is:

```
y = b₀ + b₁x₁ + b₂x₂ + b₃x₃ + ... + bₙxₙ
```

Where:
- **y** = target variable (what we want to predict)
- **b₀** = intercept (bias term)
- **b₁, b₂, ..., bₙ** = coefficients for each feature
- **x₁, x₂, ..., xₙ** = input features (independent variables)

---

## Simple vs Multiple Regression

| Aspect | Simple Linear Regression | Multiple Linear Regression |
|--------|-------------------------|---------------------------|
| **Number of Features** | 1 feature | 2 or more features |
| **Equation** | y = b₀ + b₁x | y = b₀ + b₁x₁ + b₂x₂ + ... |
| **Visualization** | 2D line | 3D plane or higher-dimensional hyperplane |
| **Example** | Price vs Size | Price vs Size, Bedrooms, Location |
| **Complexity** | Simpler to visualize | More complex but more accurate |

---

## The Mathematical Foundation

### Matrix Representation

Multiple regression can be elegantly expressed using matrices:

```
Y = Xθ
```

Where:
- **Y** is an (n×1) vector of target values
- **X** is an (n×m) matrix of features (n samples, m features)
- **θ** is an (m×1) vector of coefficients

Notice that `Y = Xθ` has no intercept in it, while the scalar equation above has b₀.
The two are reconciled by **absorbing the intercept into θ**: we prepend a column of
ones to X, so X becomes (n×(m+1)) and θ becomes ((m+1)×1) with θ₀ = b₀. A "1" times
b₀ is just b₀, so the intercept rides along as one more coefficient. That single trick
is why the implementation begins with `np.hstack((np.ones(...), X))` and why
`model.coefficients` is one element longer than the number of features. From here on,
**X means the design matrix with the ones column already attached**.

### The Normal Equation

To find the best coefficients that minimize the error, we use the **Normal Equation**:

```
θ = (XᵀX)⁻¹Xᵀy
```

This formula gives us the optimal coefficients in one shot (closed-form solution)!

#### Where does it come from?

It is not magic - it falls out of one line of calculus. We want the θ that makes the
predictions Xθ as close to y as possible, where "close" means the smallest **sum of
squared errors**. So the cost function we are minimizing is:

```
J(θ) = (y - Xθ)ᵀ(y - Xθ)
```

This is a quadratic bowl in θ: it curves upward in every direction, so it has exactly
one minimum, and that minimum sits wherever the slope is zero. Expand it, then
differentiate with respect to the whole vector θ:

```
J(θ)  = yᵀy - 2θᵀXᵀy + θᵀXᵀXθ
dJ/dθ = -2Xᵀy + 2XᵀXθ
```

Set that gradient to zero - the first-order condition for a minimum:

```
-2Xᵀy + 2XᵀXθ = 0
         XᵀXθ = Xᵀy        <- this is the Normal Equation
```

and solve for θ by left-multiplying with (XᵀX)⁻¹:

```
θ = (XᵀX)⁻¹Xᵀy
```

There is a nice geometric reading of `XᵀXθ = Xᵀy`. Rearranged it says
`Xᵀ(y - Xθ) = 0`: the residual vector is **orthogonal to every feature column**. No
feature has any leftover linear correlation with the errors, so there is nothing left
to squeeze out - which is exactly what "best fit" ought to mean.

**Breaking it down**:
1. **Xᵀ** = transpose of X matrix
2. **XᵀX** = matrix multiplication
3. **(XᵀX)⁻¹** = inverse of the matrix
4. **Xᵀy** = transpose of X multiplied by y

#### When does that inverse exist?

This is the one precondition of the whole algorithm, and it is worth knowing before
you meet it as an error message. `(XᵀX)⁻¹` exists only when X has **full column
rank** - no feature column can be written as a combination of the others. That needs:

- **n ≥ m + 1**: at least as many samples as parameters. With fewer, infinitely many
  hyperplanes pass through every training point exactly, and none is "the" answer.
- **No perfectly collinear features**: not a duplicated column, not
  `total = part_a + part_b`, and not the classic *dummy variable trap* (one-hot
  encoding all k levels of a category while an intercept is also present).

When that condition fails, `np.linalg.inv` does one of two things, and the second is
far more dangerous than the first:

1. it raises `LinAlgError: Singular matrix`, or
2. it silently returns a θ that is **not** the least-squares solution at all.

This is why `fit()` evaluates the Normal Equation with the **pseudo-inverse** rather
than an explicit inverse - see [Understanding the Code](#understanding-the-code).
Demo 3 in the [Quick Start](#quick-start-plug-and-play-example) measures the gap on a
duplicated column: the explicit inverse lands at SSE 159.948 while the pseudo-inverse
finds the true minimum, 89.206.

### The Iterative Alternative: Gradient Descent

The Normal Equation is exact and needs no tuning, but solving it means factorizing an
(m+1)-column matrix - an inverse, or the SVD this code uses - which costs roughly
O(n·m² + m³) time and O(m²) memory. Gradient descent instead walks downhill on the
same J(θ):

```
θ := θ - α · (2/n) · Xᵀ(Xθ - y)
```

repeated until it converges, at O(n·m) per step. It needs a learning rate α and a
stopping rule, and it only approaches the answer - but it stays practical when m is
in the thousands, and it is the same machinery that trains logistic regression and
neural networks. Rule of thumb: **closed form up to a few thousand features,
gradient descent beyond that.** This implementation uses the closed form.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class MultipleRegression:
    def __init__(self):
        self.coefficients = None          # [b0, b1, ..., bn]  (n_features + 1 long)
        self.intercept = None             # b0
        self.feature_coefficients = None  # [b1, ..., bn]      (n_features long)
```

There is no hyperparameter to set. Ordinary least squares has one exact solution,
so there is no learning rate, no iteration count and no regularization strength -
which is a large part of why it makes such a good first algorithm.

> **Two things named "coefficients".** `self.coefficients` is the full parameter
> vector **including** the intercept in slot 0, so it is one element longer than the
> feature count. `self.feature_coefficients` is `self.coefficients[1:]` - the slopes
> **without** the intercept, and it is what `get_coefficients()` returns under the
> `'coefficients'` key. Keep the distinction straight and the shapes always line up.

### Core Methods

1. **`fit(X, y)`** - Train the model
   - Adds bias term (column of ones)
   - Calculates coefficients using Normal Equation
   - Stores intercept and feature coefficients separately, in
     `self.intercept` and `self.feature_coefficients`
   - Accepts Python lists, 1-D `X`, and column-vector `y` of shape (n, 1);
     `y` is flattened so `predict()` always returns shape (n_samples,)
   - Returns `self`, so `model.fit(X, y).predict(X_new)` works

2. **`predict(X)`** - Make predictions
   - Adds bias term to new data
   - Multiplies features by coefficients
   - Returns predicted values
   - Raises `ValueError("Model is not fitted. Call fit(X, y) first.")` if called
     before `fit()`, and also checks that `X` has the expected number of features

3. **`get_coefficients()`** - Get model parameters
   - Returns `{'intercept': b0, 'coefficients': [b1, ..., bn]}`
   - Useful for interpreting the model

4. **`score(X, y)`** - Calculate R² score
   - Measures how well the model fits the data
   - Returns 1.0 for a perfect fit and 0.0 for a model no better than predicting the
     mean of y; the score **can be negative** on unseen data (see the interpretation
     scale in [Understanding the Code](#understanding-the-code))
   - If y is constant, R² is undefined (0/0); following scikit-learn's convention this
     returns 1.0 when the predictions match and 0.0 otherwise

---

## Step-by-Step Example

Let's walk through a complete example predicting **house prices** based on three features:

### The Data

```python
import numpy as np

# Features: [square_feet, bedrooms, age_of_house]
X_train = np.array([
    [1500, 3, 10],  # House 1
    [2000, 4, 5],   # House 2
    [1200, 2, 15],  # House 3
    [1800, 3, 8],   # House 4
    [2500, 5, 2]    # House 5
])

# Target: house prices in dollars
y_train = np.array([300000, 400000, 250000, 350000, 500000])
```

### Training the Model

```python
model = MultipleRegression()
model.fit(X_train, y_train)
```

**What happens internally**:
1. Adds a column of ones to X_train → becomes [1, 1500, 3, 10], [1, 2000, 4, 5], ...
2. Solves XᵀXθ = Xᵀy for θ, evaluating the Normal Equation as `pinv(X) @ y`
3. Splits the result into the intercept θ₀ and the feature coefficients θ₁..θ₃
4. Stores them in `self.coefficients`, `self.intercept`, `self.feature_coefficients`

### Making Predictions

```python
# New houses to predict
X_test = np.array([
    [1600, 3, 7],   # 1600 sq ft, 3 bedrooms, 7 years old
    [2200, 4, 3]    # 2200 sq ft, 4 bedrooms, 3 years old
])

predictions = model.predict(X_test)
print("Predicted prices:", predictions)
```

### Interpreting Coefficients

```python
coeffs = model.get_coefficients()
print(f"Intercept: {coeffs['intercept']}")
print(f"Square Feet Coefficient: {coeffs['coefficients'][0]}")
print(f"Bedrooms Coefficient: {coeffs['coefficients'][1]}")
print(f"Age Coefficient: {coeffs['coefficients'][2]}")
```

Output:
```
Intercept: -112745.0980392173
Square Feet Coefficient: 196.07843137254906
Bedrooms Coefficient: 22549.019607843366
Age Coefficient: 5392.156862745163
```

**What do these mean?**
- **Intercept**: Base price when all features are 0
- **Square Feet Coefficient**: Price increase per square foot
- **Bedrooms Coefficient**: Price increase per bedroom
- **Age Coefficient**: Price change per year of age

### Why is the age coefficient *positive*?

You would expect an older house to be worth **less**, yet this fit says every extra
year adds $5,392. The model is not broken - scikit-learn's `LinearRegression` returns
the same numbers on this data to nine decimal places. The dataset is.

There are 5 houses and 4 parameters to estimate (intercept + 3 features), which leaves
a single residual degree of freedom. On top of that, the three features move almost in
lockstep: bigger houses in this table also happen to be newer, so square footage and
age are close to collinear (the condition number of XᵀX is 3.3 × 10⁹). When two
columns carry nearly the same information, OLS can shuffle a large effect between them
in any proportion and still fit the data - so the individual coefficients become
arbitrary even while the predictions stay sensible. That is precisely the
[multicollinearity](#2-multicollinearity) problem, met in the wild.

The cure is more data. Demo 2 of the [Quick Start](#quick-start-plug-and-play-example)
runs the identical model on 200 houses generated from a known rule, and the age
coefficient comes out at **-857 per year** against a planted truth of -800 - the sign
you expected all along.

**Takeaway**: a high R² (this fit scores 0.9993 on its own training rows) tells you
nothing about whether the individual coefficients are trustworthy.

---

## Real-World Applications

### 1. **Real Estate Pricing**
Predict house prices based on:
- Square footage
- Number of bedrooms/bathrooms
- Location (zip code)
- Age of property
- School district rating

### 2. **Sales Forecasting**
Predict product sales based on:
- Advertising spend (TV, radio, online)
- Season
- Competitor pricing
- Economic indicators

### 3. **Medical Predictions**
Predict disease progression based on:
- Age
- BMI
- Blood pressure
- Blood sugar level
- Family history

### 4. **Student Performance**
Predict test scores based on:
- Study hours
- Attendance
- Previous grades
- Socioeconomic factors

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Adding the Bias Term

```python
X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
```

**Why?** The bias (intercept) represents the base value when all features are zero. By adding a column of ones, we can include it in our matrix multiplication.

**Example transformation**:
```
Before: [[1500, 3, 10],      After: [[1, 1500, 3, 10],
         [2000, 4, 5]]                [1, 2000, 4, 5]]
```

### 2. Normal Equation Implementation

The formula we derived is θ = (XᵀX)⁻¹Xᵀy. Written out literally in NumPy that is:

```python
# The textbook transcription - correct, but fragile. NOT what the code does.
self.coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
```

**Step-by-step**:
1. `X_with_bias.T` → Transpose the matrix
2. `X_with_bias.T @ X_with_bias` → Matrix multiplication (XᵀX)
3. `np.linalg.inv(...)` → Find inverse (XᵀX)⁻¹
4. `@ X_with_bias.T @ y` → Multiply by Xᵀy
5. Result → Optimal coefficients!

The implementation computes **the same quantity a safer way**:

```python
self.coefficients = np.linalg.pinv(X_with_bias) @ y
```

`np.linalg.pinv` is the **Moore-Penrose pseudo-inverse**, written X⁺. For any matrix
with full column rank there is an identity:

```
X⁺ = (XᵀX)⁻¹Xᵀ
```

so `pinv(X) @ y` *is* θ = (XᵀX)⁻¹Xᵀy - the very formula in the section above, just
evaluated through an SVD instead of by building and inverting XᵀX. Two things improve:

- **Accuracy.** Forming XᵀX squares the condition number of X, so a design that is
  merely awkward becomes numerically hostile. The SVD route never forms XᵀX at all.
- **Robustness.** When X is rank deficient (a duplicated column, a dummy-variable
  trap, or more features than samples) there is no inverse to find. The explicit
  inverse then either raises `LinAlgError` or - much worse - quietly returns a vector
  that does not minimize the squared error. `pinv` always returns a genuine
  least-squares solution, and among the infinitely many that exist it picks the one
  with the smallest ‖θ‖ (the *minimum-norm* solution).

Verified against `sklearn.linear_model.LinearRegression`, which solves the same
problem by the same family of method: on a well-conditioned 200×5 design the
coefficients agree to 3×10⁻¹⁵, on the 5-house table above to 2×10⁻¹⁰, and on the
diabetes dataset the test R² is 0.4526027630 from both. On a rank-deficient design
the explicit-inverse version diverges completely while `pinv` still matches sklearn.

> One trap worth naming: `np.linalg.pinv(XᵀX) @ Xᵀ @ y` looks like a smaller edit and
> is **not** equivalent. It still forms XᵀX, so the condition number is still squared,
> and `pinv`'s cutoff then discards real signal along with the noise. Take the
> pseudo-inverse of the **design matrix**, not of XᵀX.

### 3. Making Predictions

```python
return X_with_bias @ self.coefficients
```

**What it does**: Multiplies each sample's features by the learned coefficients and sums them up.

**Example calculation**:
```
For house [1600, 3, 7]:
price = b₀×1 + b₁×1600 + b₂×3 + b₃×7
```

### 4. R² Score (Model Evaluation)

```python
ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
r2_score = 1 - (ss_res / ss_tot)
```

**Interpretation**:
- **R² = 1.0** → Perfect predictions
- **R² = 0.8** → Model explains 80% of variance (very good)
- **R² = 0.5** → Model explains 50% of variance (moderate)
- **R² = 0.0** → Model no better than predicting the mean
- **R² < 0.0** → Model worse than predicting the mean

There is no lower bound: on a test set drawn from a different process the score can be
-7 or -70. Note also that R² on the *training* set never falls when you add a feature,
even a column of pure noise - which is why it is a poor tool for choosing how many
features to keep (see [Advantages & Limitations](#advantages--limitations)).

**One edge case the code guards**: if `y` is constant, `ss_tot` is 0 and R² is a 0/0.
The implementation follows scikit-learn's convention and returns 1.0 when the
predictions match, 0.0 otherwise, rather than emitting `nan`:

```python
if ss_tot == 0:
    return 1.0 if ss_res <= 1e-12 * max(np.sum(y ** 2), 1.0) else 0.0
```

---

## Key Concepts to Remember

### 1. **Feature Scaling**
When features have different scales (e.g., square feet: 1000-5000, bedrooms: 1-5), consider normalizing them for better performance.

### 2. **Multicollinearity**
When features are highly correlated with each other, it can cause problems. For example, "square feet" and "number of rooms" might be highly correlated.

### 3. **Overfitting**
With too many features relative to samples, the model might fit the training data perfectly but fail on new data.

### 4. **Assumptions**
Multiple regression assumes:
- Linear relationship between features and target
- No *perfect* multicollinearity among features (XᵀX must be invertible). Note that
  correlated features are perfectly allowed - only exact linear dependence is fatal,
  which is why section 2 above is a caution and not a prohibition
- Errors are independent of one another (no autocorrelation - a real concern with
  time-series data, where today's residual predicts tomorrow's)
- Errors are normally distributed
- Constant variance of errors (homoscedasticity)

Only the first two matter for the coefficients themselves: OLS gives the best linear
unbiased estimates without any normality assumption. Normality and homoscedasticity
are what confidence intervals and p-values rest on.

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load diabetes dataset (10 features)
data = load_diabetes()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = MultipleRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
r2 = model.score(X_test, y_test)
print(f"R2 Score: {r2:.4f}")

# Examine coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.2f}")
print("\nFeature Coefficients:")
for i, coef in enumerate(coeffs['coefficients'], 1):
    print(f"  Feature {i}: {coef:.2f}")
```

Output:
```
R2 Score: 0.4526

Intercept: 151.35

Feature Coefficients:
  Feature 1: 37.90
  Feature 2: -241.96
  Feature 3: 542.43
  Feature 4: 347.70
  Feature 5: -931.49
  Feature 6: 518.06
  Feature 7: 163.42
  Feature 8: 275.32
  Feature 9: 736.20
  Feature 10: 48.67
```

`sklearn.linear_model.LinearRegression` scores **0.4526027630** on this same split -
identical to ten decimal places, with coefficients agreeing to 9×10⁻¹³. An R² of 0.45
is not a weak implementation; it is what a linear model can honestly extract from this
dataset.

---

## Advantages & Limitations

### Advantages

- **Closed form.** One SVD and you are done. No learning rate, no epochs, no
  convergence check, no random restarts - and the answer is exact, not approximate.
- **No hyperparameters.** Nothing to tune means nothing to tune *wrongly*, and no
  validation budget spent on a search.
- **Fully interpretable.** Every coefficient is "holding the other features fixed, one
  more unit of this feature moves the prediction by *this much*". Very few models can
  be read that directly, which is why regression still dominates in economics,
  epidemiology and any setting where a decision must be defended.
- **Fast on typical tabular data.** Fitting 20,000 samples × 100 features takes about
  0.05 s with this implementation (0.03 s for scikit-learn).
- **A hard baseline to beat.** If a gradient-boosted forest cannot clear a linear
  model by a worthwhile margin, the extra complexity is not paying for itself.

### Limitations

- **Requires full column rank.** Duplicated features, a dummy-variable trap, or more
  features than samples all break the inverse. The pseudo-inverse keeps the code
  running, but a minimum-norm solution among infinitely many is a warning, not a fix -
  drop the redundant column or switch to Ridge.
- **O(m³) in the number of features.** The closed form stops being attractive
  somewhere in the thousands of features; that is gradient descent's territory.
- **Sensitive to outliers.** Squaring the error means one point at ten times the
  typical residual contributes a hundred times the pull. Huber loss or RANSAC exist
  for a reason.
- **Unstable under multicollinearity.** Correlated features leave the *predictions*
  fine but make the individual coefficients swing wildly - exactly the sign flip in
  the [Step-by-Step Example](#why-is-the-age-coefficient-positive).
- **Linear only.** Curvature and interactions must be engineered in by hand
  (`x²`, `x₁·x₂`, `log x`). The model cannot discover them.
- **Sensitive to feature scale for interpretation.** The fit itself is scale
  invariant, but comparing a coefficient in dollars-per-square-foot against one in
  dollars-per-bedroom is meaningless until the features are standardized.
- **Training R² always rises when you add features**, even useless ones - so it cannot
  be used to decide how many to keep. Adjusted R², a held-out set, or cross-validation
  can.

### Simplifications vs. canonical OLS

This implementation is deliberately the *estimation* half of ordinary least squares.
A full statistical package (statsmodels' `OLS`, R's `lm`) also reports the
**inferential** half, which is not implemented here:

| Not implemented | What canonical OLS provides | Why it is omitted |
|---|---|---|
| Standard errors | `se(θ) = sqrt(diag(σ̂²(XᵀX)⁻¹))`, with `σ̂² = SSE / (n - m - 1)` | Genuinely cheap here - `(XᵀX)⁻¹` is `X⁺(X⁺)ᵀ`, already available from the pinv. Left out only to keep the class to the estimation half |
| t-statistics & p-values | `t = θ / se(θ)`, compared against a t-distribution with n-m-1 degrees of freedom | Requires a t-distribution CDF, which numpy alone does not provide (scipy does) |
| Confidence intervals | `θ ± t* · se(θ)` | Same dependency as above |
| Adjusted R² | `1 - (1-R²)(n-1)/(n-m-1)` | One line of arithmetic; it belongs with the rest of the inferential output rather than alone |
| Regularization | Ridge adds `λI` to XᵀX; Lasso adds an L1 penalty | A different algorithm - Ridge and Lasso are their own topic |

**Practical consequence:** you can read *what* this model learned, but not *how sure*
it is. A coefficient of 5,392 and a coefficient of 5,392 ± 40,000 look identical
through `get_coefficients()`. When that distinction matters - and in any scientific or
regulatory setting it does - fit with statsmodels and read the summary table.

---

## Conclusion

Multiple Linear Regression is a powerful and interpretable technique for prediction tasks. By understanding how multiple features contribute to the target variable, we can:
- Make accurate predictions
- Understand feature importance
- Identify relationships in data
- Make data-driven decisions

The beauty of implementing it from scratch is that you now understand exactly what's happening under the hood! 🎯

**Next Steps**:
- Try with your own data
- Experiment with different features
- Compare with scikit-learn's LinearRegression
- Learn about Ridge and Lasso regression (regularized versions)

Happy coding! 💻📈

