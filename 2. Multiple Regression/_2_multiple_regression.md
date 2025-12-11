# Multiple Linear Regression from Scratch: A Comprehensive Guide

Welcome to the world of Multiple Linear Regression! 📊 In this detailed guide, we'll explore how to predict outcomes using multiple input features. Think of it as upgrading from drawing a line in 2D to fitting a plane (or hyperplane) in multi-dimensional space!

## Table of Contents
1. [What is Multiple Linear Regression?](#what-is-multiple-linear-regression)
2. [Simple vs Multiple Regression](#simple-vs-multiple-regression)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)

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

### The Normal Equation

To find the best coefficients that minimize the error, we use the **Normal Equation**:

```
θ = (XᵀX)⁻¹Xᵀy
```

This formula gives us the optimal coefficients in one shot (closed-form solution)!

**Breaking it down**:
1. **Xᵀ** = transpose of X matrix
2. **XᵀX** = matrix multiplication
3. **(XᵀX)⁻¹** = inverse of the matrix
4. **Xᵀy** = transpose of X multiplied by y

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class MultipleRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
```

### Core Methods

1. **`fit(X, y)`** - Train the model
   - Adds bias term (column of ones)
   - Calculates coefficients using Normal Equation
   - Stores intercept and feature coefficients separately

2. **`predict(X)`** - Make predictions
   - Adds bias term to new data
   - Multiplies features by coefficients
   - Returns predicted values

3. **`get_coefficients()`** - Get model parameters
   - Returns intercept and all feature coefficients
   - Useful for interpreting the model

4. **`score(X, y)`** - Calculate R² score
   - Measures how well the model fits the data
   - Returns value between 0 and 1 (1 = perfect fit)

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
2. Computes (XᵀX)⁻¹
3. Multiplies by Xᵀy
4. Stores the resulting coefficients

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

**What do these mean?**
- **Intercept**: Base price when all features are 0
- **Square Feet Coefficient**: Price increase per square foot
- **Bedrooms Coefficient**: Price increase per bedroom
- **Age Coefficient**: Price change per year of age (likely negative)

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

```python
self.coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
```

**Step-by-step**:
1. `X_with_bias.T` → Transpose the matrix
2. `X_with_bias.T @ X_with_bias` → Matrix multiplication (XᵀX)
3. `np.linalg.inv(...)` → Find inverse (XᵀX)⁻¹
4. `@ X_with_bias.T @ y` → Multiply by Xᵀy
5. Result → Optimal coefficients!

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
- Features are independent
- Errors are normally distributed
- Constant variance of errors (homoscedasticity)

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
print(f"R² Score: {r2:.4f}")

# Examine coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.2f}")
print("\nFeature Coefficients:")
for i, coef in enumerate(coeffs['coefficients'], 1):
    print(f"  Feature {i}: {coef:.2f}")
```

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

