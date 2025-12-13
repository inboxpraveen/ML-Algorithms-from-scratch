# Logistic Regression from Scratch: A Comprehensive Guide

Welcome to the world of Logistic Regression! 🎯 In this comprehensive guide, we'll explore how to solve binary classification problems. Think of it as the go-to algorithm when you need to answer yes/no questions based on data!

## Table of Contents
1. [What is Logistic Regression?](#what-is-logistic-regression)
2. [Regression vs Classification](#regression-vs-classification)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)

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
1. Initialize coefficients randomly
2. For each iteration:
   a. Compute predictions: p = sigmoid(X @ θ)
   b. Compute error: error = p - y
   c. Compute gradients: gradients = (1/n) * X^T @ error
   d. Update coefficients: θ = θ - learning_rate * gradients
3. Repeat until convergence
```

**Key Parameters**:
- **Learning Rate (α)**: Step size for updates
  - Too large → Overshooting, unstable
  - Too small → Slow convergence
  - Typical values: 0.001 to 0.1

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
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None
        self.intercept = None
        self.losses = []  # Track training progress
```

### Core Methods

1. **`__init__(learning_rate, iterations)`** - Initialize model
   - Set hyperparameters for training
   - learning_rate: Controls step size
   - iterations: Number of training steps

2. **`_sigmoid(z)`** - Private helper method
   - Applies sigmoid activation function
   - Maps linear output to probabilities
   - Handles numerical stability

3. **`fit(X, y)`** - Train the model
   - Implements gradient descent optimization
   - Minimizes binary cross-entropy loss
   - Updates coefficients iteratively

4. **`predict_proba(X)`** - Get probabilities
   - Returns probabilities for class 1
   - Values between 0 and 1
   - Useful for understanding confidence

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
    [3, 60],    # 3 hours study, 60% attendance → Pass
    [4, 80],    # 4 hours study, 80% attendance → Pass
    [5, 100],   # 5 hours study, 100% attendance → Pass
    [1.5, 30],  # Low effort → Fail
    [2.5, 50],  # Medium effort → Pass
    [3.5, 70],  # High effort → Pass
    [4.5, 90]   # High effort → Pass
])

# Target: 0 = Fail, 1 = Pass
y_train = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])
```

### Training the Model

```python
model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)
```

**What happens internally**:
1. Coefficients initialized randomly: θ = [0.003, -0.001, 0.002]
2. For 1000 iterations:
   - Compute z = X @ θ
   - Apply sigmoid: p = 1/(1+e^(-z))
   - Compute loss and gradients
   - Update coefficients
3. Final coefficients learned: θ = [intercept, coef₁, coef₂]

### Making Predictions

```python
# New students
X_test = np.array([
    [2, 30],   # Low study, low attendance
    [4, 85],   # High study, high attendance
    [3, 55]    # Medium study, medium attendance
])

# Get probabilities
probabilities = model.predict_proba(X_test)
print("Probabilities of passing:", probabilities)
# Output: [0.15, 0.92, 0.58]

# Get class predictions
predictions = model.predict(X_test)
print("Predicted outcomes:", predictions)
# Output: [0, 1, 1]  (Fail, Pass, Pass)
```

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

**Example**:
```
Intercept: -5.2
Study Hours Coefficient: 0.8 (positive → more study = higher pass probability)
Attendance Coefficient: 0.04 (positive → more attendance = higher pass probability)
```

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
- Large negative z → e^(-z) becomes huge → overflow
- Large positive z → e^(-z) becomes tiny → precision issues
- Clipping to [-500, 500] prevents these problems

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

### 4. **No Closed-Form Solution**
Unlike linear regression, we must use iterative optimization (gradient descent)

### 5. **Assumptions**
Logistic regression assumes:
- Binary outcome (can be extended to multiclass)
- Linear decision boundary
- Features are independent
- Large sample size for reliable estimates

### 6. **Limitations**
- Only works for linearly separable data
- Assumes linear relationship between features and log-odds
- Sensitive to outliers
- May underperform with highly correlated features

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
- How gradient descent optimizes the model
- How to interpret probabilities and make decisions
- How to evaluate classification performance

You've gained a crucial tool in your machine learning toolkit! 🎯

**When to Use Logistic Regression**:
- ✅ Binary classification problems
- ✅ Need probability estimates
- ✅ Want interpretable model
- ✅ Linearly separable classes
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
- Learn about regularized logistic regression (L1/L2)
- Explore ROC curves and AUC scores
- Study multinomial logistic regression for multi-class

Happy coding! 💻🎯


