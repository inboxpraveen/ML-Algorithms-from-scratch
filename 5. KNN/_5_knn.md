# K-Nearest Neighbors (KNN) from Scratch: A Comprehensive Guide

Welcome to the world of K-Nearest Neighbors! 🎯 In this comprehensive guide, we'll explore one of the simplest yet most powerful machine learning algorithms. Think of it as the "birds of a feather flock together" algorithm!

## Table of Contents
1. [What is K-Nearest Neighbors?](#what-is-k-nearest-neighbors)
2. [How KNN Works](#how-knn-works)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)

---

## What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a **simple, intuitive, non-parametric algorithm** used for both classification and regression. It makes predictions based on the principle that similar things are near each other.

**Real-world analogy**: 
Imagine you move to a new neighborhood and want to know if it's safe. You ask your 5 nearest neighbors about crime rates. If 4 out of 5 say it's safe, you'd conclude it's probably a safe neighborhood. That's exactly how KNN works!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Instance-based, Non-parametric |
| **Learning Style** | Lazy learning (no training phase) |
| **Tasks** | Classification and Regression |
| **Decision** | Based on k nearest neighbors |
| **Key Parameter** | k (number of neighbors) |

### The Core Idea

```
"You are the average of the k closest examples to you"
```

KNN doesn't learn a model! Instead, it:
1. **Memorizes** all training data
2. **Finds** the k most similar examples when predicting
3. **Votes** (classification) or **averages** (regression) their labels

---

## How KNN Works

### The Algorithm in 5 Steps

```
Step 1: Store all training data (X_train, y_train)
         ↓
Step 2: For new point x, calculate distance to ALL training points
         ↓
Step 3: Find the k nearest neighbors (smallest distances)
         ↓
Step 4: Classification: Vote for most common class
        Regression: Average the values
         ↓
Step 5: Return prediction
```

### Visual Example

```
Training Data (2D):
    
    Class A: ●  ●  ●
    Class B: ■  ■  ■
    New Point: ?

Step 1: Calculate distances
    
    ●(2.1)  ●(3.5)  ●(5.2)
         ?
    ■(1.8)  ■(4.1)  ■(6.3)

Step 2: Sort by distance, pick k=3 nearest
    
    Nearest: ■(1.8), ●(2.1), ●(3.5)
    
Step 3: Vote (2 Class A, 1 Class B)
    
    Prediction: Class A ●
```

### Why "K-Nearest"?

The "K" in KNN is crucial:

```
k=1: Look at 1 nearest neighbor
     → Very flexible, sensitive to noise
     → High variance, low bias

k=3: Look at 3 nearest neighbors
     → More stable, some noise tolerance
     → Balanced

k=10: Look at 10 nearest neighbors
      → Very stable, robust to noise
      → Low variance, high bias
```

**Visual Comparison**:
```
k=1:  Decision boundary is wiggly, complex
      ●●●●●●●●●●
       ■■■■■■■■■■
      
k=5:  Decision boundary is smoother
      ●●●●●●●●●●
      ----------
      ■■■■■■■■■■
      
k=20: Decision boundary is very smooth
      ●●●●●●●●●●
      ==========
      ■■■■■■■■■■
```

---

## The Mathematical Foundation

### Distance Metrics

KNN relies on measuring "distance" between points. The most common metrics are:

#### 1. Euclidean Distance (L2)

The straight-line distance between two points:

```
d(x, y) = √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]
```

**Example**:
```python
Point A: [1, 2]
Point B: [4, 6]

d = √[(1-4)² + (2-6)²]
d = √[(-3)² + (-4)²]
d = √[9 + 16]
d = √25 = 5
```

**Visualization**:
```
    y
    6 |      B
    5 |     /
    4 |    /  d=5
    3 |   /
    2 | A
    1 |
    0 +----------- x
      0 1 2 3 4
```

**When to use**: Most cases, natural measure of distance

#### 2. Manhattan Distance (L1)

The city-block distance (sum of absolute differences):

```
d(x, y) = |x₁-y₁| + |x₂-y₂| + ... + |xₙ-yₙ|
```

**Example**:
```python
Point A: [1, 2]
Point B: [4, 6]

d = |1-4| + |2-6|
d = 3 + 4
d = 7
```

**Visualization**:
```
    y
    6 |      B
    5 |      ↑
    4 |      ↑  4 blocks up
    3 |      ↑
    2 | A→→→→↑  3 blocks right
    1 |
    0 +----------- x
      0 1 2 3 4
```

**When to use**: High-dimensional data, when features are independent

### Classification Decision Rule

For a new point x, find k nearest neighbors and predict:

```
ŷ = mode(y₁, y₂, ..., yₖ)
```

Where:
- ŷ = predicted class
- y₁, y₂, ..., yₖ = labels of k nearest neighbors
- mode = most frequent value

**Example**:
```
k = 5
Neighbor labels: [A, A, B, A, C]

Count: A=3, B=1, C=1
Prediction: A (majority vote)
```

### Regression Decision Rule

For regression, predict the average:

```
ŷ = (1/k) × Σ(y₁ + y₂ + ... + yₖ)
```

**Example**:
```
k = 3
Neighbor values: [100, 150, 125]

Prediction: (100 + 150 + 125) / 3 = 125
```

### Probability Estimation

KNN can also provide probability estimates:

```
P(class=c|x) = (number of neighbors with class c) / k
```

**Example**:
```
k = 5
Neighbor labels: [A, A, B, A, C]

P(A) = 3/5 = 0.60 (60%)
P(B) = 1/5 = 0.20 (20%)
P(C) = 1/5 = 0.20 (20%)
```

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class KNearestNeighbors:
    def __init__(self, k=5, distance_metric='euclidean', task='classification'):
        self.k = k
        self.distance_metric = distance_metric
        self.task = task
        self.X_train = None
        self.y_train = None
```

### Core Methods

1. **`__init__(k, distance_metric, task)`** - Initialize model
   - k: Number of neighbors to consider
   - distance_metric: 'euclidean' or 'manhattan'
   - task: 'classification' or 'regression'

2. **`_calculate_distance(x1, x2)`** - Private helper method
   - Computes distance between two points
   - Supports multiple distance metrics
   - Returns a single float value

3. **`fit(X, y)`** - "Train" the model
   - Simply stores the training data
   - No actual learning happens (lazy learning)
   - O(1) time complexity

4. **`_predict_single(x)`** - Predict for one sample
   - Calculates distances to all training points
   - Finds k nearest neighbors
   - Returns single prediction

5. **`predict(X)`** - Predict for multiple samples
   - Calls _predict_single for each sample
   - Returns array of predictions
   - Main prediction interface

6. **`predict_proba(X)`** - Get class probabilities
   - Only for classification tasks
   - Returns probability distribution over classes
   - Based on neighbor label frequencies

7. **`score(X, y)`** - Calculate performance
   - Accuracy for classification
   - R² score for regression
   - Returns value between 0 and 1

---

## Step-by-Step Example

Let's walk through a complete example predicting **fruit type** based on weight and sweetness:

### The Data

```python
import numpy as np

# Features: [weight (grams), sweetness (1-10)]
X_train = np.array([
    [150, 8],   # Apple
    [170, 9],   # Apple
    [140, 7],   # Apple
    [350, 4],   # Orange
    [380, 5],   # Orange
    [340, 3],   # Orange
    [200, 9],   # Strawberry
    [180, 10],  # Strawberry
    [190, 8]    # Strawberry
])

# Labels: 0=Apple, 1=Orange, 2=Strawberry
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
```

### Training the Model

```python
model = KNearestNeighbors(k=3, task='classification')
model.fit(X_train, y_train)
```

**What happens internally**:
- The model simply stores X_train and y_train
- No coefficients are learned (unlike regression)
- Training is instantaneous!

### Making Predictions

```python
# New fruit to classify
X_test = np.array([[160, 8]])  # 160g, sweetness 8

# Step 1: Calculate distances to all training points
distances = [
    dist([160,8], [150,8]) = 10.0,    # Apple 1
    dist([160,8], [170,9]) = 10.05,   # Apple 2
    dist([160,8], [140,7]) = 20.02,   # Apple 3
    dist([160,8], [350,4]) = 195.92,  # Orange 1
    ...
]

# Step 2: Find k=3 nearest
Nearest: [10.0 (Apple), 10.05 (Apple), 20.02 (Apple)]

# Step 3: Vote
Classes: [0, 0, 0]
Prediction: 0 (Apple) ✓
```

### Complete Prediction Code

```python
# Predict for multiple samples
X_test = np.array([
    [160, 8],   # Should be Apple
    [360, 4],   # Should be Orange
    [185, 9]    # Should be Strawberry
])

predictions = model.predict(X_test)
print("Predicted classes:", predictions)
# Output: [0, 1, 2] (Apple, Orange, Strawberry)

# Get probabilities
probabilities = model.predict_proba(X_test)
print("\nProbabilities:")
for i, probs in enumerate(probabilities):
    print(f"Sample {i+1}: Apple={probs[0]:.2f}, Orange={probs[1]:.2f}, Strawberry={probs[2]:.2f}")
# Output:
# Sample 1: Apple=1.00, Orange=0.00, Strawberry=0.00
# Sample 2: Apple=0.00, Orange=1.00, Strawberry=0.00
# Sample 3: Apple=0.00, Orange=0.00, Strawberry=1.00
```

---

## Real-World Applications

### 1. **Recommender Systems**
Recommend products based on similar users:
- Input: User preferences, purchase history
- Output: Recommended products
- Example: "Customers like you also bought..."

### 2. **Image Recognition**
Classify images based on similar images:
- Input: Image features (pixels, edges, colors)
- Output: Object class (cat, dog, car, etc.)
- Example: "This image looks most like a cat"

### 3. **Medical Diagnosis**
Diagnose diseases based on similar patient profiles:
- Input: Symptoms, test results, medical history
- Output: Disease diagnosis
- Example: "Patient profile matches diabetes cases"

### 4. **Credit Risk Assessment**
Assess loan risk based on similar applicants:
- Input: Income, credit score, employment, debt
- Output: Risk level (low, medium, high)
- Example: "Similar profiles have 15% default rate"

### 5. **Handwriting Recognition**
Recognize handwritten digits:
- Input: Pixel intensities of handwritten digit
- Output: Digit (0-9)
- Example: "This handwriting looks like a '7'"

### 6. **Anomaly Detection**
Detect unusual patterns:
- Input: Transaction features, user behavior
- Output: Normal or anomalous
- Example: "This transaction differs from normal patterns"

### 7. **Real Estate Price Prediction**
Predict house prices based on similar properties:
- Input: Size, location, age, bedrooms
- Output: Estimated price
- Example: "Similar houses sold for $350k-$400k"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Distance Calculation

```python
def _calculate_distance(self, x1, x2):
    if self.distance_metric == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif self.distance_metric == 'manhattan':
        return np.sum(np.abs(x1 - x2))
```

**How it works**:
```python
# Euclidean example
x1 = [1, 2, 3]
x2 = [4, 5, 6]

diff = x1 - x2 = [-3, -3, -3]
squared = diff² = [9, 9, 9]
sum_squared = 27
distance = √27 ≈ 5.196

# Manhattan example
absolute_diff = |x1 - x2| = [3, 3, 3]
distance = sum([3, 3, 3]) = 9
```

**Why these metrics?**
- **Euclidean**: Natural measure, like using a ruler
- **Manhattan**: Useful in high dimensions, less affected by outliers

### 2. Finding Nearest Neighbors

```python
# Calculate all distances
distances = []
for x_train in self.X_train:
    distance = self._calculate_distance(x, x_train)
    distances.append(distance)

# Sort and get k smallest
k_indices = np.argsort(distances)[:self.k]
```

**Step-by-step**:
```python
# Example with k=3
distances = [5.2, 2.1, 8.3, 1.5, 4.7, 3.2]

# argsort returns indices that would sort the array
sorted_indices = [3, 1, 5, 4, 0, 2]
                  ↓  ↓  ↓
# Take first k=3
k_indices = [3, 1, 5]  # Points with distances 1.5, 2.1, 3.2
```

### 3. Making Classification Predictions

```python
# Get labels of k nearest neighbors
k_nearest_labels = self.y_train[k_indices]

# Find most common class
unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
prediction = unique_labels[np.argmax(counts)]
```

**Example**:
```python
k_nearest_labels = [0, 1, 0]  # Classes of 3 nearest neighbors

# Count occurrences
unique_labels = [0, 1]
counts = [2, 1]

# Most common
argmax(counts) = 0  # Index of maximum count
prediction = unique_labels[0] = 0
```

### 4. Making Regression Predictions

```python
# Get values of k nearest neighbors
k_nearest_values = self.y_train[k_indices]

# Calculate mean
prediction = np.mean(k_nearest_values)
```

**Example**:
```python
k_nearest_values = [100, 150, 125]

prediction = (100 + 150 + 125) / 3 = 125
```

### 5. Computing Probabilities

```python
# For each class, calculate proportion
for c in classes:
    prob = np.sum(k_nearest_labels == c) / self.k
    class_probs.append(prob)
```

**Example**:
```python
k = 5
k_nearest_labels = [0, 0, 1, 0, 2]
classes = [0, 1, 2]

# Count for each class
count_0 = 3  → prob_0 = 3/5 = 0.6
count_1 = 1  → prob_1 = 1/5 = 0.2
count_2 = 1  → prob_2 = 1/5 = 0.2

probabilities = [0.6, 0.2, 0.2]
```

---

## Model Evaluation

### For Classification

#### 1. Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Example**:
```python
y_true = [0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1]

correct = 4
total = 5
accuracy = 4/5 = 0.8 (80%)
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

#### R² Score (Coefficient of Determination)

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

**Example**:
```python
y_true = [100, 200, 150, 250]
y_pred = [110, 190, 160, 240]
y_mean = 175

SS_res = (100-110)² + (200-190)² + (150-160)² + (250-240)²
       = 100 + 100 + 100 + 100 = 400

SS_tot = (100-175)² + (200-175)² + (150-175)² + (250-175)²
       = 5625 + 625 + 625 + 5625 = 12500

R² = 1 - (400/12500) = 1 - 0.032 = 0.968 (96.8% of variance explained)
```

---

## Choosing the Right k

### The k-Value Trade-off

```
Small k (1-3):
  Pros: Flexible, captures local patterns
  Cons: Sensitive to noise, overfitting
  
Medium k (5-9):
  Pros: Balanced, good generalization
  Cons: May miss some local patterns
  
Large k (15+):
  Pros: Robust to noise, smooth boundaries
  Cons: May miss important patterns, underfitting
```

### Visual Comparison

```
k=1: Very Complex Decision Boundary
    ●●●■●●
    ●■■■●●
    ■■■●●●
    
k=5: Moderate Complexity
    ●●●●●●
    ------
    ■■■■■■
    
k=20: Very Simple Boundary
    ●●●●●●
    ======
    ■■■■■■
```

### Rule of Thumb

1. **Start with k = √n**
   - n = number of training samples
   - Example: 100 samples → k ≈ 10

2. **Use odd k for binary classification**
   - Avoids ties in voting
   - Example: k = 3, 5, 7, not 2, 4, 6

3. **Cross-validation**
   - Try multiple k values
   - Choose k with best validation performance

### Example: Finding Optimal k

```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
scores = []

for k in k_values:
    model = KNearestNeighbors(k=k, task='classification')
    # Imagine we have a cross_val_score implementation
    score = model.score(X_test, y_test)
    scores.append(score)

best_k = k_values[np.argmax(scores)]
print(f"Best k: {best_k}")
```

---

## Feature Scaling: Critical for KNN

### Why Scaling Matters

KNN uses distances, so feature scales matter greatly!

**Example without scaling**:
```python
Feature 1: Age (20-80)          → Range = 60
Feature 2: Income (20k-200k)    → Range = 180,000

Distance dominated by income!
Age difference of 30 years ≈ Income difference of $30
```

**Example with scaling**:
```python
Feature 1: Age (scaled to 0-1)        → Range = 1
Feature 2: Income (scaled to 0-1)     → Range = 1

Both features contribute equally!
```

### Standardization (Z-score)

Most common approach:

```
x_scaled = (x - mean) / std_dev
```

**Example**:
```python
from sklearn.preprocessing import StandardScaler

# Original data
X = [[20, 30000],
     [40, 60000],
     [60, 90000]]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Now: mean=0, std=1 for each feature
```

### Min-Max Scaling

Alternative approach:

```
x_scaled = (x - min) / (max - min)
```

**When to use each**:
- **Standardization**: When features have different distributions
- **Min-Max**: When you need values in specific range (0-1)

---

## Advantages and Limitations

### Advantages ✅

1. **Simple and Intuitive**
   - Easy to understand and explain
   - No complex math required

2. **No Training Phase**
   - Instant "training" (just storing data)
   - Can add new data easily

3. **Naturally Handles Multi-class**
   - No modification needed for multiple classes
   - Works with any number of classes

4. **No Assumptions**
   - Non-parametric (no assumptions about data distribution)
   - Flexible decision boundaries

5. **Good for Small to Medium Datasets**
   - Often performs well with limited data
   - Can capture complex patterns

### Limitations ❌

1. **Computationally Expensive**
   - O(n×d) for each prediction
   - n = training samples, d = dimensions
   - Slow on large datasets

2. **Memory Intensive**
   - Must store all training data
   - Large datasets require lots of memory

3. **Sensitive to Feature Scaling**
   - MUST scale features appropriately
   - Otherwise dominated by large-scale features

4. **Curse of Dimensionality**
   - Performance degrades in high dimensions
   - Distances become meaningless when d is large

5. **Sensitive to Irrelevant Features**
   - Noisy features affect all distance calculations
   - Feature selection is crucial

6. **Imbalanced Data Issues**
   - Majority class can dominate predictions
   - May need weighted KNN

### When to Use KNN

**Good Use Cases**:
- ✅ Small to medium datasets (< 100k samples)
- ✅ Low to moderate dimensions (< 20 features)
- ✅ No time constraints for prediction
- ✅ Need interpretable results
- ✅ Complex, non-linear decision boundaries

**Bad Use Cases**:
- ❌ Large datasets (millions of samples)
- ❌ High-dimensional data (100+ features)
- ❌ Real-time predictions required
- ❌ Features have very different scales (unless scaled)
- ❌ Many irrelevant features

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load iris dataset
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CRITICAL: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = KNearestNeighbors(k=5, distance_metric='euclidean', task='classification')
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
print("\nSample Predictions:")
for i in range(5):
    print(f"True: {data.target_names[y_test[i]]}")
    print(f"Predicted: {data.target_names[y_pred[i]]}")
    print(f"Probabilities: Setosa={y_proba[i][0]:.2f}, "
          f"Versicolor={y_proba[i][1]:.2f}, Virginica={y_proba[i][2]:.2f}\n")

# Try different k values
print("Testing different k values:")
for k in [1, 3, 5, 7, 9, 11]:
    model = KNearestNeighbors(k=k, task='classification')
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    print(f"k={k:2d}: Accuracy={score:.4f}")
```

---

## Optimizations and Variants

### 1. Weighted KNN

Give more weight to closer neighbors:

```
weight_i = 1 / distance_i

Prediction = Σ(weight_i × vote_i) / Σ(weight_i)
```

**Benefit**: Closer neighbors have more influence

### 2. Ball Tree / KD Tree

Data structures to speed up neighbor search:

```
Naive approach: O(n×d) per prediction
With tree: O(d×log(n)) per prediction
```

**Benefit**: Much faster for moderate dimensions

### 3. Approximate Nearest Neighbors

Trade accuracy for speed:

```
Find "approximately" nearest neighbors
Speedup: 10-100x faster
Accuracy loss: 1-5%
```

**Benefit**: Enables use on large datasets

---

## Key Concepts to Remember

### 1. **KNN is a Lazy Learner**
No training phase! Just stores data and computes at prediction time.

### 2. **Feature Scaling is CRITICAL**
Always standardize or normalize features before using KNN.

### 3. **k is the Most Important Hyperparameter**
- Too small → overfitting, noise sensitivity
- Too large → underfitting, over-smoothing
- Use cross-validation to find optimal k

### 4. **Distance Metric Matters**
- Euclidean: Most common, natural measure
- Manhattan: Better in high dimensions
- Choose based on your data and domain

### 5. **Computational Cost**
- Training: O(1) - instant
- Prediction: O(n×d) - expensive
- Memory: O(n×d) - stores all data

### 6. **Curse of Dimensionality**
Performance degrades as dimensions increase:
- Distances become similar in high dimensions
- "Nearest" neighbors aren't actually near
- Solution: Feature selection, dimensionality reduction

---

## Conclusion

K-Nearest Neighbors is a simple yet powerful algorithm! By understanding:
- How distance metrics measure similarity
- How voting/averaging produces predictions
- How k controls model complexity
- How feature scaling affects performance

You've gained a fundamental tool in your machine learning toolkit! 🎯

**When to Use KNN**:
- ✅ Small to medium datasets
- ✅ Non-linear decision boundaries
- ✅ Need interpretable predictions
- ✅ Multi-class classification
- ✅ Both classification and regression

**When to Use Something Else**:
- ❌ Large datasets → Use decision trees, random forests
- ❌ High dimensions → Use dimensionality reduction first
- ❌ Need fast predictions → Use logistic regression, SVM
- ❌ Many irrelevant features → Use regularized models

**Next Steps**:
- Try KNN on your own datasets
- Experiment with different k values and distance metrics
- Compare with other algorithms (Logistic Regression, Decision Trees)
- Learn about weighted KNN and approximate methods
- Explore KNN for anomaly detection
- Study curse of dimensionality in depth

Happy coding! 💻🎯

