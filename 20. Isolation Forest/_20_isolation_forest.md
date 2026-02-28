# Isolation Forest - Complete Guide

![Isolation Forest](https://img.shields.io/badge/Algorithm-Anomaly%20Detection-red)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow)
![Type](https://img.shields.io/badge/Type-Unsupervised-blue)

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [When to Use Isolation Forest](#when-to-use)
3. [How It Works](#how-it-works)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Usage Examples](#usage-examples)
7. [Hyperparameters Explained](#hyperparameters)
8. [Advantages & Limitations](#advantages-limitations)
9. [Comparison with Other Methods](#comparison)
10. [Tips & Best Practices](#tips)

---

## 🎯 Introduction

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

## 🎓 When to Use Isolation Forest

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

## 🔧 How It Works

Isolation Forest builds an ensemble of **Isolation Trees** (similar to Random Forests) but with a completely different goal and construction method.

### Step-by-Step Process

#### 1. Build Multiple Isolation Trees

For each tree:
1. **Randomly subsample** data (typically 256 samples)
2. **Randomly select** a feature
3. **Randomly select** a split value between min and max of that feature
4. **Recursively partition** until:
   - Only one sample in node, OR
   - Maximum depth reached, OR
   - All samples are identical

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

```
Isolation Tree Example (depth = 3):

                [Feature 2 < 5.3]
               /                 \
     [Feature 0 < 2.1]      [Feature 1 < 8.4]
      /            \          /            \
  [Anomaly]    [Normal]  [Normal]      [Normal]
  (depth=2)    (depth=2)  (depth=2)     (depth=2)

Anomaly isolated at depth 2 (shorter path)
Normal points isolated at depth 2-3 (longer paths)
```

---

## 📐 Mathematical Foundation

### Path Length

For a sample **x** in an isolation tree:
- **h(x)** = number of edges from root to leaf where x lands

### Average Path Length

Over all trees:

```
E[h(x)] = (1/n) × Σ h_i(x)
```

where n is the number of trees

### Normalization Constant

To normalize path lengths, we use the average path length of unsuccessful search in a Binary Search Tree with n nodes:

```
c(n) = 2H(n-1) - 2(n-1)/n
```

where H(i) ≈ ln(i) + 0.5772 (Euler's constant)

This represents the expected path length for a normal point.

### Anomaly Score

The final anomaly score is:

```
s(x, n) = 2^(-E[h(x)] / c(n))
```

**Interpretation:**
- **E[h(x)] ≪ c(n)** → s(x) → 1 (very anomalous)
- **E[h(x)] ≈ c(n)** → s(x) ≈ 0.5 (borderline)
- **E[h(x)] ≫ c(n)** → s(x) → 0 (very normal)

### Why This Works

The math formalizes the intuition:
1. **Anomalies are isolated faster** → shorter path h(x)
2. **Shorter path relative to c(n)** → score approaches 1
3. **Exponential transformation** → emphasizes differences

---

## 💻 Implementation Details

### Core Algorithm

```python
class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', 
                 contamination=0.1, random_state=None):
        # Initialize parameters
        
    def fit(self, X):
        # 1. Determine subsample size
        # 2. Build n_estimators isolation trees
        # 3. Calculate anomaly threshold
        
    def predict(self, X):
        # 1. Calculate average path lengths
        # 2. Convert to anomaly scores
        # 3. Apply threshold to classify
```

### Building an Isolation Tree

```python
def _build_tree(X, height_limit, current_height=0):
    # Base cases: stop if max depth or single sample
    if current_height >= height_limit or len(X) <= 1:
        return leaf_node
    
    # Randomly select feature and split value
    feature = random.choice(features)
    min_val, max_val = X[:, feature].min(), X[:, feature].max()
    split_value = random.uniform(min_val, max_val)
    
    # Partition data
    left = X[X[:, feature] < split_value]
    right = X[X[:, feature] >= split_value]
    
    # Recursively build subtrees
    return {
        'feature': feature,
        'split': split_value,
        'left': _build_tree(left, height_limit, current_height + 1),
        'right': _build_tree(right, height_limit, current_height + 1)
    }
```

### Key Differences from Random Forest

| Aspect | Isolation Forest | Random Forest |
|--------|------------------|---------------|
| **Goal** | Isolate anomalies | Classify/predict |
| **Splits** | Randomly selected | Optimized (information gain) |
| **Training** | Unsupervised | Supervised |
| **Output** | Anomaly score | Class/value |
| **Depth** | Limited (log n) | Can be deep |

---

## 📚 Usage Examples

### Example 1: Basic Anomaly Detection

```python
import numpy as np
from _20_isolation_forest import IsolationForest

# Generate data
np.random.seed(42)
X_normal = np.random.randn(300, 2) * 0.5
X_anomalies = np.random.uniform(-4, 4, (20, 2))
X = np.vstack([X_normal, X_anomalies])

# Train model
model = IsolationForest(n_estimators=100, contamination=0.1)
model.fit(X)

# Predict
predictions = model.predict(X)
scores = model.decision_function(X)

print(f"Anomalies detected: {np.sum(predictions == -1)}")
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

# Monitor new data
while True:
    new_sample = get_latest_metrics()
    score = model.decision_function([new_sample])[0]
    
    if score > threshold:
        alert_admin(f"Anomaly detected! Score: {score}")
```

---

## ⚙️ Hyperparameters Explained

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
```

- **1.0**: All features (default)
- **< 1.0**: Increases randomness and diversity
- **Use when**: High-dimensional data or correlated features

### random_state

**Random seed for reproducibility**

```python
model = IsolationForest(random_state=42)
```

- Set for reproducible results
- Important for debugging and comparison

---

## ✅ Advantages & Limitations

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

5. **Not Ideal for Streaming Data**
   - Need to retrain for concept drift
   - Online learning not straightforward

6. **Boundary Cases**
   - Samples near decision boundary can be unstable
   - May classify differently between runs

---

## 🔄 Comparison with Other Anomaly Detection Methods

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

## 💡 Tips & Best Practices

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

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_samples': [128, 256, 512],
    'contamination': [0.05, 0.1, 0.15]
}

# Note: Need labeled validation data for this
best_model = GridSearchCV(
    IsolationForest(),
    param_grid,
    scoring='accuracy',
    cv=3
).fit(X_train, y_train)
```

---

## 📊 Performance Characteristics

### Time Complexity

- **Training**: O(t × ψ × log ψ × d)
  - t = number of trees
  - ψ = subsample size
  - d = number of features

- **Prediction**: O(t × log ψ × d)

### Space Complexity

- O(t × ψ)
- Much lower than methods that store all training data

### Typical Runtimes (approximate)

| Dataset Size | Features | Trees | Training Time | Prediction (1000 samples) |
|--------------|----------|-------|---------------|---------------------------|
| 1,000 | 10 | 100 | < 1 sec | < 0.1 sec |
| 10,000 | 50 | 100 | ~5 sec | < 0.5 sec |
| 100,000 | 100 | 100 | ~30 sec | ~2 sec |
| 1,000,000 | 100 | 100 | ~5 min | ~20 sec |

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
- `predict(X)`: Return -1 for anomaly, 1 for normal
- `decision_function(X)`: Return anomaly scores (0-1)
- `score_samples(X)`: Return negative anomaly scores

### Key Attributes After Fitting
- `model.trees`: List of isolation trees
- `model.threshold`: Anomaly score threshold
- `model.max_samples_`: Actual subsample size used

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
