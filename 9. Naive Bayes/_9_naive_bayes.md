# Naive Bayes from Scratch: A Comprehensive Guide

Welcome to the world of Naive Bayes! 🎯 In this comprehensive guide, we'll explore one of the most elegant and efficient machine learning algorithms. Think of it as the "assume the best, calculate probabilities" algorithm!

## Table of Contents
1. [What is Naive Bayes?](#what-is-naive-bayes)
2. [How Naive Bayes Works](#how-naive-bayes-works)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)

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

Where α is a smoothing parameter (usually 1, called Laplace smoothing)

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
        self.variant = variant
        self.classes = None
        self.class_priors = None
        self.means = None          # For Gaussian
        self.variances = None      # For Gaussian
        self.feature_probs = None  # For Multinomial
```

### Core Methods

1. **`__init__(variant)`** - Initialize model
   - variant: 'gaussian' or 'multinomial'
   - Determines which probability distribution to use

2. **`fit(X, y)`** - Train the model
   - Calculates prior probabilities P(class)
   - For Gaussian: learns mean and variance per feature per class
   - For Multinomial: learns feature probability distributions
   - Time complexity: O(n×d) where n=samples, d=features

3. **`_calculate_gaussian_likelihood(x, class_idx)`** - Private helper
   - Calculates P(features|class) using Gaussian distribution
   - Uses log probabilities to avoid numerical underflow
   - Returns log likelihood

4. **`_calculate_multinomial_likelihood(x, class_idx)`** - Private helper
   - Calculates P(features|class) for multinomial distribution
   - Uses Laplace smoothing to avoid zero probabilities
   - Returns log likelihood

5. **`_predict_single(x)`** - Predict for one sample
   - Calculates posterior for each class
   - Returns class with highest posterior
   - Uses log probabilities for numerical stability

6. **`predict(X)`** - Predict for multiple samples
   - Calls _predict_single for each sample
   - Returns array of predictions
   - Main prediction interface

7. **`predict_proba(X)`** - Get class probabilities
   - Returns posterior probability for each class
   - Probabilities sum to 1
   - Useful for confidence estimation

8. **`score(X, y)`** - Calculate accuracy
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
Weight: mean=155g, variance=150
Diameter: mean=7.25cm, variance=0.31
```

For Oranges (class 1):
```
Weight: mean=360g, variance=266.67
Diameter: mean=9.05cm, variance=0.19
```

### Making Predictions

```python
# New fruit to classify
X_test = np.array([[155, 7.2]])  # 155g, 7.2cm diameter

# Calculate posterior for Apple (class 0)
P(Apple) = 0.5
P(weight=155|Apple) = Gaussian(155, mean=155, var=150) = high probability
P(diameter=7.2|Apple) = Gaussian(7.2, mean=7.25, var=0.31) = high probability

Posterior(Apple) ∝ 0.5 × high × high = VERY HIGH

# Calculate posterior for Orange (class 1)
P(Orange) = 0.5
P(weight=155|Orange) = Gaussian(155, mean=360, var=266.67) = very low probability
P(diameter=7.2|Orange) = Gaussian(7.2, mean=9.05, var=0.19) = very low probability

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
# Output: [0, 1] (Apple, Orange)

# Get probabilities
probabilities = model.predict_proba(X_test)
print("\nProbabilities:")
for i, probs in enumerate(probabilities):
    print(f"Sample {i+1}: Apple={probs[0]:.4f}, Orange={probs[1]:.4f}")
# Output:
# Sample 1: Apple=0.9999, Orange=0.0001
# Sample 2: Apple=0.0001, Orange=0.9999
```

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
for idx, c in enumerate(self.classes):
    X_c = X[y == c]  # Get all samples of class c
    self.means[idx, :] = np.mean(X_c, axis=0)
    self.variances[idx, :] = np.var(X_c, axis=0)
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

```python
# Convert log posteriors to probabilities
posteriors = np.array(posteriors)
posteriors = np.exp(posteriors - np.max(posteriors))  # Numerical stability
posteriors = posteriors / np.sum(posteriors)  # Normalize to sum to 1
```

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

Test model on multiple train/test splits:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
mean_accuracy = np.mean(scores)
std_accuracy = np.std(scores)

print(f"Accuracy: {mean_accuracy:.3f} (+/- {std_accuracy:.3f})")
```

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

6. **Handles Missing Data**
   - Can ignore features with missing values
   - Just don't include them in probability calculation

7. **Online Learning**
   - Can update model with new data easily
   - No need to retrain from scratch

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

Adjust priors:
```python
# Give equal weight to all classes
n_classes = len(np.unique(y))
self.class_priors = np.ones(n_classes) / n_classes
```

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

