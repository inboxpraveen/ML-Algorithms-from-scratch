# Principal Component Analysis (PCA) from Scratch: A Comprehensive Guide

Welcome to the world of Principal Component Analysis! 📊 In this comprehensive guide, we'll explore one of the most powerful dimensionality reduction techniques in machine learning. Think of it as finding the "essence" of your data!

## Table of Contents
1. [What is PCA?](#what-is-pca)
2. [How PCA Works](#how-pca-works)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)

---

## What is PCA?

Principal Component Analysis (PCA) is a **dimensionality reduction technique** that transforms high-dimensional data into a lower-dimensional space while preserving as much information (variance) as possible.

**Real-world analogy**: 
Imagine taking a photo of a 3D sculpture. The photo is 2D, but if you position the camera correctly, you can capture most of the important details. PCA does exactly this - it finds the best "angles" (principal components) to view your data!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Unsupervised, Linear Transformation |
| **Purpose** | Dimensionality Reduction, Feature Extraction |
| **Tasks** | Data Compression, Visualization, Noise Reduction |
| **Output** | Lower-dimensional representation of data |
| **Key Benefit** | Reduces complexity while preserving information |

### The Core Idea

```
"Find the directions of maximum variance in your data"
```

PCA works by:
1. **Finding directions** (principal components) where data varies most
2. **Ranking them** by importance (variance explained)
3. **Projecting data** onto the top components
4. **Discarding** less important dimensions

---

## How PCA Works

### The Algorithm in 6 Steps

```
Step 1: Center the data (subtract mean from each feature)
         ↓
Step 2: Compute covariance matrix (how features vary together)
         ↓
Step 3: Calculate eigenvectors & eigenvalues of covariance matrix
         ↓
Step 4: Sort eigenvectors by eigenvalues (descending)
         ↓
Step 5: Select top k eigenvectors (principal components)
         ↓
Step 6: Project data onto principal components
```

### Visual Example: 2D to 1D

```
Original Data (2D):
    y
    4 |     ●
    3 |   ●   ●
    2 | ●   ●
    1 | ●
    0 +--------- x
      0 1 2 3 4

PCA finds direction of maximum spread:

    y
    4 |     ●
    3 |   ● / ●    ← PC1 (diagonal direction)
    2 | ● / ●          Maximum variance here!
    1 | ●
    0 +--------- x
      0 1 2 3 4

After projection to 1D:
    ●-●-●-●-●  ← All points on PC1 line
    (Most information preserved!)
```

### Why Reduce Dimensions?

**Benefits of PCA:**

```
High Dimensions (100 features):
  ❌ Hard to visualize
  ❌ Slow to train models
  ❌ Risk of overfitting
  ❌ Curse of dimensionality
  ❌ Lots of noise

         ↓ PCA ↓

Low Dimensions (10 features):
  ✅ Easy to visualize
  ✅ Fast training
  ✅ Better generalization
  ✅ Reduced complexity
  ✅ Noise filtered out
```

### Variance Preservation

```
Original data: 100 features
PCA with 10 components: 95% variance retained

Information preserved: 95%
Dimensions reduced: 90%
Speed improvement: 10x faster!
```

---

## The Mathematical Foundation

### 1. Data Centering

First, we center the data by subtracting the mean:

```
X_centered = X - mean(X)
```

**Why?** PCA finds directions of variance from the mean. If data isn't centered, the first component might just point toward the mean!

**Example:**
```python
Original data:
X = [[1, 2],
     [2, 4],
     [3, 6]]

Mean: [2, 4]

Centered:
X_centered = [[-1, -2],
              [ 0,  0],
              [ 1,  2]]
```

**Visualization:**
```
Before centering:        After centering:
    y                        y
    6 |   ●                  2 |   ●
    4 | ●                    0 | ●  (mean at origin)
    2 | ●                   -2 | ●
    0 +---- x                0 +---- x
      0 2 4                    -2 0 2
```

### 2. Covariance Matrix

The covariance matrix measures how features vary together:

```
Cov(X) = (X^T × X) / (n - 1)
```

For 2 features:
```
Cov = [ Var(x₁)      Cov(x₁,x₂) ]
      [ Cov(x₂,x₁)   Var(x₂)    ]
```

**Interpretation:**
- **Diagonal elements**: Variance of each feature
- **Off-diagonal elements**: Covariance between features
- **Positive covariance**: Features increase together
- **Negative covariance**: One increases, other decreases

**Example:**
```python
X = [[-1, -2],
     [ 0,  0],
     [ 1,  2]]

Cov = [[ 1.0,  2.0],   # Var(x₁)=1, Cov(x₁,x₂)=2
       [ 2.0,  4.0]]   # Cov(x₂,x₁)=2, Var(x₂)=4
```

**Meaning**: x₂ has more variance (4 vs 1), and they're positively correlated (cov=2)

### 3. Eigenvalues and Eigenvectors

We decompose the covariance matrix:

```
Cov × v = λ × v

where:
  v = eigenvector (direction of principal component)
  λ = eigenvalue (variance along that direction)
```

**Intuitive Meaning:**
- **Eigenvector**: Direction in space
- **Eigenvalue**: How much data spreads in that direction

**Example:**
```python
Cov = [[1, 2],
       [2, 4]]

Eigenvalues:  λ₁ = 5.0,  λ₂ = 0.0
Eigenvectors: v₁ = [0.45, 0.89],  v₂ = [-0.89, 0.45]
```

**Visualization:**
```
    y
    4 |     ●
    3 |   ●↗  ●   ← PC1: direction of v₁ (most variance)
    2 | ●↗  ●
    1 | ●          PC2: direction of v₂ (no variance)
    0 +--------- x
      0 1 2 3 4

PC1 (eigenvalue=5.0): Main direction of data spread
PC2 (eigenvalue=0.0): Perpendicular, no spread
```

### 4. Principal Components

Principal components are the eigenvectors sorted by eigenvalues:

```
PC1 = eigenvector with largest eigenvalue  (most important)
PC2 = eigenvector with 2nd largest eigenvalue
...
PCₙ = eigenvector with smallest eigenvalue  (least important)
```

**Properties:**
1. **Orthogonal**: All PCs are perpendicular to each other
2. **Ordered**: PC1 explains most variance, PC2 second-most, etc.
3. **Uncorrelated**: Features in PC space are independent

### 5. Projection Formula

To transform data to principal component space:

```
X_transformed = X_centered × PC^T

where:
  X_centered: centered data (n_samples × n_features)
  PC: principal components matrix (n_components × n_features)
  X_transformed: projected data (n_samples × n_components)
```

**Example:**
```python
X_centered = [[-1, -2],    PC = [[0.45, 0.89]]  (just PC1)
              [ 0,  0],
              [ 1,  2]]

X_transformed = X_centered × PC^T
              = [[-1×0.45 + -2×0.89],   = [[-2.23],
                 [ 0×0.45 +  0×0.89],      [ 0.00],
                 [ 1×0.45 +  2×0.89]]      [ 2.23]]

Reduced from 2D to 1D!
```

### 6. Explained Variance Ratio

How much information each component captures:

```
Explained Variance Ratio = λᵢ / Σλⱼ

where:
  λᵢ = eigenvalue of component i
  Σλⱼ = sum of all eigenvalues
```

**Example:**
```python
Eigenvalues: [5.0, 3.0, 1.0, 0.5]
Total: 9.5

Variance ratios:
  PC1: 5.0/9.5 = 0.526 (52.6%)
  PC2: 3.0/9.5 = 0.316 (31.6%)
  PC3: 1.0/9.5 = 0.105 (10.5%)
  PC4: 0.5/9.5 = 0.053 (5.3%)

Cumulative:
  PC1: 52.6%
  PC1+PC2: 84.2%
  PC1+PC2+PC3: 94.7%  ← Keep 3 components for ~95% variance!
  All: 100%
```

### 7. Reconstruction

To transform back to original space:

```
X_reconstructed = X_transformed × PC + mean

where:
  X_transformed: data in PC space
  PC: principal components
  mean: original data mean
```

**Note**: If we kept all components, reconstruction is perfect. If we dropped some, there's information loss.

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class PrincipalComponentAnalysis:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None           # Principal components
        self.mean_ = None                 # Data mean
        self.explained_variance_ = None   # Variance per component
        self.explained_variance_ratio_ = None  # Proportion of variance
```

### Core Methods

1. **`__init__(n_components)`** - Initialize model
   - n_components: Number of components to keep
   - Can be int (exact number) or float (variance threshold)
   - None = keep all components

2. **`fit(X)`** - Compute principal components
   - Centers the data
   - Computes covariance matrix
   - Finds eigenvectors and eigenvalues
   - Sorts and selects top components

3. **`transform(X)`** - Project data to PC space
   - Centers data using training mean
   - Multiplies by principal components
   - Returns lower-dimensional representation

4. **`fit_transform(X)`** - Convenience method
   - Combines fit() and transform()
   - Returns transformed data directly

5. **`inverse_transform(X_transformed)`** - Reconstruct data
   - Projects back to original space
   - Adds back the mean
   - Returns approximation of original data

6. **`score(X)`** - Evaluate model fit
   - Measures reconstruction error
   - Lower error = better fit
   - Based on mean squared error

---

## Step-by-Step Example

Let's walk through a complete example with **simple 2D data** to understand every step:

### The Data

```python
import numpy as np

# Original data: 5 samples, 2 features
X = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10]
])

print("Original data shape:", X.shape)  # (5, 2)
```

**Visualization:**
```
    y
   10 |        ●
    8 |      ●
    6 |    ●
    4 |  ●
    2 | ●
    0 +----------- x
      0 1 2 3 4 5

Data lies roughly on a line!
Perfect for PCA to 1D
```

### Step 1: Center the Data

```python
mean = np.mean(X, axis=0)  # [3, 6]
X_centered = X - mean

print("Centered data:")
print(X_centered)
# [[-2, -4],
#  [-1, -2],
#  [ 0,  0],
#  [ 1,  2],
#  [ 2,  4]]
```

**After centering:**
```
    y
    4 |        ●
    2 |      ●
    0 |    ●  (now centered at origin)
   -2 |  ●
   -4 | ●
      +----------- x
     -2 -1 0 1 2
```

### Step 2: Compute Covariance Matrix

```python
cov = np.cov(X_centered.T)

print("Covariance matrix:")
print(cov)
# [[ 2.5,  5.0],
#  [ 5.0, 10.0]]
```

**Interpretation:**
- Var(x) = 2.5: x varies from -2 to 2
- Var(y) = 10.0: y varies from -4 to 4 (more variance!)
- Cov(x,y) = 5.0: strong positive correlation

### Step 3: Compute Eigenvalues & Eigenvectors

```python
eigenvalues, eigenvectors = np.linalg.eig(cov)

print("Eigenvalues:", eigenvalues)
# [12.5, 0.0]

print("Eigenvectors:")
print(eigenvectors)
# [[ 0.447,  0.894],  ← eigenvector 1
#  [ 0.894, -0.447]]  ← eigenvector 2
```

**Analysis:**
- λ₁ = 12.5: First component captures ALL variance!
- λ₂ = 0.0: Second component has NO variance
- This makes sense - data is perfectly linear

### Step 4: Sort by Eigenvalues

```python
idx = eigenvalues.argsort()[::-1]  # [0, 1] (already sorted)
eigenvalues = eigenvalues[idx]     # [12.5, 0.0]
eigenvectors = eigenvectors[:, idx]

explained_variance_ratio = eigenvalues / sum(eigenvalues)
print("Variance explained:", explained_variance_ratio)
# [1.0, 0.0]  ← PC1 explains 100% of variance!
```

### Step 5: Select Principal Components

```python
# Keep just 1 component (reduces 2D → 1D)
n_components = 1
PC = eigenvectors[:, :n_components].T

print("Principal Component:")
print(PC)  # [[0.447, 0.894]]
```

**Visualization:**
```
    y
    4 |        ●
    2 |      ●↗     ← PC1 direction: [0.447, 0.894]
    0 |    ●↗
   -2 |  ●↗
   -4 | ●
      +----------- x
     -2 -1 0 1 2

PC1 points along the data spread!
```

### Step 6: Transform Data

```python
X_transformed = X_centered @ PC.T

print("Transformed data (1D):")
print(X_transformed)
# [[-4.47],
#  [-2.24],
#  [ 0.00],
#  [ 2.24],
#  [ 4.47]]
```

**Result:**
```
Original (2D):          Transformed (1D):
    y                       
    4 |        ●            4.47  ●
    2 |      ●             2.24  ●
    0 |    ●         →     0.00  ●
   -2 |  ●                -2.24  ●
   -4 | ●                 -4.47  ●
      +---- x

Reduced from 2D to 1D!
Information preserved: 100%
```

### Using Our PCA Class

```python
from _11_pca import PrincipalComponentAnalysis

# Create and fit PCA
pca = PrincipalComponentAnalysis(n_components=1)
X_reduced = pca.fit_transform(X)

print("Reduced data shape:", X_reduced.shape)  # (5, 1)
print("Variance explained:", pca.explained_variance_ratio_[0])  # 1.0

# Reconstruct
X_reconstructed = pca.inverse_transform(X_reduced)
print("Reconstruction error:", np.mean((X - X_reconstructed)**2))  # ~0
```

---

## Real-World Applications

### 1. **Image Compression**
Reduce image file size while preserving quality:
- Input: High-resolution image (millions of pixels)
- Output: Compressed representation
- Example: "Compress 1MB image to 100KB with 95% quality"

### 2. **Data Visualization**
Visualize high-dimensional data in 2D/3D:
- Input: Dataset with 100+ features
- Output: 2D/3D projection for plotting
- Example: "Visualize customer segments in 2D"

### 3. **Noise Reduction**
Remove noise while keeping signal:
- Input: Noisy measurements
- Output: Clean data (keeping top PCs)
- Example: "Clean sensor data by removing noise components"

### 4. **Feature Engineering**
Create better features for ML models:
- Input: Many correlated features
- Output: Fewer, uncorrelated features
- Example: "Transform 50 features → 10 principal components"

### 5. **Face Recognition (Eigenfaces)**
Represent faces compactly:
- Input: Face images (thousands of pixels)
- Output: Compact face representation (50 components)
- Example: "Recognize faces using 50 eigenfaces"

### 6. **Genomics**
Analyze gene expression data:
- Input: Expression levels of 20,000 genes
- Output: Key patterns (principal components)
- Example: "Find main patterns in cancer gene expression"

### 7. **Recommender Systems**
Find latent factors in user preferences:
- Input: User-item ratings matrix
- Output: Latent features (user/item factors)
- Example: "Find movie preference patterns"

### 8. **Finance**
Identify main market factors:
- Input: Returns of 500 stocks
- Output: Main market factors (5-10 components)
- Example: "Find main drivers of stock market"

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Centering the Data

```python
self.mean_ = np.mean(X, axis=0)
X_centered = X - self.mean_
```

**What this does:**
```python
# Example
X = [[1, 10],
     [2, 20],
     [3, 30]]

mean = [2, 20]  # average of each column

X_centered = [[-1, -10],  # 1-2=-1, 10-20=-10
              [ 0,   0],  # 2-2=0,  20-20=0
              [ 1,  10]]  # 3-2=1,  30-20=10
```

**Why necessary?** 
- PCA measures variance from the mean
- Without centering, results are wrong!
- The first PC might just point to the data mean

### 2. Computing Covariance

```python
covariance_matrix = np.cov(X_centered.T)
```

**What `np.cov` does:**
```python
# For each pair of features i,j:
cov[i,j] = sum((X[:,i] - mean[i]) * (X[:,j] - mean[j])) / (n-1)

# Result is symmetric matrix:
#     feature1  feature2
# f1 [  var1      cov12  ]
# f2 [  cov21     var2   ]
```

**Interpretation:**
```python
cov = [[2.0, 1.5],
       [1.5, 3.0]]

# Means:
# - Feature 1 has variance 2.0
# - Feature 2 has variance 3.0
# - They're positively correlated (cov=1.5)
```

### 3. Eigendecomposition

```python
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
```

**What this finds:**
```
For covariance matrix C, finds vectors v and values λ such that:
C × v = λ × v

Results:
- eigenvalues: [λ₁, λ₂, ..., λₙ]  (variance along each PC)
- eigenvectors: [v₁, v₂, ..., vₙ] (direction of each PC)
```

**Example:**
```python
C = [[2, 1],
     [1, 2]]

eigenvalues = [3, 1]
eigenvectors = [[0.707,  0.707],   # PC1: diagonal direction
                [0.707, -0.707]]   # PC2: other diagonal
```

### 4. Sorting Components

```python
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

**Step by step:**
```python
# Before sorting
eigenvalues = [1.0, 5.0, 3.0]

# argsort gives indices that would sort
idx = eigenvalues.argsort()  # [0, 2, 1]

# [::-1] reverses to get descending order
idx = idx[::-1]  # [1, 2, 0]

# Apply sorting
eigenvalues = eigenvalues[idx]  # [5.0, 3.0, 1.0] ✓
eigenvectors = eigenvectors[:, idx]  # columns reordered
```

### 5. Selecting Components

```python
if isinstance(self.n_components, float) and 0 < self.n_components < 1:
    # Keep enough for desired variance
    cumsum = np.cumsum(self.explained_variance_ratio_)
    self.n_components_ = np.argmax(cumsum >= self.n_components) + 1
```

**How this works:**
```python
explained_variance_ratio = [0.5, 0.3, 0.15, 0.05]
cumsum = [0.5, 0.8, 0.95, 1.0]

# Want 95% variance
n_components = 0.95

# Find first index where cumsum >= 0.95
np.argmax(cumsum >= 0.95)  # returns 2
n_components_ = 2 + 1 = 3  # need 3 components
```

### 6. Projection

```python
X_transformed = np.dot(X_centered, self.components_.T)
```

**Matrix multiplication:**
```python
X_centered: (n_samples × n_features)
components: (n_components × n_features)
components.T: (n_features × n_components)

Result: (n_samples × n_components)

# Example:
X_centered = [[-1, -2, -3],    components = [[0.5, 0.5, 0.7],
              [ 0,  0,  0],                  [0.7, 0.0, 0.7]]
              [ 1,  2,  3]]    

components.T = [[0.5, 0.7],
                [0.5, 0.0],
                [0.7, 0.7]]

X_transformed = X_centered × components.T
              = [[-1×0.5+-2×0.5+-3×0.7, -1×0.7+-2×0.0+-3×0.7],
                 [ 0×0.5+ 0×0.5+ 0×0.7,  0×0.7+ 0×0.0+ 0×0.7],
                 [ 1×0.5+ 2×0.5+ 3×0.7,  1×0.7+ 2×0.0+ 3×0.7]]
              = [[-3.6, -2.8],
                 [ 0.0,  0.0],
                 [ 3.6,  2.8]]

Reduced from 3D to 2D!
```

### 7. Inverse Transform

```python
X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
```

**How it works:**
```python
# Forward: X_centered × PC.T = X_transformed
# Backward: X_transformed × PC = X_centered_reconstructed

X_transformed: (n_samples × n_components)
components: (n_components × n_features)
Result: (n_samples × n_features)

Then add back mean to get back to original scale
```

**Example:**
```python
X_transformed = [[2.0],
                 [0.0],
                 [-2.0]]

components = [[0.7, 0.7]]  # 1 component

X_centered_recon = X_transformed × components
                 = [[2.0×0.7, 2.0×0.7],
                    [0.0×0.7, 0.0×0.7],
                    [-2.0×0.7, -2.0×0.7]]
                 = [[ 1.4,  1.4],
                    [ 0.0,  0.0],
                    [-1.4, -1.4]]

X_reconstructed = X_centered_recon + mean
                = [[ 1.4,  1.4],    + [3, 5]
                   [ 0.0,  0.0],
                   [-1.4, -1.4]]
                = [[4.4, 6.4],
                   [3.0, 5.0],
                   [1.6, 3.6]]
```

---

## Model Evaluation

### 1. Explained Variance

Shows how much information each component captures:

```python
pca = PrincipalComponentAnalysis(n_components=None)
pca.fit(X)

print("Explained variance:")
for i, var in enumerate(pca.explained_variance_):
    print(f"  PC{i+1}: {var:.2f}")

print("\nExplained variance ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
```

**Output example:**
```
Explained variance:
  PC1: 12.50
  PC2: 3.20
  PC3: 0.80
  PC4: 0.20

Explained variance ratio:
  PC1: 0.7561 (75.61%)  ← Most important!
  PC2: 0.1935 (19.35%)
  PC3: 0.0484 (4.84%)
  PC4: 0.0121 (1.21%)   ← Least important
```

### 2. Cumulative Variance

Shows total variance retained with k components:

```python
cumulative = np.cumsum(pca.explained_variance_ratio_)

print("Cumulative variance:")
for i, cum in enumerate(cumulative):
    print(f"  First {i+1} components: {cum:.4f} ({cum*100:.2f}%)")
```

**Output:**
```
Cumulative variance:
  First 1 components: 0.7561 (75.61%)
  First 2 components: 0.9496 (94.96%)  ← 2 components for ~95%!
  First 3 components: 0.9980 (99.80%)
  First 4 components: 1.0000 (100.00%)
```

**Decision**: Keep 2 components to retain 95% variance!

### 3. Reconstruction Error

Measures information loss from dimensionality reduction:

```python
# Original data
X_original = X

# Reduce dimensions
X_reduced = pca.transform(X)

# Reconstruct
X_reconstructed = pca.inverse_transform(X_reduced)

# Calculate error
mse = np.mean((X_original - X_reconstructed) ** 2)
relative_error = mse / np.var(X_original)

print(f"Reconstruction MSE: {mse:.6f}")
print(f"Relative error: {relative_error:.6f}")
```

**Interpretation:**
```
MSE = 0.001: Excellent reconstruction (very little loss)
MSE = 0.1: Good reconstruction (acceptable loss)
MSE = 1.0: Poor reconstruction (significant loss)
```

### 4. Visual Evaluation: Scree Plot

A scree plot shows variance explained by each component:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Plot variance per component
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')

# Plot cumulative variance
plt.subplot(1, 2, 2)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.title('Cumulative Variance')
plt.legend()

plt.tight_layout()
plt.show()
```

**What to look for:**
```
Good scree plot:         Bad scree plot:
    |█                       |█
    |█                       |█
    |▓                       |█
    |▒                       |█
    |░ ░ ░ ░                 |▓
    +-------                 |▒ ░ ░ ░
    ↑                        +-------
"Elbow" here:               No clear elbow
Keep ~3 components          Hard to decide
```

### 5. Component Loadings

Shows how much each original feature contributes to each PC:

```python
# Get loadings
loadings = pca.components_

print("Component loadings:")
print("Feature contributions to each PC:")
for i, pc in enumerate(loadings):
    print(f"\nPC{i+1}:")
    for j, loading in enumerate(pc):
        print(f"  Feature {j+1}: {loading:.4f}")
```

**Interpretation:**
```
PC1: [0.7, 0.7, 0.0, 0.0]
  → PC1 is combination of Features 1 and 2

PC2: [0.0, 0.0, 0.7, -0.7]
  → PC2 is difference between Features 3 and 4
```

---

## Choosing Number of Components

### Method 1: Fixed Number

Use when you know exact dimensionality needed:

```python
# Reduce to 2D for visualization
pca = PrincipalComponentAnalysis(n_components=2)
X_2d = pca.fit_transform(X)
```

**When to use:**
- Visualization (2D or 3D)
- Fixed architecture requirements
- Specific dimensionality reduction goal

### Method 2: Variance Threshold

Keep components until reaching variance threshold:

```python
# Keep 95% of variance
pca = PrincipalComponentAnalysis(n_components=0.95)
X_reduced = pca.fit_transform(X)

print(f"Kept {pca.n_components_} components")
print(f"Variance retained: {sum(pca.explained_variance_ratio_[:pca.n_components_]):.2%}")
```

**Common thresholds:**
- 0.95 (95%): Standard choice, good balance
- 0.99 (99%): High fidelity, minimal loss
- 0.90 (90%): Aggressive reduction, faster
- 0.80 (80%): Very aggressive, major speedup

### Method 3: Elbow Method

Look for "elbow" in scree plot:

```
Variance explained:
    |
0.5 |●               ← PC1: 50%
0.3 |  ●             ← PC2: 30%
0.1 |    ●           ← PC3: 10% (elbow here!)
0.05|      ● ● ●     ← PC4+: small contributions
    +------------
     1  2  3  4  5

Choose 3 components (before elbow flattens)
```

### Method 4: Cross-Validation

Evaluate with downstream task:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

best_score = 0
best_n = 0

for n in [2, 5, 10, 15, 20, 25]:
    pca = PrincipalComponentAnalysis(n_components=n)
    X_reduced = pca.fit_transform(X_train)
    
    clf = LogisticRegression()
    scores = cross_val_score(clf, X_reduced, y_train, cv=5)
    avg_score = np.mean(scores)
    
    print(f"n={n:2d}: {avg_score:.4f}")
    
    if avg_score > best_score:
        best_score = avg_score
        best_n = n

print(f"\nBest n_components: {best_n}")
```

### Decision Framework

```
Use Fixed Number when:
  ✓ Visualization requirement (2D/3D)
  ✓ Hardware constraints
  ✓ Fixed model architecture

Use Variance Threshold when:
  ✓ Want to preserve information
  ✓ Balance speed vs accuracy
  ✓ Standard use case

Use Elbow Method when:
  ✓ Exploratory analysis
  ✓ Want intuitive selection
  ✓ No specific target

Use Cross-Validation when:
  ✓ Have labeled data
  ✓ Downstream task defined
  ✓ Need optimal performance
```

---

## Feature Scaling: Critical for PCA

### Why Scaling Matters

PCA is sensitive to feature scales because it measures variance:

**Without scaling:**
```python
Feature 1: Age (20-80)           → Variance ≈ 400
Feature 2: Income (20k-200k)     → Variance ≈ 3,000,000,000

PCA will be dominated by income!
Age is virtually ignored
```

**With scaling:**
```python
Feature 1: Age (scaled)      → Variance ≈ 1
Feature 2: Income (scaled)   → Variance ≈ 1

Both features contribute fairly!
```

### Standardization (Z-score)

Most common for PCA:

```
x_scaled = (x - mean) / std
```

**Code:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now: mean=0, std=1 for each feature
pca = PrincipalComponentAnalysis(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
```

**After standardization:**
```
All features have:
  - Mean = 0
  - Standard deviation = 1
  - Equal contribution to PCA
```

### Min-Max Scaling

Alternative when you need specific range:

```
x_scaled = (x - min) / (max - min)
```

**Code:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Now: all features in range [0, 1]
pca = PrincipalComponentAnalysis(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
```

### When Scaling is Optional

Don't scale when:
- All features already on same scale
- Features are same units (e.g., all pixels in image)
- Variance differences are meaningful

**Example: Image pixels**
```python
# All pixels are 0-255, same scale
X_images = load_images()  # shape: (n_images, height*width)

# Can apply PCA directly
pca = PrincipalComponentAnalysis(n_components=50)
X_pca = pca.fit_transform(X_images)
```

---

## Advantages and Limitations

### Advantages ✅

1. **Dimensionality Reduction**
   - Reduces features while keeping information
   - Makes data manageable
   - Speeds up training

2. **Removes Correlation**
   - Principal components are uncorrelated
   - Better for algorithms sensitive to correlation
   - Cleaner feature space

3. **Noise Reduction**
   - Small components often represent noise
   - Keeping top components filters noise
   - Improves signal-to-noise ratio

4. **Visualization**
   - Reduce to 2D/3D for plotting
   - Understand data structure
   - Identify patterns and clusters

5. **Computational Efficiency**
   - Fewer features = faster algorithms
   - Lower memory requirements
   - Scalable to large datasets

6. **No Labels Needed**
   - Unsupervised method
   - Works without target variable
   - Useful for exploratory analysis

### Limitations ❌

1. **Interpretability Loss**
   - PCs are linear combinations of features
   - Hard to interpret what PCs mean
   - Original features more interpretable

2. **Linear Assumption**
   - Only captures linear relationships
   - Misses non-linear patterns
   - May need kernel PCA for non-linear data

3. **Scale Sensitive**
   - MUST scale features appropriately
   - Results change with scaling method
   - Can be misleading if not scaled

4. **Variance ≠ Importance**
   - High variance doesn't always mean important
   - Low variance features might be crucial
   - May lose important information

5. **Outlier Sensitive**
   - Outliers affect mean and variance
   - Can skew principal components
   - May need outlier removal first

6. **Not Sparse**
   - PCs use all features (non-zero coefficients)
   - Cannot remove irrelevant features
   - Consider sparse PCA for feature selection

### When to Use PCA

**Good Use Cases:**
- ✅ Many correlated features
- ✅ Need faster training
- ✅ High-dimensional visualization
- ✅ Remove multicollinearity
- ✅ Compress data
- ✅ Noise reduction

**Bad Use Cases:**
- ❌ Features already independent
- ❌ Need interpretable features
- ❌ Non-linear relationships
- ❌ Very sparse data
- ❌ Few features to begin with
- ❌ Outliers present (clean first)

---

## Complete Usage Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load breast cancer dataset (30 features)
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Original dataset: {X.shape}")
print(f"Features: {data.feature_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CRITICAL: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit PCA with all components (for analysis)
pca_full = PrincipalComponentAnalysis(n_components=None)
pca_full.fit(X_train_scaled)

# Analyze variance
print("\n=== Variance Analysis ===")
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
for i, (var, cum) in enumerate(zip(pca_full.explained_variance_ratio_[:10], cumsum[:10])):
    print(f"PC{i+1:2d}: {var:6.4f} ({var*100:5.2f}%)  |  Cumulative: {cum:6.4f} ({cum*100:5.2f}%)")

# Find number of components for 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"\nComponents for 95% variance: {n_components_95}")

# Apply PCA with optimal components
pca = PrincipalComponentAnalysis(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\n=== Dimensionality Reduction ===")
print(f"Original dimensions: {X_train_scaled.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")
print(f"Reduction: {(1 - X_train_pca.shape[1]/X_train_scaled.shape[1])*100:.1f}%")
print(f"Variance retained: {sum(pca.explained_variance_ratio_[:pca.n_components_]):.4f}")

# Train classifier on original data
print("\n=== Model Performance ===")
clf_original = LogisticRegression(max_iter=10000, random_state=42)
clf_original.fit(X_train_scaled, y_train)
y_pred_original = clf_original.predict(X_test_scaled)
acc_original = accuracy_score(y_test, y_pred_original)
print(f"Accuracy (original 30D): {acc_original:.4f}")

# Train classifier on PCA data
clf_pca = LogisticRegression(max_iter=10000, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy (PCA {pca.n_components_}D):   {acc_pca:.4f}")

# Calculate reconstruction error
X_reconstructed = pca.inverse_transform(X_train_pca)
recon_error = np.mean((X_train_scaled - X_reconstructed) ** 2)
print(f"\nReconstruction MSE: {recon_error:.6f}")

# Visualize in 2D
pca_2d = PrincipalComponentAnalysis(n_components=2)
X_train_2d = pca_2d.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 6))
colors = ['red', 'blue']
labels = ['Malignant', 'Benign']

for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y_train == i
    plt.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                c=color, alpha=0.6, label=label, edgecolors='k', s=50)

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Breast Cancer Dataset - PCA Projection (2D)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n2D projection retains {sum(pca_2d.explained_variance_ratio_):.2%} of variance")
```

---

## PCA vs Other Dimensionality Reduction Methods

### PCA vs t-SNE

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| Type | Linear | Non-linear |
| Speed | Fast | Slow |
| Deterministic | Yes | No |
| Global structure | Preserved | Not preserved |
| Local structure | Not emphasized | Preserved |
| Use case | General purpose | Visualization |

### PCA vs LDA

| Aspect | PCA | LDA |
|--------|-----|-----|
| Supervision | Unsupervised | Supervised |
| Goal | Maximum variance | Maximum separation |
| Labels needed | No | Yes |
| Use case | Reduce dimensions | Classification |

### PCA vs Autoencoders

| Aspect | PCA | Autoencoders |
|--------|-----|--------------|
| Complexity | Linear | Non-linear |
| Training | Instant | Time-consuming |
| Interpretability | Better | Worse |
| Flexibility | Limited | High |

---

## Key Concepts to Remember

### 1. **PCA Finds Directions of Maximum Variance**
Not necessarily the most "important" features, but directions where data spreads most.

### 2. **Always Standardize Features**
PCA is extremely sensitive to feature scales. Standardization is almost always necessary.

### 3. **Principal Components are Orthogonal**
All PCs are perpendicular to each other and uncorrelated.

### 4. **Variance Explained is Key**
Use explained variance ratio to decide how many components to keep.

### 5. **PCA is Linear**
Can only capture linear relationships. For non-linear patterns, use kernel PCA or other methods.

### 6. **Information Loss is Inevitable**
Keeping fewer components means losing some information. This is a tradeoff for dimensionality reduction.

---

## Conclusion

Principal Component Analysis is a fundamental technique in data science! By understanding:
- How PCA finds directions of maximum variance
- How eigenvalues and eigenvectors work
- How to choose the right number of components
- How to properly scale features

You've gained a powerful tool for:
- ✅ Reducing dimensionality
- ✅ Visualizing high-dimensional data
- ✅ Removing noise
- ✅ Speeding up ML algorithms
- ✅ Understanding data structure

**When to Use PCA:**
- ✅ Many correlated features
- ✅ Need visualization (2D/3D)
- ✅ Speed up training
- ✅ Remove multicollinearity
- ✅ Compress data

**When to Consider Alternatives:**
- ❌ Need interpretable features → Use feature selection
- ❌ Non-linear relationships → Use kernel PCA or t-SNE
- ❌ Labeled data for classification → Use LDA
- ❌ Very sparse data → Use sparse PCA
- ❌ Complex patterns → Use autoencoders

**Next Steps:**
- Apply PCA to your datasets
- Experiment with different numbers of components
- Combine PCA with classification/regression
- Learn about kernel PCA for non-linear data
- Explore other dimensionality reduction techniques (t-SNE, UMAP)
- Study applications in specific domains (images, genomics, finance)

Happy dimensionality reducing! 📊🎯

