# t-SNE Algorithm from Scratch: A Comprehensive Guide

Welcome to the world of dimensionality reduction and visualization! In this comprehensive guide, we'll explore t-SNE (t-Distributed Stochastic Neighbor Embedding) - one of the most powerful algorithms for visualizing high-dimensional data in 2D or 3D space.

## Table of Contents
1. [What is t-SNE?](#what-is-t-sne)
2. [How t-SNE Works](#how-t-sne-works)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)

---

## What is t-SNE?

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a **non-linear dimensionality reduction technique** primarily used for **visualizing high-dimensional data**. It excels at preserving local structure - keeping similar data points close together while revealing cluster patterns.

**Real-world analogy**: 
Imagine you have a 3D sculpture that you want to photograph (project to 2D). Instead of just taking a flat projection, t-SNE is like an intelligent artist who arranges objects on a canvas to preserve which objects were near each other in 3D, making clusters and relationships visible even in 2D!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Non-linear Dimensionality Reduction |
| **Learning Style** | Unsupervised Learning |
| **Primary Use** | Data Visualization, Exploratory Analysis |
| **Output** | 2D or 3D embedding coordinates |
| **Key Strength** | Preserves local structure, reveals clusters |

### The Core Idea

```
"Convert high-dimensional distances to probabilities representing similarities,
then find a low-dimensional map that matches these similarity patterns"
```

Unlike PCA (which preserves global structure linearly), t-SNE:
- Focuses on preserving **local neighborhoods**
- Can capture **non-linear relationships**
- Excels at **revealing cluster structure**
- Uses **probability distributions** to model similarities

### When to Use t-SNE

**Perfect for:**
- 📊 Visualizing high-dimensional datasets (100+ dimensions)
- 🔍 Exploring cluster structure in your data
- 🖼️ Visualizing image embeddings, word vectors, or features
- 🧬 Analyzing biological data (gene expression, protein structures)
- 🎨 Understanding neural network representations

**Not ideal for:**
- ❌ Exact distance preservation (use MDS)
- ❌ Interpretable dimensions (use PCA)
- ❌ Very large datasets (>10,000 points) without optimization
- ❌ Finding outliers (t-SNE can hide them)

---

## How t-SNE Works

### The Algorithm in 5 Steps

```
Step 1: Compute pairwise distances in high-dimensional space
         ↓
Step 2: Convert distances to probabilities (Gaussian distribution)
         Adjust variance to achieve target "perplexity"
         ↓
Step 3: Initialize random low-dimensional embedding (2D or 3D)
         ↓
Step 4: Compute low-dimensional probabilities (Student t-distribution)
         ↓
Step 5: Minimize KL divergence using gradient descent
         Move points to match high-D and low-D probabilities
```

### Visual Example

Let's say we have 5 points in 3D space that we want to map to 2D:

```
High-Dimensional Space (3D):
Points: A, B, C, D, E

Compute distances:
  A-B: close    (distance = 1.2)
  A-C: far      (distance = 5.8)
  B-C: medium   (distance = 3.1)
  ...

Convert to probabilities (using Gaussian):
  P(A similar to B) = 0.35  (high - they're close)
  P(A similar to C) = 0.02  (low - they're far)
  ...
```

**Step 1: High-Dimensional Similarities**

```
Gaussian Kernel: 
  Similarity ∝ exp(-distance²/(2σ²))

For point A, compute similarity to all others:
  sim(A→B) = exp(-1.2²/(2σ²)) = 0.68  (after normalization)
  sim(A→C) = exp(-5.8²/(2σ²)) = 0.01
  
Normalize to get probability distribution
```

**Step 2: Choose Variance (σ) Based on Perplexity**

```
Perplexity = 2^(Entropy)

Controls effective number of neighbors:
  Low perplexity (5):  Very local, few neighbors matter
  Med perplexity (30): Balanced approach
  High perplexity (50): More global structure

For each point, binary search to find σ that gives target perplexity
```

**Step 3: Initialize 2D Positions Randomly**

```
Initial Random Placement:
    C
      A    
  E       B
         D
         
(Random positions near origin)
```

**Step 4: Compute Low-D Similarities (Student t-distribution)**

```
Student t-distribution with 1 DOF:
  Q(i,j) = (1 + ||yi - yj||²)⁻¹ / Σ(1 + ||yk - yl||²)⁻¹

This has heavier tails than Gaussian, which helps prevent crowding
```

**Step 5: Optimize via Gradient Descent**

```
Goal: Make Q (low-D probabilities) match P (high-D probabilities)

KL Divergence: 
  KL(P||Q) = Σ P_ij log(P_ij/Q_ij)
  
Gradient tells us how to move each point:
  - Attractive force: Pull together points that should be close (high P_ij)
  - Repulsive force: Push apart points that should be far (low P_ij)

After many iterations:
    C
         
  E   A  B    
         
       D

Clusters emerge! A and B stay close, C moves away, etc.
```

### Why Student t-distribution?

**The Crowding Problem:**

```
High dimensions → Low dimensions
  Many points at distance 5 → Cannot all be at distance 5 in 2D!
  
Solution: Use Student t-distribution
  - Heavier tails than Gaussian
  - Allows moderate distances in high-D to become larger distances in low-D
  - Points can spread out without losing similarity structure
```

**Visual Comparison:**

```
Gaussian (in high-D):
  Quick decay, most probability near center
  |
  |      ___
  |   __/   \__
  |__/         \__
  |________________

Student t (in low-D):
  Slower decay, heavier tails
  |
  |    _____
  |   /     \
  |  /       \____
  |_/____________
  
Heavier tails allow more distance variation in low-D
```

---

## The Mathematical Foundation

### 1. High-Dimensional Similarities (Gaussian)

For each point i, we define conditional probability that i would pick j as neighbor:

```
p(j|i) = exp(-||xi - xj||² / (2σi²)) / Σ(k≠i) exp(-||xi - xk||² / (2σi²))
```

**Components:**
- `||xi - xj||²`: Squared Euclidean distance between points i and j
- `σi`: Bandwidth (variance) for point i, adapted based on perplexity
- Denominator: Normalization to make it a probability distribution

**Example:**
```
Point A at [1, 2, 3], Point B at [1.5, 2.2, 3.1]
Distance² = (1-1.5)² + (2-2.2)² + (3-3.1)² = 0.25 + 0.04 + 0.01 = 0.30

With σA = 1.0:
  p(B|A) = exp(-0.30/(2×1.0²)) / Σk exp(-||xA - xk||²/(2×1.0²))
         = exp(-0.15) / Z
         = 0.86 / Z
         
High probability because B is close to A
```

### 2. Perplexity and Entropy

Perplexity determines the effective number of neighbors:

```
Perplexity(Pi) = 2^(H(Pi))

where H(Pi) = -Σj p(j|i) log₂(p(j|i))  (Shannon entropy)
```

**Interpretation:**
```
Perplexity = 30 means:
  "Consider roughly 30 nearest neighbors for each point"
  
Perplexity controls σi:
  - For dense regions: smaller σ (local focus)
  - For sparse regions: larger σ (look further for neighbors)
```

**Example:**
```
Uniform distribution over k items:
  p(j) = 1/k for all j
  H = -Σ(1/k)×log₂(1/k) = log₂(k)
  Perplexity = 2^(log₂(k)) = k

So perplexity ≈ effective number of neighbors
```

### 3. Symmetrized Probabilities

To make the similarity matrix symmetric:

```
pij = (p(j|i) + p(i|j)) / (2n)
```

where n is the number of points.

**Why symmetrize?**
- Conditional probabilities p(j|i) and p(i|j) may differ
- Joint probabilities are easier to work with
- Ensures consistent similarity relationships

### 4. Low-Dimensional Similarities (Student t)

In the embedded space, we use Student t-distribution with 1 degree of freedom:

```
qij = (1 + ||yi - yj||²)⁻¹ / Σ(k≠l) (1 + ||yk - yl||²)⁻¹
```

**Why Student t?**
- Heavier tails than Gaussian
- Allows dissimilar points to be far apart
- Prevents crowding of moderate distances

**Example:**
```
Points Y1 = [0, 0], Y2 = [2, 1]
Distance² = 2² + 1² = 5

qij = (1 + 5)⁻¹ / Z
    = 0.167 / Z
    
Compare to Gaussian: exp(-5/2) / Z = 0.082 / Z
Student t assigns higher probability (heavier tail)
```

### 5. Cost Function: KL Divergence

The optimization minimizes Kullback-Leibler divergence:

```
KL(P||Q) = Σi Σj pij log(pij/qij)
```

**Interpretation:**
- Measures how different Q is from P
- Asymmetric: KL(P||Q) ≠ KL(Q||P)
- Minimum (0) when P = Q
- Larger values = worse match

**Components:**
```
When pij is large (points should be similar):
  If qij is small (points far apart): large penalty log(pij/qij)
  If qij is large (points close): small penalty
  
When pij is small (points should be dissimilar):
  Contributes little to cost (pij × log(...) ≈ 0)
  
Effect: Focuses on keeping similar points together
```

### 6. Gradient Computation

The gradient with respect to low-dimensional coordinates:

```
∂(KL(P||Q))/∂yi = 4 Σj (pij - qij)(yi - yj)(1 + ||yi - yj||²)⁻¹
```

**Physical Interpretation:**

```
For each point i, the gradient has two forces:

Attractive Force (pij > qij):
  - Points that should be close but are far
  - Gradient points from yi toward yj
  - Strength ∝ (pij - qij)
  
Repulsive Force (qij > pij):
  - Points that are close but should be far
  - Gradient points from yj away from yi
  - Strength ∝ (qij - pij)
  
The (1 + ||yi - yj||²)⁻¹ term:
  - Moderates force based on current distance
  - Prevents very large updates for distant points
```

**Example:**
```
Point A at [0, 0], Point B at [1, 1]
pAB = 0.3 (should be similar)
qAB = 0.1 (currently far)

Difference: 0.3 - 0.1 = 0.2 (positive → attractive)

Distance² = 2
Force = 4 × 0.2 × ([0,0] - [1,1]) × (1 + 2)⁻¹
      = 0.8 × [-1,-1] × 0.33
      = [-0.27, -0.27]
      
Gradient tells A to move toward B (negative direction toward [1,1])
```

### 7. Optimization with Momentum

To speed up convergence and avoid local minima:

```
Yt+1 = Yt + α × ΔYt + momentum × (Yt - Yt-1)
```

**Parameters:**
- `α`: Learning rate (how far to move)
- `momentum`: Fraction of previous velocity to keep (typically 0.5 → 0.8)

**Effect:**
```
Without momentum:
  - Can get stuck in local minima
  - Oscillates in narrow valleys
  
With momentum:
  - Builds up speed in consistent directions
  - Dampens oscillations
  - Escapes shallow local minima
```

### 8. Early Exaggeration

In initial iterations, multiply P by a factor (typically 12):

```
P_early = early_exaggeration × P
```

**Why?**
```
Creates tight clusters initially:
  - High P values → strong attractive forces
  - Points form dense clusters
  - Easier to separate clusters later
  
Without early exaggeration:
  - Clusters may overlap from start
  - Harder to untangle later
  
Timeline:
  Iterations 0-250: P × 12 (tight clusters form)
  Iterations 250+:  P × 1  (clusters adjust and spread)
```

---

## Implementation Details

Our implementation includes the following key components:

### Class Structure

```python
class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, random_state=None, early_exaggeration=12.0,
                 early_exaggeration_iter=250, min_grad_norm=1e-7, verbose=0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        # ... other parameters
```

### Core Methods

1. **`__init__(...)`** - Initialize model
   - Set all hyperparameters
   - Typical defaults work well for most datasets

2. **`_compute_pairwise_distances(X)`** - Distance computation
   - Efficiently compute all pairwise squared Euclidean distances
   - Uses matrix operations for speed

3. **`_compute_joint_probabilities(distances, target_perplexity)`** - High-D similarities
   - For each point, binary search for optimal σ
   - Compute Gaussian similarities
   - Symmetrize the probability matrix

4. **`_compute_low_dim_affinities(Y)`** - Low-D similarities
   - Compute distances in embedded space
   - Apply Student t-distribution
   - Normalize to probabilities

5. **`_compute_gradient(P, Q, Y)`** - Gradient calculation
   - Compute attractive and repulsive forces
   - Return gradient for all points

6. **`_compute_kl_divergence(P, Q)`** - Cost function
   - Measure how well Q matches P
   - Lower is better

7. **`fit_transform(X)`** - Main algorithm
   - Compute high-D probabilities
   - Initialize embedding
   - Run gradient descent optimization
   - Return final embedding

8. **`fit(X)`** - Fit interface
   - Calls fit_transform
   - Stores embedding in self.embedding_

---

## Step-by-Step Example

Let's walk through a complete example of **visualizing handwritten digits**:

### The Data

```python
from sklearn.datasets import load_digits
import numpy as np

# Load digits dataset
digits = load_digits()
X = digits.data   # 1797 samples, 64 features (8×8 pixel images)
y = digits.target # Labels (0-9)

# Each sample is an 8×8 grayscale image flattened to 64 features
# Sample: [0, 0, 5, 13, 9, ..., 0, 0] represents pixel intensities
```

### Applying t-SNE

```python
from tsne import TSNE

# Create t-SNE model
tsne = TSNE(
    n_components=2,      # Map to 2D
    perplexity=30,       # Consider ~30 neighbors
    learning_rate=200,   # Step size for optimization
    n_iter=1000,         # 1000 gradient descent iterations
    random_state=42,     # For reproducibility
    verbose=1            # Show progress
)

# Fit and transform
X_embedded = tsne.fit_transform(X)
# Output: (1797, 2) - 2D coordinates for each digit
```

**What happens internally:**

**Iteration 0-50:**
```
[t-SNE] Computing pairwise distances...
  - Calculate 1797×1797 distance matrix
  - Takes a moment for large datasets

[t-SNE] Computing P-values...
  - For each of 1797 points:
    - Binary search for σi (50 iterations max)
    - Target perplexity = 30
  - Symmetrize probabilities
  - Result: 1797×1797 probability matrix

[t-SNE] Starting optimization...
Iteration 50/1000, KL divergence: 2.3456, Gradient norm: 15.234
  - Early exaggeration phase (P × 12)
  - Tight clusters forming
```

**Iteration 250-500:**
```
Iteration 250/1000, KL divergence: 1.8723, Gradient norm: 8.456
  - Early exaggeration ends
  - Clusters adjusting positions
  - KL divergence decreasing

Iteration 500/1000, KL divergence: 1.2341, Gradient norm: 2.123
  - Refinement phase
  - Fine-tuning cluster boundaries
```

**Iteration 750-1000:**
```
Iteration 1000/1000, KL divergence: 0.9876, Gradient norm: 0.523
[t-SNE] Optimization finished!
[t-SNE] Final KL divergence: 0.9876
```

### Visualizing Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE visualization of Handwritten Digits')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

**What you see:**

```
Visual Result:
          
    ⑧ ⑧      ⓪ ⓪
   ⑧ ⑧ ⑧    ⓪ ⓪ ⓪
              
  ⑨ ⑨         ① ①
 ⑨ ⑨ ⑨      ① ① ①
              
   ⑦ ⑦       ② ②
  ⑦ ⑦       ② ② ②
           
    ⑥        ③ ③
   ⑥ ⑥      ③ ③
              
   ⑤ ⑤       ④ ④
  ⑤ ⑤       ④ ④ ④

10 distinct clusters, one for each digit!
```

**Interpretation:**
- **Cluster separation**: Digits 0, 1 are very distinct
- **Overlap regions**: 3 and 5 might be close (similar shapes)
- **Outliers**: Misclassified or ambiguous digits
- **Distance between clusters**: Not meaningful in t-SNE!
  - Only local structure (within clusters) is preserved
  - Inter-cluster distances can be arbitrary

### Analyzing Parameter Effects

**Perplexity Comparison:**

```python
perplexities = [5, 30, 50]

Perplexity = 5 (very local):
    ⓪  ①
  ⑧  ②     ③   ④
   ⑨  ⑦  ⑥  ⑤
   
Many small, tight clusters
Over-fragments the data

Perplexity = 30 (balanced):
    ⓪ ⓪    ① ①
   ⑧ ⑧    ② ②
  ⑨ ⑨    ③ ③
   ⑦ ⑦   ④ ④
  ⑥ ⑥   ⑤ ⑤
  
Clear clusters, good separation
BEST CHOICE

Perplexity = 50 (global):
    ⓪ ①
   ⑧ ②
  ⑨ ⑦ ③
   ⑥ ⑤ ④
   
Clusters merge, less separated
More global structure
```

---

## Real-World Applications

### 1. **Image Analysis & Computer Vision**
Visualizing high-dimensional image features:
- Input: Images or CNN features (1000+ dimensions)
- Output: 2D scatter plot revealing image clusters
- Example: Group similar products, detect image duplicates
- **Business Value**: Visual search, quality control, dataset exploration

**Specific Applications:**
```
Fashion E-commerce:
  Extract features from product images
  → Apply t-SNE
  → See which products look similar
  → Improve recommendation system
  
Medical Imaging:
  Tumor scan features (shape, texture, intensity)
  → t-SNE visualization
  → Identify tumor subtypes
  → Aid diagnosis and treatment planning
```

### 2. **Natural Language Processing**
Visualizing word embeddings and document representations:
- Input: Word2Vec, GloVe, BERT embeddings (300+ dimensions)
- Output: Word relationship maps
- Example: "king" - "man" + "woman" ≈ "queen" visualized
- **Business Value**: Understanding semantic relationships, debugging NLP models

**Example:**
```
Word Categories in Embedding Space:

    animals          food
   🐕 🐈 🐎      🍎 🍊 🥖
     🐘             🍕 🍔
     
         countries        sports
        🇺🇸 🇫🇷 🇯🇵      ⚽ 🏀 🎾
         🇬🇧 🇩🇪         ⚾ 🏈

Semantic relationships preserved!
```

### 3. **Single Cell Biology**
Analyzing gene expression data:
- Input: Gene expression profiles (10,000+ genes per cell)
- Output: Cell type clusters
- Example: Identify rare cell populations, disease signatures
- **Business Value**: Drug discovery, disease understanding

**Example:**
```
Cell Types from Expression:

  T-cells          B-cells
   🔴 🔴          🔵 🔵
  🔴 🔴 🔴      🔵 🔵 🔵
  
         Stem cells
          🟢 🟢
         🟢 🟢 🟢
         
    Macrophages    Neurons
     🟡 🟡        🟣 🟣
    🟡 🟡 🟡      🟣 🟣

Different cell types cluster by expression profile
```

### 4. **Recommender Systems**
Understanding user-item relationships:
- Input: User embeddings, item embeddings
- Output: Visual map of preferences
- Example: Netflix - see which movies are similar
- **Business Value**: Better recommendations, content organization

**Applications:**
```
Movie Recommendations:
  
  Action        SciFi
   🎬 🎬        🚀 🚀
  🎬 🎬 🎬    🚀 🚀 🚀
  
       Drama        Comedy
      😢 😢        😂 😂
     😢 😢 😢      😂 😂
     
User A likes action → recommend nearby movies
```

### 5. **Anomaly Detection**
Visualizing normal vs anomalous patterns:
- Input: Transaction features, log features, sensor data
- Output: Outlier detection in 2D
- Example: Fraud detection, equipment failure prediction
- **Business Value**: Early warning systems, security

**Example:**
```
Network Traffic Patterns:

Normal traffic (dense cluster):
  ✓ ✓ ✓ ✓
 ✓ ✓ ✓ ✓ ✓
  ✓ ✓ ✓
  
Anomalies (far from cluster):
        ⚠ 
           ⚠
  ⚠
  
Outliers may indicate attacks or failures
```

### 6. **Neural Network Interpretation**
Understanding what networks learn:
- Input: Activations from hidden layers
- Output: Feature visualization
- Example: See how CNN filters group images
- **Business Value**: Model debugging, interpretability, trust

**Example:**
```
CNN Layer 5 Activations:

  Dog faces    Cat faces
   🐕 🐕        🐈 🐈
  🐕 🐕 🐕    🐈 🐈 🐈
  
      Car fronts    Car sides
       🚗 🚗        🚙 🚙
      🚗 🚗 🚗      🚙 🚙

Network learns meaningful features!
```

### 7. **Customer Segmentation**
Visualizing customer groups:
- Input: Customer features (purchases, demographics, behavior)
- Output: Customer segments
- Example: Identify distinct shopper types
- **Business Value**: Targeted marketing, personalization

**Example:**
```
Customer Segments:

  Budget          Premium
 shoppers        shoppers
   💰 💰          💎 💎
  💰 💰 💰      💎 💎 💎
  
      Seasonal      Loyal
      buyers       customers
       🎁 🎁        ⭐ ⭐
      🎁 🎁 🎁      ⭐ ⭐

Different segments need different strategies
```

---

## Understanding the Code

Let's break down the key parts of our implementation:

### 1. Computing Pairwise Distances

```python
def _compute_pairwise_distances(self, X):
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
    D = np.maximum(D, 0)
    return D
```

**How it works:**
```python
# Efficient computation of ||xi - xj||²
# Expansion: ||xi - xj||² = ||xi||² + ||xj||² - 2xi·xj

X = [[1, 2],    # Point 1
     [3, 4]]    # Point 2

sum_X = [1² + 2², 3² + 4²] = [5, 25]

# Broadcasting magic:
sum_X[:, newaxis]:     sum_X[newaxis, :]:
  [[5],                [[5, 25]]
   [25]]

# Addition broadcasts to:
  [[5+5,  5+25],
   [25+5, 25+25]]
  
# Subtract 2*dot product:
dot(X, X.T) = [[5,  11],
               [11, 25]]

D = [[10-10,  30-22],    [[0, 8],
     [30-22,  50-50]]  =  [8, 0]]

Distance from point 1 to 2: √8 ≈ 2.83
```

**Why this trick?**
- Avoid explicit loops over all pairs
- Uses optimized BLAS operations
- O(n²d) instead of O(n²d) with loops (same complexity but much faster)

### 2. Computing Joint Probabilities (Perplexity)

```python
def _compute_joint_probabilities(self, distances, target_perplexity):
    n = distances.shape[0]
    P = np.zeros((n, n))
    target_entropy = np.log(target_perplexity)
    
    for i in range(n):
        # Binary search for optimal beta (precision)
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0
        
        Di = distances[i, ...]  # Distances from point i
        
        for _ in range(50):
            P_i = np.exp(-Di * beta)
            sum_P_i = np.sum(P_i)
            P_i = P_i / sum_P_i
            
            entropy = -np.sum(P_i * np.log2(P_i + 1e-8))
            entropy_diff = entropy - target_entropy
            
            if abs(entropy_diff) < 1e-5:
                break
            
            # Adjust beta
            if entropy_diff > 0:
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2
        
        P[i, :] = P_i
    
    # Symmetrize
    P = (P + P.T) / (2 * n)
    return P
```

**Step-by-step example:**
```python
Point i, distances to 4 others: [1.0, 2.0, 3.0, 10.0]
Target perplexity: 3.0
Target entropy: log(3.0) = 1.585

Iteration 1: beta = 1.0
  P_i = exp(-[1, 2, 3, 10] * 1.0) = [0.368, 0.135, 0.050, 0.000]
  Normalize: [0.665, 0.244, 0.090, 0.001]
  Entropy: -sum(p × log2(p)) = 1.234
  Too low! Need higher entropy → decrease beta
  
Iteration 2: beta = 0.5
  P_i = exp(-[1, 2, 3, 10] * 0.5) = [0.607, 0.368, 0.223, 0.007]
  Normalize: [0.503, 0.305, 0.185, 0.006]
  Entropy: 1.502
  Still too low → decrease beta more
  
Iteration 3: beta = 0.25
  ...
  Entropy: 1.580 ≈ 1.585 ✓
  
This beta gives perplexity ≈ 3.0
```

### 3. Computing Low-Dimensional Affinities

```python
def _compute_low_dim_affinities(self, Y):
    distances = self._compute_pairwise_distances(Y)
    Q = 1 / (1 + distances)
    np.fill_diagonal(Q, 0)
    sum_Q = np.sum(Q)
    Q = Q / sum_Q
    Q = np.maximum(Q, 1e-12)
    return Q
```

**Example:**
```python
Y = [[0, 0],     # Point 1
     [1, 1],     # Point 2
     [5, 5]]     # Point 3

Distances²:
  [[0, 2, 50],
   [2, 0, 32],
   [50, 32, 0]]

Student t-distribution:
  Q_unnorm = 1 / (1 + distances²)
           = [[inf, 0.333, 0.020],
              [0.333, inf, 0.031],
              [0.020, 0.031, inf]]
  
  Set diagonal to 0:
           = [[0, 0.333, 0.020],
              [0.333, 0, 0.031],
              [0.020, 0.031, 0]]
  
  Normalize (sum = 0.768):
  Q = [[0, 0.434, 0.026],
       [0.434, 0, 0.040],
       [0.026, 0.040, 0]]

High Q (0.434) for nearby points (1-2)
Low Q (0.026) for distant points (1-3)
```

### 4. Computing Gradient

```python
def _compute_gradient(self, P, Q, Y):
    n = Y.shape[0]
    Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    distances = self._compute_pairwise_distances(Y)
    inv_distances = 1 / (1 + distances)
    np.fill_diagonal(inv_distances, 0)
    
    PQ_diff = P - Q
    gradient = 4 * np.sum((PQ_diff[:, :, np.newaxis] * 
                           Y_diff * 
                           inv_distances[:, :, np.newaxis]), axis=1)
    return gradient
```

**Example:**
```python
3 points in 2D:
Y = [[0, 0],    P = [[0,   0.4, 0.1],    Q = [[0,   0.3, 0.05],
     [1, 0],         [0.4, 0,   0.2],         [0.3, 0,   0.15],
     [0, 1]]         [0.1, 0.2, 0  ]]         [0.05, 0.15, 0   ]]

For point 0:
  PQ_diff[0,:] = [0, 0.1, 0.05]  # Should move closer to 1 and 2
  
  Y_diff[0,:,:] = [[0,0], [0-1,0-0], [0-0,0-1]]
                = [[0,0], [-1,0], [0,-1]]
  
  distances[0,:] = [0, 1, 1]
  inv_dist[0,:] = [0, 0.5, 0.5]
  
  Gradient contribution from point 1:
    4 × 0.1 × [-1,0] × 0.5 = [-0.2, 0]  (pull toward point 1)
  
  Gradient contribution from point 2:
    4 × 0.05 × [0,-1] × 0.5 = [0, -0.1]  (pull toward point 2)
  
  Total gradient[0] = [-0.2, -0.1]
  
Update: Y[0] = Y[0] - learning_rate × gradient[0]
             = [0,0] - 200 × [-0.2, -0.1]
             = [40, 20]  (moves toward 1 and 2)
```

### 5. Main Optimization Loop

```python
def fit_transform(self, X):
    # Setup
    distances = self._compute_pairwise_distances(X)
    P = self._compute_joint_probabilities(distances, self.perplexity)
    Y = np.random.randn(n_samples, self.n_components) * 1e-4
    Y_velocity = np.zeros_like(Y)
    
    # Optimization
    for iteration in range(self.n_iter):
        # Early exaggeration
        if iteration < self.early_exaggeration_iter:
            P_effective = P * self.early_exaggeration
        else:
            P_effective = P
        
        # Compute Q and gradient
        Q = self._compute_low_dim_affinities(Y)
        gradient = self._compute_gradient(P_effective, Q, Y)
        
        # Check convergence
        if np.linalg.norm(gradient) < self.min_grad_norm:
            break
        
        # Momentum update
        momentum = 0.5 if iteration < 250 else 0.8
        Y_velocity = momentum * Y_velocity - self.learning_rate * gradient
        Y = Y + Y_velocity
        
        # Recenter
        Y = Y - np.mean(Y, axis=0)
    
    return Y
```

**Optimization trace:**
```
Iteration 0:
  P: High-D similarities (fixed)
  Y: Random [-0.0001, 0.0001] (initialization)
  Q: Almost uniform (random positions)
  Gradient: Large (big mismatch between P and Q)
  
Iteration 50 (early exaggeration):
  P_effective: P × 12 (exaggerated)
  Y: Points moving into rough clusters
  Q: Starting to match P
  Gradient: Still large but decreasing
  
Iteration 250 (exaggeration ends):
  P_effective: P × 1 (normal)
  Y: Clusters formed, need refinement
  Q: Better match to P
  Gradient: Moderate
  
Iteration 1000 (final):
  Y: Well-separated clusters
  Q: Close match to P
  Gradient: Small (converged)
  KL divergence: ~1.0 (good)
```

---

## Model Evaluation

### Hyperparameter Selection

t-SNE has several important hyperparameters:

#### 1. Perplexity

```
Range: 5 to 50 (typically)

Low Perplexity (5-15):
  ✓ Very local structure
  ✓ Fine-grained clusters
  ✗ May over-fragment
  ✗ Sensitive to noise
  
Medium Perplexity (25-35):
  ✓ Balanced local/global
  ✓ Robust default choice
  ✓ Works for most datasets
  
High Perplexity (40-50):
  ✓ More global structure
  ✓ Broader patterns
  ✗ May merge distinct clusters
  ✗ Slower computation
```

**Rule of thumb:**
```
Perplexity should be less than n_samples / 3

For different dataset sizes:
  100 samples: perplexity = 10-20
  1,000 samples: perplexity = 20-40
  10,000 samples: perplexity = 30-50

When in doubt: Start with 30
```

#### 2. Learning Rate

```
Range: 10 to 1000

Low Learning Rate (10-100):
  ✓ Stable optimization
  ✓ Less likely to diverge
  ✗ Slow convergence
  ✗ May get stuck in local minima
  
Medium Learning Rate (150-250):
  ✓ Good balance
  ✓ Reasonable convergence speed
  ✓ Default choice: 200
  
High Learning Rate (500-1000):
  ✓ Fast convergence
  ✗ May overshoot
  ✗ Unstable, points bounce around
  ✗ Worse final quality
```

**Guidelines:**
```
If optimization looks unstable:
  → Decrease learning rate
  
If converging too slowly:
  → Increase learning rate
  
If you see "ball" shape (all points in circle):
  → Learning rate too high OR not enough iterations
```

#### 3. Number of Iterations

```
Minimum: 250 (for early exaggeration)
Typical: 1000
High quality: 2000-5000

Trade-off:
  More iterations → Better convergence, slower
  Fewer iterations → Faster, may not converge
  
Convergence indicators:
  ✓ KL divergence stops decreasing
  ✓ Gradient norm very small
  ✓ Visual appearance stops changing
```

### Quality Metrics

#### 1. KL Divergence

```
KL(P||Q) = Σij Pij log(Pij / Qij)

Interpretation:
  KL = 0.5-1.0:  Excellent match
  KL = 1.0-2.0:  Good match
  KL = 2.0-3.0:  Acceptable
  KL > 3.0:      Poor, increase iterations
  
Note: KL values not comparable across different datasets!
```

#### 2. Visual Inspection

```
Good t-SNE visualization:
  ✓ Clear cluster separation
  ✓ Clusters have consistent density
  ✓ Similar points (same class) grouped
  ✓ Different classes separated
  
Warning signs:
  ✗ All points in a ball (increase iterations/decrease lr)
  ✗ Severe crowding (try different perplexity)
  ✗ Expected clusters merged (increase perplexity)
  ✗ Over-fragmented (decrease perplexity)
```

#### 3. Cluster Preservation

```
Compare with known labels (if available):

from sklearn.metrics import silhouette_score

# Silhouette score measures cluster quality
score = silhouette_score(X_embedded, y)

Score interpretation:
  0.7-1.0: Strong, well-separated clusters
  0.5-0.7: Reasonable structure
  0.25-0.5: Weak structure, overlapping clusters
  < 0.25:  No meaningful clustering

Note: This only makes sense if data truly has clusters!
```

### Common Issues and Solutions

#### Issue 1: Crowding (Dense Ball)

```
Symptom: All points clustered in a tight ball

Causes:
  - Learning rate too high
  - Not enough iterations
  - Perplexity too high
  
Solutions:
  → Reduce learning rate (e.g., 200 → 50)
  → Increase iterations (e.g., 1000 → 2000)
  → Reduce perplexity (e.g., 50 → 30)
```

#### Issue 2: Over-fragmentation

```
Symptom: Too many small clusters, expected groups split

Cause: Perplexity too low

Solution:
  → Increase perplexity (e.g., 5 → 30)
```

#### Issue 3: Merged Clusters

```
Symptom: Distinct groups merged together

Causes:
  - Perplexity too high
  - Not enough iterations
  
Solutions:
  → Decrease perplexity
  → Increase iterations
  → Try different random initialization
```

#### Issue 4: Different Results Each Run

```
Symptom: Results look different every time

Cause: Random initialization

Solutions:
  ✓ Set random_state for reproducibility
  ✓ Run multiple times, choose best (lowest KL divergence)
  ✓ This is normal! t-SNE is stochastic
```

### Best Practices

1. **Preprocessing:**
```python
# Center and normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Or use PCA first (for very high dimensions)
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
X_embedded = tsne.fit_transform(X_reduced)
```

2. **Multiple Runs:**
```python
# Run several times with different random seeds
best_kl = np.inf
best_embedding = None

for seed in range(5):
    tsne = TSNE(random_state=seed, verbose=0)
    embedding = tsne.fit_transform(X)
    
    if tsne.kl_divergence_ < best_kl:
        best_kl = tsne.kl_divergence_
        best_embedding = embedding

# Use best_embedding
```

3. **Parameter Sweep:**
```python
# Try different perplexity values
for perplexity in [10, 20, 30, 40, 50]:
    tsne = TSNE(perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)
    # Visualize and compare
```

---

## Comparing with Other Methods

### t-SNE vs PCA

```
PCA (Principal Component Analysis):
  ✓ Fast (closed-form solution)
  ✓ Interpretable components
  ✓ Preserves global structure
  ✗ Linear, misses non-linear patterns
  ✗ Poor at revealing clusters
  
t-SNE:
  ✓ Reveals non-linear patterns
  ✓ Excellent cluster visualization
  ✓ Preserves local structure
  ✗ Slow (iterative optimization)
  ✗ Components not interpretable
  ✗ Different each run

When to use:
  PCA: Quick exploration, linear relationships
  t-SNE: Cluster visualization, non-linear data
```

### t-SNE vs UMAP

```
UMAP (Uniform Manifold Approximation and Projection):
  ✓ Much faster than t-SNE
  ✓ Preserves more global structure
  ✓ Better for large datasets
  ✗ More hyperparameters
  ✗ Less mature/tested
  
t-SNE:
  ✓ Better local structure preservation
  ✓ More established, well-understood
  ✓ Extensive research on behavior
  ✗ Slower
  ✗ Less global structure

When to use:
  UMAP: Large datasets (>10k), need speed
  t-SNE: Moderate datasets, best cluster separation
```

### t-SNE vs MDS

```
MDS (Multidimensional Scaling):
  ✓ Preserves all pairwise distances
  ✓ Global distance relationships
  ✗ Computationally expensive
  ✗ Less flexible than t-SNE
  
t-SNE:
  ✓ Focuses on important (local) distances
  ✓ Better visual separation
  ✗ Distances between clusters meaningless

When to use:
  MDS: Need accurate distance preservation
  t-SNE: Need cluster visualization
```

---

## Computational Complexity

### Time Complexity

```
Standard Algorithm:
  - Distance computation: O(n² × d)
  - P computation: O(n² × log(perplexity))
  - Each gradient descent iteration: O(n²)
  - Total: O(n² × d + n² × iterations)
  
For n=10,000, d=100, iterations=1000:
  ~10¹⁰ operations ≈ several minutes to hours

Practical limits:
  - Up to ~10,000 points: Feasible
  - 10,000-50,000: Slow but possible
  - >50,000: Need approximations (Barnes-Hut, FFT)
```

### Space Complexity

```
Storage required:
  - Input data: O(n × d)
  - Distance matrix: O(n²)
  - Probability matrices P, Q: O(n²)
  - Embedding: O(n × n_components)
  
Total: O(n² + n×d)

For n=10,000:
  - P matrix: 10,000² × 8 bytes ≈ 800 MB
  - Feasible on modern computers
  
For n=100,000:
  - P matrix: 100,000² × 8 bytes ≈ 80 GB
  - Need sparse approximations
```

### Optimization Strategies

1. **PCA Preprocessing**
```python
# Reduce to ~50 dimensions first
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
X_embedded = tsne.fit_transform(X_reduced)

Benefits:
  - Faster distance computation
  - Removes noise
  - Often improves results
```

2. **Barnes-Hut Approximation**
```
Not implemented in our version, but:
  - Uses spatial indexing (quadtree/octree)
  - Approximates far-away interactions
  - Reduces complexity to O(n log n)
  - Enables 50,000+ points
```

3. **Mini-batch Approach**
```python
# For very large datasets
# Train on sample, embed rest
n_sample = 5000
sample_idx = np.random.choice(len(X), n_sample, replace=False)
X_sample = X[sample_idx]

tsne = TSNE()
X_sample_embedded = tsne.fit_transform(X_sample)

# Then embed remaining points (out of scope for this implementation)
```

---

## Advantages and Limitations

### Advantages ✅

1. **Excellent Visualization**
   - Creates beautiful, interpretable plots
   - Reveals cluster structure clearly
   - Non-linear patterns visible

2. **Preserves Local Structure**
   - Similar points stay close together
   - Captures manifold structure
   - Better than PCA for complex data

3. **Flexible**
   - Works with any distance metric
   - No assumptions about data distribution
   - Handles non-linear relationships

4. **Well-Established**
   - Extensive research and validation
   - Well-understood behavior
   - Many successful applications

5. **Unsupervised**
   - No labels needed
   - Exploratory analysis tool
   - Discovers hidden patterns

### Limitations ❌

1. **Computationally Expensive**
   ```
   O(n²) complexity:
     - Slow for large datasets (>10k points)
     - Minutes to hours for moderate datasets
     - Not suitable for real-time applications
   ```

2. **Non-Deterministic**
   ```
   Different runs give different results:
     - Random initialization
     - Non-convex optimization
     - Solution: Set random_state, run multiple times
   ```

3. **Hyperparameter Sensitive**
   ```
   Results depend on parameters:
     - Perplexity significantly affects output
     - Learning rate affects convergence
     - No automatic way to choose
     - Requires experimentation
   ```

4. **Global Structure Not Preserved**
   ```
   Distances between clusters are meaningless:
     - Cannot interpret inter-cluster distances
     - Cannot compare sizes of clusters
     - Only local neighborhoods are meaningful
   ```

5. **Curse of Dimensionality**
   ```
   For very high dimensions:
     - All distances become similar
     - Harder to preserve structure
     - May need PCA preprocessing
   ```

6. **Cannot Embed New Points**
   ```
   No transform() method:
     - Must retrain for new data
     - Expensive for incrementing updates
     - Unlike PCA which has clear projection
   ```

### When to Use t-SNE

**Good Use Cases:**
- ✅ Visualizing high-dimensional data (images, embeddings)
- ✅ Exploring cluster structure
- ✅ Comparing different datasets visually
- ✅ Understanding neural network representations
- ✅ Exploratory data analysis
- ✅ Presentation and communication

**Bad Use Cases:**
- ❌ Feature extraction for downstream tasks → Use PCA or autoencoders
- ❌ Embedding new/unseen data → Use parametric methods
- ❌ Very large datasets (>50k) → Use UMAP or approximations
- ❌ Measuring exact distances → Use MDS
- ❌ Need deterministic results → Use PCA
- ❌ Real-time applications → Use PCA or random projections

---

## Key Concepts to Remember

### 1. **Local vs Global Structure**
t-SNE preserves local structure (nearby points) but not global structure (far points). Cluster positions and distances between clusters are not meaningful.

### 2. **The Perplexity Balance**
Perplexity controls the local/global trade-off:
- Low perplexity: Very local, fine details
- High perplexity: More global, broader patterns
- Default 30 works for most cases

### 3. **Student t-Distribution is Key**
Using Student t in low-D (instead of Gaussian) prevents crowding:
- Heavier tails allow moderate distances to spread out
- Points can separate without losing similarities

### 4. **Optimization is Stochastic**
Results vary each run due to:
- Random initialization
- Non-convex optimization
- Solution: Run multiple times, use random_state

### 5. **Not a Dimension Reduction for ML**
t-SNE is for visualization, not feature extraction:
- Cannot embed new points
- Components not interpretable
- Use PCA/autoencoders for feature extraction

### 6. **Visual Interpretation**

```
What you CAN interpret:
  ✓ Which points cluster together
  ✓ Relative densities within clusters
  ✓ Outliers within clusters
  
What you CANNOT interpret:
  ✗ Distances between clusters
  ✗ Cluster sizes (can be distorted)
  ✗ Orientation/rotation of plot
  ✗ Exact positions (vary each run)
```

---

## Conclusion

t-SNE is a powerful tool for visualizing and exploring high-dimensional data! By understanding:
- How it converts distances to probabilities
- The role of perplexity in balancing local/global structure
- Why Student t-distribution prevents crowding
- How gradient descent optimizes the embedding
- Best practices for hyperparameter selection

You've gained an essential technique for making sense of complex, high-dimensional datasets! 🎨📊

**When to Use t-SNE:**
- ✅ Visualizing complex datasets
- ✅ Exploring cluster structure
- ✅ Understanding embeddings and features
- ✅ Communicating patterns to stakeholders
- ✅ Moderate-sized datasets (<10k points)

**When to Use Something Else:**
- ❌ Need fast results → PCA, random projection
- ❌ Large datasets → UMAP, PCA
- ❌ Feature extraction → PCA, autoencoders
- ❌ Preserve exact distances → MDS
- ❌ Interpretable components → PCA, ICA

**Next Steps:**
- Try t-SNE on your own dataset
- Experiment with different perplexity values
- Compare with PCA to see the difference
- Learn about UMAP for faster alternative
- Study Barnes-Hut t-SNE for large datasets
- Explore parametric t-SNE for new point embedding

Happy visualizing! 💻🎨📈

