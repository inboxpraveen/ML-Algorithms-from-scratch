# Autoencoders from Scratch: A Comprehensive Guide

Welcome to the fascinating world of Autoencoders! 🧠 In this comprehensive guide, we'll explore autoencoders - neural networks that learn to compress and reconstruct data. Think of them as intelligent data compressors that learn the most important features automatically!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is an Autoencoder?](#what-is-an-autoencoder)
3. [How Autoencoders Work](#how-autoencoders-work)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Tips and Best Practices](#tips-and-best-practices)
11. [Comparison: Autoencoder vs Other Methods](#comparison-autoencoder-vs-other-methods)
12. [Simplification vs. canonical Autoencoders](#simplification-vs-canonical-autoencoders)
13. [Mathematical Intuition](#mathematical-intuition)
14. [Conclusion](#conclusion)
15. [Further Reading](#further-reading)

---

## Quick Start: Plug-and-Play Example

This is exactly the `if __name__ == "__main__":` block at the bottom of
`_24_autoencoders.py`. Run `python _24_autoencoders.py` and you get the output
below verbatim. Nothing but NumPy is required, and it finishes in a few seconds.

The first demo is also the *correctness proof* for this implementation: with
`activation='linear'` an autoencoder provably spans the same subspace as PCA, and
PCA's rank-3 reconstruction error is a floor no linear autoencoder can beat. We
compute that floor with a plain `np.linalg.svd` and check how close we got.

```python
import numpy as np

# ---- Paste the Autoencoder class here (from _24_autoencoders.py) ----
# class Autoencoder: ...

np.random.seed(42)

print("=" * 64)
print("AUTOENCODER FROM SCRATCH - PLUG-AND-PLAY DEMO")
print("=" * 64)

# ================================================================
# DEMO 1: a LINEAR autoencoder should rediscover PCA.
# This is the known-answer test for autoencoders: with no
# nonlinearity, the optimal encoder/decoder pair spans exactly the
# top-k principal subspace, so we can check the code against an
# answer we can compute in closed form with an SVD.
# ================================================================
print("\n" + "=" * 64)
print("DEMO 1 - A linear autoencoder rediscovers PCA")
print("=" * 64)
print("Data: 500 points that really live on a 3-D plane inside 20-D space")
print("      X = Z @ B + noise,  Z is 500x3,  B is 3x20")

Z = np.random.randn(500, 3)
B = np.random.randn(3, 20)
X = Z @ B + 0.05 * np.random.randn(500, 20)

# Shuffle BEFORE splitting so train and test come from the same region
idx = np.random.permutation(500)
X = X[idx]
X_train, X_test = X[:400], X[400:]      # 400 / 100, no overlap

ae = Autoencoder(
    input_dim=20,
    encoding_dim=3,
    activation='linear',    # linear bottleneck -> equivalent to PCA
    learning_rate=0.1,
    epochs=1500,
    batch_size=32,
    verbose=500
)
ae.fit(X_train)

print(f"\nLoss on the internal [0,1] scale: epoch 1 = "
      f"{ae.history['loss'][0]:.6f} -> epoch {ae.epochs} = {ae.history['loss'][-1]:.6f}")
print(f"Compression ratio        : {ae.get_compression_ratio():.1f}x (20 numbers -> 3)")
print(f"Train reconstruction MSE : {-ae.score(X_train):.6f}")
print(f"Test  reconstruction MSE : {-ae.score(X_test):.6f}")

# Known-answer check. fit() min-max scales X internally, so compare on
# that same [0,1] scale: PCA's rank-3 reconstruction is the floor no
# linear autoencoder can beat.
scale = ae.data_max - ae.data_min + 1e-8
Xn_train = (X_train - ae.data_min) / scale
Xn_test = (X_test - ae.data_min) / scale
mu = np.mean(Xn_train, axis=0)
_, _, Vt = np.linalg.svd(Xn_train - mu, full_matrices=False)
P = Vt[:3]                                        # top-3 principal directions
pca_mse = np.mean((Xn_test - ((Xn_test - mu) @ P.T @ P + mu)) ** 2)
ae_mse = np.mean((Xn_test - (ae.reconstruct(X_test) - ae.data_min) / scale) ** 2)

print("\nKnown-answer check (all on the internal [0,1] scale):")
print(f"  PCA(3) test MSE - the theoretical floor : {pca_mse:.3e}")
print(f"  Autoencoder test MSE                   : {ae_mse:.3e}")
print(f"  Ratio AE / PCA                         : {ae_mse / pca_mse:.3f}x")

# The decoder's rows span the subspace the reconstruction lives in.
# Principal angles near 0 degrees mean it found the PCA plane itself.
Qa, _ = np.linalg.qr(ae.weights[ae.n_layers - 1].T)
Qb, _ = np.linalg.qr(P.T)
cosines = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
print(f"  Angles between AE and PCA subspaces    : {np.round(angles, 3)} degrees")

print("\nSample reconstructions (first 5 of 20 features):")
recon = ae.reconstruct(X_test)
for i in range(3):
    print(f"  sample {i}  true : {np.round(X_test[i, :5], 3)}")
    print(f"            recon: {np.round(recon[i, :5], 3)}")

# ================================================================
# DEMO 2: anomaly detection done WITHOUT leakage.
# Three disjoint splits: fit on one, choose the threshold on a
# second, report metrics on a third.
# ================================================================
print("\n" + "=" * 64)
print("DEMO 2 - Anomaly detection with a held-out threshold")
print("=" * 64)
print("Train on normal rows only, pick the threshold on a CALIBRATION")
print("split, and report metrics on a third, never-seen EVAL split.")

# Normal behaviour lives on a 3-D manifold in 10-D; anomalies do not.
Zn = np.random.randn(700, 3)
X_normal = Zn @ np.random.randn(3, 10) + 0.2 * np.random.randn(700, 10)
X_anomaly = np.random.uniform(-5, 5, (50, 10))

X_fit = X_normal[:400]           # train
X_calib = X_normal[400:550]      # threshold selection only
X_eval_norm = X_normal[550:]     # scored, never seen before

det = Autoencoder(
    input_dim=10,
    encoding_dim=3,
    hidden_dims=[8, 5],
    activation='tanh',           # 'relu' here risks a dead bottleneck
    learning_rate=0.05,
    epochs=400,
    batch_size=32
)
det.fit(X_fit)

threshold = np.percentile(det.reconstruction_error(X_calib), 95)

X_scored = np.vstack([X_eval_norm, X_anomaly])
y_true = np.array([0] * len(X_eval_norm) + [1] * len(X_anomaly))
errors = det.reconstruction_error(X_scored)
y_pred = (errors > threshold).astype(int)

tp = int(np.sum((y_pred == 1) & (y_true == 1)))
fp = int(np.sum((y_pred == 1) & (y_true == 0)))
tn = int(np.sum((y_pred == 0) & (y_true == 0)))
fn = int(np.sum((y_pred == 0) & (y_true == 1)))
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"\nTrain reconstruction MSE : {-det.score(X_fit):.6f}")
print(f"Eval  reconstruction MSE : {-det.score(X_eval_norm):.6f}  (normal rows only)")
print(f"Mean error, normal rows  : {np.mean(errors[y_true == 0]):.6f}")
print(f"Mean error, anomaly rows : {np.mean(errors[y_true == 1]):.6f}")
print(f"Threshold (95th pct of calibration errors): {threshold:.6f}")
print(f"\nPrecision {precision:.2%}   Recall {recall:.2%}   F1 {f1:.4f}")
print("                Predicted Normal  Predicted Anomaly")
print(f"Actual Normal         {tn:5d}            {fp:5d}")
print(f"Actual Anomaly        {fn:5d}            {tp:5d}")
print("A 95th-percentile threshold accepts ~5% false positives by")
print("construction, so precision here is capped, not a model failure.")

print("\nSample scores (3 normal rows, then 3 anomalies):")
for i in list(range(3)) + list(range(len(X_eval_norm), len(X_eval_norm) + 3)):
    label = "normal " if y_true[i] == 0 else "anomaly"
    verdict = "FLAG" if y_pred[i] else "ok"
    print(f"  true={label}  error={errors[i]:9.6f}  -> {verdict}")

# ================================================================
# DEMO 3: the failure mode this implementation is most prone to.
# ================================================================
print("\n" + "=" * 64)
print("DEMO 3 - Dying ReLU: why the default can silently fail")
print("=" * 64)
print("Same 20-D data and budget as DEMO 1; only the activation changes.")
print("A 'dead' code unit has zero variance across the whole test set.")
print("hidden_dims=None here, so the bottleneck is the ONLY nonlinear")
print("layer - which is why row 2 and row 4 come out identical.")
print()
print(f"  {'setting':<30}{'dead units':<14}{'test MSE'}")
print("  " + "-" * 56)
for act, bneck, label in [('relu', None, "activation='relu' (default)"),
                          ('relu', 'linear', "relu + bottleneck='linear'"),
                          ('tanh', None, "activation='tanh'"),
                          ('linear', None, "activation='linear'")]:
    m = Autoencoder(input_dim=20, encoding_dim=3,
                    activation=act, bottleneck_activation=bneck,
                    learning_rate=0.1, epochs=1500, batch_size=32)
    m.fit(X_train)
    dead = int(np.sum(np.std(m.encode(X_test), axis=0) < 1e-12))
    print(f"  {label:<30}{str(dead) + '/3':<14}{-m.score(X_test):.6f}")

print("\nTip: a ReLU bottleneck can die permanently and never recover -")
print("     its gradient is exactly zero once it is off for every sample.")
print("     Use 'tanh' or 'linear' (or bottleneck_activation='linear')")
print("     whenever encoding_dim is small.")
```

Expected output:
```
================================================================
AUTOENCODER FROM SCRATCH - PLUG-AND-PLAY DEMO
================================================================

================================================================
DEMO 1 - A linear autoencoder rediscovers PCA
================================================================
Data: 500 points that really live on a 3-D plane inside 20-D space
      X = Z @ B + noise,  Z is 500x3,  B is 3x20
Epoch 500/1500, Loss: 0.000059
Epoch 1000/1500, Loss: 0.000054
Epoch 1500/1500, Loss: 0.000050

Loss on the internal [0,1] scale: epoch 1 = 0.116951 -> epoch 1500 = 0.000050
Compression ratio        : 6.7x (20 numbers -> 3)
Train reconstruction MSE : 0.003309
Test  reconstruction MSE : 0.003333

Known-answer check (all on the internal [0,1] scale):
  PCA(3) test MSE - the theoretical floor : 3.921e-05
  Autoencoder test MSE                   : 4.991e-05
  Ratio AE / PCA                         : 1.273x
  Angles between AE and PCA subspaces    : [0.015 0.034 0.051] degrees

Sample reconstructions (first 5 of 20 features):
  sample 0  true : [-0.922 -1.281 -0.415  1.423  0.21 ]
            recon: [-0.945 -1.323 -0.391  1.433  0.275]
  sample 1  true : [-1.147  1.815  2.295  0.053 -0.226]
            recon: [-1.213  1.772  2.262  0.035 -0.228]
  sample 2  true : [ 2.05   1.551  0.151 -1.804 -0.834]
            recon: [ 2.056  1.579  0.176 -1.752 -0.835]

================================================================
DEMO 2 - Anomaly detection with a held-out threshold
================================================================
Train on normal rows only, pick the threshold on a CALIBRATION
split, and report metrics on a third, never-seen EVAL split.

Train reconstruction MSE : 0.154894
Eval  reconstruction MSE : 0.169384  (normal rows only)
Mean error, normal rows  : 0.169384
Mean error, anomaly rows : 17.855301
Threshold (95th pct of calibration errors): 0.527696

Precision 86.21%   Recall 100.00%   F1 0.9259
                Predicted Normal  Predicted Anomaly
Actual Normal           142                8
Actual Anomaly            0               50
A 95th-percentile threshold accepts ~5% false positives by
construction, so precision here is capped, not a model failure.

Sample scores (3 normal rows, then 3 anomalies):
  true=normal   error= 0.242419  -> ok
  true=normal   error= 0.059018  -> ok
  true=normal   error= 0.043816  -> ok
  true=anomaly  error=14.053581  -> FLAG
  true=anomaly  error=25.120692  -> FLAG
  true=anomaly  error=24.721340  -> FLAG

================================================================
DEMO 3 - Dying ReLU: why the default can silently fail
================================================================
Same 20-D data and budget as DEMO 1; only the activation changes.
A 'dead' code unit has zero variance across the whole test set.
hidden_dims=None here, so the bottleneck is the ONLY nonlinear
layer - which is why row 2 and row 4 come out identical.

  setting                       dead units    test MSE
  --------------------------------------------------------
  activation='relu' (default)   3/3           2.946833
  relu + bottleneck='linear'    0/3           0.003333
  activation='tanh'             0/3           0.008612
  activation='linear'           0/3           0.003333

Tip: a ReLU bottleneck can die permanently and never recover -
     its gradient is exactly zero once it is off for every sample.
     Use 'tanh' or 'linear' (or bottleneck_activation='linear')
     whenever encoding_dim is small.
```

**Read those three blocks before anything else** - they are the whole guide in
miniature:

- **DEMO 1** says the maths is right. The autoencoder lands within 1.27x of the
  theoretical PCA floor, and the subspace it found sits within **0.05 degrees**
  of the true principal subspace.
- **DEMO 2** says how to evaluate honestly. Three disjoint splits: fit, calibrate
  the threshold, score. If you pick the threshold on the same rows you score,
  your precision is fiction.
- **DEMO 3** says what goes wrong. `activation='relu'` (the default!) can kill
  every unit in a narrow bottleneck, and the model then silently degrades to
  "predict the training mean" - 884x worse here than the linear model. This is
  the single most important practical warning in this file.

---

## What is an Autoencoder?

An Autoencoder is a **neural network architecture** used for **unsupervised learning** that learns to compress (encode) data into a lower-dimensional representation and then reconstruct (decode) it back. It's trained to make the output as similar to the input as possible.

**Real-world analogy**: 
Imagine you're packing a suitcase for vacation. You can't fit everything, so you compress clothes, take only essentials, and later unpack and "reconstruct" your wardrobe. An autoencoder does the same with data - it learns which features are essential and which can be discarded!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Algorithm Type** | Neural Network, Unsupervised Learning |
| **Learning Style** | Self-supervised (input = output) |
| **Primary Use** | Dimensionality Reduction, Feature Learning |
| **Output** | Encoded representation + Reconstructed data |
| **Key Principle** | Minimize reconstruction error |

### The Core Idea

```
"Learn to compress data to its essence, then reconstruct it"
```

The network is forced to learn the most important features because:
- The bottleneck (encoding layer) has fewer dimensions than input
- It must preserve enough information to reconstruct the original
- Only the most meaningful patterns are captured

### Architecture Components

**1. Encoder**: Compresses input to latent representation
```
Input (high-dimensional) → Hidden Layers → Encoding (low-dimensional)
Example: Image (784 pixels) → 128 → 64 → 32 → Code (10 dimensions)
```

**2. Latent Space (Bottleneck)**: Compressed representation
```
The "essence" of the data
Lower-dimensional feature vector
Contains learned representations
```

**3. Decoder**: Reconstructs from latent representation
```
Encoding (low-dimensional) → Hidden Layers → Output (high-dimensional)
Example: Code (10 dimensions) → 32 → 64 → 128 → Reconstruction (784 pixels)
```

### Key Concepts

**1. Encoding (Compression)**
```python
# Example: 100D → 10D
input: [x1, x2, x3, ..., x100]
         ↓
encoding: [z1, z2, z3, ..., z10]
```

**2. Decoding (Reconstruction)**
```python
# Example: 10D → 100D
encoding: [z1, z2, z3, ..., z10]
            ↓
output: [x̂1, x̂2, x̂3, ..., x̂100]
```

**3. Reconstruction Loss**
```
Loss = MSE(input, output)
     = (1/n) * Σ(x - x̂)²

Goal: Minimize this loss
```

**4. Compression Ratio**
```
Compression = input_dim / encoding_dim

Example: 784 / 32 = 24.5x compression
```

---

## How Autoencoders Work

### The Algorithm in 5 Steps

```
Step 1: Forward Pass (Encoding)
         Input → Hidden Layers → Compressed Representation
         ↓
Step 2: Forward Pass (Decoding)
         Compressed → Hidden Layers → Reconstructed Output
         ↓
Step 3: Calculate Loss
         Loss = |Input - Reconstructed Output|²
         ↓
Step 4: Backward Pass (Backpropagation)
         Compute gradients, update weights
         ↓
Step 5: Repeat Steps 1-4 until convergence
```

### Visual Example: Encoding Process

Let's compress 6-dimensional data to 2 dimensions:

```
Original Data Point:
x = [2.5, 1.3, 0.8, 3.2, 1.9, 2.1]  (6 dimensions)

Encoder Forward Pass:

Layer 1 (Input → Hidden):
z1 = W1·x + b1 = [4.2, 3.1, 2.8, 1.5]  (4 dimensions)
a1 = ReLU(z1) = [4.2, 3.1, 2.8, 1.5]

Layer 2 (Hidden → Encoding):
z2 = W2·a1 + b2 = [2.1, -0.5]  (2 dimensions)
encoding = ReLU(z2) = [2.1, 0.0]  ← Compressed!

Compression achieved: 6D → 2D (3x compression)
```

### Decoding Process

```
Compressed Representation:
encoding = [2.1, 0.0]  (2 dimensions)

Decoder Forward Pass:

Layer 3 (Encoding → Hidden):
z3 = W3·encoding + b3 = [1.8, 2.5, 3.0, 1.2]  (4 dimensions)
a3 = ReLU(z3) = [1.8, 2.5, 3.0, 1.2]

Layer 4 (Hidden → Output):
z4 = W4·a3 + b4 = [2.4, 1.4, 0.9, 3.0, 1.8, 2.0]  (6 dimensions)
output = z4 = [2.4, 1.4, 0.9, 3.0, 1.8, 2.0]  ← Reconstructed!

Reconstruction Error:
MSE = (1/6) * [(0.1)² + (-0.1)² + (-0.1)² + (0.2)² + (0.1)² + (0.1)²]
    = (1/6) * [0.01 + 0.01 + 0.01 + 0.04 + 0.01 + 0.01]
    = (1/6) * 0.09
    = 0.015
```

### Training Process

**Initialization**:
```
Input: X (n_samples × input_dim)
Target: X (same as input! - this is key)
Initialize: Random weights for encoder and decoder
```

**Epoch 1**:
```
Batch 1: [samples 1-32]
  Forward: x → encode → decode → x̂
  Loss: 0.521
  Backward: Update all weights
  
Batch 2: [samples 33-64]
  Forward: x → encode → decode → x̂
  Loss: 0.498
  Backward: Update all weights
  
... (continue for all batches)

Epoch 1 Average Loss: 0.485
```

**Epoch 50**:
```
Average Loss: 0.052  ← Much better!
Network has learned good representations
```

### Learning Representations

What the autoencoder learns at each layer:

```
Input Layer:
Raw features: [x1, x2, x3, ..., x100]

First Hidden Layer:
Low-level patterns:
- Feature combinations
- Basic correlations
- Simple patterns

Encoding Layer (Bottleneck):
High-level representations:
- Abstract concepts
- Essential information
- Compressed features

Decoder Layers:
Reconstruction process:
- Decode abstract concepts
- Reconstruct patterns
- Restore details
```

---

## The Mathematical Foundation

### 1. Network Architecture

For an autoencoder with L layers:

**Forward Propagation (Encoding)**:
```
Layer 1: a⁽¹⁾ = σ(W⁽¹⁾x + b⁽¹⁾)
Layer 2: a⁽²⁾ = σ(W⁽²⁾a⁽¹⁾ + b⁽²⁾)
...
Encoding: z = a⁽ᴸᵉ⁾ = σ(W⁽ᴸᵉ⁾a⁽ᴸᵉ⁻¹⁾ + b⁽ᴸᵉ⁾)

Where:
- σ = activation function (ReLU, sigmoid, tanh)
- W⁽ˡ⁾ = weight matrix for layer l
- b⁽ˡ⁾ = bias vector for layer l
- Lₑ = encoding layer index
```

**Forward Propagation (Decoding)**:
```
Layer Lₑ+1: a⁽ᴸᵉ⁺¹⁾ = σ(W⁽ᴸᵉ⁺¹⁾z + b⁽ᴸᵉ⁺¹⁾)
...
Output: x̂ = a⁽ᴸ⁾ = W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾  (linear activation)

Where:
- L = total number of layers
- x̂ = reconstructed output
```

### 2. Loss Function

**Mean Squared Error (MSE)**:
```
L(x, x̂) = (1/2) * Σᵢ₌₁ⁿ (xᵢ - x̂ᵢ)²

Or in matrix form, averaged over a batch of m samples:
L(X, X̂) = (1/2m) * ||X - X̂||²_F

Where:
- m = number of samples
- n = number of features
- ||·||_F = Frobenius norm
```

The **1/2** is there purely so the derivative comes out clean: it cancels the
2 from the power rule, giving `∂L/∂x̂ = x̂ - x` with no stray factor. That is
exactly the one line `delta = (output - X)` in `_backward_pass`.

**Objective**:
```
minimize L(θ) = minimize (1/2m) * Σⱼ₌₁ᵐ ||x⁽ʲ⁾ - x̂⁽ʲ⁾||²

Where:
- θ = {W⁽¹⁾, b⁽¹⁾, ..., W⁽ᴸ⁾, b⁽ᴸ⁾} (all parameters)
- j = sample index
```

> **Two "losses", one objective.** The gradients this implementation computes
> are the exact gradients of the `1/2m` loss above (checked against central
> differences to ~5e-10). The number stored in `self.history['loss']` is a
> *different but proportional* quantity: the plain per-element MSE,
> `np.mean((x̂ - x)**2)`, which equals `(2 / input_dim) * L`. The constant is
> absorbed into `learning_rate`, so the optimisation is correct - but it does
> mean the effective step size grows with `input_dim`, and you should scale
> `learning_rate` **down** for wide inputs. USAGE EXAMPLE 8 shows a 40-dim
> problem diverging to `nan` at `learning_rate=0.5`.

### 3. Backpropagation

**Output Layer Gradient**:
```
For MSE loss with linear output:
δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾ = x̂ - x = a⁽ᴸ⁾ - x

Dimensions: (m × n) where m = batch size, n = features
```

**Hidden Layer Gradients (Backward)**:
```
For layer l:
δ⁽ˡ⁾ = (δ⁽ˡ⁺¹⁾ · (W⁽ˡ⁺¹⁾)ᵀ) ⊙ σ'(z⁽ˡ⁾)

Where:
- δ⁽ˡ⁾ = error term for layer l
- ⊙ = element-wise multiplication
- σ'(z⁽ˡ⁾) = derivative of activation function
```

**Activation Function Derivatives**:
```
ReLU:
σ(z) = max(0, z)
σ'(z) = 1 if z > 0, else 0

Sigmoid:
σ(z) = 1 / (1 + e⁻ᶻ)
σ'(z) = σ(z) · (1 - σ(z))

Tanh:
σ(z) = tanh(z)
σ'(z) = 1 - tanh²(z)
```

**Weight and Bias Gradients**:
```
∂L/∂W⁽ˡ⁾ = (1/m) * (a⁽ˡ⁻¹⁾)ᵀ · δ⁽ˡ⁾

∂L/∂b⁽ˡ⁾ = (1/m) * Σⱼ₌₁ᵐ δ⁽ˡ⁾⁽ʲ⁾

Where:
- m = batch size
- j = sample index in batch
```

### 4. Gradient Descent Update

**Parameter Update Rule**:
```
W⁽ˡ⁾ ← W⁽ˡ⁾ - α · ∂L/∂W⁽ˡ⁾
b⁽ˡ⁾ ← b⁽ˡ⁾ - α · ∂L/∂b⁽ˡ⁾

Where:
- α = learning rate
- Typical values: 0.001 - 0.1
```

**Mini-batch Gradient Descent**:
```
For each epoch:
  Shuffle dataset
  For each mini-batch B:
    1. Forward pass: compute x̂ for samples in B
    2. Compute loss: L = MSE(x, x̂)
    3. Backward pass: compute gradients
    4. Update parameters: W, b ← W, b - α∇L
```

### 5. Weight Initialization

**Xavier/Glorot Initialization**:
```
W⁽ˡ⁾ ~ Uniform(-√(6/(nᵢₙ + nₒᵤₜ)), √(6/(nᵢₙ + nₒᵤₜ)))

Or Normal distribution:
W⁽ˡ⁾ ~ N(0, √(2/(nᵢₙ + nₒᵤₜ)))

Where:
- nᵢₙ = number of input units
- nₒᵤₜ = number of output units

This helps prevent vanishing/exploding gradients!
```

### 6. Reconstruction Error (Per Sample)

```
For anomaly detection:

Error(x⁽ⁱ⁾) = ||x⁽ⁱ⁾ - x̂⁽ⁱ⁾||²

Interpretation:
- Low error: Normal sample, well-reconstructed
- High error: Anomaly, poorly reconstructed
```

---

## Implementation Details

### Network Structure

```python
# Example: 100D input → 50D hidden → 10D encoding → 50D hidden → 100D output

Encoder:
  Input Layer:    100 neurons
  Hidden Layer:    50 neurons (ReLU)
  Encoding Layer:  10 neurons (ReLU)  ← Bottleneck

Decoder:
  Hidden Layer:    50 neurons (ReLU)
  Output Layer:   100 neurons (Linear)

Total Parameters:
  Encoder: (100×50 + 50) + (50×10 + 10) = 5,050 + 510 = 5,560
  Decoder: (10×50 + 50) + (50×100 + 100) = 550 + 5,100 = 5,650
  Total: 11,210 parameters
```

### Data Preprocessing

**Normalization** (Important!):
```python
# Scale data to [0, 1] for better training
X_min = min(X)
X_max = max(X)
X_normalized = (X - X_min) / (X_max - X_min)

# After reconstruction, denormalize:
X_reconstructed = X̂ * (X_max - X_min) + X_min
```

Why normalize?
- Keeps gradients stable
- Faster convergence
- Prevents saturation in sigmoid/tanh
- All features on similar scale

**This happens automatically, inside `fit()`.** You do not scale your data
yourself. Three consequences that are easy to trip over:

1. **`fit()` stores `self.data_min` / `self.data_max` (per feature) and every
   later call reuses them.** `encode()` scales incoming data with the *training*
   min/max, and `decode()` un-scales with the same numbers, so
   `reconstruct()` and `reconstruction_error()` always return values on your
   original scale. If you standardized your data yourself first, it is simply
   rescaled again - harmless, but not what you intended.
2. **`history['loss']` is on the internal [0, 1] scale, `reconstruction_error()`
   is on your scale.** They will not match, and they are not supposed to. In
   DEMO 1 above the final `history['loss']` is `0.000050` while the test
   reconstruction MSE is `0.003333`; both describe the same model.
3. **Out-of-range inputs are not clipped.** A test point below `data_min` or
   above `data_max` maps outside [0, 1]. For anomaly detection that is a
   *feature*, not a bug - it is precisely why anomalies reconstruct badly - but
   it does mean the code layer sees inputs the network never trained on.

The `+ 1e-8` you will see in the code (`(data_max - data_min + 1e-8)`) exists so
a **constant feature**, where `data_max == data_min`, divides by 1e-8 instead of
by zero. The same `+ 1e-8` must appear on the denormalization side in `decode()`
or the round-trip would not be exact.

### The Bottleneck Activation, and the Dying-ReLU Trap

This is the most important practical section in this guide, because it is the
failure mode you will actually hit.

In this implementation the bottleneck is not special: it is just another layer,
so it passes through `self.activation` like any hidden layer (see
`_layer_activation`). With the default `activation='relu'` that is a trap:

```
1. fit() min-max scales X into [0, 1]  ->  every input is non-negative
2. biases are initialized to exactly 0
3. Xavier weights are symmetric about 0, so roughly half of the
   pre-activations z = a.W + b come out negative
4. ReLU sends those to exactly 0
5. ReLU'(z) = 0 for z < 0, so the gradient flowing into that unit is
   EXACTLY zero -> the unit can never be revived
```

A code unit that is off for every sample in the batch is dead permanently. With
a small `encoding_dim` (2, 3, 5) a large fraction - sometimes *all* - of the code
units die, and the "compressed representation" becomes a constant. The decoder
then does the only thing it can: it outputs the training mean for every input.

**How to spot it** (one line):

```python
dead = np.sum(np.std(model.encode(X), axis=0) < 1e-12)
print(f"{dead} of {model.encoding_dim} code units are dead")
```

**How to fix it** - any one of these:

| Fix | What it does |
|-----|--------------|
| `activation='tanh'` | Smooth and two-sided; `tanh'(z) > 0` everywhere, so no unit can freeze |
| `activation='linear'` | No nonlinearity at all; the network becomes PCA (see DEMO 1) |
| `bottleneck_activation='linear'` | Keeps ReLU in the hidden layers but frees the code layer |

Measured on the DEMO 1 data (20-D, 3 hidden factors, 1500 epochs, identical
seed and identical everything else):

| Setting | Dead code units | Test MSE |
|---------|-----------------|----------|
| `activation='relu'` (default) | 3 / 3 | 2.946833 |
| `activation='relu'`, `bottleneck_activation='linear'` | 0 / 3 | 0.003333 |
| `activation='tanh'` | 0 / 3 | 0.008612 |
| `activation='linear'` | 0 / 3 | 0.003333 |

The default is 884x worse than the linear model, and it fails *silently* - the
loss curve flattens and everything still "runs". Whenever `encoding_dim` is
small, reach for `'tanh'` or `'linear'`.

### Training Strategy

**1. Mini-batch Training**:
```
Batch size = 32 (typical)

Advantages:
- More stable than single-sample SGD
- Faster than full-batch GD
- Better generalization
- Efficient GPU utilization
```

**2. Learning Rate Selection**:
```
Too low (0.0001):  Slow convergence, many epochs needed
Good (0.001-0.01): Stable training, good convergence
Too high (0.5):    Unstable, may diverge
```

**3. Number of Epochs**:
```
Monitor training loss:
- Still decreasing → continue training
- Plateaued → stop (converged)
- Increasing → learning rate too high
```

### Encoding Dimension Selection

**Rule of Thumb**:
```
For dimensionality reduction:
  encoding_dim ≈ 10-30% of input_dim
  
For feature learning:
  encoding_dim ≈ number of underlying factors
  
For compression:
  encoding_dim = as small as possible while maintaining acceptable reconstruction
```

**Example Trade-offs**:
```
Input: 784 dimensions (28×28 image)

encoding_dim = 2:
  ✓ 392x compression!
  ✓ Easy to visualize
  ✗ Significant information loss
  
encoding_dim = 32:
  ✓ 24.5x compression
  ✓ Good reconstruction
  ✓ Captures important features
  
encoding_dim = 100:
  ✓ 7.8x compression
  ✓ Excellent reconstruction
  ✗ Less aggressive compression
```

---

## Step-by-Step Example

Let's walk through a complete example: compressing 8-dimensional data to 2 dimensions.

### Dataset

```python
# 5 samples, 8 features each
X = [
    [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2, 1.3],  # Sample 1
    [1.2, 1.9, 1.6, 2.8, 2.3, 1.7, 2.1, 1.4],  # Sample 2
    [0.9, 2.1, 1.4, 3.2, 2.6, 1.9, 2.3, 1.2],  # Sample 3
    [1.1, 2.0, 1.5, 2.9, 2.4, 1.8, 2.0, 1.3],  # Sample 4
    [1.0, 1.8, 1.7, 3.1, 2.7, 2.0, 2.4, 1.5],  # Sample 5
]
```

### Step 1: Initialize Network

```
Architecture: 8 → 4 → 2 → 4 → 8

Encoder:
  W⁽¹⁾: 8×4 matrix (randomly initialized)
  b⁽¹⁾: 4-vector (zeros)
  W⁽²⁾: 4×2 matrix (randomly initialized)
  b⁽²⁾: 2-vector (zeros)

Decoder:
  W⁽³⁾: 2×4 matrix (randomly initialized)
  b⁽³⁾: 4-vector (zeros)
  W⁽⁴⁾: 4×8 matrix (randomly initialized)
  b⁽⁴⁾: 8-vector (zeros)
```

### Step 2: Forward Pass (Epoch 1, Sample 1)

**Input**:
```
x = [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2, 1.3]
```

**Encoder Layer 1**:
```
z⁽¹⁾ = W⁽¹⁾·x + b⁽¹⁾ = [1.2, 0.8, 1.5, 0.9]  (example values)
a⁽¹⁾ = ReLU(z⁽¹⁾) = [1.2, 0.8, 1.5, 0.9]
```

**Encoding Layer**:
```
z⁽²⁾ = W⁽²⁾·a⁽¹⁾ + b⁽²⁾ = [0.7, -0.3]
encoding = ReLU(z⁽²⁾) = [0.7, 0.0]  ← Compressed to 2D!
```

**Decoder Layer 1**:
```
z⁽³⁾ = W⁽³⁾·encoding + b⁽³⁾ = [0.9, 1.1, 0.6, 1.3]
a⁽³⁾ = ReLU(z⁽³⁾) = [0.9, 1.1, 0.6, 1.3]
```

**Output Layer**:
```
x̂ = W⁽⁴⁾·a⁽³⁾ + b⁽⁴⁾ = [1.1, 1.8, 1.6, 2.7, 2.3, 1.9, 2.0, 1.4]
```

### Step 3: Compute Loss

```
Target: x = [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2, 1.3]
Output: x̂ = [1.1, 1.8, 1.6, 2.7, 2.3, 1.9, 2.0, 1.4]

Errors = x - x̂ = [-0.1, 0.2, -0.1, 0.3, 0.2, -0.1, 0.2, -0.1]

MSE = (1/8) * [0.01 + 0.04 + 0.01 + 0.09 + 0.04 + 0.01 + 0.04 + 0.01]
    = 0.03125
```

### Step 4: Backward Pass

**Output Layer Gradient**:
```
δ⁽⁴⁾ = x̂ - x = [0.1, -0.2, 0.1, -0.3, -0.2, 0.1, -0.2, 0.1]

∂L/∂W⁽⁴⁾ = a⁽³⁾ᵀ · δ⁽⁴⁾  (4×8 matrix)
∂L/∂b⁽⁴⁾ = δ⁽⁴⁾  (8-vector)
```

**Propagate to Layer 3**:
```
δ⁽³⁾ = (δ⁽⁴⁾ · (W⁽⁴⁾)ᵀ) ⊙ ReLU'(z⁽³⁾)
     = [0.05, 0.08, 0.03, 0.09]  (example)

∂L/∂W⁽³⁾ = encodingᵀ · δ⁽³⁾  (2×4 matrix)
∂L/∂b⁽³⁾ = δ⁽³⁾  (4-vector)
```

**Continue propagating back through encoder...**

### Step 5: Update Weights

```
Learning rate α = 0.01

W⁽⁴⁾ ← W⁽⁴⁾ - 0.01 * ∂L/∂W⁽⁴⁾
b⁽⁴⁾ ← b⁽⁴⁾ - 0.01 * ∂L/∂b⁽⁴⁾

(Repeat for all layers)
```

### Step 6: Repeat for All Samples

```
Process samples 2, 3, 4, 5 the same way
Average loss for epoch 1: 0.048
```

### Step 7: Continue Training

Steps 2-6 above used illustrative round numbers so the arithmetic stays
readable. From here on **every number is real**: it comes from actually running
the 5-sample dataset at the top of this section through the implementation, with

```python
autoencoder = Autoencoder(
    input_dim=8, encoding_dim=2, hidden_dims=[4],
    activation='tanh',        # NOT the 'relu' default - see Step 8's warning
    learning_rate=0.05, epochs=500, batch_size=5
)
autoencoder.fit(X)
```

`autoencoder.history['loss']` (on the internal [0, 1] scale) then contains:

```
Epoch 1:   Loss = 0.454650
Epoch 10:  Loss = 0.138608
Epoch 20:  Loss = 0.102110
Epoch 50:  Loss = 0.071684
Epoch 100: Loss = 0.035698
Epoch 200: Loss = 0.006442
Epoch 500: Loss = 0.003084  ← flat, converged
```

### Step 8: Use Trained Model

**Encode new data**:
```python
new_sample = [1.05, 1.95, 1.55, 2.95, 2.45, 1.75, 2.15, 1.35]

encoded = autoencoder.encode(new_sample)
# Output: [-0.0004, 0.4168]  ← Compressed to 2D!
```

**Reconstruct**:
```python
reconstructed = autoencoder.reconstruct(new_sample)
# Output: [1.0994, 1.9212, 1.5911, 2.9287, 2.4590, 1.8053, 2.1547, 1.3849]

reconstruction_error = autoencoder.reconstruction_error(new_sample)[0]
# Output: 0.001225  ← Very good reconstruction!
```

Note `[0]`: `reconstruction_error` always returns one value **per row**, so a
single 1-D sample gives you an array of length 1.

> **Try the default and watch it break.** Drop `activation='tanh'` from the call
> above and rerun with `learning_rate=0.01, epochs=50` (the naive settings) and
> you get `encoded = [0., 0.]` - both ReLU code units dead - with the
> reconstruction collapsing to the training mean,
> `[0.9582, 1.8617, 1.4545, 2.8775, 2.3772, 1.7577, 2.0797, 1.2574]`, and MSE
> `0.006184`. Five times worse, and with a code that carries no information at
> all. See [The Bottleneck Activation, and the Dying-ReLU
> Trap](#the-bottleneck-activation-and-the-dying-relu-trap).

---

## Real-World Applications

### 1. Image Compression

**Problem**: Store or transmit images efficiently

**Solution**: Use autoencoder to compress images

```python
# MNIST digits: 28×28 = 784 pixels
autoencoder = Autoencoder(
    input_dim=784,
    encoding_dim=32,  # Compress to 32 numbers
    hidden_dims=[256, 128, 64]
)

# Train on image dataset
autoencoder.fit(mnist_images)

# Compress: 784 → 32 (24.5x compression!)
compressed = autoencoder.encode(image)

# Transmit only 32 numbers instead of 784

# Decompress: 32 → 784
reconstructed = autoencoder.decode(compressed)
```

**Real-world use**: JPG-like compression, streaming services

### 2. Anomaly Detection

**Problem**: Detect credit card fraud, network intrusions, defective products

**Solution**: Train on normal data, flag high reconstruction error

```python
# Train on normal transactions only
autoencoder = Autoencoder(input_dim=30, encoding_dim=10, activation='tanh')
autoencoder.fit(normal_transactions)

# Choose the threshold on a CALIBRATION split you did not train on,
# then score data you used for neither. See DEMO 2 in the Quick Start.
threshold = np.percentile(autoencoder.reconstruction_error(calibration_rows), 95)

# Test on a new transaction. reconstruction_error returns one value per
# ROW, so a single 1-D sample gives an array of length 1 - take [0].
new_transaction = [...]        # 30 numbers
error = autoencoder.reconstruction_error(new_transaction)[0]

# High error → Anomaly!
if error > threshold:
    print("FRAUD DETECTED!")
```

**Why it works**: 
- Autoencoder learns "normal" patterns
- Normal data: reconstructs well (low error)
- Anomalies: doesn't know how to reconstruct (high error)

**Real-world use**: 
- Credit card fraud detection
- Manufacturing defect detection
- Network intrusion detection
- Medical diagnosis

### 3. Dimensionality Reduction

**Problem**: Visualize high-dimensional data

**Solution**: Compress to 2D or 3D for plotting

```python
# 100-dimensional customer data
autoencoder = Autoencoder(
    input_dim=100,
    encoding_dim=2  # Compress to 2D for visualization
)
autoencoder.fit(customer_data)

# Get 2D representation
coords_2d = autoencoder.encode(customer_data)

# Now plot in 2D!
plt.scatter(coords_2d[:, 0], coords_2d[:, 1])
```

**Advantage over PCA**: Can capture non-linear relationships

**Real-world use**:
- Customer segmentation visualization
- Gene expression analysis
- Document similarity visualization

### 4. Feature Learning

**Problem**: Extract meaningful features for downstream tasks

**Solution**: Use encoding as features for classification/regression

```python
# Train autoencoder on raw data
autoencoder = Autoencoder(input_dim=200, encoding_dim=20)
autoencoder.fit(raw_data)

# Extract learned features
features = autoencoder.encode(raw_data)

# Use features for classification
classifier.fit(features, labels)
```

**Benefits**:
- Reduces dimensionality (faster training)
- Removes noise (better accuracy)
- Learns task-agnostic features

**Real-world use**: 
- Transfer learning
- Preprocessing for ML pipelines
- Feature extraction from images, text, audio

### 5. Denoising

**Problem**: Remove noise from corrupted data

**Solution**: Train autoencoder to map noisy → clean

```python
# Add noise to clean images
noisy_images = clean_images + noise

# Train to reconstruct clean from noisy
denoiser = Autoencoder(input_dim=784, encoding_dim=64)
denoiser.fit(noisy_images)  # Note: traditionally trained with clean as target

# Denoise new noisy image
denoised = denoiser.reconstruct(noisy_image)
```

**Real-world use**:
- Image denoising (photography, medical imaging)
- Audio denoising (speech recognition, music)
- Signal processing

### 6. Data Generation

**Problem**: Generate new similar samples

**Solution**: Sample from latent space, decode

```python
# Train autoencoder
autoencoder.fit(training_data)

# Sample from latent space - but sample from the range the encoder ACTUALLY
# produces, not from a standard normal. Under the default ReLU bottleneck every
# real code is >= 0, so half of np.random.randn's draws would be codes the
# decoder has never seen.
codes = autoencoder.encode(training_data)
low, high = codes.min(axis=0), codes.max(axis=0)
random_encoding = np.random.uniform(low, high, size=(1, autoencoder.encoding_dim))

# Generate new data
generated = autoencoder.decode(random_encoding)
```

**Note**: Variational Autoencoders (VAEs) are better for generation

**Real-world use**:
- Data augmentation
- Synthetic data generation
- Content creation

---

## Understanding the Code

Let's break down the key components of our implementation:

### 1. Network Initialization

```python
def _initialize_network(self):
    # Build encoder: input → hidden layers → encoding
    encoder_dims = [self.input_dim] + self.hidden_dims + [self.encoding_dim]
    
    # Build decoder: encoding → hidden layers (reversed) → output
    decoder_dims = [self.encoding_dim] + self.hidden_dims[::-1] + [self.input_dim]
    
    # Combine
    all_dims = encoder_dims + decoder_dims[1:]
    
    # Xavier initialization for each layer
    for i in range(len(all_dims) - 1):
        limit = np.sqrt(6 / (all_dims[i] + all_dims[i+1]))
        weight = self._rng.uniform(-limit, limit, (all_dims[i], all_dims[i+1]))
        bias = np.zeros(all_dims[i+1])
        
        self.weights.append(weight)
        self.biases.append(bias)

    self.n_layers = len(self.weights)
    self.n_encoder_layers = len(self.hidden_dims) + 1
```

**What it does**:
- Creates symmetric encoder/decoder architecture
- Initializes weights using Xavier method (prevents gradient problems)
- Stores all weights and biases in lists

**Note `self._rng`**: this is a private `np.random.RandomState(random_state)`
created in `__init__`. Constructing a model therefore does **not** reseed or
consume NumPy's global RNG - your own `np.random.seed(...)` stream is left
exactly as you set it - while two models built with the same `random_state`
still get identical weights.

**Which activation does each layer use?** One helper answers that for the whole
class, so the forward pass, the backward pass, `encode()` and `decode()` can
never disagree:

```python
def _layer_activation(self, layer_index):
    if layer_index == self.n_layers - 1:
        return 'linear'                       # output must be unbounded
    if layer_index == self.n_encoder_layers - 1 and self.bottleneck_activation:
        return self.bottleneck_activation     # optional override at the code layer
    return self.activation
```

**Example**:
```
Input: 100D, Hidden: [50, 25], Encoding: 10D

Encoder: 100 → 50 → 25 → 10
Decoder: 10 → 25 → 50 → 100

Weights: [W₁(100×50), W₂(50×25), W₃(25×10), W₄(10×25), W₅(25×50), W₆(50×100)]
```

### 2. Forward Pass

```python
def _forward_pass(self, X):
    activations = [X]
    pre_activations = []
    
    for i in range(self.n_layers):
        # Linear transformation
        z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
        pre_activations.append(z)
        
        # Activation function: linear on the output layer,
        # bottleneck_activation on the code layer, self.activation elsewhere
        a = self._activate(z, self._layer_activation(i))
        
        activations.append(a)
    
    return activations, pre_activations
```

**What it does**:
- Propagates input through all layers
- Stores activations (needed for backprop)
- Uses specified activation for hidden layers, linear for output

**Flow**:
```
Input x
  ↓ W⁽¹⁾x + b⁽¹⁾
z⁽¹⁾
  ↓ ReLU
a⁽¹⁾
  ↓ W⁽²⁾a⁽¹⁾ + b⁽²⁾
z⁽²⁾ (encoding)
  ↓ ReLU
... (decoder)
  ↓ W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾
Output x̂
```

### 3. Backward Pass

```python
def _backward_pass(self, X, activations, pre_activations):
    n_samples = X.shape[0]
    weight_gradients = []
    bias_gradients = []
    
    # Output error
    output = activations[-1]
    delta = (output - X)  # X is target for autoencoder!
    
    # Backpropagate
    for i in range(self.n_layers - 1, -1, -1):
        # Compute gradients
        weight_grad = np.dot(activations[i].T, delta) / n_samples
        bias_grad = np.mean(delta, axis=0)
        
        weight_gradients.insert(0, weight_grad)
        bias_gradients.insert(0, bias_grad)
        
        # Propagate error backwards
        if i > 0:
            delta = np.dot(delta, self.weights[i].T)
            delta = delta * self._activate_derivative(
                pre_activations[i-1], self._layer_activation(i-1))
    
    return weight_gradients, bias_gradients
```

**What it does**:
- Computes gradients for all weights and biases
- Uses chain rule to propagate error backwards
- Key insight: Target is input itself (x̂ should match x)

**Line-by-line against the maths above**:

| Code | Formula |
|------|---------|
| `delta = (output - X)` | δ⁽ᴸ⁾ = x̂ - x |
| `np.dot(activations[i].T, delta) / n_samples` | ∂L/∂W⁽ˡ⁾ = (a⁽ˡ⁻¹⁾)ᵀ · δ⁽ˡ⁾ / m |
| `np.mean(delta, axis=0)` | ∂L/∂b⁽ˡ⁾ = (1/m) Σⱼ δ⁽ˡ⁾⁽ʲ⁾ |
| `np.dot(delta, self.weights[i].T)` | δ⁽ˡ⁺¹⁾ · (W⁽ˡ⁺¹⁾)ᵀ |
| `* self._activate_derivative(...)` | ⊙ σ'(z⁽ˡ⁾) |

These gradients were verified against central differences (eps = 1e-6) on a
4-3-2-3-4 network for `linear`, `tanh` and `sigmoid`: the largest disagreement
anywhere was about **5e-10**, i.e. the backpropagation is exact.

**Chain rule in action**:
```
∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
           = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
```

### 4. Training Loop

```python
def fit(self, X):
    # Normalize data (min/max are stored on the model for encode/decode)
    self.data_min = np.min(X, axis=0)
    self.data_max = np.max(X, axis=0)
    X_normalized = (X - self.data_min) / (self.data_max - self.data_min + 1e-8)
    
    # Fresh loss curve; weights are NOT re-initialized, so a second
    # fit() call continues training (a warm start)
    self.history = {'loss': []}
    
    for epoch in range(self.epochs):
        # Shuffle with the model's PRIVATE rng, never the global one
        indices = self._rng.permutation(n_samples)
        X_shuffled = X_normalized[indices]
        
        # Mini-batch training
        for batch_start in range(0, n_samples, self.batch_size):
            X_batch = X_shuffled[batch_start:batch_end]
            
            # Forward pass
            activations, pre_activations = self._forward_pass(X_batch)
            
            # Compute loss
            reconstruction = activations[-1]
            batch_loss = np.mean((reconstruction - X_batch) ** 2)
            
            # Backward pass
            weight_grads, bias_grads = self._backward_pass(X_batch, activations, pre_activations)
            
            # Update parameters
            self._update_parameters(weight_grads, bias_grads)
```

**What it does**:
- Normalizes data (crucial for good training!)
- Processes data in mini-batches
- Alternates forward pass (prediction) and backward pass (learning)
- Updates weights using gradient descent

### 5. Encoding and Decoding

```python
def encode(self, X):
    if self.data_min is None:
        raise ValueError("Model is not fitted yet. Call fit(X) first.")
    
    # Normalize with the SAME min/max fit() learned
    X_normalized = (X - self.data_min) / (self.data_max - self.data_min + 1e-8)
    
    # Only forward through encoder layers
    activation = X_normalized
    n_encoder_layers = len(self.hidden_dims) + 1
    
    for i in range(n_encoder_layers):
        z = np.dot(activation, self.weights[i]) + self.biases[i]
        activation = self._activate(z, self._layer_activation(i))
    
    return activation  # This is the compressed representation

def decode(self, encoded):
    if self.data_min is None:
        raise ValueError("Model is not fitted yet. Call fit(X) first.")
    
    # Only forward through decoder layers
    activation = encoded
    n_encoder_layers = len(self.hidden_dims) + 1
    
    for i in range(n_encoder_layers, self.n_layers):
        z = np.dot(activation, self.weights[i]) + self.biases[i]
        activation = self._activate(z, self._layer_activation(i))
    
    # Denormalize (the + 1e-8 must match fit()'s, or the round-trip drifts)
    decoded = activation * (self.data_max - self.data_min + 1e-8) + self.data_min
    return decoded
```

**What it does**:
- `encode()`: Compresses input to latent representation
- `decode()`: Reconstructs from latent representation
- Can be used separately after training

### 6. API Reference

Everything the class exposes, in one place.

**Constructor**

```python
Autoencoder(input_dim, encoding_dim, hidden_dims=None, learning_rate=0.01,
            activation='relu', epochs=100, batch_size=32, verbose=False,
            bottleneck_activation=None, random_state=42)
```

| Parameter | Meaning |
|-----------|---------|
| `input_dim` | Number of input features (required) |
| `encoding_dim` | Size of the bottleneck / code (required) |
| `hidden_dims` | Encoder hidden sizes, e.g. `[128, 64]`. The decoder mirrors them. `None` = no hidden layers |
| `learning_rate` | SGD step size. Scale it **down** as `input_dim` grows |
| `activation` | `'relu'`, `'sigmoid'`, `'tanh'` or `'linear'` for hidden layers and the bottleneck |
| `epochs` | Passes over the training set |
| `batch_size` | Samples per gradient update |
| `verbose` | `False` = silent, `True` = print every epoch, `int N` = print every N epochs |
| `bottleneck_activation` | Overrides `activation` at the code layer only. `None` keeps the old behaviour; `'linear'` cures dead ReLU code units |
| `random_state` | Seed for the model's **private** RNG (weights + shuffling). Never touches the global NumPy RNG |

**Methods**

| Call | Returns | Notes |
|------|---------|-------|
| `fit(X)` | `self` | Min-max scales X internally; resets `history`; does **not** reset weights (a second `fit` warm-starts) |
| `encode(X)` | `(n_samples, encoding_dim)` | A 1-D sample of shape `(input_dim,)` returns shape `(encoding_dim,)` |
| `decode(Z)` | `(n_samples, input_dim)` | Denormalized back to your original scale |
| `reconstruct(X)` | `(n_samples, input_dim)` | `decode(encode(X))` |
| `reconstruction_error(X)` | `(n_samples,)` | Per-sample MSE on the original scale. A 1-D sample gives an array of length 1 |
| `score(X)` | `float` | `-mean(reconstruction_error(X))`; higher is better, 0.0 is perfect. Unsupervised, so no `y` |
| `get_compression_ratio()` | `float` | `input_dim / encoding_dim` |
| `transform(X)` | same as `encode` | Transformer-family alias |
| `fit_transform(X)` | same as `encode` | `fit(X).encode(X)` |
| `inverse_transform(Z)` | same as `decode` | Transformer-family alias |

`encode`, `decode` and everything built on them raise
`ValueError("Model is not fitted yet. Call fit(X) first.")` if you call them
before `fit`.

**Attributes**

| Attribute | Meaning |
|-----------|---------|
| `history['loss']` | One average loss per epoch, on the internal **[0, 1]** scale. Reset at the start of every `fit()` |
| `data_min`, `data_max` | Per-feature min/max learned by `fit()`; `None` before fitting |
| `weights`, `biases` | Lists of arrays, one per layer, encoder first then decoder |
| `n_layers` | Total number of weight matrices |
| `n_encoder_layers` | `len(hidden_dims) + 1`; the bottleneck is layer `n_encoder_layers - 1` |

---

## Model Evaluation

### 1. Reconstruction Error

**Primary Metric**:
```python
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
```

**Interpretation**:
- Lower is better
- Measures how well the model preserves information
- Compare to baseline (e.g., using PCA)

**What's a good error?**
```
Depends on data scale:
- Normalized [0,1]: Error < 0.01 is excellent
- Original scale: Compare to variance of data

Rule of thumb: Error < 5% of data variance is good
```

### 2. Visualization (for 2D/3D encodings)

```python
# Encode to 2D
encoded = autoencoder.encode(X)

# Plot
plt.scatter(encoded[:, 0], encoded[:, 1], c=labels)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
```

**What to look for**:
- Clusters (similar samples close together)
- Smooth transitions
- Meaningful structure

### 3. Anomaly Detection Metrics

```python
# Three DISJOINT splits: fit, calibrate, score. Never reuse one for two jobs.
autoencoder.fit(X_fit)

# The threshold is a fitted parameter too - choose it on its own split
calib_errors = autoencoder.reconstruction_error(X_calib)
threshold = np.percentile(calib_errors, 95)

errors = autoencoder.reconstruction_error(X_test)
anomalies = errors > threshold

# Calculate metrics
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
```

A 95th-percentile threshold *by construction* lets about 5% of normal rows
through as false positives, so precision is capped before the model has said a
word. If you want higher precision, raise the percentile - and expect recall to
fall. DEMO 2 in the Quick Start is a complete worked version of this.

**ROC Curve**: Plot TPR vs FPR at different thresholds

### 4. Compression vs Quality Trade-off

```python
encoding_dims = [2, 5, 10, 20, 50]
errors = []

for dim in encoding_dims:
    # 'tanh' matters here: at dim=2 a ReLU bottleneck usually dies and the
    # curve you plot would be the dying-ReLU curve, not the compression curve
    model = Autoencoder(input_dim=100, encoding_dim=dim,
                        activation='tanh', learning_rate=0.05, epochs=400)
    model.fit(X_train)
    error = -model.score(X_test)
    errors.append(error)

plt.plot(encoding_dims, errors)
plt.xlabel('Encoding Dimension')
plt.ylabel('Reconstruction Error')
```

**Find "elbow"**: Point where increasing dimension doesn't help much

### 5. Comparison with PCA

```python
# Autoencoder
ae = Autoencoder(input_dim=100, encoding_dim=10, activation='tanh',
                 learning_rate=0.05, epochs=400)
ae.fit(X_train)
ae_error = -ae.score(X_test)

# PCA, evaluated on the SAME held-out rows (fit on train, score on test)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X_train)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
pca_error = np.mean((X_test - X_reconstructed_pca) ** 2)

print(f"Autoencoder MSE: {ae_error}")
print(f"PCA MSE: {pca_error}")
```

No sklearn? PCA's reconstruction is four lines of NumPy - this is exactly what
DEMO 1 in the Quick Start uses:

```python
mu = X_train.mean(axis=0)
_, _, Vt = np.linalg.svd(X_train - mu, full_matrices=False)
P = Vt[:10]                                        # top-10 principal directions
X_reconstructed_pca = (X_test - mu) @ P.T @ P + mu
pca_error = np.mean((X_test - X_reconstructed_pca) ** 2)
```

**When autoencoder wins**: Non-linear relationships in data
**When PCA wins**: Linear relationships, less data, faster needed

**Sanity check first**: on data that is genuinely linear, a *linear*
autoencoder should land close to the PCA floor - within 1.27x of it, and with
its subspace within 0.05 degrees of the true principal subspace, in DEMO 1. If
your autoencoder loses badly to PCA on linear data, the bug is in your setup
(dead ReLU code units, too few epochs, or a learning rate too small for
`input_dim`), not in the idea.

---

## Tips and Best Practices

### 1. Architecture Design

**Encoding Dimension**:
```
Start with: encoding_dim = input_dim / 5

Example:
- 100D input → Try 20D encoding first
- Adjust based on reconstruction quality
```

**Hidden Layers**:
```
Gradual compression:
Input → 0.7×input → 0.5×input → encoding

Example (100D input, 10D encoding):
100 → 70 → 50 → 30 → 10 (encoder)
10 → 30 → 50 → 70 → 100 (decoder)
```

### 2. Hyperparameter Tuning

**Learning Rate**:
```python
# Start with 0.01
learning_rate = 0.01

# If loss not decreasing: Increase slightly
# If loss oscillating: Decrease
```

**Batch Size**:
```
Small dataset (<1000): batch_size = 16-32
Medium (1000-10000): batch_size = 32-64
Large (>10000): batch_size = 64-128
```

**Epochs**:
```
Monitor loss curve:
- Still decreasing → Continue
- Flat for 20+ epochs → Stop
- Increasing → Reduce learning rate
```

### 3. Preventing Overfitting

**1. Use more data**
**2. Reduce model complexity** (smaller hidden layers)
**3. Add regularization** (L2, dropout - not in basic implementation)
**4. Early stopping** (monitor validation loss)

### 4. Debugging

**Loss not decreasing**:
- Check learning rate (try 10x smaller)
- Check data normalization
- Verify gradient computation

**Loss exploding**:
- Reduce learning rate
- Check for NaN in data
- Verify weight initialization

**Poor reconstruction**:
- Increase encoding dimension
- Add more hidden layers
- Train for more epochs
- Check activation functions

---

## Comparison: Autoencoder vs Other Methods

### Autoencoder vs PCA

| Aspect | Autoencoder | PCA |
|--------|------------|-----|
| **Type** | Non-linear | Linear |
| **Training** | Iterative, needs epochs | Closed-form, fast |
| **Flexibility** | Very flexible | Limited to linear |
| **Performance** | Better for complex data | Better for simple data |
| **Interpretability** | Lower | Higher (eigenvectors) |
| **Speed** | Slower | Faster |

**When to use what**:
- **Autoencoder**: Images, audio, complex patterns
- **PCA**: Quick analysis, linear data, interpretability needed

### Autoencoder vs t-SNE

| Aspect | Autoencoder | t-SNE |
|--------|------------|-------|
| **Purpose** | Compression + reconstruction | Visualization only |
| **Encoding New Data** | Yes (fast) | No (must rerun) |
| **Reconstruction** | Yes | No |
| **Computation** | Moderate | Slow |
| **Dimensions** | Any | Typically 2-3 |

**When to use what**:
- **Autoencoder**: Need to encode new data, need reconstruction
- **t-SNE**: Only for visualization, one-time analysis

### Types of Autoencoders

**Vanilla Autoencoder** (this implementation):
- Basic: Encode → Decode
- Use: Dimensionality reduction, feature learning

**Denoising Autoencoder**:
- Train on: Noisy input → Clean output
- Use: Remove noise, robust features

**Sparse Autoencoder**:
- Regularization: Encourage sparse activations
- Use: Feature learning, interpretability

**Variational Autoencoder (VAE)**:
- Probabilistic encoding
- Use: Generation, sampling

**Convolutional Autoencoder**:
- Uses CNN layers
- Use: Images specifically

---

## Simplification vs. canonical Autoencoders

This implementation is a complete, correct **vanilla autoencoder**: a fully
connected symmetric encoder/decoder trained by mini-batch gradient descent on
the squared reconstruction loss, with exact backpropagation (verified against
central differences to ~5e-10). Everything a modern deep-learning autoencoder
adds on top of that is deliberately left out, so the ~500 lines of the class
stay readable. Here is precisely what is missing and what it costs you.

### 1. Optimizer: plain SGD only

**Canonical**: Adam is the default in practice -
`m_t = β₁m_{t-1} + (1-β₁)g_t`, `v_t = β₂v_{t-1} + (1-β₂)g_t²`, then
`θ ← θ - α·m̂_t / (√v̂_t + ε)`, with β₁ = 0.9, β₂ = 0.999.

**Here**: `θ ← θ - α·g_t`. No momentum, no per-parameter step size, no learning
rate schedule.

**Consequence**: convergence is slower and much more sensitive to
`learning_rate`. DEMO 1 needs 1500 epochs to get within 1.27x of the PCA floor;
Adam would get there in a fraction of that. It also means you must tune
`learning_rate` per problem, and scale it down as `input_dim` grows (see the
note in the Loss Function section).

### 2. No regularization

**Canonical**: weight decay (`L += λ||W||²`), dropout, and for a *sparse*
autoencoder a KL penalty on the mean code activation,
`Σⱼ KL(ρ || ρ̂ⱼ)`, that pushes each unit's average activation toward a small ρ.

**Here**: none of these. The only capacity control is the width of the
bottleneck.

**Consequence**: with `encoding_dim` close to `input_dim` and enough epochs the
network can approach the identity function and "compress" nothing. Watch the
gap between train and test reconstruction MSE - USAGE EXAMPLE 6 prints exactly
that column.

### 3. No early stopping or validation monitoring

**Canonical**: hold out a validation split, stop when its loss stops improving,
restore the best weights.

**Here**: `fit()` runs exactly `epochs` iterations on exactly the data you gave
it, and never looks at a validation set.

**Consequence**: you choose `epochs` yourself. `history['loss']` is there so you
can see whether the curve is still descending or has gone flat.

### 4. Denoising is approximate

**Canonical** (Vincent et al., 2008): corrupt the input, but keep the **clean**
sample as the target - minimize `||x - g(f(x̃))||²` where `x̃ = x + noise`.

**Here**: `fit(X)` uses its single argument as both input and target, so
USAGE EXAMPLE 4 trains on the noisy data with the noisy data as target.

**Consequence**: it still denoises well (95.4% noise reduction on held-out rows
in USAGE EXAMPLE 4) because a 5-unit bottleneck physically cannot carry the
noise - but it is the "bottleneck denoises" mechanism, not the true
denoising-autoencoder objective. Implementing the real thing would mean a
second argument to `fit()`, changing the public API.

### 5. No tied weights

**Canonical**: many formulations set `W_decoder = W_encoderᵀ`, halving the
parameter count and regularizing the model.

**Here**: encoder and decoder weights are independent.

**Consequence**: about twice the parameters, and the linear case converges to
*a* basis of the principal subspace rather than an orthonormal one. This is why
DEMO 1 checks **principal angles** between subspaces rather than comparing
weight matrices directly.

### 6. No convolutional, recurrent, or variational variants

Convolutional autoencoders (weight-sharing over image patches), sequence
autoencoders, and VAEs (which encode a *distribution* `q(z|x) = N(μ(x), σ(x))`
and add a KL term to the loss) are all out of scope here. The Types of
Autoencoders table above sketches what each is for.

### What is NOT simplified

The parts that make it an autoencoder are exact:

- The forward pass, the loss, and every gradient - checked numerically.
- Xavier/Glorot uniform initialization with the correct `sqrt(6/(n_in+n_out))`.
- Mini-batch shuffling every epoch.
- The PCA-equivalence property of the linear case, demonstrated to within
  0.05 degrees of subspace agreement in DEMO 1.

---

## Mathematical Intuition

### Why Does It Work?

**Information Bottleneck**:
```
High-dimensional input contains:
- Signal (useful information)
- Noise (redundant, random)

Bottleneck forces network to:
- Keep only signal (necessary for reconstruction)
- Discard noise (doesn't fit through bottleneck)

Result: Compressed representation of essential information
```

**Learning Manifold Structure**:
```
High-dimensional data often lies on lower-dimensional manifold

Example: 3D object images (3 rotation angles)
- 100×100 pixel images = 10,000 dimensions
- Actually controlled by 3 angles
- Autoencoder learns this 3D manifold
```

### Connection to Information Theory

**Rate-Distortion Trade-off**:
```
Rate (compression): How few bits to encode
Distortion (error): How much information lost

Autoencoder solves:
minimize Distortion
subject to Rate ≤ encoding_dim

Optimal solution: Keep most important information
```

---

## Conclusion

Autoencoders are powerful unsupervised learning tools that learn efficient data representations by solving a seemingly simple task: reconstruct the input. This self-supervised approach forces the network to discover the underlying structure of data.

**Key Takeaways**:

1. **Compression**: Autoencoders learn to compress data to essential features
2. **Unsupervised**: No labels needed, learns from data itself
3. **Versatile**: Many applications (compression, denoising, anomaly detection)
4. **Non-linear**: Can capture complex patterns that PCA misses
5. **Scalable**: Works with any data type and dimension

**When to Use Autoencoders**:
- ✅ Need non-linear dimensionality reduction
- ✅ Have unlabeled data
- ✅ Want to learn features automatically
- ✅ Need to detect anomalies
- ✅ Want to compress/denoise data

**When to Use Alternatives**:
- ❌ Need interpretable components → Use PCA
- ❌ Only need visualization → Use t-SNE
- ❌ Have small dataset → Use PCA (more stable)
- ❌ Need very fast processing → Use PCA

### Next Steps

To deepen your understanding:

1. **Experiment**: Try different architectures (deep, wide, shallow)
2. **Visualize**: Plot encodings, see what the network learned
3. **Compare**: Test against PCA on your data
4. **Apply**: Use for real problems (anomaly detection, compression)
5. **Extend**: Learn about VAEs, denoising autoencoders, sparse autoencoders

Happy encoding! 🚀

---

## Further Reading

- **Original Paper**: "Reducing the Dimensionality of Data with Neural Networks" (Hinton & Salakhutdinov, 2006)
- **Denoising**: "Extracting and Composing Robust Features with Denoising Autoencoders" (Vincent et al., 2008)
- **VAE**: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- **Deep Learning Book**: Chapter 14 on Autoencoders (Goodfellow et al.)

