# Autoencoders from Scratch: A Comprehensive Guide

Welcome to the fascinating world of Autoencoders! 🧠 In this comprehensive guide, we'll explore autoencoders - neural networks that learn to compress and reconstruct data. Think of them as intelligent data compressors that learn the most important features automatically!

## Table of Contents
1. [What is an Autoencoder?](#what-is-an-autoencoder)
2. [How Autoencoders Work](#how-autoencoders-work)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)

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
MSE = (1/6) * [(2.5-2.4)² + (1.3-1.4)² + ... + (2.1-2.0)²]
    = 0.025
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
L(x, x̂) = (1/n) * Σᵢ₌₁ⁿ (xᵢ - x̂ᵢ)²

Or in matrix form:
L(X, X̂) = (1/m) * ||X - X̂||²_F

Where:
- m = number of samples
- n = number of features
- ||·||_F = Frobenius norm
```

**Objective**:
```
minimize L(θ) = minimize (1/m) * Σⱼ₌₁ᵐ ||x⁽ʲ⁾ - x̂⁽ʲ⁾||²

Where:
- θ = {W⁽¹⁾, b⁽¹⁾, ..., W⁽ᴸ⁾, b⁽ᴸ⁾} (all parameters)
- j = sample index
```

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

```
Epoch 1: Loss = 0.048
Epoch 10: Loss = 0.022
Epoch 20: Loss = 0.012
Epoch 50: Loss = 0.005  ← Converged!
```

### Step 8: Use Trained Model

**Encode new data**:
```python
new_sample = [1.05, 1.95, 1.55, 2.95, 2.45, 1.75, 2.15, 1.35]

encoded = autoencoder.encode(new_sample)
# Output: [0.68, 0.02]  ← Compressed to 2D!
```

**Reconstruct**:
```python
reconstructed = autoencoder.reconstruct(new_sample)
# Output: [1.02, 1.98, 1.53, 2.97, 2.43, 1.77, 2.13, 1.36]

reconstruction_error = MSE(new_sample, reconstructed)
# Output: 0.0042  ← Very good reconstruction!
```

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
autoencoder = Autoencoder(input_dim=30, encoding_dim=10)
autoencoder.fit(normal_transactions)

# Test on new transaction
new_transaction = [...]
error = autoencoder.reconstruction_error(new_transaction)

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

# Sample from latent space
random_encoding = np.random.randn(1, encoding_dim)

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
        weight = np.random.uniform(-limit, limit, (all_dims[i], all_dims[i+1]))
        bias = np.zeros(all_dims[i+1])
        
        self.weights.append(weight)
        self.biases.append(bias)
```

**What it does**:
- Creates symmetric encoder/decoder architecture
- Initializes weights using Xavier method (prevents gradient problems)
- Stores all weights and biases in lists

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
        
        # Activation function
        if i == self.n_layers - 1:
            a = self._activate(z, 'linear')  # Linear output
        else:
            a = self._activate(z, self.activation)  # ReLU/sigmoid/tanh
        
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
            delta *= self._activate_derivative(pre_activations[i-1], self.activation)
    
    return weight_gradients, bias_gradients
```

**What it does**:
- Computes gradients for all weights and biases
- Uses chain rule to propagate error backwards
- Key insight: Target is input itself (x̂ should match x)

**Chain rule in action**:
```
∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
           = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
```

### 4. Training Loop

```python
def fit(self, X):
    # Normalize data
    X_normalized = (X - self.data_min) / (self.data_max - self.data_min + 1e-8)
    
    for epoch in range(self.epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
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
    # Only forward through encoder layers
    activation = X_normalized
    n_encoder_layers = len(self.hidden_dims) + 1
    
    for i in range(n_encoder_layers):
        z = np.dot(activation, self.weights[i]) + self.biases[i]
        activation = self._activate(z, self.activation)
    
    return activation  # This is the compressed representation

def decode(self, encoded):
    # Only forward through decoder layers
    activation = encoded
    n_encoder_layers = len(self.hidden_dims) + 1
    
    for i in range(n_encoder_layers, self.n_layers):
        z = np.dot(activation, self.weights[i]) + self.biases[i]
        if i == self.n_layers - 1:
            activation = self._activate(z, 'linear')
        else:
            activation = self._activate(z, self.activation)
    
    # Denormalize
    decoded = activation * (self.data_max - self.data_min) + self.data_min
    return decoded
```

**What it does**:
- `encode()`: Compresses input to latent representation
- `decode()`: Reconstructs from latent representation
- Can be used separately after training

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
# For anomaly detection
errors = autoencoder.reconstruction_error(X_test)
threshold = np.percentile(errors_train, 95)
anomalies = errors > threshold

# Calculate metrics
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
```

**ROC Curve**: Plot TPR vs FPR at different thresholds

### 4. Compression vs Quality Trade-off

```python
encoding_dims = [2, 5, 10, 20, 50]
errors = []

for dim in encoding_dims:
    model = Autoencoder(input_dim=100, encoding_dim=dim)
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
ae = Autoencoder(input_dim=100, encoding_dim=10)
ae.fit(X_train)
ae_error = -ae.score(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_train)
X_reconstructed_pca = pca.inverse_transform(X_pca)
pca_error = np.mean((X_train - X_reconstructed_pca) ** 2)

print(f"Autoencoder MSE: {ae_error}")
print(f"PCA MSE: {pca_error}")
```

**When autoencoder wins**: Non-linear relationships in data
**When PCA wins**: Linear relationships, less data, faster needed

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

