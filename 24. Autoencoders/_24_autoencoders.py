import numpy as np

class Autoencoder:
    """
    Autoencoder Implementation from Scratch
    
    An Autoencoder is a neural network that learns to compress (encode) data into
    a lower-dimensional representation and then reconstruct (decode) it back.
    It's trained to minimize the reconstruction error.
    
    Key Idea: "Learn efficient data representations in an unsupervised manner"
    
    Use Cases:
    - Dimensionality Reduction: Alternative to PCA with non-linear transformations
    - Feature Learning: Extract meaningful features automatically
    - Denoising: Remove noise from images, audio, or signals
    - Anomaly Detection: Detect outliers based on reconstruction error
    - Data Compression: Compress data while preserving important information
    - Generative Modeling: Generate new similar data samples
    
    Key Concepts:
        Encoder: Compresses input to latent representation (bottleneck)
        Latent Space: Lower-dimensional compressed representation
        Decoder: Reconstructs input from latent representation
        Reconstruction Loss: Difference between input and output (MSE)
    """
    
    def __init__(self, input_dim, encoding_dim, hidden_dims=None, 
                 learning_rate=0.01, activation='relu', epochs=100,
                 batch_size=32, verbose=False):
        """
        Initialize the Autoencoder model
        
        Parameters:
        -----------
        input_dim : int
            Dimensionality of input data
            Example: 784 for 28x28 images (MNIST)
            
        encoding_dim : int
            Dimensionality of encoded (latent) representation
            - Should be smaller than input_dim for compression
            - Smaller values: More compression, may lose information
            - Larger values: Less compression, better reconstruction
            Typical: 10-30% of input_dim
            
        hidden_dims : list of int, optional
            Hidden layer dimensions for encoder
            - If None, uses single-layer encoder/decoder
            - Example: [128, 64] creates encoder with layers: input → 128 → 64 → encoding
            - Decoder is symmetric: encoding → 64 → 128 → output
            Default: None (single layer)
            
        learning_rate : float, default=0.01
            Learning rate for gradient descent
            - Higher values: Faster learning, may overshoot
            - Lower values: Slower but more stable
            Typical values: 0.001-0.1
            
        activation : str, default='relu'
            Activation function for hidden layers
            - 'relu': Rectified Linear Unit (default, most common)
            - 'sigmoid': Smooth, bounded [0, 1]
            - 'tanh': Smooth, bounded [-1, 1]
            Output layer always uses linear activation
            
        epochs : int, default=100
            Number of training iterations over entire dataset
            - More epochs: Better fit, risk overfitting
            - Fewer epochs: Faster training, may underfit
            Typical values: 50-500
            
        batch_size : int, default=32
            Number of samples per gradient update
            - Larger batches: More stable, need more memory
            - Smaller batches: More updates, may be noisy
            Typical values: 16, 32, 64, 128
            
        verbose : bool or int, default=False
            Print training progress
            - False: No output
            - True: Print every epoch
            - int: Print every N epochs
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims if hidden_dims else []
        self.learning_rate = learning_rate
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.weights = []
        self.biases = []
        self.history = {'loss': []}
        
        self._initialize_network()
    
    def _initialize_network(self):
        """
        Initialize network weights and biases using Xavier initialization
        
        Network Architecture:
        1. Encoder: input → hidden layers → encoding
        2. Decoder: encoding → hidden layers (reversed) → output
        
        Xavier initialization: Helps prevent vanishing/exploding gradients
        weights ~ N(0, sqrt(2 / (n_in + n_out)))
        """
        np.random.seed(42)
        
        # Build encoder architecture
        encoder_dims = [self.input_dim] + self.hidden_dims + [self.encoding_dim]
        
        # Build decoder architecture (symmetric to encoder)
        decoder_dims = [self.encoding_dim] + self.hidden_dims[::-1] + [self.input_dim]
        
        # Combine all layer dimensions
        all_dims = encoder_dims + decoder_dims[1:]
        
        # Initialize weights and biases for each layer
        for i in range(len(all_dims) - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (all_dims[i] + all_dims[i+1]))
            weight = np.random.uniform(-limit, limit, (all_dims[i], all_dims[i+1]))
            bias = np.zeros(all_dims[i+1])
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        self.n_layers = len(self.weights)
    
    def _activate(self, x, activation_type=None):
        """
        Apply activation function
        
        Parameters:
        -----------
        x : np.ndarray
            Input values
        activation_type : str, optional
            Type of activation. If None, uses self.activation
            
        Returns:
        --------
        activated : np.ndarray
            Activated values
        """
        if activation_type is None:
            activation_type = self.activation
        
        if activation_type == 'relu':
            return np.maximum(0, x)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation_type == 'tanh':
            return np.tanh(x)
        elif activation_type == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation: {activation_type}")
    
    def _activate_derivative(self, x, activation_type=None):
        """
        Compute derivative of activation function
        
        Used in backpropagation to compute gradients
        
        Parameters:
        -----------
        x : np.ndarray
            Pre-activation values
        activation_type : str, optional
            Type of activation
            
        Returns:
        --------
        derivative : np.ndarray
            Activation derivative
        """
        if activation_type is None:
            activation_type = self.activation
        
        if activation_type == 'relu':
            return (x > 0).astype(float)
        elif activation_type == 'sigmoid':
            s = self._activate(x, 'sigmoid')
            return s * (1 - s)
        elif activation_type == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif activation_type == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {activation_type}")
    
    def _forward_pass(self, X):
        """
        Forward propagation through the network
        
        Process:
        1. Input → Encoder → Latent representation
        2. Latent representation → Decoder → Output
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, input_dim)
            Input data
            
        Returns:
        --------
        activations : list of np.ndarray
            Activations at each layer (for backpropagation)
        pre_activations : list of np.ndarray
            Pre-activation values at each layer (for backpropagation)
        """
        activations = [X]
        pre_activations = []
        
        # Forward through all layers
        for i in range(self.n_layers):
            # Linear transformation: z = W·x + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            
            # Apply activation function
            # Use linear activation for output layer, specified activation for others
            if i == self.n_layers - 1:
                a = self._activate(z, 'linear')
            else:
                a = self._activate(z, self.activation)
            
            activations.append(a)
        
        return activations, pre_activations
    
    def _backward_pass(self, X, activations, pre_activations):
        """
        Backward propagation to compute gradients
        
        Backpropagation Algorithm:
        1. Compute output layer error: δ = (output - target)
        2. Propagate error backwards through layers
        3. Compute gradients for weights and biases
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (target is same as input for autoencoder)
        activations : list of np.ndarray
            Activations from forward pass
        pre_activations : list of np.ndarray
            Pre-activation values from forward pass
            
        Returns:
        --------
        weight_gradients : list of np.ndarray
            Gradients for weights
        bias_gradients : list of np.ndarray
            Gradients for biases
        """
        n_samples = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        
        # Output layer error: dL/dz = (prediction - target) for MSE loss
        output = activations[-1]
        delta = (output - X)  # X is the target for autoencoder
        
        # Backpropagate through all layers
        for i in range(self.n_layers - 1, -1, -1):
            # Compute gradients
            weight_grad = np.dot(activations[i].T, delta) / n_samples
            bias_grad = np.mean(delta, axis=0)
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            # Propagate error to previous layer (if not input layer)
            if i > 0:
                # dL/da_prev = delta · W^T
                delta = np.dot(delta, self.weights[i].T)
                # dL/dz_prev = dL/da_prev · activation'(z_prev)
                delta *= self._activate_derivative(pre_activations[i-1], 
                                                   self.activation if i > 1 else self.activation)
        
        return weight_gradients, bias_gradients
    
    def _update_parameters(self, weight_gradients, bias_gradients):
        """
        Update weights and biases using gradient descent
        
        Update rule: θ = θ - learning_rate * gradient
        
        Parameters:
        -----------
        weight_gradients : list of np.ndarray
            Gradients for weights
        bias_gradients : list of np.ndarray
            Gradients for biases
        """
        for i in range(self.n_layers):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def fit(self, X):
        """
        Train the autoencoder
        
        Training Process:
        1. For each epoch:
           a. Shuffle data
           b. For each mini-batch:
              - Forward pass: compute predictions
              - Compute reconstruction loss
              - Backward pass: compute gradients
              - Update weights and biases
           c. Track average loss for epoch
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, input_dim)
            Training data
            Note: For autoencoders, there's no separate target y
            The network learns to reconstruct X from X
            
        Returns:
        --------
        self : Autoencoder
            Fitted model
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {X.shape[1]}")
        
        # Normalize data to [0, 1] range for better training
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        X_normalized = (X - self.data_min) / (self.data_max - self.data_min + 1e-8)
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_normalized[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                
                # Forward pass
                activations, pre_activations = self._forward_pass(X_batch)
                
                # Compute loss (Mean Squared Error)
                reconstruction = activations[-1]
                batch_loss = np.mean((reconstruction - X_batch) ** 2)
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass
                weight_grads, bias_grads = self._backward_pass(X_batch, activations, 
                                                                pre_activations)
                
                # Update parameters
                self._update_parameters(weight_grads, bias_grads)
            
            # Track average loss
            avg_loss = epoch_loss / n_batches
            self.history['loss'].append(avg_loss)
            
            # Print progress
            if self.verbose:
                if isinstance(self.verbose, bool) or (epoch + 1) % self.verbose == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        return self
    
    def encode(self, X):
        """
        Encode data to latent representation
        
        This compresses the input to the lower-dimensional encoding.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, input_dim)
            Data to encode
            
        Returns:
        --------
        encoded : np.ndarray, shape (n_samples, encoding_dim)
            Latent representation
        """
        X = np.array(X, dtype=float)
        
        # Normalize
        X_normalized = (X - self.data_min) / (self.data_max - self.data_min + 1e-8)
        
        # Forward pass through encoder only
        activation = X_normalized
        n_encoder_layers = len(self.hidden_dims) + 1  # Hidden layers + encoding layer
        
        for i in range(n_encoder_layers):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self._activate(z, self.activation)
        
        return activation
    
    def decode(self, encoded):
        """
        Decode latent representation back to original space
        
        This reconstructs the input from the compressed encoding.
        
        Parameters:
        -----------
        encoded : np.ndarray, shape (n_samples, encoding_dim)
            Latent representation
            
        Returns:
        --------
        decoded : np.ndarray, shape (n_samples, input_dim)
            Reconstructed data (denormalized)
        """
        encoded = np.array(encoded, dtype=float)
        
        # Forward pass through decoder only
        activation = encoded
        n_encoder_layers = len(self.hidden_dims) + 1
        
        for i in range(n_encoder_layers, self.n_layers):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            if i == self.n_layers - 1:
                activation = self._activate(z, 'linear')
            else:
                activation = self._activate(z, self.activation)
        
        # Denormalize
        decoded = activation * (self.data_max - self.data_min + 1e-8) + self.data_min
        
        return decoded
    
    def reconstruct(self, X):
        """
        Reconstruct input data (encode then decode)
        
        This is equivalent to a forward pass through entire autoencoder.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, input_dim)
            Data to reconstruct
            
        Returns:
        --------
        reconstructed : np.ndarray, shape (n_samples, input_dim)
            Reconstructed data
        """
        encoded = self.encode(X)
        reconstructed = self.decode(encoded)
        return reconstructed
    
    def reconstruction_error(self, X):
        """
        Compute reconstruction error for each sample
        
        Useful for anomaly detection: samples with high reconstruction
        error are likely anomalies.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, input_dim)
            Data to evaluate
            
        Returns:
        --------
        errors : np.ndarray, shape (n_samples,)
            Reconstruction error for each sample (MSE)
        """
        X = np.array(X, dtype=float)
        reconstructed = self.reconstruct(X)
        errors = np.mean((X - reconstructed) ** 2, axis=1)
        return errors
    
    def score(self, X):
        """
        Compute average reconstruction score (negative MSE)
        
        Parameters:
        -----------
        X : np.ndarray
            Data to evaluate
            
        Returns:
        --------
        score : float
            Negative mean reconstruction error (higher is better)
        """
        errors = self.reconstruction_error(X)
        return -np.mean(errors)
    
    def get_compression_ratio(self):
        """
        Get compression ratio achieved by the autoencoder
        
        Returns:
        --------
        ratio : float
            Compression ratio (input_dim / encoding_dim)
        """
        return self.input_dim / self.encoding_dim


"""
USAGE EXAMPLE 1: Simple Dimensionality Reduction

import numpy as np

# Generate high-dimensional data with underlying structure
np.random.seed(42)
n_samples = 1000

# Create data with 2 underlying factors but 20 dimensions
factor1 = np.random.randn(n_samples, 1)
factor2 = np.random.randn(n_samples, 1)

# Each dimension is a combination of the factors plus noise
X = np.hstack([
    factor1 * np.random.randn(1, 10) + factor2 * np.random.randn(1, 10),
    np.random.randn(n_samples, 10) * 0.1  # Some noise dimensions
])

# Split train/test
X_train, X_test = X[:800], X[800:]

# Create and train autoencoder
autoencoder = Autoencoder(
    input_dim=20,
    encoding_dim=2,  # Compress to 2 dimensions
    learning_rate=0.01,
    epochs=100,
    verbose=10
)
autoencoder.fit(X_train)

print(f"\nCompression ratio: {autoencoder.get_compression_ratio():.1f}x")

# Encode data to 2D representation
encoded = autoencoder.encode(X_test)
print(f"Original shape: {X_test.shape}, Encoded shape: {encoded.shape}")

# Reconstruct data
reconstructed = autoencoder.reconstruct(X_test)
reconstruction_loss = np.mean((X_test - reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_loss:.6f}")

# Compare original vs reconstructed
print("\nSample Comparison (first 5 features):")
print("Original:     ", X_test[0, :5])
print("Reconstructed:", reconstructed[0, :5])
"""

"""
USAGE EXAMPLE 2: Anomaly Detection

import numpy as np

# Generate normal data (Gaussian)
np.random.seed(42)
X_normal = np.random.randn(500, 10)

# Generate anomalies (different distribution)
X_anomaly = np.random.uniform(-5, 5, (50, 10))

# Train autoencoder on normal data only
autoencoder = Autoencoder(
    input_dim=10,
    encoding_dim=3,
    hidden_dims=[8, 5],
    learning_rate=0.01,
    epochs=100,
    verbose=False
)
autoencoder.fit(X_normal)

# Compute reconstruction errors
normal_errors = autoencoder.reconstruction_error(X_normal)
anomaly_errors = autoencoder.reconstruction_error(X_anomaly)

print("\nAnomaly Detection Results:")
print("="*60)
print(f"Normal data - Mean error: {np.mean(normal_errors):.6f}")
print(f"Normal data - Max error:  {np.max(normal_errors):.6f}")
print(f"\nAnomaly data - Mean error: {np.mean(anomaly_errors):.6f}")
print(f"Anomaly data - Min error:  {np.min(anomaly_errors):.6f}")

# Set threshold (e.g., 95th percentile of normal errors)
threshold = np.percentile(normal_errors, 95)
print(f"\nAnomaly threshold: {threshold:.6f}")

# Detect anomalies in combined dataset
X_combined = np.vstack([X_normal[:100], X_anomaly])
y_true = np.array([0] * 100 + [1] * 50)  # 0=normal, 1=anomaly

errors = autoencoder.reconstruction_error(X_combined)
y_pred = (errors > threshold).astype(int)

# Calculate metrics
true_positives = np.sum((y_pred == 1) & (y_true == 1))
false_positives = np.sum((y_pred == 1) & (y_true == 0))
true_negatives = np.sum((y_pred == 0) & (y_true == 0))
false_negatives = np.sum((y_pred == 0) & (y_true == 1))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"\nAnomaly Detection Performance:")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.4f}")
"""

"""
USAGE EXAMPLE 3: Feature Learning and Visualization

import numpy as np

# Generate synthetic image-like data (8x8 = 64 pixels)
np.random.seed(42)
n_samples = 1000

# Create patterns: horizontal lines, vertical lines, diagonal lines
patterns = []
for i in range(n_samples):
    pattern_type = np.random.choice([0, 1, 2])
    img = np.zeros((8, 8))
    
    if pattern_type == 0:  # Horizontal line
        row = np.random.randint(0, 8)
        img[row, :] = 1
    elif pattern_type == 1:  # Vertical line
        col = np.random.randint(0, 8)
        img[:, col] = 1
    else:  # Diagonal line
        np.fill_diagonal(img, 1)
    
    # Add noise
    img += np.random.randn(8, 8) * 0.1
    patterns.append(img.flatten())

X = np.array(patterns)

# Train autoencoder to learn features
autoencoder = Autoencoder(
    input_dim=64,
    encoding_dim=3,  # Learn 3 features
    hidden_dims=[32, 16],
    learning_rate=0.01,
    epochs=150,
    verbose=25
)
autoencoder.fit(X)

# Encode to feature space
features = autoencoder.encode(X)

print(f"\nLearned {features.shape[1]} features from {X.shape[1]}-dim data")
print(f"Feature space statistics:")
print(f"  Mean: {np.mean(features, axis=0)}")
print(f"  Std:  {np.std(features, axis=0)}")

# Test reconstruction quality
test_samples = X[:10]
reconstructed = autoencoder.reconstruct(test_samples)
mse_per_sample = np.mean((test_samples - reconstructed) ** 2, axis=1)

print(f"\nReconstruction quality (first 10 samples):")
for i, mse in enumerate(mse_per_sample):
    print(f"Sample {i+1}: MSE = {mse:.6f}")
"""

"""
USAGE EXAMPLE 4: Denoising Autoencoder

import numpy as np

# Generate clean data
np.random.seed(42)
n_samples = 500
t = np.linspace(0, 4*np.pi, 50)
X_clean = np.array([np.sin(t + phase) for phase in np.random.uniform(0, 2*np.pi, n_samples)])

# Add noise
noise_level = 0.3
X_noisy = X_clean + np.random.randn(*X_clean.shape) * noise_level

# Train on noisy data (learns to denoise)
denoiser = Autoencoder(
    input_dim=50,
    encoding_dim=5,
    hidden_dims=[30, 15],
    learning_rate=0.01,
    epochs=200,
    verbose=50
)
denoiser.fit(X_noisy)

# Denoise test samples
X_test_clean = X_clean[400:410]
X_test_noisy = X_noisy[400:410]
X_denoised = denoiser.reconstruct(X_test_noisy)

# Calculate improvement
noise_before = np.mean((X_test_noisy - X_test_clean) ** 2)
noise_after = np.mean((X_denoised - X_test_clean) ** 2)

print("\nDenoising Performance:")
print("="*60)
print(f"MSE before denoising: {noise_before:.6f}")
print(f"MSE after denoising:  {noise_after:.6f}")
print(f"Noise reduction:      {(1 - noise_after/noise_before)*100:.1f}%")

# Show sample
sample_idx = 0
print(f"\nSample Signal (first 10 points):")
print(f"Clean:    {X_test_clean[sample_idx, :10]}")
print(f"Noisy:    {X_test_noisy[sample_idx, :10]}")
print(f"Denoised: {X_denoised[sample_idx, :10]}")
"""

"""
USAGE EXAMPLE 5: Comparing Different Architectures

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(800, 100)

X_train, X_test = X[:600], X[600:]

# Test different architectures
architectures = [
    {'name': 'Shallow', 'hidden_dims': None, 'encoding_dim': 10},
    {'name': 'Deep-Narrow', 'hidden_dims': [80, 60, 40, 20], 'encoding_dim': 10},
    {'name': 'Wide-Bottleneck', 'hidden_dims': [120, 100], 'encoding_dim': 5},
    {'name': 'Balanced', 'hidden_dims': [70, 40], 'encoding_dim': 10},
]

print("Comparing Autoencoder Architectures:")
print("="*80)
print(f"{'Architecture':<20} {'Encoding Dim':<15} {'Train MSE':<15} {'Test MSE':<15}")
print("-"*80)

for arch in architectures:
    model = Autoencoder(
        input_dim=100,
        encoding_dim=arch['encoding_dim'],
        hidden_dims=arch['hidden_dims'],
        learning_rate=0.01,
        epochs=100,
        verbose=False
    )
    model.fit(X_train)
    
    train_mse = -model.score(X_train)
    test_mse = -model.score(X_test)
    
    print(f"{arch['name']:<20} {arch['encoding_dim']:<15} {train_mse:<15.6f} {test_mse:<15.6f}")

print("\nObservations:")
print("- Shallow: Fast training, may underfit complex patterns")
print("- Deep-Narrow: Better feature learning, needs more data")
print("- Wide-Bottleneck: Strong compression, may lose information")
print("- Balanced: Good trade-off for most cases")
"""

"""
USAGE EXAMPLE 6: Effect of Encoding Dimension

import numpy as np

# Generate data with structure
np.random.seed(42)
n_samples = 600
n_features = 50

# Data has 5 underlying factors
factors = np.random.randn(n_samples, 5)
X = np.dot(factors, np.random.randn(5, n_features))
X += np.random.randn(n_samples, n_features) * 0.1

X_train, X_test = X[:500], X[500:]

# Test different encoding dimensions
encoding_dims = [2, 3, 5, 8, 10, 15, 20]

print("Effect of Encoding Dimension:")
print("="*90)
print(f"{'Encoding Dim':<15} {'Compression':<15} {'Train MSE':<15} {'Test MSE':<15} {'Overfit':<15}")
print("-"*90)

for enc_dim in encoding_dims:
    model = Autoencoder(
        input_dim=50,
        encoding_dim=enc_dim,
        hidden_dims=[30],
        learning_rate=0.01,
        epochs=100,
        verbose=False
    )
    model.fit(X_train)
    
    compression = 50 / enc_dim
    train_mse = -model.score(X_train)
    test_mse = -model.score(X_test)
    overfit = test_mse - train_mse
    
    print(f"{enc_dim:<15} {compression:<15.1f}x {train_mse:<15.6f} {test_mse:<15.6f} {overfit:<15.6f}")

print("\nOptimal encoding dimension balances:")
print("- High compression (lower dim) vs Information preservation (higher dim)")
print("- Training error vs Generalization")
print("For this data (5 factors), encoding_dim=5 should be optimal")
"""

"""
USAGE EXAMPLE 7: Effect of Activation Function

import numpy as np

# Generate data with different ranges
np.random.seed(42)
X = np.random.uniform(-2, 2, (600, 30))

X_train, X_test = X[:500], X[500:]

# Test different activation functions
activations = ['relu', 'sigmoid', 'tanh']

print("Comparing Activation Functions:")
print("="*80)
print(f"{'Activation':<15} {'Train MSE':<15} {'Test MSE':<15} {'Time (epochs)':<20}")
print("-"*80)

for activation in activations:
    model = Autoencoder(
        input_dim=30,
        encoding_dim=5,
        hidden_dims=[20, 10],
        activation=activation,
        learning_rate=0.01,
        epochs=100,
        verbose=False
    )
    model.fit(X_train)
    
    train_mse = -model.score(X_train)
    test_mse = -model.score(X_test)
    
    print(f"{activation:<15} {train_mse:<15.6f} {test_mse:<15.6f} {'100':<20}")

print("\nActivation Function Characteristics:")
print("- ReLU: Fast, effective, prone to 'dying ReLU' problem")
print("- Sigmoid: Smooth, bounded [0,1], can saturate")
print("- Tanh: Smooth, bounded [-1,1], zero-centered")
"""

"""
USAGE EXAMPLE 8: Learning Rate Comparison

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(500, 40)

X_train, X_test = X[:400], X[400:]

# Test different learning rates
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]

print("Effect of Learning Rate:")
print("="*80)
print(f"{'Learning Rate':<15} {'Final Train MSE':<20} {'Final Test MSE':<20}")
print("-"*80)

for lr in learning_rates:
    model = Autoencoder(
        input_dim=40,
        encoding_dim=8,
        learning_rate=lr,
        epochs=100,
        verbose=False
    )
    model.fit(X_train)
    
    train_mse = -model.score(X_train)
    test_mse = -model.score(X_test)
    
    print(f"{lr:<15.3f} {train_mse:<20.6f} {test_mse:<20.6f}")

print("\nLearning Rate Guidelines:")
print("- Too low (0.001): Slow convergence, may need more epochs")
print("- Optimal (0.01): Good balance of speed and stability")
print("- Too high (0.1): Fast initially but unstable, may diverge")
"""

"""
USAGE EXAMPLE 9: Transfer Learning - Pre-trained Features

import numpy as np

# Generate source domain data
np.random.seed(42)
X_source = np.random.randn(1000, 50)

# Train autoencoder on source domain
pretrained = Autoencoder(
    input_dim=50,
    encoding_dim=10,
    hidden_dims=[30, 20],
    learning_rate=0.01,
    epochs=150,
    verbose=False
)
pretrained.fit(X_source)

print("Pre-trained Autoencoder:")
print(f"Trained on {X_source.shape[0]} samples")
print(f"Learned to encode {X_source.shape[1]}D → {pretrained.encoding_dim}D")

# Generate target domain data (similar but different)
X_target = np.random.randn(200, 50) + 0.5  # Shifted distribution

# Extract features using pre-trained encoder
features_target = pretrained.encode(X_target)

print(f"\nExtracted features for {X_target.shape[0]} target samples")
print(f"Feature shape: {features_target.shape}")
print(f"Feature statistics:")
print(f"  Mean: {np.mean(features_target, axis=0)[:5]}...")
print(f"  Std:  {np.std(features_target, axis=0)[:5]}...")

# These features can now be used for downstream tasks
# (classification, clustering, etc.)
"""

"""
USAGE EXAMPLE 10: Real-World - Credit Card Fraud Detection

import numpy as np

# Simulated credit card transaction data
# Features: [amount, time_of_day, merchant_type, location_distance, 
#            frequency_score, avg_transaction, num_recent_trans, ...]

np.random.seed(42)

# Normal transactions (majority)
n_normal = 900
X_normal = np.column_stack([
    np.random.exponential(50, n_normal),           # Amount
    np.random.uniform(0, 24, n_normal),            # Time
    np.random.randint(0, 10, n_normal),            # Merchant type
    np.random.exponential(10, n_normal),           # Distance
    np.random.uniform(0.5, 1.0, n_normal),         # Frequency score
    np.random.uniform(40, 80, n_normal),           # Avg transaction
    np.random.randint(5, 30, n_normal),            # Recent trans
    np.random.randn(n_normal)                      # Feature 8
])

# Fraudulent transactions (anomalies)
n_fraud = 100
X_fraud = np.column_stack([
    np.random.exponential(200, n_fraud),           # High amounts
    np.random.choice([2, 3, 4, 23], n_fraud),      # Unusual times
    np.random.randint(8, 10, n_fraud),             # Risky merchants
    np.random.exponential(50, n_fraud),            # Far locations
    np.random.uniform(0.0, 0.3, n_fraud),          # Low frequency
    np.random.uniform(20, 50, n_fraud),            # Different avg
    np.random.randint(1, 5, n_fraud),              # Few recent
    np.random.randn(n_fraud) * 2                   # Feature 8
])

# Split normal data into train/validation
X_train = X_normal[:700]
X_val = X_normal[700:900]

# Train autoencoder on normal transactions only
fraud_detector = Autoencoder(
    input_dim=8,
    encoding_dim=3,
    hidden_dims=[6, 4],
    learning_rate=0.01,
    epochs=200,
    batch_size=32,
    verbose=50
)

fraud_detector.fit(X_train)

# Compute reconstruction errors on validation set
val_errors = fraud_detector.reconstruction_error(X_val)

# Set threshold at 99th percentile of normal errors
threshold = np.percentile(val_errors, 99)
print(f"\nFraud detection threshold: {threshold:.6f}")

# Test on mixed data
X_test = np.vstack([X_val[:50], X_fraud[:50]])
y_true = np.array([0] * 50 + [1] * 50)  # 0=normal, 1=fraud

test_errors = fraud_detector.reconstruction_error(X_test)
y_pred = (test_errors > threshold).astype(int)

# Calculate metrics
tp = np.sum((y_pred == 1) & (y_true == 1))
fp = np.sum((y_pred == 1) & (y_true == 0))
tn = np.sum((y_pred == 0) & (y_true == 0))
fn = np.sum((y_pred == 0) & (y_true == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / len(y_true)

print("\nFraud Detection Performance:")
print("="*60)
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%} (of flagged transactions, how many are fraud)")
print(f"Recall:    {recall:.2%} (of fraud cases, how many we catch)")
print(f"F1 Score:  {f1:.4f}")

print(f"\nConfusion Matrix:")
print(f"                Predicted Normal  Predicted Fraud")
print(f"Actual Normal         {tn:5d}            {fp:5d}")
print(f"Actual Fraud          {fn:5d}            {tp:5d}")

print(f"\nError Statistics:")
print(f"Normal transactions - Mean error: {np.mean(test_errors[y_true==0]):.6f}")
print(f"Fraud transactions  - Mean error: {np.mean(test_errors[y_true==1]):.6f}")

# Analyze specific fraud case
fraud_idx = np.where(y_true == 1)[0][0]
fraud_transaction = X_test[fraud_idx]
fraud_error = test_errors[fraud_idx]

print(f"\nExample Fraud Detection:")
print(f"Transaction features: {fraud_transaction[:5]}...")
print(f"Reconstruction error: {fraud_error:.6f}")
print(f"Threshold:           {threshold:.6f}")
print(f"Result: {'FRAUD DETECTED' if fraud_error > threshold else 'Normal'}")
"""
