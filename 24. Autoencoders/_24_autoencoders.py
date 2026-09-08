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

    Core Formulas (all ASCII, all implemented literally in the code below):
        Forward, layer l:   z^(l) = a^(l-1) . W^(l) + b^(l),  a^(l) = f_l(z^(l))
                            f_l = activation for hidden layers,
                            f_l = bottleneck_activation at the code layer,
                            f_l = identity at the output layer.

        Optimized loss:     L = (1/(2n)) * sum_i ||x_i - xhat_i||^2
                            (n = samples in the mini-batch)

        Output delta:       delta^(L) = xhat - x            (_backward_pass)
        Hidden recursion:   delta^(l) = (delta^(l+1) . W^(l+1)^T) * f_l'(z^(l))
        Parameter grads:    dL/dW^(l) = (a^(l-1))^T . delta^(l) / n
                            dL/db^(l) = mean(delta^(l), axis=0)
        Update:             theta <- theta - learning_rate * dL/dtheta

        Xavier-uniform init: W ~ Uniform(-limit, +limit),
                             limit = sqrt(6 / (n_in + n_out))

    Note on the two "losses":
        The gradients above are the exact gradients of L = (1/(2n))*sum||x-xhat||^2,
        but history['loss'] records the PER-ELEMENT MSE, mean((xhat - x)^2), which is
        the same quantity scaled by 2/input_dim. The constant is absorbed into
        learning_rate, so nothing is wrong - but it does mean the effective step size
        grows with input_dim, and learning_rate should be scaled down for wide inputs.

    Dying-bottleneck warning:
        fit() min-max scales X into [0, 1], biases start at zero and Xavier weights push
        roughly half the pre-activations negative. With the default activation='relu' a
        code unit whose pre-activation is negative for every sample receives exactly zero
        gradient forever, so small bottlenecks frequently lose 30-100% of their units and
        the model degenerates into a mean-predictor. Use activation='tanh' (or 'linear',
        or bottleneck_activation='linear') whenever encoding_dim is small.

    Simplifications vs. canonical deep-learning autoencoders:
        This class implements plain mini-batch gradient descent on a fully connected,
        symmetric encoder/decoder. Deliberately NOT implemented (see the
        "Simplification vs. canonical Autoencoders" section of _24_autoencoders.md):
        - No momentum / Adam / RMSProp; only vanilla SGD with a fixed learning rate.
        - No weight decay, dropout, sparsity (KL) penalty, or tied weights.
        - No early stopping or validation monitoring inside fit().
        - No convolutional, recurrent, or variational (VAE) variants.
        - The denoising variant is approximated by training on noisy inputs rather
          than the canonical (noisy input -> clean target) pairing, because fit()
          takes a single X and uses it as its own target.
    """
    
    def __init__(self, input_dim, encoding_dim, hidden_dims=None,
                 learning_rate=0.01, activation='relu', epochs=100,
                 batch_size=32, verbose=False, bottleneck_activation=None,
                 random_state=42):
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
            - Example: [128, 64] creates encoder with layers: input -> 128 -> 64 -> encoding
            - Decoder is symmetric: encoding -> 64 -> 128 -> output
            Default: None (single layer)
            
        learning_rate : float, default=0.01
            Learning rate for gradient descent
            - Higher values: Faster learning, may overshoot
            - Lower values: Slower but more stable
            Typical values: 0.001-0.1
            
        activation : str, default='relu'
            Activation function for hidden layers AND for the bottleneck
            (unless bottleneck_activation overrides the bottleneck)
            - 'relu': Rectified Linear Unit (default, most common).
              Warning: with a small encoding_dim, ReLU code units routinely die
              (see "Dying-bottleneck warning" in the class docstring).
            - 'sigmoid': Smooth, bounded [0, 1]
            - 'tanh': Smooth, bounded [-1, 1]; the safest default for a narrow bottleneck
            - 'linear': No nonlinearity; makes the network equivalent to PCA
              (it then spans the same subspace as the top-encoding_dim principal
              components, which is exactly what DEMO 1 in __main__ measures)
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

        bottleneck_activation : str or None, default=None
            Activation applied to the encoding (bottleneck) layer only
            - None: use `activation` there too (the historical behaviour)
            - 'linear': keep the code unbounded and un-clipped. This is the
              single most effective cure for dead ReLU code units, and it is
              what makes a 'relu' network still able to place its code anywhere
              in R^encoding_dim
            - 'tanh' / 'sigmoid': bound the code to [-1, 1] / [0, 1]
            Typical: None, or 'linear' when encoding_dim is small

        random_state : int or None, default=42
            Seed for this model's PRIVATE random generator (weight init and the
            per-epoch mini-batch shuffle)
            - Uses np.random.RandomState internally, so it never touches or
              disturbs the global NumPy RNG
            - None: draw a fresh, unreproducible seed from the OS
            Typical: any fixed int for reproducibility
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims if hidden_dims else []
        self.learning_rate = learning_rate
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.bottleneck_activation = bottleneck_activation
        self.random_state = random_state

        # Private RNG: seeding the GLOBAL numpy RNG from a constructor would
        # silently clobber the caller's own random stream.
        self._rng = np.random.RandomState(random_state)

        self.weights = []
        self.biases = []
        self.history = {'loss': []}

        # Set by fit(); used as the "is this model fitted?" flag.
        self.data_min = None
        self.data_max = None

        self._initialize_network()
    
    def _initialize_network(self):
        """
        Initialize network weights and biases using Xavier initialization
        
        Network Architecture:
        1. Encoder: input -> hidden layers -> encoding
        2. Decoder: encoding -> hidden layers (reversed) -> output

        Xavier initialization: Helps prevent vanishing/exploding gradients
        weights ~ Uniform(-sqrt(6 / (n_in + n_out)), +sqrt(6 / (n_in + n_out)))
        biases  ~ 0
        """
        # Build encoder architecture
        encoder_dims = [self.input_dim] + self.hidden_dims + [self.encoding_dim]
        
        # Build decoder architecture (symmetric to encoder)
        decoder_dims = [self.encoding_dim] + self.hidden_dims[::-1] + [self.input_dim]
        
        # Combine all layer dimensions
        all_dims = encoder_dims + decoder_dims[1:]
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        for i in range(len(all_dims) - 1):
            # Xavier initialization: limit = sqrt(6 / (n_in + n_out))
            limit = np.sqrt(6 / (all_dims[i] + all_dims[i+1]))
            weight = self._rng.uniform(-limit, limit, (all_dims[i], all_dims[i+1]))
            bias = np.zeros(all_dims[i+1])

            self.weights.append(weight)
            self.biases.append(bias)

        self.n_layers = len(self.weights)
        # Index of the bottleneck layer inside self.weights is n_encoder_layers - 1
        self.n_encoder_layers = len(self.hidden_dims) + 1
    
    def _layer_activation(self, layer_index):
        """
        Return the name of the activation function used by one layer

        This is the single place that decides "which f_l does layer l use?", so
        _forward_pass, _backward_pass, encode() and decode() can never disagree.

        Layer roles (indices into self.weights):
            0 .. n_encoder_layers-2 : encoder hidden layers -> self.activation
            n_encoder_layers-1      : the bottleneck (code) -> bottleneck_activation
                                                               (falls back to activation)
            n_encoder_layers .. L-2 : decoder hidden layers -> self.activation
            L-1                     : output layer          -> 'linear', always

        Parameters:
        -----------
        layer_index : int
            Index of the layer in self.weights / self.biases

        Returns:
        --------
        activation_type : str
            One of 'relu', 'sigmoid', 'tanh', 'linear'
        """
        if layer_index == self.n_layers - 1:
            return 'linear'                       # reconstruction must be unbounded
        if layer_index == self.n_encoder_layers - 1 and self.bottleneck_activation:
            return self.bottleneck_activation     # optional override at the code layer
        return self.activation

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
        1. Input -> Encoder -> Latent representation
        2. Latent representation -> Decoder -> Output

        Implements, for every layer l:
            z^(l) = a^(l-1) . W^(l) + b^(l)
            a^(l) = f_l(z^(l))          with f_l given by _layer_activation(l)

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
            # Linear transformation: z = a_prev . W + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)

            # Apply activation function: linear on the output layer,
            # bottleneck_activation on the code layer, self.activation elsewhere
            a = self._activate(z, self._layer_activation(i))

            activations.append(a)
        
        return activations, pre_activations
    
    def _backward_pass(self, X, activations, pre_activations):
        """
        Backward propagation to compute gradients
        
        Backpropagation Algorithm:
        1. Compute output layer error: delta^(L) = (output - target)
        2. Propagate error backwards through layers
        3. Compute gradients for weights and biases

        These are the EXACT gradients of
            L = (1/(2n)) * sum_i ||x_i - xhat_i||^2
        (verified against central differences to ~5e-10). Note that the number
        recorded in history['loss'] is instead the per-element MSE,
        mean((xhat - x)^2) = (2/input_dim) * L; the constant factor is absorbed
        into learning_rate.

        Formulas implemented below:
            delta^(L) = xhat - x
            dL/dW^(l) = (a^(l-1))^T . delta^(l) / n
            dL/db^(l) = mean(delta^(l), axis=0)
            delta^(l) = (delta^(l+1) . W^(l+1)^T) * f_l'(z^(l))

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
                # dL/da_prev = delta . W^T
                delta = np.dot(delta, self.weights[i].T)
                # dL/dz_prev = dL/da_prev * f'(z_prev), with the SAME activation
                # that _forward_pass used on that layer
                delta = delta * self._activate_derivative(
                    pre_activations[i-1], self._layer_activation(i-1))

        return weight_gradients, bias_gradients
    
    def _update_parameters(self, weight_gradients, bias_gradients):
        """
        Update weights and biases using gradient descent
        
        Update rule: theta = theta - learning_rate * gradient
        
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

        Notes:
        ------
        - X is min-max scaled to [0, 1] per feature before training:
              X_norm = (X - data_min) / (data_max - data_min + 1e-8)
          The per-feature `data_min` / `data_max` are stored on the model and
          reused by encode(), decode(), reconstruct() and score(), so you do NOT
          need to scale your data yourself - and if you already standardized it,
          it will simply be rescaled again.
        - history['loss'] is the per-element MSE on that NORMALIZED [0, 1] scale,
          so it is not comparable to reconstruction_error(), which reports MSE on
          the ORIGINAL scale (decode() denormalizes before returning).
        - Weights are NOT re-initialized here: calling fit() a second time
          continues training from the current weights (a warm start). Only the
          loss history is reset. Build a new Autoencoder for a clean restart.
        """
        X = np.atleast_2d(np.array(X, dtype=float))
        n_samples = X.shape[0]

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {X.shape[1]}")

        # Normalize data to [0, 1] range for better training
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        X_normalized = (X - self.data_min) / (self.data_max - self.data_min + 1e-8)

        # Start a fresh loss curve (weights are kept - see Notes above)
        self.history = {'loss': []}

        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data using this model's PRIVATE rng (never the global one)
            indices = self._rng.permutation(n_samples)
            X_shuffled = X_normalized[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                
                # Forward pass
                activations, pre_activations = self._forward_pass(X_batch)
                
                # Compute loss (Mean Squared Error) on the normalized scale.
                # This is the per-element MSE = (2 / input_dim) * the objective
                # L = (1/(2n))*sum||x - xhat||^2 that _backward_pass differentiates.
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
            Latent representation.
            A single 1-D sample of shape (input_dim,) returns shape (encoding_dim,).
        """
        if self.data_min is None:
            raise ValueError("Model is not fitted yet. Call fit(X) first.")

        X = np.array(X, dtype=float)

        # Normalize with the SAME min/max fit() learned from the training data
        X_normalized = (X - self.data_min) / (self.data_max - self.data_min + 1e-8)

        # Forward pass through encoder only
        activation = X_normalized
        n_encoder_layers = len(self.hidden_dims) + 1  # Hidden layers + encoding layer

        for i in range(n_encoder_layers):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            # The bottleneck (last encoder layer) may use bottleneck_activation
            activation = self._activate(z, self._layer_activation(i))

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
            Reconstructed data (denormalized back to the ORIGINAL data scale)
        """
        if self.data_min is None:
            raise ValueError("Model is not fitted yet. Call fit(X) first.")

        encoded = np.array(encoded, dtype=float)

        # Forward pass through decoder only
        activation = encoded
        n_encoder_layers = len(self.hidden_dims) + 1

        for i in range(n_encoder_layers, self.n_layers):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self._activate(z, self._layer_activation(i))

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
        
        Formula, for sample i:
            error_i = (1/input_dim) * sum_j (x_ij - xhat_ij)^2
        computed on the ORIGINAL data scale (not the internal [0, 1] scale).

        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, input_dim)
            Data to evaluate. A single 1-D sample of shape (input_dim,) is
            promoted to (1, input_dim), so the result is always a 1-D array -
            use `reconstruction_error(one_sample)[0]` to get a scalar.

        Returns:
        --------
        errors : np.ndarray, shape (n_samples,)
            Reconstruction error for each sample (MSE)
        """
        # atleast_2d so that a single sample does not blow up on axis=1 below
        X = np.atleast_2d(np.array(X, dtype=float))
        reconstructed = self.reconstruct(X)
        errors = np.mean((X - reconstructed) ** 2, axis=1)
        return errors
    
    def score(self, X):
        """
        Compute average reconstruction score (negative MSE)

        This is an unsupervised score, so it takes no y. It is the negated mean
        of reconstruction_error(X), measured on the ORIGINAL data scale, which
        makes `-model.score(X)` the plain reconstruction MSE.

        Parameters:
        -----------
        X : np.ndarray
            Data to evaluate

        Returns:
        --------
        score : float
            Negative mean reconstruction error (higher is better, 0.0 is perfect)
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

    # ------------------------------------------------------------------
    # Transformer-family aliases (PCA / t-SNE / UMAP style).
    # These are thin wrappers so an Autoencoder can be dropped into code
    # written against the usual fit / transform / inverse_transform API.
    # ------------------------------------------------------------------

    def transform(self, X):
        """
        Alias for encode(): project data into the latent space

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, input_dim)
            Data to encode

        Returns:
        --------
        encoded : np.ndarray, shape (n_samples, encoding_dim)
            Latent representation
        """
        return self.encode(X)

    def fit_transform(self, X):
        """
        Fit the autoencoder on X and return the latent representation of X

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, input_dim)
            Training data

        Returns:
        --------
        encoded : np.ndarray, shape (n_samples, encoding_dim)
            Latent representation of the training data
        """
        return self.fit(X).encode(X)

    def inverse_transform(self, encoded):
        """
        Alias for decode(): map latent codes back to the original data space

        Parameters:
        -----------
        encoded : np.ndarray, shape (n_samples, encoding_dim)
            Latent representation

        Returns:
        --------
        decoded : np.ndarray, shape (n_samples, input_dim)
            Reconstructed data on the original scale
        """
        return self.decode(encoded)


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

# Split train/test (disjoint: rows 0-799 train, rows 800-999 test)
X_train, X_test = X[:800], X[800:]

# Create and train autoencoder.
# NOTE on activation: the default 'relu' DIES on this problem - with a
# 2-unit bottleneck both code units switch off permanently and the model
# collapses to predicting the training mean (test MSE 0.876880, i.e. no
# better than a constant). 'tanh' cannot die, so we use it here.
autoencoder = Autoencoder(
    input_dim=20,
    encoding_dim=2,      # Compress to 2 dimensions
    activation='tanh',
    learning_rate=0.05,
    epochs=500,
    verbose=100
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

# Sanity check: the best possible 2-component LINEAR reconstruction is PCA.
mu = X_train.mean(axis=0)
_, _, Vt = np.linalg.svd(X_train - mu, full_matrices=False)
pca_rec = (X_test - mu) @ Vt[:2].T @ Vt[:2] + mu
print(f"PCA(2) MSE (the floor to beat): {np.mean((X_test - pca_rec) ** 2):.6f}")
"""

"""
USAGE EXAMPLE 2: Anomaly Detection

import numpy as np

# Generate normal data (Gaussian)
np.random.seed(42)
X_normal = np.random.randn(500, 10)

# Generate anomalies (different distribution)
X_anomaly = np.random.uniform(-5, 5, (50, 10))

# Train autoencoder on the FIRST 400 normal rows only.
# Rows 400-499 are held out so the scored negatives are never trained on.
X_normal_train, X_normal_held = X_normal[:400], X_normal[400:]

autoencoder = Autoencoder(
    input_dim=10,
    encoding_dim=3,
    hidden_dims=[8, 5],
    activation='tanh',   # 'relu' risks a dead 3-unit bottleneck here
    learning_rate=0.01,
    epochs=100,
    verbose=False
)
autoencoder.fit(X_normal_train)

# Compute reconstruction errors
normal_errors = autoencoder.reconstruction_error(X_normal_train)
anomaly_errors = autoencoder.reconstruction_error(X_anomaly)

print("\nAnomaly Detection Results:")
print("="*60)
print(f"Normal data - Mean error: {np.mean(normal_errors):.6f}")
print(f"Normal data - Max error:  {np.max(normal_errors):.6f}")
print(f"\nAnomaly data - Mean error: {np.mean(anomaly_errors):.6f}")
print(f"Anomaly data - Min error:  {np.min(anomaly_errors):.6f}")

# Set threshold (e.g., 95th percentile of TRAINING errors)
threshold = np.percentile(normal_errors, 95)
print(f"\nAnomaly threshold: {threshold:.6f}")

# Detect anomalies in a combined dataset built from HELD-OUT normal rows,
# so the reported precision is not inflated by scoring training data.
X_combined = np.vstack([X_normal_held, X_anomaly])
y_true = np.array([0] * len(X_normal_held) + [1] * 50)  # 0=normal, 1=anomaly

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

# Shuffle, then split BEFORE fitting so the "test" rows are genuinely unseen
idx = np.random.permutation(len(X))
X = X[idx]
X_train, X_test = X[:800], X[800:]

# Train autoencoder to learn features.
# 'tanh' instead of the default 'relu': with only 3 code units, a ReLU
# bottleneck routinely leaves one unit permanently dead, and then the
# "3 learned features" are really only 2.
autoencoder = Autoencoder(
    input_dim=64,
    encoding_dim=3,  # Learn 3 features
    hidden_dims=[32, 16],
    activation='tanh',
    learning_rate=0.01,
    epochs=150,
    verbose=25
)
autoencoder.fit(X_train)

# Encode to feature space
features = autoencoder.encode(X_train)

print(f"\nLearned {features.shape[1]} features from {X.shape[1]}-dim data")
print(f"Feature space statistics:")
print(f"  Mean: {np.mean(features, axis=0)}")
print(f"  Std:  {np.std(features, axis=0)}")
print(f"  Dead (zero-variance) units: {np.sum(np.std(features, axis=0) < 1e-12)}/3")

# Test reconstruction quality on HELD-OUT samples
test_samples = X_test[:10]
reconstructed = autoencoder.reconstruct(test_samples)
mse_per_sample = np.mean((test_samples - reconstructed) ** 2, axis=1)

print(f"\nReconstruction quality (10 held-out samples):")
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

# Train on the FIRST 400 noisy rows only; rows 400+ stay held out.
# (This is the "approximate" denoising autoencoder: fit() uses its input as
#  its own target, so the network learns the low-dimensional signal manifold
#  and drops the noise it cannot fit through a 5-unit bottleneck. The
#  canonical denoising AE instead pairs noisy inputs with CLEAN targets.)
denoiser = Autoencoder(
    input_dim=50,
    encoding_dim=5,
    hidden_dims=[30, 15],
    activation='tanh',
    learning_rate=0.01,
    epochs=200,
    verbose=50
)
denoiser.fit(X_noisy[:400])

# Denoise HELD-OUT test samples (rows 400-409 were never trained on)
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

# Generate data with REAL structure: 100 observed dimensions that are all
# linear mixtures of 8 hidden factors, plus a little noise. Pure
# np.random.randn(800, 100) would have nothing to compress, and every
# architecture would tie at "predict the mean".
np.random.seed(42)
Z = np.random.randn(800, 8)
X = Z @ np.random.randn(8, 100) + 0.1 * np.random.randn(800, 100)

X_train, X_test = X[:600], X[600:]   # disjoint

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
        activation='tanh',      # 'relu' can kill a narrow bottleneck
        learning_rate=0.05,
        epochs=150,
        verbose=False
    )
    model.fit(X_train)

    train_mse = -model.score(X_train)
    test_mse = -model.score(X_test)

    print(f"{arch['name']:<20} {arch['encoding_dim']:<15} {train_mse:<15.6f} {test_mse:<15.6f}")

print("\nObservations (these match the table above on this dataset):")
print("- Shallow: Fast training, may underfit complex patterns")
print("- Deep-Narrow: Better feature learning, needs more data")
print("- Wide-Bottleneck: Strong compression, may lose information")
print("  (encoding_dim=5 is BELOW the 8 true factors, so it cannot win)")
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
        activation='tanh',    # 'relu' dies at enc_dim=2 and hides the elbow
        learning_rate=0.05,
        epochs=400,
        verbose=False
    )
    model.fit(X_train)
    
    compression = 50 / enc_dim
    train_mse = -model.score(X_train)
    test_mse = -model.score(X_test)
    overfit = test_mse - train_mse
    
    print(f"{enc_dim:<15} {f'{compression:.1f}x':<15} {train_mse:<15.6f} {test_mse:<15.6f} {overfit:<15.6f}")

print("\nOptimal encoding dimension balances:")
print("- High compression (lower dim) vs Information preservation (higher dim)")
print("- Training error vs Generalization")
print("For this data (5 factors), the ELBOW is at encoding_dim=5:")
print("  going 3 -> 5 cuts test MSE by roughly 30x, going 5 -> 20 barely helps.")
print("Below 5 the bottleneck physically cannot carry all 5 factors.")
"""

"""
USAGE EXAMPLE 7: Effect of Activation Function

import numpy as np

# Generate data with real low-rank structure (6 hidden factors in 30 dims),
# otherwise every activation ties at "predict the mean" and the comparison
# says nothing.
np.random.seed(42)
factors = np.random.randn(600, 6)
X = factors @ np.random.randn(6, 30) + 0.1 * np.random.randn(600, 30)

X_train, X_test = X[:500], X[500:]   # disjoint

# Test different activation functions
activations = ['relu', 'sigmoid', 'tanh']

print("Comparing Activation Functions:")
print("="*80)
print(f"{'Activation':<15} {'Train MSE':<15} {'Test MSE':<15} {'Dead code units':<20}")
print("-"*80)

for activation in activations:
    model = Autoencoder(
        input_dim=30,
        encoding_dim=5,
        hidden_dims=[20, 10],
        activation=activation,
        learning_rate=0.05,
        epochs=300,
        verbose=False
    )
    model.fit(X_train)

    train_mse = -model.score(X_train)
    test_mse = -model.score(X_test)
    # A code unit is "dead" if it outputs the same value for every sample
    dead = np.sum(np.std(model.encode(X_test), axis=0) < 1e-12)

    print(f"{activation:<15} {train_mse:<15.6f} {test_mse:<15.6f} {f'{dead}/5':<20}")

print("\nActivation Function Characteristics:")
print("- ReLU: Fast, effective, prone to 'dying ReLU' problem")
print("  (watch the Dead code units column - those units learn nothing more)")
print("- Sigmoid: Smooth, bounded [0,1], can saturate (worst fit here)")
print("- Tanh: Smooth, bounded [-1,1], zero-centered (best fit here)")
"""

"""
USAGE EXAMPLE 8: Learning Rate Comparison

import numpy as np

# Generate data with real structure (6 hidden factors in 40 dims), so that
# a better learning rate can actually show up as a better fit.
np.random.seed(42)
factors = np.random.randn(500, 6)
X = factors @ np.random.randn(6, 40) + 0.1 * np.random.randn(500, 40)

X_train, X_test = X[:400], X[400:]   # disjoint

# Test different learning rates - including one that genuinely blows up
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

print("Effect of Learning Rate:")
print("="*80)
print(f"{'Learning Rate':<15} {'Final Train MSE':<20} {'Final Test MSE':<20}")
print("-"*80)

for lr in learning_rates:
    model = Autoencoder(
        input_dim=40,
        encoding_dim=8,
        activation='tanh',
        learning_rate=lr,
        epochs=100,
        verbose=False
    )
    # lr=0.5 overflows to nan here; silence the numpy warnings so the
    # table still prints and you can SEE the divergence as 'nan'
    with np.errstate(all='ignore'):
        model.fit(X_train)

        train_mse = -model.score(X_train)
        test_mse = -model.score(X_test)

    print(f"{lr:<15.3f} {train_mse:<20.6f} {test_mse:<20.6f}")

print("\nLearning Rate Guidelines:")
print("- Too low (0.001): still far from converged after 100 epochs")
print("- Good here (0.05-0.1): fast and stable on this 40-dim problem")
print("- Too high (0.5): diverges to nan - the run is destroyed")
print("Note: the useful range shifts with input_dim. The gradients are those")
print("of L = (1/2n)*sum||x-xhat||^2 while the printed loss is the per-element")
print("MSE, so the effective step grows with input_dim; wider inputs need a")
print("SMALLER learning_rate.")
"""

"""
USAGE EXAMPLE 9: Transfer Learning - Pre-trained Features

import numpy as np

# Generate source domain data with structure worth transferring:
# 10 hidden factors expressed in 50 observed dimensions.
np.random.seed(42)
source_factors = np.random.randn(1000, 10)
mixing = np.random.randn(10, 50)
X_source = source_factors @ mixing + 0.1 * np.random.randn(1000, 50)

# Train autoencoder on source domain
pretrained = Autoencoder(
    input_dim=50,
    encoding_dim=10,
    hidden_dims=[30, 20],
    activation='tanh',
    learning_rate=0.05,
    epochs=150,
    verbose=False
)
pretrained.fit(X_source)
print(f"Source-domain reconstruction MSE: {-pretrained.score(X_source):.6f}")

print("Pre-trained Autoencoder:")
print(f"Trained on {X_source.shape[0]} samples")
print(f"Learned to encode {X_source.shape[1]}D -> {pretrained.encoding_dim}D")

# Generate target domain data: same 10 factors, shifted distribution
target_factors = np.random.randn(200, 10) + 0.5
X_target = target_factors @ mixing + 0.1 * np.random.randn(200, 50)

# Extract features using pre-trained encoder (no re-training needed)
features_target = pretrained.encode(X_target)
print(f"Target-domain reconstruction MSE: {-pretrained.score(X_target):.6f}")

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

# Split normal data into THREE disjoint parts:
#   train       - fits the autoencoder
#   calibration - chooses the threshold, nothing else
#   test        - scored, and never used for either of the above
X_train = X_normal[:700]
X_calib = X_normal[700:850]
X_test_normal = X_normal[850:900]

# Train autoencoder on normal transactions only
fraud_detector = Autoencoder(
    input_dim=8,
    encoding_dim=3,
    hidden_dims=[6, 4],
    activation='tanh',   # 'relu' risks a dead 3-unit bottleneck
    learning_rate=0.01,
    epochs=200,
    batch_size=32,
    verbose=50
)

fraud_detector.fit(X_train)

# Compute reconstruction errors on the CALIBRATION set
calib_errors = fraud_detector.reconstruction_error(X_calib)

# Set threshold at 95th percentile of calibration errors
threshold = np.percentile(calib_errors, 95)
print(f"\nFraud detection threshold: {threshold:.6f}")

# Test on mixed data drawn from the untouched test split
X_test = np.vstack([X_test_normal, X_fraud[:50]])
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


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _24_autoencoders.py
    #
    # numpy only. Seeded, so the numbers below reproduce exactly.
    # Runs in a few seconds. ASCII-only output.
    # ----------------------------------------------------------------
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
