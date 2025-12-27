import numpy as np

class TSNE:
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) Implementation from Scratch
    
    t-SNE is a powerful dimensionality reduction technique used primarily for visualization
    of high-dimensional data. It maps high-dimensional data to 2D or 3D space while preserving
    local structure and revealing clusters.
    
    Key Idea: Convert distances between points into probabilities that represent similarities,
    then find a low-dimensional representation that matches these similarities.
    
    Use Cases:
    - Visualizing high-dimensional datasets (images, embeddings, features)
    - Exploring cluster structures in data
    - Understanding relationships between data points
    - Feature visualization in neural networks
    
    Key Concepts:
        Perplexity: Balances local vs global aspects (typical: 5-50)
        KL Divergence: Measures difference between high-D and low-D similarities
        Student t-distribution: Used in low-D to spread out points (avoid crowding)
    """
    
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, 
                 n_iter=1000, random_state=None, early_exaggeration=12.0,
                 early_exaggeration_iter=250, min_grad_norm=1e-7, verbose=0):
        """
        Initialize the t-SNE model
        
        Parameters:
        -----------
        n_components : int, default=2
            Dimension of the embedded space (typically 2 or 3)
            - 2D: Best for visualization on paper/screen
            - 3D: Interactive 3D plots, more information preserved
            
        perplexity : float, default=30.0
            Related to the number of nearest neighbors considered
            - Typical range: 5-50
            - Small perplexity: Focuses on very local structure
            - Large perplexity: Considers more global structure
            - Rule of thumb: perplexity < n_samples / 3
            
        learning_rate : float, default=200.0
            Learning rate for gradient descent
            - Typical range: 10-1000
            - Too high: Points bounce around
            - Too low: Slow convergence, local minima
            - Try values: 10, 100, 200, 500, 1000
            
        n_iter : int, default=1000
            Number of gradient descent iterations
            - Minimum: 250
            - Typical: 1000
            - More iterations: Better convergence, slower computation
            
        random_state : int or None, default=None
            Random seed for reproducibility
            
        early_exaggeration : float, default=12.0
            Factor to multiply P values by in early learning
            - Helps create tight clusters that can separate later
            - Typical range: 4-24
            
        early_exaggeration_iter : int, default=250
            Number of iterations for early exaggeration phase
            
        min_grad_norm : float, default=1e-7
            Convergence threshold - stops if gradient norm < this value
            
        verbose : int, default=0
            Verbosity level
            - 0: Silent
            - 1: Show progress every 50 iterations
            - 2: Show detailed information
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.min_grad_norm = min_grad_norm
        self.verbose = verbose
        
        # Will be set during fitting
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = None
        
    def _compute_pairwise_distances(self, X):
        """
        Compute pairwise squared Euclidean distances
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        distances : numpy array of shape (n_samples, n_samples)
            Squared Euclidean distances between all pairs
        """
        # Efficient computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        sum_X = np.sum(np.square(X), axis=1)
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
        
        # Ensure non-negative (numerical stability)
        D = np.maximum(D, 0)
        
        return D
    
    def _compute_joint_probabilities(self, distances, target_perplexity):
        """
        Compute joint probability matrix P in high-dimensional space
        
        Uses Gaussian distribution with adaptive variance (sigma) for each point.
        Variance is chosen such that the perplexity of the conditional distribution
        equals the target perplexity.
        
        Parameters:
        -----------
        distances : numpy array of shape (n_samples, n_samples)
            Squared Euclidean distances
        target_perplexity : float
            Target perplexity value
            
        Returns:
        --------
        P : numpy array of shape (n_samples, n_samples)
            Joint probability matrix (symmetric)
        """
        n = distances.shape[0]
        P = np.zeros((n, n))
        
        # Target entropy based on perplexity
        # Perplexity = 2^(entropy)
        target_entropy = np.log(target_perplexity)
        
        # For each point, find the sigma that gives target perplexity
        for i in range(n):
            # Binary search for optimal sigma (variance)
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0  # beta = 1 / (2 * sigma^2)
            
            # Get distances to all other points
            Di = distances[i, np.concatenate([np.arange(0, i), np.arange(i + 1, n)])]
            
            # Binary search for beta (precision parameter)
            for _ in range(50):  # Max 50 iterations
                # Compute P given current beta
                P_i = np.exp(-Di * beta)
                sum_P_i = np.sum(P_i)
                
                # Avoid division by zero
                if sum_P_i == 0:
                    sum_P_i = 1e-8
                    
                # Normalize to get conditional probabilities
                P_i = P_i / sum_P_i
                
                # Compute entropy: H = -sum(p * log(p))
                entropy = -np.sum(P_i * np.log2(P_i + 1e-8))
                
                # Check if we've reached target entropy
                entropy_diff = entropy - target_entropy
                
                if np.abs(entropy_diff) < 1e-5:
                    break
                
                # Adjust beta based on entropy difference
                if entropy_diff > 0:
                    # Entropy too high, increase beta (decrease sigma)
                    beta_min = beta
                    if beta_max == np.inf:
                        beta = beta * 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    # Entropy too low, decrease beta (increase sigma)
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta = beta / 2
                    else:
                        beta = (beta + beta_min) / 2
            
            # Store computed probabilities for this point
            P[i, np.concatenate([np.arange(0, i), np.arange(i + 1, n)])] = P_i
        
        # Symmetrize: P_ij = (P_i|j + P_j|i) / (2n)
        P = (P + P.T) / (2 * n)
        
        # Ensure minimum probability for numerical stability
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_low_dim_affinities(self, Y):
        """
        Compute affinities (similarities) in low-dimensional space using Student t-distribution
        
        Uses Student t-distribution with 1 degree of freedom (Cauchy distribution)
        This helps prevent "crowding problem" where moderate distances in high-D
        get crowded in low-D space.
        
        Parameters:
        -----------
        Y : numpy array of shape (n_samples, n_components)
            Low-dimensional embedding
            
        Returns:
        --------
        Q : numpy array of shape (n_samples, n_samples)
            Similarity matrix in low-dimensional space
        """
        # Compute squared Euclidean distances
        distances = self._compute_pairwise_distances(Y)
        
        # Student t-distribution with 1 degree of freedom
        # Q_ij = (1 + ||y_i - y_j||^2)^(-1) / sum(1 + ||y_k - y_l||^2)^(-1)
        Q = 1 / (1 + distances)
        
        # Set diagonal to zero (point compared to itself)
        np.fill_diagonal(Q, 0)
        
        # Normalize to get probabilities
        sum_Q = np.sum(Q)
        if sum_Q == 0:
            sum_Q = 1e-8
        Q = Q / sum_Q
        
        # Ensure minimum probability for numerical stability
        Q = np.maximum(Q, 1e-12)
        
        return Q
    
    def _compute_gradient(self, P, Q, Y):
        """
        Compute gradient of KL divergence with respect to Y
        
        The gradient has an attractive force (for similar points) and
        repulsive force (for dissimilar points).
        
        Parameters:
        -----------
        P : numpy array of shape (n_samples, n_samples)
            Probabilities in high-dimensional space
        Q : numpy array of shape (n_samples, n_samples)
            Probabilities in low-dimensional space
        Y : numpy array of shape (n_samples, n_components)
            Current low-dimensional embedding
            
        Returns:
        --------
        gradient : numpy array of shape (n_samples, n_components)
            Gradient of cost function
        """
        n = Y.shape[0]
        
        # Compute pairwise differences in Y
        # Y_diff[i,j] = y_i - y_j
        Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        
        # Compute distances in low-D space
        distances = self._compute_pairwise_distances(Y)
        
        # Inverse of (1 + distance^2)
        inv_distances = 1 / (1 + distances)
        np.fill_diagonal(inv_distances, 0)
        
        # Gradient: 4 * sum_j (P_ij - Q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^(-1)
        # The factor of 4 comes from the derivative
        PQ_diff = P - Q
        gradient = 4 * np.sum((PQ_diff[:, :, np.newaxis] * 
                               Y_diff * 
                               inv_distances[:, :, np.newaxis]), axis=1)
        
        return gradient
    
    def _compute_kl_divergence(self, P, Q):
        """
        Compute Kullback-Leibler divergence between P and Q
        
        KL(P||Q) = sum_ij P_ij * log(P_ij / Q_ij)
        
        This measures how different Q is from P. Lower is better.
        
        Parameters:
        -----------
        P : numpy array
            Target distribution (high-D similarities)
        Q : numpy array
            Approximating distribution (low-D similarities)
            
        Returns:
        --------
        kl_divergence : float
            KL divergence value
        """
        # Only compute where P > 0 to avoid log(0)
        kl = np.sum(P * np.log(P / Q))
        return kl
    
    def fit_transform(self, X):
        """
        Fit t-SNE model to X and return the embedded coordinates
        
        Algorithm:
        1. Compute pairwise distances in high-dimensional space
        2. Convert distances to probabilities (with perplexity-based sigma)
        3. Initialize low-dimensional representation randomly
        4. Optimize using gradient descent:
           - Compute low-dimensional affinities
           - Compute gradient of KL divergence
           - Update positions
           - Apply momentum for faster convergence
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            High-dimensional input data
            
        Returns:
        --------
        Y : numpy array of shape (n_samples, n_components)
            Embedded coordinates in low-dimensional space
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if self.verbose > 0:
            print(f"[t-SNE] Computing pairwise distances...")
        
        # Step 1: Compute pairwise distances
        distances = self._compute_pairwise_distances(X)
        
        if self.verbose > 0:
            print(f"[t-SNE] Computing P-values...")
        
        # Step 2: Compute joint probabilities P
        P = self._compute_joint_probabilities(distances, self.perplexity)
        
        # Step 3: Initialize Y randomly (small values near origin)
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        
        # For momentum-based gradient descent
        Y_velocity = np.zeros_like(Y)
        
        if self.verbose > 0:
            print(f"[t-SNE] Starting optimization with {self.n_iter} iterations...")
        
        # Step 4: Gradient descent optimization
        for iteration in range(self.n_iter):
            # Apply early exaggeration in initial iterations
            if iteration < self.early_exaggeration_iter:
                P_effective = P * self.early_exaggeration
            else:
                P_effective = P
            
            # Compute low-dimensional affinities
            Q = self._compute_low_dim_affinities(Y)
            
            # Compute gradient
            gradient = self._compute_gradient(P_effective, Q, Y)
            
            # Check for convergence
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < self.min_grad_norm:
                if self.verbose > 0:
                    print(f"[t-SNE] Converged at iteration {iteration}")
                break
            
            # Momentum schedule
            if iteration < 250:
                momentum = 0.5
            else:
                momentum = 0.8
            
            # Update with momentum
            Y_velocity = momentum * Y_velocity - self.learning_rate * gradient
            Y = Y + Y_velocity
            
            # Zero-center the embedding (for numerical stability)
            Y = Y - np.mean(Y, axis=0)
            
            # Compute KL divergence for monitoring
            if self.verbose > 0 and (iteration + 1) % 50 == 0:
                kl_div = self._compute_kl_divergence(P, Q)
                print(f"[t-SNE] Iteration {iteration + 1}/{self.n_iter}, "
                      f"KL divergence: {kl_div:.4f}, "
                      f"Gradient norm: {grad_norm:.6f}")
        
        # Store final results
        self.embedding_ = Y
        self.kl_divergence_ = self._compute_kl_divergence(P, Q)
        self.n_iter_ = iteration + 1
        
        if self.verbose > 0:
            print(f"[t-SNE] Optimization finished!")
            print(f"[t-SNE] Final KL divergence: {self.kl_divergence_:.4f}")
        
        return Y
    
    def fit(self, X):
        """
        Fit t-SNE model to X
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            High-dimensional input data
            
        Returns:
        --------
        self : object
            Fitted model
        """
        self.fit_transform(X)
        return self


"""
USAGE EXAMPLE 1: Visualizing Digits Dataset (Simple)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load digits dataset (8x8 images of digits 0-9)
digits = load_digits()
X = digits.data  # 1797 samples, 64 features
y = digits.target  # Labels (0-9)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, 
            n_iter=1000, random_state=42, verbose=1)
X_embedded = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                     c=y, cmap='tab10', s=20, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE visualization of Digits Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# You should see 10 distinct clusters, one for each digit!
"""

"""
USAGE EXAMPLE 2: Comparing Different Perplexity Values

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

# Try different perplexity values
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, perplexity in enumerate(perplexities):
    print(f"\nRunning t-SNE with perplexity={perplexity}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, 
                learning_rate=200, n_iter=500, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    axes[idx].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                     c=y, cmap='tab10', s=10, alpha=0.7)
    axes[idx].set_title(f'Perplexity = {perplexity}')
    axes[idx].set_xlabel('t-SNE 1')
    axes[idx].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

# Observations:
# - Low perplexity (5): Many small clusters, very local structure
# - Medium perplexity (30): Balanced, usually works well
# - High perplexity (50-100): Broader structure, more global patterns
"""

"""
USAGE EXAMPLE 3: Visualizing MNIST Fashion Dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load Fashion MNIST (takes a moment to download first time)
# Using a subset for faster computation
fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto')
X = fashion.data[:5000] / 255.0  # Normalize to [0, 1]
y = fashion.target[:5000].astype(int)

# Class names
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
            n_iter=1000, random_state=42, verbose=1)
X_embedded = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                     c=y, cmap='tab10', s=5, alpha=0.6)
plt.colorbar(scatter, label='Class', ticks=range(10))
plt.clim(-0.5, 9.5)

# Add legend
for i in range(10):
    plt.scatter([], [], c=[plt.cm.tab10(i)], label=class_names[i])
plt.legend(loc='best', markerscale=2)

plt.title('t-SNE visualization of Fashion MNIST')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# You should see fashion items grouped by type!
"""

"""
USAGE EXAMPLE 4: 3D Visualization

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

# 3D t-SNE
tsne = TSNE(n_components=3, perplexity=30, learning_rate=200,
            n_iter=1000, random_state=42, verbose=1)
X_embedded = tsne.fit_transform(X)

# 3D Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
                    c=y, cmap='tab10', s=20, alpha=0.6)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')
ax.set_title('3D t-SNE visualization of Digits')
plt.colorbar(scatter, label='Digit')
plt.show()

# 3D gives more space for points to separate!
"""

"""
USAGE EXAMPLE 5: Analyzing Text Embeddings

import numpy as np
import matplotlib.pyplot as plt

# Simulate word embeddings (e.g., from Word2Vec, GloVe)
# In practice, you would load pre-trained embeddings

np.random.seed(42)

# Create synthetic embeddings for different word categories
categories = ['animals', 'food', 'countries', 'sports']
words_per_category = 20
embedding_dim = 50

embeddings = []
labels = []
word_list = []

for idx, category in enumerate(categories):
    # Generate embeddings with category-specific patterns
    center = np.random.randn(embedding_dim) * 3
    category_embeddings = center + np.random.randn(words_per_category, embedding_dim) * 0.5
    
    embeddings.append(category_embeddings)
    labels.extend([idx] * words_per_category)
    word_list.extend([f'{category}_{i}' for i in range(words_per_category)])

X = np.vstack(embeddings)
y = np.array(labels)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=15, learning_rate=200,
            n_iter=1000, random_state=42, verbose=1)
X_embedded = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(12, 10))
colors = ['red', 'blue', 'green', 'orange']

for idx, category in enumerate(categories):
    mask = y == idx
    plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
               c=colors[idx], label=category, s=50, alpha=0.6)

plt.legend()
plt.title('t-SNE visualization of Word Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# Words in the same category should cluster together!
"""

"""
USAGE EXAMPLE 6: Comparing with PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X = digits.data
y = digits.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
            n_iter=1000, random_state=42, verbose=1)
X_tsne = tsne.fit_transform(X)

# Compare side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# PCA plot
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
ax1.set_title('PCA (Linear, Global Structure)')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
plt.colorbar(scatter1, ax=ax1, label='Digit')

# t-SNE plot
scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
ax2.set_title('t-SNE (Non-linear, Local Structure)')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
plt.colorbar(scatter2, ax=ax2, label='Digit')

plt.tight_layout()
plt.show()

# Key differences:
# - PCA: Shows global structure, linear relationships
# - t-SNE: Shows local clusters, reveals non-linear patterns
"""

"""
USAGE EXAMPLE 7: Hyperparameter Tuning

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data[:500]  # Smaller subset for faster experimentation
y = digits.target[:500]

# Test different parameter combinations
param_grid = {
    'perplexity': [5, 30, 50],
    'learning_rate': [50, 200, 500]
}

fig, axes = plt.subplots(3, 3, figsize=(18, 18))

for i, perplexity in enumerate(param_grid['perplexity']):
    for j, lr in enumerate(param_grid['learning_rate']):
        print(f"\nTesting: perplexity={perplexity}, lr={lr}")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr,
                   n_iter=500, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1],
                          c=y, cmap='tab10', s=10, alpha=0.7)
        axes[i, j].set_title(f'perp={perplexity}, lr={lr}')
        axes[i, j].set_xlabel('t-SNE 1')
        axes[i, j].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

# Tips for parameter selection:
# - Start with perplexity=30, lr=200 (good defaults)
# - If clusters overlap: decrease perplexity
# - If too scattered: increase perplexity
# - If optimization unstable: decrease learning rate
# - If converges too slowly: increase learning rate
"""

"""
USAGE EXAMPLE 8: Visualizing Neural Network Features

import numpy as np
import matplotlib.pyplot as plt

# Simulate features from different layers of a neural network
# In practice, extract features from your trained model

np.random.seed(42)

# Simulate 4 classes with high-dimensional features
n_samples_per_class = 100
n_features = 128

X_list = []
y_list = []

for class_id in range(4):
    # Create class-specific pattern
    class_center = np.random.randn(n_features) * 2
    class_features = class_center + np.random.randn(n_samples_per_class, n_features) * 0.8
    
    X_list.append(class_features)
    y_list.extend([class_id] * n_samples_per_class)

X = np.vstack(X_list)
y = np.array(y_list)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
            n_iter=1000, random_state=42, verbose=1)
X_embedded = tsne.fit_transform(X)

# Visualize with density
plt.figure(figsize=(12, 10))

for class_id in range(4):
    mask = y == class_id
    plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
               label=f'Class {class_id}', s=30, alpha=0.6)

plt.legend()
plt.title('t-SNE visualization of Neural Network Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.3)
plt.show()

# This helps understand:
# - Which classes are well-separated
# - Which classes are confused
# - Quality of learned representations
"""

