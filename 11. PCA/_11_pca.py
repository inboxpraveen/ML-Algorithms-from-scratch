import numpy as np

class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) Implementation from Scratch
    
    PCA is a dimensionality reduction technique that finds the directions
    of maximum variance in high-dimensional data and projects it onto a
    lower-dimensional space while retaining most of the information.
    
    Key Idea: "Find the most important directions in your data"
    
    PCA transforms data to a new coordinate system where:
    - First axis (PC1) = direction of maximum variance
    - Second axis (PC2) = direction of second-most variance (orthogonal to PC1)
    - And so on...
    
    where:
        n_components = number of principal components to keep
        explained_variance = amount of information retained
    """
    
    def __init__(self, n_components=None):
        """
        Initialize the PCA model
        
        Parameters:
        -----------
        n_components : int or float, default=None
            Number of components to keep
            - If int: Keep exactly n_components principal components
            - If float (0 < n_components < 1): Keep enough components to retain
              this fraction of variance (e.g., 0.95 = 95% of variance)
            - If None: Keep all components
        
        Examples:
        ---------
        n_components=2      → Keep first 2 principal components
        n_components=0.95   → Keep enough components for 95% variance
        n_components=None   → Keep all components
        """
        self.n_components = n_components
        self.components_ = None          # Principal components (eigenvectors)
        self.mean_ = None                # Mean of training data
        self.explained_variance_ = None  # Variance explained by each component
        self.explained_variance_ratio_ = None  # Proportion of variance explained
        self.singular_values_ = None     # Singular values from SVD
        self.n_features_ = None          # Number of features in original data
        self.n_components_ = None        # Actual number of components kept
    
    def fit(self, X):
        """
        Fit the PCA model by computing principal components
        
        This method:
        1. Centers the data (subtract mean)
        2. Computes covariance matrix
        3. Finds eigenvectors (principal components)
        4. Sorts by eigenvalues (variance explained)
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Step 1: Center the data (subtract mean from each feature)
        # This is crucial! PCA finds directions of variance from the mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        # Covariance matrix shows how features vary together
        # Formula: Cov = (X^T × X) / (n-1)
        covariance_matrix = np.cov(X_centered.T)
        
        # Step 3: Compute eigenvalues and eigenvectors
        # Eigenvalues = variance along each principal component
        # Eigenvectors = directions of principal components
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # Step 4: Sort eigenvectors by eigenvalues (descending order)
        # We want components with highest variance first
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store explained variance
        self.explained_variance_ = eigenvalues
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
        # Step 5: Determine number of components to keep
        if self.n_components is None:
            # Keep all components
            self.n_components_ = n_features
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Keep enough components to explain desired variance
            cumsum = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = np.argmax(cumsum >= self.n_components) + 1
        else:
            # Keep specified number of components
            self.n_components_ = min(self.n_components, n_features)
        
        # Step 6: Select top components
        self.components_ = eigenvectors[:, :self.n_components_].T
        
        # Store singular values (square root of eigenvalues)
        self.singular_values_ = np.sqrt(eigenvalues[:self.n_components_] * (n_samples - 1))
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X
        
        Project data onto the principal components.
        This reduces the dimensionality while preserving the most variance.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to transform
        
        Returns:
        --------
        X_transformed : numpy array of shape (n_samples, n_components_)
            Transformed data in the new coordinate system
        """
        # Center the data using training mean
        X_centered = X - self.mean_
        
        # Project onto principal components
        # Matrix multiplication: (n_samples × n_features) × (n_features × n_components)
        # Result: (n_samples × n_components)
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit the model and apply dimensionality reduction
        
        Convenience method that combines fit() and transform()
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        
        Returns:
        --------
        X_transformed : numpy array of shape (n_samples, n_components_)
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space
        
        Reconstruct the original data from principal components.
        Note: This is an approximation if n_components < n_features
        
        Parameters:
        -----------
        X_transformed : numpy array of shape (n_samples, n_components_)
            Data in principal component space
        
        Returns:
        --------
        X_reconstructed : numpy array of shape (n_samples, n_features)
            Reconstructed data in original space
        """
        # Project back to original space
        # (n_samples × n_components) × (n_components × n_features)
        X_reconstructed = np.dot(X_transformed, self.components_)
        
        # Add back the mean
        X_reconstructed += self.mean_
        
        return X_reconstructed
    
    def get_covariance(self):
        """
        Compute data covariance with principal components
        
        Returns:
        --------
        covariance : numpy array of shape (n_features, n_features)
            Estimated covariance matrix of data
        """
        components = self.components_
        exp_var = self.explained_variance_[:self.n_components_]
        
        # Cov = components.T × diag(explained_variance) × components
        return np.dot(components.T * exp_var, components)
    
    def score(self, X):
        """
        Return the average log-likelihood of all samples
        
        This measures how well the model fits the data.
        Higher score = better fit
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to score
        
        Returns:
        --------
        score : float
            Average log-likelihood
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        # Calculate reconstruction error (mean squared error)
        mse = np.mean((X - X_reconstructed) ** 2)
        
        # Return negative MSE (higher is better)
        return -mse


"""
USAGE EXAMPLE 1: Basic Dimensionality Reduction

import numpy as np

# Sample data: 5 samples with 4 features
X = np.array([
    [2.5, 2.4, 3.1, 2.8],
    [0.5, 0.7, 1.2, 0.9],
    [2.2, 2.9, 2.7, 3.1],
    [1.9, 2.2, 2.5, 2.3],
    [3.1, 3.0, 3.3, 3.2]
])

# Create PCA model - reduce to 2 dimensions
pca = PrincipalComponentAnalysis(n_components=2)

# Fit and transform
X_reduced = pca.fit_transform(X)

print("Original shape:", X.shape)        # (5, 4)
print("Reduced shape:", X_reduced.shape)  # (5, 2)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)
# Shows how much variance each component explains

print("\nTotal variance retained:")
print(sum(pca.explained_variance_ratio_[:2]))
# Shows total information preserved (e.g., 0.95 = 95%)
"""

"""
USAGE EXAMPLE 2: Automatic Component Selection (Preserve 95% Variance)

import numpy as np
from sklearn.datasets import load_iris

# Load iris dataset (4 features)
data = load_iris()
X = data.data

# Keep enough components to preserve 95% of variance
pca = PrincipalComponentAnalysis(n_components=0.95)
X_reduced = pca.fit_transform(X)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_reduced.shape[1]}")
print(f"Components kept: {pca.n_components_}")
print(f"Variance preserved: {sum(pca.explained_variance_ratio_[:pca.n_components_]):.4f}")

# Show variance explained by each component
print("\nVariance explained by each component:")
for i, var in enumerate(pca.explained_variance_ratio_[:pca.n_components_]):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
"""

"""
USAGE EXAMPLE 3: Data Visualization (3D to 2D)

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
data = load_iris()
X = data.data
y = data.target

# Reduce to 2D for visualization
pca = PrincipalComponentAnalysis(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
target_names = data.target_names

for i, color, name in zip(range(3), colors, target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                color=color, alpha=0.8, label=name)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Iris Dataset - PCA Projection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nTotal variance retained: {sum(pca.explained_variance_ratio_[:2]):.2%}")
"""

"""
USAGE EXAMPLE 4: Feature Extraction & Reconstruction

import numpy as np

# Create sample data
X = np.array([
    [1, 2, 3, 4, 5],
    [2, 4, 5, 4, 5],
    [3, 6, 7, 8, 9],
    [4, 8, 9, 8, 9],
    [5, 10, 11, 12, 13]
])

print("Original data shape:", X.shape)
print("Original data:\n", X)

# Reduce to 2 components
pca = PrincipalComponentAnalysis(n_components=2)
X_reduced = pca.fit_transform(X)

print("\nReduced data shape:", X_reduced.shape)
print("Reduced data:\n", X_reduced)

# Reconstruct back to original space
X_reconstructed = pca.inverse_transform(X_reduced)

print("\nReconstructed data shape:", X_reconstructed.shape)
print("Reconstructed data:\n", X_reconstructed)

# Calculate reconstruction error
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"\nReconstruction error (MSE): {reconstruction_error:.6f}")
print(f"Variance preserved: {sum(pca.explained_variance_ratio_[:2]):.2%}")
"""

"""
USAGE EXAMPLE 5: PCA for Noise Reduction

import numpy as np

# Create data with noise
np.random.seed(42)
n_samples = 100
t = np.linspace(0, 10, n_samples)

# Signal: combination of two correlated features
X_clean = np.column_stack([
    np.sin(t),
    np.cos(t),
    np.sin(t) * 2,
    np.cos(t) * 2,
    np.sin(t) + np.cos(t)
])

# Add noise
noise = np.random.normal(0, 0.1, X_clean.shape)
X_noisy = X_clean + noise

# Apply PCA to reduce noise
# Keep only top components (they capture signal, not noise)
pca = PrincipalComponentAnalysis(n_components=2)
X_reduced = pca.fit_transform(X_noisy)
X_denoised = pca.inverse_transform(X_reduced)

# Calculate improvement
noise_before = np.mean((X_noisy - X_clean) ** 2)
noise_after = np.mean((X_denoised - X_clean) ** 2)

print(f"Original noise (MSE): {noise_before:.6f}")
print(f"After PCA denoising (MSE): {noise_after:.6f}")
print(f"Noise reduction: {(1 - noise_after/noise_before)*100:.2f}%")
print(f"\nVariance explained by top 2 components: {sum(pca.explained_variance_ratio_[:2]):.2%}")
"""

"""
USAGE EXAMPLE 6: PCA with Machine Learning Pipeline

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load digits dataset (64 features)
data = load_digits()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for PCA!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA - reduce from 64 to 20 dimensions
pca = PrincipalComponentAnalysis(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original dimensions: {X_train_scaled.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.2%}")

# Train a classifier on reduced data
# Note: Using sklearn's LogisticRegression for this example
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_pca, y_train)

# Evaluate
y_pred = clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy with PCA: {accuracy:.4f}")
print("\nBenefits:")
print(f"  - Training time: Much faster (20 features vs 64)")
print(f"  - Memory usage: 68.75% reduction")
print(f"  - Information loss: Only {(1-sum(pca.explained_variance_ratio_))*100:.2f}%")
"""

"""
USAGE EXAMPLE 7: Scree Plot - Choosing Number of Components

import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Load wine dataset
data = load_wine()
X = data.data

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA with all components
pca = PrincipalComponentAnalysis(n_components=None)
pca.fit(X_scaled)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Create scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Variance explained by each component
ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained')
ax1.set_title('Scree Plot')
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative variance
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
         marker='o', linestyle='-', linewidth=2)
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained')
ax2.set_title('Cumulative Variance Plot')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find optimal number of components
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nNumber of components for 95% variance: {n_components_95}")
print(f"Dimension reduction: {X.shape[1]} → {n_components_95}")
print(f"Reduction: {(1 - n_components_95/X.shape[1])*100:.1f}%")
"""

"""
USAGE EXAMPLE 8: Comparing PCA with Different Components

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load breast cancer dataset (30 features)
data = load_breast_cancer()
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different numbers of components
components_to_try = [2, 5, 10, 15, 20, 25, 30]

print(f"{'Components':<12} {'Variance':<12} {'Accuracy':<12} {'Speedup':<12}")
print("-" * 48)

import time

for n_comp in components_to_try:
    # Apply PCA
    pca = PrincipalComponentAnalysis(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train classifier
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    train_time = time.time() - start
    
    # Calculate metrics
    variance = sum(pca.explained_variance_ratio_)
    accuracy = accuracy_score(y_test, y_pred)
    speedup = 30 / n_comp  # Relative to full 30 features
    
    print(f"{n_comp:<12} {variance:<12.4f} {accuracy:<12.4f} {speedup:<12.2f}x")

# Observations:
# - More components = higher accuracy but slower
# - Sweet spot often around 80-95% variance
# - Dramatic speedup with fewer components
"""

