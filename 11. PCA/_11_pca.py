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

    Use Cases:
    - Visualization: squash 30, 64 or 20,000 features down to 2-D for a scatter plot
    - Compression: eigenfaces / image compression, keep 50 components instead of 4096 pixels
    - Denoising: drop the small-variance components, which are usually noise
    - Decorrelation: feed uncorrelated components to models that hate multicollinearity
    - Factor discovery: main market factors in finance, expression programs in genomics

    The Four Formulas This Class Implements:
        1. Covariance of the centered data (fit, Step 2):
               Cov = (X_centered^T @ X_centered) / (n_samples - 1)
        2. Eigen-decomposition of that symmetric matrix (fit, Step 3):
               Cov @ v = lambda * v
           The eigenvector v with the largest lambda is the direction of maximum
           variance; lambda itself IS the variance of the data along v.
        3. Explained variance ratio (fit, Step 4):
               explained_variance_ratio_[i] = lambda_i / sum_j(lambda_j)
        4. Singular values, the bridge to the SVD view of PCA (fit, Step 7):
               singular_value_i = sqrt(lambda_i * (n_samples - 1))

    Sign Convention:
        An eigenvector v and -v describe the same direction, so PCA component signs
        are mathematically arbitrary. Like scikit-learn's `svd_flip`, this class flips
        each component so that its largest-magnitude entry is positive. That makes
        components_ and transform() output reproducible run to run, and it matches
        sklearn.decomposition.PCA component-for-component on scikit-learn 1.5 and
        later. Version 1.5 is where sklearn started reading the sign off the component
        itself (svd_flip with u_based_decision=False); older sklearn read it off the
        transformed scores instead, so against those versions you get the same
        subspace with roughly half the components negated.

    Two Deliberate Differences From scikit-learn (see the "Simplifications vs. canonical
    PCA" section of _11_pca.md):
        - explained_variance_ and explained_variance_ratio_ are kept at FULL length
          (n_features), not truncated to n_components_, so that a scree plot can be drawn
          from a fitted model. Retained variance is therefore
              sum(explained_variance_ratio_[:n_components_])
          and NOT sum(explained_variance_ratio_), which is always 1.0.
        - score() returns the negative mean reconstruction error (-MSE), not the
          probabilistic-PCA average log-likelihood that sklearn's PCA.score returns.
          The two numbers are on completely different scales; do not compare them.

    Note on Method: production PCA (sklearn included) runs an SVD on the centered data
    matrix instead of eigendecomposing the covariance matrix. The two give the same
    answer, but SVD avoids squaring the condition number and costs
    O(n_samples * n_features * min(n_samples, n_features)) instead of
    O(n_features^3). The covariance route is used here because it makes the "variance
    -> eigenvector" story visible, which is the whole point of the algorithm.
    """
    
    def __init__(self, n_components=None):
        """
        Initialize the PCA model
        
        Parameters:
        -----------
        n_components : int or float or None, default=None
            How many principal components to keep.
            - If int: keep exactly that many components (clipped to n_features)
              Range: 1 to n_features
              Higher -> less information lost, weaker compression/denoising
              Lower  -> stronger compression, more reconstruction error
              Typical: 2 for plotting, 10-50 for an ML pipeline
            - If float strictly between 0 and 1: keep the smallest number of
              components whose cumulative explained variance reaches this fraction
              Range: 0.80 to 0.99
              Higher -> more components kept, more variance retained
              Typical: 0.95 (the standard "keep 95% of the variance" rule)
            - If None: keep all n_features components (no reduction, but you still
              get a decorrelated basis and a full scree plot)

        Examples:
        ---------
        n_components=2      -> Keep first 2 principal components
        n_components=0.95   -> Keep enough components for 95% variance
        n_components=None   -> Keep all components
        """
        self.n_components = n_components
        self.components_ = None          # Principal components (eigenvectors), shape (n_components_, n_features)
        self.mean_ = None                # Mean of training data, shape (n_features,)
        self.explained_variance_ = None  # Variance (eigenvalue) per component - FULL length n_features
        self.explained_variance_ratio_ = None  # Proportion of variance explained - FULL length n_features
        self.singular_values_ = None     # sqrt(eigenvalue * (n_samples - 1)), shape (n_components_,)
        self.n_features_ = None          # Number of features in original data
        self.n_components_ = None        # Actual number of components kept
        self.noise_variance_ = None      # Mean variance of the DISCARDED components (sigma^2)

    def fit(self, X):
        """
        Fit the PCA model by computing principal components
        
        This method:
        1. Centers the data (subtract mean)
        2. Computes covariance matrix
        3. Finds eigenvectors (principal components)
        4. Sorts by eigenvalues (variance explained)
        5. Fixes the arbitrary eigenvector signs so output is reproducible
        6. Decides how many components to keep and slices them off

        Parameters:
        -----------
        X : numpy array (or nested list) of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Accept plain Python lists as well as arrays, and fail with a clear
        # message on 1-D input rather than deep inside a matrix product later
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                "PCA expects a 2-D array of shape (n_samples, n_features); "
                "got an array with %d dimension(s). For a single feature use "
                "X.reshape(-1, 1)." % X.ndim
            )

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Step 1: Center the data (subtract mean from each feature)
        # This is crucial! PCA finds directions of variance from the mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute covariance matrix
        # Covariance matrix shows how features vary together
        # Formula: Cov = (X_centered^T @ X_centered) / (n-1)
        # np.atleast_2d keeps the single-feature case working: np.cov on one
        # column returns a 0-d scalar, which the eigen-solver would reject.
        covariance_matrix = np.atleast_2d(np.cov(X_centered.T))

        # Step 3: Compute eigenvalues and eigenvectors
        # Eigenvalues = variance along each principal component
        # Eigenvectors = directions of principal components
        # We use eigh, not eig: a covariance matrix is real and SYMMETRIC, and eigh
        # is the solver for that case. eig would use the general non-symmetric
        # routine, which returns complex128 whenever the covariance is rank
        # deficient (any time n_samples <= n_features) and does not guarantee
        # orthogonal eigenvectors for repeated eigenvalues. eigh always gives real
        # eigenvalues and a genuinely orthonormal set of eigenvectors.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort eigenvectors by eigenvalues (descending order)
        # We want components with highest variance first.
        # (eigh returns them ascending, so this sort really does reorder them.)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # A variance can never be negative; tiny negatives (-1e-16) are just
        # floating-point dust on directions with genuinely zero spread. Clamping
        # them keeps sqrt() below from producing NaNs.
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Step 5: Fix the arbitrary sign of each eigenvector
        # v and -v span the same axis, so LAPACK is free to return either. We adopt
        # scikit-learn's svd_flip convention - make the largest-magnitude entry of
        # every component positive - purely so that repeated runs (and comparisons
        # against sklearn 1.5+) give identical numbers instead of random sign flips.
        max_abs_rows = np.argmax(np.abs(eigenvectors), axis=0)
        signs = np.sign(eigenvectors[max_abs_rows, np.arange(eigenvectors.shape[1])])
        signs[signs == 0] = 1.0          # an all-zero column keeps its sign
        eigenvectors = eigenvectors * signs

        # Store explained variance
        # NOTE: these two arrays stay FULL length (n_features) so that a scree plot
        # can be drawn from a fitted model. Retained variance is therefore
        # sum(explained_variance_ratio_[:n_components_]), never sum(...) of the whole
        # array - that always equals 1.0.
        self.explained_variance_ = eigenvalues
        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio_ = eigenvalues / total_variance
        else:
            # Constant data: every direction has zero variance, so no direction
            # explains anything. Avoids a 0/0 NaN.
            self.explained_variance_ratio_ = np.zeros_like(eigenvalues)

        # Step 6: Determine number of components to keep
        if self.n_components is None:
            # Keep all components
            self.n_components_ = n_features
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Keep enough components to explain desired variance
            cumsum = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = int(np.argmax(cumsum >= self.n_components) + 1)
        else:
            # Keep specified number of components
            # (a float such as 2.0 or 1.0 is accepted and rounded down to an int,
            # since eigenvectors[:, :2.0] would raise an opaque slicing TypeError)
            if isinstance(self.n_components, float):
                if self.n_components != int(self.n_components):
                    raise ValueError(
                        "n_components as a float must be either a whole number "
                        "(e.g. 2.0) or strictly between 0 and 1 (e.g. 0.95); "
                        "got %r." % (self.n_components,)
                    )
                requested = int(self.n_components)
            else:
                requested = self.n_components
            if not isinstance(requested, (int, np.integer)) or requested < 1:
                raise ValueError(
                    "n_components must be None, an int >= 1, or a float in (0, 1); "
                    "got %r." % (self.n_components,)
                )
            self.n_components_ = int(min(requested, n_features))

        # Step 7: Select top components
        self.components_ = eigenvectors[:, :self.n_components_].T

        # Store singular values: sigma_i = sqrt(lambda_i * (n - 1)).
        # This is the bridge between the covariance/eigen view used here and the
        # SVD view used by sklearn - both describe the same decomposition.
        self.singular_values_ = np.sqrt(eigenvalues[:self.n_components_] * (n_samples - 1))

        # Average variance of everything we threw away. Probabilistic PCA models the
        # discarded directions as isotropic noise of this size; get_covariance() adds
        # it back as sigma^2 * I. It is exactly 0 when all components are kept.
        #
        # We average over only the first min(n_samples, n_features) eigenvalues: the
        # centered data can span at most that many directions, so any eigenvalue past
        # it is a structural zero, not a measurement. Including those zeros would
        # dilute the mean toward 0 on wide data (the eigenfaces case) and understate
        # the noise floor.
        #
        # This window is sklearn's, and it is not tight: centering drops the rank to
        # n_samples - 1, so when n_samples <= n_features the LAST eigenvalue inside the
        # window is a structural zero too. It also means get_covariance() does not
        # preserve the sample trace on wide data, because sigma^2 is spread over all
        # n_features directions while only min(n_samples, n_features) of them carry any
        # sample variance. Measured on a 30x100 standard-normal matrix
        # (np.random.RandomState(0), k=5): trace(get_covariance()) = 271.57 against
        # trace(np.cov) = 93.95 - and sklearn's get_covariance() returns the same
        # 271.57, so this is a shared convention, not a divergence.
        n_spectrum = min(n_samples, n_features)
        if self.n_components_ < n_spectrum:
            self.noise_variance_ = float(
                np.mean(eigenvalues[self.n_components_:n_spectrum]))
        else:
            self.noise_variance_ = 0.0

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X
        
        Project data onto the principal components.
        This reduces the dimensionality while preserving the most variance.
        
        Parameters:
        -----------
        X : numpy array (or nested list) of shape (n_samples, n_features)
            Data to transform

        Returns:
        --------
        X_transformed : numpy array of shape (n_samples, n_components_)
            Transformed data in the new coordinate system
        """
        if self.components_ is None:
            raise ValueError(
                "PCA is not fitted yet. Call fit(X) or fit_transform(X) first."
            )

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                "transform expects a 2-D array of shape (n_samples, n_features); "
                "got an array with %d dimension(s)." % X.ndim
            )
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "X has %d features, but this PCA was fitted on %d features."
                % (X.shape[1], self.n_features_)
            )

        # Center the data using training mean
        X_centered = X - self.mean_

        # Project onto principal components
        # Matrix multiplication: (n_samples, n_features) @ (n_features, n_components)
        # Result: (n_samples, n_components)
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
        if self.components_ is None:
            raise ValueError(
                "PCA is not fitted yet. Call fit(X) or fit_transform(X) first."
            )

        X_transformed = np.asarray(X_transformed, dtype=float)

        # Project back to original space
        # (n_samples, n_components) @ (n_components, n_features)
        X_reconstructed = np.dot(X_transformed, self.components_)

        # Add back the mean
        X_reconstructed += self.mean_

        return X_reconstructed
    
    def get_covariance(self):
        """
        Compute data covariance with principal components

        Formula (the probabilistic-PCA covariance, same as scikit-learn's):

            Cov ~= W^T @ diag(lambda_i - sigma^2) @ W  +  sigma^2 * I

        Read it as "signal plus noise". Every direction in feature space is given a
        baseline isotropic noise variance sigma^2 = noise_variance_ (the mean of the
        eigenvalues we discarded); the k retained directions then get topped up by
        the EXCESS variance lambda_i - sigma^2 that makes them stand out. Using
        lambda_i instead of the excess would double-count the noise along those k
        directions; dropping the sigma^2 * I term would bias every diagonal entry low,
        because the discarded directions still carry real variance.

        When all components are kept, sigma^2 = 0 and this collapses to the exact
        sample covariance matrix.

        Returns:
        --------
        covariance : numpy array of shape (n_features, n_features)
            Estimated covariance matrix of data
        """
        if self.components_ is None:
            raise ValueError(
                "PCA is not fitted yet. Call fit(X) or fit_transform(X) first."
            )

        components = self.components_
        exp_var = self.explained_variance_[:self.n_components_]

        # Excess variance of each retained direction above the noise floor
        # (clipped at 0 so a component weaker than the noise cannot subtract variance)
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.0)

        # Low-rank signal part: components.T @ diag(exp_var_diff) @ components
        covariance = np.dot(components.T * exp_var_diff, components)

        # Isotropic noise floor on every direction (0.0 when nothing was discarded)
        covariance += self.noise_variance_ * np.eye(self.n_features_)

        return covariance
    
    def score(self, X):
        """
        Return the negative mean reconstruction error, -MSE

        Projects X down to n_components_ dimensions, reconstructs it, and reports
        -mean((X - X_reconstructed)^2). Higher (i.e. closer to zero) is better, and
        0.0 means the projection lost nothing at all.

        NOTE: scikit-learn's PCA.score returns something different - the average
        Gaussian log-likelihood under the probabilistic-PCA model. That number lives
        on a completely different scale (e.g. on iris with k=2: this method gives
        -0.0253, sklearn gives -2.6998), so the two are NOT comparable. See the
        "Simplifications vs. canonical PCA" section of _11_pca.md.

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to score

        Returns:
        --------
        score : float
            Negative mean squared reconstruction error (<= 0.0; higher is better)
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
# Shows how much variance each component explains.
# NOTE: this array is FULL LENGTH (all 4 features), not just the 2 we kept.
# [9.70572245e-01 2.56962549e-02 2.91823131e-03 8.13269129e-04]

print("\nTotal variance retained:")
print(sum(pca.explained_variance_ratio_[:2]))
# Slice to the components actually kept! Summing the whole array is always 1.0.
# 0.9962684995586257  -> the 2-D projection preserves 99.6% of the variance
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
USAGE EXAMPLE 3: Data Visualization (4D to 2D)

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
# NOTE the [:pca.n_components_] slice: explained_variance_ratio_ is full length
# (all 64 ratios), so summing the whole array would always print 100.00%.
print(f"Variance retained: {sum(pca.explained_variance_ratio_[:pca.n_components_]):.2%}")

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
print(f"  - Information loss: Only {(1-sum(pca.explained_variance_ratio_[:pca.n_components_]))*100:.2f}%")
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
print(f"Dimension reduction: {X.shape[1]} -> {n_components_95}")
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

print(f"{'Components':<12} {'Variance':<12} {'Accuracy':<12} {'Fit+pred (s)':<12}")
print("-" * 48)

import time

for n_comp in components_to_try:
    # Apply PCA
    pca = PrincipalComponentAnalysis(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train classifier (timed, so the last column is a real measurement)
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    train_time = time.time() - start

    # Calculate metrics
    # Slice to n_components_: explained_variance_ratio_ holds all 30 ratios, so
    # summing the whole array would print 1.0000 on every row.
    variance = sum(pca.explained_variance_ratio_[:pca.n_components_])
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{n_comp:<12} {variance:<12.4f} {accuracy:<12.4f} {train_time:<12.4f}")

# Observations:
# - More components = more variance retained, but accuracy peaks and then flattens
# - Sweet spot often around 80-95% variance
# - kNN on this dataset is tiny, so the timings are dominated by noise; the real
#   speedup from PCA shows up on larger data or on distance-heavy models
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _11_pca.py
    # numpy-only, seeded, ASCII-only output, runs in well under a second.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # ================================================================
    # DEMO 1 - Known-answer test: recover a 2-D plane hidden inside 5-D
    # ================================================================
    print("=" * 55)
    print("DEMO 1 - Recovering a planted 2-D plane hidden in 5-D")
    print("=" * 55)
    print("We build data that TRULY lives on a known 2-D plane inside 5-D,")
    print("add a little noise and a big offset, then check PCA finds it back.")

    # A known orthonormal 5x2 basis for the plane we are planting
    basis = np.linalg.qr(np.random.randn(5, 2))[0]
    # Latent coordinates: PC1 should end up ~9x stronger than PC2 (3.0^2 vs 1.0^2)
    latent = np.random.randn(300, 2) * np.array([3.0, 1.0])
    offset = np.array([10.0, -4.0, 7.0, 0.0, 2.0])       # non-zero mean on purpose
    X1 = latent @ basis.T + 0.1 * np.random.randn(300, 5) + offset

    # PCA is unsupervised, but it still LEARNS something (a mean and a basis), so
    # it can still overfit. Hold out rows the fit never sees and score both.
    X1_train, X1_test = X1[:240], X1[240:]

    pca1 = PrincipalComponentAnalysis(n_components=2)
    Z1 = pca1.fit_transform(X1_train)

    print("\nData shape %s -> train %s / test %s"
          % (X1.shape, X1_train.shape, X1_test.shape))
    ratios = pca1.explained_variance_ratio_[:pca1.n_components_]
    print("explained_variance_ratio_[:2] : [%.4f, %.4f]" % (ratios[0], ratios[1]))
    print("variance retained by 2 of 5 PCs: %.4f" % np.sum(ratios))

    # Did we find the planted plane? components_ @ basis is 2x2; if the two
    # subspaces coincide it is orthogonal, so both its singular values are 1.
    overlap = np.linalg.svd(pca1.components_ @ basis, compute_uv=False)
    print("planted-subspace overlap (should be ~1.00): %.4f, %.4f"
          % (overlap[0], overlap[1]))

    # Reconstruction should bottom out at the noise floor, not at zero:
    # we discard 3 of 5 directions, each carrying 0.1^2 of noise variance.
    # Train and test scores should be nearly equal - the subspace generalizes.
    noise_floor = 0.1 ** 2 * 3 / 5
    print("Train score (-MSE): %.5f" % pca1.score(X1_train))
    print("Test  score (-MSE): %.5f" % pca1.score(X1_test))
    print("noise floor       : %.5f  (best achievable -MSE is -%.5f)"
          % (noise_floor, noise_floor))

    Z1_test = pca1.transform(X1_test)
    recon_test = pca1.inverse_transform(Z1_test)
    print("\nSample TEST rows (first 3 features shown):")
    print("  row   original[0:3]              PC scores          reconstructed[0:3]")
    for i in range(3):
        print("  %-3d  [%6.2f %6.2f %6.2f]  [%7.3f %7.3f]  [%6.2f %6.2f %6.2f]" % (
            i, X1_test[i, 0], X1_test[i, 1], X1_test[i, 2],
            Z1_test[i, 0], Z1_test[i, 1],
            recon_test[i, 0], recon_test[i, 1], recon_test[i, 2]))

    # ================================================================
    # DEMO 2 - Choosing k automatically with a variance threshold
    # ================================================================
    print("\n" + "=" * 55)
    print("DEMO 2 - Choosing k by variance threshold (n_components=0.95)")
    print("=" * 55)
    print("10 observed features are built from only 3 latent factors,")
    print("so PCA should decide 3 components are enough for 95% variance.")

    factors = np.random.randn(400, 3) * np.array([4.0, 3.0, 2.0])
    mixing = np.random.randn(3, 10)
    X2 = factors @ mixing + 0.05 * np.random.randn(400, 10)

    pca2 = PrincipalComponentAnalysis(n_components=0.95)
    Z2 = pca2.fit_transform(X2)

    print("\nComponents kept: %d out of %d features" % (pca2.n_components_,
                                                       pca2.n_features_))
    # IMPORTANT: explained_variance_ratio_ is FULL length (10 entries here), so
    # summing the whole array always gives 1.0. Always slice to n_components_.
    print("Variance retained: %.4f" % np.sum(
        pca2.explained_variance_ratio_[:pca2.n_components_]))

    print("\nScree table (first 6 components):")
    cumulative = np.cumsum(pca2.explained_variance_ratio_)
    for i in range(6):
        marker = "  <- cutoff" if i + 1 == pca2.n_components_ else ""
        print("  PC%-2d  var=%9.4f  ratio=%.4f  cumulative=%.4f%s" % (
            i + 1, pca2.explained_variance_[i],
            pca2.explained_variance_ratio_[i], cumulative[i], marker))

    print("\nsingular_values_ = sqrt(lambda * (n-1)) for the kept components:")
    print("  " + "  ".join("%.3f" % s for s in pca2.singular_values_))

    # ================================================================
    # DEMO 3 - PCA as a denoiser
    # ================================================================
    print("\n" + "=" * 55)
    print("DEMO 3 - PCA as a denoiser (keep signal, drop noise directions)")
    print("=" * 55)
    print("A clean rank-2 signal is buried in noise; keeping only the top 2")
    print("PCs throws the noise away because noise has no dominant direction.")

    t = np.linspace(0, 10, 200)
    X_clean = np.column_stack([
        np.sin(t), np.cos(t), 2 * np.sin(t), 2 * np.cos(t), np.sin(t) + np.cos(t),
        np.sin(t) - np.cos(t)
    ])
    X_noisy = X_clean + np.random.normal(0, 0.1, X_clean.shape)

    pca3 = PrincipalComponentAnalysis(n_components=2)
    X_denoised = pca3.inverse_transform(pca3.fit_transform(X_noisy))

    mse_before = np.mean((X_noisy - X_clean) ** 2)
    mse_after = np.mean((X_denoised - X_clean) ** 2)
    print("\nMSE(noisy,    clean) = %.6f" % mse_before)
    print("MSE(denoised, clean) = %.6f" % mse_after)
    print("Noise removed: %.2f%%" % ((1 - mse_after / mse_before) * 100))
    print("Variance held by top 2 PCs: %.2f%%" % (
        100 * np.sum(pca3.explained_variance_ratio_[:2])))

    print("\nFeature 0, first 4 rows (clean -> noisy -> denoised):")
    for i in range(4):
        print("  %7.4f -> %7.4f -> %7.4f" % (
            X_clean[i, 0], X_noisy[i, 0], X_denoised[i, 0]))
