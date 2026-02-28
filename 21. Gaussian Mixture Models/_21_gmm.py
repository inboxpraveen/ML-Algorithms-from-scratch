import numpy as np

class GaussianMixtureModel:
    """
    Gaussian Mixture Model (GMM) Implementation from Scratch
    
    GMM is a probabilistic model that assumes data is generated from a mixture of 
    several Gaussian distributions with unknown parameters. It's a powerful tool for
    soft clustering, density estimation, and anomaly detection.
    
    Key Idea: "Data comes from multiple hidden Gaussian distributions"
    
    Use Cases:
    - Customer Segmentation: Group customers with overlapping characteristics
    - Image Segmentation: Separate foreground/background in images
    - Anomaly Detection: Identify outliers in complex distributions
    - Speech Recognition: Model phoneme distributions
    - Bioinformatics: Gene expression analysis, protein structure
    - Finance: Market regime detection, portfolio optimization
    
    Key Concepts:
        Components: Individual Gaussian distributions in the mixture
        Mixing Coefficients (π): Weight/probability of each component
        Mean (μ): Center of each Gaussian component
        Covariance (Σ): Shape/spread of each Gaussian component
        Soft Assignment: Each point belongs to all clusters with different probabilities
        EM Algorithm: Expectation-Maximization for parameter estimation
    """
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, 
                 covariance_type='full', random_state=None, reg_covar=1e-6):
        """
        Initialize the Gaussian Mixture Model
        
        Parameters:
        -----------
        n_components : int, default=3
            Number of Gaussian components (clusters) in the mixture
            - More components: Can model more complex distributions
            - Fewer components: Simpler, faster, less prone to overfitting
            Typical values: 2-10
            
        max_iter : int, default=100
            Maximum number of EM iterations
            - Algorithm stops if converged before max_iter
            Typical values: 50-200
            
        tol : float, default=1e-4
            Convergence threshold (change in log-likelihood)
            - Smaller values: More precise convergence
            - Larger values: Faster convergence
            
        covariance_type : {'full', 'diag', 'spherical', 'tied'}, default='full'
            Type of covariance matrix:
            - 'full': Each component has its own general covariance matrix
            - 'diag': Diagonal covariance (features independent within component)
            - 'spherical': Single variance per component (circular/spherical clusters)
            - 'tied': All components share same covariance matrix
            
        random_state : int, optional
            Random seed for reproducibility
            
        reg_covar : float, default=1e-6
            Regularization added to covariance diagonal for numerical stability
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = reg_covar
        
        # Model parameters (learned during fit)
        self.weights_ = None      # Mixing coefficients (π)
        self.means_ = None        # Component means (μ)
        self.covariances_ = None  # Component covariances (Σ)
        self.converged_ = False   # Whether EM converged
        self.n_iter_ = 0          # Number of iterations performed
        self.lower_bound_ = None  # Log-likelihood of best fit
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, X):
        """
        Initialize GMM parameters using K-means++ strategy
        
        Strategy:
        1. Randomly select k samples as initial means
        2. Initialize weights uniformly (1/k for each component)
        3. Initialize covariances as identity matrices (or variants)
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        """
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means using k-means++ style
        # Select first center randomly
        indices = [np.random.randint(n_samples)]
        
        # Select remaining centers with probability proportional to distance
        for _ in range(1, self.n_components):
            distances = np.array([
                min([np.sum((X[i] - X[j])**2) for j in indices])
                for i in range(n_samples)
            ])
            probs = distances / distances.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            indices.append(next_idx)
        
        self.means_ = X[indices].copy()
        
        # Initialize covariances based on type
        if self.covariance_type == 'full':
            # Full covariance matrix for each component
            self.covariances_ = np.array([
                np.eye(n_features) for _ in range(self.n_components)
            ])
        elif self.covariance_type == 'diag':
            # Diagonal covariance (only variances)
            self.covariances_ = np.ones((self.n_components, n_features))
        elif self.covariance_type == 'spherical':
            # Single variance per component
            self.covariances_ = np.ones(self.n_components)
        elif self.covariance_type == 'tied':
            # Single covariance matrix for all components
            self.covariances_ = np.eye(n_features)
    
    def _compute_precision_cholesky(self, covariances):
        """
        Compute precision matrix using Cholesky decomposition
        
        Precision matrix = Inverse of covariance matrix
        Using Cholesky decomposition for numerical stability
        
        Parameters:
        -----------
        covariances : np.ndarray
            Covariance matrices
            
        Returns:
        --------
        precision_cholesky : np.ndarray
            Cholesky decomposition of precision matrices
        """
        if self.covariance_type == 'full':
            n_components, n_features, _ = covariances.shape
            precision_cholesky = np.empty((n_components, n_features, n_features))
            
            for k in range(n_components):
                # Add regularization for numerical stability
                cov_k = covariances[k] + self.reg_covar * np.eye(n_features)
                
                # Cholesky decomposition of precision matrix
                try:
                    cov_chol = np.linalg.cholesky(cov_k)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, use eigenvalue decomposition
                    cov_k = cov_k + self.reg_covar * 10 * np.eye(n_features)
                    cov_chol = np.linalg.cholesky(cov_k)
                
                precision_cholesky[k] = np.linalg.solve(cov_chol, np.eye(n_features))
            
            return precision_cholesky
        
        elif self.covariance_type == 'diag':
            # For diagonal, precision is 1/variance
            return 1.0 / np.sqrt(covariances + self.reg_covar)
        
        elif self.covariance_type == 'spherical':
            # For spherical, single precision per component
            return 1.0 / np.sqrt(covariances + self.reg_covar)
        
        elif self.covariance_type == 'tied':
            # Single precision matrix for all components
            n_features = covariances.shape[0]
            cov = covariances + self.reg_covar * np.eye(n_features)
            cov_chol = np.linalg.cholesky(cov)
            return np.linalg.solve(cov_chol, np.eye(n_features))
    
    def _estimate_log_gaussian_prob(self, X):
        """
        Estimate log probability of samples under each Gaussian component
        
        For each sample x and component k, compute:
        log N(x | μ_k, Σ_k) = -0.5 * [(x-μ_k)^T Σ_k^(-1) (x-μ_k) + log|Σ_k| + d*log(2π)]
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data samples
            
        Returns:
        --------
        log_prob : np.ndarray, shape (n_samples, n_components)
            Log probability of each sample under each component
        """
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                cov_k = self.covariances_[k] + self.reg_covar * np.eye(n_features)
                
                # Compute log determinant
                sign, logdet = np.linalg.slogdet(cov_k)
                
                # Compute Mahalanobis distance
                precision = np.linalg.inv(cov_k)
                mahalanobis = np.sum(diff @ precision * diff, axis=1)
                
                # Log probability
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + 
                                        logdet + mahalanobis)
        
        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                var_k = self.covariances_[k] + self.reg_covar
                
                log_prob[:, k] = -0.5 * (np.sum(np.log(var_k)) + 
                                        np.sum((diff ** 2) / var_k, axis=1) +
                                        n_features * np.log(2 * np.pi))
        
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff = X - self.means_[k]
                var_k = self.covariances_[k] + self.reg_covar
                
                log_prob[:, k] = -0.5 * (n_features * np.log(var_k) + 
                                        np.sum(diff ** 2, axis=1) / var_k +
                                        n_features * np.log(2 * np.pi))
        
        elif self.covariance_type == 'tied':
            cov = self.covariances_ + self.reg_covar * np.eye(n_features)
            sign, logdet = np.linalg.slogdet(cov)
            precision = np.linalg.inv(cov)
            
            for k in range(self.n_components):
                diff = X - self.means_[k]
                mahalanobis = np.sum(diff @ precision * diff, axis=1)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + 
                                        logdet + mahalanobis)
        
        return log_prob
    
    def _e_step(self, X):
        """
        E-step: Estimate responsibilities (posterior probabilities)
        
        Compute γ(z_nk) = P(z_k | x_n) = (π_k * N(x_n | μ_k, Σ_k)) / Σ_j(π_j * N(x_n | μ_j, Σ_j))
        
        This is the probability that sample n belongs to component k.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        responsibilities : np.ndarray, shape (n_samples, n_components)
            Posterior probabilities
        log_likelihood : float
            Total log-likelihood of data
        """
        # Compute log probabilities
        log_prob = self._estimate_log_gaussian_prob(X)
        
        # Add log weights
        weighted_log_prob = log_prob + np.log(self.weights_)
        
        # Compute log-sum-exp for numerical stability
        log_likelihood = np.sum(self._log_sum_exp(weighted_log_prob, axis=1))
        
        # Compute responsibilities (normalize probabilities)
        log_responsibilities = weighted_log_prob - self._log_sum_exp(
            weighted_log_prob, axis=1, keepdims=True
        )
        responsibilities = np.exp(log_responsibilities)
        
        return responsibilities, log_likelihood
    
    def _m_step(self, X, responsibilities):
        """
        M-step: Update parameters to maximize expected log-likelihood
        
        Update formulas:
        - π_k = (1/N) * Σ_n γ(z_nk)
        - μ_k = Σ_n (γ(z_nk) * x_n) / Σ_n γ(z_nk)
        - Σ_k = Σ_n (γ(z_nk) * (x_n - μ_k)(x_n - μ_k)^T) / Σ_n γ(z_nk)
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        responsibilities : np.ndarray, shape (n_samples, n_components)
            Posterior probabilities from E-step
        """
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        nk = responsibilities.sum(axis=0) + 10 * np.finfo(responsibilities.dtype).eps
        
        # Update weights (mixing coefficients)
        self.weights_ = nk / n_samples
        
        # Update means
        self.means_ = (responsibilities.T @ X) / nk[:, np.newaxis]
        
        # Update covariances
        if self.covariance_type == 'full':
            self.covariances_ = np.empty((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = (responsibilities[:, k, np.newaxis] * diff).T @ diff / nk[k]
        
        elif self.covariance_type == 'diag':
            self.covariances_ = np.empty((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(responsibilities[:, k, np.newaxis] * diff**2, 
                                             axis=0) / nk[k]
        
        elif self.covariance_type == 'spherical':
            self.covariances_ = np.empty(self.n_components)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(responsibilities[:, k, np.newaxis] * diff**2) / (nk[k] * n_features)
        
        elif self.covariance_type == 'tied':
            self.covariances_ = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_ += (responsibilities[:, k, np.newaxis] * diff).T @ diff
            self.covariances_ /= n_samples
    
    def _log_sum_exp(self, arr, axis=None, keepdims=False):
        """
        Compute log(sum(exp(arr))) in numerically stable way
        
        Uses the log-sum-exp trick: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
        """
        arr_max = np.max(arr, axis=axis, keepdims=True)
        
        if not keepdims and arr_max.ndim > 0:
            arr_max = np.squeeze(arr_max, axis=axis)
        
        out = np.log(np.sum(np.exp(arr - (arr_max if keepdims else 
                                          arr_max[..., np.newaxis] if axis is not None 
                                          else arr_max)), axis=axis, keepdims=keepdims))
        out += arr_max if keepdims else (arr_max if axis is None else arr_max)
        
        return out
    
    def fit(self, X, y=None):
        """
        Estimate GMM parameters using Expectation-Maximization (EM) algorithm
        
        EM Algorithm:
        1. Initialize parameters (means, covariances, weights)
        2. E-step: Compute responsibilities (which component generated each point)
        3. M-step: Update parameters based on responsibilities
        4. Repeat until convergence (log-likelihood stops improving)
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data
        y : ignored
            Not used, present for API consistency
            
        Returns:
        --------
        self : GaussianMixtureModel
            Fitted model
        """
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        
        # EM iterations
        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities, log_likelihood = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X, responsibilities)
            
            # Check convergence
            change = log_likelihood - prev_log_likelihood
            
            if abs(change) < self.tol:
                self.converged_ = True
                break
            
            prev_log_likelihood = log_likelihood
        
        self.n_iter_ = iteration + 1
        self.lower_bound_ = log_likelihood
        
        return self
    
    def predict(self, X):
        """
        Predict component labels for samples (hard assignment)
        
        Assigns each sample to the component with highest posterior probability.
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        labels : np.ndarray, shape (n_samples,)
            Component labels (0 to n_components-1)
        """
        X = np.array(X, dtype=float)
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """
        Predict posterior probabilities for each component (soft assignment)
        
        Returns probability that each sample belongs to each component.
        This is what makes GMM a "soft clustering" method.
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        probabilities : np.ndarray, shape (n_samples, n_components)
            Posterior probabilities for each component
        """
        X = np.array(X, dtype=float)
        responsibilities, _ = self._e_step(X)
        return responsibilities
    
    def score(self, X, y=None):
        """
        Compute log-likelihood of data under the model
        
        Higher values indicate better fit.
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to evaluate
        y : ignored
            Not used, present for API consistency
            
        Returns:
        --------
        log_likelihood : float
            Log-likelihood of data
        """
        X = np.array(X, dtype=float)
        _, log_likelihood = self._e_step(X)
        return log_likelihood
    
    def sample(self, n_samples=1):
        """
        Generate random samples from the fitted Gaussian mixture
        
        Algorithm:
        1. For each sample, randomly select a component based on weights
        2. Sample from that component's Gaussian distribution
        
        Parameters:
        -----------
        n_samples : int, default=1
            Number of samples to generate
            
        Returns:
        --------
        X : np.ndarray, shape (n_samples, n_features)
            Generated samples
        y : np.ndarray, shape (n_samples,)
            Component labels for generated samples
        """
        if self.means_ is None:
            raise ValueError("Model must be fitted before sampling")
        
        n_features = self.means_.shape[1]
        
        # Select components based on weights
        component_indices = np.random.choice(
            self.n_components, 
            size=n_samples, 
            p=self.weights_
        )
        
        # Generate samples
        X = np.empty((n_samples, n_features))
        
        for i, component_idx in enumerate(component_indices):
            if self.covariance_type == 'full':
                X[i] = np.random.multivariate_normal(
                    self.means_[component_idx],
                    self.covariances_[component_idx]
                )
            elif self.covariance_type == 'diag':
                X[i] = np.random.normal(
                    self.means_[component_idx],
                    np.sqrt(self.covariances_[component_idx])
                )
            elif self.covariance_type == 'spherical':
                X[i] = np.random.normal(
                    self.means_[component_idx],
                    np.sqrt(self.covariances_[component_idx]),
                    size=n_features
                )
            elif self.covariance_type == 'tied':
                X[i] = np.random.multivariate_normal(
                    self.means_[component_idx],
                    self.covariances_
                )
        
        return X, component_indices
    
    def bic(self, X):
        """
        Compute Bayesian Information Criterion (BIC)
        
        BIC = -2 * log-likelihood + n_parameters * log(n_samples)
        
        Lower BIC is better. Used for model selection (choosing n_components).
        
        Parameters:
        -----------
        X : np.ndarray
            Data
            
        Returns:
        --------
        bic : float
            BIC score
        """
        n_samples, n_features = X.shape
        
        # Count parameters
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2
        
        mean_params = self.n_components * n_features
        weight_params = self.n_components - 1  # Sum to 1 constraint
        
        n_parameters = mean_params + cov_params + weight_params
        
        return -2 * self.score(X) + n_parameters * np.log(n_samples)
    
    def aic(self, X):
        """
        Compute Akaike Information Criterion (AIC)
        
        AIC = -2 * log-likelihood + 2 * n_parameters
        
        Lower AIC is better. Used for model selection (choosing n_components).
        
        Parameters:
        -----------
        X : np.ndarray
            Data
            
        Returns:
        --------
        aic : float
            AIC score
        """
        n_features = X.shape[1]
        
        # Count parameters
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2
        
        mean_params = self.n_components * n_features
        weight_params = self.n_components - 1
        
        n_parameters = mean_params + cov_params + weight_params
        
        return -2 * self.score(X) + 2 * n_parameters


"""
USAGE EXAMPLE 1: Basic Clustering with Soft Assignments

import numpy as np

# Generate synthetic data from 3 Gaussians
np.random.seed(42)
X1 = np.random.randn(100, 2) + np.array([0, 0])
X2 = np.random.randn(100, 2) + np.array([5, 5])
X3 = np.random.randn(100, 2) + np.array([5, 0])
X = np.vstack([X1, X2, X3])

# Fit GMM
gmm = GaussianMixtureModel(n_components=3, random_state=42)
gmm.fit(X)

# Hard clustering (like K-means)
labels = gmm.predict(X)

# Soft clustering (unique to GMM)
probabilities = gmm.predict_proba(X)

print("Gaussian Mixture Model Clustering:")
print("="*60)
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"Log-likelihood: {gmm.lower_bound_:.2f}")

print("\nComponent Weights:")
for i, weight in enumerate(gmm.weights_):
    print(f"  Component {i}: {weight:.4f}")

print("\nComponent Means:")
for i, mean in enumerate(gmm.means_):
    print(f"  Component {i}: {mean}")

print("\nSample soft assignments (first 5 samples):")
for i in range(5):
    print(f"Sample {i}: {probabilities[i]}")
"""

"""
USAGE EXAMPLE 2: Model Selection Using BIC and AIC

import numpy as np

# Generate data
np.random.seed(42)
X1 = np.random.randn(150, 2) * 0.5 + np.array([0, 0])
X2 = np.random.randn(150, 2) * 0.5 + np.array([3, 3])
X3 = np.random.randn(150, 2) * 0.5 + np.array([0, 3])
X = np.vstack([X1, X2, X3])

# Try different numbers of components
n_components_range = range(1, 8)
bic_scores = []
aic_scores = []

print("Model Selection with BIC and AIC:")
print("="*60)
print(f"{'N Components':>15} {'BIC':>15} {'AIC':>15}")
print("-"*60)

for n_components in n_components_range:
    gmm = GaussianMixtureModel(
        n_components=n_components,
        random_state=42,
        max_iter=100
    )
    gmm.fit(X)
    
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    
    bic_scores.append(bic)
    aic_scores.append(aic)
    
    print(f"{n_components:>15} {bic:>15.2f} {aic:>15.2f}")

best_n_bic = n_components_range[np.argmin(bic_scores)]
best_n_aic = n_components_range[np.argmin(aic_scores)]

print("\nBest number of components:")
print(f"  According to BIC: {best_n_bic}")
print(f"  According to AIC: {best_n_aic}")
"""

"""
USAGE EXAMPLE 3: Customer Segmentation

import numpy as np

# Simulate customer data
# [purchase_frequency, average_order_value, recency_days, customer_lifetime_value]

np.random.seed(42)

# Segment 1: High-value frequent buyers
segment1 = np.column_stack([
    np.random.normal(20, 3, 100),    # 20 purchases/month
    np.random.normal(200, 30, 100),  # $200 avg order
    np.random.normal(5, 2, 100),     # Purchased 5 days ago
    np.random.normal(5000, 500, 100) # $5000 lifetime value
])

# Segment 2: Medium-value occasional buyers
segment2 = np.column_stack([
    np.random.normal(8, 2, 150),     # 8 purchases/month
    np.random.normal(100, 20, 150),  # $100 avg order
    np.random.normal(15, 5, 150),    # Purchased 15 days ago
    np.random.normal(1500, 300, 150) # $1500 lifetime value
])

# Segment 3: Low-value rare buyers
segment3 = np.column_stack([
    np.random.normal(2, 1, 100),     # 2 purchases/month
    np.random.normal(50, 15, 100),   # $50 avg order
    np.random.normal(60, 20, 100),   # Purchased 60 days ago
    np.random.normal(300, 100, 100)  # $300 lifetime value
])

X = np.vstack([segment1, segment2, segment3])
feature_names = ['Purchase Freq', 'Avg Order Value', 'Recency', 'Lifetime Value']

# Fit GMM
gmm = GaussianMixtureModel(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict segments
labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

print("Customer Segmentation with GMM:")
print("="*70)

# Analyze each segment
for segment_id in range(3):
    mask = labels == segment_id
    segment_size = np.sum(mask)
    segment_data = X[mask]
    
    print(f"\nSegment {segment_id} ({segment_size} customers):")
    print(f"  Mixing weight: {gmm.weights_[segment_id]:.2%}")
    print(f"  Characteristics:")
    for i, feature in enumerate(feature_names):
        mean_val = np.mean(segment_data[:, i])
        print(f"    {feature}: {mean_val:.2f}")

# Find customers with uncertain assignments (between segments)
max_probs = np.max(probabilities, axis=1)
uncertain_customers = np.where(max_probs < 0.7)[0]

print(f"\nCustomers with uncertain segment assignments: {len(uncertain_customers)}")
print("(These customers exhibit characteristics of multiple segments)")
"""

"""
USAGE EXAMPLE 4: Anomaly Detection with GMM

import numpy as np

# Generate normal data
np.random.seed(42)
X_normal = np.random.randn(400, 2) * 1.5

# Add some anomalies
X_anomalies = np.random.uniform(low=-8, high=8, size=(20, 2))

X = np.vstack([X_normal, X_anomalies])
y_true = np.array([0] * 400 + [1] * 20)  # 0=normal, 1=anomaly

# Fit GMM
gmm = GaussianMixtureModel(n_components=2, random_state=42)
gmm.fit(X_normal)  # Train only on normal data

# Compute log-likelihood for all points
log_likelihoods = []
for i in range(len(X)):
    responsibilities, log_likelihood = gmm._e_step(X[i:i+1])
    log_likelihoods.append(log_likelihood)

log_likelihoods = np.array(log_likelihoods)

# Set threshold (e.g., bottom 5 percentile)
threshold = np.percentile(log_likelihoods, 5)

# Predict anomalies
predictions = (log_likelihoods < threshold).astype(int)

# Evaluate
true_positives = np.sum((predictions == 1) & (y_true == 1))
false_positives = np.sum((predictions == 1) & (y_true == 0))
true_negatives = np.sum((predictions == 0) & (y_true == 0))
false_negatives = np.sum((predictions == 0) & (y_true == 1))

accuracy = (true_positives + true_negatives) / len(y_true)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print("Anomaly Detection with GMM:")
print("="*60)
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"\nAnomalies detected: {np.sum(predictions == 1)}")
print(f"Actual anomalies: {np.sum(y_true == 1)}")
print(f"\nThreshold log-likelihood: {threshold:.2f}")
"""

"""
USAGE EXAMPLE 5: Comparing Covariance Types

import numpy as np

# Generate data with diagonal covariance structure
np.random.seed(42)
X = np.random.randn(300, 3)
X[:, 0] *= 3  # Different variance for each feature
X[:, 1] *= 1
X[:, 2] *= 2

# Try different covariance types
covariance_types = ['full', 'diag', 'spherical', 'tied']

print("Comparing Covariance Types:")
print("="*80)
print(f"{'Type':>12} {'BIC':>12} {'AIC':>12} {'Log-Like':>12} {'N Params':>12}")
print("-"*80)

for cov_type in covariance_types:
    gmm = GaussianMixtureModel(
        n_components=2,
        covariance_type=cov_type,
        random_state=42
    )
    gmm.fit(X)
    
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    log_like = gmm.score(X)
    
    # Count parameters
    n_features = 3
    if cov_type == 'full':
        n_params = 2 * 3 + 2 * 3 * 4 / 2 + 1
    elif cov_type == 'diag':
        n_params = 2 * 3 + 2 * 3 + 1
    elif cov_type == 'spherical':
        n_params = 2 * 3 + 2 + 1
    elif cov_type == 'tied':
        n_params = 2 * 3 + 3 * 4 / 2 + 1
    
    print(f"{cov_type:>12} {bic:>12.2f} {aic:>12.2f} {log_like:>12.2f} {int(n_params):>12}")

print("\nNotes:")
print("- 'full': Most flexible, most parameters, can overfit")
print("- 'diag': Good balance, assumes independent features")
print("- 'spherical': Simplest, assumes equal variance")
print("- 'tied': All components share covariance structure")
"""

"""
USAGE EXAMPLE 6: Image Segmentation (Color-based)

import numpy as np

# Simulate image pixels as RGB values
np.random.seed(42)

# Sky (blue)
sky_pixels = np.column_stack([
    np.random.normal(135, 15, 500),  # R
    np.random.normal(206, 15, 500),  # G
    np.random.normal(235, 15, 500)   # B
])

# Grass (green)
grass_pixels = np.column_stack([
    np.random.normal(34, 10, 500),   # R
    np.random.normal(139, 15, 500),  # G
    np.random.normal(34, 10, 500)    # B
])

# Building (gray)
building_pixels = np.column_stack([
    np.random.normal(128, 20, 500),  # R
    np.random.normal(128, 20, 500),  # G
    np.random.normal(128, 20, 500)   # B
])

# Combine all pixels
X = np.vstack([sky_pixels, grass_pixels, building_pixels])
X = np.clip(X, 0, 255)  # Ensure valid RGB values

# Fit GMM
gmm = GaussianMixtureModel(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Segment image
labels = gmm.predict(X)

print("Image Segmentation with GMM:")
print("="*60)

# Analyze segments
segment_names = {0: "Unknown", 1: "Unknown", 2: "Unknown"}
for segment_id in range(3):
    mask = labels == segment_id
    segment_rgb = np.mean(X[mask], axis=0)
    
    # Identify segment based on dominant color
    if segment_rgb[2] > 200:  # High blue
        name = "Sky"
    elif segment_rgb[1] > 100 and segment_rgb[0] < 60:  # High green, low red
        name = "Grass"
    else:
        name = "Building"
    
    print(f"\nSegment {segment_id} - {name}:")
    print(f"  Pixels: {np.sum(mask)}")
    print(f"  Avg RGB: ({segment_rgb[0]:.0f}, {segment_rgb[1]:.0f}, {segment_rgb[2]:.0f})")
    print(f"  Weight: {gmm.weights_[segment_id]:.2%}")
"""

"""
USAGE EXAMPLE 7: Density Estimation and Sampling

import numpy as np

# Generate training data from complex distribution
np.random.seed(42)
X1 = np.random.randn(200, 2) * 0.5 + np.array([0, 0])
X2 = np.random.randn(150, 2) * 0.8 + np.array([3, 2])
X3 = np.random.randn(100, 2) * 0.6 + np.array([1, 3])
X_train = np.vstack([X1, X2, X3])

# Fit GMM to learn distribution
gmm = GaussianMixtureModel(n_components=3, random_state=42)
gmm.fit(X_train)

print("Density Estimation and Sampling:")
print("="*60)
print(f"Model trained on {len(X_train)} samples")
print(f"Log-likelihood: {gmm.score(X_train):.2f}")

# Generate new samples from learned distribution
X_generated, component_labels = gmm.sample(n_samples=300)

print(f"\nGenerated {len(X_generated)} new samples")
print(f"\nDistribution of generated samples by component:")
for i in range(3):
    count = np.sum(component_labels == i)
    print(f"  Component {i}: {count} samples ({count/len(X_generated):.1%})")

# Compare statistics
print("\nOriginal data statistics:")
print(f"  Mean: {np.mean(X_train, axis=0)}")
print(f"  Std: {np.std(X_train, axis=0)}")

print("\nGenerated data statistics:")
print(f"  Mean: {np.mean(X_generated, axis=0)}")
print(f"  Std: {np.std(X_generated, axis=0)}")
"""

"""
USAGE EXAMPLE 8: Speech/Audio Feature Clustering

import numpy as np

# Simulate audio features (MFCC-like features)
# [feature1, feature2, ..., feature13]
np.random.seed(42)

# Phoneme 1
phoneme1 = np.random.randn(100, 13) * 2 + np.random.randn(13) * 5

# Phoneme 2
phoneme2 = np.random.randn(100, 13) * 1.5 + np.random.randn(13) * 5

# Phoneme 3
phoneme3 = np.random.randn(100, 13) * 2.5 + np.random.randn(13) * 5

X = np.vstack([phoneme1, phoneme2, phoneme3])

# Fit GMM with diagonal covariance (common for audio)
gmm = GaussianMixtureModel(
    n_components=3,
    covariance_type='diag',
    random_state=42
)
gmm.fit(X)

# Predict phoneme clusters
labels = gmm.predict(X)

print("Speech Phoneme Clustering:")
print("="*60)
print(f"Converged: {gmm.converged_} in {gmm.n_iter_} iterations")
print(f"Log-likelihood: {gmm.lower_bound_:.2f}")

# Analyze clusters
for i in range(3):
    cluster_size = np.sum(labels == i)
    print(f"\nPhoneme cluster {i}:")
    print(f"  Samples: {cluster_size}")
    print(f"  Weight: {gmm.weights_[i]:.2%}")

# Show soft assignments for ambiguous sounds
probabilities = gmm.predict_proba(X)
ambiguous_mask = np.max(probabilities, axis=1) < 0.6
ambiguous_count = np.sum(ambiguous_mask)

print(f"\nAmbiguous sounds (confidence < 60%): {ambiguous_count}")
print("These may represent transitional sounds between phonemes")
"""

"""
USAGE EXAMPLE 9: Market Regime Detection

import numpy as np

# Simulate stock market features
# [returns, volatility, volume, momentum]
np.random.seed(42)

# Bull market
bull_market = np.column_stack([
    np.random.normal(0.05, 0.02, 150),   # Positive returns
    np.random.normal(0.15, 0.03, 150),   # Low volatility
    np.random.normal(1.0, 0.2, 150),     # Normal volume
    np.random.normal(0.03, 0.01, 150)    # Positive momentum
])

# Bear market
bear_market = np.column_stack([
    np.random.normal(-0.03, 0.03, 100),  # Negative returns
    np.random.normal(0.30, 0.05, 100),   # High volatility
    np.random.normal(1.5, 0.4, 100),     # High volume (panic)
    np.random.normal(-0.02, 0.01, 100)   # Negative momentum
])

# Sideways market
sideways_market = np.column_stack([
    np.random.normal(0.0, 0.015, 100),   # Near-zero returns
    np.random.normal(0.20, 0.04, 100),   # Medium volatility
    np.random.normal(0.8, 0.15, 100),    # Low volume
    np.random.normal(0.0, 0.005, 100)    # No momentum
])

X = np.vstack([bull_market, bear_market, sideways_market])
feature_names = ['Returns', 'Volatility', 'Volume', 'Momentum']

# Shuffle to simulate time series
indices = np.random.permutation(len(X))
X = X[indices]

# Fit GMM
gmm = GaussianMixtureModel(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Identify regimes
labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

print("Market Regime Detection:")
print("="*70)

# Characterize each regime
for regime_id in range(3):
    mask = labels == regime_id
    regime_data = X[mask]
    
    avg_return = np.mean(regime_data[:, 0])
    avg_volatility = np.mean(regime_data[:, 1])
    
    # Classify regime
    if avg_return > 0.02 and avg_volatility < 0.20:
        regime_name = "Bull Market"
    elif avg_return < -0.01:
        regime_name = "Bear Market"
    else:
        regime_name = "Sideways Market"
    
    print(f"\nRegime {regime_id} - {regime_name}:")
    print(f"  Frequency: {gmm.weights_[regime_id]:.1%}")
    print(f"  Avg Return: {avg_return:.2%}")
    print(f"  Avg Volatility: {avg_volatility:.2%}")
    print(f"  Days in regime: {np.sum(mask)}")

# Detect regime transitions
regime_uncertainty = 1 - np.max(probabilities, axis=1)
transition_periods = np.where(regime_uncertainty > 0.4)[0]

print(f"\nPotential regime transition periods: {len(transition_periods)}")
print("(High uncertainty indicates market conditions are changing)")
"""
