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
        Mixing Coefficients (pi_k): Weight/probability of each component
        Mean (mu_k): Center of each Gaussian component
        Covariance (Sigma_k): Shape/spread of each Gaussian component
        Soft Assignment: Each point belongs to all clusters with different probabilities
        EM Algorithm: Expectation-Maximization for parameter estimation

    The EM update formulas this class implements (N = n_samples, K = n_components):
        E-step:  gamma(z_nk) = pi_k * N(x_n | mu_k, Sigma_k)
                               / sum_j pi_j * N(x_n | mu_j, Sigma_j)
        M-step:  N_k      = sum_n gamma(z_nk)
                 pi_k     = N_k / N
                 mu_k     = sum_n gamma(z_nk) * x_n / N_k
                 Sigma_k  = sum_n gamma(z_nk) * (x_n - mu_k)(x_n - mu_k)^T / N_k
        Objective: log L = sum_n log( sum_k pi_k * N(x_n | mu_k, Sigma_k) )
        Every EM iteration is guaranteed not to decrease log L.

    Numerical stability:
        Densities are evaluated through the Cholesky factor P = L^-T of each
        precision matrix (Sigma = L L^T), so that
            (x - mu)^T Sigma^-1 (x - mu) = ||(x - mu) @ P||^2
            log|Sigma|                   = -2 * sum(log(diag(P)))
        and responsibilities are normalised with the log-sum-exp trick.

    Simplification vs. canonical scikit-learn GaussianMixture:
        - Initialisation: scikit-learn's default init_params='kmeans' runs a full
          k-means to convergence and derives the initial responsibilities from its
          labels. This class does a single k-means++ D^2 seeding pass for the means
          and seeds every covariance with the data's own scatter cov(X). The result
          is usually the same optimum (verified against sklearn on 4 covariance
          types), but sklearn's start is slightly better conditioned.
        - score(X) returns the TOTAL log-likelihood, not sklearn's per-sample mean
          (see score()).
        - No Bayesian / variational variant, no warm_start, no precisions_init.
    """

    def __init__(self, n_components=3, max_iter=100, tol=1e-4,
                 covariance_type='full', random_state=None, reg_covar=1e-6,
                 n_init=1):
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
            Note on scale: this is compared against the change in the TOTAL
            (summed) log-likelihood, not the per-sample mean that sklearn's
            GaussianMixture uses. So tol=1e-4 here behaves like sklearn's
            tol=1e-4/n_samples - it is a stricter stopping rule on large data.
            Typical values: 1e-3 to 1e-6

        covariance_type : {'full', 'diag', 'spherical', 'tied'}, default='full'
            Type of covariance matrix:
            - 'full': Each component has its own general covariance matrix
            - 'diag': Diagonal covariance (features independent within component)
            - 'spherical': Single variance per component (circular/spherical clusters)
            - 'tied': All components share same covariance matrix
            
        random_state : int, optional
            Random seed for reproducibility
            - Seeds NumPy's global RNG (the convention used across this repo)
            - Re-applied at the start of every fit(), so refitting the same
              object on the same data reproduces the same result

        reg_covar : float, default=1e-6
            Regularization added to covariance diagonal for numerical stability
            - Stops a component that has collapsed onto a handful of points from
              producing a singular (non-invertible) covariance matrix
            - Raise it to 1e-4 or 1e-3 if you hit LinAlgError or exploding
              log-likelihoods
            Typical values: 1e-6 to 1e-3

        n_init : int, default=1
            Number of independent EM restarts; the run with the highest final
            log-likelihood is kept
            - EM only ever reaches a LOCAL maximum, so more restarts means a
              better chance of finding the global one
            - Cost grows linearly: n_init=10 takes ~10x as long
            Typical values: 1 (default, same as sklearn) to 10 on hard data
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.n_init = n_init

        # Model parameters (learned during fit)
        self.weights_ = None      # Mixing coefficients (pi_k)
        self.means_ = None        # Component means (mu_k)
        self.covariances_ = None  # Component covariances (Sigma_k)
        self.converged_ = False   # Whether EM converged
        self.n_iter_ = 0          # Number of iterations performed
        self.lower_bound_ = None  # Log-likelihood of best fit
        self.labels_ = None       # Hard cluster labels of the training data

        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, X):
        """
        Initialize GMM parameters using K-means++ strategy

        Strategy:
        1. Pick the first mean uniformly at random from the data, then pick each
           remaining mean with probability proportional to its squared distance to
           the nearest already-chosen mean (k-means++ D^2 sampling). Only the first
           centre is uniform; the rest are spread out on purpose.
        2. Initialize weights uniformly (1/k for each component)
        3. Initialize every covariance from the DATA's OWN SCATTER (np.cov(X)),
           not from the identity matrix.

        Why the data scatter and not I? The identity says "every component starts
        with unit variance in every direction". On a feature measured in dollars
        with std 500 that Gaussian is ~500x too narrow, so almost every point gets
        a responsibility of essentially 0 or 1 on the very first E-step and EM is
        already trapped in a bad local optimum before it takes a single real step.
        Seeding with cov(X) makes the starting Gaussians as wide as the data, which
        is what scikit-learn achieves by deriving its initial covariances from a
        k-means responsibility pass. On unstandardised data this is the single
        biggest quality difference in the whole file.

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
            total = distances.sum()
            if total > 0:
                probs = distances / total
            else:
                # Every point coincides with an already-chosen centre
                # (e.g. duplicated rows) -> fall back to uniform sampling
                probs = np.ones(n_samples) / n_samples
            next_idx = np.random.choice(n_samples, p=probs)
            indices.append(next_idx)

        self.means_ = X[indices].copy()

        # Initialize covariances from the data's own scatter, so the starting
        # Gaussians live on the same scale as the data (see docstring above).
        # np.atleast_2d keeps the (1, 1) shape when X has a single feature.
        data_cov = np.atleast_2d(np.cov(X, rowvar=False))
        data_cov = data_cov + self.reg_covar * np.eye(n_features)

        if self.covariance_type == 'full':
            # Full covariance matrix for each component
            self.covariances_ = np.array([
                data_cov.copy() for _ in range(self.n_components)
            ])
        elif self.covariance_type == 'diag':
            # Diagonal covariance (only variances, one row per component)
            self.covariances_ = np.tile(np.diag(data_cov), (self.n_components, 1))
        elif self.covariance_type == 'spherical':
            # Single variance per component: average variance across features
            self.covariances_ = np.full(self.n_components, np.mean(np.diag(data_cov)))
        elif self.covariance_type == 'tied':
            # Single covariance matrix shared by all components
            self.covariances_ = data_cov.copy()
        else:
            raise ValueError(
                "Unknown covariance_type: %r. Expected one of "
                "'full', 'diag', 'spherical', 'tied'." % (self.covariance_type,)
            )

    def _compute_precision_cholesky(self, covariances):
        """
        Compute the Cholesky factor of each precision (inverse covariance) matrix

        Write Sigma = L L^T with L lower-triangular (that is the Cholesky factor of
        the covariance). Then

            Sigma^-1 = (L L^T)^-1 = L^-T L^-1 = P P^T   where   P = L^-T

        so P is "the Cholesky factor of the precision" and is what this method
        returns. Working with P instead of Sigma^-1 is more numerically stable and
        it makes both quantities the Gaussian log-density needs cheap:

            (x - mu)^T Sigma^-1 (x - mu) = || (x - mu) @ P ||^2
            log|Sigma|                   = -2 * sum(log(diag(P)))

        This is the same convention scikit-learn stores in precisions_cholesky_.
        _estimate_log_gaussian_prob calls this method for 'full' and 'tied'. For
        'diag' and 'spherical' the covariance is already diagonal, so no
        factorisation is needed - the factor is simply 1/sqrt(variance), returned
        here for completeness, and those density branches work with the variances
        directly instead.

        Parameters:
        -----------
        covariances : np.ndarray
            Covariance matrices (shape depends on covariance_type)

        Returns:
        --------
        precision_cholesky : np.ndarray
            P = L^-T for each component (full/tied), or 1/sqrt(variance)
            (diag/spherical)
        """
        if self.covariance_type == 'full':
            n_components, n_features, _ = covariances.shape
            precision_cholesky = np.empty((n_components, n_features, n_features))
            
            for k in range(n_components):
                # Add regularization for numerical stability
                cov_k = covariances[k] + self.reg_covar * np.eye(n_features)
                
                # Cholesky factor L of the COVARIANCE: Sigma = L L^T
                try:
                    cov_chol = np.linalg.cholesky(cov_k)
                except np.linalg.LinAlgError:
                    # Not positive-definite (a collapsed component): add ten times
                    # the ridge and try once more
                    cov_k = cov_k + self.reg_covar * 10 * np.eye(n_features)
                    cov_chol = np.linalg.cholesky(cov_k)
                
                # solve(L, I) is L^-1; transpose it to get P = L^-T
                precision_cholesky[k] = np.linalg.solve(cov_chol, np.eye(n_features)).T

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
            # solve(L, I) is L^-1; transpose it to get P = L^-T
            return np.linalg.solve(cov_chol, np.eye(n_features)).T
    
    def _estimate_log_gaussian_prob(self, X):
        """
        Estimate log probability of samples under each Gaussian component
        
        For each sample x and component k, compute:
        log N(x | mu_k, Sigma_k) = -0.5 * [ (x-mu_k)^T Sigma_k^-1 (x-mu_k)
                                            + log|Sigma_k| + d*log(2*pi) ]
        
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
        log_prob = np.zeros((n_samples, self.n_components))

        if self.covariance_type == 'full':
            # P[k] = L_k^-T, the Cholesky factor of the precision matrix
            precisions_chol = self._compute_precision_cholesky(self.covariances_)

            for k in range(self.n_components):
                diff = X - self.means_[k]
                P = precisions_chol[k]

                # log|Sigma_k| = -2 * sum(log(diag(P)))  (since diag(P) = 1/diag(L))
                logdet = -2.0 * np.sum(np.log(np.diag(P)))

                # (x-mu)^T Sigma^-1 (x-mu) = ||(x-mu) @ P||^2, because Sigma^-1 = P P^T
                mahalanobis = np.sum((diff @ P) ** 2, axis=1)

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
            # One shared precision Cholesky factor P for every component
            P = self._compute_precision_cholesky(self.covariances_)
            logdet = -2.0 * np.sum(np.log(np.diag(P)))

            for k in range(self.n_components):
                diff = X - self.means_[k]
                mahalanobis = np.sum((diff @ P) ** 2, axis=1)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) +
                                        logdet + mahalanobis)

        else:
            raise ValueError(
                "Unknown covariance_type: %r. Expected one of "
                "'full', 'diag', 'spherical', 'tied'." % (self.covariance_type,)
            )

        return log_prob
    
    def _e_step(self, X):
        """
        E-step: Estimate responsibilities (posterior probabilities)
        
        Compute gamma(z_nk) = P(z_k | x_n)
                            = pi_k * N(x_n | mu_k, Sigma_k)
                              / sum_j pi_j * N(x_n | mu_j, Sigma_j)
        
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
        - N_k     = sum_n gamma(z_nk)          (effective count for component k)
        - pi_k    = N_k / N
        - mu_k    = sum_n gamma(z_nk) * x_n / N_k
        - Sigma_k = sum_n gamma(z_nk) * (x_n - mu_k)(x_n - mu_k)^T / N_k
        
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
        
        Uses the log-sum-exp trick: log(sum_i exp(x_i)) = m + log(sum_i exp(x_i - m))
        with m = max(x). Subtracting the max keeps the largest exponent at exp(0) = 1,
        so nothing overflows and the smallest terms simply underflow to 0 harmlessly.

        Parameters:
        -----------
        arr : np.ndarray
            Values in log space
        axis : int or None, default=None
            Axis to reduce over (None reduces the whole array)
        keepdims : bool, default=False
            Keep the reduced axis with length 1 (handy for broadcasting a
            normalisation back onto the original array)

        Returns:
        --------
        out : np.ndarray or float
            log(sum(exp(arr))) reduced over `axis`
        """
        # Always reduce with keepdims=True first: that shape broadcasts against
        # `arr` no matter which axis was reduced, then squeeze at the very end.
        arr_max = np.max(arr, axis=axis, keepdims=True)

        out = np.log(np.sum(np.exp(arr - arr_max), axis=axis, keepdims=True))
        out = out + arr_max

        if not keepdims:
            out = np.squeeze(out, axis=axis) if axis is not None else out.reshape(())[()]

        return out

    def _check_array(self, X):
        """
        Coerce input into a float 2-D array of shape (n_samples, n_features)

        Accepts NumPy arrays and plain Python lists. A 1-D input is read as
        n_samples of a SINGLE feature - i.e. reshaped to (n_samples, 1) - which is
        the natural reading for 1-D density estimation.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(
                "X must be 1-D or 2-D, got a %d-dimensional array." % X.ndim
            )

        return X

    def _check_is_fitted(self):
        """Raise a clear error if the model has not been fitted yet"""
        if self.means_ is None or self.weights_ is None or self.covariances_ is None:
            raise ValueError(
                "This GaussianMixtureModel instance is not fitted yet. "
                "Call fit(X) before using predict / predict_proba / score / "
                "score_samples / sample / bic / aic."
            )

    def fit(self, X, y=None):
        """
        Estimate GMM parameters using Expectation-Maximization (EM) algorithm

        EM Algorithm:
        1. Initialize parameters (means, covariances, weights)
        2. E-step: Compute responsibilities (which component generated each point)
        3. M-step: Update parameters based on responsibilities
        4. Repeat until convergence (log-likelihood stops improving)

        Steps 1-4 are repeated n_init times from different random starts and the
        run with the highest final log-likelihood wins, because EM only ever finds
        a local maximum.

        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data. A 1-D input is treated as n_samples x 1 feature.
        y : ignored
            Not used, present for API consistency

        Returns:
        --------
        self : GaussianMixtureModel
            Fitted model (also sets labels_, weights_, means_, covariances_,
            converged_, n_iter_ and lower_bound_)
        """
        X = self._check_array(X)
        n_samples, n_features = X.shape

        if self.covariance_type not in ('full', 'diag', 'spherical', 'tied'):
            raise ValueError(
                "Unknown covariance_type: %r. Expected one of "
                "'full', 'diag', 'spherical', 'tied'." % (self.covariance_type,)
            )

        if self.n_components > n_samples:
            raise ValueError(
                "n_components=%d cannot exceed n_samples=%d - there are not enough "
                "points to seed that many Gaussians."
                % (self.n_components, n_samples)
            )

        # Re-apply the seed here (not only in __init__) so that calling fit() twice
        # on the same object gives the same answer twice.
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # converged_ is a property of THIS fit, so clear any previous fit's verdict
        self.converged_ = False

        best_log_likelihood = -np.inf
        best_params = None

        # n_init independent restarts (n_init=1 by default = a single EM run)
        for _ in range(self.n_init):
            # Initialize parameters
            self._initialize_parameters(X)

            prev_log_likelihood = -np.inf
            log_likelihood = -np.inf
            converged = False
            iteration = -1  # stays -1 if max_iter == 0, so n_iter_ becomes 0

            # EM iterations
            for iteration in range(self.max_iter):
                # E-step: Compute responsibilities
                responsibilities, log_likelihood = self._e_step(X)

                # M-step: Update parameters
                self._m_step(X, responsibilities)

                # Check convergence
                change = log_likelihood - prev_log_likelihood

                if abs(change) < self.tol:
                    converged = True
                    break

                prev_log_likelihood = log_likelihood

            # Keep this restart only if it beat every previous one
            if best_params is None or log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = (
                    np.array(self.weights_, copy=True),
                    np.array(self.means_, copy=True),
                    np.array(self.covariances_, copy=True),
                    converged,
                    iteration + 1,
                )

        # Restore the parameters of the best restart
        (self.weights_, self.means_, self.covariances_,
         self.converged_, self.n_iter_) = best_params
        self.lower_bound_ = best_log_likelihood

        # Clustering-family convenience attribute: labels of the training data
        self.labels_ = self.predict(X)

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
        self._check_is_fitted()
        X = self._check_array(X)
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def fit_predict(self, X, y=None):
        """
        Fit the model and return the component labels of the training data

        Convenience method of the clustering family - equivalent to
        `model.fit(X).predict(X)`, and identical to `model.fit(X).labels_`.

        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data
        y : ignored
            Not used, present for API consistency

        Returns:
        --------
        labels : np.ndarray, shape (n_samples,)
            Component labels (0 to n_components-1)
        """
        self.fit(X)
        return self.labels_

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
        self._check_is_fitted()
        X = self._check_array(X)
        responsibilities, _ = self._e_step(X)
        return responsibilities

    def score_samples(self, X):
        """
        Compute the log-likelihood of EACH sample under the mixture

        This is the per-point density, not a per-component one:

            log p(x_n) = log( sum_k pi_k * N(x_n | mu_k, Sigma_k) )

        computed with the log-sum-exp trick. score(X) is exactly the sum of this
        vector. Low values mark points the model finds surprising, which is what
        makes GMM usable as an anomaly detector.

        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to evaluate

        Returns:
        --------
        log_prob : np.ndarray, shape (n_samples,)
            Log-likelihood of each sample
        """
        self._check_is_fitted()
        X = self._check_array(X)

        # log(pi_k * N(x | mu_k, Sigma_k)) for every sample and component ...
        weighted_log_prob = self._estimate_log_gaussian_prob(X) + np.log(self.weights_)

        # ... then marginalise over the components, stably.
        return self._log_sum_exp(weighted_log_prob, axis=1)

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
            The TOTAL (summed) log-likelihood over all samples,
            sum_n log( sum_k pi_k * N(x_n | mu_k, Sigma_k) ).
            NOTE: scikit-learn's GaussianMixture.score returns the per-sample
            MEAN instead, so compare this against sk.score(X) * len(X); divide by
            len(X) yourself if you want a figure that is comparable across
            datasets of different sizes. bic(), aic() and the tol convergence
            check all rely on this summed convention.
        """
        self._check_is_fitted()
        X = self._check_array(X)
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
        self._check_is_fitted()

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
        self._check_is_fitted()
        X = self._check_array(X)
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
        self._check_is_fitted()
        X = self._check_array(X)
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

# Calibrate the threshold on the TRAINING (normal) data only.
# Thresholding over all of X would let the injected anomalies help choose the
# cut-off that is supposed to catch them, and would pin the number of detections
# at exactly 5% of X by construction instead of learning it.
# score_samples returns log p(x_n) for every row in one vectorised call.
normal_log_likelihoods = gmm.score_samples(X_normal)
threshold = np.percentile(normal_log_likelihoods, 5)  # bottom 5% of normal data

# Now score every point (normal + anomalies) against that fixed threshold
log_likelihoods = gmm.score_samples(X)

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
confidence = np.max(probabilities, axis=1)
ambiguous_count = np.sum(confidence < 0.6)

print(f"\nAmbiguous sounds (confidence < 60%): {ambiguous_count}")

# Always show the least-confident frames, even when none crosses the 60% bar -
# they are the soft assignments the hard labels above throw away.
print("\nThree least-confident frames (soft assignment in action):")
for i in np.argsort(confidence)[:3]:
    print(f"  Frame {i}: probabilities {np.round(probabilities[i], 4)} "
          f"-> cluster {labels[i]}")
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

# Always surface the most uncertain days, even if none crosses the 0.4 bar
print("\nThree most uncertain days (best candidates for a regime change):")
for i in np.argsort(regime_uncertainty)[-3:][::-1]:
    print(f"  Day {i}: regime probabilities {np.round(probabilities[i], 4)} "
          f"-> uncertainty {regime_uncertainty[i]:.4f}")
print("(High uncertainty indicates market conditions are changing)")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _21_gmm.py
    # Requires numpy only. All output is ASCII.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Build three genuinely elliptical, correlated Gaussian blobs ---
    # Elliptical on purpose: this is exactly what separates GMM from k-Means,
    # which can only ever draw spherical clusters. The centres are also placed
    # close enough that the blobs genuinely overlap, so that some points really
    # do belong to two components - otherwise every posterior would be 1.000 and
    # the "soft" in soft clustering would never be visible.
    mu_list = [np.array([0.0, 0.0]), np.array([3.5, 3.5]), np.array([3.5, 0.0])]
    cov_list = [
        np.array([[1.0, 0.7], [0.7, 0.8]]),    # tilted, positively correlated
        np.array([[2.0, -0.9], [-0.9, 1.0]]),  # tilted the other way
        np.array([[0.6, 0.0], [0.0, 1.5]]),    # axis-aligned, tall and thin
    ]
    blobs = [np.random.multivariate_normal(mu_list[k], cov_list[k], 100)
             for k in range(3)]
    X = np.vstack(blobs)
    y_true = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # Shuffle before slicing so train and test cover the same region of space
    order = np.random.permutation(len(X))
    X, y_true = X[order], y_true[order]
    X_train, X_test = X[:225], X[225:]
    y_train, y_test = y_true[:225], y_true[225:]

    print("=" * 58)
    print("DEMO 1 - Soft clustering: 3 elliptical Gaussian blobs")
    print("=" * 58)

    gmm = GaussianMixtureModel(
        n_components=3,
        covariance_type='full',
        random_state=42,
        max_iter=200
    )
    gmm.fit(X_train)

    train_ll = gmm.score(X_train)
    test_ll = gmm.score(X_test)

    print(f"Converged : {gmm.converged_} after {gmm.n_iter_} EM iterations")
    print(f"Train log-likelihood : {train_ll:10.3f} "
          f"({train_ll / len(X_train):7.3f} per sample)")
    print(f"Test  log-likelihood : {test_ll:10.3f} "
          f"({test_ll / len(X_test):7.3f} per sample)")
    print("  (score() returns the TOTAL; sklearn returns the per-sample mean)")

    print("\nMixing weights pi_k :", np.round(gmm.weights_, 3))
    print("Component means mu_k :")
    for k, mean in enumerate(gmm.means_):
        print(f"  component {k}: [{mean[0]:6.3f} {mean[1]:6.3f}]")

    # The whole point of GMM: some points genuinely belong to two components.
    proba = gmm.predict_proba(X_test)
    least_sure = np.argsort(proba.max(axis=1))[:5]
    print("\n5 least-confident held-out points (this is 'soft' clustering):")
    print("       (x, y)         ->  P(comp 0)  P(comp 1)  P(comp 2)  hard")
    for i in least_sure:
        print(f"  ({X_test[i, 0]:6.2f}, {X_test[i, 1]:6.2f})  ->  "
              f"{proba[i, 0]:9.3f}  {proba[i, 1]:9.3f}  {proba[i, 2]:9.3f}"
              f"  -> {np.argmax(proba[i])}")

    # Cluster purity: cluster labels are arbitrary, so score each cluster by the
    # true class that dominates it. 100% means every cluster is pure.
    def purity(labels, truth, n_clusters):
        correct = 0
        for k in range(n_clusters):
            members = truth[labels == k]
            if len(members) > 0:
                correct += np.bincount(members).max()
        return correct / len(truth)

    print(f"\nTrain cluster purity : {purity(gmm.labels_, y_train, 3):.2%}")
    print(f"Test  cluster purity : {purity(gmm.predict(X_test), y_test, 3):.2%}")

    # --- Demo 2: how many components? Let BIC decide. ---
    print("\n" + "=" * 58)
    print("DEMO 2 - Choosing K with BIC / AIC")
    print("=" * 58)
    print(f"{'K':>3} {'BIC':>12} {'AIC':>12} {'log-lik':>12}")
    print("-" * 42)

    bic_scores = []
    for k in range(1, 6):
        m = GaussianMixtureModel(n_components=k, covariance_type='full',
                                 random_state=42, max_iter=200)
        m.fit(X_train)
        bic_scores.append(m.bic(X_train))
        print(f"{k:>3} {m.bic(X_train):>12.2f} {m.aic(X_train):>12.2f} "
              f"{m.score(X_train):>12.2f}")

    best_k = int(np.argmin(bic_scores)) + 1
    print(f"\nBIC is minimised at K = {best_k} (true K = 3) -> lower BIC is better")

    # --- Demo 3: GMM is generative, k-Means is not ---
    print("\n" + "=" * 58)
    print("DEMO 3 - GMM is generative: sample from the fitted model")
    print("=" * 58)

    X_gen, comp_gen = gmm.sample(n_samples=300)
    shares = np.bincount(comp_gen, minlength=3) / len(comp_gen)

    print("Component share of 300 generated points vs the fitted weights:")
    print("  generated shares :", np.round(shares, 3))
    print("  fitted weights   :", np.round(gmm.weights_, 3))
    print("\nFeature statistics, real training data vs generated data:")
    print(f"  real  mean {np.round(X_train.mean(axis=0), 2)}   "
          f"std {np.round(X_train.std(axis=0), 2)}")
    print(f"  fake  mean {np.round(X_gen.mean(axis=0), 2)}   "
          f"std {np.round(X_gen.std(axis=0), 2)}")
    print("\nThe generated cloud matches the real one because GMM learned the "
          "whole density,")
    print("not just the cluster centres - that is what k-Means cannot do.")
