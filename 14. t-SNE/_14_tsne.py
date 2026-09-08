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

    The Three Formulas This Class Embeds:

        1. Perplexity / bandwidth search (per point i)
               p(j|i) = exp(-beta_i * ||x_i - x_j||^2) / sum_{k != i} exp(-beta_i * ||x_i - x_k||^2)
               H(P_i) = -sum_j p(j|i) * log2(p(j|i))        <- entropy in BITS
               Perplexity(P_i) = 2^H(P_i)
           Therefore the binary search targets  H_target = log2(perplexity).
           The base MUST be 2 on both sides. Using ln on one side and log2 on the
           other silently solves for 2^ln(perplexity) instead (perplexity 30 -> 10.56).
           Here beta_i = 1 / (2 * sigma_i^2), so a larger beta means a narrower Gaussian.
           The joint matrix is then symmetrized:  P_ij = (p(j|i) + p(i|j)) / (2n).

        2. Low-dimensional affinities (Student t with 1 degree of freedom)
               q_ij = (1 + ||y_i - y_j||^2)^-1 / sum_{k != l} (1 + ||y_k - y_l||^2)^-1
           The heavy tail is what fixes the "crowding problem".

        3. Gradient of the cost C = KL(P||Q)
               dC/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^-1
           The factor 4 comes from the chain rule (2 from d/dy of the squared
           distance, 2 more because each pair (i, j) appears twice in the
           symmetric double sum).

    Simplifications vs. canonical t-SNE (see "Simplifications vs. Canonical
    t-SNE" in _14_tsne.md for the full discussion and the measurements):
        - Plain momentum instead of the reference implementation's adaptive
          per-parameter "gains" with min_gain=0.01. Measured cost on planted
          blobs: none. On sklearn's make_blobs(n_samples=200, centers=4,
          n_features=10, cluster_std=1.0, random_state=42), perplexity 30,
          1000 iterations, this code reaches KL 0.2309 and sklearn's exact
          gains-based solver 0.2363 - a gap smaller than the spread either
          solver shows across three seeds.
        - Exact O(n^2) forces, no Barnes-Hut quadtree. This is the real
          limitation: complete runs, timed end to end, take 9.6 s for 400
          points x 1000 iterations and 190 s (3.2 min) for all 1797 digits.
          Use sklearn beyond a few thousand.
        - Random initialization only, no PCA init (sklearn's current default),
          so the arrangement of clusters relative to each other varies with
          random_state.
        - No transform() for new points - and sklearn.manifold.TSNE has none
          either. t-SNE optimizes positions, not a reusable mapping.
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
            - Hard limits, both enforced by fit_transform (ValueError):
              1 < perplexity < n_samples. The upper one holds because the
              largest entropy reachable with n-1 neighbours is log2(n-1). The
              lower one holds because the target entropy log2(perplexity) is
              then <= 0 bits, and only a degenerate point mass has entropy 0 -
              reachable only as beta -> infinity. Left unguarded, the search
              drives beta up until exp(-d^2 * beta) underflows and whole rows
              of P collapse to zero: measured on the demo blobs, sum(P) = 0.88
              instead of 1.0 at perplexity=1 (18 of the 150 rows gone), and
              ~0.0 at anything below 1 (every row gone).

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
            Seed for the random initialization of the embedding
            - Uses a private np.random.RandomState, so it never disturbs the
              caller's global numpy random stream
            - None: a fresh, non-reproducible initialization each run

        early_exaggeration : float, default=12.0
            Factor to multiply P values by in early learning
            - Helps create tight clusters that can separate later
            - Typical range: 4-24
            
        early_exaggeration_iter : int, default=250
            Number of iterations for early exaggeration phase
            - The momentum schedule switches at the same iteration (0.5 -> 0.8).
              Tying the two phases to one constant is sklearn's design
              (_EXPLORATION_MAX_ITER = 250 ends both at once). Van der Maaten's
              reference tsne.py keeps them apart - momentum flips at iteration
              20, exaggeration stops at 100 - and the 2008 paper flips momentum
              at 250 while exaggerating only the first 50 iterations.

        min_grad_norm : float, default=1e-7
            Convergence threshold - stops if gradient norm < this value
            
        verbose : int, default=0
            Verbosity level
            - 0: Silent
            - 1: Phase banners, KL divergence every 50 iterations, final KL
            - 2: Everything in level 1, plus the achieved perplexity and the
              mean sigma found by the per-point bandwidth search (a direct
              check that Perplexity = 2^H(P_i) really was solved)
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
        # Efficient computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*dot(x, y)
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

        # Target entropy based on perplexity.
        # Perplexity = 2^(entropy), so entropy = log2(perplexity).
        # The entropy below is measured in BITS (np.log2), so the target must be
        # log2 as well. Using np.log here would silently solve for a perplexity of
        # 2^ln(perplexity) instead (a request of 30 would become 10.56).
        target_entropy = np.log2(target_perplexity)

        # Diagnostics for verbose=2: the beta and entropy each point settled on
        betas = np.ones(n)
        entropies = np.zeros(n)

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
                
                # Compute Shannon entropy in bits: H = -sum(p * log2(p))
                # Clip p instead of shifting it, so tiny probabilities contribute
                # ~0 rather than biasing every term by log2(p + 1e-8).
                entropy = -np.sum(P_i * np.log2(np.maximum(P_i, 1e-12)))

                # Record what THIS beta achieved (verbose=2 diagnostics).
                # Written here, not after the loop, because beta is advanced
                # to its next candidate below before the loop can exit on
                # exhaustion rather than on the break.
                betas[i] = beta
                entropies[i] = entropy

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

        if self.verbose > 1:
            # Direct check of the invariant Perplexity = 2^H(P_i)
            achieved = 2.0 ** entropies
            sigmas = np.sqrt(1.0 / (2.0 * betas))
            print(f"[t-SNE] Achieved perplexity: mean={achieved.mean():.3f}, "
                  f"min={achieved.min():.3f}, max={achieved.max():.3f} "
                  f"(requested {target_perplexity})")
            print(f"[t-SNE] Mean sigma from bandwidth search: {sigmas.mean():.4f}")

        # Symmetrize: P_ij = (P_i|j + P_j|i) / (2n)
        P = (P + P.T) / (2 * n)

        # Ensure minimum probability for numerical stability
        P = np.maximum(P, 1e-12)

        return P
    
    def _compute_low_dim_affinities(self, Y, return_num=False):
        """
        Compute affinities (similarities) in low-dimensional space using Student t-distribution

        Uses Student t-distribution with 1 degree of freedom (Cauchy distribution)
        This helps prevent "crowding problem" where moderate distances in high-D
        get crowded in low-D space.

        Parameters:
        -----------
        Y : numpy array of shape (n_samples, n_components)
            Low-dimensional embedding
        return_num : bool, default=False
            If True, also return the un-normalized numerator
            num_ij = (1 + ||y_i - y_j||^2)^-1 (zero on the diagonal).
            The gradient needs exactly this matrix, so returning it lets
            fit_transform avoid recomputing the pairwise distances a second
            time in the same iteration. Default False keeps the original
            single-value return for any existing caller.

        Returns:
        --------
        Q : numpy array of shape (n_samples, n_samples)
            Similarity matrix in low-dimensional space
        num : numpy array of shape (n_samples, n_samples), only if return_num
            The un-normalized Student-t numerator (1 + ||y_i - y_j||^2)^-1
        """
        # Compute squared Euclidean distances
        distances = self._compute_pairwise_distances(Y)

        # Student t-distribution with 1 degree of freedom
        # Q_ij = (1 + ||y_i - y_j||^2)^(-1) / sum(1 + ||y_k - y_l||^2)^(-1)
        num = 1 / (1 + distances)

        # Set diagonal to zero (point compared to itself)
        np.fill_diagonal(num, 0)

        # Normalize to get probabilities
        sum_Q = np.sum(num)
        if sum_Q == 0:
            sum_Q = 1e-8
        Q = num / sum_Q

        # Ensure minimum probability for numerical stability
        Q = np.maximum(Q, 1e-12)

        if return_num:
            return Q, num
        return Q

    def _compute_gradient(self, P, Q, Y, inv_distances=None):
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
        inv_distances : numpy array of shape (n_samples, n_samples) or None
            Pre-computed (1 + ||y_i - y_j||^2)^-1 with a zero diagonal, as
            returned by _compute_low_dim_affinities(Y, return_num=True).
            If None (the default) it is computed here, so the method still
            works when called on its own.

        Returns:
        --------
        gradient : numpy array of shape (n_samples, n_components)
            Gradient of cost function
        """
        n = Y.shape[0]

        # Compute pairwise differences in Y
        # Y_diff[i,j] = y_i - y_j
        Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

        if inv_distances is None:
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
        X : array-like of shape (n_samples, n_features)
            High-dimensional input data. A plain Python list of lists is
            accepted; a 1-D array is treated as n_samples single-feature points.

        Returns:
        --------
        Y : numpy array of shape (n_samples, n_components)
            Embedded coordinates in low-dimensional space

        Raises:
        -------
        ValueError
            If X is empty, has more than 2 dimensions, or if perplexity falls
            outside 1 < perplexity < n_samples. On either side the entropy
            target log2(perplexity) is unreachable: the most a point can spread
            over n-1 neighbours is log2(n-1) bits, and the least is 0 bits,
            which only a degenerate point mass attains.
        """
        # Accept lists and 1-D input, as the docstring promises
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 1-D or 2-D, got an array with {X.ndim} dimensions."
            )
        if X.shape[0] < 2:
            raise ValueError(
                f"t-SNE needs at least 2 samples, got {X.shape[0]}."
            )

        n_samples, n_features = X.shape

        if self.perplexity <= 1:
            raise ValueError(
                f"perplexity ({self.perplexity}) must be greater than 1. The "
                f"bandwidth search targets an entropy of log2(perplexity) bits, "
                f"which is 0 or negative here, and only a degenerate point mass "
                f"reaches entropy 0 - i.e. only as beta -> infinity, where "
                f"exp(-d^2 * beta) underflows and whole rows of P collapse to "
                f"zero, so P no longer sums to 1 and the KL objective is "
                f"meaningless. Use the documented range 5-50, and keep "
                f"perplexity < n_samples / 3."
            )

        if self.perplexity >= n_samples:
            raise ValueError(
                f"perplexity ({self.perplexity}) must be less than n_samples "
                f"({n_samples}). The largest entropy reachable with "
                f"{n_samples - 1} neighbours is log2({n_samples - 1}) = "
                f"{np.log2(n_samples - 1):.3f} bits, i.e. a perplexity of "
                f"{n_samples - 1}. A good rule of thumb is "
                f"perplexity < n_samples / 3."
            )

        # Private random stream: seeding this never disturbs the caller's
        # global numpy RNG. RandomState(seed) reproduces the same numbers that
        # np.random.seed(seed) would have produced.
        rng = np.random.RandomState(self.random_state)

        if self.verbose > 0:
            print(f"[t-SNE] Computing pairwise distances...")
        
        # Step 1: Compute pairwise distances
        distances = self._compute_pairwise_distances(X)
        
        if self.verbose > 0:
            print(f"[t-SNE] Computing P-values...")
        
        # Step 2: Compute joint probabilities P
        P = self._compute_joint_probabilities(distances, self.perplexity)
        
        # Step 3: Initialize Y randomly (small values near origin)
        Y = rng.randn(n_samples, self.n_components) * 1e-4

        # For momentum-based gradient descent
        Y_velocity = np.zeros_like(Y)

        # Bind this before the loop so that n_iter=0 is a legal (if useless)
        # request that returns the random initialization instead of raising
        # UnboundLocalError on the n_iter_ line below.
        iteration = -1

        if self.verbose > 0:
            print(f"[t-SNE] Starting optimization with {self.n_iter} iterations...")

        # Step 4: Gradient descent optimization
        for iteration in range(self.n_iter):
            # Apply early exaggeration in initial iterations
            if iteration < self.early_exaggeration_iter:
                P_effective = P * self.early_exaggeration
            else:
                P_effective = P
            
            # Compute low-dimensional affinities.
            # Ask for the un-normalized numerator as well: it is exactly the
            # (1 + ||y_i - y_j||^2)^-1 matrix the gradient needs, so the O(n^2 d)
            # distance computation happens once per iteration instead of twice.
            Q, num = self._compute_low_dim_affinities(Y, return_num=True)

            # Compute gradient
            gradient = self._compute_gradient(P_effective, Q, Y, inv_distances=num)

            # Check for convergence
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < self.min_grad_norm:
                if self.verbose > 0:
                    print(f"[t-SNE] Converged at iteration {iteration}")
                break
            
            # Momentum schedule. The switch happens at the same iteration where
            # early exaggeration ends - the way sklearn does it, where one
            # constant (_EXPLORATION_MAX_ITER) ends both phases. The idea is
            # gentle momentum while the exaggerated clusters form, then a
            # larger momentum to refine them.
            if iteration < self.early_exaggeration_iter:
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
        
        # Store final results.
        # Recompute Q from the Y we are about to return: the Q left over from
        # the loop belongs to the embedding *before* the last gradient step, so
        # reusing it would report the KL of a configuration we never returned.
        # Note P (not P_effective) is used, so the reported KL is always the
        # true cost, never the exaggerated one.
        self.embedding_ = Y
        Q = self._compute_low_dim_affinities(Y)
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

# This implementation is the EXACT O(n^2) t-SNE: every iteration touches all
# n*n pairs. All 1797 digits at 1000 iterations takes ~3.2 minutes (190 s, a
# complete run timed end to end; the cost grows as n^2, so it is very
# sensitive to machine and dataset size).
# Subsample to keep the example interactive. The first 400 rows of
# load_digits already cover all ten classes.
X = digits.data[:400]    # 400 samples, 64 features
y = digits.target[:400]  # Labels (0-9)

# Apply t-SNE (measured: about 10 seconds, final KL divergence ~= 0.36)
# Do NOT cut n_iter much below 1000 here: at 500 iterations this dataset is
# still mid-descent (KL ~= 1.65) and the ten clusters have not separated yet.
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

# Four full-size fits would take about 13 minutes with the exact O(n^2)
# algorithm (4 x the 190 s a full 1797-digit run costs), so subsample. The
# four fits below take about 40 s together. Note perplexity must stay below
# n_samples: with only 400 points here, 100 is still legal (and under the
# n_samples/3 rule of thumb), but on a 90-point dataset fit_transform would
# raise.
X = digits.data[:400]
y = digits.target[:400]

# Try different perplexity values
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, perplexity in enumerate(perplexities):
    print(f"\nRunning t-SNE with perplexity={perplexity}")

    tsne = TSNE(n_components=2, perplexity=perplexity,
                learning_rate=200, n_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X)
    print(f"  final KL divergence: {tsne.kl_divergence_:.4f}")

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
#
# Sanity check you can run yourself: pass verbose=2 and confirm the printed
# "Achieved perplexity" matches what you asked for. That is the invariant
# Perplexity = 2^H(P_i) with H in bits.
"""

"""
USAGE EXAMPLE 3: Visualizing MNIST Fashion Dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load Fashion MNIST (takes a moment to download first time)
# Using a subset for faster computation.
# as_frame=False returns plain numpy arrays instead of a pandas DataFrame.
# fit_transform also calls np.asarray, so either would work - keep it explicit.
# Keep n small - the exact O(n^2) implementation costs about (n/400)^2 times
# as much per iteration as the 400-point examples above.
fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto', as_frame=False)
X = np.asarray(fashion.data)[:1000] / 255.0  # Normalize to [0, 1]
y = np.asarray(fashion.target)[:1000].astype(int)

# Class names
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Apply t-SNE (about 30 s to fit 1000 points for 500 iterations - timed on an
# array of the same shape, since the cost depends on n and n_features, not on
# the values - plus the one-time download, which needs a network connection)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
            n_iter=500, random_state=42, verbose=1)
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
from sklearn.datasets import load_digits

digits = load_digits()
# Subsampled for runtime: n_components=3 makes the (n, n, 3) difference
# tensor 50% larger than the 2D case, so keep n modest.
X = digits.data[:400]
y = digits.target[:400]

# 3D t-SNE (measured: about 11 seconds)
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
X = digits.data[:400]    # subsampled so the t-SNE half finishes in ~10 s
y = digits.target[:400]

# Apply PCA (instant - closed form)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE (iterative - this is the slow half)
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
X = digits.data[:250]  # Smaller subset for faster experimentation
y = digits.target[:250]  # 250 rows still cover all ten digit classes

# Test different parameter combinations
# Nine fits: keep n and n_iter small or this grid runs for many minutes.
# At 250 points x 500 iterations each fit is ~1.7 s, so the grid is ~15 s.
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
n_samples_per_class = 50   # 200 points total: the fit takes about 2 s
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


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _14_tsne.py
    # numpy only, seeded, finishes in a couple of seconds.
    # ----------------------------------------------------------------
    np.random.seed(42)

    def make_blobs(n_per_cluster=50, n_features=10, separation=6.0):
        """Three well-separated Gaussian blobs in n_features dimensions."""
        direction = np.random.randn(n_features)
        direction = direction / np.linalg.norm(direction)
        offsets = [0.0, separation, -separation]

        chunks, labels = [], []
        for cluster_id, offset in enumerate(offsets):
            center = offset * direction
            chunks.append(center + np.random.randn(n_per_cluster, n_features))
            labels.extend([cluster_id] * n_per_cluster)
        return np.vstack(chunks), np.array(labels)

    def neighbor_preservation(X_high, Y_low, k=10):
        """
        Fraction of each point's k nearest neighbours in high-D that are still
        among its k nearest neighbours in the embedding, averaged over points.

        t-SNE has no train/test split and no predict(), so this is the honest
        quality measure: it asks the only question t-SNE promises to answer,
        "did local neighbourhoods survive the projection?". 1.0 is perfect.
        Uses the same ||x - y||^2 = ||x||^2 + ||y||^2 - 2*dot(x, y) expansion
        as _compute_pairwise_distances.
        """
        def nearest(Z):
            sq = np.sum(np.square(Z), axis=1)
            D = sq[:, np.newaxis] + sq[np.newaxis, :] - 2 * np.dot(Z, Z.T)
            np.fill_diagonal(D, np.inf)   # never count a point as its own neighbour
            return np.argsort(D, axis=1)[:, :k]

        high_nn = nearest(X_high)
        low_nn = nearest(Y_low)
        shared = [len(set(high_nn[i]) & set(low_nn[i])) for i in range(len(X_high))]
        return float(np.mean(shared)) / k

    X, labels = make_blobs()

    print("=" * 60)
    print("DEMO 1 - Planted clusters survive the 10-D -> 2-D embedding")
    print("=" * 60)
    print(f"Input: {X.shape[0]} points in {X.shape[1]}-D, "
          f"{len(np.unique(labels))} planted Gaussian blobs")
    print("t-SNE is unsupervised: the labels below are ground truth used only")
    print("to score the embedding, never shown to the algorithm.")

    model = TSNE(n_components=2, perplexity=15, learning_rate=200,
                 n_iter=500, random_state=42)
    Y = model.fit_transform(X)

    # A random 2-D layout gives the floor this metric should be judged against.
    Y_random = np.random.randn(*Y.shape)
    baseline = neighbor_preservation(X, Y_random, k=10)

    print(f"\nEmbedding shape   : {Y.shape}")
    print(f"Iterations run    : {model.n_iter_}")
    print(f"Final KL(P||Q)    : {model.kl_divergence_:.4f}")
    print(f"kNN(10) preserved : {neighbor_preservation(X, Y, k=10):.3f} "
          f"(1.0 = perfect; a random 2-D layout scores {baseline:.3f})")
    print("Note: the blobs are isotropic 10-D Gaussians, so their internal")
    print("neighbour order genuinely cannot all fit in 2-D. Beating the random")
    print("baseline by this much is the local structure t-SNE did keep.")

    print("\nPer-cluster geometry in the embedding:")
    print("  cluster    centroid (x, y)        mean spread")
    centroids = []
    for cluster_id in np.unique(labels):
        pts = Y[labels == cluster_id]
        centroid = pts.mean(axis=0)
        spread = np.mean(np.sqrt(np.sum((pts - centroid) ** 2, axis=1)))
        centroids.append(centroid)
        print(f"    {cluster_id}      ({centroid[0]:8.2f}, {centroid[1]:8.2f})"
              f"      {spread:6.2f}")

    centroids = np.array(centroids)
    gaps = [np.linalg.norm(centroids[i] - centroids[j])
            for i in range(len(centroids)) for j in range(i + 1, len(centroids))]
    print(f"  Smallest gap between cluster centroids: {min(gaps):.2f}")
    print("  Separation >> spread means the clusters did not merge.")

    print("\nSample embedded points (label, x, y):")
    for i in [0, 40, 60, 110, 140]:
        print(f"  label={labels[i]}  x={Y[i, 0]:8.3f}  y={Y[i, 1]:8.3f}")

    print("\n" + "=" * 60)
    print("DEMO 2 - What perplexity actually does")
    print("=" * 60)
    print("Perplexity is the effective number of neighbours each point keeps:")
    print("  Perplexity = 2^H(P_i), H measured in bits.")
    print("A small perplexity makes P concentrate on a handful of nearest")
    print("neighbours, whose exact ordering 2-D cannot reproduce. A larger")
    print("perplexity spreads P over the whole blob, which is a smoother and")
    print("easier target - so on well-separated blobs KL FALLS as perplexity")
    print("rises. On data with fine sub-structure the trade-off reverses.")
    print("(Pass verbose=2 to print the perplexity actually achieved.)")
    print("\n  perplexity    final KL    kNN(10) preserved")
    for perplexity in [5, 15, 30]:
        sweep = TSNE(n_components=2, perplexity=perplexity, learning_rate=200,
                     n_iter=400, random_state=42)
        Y_sweep = sweep.fit_transform(X)
        preserved = neighbor_preservation(X, Y_sweep, k=10)
        print(f"     {perplexity:5.1f}       {sweep.kl_divergence_:7.4f}"
              f"           {preserved:.3f}")

    print("\nDone.")

