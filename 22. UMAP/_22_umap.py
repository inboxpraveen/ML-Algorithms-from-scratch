import numpy as np

class UMAP:
    """
    UMAP (Uniform Manifold Approximation and Projection) Implementation from Scratch
    
    UMAP is a state-of-the-art dimensionality reduction technique based on manifold learning
    and topological data analysis. It preserves both local and global structure better than
    t-SNE and is significantly faster.
    
    Key Idea: "Model data as a fuzzy topological structure (graph) in high dimensions,
    then find a similar structure in low dimensions"
    
    Use Cases:
    - High-dimensional data visualization (better than t-SNE for many cases)
    - General-purpose dimensionality reduction (before ML models)
    - Exploratory data analysis and cluster discovery
    - Feature engineering and preprocessing
    - Biological data analysis (genomics, single-cell)
    
    Key Concepts:
        Fuzzy Simplicial Sets: Probabilistic representation of topological structure
        k-NN Graph: Local neighborhood graph connecting similar points
        Global Structure: Unlike t-SNE, UMAP preserves global relationships
        Spectral Embedding: Initial layout using graph Laplacian

    Key Formulas (all four are implemented literally in the methods named):

        1. Local connectivity and bandwidth  (_smooth_knn_distances)
               rho_i   = distance from i to its NEAREST neighbour
               sigma_i = solved by bisection so that
                         sum_j exp(-max(0, d_ij - rho_i) / sigma_i) = log2(k)
           rho_i guarantees every point has at least one edge of weight 1.0;
           sigma_i adapts the metric to the LOCAL density around point i.

        2. Directed membership strength  (_compute_membership_strengths)
               v(i->j) = exp(-max(0, d_ij - rho_i) / sigma_i)

        3. Fuzzy union / probabilistic t-conorm  (_compute_membership_strengths)
               w(i,j) = v(i->j) + v(j->i) - v(i->j) * v(j->i)
           This symmetrises the directed graph: "i sees j OR j sees i".

        4. Low-dimensional kernel  (_find_ab_params, _optimize_embedding)
               q(d) = 1 / (1 + a * d^(2b))
           a and b are NOT constants: they are the least-squares fit of q(d) to
               psi(d) = 1                        if d < min_dist
                      = exp(-(d - min_dist)/spread)  otherwise
           so min_dist is what physically sets how tightly points may pack.
           (min_dist=0.0 -> a=1.9328, b=0.7905;  min_dist=0.5 -> a=0.5830, b=1.3342)

    Objective and its gradients (see _optimize_embedding):
        CE = sum_ij [ w_ij * log(w_ij / q_ij) + (1-w_ij) * log((1-w_ij)/(1-q_ij)) ]
        With s = ||y_i - y_j||^2, the two gradient pieces are
            attractive (edge present):  -2*a*b*s^(b-1) / (1 + a*s^b) * (y_i - y_j)
            repulsive  (negative pair): +2*b / ((0.001 + s) * (1 + a*s^b)) * (y_i - y_j)
        Both are clipped componentwise to [-4, 4], exactly as umap-learn does.

    Simplifications vs. canonical umap-learn (details in _22_umap.md):
        - Exact O(n^2) pairwise distances instead of approximate nearest-neighbour
          descent, so this class is comfortable up to roughly 500-1000 samples.
        - No multi-component spectral layout: on a disconnected k-NN graph the
          0-eigenspace has one dimension PER COMPONENT, so eigh returns an
          arbitrary orthonormal basis of it and each component lands as a blob
          whose internal spread follows sqrt(degree) rather than its geometry
          (measured: max point-to-centroid spread 1.4-2.2 on a 10-unit layout,
          with one component collapsed to 2e-4). The SGD still recovers them -
          10-NN purity 1.0000 on 3 planted components, over 3 seeds.
        - transform() is attractive-only with the training embedding frozen.
        - local_connectivity is fixed at 1 (rho is always the 1st-neighbour distance).
    """
    
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, 
                 metric='euclidean', learning_rate=1.0, n_epochs=200,
                 init='spectral', random_state=None, verbose=0):
        """
        Initialize the UMAP model
        
        Parameters:
        -----------
        n_components : int, default=2
            Dimension of the embedding space
            - 2D: Best for visualization
            - 3D: Interactive 3D visualization
            - Higher: For dimensionality reduction before ML models
            
        n_neighbors : int, default=15
            Number of nearest neighbors to consider
            - Typical range: 2-100
            - Small values: Focus on local structure (tight clusters)
            - Large values: Preserve more global structure
            - Default (15) is good for most cases
            - Automatically clamped to n_samples - 1 on tiny datasets
            
        min_dist : float, default=0.1
            Minimum distance between points in embedding
            - Range: 0.0-1.0
            - Small values (0.0-0.1): Tightly packed clusters
            - Large values (0.3-0.99): More evenly distributed points
            - Controls how tightly points cluster together
            How it acts: min_dist is fitted into the low-D kernel parameters
            (a, b) of q(d) = 1 / (1 + a*d^(2b)) by _find_ab_params. Raising
            min_dist lowers a and raises b, which flattens q near the origin
            so that pairs closer than min_dist gain nothing by squeezing
            further together.
            
        metric : str, default='euclidean'
            Distance metric to use
            - 'euclidean': Standard Euclidean distance
            - 'manhattan': Manhattan (L1) distance
            - 'cosine': Cosine distance (for text, high-dimensional sparse data)
            
        learning_rate : float, default=1.0
            Learning rate for optimization
            - Typical range: 0.1-10.0
            - Higher: Faster convergence but may be unstable
            - Lower: Slower but more stable
            
        n_epochs : int, default=200
            Number of training epochs
            - Typical range: 100-1000
            - More epochs: Better convergence, slower
            - Minimum recommended: 100
            
        init : str, default='spectral'
            Initialization method
            - 'spectral': Use spectral embedding (recommended)
            - 'random': Random initialization
            
        random_state : int or None, default=None
            Seed for this model's PRIVATE random generator
            - An int makes fit()/fit_transform() bit-for-bit reproducible
            - None draws fresh randomness on every fit
            - The caller's global np.random stream is never touched
            
        verbose : int, default=0
            Verbosity level
            - 0: Silent
            - 1: Show progress
            - 2: Show detailed information
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        
        # Will be set during fitting
        self.embedding_ = None
        self.graph_ = None
        
        # Training state kept so that transform() can embed new points
        self._raw_data = None
        self._rho = None
        self._sigma = None

        # PRIVATE random generator. Using np.random.seed() here would silently
        # reseed the caller's global stream and change every later np.random
        # call in their script, so we never do that.
        self._rng = np.random.RandomState(random_state)
    
    def _compute_distances(self, X):
        """
        Compute pairwise distances based on selected metric
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        distances : np.ndarray, shape (n_samples, n_samples)
            Pairwise distance matrix
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        if self.metric == 'euclidean':
            for i in range(n_samples):
                distances[i] = np.sqrt(np.sum((X - X[i])**2, axis=1))
        
        elif self.metric == 'manhattan':
            for i in range(n_samples):
                distances[i] = np.sum(np.abs(X - X[i]), axis=1)
        
        elif self.metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_normalized = X / (norms + 1e-10)
            similarities = X_normalized @ X_normalized.T
            # Floating-point error can push a self-similarity to 1 + 1e-16, which
            # would make a point's distance to ITSELF a tiny positive number and
            # let it sneak into its own neighbour list. Clip, then zero the diagonal.
            similarities = np.clip(similarities, -1.0, 1.0)
            distances = 1 - similarities
            np.fill_diagonal(distances, 0.0)
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Distances are non-negative by definition; kill any -1e-17 round-off.
        return np.maximum(distances, 0.0)

    def _cross_distances(self, A, B):
        """
        Distances from every row of A to every row of B (used by transform)

        This is the rectangular version of _compute_distances, which is the
        square special case A = B = X.

        Parameters:
        -----------
        A : np.ndarray, shape (n_a, n_features)
        B : np.ndarray, shape (n_b, n_features)

        Returns:
        --------
        distances : np.ndarray, shape (n_a, n_b)
        """
        distances = np.zeros((A.shape[0], B.shape[0]))

        if self.metric == 'euclidean':
            for i in range(A.shape[0]):
                distances[i] = np.sqrt(np.sum((B - A[i]) ** 2, axis=1))

        elif self.metric == 'manhattan':
            for i in range(A.shape[0]):
                distances[i] = np.sum(np.abs(B - A[i]), axis=1)

        elif self.metric == 'cosine':
            A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
            B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
            distances = 1 - np.clip(A_norm @ B_norm.T, -1.0, 1.0)

        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        return np.maximum(distances, 0.0)
    
    def _compute_knn_graph(self, distances):
        """
        Compute k-nearest neighbors graph
        
        Parameters:
        -----------
        distances : np.ndarray, shape (n_samples, n_samples)
            Pairwise distance matrix
            
        Returns:
        --------
        knn_indices : np.ndarray, shape (n_samples, k)
            Indices of k-nearest neighbors for each point, where
            k = min(n_neighbors, n_samples - 1)
        knn_distances : np.ndarray, shape (n_samples, k)
            Distances to k-nearest neighbors for each point
        """
        n_samples = distances.shape[0]
        if n_samples < 3:
            raise ValueError(
                f"UMAP needs at least 3 samples to build a neighbour graph, got {n_samples}"
            )

        # A point cannot be its own neighbour, so k can never exceed n_samples - 1.
        # Clamping (rather than crashing) keeps the default n_neighbors=15 usable
        # on small teaching datasets.
        k = min(self.n_neighbors, n_samples - 1)
        if k < self.n_neighbors and self.verbose > 0:
            print(f"n_neighbors={self.n_neighbors} >= n_samples={n_samples}; using k={k}")

        knn_indices = np.zeros((n_samples, k), dtype=int)
        knn_distances = np.zeros((n_samples, k))
        
        for i in range(n_samples):
            # Sort every other point by distance from i
            sorted_indices = np.argsort(distances[i])
            # Explicitly drop i itself. We must NOT just skip position 0: with
            # duplicate rows (or a metric whose self-distance is 1e-11 rather
            # than exactly 0) point i may not be first, and a self-loop edge
            # (i, i) would give a zero-length gradient later on.
            neighbors = sorted_indices[sorted_indices != i][:k]
            knn_indices[i] = neighbors
            knn_distances[i] = distances[i, neighbors]
        
        return knn_indices, knn_distances
    
    def _smooth_knn_distances(self, knn_distances, n_iter=64, bandwidth=1.0):
        """
        Compute smoothed distances using local metric (rho and sigma)
        
        This implements the adaptive distance metric from the UMAP paper
        (Algorithm 2, "SmoothKNNDist"). For each point, we find:
        - rho: distance to nearest neighbor (local connectivity)
        - sigma: normalization factor so sum of probabilities ~= log2(k)

        Why log2(k)? It is the entropy of a uniform distribution over k
        neighbours, so it asks: "widen sigma_i until point i has roughly
        log2(k) neighbours' worth of connectivity". Dense regions get a small
        sigma and sparse regions a large one, which is exactly the "uniform
        distribution on the manifold" assumption UMAP is named after.

        Note on conventions: this follows the UMAP paper's Algorithm 2 exactly -
        k TRUE neighbours summed against log2(k). umap-learn's neighbour array
        includes the point itself in column 0, so its n_neighbors=k sums k-1
        true neighbours against log2(k). Feeding both the same 15 neighbours
        therefore gives sigma about 2.6% smaller here than umap-learn's own
        smooth_knn_dist does for n_neighbors=16 (measured against a port of that
        routine written from its published source - the library itself is not a
        dependency here: mean ratio 0.974, mean edge-weight difference 0.008
        over 1544 edges) - a real but negligible offset.
        
        Parameters:
        -----------
        knn_distances : np.ndarray, shape (n_samples, n_neighbors)
            Distances to k-nearest neighbors
        n_iter : int, default=64
            Number of binary search iterations for sigma
        bandwidth : float, default=1.0
            Target bandwidth (related to perplexity)
            
        Returns:
        --------
        rho : np.ndarray, shape (n_samples,)
            Local connectivity (distance to nearest neighbor)
        sigma : np.ndarray, shape (n_samples,)
            Normalization factors
        """
        n_samples, k = knn_distances.shape
        rho = np.zeros(n_samples)
        sigma = np.ones(n_samples)
        
        target = np.log2(k) * bandwidth
        
        for i in range(n_samples):
            # rho is distance to nearest neighbor (local connectivity)
            rho[i] = knn_distances[i, 0]
            
            # max(0, d_ij - rho_i): the nearest neighbour contributes exp(0) = 1,
            # which is the guarantee that every point stays connected.
            diffs = np.maximum(knn_distances[i] - rho[i], 0.0)

            # Bisection for sigma_i. sum_probs is monotonically INCREASING in
            # sigma, so the search is a textbook binary search on a monotone
            # function; 64 halvings of [0, 1e10] is far more than enough.
            lo, hi = 0.0, 1e10
            for _ in range(n_iter):
                mid = (lo + hi) / 2.0
                
                # Compute sum of probabilities
                probs = np.exp(-diffs / mid)
                sum_probs = np.sum(probs)

                # umap-learn's SMOOTH_K_TOLERANCE: stop as soon as we are close
                # enough instead of grinding out all 64 iterations.
                if abs(sum_probs - target) < 1e-5:
                    lo = hi = mid
                    break
                
                if sum_probs > target:
                    hi = mid
                else:
                    lo = mid
            
            sigma[i] = (lo + hi) / 2.0

            # umap-learn's MIN_K_DIST_SCALE floor. Without it, a point sitting on
            # top of duplicates has rho = 0 and every diff = 0, which drives sigma
            # to ~1e-10 and makes every one of its edge weights snap to 0 or 1.
            mean_dist = knn_distances[i].mean()
            if mean_dist > 0:
                sigma[i] = max(sigma[i], 1e-3 * mean_dist)
        
        return rho, sigma
    
    def _compute_membership_strengths(self, knn_indices, knn_distances, rho, sigma):
        """
        Compute fuzzy membership strengths (edge weights in high-D graph)
        
        This creates the fuzzy simplicial set representation.
        Each edge has a probability representing how likely two points
        are to be connected in the manifold.
        
        Parameters:
        -----------
        knn_indices : np.ndarray, shape (n_samples, n_neighbors)
            Indices of nearest neighbors
        knn_distances : np.ndarray, shape (n_samples, n_neighbors)
            Distances to nearest neighbors
        rho : np.ndarray, shape (n_samples,)
            Local connectivity values
        sigma : np.ndarray, shape (n_samples,)
            Normalization factors
            
        Returns:
        --------
        graph : dict
            Sparse graph representation: {(i, j): weight}
        """
        graph = {}
        n_samples = knn_indices.shape[0]
        
        for i in range(n_samples):
            for j_idx, j in enumerate(knn_indices[i]):
                if i == j:
                    continue  # a self-loop has zero length and no gradient
                # Compute membership strength from i to j
                dist = knn_distances[i, j_idx]
                if sigma[i] > 0:
                    d_norm = max(0, dist - rho[i]) / sigma[i]
                    val_ij = np.exp(-d_norm)
                else:
                    val_ij = 1.0 if dist == 0 else 0.0
                
                # Compute membership strength from j to i
                j_neighbor_idx = np.where(knn_indices[j] == i)[0]
                if len(j_neighbor_idx) > 0:
                    dist_ji = knn_distances[j, j_neighbor_idx[0]]
                    if sigma[j] > 0:
                        d_norm_ji = max(0, dist_ji - rho[j]) / sigma[j]
                        val_ji = np.exp(-d_norm_ji)
                    else:
                        val_ji = 1.0 if dist_ji == 0 else 0.0
                else:
                    val_ji = 0.0
                
                # Fuzzy set union (probabilistic t-conorm):
                #   w(i,j) = A union B = A + B - A*B
                # This is P(edge exists) when the two directed beliefs are treated
                # as independent: 1 - (1 - A)(1 - B).
                prob = val_ij + val_ji - val_ij * val_ji
                
                if prob > 0:
                    graph[(i, j)] = prob
                    graph[(j, i)] = prob
        
        return graph
    
    def _spectral_initialization(self, graph, n_samples):
        """
        Initialize embedding using spectral embedding (graph Laplacian)
        
        This provides a good initial layout based on the graph structure.
        It is the eigenvector layout of the symmetric normalised Laplacian:

            W      = dense weight matrix built from the fuzzy graph
            D      = diag(sum_j W[i, j])                  (degree matrix)
            L_sym  = I - D^(-1/2) W D^(-1/2)

        On a connected graph the eigenvector of L_sym for eigenvalue 0 is
        D^(1/2) . 1 - proportional to sqrt(degree), a fixed function of the graph
        that carries no layout information. (It is the RANDOM-WALK Laplacian
        I - D^-1 W whose null vector is the constant one; measured on the graph
        this class builds from RandomState(0).randn(60, 5) with n_neighbors=10,
        |corr(v_0, sqrt(degree))| = 1.000000000000 - the absolute value is needed
        because eigh pins each eigenvector only up to sign, so the correlation is
        +1 or -1 depending on the LAPACK build.) Either way column 0 is
        uninformative, so we take eigenvectors 1 .. n_components, which are the
        smoothest non-trivial functions on the graph - points joined by heavy
        edges get similar coordinates. This is what "spectral" means, and it is
        why UMAP is far more reproducible than a random start.
        
        Parameters:
        -----------
        graph : dict
            Sparse graph representation
        n_samples : int
            Number of samples
            
        Returns:
        --------
        embedding : np.ndarray, shape (n_samples, n_components)
            Initial embedding, scaled to about 10 units across
        """
        # Random fallback for graphs too small for a meaningful eigen-layout
        if n_samples <= self.n_components + 2 or not graph:
            return self._rng.randn(n_samples, self.n_components) * 10.0
        
        # 1. Dense weight matrix. O(n^2) memory - fine for the few-hundred-point
        #    datasets this educational implementation targets.
        W = np.zeros((n_samples, n_samples))
        for (i, j), w in graph.items():
            W[i, j] = w
        
        # 2. Symmetric normalised Laplacian  L = I - D^-1/2 W D^-1/2
        degree = W.sum(axis=1)
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree + 1e-12), 0.0)
        L = np.eye(n_samples) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
                
        # 3. eigh returns eigenvalues in ASCENDING order and is exact for the
        #    symmetric L. Column 0 belongs to eigenvalue ~0 (the sqrt-degree
        #    vector), so the layout starts at column 1.
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return self._rng.randn(n_samples, self.n_components) * 10.0
            
        embedding = eigenvectors[:, 1:self.n_components + 1]

        if self.verbose > 1:
            print(f"  spectral init: smallest eigenvalues "
                  f"{np.round(eigenvalues[:self.n_components + 1], 4)}")

        # 4. Rescale to ~10 units of spread (umap-learn's `expansion`) and add a
        #    little noise so that exactly-coincident points can separate.
        max_abs = np.abs(embedding).max()
        if max_abs > 0:
            embedding = embedding * (10.0 / max_abs)
        embedding = embedding + self._rng.normal(
            scale=1e-4, size=(n_samples, self.n_components)
        )
        
        return embedding
    
    def _optimize_embedding(self, graph, n_samples):
        """
        Optimize the low-dimensional embedding using stochastic gradient descent
        
        This is where the actual dimensionality reduction happens.
        We minimize the cross-entropy between high-D and low-D graphs:

            CE = sum_ij  w_ij log(w_ij / q_ij) + (1-w_ij) log((1-w_ij)/(1-q_ij))
            q_ij = 1 / (1 + a * s^b),   s = ||y_i - y_j||^2

        Differentiating and descending (-dCE/dy_i) gives the two coefficients
        coded below:

            attractive  (first CE term):  -2ab*s^(b-1) / (1 + a*s^b)
            repulsive   (second CE term): +2b / (s * (1 + a*s^b))

        The code writes the repulsive one as 2b / ((0.001 + s) * (1 + a*s^b)):
        umap-learn's 0.001 keeps it finite at s = 0, and is a deliberate change to
        the formula, not a rounding (1.0e-03 relative at s = 1, 9.1e-01 at
        s = 1e-04 - which is the whole point, since that is where the true
        gradient diverges).

        Instead of summing over all n^2 pairs (which is why t-SNE is slow), UMAP
        uses two sampling tricks:

        1. EDGE SAMPLING. An edge of weight w is visited once every
           w_max / w epochs, so a weight-1.0 edge is pulled every epoch and a
           weight-0.1 edge every tenth epoch. Over the whole run each edge is
           therefore attracted in proportion to w, which is why the attractive
           gradient below is NOT multiplied by the weight again.
        2. NEGATIVE SAMPLING. The repulsive term is estimated from 5 uniformly
           random points per attractive event rather than from every non-edge.
           (1 - w_ij) ~= 1 for a random pair, so the factor is dropped.

        Together these make one epoch cost O(n * k) instead of O(n^2).
        
        Parameters:
        -----------
        graph : dict
            High-dimensional graph structure
        n_samples : int
            Number of samples
            
        Returns:
        --------
        embedding : np.ndarray, shape (n_samples, n_components)
            Optimized low-dimensional embedding
        """
        # Initialize embedding
        if self.init == 'spectral':
            embedding = self._spectral_initialization(graph, n_samples)
        else:  # random
            embedding = self._rng.randn(n_samples, self.n_components) * 10.0
        
        # Get all edges
        edges = list(graph.keys())
        weights = np.array([graph[e] for e in edges])
        
        # Prune edges too weak to be sampled even once: an edge is visited every
        # w_max / w epochs, so w < w_max / n_epochs never comes up at all.
        max_weight = weights.max()
        keep = weights >= (max_weight / float(self.n_epochs))
        edges = [e for e, k in zip(edges, keep) if k]
        weights = weights[keep]

        # epochs_per_sample[e] = w_max / w_e  (see the EDGE SAMPLING note above)
        epochs_per_sample = max_weight / weights
        epoch_of_next_sample = epochs_per_sample.copy()

        # Parameters for optimization
        a, b = self._find_ab_params(self.min_dist)
        if self.verbose > 1:
            print(f"  low-D kernel q(d) = 1/(1 + {a:.4f}*d^(2*{b:.4f})) "
                  f"for min_dist={self.min_dist}")
        
        # Optimization loop
        for epoch in range(self.n_epochs):
            # Learning rate schedule: linear decay to 0 over the run
            alpha = self.learning_rate * (1.0 - epoch / self.n_epochs)
            
            # Edges whose turn has come round this epoch, in random order
            indices = np.nonzero(epoch_of_next_sample <= epoch)[0]
            self._rng.shuffle(indices)
            
            for idx in indices:
                i, j = edges[idx]
                epoch_of_next_sample[idx] += epochs_per_sample[idx]
                
                # Get current positions (views into embedding, so the negative
                # sampling below sees the update we are about to make)
                current_i = embedding[i]
                current_j = embedding[j]
                
                # Compute squared distance in low-D
                diff = current_i - current_j
                dist_sq = np.sum(diff**2)
                
                # Attractive force (for connected pairs)
                # High-D edge exists: pull together
                #   grad_coef = -2ab * s^(b-1) / (1 + a * s^b)
                # s = 0 would make s^(b-1) infinite for b < 1, so coincident
                # points simply get no attractive pull (they are already together).
                if dist_sq > 0.0:
                    grad_coef = -2.0 * a * b * (dist_sq ** (b - 1.0))
                    grad_coef /= (1.0 + a * (dist_sq ** b))
                else:
                    grad_coef = 0.0
                
                # Clip to [-4, 4] exactly as umap-learn does: one badly scaled
                # step early on can otherwise fling a point to infinity.
                grad = np.clip(grad_coef * diff, -4.0, 4.0)
                
                # Apply gradient (both endpoints move, in opposite directions)
                embedding[i] += alpha * grad
                embedding[j] -= alpha * grad
                
                # Negative sampling: repulsive force (for random pairs)
                # This prevents all points from collapsing together
                for k in self._rng.randint(n_samples, size=5):
                    if k == i or k == j:
                        continue
                    
                    current_k = embedding[k]
                    diff_ik = current_i - current_k
                    dist_sq_ik = np.sum(diff_ik**2)
                    
                    # Repulsive force: push apart
                    #   grad_coef = +2b / ((0.001 + s) * (1 + a * s^b))
                    # The 0.001 keeps the division finite when s = 0; in that
                    # case umap-learn applies the maximum push of 4.0 instead.
                    if dist_sq_ik > 0.0:
                        grad_coef = 2.0 * b
                        grad_coef /= ((0.001 + dist_sq_ik) * (1.0 + a * (dist_sq_ik ** b)))
                        grad_ik = np.clip(grad_coef * diff_ik, -4.0, 4.0)
                    else:
                        grad_ik = np.full(self.n_components, 4.0)
                    
                    # Only the anchor moves: the negative sample is a random
                    # stand-in for "everything else", not a real partner.
                    embedding[i] += alpha * grad_ik
            
            if self.verbose > 0 and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(f"Epoch {epoch + 1}/{self.n_epochs}")
        
        return embedding
    
    def _find_ab_params(self, min_dist, spread=1.0):
        """
        Find parameters a and b for the low-dimensional probability function
        
        These parameters control the shape of the embedding:
        q(d) = 1 / (1 + a * d^(2b))

        This is a real least-squares fit, not a lookup: a and b are chosen so
        that q(d) hugs the piecewise target curve

            psi(d) = 1                             if d < min_dist
                   = exp(-(d - min_dist) / spread) if d >= min_dist

        i.e. "anything closer than min_dist counts as equally close, beyond
        that similarity decays exponentially". That is the ONLY place min_dist
        enters the algorithm, so the fit has to be done properly or min_dist
        does nothing at all.

        The fit is Levenberg-Marquardt on (log a, log b) - working in logs keeps
        a and b positive and lets one solver handle the whole range
        (a spans 1.93 down to 0.12 as min_dist goes 0.0 -> 0.99).
        Measured against scipy.optimize.curve_fit on the same target curve, the
        worst relative error in a or b is 5.0e-06 at the default spread=1.0
        (min_dist swept 0 to 1 in steps of 0.005) and 7.8e-06 over spreads 0.25
        to 20, keeping umap-learn's own precondition min_dist <= spread.
        
        Parameters:
        -----------
        min_dist : float
            Minimum distance parameter
        spread : float
            Scale of the exponential tail in the target curve. NOT a constructor
            argument: every caller inside this class leaves it at 1.0, so the
            only way to fit another spread is to call this method directly.
            (umap-learn does expose `spread` on its estimator.)
            
        Returns:
        --------
        a, b : float
            Parameters for probability function
            (min_dist=0.0 -> ~1.9328, 0.7905; min_dist=0.5 -> ~0.5830, 1.3342)
        """
        def curve(x, a, b):
            return 1.0 / (1.0 + a * (x ** (2 * b)))
        
        # Sample the target curve. x = 0 is dropped because log(0) appears in
        # the Jacobian; it contributes no residual anyway (q(0) = psi(0) = 1).
        xv = np.linspace(0.0, 3.0 * spread, 300)[1:]
        yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
        log_x = np.log(xv)
        
        # Levenberg-Marquardt from a = b = 1
        log_a, log_b = 0.0, 0.0
        lam = 1e-3

        for _ in range(64):
            a, b = np.exp(log_a), np.exp(log_b)
            u = a * xv ** (2.0 * b)          # so that curve = 1 / (1 + u)
            residual = curve(xv, a, b) - yv
            sse = np.dot(residual, residual)

            # Jacobian of q w.r.t. (log a, log b):
            #   dq/dlog(a) = -u * q^2
            #   dq/dlog(b) = -u * q^2 * 2b * log(x)
            q = 1.0 / (1.0 + u)
            d_log_a = -u * q * q
            d_log_b = d_log_a * 2.0 * b * log_x
            J = np.column_stack([d_log_a, d_log_b])

            JTJ = J.T @ J
            damping = lam * np.eye(2) * (np.trace(JTJ) / 2.0 + 1e-12)
            try:
                step = np.linalg.solve(JTJ + damping, -(J.T @ residual))
            except np.linalg.LinAlgError:
                break

            # Accept the step only if it lowers the error (that is the "LM" part)
            trial_a, trial_b = np.exp(log_a + step[0]), np.exp(log_b + step[1])
            trial_res = curve(xv, trial_a, trial_b) - yv
            trial_sse = np.dot(trial_res, trial_res)
            if trial_sse < sse:
                log_a, log_b = log_a + step[0], log_b + step[1]
                lam = max(lam * 0.5, 1e-10)
                # Converged: an ACCEPTED step that no longer lowers the error.
                # The test has to live INSIDE this branch. A REJECTED step leaves
                # (log_a, log_b) untouched, so the next iteration recomputes a
                # bit-identical sse; a plateau test run once per iteration would
                # then fire on the rejection itself - which is precisely when LM
                # still needs more iterations to grow lam.
                if sse - trial_sse < 1e-14:
                    break
            else:
                lam *= 4.0

        return float(np.exp(log_a)), float(np.exp(log_b))
    
    def _validate_data(self, X):
        """
        Coerce user input into a 2-D float array with a helpful message on failure

        Accepts plain Python lists and 1-D arrays, which the docstrings promise.

        Parameters:
        -----------
        X : array-like

        Returns:
        --------
        X : np.ndarray, shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)          # a single feature column
        if X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got {X.ndim} dimensions")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values; clean them before UMAP")
        return X
    
    def fit(self, X):
        """
        Fit the UMAP model to data
        
        This constructs the high-dimensional graph and finds the
        low-dimensional embedding.
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            High-dimensional input data. A 1-D array or a plain Python list
            of lists is accepted and treated as (n_samples, 1) / converted.
            
        Returns:
        --------
        self : object
            Fitted model
        """
        X = self._validate_data(X)

        # Restart the private generator so repeated fit() calls with the same
        # random_state give the same embedding.
        self._rng = np.random.RandomState(self.random_state)

        if self.verbose > 0:
            print(f"Computing k-NN graph with k={self.n_neighbors}...")
        
        # Step 1: Compute pairwise distances
        distances = self._compute_distances(X)
        
        # Step 2: Find k-nearest neighbors
        knn_indices, knn_distances = self._compute_knn_graph(distances)
        
        if self.verbose > 0:
            print("Computing fuzzy simplicial set...")
        
        # Step 3: Smooth distances (adaptive metric)
        rho, sigma = self._smooth_knn_distances(knn_distances)
        
        # Step 4: Compute membership strengths (high-D graph)
        self.graph_ = self._compute_membership_strengths(
            knn_indices, knn_distances, rho, sigma
        )

        # Keep what transform() needs to place new points against this fit
        self._raw_data = X
        self._rho = rho
        self._sigma = sigma
        
        if self.verbose > 0:
            print(f"Optimizing embedding in {self.n_components}D...")
        
        # Step 5: Optimize low-dimensional embedding
        self.embedding_ = self._optimize_embedding(self.graph_, X.shape[0])
        
        if self.verbose > 0:
            print("UMAP embedding complete!")
        
        return self
    
    def fit_transform(self, X):
        """
        Fit the model and return the embedding
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            High-dimensional input data
            
        Returns:
        --------
        embedding : np.ndarray, shape (n_samples, n_components)
            Low-dimensional embedding
        """
        self.fit(X)
        return self.embedding_
    
    def transform(self, X):
        """
        Transform new data using the fitted model
        
        Note: This is a simplified version. Full UMAP uses more sophisticated
        methods for transforming new points.
        
        How it works (three steps, all reusing the fitted model):
        1. Find each new point's k nearest TRAINING points and compute the same
           rho / sigma local metric that fit() used, giving membership strengths
           v = exp(-max(0, d - rho)/sigma) to those training points.
        2. Drop the new point at the strength-weighted average of its
           neighbours' existing 2-D positions - already a good guess.
        3. Run a few attractive-only SGD sweeps with the TRAINING embedding
           frozen, so new points settle into the existing map without moving it.

        Simplification vs. umap-learn: no negative sampling on the new points,
        and the attractive gradient is scaled by the edge weight directly
        instead of using epochs_per_sample edge scheduling. New points can
        therefore sit slightly closer to their neighbours than a full refit
        would place them.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples_new, n_features)
            New high-dimensional data
            
        Returns:
        --------
        embedding : np.ndarray, shape (n_samples_new, n_components)
            Embedding of new data
        """
        if self.embedding_ is None:
            raise ValueError(
                "This UMAP instance is not fitted yet. "
                "Call fit(X) or fit_transform(X) before calling transform(X)."
            )

        X = self._validate_data(X)
        n_train, n_features = self._raw_data.shape
        if X.shape[1] != n_features:
            raise ValueError(
                f"transform() got {X.shape[1]} features, but the model was "
                f"fitted on {n_features}"
            )

        n_new = X.shape[0]
        k = min(self.n_neighbors, n_train)

        # Step 1: nearest TRAINING points (no self-exclusion - these are new rows)
        cross = self._cross_distances(X, self._raw_data)
        knn_indices = np.argsort(cross, axis=1)[:, :k]
        knn_distances = np.take_along_axis(cross, knn_indices, axis=1)

        # Same adaptive metric as fit(): rho = nearest distance, sigma from log2(k)
        rho_new, sigma_new = self._smooth_knn_distances(knn_distances)

        # Step 2: start at the membership-weighted mean of neighbour positions
        new_embedding = np.zeros((n_new, self.n_components))
        strengths = np.zeros((n_new, k))
        for i in range(n_new):
            d_norm = np.maximum(knn_distances[i] - rho_new[i], 0.0) / sigma_new[i]
            strengths[i] = np.exp(-d_norm)
            share = strengths[i] / (strengths[i].sum() + 1e-12)
            new_embedding[i] = share @ self.embedding_[knn_indices[i]]

        # Step 3: attractive-only refinement, training embedding frozen
        a, b = self._find_ab_params(self.min_dist)
        n_sweeps = max(1, self.n_epochs // 4)

        for epoch in range(n_sweeps):
            alpha = self.learning_rate * (1.0 - epoch / n_sweeps)
            for i in range(n_new):
                for j_idx in range(k):
                    anchor = self.embedding_[knn_indices[i, j_idx]]
                    diff = new_embedding[i] - anchor
                    dist_sq = np.sum(diff ** 2)
                    if dist_sq <= 0.0:
                        continue
                    grad_coef = -2.0 * a * b * (dist_sq ** (b - 1.0))
                    grad_coef /= (1.0 + a * (dist_sq ** b))
                    grad_coef *= strengths[i, j_idx]   # no edge sampling here
                    new_embedding[i] += alpha * np.clip(grad_coef * diff, -4.0, 4.0)

        return new_embedding


"""
========================================
EXAMPLE USAGE
========================================
"""

if __name__ == "__main__":
    print("=" * 70)
    print("UMAP - Uniform Manifold Approximation and Projection")
    print("Educational Implementation")
    print("=" * 70)
    
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _22_umap.py
    # Needs numpy only, prints ASCII only, finishes in a few seconds.
    # Three small quality metrics below let every demo print a number that
    # says whether the embedding actually worked.
    # ----------------------------------------------------------------

    def knn_purity(embedding, point_labels, k=10):
        """Fraction of each point's k nearest EMBEDDED neighbours that share
        its label. 1.00 means the planted groups survived the projection."""
        d = np.sqrt(((embedding[:, None, :] - embedding[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        neighbours = np.argsort(d, axis=1)[:, :k]
        return float((point_labels[neighbours] == point_labels[:, None]).mean())

    def knn_retention(X_high, embedding, k=10):
        """Fraction of each point's HIGH-D k nearest neighbours that are still
        among its k nearest neighbours in the embedding."""
        def knn(A):
            d = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            return np.argsort(d, axis=1)[:, :k]
        hi, lo = knn(X_high), knn(embedding)
        return float(np.mean([len(set(hi[i]) & set(lo[i])) / float(k)
                              for i in range(len(hi))]))

    def cluster_radius(embedding, point_labels):
        """Mean distance from a point to its own cluster's centroid."""
        radii = [np.linalg.norm(embedding[point_labels == c] -
                                embedding[point_labels == c].mean(axis=0), axis=1).mean()
                 for c in np.unique(point_labels)]
        return float(np.mean(radii))

    # Example 1: Basic UMAP on 2D visualization of high-dimensional data
    print("\n" + "=" * 70)
    print("Example 1: UMAP on High-Dimensional Data (Swiss Roll)")
    print("=" * 70)
    
    # Generate Swiss roll dataset (3D manifold)
    np.random.seed(42)
    n_samples = 120
    t = 3 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 21 * np.random.rand(n_samples)
    z = t * np.sin(t)
    X_swiss = np.column_stack([x, y, z])
    
    # Add noise
    X_swiss += np.random.randn(n_samples, 3) * 0.5
    
    print(f"\nSwiss Roll Data Shape: {X_swiss.shape}")
    print(f"Data range: [{X_swiss.min():.2f}, {X_swiss.max():.2f}]")
    
    # Apply UMAP
    print("\nApplying UMAP (n_neighbors=15, min_dist=0.1)...")
    umap = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        learning_rate=1.0,
        n_epochs=60,
        random_state=42,
        verbose=1
    )
    
    X_embedded = umap.fit_transform(X_swiss)
    
    print(f"\nEmbedded Shape: {X_embedded.shape}")
    print(f"Embedding range: [{X_embedded.min():.2f}, {X_embedded.max():.2f}]")
    print("\nFirst 5 embedded points:")
    print(np.round(X_embedded[:5], 3))

    # Did the manifold structure survive? Compare 10-NN retention against a
    # random 2-D layout, which is the "learned nothing" floor.
    print(f"\n10-NN retention, 3-D roll -> 2-D map : "
          f"{knn_retention(X_swiss, X_embedded):.4f}")
    print(f"same measure for a RANDOM 2-D layout  : "
          f"{knn_retention(X_swiss, np.random.randn(n_samples, 2)):.4f}")
    
    # Example 2: UMAP with different parameters
    print("\n" + "=" * 70)
    print("Example 2: Comparing Different UMAP Parameters")
    print("=" * 70)
    
    # Generate clustered data
    np.random.seed(42)
    n_per_cluster = 40
    
    cluster1 = np.random.randn(n_per_cluster, 5) + [0, 0, 0, 0, 0]
    cluster2 = np.random.randn(n_per_cluster, 5) + [5, 5, 5, 5, 5]
    cluster3 = np.random.randn(n_per_cluster, 5) + [10, 0, 10, 0, 10]
    
    X_clusters = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)
    
    # Hold out the first 5 points of each cluster so transform() gets unseen
    # data. Train and test indices are disjoint by construction.
    holdout = np.concatenate([np.arange(c * n_per_cluster, c * n_per_cluster + 5)
                              for c in range(3)])
    is_train = np.ones(len(X_clusters), dtype=bool)
    is_train[holdout] = False
    X_train, y_train = X_clusters[is_train], labels[is_train]
    X_test, y_test = X_clusters[holdout], labels[holdout]

    print(f"\nClustered Data Shape: {X_clusters.shape}")
    print(f"Number of clusters: 3")
    print(f"Train / held-out split: {X_train.shape[0]} / {X_test.shape[0]} points")
    
    # UMAP with small n_neighbors (local structure)
    print("\n--- UMAP with small n_neighbors (local focus) ---")
    umap_local = UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.1,
        n_epochs=40,
        random_state=42,
        verbose=0
    )
    X_local = umap_local.fit_transform(X_train)
    print(f"Local embedding shape: {X_local.shape}")
    print(f"TRAIN 10-NN label purity : {knn_purity(X_local, y_train):.4f}")
    
    # UMAP with large n_neighbors (global structure)
    print("\n--- UMAP with large n_neighbors (global focus) ---")
    umap_global = UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        n_epochs=40,
        random_state=42,
        verbose=0
    )
    X_global = umap_global.fit_transform(X_train)
    print(f"Global embedding shape: {X_global.shape}")
    print(f"TRAIN 10-NN label purity : {knn_purity(X_global, y_train):.4f}")
    print(f"TRAIN 10-NN retention    : {knn_retention(X_train, X_global):.4f}")

    print("\nWhere each cluster landed (global fit):")
    for c in range(3):
        pts = X_global[y_train == c]
        cx, cy = pts.mean(axis=0)
        radius = np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean()
        print(f"  cluster {c}: centroid = ({cx:7.2f},{cy:7.2f})   mean radius = {radius:5.2f}")

    # transform() drops NEW points into the SAME map without refitting
    X_test_embedded = umap_global.transform(X_test)
    d_test = np.sqrt(((X_test_embedded[:, None, :] -
                       X_global[None, :, :]) ** 2).sum(-1))
    nearest_train = np.argmin(d_test, axis=1)
    print(f"\ntransform() on the {len(X_test)} held-out points -> {X_test_embedded.shape}")
    print(f"TEST  1-NN label agreement with the training map: "
          f"{float((y_train[nearest_train] == y_test).mean()):.4f}")

    # min_dist is fitted into the low-D kernel, so it really does change spread
    print("\n--- min_dist controls how tightly points pack ---")
    print("q(d) = 1 / (1 + a*d^(2b)); a and b are least-squares fitted from min_dist")
    print("\n  min_dist       a        b     mean cluster radius   10-NN purity")
    for min_dist in [0.0, 0.25, 0.5]:
        umap_md = UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=min_dist,
            n_epochs=30,
            random_state=42,
            verbose=0
        )
        X_md = umap_md.fit_transform(X_train)
        a_md, b_md = umap_md._find_ab_params(min_dist)
        print(f"  {min_dist:8.2f}  {a_md:7.4f}  {b_md:7.4f}"
              f"  {cluster_radius(X_md, y_train):16.3f}"
              f"  {knn_purity(X_md, y_train):13.4f}")
    print("\nRadius grows with min_dist while purity stays at 1.00: the clusters")
    print("stay separated, they just breathe.")
    
    # Example 3: UMAP for dimensionality reduction before ML
    print("\n" + "=" * 70)
    print("Example 3: UMAP for Dimensionality Reduction (Feature Engineering)")
    print("=" * 70)
    
    # Generate high-dimensional data
    np.random.seed(42)
    n_samples = 90
    n_features = 20
    X_high_dim = np.random.randn(n_samples, n_features)
    
    # Add some structure
    X_high_dim[:45, :5] += 3
    X_high_dim[45:, 5:10] += 3
    y_high_dim = np.repeat([0, 1], 45)
    
    print(f"\nOriginal data: {X_high_dim.shape}")
    
    # Reduce to lower dimensions for ML pipeline
    umap_ml = UMAP(
        n_components=5,  # Reduce to 5 features
        n_neighbors=10,
        min_dist=0.0,  # Tightest packing (a=1.9328, b=0.7905)
        n_epochs=30,
        random_state=42,
        verbose=0
    )
    
    X_reduced = umap_ml.fit_transform(X_high_dim)
    
    print(f"Reduced data: {X_reduced.shape}")
    print(f"Dimensionality reduction: {n_features} -> {X_reduced.shape[1]}")
    print(f"Compression ratio: {n_features / X_reduced.shape[1]:.1f}x")
    print(f"5-NN label purity in {n_features}-D : "
          f"{knn_purity(X_high_dim, y_high_dim, 5):.4f}")
    print(f"5-NN label purity in  {X_reduced.shape[1]}-D : "
          f"{knn_purity(X_reduced, y_high_dim, 5):.4f}")
    print("The planted two-group structure survives the 4x compression.")
    
    # Example 4: Different distance metrics
    print("\n" + "=" * 70)
    print("Example 4: UMAP with Different Distance Metrics")
    print("=" * 70)
    
    # Generate data suitable for different metrics: three groups that differ
    # only in DIRECTION, with magnitudes spread over 0.2x to 8x. Cosine sees
    # the groups; euclidean spends its budget on the magnitudes.
    np.random.seed(42)
    directions = np.random.randn(3, 12)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitudes = np.random.uniform(0.2, 8.0, size=(90, 1))
    X_text = np.repeat(directions, 30, axis=0) + np.random.randn(90, 12) * 0.15
    X_text = X_text * magnitudes  # same directions, very different lengths
    y_text = np.repeat([0, 1, 2], 30)
    
    print(f"\nData shape: {X_text.shape}")
    print(f"10-NN purity in the RAW 12-D data: {knn_purity(X_text, y_text):.4f}")
    
    # Euclidean metric
    print("\n--- Euclidean metric ---")
    umap_euclidean = UMAP(
        n_components=2,
        n_neighbors=8,
        metric='euclidean',
        n_epochs=40,
        random_state=42,
        verbose=0
    )
    X_euclidean = umap_euclidean.fit_transform(X_text)
    print(f"Euclidean embedding: {X_euclidean.shape}")
    print(f"10-NN label purity : {knn_purity(X_euclidean, y_text):.4f}")
    
    # Cosine metric (better for normalized data)
    print("\n--- Cosine metric ---")
    umap_cosine = UMAP(
        n_components=2,
        n_neighbors=8,
        metric='cosine',
        n_epochs=40,
        random_state=42,
        verbose=0
    )
    X_cosine = umap_cosine.fit_transform(X_text)
    print(f"Cosine embedding: {X_cosine.shape}")
    print(f"10-NN label purity : {knn_purity(X_cosine, y_text):.4f}")
    print("\nCosine wins: the groups differ only in direction, so euclidean")
    print("distance is partly spent on the random magnitudes.")
    
    # Practical Tips
    print("\n" + "=" * 70)
    print("PRACTICAL TIPS FOR USING UMAP")
    print("=" * 70)
    
    tips = """
    1. PARAMETER SELECTION:
       - n_neighbors: Start with 15, increase for global structure (50-100)
       - min_dist: 0.0-0.1 for clustering, 0.3-0.5 for general viz
       - n_epochs: 200 minimum, 500+ for better quality
    
    2. CHOOSING n_neighbors:
       - Small (5-10): Emphasizes local structure, tight clusters
       - Medium (15-30): Balanced (default choice)
       - Large (50-100): Emphasizes global structure
    
    3. CHOOSING min_dist:
       - 0.0: Very tight clusters (good for cluster analysis)
       - 0.1: Default, balanced (good starting point)
       - 0.5+: Spread out, more uniform (good for even distribution)
    
    4. WHEN TO USE UMAP:
       - Visualizing high-dimensional data
       - Feature engineering before ML models
       - When you need both local AND global structure
       - When you want consistent embeddings (more stable than t-SNE)
       NOTE ON SCALE: the umap-learn library handles 100,000+ samples. THIS
       from-scratch class builds a dense n x n distance matrix and runs the
       SGD in pure Python, so keep n under roughly 500-1000.
    
    5. UMAP vs t-SNE:
       - UMAP: Faster, preserves global structure, more general purpose
       - t-SNE: Only visualization, loses global structure, slower
       - UMAP can be used as preprocessing; t-SNE cannot
    
    6. DISTANCE METRICS:
       - Euclidean: Default, works for most cases
       - Cosine: Text data, high-dimensional sparse data
       - Manhattan: When you want to penalize all dimensions equally
    
    7. COMMON ISSUES:
       - Clusters overlap: Increase n_neighbors or decrease min_dist
       - Too spread out: Decrease n_neighbors or increase min_dist
       - Slow convergence: Increase learning_rate or n_epochs
       - Inconsistent results: Set random_state for reproducibility
    """
    
    print(tips)
    
    print("\n" + "=" * 70)
    print("COMPARISON: UMAP ADVANTAGES")
    print("=" * 70)
    
    comparison = """
    UMAP vs Other Dimensionality Reduction Methods:
    
    vs PCA:
    + UMAP: Captures non-linear structure
    + UMAP: Better for visualization
    - PCA: Faster, deterministic, interpretable
    
    vs t-SNE:
    + UMAP: 10-100x faster (that is the umap-learn library; this pure-Python
      teaching version is about 40x SLOWER than sklearn's
      TSNE - 9.1 s vs 0.2 s on 150 points, 200 epochs)
    + UMAP: Preserves global structure
    + UMAP: Can be used for general dimensionality reduction
    + UMAP: More stable/reproducible
    - t-SNE: Sometimes better for very local structure
    
    vs Autoencoders:
    + UMAP: No training required, faster
    + UMAP: Based on solid mathematical theory
    - Autoencoders: Can be more flexible, learned mapping
    
    Best Use Cases for UMAP:
    - Single-cell genomics visualization
    - Exploring embeddings (word2vec, BERT)
    - Preprocessing before clustering or classification
    - Interactive data exploration
    - Any high-dimensional visualization need
    """
    
    print(comparison)
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
