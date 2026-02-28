import numpy as np
from collections import defaultdict

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
            
        min_dist : float, default=0.1
            Minimum distance between points in embedding
            - Range: 0.0-1.0
            - Small values (0.0-0.1): Tightly packed clusters
            - Large values (0.3-0.99): More evenly distributed points
            - Controls how tightly points cluster together
            
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
            Random seed for reproducibility
            
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
        
        if random_state is not None:
            np.random.seed(random_state)
    
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
            distances = 1 - similarities
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return distances
    
    def _compute_knn_graph(self, distances):
        """
        Compute k-nearest neighbors graph
        
        Parameters:
        -----------
        distances : np.ndarray, shape (n_samples, n_samples)
            Pairwise distance matrix
            
        Returns:
        --------
        knn_indices : np.ndarray, shape (n_samples, n_neighbors)
            Indices of k-nearest neighbors for each point
        knn_distances : np.ndarray, shape (n_samples, n_neighbors)
            Distances to k-nearest neighbors for each point
        """
        n_samples = distances.shape[0]
        knn_indices = np.zeros((n_samples, self.n_neighbors), dtype=int)
        knn_distances = np.zeros((n_samples, self.n_neighbors))
        
        for i in range(n_samples):
            # Get indices sorted by distance (excluding self)
            sorted_indices = np.argsort(distances[i])
            # Exclude the point itself (distance 0)
            neighbors = sorted_indices[1:self.n_neighbors + 1]
            knn_indices[i] = neighbors
            knn_distances[i] = distances[i, neighbors]
        
        return knn_indices, knn_distances
    
    def _smooth_knn_distances(self, knn_distances, n_iter=64, bandwidth=1.0):
        """
        Compute smoothed distances using local metric (rho and sigma)
        
        This implements the adaptive distance metric from the UMAP paper.
        For each point, we find:
        - rho: distance to nearest neighbor (local connectivity)
        - sigma: normalization factor so sum of probabilities ≈ log2(k)
        
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
        n_samples = knn_distances.shape[0]
        rho = np.zeros(n_samples)
        sigma = np.ones(n_samples)
        
        target = np.log2(self.n_neighbors)
        
        for i in range(n_samples):
            # rho is distance to nearest neighbor (local connectivity)
            rho[i] = knn_distances[i, 0]
            
            # Binary search for sigma
            lo, hi = 0.0, 1e10
            for _ in range(n_iter):
                mid = (lo + hi) / 2.0
                
                # Compute sum of probabilities
                diffs = knn_distances[i] - rho[i]
                diffs = np.maximum(diffs, 0)
                probs = np.exp(-diffs / mid)
                sum_probs = np.sum(probs)
                
                if sum_probs > target:
                    hi = mid
                else:
                    lo = mid
            
            sigma[i] = (lo + hi) / 2.0
        
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
        graph = defaultdict(float)
        n_samples = knn_indices.shape[0]
        
        for i in range(n_samples):
            for j_idx, j in enumerate(knn_indices[i]):
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
                
                # Fuzzy set union: A ∪ B = A + B - A*B
                prob = val_ij + val_ji - val_ij * val_ji
                
                if prob > 0:
                    graph[(i, j)] = prob
                    graph[(j, i)] = prob
        
        return graph
    
    def _spectral_initialization(self, graph, n_samples):
        """
        Initialize embedding using spectral embedding (graph Laplacian)
        
        This provides a good initial layout based on the graph structure.
        Similar to using eigenvectors of the graph Laplacian.
        
        Parameters:
        -----------
        graph : dict
            Sparse graph representation
        n_samples : int
            Number of samples
            
        Returns:
        --------
        embedding : np.ndarray, shape (n_samples, n_components)
            Initial embedding
        """
        # Build adjacency matrix (sparse representation)
        # For simplicity, we'll use a simplified version
        # In production, use scipy.sparse for efficiency
        
        # Simple approach: use random initialization with structure
        embedding = np.random.randn(n_samples, self.n_components) * 0.0001
        
        # Add some structure based on average neighbor positions
        # This is a simplified spectral-like initialization
        for epoch in range(10):
            new_embedding = embedding.copy()
            for i in range(n_samples):
                neighbors = []
                weights = []
                for (a, b), w in graph.items():
                    if a == i:
                        neighbors.append(b)
                        weights.append(w)
                
                if neighbors:
                    weights = np.array(weights)
                    weights /= (weights.sum() + 1e-10)
                    neighbor_positions = embedding[neighbors]
                    new_embedding[i] = np.average(neighbor_positions, weights=weights, axis=0)
            
            embedding = 0.5 * embedding + 0.5 * new_embedding
        
        return embedding
    
    def _optimize_embedding(self, graph, n_samples):
        """
        Optimize the low-dimensional embedding using stochastic gradient descent
        
        This is where the actual dimensionality reduction happens.
        We minimize the cross-entropy between high-D and low-D graphs.
        
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
            embedding = np.random.randn(n_samples, self.n_components) * 10.0
        
        # Get all edges
        edges = list(graph.keys())
        weights = np.array([graph[e] for e in edges])
        
        # Parameters for optimization
        a, b = self._find_ab_params(self.min_dist)
        
        # Optimization loop
        for epoch in range(self.n_epochs):
            # Learning rate schedule
            alpha = self.learning_rate * (1.0 - epoch / self.n_epochs)
            
            # Shuffle edges
            indices = np.random.permutation(len(edges))
            
            for idx in indices:
                i, j = edges[idx]
                weight = weights[idx]
                
                # Get current positions
                current_i = embedding[i]
                current_j = embedding[j]
                
                # Compute distance in low-D
                diff = current_i - current_j
                dist_sq = np.sum(diff**2)
                dist = np.sqrt(dist_sq) + 1e-10
                
                # Attractive force (for connected pairs)
                # High-D edge exists: pull together
                grad_coef = -2.0 * a * b * (dist_sq ** (b - 1.0))
                grad_coef /= (1.0 + a * (dist_sq ** b))
                grad_coef *= weight
                
                grad = grad_coef * diff
                
                # Apply gradient
                embedding[i] += alpha * grad
                embedding[j] -= alpha * grad
                
                # Negative sampling: repulsive force (for random pairs)
                # This prevents all points from collapsing together
                for _ in range(5):  # Negative samples
                    k = np.random.randint(n_samples)
                    if k == i or k == j:
                        continue
                    
                    current_k = embedding[k]
                    diff_ik = current_i - current_k
                    dist_sq_ik = np.sum(diff_ik**2)
                    
                    # Repulsive force: push apart
                    grad_coef = 2.0 * b
                    grad_coef /= ((0.001 + dist_sq_ik) * (1.0 + a * (dist_sq_ik ** b)))
                    
                    grad_ik = grad_coef * diff_ik
                    
                    embedding[i] += alpha * grad_ik * 0.5
                    embedding[k] -= alpha * grad_ik * 0.5
            
            if self.verbose > 0 and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(f"Epoch {epoch + 1}/{self.n_epochs}")
        
        return embedding
    
    def _find_ab_params(self, min_dist, spread=1.0):
        """
        Find parameters a and b for the low-dimensional probability function
        
        These parameters control the shape of the embedding:
        P(d) = 1 / (1 + a * d^(2b))
        
        Parameters:
        -----------
        min_dist : float
            Minimum distance parameter
        spread : float
            Spread of points
            
        Returns:
        --------
        a, b : float
            Parameters for probability function
        """
        def curve(x, a, b):
            return 1.0 / (1.0 + a * (x ** (2 * b)))
        
        # Simplified parameter finding
        # In full UMAP, this uses curve fitting
        a = 1.0
        b = 1.0
        
        if min_dist > 0:
            # Adjust parameters based on min_dist
            # These are reasonable defaults
            a = 1.929 / spread
            b = 0.7915
        
        return a, b
    
    def fit(self, X):
        """
        Fit the UMAP model to data
        
        This constructs the high-dimensional graph and finds the
        low-dimensional embedding.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            High-dimensional input data
            
        Returns:
        --------
        self : object
            Fitted model
        """
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
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples_new, n_features)
            New high-dimensional data
            
        Returns:
        --------
        embedding : np.ndarray, shape (n_samples_new, n_components)
            Embedding of new data
        """
        raise NotImplementedError(
            "Transform method for new data requires training data storage and "
            "is not implemented in this educational version. "
            "Use fit_transform on the complete dataset."
        )


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
    
    # Example 1: Basic UMAP on 2D visualization of high-dimensional data
    print("\n" + "=" * 70)
    print("Example 1: UMAP on High-Dimensional Data (Swiss Roll)")
    print("=" * 70)
    
    # Generate Swiss roll dataset (3D manifold)
    np.random.seed(42)
    n_samples = 300
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
        n_epochs=200,
        random_state=42,
        verbose=1
    )
    
    X_embedded = umap.fit_transform(X_swiss)
    
    print(f"\nEmbedded Shape: {X_embedded.shape}")
    print(f"Embedding range: [{X_embedded.min():.2f}, {X_embedded.max():.2f}]")
    print("\nFirst 5 embedded points:")
    print(X_embedded[:5])
    
    # Example 2: UMAP with different parameters
    print("\n" + "=" * 70)
    print("Example 2: Comparing Different UMAP Parameters")
    print("=" * 70)
    
    # Generate clustered data
    np.random.seed(42)
    n_per_cluster = 100
    
    cluster1 = np.random.randn(n_per_cluster, 5) + [0, 0, 0, 0, 0]
    cluster2 = np.random.randn(n_per_cluster, 5) + [5, 5, 5, 5, 5]
    cluster3 = np.random.randn(n_per_cluster, 5) + [10, 0, 10, 0, 10]
    
    X_clusters = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)
    
    print(f"\nClustered Data Shape: {X_clusters.shape}")
    print(f"Number of clusters: 3")
    
    # UMAP with small n_neighbors (local structure)
    print("\n--- UMAP with small n_neighbors (local focus) ---")
    umap_local = UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.1,
        n_epochs=150,
        random_state=42,
        verbose=0
    )
    X_local = umap_local.fit_transform(X_clusters)
    print(f"Local embedding shape: {X_local.shape}")
    
    # UMAP with large n_neighbors (global structure)
    print("\n--- UMAP with large n_neighbors (global focus) ---")
    umap_global = UMAP(
        n_components=2,
        n_neighbors=50,
        min_dist=0.1,
        n_epochs=150,
        random_state=42,
        verbose=0
    )
    X_global = umap_global.fit_transform(X_clusters)
    print(f"Global embedding shape: {X_global.shape}")
    
    # Example 3: UMAP for dimensionality reduction before ML
    print("\n" + "=" * 70)
    print("Example 3: UMAP for Dimensionality Reduction (Feature Engineering)")
    print("=" * 70)
    
    # Generate high-dimensional data
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    X_high_dim = np.random.randn(n_samples, n_features)
    
    # Add some structure
    X_high_dim[:100, :10] += 3
    X_high_dim[100:, 10:20] += 3
    
    print(f"\nOriginal data: {X_high_dim.shape}")
    
    # Reduce to lower dimensions for ML pipeline
    umap_ml = UMAP(
        n_components=10,  # Reduce to 10 features
        n_neighbors=15,
        min_dist=0.0,  # Preserve more local structure
        n_epochs=150,
        random_state=42,
        verbose=0
    )
    
    X_reduced = umap_ml.fit_transform(X_high_dim)
    
    print(f"Reduced data: {X_reduced.shape}")
    print(f"Dimensionality reduction: {n_features} -> {X_reduced.shape[1]}")
    print(f"Compression ratio: {n_features / X_reduced.shape[1]:.1f}x")
    
    # Example 4: Different distance metrics
    print("\n" + "=" * 70)
    print("Example 4: UMAP with Different Distance Metrics")
    print("=" * 70)
    
    # Generate data suitable for different metrics
    np.random.seed(42)
    X_text = np.random.rand(150, 20)  # Simulate text embeddings
    X_text /= np.linalg.norm(X_text, axis=1, keepdims=True)  # Normalize
    
    print(f"\nData shape: {X_text.shape}")
    
    # Euclidean metric
    print("\n--- Euclidean metric ---")
    umap_euclidean = UMAP(
        n_components=2,
        n_neighbors=10,
        metric='euclidean',
        n_epochs=100,
        random_state=42,
        verbose=0
    )
    X_euclidean = umap_euclidean.fit_transform(X_text)
    print(f"Euclidean embedding: {X_euclidean.shape}")
    
    # Cosine metric (better for normalized data)
    print("\n--- Cosine metric ---")
    umap_cosine = UMAP(
        n_components=2,
        n_neighbors=10,
        metric='cosine',
        n_epochs=100,
        random_state=42,
        verbose=0
    )
    X_cosine = umap_cosine.fit_transform(X_text)
    print(f"Cosine embedding: {X_cosine.shape}")
    
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
       ✓ Visualizing high-dimensional data
       ✓ Feature engineering before ML models
       ✓ When you need both local AND global structure
       ✓ When you have >100,000 samples (faster than t-SNE)
       ✓ When you want consistent embeddings (more stable than t-SNE)
    
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
    + UMAP: 10-100x faster
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
