import numpy as np

class HierarchicalClustering:
    """
    Hierarchical Clustering Implementation from Scratch
    
    Hierarchical clustering builds a hierarchy of clusters by iteratively merging
    or splitting clusters based on their similarity. This implementation uses the
    agglomerative (bottom-up) approach.
    
    Key Idea: "Build a tree of clusters from individual points"
    
    The algorithm creates a dendrogram (tree structure) showing how data points
    are grouped at different levels of similarity. You can then "cut" this tree
    at any height to get the desired number of clusters.
    
    Advantages over k-Means:
    - No need to specify number of clusters in advance
    - Produces a dendrogram showing hierarchical relationships
    - Deterministic (no random initialization)
    - Can find clusters of different shapes
    
    Use Cases:
    - Taxonomy / phylogenetics: build evolutionary trees from genetic distances
    - Document organisation: nest topics inside categories inside sections
    - Customer segmentation: discover nested market segments without fixing k
    - Gene expression analysis: group co-expressed genes into functional families
    - Image segmentation: merge neighbouring regions into a region hierarchy

    Linkage formulas (how the distance between two CLUSTERS is defined):
        single(C1, C2)   = min  d(x, y)   for x in C1, y in C2
        complete(C1, C2) = max  d(x, y)   for x in C1, y in C2
        average(C1, C2)  = (1 / (|C1| * |C2|)) * sum_{x in C1} sum_{y in C2} d(x, y)
        ward(C1, C2)     = sqrt( (2 * |C1| * |C2|) / (|C1| + |C2|) ) * ||mu1 - mu2||

        where mu1, mu2 are the cluster centroids. Ward's value is the square root of
        twice the increase in total within-cluster sum of squares caused by the merge,
        so it is only meaningful for the Euclidean metric (enforced in fit).

    Build once, cut many times:
        fit(X) always merges all the way down to a single cluster and records every
        merge in linkage_matrix_ (SciPy-compatible). set_n_clusters(k) then re-cuts
        that stored hierarchy for a new k without re-running the clustering.

    Simplifications vs. canonical hierarchical clustering:
        Only the agglomerative (bottom-up) half is implemented; the divisive
        (top-down) variant is explained in the .md but not coded. Cuts are by
        number of clusters only - there is no distance_threshold cut, and the
        merge search is the plain O(n^2)-per-step scan rather than SciPy's
        nearest-neighbour chain. See "Simplifications vs. Canonical Hierarchical
        Clustering" in _12_hierarchical_clustering.md for the full list.
    """
    
    def __init__(self, n_clusters=2, linkage='average', distance_metric='euclidean'):
        """
        Initialize the Hierarchical Clustering model
        
        Parameters:
        -----------
        n_clusters : int, default=2
            The number of clusters to find
            - Range: 1 to n_samples (validated in fit)
            - Larger k: more, smaller, more homogeneous clusters
            - Smaller k: fewer, broader clusters
            Unlike k-Means, this can be decided AFTER seeing the dendrogram:
            call set_n_clusters(k) to re-cut an already fitted hierarchy.
            Typical: 2-10; pick k by looking for the largest gap between
            consecutive merge heights in linkage_matrix_[:, 2].
        
        linkage : str, default='average'
            Method for calculating distance between clusters
            Options: 'single', 'complete', 'average', 'ward'
            
            - 'single': Minimum distance between any two points
                       Good for: Elongated clusters
                       Bad for: Sensitive to noise (chaining effect)
            
            - 'complete': Maximum distance between any two points
                         Good for: Compact, spherical clusters
                         Bad for: Sensitive to outliers
            
            - 'average': Average distance between all point pairs (UPGMA)
                        Good for: Balanced approach, robust
                        Bad for: Computationally expensive
            
            - 'ward': Minimize within-cluster variance
                     Good for: Compact, balanced clusters
                     Bad for: Assumes spherical clusters, similar sizes
                     NOTE: Ward's criterion is derived from Euclidean sums of
                     squares, so it requires distance_metric='euclidean'.
                     fit() raises ValueError for any other metric (sklearn
                     rejects the same combination).
        
        distance_metric : str, default='euclidean'
            Distance metric to measure similarity between points
            Options: 'euclidean', 'manhattan', 'cosine'
            
            - 'euclidean': Straight-line distance (most common)
            - 'manhattan': City-block distance (sum of absolute differences)
            - 'cosine': 1 - cosine similarity; ignores vector length and
                        compares direction only (common for text/TF-IDF)
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric
        
        # Model attributes (set after fitting)
        self.labels_ = None
        self.linkage_matrix_ = None
        self.n_samples_ = None
        self._final_clusters = None   # list of index lists at the current cut
        self._X_fit = None            # training data, needed by predict()
        self._distance_matrix_ = None # cached n x n point-to-point distances
        
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points
        
        Parameters:
        -----------
        x1 : numpy array
            First point
        x2 : numpy array
            Second point
            
        Returns:
        --------
        distance : float
            Distance between the two points
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance: sqrt(sum((x1 - x2)^2))
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            # Manhattan distance: sum(|x1 - x2|)
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            # Cosine distance: 1 - (x1 . x2) / (||x1|| * ||x2||)
            # Measures the ANGLE between the vectors, not their length.
            norm_product = np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2))
            if norm_product == 0:
                # A zero vector has no direction; treat it as maximally distant
                return 0.0 if np.allclose(x1, x2) else 1.0
            return 1.0 - np.sum(x1 * x2) / norm_product
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _compute_distance_matrix(self, X):
        """
        Precompute every point-to-point distance ONCE

        The naive algorithm recomputes d(x, y) inside every merge step, which
        costs O(n^3) *distance evaluations*. Building this n x n table up front
        costs O(n^2) memory (the O(n^2) space the theory section talks about)
        and turns every later distance request into a table lookup. The
        algorithm itself is unchanged - only the bookkeeping is.

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data whose pairwise distances are needed

        Returns:
        --------
        D : numpy array of shape (n_samples, n_samples)
            D[i, j] = distance between point i and point j (symmetric, zero diagonal)
        """
        n_samples = X.shape[0]
        D = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self._calculate_distance(X[i], X[j])
                D[i, j] = distance
                D[j, i] = distance   # distance is symmetric

        return D


    def _calculate_cluster_distance(self, X, cluster1_indices, cluster2_indices):
        """
        Calculate distance between two clusters using specified linkage method
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Original data
        cluster1_indices : list
            Indices of points in first cluster
        cluster2_indices : list
            Indices of points in second cluster
            
        Returns:
        --------
        distance : float
            Distance between the two clusters
        """
        # Get the actual points in each cluster
        cluster1_points = X[cluster1_indices]
        cluster2_points = X[cluster2_indices]

        # Every cross-cluster point distance below is READ from the table built
        # once by _compute_distance_matrix, never recomputed. distances[i][j] is
        # d(point i, point j), so the loops stay exactly as literal as before.
        distances = self._distance_matrix_
        if distances is None or distances.shape[0] != X.shape[0]:
            # Called outside fit() (or on different data): build the table now
            distances = self._compute_distance_matrix(X)

        if self.linkage == 'single':
            # Single linkage: minimum distance between any two points
            min_distance = float('inf')
            for i1 in cluster1_indices:
                for i2 in cluster2_indices:
                    distance = distances[i1, i2]
                    min_distance = min(min_distance, distance)
            return min_distance

        elif self.linkage == 'complete':
            # Complete linkage: maximum distance between any two points
            max_distance = 0
            for i1 in cluster1_indices:
                for i2 in cluster2_indices:
                    distance = distances[i1, i2]
                    max_distance = max(max_distance, distance)
            return max_distance

        elif self.linkage == 'average':
            # Average linkage: average distance between all point pairs
            total_distance = 0
            count = 0
            for i1 in cluster1_indices:
                for i2 in cluster2_indices:
                    distance = distances[i1, i2]
                    total_distance += distance
                    count += 1
            return total_distance / count if count > 0 else 0

        elif self.linkage == 'ward':
            # Ward linkage: minimize within-cluster variance.
            # Merging C1 and C2 raises the total within-cluster sum of squares by
            #     dESS = (|C1| * |C2| / (|C1| + |C2|)) * ||mu1 - mu2||^2
            # Ward's reported merge height is sqrt(2 * dESS), which is what SciPy
            # and sklearn store in the linkage matrix:
            #     d(C1, C2) = sqrt(2 * |C1| * |C2| / (|C1| + |C2|)) * ||mu1 - mu2||
            # It needs CENTROIDS, so it works from the raw points, not the table.
            centroid1 = np.mean(cluster1_points, axis=0)
            centroid2 = np.mean(cluster2_points, axis=0)
            
            n1 = len(cluster1_indices)
            n2 = len(cluster2_indices)
            
            # Ward's formula
            distance = np.sqrt((2.0 * n1 * n2) / (n1 + n2)) * \
                      self._calculate_distance(centroid1, centroid2)
            return distance

        else:
            raise ValueError(f"Unknown linkage method: {self.linkage}")
    
    def fit(self, X):
        """
        Perform hierarchical clustering on the data
        
        This builds the complete dendrogram (hierarchy) by iteratively merging
        the two closest clusters until only ONE cluster remains, recording every
        merge in linkage_matrix_. The requested n_clusters is then obtained by
        cutting that finished hierarchy (see _cut_dendrogram), which is why
        set_n_clusters(k) can change k later without re-clustering.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data to cluster. A 1-D array or a plain Python list is
            accepted and is treated as a single feature column.
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            # One feature handed over as a flat array/list -> single column
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(
                "X must be a non-empty 2-D array of shape (n_samples, n_features), "
                f"got array with shape {X.shape}"
            )

        self.n_samples_ = X.shape[0]
        
        # --- Validate the configuration before doing any work ---
        if not (1 <= self.n_clusters <= self.n_samples_):
            raise ValueError(
                f"n_clusters must be between 1 and n_samples ({self.n_samples_}), "
                f"got {self.n_clusters}"
            )
        if self.linkage == 'ward' and self.distance_metric != 'euclidean':
            # Ward's criterion is a statement about Euclidean sums of squares,
            # so any other metric silently computes something that is not Ward.
            raise ValueError(
                "linkage='ward' requires distance_metric='euclidean' (got "
                f"'{self.distance_metric}')"
            )

        # Keep the training data: predict() rebuilds cluster centroids from it
        self._X_fit = X.copy()

        # Precompute every point-to-point distance ONCE (the O(n^2) memory the
        # theory section talks about). Every linkage below reads this table.
        self._distance_matrix_ = self._compute_distance_matrix(X)

        # Initialize: each point is its own cluster
        # Store as list of lists containing point indices
        clusters = [[i] for i in range(self.n_samples_)]
        
        # SciPy linkage-matrix convention for cluster IDs: the n original points
        # keep IDs 0..n-1, and the cluster formed by merge number m gets the new
        # ID n + m. cluster_ids[p] is the ID of the cluster sitting at position p.
        cluster_ids = list(range(self.n_samples_))
        next_cluster_id = self.n_samples_

        # Store merge history for dendrogram
        # Format: [cluster_i, cluster_j, distance, size]
        self.linkage_matrix_ = []
        
        # Cluster-pair distances, keyed by the pair of cluster IDs. After a merge
        # only the pairs involving the NEW cluster are unknown - every other pair
        # is untouched, so its distance is reused instead of being recomputed.
        # This is pure bookkeeping: the algorithm below is unchanged.
        distance_cache = {}
        
        # Merge until a single cluster remains -> the COMPLETE dendrogram
        while len(clusters) > 1:
            # Find the two closest clusters
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            # Check all pairs of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    pair_key = (cluster_ids[i], cluster_ids[j])
                    if pair_key not in distance_cache:
                        distance_cache[pair_key] = self._calculate_cluster_distance(
                            X, clusters[i], clusters[j]
                        )
                    distance = distance_cache[pair_key]
                    
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Store merge information for dendrogram.
            # SciPy wants the two child IDs in ascending order, then the height
            # at which they merged, then the size of the cluster they form.
            cluster_i_id = cluster_ids[merge_i]
            cluster_j_id = cluster_ids[merge_j]
            
            self.linkage_matrix_.append([
                min(cluster_i_id, cluster_j_id),
                max(cluster_i_id, cluster_j_id),
                min_distance,
                len(clusters[merge_i]) + len(clusters[merge_j])
            ])
            
            # Merge the two closest clusters IN PLACE at the lower position.
            # merge_i < merge_j always holds because j starts at i + 1, so the
            # merged cluster keeps the position of its left-most member. That
            # is what makes the final cluster numbering follow the data order.
            clusters[merge_i] = clusters[merge_i] + clusters[merge_j]
            cluster_ids[merge_i] = next_cluster_id
            clusters.pop(merge_j)
            cluster_ids.pop(merge_j)
            next_cluster_id += 1
            
        # shape (n_samples - 1, 4); reshape keeps the shape right for n_samples=1
        self.linkage_matrix_ = np.array(self.linkage_matrix_).reshape(-1, 4)
        
        # Cut the finished hierarchy down to n_clusters and label every point
        self._final_clusters = self._cut_dendrogram(self.n_clusters)
        self.labels_ = self._labels_from_clusters(self._final_clusters)
        
        return self
    
    def _cut_dendrogram(self, n_clusters):
        """
        Cut the dendrogram to get the desired number of clusters
        
        This extracts a partition from the ALREADY-BUILT hierarchy: it replays
        the first (n_samples - n_clusters) rows of linkage_matrix_ and stops.
        Nothing is re-clustered, so cutting is O(n^2) bookkeeping rather than a
        second full fit - this is the "build once, cut many times" property.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters wanted, between 1 and n_samples

        Returns:
        --------
        clusters : list of lists
            Each inner list holds the row indices belonging to one cluster
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Model must be fitted before cutting the dendrogram")

        # Start again from singletons, then replay the recorded merges
        clusters = [[i] for i in range(self.n_samples_)]
        cluster_ids = list(range(self.n_samples_))
        next_cluster_id = self.n_samples_
        
        n_merges = self.n_samples_ - n_clusters
        
        for row in self.linkage_matrix_[:n_merges]:
            id_a, id_b = int(row[0]), int(row[1])
            
            # Where do those two cluster IDs currently sit?
            pos_a = cluster_ids.index(id_a)
            pos_b = cluster_ids.index(id_b)
            low, high = min(pos_a, pos_b), max(pos_a, pos_b)
            
            # Same merge bookkeeping as fit(), so the cut reproduces exactly the
            # cluster ordering fit() would have had at this point in the run
            clusters[low] = clusters[low] + clusters[high]
            cluster_ids[low] = next_cluster_id
            clusters.pop(high)
            cluster_ids.pop(high)
            next_cluster_id += 1
        
        return clusters
        
    def _labels_from_clusters(self, clusters):
        """
        Turn a list of index lists into a flat label array
    
        Cluster number c gets label c, so labels_[i] says which cluster row i
        of the training data ended up in.

        Parameters:
        -----------
        clusters : list of lists
            Each inner list holds the row indices belonging to one cluster

        Returns:
        --------
        labels : numpy array of shape (n_samples,)
        """
        labels = np.zeros(self.n_samples_, dtype=int)
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                labels[idx] = cluster_id
        return labels

    def set_n_clusters(self, n_clusters):
        """
        Re-cut an already fitted dendrogram at a different number of clusters

        This is the headline advantage over k-Means: the hierarchy is built once
        by fit(), and any k can be read off it afterwards without re-clustering.

        Parameters:
        -----------
        n_clusters : int
            New number of clusters, between 1 and n_samples

        Returns:
        --------
        self : object
            Returns self for method chaining; labels_ is updated in place
        """
        if self.linkage_matrix_ is None:
            raise ValueError(
                "Model is not fitted yet. Call fit(X) before set_n_clusters()."
            )
        if not (1 <= n_clusters <= self.n_samples_):
            raise ValueError(
                f"n_clusters must be between 1 and n_samples ({self.n_samples_}), "
                f"got {n_clusters}"
            )

        self.n_clusters = n_clusters
        self._final_clusters = self._cut_dendrogram(n_clusters)
        self.labels_ = self._labels_from_clusters(self._final_clusters)

        return self

    def predict(self, X):
        """
        Predict cluster labels for samples
        
        Hierarchical clustering has no natural rule for unseen points (sklearn's
        AgglomerativeClustering deliberately offers no predict at all), so this
        uses the standard practical extension: each cluster is summarised by the
        centroid of ITS TRAINING POINTS, and a new point joins the cluster whose
        centroid is nearest under distance_metric.
        
        Passing the exact training matrix back returns the fitted labels_.

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to assign to clusters
            
        Returns:
        --------
        labels : numpy array of shape (n_samples,)
            Cluster assignment for each sample
        """
        if self.labels_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X) before predict(X).")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Exactly the training data -> the fitted labels are the answer.
        # (A row-count match is NOT enough: different data of the same size
        # must still go through nearest-centroid assignment.)
        if X.shape == self._X_fit.shape and np.array_equal(X, self._X_fit):
            return self.labels_
        
        if X.shape[1] != self._X_fit.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted on "
                f"{self._X_fit.shape[1]}"
            )

        # For new data, assign to nearest cluster center.
        # Centers come from the TRAINING data (self._X_fit), because
        # _final_clusters holds indices into the training rows.
        cluster_centers = []
        for cluster_indices in self._final_clusters:
            cluster_center = np.mean(self._X_fit[cluster_indices], axis=0)
            cluster_centers.append(cluster_center)
        
        # Assign each point to nearest center
        labels = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            min_distance = float('inf')
            nearest_cluster = 0
            
            for cluster_id, center in enumerate(cluster_centers):
                distance = self._calculate_distance(x, center)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = cluster_id
            
            labels[i] = nearest_cluster
        
        return labels
    
    def fit_predict(self, X):
        """
        Perform clustering and return cluster labels
        
        Convenience method that calls fit(X) and returns the resulting labels_
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : numpy array of shape (n_samples,)
            Cluster assignment for each point
        """
        self.fit(X)
        return self.labels_
    
    def get_linkage_matrix(self):
        """
        Get the linkage matrix for dendrogram visualization
        
        The linkage matrix shows the hierarchy of cluster merges.
        Compatible with scipy.cluster.hierarchy.dendrogram
        
        Format: Each row is [cluster_i, cluster_j, distance, size]
        - cluster_i, cluster_j: The clusters being merged. IDs 0..n_samples-1
          are the original points; the cluster created by row m has ID
          n_samples + m. Row m's two IDs are always strictly less than
          n_samples + m, which is what makes the matrix a valid tree.
        - distance: Distance at which they're merged (non-decreasing down the
          matrix for single/complete/average/ward)
        - size: Number of points in the new cluster
        
        Returns:
        --------
        linkage_matrix : numpy array of shape (n_samples-1, 4)
            Hierarchical clustering encoded as a linkage matrix
        """
        if self.linkage_matrix_ is None:
            raise ValueError(
                "Model is not fitted yet. Call fit(X) before get_linkage_matrix()."
            )
        
        return self.linkage_matrix_


"""
USAGE EXAMPLE 1: Simple 2D Clustering

import numpy as np

# Sample data: Geographic locations [latitude, longitude]
X = np.array([
    # West Coast cities
    [37.77, -122.42],  # San Francisco
    [34.05, -118.24],  # Los Angeles
    [47.61, -122.33],  # Seattle
    
    # East Coast cities
    [40.71, -74.01],   # New York
    [42.36, -71.06],   # Boston
    [38.91, -77.04],   # Washington DC
    
    # Midwest cities
    [41.88, -87.63],   # Chicago
    [44.98, -93.27],   # Minneapolis
])

city_names = ['SF', 'LA', 'Seattle', 'NYC', 'Boston', 'DC', 'Chicago', 'Minneapolis']

# Create and fit the model
model = HierarchicalClustering(n_clusters=3, linkage='average')
labels = model.fit_predict(X)

print("Cluster assignments:", labels)
# Output: [0 0 0 1 1 1 2 2]

# The full merge history is available too:
print("Merge heights:", np.round(model.get_linkage_matrix()[:, 2], 3))
# Output: Merge heights: [ 3.38   5.214  5.596  6.436 12.002 16.778 40.954]
# The jump from 6.44 to 12.00 is where you would cut for 3 clusters.

print("\nCities by cluster:")
for cluster in range(3):
    cities = [city_names[i] for i in range(len(labels)) if labels[i] == cluster]
    print(f"Cluster {cluster}: {', '.join(cities)}")

# Output:
# Cluster 0: SF, LA, Seattle
# Cluster 1: NYC, Boston, DC
# Cluster 2: Chicago, Minneapolis
"""

"""
USAGE EXAMPLE 2: Visualizing the Dendrogram

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Generate sample data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)

# Fit hierarchical clustering
model = HierarchicalClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)

# Get linkage matrix for dendrogram
linkage_matrix = model.get_linkage_matrix()

# Plot dendrogram
plt.figure(figsize=(12, 5))

# Subplot 1: Dendrogram
plt.subplot(1, 2, 1)
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Subplot 2: Cluster assignments
plt.subplot(1, 2, 2)
colors = ['red', 'blue', 'green']
for cluster in range(3):
    cluster_points = X[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
plt.title('Cluster Assignments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
"""

"""
USAGE EXAMPLE 3: Comparing Different Linkage Methods

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Generate non-spherical clusters (two moons)
X, y_true = make_moons(n_samples=120, noise=0.05, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']

print("Comparing Linkage Methods on Non-Spherical Data:\n")
print(f"{'Linkage':<15} {'Silhouette':<12} {'ARI vs truth':<14} {'Notes':<40}")
print("-" * 85)

for linkage in linkage_methods:
    model = HierarchicalClustering(n_clusters=2, linkage=linkage)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    ari = adjusted_rand_score(y_true, labels)
    
    notes = {
        'single': 'Best for elongated/non-spherical clusters',
        'complete': 'Creates compact clusters',
        'average': 'Balanced approach',
        'ward': 'Minimizes variance (assumes spherical)'
    }

    print(f"{linkage:<15} {score:<12.3f} {ari:<14.3f} {notes[linkage]:<40}")

# Output:
# single          0.380        1.000          Best for elongated/non-spherical clusters
# complete        0.452        0.692          Creates compact clusters
# average         0.454        0.720          Balanced approach
# ward            0.454        0.720          Minimizes variance (assumes spherical)
#
# Observation: read the two columns against each other. Single linkage RECOVERS
# THE MOONS PERFECTLY (ARI = 1.000) yet scores the LOWEST silhouette. Silhouette
# compares each point to cluster centroids, so it rewards convex blobs and
# punishes correct crescent-shaped clusters. Never pick a linkage on silhouette
# alone when the clusters are not roughly spherical.
"""

"""
USAGE EXAMPLE 4: Finding Optimal Number of Clusters

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generate data with 4 natural clusters
X, _ = make_blobs(n_samples=120, centers=4, n_features=2,
                  cluster_std=0.8, random_state=42)

# BUILD ONCE: the dendrogram does not depend on k at all
model = HierarchicalClustering(n_clusters=2, linkage='ward')
model.fit(X)

# CUT MANY TIMES: set_n_clusters re-cuts the stored hierarchy, no refitting
k_range = range(2, 9)
silhouette_scores = []

print("Finding Optimal Number of Clusters:\n")
print(f"{'k':<5} {'Silhouette Score':<20}")
print("-" * 25)

for k in k_range:
    labels = model.set_n_clusters(k).labels_
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"{k:<5} {score:<20.3f}")

# Find optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")
# Output:
# 2     0.601
# 3     0.778
# 4     0.832
# 5     0.709
# 6     0.564
# 7     0.429
# 8     0.449
#
# Optimal number of clusters: 4   <- matches the 4 planted centres
"""

"""
USAGE EXAMPLE 5: Real Dataset - Iris Species

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np

# Load iris dataset
data = load_iris()
X = data.data
y_true = data.target

# Standardize features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit hierarchical clustering
model = HierarchicalClustering(n_clusters=3, linkage='ward')
y_pred = model.fit_predict(X_scaled)

# Evaluate clustering
silhouette = silhouette_score(X_scaled, y_pred)
ari = adjusted_rand_score(y_true, y_pred)

print("Iris Dataset Clustering Results:\n")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"(ARI measures agreement with true labels)")

# Analyze each cluster
print("\nCluster Analysis:")
for cluster in range(3):
    cluster_mask = y_pred == cluster
    cluster_size = np.sum(cluster_mask)
    
    # Which true species are in this cluster?
    species_counts = [np.sum((y_true == i) & cluster_mask) for i in range(3)]
    dominant_species = np.argmax(species_counts)
    
    print(f"\nCluster {cluster} (n={cluster_size}):")
    print(f"  Setosa: {species_counts[0]}")
    print(f"  Versicolor: {species_counts[1]}")
    print(f"  Virginica: {species_counts[2]}")
    print(f"  Dominant species: {data.target_names[dominant_species]}")
"""

"""
USAGE EXAMPLE 6: Document Clustering (Text Data)

import numpy as np

# Simulate document vectors (in practice, use TF-IDF or embeddings)
# Each document represented as a feature vector
np.random.seed(42)

# Create 3 groups of documents with similar features
# Group 1: Sports documents
sports_docs = np.random.randn(10, 5) + np.array([3, 0, 0, 0, 0])

# Group 2: Technology documents
tech_docs = np.random.randn(10, 5) + np.array([0, 3, 0, 0, 0])

# Group 3: Politics documents
politics_docs = np.random.randn(10, 5) + np.array([0, 0, 3, 0, 0])

X = np.vstack([sports_docs, tech_docs, politics_docs])

document_names = (
    [f"Sports_{i}" for i in range(10)] +
    [f"Tech_{i}" for i in range(10)] +
    [f"Politics_{i}" for i in range(10)]
)

# Perform hierarchical clustering
model = HierarchicalClustering(n_clusters=3, linkage='average')
labels = model.fit_predict(X)

print("Document Clustering Results:\n")

# Show which documents are in each cluster
for cluster in range(3):
    docs_in_cluster = [document_names[i] for i in range(len(labels)) 
                       if labels[i] == cluster]
    print(f"Cluster {cluster} ({len(docs_in_cluster)} documents):")
    print(f"  {', '.join(docs_in_cluster[:5])}...")
    print()
"""

"""
USAGE EXAMPLE 7: Customer Segmentation with Multiple Features

import numpy as np

# Customer data: [Age, Income (k$), Spending Score (1-100), Years as Customer]
X = np.array([
    [25, 40, 81, 1],    # Young, medium income, high spending, new
    [28, 45, 78, 2],
    [23, 38, 85, 1],
    [45, 85, 90, 5],    # Middle-aged, high income, high spending, loyal
    [48, 90, 88, 6],
    [42, 80, 92, 4],
    [65, 60, 30, 10],   # Senior, medium income, low spending, very loyal
    [68, 55, 28, 12],
    [62, 58, 35, 9],
    [30, 40, 25, 3],    # Young, medium income, low spending
    [32, 42, 22, 2],
    [28, 38, 28, 3],
])

# Standardize features (different scales)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical clustering
model = HierarchicalClustering(n_clusters=4, linkage='ward')
labels = model.fit_predict(X_scaled)

print("Customer Segmentation Analysis:\n")
print("=" * 60)

# Analyze each segment
segment_names = []
for cluster in range(4):
    cluster_mask = labels == cluster
    cluster_data = X[cluster_mask]
    
    avg_age = np.mean(cluster_data[:, 0])
    avg_income = np.mean(cluster_data[:, 1])
    avg_spending = np.mean(cluster_data[:, 2])
    avg_years = np.mean(cluster_data[:, 3])
    
    print(f"\nSegment {cluster} (n={np.sum(cluster_mask)}):")
    print(f"  Average Age: {avg_age:.1f}")
    print(f"  Average Income: ${avg_income:.1f}k")
    print(f"  Average Spending Score: {avg_spending:.1f}")
    print(f"  Average Years as Customer: {avg_years:.1f}")
    
    # Assign descriptive names
    if avg_spending > 70:
        spending_label = "High Spenders"
    elif avg_spending > 40:
        spending_label = "Medium Spenders"
    else:
        spending_label = "Low Spenders"
    
    if avg_years > 7:
        loyalty_label = "Very Loyal"
    elif avg_years > 4:
        loyalty_label = "Loyal"
    else:
        loyalty_label = "New/Recent"
    
    segment_name = f"{spending_label}, {loyalty_label}"
    segment_names.append(segment_name)
    print(f"  Segment Name: {segment_name}")

print("\n" + "=" * 60)
print("Marketing Recommendations:")
for i, name in enumerate(segment_names):
    print(f"\nSegment {i} ({name}):")
    if "High Spenders" in name and "Loyal" in name:
        print("  -> VIP treatment, exclusive offers, loyalty rewards")
    elif "High Spenders" in name:
        print("  -> Convert to loyal customers, membership programs")
    elif "Low Spenders" in name and "Loyal" in name:
        print("  -> Understand needs, personalized offers to increase spending")
    else:
        print("  -> Engagement campaigns, incentives to increase activity")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _12_hierarchical_clustering.py
    # numpy only - no sklearn, no scipy, no matplotlib.
    # ----------------------------------------------------------------
    np.random.seed(42)

    def cluster_purity(true_labels, pred_labels):
        """
        Fraction of points that sit in a cluster dominated by their own class.

        Clustering labels are arbitrary names, so accuracy cannot be compared
        directly. Purity maps each cluster to its majority true class first.
        """
        correct = 0
        for cluster in np.unique(pred_labels):
            members = true_labels[pred_labels == cluster]
            correct += np.bincount(members).max()
        return correct / len(true_labels)

    # ================================================================
    # DEMO 1 - Do the four linkage rules find three planted blobs?
    # ================================================================
    print("=" * 62)
    print("DEMO 1 - Linkage comparison on 3 planted Gaussian blobs")
    print("=" * 62)

    blob_centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    per_blob = 30
    X_blobs = np.vstack([np.random.randn(per_blob, 2) * 0.6 + c
                         for c in blob_centers])
    y_blobs = np.repeat([0, 1, 2], per_blob)

    # Shuffle BEFORE splitting so train and test both cover all three blobs
    order = np.random.permutation(len(X_blobs))
    X_blobs, y_blobs = X_blobs[order], y_blobs[order]
    X_train, X_test = X_blobs[:60], X_blobs[60:]
    y_train, y_test = y_blobs[:60], y_blobs[60:]

    print(f"train = {X_train.shape[0]} points, test = {X_test.shape[0]} points, "
          f"3 true blobs\n")
    print(f"{'linkage':<10} {'train purity':<14} {'sizes':<16} {'last 3 merge heights'}")
    print("-" * 62)

    for linkage_name in ['single', 'complete', 'average', 'ward']:
        model = HierarchicalClustering(n_clusters=3, linkage=linkage_name)
        labels = model.fit_predict(X_train)
        purity = cluster_purity(y_train, labels)
        sizes = [int(s) for s in np.bincount(labels)]
        heights = model.linkage_matrix_[-3:, 2]
        print(f"{linkage_name:<10} {purity:<14.3f} {str(sizes):<16} "
              f"[{heights[0]:.2f}, {heights[1]:.2f}, {heights[2]:.2f}]")

    print("\nThose three heights are the last three merges: 4->3 clusters,")
    print("then 3->2, then 2->1. Look at the jump between the FIRST and the")
    print("SECOND of them - joining the final three blobs into two costs several")
    print("times more than any merge before it. Cut below that jump and you are")
    print("left with 3 clusters, which is the dendrogram telling you k=3.")

    # ================================================================
    # DEMO 2 - Generalising to unseen points via nearest centroid
    # ================================================================
    print("\n" + "=" * 62)
    print("DEMO 2 - Ward clustering, then predicting held-out points")
    print("=" * 62)

    model = HierarchicalClustering(n_clusters=3, linkage='ward')
    model.fit(X_train)

    train_purity = cluster_purity(y_train, model.labels_)
    test_labels = model.predict(X_test)
    test_purity = cluster_purity(y_test, test_labels)

    print(f"Train purity : {train_purity:.4f}")
    print(f"Test  purity : {test_purity:.4f}")
    print("\nSample predictions (x, y, true blob, predicted cluster):")
    for i in range(5):
        print(f"  ({X_test[i, 0]:6.2f}, {X_test[i, 1]:6.2f})  "
              f"true blob={y_test[i]}  predicted cluster={test_labels[i]}")

    # ================================================================
    # DEMO 3 - Cities: read the dendrogram, then re-cut it
    # ================================================================
    print("\n" + "=" * 62)
    print("DEMO 3 - Clustering 8 US cities by latitude / longitude")
    print("=" * 62)

    cities = np.array([
        [37.77, -122.42],   # San Francisco
        [34.05, -118.24],   # Los Angeles
        [47.61, -122.33],   # Seattle
        [40.71,  -74.01],   # New York
        [42.36,  -71.06],   # Boston
        [38.91,  -77.04],   # Washington DC
        [41.88,  -87.63],   # Chicago
        [44.98,  -93.27],   # Minneapolis
    ])
    city_names = ['SF', 'LA', 'Seattle', 'NYC', 'Boston', 'DC',
                  'Chicago', 'Minneapolis']

    city_model = HierarchicalClustering(n_clusters=3, linkage='average')
    city_labels = city_model.fit_predict(cities)

    print("Cluster assignments:", city_labels)
    for cluster in range(3):
        members = [city_names[i] for i in range(len(city_names))
                   if city_labels[i] == cluster]
        print(f"  Cluster {cluster}: {', '.join(members)}")

    heights = city_model.linkage_matrix_[:, 2]
    print("\nMerge heights (the dendrogram, one number per merge):")
    print("  " + "  ".join(f"{h:.2f}" for h in heights))
    # Cutting the tree just below merge number m leaves (n_samples - m) clusters,
    # so the biggest gap between consecutive heights suggests the natural k.
    gaps = np.diff(heights)
    ranked = np.argsort(gaps)[::-1]
    print("\nWhere to cut? Rank the gaps between consecutive merge heights:")
    for rank, m in enumerate(ranked[:3], start=1):
        print(f"  gap #{rank}: {heights[m]:6.2f} -> {heights[m + 1]:6.2f} "
              f"(size {gaps[m]:5.2f})  suggests k = {len(city_names) - (m + 1)}")
    print("  The dominant gap says 2 (coast vs the rest); k=3 splits off the Midwest.")

    # Build once, cut many times: no re-clustering happens here
    print("\nRe-cutting the SAME hierarchy at different k (no refit):")
    for k in [2, 3, 4]:
        city_model.set_n_clusters(k)
        groups = [[city_names[i] for i in range(len(city_names))
                   if city_model.labels_[i] == c] for c in range(k)]
        print(f"  k={k}: " + " | ".join("{" + ", ".join(g) + "}" for g in groups))

    # The metric changes the numbers, not necessarily the grouping
    manhattan_model = HierarchicalClustering(n_clusters=3, linkage='average',
                                             distance_metric='manhattan')
    manhattan_labels = manhattan_model.fit_predict(cities)
    print("\nSame data with distance_metric='manhattan':")
    print(f"  labels           : {manhattan_labels}")
    print(f"  same grouping?    {np.array_equal(manhattan_labels, city_labels)}")
    print("  first merge height: "
          f"{manhattan_model.linkage_matrix_[0, 2]:.2f} "
          f"(euclidean was {heights[0]:.2f})")
