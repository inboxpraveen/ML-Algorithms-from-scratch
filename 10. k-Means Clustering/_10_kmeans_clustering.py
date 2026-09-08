import numpy as np

class KMeansClustering:
    """
    k-Means Clustering Implementation from Scratch
    
    k-Means is an unsupervised learning algorithm that groups similar data points
    into k clusters. It works by iteratively assigning points to the nearest cluster
    center and updating centers based on the mean of assigned points.
    
    Key Idea: "Group similar things together into k clusters"
    
    Use Cases:
    - Customer segmentation: group shoppers by income and spending behavior
    - Image color quantization: reduce a photo's colors to k representative ones
    - Document clustering: organize news articles or papers by topic
    - Anomaly detection: points far from every centroid are candidate outliers
    - Market segmentation: find niches from product, price and preference features

    The algorithm finds k cluster centers (centroids) such that:
    - Points within a cluster are as close as possible to their centroid
    - Points in different clusters are as far apart as possible
    
    Objective (minimized) -- the within-cluster sum of squares, a.k.a. inertia:

        J = sum over k of  sum over x in C_k of  ||x - c_k||^2

    Lloyd's algorithm minimizes J by alternating two steps, each of which
    can only leave J the same or make it smaller:

        Assign (E-step):  label(x) = argmin_k ||x - c_k||
        Update (M-step):  c_k      = (1 / n_k) * sum of x in C_k

    The Update step is the arithmetic mean because that is exactly where the
    derivative of the objective vanishes:

        d/dc_k sum_{x in C_k} ||x - c_k||^2 = -2 * sum_{x in C_k} (x - c_k) = 0
        =>  c_k = (1 / n_k) * sum_{x in C_k} x

    J is bounded below by 0 and there are finitely many possible partitions,
    so the alternation cannot decrease J forever: k-Means always converges
    (to a local optimum, not necessarily the global one).

    where:
        k        = number of clusters (n_clusters)
        C_k      = set of points currently assigned to cluster k
        c_k      = centroid of cluster k
        n_k      = number of points in cluster k
        max_iter = maximum number of iterations
        tol      = convergence tolerance on centroid movement
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='random',
                 random_state=None, n_init=1):
        """
        Initialize the k-Means Clustering model
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters to form
            Also the number of centroids to generate
            Choose based on domain knowledge or elbow method
            - Range: 2 to roughly sqrt(n_samples)
            - More clusters always lower the inertia, so it cannot be tuned
              by minimizing inertia alone
            Typical values: 2-10, chosen with the elbow or silhouette method
        
        max_iter : int, default=300
            Maximum number of iterations
            Algorithm stops if max_iter is reached
            Typical values: 100-500
        
        tol : float, default=1e-4
            Convergence tolerance
            Algorithm stops if centroid movement < tol, where the movement is
            the Frobenius norm ||new_centroids - old_centroids|| of the whole
            stacked centroid matrix (an absolute distance in feature units).
            - Range: 1e-6 to 1e-2
            - Smaller values = more precise convergence, more iterations
            - Note: scikit-learn compares the SQUARED shift against tol scaled
              by the mean feature variance, so the same numeric tol means
              something different there (see "Simplifications" in the .md)
            Typical values: 1e-4
        
        init : str, default='random'
            Method for initialization
            Options: 'random', 'kmeans++'
            - 'random': Randomly select k points as initial centroids
            - 'kmeans++': Smart initialization for better convergence
            Typical: 'kmeans++' is recommended - 'random' with a single start
            can land in a clearly worse local optimum (the default is kept at
            'random' only for backward compatibility with older examples)
        
        random_state : int or None, default=None
            Random seed for reproducibility
            Set to an integer for consistent results across runs
            Seeds a PRIVATE numpy RandomState; the global numpy random stream
            is never touched, so fitting will not disturb your own seeding

        n_init : int, default=1
            Number of times the whole algorithm is run with a different
            initialization. The run with the lowest inertia is kept.
            - Higher values = far more robust to unlucky starts, linearly slower
            - scikit-learn's KMeans defaults to 10 restarts ('auto'); the
              default here is 1 so that the documented example outputs stay
              reproducible
            Typical values: 1 for teaching runs, 10 for real use
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.n_init = n_init

        # Private random generator (created on first use / reset by fit).
        # Using a local RandomState instead of np.random.seed keeps the
        # caller's global random stream untouched.
        self._rng = None
        
        # Model attributes (set after fitting)
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _validate_input(self, X):
        """
        Coerce X into a 2-D float array of shape (n_samples, n_features)

        Accepts plain Python lists and 1-D arrays (a single feature), so the
        docstrings' promise of "array of shape (n_samples, n_features)" is
        honoured for the convenient inputs a learner is likely to type.

        Parameters:
        -----------
        X : array-like
            Data as a list, list of lists, 1-D array or 2-D array

        Returns:
        --------
        X : numpy array of shape (n_samples, n_features)
            Validated 2-D float array
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            # A single feature: [1, 2, 3] -> [[1], [2], [3]]
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 1-D or 2-D, got an array with {X.ndim} dimensions"
            )

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")

        return X
        
    def _initialize_centroids(self, X):
        """
        Initialize cluster centroids
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        centroids : numpy array of shape (n_clusters, n_features)
            Initial centroid positions
        """
        n_samples, n_features = X.shape
        
        # Private RNG so that fitting never rewrites the caller's global
        # np.random stream. fit() resets it, so repeated fits are reproducible.
        if self._rng is None:
            self._rng = np.random.RandomState(self.random_state)
        rng = self._rng
        
        if self.init == 'random':
            # Randomly select k data points as initial centroids
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]
            
        elif self.init == 'kmeans++':
            # k-means++ initialization for better convergence
            # (Arthur & Vassilvitskii 2007: spread the seeds out, weighting
            #  each candidate by its squared distance to the nearest seed)
            centroids = []
            
            # Choose first centroid randomly
            first_idx = rng.randint(0, n_samples)
            centroids.append(X[first_idx])
            
            # Choose remaining centroids
            for _ in range(1, self.n_clusters):
                # Calculate distance to nearest centroid for each point
                distances = np.array([min([np.linalg.norm(x - c) ** 2 
                                          for c in centroids]) for x in X])
                
                # Choose next centroid with probability proportional to distance^2
                total = distances.sum()
                if total <= 0:
                    # Every remaining point already sits on a chosen centroid
                    # (e.g. duplicated rows), so D(x)^2 = 0 everywhere and the
                    # probabilities would be 0/0. Fall back to a uniform pick.
                    next_idx = rng.randint(0, n_samples)
                else:
                    probabilities = distances / total
                    next_idx = rng.choice(n_samples, p=probabilities)
                centroids.append(X[next_idx])
            
            centroids = np.array(centroids)
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
        
        return centroids
    
    def _assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data points
            
        Returns:
        --------
        labels : numpy array of shape (n_samples,)
            Cluster assignment for each point (0 to n_clusters-1)
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        # For each data point:  label(x) = argmin_k ||x - c_k||
        for i, x in enumerate(X):
            # Calculate distance to each centroid
            distances = np.linalg.norm(x - self.centroids, axis=1)
            
            # Assign to nearest centroid
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Update centroids as the mean of assigned points

        Implements the M-step:  c_k = (1 / n_k) * sum of x in C_k

        Empty-cluster rule: if a cluster loses every point, its centroid is
        relocated to the sample that is currently farthest from its own
        centroid (this is what scikit-learn does). Freezing the dead centroid
        instead would let k silently collapse to fewer effective clusters.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data points
        labels : numpy array of shape (n_samples,)
            Current cluster assignments
            
        Returns:
        --------
        new_centroids : numpy array of shape (n_clusters, n_features)
            Updated centroid positions
        """
        n_samples, n_features = X.shape
        new_centroids = np.zeros((self.n_clusters, n_features))

        # Distance of every point to its currently assigned centroid.
        # Only needed if some cluster turns out empty, so it is built lazily.
        point_distances = None
        
        # For each cluster
        for k in range(self.n_clusters):
            # Find all points assigned to this cluster
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Update centroid to mean of assigned points
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Empty cluster: relocate it to the worst-served point
                if point_distances is None:
                    point_distances = np.array([
                        np.linalg.norm(X[i] - self.centroids[labels[i]])
                        for i in range(n_samples)
                    ])
                farthest = np.argmax(point_distances)
                new_centroids[k] = X[farthest]
                # Do not hand the same point to a second empty cluster
                point_distances[farthest] = -1.0
        
        return new_centroids
    
    def _calculate_inertia(self, X, labels):
        """
        Calculate inertia (within-cluster sum of squares)
        
        Inertia measures how compact the clusters are.
        Lower inertia = tighter clusters = better fit
        
        Inertia = sum over all points of (distance from a point to its centroid)^2
        This is the objective J the algorithm minimizes.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data points
        labels : numpy array of shape (n_samples,)
            Cluster assignments
            
        Returns:
        --------
        inertia : float
            Sum of squared distances to nearest centroid
        """
        inertia = 0
        
        for i, x in enumerate(X):
            # Distance to assigned centroid
            centroid = self.centroids[labels[i]]
            inertia += np.linalg.norm(x - centroid) ** 2
        
        return inertia
    
    def fit(self, X):
        """
        Compute k-means clustering
        
        Runs Lloyd's algorithm n_init times from different initializations and
        keeps the run with the lowest inertia J, because a single start can get
        trapped in a poor local optimum.

        Each run alternates:
            E-step: labels      = argmin_k ||x - c_k||       (_assign_clusters)
            M-step: c_k         = mean of the points in C_k  (_update_centroids)
        and stops when ||new_centroids - old_centroids|| < tol.

        After the loop a FINAL E-step is performed so that labels_, inertia_
        and centroids always describe the same state (without it, labels_ would
        come from the centroids of the previous iteration).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data (lists and 1-D arrays are accepted and converted)
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        X = self._validate_input(X)
        
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot exceed the number of "
                f"samples ({X.shape[0]})"
            )

        # Reset the private RNG so that repeated fits of the same model with
        # the same random_state give identical results.
        self._rng = np.random.RandomState(self.random_state)

        best_inertia = None
        best_centroids = None
        best_labels = None
        best_n_iter = None

        # Multiple restarts: keep whichever run reaches the lowest inertia
        for _ in range(max(1, self.n_init)):
            # Initialize centroids
            self.centroids = self._initialize_centroids(X)

            n_iter = self.max_iter

            # Iterative optimization
            for iteration in range(self.max_iter):
                # Step 1: Assign each point to nearest centroid (E-step)
                labels = self._assign_clusters(X)

                # Step 2: Update centroids (M-step)
                new_centroids = self._update_centroids(X, labels)

                # Check for convergence: how far did the centroids move?
                centroid_shift = np.linalg.norm(new_centroids - self.centroids)

                self.centroids = new_centroids

                # Stop if converged
                if centroid_shift < self.tol:
                    n_iter = iteration + 1
                    break

            # Final assignment against the FINAL centroids, so that the stored
            # labels and inertia are consistent with the stored centroids.
            labels = self._assign_clusters(X)
            inertia = self._calculate_inertia(X, labels)
            
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids
                best_labels = labels
                best_n_iter = n_iter
            
        # Store the best run
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to assign to clusters
            
        Returns:
        --------
        labels : numpy array of shape (n_samples,)
            Cluster assignment for each point
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before predicting")
        
        X = self._validate_input(X)

        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """
        Compute clustering and return cluster labels
        
        Convenience method that calls fit(X) followed by predict(X).
        (fit() ends with a final assignment step, so the stored labels_ are
        exactly what predict(X) would return on the training data.)
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : numpy array of shape (n_samples,)
            Cluster assignment for each point
        """
        self.fit(X)
        return self.labels_
    
    def transform(self, X):
        """
        Transform X to cluster-distance space
        
        Returns the distance of each sample to each cluster centroid
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        distances : numpy array of shape (n_samples, n_clusters)
            Distance to each centroid for each sample
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before transforming")

        X = self._validate_input(X)
        
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i, x in enumerate(X):
            for k in range(self.n_clusters):
                distances[i, k] = np.linalg.norm(x - self.centroids[k])
        
        return distances
    
    def fit_transform(self, X):
        """
        Compute clustering and transform X to cluster-distance space
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        distances : numpy array of shape (n_samples, n_clusters)
            Distance to each centroid for each sample
        """
        self.fit(X)
        return self.transform(X)
    
    def score(self, X):
        """
        Calculate the negative inertia (for consistency with sklearn)
        
        Negative inertia is returned so that higher values indicate better fit
        (consistent with other sklearn metrics). Note this is NOT bounded like
        an accuracy or R^2 score: it is always <= 0 and its magnitude depends
        on the scale of the data and on the number of samples, so compare it
        only across models fitted on the same data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to evaluate
            
        Returns:
        --------
        score : float
            Negative inertia (-1 * within-cluster sum of squares)
        """
        X = self._validate_input(X)
        labels = self.predict(X)
        inertia = self._calculate_inertia(X, labels)
        return -inertia
    
    def get_cluster_centers(self):
        """
        Get the coordinates of cluster centers
        
        Returns:
        --------
        centroids : numpy array of shape (n_clusters, n_features)
            Coordinates of cluster centers

        Note: cluster IDs are arbitrary. Two runs that find the same geometry
        may number the clusters differently, so never rely on a specific ID.
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before accessing cluster centers")

        return self.centroids


"""
USAGE EXAMPLE 1: Simple 2D Clustering

import numpy as np

# Sample data: Customer segments based on [age, spending_score]
X = np.array([
    # Group 1: Young, low spending
    [25, 30], [28, 35], [23, 28], [26, 32],
    # Group 2: Middle-aged, high spending
    [45, 80], [48, 85], [42, 78], [47, 82],
    # Group 3: Senior, medium spending
    [65, 50], [62, 55], [68, 52], [63, 48]
])

# Create and fit the model
model = KMeansClustering(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

print("Cluster assignments:", labels)
# Output: [2 2 2 2 1 1 1 1 0 0 0 0]
# The three groups are recovered perfectly, but the cluster IDs themselves are
# arbitrary: a different seed can find the same geometry and number it
# [0 0 0 0 2 2 2 2 1 1 1 1]. Never rely on a particular ID - read the centroid.

print("\nCluster centers:")
print(model.get_cluster_centers())
# Output:
# [[64.5  51.25]    <- senior, medium spending
#  [45.5  81.25]    <- middle-aged, high spending
#  [25.5  31.25]]   <- young, low spending

print(f"\nInertia: {model.inertia_:.2f}")
# Output: Inertia: 135.25

print(f"Number of iterations: {model.n_iter_}")
# Output: Number of iterations: 3

# Predict cluster for new customers
X_new = np.array([[27, 33], [46, 81], [64, 51]])
predictions = model.predict(X_new)
print("\nPredictions for new customers:", predictions)
# Output: [2 1 0]  (young -> cluster 2, middle-aged -> 1, senior -> 0)
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Iris)

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load iris dataset (we'll ignore labels for unsupervised learning)
data = load_iris()
X = data.data

# Standardize features (important for k-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit k-means model
model = KMeansClustering(n_clusters=3, init='kmeans++', random_state=42)
labels = model.fit_predict(X_scaled)

print("Cluster assignments:", labels)
print(f"\nInertia: {model.inertia_:.2f}")
print(f"Converged in {model.n_iter_} iterations")

# Get cluster centers
centers = model.get_cluster_centers()
print("\nCluster centers (in scaled space):")
print(centers)

# Compare with true labels (for validation)
print("\nTrue labels:", data.target)

# Calculate purity (optional - for comparison with true labels)
from collections import Counter

for cluster in range(3):
    cluster_labels = data.target[labels == cluster]
    print(f"Cluster {cluster}: {Counter(cluster_labels)}")
"""

"""
USAGE EXAMPLE 3: Elbow Method to Find Optimal k

import numpy as np
from sklearn.datasets import make_blobs

# Generate synthetic data with 4 clusters
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                  cluster_std=0.6, random_state=42)

# Try different values of k
k_values = range(2, 11)
inertias = []

print("Finding optimal k using Elbow Method:\n")
print(f"{'k':<5} {'Inertia':<15}")
print("-" * 20)

for k in k_values:
    # init='kmeans++' with several restarts is essential here: a single
    # 'random' start gets trapped in a local optimum at k=4 and the elbow
    # disappears entirely (inertia 1755 instead of 204).
    model = KMeansClustering(n_clusters=k, init='kmeans++', n_init=10,
                             random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)
    print(f"{k:<5} {model.inertia_:<15.2f}")

# Output:
# 2     9051.82
# 3     1773.74
# 4     203.89          <- huge drop, then the curve flattens: the elbow
# 5     184.52
# 6     167.46
# 7     149.13
# 8     131.39
# 9     120.21
# 10    107.33

# The "elbow" point is where inertia starts decreasing more slowly
# In this case, k=4 is the elbow (since we created 4 clusters):
# 1773.74 -> 203.89 is an 88% drop, 203.89 -> 184.52 is only 10%.

print("\nLook for the 'elbow' where inertia decrease slows down")
print("That's typically the optimal number of clusters!")
"""

"""
USAGE EXAMPLE 4: Image Color Quantization

import numpy as np

# Simulate a small image (in practice, use real image data)
# Each pixel has 3 values: [Red, Green, Blue]
np.random.seed(42)

# Create image with dominant colors
image_data = []

# Add red pixels
image_data.extend([[200 + np.random.randint(-30, 30), 
                   50 + np.random.randint(-30, 30), 
                   50 + np.random.randint(-30, 30)] for _ in range(100)])

# Add blue pixels
image_data.extend([[50 + np.random.randint(-30, 30), 
                   50 + np.random.randint(-30, 30), 
                   200 + np.random.randint(-30, 30)] for _ in range(100)])

# Add green pixels
image_data.extend([[50 + np.random.randint(-30, 30), 
                   200 + np.random.randint(-30, 30), 
                   50 + np.random.randint(-30, 30)] for _ in range(100)])

X = np.array(image_data)

# Use k-means to find dominant colors
n_colors = 3
model = KMeansClustering(n_clusters=n_colors, random_state=42)
model.fit(X)

# Get dominant colors (cluster centers)
dominant_colors = model.get_cluster_centers().astype(int)

print("Dominant Colors (RGB):")
for i, color in enumerate(dominant_colors):
    print(f"  Color {i+1}: RGB({color[0]}, {color[1]}, {color[2]})")

print(f"\nOriginal image had {len(X)} pixels")
print(f"Reduced to {n_colors} representative colors")
print(f"Compression ratio: {len(X) / n_colors:.1f}x")
"""

"""
USAGE EXAMPLE 5: Customer Segmentation

import numpy as np

# Customer data: [Annual Income (k$), Spending Score (1-100)]
# This is similar to the Mall Customers dataset
X = np.array([
    [15, 39], [15, 81], [16, 6], [16, 77], [17, 40],
    [18, 76], [19, 6], [19, 94], [20, 3], [20, 72],
    [23, 14], [23, 99], [24, 15], [25, 77], [26, 13],
    [27, 79], [28, 35], [28, 97], [29, 23], [30, 69],
    [35, 14], [35, 98], [37, 15], [37, 97], [38, 16],
    [39, 96], [40, 40], [40, 71], [41, 36], [42, 73],
    [48, 12], [48, 82], [49, 15], [50, 80], [51, 17],
    [52, 85], [53, 23], [54, 73], [55, 35], [56, 92],
    [60, 13], [60, 81], [62, 17], [63, 86], [64, 18],
    [65, 83], [67, 33], [68, 92], [69, 37], [70, 75]
])

# Fit k-means with 5 clusters
model = KMeansClustering(n_clusters=5, init='kmeans++', random_state=42)
labels = model.fit_predict(X)

print("Customer Segmentation Analysis\n")
print("=" * 50)

# Analyze each cluster
for cluster in range(5):
    cluster_data = X[labels == cluster]
    n_customers = len(cluster_data)

    # Guard: np.mean of an empty slice returns nan with a RuntimeWarning
    if n_customers == 0:
        print(f"\nCluster {cluster + 1}: empty (no customers assigned)")
        continue

    avg_income = np.mean(cluster_data[:, 0])
    avg_spending = np.mean(cluster_data[:, 1])
    
    print(f"\nCluster {cluster + 1}:")
    print(f"  Number of customers: {n_customers}")
    print(f"  Average income: ${avg_income:.1f}k")
    print(f"  Average spending score: {avg_spending:.1f}")
    
    # Assign segment names based on characteristics
    if avg_spending > 70:
        segment_name = "High Spenders"
    elif avg_spending > 40:
        segment_name = "Medium Spenders"
    else:
        segment_name = "Low Spenders"
    
    if avg_income > 50:
        income_level = "High Income"
    elif avg_income > 30:
        income_level = "Medium Income"
    else:
        income_level = "Low Income"
    
    print(f"  Segment: {income_level}, {segment_name}")

print(f"\n{'=' * 50}")
print(f"Total inertia: {model.inertia_:.2f}")
print(f"Converged in: {model.n_iter_} iterations")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _10_kmeans_clustering.py
    # Requires numpy only. Everything below is seeded and reproducible.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Demo 1: can k-Means recover clusters we planted ourselves? ---
    print("=" * 55)
    print("DEMO 1 - Recovering three planted Gaussian blobs")
    print("=" * 55)

    true_centers = np.array([[0.0, 0.0], [7.0, 7.0], [0.0, 7.0]])
    X_blobs = np.vstack([c + np.random.randn(100, 2) for c in true_centers])

    # Shuffle before slicing, so the held-out points come from all three blobs
    order = np.random.permutation(len(X_blobs))
    X_blobs = X_blobs[order]
    X_train, X_test = X_blobs[:240], X_blobs[240:]

    km = KMeansClustering(n_clusters=3, init='kmeans++', n_init=10,
                          random_state=42)
    km.fit(X_train)

    # score() returns NEGATIVE inertia, so divide by n to compare train vs test
    print(f"Converged in {km.n_iter_} iterations")
    print(f"Train inertia (sum of squared distances) : {km.inertia_:8.2f}")
    print(f"Train mean squared distance to centroid  : "
          f"{-km.score(X_train) / len(X_train):8.4f}")
    print(f"Test  mean squared distance to centroid  : "
          f"{-km.score(X_test) / len(X_test):8.4f}")

    print("\nRecovered centroids vs the centers we planted:")
    for c in km.get_cluster_centers():
        nearest = true_centers[np.argmin(np.linalg.norm(true_centers - c, axis=1))]
        print(f"  found ({c[0]:5.2f}, {c[1]:5.2f})  ->  planted "
              f"({nearest[0]:.1f}, {nearest[1]:.1f})")

    print(f"\nCluster sizes on the training set: {np.bincount(km.labels_)}")
    print("(Cluster IDs are arbitrary - only the geometry is meaningful.)")

    test_labels = km.predict(X_test)
    test_distances = km.transform(X_test)
    print("\nSample test points -> assigned cluster, distance to its centroid:")
    for i in range(5):
        print(f"  ({X_test[i, 0]:6.2f}, {X_test[i, 1]:6.2f}) -> cluster "
              f"{test_labels[i]}, distance {test_distances[i, test_labels[i]]:.2f}")

    # --- Demo 2: choosing k with the elbow method ---
    print("\n" + "=" * 55)
    print("DEMO 2 - Elbow method: how many clusters are there?")
    print("=" * 55)
    print("Inertia always falls as k grows, so we look for the ELBOW:")
    print("the k after which extra clusters stop buying much.\n")

    print(f"{'k':<4}{'inertia':>12}{'drop vs k-1':>14}")
    print("-" * 30)
    previous = None
    for k in range(1, 7):
        sweep = KMeansClustering(n_clusters=k, init='kmeans++', n_init=10,
                                 random_state=42)
        sweep.fit(X_train)
        if previous is None:
            drop = "-"
        else:
            drop = f"{100.0 * (previous - sweep.inertia_) / previous:.1f} %"
        print(f"{k:<4}{sweep.inertia_:12.2f}{drop:>14}")
        previous = sweep.inertia_

    print("\nTakeaway: the drops collapse after k=3, which is exactly the")
    print("number of blobs we planted. The elbow found the right answer.")

    # --- Demo 3: color quantization (the classic k-Means application) ---
    print("\n" + "=" * 55)
    print("DEMO 3 - Color quantization: 450 pixels -> 3 colors")
    print("=" * 55)

    reds = np.column_stack([np.random.randint(190, 231, 150),
                            np.random.randint(30, 71, 150),
                            np.random.randint(30, 71, 150)])
    greens = np.column_stack([np.random.randint(30, 71, 150),
                              np.random.randint(190, 231, 150),
                              np.random.randint(50, 91, 150)])
    blues = np.column_stack([np.random.randint(20, 61, 150),
                             np.random.randint(60, 101, 150),
                             np.random.randint(190, 231, 150)])
    pixels = np.vstack([reds, greens, blues]).astype(float)

    palette = KMeansClustering(n_clusters=3, init='kmeans++', n_init=10,
                               random_state=42)
    pixel_labels = palette.fit_predict(pixels)
    colors = palette.get_cluster_centers().astype(int)
    counts = np.bincount(pixel_labels, minlength=3)

    print("Dominant colors (every pixel is replaced by its centroid):")
    for i in range(len(colors)):
        print(f"  Color {i}: RGB({colors[i, 0]:3d}, {colors[i, 1]:3d}, "
              f"{colors[i, 2]:3d})  used by {counts[i]:3d} pixels")

    print(f"\n{len(pixels)} distinct-ish RGB triples -> 3 palette entries "
          f"({len(pixels) / 3:.0f}x fewer colors)")
    print(f"score(X) = {palette.score(pixels):.2f}  "
          f"(negative inertia, so higher is better)")

    new_pixels = np.array([[205.0, 45.0, 45.0],
                           [45.0, 205.0, 70.0],
                           [35.0, 80.0, 210.0]])
    print("\nAssigning 3 unseen pixels to the fitted palette:")
    for px, lab in zip(new_pixels, palette.predict(new_pixels)):
        print(f"  RGB({px[0]:5.0f}, {px[1]:5.0f}, {px[2]:5.0f}) -> color {lab} "
              f"= RGB({colors[lab, 0]:3d}, {colors[lab, 1]:3d}, "
              f"{colors[lab, 2]:3d})")

