import numpy as np

class KMeansClustering:
    """
    k-Means Clustering Implementation from Scratch
    
    k-Means is an unsupervised learning algorithm that groups similar data points
    into k clusters. It works by iteratively assigning points to the nearest cluster
    center and updating centers based on the mean of assigned points.
    
    Key Idea: "Group similar things together into k clusters"
    
    The algorithm finds k cluster centers (centroids) such that:
    - Points within a cluster are as close as possible to their centroid
    - Points in different clusters are as far apart as possible
    
    where:
        k = number of clusters
        max_iter = maximum number of iterations
        tol = convergence tolerance
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='random', random_state=None):
        """
        Initialize the k-Means Clustering model
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters to form
            Also the number of centroids to generate
            Choose based on domain knowledge or elbow method
        
        max_iter : int, default=300
            Maximum number of iterations
            Algorithm stops if max_iter is reached
            Typical values: 100-500
        
        tol : float, default=1e-4
            Convergence tolerance
            Algorithm stops if centroid movement < tol
            Smaller values = more precise convergence
        
        init : str, default='random'
            Method for initialization
            Options: 'random', 'kmeans++'
            - 'random': Randomly select k points as initial centroids
            - 'kmeans++': Smart initialization for better convergence
        
        random_state : int or None, default=None
            Random seed for reproducibility
            Set to an integer for consistent results across runs
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        # Model attributes (set after fitting)
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
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
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.init == 'random':
            # Randomly select k data points as initial centroids
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]
            
        elif self.init == 'kmeans++':
            # k-means++ initialization for better convergence
            centroids = []
            
            # Choose first centroid randomly
            first_idx = np.random.randint(0, n_samples)
            centroids.append(X[first_idx])
            
            # Choose remaining centroids
            for _ in range(1, self.n_clusters):
                # Calculate distance to nearest centroid for each point
                distances = np.array([min([np.linalg.norm(x - c) ** 2 
                                          for c in centroids]) for x in X])
                
                # Choose next centroid with probability proportional to distance²
                probabilities = distances / distances.sum()
                next_idx = np.random.choice(n_samples, p=probabilities)
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
        
        # For each data point
        for i, x in enumerate(X):
            # Calculate distance to each centroid
            distances = np.linalg.norm(X[i] - self.centroids, axis=1)
            
            # Assign to nearest centroid
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Update centroids as the mean of assigned points
        
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
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        # For each cluster
        for k in range(self.n_clusters):
            # Find all points assigned to this cluster
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Update centroid to mean of assigned points
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If no points assigned, keep old centroid or reinitialize
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    
    def _calculate_inertia(self, X, labels):
        """
        Calculate inertia (within-cluster sum of squares)
        
        Inertia measures how compact the clusters are.
        Lower inertia = tighter clusters = better fit
        
        Inertia = Σ(distance from each point to its centroid)²
        
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
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Iterative optimization
        for iteration in range(self.max_iter):
            # Step 1: Assign each point to nearest centroid
            labels = self._assign_clusters(X)
            
            # Step 2: Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            self.centroids = new_centroids
            
            # Stop if converged
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            # Maximum iterations reached
            self.n_iter_ = self.max_iter
        
        # Store final labels and inertia
        self.labels_ = labels
        self.inertia_ = self._calculate_inertia(X, labels)
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            New data to assign to clusters
            
        Returns:
        --------
        labels : numpy array of shape (n_samples,)
            Cluster assignment for each point
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before predicting")
        
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """
        Compute clustering and return cluster labels
        
        Convenience method that calls fit(X) followed by predict(X)
        
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
    
    def transform(self, X):
        """
        Transform X to cluster-distance space
        
        Returns the distance of each sample to each cluster centroid
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        distances : numpy array of shape (n_samples, n_clusters)
            Distance to each centroid for each sample
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before transforming")
        
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
        (consistent with other sklearn metrics)
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to evaluate
            
        Returns:
        --------
        score : float
            Negative inertia (-1 * within-cluster sum of squares)
        """
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
        """
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
# Output: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] (or similar)

print("\nCluster centers:")
print(model.get_cluster_centers())

print(f"\nInertia: {model.inertia_:.2f}")
print(f"Number of iterations: {model.n_iter_}")

# Predict cluster for new customers
X_new = np.array([[27, 33], [46, 81], [64, 51]])
predictions = model.predict(X_new)
print("\nPredictions for new customers:", predictions)
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
    model = KMeansClustering(n_clusters=k, random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)
    print(f"{k:<5} {model.inertia_:<15.2f}")

# The "elbow" point is where inertia starts decreasing more slowly
# In this case, k=4 should be the elbow (since we created 4 clusters)

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
    avg_income = np.mean(cluster_data[:, 0])
    avg_spending = np.mean(cluster_data[:, 1])
    n_customers = len(cluster_data)
    
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

