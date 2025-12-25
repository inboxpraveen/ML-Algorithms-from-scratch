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
    
    where:
        n_clusters = number of clusters to form (can be changed after fitting)
        linkage = method for measuring cluster distance
        distance_metric = method for measuring point distance
    """
    
    def __init__(self, n_clusters=2, linkage='average', distance_metric='euclidean'):
        """
        Initialize the Hierarchical Clustering model
        
        Parameters:
        -----------
        n_clusters : int, default=2
            The number of clusters to find
            Can be changed by re-cutting the dendrogram
            Unlike k-Means, this can be decided after seeing the dendrogram
        
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
        
        distance_metric : str, default='euclidean'
            Distance metric to measure similarity between points
            Options: 'euclidean', 'manhattan'
            
            - 'euclidean': Straight-line distance (most common)
            - 'manhattan': City-block distance (sum of absolute differences)
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric
        
        # Model attributes (set after fitting)
        self.labels_ = None
        self.linkage_matrix_ = None
        self.n_samples_ = None
        
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
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
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
        
        if self.linkage == 'single':
            # Single linkage: minimum distance between any two points
            min_distance = float('inf')
            for x1 in cluster1_points:
                for x2 in cluster2_points:
                    distance = self._calculate_distance(x1, x2)
                    min_distance = min(min_distance, distance)
            return min_distance
        
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance between any two points
            max_distance = 0
            for x1 in cluster1_points:
                for x2 in cluster2_points:
                    distance = self._calculate_distance(x1, x2)
                    max_distance = max(max_distance, distance)
            return max_distance
        
        elif self.linkage == 'average':
            # Average linkage: average distance between all point pairs
            total_distance = 0
            count = 0
            for x1 in cluster1_points:
                for x2 in cluster2_points:
                    distance = self._calculate_distance(x1, x2)
                    total_distance += distance
                    count += 1
            return total_distance / count if count > 0 else 0
        
        elif self.linkage == 'ward':
            # Ward linkage: minimize within-cluster variance
            # Distance based on increase in sum of squares
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
        the two closest clusters until only one cluster remains.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data to cluster
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        self.n_samples_ = X.shape[0]
        
        # Initialize: each point is its own cluster
        # Store as list of lists containing point indices
        clusters = [[i] for i in range(self.n_samples_)]
        
        # Store merge history for dendrogram
        # Format: [cluster_i, cluster_j, distance, size]
        self.linkage_matrix_ = []
        
        # Keep track of cluster IDs (for linkage matrix)
        next_cluster_id = self.n_samples_
        
        # Merge until we have the desired number of clusters
        while len(clusters) > 1:
            # Find the two closest clusters
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            # Check all pairs of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._calculate_cluster_distance(
                        X, clusters[i], clusters[j]
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Store merge information for dendrogram
            cluster_i_id = clusters[merge_i][0] if len(clusters[merge_i]) == 1 else next_cluster_id - (self.n_samples_ - merge_i - 1)
            cluster_j_id = clusters[merge_j][0] if len(clusters[merge_j]) == 1 else next_cluster_id - (self.n_samples_ - merge_j - 1)
            
            self.linkage_matrix_.append([
                min(cluster_i_id, cluster_j_id),
                max(cluster_i_id, cluster_j_id),
                min_distance,
                len(clusters[merge_i]) + len(clusters[merge_j])
            ])
            
            # Merge the two closest clusters
            new_cluster = clusters[merge_i] + clusters[merge_j]
            
            # Remove old clusters (remove larger index first to avoid index shift)
            if merge_i < merge_j:
                clusters.pop(merge_j)
                clusters.pop(merge_i)
            else:
                clusters.pop(merge_i)
                clusters.pop(merge_j)
            
            # Add merged cluster
            clusters.append(new_cluster)
            next_cluster_id += 1
            
            # If we've reached the desired number of clusters, save them
            if len(clusters) == self.n_clusters:
                self._final_clusters = [c[:] for c in clusters]  # Deep copy
        
        # If n_clusters wasn't reached during merging, cut at the appropriate level
        if not hasattr(self, '_final_clusters'):
            # Rebuild clusters by cutting dendrogram at appropriate height
            self._cut_dendrogram(X)
        
        # Create labels from final clusters
        self.labels_ = np.zeros(self.n_samples_, dtype=int)
        for cluster_id, cluster_indices in enumerate(self._final_clusters):
            for idx in cluster_indices:
                self.labels_[idx] = cluster_id
        
        self.linkage_matrix_ = np.array(self.linkage_matrix_)
        
        return self
    
    def _cut_dendrogram(self, X):
        """
        Cut the dendrogram to get the desired number of clusters
        
        This is used when we want to extract a specific number of clusters
        from the already-built hierarchy.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Original data
        """
        # Start with each point as its own cluster
        clusters = [[i] for i in range(self.n_samples_)]
        
        # Replay merges until we have n_clusters
        num_merges = self.n_samples_ - self.n_clusters
        
        for merge_idx in range(num_merges):
            if merge_idx >= len(self.linkage_matrix_):
                break
                
            # Find which current clusters to merge based on linkage matrix
            # This is a simplified approach - in practice, would track cluster evolution
            # For now, merge from the beginning
            if len(clusters) > self.n_clusters:
                # Find two closest clusters
                min_distance = float('inf')
                merge_i, merge_j = -1, -1
                
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        distance = self._calculate_cluster_distance(
                            X, clusters[i], clusters[j]
                        )
                        if distance < min_distance:
                            min_distance = distance
                            merge_i, merge_j = i, j
                
                # Merge
                new_cluster = clusters[merge_i] + clusters[merge_j]
                clusters.pop(max(merge_i, merge_j))
                clusters.pop(min(merge_i, merge_j))
                clusters.append(new_cluster)
        
        self._final_clusters = clusters
    
    def predict(self, X):
        """
        Predict cluster labels for samples
        
        For new samples, assigns them to the nearest cluster center.
        For training samples, returns the stored labels.
        
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
            raise ValueError("Model must be fitted before predicting")
        
        # If X has same size as training data, might be the training data
        if X.shape[0] == self.n_samples_:
            return self.labels_
        
        # For new data, assign to nearest cluster center
        # Calculate cluster centers
        cluster_centers = []
        for cluster_indices in self._final_clusters:
            cluster_center = np.mean(X[cluster_indices], axis=0)
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
    
    def get_linkage_matrix(self):
        """
        Get the linkage matrix for dendrogram visualization
        
        The linkage matrix shows the hierarchy of cluster merges.
        Compatible with scipy.cluster.hierarchy.dendrogram
        
        Format: Each row is [cluster_i, cluster_j, distance, size]
        - cluster_i, cluster_j: The clusters being merged
        - distance: Distance at which they're merged
        - size: Number of points in the new cluster
        
        Returns:
        --------
        linkage_matrix : numpy array of shape (n_samples-1, 4)
            Hierarchical clustering encoded as a linkage matrix
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Model must be fitted first")
        
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
# Output: [0 0 0 1 1 1 2 2] (or similar)

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
from sklearn.metrics import silhouette_score

# Generate non-spherical clusters (two moons)
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']

print("Comparing Linkage Methods on Non-Spherical Data:\n")
print(f"{'Linkage':<15} {'Silhouette Score':<20} {'Notes':<40}")
print("-" * 75)

for linkage in linkage_methods:
    model = HierarchicalClustering(n_clusters=2, linkage=linkage)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    notes = {
        'single': 'Best for elongated/non-spherical clusters',
        'complete': 'Creates compact clusters',
        'average': 'Balanced approach',
        'ward': 'Minimizes variance (assumes spherical)'
    }
    
    print(f"{linkage:<15} {score:<20.3f} {notes[linkage]:<40}")

# Observation: Single linkage often works best for non-spherical clusters like moons
"""

"""
USAGE EXAMPLE 4: Finding Optimal Number of Clusters

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generate data with 4 natural clusters
X, _ = make_blobs(n_samples=200, centers=4, n_features=2, 
                  cluster_std=0.8, random_state=42)

# Try different numbers of clusters
k_range = range(2, 9)
silhouette_scores = []

print("Finding Optimal Number of Clusters:\n")
print(f"{'k':<5} {'Silhouette Score':<20}")
print("-" * 25)

for k in k_range:
    model = HierarchicalClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"{k:<5} {score:<20.3f}")

# Find optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")
# Should be close to 4 (the true number)
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
        print("  → VIP treatment, exclusive offers, loyalty rewards")
    elif "High Spenders" in name:
        print("  → Convert to loyal customers, membership programs")
    elif "Low Spenders" in name and "Loyal" in name:
        print("  → Understand needs, personalized offers to increase spending")
    else:
        print("  → Engagement campaigns, incentives to increase activity")
"""

