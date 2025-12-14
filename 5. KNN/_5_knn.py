import numpy as np

class KNearestNeighbors:
    """
    K-Nearest Neighbors (KNN) Implementation from Scratch
    
    KNN is a simple, intuitive algorithm for classification and regression.
    It predicts by finding the k most similar examples in the training data
    and using their labels to make a prediction.
    
    Key Idea: "Similar inputs should have similar outputs"
    
    For classification: Predict the majority class among k nearest neighbors
    For regression: Predict the average value among k nearest neighbors
    
    where:
        k = number of nearest neighbors to consider
        distance = measure of similarity (typically Euclidean)
    """
    
    def __init__(self, k=5, distance_metric='euclidean', task='classification'):
        """
        Initialize the K-Nearest Neighbors model
        
        Parameters:
        -----------
        k : int, default=5
            Number of nearest neighbors to use for prediction
            Larger k = smoother decision boundary, more robust to noise
            Smaller k = more flexible, captures local patterns
            Typical values: 3, 5, 7 (odd numbers avoid ties)
        
        distance_metric : str, default='euclidean'
            Distance metric to measure similarity
            Options: 'euclidean', 'manhattan'
            - Euclidean: Straight-line distance (most common)
            - Manhattan: City-block distance (sum of absolute differences)
        
        task : str, default='classification'
            Type of prediction task
            Options: 'classification', 'regression'
            - classification: Predict categorical labels
            - regression: Predict continuous values
        """
        self.k = k
        self.distance_metric = distance_metric
        self.task = task
        self.X_train = None
        self.y_train = None
    
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
    
    def fit(self, X, y):
        """
        Train the KNN model (simply store the training data)
        
        KNN is a "lazy learner" - it doesn't actually learn anything!
        It just memorizes the training data and uses it at prediction time.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Target values (labels for classification, values for regression)
        """
        self.X_train = X
        self.y_train = y
    
    def _predict_single(self, x):
        """
        Predict for a single sample
        
        Parameters:
        -----------
        x : numpy array of shape (n_features,)
            Single sample to predict
            
        Returns:
        --------
        prediction : int or float
            Predicted label (classification) or value (regression)
        """
        # Calculate distances to all training samples
        distances = []
        for x_train in self.X_train:
            distance = self._calculate_distance(x, x_train)
            distances.append(distance)
        
        # Convert to numpy array for easier manipulation
        distances = np.array(distances)
        
        # Find indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels/values of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Make prediction based on task type
        if self.task == 'classification':
            # Classification: Return most common class (mode)
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            prediction = unique_labels[np.argmax(counts)]
        else:
            # Regression: Return average value (mean)
            prediction = np.mean(k_nearest_labels)
        
        return prediction
    
    def predict(self, X):
        """
        Predict labels or values for samples
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted labels (classification) or values (regression)
        """
        # Predict for each sample
        predictions = []
        for x in X:
            prediction = self._predict_single(x)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Calculate performance score
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test data
        y : numpy array of shape (n_samples,)
            True labels (classification) or values (regression)
            
        Returns:
        --------
        score : float
            Accuracy (classification) or R² score (regression)
        """
        predictions = self.predict(X)
        
        if self.task == 'classification':
            # Classification: Calculate accuracy
            accuracy = np.mean(predictions == y)
            return accuracy
        else:
            # Regression: Calculate R² score
            # R² = 1 - (SS_res / SS_tot)
            # where SS_res = sum of squared residuals
            #       SS_tot = total sum of squares
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            return r2_score
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification tasks
        
        Only works for classification tasks.
        Returns the proportion of each class among the k nearest neighbors.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on
            
        Returns:
        --------
        probabilities : numpy array of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only works for classification tasks")
        
        # Get unique classes from training data
        classes = np.unique(self.y_train)
        
        probabilities = []
        for x in X:
            # Calculate distances to all training samples
            distances = []
            for x_train in self.X_train:
                distance = self._calculate_distance(x, x_train)
                distances.append(distance)
            
            distances = np.array(distances)
            
            # Find k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Calculate probability for each class
            class_probs = []
            for c in classes:
                prob = np.sum(k_nearest_labels == c) / self.k
                class_probs.append(prob)
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)


"""
USAGE EXAMPLE 1: Simple Classification

import numpy as np

# Sample data: Predicting fruit type based on weight (g) and sweetness (1-10)
X_train = np.array([
    [150, 8],   # Apple
    [170, 9],   # Apple
    [140, 7],   # Apple
    [350, 4],   # Orange
    [380, 5],   # Orange
    [340, 3],   # Orange
    [200, 9],   # Strawberry
    [180, 10],  # Strawberry
    [190, 8]    # Strawberry
])

# Labels: 0 = Apple, 1 = Orange, 2 = Strawberry
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# Create and train the model
model = KNearestNeighbors(k=3, task='classification')
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [160, 8],   # Should be Apple
    [360, 4],   # Should be Orange
    [185, 9]    # Should be Strawberry
])

predictions = model.predict(X_test)
print("Predicted classes:", predictions)
# Output: [0, 1, 2] (Apple, Orange, Strawberry)

# Get class probabilities
probabilities = model.predict_proba(X_test)
print("\nPredicted probabilities:")
for i, probs in enumerate(probabilities):
    print(f"  Sample {i+1}: Apple={probs[0]:.2f}, Orange={probs[1]:.2f}, Strawberry={probs[2]:.2f}")
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Iris Classification)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the KNN model
model = KNearestNeighbors(k=5, task='classification')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Display predictions for first 5 test samples
print("\nFirst 5 predictions:")
for i in range(5):
    print(f"  Sample {i+1}: True={y_test[i]}, Predicted={y_pred[i]}")

# Get class probabilities
probabilities = model.predict_proba(X_test_scaled[:5])
print("\nProbabilities for first 5 samples:")
for i, probs in enumerate(probabilities):
    print(f"  Sample {i+1}: Setosa={probs[0]:.2f}, Versicolor={probs[1]:.2f}, Virginica={probs[2]:.2f}")
"""

"""
USAGE EXAMPLE 3: KNN for Regression

import numpy as np

# Sample data: Predicting house price based on size (sq ft) and age (years)
X_train = np.array([
    [1000, 5],   # $200k
    [1500, 3],   # $300k
    [1200, 10],  # $220k
    [2000, 2],   # $400k
    [1800, 7],   # $350k
    [2500, 1],   # $500k
    [900, 15],   # $180k
    [1100, 8],   # $210k
])

# Prices in thousands
y_train = np.array([200, 300, 220, 400, 350, 500, 180, 210])

# Create and train the model for regression
model = KNearestNeighbors(k=3, task='regression')
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [1300, 6],   # Similar to training examples
    [2200, 2],   # Larger, newer house
    [950, 12]    # Smaller, older house
])

predictions = model.predict(X_test)
print("Predicted prices ($1000s):", predictions)
# Output: Approximate values based on nearest neighbors

# Calculate R² score on training data
r2_score = model.score(X_train, y_train)
print(f"\nR² Score on training data: {r2_score:.4f}")
"""

"""
USAGE EXAMPLE 4: Comparing Different k Values

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different k values
k_values = [1, 3, 5, 7, 9, 11, 15, 21]

print("Comparing Different k Values:\n")
print(f"{'k':<10} {'Train Accuracy':<20} {'Test Accuracy':<20}")
print("-" * 50)

for k in k_values:
    model = KNearestNeighbors(k=k, task='classification')
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"{k:<10} {train_acc:<20.4f} {test_acc:<20.4f}")

# Observations:
# - Small k (1-3): High train accuracy, may overfit
# - Medium k (5-9): Good balance between bias and variance
# - Large k (15+): More robust, but may underfit
"""

"""
USAGE EXAMPLE 5: Comparing Distance Metrics

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load wine dataset
data = load_wine()
X, y = data.data, data.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare distance metrics
distance_metrics = ['euclidean', 'manhattan']

print("Comparing Distance Metrics:\n")
print(f"{'Distance Metric':<20} {'Train Accuracy':<20} {'Test Accuracy':<20}")
print("-" * 60)

for metric in distance_metrics:
    model = KNearestNeighbors(k=5, distance_metric=metric, task='classification')
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"{metric:<20} {train_acc:<20.4f} {test_acc:<20.4f}")

# Euclidean: Most common, works well in most cases
# Manhattan: Better when features have different scales or in high dimensions
"""
