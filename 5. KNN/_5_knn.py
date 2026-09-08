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

    Use Cases:
    - Recommender systems: "customers similar to you also bought ..."
    - Medical diagnosis: match a patient against similar patient profiles
    - Handwriting / image recognition: nearest-template digit and object matching
    - Real-estate pricing: predict a price from comparable properties
    - Anomaly detection: flag points whose neighbors are unusually far away

    Distance Formulas Implemented (see _calculate_distance):
        d_euclidean(x, y) = sqrt(sum_i (x_i - y_i)^2)
        d_manhattan(x, y) = sum_i |x_i - y_i|

    Decision Rules Implemented (see _predict_single):
        Classification: y_hat = mode(y_(1), ..., y_(k))     # majority vote
        Regression:     y_hat = (1/k) * sum_j y_(j)         # mean of neighbors
        Probability:    P(class=c | x) = (# neighbors with class c) / k
        where y_(1), ..., y_(k) are the labels of the k nearest neighbors.

    With weights='distance' the votes/averages are weighted by w_j = 1 / d_j
    instead of counted uniformly (see the weights parameter below).

    Tie-Breaking:
        Two ties can occur and both are resolved deterministically.
        1. Vote tie (e.g. k=4 with labels [0, 0, 1, 1]): np.unique returns the
           classes in sorted order and np.argmax takes the first maximum, so the
           SMALLEST class label wins.
        2. Distance tie at the k-th neighbor: np.argsort is stable, so the
           training row that appears EARLIER in X_train is the one taken.
        Rule 1 matches scikit-learn on every backend. Rule 2 matches its
        kd_tree/ball_tree backends (what algorithm='auto' picks on small data);
        the brute backend does not guarantee which of two exactly-tied
        points it keeps, so it may pick the other one.

    Note: KNN compares raw distances, so features must be on a common scale.
    Standardize with x_scaled = (x - mean) / std before fitting real data.
    """
    
    def __init__(self, k=5, distance_metric='euclidean', task='classification',
                 weights='uniform'):
        """
        Initialize the K-Nearest Neighbors model

        Parameters:
        -----------
        k : int, default=5
            Number of nearest neighbors to use for prediction
            - Range: 1 to n_train (must not exceed the number of training samples;
              fit() raises a ValueError if it does)
            Larger k = smoother decision boundary, more robust to noise
            Smaller k = more flexible, captures local patterns
            Typical values: 3, 5, 7 (odd numbers avoid ties)
            Rule of thumb: start at k ~= sqrt(n_train), then tune by cross-validation

        distance_metric : str, default='euclidean'
            Distance metric to measure similarity
            Options: 'euclidean', 'manhattan'
            - Euclidean: Straight-line distance (most common)
            - Manhattan: City-block distance (sum of absolute differences)
            Typical: 'euclidean' unless the data is high-dimensional or has
            heavy-tailed features, where 'manhattan' is less outlier-sensitive

        task : str, default='classification'
            Type of prediction task
            Options: 'classification', 'regression'
            - classification: Predict categorical labels (score() returns accuracy)
            - regression: Predict continuous values (score() returns R^2)
            Typical: 'classification'; it also decides whether predict_proba works

        weights : str, default='uniform'
            How the k neighbors are combined into a prediction
            Options: 'uniform', 'distance'
            - 'uniform': every neighbor counts the same (the canonical KNN rule
              and scikit-learn's default)
            - 'distance': neighbor j is weighted by w_j = 1 / d_j, so closer
              neighbors pull the prediction harder. If a neighbor sits exactly on
              the query point (d_j = 0) it takes all the weight, which also
              avoids a division by zero.
            Typical: 'uniform'; try 'distance' when k is large or the class
            density varies a lot across the feature space
        """
        # --- Validate up front so mistakes are reported here, not deep in predict ---
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k}")
        if distance_metric not in ('euclidean', 'manhattan'):
            raise ValueError(
                f"Unknown distance metric: {distance_metric}. "
                "Expected 'euclidean' or 'manhattan'."
            )
        if task not in ('classification', 'regression'):
            raise ValueError(
                f"Unknown task: {task}. Expected 'classification' or 'regression'."
            )
        if weights not in ('uniform', 'distance'):
            raise ValueError(
                f"Unknown weights: {weights}. Expected 'uniform' or 'distance'."
            )

        self.k = k
        self.distance_metric = distance_metric
        self.task = task
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None

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
            Training data (a plain Python list of lists is accepted too).
            A 1-D array is treated as n_samples rows of a single feature.
        y : numpy array of shape (n_samples,)
            Target values (labels for classification, values for regression)

        Sets:
        -----
        classes_ : numpy array of shape (n_classes,)
            Sorted unique labels, for classification only. This is the column
            order used by predict_proba.
        """
        # Copy into a float array. Two reasons, both of which are real bugs otherwise:
        #   1. Integer inputs (e.g. uint8 pixels) would wrap around during
        #      (x1 - x2), turning far-apart points into near ones.
        #   2. np.array (unlike np.asarray) always copies, so later edits to the
        #      caller's array cannot silently change this model's predictions.
        self.X_train = np.array(X, dtype=float)
        # Do NOT force a dtype on y: string class labels are valid here.
        self.y_train = np.asarray(y)

        # Accept a 1-D X as "n samples, 1 feature" so single-feature data just works
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        if len(self.X_train) != len(self.y_train):
            raise ValueError(
                f"X and y have different lengths: {len(self.X_train)} vs "
                f"{len(self.y_train)}"
            )

        # k neighbors cannot be drawn from fewer than k stored samples
        if self.k > len(self.X_train):
            raise ValueError(
                f"Expected k <= n_samples_fit, but k = {self.k}, "
                f"n_samples_fit = {len(self.X_train)}"
            )

        # Column order for predict_proba, exposed so callers need not guess it
        if self.task == 'classification':
            self.classes_ = np.unique(self.y_train)

    def _check_is_fitted(self):
        """Raise a clear error if predict/score is called before fit"""
        if self.X_train is None:
            raise ValueError(
                "This KNearestNeighbors instance is not fitted yet. "
                "Call fit(X, y) before using predict/predict_proba/score."
            )

    def _prepare_input(self, X):
        """
        Coerce prediction input to a 2-D float array matching the training data

        A single sample passed as a flat array ([1.5, 2.0]) or single-feature
        data passed as a flat array is reshaped to (n_samples, n_features).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, self.X_train.shape[1])
        return X

    def _get_neighbors(self, x):
        """
        Find the k nearest training samples to x

        Implements steps 2 and 3 of the algorithm: measure the distance to every
        stored sample, then keep the k smallest.

        Parameters:
        -----------
        x : numpy array of shape (n_features,)
            Single query point

        Returns:
        --------
        k_indices : numpy array of shape (k,)
            Row indices of the k nearest training samples, nearest first
        k_distances : numpy array of shape (k,)
            Their distances to x, in the same order
        """
        # Calculate distances to all training samples
        distances = []
        for x_train in self.X_train:
            distance = self._calculate_distance(x, x_train)
            distances.append(distance)

        # Convert to numpy array for easier manipulation
        distances = np.array(distances)

        # Find indices of k nearest neighbors.
        # Full sort, O(n log n); np.argpartition would be O(n) but argsort reads
        # more clearly. argsort is stable, so a distance tie is broken in favour
        # of the training row that appears earlier in X_train.
        k_indices = np.argsort(distances)[:self.k]

        return k_indices, distances[k_indices]

    def _neighbor_weights(self, k_distances):
        """
        Turn neighbor distances into voting weights

        weights='uniform'  -> w_j = 1                    (every neighbor equal)
        weights='distance' -> w_j = 1 / d_j              (closer counts more)

        Special case for 'distance': if any neighbor sits exactly on the query
        point (d_j = 0), it gets all the weight and the rest get zero. That is
        both the sensible answer (the query IS that training point) and what
        avoids 1/0. This matches scikit-learn's behaviour.
        """
        if self.weights == 'uniform':
            return np.ones(len(k_distances))

        zero_mask = (k_distances == 0)
        if np.any(zero_mask):
            return zero_mask.astype(float)
        return 1.0 / k_distances

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
        # Steps 2-3: distances to every training sample, then the k smallest
        k_indices, k_distances = self._get_neighbors(x)

        # Get labels/values of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # How much each neighbor's vote counts (all 1.0 when weights='uniform')
        w = self._neighbor_weights(k_distances)

        # Make prediction based on task type
        if self.task == 'classification':
            # Classification: Return most common class (mode).
            # np.unique returns classes in sorted order and np.argmax takes the
            # FIRST maximum, so a vote tie resolves to the smallest class label.
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            if self.weights == 'distance':
                # Weighted vote: sum the weights of the neighbors in each class
                counts = np.array([w[k_nearest_labels == c].sum()
                                   for c in unique_labels])
            prediction = unique_labels[np.argmax(counts)]
        else:
            # Regression: Return average value (mean).
            # np.average with all-ones weights is exactly np.mean; with
            # weights='distance' it is sum(w_j * y_j) / sum(w_j).
            prediction = np.average(k_nearest_labels, weights=w)

        return prediction
    
    def predict(self, X):
        """
        Predict labels or values for samples
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on (lists and 1-D arrays are accepted)

        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted labels (classification) or values (regression)
        """
        self._check_is_fitted()
        X = self._prepare_input(X)

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
            Accuracy in [0, 1] (classification), or the R^2 score (regression).
            R^2 is at most 1.0 but has no lower bound: it goes negative when the
            model predicts worse than the mean of y.
        """
        predictions = self.predict(X)
        y = np.asarray(y)

        if self.task == 'classification':
            # Classification: Calculate accuracy
            accuracy = np.mean(predictions == y)
            return accuracy
        else:
            # Regression: Calculate R^2 score
            # R^2 = 1 - (SS_res / SS_tot)
            # where SS_res = sum of squared residuals
            #       SS_tot = total sum of squares
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if ss_tot == 0:
                # Constant y: variance is zero, so the ratio is undefined.
                # Use scikit-learn's r2_score convention instead of nan.
                return 1.0 if ss_res == 0 else 0.0
            r2_score = 1 - (ss_res / ss_tot)
            return r2_score
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification tasks
        
        Only works for classification tasks.
        Returns the proportion of each class among the k nearest neighbors:

            P(class=c | x) = (# of the k neighbors whose label is c) / k

        With weights='distance' the counts become sums of w_j = 1 / d_j,
        normalised by the total weight so each row still sums to 1.

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on (lists and 1-D arrays are accepted)

        Returns:
        --------
        probabilities : numpy array of shape (n_samples, n_classes)
            Predicted probabilities for each class. Column j corresponds to
            self.classes_[j], i.e. the sorted unique training labels. Each row
            sums to 1.
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only works for classification tasks")

        self._check_is_fitted()
        X = self._prepare_input(X)

        # Classes in the same sorted order fit() recorded, so callers can map
        # column j back to a label via self.classes_[j]
        classes = self.classes_

        probabilities = []
        for x in X:
            # Same neighbor search predict() uses - one implementation, so the
            # two paths can never disagree
            k_indices, k_distances = self._get_neighbors(x)
            k_nearest_labels = self.y_train[k_indices]
            w = self._neighbor_weights(k_distances)

            # Calculate probability for each class.
            # Divide by the total weight actually collected, not by self.k, so
            # the row always sums to exactly 1.
            total_weight = np.sum(w)
            class_probs = []
            for c in classes:
                prob = np.sum(w[k_nearest_labels == c]) / total_weight
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

# Calculate R^2 score on training data
r2_score = model.score(X_train, y_train)
print(f"\nR^2 Score on training data: {r2_score:.4f}")

# NOTE: this R^2 is measured on the same 8 rows the model memorised, so it is
# optimistic by construction - a lazy learner has every training answer on hand.
# It tells you the neighborhoods are consistent, NOT how the model generalises.
# With only 8 samples there is no room for a held-out split; USAGE EXAMPLE 4
# shows the train-vs-test comparison you should always run on real data.
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
k_values = [1, 3, 5, 11, 21]

print("Comparing Different k Values:\n")
print(f"{'k':<10} {'Train Accuracy':<20} {'Test Accuracy':<20}")
print("-" * 50)

# Every prediction scans all 455 training rows, so scoring the full training set
# for every k is the slow part. Score train accuracy on a fixed 150-row
# subsample instead: same story about overfitting, a few seconds instead of a minute.
train_subset = 150

for k in k_values:
    model = KNearestNeighbors(k=k, task='classification')
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled[:train_subset], y_train[:train_subset])
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


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _5_knn.py
    # numpy only, seeded, finishes in a couple of seconds.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Classification demo: three Gaussian blobs ---
    print("=" * 55)
    print("DEMO 1 - Classification: three Gaussian blobs")
    print("=" * 55)

    # 60 points around each of three centers. Both features are already on the
    # same scale here; on real data you MUST standardize first, because KNN
    # compares raw distances and a large-range feature would drown out the rest.
    centers = np.array([[0.0, 0.0], [4.0, 4.0], [8.0, 0.0]])
    n_per_class = 60
    spread = 1.8  # wide enough that the blobs overlap, so the k trade-off shows
    X_cls = np.vstack([np.random.randn(n_per_class, 2) * spread + c
                       for c in centers])
    y_cls = np.repeat([0, 1, 2], n_per_class)

    # Shuffle BEFORE slicing, otherwise the test set would be one whole class
    idx = np.random.permutation(len(X_cls))
    X_cls, y_cls = X_cls[idx], y_cls[idx]
    X_tr, X_te = X_cls[:135], X_cls[135:]
    y_tr, y_te = y_cls[:135], y_cls[135:]

    clf = KNearestNeighbors(k=5, distance_metric='euclidean', task='classification')
    clf.fit(X_tr, y_tr)

    print(f"Train accuracy : {clf.score(X_tr, y_tr):.4f}")
    print(f"Test  accuracy : {clf.score(X_te, y_te):.4f}")

    probas = clf.predict_proba(X_te)
    preds = clf.predict(X_te)
    print(f"\nprobability columns map to classes_ = {clf.classes_}")
    print("Sample predictions (true, pred, class probabilities):")
    for i in range(5):
        print(f"  true={y_te[i]}  pred={preds[i]}  "
              f"P0={probas[i, 0]:.2f}  P1={probas[i, 1]:.2f}  P2={probas[i, 2]:.2f}")

    # --- Regression demo: y = 3*sin(x) + noise ---
    print("\n" + "=" * 55)
    print("DEMO 2 - Regression: y = 3*sin(x) + noise")
    print("=" * 55)

    # Drawn uniformly (not with linspace), so a plain slice still gives train
    # and test sets that cover the same x range
    X_reg = np.random.uniform(-3, 3, size=(200, 1))
    y_reg = 3 * np.sin(X_reg.ravel()) + np.random.randn(200) * 0.3
    X_rtr, X_rte = X_reg[:150], X_reg[150:]
    y_rtr, y_rte = y_reg[:150], y_reg[150:]

    reg = KNearestNeighbors(k=5, task='regression')
    reg.fit(X_rtr, y_rtr)

    print(f"Train R2 : {reg.score(X_rtr, y_rtr):.4f}")
    print(f"Test  R2 : {reg.score(X_rte, y_rte):.4f}")

    reg_preds = reg.predict(X_rte)
    print("\nSample predictions (x, true, predicted):")
    for i in range(5):
        print(f"  x={X_rte[i, 0]:5.2f}  true={y_rte[i]:6.2f}  pred={reg_preds[i]:6.2f}")

    # --- What k, the metric and the weighting actually do ---
    print("\n" + "=" * 55)
    print("DEMO 3 - Effect of k, distance metric and weighting")
    print("=" * 55)
    print(f"{'metric':<12}{'weights':<12}{'k':<5}{'train acc':<12}{'test acc':<12}")
    print("-" * 53)

    for metric in ['euclidean', 'manhattan']:
        for weighting in ['uniform', 'distance']:
            for k in [1, 5, 25]:
                m = KNearestNeighbors(k=k, distance_metric=metric,
                                      task='classification', weights=weighting)
                m.fit(X_tr, y_tr)
                print(f"{metric:<12}{weighting:<12}{k:<5}"
                      f"{m.score(X_tr, y_tr):<12.4f}{m.score(X_te, y_te):<12.4f}")

    print("\nWhat to notice:")
    print("  - k=1 scores 1.0000 on train because every point is its own nearest")
    print("    neighbor. That is memorisation, not skill - watch the test column.")
    print("  - Larger k smooths the boundary: more bias, less variance.")
    print("  - weights='distance' also scores 1.0000 on train for any k: a training")
    print("    point sits at distance 0 from itself and takes all the weight.")
    print("  - fit() is O(1) (it just stores the data); every prediction then costs")
    print("    O(n_train * n_features) distance work plus an O(n_train log n) sort.")
