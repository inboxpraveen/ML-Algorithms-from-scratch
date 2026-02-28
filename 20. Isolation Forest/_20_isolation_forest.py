import numpy as np

class IsolationForest:
    """
    Isolation Forest Implementation from Scratch
    
    Isolation Forest is an unsupervised anomaly detection algorithm that isolates
    anomalies instead of profiling normal data points. It's based on the principle
    that anomalies are few and different, thus easier to isolate than normal points.
    
    Key Idea: "Anomalies have shorter average path lengths in isolation trees"
    
    Use Cases:
    - Fraud Detection: Credit card fraud, insurance fraud
    - Intrusion Detection: Network security, cybersecurity
    - Medical Diagnosis: Disease outbreak detection
    - Manufacturing: Defect detection, quality control
    - System Monitoring: Server anomalies, unusual behavior
    
    Key Concepts:
        Isolation Trees: Random trees that partition data by random splits
        Path Length: Number of edges from root to leaf (shorter for anomalies)
        Anomaly Score: Normalized path length (0-1, higher = more anomalous)
        Contamination: Expected proportion of anomalies in dataset
        Subsampling: Use subset of data to build each tree
    """
    
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, 
                 max_features=1.0, random_state=None):
        """
        Initialize the Isolation Forest model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of isolation trees to build
            - More trees: More stable and accurate detection
            - Fewer trees: Faster training and prediction
            Typical values: 50-200
            
        max_samples : int, float or 'auto', default='auto'
            Number of samples to draw from X to train each tree
            - int: Use max_samples samples
            - float: Use max_samples * n_samples samples
            - 'auto': Use min(256, n_samples)
            Subsampling creates diversity and speeds up training
            Original paper recommends 256
            
        contamination : float, default=0.1
            Expected proportion of outliers in the dataset
            - Used to define threshold for anomaly scores
            - Range: (0, 0.5)
            - 0.1 means 10% of data expected to be anomalies
            Typical values: 0.01 (1%) to 0.1 (10%)
            
        max_features : int or float, default=1.0
            Number of features to draw from X to train each tree
            - int: Use max_features features
            - float: Use max_features * n_features features
            - 1.0 means use all features
            Reducing features increases diversity
            
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        
        self.trees = []
        self.threshold = None
        self.max_samples_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _calculate_c(self, n):
        """
        Calculate average path length of unsuccessful search in Binary Search Tree
        
        This is used to normalize path lengths. For a BST with n nodes:
        - Average path length ≈ 2H(n-1) - 2(n-1)/n
        - H(i) is the harmonic number ≈ ln(i) + 0.5772 (Euler's constant)
        
        Parameters:
        -----------
        n : int
            Number of samples
            
        Returns:
        --------
        c : float
            Average path length
        """
        if n <= 1:
            return 0
        elif n == 2:
            return 1
        else:
            # H(n-1) = ln(n-1) + Euler's constant
            return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n
    
    def _build_tree(self, X, height_limit, current_height=0):
        """
        Recursively build an isolation tree
        
        Algorithm:
        1. If termination criteria met, return leaf node
        2. Randomly select a feature
        3. Randomly select a split value between min and max of that feature
        4. Split data and recursively build left and right subtrees
        
        Termination criteria:
        - Reached height limit
        - Only one sample left
        - All samples have identical values
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to partition
        height_limit : int
            Maximum tree height
        current_height : int
            Current height in the tree
            
        Returns:
        --------
        tree : dict
            Tree node with structure:
            - 'type': 'leaf' or 'internal'
            - For leaf: 'size' (number of samples)
            - For internal: 'feature', 'split_value', 'left', 'right'
        """
        n_samples, n_features = X.shape
        
        # Termination criteria
        if current_height >= height_limit or n_samples <= 1:
            return {
                'type': 'leaf',
                'size': n_samples
            }
        
        # Check if all samples are identical
        if np.all(X == X[0]):
            return {
                'type': 'leaf',
                'size': n_samples
            }
        
        # Randomly select feature to split
        if self.max_features < n_features:
            # Select subset of features
            features = np.random.choice(n_features, self.max_features, replace=False)
            feature_idx = np.random.choice(features)
        else:
            feature_idx = np.random.randint(0, n_features)
        
        # Get min and max for selected feature
        feature_values = X[:, feature_idx]
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        
        # If all values are the same, create leaf
        if min_val == max_val:
            return {
                'type': 'leaf',
                'size': n_samples
            }
        
        # Randomly select split value between min and max
        split_value = np.random.uniform(min_val, max_val)
        
        # Split data
        left_mask = feature_values < split_value
        right_mask = ~left_mask
        
        # If split doesn't separate data, create leaf
        if not np.any(left_mask) or not np.any(right_mask):
            return {
                'type': 'leaf',
                'size': n_samples
            }
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(X[left_mask], height_limit, current_height + 1)
        right_tree = self._build_tree(X[right_mask], height_limit, current_height + 1)
        
        return {
            'type': 'internal',
            'feature': feature_idx,
            'split_value': split_value,
            'left': left_tree,
            'right': right_tree
        }
    
    def _path_length(self, x, tree, current_height=0):
        """
        Calculate path length for a single sample in a tree
        
        Path length is the number of edges traversed from root to leaf.
        For anomaly detection:
        - Shorter paths → More anomalous (easier to isolate)
        - Longer paths → More normal (harder to isolate)
        
        Parameters:
        -----------
        x : np.ndarray, shape (n_features,)
            Single sample
        tree : dict
            Isolation tree
        current_height : int
            Current height in the tree
            
        Returns:
        --------
        path_length : float
            Path length for this sample
        """
        if tree['type'] == 'leaf':
            # Adjust path length for leaf size
            # If leaf has multiple samples, add average path length estimate
            return current_height + self._calculate_c(tree['size'])
        
        # Navigate to left or right subtree
        if x[tree['feature']] < tree['split_value']:
            return self._path_length(x, tree['left'], current_height + 1)
        else:
            return self._path_length(x, tree['right'], current_height + 1)
    
    def _anomaly_score(self, path_lengths):
        """
        Calculate anomaly score from average path lengths
        
        Anomaly Score Formula:
        s(x, n) = 2^(-E(h(x)) / c(n))
        
        Where:
        - E(h(x)) is the average path length of x over all trees
        - c(n) is the average path length of unsuccessful search in BST
        
        Interpretation:
        - s → 1: Very likely anomaly (very short path)
        - s → 0.5: Neither anomaly nor normal (average path)
        - s → 0: Very likely normal (very long path)
        
        Parameters:
        -----------
        path_lengths : np.ndarray
            Average path lengths for samples
            
        Returns:
        --------
        scores : np.ndarray
            Anomaly scores (0 to 1)
        """
        # Normalize by expected path length
        c = self._calculate_c(self.max_samples_)
        
        # Calculate anomaly score
        scores = np.power(2, -path_lengths / c)
        
        return scores
    
    def fit(self, X, y=None):
        """
        Train the Isolation Forest model
        
        Algorithm:
        1. Determine max_samples for subsampling
        2. Calculate height limit for trees
        3. For each tree:
           a. Randomly subsample data
           b. Build isolation tree with random splits
        4. Calculate anomaly score threshold based on contamination
        
        Note: This is unsupervised learning, so y is ignored
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data
        y : ignored
            Not used, present for API consistency
            
        Returns:
        --------
        self : IsolationForest
            Fitted model
        """
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        
        # Determine max_samples
        if self.max_samples == 'auto':
            self.max_samples_ = min(256, n_samples)
        elif isinstance(self.max_samples, int):
            self.max_samples_ = min(self.max_samples, n_samples)
        else:  # float
            self.max_samples_ = int(self.max_samples * n_samples)
        
        # Determine max_features
        if isinstance(self.max_features, int):
            self.max_features = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            self.max_features = max(1, int(self.max_features * n_features))
        
        # Height limit (ceiling of log2(max_samples))
        height_limit = int(np.ceil(np.log2(self.max_samples_)))
        
        # Build trees
        self.trees = []
        for _ in range(self.n_estimators):
            # Randomly subsample data
            sample_indices = np.random.choice(n_samples, self.max_samples_, replace=False)
            X_sample = X[sample_indices]
            
            # Build tree
            tree = self._build_tree(X_sample, height_limit)
            self.trees.append(tree)
        
        # Calculate threshold based on contamination
        # Predict on training data
        scores = self.decision_function(X)
        
        # Threshold is the (1 - contamination) quantile
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        return self
    
    def decision_function(self, X):
        """
        Calculate anomaly scores for samples
        
        The anomaly score is the average path length across all trees,
        normalized and transformed to be in range [0, 1].
        
        Higher scores indicate more anomalous samples.
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to score
            
        Returns:
        --------
        scores : np.ndarray, shape (n_samples,)
            Anomaly scores (higher = more anomalous)
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        
        # Calculate average path length for each sample
        avg_path_lengths = np.zeros(n_samples)
        
        for i in range(n_samples):
            path_lengths = []
            for tree in self.trees:
                path_length = self._path_length(X[i], tree)
                path_lengths.append(path_length)
            
            avg_path_lengths[i] = np.mean(path_lengths)
        
        # Convert to anomaly scores
        scores = self._anomaly_score(avg_path_lengths)
        
        return scores
    
    def predict(self, X):
        """
        Predict if samples are anomalies or not
        
        Uses the threshold determined during fit() based on contamination.
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted labels:
            - 1: Normal (inlier)
            - -1: Anomaly (outlier)
        """
        if self.threshold is None:
            raise ValueError("Model must be fitted before prediction")
        
        scores = self.decision_function(X)
        
        # Anomaly if score > threshold
        predictions = np.where(scores > self.threshold, -1, 1)
        
        return predictions
    
    def score_samples(self, X):
        """
        Opposite of decision_function for sklearn compatibility
        
        Returns negative anomaly scores so that:
        - More negative = more anomalous
        - Less negative = more normal
        
        Parameters:
        -----------
        X : np.ndarray or list
            Data to score
            
        Returns:
        --------
        scores : np.ndarray
            Negative anomaly scores
        """
        return -self.decision_function(X)
    
    def score(self, X, y=None):
        """
        Calculate accuracy on labeled data
        
        This is only useful if you have labeled data (1 for normal, -1 for anomaly).
        For unsupervised anomaly detection, use decision_function() instead.
        
        Parameters:
        -----------
        X : np.ndarray
            Data to evaluate
        y : np.ndarray, optional
            True labels (1 for normal, -1 for anomaly)
            
        Returns:
        --------
        score : float
            Accuracy if y is provided, mean anomaly score otherwise
        """
        if y is None:
            # Return mean decision function value
            return np.mean(self.decision_function(X))
        else:
            y = np.array(y)
            predictions = self.predict(X)
            return np.mean(predictions == y)


"""
USAGE EXAMPLE 1: Simple Anomaly Detection

import numpy as np

# Generate normal data (clustered around origin)
np.random.seed(42)
X_normal = np.random.randn(300, 2) * 0.5

# Generate anomalies (far from origin)
X_anomalies = np.random.uniform(low=-4, high=4, size=(20, 2))

# Combine data
X = np.vstack([X_normal, X_anomalies])

# Create and train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.1,
    random_state=42
)
model.fit(X)

# Predict anomalies
predictions = model.predict(X)

# Get anomaly scores
scores = model.decision_function(X)

# Analyze results
n_anomalies = np.sum(predictions == -1)
n_normal = np.sum(predictions == 1)

print(f"Total samples: {len(X)}")
print(f"Detected anomalies: {n_anomalies}")
print(f"Detected normal: {n_normal}")

print("\nTop 5 most anomalous samples:")
top_anomalies = np.argsort(scores)[-5:][::-1]
for idx in top_anomalies:
    print(f"Sample {idx}: Score = {scores[idx]:.4f}, Prediction = {predictions[idx]}")
"""

"""
USAGE EXAMPLE 2: Credit Card Fraud Detection

import numpy as np

# Simulate credit card transactions
# [amount, time_since_last, num_transactions_today, merchant_category, location_distance]

np.random.seed(42)

# Normal transactions
n_normal = 950
normal_transactions = np.column_stack([
    np.random.gamma(2, 50, n_normal),           # Amount: $0-500
    np.random.exponential(60, n_normal),        # Time: ~1 hour between
    np.random.poisson(3, n_normal),             # 3 transactions/day avg
    np.random.randint(1, 10, n_normal),         # Merchant category
    np.random.gamma(2, 5, n_normal)             # Location: nearby
])

# Fraudulent transactions
n_fraud = 50
fraud_transactions = np.column_stack([
    np.random.uniform(500, 2000, n_fraud),      # High amount
    np.random.uniform(0, 5, n_fraud),           # Rapid succession
    np.random.randint(10, 30, n_fraud),         # Many transactions
    np.random.randint(1, 10, n_fraud),          # Random merchant
    np.random.uniform(50, 500, n_fraud)         # Far location
])

X = np.vstack([normal_transactions, fraud_transactions])
y_true = np.array([1] * n_normal + [-1] * n_fraud)

# Shuffle
indices = np.random.permutation(1000)
X = X[indices]
y_true = y_true[indices]

# Train Isolation Forest
model = IsolationForest(
    n_estimators=150,
    max_samples=256,
    contamination=0.05,  # Expect 5% fraud
    random_state=42
)
model.fit(X)

# Predict
predictions = model.predict(X)

# Evaluate
accuracy = np.mean(predictions == y_true)
true_positives = np.sum((predictions == -1) & (y_true == -1))
false_positives = np.sum((predictions == -1) & (y_true == 1))
false_negatives = np.sum((predictions == 1) & (y_true == -1))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print("Credit Card Fraud Detection:")
print("="*60)
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.4f}")
print(f"\nFraud detected: {np.sum(predictions == -1)}")
print(f"Actual fraud: {np.sum(y_true == -1)}")
"""

"""
USAGE EXAMPLE 3: Effect of Number of Trees

import numpy as np

# Generate data with anomalies
np.random.seed(42)
X_normal = np.random.randn(200, 3) * 0.5
X_anomalies = np.random.uniform(low=-3, high=3, size=(20, 3))
X = np.vstack([X_normal, X_anomalies])
y_true = np.array([1] * 200 + [-1] * 20)

# Test different numbers of trees
n_trees_list = [10, 25, 50, 100, 200, 300]

print("Effect of Number of Trees:")
print("="*70)
print(f"{'N Trees':>10} {'Accuracy':>12} {'Precision':>12} {'Recall':>12}")
print("-"*70)

for n_trees in n_trees_list:
    model = IsolationForest(
        n_estimators=n_trees,
        contamination=0.1,
        random_state=42
    )
    model.fit(X)
    predictions = model.predict(X)
    
    accuracy = np.mean(predictions == y_true)
    tp = np.sum((predictions == -1) & (y_true == -1))
    fp = np.sum((predictions == -1) & (y_true == 1))
    fn = np.sum((predictions == 1) & (y_true == -1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{n_trees:>10} {accuracy:>12.2%} {precision:>12.2%} {recall:>12.2%}")

# Observation: More trees generally lead to more stable results
"""

"""
USAGE EXAMPLE 4: Effect of Contamination Parameter

import numpy as np

# Generate data
np.random.seed(42)
X_normal = np.random.randn(180, 2)
X_anomalies = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X_normal, X_anomalies])
y_true = np.array([1] * 180 + [-1] * 20)

# Actual contamination is 10%
actual_contamination = 0.1

# Try different contamination parameters
contamination_values = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]

print("Effect of Contamination Parameter:")
print("="*80)
print(f"{'Contamination':>15} {'Detected':>10} {'Precision':>12} {'Recall':>12}")
print("-"*80)

for contam in contamination_values:
    model = IsolationForest(
        n_estimators=100,
        contamination=contam,
        random_state=42
    )
    model.fit(X)
    predictions = model.predict(X)
    
    detected = np.sum(predictions == -1)
    tp = np.sum((predictions == -1) & (y_true == -1))
    fp = np.sum((predictions == -1) & (y_true == 1))
    fn = np.sum((predictions == 1) & (y_true == -1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{contam:>15.2f} {detected:>10} {precision:>12.2%} {recall:>12.2%}")

print("\nNote: Contamination parameter controls how many samples are flagged")
print("Should match expected proportion of anomalies in your data")
"""

"""
USAGE EXAMPLE 5: Feature Importance via Permutation

import numpy as np

# Generate data with different feature importance
np.random.seed(42)
n_samples = 300

# Feature 1: Most important (anomalies differ significantly)
f1_normal = np.random.randn(n_samples) * 0.3
f1_anomaly = np.random.uniform(-3, 3, 30)

# Feature 2: Somewhat important
f2_normal = np.random.randn(n_samples) * 0.5
f2_anomaly = np.random.uniform(-2, 2, 30)

# Feature 3: Not important (noise)
f3_all = np.random.randn(n_samples + 30)

X = np.column_stack([
    np.concatenate([f1_normal, f1_anomaly]),
    np.concatenate([f2_normal, f2_anomaly]),
    f3_all
])

# Train model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# Get baseline anomaly scores
baseline_scores = model.decision_function(X)

# Calculate feature importance via permutation
feature_names = ['Feature 1', 'Feature 2', 'Feature 3 (Noise)']
n_features = X.shape[1]
importances = []

for feature_idx in range(n_features):
    # Permute feature
    X_permuted = X.copy()
    np.random.shuffle(X_permuted[:, feature_idx])
    
    # Calculate scores with permuted feature
    permuted_scores = model.decision_function(X_permuted)
    
    # Importance = change in mean absolute score
    importance = np.mean(np.abs(baseline_scores - permuted_scores))
    importances.append(importance)

# Normalize
importances = np.array(importances)
importances = importances / np.sum(importances)

print("Feature Importance (via Permutation):")
print("="*60)
for name, imp in zip(feature_names, importances):
    bar = '█' * int(imp * 50)
    print(f"{name:20s}: {imp:.4f} {bar}")
"""

"""
USAGE EXAMPLE 6: Network Intrusion Detection

import numpy as np

# Simulate network traffic data
# [packets_per_sec, bytes_per_packet, connection_duration, 
#  distinct_ports, protocol_type, error_rate]

np.random.seed(42)

# Normal traffic
n_normal = 900
normal_traffic = np.column_stack([
    np.random.poisson(50, n_normal),            # Moderate packet rate
    np.random.normal(500, 100, n_normal),       # Normal packet size
    np.random.exponential(30, n_normal),        # Normal duration
    np.random.poisson(3, n_normal),             # Few distinct ports
    np.random.randint(0, 3, n_normal),          # Standard protocols
    np.random.beta(2, 50, n_normal)             # Low error rate
])

# Attack traffic
n_attack = 100
attack_traffic = np.column_stack([
    np.random.poisson(500, n_attack),           # High packet rate
    np.random.normal(100, 50, n_attack),        # Small packets
    np.random.exponential(5, n_attack),         # Short duration
    np.random.poisson(50, n_attack),            # Many ports (scanning)
    np.random.randint(0, 10, n_attack),         # Unusual protocols
    np.random.beta(10, 20, n_attack)            # High error rate
])

X = np.vstack([normal_traffic, attack_traffic])
y_true = np.array([1] * n_normal + [-1] * n_attack)

# Shuffle
indices = np.random.permutation(1000)
X = X[indices]
y_true = y_true[indices]

# Split train/test
X_train, X_test = X[:700], X[300:]
y_train, y_test = y_true[:700], y_true[300:]

# Train Isolation Forest
model = IsolationForest(
    n_estimators=150,
    max_samples='auto',
    contamination=0.1,
    random_state=42
)
model.fit(X_train)

# Test
predictions = model.predict(X_test)
scores = model.decision_function(X_test)

# Evaluate
accuracy = np.mean(predictions == y_test)
tp = np.sum((predictions == -1) & (y_test == -1))
fp = np.sum((predictions == -1) & (y_test == 1))
fn = np.sum((predictions == 1) & (y_test == -1))

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Network Intrusion Detection:")
print("="*60)
print(f"Test Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall (Detection Rate): {recall:.2%}")
print(f"\nIntrusions detected: {np.sum(predictions == -1)}")
print(f"Actual intrusions: {np.sum(y_test == -1)}")

print("\nTop 5 Suspicious Connections:")
suspicious_idx = np.argsort(scores)[-5:][::-1]
for idx in suspicious_idx:
    print(f"Connection {idx}: Anomaly Score = {scores[idx]:.4f}")
"""

"""
USAGE EXAMPLE 7: Manufacturing Defect Detection

import numpy as np

# Simulate sensor readings from manufacturing process
# [temperature, pressure, vibration, speed, current]

np.random.seed(42)

# Normal products (tight tolerances)
n_normal = 450
normal_products = np.column_stack([
    np.random.normal(25, 2, n_normal),          # Temperature: 25±2°C
    np.random.normal(100, 5, n_normal),         # Pressure: 100±5 PSI
    np.random.normal(10, 1, n_normal),          # Vibration: 10±1 Hz
    np.random.normal(1500, 50, n_normal),       # Speed: 1500±50 RPM
    np.random.normal(15, 1, n_normal)           # Current: 15±1 A
])

# Defective products (out of spec)
n_defect = 50
defect_products = np.column_stack([
    np.random.normal(25, 5, n_defect),          # More temperature variation
    np.random.normal(100, 15, n_defect),        # More pressure variation
    np.random.normal(10, 3, n_defect),          # More vibration
    np.random.normal(1500, 150, n_defect),      # More speed variation
    np.random.normal(15, 3, n_defect)           # More current variation
])

X = np.vstack([normal_products, defect_products])
y_true = np.array([1] * n_normal + [-1] * n_defect)

# Shuffle
indices = np.random.permutation(500)
X = X[indices]
y_true = y_true[indices]

# Train model
model = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.1,
    random_state=42
)
model.fit(X)

# Predict
predictions = model.predict(X)
scores = model.decision_function(X)

# Evaluate
accuracy = np.mean(predictions == y_true)
tp = np.sum((predictions == -1) & (y_true == -1))
fp = np.sum((predictions == -1) & (y_true == 1))
fn = np.sum((predictions == 1) & (y_true == -1))

print("Manufacturing Defect Detection:")
print("="*60)
print(f"Accuracy: {accuracy:.2%}")
print(f"Defects found: {np.sum(predictions == -1)}")
print(f"Actual defects: {np.sum(y_true == -1)}")
print(f"False alarms: {fp}")
print(f"Missed defects: {fn}")

# Cost analysis
cost_false_alarm = 10  # Cost to inspect good product
cost_missed_defect = 1000  # Cost of shipping defect to customer

total_cost = fp * cost_false_alarm + fn * cost_missed_defect

print(f"\nCost Analysis:")
print(f"Cost of false alarms: ${fp * cost_false_alarm}")
print(f"Cost of missed defects: ${fn * cost_missed_defect}")
print(f"Total cost: ${total_cost}")
"""

"""
USAGE EXAMPLE 8: Comparing with Different max_samples

import numpy as np

# Generate data
np.random.seed(42)
X_normal = np.random.randn(300, 4)
X_anomalies = np.random.uniform(low=-4, high=4, size=(30, 4))
X = np.vstack([X_normal, X_anomalies])
y_true = np.array([1] * 300 + [-1] * 30)

# Test different max_samples
max_samples_list = [64, 128, 256, 'auto']

print("Effect of max_samples (Subsample Size):")
print("="*80)
print(f"{'max_samples':>15} {'Samples Used':>15} {'Accuracy':>12} {'Precision':>12}")
print("-"*80)

for max_samples in max_samples_list:
    model = IsolationForest(
        n_estimators=100,
        max_samples=max_samples,
        contamination=0.1,
        random_state=42
    )
    model.fit(X)
    
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y_true)
    
    tp = np.sum((predictions == -1) & (y_true == -1))
    fp = np.sum((predictions == -1) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    samples_used = model.max_samples_ if hasattr(model, 'max_samples_') else max_samples
    
    print(f"{str(max_samples):>15} {samples_used:>15} {accuracy:>12.2%} {precision:>12.2%}")

print("\nNote: Smaller subsamples train faster but may be less stable")
print("Original paper recommends 256 as good default")
"""

"""
USAGE EXAMPLE 9: Real-time Monitoring Dashboard Simulation

import numpy as np

# Simulate server metrics over time
# [cpu_usage, memory_usage, disk_io, network_traffic, response_time]

np.random.seed(42)

# Normal operation (95% of time)
n_normal = 950
normal_metrics = np.column_stack([
    np.random.normal(40, 10, n_normal),         # CPU: 40%
    np.random.normal(60, 10, n_normal),         # Memory: 60%
    np.random.normal(100, 20, n_normal),        # Disk I/O: 100 MB/s
    np.random.normal(500, 100, n_normal),       # Network: 500 Mb/s
    np.random.normal(50, 10, n_normal)          # Response: 50ms
])

# Anomalous behavior (5% of time - potential issues)
n_anomaly = 50
anomaly_metrics = np.column_stack([
    np.random.uniform(80, 100, n_anomaly),      # High CPU
    np.random.uniform(85, 98, n_anomaly),       # High Memory
    np.random.uniform(500, 1000, n_anomaly),    # High Disk I/O
    np.random.uniform(1000, 3000, n_anomaly),   # High Network
    np.random.uniform(200, 1000, n_anomaly)     # Slow Response
])

X = np.vstack([normal_metrics, anomaly_metrics])
timestamps = np.arange(1000)

# Shuffle with timestamps
indices = np.random.permutation(1000)
X = X[indices]
timestamps = timestamps[indices]

# Train model on first 500 samples
X_train = X[:500]
model = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.05,
    random_state=42
)
model.fit(X_train)

# Monitor all samples
predictions = model.predict(X)
scores = model.decision_function(X)

# Find anomalous periods
anomaly_mask = predictions == -1
anomaly_timestamps = timestamps[anomaly_mask]
anomaly_scores = scores[anomaly_mask]

print("Real-time Server Monitoring:")
print("="*70)
print(f"Total time periods monitored: {len(X)}")
print(f"Anomalies detected: {np.sum(anomaly_mask)}")
print(f"Alert rate: {np.mean(anomaly_mask):.1%}")

print("\nTop 10 Most Critical Anomalies:")
print("-"*70)
print(f"{'Time':>8} {'CPU%':>8} {'Mem%':>8} {'Disk':>8} {'Net':>8} {'Resp':>8} {'Score':>8}")

top_10 = np.argsort(scores)[-10:][::-1]
for idx in top_10:
    t = timestamps[idx]
    metrics = X[idx]
    score = scores[idx]
    print(f"{t:>8} {metrics[0]:>8.1f} {metrics[1]:>8.1f} {metrics[2]:>8.0f} "
          f"{metrics[3]:>8.0f} {metrics[4]:>8.0f} {score:>8.4f}")

print("\nRecommendation: Investigate time periods with high anomaly scores")
"""
