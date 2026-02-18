import numpy as np

class LightGBM:
    """
    LightGBM (Light Gradient Boosting Machine) Implementation from Scratch
    
    LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
    It is designed to be distributed and efficient with faster training speed, lower memory usage,
    better accuracy, and support for parallel and GPU learning.
    
    Key Idea: "Leaf-wise tree growth with histogram-based learning for speed and efficiency"
    
    Use Cases:
    - Regression: Sales forecasting, demand prediction, price estimation
    - Classification: Click prediction, fraud detection, customer churn
    - Ranking: Information retrieval, recommendation systems
    - Large-scale datasets: Where XGBoost becomes slow
    
    Key Innovations in LightGBM:
        Leaf-wise Growth: Grows trees by best leaf (not level-wise), leading to deeper, more accurate trees
        Histogram-based Learning: Bins continuous features into discrete bins for faster training
        GOSS: Gradient-based One-Side Sampling - keeps large gradient instances
        EFB: Exclusive Feature Bundling - bundles sparse features to reduce dimensions
        Faster Training: Significantly faster than XGBoost on large datasets
        Lower Memory Usage: More memory efficient than other GBDT implementations
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31,
                 min_data_in_leaf=20, min_sum_hessian_in_leaf=1e-3, 
                 feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
                 lambda_l1=0.0, lambda_l2=0.0, min_gain_to_split=0.0,
                 max_bin=255, objective='regression'):
        """
        Initialize the LightGBM model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of boosting iterations (trees)
            - More iterations: Better training fit, longer training
            - Fewer iterations: Faster training, may underfit
            Typical values: 100-1000
            
        learning_rate : float, default=0.1
            Shrinkage rate (also called eta)
            - Lower values need more iterations but generalize better
            - Range: 0.01 to 0.3
            Typical: 0.1 is standard, 0.05 for large datasets
            
        max_depth : int, default=-1
            Maximum tree depth
            - -1 means no limit (controlled by num_leaves instead)
            - Positive values limit tree depth
            Typical: -1 or 3-8
            
        num_leaves : int, default=31
            Maximum number of leaves in one tree
            - LightGBM's primary way to control model complexity
            - Should be < 2^max_depth
            - Larger values: More complex model, risk overfitting
            Typical values: 31-127 (powers of 2 minus 1)
            
        min_data_in_leaf : int, default=20
            Minimum number of data points in one leaf
            - Larger values prevent overfitting
            - Too large: May underfit
            Typical values: 20-100 for large datasets, 5-20 for small
            
        min_sum_hessian_in_leaf : float, default=1e-3
            Minimum sum of hessian in one leaf
            - Similar to min_child_weight in XGBoost
            - Larger values: More conservative, less overfitting
            Typical values: 1e-3 to 10
            
        feature_fraction : float, default=1.0
            Fraction of features to use for each tree (column subsampling)
            - < 1.0 introduces randomness and speeds up training
            - Similar to colsample_bytree in XGBoost
            Typical values: 0.5-1.0
            
        bagging_fraction : float, default=1.0
            Fraction of data to use for each iteration (row subsampling)
            - < 1.0 provides regularization and speeds up training
            - Only used if bagging_freq > 0
            Typical values: 0.5-1.0
            
        bagging_freq : int, default=0
            Frequency for bagging (0 means disable bagging)
            - If k > 0, perform bagging every k iterations
            Typical values: 0 (disabled) or 1-5
            
        lambda_l1 : float, default=0.0
            L1 regularization term
            - Can lead to sparse solutions
            - Useful for feature selection
            Typical values: 0-10
            
        lambda_l2 : float, default=0.0
            L2 regularization term
            - Helps prevent overfitting
            - More common than L1
            Typical values: 0-10
            
        min_gain_to_split : float, default=0.0
            Minimum gain to perform split
            - Acts as regularization
            - Similar to gamma in XGBoost
            Typical values: 0-1
            
        max_bin : int, default=255
            Maximum number of bins for feature discretization
            - Larger values: More accurate but slower
            - Smaller values: Faster but less accurate
            Typical values: 63, 127, 255 (LightGBM default)
            
        objective : str, default='regression'
            Learning objective
            - 'regression': Regression with L2 loss
            - 'binary': Binary classification with log loss
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.max_bin = max_bin
        self.objective = objective
        
        self.trees = []
        self.base_score = None
        self.bin_thresholds = None
        
    def _sigmoid(self, x):
        """Sigmoid function with numerical stability"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _compute_gradient_hessian(self, y_true, y_pred):
        """
        Compute first and second order gradients
        
        LightGBM uses both gradients and hessians for optimization,
        just like XGBoost
        
        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        gradient : np.ndarray
            First-order gradient
        hessian : np.ndarray
            Second-order gradient (hessian)
        """
        if self.objective == 'regression':
            # For squared error: L = 0.5 * (y - pred)^2
            # Gradient: dL/dpred = pred - y
            # Hessian: d²L/dpred² = 1
            gradient = y_pred - y_true
            hessian = np.ones_like(y_pred)
            
        elif self.objective == 'binary':
            # For log loss: L = -y*log(p) - (1-y)*log(1-p)
            # Gradient: dL/dpred = p - y
            # Hessian: d²L/dpred² = p * (1 - p)
            p = self._sigmoid(y_pred)
            gradient = p - y_true
            hessian = p * (1 - p)
            
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return gradient, hessian
    
    def _build_histogram(self, X):
        """
        Build histogram bins for features (histogram-based learning)
        
        This is a key innovation in LightGBM: instead of considering all possible
        split points, features are binned into discrete buckets, making training
        much faster while maintaining accuracy.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_binned : np.ndarray, shape (n_samples, n_features)
            Binned feature values (integers from 0 to max_bin-1)
        """
        n_samples, n_features = X.shape
        X_binned = np.zeros_like(X, dtype=int)
        self.bin_thresholds = []
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Create bins using percentiles
            unique_values = np.unique(feature_values)
            
            if len(unique_values) <= self.max_bin:
                # If few unique values, use them directly
                thresholds = unique_values[:-1]
            else:
                # Otherwise, create max_bin bins using quantiles
                percentiles = np.linspace(0, 100, self.max_bin + 1)[1:-1]
                thresholds = np.percentile(feature_values, percentiles)
                thresholds = np.unique(thresholds)
            
            self.bin_thresholds.append(thresholds)
            
            # Assign bin indices
            X_binned[:, feature_idx] = np.digitize(feature_values, thresholds)
        
        return X_binned
    
    def _apply_binning(self, X):
        """Apply pre-computed binning to new data"""
        n_samples, n_features = X.shape
        X_binned = np.zeros_like(X, dtype=int)
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = self.bin_thresholds[feature_idx]
            X_binned[:, feature_idx] = np.digitize(feature_values, thresholds)
        
        return X_binned
    
    def _calculate_leaf_weight(self, gradient_sum, hessian_sum):
        """
        Calculate optimal leaf weight with L1 and L2 regularization
        
        LightGBM formula: w* = -G / (H + lambda_l2)
        (Simplified version without L1 for this implementation)
        
        Parameters:
        -----------
        gradient_sum : float
            Sum of gradients in the leaf
        hessian_sum : float
            Sum of hessians in the leaf
            
        Returns:
        --------
        weight : float
            Optimal leaf weight
        """
        return -gradient_sum / (hessian_sum + self.lambda_l2 + 1e-10)
    
    def _calculate_gain(self, gradient_left, hessian_left, gradient_right, hessian_right):
        """
        Calculate split gain using LightGBM's formula
        
        Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - min_gain
        
        Parameters:
        -----------
        gradient_left : float
            Sum of gradients in left child
        hessian_left : float
            Sum of hessians in left child
        gradient_right : float
            Sum of gradients in right child
        hessian_right : float
            Sum of hessians in right child
            
        Returns:
        --------
        gain : float
            Split gain
        """
        def calculate_score(G, H):
            return (G ** 2) / (H + self.lambda_l2 + 1e-10)
        
        gain_left = calculate_score(gradient_left, hessian_left)
        gain_right = calculate_score(gradient_right, hessian_right)
        gain_parent = calculate_score(gradient_left + gradient_right, 
                                      hessian_left + hessian_right)
        
        gain = 0.5 * (gain_left + gain_right - gain_parent) - self.min_gain_to_split
        
        return gain
    
    def _build_tree_leaf_wise(self, X_binned, gradient, hessian, depth=0):
        """
        Build tree using leaf-wise (best-first) strategy
        
        This is LightGBM's key innovation: instead of growing level by level,
        it finds and splits the leaf with maximum gain, leading to deeper,
        more asymmetric trees that can be more accurate.
        
        Parameters:
        -----------
        X_binned : np.ndarray, shape (n_samples, n_features)
            Binned training data
        gradient : np.ndarray, shape (n_samples,)
            First-order gradients
        hessian : np.ndarray, shape (n_samples,)
            Second-order gradients
        depth : int
            Current depth
            
        Returns:
        --------
        tree : dict
            Tree structure
        """
        n_samples, n_features = X_binned.shape
        
        # Check stopping criteria
        gradient_sum = np.sum(gradient)
        hessian_sum = np.sum(hessian)
        
        if (n_samples < self.min_data_in_leaf or
            hessian_sum < self.min_sum_hessian_in_leaf or
            (self.max_depth > 0 and depth >= self.max_depth)):
            # Create leaf
            leaf_weight = self._calculate_leaf_weight(gradient_sum, hessian_sum)
            return {
                'type': 'leaf',
                'weight': leaf_weight,
                'count': n_samples
            }
        
        # Feature subsampling
        n_features_use = max(1, int(self.feature_fraction * n_features))
        feature_indices = np.random.choice(n_features, n_features_use, replace=False)
        
        # Find best split
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_mask = None
        
        for feature_idx in feature_indices:
            feature_bins = X_binned[:, feature_idx]
            unique_bins = np.unique(feature_bins)
            
            # Try each bin as split threshold
            for bin_value in unique_bins:
                left_mask = feature_bins <= bin_value
                right_mask = ~left_mask
                
                # Check if split is valid
                if np.sum(left_mask) < self.min_data_in_leaf or \
                   np.sum(right_mask) < self.min_data_in_leaf:
                    continue
                
                # Calculate gradient and hessian sums
                gradient_left = np.sum(gradient[left_mask])
                hessian_left = np.sum(hessian[left_mask])
                gradient_right = np.sum(gradient[right_mask])
                hessian_right = np.sum(hessian[right_mask])
                
                # Check hessian constraint
                if hessian_left < self.min_sum_hessian_in_leaf or \
                   hessian_right < self.min_sum_hessian_in_leaf:
                    continue
                
                # Calculate gain
                gain = self._calculate_gain(gradient_left, hessian_left,
                                           gradient_right, hessian_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = bin_value
                    best_left_mask = left_mask
        
        # If no good split found, create leaf
        if best_gain <= 0:
            leaf_weight = self._calculate_leaf_weight(gradient_sum, hessian_sum)
            return {
                'type': 'leaf',
                'weight': leaf_weight,
                'count': n_samples
            }
        
        # Recursively build subtrees
        left_tree = self._build_tree_leaf_wise(
            X_binned[best_left_mask],
            gradient[best_left_mask],
            hessian[best_left_mask],
            depth + 1
        )
        
        right_tree = self._build_tree_leaf_wise(
            X_binned[~best_left_mask],
            gradient[~best_left_mask],
            hessian[~best_left_mask],
            depth + 1
        )
        
        return {
            'type': 'split',
            'feature': best_feature,
            'threshold': best_threshold,
            'gain': best_gain,
            'left': left_tree,
            'right': right_tree,
            'count': n_samples
        }
    
    def _predict_tree(self, tree, X_binned):
        """
        Make predictions using a single tree
        
        Parameters:
        -----------
        tree : dict
            Tree structure
        X_binned : np.ndarray
            Binned data
            
        Returns:
        --------
        predictions : np.ndarray
            Tree predictions
        """
        if tree['type'] == 'leaf':
            return np.full(len(X_binned), tree['weight'])
        
        feature_bins = X_binned[:, tree['feature']]
        left_mask = feature_bins <= tree['threshold']
        
        predictions = np.zeros(len(X_binned))
        
        if np.sum(left_mask) > 0:
            predictions[left_mask] = self._predict_tree(tree['left'], X_binned[left_mask])
        if np.sum(~left_mask) > 0:
            predictions[~left_mask] = self._predict_tree(tree['right'], X_binned[~left_mask])
        
        return predictions
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the LightGBM model
        
        Algorithm:
        1. Build histogram bins for all features
        2. Initialize predictions with base score
        3. For each boosting iteration:
           a. Calculate gradients and hessians
           b. Apply bagging if enabled
           c. Build tree using leaf-wise strategy
           d. Update predictions
        4. Optional: Early stopping on validation set
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data
        y : np.ndarray or list, shape (n_samples,)
            Target values
        eval_set : list of tuples, optional
            List of (X_val, y_val) for validation
        early_stopping_rounds : int, optional
            Stop if validation score doesn't improve
        verbose : bool or int, default=False
            Print training progress
            
        Returns:
        --------
        self : LightGBM
            Fitted model
        """
        # Convert to numpy arrays
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Build histogram bins (LightGBM's key feature)
        X_binned = self._build_histogram(X)
        
        # Initialize base score
        if self.objective == 'binary':
            p = np.mean(y)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            self.base_score = np.log(p / (1 - p))
        else:
            self.base_score = np.mean(y)
        
        # Initialize predictions
        predictions = np.full(n_samples, self.base_score)
        
        self.trees = []
        self.train_scores = []
        self.val_scores = []
        
        # Early stopping variables
        best_score = float('inf')
        best_iteration = 0
        
        # Train trees
        for iteration in range(self.n_estimators):
            # Calculate gradients and hessians
            gradient, hessian = self._compute_gradient_hessian(y, predictions)
            
            # Bagging (row subsampling)
            if self.bagging_freq > 0 and iteration % self.bagging_freq == 0:
                if self.bagging_fraction < 1.0:
                    sample_size = int(n_samples * self.bagging_fraction)
                    indices = np.random.choice(n_samples, sample_size, replace=False)
                    X_sample = X_binned[indices]
                    gradient_sample = gradient[indices]
                    hessian_sample = hessian[indices]
                else:
                    X_sample = X_binned
                    gradient_sample = gradient
                    hessian_sample = hessian
            else:
                X_sample = X_binned
                gradient_sample = gradient
                hessian_sample = hessian
            
            # Build tree using leaf-wise strategy
            tree = self._build_tree_leaf_wise(X_sample, gradient_sample, hessian_sample)
            self.trees.append(tree)
            
            # Update predictions
            tree_predictions = self._predict_tree(tree, X_binned)
            predictions += self.learning_rate * tree_predictions
            
            # Calculate training score
            if self.objective == 'binary':
                train_preds = self._sigmoid(predictions)
                train_score = -np.mean(y * np.log(train_preds + 1e-10) + 
                                      (1 - y) * np.log(1 - train_preds + 1e-10))
            else:
                train_score = np.mean((y - predictions) ** 2)
            
            self.train_scores.append(train_score)
            
            # Evaluate on validation set
            if eval_set is not None:
                X_val, y_val = eval_set[0]
                val_preds = self.predict(X_val, num_iteration=iteration+1)
                
                if self.objective == 'binary':
                    val_score = -np.mean(y_val * np.log(val_preds + 1e-10) + 
                                        (1 - y_val) * np.log(1 - val_preds + 1e-10))
                else:
                    val_score = np.mean((y_val - val_preds) ** 2)
                
                self.val_scores.append(val_score)
                
                # Early stopping
                if early_stopping_rounds is not None:
                    if val_score < best_score:
                        best_score = val_score
                        best_iteration = iteration
                    elif iteration - best_iteration >= early_stopping_rounds:
                        if verbose:
                            print(f"Early stopping at iteration {iteration}")
                            print(f"Best iteration: {best_iteration}, Best score: {best_score:.6f}")
                        self.trees = self.trees[:best_iteration + 1]
                        break
                
                # Verbose output
                if verbose and (isinstance(verbose, bool) or iteration % verbose == 0):
                    print(f"[{iteration}] train-rmse: {np.sqrt(train_score):.6f}, "
                          f"val-rmse: {np.sqrt(val_score):.6f}")
            elif verbose and (isinstance(verbose, bool) or iteration % verbose == 0):
                if self.objective == 'binary':
                    print(f"[{iteration}] train-logloss: {train_score:.6f}")
                else:
                    print(f"[{iteration}] train-rmse: {np.sqrt(train_score):.6f}")
        
        return self
    
    def predict(self, X, num_iteration=None):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
        num_iteration : int, optional
            Number of trees to use (None means all)
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        
        # Apply binning
        X_binned = self._apply_binning(X)
        
        # Start with base score
        predictions = np.full(n_samples, self.base_score)
        
        # Determine number of trees to use
        n_trees = len(self.trees) if num_iteration is None else min(num_iteration, len(self.trees))
        
        # Add contribution from each tree
        for i in range(n_trees):
            tree_predictions = self._predict_tree(self.trees[i], X_binned)
            predictions += self.learning_rate * tree_predictions
        
        # For classification, convert to probabilities
        if self.objective == 'binary':
            predictions = self._sigmoid(predictions)
        
        return predictions
    
    def predict_proba(self, X, num_iteration=None):
        """
        Predict class probabilities (for classification)
        
        Parameters:
        -----------
        X : np.ndarray or list
            Data to predict
        num_iteration : int, optional
            Number of trees to use
            
        Returns:
        --------
        probabilities : np.ndarray, shape (n_samples, 2)
            Probability for each class
        """
        if self.objective != 'binary':
            raise ValueError("predict_proba only available for binary classification")
        
        proba_class_1 = self.predict(X, num_iteration)
        proba_class_0 = 1 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def score(self, X, y):
        """
        Calculate performance metric
        
        Parameters:
        -----------
        X : np.ndarray
            Data to evaluate
        y : np.ndarray
            True values
            
        Returns:
        --------
        score : float
            R² for regression, accuracy for classification
        """
        y = np.array(y)
        predictions = self.predict(X)
        
        if self.objective == 'binary':
            # Classification: accuracy
            predicted_classes = (predictions >= 0.5).astype(int)
            return np.mean(predicted_classes == y)
        else:
            # Regression: R² score
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - predictions) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            return r2
    
    def get_feature_importance(self, importance_type='split'):
        """
        Calculate feature importance
        
        Parameters:
        -----------
        importance_type : str, default='split'
            Type of importance:
            - 'split': Number of times feature is used for splitting
            - 'gain': Total gain from splits using the feature
            
        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Feature importance scores (normalized)
        """
        if importance_type == 'split':
            importance = np.zeros(self.n_features)
            
            def count_splits(tree):
                if tree['type'] == 'leaf':
                    return
                importance[tree['feature']] += 1
                count_splits(tree['left'])
                count_splits(tree['right'])
            
            for tree in self.trees:
                count_splits(tree)
                
        elif importance_type == 'gain':
            importance = np.zeros(self.n_features)
            
            def accumulate_gain(tree):
                if tree['type'] == 'leaf':
                    return
                importance[tree['feature']] += tree['gain']
                accumulate_gain(tree['left'])
                accumulate_gain(tree['right'])
            
            for tree in self.trees:
                accumulate_gain(tree)
        else:
            raise ValueError(f"Unknown importance_type: {importance_type}")
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance


"""
USAGE EXAMPLE 1: Simple Regression with LightGBM

import numpy as np

# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Create and train LightGBM model
model = LightGBM(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=5
)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")

# Make predictions
predictions = model.predict(X_test)

print("\nSample Predictions:")
for i in range(5):
    print(f"True: {y_test[i]:.2f}, Predicted: {predictions[i]:.2f}")
"""

"""
USAGE EXAMPLE 2: Binary Classification with LightGBM

import numpy as np

# Generate classification data
np.random.seed(42)
X_class_0 = np.random.randn(100, 2) + np.array([-2, -2])
X_class_1 = np.random.randn(100, 2) + np.array([2, 2])

X = np.vstack([X_class_0, X_class_1])
y = np.array([0] * 100 + [1] * 100)

# Shuffle
indices = np.random.permutation(200)
X = X[indices]
y = y[indices]

# Split
X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Train LightGBM classifier
model = LightGBM(
    n_estimators=50,
    learning_rate=0.1,
    num_leaves=31,
    objective='binary'
)
model.fit(X_train, y_train, verbose=10)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTraining Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# Get probabilities
probabilities = model.predict_proba(X_test)

print("\nSample Predictions:")
for i in range(5):
    print(f"True: {int(y_test[i])}, "
          f"P(class=0): {probabilities[i, 0]:.3f}, "
          f"P(class=1): {probabilities[i, 1]:.3f}")
"""

"""
USAGE EXAMPLE 3: LightGBM with Early Stopping

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(500, 10)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] ** 2 + np.random.randn(500) * 0.5

# Split train/validation/test
X_train, X_val, X_test = X[:300], X[300:400], X[400:]
y_train, y_val, y_test = y[:300], y[300:400], y[400:]

# Train with early stopping
model = LightGBM(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=10
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=50
)

print(f"\nTrees trained: {len(model.trees)}")

# Evaluate on test set
test_score = model.score(X_test, y_test)
print(f"Test R²: {test_score:.4f}")
"""

"""
USAGE EXAMPLE 4: Feature Importance Analysis

import numpy as np

# Create dataset with informative and noise features
np.random.seed(42)
n_samples = 300

# Informative features
X1 = np.random.randn(n_samples, 1)
X2 = np.random.randn(n_samples, 1)
X3 = np.random.randn(n_samples, 1)

# Noise features
X_noise = np.random.randn(n_samples, 7)

X = np.hstack([X1, X2, X3, X_noise])

# Target depends on first 3 features
y = 3 * X1.ravel() + 2 * X2.ravel() - X3.ravel() + np.random.randn(n_samples) * 0.3

# Train model
model = LightGBM(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31
)
model.fit(X, y)

# Get feature importance
importance_split = model.get_feature_importance('split')
importance_gain = model.get_feature_importance('gain')

print("\nFeature Importance (by split count):")
print("="*50)
for i, imp in enumerate(importance_split):
    bar = '█' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")

print("\nFeature Importance (by gain):")
print("="*50)
for i, imp in enumerate(importance_gain):
    bar = '█' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")
"""

"""
USAGE EXAMPLE 5: Comparing with Different num_leaves Values

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(200, 5)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] + np.random.randn(200) * 0.5

X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Test different num_leaves values
num_leaves_values = [7, 15, 31, 63, 127]

print("Effect of num_leaves (Tree Complexity):")
print("="*80)
print(f"{'num_leaves':>12} {'Train R²':>15} {'Test R²':>15} {'Overfit':>15}")
print("-"*80)

for num_leaves in num_leaves_values:
    model = LightGBM(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=num_leaves,
        min_data_in_leaf=5
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfit = train_score - test_score
    
    print(f"{num_leaves:>12} {train_score:>15.4f} {test_score:>15.4f} {overfit:>15.4f}")

# Observation: Larger num_leaves can lead to overfitting
"""

"""
USAGE EXAMPLE 6: Effect of Learning Rate

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(300, 8)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 + np.sin(X[:, 2]) * X[:, 3] + 
     np.random.randn(300) * 0.5)

X_train, X_test = X[:200], X[100:]
y_train, y_test = y[:200], y[100:]

# Try different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.3]

print("\nEffect of Learning Rate:")
print("="*80)
print(f"{'Learning Rate':>15} {'Train R²':>15} {'Test R²':>15} {'Trees':>10}")
print("-"*80)

for lr in learning_rates:
    model = LightGBM(
        n_estimators=200,
        learning_rate=lr,
        num_leaves=31
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"{lr:>15.2f} {train_score:>15.4f} {test_score:>15.4f} {len(model.trees):>10}")

# Observation: Lower learning rate often needs more trees but generalizes better
"""

"""
USAGE EXAMPLE 7: LightGBM with Feature Subsampling

import numpy as np

# Wide dataset (many features)
np.random.seed(42)
X = np.random.randn(200, 20)
# Only first 5 features are informative
y = (2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] - 
     0.5 * X[:, 3] + X[:, 4] + np.random.randn(200) * 0.5)

X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Test different feature_fraction values
feature_fractions = [0.3, 0.5, 0.7, 1.0]

print("\nEffect of Feature Subsampling:")
print("="*80)
print(f"{'Feature Fraction':>18} {'Train R²':>15} {'Test R²':>15} {'Overfit':>15}")
print("-"*80)

for frac in feature_fractions:
    model = LightGBM(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=frac
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfit = train_score - test_score
    
    print(f"{frac:>18.1f} {train_score:>15.4f} {test_score:>15.4f} {overfit:>15.4f}")

# Observation: Feature subsampling can reduce overfitting
"""

"""
USAGE EXAMPLE 8: Real-World - Sales Prediction

import numpy as np

# Simulated sales data
# [advertising_spend, promotion_days, season, competitor_price, 
#  store_location_score, day_of_week, weather_score]

np.random.seed(42)
n_samples = 500

advertising = np.random.uniform(1000, 10000, n_samples)
promotion = np.random.randint(0, 31, n_samples)
season = np.random.randint(1, 5, n_samples)  # 1=Spring, 2=Summer, 3=Fall, 4=Winter
competitor_price = np.random.uniform(50, 150, n_samples)
location = np.random.uniform(1, 10, n_samples)
day_of_week = np.random.randint(1, 8, n_samples)
weather = np.random.uniform(1, 10, n_samples)

X = np.column_stack([advertising, promotion, season, competitor_price,
                     location, day_of_week, weather])

# Sales formula with interactions
sales = (
    0.5 * advertising +
    200 * promotion +
    5000 * season +
    -100 * competitor_price +
    1000 * location +
    500 * day_of_week +
    300 * weather +
    0.01 * advertising * location +  # Interaction
    np.random.randn(n_samples) * 2000
)

# Normalize to thousands
sales = sales / 1000

# Split data
X_train, X_val, X_test = X[:300], X[300:400], X[400:]
y_train, y_val, y_test = sales[:300], sales[300:400], sales[400:]

# Train LightGBM model
model = LightGBM(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=10,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=50
)

# Evaluate
test_r2 = model.score(X_test, y_test)
predictions = model.predict(X_test)

mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

print(f"\nSales Prediction Model:")
print("="*60)
print(f"Test R²: {test_r2:.4f}")
print(f"Mean Absolute Error: ${mae:.2f}k")
print(f"Root Mean Squared Error: ${rmse:.2f}k")

# Feature importance
feature_names = ['Advertising', 'Promotion Days', 'Season', 'Competitor Price',
                'Location Score', 'Day of Week', 'Weather']
importance = model.get_feature_importance('gain')

print("\nFeature Importance:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.4f}")

# Predict new scenarios
new_scenarios = np.array([
    [8000, 15, 4, 75, 9, 6, 8],   # High ad spend, promotion, good location
    [2000, 0, 1, 120, 3, 2, 4]     # Low ad spend, no promotion, poor location
])

predicted_sales = model.predict(new_scenarios)

print("\nNew Scenario Sales Predictions:")
for i, pred in enumerate(predicted_sales):
    print(f"Scenario {i+1}: ${pred:.2f}k")
"""

"""
USAGE EXAMPLE 9: Click-Through Rate Prediction

import numpy as np

# User features for ad click prediction
# [age, gender, device_type, ad_position, time_of_day, 
#  user_interests_match, previous_clicks, session_duration]

np.random.seed(42)

# Generate data for clickers and non-clickers
n_clickers = 200
n_non_clickers = 800

# Clickers (more engaged users)
X_click = np.random.randn(n_clickers, 8) * \
          np.array([10, 0.5, 0.3, 1, 3, 1.5, 2, 20]) + \
          np.array([35, 1, 1, 2, 14, 8, 5, 300])

# Non-clickers
X_no_click = np.random.randn(n_non_clickers, 8) * \
             np.array([15, 0.5, 0.3, 1, 4, 1, 1, 30]) + \
             np.array([45, 0, 2, 5, 10, 3, 1, 150])

X = np.vstack([X_click, X_no_click])
y = np.array([1] * n_clickers + [0] * n_non_clickers)

# Shuffle
indices = np.random.permutation(1000)
X = X[indices]
y = y[indices]

# Split
X_train, X_val, X_test = X[:600], X[600:800], X[800:]
y_train, y_val, y_test = y[:600], y[600:800], y[800:]

# Train CTR prediction model
model = LightGBM(
    n_estimators=150,
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    objective='binary'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=15,
    verbose=30
)

# Evaluate
test_acc = model.score(X_test, y_test)

print(f"\nClick-Through Rate Prediction:")
print("="*60)
print(f"Test Accuracy: {test_acc:.2%}")

# Calculate additional metrics
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int)

true_positives = np.sum((predicted_classes == 1) & (y_test == 1))
false_positives = np.sum((predicted_classes == 1) & (y_test == 0))
false_negatives = np.sum((predicted_classes == 0) & (y_test == 1))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.4f}")

# Feature importance
feature_names = ['Age', 'Gender', 'Device', 'Ad Position', 
                'Time of Day', 'Interest Match', 'Previous Clicks', 'Session Duration']
importance = model.get_feature_importance('gain')

print("\nTop Features for CTR:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name:20s}: {imp:.4f}")

# Predict new users
new_users = np.array([
    [28, 1, 1, 1, 14, 9, 8, 450],   # Engaged user, good targeting
    [55, 0, 2, 8, 3, 2, 0, 60]       # Less engaged user, poor ad position
])

click_probabilities = model.predict(new_users)

print("\nPredicted Click Probabilities:")
for i, prob in enumerate(click_probabilities):
    likelihood = "HIGH" if prob >= 0.5 else "LOW"
    print(f"User {i+1}: {likelihood} ({prob:.2%} probability)")
"""
