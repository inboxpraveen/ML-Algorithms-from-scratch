import numpy as np

class CatBoost:
    """
    CatBoost (Categorical Boosting) Implementation from Scratch
    
    CatBoost is a gradient boosting framework developed by Yandex that handles
    categorical features naturally and uses symmetric (oblivious) trees.
    It addresses prediction shift through ordered boosting.
    
    Key Idea: "Symmetric trees + Ordered boosting + Smart categorical encoding"
    
    Use Cases:
    - Regression: Price prediction, demand forecasting, risk scoring
    - Classification: Fraud detection, customer churn, recommendation
    - Ranking: Search engines, recommendation systems
    - Categorical-heavy datasets: E-commerce, web analytics
    
    Key Innovations in CatBoost:
        Symmetric Trees: All nodes at same level split on same feature/threshold
        Ordered Boosting: Prevents prediction shift and target leakage
        Ordered Target Statistics: Smart categorical feature encoding
        No need for extensive preprocessing: Handles categoricals natively
        Robust to overfitting: Built-in regularization through ordered boosting
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.03, depth=6,
                 l2_leaf_reg=3.0, min_data_in_leaf=1, random_strength=1.0,
                 border_count=128, objective='regression'):
        """
        Initialize the CatBoost model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of boosting iterations (trees to build)
            - More iterations: Better training fit, longer training
            - Fewer iterations: Faster training, may underfit
            Typical values: 100-1000
            
        learning_rate : float, default=0.03
            Learning rate (also called eta)
            - Lower values need more iterations but generalize better
            - Range: 0.01 to 0.3
            Typical: 0.03 is CatBoost default (lower than XGBoost/LightGBM)
            
        depth : int, default=6
            Depth of symmetric trees
            - Determines number of splits: 2^depth leaves
            - Larger values: More complex model, risk overfitting
            - CatBoost uses symmetric trees, so depth is main complexity control
            Typical values: 4-10
            
        l2_leaf_reg : float, default=3.0
            L2 regularization coefficient for leaf values
            - Higher values: More regularization, less overfitting
            - CatBoost default is 3.0 (higher than XGBoost default of 1.0)
            Typical values: 1-10
            
        min_data_in_leaf : int, default=1
            Minimum number of training samples in a leaf
            - Larger values prevent overfitting
            - CatBoost default is 1 (trusts ordered boosting for regularization)
            Typical values: 1-20
            
        random_strength : float, default=1.0
            Amount of randomness for scoring splits
            - Higher values: More randomization, better generalization
            - 0: Deterministic (no randomization)
            Typical values: 0-2
            
        border_count : int, default=128
            Number of splits for numerical features
            - Similar to LightGBM's max_bin
            - Higher values: More accurate but slower
            Typical values: 32, 64, 128, 254
            
        objective : str, default='regression'
            Learning objective
            - 'regression': Regression with RMSE loss
            - 'binary': Binary classification with logloss
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.random_strength = random_strength
        self.border_count = border_count
        self.objective = objective
        
        self.trees = []
        self.base_score = None
        self.feature_borders = None
        
    def _sigmoid(self, x):
        """Sigmoid function with numerical stability"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _compute_gradients(self, y_true, y_pred):
        """
        Compute gradients for the loss function
        
        CatBoost uses first-order gradients (unlike XGBoost which uses both
        first and second order). This is simpler but still effective with
        ordered boosting.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        gradients : np.ndarray
            First-order gradients
        """
        if self.objective == 'regression':
            # For squared error: L = 0.5 * (y - pred)^2
            # Gradient: dL/dpred = pred - y
            gradients = y_pred - y_true
            
        elif self.objective == 'binary':
            # For log loss: L = -y*log(p) - (1-y)*log(1-p)
            # Gradient: dL/dpred = p - y
            p = self._sigmoid(y_pred)
            gradients = p - y_true
            
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return gradients
    
    def _quantize_features(self, X):
        """
        Quantize continuous features into discrete bins
        
        Similar to LightGBM's histogram building, but CatBoost calls it
        "border selection". Creates discrete bins for faster split evaluation.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_quantized : np.ndarray, shape (n_samples, n_features)
            Quantized feature values (integers)
        """
        n_samples, n_features = X.shape
        X_quantized = np.zeros_like(X, dtype=int)
        self.feature_borders = []
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) <= self.border_count:
                borders = unique_values[:-1]
            else:
                # Create borders using quantiles
                percentiles = np.linspace(0, 100, self.border_count + 1)[1:-1]
                borders = np.percentile(feature_values, percentiles)
                borders = np.unique(borders)
            
            self.feature_borders.append(borders)
            X_quantized[:, feature_idx] = np.digitize(feature_values, borders)
        
        return X_quantized
    
    def _apply_quantization(self, X):
        """Apply pre-computed quantization to new data"""
        n_samples, n_features = X.shape
        X_quantized = np.zeros_like(X, dtype=int)
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            borders = self.feature_borders[feature_idx]
            X_quantized[:, feature_idx] = np.digitize(feature_values, borders)
        
        return X_quantized
    
    def _calculate_leaf_value(self, gradients, indices):
        """
        Calculate optimal leaf value with L2 regularization
        
        CatBoost formula: value = -sum(gradients) / (n_samples + l2_leaf_reg)
        
        The L2 regularization in denominator acts as smoothing:
        - More samples → less regularization effect
        - Fewer samples → more shrinkage toward zero
        
        Parameters:
        -----------
        gradients : np.ndarray
            Gradients for samples in this leaf
        indices : np.ndarray (boolean)
            Boolean mask for samples in this leaf
            
        Returns:
        --------
        value : float
            Optimal leaf value
        """
        if np.sum(indices) == 0:
            return 0.0
        
        gradient_sum = np.sum(gradients[indices])
        count = np.sum(indices)
        
        # CatBoost's leaf value formula with L2 regularization
        value = -gradient_sum / (count + self.l2_leaf_reg)
        
        return value
    
    def _build_symmetric_tree(self, X_quantized, gradients, depth=0):
        """
        Build symmetric (oblivious) tree
        
        SYMMETRIC TREES are CatBoost's key innovation:
        - All nodes at the same level use the SAME split condition
        - Creates 2^depth leaves with symmetric structure
        - Faster prediction and less overfitting
        
        Example for depth=2:
                    [Feature 3 <= 5]
                   /              \\
            [Feature 1 <= 2]    [Feature 1 <= 2]
             /        \\          /        \\
           Leaf0    Leaf1      Leaf2    Leaf3
        
        Note: Both level-1 nodes split on Feature 1!
        
        Parameters:
        -----------
        X_quantized : np.ndarray, shape (n_samples, n_features)
            Quantized training data
        gradients : np.ndarray, shape (n_samples,)
            Gradients to optimize
        depth : int
            Current depth (0 = root)
            
        Returns:
        --------
        tree : dict
            Symmetric tree structure
        """
        n_samples, n_features = X_quantized.shape
        
        # Store split conditions for each level
        splits = []
        
        # Current partition (which samples go to which leaf)
        current_partitions = [np.ones(n_samples, dtype=bool)]
        
        # Build tree level by level
        for level in range(self.depth):
            best_gain = -np.inf
            best_feature = None
            best_threshold = None
            
            # Try all features and thresholds
            for feature_idx in range(n_features):
                feature_values = X_quantized[:, feature_idx]
                unique_values = np.unique(feature_values)
                
                for threshold in unique_values:
                    # Calculate gain for this split applied to ALL current partitions
                    total_gain = 0
                    
                    for partition in current_partitions:
                        if np.sum(partition) < self.min_data_in_leaf:
                            continue
                        
                        # Split this partition
                        left_mask = partition & (feature_values <= threshold)
                        right_mask = partition & (feature_values > threshold)
                        
                        if np.sum(left_mask) < self.min_data_in_leaf or \
                           np.sum(right_mask) < self.min_data_in_leaf:
                            continue
                        
                        # Calculate gain (reduction in loss)
                        left_grad_sum = np.sum(gradients[left_mask])
                        right_grad_sum = np.sum(gradients[right_mask])
                        parent_grad_sum = np.sum(gradients[partition])
                        
                        left_count = np.sum(left_mask)
                        right_count = np.sum(right_mask)
                        parent_count = np.sum(partition)
                        
                        # Gain = loss_before - loss_after (with L2 regularization)
                        loss_before = (parent_grad_sum ** 2) / (parent_count + self.l2_leaf_reg)
                        loss_after = ((left_grad_sum ** 2) / (left_count + self.l2_leaf_reg) +
                                     (right_grad_sum ** 2) / (right_count + self.l2_leaf_reg))
                        
                        gain = loss_after - loss_before
                        total_gain += gain
                    
                    # Add randomness for split selection (random_strength)
                    if self.random_strength > 0:
                        total_gain += np.random.randn() * self.random_strength
                    
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_feature = feature_idx
                        best_threshold = threshold
            
            # If no valid split found, stop growing
            if best_feature is None:
                break
            
            # Record this level's split
            splits.append({
                'feature': best_feature,
                'threshold': best_threshold
            })
            
            # Update partitions: split each partition using this split
            new_partitions = []
            for partition in current_partitions:
                feature_values = X_quantized[:, best_feature]
                left_mask = partition & (feature_values <= best_threshold)
                right_mask = partition & (feature_values > best_threshold)
                new_partitions.append(left_mask)
                new_partitions.append(right_mask)
            
            current_partitions = new_partitions
        
        # Calculate leaf values for final partitions
        leaf_values = []
        for partition in current_partitions:
            value = self._calculate_leaf_value(gradients, partition)
            leaf_values.append(value)
        
        return {
            'type': 'symmetric',
            'splits': splits,
            'leaf_values': np.array(leaf_values),
            'depth': len(splits)
        }
    
    def _predict_tree(self, tree, X_quantized):
        """
        Make predictions using a symmetric tree
        
        For symmetric trees, prediction is fast:
        1. Start at leaf index 0
        2. For each level's split:
           - If condition true: stay in left subtree
           - If condition false: add 2^(remaining_depth) to index
        3. Return value at final leaf index
        
        Parameters:
        -----------
        tree : dict
            Symmetric tree structure
        X_quantized : np.ndarray
            Quantized data
            
        Returns:
        --------
        predictions : np.ndarray
            Tree predictions
        """
        n_samples = X_quantized.shape[0]
        predictions = np.zeros(n_samples)
        
        # Calculate leaf indices for all samples
        leaf_indices = np.zeros(n_samples, dtype=int)
        
        # Apply each split
        for level, split in enumerate(tree['splits']):
            feature_idx = split['feature']
            threshold = split['threshold']
            
            # Samples going right add 2^(depth-level-1) to their leaf index
            goes_right = X_quantized[:, feature_idx] > threshold
            remaining_depth = tree['depth'] - level - 1
            leaf_indices += goes_right * (2 ** remaining_depth)
        
        # Get predictions from leaf values
        predictions = tree['leaf_values'][leaf_indices]
        
        return predictions
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the CatBoost model
        
        Algorithm:
        1. Quantize features into discrete bins
        2. Initialize predictions with base score
        3. For each boosting iteration:
           a. Calculate gradients
           b. Build symmetric tree to minimize loss
           c. Update predictions with tree × learning_rate
        4. Optional: Early stopping on validation set
        
        CatBoost uses ORDERED BOOSTING (simplified in this implementation):
        - Prevents prediction shift and target leakage
        - Each sample's gradient uses predictions from trees trained on other samples
        - Makes the model more robust to overfitting
        
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
        self : CatBoost
            Fitted model
        """
        # Convert to numpy arrays
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Quantize features (border selection)
        X_quantized = self._quantize_features(X)
        
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
            # Calculate gradients
            gradients = self._compute_gradients(y, predictions)
            
            # Build symmetric tree
            tree = self._build_symmetric_tree(X_quantized, gradients)
            self.trees.append(tree)
            
            # Update predictions
            tree_predictions = self._predict_tree(tree, X_quantized)
            predictions += self.learning_rate * tree_predictions
            
            # Calculate training score
            if self.objective == 'binary':
                train_preds = self._sigmoid(predictions)
                train_score = -np.mean(y * np.log(train_preds + 1e-10) + 
                                      (1 - y) * np.log(1 - train_preds + 1e-10))
            else:
                train_score = np.sqrt(np.mean((y - predictions) ** 2))
            
            self.train_scores.append(train_score)
            
            # Evaluate on validation set
            if eval_set is not None:
                X_val, y_val = eval_set[0]
                val_preds = self.predict(X_val, num_iteration=iteration+1)
                
                if self.objective == 'binary':
                    val_score = -np.mean(y_val * np.log(val_preds + 1e-10) + 
                                        (1 - y_val) * np.log(1 - val_preds + 1e-10))
                else:
                    val_score = np.sqrt(np.mean((y_val - val_preds) ** 2))
                
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
                    if self.objective == 'binary':
                        print(f"[{iteration}] train-logloss: {train_score:.6f}, "
                              f"val-logloss: {val_score:.6f}")
                    else:
                        print(f"[{iteration}] train-rmse: {train_score:.6f}, "
                              f"val-rmse: {val_score:.6f}")
            elif verbose and (isinstance(verbose, bool) or iteration % verbose == 0):
                if self.objective == 'binary':
                    print(f"[{iteration}] train-logloss: {train_score:.6f}")
                else:
                    print(f"[{iteration}] train-rmse: {train_score:.6f}")
        
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
        
        # Apply quantization
        X_quantized = self._apply_quantization(X)
        
        # Start with base score
        predictions = np.full(n_samples, self.base_score)
        
        # Determine number of trees to use
        n_trees = len(self.trees) if num_iteration is None else min(num_iteration, len(self.trees))
        
        # Add contribution from each tree
        for i in range(n_trees):
            tree_predictions = self._predict_tree(self.trees[i], X_quantized)
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
            RMSE for regression, accuracy for classification
        """
        y = np.array(y)
        predictions = self.predict(X)
        
        if self.objective == 'binary':
            # Classification: accuracy
            predicted_classes = (predictions >= 0.5).astype(int)
            return np.mean(predicted_classes == y)
        else:
            # Regression: negative RMSE (higher is better)
            rmse = np.sqrt(np.mean((y - predictions) ** 2))
            return -rmse
    
    def get_feature_importance(self, importance_type='split'):
        """
        Calculate feature importance
        
        For symmetric trees, feature importance is straightforward:
        - Count how many times each feature is used in splits
        - Or sum the gain improvements from that feature
        
        Parameters:
        -----------
        importance_type : str, default='split'
            Type of importance:
            - 'split': Number of times feature is used for splitting
            
        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Feature importance scores (normalized)
        """
        importance = np.zeros(self.n_features)
        
        for tree in self.trees:
            for split in tree['splits']:
                importance[split['feature']] += 1
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance


"""
USAGE EXAMPLE 1: Simple Regression with CatBoost

import numpy as np

# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Create and train CatBoost model
model = CatBoost(
    n_estimators=100,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0
)
model.fit(X_train, y_train)

# Evaluate
train_rmse = -model.score(X_train, y_train)
test_rmse = -model.score(X_test, y_test)

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Make predictions
predictions = model.predict(X_test)

print("\nSample Predictions:")
for i in range(5):
    print(f"True: {y_test[i]:.2f}, Predicted: {predictions[i]:.2f}")
"""

"""
USAGE EXAMPLE 2: Binary Classification with CatBoost

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

# Train CatBoost classifier
model = CatBoost(
    n_estimators=50,
    learning_rate=0.05,
    depth=6,
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
USAGE EXAMPLE 3: CatBoost with Early Stopping

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(500, 10)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] ** 2 + np.random.randn(500) * 0.5

# Split train/validation/test
X_train, X_val, X_test = X[:300], X[300:400], X[400:]
y_train, y_val, y_test = y[:300], y[300:400], y[400:]

# Train with early stopping
model = CatBoost(
    n_estimators=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=50
)

print(f"\nTrees trained: {len(model.trees)}")

# Evaluate on test set
test_rmse = -model.score(X_test, y_test)
print(f"Test RMSE: {test_rmse:.4f}")
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
model = CatBoost(
    n_estimators=100,
    learning_rate=0.05,
    depth=6
)
model.fit(X, y)

# Get feature importance
importance = model.get_feature_importance('split')

print("\nFeature Importance (by split count):")
print("="*50)
for i, imp in enumerate(importance):
    bar = '█' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")
"""

"""
USAGE EXAMPLE 5: Comparing Different Tree Depths

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(200, 5)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] + np.random.randn(200) * 0.5

X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Test different depths
depths = [3, 4, 6, 8, 10]

print("Effect of Tree Depth (Complexity):")
print("="*80)
print(f"{'Depth':>8} {'Leaves':>8} {'Train RMSE':>15} {'Test RMSE':>15} {'Overfit':>15}")
print("-"*80)

for depth in depths:
    model = CatBoost(
        n_estimators=100,
        learning_rate=0.05,
        depth=depth
    )
    model.fit(X_train, y_train)
    
    train_rmse = -model.score(X_train, y_train)
    test_rmse = -model.score(X_test, y_test)
    overfit = test_rmse - train_rmse
    num_leaves = 2 ** depth
    
    print(f"{depth:>8} {num_leaves:>8} {train_rmse:>15.4f} {test_rmse:>15.4f} {overfit:>15.4f}")

# Observation: Larger depth can lead to overfitting with symmetric trees
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
learning_rates = [0.01, 0.03, 0.05, 0.1, 0.3]

print("\nEffect of Learning Rate:")
print("="*80)
print(f"{'Learning Rate':>15} {'Train RMSE':>15} {'Test RMSE':>15} {'Trees':>10}")
print("-"*80)

for lr in learning_rates:
    model = CatBoost(
        n_estimators=200,
        learning_rate=lr,
        depth=6
    )
    model.fit(X_train, y_train)
    
    train_rmse = -model.score(X_train, y_train)
    test_rmse = -model.score(X_test, y_test)
    
    print(f"{lr:>15.2f} {train_rmse:>15.4f} {test_rmse:>15.4f} {len(model.trees):>10}")

# Observation: CatBoost uses lower learning rates (0.03) by default
"""

"""
USAGE EXAMPLE 7: Effect of L2 Regularization

import numpy as np

# Generate data with potential for overfitting
np.random.seed(42)
X = np.random.randn(150, 15)  # Many features, few samples
y = 2 * X[:, 0] - X[:, 1] + np.random.randn(150) * 0.5

X_train, X_test = X[:100], X[50:]
y_train, y_test = y[:100], y[50:]

# Test different l2_leaf_reg values
l2_values = [0.1, 1.0, 3.0, 10.0, 30.0]

print("\nEffect of L2 Regularization:")
print("="*80)
print(f"{'L2 Leaf Reg':>15} {'Train RMSE':>15} {'Test RMSE':>15} {'Overfit':>15}")
print("-"*80)

for l2 in l2_values:
    model = CatBoost(
        n_estimators=100,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=l2
    )
    model.fit(X_train, y_train)
    
    train_rmse = -model.score(X_train, y_train)
    test_rmse = -model.score(X_test, y_test)
    overfit = test_rmse - train_rmse
    
    print(f"{l2:>15.1f} {train_rmse:>15.4f} {test_rmse:>15.4f} {overfit:>15.4f}")

# Observation: Higher L2 regularization reduces overfitting
"""

"""
USAGE EXAMPLE 8: Real-World - Customer Churn Prediction

import numpy as np

# Simulated customer data
# [tenure_months, monthly_charges, total_charges, num_services, 
#  support_tickets, contract_type, payment_method]

np.random.seed(42)

# Churned customers (shorter tenure, more tickets)
n_churn = 200
X_churn = np.column_stack([
    np.random.uniform(1, 12, n_churn),      # Short tenure
    np.random.uniform(70, 120, n_churn),    # High charges
    np.random.uniform(70, 1440, n_churn),   # Low total (short tenure)
    np.random.randint(1, 5, n_churn),       # Few services
    np.random.randint(3, 10, n_churn),      # Many tickets
    np.random.randint(0, 2, n_churn),       # Month-to-month
    np.random.randint(0, 3, n_churn)        # Payment method
])

# Retained customers (longer tenure, fewer tickets)
n_retain = 800
X_retain = np.column_stack([
    np.random.uniform(13, 72, n_retain),    # Long tenure
    np.random.uniform(50, 100, n_retain),   # Lower charges
    np.random.uniform(1000, 7200, n_retain),# High total (long tenure)
    np.random.randint(2, 6, n_retain),      # More services
    np.random.randint(0, 3, n_retain),      # Few tickets
    np.random.randint(1, 3, n_retain),      # Long contracts
    np.random.randint(0, 3, n_retain)       # Payment method
])

X = np.vstack([X_churn, X_retain])
y = np.array([1] * n_churn + [0] * n_retain)

# Shuffle
indices = np.random.permutation(1000)
X = X[indices]
y = y[indices]

# Split
X_train, X_val, X_test = X[:600], X[600:800], X[800:]
y_train, y_val, y_test = y[:600], y[600:800], y[800:]

# Train CatBoost model
model = CatBoost(
    n_estimators=200,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    objective='binary'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=50
)

# Evaluate
test_acc = model.score(X_test, y_test)
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int)

# Calculate metrics
true_positives = np.sum((predicted_classes == 1) & (y_test == 1))
false_positives = np.sum((predicted_classes == 1) & (y_test == 0))
false_negatives = np.sum((predicted_classes == 0) & (y_test == 1))

precision = true_positives / (true_positives + false_positives + 1e-10)
recall = true_positives / (true_positives + false_negatives + 1e-10)
f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

print(f"\nCustomer Churn Prediction:")
print("="*60)
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.4f}")

# Feature importance
feature_names = ['Tenure', 'Monthly Charges', 'Total Charges', 'Num Services',
                'Support Tickets', 'Contract Type', 'Payment Method']
importance = model.get_feature_importance('split')

print("\nTop Features for Churn Prediction:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.4f}")

# Predict churn for new customers
new_customers = np.array([
    [3, 95, 285, 2, 5, 0, 1],      # High risk: short tenure, many tickets
    [48, 65, 3120, 4, 1, 2, 0]     # Low risk: long tenure, few tickets
])

churn_probs = model.predict(new_customers)

print("\nChurn Risk Assessment:")
for i, prob in enumerate(churn_probs):
    risk = "HIGH" if prob >= 0.5 else "LOW"
    print(f"Customer {i+1}: {risk} RISK ({prob:.2%} probability of churn)")
"""

"""
USAGE EXAMPLE 9: Comparing CatBoost with Different Configurations

import numpy as np

# Generate complex data
np.random.seed(42)
X = np.random.randn(300, 10)
y = (2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] ** 2 - 
     np.sin(X[:, 3]) * X[:, 4] + np.random.randn(300) * 0.5)

X_train, X_test = X[:200], X[100:]
y_train, y_test = y[:200], y[100:]

# Test different configurations
configs = [
    {'name': 'Fast', 'params': {'n_estimators': 50, 'depth': 4, 'learning_rate': 0.1}},
    {'name': 'Balanced', 'params': {'n_estimators': 100, 'depth': 6, 'learning_rate': 0.05}},
    {'name': 'Accurate', 'params': {'n_estimators': 200, 'depth': 8, 'learning_rate': 0.03}},
    {'name': 'Regularized', 'params': {'n_estimators': 100, 'depth': 6, 'l2_leaf_reg': 10.0}}
]

print("\nComparing CatBoost Configurations:")
print("="*80)
print(f"{'Config':>15} {'Trees':>8} {'Depth':>8} {'Train RMSE':>15} {'Test RMSE':>15}")
print("-"*80)

for config in configs:
    model = CatBoost(**config['params'])
    model.fit(X_train, y_train)
    
    train_rmse = -model.score(X_train, y_train)
    test_rmse = -model.score(X_test, y_test)
    
    print(f"{config['name']:>15} {len(model.trees):>8} "
          f"{config['params'].get('depth', 6):>8} "
          f"{train_rmse:>15.4f} {test_rmse:>15.4f}")

print("\nRecommendation:")
print("- Fast: Quick training for prototyping")
print("- Balanced: Good default for most cases")
print("- Accurate: Maximum accuracy when training time is not an issue")
print("- Regularized: When overfitting is a concern")
"""
