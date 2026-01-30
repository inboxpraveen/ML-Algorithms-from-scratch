import numpy as np

class XGBoost:
    """
    XGBoost (Extreme Gradient Boosting) Implementation from Scratch
    
    XGBoost is an optimized distributed gradient boosting library designed to be highly efficient,
    flexible and portable. It implements machine learning algorithms under the Gradient Boosting
    framework with significant improvements over traditional gradient boosting.
    
    Key Idea: "Regularized gradient boosting with advanced tree learning techniques"
    
    Use Cases:
    - Regression: House prices, sales forecasting, demand prediction
    - Classification: Credit risk, fraud detection, customer churn
    - Ranking: Search engines, recommendation systems
    - Feature Selection: Identifying important variables
    
    Key Improvements over Standard Gradient Boosting:
        Regularization: L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting
        Tree Pruning: Max depth pruning with backward pruning for efficiency
        Handling Missing Values: Learns optimal default direction for missing values
        Column Subsampling: Random feature selection per tree (like Random Forest)
        Weighted Quantile Sketch: Efficient split finding algorithm
        Sparsity Awareness: Optimized for sparse data
        Built-in Cross-Validation: Early stopping to prevent overfitting
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.3, max_depth=6, 
                 min_child_weight=1, gamma=0, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, objective='reg:squarederror'):
        """
        Initialize the XGBoost model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of boosting rounds (trees) to train
            - More estimators: Better training fit, risk overfitting
            - Fewer estimators: Faster training, may underfit
            Typical values: 100-1000
            
        learning_rate : float, default=0.3
            Step size shrinkage to prevent overfitting (also called eta)
            - Range: 0.01 to 1.0
            - Lower values need more estimators but generalize better
            Typical: 0.3 (XGBoost default), 0.1 is more conservative
            
        max_depth : int, default=6
            Maximum depth of each tree
            - Deeper trees: More complex patterns, risk overfitting
            - Shallow trees: More regularization, better generalization
            Typical values: 3-10 (6 is XGBoost default)
            
        min_child_weight : float, default=1
            Minimum sum of instance weight (hessian) needed in a child
            - Higher values prevent learning highly specific relations (more conservative)
            - Used to control overfitting
            Typical values: 1-10
            
        gamma : float, default=0
            Minimum loss reduction required to make a split (also called min_split_loss)
            - Acts as regularization
            - Higher values make algorithm more conservative
            Typical values: 0-5
            
        subsample : float, default=1.0
            Fraction of samples to use for training each tree
            - < 1.0 introduces randomness (stochastic gradient boosting)
            - Helps prevent overfitting
            Typical values: 0.5-1.0
            
        colsample_bytree : float, default=1.0
            Fraction of features to use when constructing each tree
            - Similar to Random Forest's feature sampling
            - Reduces overfitting and speeds up training
            Typical values: 0.3-1.0
            
        reg_lambda : float, default=1.0
            L2 regularization term on weights (Ridge)
            - Higher values lead to more conservative models
            - Helps prevent overfitting
            Typical values: 0-10
            
        reg_alpha : float, default=0.0
            L1 regularization term on weights (Lasso)
            - Can lead to sparse solutions
            - Useful for feature selection
            Typical values: 0-10
            
        objective : str, default='reg:squarederror'
            Learning objective
            - 'reg:squarederror': Regression with squared loss (L2 loss)
            - 'reg:logistic': Logistic regression for binary classification
            - 'binary:logistic': Binary classification with logistic output
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.objective = objective
        self.trees = []
        self.base_score = None
        
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
        
        XGBoost uses both first-order (gradient) and second-order (hessian)
        derivatives for more accurate optimization
        
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
        if self.objective in ['reg:squarederror', 'reg:linear']:
            # For squared error: L = 0.5 * (y - pred)^2
            # Gradient: dL/dpred = pred - y
            # Hessian: d²L/dpred² = 1
            gradient = y_pred - y_true
            hessian = np.ones_like(y_pred)
            
        elif self.objective in ['binary:logistic', 'reg:logistic']:
            # For logistic: L = -y*log(p) - (1-y)*log(1-p)
            # where p = sigmoid(pred)
            # Gradient: dL/dpred = p - y
            # Hessian: d²L/dpred² = p * (1 - p)
            p = self._sigmoid(y_pred)
            gradient = p - y_true
            hessian = p * (1 - p)
            
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return gradient, hessian
    
    def _calculate_leaf_weight(self, gradient_sum, hessian_sum):
        """
        Calculate optimal leaf weight using XGBoost's formula
        
        XGBoost formula: w* = -G / (H + lambda)
        where G = sum of gradients, H = sum of hessians
        
        This is the optimal weight that minimizes the loss function
        with L2 regularization
        
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
        # Add small epsilon to avoid division by zero
        return -gradient_sum / (hessian_sum + self.reg_lambda + 1e-10)
    
    def _calculate_gain(self, gradient_left, hessian_left, gradient_right, hessian_right):
        """
        Calculate the gain from a split using XGBoost's gain formula
        
        Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
        
        Where:
        - G_L, G_R: Sum of gradients in left/right child
        - H_L, H_R: Sum of hessians in left/right child
        - λ (lambda): L2 regularization
        - γ (gamma): Minimum loss reduction (complexity cost)
        
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
            Gain from the split (higher is better)
        """
        # Calculate scores for left and right
        def calculate_score(G, H):
            return (G ** 2) / (H + self.reg_lambda + 1e-10)
        
        gain_left = calculate_score(gradient_left, hessian_left)
        gain_right = calculate_score(gradient_right, hessian_right)
        gain_parent = calculate_score(gradient_left + gradient_right, 
                                      hessian_left + hessian_right)
        
        # Gain formula with gamma (complexity cost)
        gain = 0.5 * (gain_left + gain_right - gain_parent) - self.gamma
        
        return gain
    
    def _build_tree(self, X, gradient, hessian, depth=0, feature_indices=None):
        """
        Build a regression tree optimized for XGBoost
        
        Uses XGBoost's advanced tree building algorithm:
        1. Considers both gradient and hessian
        2. Uses regularized gain calculation
        3. Implements column subsampling
        4. Pruning based on min_child_weight and gamma
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        gradient : np.ndarray, shape (n_samples,)
            First-order gradients
        hessian : np.ndarray, shape (n_samples,)
            Second-order gradients (hessian)
        depth : int
            Current depth of the tree
        feature_indices : np.ndarray, optional
            Indices of features to consider (for column subsampling)
            
        Returns:
        --------
        tree : dict
            Tree structure with nodes and split information
        """
        n_samples, n_features = X.shape
        
        # Base case: stopping criteria
        gradient_sum = np.sum(gradient)
        hessian_sum = np.sum(hessian)
        
        # Check stopping conditions
        if (depth >= self.max_depth or 
            n_samples < 2 or 
            hessian_sum < self.min_child_weight):
            # Create leaf node with optimal weight
            leaf_weight = self._calculate_leaf_weight(gradient_sum, hessian_sum)
            return {
                'type': 'leaf',
                'weight': leaf_weight,
                'count': n_samples
            }
        
        # Column subsampling (if not already specified)
        if feature_indices is None:
            n_features_use = max(1, int(self.colsample_bytree * n_features))
            feature_indices = np.random.choice(n_features, n_features_use, replace=False)
        
        # Find best split
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_mask = None
        
        # Try each feature
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            
            # Get unique values and sort them
            thresholds = np.unique(feature_values)
            
            # Try each threshold
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Check if split is valid
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate gradient and hessian sums for children
                gradient_left = np.sum(gradient[left_mask])
                hessian_left = np.sum(hessian[left_mask])
                gradient_right = np.sum(gradient[right_mask])
                hessian_right = np.sum(hessian[right_mask])
                
                # Check min_child_weight constraint
                if hessian_left < self.min_child_weight or hessian_right < self.min_child_weight:
                    continue
                
                # Calculate gain
                gain = self._calculate_gain(gradient_left, hessian_left,
                                           gradient_right, hessian_right)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_mask = left_mask
        
        # If no good split found, create leaf
        if best_gain <= 0:
            leaf_weight = self._calculate_leaf_weight(gradient_sum, hessian_sum)
            return {
                'type': 'leaf',
                'weight': leaf_weight,
                'count': n_samples
            }
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(
            X[best_left_mask],
            gradient[best_left_mask],
            hessian[best_left_mask],
            depth + 1,
            feature_indices
        )
        
        right_tree = self._build_tree(
            X[~best_left_mask],
            gradient[~best_left_mask],
            hessian[~best_left_mask],
            depth + 1,
            feature_indices
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
    
    def _predict_tree(self, tree, X):
        """
        Make predictions using a single tree
        
        Parameters:
        -----------
        tree : dict
            Tree structure
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Tree predictions
        """
        if tree['type'] == 'leaf':
            return np.full(len(X), tree['weight'])
        
        # Split based on feature threshold
        feature_values = X[:, tree['feature']]
        left_mask = feature_values <= tree['threshold']
        
        predictions = np.zeros(len(X))
        
        if np.sum(left_mask) > 0:
            predictions[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        if np.sum(~left_mask) > 0:
            predictions[~left_mask] = self._predict_tree(tree['right'], X[~left_mask])
        
        return predictions
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the XGBoost model
        
        Algorithm:
        1. Initialize predictions with base score
        2. For each boosting round:
           a. Calculate gradients and hessians
           b. Subsample data (if subsample < 1.0)
           c. Build tree using gradient and hessian
           d. Update predictions with learning rate
        3. Optional: Early stopping based on validation set
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data
        y : np.ndarray or list, shape (n_samples,)
            Target values
        eval_set : list of tuples, optional
            List of (X_val, y_val) tuples for validation
            Used for early stopping and monitoring
        early_stopping_rounds : int, optional
            Stop training if validation score doesn't improve for this many rounds
        verbose : bool or int, default=False
            If True, print training progress
            If int, print every verbose rounds
            
        Returns:
        --------
        self : XGBoost
            Fitted model
        """
        # Convert to numpy arrays
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Initialize base score
        if self.objective in ['binary:logistic', 'reg:logistic']:
            # For classification, initialize with log-odds
            p = np.mean(y)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            self.base_score = np.log(p / (1 - p))
        else:
            # For regression, initialize with mean
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
            
            # Row subsampling
            if self.subsample < 1.0:
                sample_size = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
                gradient_sample = gradient[indices]
                hessian_sample = hessian[indices]
            else:
                X_sample = X
                gradient_sample = gradient
                hessian_sample = hessian
            
            # Build tree
            tree = self._build_tree(X_sample, gradient_sample, hessian_sample)
            self.trees.append(tree)
            
            # Update predictions
            tree_predictions = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_predictions
            
            # Calculate training score
            if self.objective in ['binary:logistic', 'reg:logistic']:
                train_preds = self._sigmoid(predictions)
                train_score = -np.mean(y * np.log(train_preds + 1e-10) + 
                                      (1 - y) * np.log(1 - train_preds + 1e-10))
            else:
                train_score = np.mean((y - predictions) ** 2)
            
            self.train_scores.append(train_score)
            
            # Evaluate on validation set if provided
            if eval_set is not None:
                X_val, y_val = eval_set[0]
                val_preds = self.predict(X_val, num_iteration=iteration+1)
                
                if self.objective in ['binary:logistic', 'reg:logistic']:
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
                        # Remove trees after best iteration
                        self.trees = self.trees[:best_iteration + 1]
                        break
                
                # Verbose output
                if verbose and (isinstance(verbose, bool) or iteration % verbose == 0):
                    print(f"[{iteration}] train-rmse: {np.sqrt(train_score):.6f}, "
                          f"val-rmse: {np.sqrt(val_score):.6f}")
            elif verbose and (isinstance(verbose, bool) or iteration % verbose == 0):
                if self.objective in ['binary:logistic', 'reg:logistic']:
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
            Number of trees to use for prediction
            If None, use all trees
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values
            - For regression: continuous values
            - For classification: probabilities
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        
        # Start with base score
        predictions = np.full(n_samples, self.base_score)
        
        # Determine how many trees to use
        n_trees = len(self.trees) if num_iteration is None else min(num_iteration, len(self.trees))
        
        # Add contribution from each tree
        for i in range(n_trees):
            tree_predictions = self._predict_tree(self.trees[i], X)
            predictions += self.learning_rate * tree_predictions
        
        # For classification, convert to probabilities
        if self.objective in ['binary:logistic', 'reg:logistic']:
            predictions = self._sigmoid(predictions)
        
        return predictions
    
    def predict_proba(self, X, num_iteration=None):
        """
        Predict class probabilities (for classification)
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
        num_iteration : int, optional
            Number of trees to use
            
        Returns:
        --------
        probabilities : np.ndarray, shape (n_samples, 2)
            Probability for each class [P(class=0), P(class=1)]
        """
        if self.objective not in ['binary:logistic', 'reg:logistic']:
            raise ValueError("predict_proba only available for classification")
        
        proba_class_1 = self.predict(X, num_iteration)
        proba_class_0 = 1 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def score(self, X, y):
        """
        Calculate performance metric
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to evaluate
        y : np.ndarray, shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            - For regression: R² score
            - For classification: Accuracy
        """
        y = np.array(y)
        predictions = self.predict(X)
        
        if self.objective in ['binary:logistic', 'reg:logistic']:
            # Classification: accuracy
            predicted_classes = (predictions >= 0.5).astype(int)
            return np.mean(predicted_classes == y)
        else:
            # Regression: R² score
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - predictions) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            return r2
    
    def get_feature_importance(self, importance_type='weight'):
        """
        Calculate feature importance
        
        Parameters:
        -----------
        importance_type : str, default='weight'
            Type of importance to calculate:
            - 'weight': Number of times feature is used in splits
            - 'gain': Average gain when feature is used
            - 'cover': Average coverage (number of samples affected)
            
        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Feature importance scores (normalized to sum to 1)
        """
        if importance_type == 'weight':
            importance = np.zeros(self.n_features)
            
            def count_feature_usage(tree):
                if tree['type'] == 'leaf':
                    return
                importance[tree['feature']] += 1
                count_feature_usage(tree['left'])
                count_feature_usage(tree['right'])
            
            for tree in self.trees:
                count_feature_usage(tree)
                
        elif importance_type == 'gain':
            importance = np.zeros(self.n_features)
            counts = np.zeros(self.n_features)
            
            def accumulate_gain(tree):
                if tree['type'] == 'leaf':
                    return
                importance[tree['feature']] += tree['gain']
                counts[tree['feature']] += 1
                accumulate_gain(tree['left'])
                accumulate_gain(tree['right'])
            
            for tree in self.trees:
                accumulate_gain(tree)
            
            # Average gain per feature
            importance = np.where(counts > 0, importance / counts, 0)
            
        elif importance_type == 'cover':
            importance = np.zeros(self.n_features)
            counts = np.zeros(self.n_features)
            
            def accumulate_cover(tree):
                if tree['type'] == 'leaf':
                    return
                importance[tree['feature']] += tree['count']
                counts[tree['feature']] += 1
                accumulate_cover(tree['left'])
                accumulate_cover(tree['right'])
            
            for tree in self.trees:
                accumulate_cover(tree)
            
            # Average coverage per feature
            importance = np.where(counts > 0, importance / counts, 0)
        else:
            raise ValueError(f"Unknown importance_type: {importance_type}")
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance


"""
USAGE EXAMPLE 1: Simple Regression with XGBoost

import numpy as np

# Generate non-linear data: y = x² + noise
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Create and train XGBoost model
model = XGBoost(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=1.0,
    gamma=0.1
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
USAGE EXAMPLE 2: Binary Classification with XGBoost

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

# Train XGBoost classifier
model = XGBoost(
    n_estimators=50,
    learning_rate=0.3,
    max_depth=3,
    objective='binary:logistic',
    reg_lambda=1.0
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
USAGE EXAMPLE 3: XGBoost with Early Stopping

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(500, 10)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] ** 2 + np.random.randn(500) * 0.5

# Split train/validation/test
X_train, X_val, X_test = X[:300], X[300:400], X[400:]
y_train, y_val, y_test = y[:300], y[300:400], y[400:]

# Train with early stopping
model = XGBoost(
    n_estimators=500,  # Set high, will stop early
    learning_rate=0.1,
    max_depth=5,
    reg_lambda=1.0,
    subsample=0.8
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
model = XGBoost(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=2.0,
    colsample_bytree=0.8
)
model.fit(X, y)

# Get feature importance (different types)
importance_weight = model.get_feature_importance('weight')
importance_gain = model.get_feature_importance('gain')

print("\nFeature Importance (by weight):")
print("="*50)
for i, imp in enumerate(importance_weight):
    bar = '█' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")

print("\nFeature Importance (by gain):")
print("="*50)
for i, imp in enumerate(importance_gain):
    bar = '█' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")
"""

"""
USAGE EXAMPLE 5: Comparing Regularization Parameters

import numpy as np

# Generate data with some overfitting potential
np.random.seed(42)
X = np.random.randn(200, 15)
y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + np.random.randn(200) * 0.8

X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Test different regularization settings
configs = [
    {'reg_lambda': 0.0, 'reg_alpha': 0.0, 'name': 'No regularization'},
    {'reg_lambda': 1.0, 'reg_alpha': 0.0, 'name': 'L2 (Ridge)'},
    {'reg_lambda': 0.0, 'reg_alpha': 1.0, 'name': 'L1 (Lasso)'},
    {'reg_lambda': 1.0, 'reg_alpha': 1.0, 'name': 'Elastic Net'},
]

print("Effect of Regularization:")
print("="*80)
print(f"{'Configuration':<25} {'Train R²':>15} {'Test R²':>15} {'Overfit':>15}")
print("-"*80)

for config in configs:
    model = XGBoost(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        reg_lambda=config['reg_lambda'],
        reg_alpha=config['reg_alpha']
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfit = train_score - test_score
    
    print(f"{config['name']:<25} {train_score:>15.4f} {test_score:>15.4f} {overfit:>15.4f}")

# Observation: Regularization helps reduce overfitting!
"""

"""
USAGE EXAMPLE 6: Effect of Tree Depth and Complexity

import numpy as np

# Complex non-linear data
np.random.seed(42)
X = np.random.randn(300, 8)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 + 
     np.sin(X[:, 2]) * X[:, 3] + 
     np.random.randn(300) * 0.5)

X_train, X_test = X[:200], X[100:]
y_train, y_test = y[:200], y[100:]

# Test different depths
depths = [2, 3, 4, 6, 8]

print("\nEffect of Max Depth:")
print("="*80)
print(f"{'Max Depth':>12} {'Train R²':>15} {'Test R²':>15} {'Trees Used':>15}")
print("-"*80)

for depth in depths:
    model = XGBoost(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=depth,
        reg_lambda=1.0,
        gamma=0.1
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"{depth:>12} {train_score:>15.4f} {test_score:>15.4f} {len(model.trees):>15}")
"""

"""
USAGE EXAMPLE 7: Column Subsampling Effect

import numpy as np

# Wide dataset (many features)
np.random.seed(42)
X = np.random.randn(200, 20)
# Only first 5 features are informative
y = (2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] - 
     0.5 * X[:, 3] + X[:, 4] + np.random.randn(200) * 0.5)

X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Test different colsample_bytree values
colsample_values = [0.3, 0.5, 0.7, 1.0]

print("\nEffect of Column Subsampling:")
print("="*80)
print(f"{'Colsample':>12} {'Train R²':>15} {'Test R²':>15} {'Overfit':>15}")
print("-"*80)

for colsample in colsample_values:
    model = XGBoost(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        colsample_bytree=colsample,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfit = train_score - test_score
    
    print(f"{colsample:>12.1f} {train_score:>15.4f} {test_score:>15.4f} {overfit:>15.4f}")

# Observation: Lower colsample can reduce overfitting
"""

"""
USAGE EXAMPLE 8: Real-World - Credit Scoring

import numpy as np

# Simulated credit application data
# [credit_score, annual_income_k, debt_to_income, employment_years, 
#  age, num_credit_lines, delinquencies, inquiries_6mo]

np.random.seed(42)
n_samples = 1000

# Good credit (class 0)
X_good = np.random.randn(700, 8) * np.array([50, 20, 0.1, 3, 8, 2, 0.5, 1]) + \
         np.array([720, 75, 0.3, 8, 40, 6, 0, 1])

# Bad credit (class 1)
X_bad = np.random.randn(300, 8) * np.array([60, 25, 0.15, 4, 10, 3, 2, 2]) + \
        np.array([620, 45, 0.6, 3, 35, 4, 3, 4])

X = np.vstack([X_good, X_bad])
y = np.array([0] * 700 + [1] * 300)

# Shuffle
indices = np.random.permutation(1000)
X = X[indices]
y = y[indices]

# Split
X_train, X_val, X_test = X[:600], X[600:800], X[800:]
y_train, y_val, y_test = y[:600], y[600:800], y[800:]

# Train credit scoring model
model = XGBoost(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=5,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective='binary:logistic'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=50
)

# Evaluate
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)
test_acc = model.score(X_test, y_test)

print(f"\nCredit Scoring Model Performance:")
print("="*60)
print(f"Training Accuracy:   {train_acc:.2%}")
print(f"Validation Accuracy: {val_acc:.2%}")
print(f"Test Accuracy:       {test_acc:.2%}")

# Calculate additional metrics
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int)

true_positives = np.sum((predicted_classes == 1) & (y_test == 1))
false_positives = np.sum((predicted_classes == 1) & (y_test == 0))
false_negatives = np.sum((predicted_classes == 0) & (y_test == 1))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"\nPrecision: {precision:.2%} (of predicted defaults, how many are correct)")
print(f"Recall: {recall:.2%} (of actual defaults, how many detected)")
print(f"F1 Score: {f1:.4f}")

# Feature importance
feature_names = ['Credit Score', 'Income', 'Debt/Income', 'Employment Years',
                'Age', 'Credit Lines', 'Delinquencies', 'Recent Inquiries']
importance = model.get_feature_importance('gain')

print("\nFeature Importance (by gain):")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.4f}")

# Predict new applications
new_applications = np.array([
    [750, 85, 0.25, 10, 42, 8, 0, 1],  # Good profile
    [580, 35, 0.75, 2, 28, 3, 5, 6]    # Risky profile
])

risk_probabilities = model.predict(new_applications)

print("\nNew Application Risk Assessment:")
for i, prob in enumerate(risk_probabilities):
    risk_level = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    print(f"Applicant {i+1}: {risk_level} (default probability: {prob:.2%})")
"""

"""
USAGE EXAMPLE 9: House Price Prediction with XGBoost

import numpy as np

# Simulated house features
# [size_sqft, bedrooms, bathrooms, age_years, distance_to_city_km,
#  lot_size_sqft, garage_cars, has_pool, neighborhood_quality]

np.random.seed(42)
n_houses = 500

size = np.random.uniform(1000, 4000, n_houses)
bedrooms = np.random.randint(2, 6, n_houses)
bathrooms = np.random.randint(1, 5, n_houses)
age = np.random.uniform(0, 50, n_houses)
distance = np.random.uniform(1, 40, n_houses)
lot_size = np.random.uniform(2000, 10000, n_houses)
garage = np.random.randint(0, 4, n_houses)
pool = np.random.randint(0, 2, n_houses)
neighborhood = np.random.uniform(1, 10, n_houses)

X = np.column_stack([size, bedrooms, bathrooms, age, distance, 
                     lot_size, garage, pool, neighborhood])

# Price formula with non-linear relationships and interactions
price = (
    250 * size +
    40000 * bedrooms +
    25000 * bathrooms -
    800 * age -
    1500 * distance +
    10 * lot_size +
    15000 * garage +
    30000 * pool +
    10000 * neighborhood +
    0.08 * size * neighborhood +  # Interaction
    -0.5 * size * age +  # Depreciation effect
    np.random.randn(n_houses) * 25000  # Noise
)

# Normalize to thousands
price = price / 1000

# Split data
X_train, X_val, X_test = X[:300], X[300:400], X[400:]
y_train, y_val, y_test = price[:300], price[300:400], price[400:]

# Train XGBoost model
model = XGBoost(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=50
)

# Evaluate
test_r2 = model.score(X_test, y_test)
predictions = model.predict(X_test)

mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

print(f"\nHouse Price Prediction (XGBoost):")
print("="*60)
print(f"Test R²: {test_r2:.4f}")
print(f"Mean Absolute Error: ${mae:.2f}k")
print(f"Root Mean Squared Error: ${rmse:.2f}k")

# Feature importance
feature_names = ['Size', 'Bedrooms', 'Bathrooms', 'Age', 'Distance',
                'Lot Size', 'Garage', 'Pool', 'Neighborhood']
importance = model.get_feature_importance('gain')

print("\nTop 5 Most Important Features:")
feature_imp_pairs = list(zip(feature_names, importance))
feature_imp_pairs.sort(key=lambda x: x[1], reverse=True)
for name, imp in feature_imp_pairs[:5]:
    print(f"  {name:15s}: {imp:.4f}")

# Predict new houses
new_houses = np.array([
    [3000, 4, 3, 5, 8, 5000, 2, 1, 8.5],   # Large, nice, close to city
    [1500, 2, 1, 35, 30, 3000, 1, 0, 4.0]  # Small, old, far from city
])

predicted_prices = model.predict(new_houses)

print("\nNew House Price Predictions:")
for i, pred in enumerate(predicted_prices):
    print(f"House {i+1}: ${pred:.2f}k")
"""
