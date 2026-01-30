import numpy as np

class GradientBoosting:
    """
    Gradient Boosting Implementation from Scratch
    
    Gradient Boosting is an ensemble learning algorithm that builds models sequentially,
    where each new model corrects errors made by the previous models by fitting to the
    negative gradient (residuals) of the loss function.
    
    Key Idea: "Train models sequentially to correct the errors of previous models"
    
    Use Cases:
    - Regression: House price prediction, sales forecasting
    - Classification: Customer churn, fraud detection, disease prediction
    - Ranking: Search engines, recommendation systems
    - Feature Selection: Identifying important variables
    
    Key Concepts:
        Loss Function: Measures how far predictions are from true values
        Gradient: Direction of steepest descent for the loss function
        Weak Learner: Simple model (typically decision tree) that fits the gradients
        Learning Rate: Controls the contribution of each model
        Sequential Learning: Each model improves upon the ensemble
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, 
                 loss='mse', subsample=1.0):
        """
        Initialize the Gradient Boosting model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of boosting stages (trees) to train
            - More estimators: Better training fit, longer training, risk overfitting
            - Fewer estimators: Faster training, may underfit
            Typical values: 100-500 for small datasets, 500-1000+ for large datasets
            
        learning_rate : float, default=0.1
            Shrinks the contribution of each tree
            - Lower values need more estimators but generalize better
            - Range: 0.01 to 0.3
            Typical: 0.1 is a good default, 0.01-0.05 for large datasets
            
        max_depth : int, default=3
            Maximum depth of each decision tree
            - Deeper trees: Can capture complex patterns, risk overfitting
            - Shallow trees: More regularization, better generalization
            Typical values: 3-8 (3-5 recommended for most cases)
            
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node
            - Higher values prevent overfitting
            - Lower values allow more complex trees
            Typical values: 2-20
            
        loss : str, default='mse'
            Loss function to optimize
            - 'mse': Mean Squared Error (for regression)
            - 'mae': Mean Absolute Error (for robust regression)
            - 'log_loss': Logistic loss (for binary classification)
            
        subsample : float, default=1.0
            Fraction of samples to use for training each tree
            - < 1.0 introduces randomness (stochastic gradient boosting)
            - Helps prevent overfitting
            - Typical values: 0.5-1.0
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.subsample = subsample
        self.trees = []
        self.init_prediction = None
        
    def _mse_loss(self, y_true, y_pred):
        """Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def _mse_gradient(self, y_true, y_pred):
        """Gradient of MSE: negative residuals"""
        return y_pred - y_true
    
    def _mae_gradient(self, y_true, y_pred):
        """Gradient of MAE: sign of residuals"""
        return np.sign(y_pred - y_true)
    
    def _log_loss_gradient(self, y_true, y_pred):
        """Gradient of log loss for binary classification"""
        # Sigmoid of predictions
        proba = 1 / (1 + np.exp(-y_pred))
        return proba - y_true
    
    def _get_gradient(self, y_true, y_pred):
        """Calculate gradient based on loss function"""
        if self.loss == 'mse':
            return self._mse_gradient(y_true, y_pred)
        elif self.loss == 'mae':
            return self._mae_gradient(y_true, y_pred)
        elif self.loss == 'log_loss':
            return self._log_loss_gradient(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
    
    def _create_decision_tree(self, X, y, depth=0):
        """
        Create a regression tree (decision tree for continuous targets)
        
        This is a simplified decision tree that predicts the mean value at each leaf.
        It recursively splits data to minimize variance.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target values (gradients to fit)
        depth : int
            Current depth of the tree
            
        Returns:
        --------
        tree : dict
            Dictionary representing the tree structure:
            - 'type': 'leaf' or 'split'
            - For leaf: 'value' (prediction value)
            - For split: 'feature', 'threshold', 'left', 'right'
        """
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            # Create leaf node with mean value
            return {
                'type': 'leaf',
                'value': np.mean(y)
            }
        
        # Find best split
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        current_variance = np.var(y) * n_samples
        
        # Try all features
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try all thresholds
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate variance reduction
                left_variance = np.var(y[left_mask]) * np.sum(left_mask)
                right_variance = np.var(y[right_mask]) * np.sum(right_mask)
                
                gain = current_variance - (left_variance + right_variance)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_mask
                    best_right_indices = right_mask
        
        # If no good split found, create leaf
        if best_gain <= 0:
            return {
                'type': 'leaf',
                'value': np.mean(y)
            }
        
        # Create split node
        left_tree = self._create_decision_tree(
            X[best_left_indices], 
            y[best_left_indices], 
            depth + 1
        )
        
        right_tree = self._create_decision_tree(
            X[best_right_indices], 
            y[best_right_indices], 
            depth + 1
        )
        
        return {
            'type': 'split',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _predict_tree(self, tree, X):
        """
        Make predictions using a decision tree
        
        Parameters:
        -----------
        tree : dict
            Tree structure
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values
        """
        if tree['type'] == 'leaf':
            return np.full(len(X), tree['value'])
        
        # Split predictions based on threshold
        feature_values = X[:, tree['feature']]
        left_mask = feature_values <= tree['threshold']
        
        predictions = np.zeros(len(X))
        
        if np.sum(left_mask) > 0:
            predictions[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        if np.sum(~left_mask) > 0:
            predictions[~left_mask] = self._predict_tree(tree['right'], X[~left_mask])
        
        return predictions
    
    def fit(self, X, y):
        """
        Train the Gradient Boosting model
        
        Algorithm:
        1. Initialize predictions with mean (or log-odds for classification)
        2. For t = 1 to n_estimators:
           a. Calculate negative gradient (pseudo-residuals)
           b. Sample subset of data if subsample < 1.0
           c. Fit a tree to the negative gradient
           d. Update predictions: F(x) = F(x) + learning_rate × tree(x)
        3. Final model: Sum of all trees with learning rate scaling
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data
        y : np.ndarray or list, shape (n_samples,)
            Target values
            - For regression: continuous values
            - For classification: 0 or 1 (binary)
            
        Returns:
        --------
        self : GradientBoosting
            Fitted model
        """
        # Convert to numpy arrays
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Initialize predictions
        if self.loss == 'log_loss':
            # For classification, initialize with log-odds
            p = np.mean(y)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
            self.init_prediction = np.log(p / (1 - p))
        else:
            # For regression, initialize with mean
            self.init_prediction = np.mean(y)
        
        # Current predictions (start with initialization)
        current_predictions = np.full(n_samples, self.init_prediction)
        
        self.trees = []
        
        # Train trees sequentially
        for i in range(self.n_estimators):
            # Calculate negative gradient (pseudo-residuals)
            gradients = -self._get_gradient(y, current_predictions)
            
            # Subsample data
            if self.subsample < 1.0:
                sample_size = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
                gradients_sample = gradients[indices]
            else:
                X_sample = X
                gradients_sample = gradients
            
            # Fit tree to negative gradient
            tree = self._create_decision_tree(X_sample, gradients_sample)
            self.trees.append(tree)
            
            # Update predictions for all samples
            tree_predictions = self._predict_tree(tree, X)
            current_predictions += self.learning_rate * tree_predictions
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Combines initial prediction with all trees:
        F(x) = F_0 + learning_rate × Σ tree_i(x)
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values
            - For regression: continuous values
            - For classification: probabilities after sigmoid
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        
        # Start with initial prediction
        predictions = np.full(n_samples, self.init_prediction)
        
        # Add contribution from each tree
        for tree in self.trees:
            tree_predictions = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_predictions
        
        # For classification, convert to probabilities
        if self.loss == 'log_loss':
            predictions = 1 / (1 + np.exp(-predictions))
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities (for classification)
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        probabilities : np.ndarray, shape (n_samples, 2)
            Probability for each class [P(class=0), P(class=1)]
        """
        if self.loss != 'log_loss':
            raise ValueError("predict_proba only available for classification (loss='log_loss')")
        
        proba_class_1 = self.predict(X)
        proba_class_0 = 1 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def score(self, X, y):
        """
        Calculate performance metric
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to evaluate
        y : np.ndarray or list, shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            - For regression: R² score (1.0 is perfect)
            - For classification: Accuracy (1.0 is perfect)
        """
        y = np.array(y)
        predictions = self.predict(X)
        
        if self.loss == 'log_loss':
            # Classification: accuracy
            predicted_classes = (predictions >= 0.5).astype(int)
            return np.mean(predicted_classes == y)
        else:
            # Regression: R² score
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - predictions) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            return r2
    
    def staged_predict(self, X):
        """
        Generate predictions after each boosting iteration
        
        Useful for:
        - Visualizing training progress
        - Finding optimal number of estimators
        - Detecting overfitting
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        staged_predictions : list of np.ndarray
            Predictions after each tree [pred_1, pred_2, ..., pred_T]
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        
        staged_predictions = []
        current_predictions = np.full(n_samples, self.init_prediction)
        
        for tree in self.trees:
            tree_predictions = self._predict_tree(tree, X)
            current_predictions = current_predictions + self.learning_rate * tree_predictions
            
            # For classification, convert to probabilities
            if self.loss == 'log_loss':
                staged_predictions.append(1 / (1 + np.exp(-current_predictions.copy())))
            else:
                staged_predictions.append(current_predictions.copy())
        
        return staged_predictions
    
    def staged_score(self, X, y):
        """
        Calculate performance after each boosting iteration
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to evaluate
        y : np.ndarray, shape (n_samples,)
            True values
            
        Returns:
        --------
        scores : list of float
            Performance metric after each tree
        """
        y = np.array(y)
        staged_predictions = self.staged_predict(X)
        
        scores = []
        for predictions in staged_predictions:
            if self.loss == 'log_loss':
                # Classification: accuracy
                predicted_classes = (predictions >= 0.5).astype(int)
                score = np.mean(predicted_classes == y)
            else:
                # Regression: R² score
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - predictions) ** 2)
                score = 1 - (ss_residual / ss_total)
            
            scores.append(score)
        
        return scores
    
    def get_feature_importance(self):
        """
        Calculate feature importance based on total variance reduction
        
        Importance is calculated as the sum of variance reduction
        from all splits on each feature across all trees.
        
        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Normalized feature importance (sums to 1)
            importance[i] = importance of feature i
        """
        importance = np.zeros(self.n_features)
        
        def _accumulate_importance(tree):
            """Recursively accumulate importance from tree"""
            if tree['type'] == 'leaf':
                return
            
            # Add importance for this split
            importance[tree['feature']] += 1
            
            # Recurse to children
            _accumulate_importance(tree['left'])
            _accumulate_importance(tree['right'])
        
        # Accumulate from all trees
        for tree in self.trees:
            _accumulate_importance(tree)
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance


"""
USAGE EXAMPLE 1: Simple Regression

import numpy as np

# Generate sample data: y = x^2 + noise
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Create and train model
model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
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
USAGE EXAMPLE 2: Binary Classification

import numpy as np

# Generate classification data
np.random.seed(42)

# Class 0: centered at (-2, -2)
X_class_0 = np.random.randn(100, 2) + np.array([-2, -2])
# Class 1: centered at (2, 2)
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

# Train classifier
model = GradientBoosting(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3,
    loss='log_loss'
)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.2%}")
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
USAGE EXAMPLE 3: Learning Curves and Overfitting Detection

import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.randn(100) * 0.2

X_train, X_test = X[:80], X[20:]
y_train, y_test = y[:80], y[20:]

# Train model
model = GradientBoosting(n_estimators=200, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Get learning curves
train_scores = model.staged_score(X_train, y_train)
test_scores = model.staged_score(X_test, y_test)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 201), train_scores, label='Training', linewidth=2)
plt.plot(range(1, 201), test_scores, label='Testing', linewidth=2)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Gradient Boosting Learning Curves', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

# Find optimal number of trees
optimal_n = np.argmax(test_scores) + 1
print(f"Optimal number of trees: {optimal_n}")
print(f"Best test R²: {test_scores[optimal_n-1]:.4f}")
"""

"""
USAGE EXAMPLE 4: Feature Importance Analysis

import numpy as np

# Create dataset with 10 features (only first 3 are informative)
np.random.seed(42)
n_samples = 300

# Informative features
X1 = np.random.randn(n_samples, 1)
X2 = np.random.randn(n_samples, 1)
X3 = np.random.randn(n_samples, 1)

# Non-informative features (noise)
X_noise = np.random.randn(n_samples, 7)

X = np.hstack([X1, X2, X3, X_noise])

# Target depends on first 3 features
y = 2 * X1.ravel() + 3 * X2.ravel() - X3.ravel() + np.random.randn(n_samples) * 0.5

# Train model
model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=4)
model.fit(X, y)

# Get feature importance
importance = model.get_feature_importance()

print("\nFeature Importance:")
print("="*50)
for i, imp in enumerate(importance):
    bar = '█' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")

# Expected: Features 0, 1, 2 have high importance, rest are low
"""

"""
USAGE EXAMPLE 5: Comparing Learning Rates

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(200, 5)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] + np.random.randn(200) * 0.5

X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Try different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.3]

print("Effect of Learning Rate:")
print("="*70)
print(f"{'Learning Rate':>15} {'n_estimators':>15} {'Train R²':>15} {'Test R²':>15}")
print("-"*70)

for lr in learning_rates:
    model = GradientBoosting(n_estimators=200, learning_rate=lr, max_depth=3)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"{lr:>15.2f} {200:>15} {train_score:>15.4f} {test_score:>15.4f}")

# Observation: Lower learning rate often gives better generalization
"""

"""
USAGE EXAMPLE 6: Effect of Tree Depth

import numpy as np

# Generate complex non-linear data
np.random.seed(42)
X = np.random.randn(200, 3)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 - X[:, 2] + 
     np.sin(X[:, 0]) + np.random.randn(200) * 0.3)

X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Try different max depths
depths = [1, 2, 3, 5, 8]

print("\nEffect of Tree Depth:")
print("="*70)
print(f"{'Max Depth':>15} {'Train R²':>15} {'Test R²':>15} {'Difference':>15}")
print("-"*70)

for depth in depths:
    model = GradientBoosting(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=depth
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    diff = train_score - test_score
    
    print(f"{depth:>15} {train_score:>15.4f} {test_score:>15.4f} {diff:>15.4f}")

# Observation: Shallow trees (3-5) often generalize best
"""

"""
USAGE EXAMPLE 7: Stochastic Gradient Boosting (Subsampling)

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(500, 8)
y = (X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + 
     np.random.randn(500) * 0.5)

X_train, X_test = X[:400], X[100:]
y_train, y_test = y[:400], y[100:]

# Compare different subsample ratios
subsample_ratios = [0.5, 0.7, 0.9, 1.0]

print("\nEffect of Subsampling:")
print("="*70)
print(f"{'Subsample':>15} {'Train R²':>15} {'Test R²':>15} {'Overfitting':>15}")
print("-"*70)

for subsample in subsample_ratios:
    model = GradientBoosting(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=subsample
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfit = train_score - test_score
    
    print(f"{subsample:>15.1f} {train_score:>15.4f} {test_score:>15.4f} {overfit:>15.4f}")

# Observation: Subsampling (< 1.0) can reduce overfitting
"""

"""
USAGE EXAMPLE 8: Real-World Application - House Price Prediction

import numpy as np

# Simulated house features
# [size_sqft, bedrooms, bathrooms, age_years, distance_to_city_km]
np.random.seed(42)

n_houses = 200

size = np.random.uniform(800, 3000, n_houses)
bedrooms = np.random.randint(1, 6, n_houses)
bathrooms = np.random.randint(1, 4, n_houses)
age = np.random.uniform(0, 50, n_houses)
distance = np.random.uniform(1, 30, n_houses)

X = np.column_stack([size, bedrooms, bathrooms, age, distance])

# Price formula (with non-linear relationships)
price = (
    300 * size +  # Base price per sqft
    50000 * bedrooms +
    30000 * bathrooms -
    1000 * age -
    2000 * distance +
    0.05 * size ** 1.5 +  # Non-linear size effect
    np.random.randn(n_houses) * 20000  # Noise
)

# Normalize price to thousands
price = price / 1000

# Split data
X_train, X_test = X[:160], X[40:]
y_train, y_test = price[:160], price[40:]

# Train model
model = GradientBoosting(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8
)
model.fit(X_train, y_train)

# Evaluate
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print("\nHouse Price Prediction Model:")
print("="*60)
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Calculate MAE and RMSE manually
predictions = model.predict(X_test)
mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

print(f"Mean Absolute Error: ${mae:.2f}k")
print(f"Root Mean Squared Error: ${rmse:.2f}k")

# Feature importance
feature_names = ['Size (sqft)', 'Bedrooms', 'Bathrooms', 'Age (years)', 'Distance (km)']
importance = model.get_feature_importance()

print("\nFeature Importance:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.4f}")

# Predict new houses
new_houses = np.array([
    [2500, 4, 3, 5, 10],   # Large, new, close to city
    [1200, 2, 1, 30, 25]   # Small, old, far from city
])

predicted_prices = model.predict(new_houses)

print("\nNew House Price Predictions:")
for i, pred in enumerate(predicted_prices):
    print(f"House {i+1}: ${pred:.2f}k")
"""

"""
USAGE EXAMPLE 9: Medical Diagnosis with Gradient Boosting

import numpy as np

# Patient features: [age, bmi, blood_pressure, cholesterol, glucose, smoking_years]
# Target: Disease risk score (0 = low risk, 1 = high risk)

np.random.seed(42)

# Generate synthetic patient data
n_patients = 400

# High-risk patients
high_risk_features = np.random.randn(200, 6) * np.array([10, 3, 15, 20, 15, 5]) + \
                     np.array([65, 32, 145, 220, 130, 20])
high_risk_labels = np.ones(200)

# Low-risk patients
low_risk_features = np.random.randn(200, 6) * np.array([12, 2, 10, 15, 10, 3]) + \
                    np.array([40, 24, 115, 180, 95, 2])
low_risk_labels = np.zeros(200)

X = np.vstack([high_risk_features, low_risk_features])
y = np.hstack([high_risk_labels, low_risk_labels])

# Shuffle
indices = np.random.permutation(400)
X = X[indices]
y = y[indices]

# Split
X_train, X_test = X[:300], X[100:]
y_train, y_test = y[:300], y[100:]

# Train diagnostic model
model = GradientBoosting(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    loss='log_loss',
    subsample=0.8
)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("\nDisease Risk Prediction Model:")
print("="*60)
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# Calculate additional metrics
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int)

true_positives = np.sum((predicted_classes == 1) & (y_test == 1))
false_positives = np.sum((predicted_classes == 1) & (y_test == 0))
false_negatives = np.sum((predicted_classes == 0) & (y_test == 1))
true_negatives = np.sum((predicted_classes == 0) & (y_test == 0))

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2%} (of predicted high-risk, how many are correct)")
print(f"Recall: {recall:.2%} (of actual high-risk, how many detected)")
print(f"F1 Score: {f1:.4f}")

# Feature importance
feature_names = ['Age', 'BMI', 'Blood Pressure', 'Cholesterol', 'Glucose', 'Smoking Years']
importance = model.get_feature_importance()

print("\nRisk Factor Importance:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.4f}")

# Assess new patients
new_patients = np.array([
    [70, 35, 155, 240, 140, 25],  # High risk profile
    [35, 22, 110, 170, 90, 0]      # Low risk profile
])

risk_probabilities = model.predict(new_patients)

print("\nNew Patient Risk Assessment:")
for i, prob in enumerate(risk_probabilities):
    risk_level = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    print(f"Patient {i+1}: {risk_level} (probability: {prob:.2%})")

# Note: For educational purposes only!
# Real medical diagnosis requires professional evaluation
"""

