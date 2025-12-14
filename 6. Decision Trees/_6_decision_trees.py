import numpy as np

class DecisionTree:
    """
    Decision Tree Implementation from Scratch
    
    A Decision Tree is a supervised learning algorithm that makes predictions by
    recursively splitting data based on features, creating a tree-like structure
    of decisions.
    
    Key Idea: "Make decisions by asking a series of yes/no questions"
    
    For classification: Predict the most common class in each leaf
    For regression: Predict the average value in each leaf
    
    where:
        max_depth = maximum depth of the tree
        min_samples_split = minimum samples required to split a node
        criterion = measure of split quality (gini, entropy, mse)
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', task='classification'):
        """
        Initialize the Decision Tree model
        
        Parameters:
        -----------
        max_depth : int or None, default=None
            Maximum depth of the tree
            None = unlimited depth (grow until pure leaves)
            Larger depth = more complex tree, risk of overfitting
            Smaller depth = simpler tree, more generalization
        
        min_samples_split : int, default=2
            Minimum samples required to split an internal node
            Larger values = more conservative splitting
            Prevents creation of nodes with very few samples
        
        min_samples_leaf : int, default=1
            Minimum samples required to be at a leaf node
            Larger values = smoother predictions, more regularization
        
        criterion : str, default='gini'
            Function to measure split quality
            Classification: 'gini' or 'entropy'
            Regression: 'mse' (mean squared error)
        
        task : str, default='classification'
            Type of prediction task
            Options: 'classification', 'regression'
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.task = task
        self.tree = None
        self.n_features = None
        self.n_classes = None
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for classification
        
        Gini = 1 - Σ(p_i²) where p_i is proportion of class i
        
        Gini = 0: Pure node (all samples same class)
        Gini = 0.5: Maximum impurity for binary (50-50 split)
        
        Parameters:
        -----------
        y : numpy array
            Labels at this node
            
        Returns:
        --------
        gini : float
            Gini impurity value
        """
        if len(y) == 0:
            return 0
        
        # Calculate proportion of each class
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Gini impurity
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _entropy(self, y):
        """
        Calculate entropy for classification
        
        Entropy = -Σ(p_i × log2(p_i)) where p_i is proportion of class i
        
        Entropy = 0: Pure node (all samples same class)
        Entropy = 1: Maximum impurity for binary (50-50 split)
        
        Parameters:
        -----------
        y : numpy array
            Labels at this node
            
        Returns:
        --------
        entropy : float
            Entropy value
        """
        if len(y) == 0:
            return 0
        
        # Calculate proportion of each class
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Entropy (avoid log(0) by filtering out zeros)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _mse(self, y):
        """
        Calculate mean squared error for regression
        
        MSE = (1/n) × Σ(y_i - mean(y))²
        
        Parameters:
        -----------
        y : numpy array
            Values at this node
            
        Returns:
        --------
        mse : float
            Mean squared error
        """
        if len(y) == 0:
            return 0
        
        mean = np.mean(y)
        mse = np.mean((y - mean) ** 2)
        return mse
    
    def _calculate_impurity(self, y):
        """
        Calculate impurity based on criterion
        
        Parameters:
        -----------
        y : numpy array
            Labels or values at this node
            
        Returns:
        --------
        impurity : float
            Impurity measure
        """
        if self.task == 'classification':
            if self.criterion == 'gini':
                return self._gini_impurity(y)
            elif self.criterion == 'entropy':
                return self._entropy(y)
        else:  # regression
            return self._mse(y)
    
    def _information_gain(self, y, y_left, y_right):
        """
        Calculate information gain from a split
        
        Information Gain = Impurity(parent) - Weighted Average of Impurity(children)
        
        Parameters:
        -----------
        y : numpy array
            Labels/values at parent node
        y_left : numpy array
            Labels/values in left child
        y_right : numpy array
            Labels/values in right child
            
        Returns:
        --------
        gain : float
            Information gain from this split
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Weighted average of children impurity
        child_impurity = (n_left / n) * self._calculate_impurity(y_left) + \
                        (n_right / n) * self._calculate_impurity(y_right)
        
        # Information gain
        gain = parent_impurity - child_impurity
        return gain
    
    def _best_split(self, X, y):
        """
        Find the best split for a node
        
        Tests all possible splits and returns the one with highest information gain
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Features at this node
        y : numpy array of shape (n_samples,)
            Labels/values at this node
            
        Returns:
        --------
        best_split : dict or None
            Dictionary containing:
            - feature_index: Index of feature to split on
            - threshold: Value to split at
            - gain: Information gain from this split
            Returns None if no valid split found
        """
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None
        
        best_gain = -1
        best_split = None
        
        # Try splitting on each feature
        for feature_index in range(n_features):
            # Get unique values for this feature
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            
            # Try each unique value as a threshold
            for threshold in unique_values:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold
                
                # Check minimum samples per leaf
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                y_left = y[left_mask]
                y_right = y[right_mask]
                gain = self._information_gain(y, y_left, y_right)
                
                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'gain': gain
                    }
        
        return best_split
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Features at this node
        y : numpy array of shape (n_samples,)
            Labels/values at this node
        depth : int
            Current depth in the tree
            
        Returns:
        --------
        node : dict
            Dictionary representing the node:
            - If leaf: {'type': 'leaf', 'value': prediction_value}
            - If internal: {'type': 'internal', 'feature_index': int, 
                           'threshold': float, 'left': node, 'right': node}
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        # 1. Maximum depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return self._create_leaf(y)
        
        # 2. All samples have same label (pure node)
        if len(np.unique(y)) == 1:
            return self._create_leaf(y)
        
        # 3. Not enough samples to split
        if n_samples < self.min_samples_split:
            return self._create_leaf(y)
        
        # Find best split
        best_split = self._best_split(X, y)
        
        # 4. No valid split found
        if best_split is None:
            return self._create_leaf(y)
        
        # Split the data
        feature_index = best_split['feature_index']
        threshold = best_split['threshold']
        
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        # Return internal node
        return {
            'type': 'internal',
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _create_leaf(self, y):
        """
        Create a leaf node with prediction value
        
        Parameters:
        -----------
        y : numpy array
            Labels/values at this leaf
            
        Returns:
        --------
        leaf : dict
            Leaf node dictionary
        """
        if self.task == 'classification':
            # Most common class
            unique_labels, counts = np.unique(y, return_counts=True)
            value = unique_labels[np.argmax(counts)]
        else:
            # Average value
            value = np.mean(y)
        
        return {
            'type': 'leaf',
            'value': value,
            'n_samples': len(y)
        }
    
    def fit(self, X, y):
        """
        Build the decision tree from training data
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Target values
        """
        self.n_features = X.shape[1]
        
        if self.task == 'classification':
            self.n_classes = len(np.unique(y))
        
        # Build the tree recursively
        self.tree = self._build_tree(X, y)
    
    def _predict_single(self, x, node):
        """
        Predict for a single sample by traversing the tree
        
        Parameters:
        -----------
        x : numpy array of shape (n_features,)
            Single sample
        node : dict
            Current node in the tree
            
        Returns:
        --------
        prediction : int or float
            Predicted label or value
        """
        # If leaf node, return the value
        if node['type'] == 'leaf':
            return node['value']
        
        # Otherwise, traverse to left or right child
        if x[node['feature_index']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
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
            Predicted labels or values
        """
        predictions = []
        for x in X:
            prediction = self._predict_single(x, self.tree)
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
            True labels or values
            
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
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            return r2_score
    
    def get_depth(self, node=None):
        """
        Get the depth of the tree
        
        Parameters:
        -----------
        node : dict, optional
            Current node (uses root if None)
            
        Returns:
        --------
        depth : int
            Depth of the tree
        """
        if node is None:
            node = self.tree
        
        if node['type'] == 'leaf':
            return 1
        
        left_depth = self.get_depth(node['left'])
        right_depth = self.get_depth(node['right'])
        
        return 1 + max(left_depth, right_depth)
    
    def get_n_leaves(self, node=None):
        """
        Get the number of leaves in the tree
        
        Parameters:
        -----------
        node : dict, optional
            Current node (uses root if None)
            
        Returns:
        --------
        n_leaves : int
            Number of leaf nodes
        """
        if node is None:
            node = self.tree
        
        if node['type'] == 'leaf':
            return 1
        
        left_leaves = self.get_n_leaves(node['left'])
        right_leaves = self.get_n_leaves(node['right'])
        
        return left_leaves + right_leaves


"""
USAGE EXAMPLE 1: Simple Classification

import numpy as np

# Sample data: Predicting if a customer will buy (1) or not (0)
# Features: [age, income_in_thousands]
X_train = np.array([
    [25, 30],   # Young, low income → No
    [45, 80],   # Middle-aged, high income → Yes
    [35, 50],   # Middle-aged, medium income → Yes
    [20, 25],   # Young, low income → No
    [50, 90],   # Older, high income → Yes
    [30, 35],   # Young, low income → No
    [40, 70],   # Middle-aged, high income → Yes
    [22, 28],   # Young, low income → No
])

# Labels: 0 = No purchase, 1 = Purchase
y_train = np.array([0, 1, 1, 0, 1, 0, 1, 0])

# Create and train the model
model = DecisionTree(max_depth=3, criterion='gini', task='classification')
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [28, 32],   # Young, low income → Should predict No
    [42, 75],   # Middle-aged, high income → Should predict Yes
    [55, 95]    # Older, high income → Should predict Yes
])

predictions = model.predict(X_test)
print("Predicted classes:", predictions)
# Output: [0, 1, 1] (No, Yes, Yes)

# Get tree statistics
print(f"\nTree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Iris Classification)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree
model = DecisionTree(max_depth=5, criterion='gini', task='classification')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Display predictions for first 5 test samples
print("\nFirst 5 predictions:")
for i in range(5):
    print(f"  Sample {i+1}: True={data.target_names[y_test[i]]}, "
          f"Predicted={data.target_names[y_pred[i]]}")

# Tree statistics
print(f"\nTree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")
"""

"""
USAGE EXAMPLE 3: Decision Tree for Regression

import numpy as np

# Sample data: Predicting house price based on size and number of rooms
X_train = np.array([
    [1000, 2],   # 1000 sq ft, 2 rooms → $200k
    [1500, 3],   # 1500 sq ft, 3 rooms → $300k
    [1200, 2],   # 1200 sq ft, 2 rooms → $220k
    [2000, 4],   # 2000 sq ft, 4 rooms → $400k
    [1800, 3],   # 1800 sq ft, 3 rooms → $350k
    [2500, 4],   # 2500 sq ft, 4 rooms → $500k
    [900, 2],    # 900 sq ft, 2 rooms → $180k
    [1100, 2],   # 1100 sq ft, 2 rooms → $210k
])

# Prices in thousands
y_train = np.array([200, 300, 220, 400, 350, 500, 180, 210])

# Create and train the model for regression
model = DecisionTree(max_depth=4, criterion='mse', task='regression')
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [1300, 2],   # Similar to training examples
    [2200, 4],   # Larger house
    [950, 2]     # Smaller house
])

predictions = model.predict(X_test)
print("Predicted prices ($1000s):", predictions)

# Calculate R² score
r2_score = model.score(X_train, y_train)
print(f"\nR² Score: {r2_score:.4f}")

# Tree statistics
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")
"""

"""
USAGE EXAMPLE 4: Comparing Different Max Depths

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different max_depth values
depths = [2, 3, 5, 7, 10, None]

print("Comparing Different Max Depths:\n")
print(f"{'Max Depth':<12} {'Train Acc':<12} {'Test Acc':<12} {'Tree Depth':<12} {'N Leaves':<12}")
print("-" * 60)

for depth in depths:
    model = DecisionTree(max_depth=depth, criterion='gini', task='classification')
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    tree_depth = model.get_depth()
    n_leaves = model.get_n_leaves()
    
    depth_str = str(depth) if depth is not None else "None"
    print(f"{depth_str:<12} {train_acc:<12.4f} {test_acc:<12.4f} {tree_depth:<12} {n_leaves:<12}")

# Observations:
# - Shallow trees (depth 2-3): Lower train accuracy, better generalization
# - Deep trees (depth 10+): High train accuracy, may overfit
# - Unlimited depth: Perfect training fit, often overfits
"""

"""
USAGE EXAMPLE 5: Comparing Gini vs Entropy

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load wine dataset
data = load_wine()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compare criteria
criteria = ['gini', 'entropy']

print("Comparing Gini vs Entropy:\n")
print(f"{'Criterion':<15} {'Train Accuracy':<20} {'Test Accuracy':<20}")
print("-" * 55)

for criterion in criteria:
    model = DecisionTree(max_depth=5, criterion=criterion, task='classification')
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"{criterion:<15} {train_acc:<20.4f} {test_acc:<20.4f}")

# Both Gini and Entropy typically produce similar results
# Gini: Slightly faster to compute
# Entropy: More theoretically grounded in information theory
"""
