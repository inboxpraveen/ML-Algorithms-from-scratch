import numpy as np
import sys
import os

# Add the Decision Trees folder to the path to import DecisionTree
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '6. Decision Trees'))

from _6_decision_trees import DecisionTree

class RandomForest:
    """
    Random Forest Implementation from Scratch
    
    Random Forest is an ensemble learning method that combines multiple decision trees
    to create a more robust and accurate model. It uses bootstrap sampling and random
    feature selection to create diverse trees that vote together.
    
    Key Ideas:
    - "Wisdom of the crowd" - Many trees are better than one
    - Each tree trained on random subset of data (bootstrap sampling)
    - Each tree considers only random features at splits
    - Final prediction by majority vote (classification) or average (regression)
    
    For classification: Majority vote across all trees
    For regression: Average prediction across all trees
    
    where:
        n_estimators = number of trees in the forest
        max_depth = maximum depth of each tree
        bootstrap = whether to use bootstrap sampling
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, bootstrap=True, criterion='gini',
                 task='classification', random_state=None):
        """
        Initialize the Random Forest model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
            More trees = better performance, but slower training
            Typical values: 50, 100, 200
        
        max_depth : int or None, default=None
            Maximum depth of each tree
            None = unlimited depth (trees grow until pure)
            Smaller values = less overfitting, faster training
        
        min_samples_split : int, default=2
            Minimum samples required to split a node
            Larger values = more conservative trees
        
        min_samples_leaf : int, default=1
            Minimum samples required at leaf node
            Larger values = smoother predictions
        
        bootstrap : bool, default=True
            Whether to use bootstrap sampling (sample with replacement)
            True = each tree sees different random subset of data
            False = all trees see all data (not recommended)
        
        criterion : str, default='gini'
            Split quality measure for trees
            Classification: 'gini' or 'entropy'
            Regression: 'mse'
        
        task : str, default='classification'
            Type of prediction task
            Options: 'classification', 'regression'
        
        random_state : int or None, default=None
            Random seed for reproducibility
            Set to any number for consistent results
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.task = task
        self.random_state = random_state
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Store trees
        self.trees = []
        self.n_classes_ = None
    
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample (random sample with replacement)
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training features
        y : numpy array of shape (n_samples,)
            Training labels
            
        Returns:
        --------
        X_sample : numpy array
            Bootstrap sample of features
        y_sample : numpy array
            Bootstrap sample of labels
        """
        n_samples = len(X)
        
        if self.bootstrap:
            # Sample with replacement (bootstrap)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
        else:
            # Use all samples (no bootstrap)
            indices = np.arange(n_samples)
        
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """
        Train the Random Forest by building multiple decision trees
        
        Each tree is trained on a different bootstrap sample of the data.
        This creates diversity in the trees, reducing overfitting.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training features
        y : numpy array of shape (n_samples,)
            Training labels/values
        """
        X = np.array(X)
        y = np.array(y)
        
        # Store number of classes for classification
        if self.task == 'classification':
            self.n_classes_ = len(np.unique(y))
        
        # Build each tree in the forest
        self.trees = []
        for i in range(self.n_estimators):
            # Create a bootstrap sample for this tree
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Create and train a decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                task=self.task
            )
            tree.fit(X_sample, y_sample)
            
            # Add tree to forest
            self.trees.append(tree)
    
    def predict(self, X):
        """
        Make predictions using all trees in the forest
        
        For classification: Uses majority voting across all trees
        For regression: Uses average prediction across all trees
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted labels (classification) or values (regression)
        """
        X = np.array(X)
        
        # Get predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            tree_pred = tree.predict(X)
            tree_predictions.append(tree_pred)
        
        # Convert to numpy array: shape (n_estimators, n_samples)
        tree_predictions = np.array(tree_predictions)
        
        if self.task == 'classification':
            # Classification: Use majority voting
            predictions = []
            for i in range(len(X)):
                # Get all tree predictions for this sample
                sample_preds = tree_predictions[:, i]
                
                # Find most common prediction (mode)
                unique_preds, counts = np.unique(sample_preds, return_counts=True)
                majority_vote = unique_preds[np.argmax(counts)]
                predictions.append(majority_vote)
            
            return np.array(predictions)
        else:
            # Regression: Use average
            return np.mean(tree_predictions, axis=0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification tasks
        
        Returns the proportion of trees that predicted each class.
        Only available for classification.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        probabilities : numpy array of shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only works for classification tasks")
        
        X = np.array(X)
        
        # Get predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            tree_pred = tree.predict(X)
            tree_predictions.append(tree_pred)
        
        tree_predictions = np.array(tree_predictions)
        
        # Calculate probabilities
        probabilities = []
        for i in range(len(X)):
            sample_preds = tree_predictions[:, i]
            
            # Calculate proportion for each class
            class_probs = []
            for class_idx in range(self.n_classes_):
                prob = np.mean(sample_preds == class_idx)
                class_probs.append(prob)
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Calculate model performance score
        
        For classification: Accuracy (proportion of correct predictions)
        For regression: R² score (coefficient of determination)
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test features
        y : numpy array of shape (n_samples,)
            True labels/values
            
        Returns:
        --------
        score : float
            Performance score (0.0 to 1.0, higher is better)
        """
        predictions = self.predict(X)
        y = np.array(y)
        
        if self.task == 'classification':
            # Classification: Accuracy
            return np.mean(predictions == y)
        else:
            # Regression: R² score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot == 0:
                return 1.0
            
            return 1 - (ss_res / ss_tot)


"""
USAGE EXAMPLE 1: Simple Classification

import numpy as np

# Sample data: Loan approval based on [Age, Income ($k), Credit Score]
X_train = np.array([
    [25, 45, 650],   # Reject
    [35, 75, 720],   # Approve
    [45, 95, 780],   # Approve
    [30, 50, 600],   # Reject
    [40, 80, 750],   # Approve
    [50, 120, 800],  # Approve
    [28, 40, 580],   # Reject
    [42, 85, 740],   # Approve
    [32, 55, 680],   # Approve
    [27, 35, 560],   # Reject
])

# Labels: 0 = Reject, 1 = Approve
y_train = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])

# Create and train Random Forest
model = RandomForest(n_estimators=10, max_depth=3, task='classification', random_state=42)
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [38, 70, 700],   # Should approve
    [26, 35, 550],   # Should reject
    [48, 110, 790],  # Should approve
])

predictions = model.predict(X_test)
print("Predictions:", predictions)  # [1, 0, 1] = [Approve, Reject, Approve]

# Get probabilities
probabilities = model.predict_proba(X_test)
print("\nProbabilities:")
for i, probs in enumerate(probabilities):
    print(f"  Sample {i+1}: Reject={probs[0]:.2f}, Approve={probs[1]:.2f}")
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Iris Classification)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Random Forest
model = RandomForest(n_estimators=50, max_depth=5, task='classification', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Show predictions with probabilities
probabilities = model.predict_proba(X_test[:5])
print("\nFirst 5 Predictions:")
for i in range(5):
    print(f"  True: {data.target_names[y_test[i]]}, Predicted: {data.target_names[y_pred[i]]}")
    print(f"  Probabilities: Setosa={probabilities[i][0]:.2f}, "
          f"Versicolor={probabilities[i][1]:.2f}, Virginica={probabilities[i][2]:.2f}")
"""

"""
USAGE EXAMPLE 3: Random Forest for Regression

import numpy as np

# Sample data: House price prediction [Size (sq ft), Age (years), Bedrooms]
X_train = np.array([
    [1000, 5, 2],    # $200k
    [1500, 3, 3],    # $300k
    [1200, 10, 2],   # $220k
    [2000, 2, 4],    # $400k
    [1800, 7, 3],    # $350k
    [2500, 1, 4],    # $500k
    [900, 15, 2],    # $180k
    [1100, 8, 2],    # $210k
    [1400, 4, 3],    # $280k
    [2200, 3, 4],    # $420k
])

# Prices in thousands
y_train = np.array([200, 300, 220, 400, 350, 500, 180, 210, 280, 420])

# Create and train Random Forest for regression
model = RandomForest(n_estimators=20, max_depth=5, task='regression', random_state=42)
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [1300, 6, 2],    # Similar to training examples
    [2200, 2, 4],    # Larger, newer house
    [950, 12, 2],    # Smaller, older house
])

predictions = model.predict(X_test)
print("Predicted prices ($1000s):", predictions)

# Calculate R² score
r2_score = model.score(X_train, y_train)
print(f"R² Score on training data: {r2_score:.4f}")
"""

"""
USAGE EXAMPLE 4: Comparing Different Numbers of Trees

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try different numbers of trees
n_trees_values = [5, 10, 20, 50, 100]

print("Comparing Different Numbers of Trees:\n")
print(f"{'Trees':<10} {'Train Accuracy':<20} {'Test Accuracy':<20}")
print("-" * 50)

for n_trees in n_trees_values:
    model = RandomForest(n_estimators=n_trees, max_depth=10,
                        task='classification', random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"{n_trees:<10} {train_acc:<20.4f} {test_acc:<20.4f}")

# Observations:
# - More trees generally = better performance
# - Diminishing returns after ~50-100 trees
# - More trees = slower training but same prediction speed per tree
"""

"""
USAGE EXAMPLE 5: Random Forest vs Single Decision Tree

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import sys
import os

# Import Decision Tree
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '6. Decision Trees'))
from _6_decision_trees import DecisionTree

# Load wine dataset
data = load_wine()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Single Decision Tree
single_tree = DecisionTree(max_depth=10, task='classification')
single_tree.fit(X_train, y_train)
single_tree_acc = single_tree.score(X_test, y_test)

# Train Random Forest
forest = RandomForest(n_estimators=50, max_depth=10,
                     task='classification', random_state=42)
forest.fit(X_train, y_train)
forest_acc = forest.score(X_test, y_test)

print("Comparison: Single Tree vs Random Forest")
print("-" * 40)
print(f"Single Decision Tree Accuracy: {single_tree_acc:.4f}")
print(f"Random Forest Accuracy:        {forest_acc:.4f}")
print(f"Improvement:                   {(forest_acc - single_tree_acc):.4f}")

# Random Forest typically outperforms a single tree due to:
# - Reduced overfitting through ensemble averaging
# - Reduced variance through bootstrap sampling
# - Better generalization to unseen data
"""
