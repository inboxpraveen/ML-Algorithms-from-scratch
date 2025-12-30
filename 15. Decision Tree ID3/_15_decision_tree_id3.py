import numpy as np
import pandas as pd

class Node:
    """
    A node in the Decision Tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature        # Feature index/name to split on
        self.threshold = threshold    # Threshold for the split (categorical value for ID3)
        self.left = left              # Subtree for when feature value != threshold (not typically used in strict ID3 for non-binary, but useful structure)
        self.right = right            # Subtree for when feature value == threshold (not typically used in strict ID3)
        self.children = {}            # Dictionary of children nodes {feature_value: Node} - better for ID3
        self.value = value            # Leaf node value (class label)
        
    def is_leaf_node(self):
        return self.value is not None

class ID3DecisionTree:
    """
    ID3 (Iterative Dichotomiser 3) Decision Tree Implementation
    
    ID3 uses Entropy and Information Gain to build the decision tree.
    It handles categorical features effectively.
    
    Key Concepts:
    - Entropy: Measure of impurity or randomness
    - Information Gain: Reduction in entropy after splitting on a feature
    
    Algorithm:
    1. Calculate entropy of the dataset
    2. For each feature, calculate information gain
    3. Split dataset on the feature with highest information gain
    4. Recursively repeat for sub-datasets until stopping criteria met
    """
    
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X, y):
        """
        Build the decision tree.
        
        Parameters:
        X (numpy array or pandas DataFrame): Features
        y (numpy array or pandas Series): Target labels
        """
        # Convert to numpy arrays if necessary, but keep feature names if possible for better tree viz
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        
        X_data = np.array(X)
        y_data = np.array(y)
        
        self.root = self._grow_tree(X_data, y_data)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        # 1. Pure node (only one class)
        # 2. Max depth reached
        # 3. Not enough samples to split
        if (n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        feature_idx = self._best_split(X, y)
        
        # If no split gives info gain (e.g. all features identical), make leaf
        if feature_idx is None:
             leaf_value = self._most_common_label(y)
             return Node(value=leaf_value)

        # Create split node
        node = Node(feature=feature_idx)
        
        # For ID3, we create a branch for each unique value of the categorical feature
        # Note: Standard ID3 works best with categorical data. 
        # If data is continuous, it needs discretization (not covered here for pure ID3 simplicity).
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        for value in unique_values:
            # Partition data
            mask = (X[:, feature_idx] == value)
            X_subset = X[mask]
            y_subset = y[mask]
            
            if len(X_subset) > 0:
                child_node = self._grow_tree(X_subset, y_subset, depth + 1)
                node.children[value] = child_node
                
        return node
    
    def _best_split(self, X, y):
        best_gain = -1
        split_idx = None
        n_samples, n_features = X.shape
        
        # Calculate parent entropy
        parent_entropy = self._entropy(y)
        
        for feat_idx in range(n_features):
            # Calculate information gain
            feature_column = X[:, feat_idx]
            gain = self._information_gain(y, feature_column, parent_entropy)
            
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                
        return split_idx if best_gain > 0 else None
    
    def _information_gain(self, y, feature_column, parent_entropy):
        """
        IG(S, A) = Entropy(S) - Sum( (|Sv|/|S|) * Entropy(Sv) )
        """
        # Calculate weighted average entropy of children
        n = len(y)
        values, counts = np.unique(feature_column, return_counts=True)
        
        weighted_entropy = 0
        for value, count in zip(values, counts):
            # Subset of y where feature == value
            subset_y = y[feature_column == value]
            weighted_entropy += (count / n) * self._entropy(subset_y)
            
        return parent_entropy - weighted_entropy
        
    def _entropy(self, y):
        """
        Entropy = - Sum( p(x) * log2(p(x)) )
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10)) # 1e-10 to avoid log(0)
        return entropy
    
    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        # Handle empty case if needed, though stopping criteria usually prevents it
        if len(values) == 0: return None
        return values[np.argmax(counts)]

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
        
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
            
        feature_value = x[node.feature]
        
        # Navigate to the correct child
        if feature_value in node.children:
            return self._traverse_tree(x, node.children[feature_value])
        
        # If value not seen during training, return most common label (heuristic)
        # or simplified: return None or raise error. 
        # Here we just return None to indicate uncertainty or lack of path
        return None  

# ==========================================
# Example Usage with PlayTennis Dataset
# ==========================================
if __name__ == "__main__":
    print("Decision Tree ID3 Example: PlayTennis Dataset")
    print("---------------------------------------------")
    
    # Dataset
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    
    df = pd.DataFrame(data)
    
    # Features and Target
    X = df.drop('PlayTennis', axis=1)
    y = df['PlayTennis']
    
    print("Dataset Head:")
    print(df.head())
    print("\nTraining Model...")
    
    # Initialize and Train
    model = ID3DecisionTree(min_samples_split=2)
    model.fit(X, y)
    
    print("Model Trained.")
    
    # Test Prediction
    print("\nTesting Predictions:")
    test_samples = [
        ['Sunny', 'Cool', 'High', 'Strong'],   # Expected: No
        ['Overcast', 'Mild', 'Normal', 'Weak'] # Expected: Yes
    ]
    test_df = pd.DataFrame(test_samples, columns=X.columns)
    
    for i, sample in enumerate(test_samples):
        pred = model.predict(test_df.iloc[[i]])[0]
        print(f"Sample: {sample} -> Prediction: {pred}")
        
    # Verify exact matches for training data
    print("\nVerifying Training Accuracy:")
    y_pred = model.predict(X)
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Accuracy on Training Set: {accuracy * 100:.2f}%")
    
    if accuracy == 1.0:
        print("\nSUCCESS: Algorithm correctly learned the PlayTennis rules!")
    else:
        print("\nWARNING: Algorithm did not perfectly learn the training data.")
