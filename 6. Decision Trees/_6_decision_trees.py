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

    Use Cases:
    - Credit approval: transparent, auditable accept/reject rules
    - Medical triage: "fever > 100F AND cough -> likely flu" style protocols
    - Customer churn: flag at-risk accounts with rules a business team can read
    - Feature-interaction discovery: trees capture "A AND B" effects a linear model misses
    - Interpretable baseline: the model you fit before reaching for Random Forests

    Split Criterion (impurity I of a node, implemented in _gini_impurity /
    _entropy / _mse and dispatched by _calculate_impurity):
        Gini    = 1 - sum(p_i^2)                  (classification)
        Entropy = -sum(p_i * log2(p_i))           (classification)
        MSE     = (1/n) * sum((y_i - mean(y))^2)  (regression)

    Information Gain of a split (this is exactly what _best_split maximises,
    and it is computed in _information_gain):
        IG = I(parent) - (n_left/n) * I(left) - (n_right/n) * I(right)

    Candidate Thresholds:
        Like CART / scikit-learn, the thresholds tried are the MIDPOINTS
        between consecutive unique feature values, not the raw values:
            thresholds = (v_1 + v_2)/2, (v_2 + v_3)/2, ...
        Splitting at a raw value produces the same partition of the TRAINING
        set, but it parks the boundary exactly on a data point, so unseen
        samples that land between two observed values are routed to the wrong
        side. Midpoints put the boundary halfway into the gap instead.

    Simplifications vs. canonical CART:
        No post-pruning (cost-complexity / ccp_alpha), no surrogate splits for
        missing values, and no native categorical-feature handling. See the
        "Simplifications vs. Canonical CART" section of _6_decision_trees.md.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', task='classification'):
        """
        Initialize the Decision Tree model
        
        Parameters:
        -----------
        max_depth : int or None, default=None
            Maximum depth of the tree, counted in EDGES from the root
            (max_depth=1 means a root split and two leaves)
            None = unlimited depth (grow until pure leaves)
            Larger depth = more complex tree, risk of overfitting
            Smaller depth = simpler tree, more generalization
            Typical: 3-10; None only on small, clean data
            NOTE: this implementation builds and traverses the tree with plain
            Python recursion, so max_depth=None on large noisy data can exceed
            Python's default recursion limit (~1000). Set max_depth in that case.

        min_samples_split : int, default=2
            Minimum samples required to split an internal node
            Larger values = more conservative splitting
            Prevents creation of nodes with very few samples
            Typical: 2-20 (raise it when the tree is memorising noise)

        min_samples_leaf : int, default=1
            Minimum samples required to be at a leaf node
            Larger values = smoother predictions, more regularization
            Enforced when CHOOSING a split (candidates that would leave a child
            below this size are skipped) - it is not a post-hoc pruning step
            Typical: 1-10 (raise to 5+ on noisy data)

        criterion : str, default='gini'
            Function to measure split quality
            Classification: 'gini' or 'entropy'
            Regression: 'mse' (mean squared error, i.e. variance reduction)
            Typical: 'gini' for classification (cheaper, near-identical trees),
            'mse' for regression - it is the only regression criterion here

        task : str, default='classification'
            Type of prediction task
            Options: 'classification', 'regression'
            Typical: 'classification' (the default); switch to 'regression'
            and criterion='mse' together for continuous targets

        Raises:
        -------
        ValueError
            If task/criterion are not a valid pair, or if max_depth,
            min_samples_split or min_samples_leaf is out of range.
        """
        # Validate up front so a typo fails loudly here instead of surfacing as
        # a cryptic TypeError deep inside the recursion (e.g. task='rgression'
        # used to silently train a regressor).
        valid_criteria = {'classification': ('gini', 'entropy'),
                          'regression': ('mse',)}
        if task not in valid_criteria:
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}"
            )
        if criterion not in valid_criteria[task]:
            raise ValueError(
                f"criterion={criterion!r} is not valid for task={task!r}. "
                f"Valid choices: {valid_criteria[task]}"
            )
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be None or >= 1, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be >= 2, got {min_samples_split}"
            )
        if min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be >= 1, got {min_samples_leaf}"
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.task = task
        self.tree = None
        self.n_features = None
        self.n_classes = None
        self.classes_ = None
    
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
        
        # No epsilon is needed inside the log: np.unique(..., return_counts=True)
        # only ever reports classes that are PRESENT, so every count is >= 1 and
        # every probability is > 0. (An epsilon here would make a pure node score
        # a slightly NEGATIVE entropy instead of exactly 0.)
        # One cosmetic quirk to expect: a pure node computes -(1.0 * 0.0), which
        # is IEEE negative zero, so this returns -0.0 and prints as "-0.0". It
        # compares equal to 0.0 and every use of it downstream treats it as zero.
        entropy = -np.sum(probabilities * np.log2(probabilities))
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
        elif self.task == 'regression':
            if self.criterion == 'mse':
                return self._mse(y)

        # Unreachable when __init__'s validation is in force, but an explicit
        # error beats silently returning None (which used to surface as
        # "unsupported operand type(s) for *: 'float' and 'NoneType'").
        raise ValueError(
            f"Unknown criterion {self.criterion!r} for task {self.task!r}"
        )

    def _information_gain(self, y, y_left, y_right, parent_impurity=None):
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
        parent_impurity : float, optional
            I(parent), if it has already been computed. The parent is the same
            for every candidate split of a node, so _best_split computes it once
            and passes it in; leaving it None recomputes it (same answer, just
            slower). Purely a speed hint - it never changes the result.

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

        # Parent impurity: I(parent)
        if parent_impurity is None:
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

        Tests every (feature, threshold) candidate and returns the one with the
        highest information gain. Candidate thresholds are the midpoints between
        consecutive unique values of each feature, so a feature with k distinct
        values contributes k-1 candidates.

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

        # I(parent) is the same for every candidate split of this node, so
        # compute it once here instead of once per threshold inside the loop.
        parent_impurity = self._calculate_impurity(y)

        # Try splitting on each feature
        for feature_index in range(n_features):
            # Get unique values for this feature
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            # Candidate thresholds are the MIDPOINTS between consecutive unique
            # values (the CART / scikit-learn convention), not the raw values:
            #   values     20    22    25    30
            #   midpoints     21   23.5  27.5
            # Both conventions partition the TRAINING rows identically, but a
            # midpoint puts the boundary halfway into the gap, so an unseen
            # sample landing between two observed values is routed sensibly.
            # k unique values give k-1 midpoints, so a constant feature
            # (1 unique value) correctly offers no split at all.
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            # Try each midpoint as a threshold
            for threshold in thresholds:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                # Both children are normally non-empty: the midpoint of
                # a < b always lands in [a, b]. Two cases survive that - for
                # the LAST pair the midpoint can round up onto the largest
                # value, and for enormous values a + b overflows to inf.
                # This guard names them explicitly; the min_samples_leaf test
                # just below (always >= 1) would in fact catch both already.
                # What NEITHER catches is an interior midpoint that rounds onto
                # its upper neighbour: it silently repeats the next candidate's
                # partition. That needs feature values about 1 ULP apart - e.g.
                # [0, 1.0000000000000002, 1.0000000000000004, 5] gives three
                # candidate midpoints but only two distinct partitions.
                if n_left == 0 or n_right == 0:
                    continue

                # Check minimum samples per leaf
                if n_left < self.min_samples_leaf or \
                   n_right < self.min_samples_leaf:
                    continue

                # Calculate information gain
                y_left = y[left_mask]
                y_right = y[right_mask]
                gain = self._information_gain(y, y_left, y_right,
                                              parent_impurity=parent_impurity)

                # Update best split. The strict `>` means that when two splits
                # tie on gain, the FIRST one wins - i.e. the lower feature index,
                # and within a feature the lower threshold.
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
            - If leaf: {'type': 'leaf', 'value': prediction_value,
                        'n_samples': int,
                        'class_counts': array or None}   # counts only when
                                                         # task='classification'
            - If internal: {'type': 'internal', 'feature_index': int,
                           'threshold': float, 'gain': float,
                           'n_samples': int, 'left': node, 'right': node}

            'gain' and 'n_samples' on an internal node are what
            feature_importances_ later accumulates; 'class_counts' on a leaf is
            what predict_proba reads.
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
        
        # Return internal node. We keep the split's 'gain' and this node's
        # 'n_samples' so feature_importances_ can weight each split's impurity
        # decrease by the fraction of the data that reached it.
        return {
            'type': 'internal',
            'feature_index': feature_index,
            'threshold': threshold,
            'gain': best_split['gain'],
            'n_samples': n_samples,
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
            Leaf node dictionary with keys 'type', 'value', 'n_samples' and
            'class_counts' (the last is None for regression).
        """
        class_counts = None

        if self.task == 'classification':
            # Most common class
            unique_labels, counts = np.unique(y, return_counts=True)
            value = unique_labels[np.argmax(counts)]

            # Keep the FULL class histogram, not just the argmax. Scattering the
            # observed counts into a length-n_classes vector aligned with
            # self.classes_ gives every leaf the same column order, which is what
            # lets predict_proba return p(class) = count / n_samples.
            class_counts = np.zeros(len(self.classes_), dtype=float)
            positions = np.searchsorted(self.classes_, unique_labels)
            class_counts[positions] = counts
        else:
            # Average value
            value = np.mean(y)

        return {
            'type': 'leaf',
            'value': value,
            'n_samples': len(y),
            'class_counts': class_counts
        }
    
    def fit(self, X, y):
        """
        Build the decision tree from training data

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data (nested Python lists are accepted and converted)
        y : array-like of shape (n_samples,)
            Target values

        Returns:
        --------
        self : DecisionTree
            The fitted estimator, so calls can be chained:
            `DecisionTree(max_depth=3).fit(X, y).predict(X_new)`

        Attributes set (useful for introspection):
        ------------------------------------------
        tree : dict
            The root node of the fitted tree (nested dictionaries)
        n_features : int
            Number of columns seen during fit; predict() checks against it
        classes_ : numpy array or None
            Sorted unique labels; defines predict_proba's column order
            (None for regression)
        n_classes : int or None
            len(classes_) (None for regression)
        """
        # Accept Python lists as well as arrays, and fail with a readable
        # message instead of an AttributeError/IndexError from deep inside.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D of shape (n_samples, n_features), got "
                f"{X.ndim}-D with shape {X.shape}. For a single feature use "
                f"X.reshape(-1, 1)."
            )
        if len(X) != len(y):
            raise ValueError(
                f"X and y have different lengths: {len(X)} vs {len(y)}"
            )
        if len(y) == 0:
            raise ValueError("Cannot fit on an empty dataset")

        self.n_features = X.shape[1]

        if self.task == 'classification':
            # Sorted unique labels. Every leaf's class_counts vector is aligned
            # to this ordering, and predict_proba returns its columns in it.
            self.classes_ = np.unique(y)
            self.n_classes = len(self.classes_)
        else:
            self.classes_ = None
            self.n_classes = None

        # Build the tree recursively
        self.tree = self._build_tree(X, y)

        return self
    
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
    
    def _check_is_fitted(self):
        """Raise a clear error if the model has not been trained yet."""
        if self.tree is None:
            raise ValueError(
                "This DecisionTree is not fitted yet. Call fit(X, y) first."
            )

    def _validate_X(self, X):
        """Coerce X to a 2-D float array and check it against the fitted shape."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D of shape (n_samples, n_features), got "
                f"{X.ndim}-D with shape {X.shape}. For a single feature use "
                f"X.reshape(-1, 1)."
            )
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but this DecisionTree was "
                f"fitted with {self.n_features}"
            )
        return X

    def _find_leaf(self, x, node):
        """
        Walk one sample down the tree and return the LEAF NODE it lands in.

        Same traversal as _predict_single, but it hands back the whole leaf dict
        so predict_proba can read 'class_counts' rather than just 'value'.
        """
        while node['type'] != 'leaf':
            if x[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

    def predict(self, X):
        """
        Predict labels or values for samples

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on

        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted labels or values
        """
        self._check_is_fitted()
        X = self._validate_X(X)

        predictions = []
        for x in X:
            prediction = self._predict_single(x, self.tree)
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities (classification only)

        Each leaf stores the class histogram of the training samples that
        reached it, so the probability of class c for a sample is simply

            P(c | leaf) = class_counts[c] / sum(class_counts)

        These are empirical leaf frequencies, not calibrated probabilities: a
        deep tree with pure leaves returns only 0.0 and 1.0.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on

        Returns:
        --------
        probabilities : numpy array of shape (n_samples, n_classes)
            Column j is the probability of self.classes_[j]; each row sums to 1
        """
        if self.task != 'classification':
            raise ValueError(
                "predict_proba is only available for task='classification'"
            )

        self._check_is_fitted()
        X = self._validate_X(X)

        probabilities = np.zeros((len(X), self.n_classes))
        for i, x in enumerate(X):
            leaf = self._find_leaf(x, self.tree)
            counts = leaf['class_counts']
            probabilities[i] = counts / counts.sum()

        return probabilities
    
    def score(self, X, y):
        """
        Calculate performance score
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test data
        y : array-like of shape (n_samples,)
            True labels or values

        Returns:
        --------
        score : float
            Accuracy (classification) or R^2 score (regression)
        """
        self._check_is_fitted()
        y = np.asarray(y)
        predictions = self.predict(X)

        if self.task == 'classification':
            # Classification: Calculate accuracy
            accuracy = np.mean(predictions == y)
            return accuracy
        else:
            # Regression: Calculate R^2 = 1 - SS_res / SS_tot
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            # SS_tot == 0 means y is constant, so R^2 is undefined (0/0).
            # Follow sklearn: 1.0 if we predicted it exactly, else 0.0.
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0

            r2_score = 1 - (ss_res / ss_tot)
            return r2_score
    
    def get_depth(self, node=None):
        """
        Get the depth of the tree

        Depth is counted in EDGES, the same convention as max_depth and as
        scikit-learn's DecisionTreeClassifier.get_depth():
            a single leaf (no splits)     -> 0
            root split with two leaves    -> 1
        So a model fitted with max_depth=3 reports get_depth() <= 3.

        Parameters:
        -----------
        node : dict, optional
            Current node (uses root if None)

        Returns:
        --------
        depth : int
            Number of edges on the longest root-to-leaf path
        """
        if node is None:
            self._check_is_fitted()
            node = self.tree

        # A leaf is 0 edges away from itself.
        if node['type'] == 'leaf':
            return 0

        left_depth = self.get_depth(node['left'])
        right_depth = self.get_depth(node['right'])

        # One edge down to the deeper child.
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
            self._check_is_fitted()
            node = self.tree

        if node['type'] == 'leaf':
            return 1

        left_leaves = self.get_n_leaves(node['left'])
        right_leaves = self.get_n_leaves(node['right'])

        return left_leaves + right_leaves

    def _accumulate_importances(self, node, importances, n_total):
        """
        Add one node's weighted impurity decrease into `importances`, recursively.

        For an internal node t that split on feature f:
            importances[f] += (n_t / n_total) * gain_t
        where gain_t is the information gain _best_split already recorded for
        that split. Leaves contribute nothing - they do not split anything.
        """
        if node['type'] == 'leaf':
            return

        importances[node['feature_index']] += \
            (node['n_samples'] / n_total) * node['gain']

        self._accumulate_importances(node['left'], importances, n_total)
        self._accumulate_importances(node['right'], importances, n_total)

    @property
    def feature_importances_(self):
        """
        Impurity-based feature importance (Mean Decrease in Impurity, "MDI")

        A feature's importance is the total impurity it removed, summed over
        every node that split on it and weighted by how much data reached that
        node, then normalised to sum to 1:

            importance[f] = sum over nodes t splitting on f of
                                (n_t / n_total) * IG(t)
            importance    = importance / sum(importance)

        This is the same definition scikit-learn uses, which is why the two
        agree closely on the same fitted tree.

        Caveat worth knowing: MDI is biased towards high-cardinality features
        (a continuous feature offers many more candidate thresholds than a
        binary one), and correlated features split the credit between them.

        Returns:
        --------
        importances : numpy array of shape (n_features,)
            Non-negative, sums to 1 (all zeros if the tree is a single leaf)
        """
        self._check_is_fitted()

        importances = np.zeros(self.n_features)

        # A root that is already a leaf never split on anything.
        if self.tree['type'] == 'leaf':
            return importances

        self._accumulate_importances(self.tree, importances,
                                     self.tree['n_samples'])

        total = importances.sum()
        if total > 0:
            importances = importances / total

        return importances


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
# Output: [0 1 1]  (No, Yes, Yes)

# Class probabilities: each leaf reports the class mix of the training
# samples that landed in it. Both leaves here are pure, so the
# probabilities are all 0.0 / 1.0.
print("Class probabilities:\n", model.predict_proba(X_test))
# Output: [[1. 0.]
#          [0. 1.]
#          [0. 1.]]

# Get tree statistics
print(f"\nTree depth: {model.get_depth()}")
# Output: Tree depth: 1   (one split = one edge; both children are pure leaves)
print(f"Number of leaves: {model.get_n_leaves()}")
# Output: Number of leaves: 2

# Which feature did the work? Age and Income split this dataset EQUALLY
# well (both score IG = 0.5), and the tie goes to the lower feature index.
print("Feature importances:", model.feature_importances_)
# Output: Feature importances: [1. 0.]
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

# Class probabilities for the first 3 test samples
# Column j is the probability of model.classes_[j]
proba = model.predict_proba(X_test)
print("\nClass probabilities (first 3 samples):")
print("  " + "".join(f"{name:>14}" for name in data.target_names))
for i in range(3):
    print("  " + "".join(f"{p:14.3f}" for p in proba[i]))

# Which measurements actually drove the splits?
# feature_importances_ is the impurity-based (MDI) importance, the same
# quantity sklearn reports: each split's information gain, weighted by the
# fraction of data reaching it, summed per feature and normalised to 1.
print("\nFeature importances:")
for name, imp in zip(data.feature_names, model.feature_importances_):
    print(f"  {name:<20} {imp:.4f}")

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

# Calculate R2 score
# NOTE: this dataset is only 8 rows, so there is nothing left to hold out.
# The number below is an IN-SAMPLE R2 - it measures how well the tree FIT the
# training rows, NOT how well it generalises. A tree deep enough to isolate
# every training row always scores ~1.0 here. See USAGE EXAMPLE 4 for a proper
# train/test comparison.
r2_score = model.score(X_train, y_train)
print(f"\nR2 Score (in-sample): {r2_score:.4f}")

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

# Try different max_depth values.
# NOTE: depths 7, 10 and None all build the SAME tree on this dataset (the tree
# runs out of impurity to remove before depth 7), so listing them separately
# would just repeat the same row - and each unconstrained fit on 455x30 data
# costs a few seconds in this pure-Python implementation.
depths = [2, 3, 5, None]

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
# - The 'Tree Depth' column reports the depth ACTUALLY reached, in edges, the
#   same unit as max_depth. It equals max_depth while the tree still has
#   impurity to remove, then stops growing (max_depth=None here stops at 7).
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

# Gini and Entropy usually agree closely - Gini IS entropy to first order.
# Write entropy in NATS and replace ln(p) by its tangent at p = 1, i.e.
# ln(p) ~= p - 1; then -sum(p*ln(p)) collapses to 1 - sum(p^2), which is
# exactly Gini. (The expansion point is p = 1, because p = 1 is where the
# LOGARITHM is being linearised. Full derivation under "Why do Gini and
# Entropy almost always pick the same split?" in _6_decision_trees.md.)
# Agreeing to first order is why the two rank splits almost identically.
# On THIS particular wine split they still diverge by ~9 points of test
# accuracy. That gap is not a flaw in either criterion: both reach 100% train
# accuracy, and the criteria merely break a near-tie differently, which then
# cascades down the tree. It is a concrete demonstration of the "High
# Variance" limitation in _6_decision_trees.md - a single deep tree is very
# sensitive to tiny changes in how splits are chosen.
# Averaging many trees (Random Forest) is the standard cure.
#
# Gini: Slightly faster to compute (no logarithm), scikit-learn's default
# Entropy: More theoretically grounded in information theory
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _6_decision_trees.py
    # Requires numpy only. Everything below is reproducible (seed=42).
    # ----------------------------------------------------------------
    np.random.seed(42)

    print("=" * 60)
    print("DECISION TREE FROM SCRATCH - PLUG AND PLAY DEMO")
    print("=" * 60)

    # Three OVERLAPPING Gaussian blobs. They overlap on purpose: with
    # well-separated blobs a depth-2 tree already scores ~100% and the
    # overfitting story in DEMO 4 would be invisible.
    centers = np.array([[-1.2, -1.2],
                        [0.0, 1.4],
                        [1.2, -1.0]])
    X_cls = np.vstack([c + np.random.randn(100, 2) * 1.1 for c in centers])
    y_cls = np.repeat([0, 1, 2], 100)

    # Shuffle before splitting - the rows above are grouped by class, so an
    # unshuffled slice would put entire classes in the test set only.
    idx = np.random.permutation(300)
    X_cls, y_cls = X_cls[idx], y_cls[idx]
    X_tr, X_te = X_cls[:220], X_cls[220:]
    y_tr, y_te = y_cls[:220], y_cls[220:]

    # ---------------- DEMO 1: multiclass classification ----------------
    print("\n" + "-" * 60)
    print("DEMO 1 - Classification: 3 overlapping Gaussian blobs")
    print("Splits are chosen by maximising Gini information gain.")
    print("-" * 60)

    clf = DecisionTree(max_depth=3, criterion='gini', task='classification')
    clf.fit(X_tr, y_tr)

    print(f"Train accuracy : {clf.score(X_tr, y_tr):.2%}")
    print(f"Test  accuracy : {clf.score(X_te, y_te):.2%}")
    print(f"Tree depth     : {clf.get_depth()} edges (max_depth was 3)")
    print(f"Leaves         : {clf.get_n_leaves()}")
    print(f"Feature importances (MDI): {np.round(clf.feature_importances_, 4)}")

    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    print("\nSample predictions (x1, x2, true, pred, P(pred)):")
    for i in range(5):
        p = proba[i][int(preds[i])]
        print(f"  x=({X_te[i, 0]:6.2f},{X_te[i, 1]:6.2f})  "
              f"true={int(y_te[i])}  pred={int(preds[i])}  P={p:.2f}")

    # ---------------- DEMO 2: Gini vs Entropy ----------------
    print("\n" + "-" * 60)
    print("DEMO 2 - Split criterion: Gini vs Entropy (same data, depth 3)")
    print("-" * 60)
    print(f"{'Criterion':<12} {'Train Acc':<12} {'Test Acc':<12} {'Leaves':<8}")

    for criterion in ('gini', 'entropy'):
        m = DecisionTree(max_depth=3, criterion=criterion,
                         task='classification').fit(X_tr, y_tr)
        print(f"{criterion:<12} {m.score(X_tr, y_tr):<12.4f} "
              f"{m.score(X_te, y_te):<12.4f} {m.get_n_leaves():<8}")

    print("-> The two criteria usually agree closely: entropy in nats")
    print("   becomes exactly Gini when ln(p) is replaced by its tangent")
    print("   at p = 1. Gini also needs no logarithm to evaluate.")

    # ---------------- DEMO 3: regression ----------------
    print("\n" + "-" * 60)
    print("DEMO 3 - Regression: y = x^2 + noise, criterion='mse'")
    print("-" * 60)

    X_reg = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_reg = X_reg.ravel() ** 2 + np.random.randn(200) * 0.3

    # Shuffle so train and test cover the same x range. This data is generated
    # in SORTED x order, and a tree cannot extrapolate: without the shuffle
    # every test x would lie beyond the largest x the tree ever saw, and the
    # model would look broken through no fault of its own.
    idx_reg = np.random.permutation(200)
    X_reg, y_reg = X_reg[idx_reg], y_reg[idx_reg]
    X_rtr, X_rte = X_reg[:150], X_reg[150:]
    y_rtr, y_rte = y_reg[:150], y_reg[150:]

    reg = DecisionTree(max_depth=4, criterion='mse', task='regression')
    reg.fit(X_rtr, y_rtr)

    print(f"Train R2 : {reg.score(X_rtr, y_rtr):.4f}")
    print(f"Test  R2 : {reg.score(X_rte, y_rte):.4f}")
    print(f"Leaves   : {reg.get_n_leaves()} (a step function with this many steps)")

    reg_preds = reg.predict(X_rte)
    print("\nSample predictions (x, true, predicted):")
    for i in range(5):
        print(f"  x={X_rte[i, 0]:6.2f}  true={y_rte[i]:6.2f}  "
              f"pred={reg_preds[i]:6.2f}")

    # ---------------- DEMO 4: depth controls overfitting ----------------
    print("\n" + "-" * 60)
    print("DEMO 4 - Why max_depth matters (same blobs as DEMO 1)")
    print("-" * 60)
    print(f"{'max_depth':<12} {'Train Acc':<12} {'Test Acc':<12} {'Leaves':<8}")

    for depth in (1, 2, 3, 5, 8, None):
        m = DecisionTree(max_depth=depth, criterion='gini',
                         task='classification').fit(X_tr, y_tr)
        print(f"{str(depth):<12} {m.score(X_tr, y_tr):<12.4f} "
              f"{m.score(X_te, y_te):<12.4f} {m.get_n_leaves():<8}")

    print("-> Train accuracy climbs all the way to 1.0000, but test accuracy")
    print("   peaks early and then FALLS. That gap is overfitting: past the")
    print("   peak the tree is memorising noise, not learning structure.")
    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)
