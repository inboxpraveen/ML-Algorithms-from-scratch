import numpy as np
import sys
import os

# Add the Decision Trees folder to the path to import DecisionTree
# A Random Forest is built out of Decision Trees, so we reuse the DecisionTree
# class from Chapter 6 instead of duplicating it here. Only numpy is needed for
# the maths; sys/os are standard library and are used purely to locate the
# sibling folder. If you want to copy this file somewhere else, paste the
# DecisionTree class from _6_decision_trees.py above and delete lines 2-9.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '6. Decision Trees'))

from _6_decision_trees import DecisionTree


class _RandomFeatureTree(DecisionTree):
    """
    A DecisionTree that searches only a RANDOM SUBSET of features at every split.

    This tiny subclass is what turns plain bagging into a true Random Forest.
    It overrides exactly one method - the split search - and leaves everything
    else (tree growing, leaves, prediction) to the Chapter 6 DecisionTree.

    Breiman (2001) calls the number of candidate features "mtry"; sklearn calls
    it max_features. At every node we draw

        cols = k features sampled WITHOUT replacement from the p available

    and pick the best split among those k only. Two trees can therefore differ
    even when they see identical data, which is exactly what decorrelates them.

    Parameters:
    -----------
    max_features : int
        Number of features (k) to consider at each split.
    rng : numpy.random.RandomState
        Seeded generator shared with the parent forest, so the whole forest is
        reproducible from a single random_state.
    **kwargs :
        Passed straight through to DecisionTree (max_depth, min_samples_split,
        min_samples_leaf, criterion, task).
    """

    def __init__(self, max_features, rng, **kwargs):
        super().__init__(**kwargs)
        self.max_features = max_features
        self._rng = rng

    def _best_split(self, X, y):
        """
        Find the best split, but only among k randomly chosen features.

        The parent class scans every feature. Here we hand it a column subset,
        then translate the winning column back to its position in the full
        feature matrix so the rest of DecisionTree keeps working unchanged.
        """
        n_features = X.shape[1]
        k = min(self.max_features, n_features)

        # Draw k candidate features WITHOUT replacement for THIS node
        cols = self._rng.choice(n_features, size=k, replace=False)

        # Let the parent class do the actual gain computation on the subset
        split = super()._best_split(X[:, cols], y)

        if split is None and k < n_features:
            # No valid split exists among the k sampled columns. Two different
            # situations reach here, and sklearn treats them differently:
            #   - every sampled column was constant (no threshold at all):
            #     sklearn keeps drawing more features, so the full scan below
            #     is a near-equivalent shortcut;
            #   - a column was NOT constant but min_samples_leaf rejected every
            #     midpoint: sklearn would simply make this node a leaf, whereas
            #     we widen the search. That is a deliberate simplification.
            return super()._best_split(X, y)

        if split is not None:
            # Map the subset column index back to the original feature index
            split['feature_index'] = int(cols[split['feature_index']])

        return split


class RandomForest:
    """
    Random Forest Implementation from Scratch
    
    Random Forest is an ensemble learning method that combines multiple decision trees
    to create a more robust and accurate model. It uses bootstrap sampling and random
    feature selection to create diverse trees that vote together.
    
    Key Idea: "Grow many deliberately different trees on random data and random
    features, then let them vote - their errors cancel, their signal does not."
    
    Use Cases:
    - Credit risk / loan approval: default probability from income, history, ratios
    - Medical diagnosis: disease classification from symptoms and test panels
    - Fraud detection: flagging anomalous transactions in tabular payment data
    - Customer churn: who is about to leave, from usage and billing features
    - Feature selection: ranking predictors with feature_importances_
    
    The Two Sources of Randomness:
        1. Bootstrap sampling (bagging) - each tree gets n rows drawn WITH
           replacement, so it sees about 63.2% of the unique rows.
        2. Random feature subsets - at EVERY split only k of the p features are
           considered (see _RandomFeatureTree above). Without this the forest is
           just bagging, and the trees stay highly correlated. Whether this
           source is actually active depends on max_features: k < p engages it,
           k = p (the regression default) does not - see the note below.
    
    For classification: Majority vote across all trees
    For regression: Average prediction across all trees
    
    Combination rules:
        Classification:  pred(x) = mode({ T_1(x), T_2(x), ..., T_B(x) })
        Regression:      pred(x) = (1 / B) * sum_b T_b(x)
        where B = n_estimators and T_b is the b-th tree.

    Default number of candidate features per split (max_features='auto'):
        Classification:  k = floor(sqrt(p))    (Breiman's rule, sklearn's default)
        Regression:      k = p                 (sklearn's default; Breiman
                                                suggests p/3 - pass 1/3 for that)
        Note what k = p means for source of randomness 2: with every feature a
        candidate at every node, there is no column subsampling, so the DEFAULT
        REGRESSION FOREST IS PLAIN BAGGING - identical to max_features=None.
        That is sklearn's default too, but if you want the namesake mechanism on
        a regression task, pass max_features=1/3 (Breiman) or 'sqrt'.

    Variance of the ensemble (why the randomness matters):
        Var = rho * sigma^2 + (1 - rho) * sigma^2 / B
        Averaging kills the second term, but the first term - driven by the
        correlation rho between trees - only shrinks if the trees are genuinely
        different. Feature subsampling is what lowers rho.

    Simplifications vs. canonical Random Forest (see the .md for details):
    - predict_proba returns VOTE FRACTIONS (multiples of 1/n_estimators), which is
      what Breiman 2001 specifies; sklearn instead averages per-tree leaf
      distributions and so returns smoother probabilities.
    - Missing values are NOT handled (no surrogate splits). Impute before fitting.
    - Trees are grown sequentially; no parallelism.
    - When no split is found among the k sampled columns we rescan all p
      features; sklearn does the same for an all-constant draw, but makes a leaf
      when only min_samples_leaf blocked the thresholds (see _RandomFeatureTree
      above).
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, bootstrap=True, criterion='gini',
                 task='classification', random_state=None,
                 max_features='auto', oob_score=False):
        """
        Initialize the Random Forest model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
            More trees = better performance, but slower training
            Never hurts accuracy, only runtime - it cannot overfit by itself
            Typical values: 50, 100, 200
        
        max_depth : int or None, default=None
            Maximum depth of each tree
            None = unlimited depth (trees grow until pure)
            Smaller values = less overfitting, faster training
            Typical values: 5-20, or None
        
        min_samples_split : int, default=2
            Minimum samples required to split a node
            Larger values = more conservative trees
            Typical values: 2-10
        
        min_samples_leaf : int, default=1
            Minimum samples required at leaf node
            Larger values = smoother predictions
            Typical values: 1-5
        
        bootstrap : bool, default=True
            Whether to use bootstrap sampling (sample with replacement)
            True = each tree sees a different random subset of the rows
            False = every tree sees all rows; diversity then comes ONLY from
                    max_features, so never combine bootstrap=False with
                    max_features=None (that makes all trees identical)
            Typical value: True
        
        criterion : str, default='gini'
            Split quality measure for trees
            Classification: 'gini' or 'entropy'
            Regression: 'mse'
            If task='regression' and a classification criterion is passed, it is
            auto-corrected to 'mse' (the effective value is stored in criterion_)
            Typical value: 'gini' for classification, 'mse' for regression
        
        task : str, default='classification'
            Type of prediction task - changes the whole model, so set it first
            Options: 'classification', 'regression'
        
        random_state : int or None, default=None
            Random seed for reproducibility
            Seeds a PRIVATE generator (np.random.RandomState); it does not touch
            the global numpy random stream
            Typical values: any int, e.g. 42

        max_features : {'auto', 'sqrt', 'log2', None}, int or float, default='auto'
            Number of features k considered at EACH split (Breiman's "mtry").
            This is the ingredient that makes a Random Forest "random":
            - 'auto' : floor(sqrt(p)) for classification, all p for regression
                       (these are scikit-learn's defaults)
            - 'sqrt' : floor(sqrt(p))
            - 'log2' : floor(log2(p))
            - None   : all p features -> this is plain BAGGING, not a forest
                       ('auto' on a regression task resolves to the same thing)
            - int    : that many features
            - float  : that fraction of p, e.g. 0.5 or 1/3
            Smaller k = more decorrelated trees (lower variance, higher bias)
            Typical values: 'sqrt' for classification; for regression try
            max_features=1/3 (Breiman's recommendation) when p is large

        oob_score : bool, default=False
            Whether to compute the out-of-bag score after fitting.
            Each tree omits about 36.8% of the rows, so those rows act as a free
            validation set. Result is stored in oob_score_
            Typical value: True when you have no separate validation split
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.task = task
        self.random_state = random_state
        self.max_features = max_features
        self.oob_score = oob_score
        
        # PRIVATE random generator. Seeding np.random globally (the old
        # behaviour) would silently change the caller's own random stream.
        self._rng = np.random.RandomState(self.random_state)
        
        # Store trees
        self.trees = []
        self.n_classes_ = None
        self.classes_ = None
        self.n_features_ = None
        self.criterion_ = None
        self.max_features_ = None
        self.feature_importances_ = None
        self.oob_score_ = None
    
    def _check_array(self, X):
        """
        Convert X to a 2-D numpy array (accepts lists and 1-D arrays)

        A 1-D input is read as ONE feature: shape (n,) -> (n, 1).

        Parameters:
        -----------
        X : array-like
            Feature matrix or single-feature vector

        Returns:
        --------
        X : numpy array of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got {X.ndim} dimensions.")
        return X

    def _check_is_fitted(self):
        """Raise a clear error if predict/score is called before fit."""
        if not self.trees:
            raise ValueError(
                "Model is not fitted yet. Call fit(X, y) before predicting."
            )

    def _resolve_criterion(self):
        """
        Validate task/criterion and return the criterion the trees will use

        Classification accepts 'gini' or 'entropy'; regression only 'mse'.
        A classification criterion passed with task='regression' is corrected
        to 'mse' rather than being silently ignored (the old behaviour).
        """
        if self.task not in ('classification', 'regression'):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {self.task!r}."
            )

        if self.task == 'classification':
            if self.criterion not in ('gini', 'entropy'):
                raise ValueError(
                    "criterion for classification must be 'gini' or 'entropy', "
                    f"got {self.criterion!r}."
                )
            return self.criterion

        # Regression: only MSE is implemented by DecisionTree
        if self.criterion in ('gini', 'entropy'):
            return 'mse'   # auto-correct the classification default
        if self.criterion != 'mse':
            raise ValueError(
                f"criterion for regression must be 'mse', got {self.criterion!r}."
            )
        return 'mse'

    def _resolve_max_features(self, n_features):
        """
        Turn the max_features setting into an integer k

        k = floor(sqrt(p))      for 'sqrt', and for 'auto' when classifying
        k = p                   for 'auto' when regressing, and for None
        k = floor(log2(p))      for 'log2'

        The 'auto' rules match scikit-learn's defaults (sqrt for
        RandomForestClassifier, all features for RandomForestRegressor).
        Breiman's original paper recommends about p/3 for regression - pass
        max_features=1/3 to get that; it usually helps when p is large.

        Parameters:
        -----------
        n_features : int
            Total number of features p

        Returns:
        --------
        k : int
            Number of features to sample at each split, clipped to [1, p]
        """
        p = n_features
        mf = self.max_features

        if mf is None:
            k = p
        elif mf == 'auto':
            # sqrt for classification, all features for regression (sklearn's defaults)
            k = int(np.sqrt(p)) if self.task == 'classification' else p
        elif mf == 'sqrt':
            k = int(np.sqrt(p))
        elif mf == 'log2':
            k = int(np.log2(p))
        elif isinstance(mf, (int, np.integer)) and not isinstance(mf, bool):
            k = int(mf)
        elif isinstance(mf, float):
            k = int(mf * p)
        else:
            raise ValueError(
                "max_features must be 'auto', 'sqrt', 'log2', None, an int "
                f"or a float, got {mf!r}."
            )

        # Always consider at least one feature, never more than we have
        return int(np.clip(k, 1, p))

    def _bootstrap_sample(self, X, y, return_indices=False):
        """
        Create a bootstrap sample (random sample with replacement)

        We draw n indices out of n WITH replacement, so
            P(row i is drawn at least once) = 1 - (1 - 1/n)^n  ->  1 - 1/e ~= 0.632
        Each tree therefore sees roughly 63.2% of the distinct rows; the other
        ~36.8% are that tree's out-of-bag (OOB) rows and are free validation data.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training features
        y : numpy array of shape (n_samples,)
            Training labels
        return_indices : bool, default=False
            If True, also return the drawn row indices (used for OOB scoring)
            
        Returns:
        --------
        X_sample : numpy array
            Bootstrap sample of features
        y_sample : numpy array
            Bootstrap sample of labels
        indices : numpy array, only when return_indices=True
            The row indices that were drawn
        """
        n_samples = len(X)
        
        if self.bootstrap:
            # Sample with replacement (bootstrap) using the PRIVATE generator
            indices = self._rng.choice(n_samples, size=n_samples, replace=True)
        else:
            # Use all samples (no bootstrap)
            indices = np.arange(n_samples)
        
        if return_indices:
            return X[indices], y[indices], indices
        return X[indices], y[indices]

    def _accumulate_importances(self, tree, X_sample, y_sample):
        """
        Add one tree's Mean Decrease in Impurity (MDI) to feature_importances_

        For every internal node we credit the splitting feature with the
        impurity it removed, weighted by how many samples reached that node:

            importance[f] += n_node * ( I(parent) - [ (n_L/n)*I(left)
                                                    + (n_R/n)*I(right) ] )

        The bracketed part is exactly DecisionTree._information_gain, so these
        numbers agree with the gains the tree actually optimised. We walk the
        tree iteratively (a stack) rather than recursively so deep trees are safe.

        Parameters:
        -----------
        tree : DecisionTree
            A fitted tree from this forest
        X_sample, y_sample : numpy arrays
            The bootstrap sample that tree was trained on
        """
        stack = [(tree.tree, X_sample, y_sample)]

        while stack:
            node, X_node, y_node = stack.pop()
            if node['type'] == 'leaf':
                continue

            feature_index = node['feature_index']
            threshold = node['threshold']

            left_mask = X_node[:, feature_index] <= threshold
            right_mask = ~left_mask

            n = len(y_node)
            n_left = int(np.sum(left_mask))
            n_right = n - n_left

            # Impurity drop achieved by this split (same formula the tree used)
            parent_impurity = tree._calculate_impurity(y_node)
            child_impurity = ((n_left / n) * tree._calculate_impurity(y_node[left_mask])
                              + (n_right / n) * tree._calculate_impurity(y_node[right_mask]))

            self.feature_importances_[feature_index] += n * (parent_impurity - child_impurity)

            stack.append((node['left'], X_node[left_mask], y_node[left_mask]))
            stack.append((node['right'], X_node[right_mask], y_node[right_mask]))
    
    def fit(self, X, y):
        """
        Train the Random Forest by building multiple decision trees
        
        Each tree is trained on a different bootstrap sample of the ROWS and,
        at every split, considers a random subset of k = max_features_ of the
        COLUMNS (when k = p that "subset" is every column, i.e. no column
        randomness at all). Those two independent sources of randomness are
        what make the trees diverse, which is what makes averaging them
        worthwhile.

        Also computes feature_importances_ (mean decrease in impurity) and,
        when oob_score=True, the out-of-bag score.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training features
        y : numpy array of shape (n_samples,)
            Training labels/values

        Returns:
        --------
        self : RandomForest
            The fitted model
        """
        X = self._check_array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Re-seed the private generator so repeated fit() calls are identical
        self._rng = np.random.RandomState(self.random_state)

        # Resolve the settings the trees will actually run with
        self.criterion_ = self._resolve_criterion()
        self.max_features_ = self._resolve_max_features(n_features)

        # Store the class labels for classification. Keeping the labels (not
        # just how many there are) lets predict_proba work with ANY encoding -
        # strings, {1, 2}, {-1, +1} - not only 0..n_classes-1.
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

        # Accumulators
        self.trees = []
        self.feature_importances_ = np.zeros(n_features)
        if self.oob_score:
            if self.task == 'classification':
                oob_votes = np.zeros((n_samples, self.n_classes_))
            else:
                oob_sums = np.zeros(n_samples)
            oob_counts = np.zeros(n_samples)
        
        # Build each tree in the forest
        for i in range(self.n_estimators):
            # Create a bootstrap sample for this tree
            X_sample, y_sample, indices = self._bootstrap_sample(X, y, return_indices=True)
            
            # Create and train a decision tree that samples max_features_
            # candidate columns at every split (see _RandomFeatureTree)
            tree = _RandomFeatureTree(
                max_features=self.max_features_,
                rng=self._rng,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion_,
                task=self.task
            )
            tree.fit(X_sample, y_sample)
            
            # Add tree to forest
            self.trees.append(tree)
    
            # Credit each feature with the impurity this tree removed
            self._accumulate_importances(tree, X_sample, y_sample)

            # Out-of-bag: let this tree vote ONLY on the rows it never saw
            if self.oob_score:
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[indices] = True
                oob_mask = ~in_bag
                if np.any(oob_mask):
                    oob_pred = tree.predict(X[oob_mask])
                    if self.task == 'classification':
                        # searchsorted maps a label back to its column in classes_
                        cols = np.searchsorted(self.classes_, oob_pred)
                        oob_votes[np.flatnonzero(oob_mask), cols] += 1
                    else:
                        oob_sums[oob_mask] += oob_pred
                    oob_counts[oob_mask] += 1

        # Normalise importances so they sum to 1 (sklearn's convention)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        if self.oob_score:
            self._finalize_oob_score(
                y, oob_counts,
                oob_votes if self.task == 'classification' else oob_sums
            )

        return self

    def _finalize_oob_score(self, y, oob_counts, oob_totals):
        """
        Turn the accumulated out-of-bag votes/sums into oob_score_

        Only rows that were out-of-bag for at least one tree can be scored.
        With enough trees that is essentially every row, since
        P(row in-bag for all B trees) = 0.632^B.

        Parameters:
        -----------
        y : numpy array
            True labels/values
        oob_counts : numpy array of shape (n_samples,)
            How many trees voted on each row
        oob_totals : numpy array
            Vote counts per class (classification) or summed predictions (regression)
        """
        scored = oob_counts > 0
        if not np.any(scored):
            self.oob_score_ = None
            return

        if self.task == 'classification':
            oob_pred = self.classes_[np.argmax(oob_totals[scored], axis=1)]
            self.oob_score_ = float(np.mean(oob_pred == y[scored]))
        else:
            oob_pred = oob_totals[scored] / oob_counts[scored]
            y_scored = y[scored]
            ss_res = np.sum((y_scored - oob_pred) ** 2)
            ss_tot = np.sum((y_scored - np.mean(y_scored)) ** 2)
            if ss_tot == 0:
                # Same convention as score(): with a constant target R^2 is
                # undefined, so only an exact fit counts as perfect.
                self.oob_score_ = 1.0 if ss_res == 0 else 0.0
            else:
                self.oob_score_ = float(1 - ss_res / ss_tot)
    
    def predict(self, X):
        """
        Make predictions using all trees in the forest
        
        For classification: Uses majority voting across all trees
        For regression: Uses average prediction across all trees
        
        Formulas implemented below:
            Classification:  pred(x) = mode({ T_1(x), ..., T_B(x) })
            Regression:      pred(x) = (1 / B) * sum_b T_b(x)
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted labels (classification) or values (regression)
        """
        self._check_is_fitted()
        X = self._check_array(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this forest was fitted on "
                f"{self.n_features_}."
            )
        
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
        
        Returns the proportion of trees that predicted each class:

            P(class = c | x) = (# trees voting c) / B

        Column j corresponds to self.classes_[j] (the sorted unique training
        labels), so any label encoding works - strings, {1, 2}, {-1, +1}.

        Note on granularity: these are VOTE FRACTIONS, so every value is a
        multiple of 1/n_estimators (with 5 trees you can only ever see
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0). That is what Breiman (2001) specifies.
        sklearn instead averages the per-tree leaf class distributions, which
        gives smoother numbers; use more trees here for finer resolution.

        Only available for classification.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        probabilities : numpy array of shape (n_samples, n_classes)
            Class probabilities for each sample; each row sums to 1.0
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only works for classification tasks")
        
        self._check_is_fitted()
        X = self._check_array(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this forest was fitted on "
                f"{self.n_features_}."
            )
        
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
            
            # Calculate the proportion of votes for each ACTUAL class label
            class_probs = []
            for class_label in self.classes_:
                prob = np.mean(sample_preds == class_label)
                class_probs.append(prob)
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Calculate model performance score
        
        For classification: Accuracy (proportion of correct predictions)
        For regression: R^2 score (coefficient of determination)

            R^2 = 1 - SS_res / SS_tot
            SS_res = sum (y - y_pred)^2      (what the model still gets wrong)
            SS_tot = sum (y - mean(y))^2     (what predicting the mean gets wrong)
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test features
        y : numpy array of shape (n_samples,)
            True labels/values
            
        Returns:
        --------
        score : float
            Accuracy in [0, 1] for classification; R^2 for regression
            (R^2 <= 1.0, and can be negative if the model is worse than the mean)
        """
        predictions = self.predict(X)
        y = np.array(y)
        
        if self.task == 'classification':
            # Classification: Accuracy
            return np.mean(predictions == y)
        else:
            # Regression: R^2 score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot == 0:
                # y is constant: R^2 is undefined. Perfect only if we nailed it.
                return 1.0 if ss_res == 0 else 0.0
            
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
# Note: criterion='mse' is the only split measure that makes sense for regression
model = RandomForest(n_estimators=20, max_depth=5, task='regression',
                     criterion='mse', random_state=42)
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [1300, 6, 2],    # Similar to training examples
    [2200, 2, 4],    # Larger, newer house
    [950, 12, 2],    # Smaller, older house
])

predictions = model.predict(X_test)
print("Predicted prices ($1000s):", predictions)

# Calculate R^2 score
r2_score = model.score(X_train, y_train)
print(f"R2 Score on training data: {r2_score:.4f}")

# Which feature drove the predictions? (importances sum to 1.0)
names = ['Size', 'Age', 'Bedrooms']
for name, imp in zip(names, model.feature_importances_):
    print(f"  {name:<10} importance = {imp:.4f}")
"""

"""
USAGE EXAMPLE 4: Comparing Different Numbers of Trees

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load wine dataset (124 training rows x 13 features - small enough to sweep
# quickly; this whole block runs in a couple of seconds)
data = load_wine()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Try different numbers of trees
n_trees_values = [5, 10, 20]

print("Comparing Different Numbers of Trees:\n")
print(f"{'Trees':<10} {'Train Accuracy':<20} {'Test Accuracy':<20} {'OOB Score':<12}")
print("-" * 62)

for n_trees in n_trees_values:
    model = RandomForest(n_estimators=n_trees, max_depth=6,
                        task='classification', oob_score=True, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"{n_trees:<10} {train_acc:<20.4f} {test_acc:<20.4f} {model.oob_score_:<12.4f}")

# Observations:
# - More trees generally = better performance, but on a 54-row test set the
#   trend is noisy (10 trees may happen to beat 20); the OOB column, computed
#   over all 124 training rows, is the steadier signal and rises monotonically
# - Diminishing returns after ~50-100 trees
# - More trees = slower training but same prediction speed per tree
# - The OOB score estimates test performance without a held-out split
"""

"""
USAGE EXAMPLE 5: Random Forest vs Single Decision Tree

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# DecisionTree is already imported at the top of this module, so there is no
# need to touch sys.path again here.

# Load wine dataset
data = load_wine()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Single Decision Tree
single_tree = DecisionTree(max_depth=8, task='classification')
single_tree.fit(X_train, y_train)
single_tree_acc = single_tree.score(X_test, y_test)

# Train Random Forest (runs in a couple of seconds on this dataset)
forest = RandomForest(n_estimators=20, max_depth=8,
                     task='classification', random_state=42)
forest.fit(X_train, y_train)
forest_acc = forest.score(X_test, y_test)

# Plain bagging: same forest but every split sees ALL 13 features
bagging = RandomForest(n_estimators=20, max_depth=8, max_features=None,
                      task='classification', random_state=42)
bagging.fit(X_train, y_train)
bagging_acc = bagging.score(X_test, y_test)

print("Comparison: Single Tree vs Bagging vs Random Forest")
print("-" * 52)
print(f"Single Decision Tree Accuracy:  {single_tree_acc:.4f}")
print(f"Bagging (max_features=None):    {bagging_acc:.4f}")
print(f"Random Forest (sqrt features):  {forest_acc:.4f}")
print(f"Improvement over single tree:   {(forest_acc - single_tree_acc):.4f}")

# Random Forest typically outperforms a single tree due to:
# - Reduced overfitting through ensemble averaging
# - Reduced variance through bootstrap sampling
# - Reduced correlation between trees through random feature subsets
# - Better generalization to unseen data
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _7_random_forests.py
    # Needs only numpy (plus the sibling '6. Decision Trees' folder).
    # ----------------------------------------------------------------
    import time   # only used to report fit times in DEMO 3

    np.random.seed(42)

    print("=" * 55)
    print("RANDOM FOREST FROM SCRATCH - PLUG-AND-PLAY DEMO")
    print("=" * 55)

    # ---------------------------------------------------------------
    # DEMO 1 - Binary classification on two 4-D Gaussian blobs
    # ---------------------------------------------------------------
    print("\n" + "=" * 55)
    print("DEMO 1 - Classification: two Gaussian blobs (4 features)")
    print("=" * 55)

    X0 = np.random.randn(100, 4) - 1.5      # class 0 centred at (-1.5, ...)
    X1 = np.random.randn(100, 4) + 1.5      # class 1 centred at (+1.5, ...)
    X_cls = np.vstack([X0, X1])
    y_cls = np.array([0] * 100 + [1] * 100)

    # Shuffle BEFORE slicing: the rows are stacked class-by-class, so an
    # unshuffled 150/50 split would put class 0 in train and class 1 in test.
    idx = np.random.permutation(200)
    X_cls, y_cls = X_cls[idx], y_cls[idx]
    X_tr, X_te = X_cls[:150], X_cls[150:]
    y_tr, y_te = y_cls[:150], y_cls[150:]

    clf = RandomForest(
        n_estimators=25,
        max_depth=5,
        task='classification',
        oob_score=True,          # free validation from the ~36.8% held-out rows
        random_state=42
    )
    clf.fit(X_tr, y_tr)

    print(f"Features per split (k of p) : {clf.max_features_} of {clf.n_features_}")
    print(f"Train Accuracy              : {clf.score(X_tr, y_tr):.4f}")
    print(f"Test  Accuracy              : {clf.score(X_te, y_te):.4f}")
    print(f"Out-of-bag Accuracy         : {clf.oob_score_:.4f}")

    proba = clf.predict_proba(X_te)
    print("\nSample predictions (vote fractions, granularity 1/25):")
    for i in range(5):
        print(f"  true={y_te[i]}  P(class=0)={proba[i, 0]:.2f}  "
              f"P(class=1)={proba[i, 1]:.2f}")

    # ---------------------------------------------------------------
    # DEMO 2 - Regression on y = x^2 + noise
    # ---------------------------------------------------------------
    print("\n" + "=" * 55)
    print("DEMO 2 - Regression: y = x^2 + noise")
    print("=" * 55)

    X_reg = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_reg = X_reg.ravel() ** 2 + np.random.randn(200) * 0.5

    # Shuffle BEFORE slicing. np.linspace is sorted, so without this the test
    # set would be every x above the training maximum - and trees cannot
    # extrapolate beyond the range they were trained on.
    idx_reg = np.random.permutation(200)
    X_reg, y_reg = X_reg[idx_reg], y_reg[idx_reg]
    X_tr2, X_te2 = X_reg[:150], X_reg[150:]
    y_tr2, y_te2 = y_reg[:150], y_reg[150:]

    reg = RandomForest(
        n_estimators=25,
        max_depth=6,
        task='regression',
        criterion='mse',
        random_state=42
    )
    reg.fit(X_tr2, y_tr2)

    print(f"Train R2 : {reg.score(X_tr2, y_tr2):.4f}")
    print(f"Test  R2 : {reg.score(X_te2, y_te2):.4f}")

    preds = reg.predict(X_te2)
    print("\nSample predictions (x, true, predicted):")
    for i in range(5):
        print(f"  x={X_te2[i, 0]:5.2f}  true={y_te2[i]:5.2f}  pred={preds[i]:5.2f}")

    # ---------------------------------------------------------------
    # DEMO 3 - One tree vs bagging vs a real Random Forest
    #          This is where you can SEE what max_features buys you.
    # ---------------------------------------------------------------
    print("\n" + "=" * 55)
    print("DEMO 3 - Single tree vs Bagging vs Random Forest")
    print("=" * 55)

    # 12 features, only the first 3 carry signal; the other 9 are pure noise.
    # We train on 150 rows but TEST on 350 so the accuracy gaps below are not
    # just small-sample noise.
    X_mix = np.random.randn(500, 12)
    y_mix = (X_mix[:, 0] + X_mix[:, 1] - X_mix[:, 2] > 0).astype(int)
    idx_mix = np.random.permutation(500)
    X_mix, y_mix = X_mix[idx_mix], y_mix[idx_mix]
    X_tr3, X_te3 = X_mix[:150], X_mix[150:]
    y_tr3, y_te3 = y_mix[:150], y_mix[150:]

    def mean_pairwise_disagreement(forest, X):
        """Fraction of test rows on which two trees of the forest differ."""
        tree_preds = np.array([t.predict(X) for t in forest.trees])
        n_trees = len(tree_preds)
        diffs = [np.mean(tree_preds[a] != tree_preds[b])
                 for a in range(n_trees) for b in range(a + 1, n_trees)]
        return float(np.mean(diffs))

    t0 = time.time()
    single = DecisionTree(max_depth=5, task='classification')
    single.fit(X_tr3, y_tr3)
    t_single = time.time() - t0

    t0 = time.time()
    bagging = RandomForest(n_estimators=15, max_depth=5, max_features=None,
                           task='classification', random_state=42)
    bagging.fit(X_tr3, y_tr3)
    t_bag = time.time() - t0

    t0 = time.time()
    forest = RandomForest(n_estimators=15, max_depth=5, max_features='sqrt',
                          task='classification', random_state=42)
    forest.fit(X_tr3, y_tr3)
    t_forest = time.time() - t0

    print(f"{'Model':<30}{'Test Acc':>10}{'Disagreement':>14}{'Fit sec':>10}")
    print("-" * 64)
    print(f"{'Single decision tree':<30}{single.score(X_te3, y_te3):>10.4f}"
          f"{'n/a':>14}{t_single:>10.2f}")
    print(f"{'Bagging (all 12 features)':<30}{bagging.score(X_te3, y_te3):>10.4f}"
          f"{mean_pairwise_disagreement(bagging, X_te3):>14.4f}{t_bag:>10.2f}")
    print(f"{'Random Forest (k=3 feats)':<30}{forest.score(X_te3, y_te3):>10.4f}"
          f"{mean_pairwise_disagreement(forest, X_te3):>14.4f}{t_forest:>10.2f}")

    print("\nWhat to read here:")
    print("  1. Both ensembles beat the single tree - that is the bagging effect.")
    print("  2. 'Disagreement' is the average fraction of test rows on which two")
    print("     trees differ, i.e. a direct proxy for the correlation rho in")
    print("     Var = rho*sigma^2 + (1-rho)*sigma^2/B. Feature subsampling")
    print("     raises disagreement a lot, which is what lowers rho.")
    print("  3. Here the forest matches bagging's accuracy while looking at only")
    print("     3 of 12 features per split, so it also trains much faster. With")
    print("     hundreds of correlated features the decorrelation usually wins")
    print("     on accuracy too; with 12 features the gap is within noise.")

    print("\nFeature importances of the Random Forest (should favour 0, 1, 2):")
    order = np.argsort(forest.feature_importances_)[::-1]
    for rank, f_idx in enumerate(order[:5], start=1):
        print(f"  {rank}. feature {f_idx:<2} -> {forest.feature_importances_[f_idx]:.4f}")

    print("\n" + "=" * 55)
    print("Demo complete.")
    print("=" * 55)
