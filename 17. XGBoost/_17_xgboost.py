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
        Second-Order Optimization: Uses both gradient (g) and hessian (h) for better step sizes
        Regularization: L1 (alpha) and L2 (lambda) regularization with L1 soft-thresholding
        Gamma Pruning: Only split when gain exceeds gamma (complexity cost)
        Column Subsampling: Random feature selection per tree (like Random Forest)
        Row Subsampling: Stochastic boosting to reduce overfitting
        Early Stopping: Halt training when validation score stops improving
    
    L1 Soft-Thresholding:
        Leaf weights and gain scores use shrink(G, alpha) instead of G directly.
        shrink(G, alpha) = G - alpha  if G > alpha
                         = G + alpha  if G < -alpha
                         = 0          if |G| <= alpha
        This makes leaf weights exactly zero when gradient evidence is weak.

    Simplifications vs. the canonical XGBoost library:
        This class implements the exact greedy split finder from Algorithm 1 of the
        XGBoost paper. The following library features are deliberately NOT implemented
        (see the "Simplification vs. canonical XGBoost" section of _17_xgboost.md):
        - No weighted quantile sketch (approximate split finding). Every midpoint
          between consecutive distinct feature values is scored, which is exact but
          costs O(n_samples * n_features) gain evaluations per node.
        - No sparsity-aware split finding / learned default direction. NaN inputs are
          silently routed RIGHT at every node, because "NaN <= threshold" is False.
        - No colsample_bylevel / colsample_bynode; only colsample_bytree.
        - eval_set monitors only the first (X_val, y_val) tuple.
        - gamma follows the paper's halved gain (see the gamma docstring below).
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
            Note on scale: this implementation subtracts gamma from the HALVED gain,
            exactly as in Eq. (7) of the XGBoost paper:
                gain = 0.5 * [score_L + score_R - score_parent] - gamma
            The xgboost C++ library instead compares the un-halved loss change against
            min_split_loss, so a given numeric gamma prunes twice as aggressively here.
            gamma=g here corresponds to min_split_loss=2*g in the library.

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
        self.n_features = None

    def _sigmoid(self, x):
        """
        Sigmoid function with numerical stability

        Implements sigma(x) = 1 / (1 + exp(-x)), the link that turns a raw boosted
        score into a probability.

        The branch exists purely to keep exp() from overflowing:
        - for x >= 0 evaluate 1 / (1 + exp(-x))      -> exp(-x) is in (0, 1]
        - for x <  0 evaluate exp(x) / (1 + exp(x))  -> exp(x)  is in (0, 1]
        Both branches are algebraically the same function; each one only ever calls
        exp() on a non-positive argument, so a score of -1000 gives 0.0 instead of inf.

        Parameters:
        -----------
        x : np.ndarray or float
            Raw (log-odds) scores

        Returns:
        --------
        p : np.ndarray
            Probabilities in (0, 1)
        """
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
            # Hessian: d^2L/dpred^2 = 1
            gradient = y_pred - y_true
            hessian = np.ones_like(y_pred)
            
        elif self.objective in ['binary:logistic', 'reg:logistic']:
            # For logistic: L = -y*log(p) - (1-y)*log(1-p)
            # where p = sigmoid(pred)
            # Gradient: dL/dpred = p - y
            # Hessian: d^2L/dpred^2 = p * (1 - p)
            p = self._sigmoid(y_pred)
            gradient = p - y_true
            hessian = p * (1 - p)
            
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return gradient, hessian
    
    def _calculate_leaf_weight(self, gradient_sum, hessian_sum):
        """
        Calculate optimal leaf weight using XGBoost's formula
        
        With L2 only:  w* = -G / (H + lambda)
        With L1 + L2:  w* = -shrink(G, alpha) / (H + lambda)
        
        The L1 term introduces soft-thresholding on the gradient sum.
        When alpha > 0, gradients smaller than alpha are zeroed out,
        producing sparser leaf weights.
        
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
        # L1 soft-thresholding: shrink gradient toward zero by reg_alpha
        if gradient_sum > self.reg_alpha:
            g_shrunk = gradient_sum - self.reg_alpha
        elif gradient_sum < -self.reg_alpha:
            g_shrunk = gradient_sum + self.reg_alpha
        else:
            g_shrunk = 0.0
        return -g_shrunk / (hessian_sum + self.reg_lambda + 1e-10)
    
    def _calculate_gain(self, gradient_left, hessian_left, gradient_right, hessian_right):
        """
        Calculate the gain from a split using XGBoost's gain formula
        
        Gain = 0.5 * [score(G_L,H_L) + score(G_R,H_R) - score(G_L+G_R, H_L+H_R)] - gamma
        
        where score(G, H) = shrink(G, alpha)^2 / (H + lambda)
        and shrink(G, alpha) is the L1 soft-threshold of G.
        
        When reg_alpha = 0 this reduces to the standard formula:
        Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
        
        Where:
        - G_L, G_R: Sum of gradients in left/right child
        - H_L, H_R: Sum of hessians in left/right child
        - lambda: L2 regularization
        - alpha: L1 regularization (soft-threshold on G)
        - gamma: Minimum loss reduction (complexity cost)
        
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
        def calculate_score(G, H):
            # L1 soft-thresholding on gradient before squaring
            if G > self.reg_alpha:
                g = G - self.reg_alpha
            elif G < -self.reg_alpha:
                g = G + self.reg_alpha
            else:
                g = 0.0
            return (g ** 2) / (H + self.reg_lambda + 1e-10)
        
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

        Split candidates (exact greedy, Algorithm 1 of the XGBoost paper):
        For a feature with sorted distinct values u[0] < u[1] < ... < u[k-1], the
        candidate thresholds are the MIDPOINTS between consecutive values:
            thresholds = (u[:-1] + u[1:]) / 2
        Splitting at a midpoint instead of at an observed value gives the same k-1
        training partitions, but places the decision boundary halfway through the
        empty gap. An unseen point that falls inside that gap is then routed the way
        sklearn and the xgboost library would route it. Using u itself would push every
        boundary onto the left value and also waste one always-empty split at u[-1].

        Two float64 edge cases in that midpoint, documented here rather than fixed.
        When u[i] and u[i+1] are ADJACENT float64 numbers their midpoint is an exact
        tie, and round-half-to-even lands it on u[i+1] roughly half the time (20 of the
        first 40 adjacent pairs above 1.0); since the code routes with '<=', that one
        then reproduces the next one and the partition separating u[i] from u[i+1] is
        never scored. Separately, for feature values above ~9e307 the sum u[i]+u[i+1]
        overflows to inf, emitting a numpy RuntimeWarning and giving a threshold no row
        can exceed. Neither fires on ordinary data - 0 collapses across 59,700
        candidates from 300 standard-normal columns of 200 rows - so no clamp is
        applied and no warning is suppressed.

        Each node also records 'hessian' (the sum of hessians reaching it). That sum is
        XGBoost's definition of node "cover" and is what get_feature_importance('cover')
        accumulates; for squared error h=1 so it equals the sample count, but for
        logistic loss h = p(1-p) and the two differ.

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
                'count': n_samples,
                'hessian': hessian_sum
            }

        # Column subsampling (if not already specified).
        # Note that this draw runs even at colsample_bytree=1.0, where it is a no-op
        # as a subsample but still consumes global RNG state and randomizes the order
        # in which features are scanned below. Because the scan keeps only a strictly
        # better gain ('gain > best_gain'), that order decides exact ties, so two
        # unseeded fits on the same data can differ whenever two candidates tie -
        # rare on continuous features, but it does happen. Call np.random.seed(...)
        # immediately before fit() if you need bit-identical runs.
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

            # Sorted distinct values, then the midpoints between consecutive ones.
            # k distinct values -> k-1 candidate thresholds, each producing a
            # non-empty left AND right child unless the two values it separates are
            # adjacent float64 numbers (see the float64 note in the docstring above).
            unique_values = np.unique(feature_values)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            # Try each threshold
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

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
                'count': n_samples,
                'hessian': hessian_sum
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
            'count': n_samples,
            'hessian': hessian_sum
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
            Training data. A 1-D array/list is treated as a single feature column.
        y : np.ndarray or list, shape (n_samples,)
            Target values
        eval_set : list of tuples, optional
            List of (X_val, y_val) tuples for validation.
            Only the FIRST tuple is used for monitoring and early stopping; any
            further tuples are ignored (the library supports several, this does not).
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

        # Accept a 1-D X as a single-feature dataset, as the docstring promises
        if X.ndim == 1:
            X = X.reshape(-1, 1)

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

        # Validation set (only the first tuple of eval_set is monitored).
        # val_predictions is kept as a RUNNING raw score, updated by one tree per
        # round. Re-calling predict() every round would re-evaluate every tree built
        # so far, making monitoring cost O(n_estimators^2) tree evaluations.
        X_val = y_val = val_predictions = None
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            X_val = np.array(X_val, dtype=float)
            if X_val.ndim == 1:
                X_val = X_val.reshape(-1, 1)
            y_val = np.array(y_val, dtype=float)
            val_predictions = np.full(X_val.shape[0], self.base_score)

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
                is_logistic = self.objective in ['binary:logistic', 'reg:logistic']

                # Add just this round's tree to the running validation score
                val_predictions += self.learning_rate * self._predict_tree(tree, X_val)

                if is_logistic:
                    # Raw scores -> probabilities only when scoring
                    val_probs = self._sigmoid(val_predictions)
                    val_score = -np.mean(y_val * np.log(val_probs + 1e-10) +
                                        (1 - y_val) * np.log(1 - val_probs + 1e-10))
                else:
                    val_score = np.mean((y_val - val_predictions) ** 2)

                self.val_scores.append(val_score)

                # Early stopping
                if early_stopping_rounds is not None:
                    if val_score < best_score:
                        best_score = val_score
                        best_iteration = iteration
                    elif iteration - best_iteration >= early_stopping_rounds:
                        if verbose:
                            print(f"Early stopping at iteration {iteration}")
                            # Report the best score in the SAME units as the progress
                            # lines above (rmse = sqrt of the stored MSE).
                            if is_logistic:
                                print(f"Best iteration: {best_iteration}, "
                                      f"Best val-logloss: {best_score:.6f}")
                            else:
                                print(f"Best iteration: {best_iteration}, "
                                      f"Best val-rmse: {np.sqrt(best_score):.6f}")
                            print(f"Keeping {best_iteration + 1} trees "
                                  f"(iterations 0..{best_iteration})")
                        # Remove trees after best iteration
                        self.trees = self.trees[:best_iteration + 1]
                        break

                # Verbose output
                if verbose and (isinstance(verbose, bool) or iteration % verbose == 0):
                    if is_logistic:
                        print(f"[{iteration}] train-logloss: {train_score:.6f}, "
                              f"val-logloss: {val_score:.6f}")
                    else:
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
            Data to predict. A 1-D array/list is reshaped to (-1, n_features).
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
        if self.base_score is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.array(X, dtype=float)
        # A flat array is either one sample or a column of a 1-feature dataset;
        # n_features (recorded during fit) says which.
        if X.ndim == 1:
            X = X.reshape(-1, self.n_features)
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
            - For regression: R^2 score (1.0 is perfect, 0.0 is the mean baseline).
              A constant target has no variance to explain, so R^2 is undefined
              there; following sklearn's convention this returns 1.0 if the
              constant is reproduced and 0.0 if it is not.
            - For classification: Accuracy (fraction of labels predicted correctly)
        """
        y = np.array(y, dtype=float)
        predictions = self.predict(X)

        if self.objective in ['binary:logistic', 'reg:logistic']:
            # Classification: accuracy
            predicted_classes = (predictions >= 0.5).astype(int)
            return np.mean(predicted_classes == y)
        else:
            # Regression: R^2 score
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - predictions) ** 2)
            # If y is constant, ss_total is 0 (up to rounding) and R^2 = 1 - 0/0
            # is undefined. Compare against the scale of y rather than to exact
            # zero: np.full(20, 9.9) leaves ss_total ~ 6e-29, not 0.0, and
            # dividing by that would print a nonsense R^2 of -1e+31.
            #
            # The tolerance must be eps^2-small, not merely small. The ratio
            # ss_total / sum(y^2) is essentially (std(y) / mean(y))^2, so a
            # tolerance t declares y "constant" whenever its RELATIVE spread
            # drops below sqrt(t). Float64 rounding alone puts a floor of about
            # eps^2 = 4.9e-32 on that ratio (measured: 3.2e-32 for
            # np.full(20, 9.9)), so 1e-24 clears the rounding floor by ~1e7
            # while only firing below a relative spread of 1e-12. A tolerance of
            # 1e-12 would instead fire at a relative spread of 1e-6 and return
            # 1.0 for an ordinary narrow-range target whose R^2 is well defined.
            eps_sq = 1e-24
            scale = max(np.sum(y ** 2), 1.0)
            if ss_total <= eps_sq * scale:
                # Follow sklearn's convention for a constant target:
                # 1.0 if we reproduce it, 0.0 otherwise.
                return 1.0 if ss_residual <= eps_sq * scale else 0.0
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
            - 'cover': Average node "cover", i.e. the sum of hessians reaching the
              node. For squared error h=1 so cover equals the sample count; for
              logistic loss h = p(1-p), so cover measures how much *uncertain*
              mass a split touches, not merely how many rows.

        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Feature importance scores (normalized to sum to 1)
        """
        if self.n_features is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

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
                # XGBoost defines cover as the sum of second-order gradients
                # (hessians) at the node, not the raw row count.
                importance[tree['feature']] += tree['hessian']
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

# Generate non-linear data: y = x^2 + noise
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Shuffle before splitting so train and test cover the same x range
# (trees cannot extrapolate: without shuffling, test x > all train x)
idx = np.random.permutation(200)
X, y = X[idx], y[idx]

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

print(f"Training R2: {train_score:.4f}")
print(f"Test R2: {test_score:.4f}")

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
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

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
# (kept small on purpose: the exact-greedy split scan costs
#  O(n_samples * n_features) gain evaluations per node, so runtime grows
#  roughly with n_samples^2 -- see "Computational Complexity" in the .md)
np.random.seed(42)
X = np.random.randn(300, 8)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] ** 2 + np.random.randn(300) * 0.5

# Split train/validation/test (disjoint slices)
X_train, X_val, X_test = X[:180], X[180:240], X[240:]
y_train, y_val, y_test = y[:180], y[180:240], y[240:]

# Train with early stopping
model = XGBoost(
    n_estimators=150,  # Set high, will stop early
    learning_rate=0.1,
    max_depth=4,
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
print(f"Test R2: {test_score:.4f}")
"""

"""
USAGE EXAMPLE 4: Feature Importance Analysis

import numpy as np

# Create dataset with informative and noise features
np.random.seed(42)
n_samples = 200

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
    n_estimators=40,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=2.0,
    colsample_bytree=0.8
)
model.fit(X, y)

# Get feature importance (different types)
importance_weight = model.get_feature_importance('weight')
importance_gain = model.get_feature_importance('gain')
importance_cover = model.get_feature_importance('cover')

# Bars use plain ASCII '#' so the output prints on any console encoding
print("\nFeature Importance (by weight):")
print("="*50)
for i, imp in enumerate(importance_weight):
    bar = '#' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")

print("\nFeature Importance (by gain):")
print("="*50)
for i, imp in enumerate(importance_gain):
    bar = '#' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")

print("\nFeature Importance (by cover = average sum of hessians per split):")
print("="*50)
for i, imp in enumerate(importance_cover):
    bar = '#' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")
"""

"""
USAGE EXAMPLE 5: Comparing Regularization Parameters

import numpy as np

# Generate data with strong overfitting potential:
# only 3 of the 8 features carry signal, the noise is large, and deep trees
# on 100 rows can memorize almost all of it.
np.random.seed(42)
X = np.random.randn(150, 8)
y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + np.random.randn(150) * 1.5

X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]

# Test different regularization settings
configs = [
    {'reg_lambda': 0.0, 'reg_alpha': 0.0, 'name': 'No regularization'},
    {'reg_lambda': 10.0, 'reg_alpha': 0.0, 'name': 'L2 (Ridge)'},
    {'reg_lambda': 0.0, 'reg_alpha': 10.0, 'name': 'L1 (Lasso)'},
    {'reg_lambda': 10.0, 'reg_alpha': 10.0, 'name': 'Elastic Net'},
]

print("Effect of Regularization:")
print("="*80)
print(f"{'Configuration':<25} {'Train R2':>15} {'Test R2':>15} {'Overfit':>15}")
print("-"*80)

for config in configs:
    model = XGBoost(
        n_estimators=40,
        learning_rate=0.1,
        max_depth=6,
        reg_lambda=config['reg_lambda'],
        reg_alpha=config['reg_alpha']
    )
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfit = train_score - test_score

    print(f"{config['name']:<25} {train_score:>15.4f} {test_score:>15.4f} {overfit:>15.4f}")

# Observation: regularization trades training fit for honesty.
# The un-regularized model memorizes the noise (train R2 close to 1.0) and has
# by far the largest train-test gap. Both penalties shrink leaf weights, which
# costs a lot of train R2 and narrows that gap: about 17% with L2 alone, 28%
# with L1 alone, 40% with both. Test R2 itself barely moves, because with only
# 3 weak signals under heavy noise there is not much more to extract -- the win
# is that the model stops being over-confident.
# Note that at the same numeric value L1 shrinks harder than L2 here: alpha is
# subtracted from the gradient sum G, while lambda is added to the hessian sum H
# (which is ~n_samples in the leaf, so 10 barely dents it).
"""

"""
USAGE EXAMPLE 6: Effect of Tree Depth and Complexity

import numpy as np

# Complex non-linear data
np.random.seed(42)
X = np.random.randn(200, 6)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 +
     np.sin(X[:, 2]) * X[:, 3] +
     np.random.randn(200) * 0.5)

X_train, X_test = X[:140], X[140:]
y_train, y_test = y[:140], y[140:]

# Test different depths
depths = [2, 4, 6, 8]

print("\nEffect of Max Depth:")
print("="*80)
print(f"{'Max Depth':>12} {'Train R2':>15} {'Test R2':>15} {'Trees Used':>15}")
print("-"*80)

for depth in depths:
    model = XGBoost(
        n_estimators=40,
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
X = np.random.randn(150, 12)
# Only first 5 features are informative
y = (2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] -
     0.5 * X[:, 3] + X[:, 4] + np.random.randn(150) * 0.5)

X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]

# Test different colsample_bytree values
colsample_values = [0.3, 0.5, 0.7, 1.0]

print("\nEffect of Column Subsampling:")
print("="*80)
print(f"{'Colsample':>12} {'Train R2':>15} {'Test R2':>15} {'Overfit':>15}")
print("-"*80)

for colsample in colsample_values:
    model = XGBoost(
        n_estimators=40,
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

# Observation: column subsampling is NOT a free win, and this dataset shows why.
# All 5 informative features are independent and each is genuinely needed, so a
# tree that only sees 30% of the 12 columns often cannot see the column it needs:
# test R2 falls and the train-test gap widens as colsample_bytree drops.
# Column subsampling pays off in the opposite situation -- many features that are
# redundant or correlated -- where forcing trees to use different columns adds
# ensemble diversity without losing signal. Try it on your own data before
# assuming colsample_bytree < 1.0 helps.
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
print(f"Test R2: {test_r2:.4f}")
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


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _17_xgboost.py
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Regression demo: predict y = x^2 + noise ---
    print("=" * 55)
    print("DEMO 1 - Regression: y = x^2 + noise")
    print("=" * 55)

    X_reg = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_reg = X_reg.ravel() ** 2 + np.random.randn(200) * 0.5
    # Shuffle so train and test cover the same x range
    idx_reg = np.random.permutation(200)
    X_reg, y_reg = X_reg[idx_reg], y_reg[idx_reg]
    X_tr, X_te = X_reg[:150], X_reg[150:]
    y_tr, y_te = y_reg[:150], y_reg[150:]

    reg_model = XGBoost(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=1.0,
        gamma=0.1
    )
    reg_model.fit(X_tr, y_tr)

    preds = reg_model.predict(X_te)
    print(f"Train R2 : {reg_model.score(X_tr, y_tr):.4f}")
    print(f"Test  R2 : {reg_model.score(X_te, y_te):.4f}")
    print("\nSample predictions (x, true, predicted):")
    for i in range(5):
        print(f"  x={X_te[i, 0]:5.2f}  true={y_te[i]:5.2f}  pred={preds[i]:5.2f}")

    # --- Classification demo: two Gaussian blobs ---
    print("\n" + "=" * 55)
    print("DEMO 2 - Binary Classification: two Gaussian blobs")
    print("=" * 55)

    X0 = np.random.randn(100, 2) + np.array([-2, -2])
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X_cls = np.vstack([X0, X1])
    y_cls = np.array([0] * 100 + [1] * 100)
    idx = np.random.permutation(200)
    X_cls, y_cls = X_cls[idx], y_cls[idx]
    X_tr2, X_te2 = X_cls[:150], X_cls[150:]
    y_tr2, y_te2 = y_cls[:150], y_cls[150:]

    cls_model = XGBoost(
        n_estimators=50,
        learning_rate=0.3,
        max_depth=3,
        objective='binary:logistic',
        reg_lambda=1.0
    )
    cls_model.fit(X_tr2, y_tr2)

    print(f"Train Accuracy : {cls_model.score(X_tr2, y_tr2):.2%}")
    print(f"Test  Accuracy : {cls_model.score(X_te2, y_te2):.2%}")
    probas = cls_model.predict_proba(X_te2)
    print("\nSample predictions (true, P(0), P(1)):")
    for i in range(5):
        print(f"  true={int(y_te2[i])}  "
              f"P(class=0)={probas[i, 0]:.3f}  "
              f"P(class=1)={probas[i, 1]:.3f}")
