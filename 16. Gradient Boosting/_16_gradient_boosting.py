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

    Initialization and Leaf Values (Friedman, Algorithm 1):
        Two constants have to be chosen per loss function - the starting value F_0
        and the constant each tree leaf holds. The leaf constant comes from the
        "terminal region line search", step 2c of Friedman's algorithm:

            gamma_j = argmin_gamma sum_{x_i in leaf_j} L(y_i, F_{m-1}(x_i) + gamma)

        Working that argmin out per loss gives closed forms, and this class uses them:

            loss='mse'       F_0 = mean(y)              leaf = mean(r)
            loss='mae'       F_0 = median(y)            leaf = median(r)
            loss='log_loss'  F_0 = log(p / (1 - p))     leaf = sum(r) / sum(p * (1 - p))

        For 'mae' with an even number of residuals every value between the two
        central ones is an equally good minimiser - np.median's average of the
        two included. The code takes the lower central residual,
        sorted(r)[(n - 1) // 2], as a tie-break rather than as a better argmin:
        it is an actually observed value, and it is what scikit-learn's
        weighted 50th percentile returns.

        where r = y - F_{m-1}(x) are the residuals of the samples in that leaf and
        p = sigmoid(F_{m-1}(x)). The tree structure is always grown on the negative
        gradient; only the leaf constants are re-derived from the raw loss.
        See _update_leaf_values().

    Simplifications vs. canonical Gradient Boosting:
        - Only 'mse', 'mae' and 'log_loss' are supported (no Huber, no quantile,
          no user-supplied loss callable).
        - Binary classification only - no multiclass one-vs-all boosting.
        - No early stopping / validation monitoring, and no warm start: every call
          to fit() rebuilds the ensemble from scratch.
        - The split search rescans every candidate threshold, which is O(n^2) per
          node instead of the O(n log n) pre-sorted scan real libraries use.
        See "Simplifications vs. Canonical Gradient Boosting" in the .md for detail.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2,
                 loss='mse', subsample=1.0, random_state=None):
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

        random_state : int or None, default=None
            Seed for the subsampling draw, so that stochastic boosting
            (subsample < 1.0) is reproducible
            - None: use the global numpy RNG (honours np.random.seed(...))
            - int: use a private RandomState, independent of global state
            Typical: any fixed integer, e.g. 42
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.init_prediction = None
        self.n_features = None
        self.train_loss_ = []

    def _sigmoid(self, z):
        """
        Numerically stable logistic function: 1 / (1 + exp(-z))

        Raw boosting scores can grow past +/-709, where np.exp overflows.
        Clipping first keeps the result at exactly 0.0 or 1.0 without warnings.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _mse_loss(self, y_true, y_pred):
        """
        Mean Squared Error loss: mean((y_true - y_pred) ** 2)

        Watch the scale. _mse_gradient differentiates the conventional HALVED
        squared error, L = (1/2)(y - F)^2, because the 1/2 cancels the 2 the
        derivative produces and leaves the clean dL/dF = F - y. This function
        reports the plain un-halved MSE, so the number stored in train_loss_
        is exactly twice mean((1/2)(y - F)^2). A constant factor cannot move a
        minimum, so the trees are identical either way - only the printed
        value differs.
        """
        return np.mean((y_true - y_pred) ** 2)

    def _mae_loss(self, y_true, y_pred):
        """Mean Absolute Error loss"""
        return np.mean(np.abs(y_true - y_pred))

    def _log_loss(self, y_true, raw_prediction):
        """
        Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))

        `raw_prediction` is the un-squashed ensemble score F(x); p = sigmoid(F(x)).
        p is clipped away from 0 and 1 so perfectly separated data cannot give -inf.
        """
        p = np.clip(self._sigmoid(raw_prediction), 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def _compute_loss(self, y_true, raw_prediction):
        """
        Current training loss, recorded once per boosting round in fit().

        Gradient boosting is gradient descent in function space, so this
        sequence should decrease (it is what the gradients are pushing down).
        Stored in self.train_loss_. For 'mse' it is the un-halved MSE, i.e.
        twice the (1/2)(y - F)^2 the gradient comes from - see _mse_loss.
        """
        if self.loss == 'mse':
            return self._mse_loss(y_true, raw_prediction)
        elif self.loss == 'mae':
            return self._mae_loss(y_true, raw_prediction)
        elif self.loss == 'log_loss':
            return self._log_loss(y_true, raw_prediction)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

    def _mse_gradient(self, y_true, y_pred):
        """Gradient of L = (1/2)(y - F)^2: dL/dF = F - y, the negative residuals"""
        return y_pred - y_true

    def _mae_gradient(self, y_true, y_pred):
        """Gradient of MAE: sign of residuals"""
        return np.sign(y_pred - y_true)

    def _log_loss_gradient(self, y_true, y_pred):
        """Gradient of log loss for binary classification"""
        # Sigmoid of predictions
        proba = self._sigmoid(y_pred)
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

        It decides the tree STRUCTURE only. For 'mae' and 'log_loss', fit() then
        calls _update_leaf_values() to replace each leaf's mean with the constant
        that minimises the real loss (Friedman's step 2c).

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
            - For split: 'feature', 'threshold', 'gain', 'left', 'right'
              'gain' is the variance reduction this split achieved; it is what
              get_feature_importance() accumulates.
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
        # NOTE ON COST: for every candidate threshold we rebuild a boolean mask and
        # call np.var twice, so this is O(n_samples^2 * n_features) work per node.
        # Real libraries sort each feature once and sweep running sums for
        # O(n_samples * log(n_samples)). We keep the slow version on purpose - it
        # shows the variance-reduction formula literally. Keep demo data small.
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]

            # Candidate thresholds are MIDPOINTS between consecutive distinct
            # values, which is what scikit-learn uses. Splitting on a raw training
            # value puts the boundary right on top of a data point, so an unseen
            # point landing in the gap can fall on the wrong side.
            distinct_values = np.unique(feature_values)
            if len(distinct_values) < 2:
                continue
            thresholds = (distinct_values[:-1] + distinct_values[1:]) / 2

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
                
                # Strict '>' keeps the FIRST candidate seen on a tie (features in
                # index order, thresholds ascending). Exact ties do occur when the
                # gradient is two-valued, as it is for 'mae' (+-1): 5 of the first
                # 30 rounds of the .md's x^2 'mae' reference check tie at the root.
                # At round 26 (0-indexed) the tied gain is 5929/900, yet np.var
                # evaluates the two candidates 2.8e-14 apart, so which equally
                # optimal split wins is settled by rounding rather than by this
                # rule - and a library that computes the same gain by a different
                # formula can round toward the other one.
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
            'gain': best_gain,
            'left': left_tree,
            'right': right_tree
        }

    def _update_leaf_values(self, tree, X, y_true, current_predictions):
        """
        Friedman's terminal-region line search - step 2c of Algorithm 1.

        The tree structure was grown on the negative gradient, so its leaves hold
        the MEAN gradient. That is only the loss-minimising constant for MSE. For
        every other loss we have to solve, per leaf j,

            gamma_j = argmin_gamma sum_{x_i in leaf_j} L(y_i, F_{m-1}(x_i) + gamma)

        which has a closed form:
            'mae'      -> median of the residuals r_i = y_i - F_{m-1}(x_i)
                          (the value minimising sum |r_i - gamma|)
            'log_loss' -> one Newton step, sum(r_i) / sum(p_i * (1 - p_i)),
                          with r_i = y_i - p_i and p_i = sigmoid(F_{m-1}(x_i))

        Without this step 'mae' leaves hold mean(sign(r)), which lives in [-1, 1],
        so the whole ensemble could only move a prediction by n_estimators *
        learning_rate; and 'log_loss' leaves hold mean(y - p), which is bounded by
        1 and leaves the model badly under-confident.

        Parameters:
        -----------
        tree : dict
            Tree whose leaf 'value' entries are overwritten in place
        X : np.ndarray, shape (n_samples, n_features)
            Samples this subtree is responsible for
        y_true : np.ndarray, shape (n_samples,)
            Their true targets
        current_predictions : np.ndarray, shape (n_samples,)
            Their ensemble scores F_{m-1}(x) BEFORE this tree is added
        """
        if tree['type'] == 'leaf':
            if len(y_true) == 0:
                return

            if self.loss == 'mae':
                # Least-absolute-deviation: the minimiser is the median residual.
                # With an even number of residuals EVERY value between the two
                # central ones minimises sum|r_i - gamma| equally, np.median's
                # average of the two included. Taking the lower central residual
                # is therefore a tie-break, not a better argmin: it keeps the leaf
                # on an observed value and it matches scikit-learn's weighted 50th
                # percentile, which is why the two agree exactly on the uniform-X
                # 'mae' reference check in the .md. The two choices differ only in
                # even-sized leaves, and only by HALF the gap between the two
                # central residuals - widest in a two-sample leaf, where np.median
                # would average an outlier in.
                residuals = np.sort(y_true - current_predictions)
                tree['value'] = residuals[(len(residuals) - 1) // 2]

            elif self.loss == 'log_loss':
                # Newton step: -(sum of gradients) / (sum of hessians)
                p = self._sigmoid(current_predictions)
                numerator = np.sum(y_true - p)
                denominator = np.sum(p * (1 - p))
                # On perfectly separated data every p saturates and the hessian
                # sum underflows to 0; sklearn leaves such a region unchanged.
                if denominator < 1e-150:
                    tree['value'] = 0.0
                else:
                    tree['value'] = numerator / denominator

            # For 'mse' the mean of the negative gradient already IS the argmin,
            # so the leaf value the tree was built with needs no correction.
            return

        # Route each sample down the same branch predict() would take
        left_mask = X[:, tree['feature']] <= tree['threshold']
        self._update_leaf_values(tree['left'], X[left_mask], y_true[left_mask],
                                 current_predictions[left_mask])
        self._update_leaf_values(tree['right'], X[~left_mask], y_true[~left_mask],
                                 current_predictions[~left_mask])

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
        
        Algorithm (Friedman's Algorithm 1):
        1. Initialize predictions with the loss-minimising constant F_0
           (mean for 'mse', median for 'mae', log-odds for 'log_loss')
        2. For t = 1 to n_estimators:
           a. Calculate negative gradient (pseudo-residuals)
           b. Sample subset of data if subsample < 1.0
           c. Fit a tree to the negative gradient
           d. Line search: replace each leaf constant with the value that
              minimises the ORIGINAL loss for the samples in that leaf
              (see _update_leaf_values; a no-op for 'mse')
           e. Update predictions: F(x) = F(x) + learning_rate * tree(x)
        3. Final model: Sum of all trees with learning rate scaling

        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data. A 1-D array or flat list is treated as a single
            feature and reshaped to (n_samples, 1).
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

        # Accept a flat list / 1-D array as one feature column
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features = n_features

        # Reproducible subsampling: a private RandomState when a seed is given,
        # otherwise the global RNG so np.random.seed(...) still controls the run
        rng = np.random if self.random_state is None else np.random.RandomState(self.random_state)

        # Initialize predictions with F_0 = argmin_gamma sum_i L(y_i, gamma)
        if self.loss == 'log_loss':
            # For classification, initialize with log-odds
            p = np.mean(y)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
            self.init_prediction = np.log(p / (1 - p))
        elif self.loss == 'mae':
            # Absolute error is minimised by the median, not the mean
            self.init_prediction = np.median(y)
        else:
            # For squared-error regression, initialize with mean
            self.init_prediction = np.mean(y)

        # Current predictions (start with initialization)
        current_predictions = np.full(n_samples, self.init_prediction)

        self.trees = []
        self.train_loss_ = []

        # Train trees sequentially
        for i in range(self.n_estimators):
            # Calculate negative gradient (pseudo-residuals)
            gradients = -self._get_gradient(y, current_predictions)

            # Subsample data
            if self.subsample < 1.0:
                sample_size = int(n_samples * self.subsample)
                indices = rng.choice(n_samples, sample_size, replace=False)
                X_sample = X[indices]
                gradients_sample = gradients[indices]
                y_sample = y[indices]
                predictions_sample = current_predictions[indices]
            else:
                X_sample = X
                gradients_sample = gradients
                y_sample = y
                predictions_sample = current_predictions

            # Fit tree to negative gradient (this decides the tree STRUCTURE)
            tree = self._create_decision_tree(X_sample, gradients_sample)

            # Line search: re-derive each leaf CONSTANT from the original loss.
            # For 'mse' the mean gradient is already the minimiser, so skip it.
            if self.loss != 'mse':
                self._update_leaf_values(tree, X_sample, y_sample, predictions_sample)

            self.trees.append(tree)

            # Update predictions for all samples
            tree_predictions = self._predict_tree(tree, X)
            current_predictions += self.learning_rate * tree_predictions

            # Record the training loss so callers can see it going down
            self.train_loss_.append(self._compute_loss(y, current_predictions))

        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Combines initial prediction with all trees:
        F(x) = F_0 + learning_rate * sum_i tree_i(x)

        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict. A 1-D array or flat list is treated as a single
            feature and reshaped to (n_samples, 1).

        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values
            - For regression: continuous values
            - For classification: probabilities after sigmoid
        """
        if self.init_prediction is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) before predict(X).")

        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]

        # Start with initial prediction
        predictions = np.full(n_samples, self.init_prediction)

        # Add contribution from each tree
        for tree in self.trees:
            tree_predictions = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_predictions

        # For classification, convert to probabilities
        if self.loss == 'log_loss':
            predictions = self._sigmoid(predictions)

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
            # A constant y has zero variance, so R² is undefined. Follow
            # scikit-learn: 1.0 if we predict it exactly, 0.0 otherwise.
            if ss_total == 0:
                return 1.0 if ss_residual == 0 else 0.0
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
        if self.init_prediction is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) before staged_predict(X).")

        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]

        staged_predictions = []
        current_predictions = np.full(n_samples, self.init_prediction)

        for tree in self.trees:
            tree_predictions = self._predict_tree(tree, X)
            current_predictions = current_predictions + self.learning_rate * tree_predictions

            # For classification, convert to probabilities
            if self.loss == 'log_loss':
                staged_predictions.append(self._sigmoid(current_predictions))
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
                # Regression: R² score (same constant-target guard as score())
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - predictions) ** 2)
                if ss_total == 0:
                    score = 1.0 if ss_residual == 0 else 0.0
                else:
                    score = 1 - (ss_residual / ss_total)

            scores.append(score)
        
        return scores
    
    def get_feature_importance(self):
        """
        Calculate feature importance based on total variance reduction

        Importance is calculated as the sum of variance reduction
        from all splits on each feature across all trees.

        Each split node stores the gain it achieved,
            gain = var(parent) * n_parent - (var(left) * n_left + var(right) * n_right)
        and this method sums those gains per feature, then normalizes. Counting
        splits instead would rank a feature used for many tiny splits above one
        used for a few decisive ones.

        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Normalized feature importance (sums to 1)
            importance[i] = importance of feature i
        """
        if self.n_features is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) before get_feature_importance().")

        importance = np.zeros(self.n_features)

        def _accumulate_importance(tree):
            """Recursively accumulate importance from tree"""
            if tree['type'] == 'leaf':
                return

            # Add the variance reduction this split actually achieved
            importance[tree['feature']] += tree['gain']

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

# Shuffle BEFORE splitting. X was generated in sorted order, so slicing it
# directly would put every test point to the right of the training range,
# and trees cannot extrapolate - every prediction would come out identical.
indices = np.random.permutation(200)
X, y = X[indices], y[indices]

# Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Create and train model
model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
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

# Split (the slices must not overlap: 0..149 for train, 150..199 for test)
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

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

# Shuffle first (X is sorted), then split into two disjoint halves
indices = np.random.permutation(100)
X, y = X[indices], y[indices]

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

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
plt.ylabel('R2 Score', fontsize=12)
plt.title('Gradient Boosting Learning Curves', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

# Find optimal number of trees
optimal_n = np.argmax(test_scores) + 1
print(f"Optimal number of trees: {optimal_n}")
print(f"Best test R2: {test_scores[optimal_n-1]:.4f}")
"""

"""
USAGE EXAMPLE 4: Feature Importance Analysis

import numpy as np

# Create dataset with 6 features (only first 3 are informative)
# Kept small on purpose: the split search is O(n_samples^2 * n_features) per
# node, so doubling the rows roughly quadruples the training time.
np.random.seed(42)
n_samples = 150

# Informative features
X1 = np.random.randn(n_samples, 1)
X2 = np.random.randn(n_samples, 1)
X3 = np.random.randn(n_samples, 1)

# Non-informative features (noise)
X_noise = np.random.randn(n_samples, 3)

X = np.hstack([X1, X2, X3, X_noise])

# Target depends on first 3 features
y = 2 * X1.ravel() + 3 * X2.ravel() - X3.ravel() + np.random.randn(n_samples) * 0.5

# Train model
model = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=4)
model.fit(X, y)

# Get feature importance (summed variance reduction per feature, normalized)
importance = model.get_feature_importance()

print("\nFeature Importance:")
print("="*50)
for i, imp in enumerate(importance):
    bar = '#' * int(imp * 50)   # ASCII only: block characters crash cp1252 consoles
    print(f"Feature {i:2d}: {imp:.4f} {bar}")

# Expected: Features 0, 1, 2 have high importance, rest are low
"""

"""
USAGE EXAMPLE 5: Comparing Learning Rates

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(120, 3)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] + np.random.randn(120) * 0.5

# Disjoint split: rows 0..89 train, rows 90..119 test
X_train, X_test = X[:90], X[90:]
y_train, y_test = y[:90], y[90:]

# Try different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.3]
n_estimators = 60

print("Effect of Learning Rate:")
print("="*70)
print(f"{'Learning Rate':>15} {'n_estimators':>15} {'Train R2':>15} {'Test R2':>15}")
print("-"*70)

for lr in learning_rates:
    model = GradientBoosting(n_estimators=n_estimators, learning_rate=lr, max_depth=3)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"{lr:>15.2f} {n_estimators:>15} {train_score:>15.4f} {test_score:>15.4f}")

# Observation: it is learning_rate * n_estimators that sets how far the ensemble
# can travel. With the budget fixed at 60 trees, lr=0.01 has not finished fitting
# yet (it underfits), so the higher rates win here. Give lr=0.01 a few thousand
# trees instead and it overtakes them - that is the shrinkage trade-off.
"""

"""
USAGE EXAMPLE 6: Effect of Tree Depth

import numpy as np

# Cost note: this is the slowest example in the file. Five depths x 100 trees on
# 150 rows takes about 20 s, because the split search is O(n_samples^2 *
# n_features) per node and deeper trees mean more nodes (depth 8 alone is ~8 s).

# Generate complex non-linear data
np.random.seed(42)
X = np.random.randn(200, 3)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 - X[:, 2] + 
     np.sin(X[:, 0]) + np.random.randn(200) * 0.3)

# Disjoint split: rows 0..149 train, rows 150..199 test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Try different max depths
depths = [1, 2, 3, 5, 8]

print("\nEffect of Tree Depth:")
print("="*70)
print(f"{'Max Depth':>15} {'Train R2':>15} {'Test R2':>15} {'Difference':>15}")
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
X = np.random.randn(200, 4)
y = (X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] +
     np.random.randn(200) * 0.5)

# Disjoint split: rows 0..149 train, rows 150..199 test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Compare different subsample ratios
subsample_ratios = [0.5, 0.7, 0.9, 1.0]

print("\nEffect of Subsampling:")
print("="*70)
print(f"{'Subsample':>15} {'Train R2':>15} {'Test R2':>15} {'Overfitting':>15}")
print("-"*70)

for subsample in subsample_ratios:
    model = GradientBoosting(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        subsample=subsample,
        random_state=42   # makes the stochastic draw reproducible
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

# Split data (disjoint: rows 0..159 train, rows 160..199 test)
X_train, X_test = X[:160], X[160:]
y_train, y_test = price[:160], price[160:]

# Train model
model = GradientBoosting(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print("\nHouse Price Prediction Model:")
print("="*60)
print(f"Training R2: {train_r2:.4f}")
print(f"Test R2: {test_r2:.4f}")

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

# Split (disjoint: rows 0..299 train, rows 300..399 test)
X_train, X_test = X[:300], X[300:]
y_train, y_test = y[:300], y[300:]

# Train diagnostic model
model = GradientBoosting(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    loss='log_loss',
    subsample=0.8,
    random_state=42
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


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _16_gradient_boosting.py
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Regression demo: predict y = x^2 + noise ---
    print("=" * 60)
    print("DEMO 1 - Regression (loss='mse'): y = x^2 + noise")
    print("=" * 60)

    X_reg = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_reg = X_reg.ravel() ** 2 + np.random.randn(200) * 0.5
    # Shuffle so train and test cover the same x range. Trees cannot
    # extrapolate: without this the test set would lie entirely to the
    # right of the training data and every prediction would be the same.
    idx_reg = np.random.permutation(200)
    X_reg, y_reg = X_reg[idx_reg], y_reg[idx_reg]
    X_tr, X_te = X_reg[:150], X_reg[150:]
    y_tr, y_te = y_reg[:150], y_reg[150:]

    reg_model = GradientBoosting(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    reg_model.fit(X_tr, y_tr)

    preds = reg_model.predict(X_te)
    print(f"Train R2 : {reg_model.score(X_tr, y_tr):.4f}")
    print(f"Test  R2 : {reg_model.score(X_te, y_te):.4f}")
    print(f"Training loss (MSE): {reg_model.train_loss_[0]:.4f} after 1 tree "
          f"-> {reg_model.train_loss_[-1]:.4f} after 100 trees")
    print("\nSample predictions (x, true, predicted):")
    for i in range(5):
        print(f"  x={X_te[i, 0]:5.2f}  true={y_te[i]:5.2f}  pred={preds[i]:5.2f}")

    # Robustness: corrupt 8 training targets and compare squared vs absolute loss.
    # 'mae' uses the MEDIAN of the residuals in each leaf, so a few huge errors
    # cannot drag the leaf constant around the way a mean can.
    y_tr_dirty = y_tr.copy()
    y_tr_dirty[:8] += 60.0
    mse_dirty = GradientBoosting(n_estimators=100, learning_rate=0.1,
                                 max_depth=3, loss='mse').fit(X_tr, y_tr_dirty)
    mae_dirty = GradientBoosting(n_estimators=100, learning_rate=0.1,
                                 max_depth=3, loss='mae').fit(X_tr, y_tr_dirty)
    print("\nWith 8 corrupted training targets (+60), scored on the clean test set:")
    print(f"  loss='mse' Test R2 : {mse_dirty.score(X_te, y_te):.4f}")
    print(f"  loss='mae' Test R2 : {mae_dirty.score(X_te, y_te):.4f}  (robust)")

    # --- Classification demo: two Gaussian blobs ---
    print("\n" + "=" * 60)
    print("DEMO 2 - Binary Classification (loss='log_loss')")
    print("=" * 60)

    X0 = np.random.randn(100, 2) + np.array([-2, -2])
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X_cls = np.vstack([X0, X1])
    y_cls = np.array([0] * 100 + [1] * 100)
    idx = np.random.permutation(200)
    X_cls, y_cls = X_cls[idx], y_cls[idx]
    X_tr2, X_te2 = X_cls[:150], X_cls[150:]
    y_tr2, y_te2 = y_cls[:150], y_cls[150:]

    # 20 rounds is plenty here. The Newton leaf update (see _update_leaf_values)
    # makes each round push the log-odds hard, and these blobs are cleanly
    # separable, so by round 50 every probability has saturated to exactly 0 or 1.
    cls_model = GradientBoosting(
        n_estimators=20,
        learning_rate=0.3,
        max_depth=3,
        loss='log_loss'
    )
    cls_model.fit(X_tr2, y_tr2)

    print(f"Train Accuracy : {cls_model.score(X_tr2, y_tr2):.2%}")
    print(f"Test  Accuracy : {cls_model.score(X_te2, y_te2):.2%}")
    print(f"Training log loss: {cls_model.train_loss_[0]:.4f} after 1 tree "
          f"-> {cls_model.train_loss_[-1]:.6f} after 20 trees")
    probas = cls_model.predict_proba(X_te2)
    print("\nSample predictions (true, P(0), P(1)):")
    for i in range(5):
        print(f"  true={int(y_te2[i])}  "
              f"P(class=0)={probas[i, 0]:.4f}  "
              f"P(class=1)={probas[i, 1]:.4f}")

    # --- Learning curve + feature importance ---
    print("\n" + "=" * 60)
    print("DEMO 3 - Learning Curve and Feature Importance")
    print("=" * 60)

    X_fi = np.random.randn(120, 4)
    # Only features 0 and 1 carry signal; 2 and 3 are pure noise
    y_fi = 3 * X_fi[:, 0] - 2 * X_fi[:, 1] + np.random.randn(120) * 0.4
    idx_fi = np.random.permutation(120)
    X_fi, y_fi = X_fi[idx_fi], y_fi[idx_fi]
    X_tr3, X_te3 = X_fi[:90], X_fi[90:]
    y_tr3, y_te3 = y_fi[:90], y_fi[90:]

    fi_model = GradientBoosting(n_estimators=40, learning_rate=0.15, max_depth=3)
    fi_model.fit(X_tr3, y_tr3)

    stage_scores = fi_model.staged_score(X_te3, y_te3)
    print("Test R2 as trees are added:")
    for n_trees in [1, 10, 40]:
        print(f"  after {n_trees:2d} tree(s): R2 = {stage_scores[n_trees - 1]:.4f}")

    print(f"\nTrain R2 : {fi_model.score(X_tr3, y_tr3):.4f}")
    print(f"Test  R2 : {fi_model.score(X_te3, y_te3):.4f}")

    print("\nFeature importance (summed variance reduction, normalized):")
    for i, imp in enumerate(fi_model.get_feature_importance()):
        bar = '#' * int(imp * 50)
        label = "signal" if i < 2 else "noise "
        print(f"  Feature {i} ({label}): {imp:.4f} {bar}")

