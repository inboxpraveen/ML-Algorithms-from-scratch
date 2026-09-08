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
            (implemented here -- pass boosting_type='Ordered'; the default is
             'Plain', i.e. classic gradient boosting)
        Ordered Target Statistics: Smart categorical feature encoding
            (implemented here -- pass cat_features=[column indices])
        No need for extensive preprocessing: Handles categoricals natively,
            including string columns, without one-hot encoding
        Robust to overfitting: Built-in regularization through a high default
            L2 leaf penalty and the simplicity of symmetric trees

    Leaf Value and Split Score:
        Every leaf and every candidate split is scored from two sums over the
        samples that reach it -- G (gradients) and H (Hessians):

            w*    = -G / (H + l2_leaf_reg)              leaf value
            Score = G^2 / (H + l2_leaf_reg)             higher is better
            Gain  = sum over current partitions of
                    [ Score(left) + Score(right) - Score(parent) ]

        For squared loss h = 1, so H is exactly the sample count and these
        reduce to the count-based formulas usually quoted for CatBoost. For
        Logloss h = p(1-p), which is what real CatBoost's default 'Newton'
        leaf estimation uses; substituting the count there would under-step
        every update by roughly 3-4x.

    Ordered Target Statistic (categorical encoding):
        For row i in a random permutation sigma, using only earlier rows:

            TS_i = (sum of y_j for j before i with x_j == x_i + a * p)
                   / (count of those j + a)

        with p the global target mean (the prior) and a the smoothing weight.
        Row i's own target never enters TS_i, so there is no target leakage.

    See "Simplification vs. canonical CatBoost" in _19_catboost.md for the
    parts of the real library that are deliberately not reproduced here.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.03, depth=6,
                 l2_leaf_reg=3.0, min_data_in_leaf=1, random_strength=1.0,
                 border_count=128, objective='regression',
                 cat_features=None, boosting_type='Plain', random_seed=None):
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
            Note: the perturbation added to a candidate's gain is
            random_strength * std(all candidate gains at this level) * N(0,1).
            Scaling by the spread of the level's own gains is a heuristic (real
            CatBoost decays the magnitude over training); it is needed because a
            raw gain is measured in (target units)^2, so a fixed-size
            perturbation would behave completely differently for a target in
            dollars and the same target in thousands of dollars.
            
        border_count : int, default=128
            Number of splits for numerical features
            - Similar to LightGBM's max_bin
            - Higher values: More accurate but slower
            Typical values: 32, 64, 128, 254
            
        objective : str, default='regression'
            Learning objective
            - 'regression': Regression with RMSE loss
            - 'binary': Binary classification with logloss

        cat_features : list of int, default=None
            Column indices to treat as CATEGORICAL. Those columns may hold
            strings or arbitrary integer codes; they are converted to numbers
            with ordered target statistics (see `_ordered_target_statistics`)
            instead of one-hot or label encoding.
            - None or []: every column is numeric (previous behaviour)
            Typical: pass every genuinely nominal column

        boosting_type : str, default='Plain'
            How gradients are computed each iteration
            - 'Plain': classic gradient boosting -- every sample's gradient
              comes from the one shared model. Fast, and what XGBoost/LightGBM
              do, but the model has already fitted the sample it is being
              scored on (prediction shift).
            - 'Ordered': CatBoost's unbiased scheme -- a sample's gradient
              comes from a supporting model that never saw that sample.
              Costs ~log2(n) extra leaf-value computations per tree and helps
              most on small, noisy datasets.
            Typical: 'Ordered' below ~10k rows, 'Plain' above

        random_seed : int, default=None
            Seed for this model's private random generator (split jitter and
            the ordered-boosting permutation)
            - int: fully reproducible fits, independent of global RNG state
            - None: seeded from the global numpy RNG, so np.random.seed(42)
              before fitting still reproduces the run
            Typical: any fixed int when you need repeatability
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.random_strength = random_strength
        self.border_count = border_count
        self.objective = objective
        self.cat_features = cat_features
        self.boosting_type = boosting_type
        self.random_seed = random_seed
        # Number of permutations averaged when building ordered target
        # statistics. CatBoost also uses several; 4 is its usual count.
        self.ts_permutations = 4
        
        self.trees = []
        self.base_score = None
        self.feature_borders = None
        self.n_features = None
        self.train_scores = []
        self.val_scores = []
        self._cat_encodings = None
        self._rng = None
        
    def _sigmoid(self, x):
        """Sigmoid function with numerical stability"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _compute_gradients(self, y_true, y_pred):
        """
        Compute first-order gradients g = dL/dpred for the loss function
        
        Note on first vs second order: for squared loss the Hessian is the
        constant 1, so "count + l2_leaf_reg" in the leaf formula IS the exact
        Newton denominator -- this is what CatBoost calls the 'Gradient' leaf
        estimation method. For Logloss the Hessian is p(1-p) != 1, and real
        CatBoost defaults to 'Newton', dividing by sum(p*(1-p)) + l2_leaf_reg.
        See `_compute_hessians`, which supplies exactly that denominator here.
        
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
    
    def _compute_hessians(self, y_true, y_pred):
        """
        Compute second-order derivatives h = d2L/dpred2 (the Hessian diagonal)

        These are the weights that appear in the leaf value and split score:

            w*   = -G / (H + l2_leaf_reg)
            Score = G^2 / (H + l2_leaf_reg)

        where G = sum of gradients and H = sum of Hessians over a partition.

        - regression (squared loss):  L = 0.5*(y - pred)^2  ->  h = 1
          so H == number of samples, and the formulas reduce to the familiar
          count-based ones used in the .md's regression walkthrough.
        - binary (log loss):          h = p * (1 - p),  p = sigmoid(pred)
          Because p(1-p) <= 0.25, H is much smaller than the count. Using the
          count here would make every logistic step 3-4x too small, so the
          model could never become confident. This is CatBoost's 'Newton'
          leaf estimation method for Logloss.

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Current raw predictions (log-odds for the binary objective)

        Returns:
        --------
        hessians : np.ndarray
            Second-order derivatives, same shape as y_pred
        """
        if self.objective == 'regression':
            return np.ones_like(y_pred, dtype=float)
        elif self.objective == 'binary':
            p = self._sigmoid(y_pred)
            # Floor keeps the denominator from collapsing on saturated samples
            return np.maximum(p * (1 - p), 1e-6)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

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
                # Few distinct values: give every distinct value its own bin.
                # A border must fall strictly BETWEEN two neighbouring values,
                # never ON one, so we use midpoints. With k unique values this
                # gives k-1 borders and therefore k bins.
                # Example: values [0, 1] -> borders [0.5] -> bins {0 -> 0, 1 -> 1}.
                borders = (unique_values[:-1] + unique_values[1:]) / 2.0
            else:
                # Many distinct values: place borders at interior quantiles.
                # [1:-1] drops the 0th and 100th percentiles, which are the
                # min and max and would create an empty end bin.
                percentiles = np.linspace(0, 100, self.border_count + 1)[1:-1]
                borders = np.percentile(feature_values, percentiles)
                borders = np.unique(borders)
            
            self.feature_borders.append(borders)
            # digitize(v, borders) with the default right=False returns the number
            # of borders strictly below or equal to v, i.e. the bin index in 0..len(borders)
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
    
    def _calculate_leaf_value(self, gradients, indices, hessians=None):
        """
        Calculate optimal leaf value with L2 regularization
        
        CatBoost formula: value = -G / (H + l2_leaf_reg)
            G = sum of gradients in the leaf
            H = sum of Hessians in the leaf (== the sample count for squared
                loss, since h = 1 there -- which is why the .md's house-price
                walkthrough can divide by the number of houses in a leaf,
                while its logloss walkthrough must not)
        
        The L2 regularization in the denominator acts as smoothing:
        - More samples -> less regularization effect
        - Fewer samples -> more shrinkage toward zero
          (a 100-sample leaf keeps 100/103 = 97% of its unregularized value,
           a 5-sample leaf keeps only 5/8 = 62.5% of it)
        
        Parameters:
        -----------
        gradients : np.ndarray
            Gradients for all samples
        indices : np.ndarray (boolean)
            Boolean mask selecting the samples in this leaf
        hessians : np.ndarray, optional
            Hessians for all samples. None means "all ones" (squared loss).
            
        Returns:
        --------
        value : float
            Optimal leaf value
        """
        if np.sum(indices) == 0:
            return 0.0
        
        gradient_sum = np.sum(gradients[indices])
        if hessians is None:
            hessian_sum = np.sum(indices)   # h = 1 for squared loss
        else:
            hessian_sum = np.sum(hessians[indices])
        
        # CatBoost's leaf value formula with L2 regularization
        value = -gradient_sum / (hessian_sum + self.l2_leaf_reg)
        
        return value
    
    def _build_symmetric_tree(self, X_quantized, gradients, hessians=None):
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
        
        THE SCORE BEING MAXIMISED (same formula the .md states):

            Score(partition) = G^2 / (H + l2_leaf_reg)     (higher is better)
            Gain(split)      = sum over current partitions of
                               [ Score(left) + Score(right) - Score(parent) ]

        G is the sum of gradients and H the sum of Hessians in a partition
        (H == the sample count for squared loss, where h = 1). A positive gain
        means the split helps; the level keeps the single (feature, threshold)
        pair with the largest gain summed over ALL partitions -- that "summed
        over all partitions" is what makes the tree oblivious.

        HOW THE SEARCH IS ORGANISED. The straightforward way to evaluate a
        candidate is to re-mask every partition with a boolean comparison, but
        that repeats the same sums once per candidate threshold. Instead we
        accumulate, in ONE pass per feature per level, a histogram of G, H and
        counts over every (partition, bin) cell with np.bincount, then read off
        every threshold at once with a cumulative sum along the bin axis:
        cumsum up to bin t IS the left child for threshold t, and the row total
        minus it is the right child. Identical arithmetic, far fewer passes.

        Parameters:
        -----------
        X_quantized : np.ndarray, shape (n_samples, n_features)
            Quantized training data
        gradients : np.ndarray, shape (n_samples,)
            Gradients to optimize
        hessians : np.ndarray, shape (n_samples,), optional
            Hessians. None means "all ones", i.e. the squared-loss case where
            the Hessian sum is simply the sample count.
            
        Returns:
        --------
        tree : dict
            Symmetric tree structure
        """
        n_samples, n_features = X_quantized.shape
        if hessians is None:
            hessians = np.ones(n_samples)
        rng = self._rng if self._rng is not None else np.random
        lam = self.l2_leaf_reg

        # Candidate thresholds are the bin indices actually present in the data
        # (computed once per tree -- quantization does not change between levels)
        feature_thresholds = [np.unique(X_quantized[:, f]) for f in range(n_features)]
        feature_n_bins = [int(X_quantized[:, f].max()) + 1 for f in range(n_features)]
        
        # Store split conditions for each level
        splits = []
        
        # partition_id[i] tells which partition (future leaf) sample i sits in.
        # This one integer array replaces the old list of boolean masks. When a
        # partition p is split, its children are numbered 2p (left) and 2p+1
        # (right) -- exactly the leaf numbering _predict_tree rebuilds bit by bit.
        partition_id = np.zeros(n_samples, dtype=int)
        n_partitions = 1
        
        # Build tree level by level
        for level in range(self.depth):
            cand_features = []
            cand_thresholds = []
            cand_gains = []
            
            # Try all features; all their thresholds are scored in one shot
            for feature_idx in range(n_features):
                bins = X_quantized[:, feature_idx]
                n_bins = feature_n_bins[feature_idx]
                
                # Histogram of G, H and counts over every (partition, bin) cell
                code = partition_id * n_bins + bins
                size = n_partitions * n_bins
                g_hist = np.bincount(code, weights=gradients, minlength=size)
                h_hist = np.bincount(code, weights=hessians, minlength=size)
                c_hist = np.bincount(code, minlength=size).astype(float)
                g_hist = g_hist.reshape(n_partitions, n_bins)
                h_hist = h_hist.reshape(n_partitions, n_bins)
                c_hist = c_hist.reshape(n_partitions, n_bins)
                    
                # cumsum along bins = "everything with bin <= threshold" = LEFT child
                g_left = np.cumsum(g_hist, axis=1)
                h_left = np.cumsum(h_hist, axis=1)
                c_left = np.cumsum(c_hist, axis=1)
                        
                # Row totals are the parent; parent - left = RIGHT child
                g_parent = g_left[:, -1:]
                h_parent = h_left[:, -1:]
                c_parent = c_left[:, -1:]
                g_right = g_parent - g_left
                h_right = h_parent - h_left
                c_right = c_parent - c_left
                        
                # Similarity score G^2 / (H + lambda): higher is better, so a
                # positive gain means the split improves the objective.
                # With l2_leaf_reg=0 an empty partition gives 0/0 = nan here;
                # the `valid` mask below discards those cells, so we only need
                # to keep numpy from printing a warning about them.
                with np.errstate(invalid='ignore', divide='ignore'):
                    score_parent = (g_parent ** 2) / (h_parent + lam)
                    score_children = ((g_left ** 2) / (h_left + lam) +
                                      (g_right ** 2) / (h_right + lam))
                    gains = score_children - score_parent
                        
                # A partition contributes only when the split is legal there:
                # parent big enough AND both children big enough. Illegal or
                # empty partitions contribute 0, as in the original mask loop.
                valid = ((c_parent >= self.min_data_in_leaf) &
                         (c_left >= self.min_data_in_leaf) &
                         (c_right >= self.min_data_in_leaf))
                gains = np.where(valid, gains, 0.0)
                        
                total_gain = gains.sum(axis=0)      # summed over all partitions
                any_valid = valid.any(axis=0)       # was it legal anywhere?
                        
                for threshold in feature_thresholds[feature_idx]:
                    # Keep only candidates that actually split something. Without
                    # this test a meaningless "everything goes left" split would
                    # win whenever every partition was skipped.
                    if any_valid[threshold]:
                        cand_features.append(feature_idx)
                        cand_thresholds.append(int(threshold))
                        cand_gains.append(float(total_gain[threshold]))
            
            # If no valid split found, stop growing
            if not cand_gains:
                break
            
            raw_gains = np.array(cand_gains)

            # random_strength: jitter the scores so that near-ties are broken
            # randomly instead of always favouring the first feature. The noise
            # is scaled by the spread of THIS level's gains, because a raw gain
            # is measured in (target units)^2 -- an absolute perturbation would
            # mean something different for prices in dollars and in thousands.
            scored_gains = raw_gains
            if self.random_strength > 0:
                scale = np.std(raw_gains)
                if scale == 0.0:
                    scale = np.mean(np.abs(raw_gains))
                noise = rng.normal(size=raw_gains.shape)
                scored_gains = raw_gains + noise * self.random_strength * scale

            best = int(np.argmax(scored_gains))

            # Record this level's split (gain is the un-jittered value, so that
            # get_feature_importance('gain') reports real improvements)
            splits.append({
                'feature': cand_features[best],
                'threshold': cand_thresholds[best],
                'gain': float(raw_gains[best])
            })
            
            # Apply the split to EVERY partition at once: left child keeps the
            # parent's number doubled, right child adds one
            goes_right = (X_quantized[:, cand_features[best]] >
                          cand_thresholds[best]).astype(int)
            partition_id = partition_id * 2 + goes_right
            n_partitions *= 2
        
        # Calculate leaf values for final partitions
        leaf_values = []
        for p in range(n_partitions):
            value = self._calculate_leaf_value(gradients, partition_id == p, hessians)
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
    
    def _leaf_indices(self, tree, X_quantized):
        """
        Leaf index of every sample in one symmetric tree

        Same bit arithmetic as `_predict_tree`, exposed separately because
        ordered boosting needs the indices without the leaf values (it computes
        a DIFFERENT set of leaf values per supporting model on the same tree
        structure). Note there is no per-sample branching or pointer chasing:
        one vectorised comparison per level for the whole batch.
        """
        leaf_indices = np.zeros(X_quantized.shape[0], dtype=int)
        for level, split in enumerate(tree['splits']):
            goes_right = X_quantized[:, split['feature']] > split['threshold']
            remaining_depth = tree['depth'] - level - 1
            leaf_indices += goes_right * (2 ** remaining_depth)
        return leaf_indices

    def _as_2d(self, X, allow_object=False):
        """
        Coerce X to a 2-D array, accepting plain Python lists

        A 1-D input is read as n_samples rows of ONE feature, matching the
        common `X = np.array([1, 2, 3])` case. `allow_object` keeps string
        columns intact for categorical handling.
        """
        X = np.asarray(X, dtype=object) if allow_object else np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got {X.ndim} dimensions")
        return X

    def _cat_key(self, value):
        """
        Canonical dictionary key for one categorical value

        Categories are looked up by string, so the SAME level must produce the
        same string whether it arrives as a float, an int, or text. Numpy turns
        an int column into floats during fitting, so a category written 0 would
        be stored as '0.0' and then missed at predict time when a plain Python
        list supplies '0' -- silently falling back to the prior. Collapsing
        whole-number values to their integer form removes that trap.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isfinite(number) and number == int(number):
            return str(int(number))
        return str(number)

    def _ordered_target_statistics(self, keys, y, permutation, prior, prior_count):
        """
        Ordered target statistics for one categorical column (the paper's TS)

        This is CatBoost's answer to target encoding. Plain target encoding
        replaces a category with the mean target of every row in that category
        -- including the row being encoded, which leaks the answer. CatBoost
        instead fixes a random order and lets each row see only the rows BEFORE
        it, exactly the way the .md's "Red at positions 3, 7, 12, 18" example
        walks through it:

            TS_i = (sum of y_j over j before i in sigma with x_j == x_i
                    + prior_count * prior)
                   / (count of those j + prior_count)

        The first occurrence of a category therefore gets the prior alone --
        no leakage is possible, because y_i never enters TS_i.

        Parameters:
        -----------
        keys : np.ndarray of str, shape (n_samples,)
            The raw category labels of this column
        y : np.ndarray
            Target values
        permutation : np.ndarray
            The random order sigma (sigma[k] = index of the k-th row)
        prior : float
            Global target mean, used to smooth rare categories
        prior_count : float
            Smoothing weight `a` (how many "virtual prior rows" each category
            starts with)

        Returns:
        --------
        ts : np.ndarray, shape (n_samples,)
            Encoded values, aligned with the ORIGINAL row order
        """
        ts = np.zeros(len(keys))
        running_sum = {}
        running_count = {}

        for i in permutation:
            key = keys[i]
            s = running_sum.get(key, 0.0)
            c = running_count.get(key, 0)
            ts[i] = (s + prior_count * prior) / (c + prior_count)
            # Only now does row i's own target join the running totals, so it
            # can influence later rows but never itself
            running_sum[key] = s + y[i]
            running_count[key] = c + 1

        return ts

    def _encode_categoricals(self, X, y=None):
        """
        Replace categorical columns by numbers, leaving numeric columns alone

        During fit (y given) each categorical column becomes its ordered target
        statistic, and the FULL-data statistic per category is stored for later.
        At predict time (y is None) the stored statistic is used, because there
        is no ordering to respect once training is over; unseen categories fall
        back to the prior (the training target mean).

        Returns:
        --------
        X_numeric : np.ndarray of float, shape like X
        """
        cat_idx = set(self.cat_features) if self.cat_features else set()
        X_numeric = np.empty(X.shape, dtype=float)

        for j in range(X.shape[1]):
            column = X[:, j]
            if j not in cat_idx:
                X_numeric[:, j] = column.astype(float)
                continue

            keys = np.array([self._cat_key(v) for v in column])

            if y is not None:
                prior = float(np.mean(y))
                prior_count = 1.0
                # Average over several permutations, as the paper prescribes.
                # One permutation alone is very noisy: a row that happens to
                # land early sees almost no history and gets the prior, so the
                # same category can land in wildly different bins. Averaging
                # keeps the no-leakage property (no permutation ever lets a row
                # see its own target) while tightening the encoding.
                #
                # Measured on the setup below -- Demo 3's generator, two
                # feature columns, run at n = 120 and n = 400 rows. It is
                # written out in full so it can be re-run as-is; it does not
                # depend on the demo's global RNG state:
                #
                #   plans = ['basic', 'plus', 'pro', 'enterprise']
                #   value = {basic: 10, plus: 25, pro: 60, enterprise: 150}
                #   for s in range(200):          # seeds 0-199
                #       np.random.seed(s)         # draws in exactly this order
                #       plan  = np.random.choice(plans, n)
                #       usage = np.random.uniform(0, 10, n)
                #       y     = value[plan] + 3*usage + np.random.randn(n)*5
                #       rows shuffled by np.random.permutation(n)
                #       X: column 0 = plan strings, column 1 = usage
                #       train = first 3/4 of the rows, test = the rest
                #       CatBoost(n_estimators=120, learning_rate=0.1,
                #                depth=4, cat_features=[0],
                #                random_strength=0.0, random_seed=s)
                #                # every other argument left at its default
                #       model.ts_permutations = 1 or 4, set before fit()
                #
                # The one-hot baseline swaps column 0 for four 0/1 indicator
                # columns (one per plan, in the order listed above), drops
                # cat_features, and changes nothing else. Mean test RMSE over
                # the 200 seeds, at n=120 (90 train / 30 test): 8.67 with one
                # permutation vs 7.81 with four, four winning on 115 of the
                # 200 seeds, against 7.86 for one-hot. At n=400 (300 train /
                # 100 test -- Demo 3's own shape, but seeded per run, so not
                # the RMSE the demo prints) one and four are within noise of
                # each other, 6.18 vs 6.09, while one-hot leads at 5.64.
                # Averaging earns its keep when a category has few rows to
                # average over.
                ts_sum = np.zeros(len(keys))
                for _ in range(self.ts_permutations):
                    permutation = self._rng.permutation(len(keys))
                    ts_sum += self._ordered_target_statistics(
                        keys, y, permutation, prior, prior_count)
                X_numeric[:, j] = ts_sum / self.ts_permutations

                # Full-data statistic per category, for predict time
                mapping = {}
                for key in np.unique(keys):
                    in_cat = (keys == key)
                    mapping[key] = ((np.sum(y[in_cat]) + prior_count * prior) /
                                    (np.sum(in_cat) + prior_count))
                self._cat_encodings[j] = {'mapping': mapping, 'prior': prior}
            else:
                stored = self._cat_encodings[j]
                X_numeric[:, j] = [stored['mapping'].get(k, stored['prior'])
                                   for k in keys]

        return X_numeric

    def _transform(self, X):
        """Encode categoricals (predict-time statistics) then quantize."""
        X = self._as_2d(X, allow_object=bool(self.cat_features))
        if self.cat_features:
            X = self._encode_categoricals(X, y=None)
        return self._apply_quantization(X)

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the CatBoost model
        
        Algorithm:
        1. Encode categorical columns with ordered target statistics
        2. Quantize features into discrete bins
        3. Initialize predictions with base score
        4. For each boosting iteration:
           a. Calculate gradients (and Hessians)
           b. Build symmetric tree to minimize loss
           c. Update predictions with tree * learning_rate
        5. Optional: Early stopping on validation set
        
        ORDERED BOOSTING (boosting_type='Ordered', off by default):
        Plain boosting scores every sample with a model that has already fitted
        that same sample, which biases the gradients (prediction shift). In
        Ordered mode this implementation fixes one random permutation sigma and
        keeps log2(n) supporting models M_j, where M_j has only ever been
        updated using the first 2^(j-1) rows of sigma. The row at position p in
        sigma takes its gradient from the largest such model whose prefix ends
        at or before p, so no row is ever scored by a model that saw it.
        See the .md section "Simplification vs. canonical CatBoost" for what
        the real library does on top of this (several permutations, and its own
        criterion for choosing the tree structure).
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data. A 1-D input is read as one feature column. Columns
            listed in `cat_features` may hold strings.
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
        # Private RNG. When no random_seed is given we draw the seed from the
        # global numpy RNG, so np.random.seed(42) before fit() still makes the
        # run reproducible without the class touching global state itself.
        seed = self.random_seed
        if seed is None:
            seed = np.random.randint(0, 2 ** 31 - 1)
        self._rng = np.random.default_rng(seed)

        # Convert to numpy arrays (object dtype keeps string categories intact)
        X = self._as_2d(X, allow_object=bool(self.cat_features))
        y = np.asarray(y, dtype=float).ravel()
        
        n_samples, n_features = X.shape
        if len(y) != n_samples:
            raise ValueError(f"X has {n_samples} rows but y has {len(y)}")
        self.n_features = n_features

        # Encode categorical columns with ordered target statistics
        self._cat_encodings = {}
        if self.cat_features:
            bad = [j for j in self.cat_features if not 0 <= j < n_features]
            if bad:
                raise ValueError(f"cat_features indices out of range: {bad}")
            X = self._encode_categoricals(X, y)
        else:
            X = X.astype(float)
        
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
        
        # ---- Ordered boosting bookkeeping ----------------------------------
        ordered = (self.boosting_type == 'Ordered')
        if ordered:
            sigma = self._rng.permutation(n_samples)
            position = np.empty(n_samples, dtype=int)
            position[sigma] = np.arange(n_samples)   # rank of each row in sigma

            # Supporting model j is fitted on prefix_lengths[j] rows of sigma:
            # model 0 sees nothing (stays at base_score), model j>=1 sees the
            # first 2^(j-1) rows. Row at position p uses the largest model whose
            # prefix stops at or before p, i.e. j = floor(log2(p)) + 1.
            model_of = np.zeros(n_samples, dtype=int)
            nonzero = position >= 1
            model_of[nonzero] = np.floor(np.log2(position[nonzero])).astype(int) + 1
            n_models = int(model_of.max()) + 1
            prefix_masks = [np.zeros(n_samples, dtype=bool)]
            for j in range(1, n_models):
                prefix_masks.append(position < 2 ** (j - 1))

            # M_j's raw prediction for EVERY row (a model must be able to score
            # rows it was not fitted on -- that is the whole point)
            ordered_preds = np.full((n_models, n_samples), float(self.base_score))
            row_index = np.arange(n_samples)

        # ---- Validation bookkeeping ----------------------------------------
        # Quantize the validation set ONCE and keep a running prediction vector,
        # instead of replaying every tree from scratch on every iteration.
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            y_val = np.asarray(y_val, dtype=float).ravel()
            X_val_quantized = self._transform(X_val)
            val_raw = np.full(X_val_quantized.shape[0], float(self.base_score))

        # Early stopping variables
        best_score = float('inf')
        best_iteration = 0
        
        # Train trees
        for iteration in range(self.n_estimators):
            # Calculate gradients (and Hessians) of the model being returned
            gradients = self._compute_gradients(y, predictions)
            hessians = self._compute_hessians(y, predictions)

            if ordered:
                # Unbiased gradients: each row is scored by a supporting model
                # that never saw it. These choose the tree STRUCTURE, which is
                # where prediction shift does its damage.
                unbiased_raw = ordered_preds[model_of, row_index]
                struct_gradients = self._compute_gradients(y, unbiased_raw)
                struct_hessians = self._compute_hessians(y, unbiased_raw)
            else:
                struct_gradients, struct_hessians = gradients, hessians
            
            # Build symmetric tree
            tree = self._build_symmetric_tree(X_quantized, struct_gradients,
                                              struct_hessians)

            if ordered:
                # Reuse the tree STRUCTURE for every supporting model, but give
                # each one leaf values computed from its own prefix only
                leaf_idx = self._leaf_indices(tree, X_quantized)
                n_leaves = len(tree['leaf_values'])
                for j in range(1, n_models):
                    rows = prefix_masks[j]
                    g_sum = np.bincount(leaf_idx[rows], weights=struct_gradients[rows],
                                        minlength=n_leaves)
                    h_sum = np.bincount(leaf_idx[rows], weights=struct_hessians[rows],
                                        minlength=n_leaves)
                    values_j = -g_sum / (h_sum + self.l2_leaf_reg)
                    ordered_preds[j] += self.learning_rate * values_j[leaf_idx]

                # The returned model keeps that structure but re-fits its leaf
                # values on its OWN gradients, so `predictions` stays a proper
                # descent sequence (its training loss decreases monotonically).
                g_sum = np.bincount(leaf_idx, weights=gradients, minlength=n_leaves)
                h_sum = np.bincount(leaf_idx, weights=hessians, minlength=n_leaves)
                tree['leaf_values'] = -g_sum / (h_sum + self.l2_leaf_reg)

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
                # Running update, mirroring the training-side update above
                val_raw += self.learning_rate * self._predict_tree(tree, X_val_quantized)
                val_preds = self._sigmoid(val_raw) if self.objective == 'binary' else val_raw
                
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
                        # Trim the model AND the learning curves together, so
                        # train_scores/val_scores describe the trees we kept
                        self.trees = self.trees[:best_iteration + 1]
                        self.train_scores = self.train_scores[:best_iteration + 1]
                        self.val_scores = self.val_scores[:best_iteration + 1]
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
            Data to predict. A 1-D input is read as one feature column.
        num_iteration : int, optional
            Number of trees to use (None means all)
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values (probabilities when objective='binary')
        """
        if self.base_score is None:
            raise ValueError("This CatBoost model is not fitted yet. "
                             "Call fit(X, y) before predict().")
        
        # Encode categoricals with the stored statistics, then quantize
        X_quantized = self._transform(X)
        n_samples = X_quantized.shape[0]
        if X_quantized.shape[1] != self.n_features:
            raise ValueError(f"X has {X_quantized.shape[1]} features, "
                             f"but this model was fitted on {self.n_features}")
        
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
            NEGATIVE RMSE for regression (always <= 0, higher is better, so
            write `rmse = -model.score(X, y)` to read it as an error), or
            accuracy in [0, 1] for classification.
        """
        y = np.asarray(y, dtype=float).ravel()
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
            - 'gain' : Total gain (sum of the split scores recorded when the
                       level was chosen) contributed by that feature
            
        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Feature importance scores (normalized to sum to 1)
        """
        if self.n_features is None:
            raise ValueError("This CatBoost model is not fitted yet. "
                             "Call fit(X, y) before get_feature_importance().")
        if importance_type not in ('split', 'gain'):
            raise ValueError(f"Unknown importance_type: {importance_type!r} "
                             "(expected 'split' or 'gain')")

        importance = np.zeros(self.n_features)
        
        for tree in self.trees:
            for split in tree['splits']:
                if importance_type == 'split':
                    importance[split['feature']] += 1
                else:
                    # 'gain' stored by _build_symmetric_tree is the un-jittered
                    # sum of score improvements this split bought
                    importance[split['feature']] += max(split.get('gain', 0.0), 0.0)
        
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

# Shuffle so train and test cover the same x range.
# linspace produces SORTED x, so slicing straight away would put every test
# point beyond the training maximum - and a tree cannot extrapolate.
idx = np.random.permutation(200)
X, y = X[idx], y[idx]

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
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

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
    bar = '#' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")
"""

"""
USAGE EXAMPLE 5: Comparing Different Tree Depths

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(200, 5)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] + np.random.randn(200) * 0.5

X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Test different depths
depths = [3, 4, 5, 6, 8]

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

X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]

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

X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]

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

X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]

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


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _19_catboost.py
    # Requires numpy only. Everything below is seeded and reproducible.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Regression demo: predict y = x^2 + noise ---
    print("=" * 60)
    print("DEMO 1 - Regression: y = x^2 + noise")
    print("=" * 60)

    X_reg = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_reg = X_reg.ravel() ** 2 + np.random.randn(200) * 0.5
    # Shuffle so train and test cover the same x range: linspace is sorted,
    # and a tree can never extrapolate past the range it was trained on.
    idx_reg = np.random.permutation(200)
    X_reg, y_reg = X_reg[idx_reg], y_reg[idx_reg]
    X_tr, X_te = X_reg[:150], X_reg[150:]
    y_tr, y_te = y_reg[:150], y_reg[150:]

    reg_model = CatBoost(
        n_estimators=100,
        learning_rate=0.1,
        depth=4,
        l2_leaf_reg=3.0,
        random_seed=42
    )
    reg_model.fit(X_tr, y_tr)

    def r2(model, X, y):
        """R^2 = 1 - SS_res / SS_tot (score() returns negative RMSE instead)"""
        residual = np.sum((y - model.predict(X)) ** 2)
        total = np.sum((y - np.mean(y)) ** 2)
        return 1 - residual / total

    # score() returns NEGATIVE RMSE for regression, so negate it to read an error
    print(f"Train RMSE : {-reg_model.score(X_tr, y_tr):.4f}")
    print(f"Test  RMSE : {-reg_model.score(X_te, y_te):.4f}")
    print(f"Train R2   : {r2(reg_model, X_tr, y_tr):.4f}")
    print(f"Test  R2   : {r2(reg_model, X_te, y_te):.4f}")

    preds = reg_model.predict(X_te)
    print("\nSample predictions (x, true, predicted):")
    for i in range(5):
        print(f"  x={X_te[i, 0]:5.2f}  true={y_te[i]:5.2f}  pred={preds[i]:5.2f}")

    # --- Classification demo: two Gaussian blobs ---
    print("\n" + "=" * 60)
    print("DEMO 2 - Binary Classification: two Gaussian blobs")
    print("=" * 60)

    X0 = np.random.randn(100, 2) + np.array([-2, -2])
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X_cls = np.vstack([X0, X1])
    y_cls = np.array([0] * 100 + [1] * 100)
    idx_cls = np.random.permutation(200)
    X_cls, y_cls = X_cls[idx_cls], y_cls[idx_cls]
    X_tr2, X_te2 = X_cls[:150], X_cls[150:]
    y_tr2, y_te2 = y_cls[:150], y_cls[150:]

    cls_model = CatBoost(
        n_estimators=50,
        learning_rate=0.3,
        depth=3,
        objective='binary',
        random_seed=42
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

    # --- Categorical demo: the feature CatBoost is named after ---
    print("\n" + "=" * 60)
    print("DEMO 3 - Categorical Features: string column, no one-hot")
    print("=" * 60)

    plans = np.array(['basic', 'plus', 'pro', 'enterprise'])
    monthly_value = {'basic': 10.0, 'plus': 25.0, 'pro': 60.0, 'enterprise': 150.0}

    n_cat = 400
    plan_col = np.random.choice(plans, n_cat)
    usage_col = np.random.uniform(0, 10, n_cat)
    # Revenue depends on the PLAN (a string!) plus usage, plus noise
    revenue = (np.array([monthly_value[p] for p in plan_col])
               + 3.0 * usage_col
               + np.random.randn(n_cat) * 5.0)

    X_cat = np.empty((n_cat, 2), dtype=object)
    X_cat[:, 0] = plan_col          # categorical: raw strings
    X_cat[:, 1] = usage_col         # numeric
    idx_cat = np.random.permutation(n_cat)
    X_cat, revenue = X_cat[idx_cat], revenue[idx_cat]
    X_tr3, X_te3 = X_cat[:300], X_cat[300:]
    y_tr3, y_te3 = revenue[:300], revenue[300:]

    cat_model = CatBoost(
        n_estimators=120,
        learning_rate=0.1,
        depth=4,
        cat_features=[0],           # column 0 holds strings
        random_seed=42
    )
    cat_model.fit(X_tr3, y_tr3)

    print("Column 0 is raw text: " + ", ".join(str(v) for v in X_tr3[:4, 0]))
    print(f"Train RMSE : {-cat_model.score(X_tr3, y_tr3):.4f}")
    print(f"Test  RMSE : {-cat_model.score(X_te3, y_te3):.4f}")
    print(f"(std of test target = {np.std(y_te3):.4f}, so lower is better)")

    print("\nLearned target statistic per category (predict-time value):")
    mapping = cat_model._cat_encodings[0]['mapping']
    for plan in plans:
        print(f"  {plan:11s} -> {mapping[plan]:7.2f}   "
              f"(true plan value {monthly_value[plan]:6.2f} + avg usage effect)")

    preds3 = cat_model.predict(X_te3)
    print("\nSample predictions (plan, usage, true, predicted):")
    for i in range(5):
        print(f"  {str(X_te3[i, 0]):11s} usage={float(X_te3[i, 1]):5.2f}  "
              f"true={y_te3[i]:7.2f}  pred={preds3[i]:7.2f}")

    print("\n" + "=" * 60)
    print("Feature importance (Demo 3, normalized split counts):")
    for name, imp in zip(['plan (categorical)', 'usage (numeric)'],
                         cat_model.get_feature_importance('split')):
        print(f"  {name:20s}: {imp:.4f} {'#' * int(imp * 40)}")
    print("=" * 60)
