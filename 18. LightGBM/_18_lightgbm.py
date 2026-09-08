import numpy as np

class LightGBM:
    """
    LightGBM (Light Gradient Boosting Machine) Implementation from Scratch
    
    LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
    It is designed to be distributed and efficient with faster training speed, lower memory usage,
    better accuracy, and support for parallel and GPU learning.
    
    Key Idea: "Leaf-wise tree growth with histogram-based learning for speed and efficiency"
    
    Use Cases:
    - Regression: Sales forecasting, demand prediction, price estimation
    - Classification: Click prediction, fraud detection, customer churn
    - Ranking: Information retrieval, recommendation systems
    - Large-scale datasets: Where XGBoost becomes slow
    
    Key Innovations in LightGBM (and what this file implements):
        Leaf-wise Growth: IMPLEMENTED. Grows a tree by repeatedly splitting the single
            leaf with the highest gain anywhere in the tree, until num_leaves is reached.
            This is best-first growth, not level-wise, and it is what makes LightGBM's
            trees deep and asymmetric. See _build_tree_leaf_wise.
        Histogram-based Learning: IMPLEMENTED. Continuous features are binned once
            (_build_histogram), then every split search accumulates gradient/hessian
            histograms with np.bincount and scans them with a cumulative sum, which is
            the paper's O(#data) build + O(#bins) scan. See _find_best_split.
        GOSS: IMPLEMENTED (opt-in). Gradient-based One-Side Sampling keeps the
            top_rate fraction of large-|gradient| rows, randomly samples other_rate of
            the rest, and amplifies those by (1 - top_rate) / other_rate so the gain
            estimate stays unbiased. Enable with boosting_type='goss'. See _goss_sample.
        EFB: NOT IMPLEMENTED here. Exclusive Feature Bundling is explained in
            _18_lightgbm.md but is deliberately left out of this teaching
            implementation - see "Simplifications vs. canonical LightGBM" below.

    Core formulas this implementation embeds:
        ThresholdL1(G, a) = sign(G) * max(|G| - a, 0)      # L1 soft-thresholding
        Leaf weight:  w* = -ThresholdL1(G, lambda_l1) / (H + lambda_l2)
        Split score:  score(G, H) = ThresholdL1(G, lambda_l1)^2 / (H + lambda_l2)
        Split gain:   Gain = score(G_L, H_L) + score(G_R, H_R) - score(G_P, H_P)
                             - min_gain_to_split
        Note there is NO factor of 1/2 in the gain: that is XGBoost's convention.
        LightGBM's GetLeafSplitGain returns G^2/(H+lambda) directly, so the number this
        class reports is directly comparable to real LightGBM's split gains.

    Simplifications vs. canonical LightGBM (documented, not implemented):
        - EFB (Exclusive Feature Bundling): no conflict graph, no bundling of mutually
          exclusive sparse features into one bin range.
        - Missing values: X is cast with np.array(X, dtype=float) and NaN is not routed
          to a learned default side. Impute before calling fit.
        - Native categorical splits: every feature is treated as ordered/numeric, so a
          categorical column is split as "bin <= k", not as an optimal subset of levels.
        See _18_lightgbm.md, section "Simplifications vs. canonical LightGBM".
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31,
                 min_data_in_leaf=20, min_sum_hessian_in_leaf=1e-3, 
                 feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
                 lambda_l1=0.0, lambda_l2=0.0, min_gain_to_split=0.0,
                 max_bin=255, objective='regression',
                 boosting_type='gbdt', top_rate=0.2, other_rate=0.1,
                 random_state=None):
        """
        Initialize the LightGBM model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of boosting iterations (trees)
            - More iterations: Better training fit, longer training
            - Fewer iterations: Faster training, may underfit
            Typical values: 100-1000
            
        learning_rate : float, default=0.1
            Shrinkage rate (also called eta)
            - Lower values need more iterations but generalize better
            - Range: 0.01 to 0.3
            Typical: 0.1 is standard, 0.05 for large datasets
            
        max_depth : int, default=-1
            Maximum tree depth
            - -1 means no limit (controlled by num_leaves instead)
            - Positive values limit tree depth
            Typical: -1 or 3-8
            
        num_leaves : int, default=31
            Maximum number of leaves in one tree
            - LightGBM's primary way to control model complexity
            - Should be < 2^max_depth
            - Larger values: More complex model, risk overfitting
            Typical values: 31-127 (powers of 2 minus 1)
            
        min_data_in_leaf : int, default=20
            Minimum number of data points in one leaf
            - Larger values prevent overfitting
            - Too large: May underfit
            Typical values: 20-100 for large datasets, 5-20 for small
            
        min_sum_hessian_in_leaf : float, default=1e-3
            Minimum sum of hessian in one leaf
            - Similar to min_child_weight in XGBoost
            - Larger values: More conservative, less overfitting
            Typical values: 1e-3 to 10
            
        feature_fraction : float, default=1.0
            Fraction of features to use for each tree (column subsampling)
            - < 1.0 introduces randomness and speeds up training
            - Similar to colsample_bytree in XGBoost
            Typical values: 0.5-1.0
            
        bagging_fraction : float, default=1.0
            Fraction of data to use for each iteration (row subsampling)
            - < 1.0 provides regularization and speeds up training
            - Only used if bagging_freq > 0
            Typical values: 0.5-1.0
            
        bagging_freq : int, default=0
            Frequency for bagging (0 means disable bagging)
            - If k > 0, perform bagging every k iterations
            Typical values: 0 (disabled) or 1-5
            
        lambda_l1 : float, default=0.0
            L1 regularization term
            - Can lead to sparse solutions
            - Useful for feature selection
            Typical values: 0-10
            
        lambda_l2 : float, default=0.0
            L2 regularization term
            - Helps prevent overfitting
            - More common than L1
            Typical values: 0-10
            
        min_gain_to_split : float, default=0.0
            Minimum gain to perform split
            - Acts as regularization
            - Similar to gamma in XGBoost
            Typical values: 0-1
            
        max_bin : int, default=255
            Maximum number of bins for feature discretization
            - Larger values: More accurate but slower
            - Smaller values: Faster but less accurate
            Typical values: 63, 127, 255 (LightGBM default)
            
        objective : str, default='regression'
            Learning objective
            - 'regression': Regression with L2 loss
            - 'binary': Binary classification with log loss

        boosting_type : str, default='gbdt'
            Row-sampling strategy for each boosting iteration
            - 'gbdt': plain gradient boosting; uses bagging_fraction/bagging_freq
            - 'goss': Gradient-based One-Side Sampling (paper Algorithm 2)
            - GOSS and bagging are mutually exclusive: with 'goss' the bagging
              parameters are ignored, exactly as in the real library
            Typical: 'gbdt'; try 'goss' when training data is large

        top_rate : float, default=0.2
            GOSS only: fraction of rows with the LARGEST |gradient| that are always kept
            - Higher values: closer to training on all data, less speedup
            - Lower values: faster, noisier gain estimates
            Typical values: 0.1-0.3 (LightGBM's default is 0.2)

        other_rate : float, default=0.1
            GOSS only: fraction randomly sampled from the remaining small-gradient rows
            - Their gradients/hessians are amplified by (1 - top_rate) / other_rate
              so the information gain stays unbiased
            - Higher values: more of the "easy" rows retained, less speedup
            Typical values: 0.05-0.2 (LightGBM's default is 0.1)

        random_state : int or None, default=None
            Seed for the private random generator used by feature subsampling,
            bagging and GOSS
            - None: use numpy's global RNG (so np.random.seed(...) still controls it)
            - int: use a private np.random.RandomState, so results are reproducible
              regardless of other code drawing from the global RNG
            Typical: 42 for reproducible experiments
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.max_bin = max_bin
        self.objective = objective
        self.boosting_type = boosting_type
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.random_state = random_state
        
        self.trees = []
        self.base_score = None
        self.bin_thresholds = None
        self.best_iteration_ = None
        # Private RNG: np.random (global) when random_state is None, so existing
        # np.random.seed(...) scripts keep working; a private stream otherwise.
        self._rng = np.random if random_state is None else np.random.RandomState(random_state)
        
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
        
        LightGBM uses both gradients and hessians for optimization,
        just like XGBoost
        
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
        if self.objective == 'regression':
            # For squared error: L = 0.5 * (y - pred)^2
            # Gradient: dL/dpred = pred - y
            # Hessian: d²L/dpred² = 1
            gradient = y_pred - y_true
            hessian = np.ones_like(y_pred)
            
        elif self.objective == 'binary':
            # For log loss: L = -y*log(p) - (1-y)*log(1-p)
            # Gradient: dL/dpred = p - y
            # Hessian: d²L/dpred² = p * (1 - p)
            p = self._sigmoid(y_pred)
            gradient = p - y_true
            hessian = p * (1 - p)
            
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return gradient, hessian
    
    def _build_histogram(self, X):
        """
        Build histogram bins for features (histogram-based learning)
        
        This is a key innovation in LightGBM: instead of considering all possible
        split points, features are binned into discrete buckets, making training
        much faster while maintaining accuracy.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_binned : np.ndarray, shape (n_samples, n_features)
            Binned feature values (integers from 0 to max_bin-1)
        """
        n_samples, n_features = X.shape
        X_binned = np.zeros_like(X, dtype=int)
        self.bin_thresholds = []
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Create bins using percentiles
            unique_values = np.unique(feature_values)
            
            if len(unique_values) <= self.max_bin:
                # If few unique values, use them directly
                thresholds = unique_values[:-1]
            else:
                # Otherwise, create max_bin bins using quantiles
                percentiles = np.linspace(0, 100, self.max_bin + 1)[1:-1]
                thresholds = np.percentile(feature_values, percentiles)
                thresholds = np.unique(thresholds)
            
            self.bin_thresholds.append(thresholds)
            
            # Assign bin indices.
            # right=True gives "bin k contains x with threshold[k-1] < x <= threshold[k]",
            # i.e. the split semantics "x <= threshold" that the trees use later.
            # With the default right=False a value equal to a threshold would fall into
            # the NEXT bin, which merges the two largest values of every low-cardinality
            # feature (a 0/1 flag would collapse into a single constant bin).
            X_binned[:, feature_idx] = np.digitize(feature_values, thresholds, right=True)
        
        return X_binned
    
    def _apply_binning(self, X):
        """Apply pre-computed binning to new data (same right=True semantics as fit)"""
        n_samples, n_features = X.shape
        X_binned = np.zeros_like(X, dtype=int)
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = self.bin_thresholds[feature_idx]
            X_binned[:, feature_idx] = np.digitize(feature_values, thresholds, right=True)
        
        return X_binned
    
    def _threshold_l1(self, gradient_sum):
        """
        L1 soft-thresholding of a gradient sum (LightGBM's ThresholdL1)

        ThresholdL1(G, a) = sign(G) * max(|G| - a, 0)
                          = G - a  if G >  a
                            G + a  if G < -a
                            0      if |G| <= a

        Shrinking G toward zero by lambda_l1 is what makes L1 regularization
        produce leaf weights that are exactly zero when the gradient evidence
        in a leaf is weaker than lambda_l1. Written with np.sign/np.maximum so
        the same helper works on a single float and on a whole histogram of
        candidate splits at once.

        Parameters:
        -----------
        gradient_sum : float or np.ndarray
            Sum of gradients (G) for a leaf or for many candidate children

        Returns:
        --------
        shrunk : float or np.ndarray
            G shrunk toward zero by lambda_l1
        """
        return np.sign(gradient_sum) * np.maximum(np.abs(gradient_sum) - self.lambda_l1, 0.0)
    
    def _calculate_leaf_weight(self, gradient_sum, hessian_sum):
        """
        Calculate optimal leaf weight with L1 and L2 regularization
        
        LightGBM formula: w* = -ThresholdL1(G, lambda_l1) / (H + lambda_l2)

        With lambda_l1 = 0 this is the familiar w* = -G / (H + lambda_l2).
        The tiny 1e-10 only guards against a division by zero when both the
        hessian sum and lambda_l2 are zero.
        
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
        return -self._threshold_l1(gradient_sum) / (hessian_sum + self.lambda_l2 + 1e-10)
    
    def _calculate_gain(self, gradient_left, hessian_left, gradient_right, hessian_right):
        """
        Calculate split gain using LightGBM's formula
        
        score(G, H) = ThresholdL1(G, lambda_l1)^2 / (H + lambda_l2)

        Gain = score(G_L, H_L) + score(G_R, H_R) - score(G_L+G_R, H_L+H_R)
               - min_gain_to_split

        Note: unlike XGBoost there is NO factor of 1/2 here. LightGBM's
        GetLeafSplitGain returns G^2/(H+lambda) directly, so a gain printed by
        this class is on the same scale as one printed by the real library.

        Note: the value returned is NET of min_gain_to_split, and it is the value
        stored in node['gain'] and summed by get_feature_importance('gain').
        Raising min_gain_to_split lowers every candidate by the SAME amount, so
        it does not change which candidate wins at a node - a uniform shift
        preserves the argmax, and the frontier ordering with it. It does change
        how many splits happen, because _build_tree_leaf_wise only offers a leaf
        whose best gain is still > 0. Measured on
            np.random.seed(7)
            X = np.random.randn(300, 4)
            y = 3*X[:,0] - 2*X[:,1] + X[:,2] + np.random.randn(300)*0.5
        with lambda_l2=1, num_leaves=31 and min_data_in_leaf=5: the root split
        stays feature 0 / bin 146 at min_gain_to_split = 0, 2.5, 50 and 500, its
        stored gain falls by exactly the parameter (1459.8623 -> 1457.3623 ->
        1409.8623 -> 959.8623), while the tree shrinks from 30 splits to 27,
        then 8, then 2.

        All four arguments may be numpy arrays, in which case an array of gains
        for many candidate splits is returned at once - that is how the
        histogram scan in _find_best_split evaluates every threshold in one shot.
        
        Parameters:
        -----------
        gradient_left : float or np.ndarray
            Sum of gradients in left child
        hessian_left : float or np.ndarray
            Sum of hessians in left child
        gradient_right : float or np.ndarray
            Sum of gradients in right child
        hessian_right : float or np.ndarray
            Sum of hessians in right child
            
        Returns:
        --------
        gain : float or np.ndarray
            Split gain, net of min_gain_to_split
        """
        def calculate_score(G, H):
            # L1 soft-threshold first, then the usual G^2 / (H + lambda_l2)
            return (self._threshold_l1(G) ** 2) / (H + self.lambda_l2 + 1e-10)
        
        gain_left = calculate_score(gradient_left, hessian_left)
        gain_right = calculate_score(gradient_right, hessian_right)
        gain_parent = calculate_score(gradient_left + gradient_right, 
                                      hessian_left + hessian_right)
        
        gain = (gain_left + gain_right - gain_parent) - self.min_gain_to_split
        
        return gain
    
    def _select_features(self, n_features):
        """
        Choose the feature subset for ONE tree (LightGBM's feature_fraction)
        
        Drawn once per boosting iteration, not once per node - that is what
        feature_fraction means (LightGBM spells the per-node variant
        feature_fraction_bynode). When feature_fraction >= 1.0 no random draw
        happens at all, so a run without column subsampling is deterministic.

        Parameters:
        -----------
        n_features : int
            Total number of features

        Returns:
        --------
        feature_indices : np.ndarray
            Indices of the features this tree may split on
        """
        if self.feature_fraction >= 1.0:
            return np.arange(n_features)
        n_features_use = max(1, int(self.feature_fraction * n_features))
        return self._rng.choice(n_features, n_features_use, replace=False)

    def _find_best_split(self, X_binned, gradient, hessian, indices, feature_indices, n_bins):
        """
        Histogram-based split finding for ONE leaf - the paper's core speed trick

        For each candidate feature:
          1. Accumulate the gradient / hessian / count histogram over the leaf's
             rows with np.bincount                      -> O(#data_in_leaf)
          2. Turn it into every possible "bin <= k" split with ONE cumulative sum:
                 G_L[k] = sum of gradients in bins 0..k
                 G_R[k] = G_total - G_L[k]              -> O(#bins)
             The right child costs nothing extra: it is the parent minus the left.
          3. Score every threshold at once with _calculate_gain (which accepts
             arrays), then take the argmax.

        That is O(#data + #bins) per feature. The naive alternative - rebuilding a
        boolean mask and re-summing the gradients for every candidate threshold -
        is O(#data x #bins), which is exactly the cost the histogram removes.

        Parameters:
        -----------
        X_binned : np.ndarray, shape (n_samples, n_features)
            Binned training data for the whole tree
        gradient : np.ndarray, shape (n_samples,)
            First-order gradients for the whole tree
        hessian : np.ndarray, shape (n_samples,)
            Second-order gradients for the whole tree
        indices : np.ndarray
            Row indices belonging to this leaf
        feature_indices : np.ndarray
            Features this tree is allowed to split on
        n_bins : int
            Histogram width (number of distinct bin ids in the data)

        Returns:
        --------
        best_gain : float
            Gain of the best valid split (0.0 if none)
        best_feature : int or None
            Feature index of the best split
        best_bin : int or None
            Threshold as a bin index; the left child is "bin <= best_bin"
        """
        g = gradient[indices]
        h = hessian[indices]
        G_total = np.sum(g)
        H_total = np.sum(h)
        n_node = len(indices)
        min_leaf = max(1, self.min_data_in_leaf)

        best_gain = 0.0
        best_feature = None
        best_bin = None

        for feature_idx in feature_indices:
            bins = X_binned[indices, feature_idx]

            # 1. One pass over the leaf's rows builds the histogram
            hist_gradient = np.bincount(bins, weights=g, minlength=n_bins)
            hist_hessian = np.bincount(bins, weights=h, minlength=n_bins)
            hist_count = np.bincount(bins, minlength=n_bins)

            # 2. Prefix sums give the left child of every "bin <= k" split;
            #    the right child is the complement
            G_left = np.cumsum(hist_gradient)
            H_left = np.cumsum(hist_hessian)
            C_left = np.cumsum(hist_count)
            G_right = G_total - G_left
            H_right = H_total - H_left
            C_right = n_node - C_left

            # 3. Apply the leaf constraints, then score all thresholds at once
            valid = ((C_left >= min_leaf) & (C_right >= min_leaf) &
                     (H_left >= self.min_sum_hessian_in_leaf) &
                     (H_right >= self.min_sum_hessian_in_leaf))
            if not np.any(valid):
                continue

            gains = self._calculate_gain(G_left, H_left, G_right, H_right)
            gains = np.where(valid, gains, -np.inf)

            k = int(np.argmax(gains))
            if gains[k] > best_gain:
                best_gain = float(gains[k])
                best_feature = int(feature_idx)
                best_bin = k

        return best_gain, best_feature, best_bin

    def _build_tree_leaf_wise(self, X_binned, gradient, hessian, depth=0, feature_indices=None):
        """
        Build a tree with the leaf-wise (best-first) strategy - LightGBM Algorithm 1

        This is LightGBM's defining innovation. XGBoost grows LEVEL-WISE: it splits
        every leaf of the current level before moving down, so the tree stays
        balanced. LightGBM keeps a FRONTIER of candidate leaves, each carrying the
        best split it could make and that split's gain, and repeatedly splits the
        single leaf whose gain is the highest anywhere in the tree. Growth stops at
        num_leaves leaves, so the leaf budget - not the depth - controls complexity,
        and the resulting trees are deep and asymmetric.

        The loop:
            1. The root starts as one leaf; score its best split.
            2. While the tree has fewer than num_leaves leaves and some candidate
               still has positive gain:
                 a. Pop the frontier entry with the MAXIMUM gain     <- "best-first"
                 b. Turn that leaf into a split node, in place
                 c. Score the best split of each new child, push them on the frontier
            3. Whatever is still on the frontier simply stays a leaf.

        Guards applied before a leaf is ever offered for splitting:
            max_depth (when > 0), min_data_in_leaf, min_sum_hessian_in_leaf; and
            min_gain_to_split, which is already subtracted inside _calculate_gain,
            so requiring gain > 0 enforces it.
        
        Parameters:
        -----------
        X_binned : np.ndarray, shape (n_samples, n_features)
            Binned training data
        gradient : np.ndarray, shape (n_samples,)
            First-order gradients
        hessian : np.ndarray, shape (n_samples,)
            Second-order gradients
        depth : int
            Depth of the root of this tree (0 for a normal tree)
        feature_indices : np.ndarray, optional
            Features this tree may split on. None means draw them with
            _select_features (i.e. honour feature_fraction).
            
        Returns:
        --------
        tree : dict
            Tree structure. A leaf is {'type','weight','count'}; an internal node is
            {'type','feature','threshold','gain','left','right','count'}.
        """
        n_samples, n_features = X_binned.shape
        
        if feature_indices is None:
            feature_indices = self._select_features(n_features)
        
        # Histogram width: bin ids are integers 0 .. max
        n_bins = int(X_binned.max()) + 1 if n_samples > 0 else 1

        def make_leaf(indices):
            """A leaf node holding the optimal weight for the rows it covers"""
            gradient_sum = np.sum(gradient[indices])
            hessian_sum = np.sum(hessian[indices])
            return {
                'type': 'leaf',
                'weight': self._calculate_leaf_weight(gradient_sum, hessian_sum),
                'count': len(indices)
            }
        
        def offer(node, indices, node_depth):
            """Score this leaf's best split and put it on the frontier if splittable"""
            if len(indices) < self.min_data_in_leaf:
                return
            if np.sum(hessian[indices]) < self.min_sum_hessian_in_leaf:
                return
            if self.max_depth > 0 and node_depth >= self.max_depth:
                return
        
            gain, feature, bin_threshold = self._find_best_split(
                X_binned, gradient, hessian, indices, feature_indices, n_bins
            )
            if feature is not None and gain > 0:
                frontier.append({
                    'node': node, 'indices': indices, 'depth': node_depth,
                    'gain': gain, 'feature': feature, 'bin': bin_threshold
                })
        
        root_indices = np.arange(n_samples)
        root = make_leaf(root_indices)
        frontier = []
        offer(root, root_indices, depth)
            
        n_leaves = 1
        while frontier and n_leaves < self.num_leaves:
            # BEST-FIRST: split the highest-gain leaf anywhere in the tree
            best_i = max(range(len(frontier)), key=lambda i: frontier[i]['gain'])
            candidate = frontier.pop(best_i)
                
            indices = candidate['indices']
            left_mask = X_binned[indices, candidate['feature']] <= candidate['bin']
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
                
            left_node = make_leaf(left_indices)
            right_node = make_leaf(right_indices)
                
            # Convert the chosen leaf into a split node IN PLACE, so the parent -
            # which already holds a reference to this dict - sees the new subtree
            node = candidate['node']
            node.clear()
            node.update({
                'type': 'split',
                'feature': candidate['feature'],
                'threshold': candidate['bin'],
                'gain': candidate['gain'],
                'left': left_node,
                'right': right_node,
                'count': len(indices)
            })
            n_leaves += 1  # one leaf was replaced by two
                
            offer(left_node, left_indices, candidate['depth'] + 1)
            offer(right_node, right_indices, candidate['depth'] + 1)
                
        return root
    
    def _predict_tree(self, tree, X_binned):
        """
        Make predictions using a single tree
        
        Parameters:
        -----------
        tree : dict
            Tree structure
        X_binned : np.ndarray
            Binned data
            
        Returns:
        --------
        predictions : np.ndarray
            Tree predictions
        """
        if tree['type'] == 'leaf':
            return np.full(len(X_binned), tree['weight'])
        
        feature_bins = X_binned[:, tree['feature']]
        left_mask = feature_bins <= tree['threshold']
        
        predictions = np.zeros(len(X_binned))
        
        if np.sum(left_mask) > 0:
            predictions[left_mask] = self._predict_tree(tree['left'], X_binned[left_mask])
        if np.sum(~left_mask) > 0:
            predictions[~left_mask] = self._predict_tree(tree['right'], X_binned[~left_mask])
        
        return predictions
    
    def _goss_sample(self, gradient, hessian):
        """
        Gradient-based One-Side Sampling (GOSS) - LightGBM paper, Algorithm 2

        Idea: rows with a LARGE |gradient| are the ones the model still gets wrong,
        so they carry most of the information about where the next split should go.
        Rows with a small gradient are already well fitted. Simply dropping them
        would bias the gain toward the hard rows, so the few that survive are
        amplified to stand in for all the ones removed.

        Steps (a = top_rate, b = other_rate):
            1. Sort rows by |gradient|, descending
            2. Keep the top a fraction  -> set A ("large gradient")
            3. Randomly sample b fraction of the REST -> set B ("small gradient")
            4. Multiply the gradients and hessians of B by fact = (1 - a) / b, which
               restores the total weight the discarded rows would have contributed
            5. Build the tree on A + B only

        Example: 100k rows with a=0.2, b=0.1 -> train on 20k + 8k = 28k rows,
        with the 8k sampled rows counted 8x, approximating the full 100k.

        Parameters:
        -----------
        gradient : np.ndarray, shape (n_samples,)
            First-order gradients
        hessian : np.ndarray, shape (n_samples,)
            Second-order gradients (unused for the ranking, kept for symmetry)

        Returns:
        --------
        indices : np.ndarray
            Row indices of the sampled subset (set A first, then set B)
        weights : np.ndarray
            1.0 for the A rows, (1 - top_rate) / other_rate for the B rows
        """
        n_samples = len(gradient)
        top_n = int(self.top_rate * n_samples)
        other_n = int(self.other_rate * n_samples)

        # Degenerate settings (tiny data, or the two rates already cover
        # everything): fall back to the full, unweighted dataset
        if other_n < 1 or top_n + other_n >= n_samples:
            return np.arange(n_samples), np.ones(n_samples)

        # 1-2. The largest-|gradient| rows are always kept
        sorted_idx = np.argsort(-np.abs(gradient))
        top_idx = sorted_idx[:top_n]

        # 3. Uniform sample of the remaining, small-gradient rows
        rest_idx = sorted_idx[top_n:]
        other_idx = self._rng.choice(rest_idx, other_n, replace=False)

        indices = np.concatenate([top_idx, other_idx])

        # 4. Amplification keeps the information gain unbiased
        fact = (1.0 - self.top_rate) / self.other_rate
        weights = np.ones(len(indices))
        weights[top_n:] = fact

        return indices, weights

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the LightGBM model
        
        Algorithm:
        1. Build histogram bins for all features
        2. Initialize predictions with base score
        3. For each boosting iteration:
           a. Calculate gradients and hessians
           b. Apply GOSS (boosting_type='goss') or bagging, if enabled
           c. Draw this tree's feature subset (feature_fraction)
           d. Build tree leaf-wise (best-first), bounded by num_leaves
           e. Update predictions: F(x) <- F(x) + learning_rate * tree(x)
        4. Optional: Early stopping on validation set
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Training data
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
        self : LightGBM
            Fitted model
        """
        # Convert to numpy arrays (plain Python lists are accepted)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()

        # A single feature may be passed as a flat 1-D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Reset the private RNG so re-fitting with a random_state is reproducible
        self._rng = (np.random if self.random_state is None
                     else np.random.RandomState(self.random_state))
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Build histogram bins (LightGBM's key feature)
        X_binned = self._build_histogram(X)
        
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
        self.best_iteration_ = None
        
        # Early stopping variables
        best_score = float('inf')
        best_iteration = 0
        bag_indices = None
        
        # Train trees
        for iteration in range(self.n_estimators):
            # Calculate gradients and hessians
            gradient, hessian = self._compute_gradient_hessian(y, predictions)
            
            # Row sampling: GOSS or bagging. They are mutually exclusive here,
            # exactly as in the real library.
            if self.boosting_type == 'goss':
                goss_indices, goss_weights = self._goss_sample(gradient, hessian)
                X_sample = X_binned[goss_indices]
                gradient_sample = gradient[goss_indices] * goss_weights
                hessian_sample = hessian[goss_indices] * goss_weights
            elif self.bagging_freq > 0 and self.bagging_fraction < 1.0:
                # Re-draw the subsample every bagging_freq iterations and REUSE it
                # for the iterations in between. (Handing back the full dataset in
                # between would mean only one tree in every bagging_freq was
                # actually regularized.)
                if bag_indices is None or iteration % self.bagging_freq == 0:
                    sample_size = int(n_samples * self.bagging_fraction)
                    bag_indices = self._rng.choice(n_samples, sample_size, replace=False)
                X_sample = X_binned[bag_indices]
                gradient_sample = gradient[bag_indices]
                hessian_sample = hessian[bag_indices]
            else:
                X_sample = X_binned
                gradient_sample = gradient
                hessian_sample = hessian
            
            # Column sampling: one feature subset per tree (feature_fraction)
            feature_indices = self._select_features(n_features)

            # Build tree using leaf-wise (best-first) strategy
            tree = self._build_tree_leaf_wise(X_sample, gradient_sample, hessian_sample,
                                              feature_indices=feature_indices)
            self.trees.append(tree)
            
            # Update predictions
            tree_predictions = self._predict_tree(tree, X_binned)
            predictions += self.learning_rate * tree_predictions
            
            # Calculate training score
            if self.objective == 'binary':
                train_preds = self._sigmoid(predictions)
                train_score = -np.mean(y * np.log(train_preds + 1e-10) + 
                                      (1 - y) * np.log(1 - train_preds + 1e-10))
            else:
                train_score = np.mean((y - predictions) ** 2)
            
            self.train_scores.append(train_score)
            
            # Evaluate on validation set
            if eval_set is not None:
                X_val, y_val = eval_set[0]
                val_preds = self.predict(X_val, num_iteration=iteration+1)
                
                if self.objective == 'binary':
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
                        # Truncate the trees AND both score curves together, so a
                        # learning curve plotted against the surviving trees lines up
                        self.trees = self.trees[:best_iteration + 1]
                        self.train_scores = self.train_scores[:best_iteration + 1]
                        self.val_scores = self.val_scores[:best_iteration + 1]
                        self.best_iteration_ = best_iteration
                        break
                
                # Verbose output (the metric name must match the objective:
                # squared loss is reported as RMSE, log loss as logloss)
                if verbose and (isinstance(verbose, bool) or iteration % verbose == 0):
                    if self.objective == 'binary':
                        print(f"[{iteration}] train-logloss: {train_score:.6f}, "
                              f"val-logloss: {val_score:.6f}")
                    else:
                        print(f"[{iteration}] train-rmse: {np.sqrt(train_score):.6f}, "
                              f"val-rmse: {np.sqrt(val_score):.6f}")
            elif verbose and (isinstance(verbose, bool) or iteration % verbose == 0):
                if self.objective == 'binary':
                    print(f"[{iteration}] train-logloss: {train_score:.6f}")
                else:
                    print(f"[{iteration}] train-rmse: {np.sqrt(train_score):.6f}")

        # Without early stopping the last tree is the best one we have
        if self.best_iteration_ is None:
            self.best_iteration_ = len(self.trees) - 1
        
        return self
    
    def predict(self, X, num_iteration=None):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : np.ndarray or list, shape (n_samples, n_features)
            Data to predict
        num_iteration : int, optional
            Number of trees to use (None means all)
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted values
        """
        if self.base_score is None or self.bin_thresholds is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) before predict().")

        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        
        # Apply binning
        X_binned = self._apply_binning(X)
        
        # Start with base score
        predictions = np.full(n_samples, self.base_score)
        
        # Determine number of trees to use
        n_trees = len(self.trees) if num_iteration is None else min(num_iteration, len(self.trees))
        
        # Add contribution from each tree
        for i in range(n_trees):
            tree_predictions = self._predict_tree(self.trees[i], X_binned)
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
            R^2 for regression, accuracy for classification
        """
        y = np.array(y).ravel()
        predictions = self.predict(X)
        
        if self.objective == 'binary':
            # Classification: accuracy
            predicted_classes = (predictions >= 0.5).astype(int)
            return np.mean(predicted_classes == y)
        else:
            # Regression: R^2 score
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - predictions) ** 2)
            if ss_total == 0:
                # y is constant, so R^2 is undefined (the variance it explains is 0).
                # Report a perfect 1.0 only when the predictions are exact.
                return 1.0 if ss_residual == 0 else 0.0
            r2 = 1 - (ss_residual / ss_total)
            return r2
    
    def get_feature_importance(self, importance_type='split'):
        """
        Calculate feature importance
        
        Parameters:
        -----------
        importance_type : str, default='split'
            Type of importance:
            - 'split': Number of times feature is used for splitting
            - 'gain': Total gain from splits using the feature. The stored gains
              are NET of min_gain_to_split (see _calculate_gain), so a non-zero
              min_gain_to_split both lowers each stored gain and prunes the
              low-gain splits away, changing these totals by more than a
              constant shift.
            
        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Feature importance scores (normalized to sum to 1)
        """
        if not self.trees:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        if importance_type == 'split':
            importance = np.zeros(self.n_features)
            
            def count_splits(tree):
                if tree['type'] == 'leaf':
                    return
                importance[tree['feature']] += 1
                count_splits(tree['left'])
                count_splits(tree['right'])
            
            for tree in self.trees:
                count_splits(tree)
                
        elif importance_type == 'gain':
            importance = np.zeros(self.n_features)
            
            def accumulate_gain(tree):
                if tree['type'] == 'leaf':
                    return
                importance[tree['feature']] += tree['gain']
                accumulate_gain(tree['left'])
                accumulate_gain(tree['right'])
            
            for tree in self.trees:
                accumulate_gain(tree)
        else:
            raise ValueError(f"Unknown importance_type: {importance_type}")
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance


"""
USAGE EXAMPLE 1: Simple Regression with LightGBM

import numpy as np

# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.randn(200) * 0.5

# Shuffle BEFORE splitting: linspace produces sorted x, so slicing directly would
# put every test point beyond the training range. Trees cannot extrapolate - they
# would all predict the training maximum and the test R2 would go negative.
idx = np.random.permutation(200)
X, y = X[idx], y[idx]

# Split train/test
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Create and train LightGBM model
model = LightGBM(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=5
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
USAGE EXAMPLE 2: Binary Classification with LightGBM

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

# Train LightGBM classifier
model = LightGBM(
    n_estimators=50,
    learning_rate=0.1,
    num_leaves=31,
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
USAGE EXAMPLE 3: LightGBM with Early Stopping

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(500, 10)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] ** 2 + np.random.randn(500) * 0.5

# Split train/validation/test
X_train, X_val, X_test = X[:300], X[300:400], X[400:]
y_train, y_val, y_test = y[:300], y[300:400], y[400:]

# Train with early stopping
model = LightGBM(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=10
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
model = LightGBM(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31
)
model.fit(X, y)

# Get feature importance
importance_split = model.get_feature_importance('split')
importance_gain = model.get_feature_importance('gain')

print("\nFeature Importance (by split count):")
print("="*50)
for i, imp in enumerate(importance_split):
    bar = '#' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")

print("\nFeature Importance (by gain):")
print("="*50)
for i, imp in enumerate(importance_gain):
    bar = '#' * int(imp * 50)
    print(f"Feature {i:2d}: {imp:.4f} {bar}")
"""

"""
USAGE EXAMPLE 5: Comparing with Different num_leaves Values

import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(200, 5)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] + np.random.randn(200) * 0.5

X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Test different num_leaves values
num_leaves_values = [7, 15, 31, 63, 127]

print("Effect of num_leaves (Tree Complexity):")
print("="*80)
print(f"{'num_leaves':>12} {'Train R2':>15} {'Test R2':>15} {'Overfit':>15}")
print("-"*80)

for num_leaves in num_leaves_values:
    model = LightGBM(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=num_leaves,
        min_data_in_leaf=5
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfit = train_score - test_score
    
    print(f"{num_leaves:>12} {train_score:>15.4f} {test_score:>15.4f} {overfit:>15.4f}")

# Observation: Larger num_leaves can lead to overfitting - Train R2 rises while
# Test R2 falls, so the Overfit gap widens (0.0800 -> 0.1064 -> 0.1276).
# The last three rows are identical because with only 150 training rows and
# min_data_in_leaf=5 the trees run out of splittable leaves before they reach
# 31 leaves; num_leaves stops binding once another guard binds first.
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
learning_rates = [0.01, 0.05, 0.1, 0.3]

print("\nEffect of Learning Rate:")
print("="*80)
print(f"{'Learning Rate':>15} {'Train R2':>15} {'Test R2':>15} {'Trees':>10}")
print("-"*80)

for lr in learning_rates:
    model = LightGBM(
        n_estimators=200,
        learning_rate=lr,
        num_leaves=31
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"{lr:>15.2f} {train_score:>15.4f} {test_score:>15.4f} {len(model.trees):>10}")

# Observation: Lower learning rate often needs more trees but generalizes better.
# Read the two columns together. At lr=0.01 the 200 trees are not enough to fit
# even the training set (Train R2 0.5051) - that is undertraining, not
# regularization. At lr=0.3 the model memorizes (Train R2 0.9975) and Test R2
# falls back to 0.3808. The best test score here is the middle of the range,
# lr=0.1 with Test R2 0.4303: low enough to take careful steps, high enough that
# 200 trees suffice. Halving the learning rate roughly doubles the trees needed.
"""

"""
USAGE EXAMPLE 7: LightGBM with Feature Subsampling

import numpy as np

# Wide dataset (many features)
np.random.seed(42)
X = np.random.randn(200, 20)
# Only first 5 features are informative
y = (2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] - 
     0.5 * X[:, 3] + X[:, 4] + np.random.randn(200) * 0.5)

X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Test different feature_fraction values
feature_fractions = [0.3, 0.5, 0.7, 1.0]

print("\nEffect of Feature Subsampling:")
print("="*80)
print(f"{'Feature Fraction':>18} {'Train R2':>15} {'Test R2':>15} {'Overfit':>15}")
print("-"*80)

for frac in feature_fractions:
    model = LightGBM(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=frac
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfit = train_score - test_score
    
    print(f"{frac:>18.1f} {train_score:>15.4f} {test_score:>15.4f} {overfit:>15.4f}")

# Observation: feature_fraction draws a fresh feature subset for every tree, so it
# adds diversity the way Random Forest's column sampling does. It is not a free win:
# on this run the Overfit gap actually SHRINKS as the fraction grows
# (0.2883 -> 0.2506 -> 0.2536 -> 0.2310), because only 5 of the 20 features are
# informative and 150 training rows are few - a tree restricted to 6 random columns
# often sees none of the useful ones. Subsampling pays off when features are many
# AND correlated; with few informative features it mostly adds noise.
"""

"""
USAGE EXAMPLE 8: Real-World - Sales Prediction

import numpy as np

# Simulated sales data
# [advertising_spend, promotion_days, season, competitor_price, 
#  store_location_score, day_of_week, weather_score]

np.random.seed(42)
n_samples = 500

advertising = np.random.uniform(1000, 10000, n_samples)
promotion = np.random.randint(0, 31, n_samples)
season = np.random.randint(1, 5, n_samples)  # 1=Spring, 2=Summer, 3=Fall, 4=Winter
competitor_price = np.random.uniform(50, 150, n_samples)
location = np.random.uniform(1, 10, n_samples)
day_of_week = np.random.randint(1, 8, n_samples)
weather = np.random.uniform(1, 10, n_samples)

X = np.column_stack([advertising, promotion, season, competitor_price,
                     location, day_of_week, weather])

# Sales formula with interactions
sales = (
    0.5 * advertising +
    200 * promotion +
    5000 * season +
    -100 * competitor_price +
    1000 * location +
    500 * day_of_week +
    300 * weather +
    0.01 * advertising * location +  # Interaction
    np.random.randn(n_samples) * 2000
)

# Normalize to thousands
sales = sales / 1000

# Split data
X_train, X_val, X_test = X[:300], X[300:400], X[400:]
y_train, y_val, y_test = sales[:300], sales[300:400], sales[400:]

# Train LightGBM model
model = LightGBM(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=10,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=50
)

# Evaluate
test_r2 = model.score(X_test, y_test)
predictions = model.predict(X_test)

mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

print(f"\nSales Prediction Model:")
print("="*60)
print(f"Test R2: {test_r2:.4f}")
print(f"Mean Absolute Error: ${mae:.2f}k")
print(f"Root Mean Squared Error: ${rmse:.2f}k")

# Feature importance
feature_names = ['Advertising', 'Promotion Days', 'Season', 'Competitor Price',
                'Location Score', 'Day of Week', 'Weather']
importance = model.get_feature_importance('gain')

print("\nFeature Importance:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.4f}")

# Predict new scenarios
new_scenarios = np.array([
    [8000, 15, 4, 75, 9, 6, 8],   # High ad spend, promotion, good location
    [2000, 0, 1, 120, 3, 2, 4]     # Low ad spend, no promotion, poor location
])

predicted_sales = model.predict(new_scenarios)

print("\nNew Scenario Sales Predictions:")
for i, pred in enumerate(predicted_sales):
    print(f"Scenario {i+1}: ${pred:.2f}k")
"""

"""
USAGE EXAMPLE 9: Click-Through Rate Prediction

import numpy as np

# User features for ad click prediction
# [age, gender, device_type, ad_position, time_of_day, 
#  user_interests_match, previous_clicks, session_duration]

np.random.seed(42)

# Generate data for clickers and non-clickers
n_clickers = 200
n_non_clickers = 800

# Clickers (more engaged users)
X_click = np.random.randn(n_clickers, 8) * \
          np.array([10, 0.5, 0.3, 1, 3, 1.5, 2, 20]) + \
          np.array([35, 1, 1, 2, 14, 8, 5, 300])

# Non-clickers
X_no_click = np.random.randn(n_non_clickers, 8) * \
             np.array([15, 0.5, 0.3, 1, 4, 1, 1, 30]) + \
             np.array([45, 0, 2, 5, 10, 3, 1, 150])

X = np.vstack([X_click, X_no_click])
y = np.array([1] * n_clickers + [0] * n_non_clickers)

# Shuffle
indices = np.random.permutation(1000)
X = X[indices]
y = y[indices]

# Split
X_train, X_val, X_test = X[:600], X[600:800], X[800:]
y_train, y_val, y_test = y[:600], y[600:800], y[800:]

# Train CTR prediction model
model = LightGBM(
    n_estimators=150,
    learning_rate=0.1,
    num_leaves=31,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    objective='binary'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=15,
    verbose=30
)

# Evaluate
test_acc = model.score(X_test, y_test)

print(f"\nClick-Through Rate Prediction:")
print("="*60)
print(f"Test Accuracy: {test_acc:.2%}")

# Calculate additional metrics
predictions = model.predict(X_test)
predicted_classes = (predictions >= 0.5).astype(int)

true_positives = np.sum((predicted_classes == 1) & (y_test == 1))
false_positives = np.sum((predicted_classes == 1) & (y_test == 0))
false_negatives = np.sum((predicted_classes == 0) & (y_test == 1))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.4f}")

# Feature importance
feature_names = ['Age', 'Gender', 'Device', 'Ad Position', 
                'Time of Day', 'Interest Match', 'Previous Clicks', 'Session Duration']
importance = model.get_feature_importance('gain')

print("\nTop Features for CTR:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name:20s}: {imp:.4f}")

# Predict new users
new_users = np.array([
    [28, 1, 1, 1, 14, 9, 8, 450],   # Engaged user, good targeting
    [55, 0, 2, 8, 3, 2, 0, 60]       # Less engaged user, poor ad position
])

click_probabilities = model.predict(new_users)

print("\nPredicted Click Probabilities:")
for i, prob in enumerate(click_probabilities):
    likelihood = "HIGH" if prob >= 0.5 else "LOW"
    print(f"User {i+1}: {likelihood} ({prob:.2%} probability)")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _18_lightgbm.py
    # numpy is the only requirement. Output is ASCII-only.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Regression demo: predict y = x^2 + noise ---
    print("=" * 55)
    print("DEMO 1 - Regression: y = x^2 + noise")
    print("Shows leaf-wise boosting fitting a curve trees cannot")
    print("express directly - only as a staircase of leaf weights.")
    print("=" * 55)

    X_reg = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_reg = X_reg.ravel() ** 2 + np.random.randn(200) * 0.5
    # Shuffle so train and test cover the same x range. linspace is sorted, and a
    # tree cannot extrapolate: without this the test set sits entirely to the right
    # of the training data and every prediction clamps at the training maximum.
    idx_reg = np.random.permutation(200)
    X_reg, y_reg = X_reg[idx_reg], y_reg[idx_reg]
    X_tr, X_te = X_reg[:150], X_reg[150:]
    y_tr, y_te = y_reg[:150], y_reg[150:]

    reg_model = LightGBM(
        n_estimators=60,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=10,
        lambda_l2=1.0
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
    print("Same trees, different loss: gradient = p - y and")
    print("hessian = p(1-p), then sigmoid turns scores into P(class).")
    print("=" * 55)

    X0 = np.random.randn(100, 2) + np.array([-2, -2])
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X_cls = np.vstack([X0, X1])
    y_cls = np.array([0] * 100 + [1] * 100)
    idx_cls = np.random.permutation(200)
    X_cls, y_cls = X_cls[idx_cls], y_cls[idx_cls]
    X_tr2, X_te2 = X_cls[:150], X_cls[150:]
    y_tr2, y_te2 = y_cls[:150], y_cls[150:]

    cls_model = LightGBM(
        n_estimators=40,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=10,
        objective='binary'
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

    # --- The two mechanics that make LightGBM LightGBM ---
    print("\n" + "=" * 55)
    print("DEMO 3 - Inside the model: histogram bins and the")
    print("num_leaves budget, on a mix of continuous, ordinal")
    print("and binary features (feature 5 is pure noise).")
    print("=" * 55)

    n = 200
    f0, f1, f2 = np.random.randn(n), np.random.randn(n), np.random.randn(n)
    f3 = np.random.randint(1, 5, n).astype(float)     # 4-level ordinal
    f4 = np.random.randint(0, 2, n).astype(float)     # binary flag
    f5 = np.random.randn(n)                           # pure noise
    X_mix = np.column_stack([f0, f1, f2, f3, f4, f5])
    y_mix = 3 * f0 - 2 * f1 + f2 + 2 * f3 + 6 * f4 + np.random.randn(n) * 0.5
    idx_mix = np.random.permutation(n)
    X_mix, y_mix = X_mix[idx_mix], y_mix[idx_mix]
    X_tr3, X_te3 = X_mix[:150], X_mix[150:]
    y_tr3, y_te3 = y_mix[:150], y_mix[150:]

    mix_model = LightGBM(
        n_estimators=40,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=10
    )
    mix_model.fit(X_tr3, y_tr3)
    print(f"Train R2 : {mix_model.score(X_tr3, y_tr3):.4f}")
    print(f"Test  R2 : {mix_model.score(X_te3, y_te3):.4f}")

    # (a) Histogram binning: max_bin is a CEILING, not a quota. A feature with
    #     only k distinct values must produce exactly k bins.
    binned = mix_model._apply_binning(X_tr3)
    print(f"\nBins actually used per feature (max_bin={mix_model.max_bin}):")
    kinds = ["continuous", "continuous", "continuous",
             "ordinal 1-4", "binary 0/1", "noise"]
    for j in range(X_tr3.shape[1]):
        print(f"  feature {j} ({kinds[j]:11s}): {len(np.unique(binned[:, j])):3d} bins")

    # (b) Leaf-wise growth is bounded by num_leaves, not by depth.
    def count_leaves(node):
        if node['type'] == 'leaf':
            return 1
        return count_leaves(node['left']) + count_leaves(node['right'])

    def tree_depth(node):
        if node['type'] == 'leaf':
            return 0
        return 1 + max(tree_depth(node['left']), tree_depth(node['right']))

    leaves = [count_leaves(t) for t in mix_model.trees]
    depths = [tree_depth(t) for t in mix_model.trees]
    print(f"\nLeaf budget num_leaves={mix_model.num_leaves}")
    print(f"  leaves per tree : min={min(leaves)}  max={max(leaves)}")
    print(f"  depth per tree  : min={min(depths)}  max={max(depths)}")
    print(f"  no tree exceeds the budget: {max(leaves) <= mix_model.num_leaves}")
    print("  Some trees stop short of 15 because min_data_in_leaf=10 binds first.")
    print("  A level-wise tree with 13 leaves would be 4 levels deep and balanced;")
    print("  best-first growth spends its budget where the gain is, so it goes deeper.")

    # (c) Which features did the ensemble actually rely on?
    print("\nFeature importance by gain:")
    importance = mix_model.get_feature_importance('gain')
    for j, imp in enumerate(importance):
        print(f"  feature {j} ({kinds[j]:11s}): {imp:.4f} {'#' * int(imp * 40)}")
    print("\nFeature 5 is noise and should sit at the bottom.")
