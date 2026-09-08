import numpy as np

class LearningToRank:
    """
    Learning-to-Rank (LambdaRank-based) Implementation from Scratch
    
    Learning-to-Rank (LTR) is a machine learning approach for ranking items in information
    retrieval and recommendation systems. Instead of predicting exact relevance scores,
    it learns to order items correctly based on their relative importance.
    
    Key Idea: "Learn to order items by their relevance, not predict exact scores"
    
    Use Cases:
    - Search Engines: Rank web pages by relevance to query
    - Recommendation Systems: Order products/movies by user preference
    - Question Answering: Rank candidate answers by correctness
    - Document Retrieval: Order documents by relevance to query
    - E-commerce: Rank products by purchase likelihood
    - Job Matching: Rank job candidates or job postings
    
    Key Concepts:
        Query-Document Pairs: Each training example is a query with multiple documents
        Relevance Labels: Graded relevance (0=irrelevant, 1=somewhat, 2=relevant, 3=highly relevant)
        Pairwise Comparison: Learn from pairs of documents (which should rank higher)
        NDCG: Normalized Discounted Cumulative Gain (standard evaluation metric)
        LambdaRank: Use gradients based on ranking metrics (not loss directly)

    LambdaRank Gradient (Burges 2010, "From RankNet to LambdaRank to LambdaMART"):
        For every pair (i, j) inside one query whose labels differ, let i be the
        MORE relevant document and j the less relevant one. Write s_i, s_j for the
        current model scores and r_i, r_j for their current 1-based rank positions
        (rank 1 = highest score). Then

            dNDCG_ij = |2^rel_i - 2^rel_j| * |1/log2(1 + r_i) - 1/log2(1 + r_j)| / IDCG
            lambda_ij = dNDCG_ij * sigma / (1 + exp(sigma * (s_i - s_j)))
            lambda_i  = sum_j lambda_ij  -  sum_j lambda_ji

        with sigma = 1.0. dNDCG_ij is exactly the NDCG change from exchanging the two
        documents' POSITIONS, in closed form - it is not obtained by swapping scores.
        The logistic factor sigma/(1 + exp(sigma*(s_i - s_j))) -> 1 when the pair is
        badly mis-ordered (s_i << s_j) and -> 0 when the pair is already confidently
        correct, so no explicit "only if mis-ordered" test is needed (and adding one
        would freeze the model: on round 0 every score is the same constant).

        A regression tree is then fitted to lambda_i and ADDED to the scores, so a
        document with a positive lambda is pushed up the ranking.

    Simplifications vs. canonical LambdaMART:
        - Leaf value = mean(lambda) in the leaf. Production LambdaMART instead takes a
          Newton step, sum_i lambda_i / sum_i w_i, where w_i is the second derivative
          of the pairwise loss. See "Simplification vs. canonical LambdaMART" in
          _27_learning_to_rank.md.
        - The split search scores at most 10 candidate thresholds per feature
          (evenly spaced through the sorted midpoints), not every midpoint.
        - dNDCG is computed on the full result list by default. Set ndcg_k=k to
          truncate it at the same cutoff you evaluate with.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_samples_split=10, subsample=0.7, random_state=None,
                 verbose=20, ndcg_k=None):
        """
        Initialize the Learning-to-Rank model
        
        This implementation uses a gradient boosting approach with LambdaRank-style
        gradients that directly optimize ranking metrics (NDCG).
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of boosting stages (trees) to build
            - More trees: Better performance but slower, risk of overfitting
            - Fewer trees: Faster but may underfit
            Typical values: 50-500
            
        learning_rate : float, default=0.1
            Learning rate (shrinkage parameter)
            - Smaller values: More robust, needs more trees
            - Larger values: Faster learning, may overfit
            Typical range: 0.01-0.3
            
        max_depth : int, default=6
            Maximum depth of each tree
            - Deeper trees: Can capture complex patterns
            - Shallow trees: More regularization, faster
            Typical range: 3-10
            
        min_samples_split : int, default=10
            Minimum samples required to split a node
            - Higher values: More regularization
            - Lower values: More flexible, may overfit
            Typical range: 5-50
            
        subsample : float, default=0.7
            Fraction of samples to use for each tree
            - Less than 1.0: Stochastic gradient boosting (more robust)
            - Range: 0.5-1.0
            
        random_state : int or None, default=None
            Seed for the model's own random number generator
            - Any int makes the per-tree row subsampling reproducible
            - None draws fresh randomness on every run
            Note: the seed is kept in a PRIVATE np.random.RandomState, so building a
            model never disturbs the caller's global numpy random stream.
            Typical values: 0, 42, or None

        verbose : int, default=20
            Print an "Iteration t/T, Avg NDCG" line every `verbose` boosting rounds
            - 0 (or None) silences training completely
            - Larger values print less often
            - The NDCG it reports is measured at the ndcg_k cutoff below (full list
              when ndcg_k is None), i.e. the objective the gradients optimise, and
              the label becomes "Avg NDCG@k" when ndcg_k is set
            Typical values: 0 when scripting or grid-searching, 10-50 when exploring

        ndcg_k : int or None, default=None
            Truncation cutoff used when computing dNDCG inside the lambda gradients
            - None optimises the FULL result list (every position counts)
            - An int k makes training optimise NDCG@k, matching evaluate(..., k=k)
            - Aligning the two usually helps. Measured on the e-commerce data of the
              demo below, ndcg_k=5 lifts held-out NDCG@5 from 0.9151 to 0.9472 and
              gets there in fewer rounds.
            - Caveat: pairs whose documents BOTH sit below position k contribute no
              gradient, so a very small k on a very long result list thins the signal
            Typical values: None, or the same k you report (5 or 10)
            The default is None only because that is the behaviour earlier versions
            of this file had; ndcg_k=5 is usually the better choice.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose
        self.ndcg_k = ndcg_k
        
        # Model components
        self.trees_ = []  # List of decision trees
        self.base_score_ = None  # Initial prediction; set by fit()
        
        # Private RNG: seeding this does NOT touch the caller's global numpy stream
        self._rng = np.random.RandomState(random_state)
    
    def _compute_dcg(self, relevances, k=None):
        """
        Compute Discounted Cumulative Gain
        
        DCG measures the quality of ranking by giving more weight to
        highly relevant documents and documents appearing earlier
        
        Formula: DCG@k = sum_i (2^rel_i - 1) / log2(i + 1)
        
        Parameters:
        -----------
        relevances : array-like, shape (n_docs,)
            Relevance labels in ranked order (list or np.ndarray)
        k : int, optional
            Compute DCG@k (top-k positions only)
            
        Returns:
        --------
        dcg : float
            Discounted Cumulative Gain score
        """
        # Accept plain Python lists as well as arrays; float avoids the silent
        # integer overflow of 2**rel for int64 labels >= 63.
        relevances = np.asarray(relevances, dtype=float)

        if k is not None:
            relevances = relevances[:k]
        
        if len(relevances) == 0:
            return 0.0
        
        # DCG = sum (2^rel - 1) / log2(position + 1)
        gains = 2.0 ** relevances - 1.0
        discounts = np.log2(np.arange(len(relevances)) + 2)  # +2 because positions start at 1
        return np.sum(gains / discounts)
    
    def _compute_ndcg(self, relevances, predicted_scores, k=None):
        """
        Compute Normalized Discounted Cumulative Gain
        
        NDCG normalizes DCG by the ideal DCG (perfect ranking)
        Range: [0, 1] where 1 is perfect ranking
        
        Parameters:
        -----------
        relevances : array-like, shape (n_docs,)
            True relevance labels (list or np.ndarray)
        predicted_scores : array-like, shape (n_docs,)
            Predicted relevance scores (list or np.ndarray)
        k : int, optional
            Compute NDCG@k (evaluate top-k positions only)
            
        Returns:
        --------
        ndcg : float
            Normalized DCG score (0 to 1)
        """
        relevances = np.asarray(relevances, dtype=float)
        predicted_scores = np.asarray(predicted_scores, dtype=float)

        # Sort by predicted scores (descending). kind='stable' keeps tied scores in
        # their input order, so an untrained (constant-score) model is scored on the
        # input order rather than on an arbitrary permutation.
        sorted_indices = np.argsort(-predicted_scores, kind='stable')
        sorted_relevances = relevances[sorted_indices]
        
        # Compute DCG
        dcg = self._compute_dcg(sorted_relevances, k)
        
        # Compute ideal DCG (sort by true relevances)
        ideal_relevances = np.sort(relevances)[::-1]
        idcg = self._compute_dcg(ideal_relevances, k)
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _pairwise_weight(self, score_higher, score_lower, sigma=1.0):
        """
        Logistic weight of a LambdaRank pair

        Implements    sigma / (1 + exp(sigma * (s_i - s_j)))
        where s_i is the score of the MORE relevant document of the pair.

        Behaviour (this is the whole point of the term):
        - pair badly mis-ordered (s_i much smaller than s_j) -> weight -> sigma
        - pair exactly tied                                  -> weight  = sigma / 2
        - pair confidently correct (s_i much larger than s_j)-> weight -> 0

        So a correctly-ordered pair fades out on its own and no "only if the pair is
        mis-ordered" precondition is needed. (An earlier version of this file used the
        sigmoid DERIVATIVE sigma'(x) = sigma(x)(1 - sigma(x)) instead, which peaks at a
        TIE and decays for badly mis-ordered pairs - exactly backwards.)

        The two algebraically identical branches exist only to keep exp() from
        overflowing: each one calls exp() on a non-positive argument.

        Parameters:
        -----------
        score_higher : float
            Current score of the more relevant document
        score_lower : float
            Current score of the less relevant document
        sigma : float, default=1.0
            Shape parameter of the logistic; larger = sharper transition

        Returns:
        --------
        weight : float
            Value in (0, sigma)
        """
        x = sigma * (score_higher - score_lower)
        if x >= 0:
            e = np.exp(-x)
            return sigma * e / (1.0 + e)
        e = np.exp(x)
        return sigma / (1.0 + e)

    def _compute_lambda_gradients(self, query_relevances, query_scores):
        """
        Compute LambdaRank gradients for the documents of ONE query
        
        For every pair (i, j) with DIFFERENT relevance labels, let i be the more
        relevant document. Two factors are multiplied:

            dNDCG_ij  - how much NDCG would change if i and j exchanged their current
                        rank POSITIONS. Closed form (no re-sorting needed):

                        |2^rel_i - 2^rel_j| * |1/log2(1+r_i) - 1/log2(1+r_j)| / IDCG

                        where r is the 1-based rank position under the current scores.
            weight    - sigma / (1 + exp(sigma * (s_i - s_j))), see _pairwise_weight.

            lambda_ij = dNDCG_ij * weight
            lambda_i  = sum_j lambda_ij - sum_j lambda_ji

        Note there is NO "skip correctly ordered pairs" test: the weight already
        handles that smoothly, and skipping would produce an all-zero gradient on
        round 0 (where every score equals the constant base score) and the model
        would never leave its initialisation.
        
        Parameters:
        -----------
        query_relevances : array-like, shape (n_docs,)
            Relevance labels for documents in this query
        query_scores : array-like, shape (n_docs,)
            Current predicted scores for documents
            
        Returns:
        --------
        gradients : np.ndarray, shape (n_docs,)
            Lambda gradients for each document; positive = should move up
        """
        relevances = np.asarray(query_relevances, dtype=float)
        scores = np.asarray(query_scores, dtype=float)
        n_docs = len(relevances)
        gradients = np.zeros(n_docs)
        
        # IDCG of this query: the normaliser that turns a DCG change into an NDCG
        # change. A query whose documents are all irrelevant has IDCG = 0 and
        # carries no ranking signal at all.
        idcg = self._compute_dcg(np.sort(relevances)[::-1], self.ndcg_k)
        if idcg == 0:
            return gradients

        # Current rank position of every document: 1 = best score.
        order = np.argsort(-scores, kind='stable')
        ranks = np.empty(n_docs, dtype=int)
        ranks[order] = np.arange(1, n_docs + 1)

        # Position discount 1/log2(1 + rank). With ndcg_k set, anything ranked below
        # the cutoff contributes nothing, which is what NDCG@k truncation means.
        discounts = 1.0 / np.log2(ranks + 1.0)
        if self.ndcg_k is not None:
            discounts = np.where(ranks <= self.ndcg_k, discounts, 0.0)

        gains = 2.0 ** relevances - 1.0
        
        # For each pair of documents
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                # Skip if relevances are the same: exchanging them cannot change NDCG
                if relevances[i] == relevances[j]:
                    continue
                
                # Determine which document should rank higher
                if relevances[i] > relevances[j]:
                    higher_idx, lower_idx = i, j
                else:
                    higher_idx, lower_idx = j, i
                
                # |dNDCG| for exchanging the two POSITIONS (closed form)
                delta_ndcg = abs(
                    (gains[higher_idx] - gains[lower_idx]) *
                    (discounts[higher_idx] - discounts[lower_idx])
                ) / idcg
                    
                # lambda_ij = |dNDCG_ij| * sigma / (1 + exp(sigma * (s_i - s_j)))
                lambda_val = delta_ndcg * self._pairwise_weight(
                    scores[higher_idx], scores[lower_idx]
                )
                    
                # Update gradients
                gradients[higher_idx] += lambda_val  # Push higher
                gradients[lower_idx] -= lambda_val   # Push lower
        
        return gradients
    
    def _build_tree(self, X, gradients, depth=0):
        """
        Build a regression tree to fit gradients
        
        This is a simplified gradient boosting tree that fits the
        LambdaRank gradients.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        gradients : np.ndarray, shape (n_samples,)
            Target gradients to fit
        depth : int
            Current depth of the tree
            
        Returns:
        --------
        tree : dict
            Tree structure with split information or leaf value
        """
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            # Leaf node: return mean of gradients.
            # Canonical LambdaMART uses the Newton step sum(lambda_i) / sum(w_i)
            # instead, where w_i is the second derivative of the pairwise loss; the
            # plain mean is a deliberate teaching simplification (see the class
            # docstring and the .md's "Simplification vs. canonical LambdaMART").
            return {'leaf': True, 'value': np.mean(gradients)}
        
        # Find best split.
        # best_gain starts at 0.0, not -inf, so a split is only accepted if it
        # genuinely reduces the gradient variance (the "gamma pruning" idea).
        # By the law of total variance the gain computed below IS the between-group
        # variance, so it can never be negative; in exact arithmetic what starting
        # at 0.0 rejects is therefore the zero-gain candidate - a split whose two
        # sides have the same mean gradient, which carries no information. In
        # floating point that rejection is not airtight: a genuinely zero-gain
        # candidate can round to a positive gain many orders of magnitude below the
        # gradient scale and be accepted, so a constant-gradient node often splits
        # anyway. Both children then carry the same leaf value, so predictions are
        # unchanged - it costs a wasted node, not accuracy. If nothing beats 0.0
        # the node becomes a leaf instead of being split pointlessly.
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            # Try sorted unique values as potential split points
            unique_values = np.unique(X[:, feature_idx])
            if len(unique_values) <= 1:
                continue
            
            # Try midpoints between unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            # Limit the search to at most 10 candidates for efficiency, but take
            # them EVENLY SPACED through the sorted midpoints rather than the 10
            # smallest - otherwise the upper part of a feature's range is never
            # considered and the tree can only ever split near the minimum.
            if len(thresholds) > 10:
                picks = np.linspace(0, len(thresholds) - 1, 10).astype(int)
                thresholds = thresholds[picks]

            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                
                # Compute gain (reduction in variance)
                left_gradients = gradients[left_mask]
                right_gradients = gradients[right_mask]
                
                total_var = np.var(gradients)
                left_var = np.var(left_gradients)
                right_var = np.var(right_gradients)
                
                n_left = len(left_gradients)
                n_right = len(right_gradients)
                
                gain = total_var - (n_left * left_var + n_right * right_var) / n_samples
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # If no valid split found, create leaf
        if best_feature is None:
            return {'leaf': True, 'value': np.mean(gradients)}
        
        # Split and recurse
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], gradients[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], gradients[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _predict_tree(self, tree, X):
        """
        Make predictions using a single tree
        
        Parameters:
        -----------
        tree : dict
            Tree structure from _build_tree
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Tree predictions
        """
        if tree['leaf']:
            return np.full(X.shape[0], tree['value'])
        
        predictions = np.zeros(X.shape[0])
        
        left_mask = X[:, tree['feature']] <= tree['threshold']
        right_mask = ~left_mask
        
        if np.any(left_mask):
            predictions[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        if np.any(right_mask):
            predictions[right_mask] = self._predict_tree(tree['right'], X[right_mask])
        
        return predictions
    
    def fit(self, X, y, query_ids):
        """
        Train the Learning-to-Rank model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix where each row is a query-document pair
            Features describe the document and its relation to the query
            A 1-D input is read as a single feature column
            
        y : array-like, shape (n_samples,)
            Relevance labels (e.g., 0=irrelevant, 1=somewhat relevant, 
            2=relevant, 3=highly relevant)
            
        query_ids : array-like, shape (n_samples,)
            Query ID for each sample
            Documents with the same query_id belong to the same query
            
        Returns:
        --------
        self : object
            Fitted model
        """
        # Accept lists / 1-D input, not just 2-D float arrays
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float)
        query_ids = np.asarray(query_ids)

        n_samples = X.shape[0]

        # Start from a clean ensemble so re-fitting replaces the model
        # instead of stacking a second ensemble on top of the first
        self.trees_ = []
        
        # Initialize predictions with base score
        self.base_score_ = np.mean(y)
        predictions = np.full(n_samples, self.base_score_)
        
        # Get unique queries
        unique_queries = np.unique(query_ids)
        
        # Build boosting ensemble
        for iteration in range(self.n_estimators):
            # Compute gradients for each query
            all_gradients = np.zeros(n_samples)
            
            for query_id in unique_queries:
                # Get documents for this query
                query_mask = query_ids == query_id
                query_relevances = y[query_mask]
                query_scores = predictions[query_mask]
                
                # Compute lambda gradients
                gradients = self._compute_lambda_gradients(query_relevances, query_scores)
                all_gradients[query_mask] = gradients
            
            # Subsample data for this iteration
            if self.subsample < 1.0:
                n_subsample = int(self.subsample * n_samples)
                subsample_indices = self._rng.choice(n_samples, n_subsample, replace=False)
            else:
                subsample_indices = np.arange(n_samples)
            
            # Build tree to fit gradients
            tree = self._build_tree(
                X[subsample_indices],
                all_gradients[subsample_indices]
            )
            
            self.trees_.append(tree)
            
            # Update predictions
            tree_predictions = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_predictions
            
            # Print progress (verbose=0 silences training entirely).
            # The diagnostic is measured at the SAME cutoff the gradients optimise
            # (self.ndcg_k), so the number printed here is the objective actually
            # being trained. With ndcg_k=None that is the full-list NDCG.
            if self.verbose and (iteration + 1) % self.verbose == 0:
                avg_ndcg = 0.0
                for query_id in unique_queries:
                    query_mask = query_ids == query_id
                    ndcg = self._compute_ndcg(y[query_mask], predictions[query_mask],
                                              self.ndcg_k)
                    avg_ndcg += ndcg
                avg_ndcg /= len(unique_queries)
                label = "Avg NDCG" if self.ndcg_k is None else f"Avg NDCG@{self.ndcg_k}"
                print(f"Iteration {iteration + 1}/{self.n_estimators}, {label}: {avg_ndcg:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict relevance scores for query-document pairs
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of query-document pairs
            A 1-D input is read as a single feature column
            
        Returns:
        --------
        scores : np.ndarray, shape (n_samples,)
            Predicted relevance scores (higher = more relevant)
        """
        if self.base_score_ is None:
            raise ValueError("Model is not fitted. Call fit(X, y, query_ids) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = np.full(X.shape[0], self.base_score_)
        
        for tree in self.trees_:
            tree_predictions = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_predictions
        
        return predictions
    
    def rank(self, X, query_ids):
        """
        Rank documents for each query
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix of query-document pairs
        query_ids : np.ndarray, shape (n_samples,)
            Query ID for each sample
            
        Returns:
        --------
        rankings : dict
            Dictionary mapping query_id -> np.ndarray of GLOBAL row indices into X,
            ordered most- to least-relevant (highest predicted score first).
            They index X directly, so X[rankings[q][0]] is the top document for
            query q; they are NOT positions within the query's own block.
        """
        scores = self.predict(X)
        query_ids = np.asarray(query_ids)
        rankings = {}
        
        unique_queries = np.unique(query_ids)
        for query_id in unique_queries:
            query_mask = query_ids == query_id
            query_scores = scores[query_mask]
            query_indices = np.where(query_mask)[0]
            
            # Sort by score (descending). kind='stable' so documents with equal
            # scores keep their input order - the same tie-breaking _compute_ndcg
            # uses, otherwise rank() and evaluate() could disagree on a tied query.
            sorted_order = np.argsort(-query_scores, kind='stable')
            rankings[query_id] = query_indices[sorted_order]
        
        return rankings
    
    def evaluate(self, X, y, query_ids, k=10):
        """
        Evaluate model performance using NDCG@k
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            True relevance labels
        query_ids : np.ndarray, shape (n_samples,)
            Query IDs
        k : int, default=10
            Evaluate NDCG at top-k positions
            
        Returns:
        --------
        ndcg_scores : dict
            Dictionary with NDCG scores for each query and average.
            Keys are 'query_<id>' for every query plus 'average'.
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=float)
        query_ids = np.asarray(query_ids)
        unique_queries = np.unique(query_ids)
        
        ndcg_scores = {}
        total_ndcg = 0.0
        
        for query_id in unique_queries:
            query_mask = query_ids == query_id
            query_relevances = y[query_mask]
            query_predictions = predictions[query_mask]
            
            ndcg = self._compute_ndcg(query_relevances, query_predictions, k)
            ndcg_scores[f'query_{query_id}'] = ndcg
            total_ndcg += ndcg
        
        ndcg_scores['average'] = total_ndcg / len(unique_queries)
        
        return ndcg_scores


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
USAGE EXAMPLE 1: Ranking three candidate pages for one search query

import numpy as np

# Two training queries. Features: [pagerank, query_match, freshness, domain_authority]
X_q1 = np.array([
    [0.8, 1.0, 0.90, 0.85],   # High quality Python tutorial
    [0.3, 0.5, 0.10, 0.40],   # Barely relevant
    [0.9, 1.0, 0.95, 0.90],   # Excellent Python docs
    [0.2, 0.0, 0.30, 0.30],   # Irrelevant
])
y_q1 = np.array([3, 1, 3, 0])          # graded relevance, 0=irrelevant .. 3=perfect
qid_q1 = np.array([1, 1, 1, 1])

X_q2 = np.array([
    [0.7, 0.8, 0.70, 0.75],   # Good ML intro
    [0.9, 1.0, 0.90, 0.95],   # Excellent ML course
    [0.4, 0.4, 0.40, 0.50],   # Somewhat related
])
y_q2 = np.array([2, 3, 1])
qid_q2 = np.array([2, 2, 2])

X = np.vstack([X_q1, X_q2])
y = np.concatenate([y_q1, y_q2])
query_ids = np.concatenate([qid_q1, qid_q2])

# Only 7 rows here, so min_samples_split must drop below the row count or every
# tree stays a bare leaf and the model can never learn anything.
ltr = LearningToRank(
    n_estimators=60,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=4,
    subsample=1.0,
    random_state=42,
    verbose=0            # silence the per-iteration progress lines
)
ltr.fit(X, y, query_ids)

print(f"Training NDCG@3: {ltr.evaluate(X, y, query_ids, k=3)['average']:.4f}")

# Rank three unseen candidate documents for a brand-new query id
X_new = np.array([
    [0.6, 0.8, 0.5, 0.70],   # Candidate A
    [0.9, 1.0, 0.9, 0.90],   # Candidate B
    [0.4, 0.6, 0.3, 0.50],   # Candidate C
])
query_ids_new = np.array([3, 3, 3])

rankings = ltr.rank(X_new, query_ids_new)      # GLOBAL row indices into X_new
print("Ranked documents for query 3:", rankings[3])
print("Raw scores:", np.round(ltr.predict(X_new), 4))
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _27_learning_to_rank.py
    # numpy only, seeded, ASCII-only output.
    # ----------------------------------------------------------------
    np.random.seed(42)

    print("=" * 70)
    print("LEARNING-TO-RANK - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    # ========================================================================
    # Example 1: Search Ranking - Simple Case
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Search Engine Ranking")
    print("=" * 70)
    
    # Simulate search engine data
    # Query 1: "python programming"
    # Documents: 5 web pages with different relevance
    
    # Features: [pagerank, num_query_terms, freshness, domain_authority]
    X_query1 = np.array([
        [0.8, 2, 0.9, 0.85],  # Doc 0: Highly relevant Python tutorial
        [0.3, 1, 0.1, 0.40],  # Doc 1: Barely related page
        [0.7, 2, 0.7, 0.75],  # Doc 2: Good Python guide
        [0.2, 0, 0.3, 0.30],  # Doc 3: Irrelevant page
        [0.9, 2, 0.95, 0.90], # Doc 4: Excellent Python documentation
    ])
    
    # Relevance labels: 0=irrelevant, 1=somewhat, 2=relevant, 3=highly relevant
    y_query1 = np.array([3, 1, 2, 0, 3])
    query_ids_1 = np.array([1, 1, 1, 1, 1])
    
    # Query 2: "machine learning"
    X_query2 = np.array([
        [0.6, 2, 0.6, 0.70],  # Doc 5: ML basics
        [0.9, 2, 0.9, 0.95],  # Doc 6: Excellent ML course
        [0.4, 1, 0.4, 0.50],  # Doc 7: Somewhat related
        [0.8, 2, 0.85, 0.80], # Doc 8: Great ML tutorial
    ])
    
    y_query2 = np.array([2, 3, 1, 3])
    query_ids_2 = np.array([2, 2, 2, 2])
    
    # Combine all data
    X_train = np.vstack([X_query1, X_query2])
    y_train = np.concatenate([y_query1, y_query2])
    query_ids_train = np.concatenate([query_ids_1, query_ids_2])
    
    print("\nTraining Data:")
    print(f"Total documents: {len(X_train)}")
    print(f"Number of queries: {len(np.unique(query_ids_train))}")
    print(f"Features per document: {X_train.shape[1]}")
    
    # Train the model.
    # NOTE min_samples_split=4: with only 9 rows the default of 10 would stop every
    # tree at its root, so the ensemble could never learn a single split.
    print("\nTraining Learning-to-Rank model...")
    ltr = LearningToRank(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=4,
        subsample=1.0,
        random_state=42,
        verbose=0
    )

    # Baseline: before training every document scores the same constant, so the
    # "ranking" is just the input order. This is what we have to beat.
    baseline = np.full(len(y_train), float(np.mean(y_train)))
    base_ndcg = np.mean([
        ltr._compute_ndcg(y_train[query_ids_train == q], baseline[query_ids_train == q], 5)
        for q in np.unique(query_ids_train)
    ])
    print(f"NDCG@5 before training (input order): {base_ndcg:.4f}")

    ltr.fit(X_train, y_train, query_ids_train)
    
    # Get rankings
    rankings = ltr.rank(X_train, query_ids_train)
    
    print("\nRanking Results:")
    for query_id, doc_indices in rankings.items():
        print(f"\nQuery {query_id} - Ranked Documents:")
        
        # rank() returns GLOBAL row indices into X_train, so y_train can be
        # indexed with them directly - no local/global arithmetic needed.
        for rank, doc_idx in enumerate(doc_indices, 1):
            relevance = y_train[doc_idx]
            print(f"  Rank {rank}: Document {doc_idx} (Relevance: {relevance})")
    
    # Evaluate
    print("\nEvaluation Metrics:")
    ndcg_scores = ltr.evaluate(X_train, y_train, query_ids_train, k=5)
    for metric, score in ndcg_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # ========================================================================
    # Example 2: E-commerce Product Ranking
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: E-commerce Product Ranking")
    print("=" * 70)
    
    # User search: "wireless headphones"
    # Features: [price_score, rating, num_reviews, relevance_score, in_stock]
    
    n_queries = 12
    n_products_per_query = 8
    
    X_ecommerce = []
    y_ecommerce = []
    query_ids_ecommerce = []
    
    for query_id in range(n_queries):
        for _ in range(n_products_per_query):
            # Random features
            price_score = np.random.rand()  # 0=expensive, 1=cheap
            rating = np.random.uniform(3.0, 5.0)
            num_reviews = np.random.randint(0, 1000)
            relevance = np.random.rand()
            in_stock = np.random.choice([0, 1], p=[0.2, 0.8])
            
            features = [
                price_score,
                rating / 5.0,  # Normalize
                min(num_reviews / 1000, 1.0),  # Normalize
                relevance,
                in_stock
            ]
            
            # Generate label based on features
            label_score = (
                0.2 * price_score +
                0.3 * (rating / 5.0) +
                0.2 * min(num_reviews / 1000, 1.0) +
                0.3 * relevance
            )
            
            if label_score > 0.7:
                label = 3
            elif label_score > 0.5:
                label = 2
            elif label_score > 0.3:
                label = 1
            else:
                label = 0
            
            X_ecommerce.append(features)
            y_ecommerce.append(label)
            query_ids_ecommerce.append(query_id)
    
    X_ecommerce = np.array(X_ecommerce)
    y_ecommerce = np.array(y_ecommerce)
    query_ids_ecommerce = np.array(query_ids_ecommerce)
    
    print(f"\nE-commerce Dataset:")
    print(f"Total products: {len(X_ecommerce)}")
    print(f"Number of search queries: {n_queries}")
    print(f"Products per query: {n_products_per_query}")
    
    # SPLIT BY QUERY, NOT BY DOCUMENT.
    # Holding out whole queries (8-11) is the only honest test: if we held out
    # individual products we would still be training on their siblings from the
    # same query, and the reported NDCG would be optimistic.
    train_mask = query_ids_ecommerce < 8
    test_mask = ~train_mask
    X_tr_ec, y_tr_ec, q_tr_ec = X_ecommerce[train_mask], y_ecommerce[train_mask], query_ids_ecommerce[train_mask]
    X_te_ec, y_te_ec, q_te_ec = X_ecommerce[test_mask], y_ecommerce[test_mask], query_ids_ecommerce[test_mask]
    print(f"Train queries: 0-7 ({len(y_tr_ec)} products) | "
          f"Test queries: 8-11 ({len(y_te_ec)} products)")

    # Train model
    print("\nTraining model...")
    ltr_ecommerce = LearningToRank(
        n_estimators=60,
        learning_rate=0.15,
        max_depth=5,
        min_samples_split=4,
        subsample=1.0,
        random_state=42,
        verbose=0
    )
    ltr_ecommerce.fit(X_tr_ec, y_tr_ec, q_tr_ec)
    
    # Evaluate on the training queries AND on the four unseen queries
    print("\nE-commerce Model Performance:")
    base_tr = np.mean([
        ltr_ecommerce._compute_ndcg(y_tr_ec[q_tr_ec == q],
                                    np.zeros(np.sum(q_tr_ec == q)), 5)
        for q in np.unique(q_tr_ec)
    ])
    base_te = np.mean([
        ltr_ecommerce._compute_ndcg(y_te_ec[q_te_ec == q],
                                    np.zeros(np.sum(q_te_ec == q)), 5)
        for q in np.unique(q_te_ec)
    ])
    ndcg_tr = ltr_ecommerce.evaluate(X_tr_ec, y_tr_ec, q_tr_ec, k=5)
    ndcg = ltr_ecommerce.evaluate(X_te_ec, y_te_ec, q_te_ec, k=5)
    print(f"  Train NDCG@5: {base_tr:.4f} (input order) -> {ndcg_tr['average']:.4f} (trained)")
    print(f"  Test  NDCG@5: {base_te:.4f} (input order) -> {ndcg['average']:.4f} (trained)")
    
    # Show the full ranked list for one HELD-OUT query
    print("\nRanked products for held-out Query 8 (best first):")
    rankings = ltr_ecommerce.rank(X_te_ec, q_te_ec)
    for rank, doc_idx in enumerate(rankings[8], 1):
        features = X_te_ec[doc_idx]
        print(f"  Rank {rank}: true relevance {y_te_ec[doc_idx]}  "
              f"price={features[0]:.2f} rating={features[1] * 5:.1f}/5.0 "
              f"reviews={int(features[2] * 1000):4d} "
              f"in_stock={'Yes' if features[4] > 0.5 else 'No '}")
    print(f"  NDCG@5 for this query: {ndcg['query_8']:.4f}  (perfect = 1.0000)")
    
    # ========================================================================
    # Example 3: Hyperparameter Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Impact of Hyperparameters")
    print("=" * 70)
    
    # Use the e-commerce data from Example 2, scored on the HELD-OUT queries so
    # the comparison measures generalisation rather than memorisation.
    configs = [
        {'n_estimators': 30, 'learning_rate': 0.05, 'max_depth': 3},
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 4},
        {'n_estimators': 80, 'learning_rate': 0.2, 'max_depth': 5},
    ]
    
    print("\nComparing different hyperparameter configurations:")
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        model = LearningToRank(**config, min_samples_split=4, subsample=1.0,
                               random_state=42, verbose=0)
        model.fit(X_tr_ec, y_tr_ec, q_tr_ec)
        
        ndcg_train = model.evaluate(X_tr_ec, y_tr_ec, q_tr_ec, k=5)
        ndcg = model.evaluate(X_te_ec, y_te_ec, q_te_ec, k=5)
        print(f"  Train NDCG@5: {ndcg_train['average']:.4f}   "
              f"Test NDCG@5: {ndcg['average']:.4f}")

    # Learning curve: the whole point of LambdaRank is that ranking quality climbs
    # as trees are added. Each row refits from scratch with a different
    # n_estimators. Test NDCG rises sharply, then wobbles on a plateau - NDCG is a
    # step function of the ordering, so it is not guaranteed to improve every round.
    print("\nLearning curve (test NDCG@5 vs. number of trees):")
    print(f"  {0:3d} trees: {base_te:.4f}   (untrained baseline = input order)")
    for n_trees in [10, 20, 40, 60]:
        curve_model = LearningToRank(n_estimators=n_trees, learning_rate=0.15,
                                     max_depth=5, min_samples_split=4,
                                     subsample=1.0, random_state=42, verbose=0)
        curve_model.fit(X_tr_ec, y_tr_ec, q_tr_ec)
        curve_ndcg = curve_model.evaluate(X_te_ec, y_te_ec, q_te_ec, k=5)
        print(f"  {n_trees:3d} trees: {curve_ndcg['average']:.4f}")
    
    # Practical Tips
    print("\n" + "=" * 70)
    print("PRACTICAL TIPS FOR LEARNING-TO-RANK")
    print("=" * 70)
    
    tips = """
    1. FEATURE ENGINEERING:
       - Query-Document features: TF-IDF, BM25, cosine similarity
       - Document features: PageRank, freshness, length, readability
       - User features: Click-through rate, dwell time, bounce rate
       - Combine multiple feature types for best results
    
    2. DATA REQUIREMENTS:
       - Need query-document pairs with relevance labels
       - Minimum: ~1000 query-document pairs
       - Better: ~10,000+ pairs with diverse queries
       - Relevance labels: 0-4 scale is common (0=irrelevant to 4=perfect)
    
    3. HYPERPARAMETER TUNING:
       - n_estimators: Start with 100, increase if underfitting
       - learning_rate: 0.05-0.2 (smaller = more robust)
       - max_depth: 4-6 for most cases (deeper may overfit)
       - Balance training time vs. performance
    
    4. EVALUATION:
       - NDCG@k: Standard metric (k=5 or k=10 common)
       - Higher k: Evaluates more positions
       - Use hold-out queries for testing (not documents!)
       - Cross-validate across queries, not documents
    
    5. WHEN TO USE LEARNING-TO-RANK:
       + Search engines and information retrieval
       + Recommendation systems with rankings
       + Question answering systems
       + Any task where order matters more than exact scores
       + When you have query-document pairs with relevance labels
    
    6. COMMON ISSUES:
       - Overfitting: Reduce n_estimators or max_depth
       - Underfitting: Increase n_estimators or max_depth
       - Imbalanced labels: Use more training data or data augmentation
       - Slow training: Reduce n_estimators or use subsample < 1.0
    """
    
    print(tips)
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
