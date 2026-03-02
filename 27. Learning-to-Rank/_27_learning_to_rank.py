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
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_samples_split=10, subsample=0.7, random_state=None):
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
            
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        
        # Model components
        self.trees_ = []  # List of decision trees
        self.base_score_ = 0.0  # Initial prediction
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_dcg(self, relevances, k=None):
        """
        Compute Discounted Cumulative Gain
        
        DCG measures the quality of ranking by giving more weight to
        highly relevant documents and documents appearing earlier
        
        Formula: DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
        
        Parameters:
        -----------
        relevances : array-like
            Relevance labels in ranked order
        k : int, optional
            Compute DCG@k (top-k positions only)
            
        Returns:
        --------
        dcg : float
            Discounted Cumulative Gain score
        """
        if k is not None:
            relevances = relevances[:k]
        
        if len(relevances) == 0:
            return 0.0
        
        # DCG = Σ (2^rel - 1) / log2(position + 1)
        gains = 2 ** relevances - 1
        discounts = np.log2(np.arange(len(relevances)) + 2)  # +2 because positions start at 1
        return np.sum(gains / discounts)
    
    def _compute_ndcg(self, relevances, predicted_scores, k=None):
        """
        Compute Normalized Discounted Cumulative Gain
        
        NDCG normalizes DCG by the ideal DCG (perfect ranking)
        Range: [0, 1] where 1 is perfect ranking
        
        Parameters:
        -----------
        relevances : array-like
            True relevance labels
        predicted_scores : array-like
            Predicted relevance scores
        k : int, optional
            Compute NDCG@k (evaluate top-k positions only)
            
        Returns:
        --------
        ndcg : float
            Normalized DCG score (0 to 1)
        """
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(-predicted_scores)
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
    
    def _compute_lambda_gradients(self, query_relevances, query_scores):
        """
        Compute LambdaRank gradients
        
        LambdaRank uses gradients that directly optimize ranking metrics.
        For each pair of documents (i, j) where i should rank higher than j,
        we compute gradients based on how swapping them affects NDCG.
        
        Parameters:
        -----------
        query_relevances : array-like, shape (n_docs,)
            Relevance labels for documents in this query
        query_scores : array-like, shape (n_docs,)
            Current predicted scores for documents
            
        Returns:
        --------
        gradients : array-like, shape (n_docs,)
            Lambda gradients for each document
        """
        n_docs = len(query_relevances)
        gradients = np.zeros(n_docs)
        
        # Compute current NDCG
        current_ndcg = self._compute_ndcg(query_relevances, query_scores)
        
        # For each pair of documents
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                # Skip if relevances are the same
                if query_relevances[i] == query_relevances[j]:
                    continue
                
                # Determine which document should rank higher
                if query_relevances[i] > query_relevances[j]:
                    higher_idx, lower_idx = i, j
                else:
                    higher_idx, lower_idx = j, i
                
                # If prediction is wrong (lower score for more relevant doc)
                if query_scores[higher_idx] < query_scores[lower_idx]:
                    # Compute score difference
                    score_diff = query_scores[lower_idx] - query_scores[higher_idx]
                    
                    # Compute |ΔNDCG| (change in NDCG if we swap these two)
                    swapped_scores = query_scores.copy()
                    swapped_scores[higher_idx], swapped_scores[lower_idx] = \
                        query_scores[lower_idx], query_scores[higher_idx]
                    
                    swapped_ndcg = self._compute_ndcg(query_relevances, swapped_scores)
                    delta_ndcg = abs(swapped_ndcg - current_ndcg)
                    
                    # Lambda = |ΔNDCG| * σ'(score_diff)
                    # Using sigmoid: σ'(x) = σ(x) * (1 - σ(x))
                    sigmoid = 1.0 / (1.0 + np.exp(-score_diff))
                    lambda_val = delta_ndcg * sigmoid * (1 - sigmoid)
                    
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
            # Leaf node: return mean of gradients
            return {'leaf': True, 'value': np.mean(gradients)}
        
        # Find best split
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            # Try sorted unique values as potential split points
            unique_values = np.unique(X[:, feature_idx])
            if len(unique_values) <= 1:
                continue
            
            # Try midpoints between unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds[:10]:  # Limit splits for efficiency
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
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix where each row is a query-document pair
            Features describe the document and its relation to the query
            
        y : np.ndarray, shape (n_samples,)
            Relevance labels (e.g., 0=irrelevant, 1=somewhat relevant, 
            2=relevant, 3=highly relevant)
            
        query_ids : np.ndarray, shape (n_samples,)
            Query ID for each sample
            Documents with the same query_id belong to the same query
            
        Returns:
        --------
        self : object
            Fitted model
        """
        n_samples = X.shape[0]
        
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
                subsample_indices = np.random.choice(n_samples, n_subsample, replace=False)
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
            
            # Print progress
            if (iteration + 1) % 20 == 0:
                avg_ndcg = 0.0
                for query_id in unique_queries:
                    query_mask = query_ids == query_id
                    ndcg = self._compute_ndcg(y[query_mask], predictions[query_mask])
                    avg_ndcg += ndcg
                avg_ndcg /= len(unique_queries)
                print(f"Iteration {iteration + 1}/{self.n_estimators}, Avg NDCG: {avg_ndcg:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict relevance scores for query-document pairs
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix of query-document pairs
            
        Returns:
        --------
        scores : np.ndarray, shape (n_samples,)
            Predicted relevance scores (higher = more relevant)
        """
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
            Dictionary mapping query_id to ranked document indices
            (sorted from most relevant to least relevant)
        """
        scores = self.predict(X)
        rankings = {}
        
        unique_queries = np.unique(query_ids)
        for query_id in unique_queries:
            query_mask = query_ids == query_id
            query_scores = scores[query_mask]
            query_indices = np.where(query_mask)[0]
            
            # Sort by score (descending)
            sorted_order = np.argsort(-query_scores)
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
            Dictionary with NDCG scores for each query and average
        """
        predictions = self.predict(X)
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

if __name__ == "__main__":
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
    
    # Train the model
    print("\nTraining Learning-to-Rank model...")
    ltr = LearningToRank(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    ltr.fit(X_train, y_train, query_ids_train)
    
    # Get rankings
    rankings = ltr.rank(X_train, query_ids_train)
    
    print("\nRanking Results:")
    for query_id, doc_indices in rankings.items():
        print(f"\nQuery {query_id} - Ranked Documents:")
        query_mask = query_ids_train == query_id
        true_relevances = y_train[query_mask]
        
        for rank, doc_idx in enumerate(doc_indices, 1):
            local_idx = doc_idx - np.where(query_mask)[0][0]
            relevance = true_relevances[local_idx]
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
    
    np.random.seed(42)
    n_queries = 5
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
    
    # Train model
    print("\nTraining model...")
    ltr_ecommerce = LearningToRank(
        n_estimators=60,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    ltr_ecommerce.fit(X_ecommerce, y_ecommerce, query_ids_ecommerce)
    
    # Evaluate
    print("\nE-commerce Model Performance:")
    ndcg = ltr_ecommerce.evaluate(X_ecommerce, y_ecommerce, query_ids_ecommerce, k=5)
    print(f"Average NDCG@5: {ndcg['average']:.4f}")
    
    # Show top-3 products for first query
    print("\nTop 3 Products for Query 0:")
    rankings = ltr_ecommerce.rank(X_ecommerce, query_ids_ecommerce)
    top_3 = rankings[0][:3]
    
    for rank, doc_idx in enumerate(top_3, 1):
        features = X_ecommerce[doc_idx]
        label = y_ecommerce[doc_idx]
        print(f"\n  Rank {rank} (Relevance: {label}):")
        print(f"    Price Score: {features[0]:.2f}")
        print(f"    Rating: {features[1] * 5:.1f}/5.0")
        print(f"    Reviews: {int(features[2] * 1000)}")
        print(f"    In Stock: {'Yes' if features[4] > 0.5 else 'No'}")
    
    # ========================================================================
    # Example 3: Hyperparameter Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Impact of Hyperparameters")
    print("=" * 70)
    
    # Use the search engine data from Example 1
    configs = [
        {'n_estimators': 30, 'learning_rate': 0.05, 'max_depth': 3},
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 4},
        {'n_estimators': 80, 'learning_rate': 0.2, 'max_depth': 5},
    ]
    
    print("\nComparing different hyperparameter configurations:")
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        model = LearningToRank(**config, random_state=42)
        model.fit(X_train, y_train, query_ids_train)
        
        ndcg = model.evaluate(X_train, y_train, query_ids_train, k=5)
        print(f"  NDCG@5: {ndcg['average']:.4f}")
    
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
