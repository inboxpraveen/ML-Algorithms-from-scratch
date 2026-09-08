import numpy as np

class MatrixFactorization:
    """
    Matrix Factorization Implementation from Scratch
    
    Matrix Factorization decomposes a matrix R into two lower-rank matrices U and V
    such that R ~= U x V^T. This technique is fundamental in collaborative filtering
    and dimensionality reduction.
    
    Key Idea: "Find hidden factors that explain observed patterns in data"
    
    Use Cases:
    - Recommender Systems: Predict user ratings for items (Netflix, Amazon)
    - Collaborative Filtering: Find similar users or items
    - Missing Value Imputation: Fill in missing entries in sparse matrices
    - Feature Extraction: Discover latent features in data
    - Dimensionality Reduction: Reduce high-dimensional data
    
    Key Concepts:
        Latent Factors: Hidden features that explain user-item interactions
        Low-Rank Approximation: Approximate large matrix with smaller matrices
        SGD Optimization: Learn factor matrices through gradient descent
        Regularization: Prevent overfitting with L2 penalty

    Prediction and Update Rules (biased MF, Koren et al. 2009):
        Prediction for user u and item i:
            r_hat_ui = mu + b_u + b_i + p_u . q_i
        where mu is the global mean rating, b_u / b_i are the user and item biases,
        and p_u / q_i are the length-n_factors latent factor vectors.

        Objective minimised over the observed ratings K (Koren et al. 2009,
        eq. 5), with the factor 1/2 that makes the 2s in the derivative cancel:
            L = (1/2) * sum_{(u,i) in K} [ (r_ui - r_hat_ui)^2
                  + lambda * (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2) ]
        The penalty sits INSIDE the sum because SGD applies the -lambda
        shrinkage on every visit to a parameter: a user rated n_u times pays
        n_u * lambda in total. Writing the penalty outside the sum, as one
        Frobenius norm per matrix, describes a DIFFERENT objective - one whose
        gradient is far from zero at the point this code converges to.
        (training_loss_ reports the Frobenius bookkeeping; see fit().)

        One SGD step on a single observed rating, with e_ui = r_ui - r_hat_ui:
            b_u <- b_u + alpha * (e_ui - lambda * b_u)
            b_i <- b_i + alpha * (e_ui - lambda * b_i)
            p_u <- p_u + alpha * (e_ui * q_i - lambda * p_u)
            q_i <- q_i + alpha * (e_ui * p_u - lambda * q_i)   [p_u = pre-update copy]
        alpha is learning_rate and lambda is regularization.

    Where min_rating / max_rating apply:
        Clipping to [min_rating, max_rating] is a PRESENTATION step. It is applied by
        predict(), recommend() and reconstruct_matrix(), never inside the SGD residual
        e_ui, because a clipped residual has zero gradient and lets a saturated factor
        run away. Set them to the true range of YOUR data - the defaults assume 1-5 stars.

    Simplifications vs. canonical Matrix Factorization:
        - SGD only. Alternating Least Squares (ALS), the standard choice for implicit
          feedback and for distributed training, is not implemented.
        - No temporal dynamics (timeSVD++), no implicit-feedback confidence weighting,
          no per-user/per-item adaptive regularization, no early stopping inside fit().
        See the "Simplification vs. canonical Matrix Factorization" section of
        _28_matrix_factorization.md for the formulas these variants add.
    """
    
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.02,
                 n_epochs=100, min_rating=1, max_rating=5, init_mean=0,
                 init_std=0.1, random_state=None, verbose=0):
        """
        Initialize the Matrix Factorization model
        
        Parameters:
        -----------
        n_factors : int, default=10
            Number of latent factors (dimensionality of factorized matrices)
            - Typical range: 5-100
            - Small values (5-20): Fast, less overfitting, less expressive
            - Large values (50-100): More expressive, slower, risk overfitting
            - Balance between model complexity and performance
            
        learning_rate : float, default=0.01
            Learning rate for gradient descent
            - Typical range: 0.001-0.1
            - Too high: Unstable training, overshooting
            - Too low: Slow convergence
            - Can use learning rate scheduling for better results
            
        regularization : float, default=0.02
            L2 regularization parameter (lambda)
            - Typical range: 0.001-0.1
            - Controls model complexity
            - Higher values: More regularization, less overfitting
            - Lower values: Less regularization, fits training data better
            
        n_epochs : int, default=100
            Number of training iterations over the dataset
            - Typical range: 50-500
            - More epochs: Better convergence, longer training time
            - Monitor training/validation loss to avoid overfitting
            
        min_rating : float, default=1
            Minimum possible rating value
            - Used to clip PREDICTIONS only (predict / recommend /
              reconstruct_matrix); the SGD residual inside fit() is never clipped
            - Set it to the real lower bound of your data. The defaults assume
              1-5 stars; for 0/1 implicit feedback pass min_rating=0, max_rating=1
            - Example: Movie ratings typically 1-5
            
        max_rating : float, default=5
            Maximum possible rating value
            - Used to clip PREDICTIONS only, exactly like min_rating
            - A too-narrow window does not break training (the gradient ignores it)
              but it does squash everything you read back out of the model
            - Example: Movie ratings typically 1-5
            
        init_mean : float, default=0
            Mean for random initialization of factor matrices
            
        init_std : float, default=0.1
            Standard deviation for random initialization
            - Small values (0.01-0.1) are recommended
            - Too large: Training instability
            
        random_state : int or None, default=None
            Random seed for reproducibility
            - Seeds a PRIVATE np.random.RandomState used for the factor
              initialisation and for the per-epoch shuffle
            - The caller's global numpy RNG is never touched, so fitting a model
              does not change the numbers your surrounding script draws
            
        verbose : int, default=0
            Verbosity level
            - 0: Silent
            - 1: Show epoch progress
            - 2: Show detailed training information
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.init_mean = init_mean
        self.init_std = init_std
        self.random_state = random_state
        self.verbose = verbose
        
        # Will be set during fitting
        self.user_factors_ = None  # U matrix: (n_users, n_factors)
        self.item_factors_ = None  # V matrix: (n_items, n_factors)
        self.user_bias_ = None     # User bias terms
        self.item_bias_ = None     # Item bias terms
        self.global_bias_ = None   # Global mean rating
        self.n_users_ = None
        self.n_items_ = None
        self.training_loss_ = []   # Regularised training curve per epoch (see fit)
        self.training_rmse_ = []   # Plain train RMSE per epoch (no penalty term)
        self.user_rated_items_ = {}  # user_id -> set of item_ids seen in fit()
        
        # User and item mappings (for handling arbitrary IDs)
        self.user_id_map_ = {}
        self.item_id_map_ = {}
        self.user_id_reverse_ = {}
        self.item_id_reverse_ = {}
    
    def _initialize_factors(self, rng):
        """
        Initialize user and item factor matrices with random values
        
        Uses small random values centered at init_mean with init_std variance.
        This helps with gradient descent convergence.

        Parameters:
        -----------
        rng : numpy.random.RandomState
            Private generator owned by fit(). Using it instead of the module-level
            np.random keeps the caller's global RNG state untouched.
        """
        # Initialize factor matrices
        self.user_factors_ = rng.normal(
            self.init_mean, self.init_std, 
            (self.n_users_, self.n_factors)
        )
        
        self.item_factors_ = rng.normal(
            self.init_mean, self.init_std,
            (self.n_items_, self.n_factors)
        )
        
        # Initialize biases
        self.user_bias_ = np.zeros(self.n_users_)
        self.item_bias_ = np.zeros(self.n_items_)
    
    def _create_mappings(self, user_ids, item_ids):
        """
        Create mappings between user/item IDs and matrix indices
        
        This allows the model to work with arbitrary user/item identifiers
        (strings, non-sequential integers, etc.), provided each array is of a
        SINGLE type. np.unique casts to a common dtype, so mixing ints and
        strings in one array turns every ID into a string and the original
        integer IDs no longer look up.

        Parameters:
        -----------
        user_ids : array-like
            Array of user identifiers
        item_ids : array-like
            Array of item identifiers
        """
        unique_users = np.unique(user_ids)
        unique_items = np.unique(item_ids)
        
        self.n_users_ = len(unique_users)
        self.n_items_ = len(unique_items)
        
        # Create forward and reverse mappings
        self.user_id_map_ = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_map_ = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.user_id_reverse_ = {idx: user_id for user_id, idx in self.user_id_map_.items()}
        self.item_id_reverse_ = {idx: item_id for item_id, idx in self.item_id_map_.items()}
    
    def _get_user_idx(self, user_id):
        """Convert user ID to internal index"""
        return self.user_id_map_.get(user_id, None)
    
    def _get_item_idx(self, item_id):
        """Convert item ID to internal index"""
        return self.item_id_map_.get(item_id, None)
    
    def fit(self, user_ids, item_ids, ratings):
        """
        Train the Matrix Factorization model using Stochastic Gradient Descent
        
        Learns user and item factor matrices by minimizing the reconstruction error
        with L2 regularization.
        
        Training curve stored in training_loss_ (divided by n_samples):
        L_report = sum (r_ui - r_hat_ui)^2
            + lambda * (||U||^2 + ||V||^2 + ||b_u||^2 + ||b_i||^2)
        
        where:
        - r_ui: actual rating from user u for item i
        - r_hat_ui: predicted rating mu + b_u + b_i + u . v^T, UNCLIPPED here
        - lambda: regularization parameter

        The bias norms are part of the penalty because the bias updates below
        carry the same -lambda*b shrinkage the factor updates do.

        L_report is a MONITORING quantity, not the exact function this loop
        descends. It charges every parameter's penalty once (one Frobenius norm
        per block), while the updates charge it once per observation. The
        function actually minimised is therefore the class docstring's L, with
        the penalty inside the sum over K. The data term is the same in both and
        the two curves move together, which is all a training curve is read for.

        The printed / stored RMSE is sqrt(SSE / n_samples) and excludes the
        penalty, so two models fitted with different lambda stay comparable.
        
        Parameters:
        -----------
        user_ids : array-like, shape (n_samples,)
            User identifiers for each rating
            Can be any hashable type (int, string, etc.), but ONE type per
            array: _create_mappings runs np.unique, so a mixed [5, 5, 'bob']
            array is coerced to strings, so a later predict([5], ...) silently
            returns the global mean and recommend(5) returns [].

        item_ids : array-like, shape (n_samples,)
            Item identifiers for each rating
            Can be any hashable type (int, string, etc.), one type per array
            for the same reason as user_ids

        ratings : array-like, shape (n_samples,)
            Rating values (typically 1-5 or similar scale)
            
        Returns:
        --------
        self : MatrixFactorization
            Fitted model
        """
        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        ratings = np.array(ratings, dtype=np.float64)
        
        if len(ratings) == 0:
            raise ValueError("fit() needs at least one rating.")
        if not (len(user_ids) == len(item_ids) == len(ratings)):
            raise ValueError(
                f"user_ids, item_ids and ratings must have the same length; "
                f"got {len(user_ids)}, {len(item_ids)}, {len(ratings)}."
            )

        # Create ID mappings
        self._create_mappings(user_ids, item_ids)

        # Fresh training curves for every call, so refitting the same object
        # does not append to the previous run's history
        self.training_loss_ = []
        self.training_rmse_ = []
        
        # Convert IDs to indices
        user_indices = np.array([self.user_id_map_[uid] for uid in user_ids])
        item_indices = np.array([self.item_id_map_[iid] for iid in item_ids])
        
        # Remember what each user rated so recommend() can exclude those items
        # by default (exclude_rated=True) without the caller passing them back in.
        # Keys are taken from user_id_map_ / item_id_map_ rather than from the
        # raw arrays, so this dict and the ID maps always agree on key type.
        self.user_rated_items_ = {}
        for u_idx, i_idx in zip(user_indices, item_indices):
            uid = self.user_id_reverse_[u_idx]
            iid = self.item_id_reverse_[i_idx]
            self.user_rated_items_.setdefault(uid, set()).add(iid)

        # Calculate global bias (mean rating)
        self.global_bias_ = np.mean(ratings)
        
        # One private RNG for the whole fit: reproducible without touching the
        # caller's global numpy random state
        rng = np.random.RandomState(self.random_state)

        # Initialize factor matrices
        self._initialize_factors(rng)
        
        # Training loop
        n_samples = len(ratings)
        
        for epoch in range(self.n_epochs):
            # Shuffle data for SGD (fresh order every epoch, drawn from rng)
            shuffle_idx = rng.permutation(n_samples)
            
            sse = 0.0  # sum of squared errors, the RMSE numerator
            
            # SGD: Update for each rating
            for idx in shuffle_idx:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]
                
                # Predict rating.
                # clip=False is essential: a clipped prediction has ZERO gradient
                # outside [min_rating, max_rating], so the error signal saturates
                # and a factor that has run away can never be pulled back.
                # Koren (2009) clips only when showing a prediction to a user.
                pred = self._predict_pair(u, i, clip=False)
                
                # Calculate error
                error = r - pred
                
                # Update factors using gradient descent on the loss of THIS
                # rating alone,
                #   L_ui = (1/2)*e^2
                #          + (lambda/2)*(||u||^2 + ||v||^2 + b_u^2 + b_i^2),
                # whose gradients are
                #   dL/du_f = -error * v_f + lambda * u_f
                #   dL/dv_f = -error * u_f + lambda * v_f
                # so a step of -learning_rate * gradient gives the "+=" lines below.
                # Summing L_ui over the observed ratings is Koren eq (5): the
                # penalty lands inside the sum, once per observation.
                # (Writing L_ui without the 1/2 puts a 2 in front of every
                #  gradient; that 2 is absorbed into alpha - lambda keeps the
                #  same numeric value - and yields exactly the same update.)
                
                user_factor = self.user_factors_[u].copy()
                
                # Update user factors
                self.user_factors_[u] += self.learning_rate * (
                    error * self.item_factors_[i] - 
                    self.regularization * self.user_factors_[u]
                )
                
                # Update item factors
                self.item_factors_[i] += self.learning_rate * (
                    error * user_factor - 
                    self.regularization * self.item_factors_[i]
                )
                
                # Update biases
                self.user_bias_[u] += self.learning_rate * (
                    error - self.regularization * self.user_bias_[u]
                )
                
                self.item_bias_[i] += self.learning_rate * (
                    error - self.regularization * self.item_bias_[i]
                )
                
                # Accumulate squared error (RMSE numerator, penalty excluded)
                sse += error ** 2
            
            # L2 penalty on the CURRENT parameters, added once per epoch.
            # This is a reporting convention - one Frobenius norm per block, so
            # each parameter is charged once, whereas the updates above charge
            # it once per observation. Never part of RMSE either way.
            reg_loss = self.regularization * (
                np.sum(self.user_factors_ ** 2) + 
                np.sum(self.item_factors_ ** 2) +
                np.sum(self.user_bias_ ** 2) +
                np.sum(self.item_bias_ ** 2)
            )
            epoch_loss = sse + reg_loss
            
            # Track both curves: the regularised report, and plain RMSE
            self.training_loss_.append(epoch_loss / n_samples)
            self.training_rmse_.append(np.sqrt(sse / n_samples))
            
            # Print progress
            if self.verbose > 0 and (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, "
                      f"RMSE: {self.training_rmse_[-1]:.4f}, "
                      f"Reg. loss: {self.training_loss_[-1]:.4f}")
        
        if self.verbose > 0:
            print(f"\nTraining completed!")
            if self.training_rmse_:
                print(f"Final RMSE: {self.training_rmse_[-1]:.4f}")
                print(f"Final reg. loss: {self.training_loss_[-1]:.4f}")
            else:
                print("(n_epochs=0, so no training pass was made)")
        
        return self
    
    def _check_is_fitted(self, method_name):
        """Raise a clear error instead of a cryptic one when fit() was skipped"""
        if self.user_factors_ is None:
            raise ValueError(
                f"This MatrixFactorization instance is not fitted yet. "
                f"Call fit(user_ids, item_ids, ratings) before {method_name}()."
            )

    def _predict_pair(self, user_idx, item_idx, clip=True):
        """
        Predict rating for a user-item pair using internal indices
        
        Prediction formula:
        r_hat_ui = mu + b_u + b_i + u . v^T
        
        where:
        - mu: global mean rating
        - b_u: user bias
        - b_i: item bias
        - u . v^T: dot product of user and item factors
        
        Parameters:
        -----------
        user_idx : int
            Internal user index
        item_idx : int
            Internal item index
        clip : bool, default=True
            Clip the result to [min_rating, max_rating].
            True  -> presentation value, what a user should be shown.
            False -> raw model output, what the SGD residual must be taken
                     against; clipping there would zero out the gradient
                     and let saturated factors diverge.
            
        Returns:
        --------
        prediction : float
            Predicted rating (clipped to the valid range when clip=True)
        """
        pred = (
            self.global_bias_ +
            self.user_bias_[user_idx] +
            self.item_bias_[item_idx] +
            np.dot(self.user_factors_[user_idx], self.item_factors_[item_idx])
        )
        
        if not clip:
            return pred

        # Clip to valid rating range (presentation only)
        return np.clip(pred, self.min_rating, self.max_rating)
    
    def predict(self, user_ids, item_ids):
        """
        Predict ratings for user-item pairs
        
        Parameters:
        -----------
        user_ids : array-like
            User identifiers
            
        item_ids : array-like
            Item identifiers
            
        Either argument may be a single ID; it is broadcast against the other,
        so predict('Alice', all_movies) scores Alice against the whole catalogue.
        Two arrays of different lengths (neither of them length 1) is an error.

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted ratings, clipped to [min_rating, max_rating]
        """
        self._check_is_fitted("predict")

        # Broadcast so a scalar user against many items (or vice versa) works.
        # zip() alone would silently truncate to the shorter of the two.
        user_ids = np.atleast_1d(user_ids)
        item_ids = np.atleast_1d(item_ids)
        if len(user_ids) != len(item_ids) and 1 not in (len(user_ids), len(item_ids)):
            raise ValueError(
                f"user_ids has length {len(user_ids)} but item_ids has length "
                f"{len(item_ids)}; pass equal lengths or a single ID for one of them."
            )
        user_ids, item_ids = np.broadcast_arrays(user_ids, item_ids)
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            user_idx = self._get_user_idx(user_id)
            item_idx = self._get_item_idx(item_id)
            
            # Handle unknown users/items
            if user_idx is None or item_idx is None:
                # Return global mean for unknown users/items
                predictions.append(self.global_bias_)
            else:
                predictions.append(self._predict_pair(user_idx, item_idx))
        
        return np.array(predictions)
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True, 
                 rated_items=None):
        """
        Recommend top N items for a user
        
        Parameters:
        -----------
        user_id : any
            User identifier
            
        n_recommendations : int, default=10
            Number of items to recommend
            
        exclude_rated : bool, default=True
            Whether to exclude items already rated by the user.
            When True and rated_items is None, the items this user rated during
            fit() are used (they were recorded in user_rated_items_), so the
            default really does return a fresh feed.
            
        rated_items : array-like, optional
            Explicit list of items to exclude. Pass it when the user's history
            lives outside the training set (e.g. items seen since the last fit).
            
        Returns:
        --------
        recommendations : list of tuples
            List of (item_id, predicted_rating) sorted by rating (descending).
            The returned rating is clipped to [min_rating, max_rating]; the sort
            uses the UNCLIPPED score so items that all saturate at max_rating are
            still ranked in the model's true order of preference.
        """
        self._check_is_fitted("recommend")
        user_idx = self._get_user_idx(user_id)
        
        if user_idx is None:
            print(f"Warning: User {user_id} not found in training data")
            return []
        
        # Get all items
        all_items = list(self.item_id_map_.keys())
        
        # Exclude rated items if requested. Falling back to the training history
        # is what makes exclude_rated=True honest by default.
        if exclude_rated:
            if rated_items is not None:
                rated_items_set = set(rated_items)
            else:
                rated_items_set = self.user_rated_items_.get(user_id, set())
            all_items = [item for item in all_items if item not in rated_items_set]
        
        # Predict ratings for all items.
        # raw = unclipped score, used for RANKING (ties at max_rating would
        # otherwise be ordered arbitrarily); shown = clipped, what we display.
        predictions = []
        for item_id in all_items:
            item_idx = self._get_item_idx(item_id)
            raw = self._predict_pair(user_idx, item_idx, clip=False)
            shown = np.clip(raw, self.min_rating, self.max_rating)
            predictions.append((item_id, shown, raw))
        
        # Sort by the raw predicted rating (descending)
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        return [(item_id, shown) for item_id, shown, _ in predictions[:n_recommendations]]
    
    def get_similar_items(self, item_id, n_similar=10):
        """
        Find items similar to a given item based on item factors
        
        Similarity is measured using cosine similarity of item factor vectors
        
        Parameters:
        -----------
        item_id : any
            Item identifier
            
        n_similar : int, default=10
            Number of similar items to return
            
        Returns:
        --------
        similar_items : list of tuples
            List of (item_id, similarity_score) sorted by similarity (descending).
            Cosine lives in [-1, 1]: positive means "same taste direction",
            negative means "opposite taste", near zero means "unrelated".
        """
        self._check_is_fitted("get_similar_items")
        item_idx = self._get_item_idx(item_id)
        
        if item_idx is None:
            print(f"Warning: Item {item_id} not found in training data")
            return []
        
        # Get item factor vector
        item_vector = self.item_factors_[item_idx]
        
        # Calculate cosine similarity with all items
        similarities = []
        for other_id, other_idx in self.item_id_map_.items():
            if other_id == item_id:
                continue
            
            other_vector = self.item_factors_[other_idx]
            
            # Cosine similarity
            similarity = np.dot(item_vector, other_vector) / (
                np.linalg.norm(item_vector) * np.linalg.norm(other_vector) + 1e-10
            )
            
            similarities.append((other_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]
    
    def get_similar_users(self, user_id, n_similar=10):
        """
        Find users similar to a given user based on user factors
        
        Similarity is measured using cosine similarity of user factor vectors
        
        Parameters:
        -----------
        user_id : any
            User identifier
            
        n_similar : int, default=10
            Number of similar users to return
            
        Returns:
        --------
        similar_users : list of tuples
            List of (user_id, similarity_score) sorted by similarity (descending).
            Cosine lives in [-1, 1]; negative means the two users pull in
            opposite directions in latent space, not merely "less similar".
        """
        self._check_is_fitted("get_similar_users")
        user_idx = self._get_user_idx(user_id)
        
        if user_idx is None:
            print(f"Warning: User {user_id} not found in training data")
            return []
        
        # Get user factor vector
        user_vector = self.user_factors_[user_idx]
        
        # Calculate cosine similarity with all users
        similarities = []
        for other_id, other_idx in self.user_id_map_.items():
            if other_id == user_id:
                continue
            
            other_vector = self.user_factors_[other_idx]
            
            # Cosine similarity
            similarity = np.dot(user_vector, other_vector) / (
                np.linalg.norm(user_vector) * np.linalg.norm(other_vector) + 1e-10
            )
            
            similarities.append((other_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]
    
    def score(self, user_ids, item_ids, ratings):
        """
        Calculate RMSE (Root Mean Squared Error) on test data
        
        Parameters:
        -----------
        user_ids : array-like
            User identifiers
            
        item_ids : array-like
            Item identifiers
            
        ratings : array-like
            True ratings
            
        Returns:
        --------
        rmse : float
            Root Mean Squared Error. This is an ERROR, not an R^2 score:
            lower is better and 0.0 is perfect.
        """
        self._check_is_fitted("score")
        ratings = np.asarray(ratings, dtype=np.float64)
        if ratings.size == 0:
            raise ValueError("score() needs at least one rating.")
        predictions = self.predict(user_ids, item_ids)
        mse = np.mean((ratings - predictions) ** 2)
        rmse = np.sqrt(mse)
        return rmse
    
    def reconstruct_matrix(self, clip=True):
        """
        Reconstruct the full rating matrix

        R_hat = mu + b_u + b_i + U x V^T, clipped to [min_rating, max_rating]
        when clip=True.

        Parameters:
        -----------
        clip : bool, default=True
            True  -> ratings you could show a user.
            False -> the raw low-rank + bias model. Use this for missing-value
                     imputation, feature extraction, or whenever the data does
                     not live inside [min_rating, max_rating]; clipping there
                     would flatten the very structure you are trying to inspect.
        
        Returns:
        --------
        R : array, shape (n_users_, n_items_)
            Reconstructed rating matrix. Row order follows user_id_reverse_ and
            column order follows item_id_reverse_ (both are sorted-unique order).
        """
        self._check_is_fitted("reconstruct_matrix")

        # Base: global bias + user biases + item biases
        R = np.ones((self.n_users_, self.n_items_)) * self.global_bias_
        R += self.user_bias_[:, np.newaxis]
        R += self.item_bias_[np.newaxis, :]
        
        # Add interaction term: U x V^T
        R += np.dot(self.user_factors_, self.item_factors_.T)

        if not clip:
            return R
        
        # Clip to valid range
        R = np.clip(R, self.min_rating, self.max_rating)
        
        return R


"""
USAGE EXAMPLE 1: Movie Recommendations (Simple)

import numpy as np

# Simulated movie rating data
# Users rate movies on a scale of 1-5
user_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
movie_ids = [0, 1, 2, 0, 2, 3, 1, 2, 4, 0, 3, 4, 1, 3, 4]
ratings = [5, 4, 1, 4, 2, 5, 5, 1, 4, 3, 4, 5, 4, 5, 4]

# Create and train the model
mf = MatrixFactorization(
    n_factors=5,
    learning_rate=0.01,
    regularization=0.02,
    n_epochs=100,
    random_state=42,   # reproducible: same numbers every run
    verbose=1
)

mf.fit(user_ids, movie_ids, ratings)

# Predict rating for user 0, movie 3
user = 0
movie = 3
predicted_rating = mf.predict([user], [movie])[0]
print(f"\nPredicted rating for User {user}, Movie {movie}: {predicted_rating:.2f}")

# One user against the whole catalogue: the single ID is broadcast
all_scores = mf.predict(user, [0, 1, 2, 3, 4])
print(f"User {user} scored against all 5 movies: {np.round(all_scores, 2)}")

# Recommendations for user 0.
# exclude_rated=True is the default and now uses the items seen in fit(),
# so movies 0, 1 and 2 (already rated by user 0) are filtered out and only
# movies 3 and 4 remain - hence "Top 2", not "Top 3".
recommendations = mf.recommend(user_id=0, n_recommendations=3)
print(f"\nTop {len(recommendations)} recommendations for User {user}:")
for movie_id, pred_rating in recommendations:
    print(f"  Movie {movie_id}: {pred_rating:.2f} stars")

# Output shows:
# Training progress: verbose=1 prints "RMSE" (plain train RMSE) and
# "Reg. loss" (the regularised monitoring curve), both falling
# Predicted rating for a specific user-movie pair
# The unrated movies ranked by predicted rating
"""

"""
USAGE EXAMPLE 2: Book Recommendations with String IDs

# Using actual user names and book titles.
# The dataset is deliberately built with TWO taste clusters so the latent
# factors have real structure to find. With only a handful of ratings and no
# structure, the factor vectors are noise and the "similarities" are meaningless.
#
#   Fantasy fans : Alice, Bob, Carol   like Harry Potter / LOTR / Narnia
#   Romance fans : Dave, Eve, Frank    like Twilight / Notebook / Outlander

fantasy = ['Harry Potter', 'LOTR', 'Narnia']
romance = ['Twilight', 'Notebook', 'Outlander']

users, books, ratings = [], [], []
for reader in ['Alice', 'Bob', 'Carol']:      # fantasy readers
    for title in fantasy:
        users.append(reader); books.append(title); ratings.append(5)
    for title in romance:
        users.append(reader); books.append(title); ratings.append(2)
for reader in ['Dave', 'Eve', 'Frank']:       # romance readers
    for title in romance:
        users.append(reader); books.append(title); ratings.append(5)
    for title in fantasy:
        users.append(reader); books.append(title); ratings.append(2)

# Hide three ratings so there is something left to recommend
for hide in [('Alice', 'Narnia'), ('Dave', 'Outlander'), ('Bob', 'LOTR')]:
    k = [j for j in range(len(users))
         if users[j] == hide[0] and books[j] == hide[1]][0]
    users.pop(k); books.pop(k); ratings.pop(k)

# Train model. n_factors=2 is enough for two clusters; more factors on 33
# ratings would just memorise noise.
mf = MatrixFactorization(
    n_factors=2,
    learning_rate=0.02,
    regularization=0.05,
    n_epochs=300,
    random_state=42,
    verbose=1
)

mf.fit(users, books, ratings)

# Recommend books for Alice. exclude_rated=True (the default) drops the books
# Alice already rated during fit(), so only 'Narnia' can come back.
recommendations = mf.recommend(user_id='Alice', n_recommendations=2)

print("\nRecommendations for Alice:")
for book, rating in recommendations:
    print(f"  {book}: {rating:.2f} stars")

# Find similar books to Harry Potter.
# Cosine similarity is in [-1, 1]. POSITIVE = same taste direction,
# NEGATIVE = opposite taste (not merely "less similar"), 0 = unrelated.
# The other fantasy titles should score high positive and the romance
# titles negative, because the two clusters sit on opposite sides of factor 1.
similar_books = mf.get_similar_items('Harry Potter', n_similar=5)
print("\nBooks similar to Harry Potter:")
for book, similarity in similar_books:
    print(f"  {book}: {similarity:+.3f} similarity")

# Find similar users to Alice (the other fantasy readers should win)
similar_users = mf.get_similar_users('Alice', n_similar=5)
print("\nUsers similar to Alice:")
for user, similarity in similar_users:
    print(f"  {user}: {similarity:+.3f} similarity")
"""

"""
USAGE EXAMPLE 3: E-commerce Product Recommendations

import numpy as np

# Customer product ratings (1-5 stars)
np.random.seed(42)

# Generate synthetic e-commerce data
n_customers = 50
n_products = 30
n_ratings = 500

customers = np.random.randint(0, n_customers, n_ratings)
products = np.random.randint(0, n_products, n_ratings)

# Ratings must carry LEARNABLE structure, otherwise the model cannot beat the
# global mean and the demo teaches nothing. Here the signal is a customer
# bias + a product bias + a 2-factor taste interaction; the noise is small.
customer_bias = np.random.randn(n_customers) * 0.8
product_bias = np.random.randn(n_products) * 0.8
customer_taste = np.random.randn(n_customers, 2) * 0.6
product_style = np.random.randn(n_products, 2) * 0.6

interaction = np.sum(customer_taste[customers] * product_style[products], axis=1)
ratings_data = (3.0
                + customer_bias[customers]
                + product_bias[products]
                + interaction
                + np.random.randn(n_ratings) * 0.25)   # small noise
ratings_data = np.clip(ratings_data, 1, 5)

# Split into train and test with a single shuffled permutation.
# (The old `[i for i in range(n) if i not in train_idx]` complement is an
#  O(n*m) linear scan; a permutation is O(n) and cannot overlap by construction.)
train_size = int(0.8 * n_ratings)
perm = np.random.permutation(n_ratings)
train_idx, test_idx = perm[:train_size], perm[train_size:]

train_customers = customers[train_idx]
train_products = products[train_idx]
train_ratings = ratings_data[train_idx]

test_customers = customers[test_idx]
test_products = products[test_idx]
test_ratings = ratings_data[test_idx]

# Train model
mf = MatrixFactorization(
    n_factors=10,
    learning_rate=0.01,
    regularization=0.05,
    n_epochs=50,
    random_state=42,
    verbose=1
)

mf.fit(train_customers, train_products, train_ratings)

# Evaluate on test set, ALWAYS next to a baseline.
# The baseline is "predict the training mean for everything". A model that
# cannot beat it has learned nothing, no matter how good its train RMSE looks.
train_rmse = mf.score(train_customers, train_products, train_ratings)
test_rmse = mf.score(test_customers, test_products, test_ratings)
baseline = np.sqrt(np.mean((test_ratings - train_ratings.mean()) ** 2))
print(f"\nTrain RMSE:             {train_rmse:.4f}")
print(f"Test  RMSE:             {test_rmse:.4f}")
print(f"Global-mean baseline:   {baseline:.4f}  (model must be lower)")

# Recommend products for customer 0.
# exclude_rated=True now falls back to the training history when rated_items
# is omitted; passing it explicitly is still useful when the user's history
# lives outside the training split, as it does here.
recommendations = mf.recommend(
    user_id=0,
    n_recommendations=5,
    exclude_rated=True,
    rated_items=products[customers == 0]
)

print(f"\nTop 5 product recommendations for Customer 0:")
for product_id, pred_rating in recommendations:
    print(f"  Product {product_id}: {pred_rating:.2f} stars")

# Find similar products
if 5 in products:
    similar = mf.get_similar_items(5, n_similar=3)
    print(f"\nProducts similar to Product 5:")
    for product_id, similarity in similar:
        print(f"  Product {product_id}: {similarity:.3f}")
"""

"""
USAGE EXAMPLE 4: Music Streaming Service

# User listening history with ratings
users = []
songs = []
ratings = []

# Simulate user preferences
user_song_data = {
    'User1': [('Pop Song A', 5), ('Pop Song B', 4), ('Rock Song A', 2)],
    'User2': [('Rock Song A', 5), ('Rock Song B', 4), ('Pop Song A', 2)],
    'User3': [('Pop Song A', 5), ('Pop Song B', 5), ('Jazz Song A', 3)],
    'User4': [('Jazz Song A', 5), ('Jazz Song B', 4), ('Classical A', 5)],
    'User5': [('Rock Song A', 4), ('Rock Song B', 5), ('Metal Song A', 4)],
}

for user, songs_ratings in user_song_data.items():
    for song, rating in songs_ratings:
        users.append(user)
        songs.append(song)
        ratings.append(rating)

# Train model
mf = MatrixFactorization(
    n_factors=5,
    learning_rate=0.01,
    regularization=0.01,
    n_epochs=200,
    random_state=42,
    verbose=0
)

mf.fit(users, songs, ratings)

# Create a playlist for User1.
# rated_items is redundant here (these are exactly User1's training items and
# exclude_rated=True would find them anyway) but shows the explicit form.
print("\nPersonalized playlist for User1:")
recommendations = mf.recommend(
    user_id='User1',
    n_recommendations=5,
    rated_items=['Pop Song A', 'Pop Song B', 'Rock Song A']
)

for i, (song, rating) in enumerate(recommendations, 1):
    print(f"{i}. {song} (predicted rating: {rating:.2f})")

# Find similar songs to "Pop Song A"
similar = mf.get_similar_items('Pop Song A', n_similar=3)
print("\nIf you like 'Pop Song A', you might also like:")
for song, similarity in similar:
    print(f"  {song} (similarity: {similarity:.3f})")

# HONESTY NOTE: 15 ratings over 5 users and 9 songs is far too little data for
# 5 latent factors - most songs are rated by exactly one user, so their factor
# vectors are barely more than their initialisation. The similarities printed
# above are largely noise. See USAGE EXAMPLE 2 for a dataset with enough
# overlap for the factors to mean something, and always sanity-check a
# recommender against a global-mean baseline on held-out data (EXAMPLE 3).
"""

"""
USAGE EXAMPLE 5: Restaurant Recommendations

import numpy as np

# Users rate restaurants (1-5 stars)
customers = ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob',
             'Carol', 'Carol', 'Carol', 'Dave', 'Dave', 'Eve', 'Eve']

restaurants = ['Italian A', 'Chinese A', 'Mexican A', 
               'Italian A', 'Italian B', 'French A',
               'Chinese A', 'Chinese B', 'Japanese A',
               'Mexican A', 'Mexican B', 'French A', 'French B']

ratings = [5, 3, 2, 4, 5, 3, 4, 5, 4, 3, 4, 5, 4]

# Train model
mf = MatrixFactorization(
    n_factors=4,
    learning_rate=0.01,
    regularization=0.02,
    n_epochs=150,
    random_state=42,
    verbose=1
)

mf.fit(customers, restaurants, ratings)

# Recommend restaurants for Alice
print("\nRestaurant recommendations for Alice:")
recommendations = mf.recommend(
    user_id='Alice',
    n_recommendations=3,
    rated_items=['Italian A', 'Chinese A', 'Mexican A']
)

for restaurant, rating in recommendations:
    print(f"  {restaurant}: {rating:.2f} stars")

# Find people with similar taste to Alice
similar_users = mf.get_similar_users('Alice', n_similar=2)
print("\nUsers with similar taste to Alice:")
for user, similarity in similar_users:
    print(f"  {user}: {similarity:.3f}")

# Predict Alice's rating for a specific restaurant
pred = mf.predict(['Alice'], ['French A'])[0]
print(f"\nPredicted rating for Alice at French A: {pred:.2f}")
"""

"""
USAGE EXAMPLE 6: Cross-Validation for Hyperparameter Tuning

import numpy as np

# Generate sample data.
# IMPORTANT: the ratings must be a FUNCTION of the user and the item, or the
# tuning loop is comparing three models fitted to pure noise and its "best"
# configuration is meaningless. Here ratings come from a planted rank-3 model
# plus biases plus small noise - exactly what matrix factorization can recover.
np.random.seed(42)
n_samples = 2000
n_users, n_items, k_true = 40, 50, 3
users = np.random.randint(0, n_users, n_samples)
items = np.random.randint(0, n_items, n_samples)

P = np.random.randn(n_users, k_true) * 0.7      # true user factors
Q = np.random.randn(n_items, k_true) * 0.7      # true item factors
b_u = np.random.randn(n_users) * 0.5
b_i = np.random.randn(n_items) * 0.5
true_ratings = (3.0
                + b_u[users] + b_i[items]
                + np.sum(P[users] * Q[items], axis=1)
                + np.random.randn(n_samples) * 0.2)
true_ratings = np.clip(true_ratings, 1, 5)

# Try different hyperparameters
hyperparams = [
    {'n_factors': 5, 'learning_rate': 0.01, 'regularization': 0.01},
    {'n_factors': 10, 'learning_rate': 0.01, 'regularization': 0.02},
    {'n_factors': 20, 'learning_rate': 0.005, 'regularization': 0.05},
]

# Split data with one shuffled permutation (O(n), no overlap by construction)
train_size = int(0.8 * n_samples)
perm = np.random.permutation(n_samples)
train_idx, test_idx = perm[:train_size], perm[train_size:]

train_users = users[train_idx]
train_items = items[train_idx]
train_ratings = true_ratings[train_idx]

test_users = users[test_idx]
test_items = items[test_idx]
test_ratings = true_ratings[test_idx]

# Baseline first: predict the training mean for every test pair.
# Any configuration that cannot beat this number has learned nothing.
baseline_rmse = np.sqrt(np.mean((test_ratings - train_ratings.mean()) ** 2))

# Test each configuration
print("Hyperparameter Tuning Results:")
print("=" * 60)
print(f"Global-mean baseline test RMSE: {baseline_rmse:.4f}")

best_rmse = float('inf')
best_params = None

for params in hyperparams:
    mf = MatrixFactorization(
        n_factors=params['n_factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        n_epochs=50,
        random_state=42,
        verbose=0
    )
    
    mf.fit(train_users, train_items, train_ratings)
    
    train_rmse = mf.score(train_users, train_items, train_ratings)
    test_rmse = mf.score(test_users, test_items, test_ratings)
    
    print(f"\nParams: {params}")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}  "
          f"(baseline {baseline_rmse:.4f}, gap to train {test_rmse - train_rmse:+.4f})")
    
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_params = params

print(f"\n{'='*60}")
print(f"Best parameters: {best_params}")
print(f"Best test RMSE: {best_rmse:.4f} vs baseline {baseline_rmse:.4f}")
print("A large train-test gap means that configuration is memorising;")
print("prefer the one with the lowest TEST RMSE, not the lowest train RMSE.")
"""

"""
USAGE EXAMPLE 7: Visualizing Training Progress (and spotting overfitting)

import numpy as np
# Optional: import matplotlib.pyplot as plt

# Generate PLANTED data of TRUE rank 2. Random labels would let the model
# memorise a big "loss reduction" that means nothing - a falling train loss is
# only evidence of learning when a held-out score falls with it.
np.random.seed(42)
n_users, n_items, n_ratings = 40, 50, 400
users = np.random.randint(0, n_users, n_ratings)
items = np.random.randint(0, n_items, n_ratings)
P = np.random.randn(n_users, 2) * 0.8
Q = np.random.randn(n_items, 2) * 0.8
ratings = np.clip(3.0 + np.sum(P[users] * Q[items], axis=1)
                  + np.random.randn(n_ratings) * 0.3, 1, 5)

# Hold out 25% so train and test can be compared
perm = np.random.permutation(n_ratings)
cut = int(0.75 * n_ratings)
tr, te = perm[:cut], perm[cut:]

# Refit for a growing number of epochs so we can watch train and test move
# together and then apart. n_factors=15 is far above the true rank of 2 and
# regularization is switched off, which is exactly how you provoke overfitting.
test_curve = []
for n_ep in range(20, 301, 20):
    mf = MatrixFactorization(n_factors=15, learning_rate=0.01,
                             regularization=0.0, n_epochs=n_ep,
                             random_state=42, verbose=0)
    mf.fit(users[tr], items[tr], ratings[tr])
    test_curve.append((n_ep,
                       mf.training_rmse_[-1],
                       mf.score(users[te], items[te], ratings[te])))

print("epochs | train RMSE | test RMSE")
for n_ep, tr_rmse, te_rmse in test_curve:
    print(f"{n_ep:6d} | {tr_rmse:10.4f} | {te_rmse:9.4f}")

# Plot both curves
# plt.figure(figsize=(10, 6))
# plt.plot(mf.training_rmse_, label='train RMSE (per epoch)')
# plt.plot([e for e, _, _ in test_curve], [t for _, _, t in test_curve],
#          'o-', label='test RMSE')
# plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.legend(); plt.grid(True)
# plt.title('Matrix Factorization Training Progress')
# plt.show()

# training_loss_ holds the REGULARISED LOSS per epoch - a monitoring curve, not
# the exact function the SGD descends (see fit()'s docstring); training_rmse_
# holds plain RMSE (no penalty), which is the comparable one.
print(f"\nInitial train RMSE: {mf.training_rmse_[0]:.4f}")
print(f"Final train RMSE:   {mf.training_rmse_[-1]:.4f}")
print(f"Final reg. loss:    {mf.training_loss_[-1]:.4f}")

best_ep, best_tr, best_te = min(test_curve, key=lambda row: row[2])
last_ep, last_tr, last_te = test_curve[-1]
print(f"\nBest TEST RMSE  {best_te:.4f} at {best_ep} epochs "
      f"(train was {best_tr:.4f} there)")
print(f"At {last_ep} epochs: train {last_tr:.4f} (still falling) "
      f"but test {last_te:.4f} (turned back up)")
print("That turning point is where you should stop - the extra epochs are")
print("buying memorisation, not generalisation. Regularization pushes it later.")
"""

"""
USAGE EXAMPLE 8: Cold Start Problem Handling

# The cold start problem: what happens with new users/items?

import numpy as np

# Train on existing data
train_users = [0, 0, 1, 1, 2, 2]
train_items = [0, 1, 0, 2, 1, 2]
train_ratings = [5, 4, 4, 5, 3, 4]

mf = MatrixFactorization(
    n_factors=3,
    learning_rate=0.01,
    n_epochs=100,
    random_state=42,
    verbose=0
)

mf.fit(train_users, train_items, train_ratings)

# Test with known user-item pair
pred1 = mf.predict([0], [0])[0]
print(f"Known user, known item: {pred1:.2f}")

# Test with unknown user (cold start)
pred2 = mf.predict([999], [0])[0]
print(f"Unknown user, known item: {pred2:.2f} (returns global mean)")

# Test with unknown item (cold start)
pred3 = mf.predict([0], [999])[0]
print(f"Known user, unknown item: {pred3:.2f} (returns global mean)")

print("\nHandling Cold Start:")
print("- For new users: Use global mean or item popularity")
print("- For new items: Use global mean or content-based features")
print("- Hybrid approach: Combine collaborative filtering with content-based")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _28_matrix_factorization.py
    # numpy only, seeded, ASCII-only output, finishes in a few seconds.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # ================================================================
    # DEMO 1 - Known-answer test: can we recover a planted low-rank matrix?
    # ================================================================
    print("=" * 62)
    print("DEMO 1 - Recovering a planted low-rank matrix")
    print("=" * 62)

    # Build R = P Q^T with a TRUE rank of 3, then rescale into the 1-5 star
    # window. Because we know the answer, we can check the model exactly.
    n_users, n_items, k_true = 40, 30, 3
    P_true = np.random.rand(n_users, k_true)
    Q_true = np.random.rand(n_items, k_true)
    R_true = P_true @ Q_true.T
    R_true = 1.0 + 4.0 * (R_true - R_true.min()) / (R_true.max() - R_true.min())

    # Flatten to (user, item, rating) triples, then hide 40% of the cells so
    # the problem is a realistic sparse one.
    uu, ii = np.meshgrid(np.arange(n_users), np.arange(n_items), indexing="ij")
    uu, ii, rr = uu.ravel(), ii.ravel(), R_true.ravel()
    observed = np.random.rand(uu.size) < 0.60
    uu, ii, rr = uu[observed], ii[observed], rr[observed]

    # Shuffled 75/25 split. Slicing without the permutation would hand the test
    # set only the highest user ids, none of which the model has ever seen.
    perm = np.random.permutation(uu.size)
    cut = int(0.75 * uu.size)
    tr, te = perm[:cut], perm[cut:]

    print(f"Matrix        : {n_users} users x {n_items} items, true rank {k_true}")
    print(f"Observed cells: {uu.size} of {n_users * n_items} "
          f"({100.0 * uu.size / (n_users * n_items):.0f}% dense)")
    print(f"Train / Test  : {len(tr)} / {len(te)} ratings")

    mf = MatrixFactorization(
        n_factors=3,          # match the true rank
        learning_rate=0.02,
        regularization=0.02,
        n_epochs=150,
        min_rating=1,         # the data really does live in [1, 5]
        max_rating=5,
        random_state=42,
        verbose=0
    )
    mf.fit(uu[tr], ii[tr], rr[tr])

    train_rmse = mf.score(uu[tr], ii[tr], rr[tr])
    test_rmse = mf.score(uu[te], ii[te], rr[te])
    # Baseline: predict the training mean for everything. A model that cannot
    # beat this has learned nothing, however pretty its training curve looks.
    baseline = np.sqrt(np.mean((rr[te] - rr[tr].mean()) ** 2))
    recon_rmse = np.sqrt(np.mean((mf.reconstruct_matrix(clip=False) - R_true) ** 2))

    print(f"\nTrain RMSE                       : {train_rmse:.4f}")
    print(f"Test  RMSE                       : {test_rmse:.4f}")
    print(f"Global-mean baseline (test)      : {baseline:.4f}   <- model must beat this")
    print(f"Reconstruction RMSE vs planted R : {recon_rmse:.4f}   "
          f"(all {n_users * n_items} cells, {n_users * n_items - uu.size} never seen)")

    print("\nSample test predictions:")
    preds = mf.predict(uu[te], ii[te])
    for j in range(5):
        print(f"  user={uu[te][j]:3d} item={ii[te][j]:3d}  "
              f"true={rr[te][j]:5.2f}  pred={preds[j]:5.2f}")

    # The spec's known-answer test in its purest form: show every cell, drop
    # the penalty, and the factorization should reproduce R almost exactly.
    mf_full = MatrixFactorization(n_factors=3, learning_rate=0.02,
                                  regularization=0.0, n_epochs=200,
                                  random_state=42, verbose=0)
    mf_full.fit(np.repeat(np.arange(n_users), n_items),
                np.tile(np.arange(n_items), n_users),
                R_true.ravel())
    full_rmse = np.sqrt(np.mean((mf_full.reconstruct_matrix(clip=False) - R_true) ** 2))
    print(f"\nKnown-answer check: fit ALL cells with lambda=0 -> "
          f"reconstruction RMSE = {full_rmse:.2e} (target ~0)")

    # ================================================================
    # DEMO 2 - Recommendations with string IDs and two taste clusters
    # ================================================================
    print("\n" + "=" * 62)
    print("DEMO 2 - Movie recommendations with string IDs")
    print("=" * 62)

    action = ["Die Hard", "Mad Max", "John Wick", "Top Gun"]
    romance = ["Notting Hill", "The Notebook", "Love Actually", "Titanic"]
    action_fans = ["Alice", "Bob", "Carol", "Dan"]
    romance_fans = ["Eve", "Frank", "Grace", "Heidi"]

    # A per-movie quality offset so the titles inside a cluster are not clones.
    quality = {"Die Hard": 0.4, "Mad Max": 0.0, "John Wick": 0.2, "Top Gun": -0.4,
               "Notting Hill": -0.3, "The Notebook": 0.3, "Love Actually": 0.0,
               "Titanic": 0.4}

    users, movies, ratings = [], [], []
    for fans, loved, disliked in [(action_fans, action, romance),
                                  (romance_fans, romance, action)]:
        for person in fans:
            for title in loved:
                users.append(person)
                movies.append(title)
                ratings.append(round(min(5.0, 4.6 + quality[title]), 1))
            for title in disliked:
                users.append(person)
                movies.append(title)
                ratings.append(round(max(1.0, 1.8 + quality[title]), 1))

    # Hold three ratings out so there is something genuine left to recommend
    for who, what in [("Alice", "John Wick"), ("Alice", "Top Gun"), ("Eve", "Titanic")]:
        j = [x for x in range(len(users))
             if users[x] == who and movies[x] == what][0]
        users.pop(j)
        movies.pop(j)
        ratings.pop(j)

    print(f"{len(set(users))} users x {len(set(movies))} movies, "
          f"{len(ratings)} ratings")
    print("Two taste clusters: action fans and romance fans.")

    mf2 = MatrixFactorization(
        n_factors=2,          # two clusters need only two directions
        learning_rate=0.02,
        regularization=0.05,
        n_epochs=300,
        min_rating=1,
        max_rating=5,
        random_state=42,
        verbose=0
    )
    mf2.fit(users, movies, ratings)
    print(f"\nTrain RMSE: {mf2.score(users, movies, ratings):.4f}")

    # exclude_rated=True is the default and uses the items seen during fit(),
    # so only the two titles Alice never rated can come back.
    recs = mf2.recommend("Alice", n_recommendations=3)
    print(f"\nTop {len(recs)} recommendations for Alice (an action fan):")
    for title, pred in recs:
        print(f"  {title:15s} predicted {pred:.2f} stars")

    print("\nMovies most similar to 'Die Hard' (cosine of item factors):")
    for title, sim in mf2.get_similar_items("Die Hard", n_similar=4):
        tag = "same taste" if sim > 0 else "OPPOSITE taste"
        print(f"  {title:15s} {sim:+.3f}  ({tag})")
    print("  Note: all action titles point the SAME way in latent space, so")
    print("  their cosines are ~+1. Their quality difference lives in the item")
    print("  bias b_i, not in the factor direction - direction is taste, not rank.")

    print("\nUsers with taste most like Alice:")
    for person, sim in mf2.get_similar_users("Alice", n_similar=3):
        print(f"  {person:8s} {sim:+.3f}")

    held_out = mf2.predict(["Alice"], ["John Wick"])[0]
    print(f"\nAlice vs 'John Wick' (held out, true 4.8): {held_out:.2f} stars")
    cold = mf2.predict(["NewUser"], ["Die Hard"])[0]
    print(f"NewUser vs 'Die Hard'                    : {cold:.2f} stars "
          f"(unknown user -> global mean)")
