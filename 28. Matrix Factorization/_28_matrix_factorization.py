import numpy as np

class MatrixFactorization:
    """
    Matrix Factorization Implementation from Scratch
    
    Matrix Factorization decomposes a matrix R into two lower-rank matrices U and V
    such that R ≈ U × V^T. This technique is fundamental in collaborative filtering
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
            - Used to clip predictions
            - Example: Movie ratings typically 1-5
            
        max_rating : float, default=5
            Maximum possible rating value
            - Used to clip predictions
            - Example: Movie ratings typically 1-5
            
        init_mean : float, default=0
            Mean for random initialization of factor matrices
            
        init_std : float, default=0.1
            Standard deviation for random initialization
            - Small values (0.01-0.1) are recommended
            - Too large: Training instability
            
        random_state : int or None, default=None
            Random seed for reproducibility
            
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
        self.training_loss_ = []   # Track training loss
        
        # User and item mappings (for handling arbitrary IDs)
        self.user_id_map_ = {}
        self.item_id_map_ = {}
        self.user_id_reverse_ = {}
        self.item_id_reverse_ = {}
    
    def _initialize_factors(self):
        """
        Initialize user and item factor matrices with random values
        
        Uses small random values centered at init_mean with init_std variance.
        This helps with gradient descent convergence.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize factor matrices
        self.user_factors_ = np.random.normal(
            self.init_mean, self.init_std, 
            (self.n_users_, self.n_factors)
        )
        
        self.item_factors_ = np.random.normal(
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
        (strings, non-sequential integers, etc.)
        
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
        
        Loss function:
        L = Σ (r_ui - ŕ_ui)² + λ(||U||² + ||V||²)
        
        where:
        - r_ui: actual rating from user u for item i
        - ŕ_ui: predicted rating (u · v^T + biases)
        - λ: regularization parameter
        
        Parameters:
        -----------
        user_ids : array-like, shape (n_samples,)
            User identifiers for each rating
            Can be any hashable type (int, string, etc.)
            
        item_ids : array-like, shape (n_samples,)
            Item identifiers for each rating
            Can be any hashable type (int, string, etc.)
            
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
        
        # Create ID mappings
        self._create_mappings(user_ids, item_ids)
        
        # Convert IDs to indices
        user_indices = np.array([self.user_id_map_[uid] for uid in user_ids])
        item_indices = np.array([self.item_id_map_[iid] for iid in item_ids])
        
        # Calculate global bias (mean rating)
        self.global_bias_ = np.mean(ratings)
        
        # Initialize factor matrices
        self._initialize_factors()
        
        # Training loop
        n_samples = len(ratings)
        
        for epoch in range(self.n_epochs):
            # Shuffle data for SGD
            if self.random_state is not None:
                np.random.seed(self.random_state + epoch)
            shuffle_idx = np.random.permutation(n_samples)
            
            epoch_loss = 0.0
            
            # SGD: Update for each rating
            for idx in shuffle_idx:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]
                
                # Predict rating
                pred = self._predict_pair(u, i)
                
                # Calculate error
                error = r - pred
                
                # Update factors using gradient descent
                # Gradients:
                # ∂L/∂u_f = -2 * error * v_f + 2 * λ * u_f
                # ∂L/∂v_f = -2 * error * u_f + 2 * λ * v_f
                
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
                
                # Accumulate loss
                epoch_loss += error ** 2
            
            # Add regularization to loss
            reg_loss = self.regularization * (
                np.sum(self.user_factors_ ** 2) + 
                np.sum(self.item_factors_ ** 2) +
                np.sum(self.user_bias_ ** 2) +
                np.sum(self.item_bias_ ** 2)
            )
            epoch_loss += reg_loss
            
            # Track loss
            self.training_loss_.append(epoch_loss / n_samples)
            
            # Print progress
            if self.verbose > 0 and (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                rmse = np.sqrt(epoch_loss / n_samples)
                print(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")
        
        if self.verbose > 0:
            print(f"\nTraining completed!")
            print(f"Final RMSE: {np.sqrt(self.training_loss_[-1]):.4f}")
        
        return self
    
    def _predict_pair(self, user_idx, item_idx):
        """
        Predict rating for a user-item pair using internal indices
        
        Prediction formula:
        ŕ_ui = μ + b_u + b_i + u · v^T
        
        where:
        - μ: global mean rating
        - b_u: user bias
        - b_i: item bias
        - u · v^T: dot product of user and item factors
        
        Parameters:
        -----------
        user_idx : int
            Internal user index
        item_idx : int
            Internal item index
            
        Returns:
        --------
        prediction : float
            Predicted rating (clipped to valid range)
        """
        pred = (
            self.global_bias_ +
            self.user_bias_[user_idx] +
            self.item_bias_[item_idx] +
            np.dot(self.user_factors_[user_idx], self.item_factors_[item_idx])
        )
        
        # Clip to valid rating range
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
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted ratings
        """
        user_ids = np.atleast_1d(user_ids)
        item_ids = np.atleast_1d(item_ids)
        
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
            Whether to exclude items already rated by the user
            
        rated_items : array-like, optional
            List of items already rated by the user (if exclude_rated=True)
            
        Returns:
        --------
        recommendations : list of tuples
            List of (item_id, predicted_rating) sorted by rating (descending)
        """
        user_idx = self._get_user_idx(user_id)
        
        if user_idx is None:
            print(f"Warning: User {user_id} not found in training data")
            return []
        
        # Get all items
        all_items = list(self.item_id_map_.keys())
        
        # Exclude rated items if requested
        if exclude_rated and rated_items is not None:
            rated_items_set = set(rated_items)
            all_items = [item for item in all_items if item not in rated_items_set]
        
        # Predict ratings for all items
        predictions = []
        for item_id in all_items:
            item_idx = self._get_item_idx(item_id)
            pred = self._predict_pair(user_idx, item_idx)
            predictions.append((item_id, pred))
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
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
            List of (item_id, similarity_score) sorted by similarity (descending)
        """
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
            List of (user_id, similarity_score) sorted by similarity (descending)
        """
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
            Root Mean Squared Error
        """
        predictions = self.predict(user_ids, item_ids)
        mse = np.mean((ratings - predictions) ** 2)
        rmse = np.sqrt(mse)
        return rmse
    
    def reconstruct_matrix(self):
        """
        Reconstruct the full rating matrix R ≈ U × V^T
        
        Returns:
        --------
        R : array, shape (n_users, n_items)
            Reconstructed rating matrix
        """
        # Base: global bias + user biases + item biases
        R = np.ones((self.n_users_, self.n_items_)) * self.global_bias_
        R += self.user_bias_[:, np.newaxis]
        R += self.item_bias_[np.newaxis, :]
        
        # Add interaction term: U × V^T
        R += np.dot(self.user_factors_, self.item_factors_.T)
        
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
    verbose=1
)

mf.fit(user_ids, movie_ids, ratings)

# Predict rating for user 0, movie 3
user = 0
movie = 3
predicted_rating = mf.predict([user], [movie])[0]
print(f"\nPredicted rating for User {user}, Movie {movie}: {predicted_rating:.2f}")

# Get top 3 recommendations for user 0
recommendations = mf.recommend(user_id=0, n_recommendations=3, 
                               rated_items=[0, 1, 2])
print(f"\nTop 3 recommendations for User {user}:")
for movie_id, pred_rating in recommendations:
    print(f"  Movie {movie_id}: {pred_rating:.2f} stars")

# Output shows:
# Training progress with decreasing RMSE
# Predicted rating for specific user-movie pair
# Top recommended movies for the user
"""

"""
USAGE EXAMPLE 2: Book Recommendations with String IDs

# Using actual user names and book titles
users = ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob', 
         'Carol', 'Carol', 'Dave', 'Dave', 'Dave']
books = ['Harry Potter', 'Twilight', 'LOTR', 'Harry Potter', 
         'LOTR', 'Star Wars', 'Twilight', 'Star Wars',
         'Harry Potter', 'LOTR', 'Star Wars']
ratings = [5, 4, 3, 4, 5, 2, 5, 3, 3, 4, 5]

# Train model
mf = MatrixFactorization(
    n_factors=3,
    learning_rate=0.01,
    n_epochs=200,
    verbose=1
)

mf.fit(users, books, ratings)

# Recommend books for Alice
recommendations = mf.recommend(
    user_id='Alice',
    n_recommendations=2,
    rated_items=['Harry Potter', 'Twilight', 'LOTR']
)

print("\nRecommendations for Alice:")
for book, rating in recommendations:
    print(f"  {book}: {rating:.2f} stars")

# Find similar books to Harry Potter
similar_books = mf.get_similar_items('Harry Potter', n_similar=2)
print("\nBooks similar to Harry Potter:")
for book, similarity in similar_books:
    print(f"  {book}: {similarity:.3f} similarity")

# Find similar users to Alice
similar_users = mf.get_similar_users('Alice', n_similar=2)
print("\nUsers similar to Alice:")
for user, similarity in similar_users:
    print(f"  {user}: {similarity:.3f} similarity")
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
# Ratings with some structure (not completely random)
base_ratings = np.random.randint(1, 6, n_ratings)
# Add customer/product preferences
customer_bias = np.random.randn(n_customers) * 0.5
product_bias = np.random.randn(n_products) * 0.5
ratings_data = base_ratings + customer_bias[customers] + product_bias[products]
ratings_data = np.clip(ratings_data, 1, 5)

# Split into train and test
train_size = int(0.8 * n_ratings)
train_idx = np.random.choice(n_ratings, train_size, replace=False)
test_idx = np.array([i for i in range(n_ratings) if i not in train_idx])

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
    verbose=1
)

mf.fit(train_customers, train_products, train_ratings)

# Evaluate on test set
test_rmse = mf.score(test_customers, test_products, test_ratings)
print(f"\nTest RMSE: {test_rmse:.4f}")

# Recommend products for customer 0
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
    verbose=0
)

mf.fit(users, songs, ratings)

# Create a playlist for User1
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

# Generate sample data
np.random.seed(42)
n_samples = 1000
users = np.random.randint(0, 50, n_samples)
items = np.random.randint(0, 100, n_samples)
true_ratings = np.random.randint(1, 6, n_samples)

# Try different hyperparameters
hyperparams = [
    {'n_factors': 5, 'learning_rate': 0.01, 'regularization': 0.01},
    {'n_factors': 10, 'learning_rate': 0.01, 'regularization': 0.02},
    {'n_factors': 20, 'learning_rate': 0.005, 'regularization': 0.05},
]

# Split data
train_size = int(0.8 * n_samples)
train_idx = np.random.choice(n_samples, train_size, replace=False)
test_idx = np.array([i for i in range(n_samples) if i not in train_idx])

train_users = users[train_idx]
train_items = items[train_idx]
train_ratings = true_ratings[train_idx]

test_users = users[test_idx]
test_items = items[test_idx]
test_ratings = true_ratings[test_idx]

# Test each configuration
print("Hyperparameter Tuning Results:")
print("=" * 60)

best_rmse = float('inf')
best_params = None

for params in hyperparams:
    mf = MatrixFactorization(
        n_factors=params['n_factors'],
        learning_rate=params['learning_rate'],
        regularization=params['regularization'],
        n_epochs=50,
        verbose=0
    )
    
    mf.fit(train_users, train_items, train_ratings)
    
    train_rmse = mf.score(train_users, train_items, train_ratings)
    test_rmse = mf.score(test_users, test_items, test_ratings)
    
    print(f"\nParams: {params}")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_params = params

print(f"\n{'='*60}")
print(f"Best parameters: {best_params}")
print(f"Best test RMSE: {best_rmse:.4f}")
"""

"""
USAGE EXAMPLE 7: Visualizing Training Progress

import numpy as np
# Optional: import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
users = np.random.randint(0, 20, 200)
items = np.random.randint(0, 30, 200)
ratings = np.random.randint(1, 6, 200).astype(float)

# Train model with more epochs
mf = MatrixFactorization(
    n_factors=8,
    learning_rate=0.01,
    regularization=0.02,
    n_epochs=100,
    verbose=1
)

mf.fit(users, items, ratings)

# Plot training loss
# plt.figure(figsize=(10, 6))
# plt.plot(mf.training_loss_)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Matrix Factorization Training Loss')
# plt.grid(True)
# plt.show()

print(f"\nInitial loss: {mf.training_loss_[0]:.4f}")
print(f"Final loss: {mf.training_loss_[-1]:.4f}")
print(f"Loss reduction: {(1 - mf.training_loss_[-1]/mf.training_loss_[0])*100:.2f}%")
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
