# Matrix Factorization from Scratch: A Comprehensive Guide

Welcome to the world of Matrix Factorization! 🎬 In this comprehensive guide, we'll explore one of the most powerful techniques for collaborative filtering and recommender systems. Matrix Factorization powers recommendation engines at Netflix, Amazon, Spotify, and countless other platforms!

## Table of Contents
1. [What is Matrix Factorization?](#what-is-matrix-factorization)
2. [How Matrix Factorization Works](#how-matrix-factorization-works)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [Implementation Details](#implementation-details)
5. [Step-by-Step Example](#step-by-step-example)
6. [Real-World Applications](#real-world-applications)
7. [Understanding the Code](#understanding-the-code)
8. [Model Evaluation](#model-evaluation)

---

## What is Matrix Factorization?

**Matrix Factorization** is a technique that decomposes a matrix R into two (or more) lower-rank matrices U and V such that:

```
R ≈ U × V^T
```

where:
- **R** is the original matrix (e.g., user-item ratings)
- **U** represents user factors (user preferences in latent space)
- **V** represents item factors (item characteristics in latent space)

**Real-world analogy**: 
Think of movie recommendations. You don't need to know every detail about every movie. Instead, you can describe movies using a few key attributes like "how much action," "how romantic," "how scary," etc. Similarly, users have preferences for these attributes. Matrix Factorization discovers these hidden attributes automatically!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Model Type** | Collaborative Filtering |
| **Learning Style** | Unsupervised (learns patterns from data) |
| **Primary Use** | Recommender Systems, Missing Value Imputation |
| **Output** | Predicted ratings, Recommendations |
| **Complexity** | O(k × n × iterations) where k = factors, n = ratings |

### The Core Idea

```
Original Matrix (Sparse):           ≈      User Factors × Item Factors
                                            
User-Item Ratings Matrix                   U (users × k)   V (items × k)
     [5  ?  3  ?  1]                      [0.8, 0.2]     [0.9, 0.1]
     [4  2  ?  1  ?]                      [0.7, 0.3]  ×  [0.3, 0.7]  ^T
     [?  5  4  ?  2]                      [0.2, 0.8]     [0.6, 0.4]
     [1  ?  2  5  ?]                      [0.1, 0.9]     [0.1, 0.9]
                                                         [0.5, 0.5]

         Sparse                    =         Dense Matrices
      (many missing)                      (no missing values)
```

### When to Use Matrix Factorization

**Perfect for**:
- Recommender systems with implicit or explicit feedback
- Sparse matrices with many missing values
- Collaborative filtering (user-based or item-based)
- Dimensionality reduction with interpretable factors
- Finding latent relationships in data

**Examples**:
- 🎬 Movie recommendations (Netflix, IMDb)
- 🛒 Product recommendations (Amazon, eBay)
- 🎵 Music recommendations (Spotify, Apple Music)
- 📚 Book recommendations (Goodreads)
- 🍽️ Restaurant recommendations (Yelp, Google Maps)
- 📰 News article recommendations
- 👔 Fashion recommendations

---

## How Matrix Factorization Works

### The Process

**Step 1: Represent Data as Sparse Matrix**
```
Users rate items (most entries are missing):

        Item1  Item2  Item3  Item4  Item5
User1     5      ?      3      ?      1
User2     4      2      ?      1      ?
User3     ?      5      4      ?      2
User4     1      ?      2      5      ?
```

**Step 2: Initialize Factor Matrices**
```
Randomly initialize U (user factors) and V (item factors):

U (users × k factors):           V (items × k factors):
   Factor1  Factor2                  Factor1  Factor2
U1  0.1     0.3                  I1   0.2     0.4
U2  0.2     0.1                  I2   0.3     0.1
U3  0.4     0.2                  I3   0.1     0.5
U4  0.1     0.5                  I4   0.4     0.1
                                 I5   0.2     0.3
```

**Step 3: Predict Ratings**
```
For User i and Item j:
Predicted Rating = U[i] · V[j]^T + biases

Example: User1, Item2
Prediction = [0.1, 0.3] · [0.3, 0.1] + biases
           = 0.1×0.3 + 0.3×0.1 + biases
           = 0.03 + 0.03 + biases
```

**Step 4: Calculate Error**
```
For each known rating:
Error = Actual Rating - Predicted Rating
```

**Step 5: Update Factors (Gradient Descent)**
```
For each rating (user i, item j, rating r):
1. Calculate prediction error: e = r - r_pred
2. Update user factors: U[i] += α × (e × V[j] - λ × U[i])
3. Update item factors: V[j] += α × (e × U[i] - λ × V[j])

where:
- α = learning rate
- λ = regularization parameter
```

**Step 6: Repeat Until Convergence**
```
Iterate over all ratings multiple times (epochs)
Until the loss stops decreasing significantly
```

### Key Components Explained

**1. Latent Factors**
Hidden features that explain user-item interactions:
```
For movies:
- Factor 1: Action level (0.0 = drama, 1.0 = action)
- Factor 2: Romance level (0.0 = none, 1.0 = romantic)
- Factor 3: Seriousness (0.0 = comedy, 1.0 = serious)

User preferences match these dimensions:
- User likes action → High Factor 1 weight
- User dislikes romance → Low Factor 2 weight
```

**2. Bias Terms**
Account for systematic tendencies:
```
Prediction = Global Mean + User Bias + Item Bias + U·V^T

Example:
- Global mean = 3.5 (average rating across all users/items)
- User bias = -0.5 (this user rates 0.5 lower than average)
- Item bias = +0.8 (this item is rated 0.8 higher than average)
- Interaction = U·V^T (personalized preference)
```

**3. Regularization**
Prevents overfitting by penalizing large factor values:
```
Loss = Σ(actual - predicted)² + λ × (||U||² + ||V||²)

- λ = 0: No regularization (may overfit)
- λ > 0: Penalty for large values (smoother, generalizes better)
```

---

## The Mathematical Foundation

### Problem Formulation

**Objective**: Minimize the reconstruction error with regularization

```
Loss = Σ(r_ui - r̂_ui)² + λ(||U||² + ||V||²)

where:
- r_ui = actual rating from user u for item i
- r̂_ui = predicted rating
- λ = regularization parameter
- ||·||² = sum of squared values (L2 norm)
```

### Prediction Formula

**Full Prediction Model**:
```
r̂_ui = μ + b_u + b_i + U_u · V_i^T

Components:
1. μ = global mean rating (baseline)
2. b_u = user bias (user's tendency to rate high/low)
3. b_i = item bias (item's tendency to be rated high/low)
4. U_u · V_i^T = dot product of user and item factors (personalization)
```

**Example Calculation**:
```
User: Alice, Movie: Inception

μ = 3.5 (global mean)
b_alice = +0.3 (Alice rates slightly higher than average)
b_inception = +0.8 (Inception is rated highly)
U_alice = [0.9, 0.2, 0.7] (Alice's preferences)
V_inception = [0.8, 0.1, 0.6] (Inception's characteristics)

U_alice · V_inception^T = 0.9×0.8 + 0.2×0.1 + 0.7×0.6
                        = 0.72 + 0.02 + 0.42
                        = 1.16

r̂_alice,inception = 3.5 + 0.3 + 0.8 + 1.16 = 5.76
Clipped to [1, 5] → 5.0 stars ⭐⭐⭐⭐⭐
```

### Gradient Descent Update Rules

**For each rating (user u, item i, rating r)**:

```
1. Calculate error:
   e = r - r̂

2. Calculate gradients:
   ∂L/∂U_u = -2e × V_i + 2λ × U_u
   ∂L/∂V_i = -2e × U_u + 2λ × V_i
   ∂L/∂b_u = -2e + 2λ × b_u
   ∂L/∂b_i = -2e + 2λ × b_i

3. Update parameters (SGD):
   U_u ← U_u + α × (e × V_i - λ × U_u)
   V_i ← V_i + α × (e × U_u - λ × V_i)
   b_u ← b_u + α × (e - λ × b_u)
   b_i ← b_i + α × (e - λ × b_i)

where α = learning rate
```

### Matrix Form (for understanding)

```
R_{m×n} ≈ U_{m×k} × V_{n×k}^T

where:
- m = number of users
- n = number of items
- k = number of latent factors (k << m, n)

Dimensionality reduction:
Original: m × n values
Factorized: m×k + n×k = k(m+n) values
Savings: significant when k << min(m,n)
```

### Loss Function Decomposition

```
Total Loss = Reconstruction Error + Regularization Term

L = Σ_{(u,i)∈K} (r_ui - r̂_ui)² + λ × (||U||² + ||V||² + ||b_u||² + ||b_i||²)

where K = set of known ratings
```

### Why It Works

**1. Low-Rank Assumption**
```
Real-world rating matrices have hidden structure:
- Users have preferences (like action movies)
- Items have properties (is an action movie)
- Only need a few factors to capture this structure
```

**2. Collaborative Filtering**
```
Users with similar tastes have similar factor vectors
Items with similar properties have similar factor vectors
Dot product measures compatibility
```

**3. Generalization**
```
By learning latent factors instead of memorizing ratings:
- Captures underlying patterns
- Predicts unseen user-item pairs
- Handles sparse data effectively
```

---

## Implementation Details

### Algorithm: Stochastic Gradient Descent (SGD)

```
Input: Ratings data (user_ids, item_ids, ratings)
Output: User factors U, Item factors V, biases

1. Initialize:
   - Create user/item ID mappings
   - Initialize U, V with small random values
   - Initialize biases with zeros
   - Calculate global mean μ

2. For each epoch:
   a. Shuffle training data
   
   b. For each rating (u, i, r):
      - Predict: r̂ = μ + b_u + b_i + U_u · V_i^T
      - Error: e = r - r̂
      - Update U_u: U_u += α(e × V_i - λ × U_u)
      - Update V_i: V_i += α(e × U_u - λ × V_i)
      - Update b_u: b_u += α(e - λ × b_u)
      - Update b_i: b_i += α(e - λ × b_i)
   
   c. Calculate epoch loss
   
   d. Check convergence

3. Return learned parameters
```

### Key Implementation Decisions

**1. ID Mapping**
```python
# Handle arbitrary user/item identifiers
user_id_map = {user_id: index for index, user_id in enumerate(unique_users)}

# Allows using strings, non-sequential integers, etc.
# "Alice" → 0
# "Bob" → 1
# "Carol" → 2
```

**2. Initialization Strategy**
```python
# Small random values from normal distribution
U = np.random.normal(mean=0, std=0.1, size=(n_users, n_factors))
V = np.random.normal(mean=0, std=0.1, size=(n_items, n_factors))

# Why small values?
# - Helps gradient descent converge
# - Avoids numerical instability
# - Breaks symmetry
```

**3. Prediction Clipping**
```python
# Ensure predictions are in valid range
prediction = np.clip(prediction, min_rating, max_rating)

# Example: For 1-5 star ratings
# If prediction = 5.7 → clip to 5.0
# If prediction = 0.3 → clip to 1.0
```

**4. Handling Unknown Users/Items**
```python
# Cold start problem
if user_id not in user_id_map:
    return global_mean  # Use average rating

# Better approaches:
# - Use item popularity
# - Use demographic information
# - Use content-based features
```

### Computational Complexity

```
Training:
- Time: O(iterations × n_ratings × k)
  where k = number of factors, n_ratings = number of known ratings
- Space: O((n_users + n_items) × k)

Prediction:
- Time: O(k) per user-item pair
- Space: O(1)

For n_users=1000, n_items=10000, k=10, n_ratings=50000:
- Storage: (1000 + 10000) × 10 = 110,000 values
- vs. Full matrix: 1000 × 10000 = 10,000,000 values
- Reduction: ~99% less storage!
```

### Hyperparameter Guidelines

**Number of Factors (k)**
```
- Small (5-10): Fast, less overfitting, less expressive
- Medium (10-50): Good balance for most applications
- Large (50-200): More expressive, needs more data

Typical: k=10-20 for small datasets, k=50-100 for large datasets
```

**Learning Rate (α)**
```
- Too high (>0.1): Unstable, oscillating loss
- Good (0.001-0.01): Steady convergence
- Too low (<0.001): Very slow convergence

Typical: α=0.01
Advanced: Use learning rate scheduling (decrease over time)
```

**Regularization (λ)**
```
- No regularization (λ=0): Overfitting likely
- Light (λ=0.001-0.01): Good for large datasets
- Heavy (λ=0.1-1.0): Good for small datasets

Typical: λ=0.01-0.05
Use cross-validation to tune
```

**Number of Epochs**
```
- Too few (<50): Underfitting
- Good (100-500): Sufficient for convergence
- Too many (>1000): Overfitting risk, wasted time

Typical: 100-200 epochs
Use early stopping based on validation loss
```

---

## Step-by-Step Example

Let's walk through a complete example: Movie recommendations for 3 users and 4 movies.

### Step 1: Data Preparation

```
User-Movie Ratings (1-5 stars):

        Movie1  Movie2  Movie3  Movie4
Alice     5       ?       3       ?
Bob       4       2       ?       1
Carol     ?       5       4       ?

Known ratings:
- Alice likes Movie1 (5★) and Movie3 (3★)
- Bob likes Movie1 (4★), okay with Movie2 (2★), dislikes Movie4 (1★)
- Carol loves Movie2 (5★) and likes Movie3 (4★)
```

### Step 2: Initialize Parameters

```
Number of factors: k = 2

User factors U (3 users × 2 factors):
Alice:  [0.1, 0.2]
Bob:    [0.3, 0.1]
Carol:  [0.2, 0.4]

Item factors V (4 movies × 2 factors):
Movie1: [0.2, 0.1]
Movie2: [0.1, 0.3]
Movie3: [0.3, 0.2]
Movie4: [0.1, 0.1]

Biases (initialized to 0):
User biases: [0, 0, 0]
Item biases: [0, 0, 0, 0]

Global mean μ = (5+3+4+2+1+5+4)/7 = 3.43
```

### Step 3: First Training Iteration

**Rating 1: Alice rates Movie1 as 5**

```
1. Predict:
   r̂ = 3.43 + 0 + 0 + [0.1, 0.2]·[0.2, 0.1]
     = 3.43 + 0.1×0.2 + 0.2×0.1
     = 3.43 + 0.02 + 0.02
     = 3.47

2. Error:
   e = 5 - 3.47 = 1.53

3. Update (α=0.01, λ=0.02):
   
   U_Alice:
   [0.1, 0.2] += 0.01 × (1.53×[0.2,0.1] - 0.02×[0.1,0.2])
   [0.1, 0.2] += 0.01 × ([0.306,0.153] - [0.002,0.004])
   [0.1, 0.2] += [0.00304, 0.00149]
   → [0.103, 0.201]
   
   V_Movie1:
   [0.2, 0.1] += 0.01 × (1.53×[0.1,0.2] - 0.02×[0.2,0.1])
   [0.2, 0.1] += 0.01 × ([0.153,0.306] - [0.004,0.002])
   [0.2, 0.1] += [0.00149, 0.00304]
   → [0.201, 0.103]
```

**After processing all ratings in this epoch:**
```
Factors get refined to better predict known ratings
Loss decreases from initial high value
```

### Step 4: After 100 Epochs

```
Learned User Factors:
Alice:  [0.85, 0.12]  (prefers Factor 1)
Bob:    [0.62, -0.31] (prefers Factor 1, dislikes Factor 2)
Carol:  [0.28, 0.91]  (strongly prefers Factor 2)

Learned Item Factors:
Movie1: [0.91, -0.15] (high Factor 1, low Factor 2)
Movie2: [0.08, 0.87]  (low Factor 1, high Factor 2)
Movie3: [0.58, 0.53]  (balanced)
Movie4: [0.32, -0.71] (medium Factor 1, very low Factor 2)

Learned Biases:
User biases:  [0.35, -0.28, 0.41]
Item biases:  [0.52, 0.38, -0.12, -0.78]

Interpretation:
- Factor 1: Action/Adventure level
- Factor 2: Romance/Drama level

- Alice loves action (high F1), neutral on romance (low F2)
- Bob likes action (medium F1), dislikes romance (negative F2)
- Carol loves romance (very high F2), less into action (low F1)

- Movie1: Action movie (high F1, low F2)
- Movie2: Romance movie (low F1, high F2)
- Movie3: Mixed genre (balanced F1 and F2)
- Movie4: Another genre (different pattern)
```

### Step 5: Make Predictions

**Predict: What will Carol think of Movie1?**

```
r̂_Carol,Movie1 = μ + b_Carol + b_Movie1 + U_Carol · V_Movie1^T

= 3.43 + 0.41 + 0.52 + [0.28, 0.91]·[0.91, -0.15]
= 3.43 + 0.41 + 0.52 + (0.28×0.91 + 0.91×(-0.15))
= 3.43 + 0.41 + 0.52 + (0.255 - 0.137)
= 3.43 + 0.41 + 0.52 + 0.118
= 4.48

Prediction: Carol would rate Movie1 as 4.5 stars ⭐⭐⭐⭐½
```

**Recommend movies for Bob:**

```
Bob's unrated movies: Movie3

r̂_Bob,Movie3 = 3.43 + (-0.28) + (-0.12) + [0.62,-0.31]·[0.58,0.53]
              = 3.43 - 0.28 - 0.12 + (0.360 - 0.164)
              = 3.43 - 0.28 - 0.12 + 0.196
              = 3.23

Recommendation: Bob might rate Movie3 as 3.2 stars ⭐⭐⭐
(Moderate recommendation - mixed genre movie)
```

### Step 6: Find Similar Movies

**Movies similar to Movie1 (action movie)?**

```
Similarity = cosine similarity of item factor vectors

Sim(Movie1, Movie2) = cos([0.91,-0.15], [0.08,0.87])
                    = (0.91×0.08 + (-0.15)×0.87) / (||M1|| × ||M2||)
                    = (0.073 - 0.131) / (0.924 × 0.874)
                    = -0.058 / 0.807
                    = -0.072

Negative similarity → dissimilar movies (action vs romance)

Sim(Movie1, Movie3) = cos([0.91,-0.15], [0.58,0.53])
                    = (0.91×0.58 + (-0.15)×0.53) / (0.924 × 0.785)
                    = (0.528 - 0.080) / 0.725
                    = 0.618

Positive similarity → similar movies!

Conclusion: Movie3 is more similar to Movie1 than Movie2
Makes sense: Movie3 is mixed genre, has some action
```

---

## Real-World Applications

### 1. **Netflix Movie Recommendations**

```python
# Simplified Netflix-style recommender

# Millions of ratings
users = [1, 1, 1, 2, 2, 3, 3, 3, ...]  # User IDs
movies = [101, 203, 405, 101, 203, ...]  # Movie IDs
ratings = [5, 4, 3, 4, 5, 2, 5, 4, ...]  # 1-5 stars

# Train model
mf = MatrixFactorization(
    n_factors=50,  # More factors for complex patterns
    learning_rate=0.005,
    regularization=0.05,
    n_epochs=20,  # Many ratings, converges fast
    verbose=1
)

mf.fit(users, movies, ratings)

# Recommend movies for a user
user_id = 12345
user_watched = [101, 203, 405, ...]  # Movies already watched

recommendations = mf.recommend(
    user_id=user_id,
    n_recommendations=10,
    rated_items=user_watched
)

# Display recommendations
print(f"Top 10 movies for User {user_id}:")
for movie_id, predicted_rating in recommendations:
    movie_title = get_movie_title(movie_id)
    print(f"{movie_title}: {predicted_rating:.1f}⭐")
```

**Real Impact:**
- Netflix Prize: $1 million for 10% improvement in recommendations
- Matrix Factorization-based methods won
- Saves billions in customer retention

### 2. **E-Commerce Product Recommendations**

```python
# Amazon-style product recommender

# Purchase/rating data
customers = ['C001', 'C001', 'C002', 'C002', ...]
products = ['P123', 'P456', 'P123', 'P789', ...]
ratings = [5, 4, 3, 5, ...]  # Implicit: 5=purchased, 1=viewed only

mf = MatrixFactorization(
    n_factors=20,
    learning_rate=0.01,
    n_epochs=100,
    min_rating=1,
    max_rating=5
)

mf.fit(customers, products, ratings)

# "Customers who bought this also bought..."
similar_products = mf.get_similar_items('P123', n_similar=5)

print("Customers who bought Product P123 also liked:")
for product_id, similarity in similar_products:
    print(f"  {get_product_name(product_id)}: {similarity:.2f}")

# Personalized homepage
recommendations = mf.recommend('C001', n_recommendations=20)
```

**Business Value:**
- 35% of Amazon's revenue from recommendations
- Increases average order value
- Improves customer satisfaction

### 3. **Spotify Music Recommendations**

```python
# Music streaming recommender

# Listening data (implicit feedback)
users = ['U1', 'U1', 'U1', 'U2', 'U2', ...]
songs = ['S001', 'S002', 'S003', 'S001', 'S004', ...]
# Rating proxy: 5=completed, 4=80% listened, 3=50%, etc.
implicit_ratings = [5, 4, 3, 5, 2, ...]

mf = MatrixFactorization(
    n_factors=30,  # Capture diverse music tastes
    learning_rate=0.01,
    n_epochs=100
)

mf.fit(users, songs, implicit_ratings)

# Create personalized playlist
user = 'U1'
playlist = mf.recommend(
    user_id=user,
    n_recommendations=30,
    rated_items=get_user_history(user)
)

# Find similar artists/songs
similar_songs = mf.get_similar_items('S001', n_similar=20)

# Discover users with similar taste
similar_users = mf.get_similar_users('U1', n_similar=10)
```

**Key Features:**
- Discover Weekly: Personalized playlists
- Radio: Similar song recommendations
- Daily Mix: Genre-based personalization

### 4. **Content Platform Recommendations (YouTube, Medium)**

```python
# Article/video recommendations

# Engagement data
users = [101, 101, 102, 102, 103, ...]
articles = ['A1', 'A2', 'A3', 'A1', 'A4', ...]
# Rating: 5=finished+liked, 4=finished, 3=half-read, 2=clicked, 1=shown
engagement = [5, 3, 4, 5, 2, ...]

mf = MatrixFactorization(
    n_factors=15,
    learning_rate=0.01,
    n_epochs=150
)

mf.fit(users, articles, engagement)

# Personalized feed
def generate_feed(user_id, n_articles=50):
    recommendations = mf.recommend(
        user_id=user_id,
        n_recommendations=n_articles,
        rated_items=get_user_history(user_id)
    )
    return recommendations

# "More like this"
def related_articles(article_id):
    return mf.get_similar_items(article_id, n_similar=10)
```

### 5. **Dating Apps (Tinder, Bumble)**

```python
# Match recommendations based on swipe history

# Swipe data
users = ['User1', 'User1', 'User2', 'User2', ...]
profiles = ['Profile1', 'Profile2', 'Profile1', 'Profile3', ...]
# Rating: 5=super like, 4=like, 1=dislike
swipes = [4, 1, 5, 4, ...]

mf = MatrixFactorization(
    n_factors=10,
    learning_rate=0.01,
    n_epochs=100,
    min_rating=1,
    max_rating=5
)

mf.fit(users, profiles, swipes)

# Recommend potential matches
matches = mf.recommend(
    user_id='User1',
    n_recommendations=20,
    rated_items=get_swiped_profiles('User1')
)

# Find users with similar preferences
similar_users = mf.get_similar_users('User1', n_similar=5)
```

### 6. **Restaurant/Food Delivery Recommendations**

```python
# Restaurant recommendations (Uber Eats, DoorDash)

users = ['U001', 'U001', 'U002', 'U002', ...]
restaurants = ['R1', 'R2', 'R1', 'R3', ...]
ratings = [5, 3, 4, 5, ...]  # Order + rating

mf = MatrixFactorization(
    n_factors=12,
    learning_rate=0.01,
    n_epochs=100
)

mf.fit(users, restaurants, ratings)

# Lunch recommendations
lunch_recs = mf.recommend('U001', n_recommendations=10)

# "Similar restaurants"
similar = mf.get_similar_items('R1', n_similar=5)
```

### 7. **Job Recommendations (LinkedIn, Indeed)**

```python
# Match candidates to jobs

candidates = ['C1', 'C1', 'C2', 'C2', ...]
jobs = ['J001', 'J002', 'J001', 'J003', ...]
# Rating: 5=applied, 4=saved, 3=clicked, 2=viewed, 1=shown
interest = [5, 3, 4, 2, ...]

mf = MatrixFactorization(
    n_factors=20,
    learning_rate=0.01,
    n_epochs=100
)

mf.fit(candidates, jobs, interest)

# Recommend jobs to candidate
job_recs = mf.recommend('C1', n_recommendations=20)

# Find similar candidates (for recruiters)
similar_candidates = mf.get_similar_users('C1', n_similar=10)
```

### 8. **News Feed Personalization (Facebook, Twitter)**

```python
# Personalize content feed

users = ['U1', 'U1', 'U2', 'U2', ...]
posts = ['P001', 'P002', 'P003', 'P001', ...]
# Rating based on engagement (likes, shares, comments, time spent)
engagement = [5, 2, 4, 3, ...]

mf = MatrixFactorization(
    n_factors=25,
    learning_rate=0.01,
    n_epochs=100
)

mf.fit(users, posts, engagement)

# Generate personalized feed
feed = mf.recommend('U1', n_recommendations=100)

# Content diversity: Mix with other signals
# - Friend posts
# - Trending topics
# - Diversity of sources
```

---

## Understanding the Code

Let's break down the key parts of our implementation.

### 1. **Initialization**

```python
def __init__(self, n_factors=10, learning_rate=0.01, ...):
    self.n_factors = n_factors
    self.learning_rate = learning_rate
    # ... other parameters
    
    # These will be learned
    self.user_factors_ = None
    self.item_factors_ = None
    self.user_bias_ = None
    self.item_bias_ = None
```

**What's happening:**
- Set hyperparameters
- Initialize placeholders for learned parameters
- Use trailing underscore (_) for learned attributes (scikit-learn convention)

### 2. **ID Mapping**

```python
def _create_mappings(self, user_ids, item_ids):
    unique_users = np.unique(user_ids)
    unique_items = np.unique(item_ids)
    
    self.user_id_map_ = {user_id: idx for idx, user_id in enumerate(unique_users)}
    self.item_id_map_ = {item_id: idx for idx, item_id in enumerate(unique_items)}
```

**Why this matters:**
```
Input IDs can be anything:
- Strings: ['Alice', 'Bob', 'Carol']
- Non-sequential: [1001, 2050, 3017]
- Mixed types: ['User_1', 'User_2', ...]

Internal representation uses sequential indices:
- Allows efficient NumPy array indexing
- Maps: 'Alice' → 0, 'Bob' → 1, 'Carol' → 2
```

### 3. **Factor Initialization**

```python
def _initialize_factors(self):
    self.user_factors_ = np.random.normal(
        self.init_mean,  # Usually 0
        self.init_std,   # Usually 0.1
        (self.n_users_, self.n_factors)
    )
```

**Why random initialization?**
```
1. Break symmetry: If all factors start the same, they'll stay the same
2. Small values: Help gradient descent converge
3. Normal distribution: Centered at 0, most values close to 0
```

### 4. **Training Loop (SGD)**

```python
def fit(self, user_ids, item_ids, ratings):
    # ... setup ...
    
    for epoch in range(self.n_epochs):
        # Shuffle data
        shuffle_idx = np.random.permutation(n_samples)
        
        for idx in shuffle_idx:
            u = user_indices[idx]
            i = item_indices[idx]
            r = ratings[idx]
            
            # Predict
            pred = self._predict_pair(u, i)
            
            # Calculate error
            error = r - pred
            
            # Update factors (gradient descent)
            user_factor = self.user_factors_[u].copy()
            
            self.user_factors_[u] += self.learning_rate * (
                error * self.item_factors_[i] - 
                self.regularization * self.user_factors_[u]
            )
            
            self.item_factors_[i] += self.learning_rate * (
                error * user_factor - 
                self.regularization * self.item_factors_[i]
            )
```

**Key points:**
```
1. Stochastic: Update after each rating (not batch)
2. Shuffle: Different order each epoch prevents bias
3. Copy user_factor: Need original value for item update
4. Learning rate: Controls step size
5. Regularization: Prevents overfitting
```

### 5. **Prediction**

```python
def _predict_pair(self, user_idx, item_idx):
    pred = (
        self.global_bias_ +
        self.user_bias_[user_idx] +
        self.item_bias_[item_idx] +
        np.dot(self.user_factors_[user_idx], self.item_factors_[item_idx])
    )
    return np.clip(pred, self.min_rating, self.max_rating)
```

**Components:**
```
1. global_bias_: Average rating (e.g., 3.5)
2. user_bias_: User's tendency (e.g., +0.3 for generous rater)
3. item_bias_: Item's quality (e.g., +0.8 for great movie)
4. dot product: Personalized preference
5. clip: Ensure valid range (e.g., 1-5 stars)
```

### 6. **Recommendations**

```python
def recommend(self, user_id, n_recommendations=10, exclude_rated=True, rated_items=None):
    user_idx = self._get_user_idx(user_id)
    all_items = list(self.item_id_map_.keys())
    
    # Exclude already rated
    if exclude_rated and rated_items is not None:
        all_items = [item for item in all_items if item not in rated_items]
    
    # Predict for all items
    predictions = []
    for item_id in all_items:
        item_idx = self._get_item_idx(item_id)
        pred = self._predict_pair(user_idx, item_idx)
        predictions.append((item_id, pred))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions[:n_recommendations]
```

**Process:**
```
1. Get all items user hasn't rated
2. Predict rating for each
3. Sort by predicted rating (highest first)
4. Return top N
```

### 7. **Similarity Computation**

```python
def get_similar_items(self, item_id, n_similar=10):
    item_vector = self.item_factors_[item_idx]
    
    for other_id, other_idx in self.item_id_map_.items():
        other_vector = self.item_factors_[other_idx]
        
        # Cosine similarity
        similarity = np.dot(item_vector, other_vector) / (
            np.linalg.norm(item_vector) * np.linalg.norm(other_vector)
        )
```

**Cosine Similarity:**
```
Measures angle between vectors (not magnitude)

sim = (A · B) / (||A|| × ||B||)

Range: [-1, 1]
- 1.0: Identical direction (very similar)
- 0.0: Orthogonal (unrelated)
- -1.0: Opposite direction (very different)

Example:
A = [0.9, 0.1] (action movie)
B = [0.8, 0.2] (another action movie)
sim(A,B) = 0.98 (very similar!)

A = [0.9, 0.1] (action movie)
C = [0.1, 0.9] (romance movie)
sim(A,C) = 0.18 (not similar)
```

---

## Model Evaluation

### Evaluation Metrics

**1. RMSE (Root Mean Squared Error)**
```python
def score(self, user_ids, item_ids, ratings):
    predictions = self.predict(user_ids, item_ids)
    mse = np.mean((ratings - predictions) ** 2)
    rmse = np.sqrt(mse)
    return rmse
```

**Interpretation:**
```
RMSE measures average prediction error in rating units

For 1-5 star ratings:
- RMSE = 0.5: Excellent (off by half a star)
- RMSE = 0.8: Good (typical for many systems)
- RMSE = 1.0: Okay (off by one star)
- RMSE > 1.5: Poor

Lower is better!
```

**2. MAE (Mean Absolute Error)**
```python
mae = np.mean(np.abs(ratings - predictions))
```

**Comparison:**
```
RMSE vs MAE:
- RMSE: Penalizes large errors more heavily (squared term)
- MAE: Treats all errors equally (absolute value)

Example:
Errors: [0.5, 0.5, 2.0]
- MAE = (0.5 + 0.5 + 2.0) / 3 = 1.0
- RMSE = sqrt((0.25 + 0.25 + 4.0) / 3) = 1.22

RMSE is higher because it penalizes the large error (2.0) more
```

**3. Precision@K and Recall@K**
```python
def precision_at_k(true_relevant, recommended, k):
    recommended_k = recommended[:k]
    relevant_in_k = len(set(true_relevant) & set(recommended_k))
    return relevant_in_k / k

def recall_at_k(true_relevant, recommended, k):
    recommended_k = recommended[:k]
    relevant_in_k = len(set(true_relevant) & set(recommended_k))
    return relevant_in_k / len(true_relevant)
```

**Example:**
```
True relevant items (user liked): [1, 3, 5, 7, 9]
Recommended items: [1, 2, 3, 4, 5]

Precision@5 = 3/5 = 0.6 (60% of recommendations are relevant)
Recall@5 = 3/5 = 0.6 (60% of relevant items are recommended)

For k=3: [1, 2, 3]
Precision@3 = 2/3 = 0.67
Recall@3 = 2/5 = 0.4
```

### Train-Test Split Strategy

**1. Random Split**
```python
import numpy as np

# 80-20 split
n_ratings = len(ratings)
train_size = int(0.8 * n_ratings)
indices = np.random.permutation(n_ratings)

train_idx = indices[:train_size]
test_idx = indices[train_size:]

train_users = users[train_idx]
train_items = items[train_idx]
train_ratings = ratings[train_idx]

test_users = users[test_idx]
test_items = items[test_idx]
test_ratings = ratings[test_idx]
```

**2. Temporal Split (for time-series data)**
```python
# Train on past, test on future
# Sort by timestamp
sorted_idx = np.argsort(timestamps)

cutoff = int(0.8 * len(timestamps))
train_idx = sorted_idx[:cutoff]
test_idx = sorted_idx[cutoff:]

# More realistic for production systems
```

**3. User-based Split**
```python
# Ensure all users are in training set
unique_users = np.unique(users)

train_users, test_users = [], []
train_items, test_items = [], []
train_ratings, test_ratings = [], []

for user in unique_users:
    user_mask = users == user
    user_indices = np.where(user_mask)[0]
    
    # Split this user's ratings
    n_user_ratings = len(user_indices)
    n_train = int(0.8 * n_user_ratings)
    
    user_train = user_indices[:n_train]
    user_test = user_indices[n_train:]
    
    train_users.extend(users[user_train])
    # ... (extend other lists)
```

### Cross-Validation

```python
def k_fold_cross_validation(users, items, ratings, k=5):
    n_samples = len(ratings)
    fold_size = n_samples // k
    indices = np.random.permutation(n_samples)
    
    scores = []
    
    for fold in range(k):
        # Split data
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < k-1 else n_samples
        
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        # Train model
        mf = MatrixFactorization(n_factors=10, n_epochs=100, verbose=0)
        mf.fit(users[train_idx], items[train_idx], ratings[train_idx])
        
        # Evaluate
        rmse = mf.score(users[test_idx], items[test_idx], ratings[test_idx])
        scores.append(rmse)
        print(f"Fold {fold+1}: RMSE = {rmse:.4f}")
    
    print(f"\nMean RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores

# Usage
scores = k_fold_cross_validation(users, items, ratings, k=5)
```

### Hyperparameter Tuning

```python
def grid_search(users, items, ratings):
    # Parameter grid
    param_grid = {
        'n_factors': [5, 10, 20, 50],
        'learning_rate': [0.001, 0.01, 0.1],
        'regularization': [0.01, 0.05, 0.1]
    }
    
    # Split data
    train_size = int(0.8 * len(ratings))
    train_users = users[:train_size]
    train_items = items[:train_size]
    train_ratings = ratings[:train_size]
    
    val_users = users[train_size:]
    val_items = items[train_size:]
    val_ratings = ratings[train_size:]
    
    best_score = float('inf')
    best_params = None
    
    # Try all combinations
    for n_factors in param_grid['n_factors']:
        for lr in param_grid['learning_rate']:
            for reg in param_grid['regularization']:
                mf = MatrixFactorization(
                    n_factors=n_factors,
                    learning_rate=lr,
                    regularization=reg,
                    n_epochs=50,
                    verbose=0
                )
                
                mf.fit(train_users, train_items, train_ratings)
                score = mf.score(val_users, val_items, val_ratings)
                
                print(f"Factors={n_factors}, LR={lr}, Reg={reg}: RMSE={score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'n_factors': n_factors,
                        'learning_rate': lr,
                        'regularization': reg
                    }
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best RMSE: {best_score:.4f}")
    return best_params

# Usage
best_params = grid_search(users, items, ratings)
```

### Common Issues and Solutions

**1. Overfitting**
```
Symptoms:
- Low training error, high test error
- Model memorizes training data
- Poor generalization

Solutions:
- Increase regularization (λ)
- Reduce number of factors
- Early stopping (monitor validation loss)
- Get more data
```

**2. Underfitting**
```
Symptoms:
- High training error
- Model too simple to capture patterns

Solutions:
- Increase number of factors
- Reduce regularization
- Train for more epochs
- Check data quality
```

**3. Cold Start Problem**
```
Problem:
- New users: No ratings history
- New items: No one has rated them

Solutions:
- Hybrid models: Combine with content-based features
- Use demographics: Age, location, etc.
- Use item metadata: Genre, category, etc.
- Popularity baseline: Recommend popular items
```

**4. Scalability**
```
Problem:
- Large datasets (millions of users/items)
- Training takes too long

Solutions:
- Batch updates instead of SGD
- Alternating Least Squares (ALS)
- Sampling: Train on subset of data
- Distributed computing: Spark, Dask
- Use specialized libraries: Implicit, LightFM
```

**5. Data Sparsity**
```
Problem:
- Most user-item pairs are missing
- 99%+ of matrix is empty

Solutions:
- This is expected! MF handles sparsity well
- Regularization helps
- Don't try to predict all pairs
- Focus on users/items with some data
```

### Model Improvements

**1. Implicit Feedback**
```python
# For binary data (clicked/not clicked)
class ImplicitMF(MatrixFactorization):
    def __init__(self, confidence_weight=40, **kwargs):
        super().__init__(**kwargs)
        self.confidence_weight = confidence_weight
    
    # Modify loss to weight positive examples more
    # Use Alternating Least Squares (ALS) instead of SGD
```

**2. Temporal Dynamics**
```python
# Add time-based factors
class TemporalMF(MatrixFactorization):
    def __init__(self, time_factors=5, **kwargs):
        super().__init__(**kwargs)
        self.time_factors = time_factors
    
    # User and item preferences change over time
    # Add time-dependent bias terms
```

**3. Social Network Integration**
```python
# Incorporate social connections
class SocialMF(MatrixFactorization):
    def __init__(self, social_reg=0.01, **kwargs):
        super().__init__(**kwargs)
        self.social_reg = social_reg
    
    # Add regularization: friends should have similar factors
    # Loss += λ_social × Σ ||U[i] - U[friend_of_i]||²
```

### Performance Tips

```python
# 1. Use early stopping
def fit_with_early_stopping(self, X_train, X_val, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(self.n_epochs):
        # Train
        self._train_epoch(X_train)
        
        # Validate
        val_loss = self.score(*X_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            self._save_model()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                self._load_model()
                break

# 2. Learning rate scheduling
def get_learning_rate(self, epoch):
    # Decay learning rate over time
    return self.learning_rate / (1 + 0.01 * epoch)

# 3. Adaptive regularization
def get_regularization(self, n_user_ratings):
    # Less regularization for users with many ratings
    return self.regularization / np.sqrt(n_user_ratings + 1)
```

---

## Advantages and Limitations

### Advantages ✅

1. **Handles Sparsity**
   - Works well with sparse matrices (99%+ missing values)
   - Doesn't require complete data

2. **Scalable**
   - Efficient for large datasets
   - Linear in number of ratings
   - Can be parallelized

3. **Discovers Latent Patterns**
   - Automatically finds hidden factors
   - No manual feature engineering needed

4. **Personalized Recommendations**
   - Captures individual user preferences
   - Different recommendations for different users

5. **Interpretable (somewhat)**
   - Latent factors can sometimes be interpreted
   - User and item similarities are meaningful

6. **Flexible**
   - Can incorporate biases, temporal effects, etc.
   - Extensible to hybrid models

### Limitations ⚠️

1. **Cold Start Problem**
   - Can't recommend for new users/items
   - Needs at least some data

2. **Popularity Bias**
   - Tends to recommend popular items
   - May miss niche content

3. **Data Sparsity Issues**
   - Performance degrades with extremely sparse data
   - Needs minimum amount of ratings per user/item

4. **Static Model**
   - Doesn't adapt in real-time
   - Needs retraining to incorporate new data

5. **No Content Features**
   - Doesn't use item/user metadata
   - Purely collaborative filtering

6. **Hyperparameter Sensitivity**
   - Performance depends on hyperparameter tuning
   - Need cross-validation

### When to Use Matrix Factorization

**Use MF when:**
- ✅ You have implicit or explicit feedback data
- ✅ Data is sparse (most user-item pairs missing)
- ✅ You want personalized recommendations
- ✅ Scalability is important
- ✅ You have enough data (thousands of ratings minimum)

**Consider alternatives when:**
- ❌ Cold start is a major concern → Use hybrid models
- ❌ Need real-time updates → Use online learning methods
- ❌ Have rich content features → Use content-based filtering
- ❌ Very small dataset → Use simpler methods (popularity, k-NN)

---

## Summary

Matrix Factorization is a powerful technique for collaborative filtering that:

1. **Decomposes** a sparse user-item matrix into lower-rank matrices
2. **Discovers** latent factors that explain user-item interactions
3. **Predicts** missing values (ratings) based on learned patterns
4. **Scales** to millions of users and items
5. **Powers** recommendation systems at major tech companies

**Key Takeaways:**
- Matrix Factorization learns hidden patterns in user-item interactions
- Uses gradient descent to minimize reconstruction error
- Handles sparse data effectively
- Requires hyperparameter tuning for best results
- Suffers from cold start but excellent for personalized recommendations
- Can be extended with biases, temporal effects, and hybrid approaches

**Next Steps:**
- Implement for your own dataset
- Try different number of factors
- Experiment with hyperparameters
- Compare with baseline methods (popularity, k-NN)
- Consider advanced variants (ALS, temporal MF, hybrid models)

---

## Further Reading

**Papers:**
- "Matrix Factorization Techniques for Recommender Systems" - Koren et al. (2009)
- "Collaborative Filtering for Implicit Feedback Datasets" - Hu et al. (2008)
- "BPR: Bayesian Personalized Ranking from Implicit Feedback" - Rendle et al. (2009)

**Books:**
- "Recommender Systems Handbook" - Ricci, Rokach, Shapira (2015)
- "Programming Collective Intelligence" - Toby Segaran (2007)

**Libraries:**
- Surprise: Scikit for recommender systems
- Implicit: Fast Python implementations
- LightFM: Hybrid recommender systems
- TensorFlow Recommenders (TFRS)

**Resources:**
- Coursera: "Recommender Systems" by University of Minnesota
- Fast.ai: "Practical Deep Learning for Coders" (includes RecSys)
- Netflix Prize documentation and papers

---

**Happy Recommending! 🎬📚🎵**

*Remember: Matrix Factorization powers the recommendations you see every day on Netflix, Amazon, Spotify, and more. Now you understand how it works under the hood!*
