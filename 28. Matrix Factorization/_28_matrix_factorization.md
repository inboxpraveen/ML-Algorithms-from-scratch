# Matrix Factorization from Scratch: A Comprehensive Guide

Welcome to the world of Matrix Factorization! 🎬 In this comprehensive guide, we'll explore one of the most powerful techniques for collaborative filtering and recommender systems. Matrix Factorization powers recommendation engines at Netflix, Amazon, Spotify, and countless other platforms!

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [What is Matrix Factorization?](#what-is-matrix-factorization)
3. [How Matrix Factorization Works](#how-matrix-factorization-works)
4. [The Mathematical Foundation](#the-mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Simplification vs. Canonical Matrix Factorization](#simplification-vs-canonical-matrix-factorization)
11. [Advantages and Limitations](#advantages-and-limitations)
12. [Summary](#summary)
13. [Further Reading](#further-reading)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Matrix Factorization from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _28_matrix_factorization.py  (the __main__ block runs this)
# Or copy the MatrixFactorization class from _28_matrix_factorization.py above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the MatrixFactorization class here ----
# class MatrixFactorization: ...

np.random.seed(42)

# ------ DEMO 1: recover a planted low-rank matrix (known-answer test) ------
n_users, n_items = 40, 30
P_true = np.random.rand(n_users, 3)          # true rank = 3
Q_true = np.random.rand(n_items, 3)
R_true = P_true @ Q_true.T
R_true = 1.0 + 4.0 * (R_true - R_true.min()) / (R_true.max() - R_true.min())

# Flatten to (user, item, rating) triples and hide 40% of the cells
uu, ii = np.meshgrid(np.arange(n_users), np.arange(n_items), indexing="ij")
uu, ii, rr = uu.ravel(), ii.ravel(), R_true.ravel()
seen = np.random.rand(uu.size) < 0.60
uu, ii, rr = uu[seen], ii[seen], rr[seen]

# Shuffle BEFORE slicing, or the test set gets only the highest user ids
perm = np.random.permutation(uu.size)
cut = int(0.75 * uu.size)
tr, te = perm[:cut], perm[cut:]

mf = MatrixFactorization(
    n_factors=3,          # match the true rank
    learning_rate=0.02,
    regularization=0.02,
    n_epochs=150,
    min_rating=1, max_rating=5,   # the real range of THIS data
    random_state=42               # private RNG - your global seed is untouched
)
mf.fit(uu[tr], ii[tr], rr[tr])

baseline = np.sqrt(np.mean((rr[te] - rr[tr].mean()) ** 2))
print(f"Train RMSE                  : {mf.score(uu[tr], ii[tr], rr[tr]):.4f}")
print(f"Test  RMSE                  : {mf.score(uu[te], ii[te], rr[te]):.4f}")
print(f"Global-mean baseline (test) : {baseline:.4f}   <- must beat this")

# reconstruct_matrix(clip=False) returns the raw mu + b_u + b_i + U V^T
recon = mf.reconstruct_matrix(clip=False)
print(f"Reconstruction RMSE vs true R: {np.sqrt(np.mean((recon - R_true)**2)):.4f}")

preds = mf.predict(uu[te], ii[te])
for j in range(3):
    print(f"  user={uu[te][j]:3d} item={ii[te][j]:3d}  "
          f"true={rr[te][j]:5.2f}  pred={preds[j]:5.2f}")

# ------ DEMO 2: recommendations with string IDs ------
action = ["Die Hard", "Mad Max", "John Wick", "Top Gun"]
romance = ["Notting Hill", "The Notebook", "Love Actually", "Titanic"]
quality = {"Die Hard": 0.4, "Mad Max": 0.0, "John Wick": 0.2, "Top Gun": -0.4,
           "Notting Hill": -0.3, "The Notebook": 0.3, "Love Actually": 0.0,
           "Titanic": 0.4}

users, movies, ratings = [], [], []
for fans, loved, disliked in [(["Alice", "Bob", "Carol", "Dan"], action, romance),
                              (["Eve", "Frank", "Grace", "Heidi"], romance, action)]:
    for person in fans:
        for t in loved:
            users.append(person); movies.append(t)
            ratings.append(round(min(5.0, 4.6 + quality[t]), 1))
        for t in disliked:
            users.append(person); movies.append(t)
            ratings.append(round(max(1.0, 1.8 + quality[t]), 1))

# Hold out three ratings so there is something left to recommend
for who, what in [("Alice", "John Wick"), ("Alice", "Top Gun"), ("Eve", "Titanic")]:
    j = [x for x in range(len(users)) if users[x] == who and movies[x] == what][0]
    users.pop(j); movies.pop(j); ratings.pop(j)

mf2 = MatrixFactorization(n_factors=2, learning_rate=0.02, regularization=0.05,
                          n_epochs=300, random_state=42)
mf2.fit(users, movies, ratings)
print(f"\nTrain RMSE: {mf2.score(users, movies, ratings):.4f}")

# exclude_rated=True is the DEFAULT and uses the items seen during fit()
for title, pred in mf2.recommend("Alice", n_recommendations=3):
    print(f"  recommend Alice -> {title:12s} {pred:.2f} stars")

for title, sim in mf2.get_similar_items("Die Hard", n_similar=3):
    print(f"  similar to Die Hard: {title:12s} {sim:+.3f}")

print(f"\nAlice vs 'John Wick' (held out, true 4.8): "
      f"{mf2.predict(['Alice'], ['John Wick'])[0]:.2f}")
print(f"NewUser vs 'Die Hard' (cold start)       : "
      f"{mf2.predict(['NewUser'], ['Die Hard'])[0]:.2f}  (global mean)")
```

Expected output:
```
Train RMSE                  : 0.0466
Test  RMSE                  : 0.0964
Global-mean baseline (test) : 0.6487   <- must beat this
Reconstruction RMSE vs true R: 0.0784
  user= 33 item= 15  true= 1.32  pred= 1.33
  user=  2 item= 29  true= 1.16  pred= 1.33
  user=  8 item= 26  true= 3.12  pred= 3.14

Train RMSE: 0.0510
  recommend Alice -> John Wick    4.71 stars
  recommend Alice -> Top Gun      4.14 stars
  similar to Die Hard: John Wick    +1.000
  similar to Die Hard: Mad Max      +1.000
  similar to Die Hard: Top Gun      +1.000

Alice vs 'John Wick' (held out, true 4.8): 4.71
NewUser vs 'Die Hard' (cold start)       : 3.21  (global mean)
```

**How to read those numbers**

- **Test RMSE 0.0964 against a global-mean baseline of 0.6487.** The baseline is what you get by predicting the training average for every pair. Beating it by ~6.7x is the whole claim of the algorithm; a recommender that does *not* beat it has learned nothing, no matter how low its training error.
- **Reconstruction RMSE 0.0784 over all 1200 cells**, 490 of which the model never saw. That is the low-rank assumption paying off: three latent factors carry enough structure to fill in the holes. Run the same fit on *every* cell with `regularization=0.0` and the reconstruction error drops to about `2e-07` -- the planted matrix is recovered essentially exactly.
- **`recommend("Alice", n_recommendations=3)` returns only two movies.** `exclude_rated=True` is the default and drops the six titles Alice rated during `fit()`, leaving just the two that were held out. Always print `len(recommendations)`, not the number you asked for.
- **Cosine +1.000 between all four action titles** is not a bug. Their *taste direction* is identical; what separates them (Die Hard is a better film than Top Gun in this toy world) lives in the item bias `b_i`, not in the factor direction. Cosine similarity measures taste, not quality.
- **`NewUser` gets 3.21, the global mean.** That is the cold-start fallback: with no ratings there is no `p_u` to dot with, so the model can only offer the average.

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
Loss = Σ over observed ratings of [ (actual - predicted)² + λ × (||U_u||² + ||V_i||²) ]

- λ = 0: No regularization (may overfit)
- λ > 0: Penalty for large values (smoother, generalizes better)
```

(Simplified: the biases carry the same penalty and are written out in
[Problem Formulation](#problem-formulation). Note that the penalty is *inside*
the sum — each rating pays for the two rows it touches.)

---

## The Mathematical Foundation

### Problem Formulation

**Objective**: Minimize the reconstruction error with regularization

```
Loss = Σ_{(u,i)∈K} [ (r_ui - r̂_ui)² + λ(||p_u||² + ||q_i||² + b_u² + b_i²) ]

where:
- K = the set of observed (user, item) pairs
- r_ui = actual rating from user u for item i
- r̂_ui = predicted rating
- p_u, q_i = the factor rows of U and V for this user and this item
- λ = regularization parameter
- ||·||² = sum of squared values (L2 norm)
```

This is Eq. (5) of Koren, Bell & Volinsky (2009), and the penalty sits **inside**
the sum deliberately. `fit` applies `-λ·p_u` on *every* rating user u left, so a
user rated `n_u` times is shrunk `n_u` times. Writing the penalty outside the sum
as `λ(||U||² + ||V||² + ||b_u||² + ||b_i||²)` — one Frobenius norm per matrix —
shrinks each row exactly once and is therefore a **different** objective, not a
tidier way to write this one. (Measured on 600 ratings from 20 users and 15
items, mean `n_u = 30`, `n_factors=4`, `learning_rate=0.005`, λ = 0.2, 4000
epochs — long enough to converge: the largest gradient component of the
inside-the-sum loss is 0.023 against a data-term scale of 3.65, while the
outside-the-sum loss still shows 3.53 — 97% of the data scale, nowhere near
stationary. And the ratio 3.53/3.65 = 0.968 is the value the algebra predicts:
with the inside-the-sum gradient at zero, the outside-the-sum one is left holding
(n_u - 1)/n_u = 29/30 = 0.967 of the data term.)

The two **bias norms** belong in the penalty because the code shrinks the biases
with the same `-λ·b` term it applies to the factors (`user_bias_ += α(e - λ·b_u)`
in `fit`). A guide that lists only `||U||² + ||V||²` here is describing a
slightly different model from the one the code fits.

The Frobenius form does appear in one place in the code: the number stored in
`training_loss_`, which sums each squared norm once per epoch. That is a
reporting convention — see [Training curves](#training-curves-training_loss_-and-training_rmse_)
— so read it as a monitoring curve rather than as the objective above.

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

**Where did the factor of 2 go?**
Step 2 lists gradients that all start with `-2e`, but step 3 updates with `+α × e`,
not `+2α × e`. Both are correct, and the code (`_28_matrix_factorization.py`,
inside `fit`) implements step 3 exactly as written. The reconciliation: define the
loss with a leading one-half,

```
L_ui = ½ (r_ui - r̂_ui)² + (λ/2)(||U_u||² + ||V_i||² + b_u² + b_i²)
       └─ the loss of THIS one rating; summing it over the observed
          ratings K gives the objective in Problem Formulation above,
          penalty and all, which is why the penalty is inside that sum

∂L_ui/∂U_u = -e × V_i + λ × U_u        (no 2 anywhere)
U_u ← U_u - α × ∂L_ui/∂U_u = U_u + α × (e × V_i - λ × U_u)
```

and the 2s cancel exactly. If you prefer the loss without the ½, every gradient
carries a 2 and that constant is absorbed into α alone -- halve the learning rate
and you get an identical update, with **λ unchanged**, because the 2 multiplies
the data term and the penalty equally. What you must **not** do is take the
`-2e` gradients from step 2 and plug them into step 3, because that doubles your
effective learning rate.

**One more thing step 3 hides: the order of the two factor updates.**
`V_i ← V_i + α(e × U_u - λ × V_i)` needs the **pre-update** `U_u`. The code keeps
a copy for exactly this reason:

```python
user_factor = self.user_factors_[u].copy()   # snapshot BEFORE U changes

self.user_factors_[u] += self.learning_rate * (
    error * self.item_factors_[i] - self.regularization * self.user_factors_[u])

self.item_factors_[i] += self.learning_rate * (
    error * user_factor - self.regularization * self.item_factors_[i])
#                ^^^^^^^^^^^ the snapshot, not the freshly updated value
```

Drop the `.copy()` and you are doing a half-Gauss-Seidel step instead of the
simultaneous SGD update the equations describe.

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

L = Σ_{(u,i)∈K} [ (r_ui - r̂_ui)²  +  λ × (||p_u||² + ||q_i||² + b_u² + b_i²) ]
                  └── fit term ──┘     └──────── shrinkage term ────────┘

where K = set of known ratings, p_u = U[u], q_i = V[i]
```

Both terms live under the same Σ. Each observed rating contributes its own
squared error *and* its own shrinkage, so a user rated `n_u` times is shrunk
`n_u` times: setting the gradient to zero gives an effective ridge coefficient of
`n_u·λ` on `p_u`, growing in step with that user's data. The alternative,
`λ(||U||² + ||V||² + ...)` outside the Σ, charges each row exactly once and so
leans relatively harder on sparsely-rated users. That is a legitimate objective —
it is just not the one these updates descend.

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

**3. Prediction Clipping — a presentation step, never a training step**
```python
def _predict_pair(self, user_idx, item_idx, clip=True):
    pred = (self.global_bias_ + self.user_bias_[user_idx]
            + self.item_bias_[item_idx]
            + np.dot(self.user_factors_[user_idx], self.item_factors_[item_idx]))
    if not clip:
        return pred                                     # raw score, for the gradient
    return np.clip(pred, self.min_rating, self.max_rating)   # what a user sees

# Example: For 1-5 star ratings
# If prediction = 5.7 → clip to 5.0
# If prediction = 0.3 → clip to 1.0
```

**Why the `clip` switch matters.** The SGD loop calls `self._predict_pair(u, i, clip=False)`.
That is deliberate and it is the single most important line in the file:

```
e_ui = r_ui - clip(r̂_ui)      ← WRONG.  d(clip)/d(r̂) = 0 outside the window,
                                 so the error signal saturates at a constant
                                 and the coupled updates
                                     p ← p + α(e·q - λp),  q ← q + α(e·p - λq)
                                 with a constant e have a positive eigenvalue.
                                 The factors diverge.

e_ui = r_ui - r̂_ui            ← RIGHT. Koren et al. (2009) clip only when a
                                 prediction is shown to a user.
```

Measured on this implementation. 80 random 0/1 "implicit feedback" ratings over
10 users and 8 items, `n_factors=5`, `learning_rate=0.02`, `regularization=0.02`,
left at the defaults `min_rating=1, max_rating=5`. With a clipped residual there
is no single number to quote, only a growth rate — `max|U|` reaches `1.6e+03`
after 100 epochs, `5.1e+07` after 200 and `1.6e+12` after 300 (`training_loss_`
`1.0e+22`), silently, with no exception; train longer and it gets larger. With
the unclipped residual the same three fits settle at `max|U|` of 0.65, 0.85 and
0.89 and a train RMSE of 0.31, 0.29, 0.29.

**Clipping the output hides this, which is what makes it dangerous.** Take a
planted rank-3 40×30 matrix rescaled to `[0, 10]`, every cell observed,
`n_factors=3`, `learning_rate=0.02`, `regularization=0`, 200 epochs, fitted with
the default `[1, 5]` window:

| residual | largest abs. factor value | `reconstruct_matrix()` RMSE | `reconstruct_matrix(clip=False)` RMSE |
|----------|---------------------------|-----------------------------|---------------------------------------|
| clipped (wrong) | 2.5e+27 | 1.7496 | 9.5e+53 |
| unclipped (this code) | 1.13 | 0.5573 | **1.06e-15** |

The clipped-residual run has *already blown up* — the visible `1.7496` is only
what survives `np.clip`. The unclipped run recovers the planted matrix to machine
precision, and its `0.5573` is not model error at all: `clip(R, 1, 5)` differs
from `R` by exactly 0.5573 RMSE, so that is the floor the window imposes on any
model whatever. The clipping window is not a safety net — it is a claim about
your data. Set `min_rating` and `max_rating` to the true range of your ratings.

**4. Reconstructing the full matrix**
```python
R_hat = mf.reconstruct_matrix()            # clipped to [min_rating, max_rating]
R_raw = mf.reconstruct_matrix(clip=False)  # raw  μ + b_u + b_i + U·Vᵀ
```
`reconstruct_matrix()` returns an `(n_users_, n_items_)` array with rows in
`user_id_reverse_` order and columns in `item_id_reverse_` order. Use the default
`clip=True` when the numbers will be shown as star ratings. Use `clip=False` for
**missing-value imputation, feature extraction, or any data whose real range is
not `[min_rating, max_rating]`** — clipping there flattens the very structure you
are trying to inspect. (On a matrix scaled to `[-0.46, 1.00]` fitted with the
default window, the clipped reconstruction is the constant 1.0 in every cell.)

**5. Handling Unknown Users/Items**
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

   b_Alice  (the code updates the biases too - don't skip these):
   0 += 0.01 × (1.53 - 0.02×0) = 0 + 0.0153
   → 0.0153

   b_Movie1:
   0 += 0.01 × (1.53 - 0.02×0) = 0 + 0.0153
   → 0.0153
```

> **Note on the V update**: it uses the *pre-update* `U_Alice = [0.1, 0.2]`, not the
> `[0.103, 0.201]` we just computed. That is why `fit` snapshots `user_factor =
> self.user_factors_[u].copy()` before touching `U`. Hand-simulating with the new
> value would slowly drift away from what the code does.

> **Note on the biases**: both start at 0, so the `-λ×b` shrinkage contributes
> nothing on the very first step and each bias moves by exactly `α×e = 0.0153`.
> Because every rating by Alice pushes `b_Alice` in the direction of her average
> error, the biases converge much faster than the factors and end up absorbing
> "Alice is a generous rater" and "Movie1 is a good film" — leaving the factors
> free to model *taste*, which is the whole point of the `μ + b_u + b_i` split.

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
    min_rating=1, max_rating=5,  # state your real rating range
    random_state=42,             # reproducible runs
    verbose=1
)

mf.fit(users, movies, ratings)

# Recommend movies for a user.
# rated_items is optional: exclude_rated=True already drops everything this
# user rated during fit(). Pass it when the watch history is larger than the
# training split (e.g. titles watched since the last retrain).
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
    print(f"{movie_title}: {predicted_rating:.1f} stars")  # ASCII: cp1252 consoles
                                                           # cannot encode a star glyph
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
Input IDs can be anything, as long as ONE array uses ONE type:
- Strings: ['Alice', 'Bob', 'Carol']
- Non-sequential ints: [1001, 2050, 3017]
- Prefixed strings: ['User_1', 'User_2', ...]
- NOT mixed: np.unique([5, 5, 'bob']) casts to a common dtype, so every ID
  becomes a string; predict([5], ...) then returns the global mean and
  recommend(5) returns [] with a "not found" warning

Internal representation uses sequential indices:
- Allows efficient NumPy array indexing
- Maps: 'Alice' → 0, 'Bob' → 1, 'Carol' → 2
```

### 3. **Factor Initialization**

```python
def _initialize_factors(self, rng):
    self.user_factors_ = rng.normal(
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

**Why `rng` and not `np.random`?** `fit` creates one private generator,
`rng = np.random.RandomState(self.random_state)`, and hands it to
`_initialize_factors` and to the per-epoch shuffle. Calling the module-level
`np.random.seed(...)` inside `fit` instead would reset the **caller's** global
random state as a side effect of training — so every random number the
surrounding script drew afterwards would change just because a model was fitted.
`random_state=42` still gives you byte-identical reproducibility; it just keeps
that reproducibility inside the model.

### 4. **Training Loop (SGD)**

```python
def fit(self, user_ids, item_ids, ratings):
    # ... setup ...
    
    for epoch in range(self.n_epochs):
        # Shuffle data (rng is the model's private RandomState)
        shuffle_idx = rng.permutation(n_samples)
        
        for idx in shuffle_idx:
            u = user_indices[idx]
            i = item_indices[idx]
            r = ratings[idx]
            
            # Predict - UNCLIPPED, so the gradient never saturates
            pred = self._predict_pair(u, i, clip=False)
            
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
6. clip=False: The residual is taken against the RAW score
```

**Two loss curves, not one.** Each epoch appends to both:

```python
self.training_loss_.append(epoch_loss / n_samples)     # SSE + L2 penalty
self.training_rmse_.append(np.sqrt(sse / n_samples))   # SSE only
```

Use `training_rmse_` to judge fit quality and `training_loss_` to see the penalty
term move as well. Reporting `sqrt(objective)` as "RMSE" — the penalty folded
into the number — makes two models with identical fit quality look different
purely because their λ differs, which is exactly how a λ sweep gets misread.

Two honest caveats about `training_loss_`. It is not the exact function the
updates descend: it charges each parameter's penalty once (Frobenius norms),
while the SGD step charges it once per observation, so the objective in
[Problem Formulation](#problem-formulation) is the one being minimised and this
is a monitoring curve alongside it. And neither curve is monotone — SGD is noisy
and both are accumulated *while* the parameters change, so on 600 noisy ratings
(`n_factors=6`, `learning_rate=0.01`, 300 epochs) each one still rises on 74 of
the 299 steps at λ = 0.02, and on 145 at λ = 0.5, even though the run as a whole
descends. Read the trend, not the step.

### 5. **Prediction**

```python
def _predict_pair(self, user_idx, item_idx, clip=True):
    pred = (
        self.global_bias_ +
        self.user_bias_[user_idx] +
        self.item_bias_[item_idx] +
        np.dot(self.user_factors_[user_idx], self.item_factors_[item_idx])
    )
    if not clip:
        return pred
    return np.clip(pred, self.min_rating, self.max_rating)
```

**Components:**
```
1. global_bias_: Average rating (e.g., 3.5)
2. user_bias_: User's tendency (e.g., +0.3 for generous rater)
3. item_bias_: Item's quality (e.g., +0.8 for great movie)
4. dot product: Personalized preference
5. clip: Ensure valid range (e.g., 1-5 stars) - PRESENTATION ONLY
```

This is the line-for-line implementation of the prediction formula from
[The Mathematical Foundation](#the-mathematical-foundation):
`r̂_ui = μ + b_u + b_i + U_u · V_iᵀ`. The public `predict()` wraps it and always
clips; `fit` calls it with `clip=False`. See
[Prediction Clipping](#key-implementation-decisions) for why that split is not optional.

**`predict()` broadcasts.** Either argument may be a single ID:

```python
mf.predict('Alice', all_movie_titles)   # one user vs. the whole catalogue
mf.predict(all_users, 'Titanic')        # everyone vs. one movie
mf.predict(users, items)                # elementwise, equal lengths
```

Internally it calls `np.broadcast_arrays` before zipping. Plain `zip()` would
silently truncate to the shorter sequence — `predict(0, [0,1,2,3,4])` would hand
back a single number instead of five, with no error.

### 6. **Recommendations**

```python
def recommend(self, user_id, n_recommendations=10, exclude_rated=True, rated_items=None):
    user_idx = self._get_user_idx(user_id)
    all_items = list(self.item_id_map_.keys())
    
    # Exclude already rated
    if exclude_rated:
        if rated_items is not None:
            rated_items_set = set(rated_items)
        else:
            rated_items_set = self.user_rated_items_.get(user_id, set())
        all_items = [item for item in all_items if item not in rated_items_set]
    
    # Predict for all items: rank on the raw score, display the clipped one
    predictions = []
    for item_id in all_items:
        item_idx = self._get_item_idx(item_id)
        raw = self._predict_pair(user_idx, item_idx, clip=False)
        shown = np.clip(raw, self.min_rating, self.max_rating)
        predictions.append((item_id, shown, raw))
    
    # Sort by the raw predicted rating
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    return [(i, s) for i, s, _ in predictions[:n_recommendations]]
```

**Process:**
```
1. Get all items user hasn't rated
2. Predict rating for each
3. Sort by predicted rating (highest first)
4. Return top N
```

**Three details worth pausing on:**

1. **`rated_items_set` is a `set`.** The membership test runs once per item in the
   catalogue; `in` on a list is O(n) and on a set is O(1). With a million items
   that is the difference between a page load and a timeout.

2. **`exclude_rated=True` works without `rated_items`.** `fit` records
   `self.user_rated_items_[user_id]` for every training rating, so the default
   really does return a feed of things the user has not seen. (Pass `rated_items`
   explicitly when the user's history lives *outside* the training split.) A
   version that only excludes when `rated_items is not None` advertises a default
   it does not honour, and cheerfully recommends the movie you rated yesterday.

3. **Ranking uses the raw score, display uses the clipped one.** Once several
   items all predict above `max_rating`, their *clipped* values are all exactly
   5.0 and sorting on them puts the top of your feed in arbitrary order. Sorting
   on the unclipped score preserves the model's real preference ordering. This is
   the general rule for recommenders: **rank on the score, show the rating.**

### 7. **Similarity Computation**

```python
def get_similar_items(self, item_id, n_similar=10):
    item_vector = self.item_factors_[item_idx]
    
    for other_id, other_idx in self.item_id_map_.items():
        other_vector = self.item_factors_[other_idx]
        
        # Cosine similarity
        similarity = np.dot(item_vector, other_vector) / (
            np.linalg.norm(item_vector) * np.linalg.norm(other_vector) + 1e-10
        )
```

The `+ 1e-10` is not cosmetic. A **cold item** — one nobody rated, or one whose
factors were regularized to zero — has `||V|| = 0`, and without the epsilon the
division raises a `RuntimeWarning` and returns `nan`, which then sorts
unpredictably. With it, the similarity is a harmless `0.0`.

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
A · B  = 0.9×0.8 + 0.1×0.2 = 0.74
||A||  = sqrt(0.81 + 0.01) = 0.9055
||B||  = sqrt(0.64 + 0.04) = 0.8246
sim(A,B) = 0.74 / (0.9055 × 0.8246) = 0.99 (very similar!)

A = [0.9, 0.1] (action movie)
C = [0.1, 0.9] (romance movie)
A · C  = 0.9×0.1 + 0.1×0.9 = 0.18
||A||  = ||C|| = 0.9055
sim(A,C) = 0.18 / (0.9055 × 0.9055) = 0.22 (not similar)
```

Note that `sim(A,C)` is **0.22**, not 0.18. The `0.18` is the dot product on its
own — the numerator — and forgetting to divide by the norms is the single most
common cosine bug. Always carry the denominator through.

**Reading a negative similarity.** `get_similar_items` returns the list sorted
descending, so the *last* entries can be strongly negative. That is a real
signal, not noise: `-1.0` means the two items sit at opposite ends of the same
latent axis (action vs. romance), and users who love one reliably dislike the
other. It is only meaningless when the factors themselves are meaningless — a
handful of ratings spread over many items leaves the vectors barely distinguishable
from their random initialisation. Before trusting any similarity, check that the
model beats a global-mean baseline on held-out ratings.

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

> **`score()` returns RMSE, not R².** Every other regressor in this repository
> returns R² from `score()`, where *higher* is better and 1.0 is perfect. Here
> the convention of the recommender-systems literature wins: `score()` is an
> **error**, so lower is better and 0.0 is perfect. Do not compare the two.

**Always quote RMSE next to a baseline.** An absolute RMSE means nothing on its own —
it depends entirely on the spread of your ratings. The cheapest honest reference
point is "predict the training mean for every pair":

```python
baseline = np.sqrt(np.mean((test_ratings - train_ratings.mean()) ** 2))
print(f"Model    RMSE: {mf.score(test_users, test_items, test_ratings):.4f}")
print(f"Baseline RMSE: {baseline:.4f}")
```

If the model does not beat the baseline, it has learned nothing — and with
uniformly random ratings it never will, because there is nothing to learn. The
baseline is what catches that.

### Training curves: `training_loss_` and `training_rmse_`

`fit` records two lists, one entry per epoch, and resets both at the start of
every call (so refitting the same object does not append to the previous run):

| Attribute | What it holds | Use it for |
|-----------|---------------|------------|
| `training_rmse_` | `sqrt(SSE / n_samples)` — plain RMSE, no penalty | Judging fit quality; comparable across different λ |
| `training_loss_` | `(SSE + λ·(‖U‖²+‖V‖²+‖b_u‖²+‖b_i‖²)) / n_samples` | Watching the penalty term move alongside the fit |

```python
mf = MatrixFactorization(n_factors=10, n_epochs=100, random_state=42, verbose=1)
mf.fit(train_users, train_items, train_ratings)

print(mf.training_rmse_[0], "->", mf.training_rmse_[-1])   # should fall overall
print(mf.training_loss_[0], "->", mf.training_loss_[-1])   # should fall overall too
```

**`training_loss_` is not the objective, and neither curve is monotone.** The
Frobenius norms in that formula charge each parameter once, while the SGD step
charges λ once *per observation* — so the function actually minimised is the one
in [Problem Formulation](#problem-formulation), and this is a companion diagnostic,
not the objective itself. It is a *good* companion — measured over 200 epochs
(600 ratings, `n_factors=6`, `learning_rate=0.01`) the two track each other at
correlation 0.999 for λ = 0.02, 0.83 for λ = 0.2 and 0.96 for λ = 0.5, and both
fall end to end — so use it, just do not call it the objective. Both lists are
also accumulated *during* an epoch while the parameters are still moving, so
individual steps go up: on 600 noisy ratings
(`n_factors=6`, `learning_rate=0.01`, 300 epochs) each curve rises on 74 of the
299 steps at λ = 0.02 and on 145 at λ = 0.5, while still falling from end to end
(0.9127 → 0.6247 in the first case). Compare the first entry with the last, not
neighbouring pairs.

**A falling training curve is not evidence of learning.** With enough factors a
model will drive `training_rmse_` toward zero on *any* data, including pure noise.
The only curve that can tell you something generalised is a held-out one, measured
by refitting for a growing number of epochs and scoring the test set each time
(see USAGE EXAMPLE 7 in the `.py`). What you are looking for is the epoch where
train keeps falling but test turns back up — that is where you stop.

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
        mf = MatrixFactorization(n_factors=10, n_epochs=100,
                                 random_state=42, verbose=0)
        mf.fit(users[train_idx], items[train_idx], ratings[train_idx])
        
        # Evaluate
        rmse = mf.score(users[test_idx], items[test_idx], ratings[test_idx])
        scores.append(rmse)
        print(f"Fold {fold+1}: RMSE = {rmse:.4f}")
    
    print(f"\nMean RMSE: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
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
    
    # Split data with a SHUFFLED permutation (Strategy 1 above).
    # Slicing users[:train_size] straight off the array is a trap: rating files
    # are conventionally sorted by user id, so every validation user would be
    # unseen, predict() would return the global mean for all of them, and every
    # configuration would score identically.
    n = len(ratings)
    train_size = int(0.8 * n)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:train_size], idx[train_size:]
    
    train_users, train_items, train_ratings = users[train_idx], items[train_idx], ratings[train_idx]
    val_users, val_items, val_ratings = users[val_idx], items[val_idx], ratings[val_idx]
    
    # Always score a baseline first: predict the training mean for everything.
    # A configuration that cannot beat it has learned nothing.
    baseline = np.sqrt(np.mean((val_ratings - train_ratings.mean()) ** 2))
    print(f"Global-mean baseline RMSE: {baseline:.4f}")
    
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
                    random_state=42,   # same init for every configuration
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

**What ALS actually is** (named as the scalable alternative above, so here is the
one paragraph it deserves). The objective
`Σ(r_ui - p_u·q_i)² + λ(‖P‖² + ‖Q‖²)` is *not* convex in `P` and `Q` jointly —
that is why SGD only finds a local optimum. But **hold `Q` fixed and it becomes an
ordinary ridge regression in `P`**, with a closed-form solution, and vice versa.
ALS alternates: solve all users given the items, then all items given the users,
repeat. For one user `u` who rated the item set `I_u`:

```
p_u = (Q_{I_u}ᵀ Q_{I_u} + λI)⁻¹ Q_{I_u}ᵀ r_u        (a k×k solve, k = n_factors)
```

(The `λI` goes with the `λ(‖P‖² + ‖Q‖²)` written just above. The penalty this
file's SGD actually applies sits inside the sum, and its ridge coefficient is
`n_u·λ` — see [Simplification vs. Canonical Matrix Factorization](#simplification-vs-canonical-matrix-factorization).)

Two consequences. First, every user's solve is **independent of every other
user's**, so the whole sweep is embarrassingly parallel — this is why Spark's MLlib
recommender is ALS and not SGD. Second, there is no learning rate to tune and each
sweep is guaranteed not to increase the loss. The costs: a `k×k` inverse per user
per sweep is heavier than an SGD step when the data is sparse, and ALS needs a value
for *every* cell, which is why the implicit-feedback formulation (Hu et al., 2008)
pairs it with confidence weights over the full matrix. **This implementation uses
SGD only** — see [Simplification vs. Canonical Matrix Factorization](#simplification-vs-canonical-matrix-factorization).

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

**6. Ratings That Are Not 1-5 Stars**
```
Problem:
- Your data is 0/1 clicks, 0-10 scores, log play-counts, or z-scores
- The DEFAULT min_rating=1, max_rating=5 is a claim about YOUR data

Symptoms:
- Every prediction comes back pinned at 1.0 or 5.0
- reconstruct_matrix() returns a nearly constant array
- Similarities look plausible but predictions are useless

Solution:
- Pass the real range: MatrixFactorization(min_rating=0, max_rating=1)
- For unbounded data (z-scores), use min_rating=-np.inf, max_rating=np.inf
- Inspect the raw model with reconstruct_matrix(clip=False)
```
Because clipping in this implementation applies to *output only*, a wrong window
no longer destabilises training — but it does squash everything you read back out
of the model. Measured: a planted matrix rescaled to `[-0.46, 1.00]` and fitted
with the default `[1, 5]` window returns a **constant 1.0** from
`reconstruct_matrix()`, while `reconstruct_matrix(clip=False)` on the same fitted
model recovers the structure. Set the window correctly, or read the raw output.

### Model Improvements

**1. Implicit Feedback**
```python
# For binary data (clicked/not clicked).
# NOTE the min_rating/max_rating override - binary data is NOT 1-5 stars,
# and leaving the defaults would clip every prediction into [1, 5].
class ImplicitMF(MatrixFactorization):
    def __init__(self, confidence_weight=40, **kwargs):
        kwargs.setdefault('min_rating', 0)
        kwargs.setdefault('max_rating', 1)
        super().__init__(**kwargs)
        self.confidence_weight = confidence_weight
    
    # Modify loss to weight positive examples more
    # Use Alternating Least Squares (ALS) instead of SGD
```
> These three subclasses are **sketches of directions to extend the model**, not
> working code — each one still needs its `fit` overridden. They are here to show
> what the extension points look like.

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

> **These three blocks are pseudocode sketches**, not methods of the class in
> `_28_matrix_factorization.py`. `_train_epoch`, `_save_model` and `_load_model`
> do **not** exist in this implementation, and `score()` takes three arrays
> (`score(user_ids, item_ids, ratings)`), not a single unpacked tuple. A runnable
> version of early stopping, written against the real API, follows underneath.

```python
# 1. Early stopping - SKETCH (uses helpers this class does not have)
def fit_with_early_stopping(self, X_train, X_val, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(self.n_epochs):
        # Train
        self._train_epoch(X_train)      # not implemented here
        
        # Validate
        val_loss = self.score(*X_val)   # real signature: score(u, i, r)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            self._save_model()          # not implemented here
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                self._load_model()      # not implemented here
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

**Early stopping with the API that actually exists.** `fit` is reproducible for a
given `random_state`, so refitting for a growing number of epochs traces the same
trajectory a single run would, and you can score the validation set at each stop:

```python
def fit_with_early_stopping(train, val, patience=3, step=20, max_epochs=400, **kwargs):
    """train and val are (user_ids, item_ids, ratings) tuples."""
    best_score, best_model, waited = float('inf'), None, 0

    for n_epochs in range(step, max_epochs + 1, step):
        mf = MatrixFactorization(n_epochs=n_epochs, random_state=42, **kwargs)
        mf.fit(*train)
        val_rmse = mf.score(*val)               # score(user_ids, item_ids, ratings)
        print(f"{n_epochs:4d} epochs  train {mf.training_rmse_[-1]:.4f}  "
              f"val {val_rmse:.4f}")

        if val_rmse < best_score:
            best_score, best_model, waited = val_rmse, mf, 0
        else:
            waited += 1
            if waited >= patience:
                print(f"Early stopping: no improvement for {patience} checks")
                break

    return best_model, best_score
```

This costs more compute than resuming a single fit would, but it needs no new
methods and it is exact. If you want the cheap version, snapshot the arrays
yourself — `user_factors_`, `item_factors_`, `user_bias_`, `item_bias_` and
`global_bias_` are the entire model state, so `_save_model` is four `.copy()`
calls and a float.

---

## Simplification vs. Canonical Matrix Factorization

What this implementation *does* implement is the biased matrix-factorization
model of **Koren, Bell & Volinsky (2009)**, "Matrix Factorization Techniques for
Recommender Systems", trained by stochastic gradient descent:

```
r̂_ui = μ + b_u + b_i + p_u · q_i

minimise  Σ_{(u,i)∈K} [ (r_ui - r̂_ui)² + λ(||p_u||² + ||q_i||² + b_u² + b_i²) ]
                        ^ penalty inside the Σ, exactly as in Eq. (5)
```

with the exact SGD updates of Eq. (6) of that paper. That much is faithful. The
following pieces of the canonical literature are **deliberately not implemented**,
so that the file stays readable as a teaching implementation.

### 1. Alternating Least Squares (ALS)

**What canonical does.** Instead of taking gradient steps, ALS alternates two
closed-form ridge solves — all users given the items, then all items given the
users:

```
p_u = (Q_{I_u}ᵀ Q_{I_u} + λI)⁻¹ Q_{I_u}ᵀ r_u        (I_u = items user u rated)
```

Mind the regularization convention here: that plain `λI` is the ridge solve for a
penalty written *outside* the Σ — the form Hu et al. (2008) and most ALS
write-ups quote. Take the derivative of the objective above, where
the penalty sits inside, and you get `(Q_{I_u}ᵀ Q_{I_u} + n_u·λI)⁻¹` instead —
`n_u = |I_u|`. That `n_u·λ` version is Zhou et al.'s (2008) ALS-WR, and it is the
closed form that corresponds to what this file's SGD does. The two give visibly
different vectors, not a rescaling of one another.

**Why this file omits it.** SGD is the shorter and more transparent path from the
loss function to code: one residual, four `+=` lines, no linear algebra to hide
behind. ALS would add a per-user `k×k` solve and obscure the connection to the
gradients printed in [The Mathematical Foundation](#the-mathematical-foundation).

**Practical consequence.** Training does not parallelise across users, so this
implementation is the wrong tool above roughly a million ratings. It also cannot
express the implicit-feedback objective of Hu et al. (2008), which sums over
*every* cell with confidence weights `c_ui = 1 + α·r_ui` — a sum ALS handles in
closed form and SGD cannot enumerate.

### 2. SVD++ (implicit feedback as a second user representation)

**What canonical does.** SVD++ augments the user vector with the set `N(u)` of
items the user *interacted with* at all, rated or not:

```
r̂_ui = μ + b_u + b_i + q_i · ( p_u + |N(u)|^(-1/2) × Σ_{j∈N(u)} y_j )
```

**Why this file omits it.** It needs a second `n_items × k` parameter block `y`
and a gradient that touches every item in `N(u)` on every step — roughly a
doubling of the code in `fit` for a model the reader cannot check by hand.

**Practical consequence.** The fact that a user *chose to rate* an item carries
signal here that goes unused. On the Netflix data this cost Koren et al. about
0.01 RMSE, which mattered a great deal for a $1M prize and matters much less for
learning the technique.

### 3. Temporal dynamics (timeSVD++)

**What canonical does.** Lets the biases and the user vector drift with time:
`b_u(t)`, `b_i(t)`, `p_u(t)`, since a user's standards and an item's popularity
both move over months.

**Why this file omits it.** `fit(user_ids, item_ids, ratings)` has no timestamp
argument, and adding time bins would change the public API.

**Practical consequence.** All ratings are treated as simultaneous. On a dataset
spanning years, the model will average over a genuine drift instead of tracking it.

### 4. Per-user / per-item adaptive regularization

**What canonical does.** Scales λ by the number of observations, so users with
three ratings are shrunk harder than users with three hundred.

**Why this file omits it.** A single scalar `regularization` keeps the objective
in the docstring identical to the objective in the code.

**Practical consequence.** With a single λ tuned for the average user, the
long tail of sparsely-rated users is under-regularized and overfits. The sketch
in [Performance Tips](#performance-tips) (`get_regularization`) shows the shape
of the fix.

### 5. Early stopping inside `fit`

**What canonical does.** Monitors a validation RMSE each epoch and halts when it
stops improving.

**Why this file omits it.** `fit` would need a validation split in its signature.

**Practical consequence.** You must choose `n_epochs` yourself. See the runnable
`fit_with_early_stopping` in [Performance Tips](#performance-tips), and USAGE
EXAMPLE 7 in the `.py` for how to find the turning point empirically.

### What is *not* a simplification

Two things that look like shortcuts but are the canonical behaviour:

- **Clipping is applied to output only.** `fit` calls
  `_predict_pair(u, i, clip=False)`. This matches Koren et al., where clipping to
  the rating scale is a post-processing step. Clipping inside the residual is not
  a "safer" variant — it zeroes the gradient outside the window and lets factors
  diverge without bound.
- **The `-2` in the printed gradients versus the `+α·e` in the update.** Both are
  right; the factor of 2 cancels against the ½ in the loss. See
  [Gradient Descent Update Rules](#gradient-descent-update-rules).

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
