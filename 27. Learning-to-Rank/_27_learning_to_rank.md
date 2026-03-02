# Learning-to-Rank (LambdaRank)

## Overview

**Learning-to-Rank (LTR)** is a machine learning technique specifically designed for ranking problems in information retrieval, search engines, and recommendation systems. Unlike traditional regression or classification that predicts individual scores or labels, LTR learns to order items by their relevance or importance.

### Key Concept

Imagine you're a search engine. When someone searches for "machine learning tutorials", you have thousands of web pages to show. You don't just need to know which pages are good—you need to know the **order** to show them in. The first page needs to be better than the second, the second better than the third, and so on.

Learning-to-Rank solves this by:
1. Learning from examples of "this document should rank higher than that one"
2. Optimizing directly for ranking quality (not prediction accuracy)
3. Considering the relative order of items, not just their scores

Think of it like training a judge to rank contestants in a competition—it's not about giving exact scores, it's about getting the order right!

## When to Use Learning-to-Rank

### Perfect For:
- **Search Engines**: Ranking web pages by relevance to queries
- **Recommendation Systems**: Ordering products, movies, or content by user preference
- **Question Answering**: Ranking candidate answers by correctness
- **Document Retrieval**: Ordering documents by relevance to queries
- **E-commerce**: Ranking products by purchase likelihood
- **Job Matching**: Ranking candidates or job postings
- **Ad Placement**: Ordering ads by click probability and revenue

### When LTR is Better Than Alternatives:
- **vs Regression**: You care about order, not exact scores
- **vs Classification**: You need relative ranking, not just categories
- **vs Simple Scoring**: You have complex features and interactions
- **vs Manual Rules**: You have training data with relevance judgments

## Mathematical Foundation

### 1. The Ranking Problem

**Goal**: Given a query `q` and documents `{d₁, d₂, ..., dₙ}`, learn a function `f(q, d)` that produces scores such that:

```
f(q, d₁) > f(q, d₂)  if  relevance(d₁) > relevance(d₂)
```

**Key Insight**: We don't care about the exact scores, only that more relevant documents get higher scores!

### 2. Three Approaches to Learning-to-Rank

#### a) Pointwise Approach
Treat each query-document pair independently, predict relevance score.

**Problem**: Ignores relative ordering between documents!

#### b) Pairwise Approach (Our Implementation)
Learn from pairs of documents: "document A should rank higher than document B"

**Advantage**: Directly models relative preferences!

#### c) Listwise Approach
Optimize for the entire ranked list at once.

**Advantage**: Most direct, but computationally expensive.

Our implementation uses **LambdaRank**, a pairwise approach with gradients based on listwise metrics!

### 3. NDCG - The Ranking Metric

**NDCG (Normalized Discounted Cumulative Gain)** measures ranking quality.

#### Discounted Cumulative Gain (DCG)

DCG gives more weight to:
1. Highly relevant documents (2^relevance)
2. Documents appearing earlier (1/log₂(position))

```
DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)
```

**Example**: For rankings [3, 2, 0, 1] (relevances)

```
DCG = (2³-1)/log₂(2) + (2²-1)/log₂(3) + (2⁰-1)/log₂(4) + (2¹-1)/log₂(5)
    = 7/1 + 3/1.585 + 0/2 + 1/2.322
    = 7 + 1.893 + 0 + 0.431
    = 9.324
```

#### Normalized DCG (NDCG)

Normalize by the ideal DCG (perfect ranking):

```
NDCG@k = DCG@k / IDCG@k
```

Where IDCG = DCG of the ideal ranking (sorted by true relevance)

**Range**: [0, 1] where 1.0 = perfect ranking

### 4. LambdaRank Gradients

This is where the magic happens! 🎯

**Traditional gradient descent**: Optimize loss function directly

```
∂Loss/∂score
```

**LambdaRank**: Use "lambda" gradients that directly optimize ranking metrics

```
λᵢⱼ = |ΔNDCG| × σ'(Δscore)
```

Where:
- `|ΔNDCG|` = Change in NDCG if we swap documents i and j
- `σ'(Δscore)` = Derivative of sigmoid function
- `Δscore` = Score difference between documents

**Key Idea**: 
- If a higher-relevance document has a lower score (wrong order), push it up
- Weight the push by how much swapping them would improve NDCG
- Larger NDCG improvement → stronger push!

### 5. The LambdaRank Algorithm

For each query, for each pair of documents (i, j):

```python
# 1. Check if order is wrong
if relevance[i] > relevance[j] and score[i] < score[j]:
    
    # 2. Compute NDCG improvement from swapping
    current_NDCG = compute_NDCG(relevances, scores)
    swapped_NDCG = compute_NDCG(relevances, swap(scores, i, j))
    delta_NDCG = |swapped_NDCG - current_NDCG|
    
    # 3. Compute sigmoid derivative
    score_diff = score[j] - score[i]
    sigmoid = 1 / (1 + exp(-score_diff))
    sigmoid_derivative = sigmoid × (1 - sigmoid)
    
    # 4. Compute lambda gradient
    lambda_ij = delta_NDCG × sigmoid_derivative
    
    # 5. Update gradients
    gradient[i] += lambda_ij   # Push i up
    gradient[j] -= lambda_ij   # Push j down
```

### 6. Gradient Boosting Framework

LambdaRank uses gradient boosting with lambda gradients:

```
1. Initialize: F₀(x) = baseline_score

2. For t = 1 to n_estimators:
   a. Compute lambda gradients for all documents
   b. Fit regression tree to gradients
   c. Update predictions: Fₜ(x) = Fₜ₋₁(x) + learning_rate × tree_t(x)

3. Final prediction: F(x) = F₀(x) + lr × Σ tree_t(x)
```

## Algorithm Steps

### Step 1: Data Preparation

Organize data into query-document pairs:

```python
# Each row is a query-document pair
X = [
    [pagerank, query_match, freshness, ...],  # Query 1, Doc A
    [pagerank, query_match, freshness, ...],  # Query 1, Doc B
    [pagerank, query_match, freshness, ...],  # Query 1, Doc C
    [pagerank, query_match, freshness, ...],  # Query 2, Doc A
    ...
]

# Relevance labels for each document
y = [3, 2, 0, 2, ...]  # 0=irrelevant, 3=highly relevant

# Which query each document belongs to
query_ids = [1, 1, 1, 2, ...]
```

### Step 2: Initialize Model

```python
# Start with baseline prediction
baseline_score = mean(y)
predictions = [baseline_score] * n_samples
```

### Step 3: Gradient Boosting Loop

For each boosting iteration:

#### 3a. Compute Lambda Gradients

For each query:

```python
for query_id in unique_queries:
    # Get documents for this query
    query_docs = get_query_documents(query_id)
    
    # For each pair of documents
    for i in range(len(query_docs)):
        for j in range(i+1, len(query_docs)):
            # If wrong order
            if relevance[i] > relevance[j] and score[i] < score[j]:
                # Compute lambda gradient
                lambda_ij = compute_lambda(i, j, relevances, scores)
                
                # Update gradients
                gradients[i] += lambda_ij
                gradients[j] -= lambda_ij
```

#### 3b. Build Regression Tree

Fit a tree to predict the lambda gradients:

```python
tree = build_tree(X, gradients)
```

The tree learns to predict which documents should get higher/lower scores.

#### 3c. Update Predictions

```python
tree_predictions = tree.predict(X)
predictions += learning_rate * tree_predictions
```

### Step 4: Make Predictions

For new query-document pairs:

```python
score = baseline_score
for tree in trees:
    score += learning_rate * tree.predict(features)
```

### Step 5: Rank Documents

For each query, sort documents by predicted scores (descending):

```python
# Get all documents for query
query_docs = get_query_documents(query_id)
query_scores = predict(query_docs)

# Sort by score (higher = better)
ranked_docs = sort_by_score(query_docs, query_scores, descending=True)
```

## Implementation Details

### Feature Engineering

Good features are crucial for LTR! Common feature types:

#### 1. Query-Document Match Features
- **TF-IDF**: Term frequency × inverse document frequency
- **BM25**: Advanced relevance scoring function
- **Exact Match**: Does query appear exactly in document?
- **Partial Match**: How many query terms appear?
- **Query Coverage**: Fraction of query terms in document

#### 2. Document Quality Features
- **PageRank**: Link-based authority score
- **Domain Authority**: Trustworthiness of domain
- **Freshness**: How recently was document updated?
- **Length**: Document length (with normalization)
- **Readability**: Flesch reading ease score

#### 3. User Interaction Features
- **Click-Through Rate**: % of users who click this result
- **Dwell Time**: How long users stay on page
- **Bounce Rate**: % of users who immediately leave
- **Past Rankings**: Historical performance

#### 4. Context Features
- **Device Type**: Mobile vs desktop
- **Location**: Geographic relevance
- **Time of Day**: When was query made?
- **Query Intent**: Informational, navigational, transactional

### Handling Queries with Different Numbers of Documents

LTR naturally handles varying list lengths:

```python
# Query 1: 10 documents
query_1_features = X[0:10]    # Documents 0-9
query_1_ids = [1] * 10

# Query 2: 5 documents
query_2_features = X[10:15]   # Documents 10-14
query_2_ids = [2] * 5
```

Gradients are computed independently per query, so different list lengths are fine!

### Data Split Strategy

**IMPORTANT**: Split by queries, not documents!

```python
# CORRECT: Split queries
train_queries = [1, 2, 3, 4, 5]
test_queries = [6, 7]

# WRONG: Random split of documents
# This leaks information (test documents from training queries)
```

### Relevance Label Guidelines

Common schemes:

**Binary (0-1)**:
- 0 = Irrelevant
- 1 = Relevant

**Graded (0-4)**:
- 0 = Irrelevant
- 1 = Marginally relevant
- 2 = Relevant
- 3 = Highly relevant
- 4 = Perfectly relevant

**Tips**:
- More grades allow finer distinctions
- But harder to label consistently
- 5-point scale (0-4) is a sweet spot

## Usage Example

### Complete Search Engine Example

```python
import numpy as np
from learning_to_rank import LearningToRank

# ============================================================
# 1. Prepare Training Data
# ============================================================

# Query 1: "python tutorial"
# Features: [pagerank, query_match, freshness, domain_authority]
X_q1 = np.array([
    [0.8, 1.0, 0.9, 0.85],  # High quality Python tutorial
    [0.3, 0.5, 0.1, 0.40],  # Barely relevant
    [0.9, 1.0, 0.95, 0.90], # Excellent Python docs
    [0.2, 0.0, 0.3, 0.30],  # Irrelevant
])
y_q1 = np.array([3, 1, 3, 0])  # Relevance labels
qid_q1 = np.array([1, 1, 1, 1])

# Query 2: "machine learning basics"
X_q2 = np.array([
    [0.7, 0.8, 0.7, 0.75],  # Good ML intro
    [0.9, 1.0, 0.9, 0.95],  # Excellent ML course
    [0.4, 0.4, 0.4, 0.50],  # Somewhat related
])
y_q2 = np.array([2, 3, 1])
qid_q2 = np.array([2, 2, 2])

# Combine
X = np.vstack([X_q1, X_q2])
y = np.concatenate([y_q1, y_q2])
query_ids = np.concatenate([qid_q1, qid_q2])

# ============================================================
# 2. Train Model
# ============================================================

ltr = LearningToRank(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

ltr.fit(X, y, query_ids)

# ============================================================
# 3. Rank New Documents
# ============================================================

# New query: "python tutorial" with 3 candidate documents
X_new = np.array([
    [0.6, 0.8, 0.5, 0.70],  # Candidate A
    [0.9, 1.0, 0.9, 0.90],  # Candidate B
    [0.4, 0.6, 0.3, 0.50],  # Candidate C
])
query_ids_new = np.array([3, 3, 3])

# Get rankings
rankings = ltr.rank(X_new, query_ids_new)
print("Ranked documents for query 3:", rankings[3])
# Output: [1, 0, 2]  (B is best, then A, then C)

# ============================================================
# 4. Evaluate Performance
# ============================================================

# Compute NDCG
y_test = np.array([2, 3, 1])  # True relevances
ndcg_scores = ltr.evaluate(X_new, y_test, query_ids_new, k=3)
print(f"NDCG@3: {ndcg_scores['average']:.4f}")
```

## Hyperparameter Tuning

### Key Hyperparameters

#### 1. `n_estimators` (Number of Trees)
- **What it does**: Number of boosting iterations
- **Typical range**: 50-500
- **Tuning**:
  - Too few: Underfitting
  - Too many: Overfitting, slow training
  - Start with 100, increase if validation NDCG improves

#### 2. `learning_rate`
- **What it does**: Step size for each tree
- **Typical range**: 0.01-0.3
- **Tuning**:
  - Smaller: More robust, needs more trees
  - Larger: Faster convergence, may overfit
  - Common values: 0.05, 0.1, 0.15

#### 3. `max_depth`
- **What it does**: Maximum depth of each tree
- **Typical range**: 3-10
- **Tuning**:
  - Shallow (3-4): Fast, regularized
  - Medium (5-6): Good balance
  - Deep (7-10): Captures complex patterns, may overfit

#### 4. `min_samples_split`
- **What it does**: Minimum samples to split a node
- **Typical range**: 5-50
- **Tuning**:
  - Higher: More regularization
  - Lower: More flexible

#### 5. `subsample`
- **What it does**: Fraction of samples per tree
- **Typical range**: 0.5-1.0
- **Tuning**:
  - Less than 1.0: Stochastic gradient boosting (more robust)
  - 1.0: Use all data (faster but may overfit)

### Tuning Strategy

**Step 1: Start with defaults**
```python
ltr = LearningToRank(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
```

**Step 2: Adjust for your data size**

For small datasets (< 1000 samples):
```python
ltr = LearningToRank(
    n_estimators=50,
    learning_rate=0.2,
    max_depth=4
)
```

For large datasets (> 10000 samples):
```python
ltr = LearningToRank(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8
)
```

**Step 3: Grid search**
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [4, 6, 8]
}

best_ndcg = 0
best_params = None

for n_est in param_grid['n_estimators']:
    for lr in param_grid['learning_rate']:
        for depth in param_grid['max_depth']:
            model = LearningToRank(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth
            )
            model.fit(X_train, y_train, query_ids_train)
            
            ndcg = model.evaluate(X_val, y_val, query_ids_val, k=10)
            if ndcg['average'] > best_ndcg:
                best_ndcg = ndcg['average']
                best_params = (n_est, lr, depth)

print(f"Best params: {best_params}")
print(f"Best NDCG@10: {best_ndcg:.4f}")
```

## Performance Characteristics

### Time Complexity

**Training**:
- Per iteration: O(n × d × k²)
  - n = number of samples
  - d = number of features
  - k = average documents per query

- Total: O(T × n × d × k²)
  - T = n_estimators

**Prediction**:
- O(T × d × log(max_depth))
  - Very fast! Linear in number of trees

**Space Complexity**: O(T × max_depth × d)

### Scalability

**Small datasets** (< 1,000 samples):
- Works great, fast training (seconds)
- May need regularization

**Medium datasets** (1,000 - 100,000 samples):
- Sweet spot for this implementation
- Training time: minutes
- Good performance

**Large datasets** (> 100,000 samples):
- This implementation may be slow
- Consider:
  - Reducing n_estimators
  - Using subsample < 1.0
  - Production libraries (XGBoost, LightGBM with LTR objective)

### Comparison with Other Methods

**vs Pointwise (Regression)**:
- ✓ LTR: Better ranking quality
- ✓ LTR: Optimizes for ranking metrics
- − Pointwise: Simpler, faster

**vs Pairwise (RankNet, RankSVM)**:
- ✓ LambdaRank: Directly optimizes NDCG
- ✓ LambdaRank: Faster convergence
- − Others: Simpler to implement

**vs Listwise (ListNet, ListMLE)**:
- ✓ Listwise: Most direct optimization
- ✓ LambdaRank: Good balance of quality and speed
- − Listwise: More complex, slower

## Advanced Topics

### 1. Different Ranking Metrics

Besides NDCG, you can optimize for:

**Mean Average Precision (MAP)**:
```python
def compute_MAP(relevances, predictions):
    sorted_indices = np.argsort(-predictions)
    sorted_rels = relevances[sorted_indices]
    
    precisions = []
    num_relevant = 0
    for i, rel in enumerate(sorted_rels):
        if rel > 0:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0
```

**Mean Reciprocal Rank (MRR)**:
```python
def compute_MRR(relevances, predictions):
    sorted_indices = np.argsort(-predictions)
    sorted_rels = relevances[sorted_indices]
    
    for i, rel in enumerate(sorted_rels):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0
```

### 2. Position Bias

Real user clicks are biased toward top positions. Account for this:

```python
# Position bias: Users more likely to click higher positions
position_bias = lambda pos: 1.0 / np.log2(pos + 2)

# Adjust click data for bias
adjusted_clicks = clicks / position_bias(position)
```

### 3. Diversification

Avoid redundant results by promoting diversity:

```python
def diversified_ranking(docs, scores, similarity_matrix, lambda_param=0.5):
    """
    MMR (Maximal Marginal Relevance) ranking
    Balance relevance and diversity
    """
    ranked = []
    remaining = set(range(len(docs)))
    
    # Pick highest scoring document first
    first = np.argmax(scores)
    ranked.append(first)
    remaining.remove(first)
    
    while remaining:
        best_score = -np.inf
        best_doc = None
        
        for doc in remaining:
            # Relevance term
            relevance = scores[doc]
            
            # Diversity term (similarity to already selected)
            max_sim = max([similarity_matrix[doc, r] for r in ranked])
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr > best_score:
                best_score = mmr
                best_doc = doc
        
        ranked.append(best_doc)
        remaining.remove(best_doc)
    
    return ranked
```

### 4. Online Learning

Update model with new data without full retraining:

```python
def online_update(model, X_new, y_new, query_ids_new, n_new_trees=10):
    """
    Add new trees to existing model
    """
    # Get current predictions
    current_predictions = model.predict(X_new)
    
    # Train new trees on residuals
    for i in range(n_new_trees):
        # Compute gradients based on current predictions
        gradients = model._compute_all_gradients(
            X_new, y_new, query_ids_new, current_predictions
        )
        
        # Build tree
        tree = model._build_tree(X_new, gradients)
        model.trees_.append(tree)
        
        # Update predictions
        tree_pred = model._predict_tree(tree, X_new)
        current_predictions += model.learning_rate * tree_pred
```

## Common Issues and Solutions

### Issue 1: Low NDCG Scores

**Symptoms**: NDCG < 0.5, rankings seem random

**Solutions**:
1. Check feature quality
   - Are features actually predictive?
   - Correlation with relevance labels?
2. Add more features
   - Query-document match features crucial
   - User behavior features help
3. Increase model complexity
   - More trees: `n_estimators=200`
   - Deeper trees: `max_depth=8`
4. Check data quality
   - Are relevance labels accurate?
   - Enough training queries?

### Issue 2: Overfitting

**Symptoms**: Training NDCG high, test NDCG low

**Solutions**:
1. Reduce model complexity
   - Fewer trees
   - Shallower trees
   - Higher `min_samples_split`
2. Add regularization
   - Use `subsample < 1.0`
   - Smaller `learning_rate`
3. More training data
   - Collect more query-document pairs
4. Feature selection
   - Remove redundant features

### Issue 3: Slow Training

**Symptoms**: Training takes hours

**Solutions**:
1. Reduce `n_estimators`
2. Use `subsample < 1.0`
3. Reduce `max_depth`
4. Feature selection (fewer features)
5. Sample fewer documents per query during training

### Issue 4: Imbalanced Relevance Labels

**Symptoms**: Most labels are 0 or 1, few 3s or 4s

**Solutions**:
1. Collect more high-relevance examples
2. Oversample high-relevance pairs
3. Adjust training to focus on top positions
4. Use position-weighted loss

## Further Reading

### Papers

**LambdaRank and LambdaMART**:
- Burges et al. (2006): "Learning to Rank using Gradient Descent"
- Burges (2010): "From RankNet to LambdaRank to LambdaMART: An Overview"
  - Clear explanation of lambda gradients
  - Microsoft Research, highly cited

**Ranking Metrics**:
- Järvelin & Kekäläinen (2002): "Cumulated Gain-based Evaluation of IR Techniques"
  - Original NDCG paper

**Comparisons**:
- Liu (2009): "Learning to Rank for Information Retrieval"
  - Comprehensive survey of LTR methods

### Libraries

**Production Use**:
- **XGBoost**: Has built-in `rank:ndcg` objective
- **LightGBM**: Has `lambdarank` objective
- **TensorFlow Ranking**: Deep learning for ranking
- **RankLib**: Java-based LTR library

**Datasets**:
- **Microsoft LETOR**: Standard LTR benchmark
- **Yahoo Learning to Rank Challenge**: Large-scale dataset
- **Istella LETOR**: Web search dataset

### Applications

**Search Engines**:
- Google, Bing use sophisticated LTR models
- Combine with traditional IR (BM25)

**E-commerce**:
- Amazon, eBay product ranking
- Personalized recommendations

**Social Media**:
- Feed ranking (Facebook, Twitter)
- Content recommendation

## Summary

**Learning-to-Rank is a powerful technique for ordering items by relevance.**

**Key takeaways**:
1. ✓ Optimizes for ranking quality (NDCG), not prediction accuracy
2. ✓ Uses pairwise comparisons to learn relative ordering
3. ✓ Lambda gradients directly optimize ranking metrics
4. ✓ Works with graded relevance labels (0-4)
5. ✓ Essential for search engines and recommendation systems

**When to use**:
- You have query-document pairs with relevance labels
- Order matters more than exact scores
- You want to optimize ranking metrics (NDCG, MAP)

**Default settings**:
- `n_estimators=100`
- `learning_rate=0.1`
- `max_depth=6`
- Start here and adjust based on data size

**Critical for success**:
- Good feature engineering (query-doc match features crucial)
- Quality relevance labels (consistent, graded)
- Enough training queries (1000+ ideal)
- Proper evaluation (split by queries, not documents)

---

## Implementation Notes

This implementation is educational and demonstrates core concepts. For production use:
- Use **XGBoost** or **LightGBM** with LTR objectives (highly optimized)
- Consider **TensorFlow Ranking** for neural approaches
- Use approximate k-NN for large-scale feature extraction
- Implement caching for repeated queries

**Our implementation shows how LambdaRank works under the hood!**

---

**Happy ranking!** 🏆📊🔍
