# Learning-to-Rank (LambdaRank) from Scratch: A Comprehensive Guide

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [Overview](#overview)
3. [When to Use Learning-to-Rank](#when-to-use-learning-to-rank)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Algorithm Steps](#algorithm-steps)
6. [Implementation Details](#implementation-details)
7. [Usage Example](#usage-example)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Performance Characteristics](#performance-characteristics)
10. [Simplification vs. canonical LambdaMART](#simplification-vs-canonical-lambdamart)
11. [Advanced Topics](#advanced-topics)
12. [Common Issues and Solutions](#common-issues-and-solutions)
13. [Further Reading](#further-reading)
14. [Summary](#summary)
15. [Implementation Notes](#implementation-notes)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Learning-to-Rank from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _27_learning_to_rank.py  (its __main__ block runs this,
# plus a small search-engine example and a hyperparameter comparison)
# Or copy the LearningToRank class from _27_learning_to_rank.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the LearningToRank class here (from _27_learning_to_rank.py) ----
# class LearningToRank: ...

np.random.seed(42)

# ---- Build a small e-commerce ranking dataset: 12 searches x 8 products ----
n_queries, n_per_query = 12, 8
X, y, query_ids = [], [], []

for qid in range(n_queries):
    for _ in range(n_per_query):
        price_score = np.random.rand()             # 0=expensive, 1=cheap
        rating = np.random.uniform(3.0, 5.0)
        num_reviews = np.random.randint(0, 1000)
        relevance = np.random.rand()
        in_stock = np.random.choice([0, 1], p=[0.2, 0.8])

        X.append([price_score, rating / 5.0,
                  min(num_reviews / 1000, 1.0), relevance, in_stock])

        # Graded label 0-3 derived from a hidden linear utility
        score = (0.2 * price_score + 0.3 * (rating / 5.0) +
                 0.2 * min(num_reviews / 1000, 1.0) + 0.3 * relevance)
        y.append(3 if score > 0.7 else 2 if score > 0.5 else 1 if score > 0.3 else 0)
        query_ids.append(qid)

X, y, query_ids = np.array(X), np.array(y), np.array(query_ids)

# ---- SPLIT BY QUERY, never by document ----
# Queries 0-7 train, queries 8-11 are held out completely.
train = query_ids < 8
test = ~train
X_tr, y_tr, q_tr = X[train], y[train], query_ids[train]
X_te, y_te, q_te = X[test], y[test], query_ids[test]

# ---- Train ----
# min_samples_split=4 matters: the default of 10 would leave every tree a bare
# leaf on 8-document queries and the model could not learn anything.
model = LearningToRank(
    n_estimators=60,
    learning_rate=0.15,
    max_depth=5,
    min_samples_split=4,
    subsample=1.0,
    random_state=42,
    verbose=0            # 0 silences the per-iteration progress lines
)
model.fit(X_tr, y_tr, q_tr)

# ---- Baseline: an untrained model scores every document the same, so its
# "ranking" is just the input order. That is the number we have to beat. ----
def input_order_ndcg(y_sub, q_sub):
    return np.mean([model._compute_ndcg(y_sub[q_sub == q],
                                        np.zeros(np.sum(q_sub == q)), 5)
                    for q in np.unique(q_sub)])

print(f"Train NDCG@5: {input_order_ndcg(y_tr, q_tr):.4f} (input order)"
      f" -> {model.evaluate(X_tr, y_tr, q_tr, k=5)['average']:.4f} (trained)")
print(f"Test  NDCG@5: {input_order_ndcg(y_te, q_te):.4f} (input order)"
      f" -> {model.evaluate(X_te, y_te, q_te, k=5)['average']:.4f} (trained)")

# ---- Rank the documents of one HELD-OUT query ----
# rank() returns GLOBAL row indices into X_te, best first.
rankings = model.rank(X_te, q_te)
print("\nRanked products for held-out query 8 (true relevance, best first):")
print("  ", [int(y_te[i]) for i in rankings[8]])

scores = model.predict(X_te)
print("Top-3 predicted scores:", np.round(scores[rankings[8][:3]], 4))
```

Expected output:
```
Train NDCG@5: 0.7594 (input order) -> 1.0000 (trained)
Test  NDCG@5: 0.4803 (input order) -> 0.9151 (trained)

Ranked products for held-out query 8 (true relevance, best first):
   [3, 2, 1, 1, 2, 2, 1, 1]
Top-3 predicted scores: [2.0139 1.8452 1.8187]
```

Read that output carefully, because it is the whole algorithm in three numbers:

- **0.4803 -> 0.9151 on queries the model has never seen.** The four test queries were held out *whole*; not one of their products appeared in training. So this is genuine generalisation, not memorisation.
- **Train reaches 1.0000, test stops at 0.9151.** That gap is normal for a boosted ranker on 64 training rows - see [Common Issues and Solutions](#common-issues-and-solutions).
- **The held-out list starts `[3, 2, ...]` and ends `[..., 1, 1]`.** It is not perfect (two relevance-2 products sit below a relevance-1 one), but the highly-relevant product is in position 1, which is exactly what NDCG rewards most.

---

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

For a pair (i, j) inside one query where **i is the more relevant document**, with model scores `sᵢ`, `sⱼ`:

```
λᵢⱼ = |ΔNDCGᵢⱼ| × σ / (1 + exp(σ × (sᵢ - sⱼ)))
```

Where:
- `|ΔNDCGᵢⱼ|` = Change in NDCG if documents i and j exchanged **rank positions**
- `σ` = Shape parameter of the logistic (this implementation uses σ = 1.0)
- The second factor is the **logistic** term, not the sigmoid derivative

**How the logistic factor behaves** — this is the part that makes LambdaRank work:

| Situation | `sᵢ - sⱼ` | Weight |
|---|---|---|
| Pair badly mis-ordered (relevant doc scored far too low) | very negative | → σ (maximum push) |
| Pair exactly tied (e.g. round 0, all scores equal) | 0 | σ / 2 |
| Pair already confidently correct | very positive | → 0 (no push) |

**Key Idea**: 
- Every pair with *different* labels gets a gradient — there is **no** "only if the order is wrong" test
- The logistic factor already fades a correctly-ordered pair out to zero on its own
- Weight the push by how much exchanging their positions would change NDCG
- Larger NDCG change → stronger push

> **Why no ordering guard?** It is tempting to write `if relevance[i] > relevance[j] and score[i] < score[j]`. That version cannot learn. Boosting starts every document at the *same* constant base score, so on round 0 no pair is strictly mis-ordered, every lambda is 0, the first tree is a bare leaf of value 0, the scores never move — and the model stays frozen at its initialisation forever. The logistic factor is what makes the guard unnecessary *and* harmful.

> **Why the logistic, not `σ'`?** The sigmoid derivative `σ'(x) = σ(x)(1 - σ(x))` **peaks at a tie** and decays toward 0 as a pair becomes more badly mis-ordered — precisely backwards. At a score gap of 5.0 it gives 0.0066 where the canonical logistic gives 0.9933. Burges (2010) uses the logistic form.

### 5. The LambdaRank Algorithm

For each query, first record each document's current **rank position** `r` (1 = highest score), then for each pair of documents (i, j):

```python
# 0. Once per query: rank positions under the current scores, and the normaliser
order = argsort(-scores)                      # stable sort, best score first
rank[order] = arange(1, n_docs + 1)           # rank[d] = position of document d
IDCG = DCG(sorted(relevances, descending))
    
for i in range(n_docs):
    for j in range(i + 1, n_docs):
    
        # 1. Only skip pairs that carry no ranking information
        if relevance[i] == relevance[j]:
            continue
    
        # 2. Orient the pair: hi = the MORE relevant document
        hi, lo = (i, j) if relevance[i] > relevance[j] else (j, i)
    
        # 3. |delta NDCG| from exchanging their POSITIONS - closed form, no re-sorting
        delta_NDCG = abs((2**relevance[hi] - 2**relevance[lo]) *
                         (1/log2(1 + rank[hi]) - 1/log2(1 + rank[lo]))) / IDCG

        # 4. Logistic weight (sigma = 1.0)
        weight = sigma / (1 + exp(sigma * (score[hi] - score[lo])))

        # 5. Compute lambda gradient
        lambda_ij = delta_NDCG * weight

        # 6. Update gradients
        gradient[hi] += lambda_ij   # Push the more relevant document up
        gradient[lo] -= lambda_ij   # Push the less relevant document down
```

This is exactly what `_compute_lambda_gradients` in `_27_learning_to_rank.py` does, line for line. Note step 3 in particular: `|ΔNDCG|` comes from the closed form over rank **positions**. Computing it by swapping the two *scores* and re-evaluating NDCG looks equivalent but is not — when the two scores are equal (as they all are on round 0) swapping them leaves the array byte-identical and `|ΔNDCG|` is identically zero.

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
            # Skip ONLY equal-relevance pairs: exchanging two documents with the
            # same label cannot change NDCG, so they carry no gradient.
            # There is deliberately no "and score[i] < score[j]" test here -
            # see "Why no ordering guard?" above.
            if relevance[i] == relevance[j]:
                continue
                
            # Orient the pair so hi is the more relevant document
            hi, lo = (i, j) if relevance[i] > relevance[j] else (j, i)

            # Compute lambda gradient (|delta NDCG| x logistic weight)
            lambda_ij = compute_lambda(hi, lo, relevances, scores, ranks)

            # Update gradients
            gradients[hi] += lambda_ij
            gradients[lo] -= lambda_ij
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
# Illustrative slicing only - X is the stacked feature matrix of every query's
# documents, built as in Step 1 above; this fragment is not a runnable program.

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

# In practice, with query ids 0..11 and a boolean mask:
train = query_ids < 8          # queries 0-7
test = ~train                  # queries 8-11
X_tr, y_tr, q_tr = X[train], y[train], query_ids[train]
X_te, y_te, q_te = X[test], y[test], query_ids[test]

# WRONG: Random split of documents
# This leaks information (test documents from training queries)
```

This is not just advice — it is what the Quick Start above and Examples 2 and 3 in
`_27_learning_to_rank.py` actually do. The difference is large and visible: on the
e-commerce data the model reaches **NDCG@5 = 1.0000 on its training queries** but
**0.9151 on the four queries it has never seen**. A document-level split would have
reported something in between and called it a test score.

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
# Paste the LearningToRank class from _27_learning_to_rank.py above,
# or run that file directly. There is no importable `learning_to_rank` module:
# the file is named `_27_learning_to_rank.py` inside a folder with spaces and a
# dot in its name, so `import` cannot reach it.

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

# Only 7 rows here, so min_samples_split must drop below the row count
# (and subsample must not shrink it further) or every tree stays a bare leaf.
ltr = LearningToRank(
    n_estimators=60,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=4,
    subsample=1.0,
    random_state=42,
    verbose=0            # 0 silences the per-iteration progress lines
)

ltr.fit(X, y, query_ids)

print(f"Training NDCG@3: {ltr.evaluate(X, y, query_ids, k=3)['average']:.4f}")

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

# Get rankings. rank() returns GLOBAL row indices into X_new, best first.
rankings = ltr.rank(X_new, query_ids_new)
print("Ranked documents for query 3:", rankings[3])
print("Raw scores:", np.round(ltr.predict(X_new), 4))

# ============================================================
# 4. Evaluate Performance
# ============================================================

# Compute NDCG
y_test = np.array([2, 3, 1])  # True relevances
ndcg_scores = ltr.evaluate(X_new, y_test, query_ids_new, k=3)
print(f"NDCG@3 on the new query: {ndcg_scores['average']:.4f}")
```

Expected output:
```
Training NDCG@3: 1.0000
Ranked documents for query 3: [1 0 2]
Raw scores: [1.4267 2.6406 1.4267]
NDCG@3 on the new query: 1.0000
```

Candidate B is ranked first, then A, then C — which is the right order, since B dominates
A and C on every feature. Note that A and C receive the *same* score (1.4267): the trained
trees route both of them to the same leaf, and `rank()` breaks the tie by input order.
Identical scores for distinct documents are normal for a small tree ensemble, and they are
also why `_compute_ndcg` uses a **stable** sort — so a tie is resolved reproducibly rather
than arbitrarily.

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
- **Watch out**: `subsample` and `min_samples_split` interact. A tree is built on
  `int(subsample × n_samples)` rows; if that is below `min_samples_split` the tree is a
  bare leaf and the round is wasted. With the defaults (`0.7` and `10`) any dataset under
  15 rows learns *nothing*, which is why every small example in these files passes
  `min_samples_split=4, subsample=1.0`.

#### 6. `verbose`
- **What it does**: Prints `Iteration t/T, Avg NDCG` every `verbose` rounds
- **Default**: 20
- **Tuning**: set `verbose=0` inside loops, grid searches and scripts; the training NDCG
  it prints is a useful convergence check when exploring interactively
- **Which NDCG**: the line reports the metric at the `ndcg_k` cutoff — the full list by
  default, or `Avg NDCG@k` when `ndcg_k=k` is set — so the number you watch converging is
  the objective the gradients are actually optimising

#### 7. `ndcg_k`
- **What it does**: Truncates the `|ΔNDCG|` used in the gradients at position k
- **Default**: `None` (optimise the full result list)
- **Tuning**: set it to the same k you report (`ndcg_k=5` alongside `evaluate(..., k=5)`)
  so training and evaluation optimise the same objective. On the Quick Start data this
  lifts held-out NDCG@5 from 0.9151 to 0.9472 — `None` is the backward-compatible default,
  not the best one. The exception is a very small k on a very long result list, where
  pairs that both sit below position k stop contributing any gradient

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
                max_depth=depth,
                random_state=42,
                verbose=0     # essential: without it this loop prints ~450 lines
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

**Training**, per boosting round, has two separate costs:
- Lambda gradients: O(Q × k²) where Q = number of queries and k = documents per query
  (every pair inside every query), plus one O(k log k) sort per query
- Tree building: O(n × d × 10 × max_depth), since the split search scores at most
  10 candidate thresholds per feature per node
- Total: O(T × (Q × k² + n × d × max_depth)) with T = n_estimators

**The real scaling limit is k, the number of documents per query, not n.** The pair
loop is quadratic in k, so 200 queries of 10 documents is far cheaper than 20 queries
of 100 documents even though both are 2,000 rows.

**Prediction**:
- O(T × max_depth) per query-document pair — one root-to-leaf walk per tree.
  It does **not** depend on the number of features, and it is not logarithmic in the depth.

**Space Complexity**: O(T × 2^max_depth) in the worst case (a full binary tree of that depth).

### Scalability

Measured on the Quick Start e-commerce data above (`np.random.seed(42)`, 8 documents per
query, `learning_rate=0.15`, `max_depth=5`, `min_samples_split=4`, `subsample=1.0`),
median of 5 `fit()` calls, Python 3.13 / numpy 2.3:

| Rows | Trees | Wall clock |
|---|---|---|
| 40 (queries 0-4) | 60 | 0.56 s |
| 64 (queries 0-7) | 60 | 0.82 s |
| 64 (queries 0-7) | 150 | 2.29 s |

Read these as a shape, not a benchmark: wall clock varies by machine, and the label
distribution matters as much as the row count: the pair loop visits every pair, but only
pairs with *different* labels do any work inside it — the utility-derived labels above
leave fewer such pairs than uniform random labels would.

The whole `__main__` demo — three examples and 400 boosting rounds in total — runs in
about 5 seconds.

**Small datasets** (< 1,000 rows):
- Works great, fast training (seconds)
- May need regularization

**Medium datasets** (1,000 - 100,000 rows):
- Usable, but only while documents-per-query stays small (roughly k ≤ 20).
- Expect minutes at the low end of that range and much worse at the high end;
  this is pure-Python nested loops, not a compiled library.

**Large datasets** (> 100,000 rows):
- This implementation is too slow. Use:
  - Fewer n_estimators, `subsample < 1.0`
  - Fewer documents per query (sample candidates before training)
  - Production libraries (XGBoost `rank:ndcg`, LightGBM `lambdarank`)

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

## Simplification vs. canonical LambdaMART

The gradient in this implementation is the real LambdaRank gradient. What sits *around*
it is deliberately simplified for teaching. Here is exactly what differs, and what it costs.

### 1. Leaf values are a mean, not a Newton step

**Canonical (Burges 2010, the LambdaMART section):** LambdaMART fits a regression tree to the lambdas and then
*overwrites* each leaf value with a Newton step,

```
value(leaf) = sum_{i in leaf} lambda_i / sum_{i in leaf} w_i
```

where `w_i = ∂²C/∂s_i²` is the second derivative of the pairwise cost
`C = Σ_pairs |ΔNDCG_ij| · log(1 + exp(-σ(s_i - s_j)))`, accumulated per pair as
`w_ij = σ² · |ΔNDCG_ij| · ρ_ij · (1 - ρ_ij)` with `ρ_ij = 1/(1 + exp(σ(s_i - s_j)))`.

**Mind the sign convention.** The `λ` of sections 4-5 is the *ascent* direction `-∂C/∂s_i`
(positive λ = push this document up), so `w_i = -∂λ_i/∂s_i`, **not** `+∂λ_i/∂s_i`. Writing
it the other way round flips the sign of every non-zero `w_i` — central-differencing the code's own
`_compute_lambda_gradients` on `relevances = [3, 1]` gives `∂λ_hi/∂s_hi = -0.072548` at a
tie against `w_ij = +0.072548` — and the Newton step `Σλ / Σw` would then push documents
the wrong way. Presentations that define `λ = +∂C/∂s` can write `w = +∂λ/∂s`; this file's
`λ` is the negative of that, so the minus sign has to be carried over with it.

**Here:** `_build_tree` returns `{'leaf': True, 'value': np.mean(gradients)}`.

**Why omitted:** it requires threading a second per-document accumulator through the pair
loop, the tree builder and both leaf sites — and, more importantly, it would hide the one
idea the file exists to show, that a *ranking metric* can be turned into a gradient at all.

**Consequence:** the step size per leaf is not curvature-corrected, so convergence is slower
and more sensitive to `learning_rate`. It does not change what the model converges *to*.
This is the same relationship as plain gradient boosting versus XGBoost — compare
`_17_xgboost.py`'s `_compute_gradient_hessian`, which does compute both.

### 2. The split search looks at 10 thresholds per feature

**Canonical:** exact greedy split finding evaluates every midpoint between consecutive
distinct feature values (or a weighted quantile sketch approximates them).

**Here:** if a feature has more than 10 candidate midpoints, 10 evenly spaced ones are taken:
`thresholds[np.linspace(0, len(thresholds) - 1, 10).astype(int)]`.

**Consequence:** split points are quantised. Note the *evenly spaced* part matters — an
earlier version of this file used `thresholds[:10]`, the 10 **smallest** midpoints, so the
upper part of every feature's range was never considered and trees could only split near a
feature's minimum.

### 3. dNDCG is not truncated by default

**Canonical:** LambdaMART truncates the discount at the target cutoff k, so training
optimises the same NDCG@k that you report.

**Here:** truncation *is* implemented — positions below the cutoff get a zero discount —
but `ndcg_k=None` is the default, which optimises the full result list. Pass `ndcg_k=5`
to match `evaluate(..., k=5)`.

**Consequence:** by default, training and reporting optimise slightly different things.
Measured on the Quick Start's e-commerce data, aligning them helps at every ensemble size:

| Trees | `ndcg_k=None` test NDCG@5 | `ndcg_k=5` test NDCG@5 |
|---|---|---|
| 5 | 0.8786 | 0.9354 |
| 10 | 0.8892 | 0.9516 |
| 20 | 0.8725 | 0.9557 |
| 40 | 0.9283 | 0.9472 |
| 60 | 0.9151 | 0.9472 |

So `None` is the *backward-compatible* default, not the recommended one. The one situation
where truncation hurts is a very small k on a very long result list: pairs whose documents
both sit below position k contribute no gradient at all, so the signal thins out.

### 4. No query-level normalisation or listwise extensions

No per-query gradient normalisation, no position-bias / propensity correction, no
sparsity-aware missing-value handling, and no early stopping on a validation set.
NaN feature values are silently routed right at every node (because `NaN <= threshold`
is `False`).

---

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
    
    # Train new trees on the lambda gradients
    for i in range(n_new_trees):
        # Compute gradients based on current predictions.
        # There is no single _compute_all_gradients method; fit() loops over the
        # queries itself, and so do we:
        gradients = np.zeros(len(y_new))
        for query_id in np.unique(query_ids_new):
            mask = query_ids_new == query_id
            gradients[mask] = model._compute_lambda_gradients(
                y_new[mask], current_predictions[mask]
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
