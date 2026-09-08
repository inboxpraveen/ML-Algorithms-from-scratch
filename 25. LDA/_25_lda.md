# Latent Dirichlet Allocation from Scratch: A Comprehensive Guide

## Table of Contents

1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [Overview](#overview)
3. [When to Use LDA](#when-to-use-lda)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Algorithm Steps](#algorithm-steps)
6. [Step-by-Step Example](#step-by-step-example)
7. [Parameters Explained](#parameters-explained)
8. [Code Example](#code-example)
9. [Understanding the Code](#understanding-the-code)
10. [Practical Use Cases](#practical-use-cases)
11. [Data Preprocessing Guide](#data-preprocessing-guide)
12. [Evaluation Metrics](#evaluation-metrics)
13. [MCMC Hygiene: Burn-in, Mixing and Label Switching](#mcmc-hygiene-burn-in-mixing-and-label-switching)
14. [Common Issues and Solutions](#common-issues-and-solutions)
15. [Tips for Success](#tips-for-success)
16. [LDA vs Other Methods](#lda-vs-other-methods)
17. [Advanced Topics](#advanced-topics)
18. [Performance Considerations](#performance-considerations)
19. [Further Reading](#further-reading)
20. [Summary](#summary)
21. [Implementation Notes](#implementation-notes)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra
dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# LDA from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _25_lda.py  (the __main__ block runs a superset of this)
# Or copy the LatentDirichletAllocation class from _25_lda.py and paste above.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the LatentDirichletAllocation class here (from _25_lda.py) ----
# class LatentDirichletAllocation: ...

np.random.seed(42)

# 15-word vocabulary: 5 sports words, 5 tech words, 5 food words
vocabulary = [
    'game', 'team', 'player', 'win', 'score',        # Sports words (0-4)
    'computer', 'software', 'code', 'data', 'tech',  # Tech words (5-9)
    'food', 'recipe', 'cook', 'taste', 'dish'        # Food words (10-14)
]

# Document-term matrix: 10 documents x 15 words.
# Docs 0-2 are sports, 3-5 tech, 6-8 food, and doc 9 is a deliberate mixture.
X = np.array([
    [5, 4, 3, 2, 3,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
    [4, 5, 4, 3, 4,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
    [3, 3, 5, 4, 3,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0,  5, 4, 3, 4, 3,  0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0,  4, 5, 4, 5, 4,  0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0,  3, 4, 5, 3, 5,  0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  5, 4, 3, 4, 3],
    [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  4, 5, 4, 3, 4],
    [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  3, 4, 5, 5, 4],
    [2, 1, 1, 1, 0,  1, 2, 1, 0, 1,  1, 1, 0, 1, 2],
])

lda = LatentDirichletAllocation(
    n_components=3,     # look for 3 topics
    max_iter=100,       # 100 Gibbs sweeps
    alpha=0.1,          # sparse document-topic prior
    beta=0.01,          # sparse topic-word prior
    random_state=42     # reproducible
)
doc_topics = lda.fit_transform(X)

print("Discovered topics (top 5 words each):")
for i, words in enumerate(lda.get_top_words(vocabulary, n_top_words=5)):
    print(f"  Topic {i}: {', '.join(words)}")

print("\nDocument-topic distribution (rows = documents, columns = topics):")
print(np.round(doc_topics, 3))

# ---- Did it work? The corpus has known ground truth ----
# LDA numbers topics arbitrarily, so map each true group to the topic its
# documents mostly chose, then score.
true_groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])   # doc 9 excluded: mixed
dominant = doc_topics[:9].argmax(axis=1)
group_to_topic = {g: np.bincount(dominant[true_groups == g], minlength=3).argmax()
                  for g in range(3)}
correct = sum(dominant[i] == group_to_topic[true_groups[i]] for i in range(9))

print(f"\nGroup -> topic mapping (sports, tech, food): "
      f"{[int(group_to_topic[g]) for g in range(3)]}")
print(f"Train topic-recovery accuracy: {correct}/9 = {100.0 * correct / 9:.2f}%")
print(f"Train perplexity            : {lda.perplexity(X):.4f}")
print(f"Uniform-guess perplexity    : {len(vocabulary):.4f}  (chance baseline = V)")

# ---- Fold in a document the model has never seen ----
X_new = np.array([[4, 3, 4, 2, 3,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0]])  # a sports doc
print(f"\nNew sports document -> {np.round(lda.transform(X_new)[0], 3)}")
print(f"Held-out perplexity         : {lda.perplexity(X_new):.4f}")
```

Expected output:

```
Discovered topics (top 5 words each):
  Topic 0: software, tech, code, computer, data
  Topic 1: recipe, dish, taste, food, cook
  Topic 2: game, player, team, score, win

Document-topic distribution (rows = documents, columns = topics):
[[0.006 0.006 0.988]
 [0.005 0.005 0.99 ]
 [0.005 0.005 0.989]
 [0.99  0.005 0.005]
 [0.991 0.004 0.004]
 [0.99  0.005 0.005]
 [0.005 0.99  0.005]
 [0.005 0.99  0.005]
 [0.005 0.991 0.005]
 [0.333 0.333 0.333]]

Group -> topic mapping (sports, tech, food): [2, 0, 1]
Train topic-recovery accuracy: 9/9 = 100.00%
Train perplexity            : 5.4852
Uniform-guess perplexity    : 15.0000  (chance baseline = V)

New sports document -> [0.006 0.006 0.988]
Held-out perplexity         : 4.9871
```

**How to read this output.** The three topics come out as clean word groups, and the
nine single-subject documents each put 98.8%+ of their mass on one topic. Document 9
was written as an even blend of all three subjects, and LDA reports exactly
`[0.333 0.333 0.333]` for it - it has 5 sports tokens, 5 tech tokens and 5 food
tokens, so the posterior really is uniform. Perplexity 5.49 against a
uniform-guess baseline of V = 15 means the model predicts the corpus about three
times better than chance. **The topic numbers themselves are arbitrary** - run with
a different `random_state` and "sports" may become topic 0 instead of topic 2.

---

## Overview

**Latent Dirichlet Allocation (LDA)** is a generative probabilistic model used to discover hidden topics in collections of text documents. It assumes that each document is a mixture of topics, and each topic is characterized by a distribution over words. LDA is one of the most popular and widely-used topic modeling techniques.

### Key Concept

Imagine you have a collection of news articles. Some are about sports, some about technology, and some about politics. However, many articles discuss multiple topics. LDA discovers these hidden topics automatically!

**The LDA Story:**
1. Each document is a mixture of topics (e.g., 70% sports, 20% politics, 10% technology)
2. Each topic is a mixture of words (e.g., sports topic uses words like "game", "player", "win")
3. LDA discovers these mixtures from the data automatically

Think of it like a chef analyzing recipes: Each recipe (document) combines multiple cooking techniques (topics), and each technique is characterized by specific ingredients (words). LDA figures out both the techniques and how they're used in each recipe!

## When to Use LDA

### Perfect For:
- **Topic Discovery**: Find hidden themes in large document collections
- **Document Organization**: Automatically categorize documents by content
- **Content Recommendation**: Find similar documents based on topic overlap
- **Trend Analysis**: Track topic evolution over time (news, social media)
- **Text Mining**: Extract semantic patterns from unstructured text
- **Information Retrieval**: Improve search and document ranking

### Real-World Applications:
- **News Analysis**: Discover trending topics in news articles
- **Customer Feedback**: Identify themes in product reviews or support tickets
- **Academic Research**: Organize scientific papers by research topics
- **Social Media**: Analyze conversation themes on Twitter, Reddit
- **Legal Documents**: Categorize legal cases by subject matter
- **Medical Records**: Identify disease patterns in clinical notes

## Mathematical Foundation

### 1. The Generative Process

LDA is a **generative model**, meaning it describes how documents are created:

**For each document d:**
1. Choose a topic distribution θ_d ~ Dirichlet(α)
2. For each word n in document d:
   - Choose a topic z_n ~ Multinomial(θ_d)
   - Choose a word w_n ~ Multinomial(φ_z_n)

Where:
- **θ_d** = topic distribution for document d
- **φ_k** = word distribution for topic k
- **α** = Dirichlet prior for document-topic distribution
- **β** = Dirichlet prior for topic-word distribution

### 2. Key Distributions

#### Dirichlet Distribution

The Dirichlet distribution is a distribution over probability distributions. It controls how sparse or uniform our topic/word distributions are.

```
θ ~ Dirichlet(α)
```

**Intuition:** Think of α as a pseudo-count. If α is small, distributions are sparse (few active components). If α is large, distributions are uniform (many active components).

#### Document-Topic Distribution (θ)

For each document d, θ_d is a distribution over K topics:

```
θ_d = [P(topic 1 | doc d), P(topic 2 | doc d), ..., P(topic K | doc d)]
```

**Example:**
```
Document: "New AI technology improves medical diagnosis"
θ = [0.5 (Technology), 0.4 (Medicine), 0.1 (Other topics)]
```

#### Topic-Word Distribution (φ)

For each topic k, φ_k is a distribution over V vocabulary words:

```
φ_k = [P(word 1 | topic k), P(word 2 | topic k), ..., P(word V | topic k)]
```

**Example:**
```
Technology Topic:
φ_tech = [P(computer)=0.05, P(software)=0.04, P(AI)=0.03, ...]
```

### 3. The Inference Problem

**Given:** Documents (observed words)
**Find:** θ (document-topic), φ (topic-word), z (topic assignments)

The posterior distribution is:

```
P(θ, φ, z | w, α, β) = P(θ, φ, z, w | α, β) / P(w | α, β)
```

This is **intractable** to compute exactly, so we use **Gibbs Sampling** to approximate it.

### 4. Collapsed Gibbs Sampling

#### What "collapsed" means

A plain Gibbs sampler for LDA would cycle through **three** sets of unknowns: θ, φ and z.
The *collapsed* sampler integrates θ and φ out of the joint distribution analytically and
samples **only z** — one integer per word token.

This is possible because the Dirichlet prior is **conjugate** to the multinomial
likelihood. Integrating a multinomial over a Dirichlet prior has a closed form, so

```
P(w, z | α, β) = ∫∫ P(w, z, θ, φ | α, β) dθ dφ
```

can be written down exactly, with no θ or φ left in it. Fewer variables to sample means a
lower-variance, faster-mixing chain — and it is why this file never stores a sampled θ or
φ during inference. They are reconstructed at the very end, from counts.

#### Where the formula comes from

Carrying out those two integrals gives a product of Dirichlet-multinomial normalizers
(the Δ function is the multivariate Beta function, Δ(x) = ∏ᵢΓ(xᵢ) / Γ(Σᵢxᵢ)):

```
P(w, z | α, β) = ∏_k [ Δ(n_k,· + β) / Δ(β) ]  ×  ∏_d [ Δ(n_d,· + α) / Δ(α) ]
```

The Gibbs conditional for one token is a **ratio** of that joint with and without the
token:

```
P(z_i = k | z_-i, w) = P(w, z) / P(w, z_-i)
```

Almost everything cancels — only the terms containing token *i* survive. And because
Γ(n+1)/Γ(n) = n, each surviving Gamma ratio collapses to a plain **count**. What is left
is the update rule the code implements:

```
P(z_i = k | z_-i, w, α, β) ∝ (n_d,k + α) × (n_k,w + β) / (n_k + V×β)
```

Where:
- **n_d,k** = count of words in document d assigned to topic k (excluding current word)
- **n_k,w** = count of word w assigned to topic k (excluding current word)
- **n_k** = total count of words in topic k (excluding current word)
- **V** = vocabulary size

That is the whole reason the sampler is nothing but *decrement, score K topics, sample,
increment*: the closed-form integral turned a hard Bayesian posterior into bookkeeping
with three count tables.

**Intuition:** Assign word w in document d to topic k based on:
1. How much document d likes topic k (first term)
2. How much topic k likes word w (second term)

**Note on the normalizer.** `V×β` is really `Σᵥ βᵥ`, the sum of the topic-word prior over
the whole vocabulary. The two are identical when β is a single number (a *symmetric*
prior), which is the usual case. This implementation computes `Σᵥ βᵥ` so that an
asymmetric β vector — one pseudo-count per word — also works. The same is true of `K×α`
versus `Σₖ αₖ` in the θ estimator below.

## Algorithm Steps

### Step 1: Initialize Parameters

Randomly assign topics to all word occurrences in all documents.

```python
for each document d:
    for each word w in document d:
        # Randomly assign topic
        z_w = random_topic()
        
        # Update counts
        doc_topic_count[d, z_w] += 1
        topic_word_count[z_w, w] += 1
        topic_count[z_w] += 1
```

### Step 2: Gibbs Sampling Iteration

For each word occurrence, sample a new topic based on conditional probability.

```python
for iteration in range(max_iter):
    for each document d:
        for each word occurrence (w, old_topic):
            # 1. Remove current assignment
            doc_topic_count[d, old_topic] -= 1
            topic_word_count[old_topic, w] -= 1
            topic_count[old_topic] -= 1
            
            # 2. Compute probability for each topic
            for each topic k:
                p_doc = doc_topic_count[d, k] + alpha
                p_topic = (topic_word_count[k, w] + beta) / (topic_count[k] + V*beta)
                prob[k] = p_doc * p_topic
            
            # 3. Sample new topic
            new_topic = sample_from(prob)
            
            # 4. Update with new assignment
            doc_topic_count[d, new_topic] += 1
            topic_word_count[new_topic, w] += 1
            topic_count[new_topic] += 1
```

### Step 3: Compute Final Distributions

After Gibbs sampling converges:

```python
# Document-topic distribution
for each document d:
    θ_d = (doc_topic_count[d] + alpha) / sum(doc_topic_count[d] + alpha)

# Topic-word distribution
for each topic k:
    φ_k = (topic_word_count[k] + beta) / sum(topic_word_count[k] + beta)
```

### Step 4: Evaluate and Interpret

```python
# Get top words for each topic
for each topic k:
    top_words = highest_probability_words(φ_k, n=10)
    print(f"Topic {k}: {top_words}")

# Assign documents to dominant topics
for each document d:
    dominant_topic = argmax(θ_d)
```

## Step-by-Step Example

Pseudocode is easy to nod along to. Here is **one real token, with real numbers**, taken
from the Quick Start corpus (10 documents, V = 15 words, K = 3 topics, α = 0.1, β = 0.01,
`random_state=42`). Because β is symmetric, the normalizer is `Σᵥ βᵥ = V×β = 15 × 0.01 = 0.15`.

### After random initialization (before any sampling)

Every one of the corpus's 191 tokens was handed a random topic, so the counts are noise:

```
Document 0 topic counts   n_d,·  = [4, 5, 8]     (17 tokens in doc 0)
Global topic counts       n_·    = [56, 67, 68]
Counts of word 0 "game"   n_·,0  = [4, 6, 4]
```

### Step 1 - remove the token we are about to resample

The first token of document 0 is the word **"game"** (word index 0), currently assigned to
**topic 0**. We take it out of all three tables — this is the "excluding current word"
part of the formula, and it is the only reason the conditional is tractable:

```
n_d,·  = [3, 5, 8]      (was [4, 5, 8])
n_·,0  = [3, 6, 4]      (was [4, 6, 4])
n_·    = [55, 67, 68]   (was [56, 67, 68])
```

### Step 2 - score all K topics

Now evaluate `(n_d,k + α) × (n_k,w + β) / (n_k + V×β)` for k = 0, 1, 2:

| k | document term (n_d,k + α) | word term (n_k,w + β)/(n_k + V×β) | product |
|---|---------------------------|-----------------------------------|---------|
| 0 | 3 + 0.1 = **3.1** | (3 + 0.01)/(55 + 0.15) = **0.054578** | 0.169193 |
| 1 | 5 + 0.1 = **5.1** | (6 + 0.01)/(67 + 0.15) = **0.089501** | 0.456456 |
| 2 | 8 + 0.1 = **8.1** | (4 + 0.01)/(68 + 0.15) = **0.058841** | 0.476610 |

### Step 3 - normalize and sample

```
unnormalized = [0.169193, 0.456456, 0.476610]
normalized   = [0.1535,   0.4141,   0.4324  ]     (divide by the sum, 1.102259)
```

A topic is drawn from that 3-way distribution — **not** the argmax. Sampling rather than
maximizing is what lets the chain escape a bad start. At this point the model is nearly
undecided, exactly as you would expect after one random pass.

### Step 4 - put the token back and move on

The chosen topic's three counters are incremented and the sampler moves to the next token.
That is all one Gibbs "sweep" does, 191 times over, 100 times in a row.

### The same token after 100 sweeps

Once the chain has settled, document 0 is entirely sports and topic 2 has absorbed every
"game":

```
k=0: (0 + 0.1) * (0  + 0.01)/(66 + 0.15) = 0.00001512
k=1: (0 + 0.1) * (0  + 0.01)/(65 + 0.15) = 0.00001535
k=2: (16 + 0.1) * (13 + 0.01)/(59 + 0.15) = 3.54118343
normalized = [0.0000, 0.0000, 1.0000]
```

The two factors now **agree**: document 0 has no tokens in topics 0 or 1, and topics 0 and
1 have never seen the word "game". Both terms go to (nearly) zero together, and the
sampler picks topic 2 with probability ~1. The corresponding θ row is
`[0.006, 0.006, 0.988]` — the smoothing from α is the only reason it is not exactly
`[0, 0, 1]`.

**This is the whole algorithm.** Everything else in `_25_lda.py` is bookkeeping around
this one table of numbers.

## Parameters Explained

### n_components (Number of Topics)

Controls how many topics to discover.

**Small (2-10):**
- Broad, high-level topics
- Good for small corpora or overview analysis
- Less computational cost

**Medium (10-50):**
- Balanced granularity (recommended)
- Good for most use cases
- Topics are interpretable and specific

**Large (50-200):**
- Fine-grained topics
- Good for very large corpora
- May have redundant or overly specific topics

**Rule of thumb:** Start with `sqrt(n_documents)` or use the elbow method with perplexity.

### alpha (Document-Topic Prior)

Controls how many topics each document can discuss.

**Small alpha (0.01-0.1):**
- Sparse topic distribution
- Each document focuses on few topics
- Good for specialized documents

**Medium alpha (0.1-1.0):**
- Balanced (default 0.1)
- Documents can discuss several topics
- Good general-purpose setting

**Large alpha (1.0-10.0):**
- Uniform topic distribution
- Documents spread across many topics
- Use for very diverse documents

**Formula:** `alpha = 50 / n_components` is a common default.

> **Caveat:** that is the Griffiths & Steyvers (2004) heuristic, tuned for *long*
> documents (thousands of tokens each). For `n_components=10` it gives α = 5.0, which
> the table just above classifies as "documents spread across many topics" — the
> opposite of what most people want. For the short documents used throughout this guide
> the sparser default `alpha=0.1` is the better starting point. Use `50/K` only when
> your documents are genuinely long.

### beta (Topic-Word Prior)

Controls how many words can be used to represent each topic.

**Small beta (0.01-0.1):**
- Sparse word distribution
- Topics focus on few distinctive words
- Better topic interpretability (recommended)

**Medium beta (0.1-1.0):**
- Balanced word distribution
- Topics use moderate vocabulary

**Large beta (1.0-10.0):**
- Uniform word distribution
- Topics use many words
- Less distinctive topics

**Typical:** `beta = 0.01` works well for most cases.

### max_iter (Iterations)

Number of Gibbs sampling iterations.

**Minimum (50-100):**
- Fast but may not converge
- Use for quick experimentation

**Recommended (100-500):**
- Good balance of quality and speed
- Sufficient for most datasets

**High Quality (500-1000):**
- Better convergence
- Use for final models or large corpora
- Monitor perplexity to check convergence

### burn_in and sample_lag (Posterior Averaging)

Two optional parameters, both `0` by default, that replace the single final Gibbs draw
with an average over several post-burn-in sweeps. See
[MCMC Hygiene](#mcmc-hygiene-burn-in-mixing-and-label-switching) for why this matters.

**`burn_in` (int, default 0):** number of opening sweeps to throw away. The chain starts
from a random topic assignment, which is not a sample from anything. Only takes effect
when `sample_lag > 0`. Typical: 20-50% of `max_iter`.

**`sample_lag` (int, default 0):** collect and average the count matrices every this many
sweeps after `burn_in`. `0` disables averaging and reports the final state only (the
backward-compatible default). Typical: 5-10 — consecutive sweeps are highly correlated,
so lag 1 collects near-duplicates.

```python
lda = LatentDirichletAllocation(
    n_components=5, max_iter=200,
    burn_in=100,      # discard the first 100 sweeps
    sample_lag=10,    # then average sweeps 100, 110, 120, ... 190
    random_state=42
)
lda.fit(X)
print(lda.n_gibbs_samples_)   # -> 10 states were averaged
```

Averaging costs no extra sweeps, so it is close to free. On a corpus with genuinely
ambiguous documents it visibly changes θ: in the `__main__` demo's Example 2 the largest
per-entry change is 0.0198.

**One asymmetry to know about.** `transform()` folds documents in with `min(max_iter, 50)`
sweeps, not `max_iter`, because fold-in is meant to be cheap. It therefore scales the
schedule by that same ratio before collecting, instead of taking `burn_in` literally: with
`max_iter=200, burn_in=100, sample_lag=10` the fold-in runs 50 sweeps, discards the first
25 (the same 50%), collects every 3rd one after that, and averages 9 states against the 10
that `fit` averaged. Without the rescaling a `burn_in` of 100 would swallow all 50 fold-in
sweeps and the "average" would quietly be the single final state.

## Code Example

```python
import numpy as np
# Run this from inside the "25. LDA" folder (the folder name contains a space and
# a dot, so it is not importable as a package from the repo root). The Quick Start
# section above is fully self-contained if you would rather paste the class in.
from _25_lda import LatentDirichletAllocation

# Example: News article analysis
# Assume we have document-term matrix X
# X[i, j] = count of word j in document i
# Counts must be raw integers (CountVectorizer), never TF-IDF.

# Vocabulary (for interpretation)
vocabulary = ['game', 'team', 'player', 'computer', 'code', 
              'food', 'recipe', 'cook', 'market', 'trade']

# Document-term matrix (3 documents, 10 words)
X = np.array([
    [5, 4, 3, 0, 0, 0, 0, 0, 0, 0],  # Sports document
    [0, 0, 0, 5, 4, 0, 0, 0, 0, 0],  # Tech document
    [0, 0, 0, 0, 0, 4, 5, 3, 0, 0],  # Food document
    # ... add more documents here; three rows is enough to run ...
])

# Fit LDA model
lda = LatentDirichletAllocation(
    n_components=3,      # Discover 3 topics
    max_iter=100,        # 100 iterations
    alpha=0.1,           # Sparse document-topic
    beta=0.01,           # Sparse topic-word
    random_state=42,     # Reproducibility
    verbose=1            # Show progress
)

# Fit and get document-topic distributions
doc_topics = lda.fit_transform(X)

# Display topics
print("Discovered Topics:")
top_words = lda.get_top_words(vocabulary, n_top_words=5)
for i, words in enumerate(top_words):
    print(f"Topic {i}: {', '.join(words)}")

# Display document topics
print("\nDocument-Topic Distribution:")
print(doc_topics)

# Transform new documents
X_new = np.array([[3, 4, 2, 0, 0, 0, 0, 0, 0, 0]])  # New sports doc
new_topics = lda.transform(X_new)
print(f"\nNew document topics: {new_topics}")
```

## Understanding the Code

`_25_lda.py` is one class, `LatentDirichletAllocation`, and every method maps to one step
of the math above. Read them in this order.

### The three count tables

Everything hangs off three arrays created in `_initialize_parameters`:

| Attribute | Shape | Symbol | Meaning |
|-----------|-------|--------|---------|
| `_doc_topic_count` | (D, K) | n_d,k | tokens of document d assigned to topic k |
| `_topic_word_count` | (K, V) | n_k,w | times word w is assigned to topic k |
| `_topic_count` | (K,) | n_k | total tokens assigned to topic k |
| `_topic_assignments` | list of lists | z | the current topic of every individual token |

`_topic_assignments[d]` is a list of `(word_index, topic)` pairs — one entry per **token**,
so a document with `X[d, w] = 5` contributes five entries for word `w`. That list *is* the
state of the Markov chain.

### Method by method

| Method | What it does | Formula it implements |
|--------|--------------|-----------------------|
| `_check_X(X)` | Accepts lists or arrays, promotes a 1-D vector to one document, rejects negatives, NaNs and fractional (TF-IDF) values | — |
| `_expand_priors(V)` | Turns scalar `alpha`/`beta` into vectors `_alpha_vec` (K,) and `_beta_vec` (V,), and precomputes `_beta_sum = Σᵥ βᵥ` | the Dirichlet normalizer |
| `_check_is_fitted(caller)` | Raises a readable "not fitted yet" message instead of a `NoneType` crash | — |
| `_initialize_parameters(X)` | Assigns every token a uniformly random topic and fills the three count tables | Step 1 of Algorithm Steps |
| `_sample_topic(d, w)` | Scores all K topics for one token and draws one | `(n_d,k + α)(n_k,w + β)/(n_k + Σβ)` |
| `_gibbs_sampling_iteration(X)` | One full sweep: decrement, `_sample_topic`, increment, store | Step 2 |
| `_compute_distributions()` | Turns counts into probabilities; averages the collected count tables first when `sample_lag > 0` | `θ_d,k=(n_d,k+α)/(n_d+Σα)`, `φ_k,w=(n_k,w+β)/(n_k+Σβ)` |
| `_compute_perplexity(X, doc_topic_distr=None)` | Scores a corpus given its θ | `exp(-Σ c·log Σ_k φ θ / N)` |
| `fit(X)` | Validates, reseeds the private RNG, initializes, runs `max_iter` sweeps, optionally collects samples, computes θ and φ | Steps 1-3 |
| `transform(X)` | Folds new documents in against a **frozen** φ | `(n_d,k + α) · φ_k,w` |
| `fit_transform(X)` | `fit(X)` then returns `doc_topic_distr_` | — |
| `perplexity(X)` | Public, always-correct scorer: `transform(X)` first, then `_compute_perplexity` | held-out evaluation |
| `get_top_words(names, n)` | Sorts each row of `components_` and maps indices to words | topic interpretation |

### Fitted attributes

| Attribute | Shape | Meaning |
|-----------|-------|---------|
| `components_` | (K, V) | φ — each row is a probability distribution over the vocabulary and **sums to 1** |
| `doc_topic_distr_` | (D, K) | θ — each row sums to 1 |
| `n_features_` | int | vocabulary size V |
| `n_samples_` | int | number of training documents D |
| `n_gibbs_samples_` | int | how many Gibbs states were averaged (0 when `sample_lag=0`) |

### Two things that trip people up

**`transform` is not `fit_transform` restricted to new rows.** `fit_transform` returns the
θ estimated *during training*, when φ was still moving. `transform` holds φ fixed and only
re-samples the per-document counts, which is the right thing for held-out data but gives
slightly different numbers on the training corpus.

**The nested `for k in range(self.n_components)` loop in `_sample_topic` is deliberate.**
It could be one vectorized NumPy line, but written out it matches the formula term for
term, which is the point of this repository. It is also the reason throughput is about
60,000 token-samples per second — see [Performance Considerations](#performance-considerations).

## Practical Use Cases

### 1. News Article Categorization

```python
from sklearn.feature_extraction.text import CountVectorizer

# Example news articles
articles = [
    "The team won the championship game...",
    "New AI technology improves medical diagnosis...",
    "Stock market hits record high...",
    # ... more articles ...
]

# Convert to document-term matrix
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(articles).toarray()
vocabulary = vectorizer.get_feature_names_out()

# Fit LDA
lda = LatentDirichletAllocation(n_components=10, random_state=42)
doc_topics = lda.fit_transform(X)

# Print discovered topics
top_words = lda.get_top_words(vocabulary, n_top_words=10)
for i, words in enumerate(top_words):
    print(f"Topic {i}: {', '.join(words)}")
```

### 2. Customer Review Analysis

```python
# Analyze product reviews to find common themes

reviews = [
    "Great battery life and fast charging",
    "Screen is too small and quality is poor",
    "Excellent customer service and warranty",
    # ... many more reviews ...
]

# Preprocess and vectorize
vectorizer = CountVectorizer(max_features=500, ngram_range=(1, 2))
X = vectorizer.fit_transform(reviews).toarray()

# Discover review themes
lda = LatentDirichletAllocation(n_components=5, alpha=0.1, beta=0.01)
review_topics = lda.fit_transform(X)

# Find dominant topic for each review
dominant_topics = np.argmax(review_topics, axis=1)

# Group reviews by topic
for topic_id in range(5):
    print(f"\nReviews about Topic {topic_id}:")
    topic_reviews = [reviews[i] for i in range(len(reviews)) 
                     if dominant_topics[i] == topic_id]
    for review in topic_reviews[:3]:
        print(f"  - {review}")
```

### 3. Document Recommendation

```python
# Recommend similar documents based on topic similarity

def recommend_documents(query_doc_idx, doc_topics, n_recommendations=5):
    """Find similar documents based on topic distribution"""
    query_topics = doc_topics[query_doc_idx]
    
    # Compute cosine similarity
    similarities = []
    for i in range(len(doc_topics)):
        if i != query_doc_idx:
            sim = np.dot(query_topics, doc_topics[i])
            sim /= (np.linalg.norm(query_topics) * np.linalg.norm(doc_topics[i]) + 1e-10)
            similarities.append((i, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:n_recommendations]

# Get recommendations
recommendations = recommend_documents(0, doc_topics, n_recommendations=3)
print(f"Documents similar to document 0:")
for doc_idx, similarity in recommendations:
    print(f"  Document {doc_idx}: similarity = {similarity:.3f}")
```

### 4. Topic Evolution Over Time

```python
# Track how topics change over time (e.g., in news articles)

# Documents grouped by time period
docs_by_year = {
    2020: [...],  # documents from 2020
    2021: [...],  # documents from 2021
    2022: [...],  # documents from 2022
}

# Fit separate LDA for each year
topics_by_year = {}
for year, docs in docs_by_year.items():
    X = vectorizer.fit_transform(docs).toarray()
    lda = LatentDirichletAllocation(n_components=10)
    topics_by_year[year] = lda.fit_transform(X)

# Analyze topic trends
# (Compare topic proportions, identify emerging/declining topics)
```

## Data Preprocessing Guide

**LDA quality heavily depends on preprocessing!**

### Essential Preprocessing Steps:

#### 1. Lowercase and Remove Punctuation
```python
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)
```

#### 2. Remove Stop Words
```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Remove common words like 'the', 'is', 'and'
stop_words = list(ENGLISH_STOP_WORDS) + ['additional', 'custom', 'words']
```

#### 3. Lemmatization or Stemming
```python
# Convert words to base form: running → run, better → good
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word) for word in words]
```

#### 4. Remove Rare and Common Words
```python
vectorizer = CountVectorizer(
    max_df=0.9,      # Remove words in >90% of documents
    min_df=5,        # Remove words in <5 documents
    max_features=1000  # Keep top 1000 words
)
```

#### 5. Use Bigrams/Trigrams (Optional)
```python
# Capture phrases like "machine learning", "new york"
vectorizer = CountVectorizer(ngram_range=(1, 2))
```

### Complete Preprocessing Pipeline:

```python
from sklearn.feature_extraction.text import CountVectorizer
import re

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Preprocess documents
documents = [preprocess_text(doc) for doc in raw_documents]

# Vectorize with good settings
vectorizer = CountVectorizer(
    max_df=0.9,           # Remove very common words
    min_df=5,             # Remove very rare words
    max_features=5000,    # Vocabulary size
    stop_words='english', # Remove stop words
    ngram_range=(1, 2)    # Include bigrams
)

X = vectorizer.fit_transform(documents).toarray()
vocabulary = vectorizer.get_feature_names_out()
```

## Evaluation Metrics

### 1. Perplexity

Measures how well the model predicts held-out documents. **Lower is better.**

```
Perplexity = exp(-log-likelihood / total word count)
log-likelihood = Σ_d Σ_w count_d,w × log( Σ_k φ_k,w × θ_d,k )
```

Use the **public** `perplexity(X)` method. It calls `transform(X)` first, so it infers
fresh topic mixtures for the rows of `X` and works on any corpus, of any size, that shares
the training vocabulary:

```python
perplexity = lda.perplexity(X_test)
print(f"Perplexity: {perplexity:.2f}")
```

> The private `_compute_perplexity(X)` scores `X` against the **stored training** θ. It is
> the right thing only when `X` is the training corpus itself; on held-out data it would
> silently pair document 0's words with training-document 0's topic mixture. Prefer
> `perplexity(X)`.

**Interpretation:**
- Lower perplexity = better fit to data
- V (the vocabulary size) is the chance-level baseline: it is what a model that predicts
  every word uniformly scores. A fitted model should land well below it - the Quick Start
  reaches 5.49 against V = 15. It is not an upper bound: a confidently wrong model, one
  with a sharp phi on the wrong topic, can score above V.
- But: lower perplexity ≠ better interpretability
- Use as relative metric (compare different models)
- θ here is inferred from the same words being scored, so the number is optimistic in
  absolute terms. Compare models on the same corpus; do not read it as an absolute.

### 2. Topic Coherence

Measures semantic similarity of top words in each topic. **Higher is better.**

The **UMass** score is the easiest to compute from what you already have — it needs only
the document-term matrix, no external corpus. For a topic's top words ordered from most to
least probable, it sums the log conditional probability of each word given every
higher-ranked word:

```
C_UMass(topic) = Σ_{i=2..M} Σ_{j=1..i-1} log( (D(w_i, w_j) + 1) / D(w_j) )
```

where `D(w)` is the number of documents containing `w` and `D(w_i, w_j)` the number
containing both. A coherent topic's top words co-occur, so the ratio stays near 1 and the
logs stay near 0; an incoherent topic's words never co-occur and the logs go sharply
negative.

```python
def umass_coherence(components, X, n_top_words=5):
    """UMass topic coherence for each row of components (higher = better)."""
    binary = (X > 0).astype(int)          # does document d contain word w?
    doc_freq = binary.sum(axis=0)         # D(w)
    doc_freq = np.maximum(doc_freq, 1)    # a word absent from the corpus would divide by 0
    co_doc_freq = binary.T @ binary       # D(w_i, w_j)

    scores = []
    for phi_k in components:
        top = np.argsort(phi_k)[::-1][:n_top_words]   # most probable first
        total = 0.0
        for i in range(1, len(top)):
            for j in range(i):                        # w_j is the MORE probable word
                total += np.log((co_doc_freq[top[i], top[j]] + 1.0) / doc_freq[top[j]])
        scores.append(total)
    return np.array(scores)


scores = umass_coherence(lda.components_, X, n_top_words=5)
for i, (s, words) in enumerate(zip(scores, lda.get_top_words(vocabulary, 5))):
    print(f"Topic {i}: coherence={s:7.3f}  {', '.join(words)}")
print(f"Mean coherence: {scores.mean():.3f}")
```

On the Quick Start corpus this prints:

```
Topic 0: coherence=  1.339  software, tech, code, computer, data
Topic 1: coherence=  1.339  recipe, dish, taste, food, cook
Topic 2: coherence=  1.627  game, player, team, score, win
Mean coherence: 1.435
```

Three random topic rows over the same corpus - `rows = np.random.default_rng(0).random((3, 15))`
with each row divided by its sum - average **-7.463** over those three rows. That is a
single seeded draw, not a stable constant: over 1,000 such draws the mean is **-5.8** and
individual draws range from -8.1 to -1.7. Even the luckiest of them is more than three
points below the fitted model's 1.435, so the gap between a real fit and noise is
enormous. (UMass is usually negative on realistic corpora;
it comes out positive here only because this toy corpus has near-perfect co-occurrence and
tiny document frequencies.)

**Types:**
- C_v: Based on word co-occurrence and semantic similarity
- C_uci: Based on pointwise mutual information (PMI)
- C_npmi: Normalized PMI

### 3. Human Evaluation

**Most important:** Do the topics make sense to humans?

**Check:**
- Are top words semantically related?
- Can you name each topic?
- Are topics distinct from each other?
- Do documents cluster sensibly?

## MCMC Hygiene: Burn-in, Mixing and Label Switching

Collapsed Gibbs sampling is a **Markov chain**, not an optimizer. It does not converge to
an answer; it wanders around the posterior forever. Three consequences that every LDA user
runs into:

### 1. The first sweeps are not samples — burn in

The chain starts from a uniformly random topic assignment, which is nowhere near the
posterior. The first sweeps are the chain *walking towards* the high-probability region,
and their counts are junk. The standard fix is to discard them:

```python
lda = LatentDirichletAllocation(n_components=10, max_iter=200,
                                burn_in=100, sample_lag=10, random_state=42)
```

**By default (`sample_lag=0`) this implementation reports the single final Gibbs state.**
That is one draw from the posterior rather than an average over several. It is also
exactly what Griffiths & Steyvers (2004) did — they report "a single sample taken after
2,000 iterations of Gibbs sampling". Setting `sample_lag > 0` averages the post-burn-in
count tables instead, which is ordinary MCMC variance reduction rather than a fidelity
fix. Heinrich's widely copied `LdaGibbsSampler` averages the same sweeps but accumulates
the per-sweep *normalized* θ and φ, the Rao-Blackwellized estimator of the posterior mean
E[φ | w] = Σ_z P(z|w) (n_k,w + β_w)/(n_k + Σβ); averaging counts
and adding the prior once, as this file does, gives the identical answer for θ and a
very slightly different one for φ (the docstring of `_compute_distributions` derives the
difference and measures it).

### 2. Consecutive sweeps are correlated — use a lag

Two adjacent sweeps differ by a handful of token reassignments, so averaging every sweep
mostly averages the same state with itself. Collecting one state every 5-10 sweeps gives
you nearly independent samples for the same cost. `n_gibbs_samples_` reports how many
were actually collected.

Averaging is close to free (it costs one array addition per collected sweep) and cuts the
variance of the estimate. In the `__main__` demo's Example 2 the largest per-entry change
to θ is 0.0198 — small there, because those documents are nearly unambiguous. Where the
posterior is genuinely spread out it moves much more: on a corpus of overlapping 60/40
topic blends, `transform` with averaging turned on differs from the single-final-state
answer by up to 0.11 per entry. Lower variance is not the same as a better answer on any
one run, though: across the eight corpus seeds of the benchmark below, the averaged φ was
closer to the planted truth on 5 of 8. Treat it as variance reduction, not a free win.

### 3. Topic numbers are meaningless — label switching

Nothing in the model distinguishes "topic 0" from "topic 2". A different `random_state`
gives the same topics under different numbers. In the Quick Start, sports lands on topic 2;
with another seed it might be topic 0. Practical rules:

- **Never** compare topic *indices* across two fits. Compare top-word lists.
- To score against ground truth, first map each true class to the topic its members
  mostly chose — that is exactly what the `group_to_topic` step in the Quick Start does.
- Report topics by their top words, never as "topic 3".
- This is also the limit of `sample_lag` averaging. It is safe only *within* one chain
  that keeps its topics straight; if a chain relabels topic 0 as topic 2 halfway through,
  the average blends two different topics. Griffiths & Steyvers make the same point about
  separate samples: their estimates "cannot be combined across samples for any analysis
  that relies on the content of specific topics". Never average across seeds.

### How do I know the chain has settled?

Fit with `verbose=2` and watch perplexity. It should fall steeply, then flatten:

```python
lda = LatentDirichletAllocation(n_components=10, max_iter=300, verbose=2)
lda.fit(X)
# Iteration 20/300, Perplexity: ...
# Iteration 40/300, Perplexity: ...
```

If perplexity is still falling at the last iteration, raise `max_iter`. If it flattened
long ago, you can lower it. Note that a *flat* perplexity means the chain is mixing in a
stable region — it does not prove the chain found the global mode. Multiple restarts with
different seeds are the honest check.

## Common Issues and Solutions

### Issue 1: Topics Not Interpretable

**Problem:** Topics contain random or unrelated words

**Solutions:**
- Improve preprocessing (remove stop words, rare words)
- Adjust n_components (try fewer or more topics)
- Decrease beta for more focused topics
- Increase max_iter for better convergence
- Check if vocabulary makes sense

### Issue 2: All Topics Similar

**Problem:** Topics are redundant or nearly identical

**Solutions:**
- Decrease n_components (too many topics)
- Lower beta (make topics more sparse)
- Improve document preprocessing
- Ensure corpus has sufficient diversity

### Issue 3: Documents Spread Across All Topics

**Problem:** Each document has uniform topic distribution

**Solutions:**
- Lower alpha (make documents focus on fewer topics)
- Increase max_iter (model hasn't converged)
- Check if documents are too short
- Verify preprocessing didn't remove too much information

### Issue 4: Slow Convergence

**Problem:** Model takes too long or doesn't converge

**Solutions:**
- Reduce vocabulary size (use max_features)
- Decrease max_iter for experimentation
- Use smaller alpha and beta
- Consider using optimized LDA libraries for large corpora

### Issue 5: Topics Dominated by Common Words

**Problem:** Topics show words like "said", "would", "also"

**Solutions:**
- Improve stop word removal
- Raise `max_df` filtering (drop words present in >90% of documents) — but keep raw
  integer counts. LDA is a multinomial model over word *tokens*, so TF-IDF weights are not
  valid input; this implementation raises a `ValueError` on fractional values rather than
  truncating them to zero. Filter the vocabulary, do not reweight it.
- Increase min_df threshold
- Add domain-specific stop words

## Tips for Success

### 1. Start Simple

```python
# Good first attempt
lda = LatentDirichletAllocation(
    n_components=10,     # Start with ~10 topics
    max_iter=100,        # 100 iterations is usually enough
    alpha=0.1,           # Sparse documents
    beta=0.01,           # Sparse topics
    random_state=42      # Reproducibility
)
```

### 2. Experiment with Topic Numbers

Try different values and evaluate:

```python
# Shuffle first, then split: the two slices must not overlap.
rng = np.random.default_rng(0)
X = X[rng.permutation(len(X))]
n_train = int(0.75 * len(X))
X_train, X_test = X[:n_train], X[n_train:]

for n_topics in [5, 10, 20, 30]:
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_train)
    perplexity = lda.perplexity(X_test)      # held-out, disjoint from X_train
    print(f"{n_topics} topics: perplexity = {perplexity:.2f}")
```

Score on a **held-out** split, not on the training corpus: training perplexity falls
monotonically as you add topics, so it would always pick the largest `n_topics` you try.

### 3. Visualize Topics

```python
import matplotlib.pyplot as plt

# Visualize document-topic distribution
plt.figure(figsize=(10, 6))
plt.imshow(doc_topics.T, aspect='auto', cmap='YlOrRd')
plt.xlabel('Documents')
plt.ylabel('Topics')
plt.colorbar(label='Topic Proportion')
plt.title('Document-Topic Distribution')
plt.show()
```

### 4. Interpret Topics with Top Documents

Don't just look at top words; examine documents with high topic proportion:

```python
# Find documents most representative of topic k
topic_k = 0
top_docs = np.argsort(doc_topics[:, topic_k])[::-1][:5]

print(f"Top documents for Topic {topic_k}:")
for doc_idx in top_docs:
    print(f"  Document {doc_idx}: {documents[doc_idx][:100]}...")
```

### 5. Use Domain Knowledge

- Add domain-specific stop words
- Choose n_components based on expected themes
- Manually label topics after discovery
- Validate results with domain experts

## LDA vs Other Methods

### LDA vs LSA (Latent Semantic Analysis)

| Aspect | LDA | LSA |
|--------|-----|-----|
| **Model Type** | Probabilistic | Algebraic (SVD) |
| **Interpretability** | ✓ Clear probabilistic meaning | ✗ Less interpretable |
| **Topics** | Distributions over words | Linear combinations |
| **Speed** | Slower (iterative sampling) | Faster (matrix decomposition) |
| **Sparsity** | ✓ Sparse representations | Continuous values |
| **Best For** | Topic modeling, text mining | Dimensionality reduction |

### LDA vs NMF (Non-negative Matrix Factorization)

| Aspect | LDA | NMF |
|--------|-----|-----|
| **Foundation** | Generative probabilistic | Matrix factorization |
| **Parameters** | Dirichlet priors | None |
| **Convergence** | Random (sampling) | Deterministic |
| **Flexibility** | Document and topic priors | Simpler model |
| **Interpretation** | Probabilistic | Parts-based |

### When to Use Each:

- **LDA**: Need probabilistic model, want interpretable topics, have medium-sized corpus
- **LSA**: Need fast dimensionality reduction, working with very large corpus
- **NMF**: Want deterministic results, need fast computation, parts-based representation

## Advanced Topics

### 1. Hyperparameter Tuning

Learn optimal α and β from data (not implemented here):

```python
# Hierarchical Dirichlet Process (HDP)
# Automatically determines number of topics
```

### 2. Supervised LDA (sLDA)

Incorporate document labels to guide topic discovery:

```python
# Include response variable (e.g., rating, category)
# Topics become predictive of responses
```

### 3. Dynamic Topic Models

Model topic evolution over time:

```python
# Each time slice has its own topic distributions
# Topics evolve smoothly across time
```

### 4. Correlated Topic Models (CTM)

Allow topic correlations (LDA assumes independence):

```python
# Replace Dirichlet with logistic normal
# Captures topic co-occurrence patterns
```

## Performance Considerations

### Time Complexity

- **Initialization**: O(D × V + N) where D = documents, V = vocabulary size,
  N = total word tokens. `_initialize_parameters` scans every cell of the document-term
  matrix (D × V) and creates one assignment per token (N).
- **Per iteration**: O(N × K) where K = number of topics — `_sample_topic` loops over
  all K topics for each of the N tokens
- **Total**: O(D × V + iterations × N × K)

### Measured Throughput

The K-loop in `_sample_topic` is written out longhand for readability, so this is a pure
Python sampler. Each corpus below is one `rng.multinomial(L, np.full(V, 1/V))` draw per
document with `rng = np.random.default_rng(0)` and L = 15, 100, 80 respectively, fitted
with `alpha=0.1, beta=0.01, random_state=42`. Token and sweep counts are therefore exact;
the seconds are the best of three runs on one desktop machine (Python 3.13.9, NumPy 2.3.5)
and will not reproduce on different hardware:

| Corpus | Tokens | Sweeps | Time | Rate |
|--------|--------|--------|------|------|
| D=60, V=30, K=3 | 900 | 100 | 1.2 s | ~74k token-samples/s |
| D=200, V=200, K=10 | 20,000 | 20 | 7.6 s | ~53k token-samples/s |
| D=500, V=500, K=10 | 40,000 | 10 | 7.2 s | ~55k token-samples/s |

Roughly **60,000 token-samples per second**. A 1,000-document corpus averaging 100 tokens
each, with `max_iter=100`, is 10 million samples — about three minutes. That is fine for
learning and prototyping and hopeless for production; see the library recommendations in
[Implementation Notes](#implementation-notes).

### Space Complexity

- **Count matrices**: O(D × K + K × V) where D = documents
- **Topic assignments**: O(N)
- **Total**: O(D × K + K × V + N)

### Scaling Tips

1. **For large vocabulary**: Use max_features to limit vocabulary
2. **For many documents**: Consider mini-batch LDA or online LDA
3. **For long documents**: Sample or truncate very long documents
4. **For better quality**: Increase max_iter and use multiple random restarts

## Further Reading

### Original Papers
- **Blei, Ng, Jordan (2003)**: "Latent Dirichlet Allocation"
  - Original LDA paper, highly cited and readable
  - Introduces the generative model and inference

### Tutorials and Guides
- **Griffiths & Steyvers (2004)**: "Finding Scientific Topics", PNAS 101(suppl. 1)
  - The collapsed Gibbs sampler this file implements, and the source of the
    "single sample after 2,000 iterations" estimate quoted above
- **Heinrich (2008)**: "Parameter Estimation for Text Analysis"
  - Full derivation of the collapsed conditional, and the reference
    `LdaGibbsSampler` that averages the per-sweep normalized θ and φ
- **"Introduction to Probabilistic Topic Models"** (Blei, 2012)
  - Comprehensive overview of topic modeling
- **"Topic Modeling: Beyond Bag-of-Words"** (Wallach, 2006)
  - Extensions and improvements to LDA

### Practical Guides
- **Scikit-learn LDA Documentation**
  - Practical implementation and examples
- **Gensim LDA Tutorial**
  - Popular Python library for topic modeling

### Applications
- **Digital Humanities**: Analyzing historical documents
- **Computational Biology**: Gene expression analysis
- **Social Sciences**: Survey and interview analysis
- **Business Intelligence**: Customer feedback mining

## Summary

**LDA is a powerful probabilistic model for discovering hidden topics in text collections.**

**Key takeaways:**

1. ✓ **Unsupervised**: No labels needed, discovers topics automatically
2. ✓ **Interpretable**: Topics are word distributions, easy to understand
3. ✓ **Flexible**: Works with any discrete data (not just text)
4. ✓ **Probabilistic**: Provides uncertainty estimates
5. ✓ **Educational**: this pure-Python Gibbs sampler runs at roughly 60,000
   token-samples per second, which comfortably handles small to medium corpora. Use
   gensim or scikit-learn for large document collections.

**Best practices:**

- **Preprocess carefully**: Stop words, stemming, rare word removal
- **Start simple**: 10 topics, alpha=0.1, beta=0.01
- **Iterate**: Try different n_components, evaluate interpretability
- **Validate**: Check topic coherence and human evaluation
- **Visualize**: Plot topics and document distributions

**Remember:** LDA quality depends heavily on preprocessing and hyperparameter choice. Always validate that discovered topics make sense!

---

## Implementation Notes

This implementation uses **Collapsed Gibbs Sampling** for inference, which is:
- Conceptually clear and educational
- Relatively simple to implement
- Effective for small to medium corpora

For production use with large corpora, consider:
- **Scikit-learn**: `sklearn.decomposition.LatentDirichletAllocation`
- **Gensim**: `gensim.models.LdaModel` or `gensim.models.LdaMulticore`
- **Mallet**: Java-based, very fast and high-quality

### Comparing against scikit-learn

Two differences will make a side-by-side comparison look broken if you are not expecting
them:

**1. Different inference algorithm.** scikit-learn solves the same generative model with
**variational Bayes**, not collapsed Gibbs sampling. Both approximate the same posterior;
neither is "the" answer, and their perplexity numbers are not directly comparable
(scikit-learn reports a variational-bound perplexity, this file reports
`exp(-Σ c·log Σ_k φθ / N)` from a point estimate).

**2. Different `components_` scaling.** Our `components_` rows are probability
distributions — each sums to exactly 1. scikit-learn stores the *unnormalized* variational
parameter λ, whose rows sum to hundreds or thousands. Normalize before comparing:

```python
sk_phi = sk_lda.components_ / sk_lda.components_.sum(axis=1, keepdims=True)
```

Topic *order* also differs (label switching), so align by permuting topics before you
compare anything.

Once both are normalized and aligned, the two agree closely. Here is the planted-truth
benchmark, spelled out completely so you can re-run it rather than take the numbers on
trust:

```python
import numpy as np

# K=3 topics over V=30 words with disjoint 10-word supports; D=60 documents of
# 50 tokens, 20 per topic, each a 75/25 blend of its own topic and the next one.
K, V, D, L = 3, 30, 60, 50
phi_true = np.zeros((K, V))
for k in range(K):
    phi_true[k, k * 10:(k + 1) * 10] = 0.1
labels = np.repeat(np.arange(K), D // K)
rng = np.random.default_rng(0)
X = np.array([rng.multinomial(L, 0.75 * phi_true[k] + 0.25 * phi_true[(k + 1) % K])
              for k in labels])

# ours   : LatentDirichletAllocation(n_components=3, max_iter=100,
#              alpha=0.1, beta=0.01, random_state=42)
# sklearn: LatentDirichletAllocation(n_components=3, max_iter=100, doc_topic_prior=0.1,
#              topic_word_prior=0.01, learning_method="batch", random_state=42)
# Topics come out in a different order in each fit, so align both to phi_true first.
```

| Model | mean \|φ̂ − φ_true\| per entry | dominant-topic agreement with planted labels | fit time |
|-------|-------------------------------|----------------------------------------------|----------|
| This file (collapsed Gibbs, `max_iter=100`) | 0.01539 | 1.000 | ~4 s |
| This file (`burn_in=50, sample_lag=5`) | 0.01521 | 1.000 | ~4 s |
| scikit-learn (variational Bayes) | 0.01574 | 1.000 | ~0.4 s |

The φ and agreement columns are deterministic — same seeds, same digits on any machine.
The fit times are not: on a busy machine the same three fits measured 14 s / 14 s / 1.4 s.
The *ratio* is the durable part.

Head to head, after aligning both to the planted truth: mean per-entry difference between
the two θ matrices is **0.0040** and per-document dominant-topic agreement is **100%**.
The top-5 word lists agree exactly on one topic and differ on the other two only where
words are near-tied (0.0761 vs 0.0750 for the fifth slot of one topic; a swap of two words
both at 0.0808 in another). The from-scratch implementation is algorithmically equivalent
— it is just roughly 10x slower here. Sample averaging (`burn_in`/`sample_lag`) costs
essentially nothing: it reuses the same sweeps.

The three φ errors are within noise of each other, so do not read the ordering as a
ranking. What does hold across corpus seeds `default_rng(0)` through `default_rng(7)`:
dominant-topic agreement is 1.000 for both implementations on all eight, and this file's
φ error is at or below scikit-learn's on all eight (0.0019-0.0173 against 0.0157-0.0177).
Gibbs sampling is not immune to a bad run, though: on an *easier* variant of this corpus
(documents drawn from a single topic instead of a blend) one seed leaves the chain in a
merged-topic mode — agreement 0.733, φ error 0.022 — while variational Bayes still
recovers all three. That is the multiple-restarts caveat from
[MCMC Hygiene](#mcmc-hygiene-burn-in-mixing-and-label-switching), measured.

### Simplification vs. canonical LDA

Three deliberate omissions, all documented in the class docstring too:

1. **Inference method.** Collapsed Gibbs sampling (Griffiths & Steyvers, PNAS 2004), not
   the variational Bayes of Blei et al. (2003) or the online/mini-batch variant that
   scikit-learn defaults to. This is a choice, not a shortcut — Gibbs is the more
   transparent algorithm to learn from.
2. **Fixed hyperparameters.** α and β never move. Mallet and gensim can re-estimate an
   asymmetric α from the data with a fixed-point / Newton update (Minka, 2000), which
   usually improves fit on real corpora. Not implemented here; you can pass an asymmetric
   α or β vector yourself, but the model will not learn one.
3. **Point estimate by default.** With `sample_lag=0` (the default) θ and φ come from the
   single final Gibbs state rather than an average over post-burn-in sweeps — the same
   choice Griffiths & Steyvers (2004) made for their own results. Set `burn_in` and
   `sample_lag` to average instead, which lowers the variance of the estimate; the
   averaged φ is a hair away from the exact posterior mean, and the docstring of
   `_compute_distributions` says exactly how far — see
   [MCMC Hygiene](#mcmc-hygiene-burn-in-mixing-and-label-switching).

**Our implementation demonstrates the core LDA algorithm** so you can understand how topic modeling actually works!

---

**Happy topic modeling!** 📚🔍📊
