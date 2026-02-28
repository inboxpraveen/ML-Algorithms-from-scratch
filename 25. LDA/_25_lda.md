# LDA - Latent Dirichlet Allocation

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

Gibbs sampling iteratively samples topic assignments for each word based on:

```
P(z_i = k | z_-i, w, α, β) ∝ (n_d,k + α) × (n_k,w + β) / (n_k + V×β)
```

Where:
- **n_d,k** = count of words in document d assigned to topic k (excluding current word)
- **n_k,w** = count of word w assigned to topic k (excluding current word)
- **n_k** = total count of words in topic k (excluding current word)
- **V** = vocabulary size

**Intuition:** Assign word w in document d to topic k based on:
1. How much document d likes topic k (first term)
2. How much topic k likes word w (second term)

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

## Code Example

```python
import numpy as np
from _25_lda import LatentDirichletAllocation

# Example: News article analysis
# Assume we have document-term matrix X
# X[i, j] = count of word j in document i

# Vocabulary (for interpretation)
vocabulary = ['game', 'team', 'player', 'computer', 'code', 
              'food', 'recipe', 'cook', 'market', 'trade']

# Document-term matrix (10 documents, 10 words)
X = np.array([
    [5, 4, 3, 0, 0, 0, 0, 0, 0, 0],  # Sports document
    [0, 0, 0, 5, 4, 0, 0, 0, 0, 0],  # Tech document
    [0, 0, 0, 0, 0, 4, 5, 3, 0, 0],  # Food document
    # ... more documents ...
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
```

```python
perplexity = lda._compute_perplexity(X_test)
print(f"Perplexity: {perplexity:.2f}")
```

**Interpretation:**
- Lower perplexity = better fit to data
- But: lower perplexity ≠ better interpretability
- Use as relative metric (compare different models)

### 2. Topic Coherence

Measures semantic similarity of top words in each topic. **Higher is better.**

```python
# Simplified coherence: average pairwise similarity of top words
def topic_coherence(top_words):
    # Use word co-occurrence in documents
    # Higher score = more coherent topic
    pass
```

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
- Use TF-IDF instead of raw counts
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
for n_topics in [5, 10, 20, 30]:
    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(X)
    perplexity = lda._compute_perplexity(X)
    print(f"{n_topics} topics: perplexity = {perplexity:.2f}")
```

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

- **Initialization**: O(N × V) where N = total words, V = vocabulary size
- **Per iteration**: O(N × K) where K = number of topics
- **Total**: O(iterations × N × K)

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
5. ✓ **Scalable**: Can handle large document collections

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

**Our implementation demonstrates the core LDA algorithm** so you can understand how topic modeling actually works!

---

**Happy topic modeling!** 📚🔍📊
