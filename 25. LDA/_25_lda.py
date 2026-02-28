import numpy as np

class LatentDirichletAllocation:
    """
    Latent Dirichlet Allocation (LDA) Implementation from Scratch
    
    LDA is a generative probabilistic model for collections of discrete data such as text.
    It discovers hidden topics in documents by modeling each document as a mixture of topics,
    and each topic as a mixture of words.
    
    Key Idea: "Documents are mixtures of topics, and topics are mixtures of words"
    
    Use Cases:
    - Topic Modeling: Discover hidden themes in document collections
    - Document Classification: Understand document content structure
    - Recommendation Systems: Find similar documents based on topics
    - Content Analysis: Analyze trends in news articles, social media
    - Information Retrieval: Improve search and document organization
    - Text Mining: Extract semantic patterns from large text corpora
    
    Key Concepts:
        Topics: Latent themes represented as distributions over words
        Document-Topic Distribution (θ): How much each topic appears in a document
        Topic-Word Distribution (φ): How likely each word is in a topic
        Dirichlet Prior: Hyperparameters that control distribution sparsity
        Gibbs Sampling: Iterative method to estimate posterior distributions
    """
    
    def __init__(self, n_components=10, max_iter=100, alpha=0.1, beta=0.01,
                 random_state=None, verbose=0):
        """
        Initialize the Latent Dirichlet Allocation model
        
        Parameters:
        -----------
        n_components : int, default=10
            Number of topics to discover
            - Small values (2-5): Few broad topics
            - Medium values (10-20): Balanced granularity (recommended)
            - Large values (50-100): Fine-grained topics
            Typical range: 5-50 depending on corpus size
            
        max_iter : int, default=100
            Maximum number of Gibbs sampling iterations
            - More iterations: Better convergence but slower
            - Minimum recommended: 50
            - Good quality: 100-500
            Typical values: 100-1000
            
        alpha : float or array-like, default=0.1
            Dirichlet prior for document-topic distribution
            - Small values (0.01-0.1): Sparse (few topics per document)
            - Large values (1.0-10.0): Uniform (many topics per document)
            - Can be a single float or array of length n_components
            Interpretation: Pseudo-count of topics in each document
            
        beta : float or array-like, default=0.01
            Dirichlet prior for topic-word distribution
            - Small values (0.01-0.1): Sparse (few words per topic)
            - Large values (1.0-10.0): Uniform (many words per topic)
            - Can be a single float or array of length vocabulary_size
            Interpretation: Pseudo-count of words in each topic
            
        random_state : int or None, default=None
            Random seed for reproducibility
            
        verbose : int, default=0
            Verbosity level
            - 0: Silent
            - 1: Show progress
            - 2: Show detailed information including perplexity
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose
        
        # Model parameters (learned during fit)
        self.components_ = None          # Topic-word distribution (φ)
        self.doc_topic_distr_ = None     # Document-topic distribution (θ)
        self.n_features_ = None          # Vocabulary size
        self.n_samples_ = None           # Number of documents
        
        # Internal state
        self._topic_assignments = None   # Topic assignment for each word
        self._doc_topic_count = None     # Count of topics in each document
        self._topic_word_count = None    # Count of words in each topic
        self._topic_count = None         # Total count of words per topic
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, X):
        """
        Initialize LDA parameters and topic assignments
        
        Randomly assign topics to each word occurrence in each document.
        This serves as the starting point for Gibbs sampling.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Document-term matrix (word counts)
        """
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        
        # Initialize count matrices
        self._doc_topic_count = np.zeros((n_samples, self.n_components))
        self._topic_word_count = np.zeros((self.n_components, n_features))
        self._topic_count = np.zeros(self.n_components)
        
        # Initialize topic assignments for each word occurrence
        self._topic_assignments = []
        
        for d in range(n_samples):
            doc_assignments = []
            for w in range(n_features):
                word_count = int(X[d, w])
                for _ in range(word_count):
                    # Randomly assign topic
                    topic = np.random.randint(0, self.n_components)
                    doc_assignments.append((w, topic))
                    
                    # Update counts
                    self._doc_topic_count[d, topic] += 1
                    self._topic_word_count[topic, w] += 1
                    self._topic_count[topic] += 1
            
            self._topic_assignments.append(doc_assignments)
    
    def _sample_topic(self, d, w):
        """
        Sample a new topic for word w in document d using collapsed Gibbs sampling
        
        This is the core of LDA inference. For each word, we sample a new topic
        based on the conditional probability:
        
        P(z_i = k | z_-i, w) ∝ (n_d,k + α) × (n_k,w + β) / (n_k + V×β)
        
        Where:
        - n_d,k: count of topic k in document d (excluding current word)
        - n_k,w: count of word w in topic k (excluding current word)
        - n_k: total count of words in topic k (excluding current word)
        - V: vocabulary size
        
        Parameters:
        -----------
        d : int
            Document index
        w : int
            Word index
            
        Returns:
        --------
        new_topic : int
            Newly sampled topic
        """
        # Compute probability for each topic
        probs = np.zeros(self.n_components)
        
        for k in range(self.n_components):
            # Document-topic component: P(topic k | document d)
            doc_topic_prob = self._doc_topic_count[d, k] + self.alpha
            
            # Topic-word component: P(word w | topic k)
            topic_word_prob = (self._topic_word_count[k, w] + self.beta)
            topic_word_prob /= (self._topic_count[k] + self.n_features_ * self.beta)
            
            # Combined probability
            probs[k] = doc_topic_prob * topic_word_prob
        
        # Normalize to get probability distribution
        probs /= probs.sum()
        
        # Sample new topic
        new_topic = np.random.choice(self.n_components, p=probs)
        
        return new_topic
    
    def _gibbs_sampling_iteration(self, X):
        """
        Perform one iteration of Gibbs sampling
        
        For each word occurrence in each document:
        1. Remove current topic assignment
        2. Sample new topic based on conditional probability
        3. Update counts with new assignment
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Document-term matrix
        """
        for d in range(self.n_samples_):
            for i, (w, old_topic) in enumerate(self._topic_assignments[d]):
                # Remove current topic assignment
                self._doc_topic_count[d, old_topic] -= 1
                self._topic_word_count[old_topic, w] -= 1
                self._topic_count[old_topic] -= 1
                
                # Sample new topic
                new_topic = self._sample_topic(d, w)
                
                # Update with new topic assignment
                self._doc_topic_count[d, new_topic] += 1
                self._topic_word_count[new_topic, w] += 1
                self._topic_count[new_topic] += 1
                
                # Store new assignment
                self._topic_assignments[d][i] = (w, new_topic)
    
    def _compute_distributions(self):
        """
        Compute final document-topic and topic-word distributions
        
        After Gibbs sampling converges, compute:
        - θ (theta): Document-topic distribution
        - φ (phi): Topic-word distribution
        """
        # Document-topic distribution (θ)
        # θ_d,k = (n_d,k + α) / (n_d + K×α)
        self.doc_topic_distr_ = self._doc_topic_count + self.alpha
        self.doc_topic_distr_ /= self.doc_topic_distr_.sum(axis=1, keepdims=True)
        
        # Topic-word distribution (φ) - stored as components_
        # φ_k,w = (n_k,w + β) / (n_k + V×β)
        self.components_ = self._topic_word_count + self.beta
        self.components_ /= self.components_.sum(axis=1, keepdims=True)
    
    def _compute_perplexity(self, X):
        """
        Compute perplexity of the model on data X
        
        Perplexity is a standard metric for evaluating topic models.
        Lower perplexity indicates better model fit.
        
        Perplexity = exp(-log-likelihood / total word count)
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Document-term matrix
            
        Returns:
        --------
        perplexity : float
            Model perplexity
        """
        log_likelihood = 0.0
        total_words = 0
        
        for d in range(X.shape[0]):
            for w in range(X.shape[1]):
                count = X[d, w]
                if count > 0:
                    # P(w|d) = Σ_k P(w|k) × P(k|d)
                    prob = np.dot(self.components_[:, w], self.doc_topic_distr_[d])
                    log_likelihood += count * np.log(prob + 1e-10)
                    total_words += count
        
        perplexity = np.exp(-log_likelihood / total_words)
        return perplexity
    
    def fit(self, X):
        """
        Fit the LDA model to data using Gibbs sampling
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Document-term matrix where X[i, j] is the count of word j in document i
            - n_samples: Number of documents
            - n_features: Vocabulary size
            - Values should be non-negative integers (word counts)
            
        Returns:
        --------
        self : object
            Fitted model
        """
        if self.verbose > 0:
            print(f"Fitting LDA with {self.n_components} topics...")
            print(f"Corpus: {X.shape[0]} documents, {X.shape[1]} vocabulary size")
        
        # Initialize parameters and topic assignments
        self._initialize_parameters(X)
        
        if self.verbose > 0:
            print(f"\nRunning Gibbs sampling for {self.max_iter} iterations...")
        
        # Gibbs sampling iterations
        for iteration in range(self.max_iter):
            self._gibbs_sampling_iteration(X)
            
            if self.verbose > 0 and (iteration + 1) % 20 == 0:
                if self.verbose > 1:
                    # Compute and show perplexity
                    self._compute_distributions()
                    perplexity = self._compute_perplexity(X)
                    print(f"Iteration {iteration + 1}/{self.max_iter}, Perplexity: {perplexity:.2f}")
                else:
                    print(f"Iteration {iteration + 1}/{self.max_iter}")
        
        # Compute final distributions
        self._compute_distributions()
        
        if self.verbose > 0:
            print(f"\nLDA fitting complete!")
            if self.verbose > 1:
                final_perplexity = self._compute_perplexity(X)
                print(f"Final perplexity: {final_perplexity:.2f}")
        
        return self
    
    def transform(self, X):
        """
        Transform documents to document-topic distribution
        
        For new documents, infer the topic distribution using the learned
        topic-word distributions (φ) with Gibbs sampling.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Document-term matrix
            
        Returns:
        --------
        doc_topic_distr : np.ndarray, shape (n_samples, n_components)
            Document-topic distribution for input documents
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before transform. Call fit() first.")
        
        n_samples = X.shape[0]
        
        # Initialize document-topic counts for new documents
        doc_topic_count = np.zeros((n_samples, self.n_components))
        topic_assignments = []
        
        # Initialize topic assignments
        for d in range(n_samples):
            doc_assignments = []
            for w in range(self.n_features_):
                word_count = int(X[d, w])
                for _ in range(word_count):
                    # Sample topic based on learned topic-word distribution
                    probs = self.components_[:, w] + 1e-10
                    probs /= probs.sum()
                    topic = np.random.choice(self.n_components, p=probs)
                    doc_assignments.append((w, topic))
                    doc_topic_count[d, topic] += 1
            
            topic_assignments.append(doc_assignments)
        
        # Gibbs sampling for new documents (fewer iterations)
        n_iter = min(self.max_iter, 50)
        for _ in range(n_iter):
            for d in range(n_samples):
                for i, (w, old_topic) in enumerate(topic_assignments[d]):
                    # Remove current assignment
                    doc_topic_count[d, old_topic] -= 1
                    
                    # Sample new topic
                    probs = (doc_topic_count[d] + self.alpha) * (self.components_[:, w] + 1e-10)
                    probs /= probs.sum()
                    new_topic = np.random.choice(self.n_components, p=probs)
                    
                    # Update assignment
                    doc_topic_count[d, new_topic] += 1
                    topic_assignments[d][i] = (w, new_topic)
        
        # Compute document-topic distribution
        doc_topic_distr = doc_topic_count + self.alpha
        doc_topic_distr /= doc_topic_distr.sum(axis=1, keepdims=True)
        
        return doc_topic_distr
    
    def fit_transform(self, X):
        """
        Fit the model and return document-topic distribution
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Document-term matrix
            
        Returns:
        --------
        doc_topic_distr : np.ndarray, shape (n_samples, n_components)
            Document-topic distribution
        """
        self.fit(X)
        return self.doc_topic_distr_
    
    def get_top_words(self, feature_names, n_top_words=10):
        """
        Get top words for each topic
        
        Parameters:
        -----------
        feature_names : list of str
            List of vocabulary words corresponding to feature indices
        n_top_words : int, default=10
            Number of top words to return per topic
            
        Returns:
        --------
        top_words : list of lists
            Top words for each topic
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before getting top words.")
        
        top_words = []
        for topic_idx in range(self.n_components):
            # Get indices of top words for this topic
            top_indices = np.argsort(self.components_[topic_idx])[::-1][:n_top_words]
            top_words.append([feature_names[i] for i in top_indices])
        
        return top_words


"""
========================================
EXAMPLE USAGE
========================================
"""

if __name__ == "__main__":
    print("=" * 70)
    print("LDA - Latent Dirichlet Allocation (Topic Modeling)")
    print("Educational Implementation")
    print("=" * 70)
    
    # Example 1: Basic LDA on Simple Document Collection
    print("\n" + "=" * 70)
    print("Example 1: LDA on Simple Document Collection")
    print("=" * 70)
    
    # Create a simple corpus (document-term matrix)
    # Documents: 10 documents, 15 words vocabulary
    # We'll create documents about 3 topics: sports, technology, food
    
    np.random.seed(42)
    
    # Vocabulary
    vocabulary = [
        'game', 'team', 'player', 'win', 'score',        # Sports words (0-4)
        'computer', 'software', 'code', 'data', 'tech',  # Tech words (5-9)
        'food', 'recipe', 'cook', 'taste', 'dish'        # Food words (10-14)
    ]
    
    # Create documents (each focused on different topics)
    X_simple = np.array([
        # Sports documents (docs 0-2)
        [5, 4, 3, 2, 3,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
        [4, 5, 4, 3, 4,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
        [3, 3, 5, 4, 3,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
        
        # Tech documents (docs 3-5)
        [0, 0, 0, 0, 0,  5, 4, 3, 4, 3,  0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,  4, 5, 4, 5, 4,  0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,  3, 4, 5, 3, 5,  0, 0, 0, 0, 0],
        
        # Food documents (docs 6-8)
        [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  5, 4, 3, 4, 3],
        [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  4, 5, 4, 3, 4],
        [0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  3, 4, 5, 5, 4],
        
        # Mixed document (doc 9)
        [2, 1, 1, 1, 0,  1, 2, 1, 0, 1,  1, 1, 0, 1, 2]
    ])
    
    print(f"Corpus shape: {X_simple.shape}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Sample words: {', '.join(vocabulary[:5])}...")
    
    # Fit LDA model
    print("\nFitting LDA with 3 topics...")
    lda = LatentDirichletAllocation(
        n_components=3,
        max_iter=100,
        alpha=0.1,
        beta=0.01,
        random_state=42,
        verbose=1
    )
    
    doc_topics = lda.fit_transform(X_simple)
    
    # Display discovered topics
    print("\n" + "-" * 70)
    print("DISCOVERED TOPICS (Top 5 words per topic):")
    print("-" * 70)
    
    top_words = lda.get_top_words(vocabulary, n_top_words=5)
    for topic_idx, words in enumerate(top_words):
        print(f"Topic {topic_idx}: {', '.join(words)}")
    
    # Display document-topic distribution
    print("\n" + "-" * 70)
    print("DOCUMENT-TOPIC DISTRIBUTION:")
    print("-" * 70)
    print("(Each row is a document, each column is a topic)")
    print(f"\n{np.round(doc_topics, 3)}")
    
    # Example 2: LDA with Different Hyperparameters
    print("\n" + "=" * 70)
    print("Example 2: Effect of Alpha and Beta on Topic Distribution")
    print("=" * 70)
    
    # Create larger corpus with more variation
    np.random.seed(42)
    n_docs = 20
    vocab_size = 20
    
    # Generate synthetic document-term matrix
    X_synthetic = np.random.poisson(2, size=(n_docs, vocab_size))
    
    print(f"\nSynthetic corpus: {n_docs} documents, {vocab_size} vocabulary")
    
    # Test different alpha values
    alphas = [0.01, 0.1, 1.0]
    
    for alpha_val in alphas:
        print(f"\n--- LDA with alpha={alpha_val} ---")
        lda_test = LatentDirichletAllocation(
            n_components=5,
            max_iter=50,
            alpha=alpha_val,
            beta=0.01,
            random_state=42,
            verbose=0
        )
        
        doc_topics_test = lda_test.fit_transform(X_synthetic)
        
        # Compute topic sparsity (how many topics are dominant per document)
        dominant_topics = np.sum(doc_topics_test > 0.2, axis=1)
        avg_dominant = np.mean(dominant_topics)
        
        print(f"Average dominant topics per document: {avg_dominant:.2f}")
        print(f"Document-topic distribution sample (first 3 docs):")
        print(np.round(doc_topics_test[:3], 3))
    
    # Example 3: Using LDA for Document Similarity
    print("\n" + "=" * 70)
    print("Example 3: Finding Similar Documents Using Topic Distributions")
    print("=" * 70)
    
    # Use the simple corpus from Example 1
    lda_similarity = LatentDirichletAllocation(
        n_components=3,
        max_iter=100,
        alpha=0.1,
        beta=0.01,
        random_state=42,
        verbose=0
    )
    
    doc_topics_sim = lda_similarity.fit_transform(X_simple)
    
    def cosine_similarity(v1, v2):
        """Compute cosine similarity between two vectors"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    
    # Find documents similar to document 0 (sports document)
    query_doc = 0
    similarities = []
    
    for i in range(len(doc_topics_sim)):
        if i != query_doc:
            sim = cosine_similarity(doc_topics_sim[query_doc], doc_topics_sim[i])
            similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nQuery: Document {query_doc} (Sports document)")
    print(f"Topic distribution: {np.round(doc_topics_sim[query_doc], 3)}")
    print("\nMost similar documents:")
    for doc_idx, sim in similarities[:3]:
        doc_type = "Sports" if doc_idx <= 2 else ("Tech" if doc_idx <= 5 else "Food")
        print(f"  Document {doc_idx} ({doc_type}): similarity = {sim:.3f}")
        print(f"    Topic distribution: {np.round(doc_topics_sim[doc_idx], 3)}")
    
    # Example 4: Transform New Documents
    print("\n" + "=" * 70)
    print("Example 4: Inferring Topics for New Documents")
    print("=" * 70)
    
    # Create new test documents
    X_test = np.array([
        # New sports document
        [4, 3, 4, 2, 3,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
        
        # New tech document
        [0, 0, 0, 0, 0,  4, 3, 4, 3, 4,  0, 0, 0, 0, 0],
        
        # New mixed document
        [1, 1, 0, 0, 0,  2, 1, 0, 1, 0,  1, 0, 1, 2, 1]
    ])
    
    print(f"Transforming {X_test.shape[0]} new documents...")
    
    # Transform using fitted model
    new_doc_topics = lda_similarity.transform(X_test)
    
    print("\nInferred topic distributions for new documents:")
    print(np.round(new_doc_topics, 3))
    
    doc_types = ["Sports", "Tech", "Mixed"]
    for i, doc_type in enumerate(doc_types):
        print(f"\n{doc_type} document:")
        print(f"  Topic distribution: {np.round(new_doc_topics[i], 3)}")
    
    # Practical Tips
    print("\n" + "=" * 70)
    print("PRACTICAL TIPS FOR USING LDA")
    print("=" * 70)
    
    tips = """
    1. CHOOSING NUMBER OF TOPICS (n_components):
       - Start with sqrt(n_documents) as initial guess
       - Too few: Topics too broad and mixed
       - Too many: Topics redundant and hard to interpret
       - Use perplexity or coherence metrics to evaluate
       - Try values: 5-10 for small corpus, 20-100 for large corpus
    
    2. SETTING ALPHA (document-topic prior):
       - Low alpha (0.01-0.1): Documents focus on few topics (sparse)
       - High alpha (1.0-10.0): Documents spread across many topics
       - Default 0.1 works well for most cases
       - Rule: alpha = 50/n_components is a good starting point
    
    3. SETTING BETA (topic-word prior):
       - Low beta (0.01-0.1): Topics focus on few words (sparse)
       - High beta (1.0-10.0): Topics use many words
       - Default 0.01 works well for most cases
       - Rule: beta = 0.01 for focused topics
    
    4. ITERATIONS (max_iter):
       - Minimum: 50 iterations
       - Recommended: 100-500 for good convergence
       - More iterations = better but slower
       - Monitor perplexity to check convergence
    
    5. DATA PREPROCESSING (CRITICAL):
       - Remove stop words (the, is, and, etc.)
       - Remove very rare words (appear in <5 documents)
       - Remove very common words (appear in >90% documents)
       - Lemmatize or stem words (running → run)
       - Use TF-IDF weighting or raw counts
    
    6. INTERPRETING TOPICS:
       - Look at top 10-20 words per topic
       - Examine top documents for each topic
       - Topics should be coherent and interpretable
       - If topics unclear, adjust n_components or priors
    
    7. EVALUATION:
       - Perplexity: Lower is better (but not always interpretable)
       - Topic coherence: Measures semantic similarity of top words
       - Human evaluation: Do topics make sense?
       - Held-out likelihood: Performance on test documents
    
    8. COMMON ISSUES:
       - Topics not coherent: Adjust n_components or improve preprocessing
       - All topics similar: Increase n_components or lower beta
       - Documents spread across all topics: Lower alpha
       - Slow convergence: Increase max_iter or simplify vocabulary
    """
    
    print(tips)
    
    print("\n" + "=" * 70)
    print("COMPARISON: LDA vs Other Topic Modeling Methods")
    print("=" * 70)
    
    comparison = """
    LDA vs LSA (Latent Semantic Analysis):
    + LDA: Probabilistic, interpretable topics
    + LDA: Better for word co-occurrence patterns
    - LSA: Faster, deterministic
    - LSA: Uses SVD, no probabilistic interpretation
    
    LDA vs NMF (Non-negative Matrix Factorization):
    + LDA: Theoretically grounded (generative model)
    + LDA: Handles document-topic uncertainty naturally
    - NMF: Faster, simpler optimization
    - NMF: Topics can be easier to interpret
    
    LDA vs Neural Topic Models:
    + LDA: Simpler, more interpretable
    + LDA: Works well on small-medium corpora
    - Neural: Better for very large corpora
    - Neural: Can incorporate external knowledge
    
    Best Use Cases for LDA:
    - Document classification and clustering
    - Exploratory analysis of text collections
    - Content recommendation systems
    - Trend analysis in news/social media
    - Academic paper organization
    - Understanding customer feedback themes
    """
    
    print(comparison)
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
