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
        Document-Topic Distribution (theta): How much each topic appears in a document
        Topic-Word Distribution (phi): How likely each word is in a topic
        Dirichlet Prior: Hyperparameters that control distribution sparsity
        Gibbs Sampling: Iterative method to estimate posterior distributions

    Why "collapsed"?
        The Dirichlet prior is conjugate to the multinomial likelihood, so theta and phi
        can be integrated out of the joint distribution analytically. What is left is a
        posterior over the discrete topic labels z alone. We therefore never sample
        theta or phi during inference - we only sample one integer per word token, and
        recover theta and phi at the end from the final (or averaged) counts. This is
        the "collapsed" Gibbs sampler of Griffiths & Steyvers (PNAS, 2004).

    Collapsed Gibbs conditional (the one formula this file is built around):

        P(z_i = k | z_-i, w) ~ (n_dk + alpha_k) * (n_kw + beta_w) / (n_k + sum(beta))

        n_dk = tokens in document d currently assigned to topic k
        n_kw = times word w is currently assigned to topic k
        n_k  = total tokens currently assigned to topic k
        All three counts EXCLUDE the token being resampled ("z_-i").
        Left factor  = "how much does this document like topic k?"
        Right factor = "how much does topic k like this word?"

        Implemented verbatim in _sample_topic(); the two estimators recovered from the
        final counts in _compute_distributions() are

            theta_dk = (n_dk + alpha_k) / (n_d + sum(alpha))
            phi_kw   = (n_kw + beta_w)  / (n_k + sum(beta))

    Simplifications vs. canonical LDA:
        - Inference is collapsed Gibbs sampling, not the variational Bayes used by
          scikit-learn. Both target the same model; see "Implementation Notes" in
          _25_lda.md for why the two `components_` attributes look different.
        - alpha and beta are fixed hyperparameters. Canonical implementations
          (Mallet, gensim) can additionally re-estimate an asymmetric alpha from the
          data with a Newton / fixed-point update (Minka, 2000). Not implemented here.
        - By default the estimator uses the single FINAL Gibbs state, which is
          what Griffiths & Steyvers (2004) themselves report ("a single sample
          taken after 2,000 iterations of Gibbs sampling"). Pass burn_in /
          sample_lag to average post-burn-in sweeps instead - standard MCMC
          variance reduction. See _compute_distributions for exactly what is
          averaged and how it differs from the textbook posterior mean.
    """
    
    def __init__(self, n_components=10, max_iter=100, alpha=0.1, beta=0.01,
                 random_state=None, verbose=0, burn_in=0, sample_lag=0):
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
            - Can be a single float (symmetric prior) or an array of length
              n_components (asymmetric prior, one pseudo-count per topic)
            Interpretation: Pseudo-count of topics in each document
            Typical range: 0.01 to 1.0 for short documents
            
        beta : float or array-like, default=0.01
            Dirichlet prior for topic-word distribution
            - Small values (0.01-0.1): Sparse (few words per topic)
            - Large values (1.0-10.0): Uniform (many words per topic)
            - Can be a single float (symmetric prior) or an array of length
              vocabulary_size (asymmetric prior, one pseudo-count per word)
            Interpretation: Pseudo-count of words in each topic
            Typical range: 0.001 to 0.1
            
        random_state : int or None, default=None
            Seed for a PRIVATE random generator (np.random.default_rng).
            The global numpy random stream is never touched, so fitting this
            model does not disturb the caller's own random numbers.
            fit() and transform() each restart from this seed, so repeated calls
            on the same object give identical results.
            Typical: any int; None means "different topics on every run"
            
        verbose : int, default=0
            Verbosity level
            - 0: Silent
            - 1: Show progress
            - 2: Show detailed information including perplexity

        burn_in : int, default=0
            Number of initial Gibbs sweeps to DISCARD before collecting samples.
            The chain starts from a random topic assignment, so the first sweeps
            are not draws from the posterior. Only has an effect when
            sample_lag > 0.
            - 0: no burn-in (use the single final state)
            - Higher values: safer but wastes sweeps
            Typical: 20-50% of max_iter (e.g. 50 when max_iter=100)

        sample_lag : int, default=0
            Collect (and average) the count matrices every `sample_lag` sweeps
            after burn_in. This turns the single high-variance final draw into an
            average over several post-burn-in sweeps - standard MCMC variance
            reduction. (What exactly is averaged, and how close it is to the
            textbook posterior mean, is spelled out in _compute_distributions.)
            - 0: disabled - use only the final state (backward-compatible default)
            - Higher values: less correlated samples, but fewer of them
            Typical: 5-10 (consecutive sweeps are highly autocorrelated, so 1 is
            wasteful rather than wrong)
            Averaging is only meaningful while the chain keeps its topics
            straight: nothing stops a long chain from relabelling topic 0 as
            topic 2 mid-run, and averaging across such a switch blends two
            different topics. Griffiths & Steyvers (2004) make the same point
            about samples from different chains - their estimates "cannot be
            combined across samples for any analysis that relies on the content
            of specific topics".
            After fit(), n_gibbs_samples_ reports how many states were averaged.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose
        self.burn_in = burn_in
        self.sample_lag = sample_lag
        
        # Model parameters (learned during fit)
        self.components_ = None          # Topic-word distribution (phi)
        self.doc_topic_distr_ = None     # Document-topic distribution (theta)
        self.n_features_ = None          # Vocabulary size
        self.n_samples_ = None           # Number of documents
        self.n_gibbs_samples_ = 0        # Gibbs states averaged (0 = final state only)
        
        # Internal state
        self._topic_assignments = None   # Topic assignment for each word
        self._doc_topic_count = None     # Count of topics in each document
        self._topic_word_count = None    # Count of words in each topic
        self._topic_count = None         # Total count of words per topic
        
        # Prior vectors, expanded from the scalar/array `alpha` and `beta` in fit()
        self._alpha_vec = None           # shape (n_components,)
        self._beta_vec = None            # shape (n_features_,)
        self._beta_sum = None            # sum_v beta_v  (Dirichlet normalizer)

        # Posterior-mean accumulators (used only when sample_lag > 0)
        self._doc_topic_sum = None
        self._topic_word_sum = None
        self._n_collected = 0

        # PRIVATE generator: seeding the global np.random stream would silently
        # change random numbers everywhere else in the caller's program.
        self._rng = np.random.default_rng(random_state)

    def _check_X(self, X):
        """
        Validate and standardise a document-term matrix

        Accepts a nested list or an ndarray, promotes a single document given as a
        1-D vector to shape (1, n_features), and rejects negative or non-finite
        counts. LDA counts word occurrences, so entries must be non-negative.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features) or (n_features,)
            Document-term matrix (word counts)

        Returns:
        --------
        X : np.ndarray, shape (n_samples, n_features), dtype float
        """
        X = np.asarray(X, dtype=float)

        # A single document may be passed as a flat vector
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_documents, vocabulary_size); got {X.ndim} dimensions."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values; word counts must be finite.")
        if np.any(X < 0):
            raise ValueError("X contains negative values; word counts must be non-negative.")
        if np.any(X != np.floor(X)):
            # LDA is a multinomial model over word TOKENS, so the sampler walks
            # int(X[d, w]) occurrences of each word. Fractional weights (TF-IDF!)
            # would be silently truncated to 0 and the document would vanish.
            raise ValueError(
                "X contains fractional values. LDA needs raw integer word counts "
                "(a CountVectorizer matrix), not TF-IDF or normalized weights."
            )

        return X

    def _expand_priors(self, n_features):
        """
        Turn the scalar-or-array `alpha` and `beta` into explicit prior vectors

        A symmetric prior (a single float) is broadcast to every topic / every word;
        an asymmetric prior is used as given. Storing them as vectors lets the
        sampling formula be written once, with the exact Dirichlet normalizer
        sum_v beta_v instead of the symmetric-only shortcut V * beta.
        """
        alpha_vec = np.asarray(self.alpha, dtype=float)
        if alpha_vec.ndim == 0:
            alpha_vec = np.full(self.n_components, float(self.alpha))
        elif alpha_vec.shape != (self.n_components,):
            raise ValueError(
                f"alpha must be a float or an array of length n_components="
                f"{self.n_components}; got shape {alpha_vec.shape}."
            )

        beta_vec = np.asarray(self.beta, dtype=float)
        if beta_vec.ndim == 0:
            beta_vec = np.full(n_features, float(self.beta))
        elif beta_vec.shape != (n_features,):
            raise ValueError(
                f"beta must be a float or an array of length vocabulary_size="
                f"{n_features}; got shape {beta_vec.shape}."
            )

        if np.any(alpha_vec <= 0) or np.any(beta_vec <= 0):
            raise ValueError("alpha and beta are Dirichlet pseudo-counts and must be > 0.")

        self._alpha_vec = alpha_vec
        self._beta_vec = beta_vec
        self._beta_sum = beta_vec.sum()   # sum_v beta_v; equals V*beta when symmetric

    def _check_is_fitted(self, caller):
        """Raise a readable error instead of a cryptic NoneType crash."""
        if self.components_ is None:
            raise ValueError(
                f"This LatentDirichletAllocation instance is not fitted yet. "
                f"Call fit(X) before {caller}."
            )
    
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
        
        # Expand the scalar/array priors into vectors (also validates their lengths)
        self._expand_priors(n_features)

        # Initialize count matrices
        self._doc_topic_count = np.zeros((n_samples, self.n_components))
        self._topic_word_count = np.zeros((self.n_components, n_features))
        self._topic_count = np.zeros(self.n_components)

        # Posterior-mean accumulators (only filled when sample_lag > 0)
        self._doc_topic_sum = np.zeros((n_samples, self.n_components))
        self._topic_word_sum = np.zeros((self.n_components, n_features))
        self._n_collected = 0
        self.n_gibbs_samples_ = 0
        
        # Initialize topic assignments for each word occurrence
        self._topic_assignments = []
        
        for d in range(n_samples):
            doc_assignments = []
            for w in range(n_features):
                word_count = int(X[d, w])
                for _ in range(word_count):
                    # Randomly assign topic
                    topic = self._rng.integers(0, self.n_components)
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
        
        P(z_i = k | z_-i, w) ~ (n_dk + alpha_k) * (n_kw + beta_w) / (n_k + sum(beta))
        
        Where:
        - n_dk: count of topic k in document d (excluding current word)
        - n_kw: count of word w in topic k (excluding current word)
        - n_k: total count of words in topic k (excluding current word)
        - sum(beta): sum of the topic-word prior over the whole vocabulary.
          For a symmetric prior this is exactly V * beta (V = vocabulary size);
          for an asymmetric beta vector the sum is the correct normalizer.

        The caller (_gibbs_sampling_iteration) has already decremented the counts
        for the token being resampled, which is what makes them "excluding current
        word" - that exclusion is the whole reason the conditional is tractable.
        
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
            # Document-topic component: (n_dk + alpha_k), "does doc d like topic k?"
            doc_topic_prob = self._doc_topic_count[d, k] + self._alpha_vec[k]
            
            # Topic-word component: (n_kw + beta_w) / (n_k + sum(beta)),
            # "does topic k like word w?"
            topic_word_prob = (self._topic_word_count[k, w] + self._beta_vec[w])
            topic_word_prob /= (self._topic_count[k] + self._beta_sum)
            
            # Combined probability (unnormalized posterior over topics)
            probs[k] = doc_topic_prob * topic_word_prob
        
        # Normalize to get probability distribution
        probs /= probs.sum()
        
        # Sample new topic from the private generator
        new_topic = self._rng.choice(self.n_components, p=probs)
        
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
        - theta: Document-topic distribution
        - phi:   Topic-word distribution

        Which counts are used:
        - sample_lag == 0 (default): the counts of the single FINAL Gibbs state.
          This is what Griffiths & Steyvers (2004) report - "a single sample
          taken after 2,000 iterations of Gibbs sampling".
        - sample_lag  > 0: the AVERAGE of the count matrices collected every
          sample_lag sweeps after burn_in, with the prior added once afterwards:

              phi_kw = (mean_s n_kw^s + beta_w) / (mean_s n_k^s + sum(beta))

          Be precise about what that is. The quantity a posterior mean would
          give is E[phi_kw | w] = sum_z P(z | w) (n_kw + beta_w)/(n_k + sum(beta)),
          and the Monte Carlo estimator that converges to it averages the
          per-sweep NORMALIZED rows,

              phi_kw ~= mean_s [ (n_kw^s + beta_w) / (n_k^s + sum(beta)) ]

          which is what Heinrich's widely copied LdaGibbsSampler accumulates.
          A ratio of means is not a mean of ratios, so the two differ. For theta
          they coincide exactly - the row sum n_d + sum(alpha) is identical in
          every sweep, so it factors straight out of the average. For phi the
          gap grows with how much n_k moves between sweeps. Measured max
          per-entry difference, both fits with alpha=0.1, beta=0.01,
          random_state=42, burn_in = max_iter/2 and sample_lag = 5:

            1.4e-05 on the 3000-token benchmark corpus of _25_lda.md
                    (K, V, D, L) = (3, 30, 60, 50), corpus default_rng(0),
                    max_iter=100
            0.026   on a deliberately tiny 200-token corpus from that same
                    recipe with (K, V, D, L) = (5, 20, 20, 10) - each topic
                    owning V/K = 4 words at probability 1/4 - max_iter=50

          The tiny-corpus figure is corpus-seed dependent - it ranges over
          0.003 to 0.096 across corpus seeds 0-7 - so read it as an order of
          magnitude rather than a constant. Counts are averaged here because it
          keeps the accumulators in the same units as the three count tables;
          on any realistic corpus the choice is invisible.
        """
        # Pick the raw counts to turn into probabilities
        if self.sample_lag > 0 and self._n_collected > 0:
            doc_topic_count = self._doc_topic_sum / self._n_collected
            topic_word_count = self._topic_word_sum / self._n_collected
        else:
            doc_topic_count = self._doc_topic_count
            topic_word_count = self._topic_word_count

        # Document-topic distribution (theta)
        # theta_dk = (n_dk + alpha_k) / (n_d + sum_k alpha_k)
        self.doc_topic_distr_ = doc_topic_count + self._alpha_vec
        self.doc_topic_distr_ /= self.doc_topic_distr_.sum(axis=1, keepdims=True)
        
        # Topic-word distribution (phi) - stored as components_
        # phi_kw = (n_kw + beta_w) / (n_k + sum_v beta_v)
        self.components_ = topic_word_count + self._beta_vec
        self.components_ /= self.components_.sum(axis=1, keepdims=True)
    
    def _compute_perplexity(self, X, doc_topic_distr=None):
        """
        Compute perplexity of the model on data X given its topic mixtures
        
        Perplexity is a standard metric for evaluating topic models.
        Lower perplexity indicates better model fit.
        
        Perplexity = exp(-log-likelihood / total word count)
        log-likelihood = sum_d sum_w count_dw * log( sum_k phi_kw * theta_dk )
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Document-term matrix
        doc_topic_distr : np.ndarray, shape (n_samples, n_components) or None
            Topic mixtures for the rows of X. When None, the TRAINING mixtures
            (self.doc_topic_distr_) are used, so X must then be the training
            corpus. Use the public perplexity(X) for held-out documents - it
            infers fresh mixtures with transform() first.
            
        Returns:
        --------
        perplexity : float
            Model perplexity
        """
        if doc_topic_distr is None:
            doc_topic_distr = self.doc_topic_distr_

        if X.shape[0] != doc_topic_distr.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} documents but doc_topic_distr has "
                f"{doc_topic_distr.shape[0]} rows. For held-out documents call "
                f"perplexity(X), which infers the mixtures with transform()."
            )

        log_likelihood = 0.0
        total_words = 0
        
        for d in range(X.shape[0]):
            for w in range(X.shape[1]):
                count = X[d, w]
                if count > 0:
                    # P(w|d) = sum_k P(w|k) * P(k|d)
                    prob = np.dot(self.components_[:, w], doc_topic_distr[d])
                    log_likelihood += count * np.log(prob + 1e-10)
                    total_words += count

        if total_words == 0:
            raise ValueError("X contains no word occurrences; perplexity is undefined.")
        
        perplexity = np.exp(-log_likelihood / total_words)
        return perplexity
    
    def perplexity(self, X):
        """
        Perplexity of the fitted model on a corpus X (held-out or training)

        This is the public, always-correct entry point: the topic mixtures for the
        rows of X are inferred with transform(X) before scoring, so X does NOT have
        to be the corpus the model was trained on and may have any number of rows.

        Perplexity = exp( -sum_dw count_dw * log(sum_k phi_kw * theta_dk) / N )

        Lower is better. Note that theta is inferred from the same words being
        scored (as in the original LDA papers' "empirical likelihood" style
        evaluation), so this number is optimistic in absolute terms; use it to
        COMPARE models on the same corpus, not as an absolute measure.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Document-term matrix over the SAME vocabulary used for fitting

        Returns:
        --------
        perplexity : float
            Model perplexity (lower is better)
        """
        self._check_is_fitted("perplexity")
        X = self._check_X(X)
        doc_topic_distr = self.transform(X)
        return self._compute_perplexity(X, doc_topic_distr)

    def fit(self, X):
        """
        Fit the LDA model to data using collapsed Gibbs sampling

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Document-term matrix where X[i, j] is the count of word j in document i
            - n_samples: Number of documents
            - n_features: Vocabulary size
            - Values MUST be non-negative integers (raw word counts). LDA is a
              multinomial model over tokens; TF-IDF or normalized weights are
              rejected because there is no such thing as 0.37 of a word.
            - A single document may be passed as a 1-D vector
            
        Returns:
        --------
        self : object
            Fitted model
        """
        X = self._check_X(X)

        if self.sample_lag > 0 and self.burn_in >= self.max_iter:
            raise ValueError(
                f"burn_in={self.burn_in} discards every one of the max_iter="
                f"{self.max_iter} sweeps, so no posterior sample could be collected. "
                f"Use burn_in < max_iter."
            )

        # Restart the private generator so a second fit() on this object
        # reproduces the first one exactly.
        self._rng = np.random.default_rng(self.random_state)

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

            # Posterior-mean estimator: after burn_in, accumulate the count
            # matrices every sample_lag sweeps and average them at the end.
            # (sample_lag == 0 disables this and keeps the final-state estimator.)
            if (self.sample_lag > 0 and iteration >= self.burn_in
                    and (iteration - self.burn_in) % self.sample_lag == 0):
                self._doc_topic_sum += self._doc_topic_count
                self._topic_word_sum += self._topic_word_count
                self._n_collected += 1
                self.n_gibbs_samples_ = self._n_collected
            
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
        topic-word distributions (phi) with Gibbs sampling. phi is held FIXED
        here - only the per-document topic counts move - which is why this is
        "folding in" a document rather than re-training:

            P(z_i = k | z_-i, w) ~ (n_dk + alpha_k) * phi_kw

        Note the second factor is the already-learned phi, not a count ratio.

        Folding in is deliberately cheap: it runs min(max_iter, 50) sweeps, not
        max_iter. When sample_lag > 0 the burn_in/sample_lag schedule is scaled
        by that same ratio, so the fold-in discards the same FRACTION of its
        sweeps and collects about as many states as fit() did.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Document-term matrix over the same vocabulary used for fitting.
            A single document may be passed as a 1-D vector.
            
        Returns:
        --------
        doc_topic_distr : np.ndarray, shape (n_samples, n_components)
            Document-topic distribution for input documents
        """
        self._check_is_fitted("transform")
        X = self._check_X(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with "
                f"{self.n_features_}. The vocabulary must match."
            )

        # Restart the private generator so transform() is reproducible and does
        # not depend on how many times fit()/transform() ran before it.
        rng = np.random.default_rng(self.random_state)
        
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
                    topic = rng.choice(self.n_components, p=probs)
                    doc_assignments.append((w, topic))
                    doc_topic_count[d, topic] += 1
            
            topic_assignments.append(doc_assignments)
        
        # Gibbs sampling for new documents (fewer iterations)
        n_iter = min(self.max_iter, 50)

        # Same sample averaging as fit(). Because the fold-in is capped at 50
        # sweeps, a burn_in/sample_lag schedule written for max_iter has to be
        # rescaled by the same factor: with max_iter=200, burn_in=100 the raw
        # burn-in would swallow every one of the 50 sweeps and "averaging" would
        # silently collect the single final state - a no-op. Scaling both keeps
        # the same FRACTION of sweeps discarded and roughly the same NUMBER of
        # states collected as fit() does.
        collect = self.sample_lag > 0
        if collect:
            scale = n_iter / max(self.max_iter, 1)
            burn_in = min(int(self.burn_in * scale), n_iter - 1)
            lag = max(1, int(np.ceil(self.sample_lag * scale)))
        else:
            burn_in, lag = 0, 0
        doc_topic_sum = np.zeros_like(doc_topic_count)
        n_collected = 0

        for iteration in range(n_iter):
            for d in range(n_samples):
                for i, (w, old_topic) in enumerate(topic_assignments[d]):
                    # Remove current assignment
                    doc_topic_count[d, old_topic] -= 1
                    
                    # Sample new topic: (n_dk + alpha_k) * phi_kw
                    probs = (doc_topic_count[d] + self._alpha_vec) * (self.components_[:, w] + 1e-10)
                    probs /= probs.sum()
                    new_topic = rng.choice(self.n_components, p=probs)
                    
                    # Update assignment
                    doc_topic_count[d, new_topic] += 1
                    topic_assignments[d][i] = (w, new_topic)
        
            if collect and iteration >= burn_in and (iteration - burn_in) % lag == 0:
                doc_topic_sum += doc_topic_count
                n_collected += 1

        if n_collected > 0:
            doc_topic_count = doc_topic_sum / n_collected

        # Compute document-topic distribution
        # theta_dk = (n_dk + alpha_k) / (n_d + sum_k alpha_k)
        doc_topic_distr = doc_topic_count + self._alpha_vec
        doc_topic_distr /= doc_topic_distr.sum(axis=1, keepdims=True)
        
        return doc_topic_distr

    
    def fit_transform(self, X):
        """
        Fit the model and return document-topic distribution
        
        Equivalent to fit(X) followed by reading doc_topic_distr_. This returns the
        mixtures estimated DURING training (from the Gibbs counts), which is not the
        same code path as transform(X) - that one folds documents in against a frozen
        phi. Both are valid estimates of theta; the training one is usually sharper.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
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
        self._check_is_fitted("get_top_words")

        if len(feature_names) != self.n_features_:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but the model was "
                f"fitted with {self.n_features_} features."
            )
        
        top_words = []
        for topic_idx in range(self.n_components):
            # Get indices of top words for this topic
            top_indices = np.argsort(self.components_[topic_idx])[::-1][:n_top_words]
            top_words.append([feature_names[i] for i in top_indices])
        
        return top_words


"""
========================================
PLUG-AND-PLAY DEMO  (see the __main__ block below)
========================================
"""

if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _25_lda.py
    # Requires numpy only. Finishes in a few seconds. ASCII output only.
    # ----------------------------------------------------------------
    np.random.seed(42)

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
    
    # Fit LDA model (verbose=1 prints its own "Fitting LDA with 3 topics..." banner)
    print("")
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
    
    # ---- Did it actually work? Score against the known ground truth ----
    # Docs 0-2 are sports, 3-5 tech, 6-8 food (doc 9 is deliberately mixed).
    # LDA numbers its topics arbitrarily ("label switching"), so we first map
    # each true group to the topic that group's documents mostly chose.
    print("\n" + "-" * 70)
    print("QUALITY CHECK (the corpus has known ground truth):")
    print("-" * 70)

    true_groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])   # doc 9 excluded: mixed
    dominant = doc_topics[:9].argmax(axis=1)
    group_to_topic = {}
    for g in range(3):
        topics_of_group = dominant[true_groups == g]
        group_to_topic[g] = np.bincount(topics_of_group, minlength=3).argmax()

    correct = np.sum([dominant[i] == group_to_topic[true_groups[i]] for i in range(9)])
    print(f"Group -> topic mapping (sports, tech, food): "
          f"{[int(group_to_topic[g]) for g in range(3)]}")
    print(f"Train topic-recovery accuracy: {correct}/9 = {100.0 * correct / 9:.2f}%")
    print(f"Train perplexity            : {lda.perplexity(X_simple):.4f}")
    print(f"Uniform-guess perplexity    : {len(vocabulary):.4f}  (chance baseline = V)")

    # Example 2: LDA with Different Hyperparameters
    print("\n" + "=" * 70)
    print("Example 2: Effect of Alpha on Topic Distribution")
    print("=" * 70)
    
    # Build a corpus that HAS topic structure to recover. (Pure noise, e.g.
    # np.random.poisson, has no topics, so the alpha effect would be invisible.)
    # 5 planted topics over 20 words; each document is a 90/10 blend of two of them.
    np.random.seed(42)
    n_docs = 20
    vocab_size = 20
    n_planted = 5
    
    planted_topics = np.zeros((n_planted, vocab_size))
    for k in range(n_planted):
        planted_topics[k, k * 4:(k + 1) * 4] = 0.25       # 4 words own each topic
    
    X_synthetic = np.zeros((n_docs, vocab_size), dtype=int)
    for d in range(n_docs):
        main_k, side_k = d % n_planted, (d + 1) % n_planted
        word_probs = 0.9 * planted_topics[main_k] + 0.1 * planted_topics[side_k]
        X_synthetic[d] = np.random.multinomial(40, word_probs)

    print(f"\nSynthetic corpus: {n_docs} documents, {vocab_size} vocabulary, "
          f"{n_planted} planted topics")
    print("Metric: Shannon entropy of each document's topic mixture, in nats.")
    print(f"        0.000 = one topic per document, {np.log(n_planted):.3f} = "
          f"all {n_planted} topics equally  (= log K)")
    
    # Test different alpha values
    alphas = [0.01, 0.1, 1.0]
    doc_topics_final = None   # keep the alpha=0.1 result for the comparison below
    
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
        if alpha_val == 0.1:
            doc_topics_final = doc_topics_test
        
        # Topic sparsity: low entropy = document concentrated on few topics
        entropies = -np.sum(doc_topics_test * np.log(doc_topics_test + 1e-12), axis=1)
        
        print(f"Average topic-mixture entropy: {np.mean(entropies):.3f} nats")
        print(f"Document-topic distribution sample (first 3 docs):")
        print(np.round(doc_topics_test[:3], 3))

    print("\nHigher alpha -> higher entropy -> documents spread over more topics.")

    # ---- Posterior mean vs. single final Gibbs sample ----
    # By default the model reports the LAST Gibbs state, which is one draw from
    # the posterior (and is what Griffiths & Steyvers, 2004, report too).
    # burn_in/sample_lag average the post-burn-in count matrices instead, which
    # is the usual MCMC way to cut the variance of that single draw.
    # Document 1 was planted as a blend, so its mixture is genuinely uncertain
    # and the two estimators visibly disagree.
    print("\n" + "-" * 70)
    print("POSTERIOR MEAN vs SINGLE FINAL SAMPLE (alpha=0.1):")
    print("-" * 70)

    lda_mean = LatentDirichletAllocation(
        n_components=5, max_iter=50, alpha=0.1, beta=0.01,
        random_state=42, burn_in=25, sample_lag=5
    )
    doc_topics_mean = lda_mean.fit_transform(X_synthetic)

    print(f"Gibbs states averaged        : {lda_mean.n_gibbs_samples_}")
    print(f"Doc 1 theta, final state     : {np.round(doc_topics_final[1], 3)}")
    print(f"Doc 1 theta, posterior mean  : {np.round(doc_topics_mean[1], 3)}")
    print(f"Largest theta change anywhere: "
          f"{np.abs(doc_topics_final - doc_topics_mean).max():.4f}")
    print("Averaging costs nothing extra and damps the luck of the last sweep.")

    
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
        # docs 0-2 sports, 3-5 tech, 6-8 food, 9 mixed
        doc_type = ("Sports" if doc_idx <= 2 else
                    "Tech" if doc_idx <= 5 else
                    "Food" if doc_idx <= 8 else "Mixed")
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
    print("(These 3 documents were NOT in the training corpus.)")
    
    # Transform using fitted model
    new_doc_topics = lda_similarity.transform(X_test)
    
    print("\nInferred topic distributions for new documents:")
    print(np.round(new_doc_topics, 3))
    
    doc_types = ["Sports", "Tech", "Mixed"]
    for i, doc_type in enumerate(doc_types):
        print(f"\n{doc_type} document:")
        print(f"  Topic distribution: {np.round(new_doc_topics[i], 3)}")

    # ---- Score the held-out documents ----
    # Reuse the training mixtures to learn which topic index means what, then
    # check that the two unambiguous new documents land on the right one.
    train_dominant = doc_topics_sim[:9].argmax(axis=1)
    sports_topic = np.bincount(train_dominant[0:3], minlength=3).argmax()
    tech_topic = np.bincount(train_dominant[3:6], minlength=3).argmax()

    held_out_correct = 0
    held_out_correct += int(new_doc_topics[0].argmax() == sports_topic)
    held_out_correct += int(new_doc_topics[1].argmax() == tech_topic)

    print("\n" + "-" * 70)
    print("HELD-OUT QUALITY CHECK:")
    print("-" * 70)
    print(f"Topic index for sports = {int(sports_topic)}, for tech = {int(tech_topic)}")
    print(f"Held-out topic-recovery accuracy: {held_out_correct}/2 unambiguous "
          f"documents correct")
    print(f"Held-out perplexity             : "
          f"{lda_similarity.perplexity(X_test):.4f}")
    print("(The 3rd document is a deliberate mixture, so it has no single "
          "correct topic;")
    print(f" its inferred mixture is {np.round(new_doc_topics[2], 3)}.)")

    
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
       - Rule: alpha = 50/n_components is the Griffiths & Steyvers (2004)
         heuristic, tuned for LONG documents; for the short documents used
         in these examples the sparser default alpha=0.1 works better
    
    3. SETTING BETA (topic-word prior):
       - Low beta (0.01-0.1): Topics focus on few words (sparse)
       - High beta (1.0-10.0): Topics use many words
       - Default 0.01 works well for most cases
       - Rule: beta = 0.01 for focused topics
    
    4. ITERATIONS (max_iter) AND MCMC HYGIENE:
       - Minimum: 50 iterations
       - Recommended: 100-500 for good convergence
       - More iterations = better but slower
       - Monitor perplexity to check convergence
       - Gibbs sampling is a Markov chain: the early sweeps are NOT samples
         from the posterior. Set burn_in (sweeps to discard) and sample_lag
         (collect every Nth sweep afterwards) to average post-burn-in states
         instead of trusting a single final draw.
         Example: max_iter=200, burn_in=100, sample_lag=10 -> 10 samples averaged.
       - Topic NUMBERS are arbitrary and change between seeds ("label
         switching"). Compare topics by their top words, never by index.
    
    5. DATA PREPROCESSING (CRITICAL):
       - Remove stop words (the, is, and, etc.)
       - Remove very rare words (appear in <5 documents)
       - Remove very common words (appear in >90% documents)
       - Lemmatize or stem words (running -> run)
       - Feed RAW INTEGER COUNTS (CountVectorizer), never TF-IDF:
         LDA is a multinomial model over word tokens, and this
         implementation raises an error on fractional values
    
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
