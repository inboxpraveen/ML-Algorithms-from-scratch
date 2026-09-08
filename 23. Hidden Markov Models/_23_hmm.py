import numpy as np

class HiddenMarkovModel:
    """
    Hidden Markov Model (HMM) Implementation from Scratch
    
    A Hidden Markov Model is a statistical model where the system being modeled
    is assumed to be a Markov process with hidden (unobserved) states.
    
    Key Idea: The system has hidden states that we cannot directly observe,
    but we can observe outputs/emissions that depend on these hidden states.
    
    Use Cases:
    - Speech Recognition: Hidden states = phonemes, Observations = acoustic signals
    - Part-of-Speech Tagging: Hidden states = POS tags, Observations = words
    - Weather Prediction: Hidden states = weather conditions, Observations = activities
    - Stock Market: Hidden states = market regimes, Observations = prices
    - Bioinformatics: Hidden states = gene structure, Observations = DNA sequences
    
    Key Components:
        States: Hidden states the model can be in
        Observations: Visible outputs we can observe
        Initial Probability: Probability of starting in each state
        Transition Probability: Probability of moving from one state to another
        Emission Probability: Probability of observing output given a state
    
    The model is written lambda = (pi, A, B) where
        pi[i]    = P(S_0 = i)
        A[i, j]  = P(S_t = j | S_{t-1} = i)
        B[i, k]  = P(O_t = k | S_t = i)
    
    Baum-Welch Updates (the exact formulas fit() implements):
        E-step, from the forward-backward pass:
            gamma_t(i)   = P(S_t = i | O, lambda)
            xi_t(i, j)   = P(S_t = i, S_{t+1} = j | O, lambda)
        M-step (sums run over every training sequence):
            pi_i  = gamma_0(i)                                        (averaged over sequences)
            a_ij  = sum_{t=0}^{T-2} xi_t(i, j) / sum_{t=0}^{T-2} gamma_t(i)
            b_ik  = sum_{t: O_t = k} gamma_t(i) / sum_{t=0}^{T-1} gamma_t(i)
        Note the different upper limits: a_ij averages over TRANSITIONS (there are
        only T-1 of them), b_ik averages over TIME STEPS (there are T).
    
    Scaling (Rabiner 1989, Section V.A) - why there are no 1e-10 fudge factors here:
        A raw forward variable alpha_t(i) is a product of 2t probabilities, so it
        underflows to 0.0 after roughly 30 steps in float64. This implementation
        rescales every column of the trellis to sum to 1:
            c_t         = 1 / sum_i alpha_t(i)
            alpha_hat_t = c_t * alpha_t
            beta_hat_t  = c_t * beta_t          (same c_t, applied backwards)
        Then the likelihood and the E-step quantities come out of the scaled
        variables exactly, with no division by a vanishing number:
            log P(O | lambda) = -sum_t log c_t
            gamma_t(i)        = alpha_hat_t(i) * beta_hat_t(i) / c_t
            xi_t(i, j)        = alpha_hat_t(i) * a_ij * b_j(O_{t+1}) * beta_hat_{t+1}(j)
        viterbi() uses the equivalent trick of working in log space.
    
    Simplifications vs. a canonical HMM library (e.g. hmmlearn):
        See the "Simplifications vs. Canonical HMM Implementations" section of
        _23_hmm.md. In short: discrete (categorical) emissions only - no Gaussian
        or mixture emissions; a single random restart per fit() call (pass
        different random_state values to explore local optima yourself); no
        Dirichlet priors / smoothing on the M-step counts; no left-right topology
        constraints; no posterior (MAP) decoding, only Viterbi decoding.
    """
    
    def __init__(self, n_states=None, n_observations=None):
        """
        Initialize the Hidden Markov Model
        
        Parameters:
        -----------
        n_states : int, default=None
            Number of hidden states N in the model. MUST be set before calling
            fit(); set_parameters() infers it from the shape of initial_prob.
            - Range: 2-20 for models a human can still interpret
            - More states fit finer structure but overfit quickly: the model has
              N^2 + N*M - 1 free parameters (M = number of distinct observations),
              so doubling N roughly quadruples the parameter count
            - Fewer states underfit and blur distinct regimes together
            Typical: 2-3 for weather / market regimes, 3-6 for user-intent or
            gene-region models, 12-45 for part-of-speech tagging
            Example: For weather, might be 2 (Sunny, Rainy)
        
        n_observations : int, default=None
            Number of distinct observation symbols M. This is only a hint - it is
            INFERRED and overwritten by fit() (from the training vocabulary) and by
            set_parameters() (from emission_prob.shape[1]), so you almost never need
            to pass it. It is kept because reading it after fitting tells you how
            large the learned vocabulary is.
            Example: For activities, might be 3 (Walk, Shop, Clean)
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        # Model parameters (initialized in fit or set manually)
        self.initial_prob = None      # π: P(state at t=0)
        self.transition_prob = None   # A: P(state_j | state_i)
        self.emission_prob = None     # B: P(observation_k | state_i)
        
        # For mapping between labels and indices
        self.state_map = None
        self.observation_map = None
        self.state_labels = None
        self.observation_labels = None
    
    def _initialize_parameters(self, observations, rng=None):
        """
        Initialize HMM parameters (starting point for the Baum-Welch algorithm)
        
        Baum-Welch is EM, so it only ever climbs to a LOCAL maximum: the starting
        point decides which one. The single thing that matters here is SYMMETRY
        BREAKING. If two states started with identical rows, gamma would be
        identical for both of them at every step and the M-step would keep them
        identical forever - EM can never separate states it starts on top of each
        other. Drawing each row uniformly at random and normalizing guarantees the
        states start apart, and keeps every probability strictly positive (a zero
        in A or B can never recover, because every update to it is proportional to
        its current value through gamma).
        
        Note on the `observations` argument: this initializer deliberately does not
        look at the data. Seeding the emission rows from the corpus symbol
        frequencies instead was measured on this model and gave no improvement -
        it starts every state near the same marginal distribution, which is close
        to the symmetric saddle point EM has to escape. The argument is kept so
        that you can drop in a data-driven initializer (k-means on observation
        histograms, or a supervised warm start) without touching fit().
        
        Parameters:
        -----------
        observations : list of lists
            Training sequences, already encoded as observation INDICES.
            Available to data-driven initializers; unused by this random one.
        
        rng : numpy.random.RandomState, optional
            Private random generator. If None, the global numpy RNG is used
            (so np.random.seed(...) still works as expected).
        """
        if rng is None:
            rng = np.random
        
        # Initial probabilities: uniform distribution
        self.initial_prob = np.ones(self.n_states) / self.n_states
        
        # Transition probabilities: random + normalize (rows must sum to 1)
        self.transition_prob = rng.rand(self.n_states, self.n_states)
        self.transition_prob = self.transition_prob / self.transition_prob.sum(axis=1, keepdims=True)
        
        # Emission probabilities: random + normalize (rows must sum to 1)
        self.emission_prob = rng.rand(self.n_states, self.n_observations)
        self.emission_prob = self.emission_prob / self.emission_prob.sum(axis=1, keepdims=True)

    def _check_is_fitted(self):
        """
        Raise a clear error if the model has no parameters yet
        
        Every inference method needs pi, A and B. Without this guard the failure
        surfaces deep inside numpy as an unhelpful TypeError on None.
        """
        if (self.initial_prob is None or self.transition_prob is None
                or self.emission_prob is None):
            raise ValueError(
                "This HiddenMarkovModel is not fitted yet. Call fit(sequences) to "
                "learn the parameters, or set_parameters(pi, A, B) to supply them."
            )

    def set_parameters(self, initial_prob, transition_prob, emission_prob, 
                      state_labels=None, observation_labels=None):
        """
        Manually set HMM parameters
        
        Use this when you know the parameters from domain knowledge, or have
        already estimated them from labelled data by counting. (This class does
        not estimate parameters from labelled state sequences for you - fit()
        implements the unsupervised Baum-Welch algorithm only.)
        
        All arguments accept plain Python lists as well as numpy arrays.
        
        Parameters:
        -----------
        initial_prob : array-like, shape (n_states,)
            Initial state probabilities
            Example: [0.6, 0.4] means 60% chance of starting in state 0
        
        transition_prob : array-like, shape (n_states, n_states)
            State transition probabilities
            transition_prob[i, j] = P(state_j | state_i)
            Each row must sum to 1
        
        emission_prob : array-like, shape (n_states, n_observations)
            Observation emission probabilities
            emission_prob[i, k] = P(observation_k | state_i)
            Each row must sum to 1
        
        state_labels : list, optional
            Names of states (for display purposes)
            Example: ['Sunny', 'Rainy']
        
        observation_labels : list, optional
            Names of observations. These are not just for display: they define the
            label -> column mapping, so that predict(['Walk', 'Shop']) works.
            Example: ['Walk', 'Shop', 'Clean']
        
        Returns:
        --------
        self : HiddenMarkovModel
            The configured model (enables chaining)
        """
        # Copy into float arrays first, then read the shapes off the COPIES so that
        # plain Python lists work exactly as well as numpy arrays.
        self.initial_prob = np.array(initial_prob, dtype=float)
        self.transition_prob = np.array(transition_prob, dtype=float)
        self.emission_prob = np.array(emission_prob, dtype=float)
        
        self.n_states = self.initial_prob.shape[0]
        self.n_observations = self.emission_prob.shape[1]
        
        self.state_labels = state_labels if state_labels else [f"S{i}" for i in range(self.n_states)]
        self.observation_labels = observation_labels if observation_labels else [f"O{i}" for i in range(self.n_observations)]
        
        # Label -> index lookups. Without these, predict/score/viterbi cannot turn
        # ['Walk', 'Shop'] into the column indices the matrices are indexed by.
        self.state_map = {s: i for i, s in enumerate(self.state_labels)}
        self.observation_map = {o: i for i, o in enumerate(self.observation_labels)}
        
        # Shape checks (clearer than the IndexError numpy would raise later)
        assert self.transition_prob.shape == (self.n_states, self.n_states), \
            f"transition_prob must be ({self.n_states}, {self.n_states}), got {self.transition_prob.shape}"
        assert self.emission_prob.shape[0] == self.n_states, \
            f"emission_prob must have {self.n_states} rows, got {self.emission_prob.shape[0]}"
        
        # Validate probabilities sum to 1.
        # np.allclose already reduces to a single Python bool - calling .all() on it
        # is an AttributeError on numpy 2.x, which is why this used to blow up.
        assert np.allclose(self.initial_prob.sum(), 1.0), "Initial probabilities must sum to 1"
        assert np.allclose(self.transition_prob.sum(axis=1), 1.0), "Transition probabilities must sum to 1 (each row)"
        assert np.allclose(self.emission_prob.sum(axis=1), 1.0), "Emission probabilities must sum to 1 (each row)"
        
        return self

    def _encode_sequence(self, sequence, mapping):
        """
        Convert sequence of labels to indices
        
        Parameters:
        -----------
        sequence : list
            Sequence of labels
        mapping : dict
            Mapping from labels to indices
            
        Returns:
        --------
        encoded : list
            Sequence of indices
        """
        return [mapping[item] for item in sequence]
    
    def _decode_sequence(self, sequence, labels):
        """
        Convert sequence of indices to labels
        
        Parameters:
        -----------
        sequence : list
            Sequence of indices
        labels : list
            List of labels
            
        Returns:
        --------
        decoded : list
            Sequence of labels
        """
        return [labels[idx] for idx in sequence]

    def _encode_observations(self, observations):
        """
        Turn a sequence of observation LABELS into observation INDICES
        
        Accepts either form: if the model has a label vocabulary and every token in
        the sequence is a known label, the sequence is translated; if the tokens are
        already integer indices it is passed straight through.
        
        The whole sequence is checked, not just its first token. Checking only
        observations[0] means ['Admin', 'Login'] and ['Login', 'Admin'] fail in two
        completely different ways for the same underlying reason.
        
        Parameters:
        -----------
        observations : list
            Sequence of labels or of integer indices
        
        Returns:
        --------
        encoded : list
            Sequence of integer observation indices
        """
        if len(observations) == 0:
            raise ValueError("observations must contain at least one time step")
        
        # Already indices? (numpy integers count too)
        if all(isinstance(o, (int, np.integer)) and not isinstance(o, bool)
               for o in observations):
            for o in observations:
                if o < 0 or o >= self.n_observations:
                    raise ValueError(
                        f"Observation index {o} is out of range for a model with "
                        f"{self.n_observations} observation symbols"
                    )
            return list(observations)
        
        if not self.observation_map:
            raise ValueError(
                "This model has no observation vocabulary, so it can only accept "
                "integer observation indices. Pass observation_labels to "
                "set_parameters(), or train with fit() on label sequences."
            )
        
        unknown = [o for o in observations if o not in self.observation_map]
        if unknown:
            raise ValueError(
                f"Unknown observation symbol(s) {sorted(set(map(str, unknown)))}. "
                f"Known symbols: {list(self.observation_map)}"
            )
        return self._encode_sequence(observations, self.observation_map)

    def _forward_pass(self, observations, scale=True):
        """
        Core forward recursion, optionally with Rabiner scaling
        
        Shared by forward(), score() and fit() so there is exactly one copy of the
        recursion in this file.
        
        Unscaled (scale=False):
            alpha_t(i) = P(O_0, ..., O_t, S_t = i | lambda)     <- the textbook value
        Scaled (scale=True):
            c_t             = 1 / sum_i alpha_t(i)
            alpha_hat_t(i)  = c_t * alpha_t(i)                  <- each row sums to 1
            log P(O|lambda) = -sum_t log c_t
        The scaled version is numerically safe for sequences of any length; the
        unscaled one underflows to 0.0 after roughly 30 steps.
        
        Parameters:
        -----------
        observations : list
            Sequence of observation indices
        scale : bool, default=True
            Whether to rescale each trellis column to sum to 1
        
        Returns:
        --------
        alpha : array, shape (T, n_states)
        c : array, shape (T,)
            Scaling coefficients (all 1.0 when scale=False)
        log_prob : float
            Log probability of the observation sequence
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        c = np.ones(T)
        
        # Initialization: α(0, i) = π(i) * B(i, O_0)
        alpha[0] = self.initial_prob * self.emission_prob[:, observations[0]]
        if scale:
            total = np.sum(alpha[0])
            if total <= 0.0:
                return alpha, c, -np.inf   # model gives this sequence probability 0
            c[0] = 1.0 / total
            alpha[0] = alpha[0] * c[0]
        
        # Recursion: α(t, j) = [Σ α(t-1, i) * A(i,j)] * B(j, O_t)
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_prob[:, j]) * \
                             self.emission_prob[j, observations[t]]
            if scale:
                total = np.sum(alpha[t])
                if total <= 0.0:
                    return alpha, c, -np.inf
                c[t] = 1.0 / total
                alpha[t] = alpha[t] * c[t]
        
        if scale:
            # Termination: log P(O | lambda) = -sum_t log c_t
            log_prob = -np.sum(np.log(c))
        else:
            # Termination: P(O | lambda) = sum_i alpha(T-1, i)
            total = np.sum(alpha[T-1])
            log_prob = np.log(total) if total > 0.0 else -np.inf
        
        return alpha, c, log_prob

    def forward(self, observations, scale=True):
        """
        Forward Algorithm: Calculate probability of observation sequence
        
        Computes α(t, i) = P(O_1, O_2, ..., O_t, state_t = i | model)
        
        This is the probability of:
        - Observing the sequence up to time t
        - AND being in state i at time t
        
        Parameters:
        -----------
        observations : list
            Sequence of observation indices
        
        scale : bool, default=True
            If True (recommended, and what score()/fit() use), each column of the
            trellis is rescaled to sum to 1 as it is computed, and log_prob is
            exact for a sequence of any length.
            If False, the raw textbook alpha values are returned - useful for
            checking a short worked example by hand (for the weather model in the
            guide, forward([0, 2], scale=False) gives alpha[0] = [0.36, 0.04]) -
            but they underflow to zero past roughly 30 time steps.
        
        Returns:
        --------
        alpha : array, shape (T, n_states)
            Forward probabilities. With scale=True these are the SCALED values
            alpha_hat, so every row sums to 1; with scale=False they are the raw
            joint probabilities P(O_0..O_t, S_t = i).
        
        log_prob : float
            Log probability of the observation sequence, log P(O | model)
        """
        self._check_is_fitted()
        alpha, _c, log_prob = self._forward_pass(observations, scale=scale)
        return alpha, log_prob

    def backward(self, observations, c=None):
        """
        Backward Algorithm: Calculate backward probabilities
        
        Computes β(t, i) = P(O_t+1, O_t+2, ..., O_T | state_t = i, model)
        
        This is the probability of observing the remaining sequence
        given that we are in state i at time t
        
        Parameters:
        -----------
        observations : list
            Sequence of observation indices
        
        c : array, shape (T,), optional
            Scaling coefficients from the forward pass. If given, the SAME c_t is
            applied to beta at every step, which is what makes the scaled gamma and
            xi formulas exact (see the class docstring). If None (the default), the
            raw textbook beta is returned, for which the classic identity
            sum_i alpha(t, i) * beta(t, i) = P(O | model) holds at every t.
        
        Returns:
        --------
        beta : array, shape (T, n_states)
            Backward probabilities
            beta[t, i] = probability of observations[t+1:] given state i at time t
        """
        self._check_is_fitted()

        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization: β(T-1, i) = 1   (scaled: c_{T-1})
        beta[T-1] = 1.0 if c is None else c[T-1]
        
        # Recursion: β(t, i) = Σ A(i,j) * B(j, O_t+1) * β(t+1, j)
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_prob[i] *
                                   self.emission_prob[:, observations[t+1]] *
                                   beta[t+1])
            if c is not None:
                beta[t] = beta[t] * c[t]
        
        return beta

    def viterbi(self, observations):
        """
        Viterbi Algorithm: Find most likely sequence of hidden states
        
        Uses dynamic programming to find the state sequence that
        maximizes P(states | observations)
        
        This is the "decoding" problem: given observations, what are the hidden states?
        
        Numerical note: the recursion below is done in LOG space, i.e.
            log delta(t, j) = max_i [log delta(t-1, i) + log a_ij] + log b_j(O_t)
        which is the same recursion as the hand-worked trace in _23_hmm.md with
        every product turned into a sum. Working in log space means the argmax
        (and therefore the returned path) stays correct for sequences of any
        length, where multiplying raw probabilities would underflow to 0.0 after
        roughly 300 steps and make every state look equally good.
        
        Parameters:
        -----------
        observations : list
            Sequence of observations (can be labels or indices)
        
        Returns:
        --------
        path : list
            Most likely sequence of hidden state labels
        
        prob : float
            Probability of the most likely path, exp(max_i log delta(T-1, i)).
            For long sequences this genuinely underflows to 0.0 while the path
            itself remains correct - compare paths by log probability instead.
        """
        self._check_is_fitted()
        
        # Accept labels or indices (checks the whole sequence, not just its head)
        observations = self._encode_observations(observations)
        
        T = len(observations)
        
        # log delta(t, i) = log of the max probability of any state sequence that
        # ends in state i at time t
        log_delta = np.zeros((T, self.n_states))
        # ψ(t, i) = argmax for backtracking
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # log(0) = -inf is the right answer for an impossible transition/emission,
        # so silence numpy's divide-by-zero warning rather than adding an epsilon.
        with np.errstate(divide='ignore'):
            log_pi = np.log(self.initial_prob)
            log_A = np.log(self.transition_prob)
            log_B = np.log(self.emission_prob)
        
        # Initialization: log delta(0, i) = log pi(i) + log B(i, O_0)
        log_delta[0] = log_pi + log_B[:, observations[0]]
        
        # Recursion: log delta(t, j) = max_i [log delta(t-1, i) + log A(i,j)] + log B(j, O_t)
        for t in range(1, T):
            for j in range(self.n_states):
                # Score of arriving in state j from each possible previous state
                scores = log_delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(scores)
                log_delta[t, j] = np.max(scores) + log_B[j, observations[t]]
        
        # Termination: Find best final state
        path_indices = np.zeros(T, dtype=int)
        path_indices[T-1] = np.argmax(log_delta[T-1])
        max_log_prob = np.max(log_delta[T-1])
        
        # Backtracking: trace back the best path
        for t in range(T-2, -1, -1):
            path_indices[t] = psi[t+1, path_indices[t+1]]
        
        # Convert to labels
        path = self._decode_sequence(path_indices, self.state_labels)
        
        # Report a probability, as documented; exp(-inf) = 0.0 is handled cleanly
        max_prob = float(np.exp(max_log_prob))
        
        return path, max_prob

    def fit(self, observations_sequences, n_iter=100, tolerance=1e-4, verbose=False,
            random_state=None):
        """
        Train HMM using Baum-Welch Algorithm (EM for HMM)
        
        Learns model parameters from observation sequences. This is UNSUPERVISED:
        the training data contains observations only, never state labels, so the
        learned state indices are arbitrary (see "label switching" below).
        
        The Baum-Welch algorithm is an Expectation-Maximization (EM) algorithm:
        - E-step: Calculate expected state occupancies using Forward-Backward
        - M-step: Update parameters to maximize likelihood
        
        The three M-step formulas implemented below are:
            pi_i  = gamma_0(i)                                        (averaged over sequences)
            a_ij  = sum_{t=0}^{T-2} xi_t(i, j) / sum_{t=0}^{T-2} gamma_t(i)
            b_ik  = sum_{t: O_t = k} gamma_t(i) / sum_{t=0}^{T-1} gamma_t(i)
        Each denominator is obtained by normalizing the corresponding accumulator
        row, because summing xi over j gives exactly sum_t gamma_t(i) and summing
        the emission counts over k gives exactly sum_t gamma_t(i).
        
        Why EM works: each iteration maximizes a lower bound on the log-likelihood
        that touches it at the current parameters, so log P(O | lambda_new) can
        never be smaller than log P(O | lambda_old). The likelihood therefore
        increases monotonically to a LOCAL maximum - which one depends entirely on
        the random initialization, so pass different random_state values to explore.
        
        Label switching: nothing distinguishes "state 0" from "state 1" in the
        objective, so two runs can learn the same model with the rows of A and B
        swapped. Always interpret learned states by looking at their emission rows,
        never by their index.
        
        Parameters:
        -----------
        observations_sequences : list of lists
            Multiple sequences of observations for training. Sequences may have
            different lengths.
            Example: [['Walk', 'Shop', 'Clean'], ['Walk', 'Walk', 'Clean']]
        
        n_iter : int, default=100
            Maximum number of EM iterations
            - More iterations: closer to the local optimum, slower
            - Training stops early once the likelihood gain drops below tolerance
            Typical values: 50-500
        
        tolerance : float, default=1e-4
            Convergence threshold on the INCREASE in total log-likelihood
            - Larger: stops sooner, coarser fit
            - Smaller: runs longer, marginal gains
            Typical values: 1e-6 to 1e-3
        
        verbose : bool, default=False
            Print the log-likelihood at every iteration
        
        random_state : int or numpy.random.RandomState, default=None
            Seed for the random initialization of A and B. Baum-Welch only finds a
            local optimum, so this controls WHICH optimum you land in. Pass an int
            for a reproducible run; pass None to use the global numpy RNG (so
            np.random.seed(42) still works).
        
        Returns:
        --------
        self : HiddenMarkovModel
            Fitted model (enables chaining)
        
        Note on the reported likelihood: the value printed at iteration k is
        computed with the parameters produced by iteration k-1, which is why the
        final printed number is one M-step behind the returned model.
        """
        if self.n_states is None:
            raise ValueError(
                "n_states must be set before fit(); e.g. HiddenMarkovModel(n_states=2)"
            )
        if len(observations_sequences) == 0:
            raise ValueError("observations_sequences must contain at least one sequence")
        
        # Private RNG so that fitting never disturbs (or depends on) global state
        # more than the caller asked for.
        if random_state is None:
            rng = np.random
        elif isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            rng = np.random.RandomState(random_state)
        
        # Build observation vocabulary
        unique_obs = set()
        for seq in observations_sequences:
            unique_obs.update(seq)
        
        self.observation_labels = sorted(list(unique_obs))
        self.observation_map = {obs: i for i, obs in enumerate(self.observation_labels)}
        self.n_observations = len(self.observation_labels)
        
        # Set state labels if not already set. These are arbitrary placeholders:
        # Baum-Welch never sees a state label, so S0/S1/... are cluster ids that
        # you must interpret yourself from the learned emission rows.
        if not self.state_labels or len(self.state_labels) != self.n_states:
            self.state_labels = [f"S{i}" for i in range(self.n_states)]
        self.state_map = {s: i for i, s in enumerate(self.state_labels)}
        
        # Encode observation sequences
        encoded_sequences = []
        for seq in observations_sequences:
            if len(seq) == 0:
                raise ValueError("Training sequences must contain at least one observation")
            encoded_sequences.append(self._encode_sequence(seq, self.observation_map))
        
        # Initialize parameters randomly (symmetry must be broken - see the method)
        self._initialize_parameters(encoded_sequences, rng=rng)
        
        prev_log_likelihood = float('-inf')
        
        # EM iterations
        for iteration in range(n_iter):
            # E-step and M-step combined for all sequences
            new_initial = np.zeros(self.n_states)
            new_transition = np.zeros((self.n_states, self.n_states))
            new_emission = np.zeros((self.n_states, self.n_observations))
            
            total_log_likelihood = 0
            
            for obs_seq in encoded_sequences:
                # E-step: scaled Forward-Backward algorithm.
                # c holds the per-column scaling coefficients; backward() must use
                # the SAME ones or the identities below do not hold.
                alpha, c, log_prob = self._forward_pass(obs_seq, scale=True)
                beta = self.backward(obs_seq, c=c)
                
                total_log_likelihood += log_prob
                
                T = len(obs_seq)
                
                # Calculate γ(t, i) = P(state_t = i | O, model)
                #             = alpha_hat(t, i) * beta_hat(t, i) / c_t
                # No epsilon needed: with scaling each row already sums to exactly 1.
                gamma = alpha * beta / c[:, np.newaxis]
                
                # Calculate ξ(t, i, j) = P(state_t = i, state_t+1 = j | O, model)
                #             = alpha_hat(t, i) * A(i,j) * B(j, O_t+1) * beta_hat(t+1, j)
                # The scale factors cancel exactly, so there is no denominator here.
                xi = np.zeros((max(T-1, 0), self.n_states, self.n_states))
                for t in range(T-1):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[t, i, j] = (alpha[t, i] * self.transition_prob[i, j] *
                                          self.emission_prob[j, obs_seq[t+1]] * beta[t+1, j])
                
                # M-step: accumulate the expected counts
                # Update initial probabilities: pi_i <- gamma_0(i)
                new_initial += gamma[0]
                
                # Update transition probabilities: numerator of a_ij
                if T > 1:
                    new_transition += np.sum(xi, axis=0)
                
                # Update emission probabilities: numerator of b_ik
                for k in range(self.n_observations):
                    mask = (np.array(obs_seq) == k)
                    new_emission[:, k] += np.sum(gamma[mask], axis=0)
            
            # Normalize the accumulators into probability distributions.
            # Row-normalizing is exactly the division by sum_t gamma_t(i) in the
            # formulas: summing xi over j gives sum_t gamma_t(i), and summing the
            # emission counts over k gives sum_t gamma_t(i).
            # A row can only be all-zero if a state has zero expected occupancy
            # (or every sequence has length 1, so there are no transitions at all);
            # in that case keep the previous row rather than divide by zero.
            self.initial_prob = new_initial / np.sum(new_initial)
            for i in range(self.n_states):
                trans_total = np.sum(new_transition[i])
                if trans_total > 0:
                    self.transition_prob[i] = new_transition[i] / trans_total
                emit_total = np.sum(new_emission[i])
                if emit_total > 0:
                    self.emission_prob[i] = new_emission[i] / emit_total
            
            # Check convergence
            if verbose:
                print(f"Iteration {iteration + 1}: Log-Likelihood = {total_log_likelihood:.4f}")
            
            # Signed test, not abs(): EM is monotone, so a "change" smaller than
            # tolerance only counts as convergence when the likelihood went UP.
            # A decrease means something is numerically wrong and should not be
            # mistaken for a converged model.
            improvement = total_log_likelihood - prev_log_likelihood
            if 0 <= improvement < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = total_log_likelihood
        
        return self

    def predict(self, observations):
        """
        Predict the most likely sequence of hidden states (Viterbi decoding)
        
        Parameters:
        -----------
        observations : list
            Sequence of observations
            
        Returns:
        --------
        states : list
            Most likely sequence of hidden states
        """
        states, _ = self.viterbi(observations)
        return states
    
    def score(self, observations):
        """
        Calculate the log probability of an observation sequence
        
        Uses the scaled Forward algorithm to compute log P(observations | model),
        so the value stays exact for sequences of any length.
        
        Because it is a log probability the result is always <= 0, and longer
        sequences score lower simply because there are more factors to multiply.
        To compare sequences of different lengths, divide by len(observations).
        
        Parameters:
        -----------
        observations : list
            Sequence of observations (can be labels or indices)
        
        Returns:
        --------
        log_prob : float
            Log probability of the observation sequence.
            -inf means the model assigns the sequence probability exactly zero.
        
        Raises:
        -------
        ValueError
            If a token is not in the model's observation vocabulary. Anomaly
            detectors should catch this explicitly - an unseen symbol is itself
            strong evidence of an anomaly.
        """
        self._check_is_fitted()
        observations = self._encode_observations(observations)
        
        _, _, log_prob = self._forward_pass(observations, scale=True)
        return log_prob

    def sample(self, n_samples=10):
        """
        Generate a random sequence from the model
        
        This is useful for:
        - Understanding what sequences the model generates
        - Testing the model
        - Data augmentation
        
        Parameters:
        -----------
        n_samples : int, default=10
            Length of sequence to generate
        
        Returns:
        --------
        observations : list
            Generated observation sequence (labels)
        
        states : list
            Hidden state sequence that generated the observations (labels)
        
        Note: sampling uses the global numpy RNG, so call np.random.seed(...)
        beforehand if you need reproducible sequences.
        """
        self._check_is_fitted()
        if self.state_labels is None:
            self.state_labels = [f"S{i}" for i in range(self.n_states)]
        if self.observation_labels is None:
            self.observation_labels = [f"O{i}" for i in range(self.n_observations)]
        
        states = []
        observations = []
        
        # Sample initial state
        state = np.random.choice(self.n_states, p=self.initial_prob)
        
        for t in range(n_samples):
            states.append(state)
            
            # Sample observation from current state
            obs = np.random.choice(self.n_observations, p=self.emission_prob[state])
            observations.append(obs)
            
            # Sample next state
            if t < n_samples - 1:
                state = np.random.choice(self.n_states, p=self.transition_prob[state])
        
        # Convert to labels
        state_labels = self._decode_sequence(states, self.state_labels)
        obs_labels = self._decode_sequence(observations, self.observation_labels)
        
        return obs_labels, state_labels
    
    def print_parameters(self):
        """
        Print model parameters in a readable format
        
        All output is plain ASCII so that it survives a cp1252 Windows console.
        """
        self._check_is_fitted()
        
        print("\n" + "="*70)
        print("HIDDEN MARKOV MODEL PARAMETERS")
        print("="*70)
        
        print("\n1. Initial State Probabilities (pi):")
        print("-" * 40)
        for i, label in enumerate(self.state_labels):
            print(f"  P({label}) = {self.initial_prob[i]:.4f}")
        
        print("\n2. State Transition Probabilities (A):")
        print("-" * 40)
        # The literal lives outside the f-string: a backslash inside a replacement
        # field is only legal on Python 3.12+ (PEP 701).
        transition_header = 'From \\ To'
        print(f"{transition_header:<15}", end="")
        for label in self.state_labels:
            print(f"{label:>12}", end="")
        print()
        print("-" * (15 + 12 * self.n_states))
        
        for i, from_label in enumerate(self.state_labels):
            print(f"{from_label:<15}", end="")
            for j in range(self.n_states):
                print(f"{self.transition_prob[i, j]:>12.4f}", end="")
            print()
        
        print("\n3. Emission Probabilities (B):")
        print("-" * 40)
        emission_header = 'State \\ Obs'
        print(f"{emission_header:<15}", end="")
        for label in self.observation_labels[:min(6, len(self.observation_labels))]:
            print(f"{label:>12}", end="")
        if len(self.observation_labels) > 6:
            print("   ...")
        else:
            print()
        print("-" * (15 + 12 * min(6, len(self.observation_labels))))
        
        for i, state_label in enumerate(self.state_labels):
            print(f"{state_label:<15}", end="")
            for j in range(min(6, len(self.observation_labels))):
                print(f"{self.emission_prob[i, j]:>12.4f}", end="")
            if len(self.observation_labels) > 6:
                print("   ...")
            else:
                print()


"""
USAGE EXAMPLE 1: Weather Prediction (Simple Example)

import numpy as np

# A classic HMM example: predicting weather from activities
# Hidden states: Weather (Sunny, Rainy)
# Observations: Activities (Walk, Shop, Clean)

# Scenario: You're in a room without windows. You can only observe what
# your roommate does, and you want to infer the weather outside.

# Create HMM
hmm = HiddenMarkovModel()

# Set parameters based on domain knowledge
initial_prob = [0.6, 0.4]  # 60% chance of starting sunny

transition_prob = [
    [0.7, 0.3],  # Sunny: 70% stay sunny, 30% become rainy
    [0.4, 0.6]   # Rainy: 40% become sunny, 60% stay rainy
]

emission_prob = [
    [0.6, 0.3, 0.1],  # Sunny: 60% walk, 30% shop, 10% clean
    [0.1, 0.2, 0.7]   # Rainy: 10% walk, 20% shop, 70% clean
]

hmm.set_parameters(
    initial_prob=initial_prob,
    transition_prob=transition_prob,
    emission_prob=emission_prob,
    state_labels=['Sunny', 'Rainy'],
    observation_labels=['Walk', 'Shop', 'Clean']
)

# Print model parameters
hmm.print_parameters()

# Observe activities over 5 days
observations = ['Walk', 'Shop', 'Clean', 'Clean', 'Walk']

# Predict weather (most likely hidden states)
predicted_weather = hmm.predict(observations)
print("\nObserved Activities:", observations)
print("Predicted Weather:  ", predicted_weather)

# Calculate probability of this observation sequence
log_prob = hmm.score(observations)
print(f"\nLog probability of sequence: {log_prob:.4f}")
print(f"Probability: {np.exp(log_prob):.6f}")

# Output shows:
# When activities are [Walk, Shop, Clean, Clean, Walk]
# Most likely weather is [Sunny, Sunny, Rainy, Rainy, Sunny]
# Log probability of sequence: -5.3688
# Probability: 0.004660
# (0.004660 is the sum over all 2^5 = 32 possible weather paths;
#  the single best path alone is worth only 0.00160030)
"""

"""
USAGE EXAMPLE 2: Part-of-Speech Tagging

import numpy as np

# POS tagging: Given a sentence, determine the part of speech for each word
# Hidden states: POS tags (Noun, Verb, Adjective)
# Observations: Words in the sentence

np.random.seed(42)  # Baum-Welch starts from a random point - seed it to reproduce

# Training data: untagged sentences (Baum-Welch is unsupervised - it never sees
# a POS tag, so it can only discover word clusters, not name them)
sentences = [
    ['the', 'dog', 'runs', 'fast'],
    ['a', 'cat', 'sleeps'],
    ['the', 'quick', 'fox', 'jumps'],
    ['dogs', 'run', 'quickly'],
    ['the', 'cat', 'runs']
]

# Create and train HMM
hmm = HiddenMarkovModel(n_states=3)  # 3 POS tags
hmm.fit(sentences, n_iter=50, verbose=True)

# Test: predict POS tags for new sentence
test_sentence = ['the', 'dog', 'runs']
predicted_tags = hmm.predict(test_sentence)

print("\nSentence:", test_sentence)
print("Predicted state ids:", predicted_tags)

# IMPORTANT: those are 'S0'/'S1'/'S2', not 'NOUN'/'VERB'/'DET'. Baum-Welch is
# unsupervised, so the states are anonymous clusters and their numbering is
# arbitrary (run it with a different seed and the ids permute - this is called
# "label switching"). To turn them into real POS tags you must look at each
# state's emission row and name it by hand, e.g.:
#   for i, label in enumerate(hmm.state_labels):
#       top = np.argsort(hmm.emission_prob[i])[::-1][:3]
#       print(label, [hmm.observation_labels[k] for k in top])
# A state whose top words are ['the', 'a'] is your determiner state.

# Generate random sentence from the model
generated_words, generated_tags = hmm.sample(n_samples=5)
print("\nGenerated sentence:", generated_words)
print("Generated POS tags:", generated_tags)
"""

"""
USAGE EXAMPLE 3: Stock Market Regime Detection

# Detect market regimes (Bull, Bear, Sideways) from price movements
# Hidden states: Market regime
# Observations: Daily returns (categorized)

import numpy as np

# Categorize daily returns
def categorize_returns(returns):
    # Categorize as: Large Down, Down, Flat, Up, Large Up
    categories = []
    for r in returns:
        if r < -2:
            categories.append('Large Down')
        elif r < -0.5:
            categories.append('Down')
        elif r < 0.5:
            categories.append('Flat')
        elif r < 2:
            categories.append('Up')
        else:
            categories.append('Large Up')
    return categories

# Simulated daily returns (%)
daily_returns = [1.2, 0.8, -0.3, 1.5, 2.1, -1.8, -2.5, -1.2, 0.2, 1.0]
observations = categorize_returns(daily_returns)

# Create HMM with 3 market regimes
hmm = HiddenMarkovModel()

# Set parameters (simplified example)
hmm.set_parameters(
    initial_prob=[0.4, 0.3, 0.3],  # Bull, Bear, Sideways
    transition_prob=[
        [0.8, 0.1, 0.1],  # Bull: likely stays bull
        [0.1, 0.8, 0.1],  # Bear: likely stays bear
        [0.2, 0.2, 0.6]   # Sideways: most stable
    ],
    emission_prob=[
        [0.05, 0.1, 0.15, 0.35, 0.35],  # Bull: mostly up
        [0.35, 0.35, 0.15, 0.1, 0.05],  # Bear: mostly down
        [0.1, 0.2, 0.4, 0.2, 0.1]       # Sideways: mostly flat
    ],
    state_labels=['Bull', 'Bear', 'Sideways'],
    observation_labels=['Large Down', 'Down', 'Flat', 'Up', 'Large Up']
)

# Predict market regimes
predicted_regimes = hmm.predict(observations)

print("\nDaily Returns (%):", daily_returns)
print("Return Categories:", observations)
print("Predicted Regimes:", predicted_regimes)

# This helps traders understand:
# - Current market regime
# - When regime changes occur
# - Adjust strategies accordingly
"""

"""
USAGE EXAMPLE 4: Speech Recognition (Simplified)

import numpy as np

# Phoneme recognition from acoustic features
# Hidden states: Phonemes (simplified to vowels)
# Observations: Acoustic features (categorized)

np.random.seed(42)  # reproducible Baum-Welch initialization

# Training sequences: phoneme sequences
phoneme_sequences = [
    ['A', 'E', 'I', 'O', 'U'],
    ['A', 'A', 'E', 'I', 'O'],
    ['E', 'I', 'I', 'O', 'U'],
    ['A', 'E', 'E', 'O', 'U', 'U'],
    ['O', 'U', 'A', 'E', 'I']
]

# Create and train HMM
hmm = HiddenMarkovModel(n_states=5)  # 5 vowel phonemes
hmm.fit(phoneme_sequences, n_iter=100, verbose=True)

# Test recognition
test_sequence = ['A', 'E', 'I', 'O']
predicted_phonemes = hmm.predict(test_sequence)

print("\nObserved acoustic patterns:", test_sequence)
print("Recognized phonemes:", predicted_phonemes)

# Real speech recognition uses:
# - More complex acoustic features (MFCCs)
# - More phonemes
# - Larger training data
# - Deep learning for better accuracy
"""

"""
USAGE EXAMPLE 5: DNA Sequence Analysis (Gene Finding)

import numpy as np

# Find genes in DNA sequences
# Hidden states: Gene regions (Coding, Non-coding)
# Observations: Nucleotides (A, T, G, C)

np.random.seed(42)  # reproducible Baum-Welch initialization

# DNA sequences
dna_sequences = [
    ['A', 'T', 'G', 'C', 'A', 'T'],
    ['G', 'C', 'A', 'T', 'T', 'A'],
    ['A', 'A', 'T', 'G', 'C', 'C'],
    ['T', 'A', 'G', 'C', 'A', 'T']
]

# Create and train HMM
hmm = HiddenMarkovModel(n_states=2)  # Coding vs Non-coding
hmm.fit(dna_sequences, n_iter=50, verbose=True)

# Predict gene regions in new sequence
new_dna = ['A', 'T', 'G', 'C', 'A', 'T', 'G', 'C']
gene_regions = hmm.predict(new_dna)

print("\nDNA Sequence:", ''.join(new_dna))
print("Gene Regions:", gene_regions)

# Applications:
# - Gene prediction
# - Finding regulatory regions
# - Identifying splice sites
# - Detecting mutations
"""

"""
USAGE EXAMPLE 6: User Behavior Modeling (E-commerce)

import numpy as np

# Model user journey on e-commerce website
# Hidden states: User intent (Browsing, Searching, Buying)
# Observations: Actions (View, Click, Add to Cart, Purchase)

np.random.seed(42)  # reproducible Baum-Welch initialization

# User session data
sessions = [
    ['View', 'View', 'Click', 'Add to Cart', 'Purchase'],
    ['View', 'Click', 'View', 'View'],
    ['View', 'Add to Cart', 'Purchase'],
    ['Click', 'Click', 'Add to Cart', 'Add to Cart', 'Purchase'],
    ['View', 'View', 'View', 'Click']
]

# Train HMM
hmm = HiddenMarkovModel(n_states=3)
hmm.fit(sessions, n_iter=100, verbose=True)

# Predict user intent for new session
current_session = ['View', 'View', 'Click', 'Add to Cart']
predicted_intent = hmm.predict(current_session)

print("\nUser Actions:", current_session)
print("Predicted Intent:", predicted_intent)

# Business applications:
# - Personalized recommendations at each step
# - Identify users likely to abandon cart
# - Optimize user experience based on intent
# - Targeted promotions
"""

"""
USAGE EXAMPLE 7: Comparing Different Models

# Compare HMMs with different numbers of hidden states

import numpy as np

np.random.seed(42)  # without this, every run prints different log-probabilities

# Generate training data
training_data = [
    ['A', 'B', 'A', 'C', 'A', 'B'],
    ['B', 'C', 'B', 'A', 'C', 'B'],
    ['A', 'A', 'B', 'C', 'C', 'A'],
    ['C', 'B', 'A', 'B', 'C', 'C']
]

# Test data
test_data = ['A', 'B', 'C', 'A']

# Try different numbers of states
for n_states in [2, 3, 4]:
    print(f"\n{'='*50}")
    print(f"Testing HMM with {n_states} hidden states")
    print('='*50)
    
    hmm = HiddenMarkovModel(n_states=n_states)
    hmm.fit(training_data, n_iter=50, verbose=False)
    
    # Evaluate on test data
    log_prob = hmm.score(test_data)
    predicted_states = hmm.predict(test_data)
    
    print(f"Log probability: {log_prob:.4f}")
    print(f"Test sequence: {test_data}")
    print(f"Predicted states: {predicted_states}")

# Model selection:
# - Use cross-validation
# - Compare log-likelihoods
# - Consider model complexity (avoid overfitting)
# - Domain knowledge about number of states
#
# CAUTION: log probability on the TRAINING data always improves with more states,
# because a bigger model can always memorize more. The comparison above is only
# meaningful because test_data is held out. For a principled choice, penalize the
# parameter count with BIC: -2*logL + (N^2 + N*M - 1) * log(n_observations_seen).
"""

"""
USAGE EXAMPLE 8: Anomaly Detection

import numpy as np

# Detect anomalous sequences using HMM

np.random.seed(42)  # reproducible Baum-Welch initialization

# Train on normal behavior
normal_sequences = [
    ['Login', 'Browse', 'Logout'],
    ['Login', 'Browse', 'Browse', 'Logout'],
    ['Login', 'Browse', 'Purchase', 'Logout'],
    ['Login', 'Browse', 'Browse', 'Purchase', 'Logout']
]

hmm = HiddenMarkovModel(n_states=3)
hmm.fit(normal_sequences, n_iter=100, verbose=False)

# Test sequences
test_sequences = [
    ['Login', 'Browse', 'Logout'],           # Normal
    ['Login', 'Browse', 'Purchase', 'Logout'], # Normal
    ['Login', 'Admin', 'Admin', 'Download']    # Anomalous
]

print("\nAnomaly Detection:")
print("="*50)

for i, seq in enumerate(test_sequences, 1):
    # Catch ValueError specifically: score() raises it for a symbol that was never
    # seen in training. A bare 'except:' would also swallow real bugs (a typo, a
    # missing import) and silently report every sequence as an anomaly.
    try:
        log_prob = hmm.score(seq)
        prob = np.exp(log_prob)

        # Low probability indicates anomaly. Normalize by length so that a long
        # normal sequence is not flagged just for being long.
        is_anomaly = (log_prob / len(seq)) < -2.0  # threshold, in nats per step

        print(f"\nSequence {i}: {seq}")
        print(f"Log Probability: {log_prob:.4f}  ({log_prob/len(seq):.4f} per step)")
        print(f"Probability: {prob:.6f}")
        print(f"Anomaly: {'YES' if is_anomaly else 'NO'}")
    except ValueError as e:
        print(f"\nSequence {i}: {seq}")
        print(f"Contains unknown actions - ANOMALY ({e})")

# Output (with np.random.seed(42)):
# Sequence 1: Log Probability -0.8107 (-0.2702 per step) -> Anomaly: NO
# Sequence 2: Log Probability -1.9248 (-0.4812 per step) -> Anomaly: NO
# Sequence 3: Contains unknown actions - ANOMALY

# Applications:
# - Intrusion detection
# - Fraud detection
# - Quality control
# - System monitoring
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _23_hmm.py
    # Requires numpy only. Runs in a couple of seconds.
    # ----------------------------------------------------------------
    np.random.seed(42)

    print("=" * 62)
    print("HIDDEN MARKOV MODEL - Plug-and-Play Demo")
    print("=" * 62)
    print("An HMM has hidden states you cannot see and observations you can.")
    print("Demo 1 solves the first two classic problems - Evaluation and")
    print("        Decoding - on a model whose parameters we already know.")
    print("Demo 2 solves the third - Learning - recovering a model from data")
    print("        it generated itself.")
    print("Demo 3 uses the likelihood as an anomaly detector.")

    # ================================================================
    # DEMO 1 - Parameters known: Evaluation, Decoding, and sampling
    # ================================================================
    print("\n" + "=" * 62)
    print("DEMO 1 - Weather from activities (parameters given by hand)")
    print("=" * 62)
    print("You are in a windowless room. The weather is HIDDEN; all you see is")
    print("what your roommate does: Walk, Shop or Clean.")

    weather = HiddenMarkovModel()
    weather.set_parameters(
        initial_prob=[0.6, 0.4],
        transition_prob=[[0.7, 0.3],    # Sunny -> Sunny 0.7, Sunny -> Rainy 0.3
                         [0.4, 0.6]],   # Rainy -> Sunny 0.4, Rainy -> Rainy 0.6
        emission_prob=[[0.6, 0.3, 0.1],  # Sunny: mostly walking
                       [0.1, 0.2, 0.7]], # Rainy: mostly cleaning
        state_labels=['Sunny', 'Rainy'],
        observation_labels=['Walk', 'Shop', 'Clean']
    )
    weather.print_parameters()

    # --- Problem 1: EVALUATION (forward algorithm) ---
    print("\n--- Problem 1: EVALUATION - how likely is this sequence? ---")
    obs3 = ['Walk', 'Shop', 'Clean']
    log_p3 = weather.score(obs3)
    print(f"Observations : {obs3}")
    print(f"log P(O)     : {log_p3:.4f}")
    print(f"P(O)         : {np.exp(log_p3):.6f}")
    print("The forward algorithm sums over all 2^3 = 8 weather paths in O(N^2 T).")

    # --- Problem 2: DECODING (Viterbi algorithm) ---
    print("\n--- Problem 2: DECODING - which weather explains it best? ---")
    obs5 = ['Walk', 'Shop', 'Clean', 'Clean', 'Walk']
    path, path_prob = weather.viterbi(obs5)
    print(f"Observations : {obs5}")
    print(f"Best states  : {path}")
    print(f"P(best path, O) = {path_prob:.8f}")
    print(f"P(O) over ALL paths = {np.exp(weather.score(obs5)):.8f}")
    print("The single best path carries only part of the total probability -")
    print("that gap is exactly the difference between max (Viterbi) and sum (forward).")

    # --- Sampling from the model ---
    print("\n--- Sampling: what does this model think a week looks like? ---")
    sample_obs, sample_states = weather.sample(n_samples=7)
    print(f"Hidden weather : {sample_states}")
    print(f"Seen activities: {sample_obs}")

    # ================================================================
    # DEMO 2 - Learning: recover the model from its own samples
    # ================================================================
    print("\n" + "=" * 62)
    print("DEMO 2 - LEARNING with Baum-Welch (train / test)")
    print("=" * 62)
    print("Hidden states: market regime (Bull / Bear). Observed: daily move")
    print("(Down / Flat / Up). We generate data from a KNOWN model, throw the")
    print("regimes away, and see how much Baum-Welch recovers from moves alone.")

    market = HiddenMarkovModel()
    market.set_parameters(
        initial_prob=[0.5, 0.5],
        transition_prob=[[0.90, 0.10],    # Bull regimes persist
                         [0.15, 0.85]],   # so do Bear regimes
        emission_prob=[[0.10, 0.20, 0.70],  # Bull: mostly Up days
                       [0.65, 0.25, 0.10]], # Bear: mostly Down days
        state_labels=['Bull', 'Bear'],
        observation_labels=['Down', 'Flat', 'Up']
    )

    np.random.seed(42)
    n_train, n_test, seq_len = 60, 20, 20
    train_obs, train_states = [], []
    for _ in range(n_train):
        o, s = market.sample(n_samples=seq_len)
        train_obs.append(o)
        train_states.append(s)
    test_obs, test_states = [], []
    for _ in range(n_test):
        o, s = market.sample(n_samples=seq_len)
        test_obs.append(o)
        test_states.append(s)
    print(f"\nTrain: {n_train} sequences x {seq_len} days.  Test: {n_test} x {seq_len}.")
    print(f"Example training sequence: {train_obs[0][:10]} ...")

    learned = HiddenMarkovModel(n_states=2)
    learned.fit(train_obs, n_iter=200, tolerance=1e-6, random_state=0)

    def avg_ll(model, sequences):
        """Average log-likelihood PER OBSERVATION (comparable across lengths)."""
        total = sum(model.score(seq) for seq in sequences)
        steps = sum(len(seq) for seq in sequences)
        return total / steps

    print("\nAverage log-likelihood per observation (higher = better, max 0):")
    print(f"  Train (learned model) : {avg_ll(learned, train_obs):.4f}")
    print(f"  Test  (learned model) : {avg_ll(learned, test_obs):.4f}")
    print(f"  Test  (TRUE model)    : {avg_ll(market, test_obs):.4f}   <- the target")
    print(f"  Test  (no memory)     : {np.log(1.0/3.0):.4f}   <- guessing uniformly")

    # Unsupervised state ids are arbitrary ("label switching"), so line the learned
    # states up with the true ones by trying both orderings and keeping the better.
    def decode_accuracy(model, sequences, truths, order):
        correct = total = 0
        for seq, truth in zip(sequences, truths):
            pred = model.predict(seq)
            for p, t in zip(pred, truth):
                correct += (order[model.state_map[p]] == t)
                total += 1
        return correct / total

    orders = [['Bull', 'Bear'], ['Bear', 'Bull']]
    train_accs = [decode_accuracy(learned, train_obs, train_states, o) for o in orders]
    best = int(np.argmax(train_accs))          # alignment chosen on TRAIN only
    test_acc = decode_accuracy(learned, test_obs, test_states, orders[best])
    ceiling = decode_accuracy(market, test_obs, test_states, orders[0])

    print("\nHidden-regime recovery (Viterbi decoding, per day):")
    print(f"  Train accuracy (learned) : {train_accs[best]:.2%}")
    print(f"  Test  accuracy (learned) : {test_acc:.2%}")
    print(f"  Test  accuracy (TRUE)    : {ceiling:.2%}   <- the ceiling: a Flat day")
    print("                                       is ambiguous even for the true model.")
    mapping = ", ".join(f"{learned.state_labels[i]} = {orders[best][i]}"
                        for i in range(learned.n_states))
    print(f"  Learned states line up as: {mapping}")
    print("  That numbering is arbitrary (label switching) - always read the")
    print("  emission rows to decide what a learned state means.")

    print("\nEmission matrix B, learned vs true (rows aligned as above):")
    print(f"  {'':8s}{'Down':>10s}{'Flat':>10s}{'Up':>10s}")
    true_rows = {'Bull': [0.10, 0.20, 0.70], 'Bear': [0.65, 0.25, 0.10]}
    for i, name in enumerate(orders[best]):
        row = learned.emission_prob[i]
        print(f"  {name:8s}{row[0]:10.3f}{row[1]:10.3f}{row[2]:10.3f}   learned")
        t = true_rows[name]
        print(f"  {'':8s}{t[0]:10.3f}{t[1]:10.3f}{t[2]:10.3f}   true")
    print("EM finds a LOCAL optimum, so results depend on random_state -")
    print("try 1, 2, 3 ... and keep the run with the highest training likelihood.")

    # ================================================================
    # DEMO 3 - Anomaly detection from the likelihood
    # ================================================================
    print("\n" + "=" * 62)
    print("DEMO 3 - ANOMALY DETECTION on web sessions")
    print("=" * 62)

    np.random.seed(42)
    normal_sessions = [
        ['Login', 'Browse', 'Logout'],
        ['Login', 'Browse', 'Browse', 'Logout'],
        ['Login', 'Browse', 'Purchase', 'Logout'],
        ['Login', 'Browse', 'Browse', 'Purchase', 'Logout']
    ]
    detector = HiddenMarkovModel(n_states=3)
    detector.fit(normal_sessions, n_iter=100, random_state=42)

    candidates = [
        ['Login', 'Browse', 'Logout'],              # seen in training
        ['Login', 'Browse', 'Purchase', 'Logout'],  # seen in training
        ['Logout', 'Purchase', 'Login', 'Browse'],  # known words, wrong order
        ['Login', 'Admin', 'Admin', 'Download']     # unknown actions
    ]
    print("Rule: flag a session when its log-likelihood per step drops below -2.0,")
    print("or when it uses an action the model has never seen.\n")
    for i, session in enumerate(candidates, 1):
        try:
            lp = detector.score(session)
            per_step = lp / len(session)
            verdict = "ANOMALY" if per_step < -2.0 else "normal "
            print(f"  {i}. {verdict}  log P = {lp:8.4f}  per step = {per_step:7.4f}  {session}")
        except ValueError:
            print(f"  {i}. ANOMALY  unknown action in sequence            {session}")
    print("\nlog P = -inf means the model gives that session probability exactly 0:")
    print("the actions are all known, but the model learned that no session ever")
    print("starts with 'Logout'. Catching ValueError separately keeps a genuinely")
    print("unknown action from being confused with a merely improbable session.")

    print("\n" + "=" * 62)
    print("Done. Read _23_hmm.md for the theory behind every step above.")
    print("=" * 62)
