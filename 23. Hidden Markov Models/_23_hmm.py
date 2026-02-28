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
    """
    
    def __init__(self, n_states=None, n_observations=None):
        """
        Initialize the Hidden Markov Model
        
        Parameters:
        -----------
        n_states : int, optional
            Number of hidden states in the model
            Example: For weather, might be 2 (Sunny, Rainy)
        
        n_observations : int, optional
            Number of possible observations
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
    
    def _initialize_parameters(self, observations):
        """
        Initialize HMM parameters randomly
        
        This is used as starting point for the Baum-Welch algorithm
        
        Parameters:
        -----------
        observations : list of lists
            Training sequences of observations
        """
        # Initialize with random probabilities
        # Initial probabilities: uniform distribution
        self.initial_prob = np.ones(self.n_states) / self.n_states
        
        # Transition probabilities: random + normalize
        self.transition_prob = np.random.rand(self.n_states, self.n_states)
        self.transition_prob = self.transition_prob / self.transition_prob.sum(axis=1, keepdims=True)
        
        # Emission probabilities: random + normalize
        self.emission_prob = np.random.rand(self.n_states, self.n_observations)
        self.emission_prob = self.emission_prob / self.emission_prob.sum(axis=1, keepdims=True)
    
    def set_parameters(self, initial_prob, transition_prob, emission_prob, 
                      state_labels=None, observation_labels=None):
        """
        Manually set HMM parameters (for supervised learning)
        
        Use this when you know the model parameters from domain knowledge
        or have fully labeled training data
        
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
            Names of observations (for display purposes)
            Example: ['Walk', 'Shop', 'Clean']
        """
        self.initial_prob = np.array(initial_prob)
        self.transition_prob = np.array(transition_prob)
        self.emission_prob = np.array(emission_prob)
        
        self.n_states = len(initial_prob)
        self.n_observations = emission_prob.shape[1]
        
        self.state_labels = state_labels if state_labels else [f"S{i}" for i in range(self.n_states)]
        self.observation_labels = observation_labels if observation_labels else [f"O{i}" for i in range(self.n_observations)]
        
        # Validate probabilities sum to 1
        assert np.allclose(self.initial_prob.sum(), 1.0), "Initial probabilities must sum to 1"
        assert np.allclose(self.transition_prob.sum(axis=1), 1.0).all(), "Transition probabilities must sum to 1"
        assert np.allclose(self.emission_prob.sum(axis=1), 1.0).all(), "Emission probabilities must sum to 1"
    
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
    
    def forward(self, observations):
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
            
        Returns:
        --------
        alpha : array, shape (T, n_states)
            Forward probabilities
            alpha[t, i] = probability of observations[0:t+1] and being in state i at time t
        
        log_prob : float
            Log probability of the observation sequence
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization: α(0, i) = π(i) * B(i, O_0)
        alpha[0] = self.initial_prob * self.emission_prob[:, observations[0]]
        
        # Recursion: α(t, j) = [Σ α(t-1, i) * A(i,j)] * B(j, O_t)
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_prob[:, j]) * \
                             self.emission_prob[j, observations[t]]
        
        # Termination: P(O | model) = Σ α(T-1, i)
        log_prob = np.log(np.sum(alpha[T-1]) + 1e-10)
        
        return alpha, log_prob
    
    def backward(self, observations):
        """
        Backward Algorithm: Calculate backward probabilities
        
        Computes β(t, i) = P(O_t+1, O_t+2, ..., O_T | state_t = i, model)
        
        This is the probability of observing the remaining sequence
        given that we are in state i at time t
        
        Parameters:
        -----------
        observations : list
            Sequence of observation indices
            
        Returns:
        --------
        beta : array, shape (T, n_states)
            Backward probabilities
            beta[t, i] = probability of observations[t+1:] given state i at time t
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization: β(T-1, i) = 1
        beta[T-1] = 1.0
        
        # Recursion: β(t, i) = Σ A(i,j) * B(j, O_t+1) * β(t+1, j)
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_prob[i] * 
                                   self.emission_prob[:, observations[t+1]] * 
                                   beta[t+1])
        
        return beta
    
    def viterbi(self, observations):
        """
        Viterbi Algorithm: Find most likely sequence of hidden states
        
        Uses dynamic programming to find the state sequence that
        maximizes P(states | observations)
        
        This is the "decoding" problem: given observations, what are the hidden states?
        
        Parameters:
        -----------
        observations : list
            Sequence of observations (can be labels or indices)
            
        Returns:
        --------
        path : list
            Most likely sequence of hidden state labels
        
        prob : float
            Probability of the most likely path
        """
        # Encode observations if needed
        if self.observation_map and len(observations) > 0:
            if observations[0] in self.observation_map:
                observations = self._encode_sequence(observations, self.observation_map)
        
        T = len(observations)
        
        # δ(t, i) = max probability of state sequence ending in state i at time t
        delta = np.zeros((T, self.n_states))
        # ψ(t, i) = argmax for backtracking
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization: δ(0, i) = π(i) * B(i, O_0)
        delta[0] = self.initial_prob * self.emission_prob[:, observations[0]]
        
        # Recursion: δ(t, j) = max_i [δ(t-1, i) * A(i,j)] * B(j, O_t)
        for t in range(1, T):
            for j in range(self.n_states):
                # Find best previous state
                prob_scores = delta[t-1] * self.transition_prob[:, j]
                psi[t, j] = np.argmax(prob_scores)
                delta[t, j] = np.max(prob_scores) * self.emission_prob[j, observations[t]]
        
        # Termination: Find best final state
        path_indices = np.zeros(T, dtype=int)
        path_indices[T-1] = np.argmax(delta[T-1])
        max_prob = np.max(delta[T-1])
        
        # Backtracking: trace back the best path
        for t in range(T-2, -1, -1):
            path_indices[t] = psi[t+1, path_indices[t+1]]
        
        # Convert to labels
        path = self._decode_sequence(path_indices, self.state_labels)
        
        return path, max_prob
    
    def fit(self, observations_sequences, n_iter=100, tolerance=1e-4, verbose=False):
        """
        Train HMM using Baum-Welch Algorithm (EM for HMM)
        
        Learns model parameters from observation sequences
        
        The Baum-Welch algorithm is an Expectation-Maximization (EM) algorithm:
        - E-step: Calculate expected state occupancies using Forward-Backward
        - M-step: Update parameters to maximize likelihood
        
        Parameters:
        -----------
        observations_sequences : list of lists
            Multiple sequences of observations for training
            Example: [['Walk', 'Shop', 'Clean'], ['Walk', 'Walk', 'Clean']]
        
        n_iter : int, default=100
            Maximum number of iterations
            
        tolerance : float, default=1e-4
            Convergence threshold (change in log-likelihood)
            
        verbose : bool, default=False
            Print training progress
        """
        # Build observation vocabulary
        unique_obs = set()
        for seq in observations_sequences:
            unique_obs.update(seq)
        
        self.observation_labels = sorted(list(unique_obs))
        self.observation_map = {obs: i for i, obs in enumerate(self.observation_labels)}
        self.n_observations = len(self.observation_labels)
        
        # Set state labels if not already set
        if not self.state_labels:
            self.state_labels = [f"S{i}" for i in range(self.n_states)]
        
        # Encode observation sequences
        encoded_sequences = []
        for seq in observations_sequences:
            encoded_sequences.append(self._encode_sequence(seq, self.observation_map))
        
        # Initialize parameters randomly
        self._initialize_parameters(encoded_sequences)
        
        prev_log_likelihood = float('-inf')
        
        # EM iterations
        for iteration in range(n_iter):
            # E-step and M-step combined for all sequences
            new_initial = np.zeros(self.n_states)
            new_transition = np.zeros((self.n_states, self.n_states))
            new_emission = np.zeros((self.n_states, self.n_observations))
            
            total_log_likelihood = 0
            
            for obs_seq in encoded_sequences:
                # E-step: Forward-Backward algorithm
                alpha, log_prob = self.forward(obs_seq)
                beta = self.backward(obs_seq)
                
                total_log_likelihood += log_prob
                
                T = len(obs_seq)
                
                # Calculate γ(t, i) = P(state_t = i | O, model)
                gamma = alpha * beta
                gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)
                
                # Calculate ξ(t, i, j) = P(state_t = i, state_t+1 = j | O, model)
                xi = np.zeros((T-1, self.n_states, self.n_states))
                for t in range(T-1):
                    denominator = np.sum(alpha[t] * beta[t]) + 1e-10
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[t, i, j] = (alpha[t, i] * self.transition_prob[i, j] * 
                                          self.emission_prob[j, obs_seq[t+1]] * beta[t+1, j]) / denominator
                
                # M-step: Update parameters
                # Update initial probabilities
                new_initial += gamma[0]
                
                # Update transition probabilities
                new_transition += np.sum(xi, axis=0)
                
                # Update emission probabilities
                for k in range(self.n_observations):
                    mask = (np.array(obs_seq) == k)
                    new_emission[:, k] += np.sum(gamma[mask], axis=0)
            
            # Normalize
            self.initial_prob = new_initial / (np.sum(new_initial) + 1e-10)
            self.transition_prob = new_transition / (np.sum(new_transition, axis=1, keepdims=True) + 1e-10)
            self.emission_prob = new_emission / (np.sum(new_emission, axis=1, keepdims=True) + 1e-10)
            
            # Check convergence
            if verbose:
                print(f"Iteration {iteration + 1}: Log-Likelihood = {total_log_likelihood:.4f}")
            
            if abs(total_log_likelihood - prev_log_likelihood) < tolerance:
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
        
        Uses the Forward algorithm to compute P(observations | model)
        
        Parameters:
        -----------
        observations : list
            Sequence of observations (can be labels or indices)
            
        Returns:
        --------
        log_prob : float
            Log probability of the observation sequence
        """
        # Encode observations if needed
        if self.observation_map and len(observations) > 0:
            if observations[0] in self.observation_map:
                observations = self._encode_sequence(observations, self.observation_map)
        
        _, log_prob = self.forward(observations)
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
            Generated observation sequence
        
        states : list
            Hidden state sequence that generated the observations
        """
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
        """
        print("\n" + "="*70)
        print("HIDDEN MARKOV MODEL PARAMETERS")
        print("="*70)
        
        print("\n1. Initial State Probabilities (π):")
        print("-" * 40)
        for i, label in enumerate(self.state_labels):
            print(f"  P({label}) = {self.initial_prob[i]:.4f}")
        
        print("\n2. State Transition Probabilities (A):")
        print("-" * 40)
        print(f"{'From \\ To':<15}", end="")
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
        print(f"{'State \\ Obs':<15}", end="")
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
"""

"""
USAGE EXAMPLE 2: Part-of-Speech Tagging

# POS tagging: Given a sentence, determine the part of speech for each word
# Hidden states: POS tags (Noun, Verb, Adjective)
# Observations: Words in the sentence

# Training data: sentences with POS tags
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
print("Predicted POS tags:", predicted_tags)

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

# Phoneme recognition from acoustic features
# Hidden states: Phonemes (simplified to vowels)
# Observations: Acoustic features (categorized)

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

# Find genes in DNA sequences
# Hidden states: Gene regions (Coding, Non-coding)
# Observations: Nucleotides (A, T, G, C)

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

# Model user journey on e-commerce website
# Hidden states: User intent (Browsing, Searching, Buying)
# Observations: Actions (View, Click, Add to Cart, Purchase)

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
"""

"""
USAGE EXAMPLE 8: Anomaly Detection

# Detect anomalous sequences using HMM

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
    try:
        log_prob = hmm.score(seq)
        prob = np.exp(log_prob)
        
        # Low probability indicates anomaly
        is_anomaly = log_prob < -10  # threshold
        
        print(f"\nSequence {i}: {seq}")
        print(f"Log Probability: {log_prob:.4f}")
        print(f"Anomaly: {'YES' if is_anomaly else 'NO'}")
    except:
        print(f"\nSequence {i}: {seq}")
        print("Contains unknown actions - ANOMALY")

# Applications:
# - Intrusion detection
# - Fraud detection
# - Quality control
# - System monitoring
"""
