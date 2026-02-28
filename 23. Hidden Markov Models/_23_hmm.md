# Hidden Markov Models (HMM) from Scratch: A Comprehensive Guide

Welcome to the fascinating world of Hidden Markov Models! 🔮 In this comprehensive guide, we'll explore HMMs - powerful statistical models for sequential data where the underlying process is hidden but observable through outputs. Think of it as understanding the "hidden story" behind what you can see!

## Table of Contents
1. [What are Hidden Markov Models?](#what-are-hidden-markov-models)
2. [How HMMs Work](#how-hmms-work)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [The Three Fundamental Problems](#the-three-fundamental-problems)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)

---

## What are Hidden Markov Models?

A **Hidden Markov Model (HMM)** is a statistical model where:
- The system has **hidden states** that we cannot directly observe
- We can only observe **outputs/emissions** that depend on these hidden states
- The system follows a **Markov process** where the next state depends only on the current state

**Real-world analogy**: 
Imagine you're in a room without windows, trying to figure out the weather outside. You can't see the weather (hidden state), but you can see what your roommate is doing - walking, shopping, or cleaning (observations). Over time, you learn that certain activities are more likely in certain weather conditions!

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Model Type** | Sequential Statistical Model |
| **Learning Style** | Supervised or Unsupervised |
| **Primary Use** | Sequence Analysis, Pattern Recognition |
| **Output** | State Sequences, Probabilities |
| **Key Property** | Markov Property (memoryless) |

### The Core Components

```
1. States (S): Hidden states the system can be in
   Example: Weather = {Sunny, Rainy}

2. Observations (O): Visible outputs we observe
   Example: Activities = {Walk, Shop, Clean}

3. Initial Probability (π): P(starting in each state)
   Example: π = [0.6, 0.4] → 60% chance of starting sunny

4. Transition Probability (A): P(moving from state i to state j)
   Example: A[Sunny→Rainy] = 0.3 → 30% chance of becoming rainy

5. Emission Probability (B): P(observing output k in state i)
   Example: B[Sunny→Walk] = 0.6 → 60% chance of walking when sunny
```

### When to Use HMMs

**Perfect for**:
- Sequential/temporal data
- Hidden process with observable outputs
- Pattern recognition in sequences
- State estimation problems

**Examples**:
- 🗣️ Speech recognition (phonemes → acoustic signals)
- 📝 Part-of-speech tagging (POS tags → words)
- 🧬 Gene finding (gene regions → DNA sequences)
- 📈 Market regime detection (bull/bear → price movements)
- 🌤️ Weather prediction (weather → observations)

---

## How HMMs Work

### The HMM Structure

```
Time:      t=0         t=1         t=2         t=3
           
States:    [S₀] ----→ [S₁] ----→ [S₂] ----→ [S₃]
(Hidden)    ↓          ↓          ↓          ↓
            
Observations: O₀        O₁         O₂         O₃
(Visible)
```

**Key Properties**:

1. **Markov Property** (memoryless):
   ```
   P(Sₜ | S₀, S₁, ..., Sₜ₋₁) = P(Sₜ | Sₜ₋₁)
   
   "The future depends only on the present, not the past"
   ```

2. **Output Independence**:
   ```
   P(Oₜ | S₀, S₁, ..., Sₜ, O₀, O₁, ..., Oₜ₋₁) = P(Oₜ | Sₜ)
   
   "The observation depends only on the current state"
   ```

### Visual Example: Weather & Activities

```
Weather Model:
┌─────────────────────────────────────────────┐
│ Hidden States: Sunny (S), Rainy (R)        │
│ Observations: Walk (W), Shop (Sh), Clean (C)│
└─────────────────────────────────────────────┘

Day 1: Weather=Sunny → Activity=Walk
       ↓ (transition: Sunny→Sunny with prob 0.7)
Day 2: Weather=Sunny → Activity=Shop
       ↓ (transition: Sunny→Rainy with prob 0.3)
Day 3: Weather=Rainy → Activity=Clean
       ↓ (transition: Rainy→Rainy with prob 0.6)
Day 4: Weather=Rainy → Activity=Clean
```

**Model Parameters**:

```
Initial Probabilities (π):
  P(start in Sunny) = 0.6
  P(start in Rainy) = 0.4

Transition Probabilities (A):
  From Sunny: P(Sunny→Sunny) = 0.7, P(Sunny→Rainy) = 0.3
  From Rainy: P(Rainy→Sunny) = 0.4, P(Rainy→Rainy) = 0.6

Emission Probabilities (B):
  In Sunny: P(Walk|Sunny) = 0.6, P(Shop|Sunny) = 0.3, P(Clean|Sunny) = 0.1
  In Rainy: P(Walk|Rainy) = 0.1, P(Shop|Rainy) = 0.2, P(Clean|Rainy) = 0.7
```

### The Three Types of Problems

HMMs are used to solve three fundamental problems:

```
1. EVALUATION (Forward Algorithm)
   Q: What is the probability of an observation sequence?
   Input: Observations [Walk, Shop, Clean]
   Output: P(observations | model)
   Use: Model comparison, anomaly detection

2. DECODING (Viterbi Algorithm)
   Q: What is the most likely sequence of hidden states?
   Input: Observations [Walk, Shop, Clean]
   Output: Most likely states [Sunny, Sunny, Rainy]
   Use: State estimation, classification

3. LEARNING (Baum-Welch Algorithm)
   Q: What are the model parameters?
   Input: Training sequences
   Output: Optimal π, A, B parameters
   Use: Model training from data
```

---

## The Mathematical Foundation

### 1. Model Parameters

An HMM is fully specified by **λ = (π, A, B)**:

**Initial State Distribution (π)**:
```
π = [π₁, π₂, ..., πₙ]

where πᵢ = P(S₀ = i)
      πᵢ ≥ 0
      Σᵢ πᵢ = 1
```

**State Transition Matrix (A)**:
```
A = [aᵢⱼ]  where aᵢⱼ = P(Sₜ = j | Sₜ₋₁ = i)

Properties:
  - aᵢⱼ ≥ 0
  - Σⱼ aᵢⱼ = 1  (each row sums to 1)
```

**Emission Probability Matrix (B)**:
```
B = [bᵢₖ]  where bᵢₖ = P(Oₜ = k | Sₜ = i)

Properties:
  - bᵢₖ ≥ 0
  - Σₖ bᵢₖ = 1  (each row sums to 1)
```

### 2. Problem 1: Evaluation (Forward Algorithm)

**Goal**: Calculate P(O | λ) - probability of observation sequence

**Forward Variable**:
```
αₜ(i) = P(O₁, O₂, ..., Oₜ, Sₜ = i | λ)

"Probability of:
 - Seeing observations O₁ through Oₜ
 - AND being in state i at time t"
```

**Algorithm**:
```
Initialization (t=0):
  α₀(i) = πᵢ · bᵢ(O₀)

Recursion (t=1 to T-1):
  αₜ(j) = [Σᵢ αₜ₋₁(i) · aᵢⱼ] · bⱼ(Oₜ)

Termination:
  P(O | λ) = Σᵢ αₜ₋₁(i)
```

**Intuition**:
```
Forward algorithm builds up the probability by:
1. Starting with initial state probabilities
2. At each step, summing over all ways to reach the next state
3. Multiplying by the emission probability of the observation
4. Final sum gives total probability
```

**Example Calculation**:
```
States: S₀=Sunny, S₁=Rainy
Observations: [Walk, Clean]

Step 1 (t=0, Obs=Walk):
  α₀(Sunny) = π(Sunny) · B(Sunny→Walk) = 0.6 · 0.6 = 0.36
  α₀(Rainy) = π(Rainy) · B(Rainy→Walk) = 0.4 · 0.1 = 0.04

Step 2 (t=1, Obs=Clean):
  α₁(Sunny) = [α₀(Sunny)·A(Sunny→Sunny) + α₀(Rainy)·A(Rainy→Sunny)] · B(Sunny→Clean)
            = [0.36·0.7 + 0.04·0.4] · 0.1
            = [0.252 + 0.016] · 0.1 = 0.0268
  
  α₁(Rainy) = [α₀(Sunny)·A(Sunny→Rainy) + α₀(Rainy)·A(Rainy→Rainy)] · B(Rainy→Clean)
            = [0.36·0.3 + 0.04·0.6] · 0.7
            = [0.108 + 0.024] · 0.7 = 0.0924

Result:
  P([Walk, Clean] | λ) = α₁(Sunny) + α₁(Rainy)
                       = 0.0268 + 0.0924 = 0.1192
```

### 3. Problem 2: Decoding (Viterbi Algorithm)

**Goal**: Find most likely state sequence S* = argmax P(S | O, λ)

**Viterbi Variable**:
```
δₜ(i) = max P(S₁, S₂, ..., Sₜ₋₁, Sₜ=i, O₁, ..., Oₜ | λ)
        S₁...Sₜ₋₁

"Maximum probability of any state sequence ending in state i at time t"
```

**Algorithm**:
```
Initialization (t=0):
  δ₀(i) = πᵢ · bᵢ(O₀)
  ψ₀(i) = 0

Recursion (t=1 to T-1):
  δₜ(j) = max[δₜ₋₁(i) · aᵢⱼ] · bⱼ(Oₜ)
          i
  ψₜ(j) = argmax[δₜ₋₁(i) · aᵢⱼ]
          i

Termination:
  P* = max[δₜ₋₁(i)]
       i
  S*ₜ₋₁ = argmax[δₜ₋₁(i)]
          i

Backtracking (t=T-2 to 0):
  S*ₜ = ψₜ₊₁(S*ₜ₊₁)
```

**Difference from Forward Algorithm**:
```
Forward: SUMS over all possible paths
  αₜ(j) = Σᵢ [αₜ₋₁(i) · aᵢⱼ] · bⱼ(Oₜ)
  → Total probability

Viterbi: Takes MAX over all possible paths
  δₜ(j) = maxᵢ [δₜ₋₁(i) · aᵢⱼ] · bⱼ(Oₜ)
  → Best path probability
```

**Example**:
```
Observations: [Walk, Shop, Clean]

Finding best path:
┌────────────────────────────────────────┐
│ t=0: Walk                              │
│   δ₀(Sunny) = 0.6 · 0.6 = 0.36  ← Best│
│   δ₀(Rainy) = 0.4 · 0.1 = 0.04        │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ t=1: Shop                              │
│   Best to Sunny:                       │
│     From Sunny: 0.36·0.7·0.3 = 0.0756 ← Best
│     From Rainy: 0.04·0.4·0.3 = 0.0048 │
│   Best to Rainy:                       │
│     From Sunny: 0.36·0.3·0.2 = 0.0216 │
│     From Rainy: 0.04·0.6·0.2 = 0.0048 │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ t=2: Clean                             │
│   Best to Sunny:                       │
│     From Sunny: 0.0756·0.7·0.1 = 0.00529│
│   Best to Rainy:                       │
│     From Sunny: 0.0756·0.3·0.7 = 0.0159 ← Best
└────────────────────────────────────────┘

Best path: [Sunny, Sunny, Rainy]
Probability: 0.0159
```

### 4. Problem 3: Learning (Baum-Welch Algorithm)

**Goal**: Learn parameters λ = (π, A, B) from observation sequences

**Algorithm**: Expectation-Maximization (EM)

**Backward Variable** (needed for learning):
```
βₜ(i) = P(Oₜ₊₁, Oₜ₊₂, ..., Oₜ | Sₜ = i, λ)

"Probability of seeing remaining observations
 given that we're in state i at time t"

Recursion (backward):
  βₜ(i) = Σⱼ aᵢⱼ · bⱼ(Oₜ₊₁) · βₜ₊₁(j)
```

**State Occupation Probability**:
```
γₜ(i) = P(Sₜ = i | O, λ)
      = αₜ(i) · βₜ(i) / P(O | λ)

"Probability of being in state i at time t
 given the full observation sequence"
```

**State Transition Probability**:
```
ξₜ(i,j) = P(Sₜ = i, Sₜ₊₁ = j | O, λ)
        = αₜ(i) · aᵢⱼ · bⱼ(Oₜ₊₁) · βₜ₊₁(j) / P(O | λ)

"Probability of being in state i at time t
 and state j at time t+1"
```

**Parameter Updates**:
```
π̂ᵢ = γ₀(i)
"Expected frequency in state i at time 0"

âᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i)
"Expected transitions from i to j / Expected time in state i"

b̂ᵢₖ = Σₜ (Oₜ=k) γₜ(i) / Σₜ γₜ(i)
"Expected time in state i observing k / Expected time in state i"
```

**Algorithm Steps**:
```
1. Initialize π, A, B randomly
2. Repeat until convergence:
   a) E-step: Calculate γₜ(i) and ξₜ(i,j) using Forward-Backward
   b) M-step: Update π, A, B using formulas above
   c) Check log-likelihood improvement
3. Return learned parameters
```

---

## The Three Fundamental Problems

### Problem 1: Evaluation

**Question**: Given a model and observation sequence, what is P(O|λ)?

**Algorithm**: Forward Algorithm (or Backward)

**Use Cases**:
- Model comparison: Which model better explains the data?
- Anomaly detection: Is this sequence unusual?
- Speech recognition: Which word model matches best?

**Example**:
```python
hmm1 = HiddenMarkovModel()  # Model for "hello"
hmm2 = HiddenMarkovModel()  # Model for "world"

acoustic_signal = [...]

score1 = hmm1.score(acoustic_signal)
score2 = hmm2.score(acoustic_signal)

if score1 > score2:
    recognized_word = "hello"
else:
    recognized_word = "world"
```

### Problem 2: Decoding

**Question**: Given observations, what is the most likely state sequence?

**Algorithm**: Viterbi Algorithm

**Use Cases**:
- POS tagging: What are the parts of speech?
- Weather prediction: What was the actual weather?
- Gene finding: Where are the genes?
- Market regimes: What regime is the market in?

**Example**:
```python
observations = ['Walk', 'Shop', 'Clean', 'Clean']
states = hmm.predict(observations)
# states = ['Sunny', 'Sunny', 'Rainy', 'Rainy']
```

### Problem 3: Learning

**Question**: Given observation sequences, what are the best parameters?

**Algorithm**: Baum-Welch Algorithm (EM)

**Use Cases**:
- Training from unlabeled data
- Discovering hidden patterns
- Parameter estimation

**Example**:
```python
training_data = [
    ['Walk', 'Walk', 'Shop'],
    ['Clean', 'Clean', 'Walk'],
    ['Shop', 'Walk', 'Clean']
]

hmm = HiddenMarkovModel(n_states=2)
hmm.fit(training_data, n_iter=100)
```

---

## Implementation Details

### Class Structure

```python
class HiddenMarkovModel:
    def __init__(self, n_states=None, n_observations=None):
        self.n_states = n_states
        self.n_observations = n_observations
        self.initial_prob = None      # π
        self.transition_prob = None   # A
        self.emission_prob = None     # B
```

### Core Methods

1. **`set_parameters(initial_prob, transition_prob, emission_prob)`**
   - Manually set model parameters (supervised learning)
   - Use when you know the parameters from domain knowledge

2. **`forward(observations)`**
   - Forward Algorithm implementation
   - Returns: forward probabilities α and log P(O|λ)

3. **`backward(observations)`**
   - Backward Algorithm implementation
   - Returns: backward probabilities β

4. **`viterbi(observations)`**
   - Viterbi Algorithm implementation
   - Returns: most likely state sequence and its probability

5. **`fit(observation_sequences, n_iter, tolerance)`**
   - Baum-Welch Algorithm (EM) for learning
   - Learns parameters from training data

6. **`predict(observations)`**
   - Wrapper for Viterbi (decoding)
   - Returns: predicted state sequence

7. **`score(observations)`**
   - Wrapper for Forward (evaluation)
   - Returns: log probability of observations

8. **`sample(n_samples)`**
   - Generate random sequences from the model
   - Returns: observations and hidden states

---

## Step-by-Step Example

Let's walk through a complete example: **Weather Prediction from Activities**

### The Scenario

You're in a room without windows. You observe your roommate's activities and want to infer the weather outside.

### Setup

```python
from hmm import HiddenMarkovModel
import numpy as np

# Create HMM
hmm = HiddenMarkovModel()

# Define parameters based on domain knowledge
initial_prob = [0.6, 0.4]  # 60% sunny, 40% rainy to start

transition_prob = [
    [0.7, 0.3],  # From Sunny: 70% stay sunny, 30% → rainy
    [0.4, 0.6]   # From Rainy: 40% → sunny, 60% stay rainy
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
```

### Understanding the Model

```
Model Structure:
┌─────────────────────────────────────────┐
│ Initial State Probabilities:            │
│   P(Sunny at start) = 0.6               │
│   P(Rainy at start) = 0.4               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ State Transitions:                      │
│                                         │
│     ┌──── 0.7 ────┐                    │
│     ↓             │                     │
│  [Sunny] ──0.3──→ [Rainy]              │
│     ↑             │                     │
│     └──── 0.4 ──←─┘ 0.6                │
│                   │                     │
│                   ↓                     │
│                                         │
│ Sunny tends to stay sunny (70%)        │
│ Rainy tends to stay rainy (60%)        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Emissions (Observations):               │
│                                         │
│ Sunny → Walk (60%), Shop (30%), Clean (10%)
│ Rainy → Walk (10%), Shop (20%), Clean (70%)
│                                         │
│ Walking is common when sunny            │
│ Cleaning is common when rainy           │
└─────────────────────────────────────────┘
```

### Problem 1: Evaluation

**Question**: What is the probability of observing [Walk, Shop, Clean]?

```python
observations = ['Walk', 'Shop', 'Clean']
log_prob = hmm.score(observations)
prob = np.exp(log_prob)

print(f"P([Walk, Shop, Clean]) = {prob:.6f}")
print(f"Log probability = {log_prob:.4f}")
```

**Manual Calculation**:
```
All possible state sequences for 3 observations:
1. [Sunny, Sunny, Sunny]
2. [Sunny, Sunny, Rainy]
3. [Sunny, Rainy, Sunny]
4. [Sunny, Rainy, Rainy]
5. [Rainy, Sunny, Sunny]
6. [Rainy, Sunny, Rainy]
7. [Rainy, Rainy, Sunny]
8. [Rainy, Rainy, Rainy]

For each path, calculate: P(states) × P(observations|states)

Example - Path [Sunny, Sunny, Rainy]:
  P(states) = π(Sunny) × A(S→S) × A(S→R)
            = 0.6 × 0.7 × 0.3 = 0.126
  
  P(obs|states) = B(S→Walk) × B(S→Shop) × B(R→Clean)
                = 0.6 × 0.3 × 0.7 = 0.126
  
  P(path) = 0.126 × 0.126 = 0.015876

Sum over all 8 paths to get total probability
(Forward algorithm does this efficiently!)
```

**Output**:
```
P([Walk, Shop, Clean]) = 0.033194
Log probability = -3.4048

Interpretation: 3.3% chance of this sequence
```

### Problem 2: Decoding

**Question**: Given observations, what is the most likely weather sequence?

```python
observations = ['Walk', 'Shop', 'Clean', 'Clean', 'Walk']
predicted_weather = hmm.predict(observations)

print("Observed Activities:", observations)
print("Predicted Weather:  ", predicted_weather)
```

**Step-by-Step Viterbi**:
```
t=0: Walk
  δ₀(Sunny) = 0.6 × 0.6 = 0.36  ← BEST
  δ₀(Rainy) = 0.4 × 0.1 = 0.04
  Best: Sunny

t=1: Shop
  To Sunny: max(0.36×0.7, 0.04×0.4) × 0.3 = 0.0756  ← BEST
  To Rainy: max(0.36×0.3, 0.04×0.6) × 0.2 = 0.0216
  Best: Sunny (from Sunny)

t=2: Clean
  To Sunny: max(0.0756×0.7, 0.0216×0.4) × 0.1 = 0.00529
  To Rainy: max(0.0756×0.3, 0.0216×0.6) × 0.7 = 0.0159  ← BEST
  Best: Rainy (from Sunny)

t=3: Clean
  To Sunny: max(0.00529×0.7, 0.0159×0.4) × 0.1 = 0.000637
  To Rainy: max(0.00529×0.3, 0.0159×0.6) × 0.7 = 0.00668  ← BEST
  Best: Rainy (from Rainy)

t=4: Walk
  To Sunny: max(0.000637×0.7, 0.00668×0.4) × 0.6 = 0.00160  ← BEST
  To Rainy: max(0.000637×0.3, 0.00668×0.6) × 0.1 = 0.000401
  Best: Sunny (from Rainy)

Backtrack:
  t=4: Sunny ← t=3: Rainy ← t=2: Rainy ← t=1: Sunny ← t=0: Sunny
```

**Output**:
```
Observed Activities: ['Walk', 'Shop', 'Clean', 'Clean', 'Walk']
Predicted Weather:   ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny']

Interpretation:
- Started sunny (Walk, Shop are sunny activities)
- Became rainy (two Clean observations)
- Back to sunny (Walk observation)
```

### Problem 3: Learning

**Question**: Learn model parameters from data

```python
# Training data: observation sequences (no state labels!)
training_data = [
    ['Walk', 'Walk', 'Shop'],
    ['Clean', 'Clean', 'Walk'],
    ['Shop', 'Walk', 'Clean'],
    ['Walk', 'Shop', 'Walk'],
    ['Clean', 'Clean', 'Clean']
]

# Create HMM with 2 hidden states
hmm_learned = HiddenMarkovModel(n_states=2)

# Learn parameters using Baum-Welch
hmm_learned.fit(training_data, n_iter=100, verbose=True)

# Print learned parameters
hmm_learned.print_parameters()
```

**What Baum-Welch Does**:
```
Iteration 1:
  Random initialization
  Calculate expected state occupancies
  Update parameters
  Log-likelihood: -8.5234

Iteration 2:
  Use new parameters
  Recalculate expectations
  Update again
  Log-likelihood: -7.1456  (improved!)

...continues until convergence...

Iteration 47:
  Log-likelihood: -5.2103
  Change < tolerance → CONVERGED
```

**Output**:
```
Learned Initial Probabilities:
  State 0: 0.6234
  State 1: 0.3766

Learned Transition Probabilities:
  From State 0: [0.6891, 0.3109]
  From State 1: [0.4123, 0.5877]

Learned Emission Probabilities:
  State 0: Walk=0.5821, Shop=0.2912, Clean=0.1267
  State 1: Walk=0.0923, Shop=0.1845, Clean=0.7232

Interpretation:
  State 0 ≈ Sunny (high Walk/Shop, low Clean)
  State 1 ≈ Rainy (high Clean, low Walk)
```

---

## Real-World Applications

### 1. **Speech Recognition**

The original killer app for HMMs!

**Problem**: Convert audio to text

**Setup**:
- Hidden States: Phonemes (basic speech sounds)
- Observations: Acoustic features (MFCCs, spectrograms)
- Goal: Decode phoneme sequence from audio

**How It Works**:
```
Audio Signal
    ↓ (extract features)
Acoustic Features: [f1, f2, f3, ..., fn]
    ↓ (HMM decoding)
Phoneme Sequence: [/h/, /ə/, /l/, /oʊ/]
    ↓ (language model)
Word: "hello"
```

**Why HMM**:
- Speech is sequential
- Phonemes (hidden) produce acoustic signals (observed)
- Pronunciation varies (HMM handles uncertainty)

**Modern Note**: Deep learning has largely replaced HMMs for speech recognition, but HMMs laid the foundation!

### 2. **Part-of-Speech (POS) Tagging**

**Problem**: Label each word in a sentence with its grammatical role

**Setup**:
- Hidden States: POS tags (Noun, Verb, Adjective, etc.)
- Observations: Words in the sentence
- Goal: Find most likely POS tag sequence

**Example**:
```
Sentence: "The quick brown fox jumps"

Hidden (POS):  [DET] [ADJ] [ADJ] [NOUN] [VERB]
Observed:       The  quick brown  fox   jumps

Training: Learn P(word|POS) and P(POS_next|POS_current)
Testing: Given new sentence, predict POS tags
```

**Why HMM**:
```
- Sequential: POS depends on previous POS
- Hidden: We don't see POS directly, only words
- Ambiguous: "bank" can be noun or verb
  HMM uses context to decide
```

**Application**:
- Text analysis
- Information extraction
- Machine translation
- Grammar checking

### 3. **Bioinformatics: Gene Finding**

**Problem**: Identify genes in DNA sequences

**Setup**:
- Hidden States: Gene regions (Coding, Non-coding, Intron, Exon)
- Observations: DNA nucleotides (A, T, G, C)
- Goal: Segment DNA into functional regions

**Example**:
```
DNA:    A T G C C A T A T G A C G T A A
States: [Exon----] [Intron] [Exon----]
        (coding)   (non-cod) (coding)
```

**Gene Structure**:
```
┌────────┬─────────┬────────┬──────────┐
│ Promoter│  Exon  │ Intron │  Exon   │
│  (NC)   │  (C)   │  (NC)  │  (C)    │
└────────┴─────────┴────────┴──────────┘

NC = Non-coding (not expressed)
C = Coding (expressed as protein)
```

**Why HMM**:
- Different regions have different nucleotide patterns
- Exons: more structured, specific codon usage
- Introns: more random, different statistics
- HMM learns these patterns

**Other Bioinformatics Uses**:
- Protein structure prediction
- Sequence alignment
- Motif discovery
- RNA structure prediction

### 4. **Stock Market Regime Detection**

**Problem**: Identify market states (Bull, Bear, Sideways)

**Setup**:
- Hidden States: Market regimes
- Observations: Returns, volatility, volume
- Goal: Detect regime changes

**Example**:
```
Time:    Jan  Feb  Mar  Apr  May  Jun  Jul  Aug
Returns: +5%  +3%  +2%  -1%  -3%  -2%  +1%  +2%
Regime:  [Bull----]  [Bear----]  [Sideways-]
```

**Market Regimes**:
```
Bull Market:
  - Positive returns
  - Lower volatility
  - High volume
  - Optimistic sentiment

Bear Market:
  - Negative returns
  - Higher volatility
  - High volume
  - Pessimistic sentiment

Sideways Market:
  - Mixed returns
  - Moderate volatility
  - Lower volume
  - Uncertain sentiment
```

**Trading Application**:
```python
observations = get_market_data(last_30_days)
regime = hmm.predict(observations)[-1]

if regime == 'Bull':
    strategy = 'aggressive_long'
elif regime == 'Bear':
    strategy = 'defensive_short'
else:  # Sideways
    strategy = 'range_trading'
```

**Benefits**:
- Early detection of regime changes
- Adaptive trading strategies
- Risk management
- Portfolio allocation

### 5. **Natural Language Processing: Text Generation**

**Problem**: Generate realistic text sequences

**Setup**:
- Hidden States: Topics or latent semantics
- Observations: Words
- Goal: Generate coherent text

**Example**:
```
Hidden Topic:    [Sports]  [Sports] [Weather] [Weather]
Generated Words:  game     score    sunny     warm

Transitions: Sports → Weather (topic shift)
Emissions: In "Sports" topic, likely words are {game, score, team, ...}
```

**Application**:
- Chatbots
- Text completion
- Story generation
- Dialogue systems

### 6. **Gesture Recognition**

**Problem**: Recognize hand gestures from sensor data

**Setup**:
- Hidden States: Gesture phases
- Observations: Hand positions, accelerometer data
- Goal: Classify gestures

**Example - "Swipe Right" Gesture**:
```
States: [Start] → [Moving Right] → [End]
Sensor:  (x=0)     (x=1,2,3,4,5)    (x=5)
```

**Use Cases**:
- Smartphone gesture controls
- Sign language recognition
- Virtual reality interfaces
- Gaming controls

### 7. **Activity Recognition**

**Problem**: Recognize human activities from smartphone sensors

**Setup**:
- Hidden States: Activities (Walking, Running, Sitting, Standing)
- Observations: Accelerometer, gyroscope readings
- Goal: Classify current activity

**Example**:
```
Time:    0s   1s   2s   3s   4s   5s
Accel:  High High High Low  Low  Low
State:  [Walking-----] [Sitting-----]
```

**Applications**:
- Fitness tracking
- Health monitoring
- Elderly care (fall detection)
- Context-aware apps

---

## Understanding the Code

### 1. Forward Algorithm Implementation

```python
def forward(self, observations):
    T = len(observations)
    alpha = np.zeros((T, self.n_states))
    
    # Initialization
    alpha[0] = self.initial_prob * self.emission_prob[:, observations[0]]
    
    # Recursion
    for t in range(1, T):
        for j in range(self.n_states):
            alpha[t, j] = np.sum(alpha[t-1] * self.transition_prob[:, j]) * \
                         self.emission_prob[j, observations[t]]
    
    log_prob = np.log(np.sum(alpha[T-1]) + 1e-10)
    return alpha, log_prob
```

**How It Works**:
```python
# Example: 2 states, observations = [0, 1]

# Step 1: Initialization (t=0)
alpha[0, 0] = π[0] × B[0, obs[0]]  # State 0, obs 0
alpha[0, 1] = π[1] × B[1, obs[0]]  # State 1, obs 0

# Step 2: Recursion (t=1)
alpha[1, 0] = (alpha[0,0]×A[0,0] + alpha[0,1]×A[1,0]) × B[0, obs[1]]
            = (prob via state 0 + prob via state 1) × emission prob

alpha[1, 1] = (alpha[0,0]×A[0,1] + alpha[0,1]×A[1,1]) × B[1, obs[1]]

# Step 3: Sum for total probability
P(O|λ) = alpha[1, 0] + alpha[1, 1]
```

**Computational Complexity**:
```
Naive: O(N^T × T) where N=states, T=time steps
  - Enumerate all N^T possible state sequences
  - Exponential! Infeasible for T>10

Forward: O(N^2 × T)
  - For each time step: O(T)
  - For each state: O(N)
  - For each previous state: O(N)
  - Total: O(N^2 × T)
  - Polynomial! Much better!

Example:
  N=10 states, T=100 time steps
  Naive: 10^100 operations (impossible!)
  Forward: 10,000 operations (instant!)
```

### 2. Viterbi Algorithm Implementation

```python
def viterbi(self, observations):
    T = len(observations)
    delta = np.zeros((T, self.n_states))
    psi = np.zeros((T, self.n_states), dtype=int)
    
    # Initialization
    delta[0] = self.initial_prob * self.emission_prob[:, observations[0]]
    
    # Recursion
    for t in range(1, T):
        for j in range(self.n_states):
            prob_scores = delta[t-1] * self.transition_prob[:, j]
            psi[t, j] = np.argmax(prob_scores)  # Best previous state
            delta[t, j] = np.max(prob_scores) * self.emission_prob[j, observations[t]]
    
    # Termination
    path_indices = np.zeros(T, dtype=int)
    path_indices[T-1] = np.argmax(delta[T-1])
    
    # Backtracking
    for t in range(T-2, -1, -1):
        path_indices[t] = psi[t+1, path_indices[t+1]]
    
    return path_indices, np.max(delta[T-1])
```

**Visualization**:
```
Trellis Diagram:

t=0        t=1        t=2
State 0:   •--------->•--------->•
           |  \    /  |  \    /
           |   \  /   |   \  /
           |    \/    |    \/
           |    /\    |    /\
           |   /  \   |   /  \
State 1:   •--------->•--------->•

At each node, keep track of:
- δ: Best path probability to this node
- ψ: Best previous state

Final: Backtrack from best final state
```

**Key Difference**:
```python
# Forward: SUM over paths
alpha[t, j] = sum(alpha[t-1, i] * A[i,j] for i in states) * B[j, obs[t]]

# Viterbi: MAX over paths
delta[t, j] = max(delta[t-1, i] * A[i,j] for i in states) * B[j, obs[t]]
              ^
              |
            Only difference!
```

### 3. Baum-Welch Algorithm Implementation

```python
def fit(self, observations_sequences, n_iter=100, tolerance=1e-4):
    # Initialize parameters randomly
    self._initialize_parameters(observations_sequences)
    
    prev_log_likelihood = float('-inf')
    
    for iteration in range(n_iter):
        # Accumulators for new parameters
        new_initial = np.zeros(self.n_states)
        new_transition = np.zeros((self.n_states, self.n_states))
        new_emission = np.zeros((self.n_states, self.n_observations))
        
        total_log_likelihood = 0
        
        for obs_seq in observations_sequences:
            # E-step: Forward-Backward
            alpha, log_prob = self.forward(obs_seq)
            beta = self.backward(obs_seq)
            
            total_log_likelihood += log_prob
            
            # Calculate γ (state occupation)
            gamma = alpha * beta
            gamma = gamma / gamma.sum(axis=1, keepdims=True)
            
            # Calculate ξ (state transition)
            xi = self._calculate_xi(alpha, beta, obs_seq)
            
            # M-step: Accumulate statistics
            new_initial += gamma[0]
            new_transition += xi.sum(axis=0)
            new_emission += self._accumulate_emissions(gamma, obs_seq)
        
        # Normalize
        self.initial_prob = new_initial / new_initial.sum()
        self.transition_prob = new_transition / new_transition.sum(axis=1, keepdims=True)
        self.emission_prob = new_emission / new_emission.sum(axis=1, keepdims=True)
        
        # Check convergence
        if abs(total_log_likelihood - prev_log_likelihood) < tolerance:
            break
        
        prev_log_likelihood = total_log_likelihood
```

**Intuition**:
```
E-step (Expectation):
  "Given current parameters, what are the expected state occupancies?"
  
  Calculate:
  - γ(i,t): Probability of being in state i at time t
  - ξ(i,j,t): Probability of transitioning i→j at time t

M-step (Maximization):
  "Given expected occupancies, what are the best parameters?"
  
  Update:
  - π[i] = expected frequency in state i at t=0
  - A[i,j] = expected transitions i→j / expected time in i
  - B[i,k] = expected emissions k in i / expected time in i

Repeat until parameters converge (likelihood stops increasing)
```

**Why It Works**:
```
EM Algorithm guarantees:
- Likelihood never decreases
- Converges to local maximum
- Finds parameters that best explain the data

Note: May not find global maximum (depends on initialization)
Solution: Run multiple times with different initializations
```

---

## Model Evaluation

### 1. Choosing the Number of States

**Too Few States**:
```
Problem: Underfitting
- Model too simple
- Cannot capture complexity
- Poor predictions

Example: 1 state for weather
  Cannot distinguish sunny vs rainy!
```

**Too Many States**:
```
Problem: Overfitting
- Model too complex
- Memorizes training data
- Poor generalization

Example: 100 states for weather
  Overfits to training sequences
```

**Methods for Selection**:

**a) Cross-Validation**:
```python
from sklearn.model_selection import KFold

n_states_options = [2, 3, 4, 5]
cv_scores = []

for n_states in n_states_options:
    kf = KFold(n_splits=5)
    scores = []
    
    for train_idx, val_idx in kf.split(sequences):
        train = [sequences[i] for i in train_idx]
        val = [sequences[i] for i in val_idx]
        
        hmm = HiddenMarkovModel(n_states=n_states)
        hmm.fit(train)
        
        # Evaluate on validation set
        val_score = sum(hmm.score(seq) for seq in val)
        scores.append(val_score)
    
    cv_scores.append(np.mean(scores))

best_n_states = n_states_options[np.argmax(cv_scores)]
```

**b) Bayesian Information Criterion (BIC)**:
```
BIC = -2 × log(L) + k × log(n)

where:
  L = likelihood
  k = number of parameters
  n = number of observations

Lower BIC = better model

Number of parameters:
  π: N - 1 (sum to 1)
  A: N × (N-1) (each row sums to 1)
  B: N × (M-1) (each row sums to 1)
  Total: N-1 + N(N-1) + N(M-1) = N^2 + NM - 1
```

**c) Domain Knowledge**:
```
Best approach: Use domain expertise!

Examples:
- Weather: 2-3 states (Sunny, Rainy, [Cloudy])
- POS tagging: 12-45 states (number of POS tags)
- Market regimes: 3-4 states (Bull, Bear, Sideways, [Volatile])
```

### 2. Evaluation Metrics

**For Supervised Learning** (known states):

**a) Accuracy**:
```python
predicted = hmm.predict(observations)
actual = true_states

accuracy = np.mean(predicted == actual)
```

**b) Confusion Matrix**:
```
              Predicted
              Sunny Rainy
Actual Sunny    45     5
       Rainy     3    47

Accuracy = (45+47)/(45+5+3+47) = 0.92
```

**c) F1-Score per State**:
```python
from sklearn.metrics import f1_score

f1_sunny = f1_score(actual == 'Sunny', predicted == 'Sunny')
f1_rainy = f1_score(actual == 'Rainy', predicted == 'Rainy')
```

**For Unsupervised Learning** (unknown states):

**a) Log-Likelihood**:
```python
# Higher log-likelihood = better model
log_likelihood = hmm.score(test_sequence)
```

**b) Perplexity**:
```
Perplexity = exp(-log(P(O|λ)) / T)

Lower perplexity = better model
```

**c) Qualitative Evaluation**:
```
- Do learned states make sense?
- Do state transitions match expectations?
- Do emissions align with domain knowledge?
```

### 3. Common Pitfalls

**a) Underflow in Probabilities**:
```
Problem: Multiplying many small probabilities → 0

Bad:
  prob = p1 * p2 * p3 * ... * p100
  # prob becomes 0 due to floating point underflow

Good:
  log_prob = log(p1) + log(p2) + ... + log(p100)
  # Use log space, more stable
```

**b) Local Maxima in EM**:
```
Problem: Baum-Welch finds local, not global maximum

Solution:
- Run multiple times with different initializations
- Use k-means to initialize emission probabilities
- Use domain knowledge for initialization
```

**c) Zero Probabilities**:
```
Problem: Unseen transitions/emissions have probability 0

Bad:
  P(state_i → state_j) = 0  # Never seen in training
  # Causes problems for new sequences

Good: Smoothing
  P(state_i → state_j) = (count + ε) / (total + ε×N)
  # Add small constant ε (e.g., 0.01)
```

**d) Choosing Wrong Number of States**:
```
Too few: Underfitting
Too many: Overfitting

Solution: Cross-validation, BIC, domain knowledge
```

### 4. Model Interpretation

**Examine Learned Parameters**:

```python
hmm.print_parameters()

# Check if learned states match expectations:

# Initial probabilities
# - Do states have reasonable starting probabilities?

# Transition matrix
# - Are state durations reasonable?
# - Self-transition prob close to 1 → state persists
# - Self-transition prob close to 0 → state changes often

# Emission matrix
# - Do states have distinct emission patterns?
# - Can you assign meaningful labels to states?
```

**Example Analysis**:
```
Learned Transition Matrix:
          S0    S1
    S0  [0.9  0.1]
    S1  [0.2  0.8]

Interpretation:
- S0 is very stable (90% self-transition)
- S1 is stable (80% self-transition)
- S0 → S1 less common than S1 → S0
- Possible: S0=Normal, S1=Abnormal state

Learned Emission Matrix:
      Obs0  Obs1  Obs2
S0   [0.7   0.2   0.1]
S1   [0.1   0.2   0.7]

Interpretation:
- S0 strongly associated with Obs0
- S1 strongly associated with Obs2
- Obs1 is neutral (similar in both states)
```

---

## Advantages and Limitations

### Advantages ✅

1. **Handles Uncertainty**
   - Models probabilistic relationships
   - Accounts for noise in observations
   - Provides confidence measures

2. **Sequences & Temporal Data**
   - Natural for sequential problems
   - Captures temporal dependencies
   - Learns transition dynamics

3. **Unsupervised Learning**
   - Can learn from unlabeled data
   - Discovers hidden patterns
   - No need for state annotations

4. **Mathematically Rigorous**
   - Well-founded probability theory
   - Efficient algorithms (Dynamic Programming)
   - Convergence guarantees (Baum-Welch)

5. **Interpretable**
   - Parameters have clear meanings
   - States can be understood
   - Transitions are explainable

6. **Multiple Inference Tasks**
   - Evaluation: P(observations)
   - Decoding: Most likely states
   - Learning: Find parameters
   - Prediction: Future observations

### Limitations ❌

1. **Markov Assumption**
   ```
   Assumption: P(Sₜ | S₀...Sₜ₋₁) = P(Sₜ | Sₜ₋₁)
   
   Problem: Future depends only on immediate past
   
   Real world: May need longer history
   Example: In language, "bank" depends on sentence context,
            not just previous word
   
   Solution: Higher-order HMMs (but more parameters)
   ```

2. **Output Independence**
   ```
   Assumption: P(Oₜ | O₀...Oₜ₋₁, S₀...Sₜ) = P(Oₜ | Sₜ)
   
   Problem: Observations may be correlated
   
   Example: In speech, acoustic features are correlated
   
   Solution: Use richer observation models
   ```

3. **Local Maxima**
   ```
   Baum-Welch (EM) finds local, not global maximum
   
   Problem: Results depend on initialization
   
   Solution:
   - Run multiple times
   - Use informed initialization
   - Try different numbers of states
   ```

4. **Fixed Number of States**
   ```
   Must specify N before training
   
   Problem: Wrong N → poor performance
   
   Solution:
   - Cross-validation
   - BIC model selection
   - Hierarchical/infinite HMMs
   ```

5. **Computational Cost**
   ```
   Training: O(N² × T × I) where:
     N = number of states
     T = sequence length
     I = number of iterations
   
   Problem: Slow for large N or long sequences
   
   Solutions:
   - Sparse transition matrices
   - Parallel processing
   - Approximate inference
   ```

6. **Discrete Observations**
   ```
   Standard HMM assumes discrete observations
   
   Problem: Real-valued features need discretization
   
   Solutions:
   - Gaussian HMM (continuous observations)
   - Vector quantization
   - Deep learning features
   ```

### When to Use HMMs

**Good Use Cases**:
- ✅ Sequential data with temporal dependencies
- ✅ Hidden process with observable outputs
- ✅ Moderate number of states (<20)
- ✅ Need probabilistic predictions
- ✅ Need interpretable model
- ✅ Limited training data

**Bad Use Cases**:
- ❌ Very long sequences (use RNNs/LSTMs)
- ❌ Complex dependencies (use Deep Learning)
- ❌ High-dimensional observations (use dimensionality reduction first)
- ❌ Non-sequential data (use other models)
- ❌ Need end-to-end differentiability (use neural networks)

---

## Comparing with Alternatives

### HMM vs. Conditional Random Fields (CRF)

```
HMM:
  Model: Generative (models P(O,S))
  ✓ Can generate samples
  ✓ Simpler
  ✗ Makes independence assumptions
  
CRF:
  Model: Discriminative (models P(S|O) directly)
  ✓ Fewer independence assumptions
  ✓ Can use rich features
  ✗ Cannot generate samples
  ✗ More complex training
```

### HMM vs. Recurrent Neural Networks (RNN/LSTM)

```
HMM:
  ✓ Works with small data
  ✓ Faster training
  ✓ Interpretable
  ✗ Limited expressiveness
  ✗ Manual feature engineering
  
RNN/LSTM:
  ✓ More expressive
  ✓ Learns features automatically
  ✗ Needs lots of data
  ✗ Slower training
  ✗ Less interpretable
```

### HMM vs. Naive Bayes

```
HMM:
  ✓ Sequential data
  ✓ Temporal dependencies
  ✗ More complex
  
Naive Bayes:
  ✓ Simpler
  ✓ Faster
  ✗ Assumes independence (no sequences)
```

---

## Key Concepts to Remember

### 1. **Three Fundamental Problems**
- **Evaluation**: What is P(observations | model)?
- **Decoding**: What are the most likely hidden states?
- **Learning**: What are the best parameters?

### 2. **Key Algorithms**
- **Forward**: Calculate P(observations) - O(N²T)
- **Viterbi**: Find best state sequence - O(N²T)
- **Baum-Welch**: Learn parameters - O(N²TI)

### 3. **Markov Property**
```
P(Sₜ | S₀...Sₜ₋₁) = P(Sₜ | Sₜ₋₁)

Future depends only on present, not past
```

### 4. **Model Parameters**
```
λ = (π, A, B)

π: Initial state probabilities
A: State transition probabilities
B: Emission probabilities
```

### 5. **Dynamic Programming**
```
HMM algorithms use DP to avoid exponential complexity

Instead of checking all N^T paths:
- Reuse calculations
- Build solutions incrementally
- Achieve O(N²T) complexity
```

---

## Conclusion

Hidden Markov Models are powerful tools for modeling sequential data with hidden structure! By understanding:
- The three components (π, A, B)
- The three problems (Evaluation, Decoding, Learning)
- The three algorithms (Forward, Viterbi, Baum-Welch)

You've gained a fundamental technique used across many domains! 🔮

**When to Use HMM**:
- ✅ Sequential/temporal data
- ✅ Hidden states with observable outputs
- ✅ Need probabilistic model
- ✅ Want interpretability
- ✅ Moderate complexity

**When to Use Something Else**:
- ❌ Very long sequences → RNN/LSTM
- ❌ Complex patterns → Deep Learning
- ❌ Non-sequential data → Other models
- ❌ Very large state space → Approximate methods

**Next Steps**:
- Try HMM on your sequential data
- Experiment with different numbers of states
- Learn about Gaussian HMMs (continuous observations)
- Study Conditional Random Fields (discriminative alternative)
- Explore modern deep learning sequence models (RNN, LSTM, Transformers)
- Read about hierarchical and infinite HMMs

Happy sequence modeling! 💻🔮📊
