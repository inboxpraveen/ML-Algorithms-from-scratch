# Hidden Markov Models (HMM) from Scratch: A Comprehensive Guide

Welcome to the fascinating world of Hidden Markov Models! 🔮 In this comprehensive guide, we'll explore HMMs - powerful statistical models for sequential data where the underlying process is hidden but observable through outputs. Think of it as understanding the "hidden story" behind what you can see!

## Table of Contents
0. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
1. [What are Hidden Markov Models?](#what-are-hidden-markov-models)
2. [How HMMs Work](#how-hmms-work)
3. [The Mathematical Foundation](#the-mathematical-foundation)
4. [The Three Fundamental Problems](#the-three-fundamental-problems)
5. [Implementation Details](#implementation-details)
6. [Step-by-Step Example](#step-by-step-example)
7. [Real-World Applications](#real-world-applications)
8. [Understanding the Code](#understanding-the-code)
9. [Model Evaluation](#model-evaluation)
10. [Simplifications vs. Canonical HMM Implementations](#simplifications-vs-canonical-hmm-implementations)
11. [Advantages and Limitations](#advantages-and-limitations)
12. [Comparing with Alternatives](#comparing-with-alternatives)
13. [Key Concepts to Remember](#key-concepts-to-remember)
14. [Conclusion](#conclusion)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra
dependencies beyond NumPy. (Running `python _23_hmm.py` executes the fuller version
of exactly this, including a Baum-Welch learning demo and an anomaly detector.)

```python
# ---------------------------------------------------------------
# Hidden Markov Model from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _23_hmm.py  (the __main__ block runs a fuller version)
# Or paste the HiddenMarkovModel class from _23_hmm.py above this line.
# ---------------------------------------------------------------
import numpy as np

# ---- Paste the HiddenMarkovModel class here (from _23_hmm.py) ----
# class HiddenMarkovModel: ...

np.random.seed(42)

# ---------- PART 1: parameters known - the three classic problems ----------
# Hidden states: the weather (Sunny / Rainy) - you cannot see it.
# Observations : what your roommate does (Walk / Shop / Clean) - you can.
weather = HiddenMarkovModel()
weather.set_parameters(
    initial_prob=[0.6, 0.4],
    transition_prob=[[0.7, 0.3],      # Sunny -> Sunny 0.7, Sunny -> Rainy 0.3
                     [0.4, 0.6]],     # Rainy -> Sunny 0.4, Rainy -> Rainy 0.6
    emission_prob=[[0.6, 0.3, 0.1],   # Sunny: mostly walking
                   [0.1, 0.2, 0.7]],  # Rainy: mostly cleaning
    state_labels=['Sunny', 'Rainy'],
    observation_labels=['Walk', 'Shop', 'Clean']
)

# Problem 1 - EVALUATION: how likely is this sequence? (forward algorithm)
obs3 = ['Walk', 'Shop', 'Clean']
log_p = weather.score(obs3)
print(f"log P(O) = {log_p:.4f}    P(O) = {np.exp(log_p):.6f}")

# Problem 2 - DECODING: which weather explains it best? (Viterbi algorithm)
obs5 = ['Walk', 'Shop', 'Clean', 'Clean', 'Walk']
path, path_prob = weather.viterbi(obs5)
print(f"Best states = {path}")
print(f"P(best path) = {path_prob:.8f}   P(O) all paths = {np.exp(weather.score(obs5)):.8f}")

# ---------- PART 2: parameters unknown - LEARNING with Baum-Welch ----------
# Hidden states: market regime (Bull / Bear). Observed: daily move.
market = HiddenMarkovModel()
market.set_parameters(
    initial_prob=[0.5, 0.5],
    transition_prob=[[0.90, 0.10],
                     [0.15, 0.85]],
    emission_prob=[[0.10, 0.20, 0.70],   # Bull: mostly Up days
                   [0.65, 0.25, 0.10]],  # Bear: mostly Down days
    state_labels=['Bull', 'Bear'],
    observation_labels=['Down', 'Flat', 'Up']
)

np.random.seed(42)
train = [market.sample(n_samples=20)[0] for _ in range(60)]   # [0] = observations only
test  = [market.sample(n_samples=20)[0] for _ in range(20)]

learned = HiddenMarkovModel(n_states=2)
learned.fit(train, n_iter=200, tolerance=1e-6, random_state=0)

def avg_ll(model, seqs):
    """Log-likelihood per observation, so lengths are comparable."""
    return sum(model.score(s) for s in seqs) / sum(len(s) for s in seqs)

print(f"\nTrain log-lik/obs (learned): {avg_ll(learned, train):.4f}")
print(f"Test  log-lik/obs (learned): {avg_ll(learned, test):.4f}")
print(f"Test  log-lik/obs (TRUE)   : {avg_ll(market, test):.4f}   <- the target")
print("Learned emission matrix B:")
print(np.round(learned.emission_prob, 3))
print("True    emission matrix B:")
print(np.round(market.emission_prob, 3))
```

Expected output:
```
log P(O) = -3.3623    P(O) = 0.034656
Best states = ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny']
P(best path) = 0.00160030   P(O) all paths = 0.00465961

Train log-lik/obs (learned): -0.9609
Test  log-lik/obs (learned): -1.0220
Test  log-lik/obs (TRUE)   : -1.0133   <- the target
Learned emission matrix B:
[[0.098 0.181 0.721]
 [0.649 0.268 0.083]]
True    emission matrix B:
[[0.1  0.2  0.7 ]
 [0.65 0.25 0.1 ]]
```

Read that last comparison carefully - it is the whole point of Baum-Welch. From
1200 observations and **no state labels at all**, the algorithm recovered the true
emission matrix to within about 0.02 per entry. Two caveats the numbers make
visible:

- **Row order is arbitrary.** Here row 0 happened to land on Bull, but a different
  `random_state` can swap them. Nothing in the likelihood distinguishes "state 0"
  from "state 1" - this is called *label switching*. Always identify a learned
  state by its emission row, never by its index.
- **EM finds a local optimum.** The learned model's test likelihood (-1.0220) sits
  just below the true model's (-1.0133) and well above a memoryless baseline
  (log(1/3) = -1.0986). Different seeds land in different optima; the honest
  procedure is to fit several and keep the one with the highest *training*
  likelihood.

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
| **Learning Style** | Unsupervised (Baum-Welch); parameters can also be supplied directly |
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
  P(O | λ) = Σᵢ α_{T-1}(i)     (sum the LAST column of the trellis)
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
βₜ(i) = P(Oₜ₊₁, Oₜ₊₂, ..., O_T | Sₜ = i, λ)

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

âᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i)     for t = 0 .. T-2
"Expected transitions from i to j / Expected time in state i"

  The range matters: there are only T-1 transitions in a sequence of length T,
  so both sums stop at T-2. The emission update below runs to T-1 instead,
  because every one of the T time steps emits a symbol.

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

1. **`set_parameters(initial_prob, transition_prob, emission_prob, state_labels, observation_labels)`**
   - Manually set model parameters, from domain knowledge or from counts you
     estimated yourself on labelled data
   - Accepts plain Python lists as well as numpy arrays
   - The labels are not decoration: they define the label -> index mapping that
     lets `predict(['Walk', 'Shop'])` work
   - Returns: `self`, so you can chain

2. **`forward(observations, scale=True)`**
   - Forward Algorithm implementation
   - Returns: forward probabilities α and log P(O|λ)
   - With `scale=True` (default) the α returned are the SCALED values (each row
     sums to 1) and the log-probability is exact for any sequence length. With
     `scale=False` you get the raw textbook α - useful for checking a short
     worked example by hand, but they underflow past roughly 30 steps

3. **`backward(observations, c=None)`**
   - Backward Algorithm implementation
   - Returns: backward probabilities β
   - Pass the scaling coefficients `c` from the forward pass to get the scaled
     β that the E-step needs; leave it `None` for the raw textbook β, which
     satisfies Σᵢ αₜ(i)·βₜ(i) = P(O|λ) at every t

4. **`viterbi(observations)`**
   - Viterbi Algorithm implementation, computed in log space
   - Returns: most likely state sequence (as LABELS) and its probability

5. **`fit(observations_sequences, n_iter, tolerance, verbose, random_state)`**
   - Baum-Welch Algorithm (EM) for learning
   - Learns parameters from training data

6. **`predict(observations)`**
   - Wrapper for Viterbi (decoding)
   - Returns: predicted state sequence

7. **`score(observations)`**
   - Wrapper for the scaled Forward algorithm (evaluation)
   - Returns: log probability of observations. Always <= 0, and longer sequences
     score lower simply because more factors are multiplied - divide by
     `len(observations)` to compare different lengths
   - Raises `ValueError` naming the symbol if a token was never seen in training.
     For anomaly detection, catch that explicitly rather than with a bare
     `except:` - an unseen symbol is evidence, not an error to swallow

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
# Paste the HiddenMarkovModel class from _23_hmm.py above this line,
# or run the file directly: python _23_hmm.py
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
P([Walk, Shop, Clean]) = 0.034656
Log probability = -3.3623

Interpretation: 3.5% chance of this sequence

(Check it by hand: sum the eight path probabilities listed above and you get
exactly 0.034656. The forward algorithm computes that same number without ever
enumerating a path.)
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

# Learn parameters using Baum-Welch.
# random_state fixes the random initialization, so this run is reproducible;
# without it every run lands in a different local optimum.
hmm_learned.fit(training_data, n_iter=100, verbose=True, random_state=42)

# Print learned parameters
hmm_learned.print_parameters()
```

**What Baum-Welch Does** (actual output, abridged):
```
Iteration 1: Log-Likelihood = -16.8714     <- random initialization
Iteration 2: Log-Likelihood = -15.7199     <- improved
Iteration 3: Log-Likelihood = -15.6572
...
Iteration 21: Log-Likelihood = -14.2176
Iteration 22: Log-Likelihood = -14.2175
Iteration 23: Log-Likelihood = -14.2175
Converged after 23 iterations              <- gain fell below tolerance
```

Notice the likelihood **never decreases**. That is not luck; it is the defining
guarantee of EM (see "Why It Works" under Understanding the Code). If you ever see
it fall, the implementation has a numerical bug.

**Output**:
```
1. Initial State Probabilities (pi):
  P(S0) = 0.3024
  P(S1) = 0.6976

2. State Transition Probabilities (A):
From \ To                S0          S1
S0                   0.0000      1.0000
S1                   0.7235      0.2765

3. Emission Probabilities (B):
State \ Obs           Clean        Shop        Walk
S0                   0.5008      0.4992      0.0000
S1                   0.3326      0.0000      0.6674
```

**Interpretation** - and an honest warning:
```
S1 emits Walk 67% of the time and never Shop  -> looks like the "Sunny" state
S0 emits Clean/Shop and never Walk            -> looks like the "Rainy" state
```
Three things this small run teaches:

1. **The state numbering is arbitrary.** Baum-Welch never sees a label, so
   "S0" and "S1" are anonymous clusters. Run it with `random_state=7` and the two
   rows may swap. This is *label switching*: always identify a learned state by
   its emission row, never by its index.
2. **Note the observation column order.** `fit()` builds its vocabulary with
   `sorted(set(...))`, so the columns come out alphabetically: Clean, Shop, Walk -
   not the order you wrote them in. Check `hmm.observation_labels`.
3. **The zeros are overfitting, not truth.** Five sequences of three observations
   is 15 data points for a model with N^2 + N*M - 1 = 4 + 6 - 1 = 9 free
   parameters. Baum-Welch drives unsupported entries to exactly 0, and a zero can
   never recover (every update to a parameter is proportional to its current
   value). Any future sequence containing "S0 then Shop then Walk" would then score
   -inf. Real systems add smoothing, or simply train on more data - see the Quick
   Start above, where 1200 observations recover the true matrix to within 0.02.

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
def _forward_pass(self, observations, scale=True):
    T = len(observations)
    alpha = np.zeros((T, self.n_states))
    c = np.ones(T)                    # the scaling coefficients c_t

    # Initialization: alpha(0, i) = pi(i) * B(i, O_0)
    alpha[0] = self.initial_prob * self.emission_prob[:, observations[0]]
    if scale:
        total = np.sum(alpha[0])
        if total <= 0.0:
            return alpha, c, -np.inf   # model gives this sequence probability 0
        c[0] = 1.0 / total
        alpha[0] = alpha[0] * c[0]

    # Recursion: alpha(t, j) = [sum_i alpha(t-1, i) * A(i,j)] * B(j, O_t)
    for t in range(1, T):
        for j in range(self.n_states):
            alpha[t, j] = np.sum(alpha[t-1] * self.transition_prob[:, j]) * \
                         self.emission_prob[j, observations[t]]
        if scale:
            total = np.sum(alpha[t])
            if total <= 0.0:
                return alpha, c, -np.inf
            c[t] = 1.0 / total         # rescale this column to sum to 1
            alpha[t] = alpha[t] * c[t]

    if scale:
        log_prob = -np.sum(np.log(c))       # log P(O|lambda) = -sum_t log c_t
    else:
        total = np.sum(alpha[T-1])          # P(O|lambda) = sum_i alpha(T-1, i)
        log_prob = np.log(total) if total > 0.0 else -np.inf

    return alpha, c, log_prob
```

`forward(observations)` is the public wrapper: it calls this and drops `c`.

The recursion in the middle is exactly the textbook formula from the Mathematical
Foundation section. Everything else is **scaling**, and it is not optional - see
"Underflow in Probabilities" under Common Pitfalls for the measurements. If you
want the raw textbook values, ask for them: `forward([0, 2], scale=False)` returns
`alpha[0] = [0.36, 0.04]` and `alpha[1] = [0.0268, 0.0924]`, reproducing the
Example Calculation above digit for digit.


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

That trace is the `scale=False` view - the raw joint probabilities. With the
default `scale=True`, each row is rescaled to sum to 1, so `alpha[1,0]+alpha[1,1]`
would be exactly 1.0 and the probability lives in the coefficients instead:
`P(O|λ) = 1 / (c_0 · c_1)`. Same answer, no underflow.

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
    observations = self._encode_observations(observations)   # labels -> indices
    T = len(observations)

    # log delta(t, i): log of the best path probability ending in state i at t
    log_delta = np.zeros((T, self.n_states))
    psi = np.zeros((T, self.n_states), dtype=int)

    # log(0) = -inf is the correct answer for an impossible move, so allow it
    with np.errstate(divide='ignore'):
        log_pi = np.log(self.initial_prob)
        log_A = np.log(self.transition_prob)
        log_B = np.log(self.emission_prob)

    # Initialization
    log_delta[0] = log_pi + log_B[:, observations[0]]

    # Recursion (products become sums in log space)
    for t in range(1, T):
        for j in range(self.n_states):
            scores = log_delta[t-1] + log_A[:, j]
            psi[t, j] = np.argmax(scores)          # best previous state
            log_delta[t, j] = np.max(scores) + log_B[j, observations[t]]

    # Termination
    path_indices = np.zeros(T, dtype=int)
    path_indices[T-1] = np.argmax(log_delta[T-1])
    max_log_prob = np.max(log_delta[T-1])

    # Backtracking
    for t in range(T-2, -1, -1):
        path_indices[t] = psi[t+1, path_indices[t+1]]

    # Convert indices back to state labels, and report a probability
    path = self._decode_sequence(path_indices, self.state_labels)
    return path, float(np.exp(max_log_prob))
```

This is the same recursion as the hand trace above, with every **product turned
into a sum of logarithms** - `a * b` becomes `log a + log b`. `argmax` is
unaffected by the monotone `log`, so the path is identical to the one you would
get by multiplying, but it no longer underflows: multiplying raw probabilities
drives `delta` below 1e-308 after a few hundred steps, at which point every state
looks equally good (all zero) and the decoded path becomes noise.

The returned `prob` is `exp(max_log_prob)`, so it matches the hand trace for short
sequences (0.00160030 for the five-day example) and genuinely underflows to 0.0
for long ones - while the path stays correct. Compare long sequences by log
probability, not by this number.

Note also what `viterbi` returns: **state labels**, not indices. `predict()` is a
one-line wrapper around it that keeps only the path.


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
def fit(self, observations_sequences, n_iter=100, tolerance=1e-4, verbose=False,
        random_state=None):
    # ... vocabulary building and encoding omitted ...

    # Initialize parameters randomly (symmetry must be broken - see below)
    self._initialize_parameters(encoded_sequences, rng=rng)

    prev_log_likelihood = float('-inf')

    for iteration in range(n_iter):
        # Accumulators for the numerators of pi, A and B
        new_initial = np.zeros(self.n_states)
        new_transition = np.zeros((self.n_states, self.n_states))
        new_emission = np.zeros((self.n_states, self.n_observations))

        total_log_likelihood = 0

        for obs_seq in encoded_sequences:
            # E-step: SCALED forward-backward. backward() must reuse the same c.
            alpha, c, log_prob = self._forward_pass(obs_seq, scale=True)
            beta = self.backward(obs_seq, c=c)
            total_log_likelihood += log_prob

            T = len(obs_seq)

            # gamma(t, i) = alpha_hat(t, i) * beta_hat(t, i) / c_t
            # Rows sum to exactly 1 - no epsilon, no renormalization needed.
            gamma = alpha * beta / c[:, np.newaxis]

            # xi(t, i, j) = alpha_hat(t,i) * A(i,j) * B(j,O_t+1) * beta_hat(t+1,j)
            # The scale factors cancel exactly, so there is no denominator.
            xi = np.zeros((max(T-1, 0), self.n_states, self.n_states))
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.transition_prob[i, j] *
                                      self.emission_prob[j, obs_seq[t+1]] * beta[t+1, j])

            # M-step: accumulate expected counts
            new_initial += gamma[0]
            if T > 1:
                new_transition += np.sum(xi, axis=0)
            for k in range(self.n_observations):
                mask = (np.array(obs_seq) == k)
                new_emission[:, k] += np.sum(gamma[mask], axis=0)

        # Normalize. Row-normalizing IS the division by sum_t gamma_t(i):
        # summing xi over j gives sum_t gamma_t(i), and summing the emission
        # counts over k gives sum_t gamma_t(i).
        self.initial_prob = new_initial / np.sum(new_initial)
        for i in range(self.n_states):
            trans_total = np.sum(new_transition[i])
            if trans_total > 0:
                self.transition_prob[i] = new_transition[i] / trans_total
            emit_total = np.sum(new_emission[i])
            if emit_total > 0:
                self.emission_prob[i] = new_emission[i] / emit_total

        # Signed test, not abs(): EM is monotone, so a change smaller than
        # tolerance only counts as convergence when the likelihood went UP.
        improvement = total_log_likelihood - prev_log_likelihood
        if 0 <= improvement < tolerance:
            break
        prev_log_likelihood = total_log_likelihood
```

**Reading the code against the formulas**: the four lines that matter map
one-for-one onto the update rules from the Mathematical Foundation section.

| Formula | Line in `fit()` |
|---------|-----------------|
| `pi_i = gamma_0(i)` | `new_initial += gamma[0]`, then divide by the total |
| `a_ij = sum_t xi_t(i,j) / sum_t gamma_t(i)`, t = 0..T-2 | `new_transition += np.sum(xi, axis=0)`, then row-normalize |
| `b_ik = sum_{t: O_t=k} gamma_t(i) / sum_t gamma_t(i)`, t = 0..T-1 | the `mask` loop, then row-normalize |

Row-normalizing is not a shortcut - it is the exact denominator. Summing
`xi[t, i, j]` over `j` gives `gamma_t(i)` (you either transition somewhere or the
sequence ended), so the row sum of `new_transition[i]` is precisely
`sum_{t=0}^{T-2} gamma_t(i)`. The same argument over `k` gives the emission
denominator, which runs to `T-1` instead because every time step emits.

**One subtlety about the reported likelihood**: `total_log_likelihood` at
iteration `k` is computed in the E-step, i.e. with the parameters produced by
iteration `k-1`. So the last number printed is one M-step behind the model you get
back. That is why the trace can print the same value twice and then stop.


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

The monotonicity is not a happy accident, and it is worth knowing where it comes
from - it is the one property that lets you debug an EM implementation.

We want to maximize `log P(O | lambda)`, but that involves a sum over all N^T state
paths and has no closed form. EM sidesteps it. For any distribution `q` over the
hidden paths S,

```
log P(O | lambda)  =  Q(lambda | lambda_old)  -  H(lambda | lambda_old)

where  Q(lambda | lambda_old) = sum_S P(S | O, lambda_old) * log P(O, S | lambda)
       H(lambda | lambda_old) = sum_S P(S | O, lambda_old) * log P(S | O, lambda)
```

- The **E-step** computes `P(S | O, lambda_old)` - which is exactly what gamma and
  xi are: its per-time-step marginals.
- The **M-step** maximizes `Q` over lambda. Because `log P(O, S | lambda)` splits
  into independent sums over pi, A and B, that maximization has the closed form we
  implemented: just normalized expected counts.
- Gibbs' inequality guarantees `H(lambda | lambda_old) <= H(lambda_old | lambda_old)`
  for every lambda. So increasing `Q` cannot be cancelled out by `H`, and
  `log P(O | lambda_new) >= log P(O | lambda_old)`.

```
Consequences you can rely on:
- The likelihood NEVER decreases. A decrease means a bug, not a bad seed.
- It converges to a LOCAL maximum (or a saddle point).
- It never tells you the global maximum was found.

Note: which optimum you reach depends entirely on the initialization
Solution: fit several times with different random_state values and keep the
          run with the highest TRAINING likelihood
```

```python
# Multiple restarts, the standard defence against local maxima
best_model, best_score = None, -np.inf
for seed in range(10):
    candidate = HiddenMarkovModel(n_states=2)
    candidate.fit(training_data, n_iter=200, tolerance=1e-6, random_state=seed)
    total = sum(candidate.score(seq) for seq in training_data)
    if total > best_score:
        best_model, best_score = candidate, total
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

# predict() returns a LIST, and list == list is a single bool, not a mask.
# Wrap both sides in np.array to get a per-position comparison.
accuracy = np.mean(np.array(predicted) == np.array(actual))
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

actual_arr, predicted_arr = np.array(actual), np.array(predicted)
f1_sunny = f1_score(actual_arr == 'Sunny', predicted_arr == 'Sunny')
f1_rainy = f1_score(actual_arr == 'Rainy', predicted_arr == 'Rainy')
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

This is *the* practical issue in HMM implementation, and it is worth being precise
about, because the forward-backward recursion cannot simply be moved into log
space: `alpha_t(j)` is a SUM of products, and there is no cheap logarithm of a sum.
(You can use log-sum-exp, and Viterbi - which only ever takes a max - does move
straight into log space; `viterbi()` in `_23_hmm.py` does exactly that.)

**Scaling (Rabiner 1989, Section V.A)** is the standard answer for
forward-backward. Instead of letting `alpha` shrink, rescale each column of the
trellis to sum to 1 as you build it, and remember the factor you used:

```
c_t          = 1 / sum_i alpha_t(i)            (computed after the recursion step)
alpha_hat_t  = c_t * alpha_t                   (so sum_i alpha_hat_t(i) = 1)
beta_hat_t   = c_t * beta_t                    (the SAME c_t, applied backwards)

Then, exactly (not approximately):
  log P(O | lambda) = -sum_t log c_t
  gamma_t(i)        = alpha_hat_t(i) * beta_hat_t(i) / c_t
  xi_t(i,j)         = alpha_hat_t(i) * a_ij * b_j(O_t+1) * beta_hat_t+1(j)
```

Why the last two are exact: writing `C_t = prod_{s<=t} c_s` and
`D_t = prod_{s>=t} c_s`, we have `alpha_hat_t = C_t * alpha_t` and
`beta_hat_t = D_t * beta_t`, and `C_t * D_t = c_t * prod_all c_s = c_t / P(O)`.
The `P(O)` that gamma and xi are supposed to be divided by cancels against the
scale factors, which is why the code has **no denominator and no epsilon** in the
E-step at all.

**What happens without it.** Measured on the weather model of this guide, over
random observation sequences, comparing two terminations against an independent
log-sum-exp reference: the naive `log(sum(alpha[T-1]) + 1e-10)` on raw alphas,
versus the scaled `-sum(log c_t)` the code actually uses.

| T | naive log P(O) | scaled log P(O) | naive error (nats) | scaled error (nats) |
|---|----------------|-----------------|--------------------|---------------------|
| 15 | -16.4477 | -16.4491 | 0.0014 | 3.6e-15 |
| 20 | -22.2974 | -22.9564 | 0.6590 | 3.6e-15 |
| 25 | -23.0203 | -28.2106 | 5.1903 | 0 |
| 30 | -23.0258 | -34.0548 | 11.0290 | 0 |
| 100 | -23.0259 | -113.2821 | 90.2562 | 1.9e-13 |
| 400 | -23.0259 | -447.3161 | 424.2903 | 1.4e-12 |

Read the naive column downwards: `alpha` underflows to zero, the epsilon takes
over, and the answer freezes at `ln(1e-10) = -23.0259` no matter what the data
says. It is not merely imprecise - it stops depending on the input.

The same collapse hits the E-step, which is worse, because it is silent. The
minimum row sum of gamma (which must be exactly 1 at every t):

| T | naive gamma row sum | scaled gamma row sum |
|---|---------------------|----------------------|
| 10 | 0.999993 | 1.000000000000 |
| 30 | 0.000016 | 1.000000000000 |
| 60 | 0.000000 | 1.000000000000 |
| 120 | 0.000000 | 1.000000000000 |

With gamma collapsed to zero, Baum-Welch reports the same clamped likelihood every
iteration, decides it has converged after two of them, and returns an all-zero
emission matrix. **A silently wrong model that reports convergence is the worst
possible failure mode** - which is why `_23_hmm.py` scales rather than clamps, and
why its convergence test is `0 <= improvement < tolerance` rather than
`abs(change) < tolerance`: a non-improvement should never be mistaken for success.

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

## Simplifications vs. Canonical HMM Implementations

`_23_hmm.py` implements the full Rabiner triad - scaled forward-backward,
log-space Viterbi, and Baum-Welch with the exact re-estimation formulas - so its
numbers match a reference implementation. What it deliberately leaves out are the
*variants* a production library such as `hmmlearn` also carries. Each is listed
with what canonical does, why it is omitted, and what it costs you.

### 1. Discrete (categorical) emissions only

**Canonical**: `hmmlearn` offers `CategoricalHMM`, `GaussianHMM`, `GMMHMM` and
`PoissonHMM`. A Gaussian HMM replaces the emission matrix `B` with a mean vector
and covariance per state, and the M-step becomes a weighted mean and covariance:

```
mu_i    = sum_t gamma_t(i) * o_t / sum_t gamma_t(i)
Sigma_i = sum_t gamma_t(i) * (o_t - mu_i)(o_t - mu_i)^T / sum_t gamma_t(i)
```

**Here**: `B` is an `n_states x n_observations` table of categorical probabilities.

**Consequence**: real-valued observations must be discretized first (bucketed, or
vector-quantized with k-means) before you can use this class. That loses
resolution inside a bucket, and the bucket edges become hyperparameters. Note the
forward, backward, Viterbi and E-step code would not change at all - only the
emission model and its M-step would.

### 2. One random restart per `fit()` call

**Canonical**: `hmmlearn` has `init_params` and users commonly wrap `fit` in a
restart loop; some libraries do `n_init` internally and keep the best run.

**Here**: `fit()` does a single random initialization, controlled by
`random_state`. The restart loop is shown explicitly under "Why It Works" - it is
five lines, and writing it yourself makes the local-maximum problem concrete
rather than hidden behind a parameter.

**Consequence**: a single `fit()` can land in a poor optimum. Always compare a few
seeds by training likelihood.

### 3. No priors / smoothing on the M-step counts

**Canonical**: many implementations add Dirichlet pseudo-counts (`+ alpha`) to the
expected counts before normalizing, or floor the parameters at a small epsilon.

**Here**: the counts are normalized as-is, which is the plain maximum-likelihood
update.

**Consequence**: parameters that receive zero expected count become exactly `0`,
and **a zero is absorbing** - every update to a parameter is proportional to its
current value, so it can never come back. You can see this in the Step-by-Step
learning example, where five short sequences produce `B[S0, Walk] = 0.0000`. Any
later sequence needing that emission scores `-inf`. The fix is more data, or add
pseudo-counts yourself before normalizing:

```python
# Laplace-style smoothing, applied to the accumulators inside the M-step
new_emission += 0.01
new_transition += 0.01
```

### 4. No topology constraints (no left-right models)

**Canonical**: speech and gesture models usually constrain `A` to be upper
triangular ("left-right"), so a state can only advance, never go back. This is
imposed by zeroing the forbidden entries at initialization; because zeros are
absorbing, Baum-Welch then preserves the structure for free.

**Here**: `A` is fully connected (ergodic).

**Consequence**: for a strictly sequential process you are fitting more parameters
than the problem has, which costs data efficiency. You can impose the constraint
yourself by zeroing entries of `transition_prob` after `fit()` initializes - but
this class gives you no hook for it.

### 5. Viterbi decoding only (no posterior / MAP decoding)

**Canonical**: `hmmlearn` offers `algorithm='viterbi'` and `algorithm='map'`. MAP
decoding picks `argmax_i gamma_t(i)` independently at each `t`, which maximizes
per-position accuracy but can return a path with zero probability (it may use a
transition that `A` forbids).

**Here**: `predict()` is always Viterbi, which returns the single most likely
*whole path*.

**Consequence**: none for correctness. Note the distinction though: Viterbi
optimizes the joint path, MAP optimizes each position. They can legitimately
disagree, and `gamma` - which the code already computes inside `fit()` - is all
you need to implement MAP if you want it.

### 6. Single-precision performance work

**Canonical**: libraries vectorize the E-step over states and often over
sequences, and drop into Cython or C for the inner loops.

**Here**: the `xi` accumulation is written as an explicit triple loop over
`t, i, j`, because that loop is a line-by-line transcription of the formula
`xi_t(i,j) = alpha_hat_t(i) * a_ij * b_j(O_t+1) * beta_hat_t+1(j)`.

**Consequence**: fitting is `O(N^2 * T * R * iterations)` with a Python-level
constant factor. For the sizes in this guide (60 sequences x 20 steps, 2 states,
200 iterations) that is under two seconds. For thousands of long sequences or
dozens of states, use a compiled library - the repo's standing advice.

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
