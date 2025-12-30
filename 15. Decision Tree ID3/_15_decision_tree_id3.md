# Decision Tree ID3

The **ID3 (Iterative Dichotomiser 3)** algorithm is a classic decision tree algorithm introduced by Ross Quinlan. It is used to generate a decision tree from a dataset by recursively splitting data based on the attribute that provides the highest **Information Gain**.

ID3 is designed primarily for **categorical features**. It builds the tree top-down, starting from the root node, and greedy selects the best attribute at each step.

---

### Key Concepts

#### 1. Entropy
Entropy measures the impurity or uncertainty in a dataset. If a dataset is perfectly classified (all samples belong to one class), entropy is 0. If samples are evenly distributed across classes, entropy is high.

Formula for Entropy $H(S)$:
$$H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $S$ is the dataset.
- $c$ is the number of classes.
- $p_i$ is the proportion of samples belonging to class $i$.

#### 2. Information Gain (IG)
Information Gain measures the reduction in entropy achieved by splitting the dataset on a specific attribute $A$. The attribute with the highest Information Gain is chosen as the splitting node.

Formula for Information Gain $IG(S, A)$:
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $H(S)$ is the entropy of the original dataset.
- $Values(A)$ are the unique values of attribute $A$.
- $S_v$ is the subset of $S$ where attribute $A$ has value $v$.
- $|S_v| / |S|$ is the weight of the subset (proportion of samples).

---

### Algorithm Steps

1.  **Calculate Entropy** of the entire dataset.
2.  **Calculate Information Gain** for every attribute (feature).
3.  **Select the Best Attribute** with the highest Information Gain.
4.  **Create a Decision Node** for that attribute.
5.  **Split the Dataset** into subsets based on the attribute's unique values.
6.  **Recursively Repeat** steps 1-5 for each subset until:
    - All samples in a branch belong to the same class (Entropy = 0).
    - There are no more attributes to split on.
    - No samples are left.

---

### Example: Play Tennis Dataset

Imagine we want to predict if we can play tennis based on weather conditions.

| Outlook | Temperature | Humidity | Wind | Play Tennis |
| :--- | :--- | :--- | :--- | :--- |
| Sunny | Hot | High | Weak | No |
| Sunny | Hot | High | Strong | No |
| Overcast | Hot | High | Weak | Yes |
| Rain | Mild | High | Weak | Yes |
| Rain | Cool | Normal | Weak | Yes |
| Rain | Cool | Normal | Strong | No |
| ... | ... | ... | ... | ... |

**Step 1: Root Entropy**
Calculate entropy of 'Play Tennis' (9 Yes, 5 No).
$$H(S) = - \frac{9}{14} \log_2(\frac{9}{14}) - \frac{5}{14} \log_2(\frac{5}{14}) \approx 0.940$$

**Step 2: Info Gain for 'Outlook'**
- **Sunny (5 samples)**: 2 Yes, 3 No $\rightarrow H(Sunny) \approx 0.971$
- **Overcast (4 samples)**: 4 Yes, 0 No $\rightarrow H(Overcast) = 0$
- **Rain (5 samples)**: 3 Yes, 2 No $\rightarrow H(Rain) \approx 0.971$

Weighted Entropy:
$$ (5/14 * 0.971) + (4/14 * 0) + (5/14 * 0.971) \approx 0.693 $$

Information Gain:
$$ IG(S, Outlook) = 0.940 - 0.693 = 0.247 $$

Repeat for other features (Temp, Humidity, Wind). 'Outlook' has the highest gain, so it becomes the root node.

**Step 3: Recursion**
- **Outlook = Overcast**: Entropy is 0 (All Yes). This becomes a Leaf Node (Yes).
- **Outlook = Sunny**: Recurse on remaining data.
- **Outlook = Rain**: Recurse on remaining data.

---

### Advantages & Disadvantages

**Pros:**
- Simple to understand and interpret.
- Handles categorical features naturally.
- Builds short trees (greedy approach usually finds simple trees).

**Cons:**
- **Overfitting**: Can create very complex trees if not pruned or depth-limited.
- **Categorical Only**: Standard ID3 doesn't handle continuous data (needs C4.5).
- **No Missing Data**: Cannot handle missing values natively.

---

### Reference
- [ID3 Algorithm - Wikipedia](https://en.wikipedia.org/wiki/ID3_algorithm)
