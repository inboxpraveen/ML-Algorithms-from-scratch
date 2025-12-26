import numpy as np
from itertools import combinations
from collections import defaultdict

class Apriori:
    """
    Apriori Algorithm Implementation from Scratch
    
    Apriori is a classic algorithm for association rule mining and frequent itemset discovery.
    It finds patterns and relationships in transactional data (e.g., market basket analysis).
    
    Key Idea: "If an itemset is frequent, then all of its subsets must also be frequent"
    
    Use Cases:
    - Market Basket Analysis: "Customers who buy X also buy Y"
    - Recommendation Systems: "Users who like A also like B"
    - Medical Diagnosis: "Symptoms that occur together"
    
    Key Concepts:
        Support: How often an itemset appears in the dataset
        Confidence: How often a rule is true
        Lift: How much more likely Y is purchased when X is purchased
    """
    
    def __init__(self, min_support=0.5, min_confidence=0.7):
        """
        Initialize the Apriori model
        
        Parameters:
        -----------
        min_support : float, default=0.5
            Minimum support threshold (0 to 1)
            - Support = (transactions containing itemset) / (total transactions)
            - Higher value = only very frequent patterns
            - Lower value = more patterns, but may include noise
            Typical values: 0.01-0.5 depending on dataset size
        
        min_confidence : float, default=0.7
            Minimum confidence threshold for rules (0 to 1)
            - Confidence = P(Y|X) = support(X,Y) / support(X)
            - How often the rule is true
            - Higher value = stronger, more reliable rules
            Typical values: 0.5-0.9
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []
        self.support_data = {}
    
    def _get_unique_items(self, transactions):
        """
        Get all unique items from transactions
        
        Parameters:
        -----------
        transactions : list of lists
            Each sublist represents a transaction containing items
            
        Returns:
        --------
        unique_items : set
            Set of all unique items across all transactions
        """
        unique_items = set()
        for transaction in transactions:
            for item in transaction:
                unique_items.add(frozenset([item]))
        return unique_items
    
    def _calculate_support(self, itemset, transactions):
        """
        Calculate support for an itemset
        
        Support = (number of transactions containing itemset) / (total transactions)
        
        Parameters:
        -----------
        itemset : frozenset
            Set of items to calculate support for
        transactions : list of lists
            All transactions in the dataset
            
        Returns:
        --------
        support : float
            Support value between 0 and 1
        """
        count = 0
        for transaction in transactions:
            if itemset.issubset(set(transaction)):
                count += 1
        return count / len(transactions)
    
    def _filter_candidates(self, candidates, transactions):
        """
        Filter candidates by minimum support threshold
        
        Parameters:
        -----------
        candidates : set of frozensets
            Candidate itemsets to evaluate
        transactions : list of lists
            All transactions in the dataset
            
        Returns:
        --------
        frequent_items : dict
            Dictionary mapping frequent itemsets to their support values
        """
        frequent_items = {}
        
        for candidate in candidates:
            support = self._calculate_support(candidate, transactions)
            if support >= self.min_support:
                frequent_items[candidate] = support
                self.support_data[candidate] = support
        
        return frequent_items
    
    def _generate_candidates(self, frequent_itemsets, k):
        """
        Generate candidate itemsets of size k from frequent itemsets of size k-1
        
        This is the "join" step in Apriori algorithm
        
        Parameters:
        -----------
        frequent_itemsets : list of frozensets
            Frequent itemsets of size k-1
        k : int
            Size of candidates to generate
            
        Returns:
        --------
        candidates : set of frozensets
            Candidate itemsets of size k
        """
        candidates = set()
        n = len(frequent_itemsets)
        
        # Join step: combine pairs of (k-1)-itemsets
        for i in range(n):
            for j in range(i + 1, n):
                # Union of two (k-1)-itemsets
                union = frequent_itemsets[i] | frequent_itemsets[j]
                # Only add if size is exactly k
                if len(union) == k:
                    candidates.add(union)
        
        return candidates
    
    def fit(self, transactions):
        """
        Find frequent itemsets in the transaction data
        
        Uses the Apriori principle: all subsets of frequent itemsets are frequent
        
        Algorithm:
        1. Find frequent 1-itemsets
        2. Generate candidate k-itemsets from frequent (k-1)-itemsets
        3. Filter candidates by minimum support
        4. Repeat until no more frequent itemsets found
        
        Parameters:
        -----------
        transactions : list of lists
            Each sublist represents a transaction containing items
            Example: [['milk', 'bread'], ['milk', 'eggs', 'bread'], ['eggs']]
        """
        self.transactions = transactions
        self.frequent_itemsets = {}
        self.support_data = {}
        
        # Step 1: Find frequent 1-itemsets
        candidates_1 = self._get_unique_items(transactions)
        frequent_1 = self._filter_candidates(candidates_1, transactions)
        
        if not frequent_1:
            print("Warning: No frequent itemsets found with current min_support threshold")
            return
        
        self.frequent_itemsets[1] = frequent_1
        k = 2
        
        # Step 2-4: Iteratively find frequent k-itemsets
        while True:
            # Generate candidates of size k
            previous_frequent = list(self.frequent_itemsets[k-1].keys())
            candidates_k = self._generate_candidates(previous_frequent, k)
            
            if not candidates_k:
                break
            
            # Filter by minimum support
            frequent_k = self._filter_candidates(candidates_k, transactions)
            
            if not frequent_k:
                break
            
            self.frequent_itemsets[k] = frequent_k
            k += 1
        
        print(f"Found {sum(len(items) for items in self.frequent_itemsets.values())} frequent itemsets")
    
    def generate_rules(self):
        """
        Generate association rules from frequent itemsets
        
        A rule is X → Y where:
        - X and Y are itemsets
        - X ∪ Y is a frequent itemset
        - X ∩ Y = ∅ (X and Y are disjoint)
        
        Rules are filtered by minimum confidence threshold
        
        Returns:
        --------
        rules : list of tuples
            Each tuple contains (antecedent, consequent, confidence, lift, support)
            - antecedent: X (items in "if" part)
            - consequent: Y (items in "then" part)
            - confidence: P(Y|X)
            - lift: confidence / P(Y)
            - support: P(X ∪ Y)
        """
        self.rules = []
        
        # Only consider itemsets with 2 or more items
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset in self.frequent_itemsets[k].keys():
                items = list(itemset)
                
                # Generate all possible splits of the itemset
                for i in range(1, len(items)):
                    # Generate all combinations of size i for antecedent
                    for antecedent_items in combinations(items, i):
                        antecedent = frozenset(antecedent_items)
                        consequent = itemset - antecedent
                        
                        if len(consequent) == 0:
                            continue
                        
                        # Calculate confidence: support(X∪Y) / support(X)
                        confidence = self.support_data[itemset] / self.support_data[antecedent]
                        
                        if confidence >= self.min_confidence:
                            # Calculate lift: confidence / support(Y)
                            lift = confidence / self.support_data[consequent]
                            support = self.support_data[itemset]
                            
                            self.rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'confidence': confidence,
                                'lift': lift,
                                'support': support
                            })
        
        # Sort rules by confidence (descending)
        self.rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Generated {len(self.rules)} association rules")
        return self.rules
    
    def get_frequent_itemsets(self, min_size=1):
        """
        Get all frequent itemsets with at least min_size items
        
        Parameters:
        -----------
        min_size : int, default=1
            Minimum number of items in returned itemsets
            
        Returns:
        --------
        itemsets : list of tuples
            Each tuple contains (itemset, support)
            Sorted by support (descending)
        """
        all_itemsets = []
        
        for k in range(min_size, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
            for itemset, support in self.frequent_itemsets[k].items():
                all_itemsets.append((set(itemset), support))
        
        # Sort by support (descending)
        all_itemsets.sort(key=lambda x: x[1], reverse=True)
        return all_itemsets
    
    def get_rules(self, min_confidence=None, min_lift=None):
        """
        Get association rules filtered by confidence and/or lift
        
        Parameters:
        -----------
        min_confidence : float, optional
            Override minimum confidence threshold
        min_lift : float, optional
            Minimum lift threshold (typically > 1.0)
            - Lift > 1: X and Y occur together more than by chance
            - Lift = 1: X and Y are independent
            - Lift < 1: X and Y occur together less than by chance
            
        Returns:
        --------
        rules : list of dicts
            Filtered association rules
        """
        if not self.rules:
            print("No rules generated yet. Call generate_rules() first.")
            return []
        
        filtered_rules = self.rules
        
        # Filter by confidence
        if min_confidence is not None:
            filtered_rules = [r for r in filtered_rules if r['confidence'] >= min_confidence]
        
        # Filter by lift
        if min_lift is not None:
            filtered_rules = [r for r in filtered_rules if r['lift'] >= min_lift]
        
        return filtered_rules
    
    def predict(self, basket):
        """
        Recommend items based on items in the basket
        
        Parameters:
        -----------
        basket : list
            Items currently in the basket
            
        Returns:
        --------
        recommendations : list of tuples
            Each tuple contains (item, confidence, lift)
            Sorted by confidence (descending)
        """
        if not self.rules:
            print("No rules generated yet. Call generate_rules() first.")
            return []
        
        basket_set = set(basket)
        recommendations = {}
        
        # Find rules where antecedent is subset of basket
        for rule in self.rules:
            if rule['antecedent'].issubset(basket_set):
                # Recommend items in consequent that are not in basket
                for item in rule['consequent']:
                    if item not in basket_set:
                        # Keep the highest confidence for each item
                        if item not in recommendations or rule['confidence'] > recommendations[item][0]:
                            recommendations[item] = (rule['confidence'], rule['lift'])
        
        # Convert to sorted list
        rec_list = [(item, conf, lift) for item, (conf, lift) in recommendations.items()]
        rec_list.sort(key=lambda x: x[1], reverse=True)
        
        return rec_list
    
    def print_frequent_itemsets(self, max_display=10):
        """
        Print frequent itemsets in a readable format
        
        Parameters:
        -----------
        max_display : int, default=10
            Maximum number of itemsets to display
        """
        itemsets = self.get_frequent_itemsets()
        
        print(f"\n{'='*70}")
        print(f"FREQUENT ITEMSETS (showing top {min(max_display, len(itemsets))})")
        print(f"{'='*70}")
        print(f"{'Itemset':<40} {'Support':>10}")
        print(f"{'-'*70}")
        
        for i, (itemset, support) in enumerate(itemsets[:max_display]):
            itemset_str = '{' + ', '.join(sorted(str(item) for item in itemset)) + '}'
            print(f"{itemset_str:<40} {support:>10.3f}")
        
        if len(itemsets) > max_display:
            print(f"\n... and {len(itemsets) - max_display} more itemsets")
    
    def print_rules(self, max_display=10):
        """
        Print association rules in a readable format
        
        Parameters:
        -----------
        max_display : int, default=10
            Maximum number of rules to display
        """
        if not self.rules:
            print("No rules generated yet. Call generate_rules() first.")
            return
        
        print(f"\n{'='*90}")
        print(f"ASSOCIATION RULES (showing top {min(max_display, len(self.rules))})")
        print(f"{'='*90}")
        print(f"{'Rule':<45} {'Confidence':>12} {'Lift':>10} {'Support':>10}")
        print(f"{'-'*90}")
        
        for i, rule in enumerate(self.rules[:max_display]):
            ant = '{' + ', '.join(sorted(str(item) for item in rule['antecedent'])) + '}'
            con = '{' + ', '.join(sorted(str(item) for item in rule['consequent'])) + '}'
            rule_str = f"{ant} → {con}"
            
            print(f"{rule_str:<45} {rule['confidence']:>12.3f} {rule['lift']:>10.3f} {rule['support']:>10.3f}")
        
        if len(self.rules) > max_display:
            print(f"\n... and {len(self.rules) - max_display} more rules")


"""
USAGE EXAMPLE 1: Simple Market Basket Analysis

import numpy as np

# Sample transaction data: grocery store purchases
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'eggs'],
    ['bread', 'eggs'],
    ['milk', 'butter'],
    ['milk', 'bread', 'eggs'],
    ['bread', 'butter', 'eggs'],
    ['milk', 'bread']
]

# Create and fit the model
model = Apriori(min_support=0.4, min_confidence=0.7)
model.fit(transactions)

# Display frequent itemsets
model.print_frequent_itemsets(max_display=10)

# Generate and display association rules
rules = model.generate_rules()
model.print_rules(max_display=10)

# Output shows patterns like:
# {milk, bread} → {butter} with high confidence
# Meaning: Customers who buy milk and bread often buy butter too
"""

"""
USAGE EXAMPLE 2: Product Recommendations

# Customer's current shopping basket
current_basket = ['milk', 'bread']

# Get recommendations
recommendations = model.predict(current_basket)

print("\nProduct Recommendations:")
print(f"{'Item':<20} {'Confidence':>12} {'Lift':>10}")
print("-" * 45)

for item, confidence, lift in recommendations:
    print(f"{item:<20} {confidence:>12.3f} {lift:>10.3f}")

# Output might show:
# butter               0.857       1.200
# eggs                 0.714       1.050
# Suggesting that butter is a strong recommendation
"""

"""
USAGE EXAMPLE 3: Online Store - Electronics

# E-commerce transactions
transactions = [
    ['laptop', 'mouse', 'keyboard'],
    ['laptop', 'mouse', 'usb_drive'],
    ['desktop', 'monitor', 'keyboard'],
    ['laptop', 'mouse', 'laptop_bag'],
    ['tablet', 'stylus', 'case'],
    ['laptop', 'mouse', 'keyboard', 'usb_drive'],
    ['desktop', 'monitor', 'mouse', 'keyboard'],
    ['laptop', 'laptop_bag'],
    ['tablet', 'case'],
    ['laptop', 'mouse', 'usb_drive', 'laptop_bag']
]

# Find associations
model = Apriori(min_support=0.3, min_confidence=0.6)
model.fit(transactions)
rules = model.generate_rules()

# Show strong associations (lift > 1.5)
strong_rules = model.get_rules(min_lift=1.5)

print("\nStrong Associations (Lift > 1.5):")
for rule in strong_rules[:5]:
    ant = ', '.join(rule['antecedent'])
    con = ', '.join(rule['consequent'])
    print(f"If customer buys [{ant}]")
    print(f"  → They likely also buy [{con}]")
    print(f"  Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f}\n")
"""

"""
USAGE EXAMPLE 4: Movie Recommendations

# User movie viewing history
transactions = [
    ['Inception', 'Interstellar', 'The Prestige'],
    ['Inception', 'The Dark Knight', 'Batman Begins'],
    ['Interstellar', 'The Martian', 'Gravity'],
    ['Inception', 'Shutter Island', 'The Prestige'],
    ['The Dark Knight', 'Batman Begins', 'Man of Steel'],
    ['Inception', 'Interstellar', 'Shutter Island'],
    ['Interstellar', 'The Martian'],
    ['Inception', 'The Prestige', 'Shutter Island'],
    ['The Dark Knight', 'Batman Begins'],
    ['Inception', 'Interstellar', 'The Prestige', 'Shutter Island']
]

# Find movie associations
model = Apriori(min_support=0.3, min_confidence=0.65)
model.fit(transactions)
rules = model.generate_rules()

# Recommend movies for user
user_watched = ['Inception', 'The Dark Knight']
recommendations = model.predict(user_watched)

print("Because you watched:", ', '.join(user_watched))
print("\nYou might also enjoy:")
for i, (movie, confidence, lift) in enumerate(recommendations[:5], 1):
    print(f"{i}. {movie} (confidence: {confidence:.1%})")
"""

"""
USAGE EXAMPLE 5: Medical Diagnosis - Symptom Patterns

# Patient symptoms data
transactions = [
    ['fever', 'cough', 'fatigue'],
    ['fever', 'headache', 'body_ache'],
    ['cough', 'sore_throat', 'runny_nose'],
    ['fever', 'cough', 'fatigue', 'body_ache'],
    ['headache', 'nausea', 'dizziness'],
    ['fever', 'cough', 'sore_throat'],
    ['fever', 'body_ache', 'fatigue'],
    ['cough', 'runny_nose', 'sore_throat'],
    ['fever', 'headache', 'body_ache', 'fatigue'],
    ['cough', 'sore_throat', 'runny_nose', 'fatigue']
]

# Find symptom associations
model = Apriori(min_support=0.3, min_confidence=0.6)
model.fit(transactions)
rules = model.generate_rules()

model.print_rules(max_display=8)

# Analyze specific symptom combination
observed_symptoms = ['fever', 'cough']
likely_symptoms = model.predict(observed_symptoms)

print(f"\nObserved symptoms: {', '.join(observed_symptoms)}")
print("Likely co-occurring symptoms:")
for symptom, confidence, lift in likely_symptoms[:3]:
    print(f"  - {symptom}: {confidence:.1%} confidence")

# Note: This is for educational purposes only
# Real medical diagnosis requires professional medical evaluation
"""

"""
USAGE EXAMPLE 6: Web Clickstream Analysis

# Pages visited in sessions
transactions = [
    ['home', 'products', 'cart', 'checkout'],
    ['home', 'products', 'details'],
    ['home', 'search', 'products', 'details'],
    ['home', 'products', 'cart'],
    ['home', 'blog', 'products'],
    ['home', 'products', 'details', 'cart', 'checkout'],
    ['home', 'search', 'products'],
    ['home', 'products', 'details', 'cart'],
    ['home', 'about', 'contact'],
    ['home', 'products', 'search', 'details']
]

# Find navigation patterns
model = Apriori(min_support=0.25, min_confidence=0.5)
model.fit(transactions)
rules = model.generate_rules()

print("\nUser Navigation Patterns:")
print("="*70)

# Show paths that lead to checkout
checkout_rules = [r for r in model.rules if 'checkout' in r['consequent']]

for rule in checkout_rules:
    path = ' → '.join(rule['antecedent'])
    print(f"Path: {path}")
    print(f"  Leads to checkout: {rule['confidence']:.1%} of the time\n")

# This helps identify:
# - Which page sequences lead to conversions
# - Where users drop off
# - Opportunities for optimization
"""

"""
USAGE EXAMPLE 7: Finding Optimal Support and Confidence

# Experiment with different thresholds
transaction_data = [
    ['A', 'B', 'C'],
    ['A', 'B'],
    ['A', 'C'],
    ['B', 'C'],
    ['A', 'B', 'C', 'D'],
    ['B', 'D'],
    ['A', 'C', 'D'],
    ['A', 'B', 'D'],
    ['B', 'C', 'D'],
    ['A', 'B', 'C']
]

# Test different parameter combinations
support_values = [0.2, 0.3, 0.4, 0.5]
confidence_values = [0.5, 0.6, 0.7, 0.8]

print("Testing Different Parameters:")
print("="*70)
print(f"{'Support':>10} {'Confidence':>12} {'# Itemsets':>15} {'# Rules':>12}")
print("-"*70)

for sup in support_values:
    for conf in confidence_values:
        model = Apriori(min_support=sup, min_confidence=conf)
        model.fit(transaction_data)
        rules = model.generate_rules()
        
        n_itemsets = sum(len(items) for items in model.frequent_itemsets.values())
        n_rules = len(rules)
        
        print(f"{sup:>10.1f} {conf:>12.1f} {n_itemsets:>15} {n_rules:>12}")

# Observations:
# - Lower support = more itemsets and rules
# - Higher confidence = fewer but stronger rules
# - Balance depends on your use case and data size
"""

