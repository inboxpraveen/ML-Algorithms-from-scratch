import numpy as np
from itertools import combinations

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
    
    Key Concepts (the four formulas this code actually computes, ASCII notation):
        Support:    support(X)      = |{t in T : X subset of t}| / |T|
                    "How often an itemset appears in the dataset"
        Confidence: confidence(X -> Y) = support(X union Y) / support(X) = P(Y|X)
                    "How often the rule is true"
        Lift:       lift(X -> Y)      = confidence(X -> Y) / support(Y) = P(Y|X) / P(Y)
                    "How much more likely Y is when X is present, vs. by chance"
        Conviction: conviction(X -> Y) = (1 - support(Y)) / (1 - confidence(X -> Y))
                    "How much more often the rule would be wrong if X and Y were
                     independent"; inf when confidence == 1.0

    Candidate Generation (the step the algorithm is named for) is JOIN + PRUNE:
        JOIN:  union every pair of frequent (k-1)-itemsets, keep unions of size k
        PRUNE: discard a candidate C unless EVERY (k-1)-subset of C is frequent
               (contrapositive of the Apriori property: if any subset is
                infrequent, C cannot possibly be frequent, so never scan for it)

    Note on predict(): confidence and lift reported for a recommended item are the
    values of the winning RULE. If that rule has a multi-item consequent, they are
    the set-level metrics for the whole consequent, not per-item metrics.
    """

    def __init__(self, min_support=0.5, min_confidence=0.7, verbose=True):
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

        verbose : bool, default=True
            Print progress messages ("Found N frequent itemsets", etc.)
            - True  = interactive/teaching mode (the historical behaviour)
            - False = library mode; useful inside loops such as a parameter sweep,
              where the progress chatter would shred a results table
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.verbose = verbose
        self.frequent_itemsets = {}
        self.rules = []
        self.support_data = {}
        self.transactions = None   # set by fit(); None means "not fitted yet"

    def _check_fitted(self):
        """
        Raise a clear error if fit() has not been called yet.

        Without this guard the caller gets an empty result or a cryptic KeyError
        instead of being told what they actually forgot to do.
        """
        if self.transactions is None:
            raise ValueError(
                "This Apriori instance is not fitted yet. "
                "Call fit(transactions) before using this method."
            )

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

        This is the "join" step FOLLOWED BY the "prune" step of Apriori - the prune
        step is the optimization the whole algorithm is named for.

        JOIN:  union every pair of frequent (k-1)-itemsets; keep unions of size k.
        PRUNE: keep a candidate C only if ALL k of its (k-1)-subsets are frequent.

        Why prune is valid (the Apriori / downward-closure property):
            X subset of Y  =>  support(Y) <= support(X)
            So if even ONE (k-1)-subset of C is infrequent, C is infrequent too.
            We can throw C away WITHOUT scanning the database for it.

        Why prune matters: it removes candidates BEFORE they cost a full database
        scan in _filter_candidates. Measured on 300 random baskets of 5-10 items
        drawn from 14 items at min_support=0.15: at k=4 the join alone proposes
        342 candidates while join+prune proposes 50. Summed over k >= 2 that is
        797 database scans versus 505 (+57.8% wasted work without the prune).
        The frequent itemsets found are identical either way - prune costs
        nothing in correctness, only removes provably hopeless candidates.

        Parameters:
        -----------
        frequent_itemsets : list of frozensets
            Frequent itemsets of size k-1
        k : int
            Size of candidates to generate

        Returns:
        --------
        candidates : set of frozensets
            Candidate itemsets of size k (already subset-pruned)
        """
        candidates = set()
        n = len(frequent_itemsets)
        # Membership set for the prune test: "is this (k-1)-itemset frequent?"
        previous_frequent = set(frequent_itemsets)

        # Join step: combine pairs of (k-1)-itemsets
        for i in range(n):
            for j in range(i + 1, n):
                # Union of two (k-1)-itemsets
                union = frequent_itemsets[i] | frequent_itemsets[j]
                # Only consider it if size is exactly k
                if len(union) == k:
                    # Prune step: every (k-1)-subset must itself be frequent
                    subsets_all_frequent = all(
                        frozenset(subset) in previous_frequent
                        for subset in combinations(sorted(union, key=str), k - 1)
                    )
                    if subsets_all_frequent:
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

        Returns:
        --------
        self : Apriori
            The fitted model, so that Apriori(...).fit(tx) works as one expression
        """
        transactions = list(transactions)
        if len(transactions) == 0:
            raise ValueError(
                "fit() needs at least one transaction; got an empty list. "
                "Expected a list of lists, e.g. [['milk', 'bread'], ['eggs']]."
            )

        self.transactions = transactions
        self.frequent_itemsets = {}
        self.support_data = {}
        # Rules belong to the data we were last fitted on - drop stale ones,
        # otherwise predict() would answer from a previous dataset's rules.
        self.rules = []

        # Step 1: Find frequent 1-itemsets
        candidates_1 = self._get_unique_items(transactions)
        frequent_1 = self._filter_candidates(candidates_1, transactions)

        if not frequent_1:
            if self.verbose:
                print("Warning: No frequent itemsets found with current min_support threshold")
            return self

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

        if self.verbose:
            print(f"Found {sum(len(items) for items in self.frequent_itemsets.values())} frequent itemsets")
        return self

    def generate_rules(self):
        """
        Generate association rules from frequent itemsets
        
        A rule is X -> Y where:
        - X and Y are itemsets
        - X union Y is a frequent itemset
        - X intersect Y is empty (X and Y are disjoint)
        
        Rules are filtered by minimum confidence threshold

        Returns:
        --------
        rules : list of dicts
            Each rule is a dict with these keys (NOT a tuple - index by key):
            - 'antecedent' : set    X, the items in the "if" part
            - 'consequent' : set    Y, the items in the "then" part
            - 'confidence' : float  P(Y|X) = support(X u Y) / support(X)
            - 'lift'       : float  confidence / support(Y)
            - 'support'    : float  support(X u Y)
            - 'conviction' : float  (1 - support(Y)) / (1 - confidence),
                                    float('inf') when confidence == 1.0

            Iterate like this:
                for rule in model.generate_rules():
                    print(rule['antecedent'], rule['consequent'], rule['confidence'])

            Sorted by confidence descending, with a deterministic tie-break on the
            sorted antecedent then consequent, so repeated runs print the same order.
        """
        self._check_fitted()
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
                        
                        # Calculate confidence: support(X union Y) / support(X)
                        confidence = self.support_data[itemset] / self.support_data[antecedent]
                        
                        if confidence >= self.min_confidence:
                            # Calculate lift: confidence / support(Y)
                            support_consequent = self.support_data[consequent]
                            lift = confidence / support_consequent
                            support = self.support_data[itemset]

                            # Conviction: (1 - support(Y)) / (1 - confidence).
                            # A perfect rule (confidence 1.0) never fails, so the
                            # denominator is 0 and conviction is infinite.
                            if confidence >= 1.0:
                                conviction = float('inf')
                            else:
                                conviction = (1 - support_consequent) / (1 - confidence)

                            self.rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'confidence': confidence,
                                'lift': lift,
                                'support': support,
                                'conviction': conviction
                            })

        # Sort rules by confidence (descending). Itemsets come out of Python sets,
        # whose iteration order depends on PYTHONHASHSEED, so ties would otherwise
        # print in a different order on every run. The sorted antecedent and
        # consequent are the deterministic tie-break.
        self.rules.sort(key=lambda r: (-r['confidence'],
                                       sorted(map(str, r['antecedent'])),
                                       sorted(map(str, r['consequent']))))

        if self.verbose:
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
            Sorted by support (descending), ties broken by the sorted item names
            so the output order is reproducible across runs
        """
        self._check_fitted()
        all_itemsets = []

        for k in range(min_size, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
            for itemset, support in self.frequent_itemsets[k].items():
                all_itemsets.append((set(itemset), support))

        # Sort by support (descending). The sorted item names are a deterministic
        # tie-break - without one, equally-supported itemsets come out of the
        # underlying Python sets in a PYTHONHASHSEED-dependent order.
        all_itemsets.sort(key=lambda x: (-x[1], sorted(map(str, x[0]))))
        return all_itemsets
    
    def get_rules(self, min_confidence=None, min_lift=None, min_conviction=None):
        """
        Get association rules filtered by confidence, lift and/or conviction

        IMPORTANT: this filters the list that generate_rules() already produced.
        It can only TIGHTEN the thresholds, never loosen them. Rules below the
        min_confidence passed to __init__ were never created, so asking here for a
        lower min_confidence cannot bring them back - refit with a lower
        min_confidence instead.

        Parameters:
        -----------
        min_confidence : float, optional
            Additional (stricter) confidence filter on already-generated rules.
            Must be >= the min_confidence used at generate_rules() time to have
            the meaning you expect; a lower value simply changes nothing.
        min_lift : float, optional
            Minimum lift threshold (typically > 1.0)
            - Lift > 1: X and Y occur together more than by chance
            - Lift = 1: X and Y are independent
            - Lift < 1: X and Y occur together less than by chance
        min_conviction : float, optional
            Minimum conviction threshold (typically > 1.5 for a "rarely wrong" rule)
            - conviction = (1 - support(Y)) / (1 - confidence)
            - conviction = 1: X and Y independent; inf: rule never fails

        Returns:
        --------
        rules : list of dicts
            Filtered association rules, same dict shape as generate_rules()
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

        # Filter by conviction
        if min_conviction is not None:
            filtered_rules = [r for r in filtered_rules if r['conviction'] >= min_conviction]

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
            Sorted by confidence (descending), ties broken by item name.

            Only rules whose antecedent is fully contained in the basket can fire,
            and items already in the basket are never recommended back.
            For each candidate item the HIGHEST-confidence firing rule wins.

            Caveat: confidence and lift are the winning RULE's values. If that rule
            has a multi-item consequent, e.g. {usb_drive} -> {laptop, mouse}, the
            reported numbers describe the whole consequent set, not the single
            item. Read them as "the rule that recommends this item scores X",
            not "this item alone has lift X".
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
        
        # Convert to sorted list. Ties are broken by item name so that repeated
        # runs return the same ordering (dict order here follows rule order,
        # which itself depends on set iteration without a tie-break).
        rec_list = [(item, conf, lift) for item, (conf, lift) in recommendations.items()]
        rec_list.sort(key=lambda x: (-x[1], str(x[0])))

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
            # ASCII arrow: a literal '->' survives the Windows cp1252 console,
            # a U+2192 arrow raises UnicodeEncodeError and kills the whole table.
            rule_str = f"{ant} -> {con}"
            
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

# Actual output on this data - 7 frequent itemsets and exactly 3 rules:
#   {butter} -> {bread}   confidence 0.800   lift 1.000   support 0.400
#   {eggs}   -> {bread}   confidence 0.800   lift 1.000   support 0.400
#   {milk}   -> {bread}   confidence 0.714   lift 0.893   support 0.500
# Meaning: 80% of customers who buy butter also buy bread.
# Note: no 3-itemset survives min_support=0.4 here. {milk, butter} has
# support 0.3 and {butter, eggs} has support 0.2, so the PRUNE step
# discards every 3-candidate without a single database scan.
"""

"""
USAGE EXAMPLE 2: Product Recommendations

# Continues USAGE EXAMPLE 1 - it reuses the `model` fitted there.
# Run EXAMPLE 1 first, or this will raise NameError.

# Customer's current shopping basket.
# Note: predict(['milk', 'bread']) returns [] on this model, because every
# surviving rule has {bread} as its consequent and bread is already in the
# basket. Drop bread from the basket to see a recommendation fire.
current_basket = ['milk', 'butter']

# Get recommendations
recommendations = model.predict(current_basket)

print("\nProduct Recommendations:")
print(f"{'Item':<20} {'Confidence':>12} {'Lift':>10}")
print("-" * 45)

for item, confidence, lift in recommendations:
    print(f"{item:<20} {confidence:>12.3f} {lift:>10.3f}")

# Actual output:
# Item                   Confidence       Lift
# ---------------------------------------------
# bread                       0.800      1.000
# Only bread is recommended: the {butter} -> {bread} rule fires at
# confidence 0.800, beating {milk} -> {bread} at 0.714, and predict()
# keeps the highest-confidence rule per recommended item.
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
    # sorted() so the printed item order is the same on every run - antecedent
    # and consequent are plain sets, whose iteration order is not stable
    ant = ', '.join(sorted(rule['antecedent']))
    con = ', '.join(sorted(rule['consequent']))
    print(f"If customer buys [{ant}]")
    print(f"  -> They likely also buy [{con}]")
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

# Find navigation patterns.
# min_support must be <= 0.2 here: 'checkout' appears in 2 of the 10
# sessions (support 0.2), so at 0.25 it is never frequent and the
# checkout analysis below would print nothing at all.
model = Apriori(min_support=0.2, min_confidence=0.5)
model.fit(transactions)
rules = model.generate_rules()

print("\nUser Navigation Patterns:")
print("="*70)

# Show paths that lead to checkout.
# Match the consequent EXACTLY: `'checkout' in r['consequent']` would also
# match rules like {cart} -> {checkout, home}, printing the same antecedent
# path several times over.
checkout_rules = [r for r in model.rules if r['consequent'] == {'checkout'}]

for rule in checkout_rules:
    path = ' -> '.join(sorted(rule['antecedent']))
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
        # verbose=False keeps 'Found N frequent itemsets' out of the table
        model = Apriori(min_support=sup, min_confidence=conf, verbose=False)
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


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python "_13_apriori.py"
    # Requires numpy only. ASCII-only output. Runs in well under a second.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # ================================================================
    # DEMO 1 - Hand-checkable market basket (known-answer test)
    # ================================================================
    print("=" * 55)
    print("DEMO 1 - Market basket you can verify with a pencil")
    print("=" * 55)

    groceries = [
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
    print("10 transactions, min_support=0.4, min_confidence=0.7")
    print("Count by hand: bread is in 8 of 10 baskets -> support 0.8")

    market = Apriori(min_support=0.4, min_confidence=0.7)
    market.fit(groceries)
    market.print_frequent_itemsets(max_display=10)
    market.generate_rules()
    market.print_rules(max_display=10)

    print("\nWhy the search stops at size 2:")
    print("  Joining the frequent 2-itemsets proposes {milk,bread,butter},")
    print("  {milk,bread,eggs} and {bread,butter,eggs}. The PRUNE step kills all")
    print("  three - support({milk,butter})=0.3, support({milk,eggs})=0.3 and")
    print("  support({butter,eggs})=0.2 are all below 0.4 - so zero database")
    print("  scans happen at k=3 and the algorithm terminates.")

    # ================================================================
    # DEMO 2 - Planted associations, train vs held-out transactions
    # ================================================================
    print("\n" + "=" * 55)
    print("DEMO 2 - Planted associations: train vs held-out baskets")
    print("=" * 55)

    # Build 250 synthetic baskets with TWO associations planted in them:
    #   bread  -> butter  in 80% of the baskets that contain bread
    #   laptop -> mouse   in 90% of the baskets that contain laptop
    # Everything else is independent noise. A correct miner must recover
    # confidences near 0.80 and 0.90 and must NOT invent rules among the noise.
    n_baskets = 250
    noise_items = ['pen', 'soap', 'tea', 'jam', 'rice', 'salt']
    noise_probs = [0.30, 0.25, 0.20, 0.30, 0.25, 0.15]

    baskets = []
    for _ in range(n_baskets):
        basket = []
        if np.random.rand() < 0.50:                 # bread in ~50% of baskets
            basket.append('bread')
            if np.random.rand() < 0.80:             # ...and butter in 80% of those
                basket.append('butter')
        if np.random.rand() < 0.40:                 # laptop in ~40% of baskets
            basket.append('laptop')
            if np.random.rand() < 0.90:             # ...and mouse in 90% of those
                basket.append('mouse')
        for item, prob in zip(noise_items, noise_probs):
            if np.random.rand() < prob:
                basket.append(item)
        baskets.append(basket)

    # Clean split - no overlap. Baskets are i.i.d., so no shuffle is needed.
    train_baskets = baskets[:200]
    test_baskets = baskets[200:]
    print(f"Baskets: {len(train_baskets)} train (0:200), {len(test_baskets)} held-out (200:250)")

    miner = Apriori(min_support=0.15, min_confidence=0.60)
    miner.fit(train_baskets)
    train_rules = miner.generate_rules()

    def empirical_confidence(rules_antecedent, rules_consequent, transaction_list):
        """
        Measure confidence(X -> Y) directly on a set of transactions.

        confidence = |{t : X and Y both subset of t}| / |{t : X subset of t}|
        This is the same formula generate_rules() uses, evaluated on data the
        model never saw - the honest "test metric" for an unsupervised miner.
        """
        fired = 0
        correct = 0
        for transaction in transaction_list:
            items = set(transaction)
            if rules_antecedent.issubset(items):
                fired += 1
                if rules_consequent.issubset(items):
                    correct += 1
        if fired == 0:
            return float('nan'), 0
        return correct / fired, fired

    def show_rule(rule):
        """Print one rule with its training confidence and its held-out confidence."""
        ant = '{' + ','.join(sorted(rule['antecedent'])) + '}'
        con = '{' + ','.join(sorted(rule['consequent'])) + '}'
        test_conf, n_fired = empirical_confidence(
            set(rule['antecedent']), set(rule['consequent']), test_baskets)
        print(f"{ant + ' -> ' + con:<28} {rule['confidence']:>11.3f} "
              f"{test_conf:>11.3f} {n_fired:>8d} {rule['lift']:>7.3f}")

    header = (f"{'Rule':<28} {'Train conf':>11} {'Test conf':>11} "
              f"{'Test n':>8} {'Lift':>7}")

    # -- Known-answer check: did the miner recover the two PLANTED rules? --
    print("\nKnown-answer check - the two rules planted in the data:")
    print(header)
    print("-" * 68)
    planted = [({'bread'}, {'butter'}, 0.80), ({'laptop'}, {'mouse'}, 0.90)]
    for antecedent, consequent, true_conf in planted:
        found = [r for r in train_rules
                 if r['antecedent'] == antecedent and r['consequent'] == consequent]
        if found:
            show_rule(found[0])
            print(f"    planted confidence was {true_conf:.2f}")
        else:
            print(f"    MISSED: {antecedent} -> {consequent}")

    # -- The strongest rules the miner found on its own --
    print("\nTop 5 rules by confidence (train vs held-out):")
    print(header)
    print("-" * 68)
    for rule in train_rules[:5]:
        show_rule(rule)

    print("\nReading the table: the miner recovers both planted associations with")
    print("train AND held-out confidence close to the 0.80 / 0.90 that generated")
    print("the data, and lift well above 1.0. The top-5 list is dominated by the")
    print("REVERSE rules ({butter} -> {bread}, {mouse} -> {laptop}) at confidence")
    print("1.000 - butter was only ever added to a basket that already had bread,")
    print("so that direction is deterministic by construction. Note also that the")
    print("six independent noise items produce no rules at all: correct behaviour.")

    # ================================================================
    # DEMO 3 - Using the mined rules to recommend
    # ================================================================
    print("\n" + "=" * 55)
    print("DEMO 3 - Recommendations from the mined rules")
    print("=" * 55)

    for sample_basket in (['bread'], ['laptop'], ['tea']):
        recommendations = miner.predict(sample_basket)
        print(f"\nBasket {sample_basket}:")
        if not recommendations:
            print("  (no rule fires - nothing to recommend)")
        for item, confidence, lift in recommendations:
            print(f"  -> {item:<8} confidence={confidence:.3f}  lift={lift:.3f}")
