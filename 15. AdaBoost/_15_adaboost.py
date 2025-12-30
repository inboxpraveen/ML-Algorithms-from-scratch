import numpy as np

class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) Implementation from Scratch
    
    AdaBoost is an ensemble learning algorithm that combines multiple weak classifiers
    (typically decision stumps) to create a strong classifier. It sequentially trains
    learners, with each new learner focusing on examples that previous learners got wrong.
    
    Key Idea: "Combine weak learners through weighted voting to create a strong learner"
    
    Use Cases:
    - Binary Classification: Face detection, spam filtering
    - Medical Diagnosis: Disease prediction from symptoms
    - Fraud Detection: Identifying fraudulent transactions
    - Customer Analytics: Churn prediction, conversion prediction
    
    Key Concepts:
        Weak Learner: A classifier slightly better than random guessing (e.g., decision stump)
        Sample Weights: Importance of each training example (adaptive)
        Learner Weight (Alpha): How much to trust each weak learner
        Final Prediction: Weighted majority vote of all learners
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        Initialize the AdaBoost classifier
        
        Parameters:
        -----------
        n_estimators : int, default=50
            Number of weak learners to train sequentially
            - More estimators: Better training fit, longer training, risk overfitting
            - Fewer estimators: Faster training, may underfit
            Typical values: 50-200
            
        learning_rate : float, default=1.0
            Shrinks the contribution of each classifier
            - Lower values need more estimators but generalize better
            - learning_rate * n_estimators ≈ constant for similar performance
            - Range: 0.1 to 1.0
            Typical: 0.5-1.0 for small datasets, 0.1-0.3 for large datasets
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alphas = []  # Weights for each weak learner
        self.weak_learners = []  # Store trained weak learners
        
    def _create_decision_stump(self, X, y, weights):
        """
        Create a decision stump (1-level decision tree)
        
        A decision stump makes predictions based on a single feature threshold:
        - If feature_i <= threshold: predict class_left
        - If feature_i > threshold: predict class_right
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target labels (-1 or +1)
        weights : np.ndarray, shape (n_samples,)
            Sample weights (normalized, sum to 1)
            
        Returns:
        --------
        stump : dict
            Dictionary containing:
            - 'feature': Feature index to split on
            - 'threshold': Threshold value
            - 'left_prediction': Prediction for samples <= threshold
            - 'right_prediction': Prediction for samples > threshold
        error : float
            Weighted classification error
        """
        n_samples, n_features = X.shape
        best_error = float('inf')
        best_stump = None
        
        # Try all features
        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try all possible thresholds
            for threshold in thresholds:
                # Predict -1 for values <= threshold, +1 for values > threshold
                predictions = np.ones(n_samples)
                predictions[feature_values <= threshold] = -1
                
                # Calculate weighted error
                misclassified = (predictions != y).astype(float)
                error = np.sum(weights * misclassified)
                
                # Keep track of best split
                if error < best_error:
                    best_error = error
                    best_stump = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_prediction': -1,
                        'right_prediction': 1
                    }
                
                # Also try opposite predictions
                predictions = np.ones(n_samples) * -1
                predictions[feature_values <= threshold] = 1
                
                error = np.sum(weights * (predictions != y).astype(float))
                
                if error < best_error:
                    best_error = error
                    best_stump = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_prediction': 1,
                        'right_prediction': -1
                    }
        
        return best_stump, best_error
    
    def _stump_predict(self, stump, X):
        """
        Make predictions using a decision stump
        
        Parameters:
        -----------
        stump : dict
            Decision stump parameters
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted labels (-1 or +1)
        """
        n_samples = X.shape[0]
        feature_values = X[:, stump['feature']]
        
        predictions = np.ones(n_samples) * stump['right_prediction']
        predictions[feature_values <= stump['threshold']] = stump['left_prediction']
        
        return predictions
    
    def fit(self, X, y):
        """
        Train the AdaBoost classifier
        
        Algorithm:
        1. Initialize sample weights equally: w_i = 1/N
        2. For t = 1 to n_estimators:
           a. Train weak learner h_t on weighted data
           b. Calculate weighted error: ε_t
           c. Calculate learner weight: α_t = 0.5 × ln((1-ε_t)/ε_t)
           d. Update sample weights:
              - Correct predictions: w_i × e^(-α_t)
              - Wrong predictions: w_i × e^(α_t)
           e. Normalize weights: w_i = w_i / Σw_j
        3. Final model: H(x) = sign(Σ α_t × h_t(x))
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target labels (must be -1 or +1)
            
        Returns:
        --------
        self : AdaBoost
            Fitted classifier
        """
        # Validate input
        X = np.array(X)
        y = np.array(y)
        
        # Ensure labels are -1 and +1
        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [-1, 1])):
            raise ValueError("Labels must be -1 or +1. Got: {}".format(unique_labels))
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Initialize sample weights uniformly
        weights = np.ones(n_samples) / n_samples
        
        self.alphas = []
        self.weak_learners = []
        
        # Train weak learners sequentially
        for t in range(self.n_estimators):
            # Train decision stump on weighted data
            stump, error = self._create_decision_stump(X, y, weights)
            
            # Prevent error from being 0 or 1 (numerical stability)
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Calculate learner weight (alpha)
            # Higher alpha = lower error = more trust
            alpha = 0.5 * np.log((1 - error) / error)
            alpha = alpha * self.learning_rate  # Apply learning rate
            
            # Make predictions with this stump
            predictions = self._stump_predict(stump, X)
            
            # Update sample weights
            # Correct: multiply by e^(-alpha) (decrease weight)
            # Wrong: multiply by e^(alpha) (increase weight)
            misclassified = (predictions != y).astype(float)
            weights = weights * np.exp(alpha * misclassified)
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Store learner and its weight
            self.weak_learners.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Combines all weak learners using weighted majority voting:
        H(x) = sign(Σ α_t × h_t(x))
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted labels (-1 or +1)
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize weighted sum
        weighted_sum = np.zeros(n_samples)
        
        # Add weighted predictions from each learner
        for alpha, stump in zip(self.alphas, self.weak_learners):
            predictions = self._stump_predict(stump, X)
            weighted_sum += alpha * predictions
        
        # Return sign of weighted sum
        return np.sign(weighted_sum)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Converts weighted sum to probability using sigmoid-like transformation
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        probabilities : np.ndarray, shape (n_samples,)
            Probability of positive class (between 0 and 1)
            - Close to 0: Strong prediction for class -1
            - Close to 0.5: Uncertain
            - Close to 1: Strong prediction for class +1
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Calculate weighted sum
        weighted_sum = np.zeros(n_samples)
        for alpha, stump in zip(self.alphas, self.weak_learners):
            predictions = self._stump_predict(stump, X)
            weighted_sum += alpha * predictions
        
        # Convert to probability (sigmoid transformation)
        # Normalize by sum of alphas
        total_alpha = np.sum(np.abs(self.alphas))
        probabilities = 1 / (1 + np.exp(-2 * weighted_sum / total_alpha))
        
        return probabilities
    
    def score(self, X, y):
        """
        Calculate accuracy on given data
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to evaluate
        y : np.ndarray, shape (n_samples,)
            True labels
            
        Returns:
        --------
        accuracy : float
            Fraction of correct predictions (0 to 1)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def staged_score(self, X, y):
        """
        Calculate accuracy after each weak learner (learning curve)
        
        Useful for:
        - Visualizing training progress
        - Finding optimal number of estimators
        - Detecting overfitting
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to evaluate
        y : np.ndarray, shape (n_samples,)
            True labels
            
        Returns:
        --------
        scores : list of float
            Accuracy after each learner [accuracy_1, accuracy_2, ..., accuracy_T]
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        
        scores = []
        weighted_sum = np.zeros(n_samples)
        
        # Incrementally add each learner
        for alpha, stump in zip(self.alphas, self.weak_learners):
            predictions = self._stump_predict(stump, X)
            weighted_sum += alpha * predictions
            
            # Calculate accuracy with learners up to this point
            current_predictions = np.sign(weighted_sum)
            accuracy = np.mean(current_predictions == y)
            scores.append(accuracy)
        
        return scores
    
    def get_feature_importance(self):
        """
        Calculate feature importance
        
        Importance is based on:
        - How often a feature is used for splitting
        - The alpha (weight) of learners that use that feature
        
        Features used by high-alpha learners are more important
        
        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Normalized feature importance (sums to 1)
            importance[i] = importance of feature i
        """
        importance = np.zeros(self.n_features)
        
        for alpha, stump in zip(self.alphas, self.weak_learners):
            feature_idx = stump['feature']
            importance[feature_idx] += abs(alpha)
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def print_learners(self, max_display=10):
        """
        Print information about trained weak learners
        
        Parameters:
        -----------
        max_display : int, default=10
            Maximum number of learners to display
        """
        print(f"\n{'='*70}")
        print(f"TRAINED WEAK LEARNERS (showing top {min(max_display, len(self.weak_learners))})")
        print(f"{'='*70}")
        print(f"{'#':>3} {'Feature':>10} {'Threshold':>12} {'Alpha':>12} {'L→R':>10}")
        print(f"{'-'*70}")
        
        for i, (alpha, stump) in enumerate(zip(self.alphas[:max_display], 
                                                self.weak_learners[:max_display])):
            feature = stump['feature']
            threshold = stump['threshold']
            left = stump['left_prediction']
            right = stump['right_prediction']
            direction = f"{left:+d}→{right:+d}"
            
            print(f"{i+1:>3} {feature:>10} {threshold:>12.3f} {alpha:>12.3f} {direction:>10}")
        
        if len(self.weak_learners) > max_display:
            print(f"\n... and {len(self.weak_learners) - max_display} more learners")


"""
USAGE EXAMPLE 1: Simple Binary Classification

import numpy as np

# Create simple 2D dataset
np.random.seed(42)

# Generate linearly separable data
X_class_0 = np.random.randn(50, 2) + np.array([-2, -2])
X_class_1 = np.random.randn(50, 2) + np.array([2, 2])

X = np.vstack([X_class_0, X_class_1])
y = np.array([-1] * 50 + [1] * 50)

# Shuffle data
indices = np.random.permutation(100)
X = X[indices]
y = y[indices]

# Split train/test
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Create and train AdaBoost
model = AdaBoost(n_estimators=50, learning_rate=1.0)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("\nSample Predictions:")
for i in range(5):
    print(f"True: {y_test[i]:+d}, Predicted: {predictions[i]:+.0f}, "
          f"Probability: {probabilities[i]:.3f}")
"""

"""
USAGE EXAMPLE 2: Learning Curves and Model Selection

import numpy as np
import matplotlib.pyplot as plt

# Generate data (same as Example 1)
np.random.seed(42)
X_class_0 = np.random.randn(50, 2) + np.array([-2, -2])
X_class_1 = np.random.randn(50, 2) + np.array([2, 2])
X = np.vstack([X_class_0, X_class_1])
y = np.array([-1] * 50 + [1] * 50)

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Train model
model = AdaBoost(n_estimators=100, learning_rate=1.0)
model.fit(X_train, y_train)

# Get learning curves
train_scores = model.staged_score(X_train, y_train)
test_scores = model.staged_score(X_test, y_test)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_scores, label='Training', linewidth=2)
plt.plot(range(1, 101), test_scores, label='Testing', linewidth=2)
plt.xlabel('Number of Estimators', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('AdaBoost Learning Curves', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

# Find optimal number of estimators
optimal_n = np.argmax(test_scores) + 1
print(f"Optimal number of estimators: {optimal_n}")
print(f"Best test accuracy: {test_scores[optimal_n-1]:.2%}")
"""

"""
USAGE EXAMPLE 3: Feature Importance

import numpy as np

# Create dataset with 5 features (only first 2 are informative)
np.random.seed(42)
n_samples = 200

# Informative features
X1 = np.random.randn(n_samples, 1)
X2 = np.random.randn(n_samples, 1)

# Non-informative features (noise)
X_noise = np.random.randn(n_samples, 3)

X = np.hstack([X1, X2, X_noise])

# Target depends only on X1 and X2
y = np.where(X1.ravel() + X2.ravel() > 0, 1, -1)

# Train model
model = AdaBoost(n_estimators=50, learning_rate=1.0)
model.fit(X, y)

# Get feature importance
importance = model.get_feature_importance()

print("\nFeature Importance:")
print("="*40)
for i, imp in enumerate(importance):
    print(f"Feature {i}: {imp:.3f} {'█' * int(imp * 50)}")

# Expected: Features 0 and 1 have high importance, rest are low
"""

"""
USAGE EXAMPLE 4: Comparison with Single Decision Stump

import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Generate non-linearly separable data
np.random.seed(42)

# Create XOR-like pattern
n = 100
X = np.random.randn(n, 2) * 2
y = np.where((X[:, 0] > 0) == (X[:, 1] > 0), 1, -1)

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Train single decision stump
stump = DecisionTreeClassifier(max_depth=1)
stump.fit(X_train, y_train)
stump_acc = stump.score(X_test, y_test)

# Train AdaBoost with multiple stumps
adaboost = AdaBoost(n_estimators=50, learning_rate=1.0)
adaboost.fit(X_train, y_train)
adaboost_acc = adaboost.score(X_test, y_test)

print("Performance Comparison:")
print("="*40)
print(f"Single Decision Stump: {stump_acc:.2%}")
print(f"AdaBoost (50 stumps):  {adaboost_acc:.2%}")
print(f"Improvement:           {(adaboost_acc - stump_acc):.2%}")

# Show some weak learners
adaboost.print_learners(max_display=5)
"""

"""
USAGE EXAMPLE 5: Effect of Learning Rate

import numpy as np

# Generate data
np.random.seed(42)
X_class_0 = np.random.randn(100, 2) + np.array([-1, -1])
X_class_1 = np.random.randn(100, 2) + np.array([1, 1])
X = np.vstack([X_class_0, X_class_1])
y = np.array([-1] * 100 + [1] * 100)

X_train, X_test = X[:160], X[40:]
y_train, y_test = y[:160], y[40:]

# Try different learning rates
learning_rates = [0.1, 0.5, 1.0]

print("Effect of Learning Rate:")
print("="*60)
print(f"{'Learning Rate':>15} {'n_estimators':>15} {'Train Acc':>15} {'Test Acc':>15}")
print("-"*60)

for lr in learning_rates:
    model = AdaBoost(n_estimators=100, learning_rate=lr)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"{lr:>15.1f} {100:>15} {train_acc:>15.2%} {test_acc:>15.2%}")

# Note: Lower learning rate often gives better generalization
#       but needs more estimators for same training performance
"""

"""
USAGE EXAMPLE 6: Real-World Application - Spam Detection

import numpy as np

# Simulated email features
# Features: [word_count_free, word_count_click, exclamation_marks, 
#            caps_ratio, link_count, sender_reputation]

# Spam emails (higher values for suspicious features)
spam_features = np.random.randn(100, 6) * 0.5 + np.array([3, 2, 4, 0.5, 3, -1])
spam_labels = np.ones(100)

# Ham emails (normal values)
ham_features = np.random.randn(100, 6) * 0.5 + np.array([0.5, 0.2, 0.5, 0.1, 0.5, 1])
ham_labels = np.ones(100) * -1

# Combine data
X = np.vstack([spam_features, ham_features])
y = np.hstack([spam_labels, ham_labels])

# Shuffle
indices = np.random.permutation(200)
X = X[indices]
y = y[indices]

# Split
X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Train spam filter
spam_filter = AdaBoost(n_estimators=50, learning_rate=0.8)
spam_filter.fit(X_train, y_train)

# Evaluate
accuracy = spam_filter.score(X_test, y_test)
print(f"\nSpam Filter Accuracy: {accuracy:.2%}")

# Feature importance
feature_names = ['word_free', 'word_click', 'exclamation', 
                'caps_ratio', 'links', 'reputation']
importance = spam_filter.get_feature_importance()

print("\nMost Important Features for Spam Detection:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:15s}: {imp:.3f}")

# Test on new emails
new_emails = np.array([
    [5, 3, 6, 0.8, 4, -2],  # Likely spam
    [1, 0, 1, 0.05, 1, 2]   # Likely ham
])

predictions = spam_filter.predict(new_emails)
probabilities = spam_filter.predict_proba(new_emails)

print("\nNew Email Classifications:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"Email {i+1}: {label} (confidence: {prob if pred == 1 else 1-prob:.2%})")
"""

"""
USAGE EXAMPLE 7: Medical Diagnosis

import numpy as np

# Patient features: [age, blood_pressure, cholesterol, bmi, smoking, family_history]
# Target: Heart disease (1) or Healthy (-1)

np.random.seed(42)

# Generate synthetic patient data
# Patients with heart disease (higher risk factors)
diseased = np.random.randn(80, 6) * np.array([10, 15, 20, 3, 0.3, 0.3]) + \
           np.array([65, 150, 240, 30, 0.7, 0.8])

# Healthy patients (lower risk factors)
healthy = np.random.randn(80, 6) * np.array([15, 10, 15, 2, 0.3, 0.3]) + \
          np.array([45, 120, 180, 24, 0.2, 0.3])

X = np.vstack([diseased, healthy])
y = np.array([1] * 80 + [-1] * 80)

# Shuffle
indices = np.random.permutation(160)
X = X[indices]
y = y[indices]

# Split
X_train, X_test = X[:120], X[40:]
y_train, y_test = y[:120], y[40:]

# Train diagnostic model
model = AdaBoost(n_estimators=30, learning_rate=0.7)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
predictions = model.predict(X_test)

# Calculate precision and recall manually
true_positives = np.sum((predictions == 1) & (y_test == 1))
false_positives = np.sum((predictions == 1) & (y_test == -1))
false_negatives = np.sum((predictions == -1) & (y_test == 1))

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print("\nHeart Disease Diagnosis Model:")
print("="*50)
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%} (of predicted diseases, how many correct)")
print(f"Recall:    {recall:.2%} (of actual diseases, how many detected)")

# Feature importance for interpretation
feature_names = ['Age', 'Blood Pressure', 'Cholesterol', 
                'BMI', 'Smoking', 'Family History']
importance = model.get_feature_importance()

print("\nRisk Factor Importance:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.3f}")

# Diagnose new patients
new_patients = np.array([
    [70, 160, 250, 32, 0.9, 0.9],  # High risk
    [40, 110, 170, 22, 0.0, 0.1]   # Low risk
])

diagnoses = model.predict(new_patients)
probabilities = model.predict_proba(new_patients)

print("\nNew Patient Diagnoses:")
for i, (diag, prob) in enumerate(zip(diagnoses, probabilities)):
    risk = "HIGH RISK" if diag == 1 else "LOW RISK"
    confidence = prob if diag == 1 else 1 - prob
    print(f"Patient {i+1}: {risk} (confidence: {confidence:.2%})")

# Note: This is for educational purposes only!
# Real medical diagnosis requires professional medical evaluation
"""

