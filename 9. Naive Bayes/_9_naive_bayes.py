import numpy as np

class NaiveBayes:
    """
    Naive Bayes Classifier Implementation from Scratch
    
    Naive Bayes is a simple, fast, and effective probabilistic classifier
    based on Bayes' Theorem with the "naive" assumption that features are
    independent of each other given the class label.
    
    Key Idea: "Calculate the probability of each class given the features,
               then predict the class with highest probability"
    
    Formula: P(class|features) ∝ P(class) × P(features|class)
    
    where:
        P(class|features) = probability of class given the features (posterior)
        P(class) = probability of class (prior)
        P(features|class) = probability of features given class (likelihood)
    """
    
    def __init__(self, variant='gaussian'):
        """
        Initialize the Naive Bayes classifier
        
        Parameters:
        -----------
        variant : str, default='gaussian'
            Type of Naive Bayes to use
            Options: 'gaussian', 'multinomial'
            - gaussian: For continuous features (assumes Gaussian distribution)
            - multinomial: For discrete features (word counts, frequencies)
        """
        self.variant = variant
        self.classes = None
        self.class_priors = None
        
        # For Gaussian Naive Bayes
        self.means = None
        self.variances = None
        
        # For Multinomial Naive Bayes
        self.feature_probs = None
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Learns the prior probabilities P(class) and the likelihood 
        probabilities P(features|class) from the training data.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Target class labels
        """
        n_samples, n_features = X.shape
        
        # Get unique classes and their counts
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Calculate prior probabilities P(class)
        # Prior = (count of samples in class) / (total samples)
        self.class_priors = np.zeros(n_classes)
        for idx, c in enumerate(self.classes):
            self.class_priors[idx] = np.sum(y == c) / n_samples
        
        if self.variant == 'gaussian':
            # For Gaussian Naive Bayes: calculate mean and variance for each feature per class
            self.means = np.zeros((n_classes, n_features))
            self.variances = np.zeros((n_classes, n_features))
            
            for idx, c in enumerate(self.classes):
                # Get all samples belonging to class c
                X_c = X[y == c]
                
                # Calculate mean and variance for each feature
                self.means[idx, :] = np.mean(X_c, axis=0)
                self.variances[idx, :] = np.var(X_c, axis=0) + 1e-9  # Add small value to avoid division by zero
        
        elif self.variant == 'multinomial':
            # For Multinomial Naive Bayes: calculate feature probabilities
            # Feature probability = (count of feature in class + 1) / (total count in class + n_features)
            # The +1 is Laplace smoothing to avoid zero probabilities
            self.feature_probs = np.zeros((n_classes, n_features))
            
            for idx, c in enumerate(self.classes):
                X_c = X[y == c]
                # Count occurrences and apply Laplace smoothing
                feature_counts = np.sum(X_c, axis=0)
                total_count = np.sum(feature_counts)
                self.feature_probs[idx, :] = (feature_counts + 1) / (total_count + n_features)
    
    def _calculate_gaussian_likelihood(self, x, class_idx):
        """
        Calculate likelihood P(features|class) using Gaussian distribution
        
        For each feature, calculate probability density using:
        P(x|class) = (1/√(2π×σ²)) × exp(-(x-μ)²/(2×σ²))
        
        Parameters:
        -----------
        x : numpy array of shape (n_features,)
            Single sample features
        class_idx : int
            Index of the class
            
        Returns:
        --------
        log_likelihood : float
            Log likelihood of the sample given the class
        """
        mean = self.means[class_idx]
        variance = self.variances[class_idx]
        
        # Calculate log likelihood to avoid numerical underflow
        # log(P(x|class)) = -0.5 × [log(2π) + log(σ²) + (x-μ)²/σ²]
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance))
        log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / variance)
        
        return log_likelihood
    
    def _calculate_multinomial_likelihood(self, x, class_idx):
        """
        Calculate likelihood P(features|class) for multinomial distribution
        
        Parameters:
        -----------
        x : numpy array of shape (n_features,)
            Single sample features (counts or frequencies)
        class_idx : int
            Index of the class
            
        Returns:
        --------
        log_likelihood : float
            Log likelihood of the sample given the class
        """
        # Calculate log likelihood: log(P(x|class)) = Σ x_i × log(p_i)
        feature_probs = self.feature_probs[class_idx]
        log_likelihood = np.sum(x * np.log(feature_probs + 1e-9))  # Add small value to avoid log(0)
        
        return log_likelihood
    
    def _predict_single(self, x):
        """
        Predict class for a single sample
        
        Uses Bayes' Theorem:
        P(class|x) ∝ P(class) × P(x|class)
        
        Parameters:
        -----------
        x : numpy array of shape (n_features,)
            Single sample to predict
            
        Returns:
        --------
        prediction : int
            Predicted class label
        """
        posteriors = []
        
        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            # Prior: log(P(class))
            prior = np.log(self.class_priors[idx])
            
            # Likelihood: log(P(x|class))
            if self.variant == 'gaussian':
                likelihood = self._calculate_gaussian_likelihood(x, idx)
            else:  # multinomial
                likelihood = self._calculate_multinomial_likelihood(x, idx)
            
            # Posterior: log(P(class|x)) = log(P(class)) + log(P(x|class))
            posterior = prior + likelihood
            posteriors.append(posterior)
        
        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        """
        Predict class labels for samples
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted class labels
        """
        predictions = []
        for x in X:
            prediction = self._predict_single(x)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples
        
        Returns the posterior probability P(class|features) for each class
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on
            
        Returns:
        --------
        probabilities : numpy array of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        probabilities = []
        
        for x in X:
            posteriors = []
            
            # Calculate posterior for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.class_priors[idx])
                
                if self.variant == 'gaussian':
                    likelihood = self._calculate_gaussian_likelihood(x, idx)
                else:
                    likelihood = self._calculate_multinomial_likelihood(x, idx)
                
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Convert log probabilities to actual probabilities
            # Use exp and normalize to get probabilities that sum to 1
            posteriors = np.array(posteriors)
            posteriors = np.exp(posteriors - np.max(posteriors))  # Subtract max for numerical stability
            posteriors = posteriors / np.sum(posteriors)
            
            probabilities.append(posteriors)
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test data
        y : numpy array of shape (n_samples,)
            True class labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score (proportion of correct predictions)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


"""
USAGE EXAMPLE 1: Simple Classification with Gaussian Naive Bayes

import numpy as np

# Sample data: Predicting fruit type based on weight (g) and diameter (cm)
X_train = np.array([
    [150, 7],   # Apple
    [170, 8],   # Apple
    [140, 6.5], # Apple
    [160, 7.5], # Apple
    [350, 9],   # Orange
    [380, 9.5], # Orange
    [340, 8.5], # Orange
    [370, 9.2], # Orange
    [20, 3],    # Cherry
    [25, 3.2],  # Cherry
    [18, 2.8],  # Cherry
    [22, 3.1]   # Cherry
])

# Labels: 0 = Apple, 1 = Orange, 2 = Cherry
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

# Create and train the model
model = NaiveBayes(variant='gaussian')
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [155, 7.2],  # Should be Apple
    [360, 9.1],  # Should be Orange
    [21, 3.0]    # Should be Cherry
])

predictions = model.predict(X_test)
print("Predicted classes:", predictions)
# Output: [0, 1, 2] (Apple, Orange, Cherry)

# Get class probabilities
probabilities = model.predict_proba(X_test)
print("\nPredicted probabilities:")
for i, probs in enumerate(probabilities):
    print(f"  Sample {i+1}: Apple={probs[0]:.4f}, Orange={probs[1]:.4f}, Cherry={probs[2]:.4f}")
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Iris Classification)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Naive Bayes model
model = NaiveBayes(variant='gaussian')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Display predictions for first 5 test samples
print("\nFirst 5 predictions:")
for i in range(5):
    print(f"  Sample {i+1}: True={data.target_names[y_test[i]]}, Predicted={data.target_names[y_pred[i]]}")

# Get class probabilities
probabilities = model.predict_proba(X_test[:5])
print("\nProbabilities for first 5 samples:")
for i, probs in enumerate(probabilities):
    print(f"  Sample {i+1}: Setosa={probs[0]:.3f}, Versicolor={probs[1]:.3f}, Virginica={probs[2]:.3f}")
"""

"""
USAGE EXAMPLE 3: Spam Detection with Multinomial Naive Bayes

import numpy as np

# Sample data: Email word frequencies (simplified)
# Features: [count of 'free', 'money', 'urgent', 'meeting', 'report']
X_train = np.array([
    [5, 4, 3, 0, 0],  # Spam
    [4, 5, 4, 0, 0],  # Spam
    [6, 3, 5, 0, 0],  # Spam
    [0, 0, 0, 4, 5],  # Not Spam
    [0, 0, 0, 5, 4],  # Not Spam
    [0, 0, 1, 3, 6],  # Not Spam
])

# Labels: 1 = Spam, 0 = Not Spam
y_train = np.array([1, 1, 1, 0, 0, 0])

# Create and train the model (multinomial for word counts)
model = NaiveBayes(variant='multinomial')
model.fit(X_train, y_train)

# Test emails
X_test = np.array([
    [3, 2, 2, 0, 0],  # Should be Spam
    [0, 0, 0, 4, 3],  # Should be Not Spam
    [1, 1, 1, 2, 2],  # Mixed features
])

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Email Classification:")
for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
    spam_prob = probs[1]
    status = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"  Email {i+1}: {status} (confidence: {max(probs):.2%})")
"""

"""
USAGE EXAMPLE 4: Medical Diagnosis

import numpy as np

# Sample data: Patient symptoms [fever_days, cough_severity, fatigue_level, body_ache]
X_train = np.array([
    [3, 7, 8, 7],    # Flu
    [4, 8, 9, 8],    # Flu
    [2, 6, 7, 6],    # Flu
    [1, 5, 3, 2],    # Cold
    [1, 6, 2, 1],    # Cold
    [2, 5, 4, 2],    # Cold
    [0, 2, 5, 1],    # Allergy
    [0, 3, 4, 0],    # Allergy
    [0, 2, 3, 1],    # Allergy
])

# Labels: 0 = Cold, 1 = Flu, 2 = Allergy
y_train = np.array([1, 1, 1, 0, 0, 0, 2, 2, 2])

# Train model
model = NaiveBayes(variant='gaussian')
model.fit(X_train, y_train)

# New patients
X_test = np.array([
    [3, 7, 8, 6],    # Likely Flu
    [1, 5, 2, 1],    # Likely Cold
    [0, 3, 4, 0],    # Likely Allergy
])

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

diagnosis_names = ['Cold', 'Flu', 'Allergy']

print("Patient Diagnosis:")
for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
    print(f"\n  Patient {i+1}:")
    print(f"    Diagnosis: {diagnosis_names[pred]}")
    print(f"    Probabilities:")
    for j, name in enumerate(diagnosis_names):
        print(f"      {name}: {probs[j]:.2%}")
"""

"""
USAGE EXAMPLE 5: Comparing with sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Our implementation
our_model = NaiveBayes(variant='gaussian')
our_model.fit(X_train, y_train)
our_predictions = our_model.predict(X_test)
our_accuracy = our_model.score(X_test, y_test)

# Sklearn implementation
sklearn_model = GaussianNB()
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

print("Performance Comparison:")
print(f"  Our Implementation:    {our_accuracy:.4f}")
print(f"  Sklearn Implementation: {sklearn_accuracy:.4f}")
print(f"  Difference:            {abs(our_accuracy - sklearn_accuracy):.4f}")

print("\nOur Model - Classification Report:")
print(classification_report(y_test, our_predictions, target_names=data.target_names))
"""

