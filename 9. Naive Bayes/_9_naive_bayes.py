import numpy as np

class NaiveBayes:
    """
    Naive Bayes Classifier Implementation from Scratch
    
    Naive Bayes is a simple, fast, and effective probabilistic classifier
    based on Bayes' Theorem with the "naive" assumption that features are
    independent of each other given the class label.
    
    Key Idea: "Calculate the probability of each class given the features,
               then predict the class with highest probability"

    Formula: P(class|features) proportional to P(class) * P(features|class)

    where:
        P(class|features) = probability of class given the features (posterior)
        P(class) = probability of class (prior)
        P(features|class) = probability of features given class (likelihood)

    Use Cases:
    - Spam / text classification: word counts -> spam vs. not spam
    - Sentiment analysis: review or tweet -> positive / negative / neutral
    - Medical triage: symptom measurements -> likely diagnosis
    - Document topic tagging: term frequencies -> sports / politics / tech
    - Real-time scoring on high-dimensional data (thousands of features, O(k*d) per prediction)

    Two variants are implemented (see the `variant` argument):
        'gaussian'    - continuous features, one (mean, variance) per feature per class
        'multinomial' - count features, one probability per feature per class
    Bernoulli Naive Bayes, a tunable Laplace alpha, user-supplied priors, sample
    weights and partial_fit are NOT implemented here. See the section
    "Simplifications vs. Canonical Naive Bayes" in _9_naive_bayes.md for the full
    list and what each omission costs you.

    Non-obvious formulas embedded in this file:

        Laplace smoothing (multinomial likelihood table, see fit):
            p(i|class) = (count_i + 1) / (total_count + n_features)
            The +1 numerator guarantees p(i|class) > 0, so a word never seen in a
            class does not veto that class with a zero probability.

        Variance smoothing (gaussian likelihood, see fit):
            epsilon = 1e-9 * max_over_features( var(X[:, j]) )
            variance = var(X_c, axis=0) + epsilon
            A feature that is constant inside one class has variance 0 and would
            divide by zero. The epsilon is scaled by the data's own variance so
            that it stays negligible whatever units the features are in. This is
            exactly scikit-learn's GaussianNB convention (var_smoothing=1e-9).

        Log-sum-exp normalization (see predict_proba):
            P(class|x) = exp(s_c - max_s) / sum_k exp(s_k - max_s)
            where s_c = log P(class) + log P(x|class). Subtracting the max before
            exponentiating prevents overflow/underflow; the evidence term P(x)
            cancels out, which is why it never has to be computed.
    """
    
    def __init__(self, variant='gaussian'):
        """
        Initialize the Naive Bayes classifier
        
        Parameters:
        -----------
        variant : str, default='gaussian'
            Which likelihood model P(features|class) to fit
            - Options: 'gaussian' or 'multinomial' (no other value is accepted)
            - 'gaussian': continuous features (measurements, sensor readings,
              scaled numeric data). Learns a mean and a variance per feature
              per class. Effect: assumes each feature is bell-shaped inside a
              class; badly skewed features hurt accuracy, so transform them first.
            - 'multinomial': non-negative count features (word counts, term
              frequencies, histogram bins). Learns one probability per feature
              per class with Laplace smoothing. Effect: negative values are
              meaningless here and will produce nonsense likelihoods.
            Typical: 'gaussian' for tabular numeric data, 'multinomial' for text.

        Raises:
        -------
        ValueError
            If `variant` is anything other than 'gaussian' or 'multinomial'.
            Without this check an unknown variant would train nothing and then
            fail later with an opaque TypeError at predict time.
        """
        if variant not in ('gaussian', 'multinomial'):
            raise ValueError(
                "variant must be 'gaussian' or 'multinomial', got {!r}".format(variant)
            )

        self.variant = variant
        self.classes = None
        self.class_priors = None
        
        # For Gaussian Naive Bayes
        self.means = None
        self.variances = None
        
        # For Multinomial Naive Bayes
        self.feature_probs = None

        # Number of features seen during fit (used to validate predict inputs)
        self.n_features = None

    def _as_2d(self, X):
        """
        Coerce X into a 2-D float array of shape (n_samples, n_features)

        Accepts plain Python lists as well as numpy arrays so that the usage
        examples in this file work whether the caller wraps their data in
        np.array(...) or not.

        A 1-D input is ambiguous, so it is resolved like this:
        - before fit (self.n_features is None): treat it as ONE feature,
          i.e. n_samples rows of a single column -> shape (len(X), 1)
        - after fit: if its length equals n_features it is a SINGLE sample
          -> shape (1, n_features); otherwise it is a single-feature column.

        Parameters:
        -----------
        X : array-like
            Data to coerce

        Returns:
        --------
        X : numpy array of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            if self.n_features is not None and X.shape[0] == self.n_features:
                X = X.reshape(1, -1)     # one sample, n_features columns
            else:
                X = X.reshape(-1, 1)     # n_samples rows, one feature
        elif X.ndim != 2:
            raise ValueError(
                "X must be 1-D or 2-D, got an array with {} dimensions".format(X.ndim)
            )

        # After fit, the feature count must match what was learned. Without this
        # check the per-feature arithmetic in the likelihood helpers would simply
        # BROADCAST a wrong-width row against the stored means and return a
        # confident but meaningless prediction instead of an error.
        if self.n_features is not None and X.shape[1] != self.n_features:
            raise ValueError(
                "X has {} features but the model was fitted on {}".format(
                    X.shape[1], self.n_features)
            )

        return X

    def _check_is_fitted(self):
        """
        Raise a clear error if the model has not been trained yet

        Without this guard, calling predict() before fit() dies with an opaque
        'NoneType is not subscriptable' deep inside a likelihood helper.
        """
        if self.classes is None:
            raise ValueError(
                "This NaiveBayes instance is not fitted yet. "
                "Call fit(X, y) with training data before using this method."
            )

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier

        Learns the prior probabilities P(class) and the likelihood
        probabilities P(features|class) from the training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data (numpy array or nested Python list)
        y : array-like of shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : NaiveBayes
            The fitted model, so calls can be chained: model.fit(X, y).predict(X)
        """
        # Forget any previous fit first, so a 1-D X is interpreted as
        # "one feature" rather than against a stale n_features.
        self.n_features = None
        X = self._as_2d(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features = n_features

        if n_samples != y.shape[0]:
            raise ValueError(
                "X has {} samples but y has {} labels".format(n_samples, y.shape[0])
            )

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

            # Variance smoothing.
            # A feature that happens to be CONSTANT inside one class has
            # variance exactly 0, and the likelihood divides by that variance
            # (see _calculate_gaussian_likelihood) -> division by zero.
            # We add a tiny floor. The floor is scaled by the largest feature
            # variance in the whole training set so that it stays negligible
            # no matter what units the features are in: a flat 1e-9 would be
            # invisible for weights in grams but enormous for a feature whose
            # own variance is 1e-12. This is exactly scikit-learn's
            # GaussianNB rule: epsilon_ = var_smoothing * max(var(X, axis=0)),
            # with var_smoothing = 1e-9.
            epsilon = 1e-9 * np.var(X, axis=0).max()

            for idx, c in enumerate(self.classes):
                # Get all samples belonging to class c
                X_c = X[y == c]

                # Calculate mean and variance for each feature
                self.means[idx, :] = np.mean(X_c, axis=0)
                self.variances[idx, :] = np.var(X_c, axis=0) + epsilon

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

        return self

    def _calculate_gaussian_likelihood(self, x, class_idx):
        """
        Calculate likelihood P(features|class) using Gaussian distribution

        The naive independence assumption says the features are conditionally
        independent given the class, so the joint likelihood is a PRODUCT of
        per-feature densities, and its logarithm is a SUM:

            P(x|class)      = prod_j  P(x_j|class)
            log P(x|class)  = sum_j   log P(x_j|class)

        For each feature, calculate probability density using:
        P(x_j|class) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x_j - mu)^2 / (2*sigma^2))

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
        
        # Calculate log likelihood to avoid numerical underflow.
        # Taking log of the Gaussian density and summing over features j:
        #   log(P(x|class)) = sum_j -0.5 * [log(2*pi*sigma_j^2) + (x_j - mu_j)^2 / sigma_j^2]
        # which splits into a constant term (line 1) and a distance term (line 2).
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
        # Calculate log likelihood: log(P(x|class)) = sum_i  x_i * log(p_i)
        # No epsilon is needed inside the log: the Laplace smoothing applied in
        # fit, (count_i + 1) / (total + n_features), has a numerator of at least
        # 1, so every p_i is strictly positive and log(p_i) is always finite.
        # (Adding an epsilon here would quietly bias every log-likelihood.)
        feature_probs = self.feature_probs[class_idx]
        log_likelihood = np.sum(x * np.log(feature_probs))

        return log_likelihood
    
    def _predict_single(self, x):
        """
        Predict class for a single sample
        
        Uses Bayes' Theorem:
        P(class|x) proportional to P(class) * P(x|class)

        The evidence term P(x) from Bayes' theorem is deliberately dropped.
        It is the SAME number for every class, so dividing all the scores by
        it cannot change which score is largest, and argmax is all we need
        here. (predict_proba does need it, and recovers it for free by
        normalizing the scores to sum to 1 - see that method.)

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
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on (numpy array or nested Python list).
            A 1-D array of length n_features is treated as a single sample.

        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted class labels
        """
        self._check_is_fitted()
        X = self._as_2d(X)

        predictions = []
        for x in X:
            prediction = self._predict_single(x)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples
        
        Returns the posterior probability P(class|features) for each class

        This is where the dropped evidence term P(x) comes back. Bayes' theorem
        says P(class|x) = P(class) * P(x|class) / P(x), and P(x) is just the sum
        of the numerators over all classes. So instead of computing P(x)
        separately we normalize the per-class scores to sum to 1 - the
        log-sum-exp step at the bottom of this method does exactly that.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on (numpy array or nested Python list).
            A 1-D array of length n_features is treated as a single sample.

        Returns:
        --------
        probabilities : numpy array of shape (n_samples, n_classes)
            Predicted probabilities for each class; each row sums to 1.0
        """
        self._check_is_fitted()
        X = self._as_2d(X)

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
        
        This is a CLASSIFIER, so score() returns accuracy (not R^2):
            accuracy = (number of correct predictions) / (total predictions)

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test data
        y : array-like of shape (n_samples,)
            True class labels

        Returns:
        --------
        accuracy : float
            Accuracy score between 0.0 and 1.0
        """
        y = np.asarray(y)

        if y.shape[0] == 0:
            raise ValueError("Cannot score on an empty dataset (y has 0 samples).")

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return float(accuracy)


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
# Output: [0 1 2]   (Apple, Orange, Cherry)

# Get class probabilities
probabilities = model.predict_proba(X_test)
print("\nPredicted probabilities:")
for i, probs in enumerate(probabilities):
    print(f"  Sample {i+1}: Apple={probs[0]:.4f}, Orange={probs[1]:.4f}, Cherry={probs[2]:.4f}")
# Output:
#   Sample 1: Apple=1.0000, Orange=0.0000, Cherry=0.0000
#   Sample 2: Apple=0.0000, Orange=1.0000, Cherry=0.0000
#   Sample 3: Apple=0.0000, Orange=0.0000, Cherry=1.0000
# The classes are so far apart that the losing probabilities are ~1e-43 or
# smaller (some underflow to exactly 0.0 once exponentiated), not the 0.0000
# that .4f formatting shows. This is precisely why the implementation adds
# log-probabilities instead of multiplying probabilities.
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
# Output:
#   Our Implementation:    0.9737
#   Sklearn Implementation: 0.9737
#   Difference:            0.0000
#
# The difference is exactly 0 because fit() uses sklearn's variance-smoothing
# convention, epsilon = 1e-9 * max(var(X, axis=0)), instead of a flat 1e-9.
# On this raw (unscaled) dataset sklearn's epsilon_ is 3.2154e-04, which is
# ~1e5 times larger than a flat 1e-9 and nearly doubles the smallest per-class
# variance - enough to flip one of the 114 test rows. Matching the convention
# makes the two models agree on 114/114 predictions, with predict_proba
# agreeing to ~9e-16.

# Direct parameter comparison: the learned statistics should be identical
print("\nLearned parameter agreement with sklearn:")
print(f"  means      max|diff|: {np.abs(our_model.means - sklearn_model.theta_).max():.3e}")
print(f"  variances  max|diff|: {np.abs(our_model.variances - sklearn_model.var_).max():.3e}")
print(f"  priors     max|diff|: {np.abs(our_model.class_priors - sklearn_model.class_prior_).max():.3e}")
# Output: all three print 0.000e+00

print("\nOur Model - Classification Report:")
print(classification_report(y_test, our_predictions, target_names=data.target_names))
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _9_naive_bayes.py
    # numpy only - no sklearn, no matplotlib, no plotting.
    # ----------------------------------------------------------------
    np.random.seed(42)

    print("=" * 55)
    print("NAIVE BAYES FROM SCRATCH - PLUG AND PLAY DEMO")
    print("=" * 55)

    # --- Demo 1: Gaussian NB on three separated Gaussian blobs ---
    print("\n" + "=" * 55)
    print("DEMO 1 - Gaussian NB: 3 blobs in 3-D continuous space")
    print("=" * 55)
    print("Continuous features -> each class is modelled as one")
    print("Gaussian per feature, i.e. a (mean, variance) pair.")

    X0 = np.random.randn(100, 3) + np.array([0, 0, 0])
    X1 = np.random.randn(100, 3) + np.array([3, 3, -3])
    X2 = np.random.randn(100, 3) + np.array([-3, 2, 2])
    X_g = np.vstack([X0, X1, X2])
    y_g = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # Shuffle BEFORE slicing: the rows above are grouped by class, so a raw
    # X[:220] split would put almost all of class 2 in the test set only.
    idx = np.random.permutation(300)
    X_g, y_g = X_g[idx], y_g[idx]
    X_tr, X_te = X_g[:220], X_g[220:]
    y_tr, y_te = y_g[:220], y_g[220:]

    gnb = NaiveBayes(variant='gaussian')
    gnb.fit(X_tr, y_tr)

    print("\nWhat fit() learned:")
    print("  class priors P(class) : " +
          "  ".join("{:.3f}".format(p) for p in gnb.class_priors))
    for i, c in enumerate(gnb.classes):
        print("  class {} mean vector   : ".format(c) +
              "  ".join("{:6.2f}".format(v) for v in gnb.means[i]))

    print("\nTrain Accuracy : {:.2%}".format(gnb.score(X_tr, y_tr)))
    print("Test  Accuracy : {:.2%}".format(gnb.score(X_te, y_te)))

    proba_g = gnb.predict_proba(X_te)
    pred_g = gnb.predict(X_te)
    print("\nSample predictions (first 5 test rows):")
    for i in range(5):
        print("  true={}  pred={}   P(0)={:.3f}  P(1)={:.3f}  P(2)={:.3f}".format(
            y_te[i], pred_g[i], proba_g[i, 0], proba_g[i, 1], proba_g[i, 2]))

    # --- Demo 2: Multinomial NB on planted document topics ---
    print("\n" + "=" * 55)
    print("DEMO 2 - Multinomial NB: 3 planted document topics")
    print("=" * 55)
    print("Count features -> each class is modelled as one")
    print("probability per word, smoothed by Laplace's +1 rule.")

    # Known-answer setup: invent 3 topic word-distributions over a
    # 40-word vocabulary, then sample 40-word documents from them.
    vocab_size = 40
    topic_word = np.random.dirichlet(np.ones(vocab_size) * 0.3, 3)
    doc_topic = np.random.randint(0, 3, 300)
    X_m = np.array([np.random.multinomial(40, topic_word[t]) for t in doc_topic])
    y_m = doc_topic

    idx_m = np.random.permutation(300)
    X_m, y_m = X_m[idx_m], y_m[idx_m]
    X_tr2, X_te2 = X_m[:220], X_m[220:]
    y_tr2, y_te2 = y_m[:220], y_m[220:]

    mnb = NaiveBayes(variant='multinomial')
    mnb.fit(X_tr2, y_tr2)

    print("\nTrain Accuracy : {:.2%}".format(mnb.score(X_tr2, y_tr2)))
    print("Test  Accuracy : {:.2%}".format(mnb.score(X_te2, y_te2)))

    print("\nTop 3 words per topic from the learned likelihood table:")
    for i, c in enumerate(mnb.classes):
        top = np.argsort(mnb.feature_probs[i])[::-1][:3]
        print("  topic {} -> word ids {}  with p = {}".format(
            c, [int(j) for j in top],
            "  ".join("{:.3f}".format(mnb.feature_probs[i, j]) for j in top)))

    proba_m = mnb.predict_proba(X_te2)
    pred_m = mnb.predict(X_te2)
    print("\nSample predictions (first 5 test documents):")
    for i in range(5):
        print("  true topic={}  predicted={}  confidence={:.2%}".format(
            y_te2[i], pred_m[i], np.max(proba_m[i])))

    print("\n" + "=" * 55)
    print("Done. Both variants use the same Bayes rule:")
    print("  score(class) = log P(class) + log P(x|class)")
    print("and differ only in how log P(x|class) is computed.")
    print("=" * 55)

