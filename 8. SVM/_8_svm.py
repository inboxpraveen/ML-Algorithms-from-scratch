import numpy as np

class SupportVectorMachine:
    """
    Support Vector Machine (SVM) Implementation from Scratch
    
    SVM is a powerful algorithm for classification that finds the optimal
    hyperplane that maximally separates different classes in the feature space.
    
    Key Idea: "Find the widest street that separates the two classes"

    Use Cases:
    - Text classification: spam filtering, sentiment, topic tagging (many sparse features)
    - Medical diagnosis: benign vs malignant from a panel of tabular test results
    - Image and handwriting recognition: digit / character classification
    - Credit approval and fraud screening: approve vs decline decisions
    - Bioinformatics: cancer-type classification from gene-expression profiles

    The decision boundary is:
        f(x) = w.x + b

    Classification rule:
        y = +1 if w.x + b >= 0
        y = -1 if w.x + b < 0

    where:
        w = weight vector (perpendicular to decision boundary)
        b = bias term (position of decision boundary)
        x = input features

    Objective minimised by fit() (primal soft-margin SVM):
        L(w, b) = lambda*||w||^2 + (1/n) * sum_i max(0, 1 - y_i * (w.x_i + b))
                  |_____________|   |_____________________________________|
                    wide margin              hinge loss (few errors)

    The street between the two margin lines w.x + b = +1 and w.x + b = -1 has
    width 2/||w||, so shrinking ||w|| is exactly what widens the street.

    Support vectors are the points with y_i * (w.x_i + b) <= 1: the only points
    whose hinge term is still active. get_support_vectors() returns them, and
    fit() stores their indices in self.support_vector_indices_.

    Labels: fit() accepts either 0/1 or -1/+1 labels, but predict() ALWAYS
    returns -1/+1. Evaluate with score(), or convert the predictions yourself
    with np.where(pred == -1, 0, 1).

    Simplification: this is the *primal* linear SVM trained by sub-gradient
    descent. The dual formulation, SMO, and kernels (RBF, polynomial) are not
    implemented here - see "Simplification vs. Canonical SVM" in _8_svm.md.
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000):
        """
        Initialize the Support Vector Machine model
        
        Parameters:
        -----------
        learning_rate : float, default=0.001
            Step size for each sub-gradient descent update
            - Range: 0.0001 to 0.01
            - Higher values learn faster but bounce around the optimum
            - Lower values are stable but need more iterations to get there
            Typical: 0.001 on standardized features

        lambda_param : float, default=0.01
            L2 regularization strength; sets the margin-vs-error tradeoff
            - Range: 0.0001 to 1.0
            - Higher values = wider margin, simpler model, tolerates errors
            - Lower values = narrower margin, fits the training data harder
            Typical: 0.01. Equivalent to scikit-learn's C = 1 / (2 * lambda * n)

        iterations : int, default=1000
            Number of training epochs. One "iteration" here is a full pass over
            all n samples, doing one weight update per sample (n updates), so
            the total work is n * iterations updates.
            - Range: 100 to 5000
            - More iterations = better convergence, but longer training
            Typical: 1000
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []  # Track loss history
        self.classes_ = None  # The two original labels seen by fit()
        self.support_vector_indices_ = None  # Filled in at the end of fit()
    
    def _compute_loss(self, X, y):
        """
        Compute hinge loss with L2 regularization
        
        Hinge Loss: max(0, 1 - y * (w.x + b))
        Total Loss: lambda*||w||^2 + (1/n) * sum_i max(0, 1 - y_i * (w.x_i + b))
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Target labels (-1 or +1)
            
        Returns:
        --------
        loss : float
            Total loss (regularization + hinge loss)
        """
        n_samples = X.shape[0]
        
        # Calculate distances from hyperplane
        distances = 1 - y * (X @ self.weights + self.bias)
        
        # Hinge loss: max(0, distance)
        hinge_loss = np.maximum(0, distances)
        
        # Total loss: regularization + mean hinge loss
        loss = self.lambda_param * np.dot(self.weights, self.weights) + np.mean(hinge_loss)
        
        return loss
    
    def fit(self, X, y):
        """
        Train the SVM model using sub-gradient descent

        The SVM optimization problem:
        Minimize: lambda*||w||^2 + (1/n) * sum_i max(0, 1 - y_i * (w.x_i + b))

        Gradient when y_i * (w.x_i + b) < 1 (misclassified or within margin):
            dL/dw = 2*lambda*w - y_i * x_i
            dL/db = -y_i

        Gradient when y_i * (w.x_i + b) >= 1 (correctly classified outside margin):
            dL/dw = 2*lambda*w
            dL/db = 0

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data (a plain nested list works too)
        y : numpy array of shape (n_samples,)
            Target labels (should be -1 or +1)
            If labels are 0 and 1, they will be converted to -1 and +1

        Notes:
        ------
        self.losses is CLEARED at the start of every call, so re-fitting the
        same object gives a clean curve. It ends up with iterations + 1 entries:
        losses[0] is the loss before any update (always exactly 1.0, because w
        and b start at zero) and losses[t] is the loss AFTER epoch t, so
        losses[-1] really is the final training loss.
        """
        # Accept plain Python lists as well as arrays, and force float maths
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Convert labels to -1 and +1 if they are 0 and 1
        y_labels = np.where(y <= 0, -1, 1)

        # Remember the original encoding (predict() still emits -1/+1)
        self.classes_ = np.unique(y)
        if np.unique(y_labels).size < 2:
            raise ValueError(
                "fit() needs both classes to be present, but y contains only "
                "one class: %s" % (np.unique(y),)
            )

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Start a fresh loss history; entry 0 is the loss before any update
        self.losses = [self._compute_loss(X, y_labels)]

        # Sub-gradient descent optimization
        for iteration in range(self.iterations):
            # For each sample, compute gradient
            for idx, x_i in enumerate(X):
                # Check if sample is misclassified or within margin
                condition = y_labels[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
                if condition:
                    # Correctly classified outside margin
                    # Only regularization gradient
                    dw = 2 * self.lambda_param * self.weights
                    db = 0
                else:
                    # Misclassified or within margin
                    # Regularization + hinge loss gradient
                    dw = 2 * self.lambda_param * self.weights - y_labels[idx] * x_i
                    db = -y_labels[idx]
                
                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Record the loss AFTER this epoch's updates, so losses[-1] is
            # the loss of the model you actually end up with
            self.losses.append(self._compute_loss(X, y_labels))

        # Identify the support vectors: the points sitting on or inside the
        # margin, y_i * (w.x_i + b) <= 1. They are the only points with an
        # active hinge term, i.e. the only ones still shaping the boundary.
        margins = y_labels * (X @ self.weights + self.bias)
        self.support_vector_indices_ = np.where(margins <= 1 + 1e-3)[0]

    def predict(self, X):
        """
        Predict class labels for samples
        
        Decision rule: sign(w.x + b)
        Returns +1 if w.x + b >= 0, else -1

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on. A single 1-D sample of shape
            (n_features,) is accepted as well.

        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted class labels, ALWAYS -1.0 or +1.0 - even when fit() was
            given 0/1 labels. Comparing these directly against 0/1 targets
            would score about 50%; use score(X, y) to evaluate, or convert
            with np.where(predictions == -1, 0, 1).
        """
        # Calculate the decision function, then take its sign.
        # np.where also settles the exact-zero case: f(x) == 0 -> +1.
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def predict_proba(self, X):
        """
        Heuristic class confidences obtained by squashing the decision function

        A linear SVM has no probability model - it only knows how far a point
        lies from the boundary. We push that signed distance through a logistic
        sigmoid so it can be read as a confidence score:

            p(+1 | x) = 1 / (1 + exp(-f(x)))    where f(x) = w.x + b

        WARNING: these are NOT calibrated probabilities. Proper SVM probability
        estimates need Platt scaling (fitting a logistic regression on f(x) with
        cross-validation), which is not implemented here. Use these values to
        rank confidence, not as true likelihoods.

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to score

        Returns:
        --------
        proba : numpy array of shape (n_samples, 2)
            Column 0 = confidence for class -1, column 1 = confidence for
            class +1. Each row sums to 1.
        """
        f = self.decision_function(X)

        # Numerically stable sigmoid: exp(-|f|) can never overflow
        exp_neg = np.exp(-np.abs(f))
        p_positive = np.where(f >= 0, 1.0 / (1.0 + exp_neg), exp_neg / (1.0 + exp_neg))

        return np.column_stack([1.0 - p_positive, p_positive])

    def decision_function(self, X):
        """
        Calculate the distance of samples from the decision boundary

        Distance = w.x + b

        Positive values = predicted as class +1
        Negative values = predicted as class -1
        Magnitude = confidence (larger absolute value = more confident)
        (This is the distance in units of 1/||w||; multiply by 1/||w|| for the
        true geometric distance in feature space.)

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to calculate distances for. A single 1-D sample of shape
            (n_features,) is accepted as well.

        Returns:
        --------
        distances : numpy array of shape (n_samples,)
            Signed distances from decision boundary
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        # atleast_2d lets a single 1-D sample through as one row
        X = np.atleast_2d(np.asarray(X, dtype=float))

        return X @ self.weights + self.bias

    def get_support_vectors(self, X, y, tol=1e-3):
        """
        Find the support vectors - the points that hold up the boundary

        A point is a support vector when it lies on or inside the margin:

            y_i * (w.x_i + b) <= 1 + tol

        Those are exactly the points whose hinge loss is still active, which
        makes them the only points that can still push w and b around. Every
        other training point could be deleted without moving the boundary at
        all - that is the sense in which these points "support" it.

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            The data the model was trained on
        y : numpy array of shape (n_samples,)
            Its labels (-1/+1 or 0/1)
        tol : float, default=1e-3
            Slack on the margin test. Sub-gradient descent never lands exactly
            on y * f(x) == 1, so a small tolerance is needed.

        Returns:
        --------
        indices : numpy array of ints
            Row indices of X that are support vectors. fit() stores the same
            thing for its own training set in self.support_vector_indices_.
        """
        y_labels = np.where(np.asarray(y) <= 0, -1, 1)

        # decision_function() also raises a clear error if the model is unfitted
        margins = y_labels * self.decision_function(X)

        return np.where(margins <= 1 + tol)[0]

    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test data
        y : numpy array of shape (n_samples,)
            True labels (can be -1/+1 or 0/1)
            
        Returns:
        --------
        accuracy : float
            Proportion of correct predictions
        """
        # Convert labels if necessary
        y_labels = np.where(y <= 0, -1, 1)
        
        # Get predictions
        predictions = self.predict(X)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_labels)
        
        return accuracy
    
    def get_params(self):
        """
        Get the model parameters
        
        Returns:
        --------
        dict : Dictionary containing weights and bias
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'norm_w': np.linalg.norm(self.weights)  # Magnitude of weight vector
        }


"""
USAGE EXAMPLE 1: Simple Binary Classification

import numpy as np

# Sample data: Classifying fruits as Apple (+1) or Orange (-1)
# Features: [weight (grams), sweetness (1-10)]
# NOTE: these features are deliberately left UNSCALED so you can see the raw
# numbers. It only works because the 8 points are wildly separated - on real
# data always standardize first (see "Feature Scaling: Critical for SVM").
X_train = np.array([
    [150, 8],   # Apple
    [170, 9],   # Apple
    [140, 7],   # Apple
    [160, 8],   # Apple
    [350, 4],   # Orange
    [380, 5],   # Orange
    [340, 3],   # Orange
    [360, 4]    # Orange
])

# Labels: +1 = Apple, -1 = Orange
y_train = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# Create and train the model
model = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [155, 8],   # Should be Apple (+1)
    [360, 4],   # Should be Orange (-1)
    [250, 6]    # Boundary case
])

predictions = model.predict(X_test)
print("Predicted classes:", predictions)
# Output: [ 1. -1. -1.]  (Apple, Orange, Orange - the boundary case [250, 6]
# falls just on the Orange side). The fit is deterministic: w and b start at
# zero and nothing is random, so you get these exact numbers every run.

# Get decision function values (distances from boundary)
distances = model.decision_function(X_test)
print("\nDistances from decision boundary:", distances)
# Output: [ 1.85937417 -3.97680952 -0.92776644]
# Positive = Apple, Negative = Orange, Magnitude = Confidence

# Get model parameters
params = model.get_params()
print(f"\nWeights: {params['weights']}")
print(f"Bias: {params['bias']:.4f}")
print(f"Weight norm: {params['norm_w']:.4f}")
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Breast Cancer - Binary Classification)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CRITICAL: Standardize features (very important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
model = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, iterations=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Training accuracy
train_accuracy = model.score(X_train_scaled, y_train)
print(f"Train Accuracy: {train_accuracy:.4f}")

# Display predictions for first 5 test samples
print("\nFirst 5 predictions:")
distances = model.decision_function(X_test_scaled[:5])
for i in range(5):
    true_label = "Malignant" if y_test[i] == 0 else "Benign"
    pred_label = "Malignant" if y_pred[i] == -1 else "Benign"
    confidence = abs(distances[i])
    print(f"  Sample {i+1}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.4f}")
"""

"""
USAGE EXAMPLE 3: Visualizing Decision Boundary and Margins

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic linearly separable data
np.random.seed(42)
n_samples = 100

# Class 1 (centered at [2, 2])
X_class1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([2, 2])
y_class1 = np.ones(n_samples // 2)

# Class -1 (centered at [4, 4])
X_class_neg1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([4, 4])
y_class_neg1 = -np.ones(n_samples // 2)

# Combine data
X_train = np.vstack([X_class1, X_class_neg1])
y_train = np.hstack([y_class1, y_class_neg1])

# Train model
model = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, iterations=1000)
model.fit(X_train, y_train)

# Plot results
plt.figure(figsize=(12, 5))

# Plot 1: Training loss
plt.subplot(1, 2, 1)
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)

# Plot 2: Decision boundary with margins
plt.subplot(1, 2, 2)

# Create mesh for decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Calculate decision function for all points
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary (Z=0) and margins (Z=-1, Z=+1)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
            linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

# Plot data points
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], 
            c='red', marker='o', s=100, label='Class +1', edgecolors='k')
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1],
            c='blue', marker='s', s=100, label='Class -1', edgecolors='k')

# Circle the support vectors: the points on or inside the margin
sv = model.get_support_vectors(X_train, y_train)
plt.scatter(X_train[sv][:, 0], X_train[sv][:, 1],
            s=250, facecolors='none', edgecolors='green', linewidths=2,
            label=f'Support Vectors ({len(sv)})')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'SVM Decision Boundary and Margins\nAccuracy: {model.score(X_train, y_train):.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal training accuracy: {model.score(X_train, y_train):.4f}")
print(f"Initial loss (before any update): {model.losses[0]:.4f}")
print(f"Final loss: {model.losses[-1]:.4f}")
print(f"Support vectors: {len(model.support_vector_indices_)} of {len(X_train)} points")
print(f"\nModel parameters:")
params = model.get_params()
print(f"  Weights: {params['weights']}")
print(f"  Bias: {params['bias']:.4f}")
print(f"  Weight norm: {params['norm_w']:.4f}")
"""

"""
USAGE EXAMPLE 4: Comparing Different Regularization Parameters

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic classification dataset
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, 
                           class_sep=1.5, random_state=42)

# Convert labels to -1 and +1
y = np.where(y == 0, -1, 1)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different lambda values (regularization)
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1.0]

print("Comparing Different Regularization Parameters (lambda):\n")
print(f"{'Lambda':<15} {'Train Accuracy':<20} {'Test Accuracy':<20} {'Final Loss':<15}")
print("-" * 70)

for lambda_param in lambda_values:
    model = SupportVectorMachine(learning_rate=0.001, lambda_param=lambda_param, iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    final_loss = model.losses[-1]
    
    print(f"{lambda_param:<15.4f} {train_acc:<20.4f} {test_acc:<20.4f} {final_loss:<15.4f}")

print("\nObservations:")
print("- Small lambda (0.0001-0.001): Narrow margin, may overfit")
print("- Medium lambda (0.01-0.1): Balanced, good generalization")
print("- Large lambda (1.0+): Wide margin, may underfit")
"""

"""
USAGE EXAMPLE 5: Iris Dataset (Convert to Binary Classification)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load iris dataset
data = load_iris()
X, y = data.data, data.target

# Convert to binary classification: Setosa (0) vs Versicolor (1)
# Filter out Virginica (class 2)
mask = y != 2
X_binary = X[mask]
y_binary = y[mask]

# Convert labels to -1 and +1
y_binary = np.where(y_binary == 0, -1, 1)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
model = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, iterations=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Show predictions with confidence
y_pred = model.predict(X_test_scaled)
distances = model.decision_function(X_test_scaled)

print("\nPredictions with Confidence:")
class_names = ['Setosa', 'Versicolor']
for i in range(min(10, len(y_test))):
    true_label = class_names[0] if y_test[i] == -1 else class_names[1]
    pred_label = class_names[0] if y_pred[i] == -1 else class_names[1]
    confidence = abs(distances[i])
    status = "OK" if y_pred[i] == y_test[i] else "XX"
    print(f"  {status} True: {true_label:12s} | Predicted: {pred_label:12s} | Confidence: {confidence:.4f}")

# Display model parameters
params = model.get_params()
print(f"\nModel Parameters:")
print(f"  Number of features: {len(params['weights'])}")
print(f"  Bias: {params['bias']:.4f}")
print(f"  Weight vector norm: {params['norm_w']:.4f}")
print(f"\nFeature importance (absolute weights):")
for i, weight in enumerate(params['weights']):
    print(f"  {data.feature_names[i]:20s}: {abs(weight):8.4f}")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _8_svm.py
    # Requires numpy only - no sklearn, no matplotlib, no plotting.
    # ----------------------------------------------------------------
    np.random.seed(42)

    print("=" * 62)
    print("SUPPORT VECTOR MACHINE FROM SCRATCH - PLUG-AND-PLAY DEMO")
    print("=" * 62)

    # ---- Build one binary problem: two overlapping Gaussian blobs ----
    X_neg = np.random.randn(100, 2) + np.array([-1.5, -1.5])   # class -1
    X_pos = np.random.randn(100, 2) + np.array([1.5, 1.5])     # class +1
    X_all = np.vstack([X_neg, X_pos])
    y_all = np.array([-1] * 100 + [1] * 100)

    # Shuffle BEFORE slicing - otherwise the "test set" would be one class only
    shuffle_idx = np.random.permutation(200)
    X_all, y_all = X_all[shuffle_idx], y_all[shuffle_idx]

    X_train_raw, X_test_raw = X_all[:140], X_all[140:]
    y_train, y_test = y_all[:140], y_all[140:]

    # Standardize with TRAIN statistics only (never peek at the test set)
    mu, sigma = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    X_train = (X_train_raw - mu) / sigma
    X_test = (X_test_raw - mu) / sigma

    # ---------------------------------------------------------------
    # DEMO 1: fit the maximum-margin boundary and read it back
    # ---------------------------------------------------------------
    print("\n" + "=" * 62)
    print("DEMO 1 - Binary classification of two Gaussian blobs")
    print("=" * 62)
    print("140 train / 60 test points, 2 standardized features.")

    model = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01,
                                 iterations=500)
    model.fit(X_train, y_train)

    print(f"\nTrain Accuracy : {model.score(X_train, y_train):.4f}")
    print(f"Test  Accuracy : {model.score(X_test, y_test):.4f}")

    params = model.get_params()
    print(f"\nw              : [{params['weights'][0]:+.4f}, "
          f"{params['weights'][1]:+.4f}]")
    print(f"b              : {params['bias']:+.4f}")
    print(f"||w||          : {params['norm_w']:.4f}")
    print(f"Margin 2/||w|| : {2.0 / params['norm_w']:.4f}")
    print(f"Loss           : {model.losses[0]:.4f} -> {model.losses[-1]:.4f}")
    print(f"Support vectors: {len(model.support_vector_indices_)} of "
          f"{len(X_train)} training points hold up the boundary")

    print("\nSample test predictions (f(x) = w.x + b, conf = |f(x)|):")
    distances = model.decision_function(X_test)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    for i in range(5):
        print(f"  true={y_test[i]:+d}  pred={int(predictions[i]):+d}  "
              f"f(x)={distances[i]:+.3f}  conf={abs(distances[i]):.3f}  "
              f"p(+1)={probabilities[i, 1]:.3f}")

    # ---------------------------------------------------------------
    # DEMO 2: lambda is the margin dial - bigger lambda, wider street
    # ---------------------------------------------------------------
    print("\n" + "=" * 62)
    print("DEMO 2 - Regularization: larger lambda -> wider margin")
    print("=" * 62)
    print("Same split, 300 iterations each. Watch 2/||w|| grow with lambda.\n")

    print(f"{'lambda':>8}  {'train acc':>9}  {'test acc':>8}  "
          f"{'2/||w||':>8}  {'#SV':>4}")
    print("-" * 46)
    for lam in [0.001, 0.01, 0.1]:
        m = SupportVectorMachine(learning_rate=0.001, lambda_param=lam,
                                 iterations=300)
        m.fit(X_train, y_train)
        margin = 2.0 / np.linalg.norm(m.weights)
        print(f"{lam:>8.3f}  {m.score(X_train, y_train):>9.4f}  "
              f"{m.score(X_test, y_test):>8.4f}  {margin:>8.4f}  "
              f"{len(m.support_vector_indices_):>4d}")

    print("\nA wider margin normally trades training fit for generalization,")
    print("and it always pulls more points inside the margin (#SV grows).")
    print("On this easy, well-separated data the widest margin happens to win")
    print("on both counts; the tradeoff bites on noisier, overlapping data.")

    # ---------------------------------------------------------------
    # DEMO 3: why feature scaling is not optional for SVM
    # ---------------------------------------------------------------
    print("\n" + "=" * 62)
    print("DEMO 3 - Feature scaling is mandatory for SVM")
    print("=" * 62)
    print("Blow up feature 2 by 300x, then fit with and without scaling.\n")

    # Same data, but feature 2 is now measured in a much larger unit
    X_train_big = X_train_raw * np.array([1.0, 300.0])
    X_test_big = X_test_raw * np.array([1.0, 300.0])

    unscaled = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01,
                                    iterations=300)
    unscaled.fit(X_train_big, y_train)

    mu_big, sigma_big = X_train_big.mean(axis=0), X_train_big.std(axis=0)
    scaled = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01,
                                  iterations=300)
    scaled.fit((X_train_big - mu_big) / sigma_big, y_train)

    print(f"{'version':>12}  {'final loss':>10}  {'test acc':>8}")
    print("-" * 34)
    print(f"{'unscaled':>12}  {unscaled.losses[-1]:>10.4f}  "
          f"{unscaled.score(X_test_big, y_test):>8.4f}")
    print(f"{'standardized':>12}  {scaled.losses[-1]:>10.4f}  "
          f"{scaled.score((X_test_big - mu_big) / sigma_big, y_test):>8.4f}")

    print("\nThe accuracy gap is small here, but look at the loss: the")
    print("unscaled fit is still far from the optimum after the same number")
    print("of epochs, because the huge feature dominates every gradient step.")
    print("\n" + "=" * 62)
    print("Demo complete. Try editing lambda_param or iterations above.")
    print("=" * 62)
