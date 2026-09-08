import numpy as np

class LogisticRegression:
    """
    Logistic Regression Implementation from Scratch
    
    Logistic regression is used for binary classification problems where
    we want to predict if something belongs to one class or another (0 or 1).
    
    Key Idea: "Fit a straight line in log-odds space, then squash it through a
    sigmoid so the output is always a probability between 0 and 1"

    Use Cases:
    - Medical diagnosis: disease present vs absent from symptoms and lab tests
    - Email spam detection: spam vs not spam from word frequencies
    - Credit risk scoring: will default vs will repay
    - Customer churn prediction: will leave vs will stay
    - Fraud detection: fraudulent vs legitimate transaction

    Formula: p(y=1|x) = 1 / (1 + e^(-(b0 + b1*x1 + b2*x2 + ... + bn*xn)))
    
    where:
        p(y=1|x) = probability that y equals 1 given x
        e = Euler's number (~= 2.718)
        b0 = intercept (bias term)
        b1, b2, ..., bn = coefficients for each feature
        x1, x2, ..., xn = independent variables (features)

    Log-odds (logit) view - why the coefficients are interpretable:
        log(p / (1 - p)) = b0 + b1*x1 + ... + bn*xn
        The quantity z = X @ theta that the code computes IS the log-odds.
        A one-unit increase in x_j adds b_j to the log-odds, which multiplies
        the odds by exp(b_j) - the "odds ratio" for that feature.

    Training - gradient of the mean binary cross-entropy w.r.t. theta:
        grad = (1/n) * X^T @ (sigmoid(X @ theta) - y)
        The sigmoid derivative sigmoid(z)*(1 - sigmoid(z)) produced by the chain
        rule cancels exactly against the log terms of the loss, so this collapses
        to the same (1/n) * X^T @ error form as linear regression. That
        cancellation is the mathematical payoff of pairing sigmoid with
        cross-entropy (see fit() and the .md derivation).

    Optional L2 regularization (reg_lambda > 0) adds (reg_lambda/n) * theta_j to
    the gradient of every coefficient except the intercept. This is what keeps
    the weights finite on perfectly separable data, where the unregularized
    maximum-likelihood solution has no finite optimum.
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000, fit_intercept=True,
                 reg_lambda=0.0):
        """
        Initialize the Logistic Regression model
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Step size for gradient descent optimization
            - Range: 0.001 to 1.0
            - Larger values = faster convergence but risk overshooting
            - Smaller values = slower but more stable convergence
            Typical: 0.01 on raw features, 0.1 to 0.5 once features are
            standardized (scaling is what lets you take the bigger steps)
        
        iterations : int, default=1000
            Number of iterations for gradient descent
            - More iterations = better convergence (but longer training)
            - There is no early stopping: every iteration always runs
            Typical: 500 to 5000. Watch model.losses and stop increasing this
            once the curve plateaus.
        
        fit_intercept : bool, default=True
            Whether to prepend a column of ones so the model learns a bias term
            - True: the decision boundary can sit anywhere
            - False: the boundary is forced through the origin
            Set False only when the data is already centered or a bias column
            is already present in X.

        reg_lambda : float, default=0.0
            L2 (ridge) regularization strength applied to the feature
            coefficients. The intercept is never penalized.
            - Range: 0.0 to ~100
            - 0.0 = plain (unregularized) maximum-likelihood fit
            - Higher values shrink coefficients toward zero, trading a little
              training fit for stability and better generalization
            Typical: 0.0 or 1.0. Equivalent to scikit-learn's C through
            reg_lambda = 1 / C, so sklearn's default C=1.0 corresponds to
            reg_lambda=1.0 here.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.reg_lambda = reg_lambda
        self.coefficients = None
        self.intercept = None
        self.feature_coefficients = None
        self.losses = []  # Track loss history
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function
        
        Maps any real number to a value between 0 and 1
        Formula: sigmoid(z) = 1 / (1 + e^(-z))
        
        Parameters:
        -----------
        z : numpy array
            Linear combination of features and coefficients (the log-odds)
            
        Returns:
        --------
        sigmoid : numpy array
            Values between 0 and 1 (probabilities)

        Numerical notes:
        ----------------
        The clip only matters on the negative side: np.exp overflows past
        z = -709, so without it a very negative z gives inf and then nan.
        On the positive side float64 runs out of resolution long before the
        clip bites - sigmoid(z) already returns exactly 1.0 for z > ~37. That
        saturation is harmless here because _compute_loss clips probabilities
        to [1e-15, 1 - 1e-15] before taking any logarithm.
        """
        # Clip values to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss
        
        Loss = -1/n * sum[y*log(p) + (1-y)*log(1-p)]
        
        Parameters:
        -----------
        y_true : numpy array
            True labels (0 or 1)
        y_pred : numpy array
            Predicted probabilities (between 0 and 1)
            
        Returns:
        --------
        loss : float
            Binary cross-entropy loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def _objective(self, y_true, y_pred):
        """
        Full training objective = data loss + L2 penalty

        objective = _compute_loss(y_true, y_pred)
                    + (reg_lambda / (2n)) * sum(theta_j^2)

        The sum runs over feature coefficients only; the intercept is never
        penalized. With the default reg_lambda=0.0 this equals _compute_loss
        exactly, so self.losses is plain binary cross-entropy unless you asked
        for regularization.

        Parameters:
        -----------
        y_true : numpy array
            True labels (0 or 1)
        y_pred : numpy array
            Predicted probabilities produced by the CURRENT self.coefficients

        Returns:
        --------
        objective : float
            The quantity gradient descent is actually minimizing
        """
        loss = self._compute_loss(y_true, y_pred)
        if self.reg_lambda > 0:
            weights = self.coefficients[1:] if self.fit_intercept else self.coefficients
            loss += (self.reg_lambda / (2 * len(y_true))) * np.sum(weights ** 2)
        return loss

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        
        The gradient of the mean binary cross-entropy with respect to the
        coefficient vector theta is

            grad = (1/n) * X^T @ (sigmoid(X @ theta) - y)

        The sigmoid derivative sigmoid(z)*(1 - sigmoid(z)) that the chain rule
        produces cancels exactly against the 1/p and 1/(1-p) coming from the
        logs, so the cross-entropy gradient collapses to the same
        (1/n) * X^T @ error shape as ordinary linear regression. The line
        `gradients = (1 / n_samples) * (X_with_bias.T @ error)` below is that
        formula verbatim. With reg_lambda > 0 the L2 term
        (reg_lambda / n) * theta_j is added for every coefficient except the
        intercept.

        Parameters:
        -----------
        X : numpy array (or list of lists) of shape (n_samples, n_features)
            Training data. A 1-D array is read as a single feature column.
        y : numpy array (or list) of shape (n_samples,)
            Target values (must contain only 0 and 1)

        Returns:
        --------
        self : LogisticRegression
            The fitted model, so calls can be chained:
                model = LogisticRegression().fit(X, y)

        Notes:
        ------
        Coefficients start at ZERO, not at random values. Binary cross-entropy
        with a linear model is convex, so there is no symmetry to break and no
        local minimum to escape: zeros make every run of this file exactly
        reproducible, and they are what scikit-learn and statsmodels use.

        self.losses is reset on every call (refitting does not concatenate
        histories) and records the objective once per iteration PLUS once after
        the final update, so len(self.losses) == iterations + 1.
        losses[0] is the loss of the all-zeros start (always log(2) = 0.6931)
        and losses[-1] is the loss of the model you get back.
        """
        # Accept lists and 1-D input, and fail loudly on labels that are not 0/1
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]} labels."
            )
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError(
                "y must contain only 0 and 1 for binary logistic regression. "
                "Encode labels first, e.g. y = (y == positive_class).astype(int)"
            )

        n_samples, n_features = X.shape
        
        # Add bias term (column of ones) if fit_intercept is True
        if self.fit_intercept:
            X_with_bias = np.hstack((np.ones((n_samples, 1)), X))
        else:
            X_with_bias = X
        
        # Which coefficients the L2 penalty touches (never the intercept)
        penalized = slice(1, None) if self.fit_intercept else slice(None)

        # Initialize coefficients at zero (convex problem - see Notes above)
        self.coefficients = np.zeros(X_with_bias.shape[1])

        # Start a fresh loss history for this fit
        self.losses = []
        
        # Gradient descent optimization
        for i in range(self.iterations):
            # Forward pass: z = X @ theta, then p = sigmoid(z)
            linear_model = X_with_bias @ self.coefficients
            y_pred = self._sigmoid(linear_model)
            
            # Compute the objective (for tracking) at the CURRENT coefficients
            self.losses.append(self._objective(y, y_pred))
            
            # Backward pass: grad = (1/n) * X^T @ (p - y)
            error = y_pred - y
            gradients = (1 / n_samples) * (X_with_bias.T @ error)
            
            # L2 penalty gradient: d/dtheta_j of (reg_lambda / 2n) * theta_j^2
            if self.reg_lambda > 0:
                gradients[penalized] += (
                    (self.reg_lambda / n_samples) * self.coefficients[penalized]
                )

            # Update coefficients: theta = theta - learning_rate * grad
            self.coefficients -= self.learning_rate * gradients

        # Record the loss of the FINAL coefficients too. Without this,
        # losses[-1] would describe the second-to-last model, not the one
        # this method returns.
        self.losses.append(
            self._objective(y, self._sigmoid(X_with_bias @ self.coefficients))
        )
        
        # Separate intercept from feature coefficients
        if self.fit_intercept:
            self.intercept = self.coefficients[0]
            self.feature_coefficients = self.coefficients[1:]
        else:
            self.intercept = 0
            self.feature_coefficients = self.coefficients
    
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples
        
        Parameters:
        -----------
        X : numpy array (or list of lists) of shape (n_samples, n_features)
            Data to make predictions on. A 1-D array is read as a single
            feature column.
            
        Returns:
        --------
        probabilities : numpy array of shape (n_samples,)
            P(y=1) only - values between 0 and 1.
            NOTE: this is 1-D. Unlike scikit-learn, which returns an
            (n_samples, 2) matrix of [P(y=0), P(y=1)], this method returns just
            the positive-class column. Use `1 - p` when you need P(y=0).
        """
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add bias term for prediction
        if self.fit_intercept:
            X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            X_with_bias = X
        
        # Calculate probabilities: z = X @ theta (the log-odds), p = sigmoid(z)
        linear_model = X_with_bias @ self.coefficients
        probabilities = self._sigmoid(linear_model)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on
        threshold : float, default=0.5
            Decision threshold for classification
            If probability >= threshold, predict 1, else predict 0
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions
    
    def get_coefficients(self):
        """
        Get the model coefficients
        
        Returns:
        --------
        dict : Dictionary containing intercept and feature coefficients
            'intercept'    -> float, the bias b0
            'coefficients' -> numpy array of shape (n_features,)

        Each coefficient is the change in LOG-ODDS per one-unit increase of its
        feature, so exp(coefficient) is the odds ratio. If features were
        standardized before fitting, "one unit" means "one standard deviation".
        """
        if self.feature_coefficients is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        return {
            'intercept': self.intercept,
            'coefficients': self.feature_coefficients
        }
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        This is a classifier, so score() returns ACCURACY (the fraction of
        labels predicted correctly), not the R^2 that the repo's regressors
        return.

        Parameters:
        -----------
        X : numpy array (or list of lists) of shape (n_samples, n_features)
            Test data
        y : numpy array (or list) of shape (n_samples,)
            True labels (0 or 1)
            
        Returns:
        --------
        accuracy : float
            Accuracy score (proportion of correct predictions), 0.0 to 1.0
        """
        y = np.asarray(y, dtype=float).ravel()
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


"""
USAGE EXAMPLE 1: Simple Binary Classification

import numpy as np

# Sample data: Predicting if a student passes (1) or fails (0) based on study hours and attendance
X_train = np.array([
    [1, 20],   # 1 hour study, 20% attendance
    [2, 40],   # 2 hours study, 40% attendance
    [3, 60],   # 3 hours study, 60% attendance
    [4, 90],   # 4 hours study, 90% attendance
    [5, 75],   # 5 hours study, 75% attendance
    [1.5, 30],
    [2.5, 50],
    [3.5, 70],
    [4.5, 90]
])

y_train = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Standardize the features BEFORE fitting.
# Attendance spans 20-100 while study hours span 1-5, so on the raw scale the
# attendance column dominates every gradient: at learning_rate=0.01 the loss
# actually CLIMBS and the model predicts one class for everything. Putting both
# features on the same scale is what makes a large, fast step size safe.
mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
X_train_scaled = (X_train - mu) / sd

# Create and train the model
model = LogisticRegression(learning_rate=0.5, iterations=5000)
model.fit(X_train_scaled, y_train)

print(f"Training accuracy: {model.score(X_train_scaled, y_train):.4f}")
print(f"Loss: {model.losses[0]:.4f} -> {model.losses[-1]:.4f}")
# Training accuracy: 0.7778
# Loss: 0.6931 -> 0.2776
# (0.7778 = 7 of 9. The [3, 60] -> Fail student sits between two Pass
#  students, so no straight line can get all nine right.)

# Make predictions - scale new data with the TRAINING mu and sd, never with
# statistics recomputed on the test set
X_test = np.array([
    [2, 30],   # Low study, low attendance
    [4, 85],   # High study, high attendance
    [3, 55]    # Medium study, medium attendance
])
X_test_scaled = (X_test - mu) / sd

# Get probabilities
probabilities = model.predict_proba(X_test_scaled)
print("Predicted probabilities:", probabilities)
# Predicted probabilities: [0.08038779 0.96969598 0.59623216]

# Get class predictions
predictions = model.predict(X_test_scaled)
print("Predicted classes:", predictions)
# Predicted classes: [0 1 1]   (Fail, Pass, Pass)

# Get coefficients. These are on the STANDARDIZED scale, so "one unit" means
# "one standard deviation of that feature".
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.4f}")
print(f"Coefficients: {coeffs['coefficients']}")
# Intercept: 0.5558
# Coefficients: [2.04198955 1.18563097]
# exp(2.0420) = 7.71 -> one extra standard deviation of study hours multiplies
# the odds of passing by about 7.7. exp(1.1856) = 3.27 for attendance.
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Breast Cancer Dataset)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for gradient descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(learning_rate=0.1, iterations=2000)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Display probabilities for first 5 test samples
probabilities = model.predict_proba(X_test_scaled[:5])
print(f"\nProbabilities for first 5 samples:")
for i, (prob, true_label, pred_label) in enumerate(zip(probabilities, y_test[:5], y_pred[:5])):
    print(f"  Sample {i+1}: P(y=1)={prob:.4f}, True={true_label}, Predicted={pred_label}")

# Display coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.4f}")
print(f"Number of features: {len(coeffs['coefficients'])}")
"""

"""
USAGE EXAMPLE 3: Visualizing Training Progress and Decision Boundary

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_samples = 200

# Class 0 (centered at [2, 2])
X_class0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
y_class0 = np.zeros(n_samples // 2)

# Class 1 (centered at [5, 5])
X_class1 = np.random.randn(n_samples // 2, 2) + np.array([5, 5])
y_class1 = np.ones(n_samples // 2)

# Combine data, then SHUFFLE before splitting - the rows are currently sorted
# by class, so slicing without a shuffle would put every class-1 point in the
# test set
X_all = np.vstack([X_class0, X_class1])
y_all = np.hstack([y_class0, y_class1])
idx = np.random.permutation(n_samples)
X_all, y_all = X_all[idx], y_all[idx]

# Hold out the last 50 rows. Note [:150] and [150:] - not [:150] and [50:],
# which would leak 100 training rows into the test set.
X_train, X_test = X_all[:150], X_all[150:]
y_train, y_test = y_all[:150], y_all[150:]

# Train model
model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train, y_train)

# Plot loss curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)

# Plot decision boundary
plt.subplot(1, 2, 2)

# Create mesh
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict on mesh
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Plot data points
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], 
            c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], 
            c='red', label='Class 1', alpha=0.6, edgecolors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Decision Boundary (Accuracy: {model.score(X_train, y_train):.2f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nTrain accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test  accuracy: {model.score(X_test, y_test):.4f}")
print(f"Final loss: {model.losses[-1]:.4f}")
"""

"""
USAGE EXAMPLE 4: Comparing Different Learning Rates

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic classification dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=8,
                           n_redundant=2, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different learning rates.
# Only 100 iterations, so the point of the table is visible: the small rates
# have barely moved off the starting loss of log(2) = 0.6931, while the large
# rates have already converged. Run the same loop with iterations=1000 and
# every rate from 0.1 upward collapses to the SAME numbers - once gradient
# descent has converged, the learning rate no longer matters.
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]

print("Comparing Different Learning Rates (100 iterations):\n")
print(f"{'Learning Rate':<15} {'Train Accuracy':<15} {'Test Accuracy':<15} {'Final Loss':<15}")
print("-" * 60)

for lr in learning_rates:
    model = LogisticRegression(learning_rate=lr, iterations=100)
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    final_loss = model.losses[-1]
    
    print(f"{lr:<15.3f} {train_acc:<15.4f} {test_acc:<15.4f} {final_loss:<15.4f}")

# NOTE: this loop reads the TEST accuracy for every setting. That is fine for a
# demonstration, but in a real workflow you would pick the learning rate on a
# separate validation split and touch the test set only once, at the end.
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _4_logistic_regression.py
    # Requires numpy only. Output is ASCII-only and fully reproducible.
    # ----------------------------------------------------------------
    np.random.seed(42)

    print("=" * 64)
    print("LOGISTIC REGRESSION FROM SCRATCH - PLUG-AND-PLAY DEMO")
    print("=" * 64)

    # ------------------------------------------------------------------
    # DEMO 1 - Binary classification on two Gaussian blobs
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("DEMO 1 - Binary classification: two overlapping Gaussian blobs")
    print("Look for: the loss falls on every step, and the boundary learned")
    print("from 150 points generalizes to 50 points the model never saw.")
    print("The blobs deliberately overlap, so 100% is not achievable.")
    print("=" * 64)

    X0 = np.random.randn(100, 2) + np.array([-1, -1])   # class 0 cloud
    X1 = np.random.randn(100, 2) + np.array([1, 1])     # class 1 cloud
    X_cls = np.vstack([X0, X1])
    y_cls = np.array([0] * 100 + [1] * 100)

    # Shuffle before slicing. The rows are stacked class-0-then-class-1, so an
    # unshuffled split would hand the test set every single class-1 point.
    idx = np.random.permutation(200)
    X_cls, y_cls = X_cls[idx], y_cls[idx]
    X_tr, X_te = X_cls[:150], X_cls[150:]   # [150:], NOT [50:] - no overlap
    y_tr, y_te = y_cls[:150], y_cls[150:]

    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X_tr, y_tr)

    print(f"Train accuracy : {model.score(X_tr, y_tr):.4f}")
    print(f"Test  accuracy : {model.score(X_te, y_te):.4f}")
    print(f"Loss           : {model.losses[0]:.4f} (start) -> "
          f"{model.losses[-1]:.4f} (end)")
    decreasing = all(model.losses[i + 1] <= model.losses[i]
                     for i in range(len(model.losses) - 1))
    print(f"Loss fell on every step: {decreasing}")

    coeffs = model.get_coefficients()
    print(f"Intercept      : {coeffs['intercept']:.4f}")
    print(f"Coefficients   : [{coeffs['coefficients'][0]:.4f}, "
          f"{coeffs['coefficients'][1]:.4f}]")

    probas = model.predict_proba(X_te)
    preds = model.predict(X_te)
    print("\nSample test predictions (true, P(y=1), predicted):")
    for i in range(5):
        print(f"  true={int(y_te[i])}  P(y=1)={probas[i]:.4f}  pred={preds[i]}")

    # threshold is a public argument of predict(): moving it trades precision
    # against recall without refitting anything.
    print("\nEffect of the decision threshold on the 50 test points:")
    for t in [0.3, 0.5, 0.7]:
        t_preds = model.predict(X_te, threshold=t)
        tp = int(np.sum((t_preds == 1) & (y_te == 1)))
        fp = int(np.sum((t_preds == 1) & (y_te == 0)))
        fn = int(np.sum((t_preds == 0) & (y_te == 1)))
        print(f"  threshold={t:.1f} -> positives={int(np.sum(t_preds)):2d}  "
              f"accuracy={np.mean(t_preds == y_te):.4f}  "
              f"precision={tp / max(tp + fp, 1):.3f}  "
              f"recall={tp / max(tp + fn, 1):.3f}")
    print("  Raising the threshold buys precision by giving up recall.")

    # ------------------------------------------------------------------
    # DEMO 2 - Student pass/fail (the worked example from the .md)
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("DEMO 2 - Student pass/fail, with feature scaling")
    print("Look for: attendance spans 20-100 while study hours span 1-5, so")
    print("the raw columns must be standardized before gradient descent can")
    print("make progress. Coefficients are then read as odds ratios.")
    print("=" * 64)

    X_stud = np.array([
        [1.0, 20], [2.0, 40], [3.0, 60], [4.0, 90], [5.0, 75],
        [1.5, 30], [2.5, 50], [3.5, 70], [4.5, 90]
    ])
    y_stud = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])   # 0 = Fail, 1 = Pass

    mu, sd = X_stud.mean(axis=0), X_stud.std(axis=0)
    X_stud_scaled = (X_stud - mu) / sd

    stud_model = LogisticRegression(learning_rate=0.5, iterations=5000)
    stud_model.fit(X_stud_scaled, y_stud)

    print(f"Train accuracy : {stud_model.score(X_stud_scaled, y_stud):.4f}"
          "  (7 of 9)")
    print("  The [3, 60] -> Fail student sits between two Pass students, so no")
    print("  straight line can label all nine correctly. That is the point.")
    print(f"Loss           : {stud_model.losses[0]:.4f} -> "
          f"{stud_model.losses[-1]:.4f}")

    c = stud_model.get_coefficients()
    print(f"Intercept      : {c['intercept']:.4f}")
    print(f"Study hours    : coef={c['coefficients'][0]:7.4f}  "
          f"odds ratio exp(coef)={np.exp(c['coefficients'][0]):5.2f} per std dev")
    print(f"Attendance     : coef={c['coefficients'][1]:7.4f}  "
          f"odds ratio exp(coef)={np.exp(c['coefficients'][1]):5.2f} per std dev")

    X_new = np.array([[2, 30], [4, 85], [3, 55]])
    descriptions = ["low study,    low attendance   ",
                    "high study,   high attendance  ",
                    "medium study, medium attendance"]
    X_new_scaled = (X_new - mu) / sd
    p_new = stud_model.predict_proba(X_new_scaled)
    pred_new = stud_model.predict(X_new_scaled)
    print("\nThree new students (scaled with the TRAINING mu and sd):")
    for i in range(3):
        outcome = "Pass" if pred_new[i] == 1 else "Fail"
        print(f"  {descriptions[i]}  P(pass)={p_new[i]:.4f} -> {outcome}")

    # ------------------------------------------------------------------
    # DEMO 3 - What reg_lambda is for
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("DEMO 3 - Why reg_lambda exists: perfectly separable data")
    print("Look for: with no penalty the coefficient just keeps growing (the")
    print("unregularized maximum likelihood has no finite optimum here);")
    print("L2 regularization pins it down.")
    print("=" * 64)

    X_sep = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_sep = np.array([0, 0, 1, 1])
    print(f"{'reg_lambda':<12}{'coefficient':<14}{'objective':<12}{'train accuracy':<15}")
    print("-" * 53)
    for lam in [0.0, 1.0, 10.0]:
        sep_model = LogisticRegression(learning_rate=0.5, iterations=20000,
                                       reg_lambda=lam).fit(X_sep, y_sep)
        sep_c = sep_model.get_coefficients()
        print(f"{lam:<12.1f}{sep_c['coefficients'][0]:<14.4f}"
              f"{sep_model.losses[-1]:<12.4f}"
              f"{sep_model.score(X_sep, y_sep):<15.4f}")

    print("\n" + "=" * 64)
    print("Demo complete.")
    print("=" * 64)
