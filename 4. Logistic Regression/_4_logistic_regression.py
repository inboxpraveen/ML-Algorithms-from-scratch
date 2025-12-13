import numpy as np

class LogisticRegression:
    """
    Logistic Regression Implementation from Scratch
    
    Logistic regression is used for binary classification problems where
    we want to predict if something belongs to one class or another (0 or 1).
    
    Formula: p(y=1|x) = 1 / (1 + e^(-(b0 + b1*x1 + b2*x2 + ... + bn*xn)))
    
    where:
        p(y=1|x) = probability that y equals 1 given x
        e = Euler's number (≈ 2.718)
        b0 = intercept (bias term)
        b1, b2, ..., bn = coefficients for each feature
        x1, x2, ..., xn = independent variables (features)
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000, fit_intercept=True):
        """
        Initialize the Logistic Regression model
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Step size for gradient descent optimization
            Larger values = faster convergence but risk overshooting
            Smaller values = slower but more stable convergence
        
        iterations : int, default=1000
            Number of iterations for gradient descent
            More iterations = better convergence (but longer training)
        
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.losses = []  # Track loss history
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function
        
        Maps any real number to a value between 0 and 1
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Parameters:
        -----------
        z : numpy array
            Linear combination of features and coefficients
            
        Returns:
        --------
        sigmoid : numpy array
            Values between 0 and 1 (probabilities)
        """
        # Clip values to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss
        
        Loss = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
        
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
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Target values (must be 0 or 1)
        """
        n_samples, n_features = X.shape
        
        # Add bias term (column of ones) if fit_intercept is True
        if self.fit_intercept:
            X_with_bias = np.hstack((np.ones((n_samples, 1)), X))
        else:
            X_with_bias = X
        
        # Initialize coefficients with small random values
        self.coefficients = np.random.randn(X_with_bias.shape[1]) * 0.01
        
        # Gradient descent optimization
        for i in range(self.iterations):
            # Forward pass: compute predictions
            linear_model = X_with_bias @ self.coefficients
            y_pred = self._sigmoid(linear_model)
            
            # Compute loss (for tracking)
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward pass: compute gradients
            error = y_pred - y
            gradients = (1 / n_samples) * (X_with_bias.T @ error)
            
            # Update coefficients
            self.coefficients -= self.learning_rate * gradients
        
        # Separate intercept from feature coefficients
        if self.fit_intercept:
            self.intercept = self.coefficients[0]
            self.feature_coefficients = self.coefficients[1:]
        else:
            self.intercept = 0
            self.feature_coefficients = self.coefficients
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on
            
        Returns:
        --------
        probabilities : numpy array of shape (n_samples,)
            Predicted probabilities for class 1 (values between 0 and 1)
        """
        # Add bias term for prediction
        if self.fit_intercept:
            X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            X_with_bias = X
        
        # Calculate probabilities
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
        """
        return {
            'intercept': self.intercept,
            'coefficients': self.feature_coefficients
        }
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test data
        y : numpy array of shape (n_samples,)
            True labels (0 or 1)
            
        Returns:
        --------
        accuracy : float
            Accuracy score (proportion of correct predictions)
        """
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
    [4, 80],   # 4 hours study, 80% attendance
    [5, 100],  # 5 hours study, 100% attendance
    [1.5, 30],
    [2.5, 50],
    [3.5, 70],
    [4.5, 90]
])

y_train = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])  # 0 = Fail, 1 = Pass

# Create and train the model
model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [2, 30],   # Low study, low attendance
    [4, 85],   # High study, high attendance
    [3, 55]    # Medium study, medium attendance
])

# Get probabilities
probabilities = model.predict_proba(X_test)
print("Predicted probabilities:", probabilities)

# Get class predictions
predictions = model.predict(X_test)
print("Predicted classes:", predictions)

# Get coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.4f}")
print(f"Coefficients: {coeffs['coefficients']}")
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

# Combine data
X_train = np.vstack([X_class0, X_class1])
y_train = np.hstack([y_class0, y_class1])

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

print(f"\nFinal training accuracy: {model.score(X_train, y_train):.4f}")
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

# Try different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]

print("Comparing Different Learning Rates:\n")
print(f"{'Learning Rate':<15} {'Train Accuracy':<15} {'Test Accuracy':<15} {'Final Loss':<15}")
print("-" * 60)

for lr in learning_rates:
    model = LogisticRegression(learning_rate=lr, iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    final_loss = model.losses[-1]
    
    print(f"{lr:<15.3f} {train_acc:<15.4f} {test_acc:<15.4f} {final_loss:<15.4f}")
"""

