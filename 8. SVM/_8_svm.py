import numpy as np

class SupportVectorMachine:
    """
    Support Vector Machine (SVM) Implementation from Scratch
    
    SVM is a powerful algorithm for classification that finds the optimal
    hyperplane that maximally separates different classes in the feature space.
    
    Key Idea: "Find the widest street that separates the two classes"
    
    The decision boundary is:
        f(x) = w·x + b
        
    Classification rule:
        y = +1 if w·x + b >= 0
        y = -1 if w·x + b < 0
    
    where:
        w = weight vector (perpendicular to decision boundary)
        b = bias term (position of decision boundary)
        x = input features
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000):
        """
        Initialize the Support Vector Machine model
        
        Parameters:
        -----------
        learning_rate : float, default=0.001
            Step size for gradient descent optimization
            Controls how much we update weights in each iteration
            Typical range: 0.0001 to 0.01
        
        lambda_param : float, default=0.01
            Regularization parameter (controls margin width)
            Larger values = wider margin, simpler model, more tolerance for misclassification
            Smaller values = narrower margin, more complex model, less tolerance
            Typical range: 0.001 to 1.0
        
        iterations : int, default=1000
            Number of iterations for training
            More iterations = better convergence (but longer training)
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []  # Track loss history
    
    def _compute_loss(self, X, y):
        """
        Compute hinge loss with L2 regularization
        
        Hinge Loss: max(0, 1 - y * (w·x + b))
        Total Loss: λ||w||² + (1/n)Σ max(0, 1 - y * (w·x + b))
        
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
        Train the SVM model using gradient descent
        
        The SVM optimization problem:
        Minimize: λ||w||² + (1/n)Σ max(0, 1 - y_i * (w·x_i + b))
        
        Gradient when y_i * (w·x_i + b) < 1 (misclassified or within margin):
            ∂L/∂w = 2λw - y_i * x_i
            ∂L/∂b = -y_i
        
        Gradient when y_i * (w·x_i + b) >= 1 (correctly classified outside margin):
            ∂L/∂w = 2λw
            ∂L/∂b = 0
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Target labels (should be -1 or +1)
            If labels are 0 and 1, they will be converted to -1 and +1
        """
        n_samples, n_features = X.shape
        
        # Convert labels to -1 and +1 if they are 0 and 1
        y_labels = np.where(y <= 0, -1, 1)
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent optimization
        for iteration in range(self.iterations):
            # Compute loss for tracking
            loss = self._compute_loss(X, y_labels)
            self.losses.append(loss)
            
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
    
    def predict(self, X):
        """
        Predict class labels for samples
        
        Decision rule: sign(w·x + b)
        Returns +1 if w·x + b >= 0, else -1
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted class labels (-1 or +1)
        """
        # Calculate decision function values
        linear_output = X @ self.weights + self.bias
        
        # Apply sign function (return +1 or -1)
        predictions = np.sign(linear_output)
        
        # Handle case where linear_output is exactly 0
        predictions[predictions == 0] = 1
        
        return predictions
    
    def decision_function(self, X):
        """
        Calculate the distance of samples from the decision boundary
        
        Distance = w·x + b
        
        Positive values = predicted as class +1
        Negative values = predicted as class -1
        Magnitude = confidence (larger absolute value = more confident)
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to calculate distances for
            
        Returns:
        --------
        distances : numpy array of shape (n_samples,)
            Signed distances from decision boundary
        """
        return X @ self.weights + self.bias
    
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
# Output: [1, -1, ?] (Apple, Orange, depends on decision boundary)

# Get decision function values (distances from boundary)
distances = model.decision_function(X_test)
print("\nDistances from decision boundary:", distances)
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

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'SVM Decision Boundary and Margins\nAccuracy: {model.score(X_train, y_train):.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal training accuracy: {model.score(X_train, y_train):.4f}")
print(f"Final loss: {model.losses[-1]:.4f}")
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

print("Comparing Different Regularization Parameters (λ):\n")
print(f"{'Lambda (λ)':<15} {'Train Accuracy':<20} {'Test Accuracy':<20} {'Final Loss':<15}")
print("-" * 70)

for lambda_param in lambda_values:
    model = SupportVectorMachine(learning_rate=0.001, lambda_param=lambda_param, iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    final_loss = model.losses[-1]
    
    print(f"{lambda_param:<15.4f} {train_acc:<20.4f} {test_acc:<20.4f} {final_loss:<15.4f}")

print("\nObservations:")
print("- Small λ (0.0001-0.001): Narrow margin, may overfit")
print("- Medium λ (0.01-0.1): Balanced, good generalization")
print("- Large λ (1.0+): Wide margin, may underfit")
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
    status = "✓" if y_pred[i] == y_test[i] else "✗"
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

