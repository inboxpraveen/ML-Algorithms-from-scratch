import numpy as np

class RidgeRegression:
    """
    Ridge Regression Implementation from Scratch
    
    Ridge regression is a regularized version of linear regression that adds
    L2 regularization to prevent overfitting and handle multicollinearity.
    
    Formula: y = b0 + b1*x1 + b2*x2 + ... + bn*xn
    
    The Normal Equation with L2 regularization:
    θ = (X^T * X + λI)^(-1) * X^T * y
    
    where:
        y = target variable (dependent variable)
        x1, x2, ..., xn = independent variables (features)
        b0 = intercept (bias term)
        b1, b2, ..., bn = coefficients for each feature
        λ (lambda/alpha) = regularization parameter
        I = identity matrix
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the Ridge Regression model
        
        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization strength. Must be a positive float.
            Larger values specify stronger regularization.
            - alpha = 0: Equivalent to ordinary least squares (no regularization)
            - alpha > 0: Adds penalty to large coefficients
        """
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """
        Train the ridge regression model using the regularized Normal Equation
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data with multiple features
        y : numpy array of shape (n_samples,)
            Target values
            
        The Regularized Normal Equation: θ = (X^T * X + λI)^(-1) * X^T * y
        """
        # Add bias term (column of ones) for the intercept
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Create identity matrix for regularization
        # Note: We don't regularize the bias term (first element)
        identity = np.eye(X_with_bias.shape[1])
        identity[0, 0] = 0  # Don't penalize the intercept
        
        # Calculate coefficients using the regularized Normal Equation
        # θ = (X^T * X + λI)^(-1) * X^T * y
        regularization_term = self.alpha * identity
        self.coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias + regularization_term) @ X_with_bias.T @ y
        
        # Separate intercept from feature coefficients for clarity
        self.intercept = self.coefficients[0]
        self.feature_coefficients = self.coefficients[1:]
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted values
        """
        # Add bias term for prediction
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Calculate predictions: y = X * θ
        return X_with_bias @ self.coefficients
    
    def get_coefficients(self):
        """
        Get the model coefficients
        
        Returns:
        --------
        dict : Dictionary containing intercept and feature coefficients
        """
        return {
            'intercept': self.intercept,
            'coefficients': self.feature_coefficients,
            'alpha': self.alpha
        }
    
    def score(self, X, y):
        """
        Calculate R² (coefficient of determination) score
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test data
        y : numpy array of shape (n_samples,)
            True values
            
        Returns:
        --------
        r2_score : float
            R² score (1.0 is perfect prediction)
        """
        y_pred = self.predict(X)
        
        # Calculate R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score


"""
USAGE EXAMPLE 1: Ridge Regression with Different Alpha Values

import numpy as np

# Sample data: Predicting house prices based on [square_feet, bedrooms, age]
X_train = np.array([
    [1500, 3, 10],  # 1500 sq ft, 3 bedrooms, 10 years old
    [2000, 4, 5],   # 2000 sq ft, 4 bedrooms, 5 years old
    [1200, 2, 15],  # 1200 sq ft, 2 bedrooms, 15 years old
    [1800, 3, 8],   # 1800 sq ft, 3 bedrooms, 8 years old
    [2500, 5, 2]    # 2500 sq ft, 5 bedrooms, 2 years old
])

y_train = np.array([300000, 400000, 250000, 350000, 500000])  # House prices

# Try different regularization strengths
alphas = [0.0, 0.1, 1.0, 10.0, 100.0]

for alpha in alphas:
    model = RidgeRegression(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Get coefficients
    coeffs = model.get_coefficients()
    print(f"\nAlpha = {alpha}")
    print(f"Intercept: ${coeffs['intercept']:.2f}")
    print(f"Coefficients: {coeffs['coefficients']}")
    
    # Make predictions
    X_test = np.array([[1600, 3, 7]])  # 1600 sq ft, 3 bedrooms, 7 years old
    prediction = model.predict(X_test)
    print(f"Predicted price: ${prediction[0]:.2f}")
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Diabetes Dataset)

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset (has 10 features)
data = load_diabetes()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (recommended for Ridge Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different alpha values
print("Comparing Ridge Regression with different alpha values:\n")

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    # Create and train the ridge regression model
    model = RidgeRegression(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate R² score
    r2 = model.score(X_test_scaled, y_test)
    print(f"Alpha = {alpha:6.2f} | R² Score: {r2:.4f}")

# Train with optimal alpha
print("\n" + "="*50)
print("Training with optimal alpha = 1.0")
print("="*50)

model = RidgeRegression(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Evaluate
r2 = model.score(X_test_scaled, y_test)
print(f"\nR² Score: {r2:.4f}")

# Display coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.2f}")
print("\nFeature Coefficients:")
for i, coef in enumerate(coeffs['coefficients']):
    print(f"  Feature {i+1}: {coef:.2f}")
"""

"""
USAGE EXAMPLE 3: Comparing Ridge vs Multiple Regression

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load dataset
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multiple Regression (no regularization)
from _2_multiple_regression import MultipleRegression
model_ols = MultipleRegression()
model_ols.fit(X_train, y_train)
r2_ols = model_ols.score(X_test, y_test)

# Train Ridge Regression (with regularization)
model_ridge = RidgeRegression(alpha=1.0)
model_ridge.fit(X_train, y_train)
r2_ridge = model_ridge.score(X_test, y_test)

print("Comparison: Multiple Regression vs Ridge Regression")
print(f"Multiple Regression R²: {r2_ols:.4f}")
print(f"Ridge Regression R²:    {r2_ridge:.4f}")
print(f"\nDifference: {abs(r2_ridge - r2_ols):.4f}")

# Compare coefficient magnitudes
coeffs_ols = model_ols.get_coefficients()['coefficients']
coeffs_ridge = model_ridge.get_coefficients()['coefficients']

print("\nCoefficient Magnitudes:")
print(f"Multiple Regression: {np.linalg.norm(coeffs_ols):.2f}")
print(f"Ridge Regression:    {np.linalg.norm(coeffs_ridge):.2f}")
print(f"\nRidge reduces coefficient magnitudes by: {(1 - np.linalg.norm(coeffs_ridge)/np.linalg.norm(coeffs_ols))*100:.1f}%")
"""

