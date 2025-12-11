import numpy as np

class MultipleRegression:
    """
    Multiple Linear Regression Implementation from Scratch
    
    Multiple regression is used when we want to predict a target variable
    using multiple features (independent variables).
    
    Formula: y = b0 + b1*x1 + b2*x2 + ... + bn*xn
    where:
        y = target variable (dependent variable)
        x1, x2, ..., xn = independent variables (features)
        b0 = intercept (bias term)
        b1, b2, ..., bn = coefficients for each feature
    """
    
    def __init__(self):
        """Initialize the Multiple Regression model"""
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """
        Train the multiple regression model using the Normal Equation
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data with multiple features
        y : numpy array of shape (n_samples,)
            Target values
            
        The Normal Equation: θ = (X^T * X)^(-1) * X^T * y
        """
        # Add bias term (column of ones) for the intercept
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Calculate coefficients using the Normal Equation
        # θ = (X^T * X)^(-1) * X^T * y
        self.coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
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
            'coefficients': self.feature_coefficients
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
USAGE EXAMPLE 1: Simple Multiple Regression with 3 Features

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

# Create and train the model
model = MultipleRegression()
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([
    [1600, 3, 7],   # 1600 sq ft, 3 bedrooms, 7 years old
    [2200, 4, 3]    # 2200 sq ft, 4 bedrooms, 3 years old
])

predictions = model.predict(X_test)
print("Predicted prices:", predictions)

# Get coefficients
coeffs = model.get_coefficients()
print(f"Intercept: {coeffs['intercept']}")
print(f"Coefficients: {coeffs['coefficients']}")
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Diabetes Dataset)

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load the diabetes dataset (has 10 features)
data = load_diabetes()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the multiple regression model
model = MultipleRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R² score
r2 = model.score(X_test, y_test)
print(f"R² Score: {r2:.4f}")

# Display coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.2f}")
print("\nFeature Coefficients:")
for i, coef in enumerate(coeffs['coefficients']):
    print(f"  Feature {i+1}: {coef:.2f}")
"""

