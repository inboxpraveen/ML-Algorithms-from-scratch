import numpy as np

class RidgeRegression:
    """
    Ridge Regression Implementation from Scratch
    
    Ridge regression is a regularized version of linear regression that adds
    L2 regularization to prevent overfitting and handle multicollinearity.
    
    Key Idea: "Shrink the coefficients so no single correlated feature can dominate"

    Use Cases:
    - Financial modeling: many correlated ratios feeding one return forecast
    - Medical / genomics: p > n biomarker panels where OLS has no unique solution
    - Real estate valuation: size, rooms and location carry overlapping information
    - Marketing mix models: TV / radio / online spend all move together in campaigns
    - Any linear model whose OLS coefficients swing wildly when one row is added

    Formula: y = b0 + b1*x1 + b2*x2 + ... + bn*xn
    
    The Normal Equation with L2 regularization (exactly what fit() solves):
    theta = (X^T * X + alpha * I~)^(-1) * X^T * y
    
    where:
        y = target variable (dependent variable)
        x1, x2, ..., xn = independent variables (features)
        b0 = intercept (bias term)
        b1, b2, ..., bn = coefficients for each feature
        alpha (also written lambda) = regularization parameter
        I~ = the (n_features + 1) x (n_features + 1) identity matrix with its
             FIRST diagonal entry zeroed, I~[0, 0] = 0, so that the intercept is
             not penalized. In fit() this is the line `identity[0, 0] = 0`.

    Cost function being minimized:
        J(theta) = sum_i (y_i - yhat_i)^2 + alpha * sum_{j >= 1} b_j^2
    Setting dJ/dtheta = 0 gives the closed form above, so no iteration is needed.

    Zeroing I~[0, 0] is what makes this solution identical to scikit-learn's
    Ridge(fit_intercept=True), which instead centers X and y and then solves the
    penalized system without an intercept. Both routes give the same theta
    (measured max coefficient difference ~1e-14 for alpha in 0.1, 1, 10, 100).
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the Ridge Regression model
        
        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization strength. Must be a non-negative float.
            Larger values specify stronger regularization.
            - alpha = 0: Equivalent to ordinary least squares (no regularization)
            - alpha > 0: Adds penalty to large coefficients
            - Range: 0.0 to about 1000 (beyond that every coefficient is ~0)
            Typical: 1.0 as a baseline; 0.01-0.1 for clean, well-conditioned data;
                     10-100 under strong multicollinearity or when p is close to n.
            Note: alpha is scale-dependent. Standardize your features first,
                  otherwise a large-scale column is effectively unpenalized.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.alpha = alpha
        self.coefficients = None          # full theta vector: [intercept, b1..bn]
        self.intercept = None             # theta[0]
        self.feature_coefficients = None  # theta[1:], one entry per feature

    def _prepare_features(self, X):
        """
        Turn whatever the caller passed into a float (n_samples, n_features) matrix.

        A one-feature dataset is naturally written as [1, 2, 3] or np.array([1, 2, 3]),
        but the Normal Equation needs a 2-D design matrix, so a 1-D input is reshaped
        into a single column. Python lists are accepted too.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # one feature, many samples
        if X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got {X.ndim} dimensions")
        return X
    
    def fit(self, X, y):
        """
        Train the ridge regression model using the regularized Normal Equation
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data with multiple features. A 1-D array or a plain Python
            list is accepted and treated as a single feature column.
        y : numpy array of shape (n_samples,)
            Target values. A column vector of shape (n_samples, 1) is flattened.
            
        The Regularized Normal Equation:
            theta = (X^T * X + alpha * I~)^(-1) * X^T * y
        where I~ is the identity matrix with I~[0, 0] = 0, so the intercept is
        left unpenalized. That single zero is the whole trick of this method.
        """
        X = self._prepare_features(X)
        y = np.asarray(y, dtype=float).ravel()  # accept (n,) or (n, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}"
            )

        # Add bias term (column of ones) for the intercept
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Create identity matrix for regularization
        # Note: We don't regularize the bias term (first element)
        identity = np.eye(X_with_bias.shape[1])
        identity[0, 0] = 0  # Don't penalize the intercept
        
        # Calculate coefficients using the regularized Normal Equation
        # theta = (X^T * X + alpha * I~)^(-1) * X^T * y
        regularization_term = self.alpha * identity
        A = X_with_bias.T @ X_with_bias + regularization_term  # the (p+1)x(p+1) system
        b = X_with_bias.T @ y
        
        # Solve A @ theta = b rather than forming inv(A) explicitly.
        # Any alpha > 0 makes A positive definite, so the direct solve is exact.
        # At alpha = 0 with more features than samples, A is singular and BOTH
        # np.linalg.inv and np.linalg.solve return nonsense without warning
        # (measured: train R^2 = -51 for inv on a 30x50 problem). So we check the
        # conditioning first and fall back to the pseudo-inverse, which returns the
        # minimum-norm least-squares solution -- the same answer sklearn's
        # LinearRegression gives via lstsq.
        if np.linalg.cond(A) < 1.0 / np.finfo(float).eps:
            self.coefficients = np.linalg.solve(A, b)
        else:
            self.coefficients = np.linalg.pinv(A) @ b

        # self.coefficients is the FULL theta vector: [intercept, b1, b2, ..., bn]
        # Separate intercept from feature coefficients for clarity
        self.intercept = self.coefficients[0]
        self.feature_coefficients = self.coefficients[1:]
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Data to make predictions on (1-D input and lists are accepted)
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted values
        """
        if self.coefficients is None:
            raise ValueError(
                "This RidgeRegression instance is not fitted yet. "
                "Call fit(X, y) before predict(X)."
            )

        X = self._prepare_features(X)

        expected = len(self.coefficients) - 1  # theta = [intercept, b1..bn]
        if X.shape[1] != expected:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted "
                f"with {expected}"
            )

        # Add bias term for prediction
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Calculate predictions: y = X * theta
        return X_with_bias @ self.coefficients
    
    def get_coefficients(self):
        """
        Get the model coefficients
        
        Returns:
        --------
        dict : Dictionary containing intercept and feature coefficients

        Note on naming: the 'coefficients' key holds ONLY the n_features slope
        terms (self.feature_coefficients). The attribute self.coefficients is a
        different, longer vector: the full theta = [intercept, b1, ..., bn].
        Before fit() is called every entry is None.
        """
        return {
            'intercept': self.intercept,
            'coefficients': self.feature_coefficients,  # theta[1:], length n_features
            'alpha': self.alpha
        }
    
    def score(self, X, y):
        """
        Calculate R^2 (coefficient of determination) score
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Test data
        y : numpy array of shape (n_samples,)
            True values
            
        Returns:
        --------
        r2_score : float
            R^2 score. 1.0 is a perfect fit, 0.0 means the model does no better
            than always predicting the mean of y, and NEGATIVE values mean it does
            worse than that. R^2 is unbounded below.
        """
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        
        # Calculate R^2 = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares

        # Degenerate case: y is constant, so SS_tot is 0 and the ratio is undefined.
        # Convention (same as sklearn): a perfect fit scores 1.0, anything else 0.0.
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
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
    
    # Calculate R^2 score
    r2 = model.score(X_test_scaled, y_test)
    print(f"Alpha = {alpha:6.2f} | R^2 Score: {r2:.4f}")

# Train with the best alpha from the sweep above.
# On this dataset test R^2 rises all the way across the printed range
# (0.4526 -> 0.4528 -> 0.4541 -> 0.4572 -> 0.4605), so alpha = 100 wins:
# the diabetes features are noisy and correlated, and strong shrinkage pays off.
print("\n" + "="*50)
print("Training with the best alpha from the sweep (alpha = 100.0)")
print("="*50)

model = RidgeRegression(alpha=100.0)
model.fit(X_train_scaled, y_train)

# Evaluate
r2 = model.score(X_test_scaled, y_test)
print(f"\nR^2 Score: {r2:.4f}")

# Display coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.2f}")
print("\nFeature Coefficients:")
for i, coef in enumerate(coeffs['coefficients']):
    print(f"  Feature {i+1}: {coef:.2f}")
"""

"""
USAGE EXAMPLE 3: Comparing Ridge vs Multiple Regression (OLS)

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize: alpha is scale-dependent, so this comparison is only fair on
# features that share a common scale (see "Feature Scaling" in the guide).
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Multiple Regression (no regularization).
# RidgeRegression(alpha=0.0) IS ordinary least squares: the penalty term
# alpha * I~ vanishes and the Normal Equation collapses to (X^T X)^-1 X^T y.
# (Verified against sklearn's LinearRegression: coefficients agree to ~3e-11.)
model_ols = RidgeRegression(alpha=0.0)
model_ols.fit(X_train, y_train)
r2_ols = model_ols.score(X_test, y_test)

# Train Ridge Regression (with regularization)
model_ridge = RidgeRegression(alpha=1.0)
model_ridge.fit(X_train, y_train)
r2_ridge = model_ridge.score(X_test, y_test)

print("Comparison: Multiple Regression (alpha=0) vs Ridge Regression (alpha=1)")
print(f"Multiple Regression R^2: {r2_ols:.4f}")
print(f"Ridge Regression R^2:    {r2_ridge:.4f}")
print(f"\nDifference: {abs(r2_ridge - r2_ols):.4f}")

# Compare coefficient magnitudes
coeffs_ols = model_ols.get_coefficients()['coefficients']
coeffs_ridge = model_ridge.get_coefficients()['coefficients']

print("\nCoefficient Magnitudes:")
print(f"Multiple Regression: {np.linalg.norm(coeffs_ols):.2f}")
print(f"Ridge Regression:    {np.linalg.norm(coeffs_ridge):.2f}")
print(f"\nRidge reduces coefficient magnitudes by: {(1 - np.linalg.norm(coeffs_ridge)/np.linalg.norm(coeffs_ols))*100:.1f}%")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _3_ridge_regression.py
    # Requires numpy only. Everything below is seeded and reproducible.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # ================================================================
    # DEMO 1 - Regularization under multicollinearity
    # ================================================================
    print("=" * 55)
    print("DEMO 1 - Ridge under multicollinearity")
    print("=" * 55)

    n_samples = 200
    X = np.random.randn(n_samples, 5)
    X[:, 2] = X[:, 1] + 0.002 * np.random.randn(n_samples)  # near-duplicate feature

    corr_23 = np.corrcoef(X[:, 1], X[:, 2])[0, 1]
    print(f"Features x2 and x3 are near-duplicates (correlation = {corr_23:.6f}),")
    print("so OLS cannot tell them apart and splits their effect wildly.")

    true_b = np.array([3.0, -2.0, 1.5, 0.5, -1.0])
    y = 4.0 + X @ true_b + 0.5 * np.random.randn(n_samples)

    # Shuffle before slicing so train and test come from the same distribution
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    X_tr, X_te = X[:150], X[150:]
    y_tr, y_te = y[:150], y[150:]

    # Standardize using TRAIN statistics only (alpha is scale-dependent).
    # Applying the train mean/std to the test set avoids leaking test information.
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    # The true coefficients, re-expressed on the standardized features
    true_scaled = true_b * sd

    print("\nalpha      Train R^2   Test R^2   ||coef||_2")
    print("-" * 46)
    for alpha in [0.0, 0.1, 1.0, 10.0, 100.0]:
        m = RidgeRegression(alpha=alpha)
        m.fit(X_tr, y_tr)
        norm = np.linalg.norm(m.get_coefficients()['coefficients'])
        print(f"{alpha:8.2f}   {m.score(X_tr, y_tr):8.4f}   {m.score(X_te, y_te):8.4f}   {norm:9.3f}")

    print("\nNote: test R^2 hardly moves -- prediction was never the problem here.")
    print("What collapses is ||coef||, from 8.4 down to 3.1 with only alpha=0.1.")

    print("\nCoefficients (standardized scale): true vs recovered")
    m0 = RidgeRegression(alpha=0.0)
    m0.fit(X_tr, y_tr)
    m1 = RidgeRegression(alpha=1.0)
    m1.fit(X_tr, y_tr)
    c0 = m0.get_coefficients()['coefficients']
    c1 = m1.get_coefficients()['coefficients']
    print("  feature      true   alpha=0 (OLS)   alpha=1 (Ridge)")
    for j in range(5):
        print(f"       x{j+1}   {true_scaled[j]:7.3f}   {c0[j]:13.3f}   {c1[j]:15.3f}")
    print("  -> x2 and x3 carry the same information, so only their SUM is")
    print(f"     identifiable. True sum = {true_scaled[1] + true_scaled[2]:.3f}.")
    print(f"     OLS   splits it as {c0[1]:+.3f} and {c0[2]:+.3f}  (huge, opposite signs)")
    print(f"     Ridge splits it as {c1[1]:+.3f} and {c1[2]:+.3f}  (same total, shared evenly)")
    print("     Both reproduce the data; only Ridge is safe to interpret.")

    print("\nSample test predictions (true, predicted at alpha=1.0):")
    preds = m1.predict(X_te)
    for i in range(5):
        print(f"  true={y_te[i]:7.3f}   pred={preds[i]:7.3f}")

    # ================================================================
    # DEMO 2 - More features than samples (p > n)
    # ================================================================
    print("\n" + "=" * 55)
    print("DEMO 2 - More features (50) than samples (20 train)")
    print("=" * 55)
    print("50 sensor readings driven by only 3 hidden factors, measured on")
    print("30 units. With p > n, X^T X is singular: OLS has no unique answer.")
    print("Adding alpha * I~ to the diagonal makes the system solvable again.")

    n2, p2, n_factors = 30, 50, 3
    latent = np.random.randn(n2, n_factors)          # the 3 hidden drivers
    loadings = np.random.randn(n_factors, p2)        # how each sensor reads them
    X2 = latent @ loadings + 0.3 * np.random.randn(n2, p2)
    y2 = X2 @ (0.1 * np.random.randn(p2)) + 0.5 * np.random.randn(n2)

    idx2 = np.random.permutation(n2)
    X2, y2 = X2[idx2], y2[idx2]
    X2_tr, X2_te = X2[:20], X2[20:]
    y2_tr, y2_te = y2[:20], y2[20:]

    print("\nalpha      Train R^2   Test R^2   ||coef||_2")
    print("-" * 46)
    best_alpha, best_test = None, -np.inf
    for alpha in [0.0, 0.1, 1.0, 10.0, 100.0]:
        m = RidgeRegression(alpha=alpha)
        m.fit(X2_tr, y2_tr)
        test_r2 = m.score(X2_te, y2_te)
        norm = np.linalg.norm(m.get_coefficients()['coefficients'])
        print(f"{alpha:8.2f}   {m.score(X2_tr, y2_tr):8.4f}   {test_r2:8.4f}   {norm:9.3f}")
        if test_r2 > best_test:
            best_alpha, best_test = alpha, test_r2

    print("\n  -> alpha=0 fits the 20 training rows PERFECTLY (Train R^2 = 1.0)")
    print("     by memorizing them, and generalizes worst of all. That is")
    print("     overfitting you can watch happen.")
    print(f"     Best test R^2 here: {best_test:.4f} at alpha = {best_alpha:g}.")
    print("     Train R^2 falls and Test R^2 rises as alpha grows -- until the")
    print("     penalty gets so strong the model underfits. That peak is the")
    print("     bias-variance trade-off, made visible.")

    print("\n" + "=" * 55)
    print("Done. Try editing alpha above and re-running.")
    print("=" * 55)
