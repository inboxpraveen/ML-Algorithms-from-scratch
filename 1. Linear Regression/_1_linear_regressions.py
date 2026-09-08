import numpy as np

class LinearRegression:
    """
    Simple Linear Regression Implementation from Scratch
    
    Linear regression is used when we want to predict a target variable
    using a single feature (independent variable).

    Key Idea: "Fit the straight line that minimises the sum of squared
               vertical distances between the line and the data points."

    Use Cases:
    - Sales forecasting: predict revenue from advertising spend
    - Real estate: predict house price from square footage
    - Medical research: predict disease progression from time since diagnosis
    - Economics: predict GDP growth from investment rate
    - Education: predict test scores from study hours

    Formula: y = b0 + b1*x
    where:
        y = target variable (dependent variable)
        x = independent variable (feature)
        b0 = intercept (bias term)
        b1 = slope (coefficient)

    Closed form for the single-feature case (what the matrix code below
    computes, and what you would work out by hand):
        b1 = sum((x - xbar) * (y - ybar)) / sum((x - xbar)^2)
        b0 = ybar - b1 * xbar
    That is, the slope is cov(x, y) / var(x).

    This file solves it in matrix form via the Normal Equation, which is
    written for one feature here for clarity but generalises unchanged to
    any number of features - that is exactly what algorithm #2 (Multiple
    Linear Regression) builds on. If you do fit several features, note that
    get_coefficients() reports only the first slope; use the attribute
    .feature_coefficients for the full coefficient vector.
    """
    
    def __init__(self):
        """
        Initialize the Linear Regression model

        This model takes no hyperparameters - the Normal Equation has a
        single exact solution, so there is nothing to tune.

        Attributes:
        -----------
        coefficients : numpy array of shape (n_features + 1,), None until fit()
            The full parameter vector [intercept, slope_1, ..., slope_p]
        intercept : float, None until fit()
            The bias term b0, i.e. coefficients[0]
        feature_coefficients : numpy array of shape (n_features,), None until fit()
            The slopes only, i.e. coefficients[1:]. For single-feature
            regression this holds one value, the slope b1.
        """
        self.coefficients = None
        self.intercept = None
        self.feature_coefficients = None

    def fit(self, X, y):
        """
        Train the linear regression model using the Normal Equation
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, 1) or (n_samples,)
            Training data with single feature
            A plain Python list or flat 1-D array is also accepted and is
            reshaped to a column automatically.
        y : numpy array of shape (n_samples,)
            Target values
            
        The Normal Equation: theta = (X^T X)^-1 X^T y

        This is the exact minimiser of the squared-error cost
        J(theta) = sum((y - X*theta)^2). Setting dJ/dtheta = 0 gives
        (X^T X) theta = X^T y, and solving that linear system is the
        whole of "training" for this model - there is no iteration.

        The line below computes that solution as theta = pinv(X) @ y,
        the Moore-Penrose pseudo-inverse of X applied to y. There are two
        reasons not to transcribe the formula literally:

        - inv() needs X^T X to be invertible, and it is not whenever two
          columns are perfectly collinear - a constant column, a
          duplicated column, or more features than samples. There inv()
          either raises LinAlgError or silently returns garbage
          coefficients, while pinv() returns the minimum-norm
          least-squares solution.
        - pinv is applied to X, NOT to X^T X. The identity
          pinv(X^T X) X^T = pinv(X) is exact in algebra, but forming
          X^T X squares the condition number, so pinv's rank cutoff ends
          up testing SQUARED singular values and discards a direction the
          data really had once cond(X) > ~3e7. That threshold is not
          exotic - one feature with a large offset (a meter reading, a
          price in cents, a timestamp) reaches it on its own. Measured on
          x = 100000 + Uniform(0, 100), y = 3*(x - 100000) + 50 + N(0, 1),
          n = 100, full rank, cond(X) = 3.5e8: pinv(X^T X) @ X^T @ y
          scores R^2 ~= 0.001, pinv(X) @ y scores R^2 ~= 0.9999.

        scikit-learn also solves this with an SVD-based least squares on
        X, and goes one step further by centring X and y first, which
        conditions the problem better still. On rank-deficient data our
        fitted values match scikit-learn's, but the coefficient vector
        itself can differ: that centring makes scikit-learn minimise the
        norm of the slopes only, while we include the intercept. Both are
        valid least-squares solutions.
        """
        # Accept plain lists and flat (n_samples,) input, as the docstring promises
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()

        # Add bias term (column of ones) for the intercept
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calculate coefficients using the Normal Equation
        # theta = (X^T X)^-1 X^T y, evaluated as pinv(X) @ y - the same
        # solution, but without ever forming X^T X (see fit's docstring).
        # Deliberately not np.linalg.lstsq(X_with_bias, y, rcond=None):
        # its default cutoff is max(n, p) * eps, so it starts discarding
        # directions above cond(X) ~ 1 / (max(n, p) * eps) - about 5e13
        # for n = 100 - where pinv's fixed 1e-15 cutoff still keeps them.
        # Swept by enlarging the offset in the docstring example above and
        # varying n: at n = 100 lstsq still scores
        # R^2 0.9999 at cond(X) = 3.5e13 and has truncated by 3.5e14, and
        # the onset drops a decade for every 10x in n (n = 1000 truncates
        # by cond 3.4e13, n = 10000 by 3.5e12), while pinv's fixed cutoff
        # does not move with n - it held to cond ~3.5e14 at every n tried.
        self.coefficients = np.linalg.pinv(X_with_bias) @ y

        # Separate intercept from feature coefficient for clarity
        self.intercept = self.coefficients[0]
        self.feature_coefficients = self.coefficients[1:]
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, 1) or (n_samples,)
            Data to make predictions on
            A plain Python list or flat 1-D array is also accepted and is
            reshaped to a column automatically.

        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        # Accept plain lists and flat (n_samples,) input, as the docstring promises
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add bias term for prediction
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calculate predictions: y = X * theta
        return X_with_bias @ self.coefficients
    
    def get_coefficients(self):
        """
        Get the model coefficients
        
        Returns:
        --------
        dict : Dictionary containing intercept and slope

            'intercept'    : float, the bias term b0
            'slope'        : float, the FIRST feature coefficient b1
            'coefficients' : numpy array of every feature coefficient

        For single-feature regression 'slope' is all you need. If you fit
        more than one feature, 'slope' reports only the first one - read
        'coefficients' (or .feature_coefficients) for the full vector.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        return {
            'intercept': self.intercept,
            'slope': self.feature_coefficients[0] if len(self.feature_coefficients) > 0 else None,
            'coefficients': self.feature_coefficients
        }
    
    def score(self, X, y):
        """
        Calculate R^2 (coefficient of determination) score

        Parameters:
        -----------
        X : numpy array of shape (n_samples, 1) or (n_samples,)
            Test data
        y : numpy array of shape (n_samples,)
            True values
            
        Returns:
        --------
        r2_score : float
            R^2 score. 1.0 is a perfect prediction, 0.0 means the model is
            no better than always predicting the mean of y, and the value
            is NEGATIVE when the model is worse than that. R^2 is NOT
            bounded below by 0 - it has no lower limit.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)

        # Calculate R^2 = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares

        # Guard the degenerate case where y is constant: SS_tot is 0 and the
        # ratio would be a division by zero. Follow scikit-learn's convention -
        # a perfect fit of a constant target scores 1.0, anything else 0.0.
        # We compare with a tolerance where scikit-learn tests ss_res == 0
        # exactly, because our solve leaves a tiny floating-point residual on
        # a fit scikit-learn's centred solve gets exactly right: on
        # X = [[1], [2], [3]], y = [5, 5, 5] ours leaves ss_res = 3.9e-29
        # against scikit-learn's 0.0, so an exact test would score that fit
        # 0.0. The tolerance is therefore more forgiving than scikit-learn,
        # and np.allclose's rtol=1e-5 is RELATIVE, so how forgiving scales
        # with the size of y: a constant y = 1e10 predicted as 1.000001e10
        # (out by 10000) still scores 1.0 here and 0.0 in scikit-learn.
        if ss_tot == 0:
            return 1.0 if np.allclose(y_pred, y) else 0.0

        r2_score = 1 - (ss_res / ss_tot)
        return r2_score


"""
USAGE EXAMPLE 1: Simple Linear Regression with Single Feature

import numpy as np

# Sample data: Predicting salary based on years of experience
X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Years of experience
y_train = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000])  # Salary

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([11, 12, 15]).reshape(-1, 1)  # 11, 12, and 15 years of experience
predictions = model.predict(X_test)
print("Predicted salaries:", predictions)

# Get coefficients
coeffs = model.get_coefficients()
print(f"Intercept: ${coeffs['intercept']:.2f}")
print(f"Slope: ${coeffs['slope']:.2f} per year")
"""

"""
USAGE EXAMPLE 2: Using Real Dataset (Diabetes Dataset - Single Feature)

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
data = load_diabetes()
# Use only the first feature (BMI) for simple linear regression
X, y = data.data[:, 2:3], data.target  # Taking only BMI column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = model.score(X_test, y_test)
print(f"R^2 Score: {r2:.4f}")

# Display coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.2f}")
print(f"Slope: {coeffs['slope']:.2f}")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _1_linear_regressions.py
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Demo 1: recover a line we planted exactly ---
    print("=" * 55)
    print("DEMO 1 - Exact recovery: salary = 25000 + 5000 * years")
    print("=" * 55)

    X_years = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    y_salary = np.array([30000, 35000, 40000, 45000, 50000,
                         55000, 60000, 65000, 70000, 75000])

    salary_model = LinearRegression()
    salary_model.fit(X_years, y_salary)
    coeffs = salary_model.get_coefficients()

    print(f"Recovered intercept : {coeffs['intercept']:.2f}   (true 25000.00)")
    print(f"Recovered slope     : {coeffs['slope']:.2f}   (true 5000.00)")
    print(f"Train R2            : {salary_model.score(X_years, y_salary):.4f}")

    # A flat 1-D array works too - fit/predict reshape it for you.
    future = salary_model.predict(np.array([11, 12, 15]))
    print(f"Predictions for 11, 12, 15 years: {np.round(future, 2)}")

    # --- Demo 2: noisy data with a real held-out test split ---
    print("\n" + "=" * 55)
    print("DEMO 2 - Noisy data: y = 3.5x - 2.0 + noise")
    print("=" * 55)

    n = 200
    X_noisy = np.random.uniform(0, 10, n).reshape(-1, 1)
    y_noisy = 3.5 * X_noisy.ravel() - 2.0 + np.random.randn(n) * 1.5

    # Shuffle before slicing so train and test cover the same x range,
    # and slice at the SAME index so the two sets never overlap.
    idx = np.random.permutation(n)
    X_noisy, y_noisy = X_noisy[idx], y_noisy[idx]
    X_train, X_test = X_noisy[:150], X_noisy[150:]
    y_train, y_test = y_noisy[:150], y_noisy[150:]

    noisy_model = LinearRegression()
    noisy_model.fit(X_train, y_train)
    nc = noisy_model.get_coefficients()

    print(f"True   b0=-2.00  b1=3.50")
    print(f"Fitted b0={nc['intercept']:.4f}  b1={nc['slope']:.4f}")
    print(f"Train R2 : {noisy_model.score(X_train, y_train):.4f}")
    print(f"Test  R2 : {noisy_model.score(X_test, y_test):.4f}")

    preds = noisy_model.predict(X_test)
    print("\nSample predictions (x, true, predicted):")
    for i in range(5):
        print(f"  x={X_test[i, 0]:5.2f}  true={y_test[i]:7.2f}  pred={preds[i]:7.2f}")

    # --- Demo 3: what R2 looks like when there is no signal at all ---
    print("\n" + "=" * 55)
    print("DEMO 3 - Sanity check: R2 on pure noise")
    print("=" * 55)

    X_junk = np.random.randn(100, 1)
    y_junk = np.random.randn(100)  # independent of X - nothing to learn

    junk_model = LinearRegression()
    junk_model.fit(X_junk, y_junk)
    print(f"Train R2 on pure noise: {junk_model.score(X_junk, y_junk):.4f}"
          "  (near 0 = no signal)")

    # --- Demo 4: the closed form matches the matrix solution ---
    print("\n" + "=" * 55)
    print("DEMO 4 - Closed form b1 = cov(x,y)/var(x) matches the code")
    print("=" * 55)

    x_flat = X_train.ravel()
    b1_closed = (np.sum((x_flat - x_flat.mean()) * (y_train - y_train.mean()))
                 / np.sum((x_flat - x_flat.mean()) ** 2))
    b0_closed = y_train.mean() - b1_closed * x_flat.mean()

    print(f"Closed form      : b0={b0_closed:.6f}  b1={b1_closed:.6f}")
    print(f"Normal Equation  : b0={nc['intercept']:.6f}  b1={nc['slope']:.6f}")
    print(f"Max difference   : {max(abs(b0_closed - nc['intercept']), abs(b1_closed - nc['slope'])):.2e}")
