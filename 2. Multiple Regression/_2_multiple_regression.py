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

    Key Idea: "Fit one hyperplane through the data by solving for every
    coefficient at once, in closed form - no iteration, no learning rate."

    Use Cases:
    - Real estate pricing: price from square footage, bedrooms, age, location score
    - Sales forecasting: revenue from TV / radio / online advertising spend
    - Medical prediction: disease progression from age, BMI, blood pressure
    - Student performance: exam score from study hours, attendance, prior grades

    The Closed-Form Solution (exactly what fit() computes):
        Minimize the squared error   J(theta) = (y - X.theta)^T (y - X.theta)
        Setting dJ/dtheta = 0 gives the Normal Equation
            X^T X theta = X^T y      ->      theta = (X^T X)^-1 X^T y
        The code evaluates this as   theta = pinv(X) @ y   using the
        Moore-Penrose pseudo-inverse of the DESIGN matrix. For a full-column-rank
        X the pseudo-inverse IS (X^T X)^-1 X^T exactly, so the answer is
        identical; when X is rank deficient (duplicate or perfectly collinear
        features, or fewer samples than features) it still returns the
        minimum-norm least-squares solution. Forming and inverting X^T X, by
        contrast, then either raises LinAlgError or - worse - silently returns a
        vector that does not minimize the squared error at all. DEMO 3 in this
        file measures exactly that on a duplicated column: SSE 159.948 from the
        explicit inverse against the true minimum of 89.206 from pinv.

    Attributes set by fit():
        coefficients         : shape (n_features + 1,) -> [b0, b1, ..., bn]
        intercept            : float, the same value as coefficients[0]
        feature_coefficients : shape (n_features,)     -> [b1, ..., bn]

    Simplification vs. canonical OLS:
        This is the ESTIMATION half of ordinary least squares. A full statistical
        package (statsmodels OLS, R's lm) also reports the INFERENTIAL half, which
        is not implemented here: standard errors se(theta) = sqrt(diag(s2 (X^T X)^-1))
        with s2 = SSE / (n - m - 1), the t-statistics and p-values built from them,
        confidence intervals, and adjusted R2 = 1 - (1-R2)(n-1)/(n-m-1). Standard
        errors and adjusted R2 are omitted purely to keep this class the estimation
        half of the method; p-values and confidence intervals additionally need a
        t-distribution CDF, which numpy does not provide (scipy does).
        Practical consequence: you can read WHAT this model learned but not HOW SURE
        it is - a coefficient of 5392 and one of 5392 +/- 40000 look identical
        through get_coefficients(). See "Simplifications vs. canonical OLS" in
        _2_multiple_regression.md.
    """

    def __init__(self):
        """
        Initialize the Multiple Regression model

        This model has no hyperparameters. Ordinary least squares has a single
        exact solution, so there is no learning rate, no iteration count and no
        regularization strength to choose - which is precisely why the Normal
        Equation is such a satisfying place to start.

        Attributes:
        -----------
        coefficients : numpy array of shape (n_features + 1,) or None
            The full parameter vector [b0, b1, ..., bn] after fit().
            Slot 0 is the intercept, so this is ONE longer than n_features.
        intercept : float or None
            The bias term b0 (a copy of coefficients[0]).
        feature_coefficients : numpy array of shape (n_features,) or None
            The per-feature slopes [b1, ..., bn] - self.coefficients with the
            intercept stripped off. This is what get_coefficients() returns
            under the 'coefficients' key.
        """
        self.coefficients = None
        self.intercept = None
        self.feature_coefficients = None

    def fit(self, X, y):
        """
        Train the multiple regression model using the Normal Equation
        
        Parameters:
        -----------
        X : numpy array (or list of lists) of shape (n_samples, n_features)
            Training data with multiple features. A 1-D array of shape
            (n_samples,) is accepted and treated as a single feature column.
        y : numpy array (or list) of shape (n_samples,)
            Target values. A column vector of shape (n_samples, 1) is accepted
            too and is flattened, so predict() always returns shape (n_samples,).

        Returns:
        --------
        self : MultipleRegression
            The fitted model, so that calls can be chained.

        The Normal Equation: θ = (X^T * X)^(-1) * X^T * y

        Precondition:
            That inverse exists only when the design matrix (X plus the bias
            column) has full column rank, which requires n_samples >=
            n_features + 1 AND no perfectly collinear features. This code uses
            the pseudo-inverse rather than an explicit inverse, so a rank
            deficient design still returns a valid (minimum-norm) least-squares
            solution instead of failing - see the class docstring.

        Sets:
            self.coefficients          -> [b0, b1, ..., bn]  (n_features + 1 values)
            self.intercept             -> b0
            self.feature_coefficients  -> [b1, ..., bn]      (n_features values)
        """
        # Accept Python lists and 1-D input; work in float64 throughout
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)          # a single feature given as (n,) -> (n, 1)
        y = np.asarray(y, dtype=float).ravel()   # a (n, 1) target becomes (n,)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}."
            )

        # Add bias term (column of ones) for the intercept
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calculate coefficients using the Normal Equation
        # θ = (X^T * X)^(-1) * X^T * y
        # Evaluated as θ = pinv(X) @ y. For a full-column-rank design matrix
        # pinv(X) IS (X^T X)^(-1) X^T exactly, so this is the same answer; it is
        # simply computed by an SVD instead of by forming and inverting X^T X,
        # which squares the condition number and loses accuracy. On a rank
        # deficient design (duplicate features, or n_features >= n_samples) the
        # explicit inverse either raises LinAlgError or silently returns a
        # vector that is NOT the least-squares solution; the pseudo-inverse
        # always returns one - the minimum-norm one. DEMO 3 measures both.
        self.coefficients = np.linalg.pinv(X_with_bias) @ y

        # Separate intercept from feature coefficients for clarity
        self.intercept = self.coefficients[0]
        self.feature_coefficients = self.coefficients[1:]

        return self

    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : numpy array (or list of lists) of shape (n_samples, n_features)
            Data to make predictions on. A 1-D array is read as a single
            feature column if the model was fitted on one feature, and as a
            single sample otherwise.

        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model is not fitted. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            # (n,) is ambiguous, so resolve it with what the model was fitted on
            X = X.reshape(-1, 1) if len(self.feature_coefficients) == 1 else X.reshape(1, -1)
        if X.shape[1] != len(self.feature_coefficients):
            raise ValueError(
                f"Model was fitted on {len(self.feature_coefficients)} features "
                f"but X has {X.shape[1]}."
            )

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
            'intercept'    -> b0, a single float
            'coefficients' -> [b1, ..., bn], length n_features
            Note the naming: this returns self.feature_coefficients, i.e. the
            slopes WITHOUT the intercept. The attribute self.coefficients is
            the full vector [b0, b1, ..., bn] and is one element longer.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fitted. Call fit(X, y) first.")

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
            1.0 is a perfect prediction, 0.0 means the model does no better
            than always predicting the mean of y, and the score is NEGATIVE
            when the model does worse than that - which is entirely possible
            on unseen data. There is no lower bound.
            If y is constant its variance is zero and R² is undefined (0/0);
            following scikit-learn's convention this returns 1.0 when the
            predictions match and 0.0 otherwise.
        """
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)

        if y.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"X has {y_pred.shape[0]} samples but y has {y.shape[0]}."
            )

        # Calculate R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares

        # Guard the undefined case: a constant y has SS_tot = 0, making R² a 0/0.
        # scikit-learn's convention is 1.0 when the predictions match and 0.0
        # otherwise. The comparison is relative to the scale of y because a
        # floating-point fit of a constant target leaves residuals of order
        # 1e-29 rather than exactly zero.
        if ss_tot == 0:
            return 1.0 if ss_res <= 1e-12 * max(np.sum(y ** 2), 1.0) else 0.0

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

# NOTE: 5 samples for 4 parameters (intercept + 3 features) is a deliberately
# tiny illustration. With only 1 residual degree of freedom and square footage
# nearly collinear with age, the fitted coefficients are NOT stable estimates -
# the age coefficient even comes out positive here. See the __main__ demo below
# for the same problem with 200 houses, where the signs behave as expected.

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

# Calculate R2 score
r2 = model.score(X_test, y_test)
print(f"R2 Score: {r2:.4f}")

# Display coefficients
coeffs = model.get_coefficients()
print(f"\nIntercept: {coeffs['intercept']:.2f}")
print("\nFeature Coefficients:")
for i, coef in enumerate(coeffs['coefficients']):
    print(f"  Feature {i+1}: {coef:.2f}")
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _2_multiple_regression.py
    # Requires numpy only. Everything below is seeded and reproducible.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Demo 1: can the Normal Equation recover coefficients we planted? ---
    print("=" * 55)
    print("DEMO 1 - Recovering known coefficients from noisy data")
    print("=" * 55)

    n_samples, n_features = 500, 4
    true_intercept = 5.0
    true_coefs = np.array([3.0, -2.0, 0.5, 0.0])  # feature 4 is pure noise

    X = np.random.randn(n_samples, n_features)
    y = true_intercept + X @ true_coefs + np.random.randn(n_samples) * 0.5

    # Shuffle BEFORE slicing so train and test come from the same distribution
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    X_train, X_test = X[:400], X[400:]   # disjoint: 400 train, 100 test
    y_train, y_test = y[:400], y[400:]

    model = MultipleRegression()
    model.fit(X_train, y_train)
    coeffs = model.get_coefficients()

    print("Parameter       True    Recovered")
    print(f"  intercept  {true_intercept:8.3f} {coeffs['intercept']:12.3f}")
    for i in range(n_features):
        print(f"  b{i + 1}         {true_coefs[i]:8.3f} "
              f"{coeffs['coefficients'][i]:12.3f}")
    print("(b4 is planted at 0.0 - the model correctly finds no effect)")

    print(f"\nTrain R2 : {model.score(X_train, y_train):.4f}")
    print(f"Test  R2 : {model.score(X_test, y_test):.4f}")
    print("\nSample predictions (true -> predicted):")
    preds = model.predict(X_test)
    for i in range(5):
        print(f"  true={y_test[i]:7.3f}  ->  pred={preds[i]:7.3f}")

    # --- Demo 2: the house-price story with enough rows to be stable ---
    print("\n" + "=" * 55)
    print("DEMO 2 - House prices from square feet, bedrooms, age")
    print("=" * 55)

    n_houses = 200
    sqft = np.random.uniform(800, 4000, n_houses)
    bedrooms = np.random.randint(2, 6, n_houses)
    age = np.random.uniform(0, 50, n_houses)
    X_house = np.column_stack((sqft, bedrooms, age))

    # True pricing rule, plus noise of about +/- $15k
    y_house = (50000 + 150 * sqft + 10000 * bedrooms - 800 * age
               + np.random.randn(n_houses) * 15000)

    idx = np.random.permutation(n_houses)
    X_house, y_house = X_house[idx], y_house[idx]
    X_tr, X_te = X_house[:160], X_house[160:]   # disjoint: 160 train, 40 test
    y_tr, y_te = y_house[:160], y_house[160:]

    house_model = MultipleRegression()
    house_model.fit(X_tr, y_tr)
    hc = house_model.get_coefficients()

    print(f"  Base price (intercept) : {hc['intercept']:12,.0f}   (true 50,000)")
    print(f"  Per square foot        : {hc['coefficients'][0]:12,.2f}   (true 150.00)")
    print(f"  Per bedroom            : {hc['coefficients'][1]:12,.0f}   (true 10,000)")
    print(f"  Per year of age        : {hc['coefficients'][2]:12,.0f}   (true -800)")
    print("The age coefficient is negative -> older houses are worth less,")
    print("which the 5-row example in USAGE EXAMPLE 1 is far too small to show.")

    print(f"\nTrain R2 : {house_model.score(X_tr, y_tr):.4f}")
    print(f"Test  R2 : {house_model.score(X_te, y_te):.4f}")
    print("\nSample predictions (actual vs predicted price):")
    house_preds = house_model.predict(X_te)
    for i in range(3):
        print(f"  {X_te[i, 0]:6.0f} sqft, {int(X_te[i, 1])} bed, {X_te[i, 2]:4.1f} yrs"
              f"  ->  actual {y_te[i]:10,.0f}   predicted {house_preds[i]:10,.0f}")

    # --- Demo 3: what happens when two features are identical? ---
    print("\n" + "=" * 55)
    print("DEMO 3 - Perfectly collinear features (the singular case)")
    print("=" * 55)

    # Column 5 is an exact copy of column 2 from Demo 1
    X_dup = np.hstack((X_train, X_train[:, [1]]))
    dup_model = MultipleRegression()
    dup_model.fit(X_dup, y_train)
    b = dup_model.feature_coefficients

    print("Feature 5 is an exact copy of feature 2, so X^T X is singular and")
    print("infinitely many coefficient vectors give identical predictions.")
    print("The pseudo-inverse picks the minimum-norm one, which splits the")
    print("shared effect evenly between the two copies:")
    print(f"  b2 = {b[1]:7.3f}    b5 = {b[4]:7.3f}    b2 + b5 = {b[1] + b[4]:7.3f}"
          f"   (true -2.000)")
    print(f"  Train R2 : {dup_model.score(X_dup, y_train):.4f}   "
          f"(identical to the {model.score(X_train, y_train):.4f} above)")

    # What the textbook (X^T X)^-1 X^T y would have produced on this same data
    X_bias = np.hstack((np.ones((len(X_dup), 1)), X_dup))
    pinv_sse = np.sum((y_train - dup_model.predict(X_dup)) ** 2)
    try:
        naive = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y_train
        naive_sse = np.sum((y_train - X_bias @ naive) ** 2)
        print("\nThe explicit inverse does not even fail loudly here - it returns")
        print("a vector that is simply not the least-squares solution:")
        print(f"  inv(X^T X) X^T y  ->  SSE {naive_sse:10.3f}")
    except np.linalg.LinAlgError as err:
        # Whether inv raises or silently returns garbage depends on the LAPACK
        # build; both outcomes are failures, and both are why fit() avoids it.
        print("\nThe explicit inverse fails outright here:")
        print(f"  inv(X^T X) X^T y  ->  LinAlgError: {err}")
    print(f"  pinv(X) y         ->  SSE {pinv_sse:10.3f}   (the true minimum)")
    print("\nSame fit, unstable coefficients: that is multicollinearity, and")
    print("that silent wrongness is why fit() uses the pseudo-inverse.")

