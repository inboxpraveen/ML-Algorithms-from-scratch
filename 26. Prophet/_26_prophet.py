import numpy as np
from datetime import datetime, timedelta


class Prophet:
    """
    Prophet Time Series Forecasting - Implementation from Scratch

    Prophet decomposes a time series into interpretable additive components:

        y(t) = trend(t) + seasonality(t) + error

    Where:
        - trend(t):       Piecewise linear growth with automatic changepoints
        - seasonality(t): Fourier series capturing yearly and/or weekly patterns
        - error:          Gaussian noise (residual)

    Key Idea: "A forecast you can read - a bendable trend plus stacked seasonal
    waves, all estimated in one regularized linear solve."

        Instead of treating a time series as a black-box ARIMA process, Prophet
        explicitly models WHAT is happening at each point in time. This makes
        forecasts easy to understand, debug, and explain to stakeholders.

    Trend (Piecewise Linear):
        The trend is linear but can "bend" at changepoints:

            trend(t) = m + k*t + Σ_j δ_j * max(0, t - s_j)

        Where:
            m = intercept, k = slope
            s_j = changepoint location, δ_j = rate change at s_j
            max(0, t - s_j) = hinge function (0 before s_j, linear after)

    Seasonality (Fourier Series):
        Seasonal patterns are modeled as sums of sin/cos waves:

            S(t) = Σ_{n=1}^{N} [a_n * cos(2π*n*t/P) + b_n * sin(2π*n*t/P)]

        Where P = period (365.25 days for yearly, 7 days for weekly)
              N = Fourier order (number of harmonics)

    Fitting (penalized least squares - the Laplace prior on delta):
        All parameters (trend + seasonality) are estimated simultaneously in a
        single linear solve. Canonical Prophet puts a Laplace(0, tau) prior on
        the changepoint rate adjustments delta; in MAP form that is an L1
        penalty applied to the delta block ONLY:

            minimize  0.5 * ||y - X @ theta||^2  +  lam * sum_j |delta_j|
            with      lam = sigma^2 / changepoint_prior_scale

        sigma is the residual scale of the unpenalized fit, and both y and t are
        standardized first (t -> [0, 1] over the training window, y -> y/max|y|)
        so that one value of changepoint_prior_scale means the same thing for a
        2-year series as for a 10-year one. fit() converts the solution back to
        raw "days" units, so params_ is always interpretable as
        [intercept, slope per day, delta per day, ..., Fourier amplitudes].

        The penalty is solved by coordinate descent. Its update is exactly the
        soft-thresholding operator that _17_xgboost.py uses for L1 leaf weights:

            shrink(z, lam) = z - lam   if z >  lam
                           = z + lam   if z < -lam
                           = 0         if |z| <= lam

        The unpenalized columns are not iterated at all: they are profiled out
        first (Frisch-Waugh-Lovell), leaving a plain lasso in delta, and put
        back by least squares at the end. The sweeps then stop on the KKT
        conditions, which for a convex problem PROVE the returned theta is the
        global minimum - see _solve_penalized for both steps and for what
        happens if the sweep budget runs out first. Setting
        changepoint_prior_scale=np.inf switches the penalty off and recovers
        ordinary least squares, params = (X^T X)^{-1} X^T y.

    WHY THE PRIOR IS NOT OPTIONAL:
        With 25 changepoint hinges next to 20 yearly Fourier columns, the design
        matrix is badly collinear. Unpenalized OLS fits the HISTORY just as well
        (in-sample R2 0.9796 vs 0.9788 on this file's demo series) but
        extrapolates nonsense: the 130-day holdout R2 on that same series is
        +0.9102 with the prior and -17.84 without it. The prior is also what
        performs the actual changepoint *selection* - soft-thresholding drives
        most deltas to exactly 0, leaving only the bends the data insists on
        (on that series 23 of the 25 deltas are zeroed and 2 bends survive).
        What the prior does NOT do is guarantee that the surviving bends are
        the real ones: Example 2 in the demo shows the same model, same prior
        and same generative process getting the break right on 900 days of
        history and wrong on 600.

    Simplifications vs. the canonical Prophet library:
        (see the "Simplification vs. canonical Prophet" section of _26_prophet.md)
        - Linear growth only. No logistic/saturating trend, no capacity column.
        - No holiday regressors and no extra user regressors.
        - Point forecasts only: no MCMC posterior, no uncertainty intervals.
        - sigma is estimated once from the unpenalized fit rather than sampled
          jointly with the other parameters as Stan does.
        - Seasonality is left unpenalized (canonical Prophet puts a
          Normal(0, seasonality_prior_scale=10) prior on it, which is nearly
          flat in the standardized units used here).
        - Daily/sub-daily seasonality and multiplicative seasonality are absent.

    Use Cases:
        - Business forecasting: Sales, revenue, user growth
        - Web analytics: Page views, session counts, conversion rates
        - Energy demand: Electricity consumption, solar generation
        - Retail: Inventory planning, demand forecasting
        - Any time series with clear trend and repeating seasonal patterns

    Key Concepts:
        Changepoints:      Breakpoints where the trend rate changes
        Fourier Order:     Number of harmonics (higher = more flexible seasonality)
        Piecewise Linear:  Trend that can change slope at changepoints
        Component Model:   Forecasts decomposed into interpretable parts
    """

    def __init__(self, n_changepoints=25, yearly_seasonality=True, weekly_seasonality=True,
                 yearly_fourier_order=10, weekly_fourier_order=3, changepoint_range=0.8,
                 changepoint_prior_scale=0.05):
        """
        Initialize Prophet model.

        Parameters:
        -----------
        n_changepoints : int, default=25
            Number of potential trend changepoints to place automatically.

            Changepoints are placed uniformly in the first changepoint_range
            fraction of the training data. The model learns how much the
            trend actually changes at each changepoint.

            - 0:     Simple linear trend (no bends at all)
            - 5-10:  Few major direction changes
            - 25:    Default, allows moderate flexibility (recommended)
            - 50+:   Very flexible trend (risk of overfitting short data)

        yearly_seasonality : bool, default=True
            Whether to model yearly seasonal patterns (period = 365.25 days).

            - True:  Captures annual cycles (summer highs, winter lows, holidays)
            - False: Disable if data spans less than 1 year or has no annual cycle

        weekly_seasonality : bool, default=True
            Whether to model weekly seasonal patterns (period = 7 days).

            - True:  Captures weekday vs weekend patterns (e.g., higher traffic Mon-Fri)
            - False: Disable for non-daily data or data without weekly pattern

        yearly_fourier_order : int, default=10
            Number of Fourier harmonics for yearly seasonality.

            Each harmonic adds a sin + cos pair:
            - 3:   Simple, smooth annual curve
            - 10:  Default, good balance of flexibility and smoothness
            - 20+: Complex annual pattern with many sub-yearly fluctuations

        weekly_fourier_order : int, default=3
            Number of Fourier harmonics for weekly seasonality.

            - 2-3: Simple weekly pattern (recommended for most cases)
            - 5+:  Complex day-by-day variation within the week

        changepoint_range : float (0, 1], default=0.8
            Fraction of training history where changepoints can be placed.

            - 0.8:  Changepoints in first 80% (default)
            - 1.0:  Changepoints anywhere in training data
            Keeping < 1.0 prevents overfitting to the end of training data
            where there is less context for the model.

        changepoint_prior_scale : float > 0 (or np.inf), default=0.05
            Scale tau of the Laplace(0, tau) prior on the changepoint rate
            adjustments delta. This is the single most important knob in
            Prophet: it decides how willing the trend is to bend.

            Larger = MORE flexible trend (weaker penalty, more non-zero deltas).
            Smaller = MORE rigid trend (stronger penalty, deltas shrink to 0).

            - 0.001: essentially a straight line; identical to n_changepoints=0
            - 0.05:  Prophet's default, works for most business series
            - 0.5:   very flexible; use when the trend genuinely breaks often
            - np.inf: penalty OFF -> plain OLS. Fits history equally well but
              extrapolates wildly (see "WHY THE PRIOR IS NOT OPTIONAL" above).
            Typical: 0.01 to 0.5. Tune it on a holdout window, never in-sample -
            the in-sample R2 barely moves while the forecast falls apart.
        """
        if not (changepoint_prior_scale > 0):
            raise ValueError("changepoint_prior_scale must be > 0 "
                             "(use np.inf to disable the penalty entirely).")

        self.n_changepoints = n_changepoints
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_fourier_order = yearly_fourier_order
        self.weekly_fourier_order = weekly_fourier_order
        self.changepoint_range = changepoint_range
        self.changepoint_prior_scale = changepoint_prior_scale

        # Learned attributes (populated during fit)
        self.params_ = None           # All fitted parameters (penalized LS solution)
        self.changepoints_t_ = None   # Candidate changepoint locations (days)
        self._start_date = None       # Reference date for numeric conversion
        self._t_train = None          # Numeric time values used in training
        self._is_fitted = False

        # Standardization constants recorded by fit() (see the class docstring)
        self._t_scale = None          # t.max() - t.min(), the training span in days
        self._y_scale = None          # max|y|, Prophet's "absmax" scaling of y
        self._lambda_ = None          # Effective L1 weight actually used

        # Solver diagnostics recorded by _solve_penalized()
        self._n_sweeps_ = None        # Coordinate-descent sweeps actually run
        self._solver_certified_ = None  # True if the KKT conditions were proven

        # Parameter index tracking (for get_components)
        self._n_trend_params = None
        self._n_yearly_params = None
        self._n_weekly_params = None

    def _parse_dates(self, dates):
        """
        Convert dates to numeric values (days since training start date).

        Supports three input formats:
        1. List of 'YYYY-MM-DD' strings  → parsed with strptime
        2. List of Python datetime objects → difference in days
        3. List/array of numbers          → used as-is

        Parameters:
        -----------
        dates : list
            Dates in any supported format

        Returns:
        --------
        t : np.ndarray
            Numeric time array (days from reference point)
        """
        if len(dates) == 0:
            return np.array([], dtype=float)

        sample = dates[0]
        if isinstance(sample, str):
            parsed = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
            return np.array([(d - self._start_date).days for d in parsed], dtype=float)
        elif hasattr(sample, 'year'):  # datetime / date objects
            # Normalize date AND datetime (and pandas Timestamp) down to whole
            # days so that a model fitted on date objects can still be predicted
            # with the 'YYYY-MM-DD' strings that make_future_dataframe() returns.
            parsed = [datetime(d.year, d.month, d.day) for d in dates]
            return np.array([(d - self._start_date).days for d in parsed], dtype=float)
        else:
            return np.asarray(dates, dtype=float)

    def _make_fourier_features(self, t, period, n_terms):
        """
        Create Fourier series features for modeling a seasonal pattern.

        For each harmonic n = 1, 2, ..., n_terms, adds two columns:
            cos(2π * n * t / period)   ← cosine component
            sin(2π * n * t / period)   ← sine component

        Why Fourier series?
            Any smooth repeating function with period P can be written as an
            infinite sum of sin/cos harmonics (Fourier's theorem). With enough
            harmonics (n_terms), we can approximate any seasonal shape —
            whether it's a simple summer peak or a complex multi-modal pattern.

        Visual intuition:
            - Harmonic 1 (n=1): One full wave per period (lowest frequency)
            - Harmonic 2 (n=2): Two waves per period
            - Harmonic N (n=N): N waves per period (finest detail)
            Adding them together lets the model shape the seasonal curve freely.

        Parameters:
        -----------
        t : np.ndarray, shape (n_samples,)
            Numeric time values in days
        period : float
            Duration of one complete seasonal cycle in days
            (365.25 for yearly, 7.0 for weekly)
        n_terms : int
            Number of Fourier harmonics; output has 2 * n_terms columns

        Returns:
        --------
        features : np.ndarray, shape (n_samples, 2 * n_terms)
        """
        features = []
        for i in range(1, n_terms + 1):
            features.append(np.cos(2.0 * np.pi * i * t / period))
            features.append(np.sin(2.0 * np.pi * i * t / period))
        return np.column_stack(features)

    def _make_design_matrix(self, t):
        """
        Build the complete regression design matrix X.

        Column layout:
            [1 | t | max(0,t-s_1) | ... | max(0,t-s_S) | cos1_yr | sin1_yr | ... | cos1_wk | sin1_wk | ...]
             ↑   ↑         ↑ S changepoint hinges            ↑ yearly Fourier terms   ↑ weekly Fourier terms
          bias slope

        The hinge function max(0, t - s) is 0 for all time before changepoint s
        and increases linearly after s. This creates the "bend" in the trend.

        The full model prediction is simply:
            y_hat = X @ params

        Parameters:
        -----------
        t : np.ndarray, shape (n_samples,)
            Numeric time values in days

        Returns:
        --------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix combining trend and seasonality features
        """
        n = len(t)

        # Trend: intercept (ones) + linear slope + one hinge per changepoint
        trend_cols = [np.ones(n), t]
        for s in self.changepoints_t_:
            trend_cols.append(np.maximum(0.0, t - s))
        X = np.column_stack(trend_cols)

        # Yearly seasonality Fourier features
        if self.yearly_seasonality:
            X_yearly = self._make_fourier_features(t, period=365.25,
                                                   n_terms=self.yearly_fourier_order)
            X = np.hstack([X, X_yearly])

        # Weekly seasonality Fourier features
        if self.weekly_seasonality:
            X_weekly = self._make_fourier_features(t, period=7.0,
                                                   n_terms=self.weekly_fourier_order)
            X = np.hstack([X, X_weekly])

        return X

    @staticmethod
    def _soft_threshold(z, lam):
        """
        Soft-thresholding (shrinkage) operator - the L1 proximal step.

            shrink(z, lam) = z - lam   if z >  lam
                           = z + lam   if z < -lam
                           = 0         if |z| <= lam

        This is the same operator _17_xgboost.py applies to its leaf weights.
        It is what turns "L1 penalty" into "exact zeros": a changepoint whose
        evidence z is weaker than the prior's pull lam is switched off entirely,
        which is how Prophet performs changepoint SELECTION rather than just
        changepoint shrinkage.

        Parameters:
        -----------
        z : float
            Unpenalized coordinate-descent numerator (X_j . residual + ...)
        lam : float
            L1 penalty weight

        Returns:
        --------
        float : shrunk value of z
        """
        if z > lam:
            return z - lam
        if z < -lam:
            return z + lam
        return 0.0

    def _solve_penalized(self, X, y, lam, penalized, max_iter=50000, tol=1e-9):
        """
        Minimize  0.5 * ||y - X @ theta||^2 + lam * sum_{j penalized} |theta_j|
        by cyclic coordinate descent - and PROVE that the answer returned is
        the minimum rather than just a point on the way there.

        Two ideas do the work.

        (1) PROFILE OUT THE UNPENALIZED BLOCK (Frisch-Waugh-Lovell).
            Split the columns into the free block A (intercept, base slope,
            Fourier) and the penalized block H (the changepoint hinges). For
            any fixed delta the free coefficients have a closed form - they are
            just least squares of (y - H @ delta) on A. Substituting that back
            leaves an ordinary lasso in delta alone:

                minimize 0.5 * ||y_p - H_p @ delta||^2 + lam * sum_j |delta_j|

            where y_p and H_p are y and the hinge columns with their
            projections onto A removed ("what is left of a hinge once the
            intercept, the slope and the seasonal waves have taken what they
            can explain"). Same optimum, but 25 coordinates instead of 53 and
            far less collinearity, because the worst of it - a slow bend versus
            a slow annual wave - has been projected away. Cycling over the free
            columns can no longer stall the descent.

        (2) STOP ON THE KKT CONDITIONS, not on "nothing moved much".
            Cyclic descent on this problem can crawl for tens of thousands of
            sweeps while each individual step is tiny, so "no coordinate moved
            by more than tol" is not evidence of optimality. Measured on
            Example 1's 600-day holdout fit, a 1000-sweep run of the
            un-profiled version stopped with the objective still wrong in the
            4th decimal (0.128838 vs 0.128798) and 4 changepoints selected
            where the true minimum keeps 2. Instead, once a sweep leaves the
            sign pattern alone, finish the job exactly:
            on the active set S (the deltas that are non-zero, with signs s)
            the objective is smooth and its minimizer solves the small linear
            system

                G[S,S] @ delta_S = c_S - lam * s_S

            If that solution keeps the same signs, and every switched-off
            coordinate obeys |gradient_j| <= lam, then the KKT conditions hold.
            The problem is convex, so satisfied KKT conditions are a PROOF of
            global optimality - not a heuristic. If the signs or the bound fail,
            the support is still changing and the sweeps continue.

        Coordinate descent itself is unchanged. Holding every other coefficient
        fixed, the best delta_j has a closed form; with G = H_p^T H_p and
        c = H_p^T y_p precomputed (so we never touch the n rows again), the
        "partial residual" numerator for column j is

            z_j = c_j - (G[j] . delta) + G[j, j] * delta_j

        and the update is  delta_j = shrink(z_j, lam) / G[j, j].
        delta starts at ZERO, which is the natural warm start when most
        coordinates end up switched off anyway.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Design matrix, already standardized by fit()
        y : np.ndarray, shape (n_samples,)
            Target, already standardized by fit()
        lam : float
            L1 penalty weight (0.0 means "no penalty" -> exact OLS via lstsq)
        penalized : np.ndarray of bool, shape (n_features,)
            True for the columns the L1 penalty applies to (the delta block)
        max_iter : int, default=50000
            Sweep budget. This is a fallback, not the normal exit: all 15 fits
            the demo below performs certify, the worst of them (Example 2's
            600-day panel) at 4808 sweeps and most in under 100. No fit in
            this file reaches the budget, so the uncertified return below is a
            safety valve the demo never exercises.
            If the budget does run out the last iterate is returned anyway and
            _solver_certified_ is set to False, so a caller can always tell
            whether the returned theta is the proven optimum.
        tol : float, default=1e-9
            RELATIVE tolerance on the KKT conditions above, not a step-size
            tolerance: the bounds are checked as |g_j| <= lam + tol * scale,
            with scale = max(lam, max_j |c_j|).

        Returns:
        --------
        theta : np.ndarray, shape (n_features,)
            Penalized least-squares solution
        """
        # No penalty (changepoint_prior_scale=inf) or nothing to penalize
        # (n_changepoints=0): the problem is plain OLS, solve it exactly.
        if lam <= 0.0 or not np.any(penalized):
            self._n_sweeps_ = 0
            self._solver_certified_ = True
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return theta

        free = ~penalized
        A = X[:, free]                # intercept, base slope, Fourier columns
        H = X[:, penalized]           # the delta block (changepoint hinges)

        # (1) Frisch-Waugh-Lovell: regress y and each hinge column on the free
        # block, keep the residuals. lstsq rather than a QR projector, because
        # A can be rank deficient (weekly_fourier_order >= 4 aliases exactly on
        # integer days, and a very short series has fewer rows than free
        # columns); lstsq still projects onto col(A) exactly in that case.
        y_p = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        H_p = H - A @ np.linalg.lstsq(A, H, rcond=None)[0]

        gram = H_p.T @ H_p            # G, shape (S, S)
        corr = H_p.T @ y_p            # c, shape (S,)
        n_delta = H_p.shape[1]
        delta = np.zeros(n_delta)

        # Degenerate case: if the free block already reproduces y, the residual
        # variance sigma^2 - and with it lam = sigma^2 / tau - collapses to
        # roundoff. A lam that far below the gradient scale cannot switch any
        # coordinate off, so the problem is numerically unpenalized; solve it
        # as OLS instead of chasing a KKT bound that lives below machine
        # precision. (A constant series is the obvious way to trigger this.)
        # The 1e-9 ratio is a threshold chosen here, not a derived bound: one
        # decade above it the penalized and the OLS solutions already agree to
        # ~2e-8 relative, so nothing observable is lost by taking this branch.
        if lam <= 1e-9 * float(np.max(np.abs(corr))):
            self._n_sweeps_ = 0
            self._solver_certified_ = True
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return theta

        # A hinge that lies inside col(A) leaves a numerically-zero residual
        # column, so compare its norm against the largest one, not against 0.
        tiny = 1e-12 * float(np.max(np.diag(gram)))

        # Absolute slack allowed in the KKT check below. It is relative to the
        # SIZE OF THE PROBLEM (lam, or the gradient at delta = 0, whichever is
        # bigger) rather than to lam alone: on a series the free block already
        # fits exactly, sigma^2 and therefore lam collapse to ~1e-27 and a
        # lam-relative tolerance would be far below double precision, so no
        # iterate could ever certify.
        atol = tol * max(lam, float(np.max(np.abs(corr))))

        sweeps = 0
        certified = False
        signs_prev = np.full(n_delta, np.nan)   # force one plain sweep first
        for sweeps in range(1, max_iter + 1):
            for j in range(n_delta):
                gjj = gram[j, j]
                if gjj <= tiny:       # hinge with no independent support left
                    delta[j] = 0.0
                    continue
                z_j = corr[j] - gram[j] @ delta + gjj * delta[j]
                delta[j] = self._soft_threshold(z_j, lam) / gjj

            # (2) Try to certify. Only worth attempting once a whole sweep has
            # left the sign pattern (which deltas are on, and in which
            # direction) unchanged - before that the support is still moving.
            signs = np.sign(delta)
            if not np.array_equal(signs, signs_prev):
                signs_prev = signs
                continue

            active = signs != 0.0
            cand = np.zeros(n_delta)
            if np.any(active):
                cand[active] = np.linalg.lstsq(gram[np.ix_(active, active)],
                                               corr[active] - lam * signs[active],
                                               rcond=None)[0]
                if not np.array_equal(np.sign(cand[active]), signs[active]):
                    continue          # a delta wanted to change sign: keep going
            grad = gram @ cand - corr           # d/d delta of the squared-error half
            # Stationary on the active set, and inside the [-lam, lam]
            # subgradient band everywhere it is switched off:
            ok_active = (not np.any(active) or
                         np.max(np.abs(grad[active] + lam * signs[active])) <= atol)
            ok_idle = (not np.any(~active) or
                       np.max(np.abs(grad[~active])) <= lam + atol)
            if ok_active and ok_idle:
                delta = cand
                certified = True
                break

        self._n_sweeps_ = sweeps
        self._solver_certified_ = certified

        # Put back the free block that step (1) profiled away: given the final
        # delta it is an ordinary least-squares fit of what the hinges left.
        theta = np.empty(X.shape[1])
        theta[penalized] = delta
        theta[free] = np.linalg.lstsq(A, y - H @ delta, rcond=None)[0]
        return theta

    def fit(self, ds, y):
        """
        Fit Prophet model to a time series.

        Steps performed internally:
        1. Parse dates → numeric time values (days)
        2. Place candidate changepoints: n_changepoints of them, spread evenly
           over the first changepoint_range fraction of the training period.
           They are CANDIDATES, not detections - step 5 decides which survive.
        3. Build design matrix X = [trend features | seasonality features]
        4. Standardize the way canonical Prophet does, so the prior's units do
           not depend on how long or how large the series is:
               t -> (t - t.min()) / (t.max() - t.min())      [span becomes 1.0]
               y -> y / max|y|                               [absmax scaling]
        5. Solve the MAP problem, an L1 penalty on the delta block only:
               minimize 0.5*||y_s - X_s @ theta||^2 + lam * sum_j |delta_j|
               lam = sigma^2 / changepoint_prior_scale
           where sigma^2 is the residual variance of the unpenalized fit.
           Deltas that soft-threshold to exactly 0 are the changepoints the
           data rejected.
        6. Undo the standardization so params_ is back in raw "per day" units
           and predict()/get_components() can keep using _make_design_matrix().

        Parameters:
        -----------
        ds : list of str ('YYYY-MM-DD'), datetime objects, or numbers
            One date per observation. Must be sorted in ascending order
            (a ValueError is raised if it is not).
            - String example: ['2020-01-01', '2020-01-02', '2020-01-03', ...]
            - Numeric example: [0, 1, 2, 3, ...] (day indices)

        y : array-like, shape (n_samples,)
            Observed time series values corresponding to each date in ds.
            Lists, tuples and 1-D numpy arrays all work.

        Returns:
        --------
        self : Prophet
            Fitted model (enables chaining: Prophet().fit(ds, y).predict(future))
        """
        ds = list(ds)
        y = np.asarray(y, dtype=float)

        if len(ds) != len(y):
            raise ValueError(f"ds and y must be the same length. "
                             f"Got ds={len(ds)}, y={len(y)}.")

        # Set reference date for numeric conversion. datetime.date has no time
        # part, so normalize everything to a whole-day datetime; otherwise a
        # model fitted on date objects cannot be predicted with the strings
        # make_future_dataframe() hands back.
        sample = ds[0]
        if isinstance(sample, str):
            self._start_date = datetime.strptime(sample, '%Y-%m-%d')
        elif hasattr(sample, 'year'):
            self._start_date = datetime(sample.year, sample.month, sample.day)
        else:
            self._start_date = None

        # Convert dates to numeric (days)
        t = self._parse_dates(ds)
        if np.any(np.diff(t) < 0):
            raise ValueError("ds must be sorted in ascending order. "
                             "Sort ds (and y with it) before calling fit().")
        self._t_train = t.copy()

        # Step 2: PLACE candidate changepoints, evenly spread over the first
        # changepoint_range fraction of history. np.round (not truncation) and
        # dropping index 0 matter: max(0, t - t.min()) is an exact duplicate of
        # the linear slope column, which would make X rank deficient.
        t_end = t.min() + self.changepoint_range * (t.max() - t.min())
        t_eligible = t[t <= t_end]
        n_cp = min(self.n_changepoints, max(0, len(t_eligible) - 1))

        if n_cp > 0:
            cp_idx = np.round(np.linspace(0, len(t_eligible) - 1, n_cp + 1)).astype(int)
            self.changepoints_t_ = t_eligible[np.unique(cp_idx[1:])]
        else:
            self.changepoints_t_ = np.array([])

        # Record parameter counts for component extraction
        n_cp_used = len(self.changepoints_t_)
        self._n_trend_params = 2 + n_cp_used
        self._n_yearly_params = (2 * self.yearly_fourier_order
                                 if self.yearly_seasonality else 0)
        self._n_weekly_params = (2 * self.weekly_fourier_order
                                 if self.weekly_seasonality else 0)

        # Step 3: build the design matrix in raw "days" units - exactly the
        # matrix predict() will rebuild later.
        X = self._make_design_matrix(t)

        # Step 4: standardize as canonical Prophet does, so that one value of
        # changepoint_prior_scale means the same thing at any series length or
        # scale. Only the two time-valued blocks are rescaled; the Fourier
        # columns are already bounded in [-1, 1].
        t_start = t.min()
        self._t_scale = float(t.max() - t.min()) or 1.0     # training span, days
        self._y_scale = float(np.max(np.abs(y))) or 1.0     # Prophet "absmax"

        X_std = X.copy()
        X_std[:, 1] = (t - t_start) / self._t_scale                 # slope column
        if n_cp_used > 0:                                           # hinge columns
            X_std[:, 2:2 + n_cp_used] = X[:, 2:2 + n_cp_used] / self._t_scale
        y_std = y / self._y_scale

        # Step 5: MAP solve. lam = sigma^2 / tau is the Laplace prior written as
        # an L1 weight; sigma^2 comes from the unpenalized fit, which is not
        # inflated by a genuine trend break the way a changepoint-free fit is.
        if len(y_std) - X_std.shape[1] >= 1:
            fit_cols = X_std                      # enough rows: use the full model
        else:
            # More columns than observations (e.g. 25 changepoints on 30 days):
            # the full model interpolates, so its residual is 0 and would give
            # lam = 0. Estimate the noise from the changepoint-free model, which
            # is still over-determined.
            fit_cols = np.delete(X_std, np.s_[2:2 + n_cp_used], axis=1)
        theta_ols, _, _, _ = np.linalg.lstsq(fit_cols, y_std, rcond=None)
        dof = max(1, len(y_std) - fit_cols.shape[1])
        sigma2 = float(np.sum((y_std - fit_cols @ theta_ols) ** 2) / dof)

        penalized = np.zeros(X.shape[1], dtype=bool)
        penalized[2:2 + n_cp_used] = True                # the delta block only
        self._lambda_ = (0.0 if np.isinf(self.changepoint_prior_scale)
                         else sigma2 / self.changepoint_prior_scale)
        theta_std = self._solve_penalized(X_std, y_std, self._lambda_, penalized)

        # Step 6: undo the standardization. Substituting
        # (t - t_start)/t_scale back into the trend gives an exact rescaling of
        # the slope and delta columns plus one offset correction on the
        # intercept, so X @ params_ reproduces y_scale * (X_std @ theta_std).
        params = np.empty_like(theta_std)
        params[0] = self._y_scale * (theta_std[0]
                                     - theta_std[1] * t_start / self._t_scale)
        params[1] = self._y_scale * theta_std[1] / self._t_scale
        if n_cp_used > 0:
            params[2:2 + n_cp_used] = (self._y_scale
                                       * theta_std[2:2 + n_cp_used] / self._t_scale)
        params[2 + n_cp_used:] = self._y_scale * theta_std[2 + n_cp_used:]
        self.params_ = params

        self._is_fitted = True
        return self

    def predict(self, future_ds):
        """
        Generate predictions for given dates.

        Works for both in-sample (training) dates and future (out-of-sample) dates.
        The further you forecast into the future, the wider the uncertainty band
        should be conceptually (this implementation returns point forecasts).

        Parameters:
        -----------
        future_ds : list of str, datetime, or numbers
            Dates to forecast. Can include past dates (in-sample fit) or
            future dates beyond the training period.

        Returns:
        --------
        yhat : np.ndarray, shape (n_samples,)
            Predicted values (trend + all seasonality components summed).

        Example:
        --------
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting. Call fit() first.")

        t = self._parse_dates(list(future_ds))
        X = self._make_design_matrix(t)
        return X @ self.params_

    def get_components(self, future_ds):
        """
        Decompose predictions into individual interpretable components.

        This is Prophet's most powerful feature: instead of a single opaque
        forecast, you get separate trend and seasonality curves. You can
        answer questions like:
        - "Is growth slowing down?" (look at trend)
        - "Which season is our peak?" (look at yearly)
        - "Are weekends higher or lower?" (look at weekly)

        Parameters:
        -----------
        future_ds : list of str, datetime, or numbers
            Dates for which to compute components.

        Returns:
        --------
        components : dict
            'trend'  : np.ndarray - piecewise linear trend
            'yearly' : np.ndarray - yearly seasonal component (zeros if disabled)
            'weekly' : np.ndarray - weekly seasonal component (zeros if disabled)
            'yhat'   : np.ndarray - total forecast = trend + yearly + weekly

        Example:
        --------
        comps = model.get_components(all_dates)
        trend = comps['trend']
        yearly = comps['yearly']
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting components.")

        t = self._parse_dates(list(future_ds))
        n = len(t)
        components = {}

        # Trend component
        trend_cols = [np.ones(n), t]
        for s in self.changepoints_t_:
            trend_cols.append(np.maximum(0.0, t - s))
        X_trend = np.column_stack(trend_cols)
        trend_params = self.params_[:self._n_trend_params]
        components['trend'] = X_trend @ trend_params

        # Yearly seasonality component
        idx = self._n_trend_params
        if self.yearly_seasonality and self._n_yearly_params > 0:
            X_yearly = self._make_fourier_features(t, 365.25, self.yearly_fourier_order)
            yearly_params = self.params_[idx: idx + self._n_yearly_params]
            components['yearly'] = X_yearly @ yearly_params
            idx += self._n_yearly_params
        else:
            components['yearly'] = np.zeros(n)

        # Weekly seasonality component
        if self.weekly_seasonality and self._n_weekly_params > 0:
            X_weekly = self._make_fourier_features(t, 7.0, self.weekly_fourier_order)
            weekly_params = self.params_[idx: idx + self._n_weekly_params]
            components['weekly'] = X_weekly @ weekly_params
        else:
            components['weekly'] = np.zeros(n)

        components['yhat'] = (components['trend']
                              + components['yearly']
                              + components['weekly'])
        return components

    def make_future_dataframe(self, periods, freq='D'):
        """
        Create a list of future dates extending beyond the training period.

        Use this to build the input for predict() when you want to forecast
        into the future.

        Parameters:
        -----------
        periods : int
            Number of future time steps to generate.

        freq : str, default='D'
            Step size between consecutive future dates. Case-insensitive;
            anything else raises ValueError.
            - 'D': Daily  (step = 1 day)
            - 'W': Weekly (step = 7 days)
            - 'M': Monthly (step = a FIXED 30 days, not a calendar month)

        Returns:
        --------
        future_dates : list of str ('YYYY-MM-DD') or list of float
            Future dates ready to pass directly to predict() or get_components().

            Caveat for freq='M': the step is a fixed 30 days, so the generated
            dates drift off the calendar by roughly 5 days per half year
            (e.g. 6 steps from 2023-12-31 land on 2024-01-30, 02-29, 03-30,
            04-29, 05-29, 06-28). For calendar-exact month ends, build the date
            list yourself and hand it straight to predict().

        Example:
        --------
        future = model.make_future_dataframe(periods=365, freq='D')
        forecast = model.predict(future)
        """
        if not self._is_fitted:
            raise ValueError("Call fit() before make_future_dataframe().")

        freq_days = {'D': 1, 'W': 7, 'M': 30}
        if freq.upper() not in freq_days:
            raise ValueError(f"Unknown freq {freq!r}. Supported: "
                             f"{sorted(freq_days)} ('M' = fixed 30-day step).")
        step = freq_days[freq.upper()]

        if self._start_date is None:
            # Numeric mode: just extend by step
            last_t = self._t_train.max()
            return [last_t + step * (i + 1) for i in range(periods)]

        # Date mode: extend from last training date
        last_date = self._start_date + timedelta(days=int(self._t_train.max()))
        future_dates = []
        current = last_date
        for _ in range(periods):
            current = current + timedelta(days=step)
            future_dates.append(current.strftime('%Y-%m-%d'))
        return future_dates

    def score(self, ds, y):
        """
        Compute R² (coefficient of determination) on given data.

        R² = 1 - SS_residual / SS_total

        Interpretation:
        - 1.0: Perfect fit
        - 0.0: Predicts the mean of y (no better than a flat line)
        - <0:  Worse than predicting the mean

        Parameters:
        -----------
        ds : list of dates or numbers
        y : array-like of true values

        Returns:
        --------
        r2 : float
        """
        yhat = self.predict(ds)
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot <= 0.0:
            # y is constant: R^2 is undefined (0/0). Follow scikit-learn and
            # report a perfect 1.0 only if the predictions are exact.
            return 1.0 if ss_res <= 1e-12 else 0.0
        return 1.0 - ss_res / ss_tot

    def mae(self, ds, y):
        """
        Compute Mean Absolute Error (MAE).

        MAE = (1/n) * Σ |y_i - ŷ_i|

        Gives the average prediction error in the same units as y.
        Easy to interpret: "On average, predictions are off by X units."

        Parameters:
        -----------
        ds : list of dates or numbers
        y : array-like of true values

        Returns:
        --------
        mae : float
        """
        yhat = self.predict(ds)
        return float(np.mean(np.abs(np.asarray(y, dtype=float) - yhat)))

    def rmse(self, ds, y):
        """
        Compute Root Mean Squared Error (RMSE).

        RMSE = sqrt((1/n) * Σ (y_i - ŷ_i)²)

        Penalizes large errors more heavily than MAE.
        Useful when large forecast errors are especially costly.

        Parameters:
        -----------
        ds : list of dates or numbers
        y : array-like of true values

        Returns:
        --------
        rmse : float
        """
        yhat = self.predict(ds)
        return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - yhat) ** 2)))


"""
========================================
EXAMPLE USAGE
(run this file directly to execute them all)
========================================
"""

if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _26_prophet.py
    # Requires numpy only. Everything below is seeded and reproducible.
    # ----------------------------------------------------------------
    np.random.seed(42)

    print("=" * 70)
    print("Prophet - Time Series Forecasting")
    print("Educational Implementation from Scratch")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Example 1: Basic Time Series with Trend + Seasonality
    #            (recover the planted signal, then forecast a holdout window)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 1: Daily Sales Data with Trend + Yearly + Weekly Patterns")
    print("=" * 70)

    # Generate 2 years of synthetic daily sales data
    n_days = 730
    start = datetime(2022, 1, 1)
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]
    t = np.arange(n_days, dtype=float)

    # True components
    trend      = 100.0 + 0.15 * t                                  # Grows ~55 units over 2 years
    yearly     = 30.0 * np.sin(2 * np.pi * t / 365.25)             # Annual seasonal swing ±30
    weekly     = 15.0 * np.sin(2 * np.pi * t / 7.0)               # Weekly swing ±15
    noise      = np.random.normal(0, 5, n_days)                    # Gaussian noise
    sales      = trend + yearly + weekly + noise

    print(f"Data: {n_days} days from {dates[0]} to {dates[-1]}")
    print(f"Sales range: [{sales.min():.1f}, {sales.max():.1f}]")
    print(f"True trend slope: 0.15 units/day")
    print(f"True yearly amplitude: +/-30 units")
    print(f"True weekly amplitude: +/-15 units")

    # Fit model
    # (these ARE the library defaults, spelled out here for clarity)
    model = Prophet(
        n_changepoints=25,
        yearly_seasonality=True,
        weekly_seasonality=True,
        yearly_fourier_order=10,
        weekly_fourier_order=3,
        changepoint_prior_scale=0.05
    )
    model.fit(dates, sales)

    # In-sample performance
    r2   = model.score(dates, sales)
    mae  = model.mae(dates, sales)
    rmse = model.rmse(dates, sales)

    print(f"\nIn-sample performance:")
    print(f"  R2   = {r2:.4f}  (1.0 = perfect)")
    print(f"  MAE  = {mae:.2f}  (avg absolute error in sales units)")
    print(f"  RMSE = {rmse:.2f}  (penalizes large errors more)")

    # Read the planted values straight back out of params_.
    # Layout: [intercept, slope, delta_1..delta_S, a1_yr, b1_yr, ..., a1_wk, b1_wk, ...]
    n_cp = len(model.changepoints_t_)
    fit_intercept = model.params_[0]
    fit_slope     = model.params_[1]
    deltas        = model.params_[2:2 + n_cp]
    yr_a, yr_b    = model.params_[model._n_trend_params],     model.params_[model._n_trend_params + 1]
    wk_i          = model._n_trend_params + model._n_yearly_params
    wk_a, wk_b    = model.params_[wk_i], model.params_[wk_i + 1]
    # Amplitude of harmonic 1 is sqrt(a^2 + b^2) because a*cos + b*sin is one
    # shifted wave of that size.
    print(f"\nRecovered vs planted (this is the known-answer test):")
    print(f"  {'quantity':<26} {'recovered':>10} {'planted':>10}")
    print(f"  {'-' * 48}")
    print(f"  {'intercept (day 0)':<26} {fit_intercept:>10.3f} {100.0:>10.3f}")
    print(f"  {'trend slope (units/day)':<26} {fit_slope:>10.4f} {0.15:>10.4f}")
    print(f"  {'yearly amplitude':<26} {np.hypot(yr_a, yr_b):>10.3f} {30.0:>10.3f}")
    print(f"  {'weekly amplitude':<26} {np.hypot(wk_a, wk_b):>10.3f} {15.0:>10.3f}")
    print(f"\nPlaced {n_cp} candidate changepoints; the Laplace prior kept only "
          f"{int(np.sum(np.abs(deltas) > 0))} of them")
    print(f"  (the rest were soft-thresholded to exactly 0 -> no bend there)")
    print(f"  Solver: {model._n_sweeps_} coordinate-descent sweeps, then the KKT")
    print(f"  conditions were checked and hold exactly "
          f"(certified={model._solver_certified_}). That matters: the SUPPORT")
    print(f"  above is only meaningful if the solve actually reached the")
    print(f"  minimum - stop it early and you get a different set of bends.")

    # --- Same example, out-of-sample: a chronological 600 / 130 holdout ------
    # In-sample R2 tells you almost nothing about a forecaster (see Example 2),
    # so always keep a block of the FUTURE back and score on that.
    print("\n" + "-" * 70)
    print("Holdout check: train on the first 600 days, forecast the last 130")
    print("-" * 70)

    train_dates = dates[:600]
    train_sales = sales[:600]
    test_dates  = dates[600:]
    test_sales  = sales[600:]

    forecast_model = Prophet(n_changepoints=25, yearly_fourier_order=10,
                             weekly_fourier_order=3)
    forecast_model.fit(train_dates, train_sales)

    # Forecast the held-out days
    forecast = forecast_model.predict(test_dates)

    mae_test  = forecast_model.mae(test_dates, test_sales)
    rmse_test = forecast_model.rmse(test_dates, test_sales)
    r2_test   = forecast_model.score(test_dates, test_sales)

    # Ceiling: what an ORACLE that knew the true generative components would
    # score on the same window. It is not 1.0 because the noise is unpredictable.
    oracle = (trend + yearly + weekly)[600:]
    ss_res = np.sum((test_sales - oracle) ** 2)
    ss_tot = np.sum((test_sales - test_sales.mean()) ** 2)
    r2_oracle = 1.0 - ss_res / ss_tot

    print(f"Training period: {train_dates[0]} to {train_dates[-1]}  ({len(train_dates)} days)")
    print(f"Forecast period: {test_dates[0]} to {test_dates[-1]}  ({len(test_dates)} days)")
    print(f"\nFirst 10 forecasted vs actual values:")
    print(f"  {'Date':<12} {'Forecast':>10} {'Actual':>10} {'Error':>10}")
    print(f"  {'-'*44}")
    for i in range(10):
        err = forecast[i] - test_sales[i]
        print(f"  {test_dates[i]:<12} {forecast[i]:>10.1f} {test_sales[i]:>10.1f} {err:>+10.1f}")

    print(f"\nForecast performance (130-day horizon):")
    print(f"  MAE  = {mae_test:.2f} units")
    print(f"  RMSE = {rmse_test:.2f} units")
    print(f"  R2   = {r2_test:.4f}")
    print(f"  ORACLE R2 (knowing the true components) = {r2_oracle:.4f}  <- the ceiling")
    print(f"  The noise std of this series is 5.0, so an RMSE near 5 is as good")
    print(f"  as any forecaster can possibly do here.")

    # Also demonstrate make_future_dataframe
    future_dates = forecast_model.make_future_dataframe(periods=30, freq='D')
    future_forecast = forecast_model.predict(future_dates)
    print(f"\nNext 30 days beyond training (make_future_dataframe):")
    print(f"  {future_dates[0]}  : predicted = {future_forecast[0]:.1f}")
    print(f"  {future_dates[14]} : predicted = {future_forecast[14]:.1f}")
    print(f"  {future_dates[29]} : predicted = {future_forecast[29]:.1f}")

    # -------------------------------------------------------------------------
    # Example 2: Why Automatic Changepoints Need a Prior
    #            (and why they also need enough history)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 2: Why Automatic Changepoints Need a Prior")
    print("=" * 70)

    # Plant a REAL trend break: +0.15 per day for a year, then -0.10 per day.
    # Three years of history, so the SAME series can be refitted on a shorter
    # window in panel B below.
    np.random.seed(7)
    n_b = 1095
    t_b = np.arange(n_b, dtype=float)
    break_day = 365.0
    true_slope_before, true_slope_after = 0.15, -0.10
    trend_b = (100.0 + true_slope_before * t_b
               + (true_slope_after - true_slope_before) * np.maximum(0.0, t_b - break_day))
    dates_b = [(start + timedelta(days=int(i))).strftime('%Y-%m-%d') for i in t_b]
    y_b = (trend_b
           + 20.0 * np.sin(2 * np.pi * t_b / 365.25)
           + 10.0 * np.sin(2 * np.pi * t_b / 7.0)
           + np.random.normal(0, 4, n_b))

    print(f"Planted trend: {true_slope_before:+.2f} units/day until day {int(break_day)}, "
          f"then {true_slope_after:+.2f} units/day")
    print(f"Series: {n_b} days (three yearly cycles), yearly swing +/-20, "
          f"weekly +/-10, noise std 4.")

    def changepoint_panel(n_train, n_test):
        """Fit four configurations on days [0, n_train) and score the next
        n_test days. Returns {config name: (in-R2, holdout R2, end slope,
        bends kept)}, so the commentary underneath can quote measured numbers
        instead of asserting them.

        The last configuration is an ORACLE: it is handed a single candidate
        changepoint sitting exactly on the planted break (n_changepoints=1 plus
        a changepoint_range that makes the last eligible day the break day).
        Nobody has that information in real life; it is here as the reference
        the automatic grid is trying to find."""
        cr_oracle = break_day / (n_train - 1)      # last eligible day = break
        configs_cp = [
            ("n_changepoints=0 (straight line)", dict(n_changepoints=0)),
            ("25 cps, prior 0.05 (default)", dict(n_changepoints=25)),
            ("25 cps, prior OFF (plain OLS)",
             dict(n_changepoints=25, changepoint_prior_scale=np.inf)),
            (f"ORACLE: 1 cp planted on day {int(break_day)}",
             dict(n_changepoints=1, changepoint_range=cr_oracle)),
        ]
        tr_b, te_b = dates_b[:n_train], dates_b[n_train:n_train + n_test]
        ytr_b, yte_b = y_b[:n_train], y_b[n_train:n_train + n_test]
        print(f"\n  Train on days 0-{n_train - 1} "
              f"({n_train / 365.25:.1f} yearly cycles), "
              f"forecast days {n_train}-{n_train + n_test - 1}")
        print(f"  {'Configuration':<34} {'in-R2':>8} {'hold-R2':>10} "
              f"{'end slope':>10} {'bends':>7}")
        print(f"  {'-' * 72}")
        out = {}
        default_model = None
        for name, kw in configs_cp:
            m_cp = Prophet(yearly_fourier_order=10, weekly_fourier_order=3, **kw)
            m_cp.fit(tr_b, ytr_b)
            s_cp = len(m_cp.changepoints_t_)
            # Slope at the end of history = base slope + every delta that fired
            end_slope = m_cp.params_[1] + np.sum(m_cp.params_[2:2 + s_cp])
            n_bends = int(np.sum(np.abs(m_cp.params_[2:2 + s_cp]) > 0))
            in_r2, hold_r2 = m_cp.score(tr_b, ytr_b), m_cp.score(te_b, yte_b)
            print(f"  {name:<34} {in_r2:>8.4f} {hold_r2:>10.4f} "
                  f"{end_slope:>+10.4f} {n_bends:>4d}/{s_cp:<2d}")
            out[name] = (in_r2, hold_r2, end_slope, n_bends)
            if kw == dict(n_changepoints=25):
                default_model = m_cp
        s_cp = len(default_model.changepoints_t_)
        kept = default_model.changepoints_t_[
            np.abs(default_model.params_[2:2 + s_cp]) > 0]
        print(f"  Surviving bends (default model): days {kept}   "
              f"true break = day {int(break_day)}")
        print(f"  Solver: certified={default_model._solver_certified_} "
              f"(KKT conditions proven) after {default_model._n_sweeps_} sweeps")
        out['kept'] = kept
        return out

    print("\n  --- Panel A: 900 days of history (1.5 years of it AFTER the "
          "break) ---")
    a = changepoint_panel(900, 195)
    prior_a = a["25 cps, prior 0.05 (default)"]
    ols_a = a["25 cps, prior OFF (plain OLS)"]
    print(f"\n  (true end-of-history slope = {true_slope_after:+.2f})")
    print("  - Without changepoints the trend cannot bend at all, so it splits")
    print("    the difference and the forecast drifts the wrong way.")
    print(f"  - With the prior OFF, 25 collinear hinges fit the history a hair "
          f"better")
    print(f"    ({ols_a[0]:.4f} vs {prior_a[0]:.4f} in-sample) and then extrapolate "
          f"much worse")
    print(f"    ({ols_a[1]:+.4f} vs {prior_a[1]:+.4f} on the holdout).")
    print(f"  - With the Laplace prior, {25 - prior_a[3]} of the 25 deltas are "
          f"thresholded to exactly 0,")
    print(f"    the {prior_a[3]} survivors straddle the real break at day "
          f"{int(break_day)}, and the")
    print(f"    extrapolated slope is {prior_a[2]:+.4f} against a true "
          f"{true_slope_after:+.2f}.")
    print(f"  - That matches the ORACLE row ({prior_a[1]:+.4f} vs "
          f"{a[f'ORACLE: 1 cp planted on day {int(break_day)}'][1]:+.4f}), so searching")
    print(f"    25 candidates cost essentially nothing here. Panel B is where")
    print(f"    that stops being true.")

    print("\n  --- Panel B: the SAME series, but only the first 600 days ---")
    b = changepoint_panel(600, 130)
    prior_b = b["25 cps, prior 0.05 (default)"]
    oracle_b = b[f"ORACLE: 1 cp planted on day {int(break_day)}"]
    print(f"\n  Nothing changed except how much history the model was given, and")
    print(f"  now the automatic grid forecasts badly too ({prior_b[1]:+.4f}); the prior")
    print(f"  is merely the least bad of the three. Note what did NOT happen:")
    print(f"  this is not a solver failure. All four fits are certified optima")
    print(f"  of their own objectives. It is an IDENTIFIABILITY failure.")
    print(f"  Compare the last two rows. Told exactly where to bend, the model")
    print(f"  forecasts at {oracle_b[1]:+.4f}; left to pick from 25 candidates it gets")
    print(f"  {prior_b[1]:+.4f} - and in-sample the two are indistinguishable "
          f"({oracle_b[0]:.4f}")
    print(f"  vs {prior_b[0]:.4f}). Over 1.6 yearly cycles, with 20 unpenalised yearly")
    print(f"  Fourier columns free to move, 'the trend bent down at day "
          f"{int(break_day)}' and")
    print(f"  'the annual wave is lower and later' are nearly the same shape.")
    print(f"  Forcing the 25-candidate model to spend its one bend on the")
    print(f"  candidate nearest the break costs just +2.4e-04 of standardised")
    print(f"  objective (0.2%) - and that 0.2% is the difference between a")
    print(f"  holdout of +0.75 and one of {prior_b[1]:+.2f}. That +2.4e-04 is the one")
    print(f"  FIXED number here, not recomputed above: it needs a refit at the")
    print(f"  25-candidate model's lam with every delta but the day-364 hinge")
    print(f"  pinned to 0, and fit() always recomputes lam from the design it is")
    print(f"  handed. The +0.75 is the ORACLE row above.")
    print(f"  Panel A has 2.5 cycles, which breaks the tie.")
    print(f"  Practical rule: a changepoint prior cannot rescue a series too")
    print(f"  short to separate a bend from a seasonal wave. Get more history,")
    print(f"  or constrain the seasonality (lower yearly_fourier_order, or")
    print(f"  yearly_seasonality=False), before trusting an automatic bend.")

    # -------------------------------------------------------------------------
    # Example 3: Decomposing Components (Trend + Yearly + Weekly)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 3: Component Decomposition")
    print("=" * 70)

    comps = model.get_components(dates)

    trend_vals  = comps['trend']
    yearly_vals = comps['yearly']
    weekly_vals = comps['weekly']
    yhat_vals   = comps['yhat']

    print("Component magnitudes (ranges across all 730 days):")
    print(f"  Trend   : [{trend_vals.min():.1f}, {trend_vals.max():.1f}]  "
          f"  range = {trend_vals.max() - trend_vals.min():.1f}")
    print(f"  Yearly  : [{yearly_vals.min():.1f}, {yearly_vals.max():.1f}]  "
          f"  amplitude ~= {(yearly_vals.max() - yearly_vals.min()) / 2:.1f}")
    print(f"  Weekly  : [{weekly_vals.min():.1f}, {weekly_vals.max():.1f}]  "
          f"  amplitude ~= {(weekly_vals.max() - weekly_vals.min()) / 2:.1f}")

    # Show how the trend changes over time (estimated slope changes)
    print(f"\nTrend evolution (sample points):")
    checkpoints = [0, 182, 365, 547, 729]
    for cp in checkpoints:
        print(f"  Day {cp:3d} ({dates[cp]}): trend = {trend_vals[cp]:.1f}")

    # Verify components sum to total
    max_diff = np.max(np.abs(yhat_vals - (trend_vals + yearly_vals + weekly_vals)))
    print(f"\nVerification: max |yhat - (trend + yearly + weekly)| = {max_diff:.10f}  (OK)")

    # Peak season analysis
    yearly_day_idx = np.arange(365)
    start_date_obj = datetime(2022, 1, 1)
    one_year_dates = [(start_date_obj + timedelta(days=i)).strftime('%Y-%m-%d')
                      for i in range(365)]
    yr_comps = model.get_components(one_year_dates)
    peak_day = np.argmax(yr_comps['yearly'])
    trough_day = np.argmin(yr_comps['yearly'])
    # The planted signal is 30*sin(2*pi*t/365.25), so its peak is a quarter of a
    # period in: 365.25/4 = 91.3 days after the start.
    print(f"\nYearly seasonality peak:   day {peak_day} ({one_year_dates[peak_day]}), "
          f"value = +{yr_comps['yearly'][peak_day]:.1f}")
    print(f"Yearly seasonality trough: day {trough_day} ({one_year_dates[trough_day]}), "
          f"value = {yr_comps['yearly'][trough_day]:.1f}")
    print(f"  planted peak:   day 91  value +30.0   (365.25/4 into the cycle)")
    print(f"  planted trough: day 274 value -30.0   (3*365.25/4 into the cycle)")

    # Weekly effect: ask get_components for seven consecutive days instead of
    # rebuilding the Fourier basis by hand. Day names must come from the DATA -
    # the series starts on 2022-01-01, which is a Saturday, not a Monday.
    week_dates = [(start + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(7)]
    day_names = [(start + timedelta(days=d)).strftime('%a') for d in range(7)]
    wk_effect = model.get_components(week_dates)['weekly']
    print(f"\nWeekly effect by day (series starts {start.strftime('%A')} {dates[0]}):")
    for d, name in enumerate(day_names):
        bar = "+" * int(max(0, wk_effect[d])) + "-" * int(max(0, -wk_effect[d]))
        print(f"  {name}: {wk_effect[d]:+6.1f}  {bar[:20]}")

    # How much does each seasonal block actually buy? Compute it, do not assert it.
    print(f"\nContribution of each block (in-sample R2 on the same 730 days):")
    seasonal_configs = [
        ("Trend only (no seasonality)",
         dict(yearly_seasonality=False, weekly_seasonality=False)),
        ("Trend + yearly", dict(yearly_seasonality=True, weekly_seasonality=False)),
        ("Trend + weekly", dict(yearly_seasonality=False, weekly_seasonality=True)),
        ("Full model", dict(yearly_seasonality=True, weekly_seasonality=True)),
    ]
    block_r2 = {}
    for name, kw in seasonal_configs:
        m_blk = Prophet(n_changepoints=25, yearly_fourier_order=10,
                        weekly_fourier_order=3, **kw)
        m_blk.fit(dates, sales)
        block_r2[name] = m_blk.score(dates, sales)
        print(f"  {name:<28} R2 = {block_r2[name]:.4f}")
    gain_yearly = block_r2["Trend + yearly"] - block_r2["Trend only (no seasonality)"]
    gain_weekly = block_r2["Trend + weekly"] - block_r2["Trend only (no seasonality)"]
    bigger = "weekly" if gain_weekly > gain_yearly else "yearly"
    print(f"  -> yearly adds {gain_yearly:+.4f} R2, weekly adds {gain_weekly:+.4f} R2, "
          f"so {bigger} wins here")
    print(f"     (the weekly swing has a 7-day period, so it explains far more")
    print(f"      day-to-day variance than a single slow annual wave)")

    # -------------------------------------------------------------------------
    # Example 4: Retail Sales Simulation (Real-World Scenario)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 4: Retail Sales Simulation (Holiday Spike)")
    print("=" * 70)

    np.random.seed(0)
    n_days_retail = 365 * 3  # 3 years
    start_retail = datetime(2021, 1, 1)
    retail_dates = [(start_retail + timedelta(days=i)).strftime('%Y-%m-%d')
                    for i in range(n_days_retail)]
    t_retail = np.arange(n_days_retail, dtype=float)

    # Components of retail sales
    base_trend    = 500 + 0.20 * t_retail                         # Growing base
    yearly_retail = 120 * np.sin(2 * np.pi * (t_retail - 60) / 365.25)  # Peak in April
    weekly_retail = 60 * np.sin(2 * np.pi * t_retail / 7.0 + np.pi)     # Weekends higher
    holiday_spike = np.zeros(n_days_retail)

    # Add Black Friday / Christmas spikes each year
    for year_offset in [0, 365, 730]:
        # ~Nov 25 = day 328 of year
        bf_day = year_offset + 328
        if bf_day + 5 < n_days_retail:
            holiday_spike[bf_day: bf_day + 5] += 300
        # Dec 15-25 = days 348-358
        xmas_start = year_offset + 348
        if xmas_start + 10 < n_days_retail:
            holiday_spike[xmas_start: xmas_start + 10] += 200

    noise_retail = np.random.normal(0, 15, n_days_retail)
    retail_sales = base_trend + yearly_retail + weekly_retail + holiday_spike + noise_retail

    # Train on first 2.5 years, forecast last 0.5 year
    split = int(2.5 * 365)
    train_rd = retail_dates[:split]
    train_rs = retail_sales[:split]
    test_rd  = retail_dates[split:]
    test_rs  = retail_sales[split:]

    retail_model = Prophet(
        n_changepoints=20,
        yearly_seasonality=True,
        weekly_seasonality=True,
        yearly_fourier_order=10,
        weekly_fourier_order=3,
        changepoint_range=0.8
    )
    retail_model.fit(train_rd, train_rs)

    forecast_retail = retail_model.predict(test_rd)
    mae_r  = retail_model.mae(test_rd, test_rs)
    rmse_r = retail_model.rmse(test_rd, test_rs)
    r2_r   = retail_model.score(test_rd, test_rs)

    print(f"Retail simulation: {n_days_retail} days  |  Train: {split}  |  Test: {len(test_rd)}")
    print(f"Sales range: [{retail_sales.min():.0f}, {retail_sales.max():.0f}]  "
          f"(holiday spikes to {retail_sales.max():.0f})")
    print(f"\nIn-sample R2 (2.5-year train window) = "
          f"{retail_model.score(train_rd, train_rs):.4f}")
    print(f"\n6-month holdout forecast performance:")
    print(f"  MAE  = {mae_r:.1f} units")
    print(f"  RMSE = {rmse_r:.1f} units  (higher than MAE due to holiday spike errors)")
    print(f"  R2   = {r2_r:.4f}")

    print(f"\nNote: Prophet forecasts the seasonal baseline well, but sharp one-off")
    print(f"holiday spikes require explicit holiday indicators for best accuracy.")

    # Quantify that claim instead of just asserting it: split the holdout error
    # into the days that carry an engineered holiday spike and the days that do
    # not. This implementation has no holiday regressors (see the .md).
    spike_mask = holiday_spike[split:] > 0
    abs_err = np.abs(test_rs - forecast_retail)
    print(f"  MAE on the {int(np.sum(~spike_mask)):3d} ordinary days : {abs_err[~spike_mask].mean():6.1f}")
    print(f"  MAE on the {int(np.sum(spike_mask)):3d} spike days    : {abs_err[spike_mask].mean():6.1f}")

    comps_test = retail_model.get_components(test_rd)
    print(f"\nComponent ranges in forecast period:")
    print(f"  Trend  : [{comps_test['trend'].min():.0f}, {comps_test['trend'].max():.0f}]")
    print(f"  Yearly : [{comps_test['yearly'].min():.0f}, {comps_test['yearly'].max():.0f}]")
    print(f"  Weekly : [{comps_test['weekly'].min():.0f}, {comps_test['weekly'].max():.0f}]")

    # -------------------------------------------------------------------------
    # Practical Tips
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRACTICAL TIPS FOR USING PROPHET")
    print("=" * 70)

    tips = """
    1. DATA REQUIREMENTS:
       - Minimum ~2 seasonal cycles for yearly seasonality (2+ years of daily data).
         Under that, a trend bend and a shift in the annual wave look the same and
         automatic changepoints land in the wrong place - Example 2 measures it
       - Minimum ~4 weeks for reliable weekly seasonality
       - Works best with 100+ data points
       - Data must be sorted by date in ascending order

    2. CHOOSING n_changepoints:
       - Default 25 works well for 1-3 years of daily data
       - Use fewer (5-10) for short series or smooth trends
       - Use 0 if you know the trend is strictly linear
       - Too many changepoints: wiggly trend that overfits - except that the
         Laplace prior below switches off the ones the data does not support

    2b. CHOOSING changepoint_prior_scale (the important one):
       - 0.05 is the default and a good starting point
       - Larger -> more flexible trend, smaller -> more rigid trend
       - Tune it on a HOLDOUT window; in-sample R2 barely moves while the
         forecast quality swings wildly (Example 2 shows this)
       - np.inf turns the penalty off completely -> plain OLS -> wild
         extrapolation. Only useful as a teaching contrast.

    3. SEASONALITY SETTINGS:
       - yearly_fourier_order=10 is good for most annual patterns
       - Increase to 15-20 for very complex intra-year patterns
       - Set yearly_seasonality=False if data is < 1 year
       - weekly_fourier_order=3 works for most day-of-week patterns

    4. CHANGEPOINT_RANGE:
       - Default 0.8 prevents overfitting at the end of training
       - Use 1.0 if you want changepoints up to the final day
       - Keep 0.8 unless you have a strong reason to change it

    5. FORECASTING HORIZON:
       - Short-term (days/weeks): High accuracy, model captures recent patterns
       - Medium-term (months): Good for seasonal patterns, trend extrapolates
       - Long-term (years): Uncertainty grows; trend direction is the key signal

    6. WHEN PROPHET WORKS WELL:
       - Multiple years of daily/weekly data
       - Clear upward/downward growth trend
       - Strong seasonal patterns (yearly, weekly)
       - Occasional missing dates (gaps are handled naturally)

    7. WHEN TO USE ALTERNATIVES:
       - Very few observations (<50): Consider simple exponential smoothing
       - High-frequency data (hourly/minute): LSTM or ARIMA may be better
       - Complex lagged dependencies: Use ARIMA or VAR models
       - Pure stationarity focus: ARIMA/SARIMA is more appropriate
    """
    print(tips)
    print("    (Prophet vs ARIMA / ETS / LSTM: see the 'Prophet vs Other")
    print("     Methods' comparison tables in _26_prophet.md)")

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
