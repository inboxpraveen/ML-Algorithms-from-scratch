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

    Core Idea:
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

    Fitting:
        All parameters (trend + seasonality) are estimated simultaneously using
        Ordinary Least Squares (OLS), which has an efficient closed-form solution:

            params = (X^T X)^{-1} X^T y

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
                 yearly_fourier_order=10, weekly_fourier_order=3, changepoint_range=0.8):
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
        """
        self.n_changepoints = n_changepoints
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_fourier_order = yearly_fourier_order
        self.weekly_fourier_order = weekly_fourier_order
        self.changepoint_range = changepoint_range

        # Learned attributes (populated during fit)
        self.params_ = None           # All fitted parameters (OLS solution)
        self.changepoints_t_ = None   # Detected changepoint locations (days)
        self._start_date = None       # Reference date for numeric conversion
        self._t_train = None          # Numeric time values used in training
        self._is_fitted = False

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
            return np.array([(d - self._start_date).days for d in dates], dtype=float)
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

    def fit(self, ds, y):
        """
        Fit Prophet model to a time series.

        Steps performed internally:
        1. Parse dates → numeric time values (days)
        2. Detect changepoints: place n_changepoints uniformly in the first
           changepoint_range fraction of the training period
        3. Build design matrix X = [trend features | seasonality features]
        4. Solve OLS: params = lstsq(X, y)  →  minimizes ||X @ params - y||²

        Parameters:
        -----------
        ds : list of str ('YYYY-MM-DD'), datetime objects, or numbers
            One date per observation. Must be sorted in ascending order.
            - String example: ['2020-01-01', '2020-01-02', '2020-01-03', ...]
            - Numeric example: [0, 1, 2, 3, ...] (day indices)

        y : array-like, shape (n_samples,)
            Observed time series values corresponding to each date in ds.

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

        # Set reference date for numeric conversion
        sample = ds[0]
        if isinstance(sample, str):
            self._start_date = datetime.strptime(sample, '%Y-%m-%d')
        elif hasattr(sample, 'year'):
            self._start_date = sample
        else:
            self._start_date = None

        # Convert dates to numeric (days)
        t = self._parse_dates(ds)
        self._t_train = t.copy()

        # Detect changepoints: uniformly placed in first changepoint_range of data
        t_end = t.min() + self.changepoint_range * (t.max() - t.min())
        t_eligible = t[t <= t_end]
        n_cp = min(self.n_changepoints, max(0, len(t_eligible) - 1))

        if n_cp > 0:
            cp_idx = np.linspace(0, len(t_eligible) - 1, n_cp + 2, dtype=int)[1:-1]
            self.changepoints_t_ = t_eligible[cp_idx]
        else:
            self.changepoints_t_ = np.array([])

        # Record parameter counts for component extraction
        self._n_trend_params = 2 + len(self.changepoints_t_)
        self._n_yearly_params = (2 * self.yearly_fourier_order
                                 if self.yearly_seasonality else 0)
        self._n_weekly_params = (2 * self.weekly_fourier_order
                                 if self.weekly_seasonality else 0)

        # Build design matrix and fit OLS
        X = self._make_design_matrix(t)
        self.params_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

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
            Step size between consecutive future dates:
            - 'D': Daily  (step = 1 day)
            - 'W': Weekly (step = 7 days)
            - 'M': Monthly (step ≈ 30 days)

        Returns:
        --------
        future_dates : list of str ('YYYY-MM-DD') or list of float
            Future dates ready to pass directly to predict() or get_components().

        Example:
        --------
        future = model.make_future_dataframe(periods=365, freq='D')
        forecast = model.predict(future)
        """
        if not self._is_fitted:
            raise ValueError("Call fit() before make_future_dataframe().")

        freq_days = {'D': 1, 'W': 7, 'M': 30}
        step = freq_days.get(freq.upper(), 1)

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
        return 1.0 - ss_res / (ss_tot + 1e-10)

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
========================================
"""

if __name__ == "__main__":
    print("=" * 70)
    print("Prophet - Time Series Forecasting")
    print("Educational Implementation from Scratch")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Example 1: Basic Time Series with Trend + Seasonality
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 1: Daily Sales Data with Trend + Yearly + Weekly Patterns")
    print("=" * 70)

    np.random.seed(42)

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
    model = Prophet(
        n_changepoints=15,
        yearly_seasonality=True,
        weekly_seasonality=True,
        yearly_fourier_order=10,
        weekly_fourier_order=3
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
    print(f"\nDetected {len(model.changepoints_t_)} changepoints")

    # -------------------------------------------------------------------------
    # Example 2: Forecasting Future Values
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 2: Forecasting 90 Days into the Future")
    print("=" * 70)

    # Use first 600 days to train, then forecast the next 90
    train_dates = dates[:600]
    train_sales = sales[:600]
    test_dates  = dates[600:690]
    test_sales  = sales[600:690]

    forecast_model = Prophet(n_changepoints=15, yearly_fourier_order=10,
                             weekly_fourier_order=3)
    forecast_model.fit(train_dates, train_sales)

    # Forecast the held-out 90 days
    forecast = forecast_model.predict(test_dates)

    mae_test  = forecast_model.mae(test_dates, test_sales)
    rmse_test = forecast_model.rmse(test_dates, test_sales)
    r2_test   = forecast_model.score(test_dates, test_sales)

    print(f"Training period: {train_dates[0]} to {train_dates[-1]}  ({len(train_dates)} days)")
    print(f"Forecast period: {test_dates[0]} to {test_dates[-1]}  ({len(test_dates)} days)")
    print(f"\nFirst 10 forecasted vs actual values:")
    print(f"  {'Date':<12} {'Forecast':>10} {'Actual':>10} {'Error':>10}")
    print(f"  {'-'*44}")
    for i in range(10):
        err = forecast[i] - test_sales[i]
        print(f"  {test_dates[i]:<12} {forecast[i]:>10.1f} {test_sales[i]:>10.1f} {err:>+10.1f}")

    print(f"\nForecast performance (90-day horizon):")
    print(f"  MAE  = {mae_test:.2f} units")
    print(f"  RMSE = {rmse_test:.2f} units")
    print(f"  R2   = {r2_test:.4f}")

    # Also demonstrate make_future_dataframe
    future_dates = forecast_model.make_future_dataframe(periods=30, freq='D')
    future_forecast = forecast_model.predict(future_dates)
    print(f"\nNext 30 days beyond training (make_future_dataframe):")
    print(f"  {future_dates[0]}  : predicted = {future_forecast[0]:.1f}")
    print(f"  {future_dates[14]} : predicted = {future_forecast[14]:.1f}")
    print(f"  {future_dates[29]} : predicted = {future_forecast[29]:.1f}")

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
    print(f"\nYearly seasonality peak:   day {peak_day} ({one_year_dates[peak_day]}), "
          f"value = +{yr_comps['yearly'][peak_day]:.1f}")
    print(f"Yearly seasonality trough: day {trough_day} ({one_year_dates[trough_day]}), "
          f"value = {yr_comps['yearly'][trough_day]:.1f}")

    week_dates_num = list(range(7))
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    wk_t = np.array(week_dates_num, dtype=float)
    wk_features = []
    for i in range(1, model.weekly_fourier_order + 1):
        wk_features.append(np.cos(2.0 * np.pi * i * wk_t / 7.0))
        wk_features.append(np.sin(2.0 * np.pi * i * wk_t / 7.0))
    wk_X = np.column_stack(wk_features)
    idx = model._n_trend_params + model._n_yearly_params
    wk_params = model.params_[idx: idx + model._n_weekly_params]
    wk_effect = wk_X @ wk_params
    print(f"\nWeekly effect by day (Mon=0 in this dataset start):")
    for d, name in enumerate(day_names):
        bar = "+" * int(max(0, wk_effect[d])) + "-" * int(max(0, -wk_effect[d]))
        print(f"  {name}: {wk_effect[d]:+6.1f}  {bar[:20]}")

    # -------------------------------------------------------------------------
    # Example 4: Comparing Model Configurations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 4: Comparing Different Prophet Configurations")
    print("=" * 70)

    configs = [
        {"name": "No seasonality (trend only)",
         "params": dict(n_changepoints=15, yearly_seasonality=False,
                        weekly_seasonality=False)},
        {"name": "Yearly only",
         "params": dict(n_changepoints=15, yearly_seasonality=True,
                        weekly_seasonality=False, yearly_fourier_order=10)},
        {"name": "Weekly only",
         "params": dict(n_changepoints=15, yearly_seasonality=False,
                        weekly_seasonality=True, weekly_fourier_order=3)},
        {"name": "Full model (yearly + weekly)",
         "params": dict(n_changepoints=15, yearly_seasonality=True,
                        weekly_seasonality=True, yearly_fourier_order=10,
                        weekly_fourier_order=3)},
        {"name": "Full model, no changepoints",
         "params": dict(n_changepoints=0, yearly_seasonality=True,
                        weekly_seasonality=True)},
    ]

    print(f"\n{'Configuration':<35} {'R2':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 65)
    for cfg in configs:
        m = Prophet(**cfg["params"])
        m.fit(dates, sales)
        r2   = m.score(dates, sales)
        mae  = m.mae(dates, sales)
        rmse = m.rmse(dates, sales)
        print(f"{cfg['name']:<35} {r2:>8.4f} {mae:>8.2f} {rmse:>8.2f}")

    print("\nKey observations:")
    print("  - Adding yearly seasonality gives the biggest R2 boost")
    print("  - Adding weekly seasonality captures day-of-week variation")
    print("  - Changepoints allow the trend to bend (flexible growth)")
    print("  - Full model with changepoints achieves best fit")

    # -------------------------------------------------------------------------
    # Example 5: Retail Sales Simulation (Real-World Scenario)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 5: Retail Sales Simulation (Holiday Spike)")
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
    print(f"\n6-month holdout forecast performance:")
    print(f"  MAE  = {mae_r:.1f} units")
    print(f"  RMSE = {rmse_r:.1f} units  (higher than MAE due to holiday spike errors)")
    print(f"  R2   = {r2_r:.4f}")
    print(f"\nNote: Prophet forecasts the seasonal baseline well, but sharp one-off")
    print(f"holiday spikes require explicit holiday indicators for best accuracy.")

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
       - Minimum ~2 seasonal cycles for yearly seasonality (2+ years of daily data)
       - Minimum ~4 weeks for reliable weekly seasonality
       - Works best with 100+ data points
       - Data must be sorted by date in ascending order

    2. CHOOSING n_changepoints:
       - Default 25 works well for 1-3 years of daily data
       - Use fewer (5-10) for short series or smooth trends
       - Use 0 if you know the trend is strictly linear
       - Too many changepoints: wiggly trend that overfits

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
       - Occasional missing values (OLS handles this naturally)

    7. WHEN TO USE ALTERNATIVES:
       - Very few observations (<50): Consider simple exponential smoothing
       - High-frequency data (hourly/minute): LSTM or ARIMA may be better
       - Complex lagged dependencies: Use ARIMA or VAR models
       - Pure stationarity focus: ARIMA/SARIMA is more appropriate
    """
    print(tips)

    print("\n" + "=" * 70)
    print("COMPARISON: Prophet vs Other Time Series Methods")
    print("=" * 70)

    comparison = """
    Prophet vs ARIMA:
    + Prophet: Handles multiple seasonalities easily (yearly + weekly)
    + Prophet: Automatic changepoint detection, no stationarity requirement
    + Prophet: Interpretable components (trend / seasonality separated)
    - ARIMA: Better for stationary series with complex autocorrelation
    - ARIMA: No external regressors needed (no feature engineering)

    Prophet vs Exponential Smoothing (ETS):
    + Prophet: Explicit trend changepoints (ETS uses smooth exponential)
    + Prophet: Multiple simultaneous seasonalities
    - ETS: Faster, simpler, excellent for short series
    - ETS: Handles multiplicative seasonality more naturally

    Prophet vs LSTM / Deep Learning:
    + Prophet: Interpretable, fast, no GPU needed
    + Prophet: Works well with limited data (hundreds of points)
    - LSTM: Learns complex non-linear patterns automatically
    - LSTM: Better for very high-frequency or multivariate series

    Best Use Cases for Prophet:
    - Business KPIs with clear growth trends (users, revenue, orders)
    - Retail demand with yearly + weekly cycles
    - Website traffic forecasting
    - Energy demand planning
    - Any daily/weekly series with 1+ years of history
    """
    print(comparison)

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
