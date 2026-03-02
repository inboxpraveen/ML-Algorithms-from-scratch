# Prophet - Time Series Forecasting

## Overview

**Prophet** is a time series forecasting model developed by Facebook (Meta) that decomposes a time series into interpretable components. Rather than treating a time series as a black-box process, Prophet explicitly models **what** is happening at each point in time.

### Key Concept

Imagine you run an e-commerce store. Your daily sales data has three obvious patterns:
1. A **long-term growth trend** — sales have been increasing year over year
2. A **yearly seasonal pattern** — sales spike around holidays and dip in January
3. A **weekly pattern** — sales are higher on weekdays than weekends

Prophet models these three components separately and adds them together:

```
y(t) = trend(t) + seasonality(t) + error
```

This is called an **additive decomposition model**. The result is not just an accurate forecast, but a fully interpretable one — you can see *exactly* how much each component contributes to any given prediction.

### The Prophet Story

Think of it like a detective analysing crime rates in a city:
- **Trend**: Is the overall crime rate going up or down?
- **Yearly seasonality**: Does crime spike in summer?
- **Weekly seasonality**: Is it worse on weekends?

By separating these signals, you understand *why* the numbers look the way they do — and can forecast more reliably because each component is modelled on its own terms.

## When to Use Prophet

### Perfect For:
- **Business forecasting**: Sales, revenue, active users, order volumes
- **Web analytics**: Page views, click-through rates, session counts
- **Energy demand**: Electricity consumption, solar/wind generation
- **Retail planning**: Inventory management, demand forecasting
- **Any daily/weekly time series** with a clear trend and repeating seasonal patterns

### Real-World Applications:
- **E-commerce**: Forecast daily orders for supply chain planning
- **SaaS products**: Predict monthly active users for capacity planning
- **Media**: Forecast website traffic to allocate server resources
- **Finance**: Model seasonal patterns in revenue streams
- **Healthcare**: Forecast hospital admissions with weekday patterns
- **IoT / Energy**: Predict power demand with hourly or daily data

### When NOT to Use Prophet:
- Very short series (< 50 data points)
- High-frequency data (sub-hourly) with complex lags → LSTM / ARIMA
- Stationary series with no trend or seasonality → ARIMA
- Multi-variable forecasting with strong cross-variable dependencies → VAR

## Mathematical Foundation

### 1. The Additive Model

Prophet's full model is:

```
y(t) = trend(t) + S_yearly(t) + S_weekly(t) + ε
```

Where:
- **y(t)**: Observed value at time t
- **trend(t)**: Piecewise linear trend
- **S_yearly(t)**: Yearly seasonal component (Fourier series)
- **S_weekly(t)**: Weekly seasonal component (Fourier series)
- **ε**: Gaussian noise (error term)

All components are estimated simultaneously by solving a single linear system.

---

### 2. Trend — Piecewise Linear with Changepoints

The trend is **linear but can change slope** at designated "changepoints":

```
trend(t) = m + k·t + Σ_j δ_j · max(0, t - s_j)
```

Where:
- **m**: Intercept (baseline level at t = 0)
- **k**: Base growth rate (slope of the initial segment)
- **s_j**: Location of changepoint j (in days)
- **δ_j**: Rate change at changepoint j (positive = accelerating, negative = decelerating)
- **max(0, t - s_j)**: Hinge function — equals 0 before s_j, then grows linearly after

#### Why Hinge Functions?

A hinge function `max(0, t - s)` is:
- **0** for all time before changepoint s (no effect)
- **Linearly increasing** after changepoint s (adds a new "bend")

Adding S hinge functions allows the trend to change direction at S different points. The model learns the **magnitude** of each bend (δ_j) from data.

#### Visual Example

```
Sales
 │         /← changepoint: growth accelerates
 │        /
 │   ____/
 │  /
 │ /← changepoint: growth slows here
 │/
 └──────────────────── Time
```

#### Changepoint Detection

Changepoints are placed **uniformly** in the first `changepoint_range` fraction of training data (default: 80%). The model then learns which changepoints matter (large δ) and which are irrelevant (δ ≈ 0).

---

### 3. Seasonality — Fourier Series

Seasonal patterns are modelled using **Fourier series** — sums of sine and cosine waves:

```
S(t) = Σ_{n=1}^{N} [ a_n · cos(2π·n·t/P) + b_n · sin(2π·n·t/P) ]
```

Where:
- **P**: Period of the seasonality (365.25 days for yearly, 7 days for weekly)
- **N**: Fourier order (number of harmonics)
- **a_n, b_n**: Amplitudes of each harmonic (learned from data)

#### Why Fourier Series?

By **Fourier's theorem**, any smooth repeating function can be expressed as a sum of sine and cosine waves. With enough harmonics, you can approximate any seasonal shape:

| Harmonics (N) | Captures |
|---|---|
| N = 1 | Simple single-peak annual curve |
| N = 3 | Medium complexity (e.g., summer + winter peaks) |
| N = 10 | Fine detail (multiple sub-annual fluctuations) |
| N = 20 | Very complex shape (risk of overfitting) |

#### Feature Matrix

For yearly seasonality with N = 3 Fourier terms, the features for time t are:

```
[cos(2πt/365.25), sin(2πt/365.25),
 cos(4πt/365.25), sin(4πt/365.25),
 cos(6πt/365.25), sin(6πt/365.25)]
```

This gives 6 columns in the design matrix. Each column is a basis function; the model learns the weights that best fit the observed seasonal pattern.

---

### 4. Fitting — Ordinary Least Squares (OLS)

Once the design matrix X is built by stacking all features (trend + seasonality), the full model is simply:

```
y = X · θ + ε
```

Where θ is the vector of all parameters to estimate. OLS finds θ by minimising the sum of squared errors:

```
θ* = argmin_θ ||y - X·θ||²
```

This has the exact closed-form solution:

```
θ* = (X^T X)^{-1} X^T y
```

In practice, we use `numpy.linalg.lstsq` which is numerically stable.

#### Why OLS Works Here

Because all components are **additive and linear in the parameters**, the entire model reduces to a single linear regression. This makes Prophet:
- **Fast** to fit (one matrix solve, no iterations needed)
- **Stable** (convex optimisation, no local minima)
- **Transparent** (every parameter has a clear geometric interpretation)

---

### 5. Full Design Matrix Layout

```
X = [1 | t | max(0,t-s₁) | ... | max(0,t-sₛ) | cos₁_yr | sin₁_yr | ... | cos₁_wk | sin₁_wk | ...]
     ↑   ↑         ↑ S changepoint hinges          ↑ yearly Fourier       ↑ weekly Fourier
   bias slope
```

| Block | Columns | Parameters |
|---|---|---|
| Trend | 2 + S | intercept, slope, S changepoint rates |
| Yearly seasonality | 2·N_yr | Fourier amplitudes for annual cycle |
| Weekly seasonality | 2·N_wk | Fourier amplitudes for weekly cycle |

For default settings (S=25, N_yr=10, N_wk=3): **total = 2 + 25 + 20 + 6 = 53 parameters**.

## Algorithm Steps

### Step 1: Parse Dates

Convert date strings or datetime objects to numeric values (days since start):

```python
# '2022-01-15' → 14 (days from '2022-01-01')
t = [(date - start_date).days for date in dates]
```

### Step 2: Detect Changepoints

Place changepoints uniformly in the first `changepoint_range` fraction of training data:

```python
t_eligible = t[t <= t.min() + changepoint_range * (t.max() - t.min())]
changepoints = np.linspace(t_eligible[0], t_eligible[-1], n_changepoints)
```

### Step 3: Build Design Matrix

```python
# Trend features
X = [ones, t, max(0, t-s₁), max(0, t-s₂), ..., max(0, t-sₛ)]

# Yearly seasonality (Fourier features, period = 365.25)
for n in 1..N_yr:
    X.append(cos(2π·n·t/365.25))
    X.append(sin(2π·n·t/365.25))

# Weekly seasonality (Fourier features, period = 7)
for n in 1..N_wk:
    X.append(cos(2π·n·t/7))
    X.append(sin(2π·n·t/7))
```

### Step 4: Fit Parameters with OLS

```python
params = np.linalg.lstsq(X, y, rcond=None)[0]
# params = [m, k, δ₁, ..., δₛ, a₁_yr, b₁_yr, ..., a₁_wk, b₁_wk, ...]
```

### Step 5: Predict and Decompose

```python
# Full forecast
y_hat = X_future @ params

# Extract trend component only
y_trend = X_trend @ params[:n_trend_params]

# Extract yearly seasonality only
y_yearly = X_yearly @ params[n_trend_params : n_trend_params + n_yearly_params]
```

## Parameters Explained

### n_changepoints

Controls how many potential "bends" the trend can have.

**Low (0–5):**
- Simple, nearly linear trend
- Good for short series (< 1 year) or very smooth growth
- Less risk of overfitting

**Medium (10–25, default):**
- Allows a few significant direction changes
- Recommended starting point for 1–3 years of data

**High (50+):**
- Very flexible, wiggly trend
- Risk of overfitting — the trend can memorise noise
- Use with caution; validate on held-out data

**Rule of thumb:** n_changepoints = 15–25 for daily data spanning 1–3 years.

---

### yearly_fourier_order

Controls the complexity of the annual seasonal shape.

| Value | Captures | Use When |
|---|---|---|
| 3–5 | Simple single summer/winter peak | Short history or known simple pattern |
| 10 (default) | Moderate detail | Most business time series |
| 15–20 | Fine sub-annual fluctuations | Complex retail calendar, energy demand |

**Warning:** Very high orders can overfit short data. Stick to 10 unless you have 3+ years.

---

### weekly_fourier_order

Controls the shape of the weekly seasonal pattern.

| Value | Captures |
|---|---|
| 2–3 (default) | Basic weekday vs weekend difference |
| 4–5 | Distinct variation across all 7 days |

**Note:** For data with no weekly pattern (e.g., monthly observations), set `weekly_seasonality=False`.

---

### changepoint_range

Fraction of training data where changepoints can occur.

- **0.8 (default):** Changepoints only in first 80%; prevents overfitting at the training tail
- **1.0:** Changepoints up to the last day of training
- **0.5:** Very conservative; only first half can bend

**Recommendation:** Keep at 0.8 unless you observe known structural changes near the end of training.

## Code Example

```python
import numpy as np
from _26_prophet import Prophet
from datetime import datetime, timedelta

# --- 1. Prepare data ---
# Generate 2 years of daily data
n_days = 730
start = datetime(2022, 1, 1)
dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]
t = np.arange(n_days, dtype=float)

# Synthetic sales: trend + seasonal patterns + noise
np.random.seed(42)
y = (100 + 0.15 * t                          # upward trend
     + 30 * np.sin(2 * np.pi * t / 365.25)   # yearly peak in summer
     + 15 * np.sin(2 * np.pi * t / 7.0)      # weekly cycle
     + np.random.normal(0, 5, n_days))        # noise

# --- 2. Fit model ---
model = Prophet(
    n_changepoints=15,
    yearly_seasonality=True,
    weekly_seasonality=True,
    yearly_fourier_order=10,
    weekly_fourier_order=3
)
model.fit(dates, y)

# --- 3. In-sample evaluation ---
print(f"R²   = {model.score(dates, y):.4f}")
print(f"MAE  = {model.mae(dates, y):.2f}")
print(f"RMSE = {model.rmse(dates, y):.2f}")

# --- 4. Forecast 90 days ahead ---
future = model.make_future_dataframe(periods=90, freq='D')
forecast = model.predict(future)
print(f"\nForecast for next 90 days: {forecast[:5].round(1)}")

# --- 5. Decompose components ---
comps = model.get_components(dates)
print(f"\nTrend range:   [{comps['trend'].min():.1f}, {comps['trend'].max():.1f}]")
print(f"Yearly range:  [{comps['yearly'].min():.1f}, {comps['yearly'].max():.1f}]")
print(f"Weekly range:  [{comps['weekly'].min():.1f}, {comps['weekly'].max():.1f}]")
```

## Practical Use Cases

### 1. E-commerce Sales Forecasting

```python
# Forecast daily orders for the next quarter
model = Prophet(n_changepoints=20, yearly_fourier_order=10, weekly_fourier_order=3)
model.fit(historical_dates, daily_orders)

future = model.make_future_dataframe(periods=90, freq='D')
forecast = model.predict(future)

# Use forecast for inventory and staffing decisions
print(f"Predicted peak day (next 90): {future[forecast.argmax()]}")
print(f"Average daily orders next month: {forecast[:30].mean():.0f}")
```

### 2. Web Traffic Forecasting

```python
# Forecast page views to plan server capacity
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    yearly_fourier_order=8,
    weekly_fourier_order=3,
    n_changepoints=25
)
model.fit(dates, page_views)

# Check if weekly pattern shows weekday vs weekend difference
comps = model.get_components(dates[-14:])
print("Weekly effect last 2 weeks:", comps['weekly'].round(0))
```

### 3. Detecting Trend Changes

```python
# Check where significant trend changes happened
model = Prophet(n_changepoints=25)
model.fit(dates, revenue)

# Get trend component to see inflection points
comps = model.get_components(dates)
trend = comps['trend']

# Compute trend velocity (rate of change)
trend_velocity = np.diff(trend)
major_changes = np.where(np.abs(trend_velocity) > trend_velocity.std() * 2)[0]
print("Major trend shifts at days:", major_changes)
```

### 4. Monthly Forecasting (Aggregated Data)

```python
# For monthly data: disable weekly seasonality, keep yearly
import numpy as np
from _26_prophet import Prophet

# Monthly revenue over 5 years (60 data points)
monthly_dates = [f"{2019 + i // 12}-{(i % 12) + 1:02d}-01" for i in range(60)]
monthly_revenue = np.array([...])  # your monthly revenue values

model = Prophet(
    n_changepoints=10,          # Fewer changepoints for monthly data
    yearly_seasonality=True,    # Capture annual cycle
    weekly_seasonality=False,   # No weekly pattern in monthly data
    yearly_fourier_order=5      # Simpler seasonality for 12 points/year
)
model.fit(monthly_dates, monthly_revenue)

# Forecast next 12 months
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)
```

### 5. Comparing Forecast vs Actual (Post-Hoc Analysis)

```python
# Train on first 80% of data, evaluate on last 20%
n = len(dates)
split = int(0.8 * n)

train_dates, test_dates = dates[:split], dates[split:]
train_y, test_y = y[:split], y[split:]

model = Prophet(n_changepoints=20)
model.fit(train_dates, train_y)

forecast = model.predict(test_dates)

# Error metrics
mae  = model.mae(test_dates, test_y)
rmse = model.rmse(test_dates, test_y)
r2   = model.score(test_dates, test_y)

print(f"Holdout MAE:  {mae:.2f}")
print(f"Holdout RMSE: {rmse:.2f}")
print(f"Holdout R²:   {r2:.4f}")
```

## Evaluation Metrics

### 1. R² (Coefficient of Determination)

```
R² = 1 - SS_residual / SS_total
   = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
```

**Interpretation:**
- **1.0**: Perfect predictions
- **0.5**: Model explains 50% of variance
- **0.0**: No better than predicting the mean
- **< 0**: Worse than the mean (model is broken)

**Usage:** `model.score(dates, y)`

---

### 2. MAE (Mean Absolute Error)

```
MAE = (1/n) · Σ |y_i - ŷ_i|
```

**Interpretation:** Average absolute error in the **same units as y**.
If sales are in dollars, MAE = 120 means "predictions are off by $120 on average."

**Usage:** `model.mae(dates, y)`

---

### 3. RMSE (Root Mean Squared Error)

```
RMSE = √[ (1/n) · Σ (y_i - ŷ_i)² ]
```

**Interpretation:** Similar to MAE but **penalises large errors more heavily**.
A few very bad forecasts will raise RMSE much more than MAE.

**Usage:** `model.rmse(dates, y)`

---

### 4. Choosing the Right Metric

| Scenario | Recommended Metric |
|---|---|
| Large errors are equally bad as small errors | MAE |
| Large errors are very costly (e.g., financial) | RMSE |
| Comparing models with different scales | R² |
| All three give consistent info | All three |

---

### 5. Time Series Cross-Validation (Best Practice)

For rigorous evaluation, use **expanding window cross-validation**:

```python
def prophet_cv(dates, y, model_params, n_folds=5, test_size=30):
    """Simple time series cross-validation."""
    results = []
    n = len(dates)
    min_train = n - n_folds * test_size

    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_end  = train_end + test_size

        m = Prophet(**model_params)
        m.fit(dates[:train_end], y[:train_end])

        mae = m.mae(dates[train_end:test_end], y[train_end:test_end])
        results.append(mae)
        print(f"Fold {fold+1}: train={train_end}, test={test_size}, MAE={mae:.2f}")

    print(f"\nAverage CV MAE: {np.mean(results):.2f} ± {np.std(results):.2f}")
    return results
```

## Common Issues and Solutions

### Issue 1: Trend is Too Wiggly / Overfits

**Symptom:** Trend oscillates up and down, doesn't look like real growth.

**Causes:**
- Too many changepoints for the length of data
- Very noisy data

**Solutions:**
```python
# Reduce number of changepoints
model = Prophet(n_changepoints=5)   # down from default 25

# Or disable changepoints entirely for linear trend
model = Prophet(n_changepoints=0)
```

---

### Issue 2: Flat or Missing Seasonality

**Symptom:** Model ignores obvious seasonal patterns.

**Causes:**
- Not enough data (< 1 full seasonal cycle)
- Seasonality disabled

**Solutions:**
```python
# Ensure you have at least 2 years for yearly seasonality
# Enable both seasonalities explicitly
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)

# For very strong but simple annual patterns, try lower fourier order
model = Prophet(yearly_fourier_order=5)
```

---

### Issue 3: Weekly Pattern Looks Wrong

**Symptom:** Weekly seasonality doesn't match day-of-week intuition.

**Causes:**
- Data is not daily (weekly/monthly)
- Weekly pattern is non-standard

**Solutions:**
```python
# Disable weekly if data is not daily
model = Prophet(yearly_seasonality=True, weekly_seasonality=False)

# Or increase Fourier order for more complex weekly patterns
model = Prophet(weekly_fourier_order=5)
```

---

### Issue 4: Poor Forecast at Training Tail

**Symptom:** The trend curves unrealistically near the end of training data.

**Cause:** Changepoints placed too close to the end (overfitting the training tail).

**Solution:**
```python
# Reduce changepoint_range (default 0.8 is usually good)
model = Prophet(changepoint_range=0.7)
```

---

### Issue 5: Sudden Spikes Not Captured

**Symptom:** Forecast misses sharp one-off spikes (Black Friday, product launch, etc.).

**Cause:** Prophet models smooth patterns; one-off spikes are noise to it.

**Solution:**
When you know holidays or events in advance, create a binary indicator feature and include it:

```python
# Create a holiday indicator (1 on holiday, 0 otherwise)
holiday_indicator = np.zeros(n_days)
holiday_indicator[328] = 1   # Black Friday

# Stack it manually into your pipeline
# (This implementation can be extended to support external regressors)
```

For production use, Facebook Prophet's official library supports a `holidays` DataFrame directly.

---

### Issue 6: Negative Forecasts for Counts/Volumes

**Symptom:** Model predicts negative values for metrics that should always be non-negative (e.g., orders, users).

**Cause:** Additive model can go below zero when trend or seasonality are low.

**Solutions:**
- Clip: `forecast = np.maximum(0, model.predict(future))`
- Use log-transform: fit on `log(y + 1)`, exponentiate after predicting

## Tips for Success

### 1. Always Plot Your Components

After fitting, inspect the decomposed components before trusting the forecast:

```python
comps = model.get_components(all_dates)

print("Trend range:   ", comps['trend'].min(), "to", comps['trend'].max())
print("Yearly range:  ", comps['yearly'].min(), "to", comps['yearly'].max())
print("Weekly range:  ", comps['weekly'].min(), "to", comps['weekly'].max())
```

Ask yourself:
- Does the trend make business sense?
- Is the seasonal amplitude reasonable?
- Are the components in the right direction?

### 2. Start with Defaults, Then Tune

```python
# Good first model — works for most daily business time series
model = Prophet(
    n_changepoints=25,         # Default
    yearly_seasonality=True,   # Default
    weekly_seasonality=True,   # Default
    yearly_fourier_order=10,   # Default
    weekly_fourier_order=3     # Default
)
```

Only tune after evaluating this baseline on held-out data.

### 3. Use a Proper Holdout Set

```python
# Always evaluate on data the model has NOT seen
split = int(0.85 * len(dates))
model.fit(dates[:split], y[:split])
print("Test MAE:", model.mae(dates[split:], y[split:]))
```

### 4. Check for Data Quality Issues

Before fitting, verify:
- Are there any missing dates? (OLS handles missing data if dates are explicit)
- Are there any outliers / impossible values? (Check for negative counts, extreme spikes)
- Is the series sorted chronologically?

```python
# Quick data sanity check
y = np.asarray(sales)
print(f"Min: {y.min():.1f}, Max: {y.max():.1f}")
print(f"NaN count: {np.isnan(y).sum()}")
print(f"Negative count: {(y < 0).sum()}")
```

### 5. Understand the Forecast Horizon

| Horizon | Reliability | Notes |
|---|---|---|
| 1–7 days | Very high | Seasonal patterns are clear |
| 1–4 weeks | High | Weekly patterns hold well |
| 1–3 months | Moderate | Trend extrapolation dominates |
| 6–12 months | Lower | Seasonal patterns still informative, trend uncertainty grows |
| 1+ year | Use with caution | Structural changes likely, model has not seen them |

## Prophet vs Other Methods

### Prophet vs ARIMA

| Aspect | Prophet | ARIMA |
|---|---|---|
| **Model type** | Additive decomposition | Differenced autoregressive |
| **Stationarity** | Not required | Required (must difference data) |
| **Seasonality** | Multiple, automatic | One, manual (SARIMA) |
| **Trend** | Piecewise, flexible | Linear or differenced |
| **Interpretability** | ✓ Clear components | ✗ Lag coefficients only |
| **Missing data** | ✓ Handles naturally | ✗ Requires imputation |
| **Fitting speed** | Fast (OLS) | Fast |
| **Best for** | Business KPIs with trend + seasonality | Stationary or simple trending series |

### Prophet vs Exponential Smoothing (ETS)

| Aspect | Prophet | ETS |
|---|---|---|
| **Trend** | Piecewise linear, changepoints | Smooth exponential |
| **Seasonality** | Fourier, multiple periods | One additive or multiplicative |
| **Parameters** | Many (OLS) | Few (MLE) |
| **Short data** | ✗ Needs 100+ points | ✓ Works with 20–30 points |
| **Interpretability** | ✓ Explicit components | Moderate |
| **Best for** | Long, multi-seasonal series | Short series, single seasonality |

### Prophet vs LSTM / Deep Learning

| Aspect | Prophet | LSTM |
|---|---|---|
| **Data needed** | 100s of points | 1000s of points |
| **Training time** | < 1 second | Minutes to hours |
| **Interpretability** | ✓ Full decomposition | ✗ Black box |
| **GPU required** | No | Beneficial |
| **Non-linearity** | ✗ Linear components | ✓ Learns any pattern |
| **Best for** | Business forecasting, limited data | High-frequency, complex patterns |

### When to Use Each

- **Prophet**: Daily/weekly business metrics, multiple seasonalities, trend with bends, need for interpretability
- **ARIMA**: Stationary series, univariate, focus on autocorrelation structure
- **ETS**: Short series, single clear seasonality, fast baseline
- **LSTM**: High-frequency data, multi-variate, very large datasets

## Advanced Topics

### 1. Multiplicative Seasonality

The additive model (`y = trend + seasonality`) works when seasonal swings are roughly **constant in magnitude** regardless of the trend level.

If seasonal swings **grow proportionally with the trend** (e.g., 10% higher in summer always), use log-transform:

```python
import numpy as np

# Log-transform before fitting
y_log = np.log1p(y)   # log(1 + y) — handles zeros safely

model = Prophet(yearly_seasonality=True)
model.fit(dates, y_log)

# Predict and back-transform
forecast_log = model.predict(future)
forecast = np.expm1(forecast_log)   # exp(x) - 1
```

### 2. External Regressors (Custom Features)

This implementation focuses on core Prophet components. To add external features (e.g., a holiday binary flag, a marketing spend variable), you can extend the design matrix:

```python
import numpy as np

# Subclass Prophet and override _make_design_matrix
class ProphetWithRegressors(Prophet):
    def fit(self, ds, y, X_extra=None):
        self._X_extra_train = X_extra
        return super().fit(ds, y)  # base class builds X, then we'll augment

    # Alternatively, manually concatenate extra features to X before lstsq
```

### 3. Uncertainty Intervals

The current implementation returns point forecasts. To add prediction intervals, you can:
- Bootstrap residuals and re-predict
- Use the OLS covariance matrix for analytical confidence intervals:

```python
# Analytical OLS confidence interval (simplified)
X = model._make_design_matrix(t)
residuals = y - X @ model.params_
sigma2 = np.sum(residuals**2) / (len(y) - len(model.params_))
cov = sigma2 * np.linalg.pinv(X.T @ X)

# Prediction standard error for new point x_new:
# se = sqrt(x_new^T @ cov @ x_new)
```

### 4. Trend Saturation (Logistic Growth)

For metrics with natural capacity limits (e.g., market share, user adoption), a logistic (S-curve) trend is more realistic than unbounded linear growth. This requires replacing the linear trend formula with a logistic function — not implemented here but a natural extension.

### 5. Cross-Validation Strategy for Time Series

Standard k-fold cross-validation is invalid for time series (it leaks future data into training). Always use **forward-chaining** (expanding window):

```
Fold 1: Train [t=0..100],   Test [t=101..130]
Fold 2: Train [t=0..130],   Test [t=131..160]
Fold 3: Train [t=0..160],   Test [t=161..190]
```

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|---|---|---|
| `fit` — build X | O(n · F) | n = time points, F = total features |
| `fit` — solve OLS | O(n · F² + F³) | dominated by matrix solve for large F |
| `predict` | O(m · F) | m = future time points |
| `get_components` | O(m · F) | similar to predict |

For default settings (F ≈ 53 features) and n = 1000 days: fitting completes in milliseconds.

### Space Complexity

- Design matrix: O(n × F)
- Parameters: O(F)  → ~53 floats for defaults
- Changepoints array: O(S)

### Scaling Tips

1. **For large n (> 100K points):** Sub-sample data to daily/weekly aggregations before fitting — Prophet is a macro-level model
2. **For very high F (many changepoints + high Fourier orders):** Regularise by reducing `n_changepoints` or `yearly_fourier_order`
3. **For many predictions:** Batch `predict()` calls — vectorised matrix multiply handles large future arrays efficiently

## Further Reading

### Original Papers and Documentation

- **Taylor & Letham (2018)**: "Forecasting at Scale" — Original Prophet paper from Facebook
  - Introduces the additive model, changepoints, and Fourier seasonality
  - [Available at: research.fb.com/publications](https://research.fb.com/publications/)

### Books

- **"Forecasting: Principles and Practice" (Hyndman & Athanasopoulos, 2021)**
  - Comprehensive textbook covering ARIMA, ETS, and decomposition methods
  - Free online at: [otexts.com/fpp3](https://otexts.com/fpp3)

- **"Time Series Analysis and Its Applications" (Shumway & Stoffer)**
  - Graduate-level introduction to time series theory

### Libraries for Production Use

- **[Facebook Prophet (Python)](https://facebook.github.io/prophet/)**: `pip install prophet`
  - Official library with Stan-based MCMC sampling, holiday support, uncertainty intervals
- **[statsmodels](https://www.statsmodels.org/)**: ARIMA, SARIMA, ETS, state-space models
- **[sktime](https://www.sktime.net/)**: Unified time series ML interface

## Summary

**Prophet is an additive decomposition model for time series forecasting.**

**Key takeaways:**

1. ✓ **Interpretable**: Decomposes forecast into trend + yearly + weekly — you always know *why* the model predicts what it does
2. ✓ **Automatic**: Changepoints and seasonality detected automatically from data
3. ✓ **Flexible**: Handles trend bends, multiple seasonal periods, and custom frequencies
4. ✓ **Fast**: Fitted via OLS — one matrix solve, no iterations required
5. ✓ **Robust**: Works with gaps, outliers, and series of varying length

**Best practices:**

- **Start with defaults**: `n_changepoints=25`, `yearly_fourier_order=10`, `weekly_fourier_order=3`
- **Evaluate on held-out data**: Always test on dates the model has never seen
- **Inspect components**: Plot trend, yearly, and weekly before trusting the forecast
- **Match seasonality to data frequency**: Disable weekly for monthly/annual data
- **Reduce changepoints** if the trend looks too wiggly

**Remember:** Prophet is a powerful tool for business time series with clear trend and seasonal patterns. For stationary series with complex autocorrelation, ARIMA may be more appropriate. For very large high-frequency datasets, deep learning (LSTM) may outperform it.

---

## Implementation Notes

This implementation uses **Ordinary Least Squares** for fitting, which is:
- Conceptually clear and educational (single linear solve)
- Numerically stable via `numpy.linalg.lstsq`
- Fast for typical business time series (up to ~100K points)

The official Facebook Prophet library uses **Stan MCMC / L-BFGS** optimisation with:
- Laplace prior on changepoint magnitudes (sparse changepoints)
- Full posterior uncertainty intervals
- Holiday effects as extra regressors
- Logistic growth option for saturating trends

**Our implementation demonstrates the core mathematical ideas** of Prophet so you can understand exactly how trend decomposition, Fourier seasonality, and piecewise linear changepoints work in practice.

For production forecasting, use the official library:
```bash
pip install prophet
```

---

**Happy forecasting!** 📈🔮📊
