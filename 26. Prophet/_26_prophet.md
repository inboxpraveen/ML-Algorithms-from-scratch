# Prophet from Scratch: A Comprehensive Guide

Prophet is Facebook's answer to a very practical question: *how do you forecast thousands of business time series without a time series expert babysitting each one?* Its answer is an additive decomposition you can read off the page — a bendable trend, plus stacked seasonal waves — fitted in a single regularised linear solve.

## Table of Contents
1. [Quick Start: Plug-and-Play Example](#quick-start-plug-and-play-example)
2. [Overview](#overview)
3. [When to Use Prophet](#when-to-use-prophet)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Algorithm Steps](#algorithm-steps)
6. [Parameters Explained](#parameters-explained)
7. [Code Example](#code-example)
8. [Practical Use Cases](#practical-use-cases)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Common Issues and Solutions](#common-issues-and-solutions)
11. [Tips for Success](#tips-for-success)
12. [Prophet vs Other Methods](#prophet-vs-other-methods)
13. [Advanced Topics](#advanced-topics)
14. [Performance Considerations](#performance-considerations)
15. [Simplification vs. canonical Prophet](#simplification-vs-canonical-prophet)
16. [Further Reading](#further-reading)
17. [Summary](#summary)
18. [Implementation Notes](#implementation-notes)

---

## Quick Start: Plug-and-Play Example

This is a complete, self-contained script. Copy it, paste it, and run it. No extra dependencies beyond NumPy.

```python
# ---------------------------------------------------------------
# Prophet from Scratch - Complete Runnable Example
# Requires: numpy only
# Run with: python _26_prophet.py  (the __main__ block runs this)
# Or copy the Prophet class from _26_prophet.py and paste above.
# ---------------------------------------------------------------
import numpy as np
from datetime import datetime, timedelta

# ---- Paste the Prophet class here (from _26_prophet.py) ----
# class Prophet: ...

np.random.seed(42)

# ------ 2 years of daily sales: trend + yearly + weekly + noise ------
n_days = 730
start = datetime(2022, 1, 1)
dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]
t = np.arange(n_days, dtype=float)

trend  = 100.0 + 0.15 * t                        # planted slope 0.15/day
yearly = 30.0 * np.sin(2 * np.pi * t / 365.25)   # planted amplitude 30
weekly = 15.0 * np.sin(2 * np.pi * t / 7.0)      # planted amplitude 15
sales  = trend + yearly + weekly + np.random.normal(0, 5, n_days)

# ------ Fit with the defaults and read the planted values back out ------
model = Prophet(n_changepoints=25, changepoint_prior_scale=0.05)
model.fit(dates, sales)

n_cp = len(model.changepoints_t_)
yr_i = model._n_trend_params
wk_i = model._n_trend_params + model._n_yearly_params
print(f"In-sample R2   : {model.score(dates, sales):.4f}")
print(f"intercept      : {model.params_[0]:8.3f}   (planted 100.00)")
print(f"trend slope    : {model.params_[1]:8.4f}   (planted   0.15)")
print(f"yearly amp     : {np.hypot(*model.params_[yr_i:yr_i+2]):8.3f}   (planted  30.00)")
print(f"weekly amp     : {np.hypot(*model.params_[wk_i:wk_i+2]):8.3f}   (planted  15.00)")
print(f"changepoints   : {n_cp} placed, "
      f"{int(np.sum(np.abs(model.params_[2:2+n_cp]) > 0))} kept by the Laplace prior")

# ------ Chronological holdout: train on 600 days, forecast the last 130 ------
tr_d, tr_y = dates[:600], sales[:600]
te_d, te_y = dates[600:], sales[600:]

tuned = Prophet(n_changepoints=25).fit(tr_d, tr_y)                    # prior ON
naive = Prophet(n_changepoints=25,
                changepoint_prior_scale=np.inf).fit(tr_d, tr_y)       # prior OFF

print(f"\n{'model':<28}{'in-sample R2':>14}{'holdout R2':>13}")
print(f"{'prior 0.05 (default)':<28}{tuned.score(tr_d, tr_y):>14.4f}"
      f"{tuned.score(te_d, te_y):>13.4f}")
print(f"{'prior OFF (plain OLS)':<28}{naive.score(tr_d, tr_y):>14.4f}"
      f"{naive.score(te_d, te_y):>13.4f}")
print("The in-sample column barely moves; the holdout column is the whole story.")

# ------ Decompose the forecast into interpretable parts ------
comps = model.get_components(dates)
print(f"\nTrend  range: [{comps['trend'].min():.1f}, {comps['trend'].max():.1f}]")
print(f"Yearly range: [{comps['yearly'].min():.1f}, {comps['yearly'].max():.1f}]")
print(f"Weekly range: [{comps['weekly'].min():.1f}, {comps['weekly'].max():.1f}]")

# ------ Forecast 90 days past the end of history ------
future = model.make_future_dataframe(periods=90, freq='D')
forecast = model.predict(future)
print(f"\nNext 90 days start at {future[0]}, end at {future[-1]}")
print(f"First 5 forecasts: {np.round(forecast[:5], 1)}")
```

Expected output:
```
In-sample R2   : 0.9777
intercept      :   99.302   (planted 100.00)
trend slope    :   0.1568   (planted   0.15)
yearly amp     :   29.747   (planted  30.00)
weekly amp     :   14.813   (planted  15.00)
changepoints   : 25 placed, 1 kept by the Laplace prior

model                         in-sample R2   holdout R2
prior 0.05 (default)                0.9788       0.9102
prior OFF (plain OLS)               0.9796     -17.8360
The in-sample column barely moves; the holdout column is the whole story.

Trend  range: [99.3, 208.8]
Yearly range: [-31.2, 29.6]
Weekly range: [-14.0, 13.9]

Next 90 days start at 2024-01-01, end at 2024-03-30
First 5 forecasts: [224.  217.4 204.4 198.  200.8]
```

The two lines that matter most are the two rows of that table. Both models describe the *history* equally well (R2 0.9788 vs 0.9796), and yet one forecasts the future almost perfectly while the other is catastrophically wrong. That gap is what the `changepoint_prior_scale` prior buys you, and it is the single most important idea on this page. Section 4 of the Mathematical Foundation explains why.

One caveat to carry with you from the start: the prior buys you a *sparse, stable* trend, not a guarantee that the surviving bend is the real one. Section 4's "When the prior is not enough" shows the same model, the same prior and the same generating process getting a planted break right on 900 days of history and wrong on 600.

---

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

#### Where Changepoints Come From

Nothing is "detected" up front. Candidate changepoints are placed **uniformly** in the first `changepoint_range` fraction of training data (default: 80%), and the fit then decides which of them matter (large δ) and which are irrelevant (δ **exactly** 0).

That last word is doing a lot of work. With plain least squares nothing is ever exactly zero: 25 hinge columns will all take some non-zero value, the trend becomes a wiggle that chases noise, and the extrapolated slope at the end of history is whatever the last few hinges happened to land on. Selection only happens because of the **Laplace prior on δ** — see Section 4, "Fitting — Penalised Least Squares", below. Read "changepoint detection" throughout this page as "place many candidates, then let the prior switch off the ones the data does not insist on."

And read it with one further reservation: switching candidates off is not the same as switching the *right* ones off. On a series too short to distinguish a trend bend from a shift in the annual wave, the surviving bends can land nowhere near the real break — Section 4's "When the prior is not enough" measures exactly that.

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

### 4. Fitting — Penalised Least Squares (the Laplace prior on δ)

Once the design matrix X is built by stacking all features (trend + seasonality), the full model is simply:

```
y = X · θ + ε
```

Where θ is the vector of all parameters to estimate. The obvious thing to do is ordinary least squares — minimise the sum of squared errors:

```
θ_OLS = argmin_θ ||y - X·θ||²    with closed form   θ = (X^T X)^{-1} X^T y
```

**And that is exactly what you must not do.** Here is why, measured on the demo series in `_26_prophet.py` (730 days, planted slope 0.15/day, planted yearly amplitude 30; trained on the first 600 days, scored on the last 130):

| Fit | In-sample R² | 130-day holdout R² | Recovered yearly amplitude |
|---|---|---|---|
| OLS, 25 changepoints | 0.9796 | **-17.84** | 69.1 (truth: 30) |
| Penalised, 25 changepoints | 0.9788 | **+0.9102** | 29.9 (truth: 30) |
| Oracle (knows the true components) | — | +0.9134 | 30 |

The 25 hinge columns and the 20 low-order yearly Fourier columns are almost collinear: a slow bend in the trend and a slow annual wave look nearly the same over two years. OLS is unbiased but has enormous *variance* under collinearity, so it splits the signal between the two blocks in a wild, noise-driven way. In-sample it does not matter — the two errors cancel on the training window. Out of sample they do not cancel at all.

#### The fix: a prior on δ

Canonical Prophet puts a **Laplace(0, τ) prior on the changepoint rate adjustments δ** (and only on δ — the intercept, base slope and Fourier amplitudes stay unpenalised). Writing the negative log-posterior and dropping constants gives the objective this implementation minimises:

```
θ* = argmin_θ  ½·||y - X·θ||²  +  λ · Σ_j |δ_j|

with  λ = σ² / changepoint_prior_scale
```

- **σ²** is the residual variance of the unpenalised fit — the noise level of the series.
- **`changepoint_prior_scale` (τ)** is the prior scale. Larger τ → smaller λ → more flexible trend. τ = ∞ removes the penalty and recovers plain OLS.
- The penalty is an **L1** (absolute value) penalty, not L2. That is not an accident: L1 produces *exactly zero* coefficients, so the fit performs changepoint **selection**, not just changepoint shrinkage.

#### Standardisation makes λ mean the same thing everywhere

Before solving, `fit()` rescales the way canonical Prophet does:

```
t  ->  (t - t_min) / (t_max - t_min)     # training span becomes 1.0
y  ->  y / max|y|                        # Prophet's "absmax" scaling
```

Least squares is scale-equivariant, so this changes *nothing* about an unpenalised fit — but it changes everything about a penalised one. Without it, `changepoint_prior_scale=0.05` would mean one thing for a 2-year series and something completely different for a 10-year series. After solving, `fit()` converts θ back to raw per-day units so `params_`, `predict()` and `get_components()` all stay in "days".

#### Solving it: coordinate descent + soft-thresholding

An L1 penalty has no closed form, but the problem is convex and separable, so **coordinate descent** solves it: optimise one coefficient at a time, holding the rest fixed. Two refinements make that actually land on the minimum, and both are in `Prophet._solve_penalized`.

**Refinement 1 — profile the unpenalised block away first.** Split the columns into the free block **A** (intercept, base slope, all Fourier terms) and the penalised block **H** (the 25 hinges). For any fixed δ the free coefficients have a closed form — they are just least squares of `y - H·δ` on `A`. Substituting that back (this is the Frisch–Waugh–Lovell theorem) leaves an ordinary lasso in δ alone:

```
δ* = argmin_δ  ½·||y_p - H_p·δ||²  +  λ · Σ_j |δ_j|

with   y_p = y - A·lstsq(A, y)      # y with its projection onto col(A) removed
       H_p = H - A·lstsq(A, H)      # each hinge, likewise
```

Same optimum, 25 coordinates instead of 53, and — the point — far less collinearity, because the worst of it (a slow bend versus a slow annual wave) has been projected out. Sweeping over the free columns as well, which is the obvious way to write this, makes the descent crawl. Holding the update rule and the stopping rule fixed and changing only that, Example 1's series needs **2814** sweeps to reach the same certified optimum instead of **18**, and Example 2's hardest panel needs **81,413** instead of **4808**.

With G = H_p^T H_p and c = H_p^T y_p precomputed, the partial-residual numerator for hinge j is

```
z_j = c_j - (G[j] · δ) + G[j,j] · δ_j
```

and the update is

```
δ_j = shrink(z_j, λ) / G[j,j]
```

where `shrink` is the **soft-thresholding operator**:

```
shrink(z, λ) = z - λ    if z >  λ
             = z + λ    if z < -λ
             = 0        if |z| <= λ
```

This is the same operator `_17_xgboost.py` uses for its L1 leaf weights, and it is the line of code that turns "penalty" into "selection": a changepoint whose evidence `z` is weaker than the prior's pull `λ` is switched off completely. On the demo series (all 730 days, τ = 0.05) **24 of the 25 candidate deltas come back as exact zeros**, leaving a single bend at day 163.

**Refinement 2 — stop on the KKT conditions, not on "nothing moved".** The tempting stopping rule is "quit when no coordinate moved by more than `tol` in a full sweep". It is a trap here. Coordinate descent on nearly-collinear columns takes tiny steps for a very long time, so a small step size says nothing about how far the objective still has to fall. Stop it early and you do not get an approximate lasso solution — you get *a different estimator*, with different changepoints selected, whose behaviour depends on a sweep count nobody documented.

The honest stopping rule uses the optimality conditions themselves. Once a sweep leaves the sign pattern unchanged, take the active set S (the non-zero δ, with signs s) and solve the small linear system that the smooth part of the objective demands there:

```
G[S,S] · δ_S = c_S - λ · s_S
```

Then check two things: that δ_S came back with the signs it was supposed to have, and that every switched-off coordinate satisfies `|G[j]·δ - c_j| ≤ λ`. Those are the KKT conditions of the lasso, and because the problem is convex they are **sufficient** — a point that satisfies them is the global minimum, full stop. Not "close to"; *is*. If either check fails, the support is still moving and the sweeps continue. `_solve_penalized` records the outcome in `_n_sweeps_` and `_solver_certified_`, and the demo prints both.

#### Why this still counts as "fast and transparent"

Because all components are **additive and linear in the parameters**, the entire model is still one convex regression. Prophet remains:
- **Fast** to fit — the Gram matrix is only 25×25 after profiling. Example 1's 730-day fit certifies after 18 sweeps in about 10 ms; the slowest fit in the whole demo (Example 2's 600-day panel, a genuinely ill-conditioned case) needs 4808 sweeps and about 0.3 s
- **Stable** (convex objective, no local minima, no random restarts)
- **Verifiable** (the solver does not report a guess — it reports a KKT certificate, or tells you it ran out of sweeps)
- **Transparent** (every parameter has a clear geometric interpretation, and the zeros tell you which changepoints the data rejected)

#### When the prior is not enough

Sparsity is not the same as correctness, and it is worth seeing where this model's automatic changepoints genuinely fail. Example 2 in `_26_prophet.py` plants a real break — +0.15/day for a year, then -0.10/day — in a three-year series and fits the same default model on two different amounts of history:

| Training window | Holdout R² (25 candidates) | Surviving bends | Holdout R² if you *tell* it the break is at day 365 |
|---|---|---|---|
| 900 days (2.5 yearly cycles) | **+0.9152** | days 345, 374 | +0.9155 |
| 600 days (1.6 yearly cycles) | **-1.6993** | days 172, 268, 326, 383 | +0.7544 |

Both fits are KKT-certified optima. The 600-day fit is not under-solved; it is *under-identified*. Over 1.6 yearly cycles, with 20 unpenalised yearly Fourier columns free to move, "the trend bent down at day 365" and "the annual wave sits a little lower and later" describe almost the same curve. Forcing the 600-day model to spend its single bend on the candidate nearest the true break — refitting at the same λ with the day-364 hinge free and every other δ pinned to 0 — raises the standardised objective by only **+2.4e-04** (0.2%) while turning the holdout from -1.6993 into +0.7535. Only the objective figure needs that hand-built fit — `fit()` recomputes σ², and with it λ, from whatever design it is handed, so λ cannot be pinned to the 25-candidate model's value through the public API. The forecast half is reproducible: `Prophet(n_changepoints=1, changepoint_range=364/599)` puts a single candidate on day 364 and scores +0.7537 on the same holdout, and the ORACLE row above (+0.7544, one candidate planted on day 365) is the version the demo prints. The likelihood surface simply does not care which of the two stories it tells, and the prior only penalises the *size* of a bend, not its *location*.

Two practical consequences:

1. Give the model at least two full seasonal cycles before you trust an automatically-placed changepoint, and preferably a cycle of history on each side of the bend you care about.
2. If you cannot get more history, take flexibility away from the competing block instead: a lower `yearly_fourier_order`, or `yearly_seasonality=False`, leaves the hinges as the only way to explain a bend.

This is also a caution about how such demos are read. The reason to print the certificate is that a solver stopped short of the optimum will happily produce a *better-looking* forecast here — early stopping is itself a regulariser — and you would be reporting the accident, not the model.

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

> The five snippets in this section are **excerpts from inside the class**, quoted
> with the same variable names `_26_prophet.py` uses. They illustrate what `fit()`
> and `predict()` do internally; they are not standalone scripts. For copy-paste
> code, use the [Quick Start](#quick-start-plug-and-play-example) or
> [Code Example](#code-example) sections.

### Step 1: Parse Dates

Convert date strings or datetime objects to numeric values (days since start):

```python
# '2022-01-15' → 14 (days from '2022-01-01')
t = [(date - start_date).days for date in dates]
```

### Step 2: Place Candidate Changepoints

Spread `n_changepoints` candidates evenly over the first `changepoint_range` fraction of the training data. This is the exact code from `fit()`:

```python
t_eligible = t[t <= t.min() + changepoint_range * (t.max() - t.min())]
n_cp = min(n_changepoints, max(0, len(t_eligible) - 1))
cp_idx = np.round(np.linspace(0, len(t_eligible) - 1, n_cp + 1)).astype(int)
changepoints_t_ = t_eligible[np.unique(cp_idx[1:])]
```

Three details are load-bearing:

1. **The grid is built over INDICES, not time values.** That keeps the candidates evenly spread across *observations*, which is what you want when the dates are irregular.
2. **Index 0 is dropped** (`cp_idx[1:]`). A changepoint at `t.min()` gives the hinge column `max(0, t - t_min) = t - t_min`, which is an exact duplicate of the linear slope column — it makes X rank deficient for no benefit.
3. **`np.round`, not truncation.** Truncating (`dtype=int`) collapses candidates together on short series; `np.unique` then mops up any remaining duplicates.

### Step 3: Build Design Matrix

```
# Trend features
X = [ones, t, max(0, t-s1), max(0, t-s2), ..., max(0, t-sS)]

# Yearly seasonality (Fourier features, period = 365.25)
for n in 1..N_yr:
    X.append(cos(2*pi*n*t/365.25))
    X.append(sin(2*pi*n*t/365.25))

# Weekly seasonality (Fourier features, period = 7)
for n in 1..N_wk:
    X.append(cos(2*pi*n*t/7))
    X.append(sin(2*pi*n*t/7))
```

### Step 4: Fit Parameters by Penalised Least Squares

Standardise, solve the L1 problem on the δ block, then convert back to per-day units:

```python
# t -> [0, 1] and y -> y / max|y| so the prior's units are series-independent
X_std, y_std = standardize(X, t, y)

# lam = sigma^2 / changepoint_prior_scale, penalizing columns 2 .. 2+S only
penalized = np.zeros(X.shape[1], dtype=bool)
penalized[2:2 + S] = True
theta_std = self._solve_penalized(X_std, y_std, lam, penalized)

params_ = unstandardize(theta_std)
# params_ = [m, k, d1, ..., dS, a1_yr, b1_yr, ..., a1_wk, b1_wk, ...]
```

Set `changepoint_prior_scale=np.inf` and `lam` becomes 0, at which point `_solve_penalized` short-circuits to `np.linalg.lstsq(X, y, rcond=None)[0]` — the plain OLS fit, for comparison.

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

**But note:** with the Laplace prior in place, `n_changepoints` is much less dangerous than it looks. Extra candidates that the data does not support are soft-thresholded to exactly zero, so raising it mostly costs a little compute. The knob that actually controls trend flexibility is `changepoint_prior_scale`.

---

### changepoint_prior_scale

The scale τ of the Laplace(0, τ) prior on the changepoint rate adjustments δ — **the most important parameter in Prophet**.

| Value | Effect | Use when |
|---|---|---|
| 0.001 | Essentially a straight line; equivalent to `n_changepoints=0` | You know the trend is linear |
| 0.05 (default) | Bends only where the data insists | Almost always start here |
| 0.5 | Very flexible trend | The series genuinely breaks direction often |
| `np.inf` | Penalty OFF → plain OLS | Only as a teaching contrast — the forecast will be wild |

**Direction:** larger = more flexible trend (weaker penalty, more non-zero δ); smaller = more rigid.

**How to tune it:** on a **holdout window**, never in-sample. Here is the same 730-day series fitted with 25 changepoints at four prior scales, trained on the first 600 days and scored on the last 130:

| changepoint_prior_scale | In-sample R² | Holdout R² | Non-zero δ |
|---|---|---|---|
| 0.001 | 0.9786 | +0.9119 | 0 / 25 |
| 0.05 (default) | 0.9788 | +0.9102 | 2 / 25 |
| 0.5 | 0.9789 | +0.8975 | 4 / 25 |
| `np.inf` (OLS) | 0.9796 | -17.8360 | 25 / 25 |

The in-sample column spans 0.001 of R². The holdout column spans **eighteen**. If you tune this parameter on the training fit, you will pick the worst model every time.

Notice how few deltas survive even at τ = 0.5. That is the L1 penalty doing selection, and it only reads this way because the solve is run to a certified optimum: an under-converged coordinate descent leaves many more coefficients stranded at small non-zero values (19 / 25 on this same row) and reports a sparsity that is really a property of the sweep budget.

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
    n_changepoints=25,
    yearly_seasonality=True,
    weekly_seasonality=True,
    yearly_fourier_order=10,
    weekly_fourier_order=3,
    changepoint_prior_scale=0.05
)
model.fit(dates, y)

# --- 3. In-sample evaluation ---
print(f"R2   = {model.score(dates, y):.4f}")
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

# --- 6. Which candidate changepoints survived? ---
n_cp = len(model.changepoints_t_)
deltas = model.params_[2:2 + n_cp]
print(f"\n{n_cp} candidates, {int(np.sum(np.abs(deltas) > 0))} non-zero deltas")
print(f"Surviving changepoint days: {model.changepoints_t_[np.abs(deltas) > 0]}")
```

Expected output:
```
R2   = 0.9777
MAE  = 3.82
RMSE = 4.83

Forecast for next 90 days: [224.  217.4 204.4 198.  200.8]

Trend range:   [99.3, 208.8]
Yearly range:  [-31.2, 29.6]
Weekly range:  [-14.0, 13.9]

25 candidates, 1 non-zero deltas
Surviving changepoint days: [163.]
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
# Substitute your own values for monthly_revenue; this synthetic series keeps
# the snippet runnable end to end.
np.random.seed(0)
monthly_dates = [f"{2019 + i // 12}-{(i % 12) + 1:02d}-01" for i in range(60)]
month_idx = np.arange(60)
monthly_revenue = (1000 + 8 * month_idx
                   + 150 * np.sin(2 * np.pi * month_idx / 12)
                   + np.random.normal(0, 40, 60))

model = Prophet(
    n_changepoints=10,          # Fewer changepoints for monthly data
    yearly_seasonality=True,    # Capture annual cycle
    weekly_seasonality=False,   # No weekly pattern in monthly data
    yearly_fourier_order=5      # Simpler seasonality for 12 points/year
)
model.fit(monthly_dates, monthly_revenue)
print(f"In-sample R2: {model.score(monthly_dates, monthly_revenue):.4f}")

# Forecast next 12 months.
# CAVEAT: freq='M' advances by a FIXED 30 days, not a calendar month, so the
# generated dates drift ~10 days over 12 steps. For calendar-exact month
# starts, build the list yourself and pass it straight to predict():
future = [f"{2024 + i // 12}-{(i % 12) + 1:02d}-01" for i in range(12)]
forecast = model.predict(future)
print(f"Next 12 months: {forecast.round(0)}")
```

Expected output:
```
In-sample R2: 0.9497
Next 12 months: [1486. 1534. 1600. 1646. 1639. 1547. 1497. 1445. 1353. 1365. 1393. 1476.]
```

### 5. Comparing Forecast vs Actual (Post-Hoc Analysis)

```python
# Uses the `dates` / `y` series built in the Code Example section above.
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
print(f"Holdout R2:   {r2:.4f}")
```

Expected output (on the 730-day series from the Code Example, split 584 / 146):
```
Holdout MAE:  4.01
Holdout RMSE: 5.11
Holdout R2:   0.9116
```

(The noise standard deviation of that series is 5.0, so an RMSE of 5.11 is essentially the theoretical floor. Before the changepoint prior was added, this same snippet scored **R2 = -38.9**.)

Note the split indices: `dates[:split]` and `dates[split:]` share **no** rows. A split written as `dates[:584], dates[100:]` would silently score the model on 484 rows it was trained on, and the "holdout" number would be meaningless.

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

    print(f"\nAverage CV MAE: {np.mean(results):.2f} +/- {np.std(results):.2f}")
    return results


# On the 730-day series from the Code Example section:
prophet_cv(dates, y, dict(n_changepoints=25, changepoint_prior_scale=0.05))
```

Expected output:
```
Fold 1: train=580, test=30, MAE=4.20
Fold 2: train=610, test=30, MAE=4.25
Fold 3: train=640, test=30, MAE=4.04
Fold 4: train=670, test=30, MAE=4.00
Fold 5: train=700, test=30, MAE=3.98

Average CV MAE: 4.09 +/- 0.11
```

This is also the right harness for tuning `changepoint_prior_scale`: run it once per candidate value and pick the smallest average CV MAE. Never pick on the in-sample fit.

## Common Issues and Solutions

### Issue 1: Trend is Too Wiggly / Overfits

**Symptom:** Trend oscillates up and down, doesn't look like real growth.

**Causes:**
- Too many changepoints for the length of data
- Very noisy data

**Solutions:** reach for the prior first — it is the knob designed for this.
```python
# BEST: tighten the Laplace prior. Keeps all 25 candidates available but
# demands stronger evidence before any of them is allowed to bend the trend.
model = Prophet(changepoint_prior_scale=0.01)   # down from default 0.05

# Also fine: reduce the number of candidates
model = Prophet(n_changepoints=5)   # down from default 25

# Or disable changepoints entirely for a strictly linear trend
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

**Cause:** Changepoints placed too close to the end (overfitting the training tail). A hinge near the end of history has very few observations after it, so its δ is estimated from almost nothing — and yet it sets the slope that gets extrapolated forever.

**Solution:**
```python
# Reduce changepoint_range (default 0.8 is usually good)
model = Prophet(changepoint_range=0.7)

# Or tighten the prior so a poorly-evidenced late bend is thresholded to zero
model = Prophet(changepoint_range=0.8, changepoint_prior_scale=0.01)
```

**Diagnostic:** print the extrapolated end-of-history slope and sanity-check it
against the trend you can see with your own eyes.
```python
S = len(model.changepoints_t_)
end_slope = model.params_[1] + np.sum(model.params_[2:2 + S])
print(f"Extrapolated slope at end of history: {end_slope:+.4f} units/day")
```

---

### Issue 5: Sudden Spikes Not Captured

**Symptom:** Forecast misses sharp one-off spikes (Black Friday, product launch, etc.).

**Cause:** Prophet models smooth patterns; one-off spikes are noise to it.

**Solution:**
When you know holidays or events in advance, create a binary indicator feature and include it:

```python
# Subclass and append the indicator as an extra design-matrix column.
# (Full working version: "External Regressors" under Advanced Topics.)
class ProphetWithRegressors(Prophet):
    def __init__(self, extra_regressors, **kwargs):
        super().__init__(**kwargs)
        self.extra_regressors = extra_regressors

    def _make_design_matrix(self, t):
        X = super()._make_design_matrix(t)
        return np.hstack([X, self.extra_regressors(np.asarray(t, dtype=float))])

def black_friday(t):
    doy = np.mod(t, 365.25)                      # day-of-year, roughly
    return ((doy >= 328) & (doy <= 332)).astype(float).reshape(-1, 1)

model = ProphetWithRegressors(black_friday, n_changepoints=25)
model.fit(dates, y)
```

Example 4 in `_26_prophet.py` measures exactly how much this costs you when you
skip it: on a retail series with engineered Black Friday and Christmas spikes,
the holdout MAE is 23.1 on ordinary days and 147.0 on spike days.

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
comps = model.get_components(dates)

print("Trend range:   ", comps['trend'].min(), "to", comps['trend'].max())
print("Yearly range:  ", comps['yearly'].min(), "to", comps['yearly'].max())
print("Weekly range:  ", comps['weekly'].min(), "to", comps['weekly'].max())

# Also worth a look: which candidate changepoints actually fired?
S = len(model.changepoints_t_)
deltas = model.params_[2:2 + S]
print("Non-zero deltas:", int(np.sum(np.abs(deltas) > 0)), "of", S)
print("Bend days:      ", model.changepoints_t_[np.abs(deltas) > 0])
```

Ask yourself:
- Does the trend make business sense?
- Is the seasonal amplitude reasonable?
- Are the components in the right direction?

### 2. Start with Defaults, Then Tune

```python
# Good first model — works for most daily business time series
model = Prophet(
    n_changepoints=25,            # Default
    yearly_seasonality=True,      # Default
    weekly_seasonality=True,      # Default
    yearly_fourier_order=10,      # Default
    weekly_fourier_order=3,       # Default
    changepoint_prior_scale=0.05  # Default - tune this one first
)
```

Only tune after evaluating this baseline on held-out data. If exactly one
parameter is worth a search, it is `changepoint_prior_scale`; try
`[0.005, 0.01, 0.05, 0.1, 0.5]` with the `prophet_cv` helper above.

### 3. Use a Proper Holdout Set

```python
# Always evaluate on data the model has NOT seen
split = int(0.85 * len(dates))
model.fit(dates[:split], y[:split])
print("Test MAE:", model.mae(dates[split:], y[split:]))
```

### 4. Check for Data Quality Issues

Before fitting, verify:
- Are there any missing dates? (gaps are fine - the design matrix is built from whatever dates you supply)
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
| **Fitting speed** | Fast (one penalised linear solve) | Fast |
| **Best for** | Business KPIs with trend + seasonality | Stationary or simple trending series |

### Prophet vs Exponential Smoothing (ETS)

| Aspect | Prophet | ETS |
|---|---|---|
| **Trend** | Piecewise linear, changepoints | Smooth exponential |
| **Seasonality** | Fourier, multiple periods | One additive or multiplicative |
| **Parameters** | Many (penalised least squares) | Few (MLE) |
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

This implementation focuses on core Prophet components. To add external features (e.g. a holiday binary flag, a marketing spend variable), extend the design matrix. The clean hook is `_make_design_matrix`, because `fit`, `predict` and `get_components` all go through it — override it once and every path stays consistent:

```python
import numpy as np

class ProphetWithRegressors(Prophet):
    """Prophet plus arbitrary extra columns, indexed by numeric time t."""

    def __init__(self, extra_regressors, **kwargs):
        # extra_regressors: callable t -> array of shape (len(t), n_extra)
        super().__init__(**kwargs)
        self.extra_regressors = extra_regressors

    def _make_design_matrix(self, t):
        X = super()._make_design_matrix(t)          # trend + seasonality blocks
        return np.hstack([X, self.extra_regressors(np.asarray(t, dtype=float))])


# Example: a Black-Friday indicator that fires on day-of-year 328..332
def black_friday(t):
    doy = np.mod(t, 365.25)
    return ((doy >= 328) & (doy <= 332)).astype(float).reshape(-1, 1)

model = ProphetWithRegressors(black_friday, n_changepoints=25)
model.fit(dates, y)
print("Black Friday effect:", model.params_[-1].round(2))
```

Two caveats. The extra columns land at the END of `params_`, so `get_components()` (which slices by the trend/yearly/weekly counts) will still return only the three built-in components — the extra effect shows up in `yhat` but not in any component array. And the extra columns are **unpenalised**, because `fit()` marks only the δ block for the L1 penalty.

### 3. Uncertainty Intervals

The current implementation returns point forecasts. To add prediction intervals, you can:
- Bootstrap residuals and re-predict
- Use the OLS covariance matrix for analytical confidence intervals:

```python
# Analytical confidence band (simplified - see the caveat below).
# `dates` / `y` are the training series; `future` is a list of future dates.
t_train  = model._parse_dates(dates)
X_train  = model._make_design_matrix(t_train)
residuals = y - X_train @ model.params_
sigma2 = np.sum(residuals ** 2) / (len(y) - len(model.params_))
cov = sigma2 * np.linalg.pinv(X_train.T @ X_train)

X_future = model._make_design_matrix(model._parse_dates(future))
yhat = X_future @ model.params_
# Standard error of the MEAN prediction, row by row
se_mean = np.sqrt(np.einsum('ij,jk,ik->i', X_future, cov, X_future))
# Standard error of a single new OBSERVATION adds the noise term
se_obs = np.sqrt(sigma2 + se_mean ** 2)
lower, upper = yhat - 1.96 * se_obs, yhat + 1.96 * se_obs
print(f"Day 1 forecast: {yhat[0]:.1f}  95% interval [{lower[0]:.1f}, {upper[0]:.1f}]")
```

**Caveat:** this is the *OLS* covariance formula, and the fitted model is not OLS
— the L1 penalty on δ makes the estimator biased and its true sampling
distribution non-Gaussian. Treat the band as a rough guide, not a calibrated
interval. Canonical Prophet gets honest intervals by sampling the posterior
(MCMC) and by simulating future changepoints; a from-scratch equivalent would
bootstrap the residuals and refit.

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
| `fit` — unpenalised solve (for σ²) | O(n · F² + F³) | one `lstsq` |
| `fit` — profiling out the free block | O(n · F · S) | two `lstsq` solves, computed once |
| `fit` — Gram matrix | O(n · S²) | H_p^T H_p and H_p^T y_p, computed once |
| `fit` — coordinate descent | O(K · S²) | K = sweeps; independent of n |
| `fit` — KKT certificate | O(S³) per check | one small solve on the active set |
| `predict` | O(m · F) | m = future time points |
| `get_components` | O(m · F) | similar to predict |

Here F is the total number of columns (≈ 53 for the defaults) and S is the number of changepoint candidates (25) — only the S penalised columns are iterated, because the rest are profiled out first.

Measured on the demo series with default settings and n = 730 days: the fit certifies after 18 sweeps in about **10 ms**. K is data-dependent, not a fixed budget: an easy series certifies in a handful of sweeps, while Example 2's deliberately ill-conditioned 600-day panel needs 4808 sweeps and about 0.3 s — and roughly half of that is the certificate rather than the sweeps, because on that fit the sign pattern is unchanged on almost every sweep, so ~4.8k active-set solves get attempted and rejected. The whole 4-example `__main__` demo runs in about **1.2 s** of wall clock for `python _26_prophet.py`, interpreter start-up and the NumPy import included (median 1.20 s over 9 runs, spread 1.16-1.31 s). Every timing on this page is a quiet-machine figure; the same code on a loaded machine can measure several times slower. `max_iter=50000` is a backstop, not the normal exit; if it is ever reached, `_solver_certified_` comes back `False`. Because coordinate descent works on the S×S Gram matrix rather than the n×F design matrix, its per-sweep cost does not grow with the length of the series — only the one-off Gram and projection construction does.

### Space Complexity

- Design matrix: O(n × F)
- Parameters: O(F)  → ~53 floats for defaults
- Changepoints array: O(S)

### Scaling Tips

1. **For large n (> 100K points):** Sub-sample data to daily/weekly aggregations before fitting — Prophet is a macro-level model
2. **For very high F (many changepoints + high Fourier orders):** Tighten `changepoint_prior_scale` first; if it is still slow, reduce `n_changepoints` (coordinate descent is O(S²) per sweep in the number of changepoints, and badly-conditioned hinge blocks need many more sweeps)
3. **For many predictions:** Batch `predict()` calls — vectorised matrix multiply handles large future arrays efficiently

## Simplification vs. canonical Prophet

This file implements the core of Taylor & Letham (2018): the additive
decomposition, the piecewise-linear trend with hinge changepoints, Fourier
seasonality, and — crucially — the **Laplace prior on the changepoint
magnitudes**, which is what makes automatic changepoints usable at all.

The following features of the official `prophet` library are deliberately
**not** implemented here. Each entry says what canonical Prophet does, why this
file omits it, and what it costs you in practice.

| Feature | What canonical Prophet does | Why omitted here | Practical consequence |
|---|---|---|---|
| **Uncertainty intervals** | Samples the posterior with Stan (MCMC), and simulates *future* changepoints drawn from the fitted δ distribution, to produce `yhat_lower` / `yhat_upper` | Requires an MCMC sampler and a changepoint-simulation loop — well past the size budget for one teaching file | `predict()` returns point forecasts only. See "Uncertainty Intervals" under Advanced Topics for a rough analytical band and its caveats |
| **Logistic (saturating) growth** | `growth='logistic'` with a per-row `cap` column: `trend(t) = C(t) / (1 + exp(-(k + A·δ)(t - (m + A·γ))))` | A different trend family with its own non-linear solve | Forecasts grow (or shrink) without bound. For a metric with a real ceiling — market share, adoption rate — extrapolate with care |
| **Holiday effects** | A `holidays` DataFrame becomes extra binary regressors with their own `holiday_prior_scale` | Needs calendar handling plus a second prior block | One-off spikes are pure error to this model. Example 4 in the `.py` demo quantifies exactly that: MAE 23.1 on ordinary days vs 147.0 on holiday-spike days. See "External Regressors" for the subclass hook |
| **Multiplicative seasonality** | `seasonality_mode='multiplicative'`: seasonality multiplies the trend instead of adding to it | Changes the model from linear-in-parameters to a product form | Use the log-transform recipe in Advanced Topics: fit on `log1p(y)`, back-transform with `expm1` |
| **Daily / custom seasonalities** | `add_seasonality(name, period, fourier_order)` for any period | Only yearly (365.25) and weekly (7.0) are wired in | No sub-daily patterns. Adding one is a two-line change to `_make_design_matrix` |
| **Seasonality prior** | `beta ~ Normal(0, seasonality_prior_scale)` with default 10 | In the standardised units used here that prior is nearly flat, so it changes almost nothing | Fourier amplitudes are unpenalised. Only matters if you push `yearly_fourier_order` very high on a short series |
| **Joint estimation of σ** | `sigma_obs` is a free parameter sampled/optimised jointly with everything else | Would make the objective non-convex and the solve iterative | `fit()` estimates σ² once from the unpenalised residuals and holds it fixed while computing `lam = σ² / changepoint_prior_scale` |
| **Automatic seasonality selection** | Turns yearly seasonality off automatically when the history is under two cycles | Explicit is better than implicit for teaching | You must set `yearly_seasonality=False` yourself on series shorter than ~2 years |

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
2. ✓ **Automatic**: Candidate changepoints are placed on a fixed grid, and the Laplace prior selects which ones survive (most δ come back exactly 0). Selection is not the same as *correct* selection — see "When the prior is not enough" in Section 4
3. ✓ **Flexible**: Handles trend bends, multiple seasonal periods, and custom frequencies
4. ✓ **Fast**: Fitted via penalised least squares — one 25×25 Gram matrix and coordinate-descent sweeps to a KKT-certified optimum; about 10 ms on a well-conditioned two-year series, a few tenths of a second on a badly conditioned one
5. ✓ **Handles gaps**: Any set of dates works, regular or not — the design matrix is built from the numeric times you supply
6. ✗ **Not outlier-robust**: the loss is squared error, so a single extreme point moves the whole fit in proportion to its magnitude. Winsorise or log-transform obvious outliers before fitting

**Best practices:**

- **Start with defaults**: `n_changepoints=25`, `changepoint_prior_scale=0.05`, `yearly_fourier_order=10`, `weekly_fourier_order=3`
- **Evaluate on held-out data**: Always test on dates the model has never seen. In-sample R² is nearly blind to the failure mode that matters here
- **Inspect components**: Plot trend, yearly, and weekly before trusting the forecast
- **Match seasonality to data frequency**: Disable weekly for monthly/annual data
- **Tighten `changepoint_prior_scale`**, not `n_changepoints`, if the trend looks too wiggly

**Remember:** Prophet is a powerful tool for business time series with clear trend and seasonal patterns. For stationary series with complex autocorrelation, ARIMA may be more appropriate. For very large high-frequency datasets, deep learning (LSTM) may outperform it.

---

## Implementation Notes

This implementation fits the model by **penalised least squares** — the MAP
estimate under Prophet's Laplace prior on δ — which is:
- Conceptually clear and educational (a convex objective, one soft-thresholding rule)
- Numerically stable (the unpenalised path uses `numpy.linalg.lstsq`; the penalised path works on the Gram matrix)
- Verifiable: the solver stops on the lasso KKT conditions, so it either returns a proven global optimum or reports `_solver_certified_ = False`
- Fast for typical business time series (about 10 ms for 730 days with 53 parameters)

The objective, spelled out once more:

```
minimise  0.5 * ||y - X @ theta||^2  +  lam * sum_j |delta_j|
with      lam = sigma^2 / changepoint_prior_scale
```

after standardising `t` to [0, 1] and `y` to `y / max|y|`, exactly as canonical
Prophet does. `Prophet._solve_penalized` profiles the unpenalised columns out
(Frisch–Waugh–Lovell), runs the coordinate-descent sweeps on the 25 hinge
columns that remain, and certifies the result against the KKT conditions;
`Prophet._soft_threshold` implements `shrink(z, lam)`.

The official Facebook Prophet library uses **Stan MCMC / L-BFGS** optimisation with:
- The same Laplace prior on changepoint magnitudes (sparse changepoints) — implemented here
- Full posterior uncertainty intervals — not implemented here
- Holiday effects as extra regressors — not implemented here
- Logistic growth option for saturating trends — not implemented here
- A Normal prior on the seasonality amplitudes — not implemented here

See the "Simplification vs. canonical Prophet" section above for what each of
those omissions costs you in practice.

**Our implementation demonstrates the core mathematical ideas** of Prophet so you can understand exactly how trend decomposition, Fourier seasonality, piecewise linear changepoints, and the sparsity-inducing prior work in practice.

For production forecasting, use the official library:
```bash
pip install prophet
```

---

**Happy forecasting!** 📈🔮📊
