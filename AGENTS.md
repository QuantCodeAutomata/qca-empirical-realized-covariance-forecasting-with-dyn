# AGENTS.md — Realized Covariance Forecasting Project

## Project Overview

This project implements the core infrastructure for a **realized covariance
forecasting** system inspired by the high-frequency econometrics literature
(principally Barndorff-Nielsen et al., 2011 and related papers).  The goal is
to forecast the realized covariance matrix of a portfolio of equity assets using
intraday return data and time-varying uncertainty measures.

Because the original paper's proprietary TAQ dataset (NYSE TAQ, 1993–2013) is
not publicly available, the project ships a synthetic data engine
(`data/data_loader.py`) that simulates realistic covariance dynamics via a
DCC-GARCH(1,1) model.

---

## Repository Layout

```
project/
├── AGENTS.md              ← this file (persistent agent memory)
├── README.md              ← human-facing project description
├── .gitignore
│
├── src/                   ← core library code
│   ├── __init__.py
│   ├── utils.py           ← matrix helpers (vech/ivech/nearest_pd/safe_inv/is_pd)
│   ├── realized_measures.py  ← realized kernel & quarticity estimators
│   ├── evaluation.py      ← frobenius_loss, qlike_loss, compute_losses, split_by_pi_norm
│   ├── statistical_tests.py  ← DM test, MCS (range stat + stationary bootstrap), markers
│   ├── portfolio.py       ← GMV weights, turnover, net returns, CEG, WRC, PortfolioEvaluator
│   ├── simulate_intraday.py  ← diurnal_sigma, 1-sec simulation, aggregation, Exp3 study
│   ├── shrinkage.py       ← identity/equicorr/single-factor targets, LW intensity
│   └── rolling_forecast.py  ← 8-model rolling engine: HAR, HARQ, HAR-DRD, HARQ-DRD,
│                               EWMA, EWMAQ, HEAVY, HEAVYQ
│
├── data/                  ← data layer
│   ├── __init__.py
│   └── data_loader.py     ← synthetic & Massive API data loading
│
├── exp/                   ← experiment scripts (one file per experiment)
├── results/               ← saved model outputs, figures, tables
└── tests/                 ← pytest test suite
```

---

## Key Design Decisions

### vech / ivech ordering (column-major)

The `vech` operator stacks the **lower triangle column by column** (the standard
econometric convention, also matching Matlab's `vech`).  For a 3 × 3 matrix:

```
vech(M) = [M[0,0], M[1,0], M[2,0], M[1,1], M[2,1], M[2,2]]
```

Always use `src.utils.vech` / `ivech` rather than ad-hoc indexing.

### Positive-definiteness contract

All covariance and quarticity matrices returned by the public API are
**guaranteed symmetric PD**.  Internal helpers (`nearest_pd`, `is_pd`) are used
to enforce this.  A `logging.WARNING` is emitted whenever a projection is
required; callers should monitor logs during model development.

### Realized kernel bandwidth

Default bandwidth: `H = int(M^(2/3))` following Barndorff-Nielsen et al.
(2011).  This can be overridden by passing an explicit `H` argument.

### Diurnal pattern

The U-shaped intraday volatility pattern uses the functional form from the
paper:

```
σ_d(u) = C + A·exp(−a·u) + B·exp(−b·(1−u))
```

with `A=0.75, B=0.25, C=0.88929198, a=b=10`.  These parameters can be
overridden via the `diurnal_params` dictionary argument in
`generate_intraday_returns`.

### Random seeds

* `simulate_garch_covariance(seed=42)` — master seed for DCC-GARCH paths.
* `generate_intraday_returns(seed_offset=t)` — per-day seed derived from
  `seed * n_days + t` to ensure reproducibility without correlating day
  shocks.

### Massive API

Real data loading via `load_massive_data` requires:
1. `pip install massive` (proprietary package, not on PyPI — install from
   internal index or vendor).
2. Environment variable `MASSIVE_TOKEN=<your_token>`.

The function returns `None` gracefully when unavailable; callers should fall
back to `generate_dataset`.

---

## Dependencies

| Package      | Purpose                                      |
|--------------|----------------------------------------------|
| numpy        | All numerical operations                     |
| scipy        | Cholesky-based linear algebra (`safe_inv`)   |
| pandas       | Time series utilities, rolling means         |
| statsmodels  | Available for ARMA/HAR models (future work)  |

No external API calls are made in `src/utils.py` or `src/realized_measures.py`.

---

## Coding Conventions

* **Type hints everywhere** — all public functions are fully annotated.
* **Docstrings** — NumPy-style docstrings with Parameters / Returns / Raises.
* **Imports at top** — no inline imports except for optional dependencies in
  `data_loader.py` (`massive`, handled with `try/except`).
* **Fail fast** — raise `ValueError` with informative messages for invalid
  inputs rather than silently returning garbage.
* **Logging** — use `logging.getLogger(__name__)` in every module; never
  `print` in library code.
* **No mocks in tests** — tests exercise real code paths on small synthetic
  inputs.

---

## Extending the Project

### Adding a new estimator

1. Add a function to `src/realized_measures.py` (or a new `src/` module).
2. Write a test in `tests/`.
3. Add an experiment script in `exp/`.

### ivech signature

The existing `ivech` in `utils.py` requires **two** arguments:
`ivech(v, n)` where `n` is the matrix dimension.  Always pass `n` explicitly;
do not rely on inferring `n` from `len(v)`.

### rolling_forecast.py model conventions

* **HAR / HARQ**: element-wise OLS on pooled scalar HAR features
  (daily/weekly/monthly means of the full vech vector).  Beta shape:
  `(4, K)` for HAR, `(5, K)` for HARQ.
* **HAR-DRD / HARQ-DRD**: DRD decomposition — HAR on log-variances and
  Fisher-z correlations separately; reconstruction via `tanh(z)` and `D R D`.
* **EWMA / EWMAQ**: decay rate `lambda` optimised over `[0.80, 0.9999]` by
  minimising element-wise QLIKE.  EWMAQ additionally estimates `beta_Q`
  controlling uncertainty-dependent smoothing.
* **HEAVY / HEAVYQ**: common `(a, b)` parameters (elementspecific intercept
  `c = (1-a-b)*mean_s`) optimised via Nelder-Mead on element-wise QLIKE.
  HEAVYQ adds `beta_Q` scaling the forcing variable by `(1 + beta_Q * |pi|)`.

Parameters are stored as Python dicts and passed between
`estimate_window_params` → `forecast_one_step`.

### evaluation.py QLIKE implementation

QLIKE uses two triangular solves to avoid an explicit matrix inverse:
```
Z = solve_triangular(L, S, lower=True)   # Z = L^{-1} S
W = solve_triangular(L, Z.T, lower=True) # W = L^{-1} S L^{-T}
Tr(H^{-1} S) = Tr(W)
```
Returns `np.inf` when H is not PD (Cholesky fails).

### statistical_tests.py bootstrap

`stationary_bootstrap_mean` and the MCS bootstrap use
`np.random.default_rng(seed)` for reproducibility.  Block lengths are drawn
from Geometric(p=1/block_size) with circular (wrap-around) indexing.

### portfolio.py certainty-equivalent gain

Solves the implied quadratic equation for `Delta_gamma` exactly (with a
linear fallback when the quadratic is degenerate).  Returns annualized basis
points: `delta_daily * 252 * 10_000`.

### Adding a forecasting model

Models should accept:
* `S_list` — list of (N, N) realized covariance matrices.
* `pi_list` — list of (K,) uncertainty vectors.

and return a list of (N, N) forecast matrices.

### Running experiments

```bash
python -m exp.<script_name>
```

Results (figures, CSV tables) go in `results/`.

---

## Model Modules (src/)

### har_models.py

Implements `HARModel` and `HARQModel` for vech-stacked covariance forecasting.

* **Estimation**: pooled OLS via `statsmodels.OLS` on `(T_eff*K, 4)` stacked
  design matrix; shared scalar slopes `theta_1`, `theta_2`, `theta_3` across
  all K vech elements; element-specific intercepts `theta_0[k]` recovered via
  the mean-condition after fitting (no OLS dummy variables needed).
* **HARQ extra regressor**: `pi_{t-1} ⊙ s_{t-1}` — element-wise product of
  the demeaned measurement-error proxy and the lagged covariance vech.
  Estimated column order in X: `[1, s_{t-1}, pi_{t-1}*s_{t-1}, s_avg5, s_avg22]`.
* `compute_lag_averages(s_series)` → `(s_lag1, s_avg5, s_avg22)`, all shape
  `(T-22, K)`, valid for `t = 22, …, T-1`.

### drd_models.py

Implements `HARDRDModel` and `HARQDRDModel` with DRD decomposition.

* `DRDDecomposition.decompose(S)` → `(d_t, R_t)`: `d_t = sqrt(diag(S))`,
  `R_t` = correlation matrix (guaranteed PD with unit diagonal via
  `_ensure_valid_corr`).
* `DRDDecomposition.recompose(d_t, R_t)` → `S = diag(d_t) @ R_t @ diag(d_t)`.
* Variance model: N independent **univariate** HAR / HARQ fits on
  `d_{n,t}^2 = S_t[n, n]`.
* Correlation model: scalar HAR (`HARModel`) on off-diagonal lower-triangle
  elements of `R_t`; these are in column-major order (same convention as
  `vech` but excluding diagonal).
* `vech_diag_indices(n)` returns vech positions of diagonal elements:
  `[j*n - j*(j-1)//2 for j in range(n)]` (column-major).
* `_extract_offdiag(R, n)` / `_build_corr_matrix(r, n)`: conversion between
  off-diagonal vech vectors and full correlation matrices.

### heavy_ewma_models.py

Implements `EWMAModel`, `EWMAQModel`, `HEAVYModel`, `HEAVYQModel`.

* **Composite likelihood** (`composite_log_likelihood`): bivariate Gaussian
  CL over N-1 contiguous pairs; normalised by `T*(N-1)`.
* **Initialisation**: all models set `V_0 = sample_mean(S)` (= `lambda_V`).
* **EWMAQ / HEAVYQ** operate in vech space using `vech`/`ivech` from utils;
  element-wise `alpha_t` or `a_t` vectors are clipped to `[1e-4, 1-1e-4]`.
* **Optimisation**: EWMA/EWMAQ use `L-BFGS-B`; HEAVY/HEAVYQ use `SLSQP`
  with `a + b < 1` constraint.
* Internal series generators `_ewma_series`, `_ewmaq_series`, `_heavy_series`,
  `_heavyq_series` are standalone functions used by both the class methods and
  the public `fit_*` helpers to avoid circular dependencies.

