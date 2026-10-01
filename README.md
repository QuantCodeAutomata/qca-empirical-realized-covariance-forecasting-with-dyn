# Realized Covariance Forecasting

> Core infrastructure for forecasting realized covariance matrices of equity
> portfolios using high-frequency intraday returns and time-varying uncertainty
> measures.

---

## Background

This project reproduces and extends the methodology of
**Barndorff-Nielsen, Hansen, Lunde & Shephard (2011)** — *"Multivariate
Realised Kernels: Consistent Positive Semi-Definite Estimators of the
Covariation of Equity Prices with Noise and Non-Synchronous Trading"* — and
related work on realized covariance forecasting with heterogeneous autoregressive
(HAR) and uncertainty-augmented models.

**Core idea:** estimate the daily covariance matrix Σ_t from high-frequency
(e.g. 5-minute) intraday returns using the multivariate realized kernel (MK)
estimator, then build forecasting models that leverage both the realized measure
and a time-varying uncertainty proxy Π_t derived from the integrated quarticity.

---

## Asset Universe

The experiments target a 10-stock subset of the Dow Jones Industrial Average:

| Ticker | Company                |
|--------|------------------------|
| AXP    | American Express       |
| BA     | Boeing                 |
| CVX    | Chevron                |
| DD     | DuPont                 |
| GE     | General Electric       |
| IBM    | IBM                    |
| JPM    | JPMorgan Chase         |
| KO     | Coca-Cola              |
| MSFT   | Microsoft              |
| XOM    | ExxonMobil             |

---

## Project Structure

```
project/
├── src/
│   ├── utils.py              # Matrix helpers, Parzen kernel, PD projection
│   └── realized_measures.py  # Realized kernel & quarticity estimators
├── data/
│   └── data_loader.py        # Synthetic DCC-GARCH data; Massive API loader
├── exp/                      # Experiment scripts
├── results/                  # Output figures and tables
├── tests/                    # pytest test suite
├── AGENTS.md                 # Agent persistent memory / developer notes
└── README.md                 # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy scipy pandas statsmodels
```

### 2. Generate a synthetic dataset

```python
from data.data_loader import generate_dataset

dataset = generate_dataset(
    n_assets=10,
    n_days=2300,
    M_intraday=78,   # 5-minute bars, 6.5-hour session
    seed=42,
)

print(dataset["daily_returns"].shape)      # (2300, 10)
print(len(dataset["intraday_returns"]))    # 2300
print(dataset["intraday_returns"][0].shape)  # (78, 10)
```

### 3. Compute realized covariance measures

```python
from src.realized_measures import compute_daily_rc_series

S_list, IQ_list, Pi_list, pi_list = compute_daily_rc_series(
    dataset["intraday_returns"]
)

print(S_list[0].shape)   # (10, 10)
print(pi_list[0].shape)  # (55,)   — K = 10*11//2
```

### 4. Use individual utility functions

```python
import numpy as np
from src.utils import vech, ivech, nearest_pd, safe_inv

S = S_list[0]
s = vech(S)          # vectorize lower triangle (column-major), length 55
S2 = ivech(s, 10)    # reconstruct symmetric matrix
```

---

## Key Algorithms

### Multivariate Realized Kernel

```
MK_t = Σ_{h=-H}^{H} k(h/H) · Γ_h
```

where Γ_h = Σ_i r_i r_{i-h}′ is the sample autocovariance matrix at lag h and
k(·) is the Parzen kernel.  Default bandwidth: H = ⌊M^(2/3)⌋.

### Pre-Averaging Integrated Quarticity

```
IQ̂_t = M · Σ_{i=1}^{M-H} x_i x_i′  −  (M/2) · Σ_{i=1}^{M-H-1} (x_i x_{i+1}′ + x_{i+1} x_i′)
```

where x_i = vech(r_i r_i′).

### Uncertainty Measure

```
Π_t = 3.777 · IQ̂_t,    π_t = √diag(Π_t)
```

### Diurnal Volatility Pattern

```
σ_d(u) = 0.88929198 + 0.75 · exp(−10u) + 0.25 · exp(−10(1−u)),   u ∈ [0, 1]
```

---

## Data Sources

| Source       | Access                        | Notes                                      |
|--------------|-------------------------------|--------------------------------------------|
| Synthetic    | `generate_dataset()`          | DCC-GARCH(1,1) simulation; always available |
| Massive API  | `load_massive_data()`         | Requires `MASSIVE_TOKEN` env variable      |

Set `MASSIVE_TOKEN` in your shell or `.env` file before calling
`load_massive_data`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## References

* Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A. & Shephard, N. (2011).
  *Multivariate Realised Kernels: Consistent Positive Semi-Definite Estimators
  of the Covariation of Equity Prices with Noise and Non-Synchronous Trading.*
  Journal of Econometrics, 162(2), 149–169.

* Engle, R.F. (2002). *Dynamic Conditional Correlation: A Simple Class of
  Multivariate Generalized Autoregressive Conditional Heteroskedasticity
  Models.* Journal of Business & Economic Statistics, 20(3), 339–350.

* Jacod, J., Li, Y., Mykland, P.A., Podolskij, M. & Vetter, M. (2009).
  *Microstructure Noise in the Continuous Case: The Pre-Averaging Approach.*
  Stochastic Processes and their Applications, 119(7), 2249–2276.
