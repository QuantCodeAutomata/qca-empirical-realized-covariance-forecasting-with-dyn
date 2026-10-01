"""Data loading and synthetic dataset generation for realized covariance experiments.

Because the original paper's TAQ data (1993–2013) is not publicly available,
this module provides:

* A DCC-GARCH simulation engine that replicates realistic covariance dynamics.
* An intraday return generator with a U-shaped diurnal volatility pattern.
* A ``generate_dataset`` entry-point that wires both together.
* A ``load_massive_data`` function that attempts to pull real intraday data
  from the Massive financial data API (requires ``MASSIVE_TOKEN`` in the
  environment) and falls back gracefully to ``None``.

Ticker universe and index are defined as module-level constants so that other
modules can import them without instantiating any data.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import nearest_pd, is_pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSET_UNIVERSE = ["AXP", "BA", "CVX", "DD", "GE", "IBM", "JPM", "KO", "MSFT", "XOM"]
SPY_TICKER = "SPY"


# ---------------------------------------------------------------------------
# DCC-GARCH covariance simulation
# ---------------------------------------------------------------------------

def _make_factor_correlation(n_assets: int, n_factors: int, rng: np.random.Generator) -> np.ndarray:
    """Build a realistic factor-model correlation matrix.

    Parameters
    ----------
    n_assets : int
    n_factors : int
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Correlation matrix of shape (n_assets, n_assets), PD with ones on the diagonal.
    """
    # Factor loadings in (0.3, 0.8); idiosyncratic noise keeps it from being rank-deficient
    loadings = rng.uniform(0.3, 0.7, (n_assets, n_factors))
    noise = np.diag(rng.uniform(0.2, 0.5, n_assets))
    cov = loadings @ loadings.T + noise
    # Convert to correlation
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    # Numerical symmetry
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    if not is_pd(corr):
        corr = nearest_pd(corr)
        # Re-normalize diagonal to 1
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
    return corr


def simulate_garch_covariance(
    n_assets: int, n_days: int, seed: int = 42
) -> list:
    """Simulate a panel of daily covariance matrices with DCC-GARCH(1,1) dynamics.

    The model follows Engle (2002):

    **Univariate GARCH(1,1) for each asset** *i*:

        h_{i,t} = ω_i + α_i · r_{i,t-1}² + β_i · h_{i,t-1}

    **DCC quasi-correlation dynamics**:

        Q_t = (1 − a − b) · Q̄ + a · u_{t-1} u_{t-1}′ + b · Q_{t-1}
        R_t = diag(Q_t)^{-½} Q_t diag(Q_t)^{-½}

    **Daily covariance**:

        Σ_t = D_t R_t D_t,   D_t = diag(√h_{1,t}, …, √h_{N,t})

    Parameters are chosen to be empirically plausible for U.S. equity returns:
    α ∈ [0.06, 0.10], β ∈ [0.88, 0.92], a = 0.04, b = 0.94, daily
    unconditional volatilities ∈ [1 %, 3 %].

    Parameters
    ----------
    n_assets : int
        Number of assets N (must be ≥ 1).
    n_days : int
        Number of simulated trading days T (must be ≥ 1).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of np.ndarray
        List of length T, each element a symmetric PD (N, N) covariance matrix.

    Raises
    ------
    ValueError
        If *n_assets* < 1 or *n_days* < 1.
    """
    if n_assets < 1:
        raise ValueError(f"simulate_garch_covariance: n_assets must be ≥ 1; got {n_assets}.")
    if n_days < 1:
        raise ValueError(f"simulate_garch_covariance: n_days must be ≥ 1; got {n_days}.")

    rng = np.random.default_rng(seed)

    # -- GARCH(1,1) parameters (asset-specific) --
    alpha = rng.uniform(0.05, 0.09, n_assets)   # ARCH coefficient
    beta = rng.uniform(0.88, 0.91, n_assets)    # GARCH coefficient
    # Enforce strict covariance-stationarity: alpha + beta < 1
    persist = alpha + beta
    max_persist = 0.98
    scale = np.where(persist > max_persist, max_persist / persist, 1.0)
    alpha *= scale
    beta *= scale
    # Unconditional daily std: roughly 1–3 % (annualised ~16–48 %)
    sig_bar = rng.uniform(0.010, 0.030, n_assets)
    omega = sig_bar ** 2 * (1.0 - alpha - beta)  # guaranteed positive

    # -- DCC parameters --
    a_dcc = 0.04
    b_dcc = 0.94

    # -- Unconditional correlation matrix Q̄ (factor model) --
    n_factors = max(1, n_assets // 3)
    Q_bar = _make_factor_correlation(n_assets, n_factors, rng)

    # -- Initialization --
    h = sig_bar ** 2           # current per-asset variances
    Q = Q_bar.copy()           # DCC quasi-correlation matrix

    # Burn-in to reach stationarity
    n_burnin = 500
    for _ in range(n_burnin):
        q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q), 1e-12)))
        R = q_diag_inv_sqrt @ Q @ q_diag_inv_sqrt
        R = (R + R.T) / 2.0
        np.fill_diagonal(R, 1.0)
        if not is_pd(R):
            R = nearest_pd(R)
            d = np.sqrt(np.diag(R))
            R = R / np.outer(d, d)
            np.fill_diagonal(R, 1.0)

        L_R = np.linalg.cholesky(R)
        z = rng.standard_normal(n_assets)
        u = L_R @ z                   # correlated standardized shock
        D = np.diag(np.sqrt(h))
        r = D @ u                     # actual return

        h = omega + alpha * r ** 2 + beta * h
        h = np.maximum(h, 1e-12)
        Q = (1.0 - a_dcc - b_dcc) * Q_bar + a_dcc * np.outer(u, u) + b_dcc * Q

    # -- Main simulation --
    cov_list = []
    for _ in range(n_days):
        # Current R_t from Q_t
        q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q), 1e-12)))
        R_t = q_diag_inv_sqrt @ Q @ q_diag_inv_sqrt
        R_t = (R_t + R_t.T) / 2.0
        np.fill_diagonal(R_t, 1.0)
        if not is_pd(R_t):
            R_t = nearest_pd(R_t)
            d = np.sqrt(np.diag(R_t))
            R_t = R_t / np.outer(d, d)
            np.fill_diagonal(R_t, 1.0)

        D_t = np.diag(np.sqrt(h))
        Sigma_t = D_t @ R_t @ D_t
        Sigma_t = (Sigma_t + Sigma_t.T) / 2.0
        if not is_pd(Sigma_t):
            Sigma_t = nearest_pd(Sigma_t)
        cov_list.append(Sigma_t)

        # Generate standardized shock for DCC/GARCH update
        L_R = np.linalg.cholesky(R_t)
        z = rng.standard_normal(n_assets)
        u = L_R @ z
        r = D_t @ u

        # Update GARCH variances (clamp to stay positive)
        h = omega + alpha * r ** 2 + beta * h
        h = np.maximum(h, 1e-12)
        # Update DCC quasi-correlation
        Q = (1.0 - a_dcc - b_dcc) * Q_bar + a_dcc * np.outer(u, u) + b_dcc * Q

    return cov_list


# ---------------------------------------------------------------------------
# Intraday return generation
# ---------------------------------------------------------------------------

def generate_intraday_returns(
    Sigma: np.ndarray,
    M: int,
    diurnal_params: Optional[dict] = None,
    seed_offset: int = 0,
) -> np.ndarray:
    """Generate M intraday returns for one day with integrated covariance Σ.

    Applies a U-shaped diurnal volatility pattern::

        σ_d(u) = C + A·exp(−a·u) + B·exp(−b·(1−u)),   u ∈ [0, 1]

    with default parameters A = 0.75, B = 0.25, C = 0.88929198, a = b = 10.

    The per-interval weights are normalized so that the expected integrated
    covariance equals Σ exactly:

        w_k = σ_d(k/M) / √(Σ_k σ_d(k/M)²),   k = 1, …, M
        r_k = w_k · L · z_k,                   z_k ~ N(0, I)

    where L is the Cholesky factor of Σ.  Then Σ_{k=1}^M E[r_k r_k′] = Σ.

    Parameters
    ----------
    Sigma : np.ndarray
        (N, N) positive-definite integrated covariance matrix for the day.
    M : int
        Number of intraday intervals (e.g. 78 for 5-minute bars in a 6.5-hour
        trading day).
    diurnal_params : dict, optional
        Override any of the diurnal parameters: keys ``'A'``, ``'B'``, ``'C'``,
        ``'a'``, ``'b'``.  Missing keys default to their standard values.
    seed_offset : int, optional
        Seed for the random number generator (allows per-day reproducibility).

    Returns
    -------
    np.ndarray
        Intraday return matrix of shape (M, N).

    Raises
    ------
    ValueError
        If Σ is not a square PD array, or if M < 1.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(
            f"generate_intraday_returns: Sigma must be square 2-D; got {Sigma.shape}."
        )
    if M < 1:
        raise ValueError(f"generate_intraday_returns: M must be ≥ 1; got {M}.")

    if not is_pd(Sigma):
        logger.debug("generate_intraday_returns: Sigma is not PD; projecting.")
        Sigma = nearest_pd(Sigma)

    N = Sigma.shape[0]
    rng = np.random.default_rng(seed_offset)

    # -- Diurnal parameters --
    defaults = {"A": 0.75, "B": 0.25, "C": 0.88929198, "a": 10.0, "b": 10.0}
    if diurnal_params is not None:
        defaults.update(diurnal_params)
    A_d = defaults["A"]
    B_d = defaults["B"]
    C_d = defaults["C"]
    a_d = defaults["a"]
    b_d = defaults["b"]

    # u_k = k/M for k = 1, …, M (end-of-interval time points)
    u = np.arange(1, M + 1) / M
    sigma_d = C_d + A_d * np.exp(-a_d * u) + B_d * np.exp(-b_d * (1.0 - u))

    # Normalize weights so Σ_k w_k² = 1  ⟹  E[Σ_k r_k r_k′] = Σ
    weights = sigma_d / np.sqrt(np.sum(sigma_d ** 2))  # shape (M,)

    L = np.linalg.cholesky(Sigma)                # (N, N)
    Z = rng.standard_normal((M, N))              # (M, N)
    # r_k = w_k · L · z_k  ⟺  returns[k, :] = weights[k] * (L @ Z[k, :])
    returns = (Z @ L.T) * weights[:, np.newaxis]  # (M, N)

    return returns


# ---------------------------------------------------------------------------
# Full synthetic dataset
# ---------------------------------------------------------------------------

def generate_dataset(
    n_assets: int = 10,
    n_days: int = 2300,
    M_intraday: int = 78,
    estimation_window: int = 1000,
    seed: int = 42,
) -> dict:
    """Generate a complete synthetic dataset for realized covariance experiments.

    Simulates *n_days* days of intraday returns (each day has *M_intraday*
    intervals) using DCC-GARCH covariance dynamics.  Daily open-to-close
    returns are obtained by summing the intraday returns within each day.

    Parameters
    ----------
    n_assets : int, optional
        Number of assets (default: 10, matching the paper's DJIA subset).
    n_days : int, optional
        Total number of trading days (default: 2300, ≈ 9 years at 252 days/year).
    M_intraday : int, optional
        Number of intraday intervals per day (default: 78, i.e. 5-minute bars
        in a 6.5-hour trading session).
    estimation_window : int, optional
        Rolling estimation window length stored in the metadata (default: 1000).
        Not used for data generation itself.
    seed : int, optional
        Master random seed.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``'intraday_returns'``: list of (M, N) arrays, one per day.
        * ``'daily_returns'``: (T, N) array of daily open-to-close log-returns.
        * ``'true_cov'``: list of (N, N) latent covariance matrices.
        * ``'asset_names'``: list of N ticker-like strings.
        * ``'n_days'``: int
        * ``'n_assets'``: int
        * ``'M_intraday'``: int

    Raises
    ------
    ValueError
        If any of the dimension arguments are non-positive.
    """
    for name, val in [
        ("n_assets", n_assets),
        ("n_days", n_days),
        ("M_intraday", M_intraday),
        ("estimation_window", estimation_window),
    ]:
        if val < 1:
            raise ValueError(f"generate_dataset: {name} must be ≥ 1; got {val}.")

    logger.info(
        "Generating synthetic dataset: n_assets=%d, n_days=%d, M_intraday=%d, seed=%d",
        n_assets, n_days, M_intraday, seed,
    )

    # Simulate latent covariance matrices
    true_cov = simulate_garch_covariance(n_assets, n_days, seed=seed)

    # Generate intraday returns for each day
    intraday_returns = []
    daily_returns = np.zeros((n_days, n_assets), dtype=float)

    for t in range(n_days):
        # Each day gets a unique seed derived from the master seed
        ret_t = generate_intraday_returns(
            true_cov[t], M_intraday, seed_offset=seed * n_days + t
        )
        intraday_returns.append(ret_t)
        # Daily return = sum of intraday log-returns (continuously compounded)
        daily_returns[t] = ret_t.sum(axis=0)

    # Asset names: use the standard universe if n_assets matches, else generic
    if n_assets == len(ASSET_UNIVERSE):
        asset_names = list(ASSET_UNIVERSE)
    else:
        asset_names = [f"ASSET_{i + 1:02d}" for i in range(n_assets)]

    logger.info("Dataset generation complete.")

    return {
        "intraday_returns": intraday_returns,
        "daily_returns": daily_returns,
        "true_cov": true_cov,
        "asset_names": asset_names,
        "n_days": n_days,
        "n_assets": n_assets,
        "M_intraday": M_intraday,
    }


# ---------------------------------------------------------------------------
# Massive API loader
# ---------------------------------------------------------------------------

def _build_intraday_returns_from_ohlcv(
    df: pd.DataFrame, tickers: list
) -> tuple:
    """Convert a multi-ticker OHLCV dataframe to intraday and daily return arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``['ticker', 'timestamp', 'open', 'close']`` at minimum.
    tickers : list
        Ordered list of tickers defining column order in output arrays.

    Returns
    -------
    tuple
        (intraday_returns_list, daily_returns_array) where
        *intraday_returns_list* is a list of (M_t, N) per-day log-return matrices
        and *daily_returns_array* is (T, N).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["log_ret"] = np.log(df["close"] / df["open"])

    dates = sorted(df["date"].unique())
    N = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    intraday_returns_list = []
    daily_returns = []

    for date in dates:
        day_df = df[df["date"] == date].sort_values("timestamp")
        times = day_df["timestamp"].unique()
        M = len(times)

        mat = np.zeros((M, N), dtype=float)
        for j, ts in enumerate(sorted(times)):
            slice_ts = day_df[day_df["timestamp"] == ts]
            for _, row in slice_ts.iterrows():
                if row["ticker"] in ticker_idx:
                    mat[j, ticker_idx[row["ticker"]]] = row["log_ret"]

        intraday_returns_list.append(mat)
        daily_returns.append(mat.sum(axis=0))

    return intraday_returns_list, np.array(daily_returns, dtype=float)


def load_massive_data(
    tickers: list,
    start_date: str,
    end_date: str,
    timespan: str = "minute",
    multiplier: int = 5,
) -> Optional[dict]:
    """Attempt to load intraday data from the Massive financial data API.

    Requires the environment variable ``MASSIVE_TOKEN`` to be set.  Returns
    ``None`` immediately if the token is absent or if any network/import error
    occurs, so callers can fall back to synthetic data transparently.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols to fetch.
    start_date : str
        Start date in ``'YYYY-MM-DD'`` format (inclusive).
    end_date : str
        End date in ``'YYYY-MM-DD'`` format (inclusive).
    timespan : str, optional
        Bar resolution (default: ``'minute'``).
    multiplier : int, optional
        Bar size multiplier; together with *timespan* defines the bar width.
        E.g. multiplier=5, timespan='minute' gives 5-minute bars.

    Returns
    -------
    dict or None
        On success, a dict with keys:

        * ``'intraday_returns'``: list of (M, N) per-day log-return matrices.
        * ``'daily_returns'``: (T, N) daily return array.
        * ``'tickers'``: ordered list of tickers.
        * ``'dates'``: sorted list of trading dates (as Python ``date`` objects).

        Returns ``None`` if the API is unavailable or the token is missing.
    """
    token = os.environ.get("MASSIVE_TOKEN")
    if not token:
        logger.warning(
            "load_massive_data: MASSIVE_TOKEN environment variable not set; "
            "returning None."
        )
        return None

    try:
        import massive  # type: ignore[import]
    except ImportError:
        logger.warning(
            "load_massive_data: 'massive' package is not installed; returning None."
        )
        return None

    try:
        client = massive.Client(token=token)
        frames = []
        for ticker in tickers:
            logger.info("load_massive_data: fetching %s …", ticker)
            df_ticker = client.get_aggregates(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
            )
            if df_ticker is None or len(df_ticker) == 0:
                logger.warning("load_massive_data: no data returned for %s.", ticker)
                continue
            df_ticker = df_ticker.copy()
            df_ticker["ticker"] = ticker
            frames.append(df_ticker)

        if not frames:
            logger.warning("load_massive_data: no data retrieved for any ticker.")
            return None

        df_all = pd.concat(frames, ignore_index=True)
        intraday, daily = _build_intraday_returns_from_ohlcv(df_all, tickers)

        dates = sorted(
            pd.to_datetime(df_all["timestamp"]).dt.date.unique()
        )

        return {
            "intraday_returns": intraday,
            "daily_returns": daily,
            "tickers": tickers,
            "dates": dates,
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_massive_data: failed to load data (%s: %s); returning None.",
            type(exc).__name__,
            exc,
        )
        return None
