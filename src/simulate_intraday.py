"""Intraday return simulation and aggregation for the Experiment 3 study.

Generates synthetic 1-second price paths with a U-shaped diurnal volatility
pattern, aggregates to multiple sampling frequencies, and runs the full
simulation study comparing HAR/HARQ models across aggregation levels.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import vech, ivech, vech_size, nearest_pd, safe_inv, is_pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diurnal volatility
# ---------------------------------------------------------------------------

def diurnal_sigma(
    u: float,
    A: float = 0.75,
    B: float = 0.25,
    C: float = 0.88929198,
    a: float = 10.0,
    b: float = 10.0,
) -> float:
    """U-shaped diurnal volatility function.

    .. math::

        \\sigma_d(u) = C + A \\exp(-a u) + B \\exp(-b(1-u)), \\quad u \\in [0, 1]

    Captures the well-known U-shape pattern in intraday equity volatility
    (high at open and close, lower at midday).

    Parameters
    ----------
    u : float
        Normalized time within the trading day, ``u ∈ [0, 1]``.
    A : float, optional
        Amplitude of the open-of-day decay component (default: 0.75).
    B : float, optional
        Amplitude of the end-of-day surge component (default: 0.25).
    C : float, optional
        Baseline level ensuring the function integrates to 1 over [0,1]
        (default: 0.88929198).
    a : float, optional
        Decay rate for the morning component (default: 10.0).
    b : float, optional
        Decay rate for the afternoon component (default: 10.0).

    Returns
    -------
    float
        Diurnal volatility weight at normalized time *u* (positive).

    Raises
    ------
    ValueError
        If *u* is outside [0, 1].
    """
    if not (0.0 <= u <= 1.0):
        raise ValueError(f"diurnal_sigma: u must be in [0, 1]; got {u}.")
    return C + A * np.exp(-a * u) + B * np.exp(-b * (1.0 - u))


def compute_diurnal_weights(n_seconds: int = 23400) -> np.ndarray:
    """Compute normalized diurnal scaling weights for each intraday second.

    Evaluates :func:`diurnal_sigma` at each second ``i/n_seconds`` for
    ``i = 0, 1, …, n_seconds - 1``, then normalizes so that the sum equals
    *n_seconds* (i.e., the discrete integral approximates 1 when the weights
    are used to scale a covariance that already integrates to Σ).

    Parameters
    ----------
    n_seconds : int, optional
        Number of 1-second intervals in the trading day (default: 23400,
        corresponding to a 6.5-hour session: 6.5 × 3600).

    Returns
    -------
    np.ndarray
        Weight array of shape (n_seconds,) with ``sum == n_seconds``.

    Raises
    ------
    ValueError
        If *n_seconds* < 1.
    """
    if n_seconds < 1:
        raise ValueError(
            f"compute_diurnal_weights: n_seconds must be ≥ 1; got {n_seconds}."
        )
    raw = np.array(
        [diurnal_sigma(i / n_seconds) for i in range(n_seconds)],
        dtype=float,
    )
    # Normalize so that mean(raw) = 1  →  sum(raw) = n_seconds
    weights = raw / raw.mean()
    return weights


# ---------------------------------------------------------------------------
# One-day simulation
# ---------------------------------------------------------------------------

def simulate_one_day_intraday(
    Sigma: np.ndarray,
    n_seconds: int = 23400,
    seed: int = 0,
) -> np.ndarray:
    """Simulate one day of 1-second intraday returns with latent covariance Σ.

    Each second *i* draws an i.i.d. Gaussian increment whose covariance
    is proportional to the diurnal weight ``w_i``.  The weights are
    normalized so that the integrated covariance equals Σ exactly in
    expectation:

    .. math::

        r_i \\sim \\mathcal{N}\\!\\left(0,\\; \\frac{w_i}{n} \\Sigma\\right),
        \\quad \\sum_{i=1}^{n} \\frac{w_i}{n} = 1

    Parameters
    ----------
    Sigma : np.ndarray
        Latent (N, N) positive-definite covariance matrix for the day.
    n_seconds : int, optional
        Number of 1-second intervals (default: 23400).
    seed : int, optional
        Random seed for this day (default: 0).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_seconds, N)`` containing 1-second log-returns.

    Raises
    ------
    ValueError
        If *Sigma* is not a 2-D square array or *n_seconds* < 1.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(
            f"simulate_one_day_intraday: Sigma must be square 2-D; got {Sigma.shape}."
        )
    if n_seconds < 1:
        raise ValueError(
            f"simulate_one_day_intraday: n_seconds must be ≥ 1; got {n_seconds}."
        )

    N = Sigma.shape[0]
    if not is_pd(Sigma):
        Sigma = nearest_pd(Sigma)

    L = np.linalg.cholesky(Sigma)           # Cholesky factor of Σ
    weights = compute_diurnal_weights(n_seconds)   # shape (n_seconds,), sum = n_seconds

    # Scale so each increment has covariance (w_i / n_seconds) * Sigma
    # → sum over i of E[r_i r_i'] = Sigma
    scale_factors = np.sqrt(weights / n_seconds)   # shape (n_seconds,)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_seconds, N))         # i.i.d. N(0, I) shocks
    returns = (Z @ L.T) * scale_factors[:, np.newaxis]  # (n_seconds, N)

    return returns


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_returns(intraday_returns: np.ndarray, M: int) -> np.ndarray:
    """Aggregate 1-second returns to *M* equally-spaced intervals.

    Sums consecutive blocks of ``n_seconds // M`` 1-second returns.
    *n_seconds* must be divisible by ``n_seconds // M`` (equivalently, *M*
    must divide *n_seconds*).

    Parameters
    ----------
    intraday_returns : np.ndarray
        Array of shape ``(n_seconds, N)`` of 1-second returns.
    M : int
        Number of intervals after aggregation (must be a divisor of
        *n_seconds*).

    Returns
    -------
    np.ndarray
        Aggregated return array of shape ``(M, N)``.

    Raises
    ------
    ValueError
        If *intraday_returns* is not 2-D, *M* < 1, or *n_seconds* is not
        divisible by ``n_seconds // M``.
    """
    intraday_returns = np.asarray(intraday_returns, dtype=float)
    if intraday_returns.ndim != 2:
        raise ValueError(
            "aggregate_returns: intraday_returns must be 2-D; "
            f"got {intraday_returns.ndim}-D."
        )
    n_seconds, N = intraday_returns.shape
    if M < 1:
        raise ValueError(f"aggregate_returns: M must be ≥ 1; got {M}.")
    if n_seconds % M != 0:
        raise ValueError(
            f"aggregate_returns: n_seconds={n_seconds} is not divisible by M={M}."
        )

    block = n_seconds // M
    # Reshape to (M, block, N) and sum over the block dimension
    reshaped = intraday_returns.reshape(M, block, N)
    return reshaped.sum(axis=1)


# ---------------------------------------------------------------------------
# Weight-distance helpers
# ---------------------------------------------------------------------------

def compute_weight_distance(w1: np.ndarray, w2: np.ndarray) -> float:
    """Euclidean distance between two portfolio weight vectors.

    Parameters
    ----------
    w1 : np.ndarray
        Weight vector of shape (N,).
    w2 : np.ndarray
        Weight vector of shape (N,).

    Returns
    -------
    float
        Euclidean (L2) distance.

    Raises
    ------
    ValueError
        If the arrays have different shapes.
    """
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    if w1.shape != w2.shape:
        raise ValueError(
            f"compute_weight_distance: shape mismatch {w1.shape} vs {w2.shape}."
        )
    return float(np.linalg.norm(w1 - w2))


def compute_optimal_gmv_weights(true_cov_path: list) -> np.ndarray:
    """Compute ex-post optimal GMV weights using true latent covariance matrices.

    These weights represent the infeasible oracle portfolio that uses the
    true (unobservable) covariance matrix each day.

    Parameters
    ----------
    true_cov_path : list of np.ndarray
        List of T (N, N) true covariance matrices.

    Returns
    -------
    np.ndarray
        Weight matrix of shape (T, N).

    Raises
    ------
    ValueError
        If *true_cov_path* is empty.
    """
    if len(true_cov_path) == 0:
        raise ValueError("compute_optimal_gmv_weights: true_cov_path is empty.")

    from src.portfolio import gmv_weights_unconstrained

    T = len(true_cov_path)
    N = np.asarray(true_cov_path[0]).shape[0]
    weights = np.zeros((T, N), dtype=float)

    for t, Sigma_t in enumerate(true_cov_path):
        Sigma_t = np.asarray(Sigma_t, dtype=float)
        if not is_pd(Sigma_t):
            Sigma_t = nearest_pd(Sigma_t)
        weights[t] = gmv_weights_unconstrained(Sigma_t)

    return weights


def compute_har_infinity_weights(
    true_cov_path: list,
    estimation_window: int,
) -> np.ndarray:
    """Fit HAR on true latent covariances and derive GMV weights.

    Fits the HAR model directly on the latent (true) covariance matrices
    using a rolling window, then uses the forecasts to form GMV weights.
    This represents the "oracle HAR" that has access to noiseless realized
    measures.

    Parameters
    ----------
    true_cov_path : list of np.ndarray
        List of T (N, N) true covariance matrices.
    estimation_window : int
        Rolling estimation window length (must be ≥ 23).

    Returns
    -------
    np.ndarray
        Weight matrix of shape (T - estimation_window, N).

    Raises
    ------
    ValueError
        If *true_cov_path* is too short or *estimation_window* is invalid.
    """
    T = len(true_cov_path)
    if T < estimation_window + 1:
        raise ValueError(
            f"compute_har_infinity_weights: need T ≥ estimation_window + 1; "
            f"got T={T}, window={estimation_window}."
        )
    if estimation_window < 23:
        raise ValueError(
            f"compute_har_infinity_weights: estimation_window must be ≥ 23; "
            f"got {estimation_window}."
        )

    N = np.asarray(true_cov_path[0]).shape[0]
    K = vech_size(N)

    # Build full vech series
    s_series = np.zeros((T, K), dtype=float)
    for t, Sigma_t in enumerate(true_cov_path):
        s_series[t] = vech(np.asarray(Sigma_t, dtype=float))

    from src.portfolio import gmv_weights_unconstrained

    T_oos = T - estimation_window
    weights = np.zeros((T_oos, N), dtype=float)

    for idx in range(T_oos):
        t = estimation_window + idx
        window_s = s_series[t - estimation_window : t]   # (W, K)
        W = len(window_s)

        # Build HAR design matrix for the window
        # Target: window_s[22:W], shape (W-22, K)
        Y = window_s[22:W]
        n_obs = W - 22

        X = np.ones((n_obs, 4), dtype=float)
        for row in range(n_obs):
            i = row + 22    # index in window_s
            X[row, 1] = window_s[i - 1].mean()          # scalar lag-1 (pooled)
            X[row, 2] = window_s[i - 5 : i].mean()      # scalar lag-5
            X[row, 3] = window_s[i - 22 : i].mean()     # scalar lag-22

        # OLS: beta shape (4, K)
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

        # Forecast for day t
        x_new = np.array([
            1.0,
            window_s[-1].mean(),
            window_s[-5:].mean(),
            window_s[-22:].mean(),
        ])
        h_vech = x_new @ beta   # (K,)
        H_t = ivech(h_vech, N)
        H_t = (H_t + H_t.T) / 2.0
        if not is_pd(H_t):
            H_t = nearest_pd(H_t)

        weights[idx] = gmv_weights_unconstrained(H_t)

    return weights


# ---------------------------------------------------------------------------
# Main simulation study
# ---------------------------------------------------------------------------

def run_simulation_study(
    true_cov_path: list,
    estimation_window: int = 1000,
    n_oos: int = 1000,
    aggregation_levels: list = None,
    seed: int = 42,
) -> dict:
    """Run the Experiment 3 simulation study across aggregation levels.

    For each sampling frequency M in *aggregation_levels*:

    1. Simulate 1-second returns for each of the ``estimation_window + n_oos``
       days using the true latent covariances.
    2. Aggregate to M-frequency returns.
    3. Compute realized covariance estimates (realized kernel proxy) at each
       frequency.
    4. Fit HAR and HARQ on rolling *estimation_window*-day windows.
    5. Compute GMV weights for each model and compare to oracle weights.
    6. Report mean turnover, ex-post portfolio variance, and mean weight
       distance.

    Parameters
    ----------
    true_cov_path : list of np.ndarray
        List of T (N, N) true latent covariance matrices (T ≥
        ``estimation_window + n_oos``).
    estimation_window : int, optional
        Rolling estimation window length (default: 1000).
    n_oos : int, optional
        Number of out-of-sample days (default: 1000).
    aggregation_levels : list of int, optional
        Sampling frequencies M (default: [390, 78, 26] corresponding to
        1-min, 5-min, and 15-min bars in a 6.5-hour day).
    seed : int, optional
        Master random seed (default: 42).

    Returns
    -------
    dict
        ``{M: {'har': {...}, 'harq': {...}, 'optimal': {...}}}`` where the
        inner dicts contain ``'turnover'``, ``'ex_post_var'``, and
        ``'weight_dist'`` arrays of length *n_oos*.

    Raises
    ------
    ValueError
        If *true_cov_path* has fewer than ``estimation_window + n_oos`` days.
    """
    if aggregation_levels is None:
        aggregation_levels = [390, 78, 26]

    T_total = len(true_cov_path)
    required = estimation_window + n_oos
    if T_total < required:
        raise ValueError(
            f"run_simulation_study: need {required} days; got T={T_total}."
        )

    N = np.asarray(true_cov_path[0]).shape[0]
    K = vech_size(N)

    # Oracle GMV weights using true covariances
    oracle_w = compute_optimal_gmv_weights(true_cov_path[-n_oos:])  # (n_oos, N)

    from src.portfolio import gmv_weights_unconstrained, compute_turnover

    results: dict = {}

    for M in aggregation_levels:
        logger.info("run_simulation_study: M=%d (aggregation level).", M)
        n_seconds = 23400   # seconds per trading day

        # Check divisibility
        if n_seconds % M != 0:
            logger.warning(
                "n_seconds=%d is not divisible by M=%d; skipping.", n_seconds, M
            )
            continue

        # ------------------------------------------------------------------
        # Step 1 & 2: Simulate and aggregate returns
        # ------------------------------------------------------------------
        all_rc = []   # realized covariances at frequency M (simple outer product sum)
        all_pi = []   # uncertainty proxies (diagonal of quarticity estimate)

        for day_idx in range(T_total):
            Sigma_day = np.asarray(true_cov_path[day_idx])
            day_seed = seed * T_total + day_idx
            intra = simulate_one_day_intraday(Sigma_day, n_seconds=n_seconds, seed=day_seed)
            agg = aggregate_returns(intra, M)   # (M, N)

            # Realized covariance (simple sum of outer products, no kernel)
            RC = agg.T @ agg
            RC = (RC + RC.T) / 2.0
            if not is_pd(RC):
                RC = nearest_pd(RC)
            all_rc.append(RC)

            # Uncertainty proxy: diagonal of quarticity (sum of r^4 elements)
            # Simple proxy: use the diagonal of M * (agg^2)' (agg^2)
            agg_sq = agg ** 2   # element-wise square
            IQ_diag = M * np.sum(agg_sq ** 2, axis=0)   # (N,)
            # Extend to vech diagonal as sqrt of per-element quarticity
            pi_t = np.zeros(K, dtype=float)
            for k_idx in range(K):
                pi_t[k_idx] = np.sqrt(max(IQ_diag.mean(), 0.0))
            all_pi.append(pi_t)

        s_series = np.zeros((T_total, K), dtype=float)
        for t in range(T_total):
            s_series[t] = vech(all_rc[t])
        pi_series = np.array(all_pi, dtype=float)   # (T_total, K)

        # ------------------------------------------------------------------
        # Step 3: Rolling HAR / HARQ estimation
        # ------------------------------------------------------------------
        har_weights = np.zeros((n_oos, N), dtype=float)
        harq_weights = np.zeros((n_oos, N), dtype=float)

        for oos_idx in range(n_oos):
            t = estimation_window + oos_idx
            window_s = s_series[t - estimation_window : t]
            window_pi = pi_series[t - estimation_window : t]

            # Demean pi within window
            pi_mean = window_pi.mean(axis=0)
            window_pi_dm = window_pi - pi_mean

            # HAR forecast
            H_har = _har_forecast(window_s, N)
            if not is_pd(H_har):
                H_har = nearest_pd(H_har)
            har_weights[oos_idx] = gmv_weights_unconstrained(H_har)

            # HARQ forecast
            H_harq = _harq_forecast(window_s, window_pi_dm, N)
            if not is_pd(H_harq):
                H_harq = nearest_pd(H_harq)
            harq_weights[oos_idx] = gmv_weights_unconstrained(H_harq)

        # ------------------------------------------------------------------
        # Step 4: Evaluation
        # ------------------------------------------------------------------
        # Use the OOS subset of true covariances for ex-post variance
        oos_rc = np.array(
            [vech(all_rc[estimation_window + i]) for i in range(n_oos)], dtype=float
        )

        def _ex_post_var(weights: np.ndarray) -> np.ndarray:
            """Compute ex-post portfolio variance for each day."""
            vars_ = np.zeros(n_oos, dtype=float)
            for i in range(n_oos):
                Sigma_t = all_rc[estimation_window + i]
                vars_[i] = float(weights[i] @ Sigma_t @ weights[i])
            return vars_

        def _weight_dist(weights: np.ndarray) -> np.ndarray:
            """Compute daily distance from oracle weights."""
            dists = np.zeros(n_oos, dtype=float)
            for i in range(n_oos):
                dists[i] = compute_weight_distance(weights[i], oracle_w[i])
            return dists

        def _turnover_arr(weights: np.ndarray) -> np.ndarray:
            """Compute daily turnover (set fake returns to zero for this metric)."""
            fake_rets = np.zeros((n_oos, N), dtype=float)
            return compute_turnover(weights, fake_rets)

        results[M] = {
            "har": {
                "turnover": _turnover_arr(har_weights),
                "ex_post_var": _ex_post_var(har_weights),
                "weight_dist": _weight_dist(har_weights),
                "weights": har_weights,
            },
            "harq": {
                "turnover": _turnover_arr(harq_weights),
                "ex_post_var": _ex_post_var(harq_weights),
                "weight_dist": _weight_dist(harq_weights),
                "weights": harq_weights,
            },
            "optimal": {
                "turnover": _turnover_arr(oracle_w),
                "ex_post_var": _ex_post_var(oracle_w),
                "weight_dist": np.zeros(n_oos, dtype=float),
                "weights": oracle_w,
            },
        }
        logger.info(
            "run_simulation_study: M=%d complete; "
            "HAR mean ex-post var=%.4g, HARQ=%.4g.",
            M,
            results[M]["har"]["ex_post_var"].mean(),
            results[M]["harq"]["ex_post_var"].mean(),
        )

    return results


def _har_forecast(window_s: np.ndarray, N: int) -> np.ndarray:
    """Internal helper: HAR forecast from a window of vech series."""
    W, K = window_s.shape
    if W < 23:
        # Not enough data; return average
        return ivech(window_s.mean(axis=0), N)

    n_obs = W - 22
    Y = window_s[22:W]
    X = np.ones((n_obs, 4), dtype=float)
    for row in range(n_obs):
        i = row + 22
        X[row, 1] = window_s[i - 1].mean()
        X[row, 2] = window_s[max(0, i - 5) : i].mean()
        X[row, 3] = window_s[i - 22 : i].mean()

    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (4, K)

    x_new = np.array([
        1.0,
        window_s[-1].mean(),
        window_s[-5:].mean(),
        window_s[-22:].mean(),
    ])
    h_vech = x_new @ beta
    H = ivech(h_vech, N)
    return (H + H.T) / 2.0


def _harq_forecast(window_s: np.ndarray, window_pi: np.ndarray, N: int) -> np.ndarray:
    """Internal helper: HARQ forecast from a window of vech and pi series."""
    W, K = window_s.shape
    if W < 23:
        return ivech(window_s.mean(axis=0), N)

    n_obs = W - 22
    Y = window_s[22:W]
    X = np.ones((n_obs, 5), dtype=float)
    for row in range(n_obs):
        i = row + 22
        X[row, 1] = window_s[i - 1].mean()
        X[row, 2] = window_s[max(0, i - 5) : i].mean()
        X[row, 3] = window_s[i - 22 : i].mean()
        # HARQ term: pi_{t-1} * s_{t-1} (pooled scalar version)
        X[row, 4] = window_pi[i - 1].mean() * window_s[i - 1].mean()

    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (5, K)

    x_new = np.array([
        1.0,
        window_s[-1].mean(),
        window_s[-5:].mean(),
        window_s[-22:].mean(),
        window_pi[-1].mean() * window_s[-1].mean(),
    ])
    h_vech = x_new @ beta
    H = ivech(h_vech, N)
    return (H + H.T) / 2.0


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def simulation_results_table(results: dict) -> pd.DataFrame:
    """Format simulation results as a tidy summary DataFrame.

    Parameters
    ----------
    results : dict
        Output from :func:`run_simulation_study`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['M', 'model', 'mean_turnover',
        'mean_ex_post_var', 'mean_weight_dist']``, one row per
        (aggregation level, model) pair.
    """
    rows: list = []
    for M, M_results in results.items():
        for model_name, metrics in M_results.items():
            to_arr = metrics.get("turnover", np.array([]))
            var_arr = metrics.get("ex_post_var", np.array([]))
            dist_arr = metrics.get("weight_dist", np.array([]))
            rows.append({
                "M": M,
                "model": model_name,
                "mean_turnover": float(to_arr.mean()) if len(to_arr) > 0 else np.nan,
                "mean_ex_post_var": float(var_arr.mean()) if len(var_arr) > 0 else np.nan,
                "mean_weight_dist": float(dist_arr.mean()) if len(dist_arr) > 0 else np.nan,
            })
    return pd.DataFrame(rows)
