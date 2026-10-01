"""Rolling one-step-ahead covariance forecast engine.

Implements a rolling-window estimation and forecasting loop for eight
realized covariance models:

    HAR, HARQ, HAR-DRD, HARQ-DRD, EWMA, EWMAQ, HEAVY, HEAVYQ

The models are estimated on a fixed-length rolling window and produce
one-step-ahead (N × N) covariance matrix forecasts for each out-of-sample
day.  Forecast losses are evaluated against the realized covariance proxy.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize

from src.utils import vech, ivech, vech_size, nearest_pd, safe_inv, is_pd
from src.drd_models import (
    DRDDecomposition,
    _decompose_series,
    _build_corr_matrix,
    _ensure_valid_corr,
    fit_univariate_har as _fit_univariate_har_drd,
    fit_univariate_harq as _fit_univariate_harq_drd,
    vech_diag_indices,
)
from src.har_models import HARModel

logger = logging.getLogger(__name__)

# All supported model names
MODEL_NAMES = ("HAR", "HARQ", "HAR-DRD", "HARQ-DRD", "EWMA", "EWMAQ", "HEAVY", "HEAVYQ")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rolling_forecast_all_models(
    s_series: np.ndarray,
    S_list: list,
    pi_series: np.ndarray,
    daily_returns: np.ndarray,
    estimation_window: int = 1000,
    N: int = 10,
) -> dict:
    """Rolling one-step-ahead forecast for all eight realized covariance models.

    For each out-of-sample day *t* (indices ``estimation_window`` through
    ``T-1``) the function:

    1. Extracts the current estimation window
       ``s_series[t-W : t]``, ``pi_series[t-W : t]``, etc.
    2. Demeaned pi within the window before passing to Q-models.
    3. Calls :func:`estimate_window_params` for each model.
    4. Calls :func:`forecast_one_step` to obtain H_{t|t-1}.
    5. Evaluates Frobenius and QLIKE losses against ``S_list[t]``.

    Progress is logged every 100 OOS days.

    Parameters
    ----------
    s_series : np.ndarray
        Array of shape (T, K) containing the daily vech of realized
        covariance matrices (K = N*(N+1)//2).
    S_list : list of np.ndarray
        List of T (N, N) realized covariance matrices.
    pi_series : np.ndarray
        Array of shape (T, K) containing the daily uncertainty vectors π_t.
    daily_returns : np.ndarray
        Array of shape (T, N) daily returns.
    estimation_window : int, optional
        Rolling window length W (default: 1000).  Needs ≥ 23 observations.
    N : int, optional
        Number of assets (default: 10).

    Returns
    -------
    dict
        One entry per model name (keys from :data:`MODEL_NAMES`).  Each value
        is a dict::

            {
              'forecasts':      list of (N, N) forecast matrices,
              'params_history': None,          # omitted to save memory
              'oos_indices':    list[int],
              'losses':         pd.DataFrame,  # columns: frobenius, qlike
              'pi_norms':       np.ndarray,    # ||π_t|| for each OOS day
            }

    Raises
    ------
    ValueError
        If array dimensions are inconsistent or T ≤ estimation_window.
    """
    s_series = np.asarray(s_series, dtype=float)
    pi_series = np.asarray(pi_series, dtype=float)
    daily_returns = np.asarray(daily_returns, dtype=float)

    T = s_series.shape[0]
    K = vech_size(N)

    if s_series.shape != (T, K):
        raise ValueError(
            f"rolling_forecast_all_models: s_series shape {s_series.shape} "
            f"does not match (T={T}, K={K})."
        )
    if pi_series.shape != (T, K):
        raise ValueError(
            f"rolling_forecast_all_models: pi_series shape {pi_series.shape} "
            f"does not match (T={T}, K={K})."
        )
    if len(S_list) != T:
        raise ValueError(
            f"rolling_forecast_all_models: S_list length {len(S_list)} != T={T}."
        )
    if daily_returns.shape[0] != T:
        raise ValueError(
            f"rolling_forecast_all_models: daily_returns rows {daily_returns.shape[0]} "
            f"!= T={T}."
        )
    if T <= estimation_window:
        raise ValueError(
            f"rolling_forecast_all_models: T={T} must exceed "
            f"estimation_window={estimation_window}."
        )

    # Initialise result containers
    results: dict = {
        name: {
            "forecasts": [],
            "params_history": None,
            "oos_indices": [],
            "losses": None,
            "pi_norms": [],
        }
        for name in MODEL_NAMES
    }

    from src.evaluation import frobenius_loss, qlike_loss

    n_oos = T - estimation_window
    for step, t in enumerate(range(estimation_window, T)):
        if step % 100 == 0:
            logger.info(
                "rolling_forecast_all_models: OOS step %d / %d (t=%d).",
                step + 1, n_oos, t,
            )

        # Extract windows
        s_window = s_series[t - estimation_window : t]          # (W, K)
        pi_window = pi_series[t - estimation_window : t]        # (W, K)
        S_window = S_list[t - estimation_window : t]
        returns_window = daily_returns[t - estimation_window : t]  # (W, N)

        # Demean pi within this window
        pi_mean = pi_window.mean(axis=0)
        pi_window_dm = pi_window - pi_mean

        # ||π_t|| using the demeaned value at the last window position
        pi_norm_t = float(np.linalg.norm(pi_window_dm[-1]))

        # Realized covariance for this OOS day
        S_t = np.asarray(S_list[t], dtype=float)

        for model_name in MODEL_NAMES:
            try:
                params = estimate_window_params(
                    s_window, pi_window_dm, S_window, returns_window, N, model_name
                )
                H_hat = forecast_one_step(
                    s_window, pi_window_dm, S_window, returns_window, N,
                    model_name, params
                )
            except Exception as exc:
                logger.warning(
                    "rolling_forecast_all_models: model '%s' failed at t=%d (%s); "
                    "using last observed covariance.",
                    model_name, t, exc,
                )
                H_hat = np.asarray(S_window[-1], dtype=float)

            # Ensure PD
            if not is_pd(H_hat):
                H_hat = nearest_pd(H_hat)

            fl = frobenius_loss(H_hat, S_t)
            ql = qlike_loss(H_hat, S_t)

            results[model_name]["forecasts"].append(H_hat)
            results[model_name]["oos_indices"].append(t)
            results[model_name]["pi_norms"].append(pi_norm_t)

            # Temporary loss storage
            if "frob_list" not in results[model_name]:
                results[model_name]["frob_list"] = []
                results[model_name]["qlike_list"] = []
            results[model_name]["frob_list"].append(fl)
            results[model_name]["qlike_list"].append(ql)

    # Build loss DataFrames and pi_norm arrays
    for model_name in MODEL_NAMES:
        frob = results[model_name].pop("frob_list", [])
        ql_ = results[model_name].pop("qlike_list", [])
        results[model_name]["losses"] = pd.DataFrame(
            {"frobenius": frob, "qlike": ql_}
        )
        results[model_name]["pi_norms"] = np.array(
            results[model_name]["pi_norms"], dtype=float
        )

    logger.info("rolling_forecast_all_models: complete (%d OOS days).", n_oos)
    return results


# ---------------------------------------------------------------------------
# Window-level estimation
# ---------------------------------------------------------------------------

def estimate_window_params(
    s_window: np.ndarray,
    pi_window: np.ndarray,
    S_window: list,
    returns_window: np.ndarray,
    N: int,
    model_name: str,
) -> dict:
    """Estimate model parameters for one rolling window.

    Parameters
    ----------
    s_window : np.ndarray
        Array of shape (W, K) of vech-stacked realized covariances in the
        estimation window.
    pi_window : np.ndarray
        Array of shape (W, K) of demeaned uncertainty vectors.
    S_window : list of np.ndarray
        List of W (N, N) realized covariance matrices.
    returns_window : np.ndarray
        Array of shape (W, N) of daily returns.
    N : int
        Number of assets.
    model_name : str
        One of ``'HAR'``, ``'HARQ'``, ``'HAR-DRD'``, ``'HARQ-DRD'``,
        ``'EWMA'``, ``'EWMAQ'``, ``'HEAVY'``, ``'HEAVYQ'``.

    Returns
    -------
    dict
        Model-specific parameter dictionary passed directly to
        :func:`forecast_one_step`.

    Raises
    ------
    ValueError
        If *model_name* is not recognized.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"estimate_window_params: unknown model '{model_name}'. "
            f"Valid: {MODEL_NAMES}."
        )
    K = s_window.shape[1]

    if model_name == "HAR":
        return _estimate_har(s_window)

    if model_name == "HARQ":
        return _estimate_harq(s_window, pi_window)

    if model_name == "HAR-DRD":
        return _estimate_har_drd(S_window, N)

    if model_name == "HARQ-DRD":
        return _estimate_harq_drd(S_window, pi_window, N)

    if model_name == "EWMA":
        return _estimate_ewma(s_window)

    if model_name == "EWMAQ":
        return _estimate_ewmaq(s_window, pi_window)

    if model_name == "HEAVY":
        return _estimate_heavy(s_window)

    if model_name == "HEAVYQ":
        return _estimate_heavyq(s_window, pi_window)

    raise ValueError(f"estimate_window_params: unhandled model '{model_name}'.")


def forecast_one_step(
    s_window: np.ndarray,
    pi_window: np.ndarray,
    S_window: list,
    returns_window: np.ndarray,
    N: int,
    model_name: str,
    params: dict,
) -> np.ndarray:
    """Produce one-step-ahead (N, N) covariance forecast.

    Uses the pre-estimated *params* dict returned by
    :func:`estimate_window_params` and the last observations in the window
    to generate H_{t|t-1}.

    Parameters
    ----------
    s_window : np.ndarray
        Array of shape (W, K) of in-window vech covariances.
    pi_window : np.ndarray
        Array of shape (W, K) of in-window demeaned uncertainty vectors.
    S_window : list of np.ndarray
        List of W (N, N) in-window realized covariance matrices.
    returns_window : np.ndarray
        Array of shape (W, N) in-window daily returns.
    N : int
        Number of assets.
    model_name : str
        Model identifier (one of :data:`MODEL_NAMES`).
    params : dict
        Parameter dict from :func:`estimate_window_params`.

    Returns
    -------
    np.ndarray
        Symmetric (N, N) covariance forecast, projected to the nearest PD
        matrix if necessary.

    Raises
    ------
    ValueError
        If *model_name* is not recognized.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"forecast_one_step: unknown model '{model_name}'. "
            f"Valid: {MODEL_NAMES}."
        )

    if model_name == "HAR":
        H = _forecast_har(s_window, N, params)
    elif model_name == "HARQ":
        H = _forecast_harq(s_window, pi_window, N, params)
    elif model_name == "HAR-DRD":
        H = _forecast_har_drd(S_window, N, params)
    elif model_name == "HARQ-DRD":
        H = _forecast_harq_drd(S_window, pi_window, N, params)
    elif model_name == "EWMA":
        H = _forecast_ewma(s_window, N, params)
    elif model_name == "EWMAQ":
        H = _forecast_ewmaq(s_window, pi_window, N, params)
    elif model_name == "HEAVY":
        H = _forecast_heavy(s_window, N, params)
    elif model_name == "HEAVYQ":
        H = _forecast_heavyq(s_window, pi_window, N, params)
    else:
        raise ValueError(f"forecast_one_step: unhandled model '{model_name}'.")

    H = (H + H.T) / 2.0
    if not is_pd(H):
        H = nearest_pd(H)
    return H


# ---------------------------------------------------------------------------
# HAR
# ---------------------------------------------------------------------------

def _build_har_features(s_window: np.ndarray) -> tuple:
    """Build HAR design matrix and response matrix.

    Returns (X, Y) where X has shape (n_obs, 4) = [1, s_D, s_W, s_M] and
    Y has shape (n_obs, K).  Observations start at index 22 so that the
    full 22-day lookback is available.
    """
    W, K = s_window.shape
    n_obs = W - 22
    if n_obs < 1:
        raise ValueError("_build_har_features: window too short (< 23 days).")

    X = np.ones((n_obs, 4), dtype=float)
    Y = s_window[22:W]            # response: s_t for t=22..W-1

    for row in range(n_obs):
        i = row + 22              # index in s_window
        X[row, 1] = s_window[i - 1, :].mean()          # scalar lag-1 (pooled)
        X[row, 2] = s_window[max(0, i - 5) : i, :].mean()   # 5-day average
        X[row, 3] = s_window[i - 22 : i, :].mean()          # 22-day average

    return X, Y


def _estimate_har(s_window: np.ndarray) -> dict:
    X, Y = _build_har_features(s_window)
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (4, K)
    return {"beta": beta}


def _forecast_har(s_window: np.ndarray, N: int, params: dict) -> np.ndarray:
    beta = params["beta"]   # (4, K)
    x_new = np.array([
        1.0,
        s_window[-1, :].mean(),
        s_window[-5:, :].mean(),
        s_window[-22:, :].mean(),
    ])
    h_vech = x_new @ beta   # (K,)
    H = ivech(h_vech, N)
    return H


# ---------------------------------------------------------------------------
# HARQ
# ---------------------------------------------------------------------------

def _build_harq_features(s_window: np.ndarray, pi_window: np.ndarray) -> tuple:
    """Build HARQ design matrix: HAR features + pi_{t-1} * s_{t-1} interaction."""
    W, K = s_window.shape
    n_obs = W - 22
    if n_obs < 1:
        raise ValueError("_build_harq_features: window too short (< 23 days).")

    X = np.ones((n_obs, 5), dtype=float)
    Y = s_window[22:W]

    for row in range(n_obs):
        i = row + 22
        X[row, 1] = s_window[i - 1, :].mean()
        X[row, 2] = s_window[max(0, i - 5) : i, :].mean()
        X[row, 3] = s_window[i - 22 : i, :].mean()
        # Pooled scalar HARQ term: mean(pi_{t-1}) * mean(s_{t-1})
        X[row, 4] = pi_window[i - 1, :].mean() * s_window[i - 1, :].mean()

    return X, Y


def _estimate_harq(s_window: np.ndarray, pi_window: np.ndarray) -> dict:
    X, Y = _build_harq_features(s_window, pi_window)
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (5, K)
    return {"beta": beta}


def _forecast_harq(
    s_window: np.ndarray, pi_window: np.ndarray, N: int, params: dict
) -> np.ndarray:
    beta = params["beta"]   # (5, K)
    pi_last = pi_window[-1, :].mean()
    s_last = s_window[-1, :].mean()
    x_new = np.array([
        1.0,
        s_last,
        s_window[-5:, :].mean(),
        s_window[-22:, :].mean(),
        pi_last * s_last,
    ])
    h_vech = x_new @ beta
    H = ivech(h_vech, N)
    return H


# ---------------------------------------------------------------------------
# HAR-DRD
# ---------------------------------------------------------------------------
# Uses HARDRDModel / HARQDRDModel from drd_models.py (Context7-confirmed design).
# Decompose each window into d_sq (variance) and r (off-diagonal correlation) series,
# fit per-asset HAR/HARQ for variances and a scalar HARModel for correlations,
# cache the last 22 decomposed observations for cheap one-step-ahead forecasts.

_drd_decomp = DRDDecomposition()   # reusable singleton


def _estimate_har_drd(S_window: list, N: int) -> dict:
    """Fit HAR-DRD on a rolling window of (N, N) realized covariance matrices.

    Parameters
    ----------
    S_window : list of np.ndarray
        Window of T covariance matrices, each shape (N, N).
    N : int
        Number of assets.

    Returns
    -------
    dict
        ``var_params``, ``cor_params``, ``d_sq_last22``, ``r_last22``.
    """
    d_sq_series, r_series = _decompose_series(S_window, _drd_decomp)   # (T, N), (T, N_c)

    var_params = [_fit_univariate_har_drd(d_sq_series[:, n]) for n in range(N)]

    har = HARModel()
    cor_params = har.fit(r_series)

    return {
        "var_params": var_params,
        "cor_params": cor_params,
        "d_sq_last22": d_sq_series[-22:].copy(),   # (22, N)
        "r_last22":    r_series[-22:].copy(),       # (22, N_offdiag)
    }


def _forecast_har_drd(S_window: list, N: int, params: dict) -> np.ndarray:
    """One-step-ahead HAR-DRD forecast using cached decomposed lags.

    Parameters
    ----------
    S_window : list of np.ndarray
        Unused; kept for API symmetry.
    N : int
        Number of assets.
    params : dict
        Output of :func:`_estimate_har_drd`.

    Returns
    -------
    np.ndarray
        Forecast covariance matrix, shape (N, N).
    """
    d22 = params["d_sq_last22"]   # (22, N)
    r22 = params["r_last22"]      # (22, N_offdiag)

    d_last  = d22[-1]
    r_last  = r22[-1]
    d_avg5  = d22[-5:].mean(axis=0)
    r_avg5  = r22[-5:].mean(axis=0)
    d_avg22 = d22.mean(axis=0)
    r_avg22 = r22.mean(axis=0)

    # Forecast per-asset variances
    d_sq_hat = np.array([
        params["var_params"][n]["alpha_0"]
        + params["var_params"][n]["alpha_1"] * d_last[n]
        + params["var_params"][n]["alpha_2"] * d_avg5[n]
        + params["var_params"][n]["alpha_3"] * d_avg22[n]
        for n in range(N)
    ])
    d_hat = np.sqrt(np.maximum(d_sq_hat, 0.0))

    # Forecast off-diagonal correlations (scalar HAR)
    cp = params["cor_params"]
    r_hat = (
        cp["theta_0"]
        + cp["theta_1"] * r_last
        + cp["theta_2"] * r_avg5
        + cp["theta_3"] * r_avg22
    )
    r_hat = np.clip(r_hat, -1.0 + 1e-8, 1.0 - 1e-8)

    R_hat = _build_corr_matrix(r_hat, N)
    R_hat = _ensure_valid_corr(R_hat)
    return _drd_decomp.recompose(d_hat, R_hat)


# ---------------------------------------------------------------------------
# HARQ-DRD
# ---------------------------------------------------------------------------

def _estimate_harq_drd(S_window: list, pi_window: np.ndarray, N: int) -> dict:
    """Fit HARQ-DRD on a rolling window.

    Variance equations use univariate HARQ (daily lag attenuated by pi_{n,t-1});
    correlation equation uses a standard scalar HAR with no Q term.

    Parameters
    ----------
    S_window : list of np.ndarray
        Window of T covariance matrices.
    pi_window : np.ndarray
        Demeaned uncertainty vectors, shape (T, K).
    N : int
        Number of assets.

    Returns
    -------
    dict
        ``var_params``, ``cor_params``, ``d_sq_last22``, ``r_last22``, ``pi_last``.

    Raises
    ------
    ValueError
        If window length < 23.
    """
    T = len(S_window)
    if T < 23:
        raise ValueError(f"_estimate_harq_drd: window too short (T={T}; need ≥ 23).")

    pi_window = np.asarray(pi_window, dtype=float)
    d_sq_series, r_series = _decompose_series(S_window, _drd_decomp)   # (T, N), (T, N_c)

    diag_idx = vech_diag_indices(N)   # vech index for asset n's variance uncertainty

    var_params = [
        _fit_univariate_harq_drd(d_sq_series[:, n], pi_window[:, diag_idx[n]])
        for n in range(N)
    ]

    har = HARModel()
    cor_params = har.fit(r_series)

    return {
        "var_params": var_params,
        "cor_params": cor_params,
        "d_sq_last22": d_sq_series[-22:].copy(),
        "r_last22":    r_series[-22:].copy(),
        "pi_last":     pi_window[-1].copy(),     # demeaned pi_{t-1}
    }


def _forecast_harq_drd(
    S_window: list,
    pi_window: np.ndarray,
    N: int,
    params: dict,
) -> np.ndarray:
    """One-step-ahead HARQ-DRD forecast using cached decomposed lags.

    Parameters
    ----------
    S_window : list of np.ndarray
        Unused; kept for API symmetry.
    pi_window : np.ndarray
        Unused; kept for API symmetry (pi_last already cached in params).
    N : int
        Number of assets.
    params : dict
        Output of :func:`_estimate_harq_drd`.

    Returns
    -------
    np.ndarray
        Forecast covariance matrix, shape (N, N).
    """
    d22     = params["d_sq_last22"]   # (22, N)
    r22     = params["r_last22"]      # (22, N_offdiag)
    pi_last = params["pi_last"]       # (K,) — demeaned pi_{t-1}
    diag_idx = vech_diag_indices(N)

    d_last  = d22[-1]
    r_last  = r22[-1]
    d_avg5  = d22[-5:].mean(axis=0)
    r_avg5  = r22[-5:].mean(axis=0)
    d_avg22 = d22.mean(axis=0)
    r_avg22 = r22.mean(axis=0)

    # Per-asset HARQ variance forecasts: attenuate daily lag by pi_{n,t-1}
    d_sq_hat = np.zeros(N, dtype=float)
    for n in range(N):
        p        = params["var_params"][n]
        pi_n     = float(pi_last[diag_idx[n]])
        d_sq_hat[n] = (
            p["alpha_0"]
            + (p["alpha_1"] + p["alpha_1Q"] * pi_n) * float(d_last[n])
            + p["alpha_2"] * float(d_avg5[n])
            + p["alpha_3"] * float(d_avg22[n])
        )
    d_hat = np.sqrt(np.maximum(d_sq_hat, 0.0))

    # Scalar HAR forecast for correlations (no Q term)
    cp = params["cor_params"]
    r_hat = (
        cp["theta_0"]
        + cp["theta_1"] * r_last
        + cp["theta_2"] * r_avg5
        + cp["theta_3"] * r_avg22
    )
    r_hat = np.clip(r_hat, -1.0 + 1e-8, 1.0 - 1e-8)

    R_hat = _build_corr_matrix(r_hat, N)
    R_hat = _ensure_valid_corr(R_hat)
    return _drd_decomp.recompose(d_hat, R_hat)


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------

def _ewma_filter(s_window: np.ndarray, lam: float) -> np.ndarray:
    """Compute EWMA of s_window with decay rate lambda.

    h_t = lambda * h_{t-1} + (1-lambda) * s_{t-1}
    Initialized at s_window[0].  Returns h_{t} for t = 0, …, W-1 where
    h_0 is the EWMA after seeing s_window[0].
    """
    W, K = s_window.shape
    h = s_window[0].copy()     # h_0 = S_1
    h_series = np.zeros((W, K), dtype=float)
    h_series[0] = h
    for t in range(1, W):
        h = lam * h + (1.0 - lam) * s_window[t - 1]
        h_series[t] = h
    return h_series


def _ewma_qlike_obj(log_lam: float, s_window: np.ndarray) -> float:
    """QLIKE objective for EWMA parameter (operates on scalar vech elements)."""
    lam = np.clip(np.exp(log_lam), 0.01, 0.9999)
    W, K = s_window.shape
    h = s_window[0].copy()
    loss = 0.0
    count = 0
    for t in range(1, W):
        h = lam * h + (1.0 - lam) * s_window[t - 1]
        # Element-wise QLIKE approximation: sum_k [log(h_k) + s_k/h_k]
        h_pos = np.maximum(h, 1e-14)
        loss += np.sum(np.log(h_pos) + s_window[t] / h_pos)
        count += 1
    return loss / max(count, 1)


def _estimate_ewma(s_window: np.ndarray) -> dict:
    # Grid search for initial lambda, then refine
    grid = np.linspace(0.80, 0.99, 20)
    best_loss = np.inf
    best_lam = 0.94
    for lam_val in grid:
        try:
            loss_val = _ewma_qlike_obj(np.log(lam_val), s_window)
            if loss_val < best_loss:
                best_loss = loss_val
                best_lam = lam_val
        except Exception:
            pass

    # Refine with scalar optimization
    try:
        res = optimize.minimize_scalar(
            lambda log_lam: _ewma_qlike_obj(log_lam, s_window),
            bounds=(np.log(0.80), np.log(0.9999)),
            method="bounded",
        )
        best_lam = float(np.clip(np.exp(res.x), 0.80, 0.9999))
    except Exception:
        pass

    # Store the final EWMA state h_{W-1} (used to initialize forecast)
    h_series = _ewma_filter(s_window, best_lam)
    return {"lambda": best_lam, "h_last": h_series[-1]}


def _forecast_ewma(s_window: np.ndarray, N: int, params: dict) -> np.ndarray:
    lam = params["lambda"]
    h_last = params["h_last"]   # (K,)
    # One more EWMA step: h_t = lambda * h_{W-1} + (1-lambda) * s_{W-1}
    h_next = lam * h_last + (1.0 - lam) * s_window[-1]
    H = ivech(h_next, N)
    return H


# ---------------------------------------------------------------------------
# EWMAQ
# ---------------------------------------------------------------------------

def _ewmaq_qlike_obj(
    params_raw: np.ndarray, s_window: np.ndarray, pi_window: np.ndarray
) -> float:
    """QLIKE objective for EWMAQ: (log_lambda, log_beta_Q)."""
    lam = float(np.clip(np.exp(params_raw[0]), 0.01, 0.9999))
    beta_Q = float(np.exp(params_raw[1]))   # positive

    W, K = s_window.shape
    h = s_window[0].copy()
    loss = 0.0
    count = 0
    for t in range(1, W):
        pi_prev = pi_window[t - 1]
        # Effective smoothing weight reduced when uncertainty is high
        alpha_t = (1.0 - lam) / (1.0 + beta_Q * np.abs(pi_prev).mean())
        alpha_t = np.clip(alpha_t, 0.0, 1.0 - lam)
        h = lam * h + alpha_t * s_window[t - 1]
        h_pos = np.maximum(h, 1e-14)
        loss += np.sum(np.log(h_pos) + s_window[t] / h_pos)
        count += 1
    return loss / max(count, 1)


def _estimate_ewmaq(s_window: np.ndarray, pi_window: np.ndarray) -> dict:
    # Initial guess from EWMA
    ewma_params = _estimate_ewma(s_window)
    lam0 = ewma_params["lambda"]

    try:
        res = optimize.minimize(
            _ewmaq_qlike_obj,
            x0=np.array([np.log(lam0), np.log(0.01)]),
            args=(s_window, pi_window),
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
        )
        lam_opt = float(np.clip(np.exp(res.x[0]), 0.01, 0.9999))
        beta_Q_opt = float(max(np.exp(res.x[1]), 0.0))
    except Exception:
        lam_opt = lam0
        beta_Q_opt = 0.01

    # Compute h_last with optimized params
    W, K = s_window.shape
    h = s_window[0].copy()
    for t in range(1, W):
        pi_prev = pi_window[t - 1]
        alpha_t = (1.0 - lam_opt) / (1.0 + beta_Q_opt * np.abs(pi_prev).mean())
        alpha_t = np.clip(alpha_t, 0.0, 1.0 - lam_opt)
        h = lam_opt * h + alpha_t * s_window[t - 1]

    return {"lambda": lam_opt, "beta_Q": beta_Q_opt, "h_last": h}


def _forecast_ewmaq(
    s_window: np.ndarray, pi_window: np.ndarray, N: int, params: dict
) -> np.ndarray:
    lam = params["lambda"]
    beta_Q = params["beta_Q"]
    h_last = params["h_last"]
    pi_last = pi_window[-1]
    alpha_t = (1.0 - lam) / (1.0 + beta_Q * np.abs(pi_last).mean())
    alpha_t = float(np.clip(alpha_t, 0.0, 1.0 - lam))
    h_next = lam * h_last + alpha_t * s_window[-1]
    H = ivech(h_next, N)
    return H


# ---------------------------------------------------------------------------
# HEAVY
# ---------------------------------------------------------------------------

def _heavy_filter(
    s_window: np.ndarray, a: float, b: float
) -> np.ndarray:
    """HEAVY filter: h_t = c_k + a * s_{t-1,k} + b * h_{t-1,k}.

    The intercept is set so the unconditional mean of h equals mean(s):
    c_k = (1 - a - b) * mean_s_k.

    Returns h_series of shape (W, K).
    """
    W, K = s_window.shape
    mean_s = s_window.mean(axis=0)   # (K,)
    c = (1.0 - a - b) * mean_s

    h = mean_s.copy()    # initialization
    h_series = np.zeros((W, K), dtype=float)
    h_series[0] = h
    for t in range(1, W):
        h = c + a * s_window[t - 1] + b * h
        h_series[t] = h
    return h_series


def _heavy_qlike_obj(ab_raw: np.ndarray, s_window: np.ndarray) -> float:
    """Element-wise QLIKE objective for HEAVY(1,1)."""
    # Parameterize: a = sigmoid(ab_raw[0]) * 0.5, b = sigmoid(ab_raw[1]) * 0.95
    # to enforce a > 0, b > 0, a + b < 1
    a = float(0.5 / (1.0 + np.exp(-ab_raw[0])))      # ∈ (0, 0.5)
    b_max = 0.99 - a
    b = float(b_max / (1.0 + np.exp(-ab_raw[1])))     # ∈ (0, 0.99-a)

    W, K = s_window.shape
    h_series = _heavy_filter(s_window, a, b)
    h_pos = np.maximum(h_series[1:], 1e-14)
    s_target = s_window[1:]
    loss = np.sum(np.log(h_pos) + s_target / h_pos)
    return float(loss / max(W - 1, 1))


def _estimate_heavy(s_window: np.ndarray) -> dict:
    try:
        res = optimize.minimize(
            _heavy_qlike_obj,
            x0=np.array([0.0, 2.0]),   # a≈0.25, b≈0.88
            args=(s_window,),
            method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-4},
        )
        a_opt = float(0.5 / (1.0 + np.exp(-res.x[0])))
        b_max = 0.99 - a_opt
        b_opt = float(b_max / (1.0 + np.exp(-res.x[1])))
    except Exception:
        a_opt, b_opt = 0.10, 0.88

    h_series = _heavy_filter(s_window, a_opt, b_opt)
    return {"a": a_opt, "b": b_opt, "h_last": h_series[-1]}


def _forecast_heavy(s_window: np.ndarray, N: int, params: dict) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    h_last = params["h_last"]
    mean_s = s_window.mean(axis=0)
    c = (1.0 - a - b) * mean_s
    h_next = c + a * s_window[-1] + b * h_last
    H = ivech(h_next, N)
    return H


# ---------------------------------------------------------------------------
# HEAVYQ
# ---------------------------------------------------------------------------

def _heavyq_filter(
    s_window: np.ndarray, pi_window: np.ndarray, a: float, b: float, beta_Q: float
) -> np.ndarray:
    """HEAVYQ filter with uncertainty-adjusted forcing variable."""
    W, K = s_window.shape
    mean_s = s_window.mean(axis=0)
    c = (1.0 - a - b) * mean_s

    h = mean_s.copy()
    h_series = np.zeros((W, K), dtype=float)
    h_series[0] = h
    for t in range(1, W):
        pi_prev_mean = float(np.abs(pi_window[t - 1]).mean())
        adj = 1.0 + beta_Q * pi_prev_mean
        h = c + a * adj * s_window[t - 1] + b * h
        h_series[t] = h
    return h_series


def _heavyq_qlike_obj(
    params_raw: np.ndarray, s_window: np.ndarray, pi_window: np.ndarray
) -> float:
    a = float(0.5 / (1.0 + np.exp(-params_raw[0])))
    b_max = 0.99 - a
    b = float(b_max / (1.0 + np.exp(-params_raw[1])))
    beta_Q = float(np.exp(params_raw[2]))   # positive

    W, K = s_window.shape
    h_series = _heavyq_filter(s_window, pi_window, a, b, beta_Q)
    h_pos = np.maximum(h_series[1:], 1e-14)
    s_target = s_window[1:]
    loss = np.sum(np.log(h_pos) + s_target / h_pos)
    return float(loss / max(W - 1, 1))


def _estimate_heavyq(s_window: np.ndarray, pi_window: np.ndarray) -> dict:
    heavy_params = _estimate_heavy(s_window)
    a0 = heavy_params["a"]
    b0 = heavy_params["b"]

    # Transform to unconstrained parameterization for initialization
    def _to_raw(a_val: float, b_val: float) -> np.ndarray:
        x0 = np.log(a_val / (0.5 - a_val))        # inverse sigmoid for a
        b_max = 0.99 - a_val
        x1 = np.log(b_val / (b_max - b_val))      # inverse sigmoid for b
        return np.array([x0, x1, np.log(0.01)])

    try:
        x0 = _to_raw(a0, b0)
        res = optimize.minimize(
            _heavyq_qlike_obj,
            x0=x0,
            args=(s_window, pi_window),
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4},
        )
        a_opt = float(0.5 / (1.0 + np.exp(-res.x[0])))
        b_max = 0.99 - a_opt
        b_opt = float(b_max / (1.0 + np.exp(-res.x[1])))
        beta_Q_opt = float(max(np.exp(res.x[2]), 0.0))
    except Exception:
        a_opt, b_opt, beta_Q_opt = a0, b0, 0.01

    h_series = _heavyq_filter(s_window, pi_window, a_opt, b_opt, beta_Q_opt)
    return {"a": a_opt, "b": b_opt, "beta_Q": beta_Q_opt, "h_last": h_series[-1]}


def _forecast_heavyq(
    s_window: np.ndarray, pi_window: np.ndarray, N: int, params: dict
) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    beta_Q = params["beta_Q"]
    h_last = params["h_last"]
    mean_s = s_window.mean(axis=0)
    c = (1.0 - a - b) * mean_s
    pi_last_mean = float(np.abs(pi_window[-1]).mean())
    adj = 1.0 + beta_Q * pi_last_mean
    h_next = c + a * adj * s_window[-1] + b * h_last
    H = ivech(h_next, N)
    return H
