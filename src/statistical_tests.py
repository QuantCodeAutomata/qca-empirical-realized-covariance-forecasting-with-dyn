"""Statistical tests for forecast comparison.

Implements the Diebold–Mariano (1995) test for equal predictive ability,
the Model Confidence Set (MCS) procedure of Hansen, Lunde & Nason (2011),
and supporting bootstrap utilities.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stationary bootstrap
# ---------------------------------------------------------------------------

def stationary_bootstrap_mean(
    data: np.ndarray,
    B: int,
    block_size: float,
    seed: int = 42,
) -> np.ndarray:
    """Stationary bootstrap for the mean of a 1-D or 2-D time series.

    Each bootstrap sample of length T is formed by drawing random blocks
    whose lengths are i.i.d. Geometric(1/block_size), with wrap-around
    (circular) indexing so the sample is stationary (Politis & Romano, 1994).

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (T,) or (T, K).  Rows are time periods.
    B : int
        Number of bootstrap replications (must be ≥ 1).
    block_size : float
        Expected block length; controls the Geometric distribution parameter
        p = 1 / block_size.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Bootstrap means of shape (B,) if *data* is 1-D, or (B, K) if 2-D.

    Raises
    ------
    ValueError
        If *data* is not 1-D or 2-D, *B* < 1, or *block_size* < 1.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim not in (1, 2):
        raise ValueError(
            f"stationary_bootstrap_mean: data must be 1-D or 2-D; got {data.ndim}-D."
        )
    if B < 1:
        raise ValueError(f"stationary_bootstrap_mean: B must be ≥ 1; got {B}.")
    if block_size < 1:
        raise ValueError(
            f"stationary_bootstrap_mean: block_size must be ≥ 1; got {block_size}."
        )

    rng = np.random.default_rng(seed)
    T = data.shape[0]
    p_geom = 1.0 / block_size  # Geometric success probability

    is_1d = data.ndim == 1
    arr = data if not is_1d else data[:, np.newaxis]
    K = arr.shape[1]

    boot_means = np.empty((B, K), dtype=float)

    for b in range(B):
        indices = np.empty(T, dtype=int)
        pos = 0
        while pos < T:
            start = int(rng.integers(0, T))
            block_len = int(rng.geometric(p_geom))
            block_len = min(block_len, T - pos)
            for k in range(block_len):
                indices[pos + k] = (start + k) % T
            pos += block_len
        boot_sample = arr[indices]          # (T, K)
        boot_means[b] = boot_sample.mean(axis=0)

    return boot_means[:, 0] if is_1d else boot_means


# ---------------------------------------------------------------------------
# Diebold–Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int = 1,
) -> tuple:
    """Diebold–Mariano test for equal predictive ability (two-sided).

    Computes the loss differential :math:`d_t = \\text{loss1}_t - \\text{loss2}_t`
    and tests ``H₀: E[d_t] = 0`` against the two-sided alternative using a
    Newey–West HAC-corrected t-statistic with *h* lags.

    .. math::

        DM = \\frac{\\bar{d}}{\\sqrt{\\widehat{\\mathrm{Var}}_{\\text{HAC}}(\\bar{d})}}

    Parameters
    ----------
    loss1 : np.ndarray
        Loss series for model 1, shape (T,).
    loss2 : np.ndarray
        Loss series for model 2, shape (T,).
    h : int, optional
        Number of lags for the Newey–West HAC estimator (default: 1).

    Returns
    -------
    tuple[float, float]
        *(DM statistic, two-sided p-value)*.  The p-value is based on the
        standard normal distribution (asymptotically valid for large T).

    Raises
    ------
    ValueError
        If *loss1* and *loss2* have different lengths, are not 1-D, or T < 2.
    """
    loss1 = np.asarray(loss1, dtype=float)
    loss2 = np.asarray(loss2, dtype=float)
    if loss1.ndim != 1 or loss2.ndim != 1:
        raise ValueError("diebold_mariano_test: loss1 and loss2 must be 1-D.")
    if len(loss1) != len(loss2):
        raise ValueError(
            f"diebold_mariano_test: loss1 length {len(loss1)} != "
            f"loss2 length {len(loss2)}."
        )
    T = len(loss1)
    if T < 2:
        raise ValueError(f"diebold_mariano_test: need T ≥ 2; got T={T}.")

    d = loss1 - loss2
    d_bar = d.mean()

    # Newey–West HAC variance of d_bar = Var_HAC(d) / T
    # Use statsmodels OLS with constant to extract HAC variance
    X = np.ones((T, 1), dtype=float)
    model = sm.OLS(d, X)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": h, "use_correction": True})

    # result.bse[0] is the HAC standard error of d_bar
    se_dbar = float(result.bse[0])

    if se_dbar <= 0.0:
        logger.warning(
            "diebold_mariano_test: HAC s.e. is non-positive (%.4g); returning NaN.",
            se_dbar,
        )
        return float("nan"), float("nan")

    dm_stat = d_bar / se_dbar

    # Two-sided p-value under N(0,1)
    from scipy import stats as scipy_stats
    p_value = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(dm_stat))))

    return float(dm_stat), p_value


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------

def model_confidence_set(
    losses_df: pd.DataFrame,
    alpha: float = 0.10,
    B: int = 1000,
    block_size: int = 10,
    seed: int = 42,
) -> list:
    """Model Confidence Set (MCS) at confidence level ``1 - alpha``.

    Implements the MCS procedure of Hansen, Lunde & Nason (2011) using the
    range statistic :math:`T_R` and the stationary bootstrap.

    **Algorithm**:

    1. Start with all models in :math:`\\hat{\\mathcal{M}}`.
    2. For all pairs :math:`(i, j) \\in \\hat{\\mathcal{M}}`:
       compute :math:`d_{ij,t} = L_{i,t} - L_{j,t}`.
    3. Compute
       :math:`T_R = \\max_{i,j} |\\bar{d}_{ij}| / \\sqrt{\\widehat{\\mathrm{Var}}(d_{ij})/T}`.
    4. Obtain the bootstrap critical value :math:`c_\\alpha` at level *alpha*.
    5. If :math:`T_R > c_\\alpha`: eliminate the model with the highest
       mean loss and repeat from step 2.
    6. Return the surviving model names.

    Parameters
    ----------
    losses_df : pd.DataFrame
        DataFrame of shape (T, M) where each column is a model's loss series.
    alpha : float, optional
        Significance level (default: 0.10, so the MCS has 90 % confidence).
    B : int, optional
        Bootstrap replications (default: 1000).
    block_size : int, optional
        Expected block length for the stationary bootstrap (default: 10).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    list
        List of column names (model names) surviving in the MCS.

    Raises
    ------
    ValueError
        If *losses_df* has fewer than 2 columns, fewer than 2 rows, or
        *alpha* is not in (0, 1).
    """
    if losses_df.shape[1] < 2:
        raise ValueError("model_confidence_set: need at least 2 models.")
    if losses_df.shape[0] < 2:
        raise ValueError("model_confidence_set: need at least 2 time periods.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"model_confidence_set: alpha must be in (0, 1); got {alpha}.")

    rng = np.random.default_rng(seed)
    T = losses_df.shape[0]
    surviving = list(losses_df.columns)

    def _compute_tr(models: list) -> tuple:
        """Compute T_R statistic and pairwise test results for given model set."""
        losses = losses_df[models].to_numpy(dtype=float)
        M_curr = len(models)
        best_stat = -np.inf
        best_i = best_j = -1
        t_stats = {}

        for i in range(M_curr):
            for j in range(i + 1, M_curr):
                d_ij = losses[:, i] - losses[:, j]
                d_bar = d_ij.mean()
                # HAC variance using Newey-West with block_size lags
                var_hac = _hac_variance(d_ij, lags=block_size)
                se = np.sqrt(max(var_hac / T, 1e-16))
                t_val = abs(d_bar) / se
                t_stats[(i, j)] = (t_val, d_bar)
                if t_val > best_stat:
                    best_stat = t_val
                    best_i, best_j = i, j

        return best_stat, best_i, best_j, t_stats

    def _bootstrap_critical_value(models: list) -> float:
        """Bootstrap distribution of T_R under H₀ (centred differentials)."""
        losses = losses_df[models].to_numpy(dtype=float)
        M_curr = len(models)
        T_R_boot = np.empty(B, dtype=float)
        p_geom = 1.0 / block_size

        # Pre-compute centred pairwise differentials
        diffs = []
        for i in range(M_curr):
            for j in range(i + 1, M_curr):
                d_ij = losses[:, i] - losses[:, j]
                diffs.append(d_ij - d_ij.mean())   # centre under H₀

        n_pairs = len(diffs)

        for b in range(B):
            # Single stationary bootstrap index draw for all pairs
            indices = _bootstrap_indices(T, T, p_geom, rng)
            best_t = -np.inf
            for d_ij in diffs:
                d_boot = d_ij[indices]
                d_bar_b = d_boot.mean()
                var_hac_b = _hac_variance(d_boot, lags=block_size)
                se_b = np.sqrt(max(var_hac_b / T, 1e-16))
                t_val_b = abs(d_bar_b) / se_b
                if t_val_b > best_t:
                    best_t = t_val_b
            T_R_boot[b] = best_t

        return float(np.quantile(T_R_boot, 1.0 - alpha))

    while len(surviving) > 1:
        tr_stat, bi, bj, _ = _compute_tr(surviving)
        c_alpha = _bootstrap_critical_value(surviving)

        if tr_stat <= c_alpha:
            break  # all surviving models are in the MCS

        # Eliminate model with highest mean loss
        mean_losses = losses_df[surviving].mean(axis=0)
        worst_model = mean_losses.idxmax()
        logger.debug(
            "MCS: T_R=%.4f > c_alpha=%.4f; eliminating '%s'.",
            tr_stat, c_alpha, worst_model,
        )
        surviving.remove(worst_model)

    return surviving


def _bootstrap_indices(T: int, n: int, p_geom: float, rng: np.random.Generator) -> np.ndarray:
    """Draw *n* stationary-bootstrap indices from a series of length *T*."""
    indices = np.empty(n, dtype=int)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, T))
        block_len = int(rng.geometric(p_geom))
        block_len = min(block_len, n - pos)
        for k in range(block_len):
            indices[pos + k] = (start + k) % T
        pos += block_len
    return indices


def _hac_variance(d: np.ndarray, lags: int) -> float:
    """Newey–West HAC estimator of Var(d_bar) * T (the long-run variance).

    Returns the long-run variance :math:`\\hat{\\sigma}^2` such that
    :math:`\\widehat{\\operatorname{Var}}(\\bar{d}) = \\hat{\\sigma}^2 / T`.

    Uses the Bartlett kernel: :math:`w_h = 1 - h/(L+1)` for *h* ≤ *L*.
    """
    T = len(d)
    d_c = d - d.mean()
    gamma0 = float(d_c @ d_c) / T
    lrv = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma_h = float(d_c[lag:] @ d_c[:-lag]) / T
        lrv += 2.0 * weight * gamma_h
    return max(lrv, 0.0)  # clamp at 0 to avoid numerical negatives


# ---------------------------------------------------------------------------
# DM significance markers
# ---------------------------------------------------------------------------

def dm_significance_markers(
    losses_df: pd.DataFrame,
    model_pairs: list,
) -> dict:
    """Run pairwise Diebold–Mariano tests and return significance markers.

    For each ``(query_model, baseline_model)`` pair, tests whether the query
    model's losses are significantly different from the baseline model's losses.
    Returns ``'*'`` if the two-sided p-value < 0.05, else ``''``.

    Parameters
    ----------
    losses_df : pd.DataFrame
        DataFrame with one column per model, rows are time periods.
    model_pairs : list of tuple
        List of ``(query_model, baseline_model)`` string pairs, where both
        strings must be column names in *losses_df*.

    Returns
    -------
    dict
        Keys are ``(query_model, baseline_model)`` tuples; values are
        ``'*'`` (p < 0.05) or ``''`` (p ≥ 0.05).

    Raises
    ------
    ValueError
        If any model name in *model_pairs* is not a column of *losses_df*.
    """
    markers: dict = {}
    for q_model, b_model in model_pairs:
        for name in (q_model, b_model):
            if name not in losses_df.columns:
                raise ValueError(
                    f"dm_significance_markers: '{name}' not found in losses_df columns."
                )
        loss1 = losses_df[q_model].to_numpy(dtype=float)
        loss2 = losses_df[b_model].to_numpy(dtype=float)
        _, p_val = diebold_mariano_test(loss1, loss2)
        markers[(q_model, b_model)] = "*" if (not np.isnan(p_val) and p_val < 0.05) else ""
    return markers
