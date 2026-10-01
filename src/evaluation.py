"""Forecast evaluation metrics for realized covariance forecasting.

Implements the Frobenius and QLIKE loss functions used to compare covariance
matrix forecasts H_{t|t-1} against realized proxies S_t, along with
aggregation helpers and subsample splitting by uncertainty norm.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg

from src.utils import is_pd, nearest_pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point-in-time losses
# ---------------------------------------------------------------------------

def frobenius_loss(H: np.ndarray, S: np.ndarray) -> float:
    """Frobenius loss between forecast H and realized covariance S.

    .. math::

        L_{\\text{Frob}} = \\sqrt{\\operatorname{Tr}[(H - S)(H - S)^{\\prime}]}

    Parameters
    ----------
    H : np.ndarray
        Forecast covariance matrix of shape (N, N).
    S : np.ndarray
        Realized covariance matrix of shape (N, N).

    Returns
    -------
    float
        Non-negative Frobenius loss value.

    Raises
    ------
    ValueError
        If *H* or *S* are not 2-D square arrays of the same shape.
    """
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"frobenius_loss: H must be square 2-D; got {H.shape}.")
    if S.shape != H.shape:
        raise ValueError(
            f"frobenius_loss: S shape {S.shape} does not match H shape {H.shape}."
        )
    diff = H - S
    return float(np.sqrt(np.trace(diff @ diff.T)))


def qlike_loss(H: np.ndarray, S: np.ndarray) -> float:
    """QLIKE (Gaussian quasi-likelihood) loss between forecast H and realized S.

    .. math::

        L_{\\text{QLIKE}} = \\log|H| + \\operatorname{Tr}(H^{-1} S)

    The log-determinant is computed via the Cholesky factor for numerical
    stability.  The trace term uses two triangular solves so that no explicit
    matrix inverse is formed.

    Parameters
    ----------
    H : np.ndarray
        Forecast covariance matrix of shape (N, N).  Must be positive definite;
        otherwise ``np.inf`` is returned.
    S : np.ndarray
        Realized covariance matrix of shape (N, N).

    Returns
    -------
    float
        QLIKE loss, or ``np.inf`` if *H* is not positive definite.

    Raises
    ------
    ValueError
        If *H* or *S* are not 2-D square arrays of the same shape.
    """
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"qlike_loss: H must be square 2-D; got {H.shape}.")
    if S.shape != H.shape:
        raise ValueError(
            f"qlike_loss: S shape {S.shape} does not match H shape {H.shape}."
        )

    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return np.inf

    # log|H| = 2 * sum(log(diag(L)))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    # Tr(H^{-1} S) = Tr(L^{-T} L^{-1} S)
    # Let Z = L^{-1} S  (solve L @ Z = S)
    # Then Tr(L^{-T} Z) = Tr(L^{-1} Z^T)  [cyclic: Tr(ABC) = Tr(CAB)]
    #                    = Tr(L^{-1} S^T L^{-T}) [since S symmetric, S^T=S]
    # Compute W = L^{-1} Z^T = L^{-1} S L^{-T} via second solve, then take Tr(W).
    Z = linalg.solve_triangular(L, S, lower=True)          # Z = L^{-1} S
    W = linalg.solve_triangular(L, Z.T, lower=True)        # W = L^{-1} S L^{-T}
    trace_val = float(np.trace(W))

    return float(logdet + trace_val)


# ---------------------------------------------------------------------------
# Series-level aggregation
# ---------------------------------------------------------------------------

def compute_losses(
    H_series: list,
    S_series: list,
) -> pd.DataFrame:
    """Compute Frobenius and QLIKE losses for each forecast/realisation pair.

    Parameters
    ----------
    H_series : list of np.ndarray
        Sequence of T forecast matrices, each of shape (N, N).
    S_series : list of np.ndarray
        Sequence of T realized covariance matrices, each of shape (N, N).

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (T, 2) with columns ``['frobenius', 'qlike']``.
        Integer index 0 … T-1.

    Raises
    ------
    ValueError
        If *H_series* and *S_series* have different lengths or are empty.
    """
    if len(H_series) != len(S_series):
        raise ValueError(
            f"compute_losses: H_series length {len(H_series)} != "
            f"S_series length {len(S_series)}."
        )
    if len(H_series) == 0:
        raise ValueError("compute_losses: H_series and S_series must be non-empty.")

    frob_vals: list = []
    qlike_vals: list = []

    for t, (H, S) in enumerate(zip(H_series, S_series)):
        frob_vals.append(frobenius_loss(H, S))
        qlike_vals.append(qlike_loss(H, S))

    return pd.DataFrame({"frobenius": frob_vals, "qlike": qlike_vals})


def mean_losses(losses: pd.DataFrame) -> dict:
    """Compute mean Frobenius and QLIKE losses over the evaluation period.

    Parameters
    ----------
    losses : pd.DataFrame
        DataFrame with columns ``['frobenius', 'qlike']`` as returned by
        :func:`compute_losses`.

    Returns
    -------
    dict
        ``{'frobenius': float, 'qlike': float}`` mean values.

    Raises
    ------
    ValueError
        If *losses* is empty or does not contain the required columns.
    """
    required = {"frobenius", "qlike"}
    missing = required - set(losses.columns)
    if missing:
        raise ValueError(f"mean_losses: missing columns {missing}.")
    if len(losses) == 0:
        raise ValueError("mean_losses: losses DataFrame is empty.")

    return {
        "frobenius": float(losses["frobenius"].mean()),
        "qlike": float(losses["qlike"].mean()),
    }


# ---------------------------------------------------------------------------
# Subsample splitting by uncertainty norm
# ---------------------------------------------------------------------------

def split_by_pi_norm(
    losses: pd.DataFrame,
    pi_norms: np.ndarray,
    threshold_pct: float = 95.0,
) -> tuple:
    """Split loss series into low- and high-uncertainty subsamples.

    Partitions *losses* into two sub-DataFrames based on whether each day's
    uncertainty norm ``||π_t||`` is below or above the *threshold_pct*
    percentile of *pi_norms*.

    Parameters
    ----------
    losses : pd.DataFrame
        DataFrame with columns ``['frobenius', 'qlike']`` of length T.
    pi_norms : np.ndarray
        1-D array of length T containing ``||π_t||`` for each OOS day.
    threshold_pct : float, optional
        Percentile threshold in (0, 100) used to split the sample.
        Default is 95.0 (top 5 % of uncertainty days in the "above" group).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        *(losses_below, losses_above)* — both DataFrames retain the original
        index from *losses* so that days can be traced back.

    Raises
    ------
    ValueError
        If lengths do not match, *pi_norms* is not 1-D, or *threshold_pct*
        is outside (0, 100).
    """
    pi_norms = np.asarray(pi_norms, dtype=float)
    if pi_norms.ndim != 1:
        raise ValueError(
            f"split_by_pi_norm: pi_norms must be 1-D; got shape {pi_norms.shape}."
        )
    if len(losses) != len(pi_norms):
        raise ValueError(
            f"split_by_pi_norm: losses length {len(losses)} != "
            f"pi_norms length {len(pi_norms)}."
        )
    if not (0.0 < threshold_pct < 100.0):
        raise ValueError(
            f"split_by_pi_norm: threshold_pct must be in (0, 100); got {threshold_pct}."
        )

    threshold = float(np.percentile(pi_norms, threshold_pct))
    mask_above = pi_norms >= threshold
    mask_below = ~mask_above

    losses_below = losses.loc[losses.index[mask_below]].copy()
    losses_above = losses.loc[losses.index[mask_above]].copy()

    logger.debug(
        "split_by_pi_norm: threshold=%.4g (pct=%.1f); "
        "below=%d, above=%d days.",
        threshold, threshold_pct, mask_below.sum(), mask_above.sum(),
    )

    return losses_below, losses_above
