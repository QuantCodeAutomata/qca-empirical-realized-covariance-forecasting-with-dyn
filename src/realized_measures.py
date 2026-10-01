"""Realized covariance measures for high-frequency return data.

Implements the multivariate realized kernel estimator of Barndorff-Nielsen
et al. (2011) together with a pre-averaging integrated quarticity proxy
used to construct the time-varying uncertainty measure Π_t.
"""

import logging
from typing import Optional

import numpy as np

from src.utils import nearest_pd, is_pd, parzen_kernel, vech, vech_size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def parzen_kernel_cov(h: int, H: int) -> float:
    """Return the Parzen kernel weight k(h/H) for lag *h* and bandwidth *H*.

    Thin wrapper around :func:`src.utils.parzen_kernel` with integer-lag
    interface used throughout the realized kernel computation.

    Parameters
    ----------
    h : int
        Lag index (0 ≤ h ≤ H).
    H : int
        Bandwidth parameter (positive integer).

    Returns
    -------
    float
        Parzen weight k(|h|/H) ∈ [0, 1].

    Raises
    ------
    ValueError
        If *H* ≤ 0.
    """
    if H <= 0:
        raise ValueError(f"parzen_kernel_cov: H must be positive; got {H}.")
    return parzen_kernel(abs(h) / H)


# ---------------------------------------------------------------------------
# Autocovariance
# ---------------------------------------------------------------------------

def compute_autocovariance(returns: np.ndarray, lag: int) -> np.ndarray:
    """Compute sample autocovariance matrix Γ_h for a given lag.

    Γ_h = Σ_{i=lag+1}^{M} r_i · r_{i-lag}′

    where *r_i* is the *i*-th intraday return vector (a column of length N).
    The summation convention uses 1-based indexing; in 0-based terms the sum
    runs over *i* = lag, lag+1, …, M-1.

    Note: unlike the sample covariance, no mean-centering or normalization by
    M is applied here.  The realized kernel handles this via the kernel weights.

    Parameters
    ----------
    returns : np.ndarray
        Intraday return matrix of shape (M, N) where M is the number of
        intraday intervals and N is the number of assets.
    lag : int
        Lag h ≥ 0.

    Returns
    -------
    np.ndarray
        Autocovariance matrix of shape (N, N).

    Raises
    ------
    ValueError
        If *returns* is not 2-D, or if *lag* is negative, or if *lag* ≥ M.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError(
            f"compute_autocovariance: expected 2-D array; got shape {returns.shape}."
        )
    M, N = returns.shape
    if lag < 0:
        raise ValueError(f"compute_autocovariance: lag must be ≥ 0; got {lag}.")
    if lag >= M:
        raise ValueError(
            f"compute_autocovariance: lag ({lag}) must be < M ({M})."
        )

    # Vectorised outer-product sum: returns[lag:, :].T @ returns[:M-lag, :]
    lagged = returns[:M - lag, :]     # shape (M-lag, N) — the r_{i-lag} terms
    current = returns[lag:, :]        # shape (M-lag, N) — the r_i terms
    return current.T @ lagged         # (N, N)


# ---------------------------------------------------------------------------
# Realized kernel
# ---------------------------------------------------------------------------

def compute_realized_kernel(
    intraday_returns: np.ndarray, H: Optional[int] = None
) -> np.ndarray:
    """Estimate the daily realized covariance matrix via multivariate realized kernel.

    Uses the Parzen kernel with bandwidth *H*::

        MK_t = Σ_{h=-H}^{H} k(h/H) · Γ_h

    where Γ_{-h} = Γ_h′ (by symmetry of the covariance matrix).  Equivalently::

        MK_t = Γ_0 + Σ_{h=1}^{H} k(h/H) · (Γ_h + Γ_h′)

    because k(0/H) = 1 for all valid Parzen kernels.

    The result is symmetrized and, if not positive definite, projected to the
    nearest PD matrix via eigenvalue clipping.

    Parameters
    ----------
    intraday_returns : np.ndarray
        Intraday return matrix of shape (M, N) where M is the number of
        intraday intervals and N is the number of assets.
    H : int, optional
        Bandwidth parameter.  If ``None`` the data-driven default
        H = int(M^(2/3)) is used (Barndorff-Nielsen et al., 2011).

    Returns
    -------
    np.ndarray
        Realized covariance estimate of shape (N, N), symmetric and PD.

    Raises
    ------
    ValueError
        If *intraday_returns* is not 2-D or has fewer than 2 rows.
    """
    intraday_returns = np.asarray(intraday_returns, dtype=float)
    if intraday_returns.ndim != 2:
        raise ValueError(
            "compute_realized_kernel: intraday_returns must be 2-D; "
            f"got shape {intraday_returns.shape}."
        )
    M, N = intraday_returns.shape
    if M < 2:
        raise ValueError(
            f"compute_realized_kernel: need at least 2 intraday observations; got M={M}."
        )

    if H is None:
        H = max(1, int(M ** (2.0 / 3.0)))
    H = min(H, M - 1)  # bandwidth cannot exceed M-1

    # Γ_0 term (kernel weight = 1)
    MK = compute_autocovariance(intraday_returns, lag=0)

    for h in range(1, H + 1):
        w = parzen_kernel_cov(h, H)
        if w == 0.0:
            continue
        Gamma_h = compute_autocovariance(intraday_returns, lag=h)
        MK += w * (Gamma_h + Gamma_h.T)

    # Enforce symmetry numerically
    MK = (MK + MK.T) / 2.0

    if not is_pd(MK):
        logger.warning(
            "compute_realized_kernel: realized kernel matrix is not PD; "
            "projecting to nearest PD."
        )
        MK = nearest_pd(MK)

    return MK


# ---------------------------------------------------------------------------
# Pre-averaging integrated quarticity
# ---------------------------------------------------------------------------

def compute_preaveraging_quarticity(
    intraday_returns: np.ndarray, H: Optional[int] = None
) -> np.ndarray:
    """Estimate integrated quarticity IQ̂_t using a pre-averaging proxy.

    The estimator approximates the pre-averaged quarticity measure::

        IQ̂_t = M · Σ_{i=1}^{M-H} x_i x_i′
                − (M/2) · Σ_{i=1}^{M-H-1} (x_i x_{i+1}′ + x_{i+1} x_i′)

    where x_i = vech(r_i r_i′) is the vectorized outer product of the
    *i*-th intraday return vector.  The double-sum bias-correction term
    removes the leading-order noise bias analogous to the pre-averaging
    correction in Jacod et al. (2009).

    Uses the same default bandwidth H = int(M^(2/3)) as
    :func:`compute_realized_kernel`.

    Parameters
    ----------
    intraday_returns : np.ndarray
        Intraday return matrix of shape (M, N).
    H : int, optional
        Bandwidth controlling the number of terms included in the sum.
        Defaults to int(M^(2/3)).

    Returns
    -------
    np.ndarray
        Integrated quarticity proxy of shape (K, K) where K = N*(N+1)//2.
        Guaranteed symmetric; projected to nearest PD if needed.

    Raises
    ------
    ValueError
        If *intraday_returns* is not 2-D or M < 3.
    """
    intraday_returns = np.asarray(intraday_returns, dtype=float)
    if intraday_returns.ndim != 2:
        raise ValueError(
            "compute_preaveraging_quarticity: intraday_returns must be 2-D; "
            f"got shape {intraday_returns.shape}."
        )
    M, N = intraday_returns.shape
    if M < 3:
        raise ValueError(
            "compute_preaveraging_quarticity: need at least 3 intraday "
            f"observations; got M={M}."
        )

    if H is None:
        H = max(1, int(M ** (2.0 / 3.0)))
    H = min(H, M - 2)  # need at least 2 terms in the correction sum

    K = vech_size(N)

    # Build x_i = vech(r_i r_i') for i = 0 .. M-H-1  (0-based)
    n_terms = M - H  # number of x_i vectors
    X = np.zeros((n_terms, K), dtype=float)
    for i in range(n_terms):
        ri = intraday_returns[i]
        X[i] = vech(np.outer(ri, ri))

    # First sum: M * Σ x_i x_i'
    IQ = M * (X.T @ X)

    # Bias-correction: -(M/2) * Σ (x_i x_{i+1}' + x_{i+1} x_i')
    # = -(M/2) * (X[:-1].T @ X[1:] + X[1:].T @ X[:-1])
    cross = X[:-1].T @ X[1:]  # (K, K)
    IQ -= (M / 2.0) * (cross + cross.T)

    # Enforce symmetry
    IQ = (IQ + IQ.T) / 2.0

    if not is_pd(IQ):
        logger.debug(
            "compute_preaveraging_quarticity: IQ matrix is not PD; "
            "projecting to nearest PD."
        )
        IQ = nearest_pd(IQ)

    return IQ


# ---------------------------------------------------------------------------
# Π_t computation
# ---------------------------------------------------------------------------

def compute_pi(
    IQ_hat: np.ndarray, constant: float = 3.777
) -> tuple:
    """Compute the uncertainty measure Π_t and its diagonal square root π_t.

    Π_t = constant · IQ̂_t

    π_t = sqrt(diag(Π_t))

    The constant 3.777 corresponds to the scaling used in the paper for the
    Parzen kernel (the constant depends on the kernel; for Parzen it equals
    ∫₀¹ k(x)² dx / (∫₀¹ k′(x)² dx)^{1/2}, approximately 3.777).

    Parameters
    ----------
    IQ_hat : np.ndarray
        Integrated quarticity estimate of shape (K, K).
    constant : float, optional
        Scaling constant (default: 3.777).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Π_t, π_t) where Π_t has shape (K, K) and π_t has shape (K,).

    Raises
    ------
    ValueError
        If *IQ_hat* is not square or *constant* is non-positive.
    """
    IQ_hat = np.asarray(IQ_hat, dtype=float)
    if IQ_hat.ndim != 2 or IQ_hat.shape[0] != IQ_hat.shape[1]:
        raise ValueError(
            "compute_pi: IQ_hat must be a square 2-D array; "
            f"got shape {IQ_hat.shape}."
        )
    if constant <= 0:
        raise ValueError(f"compute_pi: constant must be positive; got {constant}.")

    Pi_t = constant * IQ_hat
    diag_vals = np.diag(Pi_t)
    # Guard against numerical negatives on the diagonal before sqrt
    pi_t = np.sqrt(np.maximum(diag_vals, 0.0))
    return Pi_t, pi_t


# ---------------------------------------------------------------------------
# π_t demeaning
# ---------------------------------------------------------------------------

def demean_pi(pi_series: np.ndarray) -> np.ndarray:
    """Demean π_t within the estimation window by subtracting the time mean.

    For each of the K elements, subtracts the sample mean computed over all
    T observations in *pi_series*.  This recentres the uncertainty proxy
    before it enters regression or forecasting models.

    Parameters
    ----------
    pi_series : np.ndarray
        Array of shape (T, K) containing the daily π_t row vectors.

    Returns
    -------
    np.ndarray
        Demeaned array of shape (T, K) with zero mean along axis 0.

    Raises
    ------
    ValueError
        If *pi_series* is not 2-D or has zero rows.
    """
    pi_series = np.asarray(pi_series, dtype=float)
    if pi_series.ndim != 2:
        raise ValueError(
            f"demean_pi: expected 2-D array; got shape {pi_series.shape}."
        )
    T, K = pi_series.shape
    if T == 0:
        raise ValueError("demean_pi: pi_series has zero rows.")

    mean_pi = pi_series.mean(axis=0)  # shape (K,)
    return pi_series - mean_pi


# ---------------------------------------------------------------------------
# Daily realized covariance series
# ---------------------------------------------------------------------------

def compute_daily_rc_series(
    intraday_returns_list: list, H: Optional[int] = None
) -> tuple:
    """Compute realized covariance and quarticity measures for a panel of days.

    Iterates over a list of daily intraday return matrices and computes:

    1. **S_t** – Realized covariance matrix (N × N) via
       :func:`compute_realized_kernel`.
    2. **IQ̂_t** – Integrated quarticity proxy (K × K) via
       :func:`compute_preaveraging_quarticity`.
    3. **Π_t** – Uncertainty matrix (K × K) = 3.777 · IQ̂_t.
    4. **π_t** – Diagonal uncertainty vector (K,) = sqrt(diag(Π_t)).

    where K = N*(N+1)//2.

    Parameters
    ----------
    intraday_returns_list : list of np.ndarray
        List of length T, each element a (M_t, N) intraday return matrix for
        one trading day.  M_t may vary across days; N must be constant.
    H : int, optional
        Bandwidth for the realized kernel and quarticity estimators.  If
        ``None``, each day uses its own default H = int(M_t^(2/3)).

    Returns
    -------
    tuple
        (S_list, IQ_list, Pi_list, pi_list) where each element is a Python
        list of length T containing the per-day estimates.

    Raises
    ------
    ValueError
        If *intraday_returns_list* is empty or any element is not a valid 2-D
        array.
    """
    if len(intraday_returns_list) == 0:
        raise ValueError("compute_daily_rc_series: intraday_returns_list is empty.")

    S_list: list = []
    IQ_list: list = []
    Pi_list: list = []
    pi_list: list = []

    for t, ret_t in enumerate(intraday_returns_list):
        ret_t = np.asarray(ret_t, dtype=float)
        if ret_t.ndim != 2:
            raise ValueError(
                f"compute_daily_rc_series: element {t} is not a 2-D array; "
                f"got shape {ret_t.shape}."
            )

        S_t = compute_realized_kernel(ret_t, H=H)
        IQ_t = compute_preaveraging_quarticity(ret_t, H=H)
        Pi_t, pi_t = compute_pi(IQ_t)

        S_list.append(S_t)
        IQ_list.append(IQ_t)
        Pi_list.append(Pi_t)
        pi_list.append(pi_t)

    return S_list, IQ_list, Pi_list, pi_list
