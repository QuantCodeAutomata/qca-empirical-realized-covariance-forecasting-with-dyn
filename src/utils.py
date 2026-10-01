"""Utility functions for realized covariance forecasting.

Provides matrix helpers (vech/ivech, nearest PD, safe linear algebra),
kernel weights, and rolling statistical tools used throughout the project.
"""

import logging
import warnings
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# vech / ivech helpers
# ---------------------------------------------------------------------------

def vech(M: np.ndarray) -> np.ndarray:
    """Vectorize the lower triangle of a symmetric matrix (column-major).

    Stacks the lower-triangular elements column by column, including the
    diagonal.  For an (n, n) matrix the result has length n*(n+1)//2.

    Column-major ordering for a 3×3 matrix:
        [M[0,0], M[1,0], M[2,0], M[1,1], M[2,1], M[2,2]]

    Parameters
    ----------
    M : np.ndarray
        Square symmetric matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        1-D array of length n*(n+1)//2 containing lower-triangular elements.

    Raises
    ------
    ValueError
        If *M* is not a 2-D square array.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"vech requires a square 2-D array; got shape {M.shape}.")
    n = M.shape[0]
    # column-major: for column j take rows j..n-1
    idx_col = []
    idx_row = []
    for j in range(n):
        for i in range(j, n):
            idx_row.append(i)
            idx_col.append(j)
    return M[idx_row, idx_col]


def ivech(v: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct a symmetric matrix from a vech vector (column-major).

    Inverse of :func:`vech`.  Given a vector of length n*(n+1)//2 produced
    by column-major vech ordering, returns the corresponding (n, n) symmetric
    matrix.

    Parameters
    ----------
    v : np.ndarray
        1-D array of length n*(n+1)//2.
    n : int
        Dimension of the target square matrix.

    Returns
    -------
    np.ndarray
        Symmetric (n, n) array reconstructed from *v*.

    Raises
    ------
    ValueError
        If the length of *v* does not match n*(n+1)//2.
    """
    v = np.asarray(v, dtype=float)
    expected = vech_size(n)
    if v.shape != (expected,):
        raise ValueError(
            f"ivech: expected vector of length {expected} for n={n}; got {v.shape}."
        )
    M = np.zeros((n, n), dtype=float)
    k = 0
    for j in range(n):
        for i in range(j, n):
            M[i, j] = v[k]
            M[j, i] = v[k]  # symmetry
            k += 1
    return M


def vech_size(n: int) -> int:
    """Return the length of the vech vector for an (n, n) symmetric matrix.

    Parameters
    ----------
    n : int
        Matrix dimension (must be positive).

    Returns
    -------
    int
        n*(n+1)//2
    """
    if n < 1:
        raise ValueError(f"vech_size: n must be a positive integer; got {n}.")
    return n * (n + 1) // 2


# ---------------------------------------------------------------------------
# Parzen kernel
# ---------------------------------------------------------------------------

def parzen_kernel(x: float) -> float:
    """Evaluate the Parzen kernel weight at a single point.

    The Parzen kernel is defined as::

        k(x) = 1 - 6x² + 6|x|³    if |x| ≤ 0.5
              = 2(1 - |x|)³         if 0.5 < |x| ≤ 1
              = 0                    if |x| > 1

    This kernel is used for bandwidth weighting in multivariate realized
    kernel estimation (Barndorff-Nielsen et al., 2011).

    Parameters
    ----------
    x : float
        Evaluation point (typically h/H for lag h and bandwidth H).

    Returns
    -------
    float
        Non-negative kernel weight in [0, 1].
    """
    ax = abs(x)
    if ax <= 0.5:
        return 1.0 - 6.0 * ax ** 2 + 6.0 * ax ** 3
    elif ax <= 1.0:
        return 2.0 * (1.0 - ax) ** 3
    return 0.0


def parzen_weights(H: int) -> np.ndarray:
    """Return Parzen kernel weights k(h/H) for h = 0, 1, …, H.

    Parameters
    ----------
    H : int
        Bandwidth parameter (positive integer).  The returned array has
        length H+1.

    Returns
    -------
    np.ndarray
        1-D array of shape (H+1,) with k(0/H), k(1/H), …, k(H/H).

    Raises
    ------
    ValueError
        If *H* is not a positive integer.
    """
    if H < 1:
        raise ValueError(f"parzen_weights: H must be a positive integer; got {H}.")
    return np.array([parzen_kernel(h / H) for h in range(H + 1)], dtype=float)


# ---------------------------------------------------------------------------
# Positive-definiteness utilities
# ---------------------------------------------------------------------------

def nearest_pd(A: np.ndarray) -> np.ndarray:
    """Return the nearest positive definite matrix via eigendecomposition.

    Symmetrizes *A*, computes its eigendecomposition, clips all eigenvalues
    to a minimum of ``eps = 1e-8``, and reconstructs.

    Parameters
    ----------
    A : np.ndarray
        Square (n, n) array that may not be positive definite.

    Returns
    -------
    np.ndarray
        Nearest positive definite (n, n) array.

    Raises
    ------
    ValueError
        If *A* is not a 2-D square array.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"nearest_pd: expected square 2-D array; got shape {A.shape}.")
    eps = 1e-8
    A_sym = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def is_pd(A: np.ndarray) -> bool:
    """Check whether a matrix is positive definite via Cholesky decomposition.

    Parameters
    ----------
    A : np.ndarray
        Square (n, n) array.

    Returns
    -------
    bool
        ``True`` if *A* admits a Cholesky factorization, ``False`` otherwise.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def safe_inv(A: np.ndarray) -> np.ndarray:
    """Invert a matrix using Cholesky; fall back to nearest PD if necessary.

    First attempts Cholesky-based inversion (``scipy.linalg.cho_solve``).  If
    *A* is not positive definite, projects it to the nearest PD matrix via
    :func:`nearest_pd`, logs a warning, and retries.

    Parameters
    ----------
    A : np.ndarray
        Square (n, n) array intended to be positive definite.

    Returns
    -------
    np.ndarray
        Inverse of *A* (or of its nearest-PD projection).

    Raises
    ------
    ValueError
        If *A* is not a 2-D square array.
    """
    import scipy.linalg as la

    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"safe_inv: expected square 2-D array; got shape {A.shape}.")

    try:
        c, low = la.cho_factor(A)
        return la.cho_solve((c, low), np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        logger.warning(
            "safe_inv: matrix is not positive definite; projecting to nearest PD."
        )
        A_pd = nearest_pd(A)
        c, low = la.cho_factor(A_pd)
        return la.cho_solve((c, low), np.eye(A.shape[0]))


def safe_logdet(A: np.ndarray) -> float:
    """Compute the log-determinant of a matrix via numpy's slogdet.

    Logs a warning if *A* does not have a positive determinant (which would
    indicate it is not positive definite).

    Parameters
    ----------
    A : np.ndarray
        Square (n, n) array.

    Returns
    -------
    float
        Natural logarithm of ``|det(A)|``.

    Raises
    ------
    ValueError
        If *A* is not a 2-D square array.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"safe_logdet: expected square 2-D array; got shape {A.shape}.")
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        logger.warning(
            "safe_logdet: matrix has non-positive determinant (sign=%d); "
            "result may be unreliable.",
            sign,
        )
    return logdet


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def compute_rolling_mean(
    x: Union[pd.Series, np.ndarray], window: int
) -> np.ndarray:
    """Compute a purely backward-looking rolling mean over a fixed window.

    For each position *t*, the mean is taken over the *window* observations
    strictly preceding *t* (i.e. observations t-window through t-1, **not**
    including the current observation).  Positions with fewer than *window*
    past observations are filled with ``NaN``.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        1-D sequence of length T.
    window : int
        Number of past observations to average (must be ≥ 1).

    Returns
    -------
    np.ndarray
        Array of shape (T,) with the rolling means; the first *window*
        entries are ``NaN``.

    Raises
    ------
    ValueError
        If *window* < 1 or *x* is not 1-D.
    """
    if window < 1:
        raise ValueError(f"compute_rolling_mean: window must be ≥ 1; got {window}.")

    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:
        raise ValueError(
            f"compute_rolling_mean: expected 1-D input; got shape {arr.shape}."
        )

    T = len(arr)
    result = np.full(T, np.nan, dtype=float)
    for t in range(window, T):
        result[t] = arr[t - window : t].mean()
    return result


# ---------------------------------------------------------------------------
# Annualization helpers
# ---------------------------------------------------------------------------

def annualize_vol(daily_vol: float, trading_days: int = 252) -> float:
    """Convert a daily volatility to an annualized figure.

    Parameters
    ----------
    daily_vol : float
        Daily volatility (standard deviation, not variance).
    trading_days : int, optional
        Number of trading days per year (default: 252).

    Returns
    -------
    float
        Annualized volatility = ``daily_vol * sqrt(trading_days)``.

    Raises
    ------
    ValueError
        If *daily_vol* is negative or *trading_days* < 1.
    """
    if daily_vol < 0:
        raise ValueError(f"annualize_vol: daily_vol must be non-negative; got {daily_vol}.")
    if trading_days < 1:
        raise ValueError(f"annualize_vol: trading_days must be ≥ 1; got {trading_days}.")
    return daily_vol * np.sqrt(trading_days)


def annualize_return(daily_return: float, trading_days: int = 252) -> float:
    """Convert a mean daily log-return to an annualized figure.

    Parameters
    ----------
    daily_return : float
        Mean daily log-return.
    trading_days : int, optional
        Number of trading days per year (default: 252).

    Returns
    -------
    float
        Annualized log-return = ``daily_return * trading_days``.

    Raises
    ------
    ValueError
        If *trading_days* < 1.
    """
    if trading_days < 1:
        raise ValueError(
            f"annualize_return: trading_days must be ≥ 1; got {trading_days}."
        )
    return daily_return * trading_days
