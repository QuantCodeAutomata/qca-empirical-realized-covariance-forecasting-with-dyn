"""Ex-post covariance shrinkage benchmarks for realized covariance forecasting.

Implements structured shrinkage targets (identity, equicorrelation, single-
factor) and the Ledoit–Wolf (2004) optimal shrinkage intensity estimator.
These are used as post-processing benchmarks applied to the raw covariance
forecasts produced by HAR/HARQ and related models.
"""

import logging
from typing import Optional

import numpy as np

from src.utils import nearest_pd, is_pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shrinkage targets
# ---------------------------------------------------------------------------

def identity_target(n: int) -> np.ndarray:
    """Return the scaled identity matrix as a shrinkage target.

    Parameters
    ----------
    n : int
        Matrix dimension (must be ≥ 1).

    Returns
    -------
    np.ndarray
        Identity matrix of shape (n, n).

    Raises
    ------
    ValueError
        If *n* < 1.
    """
    if n < 1:
        raise ValueError(f"identity_target: n must be ≥ 1; got {n}.")
    return np.eye(n, dtype=float)


def equicorrelation_target(S: np.ndarray) -> np.ndarray:
    """Equicorrelation shrinkage target.

    Constructs a structured matrix that preserves the sample variances on
    the diagonal and sets all off-diagonal correlations to the average
    sample pairwise correlation:

    .. math::

        F_{ij} =
        \\begin{cases}
            S_{ii}                              & i = j \\\\
            \\bar{\\rho} \\sqrt{S_{ii} S_{jj}} & i \\neq j
        \\end{cases}

    where :math:`\\bar{\\rho}` is the mean of the off-diagonal elements of
    the sample correlation matrix.

    Parameters
    ----------
    S : np.ndarray
        Sample covariance matrix of shape (N, N).  Must be square and have
        positive diagonal entries.

    Returns
    -------
    np.ndarray
        Equicorrelation target matrix of shape (N, N), symmetric and PD.

    Raises
    ------
    ValueError
        If *S* is not a 2-D square array or has any non-positive diagonal entry.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"equicorrelation_target: S must be square 2-D; got {S.shape}.")
    N = S.shape[0]

    diag = np.diag(S)
    if np.any(diag <= 0.0):
        raise ValueError(
            "equicorrelation_target: S has non-positive diagonal entries; "
            "cannot convert to correlation."
        )

    std = np.sqrt(diag)
    # Sample correlation matrix
    R = S / np.outer(std, std)
    np.fill_diagonal(R, 1.0)    # enforce exact 1s on diagonal

    # Mean off-diagonal correlation
    mask = ~np.eye(N, dtype=bool)
    rho_bar = float(R[mask].mean())

    # Build target: equicorrelation structure
    F = rho_bar * np.outer(std, std)
    np.fill_diagonal(F, diag)

    # Ensure PD
    F = (F + F.T) / 2.0
    if not is_pd(F):
        logger.debug("equicorrelation_target: result not PD; projecting to nearest PD.")
        F = nearest_pd(F)

    return F


def single_factor_target(
    S: np.ndarray,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Single-factor (market model) shrinkage target.

    If *returns* are provided, fits a PCA single-factor model and uses the
    rank-1 factor structure as the shrinkage target (augmented with
    idiosyncratic variances on the diagonal).  If *returns* are not provided,
    falls back to :func:`equicorrelation_target`.

    **PCA factor model**:

    .. math::

        F = \\sigma_1^2 \\, v_1 v_1^{\\prime} +
            \\operatorname{diag}(S - \\sigma_1^2 v_1 v_1^{\\prime})

    where :math:`\\sigma_1^2` and :math:`v_1` are the leading eigenvalue and
    eigenvector of *S*.

    Parameters
    ----------
    S : np.ndarray
        Sample covariance matrix of shape (N, N).
    returns : np.ndarray, optional
        Return matrix of shape (T, N).  If provided, the sample covariance
        from *returns* is used to compute the PCA factor; otherwise *S*
        itself is used.

    Returns
    -------
    np.ndarray
        Single-factor shrinkage target of shape (N, N), symmetric and PD.

    Raises
    ------
    ValueError
        If *S* is not square 2-D.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"single_factor_target: S must be square 2-D; got {S.shape}.")
    N = S.shape[0]

    if returns is None:
        # Fallback to equicorrelation target
        return equicorrelation_target(S)

    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[1] != N:
        logger.warning(
            "single_factor_target: returns shape %s incompatible with N=%d; "
            "falling back to equicorrelation.",
            returns.shape, N,
        )
        return equicorrelation_target(S)

    # Compute sample covariance from returns (may differ from S if S is a
    # realized covariance proxy)
    S_ret = np.cov(returns.T, ddof=1) if returns.shape[0] > 1 else S

    # Leading eigenvector (principal component ~ market factor)
    eigvals, eigvecs = np.linalg.eigh(S_ret)
    # eigh returns eigenvalues in ascending order; take the last one
    lambda1 = float(eigvals[-1])
    v1 = eigvecs[:, -1]                     # shape (N,)

    # Factor covariance contribution
    factor_cov = lambda1 * np.outer(v1, v1)   # rank-1

    # Idiosyncratic variances: diagonal of (S - factor_cov), floored at 0
    resid_diag = np.diag(S - factor_cov)
    resid_diag = np.maximum(resid_diag, 0.0)

    F = factor_cov + np.diag(resid_diag)
    F = (F + F.T) / 2.0

    if not is_pd(F):
        logger.debug("single_factor_target: result not PD; projecting to nearest PD.")
        F = nearest_pd(F)

    return F


# ---------------------------------------------------------------------------
# Ledoit–Wolf optimal shrinkage intensity
# ---------------------------------------------------------------------------

def compute_shrinkage_intensity(
    H: np.ndarray,
    F: np.ndarray,
    S: np.ndarray,
    residuals: np.ndarray,
) -> float:
    """Ledoit–Wolf optimal shrinkage intensity α*.

    Estimates the convex combination weight α* ∈ [0, 1] such that

    .. math::

        \\hat{\\Sigma} = \\alpha^* F + (1 - \\alpha^*) H

    minimises the expected Frobenius distance to the true covariance.

    Following Ledoit & Wolf (2004, eq. 14), the intensity is estimated as:

    .. math::

        \\alpha^* = \\frac{\\sum_{i,j} \\widehat{\\mathrm{Var}}(s_{ij}) -
                    \\widehat{\\mathrm{Cov}}(f_{ij}, s_{ij})}{\\sum_{i,j}
                    (\\phi_{ij} - \\sigma_{ij})^2}

    where the variances and covariances are estimated from the residual sample
    *residuals*, and :math:`\\phi_{ij}`, :math:`\\sigma_{ij}` are elements of
    *F* and *H* respectively.  The result is clipped to [0, 1].

    Parameters
    ----------
    H : np.ndarray
        Forecast covariance matrix (model estimate), shape (N, N).
    F : np.ndarray
        Shrinkage target matrix, shape (N, N).
    S : np.ndarray
        Realized (sample) covariance matrix, shape (N, N).
    residuals : np.ndarray
        Return residual matrix of shape (T, N) used to estimate the
        asymptotic variances/covariances.

    Returns
    -------
    float
        Optimal shrinkage intensity α* clipped to [0, 1].

    Raises
    ------
    ValueError
        If any of *H*, *F*, *S* are not square 2-D of the same shape, or
        *residuals* is not 2-D with N columns.
    """
    H = np.asarray(H, dtype=float)
    F = np.asarray(F, dtype=float)
    S = np.asarray(S, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"compute_shrinkage_intensity: H must be square 2-D; got {H.shape}.")
    N = H.shape[0]
    if F.shape != (N, N):
        raise ValueError(
            f"compute_shrinkage_intensity: F shape {F.shape} != H shape {H.shape}."
        )
    if S.shape != (N, N):
        raise ValueError(
            f"compute_shrinkage_intensity: S shape {S.shape} != H shape {H.shape}."
        )
    if residuals.ndim != 2 or residuals.shape[1] != N:
        raise ValueError(
            f"compute_shrinkage_intensity: residuals must be (T, {N}); "
            f"got {residuals.shape}."
        )
    T_res = residuals.shape[0]
    if T_res < 2:
        logger.warning(
            "compute_shrinkage_intensity: residuals has T=%d < 2; returning 0.5.",
            T_res,
        )
        return 0.5

    # ------------------------------------------------------------------
    # Numerator: sum_{ij} [Var(s_ij) - Cov(f_ij, s_ij)]
    # ------------------------------------------------------------------
    # Estimate element-wise sample variance and covariance from residuals.
    # s_ij^(t) ≈ r_i^(t) * r_j^(t)  (outer product proxy for covariance element)
    # f_ij is fixed (element of F); Cov(f_ij, s_ij) = 0 since f_ij is constant.
    # Therefore numerator simplifies to sum_{ij} Var(s_ij).

    # Compute per-observation sample covariance element estimates
    # s_ij^(t) = r_i^t * r_j^t  (uncentred cross-product)
    s_obs = residuals[:, :, np.newaxis] * residuals[:, np.newaxis, :]  # (T, N, N)
    s_var = s_obs.var(axis=0)    # (N, N): Var of each element across T observations

    numerator = float(s_var.sum())

    # ------------------------------------------------------------------
    # Denominator: sum_{ij} (phi_ij - sigma_ij)^2 = ||F - H||_F^2
    # ------------------------------------------------------------------
    diff = F - H
    denominator = float(np.sum(diff ** 2))

    if denominator < 1e-14:
        logger.warning(
            "compute_shrinkage_intensity: ||F - H||_F^2 ≈ 0; "
            "target and forecast are nearly identical, returning 0.0."
        )
        return 0.0

    alpha_star = numerator / denominator
    return float(np.clip(alpha_star, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Shrinkage application
# ---------------------------------------------------------------------------

def shrink_forecast(H: np.ndarray, F: np.ndarray, alpha: float) -> np.ndarray:
    """Apply shrinkage to a forecast matrix.

    .. math::

        \\hat{\\Sigma} = \\alpha F + (1 - \\alpha) H

    Parameters
    ----------
    H : np.ndarray
        Forecast covariance matrix of shape (N, N).
    F : np.ndarray
        Shrinkage target of shape (N, N).
    alpha : float
        Shrinkage intensity in [0, 1].  ``alpha = 0`` returns *H* unchanged;
        ``alpha = 1`` returns *F*.

    Returns
    -------
    np.ndarray
        Shrunk forecast of shape (N, N).

    Raises
    ------
    ValueError
        If *H* and *F* have different shapes, are not square 2-D, or *alpha*
        is outside [0, 1].
    """
    H = np.asarray(H, dtype=float)
    F = np.asarray(F, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"shrink_forecast: H must be square 2-D; got {H.shape}.")
    if F.shape != H.shape:
        raise ValueError(
            f"shrink_forecast: F shape {F.shape} != H shape {H.shape}."
        )
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"shrink_forecast: alpha must be in [0, 1]; got {alpha}.")

    return alpha * F + (1.0 - alpha) * H
