"""HAR-DRD and HARQ-DRD models for realized covariance forecasting.

Implements the Diagonal-Rotation-Diagonal (DRD) decomposition of a covariance
matrix into its variance and correlation components, then fits separate HAR or
HARQ models to each component.

The DRD decomposition::

    S_t = D_t R_t D_t,  D_t = diag(sqrt(diag(S_t))),  R_t = correlation matrix

Variance model (per asset n)::

    d_{n,t}^2 = alpha_{0,n} + alpha_1 d_{n,t-1}^2
                + alpha_2 d_{n,t-5|t-1}^2 + alpha_3 d_{n,t-22|t-1}^2

Correlation model (scalar HAR on off-diagonal lower-triangle of R_t)::

    r_{t,m} = theta_{0,m} + theta_1 r_{t-1,m}
              + theta_2 r_{t-5|t-1,m} + theta_3 r_{t-22|t-1,m}

For HARQ-DRD, the variance model is replaced with univariate HARQ::

    d_{n,t}^2 = alpha_{0,n} + (alpha_1 + alpha_{1Q} pi_{n,t-1}) d_{n,t-1}^2
                + alpha_2 d_{n,t-5|t-1}^2 + alpha_3 d_{n,t-22|t-1}^2
"""

import logging
from typing import Optional

import numpy as np
import statsmodels.api as sm

from src.utils import nearest_pd, is_pd, vech_size
from src.har_models import HARModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correlation matrix helpers
# ---------------------------------------------------------------------------

def vech_diag_indices(n: int) -> list[int]:
    """Return vech indices corresponding to diagonal matrix elements.

    For a column-major vech of an (n, n) matrix, the diagonal element (j, j)
    occupies position  j*n - j*(j-1)//2  in the vech vector.

    Parameters
    ----------
    n : int
        Matrix dimension (≥ 1).

    Returns
    -------
    list[int]
        List of length n containing the vech index for each diagonal element,
        ordered from (0,0) to (n-1, n-1).

    Raises
    ------
    ValueError
        If *n* < 1.
    """
    if n < 1:
        raise ValueError(f"vech_diag_indices: n must be ≥ 1; got {n}.")
    return [j * n - j * (j - 1) // 2 for j in range(n)]


def _extract_offdiag(R: np.ndarray, n: int) -> np.ndarray:
    """Extract the off-diagonal lower-triangle elements of R in column-major order.

    Parameters
    ----------
    R : np.ndarray
        Square (n, n) matrix.
    n : int
        Matrix dimension.

    Returns
    -------
    np.ndarray
        1-D array of length n*(n-1)//2.
    """
    elements = []
    for j in range(n):
        for i in range(j + 1, n):
            elements.append(R[i, j])
    return np.array(elements, dtype=float)


def _build_corr_matrix(r_offdiag: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct a symmetric matrix with unit diagonal from off-diagonal elements.

    Parameters
    ----------
    r_offdiag : np.ndarray
        Off-diagonal lower-triangle values, length n*(n-1)//2.
    n : int
        Target matrix dimension.

    Returns
    -------
    np.ndarray
        Symmetric (n, n) matrix with ones on the diagonal.
    """
    R = np.eye(n, dtype=float)
    k = 0
    for j in range(n):
        for i in range(j + 1, n):
            R[i, j] = r_offdiag[k]
            R[j, i] = r_offdiag[k]
            k += 1
    return R


def _ensure_valid_corr(R: np.ndarray) -> np.ndarray:
    """Project a near-correlation matrix to a valid PD correlation matrix.

    Symmetrises, applies :func:`~src.utils.nearest_pd`, then re-standardises
    so that diagonal elements are exactly one.

    Parameters
    ----------
    R : np.ndarray
        Approximate (n, n) correlation matrix.

    Returns
    -------
    np.ndarray
        Valid (n, n) correlation matrix (unit diagonal, symmetric, PD).
    """
    R = (R + R.T) / 2.0
    if not is_pd(R):
        logger.warning(
            "_ensure_valid_corr: correlation matrix is not PD; projecting to nearest PD."
        )
        R = nearest_pd(R)
    d = np.sqrt(np.diag(R))
    d_inv = np.where(d > 0.0, 1.0 / d, 0.0)
    return R * np.outer(d_inv, d_inv)


# ---------------------------------------------------------------------------
# DRD decomposition
# ---------------------------------------------------------------------------

class DRDDecomposition:
    """Decompose and recompose covariance matrices via DRD factorisation.

    S = D R D  where  D = diag(sqrt(diag(S))).
    """

    def decompose(self, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decompose a covariance matrix into volatilities and correlation.

        Parameters
        ----------
        S : np.ndarray
            Symmetric PD covariance matrix, shape (N, N).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (d_t, R_t) where d_t = sqrt(diag(S)) has shape (N,) and R_t is
            the (N, N) correlation matrix.  R_t is guaranteed PD via
            :func:`_ensure_valid_corr`.

        Raises
        ------
        ValueError
            If *S* is not a 2-D square array.
        """
        S = np.asarray(S, dtype=float)
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(
                f"DRDDecomposition.decompose: expected square 2-D array; got {S.shape}."
            )
        n = S.shape[0]
        diag_vals = np.diag(S)
        d_t = np.sqrt(np.maximum(diag_vals, 0.0))       # shape (N,)
        d_inv = np.where(d_t > 0.0, 1.0 / d_t, 0.0)
        D_inv = np.diag(d_inv)
        R_t = D_inv @ S @ D_inv                          # correlation matrix
        R_t = _ensure_valid_corr(R_t)
        return d_t, R_t

    def recompose(self, d_t: np.ndarray, R_t: np.ndarray) -> np.ndarray:
        """Reconstruct a covariance matrix from volatilities and correlation.

        Parameters
        ----------
        d_t : np.ndarray
            Per-asset standard deviations (square roots of variances), shape (N,).
        R_t : np.ndarray
            Correlation matrix, shape (N, N).

        Returns
        -------
        np.ndarray
            Covariance matrix S = diag(d_t) @ R_t @ diag(d_t), shape (N, N).
        """
        d_t = np.asarray(d_t, dtype=float)
        D = np.diag(d_t)
        S = D @ R_t @ D
        S = (S + S.T) / 2.0
        if not is_pd(S):
            logger.warning(
                "DRDDecomposition.recompose: recomposed matrix is not PD; "
                "projecting to nearest PD."
            )
            S = nearest_pd(S)
        return S


# ---------------------------------------------------------------------------
# Univariate HAR helpers
# ---------------------------------------------------------------------------

def fit_univariate_har(y: np.ndarray) -> dict:
    """Fit a univariate HAR model via OLS.

    Model::

        y_t = alpha_0 + alpha_1 y_{t-1}
              + alpha_2 y_{t-5|t-1} + alpha_3 y_{t-22|t-1} + eps_t

    Parameters
    ----------
    y : np.ndarray
        Univariate time series, shape (T,).  Requires T ≥ 23.

    Returns
    -------
    dict
        Keys: ``alpha_0``, ``alpha_1``, ``alpha_2``, ``alpha_3`` (floats),
        ``se`` (np.ndarray of four standard errors).

    Raises
    ------
    ValueError
        If *y* is not 1-D or has fewer than 23 observations.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(
            f"fit_univariate_har: expected 1-D array; got shape {y.shape}."
        )
    T = len(y)
    if T < 23:
        raise ValueError(
            f"fit_univariate_har: need T ≥ 23 (got T={T}) to form 22-lag average."
        )

    T_eff = T - 22
    y_target = y[22:]                                   # shape (T_eff,)
    lag1 = y[21 : T - 1]                               # y_{t-1}
    avg5 = np.array([y[t - 5 : t].mean() for t in range(22, T)])
    avg22 = np.array([y[t - 22 : t].mean() for t in range(22, T)])

    X = sm.add_constant(np.column_stack([lag1, avg5, avg22]))   # (T_eff, 4)
    result = sm.OLS(y_target, X).fit()
    p = result.params
    return {
        "alpha_0": float(p[0]),
        "alpha_1": float(p[1]),
        "alpha_2": float(p[2]),
        "alpha_3": float(p[3]),
        "se": result.bse.astype(float),
    }


def predict_univariate_har(y: np.ndarray, params: dict, t: int) -> float:
    """One-step-ahead univariate HAR forecast for day *t*.

    Parameters
    ----------
    y : np.ndarray
        Full history array, shape (T,).  Must satisfy T > t and t ≥ 22.
    params : dict
        Parameter dict as returned by :func:`fit_univariate_har`.
    t : int
        Forecast target index (≥ 22); uses y[0..t-1].

    Returns
    -------
    float
        Scalar one-step-ahead forecast.

    Raises
    ------
    ValueError
        If t < 22 or t > len(y).
    """
    y = np.asarray(y, dtype=float)
    if t < 22:
        raise ValueError(f"predict_univariate_har: need t ≥ 22; got t={t}.")
    if t > len(y):
        raise ValueError(
            f"predict_univariate_har: t={t} exceeds series length {len(y)}."
        )
    lag1 = y[t - 1]
    avg5 = y[t - 5 : t].mean()
    avg22 = y[t - 22 : t].mean()
    return (
        params["alpha_0"]
        + params["alpha_1"] * lag1
        + params["alpha_2"] * avg5
        + params["alpha_3"] * avg22
    )


def fit_univariate_harq(y: np.ndarray, pi: np.ndarray) -> dict:
    """Fit a univariate HARQ model via OLS.

    Model::

        y_t = alpha_0 + alpha_1 y_{t-1} + alpha_{1Q} (pi_{t-1} y_{t-1})
              + alpha_2 y_{t-5|t-1} + alpha_3 y_{t-22|t-1} + eps_t

    Parameters
    ----------
    y : np.ndarray
        Univariate target series, shape (T,).
    pi : np.ndarray
        Demeaned measurement-error proxy series, shape (T,).

    Returns
    -------
    dict
        Keys: ``alpha_0``, ``alpha_1``, ``alpha_1Q``, ``alpha_2``, ``alpha_3``
        (floats), ``se`` (np.ndarray of five standard errors).

    Raises
    ------
    ValueError
        If *y* and *pi* do not have the same shape or T < 23.
    """
    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    if y.shape != pi.shape:
        raise ValueError(
            f"fit_univariate_harq: y shape {y.shape} != pi shape {pi.shape}."
        )
    if y.ndim != 1:
        raise ValueError(
            f"fit_univariate_harq: expected 1-D arrays; got shape {y.shape}."
        )
    T = len(y)
    if T < 23:
        raise ValueError(
            f"fit_univariate_harq: need T ≥ 23 (got T={T}) to form 22-lag average."
        )

    T_eff = T - 22
    y_target = y[22:]
    lag1 = y[21 : T - 1]                               # y_{t-1}
    pi_lag1 = pi[21 : T - 1]                           # pi_{t-1}
    pi_x1 = pi_lag1 * lag1                             # pi_{t-1} * y_{t-1}
    avg5 = np.array([y[t - 5 : t].mean() for t in range(22, T)])
    avg22 = np.array([y[t - 22 : t].mean() for t in range(22, T)])

    X = sm.add_constant(np.column_stack([lag1, pi_x1, avg5, avg22]))  # (T_eff, 5)
    result = sm.OLS(y_target, X).fit()
    p = result.params
    return {
        "alpha_0": float(p[0]),
        "alpha_1": float(p[1]),
        "alpha_1Q": float(p[2]),
        "alpha_2": float(p[3]),
        "alpha_3": float(p[4]),
        "se": result.bse.astype(float),
    }


def predict_univariate_harq(
    y: np.ndarray, pi: np.ndarray, params: dict, t: int
) -> float:
    """One-step-ahead univariate HARQ forecast for day *t*.

    Parameters
    ----------
    y : np.ndarray
        Full history array, shape (T,).
    pi : np.ndarray
        Full demeaned pi series, shape (T,).
    params : dict
        Parameter dict as returned by :func:`fit_univariate_harq`.
    t : int
        Forecast target index (≥ 22).

    Returns
    -------
    float
        Scalar one-step-ahead HARQ forecast.

    Raises
    ------
    ValueError
        If t < 22.
    """
    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    if t < 22:
        raise ValueError(f"predict_univariate_harq: need t ≥ 22; got t={t}.")
    if t > len(y):
        raise ValueError(
            f"predict_univariate_harq: t={t} exceeds series length {len(y)}."
        )
    lag1 = y[t - 1]
    pi_lag1 = pi[t - 1]
    avg5 = y[t - 5 : t].mean()
    avg22 = y[t - 22 : t].mean()
    return (
        params["alpha_0"]
        + params["alpha_1"] * lag1
        + params["alpha_1Q"] * pi_lag1 * lag1
        + params["alpha_2"] * avg5
        + params["alpha_3"] * avg22
    )


# ---------------------------------------------------------------------------
# Internal series extraction
# ---------------------------------------------------------------------------

def _decompose_series(
    S_series: list, decomp: DRDDecomposition
) -> tuple[np.ndarray, np.ndarray]:
    """Extract variance and correlation time series from a list of covariance matrices.

    Parameters
    ----------
    S_series : list of np.ndarray
        Length-T list of (N, N) realized covariance matrices.
    decomp : DRDDecomposition
        Decomposition helper instance.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (d_sq_series, r_series) where
        - d_sq_series has shape (T, N) with d_sq_series[t, n] = S_t[n, n]
        - r_series has shape (T, N_offdiag) with off-diagonal R_t elements
    """
    T = len(S_series)
    N = S_series[0].shape[0]
    N_offdiag = N * (N - 1) // 2

    d_sq_series = np.zeros((T, N), dtype=float)
    r_series = np.zeros((T, N_offdiag), dtype=float)

    for t, S in enumerate(S_series):
        d_t, R_t = decomp.decompose(S)
        d_sq_series[t] = d_t ** 2
        r_series[t] = _extract_offdiag(R_t, N)

    return d_sq_series, r_series


# ---------------------------------------------------------------------------
# HAR-DRD
# ---------------------------------------------------------------------------

class HARDRDModel:
    """HAR-DRD model: separate univariate HAR for variances; scalar HAR for correlations.

    Variances d_{n,t}^2 (n = 0, …, N-1) are each modelled by an independent
    univariate HAR fitted via OLS.  The off-diagonal elements of the correlation
    matrix R_t are jointly modelled by a scalar (pooled) HAR using
    :class:`~src.har_models.HARModel`.

    Forecast covariance matrices are guaranteed PD via
    :class:`DRDDecomposition`.
    """

    def __init__(self) -> None:
        self._decomp = DRDDecomposition()
        self._har = HARModel()

    # ------------------------------------------------------------------
    def fit(
        self, S_series: list, params: Optional[dict] = None
    ) -> dict:
        """Fit variance HARs and correlation HAR.

        Parameters
        ----------
        S_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        params : dict, optional
            Ignored; reserved for future warm-starting.

        Returns
        -------
        dict
            Keys:
            - ``var_params``: list of N dicts, one per asset, each with keys
              ``alpha_0``, ``alpha_1``, ``alpha_2``, ``alpha_3``, ``se``.
            - ``cor_params``: dict with keys ``theta_0`` (N_offdiag,),
              ``theta_1``, ``theta_2``, ``theta_3`` (scalars), and SE entries.

        Raises
        ------
        ValueError
            If *S_series* has fewer than 23 elements.
        """
        T = len(S_series)
        if T < 23:
            raise ValueError(
                f"HARDRDModel.fit: need T ≥ 23 (got T={T}) to form 22-lag average."
            )
        N = S_series[0].shape[0]

        d_sq_series, r_series = _decompose_series(S_series, self._decomp)

        # Variance models: N independent univariate HARs
        var_params = [fit_univariate_har(d_sq_series[:, n]) for n in range(N)]

        # Correlation model: scalar HAR on off-diagonal elements
        cor_params = self._har.fit(r_series)

        return {"var_params": var_params, "cor_params": cor_params}

    # ------------------------------------------------------------------
    def forecast_from_last(
        self,
        d_last: np.ndarray,
        r_last: np.ndarray,
        d_avg5: np.ndarray,
        r_avg5: np.ndarray,
        d_avg22: np.ndarray,
        r_avg22: np.ndarray,
        var_params: list,
        cor_params: dict,
    ) -> np.ndarray:
        """Produce a forecast covariance matrix from pre-computed lagged components.

        Parameters
        ----------
        d_last : np.ndarray
            Lagged variances d_{n,t-1}^2, shape (N,).
        r_last : np.ndarray
            Lagged off-diagonal correlation elements r_{t-1}, shape (N_offdiag,).
        d_avg5 : np.ndarray
            Five-day average variances, shape (N,).
        r_avg5 : np.ndarray
            Five-day average correlations, shape (N_offdiag,).
        d_avg22 : np.ndarray
            Twenty-two-day average variances, shape (N,).
        r_avg22 : np.ndarray
            Twenty-two-day average correlations, shape (N_offdiag,).
        var_params : list of dict
            Per-asset variance HAR parameters.
        cor_params : dict
            Scalar HAR parameters for off-diagonal correlations.

        Returns
        -------
        np.ndarray
            Forecast covariance matrix, shape (N, N), symmetric and PD.
        """
        N = len(var_params)

        # Forecast each variance
        d_sq_hat = np.array([
            var_params[n]["alpha_0"]
            + var_params[n]["alpha_1"] * float(d_last[n])
            + var_params[n]["alpha_2"] * float(d_avg5[n])
            + var_params[n]["alpha_3"] * float(d_avg22[n])
            for n in range(N)
        ])
        d_hat = np.sqrt(np.maximum(d_sq_hat, 0.0))     # std devs, shape (N,)

        # Forecast off-diagonal correlation elements
        r_hat = (
            cor_params["theta_0"]
            + cor_params["theta_1"] * r_last
            + cor_params["theta_2"] * r_avg5
            + cor_params["theta_3"] * r_avg22
        )

        # Clip to [-1, 1] before building correlation matrix
        r_hat = np.clip(r_hat, -1.0 + 1e-8, 1.0 - 1e-8)

        # Build and validate correlation matrix
        R_hat = _build_corr_matrix(r_hat, N)
        R_hat = _ensure_valid_corr(R_hat)

        # Recompose S
        return self._decomp.recompose(d_hat, R_hat)

    # ------------------------------------------------------------------
    def predict_one_step(
        self, S_series: list, t: int, params: dict
    ) -> np.ndarray:
        """One-step-ahead forecast for day *t*.

        Parameters
        ----------
        S_series : list of np.ndarray
            Full list of (N, N) covariance matrices (length ≥ t).
        t : int
            Forecast target index (≥ 22).  Uses S_series[0..t-1].
        params : dict
            Parameter dict as returned by :meth:`fit`.

        Returns
        -------
        np.ndarray
            Forecast covariance matrix of shape (N, N).

        Raises
        ------
        ValueError
            If t < 22 or t > len(S_series).
        """
        if t < 22:
            raise ValueError(f"HARDRDModel.predict_one_step: need t ≥ 22; got t={t}.")
        if t > len(S_series):
            raise ValueError(
                f"HARDRDModel.predict_one_step: t={t} exceeds series length {len(S_series)}."
            )

        N = S_series[0].shape[0]
        d_sq_series, r_series = _decompose_series(S_series[:t], self._decomp)

        d_last = d_sq_series[t - 1]
        r_last = r_series[t - 1]
        d_avg5 = d_sq_series[t - 5 : t].mean(axis=0)
        r_avg5 = r_series[t - 5 : t].mean(axis=0)
        d_avg22 = d_sq_series[t - 22 : t].mean(axis=0)
        r_avg22 = r_series[t - 22 : t].mean(axis=0)

        return self.forecast_from_last(
            d_last, r_last, d_avg5, r_avg5, d_avg22, r_avg22,
            params["var_params"], params["cor_params"],
        )


# ---------------------------------------------------------------------------
# HARQ-DRD
# ---------------------------------------------------------------------------

class HARQDRDModel:
    """HARQ-DRD model: univariate HARQ for variances; scalar HAR for correlations.

    Each asset variance d_{n,t}^2 is modelled by a univariate HARQ that
    incorporates the measurement-error proxy pi_{n,t-1} (the vech-diagonal
    element of Pi_t corresponding to asset n's variance).  The correlation
    component is modelled with a standard scalar HAR identical to
    :class:`HARDRDModel`.
    """

    def __init__(self) -> None:
        self._decomp = DRDDecomposition()
        self._har = HARModel()

    # ------------------------------------------------------------------
    def fit(self, S_series: list, pi_series: np.ndarray) -> dict:
        """Fit variance HARQs and correlation HAR.

        Parameters
        ----------
        S_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        pi_series : np.ndarray
            Full pi_t vectors, shape (T, K) where K = N*(N+1)//2.
            Element pi_series[t, vech_diag_indices(N)[n]] gives pi_{n,t}.

        Returns
        -------
        dict
            Keys:
            - ``var_params``: list of N dicts with keys ``alpha_0``,
              ``alpha_1``, ``alpha_1Q``, ``alpha_2``, ``alpha_3``, ``se``.
            - ``cor_params``: dict (same structure as HAR-DRD).

        Raises
        ------
        ValueError
            If *S_series* has fewer than 23 elements or *pi_series* has wrong shape.
        """
        T = len(S_series)
        if T < 23:
            raise ValueError(
                f"HARQDRDModel.fit: need T ≥ 23 (got T={T}) to form 22-lag average."
            )
        pi_series = np.asarray(pi_series, dtype=float)
        N = S_series[0].shape[0]
        K = vech_size(N)
        if pi_series.shape != (T, K):
            raise ValueError(
                f"HARQDRDModel.fit: pi_series must have shape (T={T}, K={K}); "
                f"got {pi_series.shape}."
            )

        d_sq_series, r_series = _decompose_series(S_series, self._decomp)

        # Diagonal vech indices → pi for each variance
        diag_idx = vech_diag_indices(N)                 # length-N list of ints

        # Variance models: N independent univariate HARQs
        var_params = [
            fit_univariate_harq(d_sq_series[:, n], pi_series[:, diag_idx[n]])
            for n in range(N)
        ]

        # Correlation model: scalar HAR (no HARQ for correlations)
        cor_params = self._har.fit(r_series)

        return {"var_params": var_params, "cor_params": cor_params}

    # ------------------------------------------------------------------
    def predict_one_step(
        self,
        S_series: list,
        pi_series: np.ndarray,
        t: int,
        params: dict,
    ) -> np.ndarray:
        """One-step-ahead forecast for day *t*.

        Parameters
        ----------
        S_series : list of np.ndarray
            Full list of (N, N) covariance matrices (length ≥ t).
        pi_series : np.ndarray
            Full demeaned pi series, shape (T, K).
        t : int
            Forecast target index (≥ 22).
        params : dict
            Parameter dict as returned by :meth:`fit`.

        Returns
        -------
        np.ndarray
            Forecast covariance matrix of shape (N, N).

        Raises
        ------
        ValueError
            If t < 22.
        """
        if t < 22:
            raise ValueError(
                f"HARQDRDModel.predict_one_step: need t ≥ 22; got t={t}."
            )
        if t > len(S_series):
            raise ValueError(
                f"HARQDRDModel.predict_one_step: t={t} exceeds series length {len(S_series)}."
            )

        pi_series = np.asarray(pi_series, dtype=float)
        N = S_series[0].shape[0]
        diag_idx = vech_diag_indices(N)

        d_sq_series, r_series = _decompose_series(S_series[:t], self._decomp)

        # Forecast variances with HARQ
        d_sq_hat = np.array([
            predict_univariate_harq(
                d_sq_series[:, n],
                pi_series[:t, diag_idx[n]],
                params["var_params"][n],
                t,
            )
            for n in range(N)
        ])
        d_hat = np.sqrt(np.maximum(d_sq_hat, 0.0))

        # Forecast correlations with standard HAR
        r_last = r_series[t - 1]
        r_avg5 = r_series[t - 5 : t].mean(axis=0)
        r_avg22 = r_series[t - 22 : t].mean(axis=0)
        cor_params = params["cor_params"]
        r_hat = (
            cor_params["theta_0"]
            + cor_params["theta_1"] * r_last
            + cor_params["theta_2"] * r_avg5
            + cor_params["theta_3"] * r_avg22
        )
        r_hat = np.clip(r_hat, -1.0 + 1e-8, 1.0 - 1e-8)

        R_hat = _build_corr_matrix(r_hat, N)
        R_hat = _ensure_valid_corr(R_hat)

        return self._decomp.recompose(d_hat, R_hat)
