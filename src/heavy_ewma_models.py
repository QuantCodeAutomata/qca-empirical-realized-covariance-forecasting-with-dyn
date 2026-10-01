"""HEAVY, HEAVYQ, EWMA, and EWMAQ models for realized covariance forecasting.

All four models are estimated by maximising a composite log-likelihood (CL)
formed from contiguous bivariate pairs of daily returns, as in the HEAVY
framework of Noureldin, Shephard & Sheppard (2012).

Composite likelihood::

    CL(psi) = [1 / (T * (N-1))] *
              sum_{j=0}^{N-2} sum_{t=0}^{T-1} ell(V_{j,t}; r_{j,t})

    ell(V; r) = -0.5 * log|V| - 0.5 * r^T V^{-1} r

where r_{j,t} = (r_{j,t}, r_{j+1,t})^T and V_{j,t} is the corresponding
2 × 2 submatrix of the N × N forecast covariance V_t.

Model update equations
----------------------
EWMA::

    V_t = (1 - alpha) V_{t-1} + alpha S_{t-1}

EWMAQ (element-wise via vech)::

    vech(V_t) = (1 - alpha_t) o vech(V_{t-1}) + alpha_t o vech(S_{t-1})
    alpha_t = alpha * 1_K + alpha_Q * pi_{t-1}

HEAVY (covariance-targeting)::

    V_t = (1 - b - a) lambda_V + b V_{t-1} + a S_{t-1}
    lambda_V = sample_mean(S)

HEAVYQ (element-wise via vech)::

    vech(V_t) = (1 - b - a_t) o lambda_v + b * vech(V_{t-1}) + a_t o vech(S_{t-1})
    a_t = a * 1_K + a_Q * pi_{t-1},  lambda_v = vech(lambda_V)

All V_t sequences are initialised with V_0 = lambda_V.

References
----------
Noureldin, D., Shephard, N., & Sheppard, K. (2012). Multivariate high-frequency-
    based volatility (HEAVY) models. Journal of Applied Econometrics, 27, 907-933.
"""

import logging
from typing import Optional

import numpy as np
from scipy import optimize

from src.utils import nearest_pd, is_pd, safe_inv, vech, ivech, vech_size

logger = logging.getLogger(__name__)

_EPS = 1e-8          # eigenvalue floor for PD projections
_ALPHA_LO = 1e-4     # lower bound for decay parameters
_ALPHA_HI = 1 - 1e-4 # upper bound for decay parameters


# ---------------------------------------------------------------------------
# Composite log-likelihood
# ---------------------------------------------------------------------------

def composite_log_likelihood(
    returns: np.ndarray, V_series: list
) -> float:
    """Compute the composite log-likelihood over contiguous bivariate pairs.

    For each consecutive pair of assets j and j+1 (j = 0, …, N-2) and each
    time step t, the bivariate Gaussian log-likelihood contribution is::

        ell_{j,t} = -0.5 * log|V_{j,t}| - 0.5 * r_{j,t}^T V_{j,t}^{-1} r_{j,t}

    where V_{j,t} is the 2 × 2 submatrix of V_series[t] for assets j and j+1,
    and r_{j,t} is the corresponding 2-vector from *returns[t]*.

    Parameters
    ----------
    returns : np.ndarray
        Daily return matrix, shape (T, N).
    V_series : list of np.ndarray
        Length-T list of (N, N) forecast covariance matrices.  V_series[t]
        should be a one-step-ahead forecast for period t.

    Returns
    -------
    float
        CL value normalised by T*(N-1).  A larger (less-negative) value
        indicates a better-fitting model.

    Raises
    ------
    ValueError
        If *returns* is not 2-D or len(V_series) != T.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError(
            f"composite_log_likelihood: returns must be 2-D; got {returns.shape}."
        )
    T, N = returns.shape
    if len(V_series) != T:
        raise ValueError(
            f"composite_log_likelihood: len(V_series)={len(V_series)} != T={T}."
        )
    if N < 2:
        raise ValueError(
            "composite_log_likelihood: need N ≥ 2 assets for bivariate pairs."
        )

    total = 0.0
    pair_idx = [[j, j + 1] for j in range(N - 1)]     # N-1 contiguous pairs

    for j, (i0, i1) in enumerate(pair_idx):
        for t in range(T):
            r_pair = returns[t, [i0, i1]]               # shape (2,)
            V_t = np.asarray(V_series[t], dtype=float)
            V_pair = V_t[np.ix_([i0, i1], [i0, i1])]   # 2 × 2

            if not is_pd(V_pair):
                logger.debug(
                    "composite_log_likelihood: 2x2 submatrix for pair (%d,%d) "
                    "at t=%d is not PD; projecting.",
                    i0, i1, t,
                )
                V_pair = nearest_pd(V_pair)

            sign, logdet = np.linalg.slogdet(V_pair)
            if sign <= 0:
                # Degenerate: penalise with a large negative value
                total += -1e6
                continue

            V_inv = safe_inv(V_pair)
            total += -0.5 * logdet - 0.5 * (r_pair @ V_inv @ r_pair)

    return total / (T * (N - 1))


# ---------------------------------------------------------------------------
# Internal V_series generators (shared across classes and fit helpers)
# ---------------------------------------------------------------------------

def _ewma_series(
    s_series: list, alpha: float, lambda_V: np.ndarray
) -> list:
    """Generate the EWMA forecast series V_0, V_1, …, V_{T-1}.

    V_0 = lambda_V (initialisation).
    V_t = (1 - alpha) * V_{t-1} + alpha * S_{t-1}  for t ≥ 1.
    """
    T = len(s_series)
    V_list = []
    V_prev = lambda_V.copy()
    V_list.append(V_prev.copy())
    for t in range(1, T):
        V_t = (1.0 - alpha) * V_prev + alpha * np.asarray(s_series[t - 1], dtype=float)
        V_t = (V_t + V_t.T) / 2.0
        if not is_pd(V_t):
            logger.debug("_ewma_series: V_%d not PD; projecting.", t)
            V_t = nearest_pd(V_t)
        V_list.append(V_t)
        V_prev = V_t
    return V_list


def _ewmaq_series(
    s_series: list,
    pi_series: np.ndarray,
    alpha: float,
    alpha_Q: float,
    lambda_V: np.ndarray,
) -> list:
    """Generate the EWMAQ forecast series V_0, V_1, …, V_{T-1} (vech space).

    vech(V_t) = (1 - alpha_t) o vech(V_{t-1}) + alpha_t o vech(S_{t-1})
    alpha_t = alpha + alpha_Q * pi_{t-1}   (clipped to (_ALPHA_LO, _ALPHA_HI))
    """
    T = len(s_series)
    N = lambda_V.shape[0]
    K = vech_size(N)
    pi_series = np.asarray(pi_series, dtype=float)

    V_list = []
    v_prev = vech(lambda_V)                            # shape (K,)
    V_list.append(lambda_V.copy())

    for t in range(1, T):
        pi_prev = pi_series[t - 1]                     # shape (K,)
        alpha_t = np.clip(alpha + alpha_Q * pi_prev, _ALPHA_LO, _ALPHA_HI)
        v_s = vech(np.asarray(s_series[t - 1], dtype=float))
        v_t = (1.0 - alpha_t) * v_prev + alpha_t * v_s
        V_t = ivech(v_t, N)
        V_t = (V_t + V_t.T) / 2.0
        if not is_pd(V_t):
            logger.debug("_ewmaq_series: V_%d not PD; projecting.", t)
            V_t = nearest_pd(V_t)
        V_list.append(V_t)
        v_prev = vech(V_t)

    return V_list


def _heavy_series(
    s_series: list, a: float, b: float, lambda_V: np.ndarray
) -> list:
    """Generate the HEAVY forecast series V_0, V_1, …, V_{T-1}.

    V_t = (1 - b - a) * lambda_V + b * V_{t-1} + a * S_{t-1}
    """
    T = len(s_series)
    intercept = (1.0 - b - a) * lambda_V              # (N, N)
    V_list = []
    V_prev = lambda_V.copy()
    V_list.append(V_prev.copy())

    for t in range(1, T):
        S_prev = np.asarray(s_series[t - 1], dtype=float)
        V_t = intercept + b * V_prev + a * S_prev
        V_t = (V_t + V_t.T) / 2.0
        if not is_pd(V_t):
            logger.debug("_heavy_series: V_%d not PD; projecting.", t)
            V_t = nearest_pd(V_t)
        V_list.append(V_t)
        V_prev = V_t

    return V_list


def _heavyq_series(
    s_series: list,
    pi_series: np.ndarray,
    a: float,
    a_Q: float,
    b: float,
    lambda_V: np.ndarray,
) -> list:
    """Generate the HEAVYQ forecast series V_0, V_1, …, V_{T-1} (vech space).

    vech(V_t) = (1 - b - a_t) o lambda_v + b * vech(V_{t-1}) + a_t o vech(S_{t-1})
    a_t = a + a_Q * pi_{t-1}   (clipped to (1e-8, 1-b-1e-8))
    """
    T = len(s_series)
    N = lambda_V.shape[0]
    pi_series = np.asarray(pi_series, dtype=float)
    lambda_v = vech(lambda_V)                          # shape (K,)

    V_list = []
    v_prev = lambda_v.copy()
    V_list.append(lambda_V.copy())

    a_lo = _ALPHA_LO
    a_hi = max(_ALPHA_LO + 1e-8, 1.0 - b - _ALPHA_LO)

    for t in range(1, T):
        pi_prev = pi_series[t - 1]                     # shape (K,)
        a_t = np.clip(a + a_Q * pi_prev, a_lo, a_hi)  # shape (K,)
        v_s = vech(np.asarray(s_series[t - 1], dtype=float))
        v_t = (1.0 - b - a_t) * lambda_v + b * v_prev + a_t * v_s
        V_t = ivech(v_t, N)
        V_t = (V_t + V_t.T) / 2.0
        if not is_pd(V_t):
            logger.debug("_heavyq_series: V_%d not PD; projecting.", t)
            V_t = nearest_pd(V_t)
        V_list.append(V_t)
        v_prev = vech(V_t)

    return V_list


# ---------------------------------------------------------------------------
# Standalone optimisation helpers
# ---------------------------------------------------------------------------

def fit_alpha_ewma(
    returns: np.ndarray,
    s_series: list,
    bounds: tuple = ((_ALPHA_LO, _ALPHA_HI),),
) -> float:
    """Optimise the EWMA decay parameter alpha by composite likelihood.

    Parameters
    ----------
    returns : np.ndarray
        Daily return matrix, shape (T, N).
    s_series : list of np.ndarray
        Length-T list of (N, N) realized covariance matrices.
    bounds : tuple of (float, float), optional
        Box bounds for alpha.  Default: (1e-4, 1 - 1e-4).

    Returns
    -------
    float
        MLE estimate of alpha.
    """
    returns = np.asarray(returns, dtype=float)
    lambda_V = np.mean(np.stack(s_series, axis=0), axis=0)

    def neg_cl(params: np.ndarray) -> float:
        alpha = float(params[0])
        V_series = _ewma_series(s_series, alpha, lambda_V)
        return -composite_log_likelihood(returns, V_series)

    x0 = np.array([0.06])
    result = optimize.minimize(neg_cl, x0, bounds=bounds, method="L-BFGS-B")
    return float(result.x[0])


def fit_params_ewmaq(
    returns: np.ndarray,
    s_series: list,
    pi_series: np.ndarray,
) -> tuple[float, float]:
    """Optimise EWMAQ parameters (alpha, alpha_Q) by composite likelihood.

    Parameters
    ----------
    returns : np.ndarray
        Daily return matrix, shape (T, N).
    s_series : list of np.ndarray
        Length-T list of (N, N) realized covariance matrices.
    pi_series : np.ndarray
        Demeaned measurement-error vectors, shape (T, K).

    Returns
    -------
    tuple[float, float]
        (alpha, alpha_Q) estimates.  Expected sign: alpha_Q < 0.
    """
    returns = np.asarray(returns, dtype=float)
    pi_series = np.asarray(pi_series, dtype=float)
    lambda_V = np.mean(np.stack(s_series, axis=0), axis=0)

    bounds = [(_ALPHA_LO, _ALPHA_HI), (-1.0, 1.0)]

    def neg_cl(params: np.ndarray) -> float:
        alpha, alpha_Q = float(params[0]), float(params[1])
        V_series = _ewmaq_series(s_series, pi_series, alpha, alpha_Q, lambda_V)
        return -composite_log_likelihood(returns, V_series)

    x0 = np.array([0.06, -0.01])
    result = optimize.minimize(neg_cl, x0, bounds=bounds, method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def fit_params_heavy(
    returns: np.ndarray,
    s_series: list,
) -> tuple[float, float]:
    """Optimise HEAVY parameters (a, b) by composite likelihood.

    Enforces the stationarity constraint a + b < 1 with a > 0 and b > 0.

    Parameters
    ----------
    returns : np.ndarray
        Daily return matrix, shape (T, N).
    s_series : list of np.ndarray
        Length-T list of (N, N) realized covariance matrices.

    Returns
    -------
    tuple[float, float]
        (a, b) estimates.
    """
    returns = np.asarray(returns, dtype=float)
    lambda_V = np.mean(np.stack(s_series, axis=0), axis=0)

    bounds = [(_ALPHA_LO, _ALPHA_HI), (_ALPHA_LO, _ALPHA_HI)]
    constraints = [
        {"type": "ineq", "fun": lambda p: (1.0 - _ALPHA_LO) - p[0] - p[1]},
    ]

    def neg_cl(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        # Penalise infeasible region without hard clipping
        if a + b >= 1.0 - _ALPHA_LO:
            return 1e10
        V_series = _heavy_series(s_series, a, b, lambda_V)
        return -composite_log_likelihood(returns, V_series)

    x0 = np.array([0.05, 0.80])
    result = optimize.minimize(
        neg_cl, x0, bounds=bounds, constraints=constraints, method="SLSQP"
    )
    return float(result.x[0]), float(result.x[1])


def fit_params_heavyq(
    returns: np.ndarray,
    s_series: list,
    pi_series: np.ndarray,
) -> tuple[float, float, float]:
    """Optimise HEAVYQ parameters (a, a_Q, b) by composite likelihood.

    Parameters
    ----------
    returns : np.ndarray
        Daily return matrix, shape (T, N).
    s_series : list of np.ndarray
        Length-T list of (N, N) realized covariance matrices.
    pi_series : np.ndarray
        Demeaned measurement-error vectors, shape (T, K).

    Returns
    -------
    tuple[float, float, float]
        (a, a_Q, b) estimates.  Expected sign: a_Q < 0.
    """
    returns = np.asarray(returns, dtype=float)
    pi_series = np.asarray(pi_series, dtype=float)
    lambda_V = np.mean(np.stack(s_series, axis=0), axis=0)

    # params = [a, a_Q, b]
    bounds = [(_ALPHA_LO, _ALPHA_HI), (-1.0, 1.0), (_ALPHA_LO, _ALPHA_HI)]
    constraints = [
        {"type": "ineq", "fun": lambda p: (1.0 - _ALPHA_LO) - p[0] - p[2]},
    ]

    def neg_cl(params: np.ndarray) -> float:
        a, a_Q, b = float(params[0]), float(params[1]), float(params[2])
        if a + b >= 1.0 - _ALPHA_LO:
            return 1e10
        V_series = _heavyq_series(s_series, pi_series, a, a_Q, b, lambda_V)
        return -composite_log_likelihood(returns, V_series)

    x0 = np.array([0.05, -0.01, 0.80])
    result = optimize.minimize(
        neg_cl, x0, bounds=bounds, constraints=constraints, method="SLSQP"
    )
    return float(result.x[0]), float(result.x[1]), float(result.x[2])


# ---------------------------------------------------------------------------
# EWMAModel
# ---------------------------------------------------------------------------

class EWMAModel:
    """EWMA model: V_t = (1 - alpha) V_{t-1} + alpha S_{t-1}.

    The scalar smoothing parameter alpha is estimated by maximising the
    composite log-likelihood over contiguous bivariate return pairs.
    """

    def fit(
        self,
        returns: np.ndarray,
        s_series: list,
        lambda_V: Optional[np.ndarray] = None,
    ) -> dict:
        """Estimate the EWMA smoothing parameter by composite likelihood.

        Parameters
        ----------
        returns : np.ndarray
            Daily return matrix, shape (T, N).
        s_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        lambda_V : np.ndarray, optional
            Unconditional covariance target.  Defaults to sample mean of
            *s_series*.

        Returns
        -------
        dict
            Keys: ``alpha`` (float).
        """
        returns = np.asarray(returns, dtype=float)
        alpha = fit_alpha_ewma(returns, s_series)
        return {"alpha": alpha}

    def one_step_ahead(
        self,
        V_prev: np.ndarray,
        S_prev: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Compute V_t from V_{t-1} and S_{t-1}.

        Parameters
        ----------
        V_prev : np.ndarray
            Previous forecast matrix, shape (N, N).
        S_prev : np.ndarray
            Previous realized covariance matrix, shape (N, N).
        params : dict
            Parameter dict with key ``alpha``.

        Returns
        -------
        np.ndarray
            Next-step forecast matrix, shape (N, N), symmetric and PD.
        """
        alpha = params["alpha"]
        V_t = (1.0 - alpha) * V_prev + alpha * np.asarray(S_prev, dtype=float)
        V_t = (V_t + V_t.T) / 2.0
        if not is_pd(V_t):
            logger.warning("EWMAModel.one_step_ahead: V_t not PD; projecting.")
            V_t = nearest_pd(V_t)
        return V_t

    def predict_series(
        self,
        returns: np.ndarray,
        s_series: list,
        params: dict,
    ) -> list:
        """Generate the full sequence of one-step-ahead forecasts.

        V_0 = sample_mean(S) (initialisation).
        V_t = (1 - alpha) V_{t-1} + alpha S_{t-1}  for t ≥ 1.

        Parameters
        ----------
        returns : np.ndarray
            Daily return matrix, shape (T, N).  Used only to determine T.
        s_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        params : dict
            Parameter dict with key ``alpha``.

        Returns
        -------
        list of np.ndarray
            Length-T list of (N, N) one-step-ahead forecast matrices.
        """
        returns = np.asarray(returns, dtype=float)
        T = returns.shape[0]
        lambda_V = np.mean(np.stack(s_series, axis=0), axis=0)
        return _ewma_series(s_series, params["alpha"], lambda_V)


# ---------------------------------------------------------------------------
# EWMAQModel
# ---------------------------------------------------------------------------

class EWMAQModel:
    """EWMAQ model with time-varying smoothing driven by measurement error.

    vech(V_t) = (1 - alpha_t) o vech(V_{t-1}) + alpha_t o vech(S_{t-1})
    alpha_t = alpha * 1_K + alpha_Q * pi_{t-1}

    Expected: alpha_Q < 0 (high measurement error → less weight on S_{t-1}).
    """

    def fit(
        self,
        returns: np.ndarray,
        s_series: list,
        pi_series: np.ndarray,
    ) -> dict:
        """Estimate EWMAQ parameters by composite likelihood.

        Parameters
        ----------
        returns : np.ndarray
            Daily return matrix, shape (T, N).
        s_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        pi_series : np.ndarray
            Demeaned measurement-error vectors, shape (T, K).

        Returns
        -------
        dict
            Keys: ``alpha`` (float), ``alpha_Q`` (float).
        """
        returns = np.asarray(returns, dtype=float)
        pi_series = np.asarray(pi_series, dtype=float)
        alpha, alpha_Q = fit_params_ewmaq(returns, s_series, pi_series)
        return {"alpha": alpha, "alpha_Q": alpha_Q}

    def predict_series(
        self,
        returns: np.ndarray,
        s_series: list,
        pi_series: np.ndarray,
        params: dict,
    ) -> list:
        """Generate the full sequence of EWMAQ one-step-ahead forecasts.

        Parameters
        ----------
        returns : np.ndarray
            Daily return matrix, shape (T, N).  Used only to determine T.
        s_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        pi_series : np.ndarray
            Demeaned measurement-error vectors, shape (T, K).
        params : dict
            Parameter dict with keys ``alpha``, ``alpha_Q``.

        Returns
        -------
        list of np.ndarray
            Length-T list of (N, N) one-step-ahead forecast matrices.
        """
        returns = np.asarray(returns, dtype=float)
        pi_series = np.asarray(pi_series, dtype=float)
        lambda_V = np.mean(np.stack(s_series, axis=0), axis=0)
        return _ewmaq_series(
            s_series, pi_series, params["alpha"], params["alpha_Q"], lambda_V
        )


# ---------------------------------------------------------------------------
# HEAVYModel
# ---------------------------------------------------------------------------

class HEAVYModel:
    """HEAVY model with covariance targeting.

    V_t = (1 - b - a) lambda_V + b V_{t-1} + a S_{t-1}

    where lambda_V = sample_mean(S) enforces covariance targeting.
    Stationarity requires a + b < 1 with a, b > 0.
    """

    def fit(
        self,
        returns: np.ndarray,
        s_series: list,
    ) -> dict:
        """Estimate HEAVY parameters (a, b) by composite likelihood.

        Parameters
        ----------
        returns : np.ndarray
            Daily return matrix, shape (T, N).
        s_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.

        Returns
        -------
        dict
            Keys: ``a`` (float), ``b`` (float), ``lambda_V`` (np.ndarray of
            shape (N, N)).
        """
        returns = np.asarray(returns, dtype=float)
        lambda_V = np.mean(np.stack(s_series, axis=0), axis=0)
        a, b = fit_params_heavy(returns, s_series)
        return {"a": a, "b": b, "lambda_V": lambda_V}

    def predict_series(
        self,
        returns: np.ndarray,
        s_series: list,
        params: dict,
    ) -> list:
        """Generate the full sequence of HEAVY one-step-ahead forecasts.

        V_0 = lambda_V (initialisation).
        V_t = (1 - b - a) lambda_V + b V_{t-1} + a S_{t-1}  for t ≥ 1.

        Parameters
        ----------
        returns : np.ndarray
            Daily return matrix, shape (T, N).  Used only to determine T.
        s_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        params : dict
            Parameter dict with keys ``a``, ``b``, ``lambda_V``.

        Returns
        -------
        list of np.ndarray
            Length-T list of (N, N) one-step-ahead forecast matrices.
        """
        returns = np.asarray(returns, dtype=float)
        return _heavy_series(s_series, params["a"], params["b"], params["lambda_V"])


# ---------------------------------------------------------------------------
# HEAVYQModel
# ---------------------------------------------------------------------------

class HEAVYQModel:
    """HEAVYQ model: element-wise time-varying HEAVY driven by measurement error.

    vech(V_t) = (1 - b - a_t) o lambda_v + b * vech(V_{t-1}) + a_t o vech(S_{t-1})
    a_t = a * 1_K + a_Q * pi_{t-1}

    Expected: a_Q < 0 (high measurement error → less weight on S_{t-1}).
    Stationarity requires a + b < 1 at the mean level (since pi is demeaned).
    """

    def fit(
        self,
        returns: np.ndarray,
        s_series: list,
        pi_series: np.ndarray,
    ) -> dict:
        """Estimate HEAVYQ parameters (a, a_Q, b) by composite likelihood.

        Parameters
        ----------
        returns : np.ndarray
            Daily return matrix, shape (T, N).
        s_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        pi_series : np.ndarray
            Demeaned measurement-error vectors, shape (T, K).

        Returns
        -------
        dict
            Keys: ``a`` (float), ``a_Q`` (float), ``b`` (float),
            ``lambda_V`` (np.ndarray of shape (N, N)).
        """
        returns = np.asarray(returns, dtype=float)
        pi_series = np.asarray(pi_series, dtype=float)
        lambda_V = np.mean(np.stack(s_series, axis=0), axis=0)
        a, a_Q, b = fit_params_heavyq(returns, s_series, pi_series)
        return {"a": a, "a_Q": a_Q, "b": b, "lambda_V": lambda_V}

    def predict_series(
        self,
        returns: np.ndarray,
        s_series: list,
        pi_series: np.ndarray,
        params: dict,
    ) -> list:
        """Generate the full sequence of HEAVYQ one-step-ahead forecasts.

        Parameters
        ----------
        returns : np.ndarray
            Daily return matrix, shape (T, N).  Used only to determine T.
        s_series : list of np.ndarray
            Length-T list of (N, N) realized covariance matrices.
        pi_series : np.ndarray
            Demeaned measurement-error vectors, shape (T, K).
        params : dict
            Parameter dict with keys ``a``, ``a_Q``, ``b``, ``lambda_V``.

        Returns
        -------
        list of np.ndarray
            Length-T list of (N, N) one-step-ahead forecast matrices.
        """
        returns = np.asarray(returns, dtype=float)
        pi_series = np.asarray(pi_series, dtype=float)
        return _heavyq_series(
            s_series,
            pi_series,
            params["a"],
            params["a_Q"],
            params["b"],
            params["lambda_V"],
        )
