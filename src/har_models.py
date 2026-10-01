"""HAR and HARQ models for vectorized realized covariance forecasting.

Implements the scalar-coefficient Heterogeneous Autoregressive (HAR) model
and its measurement-error-corrected variant HARQ on the vech-stacked realized
covariance series.  Estimation uses pooled OLS (statsmodels) so that a single
set of slope parameters is shared across all K = N*(N+1)/2 covariance elements,
while each element retains its own intercept.

References
----------
Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
    Journal of Financial Econometrics, 7(2), 174-196.
Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the errors:
    A simple approach for improved volatility forecasting.
    Journal of Econometrics, 192(1), 1-18.
"""

import logging
from typing import Optional

import numpy as np
import statsmodels.api as sm

from src.utils import nearest_pd, is_pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lag-average helper
# ---------------------------------------------------------------------------

def compute_lag_averages(s_series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-step, five-day and twenty-two-day lagged averages.

    For each time index t in {22, ..., T-1}, returns:
    - s_lag1[t-22]   = s_{t-1}
    - s_avg5[t-22]   = mean(s_{t-5}, ..., s_{t-1})
    - s_avg22[t-22]  = mean(s_{t-22}, ..., s_{t-1})

    All output arrays have shape (T-22, K).

    Parameters
    ----------
    s_series : np.ndarray
        Vech-stacked realized covariance series, shape (T, K).  Requires T ≥ 23.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (s_lag1, s_avg5, s_avg22), each of shape (T-22, K).

    Raises
    ------
    ValueError
        If *s_series* is not 2-D or has fewer than 23 rows.
    """
    s_series = np.asarray(s_series, dtype=float)
    if s_series.ndim != 2:
        raise ValueError(
            f"compute_lag_averages: expected 2-D array; got shape {s_series.shape}."
        )
    T, K = s_series.shape
    if T < 23:
        raise ValueError(
            f"compute_lag_averages: need T ≥ 23 (got T={T}) to form 22-lag average."
        )

    T_eff = T - 22                             # valid prediction time steps
    s_lag1 = s_series[21 : T - 1]             # s_{t-1}, shape (T_eff, K)

    # Five-day average: mean of rows [t-5, t-4, t-3, t-2, t-1]
    # For t = 22: rows 17..21; for t = T-1: rows T-6..T-2
    s_avg5 = np.empty((T_eff, K), dtype=float)
    s_avg22 = np.empty((T_eff, K), dtype=float)
    for i, t in enumerate(range(22, T)):
        s_avg5[i] = s_series[t - 5 : t].mean(axis=0)
        s_avg22[i] = s_series[t - 22 : t].mean(axis=0)

    return s_lag1, s_avg5, s_avg22


# ---------------------------------------------------------------------------
# HAR regressor builders
# ---------------------------------------------------------------------------

def build_har_regressors(s_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build pooled (X, y) arrays for scalar-coefficient HAR OLS.

    For each time index t ∈ {22, ..., T-1} and each vech element k ∈ {0, ..., K-1},
    one row is added to the stacked system:

        y_{t,k}  = s_{t,k}
        x_{t,k}  = [1, s_{t-1,k}, s_{t-5|t-1,k}, s_{t-22|t-1,k}]

    All K equations are stacked into a single (T_eff·K, 4) design matrix so
    that OLS recovers the shared scalar slopes (theta_1, theta_2, theta_3)
    alongside a common intercept.  Element-specific intercepts are back-
    calculated in :meth:`HARModel.fit`.

    Parameters
    ----------
    s_series : np.ndarray
        Vech-stacked realized covariance series, shape (T, K).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) where X has shape (T_eff*K, 4) and y has shape (T_eff*K,).

    Raises
    ------
    ValueError
        Propagated from :func:`compute_lag_averages` if T < 23.
    """
    s_series = np.asarray(s_series, dtype=float)
    T, K = s_series.shape
    T_eff = T - 22

    s_lag1, s_avg5, s_avg22 = compute_lag_averages(s_series)

    y = s_series[22:].reshape(-1)                    # (T_eff*K,)
    ones = np.ones(T_eff * K, dtype=float)
    x1 = s_lag1.reshape(-1)
    x2 = s_avg5.reshape(-1)
    x3 = s_avg22.reshape(-1)

    X = np.column_stack([ones, x1, x2, x3])          # (T_eff*K, 4)
    return X, y


def build_harq_regressors(
    s_series: np.ndarray, pi_series: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build pooled (X, y) arrays for scalar-coefficient HARQ OLS.

    Extends :func:`build_har_regressors` with the additional regressor
    (pi_{t-1} ⊙ s_{t-1}) that captures the interaction between measurement
    error magnitude and the level of the lagged covariance.

    Parameters
    ----------
    s_series : np.ndarray
        Vech-stacked realized covariance series, shape (T, K).
    pi_series : np.ndarray
        Demeaned measurement-error standard deviations, shape (T, K).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) where X has shape (T_eff*K, 5) and y has shape (T_eff*K,).

    Raises
    ------
    ValueError
        If *pi_series* does not match the shape of *s_series*.
    """
    s_series = np.asarray(s_series, dtype=float)
    pi_series = np.asarray(pi_series, dtype=float)
    if s_series.shape != pi_series.shape:
        raise ValueError(
            f"build_harq_regressors: s_series shape {s_series.shape} does not "
            f"match pi_series shape {pi_series.shape}."
        )

    T, K = s_series.shape
    T_eff = T - 22

    s_lag1, s_avg5, s_avg22 = compute_lag_averages(s_series)
    pi_lag1 = pi_series[21 : T - 1]                   # pi_{t-1}, shape (T_eff, K)

    y = s_series[22:].reshape(-1)                      # (T_eff*K,)
    ones = np.ones(T_eff * K, dtype=float)
    x1 = s_lag1.reshape(-1)
    pi_x1 = (pi_lag1 * s_lag1).reshape(-1)            # pi_{t-1} ⊙ s_{t-1}
    x2 = s_avg5.reshape(-1)
    x3 = s_avg22.reshape(-1)

    X = np.column_stack([ones, x1, pi_x1, x2, x3])    # (T_eff*K, 5)
    return X, y


# ---------------------------------------------------------------------------
# HARModel
# ---------------------------------------------------------------------------

class HARModel:
    """Scalar-coefficient HAR model on vech-stacked covariance.

    Model specification (per element k)::

        s_{t,k} = theta_{0,k} + theta_1 s_{t-1,k}
                  + theta_2 s_{t-5|t-1,k} + theta_3 s_{t-22|t-1,k} + eps_{t,k}

    The three slope parameters (theta_1, theta_2, theta_3) are shared across
    all K elements (pooled OLS).  Each element has its own intercept theta_{0,k},
    recovered from the element-wise mean condition after pooled estimation.
    """

    # ------------------------------------------------------------------
    def fit(
        self, s_series: np.ndarray, pi_series: Optional[np.ndarray] = None
    ) -> dict:
        """Estimate HAR parameters by pooled OLS.

        Parameters
        ----------
        s_series : np.ndarray
            Vech-stacked realized covariance series, shape (T, K).
        pi_series : np.ndarray, optional
            Ignored for HAR (present for API symmetry with HARQ).

        Returns
        -------
        dict
            Keys: ``theta_0`` (K,), ``theta_1``, ``theta_2``, ``theta_3``
            (all scalars), ``se_theta_1``, ``se_theta_2``, ``se_theta_3``
            (standard errors of the shared slope estimates).

        Raises
        ------
        ValueError
            If *s_series* has fewer than 23 rows.
        """
        s_series = np.asarray(s_series, dtype=float)
        T, K = s_series.shape

        X, y = build_har_regressors(s_series)
        result = sm.OLS(y, X).fit()

        # Shared slope estimates from pooled OLS
        theta_1 = float(result.params[1])
        theta_2 = float(result.params[2])
        theta_3 = float(result.params[3])

        # Element-specific intercepts via mean condition:
        # theta_0[k] = mean(y_k) - theta_1*mean(x1_k) - theta_2*mean(x2_k) - theta_3*mean(x3_k)
        s_lag1, s_avg5, s_avg22 = compute_lag_averages(s_series)
        y_mean = s_series[22:].mean(axis=0)            # (K,)
        theta_0 = (
            y_mean
            - theta_1 * s_lag1.mean(axis=0)
            - theta_2 * s_avg5.mean(axis=0)
            - theta_3 * s_avg22.mean(axis=0)
        )                                               # (K,)

        return {
            "theta_0": theta_0,
            "theta_1": theta_1,
            "theta_2": theta_2,
            "theta_3": theta_3,
            "se_theta_1": float(result.bse[1]),
            "se_theta_2": float(result.bse[2]),
            "se_theta_3": float(result.bse[3]),
        }

    # ------------------------------------------------------------------
    def forecast_from_params(
        self,
        s_last: np.ndarray,
        s_avg5: np.ndarray,
        s_avg22: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Produce a one-step forecast given pre-computed lagged values.

        Parameters
        ----------
        s_last : np.ndarray
            s_{t-1}, shape (K,).
        s_avg5 : np.ndarray
            Five-day average s_{t-5|t-1}, shape (K,).
        s_avg22 : np.ndarray
            Twenty-two-day average s_{t-22|t-1}, shape (K,).
        params : dict
            Parameter dict as returned by :meth:`fit`.

        Returns
        -------
        np.ndarray
            One-step-ahead vech forecast, shape (K,).
        """
        return (
            params["theta_0"]
            + params["theta_1"] * s_last
            + params["theta_2"] * s_avg5
            + params["theta_3"] * s_avg22
        )

    # ------------------------------------------------------------------
    def predict(
        self, s_series: np.ndarray, params: dict, t: int
    ) -> np.ndarray:
        """One-step-ahead forecast for day *t* using pre-estimated params.

        Parameters
        ----------
        s_series : np.ndarray
            Full vech-stacked series, shape (T, K).  Must satisfy t ≥ 22.
        params : dict
            Parameter dict as returned by :meth:`fit`.
        t : int
            Forecast target index (≥ 22).  Uses s_series[0..t-1].

        Returns
        -------
        np.ndarray
            Vech forecast of shape (K,).

        Raises
        ------
        ValueError
            If t < 22.
        """
        if t < 22:
            raise ValueError(f"HARModel.predict: need t ≥ 22; got t={t}.")
        s_series = np.asarray(s_series, dtype=float)
        s_last = s_series[t - 1]
        s_avg5 = s_series[t - 5 : t].mean(axis=0)
        s_avg22 = s_series[t - 22 : t].mean(axis=0)
        return self.forecast_from_params(s_last, s_avg5, s_avg22, params)


# ---------------------------------------------------------------------------
# HARQModel
# ---------------------------------------------------------------------------

class HARQModel:
    """Scalar-coefficient HARQ model on vech-stacked covariance.

    Extends HAR by allowing the lagged-level coefficient to vary with the
    measurement-error magnitude::

        theta_{1,t} = theta_1 + theta_{1Q} * pi_{t-1,k}

    Substituting into the HAR equation for element k::

        s_{t,k} = theta_{0,k} + theta_1 s_{t-1,k}
                  + theta_{1Q} (pi_{t-1,k} * s_{t-1,k})
                  + theta_2 s_{t-5|t-1,k} + theta_3 s_{t-22|t-1,k} + eps_{t,k}

    The added regressor (pi_{t-1,k} * s_{t-1,k}) is estimated jointly with
    the HAR slopes via pooled OLS.  Expected sign: theta_{1Q} < 0.
    """

    # ------------------------------------------------------------------
    def fit(self, s_series: np.ndarray, pi_series: np.ndarray) -> dict:
        """Estimate HARQ parameters by pooled OLS.

        Parameters
        ----------
        s_series : np.ndarray
            Vech-stacked realized covariance series, shape (T, K).
        pi_series : np.ndarray
            Demeaned measurement-error standard deviations, shape (T, K).

        Returns
        -------
        dict
            Keys: ``theta_0`` (K,), ``theta_1``, ``theta_1Q``, ``theta_2``,
            ``theta_3`` (scalars), ``se_theta_1``, ``se_theta_1Q``,
            ``se_theta_2``, ``se_theta_3``.

        Raises
        ------
        ValueError
            If shapes of *s_series* and *pi_series* do not match.
        """
        s_series = np.asarray(s_series, dtype=float)
        pi_series = np.asarray(pi_series, dtype=float)

        X, y = build_harq_regressors(s_series, pi_series)
        result = sm.OLS(y, X).fit()

        # Columns: [const, s_{t-1}, pi_{t-1}*s_{t-1}, s_avg5, s_avg22]
        theta_1 = float(result.params[1])
        theta_1Q = float(result.params[2])
        theta_2 = float(result.params[3])
        theta_3 = float(result.params[4])

        s_lag1, s_avg5, s_avg22 = compute_lag_averages(s_series)
        T, K = s_series.shape
        pi_lag1 = pi_series[21 : T - 1]               # (T_eff, K)

        y_mean = s_series[22:].mean(axis=0)
        theta_0 = (
            y_mean
            - theta_1 * s_lag1.mean(axis=0)
            - theta_1Q * (pi_lag1 * s_lag1).mean(axis=0)
            - theta_2 * s_avg5.mean(axis=0)
            - theta_3 * s_avg22.mean(axis=0)
        )                                               # (K,)

        return {
            "theta_0": theta_0,
            "theta_1": theta_1,
            "theta_1Q": theta_1Q,
            "theta_2": theta_2,
            "theta_3": theta_3,
            "se_theta_1": float(result.bse[1]),
            "se_theta_1Q": float(result.bse[2]),
            "se_theta_2": float(result.bse[3]),
            "se_theta_3": float(result.bse[4]),
        }

    # ------------------------------------------------------------------
    def forecast_from_params(
        self,
        s_last: np.ndarray,
        s_avg5: np.ndarray,
        s_avg22: np.ndarray,
        pi_last: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Produce a one-step HARQ forecast given pre-computed lagged values.

        Parameters
        ----------
        s_last : np.ndarray
            s_{t-1}, shape (K,).
        s_avg5 : np.ndarray
            Five-day average s_{t-5|t-1}, shape (K,).
        s_avg22 : np.ndarray
            Twenty-two-day average s_{t-22|t-1}, shape (K,).
        pi_last : np.ndarray
            Demeaned measurement-error vector pi_{t-1}, shape (K,).
        params : dict
            Parameter dict as returned by :meth:`fit`.

        Returns
        -------
        np.ndarray
            One-step-ahead vech forecast, shape (K,).
        """
        return (
            params["theta_0"]
            + params["theta_1"] * s_last
            + params["theta_1Q"] * pi_last * s_last
            + params["theta_2"] * s_avg5
            + params["theta_3"] * s_avg22
        )

    # ------------------------------------------------------------------
    def predict(
        self,
        s_series: np.ndarray,
        pi_series: np.ndarray,
        params: dict,
        t: int,
    ) -> np.ndarray:
        """One-step-ahead HARQ forecast for day *t* using pre-estimated params.

        Parameters
        ----------
        s_series : np.ndarray
            Full vech-stacked series, shape (T, K).
        pi_series : np.ndarray
            Demeaned pi series, shape (T, K).
        params : dict
            Parameter dict as returned by :meth:`fit`.
        t : int
            Forecast target index (≥ 22).

        Returns
        -------
        np.ndarray
            Vech forecast of shape (K,).

        Raises
        ------
        ValueError
            If t < 22.
        """
        if t < 22:
            raise ValueError(f"HARQModel.predict: need t ≥ 22; got t={t}.")
        s_series = np.asarray(s_series, dtype=float)
        pi_series = np.asarray(pi_series, dtype=float)
        s_last = s_series[t - 1]
        s_avg5_val = s_series[t - 5 : t].mean(axis=0)
        s_avg22_val = s_series[t - 22 : t].mean(axis=0)
        pi_last = pi_series[t - 1]
        return self.forecast_from_params(s_last, s_avg5_val, s_avg22_val, pi_last, params)
