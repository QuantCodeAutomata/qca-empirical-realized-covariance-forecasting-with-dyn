"""Portfolio construction and evaluation for realized covariance forecasting.

Provides GMV portfolio formation (unconstrained and constrained), turnover
and net-return computation, performance statistics, utility-based evaluation
measures, and the :class:`PortfolioEvaluator` orchestrator.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.utils import safe_inv, nearest_pd, is_pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def gmv_weights_unconstrained(H: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights without short-sale constraints.

    .. math::

        w = \\frac{H^{-1} \\mathbf{1}}{\\mathbf{1}^{\\prime} H^{-1} \\mathbf{1}}

    Parameters
    ----------
    H : np.ndarray
        Covariance matrix of shape (N, N).  Should be positive definite;
        if not, :func:`~src.utils.safe_inv` projects it internally.

    Returns
    -------
    np.ndarray
        Weight vector of shape (N,) summing to one.

    Raises
    ------
    ValueError
        If *H* is not a 2-D square array.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"gmv_weights_unconstrained: H must be square 2-D; got {H.shape}.")
    N = H.shape[0]
    ones = np.ones(N, dtype=float)
    H_inv = safe_inv(H)
    numerator = H_inv @ ones
    denominator = ones @ numerator
    if abs(denominator) < 1e-14:
        logger.warning(
            "gmv_weights_unconstrained: 1' H^{-1} 1 ≈ 0; returning equal weights."
        )
        return np.full(N, 1.0 / N)
    return numerator / denominator


def gmv_weights_constrained(H: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights with non-negativity constraint.

    Solves

    .. math::

        \\min_w \\; w^{\\prime} H w \\quad \\text{s.t.} \\quad
        \\mathbf{1}^{\\prime} w = 1, \\; w \\geq 0.

    Tries ``cvxpy`` first for speed; falls back to ``scipy.optimize.minimize``
    with the SLSQP method.

    Parameters
    ----------
    H : np.ndarray
        Covariance matrix of shape (N, N).

    Returns
    -------
    np.ndarray
        Weight vector of shape (N,) with non-negative entries summing to one.

    Raises
    ------
    ValueError
        If *H* is not a 2-D square array.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"gmv_weights_constrained: H must be square 2-D; got {H.shape}.")
    N = H.shape[0]
    if not is_pd(H):
        H = nearest_pd(H)

    # Try cvxpy first
    try:
        import cvxpy as cp  # type: ignore[import]
        w = cp.Variable(N)
        obj = cp.Minimize(cp.quad_form(w, H))
        constraints = [cp.sum(w) == 1.0, w >= 0.0]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            wopt = np.array(w.value, dtype=float)
            wopt = np.maximum(wopt, 0.0)
            wopt /= wopt.sum()
            return wopt
    except Exception:
        pass

    # Fallback: scipy SLSQP
    w0 = np.full(N, 1.0 / N)
    result = optimize.minimize(
        fun=lambda w: float(w @ H @ w),
        x0=w0,
        method="SLSQP",
        jac=lambda w: 2.0 * H @ w,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        bounds=[(0.0, None)] * N,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    wopt = np.maximum(result.x, 0.0)
    wopt /= wopt.sum()
    return wopt


def tracking_error_weights(H: np.ndarray, n_assets: int) -> np.ndarray:
    """Minimum tracking-error weights relative to SPY.

    Solves the GMV problem on the excess-return covariance matrix *H*,
    which is assumed to be the covariance of ``(r - r_SPY)`` for the
    *n_assets* assets already encoded in the first *n_assets* rows/columns
    of *H*.

    .. math::

        \\min_w \\; w^{\\prime} H_{\\text{excess}} w \\quad \\text{s.t.} \\quad
        \\mathbf{1}^{\\prime} w = 1

    Parameters
    ----------
    H : np.ndarray
        Excess-return covariance matrix of shape (N, N), already constructed
        as ``Cov(r - r_SPY)``.
    n_assets : int
        Number of risky assets (informational; must equal ``H.shape[0]``).

    Returns
    -------
    np.ndarray
        Weight vector of shape (N,) summing to one (unconstrained long-short).

    Raises
    ------
    ValueError
        If *H* is not square or *n_assets* does not match its dimension.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"tracking_error_weights: H must be square 2-D; got {H.shape}.")
    if H.shape[0] != n_assets:
        raise ValueError(
            f"tracking_error_weights: H dimension {H.shape[0]} != n_assets {n_assets}."
        )
    return gmv_weights_unconstrained(H)


# ---------------------------------------------------------------------------
# Turnover and net returns
# ---------------------------------------------------------------------------

def compute_turnover(
    weights_series: np.ndarray,
    returns_series: np.ndarray,
) -> np.ndarray:
    """Compute portfolio turnover for each rebalancing date.

    .. math::

        \\text{TO}_t = \\sum_n \\left| w_{t+1}^n -
        \\frac{w_t^n (1 + r_t^n)}{1 + w_t^{\\prime} r_t} \\right|

    where ``w_{t+1}`` are the target weights for the next period and
    ``w_t^n (1+r_t^n) / (1 + r_{p,t})`` are the drifted weights after
    period *t*'s returns.

    Parameters
    ----------
    weights_series : np.ndarray
        Weight matrix of shape (T, N).
    returns_series : np.ndarray
        Return matrix of shape (T, N).

    Returns
    -------
    np.ndarray
        Turnover array of shape (T-1,), one value per rebalancing event.

    Raises
    ------
    ValueError
        If *weights_series* or *returns_series* are not 2-D or have
        incompatible shapes.
    """
    weights_series = np.asarray(weights_series, dtype=float)
    returns_series = np.asarray(returns_series, dtype=float)
    if weights_series.ndim != 2 or returns_series.ndim != 2:
        raise ValueError("compute_turnover: inputs must be 2-D arrays.")
    if weights_series.shape != returns_series.shape:
        raise ValueError(
            f"compute_turnover: shape mismatch {weights_series.shape} vs "
            f"{returns_series.shape}."
        )
    T, N = weights_series.shape
    if T < 2:
        raise ValueError(f"compute_turnover: need T ≥ 2; got T={T}.")

    turnover = np.empty(T - 1, dtype=float)
    for t in range(T - 1):
        w_t = weights_series[t]
        r_t = returns_series[t]
        port_ret = float(w_t @ r_t)
        denom = 1.0 + port_ret
        if abs(denom) < 1e-14:
            denom = 1e-14
        w_drift = w_t * (1.0 + r_t) / denom   # drifted weights
        w_next = weights_series[t + 1]
        turnover[t] = float(np.abs(w_next - w_drift).sum())

    return turnover


def portfolio_net_return(
    weights_series: np.ndarray,
    returns_series: np.ndarray,
    c: float = 0.0,
) -> np.ndarray:
    """Compute portfolio net returns after transaction costs.

    .. math::

        r_{p,t} = w_t^{\\prime} r_t - c \\cdot \\text{TO}_t

    where ``TO_t`` is computed between ``w_t`` and ``w_{t+1}``.  For the last
    period there is no rebalancing, so ``TO_{T-1} = 0``.

    Parameters
    ----------
    weights_series : np.ndarray
        Weight matrix of shape (T, N).
    returns_series : np.ndarray
        Return matrix of shape (T, N).
    c : float, optional
        One-way transaction cost rate (default: 0.0, no costs).

    Returns
    -------
    np.ndarray
        Net return array of shape (T,).

    Raises
    ------
    ValueError
        If inputs are not 2-D, have mismatched shapes, or *c* is negative.
    """
    weights_series = np.asarray(weights_series, dtype=float)
    returns_series = np.asarray(returns_series, dtype=float)
    if weights_series.ndim != 2 or returns_series.ndim != 2:
        raise ValueError("portfolio_net_return: inputs must be 2-D arrays.")
    if weights_series.shape != returns_series.shape:
        raise ValueError(
            f"portfolio_net_return: shape mismatch {weights_series.shape} vs "
            f"{returns_series.shape}."
        )
    if c < 0.0:
        raise ValueError(f"portfolio_net_return: cost c must be ≥ 0; got {c}.")
    T, N = weights_series.shape

    gross_returns = (weights_series * returns_series).sum(axis=1)   # (T,)

    if c == 0.0 or T < 2:
        return gross_returns

    to = compute_turnover(weights_series, returns_series)   # (T-1,)
    # Pad TO with 0 for the last period (no subsequent rebalancing)
    to_full = np.concatenate([to, [0.0]])
    return gross_returns - c * to_full


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def compute_portfolio_stats(
    returns: np.ndarray,
    trading_days: int = 252,
) -> dict:
    """Compute annualized portfolio performance statistics.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of portfolio (net) returns, one per trading day.
    trading_days : int, optional
        Number of trading days per year (default: 252).

    Returns
    -------
    dict
        Dictionary with keys:
        ``'ann_return'``, ``'ann_vol'``, ``'sharpe'``,
        ``'mean_return'``, ``'std_return'``.

    Raises
    ------
    ValueError
        If *returns* is not 1-D, or has fewer than 2 observations.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError(f"compute_portfolio_stats: returns must be 1-D; got {returns.ndim}-D.")
    if len(returns) < 2:
        raise ValueError(f"compute_portfolio_stats: need ≥ 2 observations; got {len(returns)}.")
    if trading_days < 1:
        raise ValueError(f"compute_portfolio_stats: trading_days must be ≥ 1; got {trading_days}.")

    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    ann_return = mu * trading_days
    ann_vol = sigma * np.sqrt(trading_days)
    sharpe = (ann_return / ann_vol) if ann_vol > 1e-14 else 0.0

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mean_return": mu,
        "std_return": sigma,
    }


# ---------------------------------------------------------------------------
# Utility-based evaluation
# ---------------------------------------------------------------------------

def quadratic_utility(returns: np.ndarray, gamma: float) -> float:
    """Mean quadratic utility under power-utility approximation.

    .. math::

        U = \\operatorname{E}_t \\left[
            (1 + r_{p,t}) - \\frac{\\gamma}{2(1+\\gamma)}(1 + r_{p,t})^2
        \\right]

    Parameters
    ----------
    returns : np.ndarray
        1-D array of net portfolio returns.
    gamma : float
        Relative risk-aversion coefficient (must be > 0).

    Returns
    -------
    float
        Mean utility over the evaluation period.

    Raises
    ------
    ValueError
        If *returns* is not 1-D or *gamma* ≤ 0.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError(f"quadratic_utility: returns must be 1-D; got {returns.ndim}-D.")
    if gamma <= 0.0:
        raise ValueError(f"quadratic_utility: gamma must be > 0; got {gamma}.")

    c = gamma / (2.0 * (1.0 + gamma))
    gross = 1.0 + returns
    return float(np.mean(gross - c * gross ** 2))


def certainty_equivalent_gain(
    returns_k: np.ndarray,
    returns_l: np.ndarray,
    gamma: float,
    trading_days: int = 252,
) -> float:
    """Annualized certainty-equivalent return gain of model *k* over model *l*.

    Finds the daily fee :math:`\\Delta_\\gamma` (in basis points, annualized)
    that equates the expected utility of model *k* to the fee-adjusted utility
    of model *l*:

    .. math::

        \\sum_t U(r_t^k, \\gamma) = \\sum_t U(r_t^l - \\Delta_\\gamma, \\gamma)

    A positive value means model *k* outperforms model *l*.

    Solves the implied quadratic equation for :math:`\\Delta_\\gamma`.

    Parameters
    ----------
    returns_k : np.ndarray
        Net returns of model *k*, shape (T,).
    returns_l : np.ndarray
        Net returns of model *l* (benchmark), shape (T,).
    gamma : float
        Relative risk-aversion coefficient (must be > 0).
    trading_days : int, optional
        Trading days per year for annualization (default: 252).

    Returns
    -------
    float
        Annualized certainty-equivalent gain in **basis points**.
        Positive → model *k* is better.

    Raises
    ------
    ValueError
        If arrays are not 1-D, have different lengths, or *gamma* ≤ 0.
    """
    returns_k = np.asarray(returns_k, dtype=float)
    returns_l = np.asarray(returns_l, dtype=float)
    if returns_k.ndim != 1 or returns_l.ndim != 1:
        raise ValueError("certainty_equivalent_gain: inputs must be 1-D.")
    if len(returns_k) != len(returns_l):
        raise ValueError(
            f"certainty_equivalent_gain: length mismatch {len(returns_k)} vs "
            f"{len(returns_l)}."
        )
    if gamma <= 0.0:
        raise ValueError(f"certainty_equivalent_gain: gamma must be > 0; got {gamma}.")

    c = gamma / (2.0 * (1.0 + gamma))

    U_k = quadratic_utility(returns_k, gamma)
    U_l = quadratic_utility(returns_l, gamma)

    # Quadratic in Delta:
    # c * Delta^2 - Delta * (1 - 2*c*mu_l) + (U_l - U_k) = 0
    # where mu_l = mean(1 + r_l)
    mu_l = float((1.0 + returns_l).mean())
    a_coef = c
    b_coef = -(1.0 - 2.0 * c * mu_l)
    c_coef = U_l - U_k

    discriminant = b_coef ** 2 - 4.0 * a_coef * c_coef
    if discriminant < 0.0:
        # Numerically degenerate; fall back to linear approximation (ignore Delta^2)
        if abs(1.0 - 2.0 * c * mu_l) < 1e-14:
            delta_daily = 0.0
        else:
            delta_daily = c_coef / (1.0 - 2.0 * c * mu_l)
    else:
        # Take root closer to 0 (small fee)
        sqrt_disc = np.sqrt(discriminant)
        root1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        root2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
        delta_daily = root1 if abs(root1) < abs(root2) else root2

    # Annualize and convert to basis points
    delta_ann_bps = delta_daily * trading_days * 10_000.0
    return float(delta_ann_bps)


# ---------------------------------------------------------------------------
# White's Reality Check
# ---------------------------------------------------------------------------

def white_reality_check(
    utility_gains: np.ndarray,
    B: int = 999,
    block_size: int = 22,
    seed: int = 42,
) -> float:
    """White's (2000) Reality Check p-value via stationary bootstrap.

    Tests the null hypothesis that the mean utility gain of the best model
    relative to a benchmark is non-positive:

    .. math::

        H_0 : \\operatorname{E}[\\Delta u_t] \\leq 0

    Uses stationary bootstrap to approximate the null distribution of the
    sample mean under H₀ (centring the gains to impose H₀).

    Parameters
    ----------
    utility_gains : np.ndarray
        1-D array of shape (T,) containing daily utility differences.
    B : int, optional
        Number of bootstrap replications (default: 999).
    block_size : int, optional
        Expected block length for the stationary bootstrap (default: 22).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    float
        Bootstrap p-value for :math:`H_0 : \\operatorname{E}[\\Delta u_t] \\leq 0`.

    Raises
    ------
    ValueError
        If *utility_gains* is not 1-D or has fewer than 2 observations.
    """
    utility_gains = np.asarray(utility_gains, dtype=float)
    if utility_gains.ndim != 1:
        raise ValueError(
            f"white_reality_check: utility_gains must be 1-D; got {utility_gains.ndim}-D."
        )
    T = len(utility_gains)
    if T < 2:
        raise ValueError(f"white_reality_check: need T ≥ 2; got T={T}.")
    if B < 1:
        raise ValueError(f"white_reality_check: B must be ≥ 1; got {B}.")

    from src.statistical_tests import stationary_bootstrap_mean

    v_obs = float(utility_gains.mean())

    # Centre under H₀ (impose zero mean) before bootstrapping
    gains_centred = utility_gains - v_obs
    boot_means = stationary_bootstrap_mean(gains_centred, B=B, block_size=block_size, seed=seed)
    p_value = float(np.mean(boot_means >= v_obs))
    return p_value


# ---------------------------------------------------------------------------
# Concentration and short position
# ---------------------------------------------------------------------------

def concentration(weights: np.ndarray) -> float:
    """Portfolio concentration (Herfindahl–Hirschman index in weight space).

    .. math::

        CO = \\sqrt{\\sum_n w_n^2}

    Parameters
    ----------
    weights : np.ndarray
        Weight vector of shape (N,).

    Returns
    -------
    float
        Concentration in [1/sqrt(N), 1] (1 = fully concentrated).
    """
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights ** 2)))


def short_position(weights: np.ndarray) -> float:
    """Total short position in the portfolio.

    .. math::

        SP = \\sum_n w_n \\cdot \\mathbf{1}[w_n < 0]

    Parameters
    ----------
    weights : np.ndarray
        Weight vector of shape (N,).

    Returns
    -------
    float
        Sum of negative weights (non-positive number; 0 for long-only portfolios).
    """
    weights = np.asarray(weights, dtype=float)
    return float(weights[weights < 0.0].sum())


# ---------------------------------------------------------------------------
# PortfolioEvaluator
# ---------------------------------------------------------------------------

class PortfolioEvaluator:
    """Evaluate multiple covariance forecast models across portfolio strategies.

    Parameters
    ----------
    trading_days : int, optional
        Annualization factor (default: 252).

    Examples
    --------
    >>> evaluator = PortfolioEvaluator()
    >>> results = evaluator.run_gmv_daily(H_dict, daily_returns)
    >>> summary = evaluator.summarize_results(results)
    """

    def __init__(self, trading_days: int = 252) -> None:
        self.trading_days = trading_days

    # ------------------------------------------------------------------
    def run_gmv_daily(
        self,
        H_series_dict: dict,
        daily_returns: np.ndarray,
        c_list: list = None,
    ) -> dict:
        """Run unconstrained GMV strategy for every forecast model.

        For each model and each cost level *c* in *c_list*, computes daily
        GMV weights, portfolio net returns, and performance statistics.

        Parameters
        ----------
        H_series_dict : dict
            ``{model_name: list_of_(N,N)_forecast_matrices}`` for T days.
        daily_returns : np.ndarray
            Daily return matrix of shape (T, N).
        c_list : list of float, optional
            Transaction cost rates (default: [0.0, 0.001, 0.002]).

        Returns
        -------
        dict
            Nested dict ``{model_name: {c: {stats_dict, 'weights', 'returns'}}}``.
        """
        if c_list is None:
            c_list = [0.0, 0.001, 0.002]

        daily_returns = np.asarray(daily_returns, dtype=float)
        T, N = daily_returns.shape
        results: dict = {}

        for model_name, H_series in H_series_dict.items():
            if len(H_series) != T:
                raise ValueError(
                    f"run_gmv_daily: H_series for '{model_name}' has {len(H_series)} "
                    f"entries but daily_returns has {T} rows."
                )
            # Compute weights for each day
            weights = np.zeros((T, N), dtype=float)
            for t, H_t in enumerate(H_series):
                try:
                    w = gmv_weights_unconstrained(np.asarray(H_t))
                except Exception as exc:
                    logger.warning(
                        "run_gmv_daily: weight computation failed for model '%s' "
                        "day %d (%s); using equal weights.",
                        model_name, t, exc,
                    )
                    w = np.full(N, 1.0 / N)
                weights[t] = w

            model_results: dict = {}
            for c in c_list:
                net_ret = portfolio_net_return(weights, daily_returns, c=c)
                stats = compute_portfolio_stats(net_ret, trading_days=self.trading_days)
                stats["turnover_mean"] = (
                    float(compute_turnover(weights, daily_returns).mean())
                    if T > 1
                    else 0.0
                )
                stats["concentration_mean"] = float(
                    np.mean([concentration(weights[t]) for t in range(T)])
                )
                stats["short_mean"] = float(
                    np.mean([short_position(weights[t]) for t in range(T)])
                )
                model_results[c] = {
                    **stats,
                    "weights": weights,
                    "returns": net_ret,
                }
            results[model_name] = model_results

        return results

    # ------------------------------------------------------------------
    def run_tracking_error(
        self,
        H_excess_dict: dict,
        excess_returns: np.ndarray,
        spy_returns: np.ndarray,
    ) -> dict:
        """Run minimum tracking-error strategy for every model.

        Parameters
        ----------
        H_excess_dict : dict
            ``{model_name: list_of_(N,N)_excess-return-covariance matrices}``.
        excess_returns : np.ndarray
            Excess returns ``(r - r_SPY)`` of shape (T, N).
        spy_returns : np.ndarray
            SPY returns of shape (T,).

        Returns
        -------
        dict
            ``{model_name: {'te_ann': float, 'weights': np.ndarray,
              'active_returns': np.ndarray}}``.
        """
        excess_returns = np.asarray(excess_returns, dtype=float)
        spy_returns = np.asarray(spy_returns, dtype=float)
        T, N = excess_returns.shape
        results: dict = {}

        for model_name, H_excess_series in H_excess_dict.items():
            if len(H_excess_series) != T:
                raise ValueError(
                    f"run_tracking_error: H_excess_series for '{model_name}' has "
                    f"{len(H_excess_series)} entries but returns have {T} rows."
                )
            weights = np.zeros((T, N), dtype=float)
            for t, H_t in enumerate(H_excess_series):
                try:
                    w = tracking_error_weights(np.asarray(H_t), N)
                except Exception as exc:
                    logger.warning(
                        "run_tracking_error: weight failure for '%s' day %d (%s).",
                        model_name, t, exc,
                    )
                    w = np.full(N, 1.0 / N)
                weights[t] = w

            # Active return = portfolio return - SPY return
            port_ret = (weights * excess_returns).sum(axis=1)
            active_ret = port_ret  # already excess over SPY
            te_daily = float(active_ret.std(ddof=1))
            te_ann = te_daily * np.sqrt(self.trading_days)

            results[model_name] = {
                "te_ann": te_ann,
                "weights": weights,
                "active_returns": active_ret,
            }

        return results

    # ------------------------------------------------------------------
    def run_constrained(
        self,
        H_series_dict: dict,
        daily_returns: np.ndarray,
    ) -> dict:
        """Run long-only (constrained) GMV strategy for every model.

        Parameters
        ----------
        H_series_dict : dict
            ``{model_name: list_of_(N,N)_forecast_matrices}``.
        daily_returns : np.ndarray
            Daily return matrix of shape (T, N).

        Returns
        -------
        dict
            ``{model_name: {stats_dict, 'weights', 'returns'}}``.
        """
        daily_returns = np.asarray(daily_returns, dtype=float)
        T, N = daily_returns.shape
        results: dict = {}

        for model_name, H_series in H_series_dict.items():
            if len(H_series) != T:
                raise ValueError(
                    f"run_constrained: H_series for '{model_name}' has {len(H_series)} "
                    f"entries but daily_returns has {T} rows."
                )
            weights = np.zeros((T, N), dtype=float)
            for t, H_t in enumerate(H_series):
                try:
                    w = gmv_weights_constrained(np.asarray(H_t))
                except Exception as exc:
                    logger.warning(
                        "run_constrained: weight failure for '%s' day %d (%s).",
                        model_name, t, exc,
                    )
                    w = np.full(N, 1.0 / N)
                weights[t] = w

            net_ret = portfolio_net_return(weights, daily_returns, c=0.0)
            stats = compute_portfolio_stats(net_ret, trading_days=self.trading_days)
            stats["turnover_mean"] = (
                float(compute_turnover(weights, daily_returns).mean()) if T > 1 else 0.0
            )
            stats["concentration_mean"] = float(
                np.mean([concentration(weights[t]) for t in range(T)])
            )
            results[model_name] = {**stats, "weights": weights, "returns": net_ret}

        return results

    # ------------------------------------------------------------------
    def summarize_results(self, results: dict) -> pd.DataFrame:
        """Flatten nested results dict into a tidy summary DataFrame.

        Handles both:
        - results from :meth:`run_gmv_daily` (nested by cost level), and
        - results from :meth:`run_constrained` / :meth:`run_tracking_error`
          (flat per model).

        Parameters
        ----------
        results : dict
            Output dict from one of the ``run_*`` methods.

        Returns
        -------
        pd.DataFrame
            Summary with one row per (model, configuration) and columns for
            all numeric statistics.
        """
        rows: list = []
        for model_name, model_data in results.items():
            # Check if the value is itself a nested dict keyed by cost level
            first_val = next(iter(model_data.values()))
            if isinstance(first_val, dict):
                # run_gmv_daily structure: {c: {stats}}
                for c_val, stats in model_data.items():
                    row = {"model": model_name, "cost": c_val}
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            row[k] = v
                    rows.append(row)
            else:
                # run_constrained / run_tracking_error structure: flat stats
                row = {"model": model_name}
                for k, v in model_data.items():
                    if isinstance(v, (int, float)):
                        row[k] = v
                rows.append(row)

        return pd.DataFrame(rows)
