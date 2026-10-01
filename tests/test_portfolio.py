import numpy as np
from src.portfolio import (
    gmv_weights_unconstrained,
    gmv_weights_constrained,
    compute_turnover,
    portfolio_net_return,
    concentration,
    quadratic_utility,
)


def test_gmv_unconstrained_sums_to_one():
    rng = np.random.default_rng(10)
    N = 4
    A = rng.standard_normal((N, N))
    H = A @ A.T + np.eye(N) * 0.1
    w = gmv_weights_unconstrained(H)
    assert np.isclose(w.sum(), 1.0)


def test_gmv_constrained_non_negative_and_sums_one():
    rng = np.random.default_rng(11)
    N = 4
    A = rng.standard_normal((N, N))
    H = A @ A.T + np.eye(N) * 0.1
    w = gmv_weights_constrained(H)
    assert np.all(w >= -1e-8)
    assert np.isclose(w.sum(), 1.0)


def test_compute_turnover_shape():
    T, N = 10, 5
    rng = np.random.default_rng(12)
    weights_series = np.abs(rng.standard_normal((T, N)))
    weights_series /= weights_series.sum(axis=1, keepdims=True)
    returns_series = rng.standard_normal((T, N)) * 0.01
    turnover = compute_turnover(weights_series, returns_series)
    assert len(turnover) == T - 1
    assert np.all(turnover >= 0)


def test_portfolio_net_return_length():
    T, N = 10, 5
    rng = np.random.default_rng(13)
    weights_series = np.abs(rng.standard_normal((T, N)))
    weights_series /= weights_series.sum(axis=1, keepdims=True)
    returns_series = rng.standard_normal((T, N)) * 0.01
    net_ret = portfolio_net_return(weights_series, returns_series, c=0.0)
    assert len(net_ret) == T


def test_concentration_equal_weights():
    N = 5
    w = np.ones(N) / N
    assert np.isclose(concentration(w), 1.0 / np.sqrt(N))


def test_quadratic_utility_scalar():
    rng = np.random.default_rng(14)
    returns = rng.standard_normal(50) * 0.01
    result = quadratic_utility(returns, gamma=2.0)
    assert isinstance(result, float)
