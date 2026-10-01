import numpy as np
import pytest
from src.realized_measures import (
    parzen_kernel_cov,
    compute_realized_kernel,
    compute_preaveraging_quarticity,
    compute_pi,
    demean_pi,
    compute_daily_rc_series,
)


def test_parzen_kernel_zero_lag():
    assert parzen_kernel_cov(0, 10) == 1.0


def test_parzen_kernel_symmetry():
    h, bandwidth = 3, 10
    assert parzen_kernel_cov(-h, bandwidth) == parzen_kernel_cov(h, bandwidth)


def test_realized_kernel_positive_definite():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((78, 5)) * 0.001
    S = compute_realized_kernel(returns)
    assert S.shape == (5, 5)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals > 0)


def test_realized_kernel_single_period():
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((2, 3)) * 0.001
    S = compute_realized_kernel(returns)
    assert S.shape == (3, 3)


def test_compute_pi_shape_and_positivity():
    # K=15 corresponds to N=5 (K = N*(N+1)//2 = 15)
    K = 15
    rng = np.random.default_rng(2)
    A = rng.standard_normal((K, K))
    IQ = A @ A.T + np.eye(K) * 1e-4
    _, pi = compute_pi(IQ)
    assert pi.shape == (K,)
    assert np.all(pi >= 0)


def test_demean_pi_zero_mean():
    rng = np.random.default_rng(3)
    pi_series = rng.standard_normal((50, 6)) ** 2
    demeaned = demean_pi(pi_series)
    assert demeaned.shape == pi_series.shape
    assert np.allclose(demeaned.mean(axis=0), 0.0)


def test_demean_pi_invalid_input():
    with pytest.raises(ValueError):
        demean_pi(np.array([1.0, 2.0, 3.0]))


def test_compute_daily_rc_series():
    rng = np.random.default_rng(4)
    n_days, n_assets, n_bars = 5, 3, 26
    intraday_list = [rng.standard_normal((n_bars, n_assets)) * 0.001 for _ in range(n_days)]
    S_list, IQ_list, Pi_list, pi_list = compute_daily_rc_series(intraday_list)
    assert len(S_list) == n_days
