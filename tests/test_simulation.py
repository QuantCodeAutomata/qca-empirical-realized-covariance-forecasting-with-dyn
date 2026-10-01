import numpy as np
from src.simulate_intraday import (
    diurnal_sigma,
    simulate_one_day_intraday,
    aggregate_returns,
    compute_diurnal_weights,
    compute_weight_distance,
)


def test_diurnal_sigma_positive():
    for u in np.linspace(0, 1, 50):
        assert diurnal_sigma(u) > 0


def test_diurnal_sigma_u_shape():
    start = diurnal_sigma(0.0)
    mid = diurnal_sigma(0.5)
    end = diurnal_sigma(1.0)
    assert start > mid
    assert end > mid


def test_simulate_one_day_shape():
    Sigma = np.eye(3) * 0.0001
    result = simulate_one_day_intraday(Sigma, n_seconds=100, seed=0)
    assert result.shape == (100, 3)


def test_aggregate_returns_shape():
    rng = np.random.default_rng(20)
    intraday = rng.standard_normal((390, 4)) * 0.001
    agg = aggregate_returns(intraday, M=78)
    assert agg.shape == (78, 4)


def test_diurnal_weights_positive():
    weights = compute_diurnal_weights(100)
    assert len(weights) == 100
    assert np.all(weights > 0)


def test_weight_distance_zero():
    w = np.ones(4) / 4
    assert compute_weight_distance(w, w) == 0.0


def test_weight_distance_positive():
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    assert compute_weight_distance(e1, e2) > 0
