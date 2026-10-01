import numpy as np
import pandas as pd
from src.evaluation import compute_losses, mean_losses, split_by_pi_norm


def test_compute_losses_shapes():
    T, N = 6, 3
    H_series = [np.eye(N) for _ in range(T)]
    S_series = [np.eye(N) * 2.0 for _ in range(T)]
    df = compute_losses(H_series, S_series)
    assert "frobenius" in df.columns
    assert "qlike" in df.columns
    assert len(df) == T


def test_frobenius_loss_zero_for_identical():
    T, N = 4, 3
    mat = np.eye(N) * 0.01
    H_series = [mat for _ in range(T)]
    S_series = [mat for _ in range(T)]
    df = compute_losses(H_series, S_series)
    assert np.allclose(df["frobenius"].values, 0.0)


def test_qlike_positive():
    T, N = 4, 3
    H_series = [np.eye(N) for _ in range(T)]
    S_series = [np.eye(N) * 2.0 for _ in range(T)]
    df = compute_losses(H_series, S_series)
    assert np.all(df["qlike"].values > 0)


def test_split_by_pi_norm_counts():
    T = 100
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"frobenius": rng.random(T), "qlike": rng.random(T)})
    pi_norms = np.arange(T, dtype=float)
    low, high = split_by_pi_norm(df, pi_norms, threshold_pct=95.0)
    assert len(high) == 5
    assert len(low) + len(high) == T


def test_mean_losses_keys():
    T, N = 5, 2
    H_series = [np.eye(N) for _ in range(T)]
    S_series = [np.eye(N) * 1.5 for _ in range(T)]
    df = compute_losses(H_series, S_series)
    result = mean_losses(df)
    assert "frobenius" in result
    assert "qlike" in result
