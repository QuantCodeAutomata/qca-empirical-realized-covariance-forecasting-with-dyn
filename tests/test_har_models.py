import numpy as np
from src.rolling_forecast import estimate_window_params, MODEL_NAMES
from src.utils import vech


# ---------------------------------------------------------------------------
# Helper: build small estimation-window data for N=3 assets, W=30 days
# ---------------------------------------------------------------------------

def _make_window_data(N=3, W=30, seed=42):
    rng = np.random.default_rng(seed)
    K = N * (N + 1) // 2

    # Build W positive-definite (N, N) covariance matrices
    S_window = []
    for _ in range(W):
        A = rng.standard_normal((N, N)) * 0.01
        S = A @ A.T + np.eye(N) * 1e-4
        S_window.append(S)

    # Stack vech of each matrix into s_series
    s_series = np.array([vech(S) for S in S_window], dtype=float)  # (W, K)

    # Random positive uncertainty series, demeaned within window
    raw_pi = rng.random((W, K)) * 1e-3
    pi_series = raw_pi - raw_pi.mean(axis=0)  # (W, K), zero-mean

    # Random daily returns
    r_series = rng.standard_normal((W, N)) * 0.01  # (W, N)

    return s_series, pi_series, S_window, r_series


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_model_names_present():
    expected = ("HAR", "HARQ", "HAR-DRD", "HARQ-DRD", "EWMA", "EWMAQ", "HEAVY", "HEAVYQ")
    assert len(MODEL_NAMES) == 8
    for name in expected:
        assert name in MODEL_NAMES


def test_estimate_window_params_returns_dict():
    N, W = 3, 30
    s_series, pi_series, S_window, r_series = _make_window_data(N=N, W=W)
    for model_name in MODEL_NAMES:
        params = estimate_window_params(s_series, pi_series, S_window, r_series, N, model_name)
        assert isinstance(params, dict), f"{model_name} did not return a dict"


def test_harq_beta_matrix_has_q_row():
    """HARQ params['beta'] is shape (5, K); last row is the theta_1Q coefficient."""
    N, W = 3, 30
    K = N * (N + 1) // 2
    s_series, pi_series, S_window, r_series = _make_window_data(N=N, W=W)
    params = estimate_window_params(s_series, pi_series, S_window, r_series, N, "HARQ")
    assert "beta" in params, "HARQ must have 'beta' key"
    beta = np.asarray(params["beta"])
    assert beta.ndim == 2, "beta must be 2-D (n_regressors, K)"
    assert beta.shape[1] == K, f"beta columns must equal K={K}"
    # 5 rows: intercept, theta_1, theta_2, theta_3, theta_1Q
    assert beta.shape[0] == 5, "HARQ beta must have 5 rows (4 HAR + 1 Q)"


def test_harqdrd_alpha1q_in_var_params():
    """HARQ-DRD per-asset params contain 'alpha_1Q' key (the Q coefficient)."""
    N, W = 3, 30
    s_series, pi_series, S_window, r_series = _make_window_data(N=N, W=W)
    params = estimate_window_params(s_series, pi_series, S_window, r_series, N, "HARQ-DRD")
    assert "var_params" in params, "HARQ-DRD must have 'var_params' key"
    assert len(params["var_params"]) == N
    for p in params["var_params"]:
        assert "alpha_1Q" in p, "Each asset in HARQ-DRD var_params must have 'alpha_1Q'"


def test_ewmaq_beta_q_key():
    """EWMAQ params must contain 'beta_Q' (the alpha_Q attenuation coefficient)."""
    N, W = 3, 30
    s_series, pi_series, S_window, r_series = _make_window_data(N=N, W=W)
    params = estimate_window_params(s_series, pi_series, S_window, r_series, N, "EWMAQ")
    assert "beta_Q" in params, "EWMAQ must have 'beta_Q' key (alpha_Q)"


def test_heavyq_beta_q_key():
    """HEAVYQ params must contain 'beta_Q' (the a_Q attenuation coefficient)."""
    N, W = 3, 30
    s_series, pi_series, S_window, r_series = _make_window_data(N=N, W=W)
    params = estimate_window_params(s_series, pi_series, S_window, r_series, N, "HEAVYQ")
    assert "beta_Q" in params, "HEAVYQ must have 'beta_Q' key (a_Q)"
