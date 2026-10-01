"""Experiment 1 — Empirical realized covariance forecasting.

Reproduces the paper's one-day-ahead out-of-sample covariance forecast
comparison: HAR vs HARQ, HAR-DRD vs HARQ-DRD, EWMA vs EWMAQ, HEAVY vs HEAVYQ.

Because the original TAQ data (1993-2013) is proprietary, we use a synthetic
DCC-GARCH dataset with realistic dynamics (2300 days, 10 assets, M=78 intraday
intervals per day, rolling window=1000 days, giving ≈1300 out-of-sample days).

Outputs saved to results/:
  - exp1_mean_losses.csv         — full-sample mean Frobenius & QLIKE losses
  - exp1_subsamples.csv          — split by top/bottom 5% of ||pi_t||
  - exp1_dm_tests.json           — pairwise Diebold-Mariano test results
  - exp1_mcs_members.json        — 90% Model Confidence Set members
  - exp1_loss_series.csv         — per-day loss time-series for all models
  - exp1_loss_comparison.png     — cumulative loss plot
  - exp1_q_signs.csv             — Q-coefficient sign check

Run with:
    python -m exp.exp1_forecasting
    python -m exp.exp1_forecasting --fast   # quick smoke test
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import generate_dataset
from src.evaluation import split_by_pi_norm
from src.realized_measures import compute_daily_rc_series
from src.rolling_forecast import rolling_forecast_all_models, estimate_window_params, MODEL_NAMES
from src.statistical_tests import diebold_mariano_test, model_confidence_set
from src.utils import vech

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Experiment configuration
N_ASSETS = 10
N_DAYS = 2300          # total days (≈1300 OOS after window=1000)
M_INTRADAY = 78        # 5-min bars in 6.5-hr session
ESTIMATION_WINDOW = 1000
SEED = 42

# Q-model pairs for Diebold-Mariano tests
Q_PAIRS = [
    ("HAR",     "HARQ"),
    ("HAR-DRD", "HARQ-DRD"),
    ("EWMA",    "EWMAQ"),
    ("HEAVY",   "HEAVYQ"),
]


def build_realized_measures(dataset: dict) -> tuple[list, np.ndarray, np.ndarray]:
    """Compute S_t, pi_t, and vech(S_t) series from intraday returns.

    Parameters
    ----------
    dataset : dict
        Output of :func:`data.data_loader.generate_dataset`.

    Returns
    -------
    tuple
        ``(S_list, pi_series, s_series)`` — T-length list of (N,N) matrices,
        (T, K) uncertainty array, and (T, K) vech-stacked realized covariance.
        The returned *pi_series* is raw (global demeaning only); rolling
        window demeaning is handled inside :func:`rolling_forecast_all_models`.
    """
    logger.info("Computing realized measures …")
    S_list, _, _, pi_list = compute_daily_rc_series(dataset["intraday_returns"])

    T = len(S_list)
    N = dataset["n_assets"]
    K = N * (N + 1) // 2

    s_series = np.zeros((T, K), dtype=float)
    pi_series = np.zeros((T, K), dtype=float)

    for t in range(T):
        s_series[t] = vech(S_list[t])
        pi_series[t] = pi_list[t]

    return S_list, pi_series, s_series


def run_exp1(fast: bool = False) -> dict:
    """Run Experiment 1 and save all artifacts to ``results/``.

    Parameters
    ----------
    fast : bool
        If *True*, use a reduced dataset (n_days=400, window=200) for quick
        CI smoke testing; the paper-scale run uses the full configuration.

    Returns
    -------
    dict
        Summary dict with keys ``losses_summary``, ``dm_results``,
        ``mcs_members``, ``subsample_df``.
    """
    n_days  = 400 if fast else N_DAYS
    window  = 200 if fast else ESTIMATION_WINDOW

    logger.info(
        "Exp1 | n_assets=%d  n_days=%d  M=%d  window=%d  fast=%s",
        N_ASSETS, n_days, M_INTRADAY, window, fast,
    )
    t0 = time.time()

    # ------------------------------------------------------------------ data
    dataset = generate_dataset(
        n_assets=N_ASSETS, n_days=n_days, M_intraday=M_INTRADAY,
        estimation_window=window, seed=SEED,
    )
    logger.info("Dataset ready in %.1f s.", time.time() - t0)

    S_list, pi_series, s_series = build_realized_measures(dataset)
    logger.info("Realized measures done in %.1f s.", time.time() - t0)

    # --------------------------------------------- rolling forecast (all 8)
    logger.info("Rolling forecast (window=%d) …", window)
    results = rolling_forecast_all_models(
        s_series=s_series,
        S_list=S_list,
        pi_series=pi_series,
        daily_returns=dataset["daily_returns"],
        estimation_window=window,
        N=N_ASSETS,
    )
    logger.info("Forecasts done in %.1f s.", time.time() - t0)

    # ----------------------------------------------------------- loss tables
    # losses DataFrame per model is stored directly in results[model]["losses"]
    frob_losses = pd.DataFrame(
        {m: results[m]["losses"]["frobenius"].values for m in MODEL_NAMES}
    )
    qlike_losses = pd.DataFrame(
        {m: results[m]["losses"]["qlike"].values for m in MODEL_NAMES}
    )

    # Save per-day series
    pd.concat(
        [frob_losses.add_suffix("_frob"), qlike_losses.add_suffix("_qlike")],
        axis=1,
    ).to_csv(RESULTS_DIR / "exp1_loss_series.csv", index=False)

    # Mean losses summary
    losses_summary = pd.DataFrame({
        "Frobenius": {m: results[m]["losses"]["frobenius"].mean() for m in MODEL_NAMES},
        "QLIKE":     {m: results[m]["losses"]["qlike"].mean()     for m in MODEL_NAMES},
    })
    losses_summary.to_csv(RESULTS_DIR / "exp1_mean_losses.csv")
    logger.info("Mean losses:\n%s", losses_summary.to_string())

    # ----------------------------------------- sub-sample by ||pi_t|| rank
    # Use any model's pi_norms (identical across models for same OOS window)
    pi_norms_oos = results[MODEL_NAMES[0]]["pi_norms"]
    sub_rows = []
    for model in MODEL_NAMES:
        ldf = results[model]["losses"]
        low_df, high_df = split_by_pi_norm(ldf, pi_norms_oos)
        sub_rows.append({
            "model": model,
            "frob_low95": low_df["frobenius"].mean(),
            "frob_high5": high_df["frobenius"].mean(),
            "qlike_low95": low_df["qlike"].mean(),
            "qlike_high5": high_df["qlike"].mean(),
        })
    subsample_df = pd.DataFrame(sub_rows).set_index("model")
    subsample_df.to_csv(RESULTS_DIR / "exp1_subsamples.csv")
    logger.info("Sub-sample table:\n%s", subsample_df.to_string())

    # ---------------------------------------- Diebold-Mariano (Q vs base)
    dm_results: dict = {}
    for (base, q_model) in Q_PAIRS:
        for loss_col in ("frobenius", "qlike"):
            key = f"{base}_vs_{q_model}_{loss_col}"
            loss_b = results[base][   "losses"][loss_col].values
            loss_q = results[q_model]["losses"][loss_col].values
            stat, pval = diebold_mariano_test(loss_b, loss_q)
            dm_results[key] = {
                "statistic":       float(stat),
                "p_value":         float(pval),
                "significant_5pct": bool(pval < 0.05),
                # positive stat ⟹ base loss > Q loss ⟹ Q wins
                "q_model_better":  bool(stat > 0),
            }
            logger.info("DM %-8s vs %-8s (%s): stat=%+.3f  p=%.4f",
                        base, q_model, loss_col, stat, pval)

    with open(RESULTS_DIR / "exp1_dm_tests.json", "w") as fh:
        json.dump(dm_results, fh, indent=2)

    # ---------------------------------------- Model Confidence Set (Frob)
    logger.info("MCS (B=500) …")
    mcs_members = model_confidence_set(frob_losses, alpha=0.1, B=500, seed=SEED)
    with open(RESULTS_DIR / "exp1_mcs_members.json", "w") as fh:
        json.dump({"mcs_90pct_frobenius": mcs_members}, fh, indent=2)
    logger.info("MCS members (Frobenius): %s", mcs_members)

    # ------------------------------------------------- Q-coefficient signs
    _report_q_signs(S_list, s_series, pi_series, dataset["daily_returns"],
                    window, N_ASSETS)

    # -------------------------------------------- cumulative-loss plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, df) in zip(
        axes,
        [("Frobenius", frob_losses), ("QLIKE", qlike_losses)],
    ):
        df.cumsum().plot(ax=ax)
        ax.set_title(f"Cumulative {metric} Loss — OOS")
        ax.set_xlabel("Out-of-sample day")
        ax.set_ylabel(f"Cumulative {metric}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_loss_comparison.png", dpi=150)
    plt.close()
    logger.info("Loss comparison plot saved.")

    logger.info("Exp1 complete.  Elapsed: %.1f s.", time.time() - t0)
    return {
        "losses_summary": losses_summary,
        "dm_results":     dm_results,
        "mcs_members":    mcs_members,
        "subsample_df":   subsample_df,
    }


def _report_q_signs(
    S_list: list,
    s_series: np.ndarray,
    pi_series: np.ndarray,
    daily_returns: np.ndarray,
    window: int,
    N: int,
) -> None:
    """Fit Q-models on the first in-sample window and log coefficient signs.

    Uses :func:`src.rolling_forecast.estimate_window_params` with the correct
    positional-argument order: ``(s_window, pi_window_dm, S_window,
    returns_window, N, model_name)``.
    """
    t_end = window
    win_s  = s_series[0:t_end]
    win_pi = pi_series[0:t_end]
    win_pi_dm = win_pi - win_pi.mean(axis=0)      # demean within window
    win_S  = S_list[0:t_end]
    win_r  = daily_returns[0:t_end]

    sign_rows = []
    for model in ("HARQ", "HARQ-DRD", "EWMAQ", "HEAVYQ"):
        try:
            params = estimate_window_params(
                win_s, win_pi_dm, win_S, win_r, N, model
            )
            q_coeff = _extract_q_coeff(model, params)
            sign_rows.append({
                "model":    model,
                "Q_coeff":  q_coeff,
                "negative": bool(q_coeff < 0),
            })
            logger.info("  %-10s  Q_coeff = %+.5f  (negative=%s)",
                        model, q_coeff, q_coeff < 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Q-sign check failed for %s: %s", model, exc)

    if sign_rows:
        df = pd.DataFrame(sign_rows).set_index("model")
        logger.info("Q-coefficient sign table:\n%s", df.to_string())
        df.to_csv(RESULTS_DIR / "exp1_q_signs.csv")


def _extract_q_coeff(model: str, params: dict) -> float:
    """Extract the primary Q-attenuation coefficient from fitted parameters.

    Actual key structure (confirmed from estimate_window_params output):
      HARQ      → params['beta'] shape (5, K); row index -1 is theta_1Q
      HARQ-DRD  → params['var_params'][i]['alpha_1Q'] per asset
      EWMAQ     → params['beta_Q']
      HEAVYQ    → params['beta_Q']

    Returns
    -------
    float
        Q coefficient; expected to be negative per the paper.
    """
    if model == "HARQ":
        beta = params.get("beta")
        if beta is None:
            return np.nan
        beta = np.asarray(beta)
        # Last row of beta matrix contains theta_1Q per vech element
        theta_1Q_row = beta[-1]
        nonzero = theta_1Q_row[np.abs(theta_1Q_row) > 1e-15]
        return float(nonzero.mean()) if len(nonzero) > 0 else float(theta_1Q_row.mean())

    if model == "HARQ-DRD":
        var_params = params.get("var_params", [])
        vals = [float(p["alpha_1Q"]) for p in var_params if "alpha_1Q" in p]
        return float(np.mean(vals)) if vals else np.nan

    if model in ("EWMAQ", "HEAVYQ"):
        return float(params.get("beta_Q", np.nan))

    return np.nan


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp1 — covariance forecasting")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced dataset for quick smoke testing")
    args = parser.parse_args()
    run_exp1(fast=args.fast)
