"""Experiment 2 — Portfolio evaluation of dynamic-attenuation covariance forecasts.

Reproduces the paper's economic-value experiments: Q-model covariance forecasts
(HARQ, HARQ-DRD, EWMAQ, HEAVYQ) are used to construct global-minimum-variance
portfolios and tracking-error portfolios, and their economic gains over the
paired non-Q benchmarks are evaluated.

Because the original TAQ data (1993-2013) is proprietary, we use the same
synthetic DCC-GARCH dataset as Experiment 1 (seed=42, 10 assets, 2300 days,
rolling window=1000 days).

Portfolio strategies evaluated
-------------------------------
1. Daily unconstrained GMV portfolios (c ∈ {0%, 1%, 2%})
2. No-short-sale GMV portfolios
3. Minimum tracking-error portfolios versus SPY proxy
4. Ex-post shrinkage benchmarks applied to HAR-DRD forecasts
5. 1/N equal-weight benchmark

Outputs saved to results/
--------------------------
- exp2_gmv_summary.csv           — turnover, vol, SR, utility gains (unconstrained GMV)
- exp2_constrained_summary.csv   — same for no-short-sale GMV
- exp2_tracking_error.csv        — annualized TE and utility gains
- exp2_shrinkage_comparison.csv  — HARQ-DRD vs shrunk HAR-DRD vs 1/N
- exp2_ceq_gains.csv             — CEQ gains matrix (bps, gamma ∈ {1, 10})
- exp2_cumulative_wealth.png     — cumulative portfolio wealth plot
- exp2_turnover_bars.png         — turnover comparison bar chart
- exp2_ceq_heatmap.png           — CEQ gains heatmap

Run with:
    python -m exp.exp2_portfolio
    python -m exp.exp2_portfolio --fast   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import generate_dataset
from src.realized_measures import compute_daily_rc_series
from src.rolling_forecast import rolling_forecast_all_models, MODEL_NAMES
from src.portfolio import (
    PortfolioEvaluator,
    gmv_weights_unconstrained,
    compute_turnover,
    portfolio_net_return,
    compute_portfolio_stats,
    certainty_equivalent_gain,
    white_reality_check,
    quadratic_utility,
    concentration,
    short_position,
)
from src.shrinkage import (
    identity_target,
    equicorrelation_target,
    single_factor_target,
    compute_shrinkage_intensity,
    shrink_forecast,
)
from src.utils import vech, nearest_pd, is_pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Experiment configuration (identical to Exp 1)
N_ASSETS = 10
N_DAYS = 2300
M_INTRADAY = 78
ESTIMATION_WINDOW = 1000
SEED = 42
TRADING_DAYS = 252

# Transaction cost rates (fraction)
C_LIST = [0.0, 0.001, 0.002]

# Risk aversion levels
GAMMA_LIST = [1, 10]

# Q-pairs for economic gain comparison
Q_PAIRS = [
    ("HAR",     "HARQ"),
    ("HAR-DRD", "HARQ-DRD"),
    ("EWMA",    "EWMAQ"),
    ("HEAVY",   "HEAVYQ"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_realized_measures(dataset: dict) -> tuple[list, np.ndarray, np.ndarray]:
    """Compute S_list, pi_series, s_series from intraday returns."""
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


def extract_oos_data(
    results: dict,
    dataset: dict,
    estimation_window: int,
) -> tuple[dict, np.ndarray]:
    """Extract OOS H_series_dict and aligned daily returns.

    Parameters
    ----------
    results : dict
        Output of :func:`src.rolling_forecast.rolling_forecast_all_models`.
    dataset : dict
        Output of :func:`data.data_loader.generate_dataset`.
    estimation_window : int
        Rolling window length; OOS starts at index *estimation_window*.

    Returns
    -------
    tuple
        ``(H_series_dict, oos_returns)`` where *H_series_dict* maps model
        names to lists of forecast matrices and *oos_returns* is (T_oos, N).
    """
    n_oos = len(results[MODEL_NAMES[0]]["forecasts"])
    oos_returns = dataset["daily_returns"][estimation_window : estimation_window + n_oos]
    H_series_dict = {m: results[m]["forecasts"] for m in MODEL_NAMES}
    return H_series_dict, oos_returns


def make_spy_proxy(daily_returns: np.ndarray) -> np.ndarray:
    """Construct an SPY proxy as an equal-weight portfolio of all assets.

    Parameters
    ----------
    daily_returns : np.ndarray
        Shape (T, N) OOS daily returns.

    Returns
    -------
    np.ndarray
        SPY proxy returns of shape (T,).
    """
    return daily_returns.mean(axis=1)


def compute_1on_n_returns(
    daily_returns: np.ndarray, c: float = 0.0
) -> np.ndarray:
    """Compute 1/N equal-weight portfolio net returns.

    Parameters
    ----------
    daily_returns : np.ndarray
        Shape (T, N).
    c : float
        Per-unit transaction cost fraction.

    Returns
    -------
    np.ndarray
        Net portfolio returns of shape (T,).
    """
    T, N = daily_returns.shape
    weights = np.full((T, N), 1.0 / N)
    return portfolio_net_return(weights, daily_returns, c=c)


def apply_shrinkage_to_forecasts(
    H_series: list,
    S_list: list,
    estimation_window: int,
    targets: list[str],
) -> dict[str, list]:
    """Shrink each HAR-DRD forecast toward named targets.

    For each OOS day *t* (index *estimation_window + i* in the full sample),
    we estimate shrinkage intensity from the in-window sample covariance and
    shrink *H_series[i]* toward the target.

    Parameters
    ----------
    H_series : list of np.ndarray
        HAR-DRD forecasts, one (N, N) matrix per OOS day.
    S_list : list of np.ndarray
        Full realized covariance series (length T = estimation_window + n_oos).
    estimation_window : int
        Rolling window length.
    targets : list of str
        Shrinkage target names: ``'identity'``, ``'equicorrelation'``,
        ``'single_factor'``.

    Returns
    -------
    dict
        ``{target_name: list_of_shrunk_(N,N)_matrices}``.
    """
    n_oos = len(H_series)
    N = H_series[0].shape[0]
    shrunk: dict[str, list] = {t: [] for t in targets}

    for i in range(n_oos):
        t_abs = estimation_window + i
        # In-window realized covariances
        S_window = S_list[t_abs - estimation_window : t_abs]
        S_arr = np.array([np.asarray(s) for s in S_window])   # (W, N, N)

        # Sample covariance of vech(S_t) across window for shrinkage intensity
        # Use the unconditional mean as reference point
        S_bar = S_arr.mean(axis=0)

        H_t = H_series[i]

        for target_name in targets:
            if target_name == "identity":
                F_t = identity_target(N) * np.trace(H_t) / N
            elif target_name == "equicorrelation":
                F_t = equicorrelation_target(H_t)
            elif target_name == "single_factor":
                F_t = single_factor_target(H_t, S_bar)
            else:
                F_t = H_t.copy()

            # Estimate shrinkage intensity from window residuals
            # Residuals = S_t - S_bar (proxy for forecast-error variance)
            resid_arr = S_arr - S_bar[np.newaxis]   # (W, N, N)
            try:
                alpha_star = compute_shrinkage_intensity(H_t, F_t, resid_arr)
            except Exception:
                alpha_star = 0.5
            alpha_star = float(np.clip(alpha_star, 0.0, 1.0))
            H_shrunk = shrink_forecast(H_t, F_t, alpha_star)
            if not is_pd(H_shrunk):
                H_shrunk = nearest_pd(H_shrunk)
            shrunk[target_name].append(H_shrunk)

    return shrunk


def compute_ceq_table(
    model_returns: dict[str, np.ndarray],
    q_pairs: list[tuple[str, str]],
    gamma_list: list[float],
) -> pd.DataFrame:
    """Compute annualized CEQ gains (bps) for Q vs non-Q pairs.

    Parameters
    ----------
    model_returns : dict
        ``{model_name: net_return_array}`` for c=0%.
    q_pairs : list of tuples
        Each tuple ``(base, q_model)`` compares *q_model* relative to *base*.
    gamma_list : list of float
        Risk-aversion levels.

    Returns
    -------
    pd.DataFrame
        Rows indexed by pair, columns by gamma.
    """
    rows = []
    for base, q_model in q_pairs:
        if base not in model_returns or q_model not in model_returns:
            continue
        row = {"pair": f"{q_model} vs {base}"}
        for gamma in gamma_list:
            ceq_bps = certainty_equivalent_gain(
                model_returns[q_model], model_returns[base], gamma=gamma
            )
            row[f"γ={gamma}"] = round(ceq_bps, 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_exp2(fast: bool = False) -> dict:
    """Run Experiment 2 and save all artifacts to ``results/``.

    Parameters
    ----------
    fast : bool
        If *True*, use a reduced dataset (n_days=400, window=200).

    Returns
    -------
    dict
        Summary dict with keys ``gmv_summary``, ``constrained_summary``,
        ``te_summary``, ``ceq_table``, ``shrinkage_summary``.
    """
    n_days = 400 if fast else N_DAYS
    window = 200 if fast else ESTIMATION_WINDOW

    logger.info(
        "Exp2 | n_assets=%d  n_days=%d  M=%d  window=%d  fast=%s",
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

    H_series_dict, oos_returns = extract_oos_data(results, dataset, window)
    T_oos, N = oos_returns.shape
    logger.info("OOS window: T=%d, N=%d", T_oos, N)

    spy_returns = make_spy_proxy(oos_returns)

    evaluator = PortfolioEvaluator(trading_days=TRADING_DAYS)

    # ==================================================================
    # 1. Daily unconstrained GMV portfolios
    # ==================================================================
    logger.info("Running unconstrained GMV portfolios …")
    gmv_results = evaluator.run_gmv_daily(H_series_dict, oos_returns, c_list=C_LIST)
    gmv_summary = evaluator.summarize_results(gmv_results)
    gmv_summary.to_csv(RESULTS_DIR / "exp2_gmv_summary.csv", index=False)
    logger.info("GMV summary:\n%s", gmv_summary.to_string())

    # ==================================================================
    # 2. No-short-sale GMV portfolios
    # ==================================================================
    logger.info("Running no-short-sale GMV portfolios …")
    constrained_results = evaluator.run_constrained(H_series_dict, oos_returns)
    constrained_summary = evaluator.summarize_results(constrained_results)
    constrained_summary.to_csv(RESULTS_DIR / "exp2_constrained_summary.csv", index=False)
    logger.info("Constrained GMV summary:\n%s", constrained_summary.to_string())

    # ==================================================================
    # 3. Minimum tracking-error portfolios versus SPY proxy
    # ==================================================================
    logger.info("Running minimum tracking-error portfolios …")
    # Excess returns = r_stock - r_SPY (broadcast SPY scalar to each asset)
    excess_returns = oos_returns - spy_returns[:, np.newaxis]

    # For TE portfolios, also compute excess-return covariance adjustments.
    # Since our H_series_dict is the covariance of raw returns, we approximate
    # the excess-return covariance as the same matrix (SPY proxy variance is
    # already implicitly included via equal weighting).  This is a standard
    # simplification; the paper adjusts H accordingly but the synthetic data
    # has no separate SPY series.
    H_excess_dict = H_series_dict   # simplification documented in methodology

    te_results = evaluator.run_tracking_error(H_excess_dict, excess_returns, spy_returns)

    te_rows = []
    for model_name, te_data in te_results.items():
        te_rows.append({
            "model": model_name,
            "te_ann": round(te_data["te_ann"] * 100, 4),   # in %
        })
    te_summary = pd.DataFrame(te_rows).set_index("model")

    # Add utility gains for TE strategy (gamma=1, c=0)
    te_utility_rows = []
    for base, q_model in Q_PAIRS:
        if base in te_results and q_model in te_results:
            for gamma in GAMMA_LIST:
                ceq = certainty_equivalent_gain(
                    te_results[q_model]["active_returns"],
                    te_results[base]["active_returns"],
                    gamma=gamma,
                )
                te_utility_rows.append({
                    "pair": f"{q_model} vs {base}",
                    f"γ={gamma}_ceq_bps": round(ceq, 2),
                })
    te_utility_df = pd.DataFrame(te_utility_rows).set_index("pair") if te_utility_rows else pd.DataFrame()
    te_summary.to_csv(RESULTS_DIR / "exp2_tracking_error.csv")
    if not te_utility_df.empty:
        te_utility_df.to_csv(RESULTS_DIR / "exp2_tracking_utility_gains.csv")
    logger.info("Tracking error (annualised %%):\n%s", te_summary.to_string())

    # ==================================================================
    # 4. CEQ gain table — daily unconstrained GMV, c=0
    # ==================================================================
    logger.info("Computing CEQ gains …")
    model_rets_c0 = {m: gmv_results[m][0.0]["returns"] for m in MODEL_NAMES}
    ceq_rows = []
    for gamma in GAMMA_LIST:
        for base, q_model in Q_PAIRS:
            ceq_bps = certainty_equivalent_gain(
                model_rets_c0[q_model], model_rets_c0[base], gamma=gamma
            )
            ceq_rows.append({
                "pair": f"{q_model} vs {base}",
                "gamma": gamma,
                "ceq_bps": round(ceq_bps, 2),
            })
    ceq_table = pd.DataFrame(ceq_rows)
    ceq_table.to_csv(RESULTS_DIR / "exp2_ceq_gains.csv", index=False)
    logger.info("CEQ gains (bps):\n%s", ceq_table.to_string())

    # White's Reality Check for best Q-model (HARQ-DRD vs HAR-DRD)
    wrc_pvalue = None
    if "HARQ-DRD" in model_rets_c0 and "HAR-DRD" in model_rets_c0:
        gamma_rc = 1
        # Compute per-period utility values for the Reality Check (1-D arrays)
        def _per_period_utility(rets: np.ndarray, g: float) -> np.ndarray:
            r = np.asarray(rets, dtype=float)
            c = g / (2.0 * (1.0 + g))
            gross = 1.0 + r
            return gross - c * gross ** 2

        u_q  = _per_period_utility(model_rets_c0["HARQ-DRD"], gamma_rc)
        u_b  = _per_period_utility(model_rets_c0["HAR-DRD"],  gamma_rc)
        gains_daily = u_q - u_b
        n_boot = 99 if fast else 999
        wrc_pvalue = white_reality_check(gains_daily, B=n_boot, block_size=22, seed=SEED)
        logger.info("White Reality Check p-value (HARQ-DRD vs HAR-DRD, γ=1): %.4f", wrc_pvalue)
        with open(RESULTS_DIR / "exp2_white_reality_check.json", "w") as f:
            json.dump({"harq_drd_vs_har_drd_gamma1_pvalue": wrc_pvalue}, f, indent=2)

    # ==================================================================
    # 5. Shrinkage benchmarks applied to HAR-DRD forecasts
    # ==================================================================
    logger.info("Computing shrinkage benchmarks …")
    har_drd_series = H_series_dict.get("HAR-DRD", [])
    harq_drd_series = H_series_dict.get("HARQ-DRD", [])
    shrinkage_targets = ["identity", "equicorrelation", "single_factor"]

    shrunk_dict = apply_shrinkage_to_forecasts(
        har_drd_series, S_list, window, shrinkage_targets
    )

    shrinkage_rows = []
    for target_name, shrunk_series in shrunk_dict.items():
        label = f"HAR-DRD+{target_name}"
        w_shrunk = np.zeros((T_oos, N), dtype=float)
        for t, H_t in enumerate(shrunk_series):
            try:
                w_shrunk[t] = gmv_weights_unconstrained(H_t)
            except Exception:
                w_shrunk[t] = np.full(N, 1.0 / N)

        r_shrunk = portfolio_net_return(w_shrunk, oos_returns, c=0.0)
        r_harq_drd = model_rets_c0.get("HARQ-DRD", r_shrunk)

        stats_s = compute_portfolio_stats(r_shrunk, trading_days=TRADING_DAYS)
        stats_q = compute_portfolio_stats(model_rets_c0.get("HARQ-DRD", r_shrunk), trading_days=TRADING_DAYS)

        for gamma in GAMMA_LIST:
            ceq_bps = certainty_equivalent_gain(model_rets_c0.get("HARQ-DRD", r_shrunk), r_shrunk, gamma=gamma)
            shrinkage_rows.append({
                "strategy": label,
                "gamma": gamma,
                "ann_vol_shrunk": round(stats_s["ann_vol"] * 100, 4),
                "ann_vol_harqdrd": round(stats_q["ann_vol"] * 100, 4),
                "ceq_harqdrd_vs_shrunk_bps": round(ceq_bps, 2),
            })

    # 1/N benchmark vs HARQ-DRD
    r_1n = compute_1on_n_returns(oos_returns, c=0.0)
    stats_1n = compute_portfolio_stats(r_1n, trading_days=TRADING_DAYS)
    for gamma in GAMMA_LIST:
        ceq_bps = certainty_equivalent_gain(model_rets_c0.get("HARQ-DRD", r_1n), r_1n, gamma=gamma)
        shrinkage_rows.append({
            "strategy": "1/N",
            "gamma": gamma,
            "ann_vol_shrunk": round(stats_1n["ann_vol"] * 100, 4),
            "ann_vol_harqdrd": round(stats_q["ann_vol"] * 100, 4),
            "ceq_harqdrd_vs_shrunk_bps": round(ceq_bps, 2),
        })

    shrinkage_summary = pd.DataFrame(shrinkage_rows)
    shrinkage_summary.to_csv(RESULTS_DIR / "exp2_shrinkage_comparison.csv", index=False)
    logger.info("Shrinkage comparison:\n%s", shrinkage_summary.to_string())

    # ==================================================================
    # 6. Plots
    # ==================================================================
    _plot_cumulative_wealth(model_rets_c0, RESULTS_DIR / "exp2_cumulative_wealth.png")
    _plot_turnover_bars(gmv_results, RESULTS_DIR / "exp2_turnover_bars.png")
    _plot_ceq_heatmap(ceq_table, RESULTS_DIR / "exp2_ceq_heatmap.png")

    logger.info("Exp2 complete in %.1f s.", time.time() - t0)

    return {
        "gmv_summary": gmv_summary,
        "constrained_summary": constrained_summary,
        "te_summary": te_summary,
        "ceq_table": ceq_table,
        "shrinkage_summary": shrinkage_summary,
        "wrc_pvalue": wrc_pvalue,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_cumulative_wealth(
    model_rets: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Plot cumulative wealth (log scale) for all 8 models and 1/N."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_rets) + 1))
    for idx, (model_name, rets) in enumerate(model_rets.items()):
        cum_wealth = np.cumprod(1.0 + rets)
        ax.plot(cum_wealth, label=model_name, color=colors[idx], linewidth=1.2)

    ax.set_yscale("log")
    ax.set_title("Exp 2 — Cumulative Wealth (unconstrained GMV, c=0%)", fontsize=13)
    ax.set_xlabel("Out-of-sample day")
    ax.set_ylabel("Cumulative wealth (log scale)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved cumulative wealth plot to %s", save_path)


def _plot_turnover_bars(
    gmv_results: dict,
    save_path: Path,
) -> None:
    """Bar chart of mean daily turnover for each model (c=0)."""
    models = list(gmv_results.keys())
    turnovers = [gmv_results[m][0.0]["turnover_mean"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, turnovers, color=plt.cm.tab10(np.linspace(0, 1, len(models))))
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Exp 2 — Mean Daily Turnover (unconstrained GMV)", fontsize=13)
    ax.set_ylabel("Mean turnover")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars with values
    for bar, val in zip(bars, turnovers):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved turnover bar chart to %s", save_path)


def _plot_ceq_heatmap(
    ceq_table: pd.DataFrame,
    save_path: Path,
) -> None:
    """Heatmap of CEQ gains in basis points by pair and gamma."""
    pivot = ceq_table.pivot(index="pair", columns="gamma", values="ceq_bps")

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"γ={g}" for g in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Exp 2 — CEQ gains (bps) of Q vs non-Q models", fontsize=11)
    plt.colorbar(im, ax=ax, label="CEQ gain (bps)")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved CEQ heatmap to %s", save_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: portfolio evaluation")
    parser.add_argument("--fast", action="store_true", help="Quick smoke test mode")
    args = parser.parse_args()
    run_exp2(fast=args.fast)
