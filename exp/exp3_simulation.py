"""Experiment 3 — Simulation study: spurious vs fundamental turnover.

Reproduces the paper's simulation study that separates turnover caused by true
changes in latent covariances from spurious turnover induced by realized
covariance measurement error.

Methodology summary
-------------------
1. Build a 2,000-day empirical latent covariance path for 5 assets from the
   DCC-GARCH synthetic generator (or Massive data if available).  This path
   replaces the paper's empirical realized-kernel covariances.
2. For each intraday aggregation level M ∈ {390, 78, 26} (≈ 1-min, 5-min,
   15-min), simulate 1-second intraday returns with diurnal U-shaped scaling,
   aggregate to M bars, compute realized-covariance estimates, fit rolling
   HAR and HARQ on 1,000-day windows, and form GMV portfolios.
3. Compare against the oracle GMV portfolio (uses true latent Σ_t) and the
   HAR_∞ portfolio (HAR fitted on true latent covariances directly).
4. Report mean daily turnover, ex-post portfolio standard deviation, mean
   weight distance to the oracle, and mean weight distance to HAR_∞.

The key insight: HARQ attenuates noisy realized covariance inputs, so HARQ
portfolios should be closer to the oracle and have less spurious turnover than
HAR portfolios — especially at coarser sampling frequencies.

Outputs saved to results/
--------------------------
- exp3_simulation_results.csv   — tidy summary table (M × model metrics)
- exp3_turnover_by_freq.png     — turnover vs frequency for HAR / HARQ / oracle
- exp3_weight_dist_by_freq.png  — mean weight distance vs oracle
- exp3_decomp_table.csv         — fundamental vs spurious turnover decomposition

Run with:
    python -m exp.exp3_simulation
    python -m exp.exp3_simulation --fast   # reduced dataset for CI
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import generate_dataset
from src.realized_measures import compute_daily_rc_series
from src.simulate_intraday import (
    run_simulation_study,
    simulation_results_table,
    _har_forecast,
)
from src.portfolio import gmv_weights_unconstrained
from src.utils import vech, nearest_pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Experiment configuration
N_ASSETS_FULL = 10          # total empirical universe
N_ASSETS_SIM = 5            # 5-asset subset for simulation
TOTAL_DAYS = 2000           # paper: 2000-day empirical covariance path
ESTIMATION_WINDOW = 1000    # rolling window
N_OOS = 1000                # OOS portfolio decisions
M_INTRADAY = 78             # for the initial latent covariance estimation
SEED = 42                   # master random seed

# Sampling frequencies (bars per day): 1-min, 5-min, 15-min
AGGREGATION_LEVELS_FULL = [390, 78, 26]
AGGREGATION_LEVELS_FAST = [78, 26]   # reduced set for fast mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_latent_covariance_path(
    n_assets: int = N_ASSETS_FULL,
    n_days: int = TOTAL_DAYS,
    M_intraday: int = M_INTRADAY,
    seed: int = SEED,
) -> np.ndarray:
    """Generate the latent daily covariance path.

    Uses :func:`data.data_loader.generate_dataset` to produce synthetic
    DCC-GARCH data for *n_assets* over *n_days* days, then computes daily
    realized covariance matrices.  These realized covariances are treated as
    the **true latent** Σ_t for the simulation — following the paper's
    blueprint of using empirical MK covariances as ground truth.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    n_days : int
        Total number of trading days.
    M_intraday : int
        Intraday bars used for realized-covariance estimation.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Array of shape (T, n_assets, n_assets) containing daily covariance
        matrices.  T ≤ n_days (days with valid RC estimates only).
    """
    dataset = generate_dataset(
        n_assets=n_assets,
        n_days=n_days,
        M_intraday=M_intraday,
        estimation_window=1000,
        seed=seed,
    )
    S_list, _, _, _ = compute_daily_rc_series(dataset["intraday_returns"])
    logger.info("Latent covariance path: T=%d, N=%d", len(S_list), n_assets)
    return np.array([np.asarray(S) for S in S_list])  # (T, N, N)


def select_asset_subset(
    cov_path: np.ndarray,
    n_select: int = N_ASSETS_SIM,
    seed: int = SEED,
) -> tuple[np.ndarray, list]:
    """Select a random subset of assets from the covariance path.

    Parameters
    ----------
    cov_path : np.ndarray
        Shape (T, N, N) full-universe covariance path.
    n_select : int
        Number of assets to select.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    tuple
        ``(subset_cov_path, selected_indices)`` where *subset_cov_path* has
        shape (T, n_select, n_select).
    """
    rng = np.random.default_rng(seed)
    N = cov_path.shape[1]
    idx = sorted(rng.choice(N, size=n_select, replace=False).tolist())
    logger.info("Selected asset indices for simulation: %s", idx)
    subset = cov_path[:, idx, :][:, :, idx]   # (T, n_select, n_select)
    return subset, idx



def compute_oracle_weights(
    true_cov_path: list,
    estimation_window: int = ESTIMATION_WINDOW,
    n_oos: int = N_OOS,
) -> np.ndarray:
    """Compute oracle GMV weights using true latent Σ_t for each OOS day.

    Parameters
    ----------
    true_cov_path : list of np.ndarray
        Full-sample true covariance matrices.
    estimation_window : int
        OOS starts at *estimation_window*.
    n_oos : int
        Number of OOS days.

    Returns
    -------
    np.ndarray
        Oracle weights, shape (n_oos, N).
    """
    N = true_cov_path[0].shape[0]
    weights = np.zeros((n_oos, N), dtype=float)
    for i in range(n_oos):
        Sigma_t = np.asarray(true_cov_path[estimation_window + i])
        if not is_pd(Sigma_t):
            Sigma_t = nearest_pd(Sigma_t)
        try:
            weights[i] = gmv_weights_unconstrained(Sigma_t)
        except Exception:
            weights[i] = np.full(N, 1.0 / N)
    return weights


def is_pd(M: np.ndarray) -> bool:
    """Check if a matrix is positive definite."""
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def weight_distance(w1: np.ndarray, w2: np.ndarray) -> float:
    """Euclidean L2 distance between two weight vectors."""
    return float(np.linalg.norm(w1 - w2))


def turnover_series(weights: np.ndarray) -> np.ndarray:
    """Compute daily turnover (sum of |Δw|) assuming zero returns."""
    T, N = weights.shape
    to = np.zeros(T, dtype=float)
    for t in range(1, T):
        to[t] = np.abs(weights[t] - weights[t - 1]).sum()
    return to


def build_decomp_table(
    results: dict,
    oracle_w: np.ndarray,
    har_inf_w: np.ndarray,
    estimation_window: int,
    n_oos: int,
) -> pd.DataFrame:
    """Build fundamental vs spurious turnover decomposition table.

    Fundamental turnover  ≈ oracle turnover (driven by true Σ_t changes).
    Spurious turnover     ≈ (model_turnover - oracle_turnover).

    Parameters
    ----------
    results : dict
        Output of :func:`src.simulate_intraday.run_simulation_study`.
    oracle_w : np.ndarray
        Oracle GMV weights, shape (n_oos, N).
    har_inf_w : np.ndarray
        HAR_∞ weights, shape (n_oos, N).
    estimation_window : int
        Rolling window length (for labeling).
    n_oos : int
        Number of OOS days.

    Returns
    -------
    pd.DataFrame
        One row per (M, model) pair with fundamental and spurious turnover.
    """
    # Oracle and HAR_inf turnover (frequency-independent)
    oracle_to = turnover_series(oracle_w).mean()
    har_inf_to = turnover_series(har_inf_w).mean()

    rows = []
    for M, M_results in results.items():
        for model_name, metrics in M_results.items():
            model_to = metrics["turnover"].mean()
            spurious_to = max(model_to - oracle_to, 0.0)
            # Distance to HAR_inf
            dist_to_harinf = float(
                np.mean([
                    weight_distance(metrics["weights"][i], har_inf_w[i])
                    for i in range(n_oos)
                ])
            )
            rows.append({
                "M": M,
                "model": model_name,
                "mean_turnover": round(model_to, 6),
                "fundamental_turnover": round(oracle_to, 6),
                "spurious_turnover": round(spurious_to, 6),
                "dist_to_oracle": round(metrics["weight_dist"].mean(), 6),
                "dist_to_har_inf": round(dist_to_harinf, 6),
            })

    # Add oracle and HAR_inf rows
    oracle_dist_self = 0.0
    rows.append({
        "M": "all",
        "model": "oracle",
        "mean_turnover": round(oracle_to, 6),
        "fundamental_turnover": round(oracle_to, 6),
        "spurious_turnover": 0.0,
        "dist_to_oracle": 0.0,
        "dist_to_har_inf": round(
            float(np.mean([weight_distance(oracle_w[i], har_inf_w[i]) for i in range(n_oos)])),
            6,
        ),
    })
    rows.append({
        "M": "all",
        "model": "har_inf",
        "mean_turnover": round(har_inf_to, 6),
        "fundamental_turnover": round(oracle_to, 6),
        "spurious_turnover": round(max(har_inf_to - oracle_to, 0.0), 6),
        "dist_to_oracle": round(
            float(np.mean([weight_distance(har_inf_w[i], oracle_w[i]) for i in range(n_oos)])),
            6,
        ),
        "dist_to_har_inf": 0.0,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_turnover_by_freq(
    sim_table: pd.DataFrame,
    oracle_to: float,
    har_inf_to: float,
    save_path: Path,
) -> None:
    """Line chart: mean turnover vs sampling frequency for HAR/HARQ/optimal."""
    freq_labels = {390: "1-min", 78: "5-min", 26: "15-min"}

    models_to_plot = ["har", "harq"]
    styles = {"har": ("royalblue", "o--"), "harq": ("firebrick", "s-")}

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in models_to_plot:
        sub = sim_table[sim_table["model"] == model_name].copy()
        sub = sub[sub["M"].apply(lambda x: isinstance(x, (int, float)))]
        sub = sub.sort_values("M", ascending=False)  # decreasing M = coarser freq
        x_vals = [freq_labels.get(int(m), str(m)) for m in sub["M"]]
        y_vals = sub["mean_turnover"].values
        color, style = styles[model_name]
        ax.plot(x_vals, y_vals, style, color=color, label=model_name.upper(),
                linewidth=1.8, markersize=7)

    # Horizontal lines for oracle and HAR_∞
    x_range = [freq_labels.get(M, str(M)) for M in sorted(
        sim_table[sim_table["M"].apply(lambda x: isinstance(x, int))]["M"].unique(),
        reverse=True
    )]
    ax.axhline(oracle_to, color="green", linestyle=":", linewidth=1.5, label="Oracle (optimal Σ_t)")
    ax.axhline(har_inf_to, color="purple", linestyle="-.", linewidth=1.5, label="HAR_∞ (true Σ_t)")

    ax.set_title("Exp 3 — Mean Turnover vs Sampling Frequency", fontsize=13)
    ax.set_xlabel("Sampling frequency (bars/day)")
    ax.set_ylabel("Mean daily turnover")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved turnover-by-frequency plot to %s", save_path)


def _plot_weight_dist_by_freq(
    sim_table: pd.DataFrame,
    save_path: Path,
) -> None:
    """Line chart: mean weight distance to oracle vs sampling frequency."""
    freq_labels = {390: "1-min", 78: "5-min", 26: "15-min"}
    models_to_plot = ["har", "harq"]
    styles = {"har": ("royalblue", "o--"), "harq": ("firebrick", "s-")}

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in models_to_plot:
        sub = sim_table[sim_table["model"] == model_name].copy()
        sub = sub[sub["M"].apply(lambda x: isinstance(x, (int, float)))]
        sub = sub.sort_values("M", ascending=False)
        x_vals = [freq_labels.get(int(m), str(m)) for m in sub["M"]]
        y_vals = sub["mean_weight_dist"].values
        color, style = styles[model_name]
        ax.plot(x_vals, y_vals, style, color=color, label=model_name.upper(),
                linewidth=1.8, markersize=7)

    ax.set_title("Exp 3 — Mean Weight Distance to Oracle vs Sampling Frequency", fontsize=13)
    ax.set_xlabel("Sampling frequency (bars/day)")
    ax.set_ylabel("Mean L2 weight distance to oracle")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved weight-distance plot to %s", save_path)


def _plot_expost_var_by_freq(
    sim_table: pd.DataFrame,
    save_path: Path,
) -> None:
    """Bar chart: mean ex-post portfolio variance by model and frequency."""
    freq_labels = {390: "1-min", 78: "5-min", 26: "15-min"}
    models_of_interest = ["har", "harq", "optimal"]
    freq_vals = sorted(
        sim_table[sim_table["M"].apply(lambda x: isinstance(x, (int, float)))]["M"].unique(),
        reverse=True,
    )

    n_freq = len(freq_vals)
    n_models = len(models_of_interest)
    bar_width = 0.25
    x = np.arange(n_freq)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["royalblue", "firebrick", "green"]

    for i, model_name in enumerate(models_of_interest):
        sub = sim_table[sim_table["model"] == model_name].copy()
        sub = sub[sub["M"].apply(lambda x: isinstance(x, (int, float)))]
        sub = sub.sort_values("M", ascending=False)
        vals = []
        for freq in freq_vals:
            row = sub[sub["M"] == freq]
            vals.append(float(row["mean_ex_post_var"].values[0]) if len(row) else np.nan)
        ax.bar(
            x + i * bar_width, vals, width=bar_width,
            color=colors[i], label=model_name.upper(), alpha=0.85,
        )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([freq_labels.get(int(f), str(f)) for f in freq_vals])
    ax.set_title("Exp 3 — Mean Ex-Post Portfolio Variance by Model & Frequency", fontsize=12)
    ax.set_xlabel("Sampling frequency")
    ax.set_ylabel("Mean ex-post portfolio variance")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ex-post variance plot to %s", save_path)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_exp3(fast: bool = False) -> dict:
    """Run Experiment 3 and save all artifacts to ``results/``.

    Parameters
    ----------
    fast : bool
        If *True*, use a reduced dataset and fewer aggregation levels.

    Returns
    -------
    dict
        ``{'sim_results': dict, 'sim_table': DataFrame,
           'decomp_table': DataFrame}``.
    """
    n_assets_sim = N_ASSETS_SIM
    total_days = 600 if fast else TOTAL_DAYS
    estimation_window = 200 if fast else ESTIMATION_WINDOW
    n_oos = 200 if fast else N_OOS
    agg_levels = AGGREGATION_LEVELS_FAST if fast else AGGREGATION_LEVELS_FULL

    logger.info(
        "Exp3 | n_assets=%d  total_days=%d  window=%d  n_oos=%d  M=%s  fast=%s",
        n_assets_sim, total_days, estimation_window, n_oos, agg_levels, fast,
    )
    t0 = time.time()

    # ----------------------------------------------------------- latent path
    logger.info("Building latent covariance path (%d days) …", total_days)
    full_cov_path = build_latent_covariance_path(
        n_assets=N_ASSETS_FULL,
        n_days=total_days,
        M_intraday=M_INTRADAY,
        seed=SEED,
    )
    logger.info("Full latent path shape: %s", full_cov_path.shape)

    # Ensure we have enough days
    needed = estimation_window + n_oos
    if full_cov_path.shape[0] < needed:
        raise ValueError(
            f"Latent covariance path too short: got {full_cov_path.shape[0]} days, "
            f"need {needed} (estimation_window={estimation_window} + n_oos={n_oos})."
        )

    # --------------------------------------------------- 5-asset subset
    subset_cov_path, selected_idx = select_asset_subset(
        full_cov_path, n_select=n_assets_sim, seed=SEED
    )
    logger.info(
        "Asset subset: indices=%s, shape=%s", selected_idx, subset_cov_path.shape
    )

    # Use exactly needed days
    subset_cov_list = [subset_cov_path[t] for t in range(needed)]

    # -------------------------------------------------- simulation study
    logger.info("Running simulation study across M=%s …", agg_levels)
    sim_results = run_simulation_study(
        true_cov_path=subset_cov_list,
        estimation_window=estimation_window,
        n_oos=n_oos,
        aggregation_levels=agg_levels,
        seed=SEED,
    )
    logger.info("Simulation done in %.1f s.", time.time() - t0)

    # ------------------------------------------ oracle and HAR_∞ weights
    logger.info("Computing oracle and HAR_∞ weights …")
    oracle_w = compute_oracle_weights(
        subset_cov_list, estimation_window=estimation_window, n_oos=n_oos
    )
    oracle_to = turnover_series(oracle_w).mean()

    # HAR_∞: HAR on true latent vech(Σ_t) path
    try:
        N_sim = n_assets_sim
        s_true = np.array([vech(np.asarray(subset_cov_list[t])) for t in range(needed)])

        har_inf_w = np.zeros((n_oos, N_sim), dtype=float)
        for oos_idx in range(n_oos):
            t = estimation_window + oos_idx
            window_s = s_true[t - estimation_window : t]
            H_hat = _har_forecast(window_s, N_sim)
            if not is_pd(H_hat):
                H_hat = nearest_pd(H_hat)
            try:
                har_inf_w[oos_idx] = gmv_weights_unconstrained(H_hat)
            except Exception:
                har_inf_w[oos_idx] = np.full(N_sim, 1.0 / N_sim)
    except Exception as exc:
        logger.warning("HAR_∞ weight computation failed: %s. Using oracle.", exc)
        har_inf_w = oracle_w.copy()

    har_inf_to = turnover_series(har_inf_w).mean()
    logger.info(
        "Oracle mean TO=%.6f, HAR_∞ mean TO=%.6f", oracle_to, har_inf_to
    )

    # ------------------------------------------------ summary table
    sim_table = simulation_results_table(sim_results)
    logger.info("Simulation summary:\n%s", sim_table.to_string())
    sim_table.to_csv(RESULTS_DIR / "exp3_simulation_results.csv", index=False)

    # ----------------------------------------- decomposition table
    decomp_table = build_decomp_table(
        sim_results, oracle_w, har_inf_w, estimation_window, n_oos
    )
    decomp_table.to_csv(RESULTS_DIR / "exp3_decomp_table.csv", index=False)
    logger.info("Decomposition table:\n%s", decomp_table.to_string())

    # ------------------------------------------------------------ plots
    _plot_turnover_by_freq(
        sim_table, oracle_to, har_inf_to,
        RESULTS_DIR / "exp3_turnover_by_freq.png",
    )
    _plot_weight_dist_by_freq(
        sim_table,
        RESULTS_DIR / "exp3_weight_dist_by_freq.png",
    )
    _plot_expost_var_by_freq(
        sim_table,
        RESULTS_DIR / "exp3_expost_var_by_freq.png",
    )

    logger.info("Exp3 complete in %.1f s.", time.time() - t0)

    return {
        "sim_results": sim_results,
        "sim_table": sim_table,
        "decomp_table": decomp_table,
        "oracle_weights": oracle_w,
        "har_inf_weights": har_inf_w,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: simulation study")
    parser.add_argument("--fast", action="store_true", help="Quick smoke test mode")
    args = parser.parse_args()
    run_exp3(fast=args.fast)
