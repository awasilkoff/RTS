#!/usr/bin/env python3
"""
Generate uncertainty sets at multiple alpha levels.

Runs covariance (expensive) once, then generates NPZ for each alpha (cheap).
Prints rho summary statistics for comparison.
"""
from pathlib import Path
import numpy as np

from utils import CachedPaths
from generate_uncertainty_sets import (
    UncertaintySetConfig,
    pre_compute_covariance,
    generate_uncertainty_sets_for_alpha,
)

# ---- Configuration (match your v2_16d run) ----
ALPHAS = [0.90, 0.95, 0.99]
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_BASE = DATA_DIR / "uncertainty_sets_rts4_v2_16d"

config = UncertaintySetConfig(
    tau=0.05,
    ridge=0.001,
    k=128,
    alpha_target=0.90,  # placeholder, overridden per alpha
    n_bins=1,
    quantile_alpha=0.05,
    safety_margin=0.0,
    scaler_type="standard",
    feature_set="high_dim_16d",
    train_frac=0.75,
    omega_constraint="softmax",
    retrain_omega=True,
    omega_path=None,
    fit_step_size=0.001,
    fit_grad_clip=10.0,
    fit_max_iters=200,
    conformal_feature_cols=[
        "ens_mean", "ens_std", "ens_min", "ens_max", "n_models",
        "hour", "dow", "forecast_range", "forecast_range_normalized",
        "forecast_cv", "y_lag24", "forecast_error_lag24",
    ],
)

paths = CachedPaths(
    actuals_parquet=DATA_DIR / "actuals_filtered_rts4_constellation_v2.parquet",
    forecasts_parquet=DATA_DIR / "forecasts_filtered_rts4_constellation_v2.parquet",
)

# ---- Phase 1: Covariance (expensive, run once) ----
print("=" * 70)
print("Phase 1: Pre-computing covariance (this takes a few minutes)...")
print("=" * 70)
cov = pre_compute_covariance(config, paths)
print("Covariance done.\n")

# ---- Phase 2: Generate NPZ for each alpha ----
results = {}

for alpha in ALPHAS:
    print("=" * 70)
    print(f"Phase 2: Generating uncertainty sets for alpha={alpha}")
    print("=" * 70)

    out_name = f"sigma_rho_alpha{int(alpha * 100):02d}"
    npz_path = generate_uncertainty_sets_for_alpha(
        alpha_target=alpha,
        mu_all=cov["mu_all"],
        sigma_all=cov["sigma_all"],
        times_cov=cov["times_cov"],
        times_train=cov["times_train"],
        df_tot=cov["df_tot"],
        config=config,
        omega_hat=cov["omega_hat"],
        y_cols=cov["y_cols"],
        x_cols=cov["x_cols"],
        output_dir=OUTPUT_BASE,
        output_name=out_name,
    )

    # Load back and collect rho stats
    data = np.load(npz_path)
    rho = data["rho"]
    mu = data["mu"]
    sigma = data["sigma"]

    T, K = mu.shape
    e = np.ones(K)
    total_std = np.array([np.sqrt(float(e @ sigma[t] @ e)) for t in range(T)])
    mw_unc = rho * total_std

    results[alpha] = {
        "rho_mean": rho.mean(),
        "rho_median": np.median(rho),
        "rho_p95": np.percentile(rho, 95),
        "rho_max": rho.max(),
        "pct_zero": (rho == 0).sum() / T * 100,
        "mw_mean": mw_unc.mean(),
        "mw_median": np.median(mw_unc),
        "mw_p95": np.percentile(mw_unc, 95),
        "mw_max": mw_unc.max(),
    }
    print(f"  Saved: {npz_path}\n")

# ---- Summary comparison ----
print("\n" + "=" * 70)
print("ALPHA COMPARISON SUMMARY")
print("=" * 70)

header = f"{'Metric':<25}"
for a in ALPHAS:
    header += f"  {'alpha=' + str(a):>12}"
print(header)
print("-" * (25 + 14 * len(ALPHAS)))

for metric, label in [
    ("rho_mean", "Rho mean"),
    ("rho_median", "Rho median"),
    ("rho_p95", "Rho 95th pctile"),
    ("rho_max", "Rho max"),
    ("pct_zero", "% hours rho=0"),
    ("mw_mean", "MW unc. mean"),
    ("mw_median", "MW unc. median"),
    ("mw_p95", "MW unc. 95th pctile"),
    ("mw_max", "MW unc. max"),
]:
    row = f"{label:<25}"
    for a in ALPHAS:
        val = results[a][metric]
        if "pct" in metric or "%" in label:
            row += f"  {val:>11.1f}%"
        elif "mw" in metric or "MW" in label:
            row += f"  {val:>10.1f} MW"
        else:
            row += f"  {val:>12.3f}"

    print(row)

print("\n" + "=" * 70)
print("Files generated:")
for alpha in ALPHAS:
    print(f"  alpha={alpha}: {OUTPUT_BASE}/sigma_rho_alpha{int(alpha * 100):02d}.npz")
print("=" * 70)
