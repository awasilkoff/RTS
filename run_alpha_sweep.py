"""
run_alpha_sweep.py

Sweep conformal alpha values and measure cost / wind curtailment for DAM,
DARUC, and ARUC with time-varying uncertainty sets.

Alpha controls how conservative the uncertainty set is: higher alpha produces
a tighter conformal lower bound, which implies a larger rho and more expensive
robust commitment.

The covariance pre-computation (omega, mu, Sigma) is alpha-independent and
runs once (~5-10 min). Each alpha then only retrains the conformal model
(~30-60s) and generates a new NPZ, followed by a DARUC + ARUC solve.

This is a long-running script -- run manually:

    python run_alpha_sweep.py --alphas 0.9 0.99 --hours 6 --start-month 7 --start-day 15

Full overnight run (~40-60 min for 4 alphas at 12h with lines):

    python run_alpha_sweep.py --hours 12 --start-month 7 --start-day 15

Outputs (in --out-dir, default auto-named e.g. alpha_sweep/lines_12h_m07d15_a0.80_0.90_0.95_0.99/):
    sweep_results.csv                - Metrics for each alpha
    fig_price_of_robustness_alpha.pdf - Cost vs alpha
    fig_curtailment_vs_alpha.pdf     - Wind curtailment vs alpha
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure uncertainty_sets_refactored is importable
sys.path.insert(0, str(Path(__file__).parent / "uncertainty_sets_refactored"))

from uncertainty_sets_refactored.generate_uncertainty_sets import (
    UncertaintySetConfig,
    pre_compute_covariance,
    generate_uncertainty_sets_for_alpha,
)
from utils import CachedPaths

from run_rts_daruc import run_rts_daruc
from run_rts_aruc import run_rts_aruc
from compare_aruc_vs_daruc import (
    compute_wind_curtailment,
    compute_cost_breakdown,
    _round_commitment,
)


def _unit_hours(u_df: pd.DataFrame, times: list[str]) -> int:
    return int(_round_commitment(u_df[times]).values.sum())


# ---------------------------------------------------------------------------
# Phase 1: Pre-compute covariance (alpha-independent, ~5-10 min)
# ---------------------------------------------------------------------------


def build_covariance_artifacts(data_dir: Path) -> dict:
    """Run alpha-independent covariance pre-computation once."""
    paths = CachedPaths(
        actuals_parquet=data_dir / "actuals_filtered_rts4_constellation_v1.parquet",
        forecasts_parquet=data_dir / "forecasts_filtered_rts4_constellation_v1.parquet",
    )
    config = UncertaintySetConfig()  # defaults: softmax, focused_2d, etc.
    return pre_compute_covariance(config, paths), config, paths


# ---------------------------------------------------------------------------
# Phase 2: Generate NPZ per alpha (~30-60s each)
# ---------------------------------------------------------------------------


def generate_npz_for_alpha(
    alpha: float,
    cov: dict,
    config: UncertaintySetConfig,
    out_dir: Path,
) -> Path:
    """Generate uncertainty NPZ for a single alpha value."""
    alpha_dir = out_dir / f"alpha_{alpha:.4f}"
    return generate_uncertainty_sets_for_alpha(
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
        output_dir=alpha_dir,
        output_name="sigma_rho",
    )


# ---------------------------------------------------------------------------
# Phase 3: Run DARUC + ARUC for each alpha
# ---------------------------------------------------------------------------


def run_alpha_point(
    alpha: float,
    npz_path: Path,
    start_time: pd.Timestamp,
    hours: int,
    enforce_lines: bool,
    rho_lines_frac: float | None,
    mip_gap: float,
    provider_start: int,
) -> dict | None:
    """Run DARUC + ARUC with a given NPZ and return metrics row."""
    print(f"\n{'#' * 70}")
    print(f"# ALPHA = {alpha}")
    print(f"# NPZ   = {npz_path}")
    print(f"{'#' * 70}")
    t0 = time.time()

    # --- DARUC (includes DAM as step 1) ---
    try:
        daruc_out = run_rts_daruc(
            start_time=start_time,
            horizon_hours=hours,
            rho=1.0,  # ignored when provider used
            enforce_lines=enforce_lines,
            rho_lines_frac=rho_lines_frac,
            mip_gap=mip_gap,
            uncertainty_provider_path=str(npz_path),
            provider_start_idx=provider_start,
        )
        data = daruc_out["data"]
        daruc_res = daruc_out["daruc_results"]
        dam_res = daruc_out["dam_outputs"]["results"]
        common_times = list(data.time)

        dam_obj = dam_res["obj"]
        daruc_obj = daruc_res["obj"]
        dam_p0 = dam_res["p"]
        daruc_p0 = daruc_res["p0"]

        dam_curtail = compute_wind_curtailment(dam_p0, data, common_times)
        daruc_curtail = compute_wind_curtailment(daruc_p0, data, common_times)
        dam_uhours = _unit_hours(dam_res["u"], common_times)
        daruc_uhours = _unit_hours(daruc_res["u"], common_times)
        dam_cost = compute_cost_breakdown(dam_res["u"][common_times], dam_p0[common_times], data)
        daruc_cost = compute_cost_breakdown(daruc_res["u"][common_times], daruc_p0[common_times], data)
    except Exception as e:
        print(f"DARUC failed at alpha={alpha}: {e}")
        return None

    # --- ARUC ---
    try:
        aruc_out = run_rts_aruc(
            start_time=start_time,
            horizon_hours=hours,
            rho=1.0,  # ignored when provider used
            enforce_lines=enforce_lines,
            rho_lines_frac=rho_lines_frac,
            mip_gap=mip_gap,
            uncertainty_provider_path=str(npz_path),
            provider_start_idx=provider_start,
        )
        aruc_res = aruc_out["results"]
        aruc_obj = aruc_res["obj"]
        aruc_p0 = aruc_res["p0"]
        aruc_curtail = compute_wind_curtailment(aruc_p0, data, common_times)
        aruc_uhours = _unit_hours(aruc_res["u"], common_times)
        aruc_cost = compute_cost_breakdown(aruc_res["u"][common_times], aruc_p0[common_times], data)
    except Exception as e:
        print(f"ARUC failed at alpha={alpha}: {e}")
        return None

    elapsed = time.time() - t0

    # Mean rho from the NPZ (for reference)
    npz_data = np.load(npz_path)
    rho_mean = float(npz_data["rho"].mean())
    rho_max = float(npz_data["rho"].max())

    row = {
        "alpha": alpha,
        "rho_mean": rho_mean,
        "rho_max": rho_max,
        "dam_obj": dam_obj,
        "daruc_obj": daruc_obj,
        "aruc_obj": aruc_obj,
        "dam_cost_total": dam_cost["total"],
        "daruc_cost_total": daruc_cost["total"],
        "aruc_cost_total": aruc_cost["total"],
        "dam_curtail_mwh": dam_curtail["total_mwh"],
        "daruc_curtail_mwh": daruc_curtail["total_mwh"],
        "aruc_curtail_mwh": aruc_curtail["total_mwh"],
        "dam_curtail_pct": dam_curtail["pct"],
        "daruc_curtail_pct": daruc_curtail["pct"],
        "aruc_curtail_pct": aruc_curtail["pct"],
        "dam_unit_hours": dam_uhours,
        "daruc_unit_hours": daruc_uhours,
        "aruc_unit_hours": aruc_uhours,
        "elapsed_s": elapsed,
    }
    print(
        f"\n  alpha={alpha} done in {elapsed:.0f}s  "
        f"DAM={dam_obj:,.0f}  DARUC={daruc_obj:,.0f}  ARUC={aruc_obj:,.0f}  "
        f"rho_mean={rho_mean:.3f}"
    )
    return row


# ---------------------------------------------------------------------------
# Phase 4: Figures
# ---------------------------------------------------------------------------


def plot_price_of_robustness_alpha(df: pd.DataFrame, out_dir: Path):
    """Cost vs alpha figure with DAM baseline, DARUC, ARUC."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.axhline(
        df["dam_obj"].iloc[0], color="#2ca02c", linestyle="--",
        linewidth=1.2, label="DAM (baseline)",
    )
    ax.plot(
        df["alpha"], df["daruc_obj"], "o-", color="#ff7f0e",
        linewidth=1.5, markersize=5, label="DARUC",
    )
    ax.plot(
        df["alpha"], df["aruc_obj"], "s-", color="#1f77b4",
        linewidth=1.5, markersize=5, label="ARUC",
    )

    ax.fill_between(
        df["alpha"], df["daruc_obj"], df["aruc_obj"],
        alpha=0.15, color="#1f77b4",
    )

    ax.set_xlabel(r"Conformal coverage $\alpha$", fontsize=10)
    ax.set_ylabel("Objective cost ($)", fontsize=10)
    ax.set_title("Price of Robustness vs Conformal Alpha", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Secondary y-axis: % increase vs DAM
    dam_base = df["dam_obj"].iloc[0]
    if dam_base > 0:
        ax2 = ax.twinx()
        ax2.set_ylabel("% increase vs DAM", fontsize=9)
        y_lo, y_hi = ax.get_ylim()
        ax2.set_ylim(100 * (y_lo / dam_base - 1), 100 * (y_hi / dam_base - 1))
        ax2.tick_params(labelsize=8)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(
            out_dir / f"fig_price_of_robustness_alpha.{ext}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)
    print("  Saved fig_price_of_robustness_alpha.pdf/.png")


def plot_curtailment_vs_alpha(df: pd.DataFrame, out_dir: Path):
    """Wind curtailment vs alpha figure."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.axhline(
        df["dam_curtail_mwh"].iloc[0], color="#2ca02c", linestyle="--",
        linewidth=1.2, label="DAM (baseline)",
    )
    ax.plot(
        df["alpha"], df["daruc_curtail_mwh"], "o-", color="#ff7f0e",
        linewidth=1.5, markersize=5, label="DARUC",
    )
    ax.plot(
        df["alpha"], df["aruc_curtail_mwh"], "s-", color="#1f77b4",
        linewidth=1.5, markersize=5, label="ARUC",
    )

    ax.set_xlabel(r"Conformal coverage $\alpha$", fontsize=10)
    ax.set_ylabel("Wind Curtailment (MWh)", fontsize=10)
    ax.set_title("Wind Curtailment vs Conformal Alpha", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(
            out_dir / f"fig_curtailment_vs_alpha.{ext}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)
    print("  Saved fig_curtailment_vs_alpha.pdf/.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Alpha-sweep price of robustness (conformal alpha -> NPZ -> DARUC/ARUC)"
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.80, 0.90, 0.95, 0.99], help="Conformal alpha values to sweep (default: 0.80 0.90 0.95 0.99)")
    parser.add_argument("--hours", type=int, default=12, help="Horizon hours (default: 12)")
    parser.add_argument("--start-month", type=int, default=7, help="Start month (default: 7)")
    parser.add_argument("--start-day", type=int, default=15, help="Start day (default: 15)")
    parser.add_argument("--start-hour", type=int, default=0, help="Start hour (default: 0)")
    parser.add_argument("--provider-start", type=int, default=0, help="Start index into NPZ time series (default: 0)")
    parser.add_argument("--rho-lines-frac", type=float, default=0.25, help="Fraction of rho for line constraints (default: 0.25)")
    parser.add_argument("--mip-gap", type=float, default=0.005, help="MIP gap (default: 0.005)")
    parser.add_argument("--no-enforce-lines", dest="enforce_lines", action="store_false", help="Disable line flow limits (default: enforce lines)")
    parser.set_defaults(enforce_lines=True)
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: auto-generated from params)")
    parser.add_argument("--data-dir", type=str, default="uncertainty_sets_refactored/data", help="Data directory for uncertainty set generation")
    args = parser.parse_args()

    # Auto-generate output dir name from run parameters if not specified
    if args.out_dir is None:
        net = "lines" if args.enforce_lines else "copper"
        alphas_tag = "_".join(f"{a:.2f}" for a in sorted(args.alphas))
        args.out_dir = f"alpha_sweep/{net}_{args.hours}h_m{args.start_month:02d}d{args.start_day:02d}_a{alphas_tag}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    start_time = pd.Timestamp(
        year=2020, month=args.start_month, day=args.start_day, hour=args.start_hour,
    )

    print("=" * 70)
    print("ALPHA-SWEEP PRICE OF ROBUSTNESS")
    print("=" * 70)
    print(f"  Alphas:   {args.alphas}")
    print(f"  Horizon:  {args.hours}h")
    print(f"  Start:    {start_time}")
    print(f"  Network:  {'with line limits' if args.enforce_lines else 'copperplate'}")
    print(f"  rho_lines_frac: {args.rho_lines_frac}")
    print(f"  MIP gap:  {args.mip_gap}")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Pre-compute covariance (alpha-independent)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: PRE-COMPUTE COVARIANCE (alpha-independent)")
    print("=" * 70)
    t0_cov = time.time()
    cov, config, paths = build_covariance_artifacts(data_dir)
    cov_elapsed = time.time() - t0_cov
    print(f"\nCovariance pre-computation done in {cov_elapsed:.0f}s")

    # ------------------------------------------------------------------
    # Phase 2+3: For each alpha, generate NPZ then run DARUC + ARUC
    # ------------------------------------------------------------------
    rows = []
    for alpha in sorted(args.alphas):
        print(f"\n{'=' * 70}")
        print(f"PHASE 2: GENERATE NPZ FOR ALPHA={alpha}")
        print(f"{'=' * 70}")
        t0_npz = time.time()
        npz_path = generate_npz_for_alpha(alpha, cov, config, out_dir)
        npz_elapsed = time.time() - t0_npz
        print(f"  NPZ generated in {npz_elapsed:.0f}s: {npz_path}")

        print(f"\n{'=' * 70}")
        print(f"PHASE 3: DARUC + ARUC SOLVES FOR ALPHA={alpha}")
        print(f"{'=' * 70}")
        row = run_alpha_point(
            alpha=alpha,
            npz_path=npz_path,
            start_time=start_time,
            hours=args.hours,
            enforce_lines=args.enforce_lines,
            rho_lines_frac=args.rho_lines_frac,
            mip_gap=args.mip_gap,
            provider_start=args.provider_start,
        )
        if row is not None:
            rows.append(row)

    # ------------------------------------------------------------------
    # Phase 4: Aggregate and plot
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo successful runs. Exiting.")
        return

    csv_path = out_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved {csv_path}")

    print("\nGenerating figures...")
    plot_price_of_robustness_alpha(df, out_dir)
    plot_curtailment_vs_alpha(df, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
