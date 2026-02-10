"""
Quick DARUC test with short horizon to validate SOCP formulation + numeric tuning.

Runs the full two-step pipeline (DAM -> DARUC) with only 4 hours instead of 48.
Same generators, PTDF, and uncertainty set — just fewer time periods.

Saves all results + Z matrix analysis to daruc_outputs/quick_test/.

Usage:
    python test_daruc_quick.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_rts_daruc import run_rts_daruc


def analyze_Z(Z_df: pd.DataFrame, data, out_dir: Path, rho=None):
    """Analyze Z matrix structure: row sums, wind vs non-wind, per-period."""
    gen_ids = list(Z_df.index)
    gen_type = data.gen_type
    time_labels = Z_df.columns.get_level_values("time").unique()
    K = Z_df.columns.get_level_values("k").unique().size

    # Build per-period rho lookup
    rho_arr = np.atleast_1d(rho) if rho is not None else None
    time_varying_rho = rho_arr is not None and rho_arr.shape[0] > 1

    rows = []
    for t in time_labels:
        Z_t = Z_df[t].values  # (I, K) for this time period

        for i, gid in enumerate(gen_ids):
            z_row = Z_t[i, :]
            rows.append({
                "gen_id": gid,
                "gen_type": gen_type[i],
                "time": t,
                "Z_row_sum": z_row.sum(),
                "Z_row_abs_sum": np.abs(z_row).sum(),
                "Z_row_norm": np.linalg.norm(z_row),
                **{f"Z_k{k}": z_row[k] for k in range(K)},
            })

    df = pd.DataFrame(rows)

    # Save full per-generator-per-period detail
    df.to_csv(out_dir / "Z_analysis_full.csv", index=False)

    # Summary: aggregate across time for each generator
    agg = (
        df.groupby(["gen_id", "gen_type"])
        .agg({
            "Z_row_sum": ["mean", "std", "min", "max"],
            "Z_row_abs_sum": ["mean", "max"],
            "Z_row_norm": ["mean", "max"],
        })
        .reset_index()
    )
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg.to_csv(out_dir / "Z_analysis_per_gen.csv", index=False)

    # Column sums per period (should sum to 0 for power balance if no slack)
    print("\n" + "=" * 70)
    print("Z MATRIX ANALYSIS")
    print("=" * 70)

    for t_idx, t in enumerate(time_labels):
        Z_t = Z_df[t].values
        col_sums = Z_t.sum(axis=0)  # sum over all generators for each k
        wind_mask = np.array([gt.upper() == "WIND" for gt in gen_type])
        wind_col_sums = Z_t[wind_mask].sum(axis=0)
        nonwind_col_sums = Z_t[~wind_mask].sum(axis=0)

        # Get rho for this period
        if time_varying_rho:
            rho_t = rho_arr[t_idx]
        elif rho_arr is not None:
            rho_t = float(rho_arr[0])
        else:
            rho_t = None
        rho_str = f"  rho={rho_t:.4f}" if rho_t is not None else ""

        print(f"\n  Period {t}:{rho_str}")
        print(f"    Column sums (all gens):  {col_sums}")
        print(f"    Column sums (wind only): {wind_col_sums}")
        print(f"    Column sums (non-wind):  {nonwind_col_sums}")

        # Per-wind-generator row sums
        wind_idx = np.where(wind_mask)[0]
        for wi in wind_idx:
            z_row = Z_t[wi, :]
            print(f"    {gen_ids[wi]:20s}  Z=[{', '.join(f'{v:+.4f}' for v in z_row)}]  sum={z_row.sum():+.4f}")

    # Non-zero Z rows for non-wind generators
    nonwind_active = df[(df["gen_type"] != "WIND") & (df["Z_row_abs_sum"] > 1e-6)]
    if not nonwind_active.empty:
        print(f"\n  Non-wind generators with non-zero Z ({len(nonwind_active)} entries):")
        summary = (
            nonwind_active.groupby("gen_id")
            .agg({"Z_row_sum": "mean", "Z_row_abs_sum": "mean"})
            .sort_values("Z_row_abs_sum", ascending=False)
            .head(20)
        )
        print(summary.to_string())
    else:
        print("\n  No non-wind generators have non-zero Z.")

    print("=" * 70)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick DARUC test")
    parser.add_argument("--hours", type=int, default=4, help="Horizon hours")
    parser.add_argument("--rho", type=float, default=2.0, help="Ellipsoid radius (static mode only)")
    parser.add_argument("--no-lines", action="store_true", help="Copper-plate (no line limits)")
    parser.add_argument(
        "--uncertainty-npz", type=str, default=None,
        help="Path to time-varying uncertainty NPZ (enables time-varying mode)",
    )
    parser.add_argument("--provider-start", type=int, default=0, help="Start index into NPZ time series")
    args = parser.parse_args()

    parts = []
    if args.no_lines:
        parts.append("copperplate")
    if args.uncertainty_npz:
        parts.append("tv")
    suffix = ("_" + "_".join(parts)) if parts else ""
    out_dir = Path(f"daruc_outputs/quick_test{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = run_rts_daruc(
        horizon_hours=args.hours,
        rho=args.rho,
        enforce_lines=not args.no_lines,
        uncertainty_provider_path=args.uncertainty_npz,
        provider_start_idx=args.provider_start,
    )

    daruc_results = outputs["daruc_results"]
    data = outputs["data"]
    dev_df = outputs["deviation_summary"]

    # Save standard outputs
    daruc_results["u"].to_csv(out_dir / "commitment_u.csv")
    daruc_results["p0"].to_csv(out_dir / "dispatch_p0.csv")
    daruc_results["Z"].to_csv(out_dir / "Z_coefficients.csv")
    dev_df.to_csv(out_dir / "deviation_summary.csv", index=False)

    # Save Sigma and rho
    np.save(out_dir / "Sigma.npy", outputs["Sigma"])
    np.save(out_dir / "rho.npy", np.atleast_1d(outputs["rho"]))

    # Z analysis
    z_analysis = analyze_Z(daruc_results["Z"], data, out_dir, rho=outputs["rho"])

    # Save DAM results for 3-way comparison
    # DAM uses key "p" (not "p0" like ARUC/DARUC)
    dam_results = outputs["dam_outputs"]["results"]
    dam_results["u"].to_csv(out_dir / "dam_commitment_u.csv")
    dam_results["p"].to_csv(out_dir / "dam_dispatch_p0.csv")

    # Save summary for comparison scripts
    summary = {
        "daruc_objective": daruc_results["obj"],
        "dam_objective": dam_results["obj"],
        "hours": args.hours,
        "rho_input": args.rho,
        "time_varying": outputs["time_varying"],
        "enforce_lines": not args.no_lines,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDARUC objective: {daruc_results['obj']:,.2f}")
    print(f"DAM objective:   {dam_results['obj']:,.2f}")
    print(f"Time-varying: {outputs['time_varying']}")
    print(f"\nResults saved to {out_dir}/")
    print("  commitment_u.csv, dispatch_p0.csv, Z_coefficients.csv")
    print("  dam_commitment_u.csv, dam_dispatch_p0.csv")
    print("  Z_analysis_full.csv, Z_analysis_per_gen.csv")
    print("  Sigma.npy, rho.npy, deviation_summary.csv, summary.json")
