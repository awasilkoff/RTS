"""
Quick ARUC-LDR test with short horizon to validate one-shot robust formulation.

Runs the standalone ARUC-LDR pipeline with only 4 hours instead of 48.
Same generators, PTDF, and uncertainty set — just fewer time periods.

Saves all results + Z matrix analysis to aruc_outputs/quick_test/.

Usage:
    python test_aruc_quick.py
    python test_aruc_quick.py --hours 4 --rho 2.0 --no-lines
"""
from pathlib import Path

import numpy as np
import pandas as pd

from run_rts_aruc import run_rts_aruc
from test_daruc_quick import analyze_Z  # reuse Z analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick ARUC-LDR test")
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
    out_dir = Path(f"aruc_outputs/quick_test{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = run_rts_aruc(
        horizon_hours=args.hours,
        rho=args.rho,
        enforce_lines=not args.no_lines,
        uncertainty_provider_path=args.uncertainty_npz,
        provider_start_idx=args.provider_start,
    )

    results = outputs["results"]
    data = outputs["data"]

    # Save standard outputs
    results["u"].to_csv(out_dir / "commitment_u.csv")
    results["p0"].to_csv(out_dir / "dispatch_p0.csv")
    results["Z"].to_csv(out_dir / "Z_coefficients.csv")

    # Save Sigma and rho
    np.save(out_dir / "Sigma.npy", outputs["Sigma"])
    np.save(out_dir / "rho.npy", np.atleast_1d(outputs["rho"]))

    # Z analysis
    z_analysis = analyze_Z(results["Z"], data, out_dir, rho=outputs["rho"])

    print(f"\nARUC-LDR objective: {results['obj']:,.2f}")
    print(f"Time-varying: {outputs['time_varying']}")
    print(f"\nResults saved to {out_dir}/")
    print("  commitment_u.csv, dispatch_p0.csv, Z_coefficients.csv")
    print("  Z_analysis_full.csv, Z_analysis_per_gen.csv")
    print("  Sigma.npy, rho.npy")
