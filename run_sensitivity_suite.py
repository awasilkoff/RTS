#!/usr/bin/env python3
"""
run_sensitivity_suite.py

Systematic sensitivity decomposition: runs multiple configurations of
run_comparison.py to isolate the contribution of each robustness feature.

Scenarios (2x2 matrix + baselines):
  1. full_robust        — lines + robust ramps (full model)
  2. lines_only         — lines, nominal ramps (isolates robust ramp value)
  3. ramps_only         — copperplate, robust ramps (isolates line value)
  4. stripped            — copperplate, nominal ramps (pure per-gen hedging)
  5. dam_reserve         — DAM + spinning reserve (system-level baseline)
  6. stripped_no_wcc     — like stripped but no worst-case cost epigraph
  7. stripped_free_z     — like stripped but wind Z optimized freely (not fixed)
  8. reserve_then_daruc  — DAM+Reserve commitment -> DARUC with lines + robust ramps

All scenarios share the same start time, rho, horizon, and uncertainty set.
Produces a combined CSV summary and prints a comparison table.

Usage:
    python run_sensitivity_suite.py --rho 2.0 --start-month 7 --start-day 15
    python run_sensitivity_suite.py --uncertainty-npz path/to/unc.npz --enforce-lines-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def build_base_args(args, script: str = "run_comparison.py") -> list[str]:
    """Common args shared across all scenarios."""
    base = [
        sys.executable, script,
        "--hours", str(args.hours),
        "--rho", str(args.rho),
        "--start-month", str(args.start_month),
        "--start-day", str(args.start_day),
        "--start-hour", str(args.start_hour),
        "--mip-gap", str(args.mip_gap),
        "--day2-interval", str(args.day2_interval),
        "--day1-only-robust",
        "--time-limit", str(args.time_limit),
        "--bar-qcp-conv-tol", str(args.bar_qcp_conv_tol),
    ]
    if args.uncertainty_npz:
        base += ["--uncertainty-npz", args.uncertainty_npz,
                 "--provider-start", str(args.provider_start)]
    if args.rho_lines_frac is not None:
        base += ["--rho-lines-frac", str(args.rho_lines_frac)]
    return base


SCENARIOS = [
    {
        "name": "full_robust",
        "desc": "Full model: lines + robust ramps",
        "extra": ["--enforce-lines", "--robust-ramp", "--with-reserve"],
    },
    {
        "name": "lines_only",
        "desc": "Lines enabled, nominal ramps",
        "extra": ["--enforce-lines", "--no-robust-ramp"],
    },
    {
        "name": "ramps_only",
        "desc": "Copperplate, robust ramps",
        "extra": ["--robust-ramp"],
    },
    {
        "name": "stripped",
        "desc": "Copperplate, nominal ramps (per-gen hedging only)",
        "extra": ["--no-robust-ramp"],
    },
    {
        "name": "dam_reserve",
        "desc": "DAM + spinning reserve baseline",
        "extra": ["--no-robust-ramp", "--with-reserve"],
    },
    {
        "name": "stripped_no_wcc",
        "desc": "Stripped + no worst-case cost epigraph",
        "extra": ["--no-robust-ramp", "--no-worst-case-cost"],
    },
    {
        "name": "stripped_free_z",
        "desc": "Stripped + wind Z free (not fixed to identity)",
        "extra": ["--no-robust-ramp", "--no-fix-wind-z"],
    },
    {
        "name": "reserve_then_daruc",
        "desc": "DAM+Reserve -> DARUC (lines + robust ramps, incremental obj)",
        "script": "run_reserve_then_daruc.py",  # uses different script
        "extra": [],
    },
]


def run_scenario(name: str, desc: str, base_args: list[str], extra: list[str],
                 out_root: Path,
                 alt_base_args: list[str] | None = None) -> Path:
    """Run a single scenario via subprocess.

    alt_base_args overrides base_args when the scenario uses a different script.
    """
    out_dir = out_root / name
    cmd = list(alt_base_args if alt_base_args is not None else base_args)
    cmd += extra + ["--out-dir", str(out_dir)]

    print(f"\n{'='*70}")
    print(f"SCENARIO: {name} — {desc}")
    print(f"{'='*70}")
    print(f"  CMD: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print(f"  WARNING: {name} exited with code {result.returncode}")
    return out_dir


def load_summary(out_dir: Path) -> dict | None:
    """Load comparison_summary or summary.json from a scenario output."""
    # run_comparison.py writes summary.json in aruc/ and daruc/ subdirs
    summaries = {}
    for subdir in ["aruc", "daruc", "dam_reserve"]:
        p = out_dir / subdir / "summary.json"
        if p.exists():
            with open(p) as f:
                summaries[subdir] = json.load(f)
    # run_reserve_then_daruc.py writes a top-level summary.json
    top_level = out_dir / "summary.json"
    if top_level.exists() and not summaries:
        with open(top_level) as f:
            summaries["_top"] = json.load(f)
    return summaries if summaries else None


def collect_results(out_root: Path, scenarios: list[dict]) -> list[dict]:
    """Collect key metrics from all scenario outputs."""
    rows = []
    for sc in scenarios:
        name = sc["name"]
        out_dir = out_root / name
        summaries = load_summary(out_dir)

        row = {"scenario": name, "description": sc["desc"]}
        if summaries:
            if "aruc" in summaries:
                row["aruc_obj"] = summaries["aruc"].get("objective")
            if "daruc" in summaries:
                row["daruc_obj"] = summaries["daruc"].get("daruc_objective")
                row["dam_obj"] = summaries["daruc"].get("dam_objective")
            if "dam_reserve" in summaries:
                row["reserve_obj"] = summaries["dam_reserve"].get("objective")
            # reserve_then_daruc writes a top-level summary with cost dicts
            if "_top" in summaries:
                top = summaries["_top"]
                if "reserve_cost" in top:
                    row["reserve_obj"] = top["reserve_cost"].get("total")
                if "daruc_cost" in top:
                    row["daruc_obj"] = top["daruc_cost"].get("total")
                if "extra_unit_hours_full_horizon" in top:
                    row["extra_unit_hours"] = top["extra_unit_hours_full_horizon"]
        rows.append(row)
    return rows


def print_comparison(rows: list[dict]):
    """Print a compact comparison table."""
    print("\n" + "=" * 90)
    print("SENSITIVITY SUITE SUMMARY (Gurobi objectives, full horizon)")
    print("=" * 90)
    print(f"{'Scenario':<20s} {'DAM':>14s} {'DAM+Res':>14s} {'DARUC':>14s} {'ARUC':>14s}")
    print("-" * 90)
    for r in rows:
        dam = f"{r['dam_obj']:>14,.0f}" if r.get("dam_obj") else f"{'--':>14s}"
        res = f"{r['reserve_obj']:>14,.0f}" if r.get("reserve_obj") else f"{'--':>14s}"
        daruc = f"{r['daruc_obj']:>14,.0f}" if r.get("daruc_obj") else f"{'--':>14s}"
        aruc = f"{r['aruc_obj']:>14,.0f}" if r.get("aruc_obj") else f"{'--':>14s}"
        print(f"{r['scenario']:<20s} {dam} {res} {daruc} {aruc}")
    print("=" * 90)
    print()

    # Print the interpretation guide
    print("Interpretation:")
    print("  full_robust - lines_only   = value of robust ramps (with lines)")
    print("  full_robust - ramps_only   = value of line constraints (with robust ramps)")
    print("  lines_only  - stripped     = value of line constraints (nominal ramps)")
    print("  ramps_only  - stripped     = value of robust ramps (copperplate)")
    print("  stripped    - dam_reserve  = value of per-gen Z hedging vs system reserve")
    print("  stripped    - stripped_no_wcc = value of worst-case cost epigraph")
    print("  stripped    - stripped_free_z = value of fixing wind Z to identity")
    print("  reserve_then_daruc           = robustness gap: extra commitments to make reserve fully robust")


def main():
    parser = argparse.ArgumentParser(
        description="Run sensitivity suite: decompose robustness feature contributions"
    )
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--start-month", type=int, default=7)
    parser.add_argument("--start-day", type=int, default=15)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--uncertainty-npz", type=str, default="uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha90.npz")
    parser.add_argument("--provider-start", type=int, default=2448)
    parser.add_argument("--rho-lines-frac", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--day2-interval", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=12000)
    parser.add_argument("--bar-qcp-conv-tol", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Root output directory (default: auto-generated)")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None,
                        help="Run only these scenarios (default: all). "
                             "Names: full_robust, lines_only, ramps_only, stripped, "
                             "dam_reserve, stripped_no_wcc, stripped_free_z, reserve_then_daruc")
    parser.add_argument("--resume", action="store_true",
                        help="Skip scenarios that already have output directories")
    args = parser.parse_args()

    tag = f"rho{args.rho}_{args.hours}h_m{args.start_month:02d}d{args.start_day:02d}"
    out_root = Path(args.out_dir) if args.out_dir else Path("sensitivity_suite") / tag
    out_root.mkdir(parents=True, exist_ok=True)

    base_args = build_base_args(args)

    # Filter scenarios if requested
    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s["name"] in args.scenarios]
        if not scenarios:
            print(f"ERROR: no matching scenarios. Available: "
                  f"{[s['name'] for s in SCENARIOS]}")
            sys.exit(1)

    # Run each scenario
    for sc in scenarios:
        sc_dir = out_root / sc["name"]
        if args.resume and sc_dir.exists():
            print(f"\nSkipping {sc['name']} (already exists, --resume)")
            continue

        # Scenarios with a different script get their own base args
        alt_base = None
        if "script" in sc:
            alt_base = build_base_args(args, script=sc["script"])

        run_scenario(
            sc["name"], sc["desc"], base_args, sc["extra"],
            out_root, alt_base_args=alt_base,
        )

    # Collect and display results
    rows = collect_results(out_root, SCENARIOS)

    # Save CSV
    csv_path = out_root / "sensitivity_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {csv_path}")

    print_comparison(rows)

    # Save run config
    with open(out_root / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nAll outputs in {out_root}/")


if __name__ == "__main__":
    main()
