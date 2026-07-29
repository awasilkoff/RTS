#!/usr/bin/env python3
"""Run every solve and post-processing step the NewRAMP paper needs, unattended.

Designed for an overnight run: it refuses to start on a network model known to be
broken, never silently continues past a failed stage, and finishes by stating
whether the results are trustworthy enough to publish.

Stages
------
  0  preflight   PTDF validation, Gurobi licence, provenance capture
  1  solves      reserve_then_daruc, then full_free_z (via run_sensitivity_suite)
  2  gate        every solve converged? line-resolve loop converged?
  3  post        worst-case flows, committed units, line-flow and paper figures
  4  report      four-case table + verdict, written to PAPER_RUN_REPORT.md

Why the preflight matters
-------------------------
``build_dc_ptdf`` previously assembled the PTDF slack column with a
distributed-slack pattern, putting entries up to 31.3 where a single-slack PTDF
is bounded by 1.0, and corrupting every line-flow constraint.  A run on that code
looks completely normal and produces publishable-looking numbers.  Stage 0 runs
``utils/verify_ptdf.py`` and refuses to proceed if it fails, so an overnight run
can never quietly produce another set of invalid results.

Usage
-----
    python run_paper_overnight.py
    python run_paper_overnight.py --out-dir sensitivity_suite/paper_final --time-limit 21600
    python run_paper_overnight.py --dry-run
    python run_paper_overnight.py --resume        # skip finished scenarios

The four reported cases come from two scenarios.  Note that **DAM w/Res is read
from reserve_then_daruc**, not from full_free_z/dam_reserve: Case 4 must be
compared against the first stage it actually amends, and the two solves of that
same problem differ inside the MIP gap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
PY = sys.executable

#: Scenario order.  reserve_then_daruc runs FIRST: it is the harder solve and the
#: one that has failed before, so an unattended run should discover that early
#: rather than after the other scenario has consumed hours.
SCENARIOS = ["reserve_then_daruc", "full_free_z"]

#: Canonical paper configuration.  Kept here rather than relying on argparse
#: defaults so the run is reproducible from this file alone.
CANONICAL = {
    "start_month": 7,
    "start_day": 15,
    "hours": 48,
    "day2_interval": 2,
    "uncertainty_npz": "uncertainty_sets_refactored/data/uncertainty_sets_rts4_v2_16d/sigma_rho_alpha99.npz",
    "provider_start": 2448,
    "line_monitor_threshold": 0.9,
    "mip_gap": 0.005,
}


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def banner(text: str, char: str = "=") -> None:
    print("\n" + char * 78)
    print(text)
    print(char * 78, flush=True)


def run(cmd: list[str], log_path: Path | None = None, label: str = "") -> int:
    """Run *cmd*, mirroring output to *log_path* in UTF-8.

    Windows consoles default to cp1252, which cannot encode the dashes in these
    scripts' summary text; a redirected run would die at reporting time, after
    the solves but before results are written.  Child processes force UTF-8
    themselves, and we write the log in UTF-8 too.
    """
    print(f"  $ {' '.join(cmd)}", flush=True)
    if log_path is None:
        return subprocess.run(cmd, cwd=str(ROOT)).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# {label}\n# {' '.join(cmd)}\n# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
        rc = proc.wait()
        fh.write(f"\n# exit={rc} after {time.time()-t0:.0f}s\n")
    print(f"  -> exit={rc} after {time.time()-t0:.0f}s   log: {log_path}", flush=True)
    return rc


def load_json(p: Path):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# stage 0 — preflight
# ---------------------------------------------------------------------------

def preflight(out_root: Path) -> list[str]:
    """Return a list of blocking problems; empty means good to go."""
    banner("STAGE 0 — PREFLIGHT")
    blockers: list[str] = []

    # Provenance: which code produced these results?
    for desc, cmd in (("commit", ["git", "rev-parse", "--short", "HEAD"]),
                      ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            val = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                                 text=True).stdout.strip()
            print(f"  {desc:8s} {val}")
        except OSError:
            print(f"  {desc:8s} (git unavailable)")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                           capture_output=True, text=True).stdout.strip()
    tracked_dirty = [l for l in dirty.splitlines() if not l.startswith("??")]
    if tracked_dirty:
        print(f"  WARNING: {len(tracked_dirty)} modified tracked file(s) — results "
              f"will not correspond to any commit:")
        for l in tracked_dirty[:8]:
            print(f"      {l}")

    # The one that matters: is the network model sane?
    print("\n  Validating DC PTDF (utils/verify_ptdf.py)...")
    rc = run([PY, "utils/verify_ptdf.py", "--trials", "200"],
             out_root / "logs" / "00_verify_ptdf.log", "PTDF validation")
    if rc != 0:
        blockers.append(
            "PTDF validation FAILED. The network model is wrong, so every "
            "line-flow, binding-line and robust-margin result this run would "
            "produce is invalid. Do not run overnight on this code."
        )

    # Gurobi licence: fail now, not after midnight.
    print("\n  Checking Gurobi...")
    probe = (
        "import gurobipy as gp;"
        "m=gp.Model();m.Params.OutputFlag=0;"
        "x=m.addVar(ub=1);m.setObjective(x,gp.GRB.MAXIMIZE);m.optimize();"
        "print('gurobi',gp.gurobi.version(),'status',m.Status)"
    )
    r = subprocess.run([PY, "-c", probe], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        blockers.append(f"Gurobi unavailable or unlicensed: {r.stderr.strip()[:300]}")
    else:
        print(f"    {r.stdout.strip()}")

    npz = ROOT / CANONICAL["uncertainty_npz"]
    if not npz.exists():
        blockers.append(f"Uncertainty set not found: {npz}")
    else:
        print(f"  uncertainty set present: {npz.name}")

    return blockers


# ---------------------------------------------------------------------------
# stage 1 — solves
# ---------------------------------------------------------------------------

def suite_cmd(scenario: str, out_root: Path, time_limit: float, resume: bool) -> list[str]:
    cmd = [
        PY, "run_sensitivity_suite.py",
        "--scenarios", scenario,
        "--start-month", str(CANONICAL["start_month"]),
        "--start-day", str(CANONICAL["start_day"]),
        "--hours", str(CANONICAL["hours"]),
        "--day2-interval", str(CANONICAL["day2_interval"]),
        "--uncertainty-npz", CANONICAL["uncertainty_npz"],
        "--provider-start", str(CANONICAL["provider_start"]),
        "--line-monitor-threshold", str(CANONICAL["line_monitor_threshold"]),
        "--mip-gap", str(CANONICAL["mip_gap"]),
        "--time-limit", str(time_limit),
        "--out-dir", str(out_root),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def solve_all(out_root: Path, time_limit: float, resume: bool) -> list[str]:
    banner("STAGE 1 — SOLVES")
    failures = []
    for i, sc in enumerate(SCENARIOS, 1):
        summary = out_root / sc / "summary.json"
        if resume and summary.exists():
            print(f"\n[{i}/{len(SCENARIOS)}] {sc}: already has summary.json — skipping (--resume)")
            continue
        banner(f"[{i}/{len(SCENARIOS)}] {sc}", "-")
        rc = run(suite_cmd(sc, out_root, time_limit, resume),
                 out_root / "logs" / f"1{i}_{sc}.log", f"scenario {sc}")
        # The suite only prints a warning when a scenario exits non-zero, and it
        # writes nothing until a solve completes -- so check for the artifact,
        # not just the exit code.
        if rc != 0:
            failures.append(f"{sc}: exited {rc}")
        if not summary.exists():
            failures.append(f"{sc}: produced no summary.json (solve did not complete)")
    return failures


# ---------------------------------------------------------------------------
# stage 2 — trust gate
# ---------------------------------------------------------------------------

def gate(out_root: Path) -> tuple[bool, list[str]]:
    """Check solve provenance. Returns (trustworthy, notes)."""
    banner("STAGE 2 — TRUST GATE")
    notes, ok = [], True
    for sc in SCENARIOS:
        d = load_json(out_root / sc / "summary.json")
        if d is None:
            notes.append(f"{sc}: no summary.json to check")
            ok = False
            continue
        print(f"\n  {sc}")
        diags = d.get("solve_diagnostics") or {}
        if not diags:
            notes.append(f"{sc}: no solve_diagnostics — cannot verify convergence")
            ok = False
        for name, v in diags.items():
            if v is None:
                continue
            gap = v.get("mip_gap_achieved")
            gap_s = f"{gap:.5f}" if isinstance(gap, (int, float)) else "n/a"
            rt = v.get("runtime_seconds") or 0.0
            print(f"    {name:9s} {v.get('status_name','?'):11s} "
                  f"converged={v.get('converged')} gap={gap_s} runtime={rt:.0f}s")
            if not v.get("converged"):
                notes.append(
                    f"{sc}/{name}: {v.get('status_name')} at gap {gap_s} — this cost "
                    f"is NOT a converged {CANONICAL['mip_gap']} solution"
                )
                ok = False
        hit = d.get("line_resolve_max_iter_hit")
        # run_comparison stores a dict per model; run_reserve_then_daruc a bool
        hits = hit.values() if isinstance(hit, dict) else [hit]
        if any(h is True for h in hits):
            notes.append(
                f"{sc}: line-resolve loop exhausted max_iter — the solution may "
                f"still violate worst-case line limits"
            )
            ok = False
        print(f"    line_resolve_max_iter_hit={hit}  iterations="
              f"{d.get('daruc_line_iterations', d.get('line_iterations'))}")
    return ok, notes


# ---------------------------------------------------------------------------
# stage 3 — post-processing
# ---------------------------------------------------------------------------

def post_process(out_root: Path) -> list[str]:
    banner("STAGE 3 — POST-PROCESSING")
    warns = []
    logs = out_root / "logs"

    steps = [
        ("worst-case line flows",
         [PY, "utils/backfill_worst_case_flows.py", "--scan-dir", str(out_root)],
         "30_backfill_flows.log"),
        ("committed units / binding lines",
         [PY, "utils/count_committed_units.py", str(out_root),
          "--out", str(out_root / "case_summary.csv")],
         "31_count_units.log"),
        ("line-flow figures",
         [PY, "plot_line_flows.py", "--case-dir", str(out_root / "reserve_then_daruc")],
         "32_plot_line_flows.log"),
    ]
    if (ROOT / "make_paper_figures.py").exists():
        steps.append(("paper figure suite",
                      [PY, "make_paper_figures.py", "--scenario-dir", str(out_root),
                       "--out-dir", str(out_root / "paper_figures")],
                      "33_paper_figures.log"))
    else:
        warns.append("make_paper_figures.py not present — paper figure suite skipped "
                     "(it is untracked, so a fresh clone will not have it)")

    for label, cmd, logname in steps:
        banner(label, "-")
        if run(cmd, logs / logname, label) != 0:
            warns.append(f"post-processing step failed: {label} (see logs/{logname})")
    return warns


# ---------------------------------------------------------------------------
# stage 4 — report
# ---------------------------------------------------------------------------

def _k(x):
    return f"{x/1000:,.0f}" if isinstance(x, (int, float)) else "--"


def build_report(out_root: Path, trustworthy: bool, notes: list[str],
                 warns: list[str], elapsed: float) -> str:
    ffz = load_json(out_root / "full_free_z" / "summary.json") or {}
    rtd = load_json(out_root / "reserve_then_daruc" / "summary.json") or {}

    # Case -> (cost dict, unit-hours, curtailment).  DAM w/Res deliberately comes
    # from reserve_then_daruc, the first stage Case 4 actually amends.
    dam = ffz.get("dam_cost")
    ruc = ffz.get("daruc_cost")
    res = rtd.get("reserve_cost")
    resruc = rtd.get("daruc_cost")

    uh = rtd.get("unit_hours") or {}
    wc = rtd.get("wind_curtailment_mwh") or {}
    rows = [
        ("DAM", dam, (ffz.get("dam_metrics") or {}).get("unit_hours"),
         (ffz.get("dam_metrics") or {}).get("wind_curtailment_mwh")),
        ("DAM w/Res", res, uh.get("reserve"), wc.get("reserve")),
        ("DAM + RUC", ruc, (ffz.get("daruc_metrics") or {}).get("unit_hours"),
         (ffz.get("daruc_metrics") or {}).get("wind_curtailment_mwh")),
        ("DAM w/Res + RUC", resruc, uh.get("daruc"), wc.get("daruc")),
    ]

    L = []
    L.append("# NewRAMP paper run report")
    L.append("")
    L.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- output: `{out_root}`")
    L.append(f"- wall time: {elapsed/3600:.2f} h")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                                capture_output=True, text=True).stdout.strip()
        L.append(f"- commit: `{commit}`")
    except OSError:
        pass
    L.append("")
    L.append(f"## Verdict: {'TRUSTWORTHY' if trustworthy else 'DO NOT PUBLISH'}")
    L.append("")
    if notes:
        L.append("Blocking problems:")
        L.extend(f"- {n}" for n in notes)
    else:
        L.append("All solves converged and the line-resolve loop converged in every case.")
    if warns:
        L.append("")
        L.append("Non-blocking warnings:")
        L.extend(f"- {w}" for w in warns)
    L.append("")

    L.append("## Cost table (day 1, $K)")
    L.append("")
    L.append("| | " + " | ".join(r[0] for r in rows) + " |")
    L.append("|---|" + "---|" * len(rows))
    for field, label in (("total", "**Total**"), ("energy", "Energy"),
                         ("commitment", "Commitment"), ("startup", "— Startup"),
                         ("no_load", "— Min. Load")):
        L.append(f"| {label} | " + " | ".join(
            _k((r[1] or {}).get(field)) for r in rows) + " |")
    L.append("| (Unit-hours) | " + " | ".join(
        f"{r[2]:,.0f}" if isinstance(r[2], (int, float)) else "--" for r in rows) + " |")
    L.append("| (Curtail MWh) | " + " | ".join(
        f"{r[3]:,.0f}" if isinstance(r[3], (int, float)) else "--" for r in rows) + " |")
    L.append("")

    if all(isinstance((r[1] or {}).get("total"), (int, float)) for r in rows):
        t = [(r[1] or {})["total"] for r in rows]
        L.append("## Key increments ($K)")
        L.append("")
        L.append(f"- DAM w/Res − DAM: {(t[1]-t[0])/1000:+,.0f}")
        L.append(f"- DAM + RUC − DAM: {(t[2]-t[0])/1000:+,.0f}")
        L.append(f"- DAM w/Res + RUC − DAM w/Res: {(t[3]-t[1])/1000:+,.0f}")
        L.append(f"- **DAM w/Res + RUC − DAM + RUC: {(t[3]-t[2])/1000:+,.0f}**  "
                 f"(the locational-inefficiency result)")
        L.append("")
        span = abs(t[3] - t[2]) / t[3] * 100
        L.append(f"That headline increment is {span:.2f}% of total cost. Compare it "
                 f"against the achieved MIP gaps above: if the gap is of the same "
                 f"order, the increment is not resolvable from the optimisation and "
                 f"must not be reported as a finding.")
        L.append("")

    L.append("## Solve diagnostics")
    L.append("")
    for sc in SCENARIOS:
        d = load_json(out_root / sc / "summary.json") or {}
        L.append(f"### {sc}")
        L.append("")
        L.append("| solve | status | converged | achieved gap | runtime (s) |")
        L.append("|---|---|---|---|---|")
        for name, v in (d.get("solve_diagnostics") or {}).items():
            if v is None:
                continue
            g = v.get("mip_gap_achieved")
            L.append(f"| {name} | {v.get('status_name')} | {v.get('converged')} | "
                     f"{g:.5f} | {(v.get('runtime_seconds') or 0):.0f} |"
                     if isinstance(g, (int, float)) else
                     f"| {name} | {v.get('status_name')} | {v.get('converged')} | n/a | "
                     f"{(v.get('runtime_seconds') or 0):.0f} |")
        L.append("")
        ts = d.get("timings_seconds") or {}
        if ts:
            L.append("Timings: " + ", ".join(f"{k}={v:.0f}s" for k, v in ts.items()))
            L.append("")

    L.append("## Artifacts")
    L.append("")
    L.append("- `case_summary.csv` — committed units and binding line-hours per case")
    L.append("- `*/worst_case_flow_analysis_*.csv` — reserve vs LDR dispatch at worst-case wind")
    L.append("- `reserve_then_daruc/fig_*.png` — line-flow and reserve figures")
    L.append("- `paper_figures/` — figure suite (if make_paper_figures.py was present)")
    L.append("- `logs/` — full solver output per stage")
    return "\n".join(L)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="sensitivity_suite/paper_run",
                    help="Output root (default: sensitivity_suite/paper_run)")
    ap.add_argument("--time-limit", type=float, default=21600,
                    help="Gurobi time limit per solve, seconds (default: 21600 = 6 h)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip scenarios that already have a summary.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit without solving")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="Bypass PTDF/Gurobi checks. Do not use for paper runs.")
    ap.add_argument("--quick", action="store_true",
                    help="Validate the whole pipeline on an 8 h horizon in minutes. "
                         "Exercises every stage end-to-end. NOT paper numbers.")
    args = ap.parse_args()

    if args.quick:
        CANONICAL.update(hours=8, day2_interval=1, mip_gap=0.02)
        args.time_limit = min(args.time_limit, 120)
        if args.out_dir == ap.get_default("out_dir"):
            args.out_dir = "sensitivity_suite/paper_run_quick"

    out_root = Path(args.out_dir)
    t0 = time.time()

    banner("NewRAMP PAPER RUN")
    print(f"  output      {out_root}")
    print(f"  time limit  {args.time_limit:.0f}s per solve "
          f"({args.time_limit/3600:.1f} h; worst case "
          f"{len(SCENARIOS)*args.time_limit/3600:.1f} h total)")
    print(f"  scenarios   {' -> '.join(SCENARIOS)}")
    for k, v in CANONICAL.items():
        print(f"  {k:24s} {v}")

    if args.dry_run:
        print("\n  --dry-run: commands that would run\n")
        for sc in SCENARIOS:
            print("   ", " ".join(suite_cmd(sc, out_root, args.time_limit, args.resume)))
        return 0

    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        blockers = preflight(out_root)
        if blockers:
            banner("PREFLIGHT FAILED — NOT STARTING", "!")
            for b in blockers:
                print(f"  - {b}")
            return 2
        print("\n  Preflight OK.")
    else:
        print("\n  WARNING: preflight skipped (--skip-preflight).")

    failures = solve_all(out_root, args.time_limit, args.resume)
    if failures:
        banner("SOLVE STAGE FAILED", "!")
        for f in failures:
            print(f"  - {f}")
        print("\n  Skipping post-processing; re-run with --resume after fixing.")
        (out_root / "PAPER_RUN_REPORT.md").write_text(
            build_report(out_root, False, failures, [], time.time() - t0),
            encoding="utf-8")
        return 1

    trustworthy, notes = gate(out_root)
    warns = post_process(out_root)

    report = build_report(out_root, trustworthy, notes, warns, time.time() - t0)
    rp = out_root / "PAPER_RUN_REPORT.md"
    rp.write_text(report, encoding="utf-8")

    banner(f"DONE in {(time.time()-t0)/3600:.2f} h — "
           f"{'TRUSTWORTHY' if trustworthy else 'DO NOT PUBLISH'}")
    print(report)
    print(f"\nReport written to {rp}")
    return 0 if trustworthy else 1


if __name__ == "__main__":
    sys.exit(main())
