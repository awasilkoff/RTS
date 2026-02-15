#!/usr/bin/env python3
"""
Diagnose NPZ uncertainty set scaling.

Checks whether mu, Sigma, and rho are internally consistent and
at the correct scale for RTS wind generators.

Key relationship:
    rho = (sum(mu) - total_lower_bound) / sqrt(e^T Sigma e)
    Actual MW uncertainty = rho * sqrt(e^T Sigma e)
"""
import numpy as np
from pathlib import Path
import json

# RTS wind capacities (MW)
RTS_CAPS = {
    "309_WIND_1": 148.3,
    "317_WIND_1": 799.1,
    "303_WIND_1": 847.0,
    "122_WIND_1": 713.5,
}
TOTAL_CAP = sum(RTS_CAPS.values())

npz_path = Path("data/uncertainty_sets_rts4_v2_16d/sigma_rho.npz")
meta_path = npz_path.parent / "metadata.json"

print("=" * 70)
print("NPZ Uncertainty Set Diagnostic")
print("=" * 70)
print(f"\nRTS total wind capacity: {TOTAL_CAP:.1f} MW")
print(f"Loading: {npz_path}\n")

data = np.load(npz_path, allow_pickle=True)
mu = data["mu"]          # (T, K)
sigma = data["sigma"]    # (T, K, K)
rho = data["rho"]        # (T,)

T, K = mu.shape
e = np.ones(K)

if "y_cols" in data.files:
    wind_ids = [str(w) for w in data["y_cols"]]
else:
    wind_ids = [f"Wind_{i}" for i in range(K)]

print(f"Shape: mu={mu.shape}, sigma={sigma.shape}, rho={rho.shape}")
print(f"Wind IDs: {wind_ids}\n")

# =====================================================================
# 1. MU ANALYSIS (conditional mean from covariance model)
# =====================================================================
print("=" * 70)
print("1. MU (Conditional Mean) — Are these in MW or normalized?")
print("=" * 70)

mu_system = mu.sum(axis=1)  # Total across farms
print(f"\nPer-farm mu statistics:")
for k in range(K):
    wid = wind_ids[k] if k < len(wind_ids) else f"Farm_{k}"
    cap = RTS_CAPS.get(wid, 0)
    print(f"  {wid}: mean={mu[:, k].mean():.1f}, "
          f"min={mu[:, k].min():.1f}, max={mu[:, k].max():.1f}, "
          f"RTS_cap={cap:.1f}")

print(f"\nSystem total mu:")
print(f"  Mean: {mu_system.mean():.1f}")
print(f"  Min:  {mu_system.min():.1f}")
print(f"  Max:  {mu_system.max():.1f}")
print(f"  Ratio to RTS capacity: {mu_system.max() / TOTAL_CAP:.1%}")

if mu_system.max() > 500:
    print("  -> Mu appears to be in SCALED MW units")
elif mu_system.max() > 50:
    print("  -> Mu appears to be in UNSCALED MW units (raw SPP)")
else:
    print("  -> Mu appears to be NORMALIZED (not in MW)")

# =====================================================================
# 2. SIGMA ANALYSIS (covariance matrix)
# =====================================================================
print("\n" + "=" * 70)
print("2. SIGMA (Covariance Matrix) — Variance magnitude")
print("=" * 70)

# Diagonal = variance per farm
all_diag = np.array([np.diag(sigma[t]) for t in range(T)])
all_std = np.sqrt(all_diag)

print(f"\nPer-farm std dev (sqrt of diagonal) statistics (MW):")
for k in range(K):
    wid = wind_ids[k] if k < len(wind_ids) else f"Farm_{k}"
    print(f"  {wid}: mean={all_std[:, k].mean():.1f}, "
          f"min={all_std[:, k].min():.1f}, max={all_std[:, k].max():.1f}")

# Total system variance: e^T Sigma e
total_var = np.array([float(e @ sigma[t] @ e) for t in range(T)])
total_std = np.sqrt(total_var)

print(f"\nSystem total std (sqrt(e^T Sigma e)):")
print(f"  Mean: {total_std.mean():.1f}")
print(f"  Min:  {total_std.min():.1f}")
print(f"  Max:  {total_std.max():.1f}")

# Eigenvalue analysis
print(f"\nEigenvalue analysis (sample timesteps):")
for t in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
    eigs = np.linalg.eigvalsh(sigma[t])
    print(f"  t={t}: eigenvalues = {eigs}, condition = {eigs[-1]/max(eigs[0], 1e-15):.1f}")

# =====================================================================
# 3. RHO ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("3. RHO (Ellipsoid Radius) — Mahalanobis distance units")
print("=" * 70)

print(f"\nRho statistics:")
print(f"  Mean:   {rho.mean():.4f}")
print(f"  Median: {np.median(rho):.4f}")
print(f"  Std:    {rho.std():.4f}")
print(f"  Min:    {rho.min():.4f}")
print(f"  Max:    {rho.max():.4f}")
print(f"  % zero: {(rho == 0).sum() / T * 100:.1f}%")
print(f"  % < 0.1: {(rho < 0.1).sum() / T * 100:.1f}%")
print(f"  % < 1.0: {(rho < 1.0).sum() / T * 100:.1f}%")
print(f"  % > 2.0: {(rho > 2.0).sum() / T * 100:.1f}%")
print(f"  % > 5.0: {(rho > 5.0).sum() / T * 100:.1f}%")

# =====================================================================
# 4. IMPLIED MW UNCERTAINTY
# =====================================================================
print("\n" + "=" * 70)
print("4. IMPLIED MW UNCERTAINTY — rho * sqrt(e^T Sigma e)")
print("=" * 70)

mw_uncertainty = rho * total_std  # Actual MW range for system total

print(f"\nSystem-total MW uncertainty = rho * sqrt(e^T Sigma e):")
print(f"  Mean:   {mw_uncertainty.mean():.1f} MW")
print(f"  Median: {np.median(mw_uncertainty):.1f} MW")
print(f"  Max:    {mw_uncertainty.max():.1f} MW")
print(f"  Min:    {mw_uncertainty.min():.1f} MW")
print(f"  % < 10 MW:  {(mw_uncertainty < 10).sum() / T * 100:.1f}%")
print(f"  % < 50 MW:  {(mw_uncertainty < 50).sum() / T * 100:.1f}%")
print(f"  % < 100 MW: {(mw_uncertainty < 100).sum() / T * 100:.1f}%")
print(f"  % > 200 MW: {(mw_uncertainty > 200).sum() / T * 100:.1f}%")

print(f"\nAs % of total capacity ({TOTAL_CAP:.0f} MW):")
pct = mw_uncertainty / TOTAL_CAP * 100
print(f"  Mean:   {pct.mean():.1f}%")
print(f"  Max:    {pct.max():.1f}%")

# =====================================================================
# 5. BACK-CHECK: Reconstruct what conformal lower bound was
# =====================================================================
print("\n" + "=" * 70)
print("5. BACK-CHECK: Implied conformal lower bound")
print("=" * 70)
print("  rho = (sum(mu) - lower_bound) / sqrt(e^T Sigma e)")
print("  => lower_bound = sum(mu) - rho * sqrt(e^T Sigma e)")

implied_lower = mu_system - rho * total_std

print(f"\nImplied total lower bound:")
print(f"  Mean:   {implied_lower.mean():.1f} MW")
print(f"  Min:    {implied_lower.min():.1f} MW")
print(f"  Max:    {implied_lower.max():.1f} MW")
print(f"  % negative: {(implied_lower < 0).sum() / T * 100:.1f}%")

print(f"\nMu vs lower bound gap (= MW uncertainty):")
gap = mu_system - implied_lower
print(f"  Mean gap: {gap.mean():.1f} MW")
print(f"  Max gap:  {gap.max():.1f} MW")

# =====================================================================
# 6. COMPARISON WITH STATIC FALLBACK
# =====================================================================
print("\n" + "=" * 70)
print("6. COMPARISON: NPZ vs Static Fallback (15% std, rho=2.0)")
print("=" * 70)

# Static fallback uses 15% of average Pmax as std, rho=2.0
static_rho = 2.0
avg_pmax = np.array([mu[:, k].mean() for k in range(K)])
static_var = (0.15 * avg_pmax) ** 2
static_sigma = np.diag(static_var)
static_total_std = float(np.sqrt(e @ static_sigma @ e))
static_mw = static_rho * static_total_std

print(f"\nStatic (wind_std_fraction=0.15, rho=2.0):")
print(f"  Per-farm std: {np.sqrt(static_var)}")
print(f"  Total system std: {static_total_std:.1f} MW")
print(f"  MW uncertainty: {static_mw:.1f} MW ({static_mw/TOTAL_CAP*100:.1f}% of capacity)")

print(f"\nNPZ time-varying (median hour):")
med_idx = np.argsort(rho)[T // 2]
npz_total_std_med = total_std[med_idx]
npz_mw_med = mw_uncertainty[med_idx]
print(f"  rho: {rho[med_idx]:.3f}")
print(f"  Total system std: {npz_total_std_med:.1f} MW")
print(f"  MW uncertainty: {npz_mw_med:.1f} MW ({npz_mw_med/TOTAL_CAP*100:.1f}% of capacity)")

print(f"\nNPZ time-varying (95th percentile hour):")
p95_idx = np.argsort(rho)[int(T * 0.95)]
npz_total_std_95 = total_std[p95_idx]
npz_mw_95 = mw_uncertainty[p95_idx]
print(f"  rho: {rho[p95_idx]:.3f}")
print(f"  Total system std: {npz_total_std_95:.1f} MW")
print(f"  MW uncertainty: {npz_mw_95:.1f} MW ({npz_mw_95/TOTAL_CAP*100:.1f}% of capacity)")

# =====================================================================
# 7. POTENTIAL ISSUES
# =====================================================================
print("\n" + "=" * 70)
print("7. POTENTIAL ISSUES TO INVESTIGATE")
print("=" * 70)

issues = []

if mu_system.max() < 500:
    issues.append("MU SCALE: mu values too small - likely unscaled SPP data")

if total_std.mean() > 500:
    issues.append("SIGMA TOO LARGE: Very high variance dilutes rho")

if (rho == 0).sum() / T > 0.3:
    issues.append(f"MANY ZERO RHO: {(rho == 0).sum()}/{T} hours have rho=0 "
                  "(lower bound >= mean => no uncertainty)")

if mw_uncertainty.mean() < 20:
    issues.append("MW UNCERTAINTY TOO SMALL: Average <20 MW system uncertainty")

if np.median(rho) < 0.5 and total_std.mean() < 100:
    issues.append("BOTH rho AND Sigma small: data may be unscaled")

if np.median(rho) < 0.5 and total_std.mean() > 200:
    issues.append("LARGE Sigma + SMALL rho: conformal bounds very close to mean "
                  "(alpha too permissive? n_bins=1 too coarse?)")

if not issues:
    issues.append("No obvious issues detected - values look reasonable")

for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")

print("\n" + "=" * 70)
