#!/usr/bin/env python3
"""Quick check: Are Sigma values in the NPZ scaled to RTS units?"""
import numpy as np
from pathlib import Path

# RTS wind capacities (MW)
RTS_CAPACITIES = {
    "309_WIND_1": 148.3,
    "317_WIND_1": 799.1,
    "303_WIND_1": 847.0,
    "122_WIND_1": 713.5,
}
TOTAL_CAPACITY = sum(RTS_CAPACITIES.values())

print("="*70)
print("NPZ Sigma Verification - RTS Scaling Check")
print("="*70)
print(f"\nRTS Wind Capacities:")
for name, cap in RTS_CAPACITIES.items():
    print(f"  {name}: {cap:.1f} MW")
print(f"  Total: {TOTAL_CAPACITY:.1f} MW\n")

npz_path = Path("uncertainty_sets_refactored/data/uncertainty_sets_rts4/sigma_rho.npz")

if not npz_path.exists():
    print(f"ERROR: NPZ file not found at {npz_path}")
    exit(1)

print(f"Loading NPZ from: {npz_path}\n")
data = np.load(npz_path)

print(f"NPZ Contents:")
print(f"  Keys: {list(data.keys())}")
print(f"  mu shape: {data['mu'].shape}")
print(f"  Sigma shape: {data['sigma'].shape}")
print(f"  rho shape: {data['rho'].shape}")
print()

# Check first few timesteps
print("-"*70)
print("Sigma Analysis (first 5 timesteps)")
print("-"*70)

for t in range(min(5, data['sigma'].shape[0])):
    sigma_t = data['sigma'][t]
    diag = np.diag(sigma_t)
    std_dev = np.sqrt(diag)

    print(f"\nTimestep {t}:")
    print(f"  Variance (diagonal): {diag}")
    print(f"  Std Dev (sqrt):      {std_dev}")
    print(f"  Mean std dev: {std_dev.mean():.1f} MW")
    print(f"  Max std dev:  {std_dev.max():.1f} MW")

# Overall statistics
print("\n" + "="*70)
print("Overall Statistics Across All Timesteps")
print("="*70)

all_diag = np.array([np.diag(data['sigma'][t]) for t in range(data['sigma'].shape[0])])
all_std = np.sqrt(all_diag)

print(f"\nStd Dev per farm (MW):")
print(f"  Min across all times: {all_std.min(axis=0)}")
print(f"  Mean across all times: {all_std.mean(axis=0)}")
print(f"  Max across all times: {all_std.max(axis=0)}")
print(f"\nOverall mean std dev: {all_std.mean():.1f} MW")
print(f"Overall max std dev:  {all_std.max():.1f} MW")

# Check rho values
print("\n" + "="*70)
print("Rho (Ellipsoid Radius) Statistics")
print("="*70)
print(f"  Min rho: {data['rho'].min():.3f}")
print(f"  Mean rho: {data['rho'].mean():.3f}")
print(f"  Median rho: {np.median(data['rho']):.3f}")
print(f"  Max rho: {data['rho'].max():.3f}")

# Verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)

avg_std = all_std.mean()
if avg_std > 20:
    print(f"✓ PASS: Sigma appears to use SCALED data (avg std = {avg_std:.1f} MW)")
    print("  Expected range for scaled data: 30-200 MW")
    print("  This is consistent with RTS-scaled wind uncertainty.")
else:
    print(f"✗ FAIL: Sigma appears to use UNSCALED data (avg std = {avg_std:.1f} MW)")
    print("  Unscaled data would have std dev ~2-10 MW")
    print("  You may need to regenerate with --data-version v2")

print("="*70)
