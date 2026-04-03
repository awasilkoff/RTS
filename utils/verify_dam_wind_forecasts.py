#!/usr/bin/env python3
"""
Verify wind forecasts being used in DAM model.

This loads DAMData exactly as the runner scripts do and visualizes
the wind Pmax values to confirm they're using scaled SPP forecasts.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from io_rts import build_damdata_from_rts

# RTS wind capacities (nameplate)
RTS_WIND_CAPS = {
    "309_WIND_1": 148.3,
    "317_WIND_1": 799.1,
    "303_WIND_1": 847.0,
    "122_WIND_1": 713.5,
}
TOTAL_CAPACITY = sum(RTS_WIND_CAPS.values())

print("=" * 70)
print("DAM Wind Forecast Verification")
print("=" * 70)
print(f"\nRTS Wind Nameplate Capacities:")
for name, cap in RTS_WIND_CAPS.items():
    print(f"  {name}: {cap:.1f} MW")
print(f"  Total: {TOTAL_CAPACITY:.1f} MW\n")

# Build DAMData exactly as run_rts_dam.py does
print("Building DAMData (as DAM runner does)...")
start_time = pd.Timestamp(year=2020, month=7, day=15, hour=0)
horizon_hours = 48

source_dir = Path("RTS_Data/SourceData")
ts_dir = Path("RTS_Data/timeseries_data_files")
spp_forecasts = Path(
    "uncertainty_sets_refactored/data/"
    "forecasts_filtered_rts4_constellation_v2.parquet"
)

data = build_damdata_from_rts(
    source_dir=source_dir,
    ts_dir=ts_dir,
    start_time=start_time,
    horizon_hours=horizon_hours,
    spp_forecasts_parquet=spp_forecasts,
    spp_start_idx=0,
)

I = data.n_gens
T = data.n_periods

print(f"Loaded DAMData:")
print(f"  Generators: {I}")
print(f"  Time periods: {T}")
print(f"  Total hours: {data.total_hours}")

# Extract wind generators
wind_mask = np.array([gt == "WIND" for gt in data.gen_type])
wind_indices = np.where(wind_mask)[0]
wind_gen_ids = [data.gen_ids[i] for i in wind_indices]

print(f"\nWind generators found: {len(wind_indices)}")
for i in wind_indices:
    print(f"  [{i}] {data.gen_ids[i]}")

# Extract wind Pmax via Pmax_2d() method
Pmax_2d = data.Pmax_2d()  # (I, T) array
wind_forecasts = Pmax_2d[wind_indices, :]  # (n_wind, T)
times = pd.date_range(start=start_time, periods=T, freq="h")

# Compute system total
system_total = wind_forecasts.sum(axis=0)

print("\n" + "=" * 70)
print("Wind Forecast Statistics (MW)")
print("=" * 70)

print("\nPer-Farm Statistics:")
for idx, (i, gen_id) in enumerate(zip(wind_indices, wind_gen_ids)):
    fc = wind_forecasts[idx, :]
    cap = RTS_WIND_CAPS.get(gen_id, 0)
    print(f"\n{gen_id} (nameplate: {cap:.1f} MW):")
    print(f"  Mean:  {fc.mean():.1f} MW")
    print(f"  Min:   {fc.min():.1f} MW")
    print(f"  Max:   {fc.max():.1f} MW ({fc.max()/cap*100:.1f}% of capacity)")
    if fc.max() > cap * 1.01:
        print(f"  WARNING: Max exceeds nameplate by {fc.max()-cap:.1f} MW!")

print(f"\nSystem Total:")
print(f"  Mean:  {system_total.mean():.1f} MW")
print(f"  Min:   {system_total.min():.1f} MW")
print(f"  Max:   {system_total.max():.1f} MW "
      f"({system_total.max()/TOTAL_CAPACITY*100:.1f}% of capacity)")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if system_total.max() > 0.5 * TOTAL_CAPACITY:
    print(f"PASS: Forecasts appear SCALED")
    print(f"  System max = {system_total.max():.1f} MW "
          f"(expected range: 1000-2500 MW)")
    print(f"  This is {system_total.max()/TOTAL_CAPACITY*100:.0f}% "
          f"of total capacity.")
else:
    print(f"FAIL: Forecasts appear UNSCALED")
    print(f"  System max = {system_total.max():.1f} MW (too low!)")
    print(f"  Expected: 1000-2500 MW for scaled data")
    print(f"  Check: Are you using the v2 parquet file?")

print("=" * 70)

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Individual farm forecasts
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for idx, (gen_id, color) in enumerate(zip(wind_gen_ids, colors)):
    ax1.plot(times, wind_forecasts[idx, :], label=gen_id, color=color, linewidth=2)
    cap = RTS_WIND_CAPS.get(gen_id, 0)
    ax1.axhline(y=cap, color=color, linestyle="--", alpha=0.3, linewidth=1)

ax1.set_ylabel("Wind Forecast (MW)", fontsize=12)
ax1.set_title("Per-Farm Wind Forecasts (used in DAM)", fontsize=14, fontweight="bold")
ax1.legend(loc="best")
ax1.grid(True, alpha=0.3)

# Plot 2: System total
ax2.plot(times, system_total, color="navy", linewidth=2.5, label="System Total")
ax2.axhline(
    y=TOTAL_CAPACITY, color="red", linestyle="--", linewidth=2,
    label=f"Total Capacity ({TOTAL_CAPACITY:.0f} MW)", alpha=0.6,
)
ax2.fill_between(times, 0, system_total, alpha=0.2, color="navy")

ax2.set_xlabel("Time", fontsize=12)
ax2.set_ylabel("Total Wind Forecast (MW)", fontsize=12)
ax2.set_title("System-Wide Wind Forecast", fontsize=14, fontweight="bold")
ax2.legend(loc="best")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = Path("dam_wind_forecast_check.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {output_path.absolute()}")

plt.close()
