"""
Interactive visualization of uncertainty sets (rho, Sigma, forecasts) from NPZ file.

Helps identify interesting days for ARUC/DARUC testing by showing:
- Time-varying rho (uncertainty budget)
- Mean wind forecast across all farms
- Individual farm forecasts
- Covariance structure (trace of Sigma)

Usage:
    python visualize_uncertainty_npz.py [--npz-path PATH]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_npz(npz_path: Path):
    """Load NPZ file with uncertainty sets."""
    data = np.load(npz_path, allow_pickle=True)

    mu = data["mu"]  # (T, K) - mean forecasts
    sigma = data["sigma"]  # (T, K, K) - covariance matrices
    rho = data["rho"]  # (T,) - ellipsoid radius
    times = pd.to_datetime(data["times"])

    # Extract wind IDs if available
    if "y_cols" in data.files:
        wind_ids = [str(w) for w in data["y_cols"]]
    else:
        K = mu.shape[1]
        wind_ids = [f"Wind_{i}" for i in range(K)]

    return {
        "mu": mu,
        "sigma": sigma,
        "rho": rho,
        "times": times,
        "wind_ids": wind_ids,
    }


def compute_metrics(data_dict):
    """Compute additional metrics for visualization."""
    mu = data_dict["mu"]
    sigma = data_dict["sigma"]
    rho = data_dict["rho"]
    times = data_dict["times"]

    T, K = mu.shape

    # System-level metrics
    system_mean = mu.sum(axis=1)  # Total wind forecast

    # Covariance trace (total variance)
    sigma_trace = np.array([np.trace(sigma[t]) for t in range(T)])

    # Determinant (volume of uncertainty ellipsoid)
    sigma_det = np.array([np.linalg.det(sigma[t]) for t in range(T)])

    # Max eigenvalue (principal uncertainty direction)
    sigma_max_eig = np.array([np.linalg.eigvalsh(sigma[t])[-1] for t in range(T)])

    return {
        "system_mean": system_mean,
        "sigma_trace": sigma_trace,
        "sigma_det": sigma_det,
        "sigma_max_eig": sigma_max_eig,
    }


def create_interactive_plot(data_dict, metrics):
    """Create interactive Plotly figure with multiple subplots."""
    times = data_dict["times"]
    rho = data_dict["rho"]
    mu = data_dict["mu"]
    wind_ids = data_dict["wind_ids"]

    system_mean = metrics["system_mean"]
    sigma_trace = metrics["sigma_trace"]

    # Create figure with 3 subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Ellipsoid Radius (ρ) - Uncertainty Budget",
            "System-Wide Mean Forecast (Total MW)",
            "Per-Farm Mean Forecasts (MW)",
        ),
        vertical_spacing=0.08,
        row_heights=[0.3, 0.3, 0.4],
    )

    # Subplot 1: Rho (uncertainty budget)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=rho,
            mode="lines",
            name="ρ (radius)",
            line=dict(color="red", width=2),
            hovertemplate="<b>%{x}</b><br>ρ = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: System mean forecast
    fig.add_trace(
        go.Scatter(
            x=times,
            y=system_mean,
            mode="lines",
            name="Total Forecast",
            line=dict(color="blue", width=2),
            hovertemplate="<b>%{x}</b><br>Total = %{y:.1f} MW<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Subplot 3: Per-farm forecasts
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for k, wind_id in enumerate(wind_ids):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=mu[:, k],
                mode="lines",
                name=wind_id,
                line=dict(color=colors[k % len(colors)], width=1.5),
                hovertemplate=f"<b>%{{x}}</b><br>{wind_id}: %{{y:.1f}} MW<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # Add vertical lines for day boundaries (midnight)
    midnight_mask = (times.hour == 0) & (times.minute == 0)
    midnight_times = times[midnight_mask]

    for midnight in midnight_times:
        for row in [1, 2, 3]:
            fig.add_vline(
                x=midnight,
                line_width=1,
                line_dash="dash",
                line_color="gray",
                opacity=0.3,
                row=row,
                col=1,
            )

    # Update axes labels
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="ρ", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="MW", row=3, col=1)

    # Update layout
    fig.update_layout(
        height=1000,
        title_text="Time-Varying Uncertainty Sets - Interactive Explorer",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def print_summary_stats(data_dict, metrics):
    """Print summary statistics to help identify interesting periods."""
    times = data_dict["times"]
    rho = data_dict["rho"]
    mu = data_dict["mu"]
    system_mean = metrics["system_mean"]

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nTime range: {times[0]} to {times[-1]}")
    print(f"Total hours: {len(times)}")

    print(f"\nRho (uncertainty budget):")
    print(f"  Mean:   {rho.mean():.3f}")
    print(f"  Std:    {rho.std():.3f}")
    print(f"  Min:    {rho.min():.3f} (at {times[rho.argmin()]})")
    print(f"  Max:    {rho.max():.3f} (at {times[rho.argmax()]})")

    print(f"\nSystem forecast (total MW):")
    print(f"  Mean:   {system_mean.mean():.1f}")
    print(f"  Std:    {system_mean.std():.1f}")
    print(f"  Min:    {system_mean.min():.1f} (at {times[system_mean.argmin()]})")
    print(f"  Max:    {system_mean.max():.1f} (at {times[system_mean.argmax()]})")

    print(f"\n" + "=" * 70)
    print("INTERESTING PERIODS (for testing)")
    print("=" * 70)

    # Find periods with high rho (high uncertainty)
    top_rho_idx = np.argsort(rho)[-10:][::-1]
    print("\nTop 10 hours by rho (highest uncertainty):")
    for i, idx in enumerate(top_rho_idx, 1):
        print(
            f"  {i:2d}. {times[idx]}  ρ={rho[idx]:.3f}  forecast={system_mean[idx]:.1f} MW"
        )

    # Find periods with high wind and high rho (stress test)
    wind_rho_product = system_mean * rho
    top_stress_idx = np.argsort(wind_rho_product)[-10:][::-1]
    print("\nTop 10 hours by windxrho (high wind + high uncertainty):")
    for i, idx in enumerate(top_stress_idx, 1):
        print(
            f"  {i:2d}. {times[idx]}  wind={system_mean[idx]:.1f} MW  ρ={rho[idx]:.3f}"
        )

    # Find periods with low rho (deterministic-like)
    low_rho_idx = np.argsort(rho)[:10]
    print("\nTop 10 hours by low rho (most deterministic):")
    for i, idx in enumerate(low_rho_idx, 1):
        print(
            f"  {i:2d}. {times[idx]}  ρ={rho[idx]:.3f}  forecast={system_mean[idx]:.1f} MW"
        )

    # Find days (48h windows) with highest average rho
    print("\nBest 48h windows for testing (high average rho):")
    window_size = 48
    avg_rho_48h = np.array(
        [rho[i : i + window_size].mean() for i in range(len(rho) - window_size + 1)]
    )
    top_windows = np.argsort(avg_rho_48h)[-5:][::-1]
    for i, idx in enumerate(top_windows, 1):
        start_time = times[idx]
        end_time = times[idx + window_size - 1]
        avg_rho = avg_rho_48h[idx]
        avg_wind = system_mean[idx : idx + window_size].mean()
        print(f"  {i}. {start_time.date()} to {end_time.date()}")
        print(f"     Avg ρ={avg_rho:.3f}, Avg wind={avg_wind:.1f} MW")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize uncertainty sets from NPZ file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=Path(__file__).parent
        / "data"
        / "uncertainty_sets_rts4_v2_16d"
        / "sigma_rho_alpha90.npz",
        help="Path to NPZ file",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Save interactive plot to HTML (default: show in browser)",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.npz_path}...")
    data_dict = load_npz(args.npz_path)

    print(
        f"Loaded {len(data_dict['times'])} hours for {len(data_dict['wind_ids'])} wind farms"
    )
    print(f"Wind farms: {', '.join(data_dict['wind_ids'])}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(data_dict)

    # Print summary
    print_summary_stats(data_dict, metrics)

    # Create plot
    print("\nCreating interactive plot...")
    fig = create_interactive_plot(data_dict, metrics)

    # Show or save
    if args.output_html:
        print(f"\nSaving to {args.output_html}...")
        fig.write_html(args.output_html)
        print(f"(ok) Saved. Open in browser: file://{args.output_html.absolute()}")
    else:
        print("\nOpening in browser...")
        fig.show()


if __name__ == "__main__":
    main()
