"""
Experiment: Compare different weighting schemes for conformal prediction.

Tests various approaches to weight calibration points:
1. Uniform (binned conformal baseline)
2. Feature-based kernel (current weighted conformal)
3. Time-based exponential decay (recency weighting)
4. Time-based sliding window (recent points only)
5. Combined feature + time kernel (hybrid approach)

Key question: Is temporal drift significant enough to warrant time-based weighting?

Usage:
    python experiment_weighting_schemes.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from conformal_prediction import (
    train_wind_lower_model_conformal_binned,
    weighted_quantile,
    _compute_kernel_distances,
    _sanitize_scale,
    WeightedConformalLowerBundle,
)
from data_processing import build_conformal_totals_df
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor


def compute_time_weights_exponential(
    times_cal: pd.DatetimeIndex,
    times_query: pd.DatetimeIndex,
    half_life_days: float = 30.0,
    causal: bool = True,
    min_lag_days: float = 0.0,
) -> np.ndarray:
    """
    Compute exponential decay weights based on time distance.

    Weights decay exponentially: w = exp(-λ * Δt)
    where λ = ln(2) / half_life

    Recent calibration points get more weight than older ones.

    Parameters
    ----------
    times_cal : pd.DatetimeIndex, shape (n_cal,)
        Calibration timestamps
    times_query : pd.DatetimeIndex, shape (n_query,)
        Query timestamps
    half_life_days : float
        Time (in days) for weight to decay to 50%
    causal : bool, default=True
        If True, only use calibration data BEFORE query time (realistic for day-ahead)
        If False, use all calibration data (symmetric window, for experiments only)
    min_lag_days : float, default=0.0
        Minimum time lag (in days) between calibration and query
        E.g., min_lag_days=1.0 means "only use data from yesterday or earlier"
        Useful for operational day-ahead: can't use same-day data

    Returns
    -------
    weights : np.ndarray, shape (n_query, n_cal)
        Time-based weights for each query-calibration pair
        Weights are 0 for future calibration points (if causal=True)
    """
    # Convert to numeric (days since epoch)
    t_cal = (times_cal - pd.Timestamp('2020-01-01', tz='UTC')).total_seconds().values / 86400.0
    t_query = (times_query - pd.Timestamp('2020-01-01', tz='UTC')).total_seconds().values / 86400.0

    # Time differences (n_query, n_cal)
    # Positive if calibration point is before query (usual case)
    dt = t_query[:, np.newaxis] - t_cal[np.newaxis, :]

    # Exponential decay: w = exp(-λ * dt) for dt > 0
    # where λ = ln(2) / half_life
    lam = np.log(2.0) / half_life_days

    if causal:
        # Only use past calibration data
        # dt > 0: calibration is in the past -> apply decay
        # dt <= 0: calibration is in the future -> weight = 0
        weights = np.where(
            dt > min_lag_days,
            np.exp(-lam * (dt - min_lag_days)),  # Decay from (t - min_lag)
            0.0  # Future points or within min_lag get zero weight
        )
    else:
        # Non-causal (for experiments): use absolute time distance
        weights = np.exp(-lam * np.abs(dt))

    return weights


def compute_time_weights_sliding_window(
    times_cal: pd.DatetimeIndex,
    times_query: pd.DatetimeIndex,
    window_days: float = 30.0,
    causal: bool = True,
    min_lag_days: float = 0.0,
) -> np.ndarray:
    """
    Compute sliding window weights (0/1 based on time distance).

    Only calibration points within the time window get weight 1, others get 0.

    Parameters
    ----------
    times_cal : pd.DatetimeIndex, shape (n_cal,)
        Calibration timestamps
    times_query : pd.DatetimeIndex, shape (n_query,)
        Query timestamps
    window_days : float
        Window size in days (points within window get weight 1)
    causal : bool, default=True
        If True, only use calibration data BEFORE query time (realistic for day-ahead)
        Window is backwards-looking: [t - window_days, t - min_lag_days]
    min_lag_days : float, default=0.0
        Minimum time lag (in days) between calibration and query
        E.g., min_lag_days=1.0 means "only use data from yesterday or earlier"

    Returns
    -------
    weights : np.ndarray, shape (n_query, n_cal)
        Binary weights (0 or 1)
    """
    # Convert to numeric (days)
    t_cal = (times_cal - pd.Timestamp('2020-01-01', tz='UTC')).total_seconds().values / 86400.0
    t_query = (times_query - pd.Timestamp('2020-01-01', tz='UTC')).total_seconds().values / 86400.0

    # Time differences (positive if cal is in past)
    dt = t_query[:, np.newaxis] - t_cal[np.newaxis, :]

    if causal:
        # Only use past calibration data within window
        # Window: [min_lag_days, min_lag_days + window_days] days ago
        weights = ((dt >= min_lag_days) & (dt <= min_lag_days + window_days)).astype(float)
    else:
        # Non-causal: symmetric window
        dt_abs = np.abs(dt)
        weights = (dt_abs <= window_days).astype(float)

    return weights


def train_time_weighted_conformal(
    df: pd.DataFrame,
    weighting: Literal['exponential', 'sliding_window', 'combined'],
    *,
    time_col: str = 'TIME_HOURLY',
    feature_cols: list[str] = None,
    kernel_feature_cols: list[str] = None,
    scale_col: str = 'ens_std',
    alpha_target: float = 0.95,
    half_life_days: float = 30.0,
    window_days: float = 30.0,
    omega: np.ndarray = None,
    tau: float = 5.0,
    causal: bool = True,
    min_lag_days: float = 0.0,
    safety_margin: float = 0.0,
    split_method: str = 'random',
    random_seed: int = 42,
    test_frac: float = 0.2,
    cal_frac: float = 0.2,
):
    """
    Train conformal prediction with time-based weighting schemes.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with features, targets, and timestamps
    weighting : {'exponential', 'sliding_window', 'combined'}
        Weighting scheme:
        - 'exponential': Exponential decay based on time distance
        - 'sliding_window': Binary weights within time window
        - 'combined': Feature kernel x time decay (hybrid)
    time_col : str
        Column name for timestamps
    feature_cols : list[str]
        Features for quantile model
    kernel_feature_cols : list[str]
        Features for kernel weighting (if weighting='combined')
    scale_col : str
        Column for nonconformity scaling
    alpha_target : float
        Target coverage level
    half_life_days : float
        Half-life for exponential decay (if weighting='exponential' or 'combined')
    window_days : float
        Window size for sliding window (if weighting='sliding_window')
    omega : np.ndarray
        Feature weights for kernel (if weighting='combined')
    tau : float
        Bandwidth for feature kernel (if weighting='combined')
    split_method : str
        'random' or 'time_ordered'
    random_seed : int
        Random seed for reproducibility
    test_frac : float
        Fraction of data for test set
    cal_frac : float
        Fraction of data for calibration set

    Returns
    -------
    bundle : WeightedConformalLowerBundle or similar
        Trained model bundle
    metrics : dict
        Evaluation metrics
    df_test : pd.DataFrame
        Test predictions
    """
    if feature_cols is None:
        feature_cols = ['ens_mean', 'ens_std']
    if kernel_feature_cols is None:
        kernel_feature_cols = ['SYS_MEAN', 'SYS_STD']

    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    df = df[df['y'].notna()].reset_index(drop=True)

    # Extract arrays
    X = df[feature_cols]
    y = df['y'].to_numpy(dtype=float)
    times = pd.to_datetime(df[time_col])
    scale = _sanitize_scale(df[scale_col].to_numpy(dtype=float), min_scale=1e-3)

    # Split data
    n = len(df)
    n_test = int(n * test_frac)
    n_cal = int(n * cal_frac)
    n_train = n - n_test - n_cal

    if split_method == 'time_ordered':
        idx_train = np.arange(n_train)
        idx_cal = np.arange(n_train, n_train + n_cal)
        idx_test = np.arange(n_train + n_cal, n)
    elif split_method == 'random':
        rng = np.random.RandomState(random_seed)
        idx = rng.permutation(n)
        idx_train = idx[:n_train]
        idx_cal = idx[n_train:n_train + n_cal]
        idx_test = idx[n_train + n_cal:]
    else:
        raise ValueError(f"split_method must be 'time_ordered' or 'random'")

    X_train, y_train = X.iloc[idx_train], y[idx_train]
    X_cal, y_cal = X.iloc[idx_cal], y[idx_cal]
    X_test, y_test = X.iloc[idx_test], y[idx_test]

    scale_cal = scale[idx_cal]
    scale_test = scale[idx_test]

    # Convert to DatetimeIndex (not Series)
    times_cal = pd.DatetimeIndex(times.iloc[idx_cal])
    times_test = pd.DatetimeIndex(times.iloc[idx_test])

    # Train quantile model
    model = LGBMRegressor(
        objective='quantile',
        alpha=1.0 - alpha_target,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_cal = np.asarray(model.predict(X_cal), dtype=float)
    y_pred_test = np.asarray(model.predict(X_test), dtype=float)

    # Nonconformity scores (one-sided for lower bound)
    r_cal = np.maximum(0.0, (y_pred_cal - y_cal) / scale_cal)

    # Compute weights based on scheme
    if weighting == 'exponential':
        # Pure time-based exponential decay
        W = compute_time_weights_exponential(times_cal, times_test, half_life_days, causal, min_lag_days)

    elif weighting == 'sliding_window':
        # Pure time-based sliding window
        W = compute_time_weights_sliding_window(times_cal, times_test, window_days, causal, min_lag_days)

    elif weighting == 'combined':
        # Feature kernel x time decay
        if omega is None:
            omega = np.ones(len(kernel_feature_cols))

        X_kernel_cal = df[kernel_feature_cols].iloc[idx_cal].to_numpy(dtype=float)
        X_kernel_test = df[kernel_feature_cols].iloc[idx_test].to_numpy(dtype=float)

        W_feature = _compute_kernel_distances(X_kernel_test, X_kernel_cal, omega, tau)
        W_time = compute_time_weights_exponential(times_cal, times_test, half_life_days, causal, min_lag_days)

        # Combined: element-wise product
        W = W_feature * W_time

    else:
        raise ValueError(f"Invalid weighting: {weighting}")

    # Compute weighted quantiles for each test point
    n_test = len(y_test)
    q_hat = np.zeros(n_test)

    # Apply safety margin to quantile level
    # Negative margin reduces conservatism (e.g., -0.02 for over-coverage)
    # Positive margin increases conservatism
    alpha_adjusted = np.clip(alpha_target + safety_margin, 0.0, 1.0)

    for i in range(n_test):
        w_i = W[i, :]  # Weights for test point i

        # Handle edge case: all weights zero
        if w_i.sum() < 1e-10:
            # Fall back to uniform weights
            w_i = np.ones_like(w_i)

        q_hat[i] = weighted_quantile(
            values=r_cal,
            weights=w_i,
            q=alpha_adjusted,
            include_test_weight=True,
        )

    # Construct conformal predictions
    margin = q_hat * scale_test
    y_pred_conf = y_pred_test - margin

    # Metrics
    coverage = float((y_test >= y_pred_conf).mean())
    pre_conf_coverage = float((y_test >= y_pred_test).mean())

    metrics = {
        'coverage': coverage,
        'gap': abs(coverage - alpha_target),
        'pre_conformal_coverage': pre_conf_coverage,
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_conf))),
        'mae': float(mean_absolute_error(y_test, y_pred_conf)),
        'q_hat_mean': float(q_hat.mean()),
        'q_hat_std': float(q_hat.std()),
        'q_hat_min': float(q_hat.min()),
        'q_hat_max': float(q_hat.max()),
        'n_train': len(y_train),
        'n_cal': len(y_cal),
        'n_test': len(y_test),
        'weighting': weighting,
        'safety_margin': safety_margin,
        'alpha_adjusted': float(alpha_adjusted),
    }

    # Return test predictions
    df_test = pd.DataFrame({
        'y': y_test,
        'y_pred_base': y_pred_test,
        'y_pred_conf': y_pred_conf,
        'q_hat_local': q_hat,
        'margin': margin,
        'scale': scale_test,
    })

    # Add timestamps for analysis
    df_test[time_col] = times_test.values

    return None, metrics, df_test  # Bundle not needed for comparison


def run_weighting_schemes_experiment(
    alpha_values: list[float] = None,
    tau_values: list[float] = None,
    half_life_values: list[float] = None,
    window_values: list[float] = None,
    causal: bool = True,
    min_lag_days: float = 0.0,
    omega_path: str = "data/viz_artifacts/focused_2d/best_omega.npy",
):
    """
    Compare different weighting schemes for conformal prediction.

    Tests:
    1. Binned (uniform weights within bins)
    2. Feature kernel (current weighted conformal)
    3. Time exponential decay
    4. Time sliding window
    5. Combined (feature + time)
    """
    if alpha_values is None:
        alpha_values = [0.95]
    if tau_values is None:
        tau_values = [2.0, 5.0]
    if half_life_values is None:
        half_life_values = [14.0, 30.0, 60.0]  # 2 weeks, 1 month, 2 months
    if window_values is None:
        window_values = [14.0, 30.0, 60.0]

    print("\n" + "="*80)
    print("EXPERIMENT: Comparing Weighting Schemes for Conformal Prediction")
    print("="*80)
    print(f"Alpha values: {alpha_values}")
    print(f"Tau values (feature kernel): {tau_values}")
    print(f"Half-life values (exponential decay): {half_life_values} days")
    print(f"Window values (sliding window): {window_values} days")
    print("="*80 + "\n")

    # Load data
    print("Loading RTS data...")
    actuals_path = Path("data/actuals_filtered_rts3_constellation_v1.parquet")
    forecasts_path = Path("data/forecasts_filtered_rts3_constellation_v1.parquet")

    actuals = pd.read_parquet(actuals_path)
    forecasts = pd.read_parquet(forecasts_path)
    df = build_conformal_totals_df(actuals, forecasts)

    print(f"Loaded {len(df)} rows")
    print(f"Time range: {df['TIME_HOURLY'].min()} to {df['TIME_HOURLY'].max()}")
    print(f"Duration: {(df['TIME_HOURLY'].max() - df['TIME_HOURLY'].min()).days} days\n")

    # Load omega
    omega_file = Path(omega_path)
    if omega_file.exists():
        omega = np.load(omega_path)
        print(f"Loaded omega: {omega}")
    else:
        print("⚠ Omega not found, using uniform weights")
        omega = np.array([1.0, 1.0])

    # Add kernel features
    if 'SYS_MEAN' not in df.columns:
        df['SYS_MEAN'] = df['ens_mean']
    if 'SYS_STD' not in df.columns:
        df['SYS_STD'] = df['ens_std']

    print()

    results = []

    for alpha in alpha_values:
        print(f"\n{'='*80}")
        print(f"Alpha = {alpha}")
        print(f"{'='*80}\n")

        # =====================================================================
        # 1. BINNED (baseline)
        # =====================================================================
        print("  [1/5] Binned conformal (baseline)...")
        try:
            bundle, metrics, df_test = train_wind_lower_model_conformal_binned(
                df,
                feature_cols=['ens_mean', 'ens_std'],
                scale_col='ens_std',
                binning='y_pred',
                n_bins=5,
                alpha_target=alpha,
                split_method='random',
                random_seed=42,
            )

            # Compute gap manually if not in metrics
            gap = abs(metrics['coverage'] - alpha)

            results.append({
                'method': 'binned',
                'alpha_target': alpha,
                'param': np.nan,
                'coverage': metrics['coverage'],
                'gap': gap,
                'q_hat_mean': metrics.get('q_hat_global_r', np.nan),
                'q_hat_std': 0.0,  # Global q_hat per bin
            })

            print(f"    Coverage: {metrics['coverage']:.3f} (gap: {gap:.3f})")
            print(f"    q_hat: {metrics.get('q_hat_global_r', np.nan):.3f}\n")
        except Exception as e:
            print(f"    (x) Failed: {e}\n")

        # =====================================================================
        # 2. FEATURE KERNEL (current weighted conformal)
        # =====================================================================
        print("  [2/5] Feature-based kernel weighting...")
        for tau in tau_values:
            print(f"    tau={tau}...")
            try:
                _, metrics, df_test = train_time_weighted_conformal(
                    df,
                    weighting='combined',  # Will use feature only if we set time weight to 1
                    tau=tau,
                    half_life_days=1e6,  # Effectively infinite = no time decay
                    omega=omega,
                    alpha_target=alpha,
                    causal=causal,
                    min_lag_days=min_lag_days,
                    split_method='random',
                    random_seed=42,
                )

                results.append({
                    'method': 'feature_kernel',
                    'alpha_target': alpha,
                    'param': tau,
                    'coverage': metrics['coverage'],
                    'gap': metrics['gap'],
                    'q_hat_mean': metrics['q_hat_mean'],
                    'q_hat_std': metrics['q_hat_std'],
                })

                print(f"      Coverage: {metrics['coverage']:.3f} (gap: {metrics['gap']:.3f})")
            except Exception as e:
                print(f"      (x) Failed: {e}")
        print()

        # =====================================================================
        # 3. TIME EXPONENTIAL DECAY
        # =====================================================================
        print("  [3/5] Time-based exponential decay...")
        for half_life in half_life_values:
            print(f"    half_life={half_life} days...")
            try:
                _, metrics, df_test = train_time_weighted_conformal(
                    df,
                    weighting='exponential',
                    half_life_days=half_life,
                    alpha_target=alpha,
                    causal=causal,
                    min_lag_days=min_lag_days,
                    split_method='random',
                    random_seed=42,
                )

                results.append({
                    'method': 'time_exponential',
                    'alpha_target': alpha,
                    'param': half_life,
                    'coverage': metrics['coverage'],
                    'gap': metrics['gap'],
                    'q_hat_mean': metrics['q_hat_mean'],
                    'q_hat_std': metrics['q_hat_std'],
                })

                print(f"      Coverage: {metrics['coverage']:.3f} (gap: {metrics['gap']:.3f})")
            except Exception as e:
                print(f"      (x) Failed: {e}")
        print()

        # =====================================================================
        # 4. TIME SLIDING WINDOW
        # =====================================================================
        print("  [4/5] Time-based sliding window...")
        for window in window_values:
            print(f"    window={window} days...")
            try:
                _, metrics, df_test = train_time_weighted_conformal(
                    df,
                    weighting='sliding_window',
                    window_days=window,
                    alpha_target=alpha,
                    causal=causal,
                    min_lag_days=min_lag_days,
                    split_method='random',
                    random_seed=42,
                )

                results.append({
                    'method': 'time_window',
                    'alpha_target': alpha,
                    'param': window,
                    'coverage': metrics['coverage'],
                    'gap': metrics['gap'],
                    'q_hat_mean': metrics['q_hat_mean'],
                    'q_hat_std': metrics['q_hat_std'],
                })

                print(f"      Coverage: {metrics['coverage']:.3f} (gap: {metrics['gap']:.3f})")
            except Exception as e:
                print(f"      (x) Failed: {e}")
        print()

        # =====================================================================
        # 5. COMBINED (feature + time)
        # =====================================================================
        print("  [5/5] Combined feature + time kernel...")
        for tau in tau_values:
            for half_life in half_life_values:
                print(f"    tau={tau}, half_life={half_life} days...")
                try:
                    _, metrics, df_test = train_time_weighted_conformal(
                        df,
                        weighting='combined',
                        tau=tau,
                        half_life_days=half_life,
                        omega=omega,
                        alpha_target=alpha,
                        causal=causal,
                        min_lag_days=min_lag_days,
                        split_method='random',
                        random_seed=42,
                    )

                    results.append({
                        'method': 'combined',
                        'alpha_target': alpha,
                        'param': f"τ={tau},hl={half_life}",
                        'coverage': metrics['coverage'],
                        'gap': metrics['gap'],
                        'q_hat_mean': metrics['q_hat_mean'],
                        'q_hat_std': metrics['q_hat_std'],
                    })

                    print(f"      Coverage: {metrics['coverage']:.3f} (gap: {metrics['gap']:.3f})")
                except Exception as e:
                    print(f"      (x) Failed: {e}")
        print()

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "weighting_schemes_comparison.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n(ok) Saved results to {results_path}")

    # Generate visualizations
    plot_weighting_schemes(df_results, output_dir)

    # Print summary
    print_weighting_summary(df_results)

    return df_results


def plot_weighting_schemes(df_results: pd.DataFrame, output_dir: Path):
    """Visualize comparison of weighting schemes."""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    alpha = df_results['alpha_target'].iloc[0]

    # -------------------------------------------------------------------------
    # 1. Coverage gap by method
    # -------------------------------------------------------------------------
    ax = axes[0, 0]

    method_order = ['binned', 'feature_kernel', 'time_exponential', 'time_window', 'combined']
    method_labels = ['Binned', 'Feature\nKernel', 'Time\nExponential', 'Time\nWindow', 'Combined']

    gaps = []
    for method in method_order:
        df_method = df_results[df_results['method'] == method]
        if not df_method.empty:
            gaps.append(df_method['gap'].min())
        else:
            gaps.append(np.nan)

    colors = ['#E63946', '#F77F00', '#06AED5', '#2A9D8F', '#9B59B6']
    bars = ax.bar(range(len(gaps)), gaps, color=colors, alpha=0.8)

    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.set_ylabel('Coverage Gap (best config)', fontsize=11, fontweight='bold')
    ax.set_title(f'Coverage Gap by Method (α={alpha})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='5% threshold')
    ax.legend(fontsize=9)

    # -------------------------------------------------------------------------
    # 2. Time exponential: coverage vs half-life
    # -------------------------------------------------------------------------
    ax = axes[0, 1]

    df_exp = df_results[df_results['method'] == 'time_exponential']
    if not df_exp.empty:
        ax.plot(df_exp['param'], df_exp['coverage'], 'o-', linewidth=2, markersize=8, color='#06AED5')
        ax.axhline(alpha, color='red', linestyle='--', linewidth=2, label=f'Target ({alpha})')
        ax.set_xlabel('Half-life (days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Coverage', fontsize=11, fontweight='bold')
        ax.set_title('Time Exponential Decay', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # 3. Time window: coverage vs window size
    # -------------------------------------------------------------------------
    ax = axes[1, 0]

    df_win = df_results[df_results['method'] == 'time_window']
    if not df_win.empty:
        ax.plot(df_win['param'], df_win['coverage'], 's-', linewidth=2, markersize=8, color='#2A9D8F')
        ax.axhline(alpha, color='red', linestyle='--', linewidth=2, label=f'Target ({alpha})')
        ax.set_xlabel('Window size (days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Coverage', fontsize=11, fontweight='bold')
        ax.set_title('Time Sliding Window', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # 4. Spatial variation (q_hat std) by method
    # -------------------------------------------------------------------------
    ax = axes[1, 1]

    stds = []
    for method in method_order:
        df_method = df_results[df_results['method'] == method]
        if not df_method.empty:
            # Find best config
            best_idx = df_method['gap'].idxmin()
            stds.append(df_method.loc[best_idx, 'q_hat_std'])
        else:
            stds.append(np.nan)

    bars = ax.bar(range(len(stds)), stds, color=colors, alpha=0.8)
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.set_ylabel('q_hat std (spatial variation)', fontsize=11, fontweight='bold')
    ax.set_title('Spatial Variation in Corrections', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / 'weighting_schemes_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"(ok) Saved visualization to {output_path}")

    plt.close()


def print_weighting_summary(df_results: pd.DataFrame):
    """Print summary of weighting scheme comparison."""
    print("\n" + "="*80)
    print("SUMMARY: Weighting Schemes Comparison")
    print("="*80 + "\n")

    methods = df_results['method'].unique()

    for method in methods:
        df_method = df_results[df_results['method'] == method]
        if df_method.empty:
            continue

        best_idx = df_method['gap'].idxmin()
        best = df_method.loc[best_idx]

        print(f"{method.upper()}:")
        print(f"  Best config: param={best['param']}")
        print(f"  Coverage: {best['coverage']:.3f} (gap: {best['gap']:.3f})")
        print(f"  q_hat: mean={best['q_hat_mean']:.3f}, std={best['q_hat_std']:.3f}")
        print()

    # Find overall best
    best_overall_idx = df_results['gap'].idxmin()
    best_overall = df_results.loc[best_overall_idx]

    print("="*80)
    print("BEST OVERALL:")
    print(f"  Method: {best_overall['method']}")
    print(f"  Config: {best_overall['param']}")
    print(f"  Coverage: {best_overall['coverage']:.3f} (gap: {best_overall['gap']:.3f})")
    print("="*80)

    print("\nKEY INSIGHTS:")
    print("- If time-based methods have similar gaps to feature-kernel:")
    print("  -> Temporal drift is not significant")
    print("- If time-based methods have better coverage:")
    print("  -> Recent calibration points are more relevant")
    print("- If combined (feature+time) is best:")
    print("  -> Both feature similarity AND recency matter")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run experiments with day-ahead realistic settings
    # causal=True: only use past calibration data
    # min_lag_days=1.0: can't use same-day data (day-ahead constraint)
    df_results = run_weighting_schemes_experiment(
        alpha_values=[0.95],
        tau_values=[2.0, 5.0],
        half_life_values=[14.0, 30.0, 60.0],
        window_values=[14.0, 30.0, 60.0],
        causal=True,  # Only use past data (realistic)
        min_lag_days=1.0,  # Day-ahead: yesterday or earlier only
        omega_path="data/viz_artifacts/focused_2d/best_omega.npy",
    )

    print("\n(ok) Weighting schemes experiment complete!")
