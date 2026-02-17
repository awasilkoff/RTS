#!/usr/bin/env python
"""
Comprehensive comparison visualization script.

Compares four approaches:
1. Learned omega (adaptive k-NN with learned feature weights)
2. Kernel with equal omega (kernel distance, ω=[1,1,...])
3. Euclidean k-NN (true uniform 1/k weights)
4. Global covariance baseline (no adaptation)

Usage modes:
1. Load from config (default):
   python create_comprehensive_comparison.py --feature-set focused_2d

2. Specify omega directly:
   python create_comprehensive_comparison.py --feature-set focused_2d --omega 0.5 0.3 --tau 2.0

3. Learn omega from scratch:
   python create_comprehensive_comparison.py --feature-set focused_2d --learn-omega --tau 2.0 --omega-l2-reg 0.01

Generates:
- Pairwise kernel distance comparisons
- Box-and-whisker plots of NLL distributions
- Summary statistics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

from data_processing_extended import FEATURE_BUILDERS
from utils import fit_scaler, apply_scaler
from viz_kernel_distance import (
    plot_kernel_distance_comparison,
    compute_kernel_weights,
    plot_kernel_distance,
    plot_global_uniform_weights,
    plot_knn_binary_weights,
)
from viz_ellipsoid_plane_3d import ellipsoid_mesh
from covariance_optimization import (
    CovPredictConfig,
    predict_mu_sigma_topk_cross,
    predict_mu_sigma_knn,
    KernelCovConfig,
    FitConfig,
    fit_omega,
)


def compute_nll_per_point(Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute NLL for each point individually.

    Returns:
        nll_per_point: (N,) array of NLL values
    """
    N, M = Y.shape
    nll = np.empty(N, dtype=float)

    for i in range(N):
        S = Sigma[i]
        r = (Y[i] - Mu[i]).reshape(M, 1)
        try:
            L = np.linalg.cholesky(S)
            logdet = 2.0 * np.log(np.diag(L)).sum()
            x = np.linalg.solve(L, r)
            x = np.linalg.solve(L.T, x)
            quad = float(r.T @ x)
            nll[i] = 0.5 * (logdet + quad + M * np.log(2.0 * np.pi))
        except np.linalg.LinAlgError:
            # If Cholesky fails, use large NLL
            nll[i] = 1e6

    return nll


def create_nll_boxplot(nll_dict, save_path=None, outlier_percentile=99):
    """Create box-and-whisker plot comparing NLL distributions.

    Parameters
    ----------
    nll_dict : dict
        Method name -> NLL values array.
    save_path : Path, optional
        Where to save the figure.
    outlier_percentile : float
        If max value exceeds this percentile by more than 2x, clip y-axis
        to show the main distribution clearly. Default: 99.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(nll_dict.keys())
    data = [nll_dict[m] for m in methods]

    # Combine all data to detect outliers
    all_nll = np.concatenate(data)
    p_low = np.percentile(all_nll, 1)
    p_high = np.percentile(all_nll, outlier_percentile)
    data_max = np.max(all_nll)
    data_min = np.min(all_nll)

    # Detect if outliers make plot illegible
    # Criterion: max value is more than 3x the 99th percentile
    has_extreme_outliers = data_max > 3 * p_high or data_max > p_high + 10 * (p_high - p_low)
    n_outliers_above = np.sum(all_nll > p_high)

    # Create box plot (don't show fliers if we're going to clip)
    bp = ax.boxplot(data, labels=methods, patch_artist=True,
                     showmeans=True, meanline=True,
                     showfliers=not has_extreme_outliers)

    # Color boxes
    colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)

    # If extreme outliers, clip y-axis and add annotation
    if has_extreme_outliers:
        # Set y-limits to show main distribution
        y_margin = 0.1 * (p_high - p_low)
        ax.set_ylim(p_low - y_margin, p_high + y_margin)

        # Add annotation about clipped outliers
        ax.annotate(
            f"Note: {n_outliers_above} outliers above {p_high:.1f} not shown\n(max={data_max:.1f})",
            xy=(0.98, 0.98), xycoords='axes fraction',
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    # Styling
    ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax.set_title('NLL Distribution Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean values as text (use median position if mean is outside visible range)
    y_min, y_max = ax.get_ylim()
    for i, (method, nll_vals) in enumerate(nll_dict.items()):
        mean_val = np.mean(nll_vals)
        median_val = np.median(nll_vals)

        # Position label at mean if visible, otherwise at top of visible range
        if y_min <= mean_val <= y_max:
            label_y = mean_val
            label_text = f'μ={mean_val:.2f}'
        else:
            label_y = y_max - 0.05 * (y_max - y_min)
            label_text = f'μ={mean_val:.1f}^' if mean_val > y_max else f'μ={mean_val:.1f}v'

        ax.text(i+1, label_y, label_text,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        if has_extreme_outliers:
            print(f"  (Note: {n_outliers_above} extreme outliers clipped from view)")

    return fig


def plot_ellipsoid_comparison(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    y_actual: np.ndarray | None = None,
    name1: str = "Method 1",
    name2: str = "Method 2",
    rho: float = 1.0,
    dims3: tuple = (0, 1, 2),
    y_cols: list | None = None,
    title: str | None = None,
    ax=None,
):
    """
    Plot two 3D covariance ellipsoids side-by-side for comparison.

    Parameters
    ----------
    mu1, sigma1 : np.ndarray
        Mean (M,) and covariance (M, M) for method 1.
    mu2, sigma2 : np.ndarray
        Mean (M,) and covariance (M, M) for method 2.
    y_actual : np.ndarray, optional
        Actual observation to mark on plot.
    name1, name2 : str
        Labels for the two methods.
    rho : float
        Ellipsoid radius (number of standard deviations).
    dims3 : tuple
        Which 3 dimensions to plot if M > 3.
    y_cols : list, optional
        Column names for axis labels.
    title : str, optional
        Plot title.
    ax : matplotlib Axes3D, optional
        Existing axes to plot on.
    """
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    d0, d1, d2 = dims3

    # Extract 3D subsets
    mu1_3d = mu1[[d0, d1, d2]]
    sigma1_3d = sigma1[np.ix_([d0, d1, d2], [d0, d1, d2])]
    mu2_3d = mu2[[d0, d1, d2]]
    sigma2_3d = sigma2[np.ix_([d0, d1, d2], [d0, d1, d2])]

    # Build ellipsoid meshes
    X1, Y1, Z1 = ellipsoid_mesh(mu1_3d, sigma1_3d, rho)
    X2, Y2, Z2 = ellipsoid_mesh(mu2_3d, sigma2_3d, rho)

    # Plot ellipsoids as wireframes
    ax.plot_wireframe(X1, Y1, Z1, alpha=0.5, linewidth=0.5, color='green', label=name1)
    ax.plot_wireframe(X2, Y2, Z2, alpha=0.5, linewidth=0.5, color='blue', label=name2)

    # Mark centers
    ax.scatter([mu1_3d[0]], [mu1_3d[1]], [mu1_3d[2]], s=80, c='green', marker='o', edgecolors='black')
    ax.scatter([mu2_3d[0]], [mu2_3d[1]], [mu2_3d[2]], s=80, c='blue', marker='o', edgecolors='black')

    # Mark actual observation if provided
    if y_actual is not None:
        y3 = y_actual[[d0, d1, d2]]
        ax.scatter([y3[0]], [y3[1]], [y3[2]], s=150, c='red', marker='*', label='Actual Y', edgecolors='black')

    ax.legend(loc='upper right')

    # Set axis labels
    if y_cols is not None and len(y_cols) > max(dims3):
        ax.set_xlabel(y_cols[d0])
        ax.set_ylabel(y_cols[d1])
        ax.set_zlabel(y_cols[d2])
    else:
        ax.set_xlabel(f'Y[{d0}]')
        ax.set_ylabel(f'Y[{d1}]')
        ax.set_zlabel(f'Y[{d2}]')

    if title:
        ax.set_title(title)

    # Set axis limits based on both ellipsoids
    all_x = np.concatenate([X1.ravel(), X2.ravel()])
    all_y = np.concatenate([Y1.ravel(), Y2.ravel()])
    all_z = np.concatenate([Z1.ravel(), Z2.ravel()])

    if y_actual is not None:
        y3 = y_actual[[d0, d1, d2]]
        all_x = np.append(all_x, y3[0])
        all_y = np.append(all_y, y3[1])
        all_z = np.append(all_z, y3[2])

    all_x = all_x[np.isfinite(all_x)]
    all_y = all_y[np.isfinite(all_y)]
    all_z = all_z[np.isfinite(all_z)]

    pad = 0.1
    x_range = all_x.max() - all_x.min() if len(all_x) > 0 else 1.0
    y_range = all_y.max() - all_y.min() if len(all_y) > 0 else 1.0
    z_range = all_z.max() - all_z.min() if len(all_z) > 0 else 1.0

    ax.set_xlim(all_x.min() - pad * x_range, all_x.max() + pad * x_range)
    ax.set_ylim(all_y.min() - pad * y_range, all_y.max() + pad * y_range)
    ax.set_zlim(all_z.min() - pad * z_range, all_z.max() + pad * z_range)

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive comparison visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from existing config
  python create_comprehensive_comparison.py --feature-set focused_2d

  # Specify omega directly
  python create_comprehensive_comparison.py --feature-set focused_2d \\
      --omega 0.5 0.3 --tau 2.0 --scaler-type standard

  # Learn omega from scratch
  python create_comprehensive_comparison.py --feature-set focused_2d \\
      --learn-omega --tau 2.0 --omega-l2-reg 0.01 --scaler-type minmax
        """
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="focused_2d",
        choices=list(FEATURE_BUILDERS.keys()),
        help="Feature set to use"
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        default=None,
        choices=["none", "standard", "minmax"],
        help="Scaler type (required if --omega or --learn-omega is used)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory with feature_config.json and best_config_summary.json (default: data/viz_artifacts/<feature-set>)"
    )
    parser.add_argument(
        "--target-idx",
        type=int,
        default=None,
        help="Target index for kernel distance plots (default: middle of dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <config-dir>/comprehensive_comparison)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=128,
        help="Number of neighbors for k-NN (default: 128)"
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Ridge regularization (default: 1e-3)"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=2.0,
        help="Ellipsoid radius in standard deviations for 3D plots (default: 2.0)"
    )
    parser.add_argument(
        "--skip-ellipsoids",
        action="store_true",
        help="Skip 3D ellipsoid comparison plots"
    )

    # Omega specification options
    omega_group = parser.add_argument_group("Omega specification (choose one)")
    omega_group.add_argument(
        "--omega",
        type=float,
        nargs="+",
        default=None,
        help="Directly specify omega values (e.g., --omega 0.5 0.3)"
    )
    omega_group.add_argument(
        "--learn-omega",
        action="store_true",
        help="Learn omega from scratch using specified hyperparameters"
    )

    # Hyperparameters (used when --omega or --learn-omega is specified)
    hyper_group = parser.add_argument_group("Hyperparameters (for --omega or --learn-omega)")
    hyper_group.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Temperature parameter for kernel (required if --omega or --learn-omega)"
    )
    hyper_group.add_argument(
        "--omega-l2-reg",
        type=float,
        default=0.0,
        help="L2 regularization for omega learning (default: 0.0)"
    )
    hyper_group.add_argument(
        "--omega-constraint",
        type=str,
        default="none",
        choices=["none", "softmax", "simplex", "normalize"],
        help="Constraint on omega: 'none' (no constraint), 'softmax' (sum=1 via softmax), "
             "'simplex' (project to simplex), 'normalize' (divide by sum at end). Default: none"
    )
    hyper_group.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="Max iterations for omega learning (default: 100)"
    )
    hyper_group.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Step size for omega learning (default: 0.1)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.omega is not None and args.learn_omega:
        parser.error("Cannot specify both --omega and --learn-omega")

    if (args.omega is not None or args.learn_omega) and args.tau is None:
        parser.error("--tau is required when using --omega or --learn-omega")

    if (args.omega is not None or args.learn_omega) and args.scaler_type is None:
        parser.error("--scaler-type is required when using --omega or --learn-omega")

    # Set default config directory
    if args.config_dir is None:
        args.config_dir = Path(f"data/viz_artifacts/{args.feature_set}")

    # Set default output directory
    if args.output_dir is None:
        if args.omega is not None or args.learn_omega:
            args.output_dir = Path(f"data/viz_artifacts/{args.feature_set}/custom_comparison")
        else:
            args.output_dir = args.config_dir / "comprehensive_comparison"

    args.output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("COMPREHENSIVE COMPARISON VISUALIZATION")
    print("=" * 80)
    print()

    # Load data
    data_dir = Path("data")
    actuals = pd.read_parquet(data_dir / "actuals_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(data_dir / "forecasts_filtered_rts3_constellation_v1.parquet")

    # Build features
    feature_set = args.feature_set
    build_fn = FEATURE_BUILDERS[feature_set]
    X_raw, Y, times, x_cols, y_cols = build_fn(forecasts, actuals, drop_any_nan_rows=True)

    # Determine parameters based on mode
    if args.omega is not None:
        # Mode 1: Directly specified omega
        omega_learned = np.array(args.omega)
        if len(omega_learned) != X_raw.shape[1]:
            parser.error(f"--omega requires {X_raw.shape[1]} values for {feature_set}, got {len(omega_learned)}")
        tau = args.tau
        scaler_type = args.scaler_type
        print(f"Mode: Direct omega specification")
        print(f"  Omega: {omega_learned}")

    elif args.learn_omega:
        # Mode 2: Learn omega from scratch
        tau = args.tau
        scaler_type = args.scaler_type
        print(f"Mode: Learning omega from scratch")
        print(f"  Tau: {tau}")
        print(f"  L2 regularization: {args.omega_l2_reg}")
        print(f"  Omega constraint: {args.omega_constraint}")
        print(f"  Max iterations: {args.max_iters}")
        omega_learned = None  # Will be set after learning

    else:
        # Mode 3: Load from config file
        config_path = args.config_dir / "feature_config.json"
        best_config_path = args.config_dir / "best_config_summary.json"

        if not config_path.exists() or not best_config_path.exists():
            parser.error(f"Config files not found in {args.config_dir}. "
                        f"Use --omega or --learn-omega to specify parameters directly.")

        with open(config_path) as f:
            config = json.load(f)
        with open(best_config_path) as f:
            best_config = json.load(f)

        scaler_type = config["scaler_type"]
        tau = best_config["best_config"]["tau"]
        omega_learned = np.array(best_config["best_omega"])
        print(f"Mode: Loaded from config")
        print(f"  Config dir: {args.config_dir}")

    # Apply scaler
    scaler = fit_scaler(X_raw, scaler_type)
    X = apply_scaler(X_raw, scaler)

    print(f"Configuration:")
    print(f"  Feature set: {feature_set}")
    print(f"  Features: {x_cols}")
    print(f"  Scaler: {scaler_type}")
    print(f"  Tau: {tau}")
    print(f"  k (neighbors): {args.k}")
    print(f"  Ridge: {args.ridge}")
    print()

    print(f"Data: {X.shape[0]} points, {X.shape[1]} features")
    print()

    # Split into train/eval (same as sweep)
    n = X.shape[0]
    train_frac = 0.75
    n_train = int(train_frac * n)

    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx = np.sort(indices[:n_train])
    eval_idx = np.sort(indices[n_train:])

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_eval, Y_eval = X[eval_idx], Y[eval_idx]
    times_eval = times[eval_idx]

    print(f"Train: {len(train_idx)}, Eval: {len(eval_idx)}")
    print()

    # Learn omega if requested
    if args.learn_omega:
        print("Learning omega...")
        kernel_cfg = KernelCovConfig(
            tau=tau,
            ridge=args.ridge,
            enforce_nonneg_omega=True,
        )
        fit_cfg = FitConfig(
            max_iters=args.max_iters,
            step_size=args.step_size,
            omega_l2_reg=args.omega_l2_reg,
            omega_constraint=args.omega_constraint,
            tol=1e-7,
            grad_clip=10.0,
            verbose_every=50,
        )
        omega0 = np.ones(X.shape[1])
        omega_learned = fit_omega(
            X=X,
            Y=Y,
            omega0=omega0,
            train_idx=train_idx,
            cfg=kernel_cfg,
            fit_cfg=fit_cfg,
        )
        print(f"  Learned omega: {omega_learned}")
        print()

    omega_equal = np.ones_like(omega_learned)
    k = args.k
    ridge = args.ridge

    print(f"Final omega values:")
    print(f"  Omega learned: {omega_learned}")
    print(f"  Omega equal: {omega_equal}")
    print()

    # Set up prediction config
    pred_cfg = CovPredictConfig(
        tau=float(tau),
        ridge=float(ridge),
        enforce_nonneg_omega=True,
        dtype="float32",
        device="cpu",
    )

    print("Computing predictions...")
    print()

    # 1. Learned omega predictions
    print("  1/4: Learned omega...")
    Mu_learned, Sigma_learned = predict_mu_sigma_topk_cross(
        X_query=X_eval,
        X_ref=X_train,
        Y_ref=Y_train,
        omega=omega_learned,
        cfg=pred_cfg,
        k=k,
        exclude_self_if_same=False,
        return_type="numpy",
    )
    nll_learned = compute_nll_per_point(Y_eval, Mu_learned, Sigma_learned)

    # 2. Kernel with equal feature weights (higher ridge for stability)
    print("  2/4: Kernel with equal omega...")
    pred_cfg_stable = CovPredictConfig(
        tau=float(tau),
        ridge=1e-2,  # 10x higher ridge for numerical stability
        enforce_nonneg_omega=True,
        dtype="float64",  # Use float64 for better precision
        device="cpu",
    )
    Mu_kernel_equal, Sigma_kernel_equal = predict_mu_sigma_topk_cross(
        X_query=X_eval,
        X_ref=X_train,
        Y_ref=Y_train,
        omega=omega_equal,
        cfg=pred_cfg_stable,
        k=k,
        exclude_self_if_same=False,
        return_type="numpy",
    )
    nll_kernel_equal = compute_nll_per_point(Y_eval, Mu_kernel_equal, Sigma_kernel_equal)

    # 3. Standard Euclidean k-NN (uniform 1/k weights)
    print("  3/4: Euclidean k-NN...")
    Mu_euclidean_knn, Sigma_euclidean_knn = predict_mu_sigma_knn(
        X_query=X_eval,
        X_ref=X_train,
        Y_ref=Y_train,
        k=k,
        ridge=ridge,
    )
    nll_euclidean_knn = compute_nll_per_point(Y_eval, Mu_euclidean_knn, Sigma_euclidean_knn)

    # 4. Global covariance predictions
    print("  4/4: Global covariance...")
    Mu_global = np.mean(Y_train, axis=0)
    Sigma_global = np.cov(Y_train, rowvar=False)
    Sigma_global += ridge * np.eye(Sigma_global.shape[0])

    Mu_global_eval = np.tile(Mu_global, (len(Y_eval), 1))
    Sigma_global_eval = np.tile(Sigma_global[None, :, :], (len(Y_eval), 1, 1))
    nll_global = compute_nll_per_point(Y_eval, Mu_global_eval, Sigma_global_eval)

    print()
    print("=" * 80)
    print("NLL STATISTICS")
    print("=" * 80)
    print()

    # Print statistics
    methods = {
        "Learned Omega": nll_learned,
        "Kernel (ω=1)": nll_kernel_equal,
        "Euclidean k-NN": nll_euclidean_knn,
        "Global Cov": nll_global,
    }

    for method, nll_vals in methods.items():
        print(f"{method}:")
        print(f"  Mean:   {np.mean(nll_vals):.4f}")
        print(f"  Median: {np.median(nll_vals):.4f}")
        print(f"  Std:    {np.std(nll_vals):.4f}")
        print(f"  Min:    {np.min(nll_vals):.4f}")
        print(f"  Max:    {np.max(nll_vals):.4f}")
        print(f"  Q25:    {np.percentile(nll_vals, 25):.4f}")
        print(f"  Q75:    {np.percentile(nll_vals, 75):.4f}")
        print()

    # Improvements (Δ NLL, positive = better)
    print("Mean NLL Improvements (Δ NLL, positive = better):")
    print(f"  Learned vs Kernel(ω=1):     {np.mean(nll_kernel_equal) - np.mean(nll_learned):.4f}")
    print(f"  Learned vs Euclidean k-NN:  {np.mean(nll_euclidean_knn) - np.mean(nll_learned):.4f}")
    print(f"  Learned vs Global:          {np.mean(nll_global) - np.mean(nll_learned):.4f}")
    print(f"  Euclidean k-NN vs Global:   {np.mean(nll_global) - np.mean(nll_euclidean_knn):.4f}")
    print(f"  Kernel(ω=1) vs Euclidean:   {np.mean(nll_euclidean_knn) - np.mean(nll_kernel_equal):.4f}")
    print()

    # Percentage of samples where learned omega is better
    pct_better_kernel = 100.0 * np.mean(nll_learned < nll_kernel_equal)
    pct_better_knn = 100.0 * np.mean(nll_learned < nll_euclidean_knn)
    pct_better_global = 100.0 * np.mean(nll_learned < nll_global)

    print("Percentage of samples where Learned Omega is better:")
    print(f"  vs Kernel(ω=1):     {pct_better_kernel:.1f}%")
    print(f"  vs Euclidean k-NN:  {pct_better_knn:.1f}%")
    print(f"  vs Global:          {pct_better_global:.1f}%")
    print()

    # Likelihood ratio: exp(baseline_NLL - learned_NLL)
    # Use median to be robust to outliers, or geometric mean
    # Geometric mean of likelihood ratios = exp(mean of NLL differences)
    def likelihood_ratio_stats(nll_baseline, nll_learned):
        """Compute likelihood ratio statistics."""
        nll_diff = nll_baseline - nll_learned  # positive means learned is better
        # Clip extreme values to avoid overflow
        nll_diff_clipped = np.clip(nll_diff, -50, 50)
        lr = np.exp(nll_diff_clipped)
        return {
            'geometric_mean': np.exp(np.mean(nll_diff_clipped)),
            'median': np.median(lr),
            'mean': np.mean(lr),
            'pct_>1': 100.0 * np.mean(lr > 1),  # % where learned is more likely
        }

    lr_kernel = likelihood_ratio_stats(nll_kernel_equal, nll_learned)
    lr_knn = likelihood_ratio_stats(nll_euclidean_knn, nll_learned)
    lr_global = likelihood_ratio_stats(nll_global, nll_learned)

    print("Likelihood Ratio (Learned vs Baseline, >1 means learned is better):")
    print(f"  vs Kernel(ω=1):     geom_mean={lr_kernel['geometric_mean']:.3f}, median={lr_kernel['median']:.3f}")
    print(f"  vs Euclidean k-NN:  geom_mean={lr_knn['geometric_mean']:.3f}, median={lr_knn['median']:.3f}")
    print(f"  vs Global:          geom_mean={lr_global['geometric_mean']:.3f}, median={lr_global['median']:.3f}")
    print()

    # Create box-and-whisker plot
    print("Creating NLL box-and-whisker plot...")
    create_nll_boxplot(
        methods,
        save_path=args.output_dir / "nll_boxplot.png"
    )
    print()

    # Select target for kernel distance plots
    if args.target_idx is None:
        target_idx = len(X) // 2
    else:
        target_idx = args.target_idx

    target_time = times[target_idx]

    print(f"Creating kernel distance comparison plots...")
    print(f"  Target: Index {target_idx} - {target_time}")
    print()

    # Pairwise comparisons: Learned Omega vs each baseline
    # Each comparison shows: Left = Learned Omega (kernel weights), Right = Baseline
    comparisons = [
        ("Kernel (ω=1)", "learned_vs_kernel_equal"),
        ("Euclidean k-NN", "learned_vs_euclidean_knn"),
        ("Global Cov", "learned_vs_global"),
    ]

    for i, (baseline_name, filename) in enumerate(comparisons):
        print(f"  {i+1}/3: Learned Omega vs {baseline_name}...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # For kernel-based comparisons, compute shared colorbar scale
        if baseline_name == "Kernel (ω=1)":
            # Compute weights for both to get shared scale
            X_target = X[target_idx]
            weights_learned = compute_kernel_weights(X_target, X, omega_learned, tau)
            weights_equal = compute_kernel_weights(X_target, X, omega_equal, tau)

            # Shared min/max for fair comparison
            # Use very small floor to show all points (weights are clipped to 1e-300)
            vmin = max(min(weights_learned.min(), weights_equal.min()), 1e-300)
            vmax = max(weights_learned.max(), weights_equal.max())

            # Left plot: Learned Omega
            plot_kernel_distance(
                X, x_cols, times, target_idx,
                omega=omega_learned, tau=tau,
                title=f"Learned Omega: ω = [{', '.join(f'{w:.3f}' for w in omega_learned)}]",
                ax=axes[0], show_colorbar=True,
                vmin=vmin, vmax=vmax
            )

            # Right plot: Kernel with equal omega (same scale)
            plot_kernel_distance(
                X, x_cols, times, target_idx,
                omega=omega_equal, tau=tau,
                title=f"Kernel (ω=1): ω = [{', '.join(f'{w:.1f}' for w in omega_equal)}]",
                ax=axes[1], show_colorbar=True,
                vmin=vmin, vmax=vmax
            )
        else:
            # For non-kernel baselines, left plot uses its own scale
            plot_kernel_distance(
                X, x_cols, times, target_idx,
                omega=omega_learned, tau=tau,
                title=f"Learned Omega: ω = [{', '.join(f'{w:.3f}' for w in omega_learned)}]",
                ax=axes[0], show_colorbar=True
            )

            if baseline_name == "Euclidean k-NN":
                # Binary k-NN: k nearest neighbors highlighted
                plot_knn_binary_weights(
                    X, x_cols, times, target_idx, k=k,
                    title=f"Euclidean k-NN: k={k} nearest (uniform 1/k weights)",
                    ax=axes[1],
                )
            elif baseline_name == "Global Cov":
                # Global: all points same color (uniform weight)
                plot_global_uniform_weights(
                    X, x_cols, times, target_idx,
                    title="Global Cov: All points equal weight",
                    ax=axes[1],
                )

        fig.suptitle(
            f"Learned Omega vs {baseline_name}\nTarget: {target_time.strftime('%Y-%m-%d %H:%M')}",
            fontsize=14, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = args.output_dir / f"comparison_{filename}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {save_path}")
        plt.close()

    # =========================================================================
    # 3D Ellipsoid Comparison Plots
    # =========================================================================
    if not args.skip_ellipsoids and Y.shape[1] >= 3:
        print()
        print("Creating 3D ellipsoid comparison plots...")
        print(f"  Rho (ellipsoid radius): {args.rho}")

        # Use the same target index for consistency
        # Get predictions at target index (need to find which eval index corresponds)
        # Since we're using eval set, find closest eval index to target_idx
        eval_target_idx = np.argmin(np.abs(eval_idx - target_idx))

        print(f"  Using eval point {eval_target_idx} (original idx {eval_idx[eval_target_idx]})")

        # Extract predictions at target
        mu_learned_t = Mu_learned[eval_target_idx]
        sigma_learned_t = Sigma_learned[eval_target_idx]
        mu_kernel_t = Mu_kernel_equal[eval_target_idx]
        sigma_kernel_t = Sigma_kernel_equal[eval_target_idx]
        mu_knn_t = Mu_euclidean_knn[eval_target_idx]
        sigma_knn_t = Sigma_euclidean_knn[eval_target_idx]
        mu_global_t = Mu_global_eval[eval_target_idx]
        sigma_global_t = Sigma_global_eval[eval_target_idx]
        y_actual_t = Y_eval[eval_target_idx]

        # Pairwise ellipsoid comparisons
        ellipsoid_comparisons = [
            ("Learned Omega", mu_learned_t, sigma_learned_t,
             "Kernel (ω=1)", mu_kernel_t, sigma_kernel_t,
             "ellipsoid_learned_vs_kernel_equal"),
            ("Learned Omega", mu_learned_t, sigma_learned_t,
             "Euclidean k-NN", mu_knn_t, sigma_knn_t,
             "ellipsoid_learned_vs_euclidean_knn"),
            ("Learned Omega", mu_learned_t, sigma_learned_t,
             "Global Cov", mu_global_t, sigma_global_t,
             "ellipsoid_learned_vs_global"),
        ]

        for name1, mu1, sig1, name2, mu2, sig2, filename in ellipsoid_comparisons:
            print(f"  Creating {name1} vs {name2}...")

            fig, ax = plot_ellipsoid_comparison(
                mu1=mu1, sigma1=sig1,
                mu2=mu2, sigma2=sig2,
                y_actual=y_actual_t,
                name1=name1, name2=name2,
                rho=args.rho,
                dims3=(0, 1, 2),
                y_cols=y_cols,
                title=f"{name1} vs {name2}\nTarget: {times_eval[eval_target_idx].strftime('%Y-%m-%d %H:%M')}"
            )

            save_path = args.output_dir / f"{filename}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {save_path}")
            plt.close()

        # Also create an overlay of all 4 methods
        print("  Creating all methods overlay...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        all_methods = [
            ("Learned Omega", mu_learned_t, sigma_learned_t, "green"),
            ("Kernel (ω=1)", mu_kernel_t, sigma_kernel_t, "blue"),
            ("Euclidean k-NN", mu_knn_t, sigma_knn_t, "orange"),
            ("Global Cov", mu_global_t, sigma_global_t, "purple"),
        ]

        d0, d1, d2 = 0, 1, 2
        for name, mu, sig, color in all_methods:
            mu3 = mu[[d0, d1, d2]]
            sig3 = sig[np.ix_([d0, d1, d2], [d0, d1, d2])]
            X, Y_mesh, Z = ellipsoid_mesh(mu3, sig3, args.rho)
            ax.plot_wireframe(X, Y_mesh, Z, alpha=0.4, linewidth=0.5, color=color, label=name)
            ax.scatter([mu3[0]], [mu3[1]], [mu3[2]], s=60, c=color, marker='o')

        # Mark actual observation
        y3 = y_actual_t[[d0, d1, d2]]
        ax.scatter([y3[0]], [y3[1]], [y3[2]], s=150, c='red', marker='*', label='Actual Y', edgecolors='black')

        ax.legend(loc='upper right')
        ax.set_xlabel(y_cols[d0] if y_cols else f'Y[{d0}]')
        ax.set_ylabel(y_cols[d1] if y_cols else f'Y[{d1}]')
        ax.set_zlabel(y_cols[d2] if y_cols else f'Y[{d2}]')
        ax.set_title(f"All Methods Comparison (ρ={args.rho})\nTarget: {times_eval[eval_target_idx].strftime('%Y-%m-%d %H:%M')}")

        save_path = args.output_dir / "ellipsoid_all_methods.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {save_path}")
        plt.close()

    elif args.skip_ellipsoids:
        print("\nSkipping 3D ellipsoid plots (--skip-ellipsoids)")
    elif Y.shape[1] < 3:
        print(f"\nSkipping 3D ellipsoid plots (Y has only {Y.shape[1]} dimensions, need >= 3)")

    # Save configuration for reproducibility
    config_out = {
        "feature_set": feature_set,
        "scaler_type": scaler_type,
        "tau": tau,
        "k": k,
        "ridge": ridge,
        "rho": args.rho,
        "omega_learned": omega_learned.tolist(),
        "omega_equal": omega_equal.tolist(),
        "mode": "direct" if args.omega else ("learned" if args.learn_omega else "config"),
        # NLL statistics
        "nll_stats": {
            "learned_mean": float(np.mean(nll_learned)),
            "kernel_equal_mean": float(np.mean(nll_kernel_equal)),
            "euclidean_knn_mean": float(np.mean(nll_euclidean_knn)),
            "global_mean": float(np.mean(nll_global)),
        },
        # Improvement metrics
        "improvements": {
            "learned_vs_kernel_equal": float(np.mean(nll_kernel_equal) - np.mean(nll_learned)),
            "learned_vs_euclidean_knn": float(np.mean(nll_euclidean_knn) - np.mean(nll_learned)),
            "learned_vs_global": float(np.mean(nll_global) - np.mean(nll_learned)),
        },
        "pct_learned_better": {
            "vs_kernel_equal": float(pct_better_kernel),
            "vs_euclidean_knn": float(pct_better_knn),
            "vs_global": float(pct_better_global),
        },
        "likelihood_ratio_geom_mean": {
            "vs_kernel_equal": float(lr_kernel['geometric_mean']),
            "vs_euclidean_knn": float(lr_knn['geometric_mean']),
            "vs_global": float(lr_global['geometric_mean']),
        },
    }
    if args.learn_omega:
        config_out["omega_l2_reg"] = args.omega_l2_reg
        config_out["omega_constraint"] = args.omega_constraint
        config_out["max_iters"] = args.max_iters
        config_out["step_size"] = args.step_size

    with open(args.output_dir / "comparison_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {args.output_dir}/")
    print()
    print("Generated files:")
    print("  1. nll_boxplot.png - Box-and-whisker plot of NLL distributions (4 baselines)")
    print("  2. comparison_learned_vs_kernel_equal.png - Learned Omega vs Kernel(ω=1)")
    print("  3. comparison_learned_vs_euclidean_knn.png - Learned Omega vs Euclidean k-NN (binary)")
    print("  4. comparison_learned_vs_global.png - Learned Omega vs Global Cov (uniform)")
    if not args.skip_ellipsoids and Y.shape[1] >= 3:
        print("  5. ellipsoid_learned_vs_kernel_equal.png - 3D ellipsoid comparison")
        print("  6. ellipsoid_learned_vs_euclidean_knn.png - 3D ellipsoid comparison")
        print("  7. ellipsoid_learned_vs_global.png - 3D ellipsoid comparison")
        print("  8. ellipsoid_all_methods.png - All 4 methods overlaid")
    print("  N. comparison_config.json - Configuration used for this run")
    print()


if __name__ == "__main__":
    main()
