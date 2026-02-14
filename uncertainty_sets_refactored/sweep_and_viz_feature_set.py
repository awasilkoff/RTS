"""
Unified sweep and visualization for different feature sets.

Supports three feature engineering approaches:
1. temporal_3d: [SYS_MEAN, SYS_STD, HOUR_SIN]
2. per_resource_4d: [WIND_122_MEAN, WIND_309_MEAN, WIND_317_MEAN, HOUR_SIN]
3. unscaled_2d: [SYS_MEAN_MW, SYS_STD_MW] (raw, unscaled)

Usage:
    python sweep_and_viz_feature_set.py --feature-set temporal_3d
    python sweep_and_viz_feature_set.py --feature-set per_resource_4d
    python sweep_and_viz_feature_set.py --feature-set unscaled_2d
"""
from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import re
import numpy as np
import pandas as pd

from utils import fit_standard_scaler, fit_scaler, apply_scaler
from data_processing_extended import (
    build_XY_temporal_nuisance_3d,
    build_XY_per_resource_4d,
    build_XY_unscaled_2d,
    build_XY_focused_2d,
    build_XY_high_dim_8d,
    FEATURE_BUILDERS,
    FEATURE_SET_DESCRIPTIONS,
)
from covariance_optimization import (
    KernelCovConfig,
    FitConfig,
    CovPredictConfig,
    fit_omega,
    predict_mu_sigma_topk_cross,
    predict_mu_sigma_knn,
)
from viz_artifacts_utils import (
    setup_feature_set_directory,
    save_sweep_summary,
    update_readme_file_list,
    resolve_residuals_config,
)
from viz_projections import (
    plot_kernel_distance_2d_projection,
    plot_omega_bar_chart,
)


# Feature set dispatch - imported from data_processing_extended


def _per_point_gaussian_nll(
    Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray
) -> np.ndarray:
    """Per-point NLL of Y under N(Mu, Sigma) row-wise."""
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
            nll[i] = 1e6  # Large NLL if Cholesky fails

    return nll


def _mean_gaussian_nll(Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Mean NLL of Y under N(Mu, Sigma) row-wise."""
    return float(_per_point_gaussian_nll(Y, Mu, Sigma).mean())


def run_sweep(
    feature_set: str,
    forecasts_parquet: Path,
    actuals_parquet: Path,
    artifact_dir: Path,
    *,
    target_col: str = "ACTUAL",
    scaler_types: tuple[str, ...] = ("standard", "minmax"),
    taus: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0),
    omega_l2_regs: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0),
    omega_constraints: tuple[str, ...] = (
        "none",
    ),  # "none", "softmax", "simplex", or "normalize"
    k: int = 64,  # Number of neighbors (tuned via sweep_knn_k_values.py)
    ridge: float = 1e-3,
    max_iters: int = 250,  # Reduced from 300 (3x speedup, usually converges by 100)
    step_size: float = 0.1,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    # test_frac = 1 - train_frac - val_frac = 0.20
):
    """
    Run hyperparameter sweep for a given feature set.

    Returns
    -------
    sweep_df : DataFrame
        Results of sweep
    best_row_idx : int
        Index of best configuration
    omega_best : ndarray
        Best learned omega
    X_raw, Y, times, x_cols, y_cols : full dataset
    """
    # Load data using appropriate feature builder
    build_fn = FEATURE_BUILDERS[feature_set]

    actuals = pd.read_parquet(actuals_parquet)
    forecasts = pd.read_parquet(forecasts_parquet)

    X_raw, Y, times, x_cols, y_cols = build_fn(
        forecasts, actuals, actual_col=target_col, drop_any_nan_rows=True
    )

    print(f"Feature set: {feature_set}")
    print(f"  Description: {FEATURE_SET_DESCRIPTIONS[feature_set]}")
    print(f"  Data shape: X={X_raw.shape}, Y={Y.shape}")
    print(f"  Features: {x_cols}")
    print(f"  Targets: {y_cols}")
    print()

    # Random train/val/test split (proper 3-way for unbiased evaluation)
    # - Train: fit omega
    # - Val: select hyperparameters (tau, l2_reg)
    # - Test: final unbiased evaluation (used ONCE at the end)
    n = X_raw.shape[0]
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    # Fixed random seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train : n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val :])

    test_frac = 1.0 - train_frac - val_frac
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Split: Random {train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%} (seed=42)")
    print(f"  - Val used for hyperparameter selection")
    print(f"  - Test used ONCE for final unbiased evaluation")
    print()

    rows = []

    # Build grid: when using omega constraints (softmax/simplex/normalize), L2 reg is ignored
    # So we create (constraint, l2_reg) pairs where constrained methods use l2_reg=0.0
    constraint_reg_pairs = []
    for constraint in omega_constraints:
        if constraint == "none":
            # L2 reg matters for unconstrained optimization
            for reg in omega_l2_regs:
                constraint_reg_pairs.append((constraint, reg))
        else:
            # L2 reg is ignored for constrained optimization
            constraint_reg_pairs.append((constraint, 0.0))

    for scaler_type, tau, (omega_constraint, omega_l2_reg) in itertools.product(
        scaler_types, taus, constraint_reg_pairs
    ):
        # Fit scaler on training data only
        scaler = fit_scaler(X_raw[train_idx], scaler_type)
        X = apply_scaler(X_raw, scaler)

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # Initial omega
        if scaler_type == "none":
            # Start with inverse variance for raw features
            omega0 = 1.0 / (X_train.var(axis=0) + 1e-6)
        else:
            # Scaled features start with equal weights
            omega0 = np.ones(X.shape[1], dtype=float)

        # Use zero_mean=True for residuals (mean already removed by forecast)
        use_zero_mean = target_col == "RESIDUAL"

        cfg = KernelCovConfig(
            tau=float(tau),
            ridge=float(ridge),
            zero_mean=use_zero_mean,
        )
        fit_cfg = FitConfig(
            max_iters=max_iters,
            step_size=float(step_size),
            grad_clip=10.0,
            tol=1e-7,
            verbose_every=999999,
            dtype="float32",
            device="cpu",
            omega_l2_reg=float(omega_l2_reg),
            omega_constraint=omega_constraint,
            # k_fit=None: use all neighbors during training (better omega learning)
            # Prediction uses top-k which filters noise - this "mismatch" is beneficial
        )

        # Fit omega
        omega_hat, hist = fit_omega(
            X,
            Y,
            omega0=omega0,
            train_idx=train_idx,
            cfg=cfg,
            fit_cfg=fit_cfg,
            return_history=True,
        )

        dfh = pd.DataFrame(hist)
        last = dfh.iloc[-1]

        # Evaluate on held-out set
        pred_cfg = CovPredictConfig(
            tau=float(tau),
            ridge=float(ridge),
            enforce_nonneg_omega=True,
            dtype="float32",
            device="cpu",
            zero_mean=use_zero_mean,
        )

        # Learned omega eval
        Mu_eval, Sigma_eval = predict_mu_sigma_topk_cross(
            X_query=X_val,
            X_ref=X_train,
            Y_ref=Y_train,
            omega=omega_hat,
            cfg=pred_cfg,
            k=k,
            exclude_self_if_same=False,
            return_type="numpy",
        )
        nll_learned = _mean_gaussian_nll(Y_val, Mu_eval, Sigma_eval)

        # Equal weights k-NN baseline
        # For normalized constraints (softmax, simplex, normalize), use [1/d, ...] so sum=1
        # For unconstrained (none), use [1, 1, ...] to match historical behavior
        d = X.shape[1]
        if omega_constraint in ("softmax", "simplex", "normalize"):
            omega_baseline = np.ones(d, dtype=float) / d
        else:
            omega_baseline = np.ones(d, dtype=float)
        Mu_base, Sigma_base = predict_mu_sigma_topk_cross(
            X_query=X_val,
            X_ref=X_train,
            Y_ref=Y_train,
            omega=omega_baseline,
            cfg=pred_cfg,
            k=k,
            exclude_self_if_same=False,
            return_type="numpy",
        )
        nll_baseline = _mean_gaussian_nll(Y_val, Mu_base, Sigma_base)

        # Global covariance baseline (no adaptation - uses all training data)
        # This is like k-NN with k=N (all points equally weighted)
        Mu_global = np.mean(Y_train, axis=0)  # Global mean
        Sigma_global = np.cov(Y_train, rowvar=False)  # Global covariance
        # Add small ridge for numerical stability
        Sigma_global += ridge * np.eye(Sigma_global.shape[0])

        # Predict using same global mean/cov for all eval points
        Mu_global_eval = np.tile(Mu_global, (len(Y_val), 1))
        Sigma_global_eval = np.tile(Sigma_global[None, :, :], (len(Y_val), 1, 1))
        nll_global = _mean_gaussian_nll(Y_val, Mu_global_eval, Sigma_global_eval)

        # Euclidean k-NN baseline (true uniform 1/k weights)
        Mu_euclidean, Sigma_euclidean = predict_mu_sigma_knn(
            X_query=X_val,
            X_ref=X_train,
            Y_ref=Y_train,
            k=k,
            ridge=ridge,
        )
        nll_euclidean = _mean_gaussian_nll(Y_val, Mu_euclidean, Sigma_euclidean)

        # Compute per-point NLL for detailed metrics
        nll_learned_per_point = _per_point_gaussian_nll(Y_val, Mu_eval, Sigma_eval)
        nll_baseline_per_point = _per_point_gaussian_nll(Y_val, Mu_base, Sigma_base)
        nll_euclidean_per_point = _per_point_gaussian_nll(
            Y_val, Mu_euclidean, Sigma_euclidean
        )
        nll_global_per_point = _per_point_gaussian_nll(
            Y_val, Mu_global_eval, Sigma_global_eval
        )

        # Improvements over baselines (mean NLL)
        nll_improvement = nll_baseline - nll_learned
        nll_improvement_vs_global = nll_global - nll_learned
        nll_improvement_vs_euclidean = nll_euclidean - nll_learned

        # Percentage of samples where learned is better
        pct_better_kernel = 100.0 * np.mean(
            nll_learned_per_point < nll_baseline_per_point
        )
        pct_better_euclidean = 100.0 * np.mean(
            nll_learned_per_point < nll_euclidean_per_point
        )
        pct_better_global = 100.0 * np.mean(
            nll_learned_per_point < nll_global_per_point
        )

        # Likelihood ratio (geometric mean)
        def _geom_mean_likelihood_ratio(nll_base, nll_learned):
            nll_diff = np.clip(nll_base - nll_learned, -50, 50)
            return float(np.exp(np.mean(nll_diff)))

        lr_kernel = _geom_mean_likelihood_ratio(
            nll_baseline_per_point, nll_learned_per_point
        )
        lr_euclidean = _geom_mean_likelihood_ratio(
            nll_euclidean_per_point, nll_learned_per_point
        )
        lr_global = _geom_mean_likelihood_ratio(
            nll_global_per_point, nll_learned_per_point
        )

        row = {
            "scaler_type": scaler_type,
            "tau": tau,
            "omega_l2_reg": omega_l2_reg,
            "omega_constraint": omega_constraint,
            "train_nll_final": float(last.get("nll_loss", last["loss"])),
            "val_nll_learned": nll_learned,
            "val_nll_kernel_equal": nll_baseline,
            "val_nll_euclidean_knn": nll_euclidean,
            "val_nll_global": nll_global,
            "nll_improvement_vs_kernel": nll_improvement,
            "nll_improvement_vs_euclidean": nll_improvement_vs_euclidean,
            "nll_improvement_vs_global": nll_improvement_vs_global,
            "pct_better_vs_kernel": pct_better_kernel,
            "pct_better_vs_euclidean": pct_better_euclidean,
            "pct_better_vs_global": pct_better_global,
            "likelihood_ratio_vs_kernel": lr_kernel,
            "likelihood_ratio_vs_euclidean": lr_euclidean,
            "likelihood_ratio_vs_global": lr_global,
            "omega_reg_term": float(last.get("omega_reg", 0.0)),
            # Backward compatibility aliases
            "nll_improvement": nll_improvement,  # alias for nll_improvement_vs_kernel
            "val_nll_baseline": nll_baseline,  # alias for val_nll_kernel_equal
        }

        # Add per-feature omega values
        for i, col in enumerate(x_cols):
            row[f"omega_{i}_{col}"] = float(omega_hat[i])

        rows.append(row)

        status_kernel = "+" if nll_improvement > 0 else "-"
        status_euclidean = "+" if nll_improvement_vs_euclidean > 0 else "-"
        status_global = "+" if nll_improvement_vs_global > 0 else "-"
        omega_str = ", ".join([f"{w:.3f}" for w in omega_hat])
        constraint_str = f"/{omega_constraint}" if omega_constraint != "none" else ""
        print(
            f"scaler={scaler_type:8s}, tau={tau:5.3f}, reg={omega_l2_reg:.2f}{constraint_str} => "
            f"ω=[{omega_str}], "
            f"NLL: {nll_learned:.3f} vs kernel({status_kernel}){nll_baseline:.3f} "
            f"euclid({status_euclidean}){nll_euclidean:.3f} "
            f"global({status_global}){nll_global:.3f} | "
            f"win%: {pct_better_kernel:.0f}/{pct_better_euclidean:.0f}/{pct_better_global:.0f}"
        )

    # Find best configuration (sort by improvement vs kernel equal weights)
    sweep_df = pd.DataFrame(rows).sort_values(
        "nll_improvement_vs_kernel", ascending=False
    )

    best_row_idx = 0  # Rank in sorted dataframe
    best_row = sweep_df.iloc[0]

    # Reconstruct best omega (strict regex to avoid matching omega_l2_reg)
    omega_cols = [c for c in sweep_df.columns if re.match(r"^omega_\d+_", c)]
    omega_cols = sorted(
        omega_cols, key=lambda c: int(c.split("_")[1])
    )  # Sort by feature index
    omega_best = np.array([best_row[c] for c in omega_cols], dtype=float)

    # Sanity check: omega dimensionality should match number of features
    assert omega_best.shape[0] == len(x_cols), (
        f"Omega dimension mismatch: got {omega_best.shape[0]} values "
        f"but expected {len(x_cols)} features. Omega cols: {omega_cols}"
    )

    # Baseline label depends on constraint
    constraint = best_row["omega_constraint"]
    if constraint in ("softmax", "simplex", "normalize"):
        baseline_label = f"Uniform(ω=1/{len(x_cols)})"
    else:
        baseline_label = "Kernel(ω=1)"

    print(f"\nBest configuration (selected on VALIDATION set):")
    print(
        f"  scaler_type={best_row['scaler_type']}, tau={best_row['tau']}, "
        f"omega_l2_reg={best_row['omega_l2_reg']}, constraint={constraint}"
    )
    print(f"  Learned omega: {omega_best}")
    print(f"  Validation NLL:")
    print(f"    Learned:       {best_row['val_nll_learned']:.4f}")
    print(f"    {baseline_label}: {best_row['val_nll_kernel_equal']:.4f}")
    print(f"    Euclidean kNN: {best_row['val_nll_euclidean_knn']:.4f}")
    print(f"    Global:        {best_row['val_nll_global']:.4f}")

    # =========================================================================
    # FINAL EVALUATION ON HELD-OUT TEST SET (unbiased estimate)
    # =========================================================================
    print(f"\n--- Final Evaluation on TEST set (unbiased) ---")

    # Apply best scaler
    best_scaler = fit_scaler(X_raw[train_idx], best_row["scaler_type"])
    X_scaled = apply_scaler(X_raw, best_scaler)
    X_train_scaled = X_scaled[train_idx]
    Y_train_data = Y[train_idx]
    X_test = X_scaled[test_idx]
    Y_test = Y[test_idx]

    # Prediction config
    use_zero_mean = target_col == "RESIDUAL"
    pred_cfg = CovPredictConfig(
        tau=float(best_row["tau"]),
        ridge=float(ridge),
        enforce_nonneg_omega=True,
        dtype="float32",
        device="cpu",
        zero_mean=use_zero_mean,
    )

    # Learned omega on test
    Mu_test_learned, Sigma_test_learned = predict_mu_sigma_topk_cross(
        X_query=X_test,
        X_ref=X_train_scaled,
        Y_ref=Y_train_data,
        omega=omega_best,
        cfg=pred_cfg,
        k=k,
        exclude_self_if_same=False,
        return_type="numpy",
    )
    test_nll_learned = _mean_gaussian_nll(Y_test, Mu_test_learned, Sigma_test_learned)

    # Baseline omega on test
    d = len(x_cols)
    if constraint in ("softmax", "simplex", "normalize"):
        omega_baseline = np.ones(d, dtype=float) / d
    else:
        omega_baseline = np.ones(d, dtype=float)

    Mu_test_baseline, Sigma_test_baseline = predict_mu_sigma_topk_cross(
        X_query=X_test,
        X_ref=X_train_scaled,
        Y_ref=Y_train_data,
        omega=omega_baseline,
        cfg=pred_cfg,
        k=k,
        exclude_self_if_same=False,
        return_type="numpy",
    )
    test_nll_baseline = _mean_gaussian_nll(
        Y_test, Mu_test_baseline, Sigma_test_baseline
    )

    # Euclidean k-NN on test
    Mu_test_euclidean, Sigma_test_euclidean = predict_mu_sigma_knn(
        X_query=X_test,
        X_ref=X_train_scaled,
        Y_ref=Y_train_data,
        k=k,
        ridge=ridge,
    )
    test_nll_euclidean = _mean_gaussian_nll(
        Y_test, Mu_test_euclidean, Sigma_test_euclidean
    )

    # Global covariance on test
    Mu_global = np.mean(Y_train_data, axis=0)
    Sigma_global = np.cov(Y_train_data, rowvar=False) + ridge * np.eye(Y.shape[1])
    Mu_test_global = np.tile(Mu_global, (len(Y_test), 1))
    Sigma_test_global = np.tile(Sigma_global[None, :, :], (len(Y_test), 1, 1))
    test_nll_global = _mean_gaussian_nll(Y_test, Mu_test_global, Sigma_test_global)

    # Compute test improvements
    test_improvement_vs_baseline = test_nll_baseline - test_nll_learned
    test_improvement_vs_euclidean = test_nll_euclidean - test_nll_learned
    test_improvement_vs_global = test_nll_global - test_nll_learned

    print(f"  Test NLL (unbiased):")
    print(f"    Learned:       {test_nll_learned:.4f}")
    print(f"    {baseline_label}: {test_nll_baseline:.4f}")
    print(f"    Euclidean kNN: {test_nll_euclidean:.4f}")
    print(f"    Global:        {test_nll_global:.4f}")
    print(f"  Test Improvement (positive = learned is better):")
    print(f"    vs {baseline_label}: {test_improvement_vs_baseline:.4f}")
    print(f"    vs Euclidean kNN: {test_improvement_vs_euclidean:.4f}")
    print(f"    vs Global:        {test_improvement_vs_global:.4f}")

    # Store test results
    test_results = {
        "test_nll_learned": test_nll_learned,
        "test_nll_baseline": test_nll_baseline,
        "test_nll_euclidean": test_nll_euclidean,
        "test_nll_global": test_nll_global,
        "test_improvement_vs_baseline": test_improvement_vs_baseline,
        "test_improvement_vs_euclidean": test_improvement_vs_euclidean,
        "test_improvement_vs_global": test_improvement_vs_global,
    }

    return (
        sweep_df,
        best_row_idx,
        omega_best,
        X_raw,
        Y,
        times,
        x_cols,
        y_cols,
        test_results,
    )


def run_multi_seed_validation(
    feature_set: str,
    forecasts_parquet: Path,
    actuals_parquet: Path,
    *,
    best_config: dict,
    taus: tuple[float, ...],
    n_seeds: int = 10,
    k: int = 64,
    max_iters: int = 250,
    step_size: float = 0.1,
    target_col: str = "ACTUAL",
    train_frac: float = 0.5,
    val_frac: float = 0.25,
):
    """
    Multi-seed validation: re-run omega optimization at each tau with N random
    omega initializations, using the best config's l2_reg and constraint.

    Returns (results_df, stats_df, best_omega) where best_omega has lowest
    val_nll across all (tau, seed) combinations.
    """
    build_fn = FEATURE_BUILDERS[feature_set]
    actuals = pd.read_parquet(actuals_parquet)
    forecasts = pd.read_parquet(forecasts_parquet)
    X_raw, Y, times, x_cols, y_cols = build_fn(
        forecasts, actuals, actual_col=target_col, drop_any_nan_rows=True
    )

    # Same split as run_sweep (seed=42)
    n = X_raw.shape[0]
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    rng_split = np.random.RandomState(42)
    indices = rng_split.permutation(n)
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train : n_train + n_val])

    # Apply best scaler
    scaler_type = best_config.get("scaler_type", "standard")
    scaler = fit_scaler(X_raw[train_idx], scaler_type)
    X = apply_scaler(X_raw, scaler)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    omega_constraint = best_config.get("omega_constraint", "none")
    omega_l2_reg = best_config.get("omega_l2_reg", 0.0)
    ridge = best_config.get("ridge", 1e-3)
    use_zero_mean = target_col == "RESIDUAL"
    d = X.shape[1]

    print(f"\n{'=' * 80}")
    print(f"MULTI-SEED VALIDATION ({n_seeds} seeds x {len(taus)} taus)")
    print(
        f"  constraint={omega_constraint}, l2_reg={omega_l2_reg}, scaler={scaler_type}"
    )
    print(f"{'=' * 80}")

    rows = []
    best_nll = float("inf")
    best_omega = None

    for tau in taus:
        print(f"\ntau={tau}:")
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            omega0 = np.abs(rng.randn(d)) + 0.5

            cfg = KernelCovConfig(
                tau=float(tau), ridge=float(ridge), zero_mean=use_zero_mean
            )
            fit_cfg = FitConfig(
                max_iters=max_iters,
                step_size=float(step_size),
                grad_clip=10.0,
                tol=1e-7,
                verbose_every=999999,
                dtype="float32",
                device="cpu",
                omega_l2_reg=float(omega_l2_reg),
                omega_constraint=omega_constraint,
            )

            omega_hat, hist = fit_omega(
                X,
                Y,
                omega0=omega0,
                train_idx=train_idx,
                cfg=cfg,
                fit_cfg=fit_cfg,
                return_history=True,
            )

            dfh = pd.DataFrame(hist)
            train_nll = float(dfh.iloc[-1].get("nll_loss", dfh.iloc[-1]["loss"]))
            n_iters = len(dfh)

            pred_cfg = CovPredictConfig(
                tau=float(tau),
                ridge=float(ridge),
                enforce_nonneg_omega=True,
                dtype="float32",
                device="cpu",
                zero_mean=use_zero_mean,
            )
            Mu_val, Sigma_val = predict_mu_sigma_topk_cross(
                X_val,
                X_train,
                Y_train,
                omega=omega_hat,
                cfg=pred_cfg,
                k=k,
                exclude_self_if_same=False,
                return_type="numpy",
            )
            val_nll = _mean_gaussian_nll(Y_val, Mu_val, Sigma_val)

            row = {
                "tau": tau,
                "seed": seed,
                "val_nll": val_nll,
                "train_nll": train_nll,
                "n_iters": n_iters,
                "converged": n_iters < max_iters,
            }
            for i in range(len(omega_hat)):
                row[f"omega_{i}"] = float(omega_hat[i])
            rows.append(row)

            if val_nll < best_nll:
                best_nll = val_nll
                best_omega = omega_hat.copy()

            omega_str = ", ".join([f"{w:.3f}" for w in omega_hat])
            print(f"  seed={seed}: val_NLL={val_nll:.4f}, ω=[{omega_str}]")

    results_df = pd.DataFrame(rows)

    # Compute per-tau statistics
    omega_cols = [c for c in results_df.columns if re.match(r"^omega_\d+$", c)]
    stats_agg = (
        results_df.groupby("tau")
        .agg(
            {
                "val_nll": ["mean", "std", "min", "max"],
                "train_nll": ["mean", "std"],
            }
        )
        .reset_index()
    )
    stats_agg.columns = [
        "tau",
        "val_nll_mean",
        "val_nll_std",
        "val_nll_min",
        "val_nll_max",
        "train_nll_mean",
        "train_nll_std",
    ]
    for col in omega_cols:
        omega_agg = results_df.groupby("tau")[col].agg(["mean", "std"]).reset_index()
        stats_agg[f"{col}_mean"] = omega_agg["mean"]
        stats_agg[f"{col}_std"] = omega_agg["std"]

    stats_df = stats_agg

    best_tau_idx = stats_df["val_nll_mean"].idxmin()
    best_tau = stats_df.loc[best_tau_idx, "tau"]
    best_mean = stats_df.loc[best_tau_idx, "val_nll_mean"]
    best_std = stats_df.loc[best_tau_idx, "val_nll_std"]
    print(f"\nBest tau (by mean): {best_tau} (NLL = {best_mean:.4f} ± {best_std:.4f})")
    print(
        f"Best overall: tau={results_df.loc[results_df['val_nll'].idxmin(), 'tau']}, "
        f"seed={results_df.loc[results_df['val_nll'].idxmin(), 'seed']}, "
        f"NLL={best_nll:.4f}"
    )

    return results_df, stats_df, best_omega


def plot_nll_vs_tau(
    sweep_df: pd.DataFrame,
    artifact_dir: Path,
    feature_set: str,
    scaler_type: str = None,
):
    """
    Plot NLL vs tau curve showing hyperparameter tuning.

    Only shows learned omega (no equal omega baseline).
    """
    import matplotlib.pyplot as plt
    from plot_config import setup_plotting

    setup_plotting()

    # Filter to best scaler if specified, otherwise use all
    if scaler_type is not None:
        df = sweep_df[sweep_df["scaler_type"] == scaler_type].copy()
    else:
        # Use the scaler type from the best config
        best_scaler = sweep_df.iloc[0]["scaler_type"]
        df = sweep_df[sweep_df["scaler_type"] == best_scaler].copy()
        scaler_type = best_scaler

    # Group by tau and get mean NLL (in case of multiple L2 regs)
    tau_nll = df.groupby("tau")["val_nll_learned"].min().reset_index()
    tau_nll = tau_nll.sort_values("tau")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot learned omega NLL vs tau
    ax.plot(
        tau_nll["tau"],
        tau_nll["val_nll_learned"],
        "o-",
        linewidth=2.5,
        markersize=10,
        color="#E63946",
        label="Learned ω",
    )

    # Mark best tau
    best_idx = tau_nll["val_nll_learned"].idxmin()
    best_tau = tau_nll.loc[best_idx, "tau"]
    best_nll = tau_nll.loc[best_idx, "val_nll_learned"]

    ax.scatter(
        [best_tau],
        [best_nll],
        s=300,
        c="#E63946",
        marker="*",
        zorder=10,
        edgecolors="black",
        linewidths=2,
        label=f"Best: τ={best_tau}",
    )

    # Styling
    ax.set_xlabel("τ (Kernel Bandwidth)")
    ax.set_ylabel("Validation NLL")
    ax.set_title(f"Hyperparameter Tuning: {feature_set} (scaler={scaler_type})")
    ax.set_xscale("log")
    ax.legend(loc="upper right")

    plt.tight_layout()

    # Save
    output_path = artifact_dir / "nll_vs_tau.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved NLL vs tau plot to {output_path}")

    output_path_pdf = artifact_dir / "nll_vs_tau.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()

    return best_tau, best_nll


def generate_visualizations(
    feature_set: str,
    artifact_dir: Path,
    X_raw: np.ndarray,
    Y: np.ndarray,
    times: pd.DatetimeIndex,
    x_cols: list[str],
    y_cols: list[str],
    omega_best: np.ndarray,
    tau_best: float,
    scaler_type_best: str,
    omega_constraint_best: str = "none",
    sweep_df: pd.DataFrame = None,
):
    """Generate feature-set-specific visualizations."""
    print("\nGenerating visualizations...")

    # 0. NLL vs tau hyperparameter tuning plot
    if sweep_df is not None:
        plot_nll_vs_tau(sweep_df, artifact_dir, feature_set, scaler_type_best)

    # Apply best scaler
    scaler = fit_scaler(X_raw, scaler_type_best)
    X = apply_scaler(X_raw, scaler)

    # 1. Omega bar chart (always generate)
    plot_omega_bar_chart(
        omega_best,
        x_cols,
        out_path=artifact_dir / "omega_bar_chart.png",
        title=f"Learned Feature Weights: {feature_set}",
        highlight_top_k=2 if len(omega_best) > 2 else len(omega_best),
    )

    # 2. Kernel distance visualization
    # Pick a sample index for visualization (middle of dataset is safe with random split)
    sample_idx = len(X) // 2  # Middle sample (guaranteed to be valid)

    if len(omega_best) > 2:
        # High-D: use projection
        plot_kernel_distance_2d_projection(
            X,
            omega_best,
            x_cols,
            times,
            target_idx=sample_idx,
            tau=tau_best,
            projection_method="top2",
            out_path=artifact_dir / "kernel_distance_projection_2d.png",
            title_suffix=feature_set,
        )
    else:
        # 2D: use standard visualization
        from viz_kernel_distance import plot_kernel_distance_comparison

        # Use [1/d, ...] baseline for normalized constraints, [1, 1, ...] otherwise
        d = len(omega_best)
        if omega_constraint_best in ("softmax", "simplex", "normalize"):
            omega_equal = np.ones(d) / d
        else:
            omega_equal = np.ones(d)
        plot_kernel_distance_comparison(
            X,
            x_cols,
            times,
            sample_idx,
            omega_equal,
            omega_best,
            tau_best,
            save_path=artifact_dir / "kernel_distance_comparison.png",
        )

    print("Visualizations complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep and visualize different feature sets for learned omega"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        required=True,
        choices=list(FEATURE_BUILDERS.keys()),
        help="Feature set to use",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Data directory",
    )
    parser.add_argument(
        "--scaler-types",
        nargs="+",
        type=str,
        default=[
            "standard",
        ],
        choices=["none", "standard", "minmax"],
        help="Scaler types to test",
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
            20.0,
        ],
        help="Tau values to sweep (finer at low tau where performance is better)",
    )
    parser.add_argument(
        "--omega-l2-regs",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.1, 1.0],
        help="Omega L2 regularization values (only used when omega-constraint=none)",
    )
    parser.add_argument(
        "--omega-constraints",
        nargs="+",
        type=str,
        default=["softmax"],
        choices=["none", "softmax", "simplex", "normalize"],
        help="Omega constraint types to sweep: none (L2 reg), softmax (sum=1 via softmax), "
        "simplex (projected sum=1), normalize (post-hoc normalize)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of neighbors for k-NN (default 16, tuned via sweep_knn_k_values.py)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix to append to output directory name (e.g., '_softmax' -> 'focused_2d_softmax')",
    )
    parser.add_argument(
        "--use-residuals",
        action="store_true",
        help="Use residuals (ACTUAL - MEAN_FORECAST) instead of raw actuals as target",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Multi-seed validation at best config (default: 1 = skip)",
    )

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    feature_set = args.feature_set
    output_suffix = args.output_suffix

    # Determine data source and target column based on --use-residuals flag
    rcfg = resolve_residuals_config(args.use_residuals, DATA_DIR)
    data_parquet = rcfg["actuals_parquet"]
    target_col = rcfg["actual_col"]
    if args.use_residuals:
        print("\n*** Using RESIDUALS as target (ACTUAL - MEAN_FORECAST) ***\n")
    else:
        print("\n*** Using raw ACTUALS as target ***\n")

    # Auto-append _residuals suffix when using residuals and no explicit suffix given
    if args.use_residuals and not output_suffix:
        output_suffix = "_residuals"

    # Build output directory name with optional suffix
    output_name = f"{feature_set}{output_suffix}" if output_suffix else feature_set

    # Setup artifact directory
    artifact_dir = setup_feature_set_directory(
        feature_set_name=output_name,
        feature_config={
            "description": FEATURE_SET_DESCRIPTIONS[feature_set],
            "feature_set_base": feature_set,  # Original feature set name
            "output_suffix": output_suffix,
            "use_residuals": args.use_residuals,
            "target_col": target_col,
            "scaler_types": args.scaler_types,
            "taus": args.taus,
            "omega_l2_regs": args.omega_l2_regs,
            "omega_constraints": args.omega_constraints,
        },
        base_dir=DATA_DIR / "viz_artifacts",
    )

    # Run sweep
    print("=" * 80)
    print(f"Running sweep for feature set: {feature_set}")
    print("=" * 80)

    (
        sweep_df,
        best_row_idx,
        omega_best,
        X_raw,
        Y,
        times,
        x_cols,
        y_cols,
        test_results,
    ) = run_sweep(
        feature_set=feature_set,
        forecasts_parquet=DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet",
        actuals_parquet=data_parquet,
        artifact_dir=artifact_dir,
        target_col=target_col,
        scaler_types=tuple(args.scaler_types),
        taus=tuple(args.taus),
        omega_l2_regs=tuple(args.omega_l2_regs),
        omega_constraints=tuple(args.omega_constraints),
        k=args.k,
    )

    # Save results (including test results)
    save_sweep_summary(
        artifact_dir, sweep_df, best_row_idx, omega_best, test_results=test_results
    )

    # Multi-seed validation (optional)
    if args.n_seeds > 1:
        best_row_ms = sweep_df.iloc[best_row_idx]
        best_config = {
            "tau": float(best_row_ms["tau"]),
            "omega_l2_reg": float(best_row_ms["omega_l2_reg"]),
            "omega_constraint": str(best_row_ms.get("omega_constraint", "none")),
            "scaler_type": str(best_row_ms["scaler_type"]),
            "ridge": 1e-3,
        }
        ms_results, ms_stats, ms_best_omega = run_multi_seed_validation(
            feature_set=feature_set,
            forecasts_parquet=DATA_DIR
            / "forecasts_filtered_rts3_constellation_v1.parquet",
            actuals_parquet=data_parquet,
            best_config=best_config,
            taus=tuple(args.taus),
            n_seeds=args.n_seeds,
            k=args.k,
            target_col=target_col,
        )
        ms_results.to_csv(artifact_dir / "multi_seed_results.csv", index=False)
        ms_stats.to_csv(artifact_dir / "multi_seed_stats.csv", index=False)
        # Update best_omega if multi-seed found better
        np.save(artifact_dir / "best_omega.npy", ms_best_omega)
        omega_best = ms_best_omega
        print(f"\nSaved multi-seed results to {artifact_dir}/")

    # Update config with actual feature info
    best_row = sweep_df.iloc[best_row_idx]
    from viz_artifacts_utils import load_feature_config
    import json

    config = load_feature_config(artifact_dir)
    config.update(
        {
            "x_cols": x_cols,
            "y_cols": y_cols,
            "n_features": len(x_cols),
            "scaler_type": str(best_row["scaler_type"]),
            "tau": float(best_row["tau"]),
            "omega_l2_reg": float(best_row["omega_l2_reg"]),
            "omega_constraint": str(best_row.get("omega_constraint", "none")),
            "ridge": 1e-3,
            "k": 128,
        }
    )
    with open(artifact_dir / "feature_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate visualizations
    generate_visualizations(
        feature_set=feature_set,
        artifact_dir=artifact_dir,
        X_raw=X_raw,
        Y=Y,
        times=times,
        x_cols=x_cols,
        y_cols=y_cols,
        omega_best=omega_best,
        tau_best=best_row["tau"],
        scaler_type_best=best_row["scaler_type"],
        omega_constraint_best=best_row["omega_constraint"],
        sweep_df=sweep_df,
    )

    # Update README with file list
    update_readme_file_list(artifact_dir)

    print(f"\nAll outputs saved to: {artifact_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
