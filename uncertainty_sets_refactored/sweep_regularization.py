"""
Sweep over regularization and standardization options for covariance fitting.

Tests:
1. Standardized vs raw features
2. Different omega_l2_reg values
3. Different tau values
"""
from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from utils import fit_standard_scaler
from data_processing import build_XY_for_covariance_system_only
from covariance_optimization import (
    KernelCovConfig,
    FitConfig,
    CovPredictConfig,
    fit_omega,
    predict_mu_sigma_topk_cross,
)


def _mean_gaussian_nll(Y: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Mean NLL of Y under N(Mu, Sigma) row-wise."""
    N, M = Y.shape
    nll = np.empty(N, dtype=float)

    for i in range(N):
        S = Sigma[i]
        r = (Y[i] - Mu[i]).reshape(M, 1)
        L = np.linalg.cholesky(S)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        x = np.linalg.solve(L, r)
        x = np.linalg.solve(L.T, x)
        quad = float(r.T @ x)
        nll[i] = 0.5 * (logdet + quad + M * np.log(2.0 * np.pi))

    return float(nll.mean())


def run_sweep(
    forecasts_parquet: Path,
    actuals_parquet: Path,
    out_csv: Path,
    *,
    standardize_options=(True, False),
    taus=(1.0, 5.0, 20.0),
    omega_l2_regs=(0.0, 0.01, 0.1, 1.0),
    k: int = 128,  # Number of neighbors (hyperparameter, not just for speed)
    ridge: float = 1e-3,
    max_iters: int = 100,  # Reduced from 300 (3x speedup, usually converges by 100)
    step_size: float = 0.1,
    train_frac: float = 0.75,
):
    """Run sweep over standardization and regularization options."""
    actuals = pd.read_parquet(actuals_parquet)
    forecasts = pd.read_parquet(forecasts_parquet)

    X_raw, Y, times, x_cols, y_cols = build_XY_for_covariance_system_only(
        forecasts, actuals, drop_any_nan_rows=True
    )

    # Prepare both standardized and raw versions
    scaler = fit_standard_scaler(X_raw)
    X_std = scaler.transform(X_raw)

    # Random train/eval split (more representative than temporal holdout)
    n = X_raw.shape[0]
    n_train = int(train_frac * n)

    # Fixed random seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx = np.sort(indices[:n_train])  # Sort for deterministic behavior
    eval_idx = np.sort(indices[n_train:])

    print(f"Data: {n} samples, {X_raw.shape[1]} features")
    print(f"Train: {n_train}, Eval: {n - n_train}")
    print(f"Split: Random (seed=42) - more representative than temporal holdout")
    print(f"Raw feature ranges: {X_raw.min(axis=0)} to {X_raw.max(axis=0)}")
    print()

    rows = []

    for standardize, tau, omega_l2_reg in itertools.product(
        standardize_options, taus, omega_l2_regs
    ):
        X = X_std if standardize else X_raw
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_eval, Y_eval = X[eval_idx], Y[eval_idx]

        # Initial omega: [1, 1] for standardized, feature variances for raw
        if standardize:
            omega0 = np.ones(X.shape[1], dtype=float)
        else:
            # Start with inverse variance as initial guess
            omega0 = 1.0 / (X_train.var(axis=0) + 1e-6)

        cfg = KernelCovConfig(tau=float(tau), ridge=float(ridge))
        fit_cfg = FitConfig(
            max_iters=max_iters,
            step_size=float(step_size),
            grad_clip=10.0,
            tol=1e-7,
            verbose_every=999999,
            dtype="float32",
            device="cpu",
            omega_l2_reg=float(omega_l2_reg),
        )

        # Fit omega on training set
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
        )

        # Learned omega eval
        Mu_eval, Sigma_eval = predict_mu_sigma_topk_cross(
            X_query=X_eval,
            X_ref=X_train,
            Y_ref=Y_train,
            omega=omega_hat,
            cfg=pred_cfg,
            k=k,
            exclude_self_if_same=False,
            return_type="numpy",
        )
        nll_learned = _mean_gaussian_nll(Y_eval, Mu_eval, Sigma_eval)

        # Equal weights baseline (always [1, 1, ...] for fair comparison)
        omega_baseline = np.ones(X.shape[1], dtype=float)
        Mu_base, Sigma_base = predict_mu_sigma_topk_cross(
            X_query=X_eval,
            X_ref=X_train,
            Y_ref=Y_train,
            omega=omega_baseline,
            cfg=pred_cfg,
            k=k,
            exclude_self_if_same=False,
            return_type="numpy",
        )
        nll_baseline = _mean_gaussian_nll(Y_eval, Mu_base, Sigma_base)

        # Compute improvement
        nll_improvement = nll_baseline - nll_learned

        row = {
            "standardize": standardize,
            "tau": tau,
            "omega_l2_reg": omega_l2_reg,
            "omega_0": float(omega_hat[0]),
            "omega_1": float(omega_hat[1]),
            "omega_ratio": float(omega_hat[0] / omega_hat[1]) if omega_hat[1] > 0 else np.nan,
            "train_nll_final": float(last.get("nll_loss", last["loss"])),
            "eval_nll_learned": nll_learned,
            "eval_nll_baseline": nll_baseline,
            "nll_improvement": nll_improvement,
            "omega_reg_term": float(last.get("omega_reg", 0.0)),
        }
        rows.append(row)

        status = "BETTER" if nll_improvement > 0 else "WORSE"
        print(
            f"std={standardize}, tau={tau:5.1f}, reg={omega_l2_reg:.2f} => "
            f"omega=[{omega_hat[0]:.3f}, {omega_hat[1]:.3f}], "
            f"eval_nll={nll_learned:.3f} vs baseline={nll_baseline:.3f} ({status})"
        )

    # Save results
    res = pd.DataFrame(rows).sort_values("nll_improvement", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")
    print("\nTop 10 by NLL improvement:")
    print(res.head(10).to_string(index=False))

    return res


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    ART = DATA_DIR / "viz_artifacts"

    run_sweep(
        forecasts_parquet=DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet",
        actuals_parquet=DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet",
        out_csv=ART / "regularization_sweep_results.csv",
        standardize_options=[True, False],
        taus=[1.0, 5.0, 20.0],
        omega_l2_regs=[0.0, 0.01, 0.1, 1.0],
        k=128,
        ridge=1e-3,
        max_iters=300,
        step_size=0.1,
    )
