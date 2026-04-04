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
    """
    Mean NLL of Y under N(Mu, Sigma) row-wise.
    Y:     (N, M)
    Mu:    (N, M)
    Sigma: (N, M, M)
    """
    import numpy as _np

    Y = _np.asarray(Y, dtype=float)
    Mu = _np.asarray(Mu, dtype=float)
    Sigma = _np.asarray(Sigma, dtype=float)

    N, M = Y.shape
    nll = _np.empty(N, dtype=float)

    # Per-row Cholesky
    for i in range(N):
        S = Sigma[i]
        r = (Y[i] - Mu[i]).reshape(M, 1)
        # Cholesky; if this fails, your ridge/jitter is too small
        L = _np.linalg.cholesky(S)
        logdet = 2.0 * _np.log(_np.diag(L)).sum()
        # solve S^{-1} r via Cholesky
        x = _np.linalg.solve(L, r)
        x = _np.linalg.solve(L.T, x)
        quad = float(r.T @ x)
        nll[i] = 0.5 * (logdet + quad + M * _np.log(2.0 * _np.pi))

    return float(nll.mean())


def run_sweep(
    forecasts_parquet: Path,
    actuals_parquet: Path,
    out_csv: Path,
    *,
    taus=(5.0, 10.0, 20.0, 40.0),
    ks=(32, 64, 128, 256),
    ridge: float = 1e-3,
    max_iters: int = 150,
    train_frac: float = 0.75,
    step_sizes=(1e-3, 3e-3, 1e-2),
    grad_clips=(None, 10.0),
    dtype_list=("float32",),
    seeds=(0,),
):
    actuals = pd.read_parquet(actuals_parquet)
    forecasts = pd.read_parquet(forecasts_parquet)

    X, Y, times, x_cols, y_cols = build_XY_for_covariance_system_only(
        forecasts, actuals, drop_any_nan_rows=True
    )

    scaler = fit_standard_scaler(X)
    Xs = scaler.transform(X)

    n = Xs.shape[0]
    n_train = int(train_frac * n)
    train_idx = np.arange(n_train)
    eval_idx = np.arange(n_train, n)

    omega0 = np.ones(Xs.shape[1], dtype=float)

    X_train, Y_train = Xs[train_idx], Y[train_idx]
    X_eval, Y_eval = Xs[eval_idx], Y[eval_idx]

    rows = []

    for (tau, k, step_size, grad_clip, dtype, seed) in itertools.product(
        taus, ks, step_sizes, grad_clips, dtype_list, seeds
    ):
        fit_cfg = FitConfig(
            max_iters=max_iters,
            step_size=float(step_size),
            grad_clip=grad_clip,
            tol=0.0,  # no early stop during sweep
            verbose_every=999999,
            dtype=dtype,
            device="cpu",
        )
        cfg = KernelCovConfig(tau=float(tau), ridge=float(ridge))

        # Fit omega on TRAIN only
        omega_hat, hist = fit_omega(
            Xs,
            Y,
            omega0=omega0,
            train_idx=train_idx,
            cfg=cfg,
            fit_cfg=fit_cfg,
            return_history=True,
        )

        dfh = pd.DataFrame(hist)
        last = dfh.iloc[-1]

        # Eval moments using TRAIN as reference set (no leakage)
        pred_cfg = CovPredictConfig(
            tau=float(tau),
            ridge=float(ridge),
            enforce_nonneg_omega=True,
            dtype=dtype,
            device="cpu",
        )
        Mu_eval, Sigma_eval = predict_mu_sigma_topk_cross(
            X_query=X_eval,
            X_ref=X_train,
            Y_ref=Y_train,
            omega=omega_hat,
            cfg=pred_cfg,
            k=int(k),
            exclude_self_if_same=True,
            return_type="numpy",
        )

        loss_eval_mean = _mean_gaussian_nll(Y_eval, Mu_eval, Sigma_eval)

        rows.append(
            {
                "tau": float(tau),
                "k": int(k),
                "step_size": float(step_size),
                "grad_clip": grad_clip if grad_clip is not None else "None",
                "dtype": dtype,
                "seed": seed,
                "train_loss_final": float(last["loss"]),
                "train_loss_start": float(dfh.iloc[0]["loss"]),
                "train_loss_delta": float(dfh.iloc[0]["loss"] - last["loss"]),
                "loss_eval_mean": float(loss_eval_mean),
                "grad_norm_final": float(last["grad_norm"]),
                "omega_nnz_final": int(last["omega_nnz"]),
                "omega_min_final": float(last.get("omega_min", np.nan)),
                "omega_max_final": float(last["omega_max"]),
                "omega_mean_final": float(last["omega_mean"]),
            }
        )

        # Optional per-iter history
        run_tag = f"tau{tau}_k{k}_lr{step_size}_clip{grad_clip}_dtype{dtype}_seed{seed}".replace(
            ".", "p"
        ).replace(
            "None", "none"
        )
        dfh.to_csv(
            out_csv.with_suffix("").with_name(out_csv.stem + f"__hist__{run_tag}.csv"),
            index=False,
        )

    res = pd.DataFrame(rows).sort_values(["loss_eval_mean", "train_loss_final"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)
    print(res.head(15))


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    ART = DATA_DIR / "viz_artifacts"

    run_sweep(
        forecasts_parquet=DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet",
        actuals_parquet=DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet",
        out_csv=ART / "cov_fit_sweep_summary_rts3.csv",
        taus=(1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
        ks=(16, 32, 64, 128, 256),
        ridge=1e-3,
        max_iters=300,
        step_sizes=(1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1),
        grad_clips=(None, 5.0, 10.0),
        dtype_list=("float32",),
    )
