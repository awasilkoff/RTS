from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, overload
import logging

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

logger = logging.getLogger(__name__)


# -----------------------------
# Configs
# -----------------------------


@dataclass(frozen=True)
class KernelCovConfig:
    """Configuration for kernel-weighted local moment estimation."""

    tau: float
    ridge: float = 1e-4
    enforce_nonneg_omega: bool = True


@dataclass(frozen=True)
class FitConfig:
    """Optimizer configuration for fitting omega."""

    max_iters: int = 200
    step_size: float = 1e-3
    tol: float = 1e-6
    grad_clip: Optional[float] = 10.0
    verbose_every: int = 10
    projection_nonneg: bool = True
    dtype: str = "float32"  # "float32" or "float64"
    cholesky_jitter_tries: int = 8
    cholesky_jitter_mult: float = 10.0
    finite_check: bool = True
    device: str = "cpu"
    omega_l2_reg: float = 0.0  # L2 regularization on omega (only used when omega_constraint="none")
    omega_constraint: str = "none"  # "none", "softmax", "simplex", or "normalize"
    # - "none": No constraint on omega scale. L2 reg pulls toward 1.0.
    # - "softmax": Learn unconstrained alpha, omega = softmax(alpha). Ensures sum(omega)=1. L2 reg ignored.
    # - "simplex": Project omega onto probability simplex after each step. Ensures sum(omega)=1. L2 reg ignored.
    # - "normalize": Divide omega by sum(omega) at the end (post-hoc normalization). L2 reg ignored.


@dataclass(frozen=True)
class CovPredictConfig:
    """Prediction-time configuration for local moments."""

    tau: float
    ridge: float = 1e-4
    enforce_nonneg_omega: bool = True
    dtype: str = "float32"  # "float32" or "float64"
    device: str = "cpu"
    finite_check: bool = True


def _check_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for covariance_optimization. Install torch or vendor a numpy implementation."
        )


def _torch_dtype(dtype: str):
    _check_torch()
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype={dtype!r}; use 'float32' or 'float64'.")


def _as_2d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {a.shape}")
    return a


def _as_1d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {a.shape}")
    return a


def _project_simplex_torch(v: "torch.Tensor") -> "torch.Tensor":
    """
    Project vector v onto the probability simplex (sum=1, all >= 0).

    Uses the algorithm from "Efficient Projections onto the L1-Ball
    for Learning in High Dimensions" (Duchi et al., 2008).
    """
    n = v.shape[0]
    # Sort in descending order
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0)
    # Find rho: largest j such that u[j] - (cssv[j] - 1) / (j+1) > 0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - (cssv - 1) / ind > 0
    rho = torch.where(cond)[0][-1]  # Last index where condition holds
    theta = (cssv[rho] - 1) / (rho + 1)
    return torch.clamp(v - theta, min=0)


# -----------------------------
# Core math helpers (torch)
# -----------------------------


def _pairwise_sqdist(
    Xq: "torch.Tensor",
    Xr: "torch.Tensor",
    omega: "torch.Tensor",
) -> "torch.Tensor":
    """Compute weighted squared distances between query rows and reference rows.

    Returns D with shape (Nq, Nr): D[i, j] = sum_k omega[k] * (Xq[i,k]-Xr[j,k])^2
    """
    # Apply sqrt weights for numerical stability
    w = torch.sqrt(torch.clamp(omega, min=0.0))
    Xq_w = Xq * w
    Xr_w = Xr * w
    aq = (Xq_w * Xq_w).sum(dim=1, keepdim=True)  # (Nq, 1)
    ar = (Xr_w * Xr_w).sum(dim=1, keepdim=True).T  # (1, Nr)
    D = aq + ar - 2.0 * (Xq_w @ Xr_w.T)
    return torch.clamp(D, min=0.0)


def _safe_cholesky(
    Sigma: "torch.Tensor",
    I: "torch.Tensor",
    ridge: float,
    tries: int,
    mult: float,
) -> "torch.Tensor":
    """Cholesky with increasing jitter."""
    jitter = float(ridge)
    last_err: Optional[Exception] = None
    for _ in range(max(1, int(tries))):
        try:
            return torch.linalg.cholesky(Sigma + jitter * I)
        except Exception as e:  # pragma: no cover (depends on data)
            last_err = e
            jitter *= float(mult)
    # final attempt (let it raise if still failing)
    if last_err is not None:
        logger.debug("Cholesky still failing after jitter; last_err=%r", last_err)
    return torch.linalg.cholesky(Sigma + jitter * I)


# -----------------------------
# Omega fitting
# -----------------------------


def fit_omega(
    X: np.ndarray,
    Y: np.ndarray,
    omega0: np.ndarray,
    train_idx: np.ndarray,
    cfg: KernelCovConfig,
    fit_cfg: FitConfig,
    return_history: bool = False,
) -> np.ndarray:
    """Gradient descent to optimize feature weights (omega) for covariance estimation.

    IMPORTANT: This performs leave-one-out moment estimation on the *training* set indices.

    The omega_constraint option in fit_cfg controls how omega is parameterized:
    - "none": Learn omega directly (current behavior, no sum constraint)
    - "softmax": Learn alpha, omega = softmax(alpha). Ensures sum(omega)=1, omega>=0.
    - "simplex": Project omega onto probability simplex after each step.
    - "normalize": Divide omega by sum(omega) at the end (post-hoc).

    When omega_constraint is "softmax" or "simplex", omega represents relative feature
    importance (sums to 1), separating feature weighting from kernel bandwidth (tau).
    """
    _check_torch()

    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    omega0 = _as_1d(omega0, "omega0")
    train_idx = np.asarray(train_idx, dtype=int)

    dtype = _torch_dtype(fit_cfg.dtype)
    device = fit_cfg.device
    constraint = fit_cfg.omega_constraint

    if constraint not in ("none", "softmax", "simplex", "normalize"):
        raise ValueError(
            f"omega_constraint must be 'none', 'softmax', 'simplex', or 'normalize', got {constraint!r}"
        )

    Xtr = torch.as_tensor(X[train_idx], dtype=dtype, device=device)
    Ytr = torch.as_tensor(Y[train_idx], dtype=dtype, device=device)

    Nt, M = Xtr.shape[0], Ytr.shape[1]
    I = torch.eye(M, dtype=dtype, device=device)

    # Initialize parameters based on constraint type
    if constraint == "softmax":
        # Learn in log-space: alpha, then omega = softmax(alpha)
        # Initialize alpha so that softmax(alpha) ≈ omega0 / sum(omega0)
        omega0_normalized = omega0 / omega0.sum()
        # alpha = log(omega0_normalized) + constant (constant cancels in softmax)
        alpha_init = np.log(omega0_normalized + 1e-10)
        param_t = torch.tensor(
            alpha_init.astype(np.float64),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
    else:
        # Learn omega directly
        param_t = torch.tensor(
            np.asarray(omega0, dtype=np.float64),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

    history = []
    prev_loss = None
    opt = torch.optim.Adam([param_t], lr=float(fit_cfg.step_size))

    for it in range(1, fit_cfg.max_iters + 1):
        opt.zero_grad()

        # Convert parameter to omega based on constraint type
        if constraint == "softmax":
            # omega = softmax(alpha), ensures sum(omega)=1 and omega>=0
            om = torch.softmax(param_t, dim=0)
        elif constraint == "simplex":
            # Will project after gradient step; for now use clamped param
            om = torch.clamp(param_t, min=0.0) if cfg.enforce_nonneg_omega else param_t
        else:
            # "none" or "normalize": use clamped param directly
            om = torch.clamp(param_t, min=0.0) if cfg.enforce_nonneg_omega else param_t

        # leave-one-out distances and masked softmax on training set
        D = _pairwise_sqdist(Xtr, Xtr, om)  # (Nt, Nt)
        logits = -D / float(cfg.tau)

        mask = ~torch.eye(Nt, dtype=torch.bool, device=device)
        logits = logits.masked_fill(~mask, -1e9)  # exclude self

        Phi = torch.softmax(logits, dim=1) * mask.to(dtype)

        Mu = Phi @ Ytr  # (Nt, M)

        C = Ytr.unsqueeze(0) - Mu.unsqueeze(1)  # (Nt, Nt, M)
        CW = C * Phi.unsqueeze(-1)  # (Nt, Nt, M)

        Sigma = torch.matmul(C.transpose(1, 2), CW)  # (Nt, M, M)
        Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2)) + float(cfg.ridge) * I

        # NLL under Gaussian with per-row Sigma
        r = (Ytr - Mu).unsqueeze(-1)  # (Nt, M, 1)
        Lchol = _safe_cholesky(
            Sigma,
            I,
            ridge=float(cfg.ridge),
            tries=fit_cfg.cholesky_jitter_tries,
            mult=fit_cfg.cholesky_jitter_mult,
        )
        logdet = 2.0 * torch.log(torch.diagonal(Lchol, dim1=-2, dim2=-1)).sum(
            dim=-1
        )  # (Nt,)
        x = torch.cholesky_solve(r, Lchol)  # (Nt, M, 1)
        quad = (r.transpose(1, 2) @ x).squeeze(-1).squeeze(-1)  # (Nt,)
        nll_loss = (logdet + quad).mean()

        # L2 regularization on omega (only when no constraint)
        # With constraints (softmax/simplex/normalize), the constraint itself acts as regularization
        if constraint == "none" and fit_cfg.omega_l2_reg > 0:
            omega_reg = fit_cfg.omega_l2_reg * ((om - 1.0) ** 2).sum()
        else:
            omega_reg = torch.tensor(0.0, dtype=dtype, device=device)
        loss = nll_loss + omega_reg

        if fit_cfg.finite_check and (not torch.isfinite(loss)):
            logger.error("Non-finite loss at iter=%s; stopping.", it)
            break

        loss.backward()

        if fit_cfg.grad_clip is not None and float(fit_cfg.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_([param_t], float(fit_cfg.grad_clip))

        opt.step()

        # Post-step projection/clamping based on constraint type
        with torch.no_grad():
            if constraint == "softmax":
                # No clamping needed - softmax handles constraints
                pass
            elif constraint == "simplex":
                # Project onto probability simplex
                param_t.copy_(_project_simplex_torch(param_t))
            else:
                # "none" or "normalize": just clamp to positive
                eps_w = 1e-2
                param_t.clamp_(min=eps_w)

        gn = float(torch.linalg.norm(param_t.grad.detach()).item())
        if return_history:
            with torch.no_grad():
                # Compute actual omega for logging
                if constraint == "softmax":
                    om_now = torch.softmax(param_t, dim=0)
                elif constraint == "simplex":
                    om_now = param_t.clone()
                else:
                    om_now = (
                        torch.clamp(param_t, min=0.0)
                        if cfg.enforce_nonneg_omega
                        else param_t
                    )
                history.append(
                    {
                        "iter": it,
                        "loss": float(loss.item()),
                        "nll_loss": float(nll_loss.item()),
                        "omega_reg": float(omega_reg.item()),
                        "grad_norm": gn,
                        "omega_min": float(om_now.min().item()),
                        "omega_max": float(om_now.max().item()),
                        "omega_mean": float(om_now.mean().item()),
                        "omega_sum": float(om_now.sum().item()),
                        "omega_nnz": int((om_now > 0).sum().item()),
                    }
                )
        if it % fit_cfg.verbose_every == 0:
            logger.info(
                "Iter %s: loss=%.6f, grad_norm=%.3e", it, float(loss.item()), gn
            )
            with torch.no_grad():
                # Compute actual omega for logging
                if constraint == "softmax":
                    om_now = torch.softmax(param_t, dim=0)
                elif constraint == "simplex":
                    om_now = param_t.clone()
                else:
                    om_now = (
                        torch.clamp(param_t, min=0.0)
                        if cfg.enforce_nonneg_omega
                        else param_t
                    )
                logger.info(
                    "omega: min=%.3e, max=%.3e, mean=%.3e, sum=%.3f, nnz=%d/%d",
                    float(om_now.min().item()),
                    float(om_now.max().item()),
                    float(om_now.mean().item()),
                    float(om_now.sum().item()),
                    int((om_now > 0).sum().item()),
                    int(om_now.numel()),
                )

        # Convergence check
        if prev_loss is not None and abs(prev_loss - float(loss.item())) < float(
            fit_cfg.tol
        ):
            if it % fit_cfg.verbose_every != 0:
                logger.info("Converged at iter %s: loss=%.6f", it, float(loss.item()))
            break
        prev_loss = float(loss.item())

    # Convert parameter to final omega based on constraint type
    with torch.no_grad():
        if constraint == "softmax":
            omega_out = torch.softmax(param_t, dim=0).cpu().numpy()
        elif constraint == "simplex":
            omega_out = param_t.cpu().numpy()
        elif constraint == "normalize":
            # Post-hoc normalization: divide by sum
            omega_clamped = torch.clamp(param_t, min=0.0)
            omega_out = (omega_clamped / omega_clamped.sum()).cpu().numpy()
        else:
            # "none": return as-is
            omega_out = param_t.cpu().numpy()

    if return_history:
        return omega_out, history
    return omega_out


# -----------------------------
# Prediction (no leakage by construction)
# -----------------------------


@overload
def predict_mu_sigma_topk_cross(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    omega: np.ndarray,
    cfg: CovPredictConfig,
    k: int = 128,
    *,
    exclude_self_if_same: bool = True,
    return_type: Literal["numpy"] = "numpy",
) -> Tuple[np.ndarray, np.ndarray]:
    ...


@overload
def predict_mu_sigma_topk_cross(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    omega: np.ndarray,
    cfg: CovPredictConfig,
    k: int = 128,
    *,
    exclude_self_if_same: bool = True,
    return_type: Literal["torch"],
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    ...


def predict_mu_sigma_topk_cross(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    omega: np.ndarray,
    cfg: CovPredictConfig,
    k: int = 128,
    *,
    exclude_self_if_same: bool = True,
    return_type: Literal["numpy", "torch"] = "numpy",
):
    """Compute (mu, Sigma) at query points using reference set moments.

    This is the *safe* API for evaluation time:
      - Use X_query for the timestamps you want moments for.
      - Use (X_ref, Y_ref) from *training* only.

    If X_query and X_ref are the same object/array and exclude_self_if_same=True,
    we exclude self from the neighbor set by setting diagonal to +inf.
    """
    _check_torch()

    X_query = _as_2d(X_query, "X_query")
    X_ref = _as_2d(X_ref, "X_ref")
    Y_ref = _as_2d(Y_ref, "Y_ref")
    omega = _as_1d(omega, "omega")

    if X_ref.shape[0] != Y_ref.shape[0]:
        raise ValueError(
            f"X_ref rows ({X_ref.shape[0]}) must match Y_ref rows ({Y_ref.shape[0]})."
        )

    dtype = _torch_dtype(cfg.dtype)
    device = cfg.device

    Xq = torch.as_tensor(X_query, dtype=dtype, device=device)
    Xr = torch.as_tensor(X_ref, dtype=dtype, device=device)
    Yr = torch.as_tensor(Y_ref, dtype=dtype, device=device)

    om = torch.as_tensor(omega, dtype=dtype, device=device)
    if cfg.enforce_nonneg_omega:
        om = torch.clamp(om, min=0.0)

    # Distances (Nq, Nr)
    D = _pairwise_sqdist(Xq, Xr, om)

    if (
        exclude_self_if_same
        and (X_query is X_ref or np.shares_memory(X_query, X_ref))
        and D.shape[0] == D.shape[1]
    ):
        # in-sample leave-one-out
        D.fill_diagonal_(float("inf"))

    Nr = D.shape[1]
    k_eff = int(
        min(
            max(1, k),
            max(
                1, Nr - 1 if (exclude_self_if_same and D.shape[0] == D.shape[1]) else Nr
            ),
        )
    )
    if k_eff != k:
        logger.debug("Adjusted k from %s to %s based on reference set size.", k, k_eff)

    d_k, idx = torch.topk(D, k=k_eff, dim=1, largest=False, sorted=False)  # (Nq, k)
    logits = -d_k / float(cfg.tau)
    Phi_k = torch.softmax(logits, dim=1)  # (Nq, k)

    Y_nb = Yr[idx]  # (Nq, k, M)

    Mu = (Phi_k.unsqueeze(-1) * Y_nb).sum(dim=1)  # (Nq, M)
    C = Y_nb - Mu.unsqueeze(1)  # (Nq, k, M)
    CW = C * Phi_k.unsqueeze(-1)  # (Nq, k, M)
    Sigma = torch.matmul(C.transpose(1, 2), CW)  # (Nq, M, M)

    I = torch.eye(Y_ref.shape[1], dtype=dtype, device=device)
    Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2)) + float(cfg.ridge) * I

    if cfg.finite_check:
        if (not torch.isfinite(Mu).all()) or (not torch.isfinite(Sigma).all()):
            raise FloatingPointError(
                "Non-finite values encountered in predicted moments."
            )

    if return_type == "torch":
        return Mu, Sigma
    return Mu.detach().cpu().numpy(), Sigma.detach().cpu().numpy()


def predict_mu_sigma_topk(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    omega_np: np.ndarray,
    cfg: CovPredictConfig,
    k: int = 128,
    *,
    return_type: Literal["numpy", "torch"] = "numpy",
):
    """Backward-compatible wrapper for in-sample prediction.

    WARNING: This uses Y_np as the reference set; for evaluation you almost always want
    `predict_mu_sigma_topk_cross(X_eval, X_train, Y_train, ...)` to avoid leakage.
    """
    return predict_mu_sigma_topk_cross(
        X_query=X_np,
        X_ref=X_np,
        Y_ref=Y_np,
        omega=omega_np,
        cfg=cfg,
        k=k,
        exclude_self_if_same=True,
        return_type=return_type,
    )


# -----------------------------
# Scalar rho helper
# -----------------------------


# -----------------------------
# KNN Baseline (no learned weights)
# -----------------------------


def predict_mu_sigma_knn(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    k: int = 64,
    ridge: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple KNN: uniform weights over k nearest neighbors.

    This is a baseline that uses standard Euclidean distance (no learned omega)
    with uniform weights over the k nearest neighbors.

    Parameters
    ----------
    X_query : np.ndarray
        Query features, shape (Nq, K).
    X_ref : np.ndarray
        Reference features (standardized), shape (Nr, K).
    Y_ref : np.ndarray
        Reference targets, shape (Nr, M).
    k : int
        Number of neighbors to use.
    ridge : float
        Ridge regularization for covariance matrix.

    Returns
    -------
    Mu : np.ndarray
        Predicted means, shape (Nq, M).
    Sigma : np.ndarray
        Predicted covariance matrices, shape (Nq, M, M).
    """
    X_query = _as_2d(X_query, "X_query")
    X_ref = _as_2d(X_ref, "X_ref")
    Y_ref = _as_2d(Y_ref, "Y_ref")

    if X_ref.shape[0] != Y_ref.shape[0]:
        raise ValueError(
            f"X_ref rows ({X_ref.shape[0]}) must match Y_ref rows ({Y_ref.shape[0]})."
        )
    if X_query.shape[1] != X_ref.shape[1]:
        raise ValueError(
            f"X_query features ({X_query.shape[1]}) must match X_ref features ({X_ref.shape[1]})."
        )

    Nq = X_query.shape[0]
    Nr = X_ref.shape[0]
    M = Y_ref.shape[1]
    k_eff = min(k, Nr)

    # Euclidean distance: ||x_q - x_r||^2 for all pairs
    # Using broadcasting: (Nq, 1, K) - (1, Nr, K) -> (Nq, Nr, K) -> sum -> (Nq, Nr)
    diff = X_query[:, np.newaxis, :] - X_ref[np.newaxis, :, :]  # (Nq, Nr, K)
    D = np.sum(diff**2, axis=2)  # (Nq, Nr)

    # Find k nearest neighbors for each query
    idx = np.argpartition(D, k_eff - 1, axis=1)[:, :k_eff]  # (Nq, k_eff)

    # Compute uniform-weighted mean and covariance for each query
    Mu = np.zeros((Nq, M))
    Sigma = np.zeros((Nq, M, M))
    I = np.eye(M)

    for i in range(Nq):
        Y_neighbors = Y_ref[idx[i]]  # (k_eff, M)
        mu_i = Y_neighbors.mean(axis=0)
        Mu[i] = mu_i

        # Sample covariance with uniform weights
        C = Y_neighbors - mu_i  # (k_eff, M)
        Sigma_i = (C.T @ C) / k_eff  # (M, M)
        # Symmetrize and add ridge
        Sigma[i] = 0.5 * (Sigma_i + Sigma_i.T) + ridge * I

    return Mu, Sigma


def implied_rho_from_total_lower_bound(
    Sigma: np.ndarray,
    mean: np.ndarray,
    total_lower_bound: float,
    e: np.ndarray | None = None,
    clip_nonneg: bool = True,
    eps: float = 1e-12,
) -> float:
    """Convert a scalar lower bound on e^T y into ellipsoid radius rho.

    We assume constraint:  e^T y >= total_lower_bound  for y in ellipsoid centered at mean with covariance Sigma.

    rho = (e^T mean - lower_bound) / sqrt(e^T Sigma e)
    """
    Sigma = np.asarray(Sigma)
    mean = np.asarray(mean)

    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be square 2D, got shape {Sigma.shape}")
    if mean.ndim != 1 or mean.shape[0] != Sigma.shape[0]:
        raise ValueError(
            f"mean must be shape (M,), got {mean.shape} for M={Sigma.shape[0]}"
        )
    M = Sigma.shape[0]
    e = np.ones(M) if e is None else np.asarray(e)
    if e.shape != (M,):
        raise ValueError(f"e must be shape (M,), got {e.shape}")

    mean_total = float(e @ mean)
    var_total = float(e @ (Sigma @ e))
    denom = float(np.sqrt(max(var_total, eps)))

    rho = (mean_total - float(total_lower_bound)) / denom
    return max(rho, 0.0) if clip_nonneg else rho
