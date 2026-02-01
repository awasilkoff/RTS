from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ellipsoid_mesh(
    mu3: np.ndarray, Sigma3: np.ndarray, rho: float, n_u: int = 60, n_v: int = 30
):
    """
    Parametric ellipsoid surface:
      x = mu + rho * A * s(u,v)
    where A satisfies A A^T = Sigma3 and s is unit sphere point.
    """
    # Eigen-decomp for symmetric PSD
    w, V = np.linalg.eigh(Sigma3)
    w = np.clip(w, 0.0, None)
    A = V @ np.diag(np.sqrt(w))  # so that A A^T = Sigma3

    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)

    # unit sphere
    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)
    S = np.stack([xs, ys, zs], axis=-1)  # (n_v, n_u, 3)

    X = mu3[None, None, :] + rho * (S @ A.T)
    return X[..., 0], X[..., 1], X[..., 2]


def tangent_point(mu: np.ndarray, Sigma: np.ndarray, rho: float) -> np.ndarray:
    e = np.ones(mu.shape[0], dtype=float)
    denom = float(np.sqrt(max(e @ Sigma @ e, 0.0)))
    if denom <= 0:
        return mu.copy()
    return mu - rho * (Sigma @ e) / denom


def plot_3d_ellipsoid_and_plane(
    *,
    npz_path: Path,
    parquet_path: Path,
    out_png: Path,
    idx: int = 0,
    dims3: tuple[int, int, int] = (0, 1, 2),
    fixed_dim: int = 3,
    fixed_mode: str = "tangent",  # "tangent" or "mu"
    plane_grid: float = 3.0,  # scale plane extent relative to ellipsoid std
):
    # Load tensors
    z = np.load(npz_path, allow_pickle=True)
    times = z["times"]
    mu = z["mu"]  # (T, M)
    sigma = z["sigma"]  # (T, M, M)
    rho_arr = z["rho"]  # (T,)

    df = pd.read_parquet(parquet_path)

    if idx < 0 or idx >= mu.shape[0]:
        raise ValueError(f"idx out of range: idx={idx}, available=0..{mu.shape[0]-1}")

    mu_t = mu[idx].astype(float)
    Sigma_t = sigma[idx].astype(float)
    rho_t = float(rho_arr[idx])

    # Get LB from parquet (aligned to same kept times)
    # safest: take matching row idx if parquet rows correspond to npz
    lb = float(df.iloc[idx]["total_lower_bound"])

    # Tangency point in full 4D
    x_star = tangent_point(mu_t, Sigma_t, rho_t)

    # Choose which 3 dims we plot
    d0, d1, d2 = dims3
    mu3 = mu_t[[d0, d1, d2]]
    Sigma3 = Sigma_t[np.ix_([d0, d1, d2], [d0, d1, d2])]

    # Decide fixed value of dropped dimension for plane equation
    if fixed_mode == "tangent":
        x_fixed = float(x_star[fixed_dim])
    elif fixed_mode == "mu":
        x_fixed = float(mu_t[fixed_dim])
    else:
        raise ValueError("fixed_mode must be 'tangent' or 'mu'")

    # Plane: x_d0 + x_d1 + x_d2 = lb - x_fixed  (assuming original was sum over all 4 dims)
    rhs = lb - x_fixed

    # Build ellipsoid surface in 3D
    X, Y, Z = ellipsoid_mesh(mu3, Sigma3, rho_t)

    # Tangent point projection
    x_star3 = x_star[[d0, d1, d2]]

    # Plane grid extents based on ellipsoid spread
    std3 = np.sqrt(np.clip(np.diag(Sigma3), 0.0, None))
    extent = plane_grid * float(np.max(std3) * max(rho_t, 1e-6))
    if not np.isfinite(extent) or extent <= 0:
        extent = 1.0

    # Choose a grid in x-y, solve for z from x+y+z=rhs (in plotted coords)
    # Here we interpret plotted axes as (d0,d1,d2), so plane is X+Y+Z=rhs
    gx = np.linspace(mu3[0] - extent, mu3[0] + extent, 60)
    gy = np.linspace(mu3[1] - extent, mu3[1] + extent, 60)
    GX, GY = np.meshgrid(gx, gy)
    GZ = rhs - GX - GY

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, alpha=0.25, linewidth=0)
    ax.plot_surface(GX, GY, GZ, alpha=0.25, linewidth=0)

    ax.scatter([mu3[0]], [mu3[1]], [mu3[2]], s=60)
    ax.scatter([x_star3[0]], [x_star3[1]], [x_star3[2]], s=80)

    title_t = str(times[idx])
    ax.set_title(
        f"3D Ellipsoid (dims {dims3}) + LB Plane (fixed dim {fixed_dim}={fixed_mode})\n"
        f"time={title_t}, rho={rho_t:.3f}"
    )
    ax.set_xlabel(f"x[{d0}]")
    ax.set_ylabel(f"x[{d1}]")
    ax.set_zlabel(f"x[{d2}]")
    # --- Axis limits: force lower bounds to 0 ---
    # Collect finite values from what we plotted (ellipsoid + plane grid + points)
    xs = np.concatenate([X.ravel(), GX.ravel(), np.array([mu3[0], x_star3[0]])])
    ys = np.concatenate([Y.ravel(), GY.ravel(), np.array([mu3[1], x_star3[1]])])
    zs = np.concatenate([Z.ravel(), GZ.ravel(), np.array([mu3[2], x_star3[2]])])

    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    zs = zs[np.isfinite(zs)]

    # Upper bounds: max(0, max_value) with a small pad
    pad = 0.05
    x_max = float(max(0.0, xs.max())) if xs.size else 1.0
    y_max = float(max(0.0, ys.max())) if ys.size else 1.0
    z_max = float(max(0.0, zs.max())) if zs.size else 1.0

    # add a small padding
    ax.set_xlim(0.0, x_max * (1.0 + pad) if x_max > 0 else 1.0)
    ax.set_ylim(0.0, y_max * (1.0 + pad) if y_max > 0 else 1.0)
    ax.set_zlim(0.0, z_max * (1.0 + pad) if z_max > 0 else 1.0)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_multi_ellipsoid_comparison(
    *,
    mu_dict: dict[str, np.ndarray],
    sigma_dict: dict[str, np.ndarray],
    rho: float,
    y_actual: np.ndarray | None = None,
    out_png: Path,
    dims3: tuple[int, int, int] = (0, 1, 2),
    colors: dict[str, str] | None = None,
    title: str | None = None,
):
    """Overlay multiple covariance ellipsoids for method comparison.

    Parameters
    ----------
    mu_dict : dict[str, np.ndarray]
        Method name -> mean vector (M,).
    sigma_dict : dict[str, np.ndarray]
        Method name -> covariance matrix (M, M).
    rho : float
        Ellipsoid radius (same for all methods for fair comparison).
    y_actual : np.ndarray, optional
        Actual observation point to mark on plot (M,).
    out_png : Path
        Output path for PNG file.
    dims3 : tuple[int, int, int]
        Which 3 dimensions to plot if M > 3.
    colors : dict[str, str], optional
        Method name -> color. Defaults provided if None.
    title : str, optional
        Plot title.
    """
    default_colors = {
        "Equal Weights": "blue",
        "Learned Omega": "green",
        "KNN": "orange",
    }
    if colors is None:
        colors = default_colors

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    d0, d1, d2 = dims3
    all_surfaces = []

    for method_name in mu_dict.keys():
        mu = mu_dict[method_name]
        Sigma = sigma_dict[method_name]
        color = colors.get(method_name, "gray")

        # Extract 3D subset
        mu3 = mu[[d0, d1, d2]]
        Sigma3 = Sigma[np.ix_([d0, d1, d2], [d0, d1, d2])]

        # Build ellipsoid mesh
        X, Y, Z = ellipsoid_mesh(mu3, Sigma3, rho)
        all_surfaces.append((X, Y, Z))

        # Plot as wireframe for better visibility
        ax.plot_wireframe(
            X, Y, Z, alpha=0.4, linewidth=0.5, color=color, label=method_name
        )

        # Mark center
        ax.scatter([mu3[0]], [mu3[1]], [mu3[2]], s=60, c=color, marker="o")

    # Mark actual observation if provided
    if y_actual is not None:
        y3 = y_actual[[d0, d1, d2]]
        ax.scatter(
            [y3[0]], [y3[1]], [y3[2]], s=100, c="red", marker="*", label="Actual Y"
        )

    ax.legend(loc="upper right")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Covariance Ellipsoid Comparison (dims {dims3})")

    ax.set_xlabel(f"x[{d0}]")
    ax.set_ylabel(f"x[{d1}]")
    ax.set_zlabel(f"x[{d2}]")

    # Set axis limits based on all surfaces
    xs = np.concatenate([s[0].ravel() for s in all_surfaces])
    ys = np.concatenate([s[1].ravel() for s in all_surfaces])
    zs = np.concatenate([s[2].ravel() for s in all_surfaces])

    if y_actual is not None:
        y3 = y_actual[[d0, d1, d2]]
        xs = np.append(xs, y3[0])
        ys = np.append(ys, y3[1])
        zs = np.append(zs, y3[2])

    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    zs = zs[np.isfinite(zs)]

    pad = 0.1
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    z_range = z_max - z_min if z_max > z_min else 1.0

    ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
    ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)
    ax.set_zlim(z_min - pad * z_range, z_max + pad * z_range)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    DATA_DIR = Path("/Users/alexwasilkoff/PycharmProjects/RTS/uncertainty_sets/data/")
    ART = DATA_DIR / "viz_artifacts"

    plot_3d_ellipsoid_and_plane(
        npz_path=ART / "full_tensors.npz",
        parquet_path=ART / "viz_integration_results.parquet",
        out_png=ART / "ellipsoid_plane_3d.png",
        idx=0,  # choose which timestamp row to visualize
        dims3=(0, 1, 2),
        fixed_dim=3,
        fixed_mode="tangent",  # most consistent with tangency
    )
    print(f"Wrote {ART / 'ellipsoid_plane_3d.png'}")
