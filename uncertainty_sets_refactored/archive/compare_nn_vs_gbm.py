#!/usr/bin/env python3
"""
Empirical comparison: Neural Networks vs Gradient Boosting for conformal prediction.

Tests whether neural networks can outperform LightGBM on this dataset (~1,770 training samples).

Models tested:
1. LightGBM (current baseline)
2. Simple MLP (Multi-Layer Perceptron)
3. Deeper MLP with dropout
4. Quantile regression NN (custom pinball loss)

Goal: Determine if NNs provide better coverage calibration on this small-medium dataset.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_processing import build_conformal_totals_df
from conformal_prediction import (
    train_wind_lower_model_conformal_binned,
    compute_binned_adaptive_conformal_corrections_lower,
    _time_ordered_split,
)


# ============================================================================
# Neural Network Models
# ============================================================================


class QuantileMLP(nn.Module):
    """
    Simple MLP for quantile regression with pinball loss.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def pinball_loss(predictions, targets, quantile: float):
    """
    Pinball loss (quantile loss) for quantile regression.

    L(y, ŷ) = max(q(y - ŷ), (q-1)(y - ŷ))
    """
    errors = targets - predictions
    loss = torch.mean(
        torch.max(quantile * errors, (quantile - 1) * errors)
    )
    return loss


def train_quantile_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantile: float = 0.10,
    hidden_dims: list[int] = [128, 64, 32],
    dropout: float = 0.2,
    lr: float = 0.001,
    batch_size: int = 64,
    epochs: int = 200,
    early_stop_patience: int = 20,
    verbose: bool = True,
) -> tuple[QuantileMLP, StandardScaler, list]:
    """
    Train a quantile regression neural network.

    Returns
    -------
    model : QuantileMLP
        Trained model
    scaler : StandardScaler
        Feature scaler
    history : list
        Training history (loss per epoch)
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = X_train.shape[1]
    model = QuantileMLP(input_dim, hidden_dims, dropout)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = pinball_loss(predictions, batch_y, quantile)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = pinball_loss(val_predictions, y_val_t, quantile).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    return model, scaler, history


def predict_quantile_nn(
    model: QuantileMLP, scaler: StandardScaler, X: np.ndarray
) -> np.ndarray:
    """Predict with trained quantile NN."""
    model.eval()
    X_scaled = scaler.transform(X)
    X_t = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        predictions = model(X_t).numpy()
    return predictions


# ============================================================================
# Comparison Functions
# ============================================================================


def compare_models_on_conformal(
    df: pd.DataFrame,
    feature_cols: list[str],
    alpha_target: float = 0.95,
    quantile_alpha: float = 0.10,
):
    """
    Compare LightGBM vs Neural Network for conformal prediction.

    Returns
    -------
    results : dict
        Comparison metrics
    """
    print("\n" + "=" * 80)
    print("COMPARING NEURAL NETWORK vs LIGHTGBM")
    print("=" * 80)

    # Prepare data
    df_clean = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    X = df_clean[feature_cols].values
    y = df_clean["y"].values
    scale = df_clean["ens_std"].values

    print(f"\nDataset: {len(df_clean)} samples, {len(feature_cols)} features")

    # Time-ordered split
    sl_train, sl_cal, sl_test = _time_ordered_split(
        len(df_clean), test_frac=0.2, cal_frac=0.2
    )

    X_train, y_train = X[sl_train], y[sl_train]
    X_cal, y_cal = X[sl_cal], y[sl_cal]
    X_test, y_test = X[sl_test], y[sl_test]
    scale_cal = scale[sl_cal]
    scale_test = scale[sl_test]

    print(f"Split: train={len(y_train)}, cal={len(y_cal)}, test={len(y_test)}")

    results = {}

    # ========================================================================
    # Model 1: LightGBM (baseline)
    # ========================================================================
    print("\n[1/4] Training LightGBM (baseline)...")

    from lightgbm import LGBMRegressor

    lgbm = LGBMRegressor(
        objective="quantile",
        alpha=quantile_alpha,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    lgbm.fit(X_train, y_train)
    y_pred_lgbm_cal = lgbm.predict(X_cal)
    y_pred_lgbm_test = lgbm.predict(X_test)

    pre_coverage_lgbm = (y_test >= y_pred_lgbm_test).mean()
    print(f"  Pre-conformal coverage: {pre_coverage_lgbm:.3f}")

    # Apply conformal correction
    q_hat_global_lgbm, q_hat_by_bin_lgbm = compute_binned_adaptive_conformal_corrections_lower(
        bin_feature_cal=y_pred_lgbm_cal,
        y_cal=y_cal,
        y_pred_cal=y_pred_lgbm_cal,
        scale_cal=scale_cal,
        alpha_target=alpha_target,
        bins=[0, 200, 400, 600, 800, 1000, 1200, 1400],
    )

    margin_lgbm = q_hat_global_lgbm * scale_test
    y_pred_conf_lgbm = y_pred_lgbm_test - margin_lgbm
    coverage_lgbm = (y_test >= y_pred_conf_lgbm).mean()

    print(f"  Post-conformal coverage: {coverage_lgbm:.3f} (target: {alpha_target})")
    print(f"  Coverage gap: {abs(coverage_lgbm - alpha_target):.3f}")

    results["lightgbm"] = {
        "pre_coverage": pre_coverage_lgbm,
        "coverage": coverage_lgbm,
        "coverage_gap": abs(coverage_lgbm - alpha_target),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_conf_lgbm)),
        "mae": mean_absolute_error(y_test, y_pred_conf_lgbm),
    }

    # ========================================================================
    # Model 2: Simple MLP
    # ========================================================================
    print("\n[2/4] Training Simple MLP...")

    model_mlp_simple, scaler_simple, _ = train_quantile_nn(
        X_train,
        y_train,
        X_cal,
        y_cal,
        quantile=quantile_alpha,
        hidden_dims=[64, 32],
        dropout=0.1,
        lr=0.001,
        epochs=200,
        verbose=False,
    )

    y_pred_mlp_simple_cal = predict_quantile_nn(model_mlp_simple, scaler_simple, X_cal)
    y_pred_mlp_simple_test = predict_quantile_nn(model_mlp_simple, scaler_simple, X_test)

    pre_coverage_mlp_simple = (y_test >= y_pred_mlp_simple_test).mean()
    print(f"  Pre-conformal coverage: {pre_coverage_mlp_simple:.3f}")

    # Apply conformal correction
    q_hat_global_mlp_simple, _ = compute_binned_adaptive_conformal_corrections_lower(
        bin_feature_cal=y_pred_mlp_simple_cal,
        y_cal=y_cal,
        y_pred_cal=y_pred_mlp_simple_cal,
        scale_cal=scale_cal,
        alpha_target=alpha_target,
        bins=[0, 200, 400, 600, 800, 1000, 1200, 1400],
    )

    margin_mlp_simple = q_hat_global_mlp_simple * scale_test
    y_pred_conf_mlp_simple = y_pred_mlp_simple_test - margin_mlp_simple
    coverage_mlp_simple = (y_test >= y_pred_conf_mlp_simple).mean()

    print(f"  Post-conformal coverage: {coverage_mlp_simple:.3f} (target: {alpha_target})")
    print(f"  Coverage gap: {abs(coverage_mlp_simple - alpha_target):.3f}")

    results["mlp_simple"] = {
        "pre_coverage": pre_coverage_mlp_simple,
        "coverage": coverage_mlp_simple,
        "coverage_gap": abs(coverage_mlp_simple - alpha_target),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_conf_mlp_simple)),
        "mae": mean_absolute_error(y_test, y_pred_conf_mlp_simple),
    }

    # ========================================================================
    # Model 3: Deeper MLP with regularization
    # ========================================================================
    print("\n[3/4] Training Deeper MLP (regularized)...")

    model_mlp_deep, scaler_deep, _ = train_quantile_nn(
        X_train,
        y_train,
        X_cal,
        y_cal,
        quantile=quantile_alpha,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        lr=0.0005,
        epochs=300,
        verbose=False,
    )

    y_pred_mlp_deep_cal = predict_quantile_nn(model_mlp_deep, scaler_deep, X_cal)
    y_pred_mlp_deep_test = predict_quantile_nn(model_mlp_deep, scaler_deep, X_test)

    pre_coverage_mlp_deep = (y_test >= y_pred_mlp_deep_test).mean()
    print(f"  Pre-conformal coverage: {pre_coverage_mlp_deep:.3f}")

    # Apply conformal correction
    q_hat_global_mlp_deep, _ = compute_binned_adaptive_conformal_corrections_lower(
        bin_feature_cal=y_pred_mlp_deep_cal,
        y_cal=y_cal,
        y_pred_cal=y_pred_mlp_deep_cal,
        scale_cal=scale_cal,
        alpha_target=alpha_target,
        bins=[0, 200, 400, 600, 800, 1000, 1200, 1400],
    )

    margin_mlp_deep = q_hat_global_mlp_deep * scale_test
    y_pred_conf_mlp_deep = y_pred_mlp_deep_test - margin_mlp_deep
    coverage_mlp_deep = (y_test >= y_pred_conf_mlp_deep).mean()

    print(f"  Post-conformal coverage: {coverage_mlp_deep:.3f} (target: {alpha_target})")
    print(f"  Coverage gap: {abs(coverage_mlp_deep - alpha_target):.3f}")

    results["mlp_deep"] = {
        "pre_coverage": pre_coverage_mlp_deep,
        "coverage": coverage_mlp_deep,
        "coverage_gap": abs(coverage_mlp_deep - alpha_target),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_conf_mlp_deep)),
        "mae": mean_absolute_error(y_test, y_pred_conf_mlp_deep),
    }

    # ========================================================================
    # Model 4: Wide MLP (more capacity)
    # ========================================================================
    print("\n[4/4] Training Wide MLP...")

    model_mlp_wide, scaler_wide, _ = train_quantile_nn(
        X_train,
        y_train,
        X_cal,
        y_cal,
        quantile=quantile_alpha,
        hidden_dims=[256, 128],
        dropout=0.2,
        lr=0.001,
        epochs=200,
        verbose=False,
    )

    y_pred_mlp_wide_cal = predict_quantile_nn(model_mlp_wide, scaler_wide, X_cal)
    y_pred_mlp_wide_test = predict_quantile_nn(model_mlp_wide, scaler_wide, X_test)

    pre_coverage_mlp_wide = (y_test >= y_pred_mlp_wide_test).mean()
    print(f"  Pre-conformal coverage: {pre_coverage_mlp_wide:.3f}")

    # Apply conformal correction
    q_hat_global_mlp_wide, _ = compute_binned_adaptive_conformal_corrections_lower(
        bin_feature_cal=y_pred_mlp_wide_cal,
        y_cal=y_cal,
        y_pred_cal=y_pred_mlp_wide_cal,
        scale_cal=scale_cal,
        alpha_target=alpha_target,
        bins=[0, 200, 400, 600, 800, 1000, 1200, 1400],
    )

    margin_mlp_wide = q_hat_global_mlp_wide * scale_test
    y_pred_conf_mlp_wide = y_pred_mlp_wide_test - margin_mlp_wide
    coverage_mlp_wide = (y_test >= y_pred_conf_mlp_wide).mean()

    print(f"  Post-conformal coverage: {coverage_mlp_wide:.3f} (target: {alpha_target})")
    print(f"  Coverage gap: {abs(coverage_mlp_wide - alpha_target):.3f}")

    results["mlp_wide"] = {
        "pre_coverage": pre_coverage_mlp_wide,
        "coverage": coverage_mlp_wide,
        "coverage_gap": abs(coverage_mlp_wide - alpha_target),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_conf_mlp_wide)),
        "mae": mean_absolute_error(y_test, y_pred_conf_mlp_wide),
    }

    return results


def plot_model_comparison(results: dict, output_path: Path):
    """Plot comparison bar charts."""
    models = list(results.keys())
    coverage_gaps = [results[m]["coverage_gap"] for m in models]
    coverages = [results[m]["coverage"] for m in models]
    pre_coverages = [results[m]["pre_coverage"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Coverage gap
    ax = axes[0]
    ax.bar(models, coverage_gaps, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_ylabel("Coverage Gap", fontweight="bold")
    ax.set_title("Coverage Gap (Lower is Better)", fontweight="bold")
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Post-conformal coverage
    ax = axes[1]
    ax.bar(models, coverages, color="green", alpha=0.7, edgecolor="black")
    ax.axhline(0.95, color="red", linestyle="--", label="Target")
    ax.set_ylabel("Coverage", fontweight="bold")
    ax.set_title("Post-Conformal Coverage", fontweight="bold")
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Pre-conformal coverage
    ax = axes[2]
    ax.bar(models, pre_coverages, color="orange", alpha=0.7, edgecolor="black")
    ax.set_ylabel("Pre-Conformal Coverage", fontweight="bold")
    ax.set_title("Base Model Coverage", fontweight="bold")
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved comparison plot: {output_path}")


if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "nn_vs_gbm"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NEURAL NETWORK vs GRADIENT BOOSTING COMPARISON")
    print("=" * 80)
    print("\nThis will test whether NNs can outperform LightGBM on ~1,770 training samples")

    # Load data
    print("\nLoading data...")
    actuals = pd.read_parquet(DATA_DIR / "actuals_filtered_rts3_constellation_v1.parquet")
    forecasts = pd.read_parquet(DATA_DIR / "forecasts_filtered_rts3_constellation_v1.parquet")
    df = build_conformal_totals_df(actuals, forecasts)

    feature_cols = [
        "ens_mean",
        "ens_std",
        "ens_min",
        "ens_max",
        "n_models",
        "hour",
        "dow",
    ]

    # Run comparison
    results = compare_models_on_conformal(df, feature_cols, alpha_target=0.95)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results).T
    print(results_df)

    # Identify winner
    best_model = min(results.keys(), key=lambda m: results[m]["coverage_gap"])
    print(f"\n🏆 WINNER: {best_model}")
    print(f"   Coverage gap: {results[best_model]['coverage_gap']:.3f}")
    print(f"   Post-conformal coverage: {results[best_model]['coverage']:.3f}")

    # Plot
    plot_path = OUTPUT_DIR / "nn_vs_gbm_comparison.png"
    plot_model_comparison(results, plot_path)

    # Save results
    import json
    results_json = OUTPUT_DIR / "nn_vs_gbm_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_json}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    if best_model == "lightgbm":
        print("\nLightGBM outperforms neural networks on this dataset.")
        print("Recommendation: Stick with LightGBM, focus on feature engineering.")
    else:
        print(f"\n{best_model} outperforms LightGBM!")
        print(f"Recommendation: Consider switching to neural networks.")
