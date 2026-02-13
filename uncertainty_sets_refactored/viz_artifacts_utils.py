"""
Utilities for organizing visualization artifacts and experiment configs.

Provides structured directory management and metadata tracking for
feature engineering experiments.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Any


def resolve_residuals_config(use_residuals: bool, data_dir: Path) -> dict[str, Any]:
    """
    Single source of truth for residuals-mode configuration.

    Returns dict with keys: actuals_parquet, actual_col, zero_mean, suffix.
    """
    if use_residuals:
        return {
            "actuals_parquet": data_dir / "residuals_filtered_rts3_constellation_v1.parquet",
            "actual_col": "RESIDUAL",
            "zero_mean": True,
            "suffix": "_residuals",
        }
    return {
        "actuals_parquet": data_dir / "actuals_filtered_rts3_constellation_v1.parquet",
        "actual_col": "ACTUAL",
        "zero_mean": False,
        "suffix": "",
    }


def setup_feature_set_directory(
    feature_set_name: str,
    feature_config: dict[str, Any],
    base_dir: Path | str = Path("data/viz_artifacts"),
) -> Path:
    """
    Create organized directory structure with metadata for a feature set.

    Parameters
    ----------
    feature_set_name : str
        Name of feature set (e.g., "temporal_nuisance_3d")
    feature_config : dict
        Configuration containing:
        - x_cols: list of feature names
        - y_cols: list of target names
        - n_features: feature dimensionality
        - standardize: whether features are standardized
        - tau, ridge, k: hyperparameters
    base_dir : Path
        Base directory for artifacts

    Returns
    -------
    artifact_dir : Path
        Path to created artifact directory
    """
    base_dir = Path(base_dir)
    artifact_dir = base_dir / feature_set_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save config as JSON
    config_path = artifact_dir / "feature_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                **feature_config,
                "created_at": datetime.now().isoformat(),
                "feature_set": feature_set_name,
            },
            f,
            indent=2,
        )

    # Create README
    readme = _generate_readme(feature_set_name, feature_config)
    readme_path = artifact_dir / "README.md"
    readme_path.write_text(readme)

    print(f"Created artifact directory: {artifact_dir}")
    print(f"  - Config: {config_path}")
    print(f"  - README: {readme_path}")

    return artifact_dir


def _generate_readme(feature_set_name: str, config: dict[str, Any]) -> str:
    """Generate README content for feature set directory."""
    x_cols = config.get("x_cols", [])
    y_cols = config.get("y_cols", [])
    n_features = config.get("n_features", len(x_cols))
    standardize = config.get("standardize", True)

    readme = f"""# {feature_set_name} Artifacts

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Feature Configuration

**Feature Set:** {feature_set_name}

**Input Features (X):** {n_features}D
{_format_list(x_cols)}

**Target Variables (Y):** {len(y_cols)} resources
{_format_list(y_cols)}

**Standardization:** {"Yes (zero mean, unit variance)" if standardize else "No (raw features)"}

## Hyperparameters

- **tau:** {config.get('tau', 'N/A')} (kernel bandwidth)
- **ridge:** {config.get('ridge', 'N/A')} (covariance regularization)
- **k:** {config.get('k', 'N/A')} (number of neighbors)
- **omega_l2_reg:** {config.get('omega_l2_reg', 'N/A')} (omega regularization)

## Expected Outputs

- `feature_config.json` - Full configuration
- `sweep_results.csv` - Hyperparameter sweep results
- `best_omega.npy` - Learned feature weights
- `comparison_nll_bar.png` - NLL comparison bar chart
- `omega_weights.png` - Learned omega visualization
- `kernel_distance.png` - Kernel weight visualization
- Additional visualizations based on dimensionality

## Interpretation

### Expected Omega Behavior

{_get_interpretation_notes(feature_set_name)}

### How to Compare

1. Check `sweep_results.csv` for NLL improvements
2. Compare learned omega in `best_omega.npy` to equal weights baseline
3. Look for clear visual differences in kernel weight distributions
4. Verify omega values match expected pattern (see above)

## Files in This Directory

(Auto-populated as outputs are generated)
"""
    return readme


def _format_list(items: list[str], max_inline: int = 5) -> str:
    """Format list for markdown display."""
    if len(items) <= max_inline:
        return "- " + ", ".join(items)
    else:
        return "\n".join(f"  - {item}" for item in items)


def _get_interpretation_notes(feature_set_name: str) -> str:
    """Get expected behavior notes for each feature set."""
    notes = {
        "temporal_nuisance_3d": """
**Expected:** ω ≈ [α, β, ~0]

The learned omega should downweight the HOUR_SIN feature (weakly-relevant temporal)
while preserving weights for SYS_MEAN and SYS_STD. Look for ω[2] ≈ 0.

**Why this matters:** Demonstrates omega learning can identify and suppress nuisance features.
""",
        "per_resource_4d": """
**Expected:** ω discovers differential farm importance

Different wind farms have different predictive value for covariance. The learned omega
should identify which farms are most informative (highest ω).

**Why this matters:** Shows omega can learn feature relevance in natural multi-source data.
Visualize via 2D projection of top 2 ω-weighted features.
""",
        "unscaled_2d": """
**Expected:** ω learns feature rescaling

SYS_MEAN_MW (~500) and SYS_STD_MW (~50) have different scales. Equal weights
are dominated by large-scale feature. Learned ω should discover correct rescaling.

**Why this matters:** Demonstrates omega automatically handles feature scale differences,
equivalent to learned standardization. Compare ω to 1/variance of each feature.
""",
        "baseline_2d": """
**Baseline:** Standard 2D system-level features

SYS_MEAN and SYS_STD after standardization. This is the reference case.
Expected improvement from learned omega is modest but measurable.
""",
    }
    return notes.get(feature_set_name, "No specific interpretation notes available.")


def load_feature_config(artifact_dir: Path | str) -> dict[str, Any]:
    """Load feature configuration from artifact directory."""
    artifact_dir = Path(artifact_dir)
    config_path = artifact_dir / "feature_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {config_path}")

    with open(config_path) as f:
        return json.load(f)


def save_sweep_summary(
    artifact_dir: Path | str,
    sweep_df: "pd.DataFrame",
    best_row_idx: int,
    omega_best: "np.ndarray",
    test_results: dict | None = None,
):
    """
    Save sweep results and best omega to artifact directory.

    Parameters
    ----------
    artifact_dir : Path
        Artifact directory
    sweep_df : DataFrame
        Sweep results with columns: tau, omega_l2_reg, val_nll_learned, etc.
    best_row_idx : int
        Index of best configuration in sweep_df
    omega_best : ndarray
        Best learned omega
    test_results : dict, optional
        Final test set results (unbiased evaluation)
    """
    import numpy as np
    import pandas as pd

    artifact_dir = Path(artifact_dir)

    # Save sweep CSV
    sweep_path = artifact_dir / "sweep_results.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved sweep results: {sweep_path}")

    # Save best omega
    omega_path = artifact_dir / "best_omega.npy"
    np.save(omega_path, omega_best)
    print(f"Saved best omega: {omega_path}")

    # Save best config summary
    best_row = sweep_df.iloc[best_row_idx]
    summary = {
        "best_config": {
            "tau": float(best_row.get("tau", 0)),
            "omega_l2_reg": float(best_row.get("omega_l2_reg", 0)),
            "omega_constraint": str(best_row.get("omega_constraint", "none")),
            "ridge": float(best_row.get("ridge", 1e-3)),
        },
        "best_omega": omega_best.tolist(),
        # Validation metrics (used for hyperparameter selection)
        "val_nll_learned": float(best_row.get("val_nll_learned", 0)),
        "val_nll_baseline": float(best_row.get("val_nll_baseline", 0)),
        "val_nll_improvement": float(best_row.get("nll_improvement_vs_kernel", 0)),
    }

    # Add test results if provided (unbiased evaluation)
    if test_results is not None:
        summary["test_results"] = test_results

    summary_path = artifact_dir / "best_config_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved best config summary: {summary_path}")


def update_readme_file_list(artifact_dir: Path | str):
    """Update README with list of generated files (called after all outputs created)."""
    artifact_dir = Path(artifact_dir)
    readme_path = artifact_dir / "README.md"

    if not readme_path.exists():
        return

    # Get all files (excluding README and config)
    files = sorted(
        [
            f.name
            for f in artifact_dir.iterdir()
            if f.is_file() and f.name not in ["README.md", "feature_config.json"]
        ]
    )

    # Append file list to README
    file_list_section = "\n\n## Generated Files\n\n"
    for fname in files:
        file_list_section += f"- `{fname}`\n"

    readme_content = readme_path.read_text()

    # Replace old file list if exists, otherwise append
    if "## Generated Files" in readme_content:
        # Replace existing section
        lines = readme_content.split("\n")
        new_lines = []
        skip = False
        for line in lines:
            if line.startswith("## Generated Files"):
                skip = True
            elif skip and line.startswith("##"):
                skip = False

            if not skip:
                new_lines.append(line)

        readme_content = "\n".join(new_lines)

    readme_path.write_text(readme_content + file_list_section)
