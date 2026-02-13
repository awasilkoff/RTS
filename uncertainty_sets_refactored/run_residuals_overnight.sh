#!/usr/bin/env bash
# Overnight orchestrator: run all sweeps with --use-residuals, then generate paper figures.
# Usage: cd uncertainty_sets_refactored && bash run_residuals_overnight.sh 2>&1 | tee residuals_overnight.log
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 1/5: sweep_and_viz focused_2d (residuals) ==="
python sweep_and_viz_feature_set.py --feature-set focused_2d --use-residuals

echo "=== Step 2/5: sweep_and_viz high_dim_16d (residuals) ==="
python sweep_and_viz_feature_set.py --feature-set high_dim_16d --use-residuals

echo "=== Step 3/5: sweep_knn_k_values (residuals) ==="
python sweep_knn_k_values.py --use-residuals

echo "=== Step 4/5: diagnose_tau_omega_seeds (residuals) ==="
python diagnose_tau_omega_seeds.py --use-residuals

echo "=== Step 5/5: generate_paper_figures (USE_RESIDUALS=True) ==="
python -c "
import generate_paper_figures as gpf
gpf.USE_RESIDUALS = True
from pathlib import Path
gpf.ACTUALS_PARQUET = gpf.DATA_DIR / 'residuals_filtered_rts3_constellation_v1.parquet'
gpf.ACTUAL_COL = 'RESIDUAL'
gpf.OUTPUT_DIR = gpf.VIZ_ARTIFACTS / 'paper_final_residuals'
gpf.FOCUSED_2D_DIR = gpf.VIZ_ARTIFACTS / 'focused_2d_residuals'
gpf.HIGH_DIM_16D_DIR = gpf.VIZ_ARTIFACTS / 'high_dim_16d_residuals'
gpf.KNN_SWEEP_DIR = gpf.VIZ_ARTIFACTS / 'knn_k_sweep_residuals'
gpf.TAU_DIAGNOSIS_DIR = gpf.VIZ_ARTIFACTS / 'tau_omega_diagnosis_residuals'
gpf.generate_all_figures()
"

echo "=== Done! Outputs in data/viz_artifacts/paper_final_residuals/ ==="
