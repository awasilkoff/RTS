#!/usr/bin/env bash
# Regenerate all sweep data + paper figures (actuals AND residuals).
# Usage: cd uncertainty_sets_refactored && bash run_all_sweeps.sh 2>&1 | tee all_sweeps.log
set -euo pipefail
cd "$(dirname "$0")"

echo "============ ACTUALS MODE ============"

echo "=== [1/10] sweep_and_viz focused_2d ==="
python sweep_and_viz_feature_set.py --feature-set focused_2d

echo "=== [2/10] sweep_and_viz high_dim_16d ==="
python sweep_and_viz_feature_set.py --feature-set high_dim_16d

echo "=== [3/10] sweep_knn_k_values ==="
python sweep_knn_k_values.py

echo "=== [4/10] diagnose_tau_omega_seeds ==="
python diagnose_tau_omega_seeds.py

echo "=== [5/10] generate_paper_figures ==="
python generate_paper_figures.py

echo "============ RESIDUALS MODE ============"

echo "=== [6/10] sweep_and_viz focused_2d (residuals) ==="
python sweep_and_viz_feature_set.py --feature-set focused_2d --use-residuals

echo "=== [7/10] sweep_and_viz high_dim_16d (residuals) ==="
python sweep_and_viz_feature_set.py --feature-set high_dim_16d --use-residuals

echo "=== [8/10] sweep_knn_k_values (residuals) ==="
python sweep_knn_k_values.py --use-residuals

echo "=== [9/10] diagnose_tau_omega_seeds (residuals) ==="
python diagnose_tau_omega_seeds.py --use-residuals

echo "=== [10/10] generate_paper_figures (residuals) ==="
python generate_paper_figures.py --use-residuals

echo "=== Done! Outputs: paper_final/ and paper_final_residuals/ ==="
