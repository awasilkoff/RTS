#!/usr/bin/env bash
# Regenerate all sweep data + paper figures (actuals AND residuals).
# Usage: cd uncertainty_sets_refactored && bash run_all_sweeps.sh 2>&1 | tee all_sweeps.log
set -euo pipefail
cd "$(dirname "$0")"

echo "============ ACTUALS MODE ============"

echo "=== [1/8] sweep_and_viz focused_2d ==="
python sweep_and_viz_feature_set.py --feature-set focused_2d --n-seeds 10

echo "=== [2/8] sweep_and_viz high_dim_16d ==="
python sweep_and_viz_feature_set.py --feature-set high_dim_16d --n-seeds 10

echo "=== [3/8] sweep_knn_k_values (multi-split) ==="
python sweep_knn_k_values.py --multi-split

echo "=== [4/8] generate_paper_figures ==="
python generate_paper_figures.py

echo "============ RESIDUALS MODE ============"

echo "=== [5/8] sweep_and_viz focused_2d (residuals) ==="
python sweep_and_viz_feature_set.py --feature-set focused_2d --use-residuals --n-seeds 10

echo "=== [6/8] sweep_and_viz high_dim_16d (residuals) ==="
python sweep_and_viz_feature_set.py --feature-set high_dim_16d --use-residuals --n-seeds 10

echo "=== [7/8] sweep_knn_k_values (residuals, multi-split) ==="
python sweep_knn_k_values.py --use-residuals --multi-split

echo "=== [8/8] generate_paper_figures (residuals) ==="
python generate_paper_figures.py --use-residuals

echo "=== Done! Outputs: paper_final/ and paper_final_residuals/ ==="
