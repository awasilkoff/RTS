#!/bin/bash
#
# Run all three feature set experiments and generate comparison summary.
#
# Usage: ./run_all_feature_sets.sh
#

set -e  # Exit on error

echo "=============================================================================="
echo "Feature Engineering Experiment Suite for Learned Omega Visualization"
echo "=============================================================================="
echo ""

# Feature sets to run
FEATURE_SETS=("temporal_3d" "per_resource_4d" "unscaled_2d")

# Track start time
START_TIME=$(date +%s)

# Run each feature set
for fs in "${FEATURE_SETS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "Running feature set: $fs"
    echo "------------------------------------------------------------------------------"
    echo ""

    python sweep_and_viz_feature_set.py --feature-set "$fs"

    echo ""
    echo "✓ Completed: $fs"
    echo ""
done

# Track end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=============================================================================="
echo "All feature sets complete!"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "=============================================================================="
echo ""

# Generate comparison summary
echo "Generating comparison summary..."
echo ""

python - << 'EOF'
import pandas as pd
from pathlib import Path
import numpy as np

artifact_dir = Path("data/viz_artifacts")
feature_sets = ["temporal_3d", "per_resource_4d", "unscaled_2d"]

results = []
for fs in feature_sets:
    df = pd.read_csv(artifact_dir / fs / "sweep_results.csv")
    best = df.sort_values("nll_improvement", ascending=False).iloc[0]

    omega = np.load(artifact_dir / fs / "best_omega.npy")

    results.append({
        "feature_set": fs,
        "nll_improvement": best["nll_improvement"],
        "eval_nll_learned": best["eval_nll_learned"],
        "eval_nll_baseline": best["eval_nll_baseline"],
        "improvement_pct": 100 * best["nll_improvement"] / best["eval_nll_baseline"],
        "tau": best["tau"],
        "omega_l2_reg": best["omega_l2_reg"],
        "standardize": best["standardize"],
        "omega": str(omega.round(3)),
    })

comparison = pd.DataFrame(results).sort_values("nll_improvement", ascending=False)

print("=" * 100)
print("COMPARISON SUMMARY")
print("=" * 100)
print()
print(comparison.to_string(index=False))
print()
print("=" * 100)
print(f"Best feature set for paper: {comparison.iloc[0]['feature_set']}")
print(f"  NLL improvement: {comparison.iloc[0]['nll_improvement']:.3f} ({comparison.iloc[0]['improvement_pct']:.2f}%)")
print(f"  Learned omega: {comparison.iloc[0]['omega']}")
print("=" * 100)
print()
print("Outputs saved to: data/viz_artifacts/<feature_set>/")
print("  - sweep_results.csv")
print("  - best_omega.npy")
print("  - omega_bar_chart.png")
print("  - kernel_distance_*.png")
print("  - feature_config.json")
print("  - README.md")
print()
EOF

echo "Done!"
echo ""
