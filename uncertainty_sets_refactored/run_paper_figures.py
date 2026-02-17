#!/usr/bin/env python3
"""
Generate IEEE paper figures for conformal prediction section.

This script creates three publication-quality figures:
1. Timeseries overlay - shows method visually
2. Calibration curve - validates conformal guarantee
3. Adaptive correction summary - explains why binning helps

Outputs are saved to data/viz_artifacts/paper_figures/ in both PDF and PNG formats.
"""
from pathlib import Path
from viz_conformal_paper import generate_paper_figures

if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent / "data"
    OUTPUT_DIR = DATA_DIR / "viz_artifacts" / "paper_figures"

    print("\n" + "=" * 80)
    print("IEEE PAPER FIGURE GENERATOR")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nThis will generate:")
    print("  - Timeseries overlay (PNG)")
    print("  - Calibration curve (PDF + PNG)")
    print("  - Adaptive correction summary (PDF + PNG)")
    print("\nEstimated time: 2-3 minutes")
    print("=" * 80 + "\n")

    paths = generate_paper_figures(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        alpha_values=[0.80, 0.85, 0.90, 0.95, 0.99],
        primary_alpha=0.95,
    )

    print("\n" + "=" * 80)
    print("SUCCESS - FIGURES READY FOR PAPER")
    print("=" * 80)
    print("\nGenerated files:")
    for name, path in paths.items():
        if path.exists():
            print(f"  (ok) {name}: {path}")
        else:
            print(f"  (x) {name}: {path} (NOT FOUND)")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Copy PDF files to your LaTeX figures/ directory:")
    print(f"   cp {paths['calibration_pdf']} /path/to/paper/figures/")
    print(f"   cp {paths['correction_pdf']} /path/to/paper/figures/")
    print(f"   cp {paths['timeseries']} /path/to/paper/figures/")
    print("\n2. Include in LaTeX with:")
    print("   \\includegraphics[width=\\columnwidth]{fig_calibration_curve.pdf}")
    print("   \\includegraphics[width=\\textwidth]{fig_adaptive_correction.pdf}")
    print("\n3. Review metadata for paper text:")
    print(f"   cat {paths['metadata']}")
    print("=" * 80 + "\n")
