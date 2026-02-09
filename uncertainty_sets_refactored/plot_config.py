"""
Central plotting configuration for paper-ready figures.

Import and call `setup_plotting()` at the start of any visualization script.

Usage:
    from plot_config import setup_plotting
    setup_plotting()
"""
import matplotlib.pyplot as plt


# IEEE single-column figure sizes (3.5" wide)
# Fonts must be 7pt+ when printed
FONT_SIZES = {
    "small": 8,      # Ticks, legend
    "medium": 9,     # Axis labels
    "large": 9,      # Titles (often omitted for IEEE)
    "xlarge": 10,    # Figure suptitle
}

# For two-column figures (7" wide), double these
FONT_SIZES_TWO_COL = {
    "small": 9,
    "medium": 10,
    "large": 11,
    "xlarge": 12,
}

# IEEE figure widths
IEEE_COL_WIDTH = 3.5      # Single column (inches)
IEEE_TWO_COL_WIDTH = 7.16  # Two-column / full width (inches)

# Figure defaults
FIGURE_DEFAULTS = {
    "dpi": 150,
    "dpi_pdf": 300,
}

# Colors
COLORS = {
    "global": "#6c757d",      # Gray - baseline
    "knn": "#2A9D8F",         # Teal - k-NN
    "learned": "#E63946",     # Red - learned omega
    "actual": "#32CD32",      # Lime green - actual observation
}


def setup_plotting():
    """Configure matplotlib for paper-ready figures with larger fonts."""
    plt.rcParams.update({
        # Font sizes
        "font.size": FONT_SIZES["medium"],
        "axes.titlesize": FONT_SIZES["large"],
        "axes.labelsize": FONT_SIZES["medium"],
        "xtick.labelsize": FONT_SIZES["small"],
        "ytick.labelsize": FONT_SIZES["small"],
        "legend.fontsize": FONT_SIZES["small"],
        "figure.titlesize": FONT_SIZES["xlarge"],

        # Font weight
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",

        # Figure
        "figure.dpi": FIGURE_DEFAULTS["dpi"],
        "savefig.dpi": FIGURE_DEFAULTS["dpi"],
        "savefig.bbox": "tight",

        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 8,

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,

        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "gray",
    })


def get_color(name: str) -> str:
    """Get a named color from the palette."""
    return COLORS.get(name, "#000000")
