from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_conformal_timeseries(
    df_eval_pred: pd.DataFrame,
    *,
    time_col: str = "TIME_HOURLY",
    actual_col: str = "y",
    mean_col: str = "ens_mean",
    base_pred_col: str = "y_pred_base",
    conf_pred_col: str = "y_pred_conf",
    out_png: Path,
    title: str = "Forecast vs Conformal Lower Bound vs Actual",
    max_points: int | None = None,
):
    """
    Line chart showing:
      - Ensemble mean forecast
      - Base quantile forecast
      - Conformalized lower bound
      - Actual total

    df_eval_pred must already include predictions from bundle.predict_df()
    and actual values.
    """

    df = df_eval_pred.copy()
    df = df.sort_values(time_col)

    if max_points is not None and len(df) > max_points:
        df = df.iloc[-max_points:]  # keep most recent window

    t = df[time_col]

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    ax.plot(t, df[actual_col], label="Actual", linewidth=2)
    ax.plot(t, df[mean_col], label="Ensemble Mean Forecast", linestyle="--")
    ax.plot(t, df[base_pred_col], label="Base Quantile Prediction")
    ax.plot(t, df[conf_pred_col], label="Conformal Lower Bound", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Output")

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    DATA_DIR = Path("/Users/alexwasilkoff/PycharmProjects/RTS/uncertainty_sets/data/")
    ART = DATA_DIR / "viz_artifacts"

    # Load the evaluation dataframe you already generated in main:
    # This should be df_pred_eval with y_pred_base, y_pred_conf, ens_mean, y, TIME_HOURLY
    df_eval_pred = pd.read_parquet(ART / "viz_eval_conformal_frame.parquet")

    plot_conformal_timeseries(
        df_eval_pred,
        out_png=ART / "timeseries_conformal_overlay.png",
        max_points=500,  # optional: zoom recent window
        title="System Forecast vs Conformal Lower Bound (α=0.95)",
    )

    print("Saved:", ART / "timeseries_conformal_overlay.png")
