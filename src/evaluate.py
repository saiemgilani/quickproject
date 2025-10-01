from __future__ import annotations

import os

import pandas as pd
from rich import print

PRED_DIR = os.path.join("data", "predictions")
REPORTS_DIR = "reports"


def build_calibration_plot(calib: pd.DataFrame) -> None:
    """Builds and saves a calibration plot comparing predicted win probabilities to actual outcomes.

    This function creates a scatter plot with a reference line to visualize the calibration of predicted probabilities.

    Args:
        calib (pd.DataFrame): DataFrame containing binned predicted probabilities and actual win rates.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.scatter(calib["pred_home_wp_mean"], calib["actual_home_wp"], s=calib["n_games"] * 3, alpha=0.7)
    plt.xlabel("Mean Predicted Home Win Probability")
    plt.ylabel("Actual Home Win Rate")
    plt.title("Calibration Plot")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "calibration_plot.png")
    plt.savefig(out_path)
    print(f"[bold blue]Wrote[/bold blue] {out_path}")
    plt.close()


def make_report() -> str:
    """Generates a markdown report summarizing model performance by season.

    This function reads evaluation metrics from a CSV file, formats them into a markdown table, writes the report to disk, and returns the report path.

    Returns:
        str: The file path to the generated markdown report.
    """
    eval_seasons_csv = os.path.join(PRED_DIR, "evaluation_seasons.csv")
    eval_weeks_csv = os.path.join(PRED_DIR, "evaluation_weeks.csv")
    calib_csv = os.path.join(PRED_DIR, "calibration_data_wp.csv")
    if not os.path.exists(eval_seasons_csv):
        raise FileNotFoundError("Run the pipeline first to create data/predictions/evaluation_seasons.csv")
    if not os.path.exists(eval_weeks_csv):
        raise FileNotFoundError("Run the pipeline first to create data/predictions/evaluation_weeks.csv")
    if not os.path.exists(calib_csv):
        raise FileNotFoundError("Run the pipeline first to create data/predictions/calibration_data_wp.csv")
    calib_df = pd.read_csv(calib_csv)
    build_calibration_plot(calib_df)
    seasons_df = pd.read_csv(eval_seasons_csv)
    weeks_df = pd.read_csv(eval_weeks_csv)

    seasom_lines = [
        "# Model Performance Summary by Season\n",
        "| Season | Games | MAE Home Margin | MAE Home | MAE Away | RMSE Home | RMSE Away | RMSE Margin | MAE Margin | Brier Score WP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in seasons_df.iterrows():
        seasom_lines.append(
            f"| {int(r['season'])} | {int(r['n_games'])} | {r['mae_home_margin']:.2f} | {r['mae_home']:.2f} | {r['mae_away']:.2f} | {r['rmse_home']:.2f} | {r['rmse_away']:.2f} | {r['rmse_margin']:.2f} | {r['mae_margin']:.2f} | {r['brier_score_loss_home_wp']:.3f} |"
        )

    lines = [
        "# Model Performance Summary by Week\n",
        "| Season | Week | Games | MAE Home Margin | MAE Home | MAE Away | RMSE Home | RMSE Away | RMSE Margin | MAE Margin | Brier Score WP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in weeks_df.iterrows():
        lines.append(
            f"| {int(r['season'])} | {r['week']} | {int(r['n_games'])} | {r['mae_home_margin']:.2f} | {r['mae_home']:.2f} | {r['mae_away']:.2f} | {r['rmse_home']:.2f} | {r['rmse_away']:.2f} | {r['rmse_margin']:.2f} | {r['mae_margin']:.2f} | {r['brier_score_loss_home_wp']:.3f} |"
        )
    md = "\n".join(seasom_lines) + "\n" + "\n" + "\n".join(lines)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "PERFORMANCE.md")
    open(out_path, "w").write(md)
    print(f"[bold blue]Wrote[/bold blue] {out_path}")
    return out_path
