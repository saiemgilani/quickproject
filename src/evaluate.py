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
    """Generates a markdown report summarizing model performance and comparing model types.

    This function reads evaluation metrics for different models, creates a comparison summary,
    adds a discussion of the results, and appends the detailed performance report for the primary (Poisson) model.

    Returns:
        str: The file path to the generated markdown report.
    """
    # --- Check for primary model (poisson) files ---
    primary_model_name = "poisson"
    eval_seasons_csv = os.path.join(PRED_DIR, f"evaluation_seasons_{primary_model_name}.csv")
    eval_weeks_csv = os.path.join(PRED_DIR, f"evaluation_weeks_{primary_model_name}.csv")
    calib_csv = os.path.join(PRED_DIR, f"calibration_data_wp_{primary_model_name}.csv")

    if not all(os.path.exists(f) for f in [eval_seasons_csv, eval_weeks_csv, calib_csv]):
        raise FileNotFoundError(
            f"Primary model ('{primary_model_name}') evaluation files not found. Run the pipeline first."
        )

    # --- Build Calibration Plot for Primary Model ---
    calib_df = pd.read_csv(calib_csv)
    build_calibration_plot(calib_df)

    # --- Model Comparison Section ---
    model_names = ["poisson", "baseline_rmse", "poisson_prior_seasons"]
    model_evals = {}
    for name in model_names:
        fpath = os.path.join(PRED_DIR, f"evaluation_seasons_{name}.csv")
        if os.path.exists(fpath):
            model_evals[name] = pd.read_csv(fpath)

    comparison_lines = []
    if model_evals:
        # Build header and divider
        header_parts = ["| Season |"]
        divider_parts = ["|---:|"]
        for name in model_names:
            if name in model_evals:
                header_parts.append(f" MAE Margin ({name}) | Brier Score WP ({name}) |")
                divider_parts.append("---:|---:|")
        header = "".join(header_parts)
        divider = "".join(divider_parts)
        comparison_lines.extend(["# Model Comparison\n", header, divider])

        # Merge all available dataframes to create a comprehensive comparison table
        all_seasons = sorted(list(set(s for df in model_evals.values() for s in df["season"])))
        comp_df = pd.DataFrame({"season": all_seasons})
        for name, df in model_evals.items():
            df_to_merge = df[["season", "mae_margin", "brier_score_loss_home_wp"]].rename(
                columns={
                    "mae_margin": f"mae_margin_{name}",
                    "brier_score_loss_home_wp": f"brier_score_loss_home_wp_{name}",
                }
            )
            comp_df = comp_df.merge(df_to_merge, on="season", how="left")

        # Build rows with NaN handling
        for _, r in comp_df.iterrows():
            row_parts = [f"| {int(r['season'])} |"]
            for name in model_names:
                if name in model_evals:
                    mae_val = r.get(f"mae_margin_{name}")
                    brier_val = r.get(f"brier_score_loss_home_wp_{name}")

                    mae_str = f"{mae_val:.2f}" if pd.notna(mae_val) else "N/A"
                    brier_str = f"{brier_val:.3f}" if pd.notna(brier_val) else "N/A"

                    row_parts.append(f" {mae_str} | {brier_str} |")
            comparison_lines.append("".join(row_parts))

    # --- Analysis and Discussion Section ---
    discussion = """
# Analysis and Discussion

This report summarizes the performance of multiple models:
- **`Poisson`**: The primary model using an XGBoost regressor with a `count:poisson` objective, trained only on data from the current season (walk-forward). This adheres to the original "cold start" per-season design.
- **`Baseline RMSE`**: An XGBoost regressor with a standard `reg:squarederror` objective, also trained only on data from the current season.
- **`Poisson Prior Seasons`**: Same as the primary Poisson model, but trained on all available data from previous seasons plus the current season's data up to the prediction week. This tests the impact of using a larger, historical training set.
- **`Logistic Regression WP`**: An XGBoost classifier with a `binary:logistic` objective, trained on the same features but predicting win probabilities directly.

- **Performance**: The comparison table highlights the differences in key metrics. Comparing the `Poisson` and `Poisson Prior Seasons` models shows the impact of using historical data, which may improve performance in early weeks but could also introduce noise from outdated team dynamics.
- **Feature Importance**: Feature importances for all models have been saved to the `reports/` directory. This allows for analysis of which factors are most influential for each modeling approach.

## Limitations and Next Steps
The current approach uses fixed hyperparameters. Future improvements could include:
- Hyperparameter tuning for all models.
- Ensembling predictions from multiple models.
- Adding more features related to player availability, injuries, or weather.
"""

    # --- Detailed Report for Primary Model ---
    seasons_df = pd.read_csv(eval_seasons_csv)
    weeks_df = pd.read_csv(eval_weeks_csv)

    season_lines = [
        f"# Model Performance Summary by Season ({primary_model_name.capitalize()})\n",
        "| Season | MAE Home Margin | MAE Home | MAE Away | RMSE Home | RMSE Away | RMSE Margin | MAE Margin | Brier Score WP | LogLoss WP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in seasons_df.iterrows():
        season_lines.append(
            f"| {int(r['season'])} | {r['mae_margin']:.2f} | {r['mae_home']:.2f} | {r['mae_away']:.2f} | {r['rmse_home']:.2f} | {r['rmse_away']:.2f} | {r['rmse_margin']:.2f} | {r['mae_margin']:.2f} | {r['brier_score_loss_home_wp']:.3f} | {r['log_loss_home_wp']:.3f} |"
        )

    week_lines = [
        f"# Model Performance Summary by Week ({primary_model_name.capitalize()})\n",
        "| Season | Week | MAE Home Margin | MAE Home | MAE Away | RMSE Home | RMSE Away | RMSE Margin | MAE Margin | Brier Score WP | LogLoss WP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in weeks_df.iterrows():
        week_lines.append(
            f"| {int(r['season'])} | {r['week']} | {r['mae_margin']:.2f} | {r['mae_home']:.2f} | {r['mae_away']:.2f} | {r['rmse_home']:.2f} | {r['rmse_away']:.2f} | {r['rmse_margin']:.2f} | {r['mae_margin']:.2f} | {r['brier_score_loss_home_wp']:.3f} | {r['log_loss_home_wp']:.3f} |"
        )

    # --- Assemble and Write Report ---
    md_parts = []
    if comparison_lines:
        md_parts.extend(comparison_lines)
        md_parts.append("\n")
    md_parts.append(discussion)
    md_parts.append("\n")
    md_parts.extend(season_lines)
    md_parts.append("\n\n")
    md_parts.extend(week_lines)
    md = "\n".join(md_parts)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "PERFORMANCE.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"[bold blue]Wrote[/bold blue] {out_path}")
    return out_path
