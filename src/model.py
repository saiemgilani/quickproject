from __future__ import annotations

import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from rich import print
from scipy.stats import skellam
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEAT_DIR = os.path.join("data", "features")
PRED_DIR = os.path.join("data", "predictions")
MODELS_DIR = os.path.join("data", "models")

RANDOM_SEED = 42


def _load_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads processed game features from parquet files.

     This function reads the game features dataset from the features directory and returns it as a pandas DataFrame.

     Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The full game features DataFrame.
            - A slimmed-down version of the game features DataFrame with reduced features.
    """
    return pd.read_parquet(os.path.join(FEAT_DIR, "game_features.parquet")), pd.read_parquet(os.path.join(FEAT_DIR, "game_features_slim.parquet"))




def _feature_matrix(df: pd.DataFrame, feature_columns_selected: list[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Creates the feature matrix and target variables for model training.

    This function extracts numeric features and separates the home and away scores.

    Args:
        df (pd.DataFrame): The input DataFrame containing game features and scores.
        feature_columns_selected (list[str]): The list of selected feature columns to use for training.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing the feature matrix, home scores, and away scores.
    """
    y_home = df["home_score"]
    y_away = df["away_score"]
    drop_cols = ["season", "week", "home_score", "away_score", "game_id", "home_team", "away_team", "game_type"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    if feature_columns_selected:
        X = X[[c for c in feature_columns_selected if c in X.columns]]

    return X, y_home, y_away


def _make_model(poisson: bool = True) -> XGBRegressor:
    """Creates and configures an XGBoost regression model for training.

    This function initializes an XGBRegressor with specified hyperparameters and sets the objective based on the poisson argument.

    Args:
        poisson (bool): If True, uses Poisson regression objective; otherwise uses squared error regression.

    Returns:
        XGBRegressor: The configured XGBoost regression model.
    """
    params = dict(
        n_estimators=50,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        reg_alpha=0.0,
        reg_lambda=1.0,
        verbosity=1
    )
    params["objective"] = "count:poisson" if poisson else "reg:squarederror"
    return XGBRegressor(**params)


def _generate_shap_summary_plot(
    model: XGBRegressor, X: pd.DataFrame, model_name: str, season: int, model_type: str
) -> None:
    """Generates and saves a SHAP summary plot for a given model.

    Args:
        model (XGBRegressor): The trained XGBoost model.
        X (pd.DataFrame): The feature matrix used for training.
        model_name (str): The name of the model configuration (e.g., 'poisson').
        season (int): The season for which the model was trained.
        model_type (str): The type of model ('home' or 'away').
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        print("[bold yellow]SHAP library not installed. Skipping SHAP plots. Run `pip install shap`.[/bold yellow]")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance for {model_name} ({model_type}, {season})")
    plt.tight_layout()

    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"shap_summary_{model_name}_{model_type}_{season}.png")
    plt.savefig(out_path)
    plt.close()
    # print(f"[bold blue]Wrote[/bold blue] SHAP summary plot to {out_path}")


def walkforward_train_predict(
    df: pd.DataFrame, feature_columns_selected: list[str], model_name: str, poisson: bool = True, use_prior_seasons: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trains and predicts game outcomes using a seasonal 80/20 split.

    For each season, this function splits the data into an 80% training set and a 20% test set,
    stratified by week. It trains models on the training set and predicts on the test set.
    It also captures feature importances and saves the trained models.

    Args:
        df (pd.DataFrame): The DataFrame containing game features and scores.
        feature_columns_selected (list[str]): The list of selected feature columns to use for training.
        model_name (str): The name of the model configuration, used for saving model files.
        poisson (bool): If True, uses Poisson regression objective; otherwise uses squared error regression.
        use_prior_seasons (bool): If True, includes all prior seasons in the training data for each week.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - A DataFrame with predictions for each game.
            - A DataFrame with feature importances from all trained models.
    """
    preds = []
    importances = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season].copy()

        # Split season data into 80% train and 20% test, stratified by week
        train, test = train_test_split(sdf, test_size=0.4, random_state=RANDOM_SEED, stratify=sdf["week"])

        if use_prior_seasons:
            train_prior = df[df["season"] < season].copy()
            train = pd.concat([train_prior, train], ignore_index=True)

        if len(train) == 0 or len(test) == 0:
            continue

        X_tr, yh_tr, ya_tr = _feature_matrix(train, feature_columns_selected)
        X_te, _, _ = _feature_matrix(test, feature_columns_selected)
        m_h = _make_model(poisson=poisson)
        m_a = _make_model(poisson=poisson)
        m_h.fit(X_tr, yh_tr)
        _generate_shap_summary_plot(m_h, X_tr, model_name, season, "home")
        m_a.fit(X_tr, ya_tr)
        _generate_shap_summary_plot(m_a, X_tr, model_name, season, "away")

        # Save models
        model_h_path = os.path.join(MODELS_DIR, f"{model_name}_home_{season}.pkl")
        with open(model_h_path, "wb") as f:
            pickle.dump(m_h, f)

        model_a_path = os.path.join(MODELS_DIR, f"{model_name}_away_{season}.pkl")
        with open(model_a_path, "wb") as f:
            pickle.dump(m_a, f)

        # Store feature importances
        imp_h = pd.DataFrame(
            {"feature": X_tr.columns, "importance": m_h.feature_importances_, "model": "home", "season": season}
        )
        imp_a = pd.DataFrame(
            {"feature": X_tr.columns, "importance": m_a.feature_importances_, "model": "away", "season": season}
        )
        importances.extend([imp_h, imp_a])

        test = test.copy()
        test["pred_home_points"] = np.select([(test["week"]==1) & (~use_prior_seasons)], [22.0 + 2.5], default=np.clip(m_h.predict(X_te), 0, None))
        test["pred_away_points"] = np.select([(test["week"]==1) & (~use_prior_seasons)], [22.0], default=np.clip(m_a.predict(X_te), 0, None))
        preds.append(test)

    if not preds:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.concat(preds, ignore_index=True)
    out["actual_margin_home"] = out["home_score"] - out["away_score"]
    out["pred_margin_home"] = out["pred_home_points"] - out["pred_away_points"]
    lam_h = np.clip(out["pred_home_points"].values, 0.01, None)
    lam_a = np.clip(out["pred_away_points"].values, 0.01, None)
    out["pred_home_wp"] = 1.0 - skellam.cdf(0, lam_h, lam_a) + 0.5 * skellam.pmf(0, lam_h, lam_a)
    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)

    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    out = out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return out, imp_df


def evaluate_season(preds: pd.DataFrame) -> pd.DataFrame:
    """Evaluates prediction accuracy for each season using error metrics.

    This function computes mean absolute error and root mean squared error for home and away scores, as well as margin, grouped by season.

    Args:
        preds (pd.DataFrame): DataFrame containing actual and predicted scores for each game.

    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each season.
    """
    rows = []
    for season, sdf in preds.groupby("season"):
        mae_home_margin = abs(sdf["actual_margin_home"] - sdf["pred_margin_home"]).mean()
        mae_home = mean_absolute_error(sdf["home_score"], sdf["pred_home_points"])
        mae_away = mean_absolute_error(sdf["away_score"], sdf["pred_away_points"])
        rmse_home = mean_squared_error(sdf["home_score"], sdf["pred_home_points"])
        rmse_away = mean_squared_error(sdf["away_score"], sdf["pred_away_points"])
        margin_true = sdf["home_score"] - sdf["away_score"]
        rmse_margin = mean_squared_error(margin_true, sdf["pred_margin_home"])
        mae_margin = mean_absolute_error(margin_true, sdf["pred_margin_home"])
        brier_score_loss_home_wp = brier_score_loss(sdf["home_win"], sdf["pred_home_wp"])
        rows.append(
            dict(
                season=int(season),
                mae_home_margin=mae_home_margin,
                mae_home=mae_home,
                mae_away=mae_away,
                rmse_home=rmse_home,
                rmse_away=rmse_away,
                rmse_margin=rmse_margin,
                mae_margin=mae_margin,
                brier_score_loss_home_wp=brier_score_loss_home_wp,
                n_games=len(sdf),
            )
        )
    return pd.DataFrame(rows)


def evaluate_weeks(preds: pd.DataFrame) -> pd.DataFrame:
    """Evaluates prediction accuracy for each week using error metrics.

    This function computes mean absolute error and root mean squared error for home and away scores, as well as margin, grouped by season and week.

    Args:
        preds (pd.DataFrame): DataFrame containing actual and predicted scores for each game.

    Returns:
        pd.DataFrame: DataFrame with evaluation metrics for each season and week.
    """
    rows = []
    for (season, week), sdf in preds.groupby(["season", "week"]):
        row = {
            "season": season,
            "week": week,
            "n_games": len(sdf),
            "mae_home": mean_absolute_error(sdf["home_score"], sdf["pred_home_points"]),
            "mae_away": mean_absolute_error(sdf["away_score"], sdf["pred_away_points"]),
            "mae_margin": mean_absolute_error(sdf["actual_margin_home"], sdf["pred_margin_home"]),
            "rmse_home": mean_squared_error(sdf["home_score"], sdf["pred_home_points"]),
            "rmse_away": mean_squared_error(sdf["away_score"], sdf["pred_away_points"]),
            "rmse_margin": mean_squared_error(sdf["actual_margin_home"], sdf["pred_margin_home"]),
            "brier_score_loss_home_wp": brier_score_loss(sdf["home_win"], sdf["pred_home_wp"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_calibration_data(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Aggregates data for calibration plots by binning predicted win probabilities.

    This function bins the predicted home win probabilities and computes the mean predicted probability and actual win rate for each bin.

    Args:
        preds (pd.DataFrame): DataFrame containing actual and predicted scores for each game.
        n_bins (int): The number of bins to use for calibration.

    Returns:
        pd.DataFrame: DataFrame with binned predicted probabilities and actual win rates.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    labels = (bins[:-1] + bins[1:]) / 2.0
    preds = preds.copy()
    preds["pred_bin"] = pd.cut(preds["pred_home_wp"], bins=bins, labels=labels, include_lowest=True)
    calib = preds.groupby("pred_bin", observed=False).agg(
        n_games=("home_win", "size"),
        pred_home_wp_mean=("pred_home_wp", "mean"),
        actual_home_wp=("home_win", "mean"),
    )
    calib = calib[calib["n_games"] > 0].reset_index()
    calib["pred_bin"] = calib["pred_bin"].astype(float)
    return calib


def _save_feature_importances(importances: pd.DataFrame, model_name: str) -> None:
    """Aggregates and saves feature importances.

    Args:
        importances (pd.DataFrame): DataFrame with feature importances from walk-forward training.
        model_name (str): Name of the model (e.g., 'poisson') for the output filename.
    """
    if importances.empty:
        return

    # Aggregate by taking the mean importance across all training weeks
    agg_imp = importances.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()

    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"feature_importances_{model_name}.csv")
    agg_imp.to_csv(out_path, index=False)
    print(f"[bold blue]Wrote[/bold blue] feature importances to {out_path}")


def plot_feature_importances(importances: pd.DataFrame, model_name: str) -> None:
    """Plots feature importances from the aggregated importances DataFrame.

    Args:
        importances (pd.DataFrame): DataFrame with feature importances.
        model_name (str): Name of the model (e.g., 'poisson') for the plot title and filename.
    """
    import matplotlib.pyplot as plt

    if importances.empty:
        print("[bold red]No importances to plot.[/bold red]")
        return

    plt.figure(figsize=(10, 6))
    plt.barh(importances["feature"], importances["importance"], color="skyblue")
    plt.xlabel("Mean Importance")
    plt.title(f"Feature Importances for {model_name} Model")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"feature_importances_{model_name}.png")
    plt.savefig(out_path)
    print(f"[bold blue]Wrote[/bold blue] feature importances plot to {out_path}")
    plt.close()

def run() -> pd.DataFrame:
    """Runs the full model pipeline for training, prediction, and evaluation for multiple model types.

    This function loads features, trains both a Poisson and a standard regression model,
    generates predictions, evaluates results, saves feature importances and trained models,
    and writes all outputs to disk.

    Returns:
        pd.DataFrame: The predictions DataFrame from the primary (Poisson) model.
    """
    os.makedirs(PRED_DIR, exist_ok=True)
    df, df_slim = _load_features()

    models_to_run = {
        "poisson": {"is_poisson": True, "use_prior_seasons": False},
        "baseline_rmse": {"is_poisson": False, "use_prior_seasons": False},
        "poisson_prior_seasons": {"is_poisson": True, "use_prior_seasons": True},
    }
    primary_preds = None

    for model_name, config in models_to_run.items():
        is_poisson = config["is_poisson"]
        use_prior = config["use_prior_seasons"]
        print(
            f"[bold yellow]Running model:[/bold yellow] {model_name} (poisson={is_poisson}, use_prior_seasons={use_prior})"
        )

        preds, importances = walkforward_train_predict(
            df, feature_columns_selected=df_slim.columns.tolist(), model_name=model_name, poisson=is_poisson, use_prior_seasons=use_prior
        )
        if model_name == "poisson":
            primary_preds = preds

        eval_seasons = evaluate_season(preds)
        eval_weeks = evaluate_weeks(preds)
        calib = aggregate_calibration_data(preds, n_bins=10)

        # Save outputs with model-specific names
        preds.to_parquet(os.path.join(PRED_DIR, f"predictions_{model_name}.parquet"))
        preds.to_csv(os.path.join(PRED_DIR, f"predictions_{model_name}.csv"), index=False)
        eval_seasons.to_csv(os.path.join(PRED_DIR, f"evaluation_seasons_{model_name}.csv"), index=False)
        eval_weeks.to_csv(os.path.join(PRED_DIR, f"evaluation_weeks_{model_name}.csv"), index=False)
        calib.to_csv(os.path.join(PRED_DIR, f"calibration_data_wp_{model_name}.csv"), index=False)

        _save_feature_importances(importances, model_name)
        plot_feature_importances(importances.groupby("feature")["importance"].mean().reset_index(), model_name)
        print(f"[bold green]Wrote[/bold green] predictions and evaluation for {model_name} to data/predictions/")

    print(f"[bold green]Saved[/bold green] trained models to {MODELS_DIR}")
    print("[bold green]Completed model runs.[/bold green]")
    return primary_preds
