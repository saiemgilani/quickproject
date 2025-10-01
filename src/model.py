from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from rich import print
from scipy.stats import skellam
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

FEAT_DIR = os.path.join("data", "features")
PRED_DIR = os.path.join("data", "predictions")

RANDOM_SEED = 42


def _load_features() -> pd.DataFrame:
    """Loads processed game features pd.DataFrame:
    from a parquet file.

     This function reads the game features dataset from the features directory and returns it as a pandas DataFrame.

     Returns:
         pd.DataFrame: The game features DataFrame.
    """
    return pd.read_parquet(os.path.join(FEAT_DIR, "game_features.parquet"))


def _feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Creates the feature matrix and target variables for model training.

    This function extracts numeric features from the input DataFrame and separates the home and away scores as target variables.

    Args:
        df (pd.DataFrame): The input DataFrame containing game features and scores.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing the feature matrix, home scores, and away scores.
    """
    y_home = df["home_score"]
    y_away = df["away_score"]
    drop_cols = ["home_score", "away_score", "game_id", "home_team", "away_team", "game_type"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
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
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        reg_alpha=0.0,
        reg_lambda=1.0,
        verbosity=1,
    )
    params["objective"] = "count:poisson" if poisson else "reg:squarederror"
    return XGBRegressor(**params)


def walkforward_train_predict(df: pd.DataFrame, poisson: bool = True) -> pd.DataFrame:
    """Trains and predicts game outcomes using walk-forward validation.

    This function iteratively trains models on past weeks and predicts scores for the next week, simulating a real-time prediction scenario.

    Args:
        df (pd.DataFrame): The DataFrame containing game features and scores.
        poisson (bool): If True, uses Poisson regression objective; otherwise uses squared error regression.

    Returns:
        pd.DataFrame: DataFrame containing predictions for each game, including predicted scores, margin, and win probability.
    """
    preds = []
    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season].copy()
        for week in sorted(sdf["week"].unique()):
            train = sdf[sdf["week"] < week].copy()
            test = sdf[sdf["week"] == week].copy()
            if len(train) < 16:
                test = test.copy()
                test["pred_home_points"] = 22.0 + 2.5
                test["pred_away_points"] = 22.0
                preds.append(test)
                continue
            X_tr, yh_tr, ya_tr = _feature_matrix(train)
            X_te, _, _ = _feature_matrix(test)
            m_h = _make_model(poisson=poisson)
            m_a = _make_model(poisson=poisson)
            m_h.fit(X_tr, yh_tr)
            m_a.fit(X_tr, ya_tr)
            test = test.copy()
            test["pred_home_points"] = np.clip(m_h.predict(X_te), 0, None)
            test["pred_away_points"] = np.clip(m_a.predict(X_te), 0, None)
            preds.append(test)
    out = pd.concat(preds, ignore_index=True)
    out["actual_margin_home"] = out["home_score"] - out["away_score"]
    out["pred_margin_home"] = out["pred_home_points"] - out["pred_away_points"]
    lam_h = np.clip(out["pred_home_points"].values, 0.01, None)
    lam_a = np.clip(out["pred_away_points"].values, 0.01, None)
    out["pred_home_wp"] = 1.0 - skellam.cdf(0, lam_h, lam_a) + 0.5 * skellam.pmf(0, lam_h, lam_a)
    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)

    return out


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
                week=int(week),
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
    calib = preds.groupby("pred_bin").agg(
        n_games=("home_win", "size"),
        pred_home_wp_mean=("pred_home_wp", "mean"),
        actual_home_wp=("home_win", "mean"),
    )
    calib = calib[calib["n_games"] > 0].reset_index()
    calib["pred_bin"] = calib["pred_bin"].astype(float)
    return calib


def run(poisson: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the full model pipeline for training, prediction, and evaluation.

    This function loads features, trains models, generates predictions, evaluates results, and writes outputs to disk.

    Args:
        poisson (bool): If True, uses Poisson regression objective; otherwise uses squared error regression.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the predictions DataFrame and the evaluation DataFrame.
    """
    os.makedirs(PRED_DIR, exist_ok=True)
    df = _load_features()
    preds = walkforward_train_predict(df, poisson=poisson)
    eval_seasons = evaluate_season(preds)
    eval_weeks = evaluate_weeks(preds)
    calib = aggregate_calibration_data(preds, n_bins=10)

    # evaldf = pd.concat([evaldf, eval_weeks], ignore_index=True)
    preds.to_parquet(os.path.join(PRED_DIR, "predictions.parquet"))
    preds.to_csv(os.path.join(PRED_DIR, "predictions.csv"), index=False)
    eval_seasons.to_csv(os.path.join(PRED_DIR, "evaluation_seasons.csv"), index=False)
    eval_weeks.to_csv(os.path.join(PRED_DIR, "evaluation_weeks.csv"), index=False)
    calib.to_csv(os.path.join(PRED_DIR, "calibration_data_wp.csv"), index=False)

    print("[bold green]Wrote[/bold green] predictions and evaluation to data/predictions/")
    return preds, eval_seasons, eval_weeks
