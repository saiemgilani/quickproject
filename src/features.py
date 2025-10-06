from __future__ import annotations

import os
from typing import List

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.utils.common import bin_yardline_100, bin_ydstogo, safe_div




RAW_DIR = os.path.join("data", "raw")
FEAT_DIR = os.path.join("data", "features")


def _load_raw():
    """Loads raw play-by-play and schedule data from parquet files.

    This function reads the play-by-play and schedule datasets from the raw data directory and returns them as pandas DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the play-by-play DataFrame and the schedule DataFrame.
    """
    pbp = pd.read_parquet(os.path.join(RAW_DIR, "pbp.parquet"))
    sched = pd.read_parquet(os.path.join(RAW_DIR, "schedules.parquet"))
    return pbp, sched


def _prep_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Prepares and standardizes play-by-play data for feature engineering.

    This function ensures required columns exist, fills missing values, creates derived features, and flags valid plays for modeling.

    Args:
        pbp (pd.DataFrame): The raw play-by-play DataFrame.

    Returns:
        pd.DataFrame: The processed play-by-play DataFrame ready for feature engineering.
    """
    df = pbp.copy()
    if "pass" not in df.columns and "pass_attempt" in df.columns:
        df["pass"] = df["pass_attempt"].astype("float").fillna(0.0) == 1.0
    if "rush" not in df.columns and "rush_attempt" in df.columns:
        df["rush"] = df["rush_attempt"].astype("float").fillna(0.0) == 1.0
    if "epa" not in df.columns:
        df["epa"] = 0.0
    if "posteam" not in df.columns and "pos_team" in df.columns:
        df["posteam"] = df["pos_team"]
    if "defteam" not in df.columns and "def_team" in df.columns:
        df["defteam"] = df["def_team"]
    if "yardline_100" not in df.columns and "yardline" in df.columns:
        df["yardline_100"] = df["yardline"]
    for c in ["down", "ydstogo"]:
        if c not in df.columns:
            df[c] = np.nan
    if "week" not in df.columns and "game_week" in df.columns:
        df["week"] = df["game_week"]
    if "season" not in df.columns:
        if "season" in df.columns:  # redundant but safe
            pass
        elif "season_year" in df.columns:
            df["season"] = df["season_year"]
    # Filter common non-plays if present
    df["is_valid_model_play"] = np.select(
        [
            df["season_type"].fillna("REG") != "REG",
            df["play_type"].isin(["no_play", "kickoff", "extra_point", "two_point_attempt", "punt"]) == True,
            df["play_type"].isna(),
            df["special_teams_play"].fillna(0.0) == 1.0,
            df["timeout"].fillna(0.0) == 1.0,
            df["penalty"].fillna(0.0) == 1.0,
            df["qb_spike"].fillna(0.0) == 1.0,
            df["qb_kneel"].fillna(0.0) == 1.0,
        ],
        [False, False, False, False, False, False, False, False],
        default=True,
    )
    df["ydstogo_bin"] = df["ydstogo"].apply(bin_ydstogo)
    df["yardline_bin"] = df["yardline_100"].apply(bin_yardline_100)
    df["is_pass"] = df["pass"].fillna(False).astype(bool)
    df["is_rush"] = df["rush"].fillna(False).astype(bool)
    return df


def _expected_pass_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates expected pass probability for each game situation.

    This function groups plays by season, down, distance, and field position, and computes the mean xpass for each group.

    Args:
        df (pd.DataFrame): The processed play-by-play DataFrame.

    Returns:
        pd.DataFrame: DataFrame with expected pass probability for each situation.
    """
    grp = df.groupby(["season", "down", "ydstogo_bin", "yardline_bin"], dropna=False)["xpass"].mean().reset_index()
    grp = grp.rename(columns={"xpass": "exp_pass_prob"})
    return grp


def _team_week_offense(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates weekly offensive statistics for each team.

    This function groups play-by-play data by team and week, calculating totals and averages for key offensive metrics.

    Args:
        df (pd.DataFrame): The processed play-by-play DataFrame.

    Returns:
        pd.DataFrame: DataFrame with weekly offensive statistics for each team.
    """
    off = (
        df.groupby(["season", "week", "posteam"], dropna=False)
        .agg(
            plays=("epa", "size"),
            epa_per_play=("epa", "mean"),
            pass_plays=("is_pass", "sum"),
            rush_plays=("is_rush", "sum"),
            pass_epa=(
                "epa",
                lambda s: s[df.loc[s.index, "is_pass"]].mean() if (df.loc[s.index, "is_pass"]).any() else np.nan,
            ),
            rush_epa=(
                "epa",
                lambda s: s[df.loc[s.index, "is_rush"]].mean() if (df.loc[s.index, "is_rush"]).any() else np.nan,
            ),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off["pass_rate"] = off["pass_plays"] / off["plays"].replace(0.0, np.nan)
    return off


def _team_week_defense(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates weekly defensive statistics for each team.

    This function groups play-by-play data by defensive team and week, calculating totals and averages for key defensive metrics.

    Args:
        df (pd.DataFrame): The processed play-by-play DataFrame.

    Returns:
        pd.DataFrame: DataFrame with weekly defensive statistics for each team.
    """
    deff = (
        df.groupby(["season", "week", "defteam"], dropna=False)
        .agg(
            plays_def=("epa", "size"),
            epa_per_play_def=("epa", "mean"),
            pass_plays_def=("is_pass", "sum"),
            rush_plays_def=("is_rush", "sum"),
            pass_epa_def=(
                "epa",
                lambda s: s[df.loc[s.index, "is_pass"]].mean() if (df.loc[s.index, "is_pass"]).any() else np.nan,
            ),
            rush_epa_def=(
                "epa",
                lambda s: s[df.loc[s.index, "is_rush"]].mean() if (df.loc[s.index, "is_rush"]).any() else np.nan,
            ),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    deff["pass_rate_def"] = deff["pass_plays_def"] / deff["plays_def"].replace(0.0, np.nan)
    return deff


def _merge_expected_pass(off: pd.DataFrame, df: pd.DataFrame, exp_tbl: pd.DataFrame) -> pd.DataFrame:
    """Merges expected pass rates into team offense data and computes pass/rush over expected.

    This function joins expected pass probabilities to team offense stats and calculates pass over expected (proe) and rush over expected (rroe) metrics.

    Args:
        off (pd.DataFrame): Team offense statistics DataFrame.
        df (pd.DataFrame): Processed play-by-play DataFrame.
        exp_tbl (pd.DataFrame): Expected pass probability table.

    Returns:
        pd.DataFrame: Team offense DataFrame with expected pass rates and over-expected metrics.
    """
    plays = df[["season", "week", "posteam", "down", "ydstogo_bin", "yardline_bin", "is_pass"]].copy()
    plays = plays.merge(exp_tbl, on=["season", "down", "ydstogo_bin", "yardline_bin"], how="left")
    exp_team = plays.groupby(["season", "week", "posteam"], dropna=False)["exp_pass_prob"].mean().reset_index()
    exp_team = exp_team.rename(columns={"posteam": "team", "exp_pass_prob": "exp_pass_rate"})
    out = off.merge(exp_team, on=["season", "week", "team"], how="left")
    out["proe"] = out["pass_rate"] - out["exp_pass_rate"]
    out["rroe"] = (1.0 - out["pass_rate"]) - (1.0 - out["exp_pass_rate"])
    return out


def _cume_prior_to_week(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Calculates cumulative and rolling prior statistics for each team up to each week.

    This function computes expanding, rolling, and exponentially weighted means for specified columns, shifted to exclude the current week.

    Args:
        df (pd.DataFrame): DataFrame containing team-week statistics.
        value_cols (list[str]): List of column names to compute prior statistics for.

    Returns:
        pd.DataFrame: DataFrame with additional columns for cumulative, rolling, and exponentially weighted means.
    """
    df = df.sort_values(["season", "team", "week"])
    out = []
    for (season, team), g in df.groupby(["season", "team"], dropna=False):
        g = g.copy()
        for c in value_cols:
            g[f"cume_{c}"] = g[c].expanding().mean().shift(1)
            g[f"roll3_{c}"] = g[c].rolling(3, min_periods=1).mean().shift(1)
            g[f"ewm_{c}"] = g[c].ewm(alpha=0.3, adjust=False).mean().shift(1)
        out.append(g)
    return pd.concat(out, ignore_index=True)

def _auto_correlation_selection(df: pd.DataFrame, min_correlation_cutoff: float, max_features: int) -> pd.DataFrame:
    """Selects features based on their correlation with the target variables.

    This function computes the absolute correlation of each numeric feature with the home and away scores,
    selects features that meet the minimum correlation cutoff, and limits the number of features to the specified maximum.

    Args:
        df (pd.DataFrame): The input DataFrame containing game features and scores.
        min_correlation_cutoff (float): Minimum absolute correlation required to include a feature.
        max_features (int): Maximum number of features to select based on correlation.

    Returns:
        pd.DataFrame: A DataFrame containing only the selected features and target variables.
    """
    y_home = df["home_score"]
    y_away = df["away_score"]
    drop_cols = ["season", "week", "home_score", "away_score", "game_id", "home_team", "away_team", "game_type"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Filter out columns with zero variance to avoid division by zero in correlation calculation
    X = X.loc[:, X.std() > 0]

    corrs_home = X.apply(lambda x: x.corr(y_home)).abs()
    corrs_away = X.apply(lambda x: x.corr(y_away)).abs()
    corrs = pd.DataFrame({"feature": X.columns, "corr_home": corrs_home, "corr_away": corrs_away})
    corrs["max_corr"] = corrs[["corr_home", "corr_away"]].max(axis=1)
    selected_features = (
        corrs[corrs["max_corr"] >= min_correlation_cutoff]
        .sort_values("max_corr", ascending=False)
        .head(max_features)["feature"]
        .tolist()
    )

    selected_cols = selected_features + ["home_score", "away_score"]
    return df[selected_cols], corrs


def plot_feature_correlations(corrs: pd.DataFrame) -> None:
    """Plots feature correlations with the target variables.

    This function creates a bar plot for the absolute correlations of each feature with the home and away scores.

    Args:
        corrs (pd.DataFrame): DataFrame containing feature correlations.
    """
    plt.figure(figsize=(15, 6))
    plt.bar(corrs["feature"], corrs["max_corr"], color="skyblue")
    plt.axhline(y=0.35, color="r", linestyle="--", label="Min Correlation Cutoff")
    plt.title("Feature Correlations with Target Variables")
    plt.xlabel("Features")
    plt.ylabel("Absolute Correlation")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/feature_correlations.png")


def plot_feature_correlations_heatmap(corrs: pd.DataFrame) -> None:
    """Plots a heatmap of feature correlations.

    This function creates a heatmap to visualize the correlation matrix of the features.

    Args:
        corrs (pd.DataFrame): DataFrame containing feature correlations.
    """
    import seaborn as sns

    plt.figure(figsize=(14, 10))
    corr_matrix = corrs.set_index('feature')[['corr_home', 'corr_away']]
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("reports/feature_correlation_heatmap.png")


def build_game_features(min_plays_team_week: int = 20) -> pd.DataFrame:
    """Builds and saves engineered features for each NFL game.

    This function processes raw play-by-play and schedule data, computes team-level statistics, merges features for home and away teams, fills missing values, and writes the resulting feature set to disk.

    Args:
        min_plays_team_week (int): Minimum number of plays required for a team's weekly stats to be included.

    Returns:
        pd.DataFrame: DataFrame containing engineered features for each game.
    """
    pbp, sched = _load_raw()
    df = _prep_pbp(pbp)
    exp_tbl = _expected_pass_table(df)
    off = _team_week_offense(df)
    deff = _team_week_defense(df)

    off = off.loc[off["plays"] >= min_plays_team_week].copy()
    deff = deff.loc[deff["plays_def"] >= min_plays_team_week].copy()

    off = _merge_expected_pass(off, df, exp_tbl)
    # Columns to compute priors for
    off_cols = ["epa_per_play", "pass_epa", "rush_epa", "pass_rate", "proe", "rroe"]
    deff_cols = ["epa_per_play_def", "pass_epa_def", "rush_epa_def", "pass_rate_def"]
    off_prior = _cume_prior_to_week(off, off_cols)
    deff_prior = _cume_prior_to_week(deff, deff_cols)
    # rename columns to remove _def suffix for defense

    deff_prior = deff_prior.rename(columns={c: c.replace("_def", "") for c in deff_cols})


    sched_cols = ["season", "week", "home_team", "away_team", "home_score", "away_score", "game_type", "game_id"]
    sk = sched[sched_cols].copy()
    sk = sk[(sk["week"] >= 1) & (sk["week"] <= 18)]
    if "game_type" in sk.columns:
        sk = sk[sk["game_type"].fillna("REG").isin(["REG", "Regular"])].copy()

    home_off = off_prior.rename(
        columns={c: f"home_off_{c}" for c in off_prior.columns if c not in ["season", "week", "team"]}
    )
    home_off = home_off.rename(columns={"team": "home_team"})
    home_def = deff_prior.rename(
        columns={c: f"home_def_{c}" for c in deff_prior.columns if c not in ["season", "week", "team"]}
    )
    home_def = home_def.rename(columns={"team": "home_team"})
    away_off = off_prior.rename(
        columns={c: f"away_off_{c}" for c in off_prior.columns if c not in ["season", "week", "team"]}
    )
    away_off = away_off.rename(columns={"team": "away_team"})
    away_def = deff_prior.rename(
        columns={c: f"away_def_{c}" for c in deff_prior.columns if c not in ["season", "week", "team"]}
    )
    away_def = away_def.rename(columns={"team": "away_team"})

    gf = (
        sk.merge(home_off, on=["season", "week", "home_team"], how="left")
        .merge(home_def, on=["season", "week", "home_team"], how="left")
        .merge(away_off, on=["season", "week", "away_team"], how="left")
        .merge(away_def, on=["season", "week", "away_team"], how="left")
    )

    gf["home_field"] = 1.0

    # Fill NaNs with season means (neutral priors)
    out = []
    for season, g in gf.groupby("season"):
        num_cols = g.select_dtypes(include=[np.number]).columns.tolist()
        means = g[num_cols].mean(numeric_only=True,)
        g[num_cols] = g[num_cols].fillna(means)
        out.append(g)
    gf = pd.concat(out, ignore_index=True)
    slim_gf, corrs = _auto_correlation_selection(gf, min_correlation_cutoff=0.35, max_features=5)

    os.makedirs(FEAT_DIR, exist_ok=True)

    exp_tbl.to_csv(os.path.join(FEAT_DIR, "expected_pass_table.csv"), index=False)
    exp_tbl.to_parquet(os.path.join(FEAT_DIR, "expected_pass_table.parquet"))
    gf.to_parquet(os.path.join(FEAT_DIR, "game_features.parquet"))
    gf.to_csv(os.path.join(FEAT_DIR, "game_features.csv"), index=False)
    slim_gf.to_parquet(os.path.join(FEAT_DIR, "game_features_slim.parquet"))
    slim_gf.to_csv(os.path.join(FEAT_DIR, "game_features_slim.csv"), index=False)
    corrs.to_parquet(os.path.join(FEAT_DIR, "feature_correlations.parquet"))
    corrs.to_csv(os.path.join(FEAT_DIR, "feature_correlations.csv"), index=False)
    plot_feature_correlations(corrs)
    plot_feature_correlations_heatmap(corrs)

    print(f"[bold green]Wrote[/bold green] data/features/game_features.parquet with {len(gf):,} rows")
    return gf
