from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TEAM_COL = "team"
OPP_COL = "opponent"
SEASON_COL = "season"
WEEK_COL = "week"

VALID_PLAY_TYPES = {"pass", "run", "rush"}


def bin_ydstogo(x: float) -> str:
    """Bins the yards to go value into descriptive categories.

    Returns a string label representing the range of yards to go for a given value.


    Args:
        x: The yards to go as a float.

    Returns:
        A string representing the binned category ('unk', 'short', 'med', 'long', 'xl', or 'xlplus').

        short - 0-3 yards
        med - 4-7 yards
        long - 8-10 yards
        xl - 11-15 yards
        xlplus - 16+ yards
    """
    if pd.isna(x):
        return "unk"
    if x <= 3:
        return "short"
    if x <= 7:
        return "med"
    if x <= 10:
        return "long"
    if x <= 15:
        return "xl"
    return "xlplus"


def bin_yardline_100(x: float) -> str:
    """Calculates the mean of a pandas Series, ignoring NaN values.

    Returns NaN if the Series is empty; otherwise, returns the mean as a float.

    Args:
        x: A pandas Series of numeric values.

    Returns:
        The mean of the Series as a float, or NaN if the Series is empty.
        red - 1-20
        front - 21-40
        mid - 41-60
        back - 61-80
        deep - 81-100
    """
    if pd.isna(x):
        return "unk"
    if x <= 20:
        return "red"
    if x <= 40:
        return "front"
    if x <= 60:
        return "mid"
    if x <= 80:
        return "back"
    return "deep"


def safe_mean(x: pd.Series) -> float:
    """Calculates the mean of a pandas Series, ignoring NaN values.

    Returns NaN if the Series is empty; otherwise, returns the mean as a float.

    Args:
        x: A pandas Series of numeric values.

    Returns:
        The mean of the Series as a float, or NaN if the Series is empty.
    """
    return float(np.nanmean(x.values)) if len(x) else np.nan


def safe_div(a: float, b: float) -> float:
    """Performs safe division of two numbers, returning 0.0 if the denominator is zero or invalid.
    Args:
        a: The numerator as a float.
        b: The denominator as a float.
    Returns:
        The result of the division as a float, or 0.0 if the denominator is zero or invalid.
    """
    return float(a) / float(b) if b not in (0, 0.0, None, np.nan) else 0.0


def fix_team_codes(df: pd.DataFrame, team_cols: List[str]) -> pd.DataFrame:
    """Normalize a few known team code variants if present.
    Args:
        df: The input DataFrame containing team codes.
        team_cols: A list of column names in the DataFrame that contain team codes to be normalized.
    Returns:
        The DataFrame with normalized team codes.
    """
    # Add mappings if needed (e.g., LA -> LAR), left minimal here.
    return df
