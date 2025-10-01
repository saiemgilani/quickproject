from __future__ import annotations

import os
from typing import List

import pandas as pd
from rich import print

RAW_DIR = os.path.join("data", "raw")


def _load_pbp_from_nflreadpy(seasons: List[int]) -> pd.DataFrame:
    """Loads NFL play-by-play data using nflreadpy for the given seasons.

    Attempts to find and use the appropriate nflreadpy loader functions for play-by-play data for the specified years.

    Args:
        seasons: List of NFL seasons (years) to load data for.

    Returns:
        pd.DataFrame: The play-by-play pandas DataFrames.

    Raises:
        AttributeError: If no suitable loader function is found in nflreadpy.
    """
    import nflreadpy as nfl

    pbp = None
    if hasattr(nfl, "load_pbp"):
        pbp = nfl.load_pbp(seasons)
    elif hasattr(nfl, "play_by_play"):
        pbp = nfl.play_by_play(seasons)
    elif hasattr(nfl, "load_pbp_seasons"):
        pbp = nfl.load_pbp_seasons(seasons)
    else:
        raise AttributeError("nflreadpy: could not find a play-by-play loader; please adjust ingest.py")

    return pbp


def _load_sched_from_nflreadpy(seasons: List[int]) -> pd.DataFrame:
    """Loads NFL schedule data using nflreadpy for the given seasons.

    Attempts to find and use the appropriate nflreadpy loader functions for schedule data for the specified years.

    Args:
        seasons: List of NFL seasons (years) to load data for.

    Returns:
        pd.DataFrame: The schedule pandas DataFrame.

    Raises:
        AttributeError: If no suitable loader function is found in nflreadpy.
    """
    import nflreadpy as nfl

    sched = None
    if hasattr(nfl, "load_schedules"):
        sched = nfl.load_schedules(seasons)
    elif hasattr(nfl, "schedules"):
        sched = nfl.schedules(seasons)
    elif hasattr(nfl, "load_schedule_seasons"):
        sched = nfl.load_schedule_seasons(seasons)
    else:
        raise AttributeError("nflreadpy: could not find a schedule loader; please adjust ingest.py")

    return sched


def _load_pbp_from_nfl_data_py(seasons: List[int]) -> pd.DataFrame:
    """Loads NFL play-by-play data using nfl_data_py for the given seasons.

    Retrieves play-by-play data for the specified years and returns it as a pandas DataFrame.

    Args:
        seasons: List of NFL seasons (years) to load data for.

    Returns:
        pd.DataFrame: The play-by-play pandas DataFrames.
    """
    import nfl_data_py as ndp

    pbp = None
    if hasattr(ndp, "import_pbp_data"):
        pbp = ndp.import_pbp_data(years=seasons, downcast=False, cache=True)
    else:
        raise AttributeError("nfl_data_py: could not find import_pbp_data; please adjust ingest.py")

    return pbp


def _load_sched_from_nfl_data_py(seasons: List[int]) -> pd.DataFrame:
    """Loads NFL schedule data using nfl_data_py for the given seasons.

    Retrieves schedule data for the specified years and returns it as a pandas DataFrame.

    Args:
        seasons: List of NFL seasons (years) to load data for.
    Returns:
        pd.DataFrame: The schedule pandas DataFrame.
    """
    import nfl_data_py as ndp

    sched = None
    if hasattr(ndp, "import_schedules"):
        sched = ndp.import_schedules(seasons, cache=True)
    else:
        raise AttributeError("nfl_data_py: could not find import_schedules; please adjust ingest.py")

    return sched


def ingest(seasons: List[int]) -> None:
    """Ingests NFL play-by-play and schedule data for the specified seasons.

    Attempts to load data using nflreadpy, falling back to nfl_data_py if necessary, and saves the results as parquet files.

    Args:
        seasons: List of NFL seasons (years) to ingest data for.

    Returns:
        None
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    try:
        pbp = _load_pbp_from_nflreadpy(seasons)
        sched = _load_sched_from_nflreadpy(seasons)
        print("[green]Loaded data via nflreadpy[/green]")
    except Exception as e:
        print(f"[yellow]nflreadpy failed ({e}); falling back to nfl_data_py[/yellow]")
        pbp = _load_pbp_from_nfl_data_py(seasons)
        sched = _load_sched_from_nfl_data_py(seasons)
        print("[green]Loaded data via nfl_data_py[/green]")

    pbp.write_parquet(os.path.join(RAW_DIR, "pbp.parquet"))
    sched.write_parquet(os.path.join(RAW_DIR, "schedules.parquet"))
    print("[bold]Saved:[/bold] data/raw/pbp.parquet, data/raw/schedules.parquet")


if __name__ == "__main__":
    ingest([2020, 2021, 2022, 2023, 2024])
