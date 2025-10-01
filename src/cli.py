from __future__ import annotations

from typing import List

import typer
from rich import print

from .evaluate import make_report
from .features import build_game_features
from .ingest import ingest as ingest_fn
from .model import run as run_models

app = typer.Typer(add_completion=False, help="NFL game outcomes pipeline CLI")


@app.command()
def ingest(
    seasons: List[int] = typer.Argument([2020, 2021, 2022, 2023, 2024], help="Seasons, e.g. 2020 2021 2022 2023 2024")
):
    """Download/cached PBP and schedules for the seasons."""
    ingest_fn(seasons)


@app.command("build-features")
def build_features():
    """Builds and saves engineered features for all NFL games.

    This command processes raw data and generates feature sets required for model training and prediction.
    """
    build_game_features()


@app.command()
def predict(poisson: bool = True):
    """Generates predictions for NFL games using the trained model.

    This command runs the prediction pipeline and outputs the number of games predicted.

    Args:
        poisson (bool): If True, uses Poisson regression objective; otherwise uses squared error regression.
    """
    preds, _ = run_models(poisson=poisson)
    print(f"[green]Predicted {len(preds)} games[/green]")


@app.command()
def evaluate():
    """Evaluates the performance of the trained model and generates a report."""
    make_report()


@app.command("build-all")
def build_all(poisson: bool = True):
    """Runs the full pipeline to ingest data, build features, train, predict, and generate a report.

    This command executes all steps in the NFL game outcomes pipeline for the specified seasons.

    Args:
        poisson (bool): If True, uses Poisson regression objective; otherwise uses squared error regression.
    """
    ingest_fn([2020, 2021, 2022, 2023, 2024])
    build_game_features()
    run_models(poisson=poisson)
    make_report()


if __name__ == "__main__":
    app()
