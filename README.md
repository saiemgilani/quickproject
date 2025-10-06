# NFL Game Outcome Pipeline (2020–2024)

Predict NFL game outcomes (score margin, team points, and win probability) for regular-season weeks using publicly available play-by-play with **nflreadpy** (fallback to `nfl_data_py` if needed).

**Key design goals**
- Cold start *within-season*: Week 1 uses neutral priors; no prior seasons used.
- Pre-game predictions at the individual game level (Weeks 1–18) for seasons **2020–2024**.
- Feature engineering from play-by-play → team-week aggregates with **opponent adjustments**, **home field**, **situational efficiency** (down/distance/field position), **rolling** and **EWMA** form.
- Separate models for **home points** and **away points**; score margin is the difference; **win probability** from logistic regression model as well as via Skellam using the two predicted means.
- Reproducible pipeline with **batch**, **Makefile**, **Docker**, and CLI (`typer`) commands.

> This repo favors clarity and reproducibility over leaderboard-chasing.

---

## Quickstart

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# End-to-end run (downloads data, builds features, trains, predicts, evaluates, and writes a markdown report)
make all
# or
python -m src.cli build-all

# or using bash commands
 bash build_models.sh

```

Data artifacts land in:
- `data/raw/` – schedule and pbp parquet files
- `data/features/` – per-game feature table (one row per game)
- `data/predictions/` – per-week predictions and evaluation CSVs
- `reports/` – markdown + CSV summaries

---

## Approach (high level)

1. **Ingest** 2020–2024 PBP + schedules using `nflreadpy` (falls back to `nfl_data_py`).
2. **Clean & label plays** → pass/rush, valid plays (filter penalties, spikes, kneels).
3. **Situational table**: league‑wide expected pass rate by (down, distance bin, yardline bin) ⇒ compute **PROE/RROE** per team‑week.
4. **Team‑week aggregates**: EPA/play (pass, rush, overall).
5. **Opponent adjustments** (within-season, through week t‑1): adjust team offense/defense by averaging opponent strengths and recent form (rolling/EWMA).
6. **Game‑level features** (pre‑match): combine home offense vs away defense (and vice versa), diffs, home field
7. **Models**:
   - **Home points** (Poisson GBM/XGB)
   - **Away points** (Poisson GBM/XGB)
   - Margin = difference of means;
   - **Win Probability** (Logistic Regression GBM/XGB) as well as via **Skellam**.
8. **Walk‑forward evaluation** (per season): only train using data from the same season **before** the prediction week.
9. **Metrics**: MAE/RMSE for points & margin; Brier/log-loss + calibration for win prob.
10. **Report**: markdown summary with assumptions, limitations, and results.

### Cold start
Week 1 uses neutral priors: league-average points baseline (constant = 22 points) and league average team ratings. This adheres to “no prior season data” while remaining stable.

---

## Repo layout

```
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                # cached pbp/schedule
│   ├── features/           # game-level features
│   ├── models/             # saved models
│   └── predictions/        # per-week preds & eval
├── reports/
├── src/
│   ├── cli.py              # typer CLI
│   ├── ingest.py           # data download + caching
│   ├── features.py         # feature engineering
│   ├── model.py            # training + inference
│   ├── evaluate.py         # metrics & calibration
│   └── utils/
│       └── common.py       # helpers & constants
│── build_models.sh         # bash orchestration
├── Makefile                # make orchestration
├── Dockerfile              # docker image
├── requirements.txt
└── README.md               # documentation
```

---

## Assumptions & Limitations

- **Public data only**; no proprietary DVOA. We approximate schedule‑adjusted efficiency with per‑play EPA and team‑week opponent‑aware adjustments.
- **No prior seasons for training**; each season is learned on the fly (neutral Week 1 baseline).
- **In‑game factors** (down/distance/field position) are incorporated into team **situational efficiency features**, not live WP.
- Travel/rest/injuries/trades are **optional** future work hooks.
- Model choices favor interpretability and speed; you can swap in LightGBM/CatBoost/NGBoost easily.
- No hyperparameter tuning; fixed XGBoost params.
- No ensembling or stacking; simple per‑season+week Poisson models.
- Feature engineering is basic and not optimized for performance.
- Model evaluation is walk‑forward within each season; no cross‑validation. Likely to be overfitting given small data size per season/week.

---

## EDA & Feature Engineering

![Feature Correlations (Absolute)](reports/feature_correlations.png)

- **Target correlations**: Home points and away points are most correlated with their respective team offensive EPA/play metrics (pass, rush, overall)
- **Feature correlations**: Many features are correlated (e.g., team offensive stats with opponent defensive stats). The model should be able to handle this, but it may affect interpretability.

![Feature Correlations Heatmap](reports/feature_correlation_heatmap.png)

- **Feature importance**: The most important features for predicting home points are home team offensive EPA/play (overall, pass, rush), away team defensive EPA/play (overall, pass, rush). Similar patterns hold for away points.
- **Model interpretability Consideration**: The use of tree-based models like XGBoost allows for some level of interpretability, but care should be taken when interpreting feature importances, especially in the presence of correlated features.

---

## Results
See `reports/PERFORMANCE.md` for the latest model performance summary.

### Model Performance Summary by Season

| Season | MAE Home Margin | MAE Home | MAE Away | RMSE Home | RMSE Away | RMSE Margin | MAE Margin | Brier Score WP | LogLoss WP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 6.80 | 5.72 | 6.18 | 54.22 | 58.31 | 83.17 | 6.80 | 0.065 | 0.228 |
| 2021 | 6.47 | 6.08 | 5.44 | 57.37 | 45.43 | 76.79 | 6.47 | 0.048 | 0.185 |
| 2022 | 4.95 | 4.75 | 5.11 | 40.28 | 40.63 | 40.56 | 4.95 | 0.078 | 0.266 |
| 2023 | 6.51 | 6.10 | 5.07 | 66.92 | 41.45 | 77.13 | 6.51 | 0.097 | 0.310 |
| 2024 | 6.40 | 5.88 | 5.15 | 54.57 | 39.97 | 67.95 | 6.40 | 0.077 | 0.259 |

### Feature Importances

![Feature importances for Poisson GBM Points models](reports/feature_importances_poisson.png)
![Feature importances for Poisson GBM Points models (Including Prior Seasons)](reports/feature_importances_poisson_prior_seasons.png)
![Feature importances for Baseline RMSE XGB Points Models](reports/feature_importances_baseline_rmse.png)

### Shapley Values

![SHAP values for Home Points model](reports/shap_summary_poisson_home_2024.png)

![SHAP values for Away Points model](reports/shap_summary_poisson_away_2024.png)

![SHAP values for Win Probability model](reports/shap_summary_logistic_wp_wp_2024.png)



### Win Probability Calibration Plot

![Calibration plot for win probability](reports/calibration_plot.png)

---

## How to extend

- Add roster/injury signals (e.g., QB status, OL continuity), weather, travel rest.
- Replace Poisson with **bivariate Poisson** to model score correlation directly.
- Swap opponent‑adjustment block with weekly **ridge APM** (off/def team effects).
