# Predict the result of NFL 2020 - 2024
Predict the result of NFL 2020 - 2024
publicly available models to predict NFL game outcomes, including score margin, team points, and win probability.
- Data source: [nflreadpy]
- GOAL: Individual game predictions for each week - regular season
- Features: Team stats, opponent adjustments, situational factors
- Seasons: 2020 to 2024 (Weeks 1-18)
- Cold start: No prior season data used for predictions
- Modeling team offense vs defense matchups
- Adjusted for home field advantage
- In game situational factors: down, distance, field position
- Strength of schedule adjustments for team offense and defense metrics
- Team Offense passing/rushing adjusted EPA/play(pass/rush), adjusting for schedule RROE/PPOE, turnovers
- Team Defense passing/rushing adjusted EPA/play(pass/rush), adjusting for schedule RROE/PPOE, turnovers
- Play-by-play data aggregation to team-level stats
- Opponent adjustments based on league averages
- Rolling averages and exponential moving averages for recent performance
- Incorporate advanced statistics (e.g., DVOA, EPA)
- Further down the line, include roster adjustments (injuries, trades, etc.)
- Model performance evaluation and iteration

Take home assessment:
- Predict score margin, team points for each game
- Not necessarily looking for the most accurate model, but a well-reasoned approach
- Focus on data cleaning, feature engineering
- Easy to follow code and documentation/project structure/repository
- Orchestration of data pipeline - up to you but consider:
    - Bash/Airflow/DVC/Makefile/Docker
- Reproducibility of results
- Plus points for evaluation and visualization of model performance but not emphasized
- Markdown documentation of approach, assumptions, limitations as well as RESULTS
