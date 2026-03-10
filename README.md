# 🏎️ Formula 1 Race Predictions

Predict Formula 1 race results using historical data from the **FastF1** library and **scikit-learn** machine learning models.

## What It Does

Pick any race (year + Grand Prix) and the notebook will:

1. **Collect training data** — all prior races that season + historical data back to 2018 via FastF1
2. **Engineer features** — grid position, driver form, team strength, tyre strategy, weather, circuit type
3. **Train & compare models** — Ridge, Random Forest, Gradient Boosting, Stacking (regression & classification)
4. **Predict** — finishing positions, podium finishes, and points finishes for your selected race

## Quick Start

```bash
pip install -r requirements.txt
jupyter lab f1_race_predictions.ipynb
```

Select a season and Grand Prix from the dropdowns, then run all cells.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data | FastF1 |
| ML | scikit-learn |
| Notebook | Jupyter |
| Viz | matplotlib, seaborn |

## Project Structure

```
├── f1_race_predictions.ipynb   # Main notebook
├── PRD.md                      # Product requirements
├── data/cache/                 # FastF1 cache
├── models/                     # Saved models
└── utils/                      # Helper modules
```

See [PRD.md](PRD.md) for full details.
