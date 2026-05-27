# TARDIS — Predicting the Unpredictable

SNCF TGV delay analysis and prediction platform built with scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow, and Streamlit.

## Project structure

```
Tardis/
├── cleaned_dataset.csv       Cleaned dataset (output of EDA notebook)
├── model.joblib              Trained model pipeline (output of model notebook)
├── route_stats.csv           Route-level median stats (output of model notebook)
├── requirements.txt
├── tardis_eda.ipynb          Data cleaning, exploration & feature engineering
├── tardis_model.ipynb        Model training orchestrator (calls scripts/model/)
├── tardis_dashboard.py       Streamlit dashboard
└── scripts/
    ├── cleaning/             Data cleaning modules
    ├── visualization/        EDA plot modules
    ├── merging/              Dataset merging utilities
    ├── audit/                Data quality tracking
    └── model/
        ├── config.py         Constants (target, features, season map)
        ├── preprocessing.py  Preprocessor pipeline + route stats builder
        ├── sklearn_models.py sklearn / XGBoost / LightGBM training & tuning
        ├── dl_models.py      PyTorch & TensorFlow MLP models
        ├── evaluation.py     Metrics and comparison utilities
        ├── predict.py        predict_delay(dep, arr, date, model, stats)
        └── train.py          Full training orchestrator (also CLI-runnable)
```

## Setup

```bash
git clone <repo-url>
cd Tardis

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

### 1. Data cleaning & EDA
Open and run `tardis_eda.ipynb` from top to bottom.
Produces `cleaned_dataset.csv`.

### 2. Model training
Open and run `tardis_model.ipynb` from top to bottom.
Produces `model.joblib` and `route_stats.csv`.

Or train from the command line:
```bash
python -m scripts.model.train cleaned_dataset.csv
```

### 3. Dashboard
```bash
streamlit run tardis_dashboard.py
```

The dashboard opens at `http://localhost:8501` with three tabs:

| Tab | Content |
|-----|---------|
| 🎯 Predict a delay | Select two stations + date → predicted delay + historical distribution + seasonal comparison |
| 📊 Explore data | KPIs, delay distribution, monthly trend, station ranking, year×season heatmap, correlation matrix, CSV export |
| 🔍 Model insights | Active model info, feature list, feature importance chart, prediction explanation |

## Models trained

| Framework | Model |
|-----------|-------|
| scikit-learn | Linear Regression, Ridge, Random Forest, Gradient Boosting |
| XGBoost | XGBRegressor |
| LightGBM | LGBMRegressor |
| PyTorch | MLP — 256 → 128 → 64 → 1 (BatchNorm + Dropout) |
| TensorFlow | MLP — 256 → 128 → 64 → 1 (BatchNorm + Dropout + EarlyStopping) |

The dashboard always uses the **best sklearn/boosting model after GridSearchCV tuning** (serialised with joblib). DL models are compared in the notebook but not deployed.

## Dataset format

The pipeline expects a CSV named `dataset.csv` (or `cleaned_dataset.csv` post-EDA) with at minimum:

- `Date` — ISO date string
- `Departure station`, `Arrival station` — station names
- `Average delay of all trains at arrival` — prediction target (minutes)
- See `tardis_eda.ipynb` for the full column list and cleaning steps
