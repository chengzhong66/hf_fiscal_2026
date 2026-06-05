# HF Fiscal 2026

This repository contains an Python research project for sovereign/fiscal crisis forecasting.

## Current State

- Repository: `chengzhong66/hf_fiscal_2026`
- Local working path:
- Git branch: `main`
- Remote: `https://github.com/chengzhong66/hf_fiscal_2026.git`

## High-Level Purpose

The project appears to forecast fiscal or sovereign crisis risk at a 12-month horizon using monthly country-panel data. The core target is `precrisis`, shifted forward into `Target_H`.

The model family evolves around:

- historical walk-forward validation;
- global versus month-specific models;
- standard features versus MIDAS/Almon lag-combined features;
- XGBoost-like gradient boosting via `HistGradientBoostingClassifier`;
- logistic regression baselines;
- SHAP-based explainability;
- country, income-group, and event-study diagnostics;
- production-style latest-date forecasts.

Despite filenames containing `XGBoost`, the implementation currently uses scikit-learn's `HistGradientBoostingClassifier`, not the external `xgboost` package.

## Data Inputs

The latest scripts expect these files in the repository root:

- `fiscal_data_HF_monthly_2025-09-15.csv`: main country-month modeling dataset. This is not currently tracked in the repo.
- `Mapping.csv`: tracked country metadata file, keyed by `IFS`, with country names, income groups, areas, currency regime flags, and reserve-policy fields.
- `Mapping.xlsx`: spreadsheet version of the mapping file.

The main data-prep flow in `Main_v110_Prod_RA_exploration_May12.py`:

1. loads the fiscal CSV;
2. normalizes country-code columns;
3. merges `Mapping.csv` on `Country == IFS`;
4. builds a monthly `Date` from `year` and `month`;
5. engineers additional variables such as `oil_price_yoy`, `oil_shock_impact`, and `debt_fx_vulnerability`;
6. shifts `precrisis` forward by `HORIZON = 12` months;
7. winsorizes predictors;
8. creates 12 monthly lags for each predictor;
9. returns both the full production frame and the historical frame with observed targets.

## Main Predictors

The latest version groups predictors into three conceptual families:

- Macro: inflation, GDP growth, current account/GDP, reserve cover, terms of trade, GDP per capita relative to the US, oil exports/GDP, oil shock impact.
- Fiscal: public debt/GDP, fiscal balance/GDP, fiscal revenue/GDP, total external debt/GDP, debt service/GDP, corruption.
- Financial: short-term rate, long-term bond yield, FX depreciation, sovereign spread, oil price, VIX.

Each predictor also has a monotonicity sign used by constrained gradient boosting where applicable: `1` for risk-increasing, `-1` for risk-reducing, and `0` for unconstrained.

## Version Evolution


### Phase 1: Early Monthly Crisis Engine

Representative files:

- `Main_v3_Feb17.py`
- `Main_v5.7_Feb17.py`
- `Main_v6_Feb18.py`

Likely focus:

- basic sovereign crisis forecasting;
- global versus month-specific model modes;
- 12-month forecast horizon;
- temporal cross-validation;
- early SHAP support;
- audit and plotting utilities.

The project starts as a relatively compact engine and quickly adds data auditing, output folders, raw-data visualizations, monotonic constraints, and optional SHAP computation.

### Phase 2: Benchmark and Linear Model Expansion

Representative files:

- `Main_BenchmarkModels_v1_Feb18.py`
- `Main_BenchmarkModels_v2_Feb18.py`
- `Main_v19_Feb18.py`
- `Main_v20_Feb18.py`

Likely focus:

- comparing XGBoost-like tree models against OLS/logit/lasso-style baselines;
- adding MIDAS logit variants;
- creating coefficient and beta heatmaps;
- comparing model risk by income group and region.

This phase looks like the author was broadening the model tournament before committing to one family.

### Phase 3: MIDAS-XGBoost Competition

Representative files:

- `Main_v22_MIDAS-XGBoost_Feb18.py`
- `Main_v23_MIDAS-XGBoost_Competition_Feb18.py`
- `Main_v29_*` through `Main_v39_*`

Likely focus:

- implementing exponential Almon lag weights;
- searching MIDAS lag-shape parameters;
- transferring a global lag shape to monthly models;
- comparing standard, monthly, MIDAS, and monthly MIDAS variants.

Important note: `Main_v39_MIDAS-XGBoost_Competition_Feb19.py` currently has a Python parse error near line 130, so it should be treated as an experimental or broken checkpoint.

### Phase 4: Diagnostics, Event Studies, and Visualization

Representative files:

- `Main_v41_MIDAS-XGBoost_Feb19.py` through `Main_v62_MIDAS-XGBoost_Feb24.py`

Likely focus:

- richer AUC comparisons;
- monthly AUC and volatility tracking;
- calibration and precision/recall diagnostics;
- country profile charts;
- stress-episode zoom charts;
- SHAP dependence plots;
- event studies;
- regularization experiments for linear models.

Notable design shifts:

- v46 notes a move from z-score event studies to base-100 indexing at event time.
- v58 notes heavier linear-model regularization and explicit Almon weight plots.
- v61/v62 add clearer walk-forward and leakage-prevention controls.

### Phase 5: Production Pipeline

Representative files:

- `Main_v100_Prod_Feb26.py`
- `Main_v101_Prod_Feb26.py`
- `Main_v102_Prod_Feb27.py`
- `Main_v103_Prod_Feb27.py`
- `Main_v104_Prod_Feb27.py`
- `Main_v105_Prod_Feb27.py`
- `Main_v106_Prod_Feb27.py`
- `Main_v107_Prod_Mar3.py`
- `Main_v108_Prod_Mar3.py`
- `Main_v109_Prod_May11.py`

Likely focus:

- turning the research workflow into a production-style forecast engine;
- training historical out-of-sample models;
- training final models on all available historical target data;
- writing latest forecast snapshots;
- producing production profile charts;
- splitting chart output into model families.

By v104 and later, the script uses explicit pipeline toggles:

- historical out-of-sample analysis;
- production forecasting and explainability;
- event-study analysis.

### Phase 6: RA Exploration and Robust Scoring

Representative file:

- `Main_v110_Prod_RA_exploration_May12.py`

Likely focus:

- research-assistant exploration on top of the production engine;
- expanded stress episodes;
- income-group comparisons;
- safer AUC scoring for folds with only one class.

The most visible code change is `safe_auc` and `safe_auc_scorer`, which return a neutral `0.5` when a validation fold has only crisis or only non-crisis observations. This suggests the author ran into sparse-class problems when filtering by country group, episode, or time window.

## Latest Script Map

`Main_v110_Prod_RA_exploration_May12.py` is the most recent file by filename. Its structure is:

- `safe_auc`: neutral fallback scorer for one-class validation folds.
- `Config`: all runtime choices, model lists, chart families, event-study settings, grids, colors, variable metadata, and file paths.
- `AlmonValueCombiner`: transforms 12 lags of each predictor into one MIDAS-style weighted value.
- Plotting helpers: AUC comparisons, Almon weights, ROC curves, risk distributions, income comparisons, country profiles, production profiles, SHAP charts, decomposition charts, zoom episodes, and event study charts.
- `prepare_data`: load, merge, clean, engineer, lag, and return modeling data.
- `run_engine`: orchestrates historical OOS, diagnostics, production forecasting, SHAP explainability, and event studies.

## Modeling Workflow

The current production-family scripts generally follow this sequence:

1. Load and prepare the panel data.
2. Build a 12-month-ahead target.
3. Select predictors present in the data.
4. Train models using temporal cross-validation.
5. Run walk-forward historical out-of-sample evaluation.
6. Compare models by AUC overall, by month, and by income group.
7. Train production models using all historical labeled observations.
8. Predict latest/future risk values.
9. Export forecast snapshots and charts.
10. Generate SHAP explanations and decomposition charts where dependencies are available.

## Environment Setup

Use a project-local virtual environment. The default `python3` on this Mac currently points to Python 3.14, which is newer than the safest target for this scientific stack. Prefer Python 3.12 through `pyenv`.

From the repository root:

```bash
cd /Users/cz/base/40_code/projects/hf_fiscal_2026
python3.12 -m venv .venv
source .venv/bin/activate
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

To check the environment:

```bash
python - <<'PY'
import pandas, numpy, scipy, sklearn, matplotlib, seaborn, statsmodels
print("core scientific stack imports OK")
PY
```

`shap` is optional in the scripts. If it fails to install or import, the model can still run, but SHAP plots will be skipped.

## Running

After the missing main data file is available in the repo root:

```bash
source .venv/bin/activate
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
python Main_v110_Prod_RA_exploration_May12.py
```

Expected output goes into the script's configured `OUTPUT_ROOT`, currently:

```text
Output_v109_ExpandedEpisodes
```

Note that v110 still uses the v109 output folder name. That may be intentional for continuity or simply a leftover from version copying.

## Known Issues and Open Questions

- The main data file `fiscal_data_HF_monthly_2025-09-15.csv` is missing from the repo.
- There is no README from the original author, so the evolution notes above are inferred from filenames, headers, comments, and code structure.
- The repo has only one git commit, so detailed version history is embedded in copied files rather than git history.
- `Main_v39_MIDAS-XGBoost_Competition_Feb19.py` has a syntax error and cannot be parsed by Python.
- Several scripts produce Python warnings about invalid escape sequences in plot labels such as `\D`; these are usually easy to fix later by using raw strings or escaping backslashes.
- The current root directory contains many script versions. This is useful for understanding evolution now, but later it may be worth adding an `archive/` directory once the lineage is documented.

## Suggested Next Steps

1. Locate the missing fiscal data CSV and decide whether it should be tracked, ignored, or stored outside git.
2. Install the project environment in `.venv`.
3. Run a lightweight import/syntax check across scripts.
4. Run the latest script once the data file is available.
5. Add a short `NOTES.md` or `docs/evolution.md` if deeper version-by-version interpretation becomes useful.
6. Later, choose a canonical script and refactor only after the modeling lineage is understood.
