# Handoff: HF Fiscal Crisis Early-Warning Model (v111) — 2026-07-09

## Context
Continuation of a session that reviewed and fixed `Main_v110_Prod_RA_exploration_May12.py`
and produced `Main_v111_Prod_Jul7.py` (same folder). Read this file, then continue helping
the user analyze results and iterate. v110 is untouched reference code.

## What v111 changed vs v110 (all verified, full run completed 2026-07-08)
1. Data: `FILE_PATH` → `HF_data_monthly_2026-07-06_ffilled.csv` (109,005 rows, 1980-01→2026-07,
   195 countries; `precrisis` labeled 1980–2023, NOT ffilled past 2023). Paths anchored to
   script dir via `BASE_DIR` so cwd doesn't matter. Output → `Output_v111_NewData/`.
2. CV fix (major): new `PurgedPanelTimeSeriesSplit` (splits on unique dates, 12-calendar-month
   embargo, groups=row dates passed to every search). Replaces TimeSeriesSplit, which was
   splitting by country blocks (data sorted by Country,Date) with a 12-ROW gap (~1.8 days).
3. Early stopping disabled in all HistGB models (random internal 10% split leaked future);
   `max_iter` tuned instead (added `clf__max_iter` to MIDAS_GRID).
4. Winsorization optional: `Config.WINSORIZE=False` default; limits [0.01,0.99] when on.
5. Metrics: PR_AUC, Brier, Base_Rate added to results CSV; new OOS-only PR + calibration
   charts; ROC chart now OOS-only (was pooling in-sample+OOS).
6. CV search log: `Results_CV_Search_Log.csv` (Context, Model, CV_AUC, CV_AUC_Std, Best_Params,
   N_Train) — comparable across models within same Context.
7. Fixed NameError when RUN_HISTORICAL_OOS=False (diagnostics now guarded).

## Key results (full run, in Output_v111_NewData/)
- All-countries OOS (2014–2022) test AUC: HistGB ("XGBoost") 0.810, Midas-XGB 0.809,
  Monthly variants ~0.802–0.804, Logit family 0.793–0.798. PR-AUC 0.606 vs base rate 0.286.
  Calibration excellent (Brier 0.155 vs 0.204 climatology). CV AUC ≈ OOS AUC (0.799 vs 0.810)
  → no search overfitting. Hyperparams stable across all 11 refits: lr .01, depth 3, L2 50,
  min-leaf 200 (note: L2, min-leaf, and Almon θ1=-0.25 all at GRID EDGES — widen grids).
- Pooled model beats income-group-specific models everywhere (EM-only .794, LIC-only .667).
  AE-only: Logit (.950) beats HistGB (.897) but base rate 2.2%, PR-AUC only .12–.32 → fragile.
  LIC base rate is 46% → target is closer to "chronic stress" than early warning there.
- Monthly per-calendar-month models add nothing (flat monthly AUC; recommend dropping).
  MIDAS adds nothing: tuned Almon decay θ1=-0.25 puts ~93% weight on lags 1–2.
- SHAP (450 files, Explainability/): learned thresholds match economic intuition —
  public debt/GDP kink at ~60–70%, FX reserve cover matters up to ~3 months then flat,
  sovereign spread trigger ~300–400bp saturating ~800bp. Top feature: GDP per capita vs US.
  Sri Lanka 2022 caught truly OOS (risk .45→.88 from 2018); Lebanon jump only ~2020 (late).
- July 2026 top risks (Prod_Risk_XGBoost): Venezuela .83, Sudan .72, Mozambique .69,
  Senegal .68, Gabon .67, Argentina .66. Bottom: Norway/Switzerland/Denmark ~.01.
  Snapshot: Results_Latest_Forecast_Snapshot.csv. XGB-vs-Logit rank corr 0.88.

## Known remaining issues (user aware, not yet fixed)
- Full-sample winsorization/lag-median imputation stats computed before split (minor leakage);
  clean fix = move into pipeline.
- 'Monthly Midas-Logit' listed in MODELS_TO_RUN but has no training branch → NaN column.
- Grids should be widened one notch (l2 > 50, min_samples_leaf > 200, θ1 < -0.25).

## Environment
Run with `.venv/bin/python` in the project folder (has sklearn 1.7.2, shap, spyder-kernels
3.1.5 for Spyder 6.1.4). Full 8-model 4-setup run ≈ 1.5–2h on this M5 Pro.

## User profile
Comfortable with econometrics/ML (IMF-macro framing appreciated); wants line-number-specific
diagnostics, asks "why" a lot — explain mechanisms, not just conclusions.
