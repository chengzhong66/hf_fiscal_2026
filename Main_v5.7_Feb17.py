# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 22:42:17 2026
@author: cmarsilli
Sovereign Crisis Forecasting Engine v6.0
Pipeline: Audit -> Estimation -> Visual Suite -> SHAP
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False

# =============================================================================
# 1. CONFIGURATION & MASTER METADATA
# =============================================================================

class Config:
    OUTPUT_NAME = "Output_v6_Global"
    
    # --- USER TOGGLES ---
    USE_MONOTONIC_CONSTRAINTS = True  # Toggle for economic logic constraints
    USE_NATIVE_IMPUTATION = True      # If True, lets HistGradientBoosting handle NaNs
    RUN_RAW_DATA_STACK = False         # Creates the "Long Picture" charts
    COMPUTE_SHAP = False
    COMPUTE_SHAP_INTERACTIONS = False
    INTERACTION_TOP_N = 5
    
    # --- MODEL SETTINGS ---
    HORIZON = 12
    MODEL_MODE = "Global"     # "Month-Specific"  
    CV_TYPE = "Temporal"
    TEST_SHARE = 0.15
    CV_FOLDS = 5
    
    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    TARGET = "precrisis"
    
    PARAM_GRID = {
        'learning_rate': [0.01, 0.03, 0.05],
        'max_iter': [100, 300, 500],
        'max_depth': [3, 4, 5],
        'l2_regularization': [0.1, 1.0, 5.0, 10.0, 20.0],
        'min_samples_leaf': [20, 50, 100],
        'max_leaf_nodes': [15, 31],
    }

    VARS = {
        "oil_price": ("Oil Price", "Global", 0),
        "VIX": ("VIX Index", "Global", 1),
      
        "debt_fx_vulnerability": ("Debt x FX Shock", "Interaction", 1), 
       
        "PCPI_PCH": ("Inflation", "Macro", 1),
        "gdp_growth": ("GDP Growth", "Macro", -1),
        "BoP_gdp": ("Current Account/GDP", "Macro", -1),
        "reserve_cover": ("FX Reserve Cover", "Macro", -1),
        "terms_of_trade": ("Terms of Trade", "Macro", -1),
        "GDP_percapita_over_US_12ma": ("GDP per Capita", "Macro", -1), 
        "oil_to_gdp": ("OilExports/GDP", "Macro", -1), 
        "oil_shock_impact": ("Oil Shock Impact", "Interaction", -1), 
       
        "govt_debt_gdp": ("Public Debt/GDP", "Fiscal", 1),
        "govt_deficit_gdp": ("Fiscal Deficit/GDP", "Fiscal", 1),
        "govt_revenue_gdp": ("Fiscal Revenue/GDP", "Fiscal", -1),
        "tot_ext_debt_gdp": ("Total Ext Debt/GDP", "Fiscal", 1),
        "debt_service_gdp": ("Debt Service/GDP", "Fiscal", 1),
        "corruption_12ma": ("Corruption", "Fiscal", -1),
       
        "deposit_rate": ("ST Rate", "Financial Conditions", 1),
        "long_term_bond_yield": ("LT Rate", "Financial Conditions", 1),
        "WUI": ("Uncertainty Idx", "Financial Conditions", 1),
        "ENDE_yoy": ("FX Depreciation", "Financial Conditions", 1),
        "spread": ("Sovereign Spread", "Financial Conditions", 1)
    }

    @classmethod
    def get_meta(cls, var):
        return cls.VARS.get(var, (var, "Macroeconomic", 0))

# =============================================================================
# 2. UTILITIES & DATA AUDIT
# =============================================================================

def setup_folders():
    subdirs = ["data", "charts/audit", "charts/performance", "country_charts", "country_raw_stacks",
               "shap/dependence", "shap/seasonal"]
    for d in subdirs: os.makedirs(os.path.join(Config.OUTPUT_NAME, d), exist_ok=True)

def prepare_data():
    df = pd.read_csv(Config.FILE_PATH)
    if 'COUNTRY_name' in df.columns: df.rename(columns={'COUNTRY_name': 'Country'}, inplace=True)
    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    df = df.sort_values(['Country', 'Date'])

    # Engineering Interactions
    df['oil_price_pch'] = df.groupby('Country')['oil_price'].pct_change()
    df['oil_shock_impact'] = df['oil_to_gdp'] * df['oil_price_pch']
    df['debt_fx_vulnerability'] = df['ENDE_yoy'] * df['tot_ext_debt_gdp']

    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)
    
    # Optional Manual Imputation
    if not Config.USE_NATIVE_IMPUTATION:
        df[predictors] = df.groupby('Country')[predictors].ffill().fillna(df[predictors].median())
    
    return df.dropna(subset=['Target_H']).copy(), predictors

def run_audit(df, predictors):
    print("\n[AUDIT] Assessing Data Quality...")
    path = os.path.join(Config.OUTPUT_NAME, "charts/audit")
    
    # 1. Missing Value Statistics
    missing_pct = df[predictors].isna().mean().sort_values(ascending=False)
    missing_pct.to_csv(os.path.join(Config.OUTPUT_NAME, "data/missing_audit.csv"))
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[predictors].isna(), cbar=False, cmap='binary')
    plt.title("Missing Data Patterns (Black = Missing)")
    plt.savefig(os.path.join(path, "missing_heatmap.png")); plt.close()

    # 2. Target Distribution
    plt.figure(figsize=(8, 5))
    df['Target_H'].value_counts(normalize=True).plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Class Balance (0: No Crisis, 1: Pre-Crisis)"); plt.savefig(os.path.join(path, "class_balance.png")); plt.close()

# =============================================================================
# 3. VISUAL SUITE (RAW STACKS)
# =============================================================================

def plot_raw_data_stack(df, predictors):
    if not Config.RUN_RAW_DATA_STACK: return
    print("\n[VISUAL] Generating Raw Data Stacks (Country-by-Country)...")
    path = os.path.join(Config.OUTPUT_NAME, "country_raw_stacks")
    
    # Sample a subset of indicators for the stack to keep it readable
    core_vars = predictors[:8] 
    
    for country, g in df.groupby('Country'):
        fig, axes = plt.subplots(len(core_vars), 1, figsize=(15, 2*len(core_vars)), sharex=True)
        plt.subplots_adjust(hspace=0.3)
        
        for i, var in enumerate(core_vars):
            axes[i].plot(g['Date'], g[var], color='navy', lw=1.5)
            # Shade crisis periods in grey
            axes[i].fill_between(g['Date'], g[var].min(), g[var].max(), where=g['Target_H']==1, 
                                 color='grey', alpha=0.3, label='Crisis Window')
            axes[i].set_ylabel(Config.get_meta(var)[0], rotation=0, labelpad=40, fontsize=8)
        
        plt.suptitle(f"Historical Data Profile: {country} (Grey = Pre-Crisis Horizon)", fontsize=14)
        plt.savefig(os.path.join(path, f"{country}_stack.png"), bbox_inches='tight'); plt.close()

# =============================================================================
# 4. ESTIMATION PIPELINE
# =============================================================================

def run_estimation(df, predictors):
    print(f"\n[ESTIMATION] Running Models (Constraints: {Config.USE_MONOTONIC_CONSTRAINTS})...")
    
    constraints = [Config.get_meta(p)[2] for p in predictors] if Config.USE_MONOTONIC_CONSTRAINTS else None
    modes = [(f"Month_{m}", df[df['month'] == m]) for m in range(1, 13)] if Config.MODEL_MODE == "Month-Specific" else [("Global", df)]
    
    reconstructed_list, perf_log, all_shap_values, all_X_data = [], [], [], []
    
    print("-" * 65)
    print(f"{'Model Mode':<15} | {'In-Sample AUC':<15} | {'OOS AUC':<15} | {'Samples':<10}")
    print("-" * 65)

    for mode_name, sub_df in modes:
        X, y = sub_df[predictors], sub_df['Target_H']
        split_idx = int(len(sub_df) * (1 - Config.TEST_SHARE))
        X_train, X_test = X.iloc[:split_idx], y.iloc[:split_idx] # Dummy X_test for logic
        
        # Proper splits
        X_tr, X_ts = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_ts = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if len(y_tr.unique()) < 2: continue

        search = RandomizedSearchCV(
            HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42),
            Config.PARAM_GRID, n_iter=10, cv=TimeSeriesSplit(n_splits=Config.CV_FOLDS), 
            scoring='roc_auc', n_jobs=-1, random_state=42
        )
        search.fit(X_tr, y_tr)
        best_clf = search.best_estimator_
        
        auc_in = roc_auc_score(y_tr, best_clf.predict_proba(X_tr)[:, 1])
        auc_out = roc_auc_score(y_ts, best_clf.predict_proba(X_ts)[:, 1])
        
        print(f"{mode_name:<15} | {auc_in:<15.4f} | {auc_out:<15.4f} | {len(sub_df):<10}")

        sub_df['Risk_Index'] = best_clf.predict_proba(X)[:, 1]
        reconstructed_list.append(sub_df)
        perf_log.append({'Mode': mode_name, 'In-Sample': auc_in, 'OOS': auc_out})

        if Config.COMPUTE_SHAP and SHAP_INSTALLED:
            explainer = shap.Explainer(best_clf.predict_proba, X)
            all_shap_values.append(explainer(X)[:, :, 1].values)
            all_X_data.append(X)

    full_df = pd.concat(reconstructed_list).sort_values(['Country', 'Date'])
    
    # Chart: AUC Comparison
    perf_df = pd.DataFrame(perf_log)
    plt.figure(figsize=(10, 5))
    plt.plot(perf_df['Mode'], perf_df['In-Sample'], marker='o', label='In-Sample AUC')
    plt.plot(perf_df['Mode'], perf_df['OOS'], marker='s', label='OOS AUC')
    plt.xticks(rotation=45); plt.legend(); plt.title("Predictive Performance by Model")
    plt.savefig(os.path.join(Config.OUTPUT_NAME, "charts/performance/auc_comparison.png")); plt.close()

    # Chart: All Country Specific Result Plots
    print("[VISUAL] Generating Country Forecast Charts...")
    for c, g in full_df.groupby('Country'):
        plt.figure(figsize=(10, 4))
        plt.plot(g['Date'], g['Risk_Index'], color='red', label='Forecast Risk')
        plt.fill_between(g['Date'], 0, 1, where=g['Target_H']==1, color='grey', alpha=0.2, label='Actual Crisis')
        plt.title(f"Forecast: {c}"); plt.legend(); plt.ylim(0, 1)
        plt.savefig(os.path.join(Config.OUTPUT_NAME, f"country_charts/{c}_forecast.png")); plt.close()

    return full_df, all_shap_values, all_X_data

# =============================================================================
# 5. EXECUTION
# =============================================================================

if __name__ == "__main__":
    setup_folders()
    data, preds = prepare_data()
    run_audit(data, preds)
    plot_raw_data_stack(data, preds)
    final_df, sv_list, x_list = run_estimation(data, preds)
    
    # SHAP Suite (Reuse stage 3 logic from previous version here)
    print(f"\n[DONE] Pipeline Success. Results in: {Config.OUTPUT_NAME}")