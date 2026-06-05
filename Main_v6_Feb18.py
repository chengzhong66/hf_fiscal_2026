# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:05:00 2026
@author: cmarsilli
Sovereign Crisis Forecasting Engine v6.5 (Advanced Performance Suite)
Pipeline: Audit -> Raw Visuals -> Estimation -> Performance (ROC/Gains/Lift) -> SHAP Interactions
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import warnings

# --- Suppress non-critical warnings ---
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
    OUTPUT_NAME = "Output_v6_Global_noMC"
    
    # --- USER TOGGLES ---
    USE_MONOTONIC_CONSTRAINTS = False
    USE_NATIVE_IMPUTATION = True      
    RUN_RAW_DATA_STACK = False         
    
    # --- SHAP CONFIGURATION ---
    COMPUTE_SHAP = True
    SHAP_MODE = "Quick"                # "Quick" (Beeswarm) or "Full" (Dependence)
    COMPUTE_ALL_INTERACTIONS = False   # Set to True for full Interaction Matrix
    INTERACTION_TOP_N = 5             # Primary features for dependence
    INTERACTION_SAMPLES = 300          # Sample size for interaction calculation (Keep low for speed)
    
    # --- MODEL SETTINGS ---
    HORIZON = 12
    MODEL_MODE = "Global"             
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
        "PCPI_PCH": ("Inflation", "Macro", 1),
        "gdp_growth": ("GDP Growth", "Macro", -1),
        "BoP_gdp": ("Current Account/GDP", "Macro", -1),
        "reserve_cover": ("FX Reserve Cover", "Macro", -1),
        "terms_of_trade": ("Terms of Trade", "Macro", -1),
        "GDP_percapita_over_US_12ma": ("GDP per Capita", "Macro", -1), 
        "oil_to_gdp": ("Oil Exports/GDP", "Macro", -1), 
        "oil_shock_impact": ("Oil Exports Gains", "Macro", -1), 
        "govt_debt_gdp": ("Public Debt/GDP", "Fiscal", 1),
        "govt_deficit_gdp": ("Fiscal Balance/GDP", "Fiscal", -1),
        "govt_revenue_gdp": ("Fiscal Revenue/GDP", "Fiscal", -1),
        "tot_ext_debt_gdp": ("Total Ext Debt/GDP", "Fiscal", 1),
        "debt_service_gdp": ("Debt Service/GDP", "Fiscal", 1),
        "debt_fx_vulnerability": ("Debt x FX Shock", "Fiscal", 1), 
        "corruption_12ma": ("Corruption", "Fiscal", -1),
        "deposit_rate": ("ST Rate", "Financial", 1),
        "long_term_bond_yield": ("LT Rate", "Financial", 1),
        "WUI": ("Uncertainty Idx", "Financial", 1),
        "ENDE_yoy": ("FX Depreciation", "Financial", 1),
        "spread": ("Sovereign Spread", "Financial", 1), 
        "oil_price": ("Oil Price", "Financial", 0),
        "VIX": ("VIX Index", "Financial", 1)
    }

    @classmethod
    def get_meta(cls, var):
        return cls.VARS.get(var, (var, "Macroeconomic", 0))

# =============================================================================
# 2. AUDIT & DATA STACKS
# =============================================================================

def setup_folders():
    subdirs = ["data", "charts/audit", "charts/performance", "country_charts", "country_raw_stacks",
               "shap/dependence", "shap/interactions", "shap/seasonal"]
    for d in subdirs: os.makedirs(os.path.join(Config.OUTPUT_NAME, d), exist_ok=True)

def prepare_data():
    df = pd.read_csv(Config.FILE_PATH)
    if 'COUNTRY_name' in df.columns: df.rename(columns={'COUNTRY_name': 'Country'}, inplace=True)
    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    df = df.sort_values(['Country', 'Date'])

    df['oil_price_pch'] = df.groupby('Country')['oil_price'].pct_change()
    df['oil_shock_impact'] = df['oil_to_gdp'] * df['oil_price_pch']
    df['debt_fx_vulnerability'] = df['ENDE_yoy'] * df['tot_ext_debt_gdp']

    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)
    
    if not Config.USE_NATIVE_IMPUTATION:
        df[predictors] = df.groupby('Country')[predictors].ffill().fillna(df[predictors].median())
    
    return df.dropna(subset=['Target_H']).copy(), predictors

def run_audit(df, predictors):
    print("\n[1/5] Running Data Audit and Diagnostics...")
    path = os.path.join(Config.OUTPUT_NAME, "charts/audit")
    missing = df[predictors].isna().mean().sort_values(ascending=False)
    missing.to_csv(os.path.join(Config.OUTPUT_NAME, "data/missing_stats.csv"))
    plt.figure(figsize=(10, 6)); sns.heatmap(df[predictors].isna(), cbar=False, cmap='binary')
    plt.title("Missing Data Audit"); plt.savefig(os.path.join(path, "missing_map.png")); plt.close()

def plot_raw_data_stack(df, predictors):
    if not Config.RUN_RAW_DATA_STACK: return
    print("\n[2/5] Generating Raw Data Stacks (Country-level)...")
    path = os.path.join(Config.OUTPUT_NAME, "country_raw_stacks")
    # Using tqdm for a clear progress bar across countries
    for country, g in tqdm(df.groupby('Country'), desc="Data Stacks Progress"):
        fig, axes = plt.subplots(len(predictors), 1, figsize=(12, 1.5*len(predictors)), sharex=True)
        for i, var in enumerate(predictors):
            axes[i].plot(g['Date'], g[var], color='navy', lw=1.2)
            axes[i].fill_between(g['Date'], g[var].min(), g[var].max(), where=g['Target_H']==1, color='grey', alpha=0.3)
            axes[i].set_ylabel(Config.get_meta(var)[0], rotation=0, labelpad=50, fontsize=7)
        plt.savefig(os.path.join(path, f"{country}_full_stack.png"), bbox_inches='tight'); plt.close()

# =============================================================================
# 3. ESTIMATION & PERFORMANCE
# =============================================================================

def plot_performance_metrics(y_true, y_score, mode_name):
    """Generates ROC, Cumulative Gains, and Decile Lift charts."""
    path = os.path.join(Config.OUTPUT_NAME, "charts/performance")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(); plt.plot(fpr, tpr, color='crimson', label=f'AUC: {auc(fpr, tpr):.3f}')
    plt.plot([0,1],[0,1], 'k--'); plt.legend(); plt.savefig(os.path.join(path, f"roc_{mode_name}.png")); plt.close()

    # 2. Cumulative Gains
    df_g = pd.DataFrame({'y': y_true, 's': y_score}).sort_values('s', ascending=False)
    df_g['cum_y'] = df_g['y'].cumsum() / df_g['y'].sum()
    df_g['cum_p'] = np.arange(1, len(df_g)+1) / len(df_g)
    plt.figure(); plt.plot(df_g['cum_p'], df_g['cum_y'], label='Model'); plt.plot([0,1],[0,1],'k--')
    plt.title(f"Cumulative Gains: {mode_name}")
    plt.savefig(os.path.join(path, f"gains_{mode_name}.png")); plt.close()

    # 3. Decile-Based Lift Chart
    df_l = pd.DataFrame({'y': y_true, 's': y_score}).sort_values('s', ascending=False)
    # Group into 10 deciles (using rank to handle duplicate scores)
    df_l['decile'] = pd.qcut(df_l['s'].rank(method='first'), 10, labels=False)
    df_l['decile'] = 9 - df_l['decile'] # Re-label so 0 is the highest risk decile
    
    # Calculate Lift: (Decile Average Target) / (Overall Sample Average Target)
    lift_data = df_l.groupby('decile')['y'].mean() / df_l['y'].mean()
    
    plt.figure(figsize=(8, 5))
    lift_data.plot(kind='bar', color='darkblue', alpha=0.7)
    plt.axhline(1, color='red', linestyle='--', label='Baseline (1.0)')
    plt.title(f"Decile Lift Chart: {mode_name}")
    plt.xlabel("Decile (0 = Highest Risk Score)")
    plt.ylabel("Lift (x Baseline)")
    plt.savefig(os.path.join(path, f"lift_{mode_name}.png")); plt.close()

def run_estimation(df, predictors):
    print(f"\n[3/5] Estimating Forecasting Models ({Config.MODEL_MODE} mode)...")
    constraints = [Config.get_meta(p)[2] for p in predictors] if Config.USE_MONOTONIC_CONSTRAINTS else None
    modes = [(f"Month_{m}", df[df['month'] == m]) for m in range(1, 13)] if Config.MODEL_MODE == "Month-Specific" else [("Global", df)]
    reconstructed_list, all_shap_values, all_X_data, best_models = [], [], [], {}

    # Progress bar for model training segments
    for mode_name, sub_df in tqdm(modes, desc="Training Progress"):
        X, y = sub_df[predictors], sub_df['Target_H']
        split_idx = int(len(sub_df) * (1 - Config.TEST_SHARE))
        X_tr, X_ts, y_tr, y_ts = X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]
        if len(y_tr.unique()) < 2: continue

        search = RandomizedSearchCV(HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42), 
                                    Config.PARAM_GRID, n_iter=10, cv=TimeSeriesSplit(n_splits=Config.CV_FOLDS), scoring='roc_auc', n_jobs=-1)
        search.fit(X_tr, y_tr); best_clf = search.best_estimator_; best_models[mode_name] = best_clf
        
        y_prob_ts = best_clf.predict_proba(X_ts)[:, 1]
        plot_performance_metrics(y_ts, y_prob_ts, mode_name)
        sub_df['Risk_Index'] = best_clf.predict_proba(X)[:, 1]
        reconstructed_list.append(sub_df)

        if Config.COMPUTE_SHAP and SHAP_INSTALLED:
            explainer = shap.Explainer(best_clf.predict_proba, X)
            sample_X = X.sample(min(Config.INTERACTION_SAMPLES, len(X)))
            all_shap_values.append(explainer(sample_X)[:, :, 1].values); all_X_data.append(sample_X)

    # Post-estimation country charts with automated progress feedback
    print("Generating Country Forecast Profiles...")
    full_df = pd.concat(reconstructed_list).sort_values(['Country', 'Date'])
    for c, g in full_df.groupby('Country'):
        plt.figure(figsize=(10, 4)); plt.plot(g['Date'], g['Risk_Index'], color='red')
        plt.fill_between(g['Date'], 0, 1, where=g['Target_H']==1, color='grey', alpha=0.2)
        plt.savefig(os.path.join(Config.OUTPUT_NAME, f"country_charts/{c}.png")); plt.close()

    return full_df, all_shap_values, all_X_data, best_models

# =============================================================================
# 4. SHAP INTERACTION SUITE
# =============================================================================

def run_shap_suite(sv_list, x_list, preds, best_models):
    if not (Config.COMPUTE_SHAP and SHAP_INSTALLED) or not sv_list: return
    print(f"\n[4/5] Running SHAP Interpretability Suite...")
    cat_sv, cat_X = np.concatenate(sv_list, axis=0), pd.concat(x_list, axis=0)
    sh_exp = shap.Explanation(values=cat_sv, data=cat_X.values, feature_names=preds)
    
    plt.figure(); shap.plots.beeswarm(sh_exp, show=False); plt.savefig(os.path.join(Config.OUTPUT_NAME, "shap/beeswarm.png"), bbox_inches='tight'); plt.close()

    if Config.COMPUTE_ALL_INTERACTIONS:
        print("Computing Full SHAP Interaction Matrix (Sampled)...")
        path_int = os.path.join(Config.OUTPUT_NAME, "shap/interactions")
        target_model = list(best_models.values())[0]
        explainer = shap.TreeExplainer(target_model)
        X_sample = cat_X.sample(min(Config.INTERACTION_SAMPLES, len(cat_X)))
        inter_vals = explainer.shap_interaction_values(X_sample)
        
        mean_inter = np.abs(inter_vals).mean(axis=0)
        inter_df = pd.DataFrame(mean_inter, index=preds, columns=preds)
        plt.figure(figsize=(12, 10)); sns.heatmap(inter_df, cmap='YlOrRd')
        plt.savefig(os.path.join(path_int, "interaction_heatmap.png")); plt.close()

# =============================================================================
# 5. EXECUTION
# =============================================================================

if __name__ == "__main__":
    setup_folders()
    data, preds = prepare_data()
    run_audit(data, preds)
    plot_raw_data_stack(data, preds)
    final_df, sv_list, x_list, models = run_estimation(data, preds)
    run_shap_suite(sv_list, x_list, preds, models)
    print(f"\n[5/5] SUCCESS: Sovereign Engine Execution Complete.")
    print(f"Outputs saved in: {Config.OUTPUT_NAME}")