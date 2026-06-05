# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 23:55:00 2026
@author: cmarsilli & Gemini
Sovereign Crisis Forecasting Engine v20.3 (Status Indicators Added)

UPDATES:
- UI: Added detailed print statements to track progress without progress bars.
- PLOTTING FIX: Forces 'Agg' backend to ensure files save.
- DIAGNOSTICS: Prints Data Shape and Class Balance.
"""

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for file saving

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import warnings
import itertools

# --- Optional Imports ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: 'shap' library not found. SHAP plots will be skipped.")

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOWESS_AVAILABLE = True
except ImportError:
    LOWESS_AVAILABLE = False
    print("WARNING: 'statsmodels' (lowess) not found. Trend lines will be skipped.")

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class Config:
    OUTPUT_ROOT = "Output_v20_Shap_Global-Month"
    MODELS_TO_RUN = ["Global", "Month-Specific"]
    
    # Core Settings
    USE_MONOTONIC = True
    USE_NATIVE_IMPUTATION = True
    CV_TYPE = "Temporal"
    CV_FOLDS = 5
    HORIZON = 12
    TARGET = "precrisis"
    
    # Data & UI
    RUN_RAW_DATA_STACK = True
    WINSORIZE_LIMITS = [0.01, 0.99]
    
    # 0: Off; 1: Beeswarm; 2: Beeswarm + Dependence; 3: Beeswarm + Dependence + Interaction
    SHAP_COMPLEXITY = 3 
    SHAP_SAMPLES = 50000
    
    CHART_SIZE = (10, 10)
    COLORS = {
        'DARK_RED': '#C0392B',    'DARK_BLUE': '#1F618D',    
        'PASTEL_BLUE': '#AED6F1', 'YELLOW': '#F1C40F',       
        'GREY': '#B0B0B0',        'CRISIS_SHADE': '#606060', 
        'OBS_LINE': '#2C3E50',    'PRED_LINE_MAIN': '#C0392B',
        'TRAIN_BAR': '#2E86C1',   'TEST_BAR': '#E74C3C',
        'GLOBAL_LINE': '#1F618D', 'MONTHLY_LINE': '#C0392B'
    }

    # SHAP Plot Settings
    DEP_PLOT_X_PCT = [1, 97]        
    DEP_PLOT_Y_PCT = [.5, 99.5]  
    DEP_PLOT_COLOR = COLORS['DARK_BLUE']
    DEP_PLOT_ALPHA = 0.6        

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    # Hyperparameters
    PARAM_GRID = {
        'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
        'max_iter': [300, 500, 800, 1000],
        'max_depth': [3, 4, 5, 6, 8],
        'l2_regularization': [0.0, 0.1, 1.0, 5.0, 15.0],
        'max_leaf_nodes': [15, 31, 40, 60],
        'min_samples_leaf': [20, 40, 60]
    }

    # Feature Metadata
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
        return cls.VARS.get(var, (var, "Macro", 0))

# =============================================================================
# 2. CHARTING SUITE
# =============================================================================

def apply_style(ax, title, xlabel, ylabel, grid=True):
    ax.set_title(title, fontsize=14, fontweight='bold', color=Config.COLORS['OBS_LINE'])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid:
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    else:
        ax.grid(False)

def plot_performance_suite(y_true, y_score, label, out_path):
    if len(np.unique(y_true)) < 2: 
        return
    clean = label.replace(" ", "_").replace("/", "_")
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=Config.CHART_SIZE)
    plt.plot(fpr, tpr, color=Config.COLORS['DARK_RED'], lw=3, label=f'AUC: {auc(fpr, tpr):.3f}')
    plt.plot([0,1],[0,1], color=Config.COLORS['GREY'], linestyle='--')
    apply_style(plt.gca(), f"ROC: {label}", "FPR", "TPR")
    plt.legend()
    plt.savefig(os.path.join(out_path, f"ROC_{clean}.png"), bbox_inches='tight')
    plt.close()

    # Silhouette
    plt.figure(figsize=Config.CHART_SIZE)
    try:
        sns.kdeplot(y_score[y_true==0], label='No Crisis', fill=True, color=Config.COLORS['DARK_BLUE'], alpha=0.3)
        sns.kdeplot(y_score[y_true==1], label='Crisis', fill=True, color=Config.COLORS['DARK_RED'], alpha=0.3)
        apply_style(plt.gca(), f"Silhouette: {label}", "Score", "Density")
        plt.legend()
        plt.savefig(os.path.join(out_path, f"Silhouette_{clean}.png"), bbox_inches='tight')
    except: pass
    plt.close()

def plot_comparisons(global_m_aucs, monthly_m_aucs, out_path):
    plt.figure(figsize=(12, 6))
    
    if global_m_aucs:
        g_df = pd.DataFrame(global_m_aucs)
        sns.lineplot(data=g_df, x='Month', y='AUC', marker='o', lw=2, color=Config.COLORS['GLOBAL_LINE'], label='Global Model')
        
    if monthly_m_aucs:
        m_df = pd.DataFrame(monthly_m_aucs)
        sns.lineplot(data=m_df, x='Month', y='AUC', marker='o', lw=3, color=Config.COLORS['MONTHLY_LINE'], label='Month-Specific Model')

    plt.xticks(range(1, 13))
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    apply_style(plt.gca(), "Comparison: Monthly AUC Evolution", "Month", "AUC")
    plt.legend()
    plt.savefig(os.path.join(out_path, "Comparison_AUCs_Monthly_Evolution.png"))
    plt.close()

def compute_loess(x, y, frac=0.3):
    if not LOWESS_AVAILABLE: return None
    # Remove NaNs for LOESS calculation
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < 20: return None
    
    # Sort for LOESS
    sorted_idxs = np.argsort(x_c)
    x_s = x_c[sorted_idxs]
    y_s = y_c[sorted_idxs]
    try:
        z = lowess(y_s, x_s, frac=frac, return_sorted=False)
        return x_s, z
    except:
        return None

def normalize_shap_values(sv):
    """
    Standardize SHAP values to (N, F) array.
    Handles list output (multiclass/binary) and 3D arrays.
    """
    if isinstance(sv, list):
        # Usually sv[1] is the positive class for binary
        if len(sv) == 2:
            return sv[1]
        else:
            return sv[0] # Fallback
    elif sv.ndim == 3:
        # If (N, F, 2), take the second slice
        return sv[:, :, 1]
    return sv

def plot_shap_advanced(model, X, path, label, predictors):
    if Config.SHAP_COMPLEXITY == 0: 
        return None, None
    
    if not SHAP_AVAILABLE:
        print(f"   [Status] SHAP library not found. Skipping plots for {label}.")
        return None, None
        
    if len(X) < 10:
        print(f"   [Status] Sample size too small ({len(X)}) for SHAP in {label}.")
        return None, None
    
    print(f"   [Status] Calculating SHAP values for {label}...")
    
    # 1. Setup Data
    X_sub = X.sample(min(len(X), Config.SHAP_SAMPLES), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_sub)
    sv = normalize_shap_values(sv_raw)
    
    shap_path = os.path.join(path, f"SHAP_{label}")
    os.makedirs(shap_path, exist_ok=True)
    
    # 2. Beeswarm (Level 1+)
    if Config.SHAP_COMPLEXITY >= 1:
        print(f"      > Drawing Beeswarm Plot...")
        plt.figure(figsize=Config.CHART_SIZE)
        shap.summary_plot(sv, X_sub, show=False)
        plt.title(f"SHAP: {label}", fontsize=14)
        plt.savefig(os.path.join(shap_path, "Beeswarm.png"), bbox_inches='tight')
        plt.close()
    
    # 3. Dependence Plots (Level 2+)
    if Config.SHAP_COMPLEXITY >= 2:
        print(f"      > Drawing Dependence Plots...")
        for i, feature in enumerate(predictors):
            x_vals = X_sub[feature].values
            y_vals = sv[:, i]
            
            if np.isnan(x_vals).all() or np.isnan(y_vals).all(): continue

            xlim = np.nanpercentile(x_vals, Config.DEP_PLOT_X_PCT)
            ylim = np.nanpercentile(y_vals, Config.DEP_PLOT_Y_PCT)
            
            mask = (x_vals >= xlim[0]) & (x_vals <= xlim[1]) & (y_vals >= ylim[0]) & (y_vals <= ylim[1])
            x_filt, y_filt = x_vals[mask], y_vals[mask]
            
            if len(x_filt) < 10: continue

            # Dots Only
            plt.figure(figsize=Config.CHART_SIZE)
            plt.scatter(x_filt, y_filt, color=Config.DEP_PLOT_COLOR, alpha=Config.DEP_PLOT_ALPHA, s=15, edgecolor='none')
            plt.xlim(xlim); plt.ylim(ylim)
            apply_style(plt.gca(), f"Dependence: {feature}", feature, "SHAP Value", grid=False)
            plt.savefig(os.path.join(shap_path, f"Dep_{feature}_Dots.png"), bbox_inches='tight')
            plt.close()
            
            # Dots + Trend
            plt.figure(figsize=Config.CHART_SIZE)
            plt.scatter(x_filt, y_filt, color=Config.DEP_PLOT_COLOR, alpha=Config.DEP_PLOT_ALPHA, s=15, edgecolor='none')
            loess_res = compute_loess(x_filt, y_filt)
            if loess_res:
                plt.plot(loess_res[0], loess_res[1], color='red', lw=3, label='Trend (LOESS)')
                plt.legend()
            plt.xlim(xlim); plt.ylim(ylim)
            apply_style(plt.gca(), f"Dependence: {feature}", feature, "SHAP Value", grid=False)
            plt.savefig(os.path.join(shap_path, f"Dep_{feature}_Trend.png"), bbox_inches='tight')
            plt.close()

    # 4. Interaction Plots (Level 3+)
    if Config.SHAP_COMPLEXITY >= 3:
        print(f"      > Drawing Interaction Plots (this takes time)...")
        for i, f1 in enumerate(predictors):
            for j, f2 in enumerate(predictors):
                if i >= j: continue 
                
                x_vals = X_sub[f1].values
                y_vals = sv[:, i] 
                c_vals = X_sub[f2].values 
                
                if np.isnan(x_vals).all() or np.isnan(y_vals).all() or np.isnan(c_vals).all(): continue

                xlim = np.nanpercentile(x_vals, Config.DEP_PLOT_X_PCT)
                ylim = np.nanpercentile(y_vals, Config.DEP_PLOT_Y_PCT)
                
                mask = (x_vals >= xlim[0]) & (x_vals <= xlim[1]) & (y_vals >= ylim[0]) & (y_vals <= ylim[1])
                x_filt, y_filt, c_filt = x_vals[mask], y_vals[mask], c_vals[mask]
                
                if len(x_filt) < 20: continue
                
                # Gradient Dots
                plt.figure(figsize=Config.CHART_SIZE)
                sc = plt.scatter(x_filt, y_filt, c=c_filt, cmap='coolwarm', alpha=0.6, s=15, edgecolor='none')
                plt.colorbar(sc, label=f2)
                plt.xlim(xlim); plt.ylim(ylim)
                apply_style(plt.gca(), f"Interaction: {f1} x {f2}", f1, f"SHAP value for {f1}", grid=False)
                plt.savefig(os.path.join(shap_path, f"Int_{f1}_x_{f2}_Gradient.png"), bbox_inches='tight')
                plt.close()
                
                # Split Trends
                plt.figure(figsize=Config.CHART_SIZE)
                #plt.scatter(x_filt, y_filt, color='gray', alpha=0.1, s=10) 
                sc = plt.scatter(x_filt, y_filt, c=c_filt, cmap='coolwarm', alpha=0.6, s=15, edgecolor='none')
                plt.colorbar(sc, label=f2)
                q1_c, q3_c = np.nanpercentile(c_filt, 20), np.nanpercentile(c_filt, 80)
                
                mask_low = c_filt <= q1_c
                if mask_low.sum() > 10:
                    loess_low = compute_loess(x_filt[mask_low], y_filt[mask_low])
                    if loess_low: plt.plot(loess_low[0], loess_low[1], color='blue', lw=2, label=f'Low {f2} (<=Q1)')
                
                mask_high = c_filt >= q3_c
                if mask_high.sum() > 10:
                    loess_high = compute_loess(x_filt[mask_high], y_filt[mask_high])
                    if loess_high: plt.plot(loess_high[0], loess_high[1], color='red', lw=2, label=f'High {f2} (>=Q3)')
                
                plt.xlim(xlim); plt.ylim(ylim)
                plt.legend()
                apply_style(plt.gca(), f"Interaction Trends: {f1} x {f2}", f1, f"SHAP value for {f1}", grid=False)
                plt.savefig(os.path.join(shap_path, f"Int_{f1}_x_{f2}_Trends.png"), bbox_inches='tight')
                plt.close()

    return sv, X_sub

def plot_combined_beeswarm(shap_list, out_path):
    if not shap_list: return
    print("   [Status] Combining SHAP values for summary plot...")
    all_sv = np.concatenate([x[0] for x in shap_list], axis=0)
    all_X = pd.concat([x[1] for x in shap_list], axis=0)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(all_sv, all_X, show=False)
    plt.title("Combined SHAP (All Months)", fontsize=16)
    plt.savefig(os.path.join(out_path, "Combined_Beeswarm.png"), bbox_inches='tight')
    plt.close()

def plot_country_chart(sub, country_name, out_path, model_label):
    p = os.path.join(out_path, "Country_Charts")
    os.makedirs(p, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(sub['Date'], sub['Risk_Index'], color=Config.COLORS['PRED_LINE_MAIN'], lw=2, label='Risk')
    plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), color=Config.COLORS['CRISIS_SHADE'], alpha=0.4, label='Crisis')
    apply_style(plt.gca(), f"{country_name} ({model_label})", "Date", "Risk")
    plt.savefig(os.path.join(p, f"{country_name}.png"), bbox_inches='tight')
    plt.close()

def plot_flashing_red(df, out_path, model_label):
    p = os.path.join(out_path, "Warnings")
    os.makedirs(p, exist_ok=True)
    df = df.sort_values(['Country', 'Date'])
    df['Delta'] = df.groupby('Country')['Risk_Index'].diff(12)
    alerts = df[(df['Delta'] > 0.15) & (df['Risk_Index'] > 0.4) & (df['Date'] > '2020-01-01')]
    for c in alerts['Country_Name'].unique()[:15]:
        sub = df[df['Country_Name'] == c]
        plt.figure(figsize=(10, 4))
        plt.plot(sub['Date'], sub['Risk_Index'], color=Config.COLORS['PRED_LINE_MAIN'], lw=2)
        plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), color=Config.COLORS['CRISIS_SHADE'], alpha=0.4)
        apply_style(plt.gca(), f"WARNING: {c}", "Date", "Risk")
        plt.savefig(os.path.join(p, f"WARNING_{c}.png"), bbox_inches='tight')
        plt.close()

# =============================================================================
# 3. DATA PREP (EXTENDED FORECAST + ROBUST MERGE)
# =============================================================================

def prepare_data():
    print("--- Loading and Preparing Data ---")
    df = pd.read_csv(Config.FILE_PATH)
    
    # 1. RENAME IMMEDIATELY
    if 'COUNTRY' in df.columns: df.rename(columns={'COUNTRY': 'Country'}, inplace=True)
    if 'COUNTRY_name' in df.columns: df.rename(columns={'COUNTRY_name': 'Country_Name'}, inplace=True)
    
    # 2. CLEAN KEY
    df['Country'] = pd.to_numeric(df['Country'], errors='coerce').fillna(0).astype(int)
    
    # 3. MERGE
    if os.path.exists(Config.MAPPING_FILE):
        map_df = pd.read_csv(Config.MAPPING_FILE, encoding='latin1')
        map_df['IFS'] = pd.to_numeric(map_df['IFS'], errors='coerce').fillna(0).astype(int)
        
        print("   [Status] Merging Main (Country) with Map (IFS)...")
        df = df.merge(map_df[['IFS', 'income', 'Area', 'Country_Name']], 
                      left_on='Country', right_on='IFS', how='left')
        
        # Fill Metadata
        df['income'] = df['income'].fillna("Unknown")
        df['Area'] = df['Area'].fillna("Unknown")
        # Handle duplicate Country_Name column from merge
        if 'Country_Name_y' in df.columns:
            df['Country_Name'] = df['Country_Name_y'].fillna(df['Country_Name_x'])
            df = df.drop(columns=['Country_Name_x', 'Country_Name_y'])
    else:
        print("   [Status] Mapping file not found.")
        df['income'] = "Unknown"
        df['Area'] = "Unknown"

    # 4. EXTEND FORECAST - DISABLED PER USER REQUEST (No carry forward)
    # df = extend_forecast_data(df) 

    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    
    # 5. TARGET
    print("   [Status] Calculating Crisis Targets...")
    df = df.sort_values(['Country', 'Date'])
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)

    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    
    print("   [Status] Winsorizing and Imputing Features...")
    for p in predictors:
        lower, upper = df[p].quantile(Config.WINSORIZE_LIMITS)
        df[p] = df[p].clip(lower, upper)
        if not Config.USE_NATIVE_IMPUTATION: df[p] = df[p].fillna(df[p].median())

    m_cst = [Config.get_meta(p)[2] for p in predictors] if Config.USE_MONOTONIC else None
    
    return df.dropna(subset=['Target_H']).copy(), df[df['Target_H'].isna()].copy(), predictors, m_cst

# =============================================================================
# 4. ENGINE
# =============================================================================

def run_engine():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    print("\n=== ENGINE START ===")
    d_tr, d_fc, preds, constraints = prepare_data()
    
    # DIAGNOSTICS
    print(f"\n--- Data Diagnostics ---")
    print(f"Training Rows: {len(d_tr)}")
    print(f"Crisis Count: {d_tr['Target_H'].sum()}")
    print(f"Crisis Rate: {d_tr['Target_H'].mean():.4f}")
    if len(d_tr) == 0:
        print("ERROR: Training set is empty. Check horizon calculation or missing data.")
        return
    
    results = []
    
    # Storage for comparison charts
    global_monthly_aucs = []
    month_specific_aucs = []
    
    cv_t = TimeSeriesSplit(n_splits=Config.CV_FOLDS) if Config.CV_TYPE == "Temporal" else StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True)

    for mode in Config.MODELS_TO_RUN:
        path = os.path.join(Config.OUTPUT_ROOT, mode)
        os.makedirs(path, exist_ok=True)
        print(f"\n>>> Running Model Loop: {mode}")
        
        d_tr['Risk_Index'] = np.nan
        d_fc['Risk_Index'] = np.nan
        shap_coll = []
        
        if mode == "Global":
            print("   [Status] Optimizing Hyperparameters (RandomizedSearch)...")
            clf = HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42)
            search = RandomizedSearchCV(clf, Config.PARAM_GRID, n_iter=15, cv=cv_t, scoring='roc_auc', n_jobs=-1)
            search.fit(d_tr[preds], d_tr['Target_H'])
            best = search.best_estimator_
            
            print("   [Status] Generating Predictions...")
            d_tr['Risk_Index'] = best.predict_proba(d_tr[preds])[:, 1]
            d_fc['Risk_Index'] = best.predict_proba(d_fc[preds])[:, 1]
            
            # RUN ADVANCED SHAP
            plot_shap_advanced(best, d_tr[preds], path, "Global", preds)
            
            # Calculate Global Model's performance per month for comparison
            for m in range(1, 13):
                m_sub = d_tr[d_tr['month'] == m]
                if len(m_sub) > 20:
                    sc = roc_auc_score(m_sub['Target_H'], m_sub['Risk_Index'])
                    global_monthly_aucs.append({'Month': m, 'AUC': sc})

        elif mode == "Month-Specific":
            for m in range(1, 13):
                print(f"   [Status] Training Month-Specific Model for Month {m}/12...")
                m_tr = d_tr[d_tr['month'] == m]
                m_fc = d_fc[d_fc['month'] == m]
                if len(m_tr) < 30: continue
                
                # FIX: PREVENT OVERFITTING/LEAK
                sub_cv = TimeSeriesSplit(n_splits=3)
                
                clf = HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42)
                search = RandomizedSearchCV(clf, Config.PARAM_GRID, n_iter=10, cv=sub_cv, scoring='roc_auc', n_jobs=-1)
                search.fit(m_tr[preds], m_tr['Target_H'])
                best = search.best_estimator_
                
                d_tr.loc[d_tr['month'] == m, 'Risk_Index'] = best.predict_proba(m_tr[preds])[:, 1]
                if not m_fc.empty:
                    # FIX: NameError 'predictors' -> 'preds'
                    d_fc.loc[d_fc['month'] == m, 'Risk_Index'] = best.predict_proba(m_fc[preds])[:, 1]
                
                # Metrics
                m_auc = roc_auc_score(m_tr['Target_H'], best.predict_proba(m_tr[preds])[:, 1])
                month_specific_aucs.append({'Month': m, 'AUC': m_auc})
                
                # Run advanced shap for each month
                sv, Xd = plot_shap_advanced(best, m_tr[preds], path, f"Month_{m:02d}", preds)
                if sv is not None: shap_coll.append((sv, Xd))
                
                plot_performance_suite(m_tr['Target_H'], best.predict_proba(m_tr[preds])[:, 1], f"Month_{m:02d}", path)

            plot_combined_beeswarm(shap_coll, path)

        # Diagnostics (All Models)
        valid = d_tr['Risk_Index'].notna()
        if valid.any():
            auc_val = roc_auc_score(d_tr.loc[valid, 'Target_H'], d_tr.loc[valid, 'Risk_Index'])
            results.append({'Model': mode, 'Test_AUC': auc_val})
            print(f"   > Overall AUC ({mode}): {auc_val:.3f}")
            
            plot_performance_suite(d_tr.loc[valid, 'Target_H'], d_tr.loc[valid, 'Risk_Index'], "Total", path)
            
            # --- GRANULAR AUC METRICS (Groups) ---
            print(f"   [Status] Calculating Granular AUCs (Income/Area)...")
            
            # 1. By Group (Income & Area)
            for cat in ['income', 'Area']:
                for val in d_tr[cat].unique():
                    sub = d_tr[(d_tr[cat] == val) & valid]
                    if len(sub) > 50 and sub['Target_H'].nunique() > 1:
                        grp_auc = roc_auc_score(sub['Target_H'], sub['Risk_Index'])
                        results.append({'Model': mode, 'Type': 'Group', 'Group': cat, 'Value': val, 'Test_AUC': grp_auc})
                        plot_performance_suite(sub['Target_H'], sub['Risk_Index'], f"{cat}_{val}", path)
            
            # 2. By Country
            for c in d_tr['Country_Name'].unique():
                sub = d_tr[(d_tr['Country_Name'] == c) & valid]
                if len(sub) > 20 and sub['Target_H'].nunique() > 1:
                    try:
                        cnt_auc = roc_auc_score(sub['Target_H'], sub['Risk_Index'])
                        results.append({'Model': mode, 'Type': 'Country', 'Group': 'Country', 'Value': c, 'Test_AUC': cnt_auc})
                    except:
                        pass
            
            # Country Charts
            full = pd.concat([d_tr, d_fc])
            plot_flashing_red(full, path, mode)
            print(f"   [Status] Generating Country Charts for {len(full['Country_Name'].unique())} entities...")
            for c in full['Country_Name'].unique():
                sub = full[full['Country_Name'] == c].sort_values('Date')
                if len(sub) > 12: plot_country_chart(sub, c, path, mode)

    # Final Comparative Charts
    plot_comparisons(global_monthly_aucs, month_specific_aucs, Config.OUTPUT_ROOT)
    
    print("\n=== FINAL RESULTS (Summary) ===")
    res_df = pd.DataFrame(results)
    print(res_df.head(15)) 
    res_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Full_Granular.csv"), index=False)
    print("\n=== PROCESS COMPLETED ===")

if __name__ == "__main__":
    run_engine()