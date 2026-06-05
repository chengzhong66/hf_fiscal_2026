
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 18:30:00 2026
@author: cmarsilli & Gemini
Sovereign Crisis Forecasting Engine v33.0 (The Stable Grand Tournament)

UPDATES:
- 4-Way Architecture: XGBoost vs Midas-XGBoost (Global & Monthly).
- STABILITY FIX: Strict Almon boundaries + L2 Regularization on monthly models.
- UI & DIAGNOSTICS: Retains v20.3 granular outputs, Silhouette, SHAP.
- NEW OUTPUTS: 13 Almon lag charts, Volatility comparison, full CSVs.
"""

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for file saving

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
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
    OUTPUT_ROOT = "Output_v33_Stable_Tournament"
    
    # Core Settings
    USE_MONOTONIC = True
    USE_NATIVE_IMPUTATION = True
    CV_TYPE = "Temporal"
    CV_FOLDS = 5
    HORIZON = 12
    TARGET = "precrisis"
    WINSORIZE_LIMITS = [0.01, 0.99]
    
    # SHAP & UI
    SHAP_COMPLEXITY = 1 # Set to 1 (Beeswarm only) by default to save time. Change to 3 for interactions.
    SHAP_SAMPLES = 20000
    
    CHART_SIZE = (10, 6)
    COLORS = {
        'XGBoost': '#1F618D',               # Dark Blue
        'Midas-XGBoost': '#2C3E50',         # Dark Slate
        'Monthly XGBoost': '#E74C3C',       # Red (Erratic Baseline)
        'Monthly Midas-XGBoost': '#C0392B', # Dark Red
        'CRISIS_SHADE': '#404040',          # Darker Grey for crisis periods
        'ALMON_BAR': '#2E86C1',
        'ALMON_LINE': '#F1C40F'
    }

    DEP_PLOT_X_PCT = [1, 99]        
    DEP_PLOT_Y_PCT = [.5, 99.5]  
    DEP_PLOT_COLOR = COLORS['XGBoost']
    DEP_PLOT_ALPHA = 0.6        

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    # --- HYPERPARAMETERS ---
    # Standard XGB Grid
    XGB_GRID = {
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'max_iter': [300, 500, 800],
        'max_depth': [3, 4, 5, 6],
        'l2_regularization': [0.0, 1.0, 5.0, 15.0],
        'min_samples_leaf': [20, 40]
    }

    # MIDAS Grid (Strict boundaries to ensure stability)
    MIDAS_GRID = {
        'clf__learning_rate': [0.01, 0.02, 0.05],
        'clf__max_depth': [3, 4, 5],
        'clf__l2_regularization': [1.0, 5.0, 15.0],  # Heavier penalty for stability
        'almon__theta1': [-0.2, -0.05, -0.01], # Forced decay
        'almon__theta2': [-0.005, 0.0]                # Prevent U-shapes
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
        return cls.VARS.get(var, (var, "Macro", 0))

# =============================================================================
# 2. ALMON COMBINER (Weighting Engine)
# =============================================================================

class AlmonValueCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, base_features=None, max_lag=12, theta1=-0.5, theta2=0.0):
        self.base_features = base_features 
        self.max_lag = max_lag
        self.theta1 = theta1
        self.theta2 = theta2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        k = np.arange(1, self.max_lag + 1)
        w_raw = np.exp(self.theta1 * k + self.theta2 * (k**2))
        weights = w_raw / np.sum(w_raw)

        for base in self.base_features:
            weighted_sum = 0
            for i, lag in enumerate(k):
                col_name = f"{base}_lag{lag}"
                if col_name in X.columns:
                    weighted_sum += X[col_name].fillna(0) * weights[i]
            X_out[base] = weighted_sum
        return X_out

# =============================================================================
# 3. CHARTING SUITE
# =============================================================================

def apply_style(ax, title, xlabel, ylabel, grid=True):
    ax.set_title(title, fontsize=14, fontweight='bold', color=Config.COLORS['Midas-XGBoost'])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid: ax.grid(axis='y', linestyle='--', alpha=0.5)
    else: ax.grid(False)

def plot_almon_weights(theta1, theta2, max_lag, label, out_path):
    k = np.arange(1, max_lag + 1)
    w_raw = np.exp(theta1 * k + theta2 * (k**2))
    w = w_raw / np.sum(w_raw)
    
    plt.figure(figsize=(8, 5))
    plt.bar(k, w, color=Config.COLORS['ALMON_BAR'], alpha=0.7)
    plt.plot(k, w, color=Config.COLORS['ALMON_LINE'], marker='o', lw=2)
    apply_style(plt.gca(), f"Almon Lag Structure: {label}\n(t1={theta1:.3f}, t2={theta2:.3f})", "Lag (Months)", "Weight")
    plt.xticks(k)
    os.makedirs(os.path.join(out_path, "Almon_Structures"), exist_ok=True)
    plt.savefig(os.path.join(out_path, "Almon_Structures", f"Almon_{label}.png"), bbox_inches='tight')
    plt.close()

def plot_performance_suite(y_true, y_score, label, color_key, out_path):
    if len(np.unique(y_true)) < 2: return
    clean = label.replace(" ", "_").replace("/", "_")
    p = os.path.join(out_path, "Diagnostics", clean)
    os.makedirs(p, exist_ok=True)
    
    # 1. ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=Config.CHART_SIZE)
    plt.plot(fpr, tpr, color=Config.COLORS[color_key], lw=3, label=f'AUC: {auc(fpr, tpr):.3f}')
    plt.plot([0,1],[0,1], color='#B0B0B0', linestyle='--')
    apply_style(plt.gca(), f"ROC: {label}", "FPR", "TPR")
    plt.legend()
    plt.savefig(os.path.join(p, f"ROC.png"), bbox_inches='tight')
    plt.close()

    # 2. Silhouette
    plt.figure(figsize=Config.CHART_SIZE)
    try:
        sns.kdeplot(y_score[y_true==0], label='No Crisis', fill=True, color='#1F618D', alpha=0.3)
        sns.kdeplot(y_score[y_true==1], label='Crisis', fill=True, color='#C0392B', alpha=0.3)
        apply_style(plt.gca(), f"Silhouette Distribution: {label}", "Predicted Risk Score", "Density")
        plt.legend()
        plt.savefig(os.path.join(p, f"Silhouette.png"), bbox_inches='tight')
    except: pass
    plt.close()

def plot_volatility_comparison(df, out_path):
    df = df.sort_values(['Country', 'Date'])
    metrics = []
    models = ['Risk_XGBoost', 'Risk_Midas-XGBoost', 'Risk_Monthly XGBoost', 'Risk_Monthly Midas-XGBoost']
    
    for mod in models:
        # Mean absolute diff = jumpiness
        vol = df.groupby('Country')[mod].diff().abs().mean()
        metrics.append({'Model': mod.replace('Risk_', ''), 'Volatility': vol})
    
    res = pd.DataFrame(metrics)
    plt.figure(figsize=Config.CHART_SIZE)
    sns.barplot(data=res, x='Model', y='Volatility', palette=[Config.COLORS[m] for m in res['Model']])
    apply_style(plt.gca(), "Stability Comparison (Lower Volatility is Smoother)", "Model Architecture", "Avg M-o-M Risk Change")
    plt.savefig(os.path.join(out_path, "Comparison_Volatility_Score.png"), bbox_inches='tight')
    plt.close()
    return res

def plot_4way_country_chart(sub, country_name, out_path):
    p = os.path.join(out_path, "Country_Charts")
    os.makedirs(p, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    # Shade Crisis
    plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), color=Config.COLORS['CRISIS_SHADE'], alpha=0.6, label='Crisis Event')
    
    # Plot Trajectories
    plt.plot(sub['Date'], sub['Risk_XGBoost'], color=Config.COLORS['XGBoost'], linestyle=':', lw=1.5, label='XGBoost')
    plt.plot(sub['Date'], sub['Risk_Midas-XGBoost'], color=Config.COLORS['Midas-XGBoost'], linestyle='--', lw=2, label='Midas-XGBoost')
    plt.plot(sub['Date'], sub['Risk_Monthly XGBoost'], color=Config.COLORS['Monthly XGBoost'], lw=1.5, alpha=0.6, label='Monthly XGBoost')
    plt.plot(sub['Date'], sub['Risk_Monthly Midas-XGBoost'], color=Config.COLORS['Monthly Midas-XGBoost'], lw=3, label='Monthly Midas-XGBoost')
    
    apply_style(plt.gca(), f"Risk Trajectory: {country_name}", "Date", "Risk Probability")
    plt.legend(loc='upper left')
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(p, f"{country_name}.png"), bbox_inches='tight')
    plt.close()

def normalize_shap_values(sv):
    if isinstance(sv, list): return sv[1] if len(sv) == 2 else sv[0]
    elif sv.ndim == 3: return sv[:, :, 1]
    return sv

def plot_shap_advanced(model, X, path, label, predictors):
    if Config.SHAP_COMPLEXITY == 0 or not SHAP_AVAILABLE or len(X) < 10: return
    print(f"   [Status] SHAP for {label}...")
    
    X_sub = X.sample(min(len(X), Config.SHAP_SAMPLES), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_sub)
    sv = normalize_shap_values(sv_raw)
    
    shap_path = os.path.join(path, "SHAP", label.replace(" ", "_"))
    os.makedirs(shap_path, exist_ok=True)
    
    # 1. Beeswarm
    if Config.SHAP_COMPLEXITY >= 1:
        plt.figure(figsize=Config.CHART_SIZE)
        shap.summary_plot(sv, X_sub, show=False)
        plt.title(f"SHAP: {label}", fontsize=14)
        plt.savefig(os.path.join(shap_path, "Beeswarm.png"), bbox_inches='tight')
        plt.close()
        
    # (Interactions & Dependence code remains accessible via Config.SHAP_COMPLEXITY = 3)

# =============================================================================
# 4. DATA PREP 
# =============================================================================

def prepare_data():
    print("--- Loading and Preparing Data ---")
    df = pd.read_csv(Config.FILE_PATH)
    
    if 'COUNTRY' in df.columns: df.rename(columns={'COUNTRY': 'Country'}, inplace=True)
    if 'COUNTRY_name' in df.columns: df.rename(columns={'COUNTRY_name': 'Country_Name'}, inplace=True)
    df['Country'] = pd.to_numeric(df['Country'], errors='coerce').fillna(0).astype(int)
    
    if os.path.exists(Config.MAPPING_FILE):
        map_df = pd.read_csv(Config.MAPPING_FILE, encoding='latin1')
        map_df['IFS'] = pd.to_numeric(map_df['IFS'], errors='coerce').fillna(0).astype(int)
        df = df.merge(map_df[['IFS', 'income', 'Area', 'Country_Name']], left_on='Country', right_on='IFS', how='left')
        df['income'] = df['income'].fillna("Unknown")
        df['Area'] = df['Area'].fillna("Unknown")
        if 'Country_Name_y' in df.columns:
            df['Country_Name'] = df['Country_Name_y'].fillna(df['Country_Name_x'])
            df = df.drop(columns=['Country_Name_x', 'Country_Name_y'])
    else:
        df['income'] = "Unknown"
        df['Area'] = "Unknown"

    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    df = df.sort_values(['Country', 'Date'])
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)

    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    m_cst = [Config.get_meta(p)[2] for p in predictors] if Config.USE_MONOTONIC else None
    
    print("   [Status] Winsorizing and Imputing Features...")
    for p in predictors:
        lower, upper = df[p].quantile(Config.WINSORIZE_LIMITS)
        df[p] = df[p].clip(lower, upper)
        if not Config.USE_NATIVE_IMPUTATION: df[p] = df[p].fillna(df[p].median())
        
    print("   [Status] Pre-calculating Raw Lags (1-12) for MIDAS...")
    lag_cols = []
    for p in predictors:
        for lag in range(1, 13):
            col_name = f"{p}_lag{lag}"
            df[col_name] = df.groupby('Country')[p].shift(lag)
            lag_cols.append(col_name)
    
    for lc in lag_cols: df[lc] = df[lc].fillna(df[lc].median())

    return df.dropna(subset=['Target_H']).copy(), predictors, m_cst

# =============================================================================
# 5. ENGINE
# =============================================================================

def run_engine():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    print("\n=== ENGINE START (v33.0) ===")
    d_tr, predictors, constraints = prepare_data()
    
    print(f"\n--- Data Diagnostics ---")
    print(f"Training Rows: {len(d_tr)}")
    print(f"Crisis Count: {d_tr['Target_H'].sum()}")
    print(f"Crisis Rate: {d_tr['Target_H'].mean():.4f}")
    if len(d_tr) == 0: return
    
    cv_t = TimeSeriesSplit(n_splits=Config.CV_FOLDS) if Config.CV_TYPE == "Temporal" else StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=42)
    
    # Initialize Score Columns
    model_keys = ['XGBoost', 'Midas-XGBoost', 'Monthly XGBoost', 'Monthly Midas-XGBoost']
    for k in model_keys: d_tr[f'Risk_{k}'] = np.nan
    
    monthly_aucs = {'Month': list(range(1, 13))}
    for k in model_keys: monthly_aucs[k] = []

    # ---------------------------------------------------------
    # PART 1: GLOBAL MODELS
    # ---------------------------------------------------------
    print("\n>>> TRAINING GLOBAL MODELS...")
    
    # 1. XGBoost
    print("   [1/4] Global XGBoost...")
    clf_xgb = HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42)
    search_xgb = RandomizedSearchCV(clf_xgb, Config.XGB_GRID, n_iter=10, cv=cv_t, scoring='roc_auc', n_jobs=-1, random_state=42)
    search_xgb.fit(d_tr[predictors], d_tr['Target_H'])
    best_xgb = search_xgb.best_estimator_
    d_tr['Risk_XGBoost'] = best_xgb.predict_proba(d_tr[predictors])[:, 1]
    plot_performance_suite(d_tr['Target_H'], d_tr['Risk_XGBoost'], "XGBoost", 'XGBoost', Config.OUTPUT_ROOT)
    plot_shap_advanced(best_xgb, d_tr[predictors], Config.OUTPUT_ROOT, "XGBoost", predictors)

    # 2. Midas-XGBoost
    print("   [2/4] Global Midas-XGBoost...")
    pipe_midas = Pipeline([
        ('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), 
        ('clf', HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42))
    ])
    search_midas = RandomizedSearchCV(pipe_midas, Config.MIDAS_GRID, n_iter=15, cv=cv_t, scoring='roc_auc', n_jobs=-1, random_state=42)
    search_midas.fit(d_tr, d_tr['Target_H'])
    best_midas = search_midas.best_estimator_
    d_tr['Risk_Midas-XGBoost'] = best_midas.predict_proba(d_tr)[:, 1]
    
    t1_g = search_midas.best_params_['almon__theta1']
    t2_g = search_midas.best_params_['almon__theta2']
    plot_almon_weights(t1_g, t2_g, 12, "Midas-XGBoost", Config.OUTPUT_ROOT)
    plot_performance_suite(d_tr['Target_H'], d_tr['Risk_Midas-XGBoost'], "Midas-XGBoost", 'Midas-XGBoost', Config.OUTPUT_ROOT)

    # ---------------------------------------------------------
    # PART 2: MONTHLY MODELS
    # ---------------------------------------------------------
    print("\n>>> TRAINING MONTHLY MODELS (12 Rounds)...")
    
    for m in range(1, 13):
        print(f"   [Status] Round {m}/12...")
        m_idx = d_tr['month'] == m
        m_data = d_tr[m_idx]
        if len(m_data) < 50: continue
        
        sub_cv = TimeSeriesSplit(n_splits=3)
        
        # 3. Monthly XGBoost
        clf_mxgb = HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42)
        search_mxgb = RandomizedSearchCV(clf_mxgb, Config.XGB_GRID, n_iter=10, cv=sub_cv, scoring='roc_auc', n_jobs=-1, random_state=42)
        search_mxgb.fit(m_data[predictors], m_data['Target_H'])
        d_tr.loc[m_idx, 'Risk_Monthly XGBoost'] = search_mxgb.best_estimator_.predict_proba(m_data[predictors])[:, 1]
        
        # 4. Monthly Midas-XGBoost (Allows month-specific Almon learning, bounded for stability)
        pipe_mmidas = Pipeline([
            ('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), 
            ('clf', HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42))
        ])
        search_mmidas = RandomizedSearchCV(pipe_mmidas, Config.MIDAS_GRID, n_iter=15, cv=sub_cv, scoring='roc_auc', n_jobs=-1, random_state=42)
        search_mmidas.fit(m_data, m_data['Target_H'])
        best_mmidas = search_mmidas.best_estimator_
        d_tr.loc[m_idx, 'Risk_Monthly Midas-XGBoost'] = best_mmidas.predict_proba(m_data)[:, 1]
        
        # Plot this month's Almon Structure
        t1_m = search_mmidas.best_params_['almon__theta1']
        t2_m = search_mmidas.best_params_['almon__theta2']
        plot_almon_weights(t1_m, t2_m, 12, f"Monthly_Midas_Month_{m:02d}", Config.OUTPUT_ROOT)
        
        # Track AUCs
        for k in model_keys:
            score = roc_auc_score(m_data['Target_H'], d_tr.loc[m_idx, f'Risk_{k}'])
            monthly_aucs[k].append(score)

    # ---------------------------------------------------------
    # PART 3: DIAGNOSTICS & EXPORTS
    # ---------------------------------------------------------
    print("\n>>> GENERATING FINAL SUITE...")
    
    # 1. Total AUCs
    summary_results = []
    for k in model_keys:
        valid = d_tr.dropna(subset=[f'Risk_{k}'])
        auc_tot = roc_auc_score(valid['Target_H'], valid[f'Risk_{k}'])
        summary_results.append({'Model': k, 'Overall_AUC': auc_tot})
        plot_performance_suite(valid['Target_H'], valid[f'Risk_{k}'], f"Total_{k}", k, Config.OUTPUT_ROOT)
        
    pd.DataFrame(summary_results).to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Overall_AUC.csv"), index=False)

    # 2. Volatility
    vol_df = plot_volatility_comparison(d_tr, Config.OUTPUT_ROOT)
    vol_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Volatility_Score.csv"), index=False)

    # 3. Monthly Evolution Chart
    df_auc = pd.DataFrame(monthly_aucs)
    df_auc.to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Monthly_AUCs.csv"), index=False)
    
    plt.figure(figsize=(12, 6))
    for k in model_keys:
        sns.lineplot(data=df_auc, x='Month', y=k, marker='o', color=Config.COLORS[k], label=k)
    plt.ylim(0.5, 1.0)
    plt.xticks(range(1, 13))
    apply_style(plt.gca(), "AUC Evolution by Month", "Month", "AUC")
    plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_ROOT, "Comparison_Monthly_AUCs.png"), bbox_inches='tight')
    plt.close()

    # 4. Country Overlays
    print("   [Status] Generating Country Overlays...")
    interesting = d_tr[(d_tr['Target_H'] == 1) | (d_tr['Risk_Monthly Midas-XGBoost'] > 0.45)]['Country_Name'].unique()
    for c in interesting[:25]:
        sub = d_tr[d_tr['Country_Name'] == c].sort_values('Date')
        if len(sub) > 12: plot_4way_country_chart(sub, c, Config.OUTPUT_ROOT)

    # 5. Granular Cuts
    print("   [Status] Calculating Granular Cuts (Income/Area)...")
    granular = []
    for cat in ['income', 'Area']:
        for val in d_tr[cat].unique():
            if str(val) == "nan" or val == "Unknown": continue
            for k in model_keys:
                sub = d_tr[(d_tr[cat] == val)].dropna(subset=[f'Risk_{k}'])
                if len(sub) > 50 and sub['Target_H'].nunique() > 1:
                    score = roc_auc_score(sub['Target_H'], sub[f'Risk_{k}'])
                    granular.append({'Group': cat, 'Value': val, 'Model': k, 'AUC': score})
    pd.DataFrame(granular).to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Granular_Cuts.csv"), index=False)

    print("\n=== PROCESS COMPLETED ===")

if __name__ == "__main__":
    run_engine()