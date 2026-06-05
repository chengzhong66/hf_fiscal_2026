# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 03:15:00 2026
@author: cmarsilli & Gemini
Sovereign Crisis Forecasting Engine v26.0 (The Final Tournament)

ARCHITECTURAL CHANGE:
- To optimize lags on Month-Specific data (non-contiguous), we pre-calculate
  raw lags (1-12) globally, then use the AlmonCombiner to optimize the
  WEIGHTING of those lags inside the monthly Cross-Validation.

FEATURES:
- Tournament: Baseline vs. MIDAS per month.
- UI: Status indicators from v20.3.
- Charts: Almon Structure, SHAP Dependence, Country Comparisons, Flashing Red.
"""

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings
import re

# --- Optional Imports ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("   [System] SHAP not found. Feature importance plots skipped.")

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOWESS_AVAILABLE = True
except ImportError:
    LOWESS_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class Config:
    OUTPUT_ROOT = "Output_v26_Final_Tournament"
    
    # Core Settings
    CV_FOLDS = 5
    HORIZON = 12
    TARGET = "precrisis"
    WINSORIZE_LIMITS = [0.01, 0.99]
    USE_NATIVE_IMPUTATION = True
    
    # Analysis Settings
    GENERATE_COUNTRY_CHARTS = True
    SHAP_SAMPLES = 1000
    TOP_N_DEPENDENCE = 3
    
    COLORS = {
        'BASELINE': '#7F8C8D',    # Grey
        'MIDAS': '#C0392B',       # Red (Challenger)
        'CRISIS_SHADE': '#E5E7E9',
        'OBS_LINE': '#2C3E50',
        'ALMON_BAR': '#2980B9',
        'ALMON_LINE': '#E74C3C',
        'DARK_RED': '#C0392B',
        'DARK_BLUE': '#1F618D',
        'GREY': '#B0B0B0'
    }

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    # --- HYPERPARAMETERS ---
    
    # 1. Baseline Grid (Standard XGBoost)
    BASELINE_GRID = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [300, 500],
        'max_depth': [3, 4, 6],
        'l2_regularization': [0.0, 1.0]
    }

    # 2. MIDAS Grid (Almon Combiner + XGBoost)
    # We search for the best Lag Shape (Theta) and Tree Params simultaneously
    MIDAS_GRID = {
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 4, 6],
        'almon__theta1': [-0.5, -0.2, -0.05, 0.0],   # Slope
        'almon__theta2': [-0.01, 0.0],               # Curvature
        # We pre-calculate 12 lags. The model chooses how to weight them.
    }

    # Feature Metadata
    VARS = {
        "PCPI_PCH": ("Inflation", "Macro"),
        "gdp_growth": ("GDP Growth", "Macro"),
        "BoP_gdp": ("Current Account/GDP", "Macro"),
        "reserve_cover": ("FX Reserve Cover", "Macro"),
        "terms_of_trade": ("Terms of Trade", "Macro"),
        "GDP_percapita_over_US_12ma": ("GDP per Capita", "Macro"), 
        "oil_to_gdp": ("Oil Exports/GDP", "Macro"), 
        "govt_debt_gdp": ("Public Debt/GDP", "Fiscal"),
        "govt_deficit_gdp": ("Fiscal Balance/GDP", "Fiscal"),
        "tot_ext_debt_gdp": ("Total Ext Debt/GDP", "Fiscal"),
        "deposit_rate": ("ST Rate", "Financial"),
        "spread": ("Sovereign Spread", "Financial"), 
        "VIX": ("VIX Index", "Financial")
    }

# =============================================================================
# 2. ALMON COMBINER (The Weighting Engine)
# =============================================================================

class AlmonValueCombiner(BaseEstimator, TransformerMixin):
    """
    Takes a dataset containing raw lags (e.g., 'CPI_lag1', 'CPI_lag2'...)
    and combines them into a single feature using Exponential Almon Weights.
    This allows optimizing 'theta' inside CV even on non-contiguous rows.
    """
    def __init__(self, base_features=None, max_lag=12, theta1=-0.5, theta2=0.0):
        self.base_features = base_features # List of base names (e.g. "gdp_growth")
        self.max_lag = max_lag
        self.theta1 = theta1
        self.theta2 = theta2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        
        # Calculate Weights
        k = np.arange(1, self.max_lag + 1)
        w_raw = np.exp(self.theta1 * k + self.theta2 * (k**2))
        weights = w_raw / np.sum(w_raw)

        for base in self.base_features:
            # We assume columns exist: f"{base}_lag{k}"
            # Weighted Sum: Sum( w_k * X_lagk )
            
            # Fast Vectorized Implementation
            weighted_sum = 0
            valid_lags = 0
            
            for i, lag in enumerate(k):
                col_name = f"{base}_lag{lag}"
                if col_name in X.columns:
                    weighted_sum += X[col_name].fillna(0) * weights[i] 
                    # Note: We rely on XGBoost robustness or earlier imputation for NaNs.
                    # Here we treat NaN as 0 contribution to the weighted index 
                    # (effectively re-weighting remaining lags).
            
            X_out[base] = weighted_sum
            
        return X_out

# =============================================================================
# 3. CHARTING SUITE
# =============================================================================

def apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold', color='#333333')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

def plot_almon_weights(theta1, theta2, max_lag, month_label, out_path):
    k = np.arange(1, max_lag + 1)
    w_raw = np.exp(theta1 * k + theta2 * (k**2))
    w = w_raw / np.sum(w_raw)
    
    plt.figure(figsize=(6, 4))
    plt.bar(k, w, color=Config.COLORS['ALMON_BAR'], alpha=0.7)
    plt.plot(k, w, color=Config.COLORS['ALMON_LINE'], marker='o', lw=2)
    apply_style(plt.gca(), f"Memory Structure: {month_label}\n(t1={theta1}, t2={theta2})", "Lag (Months)", "Weight")
    plt.savefig(os.path.join(out_path, f"Almon_Weights_{month_label}.png"), bbox_inches='tight')
    plt.close()

def plot_tournament_roc(y_true, y_base, y_midas, label, out_path):
    if len(np.unique(y_true)) < 2: return 0, 0
    plt.figure(figsize=(10, 6))
    
    fpr_b, tpr_b, _ = roc_curve(y_true, y_base)
    auc_b = auc(fpr_b, tpr_b)
    plt.plot(fpr_b, tpr_b, color=Config.COLORS['BASELINE'], lw=2, linestyle='--', label=f'Baseline ({auc_b:.3f})')
    
    fpr_m, tpr_m, _ = roc_curve(y_true, y_midas)
    auc_m = auc(fpr_m, tpr_m)
    plt.plot(fpr_m, tpr_m, color=Config.COLORS['MIDAS'], lw=3, label=f'MIDAS ({auc_m:.3f})')
    
    plt.plot([0,1],[0,1], color='black', alpha=0.2, linestyle=':')
    apply_style(plt.gca(), f"Tournament ROC: {label}", "FPR", "TPR")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_path, f"ROC_{label}.png"), bbox_inches='tight')
    plt.close()
    return auc_b, auc_m

def plot_country_comparison(df, country_name, out_path, label):
    sub = df[df['Country_Name'] == country_name].sort_values('Date')
    if len(sub) < 12: return
    
    plt.figure(figsize=(12, 5))
    plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), 
                     color=Config.COLORS['CRISIS_SHADE'], alpha=0.5, label='Crisis Event')
    
    if 'Risk_Base' in sub.columns:
        plt.plot(sub['Date'], sub['Risk_Base'], color=Config.COLORS['BASELINE'], 
                 linestyle='--', lw=1.5, alpha=0.8, label='Baseline')
    if 'Risk_Midas' in sub.columns:
        plt.plot(sub['Date'], sub['Risk_Midas'], color=Config.COLORS['MIDAS'], 
                 lw=2.5, label='MIDAS')
    
    apply_style(plt.gca(), f"{country_name} ({label})", "Date", "Risk Probability")
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(out_path, f"{country_name}_{label}.png"), bbox_inches='tight')
    plt.close()

def compute_loess(x, y, frac=0.4):
    if not LOWESS_AVAILABLE: return None
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < 20: return None
    sorted_idxs = np.argsort(x_c)
    x_s, y_s = x_c[sorted_idxs], y_c[sorted_idxs]
    try:
        z = lowess(y_s, x_s, frac=frac, return_sorted=False)
        return x_s, z
    except: return None

def plot_shap_advanced(model, X, path, label):
    if not SHAP_AVAILABLE: return
    if len(X) < 10: return
    
    # Sample
    X_sub = X.sample(min(len(X), Config.SHAP_SAMPLES), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sub)
    if isinstance(sv, list): sv = sv[1]
    
    # 1. Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sub, show=False)
    plt.title(f"Feature Importance: {label}", fontsize=14)
    plt.savefig(os.path.join(path, f"SHAP_Beeswarm_{label}.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Dependence (Top N)
    global_imp = np.abs(sv).mean(0)
    top_indices = np.argsort(global_imp)[-Config.TOP_N_DEPENDENCE:]
    top_features = X_sub.columns[top_indices]
    
    for feat in top_features:
        x_vals = X_sub[feat].values
        feat_idx = X_sub.columns.get_loc(feat)
        y_vals = sv[:, feat_idx]
        
        plt.figure(figsize=(8, 5))
        plt.scatter(x_vals, y_vals, color=Config.COLORS['DARK_BLUE'], alpha=0.5, s=15, edgecolor='none')
        
        loess_res = compute_loess(x_vals, y_vals)
        if loess_res:
            plt.plot(loess_res[0], loess_res[1], color='red', lw=2, label='Trend')
            
        apply_style(plt.gca(), f"Dependence: {feat}", feat, "SHAP Value")
        plt.savefig(os.path.join(path, f"SHAP_Dep_{feat}_{label}.png"), bbox_inches='tight')
        plt.close()

def plot_flashing_red(df, out_path, label):
    # Uses the MIDAS risk if available, else Base
    risk_col = 'Risk_Midas' if 'Risk_Midas' in df.columns else 'Risk_Base'
    
    df = df.sort_values(['Country', 'Date'])
    df['Delta'] = df.groupby('Country')[risk_col].diff(12)
    alerts = df[(df['Delta'] > 0.15) & (df[risk_col] > 0.4) & (df['Date'] > '2020-01-01')]
    
    p = os.path.join(out_path, "Warnings")
    os.makedirs(p, exist_ok=True)
    
    for c in alerts['Country_Name'].unique()[:10]:
        sub = df[df['Country_Name'] == c]
        plt.figure(figsize=(10, 4))
        plt.plot(sub['Date'], sub[risk_col], color=Config.COLORS['DARK_RED'], lw=2)
        plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), color=Config.COLORS['CRISIS_SHADE'], alpha=0.4)
        apply_style(plt.gca(), f"WARNING: {c} ({label})", "Date", "Risk")
        plt.savefig(os.path.join(p, f"WARNING_{c}.png"), bbox_inches='tight')
        plt.close()

# =============================================================================
# 4. DATA PREP
# =============================================================================

def prepare_data():
    print("--- Loading and Preparing Data ---")
    df = pd.read_csv(Config.FILE_PATH)
    
    if 'COUNTRY' in df.columns: df.rename(columns={'COUNTRY': 'Country'}, inplace=True)
    df['Country'] = pd.to_numeric(df['Country'], errors='coerce').fillna(0).astype(int)
    
    if os.path.exists(Config.MAPPING_FILE):
        map_df = pd.read_csv(Config.MAPPING_FILE, encoding='latin1')
        map_df['IFS'] = pd.to_numeric(map_df['IFS'], errors='coerce').fillna(0).astype(int)
        df = df.merge(map_df[['IFS', 'Country_Name']], left_on='Country', right_on='IFS', how='left')
        if 'Country_Name_y' in df.columns:
            df['Country_Name'] = df['Country_Name_y'].fillna(df['Country_Name_x'])
    else:
        df['Country_Name'] = df['Country'].astype(str)
        
    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    
    # Target
    df = df.sort_values(['Country', 'Date'])
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)
    
    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    
    # Winsorize & Impute
    print("   [Status] Winsorizing...")
    for p in predictors:
        lower, upper = df[p].quantile(Config.WINSORIZE_LIMITS)
        df[p] = df[p].clip(lower, upper)
        if not Config.USE_NATIVE_IMPUTATION: df[p] = df[p].fillna(df[p].median())
    
    # --- CRITICAL: PRE-CALCULATE LAGS (1-12) ---
    # This allows Month-Specific loops to use history without needing contiguous rows
    print("   [Status] Pre-calculating Raw Lags (1-12) for MIDAS...")
    lag_cols = []
    for p in predictors:
        for lag in range(1, 13):
            col_name = f"{p}_lag{lag}"
            df[col_name] = df.groupby('Country')[p].shift(lag)
            lag_cols.append(col_name)
            
    # We must fill NaN lags because AlmonCombiner will sum them.
    # Simple strategy: Forward Fill or Median. 
    # For robustness in XGBoost, we'll fill with 0 (neutral impact if centered) or Median.
    # Let's use Median of the LAG column to avoid bias.
    for lc in lag_cols:
        df[lc] = df[lc].fillna(df[lc].median())

    return df.dropna(subset=['Target_H']).copy(), predictors

# =============================================================================
# 5. TOURNAMENT ENGINE
# =============================================================================

def run_tournament():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    print("\n=== STARTING SOVEREIGN CRISIS TOURNAMENT ===")
    
    df, predictors = prepare_data()
    print(f"   [Data] Rows: {len(df)} | Crisis Events: {df['Target_H'].sum()}")
    
    results = []

    # Iterate Months 1-12
    for m in range(1, 13):
        print(f"\n>>> ROUND {m}: Analyzing Month {m}...")
        
        m_df = df[df['month'] == m].copy()
        if len(m_df) < 50: 
            print("   [Skip] Insufficient data.")
            continue
            
        m_path = os.path.join(Config.OUTPUT_ROOT, f"Month_{m:02d}")
        os.makedirs(m_path, exist_ok=True)
        
        cv_t = TimeSeriesSplit(n_splits=3)
        
        # --- MODEL 1: BASELINE (Raw Features) ---
        clf_base = HistGradientBoostingClassifier(random_state=42)
        search_base = RandomizedSearchCV(clf_base, Config.BASELINE_GRID, n_iter=10, cv=cv_t, scoring='roc_auc', n_jobs=-1)
        search_base.fit(m_df[predictors], m_df['Target_H'])
        best_base = search_base.best_estimator_
        m_df['Risk_Base'] = best_base.predict_proba(m_df[predictors])[:, 1]
        
        # --- MODEL 2: MIDAS (Almon Weighted Lags) ---
        # Note: We pass the RAW LAG columns implicitly. The AlmonCombiner looks for them.
        # But sklearn Pipeline needs X to contain them. 'm_df' has them.
        pipe_midas = Pipeline([
            ('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), 
            ('clf', HistGradientBoostingClassifier(random_state=42))
        ])
        
        search_midas = RandomizedSearchCV(pipe_midas, Config.MIDAS_GRID, n_iter=15, cv=cv_t, scoring='roc_auc', n_jobs=-1)
        # We must pass the FULL m_df (with lag cols) to fit
        search_midas.fit(m_df, m_df['Target_H'])
        best_midas = search_midas.best_estimator_
        m_df['Risk_Midas'] = best_midas.predict_proba(m_df)[:, 1]
        
        # Extract Almon Weights for visualization
        best_t1 = search_midas.best_params_['almon__theta1']
        best_t2 = search_midas.best_params_['almon__theta2']
        plot_almon_weights(best_t1, best_t2, 12, f"Month_{m:02d}", m_path)
        
        # --- RESULTS ---
        auc_b, auc_m = plot_tournament_roc(m_df['Target_H'], m_df['Risk_Base'], m_df['Risk_Midas'], f"Month_{m:02d}", m_path)
        print(f"   [Scoreboard] Base: {auc_b:.3f} | MIDAS: {auc_m:.3f} | Delta: {auc_m-auc_b:+.3f}")
        results.append({'Month': m, 'AUC_Base': auc_b, 'AUC_Midas': auc_m, 'Winner': 'MIDAS' if auc_m > auc_b else 'Base'})
        
        # --- DIAGNOSTICS (Winner Takes All) ---
        winner_label = "MIDAS" if auc_m > auc_b else "Base"
        winner_model = best_midas if auc_m > auc_b else best_base
        
        # 1. SHAP
        if SHAP_AVAILABLE:
            if winner_label == "MIDAS":
                # Must transform first
                X_trans = winner_model.named_steps['almon'].transform(m_df)
                plot_shap_advanced(winner_model.named_steps['clf'], X_trans, m_path, winner_label)
            else:
                plot_shap_advanced(winner_model, m_df[predictors], m_path, winner_label)

        # 2. Country Charts (Overlay)
        # We need to map these month-specific predictions back to the full dataset for continuity?
        # No, "Month Specific" means we only judge risk once a year per country.
        # We will plot the dots on the chart.
        interesting = m_df[(m_df['Risk_Midas'] > 0.45) | (m_df['Target_H'] == 1)]['Country_Name'].unique()
        
        if Config.GENERATE_COUNTRY_CHARTS:
            c_path = os.path.join(m_path, "Countries")
            os.makedirs(c_path, exist_ok=True)
            for c in interesting[:15]:
                # Pull full history for context
                full_c = df[df['Country_Name'] == c].copy()
                # Merge current month predictions
                full_c = full_c.merge(m_df[['Date', 'Risk_Base', 'Risk_Midas']], on='Date', how='left', suffixes=('', '_new'))
                # Combine
                if 'Risk_Base_new' in full_c.columns:
                     full_c['Risk_Base'] = full_c['Risk_Base_new'].combine_first(full_c.get('Risk_Base', pd.Series([np.nan]*len(full_c))))
                     full_c['Risk_Midas'] = full_c['Risk_Midas_new'].combine_first(full_c.get('Risk_Midas', pd.Series([np.nan]*len(full_c))))
                
                plot_country_comparison(full_c, c, c_path, f"Month_{m:02d}")
        
        # 3. Flashing Red (Warnings)
        # We look for countries where risk jumped > 0.15 compared to last year (Month M-12)
        # Since m_df rows are 12 months apart per country, simple diff works
        plot_flashing_red(m_df, m_path, winner_label)

    # --- FINAL STANDINGS ---
    res_df = pd.DataFrame(results)
    print("\n=== FINAL STANDINGS ===")
    print(res_df)
    res_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "Tournament_Standings.csv"), index=False)
    
    # Win Rate Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(data=res_df, x='Month', y='AUC_Midas', color=Config.COLORS['MIDAS'], alpha=0.6, label='MIDAS')
    sns.scatterplot(data=res_df, x='Month', y='AUC_Base', color='black', marker='x', s=100, label='Baseline', zorder=10)
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.title("Tournament: MIDAS vs Baseline by Month")
    plt.savefig(os.path.join(Config.OUTPUT_ROOT, "Tournament_Summary.png"))
    plt.close()

if __name__ == "__main__":
    run_tournament()