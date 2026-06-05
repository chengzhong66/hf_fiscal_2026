

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 23:55:00 2026
@author: cmarsilli & Gemini
Sovereign Crisis Forecasting Engine v22.0 (MIDAS-XGBoost: Global Search + Monthly Transfer)

UPDATES:
- MIDAS ENGINE: Implemented AlmonMidasTransformer with Exponential Weights.
- HYPERPARAMS: theta1/theta2 estimated via Grid Search (Global).
- MONTHLY SUPPORT: Applies Optimal Global Lag Structure to Month-Specific models.
- PIPELINE: Full integration of Feature Engineering -> Modeling.
"""

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for file saving

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings

# --- Optional Imports ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
    OUTPUT_ROOT = "Output_v22_MIDAS_Monthly"
    MODELS_TO_RUN = ["Global", "Month-Specific"] 
    
    # Core Settings
    CV_TYPE = "Temporal"
    CV_FOLDS = 5
    HORIZON = 12
    TARGET = "precrisis"
    
    # Data & UI
    WINSORIZE_LIMITS = [0.01, 0.99]
    USE_NATIVE_IMPUTATION = True 
    
    # SHAP (Complexity 0-3)
    SHAP_COMPLEXITY = 1 
    SHAP_SAMPLES = 500
    
    CHART_SIZE = (10, 6)
    COLORS = {
        'DARK_RED': '#C0392B',    'DARK_BLUE': '#1F618D',    
        'GLOBAL_LINE': '#1F618D', 'MONTHLY_LINE': '#C0392B',
        'GREY': '#B0B0B0',        'CRISIS_SHADE': '#606060', 
        'OBS_LINE': '#2C3E50',    'PRED_LINE_MAIN': '#C0392B'
    }

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    # --- HYPERPARAMETERS ---
    # 1. XGBoost Grid
    XGB_GRID = {
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'max_iter': [300, 500, 800],
        'max_depth': [3, 4, 5, 6],
        'l2_regularization': [0.0, 1.0, 5.0],
        'max_leaf_nodes': [15, 31, 40]
    }

    # 2. Global Pipeline Grid (Includes Almon Lags)
    # The pipeline step is named 'midas', so we prefix vars with 'midas__'
    PIPELINE_GRID = {
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 4, 6],
        'clf__l2_regularization': [0.0, 1.0],
        
        # MIDAS Almon Params
        # Initial defaults as requested: lag=12, t1=-0.5, t2=0
        # But we search around them:
        'midas__theta1': [-0.5, -0.2, -0.05, -0.01],  # Slope (Decay)
        'midas__theta2': [-0.01, 0.0],                # Curvature
        'midas__max_lag': [12]                        # Fixed at 12 to save time, or [6, 12]
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

# =============================================================================
# 2. MIDAS TRANSFORMER (Exponential Almon)
# =============================================================================

class AlmonMidasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_lag=12, theta1=-0.5, theta2=0.0):
        self.max_lag = max_lag
        self.theta1 = theta1
        self.theta2 = theta2
        self.weights_ = None

    def fit(self, X, y=None):
        # 
        k = np.arange(1, self.max_lag + 1)
        w_raw = np.exp(self.theta1 * k + self.theta2 * (k**2))
        self.weights_ = w_raw / np.sum(w_raw) 
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_in = X.copy()
            feature_names = X_in.columns
        else:
            X_in = pd.DataFrame(X)
            feature_names = [f"var_{i}" for i in range(X.shape[1])]
            
        X_out = pd.DataFrame(index=X_in.index)
        
        # Re-calc weights (stateless transform)
        k = np.arange(1, self.max_lag + 1)
        w_raw = np.exp(self.theta1 * k + self.theta2 * (k**2))
        weights = w_raw / np.sum(w_raw)

        # Apply convolution
        for col in feature_names:
            accumulated_data = []
            for i, lag in enumerate(k):
                shifted = X_in[col].shift(lag)
                accumulated_data.append(shifted * weights[i])
            
            df_accum = pd.concat(accumulated_data, axis=1)
            # Require at least 1 valid lag to produce a value
            X_out[col] = df_accum.sum(axis=1, min_count=1) 
            
        return X_out

# =============================================================================
# 3. CHARTING & UTILS
# =============================================================================

def apply_style(ax, title, xlabel, ylabel, grid=True):
    ax.set_title(title, fontsize=14, fontweight='bold', color=Config.COLORS['OBS_LINE'])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid: ax.grid(axis='y', linestyle='--', alpha=0.5)

def plot_almon_weights(theta1, theta2, max_lag, out_path):
    k = np.arange(1, max_lag + 1)
    w_raw = np.exp(theta1 * k + theta2 * (k**2))
    w = w_raw / np.sum(w_raw)
    
    plt.figure(figsize=(6, 4))
    plt.bar(k, w, color=Config.COLORS['DARK_BLUE'])
    plt.plot(k, w, color='red', marker='o')
    apply_style(plt.gca(), f"Optimal Almon Weights\n(t1={theta1}, t2={theta2})", "Lag (Months)", "Weight")
    plt.savefig(os.path.join(out_path, "Almon_Weights_Profile.png"), bbox_inches='tight')
    plt.close()

def plot_performance_suite(y_true, y_score, label, out_path):
    if len(np.unique(y_true)) < 2: return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=Config.CHART_SIZE)
    plt.plot(fpr, tpr, color=Config.COLORS['DARK_RED'], lw=3, label=f'AUC: {auc(fpr, tpr):.3f}')
    plt.plot([0,1],[0,1], color=Config.COLORS['GREY'], linestyle='--')
    apply_style(plt.gca(), f"ROC: {label}", "FPR", "TPR")
    plt.legend()
    plt.savefig(os.path.join(out_path, f"ROC_{label.replace(' ','_')}.png"), bbox_inches='tight')
    plt.close()

def plot_comparisons(global_m_aucs, monthly_m_aucs, out_path):
    plt.figure(figsize=(12, 6))
    if global_m_aucs:
        g_df = pd.DataFrame(global_m_aucs)
        sns.lineplot(data=g_df, x='Month', y='AUC', marker='o', lw=2, color=Config.COLORS['GLOBAL_LINE'], label='Global Model')
    if monthly_m_aucs:
        m_df = pd.DataFrame(monthly_m_aucs)
        sns.lineplot(data=m_df, x='Month', y='AUC', marker='o', lw=3, color=Config.COLORS['MONTHLY_LINE'], label='Month-Specific Model')
    plt.xticks(range(1, 13)); plt.ylim(0.5, 1.0); plt.grid(True, linestyle='--', alpha=0.5)
    apply_style(plt.gca(), "Comparison: Monthly AUC Evolution", "Month", "AUC")
    plt.legend()
    plt.savefig(os.path.join(out_path, "Comparison_AUCs_Monthly_Evolution.png"))
    plt.close()

# =============================================================================
# 4. DATA PREP
# =============================================================================

def prepare_data():
    print("--- Loading and Preparing Data ---")
    df = pd.read_csv(Config.FILE_PATH)
    
    # Keys & Date
    if 'COUNTRY' in df.columns: df.rename(columns={'COUNTRY': 'Country'}, inplace=True)
    if 'COUNTRY_name' in df.columns: df.rename(columns={'COUNTRY_name': 'Country_Name'}, inplace=True)
    df['Country'] = pd.to_numeric(df['Country'], errors='coerce').fillna(0).astype(int)
    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    
    # Target
    df = df.sort_values(['Country', 'Date'])
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)

    # Predictors
    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    
    print("   [Status] Winsorizing...")
    for p in predictors:
        lower, upper = df[p].quantile(Config.WINSORIZE_LIMITS)
        df[p] = df[p].clip(lower, upper)
        if not Config.USE_NATIVE_IMPUTATION: df[p] = df[p].fillna(df[p].median())

    return df.dropna(subset=['Target_H']).copy(), df[df['Target_H'].isna()].copy(), predictors

# =============================================================================
# 5. ENGINE
# =============================================================================

def run_engine():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    print("\n=== MIDAS ENGINE START ===")
    d_tr_raw, d_fc_raw, preds = prepare_data()
    
    # Validation
    print(f"   [Status] Training Rows: {len(d_tr_raw)} | Crisis Count: {d_tr_raw['Target_H'].sum()}")
    if len(d_tr_raw) == 0: return

    cv_t = TimeSeriesSplit(n_splits=Config.CV_FOLDS)
    
    # Variables to store Global Optima for reuse in Monthly
    best_t1, best_t2, best_lag = -0.5, 0.0, 12 # Defaults
    global_monthly_aucs = []
    month_specific_aucs = []

    # --- 1. GLOBAL MODEL (Optimizes Thetas) ---
    if "Global" in Config.MODELS_TO_RUN:
        path = os.path.join(Config.OUTPUT_ROOT, "Global")
        os.makedirs(path, exist_ok=True)
        print("\n>>> Running: Global Model (Optimizing Lags)")
        
        pipe = Pipeline([
            ('midas', AlmonMidasTransformer()), 
            ('clf', HistGradientBoostingClassifier(random_state=42))
        ])
        
        print("   [Status] Grid Search for Optimal Almon Thetas...")
        search = RandomizedSearchCV(pipe, Config.PIPELINE_GRID, n_iter=15, cv=cv_t, scoring='roc_auc', n_jobs=-1, random_state=42)
        search.fit(d_tr_raw[preds], d_tr_raw['Target_H'])
        
        best_pipe = search.best_estimator_
        best_t1 = search.best_params_.get('midas__theta1')
        best_t2 = search.best_params_.get('midas__theta2')
        best_lag = search.best_params_.get('midas__max_lag')
        
        print(f"   [RESULT] Optimal Structure: MaxLag={best_lag}, Theta1={best_t1}, Theta2={best_t2}")
        plot_almon_weights(best_t1, best_t2, best_lag, path)
        
        # Predictions & Metrics
        d_tr_raw['Risk_Global'] = best_pipe.predict_proba(d_tr_raw[preds])[:, 1]
        auc_val = roc_auc_score(d_tr_raw['Target_H'], d_tr_raw['Risk_Global'])
        print(f"   > Global AUC: {auc_val:.3f}")
        plot_performance_suite(d_tr_raw['Target_H'], d_tr_raw['Risk_Global'], "Global_Total", path)
        
        # Breakdown for comparison
        for m in range(1, 13):
            sub = d_tr_raw[d_tr_raw['month'] == m]
            if len(sub) > 20:
                sc = roc_auc_score(sub['Target_H'], sub['Risk_Global'])
                global_monthly_aucs.append({'Month': m, 'AUC': sc})

    # --- 2. MONTH-SPECIFIC MODELS ---
    if "Month-Specific" in Config.MODELS_TO_RUN:
        path = os.path.join(Config.OUTPUT_ROOT, "Month-Specific")
        os.makedirs(path, exist_ok=True)
        print("\n>>> Running: Month-Specific Models")
        print(f"   [Status] Using Optimal Global Lags (t1={best_t1}, t2={best_t2}) to transform FULL history...")
        
        # A. TRANSFORM EVERYTHING ONCE (Preserves Lag History)
        # We must transform before splitting by month, otherwise Jan 2024 can't see Dec 2023
        midas_engine = AlmonMidasTransformer(max_lag=best_lag, theta1=best_t1, theta2=best_t2)
        
        # Transform Training Data
        X_tr_midas = midas_engine.fit_transform(d_tr_raw[preds])
        # Re-attach metadata for slicing
        X_tr_midas['month'] = d_tr_raw['month'].values
        X_tr_midas['Target_H'] = d_tr_raw['Target_H'].values
        
        midas_cols = [c for c in X_tr_midas.columns if c not in ['month', 'Target_H']]
        
        # B. LOOP MONTHS
        for m in range(1, 13):
            print(f"   [Status] Training Model for Month {m}/12...")
            
            # Slice the Pre-Transformed Data
            m_tr = X_tr_midas[X_tr_midas['month'] == m]
            if len(m_tr) < 30: continue
            
            # Tune Classifier Only (Lags are fixed from Global Best)
            sub_cv = TimeSeriesSplit(n_splits=3)
            clf = HistGradientBoostingClassifier(random_state=42)
            search = RandomizedSearchCV(clf, Config.XGB_GRID, n_iter=10, cv=sub_cv, scoring='roc_auc', n_jobs=-1)
            
            try:
                search.fit(m_tr[midas_cols], m_tr['Target_H'])
                best_m = search.best_estimator_
                
                # Predict
                probs = best_m.predict_proba(m_tr[midas_cols])[:, 1]
                m_auc = roc_auc_score(m_tr['Target_H'], probs)
                month_specific_aucs.append({'Month': m, 'AUC': m_auc})
                
                plot_performance_suite(m_tr['Target_H'], probs, f"Month_{m:02d}", path)
                
            except Exception as e:
                print(f"      [Error] Failed for Month {m}: {e}")

    # Final Comparison
    plot_comparisons(global_monthly_aucs, month_specific_aucs, Config.OUTPUT_ROOT)
    
    print("\n=== PROCESS COMPLETED ===")

if __name__ == "__main__":
    run_engine()