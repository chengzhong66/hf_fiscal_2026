# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:05:29 2026

@author: cmarsilli
"""


import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import itertools
import textwrap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss, make_scorer
from sklearn.calibration import calibration_curve
from scipy.special import expit
from sklearn.preprocessing import QuantileTransformer
import warnings

# --- Global Font Styling ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

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

warnings.filterwarnings('ignore')

# =============================================================================
# 0. SAFE SCORER DEFINITION
# =============================================================================
def safe_auc(y_true, y_pred):
    # If the fold only has non-crisis or only crisis data, return a neutral score
    if len(np.unique(y_true)) < 2:
        return 0.5 
    return roc_auc_score(y_true, y_pred)

safe_auc_scorer = make_scorer(safe_auc, response_method='predict_proba')

# =============================================================================
# 1. CONFIGURATION AND ENGINE OPTIONS
# =============================================================================

class Config:
    OUTPUT_ROOT = "Output_v108_Flattened_Comparisons"
    
    # =========================================================================
    # --- 1. EXECUTION SELECTIONS & SAMPLE FILTER COMPARISONS ---
    # =========================================================================
    MODELS_TO_RUN = [
        'XGBoost', 
        'Monthly XGBoost', 
        'Midas-XGBoost', 
        'Monthly Midas-XGBoost',
        'Logit', 
        'Midas-Logit', 
        'Monthly Logit', 
        'Monthly Midas-Logit'
    ]

    # Define multiple setups to compare. The engine will loop through all of them
    # for the Historical OOS Backtest to generate AUC comparisons.
    FILTER_SETUPS = {
        "All_Countries": "ALL",
        "AE_Only": ["AE"],
        "EM_Only": ["EM"],
        "LIC_Only": ["LIC"]
    }
    
    # Select ONE setup from the dictionary above to run the full deep diagnostics,
    # production forecasting, and SHAP explainability.
    MAIN_SETUP = "All_Countries"

    # =========================================================================
    # --- 2. CHARTING SELECTIONS (SEPARATED BY FAMILY) ---
    # =========================================================================
    CHART_FAMILIES = {
        "XGBoost_Standard": ['XGBoost', 'Monthly XGBoost'],
        "XGBoost_Midas": ['Midas-XGBoost', 'Monthly Midas-XGBoost'],
        "Linear_Models": ['Logit', 'Midas-Logit', 'Monthly Logit', 'Monthly Midas-Logit']
    }
    
    # --- PIPELINE 1: HISTORICAL OUT-OF-SAMPLE (OOS) ANALYSIS ---
    RUN_HISTORICAL_OOS = True 
    
    STRESS_EPISODES = [
        ("Brazil", "2003-01-01", "2008-12-01"), ("Cyprus", "2009-01-01", "2016-12-01"),
        ("Greece", "2007-01-01", "2019-12-01"), ("Lebanon", "2017-01-01", "2024-12-01"),
        ("Egypt", "2015-01-01", None), 
    ]

    # --- PIPELINE 2: PRODUCTION FORECASTING & EXPLAINABILITY ---
    RUN_PRODUCTION_FORECASTS = True 
    PROD_TARGET_COUNTRIES = "ALL" 
    
    MODEL_FOR_SHAP = 'XGBoost'  
    PROD_DECOMP_START_DATE = "2007-01-01"
    PROD_DECOMP_END_DATE = None  
    PROD_DECOMP_TOP_N = 6  
    PROD_DECOMP_CATEGORIES = ['Macro', 'Fiscal', 'Financial'] 
    
    # Deep SHAP Settings
    # 1: Beeswarm, 2: Univariate Dependence (+ LOESS), 3: Bivariate Gradient Dependence (+ High/Low LOESS)
    SHAP_COMPLEXITY = 2                
    SHAP_ALPHA = 0.3
    SHAP_INTERACTION_PERCENTILES = [20, 80] 
    SHAP_X_LIMITS = {
        "PCPI_PCH": (-5,20),
        "gdp_growth": (-5,10),
        "BoP_gdp": (-20,20),
        "reserve_cover": (0,8),
        "terms_of_trade": (50,150),
        "GDP_percapita_over_US_12ma": (0,1.5), 
        "oil_to_gdp": (0,0.15),
        "govt_debt_gdp": (0,120),
        "govt_deficit_gdp": (-10,5),
        "govt_revenue_gdp": (0,60),
        "tot_ext_debt_gdp": (0,200),
        "debt_service_gdp": (0,25),
        "corruption_12ma": (-2,2.5),
        "deposit_rate": (0,21),
        "long_term_bond_yield": (0,20),
        #"WUI": (0,0.5),
        "ENDE_yoy": (-20,30),
        "spread": (0,1200), 
        "oil_price": (10,110),
        "VIX": (10,30)} # e.g. {'gdp_growth': (-10, 15)}
    SHAP_Y_LIMITS = {
        "PCPI_PCH": (-0.05,0.05),
        "gdp_growth": (-0.4,0.65),
        "BoP_gdp": (-0.3,0.35),
        "reserve_cover": (-0.4,0.8),
        "terms_of_trade": (-0.08,0.12),
        "GDP_percapita_over_US_12ma": (-2,1), 
        "oil_to_gdp": (-0.08,0.08), 
        "govt_debt_gdp": (-0.5,0.5),
        "govt_deficit_gdp": (-0.2,0.2),
        "govt_revenue_gdp": (-1,0.5),
        "tot_ext_debt_gdp": (-0.6,0.6),
        "debt_service_gdp": (-0.10,0.10),
        "corruption_12ma": (-0.3,0.3),
        "deposit_rate": (-0.2,0.8),
        "long_term_bond_yield": (-1.4,0.4),
        #"WUI": (-0.025,0.015),
        "ENDE_yoy": (-0.1,0.35),
        "spread": (-1,1.5), 
        "oil_price": (-0.05,0.05),
        "VIX": (-0.005,0.005)} 
    SHAP_X_PCT_LIMITS = (1, 99) 
    
    # --- PIPELINE 3: EVENT STUDY ANALYSIS ---
    RUN_EVENT_STUDY = True
    EVENT_STUDY_CONFIG = {
        'target_date': '2025-04-01',
        'income_group': 'EM',
        'spread_min': 100,
        'spread_max': 1000,
        'window_months': 12
    }
    
    # =========================================================================
    
    RANDOM_STATE = 851 
    EXPANDING_WINDOW_START = 2014 
    WALK_FORWARD_TYPE = "Expanding"
    ROLLING_WINDOW_YEARS = 35      
    PURGE_CV_OVERLAP = True   
    PURGE_OOS_OVERLAP = True  
    
    USE_MONOTONIC = True  #True              
    USE_NATIVE_IMPUTATION = True      
    CV_TYPE = "Temporal"              
    CV_FOLDS = 5                      
    HORIZON = 12                      
    TARGET = "precrisis"              
    WINSORIZE_LIMITS = [0, 1]   
    
    CHART_SIZE = (10,10) 
    
    COLORS = {
        # Core Models (Blues)
        'XGBoost': '#1A5276',               # Dark Blue          
        'Monthly XGBoost': '#5DADE2',       # Light Blue     
        # Midas Variants (Reds)
        'Midas-XGBoost': '#943126',         # Dark Red          
        'Monthly Midas-XGBoost': '#EC7063', # Light Red  
        # Linear Models (Purples)
        'Logit': '#5B2C6F',                 # Dark Purple                 
        'Monthly Logit': '#AF7AC5',         # Light Purple         
        'Midas-Logit': '#D35400',           # Dark Orange            
        'Monthly Midas-Logit': '#F5B041',   # Light Orange
        
        # UI Colors
        'CRISIS_SHADE': '#D5D8DC',          
        'ACTUAL_CRISIS_SHADE': '#E6B0AA',
        'BLACK': '#17202A',                 
        'Macro': '#5DADE2',                 
        'Fiscal': '#1A5276',                
        'Financial': '#D35400',              
        'Others': '#D5D8DC'                 
    }

    LINE_WIDTHS = {k: 1.5 if 'Monthly' not in k else 3.0 for k in COLORS.keys()}

    Z_ORDERS = {
        'Monthly Lasso-Logit': 2, 'Monthly Logit': 2, 'Monthly Midas-Lasso-Logit': 2, 'Monthly Midas-Logit': 2,
        'Lasso-Logit': 3, 'Logit': 3, 'Midas-Lasso-Logit': 3, 'Midas-Logit': 3,
        'XGBoost': 7, 'Midas-XGBoost': 6,
        'Monthly XGBoost': 5, 'Monthly Midas-XGBoost': 4 
    }

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    XGB_GRID = {
        'learning_rate': [0.005, 0.01],
        'max_iter': [300, 500],
        'max_depth': [1, 2, 3], # Lower depth for smoother, linear-like GAM behavior
        'l2_regularization': [10.0, 25.0, 50.0], # Heavier penalty
        'min_samples_leaf': [100, 150, 200] # Larger leaves
    }

    MIDAS_GRID = {
        'clf__learning_rate': [0.005, 0.01],
        'clf__max_depth': [1, 2, 3],
        'clf__l2_regularization': [10.0, 25.0, 50.0],  
        'almon__theta1': [-0.25, -0.1, -0.05], 
        'almon__theta2': [-0.5, -0.25, -0.1, -0.05]                
    }
    
    LINEAR_GRID = {'clf__C': [0.5,1,2,5]} 

    VARS = {
        "PCPI_PCH": ("Inflation", "Macro", 0),
        "gdp_growth": ("GDP Growth", "Macro", -1),
        "BoP_gdp": ("Current Account/GDP", "Macro", -1),
        "reserve_cover": ("FX Reserve Cover", "Macro", -1),
        "terms_of_trade": ("Terms of Trade", "Macro", -1),
        "GDP_percapita_over_US_12ma": ("GDP per Capita", "Macro", 0), 
        "oil_to_gdp": ("Oil Exports/GDP", "Macro", 0), 
        #"oil_price_yoy": ("Oil Price YoY", "Macro", 0),
        "oil_shock_impact": ("Oil Shock Impact", "Macro", 0), 
        "govt_debt_gdp": ("Public Debt/GDP", "Fiscal", 1),
        "govt_deficit_gdp": ("Fiscal Balance/GDP", "Fiscal", -1),
        "govt_revenue_gdp": ("Fiscal Revenue/GDP", "Fiscal", -1),
        "tot_ext_debt_gdp": ("Total Ext Debt/GDP", "Fiscal", 1),
        "debt_service_gdp": ("Debt Service/GDP", "Fiscal", 1),
        #"debt_fx_vulnerability": ("Debt FX Vuln", "Fiscal", 1), 
        "corruption_12ma": ("Corruption", "Fiscal", -1),
        "deposit_rate": ("ST Rate", "Financial", 0),
        "long_term_bond_yield": ("LT Rate", "Financial", 1),
        #"WUI": ("Uncertainty Idx", "Financial", 1),
        "ENDE_yoy": ("FX Depreciation", "Financial", 1),
        "spread": ("Sovereign Spread", "Financial", 1), 
        "oil_price": ("Oil Price", "Financial", 0),
        "VIX": ("VIX Index", "Financial", 1)
    }

    @classmethod
    def get_meta(cls, var): return cls.VARS.get(var, (var, "Macro", 0))
    @classmethod
    def get_label(cls, var): return cls.VARS.get(var, (var, "Macro", 0))[0]
    @classmethod
    def get_category(cls, var): return cls.VARS.get(var, (var, "Macro", 0))[1]

# =============================================================================
# 2. ALMON COMBINER 
# =============================================================================

class AlmonValueCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, base_features=None, max_lag=12, theta1=-0.5, theta2=0.0):
        self.base_features = base_features 
        self.max_lag = max_lag
        self.theta1 = theta1
        self.theta2 = theta2

    def fit(self, X, y=None): return self

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
# 3. ADVANCED CHARTING SUITE
# =============================================================================

def apply_style(ax, title, xlabel, ylabel, grid=False):
    ax.set_title(title, fontweight='bold', color=Config.COLORS['BLACK'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid: ax.grid(axis='y', linestyle='--', alpha=0.5)
    else: ax.grid(False)

def plot_filter_comparisons(records, out_path):
    p = os.path.join(out_path, "Metrics")
    os.makedirs(p, exist_ok=True)
    
    df_res = pd.DataFrame(records)
    df_res.to_csv(os.path.join(p, "Results_OOS_Filter_Comparisons.csv"), index=False)
    
    df_plot = df_res[df_res['Dataset'] == 'Test'].copy()
    if df_plot.empty: return
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Model', y='AUC', hue='Setup')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.axhline(0.5, color=Config.COLORS['BLACK'], linestyle='--', lw=1, zorder=1)
    apply_style(plt.gca(), f"Out-Of-Sample AUC Comparison Across Filter Setups", "Model Architecture", "Test AUC Score")
    plt.legend(title="Filter Setup", loc='best')
    plt.savefig(os.path.join(p, "Diag_Filter_Comparisons_AUC.png"), bbox_inches='tight')
    plt.close('all')

def plot_xgb_midas_auc_comparison(df_auc, out_path):
    p = os.path.join(out_path, "Metrics")
    os.makedirs(p, exist_ok=True)
    
    target_models = ['XGBoost', 'Monthly XGBoost', 'Midas-XGBoost', 'Monthly Midas-XGBoost']
    sub = df_auc[(df_auc['Dataset'] == 'Test') & (df_auc['Model'].isin(target_models))]
    if sub.empty: return
    
    avg_auc = sub.groupby('Model')['AUC'].mean().reset_index()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=avg_auc, x='Model', y='AUC', palette=[Config.COLORS.get(m, '#000') for m in avg_auc['Model']])
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=30, ha='right')
    plt.axhline(0.5, color='black', linestyle='--')
    apply_style(plt.gca(), "Out-of-Sample AUC: Standard vs MIDAS", "Model", "Average Monthly Test AUC")
    plt.savefig(os.path.join(p, "AUC_Comparison_XGB_vs_MIDAS.png"), bbox_inches='tight')
    plt.close('all')

def plot_almon_weights(trained_models_dict, out_path):
    p = os.path.join(out_path, "Explainability")
    os.makedirs(p, exist_ok=True)
    
    best_midas = trained_models_dict.get('Midas-XGBoost')
    best_mmidas_dict = trained_models_dict.get('Monthly Midas-XGBoost')
    if not best_midas and not best_mmidas_dict: return
    
    plt.figure(figsize=Config.CHART_SIZE)
    
    if best_midas and 'almon' in best_midas.named_steps:
        almon = best_midas.named_steps['almon']
        k = np.arange(1, almon.max_lag + 1)
        w_raw = np.exp(almon.theta1 * k + almon.theta2 * (k**2))
        weights = w_raw / np.sum(w_raw)
        plt.plot(k, weights, marker='o', color=Config.COLORS['Midas-XGBoost'], lw=3, label=f'Global Midas (\u03b81={almon.theta1:.3f})')
        
    if isinstance(best_mmidas_dict, dict):
        cmap = plt.get_cmap('viridis')
        for m in range(1, 13):
            if m in best_mmidas_dict and 'almon' in best_mmidas_dict[m].named_steps:
                almon_m = best_mmidas_dict[m].named_steps['almon']
                k = np.arange(1, almon_m.max_lag + 1)
                w_raw_m = np.exp(almon_m.theta1 * k + almon_m.theta2 * (k**2))
                weights_m = w_raw_m / np.sum(w_raw_m)
                plt.plot(k, weights_m, marker='.', color=cmap(m/12.0), lw=1, alpha=0.5, label=f'Month {m} (\u03b81={almon_m.theta1:.3f})')
                
    apply_style(plt.gca(), "Estimated Almon Lag Weights (XGBoost MIDAS)", "Lag (Months)", "Weight")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:13], labels[:13], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(p, "Almon_Weight_Structures.png"), bbox_inches='tight')
    plt.close('all')

def plot_monthly_auc_advanced(df_monthly, family_name, group_models, out_path):
    p = os.path.join(out_path, "Metrics")
    os.makedirs(p, exist_ok=True)
    incomes = df_monthly['Income'].unique()
    for inc in incomes:
        sub = df_monthly[(df_monthly['Income'] == inc) & (df_monthly['Model'].isin(group_models))].dropna(subset=['AUC'])
        if sub.empty: continue
        for ds in ['Train', 'Test']:
            sub_ds = sub[sub['Dataset'] == ds]
            if sub_ds.empty: continue
            plt.figure(figsize=Config.CHART_SIZE)
            ax = plt.gca()
            for mod in group_models:
                mod_data = sub_ds[sub_ds['Model'] == mod].sort_values('Month')
                if not mod_data.empty:
                    ax.plot(mod_data['Month'], mod_data['AUC'], marker='o', 
                             color=Config.COLORS.get(mod, '#000'), lw=Config.LINE_WIDTHS.get(mod, 2), 
                             label=f"{mod} (Avg: {mod_data['AUC'].mean():.3f})", zorder=Config.Z_ORDERS.get(mod, 3))
            ax.set_ylim(0.4, 1.0)
            ax.set_xticks(range(1, 13))
            ax.axhline(0.5, color=Config.COLORS['BLACK'], linestyle='--', lw=1, zorder=1)
            apply_style(ax, f"Monthly {ds} AUC Evolution: {inc} ({family_name})", "Month", "AUC Score", grid=True)
            ax.legend(loc='best', fontsize=10)
            clean_inc = str(inc).replace(' ', '_').replace('/', '_')
            plt.savefig(os.path.join(p, f"Monthly_AUC_{ds}_{clean_inc}_{family_name}.png"), bbox_inches='tight')
            plt.close('all')

def plot_comparisons(df, models_to_plot, family_name, out_path):
    p = os.path.join(out_path, "Metrics")
    os.makedirs(p, exist_ok=True)
    plt.figure(figsize=Config.CHART_SIZE)
    for mod in models_to_plot:
        col = f'Risk_{mod}'
        valid = df.dropna(subset=[col, 'Target_H'])
        if len(valid) < 10: continue
        fpr, tpr, _ = roc_curve(valid['Target_H'], valid[col])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=Config.COLORS.get(mod, '#000'), lw=Config.LINE_WIDTHS.get(mod, 2), 
                 label=f'{mod} (AUC: {auc_val:.3f})', zorder=Config.Z_ORDERS.get(mod, 3))
    plt.plot([0,1],[0,1], color=Config.COLORS['BLACK'], linestyle='--', alpha=0.2, zorder=1)
    apply_style(plt.gca(), f"ROC Curves ({family_name})", "False Positive Rate", "True Positive Rate")
    plt.legend(loc='best')
    plt.savefig(os.path.join(p, f"ROC_Curve_{family_name}.png"), bbox_inches='tight')
    plt.close('all')

def plot_risk_silhouette_distributions(df, models_to_plot, family_name, out_path):
    p = os.path.join(out_path, "Distributions")
    os.makedirs(p, exist_ok=True)
    df_oos = df[df['Date'].dt.year >= Config.EXPANDING_WINDOW_START].copy()
    if df_oos.empty: return
    
    incomes = ['All'] + [inc for inc in df_oos['income'].unique() if str(inc) not in ['Unknown', 'nan']]
    
    for mod in models_to_plot:
        col = f'Risk_{mod}'
        if col not in df_oos.columns: continue
        valid = df_oos.dropna(subset=[col, 'Target_H', 'income'])
        if valid.empty: continue
        
        for inc in incomes:
            sub = valid if inc == 'All' else valid[valid['income'] == inc]
            if sub.empty or sub['Target_H'].nunique() < 2: continue
            
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=sub[sub['Target_H'] == 0], x=col, fill=True, color='#5DADE2', alpha=0.5, label='Non-Crisis (0)', lw=1.5)
            sns.kdeplot(data=sub[sub['Target_H'] == 1], x=col, fill=True, color='#E67E22', alpha=0.5, label='Crisis (1)', lw=1.5)
            
            apply_style(plt.gca(), f"Risk Density by Crisis Status: {mod} ({inc})", "Predicted Risk Probability", "Density")
            plt.xlim(0, 1)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(p, f"Silhouette_CrisisVsNonCrisis_{inc}_{mod}_{family_name}.png"), bbox_inches='tight')
            plt.close('all')

def plot_average_risk_by_income(df, models_to_plot, family_name, out_path):
    p = os.path.join(out_path, "Distributions")
    os.makedirs(p, exist_ok=True)
    for mod in models_to_plot:
        col = f'Risk_{mod}'
        if col not in df.columns: continue
        valid = df.dropna(subset=[col, 'income'])
        if valid.empty: continue
        agg = valid[~valid['income'].isin(['Unknown', 'nan'])].groupby(['Date', 'income'])[col].mean().unstack()
        
        plt.figure(figsize=Config.CHART_SIZE)
        agg.plot(ax=plt.gca(), lw=2)
        apply_style(plt.gca(), f"Average Risk Index by Income: {mod}", "Date", "Mean Risk Probability")
        plt.legend(title="Income Group", loc='best')
        plt.savefig(os.path.join(p, f"Avg_Risk_Income_{mod}_{family_name}.png"), bbox_inches='tight')
        plt.close('all')

def plot_income_group_comparisons(df_full, models_to_plot, family_name, out_path):
    p = os.path.join(out_path, "Metrics")
    os.makedirs(p, exist_ok=True)
    
    records = []
    # Calculate AUC per income group
    for inc in df_full['income'].unique():
        if str(inc) in ['Unknown', 'nan']: continue
        sub = df_full[df_full['income'] == inc].dropna(subset=['Target_H'])
        if sub['Target_H'].nunique() < 2: continue # Skip if no crisis events
        
        for mod in models_to_plot:
            col = f'Risk_{mod}'
            if col in sub.columns:
                valid = sub.dropna(subset=[col])
                if len(valid) > 10 and valid['Target_H'].nunique() > 1:
                    score = roc_auc_score(valid['Target_H'], valid[col])
                    records.append({'Income': inc, 'Model': mod, 'AUC': score})
                    
    if not records: return
    df_plot = pd.DataFrame(records)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x='Income', y='AUC', hue='Model', 
                palette=[Config.COLORS.get(m, '#000') for m in df_plot['Model'].unique()])
    plt.ylim(0.5, 1.0)
    plt.axhline(0.5, color=Config.COLORS['BLACK'], linestyle='--', lw=1)
    apply_style(plt.gca(), f"Out-of-Sample AUC by Income Group ({family_name})", "Income Group", "AUC Score")
    plt.legend(title="Model", loc='lower right')
    plt.savefig(os.path.join(p, f"Diag_Income_AUC_{family_name}.png"), bbox_inches='tight')
    plt.close('all')

def plot_all_countries_profiles(df, out_path, models_to_plot, family_name):
    p = os.path.join(out_path, "Profiles")
    os.makedirs(p, exist_ok=True)
    countries = df['Country_Name'].unique()
    for country in countries:
        sub = df[df['Country_Name'] == country].sort_values('Date')
        if len(sub) < 12: continue
        plt.figure(figsize=Config.CHART_SIZE)
        plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), color=Config.COLORS['CRISIS_SHADE'], alpha=0.6, label='Crisis', zorder=1)
        for mod in models_to_plot:
            if f'Risk_{mod}' in sub.columns:
                plt.plot(sub['Date'], sub[f'Risk_{mod}'], color=Config.COLORS.get(mod, '#000'), 
                         lw=Config.LINE_WIDTHS.get(mod, 2), label=mod, zorder=Config.Z_ORDERS.get(mod, 3))
        plt.axvline(pd.to_datetime(f'{Config.EXPANDING_WINDOW_START}-01-01'), color=Config.COLORS['BLACK'], linestyle=':', lw=1, zorder=2, label='Walk-Forward Start')
        apply_style(plt.gca(), f"Backtest Risk Profile ({family_name}): {country}", "Date", "Risk Probability")
        plt.legend(loc='best')
        clean_name = str(country).replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(p, f"Hist_Profile_{clean_name}_{family_name}.png"), bbox_inches='tight')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(p, f"Hist_Profile_{clean_name}_FixedY_{family_name}.png"), bbox_inches='tight')
        plt.close('all')

def plot_production_profiles(df_full, last_known_target_date, out_path, models_to_plot, chart_type='Early_Warning', family_name=""):
    p = os.path.join(out_path, "Profiles")
    os.makedirs(p, exist_ok=True)
    
    if chart_type == 'Early_Warning':
        target_col = 'Target_H'
        shade_color = Config.COLORS['CRISIS_SHADE']
        shade_label = '12m Pre-Crisis Window'
        title_prefix = "Forecast vs Early Warning Target"
    else:
        target_col = 'Target_Actual'
        shade_color = Config.COLORS['ACTUAL_CRISIS_SHADE']
        shade_label = 'Observed Crisis Event'
        title_prefix = "Forecast vs Actual Event"
        
    if Config.PROD_TARGET_COUNTRIES == "ALL":
        countries = df_full['Country_Name'].unique()
    else:
        countries = [c for c in Config.PROD_TARGET_COUNTRIES if c in df_full['Country_Name'].values]
        
    for country in countries:
        sub = df_full[df_full['Country_Name'] == country].sort_values('Date')
        if len(sub) < 12: continue
        plt.figure(figsize=Config.CHART_SIZE)
        
        known_mask = sub['Date'] <= last_known_target_date if target_col == 'Target_H' else sub['Date'] <= sub['Date'].max()
        plt.fill_between(sub.loc[known_mask, 'Date'], 0, 1, where=(sub.loc[known_mask, target_col] == 1), 
                         color=shade_color, alpha=0.6, label=shade_label, zorder=1)
        
        for mod in models_to_plot:
            if f'Prod_Risk_{mod}' in sub.columns:
                plt.plot(sub['Date'], sub[f'Prod_Risk_{mod}'], color=Config.COLORS.get(mod, '#5DADE2'), 
                         lw=Config.LINE_WIDTHS.get(mod, 2), label=f"Production {mod}", zorder=Config.Z_ORDERS.get(mod, 5))
                         
        apply_style(plt.gca(), f"{title_prefix} ({family_name}): {country}", "Date", "Risk Probability")
        plt.legend(loc='best')
        clean_name = str(country).replace('/', '_').replace(' ', '_')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(p, f"Prod_Profile_{chart_type}_{clean_name}_{family_name}.png"), bbox_inches='tight')
        plt.close('all')

def apply_shap_limits(ax, feat_name, X_data):
    if feat_name in Config.SHAP_X_LIMITS:
        ax.set_xlim(Config.SHAP_X_LIMITS[feat_name])
    elif Config.SHAP_X_PCT_LIMITS:
        p_low, p_high = np.nanpercentile(X_data[feat_name], Config.SHAP_X_PCT_LIMITS)
        ax.set_xlim(p_low, p_high)
        
    if feat_name in Config.SHAP_Y_LIMITS:
        ax.set_ylim(Config.SHAP_Y_LIMITS[feat_name])

def compute_loess(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 10: return None
    return lowess(y_clean, x_clean, frac=0.6, return_sorted=True)

def plot_global_shap_explanations(df, model_or_dict, predictors, out_path, model_label, complexity=1):
    if not SHAP_AVAILABLE: return
    p = os.path.join(out_path, "Explainability")
    os.makedirs(p, exist_ok=True)

    if isinstance(model_or_dict, dict):
        X_list, sv_list = [], []
        for m in range(1, 13):
            mod = model_or_dict.get(m)
            if mod is not None:
                X_m = df[df['Date'].dt.month == m][predictors]
                if len(X_m) > 0:
                    exp = shap.TreeExplainer(mod)
                    sv = normalize_shap_values(exp.shap_values(X_m))
                    X_list.append(X_m)
                    sv_list.append(sv)
        if not X_list: return
        X = pd.concat(X_list)
        sv_combined = np.vstack(sv_list)
    else:
        X = df[predictors]
        exp = shap.TreeExplainer(model_or_dict)
        sv_combined = normalize_shap_values(exp.shap_values(X))

    # Level 1: Beeswarm Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_combined, X, show=False)
    plt.title(f"SHAP Global Summary (Beeswarm) - {model_label}", fontweight='bold')
    plt.savefig(os.path.join(p, f"SHAP_1_Beeswarm_Summary_{model_label}.png"), bbox_inches='tight')
    plt.close('all')

    # Level 2: Univariate Dependence + LOESS
    if complexity >= 2:
        for feat in predictors:
            f_idx = predictors.index(feat)
            x_vals = X[feat].values
            y_vals = sv_combined[:, f_idx]
            
            # Base Unicolor
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x_vals, y_vals, alpha=Config.SHAP_ALPHA, color=Config.COLORS['XGBoost'], s=20)
            apply_shap_limits(ax, feat, X)
            apply_style(ax, f"SHAP Dependence: {Config.get_label(feat)}", Config.get_label(feat), "SHAP Value")
            plt.savefig(os.path.join(p, f"SHAP_2A_Unicolor_{feat}_{model_label}.png"), bbox_inches='tight')
            plt.close('all')
            
            # Base + LOESS
            if LOWESS_AVAILABLE:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(x_vals, y_vals, alpha=Config.SHAP_ALPHA, color=Config.COLORS['XGBoost'], s=20)
                l_res = compute_loess(x_vals, y_vals)
                if l_res is not None:
                    ax.plot(l_res[:,0], l_res[:,1], color='red', lw=3, label='Global LOESS Trend')
                apply_shap_limits(ax, feat, X)
                apply_style(ax, f"SHAP LOESS: {Config.get_label(feat)}", Config.get_label(feat), "SHAP Value")
                ax.legend(loc='best')
                plt.savefig(os.path.join(p, f"SHAP_2B_LOESS_{feat}_{model_label}.png"), bbox_inches='tight')
                plt.close('all')

    # Level 3: Bivariate Gradient Dependence + High/Low LOESS
    if complexity >= 3:
        combinations = list(itertools.combinations(predictors, 2))
        for f1, f2 in combinations:
            f1_idx = predictors.index(f1)
            x_vals = X[f1].values
            y_vals = sv_combined[:, f1_idx] 
            c_vals = X[f2].values
            
            # Gradient Scatter
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap='viridis', alpha=Config.SHAP_ALPHA, s=20)
            plt.colorbar(sc, label=Config.get_label(f2))
            apply_shap_limits(ax, f1, X)
            apply_style(ax, f"Conditional Dependency: {Config.get_label(f1)}", Config.get_label(f1), "SHAP Value")
            plt.savefig(os.path.join(p, f"SHAP_3A_Gradient_{f1}_vs_{f2}_{model_label}.png"), bbox_inches='tight')
            plt.close('all')
            
            # Scatter + LOESS High/Low (Gradient removed here as requested)
            if LOWESS_AVAILABLE:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # CHANGED: Replaced gradient with unicolor for the LOESS chart to match Level 2A/2B
                ax.scatter(x_vals, y_vals, color=Config.COLORS['XGBoost'], alpha=Config.SHAP_ALPHA, s=20)
                
                pct_low, pct_high = Config.SHAP_INTERACTION_PERCENTILES
                v_low = np.nanpercentile(c_vals, pct_low)
                v_high = np.nanpercentile(c_vals, pct_high)
                
                mask_low = c_vals <= v_low
                mask_high = c_vals >= v_high
                
                l_low = compute_loess(x_vals[mask_low], y_vals[mask_low])
                if l_low is not None:
                    ax.plot(l_low[:,0], l_low[:,1], color='blue', lw=3, label=f'LOESS (<= {pct_low}th)')
                l_high = compute_loess(x_vals[mask_high], y_vals[mask_high])
                if l_high is not None:
                    ax.plot(l_high[:,0], l_high[:,1], color='red', lw=3, label=f'LOESS (>= {pct_high}th)')
                
                apply_shap_limits(ax, f1, X)
                apply_style(ax, f"Bifurcated Trends: {Config.get_label(f1)}", Config.get_label(f1), "SHAP Value")
                ax.legend(loc='best')
                plt.savefig(os.path.join(p, f"SHAP_3B_LOESS_Split_{f1}_vs_{f2}_{model_label}.png"), bbox_inches='tight')
                plt.close('all')

def plot_episode_shap_deviations(df, country, start, end, model_label, out_path, model_or_dict, predictors, root_folder="SHAP_Decompositions", top_n_features=5, categories=None):
    if not SHAP_AVAILABLE: return
    p = os.path.join(out_path, "Explainability")
    os.makedirs(p, exist_ok=True)
    
    mask = (df['Country_Name'] == country)
    if start is not None: mask &= (df['Date'] >= pd.to_datetime(start))
    if end is not None: mask &= (df['Date'] <= pd.to_datetime(end))
    
    sub = df[mask].sort_values('Date').copy()
    if len(sub) < 2: return
    
    risk_probs = np.zeros(len(sub))
    sv = np.zeros((len(sub), len(predictors)))
    
    if isinstance(model_or_dict, dict):
        explainers = {m: shap.TreeExplainer(mod) for m, mod in model_or_dict.items() if mod is not None}
        for idx, (i, row) in enumerate(sub.iterrows()):
            m = row['Date'].month
            if m not in explainers: continue 
            mod = model_or_dict[m]
            X_row = row[predictors].to_frame().T
            risk_probs[idx] = mod.predict_proba(X_row)[0, 1]
            sv_raw = normalize_shap_values(explainers[m].shap_values(X_row))
            sv[idx, :] = sv_raw[0]
    else:
        X = sub[predictors]
        risk_probs = model_or_dict.predict_proba(X)[:, 1]
        exp = shap.TreeExplainer(model_or_dict)
        sv = normalize_shap_values(exp.shap_values(X))
        
    risk_dev = risk_probs - risk_probs[0]
    sv_dev = sv - sv[0, :]
    sv_dev_sum = sv_dev.sum(axis=1)
    
    scaling_factor = np.divide(risk_dev, sv_dev_sum, out=np.zeros_like(risk_dev), where=sv_dev_sum!=0)
    sv_dev_scaled = sv_dev * scaling_factor[:, np.newaxis]
    
    dates_str = sub['Date'].dt.strftime('%Y-%m').tolist()

    def format_and_save_bar(df_bar, title, filename, custom_colors=None):
        fig, ax = plt.subplots(figsize=Config.CHART_SIZE)
        colors = custom_colors if custom_colors else [Config.COLORS.get(col, Config.COLORS['Others']) for col in df_bar.columns]
        df_bar.plot(kind='bar', stacked=True, color=colors, ax=ax, zorder=2, width=0.85)
        ax.plot(range(len(df_bar)), risk_dev, color=Config.COLORS['BLACK'], lw=2, marker='o', label='$\Delta$ Risk Index', zorder=4)
        if 'Target_H' in sub.columns:
            crisis_indices = np.where(sub['Target_H'] == 1)[0]
            for idx in crisis_indices:
                ax.axvspan(idx - 0.5, idx + 0.5, color=Config.COLORS['CRISIS_SHADE'], alpha=0.5, zorder=1, lw=0)
        plt.axhline(0, color=Config.COLORS['BLACK'], lw=0.8, zorder=3)
        tick_locs = []
        tick_labels = []
        for i, d in enumerate(sub['Date']):
            if d.month == 1 or i == 0 or i == len(sub)-1:
                if i not in tick_locs:
                    tick_locs.append(i)
                    tick_labels.append(d.strftime('%Y-%m'))
        ax.xaxis.set_ticks(tick_locs)
        ax.xaxis.set_ticklabels(tick_labels, rotation=45)
        apply_style(ax, title, "Date", "$\Delta$ Risk Probability Contribution (from $t_0$)")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        clean_name = str(filename).replace('/', '_')
        plt.savefig(os.path.join(p, clean_name), bbox_inches='tight')
        plt.close('all')

    labels = [Config.get_label(f) for f in predictors]
    
    top_n = min(top_n_features, len(predictors))
    top_indices = np.argsort(np.abs(sv_dev_scaled).mean(axis=0))[-top_n:]
    top_labels = [labels[i] for i in top_indices]
    
    df_feat = pd.DataFrame(sv_dev_scaled[:, top_indices], columns=top_labels, index=dates_str)
    df_feat['Others'] = sv_dev_scaled.sum(axis=1) - df_feat.sum(axis=1)
    
    cmap = plt.get_cmap('tab10')
    feat_colors = [cmap(i % 10) for i in range(top_n)] + [Config.COLORS['Others']]
    
    start_str = start[:4] if start else "Full"
    format_and_save_bar(df_feat, f"Top {top_n} Feature Deviations: {country}", f"Dev_Top_{top_n}_{country}_{start_str}_{model_label}.png", custom_colors=feat_colors)

    if categories is None:
        categories = ['Macro', 'Fiscal', 'Financial']
        
    cats = [Config.get_category(f) for f in predictors]
    df_cat = pd.DataFrame(index=dates_str)
    for cat_name in categories:
        cat_idx = [i for i, c in enumerate(cats) if c == cat_name]
        df_cat[cat_name] = sv_dev_scaled[:, cat_idx].sum(axis=1)
    df_cat = df_cat[categories] 
    format_and_save_bar(df_cat, f"Category Deviation: {country}", f"Dev_Category_{country}_{start_str}_{model_label}.png")

def plot_zoom_episodes(df, out_path, models_to_plot, episodes, prefix):
    p = os.path.join(out_path, "Profiles")
    os.makedirs(p, exist_ok=True)
    for country, start, end in episodes:
        mask = (df['Country_Name'] == country)
        if start: mask &= (df['Date'] >= pd.to_datetime(start))
        if end: mask &= (df['Date'] <= pd.to_datetime(end))
        sub = df[mask]
        if len(sub) < 5: continue
        
        plt.figure(figsize=Config.CHART_SIZE)
        plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), color=Config.COLORS['CRISIS_SHADE'], alpha=0.6, label='Crisis', zorder=1)
        for mod in models_to_plot:
            if f'Risk_{mod}' in sub.columns:
                plt.plot(sub['Date'], sub[f'Risk_{mod}'], color=Config.COLORS.get(mod, '#000'), 
                         lw=Config.LINE_WIDTHS.get(mod, 2), label=mod, zorder=Config.Z_ORDERS.get(mod, 3))
                         
        end_str = end[:4] if end else "End"
        apply_style(plt.gca(), f"Zoom: {country} ({start[:4]}-{end_str}) [{prefix}]", "Date", "Risk Probability")
        plt.legend(loc='best')
        plt.savefig(os.path.join(p, f"Zoom_{country}_{start[:4]}_{end_str}_{prefix}.png"), bbox_inches='tight')
        plt.close('all')

def plot_cross_country_event_study(df, model_or_dict, predictors, config_dict, out_path):
    print(f"   [Status] Running cross-country Event Study analysis for {config_dict['target_date']}...")
    p = os.path.join(out_path, "Event_Study")
    os.makedirs(p, exist_ok=True)
    
    t_date = pd.to_datetime(config_dict['target_date'])
    t0_df = df[df['Date'] == t_date]
    if t0_df.empty: return
    
    valid_t0 = t0_df[
        (t0_df['income'] == config_dict['income_group']) & 
        (t0_df['spread'] >= config_dict['spread_min']) & 
        (t0_df['spread'] <= config_dict['spread_max'])
    ]
    countries = valid_t0['Country_Name'].unique()
    if len(countries) == 0: 
        print("      -> No countries matched the event study criteria.")
        return
        
    window = config_dict['window_months']
    collected_data = []
    collected_shap = []
    
    exp = None
    if SHAP_AVAILABLE and not isinstance(model_or_dict, dict):
        exp = shap.TreeExplainer(model_or_dict)
        
    for c in countries:
        sub = df[(df['Country_Name'] == c) & (df['Date'] >= t_date - pd.DateOffset(months=window)) & (df['Date'] <= t_date + pd.DateOffset(months=window))].copy()
        if sub.empty: continue
        sub['T'] = ((sub['Date'] - t_date).dt.days / 30.44).round().astype(int)
        
        t0_row = sub[sub['T'] == 0]
        if t0_row.empty: continue
        
        sub_norm = pd.DataFrame({'T': sub['T']})
        for pred in predictors:
            val_0 = t0_row[pred].values[0]
            if abs(val_0) > 1e-4:
                sub_norm[pred] = (sub[pred] / val_0) * 100
            else:
                sub_norm[pred] = sub[pred] - val_0 + 100
        collected_data.append(sub_norm)
        
        if exp:
            sv = normalize_shap_values(exp.shap_values(sub[predictors]))
            sv_t0 = sv[sub['T'] == 0][0]
            sv_norm = (sv - sv_t0) + 100 
            sv_df = pd.DataFrame(sv_norm, columns=predictors)
            sv_df['T'] = sub['T'].values
            collected_shap.append(sv_df)
            
    footnote = "Countries included: " + ", ".join(countries)
    footnote_wrapped = "\n".join(textwrap.wrap(footnote, 100))
    
    def plot_avg(data_list, folder_name, title_prefix, ylabel):
        if not data_list: return
        merged = pd.concat(data_list)
        
        for col in predictors:
            if col not in merged.columns: continue
            fig, ax = plt.subplots(figsize=Config.CHART_SIZE)
            sns.lineplot(data=merged, x='T', y=col, ax=ax, color=Config.COLORS['XGBoost'], lw=3, label='Average Path')
            plt.axvline(0, color='red', linestyle='--')
            plt.axhline(100, color='black', linestyle=':', lw=1)
            apply_style(ax, f"{title_prefix}: {Config.get_label(col)}", "Months around Event (T=0)", ylabel)
            plt.figtext(0.5, -0.05, footnote_wrapped, wrap=True, horizontalalignment='center', fontsize=10, style='italic')
            plt.savefig(os.path.join(p, f"EventStudy_{folder_name}_{col}.png"), bbox_inches='tight')
            plt.close('all')

    plot_avg(collected_data, "Variables", "Variable Index (Base 100 at T=0)", "Index")
    if exp:
        plot_avg(collected_shap, "SHAP", "SHAP Contribution Index (Base 100 at T=0)", "SHAP Index")


def normalize_shap_values(sv):
    if isinstance(sv, list): return sv[1] if len(sv) == 2 else sv[0]
    elif sv.ndim == 3: return sv[:, :, 1]
    return sv

# =============================================================================
# 4. DATA PREP 
# =============================================================================

def prepare_data(income_filter="ALL"):
    print(f"--- Loading and Preparing Data (Filter: {income_filter}) ---")
    df = pd.read_csv(Config.FILE_PATH)
    
    if 'COUNTRY' in df.columns: df.rename(columns={'COUNTRY': 'Country'}, inplace=True)
    if 'COUNTRY_name' in df.columns: df.rename(columns={'COUNTRY_name': 'Country_Name'}, inplace=True)
    df['Country'] = pd.to_numeric(df['Country'], errors='coerce').fillna(0).astype(int)
    
    if os.path.exists(Config.MAPPING_FILE):
        map_df = pd.read_csv(Config.MAPPING_FILE, encoding='latin1')
        map_df['IFS'] = pd.to_numeric(map_df['IFS'], errors='coerce').fillna(0).astype(int)
        df = df.merge(map_df[['IFS', 'income', 'Area', 'Country_Name']], left_on='Country', right_on='IFS', how='left')
        if 'Country_Name_y' in df.columns:
            df['Country_Name'] = df['Country_Name_y'].fillna(df['Country_Name_x'])
            df.drop(columns=['Country_Name_x', 'Country_Name_y'], inplace=True)
        elif 'Country_Name_x' in df.columns:
            df.rename(columns={'Country_Name_x': 'Country_Name'}, inplace=True)
        df['income'] = df['income'].fillna("Unknown")
        df['Area'] = df['Area'].fillna("Unknown")
        
    if income_filter != "ALL":
        df = df[df['income'].isin(income_filter)].copy()
    
    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    df = df.sort_values(['Country', 'Date'])

    df['oil_price_yoy'] = df.groupby('Country')['oil_price'].diff(12)
    df['oil_shock_impact'] = df['oil_price'] * df['oil_to_gdp']
    df['debt_fx_vulnerability'] = df['ENDE_yoy'] * df['debt_service_gdp']
    
    df['Target_Actual'] = df[Config.TARGET]
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)

    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    for p in predictors:
        lower, upper = df[p].quantile(Config.WINSORIZE_LIMITS)
        df[p] = df[p].clip(lower, upper)
        if not Config.USE_NATIVE_IMPUTATION: df[p] = df[p].fillna(df[p].median())
        
    lag_cols = []
    for p in predictors:
        for lag in range(1, 13):
            col_name = f"{p}_lag{lag}"
            df[col_name] = df.groupby('Country')[p].shift(lag)
            lag_cols.append(col_name)
    for lc in lag_cols: df[lc] = df[lc].fillna(df[lc].median())

    df_full = df.copy() 
    df_valid = df.dropna(subset=['Target_H']).copy() 
    
    return df_full, df_valid, predictors, [Config.get_meta(p)[2] for p in predictors]

# =============================================================================
# 5. ENGINE
# =============================================================================

def run_engine():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    print("\n=== ENGINE START (Modular Execution & Family Charting) ===")
    
    models_all = Config.MODELS_TO_RUN 
    cv_gap = Config.HORIZON if Config.PURGE_CV_OVERLAP else 0
    if Config.CV_TYPE == "Temporal":
        cv_t = TimeSeriesSplit(n_splits=Config.CV_FOLDS, gap=cv_gap)
    else:
        cv_t = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

    def train_iteration(d_train, d_pred, predictors, constraints):
        preds = {k: np.full(len(d_pred), np.nan) for k in models_all}
        trained_models = {}
        xgb_kwargs = {'monotonic_cst': constraints if Config.USE_MONOTONIC else None, 'max_features': 0.75, 'early_stopping': True, 'validation_fraction': 0.1, 'n_iter_no_change': 20, 'random_state': Config.RANDOM_STATE}
        
        if 'XGBoost' in models_all:
            clf_xgb = HistGradientBoostingClassifier(**xgb_kwargs)
            search_xgb = RandomizedSearchCV(clf_xgb, Config.XGB_GRID, n_iter=10, cv=cv_t, scoring=safe_auc_scorer, n_jobs=-1, random_state=Config.RANDOM_STATE)
            search_xgb.fit(d_train[predictors], d_train['Target_H'])
            trained_models['XGBoost'] = search_xgb.best_estimator_
            if len(d_pred) > 0: preds['XGBoost'] = trained_models['XGBoost'].predict_proba(d_pred[predictors])[:, 1]

        best_xgb_theta1, best_xgb_theta2 = -0.5, 0.0
        if 'Midas-XGBoost' in models_all or 'Monthly Midas-XGBoost' in models_all:
            pipe_midas = Pipeline([('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), ('clf', HistGradientBoostingClassifier(**xgb_kwargs))])
            search_midas = RandomizedSearchCV(pipe_midas, Config.MIDAS_GRID, n_iter=15, cv=cv_t, scoring=safe_auc_scorer, n_jobs=-1, random_state=Config.RANDOM_STATE)
            search_midas.fit(d_train, d_train['Target_H'])
            trained_models['Midas-XGBoost'] = search_midas.best_estimator_
            
            best_xgb_theta1 = trained_models['Midas-XGBoost'].named_steps['almon'].theta1
            best_xgb_theta2 = trained_models['Midas-XGBoost'].named_steps['almon'].theta2
            if 'Midas-XGBoost' in models_all and len(d_pred) > 0: 
                preds['Midas-XGBoost'] = trained_models['Midas-XGBoost'].predict_proba(d_pred)[:, 1]

        if 'Logit' in models_all:
            pipe_l = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', QuantileTransformer(n_quantiles=100, random_state=Config.RANDOM_STATE)), ('clf', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=Config.RANDOM_STATE))])
            search_l = GridSearchCV(pipe_l, Config.LINEAR_GRID, cv=cv_t, scoring=safe_auc_scorer, n_jobs=-1)
            search_l.fit(d_train[predictors], d_train['Target_H'])
            trained_models['Logit'] = search_l.best_estimator_
            if len(d_pred) > 0: preds['Logit'] = trained_models['Logit'].predict_proba(d_pred[predictors])[:, 1]

        if 'Midas-Logit' in models_all:
            pipe_ml = Pipeline([('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), ('imp', SimpleImputer(strategy='median')), ('sc', QuantileTransformer(n_quantiles=100, random_state=Config.RANDOM_STATE)), ('clf', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=Config.RANDOM_STATE))])
            search_ml = GridSearchCV(pipe_ml, {'almon__theta1': [-0.2, -0.05], 'clf__C': Config.LINEAR_GRID['clf__C']}, cv=cv_t, scoring=safe_auc_scorer, n_jobs=-1)
            search_ml.fit(d_train, d_train['Target_H'])
            trained_models['Midas-Logit'] = search_ml.best_estimator_
            if len(d_pred) > 0: preds['Midas-Logit'] = trained_models['Midas-Logit'].predict_proba(d_pred)[:, 1]

        best_mxgb_dict = {}
        best_mmidas_dict = {}
        best_mlogit_dict = {}
        
        for m in range(1, 13):
            m_train_mask = d_train['month'] == m
            m_pred_mask = d_pred['month'] == m
            m_train = d_train[m_train_mask]
            m_pred_df = d_pred[m_pred_mask]
            if len(m_train) < 50 or m_train['Target_H'].nunique() < 2: continue
            sub_cv = TimeSeriesSplit(n_splits=3, gap=cv_gap)
            
            if 'Monthly XGBoost' in models_all:
                clf_mxgb = HistGradientBoostingClassifier(**xgb_kwargs)
                search_mxgb = RandomizedSearchCV(clf_mxgb, Config.XGB_GRID, n_iter=10, cv=sub_cv, scoring=safe_auc_scorer, n_jobs=-1, random_state=Config.RANDOM_STATE)
                search_mxgb.fit(m_train[predictors], m_train['Target_H'])
                best_mxgb_dict[m] = search_mxgb.best_estimator_
                if len(m_pred_df) > 0: preds['Monthly XGBoost'][m_pred_mask] = best_mxgb_dict[m].predict_proba(m_pred_df[predictors])[:, 1]
            
            if 'Monthly Midas-XGBoost' in models_all:
                fixed_almon_xgb = AlmonValueCombiner(base_features=predictors, max_lag=12, theta1=best_xgb_theta1, theta2=best_xgb_theta2)
                pipe_mmidas = Pipeline([('almon', fixed_almon_xgb), ('clf', HistGradientBoostingClassifier(**xgb_kwargs))])
                mmidas_grid = {k: v for k, v in Config.MIDAS_GRID.items() if k.startswith('clf__')}
                search_mmidas = RandomizedSearchCV(pipe_mmidas, mmidas_grid, n_iter=15, cv=sub_cv, scoring=safe_auc_scorer, n_jobs=-1, random_state=Config.RANDOM_STATE)
                search_mmidas.fit(m_train, m_train['Target_H'])
                best_mmidas_dict[m] = search_mmidas.best_estimator_
                if len(m_pred_df) > 0: preds['Monthly Midas-XGBoost'][m_pred_mask] = best_mmidas_dict[m].predict_proba(m_pred_df)[:, 1]
                
            if 'Monthly Logit' in models_all:
                search_mlg = GridSearchCV(Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', QuantileTransformer(n_quantiles=100, random_state=Config.RANDOM_STATE)), ('clf', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=Config.RANDOM_STATE))]), Config.LINEAR_GRID, cv=sub_cv, scoring=safe_auc_scorer, n_jobs=-1)
                search_mlg.fit(m_train[predictors], m_train['Target_H'])
                best_mlogit_dict[m] = search_mlg.best_estimator_
                if len(m_pred_df) > 0: preds['Monthly Logit'][m_pred_mask] = best_mlogit_dict[m].predict_proba(m_pred_df[predictors])[:, 1]
                
        if 'Monthly XGBoost' in models_all: trained_models['Monthly XGBoost'] = best_mxgb_dict
        if 'Monthly Midas-XGBoost' in models_all: trained_models['Monthly Midas-XGBoost'] = best_mmidas_dict
        if 'Monthly Logit' in models_all: trained_models['Monthly Logit'] = best_mlogit_dict

        return preds, trained_models

    # Initialize storage for cross-setup comparisons
    all_setup_auc_records = []

    # Iterate through all configured Filter Setups to gather OOS Backtest Comparisons
    for setup_name, income_filter in Config.FILTER_SETUPS.items():
        print(f"\n========================================================")
        print(f" >>> RUNNING SETUP PIPELINE: {setup_name} <<<")
        print(f"========================================================")
        
        df_full, d_tr, predictors, constraints = prepare_data(income_filter)
        if len(d_tr) == 0: 
            print("   [Warning] No data found for this setup.")
            continue

        if Config.RUN_HISTORICAL_OOS:
            for k in models_all: d_tr[f'Risk_{k}'] = np.nan
            
            base_mask = d_tr['Date'].dt.year < Config.EXPANDING_WINDOW_START
            oos_years = sorted(d_tr[~base_mask]['Date'].dt.year.unique())

            print(f"   [Phase 1A] INITIAL TRAINING (Base IS: 1980-{Config.EXPANDING_WINDOW_START-1})")
            preds_is, _ = train_iteration(d_tr[base_mask], d_tr[base_mask], predictors, constraints)
            for k in models_all: d_tr.loc[base_mask, f'Risk_{k}'] = preds_is[k]

            print(f"   [Phase 1B] {Config.WALK_FORWARD_TYPE.upper()} WINDOW OOS ({Config.EXPANDING_WINDOW_START}-{oos_years[-1]})")
            for y in oos_years:
                end_t_year = (y - 1) if Config.PURGE_OOS_OVERLAP else y
                t_mask = d_tr['Date'].dt.year < end_t_year
                if Config.WALK_FORWARD_TYPE == "Rolling":
                    t_mask &= (d_tr['Date'].dt.year >= (y - Config.ROLLING_WINDOW_YEARS))
                
                p_mask = d_tr['Date'].dt.year == y
                preds_oos, _ = train_iteration(d_tr[t_mask], d_tr[p_mask], predictors, constraints)
                for k in models_all:
                    if len(d_tr[p_mask]) > 0: d_tr.loc[p_mask, f'Risk_{k}'] = preds_oos[k]
                    
            # Collect OOS Records for this Setup
            train_mask = d_tr['Date'].dt.year < Config.EXPANDING_WINDOW_START
            d_train = d_tr[train_mask]
            d_test = d_tr[~train_mask]
            
            for mod in models_all:
                col = f'Risk_{mod}'
                if col in d_tr.columns:
                    val_train = d_train.dropna(subset=[col, 'Target_H'])
                    val_test = d_test.dropna(subset=[col, 'Target_H'])
                    auc_train = roc_auc_score(val_train['Target_H'], val_train[col]) if val_train['Target_H'].nunique() > 1 else np.nan
                    auc_test = roc_auc_score(val_test['Target_H'], val_test[col]) if val_test['Target_H'].nunique() > 1 else np.nan
                    all_setup_auc_records.append({'Setup': setup_name, 'Model': mod, 'Dataset': 'Train', 'AUC': auc_train})
                    all_setup_auc_records.append({'Setup': setup_name, 'Model': mod, 'Dataset': 'Test', 'AUC': auc_test})

        # Generate Deep Diagnostics & SHAP ONLY for the Primary setup
        if setup_name == Config.MAIN_SETUP:
            print("\n   >>> Generating Full Analytical Diagnostics & SHAP Extracts for MAIN SETUP <<<")
            
            # Sub-Extract Monthly AUC Data
            monthly_records = []
            for m in range(1, 13):
                for mod in models_all:
                    col = f'Risk_{mod}'
                    v_tr = d_train[(d_train['month'] == m)].dropna(subset=[col, 'Target_H'])
                    auc_tr = roc_auc_score(v_tr['Target_H'], v_tr[col]) if v_tr['Target_H'].nunique() > 1 else np.nan
                    monthly_records.append({'Month': m, 'Income': 'All', 'Model': mod, 'Dataset': 'Train', 'AUC': auc_tr})
                    v_te = d_test[(d_test['month'] == m)].dropna(subset=[col, 'Target_H'])
                    auc_te = roc_auc_score(v_te['Target_H'], v_te[col]) if v_te['Target_H'].nunique() > 1 else np.nan
                    monthly_records.append({'Month': m, 'Income': 'All', 'Model': mod, 'Dataset': 'Test', 'AUC': auc_te})
            df_monthly_auc = pd.DataFrame(monthly_records)
            plot_xgb_midas_auc_comparison(df_monthly_auc, Config.OUTPUT_ROOT)
            
            for family_name, family_models in Config.CHART_FAMILIES.items():
                valid_models = [m for m in family_models if m in models_all]
                if len(valid_models) > 0:
                    plot_monthly_auc_advanced(df_monthly_auc, family_name, valid_models, Config.OUTPUT_ROOT)
                    plot_comparisons(d_tr, valid_models, family_name, Config.OUTPUT_ROOT)
                    plot_income_group_comparisons(d_tr, valid_models, family_name, Config.OUTPUT_ROOT)
                    plot_risk_silhouette_distributions(d_tr, valid_models, family_name, Config.OUTPUT_ROOT)
                    plot_average_risk_by_income(d_tr, valid_models, family_name, Config.OUTPUT_ROOT)
                    plot_all_countries_profiles(d_tr, Config.OUTPUT_ROOT, valid_models, family_name)
                    plot_zoom_episodes(d_tr, Config.OUTPUT_ROOT, valid_models, Config.STRESS_EPISODES, family_name)

            if Config.RUN_PRODUCTION_FORECASTS:
                print("\n   >>> PIPELINE 2: TRUE PRODUCTION FORECASTING <<<")
                for k in models_all: df_full[f'Prod_Risk_{k}'] = np.nan
                
                preds_prod, final_prod_models = train_iteration(d_tr, df_full, predictors, constraints)
                for k in models_all: df_full[f'Prod_Risk_{k}'] = preds_prod[k]
                
                plot_almon_weights(final_prod_models, Config.OUTPUT_ROOT)
                
                latest_date = df_full['Date'].max()
                snapshot = df_full[df_full['Date'] == latest_date].copy()
                cols_to_keep = ['Country_Name', 'Date', 'income', 'Area'] + [f'Prod_Risk_{m}' for m in models_all]
                snapshot = snapshot[cols_to_keep].sort_values(by=f'Prod_Risk_{models_all[0]}', ascending=False)
                snapshot.to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Latest_Forecast_Snapshot.csv"), index=False)

                last_known_target = d_tr['Date'].max()
                
                for family_name, family_models in Config.CHART_FAMILIES.items():
                    valid_models_to_plot = [m for m in family_models if m in models_all]
                    if len(valid_models_to_plot) > 0:
                        plot_production_profiles(df_full, last_known_target, Config.OUTPUT_ROOT, valid_models_to_plot, chart_type='Early_Warning', family_name=family_name)
                        plot_production_profiles(df_full, last_known_target, Config.OUTPUT_ROOT, valid_models_to_plot, chart_type='Actual_Crisis', family_name=family_name)
                    
                model_to_explain = final_prod_models.get(Config.MODEL_FOR_SHAP)
                if SHAP_AVAILABLE and model_to_explain:
                    print(f"   [Status] Extracting Layered SHAP Interactions (Complexity {Config.SHAP_COMPLEXITY}) via [{Config.MODEL_FOR_SHAP}]...")
                    plot_global_shap_explanations(df_full, model_to_explain, predictors, Config.OUTPUT_ROOT, Config.MODEL_FOR_SHAP, complexity=Config.SHAP_COMPLEXITY)
                    
                    target_countries = df_full['Country_Name'].unique() if Config.PROD_TARGET_COUNTRIES == "ALL" else [c for c in Config.PROD_TARGET_COUNTRIES if c in df_full['Country_Name'].values]
                    for c in target_countries:
                        plot_episode_shap_deviations(df_full, country=c, start=Config.PROD_DECOMP_START_DATE, end=Config.PROD_DECOMP_END_DATE, model_label=f"Production_SHAP_{Config.MODEL_FOR_SHAP}", out_path=Config.OUTPUT_ROOT, model_or_dict=model_to_explain, predictors=predictors, root_folder="SHAP_Decompositions", top_n_features=Config.PROD_DECOMP_TOP_N, categories=Config.PROD_DECOMP_CATEGORIES)
                    for c, start, end in Config.STRESS_EPISODES:
                        plot_episode_shap_deviations(df_full, country=c, start=start, end=end, model_label=f"Historical_SHAP_{Config.MODEL_FOR_SHAP}", out_path=Config.OUTPUT_ROOT, model_or_dict=model_to_explain, predictors=predictors, top_n_features=Config.PROD_DECOMP_TOP_N, categories=Config.PROD_DECOMP_CATEGORIES)

                if Config.RUN_EVENT_STUDY:
                    plot_cross_country_event_study(df_full, model_to_explain, predictors, Config.EVENT_STUDY_CONFIG, Config.OUTPUT_ROOT)

    # Finally, Plot the cross-setup Filter Comparisons
    if len(all_setup_auc_records) > 0:
        plot_filter_comparisons(all_setup_auc_records, Config.OUTPUT_ROOT)
        
    print("\n=== PROCESS COMPLETED ===")

if __name__ == "__main__":
    run_engine()