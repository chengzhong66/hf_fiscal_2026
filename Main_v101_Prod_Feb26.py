# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 2026
Sovereign Crisis Forecasting Engine (Modular Production & Historical Pipeline)
Based on v100.1b
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
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
# 1. CONFIGURATION AND ENGINE OPTIONS
# =============================================================================

class Config:
    OUTPUT_ROOT = "Output_v101_Modular_Production"
    
    # =========================================================================
    # --- USER SELECTIONS: PIPELINE TOGGLES ---
    # =========================================================================
    
    # Choose which models to execute across the pipeline.
    MODELS_TO_RUN = [
        'XGBoost', 
        'Monthly XGBoost', 
        'Midas-XGBoost', 
        'Monthly Midas-XGBoost',
        #'Logit', 
        #'Midas-Logit', 
        #'Lasso-Logit', 
        #'Midas-Lasso-Logit', 
        #'Monthly Logit', 
        #'Monthly Midas-Logit', 
        #'Monthly Lasso-Logit', 
        #'Monthly Midas-Lasso-Logit'
    ]

    # --- PIPELINE 1: HISTORICAL OUT-OF-SAMPLE (OOS) ANALYSIS ---
    RUN_HISTORICAL_OOS = False 
    
    # Specific historical episodes for zoom-in analysis. 
    # Format: ("Country_Name", "Start_Date", "End_Date") -> Use None for End_Date to go to latest input data.
    STRESS_EPISODES = [
        ("Albania", "1996-01-01", "2001-12-01"), ("Albania", "1995-01-01", "2001-12-01"),
        ("Brazil", "2003-01-01", "2008-12-01"), ("Cyprus", "2009-01-01", "2016-12-01"),
        ("Greece", "2007-01-01", "2019-12-01"), ("Lebanon", "2017-01-01", "2024-12-01"),
        ("Barbados", "2011-01-01", "2023-12-01"), ("Pakistan", "2005-01-01", "2017-03-01"),
        ("Portugal", "2008-01-01", "2015-12-01"),("Malaysia", "1995-01-01", "2000-12-01"), 
        ("Egypt", "2015-01-01", None), # Example of plotting up to the end of input data
    ]

    # --- PIPELINE 2: PRODUCTION FORECASTING & DECOMPOSITION ---
    RUN_PRODUCTION_FORECASTS = True 
    
    # Define which countries to generate production profiles and decompositions for.
    PROD_TARGET_COUNTRIES = "ALL" # Use "ALL" or a list like ["Brazil", "Greece"]
    
    # Factor Decomposition Parameters
    PROD_DECOMP_START_DATE = "2024-01-01"
    PROD_DECOMP_END_DATE = None  # Use None to plot up to the absolute latest available data row
    
    PROD_DECOMP_TOP_N = 5  # Number of top individual predictors to show
    PROD_DECOMP_CATEGORIES = ['Macro', 'Fiscal', 'Financial'] # Categories to aggregate
    
    # =========================================================================
    
    RANDOM_STATE = 851 
    EXPANDING_WINDOW_START = 2021 
    WALK_FORWARD_TYPE = "Expanding"
    ROLLING_WINDOW_YEARS = 25     
    
    PURGE_CV_OVERLAP = True   
    PURGE_OOS_OVERLAP = True  
    
    USE_MONOTONIC = True              
    USE_NATIVE_IMPUTATION = True      
    CV_TYPE = "Temporal"              
    CV_FOLDS = 5                      
    HORIZON = 12                      
    TARGET = "precrisis"              
    WINSORIZE_LIMITS = [0, 1]   
    
    SHAP_COMPLEXITY = 1               
    SHAP_SAMPLES = 50000              
    
    CHART_SIZE = (10,10) 
    
    COLORS = {
        'XGBoost': '#154360',               
        'Monthly XGBoost': '#5DADE2',       
        'Midas-XGBoost': '#7B241C',         
        'Monthly Midas-XGBoost': '#EC7063', 
        'Logit': '#8E44AD',                 
        'Monthly Logit': '#C39BD3',         
        'Midas-Logit': '#E67E22',           
        'Monthly Midas-Logit': '#F5B041',   
        'Lasso-Logit': '#27AE60',                 
        'Monthly Lasso-Logit': '#82E0AA',         
        'Midas-Lasso-Logit': '#16A085',           
        'Monthly Midas-Lasso-Logit': '#73C6B6',   
        'CRISIS_SHADE': '#D5D8DC',          
        'ACTUAL_CRISIS_SHADE': '#E6B0AA',
        'DARK_YELLOW': '#B7950B',           
        'BLACK': '#17202A',                 
        'Macro': '#5DADE2',                 
        'Fiscal': '#154360',                
        'Financial': '#B7950B',             
        'Others': '#D5D8DC'                 
    }

    LINE_WIDTHS = {k: 1.5 if 'Monthly' not in k else 3.0 for k in COLORS.keys()}

    Z_ORDERS = {
        'Monthly Lasso-Logit': 2, 
        'Monthly Logit': 2, 
        'Monthly Midas-Lasso-Logit': 2, 
        'Monthly Midas-Logit': 2,
        'Lasso-Logit': 3, 
        'Logit': 3, 
        'Midas-Lasso-Logit': 3, 
        'Midas-Logit': 3,
        
        'XGBoost': 7, 
        'Midas-XGBoost': 6,
        'Monthly XGBoost': 5,
        'Monthly Midas-XGBoost': 4 
    }

    DEP_PLOT_X_PCT = [2, 98]        
    DEP_PLOT_Y_PCT = [.5, 99.5]  
    DEP_PLOT_COLOR = COLORS['XGBoost']
    DEP_PLOT_ALPHA = 0.6        

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    XGB_GRID = {
        'learning_rate': [0.005, 0.01, 0.02],
        'max_iter': [300, 500, 800],
        'max_depth': [2, 3, 4, 5], 
        'l2_regularization': [1, 5.0, 25.0, 50.0], 
        'min_samples_leaf': [60, 80, 100]
    }

    MIDAS_GRID = {
        'clf__learning_rate': [0.005, 0.01, 0.02],
        'clf__max_depth': [2, 3, 4],
        'clf__l2_regularization': [5.0, 25.0, 50.0],  
        'almon__theta1': [-0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01], 
        'almon__theta2': [-0.005, -0.004, -0.003, -0.002, -0.001, 0.0]                
    }
    
    LINEAR_GRID = {'clf__C': [0.001, 0.01, 0.05]} 

    VARS = {
        "PCPI_PCH": ("Inflation", "Macro", 1),
        "gdp_growth": ("GDP Growth", "Macro", -1),
        "BoP_gdp": ("Current Account/GDP", "Macro", -1),
        "reserve_cover": ("FX Reserve Cover", "Macro", -1),
        "terms_of_trade": ("Terms of Trade", "Macro", -1),
        "GDP_percapita_over_US_12ma": ("GDP per Capita", "Macro", -1), 
        "oil_to_gdp": ("Oil Exports/GDP", "Macro", -1), 
        "oil_price_yoy": ("Oil Price YoY", "Macro", 0),
        "oil_shock_impact": ("Oil Shock Impact", "Macro", -1), 
        "govt_debt_gdp": ("Public Debt/GDP", "Fiscal", 1),
        "govt_deficit_gdp": ("Fiscal Balance/GDP", "Fiscal", -1),
        "govt_revenue_gdp": ("Fiscal Revenue/GDP", "Fiscal", -1),
        "tot_ext_debt_gdp": ("Total Ext Debt/GDP", "Fiscal", 1),
        "debt_service_gdp": ("Debt Service/GDP", "Fiscal", 1),
        "debt_fx_vulnerability": ("Debt FX Vuln", "Fiscal", 1), 
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

def compute_loess(x, y, frac=0.3):
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

def map_income_segment(inc):
    if pd.isna(inc) or inc == 'Unknown': return None
    inc_str = str(inc).upper()
    if 'AE' in inc_str or 'ADVANCED' in inc_str: return 'AE'
    if 'EM' in inc_str or 'EMERGING' in inc_str: return 'EM'
    if 'LIC' in inc_str or 'LOW' in inc_str: return 'LIC'
    return str(inc)

def plot_train_test_auc(df, models, holdout_year, out_path):
    p = os.path.join(out_path, "Diagnostics")
    os.makedirs(p, exist_ok=True)
    train_mask = df['Date'].dt.year < holdout_year
    d_train = df[train_mask]
    d_test = df[~train_mask]
    records = []
    for mod in models:
        col = f'Risk_{mod}'
        if col in df.columns:
            val_train = d_train.dropna(subset=[col, 'Target_H'])
            val_test = d_test.dropna(subset=[col, 'Target_H'])
            auc_train = roc_auc_score(val_train['Target_H'], val_train[col]) if val_train['Target_H'].nunique() > 1 else np.nan
            auc_test = roc_auc_score(val_test['Target_H'], val_test[col]) if val_test['Target_H'].nunique() > 1 else np.nan
            records.append({'Model': mod, 'Train AUC (Pre-Holdout)': auc_train, 'Test AUC (Walk-Forward OOS)': auc_test})
            
    res = pd.DataFrame(records)
    res.to_csv(os.path.join(out_path, "Results_OOS_WalkForward_Metrics.csv"), index=False)
    res_melt = res.melt(id_vars='Model', var_name='Dataset', value_name='AUC')
    
    plt.figure(figsize=Config.CHART_SIZE)
    sns.barplot(data=res_melt, x='Model', y='AUC', hue='Dataset', palette=[Config.COLORS['XGBoost'], Config.COLORS['Monthly Midas-XGBoost']])
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.axhline(0.5, color=Config.COLORS['BLACK'], linestyle='--', lw=1, zorder=1)
    apply_style(plt.gca(), f"{Config.WALK_FORWARD_TYPE} Window Generalization (Start: {holdout_year})", "Model Architecture", "AUC Score")
    plt.legend(loc='best')
    plt.savefig(os.path.join(p, "Diag_OOS_WalkForward_AUC.png"), bbox_inches='tight')
    plt.close('all')

def plot_all_countries_profiles(df, out_path, models_to_plot):
    p = os.path.join(out_path, "Historical_Backtest_Profiles")
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
        apply_style(plt.gca(), f"Historical Backtest Risk Profile: {country}", "Date", "Risk Probability")
        plt.legend(loc='best')
        clean_name = str(country).replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(p, f"Hist_Profile_{clean_name}.png"), bbox_inches='tight')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(p, f"Hist_Profile_{clean_name}_FixedY.png"), bbox_inches='tight')
        plt.close('all')

def plot_production_profiles(df_full, last_known_target_date, out_path, models_to_plot, chart_type='Early_Warning'):
    """
    Plots the true out-of-sample forecast. 
    chart_type can be 'Early_Warning' (shaded Target_H) or 'Actual_Crisis' (shaded unshifted Target_Actual)
    """
    if chart_type == 'Early_Warning':
        p = os.path.join(out_path, "Production_Forecasts", "Profiles_Early_Warning_Target")
        target_col = 'Target_H'
        shade_color = Config.COLORS['CRISIS_SHADE']
        shade_label = '12m Pre-Crisis Window'
        title_prefix = "Forecast vs Early Warning Target"
    else:
        p = os.path.join(out_path, "Production_Forecasts", "Profiles_Actual_Contemporaneous_Crisis")
        target_col = 'Target_Actual'
        shade_color = Config.COLORS['ACTUAL_CRISIS_SHADE']
        shade_label = 'Observed Crisis Event'
        title_prefix = "Forecast vs Actual Event"
        
    os.makedirs(p, exist_ok=True)
    
    if Config.PROD_TARGET_COUNTRIES == "ALL":
        countries = df_full['Country_Name'].unique()
    else:
        countries = [c for c in Config.PROD_TARGET_COUNTRIES if c in df_full['Country_Name'].values]
        
    for country in countries:
        sub = df_full[df_full['Country_Name'] == country].sort_values('Date')
        if len(sub) < 12: continue
        plt.figure(figsize=Config.CHART_SIZE)
        
        # Shade known historical events (stops where target becomes inherently unknown)
        known_mask = sub['Date'] <= last_known_target_date if target_col == 'Target_H' else sub['Date'] <= sub['Date'].max()
        plt.fill_between(sub.loc[known_mask, 'Date'], 0, 1, where=(sub.loc[known_mask, target_col] == 1), 
                         color=shade_color, alpha=0.6, label=shade_label, zorder=1)
        
        for mod in models_to_plot:
            if f'Prod_Risk_{mod}' in sub.columns:
                plt.plot(sub['Date'], sub[f'Prod_Risk_{mod}'], color=Config.COLORS.get(mod, '#5DADE2'), 
                         lw=Config.LINE_WIDTHS.get(mod, 2), label=f"Production {mod}", zorder=Config.Z_ORDERS.get(mod, 5))
                         
        plt.axvline(last_known_target_date, color='red', linestyle='--', lw=2, zorder=2, label='Forecast Horizon (Target Unknown)')
        apply_style(plt.gca(), f"{title_prefix}: {country}", "Date", "Risk Probability")
        plt.legend(loc='best')
        clean_name = str(country).replace('/', '_').replace(' ', '_')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(p, f"Prod_Profile_{chart_type}_{clean_name}.png"), bbox_inches='tight')
        plt.close('all')

def plot_episode_shap_deviations(df, country, start, end, model_label, out_path, model_or_dict, predictors, root_folder="Factor_Decomposition", top_n_features=5, categories=None):
    if not SHAP_AVAILABLE: return
    p = os.path.join(out_path, root_folder, model_label.replace(' ', '_'))
    os.makedirs(p, exist_ok=True)
    
    # Allow filtering up to the end of data if 'end' is None
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
    
    # Top N Features Chart
    top_n = min(top_n_features, len(predictors))
    top_indices = np.argsort(np.abs(sv_dev_scaled).mean(axis=0))[-top_n:]
    top_labels = [labels[i] for i in top_indices]
    
    df_feat = pd.DataFrame(sv_dev_scaled[:, top_indices], columns=top_labels, index=dates_str)
    df_feat['Others'] = sv_dev_scaled.sum(axis=1) - df_feat.sum(axis=1)
    
    cmap = plt.get_cmap('tab10')
    feat_colors = [cmap(i % 10) for i in range(top_n)] + [Config.COLORS['Others']]
    
    start_str = start[:4] if start else "Full"
    format_and_save_bar(df_feat, f"Top {top_n} Feature Deviations: {country}", f"Dev_Top_{top_n}_{country}_{start_str}.png", custom_colors=feat_colors)

    # Category Chart
    if categories is None:
        categories = ['Macro', 'Fiscal', 'Financial']
        
    cats = [Config.get_category(f) for f in predictors]
    df_cat = pd.DataFrame(index=dates_str)
    for cat_name in categories:
        cat_idx = [i for i, c in enumerate(cats) if c == cat_name]
        df_cat[cat_name] = sv_dev_scaled[:, cat_idx].sum(axis=1)
    df_cat = df_cat[categories] 
    format_and_save_bar(df_cat, f"Category Deviation: {country}", f"Dev_Category_{country}_{start_str}.png")

def plot_zoom_episodes(df, out_path, models_to_plot, episodes, prefix):
    p = os.path.join(out_path, "Diagnostics", "Zoom_Episodes")
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
        plt.savefig(os.path.join(p, f"Zoom_{prefix}_{country}_{start[:4]}_{end_str}.png"), bbox_inches='tight')
        plt.close('all')

def normalize_shap_values(sv):
    if isinstance(sv, list): return sv[1] if len(sv) == 2 else sv[0]
    elif sv.ndim == 3: return sv[:, :, 1]
    return sv

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
        if 'Country_Name_y' in df.columns:
            df['Country_Name'] = df['Country_Name_y'].fillna(df['Country_Name_x'])
            df.drop(columns=['Country_Name_x', 'Country_Name_y'], inplace=True)
        elif 'Country_Name_x' in df.columns:
            df.rename(columns={'Country_Name_x': 'Country_Name'}, inplace=True)
        df['income'] = df['income'].fillna("Unknown")
        df['Area'] = df['Area'].fillna("Unknown")
    
    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    df = df.sort_values(['Country', 'Date'])

    df['oil_price_yoy'] = df.groupby('Country')['oil_price'].diff(12)
    df['oil_shock_impact'] = df['oil_price_yoy'] * df['oil_to_gdp']
    df['debt_fx_vulnerability'] = df['ENDE_yoy'] * df['debt_service_gdp']
    
    # Preserve actual crisis target for contemporaneous production evaluation
    df['Target_Actual'] = df[Config.TARGET]
    
    # Target shift for early warning prediction (Creates NaNs at the end)
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
    print("\n=== ENGINE START (Modular Production & Backtesting) ===")
    df_full, d_tr, predictors, constraints = prepare_data()
    
    if len(d_tr) == 0: return
    
    cv_gap = Config.HORIZON if Config.PURGE_CV_OVERLAP else 0
    if Config.CV_TYPE == "Temporal":
        cv_t = TimeSeriesSplit(n_splits=Config.CV_FOLDS, gap=cv_gap)
    else:
        cv_t = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    models_all = Config.MODELS_TO_RUN 
    models_2way = [m for m in ['XGBoost', 'Monthly XGBoost'] if m in models_all]

    def train_iteration(d_train, d_pred):
        preds = {k: np.full(len(d_pred), np.nan) for k in models_all}
        xgb_kwargs = {'monotonic_cst': constraints if Config.USE_MONOTONIC else None, 'max_features': 0.75, 'early_stopping': True, 'validation_fraction': 0.1, 'n_iter_no_change': 20, 'random_state': Config.RANDOM_STATE}
        
        best_xgb = None
        if 'XGBoost' in models_all:
            clf_xgb = HistGradientBoostingClassifier(**xgb_kwargs)
            search_xgb = RandomizedSearchCV(clf_xgb, Config.XGB_GRID, n_iter=10, cv=cv_t, scoring='roc_auc', n_jobs=-1, random_state=Config.RANDOM_STATE)
            search_xgb.fit(d_train[predictors], d_train['Target_H'])
            best_xgb = search_xgb.best_estimator_
            if len(d_pred) > 0: preds['XGBoost'] = best_xgb.predict_proba(d_pred[predictors])[:, 1]

        best_midas = None
        best_xgb_theta1, best_xgb_theta2 = -0.5, 0.0
        if 'Midas-XGBoost' in models_all or 'Monthly Midas-XGBoost' in models_all:
            pipe_midas = Pipeline([('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), ('clf', HistGradientBoostingClassifier(**xgb_kwargs))])
            search_midas = RandomizedSearchCV(pipe_midas, Config.MIDAS_GRID, n_iter=15, cv=cv_t, scoring='roc_auc', n_jobs=-1, random_state=Config.RANDOM_STATE)
            search_midas.fit(d_train, d_train['Target_H'])
            best_midas = search_midas.best_estimator_
            best_xgb_theta1 = best_midas.named_steps['almon'].theta1
            best_xgb_theta2 = best_midas.named_steps['almon'].theta2
            if 'Midas-XGBoost' in models_all and len(d_pred) > 0: 
                preds['Midas-XGBoost'] = best_midas.predict_proba(d_pred)[:, 1]

        best_mxgb_dict = {}
        best_mmidas_dict = {}
        for m in range(1, 13):
            m_train_mask = d_train['month'] == m
            m_pred_mask = d_pred['month'] == m
            m_train = d_train[m_train_mask]
            m_pred_df = d_pred[m_pred_mask]
            if len(m_train) < 50: continue
            sub_cv = TimeSeriesSplit(n_splits=3, gap=cv_gap)
            
            if 'Monthly XGBoost' in models_all:
                clf_mxgb = HistGradientBoostingClassifier(**xgb_kwargs)
                search_mxgb = RandomizedSearchCV(clf_mxgb, Config.XGB_GRID, n_iter=10, cv=sub_cv, scoring='roc_auc', n_jobs=-1, random_state=Config.RANDOM_STATE)
                search_mxgb.fit(m_train[predictors], m_train['Target_H'])
                best_mxgb_dict[m] = search_mxgb.best_estimator_
                if len(m_pred_df) > 0: preds['Monthly XGBoost'][m_pred_mask] = best_mxgb_dict[m].predict_proba(m_pred_df[predictors])[:, 1]
            
            if 'Monthly Midas-XGBoost' in models_all:
                fixed_almon_xgb = AlmonValueCombiner(base_features=predictors, max_lag=12, theta1=best_xgb_theta1, theta2=best_xgb_theta2)
                pipe_mmidas = Pipeline([('almon', fixed_almon_xgb), ('clf', HistGradientBoostingClassifier(**xgb_kwargs))])
                mmidas_grid = {k: v for k, v in Config.MIDAS_GRID.items() if k.startswith('clf__')}
                search_mmidas = RandomizedSearchCV(pipe_mmidas, mmidas_grid, n_iter=15, cv=sub_cv, scoring='roc_auc', n_jobs=-1, random_state=Config.RANDOM_STATE)
                search_mmidas.fit(m_train, m_train['Target_H'])
                best_mmidas_dict[m] = search_mmidas.best_estimator_
                if len(m_pred_df) > 0: preds['Monthly Midas-XGBoost'][m_pred_mask] = best_mmidas_dict[m].predict_proba(m_pred_df)[:, 1]

        return preds, best_xgb, best_mxgb_dict, best_midas, best_mmidas_dict

    # =========================================================================
    # PIPELINE 1: HISTORICAL OOS VALIDATION
    # =========================================================================
    if Config.RUN_HISTORICAL_OOS:
        print("\n>>> PIPELINE 1: HISTORICAL WALK-FORWARD EVALUATION <<<")
        for k in models_all: d_tr[f'Risk_{k}'] = np.nan
        
        base_mask = d_tr['Date'].dt.year < Config.EXPANDING_WINDOW_START
        oos_years = sorted(d_tr[~base_mask]['Date'].dt.year.unique())

        print(f"   [Phase 1A] INITIAL TRAINING (Base IS: 1980-{Config.EXPANDING_WINDOW_START-1})")
        preds_is, final_hist_xgb, _, _, _ = train_iteration(d_tr[base_mask], d_tr[base_mask])
        for k in models_all: d_tr.loc[base_mask, f'Risk_{k}'] = preds_is[k]

        print(f"   [Phase 1B] {Config.WALK_FORWARD_TYPE.upper()} WINDOW OOS ({Config.EXPANDING_WINDOW_START}-{oos_years[-1]})")
        for y in oos_years:
            if Config.WALK_FORWARD_TYPE == "Rolling":
                start_year = y - Config.ROLLING_WINDOW_YEARS
                end_t_year = (y - 1) if Config.PURGE_OOS_OVERLAP else y
                print(f"      -> Re-tuning grids on {start_year}-{end_t_year-1} to predict {y}...")
                t_mask = (d_tr['Date'].dt.year < end_t_year) & (d_tr['Date'].dt.year >= start_year)
            else:
                end_t_year = (y - 1) if Config.PURGE_OOS_OVERLAP else y
                print(f"      -> Re-tuning grids on data up to {end_t_year-1} to predict {y}...")
                t_mask = d_tr['Date'].dt.year < end_t_year
                
            p_mask = d_tr['Date'].dt.year == y
            preds_oos, _, _, _, _ = train_iteration(d_tr[t_mask], d_tr[p_mask])
            for k in models_all:
                if len(d_tr[p_mask]) > 0: d_tr.loc[p_mask, f'Risk_{k}'] = preds_oos[k]
                
        # Generate Historical Diagnostics
        print("   [Status] Generating Historical Diagnostics & AUCs...")
        plot_train_test_auc(d_tr, models_all, Config.EXPANDING_WINDOW_START, Config.OUTPUT_ROOT)
        
        if len(models_2way) > 0:
            plot_all_countries_profiles(d_tr, Config.OUTPUT_ROOT, models_2way)
            plot_zoom_episodes(d_tr, Config.OUTPUT_ROOT, models_2way, Config.STRESS_EPISODES, "2Way")

    # =========================================================================
    # PIPELINE 2: LIVE PRODUCTION FORECASTING
    # =========================================================================
    if Config.RUN_PRODUCTION_FORECASTS:
        print("\n>>> PIPELINE 2: TRUE PRODUCTION PIPELINE (Training Master Models on All Data) <<<")
        print("   -> Predicting unobserved horizon...")
        
        for k in models_all: df_full[f'Prod_Risk_{k}'] = np.nan
        
        preds_prod, final_prod_xgb, final_prod_mxgb_dict, final_prod_midas, final_prod_mmidas_dict = train_iteration(d_tr, df_full)
        for k in models_all: df_full[f'Prod_Risk_{k}'] = preds_prod[k]
        
        # Export latest Snapshot
        latest_date = df_full['Date'].max()
        snapshot = df_full[df_full['Date'] == latest_date].copy()
        cols_to_keep = ['Country_Name', 'Date', 'income', 'Area'] + [f'Prod_Risk_{m}' for m in models_all]
        snapshot = snapshot[cols_to_keep].sort_values(by=f'Prod_Risk_{models_all[0]}', ascending=False)
        snapshot.to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Latest_Forecast_Snapshot.csv"), index=False)
        print(f"   -> Generated Latest Snapshot for {latest_date.strftime('%Y-%m')}")

        last_known_target = d_tr['Date'].max()
        if len(models_all) > 0:
            print("   [Status] Generating Dual Production Profiles...")
            plot_production_profiles(df_full, last_known_target, Config.OUTPUT_ROOT, models_all, chart_type='Early_Warning')
            plot_production_profiles(df_full, last_known_target, Config.OUTPUT_ROOT, models_all, chart_type='Actual_Crisis')
            
        # PRODUCTION FACTOR DECOMPOSITION
        if SHAP_AVAILABLE and final_prod_xgb:
            print(f"   [Status] Running Customizable Global SHAP Extraction...")
            
            if Config.PROD_TARGET_COUNTRIES == "ALL":
                target_countries = df_full['Country_Name'].unique()
            else:
                target_countries = [c for c in Config.PROD_TARGET_COUNTRIES if c in df_full['Country_Name'].values]
                
            for c in target_countries:
                plot_episode_shap_deviations(
                    df_full, 
                    country=c, 
                    start=Config.PROD_DECOMP_START_DATE, 
                    end=Config.PROD_DECOMP_END_DATE, 
                    model_label="Production_XGBoost", 
                    out_path=Config.OUTPUT_ROOT, 
                    model_or_dict=final_prod_xgb, 
                    predictors=predictors, 
                    root_folder="Production_Forecasts/Factor_Decomp",
                    top_n_features=Config.PROD_DECOMP_TOP_N,
                    categories=Config.PROD_DECOMP_CATEGORIES
                )

            print("   [Status] Generating SHAP Extracts for Specific Stress Episodes...")
            for c, start, end in Config.STRESS_EPISODES:
                plot_episode_shap_deviations(
                    df_full, 
                    country=c, 
                    start=start, 
                    end=end, 
                    model_label="Historical_XGBoost_Episodes", 
                    out_path=Config.OUTPUT_ROOT, 
                    model_or_dict=final_prod_xgb, 
                    predictors=predictors,
                    top_n_features=Config.PROD_DECOMP_TOP_N,
                    categories=Config.PROD_DECOMP_CATEGORIES
                )

    print("\n=== PROCESS COMPLETED ===")

if __name__ == "__main__":
    run_engine()