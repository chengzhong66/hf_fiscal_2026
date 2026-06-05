
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:30:00 2026
@author: cmarsilli & Gemini
Sovereign Crisis Forecasting Engine v45.0 (Global Standards & Full Visualization Suite)

UPDATES:
- CHARTING: Increased font size universally for better readability.
- TIME SERIES: Added Crisis Frequency overlay to Income and All-Country Risk charts.
- EVENT STUDIES: Added T-24 to T+12 charts for raw variables (standardized globally).
- EVENT STUDIES: Added average risk trajectory grouped by Income.
- DEVIATION CHARTS: Colored the feature deviation stacked bars; legend set to "Risk Index".
- VOLATILITY: Restored M-o-M stability comparison charts (2Way & 4Way).
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
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.special import expit
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
# 1. CONFIGURATION
# =============================================================================

class Config:
    OUTPUT_ROOT = "Output_v45b"
    
    USE_MONOTONIC = True
    USE_NATIVE_IMPUTATION = True
    CV_TYPE = "Temporal"
    CV_FOLDS = 5
    HORIZON = 12
    TARGET = "precrisis"
    WINSORIZE_LIMITS = [0.01, 0.99]
    
    SHAP_COMPLEXITY = 2  
    SHAP_SAMPLES = 10000
    
    CHART_SIZE = (10, 10) 
    
    COLORS = {
        'XGBoost': '#154360',               # Dark Blue 
        'Monthly XGBoost': '#5DADE2',       # Light Blue 
        'Midas-XGBoost': '#7B241C',         # Dark Red 
        'Monthly Midas-XGBoost': '#EC7063', # Light Red 
        'CRISIS_SHADE': '#D5D8DC',          # Grey
        'DARK_YELLOW': '#B7950B',           # Dark Yellow
        'BLACK': '#17202A',                 # Black
        
        'Macro': '#5DADE2',                 # Light Blue
        'Fiscal': '#154360',                # Dark Blue
        'Financial': '#B7950B',             # Dark Yellow
        'Others': '#D5D8DC'                 # Grey
    }

    LINE_WIDTHS = {
        'XGBoost': 1.5,               
        'Monthly XGBoost': 3.0,       
        'Midas-XGBoost': 1.5,         
        'Monthly Midas-XGBoost': 3.0  
    }

    DEP_PLOT_X_PCT = [2, 98]        
    DEP_PLOT_Y_PCT = [.5, 99.5]  
    DEP_PLOT_COLOR = COLORS['XGBoost']
    DEP_PLOT_ALPHA = 0.6        

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    XGB_GRID = {
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'max_iter': [300, 500, 800],
        'max_depth': [3, 4, 5, 6],
        'l2_regularization': [0.0, 1.0, 5.0, 15.0],
        'min_samples_leaf': [20, 40]
    }

    MIDAS_GRID = {
        'clf__learning_rate': [0.01, 0.02, 0.05],
        'clf__max_depth': [3, 4, 5],
        'clf__l2_regularization': [1.0, 5.0, 15.0],  
        'almon__theta1': [-0.2, -0.05, -0.01], 
        'almon__theta2': [-0.005, 0.0]                
    }

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

def plot_volatility_comparison(df, models_to_plot, prefix, out_path):
    p = os.path.join(out_path, "Comparisons")
    os.makedirs(p, exist_ok=True)
    df = df.sort_values(['Country', 'Date'])
    metrics = []
    for mod in models_to_plot:
        col = f'Risk_{mod}'
        if col in df.columns:
            vol = df.groupby('Country')[col].diff().abs().mean()
            metrics.append({'Model': mod, 'Volatility': vol})
    res = pd.DataFrame(metrics)
    plt.figure(figsize=Config.CHART_SIZE)
    sns.barplot(data=res, x='Model', y='Volatility', palette=[Config.COLORS.get(m, Config.COLORS['BLACK']) for m in res['Model']])
    apply_style(plt.gca(), f"{prefix}: Stability Comparison (Lower is Smoother)", "Model Architecture", "Avg M-o-M Risk Change")
    plt.savefig(os.path.join(p, f"{prefix}_Volatility_Score.png"), bbox_inches='tight')
    plt.close()
    return res

def plot_performance_suite(y_true, y_score, label, out_path):
    if len(np.unique(y_true)) < 2: return
    clean = label.replace(" ", "_").replace("/", "_")
    p = os.path.join(out_path, "Diagnostics")
    os.makedirs(p, exist_ok=True)
    plt.figure(figsize=Config.CHART_SIZE)
    try:
        sns.kdeplot(y_score[y_true==0], label='No Crisis', fill=True, color=Config.COLORS['Monthly XGBoost'], alpha=0.3)
        sns.kdeplot(y_score[y_true==1], label='Crisis', fill=True, color=Config.COLORS['Midas-XGBoost'], alpha=0.3)
        auc_val = roc_auc_score(y_true, y_score)
        apply_style(plt.gca(), f"Silhouette: {label} (AUC: {auc_val:.3f})", "Predicted Risk Score", "Density")
        plt.legend()
        plt.savefig(os.path.join(p, f"Diag_Silhouette_{clean}.png"), bbox_inches='tight')
    except: pass
    plt.close()

def plot_income_silhouettes(df, models_to_plot, out_path):
    p = os.path.join(out_path, "Diagnostics", "Income_Silhouettes")
    os.makedirs(p, exist_ok=True)
    valid_df = df[~df['income'].isin(['Unknown', 'nan'])].dropna(subset=['income', 'Target_H'])
    for grp in valid_df['income'].unique():
        sub = valid_df[valid_df['income'] == grp]
        if len(np.unique(sub['Target_H'])) < 2: continue
        for mod in models_to_plot:
            col = f'Risk_{mod}'
            if col not in sub.columns or sub[col].isna().all(): continue
            plt.figure(figsize=Config.CHART_SIZE)
            try:
                sns.kdeplot(sub[sub['Target_H']==0][col], label='No Crisis', fill=True, color=Config.COLORS['Monthly XGBoost'], alpha=0.3)
                sns.kdeplot(sub[sub['Target_H']==1][col], label='Crisis', fill=True, color=Config.COLORS['Midas-XGBoost'], alpha=0.3)
                auc_val = roc_auc_score(sub['Target_H'], sub[col])
                apply_style(plt.gca(), f"Silhouette: {grp} - {mod} (AUC: {auc_val:.3f})", "Predicted Risk Score", "Density")
                plt.legend()
                plt.savefig(os.path.join(p, f"Silhouette_{grp.replace(' ','_')}_{mod.replace(' ','_')}.png"), bbox_inches='tight')
            except: pass
            plt.close()

def plot_comparisons(df, models_to_plot, prefix, out_path):
    p = os.path.join(out_path, "Comparisons")
    os.makedirs(p, exist_ok=True)
    
    # 1. ROC Curve
    plt.figure(figsize=Config.CHART_SIZE)
    for mod in models_to_plot:
        col = f'Risk_{mod}'
        valid = df.dropna(subset=[col, 'Target_H'])
        if len(valid) < 10: continue
        fpr, tpr, _ = roc_curve(valid['Target_H'], valid[col])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=Config.COLORS[mod], lw=Config.LINE_WIDTHS[mod], label=f'{mod} (AUC: {auc_val:.3f})')
    plt.plot([0,1],[0,1], color=Config.COLORS['BLACK'], linestyle='--', alpha=0.2)
    apply_style(plt.gca(), f"{prefix}: ROC Curves", "False Positive Rate", "True Positive Rate")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(p, f"{prefix}_ROC.png"), bbox_inches='tight')
    plt.close()

    # 2. Precision-Recall
    plt.figure(figsize=Config.CHART_SIZE)
    for mod in models_to_plot:
        col = f'Risk_{mod}'
        valid = df.dropna(subset=[col, 'Target_H'])
        if len(valid) < 10: continue
        precision, recall, _ = precision_recall_curve(valid['Target_H'], valid[col])
        ap = average_precision_score(valid['Target_H'], valid[col])
        plt.plot(recall, precision, color=Config.COLORS[mod], lw=Config.LINE_WIDTHS[mod], label=f'{mod} (AP: {ap:.3f})')
    baseline = df['Target_H'].mean()
    plt.axhline(y=baseline, color=Config.COLORS['BLACK'], linestyle=':', label=f'Random ({baseline:.2f})')
    apply_style(plt.gca(), f"{prefix}: Precision-Recall", "Recall (Sensitivity)", "Precision (PPV)")
    plt.legend()
    plt.savefig(os.path.join(p, f"{prefix}_Precision_Recall.png"), bbox_inches='tight')
    plt.close()

    # 3. Calibration
    plt.figure(figsize=Config.CHART_SIZE)
    plt.plot([0, 1], [0, 1], "k:", color=Config.COLORS['BLACK'], label="Perfect Calibration")
    for mod in models_to_plot:
        col = f'Risk_{mod}'
        valid = df.dropna(subset=[col, 'Target_H'])
        if len(valid) < 10: continue
        bs = brier_score_loss(valid['Target_H'], valid[col])
        fraction_of_positives, mean_predicted_value = calibration_curve(valid['Target_H'], valid[col], n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", color=Config.COLORS[mod], lw=Config.LINE_WIDTHS[mod], label=f"{mod} (Brier: {bs:.3f})")
    apply_style(plt.gca(), f"{prefix}: Calibration (Reliability)", "Mean Predicted Risk", "Actual Fraction of Crises")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(p, f"{prefix}_Calibration.png"), bbox_inches='tight')
    plt.close()

    # 4. Lift
    plt.figure(figsize=Config.CHART_SIZE)
    plt.plot([0, 1], [0, 1], '--', color=Config.COLORS['BLACK'], label="Random Guessing")
    total_positives = df['Target_H'].sum()
    for mod in models_to_plot:
        col = f'Risk_{mod}'
        valid = df.dropna(subset=[col, 'Target_H']).sort_values(by=col, ascending=False)
        if len(valid) < 10: continue
        cumulative_positives = np.cumsum(valid['Target_H'])
        gain = cumulative_positives / total_positives
        pct = np.linspace(0, 1, len(gain))
        plt.plot(pct, gain, color=Config.COLORS[mod], lw=Config.LINE_WIDTHS[mod], label=mod)
    apply_style(plt.gca(), f"{prefix}: Cumulative Gain (Efficiency)", "% of Sample Flagged", "% of Crises Caught")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(p, f"{prefix}_Cumulative_Gain.png"), bbox_inches='tight')
    plt.close()

def plot_time_series_risk(df, models_to_plot, prefix, out_path):
    p = os.path.join(out_path, f"Risk_Trajectories_{prefix}")
    os.makedirs(p, exist_ok=True)
    
    # 1. All Countries (Avg Risk vs Crisis Freq)
    all_sub = df.groupby('Date').agg({f'Risk_{m}': 'mean' for m in models_to_plot}).reset_index()
    all_sub['Target_H'] = df.groupby('Date')['Target_H'].mean().values
    
    plt.figure(figsize=Config.CHART_SIZE)
    plt.fill_between(all_sub['Date'], 0, all_sub['Target_H'], color=Config.COLORS['CRISIS_SHADE'], alpha=0.3, label='Crisis Frequency')
    for mod in models_to_plot:
        plt.plot(all_sub['Date'], all_sub[f'Risk_{mod}'], color=Config.COLORS.get(mod, '#000'), lw=Config.LINE_WIDTHS.get(mod, 2), label=mod)
    apply_style(plt.gca(), f"Avg Risk & Crisis Freq: All Countries ({prefix})", "Date", "Probability / Frequency")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(p, f"All_Countries_{prefix}.png"), bbox_inches='tight')
    plt.close()
    
    # 2. By Income Groups (Avg Risk vs Crisis Freq)
    valid_df = df[~df['income'].isin(['Unknown', 'nan'])]
    for grp in valid_df['income'].unique():
        grp_sub = valid_df[valid_df['income'] == grp]
        sub = grp_sub.groupby('Date').agg({f'Risk_{m}': 'mean' for m in models_to_plot}).reset_index()
        sub['Target_H'] = grp_sub.groupby('Date')['Target_H'].mean().values
        
        plt.figure(figsize=Config.CHART_SIZE)
        plt.fill_between(sub['Date'], 0, sub['Target_H'], color=Config.COLORS['CRISIS_SHADE'], alpha=0.3, label='Crisis Frequency')
        for mod in models_to_plot:
            plt.plot(sub['Date'], sub[f'Risk_{mod}'], color=Config.COLORS.get(mod, '#000'), lw=Config.LINE_WIDTHS.get(mod, 2), label=mod)
        apply_style(plt.gca(), f"Avg Risk & Crisis Freq: {grp} ({prefix})", "Date", "Probability / Frequency")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(p, f"Income_{grp.replace(' ','_')}_{prefix}.png"), bbox_inches='tight')
        plt.close()

def plot_zoom_episodes(df, out_path, models_to_plot, episodes):
    p = os.path.join(out_path, "Zoom_Episodes")
    os.makedirs(p, exist_ok=True)
    for country, start, end in episodes:
        target_start = pd.to_datetime(start)
        target_end = pd.to_datetime(end)
        mask = (df['Country_Name'] == country) & (df['Date'] >= target_start) & (df['Date'] <= target_end)
        sub = df[mask]
        if len(sub) < 5: continue
        
        plt.figure(figsize=Config.CHART_SIZE)
        plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), color=Config.COLORS['CRISIS_SHADE'], alpha=0.6, label='Crisis', zorder=1)
        zorder_map = {'Monthly XGBoost': 2, 'Monthly Midas-XGBoost': 3, 'XGBoost': 4, 'Midas-XGBoost': 5}
        
        for mod in models_to_plot:
            plt.plot(sub['Date'], sub[f'Risk_{mod}'], color=Config.COLORS[mod], lw=Config.LINE_WIDTHS[mod], label=mod, zorder=zorder_map.get(mod, 2))
        
        apply_style(plt.gca(), f"Zoom: {country} ({start[:4]}-{end[:4]})", "Date", "Risk Probability")
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(p, f"Zoom_{country}_{start[:4]}_{end[:4]}.png"), bbox_inches='tight')
        plt.close()

def plot_episode_shap_deviations(df, country, start, end, model_label, out_path, model, explainer, predictors):
    if not SHAP_AVAILABLE or not explainer: return
    p = os.path.join(out_path, "Factor_Decomposition")
    os.makedirs(p, exist_ok=True)
    mask = (df['Country_Name'] == country) & (df['Date'] >= pd.to_datetime(start)) & (df['Date'] <= pd.to_datetime(end))
    sub = df[mask].sort_values('Date')
    if len(sub) < 2: return

    X = sub[predictors]
    
    risk_probs = model.predict_proba(X)[:, 1]
    risk_dev = risk_probs - risk_probs[0]
    
    sv = normalize_shap_values(explainer.shap_values(X))
    sv_dev = sv - sv[0, :]
    sv_dev_sum = sv_dev.sum(axis=1)
    
    scaling_factor = np.divide(risk_dev, sv_dev_sum, out=np.zeros_like(risk_dev), where=sv_dev_sum!=0)
    sv_dev_scaled = sv_dev * scaling_factor[:, np.newaxis]
    
    dates_str = sub['Date'].dt.strftime('%Y-%m').tolist()

    def format_and_save_bar(df_bar, title, filename, custom_colors=None):
        fig, ax = plt.subplots(figsize=Config.CHART_SIZE)
        
        if custom_colors:
            colors = custom_colors
        else:
            colors = [Config.COLORS.get(col, Config.COLORS['Others']) for col in df_bar.columns]
            
        df_bar.plot(kind='bar', stacked=True, color=colors, ax=ax, zorder=2, width=0.85)
        ax.plot(range(len(df_bar)), risk_dev, color=Config.COLORS['BLACK'], lw=2, marker='o', label='Risk Index', zorder=4)
        
        crisis_indices = np.where(sub['Target_H'] == 1)[0]
        for idx in crisis_indices:
            ax.axvspan(idx - 0.5, idx + 0.5, color=Config.COLORS['CRISIS_SHADE'], alpha=0.5, zorder=1, lw=0)
            
        plt.axhline(0, color=Config.COLORS['BLACK'], lw=0.8, zorder=3)
        
        ticks = ax.xaxis.get_ticklocs()
        ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
        ax.xaxis.set_ticks(ticks[::6])
        ax.xaxis.set_ticklabels(ticklabels[::6], rotation=0)
        
        apply_style(ax, title, "Date", "Δ Risk Probability Contribution")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(p, filename), bbox_inches='tight')
        plt.close()

    # --- Feature Deviation (with bright categorical colors) ---
    labels = [Config.get_label(f) for f in predictors]
    top_indices = np.argsort(np.abs(sv_dev_scaled).mean(axis=0))[-5:]
    top_labels = [labels[i] for i in top_indices]
    
    df_feat = pd.DataFrame(sv_dev_scaled[:, top_indices], columns=top_labels, index=dates_str)
    df_feat['Others'] = sv_dev_scaled.sum(axis=1) - df_feat.sum(axis=1)
    
    feat_colors = [Config.COLORS['Monthly Midas-XGBoost'], Config.COLORS['Monthly XGBoost'], Config.COLORS['DARK_YELLOW'], Config.COLORS['XGBoost'], Config.COLORS['Midas-XGBoost']]
    feat_colors = feat_colors[:len(df_feat.columns)-1] + [Config.COLORS['Others']]
    
    format_and_save_bar(df_feat, f"Feature Deviation: {country} ({start[:4]}-{end[:4]})", f"Dev_Feature_{country}_{start[:4]}.png", custom_colors=feat_colors)

    # --- Category Deviation ---
    cats = [Config.get_category(f) for f in predictors]
    df_cat = pd.DataFrame(index=dates_str)
    for cat_name in ['Macro', 'Fiscal', 'Financial']:
        cat_idx = [i for i, c in enumerate(cats) if c == cat_name]
        df_cat[cat_name] = sv_dev_scaled[:, cat_idx].sum(axis=1)
    
    df_cat = df_cat[['Macro', 'Fiscal', 'Financial']] 
    format_and_save_bar(df_cat, f"Category Deviation: {country} ({start[:4]}-{end[:4]})", f"Dev_Category_{country}_{start[:4]}.png")

def plot_event_study(df, out_path, model=None, explainer=None, predictors=None):
    print("   [Status] Generating Event Studies...")
    p = os.path.join(out_path, "Event_Studies")
    os.makedirs(p, exist_ok=True)
    
    df = df.sort_values(['Country', 'Date'])
    df['Crisis_Start'] = (df['Target_H'] == 1) & (df.groupby('Country')['Target_H'].shift(1) == 0)
    
    # Pre-calculate Standardized Variables globally per Country
    df_std = df.copy()
    for p_val in predictors:
        df_std[p_val] = df_std.groupby('Country')[p_val].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    
    event_data = []
    event_data_std = []
    models = ['XGBoost', 'Monthly XGBoost'] 
    starts = df[df['Crisis_Start'] == True]
    
    for _, row in starts.iterrows():
        # Raw Data Slice
        mask = (df['Country'] == row['Country']) & \
               (df['Date'] >= row['Date'] - pd.DateOffset(months=24)) & \
               (df['Date'] <= row['Date'] + pd.DateOffset(months=12))
        sub = df[mask].copy()
        
        # Standardized Data Slice
        sub_std = df_std[mask].copy()
        
        if len(sub) < 5: continue
        
        sub['T'] = ((sub['Date'] - row['Date']).dt.days / 30.44).round().astype(int)
        sub_std['T'] = sub['T']
        
        if explainer and predictors and SHAP_AVAILABLE:
            X = sub[predictors]
            sv = normalize_shap_values(explainer.shap_values(X))
            cats = [Config.get_category(f) for f in predictors]
            
            for cat_name in ['Macro', 'Fiscal', 'Financial']:
                cat_idx = [i for i, c in enumerate(cats) if c == cat_name]
                sub[f'SHAP_Cat_{cat_name}'] = np.abs(sv[:, cat_idx]).sum(axis=1)
                
            for target_var in ['long_term_bond_yield', 'govt_deficit_gdp']:
                if target_var in predictors:
                    sub[f'SHAP_{target_var}'] = np.abs(sv[:, predictors.index(target_var)])
                    
        event_data.append(sub)
        event_data_std.append(sub_std)
    
    if event_data:
        # FIX: Reset the index immediately after concatenating to prevent Seaborn grouping errors
        evt_df = pd.concat(event_data).reset_index(drop=True)
        evt_std_df = pd.concat(event_data_std).reset_index(drop=True)
        
        # 1. Base Event Study
        plt.figure(figsize=Config.CHART_SIZE)
        for mod in models:
            sns.lineplot(data=evt_df, x='T', y=f'Risk_{mod}', color=Config.COLORS[mod], lw=Config.LINE_WIDTHS[mod], label=mod)
        plt.axvline(0, color=Config.COLORS['Midas-XGBoost'], linestyle='--', alpha=0.7)
        apply_style(plt.gca(), "Event Study: Average Risk Trajectory", "Months from Crisis Start", "Avg Risk Probability")
        plt.savefig(os.path.join(p, "EventStudy_Risk_Evolution.png"), bbox_inches='tight')
        plt.close()
        
        # 2. Income Group Breakout (Risk Trajectory)
        valid_evt = evt_df[~evt_df['income'].isin(['Unknown', 'nan'])]
        plt.figure(figsize=Config.CHART_SIZE)
        sns.lineplot(data=valid_evt, x='T', y='Risk_XGBoost', hue='income', lw=2)
        plt.axvline(0, color=Config.COLORS['Midas-XGBoost'], linestyle='--', alpha=0.7)
        apply_style(plt.gca(), "Event Study: Avg Risk Trajectory by Income (XGBoost)", "Months from Crisis Start", "Avg Risk Probability")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(p, "EventStudy_Risk_Income_Breakout.png"), bbox_inches='tight')
        plt.close()

        # 3. Raw Variables (Standardized) Evolution
        p_std = os.path.join(p, "Standardized_Variables")
        os.makedirs(p_std, exist_ok=True)
        print("      > Writing Standardized Variable Event Studies...")
        for feat in predictors:
            plt.figure(figsize=Config.CHART_SIZE)
            sns.lineplot(data=evt_std_df, x='T', y=feat, color=Config.COLORS['XGBoost'], lw=3)
            plt.axvline(0, color=Config.COLORS['Midas-XGBoost'], linestyle='--', alpha=0.7)
            plt.axhline(0, color=Config.COLORS['BLACK'], lw=0.8)
            apply_style(plt.gca(), f"Standardized Pre-Crisis Trajectory: {Config.get_label(feat)}", "Months from Crisis Start", "Std Dev (Z-Score)")
            plt.savefig(os.path.join(p_std, f"EventStudy_StdVar_{feat}.png"), bbox_inches='tight')
            plt.close()

    # 4. Predictor SHAP Deviations (T-12 to T+12)
    if explainer and predictors and model and SHAP_AVAILABLE:
        base_margin = explainer.expected_value
        if isinstance(base_margin, (list, np.ndarray)): 
            base_margin = base_margin[-1] if len(base_margin)>1 else base_margin[0]
        base_prob = expit(base_margin)
        
        event_data_dev = []
        for _, row in starts.iterrows():
            mask = (df['Country'] == row['Country']) & \
                   (df['Date'] >= row['Date'] - pd.DateOffset(months=12)) & \
                   (df['Date'] <= row['Date'] + pd.DateOffset(months=12))
            sub = df[mask].copy()
            sub['T'] = ((sub['Date'] - row['Date']).dt.days / 30.44).round().astype(int)
            
            if 0 not in sub['T'].values or len(sub) < 2: continue
            
            X = sub[predictors]
            sv = normalize_shap_values(explainer.shap_values(X))
            
            risk_probs = model.predict_proba(X)[:, 1]
            margin_sum = sv.sum(axis=1)
            prob_diff = risk_probs - base_prob
            scaling_factor = np.divide(prob_diff, margin_sum, out=np.zeros_like(prob_diff), where=margin_sum!=0)
            sv_scaled = sv * scaling_factor[:, np.newaxis]
            
            t0_idx = np.where(sub['T'].values == 0)[0][0]
            sv_dev = sv_scaled - sv_scaled[t0_idx, :]
            
            for i, feat in enumerate(predictors):
                sub[f'Dev_{feat}'] = sv_dev[:, i]
                
            event_data_dev.append(sub)
            
        if event_data_dev:
            # FIX: Reset the index for the SHAP deviation dataframe as well
            evt_dev_df = pd.concat(event_data_dev).reset_index(drop=True)
            p_dev = os.path.join(p, "Predictor_Deviations")
            os.makedirs(p_dev, exist_ok=True)
            
            print("      > Writing SHAP Deviation Event Studies...")
            for feat in predictors:
                plt.figure(figsize=Config.CHART_SIZE)
                sns.lineplot(data=evt_dev_df, x='T', y=f'Dev_{feat}', color=Config.COLORS['Monthly XGBoost'], lw=3)
                plt.axvline(0, color=Config.COLORS['Midas-XGBoost'], linestyle='--', alpha=0.7)
                plt.axhline(0, color=Config.COLORS['BLACK'], lw=0.8)
                apply_style(plt.gca(), f"SHAP Risk Evolution: {Config.get_label(feat)}", "Months from Crisis Start (T=0)", "Δ SHAP Contribution to Risk")
                plt.savefig(os.path.join(p_dev, f"EventStudy_Dev_{feat}.png"), bbox_inches='tight')
                plt.close()




def normalize_shap_values(sv):
    if isinstance(sv, list): return sv[1] if len(sv) == 2 else sv[0]
    elif sv.ndim == 3: return sv[:, :, 1]
    return sv

def plot_shap_advanced(model, X, path, label, predictors):
    if Config.SHAP_COMPLEXITY == 0 or not SHAP_AVAILABLE or len(X) < 10: return
    p = os.path.join(path, "SHAP", label.replace(' ', '_'))
    os.makedirs(p, exist_ok=True)
    
    X_sub = X.sample(min(len(X), Config.SHAP_SAMPLES), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_sub)
    sv = normalize_shap_values(sv_raw)
    
    if Config.SHAP_COMPLEXITY >= 1:
        plt.figure(figsize=Config.CHART_SIZE)
        X_labeled = X_sub.rename(columns={col: Config.get_label(col) for col in X_sub.columns})
        shap.summary_plot(sv, X_labeled, show=False)
        plt.title(f"SHAP: {label}", fontsize=16, fontweight='bold', color=Config.COLORS['BLACK'])
        plt.savefig(os.path.join(p, f"Beeswarm_{label.replace(' ', '_')}.png"), bbox_inches='tight')
        plt.close()
        
    if Config.SHAP_COMPLEXITY >= 2:
        for i, feature in enumerate(predictors):
            f_label = Config.get_label(feature)
            x_vals = X_sub[feature].values
            y_vals = sv[:, i]
            
            if np.isnan(x_vals).all() or np.isnan(y_vals).all(): continue
            xlim = np.nanpercentile(x_vals, Config.DEP_PLOT_X_PCT)
            ylim = np.nanpercentile(y_vals, Config.DEP_PLOT_Y_PCT)
            mask = (x_vals >= xlim[0]) & (x_vals <= xlim[1]) & (y_vals >= ylim[0]) & (y_vals <= ylim[1])
            x_filt, y_filt = x_vals[mask], y_vals[mask]
            
            if len(x_filt) < 10: continue

            plt.figure(figsize=Config.CHART_SIZE)
            plt.scatter(x_filt, y_filt, color=Config.COLORS['Monthly XGBoost'], alpha=Config.DEP_PLOT_ALPHA, s=15, edgecolor='none')
            
            loess_res = compute_loess(x_filt, y_filt)
            if loess_res:
                plt.plot(loess_res[0], loess_res[1], color=Config.COLORS['Midas-XGBoost'], lw=3, label='Trend (LOESS)')
                plt.legend()
                
            plt.xlim(xlim); plt.ylim(ylim)
            apply_style(plt.gca(), f"Dependence: {f_label} ({label})", f_label, "SHAP Value")
            plt.savefig(os.path.join(p, f"Dep_{feature}.png"), bbox_inches='tight')
            plt.close()

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

    return df.dropna(subset=['Target_H']).copy(), predictors, [Config.get_meta(p)[2] for p in predictors]

# =============================================================================
# 5. ENGINE
# =============================================================================

def run_engine():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    print("\n=== ENGINE START (v45.0) ===")
    d_tr, predictors, constraints = prepare_data()
    
    if len(d_tr) == 0: return
    
    cv_t = TimeSeriesSplit(n_splits=Config.CV_FOLDS) if Config.CV_TYPE == "Temporal" else StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=42)
    
    models_4way = ['XGBoost', 'Midas-XGBoost', 'Monthly XGBoost', 'Monthly Midas-XGBoost']
    models_2way = ['XGBoost', 'Monthly XGBoost']
    
    for k in models_4way: d_tr[f'Risk_{k}'] = np.nan

    # ---------------------------------------------------------
    # PART 1: GLOBAL MODELS
    # ---------------------------------------------------------
    print("\n>>> TRAINING GLOBAL MODELS...")
    clf_xgb = HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42)
    search_xgb = RandomizedSearchCV(clf_xgb, Config.XGB_GRID, n_iter=10, cv=cv_t, scoring='roc_auc', n_jobs=-1, random_state=42)
    search_xgb.fit(d_tr[predictors], d_tr['Target_H'])
    best_xgb = search_xgb.best_estimator_
    d_tr['Risk_XGBoost'] = best_xgb.predict_proba(d_tr[predictors])[:, 1]

    pipe_midas = Pipeline([
        ('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), 
        ('clf', HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42))
    ])
    search_midas = RandomizedSearchCV(pipe_midas, Config.MIDAS_GRID, n_iter=15, cv=cv_t, scoring='roc_auc', n_jobs=-1, random_state=42)
    search_midas.fit(d_tr, d_tr['Target_H'])
    best_midas = search_midas.best_estimator_
    d_tr['Risk_Midas-XGBoost'] = best_midas.predict_proba(d_tr)[:, 1]

    # ---------------------------------------------------------
    # PART 2: MONTHLY MODELS
    # ---------------------------------------------------------
    print("\n>>> TRAINING MONTHLY MODELS (12 Rounds)...")
    for m in range(1, 13):
        m_idx = d_tr['month'] == m
        m_data = d_tr[m_idx]
        if len(m_data) < 50: continue
        
        sub_cv = TimeSeriesSplit(n_splits=3)
        
        clf_mxgb = HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42)
        search_mxgb = RandomizedSearchCV(clf_mxgb, Config.XGB_GRID, n_iter=10, cv=sub_cv, scoring='roc_auc', n_jobs=-1, random_state=42)
        search_mxgb.fit(m_data[predictors], m_data['Target_H'])
        d_tr.loc[m_idx, 'Risk_Monthly XGBoost'] = search_mxgb.best_estimator_.predict_proba(m_data[predictors])[:, 1]
        
        pipe_mmidas = Pipeline([
            ('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), 
            ('clf', HistGradientBoostingClassifier(monotonic_cst=constraints, random_state=42))
        ])
        search_mmidas = RandomizedSearchCV(pipe_mmidas, Config.MIDAS_GRID, n_iter=15, cv=sub_cv, scoring='roc_auc', n_jobs=-1, random_state=42)
        search_mmidas.fit(m_data, m_data['Target_H'])
        d_tr.loc[m_idx, 'Risk_Monthly Midas-XGBoost'] = search_mmidas.best_estimator_.predict_proba(m_data)[:, 1]

    # ---------------------------------------------------------
    # PART 3: DIAGNOSTICS & EXPORTS
    # ---------------------------------------------------------
    print("\n>>> GENERATING FINAL SUITE...")
    
    print("   [Status] Exporting Comprehensive AUC Data...")
    auc_records = []
    for k in models_4way:
        valid = d_tr.dropna(subset=[f'Risk_{k}', 'Target_H'])
        if valid['Target_H'].nunique() > 1:
            auc_records.append({'Category': 'Total', 'Segment': 'All', 'Model': k, 'AUC': roc_auc_score(valid['Target_H'], valid[f'Risk_{k}'])})
        
        for inc in valid['income'].unique():
            if str(inc) in ['nan', 'Unknown']: continue
            sub = valid[valid['income'] == inc]
            if sub['Target_H'].nunique() > 1:
                auc_records.append({'Category': 'Income', 'Segment': inc, 'Model': k, 'AUC': roc_auc_score(sub['Target_H'], sub[f'Risk_{k}'])})
        
        periods = [('1980-1991', 1980, 1991), ('1992-2006', 1992, 2006), ('2007-2023', 2007, 2023)]
        for p_name, p_start, p_end in periods:
            sub = valid[(valid['Date'].dt.year >= p_start) & (valid['Date'].dt.year <= p_end)]
            if len(sub) > 10 and sub['Target_H'].nunique() > 1:
                auc_records.append({'Category': 'Time Period', 'Segment': p_name, 'Model': k, 'AUC': roc_auc_score(sub['Target_H'], sub[f'Risk_{k}'])})

    pd.DataFrame(auc_records).to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Comprehensive_AUCs.csv"), index=False)

    print("   [Status] Generating Graphics...")
    plot_income_silhouettes(d_tr, models_4way, Config.OUTPUT_ROOT)
    plot_comparisons(d_tr, models_4way, "Comp_4Way", Config.OUTPUT_ROOT)
    plot_comparisons(d_tr, models_2way, "Comp_2Way", Config.OUTPUT_ROOT)
    
    # Restored Volatility
    vol_df = plot_volatility_comparison(d_tr, models_4way, "Comp_4Way", Config.OUTPUT_ROOT)
    plot_volatility_comparison(d_tr, models_2way, "Comp_2Way", Config.OUTPUT_ROOT)
    vol_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Volatility_Score_4Way.csv"), index=False)

    # Time series with crisis frequency overlay (4Way and 2Way)
    plot_time_series_risk(d_tr, models_4way, "4Way", Config.OUTPUT_ROOT)
    plot_time_series_risk(d_tr, models_2way, "2Way", Config.OUTPUT_ROOT)

    episodes = [
        ("Albania", "1996-01-01", "1999-12-01"), ("Albania", "2012-01-01", "2015-12-01"),
        ("Brazil", "2003-01-01", "2008-12-01"), ("Cameroon", "2013-01-01", "2017-12-01"),
        ("Colombia", "1997-01-01", "2000-12-01"), ("Cyprus", "2010-01-01", "2014-12-01"),
        ("Greece", "2008-01-01", "2015-12-01"), ("Iceland", "2006-01-01", "2008-12-01"),
        ("Indonesia", "1995-01-01", "1998-12-01"), ("Ireland", "2008-01-01", "2011-12-01"),
        ("Jamaica", "2007-01-01", "2010-12-01"), ("Lebanon", "2017-01-01", "2023-12-01")
    ]
    plot_zoom_episodes(d_tr, Config.OUTPUT_ROOT, models_4way, episodes)
    
    if SHAP_AVAILABLE:
        print("   [Status] Calculating Scaled Global SHAP Values...")
        explainer = shap.TreeExplainer(best_xgb)
        
        base_margin = explainer.expected_value
        if isinstance(base_margin, (list, np.ndarray)): 
            base_margin = base_margin[-1] if len(base_margin) > 1 else base_margin[0]
        base_prob = expit(base_margin)
        
        X_all = d_tr[predictors]
        sv_all = normalize_shap_values(explainer.shap_values(X_all))
        risk_probs = best_xgb.predict_proba(X_all)[:, 1]
        
        margin_sum = sv_all.sum(axis=1)
        prob_diff = risk_probs - base_prob
        scaling_factor = np.divide(prob_diff, margin_sum, out=np.zeros_like(prob_diff), where=margin_sum!=0)
        sv_scaled = sv_all * scaling_factor[:, np.newaxis]
        
        shap_df = pd.DataFrame(sv_scaled, columns=[Config.get_label(f) for f in predictors], index=d_tr.index)
        shap_df['Base_Value'] = base_prob
        shap_df['Risk_XGBoost'] = risk_probs
        shap_df['Country_Name'] = d_tr['Country_Name']
        shap_df['IFS'] = d_tr['Country']
        shap_df['Date'] = d_tr['Date']
        shap_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "Results_Global_SHAP_Values.csv"), index=False)
        
        for c, start, end in episodes:
            plot_episode_shap_deviations(d_tr, c, start, end, "XGBoost", Config.OUTPUT_ROOT, best_xgb, explainer, predictors)
            
        plot_event_study(d_tr, Config.OUTPUT_ROOT, best_xgb, explainer, predictors)

    print("\n=== PROCESS COMPLETED ===")

if __name__ == "__main__":
    run_engine()