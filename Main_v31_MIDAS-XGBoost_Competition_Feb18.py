# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:00:00 2026
@author: cmarsilli & Gemini
Sovereign Crisis Forecasting Engine v31.0 (Grand Tournament + Almon Charts)

OBJECTIVE:
- Compare 4 Architectures: Global vs. Monthly x Baseline vs. MIDAS.
- Reconstruct continuous time series to test stability.
- Generate Full Metric Suite (ROC, Calibration, PRC, Lift).
- Generate Country-Specific comparative risk trajectories.
- VISUALIZE ALMON LAGS: Charts the learned memory structures.
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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings

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
    OUTPUT_ROOT = "Output_v31_Grand_Tournament"
    
    # Core Settings
    CV_FOLDS = 5
    HORIZON = 12
    TARGET = "precrisis"
    WINSORIZE_LIMITS = [0.00, 1]
    USE_NATIVE_IMPUTATION = True
    
    # Analysis Settings
    SHAP_SAMPLES = 2000
    
    CHART_SIZE = (10, 6)
    COLORS = {
        'Global_Base': '#95A5A6',   # Grey (Baseline)
        'Global_Midas': '#34495E',  # Dark Blue (Strong Baseline)
        'Month_Base': '#E74C3C',    # Bright Red (Volatile)
        'Month_Midas': '#8E44AD',   # Purple (The Challenger)
        'CRISIS_SHADE': '#E5E7E9',
        'OBS_LINE': '#2C3E50',
        'ALMON_BAR': '#2980B9',
        'ALMON_LINE': '#E74C3C'
    }

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    # --- HYPERPARAMETERS ---
    XGB_GRID = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [300, 500],
        'max_depth': [3, 4, 6],
        'l2_regularization': [0.0, 1.0, 5.0]
    }

    MIDAS_GRID = {
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 4, 6],
        'almon__theta1': [-0.2, -0.05, 0.0],  # Slope
        'almon__theta2': [-0.005, 0.0]               # Curvature
    }

    VARS = {
        "PCPI_PCH": ("Inflation", "Macro", 1),
        "gdp_growth": ("GDP Growth", "Macro", -1),
        "BoP_gdp": ("Current Account/GDP", "Macro", -1),
        "reserve_cover": ("FX Reserve Cover", "Macro", -1),
        "terms_of_trade": ("Terms of Trade", "Macro", -1),
        "GDP_percapita_over_US_12ma": ("GDP per Capita", "Macro", -1), 
        "oil_to_gdp": ("Oil Exports/GDP", "Macro", -1), 
        "govt_debt_gdp": ("Public Debt/GDP", "Fiscal", 1),
        "govt_deficit_gdp": ("Fiscal Balance/GDP", "Fiscal", -1),
        "tot_ext_debt_gdp": ("Total Ext Debt/GDP", "Fiscal", 1),
        "deposit_rate": ("ST Rate", "Financial", 1),
        "spread": ("Sovereign Spread", "Financial", 1), 
        "VIX": ("VIX Index", "Financial", 1)
    }

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
# 3. ADVANCED CHARTING SUITE
# =============================================================================

def apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold', color='#333333')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

def plot_almon_weights(theta1, theta2, max_lag, model_label, out_path):
    k = np.arange(1, max_lag + 1)
    w_raw = np.exp(theta1 * k + theta2 * (k**2))
    w = w_raw / np.sum(w_raw)
    
    plt.figure(figsize=(8, 5))
    plt.bar(k, w, color=Config.COLORS['ALMON_BAR'], alpha=0.7, label='Lag Weight')
    plt.plot(k, w, color=Config.COLORS['ALMON_LINE'], marker='o', lw=2)
    
    apply_style(plt.gca(), f"Almon Lag Memory Structure: {model_label}\n(t1={theta1:.3f}, t2={theta2:.3f})", "Lag (Months)", "Impact Weight")
    plt.xticks(k)
    
    p = os.path.join(out_path, "Almon_Structures")
    os.makedirs(p, exist_ok=True)
    plt.savefig(os.path.join(p, f"Almon_Weights_{model_label}.png"), bbox_inches='tight')
    plt.close()

def plot_four_way_roc(df, out_path):
    plt.figure(figsize=Config.CHART_SIZE)
    models = [
        ('Risk_Global_Base', 'Global Base', Config.COLORS['Global_Base']),
        ('Risk_Global_Midas', 'Global MIDAS', Config.COLORS['Global_Midas']),
        ('Risk_Month_Base', 'Monthly Base', Config.COLORS['Month_Base']),
        ('Risk_Month_Midas', 'Monthly MIDAS', Config.COLORS['Month_Midas'])
    ]
    for col, label, color in models:
        valid = df.dropna(subset=[col, 'Target_H'])
        if len(valid) < 10: continue
        fpr, tpr, _ = roc_curve(valid['Target_H'], valid[col])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC: {auc_score:.3f})')
    
    plt.plot([0,1],[0,1], 'k--', alpha=0.2)
    apply_style(plt.gca(), "Tournament ROC: All Architectures", "FPR", "TPR")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(out_path, "1_Grand_Tournament_ROC.png"), bbox_inches='tight')
    plt.close()

def plot_precision_recall(df, out_path):
    plt.figure(figsize=Config.CHART_SIZE)
    models = [
        ('Risk_Global_Base', 'Global Base', Config.COLORS['Global_Base']),
        ('Risk_Global_Midas', 'Global MIDAS', Config.COLORS['Global_Midas']),
        ('Risk_Month_Base', 'Month Base', Config.COLORS['Month_Base']),
        ('Risk_Month_Midas', 'Month MIDAS', Config.COLORS['Month_Midas'])
    ]
    for col, label, color in models:
        valid = df.dropna(subset=[col, 'Target_H'])
        if len(valid) < 10: continue
        precision, recall, _ = precision_recall_curve(valid['Target_H'], valid[col])
        ap = average_precision_score(valid['Target_H'], valid[col])
        plt.plot(recall, precision, color=color, lw=2, label=f'{label} (AP: {ap:.3f})')
    
    baseline = df['Target_H'].mean()
    plt.axhline(y=baseline, color='black', linestyle=':', label=f'Random ({baseline:.2f})')
    apply_style(plt.gca(), "Precision-Recall (Truth Teller)", "Recall", "Precision")
    plt.legend()
    plt.savefig(os.path.join(out_path, "2_Precision_Recall.png"), bbox_inches='tight')
    plt.close()

def plot_calibration_curve_custom(df, out_path):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    models = [
        ('Risk_Global_Base', 'Global Base', Config.COLORS['Global_Base']),
        ('Risk_Global_Midas', 'Global MIDAS', Config.COLORS['Global_Midas']),
        ('Risk_Month_Base', 'Month Base', Config.COLORS['Month_Base']),
        ('Risk_Month_Midas', 'Month MIDAS', Config.COLORS['Month_Midas'])
    ]
    for col, label, color in models:
        valid = df.dropna(subset=[col, 'Target_H'])
        if len(valid) < 10: continue
        bs = brier_score_loss(valid['Target_H'], valid[col])
        fraction_of_positives, mean_predicted_value = calibration_curve(valid['Target_H'], valid[col], n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", color=color, lw=2, label=f"{label} (Brier: {bs:.3f})")
    
    apply_style(plt.gca(), "Calibration (Reliability)", "Mean Predicted Risk", "Actual Fraction")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_path, "3_Calibration_Reliability.png"), bbox_inches='tight')
    plt.close()

def plot_cumulative_gain(df, out_path):
    plt.figure(figsize=Config.CHART_SIZE)
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    total_positives = df['Target_H'].sum()
    models = [
        ('Risk_Global_Base', 'Global Base', Config.COLORS['Global_Base']),
        ('Risk_Global_Midas', 'Global MIDAS', Config.COLORS['Global_Midas']),
        ('Risk_Month_Base', 'Month Base', Config.COLORS['Month_Base']),
        ('Risk_Month_Midas', 'Month MIDAS', Config.COLORS['Month_Midas'])
    ]
    for col, label, color in models:
        valid = df.dropna(subset=[col, 'Target_H']).sort_values(by=col, ascending=False)
        cumulative_positives = np.cumsum(valid['Target_H'])
        gain = cumulative_positives / total_positives
        pct = np.linspace(0, 1, len(gain))
        plt.plot(pct, gain, color=color, lw=2, label=label)
        
    apply_style(plt.gca(), "Cumulative Gain (Targeting Efficiency)", "% Flagged", "% Caught")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_path, "4_Cumulative_Gain.png"), bbox_inches='tight')
    plt.close()

def plot_auc_evolution(monthly_aucs, out_path):
    plt.figure(figsize=(12, 6))
    for col in ['Global_Base', 'Global_Midas', 'Month_Base', 'Month_Midas']:
        sns.lineplot(data=monthly_aucs, x='Month', y=col, marker='o', color=Config.COLORS[col], label=col.replace('_', ' '))
    plt.ylim(0.5, 1.0)
    plt.xticks(range(1, 13))
    apply_style(plt.gca(), "Stability Check: AUC Evolution by Month", "Month", "AUC")
    plt.legend()
    plt.savefig(os.path.join(out_path, "5_AUC_Evolution.png"), bbox_inches='tight')
    plt.close()

def plot_income_group_risk(df, out_path):
    groups = df['income'].unique()
    for grp in groups:
        if str(grp) == "nan" or grp == "Unknown": continue
        sub = df[df['income'] == grp].groupby('Date')[['Risk_Global_Midas', 'Risk_Month_Midas', 'Risk_Month_Base']].mean().reset_index()
        
        plt.figure(figsize=(12, 5))
        plt.plot(sub['Date'], sub['Risk_Month_Base'], color=Config.COLORS['Month_Base'], alpha=0.3, label='Month Base')
        plt.plot(sub['Date'], sub['Risk_Month_Midas'], color=Config.COLORS['Month_Midas'], lw=2, label='Month MIDAS')
        plt.plot(sub['Date'], sub['Risk_Global_Midas'], color=Config.COLORS['Global_Midas'], linestyle='--', label='Global MIDAS')
        
        apply_style(plt.gca(), f"Average Risk: {grp}", "Date", "Avg Risk Probability")
        plt.legend()
        plt.savefig(os.path.join(out_path, f"Risk_Profile_{grp}.png"), bbox_inches='tight')
        plt.close()

def plot_shap_comparison(model_global, X_global, model_monthly, X_monthly, out_path):
    if not SHAP_AVAILABLE: return
    
    # 1. Global Beeswarm
    plt.figure(figsize=(10, 8))
    explainer_g = shap.TreeExplainer(model_global)
    sv_g = explainer_g.shap_values(X_global.sample(min(2000, len(X_global)), random_state=42))
    if isinstance(sv_g, list): sv_g = sv_g[1]
    shap.summary_plot(sv_g, X_global.sample(min(2000, len(X_global)), random_state=42), show=False)
    plt.title("Global MIDAS: Feature Importance")
    plt.savefig(os.path.join(out_path, "SHAP_Global_Midas.png"), bbox_inches='tight')
    plt.close()

    # 2. Monthly Proxy Beeswarm
    plt.figure(figsize=(10, 8))
    explainer_m = shap.TreeExplainer(model_monthly)
    sv_m = explainer_m.shap_values(X_monthly.sample(min(2000, len(X_monthly)), random_state=42))
    if isinstance(sv_m, list): sv_m = sv_m[1]
    shap.summary_plot(sv_m, X_monthly.sample(min(2000, len(X_monthly)), random_state=42), show=False)
    plt.title("Monthly MIDAS (December Proxy): Feature Importance")
    plt.savefig(os.path.join(out_path, "SHAP_Monthly_Midas_Dec.png"), bbox_inches='tight')
    plt.close()

def plot_country_comparison(df, country_name, out_path):
    sub = df[df['Country_Name'] == country_name].sort_values('Date')
    if len(sub) < 12: return
    
    plt.figure(figsize=(12, 5))
    
    plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), 
                     color=Config.COLORS['CRISIS_SHADE'], alpha=0.5, label='Crisis Event')
    
    if 'Risk_Month_Base' in sub.columns:
        plt.plot(sub['Date'], sub['Risk_Month_Base'], color=Config.COLORS['Month_Base'], 
                 linestyle='--', lw=1.5, alpha=0.5, label='Monthly Base (Erratic)')
    if 'Risk_Month_Midas' in sub.columns:
        plt.plot(sub['Date'], sub['Risk_Month_Midas'], color=Config.COLORS['Month_Midas'], 
                 lw=2.5, label='Monthly MIDAS')
    if 'Risk_Global_Midas' in sub.columns:
        plt.plot(sub['Date'], sub['Risk_Global_Midas'], color=Config.COLORS['Global_Midas'], 
                 linestyle=':', lw=2, label='Global MIDAS')
    
    apply_style(plt.gca(), f"Risk Trajectory: {country_name}", "Date", "Risk Probability")
    plt.legend(loc='upper left')
    plt.ylim(0, 1.05)
    
    safe_name = country_name.replace(" ", "_").replace("/", "-")
    plt.savefig(os.path.join(out_path, f"{safe_name}_Comparison.png"), bbox_inches='tight')
    plt.close()

def plot_flashing_red(df, out_path):
    p = os.path.join(out_path, "Warnings")
    os.makedirs(p, exist_ok=True)
    df = df.sort_values(['Country', 'Date'])
    
    primary_risk = 'Risk_Month_Midas'
    if primary_risk not in df.columns: return
    
    df['Delta'] = df.groupby('Country')[primary_risk].diff(12)
    alerts = df[(df['Delta'] > 0.15) & (df[primary_risk] > 0.4) & (df['Date'] > '2020-01-01')]
    
    for c in alerts['Country_Name'].unique()[:15]:
        sub = df[df['Country_Name'] == c]
        plt.figure(figsize=(10, 4))
        plt.plot(sub['Date'], sub[primary_risk], color=Config.COLORS['Month_Midas'], lw=2, label='Monthly MIDAS')
        plt.fill_between(sub['Date'], 0, 1, where=(sub['Target_H'] == 1), color=Config.COLORS['CRISIS_SHADE'], alpha=0.4, label='Crisis')
        apply_style(plt.gca(), f"WARNING: {c}", "Date", "Risk")
        plt.legend()
        plt.savefig(os.path.join(p, f"WARNING_{c}.png"), bbox_inches='tight')
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
        print("   [Status] Merging Main (Country) with Map (IFS)...")
        df = df.merge(map_df[['IFS', 'income', 'Area', 'Country_Name']], 
                      left_on='Country', right_on='IFS', how='left')
        
        df['income'] = df['income'].fillna("Unknown")
        df['Area'] = df['Area'].fillna("Unknown")
        if 'Country_Name_y' in df.columns:
            df['Country_Name'] = df['Country_Name_y'].fillna(df['Country_Name_x'])
            df = df.drop(columns=['Country_Name_x', 'Country_Name_y'])
    else:
        print("   [Status] Mapping file not found.")
        df['income'] = "Unknown"
        df['Area'] = "Unknown"

    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    
    print("   [Status] Calculating Crisis Targets...")
    df = df.sort_values(['Country', 'Date'])
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)

    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    
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
    
    for lc in lag_cols:
        df[lc] = df[lc].fillna(df[lc].median())

    return df.dropna(subset=['Target_H']).copy(), predictors

# =============================================================================
# 5. THE GRAND TOURNAMENT ENGINE
# =============================================================================

def run_engine():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    print("\n=== ENGINE START ===")
    df, predictors = prepare_data()
    
    print(f"\n--- Data Diagnostics ---")
    print(f"Training Rows: {len(df)}")
    print(f"Crisis Count: {df['Target_H'].sum()}")
    print(f"Crisis Rate: {df['Target_H'].mean():.4f}")
    if len(df) == 0: return
    
    cv_t = TimeSeriesSplit(n_splits=3)
    
    monthly_aucs = {'Month': list(range(1, 13)), 
                    'Global_Base': [], 'Global_Midas': [], 
                    'Month_Base': [], 'Month_Midas': []}
    
    # ---------------------------------------------------------
    # PART 1: GLOBAL MODELS
    # ---------------------------------------------------------
    print("\n>>> TRAINING GLOBAL MODELS...")
    
    print("   [1/4] Global XGBoost (Baseline)...")
    clf_gb = HistGradientBoostingClassifier(random_state=42)
    search_gb = RandomizedSearchCV(clf_gb, Config.XGB_GRID, n_iter=8, cv=cv_t, scoring='roc_auc', n_jobs=-1)
    search_gb.fit(df[predictors], df['Target_H'])
    best_gb = search_gb.best_estimator_
    df['Risk_Global_Base'] = best_gb.predict_proba(df[predictors])[:, 1]
    
    print("   [2/4] Global MIDAS-XGBoost...")
    pipe_gm = Pipeline([
        ('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), 
        ('clf', HistGradientBoostingClassifier(random_state=42))
    ])
    search_gm = RandomizedSearchCV(pipe_gm, Config.MIDAS_GRID, n_iter=12, cv=cv_t, scoring='roc_auc', n_jobs=-1)
    search_gm.fit(df, df['Target_H']) 
    best_gm = search_gm.best_estimator_
    df['Risk_Global_Midas'] = best_gm.predict_proba(df)[:, 1]

    # Plot Global Almon Weights
    best_t1_g = search_gm.best_params_['almon__theta1']
    best_t2_g = search_gm.best_params_['almon__theta2']
    plot_almon_weights(best_t1_g, best_t2_g, 12, "Global_MIDAS", Config.OUTPUT_ROOT)

    for m in range(1, 13):
        sub = df[df['month'] == m]
        monthly_aucs['Global_Base'].append(roc_auc_score(sub['Target_H'], sub['Risk_Global_Base']))
        monthly_aucs['Global_Midas'].append(roc_auc_score(sub['Target_H'], sub['Risk_Global_Midas']))

    # ---------------------------------------------------------
    # PART 2: MONTHLY MODELS
    # ---------------------------------------------------------
    print("\n>>> TRAINING MONTHLY MODELS (12 Rounds)...")
    
    df['Risk_Month_Base'] = np.nan
    df['Risk_Month_Midas'] = np.nan
    
    dec_model_midas = None
    dec_data_midas = None
    
    for m in range(1, 13):
        print(f"   [Status] Round {m}: Training Month {m} Models...")
        m_idx = df['month'] == m
        m_data = df[m_idx]
        
        if len(m_data) < 50: continue
        
        # Month Base
        clf_mb = HistGradientBoostingClassifier(random_state=42)
        search_mb = RandomizedSearchCV(clf_mb, Config.XGB_GRID, n_iter=8, cv=cv_t, scoring='roc_auc', n_jobs=-1)
        search_mb.fit(m_data[predictors], m_data['Target_H'])
        best_mb = search_mb.best_estimator_
        df.loc[m_idx, 'Risk_Month_Base'] = best_mb.predict_proba(m_data[predictors])[:, 1]
        
        # Month MIDAS
        pipe_mm = Pipeline([
            ('almon', AlmonValueCombiner(base_features=predictors, max_lag=12)), 
            ('clf', HistGradientBoostingClassifier(random_state=42))
        ])
        search_mm = RandomizedSearchCV(pipe_mm, Config.MIDAS_GRID, n_iter=12, cv=cv_t, scoring='roc_auc', n_jobs=-1)
        search_mm.fit(m_data, m_data['Target_H'])
        best_mm = search_mm.best_estimator_
        df.loc[m_idx, 'Risk_Month_Midas'] = best_mm.predict_proba(m_data)[:, 1]
        
        # Plot Month-Specific Almon Weights
        best_t1_m = search_mm.best_params_['almon__theta1']
        best_t2_m = search_mm.best_params_['almon__theta2']
        plot_almon_weights(best_t1_m, best_t2_m, 12, f"Month_{m:02d}_MIDAS", Config.OUTPUT_ROOT)
        
        monthly_aucs['Month_Base'].append(roc_auc_score(m_data['Target_H'], df.loc[m_idx, 'Risk_Month_Base']))
        monthly_aucs['Month_Midas'].append(roc_auc_score(m_data['Target_H'], df.loc[m_idx, 'Risk_Month_Midas']))
        
        if m == 12:
            dec_model_midas = best_mm.named_steps['clf']
            dec_data_midas = best_mm.named_steps['almon'].transform(m_data)

    # ---------------------------------------------------------
    # PART 3: ANALYSIS & CHARTS
    # ---------------------------------------------------------
    print("\n>>> GENERATING FINAL METRIC SUITE...")
    
    # 1. Stability Score
    df = df.sort_values(['Country', 'Date'])
    metrics = []
    for model_col in ['Risk_Global_Base', 'Risk_Global_Midas', 'Risk_Month_Base', 'Risk_Month_Midas']:
        diffs = df.groupby('Country')[model_col].diff().abs()
        volatility = diffs.mean()
        valid = df.dropna(subset=[model_col])
        auc_tot = roc_auc_score(valid['Target_H'], valid[model_col])
        metrics.append({'Model': model_col, 'AUC': auc_tot, 'Volatility_Score': volatility})
    
    res_df = pd.DataFrame(metrics)
    print("\n=== FINAL SCOREBOARD ===")
    print(res_df)
    res_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "Scoreboard_Stability.csv"), index=False)
    
    # 2. General Plots
    plot_four_way_roc(df, Config.OUTPUT_ROOT)
    plot_precision_recall(df, Config.OUTPUT_ROOT)
    plot_calibration_curve_custom(df, Config.OUTPUT_ROOT)
    plot_cumulative_gain(df, Config.OUTPUT_ROOT)
    plot_auc_evolution(pd.DataFrame(monthly_aucs), Config.OUTPUT_ROOT)
    plot_income_group_risk(df, Config.OUTPUT_ROOT)
    
    # 3. Country-Specific Charts
    print("   [Status] Generating Country Charts...")
    c_path = os.path.join(Config.OUTPUT_ROOT, "Country_Comparisons")
    os.makedirs(c_path, exist_ok=True)
    
    interesting_countries = df[(df['Target_H'] == 1) | (df['Risk_Month_Midas'] > 0.4)]['Country_Name'].unique()
    for c in interesting_countries:
        plot_country_comparison(df, c, c_path)
        
    plot_flashing_red(df, Config.OUTPUT_ROOT)
    
    # 4. SHAP
    if SHAP_AVAILABLE:
        print("   [Status] Generating SHAP Beeswarms...")
        X_glob_trans = best_gm.named_steps['almon'].transform(df)
        plot_shap_comparison(best_gm.named_steps['clf'], X_glob_trans, dec_model_midas, dec_data_midas, Config.OUTPUT_ROOT)

    # 5. Save Final Predictions
    print("   [Status] Saving Reconstructed Data...")
    cols = ['Country_Name', 'Date', 'income', 'Target_H', 'Risk_Global_Base', 'Risk_Global_Midas', 'Risk_Month_Base', 'Risk_Month_Midas']
    df[cols].to_csv(os.path.join(Config.OUTPUT_ROOT, "Final_TimeSeries_Predictions.csv"), index=False)
    
    print("\n=== PROCESS COMPLETED ===")

if __name__ == "__main__":
    run_engine()