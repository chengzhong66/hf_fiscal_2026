# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 23:55:00 2026
@author: cmarsilli & Gemini
Sovereign Crisis Forecasting Engine v19.5 (Lasso & Lasso-MIDAS Support)

UPDATES:
- ADDED: Lasso_Logit and Lasso_MIDAS_Logit specifications.
- FIX: Ensures correct solver ('liblinear') for L1-penalized Logistic Regression.
- FIX: Dynamic Hyperparameter Grids (separates Alpha for OLS vs C for Logit).
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings

# Force Pandas Output to preserve column names for MIDAS
set_config(transform_output="pandas") 

# --- Optional Imports ---
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class Config:
    OUTPUT_ROOT = "Output_BenchmarkModels_v2.2_"
    
    # Options: 'XGBoost', 'OLS', 'Logit', 'MIDAS_Logit', 'Lasso_Logit', 'Lasso_MIDAS_Logit'
    ACTIVE_MODELS = [
        'OLS', 
        'Logit', 
        'Lasso_Logit',         
        'MIDAS_Logit', 
        'Lasso_MIDAS_Logit',   
        #'XGBoost'
    ]
    
    MODES = ["Global", "Month-Specific"]
    
    # Core Settings
    USE_MONOTONIC = True 
    USE_NATIVE_IMPUTATION = True 
    CV_FOLDS = 5
    HORIZON = 12
    TARGET = "precrisis"
    MIDAS_LAGS = 12
    
    # Data & UI
    WINSORIZE_LIMITS = [0.01, 0.99]
    SHAP_COMPLEXITY = 1 
    SHAP_SAMPLES = 500
    
    CHART_SIZE = (10, 6)
    COLORS = {
        'DARK_RED': '#C0392B',    'DARK_BLUE': '#1F618D',    
        'PASTEL_BLUE': '#AED6F1', 'YELLOW': '#F1C40F',       
        'GREY': '#B0B0B0',        'CRISIS_SHADE': '#606060', 
        'OBS_LINE': '#2C3E50',    'PRED_LINE_MAIN': '#C0392B',
        'TRAIN_BAR': '#2E86C1',   'TEST_BAR': '#E74C3C',
        'GLOBAL_LINE': '#1F618D', 'MONTHLY_LINE': '#C0392B',
        'ALMON_LINE': '#8E44AD'
    }

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.csv"

    # --- Hyperparameter Grids ---
    
    GRID_XGB = {
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_iter': [300, 500],
        'model__max_depth': [3, 5],
        'model__l2_regularization': [0.1, 1.0]
    }
    
    # Grid for OLS Lasso (Regression)
    GRID_LASSO_OLS = {
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0] 
    }
    
    # Grid for Logistic Lasso (Classification) - 'C' is inverse regularization
    # Smaller C = Stronger L1 penalty (More sparse/Lasso-like)
    GRID_LASSO_LOGIT = {
        'model__C': [0.01, 0.1, 0.5, 1, 5, 10, 50] 
    }

    GRID_MIDAS = {
        'midas__theta1': [-0.5, -0.1, -0.01, -0.001], 
        'midas__theta2': [0.0, -0.01, -0.05]
    }

    # Feature Metadata
    VARS = {
        "PCPI_PCH": ("Inflation", "Macro", 1),
        "gdp_growth": ("GDP Growth", "Macro", -1),
        "BoP_gdp": ("Current Account/GDP", "Macro", -1),
        "reserve_cover": ("FX Reserve Cover", "Macro", -1),
        "terms_of_trade": ("Terms of Trade", "Macro", -1),
        "govt_debt_gdp": ("Public Debt/GDP", "Fiscal", 1),
        "govt_deficit_gdp": ("Fiscal Balance/GDP", "Fiscal", -1),
        "tot_ext_debt_gdp": ("Total Ext Debt/GDP", "Fiscal", 1),
        "spread": ("Sovereign Spread", "Financial", 1), 
        "VIX": ("VIX Index", "Financial", 1)
    }

    @classmethod
    def get_meta(cls, var):
        return cls.VARS.get(var, (var, "Macro", 0))

# =============================================================================
# 2. MIDAS TRANSFORMER (ROBUST)
# =============================================================================

class AlmonLagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lags=12, theta1=-0.1, theta2=0.0):
        self.lags = lags
        self.theta1 = theta1
        self.theta2 = theta2
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Safety Convert to Pandas if Array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X) 
            
        # 2. Calculate Almon Weights
        k = np.arange(1, self.lags + 1)
        weights = np.exp(self.theta1 * k + self.theta2 * (k**2))
        weights = weights / np.sum(weights)
        
        X_new = pd.DataFrame(index=X.index)
        
        # 3. Find Base Columns
        base_cols = set([c.split('_L')[0] for c in X.columns if '_L' in str(c)])
        
        for col in base_cols:
            lag_cols = [f"{col}_L{i}" for i in range(1, self.lags + 1)]
            valid_lags = [c for c in lag_cols if c in X.columns]
            if not valid_lags: continue
            
            w_subset = weights[:len(valid_lags)]
            w_subset = w_subset / np.sum(w_subset)
            X_new[f"{col}_MIDAS"] = np.dot(X[valid_lags].values, w_subset)
            
        return X_new

# =============================================================================
# 3. CHARTING SUITE
# =============================================================================

def apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold', color=Config.COLORS['OBS_LINE'])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

def plot_almon_weights(theta1, theta2, lags, out_path, label):
    k = np.arange(1, lags + 1)
    weights = np.exp(theta1 * k + theta2 * (k**2))
    weights = weights / np.sum(weights)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k, weights, marker='o', linestyle='-', color=Config.COLORS['ALMON_LINE'], lw=2)
    plt.fill_between(k, 0, weights, color=Config.COLORS['ALMON_LINE'], alpha=0.2)
    apply_style(plt.gca(), f"Almon Weights: {label}\nTheta1={theta1}, Theta2={theta2}", "Lag", "Weight")
    plt.xticks(k)
    plt.savefig(os.path.join(out_path, f"Almon_Weights_{label}.png"), bbox_inches='tight')
    plt.close()

def plot_performance_suite(y_true, y_score, label, out_path):
    if len(np.unique(y_true)) < 2: return
    clean = label.replace(" ", "_").replace("/", "_")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=Config.CHART_SIZE)
    plt.plot(fpr, tpr, color=Config.COLORS['DARK_RED'], lw=3, label=f'AUC: {auc(fpr, tpr):.3f}')
    plt.plot([0,1],[0,1], color=Config.COLORS['GREY'], linestyle='--')
    apply_style(plt.gca(), f"ROC: {label}", "FPR", "TPR")
    plt.legend()
    plt.savefig(os.path.join(out_path, f"ROC_{clean}.png"), bbox_inches='tight')
    plt.close()

def plot_beta_heatmap(coef_dict, out_path, model_name):
    if not coef_dict: return
    df_coef = pd.DataFrame(coef_dict)
    df_vis = df_coef.apply(lambda x: x / x.abs().max() if x.abs().max() > 0 else x)
    
    plt.figure(figsize=(10, len(df_vis)*0.4 + 2))
    sns.heatmap(df_vis, center=0, cmap="vlag", annot=True, fmt=".2f", 
                linewidths=.5, cbar_kws={'label': 'Normalized Coef'})
    plt.title(f"Coefficient Heatmap: {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"Beta_Heatmap_{model_name}.png"))
    plt.close()

def plot_comparisons(global_aucs, monthly_aucs, out_path, model_label):
    plt.figure(figsize=(12, 6))
    if global_aucs:
        g_df = pd.DataFrame(global_aucs)
        sns.lineplot(data=g_df, x='Month', y='AUC', marker='o', lw=2, color=Config.COLORS['GLOBAL_LINE'], label='Global Model')
    if monthly_aucs:
        m_df = pd.DataFrame(monthly_aucs)
        sns.lineplot(data=m_df, x='Month', y='AUC', marker='o', lw=3, color=Config.COLORS['MONTHLY_LINE'], label='Month-Specific Model')
    plt.xticks(range(1, 13))
    plt.ylim(0.4, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    apply_style(plt.gca(), f"Monthly Evolution: {model_label}", "Month", "AUC")
    plt.legend()
    plt.savefig(os.path.join(out_path, f"Comparison_Evolution_{model_label}.png"))
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

# =============================================================================
# 4. DATA PREP
# =============================================================================

def prepare_data(add_lags=True):
    print("--- Loading Data ---")
    df = pd.read_csv(Config.FILE_PATH)
    
    if 'COUNTRY' in df.columns: df.rename(columns={'COUNTRY': 'Country'}, inplace=True)
    if 'COUNTRY_name' in df.columns: df.rename(columns={'COUNTRY_name': 'Country_Name'}, inplace=True)
    
    df['Country'] = pd.to_numeric(df['Country'], errors='coerce').fillna(0).astype(int)
    
    if os.path.exists(Config.MAPPING_FILE):
        map_df = pd.read_csv(Config.MAPPING_FILE, encoding='latin1')
        map_df['IFS'] = pd.to_numeric(map_df['IFS'], errors='coerce').fillna(0).astype(int)
        df = df.merge(map_df[['IFS', 'income', 'Area', 'Country_Name']], 
                      left_on='Country', right_on='IFS', how='left')
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

    if add_lags:
        print(f"   > Generating {Config.MIDAS_LAGS} lags for MIDAS specifications...")
        lag_cols = []
        for p in predictors:
            for l in range(1, Config.MIDAS_LAGS + 1):
                col = f"{p}_L{l}"
                df[col] = df.groupby('Country')[p].shift(l)
                lag_cols.append(col)
        all_feats = predictors + lag_cols
    else:
        all_feats = predictors

    for p in all_feats:
        lower, upper = df[p].quantile(Config.WINSORIZE_LIMITS)
        df[p] = df[p].clip(lower, upper)
        # No Imputation here (pipeline handles it)

    m_cst = [Config.get_meta(p)[2] for p in predictors] if Config.USE_MONOTONIC else None
    
    return df.dropna(subset=['Target_H']).copy(), df[df['Target_H'].isna()].copy(), predictors, m_cst

# =============================================================================
# 5. MODEL PIPELINE FACTORY
# =============================================================================

def get_model_pipeline(model_type, m_cst):
    steps = []
    param_grid = {}

    # 1. IMPUTATION (Explicit Pandas Output)
    if "XGBoost" not in model_type:
        steps.append(('imputer', SimpleImputer(strategy='median').set_output(transform="pandas")))

    # 2. MIDAS / Feature Engineering
    if "MIDAS" in model_type:
        steps.append(('midas', AlmonLagTransformer(lags=Config.MIDAS_LAGS)))
        param_grid.update(Config.GRID_MIDAS)
    else:
        if "XGBoost" not in model_type:
            steps.append(('scaler', StandardScaler().set_output(transform="pandas")))

    # 3. Estimator Selection
    
    # --- OLS Variants ---
    if model_type == 'OLS':
        steps.append(('model', LinearRegression()))
        
    elif model_type == 'Lasso_OLS':
        steps.append(('model', Lasso(max_iter=2000)))
        param_grid.update(Config.GRID_LASSO_OLS)
        
    # --- Logistic Variants ---
    elif model_type == 'Logit' or model_type == 'MIDAS_Logit':
        # Default Logit (No penalty or L2)
        steps.append(('model', LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)))
        
    elif model_type == 'Lasso_Logit' or model_type == 'Lasso_MIDAS_Logit':
        # Lasso Logit (L1 Penalty) - Requires liblinear or saga solver
        steps.append(('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000)))
        param_grid.update(Config.GRID_LASSO_LOGIT)
        
    # --- Tree Variants ---
    elif model_type == 'XGBoost':
        steps.append(('model', HistGradientBoostingClassifier(monotonic_cst=m_cst, random_state=42)))
        param_grid.update(Config.GRID_XGB)

    return Pipeline(steps), param_grid

# =============================================================================
# 6. ENGINE
# =============================================================================

def run_engine():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    d_tr, d_fc, preds, constraints = prepare_data(add_lags=True)
    
    results = []
    cv_t = TimeSeriesSplit(n_splits=Config.CV_FOLDS)

    for model_type in Config.ACTIVE_MODELS:
        g_aucs_evolution = []
        m_aucs_evolution = []
        coeffs_storage = {}

        for mode in Config.MODES:
            print(f"\n>>> Running: {model_type} - {mode}")
            path = os.path.join(Config.OUTPUT_ROOT, model_type, mode)
            os.makedirs(path, exist_ok=True)
            
            d_tr['Risk_Index'] = np.nan
            d_fc['Risk_Index'] = np.nan
            
            # Feature Selection (Lags vs Base)
            if "MIDAS" in model_type:
                lag_cols = [c for c in d_tr.columns if "_L" in c]
                X_cols = preds + lag_cols
            else:
                X_cols = preds

            # --- TRAIN ---
            if mode == "Global":
                pipe, grid = get_model_pipeline(model_type, constraints)
                search = RandomizedSearchCV(pipe, grid, n_iter=15, cv=cv_t, scoring='roc_auc', n_jobs=-1, random_state=42)
                search.fit(d_tr[X_cols], d_tr['Target_H'])
                best = search.best_estimator_
                
                # Almon Weights
                if "MIDAS" in model_type:
                    t1 = search.best_params_.get('midas__theta1', -0.1)
                    t2 = search.best_params_.get('midas__theta2', 0.0)
                    plot_almon_weights(t1, t2, Config.MIDAS_LAGS, path, f"{model_type}_{mode}")
                    print(f"    > Optimal Almon: Theta1={t1}, Theta2={t2}")
                
                # Predict
                if hasattr(best.named_steps['model'], "predict_proba"):
                    d_tr['Risk_Index'] = best.predict_proba(d_tr[X_cols])[:, 1]
                    d_fc['Risk_Index'] = best.predict_proba(d_fc[X_cols])[:, 1]
                else:
                    d_tr['Risk_Index'] = np.clip(best.predict(d_tr[X_cols]), 0, 1)
                    d_fc['Risk_Index'] = np.clip(best.predict(d_fc[X_cols]), 0, 1)

                # Coefficients
                if hasattr(best.named_steps['model'], 'coef_'):
                    c = best.named_steps['model'].coef_.flatten()
                    feat_names = [f"{p}_MIDAS" for p in preds] if "MIDAS" in model_type else preds
                    if len(c) == len(feat_names): coeffs_storage['Global'] = pd.Series(c, index=feat_names)

                for m in range(1, 13):
                    m_sub = d_tr[d_tr['month'] == m]
                    if len(m_sub) > 20:
                        sc = roc_auc_score(m_sub['Target_H'], m_sub['Risk_Index'])
                        g_aucs_evolution.append({'Month': m, 'AUC': sc})

            elif mode == "Month-Specific":
                for m in range(1, 13):
                    m_tr = d_tr[d_tr['month'] == m]
                    m_fc = d_fc[d_fc['month'] == m]
                    if len(m_tr) < 30: continue
                    
                    sub_cv = TimeSeriesSplit(n_splits=3)
                    pipe, grid = get_model_pipeline(model_type, constraints)
                    search = RandomizedSearchCV(pipe, grid, n_iter=10, cv=sub_cv, scoring='roc_auc', n_jobs=-1, random_state=42)
                    search.fit(m_tr[X_cols], m_tr['Target_H'])
                    best = search.best_estimator_
                    
                    if "MIDAS" in model_type:
                        t1 = search.best_params_.get('midas__theta1', -0.1)
                        t2 = search.best_params_.get('midas__theta2', 0.0)
                        if m in [1, 6, 12]: 
                            plot_almon_weights(t1, t2, Config.MIDAS_LAGS, path, f"{model_type}_M{m:02d}")

                    if hasattr(best.named_steps['model'], "predict_proba"):
                        d_tr.loc[d_tr['month'] == m, 'Risk_Index'] = best.predict_proba(m_tr[X_cols])[:, 1]
                        if not m_fc.empty:
                            d_fc.loc[d_fc['month'] == m, 'Risk_Index'] = best.predict_proba(m_fc[X_cols])[:, 1]
                    else:
                        d_tr.loc[d_tr['month'] == m, 'Risk_Index'] = np.clip(best.predict(m_tr[X_cols]), 0, 1)
                        if not m_fc.empty:
                            d_fc.loc[d_fc['month'] == m, 'Risk_Index'] = np.clip(best.predict(m_fc[X_cols]), 0, 1)
                    
                    sc = roc_auc_score(m_tr['Target_H'], d_tr.loc[d_tr['month'] == m, 'Risk_Index'])
                    m_aucs_evolution.append({'Month': m, 'AUC': sc})

                    if hasattr(best.named_steps['model'], 'coef_'):
                        c = best.named_steps['model'].coef_.flatten()
                        feat_names = [f"{p}_MIDAS" for p in preds] if "MIDAS" in model_type else preds
                        if len(c) == len(feat_names): coeffs_storage[f'M{m}'] = pd.Series(c, index=feat_names)

            # --- DIAGNOSTICS ---
            valid = d_tr['Risk_Index'].notna()
            if valid.any():
                auc_total = roc_auc_score(d_tr.loc[valid, 'Target_H'], d_tr.loc[valid, 'Risk_Index'])
                results.append({'Model': model_type, 'Mode': mode, 'Type': 'Total', 'Value': 'All', 'Test_AUC': auc_total})
                
                for cat in ['income', 'Area']:
                    if cat in d_tr.columns:
                        for val in d_tr[cat].unique():
                            sub = d_tr[(d_tr[cat] == val) & valid]
                            if len(sub) > 50 and sub['Target_H'].nunique() > 1:
                                ga = roc_auc_score(sub['Target_H'], sub['Risk_Index'])
                                results.append({'Model': model_type, 'Mode': mode, 'Type': 'Group', 'Group': cat, 'Value': val, 'Test_AUC': ga})

                if mode == "Global":
                    full = pd.concat([d_tr, d_fc])
                    for c in full['Country_Name'].unique():
                        sub = full[full['Country_Name'] == c].sort_values('Date')
                        if len(sub) > 12: plot_country_chart(sub, c, path, f"{model_type}_{mode}")

        comp_path = os.path.join(Config.OUTPUT_ROOT, model_type)
        os.makedirs(comp_path, exist_ok=True)
        plot_comparisons(g_aucs_evolution, m_aucs_evolution, comp_path, model_type)
        plot_beta_heatmap(coeffs_storage, comp_path, model_type)

    print("\n=== FINAL RESULTS (Summary) ===")
    res_df = pd.DataFrame(results)
    print(res_df.head(20))
    res_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "FINAL_AUC_SUMMARY.csv"), index=False)
    
    # 
    plt.figure(figsize=(12, 6))
    sns.barplot(data=res_df[res_df['Type']=='Total'], x='Model', y='Test_AUC', hue='Mode', palette='viridis')
    plt.ylim(0.5, 1.0)
    plt.title("Model Comparison (Global vs Month-Specific)")
    plt.savefig(os.path.join(Config.OUTPUT_ROOT, "Final_Model_Comparison.png"))
    plt.close()

if __name__ == "__main__":
    run_engine()