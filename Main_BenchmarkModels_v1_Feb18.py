# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 18:28:20 2026

@author: cmarsilli
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 22:00:00 2026
@author: cmarsilli & Gemini
Fiscal crisis model v1 (Multi-Model: OLS, Logit, Lasso, MIDAS)

NEW FEATURES:
- Added Linear Probability Models (OLS, Lasso OLS)
- Added Logistic Models (Logit, Lasso Logit)
- Added MIDAS (Exponential Almon Lag) specifications
- GridSearch for Almon hyperparameters and Lasso Lambda
- Coefficient (Beta) Heatmaps
- Income & Area Risk vs Frequency Charts
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class Config:
    OUTPUT_ROOT = "Output_BenchmarkModels"
    
    # Define which models to run
    # Options: 'XGBoost', 'OLS', 'Logit', 'Lasso_OLS', 'Lasso_Logit', 'MIDAS_Logit', 'Lasso_MIDAS_Logit'
    ACTIVE_MODELS = [
        'OLS', 
        'Logit', 
        'MIDAS_Logit', 
        'Lasso_MIDAS_Logit', 
        #'XGBoost' # Keeping reference
    ]
    
    MODES = ["Global", "Month-Specific"]
    
    # Core Settings
    CV_FOLDS = 5
    HORIZON = 12
    TARGET = "precrisis"
    MIDAS_LAGS = 12  # How many months back to look for MIDAS
    
    # Data & UI
    WINSORIZE_LIMITS = [0.01, 0.99]
    CHART_SIZE = (10, 8)
    COLORS = {
        'DARK_RED': '#C0392B',    'DARK_BLUE': '#1F618D',    
        'PASTEL_BLUE': '#AED6F1', 'YELLOW': '#F1C40F',       
        'GREY': '#B0B0B0',        'CRISIS_SHADE': '#606060', 
        'OBS_LINE': '#2C3E50',    'PRED_LINE_MAIN': '#C0392B',
        'TRAIN_BAR': '#2E86C1',   'TEST_BAR': '#E74C3C'
    }

    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    MAPPING_FILE = "Mapping.xlsx - Sheet2.csv"

    # --- Hyperparameter Grids ---
    
    # XGBoost
    GRID_XGB = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [300, 500],
        'max_depth': [3, 5],
        'l2_regularization': [0.1, 1.0]
    }
    
    # Lasso (Lambda/Alpha)
    GRID_LASSO = {
        'model__C': [0.01, 0.1, 1, 10, 100], # For Logistic (Inverse Lambda)
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0] # For OLS Lasso
    }

    # MIDAS Almon Params (Theta 1 & 2 for exponential polynomial)
    GRID_MIDAS = {
        'midas__theta1': [-0.1, -0.01, -0.001], # Decay rates
        'midas__theta2': [0.0, -0.01]           # Curvature
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
# 2. MIDAS / ALMON LAG TRANSFORMER
# =============================================================================

class AlmonLagTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Exponential Almon Lag weights to features to create MIDAS terms.
    The lag structure is controlled by theta1 and theta2.
    """
    def __init__(self, lags=12, theta1=-0.1, theta2=0.0):
        self.lags = lags
        self.theta1 = theta1
        self.theta2 = theta2
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expects X to be a 3D array or Panel logic handled externally.
        # However, specifically for Scikit-Learn compatibility, we assume 
        # X contains columns that need lag weighting. 
        # Since sklearn passes 2D arrays, we must generate the lagged data BEFORE 
        # this step or handle the lag generation here if we have time index.
        
        # Simplified MIDAS Approach for Panel Data:
        # We compute the weights once:
        k = np.arange(1, self.lags + 1)
        weights = np.exp(self.theta1 * k + self.theta2 * (k**2))
        weights = weights / np.sum(weights) # Normalize
        
        # We assume the input X already has columns formatted as "Var_L1", "Var_L2"...
        # OR we perform a weighted average of existing lag columns.
        
        # To make this robust: We will assume X is the RAW data and we can't do time-shifts here easily
        # without group info. 
        # STRATEGY: This transformer expects the input dataframe to HAVE lags columns present.
        # It collapses them into a single weighted "MIDAS" column per feature.
        
        X_new = pd.DataFrame(index=X.index)
        
        # Identify base features
        base_cols = set([c.split('_L')[0] for c in X.columns if '_L' in c])
        
        for col in base_cols:
            # Gather all lags for this column
            lag_cols = [f"{col}_L{i}" for i in range(1, self.lags + 1)]
            
            # If lags missing, skip
            valid_lags = [c for c in lag_cols if c in X.columns]
            if not valid_lags: continue
                
            # Apply weights (dot product of lags and weights)
            # Slice weights to match found lags
            w_subset = weights[:len(valid_lags)]
            w_subset = w_subset / np.sum(w_subset) # Re-normalize
            
            X_new[f"{col}_MIDAS"] = np.dot(X[valid_lags].values, w_subset)
            
        return X_new

# =============================================================================
# 3. CHARTING SUITE (Enhanced)
# =============================================================================

def apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold', color=Config.COLORS['OBS_LINE'])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

def plot_beta_heatmap(coef_dict, out_path, model_name):
    """Generates a heatmap of model coefficients (Betas)."""
    if not coef_dict: return
    
    # Create DataFrame from coeffs (Cols = Models/Months, Rows = Features)
    df_coef = pd.DataFrame(coef_dict)
    
    # Normalize for visualization (MinMax or Scale) to compare relative importance
    # Use simple scaling -1 to 1 for visual clarity if magnitudes vary widely
    df_vis = df_coef.apply(lambda x: x / x.abs().max() if x.abs().max() > 0 else x)
    
    plt.figure(figsize=(10, len(df_vis)*0.4 + 2))
    sns.heatmap(df_vis, center=0, cmap="vlag", annot=True, fmt=".2f", 
                linewidths=.5, cbar_kws={'label': 'Normalized Coefficient'})
    plt.title(f"Coefficient Heatmap: {model_name} (Normalized)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"Beta_Heatmap_{model_name}.png"))
    plt.close()

def plot_group_risk_charts(full_df, out_path):
    """Charts comparing Average Risk vs Crisis Frequency by Income and Area."""
    p = os.path.join(out_path, "charts/group_analysis")
    os.makedirs(p, exist_ok=True)
    
    for group_col in ['income', 'Area']:
        if group_col not in full_df.columns: continue
        
        # 1. Aggregate metrics
        agg = full_df.groupby(group_col).agg({
            'Risk_Index': 'mean',
            'Target_H': 'mean', # This approximates crisis frequency
            'Country': 'nunique'
        }).reset_index()
        
        # Melt for plotting
        melted = agg.melt(id_vars=[group_col, 'Country'], 
                          value_vars=['Risk_Index', 'Target_H'],
                          var_name='Metric', value_name='Value')
        
        melted['Metric'] = melted['Metric'].map({'Risk_Index': 'Avg Predicted Risk', 'Target_H': 'Actual Crisis Freq'})
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=melted, x=group_col, y='Value', hue='Metric', 
                    palette={ 'Avg Predicted Risk': Config.COLORS['TRAIN_BAR'], 'Actual Crisis Freq': Config.COLORS['DARK_RED'] })
        
        plt.title(f"Model Calibration by {group_col}", fontsize=14, fontweight='bold')
        plt.ylabel("Probability / Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(p, f"Risk_vs_Freq_{group_col}.png"))
        plt.close()

# =============================================================================
# 4. DATA PREP
# =============================================================================

def prepare_data(add_lags=False):
    print("--- Loading and Preparing Data ---")
    df = pd.read_csv(Config.FILE_PATH)
    
    if 'COUNTRY_name' in df.columns: 
        df.rename(columns={'COUNTRY_name': 'Country'}, inplace=True)
    
    # Merge Mapping
    if os.path.exists(Config.MAPPING_FILE):
        map_df = pd.read_csv(Config.MAPPING_FILE, encoding='latin1')
        df['Merge_Key'] = df['Country'].astype(str).str.split('.').str[0].str.strip()
        map_df['Merge_Key'] = map_df['IFS'].astype(str).str.split('.').str[0].str.strip()
        df = df.merge(map_df[['Merge_Key', 'income', 'Area', 'Country_Name']], on='Merge_Key', how='left')
        df['Country_Name'] = df['Country_Name'].fillna(df['Country'].astype(str))
    else:
        df['Country_Name'] = df['Country']

    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    df = df.sort_values(['Country', 'Date']) 
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)

    # Base Predictors
    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    
    # --- Generate Lags for MIDAS ---
    # Even if not running MIDAS, we generate them now so the Transformer can find them.
    if add_lags:
        print(f"   > Generating {Config.MIDAS_LAGS} lags for MIDAS specifications...")
        lag_cols = []
        for p in predictors:
            for l in range(1, Config.MIDAS_LAGS + 1):
                col_name = f"{p}_L{l}"
                df[col_name] = df.groupby('Country')[p].shift(l)
                lag_cols.append(col_name)
    
    # Handling missing/winsorization
    all_numeric = predictors + (lag_cols if add_lags else [])
    
    for p in all_numeric:
        # Simple Winsorization
        lower = df[p].quantile(Config.WINSORIZE_LIMITS[0])
        upper = df[p].quantile(Config.WINSORIZE_LIMITS[1])
        df[p] = df[p].clip(lower, upper)
        # Simple Fill
        df[p] = df[p].fillna(df[p].median())

    train_df = df.dropna(subset=['Target_H']).copy()
    forecast_df = df[df['Target_H'].isna()].copy()
    
    return train_df, forecast_df, predictors

# =============================================================================
# 5. ENGINE: MULTI-MODEL SUPPORT
# =============================================================================

def get_model_pipeline(model_type, predictors):
    """Constructs the sklearn pipeline based on model type."""
    
    # 1. Base Steps (Scaling is crucial for OLS/Lasso/Logit)
    steps = []
    
    # 2. MIDAS Logic
    if "MIDAS" in model_type:
        steps.append(('midas', AlmonLagTransformer(lags=Config.MIDAS_LAGS)))
        # For MIDAS, input features are the *original* ones, the transformer handles finding Lags
        # But we need to ensure the grid search knows which params to tune
        param_grid = Config.GRID_MIDAS.copy()
    else:
        # Standard Scaling for non-tree models
        if "XGBoost" not in model_type:
            steps.append(('scaler', StandardScaler()))
        param_grid = {}

    # 3. Estimator Logic
    if model_type == 'OLS':
        steps.append(('model', LinearRegression()))
        
    elif model_type == 'Logit' or model_type == 'MIDAS_Logit':
        steps.append(('model', LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)))
        
    elif model_type == 'Lasso_OLS':
        steps.append(('model', Lasso(max_iter=2000)))
        param_grid.update(Config.GRID_LASSO)
        
    elif model_type == 'Lasso_Logit' or model_type == 'Lasso_MIDAS_Logit':
        steps.append(('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)))
        param_grid.update(Config.GRID_LASSO)
        
    elif model_type == 'XGBoost':
        steps.append(('model', HistGradientBoostingClassifier(random_state=42)))
        param_grid = Config.GRID_XGB

    pipeline = Pipeline(steps)
    return pipeline, param_grid

def run_engine(d_tr, d_fc, predictors, model_type, mode):
    print(f"\n>>> Running: {model_type} ({mode})")
    
    # Define Input Columns
    # If MIDAS, we need the LAG columns in X. If standard, just base predictors.
    if "MIDAS" in model_type:
        # Include lags in input matrix
        lag_cols = [c for c in d_tr.columns if "_L" in c]
        X_cols = predictors + lag_cols # Pass everything, transformer filters
    else:
        X_cols = predictors
        
    save_path = os.path.join(Config.OUTPUT_ROOT, model_type, mode)
    os.makedirs(save_path, exist_ok=True)
    
    d_tr['Risk_Index'] = np.nan
    d_fc['Risk_Index'] = np.nan
    coeffs_storage = {} # To store betas
    
    # Cross-Validation Strategy
    cv = TimeSeriesSplit(n_splits=Config.CV_FOLDS)

    # --- Training Loop ---
    if mode == "Global":
        pipeline, grid = get_model_pipeline(model_type, predictors)
        
        # Use RandomizedSearchCV for speed, or Grid if small
        search = RandomizedSearchCV(pipeline, grid, n_iter=15, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42)
        search.fit(d_tr[X_cols], d_tr['Target_H'])
        best_model = search.best_estimator_
        
        # Predict
        if hasattr(best_model.named_steps['model'], "predict_proba"):
            # Classifiers (Logit, XGB)
            d_tr['Risk_Index'] = best_model.predict_proba(d_tr[X_cols])[:, 1]
            d_fc['Risk_Index'] = best_model.predict_proba(d_fc[X_cols])[:, 1]
        else:
            # Regressors (OLS, Lasso OLS) - Clamp Output
            d_tr['Risk_Index'] = np.clip(best_model.predict(d_tr[X_cols]), 0, 1)
            d_fc['Risk_Index'] = np.clip(best_model.predict(d_fc[X_cols]), 0, 1)

        # Store Coefficients
        if 'model' in best_model.named_steps:
            est = best_model.named_steps['model']
            if hasattr(est, 'coef_'):
                # Handle flattened coefs
                c = est.coef_.flatten()
                # Mapping names is tricky with Pipelines, simplified here:
                # If MIDAS, feature names changed. If Standard, they are predictors.
                if "MIDAS" in model_type:
                    # MIDAS creates one col per predictor
                    feats = [f"{p}_MIDAS" for p in predictors]
                    # Note: Lasso might select subsets, shape check needed
                    if len(c) == len(feats):
                        coeffs_storage['Global'] = pd.Series(c, index=feats)
                elif len(c) == len(predictors):
                    coeffs_storage['Global'] = pd.Series(c, index=predictors)

    elif mode == "Month-Specific":
        for m in range(1, 13):
            m_tr = d_tr[d_tr['month'] == m]
            m_fc = d_fc[d_fc['month'] == m]
            if len(m_tr) < 30: continue
            
            pipeline, grid = get_model_pipeline(model_type, predictors)
            search = RandomizedSearchCV(pipeline, grid, n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
            search.fit(m_tr[X_cols], m_tr['Target_H'])
            best_m = search.best_estimator_
            
            # Predict
            if hasattr(best_m.named_steps['model'], "predict_proba"):
                d_tr.loc[d_tr['month'] == m, 'Risk_Index'] = best_m.predict_proba(m_tr[X_cols])[:, 1]
                if not m_fc.empty:
                    d_fc.loc[d_fc['month'] == m, 'Risk_Index'] = best_m.predict_proba(m_fc[X_cols])[:, 1]
            else:
                d_tr.loc[d_tr['month'] == m, 'Risk_Index'] = np.clip(best_m.predict(m_tr[X_cols]), 0, 1)
                if not m_fc.empty:
                    d_fc.loc[d_fc['month'] == m, 'Risk_Index'] = np.clip(best_m.predict(m_fc[X_cols]), 0, 1)

            # Store Beta
            est = best_m.named_steps['model']
            if hasattr(est, 'coef_'):
                c = est.coef_.flatten()
                if "MIDAS" in model_type and len(c) == len(predictors):
                    coeffs_storage[f'M{m}'] = pd.Series(c, index=[f"{p}_MIDAS" for p in predictors])
                elif len(c) == len(predictors):
                    coeffs_storage[f'M{m}'] = pd.Series(c, index=predictors)

    # --- Diagnostics & Saving ---
    valid_mask = d_tr['Risk_Index'].notna()
    if not valid_mask.any(): return 0.0
    
    auc_val = roc_auc_score(d_tr.loc[valid_mask, 'Target_H'], d_tr.loc[valid_mask, 'Risk_Index'])
    
    # Generate Heatmap of Betas
    plot_beta_heatmap(coeffs_storage, save_path, f"{model_type}_{mode}")
    
    # Save Full Results
    full_df = pd.concat([d_tr, d_fc])
    full_df.to_csv(os.path.join(save_path, f"Detailed_Results_{model_type}.csv"), index=False)
    
    # Charts (Reusing logic from previous tools)
    plot_group_risk_charts(full_df, save_path)
    
    return auc_val

# =============================================================================
# 6. EXECUTION
# =============================================================================

if __name__ == "__main__":
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    
    # Prep data (Add lags for MIDAS capability)
    d_tr, d_fc, preds = prepare_data(add_lags=True)
    
    summary_results = []
    
    for model in Config.ACTIVE_MODELS:
        for mode in Config.MODES:
            try:
                auc_score = run_engine(d_tr.copy(), d_fc.copy(), preds, model, mode)
                summary_results.append({
                    'Model': model,
                    'Mode': mode,
                    'AUC': auc_score
                })
                print(f"   > Result: {model} ({mode}) = AUC {auc_score:.3f}")
            except Exception as e:
                print(f"   > FAILED: {model} ({mode}) - Error: {e}")

    # Final Comparison Chart
    res_df = pd.DataFrame(summary_results)
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=res_df, x='Model', y='AUC', hue='Mode', palette='viridis')
        plt.title("Model Comparison: AUC Scores", fontsize=14)
        plt.ylim(0.5, 1.0)
        plt.savefig(os.path.join(Config.OUTPUT_ROOT, "FINAL_Model_Comparison.png"))
        
        res_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "Final_AUC_Table.csv"), index=False)
        print("\n=== Final Results ===")
        print(res_df)
    
    print(f"\n[SUCCESS] Multi-model run complete. Check {Config.OUTPUT_ROOT}")