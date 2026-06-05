# -*- coding: utf-8 -*-
"""
Sovereign Crisis Forecasting Engine v5.5 (Deep-Dive Edition)


"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False

# =============================================================================
# 1. CONFIGURATION & MASTER METADATA
# =============================================================================

class Config:
    OUTPUT_NAME = "Output_v3_Monthly"
    
    # --- SHAP CONTROLS ---
    COMPUTE_SHAP = True
    COMPUTE_SHAP_INTERACTIONS = True  # WARNING: Computationally intensive
    INTERACTION_TOP_N = 5             # Number of top features to compute interactions for
    
    HORIZON = 12                      # Forecast horizon in months
    MODEL_MODE = "Month-Specific"     # "Global" or "Month-Specific"
    CV_TYPE = "Temporal"              # "Temporal" (TimeSeriesSplit) or "Random" (StratifiedKFold)
    TEST_SHARE = 0.15                 # Final out-of-sample holdout (last X% of data)
    CV_FOLDS = 5                      # Internal folds for hyperparameter optimization
    
    FILE_PATH = "fiscal_data_HF_monthly_2025-09-15.csv"
    TARGET = "precrisis"
    
    PARAM_GRID = {
        'learning_rate': [0.01, 0.05],
        'max_iter': [300, 500],
        'max_depth': [3, 4, 5],
        'l2_regularization': [0.1, 1.0, 10.0],
        'min_samples_leaf': [20, 50],
    }

    VARS = {
        "oil_price": ("Oil Price", "Global", 0),
        "VIX": ("VIX Index", "Global", 1),
        "oil_shock_impact": ("Oil Shock Impact", "Interaction", -1), 
        "debt_fx_vulnerability": ("Debt x FX Shock", "Interaction", 1), 
        
        "PCPI_PCH": ("Inflation", "Macro", 1),
        "gdp_growth": ("GDP Growth", "Macro", -1),
        "BoP_gdp": ("Current Account/GDP", "Macro", -1),
        "reserve_cover": ("FX Reserve Cover", "Macro", -1),
        "terms_of_trade": ("Terms of Trade", "Macro", -1),
        "GDP_percapita_over_US_12ma": ("GDP per Capita", "Macro", -1), 
        "oil_to_gdp": ("OilExports/GDP", "Macro", -1), 
        
        "govt_debt_gdp": ("Public Debt/GDP", "Fiscal", 1),
        "govt_deficit_gdp": ("Fiscal Deficit/GDP", "Fiscal", 1),
        "govt_revenue_gdp": ("Fiscal Revenue/GDP", "Fiscal", -1),
        "tot_ext_debt_gdp": ("Total Ext Debt/GDP", "Fiscal", 1),
        "debt_service_gdp": ("Debt Service/GDP", "Fiscal", 1),
        "corruption_12ma": ("Corruption", "Fiscal", -1),
        
        "deposit_rate": ("ST Rate", "Financial Conditions", 1),
        "long_term_bond_yield": ("LT Rate", "Financial Conditions", 1),
        "WUI": ("Uncertainty Idx", "Financial Conditions", 1),
        "ENDE_yoy": ("FX Depreciation", "Financial Conditions", 1),
        "spread": ("Sovereign Spread", "Financial Conditions", 1)
    }

    @classmethod
    def get_meta(cls, var):
        return cls.VARS.get(var, (var, "Macroeconomic", 0))

# =============================================================================
# 2. DATA PREPARATION & ENGINE
# =============================================================================

def prepare_data():
    df = pd.read_csv(Config.FILE_PATH)
    if 'COUNTRY_name' in df.columns: df.rename(columns={'COUNTRY_name': 'Country'}, inplace=True)
    df['Date'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str) + "-01")
    df = df.sort_values(['Country', 'Date'])

    # Engineering Interactions
    df['oil_price_pch'] = df.groupby('Country')['oil_price'].pct_change()
    df['oil_shock_impact'] = df['oil_to_gdp'] * df['oil_price_pch']
    df['debt_fx_vulnerability'] = df['ENDE_yoy'] * df['tot_ext_debt_gdp']

    predictors = [p for p in Config.VARS.keys() if p in df.columns]
    df['Target_H'] = df.groupby('Country')[Config.TARGET].shift(-Config.HORIZON)
    df[predictors] = df.groupby('Country')[predictors].ffill().fillna(df[predictors].median())
    
    return df.dropna(subset=['Target_H']).copy(), predictors

def run_engine():
    out_root = Config.OUTPUT_NAME
    subdirs = ["data", "charts", "country_charts", "shap/dependence", "shap/interactions", "shap/seasonal", "shap/country_deep_dives"]
    for d in subdirs: os.makedirs(os.path.join(out_root, d), exist_ok=True)
    
    df, predictors = prepare_data()
    constraints = [Config.get_meta(p)[2] for p in predictors]
    modes = [(f"Month_{m}", df[df['month'] == m]) for m in range(1, 13)] if Config.MODEL_MODE == "Month-Specific" else [("Global", df)]
    
    reconstructed_list, all_shap_values, all_X_data = [], [], []

    print(f"--- Running {Config.MODEL_MODE} Engine ---")
    for mode_name, sub_df in modes:
        sub_df = sub_df.sort_values('Date')
        X, y = sub_df[predictors], sub_df['Target_H']
        split_idx = int(len(sub_df) * (1 - Config.TEST_SHARE))
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        
        if len(y_train.unique()) < 2: continue
        model = RandomizedSearchCV(HistGradientBoostingClassifier(monotonic_cst=constraints), Config.PARAM_GRID, n_iter=5, cv=TimeSeriesSplit(n_splits=Config.CV_FOLDS), scoring='roc_auc', n_jobs=-1)
        model.fit(X_train, y_train)
        best_clf = model.best_estimator_
        
        sub_df['Risk_Index'] = best_clf.predict_proba(X)[:, 1]
        reconstructed_list.append(sub_df)

        if Config.COMPUTE_SHAP and SHAP_INSTALLED:
            explainer = shap.Explainer(best_clf.predict_proba, X)
            all_shap_values.append(explainer(X)[:, :, 1].values)
            all_X_data.append(X)

    full_df = pd.concat(reconstructed_list).sort_values(['Country', 'Date'])
    
    # --- ADVANCED ANALYTICS ---
    if Config.COMPUTE_SHAP and SHAP_INSTALLED:
        cat_sv = np.concatenate(all_shap_values, axis=0)
        cat_X = pd.concat(all_X_data, axis=0)
        sh_exp = shap.Explanation(values=cat_sv, data=cat_X.values, feature_names=predictors)

        # 1. Beeswarm & Dependence
        shap.plots.beeswarm(sh_exp, show=False); plt.savefig(os.path.join(out_root, "shap/beeswarm.png"), bbox_inches='tight'); plt.close()
        for feat in predictors[:Config.INTERACTION_TOP_N]:
            plt.figure(); shap.plots.scatter(sh_exp[:, feat], color=sh_exp, show=False)
            sns.regplot(x=cat_X[feat], y=cat_sv[:, predictors.index(feat)], scatter=False, color='black', lowess=True)
            plt.savefig(os.path.join(out_root, f"shap/dependence/{feat}_smooth.png")); plt.close()

        # 2. Country Deep-Dives: SHAP Decomposition Over Time
        for country, c_data in full_df.groupby('Country'):
            plt.figure(figsize=(12, 6))
            c_indices = c_data.index.tolist()
            # Simplified proxy for temporal SHAP decomposition
            c_shap = pd.DataFrame(cat_sv[c_data.index], columns=predictors, index=c_data['Date'])
            c_shap.iloc[:, :8].plot(kind='area', stacked=True, alpha=0.5)
            plt.title(f"Risk Drivers Over Time: {country}"); plt.legend(loc='upper left', bbox_to_anchor=(1,1))
            plt.savefig(os.path.join(out_root, f"shap/country_deep_dives/{country}_decomposition.png"), bbox_inches='tight'); plt.close()

    print(f"Deep-Dive Completed. Data in: {out_root}")

if __name__ == "__main__":
    run_engine()