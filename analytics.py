"""
Business Analytics & ABC-XYZ Inventory Matrix Module
Author: Nibiya Nadir
"""

import pandas as pd
import numpy as np

def compute_abc_xyz_matrix(df):
    """
    Computes ABC (based on cumulative revenue value via Pareto principle) and 
    XYZ (based on demand coefficient of variation CV = std/mean) classifications.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns ['SKU_ID', 'revenue', 'demand']
        
    Returns:
    --------
    pd.DataFrame with SKU metrics, ABC class, XYZ class, and combined ABC_XYZ segment.
    """
    # Aggregate revenue and demand statistics per SKU
    sku_summary = df.groupby('SKU_ID').agg(
        total_revenue=('revenue', 'sum'),
        total_demand=('demand', 'sum'),
        mean_demand=('demand', 'mean'),
        std_demand=('demand', 'std')
    ).reset_index()
    
    # Handle zero/missing standard deviation cases safely
    sku_summary['std_demand'] = sku_summary['std_demand'].fillna(0)
    sku_summary['cv'] = np.where(
        sku_summary['mean_demand'] > 0, 
        sku_summary['std_demand'] / sku_summary['mean_demand'], 
        0
    )
    
    # --- ABC Classification (Pareto Cumulative Revenue Rule) ---
    sku_summary = sku_summary.sort_values(by='total_revenue', ascending=False).reset_index(drop=True)
    sku_summary['cum_revenue_pct'] = sku_summary['total_revenue'].cumsum() / sku_summary['total_revenue'].sum()
    
    def assign_abc(pct):
        if pct <= 0.80:
            return 'A'
        elif pct <= 0.95:
            return 'B'
        else:
            return 'C'
        
    sku_summary['ABC_Class'] = sku_summary['cum_revenue_pct'].apply(assign_abc)
    
    # --- XYZ Classification (Demand Volatility Rule) ---
    # CV < 0.5 = High Predictability (X), 0.5 <= CV <= 1.0 = Moderate (Y), CV > 1.0 = Erratic (Z)
    def assign_xyz(cv):
        if cv < 0.5:
            return 'X'
        elif cv <= 1.0:
            return 'Y'
        else:
            return 'Z'
        
    sku_summary['XYZ_Class'] = sku_summary['cv'].apply(assign_xyz)
    sku_summary['ABC_XYZ'] = sku_summary['ABC_Class'] + sku_summary['XYZ_Class']
    
    return sku_summary
