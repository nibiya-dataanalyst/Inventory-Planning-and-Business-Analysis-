"""
Operational Inventory Policy Module (Safety Stock, ROP, EOQ)
Author: Nibiya Nadir
"""

import pandas as pd
import numpy as np

def calculate_inventory_policies(sku_summary, forecast_df, lead_time_days=7, service_level_z=1.65, holding_cost_pct=0.20, order_cost=50.0):
    """
    Translates demand forecasts into concrete operational supply chain controls:
    Safety Stock, Reorder Point (ROP), and Economic Order Quantity (EOQ).
    
    Parameters:
    -----------
    sku_summary : pd.DataFrame
        SKU metadata table including ABC_XYZ classification and unit price.
    forecast_df : pd.DataFrame
        Predictions table with columns ['SKU_ID', 'predicted_demand'].
    lead_time_days : int
        Supplier replenishment lead time in days (default 7).
    service_level_z : float
        Z-score corresponding to desired cycle service level (1.65 = 95%).
    holding_cost_pct : float
        Annual inventory holding cost percentage (default 20%).
    order_cost : float
        Fixed administrative cost per order placement (default $50).
        
    Returns:
    --------
    pd.DataFrame with full inventory policy metrics.
    """
    policies = []
    
    for _, row in sku_summary.iterrows():
        sku = row['SKU_ID']
        sku_forecasts = forecast_df[forecast_df['SKU_ID'] == sku]
        
        if sku_forecasts.empty:
            continue
            
        avg_daily_demand = sku_forecasts['predicted_demand'].mean()
        demand_std_dev = sku_forecasts['predicted_demand'].std()
        if pd.isna(demand_std_dev):
            demand_std_dev = 0.0
            
        # 1. Safety Stock = Z * sqrt(Lead Time) * StdDev(Demand)
        safety_stock = service_level_z * np.sqrt(lead_time_days) * demand_std_dev
        
        # 2. Reorder Point (ROP) = (Avg Daily Demand * Lead Time) + Safety Stock
        rop = (avg_daily_demand * lead_time_days) + safety_stock
        
        # 3. Economic Order Quantity (EOQ) = sqrt((2 * Annual Demand * Order Cost) / Unit Holding Cost)
        annual_demand = avg_daily_demand * 365
        unit_price = row.get('unit_price', 25.0)  # Default fallback unit price
        holding_cost = unit_price * holding_cost_pct
        
        if holding_cost > 0 and annual_demand > 0:
            eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
        else:
            eoq = 0.0
            
        policies.append({
            'SKU_ID': sku,
            'ABC_XYZ': row['ABC_XYZ'],
            'Avg_Daily_Demand': round(avg_daily_demand, 2),
            'Demand_StdDev': round(demand_std_dev, 2),
            'Safety_Stock_Units': int(np.ceil(safety_stock)),
            'Reorder_Point_Units': int(np.ceil(rop)),
            'EOQ_Units': int(np.ceil(eoq))
        })
        
    return pd.DataFrame(policies)
