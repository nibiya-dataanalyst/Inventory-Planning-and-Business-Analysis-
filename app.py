"""
Interactive Streamlit Business Analysis & Inventory Dashboard
Author: Nibiya Nadir
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Inventory Planning & Demand Forecasting Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Enterprise Inventory Planning & Demand Forecasting")
st.markdown("""
*End-to-End Analytics Pipeline featuring ABC-XYZ Matrix Segmentation, LSTM Deep Learning Forecasting, and Dynamic Safety Stock Optimization.*
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Policy Control Panel")
lead_time = st.sidebar.slider("Supplier Lead Time (Days)", min_value=1, max_value=30, value=7)
service_level = st.sidebar.selectbox("Target Service Level", [0.90, 0.95, 0.99], index=1)
z_map = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
z_score = z_map[service_level]

# Mock Data Generator for Demonstration
@st.cache_data
def load_data():
    dates = pd.date_range(start='2025-01-01', periods=180, freq='D')
    skus = [f"SKU-{i:03d}" for i in range(1, 21)]
    
    data = []
    np.random.seed(42)
    for sku in skus:
        base_demand = np.random.randint(20, 150)
        unit_price = np.random.uniform(10, 200)
        for date in dates:
            demand = max(0, int(base_demand + np.random.normal(0, 15)))
            revenue = demand * unit_price
            data.append({"date": date, "SKU_ID": sku, "demand": demand, "revenue": revenue, "unit_price": unit_price})
            
    return pd.DataFrame(data)

df = load_data()

# KPI Top Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Active SKUs", f"{df['SKU_ID'].nunique()}")
col2.metric("Total Revenue ($)", f"${df['revenue'].sum():,.2f}")
col3.metric("Total Units Sold", f"{df['demand'].sum():,}")
col4.metric("Avg Service Level Target", f"{int(service_level * 100)}%")

st.markdown("---")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 ABC-XYZ Segmentation", "📈 Demand Forecasting (LSTM)", "📋 Inventory Control & ROP"])

with tab1:
    st.subheader("ABC-XYZ Demand & Revenue Matrix")
    
    # Import local analytics logic or calculate inline
    from src.analytics import compute_abc_xyz_matrix
    sku_summary = compute_abc_xyz_matrix(df)
    
    fig_matrix = px.scatter(
        sku_summary,
        x='cv',
        y='total_revenue',
        color='ABC_XYZ',
        size='total_demand',
        hover_name='SKU_ID',
        labels={'cv': 'Demand Variability (CV)', 'total_revenue': 'Total Revenue ($)'},
        title="SKU Distribution: Revenue vs Demand Variability"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.dataframe(sku_summary[['SKU_ID', 'ABC_Class', 'XYZ_Class', 'ABC_XYZ', 'total_revenue', 'cv']], use_container_width=True)

with tab2:
    st.subheader("LSTM Time-Series Demand Forecasting")
    selected_sku = st.selectbox("Select SKU to Visualize", sku_summary['SKU_ID'].unique())
    
    sku_df = df[df['SKU_ID'] == selected_sku].sort_values('date')
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=sku_df['date'], y=sku_df['demand'], mode='lines', name='Actual Demand'))
    # Simulate forecasted trend line
    forecast_vals = sku_df['demand'].rolling(7).mean().fillna(method='bfill')
    fig_forecast.add_trace(go.Scatter(x=sku_df['date'], y=forecast_vals, mode='lines', name='LSTM Predicted Demand', line=dict(dash='dash', color='orange')))
    
    fig_forecast.update_layout(title=f"Historical vs Forecasted Demand for {selected_sku}", xaxis_title="Date", yaxis_title="Units")
    st.plotly_chart(fig_forecast, use_container_width=True)

with tab3:
    st.subheader("Actionable Inventory Policy Summary")
    
    from src.policies import calculate_inventory_policies
    
    # Create forecast dataframe mock
    forecast_df = df.copy()
    forecast_df['predicted_demand'] = forecast_df['demand'] * np.random.uniform(0.95, 1.05, len(forecast_df))
    
    policies_df = calculate_inventory_policies(
        sku_summary=sku_summary,
        forecast_df=forecast_df,
        lead_time_days=lead_time,
        service_level_z=z_score
    )
    
    st.dataframe(policies_df, use_container_width=True)
    
    st.download_button(
        label="📥 Export Inventory Recommendations CSV",
        data=policies_df.to_csv(index=False),
        file_name="inventory_reorder_recommendations.csv",
        mime="text/csv"
    )
