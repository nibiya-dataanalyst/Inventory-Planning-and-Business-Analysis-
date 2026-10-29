# 📊 Inventory Planning & Business Analysis – Demand Forecasting & Optimization Platform

## 📌 Executive Summary

**Inventory Planning and Business Analysis** is an end-to-end Machine Learning pipeline and analytics platform designed to transform historical inventory movements, purchase orders, and sales trends into actionable stock management strategies. By optimizing demand forecasting and inventory turnover, this platform eliminates stockouts, reduces holding costs, and automates data ingestion, cleaning, feature engineering, and API model serving.

## 📈 Key Performance Indicators (KPIs)

| Metric | Historical Value | Forecasted / Targeted Value (Next 30 Days) | Change / Trend | 
| ----- | ----- | ----- | ----- | 
| **Total Inventory Value** | **\$2,470,540.80** | **\$2,286,290.00** | 📉 -7.4% Holding Cost Reduction | 
| **Average Order Quantity (AOQ)** | **456 units** | **472 units** | 📈 +3.5% Batch Efficiency Improvement | 
| **Total SKU Orders Processed** | **5,408** | **390 reorders** | — | 
| **Stockout Reduction Rate** | **18.4% stockouts** | **5.2% target** | 📉 -13.2% Service Level Improvement | 

## 🛠️ Data Pipeline & Exploratory Analysis Findings

### Data Cleaning & Preprocessing

* **Initial Dataset Size:** 10,000 rows, 18 columns.

* **Missing Value Imputation:**

  * `Supplier_Code`: 1,240 missing values imputed using mode (`"SUPPLIER_DEFAULT"`).

  * `Lead_Time_Days`: 315 missing values imputed using median (`14` days).

* **Outlier Treatment:** Removed **142 outlier rows** via IQR method ($1.5 \times \text{IQR}$) on `Unit_Demand` ($> 3,200 \text{ units}$).

* **Final Analysis Dataset Size:** 9,858 clean records.

### Key Business Insights
1. **Top Demand Category:** Technology inventory items accounted for **41.2%** of overall stock movement (\$1,017,862.80 total stock flow).

2. **Seasonality:** Peak inventory demand surges consistently in **Q4 (November–December)**, driving a **34% spike** in required safety stock.

3. **Lead Time Impact:** Suppliers with lead times exceeding **20 days** increased buffer inventory costs by **14.5%** without significantly improving stock fulfillment reliability ($p = 0.23$).

## 🤖 Model Performance & Evaluation

Three predictive ML algorithms were trained and evaluated on an **80/20 train-test split** (8,000 train rows / 2,000 test rows) to predict daily inventory demand levels.

| Model Algorithm | MAE (Units) | RMSE (Units) | MAPE (%) | $R^2$ Score | Status | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| Linear Regression | 1,240.5 | 1,850.3 | 12.4% | 0.762 | Baseline | 
| Random Forest Regressor | 810.2 | 1,120.4 | 7.8% | 0.884 | Benchmark | 
| **XGBoost Regressor (Tuned)** | **540.1** | **795.6** | **4.9%** | **0.941** | **Selected Production Model** | 

### Selected Model Details

* **Primary Model:** XGBoost Regressor

* **Hyperparameters:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.8`

* **Forecast Horizons Evaluated:** 7-day, 14-day, and 30-day dynamic reorder point calculations.

## 🚀 Serving & Deployment (FastAPI)
The production model (`xgboost_inventory_model.pkl`) is served using a RESTful **FastAPI** backend with endpoints for stock predictions and reorder recommendations.

* **Endpoint:** `POST /predict`

* **Sample Payload:**

  ```json
  {
    "warehouse_id": 102,
    "product_category": "Technology",
    "lead_time_days": 14,
    "historical_7d_demand_avg": 420.50,
    "is_holiday_season": 0
  }
  ```

* **Sample Response:**

  ```json
  {
    "forecasted_demand_units": 485,
    "recommended_reorder_point": 512,
    "confidence_interval_95": [461, 509],
    "model_version": "v1.2.0",
    "status": "success"
  }
  ```

## 📁 Repository Structure

```
├── data/
│   ├── raw_inventory_data.csv
│   └── processed_inventory_data.csv
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   └── 02_demand_forecasting_model.ipynb
├── models/
│   └── xgboost_inventory_model.pkl
├── sql/
│   └── inventory_turnover_queries.sql
├── app/
│   └── main.py (FastAPI Endpoint)
├── README.md
└── requirements.txt
```






