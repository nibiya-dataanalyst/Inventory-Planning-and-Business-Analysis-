Inventory Planning & Business Analysis: Time-Series Demand Forecasting (LSTM & Deep Learning)
An end-to-end time-series forecasting and inventory data analysis pipeline leveraging Python, TensorFlow/Keras, Scikit-Learn, and Seaborn. This project explores large-scale historical inventory datasets spanning multi-store and multi-category retail domains to predict demand, track sales performance, and optimize inventory replenishment logic.

🚀 Project Overview
Effective inventory management is critical for minimizing stockouts and holding costs in retail supply chains. This repository contains an analytical workspace and deep learning pipeline built to:

Perform Exploratory Data Analysis (EDA): Analyze 109,500+ records of granular store-product-level inventory tracking data across multiple regions and product categories.

Feature Engineering & Preprocessing: Process temporal features, validate date ranges, check for missing values, and structure sequential data for machine learning models.

Demand Forecasting with Deep Learning: Implement an LSTM (Long Short-Term Memory) neural network architecture using TensorFlow/Keras to forecast unit sales and inventory dynamics.

Performance Evaluation: Measure predictive error using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

Regional & Categorical Insights: Aggregate sales data across product lines (Clothing, Electronics, Furniture, Groceries, Toys) and geographical regions (East, West, North, South) to identify key revenue drivers.

📊 Key Datasets & Exploratory Insights
Time Span: 729 days of continuous daily data (January 1, 2022 – December 31, 2023).

Granularity: Tracks Store IDs (S001–S005), Product IDs (P0016–P0187), regions, weather conditions, pricing, competitor pricing, seasonality, and promotional campaigns.

Top-Performing Categories: Led by Clothing (~4.2M units sold) and Electronics (~2.5M units sold).

Regional Distribution: Sales volume remains exceptionally well-balanced across Eastern, Western, Northern, and Southern regional distribution hubs (~2.6M–2.66M units per region).

📈 Model Performance & Evaluation
Baseline demand forecasts compared against actual sales figures yield robust accuracy metrics on the evaluation dataset:

Mean Absolute Error (MAE): 13.74 units

Root Mean Squared Error (RMSE): 17.79 units

🛠️ Tech Stack & Libraries
Language: Python 3

Deep Learning: TensorFlow, Keras (LSTM, Dense, Dropout layers)

Data Manipulation: Pandas, NumPy

Machine Learning Metrics: Scikit-Learn (mean_absolute_error, mean_squared_error, MinMaxScaler)

Data Visualization: Matplotlib, Seaborn

📂 Code Structure & Execution Flow
Environment Setup & Imports:

Python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
Data Ingestion & Cleaning: Mounting Google Drive and reading structured CSV inventory data, performing date-time conversions, and checking null value integrity.

Exploratory Data Analysis (EDA):

Statistical description of numerical features (Units Sold, Inventory Level, Price, Competitor Pricing, Demand Forecast).

Outlier detection via boxplots.

Time-series trend plotting for daily aggregate unit sales.

Demand vs. Actual Comparison: Evaluating forecast deviation and computing MAE/RMSE benchmarks.

Categorical & Regional Breakdown: Grouping and visualizing volume distribution across categories and geographic regions using bar plots.





