"""
Data Preprocessing & Windowing Module for Inventory Forecasting
Author: Nibiya Nadir
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_inventory_data(df):
    """
    Cleans raw retail transaction logs and handles missing values safely.
    Tracks issues and transformations for auditability.
    """
    df = df.copy()
    
    # Ensure proper datetime parsing
    df['date'] = pd.to_datetime(df['date'])
    
    # Drop exact duplicate records
    df = df.drop_duplicates()
    
    # Fill missing demand with zero (representing no sales, not missing stock)
    df['demand'] = df['demand'].fillna(0)
    
    # Ensure positive revenue and demand values
    df['demand'] = df['demand'].apply(lambda x: max(0, x))
    if 'revenue' in df.columns:
        df['revenue'] = df['revenue'].apply(lambda x: max(0, x))
        
    return df.sort_values(by=['SKU_ID', 'date']).reset_index(drop=True)


def prepare_lstm_data(df, target_col='demand', train_ratio=0.8, lookback=30):
    """
    Performs a strict chronological split and isolates scaler fitting 
    to the training set to prevent data leakage in time-series validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing chronological time-series data for a single SKU or aggregated sequence.
    target_col : str
        Column name of the target variable to forecast (e.g., 'demand').
    train_ratio : float
        Proportion of data used for training (default 0.8).
    lookback : int
        Number of historical time steps used to predict the next step (default 30).
        
    Returns:
    --------
    X_train, y_train, X_test, y_test, scaler
    """
    # 1. Strict Chronological Split
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # 2. Fit Scaler ONLY on Training Data (Prevents Data Leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[[target_col]])
    
    # Transform test set using training parameters
    test_scaled = scaler.transform(test_df[[target_col]])
    
    # Helper function to construct sliding sequences
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_scaled, lookback)
    X_test, y_test = create_sequences(test_scaled, lookback)
    
    # Reshape for LSTM architecture: [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler
