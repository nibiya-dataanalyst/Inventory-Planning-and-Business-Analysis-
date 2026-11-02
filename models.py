"""
Deep Learning LSTM Architecture & Training Pipeline
Author: Nibiya Nadir
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(input_shape):
    """
    Builds and compiles a 2-layer LSTM neural network for demand time-series forecasting.
    """
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_forecasting_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    """
    Trains the LSTM model with EarlyStopping to prevent overfitting.
    """
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history
