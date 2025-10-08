import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import constants as const

def load_and_split_data(ticker, train_end_date):
    """
    Loads historical stock data, splits it into training and testing sets by date,
    and scales it.
    """
    # Download data from 2020 to the latest available date
    data = yf.download(ticker, start=const.TRAIN_START_DATE)
    
    # Ensure the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)
    
    # Get the closing prices
    close_prices = data[['Close']]
    
    # Split the data into training and testing sets
    train_data = close_prices.loc[:train_end_date]
    test_data = close_prices.loc[train_end_date:]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit the scaler ONLY on the training data
    scaled_train_data = scaler.fit_transform(train_data)
    # Transform the test data using the same scaler
    scaled_test_data = scaler.transform(test_data)

    return scaled_train_data, scaled_test_data, scaler

def create_sequences(data, seq_length):
    """
    Creates sequences of data for time series forecasting.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_data(ticker, seq_length):
    """
    Prepares and returns data for training and testing based on a date split.
    """
    train_end_date = const.TRAIN_END_DATE # End of 2023 for training

    # Load, split, and scale data
    scaled_train_data, scaled_test_data, scaler = load_and_split_data(ticker, train_end_date)

    # Create sequences for training
    X_train, y_train = create_sequences(scaled_train_data, seq_length)
    
    # To create test sequences, we need to include the last `seq_length` days
    # of the training data to have enough data for the first test sequence.
    inputs = np.concatenate((scaled_train_data[-seq_length:], scaled_test_data))
    X_test, y_test = create_sequences(inputs, seq_length)

    return X_train, y_train, X_test, y_test, scaler