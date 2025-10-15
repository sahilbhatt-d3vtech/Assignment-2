import yfinance as yf
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import constants as const
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

def get_data_loaders(ticker, seq_length, batch_size=32):
    """
    Prepares and returns data loaders for training and testing.
    """
    train_end_date = const.TRAIN_END_DATE  # End of 2023 for training

    # Load, split, and scale data
    scaled_train_data, scaled_test_data, scaler = load_and_split_data(ticker, train_end_date)

    # Create sequences for training
    X_train, y_train = create_sequences(scaled_train_data, seq_length)
    
    # To create test sequences, we need to include the last `seq_length` days
    # of the training data to have enough data for the first test sequence.
    inputs = np.concatenate((scaled_train_data[-seq_length:], scaled_test_data))
    X_test, y_test = create_sequences(inputs, seq_length)
    
    # Reshape data for PyTorch (batch_size, seq_len, n_features)
    X_train = X_train.reshape(-1, seq_length, 1)
    X_test = X_test.reshape(-1, seq_length, 1)
    
    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Return loaders, scaler, and sequential targets for plotting
    return train_loader, test_loader, scaler, y_train, y_test