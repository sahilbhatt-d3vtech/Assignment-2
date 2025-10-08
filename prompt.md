Project Path: Assignment-2

Source Tree:

```txt
Assignment-2
├── Logs
│   ├── PyTorch_models.log
│   └── TensorFlow_models.log
├── PyTorch
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── main.py
│   ├── models.py
│   └── train.py
├── TensorFlow
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── main.py
│   ├── models.py
│   └── train.py
├── constants.py
├── pyproject.toml
└── utils.py

```

`Logs/PyTorch_models.log`:

```log
2025-10-07 21:19:08 - INFO - PyTorch - Using device: cpu

2025-10-07 21:19:09 - INFO - PyTorch - --- Training CNN Model ---
2025-10-07 21:19:24 - INFO - PyTorch - Saved trained model: E:\SAM\D3V\Assignment-2\Trained-Models\PyTorch\CNN_model.pth
2025-10-07 21:19:24 - INFO - PyTorch - --- CNN Model Evaluation ---
2025-10-07 21:19:24 - INFO - PyTorch - Test Loss: 0.014794
2025-10-07 21:19:24 - INFO - PyTorch - RMSE: 11.7333
2025-10-07 21:19:24 - INFO - PyTorch - MAE: 9.9902
2025-10-07 21:19:24 - INFO - PyTorch - Saved evaluation plot: E:\SAM\D3V\Assignment-2\Plots\PyTorch\PyTorch_CNN_prediction.png
2025-10-07 21:19:24 - INFO - PyTorch - --------------------------------------------------


2025-10-07 21:19:24 - INFO - PyTorch - --- Training RNN Model ---
2025-10-07 21:19:55 - INFO - PyTorch - Saved trained model: E:\SAM\D3V\Assignment-2\Trained-Models\PyTorch\RNN_model.pth
2025-10-07 21:19:55 - INFO - PyTorch - --- RNN Model Evaluation ---
2025-10-07 21:19:55 - INFO - PyTorch - Test Loss: 0.003719
2025-10-07 21:19:55 - INFO - PyTorch - RMSE: 5.8830
2025-10-07 21:19:55 - INFO - PyTorch - MAE: 4.2048
2025-10-07 21:19:56 - INFO - PyTorch - Saved evaluation plot: E:\SAM\D3V\Assignment-2\Plots\PyTorch\PyTorch_RNN_prediction.png
2025-10-07 21:19:56 - INFO - PyTorch - --------------------------------------------------


2025-10-07 21:19:56 - INFO - PyTorch - --- Training LSTM Model ---
2025-10-07 21:20:17 - INFO - PyTorch - Saved trained model: E:\SAM\D3V\Assignment-2\Trained-Models\PyTorch\LSTM_model.pth
2025-10-07 21:20:17 - INFO - PyTorch - --- LSTM Model Evaluation ---
2025-10-07 21:20:17 - INFO - PyTorch - Test Loss: 0.003145
2025-10-07 21:20:17 - INFO - PyTorch - RMSE: 5.4098
2025-10-07 21:20:17 - INFO - PyTorch - MAE: 3.9959
2025-10-07 21:20:17 - INFO - PyTorch - Saved evaluation plot: E:\SAM\D3V\Assignment-2\Plots\PyTorch\PyTorch_LSTM_prediction.png
2025-10-07 21:20:17 - INFO - PyTorch - --------------------------------------------------


2025-10-07 21:20:17 - INFO - PyTorch - --- Training GRU Model ---
2025-10-07 21:21:30 - INFO - PyTorch - Saved trained model: E:\SAM\D3V\Assignment-2\Trained-Models\PyTorch\GRU_model.pth
2025-10-07 21:21:30 - INFO - PyTorch - --- GRU Model Evaluation ---
2025-10-07 21:21:30 - INFO - PyTorch - Test Loss: 0.003864
2025-10-07 21:21:30 - INFO - PyTorch - RMSE: 5.9963
2025-10-07 21:21:30 - INFO - PyTorch - MAE: 4.4419
2025-10-07 21:21:30 - INFO - PyTorch - Saved evaluation plot: E:\SAM\D3V\Assignment-2\Plots\PyTorch\PyTorch_GRU_prediction.png
2025-10-07 21:21:30 - INFO - PyTorch - --------------------------------------------------



```

`Logs/TensorFlow_models.log`:

```log
2025-10-07 21:13:11 - INFO - TensorFlow - --- Training CNN Model ---
2025-10-07 21:13:27 - INFO - TensorFlow - Saved trained model: E:\SAM\D3V\Assignment-2\Trained-Models\TensorFlow\CNN_model.h5
2025-10-07 21:13:27 - INFO - TensorFlow - --- CNN Model Evaluation ---
2025-10-07 21:13:27 - INFO - TensorFlow - RMSE: 8.8362
2025-10-07 21:13:27 - INFO - TensorFlow - MAE: 6.4117
2025-10-07 21:13:28 - INFO - TensorFlow - Saved evaluation plot: E:\SAM\D3V\Assignment-2\Plots\TensorFlow\TensorFlow_CNN_prediction.png
2025-10-07 21:13:28 - INFO - TensorFlow - --------------------------------------------------


2025-10-07 21:13:28 - INFO - TensorFlow - --- Training RNN Model ---
2025-10-07 21:13:49 - INFO - TensorFlow - Saved trained model: E:\SAM\D3V\Assignment-2\Trained-Models\TensorFlow\RNN_model.h5
2025-10-07 21:13:49 - INFO - TensorFlow - --- RNN Model Evaluation ---
2025-10-07 21:13:49 - INFO - TensorFlow - RMSE: 3.4568
2025-10-07 21:13:49 - INFO - TensorFlow - MAE: 2.4086
2025-10-07 21:13:49 - INFO - TensorFlow - Saved evaluation plot: E:\SAM\D3V\Assignment-2\Plots\TensorFlow\TensorFlow_RNN_prediction.png
2025-10-07 21:13:49 - INFO - TensorFlow - --------------------------------------------------


2025-10-07 21:13:49 - INFO - TensorFlow - --- Training LSTM Model ---
2025-10-07 21:14:37 - INFO - TensorFlow - Saved trained model: E:\SAM\D3V\Assignment-2\Trained-Models\TensorFlow\LSTM_model.h5
2025-10-07 21:14:37 - INFO - TensorFlow - --- LSTM Model Evaluation ---
2025-10-07 21:14:37 - INFO - TensorFlow - RMSE: 8.5981
2025-10-07 21:14:37 - INFO - TensorFlow - MAE: 6.0574
2025-10-07 21:14:37 - INFO - TensorFlow - Saved evaluation plot: E:\SAM\D3V\Assignment-2\Plots\TensorFlow\TensorFlow_LSTM_prediction.png
2025-10-07 21:14:37 - INFO - TensorFlow - --------------------------------------------------


2025-10-07 21:14:37 - INFO - TensorFlow - --- Training GRU Model ---
2025-10-07 21:15:30 - INFO - TensorFlow - Saved trained model: E:\SAM\D3V\Assignment-2\Trained-Models\TensorFlow\GRU_model.h5
2025-10-07 21:15:30 - INFO - TensorFlow - --- GRU Model Evaluation ---
2025-10-07 21:15:30 - INFO - TensorFlow - RMSE: 4.6312
2025-10-07 21:15:30 - INFO - TensorFlow - MAE: 3.3448
2025-10-07 21:15:30 - INFO - TensorFlow - Saved evaluation plot: E:\SAM\D3V\Assignment-2\Plots\TensorFlow\TensorFlow_GRU_prediction.png
2025-10-07 21:15:30 - INFO - TensorFlow - --------------------------------------------------



```

`PyTorch/data_loader.py`:

```py
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
    
    return train_loader, test_loader, scaler
```

`PyTorch/evaluate.py`:

```py
import torch
import numpy as np


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, all_preds, all_targets
```

`PyTorch/main.py`:

```py
import os
import torch
import numpy as np
from data_loader import get_data_loaders
from models import get_model
from train import train_model
from evaluate import evaluate_model
from utils import plot_predictions, setup_logger
import constants as const

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Logger setup
    logger = setup_logger("PyTorch")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), const.TRAINED_MODELS_FOLDER, 'PyTorch')
    os.makedirs(models_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, test_loader, scaler = get_data_loaders(
        const.TICKER, const.SEQ_LENGTH, batch_size=const.BATCH_SIZE
    )
    
    # Model configurations
    input_dim = 1  # Number of features (closing price only)
    models_to_train = ['CNN', 'RNN', 'LSTM', 'GRU']
    
    for model_name in models_to_train:
        logger.info(f"--- Training {model_name} Model ---")
        
        # Initialize model
        model = get_model(model_name, input_dim, device)
        
        # Loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
        
        # Train the model
        model = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=const.NUM_EPOCHS,
            device=device
        )
        
        # Save the trained model
        model_path = os.path.join(models_dir, f'{model_name}_model.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved trained model: {model_path}")
        
        # Evaluate the model
        test_loss, y_pred, y_true = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # Inverse transform the scaled data
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_pred_inv - y_true_inv) ** 2))
        mae = np.mean(np.abs(y_pred_inv - y_true_inv))
        
        logger.info(f"--- {model_name} Model Evaluation ---")
        logger.info(f"Test Loss: {test_loss:.6f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")

        # Plot predictions
        plot_predictions(model_name, y_true_inv, y_pred_inv, "PyTorch", logger)
        logger.info("-"*50 + "\n\n")

if __name__ == "__main__":
    main()
```

`PyTorch/models.py`:

```py
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, output_dim=1, dropout=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(hidden_dim)  # Let PyTorch infer the input size
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        x = x.transpose(1, 2)  # (batch_size, 1, seq_len) for Conv1d
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=1, output_dim=1, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Only apply dropout if we have multiple layers
        dropout = dropout if num_layers > 1 else 0
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Get the last time step's output
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=1, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Only apply dropout if we have multiple layers
        dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Get the last time step's output
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=1, output_dim=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Only apply dropout if we have multiple layers
        dropout = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Get the last time step's output
        return out

def get_model(model_name, input_dim, device):
    """
    Factory function to create the requested model
    """
    model_map = {
        'CNN': CNNModel(input_dim=input_dim),
        'RNN': RNNModel(input_dim=input_dim),
        'LSTM': LSTMModel(input_dim=input_dim),
        'GRU': GRUModel(input_dim=input_dim)
    }
    
    return model_map[model_name].to(device)
```

`PyTorch/train.py`:

```py
import torch
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    Trains a PyTorch model with progress bar and loss display.
    """
    model.train()
    
    # Move model to device
    model = model.to(device)
    
    # Initialize tqdm progress bar
    pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in pbar:
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            epoch_loss += loss.item() * batch_x.size(0)
        
        # Calculate average loss for the epoch
        epoch_loss = epoch_loss / len(train_loader.dataset)
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{epoch_loss:.6f}'})
    
    return model
```

`TensorFlow/data_loader.py`:

```py
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
```

`TensorFlow/evaluate.py`:

```py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluates a trained TensorFlow model.
    """
    predictions = model.predict(X_test)

    # Inverse transform to get actual values
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    mae = mean_absolute_error(y_test_scaled, predictions)

    return rmse, mae, predictions, y_test_scaled
```

`TensorFlow/main.py`:

```py
from data_loader import get_data
from models import create_cnn_model, create_rnn_model, create_lstm_model, create_gru_model
from train import train_model
from evaluate import evaluate_model
from utils import plot_predictions, setup_logger
import constants as const
import os

def main():
    ticker = const.TICKER
    seq_length = const.SEQ_LENGTH

    # Logger setup
    logger = setup_logger("TensorFlow")

    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), const.TRAINED_MODELS_FOLDER, 'TensorFlow')
    os.makedirs(models_dir, exist_ok=True)

    X_train, y_train, X_test, y_test, scaler = get_data(ticker, seq_length)

    input_shape = (X_train.shape[1], 1)

    models = {
        "CNN": create_cnn_model(input_shape),
        "RNN": create_rnn_model(input_shape),
        "LSTM": create_lstm_model(input_shape),
        "GRU": create_gru_model(input_shape)
    }

    for name, model in models.items():
        logger.info(f"--- Training {name} Model ---")
        trained_model = train_model(model, const.OPTIMIZER, const.LOSS, const.NUM_EPOCHS, const.BATCH_SIZE, X_train, y_train)

        # Save the trained model
        model_path = os.path.join(models_dir, f'{name}_model.h5')
        trained_model.save(model_path)
        logger.info(f"Saved trained model: {model_path}")

        rmse, mae, predictions, y_test_scaled = evaluate_model(trained_model, X_test, y_test, scaler)
        logger.info(f"--- {name} Model Evaluation ---")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")

        # Plot actual vs predicted values
        plot_predictions(name, y_test_scaled, predictions, "TensorFlow", logger)
        logger.info("-"*50 + "\n\n")

if __name__ == "__main__":
    main()
```

`TensorFlow/models.py`:

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN, LSTM, GRU

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    return model

def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    return model

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    return model

def create_gru_model(input_shape):
    model = Sequential([
        GRU(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    return model
```

`TensorFlow/train.py`:

```py
def train_model(model, optimizer, loss, num_epochs, batch_size, X_train, y_train):
    """
    Trains a TensorFlow model.
    """
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
    return model
```

`constants.py`:

```py
# Plotting folder
PLOTS_FOLDER = "Plots"

# Trained model save folder
TRAINED_MODELS_FOLDER = "Trained-Models"

# Stock to follow
TICKER = "GOOGL"

# Stock price date range
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2023-12-31"

# Sequence length
SEQ_LENGTH = 60 # Taking previous 60 days stock data to predict next day stock

# TensorFlow model configuration
OPTIMIZER = "adam"
LOSS = "mean_squared_error"
NUM_EPOCHS = 100
BATCH_SIZE = 32

# PyTorch model configuration
LEARNING_RATE = 0.001
```

`pyproject.toml`:

```toml
[project]
name = "assignment-2"
version = "0.1.0"
description = "Assignment 2"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "jupyter>=1.1.1",
    "matplotlib>=3.10.6",
    "numpy>=2.3.3",
    "pandas>=2.3.3",
    "scikit-learn>=1.7.2",
    "seaborn>=0.13.2",
    "tensorflow>=2.20.0",
    "torch>=2.8.0",
    "tqdm>=4.67.1",
    "yfinance>=0.2.66",
]

```

`utils.py`:

```py
import logging
import matplotlib.pyplot as plt
import os
import constants as const

def setup_logger(framework_name):
    """
    Configures and returns a logger for a given framework (TensorFlow/PyTorch).
    """
    logs_dir = os.path.join(os.path.dirname(__file__), "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, f"{framework_name}_models.log")

    logger = logging.getLogger(framework_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def plot_predictions(model_name, y_true, y_pred, framework_name="TensorFlow", logger=None):
    """
    Plot actual vs predicted values for model evaluation.
    """
    plt.figure(figsize=(12, 6))

    # Create a range for x-axis (representing time sequence)
    x_axis = range(len(y_true))

    plt.plot(x_axis, y_true, label='Actual Values', color='blue', linewidth=2)
    plt.plot(x_axis, y_pred, label='Predicted Values', color='red', linewidth=2, linestyle='--')

    plt.title(f'{model_name} Model - Actual vs Predicted ({framework_name})')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Create Plots directory if it doesn't exist (in the same directory as utils.py)
    plots_dir = os.path.join(os.path.dirname(__file__), const.PLOTS_FOLDER, framework_name)
    os.makedirs(plots_dir, exist_ok=True)

    filename = f"{framework_name}_{model_name}_prediction.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath)
    plt.close()

    if logger:
        logger.info(f"Saved evaluation plot: {filepath}")
    else:
        print(f"Saved evaluation plot: {filepath}")
```