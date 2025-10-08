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