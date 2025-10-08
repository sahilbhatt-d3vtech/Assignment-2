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