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