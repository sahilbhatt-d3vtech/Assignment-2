import logging
import matplotlib.pyplot as plt
import os
import numpy as np
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


def plot_train_test_predictions(model_name, y_train_true, y_test_true, y_test_pred, framework_name="TensorFlow", logger=None):
    """
    Plot training data (e.g., 2020-2023), testing data (2024-current), and predictions on the testing range on one graph.
    """
    plt.figure(figsize=(12, 6))

    # Flatten
    y_train_true = np.asarray(y_train_true).reshape(-1)
    y_test_true = np.asarray(y_test_true).reshape(-1)
    y_test_pred = np.asarray(y_test_pred).reshape(-1)

    # Indices
    n_train = len(y_train_true)
    n_test = len(y_test_true)
    x_train = range(n_train)
    x_test = range(n_train, n_train + n_test)

    # Plots
    plt.plot(x_train, y_train_true, label='Train (Actual)', color='blue', linewidth=2)
    plt.plot(x_test, y_test_true, label='Test (Actual)', color='green', linewidth=2)
    plt.plot(x_test, y_test_pred, label='Test (Predicted)', color='red', linewidth=2, linestyle='--')

    plt.title(f'{model_name} - Train/Test Actual vs Test Predicted ({framework_name})')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plots_dir = os.path.join(os.path.dirname(__file__), const.PLOTS_FOLDER, framework_name)
    os.makedirs(plots_dir, exist_ok=True)

    filename = f"{framework_name}_{model_name}_train_test_prediction.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath)
    plt.close()

    if logger:
        logger.info(f"Saved evaluation plot: {filepath}")
    else:
        print(f"Saved evaluation plot: {filepath}")