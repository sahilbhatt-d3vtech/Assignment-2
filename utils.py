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