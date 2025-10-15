from data_loader import get_data
from models import create_cnn_model, create_rnn_model, create_lstm_model, create_gru_model
from train import train_model
from evaluate import evaluate_model
from utils import plot_train_test_predictions, setup_logger
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

        # Inverse transform y_train for plotting
        y_train_inv = scaler.inverse_transform(y_train)

        # Plot train actual, test actual, and test predicted
        plot_train_test_predictions(name, y_train_inv, y_test_scaled, predictions, "TensorFlow", logger)
        logger.info("-"*50 + "\n\n")

if __name__ == "__main__":
    main()