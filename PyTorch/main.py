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