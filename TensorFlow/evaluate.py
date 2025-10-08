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