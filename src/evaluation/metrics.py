from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def generate_model_metrics(predictions, test):
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(test, predictions))

    # Mean Absolute Error
    mae = mean_absolute_error(test, predictions)

    # R-squared
    r2 = r2_score(test, predictions)

    # Adjusted R-squared
    n = len(test)
    p = 1  # number of predictors (just 1 for ARIMA by default)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return rmse, mae, r2, adj_r2
