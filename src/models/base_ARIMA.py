import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def build_arima_model(series, order, train_ratio, verbose=False):
    # Ensure it's a 1D NumPy array
    series = np.array(series.dropna()).flatten().astype('float64')
    
    # split into train and test sets
    size = int(len(series) * train_ratio)
    train, test = series[:size], series[size:]
    history = list(train)
    predictions = []
    
    # Walk-forward validation
    for t in range(len(test)):
        try:
            # Convert to pure 1D array for ARIMA
            model = ARIMA(np.array(history, dtype='float64'), order=order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
        except Exception as e:
            print(f" Model failed at step {t}: {e}")
            yhat = history[-1]  # fallback: repeat last known value

        predictions.append(yhat)
        history.append(test[t])

        if verbose:
            print(f"predicted={yhat:.3f}, expected={test[t]:.3f}")

    return predictions, train, test
