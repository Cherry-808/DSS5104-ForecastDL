import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def build_arima_model():
    # 1. Extract a univariate series from combined_data
    tsla_col = [col for col in combined_data.columns if col.lower().endswith('close_scaled')][0]
    tsla_series = combined_data[tsla_col]

    # 2. Check stationarity (visual inspection or use ADF test)
    tsla_diff = tsla_series.diff().dropna()

    # Optional: ADF test
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(tsla_diff)
    print(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")

    # 3. Fit ARIMA (p,d,q)
    model = ARIMA(tsla_series, order=(5, 1, 0))  # Example (5,1,0); adjust with AIC/BIC

    return model
