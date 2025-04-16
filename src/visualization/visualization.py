import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_time_series(data, title='Time Series Data'):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Original Data')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def evaluate_forecast(test, predictions):
    error = mean_squared_error(test, predictions)
    print(f'Test Mean Squared Error: {error:.3f}')
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='Actual Data')
    plt.plot(test.index, predictions, color='red', label='Predicted Data')
    plt.title('Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
