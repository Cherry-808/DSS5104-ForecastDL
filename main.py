from src.data import generate_sample_data, train_test_split
from src.visualization import plot_time_series, evaluate_forecast
from src.models import arima_forecast

def main():
    # Generate sample data
    df = generate_sample_data()

    # Plot the original time series data
    plot_time_series(df, title='Original Time Series Data')

    # Split the data into training and testing sets
    train, test = train_test_split(df['value'])

    # Fit ARIMA model and make predictions
    predictions = arima_forecast(train, test)

    # Evaluate the forecast
    evaluate_forecast(test, predictions)

if __name__ == '__main__':
    main()
