from prophet import Prophet
import pandas as pd
import numpy as np

def build_prophet_model(df, train_ratio=0.7, verbose=False):
    """
    df: DataFrame with 'ds' (date) and 'y' (value) columns
    train_ratio: ratio of data to use for training
    return: predictions, train_df, test_df
    """
    df = df.sort_values('ds').dropna().reset_index(drop=True)
    
    # Divide training set and test set
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    predictions = []
    history_df = train_df.copy()

    for i in range(len(test_df)):
        try:
            model = Prophet()
            model.fit(history_df)

            # Construct the next time point to be predicted
            future = pd.DataFrame({'ds': [test_df.iloc[i]['ds']]})
            forecast = model.predict(future)
            yhat = forecast['yhat'].values[0]
        except Exception as e:
            print(f"Prophet model failed at step {i}: {e}")
            yhat = history_df['y'].iloc[-1]  # fallback: last known value

        predictions.append(yhat)

        # Add the current real value to history
        new_row = test_df.iloc[[i]]
        history_df = pd.concat([history_df, new_row], ignore_index=True)

        if verbose:
            print(f"Step {i}: predicted={yhat:.2f}, actual={test_df.iloc[i]['y']:.2f}")

    return predictions, train_df, test_df
