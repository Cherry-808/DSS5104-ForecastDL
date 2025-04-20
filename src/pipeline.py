# pipeline.py

import pandas as pd
from src.data.load_fin_data import load_finance_data
from src.data.preprocess_fin_data import enrich_target_stock
from src.data.remove_outliers import remove_outliers_zscore
from src.data.scaler_utils import get_scaler
from src.data.split_sequences import prepare_train_val_test
from DL_LSTM import build_lstm_model, plot_predictions
from metrics import generate_model_metrics


class LSTMFinancePipeline:
    def __init__(self, ticker, start_date, end_date, seq_length, test_ratio, outlier_threshold, plot=False):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.test_ratio = test_ratio
        self.outlier_threshold = outlier_threshold
        self.plot = plot

        self.scaler = get_scaler()
        self.model = None

    def load_and_preprocess_data(self):
        print("\n[INFO] Loading data...")
        df = load_finance_data(self.ticker, self.start_date, self.end_date)

        print("[INFO] Enriching with technical features...")
        df = enrich_target_stock(df, self.ticker)

        print("[INFO] Removing outliers...")
        df_clean = remove_outliers_zscore(df.dropna(), threshold=self.outlier_threshold)

        print("[INFO] Scaling data...")
        scaled = self.scaler.fit_transform(df_clean)
        scaled_df = pd.DataFrame(scaled, columns=df_clean.columns, index=df_clean.index)
        scaled_df['Date'] = scaled_df.index
        scaled_df.reset_index(drop=True, inplace=True)

        return df_clean, scaled_df

    def prepare_data(self, scaled_df):
        print("[INFO] Preparing sequences...")
        return prepare_train_val_test(scaled_df, self.seq_length)

    def train_model(self, X_train, y_train, X_val, y_val):
        print("[INFO] Building LSTM model...")
        self.model = build_lstm_model(self.seq_length, 1)

        print("[INFO] Training...")
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)

    def evaluate(self, X_test, y_test, df_clean):
        print("[INFO] Predicting...")
        y_pred = self.model.predict(X_test)

        print("[INFO] Inverting scaling...")
        y_test_inv = self.scaler.inverse_transform([[0]*(X_test.shape[2]-1) + [val[0]] for val in y_test])[:, -1]
        y_pred_inv = self.scaler.inverse_transform([[0]*(X_test.shape[2]-1) + [val[0]] for val in y_pred])[:, -1]

        print("[INFO] Evaluating...")
        rmse, mse, mae, mape, r2, adj_r2 = generate_model_metrics(y_pred_inv, y_test_inv)

        print(f"\nEvaluation Metrics:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\nAdj R2: {adj_r2:.4f}\n")

        if self.plot:
            print("[INFO] Plotting predictions...")
            plot_predictions(df_clean.index[-len(y_test):], y_test_inv, y_pred_inv)

        return rmse, mae, r2, adj_r2

    def run(self):
        df_clean, scaled_df = self.load_and_preprocess_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(scaled_df)
        self.train_model(X_train, y_train, X_val, y_val)
        return self.evaluate(X_test, y_test, df_clean)
