# main.py

import argparse
from pipeline import LSTMFinancePipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="TSLA")
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date", type=str, default="2023-12-31")
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--outlier_threshold", type=float, default=3.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    pipeline = LSTMFinancePipeline(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_length=args.seq_length,
        test_ratio=args.test_ratio,
        outlier_threshold=args.outlier_threshold,
        plot=args.plot
    )

    pipeline.run()
