def train_test_split(series, split_ratio=0.8):
    split_point = int(len(series) * split_ratio)
    return series.iloc[:split_point], series.iloc[split_point:]
