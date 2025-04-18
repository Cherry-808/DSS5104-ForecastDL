import os
import numpy as np
from torch.utils.data import Dataset


class M5Dataset(Dataset):
    """
    PyTorch Dataset for M5 time-series data using memory-mapped numpy array.

    The processed directory should contain a file 'ts_values.npy' of shape (n_series, n_days).
    """
    def __init__(self, ts_path: str, history: int, horizon: int):
        """
        Args:
            ts_path: Path to the numpy file with raw time series (n_series, n_days)
            history: Number of past days to use as input
            horizon: Number of future days to predict
        """
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f"Time series file not found: {ts_path}")

        # Load via mmap to avoid full memory load
        self.ts = np.load(ts_path, mmap_mode='r')
        self.history = history
        self.horizon = horizon

        # Calculate dimensions
        self.n_series, self.n_days = self.ts.shape
        self.max_start = self.n_days - self.horizon
        self.samples_per_series = max(0, self.max_start - self.history + 1)
        self.total_samples = self.n_series * self.samples_per_series

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which series and window index
        series_idx = idx // self.samples_per_series
        window_idx = idx % self.samples_per_series

        start = window_idx
        end = window_idx + self.history
        x = self.ts[series_idx, start:end]
        y = self.ts[series_idx, end:end + self.horizon]

        return x.astype(np.float32), y.astype(np.float32)


def get_dataset(name: str, **kwargs) -> Dataset:
    """
    Factory to get dataset by name.

    Supported names:
      - 'm5': returns M5Dataset
    """
    name = name.lower()
    if name == 'm5':
        return M5Dataset(
            ts_path=kwargs.get('ts_path'),
            history=kwargs.get('history'),
            horizon=kwargs.get('horizon')
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
