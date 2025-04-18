import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    BaseModel for all forecasting models. Provides common interfaces.
    """
    def __init__(self):
        super().__init__()
        # Initialize device attribute
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, *args, **kwargs):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward().")

    def to(self, *args, **kwargs):
        """Move model parameters and update device attribute."""
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return self

class LSTMModel(BaseModel):
    """
    LSTM-based time series forecasting model.
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        history: int = 365,
        horizon: int = 28
    ):
        super().__init__()
        self.history = history
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, history)
        x = x.unsqueeze(-1)  # (batch_size, history, 1)
        out, _ = self.lstm(x)  # out: (batch_size, history, hidden_size)
        last = out[:, -1, :]   # (batch_size, hidden_size)
        return self.fc(last)   # (batch_size, horizon)


class ARIMAModel(BaseModel):
    """
    ARIMA-based forecasting as a baseline. Fits per series.
    Note: requires statsmodels.
    """
    def __init__(
        self,
        order: tuple = (5, 1, 0),
        history: int = 365,
        horizon: int = 28
    ):
        super().__init__()
        self.order = order
        self.history = history
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, history)
        import numpy as _np
        from statsmodels.tsa.arima.model import ARIMA

        preds = []
        x_np = x.detach().cpu().numpy()
        for seq in x_np:
            model = ARIMA(seq, order=self.order).fit()
            p = model.predict(start=len(seq), end=len(seq) + self.horizon - 1)
            preds.append(p)
        preds_np = _np.stack(preds, axis=0)
        return torch.from_numpy(preds_np).float().to(self.device)
