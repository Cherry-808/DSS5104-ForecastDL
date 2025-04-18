import math
import torch
import torch.nn as nn
from .base_model import BaseModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding module adds sine/cosine positional embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            output: Tensor same shape as x
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """
    Transformer-based time series forecasting model.

    Input: sequence of shape (batch_size, history)
    Output: predictions of shape (batch_size, horizon)
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        history: int = 365,
        horizon: int = 28
    ):
        super().__init__()
        self.history = history
        self.horizon = horizon

        # Project scalar input -> d_model
        self.input_proj = nn.Linear(1, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=history)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final output projection
        self.fc_out = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch_size, history)
        Returns:
            out: Tensor (batch_size, horizon)
        """
        # x -> (history, batch_size, 1)
        x = x.unsqueeze(-1).permute(1, 0, 2)
        # project to d_model
        x = self.input_proj(x)  # (history, batch_size, d_model)
        # add positional encoding
        x = self.pos_encoder(x)
        # transformer encoding
        x = self.transformer_encoder(x)  # (history, batch_size, d_model)
        # use last time-step
        last = x[-1]  # (batch_size, d_model)
        out = self.fc_out(last)  # (batch_size, horizon)
        return out
