from .transformer_model import TransformerModel
from .base_model import LSTMModel, ARIMAModel

def model_factory(name: str, **params):
    name = name.lower()
    if name == 'transformer':
        return TransformerModel(**params)
    elif name == 'lstm':
        return LSTMModel(**params)
    elif name == 'arima':
        return ARIMAModel(**params)
    else:
        raise ValueError(f"Unknown model: {name}")