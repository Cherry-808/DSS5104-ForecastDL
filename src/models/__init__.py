from .base_model import BaseModel
from .transformer_model import TransformerModel
from .base_model import LSTMModel, ARIMAModel
from .model_factory import model_factory

def model_factory(name: str, **params):
    name = name.lower()
    if name == 'transformer':
        return TransformerModel(**params)
    elif name == 'lstm':
        return LSTMModel(**params)
    # 其它模型……
    else:
        raise ValueError(f"Unknown model: {name}")