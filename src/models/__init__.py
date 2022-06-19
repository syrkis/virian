"""
The models module includes neural modesl for virian inference
"""


from ._fnn import FNN
from ._cnn import CNN
from ._lstm import LSTM
from ._tnn import TNN

__all__ = [
    'CNN',
    'FNN',
    'LSTM'
    'TNN',
]
