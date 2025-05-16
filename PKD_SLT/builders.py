from functools import partial
from typing import Callable, Dict, Generator, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

from PKD_SLT.config import ConfigurationError
from PKD_SLT.helpers_for_ddp import get_logger

logger = get_logger(__name__)


def build_activation(activation: str = "relu") -> Callable:
    """
    Returns the activation function
    """
    # pylint: disable=no-else-return
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "tanh":
        return torch.tanh
    elif activation == "swish":
        return nn.SiLU
    elif activation == "softsign":
        return nn.Softsign
    else:
        raise ConfigurationError(
            "Invalid activation function. Valid options: "
            "'relu', 'gelu', 'tanh', 'swish'."
        )