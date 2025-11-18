# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    from layers.delta_net_fla import DeltaNet
except ImportError:
    pass
from .layers.delta_rule import DeltaRule
from .layers.full_matrix import FullMatrix
from .layers.diagonal import Diagonal
from models.xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from models.xlstm.components.ln import LayerNorm


@dataclass
class SimpleRecurrentConfig:
    layer_type: str
    layer_config: Optional[mLSTMLayerConfig] = None
    embedding_dim: int = 1


class SimpleRecurrentBlock(nn.Module):
    config_class = SimpleRecurrentConfig

    def __init__(self, config: SimpleRecurrentConfig) -> None:
        super().__init__()
        self.config = config
        embedding_dim = config.embedding_dim
        self.pre_norm = LayerNorm(ndim=embedding_dim, weight=True, bias=False)
        self.post_norm = LayerNorm(ndim=embedding_dim, weight=True, bias=False)

        if self.config.layer_type.lower() == 'full_matrix':
            self.recurrent_layer = FullMatrix(config=self.config)
        elif self.config.layer_type.lower() == 'diagonal':
            self.recurrent_layer = Diagonal(config=self.config)
        elif self.config.layer_type.lower() == 'delta_rule':
            self.recurrent_layer = DeltaRule(config=self.config)
        elif self.config.layer_type.lower() == 'delta_rule_fla':
            self.recurrent_layer = DeltaNet(d_model=self.config.embedding_dim, **self.config)
        elif self.config.layer_type.lower() == 'lstm':
            self.recurrent_layer = nn.LSTM(input_size=self.config.embedding_dim,
                                           hidden_size=self.config.embedding_dim,
                                           batch_first=True)
        else:
            raise ValueError("Either mlstm or slstm must be provided")

        self.ffn_norm = None
        self.ffn = None

        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.pre_norm(x)
        if self.config.layer_type.lower() == 'lstm':
            x = self.recurrent_layer(x)
        else:
            x = self.recurrent_layer(x, **kwargs)
        if isinstance(x, tuple):
            # for delta rule from fla more values are returned.
            x = x[0]
        x = self.post_norm(x)
        return x

    def step(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        raise NotImplementedError('Step method is not implemented yet')

    def reset_parameters(self) -> None:
        # check that recurrent layer has method rese_parameters
        if hasattr(self.recurrent_layer, 'reset_parameters'):
            self.recurrent_layer.reset_parameters()