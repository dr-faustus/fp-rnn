import torch.nn as nn

from torch import Tensor
from models.simple_recurrent.block import SimpleRecurrentBlock
from models.simple_recurrent.block import SimpleRecurrentConfig


class LSTM(nn.Module):
    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_vocab: int,
    ):
        """Initialize a bare LSTM module."""
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_vocab = n_vocab

        config = SimpleRecurrentConfig(layer_type='LSTM', embedding_dim=self.d_model)

        self.embedding = nn.Embedding(n_vocab, d_model)

        self.layers = nn.ModuleList(
            [
                SimpleRecurrentBlock(config=config)
                for _ in range(n_layers)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model)


    def forward(self, x: Tensor, inference_params=None) -> Tensor:
        """Perform the forward pass."""
        x = self.embedding(x)
        for layer in self.layers:
            residual = x
            x = layer(x)
        residual = (x + residual) if residual is not None else x
        x = self.norm_f(residual)
        return x

class LSTMTokenClassifier(LSTM):
    """A Mamba model with a token classification head."""

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)
        self.cl_head = nn.Linear(self.d_model, self.n_vocab)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass."""
        x = super().forward(x)
        x = self.cl_head(x)
        return x
