import math
from dataclasses import dataclass
from typing import Union
import torch.nn as nn

from models import BedouinBlock

@dataclass
class BedouinConfig:
    d_model: int  #  D
    n_layers: int
    d_state: int = 16  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 1

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    positive_and_negative: bool = True

    mixer_rank: int = 4
    c: int = 8
    max_iter: int = 100
    damping_decay_rate: float = 0.9
    convergence_threshold: float = 0.1

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    pscan: bool = True  #  use parallel scan mode or sequential mode when training
    use_cuda: bool = True  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class Bedouin(nn.Module):
    def __init__(self, config: BedouinConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([BedouinBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            expand=config.expand_factor,
            mixer_rank=config.mixer_rank,
            max_iter=config.max_iter,
            damping_decay_rate=config.damping_decay_rate,
            convergence_threshold=config.convergence_threshold
        ) for _ in range(config.n_layers)])

    def forward(self, x):
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)

        return x, residual
    
    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class BedouinLM(Bedouin):
    def __init__(self, config: BedouinConfig, vocab_size, embedding_dim, positive_and_negative):
        config.positive_and_negative = positive_and_negative
        print(f'CUDA: {config.use_cuda}')
        super().__init__(config)
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.layers = nn.ModuleList([BedouinBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            expand=config.expand_factor,
            mixer_rank=config.mixer_rank,
            max_iter=config.max_iter,
            damping_decay_rate=config.damping_decay_rate,
            convergence_threshold=config.convergence_threshold
        ) for _ in range(config.n_layers)])

        self.lm_head = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
            bias=False,
        )
        self.norm_f = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.token_embedding(x)
        x, residual = super().forward(x)
        residual = (x + residual) if residual is not None else x
        x = self.norm_f(residual)
        x = self.lm_head(x)
        return x
