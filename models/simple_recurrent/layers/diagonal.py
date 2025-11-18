import numpy as np
import torch
from torch import nn

from .full_matrix import FullMatrix


class Diagonal(FullMatrix):
    def __init__(self, config):
        super().__init__(config)
        if self.config.activation_func.lower() != 'none':
            if self.config.activation_func == 'sigmoid':
                self.activation_func = torch.nn.Sigmoid()
            elif self.config.activation_func == 'tanh':
                self.activation_func = torch.nn.Tanh()
            elif self.config.activation_func == 'relu':
                self.activation_func = torch.nn.ReLU()
            else:
                raise ValueError("Activation function must be either sigmoid or tanh")
        self.A = nn.Linear(in_features=self.config.embedding_dim,
                           out_features=self.config.embedding_dim)
        self.B = nn.Linear(in_features=self.config.embedding_dim,
                           out_features=self.config.embedding_dim)

        self.A.weight.data = self.A.weight.data / np.sqrt(self.config.embedding_dim)
        self.B.weight.data = self.B.weight.data / np.sqrt(self.config.embedding_dim)

    def forward_loop(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, emb_dim = x.shape

        # Predict diagonal elements
        A_diag = self.A(x)
        B_diag = self.B(x)

        # Activation function
        if hasattr(self, 'activation_func'):
            A_diag = self.activation_func(10 * A_diag)

        # Create diagonal matrices
        A = torch.diag_embed(A_diag)
        B = torch.diag_embed(B_diag)

        # Compute h for all timesteps after the first one
        h = self.forward_recurrence(A, B, x)
        return h
