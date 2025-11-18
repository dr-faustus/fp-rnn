import numpy as np
import torch
from torch import nn

from .full_matrix import FullMatrix


class DeltaRule(FullMatrix):
    def __init__(self, config):
        super().__init__(config)
        self.A = nn.Linear(in_features=self.config.embedding_dim, out_features=self.config.embedding_dim)
        self.B = nn.Linear(in_features=self.config.embedding_dim, out_features=self.config.embedding_dim)
        self.beta = nn.Linear(in_features=self.config.embedding_dim, out_features=1)

        self.A.weight.data = self.A.weight.data / np.sqrt(self.config.embedding_dim)
        self.B.weight.data = self.B.weight.data / np.sqrt(self.config.embedding_dim)
        self.step_size = self.config.step_size

    def forward_loop(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, emb_dim = x.shape

        # Predict diagonal elements
        A_u = self.A(x)  # Shape: (batch_size, sequence_length, emb_dim)
        B_diag = self.B(x)  # Shape: (batch_size, sequence_length, emb_dim)

        A_u = A_u / A_u.norm(dim=(-1), keepdim=True)

        # Reshape A_u for batch matrix multiplication
        A_u_reshaped = A_u.unsqueeze(-1)  # Shape: (batch_size, sequence_length, emb_dim, 1)

        # Compute u @ u.T using batch matrix multiplication
        u_uT = torch.matmul(A_u_reshaped,
                            A_u_reshaped.transpose(-1, -2))  # Shape: (batch_size, sequence_length, emb_dim, emb_dim)
        # Compute A = I + u @ u.T
        beta = torch.sigmoid(self.beta(x)).unsqueeze(-1)
        A = torch.eye(emb_dim, device=x.device).unsqueeze(0).unsqueeze(0) - self.step_size * beta * u_uT  # Shape: (batch_size, sequence_length, emb_dim, emb_dim)

        # Create diagonal matrices B
        B = torch.diag_embed(B_diag)  # Shape: (batch_size, sequence_length, emb_dim, emb_dim)

        # Compute h for all timesteps after the first one
        h = self.forward_recurrence(A, B, x)
        return h

