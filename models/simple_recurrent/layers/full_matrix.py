from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..cummulitive_matrix_product import batched_cumulative_matrix_multiplication
from models.xlstm.utils import UpProjConfigMixin


@dataclass
class FullMatrixConfig(UpProjConfigMixin):
    def __init__(self, embedding_dim: int):
        self.embedding_dim: int = embedding_dim


class FullMatrix(nn.Module):
    config_class = FullMatrixConfig

    def __init__(self, config: FullMatrixConfig):
        super().__init__()
        self.config = config
        self.A = nn.Linear(in_features=self.config.embedding_dim,
                           out_features=self.config.embedding_dim * self.config.embedding_dim)
        self.B = nn.Linear(in_features=self.config.embedding_dim,
                           out_features=self.config.embedding_dim * self.config.embedding_dim)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_loop = self.forward_loop(x)
        return h_loop

    def forward_efficient(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Efficient method is not implemented yet')
        '''
        Iteration 1:
        Ax: tensor([-1.8369,  1.6886])
        Bx: tensor([-0.1101, -0.9264])

        Iteration 2:
        Ax: tensor([-1.0966, -0.0215])
        Bx: tensor([-0.1136, -0.4069])

        Iteration 3:
        Ax: tensor([ 1.7371, -0.5649])
        Bx: tensor([-0.1339, -0.8847])

        Iteration 4:
        Ax: tensor([ 0.8469, -0.5070])
        Bx: tensor([-0.0807, -0.2044])

        Iteration 5:
        Ax: tensor([-0.7016,  0.0737])
        Bx: tensor([-0.0042, -0.0515])
        '''
        batch_size, sequence_length, emb_dim = x.shape

        A = self.A(x).view(batch_size, sequence_length, emb_dim, emb_dim) / torch.sqrt(emb_dim)
        B = self.B(x).view(batch_size, sequence_length, emb_dim, emb_dim) / torch.sqrt(emb_dim)

        # Compute B * x for all time steps
        Bx = torch.matmul(B, x.unsqueeze(-1)).squeeze(-1)

        # Initialize h with the correct initial state
        h = torch.zeros_like(x)

        # Compute cumulative products of A, starting from the second time step
        A_cumprod = batched_cumulative_matrix_multiplication(A)

        # Compute the terms of the sum, starting from the second time step
        terms = torch.matmul(A_cumprod, Bx[:, :-1].unsqueeze(-1)).squeeze(-1)

        # Compute the final sum and add it to Bx
        Ah = torch.cumsum(terms, dim=1)
        h[:, 1:] = Ah + Bx[:, 1:]

        return h

    def forward_loop(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, emb_dim = x.shape

        A = self.A(x).view(batch_size, sequence_length, emb_dim, emb_dim) / np.sqrt(emb_dim)
        B = self.B(x).view(batch_size, sequence_length, emb_dim, emb_dim) / np.sqrt(emb_dim)

        # Compute h for all timesteps after the first one
        h = self.forward_recurrence(A, B, x)
        return h

    def forward_recurrence(self, A, B, x):
        batch_size, sequence_length, emb_dim = x.shape

        # Initialize h_list with the initial state
        h_list = [torch.bmm(B[:, 0], x[:, 0].unsqueeze(-1)).squeeze(-1)]

        # Compute h for all timesteps after the first one
        for t in range(1, sequence_length):
            h_prev = h_list[-1]
            h_curr = (torch.bmm(A[:, t], h_prev.unsqueeze(-1)).squeeze(-1) +
                      torch.bmm(B[:, t], x[:, t].unsqueeze(-1)).squeeze(-1))
            h_list.append(h_curr)

        # Stack the list of h tensors to create the final output
        h = torch.stack(h_list, dim=1)
        return h

    def forward_loop_naive(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, emb_dim = x.shape

        A = self.A(x).view(batch_size, sequence_length, emb_dim, emb_dim) / np.sqrt(emb_dim)
        B = self.B(x).view(batch_size, sequence_length, emb_dim, emb_dim) / np.sqrt(emb_dim)

        # Initialize output tensor
        h = torch.zeros(batch_size, sequence_length, emb_dim, device=x.device)

        for b in range(batch_size):
            # Initialize h_list with the initial state as zero for this batch
            h_list = [torch.mv(B[b, 0], x[b, 0])]

            # Compute h for all timesteps after the first one
            for t in range(1, sequence_length):
                h_prev = h_list[-1]
                Ah = torch.mv(A[b, t], h_prev)
                Bx = torch.mv(B[b, t], x[b, t])
                print(f'Iteration {t}:\nAh: {Ah}\nBx: {Bx}')
                h_curr = Ah + Bx
                h_list.append(h_curr)

            # Stack the list of h tensors for this batch
            h[b] = torch.stack(h_list, dim=0)

        return h

    def step(self, x: torch.Tensor, mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
             conv_state: tuple[torch.Tensor] = None) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        raise NotImplementedError('FullMatrix does not support step function')

    def reset_parameters(self):
        pass
