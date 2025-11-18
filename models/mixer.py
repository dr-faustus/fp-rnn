import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from abc import ABC, abstractmethod

class Mixer(nn.Module, ABC):
    @abstractmethod
    def __init__(self, in_features, out_features, rank=1, mixer_ac='identity', proj_rank=-1, norm_eps=1e-5):
        super().__init__()
        assert mixer_ac.lower() in ['identity', 'relu', 'silu']
        if isinstance(norm_eps, str):
            norm_eps = float(norm_eps)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.proj_rank = proj_rank
        self.norm_eps = norm_eps

        if mixer_ac.lower() == 'identity':
            self.ac = nn.Identity()
        elif mixer_ac.lower() == 'relu':
            self.ac = nn.ReLU()
        elif mixer_ac.lower() == 'silu':
            self.ac = nn.SiLU()
        else:
            raise ValueError(
                "The mixer activation function needs to be either identity, relu, or silu."
            )
    
    @abstractmethod
    def forward(self, param_arg, input_arg):
        pass

class HouseholderProducts(Mixer):
    def __init__(self, in_features, out_features, rank=1, mixer_ac='identity', proj_rank=-1, norm_eps=1e-5):
        super().__init__(in_features, out_features, rank, mixer_ac, proj_rank, norm_eps=norm_eps)

        if proj_rank == -1:
            self.u_proj = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, rank * out_features, bias=False))
        else:
            self.u_proj = nn.Sequential(
                nn.utils.parametrizations.weight_norm(nn.Linear(in_features, proj_rank, bias=False)),
                nn.utils.parametrizations.weight_norm(nn.Linear(proj_rank, rank * out_features, bias=False))
            )
        self.alpha_proj = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, rank, bias=True))

    def forward(self, param_arg, input_arg):
        shape = input_arg.shape

        input_arg = input_arg.reshape(-1, self.out_features)

        u = self.u_proj(param_arg)
        u = u.reshape(-1, self.rank, self.out_features)
        u = F.normalize(self.ac(u), dim=-1, eps=self.norm_eps)

        alpha = torch.sigmoid(self.alpha_proj(param_arg).reshape(-1, self.rank, 1))

        for idx in range(self.rank):
            input_arg = input_arg - 2 * alpha[:, idx] * u[:, idx] * (u[:, idx] * input_arg).sum(dim=-1, keepdim=True)
        
        return input_arg.reshape(shape)

class DPLR(Mixer):
    def __init__(self, in_features, out_features, rank=1, mixer_ac='identity', proj_rank=-1, norm_eps=1e-5):
        super().__init__(in_features, out_features, rank, mixer_ac, proj_rank, norm_eps=norm_eps)

        if proj_rank == -1:
            self.u_proj = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, rank * out_features, bias=False))
        else:
            self.u_proj = nn.Sequential(
                nn.utils.parametrizations.weight_norm(nn.Linear(in_features, proj_rank, bias=False)),
                nn.utils.parametrizations.weight_norm(nn.Linear(proj_rank, rank * out_features, bias=False))
            )
        self.alpha_proj = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, rank, bias=True))

    def forward(self, param_arg, input_arg):
        B, T = input_arg.shape[0], input_arg.shape[1]

        u = self.u_proj(param_arg).reshape(B, T, self.rank, self.out_features)
        u = F.normalize(self.ac(u), dim=-1, eps=self.norm_eps)
        alpha = torch.sigmoid(self.alpha_proj(param_arg).reshape(B, T, self.rank, 1))

        return input_arg - 2 * (alpha * u * (u * input_arg.unsqueeze(-2)).sum(dim=-1, keepdim=True)).sum(dim=-2)

class KroneckerProduct(Mixer):
    def __init__(self, in_features, out_features, mixer_ac='identity', proj_rank=-1, symmetric=True, normalization='column-wise', norm_eps=1e-5):
        assert normalization in ['column-wise', 'spectral']
        super().__init__(in_features, out_features, mixer_ac=mixer_ac, proj_rank=proj_rank, norm_eps=norm_eps)

        root = int(np.sqrt(self.out_features))
        self.symmetric = symmetric
        self.normalization = normalization

        if root ** 2 != self.out_features:
            for a in range(root, 0, -1):
                if self.out_features % a == 0:
                    b = self.out_features // a
                    break
            self.mat_dim_1 = a
            self.mat_dim_2 = b
        else:
            self.mat_dim_1, self.mat_dim_2 = root, root

        if symmetric:
            if proj_rank == -1:
                self.uv_project = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, 
                                                                                  (self.mat_dim_1 ** 2 + self.mat_dim_1) // 2 + (self.mat_dim_2 ** 2 + self.mat_dim_2) // 2, 
                                                                                  bias=False))
                # self.uv_project = nn.Linear(in_features, (self.mat_dim_1 ** 2 + self.mat_dim_1) // 2 + (self.mat_dim_2 ** 2 + self.mat_dim_2) // 2, bias=False)
            else:
                self.uv_project = nn.Sequential(
                    nn.utils.parametrizations.weight_norm(nn.Linear(in_features, proj_rank, bias=False)),
                    nn.utils.parametrizations.weight_norm(nn.Linear(proj_rank, 
                                                                    (self.mat_dim_1 ** 2 + self.mat_dim_1) // 2 + (self.mat_dim_2 ** 2 + self.mat_dim_2) // 2, 
                                                                    bias=False))
                )

            self.triu_indices_1 = torch.triu_indices(self.mat_dim_1, self.mat_dim_1)
            self.triu_indices_2 = torch.triu_indices(self.mat_dim_2, self.mat_dim_2)
        else:
            if proj_rank == -1:
                self.uv_project = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, self.mat_dim_1 ** 2 + self.mat_dim_2 ** 2, bias=False))
                # self.uv_project = nn.Linear(in_features, self.mat_dim_1 ** 2 + self.mat_dim_2 ** 2, bias=False)
            else:
                self.uv_project = nn.Sequential(
                    nn.utils.parametrizations.weight_norm(nn.Linear(in_features, proj_rank, bias=False)),
                    nn.utils.parametrizations.weight_norm(nn.Linear(proj_rank, self.mat_dim_1 ** 2 + self.mat_dim_2 ** 2, bias=False))
                )
        
        self.diag = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, self.mat_dim_1 + self.mat_dim_2, bias=True))
        # self.diag = nn.Linear(in_features, self.mat_dim_1 + self.mat_dim_2, bias=True)

    def forward(self, param_arg, input_arg):
        B, T = input_arg.shape[0], input_arg.shape[1]

        input_arg_mat = input_arg.reshape(B * T, self.mat_dim_1, self.mat_dim_2).transpose(-1, -2)
        params = self.ac(self.uv_project(param_arg))
        diag = torch.sigmoid(self.diag(param_arg)).reshape(B * T, -1)

        params = params.reshape(B * T, -1)

        if self.symmetric:
            params_size = (self.mat_dim_1 ** 2 + self.mat_dim_1) // 2
        else:
            params_size = self.mat_dim_1 ** 2
        
        params_1 = params[:, :params_size]
        diag_1 = diag[:, :self.mat_dim_1]

        params_2 = params[:, params_size:]
        diag_2 = diag[:, self.mat_dim_1:]

        if self.symmetric:
            u = torch.zeros(B * T, self.mat_dim_1, self.mat_dim_1, device=params.device, dtype=params.dtype)
            v = torch.zeros(B * T, self.mat_dim_2, self.mat_dim_2, device=params.device, dtype=params.dtype)

            u[:, self.triu_indices_1[0], self.triu_indices_1[1]] = params_1
            v[:, self.triu_indices_2[0], self.triu_indices_2[1]] = params_2

            u = torch.bmm(u, u.transpose(-1, -2))
            v = torch.bmm(v, v.transpose(-1, -2))
        else:
            u = params_1.reshape(B * T, self.mat_dim_1, self.mat_dim_1)
            v = params_2.reshape(B * T, self.mat_dim_2, self.mat_dim_2)

        # if self.normalization == 'column-wise':
        u = F.normalize(u, dim=-1, eps=self.norm_eps)
        v = F.normalize(v, dim=-1, eps=self.norm_eps)
        if self.normalization == 'spectral':
            u = u / (self._power_iters(u, num_iters=16) + self.norm_eps)
            v = v / (self._power_iters(v, num_iters=16) + self.norm_eps)

        u = torch.bmm(torch.diag_embed(diag_1), u.reshape(B * T, self.mat_dim_1, self.mat_dim_1))
        v = torch.bmm(torch.diag_embed(diag_2), v.reshape(B * T, self.mat_dim_2, self.mat_dim_2))

        return input_arg - 2 * torch.bmm(torch.bmm(v, input_arg_mat), u).transpose(-1, -2).reshape(B, T, self.out_features)
    
    @torch.compile(fullgraph=True)
    def _power_iters(self, mat, num_iters=5):
        assert len(mat.shape) == 3 or len(mat.shape) == 2
        mat_dim = mat.shape[-1]

        v = torch.ones(mat.shape[0], mat_dim, device=mat.device) * (1 / np.sqrt(self.out_features))
        for idx in range(num_iters):
            if idx % 2 == 0:
                u = torch.bmm(mat, v.unsqueeze(-1)).squeeze()
                u = F.normalize(u, dim=-1)
            else:
                v = torch.bmm(mat, u.unsqueeze(-1)).squeeze()
                v = F.normalize(v, dim=-1)
            
        spec_norm = torch.bmm(u.unsqueeze(-2), torch.bmm(mat, v.unsqueeze(-1)))

        return spec_norm.reshape(mat.shape[0], 1, 1)
    
class MixtureOfMixers(Mixer):
    def __init__(self, in_features, out_features, rank=2, mixer_ac='identity', proj_rank=-1, norm_eps=1e-5):
        assert rank > 1
        super().__init__(in_features, out_features, mixer_ac=mixer_ac, proj_rank=proj_rank, norm_eps=norm_eps)
        self.mixers = nn.ModuleList()
        for r in range(rank):
            if r % 2 == 0:
                self.mixers.append(HouseholderProducts(in_features, out_features, rank=1, mixer_ac=mixer_ac, proj_rank=proj_rank))
            else:
                self.mixers.append(KroneckerProduct(in_features, out_features, mixer_ac=mixer_ac, proj_rank=proj_rank))
    
    def forward(self, param_arg, input_arg):
        for mixer in self.mixers:
            input_arg = mixer(param_arg=param_arg, input_arg=input_arg)
        return input_arg

        
class Monarch(Mixer):
    def __init__(self, in_features, out_features, rank=32, mixer_ac='identity', proj_rank=-1, norm_eps=1e-5):
        super().__init__(in_features, out_features, rank, mixer_ac, proj_rank, norm_eps=norm_eps)
        # make sure both rank and out features are powers of 2
        assert (out_features & (out_features - 1) == 0) and out_features != 0
        assert (rank & (rank - 1) == 0) and rank != 0

        self.n_components = int(self.out_features / self.rank)

        if proj_rank == -1:
            self.uv_project = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, 
                                                                              self.n_components * int(self.rank * (self.rank + 1) / 2), 
                                                                              bias=False))
        else:
            self.uv_project = nn.Sequential(
                nn.utils.parametrizations.weight_norm(nn.Linear(in_features, proj_rank, bias=False)),
                nn.utils.parametrizations.weight_norm(nn.Linear(proj_rank, 
                                                                self.n_components * int(self.rank * (self.rank + 1) / 2), 
                                                                bias=False))
            )
        self.alpha = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, self.n_components, bias=True))

        self.triu_indices = torch.triu_indices(rank, rank)

    def forward(self, param_arg, input_arg):
        B, T = input_arg.shape[0], input_arg.shape[1]

        params = self.ac(self.uv_project(param_arg))
        alpha = torch.sigmoid(self.alpha(param_arg)).reshape(-1, 1, 1)

        params = params.reshape(B * T * self.n_components, int((self.rank * (self.rank + 1)) / 2))

        uv = torch.zeros(B * T * self.n_components, self.rank, self.rank, device=params.device)
        uv[:, self.triu_indices[0], self.triu_indices[1]] = params
        uv = torch.bmm(uv, uv.transpose(-1, -2))
        uv = uv / (uv.norm(dim=-2, keepdim=True).max(dim=-1, keepdim=True)[0] + self.norm_eps)

        return input_arg - torch.bmm(alpha * uv, input_arg.reshape(-1, self.rank, 1)).reshape(B, T, self.out_features)

class SelectiveDense(Mixer):
    def __init__(self, in_features, out_features, mixer_ac='identity', proj_rank=-1, norm_eps=1e-5):
        super().__init__(in_features, out_features, mixer_ac=mixer_ac, proj_rank=proj_rank, norm_eps=norm_eps)

        if proj_rank == -1:
            self.proj = nn.utils.parametrizations.weight_norm(nn.Linear(in_features, 
                                                                        (out_features ** 2 + out_features) // 2, 
                                                                        bias=False))
        else:
            self.proj = nn.Sequential(
                nn.utils.parametrizations.weight_norm(nn.Linear(in_features, proj_rank, bias=False)),
                nn.utils.parametrizations.weight_norm(nn.Linear(proj_rank, out_features ** 2, bias=False))
            )
        
        self.triu_indices = torch.triu_indices(self.out_features, self.out_features)

    def forward(self, param_arg, input_arg):
        B, T = input_arg.shape[0], input_arg.shape[1]

        input_arg = input_arg.reshape(B * T, self.out_features)

        params = self.ac(self.proj(param_arg))
        params = params.reshape(B * T, (self.out_features ** 2 + self.out_features) // 2)

        Q = torch.zeros(B * T, self.out_features, self.out_features, device=params.device, dtype=params.dtype)
        Q[:, self.triu_indices[0], self.triu_indices[1]] = params
        Q = torch.bmm(Q, Q.transpose(-1, -2))
        Q = Q / (self._power_iters(Q, num_iters=32) + self.norm_eps)

        return torch.bmm(Q, input_arg.unsqueeze(-1)).reshape(B, T, self.out_features)
        
    @torch.compile(fullgraph=True)
    def _power_iters(self, mat, num_iters=5):
        assert len(mat.shape) == 3 or len(mat.shape) == 2
        mat_dim = mat.shape[-1]

        v = torch.ones(mat.shape[0], mat_dim, 1, device=mat.device) * (1 / np.sqrt(self.out_features))
        for idx in range(num_iters):
            v = mat @ v
            eig_val = v.norm(dim=-2, keepdim=True)
            v = v / (eig_val + self.norm_eps)
        return eig_val


class NonSelective(Mixer):
    def __init__(self, in_features, out_features, norm_eps=1e-5):
        super().__init__(in_features, out_features, norm_eps=norm_eps)

        self.q_proj = nn.utils.parametrizations.orthogonal(nn.Linear(in_features, out_features, bias=False))

    def forward(self, param_arg, input_arg):
        B, T = input_arg.shape[0], input_arg.shape[1]

        return self.q_proj(input_arg)
