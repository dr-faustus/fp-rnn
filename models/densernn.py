import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, List
from .mixer import HouseholderProducts, DPLR, NonSelective, KroneckerProduct, Monarch, Dense

from .model_utils import FixedPointOptimizer, shift
from .ops.linrec import linrec_hop

from torch._higher_order_ops.scan import scan

class DenseRNN(nn.Module):
    def __init__(
        self, 
        d_model,
        d_mixer,
        expand=2,
        mixer_ac='identity',
        mixer_rank=1,
        mixer_type='householderproduct',
        mixer_proj_rank=None,
        layer_idx=None,
        device=None,
        dtype=None,
        c=8
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand
        #self.mixer_rank = mixer_rank
        #self.max_iter = max_iter
        self.c = c

        a = np.log((1 / np.exp(np.log(0.999) / self.c)) - 1)
        b = np.log((1 / np.exp(np.log(0.9) / self.c)) - 1)
        self.omega = nn.Parameter(a + (b - a) * torch.rand((1, 1, self.d_inner)))
        self.omega._no_weight_decay = True         
        self.lambda_proj = nn.Linear(self.d_inner, self.d_inner)

        if mixer_type.lower() == 'householderproduct':
            self.mixer = HouseholderProducts(self.d_inner, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank)
        elif mixer_type.lower() == 'dplr':
            self.mixer = DPLR(self.d_inner, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank)
        elif mixer_type.lower() == 'kronecker':
            self.mixer = KroneckerProduct(self.d_inner, self.d_inner, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank)
        elif mixer_type.lower() == 'monarch':
            self.mixer = Monarch(self.d_inner, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank)
        elif mixer_type.lower() == 'nonselective':
            self.mixer = NonSelective(self.d_inner, self.d_inner)
        else:
            raise ValueError(
                "The mixer needs to be householder product, dplr, or non selective."
            )

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)    # x,z
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    #@torch.compile(fullgraph=True)
    def forward(self, x, inference_params=None):
        x, z = torch.tensor_split(self.in_proj(x), 2, dim=-1)                   # [B, T, D_inner]
        #x, z = xz[:, :, :self.d_inner], xz[:, :, self.d_inner:]

        log_lambda = - self.c * F.softplus(self.omega) * torch.sigmoid(self.lambda_proj(x))
        lambdA = log_lambda.exp()                                               # [B, T, D_inner]
  
        if True:
            lambdA, x = lambdA.unbind(1), x.unbind(1)
            h:List[torch.Tensor] = [x[0]]
            for t in range(1, len(x)):
                Qx = self.mixer(x[t], h[t-1])
                h.append(lambdA[t] * Qx + (1 - lambdA[t]) * x[t])
            h = torch.stack(h, dim=1)

        if False: # '2.8' == torch.__version__[:3] (about 3 times slower and throws error with compile)
            def step(h:torch.Tensor, curr):
                Qx = self.mixer(curr['x'], h)
                h = curr['lambdA'] * Qx + (1 - curr['lambdA']) * curr['x']
                return h, h
            last, h = scan(step, init=x[:,0].contiguous(), xs=dict(x=x, lambdA=lambdA), dim=1)
            h = h.transpose(0,1) # scan bug: does not move dimension back

        return self.out_proj(F.silu(z) * h)
