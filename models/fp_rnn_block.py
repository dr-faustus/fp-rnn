import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict
from .mixer import HouseholderProducts, DPLR, NonSelective, KroneckerProduct, Monarch, MixtureOfMixers, SelectiveDense

from .model_utils import FixedPointOptimizer, shift
from models.ops.linrec import linrec_hop

from causal_conv1d import causal_conv1d_fn
from einops import rearrange

class FPRNN(nn.Module):
    def __init__(
        self, 
        d_model,
        d_mixer,
        d_conv=4,
        expand=2,
        mixer_ac='silu',
        mixer_rank=1,
        mixer_type='householderproduct',
        max_iter=1000,
        
        mixer_proj_rank=None,
        symm_mixer=False,
        mixer_h_dep=True,
        n_backwards=1,

        damping_decay_rate=0.9,
        convergence_threshold=0.1,
        conv_bias=True,
        norm_eps=1e-5,
        use_short_conv=False,

        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert mixer_type.lower() in ['householderproduct', 'dplr', 'nonselective', 'kronecker', 'monarch', 'mixture', 'selectivedense']


        self.d_model = d_model
        if mixer_type != 'selectivedense':
            self.d_inner = d_model * expand
        else:
            self.d_inner = 256
        self.mixer_rank = mixer_rank
        self.max_iter = max_iter

        self.symm_mixer = symm_mixer
        self.mixer_type = mixer_type
        self.mixer_h_dep = mixer_h_dep
        self.n_backwards = n_backwards
        self.use_short_conv = use_short_conv

        self.lambda_proj = nn.Linear(self.d_inner, self.d_inner)
        if use_short_conv:
            self.conv1d_x = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )
            self.conv1d_c = nn.Conv1d(
                in_channels=2 * self.d_inner,
                out_channels=2 * self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=2 * self.d_inner,
                padding=d_conv - 1,
            )
            self.conv1d_mixer = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv-1
            )
            self.conv1d_lambda = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1
            )

        if symm_mixer or not mixer_h_dep:
            mixer_input_dim = self.d_inner
        else:
            mixer_input_dim = 2 * self.d_inner
        
        if mixer_type.lower() == 'householderproduct':
            self.mixer = HouseholderProducts(mixer_input_dim, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'dplr':
            self.mixer = DPLR(mixer_input_dim, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'kronecker':
            self.mixer = KroneckerProduct(mixer_input_dim, self.d_inner, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps, normalization='spectral')
        elif mixer_type.lower() == 'monarch':
            self.mixer = Monarch(mixer_input_dim, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'nonselective':
            self.mixer = NonSelective(self.d_inner, self.d_inner, norm_eps=norm_eps)
        elif mixer_type.lower() == 'mixture':
            self.mixer = MixtureOfMixers(mixer_input_dim, self.d_inner, rank=mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'selectivedense':
            self.mixer = SelectiveDense(mixer_input_dim, self.d_inner, mixer_ac, mixer_proj_rank, norm_eps=norm_eps)
        else:
            raise ValueError(
                "The mixer needs to be householder product, dplr, or non selective."
            )
        self.mixer = torch.compile(self.mixer, fullgraph=True)
        
        self.b_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.c_proj = nn.Linear(2 * self.d_inner, self.d_inner, bias=False)
        
        # fixed-point optimizer
        self.optimizer = FixedPointOptimizer(
            stepsize=1,
            stepsize_decay=1.0,
            decay_patience=15,
            thresh=convergence_threshold,
            outlier_quantile=0.1,
            max_iter=max_iter,
        )

        self.input_trans = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.output_trans = nn.Linear(self.d_inner, self.d_model, bias=False)

        self.output_norm = nn.RMSNorm(self.d_inner, eps=norm_eps)

        self.fixed_point_iter = []

    def forward(self, x, inference_params=None):
        # x \in R^{B, T, I}

        xz = self.input_trans(x)
        x, z = xz[:, :, :self.d_inner], xz[:, :, self.d_inner:]

        x = rearrange(x, 'b l d -> b d l')
        if self.use_short_conv:
            x_mixer = rearrange(causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d_mixer.weight, "d 1 w -> d w"),
                    bias=self.conv1d_mixer.bias,
                    activation='silu',
                ), 'b d l -> b l d').contiguous()
            
            x_b = rearrange(causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    bias=self.conv1d_b.bias,
                    activation='silu',
                ), 'b d l -> b l d').contiguous()
            
            x_lambda = rearrange(causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d_lambda.weight, "d 1 w -> d w"),
                    bias=self.conv1d_lambda.bias,
                    activation='silu',
                ), 'b d l -> b l d').contiguous()

            x = rearrange(causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d_x.weight, "d 1 w -> d w"),
                    bias=self.conv1d_x.bias,
                    activation='silu',
                ), 'b d l -> b l d').contiguous()
        else:
            x_mixer, x_b, x_lambda = x, x, x

        b = F.silu(self.b_proj(x_b))

        lambdA = torch.sigmoid(self.lambda_proj(x_lambda))

        h_star = self._fixed_point_iter(x, x_mixer, b, lambdA)
        
        h = h_star
        for n in range(self.n_backwards):
            h = self._fixed_point_step(x, x_mixer, b, h, lambdA)

        if self.use_short_conv:
            x_c = torch.cat([x, h], dim=-1)
            x_c = rearrange(causal_conv1d_fn(
                    x=rearrange(x_c, 'b l d -> b d l'),
                    weight=rearrange(self.conv1d_c.weight, "d 1 w -> d w"),
                    bias=self.conv1d_c.bias,
                    activation='silu',
                ), 'b d l -> b l d').contiguous()
        else:
            h_shifted = shift(h, 1, dim=1)
            x_c = torch.cat([x, h_shifted], dim=-1)
        c = F.silu(self.c_proj(x_c))

        y = c * h
        
        return self.output_trans(self.output_norm(F.silu(z) * y))
    
    @torch.no_grad()
    def _fixed_point_iter(self, x, x_mixer, b, lambdA, cumsum_log_lambda=None):

        state = self.optimizer.start(torch.zeros_like(x))

        @torch.compile(fullgraph=True)
        def step(state:Dict[str, torch.Tensor]):
            h = self._fixed_point_step(x, x_mixer, b, state['y'], lambdA, cumsum_log_lambda)
            return self.optimizer.step(state, h)

        while self.optimizer.cont(state):
            # https://github.com/pytorch/pytorch/issues/97155
            # torch._dynamo.graph_break() # to avoid recompilations?
            state = step(state)
         
        self.fixed_point_iter.append(state['iter_idx'].item() + 1)
        return state['y']
    
    def _fixed_point_step(self, x, x_mixer, b, h_prev, lambdA, cumsum_log_lambda=None):
        x_tilde = self._channel_mixing(x_mixer, b * x - h_prev, h_prev) + h_prev
        h = self._sequence_mixing(x_tilde, lambdA, cumsum_log_lambda=cumsum_log_lambda)

        return h

    def _channel_mixing(self, x_mixer, delta_x, h_prev):
        h_shifted = shift(h_prev, 1, dim=1)
        if self.mixer_h_dep:
            param_arg = x_mixer + h_shifted if self.symm_mixer else torch.cat([x_mixer, h_shifted], dim=-1)
        else:
            param_arg = x_mixer
        return self.mixer(param_arg=param_arg, input_arg=delta_x)
    
    def _sequence_mixing(self, x, lambdA, cumsum_log_lambda=None):
        # performs the recurrent step
        return linrec_hop(inputs=(1 - lambdA) * x, coeffs=lambdA, dim=1)
    
    def _log_space_scan(self, cumsum_log_A, x):
        # A in log space
        # this function performs associative scans for diagonal matrix product and sum.
        # super in-efficient backward pass, also we have to use the complex domain
        # to get the log of the input since it can be negative.
        # h[t] = A h[t-1] + x[t]
        scaled_x = torch.logcumsumexp(torch.log(x) - cumsum_log_A, dim=1)

        hidden = (cumsum_log_A + scaled_x).exp()

        return hidden
    
    def _seq_scan(self, A, x):
        # this function performs sequential scan for diagonal state transition matrix
        # h[t] = A h[t-1] + x[t]

        L = x.shape[1]

        hidden = [0]
        for t in range(L):
            hidden.append(A[:, t] * hidden[-1] + x[:, t])

        hidden = torch.cat([h.unsqueeze(1) for h in hidden[1:]], dim=1)

        return hidden
    
    def set_max_iter(self, max_iter):
        self.max_iter = max_iter
        self.optimizer.max_iter = max_iter