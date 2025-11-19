# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mixer import HouseholderProducts, DPLR, NonSelective, KroneckerProduct, Monarch

from einops import rearrange, repeat

# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from .ops.selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .model_utils import FixedPointOptimizer, shift


class FPMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        norm_eps=1e-5,
        bc_norm=True,
        glu_in_loop=False,
        # mixer options
        d_mixer=128,
        mixer_type='kronecker',
        mixer_rank=1,
        mixer_ac='silu',
        mixer_proj_rank=None,
        symm_mixer=False,
        mixer_h_dep=True,
        max_iter=100,
        use_short_conv=False,
        # optimizer options
        damping_decay_rate=0.9,
        convergence_threshold=0.1,
        n_backwards=1,
        # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        if isinstance(norm_eps, str):
            norm_eps = float(norm_eps)
        self.norm_eps = norm_eps
        self.bc_norm = bc_norm
        self.glu_in_loop = glu_in_loop

        self.symm_mixer = symm_mixer
        self.mixer_type = mixer_type
        self.mixer_h_dep = mixer_h_dep
        self.n_backwards = n_backwards
        self.use_short_conv = use_short_conv

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if use_short_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.bc_y_proj = nn.utils.parametrizations.weight_norm(nn.Linear(self.d_inner, 2 * self.d_state, bias=False))
        if not glu_in_loop:
            self.z_y_proj = nn.utils.parametrizations.weight_norm(nn.Linear(self.d_inner, self.d_inner, bias=False))

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        if symm_mixer or not mixer_h_dep:
            mixer_input_dim = self.d_inner
        else:
            mixer_input_dim = 2 * self.d_inner

        # fixed-point mixer
        if mixer_type.lower() == 'householderproduct':
            self.mixer = HouseholderProducts(mixer_input_dim, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'dplr':
            self.mixer = DPLR(mixer_input_dim, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'kronecker':
            self.mixer = KroneckerProduct(mixer_input_dim, self.d_inner, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'monarch':
            self.mixer = Monarch(mixer_input_dim, self.d_inner, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'nonselective':
            self.mixer = NonSelective(self.d_inner, self.d_inner, norm_eps=norm_eps)
        else:
            raise ValueError(
                "The mixer needs to be householder product, dplr, or non selective."
            )
        # self.mixer = torch.compile(self.mixer, fullgraph=True)

        # fixed-point optimizer
        self.optimizer = FixedPointOptimizer(
            stepsize=0.99,
            stepsize_decay=damping_decay_rate,
            decay_patience=15,
            thresh=convergence_threshold,
            outlier_quantile=0.25,
            max_iter=max_iter
        )

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_norm = nn.LayerNorm(self.d_inner, eps=norm_eps)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.fixed_point_iter = []

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """

        x, dt, A, B, C, z, ssm_state = self._prepare_components(hidden_states, inference_params=inference_params)
        with torch.no_grad():
            y_star = self._fixed_point_iter(x, dt, A, B, C, z, ssm_state)

        y = y_star
        for n in range(self.n_backwards):
            y = self._fixed_point_step(x, y, dt, A, B, C, z, ssm_state)
        
        if not self.glu_in_loop:
            z = rearrange(z, "b d l -> b l d") + self.z_y_proj(shift(y, 1, dim=1))
            out = self.out_proj(self.act(z) * y)
        else:
            out = self.out_proj(y)
        return out
    
    @torch.no_grad()
    def _fixed_point_iter(self, x, dt, A, B, C, z, ssm_state=None):
        state = self.optimizer.start(torch.zeros_like(x))

        # @torch.compile(fullgraph=True)
        def step(state:Dict[str, torch.Tensor]):
            y = self._fixed_point_step(x, state['y'], dt, A, B, C, z, ssm_state=ssm_state)
            return self.optimizer.step(state, y)

        while self.optimizer.cont(state):
            # https://github.com/pytorch/pytorch/issues/97155
            # torch._dynamo.graph_break() # to avoid recompilations?
            state = step(state)
        
        self.fixed_point_iter.append(state['iter_idx'].item() + 1)
        return state['y']

    def _fixed_point_step(self, x, y_prev, dt, A, B, C, z, ssm_state=None):
        x_tilde = self._channel_mixing(x, x - y_prev, y_prev) + y_prev

        y_shifted = shift(y_prev, 1, dim=1)
        BC_y = self.bc_y_proj(y_shifted)
        B_y, C_y = BC_y[:, :, :self.d_state], BC_y[:, :, self.d_state:]

        B_y = rearrange(B_y, "b l dstate -> b dstate l").contiguous()
        C_y = rearrange(C_y, "b l dstate -> b dstate l").contiguous()

        y = self._sequence_mixing(x_tilde, dt, A, B + B_y, C + C_y, z, ssm_state=ssm_state)
        return y

    def _channel_mixing(self, x, delta_x, y_prev):
        # get previous hidden-states for Q
        y_shifted = shift(y_prev, 1, dim=1)
        if self.mixer_h_dep:
            param_arg = x + y_shifted if self.symm_mixer else torch.cat([x, y_shifted], dim=-1)
        else:
            param_arg = x
        return self.mixer(param_arg=param_arg, input_arg=delta_x)

    def _sequence_mixing(self, x, dt, A, B, C, z, ssm_state=None):
        x = x.to(dtype=dt.dtype)
        if self.bc_norm:
            B = F.normalize(B, dim=1, eps=self.norm_eps).to(dtype=dt.dtype)
            C = F.normalize(C, dim=1, eps=self.norm_eps).to(dtype=dt.dtype)
        y = selective_scan_fn(
            rearrange(x, "b l d -> b d l"),
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z if self.glu_in_loop else None, # glu component
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        return self.out_norm(y)

    def _prepare_components(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        
        x, z = xz.chunk(2, dim=1)
        if self.use_short_conv:
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        x = rearrange(x, "b d l -> b l d")
        assert self.activation in ["silu", "swish"]

        return x, dt, A, B, C, z, ssm_state

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
