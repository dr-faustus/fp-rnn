# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from huggingface_hub import PyTorchModelHubMixin

from .model_utils import FixedPointOptimizer, shift

from .mixer import HouseholderProducts, DPLR, NonSelective, KroneckerProduct, Monarch


class FPMamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
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
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.bc_norm = bc_norm
        self.glu_in_loop = glu_in_loop

        self.symm_mixer = symm_mixer
        self.mixer_type = mixer_type
        self.mixer_h_dep = mixer_h_dep
        self.n_backwards = n_backwards
        if isinstance(norm_eps, str):
            norm_eps = float(norm_eps)
        self.norm_eps = norm_eps
        self.use_short_conv = use_short_conv

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        if use_short_conv:
            conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.bc_y_proj = nn.utils.parametrizations.weight_norm(nn.Linear(self.d_ssm, 2 * self.ngroups * self.d_state, bias=False))
        if not glu_in_loop:
            self.z_y_proj = nn.utils.parametrizations.weight_norm(nn.Linear(self.d_inner, self.d_inner, bias=False))

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        if symm_mixer or not mixer_h_dep:
            mixer_input_dim = self.d_ssm
        else:
            mixer_input_dim = 2 * self.d_ssm
        
        # fixed-point mixer
        if mixer_type.lower() == 'householderproduct':
            self.mixer = HouseholderProducts(mixer_input_dim, self.d_ssm, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'dplr':
            self.mixer = DPLR(mixer_input_dim, self.d_ssm, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'kronecker':
            self.mixer = KroneckerProduct(mixer_input_dim, self.d_ssm, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'monarch':
            self.mixer = Monarch(mixer_input_dim, self.d_ssm, mixer_rank, mixer_ac=mixer_ac, proj_rank=mixer_proj_rank, norm_eps=norm_eps)
        elif mixer_type.lower() == 'nonselective':
            self.mixer = NonSelective(self.d_ssm, self.d_ssm, norm_eps=norm_eps)
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
            outlier_quantile=0.1,
            max_iter=max_iter
        )

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True
        
        self.norm = nn.RMSNorm(self.d_ssm, eps=norm_eps)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)
            
        self.fixed_point_iter = []

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        x, dt, A, B, C, z, z0, x0, seqlen_og, d_mlp, dt_limit_kwargs, ssm_state = self._prepare_components(u, seqlen=seqlen,
                                                                                                           seq_idx=seq_idx, 
                                                                                                           cu_seqlens=cu_seqlens, 
                                                                                                           inference_params=inference_params)
        
        y_star = self._fixed_point_iter(x, dt, A, B, C, z,
                                        seq_idx, cu_seqlens, dt_limit_kwargs, ssm_state, inference_params)
        
        y = y_star
        for n in range(self.n_backwards):
            y = self._fixed_point_step(x, y, dt, A, B, C, z,
                                    seq_idx, cu_seqlens, dt_limit_kwargs, ssm_state, inference_params)
        
        # concatenate the ssm section with the non-ssm section
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        
        if not self.glu_in_loop:
            z = rearrange(z, "b d l -> b l d") + self.z_y_proj(shift(y, 1, dim=1))
            out = self.out_proj(self.act(z) * y)
        else:
            out = self.out_proj(y)
        
        return out
    
    @torch.no_grad()
    def _fixed_point_iter(self, x, dt, A, B, C, z, 
                          seq_idx, cu_seqlens, dt_limit_kwargs, ssm_state, inference_params):
        state = self.optimizer.start(torch.zeros_like(x))

        # @torch.compile(fullgraph=True)
        def step(state:Dict[str, torch.Tensor]):
            y = self._fixed_point_step(x, state['y'], dt, A, B, C, z, 
                                       seq_idx, cu_seqlens, dt_limit_kwargs, ssm_state, inference_params)
            return self.optimizer.step(state, y)

        while self.optimizer.cont(state):
            # https://github.com/pytorch/pytorch/issues/97155
            # torch._dynamo.graph_break() # to avoid recompilations?
            state = step(state)
         
        self.fixed_point_iter.append(state['iter_idx'].item() + 1)
        return state['y']

    def _fixed_point_step(self, x, y_prev, dt, A, B, C, z, 
                          seq_idx, cu_seqlens, dt_limit_kwargs, ssm_state, inference_params):
        x_tilde = self._channel_mixing(x, x - y_prev, y_prev) + y_prev
        
        y_shifted = shift(y_prev, 1, dim=1)
        BC_y = self.bc_y_proj(y_shifted)
        B_y, C_y = BC_y[:, :, :self.ngroups * self.d_state], BC_y[:, :, self.ngroups * self.d_state:]

        y = self._sequence_mixing(x_tilde, dt, A, B + B_y, C + C_y, z, seq_idx, cu_seqlens, dt_limit_kwargs, ssm_state, inference_params)
        return y

    def _channel_mixing(self, x, delta_x, y_prev):
        # get previous hidden-states for Q
        y_shifted = shift(y_prev, 1, dim=1)
        if self.mixer_h_dep:
            param_arg = x + y_shifted if self.symm_mixer else torch.cat([x, y_shifted], dim=-1)
        else:
            param_arg = x
        return self.mixer(param_arg=param_arg, input_arg=delta_x)
    
    def _sequence_mixing(self, x, dt, A, B, C, z, seq_idx, cu_seqlens, dt_limit_kwargs, ssm_state, inference_params):
        x = x.to(dtype=dt.dtype)
        B = F.normalize(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups), dim=-1, eps=self.norm_eps).to(dtype=dt.dtype)
        C = F.normalize(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups), dim=-1, eps=self.norm_eps).to(dtype=dt.dtype)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            B,
            C,
            chunk_size=self.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            z=z if self.glu_in_loop else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            **dt_limit_kwargs,
            return_final_states=ssm_state is not None,
            return_varlen_states=cu_seqlens is not None and inference_params is not None,
        )
        if ssm_state is not None:
            y, last_state, *rest = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                ssm_state.copy_(varlen_states)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y * self.act(z))

        return y
    
    def _prepare_components(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )
        if self.use_short_conv:
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2).contiguous(),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        return x, dt, A, B, C, z, z0, x0, seqlen_og, d_mlp, dt_limit_kwargs, ssm_state

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
