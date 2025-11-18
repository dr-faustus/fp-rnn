# Compilable version of the selective scan interface
#  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py

from typing import Optional, List
import torch
from torch.autograd.function import FunctionCtx

from einops import rearrange

import mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_cuda

#################################################################################################################################
################# CPP Wrappers
#################################################################################################################################

# Wrap selective_scan_cuda.fwd:
# https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/csrc/selective_scan/selective_scan.cpp#L227
@torch.library.custom_op("mambacuda::selective_scan_fwd", mutates_args=[], device_types="cuda")
def selective_scan_fwd(    
    u:torch.Tensor,
    delta:torch.Tensor,
    A:torch.Tensor,
    B:torch.Tensor,
    C:torch.Tensor,
    D:Optional[torch.Tensor],
    z:Optional[torch.Tensor],
    delta_bias:Optional[torch.Tensor],
    delta_softplus:bool
) -> List[torch.Tensor]: # List[Optional[torch.Tensor]] not supported
    
    result = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus) 
    return [torch.tensor(float('nan')) if t is None else t for t in result] # cast to List[torch.tensor] 

@selective_scan_fwd.register_fake
def fake_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
    batch_size, dim, seqlen, dstate = *u.shape, A.shape[1]
    n_chunks = (seqlen + 2048 - 1) // 2048

    out = torch.empty_like(delta) 
    # .cpp has x.shape = (batch_size, dim, n_chunks, dstate * 2) but this might be a typo?
    x = torch.empty((batch_size, dim, n_chunks, dstate * 2), dtype=A.dtype, layout=u.layout, device=u.device)
    result = [out, x]

    if z is not None:
        out_z = torch.empty_like(z)
        result.append(out_z)

    return [torch.tensor(float('nan')) if t is None else t for t in result] # cast to List[torch.tensor] 

# Wrap selective_scan_cuda.bwd:
# https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/csrc/selective_scan/selective_scan.cpp#L339
# Setting mutates_args=["dz"] gives the following RuntimeError: 
# Found a custom (non-ATen) operator whose output has alias annotations. We only support functionalizing operators whose outputs 
# do not have alias annotations (e.g. 'Tensor(a)' is a Tensor with an alias annotation whereas 'Tensor' is a Tensor without. 
# The '(a)' is the alias annotation). The alias annotation specifies that the output Tensor shares storage with an input that 
# has the same annotation. Please check if 
# (1) the output needs to be an output (if not, don't return it), 
# (2) if the output doesn't share storage with any inputs, then delete the alias annotation. 
# (3) if the output indeed shares storage with an input, then add a .clone() before returning it to prevent storage sharing and 
# then delete the alias annotation. Otherwise, please file an issue on GitHub.
@torch.library.custom_op("mambacuda::selective_scan_bwd", mutates_args=[], device_types="cuda")
def selective_scan_bwd(    
    u:torch.Tensor,
    delta:torch.Tensor,
    A:torch.Tensor,
    B:torch.Tensor,
    C:torch.Tensor,
    D:Optional[torch.Tensor],
    z:Optional[torch.Tensor],
    delta_bias:Optional[torch.Tensor],
    dout:torch.Tensor,
    x:Optional[torch.Tensor],
    out:Optional[torch.Tensor],
    dz:Optional[torch.Tensor],
    delta_softplus:bool,
    recompute_out_z:bool
) -> List[torch.Tensor]: # List[Optional[torch.Tensor]] not supported
    result = selective_scan_cuda.bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz, delta_softplus, recompute_out_z)
    return [torch.tensor(float('nan')) if t is None else t for t in result] # cast to List[torch.tensor] 

@selective_scan_bwd.register_fake
def fake_bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz, delta_softplus, recompute_out_z):
    du = torch.empty_like(u)
    ddelta = torch.empty_like(delta)
    dA = torch.empty_like(A)
    dB = torch.empty_like(B)
    dC = torch.empty_like(C)
    dD = None if D is None else torch.empty_like(D)
    ddelta_bias = None if delta_bias is None else torch.empty_like(delta_bias)
    result = [du, ddelta, dA, dB, dC, dD, ddelta_bias]

    if z is not None:
        if dz is None:
            dz = torch.empty_like(z)

        if recompute_out_z:
            out_z = torch.empty_like(out)

    if z is not None:
        result.append(out_z)

    if recompute_out_z:
        result.append(dz)
   
    return [torch.tensor(float('nan')) if t is None else t for t in result] # cast to List[torch.tensor] 


#################################################################################################################################
################# selective_scan_fn via autograd.Function (deprecated)
#################################################################################################################################

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


if __name__ == '__main__':
    from torch import nn

    # model params from
    # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    d_model = 1024                          # W: width of u (default: d_model=1024 in mamba1-370m)
    expand  = 2                             # expansion from d_state to d_inner
    d_inner = expand * d_model              # D: width of x (default: expand=2 => d_inner=2048)
    d_state = 16                            # N: width of one SSM-head  (default: d_state=16)
    ngroups = 1                             # G: number heads that share B and C projection vectors
    assert(d_inner % ngroups == 0)

    # prepare dummy data according to
    # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

    device = torch.device('cuda')
    dtype = torch.float32
    batchsize, seqlen = 1, 2**10

    # A is only input independent param
    Abase = torch.rand((d_inner, d_state), device=device, dtype=dtype, requires_grad=True) * 15 + 1
    A = -torch.exp(Abase.log().float()) # for completeness
    in_proj = nn.Linear(d_model, d_inner + d_inner + ngroups*d_state + ngroups*d_state + d_inner, device=device, dtype=dtype)

    # prepare input u and input-dependent params
    x = torch.randn((batchsize, seqlen, d_model), device=device, dtype=dtype, requires_grad=True)
    _, u, B, C, dt = torch.split(in_proj(x), [d_inner, d_inner, ngroups*d_state, ngroups*d_state, d_inner], dim=-1)
    B = rearrange(B, 'B L (G N) -> B G N L', G=ngroups, N=d_state).contiguous()
    C = rearrange(C, 'B L (G N) -> B G N L', G=ngroups, N=d_state).contiguous()
    u = rearrange(u, 'B L D -> B D L').contiguous()
    dt = rearrange(dt, 'B L D -> B D L').contiguous()
    dt = nn.functional.softplus(dt) # map to positive range

    selective_scan_fn = torch.compile(selective_scan_fn)
    out = selective_scan_fn(u=u, delta=dt, A=A, B=B, C=C, D=None, z=None, delta_bias=None, delta_softplus=False)
    dout = torch.rand_like(out)
    grad = torch.autograd.grad(out, (in_proj.weight, in_proj.bias, Abase, x), dout, retain_graph=True)

    out = mamba_ssm.selective_scan_fn(u=u, delta=dt, A=A, B=B, C=C, D=None, z=None, delta_bias=None, delta_softplus=False)
    sol = torch.autograd.grad(out, (in_proj.weight, in_proj.bias, Abase, x), dout)
    errors = [torch.abs(g-s).max() for (g,s) in zip(grad, sol)]
    print(f'torch.autograd.Functon works with errors: {errors}')



#################################################################################################################################
################# selective_scan_fn via torch.library.register_autograd (recommended)
#################################################################################################################################

    # Requirements for setup_context signature are descriped here:
    # https://pytorch.org/docs/stable/library.html#torch.library.register_autograd
    def setup_context(ctx:FunctionCtx, inputs, output):
        u, delta, A, B, C, D, z, delta_bias, delta_softplus = inputs # named argument must be 'inputs'
        out, x, *rest = output                                       # named argument must be 'output'
        if B.dim() == 3:
            ctx.squeeze_B = True
        if C.dim() == 3:
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)    

    def backward(ctx:FunctionCtx, grads:List[torch.Tensor]):
        dout, *rest = grads
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, *rest = selective_scan_bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, False, False)  # option to recompute out_z, not used here 
        dz = rest[-1] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD if D is not None else None, dz,
                rest[-2] if delta_bias is not None else None, None)
    torch.library.register_autograd("mambacuda::selective_scan_fwd", backward, setup_context=setup_context)


    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
        
        out, x, *rest = selective_scan_fwd(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus)
        last_state = x[:, :, -1, 1::2]
        if z is None:
            return out if not return_last_state else (out, last_state)
        else:
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)
        

    torch.library.opcheck(selective_scan_fwd, 
                            args=(u, dt, A, B, C, None, None, None, False), 
                            test_utils=['test_schema',
                                        'test_autograd_registration', 
                                        'test_faketensor', 
                                        #'test_aot_dispatch_static',
                                        'test_aot_dispatch_dynamic',
                                        ])
    
    print('torch.library.register_autograd works, but opcheck fails because gradients wrt x are ignored.')