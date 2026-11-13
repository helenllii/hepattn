"""Weights-first GMP ("Option B"): the depthwise weights are applied before the neighbour aggregation.

WHAT IS COMPUTED
Production `GeometricMessagePassing.forward` evaluates the depthwise convolution over the occupied cells as

    z    = bmm(neighbours, x_masked)              # (B, K*N, D): one aggregated copy of the tokens per kernel offset
    conv = einsum("bond,do->bnd", z, w) + bias    # y_i = sum_o w_o * (sum_j A[o, i, j] x_j)

Option B evaluates the same sum in the other order,

    y_i = sum_o sum_j A[o, i, j] (w_o * x_j)

so the only (B, K*N, D)-sized tensor is the weighted input, and the result comes out of the matmul already reduced
over the offsets. Neighbourhood construction, parameters (`dw`, `pointwise`, `norm`), masking and the residual are
production's, unchanged: the backends below are strict subclasses of `GeometricMessagePassing` that override only
`forward`, exactly like `GMPEinsumOpt`, so `PlacementEncoder(gmp_impl=...)` can retype to them.

WHY A CUSTOM AUTOGRAD FUNCTION (measured, job 7841399, tests/models/test_gmp_weights_first.py::test_saved_activations)
The reordering alone already avoids saving the K-expanded tensor: bmm's backward saves an operand only when the
other operand needs a gradient, and A (no grad) means x_w is never saved. So the plain-autograd version
(`weights_first_conv_autograd`, kept as the measured reference) saves nothing K-expanded in fp32 -- contrary to the
first guess. What it does keep, under autocast with an fp32 A, is autocast's per-call bf16 copy of A (B, K*N, N),
once per layer; production keeps the same copy, plus z. `_WeightsFirstConv` saves only the masked input (B, N, D),
the (D, K) weights and the neighbours tensor exactly as passed in (which the encoder holds for all layers anyway), and
rebuilds everything else in backward. It also fixes the compute dtype explicitly instead of relying on autocast:

    g_xw = bmm(A, g)                    # (B, K*N, D), transient in backward
    g_x  = sum_o w_o * g_xw[o]          # input gradient
    g_w  = sum_{b, i} g_xw[o] * x       # weight gradient, from the same g_xw: no recomputation of the forward

The production order cannot do that: its weight gradient needs z = bmm(A, x) itself, so dropping z costs a second
bmm in backward.

TWO STRATEGIES (same arithmetic, different memory/launch trade-off)
* "fused": one bmm over the K*N contraction. Uses the neighbourhood symmetry A[o, i, j] == A[K-1-o, j, i] (offset o
  from i to j is offset -o from j to i; production's grid is symmetric in every axis, including the periodic phi
  axis because phi_cells >= kernel_size) so that A^T is a free stride view of the stored (B, K*N, N) tensor, with the
  weights taken in reversed offset order. Transient (B, K*N, D) tensors in forward and backward; nothing K-expanded
  is saved.
* "loop": K bmms of (B, N, N) x (B, N, D), one per offset, accumulated in fp32. No K-expanded tensor at all, even
  transiently, and no symmetry assumption, at the price of K smaller matmuls per direction.

PRECISION
Computation runs in the autocast dtype when autocast is on (as production's bmm would), else in the input dtype.
Accumulation over the offsets is at least fp32 in both strategies (inside the matmul for "fused", an explicit
accumulator for "loop"); the reductions in backward likewise. fp32 is used only to widen 16-bit compute: an fp64
computation accumulates in fp64 (an earlier version hard-coded fp32 and lost fp64 precision; the fp64 tests caught
it). The neighbours tensor is saved as given and recast in backward, so an fp32 neighbours tensor does not get a
second, per-layer bf16 copy kept alive.

NUMERICAL STATUS: mathematically identical to production, not bit-identical (different summation order). See
tests/models/test_gmp_weights_first.py and REPORT_gmp_weights_first.md for the measured errors.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hepattn.models.encoder import GeometricMessagePassing

STRATEGIES = ("fused", "loop")


def compute_dtype_for(x: Tensor) -> torch.dtype:
    """The dtype production's bmm would run in: the autocast dtype if autocast is on for x's device, else x's."""
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


def accumulate_dtype_for(compute_dtype: torch.dtype) -> torch.dtype:
    """fp32 accumulation for the 16-bit compute dtypes; otherwise the compute dtype itself (never a downcast of fp64)."""
    return torch.float32 if compute_dtype in {torch.bfloat16, torch.float16} else compute_dtype


class _WeightsFirstConv(torch.autograd.Function):
    """y = sum_o A_o (w_o * x), saving only (x, A, w). See the module docstring."""

    @staticmethod
    def forward(ctx, xm: Tensor, neighbours: Tensor, weight: Tensor, compute_dtype: torch.dtype, strategy: str) -> Tensor:
        batch_size, num_tokens, dim = xm.shape
        num_offsets = weight.shape[1]
        xc = xm.to(compute_dtype)
        wc = weight.to(compute_dtype).t()  # (K, D)
        nb = neighbours.to(compute_dtype)
        if strategy == "fused":
            # A^T[b, j, o*N + i] = A[b, o*N + i, j] = A[b, (K-1-o)*N + j, i]  -> weights in reversed offset order
            xw = (xc.unsqueeze(1) * wc.flip(0)[None, :, None, :]).reshape(batch_size, num_offsets * num_tokens, dim)
            out = torch.bmm(nb.transpose(1, 2), xw)
        elif strategy == "loop":
            nb4 = nb.view(batch_size, num_offsets, num_tokens, num_tokens)
            acc = torch.zeros(batch_size, num_tokens, dim, device=xm.device, dtype=accumulate_dtype_for(compute_dtype))
            for o in range(num_offsets):
                acc = acc + torch.bmm(nb4[:, o], xc * wc[o])
            out = acc.to(compute_dtype)
        else:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        # neighbours is saved as passed in (the encoder keeps it alive for every layer anyway), not as the cast copy
        ctx.save_for_backward(xc, neighbours, wc)
        ctx.strategy = strategy
        ctx.x_dtype = xm.dtype
        ctx.w_dtype = weight.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        xc, neighbours, wc = ctx.saved_tensors
        need_x, _, need_w = ctx.needs_input_grad[:3]
        if not (need_x or need_w):
            return None, None, None, None, None
        batch_size, num_tokens, dim = xc.shape
        num_offsets = wc.shape[0]
        nb = neighbours.to(xc.dtype)
        g = grad_out.to(xc.dtype)
        acc_dtype = accumulate_dtype_for(xc.dtype)
        grad_x = grad_w = None
        if ctx.strategy == "fused":
            gxw = torch.bmm(nb, g).view(batch_size, num_offsets, num_tokens, dim)  # grad of xw, reversed offsets
            if need_x:
                grad_x = (gxw * wc.flip(0)[None, :, None, :]).sum(1, dtype=acc_dtype)
            if need_w:
                grad_w = (gxw * xc.unsqueeze(1)).sum((0, 2), dtype=acc_dtype).flip(0)  # (K, D)
        else:
            nb4 = nb.view(batch_size, num_offsets, num_tokens, num_tokens)
            grad_x = torch.zeros(batch_size, num_tokens, dim, device=xc.device, dtype=acc_dtype) if need_x else None
            rows = []
            for o in range(num_offsets):
                t = torch.bmm(nb4[:, o].transpose(1, 2), g)  # (B, N, D)
                if need_x:
                    grad_x = grad_x + t * wc[o]
                if need_w:
                    rows.append((t * xc).sum((0, 1), dtype=acc_dtype))
            grad_w = torch.stack(rows) if need_w else None
        return (
            grad_x.to(ctx.x_dtype) if grad_x is not None else None,
            None,
            grad_w.t().to(ctx.w_dtype) if grad_w is not None else None,
            None,
            None,
        )


def weights_first_conv(xm: Tensor, neighbours: Tensor, weight: Tensor, strategy: str = "fused") -> Tensor:
    """sum_o A_o (w_o * xm), without the bias. xm (B, N, D) masked tokens, neighbours (B, K*N, N), weight (D, K)."""
    return _WeightsFirstConv.apply(xm, neighbours, weight, compute_dtype_for(xm), strategy)


# ------------------------------------------------------------------------------------- reference cores (not backends)
# The same three lines of arithmetic as the production / EinsumOpt forwards, for an arbitrary number of offsets K, so
# the tests and the micro-benchmark can use K = 27 (a 3x3x3 proxy) without touching the production class.
def production_conv(xm: Tensor, neighbours: Tensor, weight: Tensor) -> Tensor:
    batch_size, num_tokens, dim = xm.shape
    z = torch.bmm(neighbours, xm).view(batch_size, weight.shape[1], num_tokens, dim)
    return torch.einsum("bond,do->bnd", z, weight)


def einsumopt_conv(xm: Tensor, neighbours: Tensor, weight: Tensor) -> Tensor:
    batch_size, num_tokens, dim = xm.shape
    z = torch.bmm(neighbours, xm).view(batch_size, weight.shape[1], num_tokens, dim)
    return (z.permute(0, 2, 3, 1) * weight).sum(-1)


def weights_first_conv_autograd(xm: Tensor, neighbours: Tensor, weight: Tensor) -> Tensor:
    """Option B written naively with plain autograd: the counter-example that keeps the K-expanded tensor."""
    batch_size, num_tokens, dim = xm.shape
    num_offsets = weight.shape[1]
    xw = (xm.unsqueeze(1) * weight.t().flip(0)[None, :, None, :]).reshape(batch_size, num_offsets * num_tokens, dim)
    return torch.bmm(neighbours.transpose(1, 2), xw)


# ------------------------------------------------------------------------------------------------------------ backends
class GMPWeightsFirst(GeometricMessagePassing):
    """Production GMP with the depthwise convolution evaluated weights-first by `_WeightsFirstConv` ("fused")."""

    strategy = "fused"  # a class attribute, not module state: retyping adds no parameter, buffer or key

    def forward(self, x: Tensor, neighbours: Tensor, in_grid: Tensor) -> Tensor:
        dim = x.shape[-1]
        # Identical masking to production: off-grid tokens are zeroed, so a padded inf/NaN cannot reach the others
        xm = torch.where(in_grid.unsqueeze(-1), x, 0.0)
        z = weights_first_conv(xm, neighbours, self.dw.weight.view(dim, self.num_offsets), self.strategy) + self.dw.bias
        return torch.where(in_grid.unsqueeze(-1), x + self.norm(self.pointwise(z)), x)


class GMPWeightsFirstLoop(GMPWeightsFirst):
    """As GMPWeightsFirst, one (B, N, N) matmul per kernel offset: no K-expanded tensor even transiently."""

    strategy = "loop"
