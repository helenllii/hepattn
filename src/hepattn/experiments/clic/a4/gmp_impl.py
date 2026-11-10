"""An optimized drop-in for the production GMP depthwise contraction.

WHAT CHANGES, AND WHAT DOES NOT
`GeometricMessagePassing.forward` is two lines of arithmetic:

    z = bmm(neighbours, masked_x).view(B, k*k, N, D)                  # aggregation
    z = einsum("bond,do->bnd", z, W.view(D, k*k)) + bias              # depthwise contraction

Only the second line is rewritten here:

    einsum("bond,do->bnd", z, w)   ->   (z.permute(0, 2, 3, 1) * w).sum(-1)

The einsum dispatches to a permute-and-copy of the entire (B, k*k*N, D) tensor before reducing;
the broadcast form reduces without materializing it. Everything else -- the neighbour matrix, the
bmm, the off-grid masking, the pointwise Linear, the LayerNorm, the residual -- is inherited from
production untouched.

WHY A SUBCLASS, NOT A SEPARATE MODULE
This class adds no state and overrides no `__init__`, so:

  * parameter creation order, shapes and dtypes are production's, hence the same `torch.manual_seed`
    produces bit-identical initial weights;
  * `state_dict()` keys are production's, so a checkpoint is interchangeable with a production one;
  * no extra module is constructed, so the global RNG stream is not advanced and every later
    random initialization (decoder, task heads) matches production as well.

That is what makes `layer.gmp.__class__ = GMPEinsumOpt` (see `encoder.PlacementEncoder`) a change
of forward arithmetic and nothing else.

NUMERICAL STATUS
Mathematically identical; floating-point results differ because the two forms reduce over the
k*k offset axis in a different order. Measured against production at tensor scale on A100:

    fp32  forward  <= 1.2e-07     gradients <= 2.1e-07
    bf16  forward  4.6e-03 .. 9.4e-03 across a (N, D) sweep

It is NOT bit-equivalent to production and is not intended to be. `tests/models/test_a4_placement.py`
pins the fp64 agreement, where the difference is at the 1e-15 level and the two forms are
distinguishable from an actual algebra error.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hepattn.models.encoder import GeometricMessagePassing


class GMPEinsumOpt(GeometricMessagePassing):
    """Production GMP with the depthwise contraction done as a broadcast-multiply-reduce."""

    def forward(self, x: Tensor, neighbours: Tensor, in_grid: Tensor) -> Tensor:
        batch_size, num_tokens, dim = x.shape
        num_offsets = self.kernel_size**2
        # Identical to production up to here, including the off-grid zeroing that stops a padded
        # inf/NaN reaching other tokens through 0 * inf.
        z = torch.bmm(neighbours, torch.where(in_grid.unsqueeze(-1), x, 0.0)).view(batch_size, num_offsets, num_tokens, dim)
        z = (z.permute(0, 2, 3, 1) * self.dw.weight.view(dim, num_offsets)).sum(-1) + self.dw.bias
        return torch.where(in_grid.unsqueeze(-1), x + self.norm(self.pointwise(z)), x)


# Both entries take production's (x, neighbours, in_grid) geometry, so `PlacementEncoder` never
# has to touch `Encoder._patch_forward`.
GMP_IMPLS: dict[str, type[GeometricMessagePassing]] = {
    "production": GeometricMessagePassing,
    "einsumopt": GMPEinsumOpt,
}
