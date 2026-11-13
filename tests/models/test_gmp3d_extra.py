"""Supplementary 3D GMP checks, alongside tests/models/test_gmp3d.py (which has the brute-force and voxel-grid references).

What this file adds: exact depth-bin edges (negative depth, bin boundaries, no upper bound); the 2D module's init RNG
stream; constructor validation; fp32 and bf16-autocast agreement of every backend with fp64 production in 3D; and the
3D weights-first backend under torch.compile(fullgraph=True) (CUDA only, run in the PBS job).
"""

from __future__ import annotations

import math
from itertools import starmap

import pytest
import torch

from hepattn.experiments.clic.a4 import GMP_IMPLS
from hepattn.models.encoder import GeometricMessagePassing, gmp_neighbourhood_3d

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
IMPLS = ["production", "einsumopt", "weightsfirst", "weightsfirst_loop"]
PARAMS = ["dw.weight", "dw.bias", "pointwise.weight", "pointwise.bias", "norm.weight", "norm.bias"]
# A small grid so the voxel reference stays small: eta in [-1, 1] / 0.25 -> 9 cells (eta = eta_max is on the grid),
# 8 phi cells, depth bins of 100 (tokens in [0, 600) -> 6 bins).
SMALL = {"kernel_size": 3, "eta_cell": 0.25, "phi_cells": 8, "eta_max": 1.0, "depth_cell": 100.0, "depth_kernel_size": 3}


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def _coords(b, n, seed, *, eta_scale=0.7, depth_max=599.0, colocate=True):
    """Random tokens with padding (NaN), some off-grid (|eta| > eta_max), co-located groups and the phi wrap."""
    g = torch.Generator().manual_seed(seed)
    eta = (torch.rand(b, n, generator=g) * 2 - 1) * eta_scale * 1.6  # some beyond eta_max = 1
    phi = torch.rand(b, n, generator=g) * 2 * math.pi - math.pi
    depth = torch.rand(b, n, generator=g) * depth_max
    if colocate:  # five tokens in one (eta, phi, depth) cell, two more across the phi wrap
        eta[:, 1:6], phi[:, 1:6], depth[:, 1:6] = 0.1, 0.3, 150.0
        phi[:, 6], phi[:, 7] = math.pi - 1e-3, -math.pi + 1e-3
        eta[:, 6:8], depth[:, 6:8] = 0.0, 50.0
    depth[:, 8] = -3.0  # a barycentre a few mm in front of the face: first bin
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[:, -4:] = False
    eta, phi, depth = (torch.where(valid, t, float("nan")) for t in (eta, phi, depth))
    return eta, phi, depth, valid


# ------------------------------------------------------------------------------------------ neighbourhood structure


def test_depth_binning_edges():
    """Bin edges: [0, 100) -> 0, [100, 200) -> 1; negative -> 0; no upper bound (a deep token is still on the grid)."""
    eta = torch.zeros(1, 6)
    phi = torch.zeros(1, 6)
    depth = torch.tensor([[-5.0, 0.0, 99.999, 100.0, 199.0, 1.0e5]])
    valid = torch.ones(1, 6, dtype=torch.bool)
    nb, ing = gmp_neighbourhood_3d(eta, phi, depth, valid, **SMALL)
    assert ing.all()
    a = nb.view(1, 9, 3, 6, 6)[0, 4]  # same (eta, phi) cell; [dw + 1, i, j]
    same_bin = a[1]
    assert same_bin[0, 1] and same_bin[0, 2] and not same_bin[0, 3]  # -5, 0, 99.999 together; 100 is the next bin
    assert a[2, 0, 3] and a[2, 2, 4]  # dw = +1: bin 0 -> bin 1
    assert not a[:, 5, :5].any()  # 1e5 mm is far from everything, but on the grid
    assert same_bin[5, 5]


# ------------------------------------------------------------------------------------------------- the GMP module
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backends_agree_low_precision(dtype, device):
    """fp32: all backends within 2e-5 of fp64 production; bf16 autocast: weights-first no worse than 2x production + 2e-3."""
    eta, phi, depth, valid = (t.to(device) for t in _coords(2, 160, seed=21, colocate=False))
    nb, in_grid = gmp_neighbourhood_3d(eta, phi, depth, valid, **SMALL)
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(2, 160, 16, generator=gen).to(device)
    g = torch.randn(2, 160, 16, generator=gen).to(device)

    def run(impl, dt, amp):
        torch.manual_seed(0)
        mod = GMP_IMPLS[impl](dim=16, **SMALL).to(device=device, dtype=torch.float64 if dt is torch.float64 else torch.float32)
        ctx = torch.autocast(device, dtype=torch.bfloat16) if amp else torch.autocast(device, enabled=False)
        xx = x.to(dt).detach().requires_grad_(True)
        with ctx:
            y = mod(xx, nb.to(dt), in_grid)
        y.backward(g.to(y.dtype))
        named = dict(mod.named_parameters())
        return [y.detach(), xx.grad] + [named[p].grad for p in PARAMS]

    ref = run("production", torch.float64, amp=False)
    amp = dtype is torch.bfloat16
    errs = {impl: list(starmap(_rel, zip(run(impl, torch.float32, amp), ref, strict=True))) for impl in IMPLS}
    print(f"\n[3D {dtype} {device}] max rel err per impl: " + " ".join(f"{k}={max(v):.1e}" for k, v in errs.items()))
    for impl in IMPLS:
        for e_impl, e_prod in zip(errs[impl], errs["production"], strict=True):
            assert e_impl <= (2 * e_prod + 2e-3 if amp else 2e-5)


def test_2d_init_rng_stream_unchanged():
    """The 2D module draws exactly the random numbers it drew before (Conv2d, Linear, LayerNorm in that order)."""
    torch.manual_seed(9)
    mod = GeometricMessagePassing(dim=8)
    torch.manual_seed(9)
    conv = torch.nn.Conv2d(8, 8, 3, groups=8, bias=True)
    lin = torch.nn.Linear(8, 8)
    assert torch.equal(mod.dw.weight, conv.weight) and torch.equal(mod.dw.bias, conv.bias)
    assert torch.equal(mod.pointwise.weight, lin.weight)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"depth_kernel_size": 3}, "needs depth_cell"),
        ({"depth_cell": 0.0}, "depth_cell must be positive"),
        ({"depth_cell": 100.0, "depth_kernel_size": 2}, "odd and positive"),
    ],
)
def test_module_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GeometricMessagePassing(dim=8, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile check runs on the GPU job only")
def test_3d_weightsfirst_compiles_fullgraph():
    eta, phi, depth, valid = (t.cuda() for t in _coords(4, 160, seed=9, colocate=False))
    nb, in_grid = gmp_neighbourhood_3d(eta, phi, depth, valid, **SMALL)
    torch.manual_seed(0)
    mod = GMP_IMPLS["weightsfirst"](dim=16, **SMALL).cuda()
    x = torch.randn(4, 160, 16, device="cuda")
    torch._dynamo.reset()  # noqa: SLF001
    comp = torch.compile(mod, fullgraph=True, dynamic=False)
    want = mod(x, nb.float(), in_grid)
    got = comp(x, nb.float(), in_grid)
    assert _rel(got, want) <= 1e-5


# ---------------------------------------------------------------------------------------------- the depth coordinate
