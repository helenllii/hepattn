"""Correctness of the weights-first GMP backends (Option B) against production (A) and EinsumOpt (A').

Every comparison uses identical inputs, parameters, neighbourhood tensors and upstream gradients. The reference is
production GMP evaluated in float64, so an fp32/bf16 result is judged by its distance from the exact answer, and in
bf16 Option B must be no less accurate than production itself (it is not required to be bit-identical to it: it sums
in a different order).

Tolerances, all relative to max|reference| of the compared tensor:
  * float64: 1e-12   -- round-off only; a real algebra error is O(1).
  * float32: 2e-5    -- ~170 x fp32 epsilon, for sums of up to K*N = 8000 terms followed by a LayerNorm.
  * bf16 autocast: err(impl) <= 2 * err(production) + 2e-3, and <= max(3e-2, err(production)).

Runs on CPU (seconds); the same tests run on CUDA when a GPU is present. The torch.compile test is CUDA-only.
Errors are printed (pytest -s) so the report can quote them.
"""

from __future__ import annotations

import math

import pytest
import torch

from hepattn.experiments.clic.a4 import GMP_IMPLS, PlacementEncoder
from hepattn.experiments.clic.a4.bench_gmp_backends import GMPLayer, make_geometry, proxy_neighbourhood_3d
from hepattn.experiments.clic.a4.gmp_weights_first import (
    GMPWeightsFirst,
    GMPWeightsFirstLoop,
    einsumopt_conv,
    production_conv,
    weights_first_conv,
    weights_first_conv_autograd,
)
from hepattn.models.encoder import GeometricMessagePassing, gmp_neighbourhood

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
GRID = {"eta_cell": 0.2, "phi_cells": 32, "eta_max": 5.2}
IMPLS = ["production", "einsumopt", "weightsfirst", "weightsfirst_loop"]
PARAMS = ["dw.weight", "dw.bias", "pointwise.weight", "pointwise.bias", "norm.weight", "norm.bias"]
TOL = {torch.float64: 1e-12, torch.float32: 2e-5}


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.double() - ref.double()).abs().max() / ref.double().abs().max().clamp_min(1e-30)).item()


def _module(impl: str, kernel_size: int, dim: int, device, dtype, seed: int = 0) -> GeometricMessagePassing:
    torch.manual_seed(seed)
    return GMP_IMPLS[impl](dim=dim, kernel_size=kernel_size, **GRID).to(device=device, dtype=dtype)


def _run(mod, x, neighbours, in_grid, g, amp: bool = False):
    """Forward + backward with the given upstream gradient; returns the output and every gradient."""
    x = x.detach().clone().requires_grad_(True)
    mod.zero_grad(set_to_none=True)
    ctx = torch.autocast(x.device.type, dtype=torch.bfloat16) if amp else torch.autocast(x.device.type, enabled=False)
    with ctx:
        y = mod(x, neighbours, in_grid)
    y.backward(g.to(y.dtype))
    grads = {"x": x.grad}
    named = dict(getattr(mod, "_orig_mod", mod).named_parameters())  # a compiled wrapper prefixes the names
    grads |= {p: named[p].grad for p in PARAMS}
    return y.detach(), grads


def _case(batch, slots, kernel_size, device, dtype, dim=16, seed=0):
    nb, in_grid = make_geometry(batch, slots, kernel_size**2, device, seed=seed)
    gen = torch.Generator().manual_seed(seed + 7)
    x = torch.randn(batch, slots, dim, generator=gen).to(device=device, dtype=dtype)
    g = torch.randn(batch, slots, dim, generator=gen).to(device=device, dtype=dtype)
    return x, nb.to(dtype), in_grid, g


def _reference(kernel_size, dim, device, x, nb, in_grid, g):
    ref = _module("production", kernel_size, dim, device, torch.float64)
    return _run(ref, x.double(), nb.double(), in_grid, g.double())


# ---------------------------------------------------------------------------------- the precondition of "fused"
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("phi_cells", [None, 1, 2])  # None: 32; else kernel_size + this - 1 (the tightest allowed)
def test_neighbourhood_is_symmetric(kernel_size, phi_cells):
    """A[o, i, j] == A[K-1-o, j, i]: what lets "fused" use A^T as a stride view. Includes phi_cells == kernel_size."""
    w = 32 if phi_cells is None else kernel_size + phi_cells - 1
    gen = torch.Generator().manual_seed(3)
    b, n = 3, 40
    eta = torch.randn(b, n, generator=gen) * 2
    phi = torch.rand(b, n, generator=gen) * 2 * math.pi - math.pi
    phi[:, :6] = torch.tensor([math.pi - 1e-4, -math.pi + 1e-4, 0.0, math.pi / 2, -math.pi / 2, 3.0])  # the wrap
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[:, -5:] = False
    eta[~valid] = float("nan")
    nb, _ = gmp_neighbourhood(eta, phi, valid, kernel_size=kernel_size, eta_cell=0.2, phi_cells=w, eta_max=5.2)
    k2 = kernel_size**2
    a = nb.view(b, k2, n, n)
    assert torch.equal(a, a.flip(1).transpose(-1, -2))


def test_proxy_3d_neighbourhood_is_symmetric():
    gen = torch.Generator().manual_seed(4)
    b, n = 2, 48
    eta, phi = torch.randn(b, n, generator=gen), torch.rand(b, n, generator=gen) * math.tau - math.pi
    depth = torch.rand(b, n, generator=gen) * 900
    valid = torch.ones(b, n, dtype=torch.bool)
    nb, _ = proxy_neighbourhood_3d(eta, phi, depth, valid, kernel_size=3, depth_cell=250.0, depth_max=2000.0, **GRID)
    a = nb.view(b, 27, n, n)
    assert torch.equal(a, a.flip(1).transpose(-1, -2))


# -------------------------------------------------------------------- module level: forward, input & param grads
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("slots", [160, 320])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("impl", IMPLS)
def test_matches_fp64_reference(impl, dtype, kernel_size, slots, device):
    x, nb, in_grid, g = _case(2, slots, kernel_size, device, dtype)
    want_y, want_g = _reference(kernel_size, 16, device, x, nb, in_grid, g)
    got_y, got_g = _run(_module(impl, kernel_size, 16, device, dtype), x, nb, in_grid, g)
    errs = {"out": _rel(got_y, want_y)} | {k: _rel(got_g[k], want_g[k]) for k in want_g}
    print(f"\n[{impl} {str(dtype)[6:]} K={kernel_size**2} N={slots} {device}] max rel err " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    bad = {k: v for k, v in errs.items() if not v <= TOL[dtype]}
    assert not bad, bad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("slots", [160, 320])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize(
    "dtypes",
    [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16), (torch.float32, torch.float32)],
    ids=["x_bf16-nb_bf16", "x_fp32-nb_bf16", "x_fp32-nb_fp32"],  # layer 0 / layers >= 1 / fp32 encoder input
)
@pytest.mark.parametrize("impl", ["einsumopt", "weightsfirst", "weightsfirst_loop"])
def test_bf16_autocast_no_less_accurate_than_production(impl, dtypes, kernel_size, slots, device):
    """Training precision: bf16 autocast with fp32 parameters. x/neighbours as the encoder would pass them."""
    x_dtype, nb_dtype = dtypes
    x, nb, in_grid, g = _case(2, slots, kernel_size, device, x_dtype)
    nb = nb.to(nb_dtype)  # 0/1 entries: exact in any dtype
    want_y, want_g = _reference(kernel_size, 16, device, x, nb, in_grid, g)
    prod_y, prod_g = _run(_module("production", kernel_size, 16, device, torch.float32), x, nb, in_grid, g, amp=True)
    got_y, got_g = _run(_module(impl, kernel_size, 16, device, torch.float32), x, nb, in_grid, g, amp=True)
    e_prod = {"out": _rel(prod_y, want_y)} | {k: _rel(prod_g[k], want_g[k]) for k in want_g}
    e_impl = {"out": _rel(got_y, want_y)} | {k: _rel(got_g[k], want_g[k]) for k in want_g}
    print(
        f"\n[{impl} amp x={str(x_dtype)[6:]} nb={str(nb_dtype)[6:]} K={kernel_size**2} N={slots} {device}] rel err impl/prod "
        + " ".join(f"{k}={e_impl[k]:.1e}/{e_prod[k]:.1e}" for k in e_impl)
    )
    assert got_y.isfinite().all()
    # Primary criterion: no less accurate than production (2x its error plus a 2e-3 floor). The sanity cap is 3e-2 or
    # production's own error, whichever is larger: job 7841399 showed production itself at 3.15e-2 on the LayerNorm
    # weight gradient (K=25, N=320, CPU), so a fixed 3e-2 cap failed an implementation that was MORE accurate.
    bad = {k: (e_impl[k], e_prod[k]) for k in e_impl if not (e_impl[k] <= 2 * e_prod[k] + 2e-3 and e_impl[k] <= max(3e-2, e_prod[k]))}
    assert not bad, bad


# ---------------------------------------------------------------------- core level, any K (incl. the 3D proxy)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_offsets", [9, 25, 27])
@pytest.mark.parametrize(
    "core",
    [
        einsumopt_conv,
        weights_first_conv_autograd,
        lambda *a: weights_first_conv(*a, strategy="fused"),
        lambda *a: weights_first_conv(*a, strategy="loop"),
    ],
    ids=["einsumopt", "wf_autograd", "wf_fused", "wf_loop"],
)
def test_cores_match_production_core_fp64(core, num_offsets, device):
    nb, in_grid = make_geometry(2, 160, num_offsets, device, seed=5)
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(2, 160, 8, generator=gen, dtype=torch.float64).to(device)
    w = torch.randn(8, num_offsets, generator=gen, dtype=torch.float64).to(device)
    g = torch.randn(2, 160, 8, generator=gen, dtype=torch.float64).to(device)
    xm = torch.where(in_grid.unsqueeze(-1), x, 0.0)
    out = {}
    for name, fn in (("ref", production_conv), ("got", core)):
        xi, wi = xm.clone().requires_grad_(True), w.clone().requires_grad_(True)
        y = fn(xi, nb.double(), wi)
        y.backward(g)
        out[name] = (y.detach(), xi.grad, wi.grad)
    for a, b, what in zip(out["got"], out["ref"], ("out", "grad_x", "grad_w"), strict=True):
        assert _rel(a, b) <= 1e-12, (what, _rel(a, b))


# ----------------------------------------------------------------------------------- masking and odd geometries
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
def test_padding_garbage_cannot_reach_real_tokens(impl, device):
    """NaN/inf in padded and off-grid slots: they pass through unchanged, and real tokens do not see them."""
    x, nb, in_grid, g = _case(2, 160, 3, device, torch.float64)
    mod = _module(impl, 3, 16, device, torch.float64)
    clean_y, _ = _run(mod, torch.where(in_grid.unsqueeze(-1), x, 0.0), nb, in_grid, g)
    odd = torch.arange(x.shape[1], device=device).unsqueeze(0) % 2 == 1
    dirty = torch.where((~in_grid).unsqueeze(-1), float("nan"), x)
    dirty = torch.where((~in_grid & odd).unsqueeze(-1), float("inf"), dirty)
    assert dirty.isinf().any() and dirty.isnan().any()
    y, grads = _run(mod, dirty, nb, in_grid, g)
    assert torch.equal(y[in_grid], clean_y[in_grid])
    assert torch.equal(y[~in_grid].nan_to_num(), dirty[~in_grid].nan_to_num())  # passed through untouched
    assert all(v.isfinite().all() for k, v in grads.items() if k != "x")
    assert torch.equal(grads["x"][~in_grid], g[~in_grid])  # identity path for off-grid tokens


def _explicit(eta, phi, valid, device):
    nb, in_grid = gmp_neighbourhood(eta.to(device), phi.to(device), valid.to(device), kernel_size=3, **GRID)
    return nb.double(), in_grid


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", ["einsumopt", "weightsfirst", "weightsfirst_loop"])
def test_empty_isolated_and_colocated(impl, device):
    """Event 0: no real token. Event 1: one token. Event 2: isolated tokens (only themselves in reach).
    Event 3: eight tokens in one cell plus one neighbour cell. All agree with production in fp64, and the
    co-located case also with the hand-computed answer.
    """
    n = 16
    eta = torch.full((4, n), float("nan"))
    phi = torch.full((4, n), float("nan"))
    valid = torch.zeros(4, n, dtype=torch.bool)
    valid[1, 0] = True
    eta[1, 0], phi[1, 0] = 0.05, 0.05
    valid[2, :5] = True
    eta[2, :5] = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
    phi[2, :5] = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0])
    valid[3, :9] = True
    eta[3, :8], phi[3, :8] = 0.05, 0.05  # one cell
    eta[3, 8], phi[3, 8] = 0.25, 0.05  # the next cell in eta
    nb, in_grid = _explicit(eta, phi, valid, device)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(4, n, 16, generator=gen, dtype=torch.float64).to(device)
    g = torch.randn(4, n, 16, generator=gen, dtype=torch.float64).to(device)
    want_y, want_g = _run(_module("production", 3, 16, device, torch.float64), x, nb, in_grid, g)
    got_y, got_g = _run(_module(impl, 3, 16, device, torch.float64), x, nb, in_grid, g)
    assert torch.equal(got_y[0], x[0])  # empty event untouched
    assert _rel(got_y, want_y) <= 1e-12
    assert all(_rel(got_g[k], want_g[k]) <= 1e-12 for k in want_g)
    # hand check of the conv for the co-located tokens: centre weight * sum of the 8, plus the (+1, 0) weight * x_8
    mod = _module(impl, 3, 16, device, torch.float64)
    w = mod.dw.weight.view(16, 9)
    xm = torch.where(in_grid.unsqueeze(-1), x, 0.0)
    conv = weights_first_conv(xm, nb, w, strategy=getattr(mod, "strategy", "fused"))[3]
    expect = w[:, 4] * x[3, :8].sum(0) + w[:, 7] * x[3, 8]  # offset (du=+1, dv=0) = (1+1)*3 + (0+1) = 7
    assert torch.allclose(conv[:8], expect.expand(8, 16), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- what autograd keeps for backward
def _saved(layer, x, nb, in_grid, amp=False):
    known = {t.untyped_storage().data_ptr() for t in [x, nb, in_grid, *layer.parameters()]}
    sizes = []

    def pack(t):
        if t.untyped_storage().data_ptr() not in known:
            sizes.append(t.numel())
        return t

    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if amp else torch.autocast("cpu", enabled=False)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), ctx:
        layer(x, nb, in_grid)
    return sizes


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("num_offsets", [9, 27])
def test_saved_activations(num_offsets, amp):
    """What autograd keeps for backward, measured with saved_tensors_hooks (job 7841399 corrected the first guess).

    fp32 neighbours in both cases; under autocast that is the regime where autocast casts A per call:
      * production keeps the K-expanded z (B, K*N, D); under autocast also the bf16 copy of A (B, K*N, N);
      * plain-autograd weights-first keeps NO K-expanded tensor (bmm saves an operand only when its partner needs a
        gradient, and A needs none), but under autocast it keeps the same per-layer bf16 copy of A;
      * the custom Function keeps neither: nothing larger than (B, N, D).
    """
    b, n, d = 2, 32, 8
    nb, in_grid = make_geometry(b, n, num_offsets, "cpu", seed=2)
    x = torch.randn(b, n, d, requires_grad=True)
    nbf = nb.float()
    k_expanded, nb_sized, token_sized = b * num_offsets * n * d, b * num_offsets * n * n, b * n * d
    cores = {
        "production": production_conv,
        "wf_autograd": weights_first_conv_autograd,
        "wf_fused": lambda *a: weights_first_conv(*a, strategy="fused"),
        "wf_loop": lambda *a: weights_first_conv(*a, strategy="loop"),
    }
    sizes = {name: _saved(GMPLayer(d, num_offsets, core), x, nbf, in_grid, amp) for name, core in cores.items()}
    facts = {name: {"k_expanded": k_expanded in v, "neighbours_copy": nb_sized in v, "largest": max(v)} for name, v in sizes.items()}
    print(f"\n[saved K={num_offsets} amp={amp}] {facts}; B*N*D={token_sized} B*K*N*D={k_expanded} B*K*N*N={nb_sized}")
    assert facts["production"]["k_expanded"]
    assert facts["production"]["neighbours_copy"] == amp
    assert not facts["wf_autograd"]["k_expanded"]
    assert facts["wf_autograd"]["neighbours_copy"] == amp
    for name in ("wf_fused", "wf_loop"):
        assert facts[name]["largest"] <= token_sized, (name, facts[name])


# ---------------------------------------------------------------------------- integration with the a4 machinery
def test_backends_registered_and_stateless():
    assert GMP_IMPLS["weightsfirst"] is GMPWeightsFirst
    assert GMP_IMPLS["weightsfirst_loop"] is GMPWeightsFirstLoop
    for cls in (GMPWeightsFirst, GMPWeightsFirstLoop):
        assert issubclass(cls, GeometricMessagePassing)
        assert "__init__" not in cls.__dict__
    assert set(GMPWeightsFirst.__dict__) <= {"__doc__", "__module__", "__qualname__", "forward", "strategy"}
    assert set(GMPWeightsFirstLoop.__dict__) <= {"__doc__", "__module__", "__qualname__", "strategy"}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", ["weightsfirst", "weightsfirst_loop"])
def test_whole_encoder_all_layer_gmp_fp64(impl, device):
    """All six patch blocks with their own GMP (gmp_layers=None), production vs the new backend, fp64, with grads."""
    kw = {"num_layers": 6, "dim": 16, "attn_type": "patch", "patch_size": 4, "attn_kwargs": {"num_heads": 2}, "gmp": {"kernel_size": 3, **GRID}}
    torch.manual_seed(11)
    prod = PlacementEncoder(gmp_layers=None, gmp_impl="production", **kw).double().to(device)
    torch.manual_seed(11)
    new = PlacementEncoder(gmp_layers=None, gmp_impl=impl, **kw).double().to(device)
    assert all(type(lyr.gmp) is GMP_IMPLS[impl] for lyr in new.layers)
    gen = torch.Generator().manual_seed(2)
    b, s = 2, 16
    x = torch.randn(b, s, 16, generator=gen, dtype=torch.float64).to(device)
    coords = torch.stack([torch.rand(b, s, generator=gen) * 8 - 4, torch.rand(b, s, generator=gen) * math.tau - math.pi], -1)
    coords = coords.double().to(device)
    kv_mask = torch.ones(b, s, dtype=torch.bool, device=device)
    kv_mask[:, -2:] = False
    outs = []
    for enc in (prod, new):
        xi = x.clone().requires_grad_(True)
        y = enc(xi, kv_mask=kv_mask, x_coords=coords)
        y.square().sum().backward()
        outs.append((y.detach(), xi.grad, [p.grad for p in enc.parameters()]))
    assert _rel(outs[1][0], outs[0][0]) <= 1e-10
    assert _rel(outs[1][1], outs[0][1]) <= 1e-10
    for gp, gn in zip(outs[0][2], outs[1][2], strict=True):
        if gp is not None:
            assert _rel(gn, gp) <= 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile check runs on the GPU job only")
@pytest.mark.parametrize("impl", ["weightsfirst", "weightsfirst_loop"])
def test_compiles_fullgraph_and_matches_eager(impl):
    """The custom autograd.Function is traced by dynamo without a graph break, and compiled == eager (fp32)."""
    x, nb, in_grid, g = _case(4, 160, 3, "cuda", torch.float32)
    mod = _module(impl, 3, 16, "cuda", torch.float32)
    torch._dynamo.reset()  # noqa: SLF001
    compiled = torch.compile(mod, fullgraph=True, dynamic=False)
    eager_y, eager_g = _run(mod, x, nb, in_grid, g)
    comp_y, comp_g = _run(compiled, x, nb, in_grid, g)
    assert _rel(comp_y, eager_y) <= 1e-5
    assert all(_rel(comp_g[k], eager_g[k]) <= 1e-5 for k in eager_g)
