"""Correctness gates for the A4 PlacementEncoder and the EinsumOpt GMP implementation.

These are the claims the benchmark rests on. If any of them is false, a speed number measured
with `gmp_impl: einsumopt` is a number for a different model:

  1. GMPEinsumOpt computes the production function. Checked in float64, where the two reduction
     orders agree to ~1e-15 and a genuine algebra error cannot hide inside the tolerance.
  2. Retyping adds and removes nothing: same parameters, same `state_dict` keys, bit-identical
     initialization under one seed. This is what makes the comparison controlled, and what lets a
     checkpoint move between the two.
  3. `gmp_layers` really deletes the other layers' GMP modules -- they leave `state_dict()` and
     `parameters()`, so they leave the optimizer -- rather than merely bypassing them.
  4. Off-grid and padded tokens pass through untouched, so a padded NaN cannot reach a real token.

All of it runs on CPU in a second; the CUDA path is exercised by the same assertions whenever a
GPU is present.
"""

from __future__ import annotations

import math

import pytest
import torch

from hepattn.experiments.clic.a4 import GMP_IMPLS, GMPEinsumOpt, PlacementEncoder
from hepattn.models.encoder import Encoder, GeometricMessagePassing, gmp_neighbourhood

GEOM = {"kernel_size": 3, "eta_cell": 0.2, "phi_cells": 32, "eta_max": 5.2}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _geometry(batch: int, tokens: int, device: str, dtype: torch.dtype, seed: int = 0):
    """A realistic (neighbours, in_grid, x) triple, including padding and off-grid tokens."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eta = (torch.rand(batch, tokens, generator=gen) * 12.0 - 6.0).to(device)  # some |eta| > eta_max
    phi = (torch.rand(batch, tokens, generator=gen) * 2 * math.pi - math.pi).to(device)
    valid = torch.ones(batch, tokens, dtype=torch.bool, device=device)
    valid[:, -3:] = False  # padding at the end, as the encoder's stable compaction produces
    eta[~valid] = float("nan")  # a padded NaN must not reach a real token
    neighbours, in_grid = gmp_neighbourhood(eta, phi, valid, **GEOM)
    x = torch.randn(batch, tokens, 8, generator=gen).to(device=device, dtype=dtype)
    return neighbours.to(dtype), in_grid, x


# ----------------------------------------------------------------- 1. same function as production
@pytest.mark.parametrize("device", DEVICES)
def test_einsumopt_matches_production_fp64(device):
    """The rewritten contraction is the same arithmetic, to float64 round-off."""
    torch.manual_seed(42)
    prod = GeometricMessagePassing(dim=8, **GEOM).to(device=device, dtype=torch.float64)
    opt = GMPEinsumOpt(dim=8, **GEOM).to(device=device, dtype=torch.float64)
    opt.load_state_dict(prod.state_dict())

    neighbours, in_grid, x = _geometry(4, 24, device, torch.float64)
    with torch.no_grad():
        want, got = prod(x, neighbours, in_grid), opt(x, neighbours, in_grid)
    assert torch.allclose(want, got, rtol=0, atol=1e-12), (want - got).abs().max().item()


@pytest.mark.parametrize("device", DEVICES)
def test_einsumopt_matches_production_gradients(device):
    """Equal values are not enough: the backward must match too, or training diverges."""
    torch.manual_seed(7)
    prod = GeometricMessagePassing(dim=8, **GEOM).to(device=device, dtype=torch.float64)
    opt = GMPEinsumOpt(dim=8, **GEOM).to(device=device, dtype=torch.float64)
    opt.load_state_dict(prod.state_dict())

    neighbours, in_grid, x = _geometry(3, 20, device, torch.float64)
    grads = []
    for mod in (prod, opt):
        xi = x.clone().requires_grad_(True)
        mod.zero_grad()
        mod(xi, neighbours, in_grid).square().sum().backward()
        grads.append((xi.grad, {k: p.grad.clone() for k, p in mod.named_parameters()}))

    assert torch.allclose(grads[0][0], grads[1][0], rtol=0, atol=1e-10)
    for key in grads[0][1]:
        a, b = grads[0][1][key], grads[1][1][key]
        assert torch.allclose(a, b, rtol=0, atol=1e-10), f"{key}: {(a - b).abs().max().item()}"


# -------------------------------------------------------- 4. padding and off-grid tokens are inert
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", sorted(GMP_IMPLS))
def test_off_grid_tokens_pass_through_unchanged(device, impl):
    """A token off the grid (padded, non-finite, or |eta| > eta_max) must come out as it went in."""
    torch.manual_seed(3)
    mod = GMP_IMPLS[impl](dim=8, **GEOM).to(device=device, dtype=torch.float64)
    neighbours, in_grid, x = _geometry(4, 24, device, torch.float64)
    assert (~in_grid).any(), "this fixture is meant to contain off-grid tokens"
    with torch.no_grad():
        out = mod(x, neighbours, in_grid)
    assert torch.equal(out[~in_grid], x[~in_grid])
    assert torch.isfinite(out[in_grid]).all(), "a padded NaN reached a real token"


# ------------------------------------------------- 2. retyping changes arithmetic and nothing else
def test_retype_preserves_parameters_and_keys():
    torch.manual_seed(1234)
    mod = GeometricMessagePassing(dim=16, **GEOM)
    before_keys = list(mod.state_dict().keys())
    before = {k: v.clone() for k, v in mod.state_dict().items()}

    mod.__class__ = GMPEinsumOpt

    assert list(mod.state_dict().keys()) == before_keys
    for k, v in mod.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert isinstance(mod, GeometricMessagePassing)


def test_einsumopt_declares_no_init_and_no_new_state():
    """The property the whole design depends on: a strict subclass with no state of its own."""
    assert "__init__" not in GMPEinsumOpt.__dict__
    assert set(GMPEinsumOpt.__dict__) <= {"__doc__", "__module__", "__qualname__", "forward"}
    assert issubclass(GMPEinsumOpt, GeometricMessagePassing)


@pytest.mark.parametrize("impl", sorted(GMP_IMPLS))
def test_init_is_bit_identical_across_impls(impl):
    """One seed, two gmp_impl values -> identical initial weights, so only the forward differs."""
    torch.manual_seed(99)
    ref = _encoder(gmp_layers=[0], gmp_impl="production").state_dict()
    torch.manual_seed(99)
    got = _encoder(gmp_layers=[0], gmp_impl=impl).state_dict()
    assert list(ref) == list(got)
    for k in ref:
        assert torch.equal(ref[k], got[k]), k


# -------------------------------------------------------------- 3. gmp_layers deletes, not bypasses
def _encoder(gmp_layers=None, gmp_impl="production", num_layers=6, dim=16):
    return PlacementEncoder(
        gmp_layers=gmp_layers,
        gmp_impl=gmp_impl,
        num_layers=num_layers,
        dim=dim,
        attn_type="patch",
        patch_size=4,
        gmp=dict(GEOM),
        attn_kwargs={"num_heads": 2},
    )


def test_gmp_layers_removes_the_other_layers_modules():
    enc = _encoder(gmp_layers=[0])
    assert [i for i, lyr in enumerate(enc.layers) if lyr.gmp is not None] == [0]
    gmp_keys = [k for k in enc.state_dict() if ".gmp." in k]
    assert gmp_keys, "expected layer 0 to keep its GMP parameters"
    assert all(k.startswith("layers.0.gmp.") for k in gmp_keys), sorted({k.split(".gmp.")[0] for k in gmp_keys})


def test_gmp_layers_removes_parameters_from_the_optimizer_view():
    """Dropped GMP parameters must leave parameters(), or the optimizer would still carry them."""
    full = _encoder(gmp_layers=None)
    one = _encoder(gmp_layers=[0])
    per_layer_gmp = sum(p.numel() for n, p in full.named_parameters() if n.startswith("layers.0.gmp."))
    assert per_layer_gmp > 0
    dropped = sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in one.parameters())
    assert dropped == 5 * per_layer_gmp, (dropped, per_layer_gmp)


def test_gmp_layers_none_is_the_production_encoder():
    """The default must be parameter-for-parameter the unmodified production architecture."""
    torch.manual_seed(5)
    prod = Encoder(num_layers=6, dim=16, attn_type="patch", patch_size=4, gmp=dict(GEOM), attn_kwargs={"num_heads": 2})
    torch.manual_seed(5)
    plc = _encoder(gmp_layers=None)
    assert list(prod.state_dict()) == list(plc.state_dict())
    for k, v in prod.state_dict().items():
        assert torch.equal(v, plc.state_dict()[k]), k


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gmp_layers": [0, 0]}, "must not repeat"),
        ({"gmp_layers": [6]}, r"must be in \[0, 5\]"),
        ({"gmp_layers": [-1]}, r"must be in \[0, 5\]"),
        ({"gmp_layers": []}, "empty"),
        ({"gmp_impl": "nope"}, "gmp_impl must be one of"),
    ],
)
def test_rejects_bad_placement(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _encoder(**kwargs)


def test_gmp_layers_needs_gmp():
    with pytest.raises(ValueError, match="needs gmp to be set"):
        PlacementEncoder(gmp_layers=[0], num_layers=6, dim=16, attn_type="patch", patch_size=4, attn_kwargs={"num_heads": 2})


# ------------------------------------------------- the whole encoder, production vs einsumopt
@pytest.mark.parametrize("device", DEVICES)
def test_whole_encoder_agrees_in_fp64(device):
    """End to end: swapping gmp_impl must not move the encoder output beyond fp64 round-off."""
    torch.manual_seed(11)
    prod = _encoder(gmp_layers=[0, 3], gmp_impl="production").to(device=device, dtype=torch.float64).eval()
    torch.manual_seed(11)
    opt = _encoder(gmp_layers=[0, 3], gmp_impl="einsumopt").to(device=device, dtype=torch.float64).eval()

    gen = torch.Generator(device="cpu").manual_seed(2)
    batch, tokens = 2, 16
    x = torch.randn(batch, tokens, 16, generator=gen).to(device=device, dtype=torch.float64)
    eta = (torch.rand(batch, tokens, generator=gen) * 8 - 4).to(device=device, dtype=torch.float64)
    phi = (torch.rand(batch, tokens, generator=gen) * 2 * math.pi - math.pi).to(device=device, dtype=torch.float64)
    coords = torch.stack([eta, phi], dim=-1)
    kv_mask = torch.ones(batch, tokens, dtype=torch.bool, device=device)
    kv_mask[:, -2:] = False

    with torch.no_grad():
        want = prod(x, kv_mask=kv_mask, x_coords=coords)
        got = opt(x, kv_mask=kv_mask, x_coords=coords)
    assert want.shape == got.shape
    assert torch.allclose(want, got, rtol=0, atol=1e-10), (want - got).abs().max().item()
