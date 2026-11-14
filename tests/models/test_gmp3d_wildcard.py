"""3D GMP with depth_wildcard: tracks at "any" depth (track-cluster pairs ignore the depth difference).

The track-depth ablation's third arm. A token with depth < 0 (tracks: calo_depth.ANY_DEPTH = -1 under CLICDataset
gmp_track_depth "any") is at every depth: in any pair involving it, the depth offset is taken as 0 (the kernel's centre
depth slice), in both directions, so the neighbourhood stays symmetric and every backend, including the fused
weights-first one, applies unchanged. Checked against an independent brute-force enumeration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hepattn.experiments.clic.a4 import GMP_IMPLS, PlacementEncoder
from hepattn.experiments.clic.calo_depth import ANY_DEPTH, TRACK_DEPTH_MODES
from hepattn.models.encoder import GeometricMessagePassing, gmp_neighbourhood_3d

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
IMPLS = ["production", "einsumopt", "weightsfirst", "weightsfirst_loop"]
PARAMS = ["dw.weight", "dw.bias", "pointwise.weight", "pointwise.bias", "norm.weight", "norm.bias"]
GEOM = {"kernel_size": 3, "eta_cell": 0.25, "phi_cells": 8, "eta_max": 1.0, "depth_cell": 100.0, "depth_kernel_size": 3}


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def _event(b, n, seed, n_tracks=6):
    """The first n_tracks tokens are tracks at ANY_DEPTH; the rest clusters at depths 0..900 mm; padding at the end."""
    g = torch.Generator().manual_seed(seed)
    eta = (torch.rand(b, n, generator=g) * 2 - 1) * 1.1  # a few beyond eta_max
    phi = torch.rand(b, n, generator=g) * 2 * math.pi - math.pi
    depth = torch.rand(b, n, generator=g) * 900.0
    depth[:, :n_tracks] = ANY_DEPTH
    eta[:, 1], phi[:, 1] = eta[:, n_tracks], phi[:, n_tracks]  # a track exactly on a deep cluster's (eta, phi) cell
    depth[:, n_tracks] = 850.0
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[:, -3:] = False
    eta, phi, depth = (torch.where(valid, t, float("nan")) for t in (eta, phi, depth))
    return eta, phi, depth, valid


def _brute(eta, phi, depth, valid, kernel_size, eta_cell, phi_cells, eta_max, depth_cell, depth_kernel_size, depth_wildcard):
    b, n = eta.shape
    r, rd = kernel_size // 2, depth_kernel_size // 2
    out = np.zeros((b, kernel_size**2 * depth_kernel_size, n, n), dtype=bool)
    for e in range(b):
        cells = {}
        for i in range(n):
            et, ph, de = float(eta[e, i]), float(phi[e, i]), float(depth[e, i])
            if valid[e, i] and all(map(math.isfinite, (et, ph, de))) and abs(et) <= eta_max:
                cells[i] = (
                    math.floor((et + eta_max) / eta_cell),
                    math.floor(ph * phi_cells / (2 * math.pi)) % phi_cells,
                    math.floor(max(de, 0.0) / depth_cell),
                    de < 0,
                )
        for i, (ui, vi, wi, ai) in cells.items():
            for j, (uj, vj, wj, aj) in cells.items():
                du, dv = uj - ui, (vj - vi + r) % phi_cells - r
                dw = 0 if depth_wildcard and (ai or aj) else wj - wi
                if abs(du) <= r and abs(dv) <= r and abs(dw) <= rd:
                    out[e, ((du + r) * kernel_size + (dv + r)) * depth_kernel_size + (dw + rd), i, j] = True
    return out.reshape(b, -1, n)


@pytest.mark.parametrize("wildcard", [False, True])
def test_matches_brute_force(wildcard):
    eta, phi, depth, valid = _event(3, 28, seed=1)
    got, _ = gmp_neighbourhood_3d(eta, phi, depth, valid, **GEOM, depth_wildcard=wildcard)
    want = _brute(eta, phi, depth, valid, **GEOM, depth_wildcard=wildcard)
    assert torch.equal(got, torch.from_numpy(want))


def test_symmetric_so_fused_backend_applies():
    eta, phi, depth, valid = _event(2, 40, seed=2)
    nb, _ = gmp_neighbourhood_3d(eta, phi, depth, valid, **GEOM, depth_wildcard=True)
    a = nb.view(2, 27, 40, 40)
    assert torch.equal(a, a.flip(1).transpose(-1, -2))


def test_track_reaches_deep_cluster_only_with_wildcard():
    """Track 1 shares its (eta, phi) cell with a cluster 850 mm deep: a neighbour only with the wildcard, at dw = 0."""
    eta, phi, depth, valid = _event(1, 20, seed=3)
    k = 6  # the deep cluster
    off_same_cell = lambda dw: ((1 * 3) + 1) * 3 + (dw + 1)  # noqa: E731 - du = dv = 0
    plain, _ = gmp_neighbourhood_3d(eta, phi, depth, valid, **GEOM)
    wild, _ = gmp_neighbourhood_3d(eta, phi, depth, valid, **GEOM, depth_wildcard=True)
    p, w = plain.view(1, 27, 20, 20)[0], wild.view(1, 27, 20, 20)[0]
    assert not p[:, 1, k].any() and not p[:, k, 1].any()  # 850 mm vs bin 0: out of reach
    assert w[off_same_cell(0), 1, k] and w[off_same_cell(0), k, 1]
    assert w[:, 1, k].sum() == 1


def test_no_negative_depth_means_no_change():
    eta, phi, depth, valid = _event(2, 30, seed=4, n_tracks=0)
    a, ia = gmp_neighbourhood_3d(eta, phi, depth, valid, **GEOM)
    b, ib = gmp_neighbourhood_3d(eta, phi, depth, valid, **GEOM, depth_wildcard=True)
    assert torch.equal(a, b) and torch.equal(ia, ib)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS[1:])
def test_backends_match_production_fp64(impl, device):
    eta, phi, depth, valid = (t.to(device) for t in _event(2, 48, seed=5))
    nb, in_grid = gmp_neighbourhood_3d(eta, phi, depth, valid, **GEOM, depth_wildcard=True)
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(2, 48, 8, generator=gen, dtype=torch.float64).to(device)
    g = torch.randn(2, 48, 8, generator=gen, dtype=torch.float64).to(device)
    out = {}
    for name in ("production", impl):
        torch.manual_seed(0)
        mod = GMP_IMPLS[name](dim=8, **GEOM, depth_wildcard=True).to(device=device, dtype=torch.float64)
        xi = x.clone().requires_grad_(True)
        y = mod(xi, nb.double(), in_grid)
        y.backward(g)
        named = dict(mod.named_parameters())
        out[name] = [y.detach(), xi.grad] + [named[p].grad for p in PARAMS]
    for a, b in zip(out[impl], out["production"], strict=True):
        assert _rel(a, b) <= 1e-12


@pytest.mark.parametrize("device", DEVICES)
def test_encoder_all_layers_with_wildcard(device):
    kw = {"num_layers": 6, "dim": 16, "attn_type": "patch", "patch_size": 4, "attn_kwargs": {"num_heads": 2}, "gmp": GEOM | {"depth_wildcard": True}}
    torch.manual_seed(11)
    prod = PlacementEncoder(gmp_layers=None, gmp_impl="production", **kw).double().to(device)
    torch.manual_seed(11)
    wf = PlacementEncoder(gmp_layers=None, gmp_impl="weightsfirst", **kw).double().to(device)
    assert all(lyr.gmp.geometry.get("depth_wildcard") for lyr in wf.layers)
    eta, phi, depth, valid = (t.to(device) for t in _event(2, 16, seed=7, n_tracks=4))
    coords = torch.stack([eta, phi, depth], -1).double()
    x = torch.randn(2, 16, 16, generator=torch.Generator().manual_seed(8), dtype=torch.float64).to(device)
    outs = []
    for enc in (prod, wf):
        xi = x.clone().requires_grad_(True)
        y = enc(xi, kv_mask=valid, x_coords=coords)
        y.square().sum().backward()
        outs.append((y.detach(), xi.grad))
    assert outs[0][0].isfinite().all() and outs[0][1].isfinite().all()
    assert _rel(outs[1][0], outs[0][0]) <= 1e-10 and _rel(outs[1][1], outs[0][1]) <= 1e-10


def test_validation_and_params():
    with pytest.raises(ValueError, match="depth_wildcard is a 3D setting"):
        GeometricMessagePassing(dim=8, depth_wildcard=True)
    a = GeometricMessagePassing(dim=32, depth_cell=200.0)
    torch.manual_seed(0)
    b = GeometricMessagePassing(dim=32, depth_cell=200.0, depth_wildcard=True)
    assert [p.shape for p in a.parameters()] == [p.shape for p in b.parameters()]  # geometry only, no new parameters
    assert "depth_wildcard" not in a.geometry and b.geometry["depth_wildcard"] is True
    assert ANY_DEPTH < 0 and TRACK_DEPTH_MODES == ("front", "any")


def test_dataset_rejects_unknown_track_depth():
    from hepattn.experiments.clic.pflow_data import CLICDataset  # noqa: PLC0415

    with pytest.raises(ValueError, match="gmp_track_depth must be one of"):
        CLICDataset(filepath="", inputs={}, targets={}, scale_dict_path="", gmp_depth="envelope", gmp_track_depth="deep")
