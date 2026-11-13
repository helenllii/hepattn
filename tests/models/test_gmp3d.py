"""3D GMP: the (eta, phi, depth) neighbourhood, the Conv3d GMP module, its backends, the encoder plumbing, calo depth.

The central check is test_3d_gmp_matches_voxel_conv3d: the pairwise GMP (no grid) against an independent dense
implementation -- scatter the tokens into an (eta, phi, depth) voxel grid, torch's conv3d with circular phi padding,
gather back -- in float64, forward and every gradient, for every backend. The 2D GMP is checked to be unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from hepattn.experiments.clic import calo_depth
from hepattn.experiments.clic.a4 import GMP_IMPLS, PlacementEncoder
from hepattn.models.encoder import Encoder, GeometricMessagePassing, gmp_neighbourhood, gmp_neighbourhood_3d

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
GRID2 = {"eta_cell": 0.2, "phi_cells": 32, "eta_max": 5.2}
PARAMS = ["dw.weight", "dw.bias", "pointwise.weight", "pointwise.bias", "norm.weight", "norm.bias"]


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def _tokens(b, n, seed=0, eta_scale=1.5, depth_scale=600.0):
    """Clustered tokens: many share cells; some real ones off grid (|eta| > eta_max, NaN depth); NaN padding."""
    g = torch.Generator().manual_seed(seed)
    valid = torch.arange(n).unsqueeze(0) < torch.randint(1, n, (b, 1), generator=g)
    c = torch.randint(0, 6, (b, n), generator=g)
    eta = (torch.randn(b, 6, generator=g) * eta_scale).gather(1, c) + torch.randn(b, n, generator=g) * 0.1
    phi = (torch.rand(b, 6, generator=g) * 2 * math.pi - math.pi).gather(1, c) + torch.randn(b, n, generator=g) * 0.1
    depth = (torch.rand(b, 6, generator=g) * depth_scale).gather(1, c) + torch.randn(b, n, generator=g) * 60
    phi[:, :4] = torch.tensor([math.pi - 1e-3, -math.pi + 1e-3, 3.1, -3.1])  # across the wrap
    depth[:, 4] = -7.0  # slightly in front of the face: joins the first depth bin
    eta[:, 5] = 6.0  # beyond eta_max: off grid
    depth[:, 6] = float("nan")  # non-finite depth: off grid
    for t in (eta, phi, depth):
        t[~valid] = float("nan")
    return eta.double(), phi.double(), depth.double(), valid


# ---------------------------------------------------------------------------------------------- neighbourhood
@pytest.mark.parametrize(("k", "kd", "w"), [(3, 3, 32), (3, 1, 32), (3, 5, 3), (5, 3, 5), (1, 3, 1)])
def test_3d_neighbourhood_matches_brute_force(k, kd, w):
    eta, phi, depth, valid = _tokens(2, 24, seed=k * 10 + kd)
    dc = 150.0
    nb, in_grid = gmp_neighbourhood_3d(eta, phi, depth, valid, k, 0.2, w, 5.2, dc, kd)
    r, rd, kk = k // 2, kd // 2, k * k * kd
    assert nb.shape == (2, kk * 24, 24)
    for bi in range(2):
        for i in range(24):
            for j in range(24):
                ok = all(bool(t) for t in (valid[bi, i], valid[bi, j]))
                ok = ok and all(math.isfinite(float(c[bi, q])) for c in (eta, phi, depth) for q in (i, j))
                ok = ok and abs(float(eta[bi, i])) <= 5.2 and abs(float(eta[bi, j])) <= 5.2
                want = np.zeros(kk, bool)
                if ok:
                    u = [math.floor((float(eta[bi, q]) + 5.2) / 0.2) for q in (i, j)]
                    v = [math.floor(float(phi[bi, q]) * w / (2 * math.pi)) % w for q in (i, j)]
                    d = [math.floor(max(float(depth[bi, q]), 0.0) / dc) for q in (i, j)]
                    du, dw = u[1] - u[0], d[1] - d[0]
                    dv = (v[1] - v[0] + r) % w - r
                    if abs(du) <= r and abs(dv) <= r and abs(dw) <= rd:
                        want[((du + r) * k + (dv + r)) * kd + (dw + rd)] = True
                got = nb[bi, torch.arange(kk) * 24 + i, j].numpy()
                assert (got == want).all(), (bi, i, j, got.nonzero(), want.nonzero())
        assert torch.equal(in_grid[bi], valid[bi] & eta[bi].isfinite() & depth[bi].isfinite() & (eta[bi].abs() <= 5.2))


@pytest.mark.parametrize(("k", "kd"), [(3, 3), (5, 1), (3, 5)])
def test_3d_neighbourhood_is_symmetric(k, kd):
    eta, phi, depth, valid = _tokens(3, 40, seed=7)
    nb, _ = gmp_neighbourhood_3d(eta, phi, depth, valid, k, 0.2, 32, 5.2, 200.0, kd)
    a = nb.view(3, k * k * kd, 40, 40)
    assert torch.equal(a, a.flip(1).transpose(-1, -2))


def test_3d_reduces_to_2d_when_depth_is_flat():
    eta, phi, depth, valid = _tokens(3, 40, seed=3)
    flat = torch.where(depth.isfinite(), torch.zeros_like(depth), depth)  # same bin for all, NaN kept where NaN
    nb2, g2 = gmp_neighbourhood(eta, phi, valid & depth.isfinite(), 3, **GRID2)
    nb1, g1 = gmp_neighbourhood_3d(eta, phi, flat, valid, 3, **GRID2, depth_cell=100.0, depth_kernel_size=1)
    assert torch.equal(nb1, nb2) and torch.equal(g1, g2)
    nb3, _ = gmp_neighbourhood_3d(eta, phi, flat, valid, 3, **GRID2, depth_cell=100.0, depth_kernel_size=3)
    a = nb3.view(3, 9, 3, 40, 40)
    assert torch.equal(a[:, :, 1], nb2.view(3, 9, 40, 40))  # the dw = 0 slice is the 2D kernel
    assert not a[:, :, 0].any() and not a[:, :, 2].any()


# -------------------------------------------------------------------------------------------- the GMP module
def test_2d_module_unchanged():
    """No depth keys (or depth_cell None): the original module -- Conv2d, k^2 offsets, same keys, same init stream."""
    torch.manual_seed(0)
    mod = GeometricMessagePassing(16, 3, **GRID2)
    torch.manual_seed(0)
    ref = (nn.Conv2d(16, 16, 3, groups=16, bias=True), nn.Linear(16, 16), nn.LayerNorm(16, eps=1e-6))
    assert type(mod.dw) is nn.Conv2d and mod.num_offsets == 9
    assert list(mod.state_dict()) == PARAMS
    for name, t in zip(
        ["dw.weight", "dw.bias", "pointwise.weight", "pointwise.bias"], [ref[0].weight, ref[0].bias, ref[1].weight, ref[1].bias], strict=True
    ):
        assert torch.equal(mod.state_dict()[name], t), name
    assert mod.geometry == {"kernel_size": 3, **GRID2}
    with pytest.raises(ValueError, match="needs depth_cell"):
        GeometricMessagePassing(16, 3, **GRID2, depth_kernel_size=3)
    with pytest.raises(ValueError, match="odd"):
        GeometricMessagePassing(16, 3, **GRID2, depth_cell=100.0, depth_kernel_size=2)


def test_3d_module_parameters():
    mod = GeometricMessagePassing(256, 3, **GRID2, depth_cell=200.0, depth_kernel_size=3)
    assert type(mod.dw) is nn.Conv3d and mod.dw.weight.shape == (256, 1, 3, 3, 3) and mod.num_offsets == 27
    assert sum(p.numel() for p in mod.parameters()) == 27 * 256 + 256 + 256 * 256 + 256 + 2 * 256  # 73,472
    assert sum(p.numel() for p in GeometricMessagePassing(256, 3, **GRID2).parameters()) == 68_864
    assert mod.geometry == {"kernel_size": 3, **GRID2, "depth_cell": 200.0, "depth_kernel_size": 3}


def _voxel_reference(mod, x, eta, phi, depth, valid):
    """Dense GMP: scatter-sum into an (eta, phi, depth) grid, conv3d (circular phi, zero eta/depth padding), gather."""
    k, kd, w = mod.kernel_size, mod.depth_kernel_size, mod.phi_cells
    r, rd = k // 2, kd // 2
    in_grid = valid & eta.isfinite() & phi.isfinite() & depth.isfinite() & (eta.abs() <= mod.eta_max)
    e0, p0, d0 = (torch.where(in_grid, t, 0.0) for t in (eta, phi, depth))
    u = torch.floor((e0 + mod.eta_max) / mod.eta_cell).long()
    v = torch.remainder(torch.floor(p0 * (w / (2 * math.pi))), w).long()
    dd = torch.floor(d0.clamp_min(0.0) / mod.depth_cell).long()
    b, n, dim = x.shape
    nu, nd = int(u.max()) + 1, int(dd.max()) + 1
    xm = torch.where(in_grid.unsqueeze(-1), x, 0.0)
    flat = ((torch.arange(b).unsqueeze(1) * nu + u) * w + v) * nd + dd  # (B, N) voxel index
    grid = torch.zeros(b * nu * w * nd, dim, dtype=x.dtype).index_add(0, flat.reshape(-1), xm.reshape(-1, dim))
    grid = grid.view(b, nu, w, nd, dim).permute(0, 4, 1, 2, 3)  # (B, D, eta, phi, depth)
    grid = F.pad(grid, (0, 0, r, r, 0, 0), mode="circular") if r else grid  # phi is periodic
    grid = F.pad(grid, (rd, rd, 0, 0, r, r))  # eta and depth are not
    conv = F.conv3d(grid, mod.dw.weight, mod.dw.bias, groups=dim)  # (B, D, nu, w, nd)
    z = conv.permute(0, 2, 3, 4, 1).reshape(b, nu * w * nd, dim)
    z = torch.gather(z, 1, ((u * w + v) * nd + dd).unsqueeze(-1).expand(b, n, dim))
    return torch.where(in_grid.unsqueeze(-1), x + mod.norm(mod.pointwise(z)), x)


@pytest.mark.parametrize("impl", sorted(GMP_IMPLS))
@pytest.mark.parametrize(("k", "kd"), [(3, 3), (3, 1), (5, 3), (3, 5)])
def test_3d_gmp_matches_voxel_conv3d(impl, k, kd):
    eta, phi, depth, valid = _tokens(2, 48, seed=k + kd, eta_scale=0.8)
    geo = {"kernel_size": k, "eta_cell": 0.25, "phi_cells": 12, "eta_max": 2.5, "depth_cell": 250.0, "depth_kernel_size": kd}
    torch.manual_seed(1)
    mod = GMP_IMPLS[impl](dim=8, **geo).double()
    with torch.no_grad():
        mod.norm.weight.uniform_(0.5, 1.5)
        mod.norm.bias.uniform_(-0.5, 0.5)
    nb, in_grid = gmp_neighbourhood_3d(eta, phi, depth, valid, **geo)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(2, 48, 8, generator=g, dtype=torch.float64)
    up = torch.randn(2, 48, 8, generator=g, dtype=torch.float64)
    outs = []
    for fn in (lambda xi: mod(xi, nb.double(), in_grid), lambda xi: _voxel_reference(mod, xi, eta, phi, depth, valid)):
        xi = x.clone().requires_grad_(True)
        mod.zero_grad(set_to_none=True)
        y = fn(xi)
        y.backward(up)
        outs.append((y.detach(), xi.grad, {n: p.grad.clone() for n, p in mod.named_parameters()}))
    (y1, gx1, gp1), (y2, gx2, gp2) = outs
    assert _rel(y1, y2) <= 1e-12 and _rel(gx1, gx2) <= 1e-12
    for name in PARAMS:
        assert _rel(gp1[name], gp2[name]) <= 1e-11, name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", ["einsumopt", "weightsfirst", "weightsfirst_loop"])
@pytest.mark.parametrize("amp", [False, True])
def test_3d_backends_match_production(impl, amp, device):
    eta, phi, depth, valid = _tokens(2, 160, seed=4)
    geo = {"kernel_size": 3, **GRID2, "depth_cell": 200.0, "depth_kernel_size": 3}
    nb, in_grid = gmp_neighbourhood_3d(eta.to(device), phi.to(device), depth.to(device), valid.to(device), **geo)
    dtype = torch.float32 if amp else torch.float64
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 160, 16, generator=g).to(device=device, dtype=dtype)
    up = torch.randn(2, 160, 16, generator=g).to(device=device, dtype=dtype)
    res = {}
    for name in ("production", impl):
        torch.manual_seed(5)
        mod = GMP_IMPLS[name](dim=16, **geo).to(device=device, dtype=dtype)
        xi = x.clone().requires_grad_(True)
        ctx = torch.autocast(device, dtype=torch.bfloat16) if amp else torch.autocast(device, enabled=False)
        with ctx:
            y = mod(xi, nb.to(dtype), in_grid)
        y.backward(up.to(y.dtype))
        res[name] = (y.detach(), xi.grad, [p.grad for p in mod.parameters()])
    tol = 3e-2 if amp else 1e-12  # bf16: two different summation orders, each ~1e-2 from exact
    assert _rel(res[impl][0], res["production"][0]) <= tol
    assert _rel(res[impl][1], res["production"][1]) <= tol
    for a, b in zip(res[impl][2], res["production"][2], strict=True):
        assert _rel(a, b) <= max(tol, 1e-11)


@pytest.mark.parametrize("impl", sorted(GMP_IMPLS))
def test_3d_padding_empty_and_colocated(impl):
    eta, phi, depth, valid = _tokens(3, 32, seed=6)
    valid[2] = False  # an empty event
    geo = {"kernel_size": 3, **GRID2, "depth_cell": 200.0, "depth_kernel_size": 3}
    # event 1: tokens 0-5 in one (eta, phi, depth) cell; token 6 same (eta, phi) cell but 2 depth bins away; 7 one bin away
    valid[1] = False
    valid[1, :8] = True
    eta[1, :8], phi[1, :8] = 0.05, 0.05
    depth[1, :6], depth[1, 6], depth[1, 7] = 10.0, 450.0, 250.0
    nb, in_grid = gmp_neighbourhood_3d(eta, phi, depth, valid, **geo)
    torch.manual_seed(3)
    mod = GMP_IMPLS[impl](dim=8, **geo).double()
    x = torch.randn(3, 32, 8, dtype=torch.float64)
    dirty = torch.where(in_grid.unsqueeze(-1), x, float("nan"))
    y = mod(dirty, nb.double(), in_grid)
    clean = mod(torch.where(in_grid.unsqueeze(-1), x, 0.0), nb.double(), in_grid)
    assert torch.equal(y[in_grid], clean[in_grid])  # padding / off-grid garbage never reaches a real token
    assert y[~in_grid].isnan().all() and not in_grid[2].any()
    w = mod.dw.weight.view(8, 27)
    xm = torch.where(in_grid.unsqueeze(-1), x, 0.0)
    z = torch.einsum("bond,do->bnd", torch.bmm(nb.double(), xm).view(3, 27, 32, 8), w)[1]
    centre, deeper = 13, 14  # (du, dv, dw) = (0, 0, 0) and (0, 0, +1)
    expect = w[:, centre] * x[1, :6].sum(0) + w[:, deeper] * x[1, 7]  # token 6 is out of depth reach
    assert torch.allclose(z[:6], expect.expand(6, 8), rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------------------------------- encoder
def _encoder(gmp, impl="production", layers=None, seed=11):
    torch.manual_seed(seed)
    return PlacementEncoder(
        gmp_layers=layers, gmp_impl=impl, num_layers=6, dim=16, attn_type="patch", patch_size=4, attn_kwargs={"num_heads": 2}, gmp=gmp
    ).double()


def _encoder_inputs(dims, seed=2):
    g = torch.Generator().manual_seed(seed)
    b, s = 2, 16
    x = torch.randn(b, s, 16, generator=g, dtype=torch.float64)
    cols = [torch.rand(b, s, generator=g) * 6 - 3, torch.rand(b, s, generator=g) * 2 * math.pi - math.pi, torch.rand(b, s, generator=g) * 900][:dims]
    kv = torch.ones(b, s, dtype=torch.bool)
    kv[:, -3:] = False
    return x, torch.stack(cols, -1).double(), kv


@pytest.mark.parametrize("impl", ["einsumopt", "weightsfirst", "weightsfirst_loop"])
def test_encoder_3d_all_layers(impl):
    gmp = {"kernel_size": 3, **GRID2, "depth_cell": 200.0, "depth_kernel_size": 3}
    prod, new = _encoder(gmp), _encoder(gmp, impl)
    assert all(type(lyr.gmp.dw) is nn.Conv3d for lyr in new.layers)
    assert len({id(lyr.gmp) for lyr in new.layers}) == 6  # one GMP per patch block, nothing shared
    x, coords, kv = _encoder_inputs(3)
    outs = []
    for enc in (prod, new):
        xi = x.clone().requires_grad_(True)
        y = enc(xi, kv_mask=kv, x_coords=coords)
        y.square().sum().backward()
        outs.append((y.detach(), xi.grad, [p.grad for p in enc.parameters()]))
    assert outs[0][0].isfinite().all()
    assert _rel(outs[1][0], outs[0][0]) <= 1e-10 and _rel(outs[1][1], outs[0][1]) <= 1e-10
    for a, b in zip(outs[1][2], outs[0][2], strict=True):
        if b is not None:
            assert a.isfinite().all() and _rel(a, b) <= 1e-9


def test_encoder_coordinate_count_is_checked():
    x, c2, kv = _encoder_inputs(2)
    _, c3, _ = _encoder_inputs(3)
    with pytest.raises(ValueError, match="needs 2 coordinates"):
        _encoder({"kernel_size": 3, **GRID2})(x, kv_mask=kv, x_coords=c3)
    with pytest.raises(ValueError, match="needs 3 coordinates"):
        _encoder({"kernel_size": 3, **GRID2, "depth_cell": 200.0})(x, kv_mask=kv, x_coords=c2)


def test_encoder_2d_with_explicit_null_depth_is_2d():
    """A config that writes depth_cell: null must be exactly the 2D GMP."""
    x, c2, kv = _encoder_inputs(2)
    torch.manual_seed(4)
    plain = Encoder(num_layers=2, dim=16, attn_type="patch", patch_size=4, attn_kwargs={"num_heads": 2}, gmp={"kernel_size": 3, **GRID2}).double()
    torch.manual_seed(4)
    null = Encoder(
        num_layers=2,
        dim=16,
        attn_type="patch",
        patch_size=4,
        attn_kwargs={"num_heads": 2},
        gmp={"kernel_size": 3, **GRID2, "depth_cell": None, "depth_kernel_size": None},
    ).double()
    assert all(type(lyr.gmp.dw) is nn.Conv2d for lyr in null.layers)
    with torch.no_grad():
        assert torch.equal(plain(x, kv_mask=kv, x_coords=c2), null(x, kv_mask=kv, x_coords=c2))


# --------------------------------------------------------------------------------------------- depth coordinate
def test_calo_depth_definitions():
    rb, ze = calo_depth.R_FRONT, calo_depth.Z_FRONT
    x = np.array([1700.0, 500.0, 1600.0, 1000.0, rb, 0.0, 1800.0])
    y = np.zeros_like(x)
    z = np.array([0.0, 2500.0, 2400.0, 1000.0, 1500.0, ze, 1200.0])
    env = calo_depth.depth_envelope(x, y, z)
    np.testing.assert_allclose(env[:5], [1700 - rb, 2500 - ze, math.hypot(1600 - rb, 2400 - ze), 0.0, 0.0], atol=1e-9)
    assert env[5] == 0.0  # on the endcap face (a track intersection point)
    proj = calo_depth.depth_projective(x, y, z)
    np.testing.assert_allclose(proj[0], env[0], atol=1e-9)  # theta = 90 deg: projective = envelope
    p = math.hypot(1800.0, 1200.0)
    np.testing.assert_allclose(proj[6], p * (1 - rb / 1800.0), rtol=1e-12)  # barrel: env / sin(theta)
    np.testing.assert_allclose(proj[6], env[6] * p / 1800.0, rtol=1e-12)
    assert proj[3] == 0.0 and proj[4] == 0.0  # inside / on the face
    comp = calo_depth.depth_compartment(np.array([9.0, 1.0, 0.0]), np.array([1.0, 9.0, 0.0]), np.zeros(3))
    np.testing.assert_array_equal(comp, [1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="depth mode"):
        calo_depth.topo_depth("layers", x, y, z, x, x, x)
