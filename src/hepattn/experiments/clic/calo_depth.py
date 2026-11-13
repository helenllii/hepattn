"""Calorimeter depth of CLIC nodes: the third GMP coordinate. Pure numpy; shared by pflow_data and the geometry study.

The calorimeter's inner front envelope is the barrel cylinder r = R_FRONT and the endcap planes |z| = Z_FRONT -- the
surfaces on which track_x/y/z_int (and so track_eta_int / track_phi_int) are given. Every mode returns 0 on that
envelope, so a track (whose position is its intersection with the envelope) has depth 0 by construction.

Modes (see gmp_ablation/GEOMETRY_3D.md for the data study behind the choice):
  "envelope"    Euclidean distance (mm) from the point to the envelope's interior {r <= R, |z| <= Z}:
                r - R in the barrel, |z| - Z in the endcap, sqrt of both squared beyond the corner, 0 inside.
                Normal to the calorimeter layers, so the ECAL/HCAL boundary sits at one depth in both barrel and endcap.
  "projective"  distance (mm) past the front face along the ray from the origin (envelope / sin(theta) in the barrel):
                the path a particle from the IP travels in the calorimeter.
  "compartment" 0 for tracks, 1 for an ECAL-dominated cluster, 2 for an HCAL-dominated one (hcal energy fraction >= 0.5
                of ecal + hcal + other); a two-layer detector picture, needs no positions.
"""

from __future__ import annotations

import numpy as np

R_FRONT = 1502.186  # mm, barrel front face (track_x/y/z_int, verified in analysis/gmp_implementation_20261020)
Z_FRONT = 2309.0  # mm, endcap front face
MODES = ("envelope", "projective", "compartment")


def depth_envelope(x, y, z):
    r = np.hypot(x, y)
    return np.hypot(np.clip(r - R_FRONT, 0, None), np.clip(np.abs(z) - Z_FRONT, 0, None))


def depth_projective(x, y, z):
    r = np.hypot(x, y)
    p = np.sqrt(r * r + z * z)
    with np.errstate(divide="ignore", invalid="ignore"):
        tb = np.where(r > 0, R_FRONT * p / r, np.inf)
        te = np.where(np.abs(z) > 0, Z_FRONT * p / np.abs(z), np.inf)
    return np.clip(p - np.minimum(tb, te), 0, None)


def depth_compartment(energy_ecal, energy_hcal, energy_other):
    tot = energy_ecal + energy_hcal + energy_other
    with np.errstate(divide="ignore", invalid="ignore"):
        hcal_frac = np.where(tot > 0, energy_hcal / np.where(tot > 0, tot, 1.0), 0.0)
    return np.where(hcal_frac >= 0.5, 2.0, 1.0)


def topo_depth(mode: str, x, y, z, energy_ecal, energy_hcal, energy_other):
    """Depth of topo clusters for `mode`, float32.

    Raises:
        ValueError: If mode is not one of MODES.
    """
    if mode == "envelope":
        d = depth_envelope(x, y, z)
    elif mode == "projective":
        d = depth_projective(x, y, z)
    elif mode == "compartment":
        d = depth_compartment(energy_ecal, energy_hcal, energy_other)
    else:
        raise ValueError(f"depth mode must be one of {MODES}, got {mode!r}")
    return np.asarray(d, dtype=np.float32)
