"""Data study for the GMP depth coordinate (task: 3D GMP). numpy + uproot only; run it in a PBS job, not on a login node.

Questions it answers from the CLIC files, not from assumptions:
  A. Where each calorimeter region's cells actually are (cell_calo_region vs r = hypot(x, y) and |z|), so the front
     envelope (barrel R, endcap Z) and the ECAL/HCAL boundaries are measured; what topo_rho is.
  B. Three candidate depth coordinates for topo clusters, all computable from topo_x/y/z (+ energy fractions):
       envelope   : Euclidean distance from the calorimeter's inner envelope {r <= R, |z| <= Z}
                    = r - R in the barrel, |z| - Z in the endcap, continuous at the corners; 0 on the front faces.
       projective : distance past the front face along the ray from the origin (= envelope / sin(theta) in the barrel),
                    clipped at 0 like calo_depth (a cluster barycentre can sit a few mm in front of the face).
       compartment: discrete 0 = track, 1 = ECAL-dominated cluster, 2 = HCAL-dominated (hcal energy fraction >= 0.5).
     Tracks: track_x/y/z_int is the track's intersection with the front face, so their envelope/projective depth is
     ~0 by construction (checked here).
     For each: how well one depth threshold separates ECAL-like (hcal_frac < 0.1) from HCAL-like (> 0.9) clusters, in
     barrel and endcap separately (a coordinate whose compartment boundary moves with theta mixes them in one bin).
  C. Neighbourhood structure of the 2D production grid and 3D grids built on each depth, against truth: occupied
     cells, neighbours per node, same-particle neighbours, whether a track's own leading cluster is in reach, and
     ECAL-HCAL pairs (all pairs, and the same-particle ones -- a hadron shower split across the two compartments).

    python -m hepattn.experiments.clic.gmp_ablation.geometry_study --events 3000 --cell-events 300 --out study.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import uproot

from hepattn.experiments.clic.calo_depth import R_FRONT, Z_FRONT, depth_envelope, depth_projective

VAL = "/eagle/hgcal-maskformer-fpga/clic_data/val_clic_fix.root"
ETA_MAX = 5.2
MAX_NODES = 160

TOPO = [
    "topo_eta", "topo_phi", "topo_x", "topo_y", "topo_z", "topo_rho", "topo_e", "topo_energy_ecal", "topo_energy_hcal",
    "topo_energy_other", "topo2particle_topo_idx", "topo2particle_particle_idx", "topo2particle_energy",
]  # fmt: skip
TRACK = ["track_eta_int", "track_phi_int", "track_x_int", "track_y_int", "track_z_int", "track_particle_idx", "track_pt"]
CELL = ["cell_x", "cell_y", "cell_z", "cell_rho", "cell_e", "cell_calo_region", "cell_topo_idx"]


def barrel_hit(x, y, z):
    """True where the ray from the origin meets the barrel face before the endcap plane."""
    r = np.hypot(x, y)
    return np.abs(z) * R_FRONT < Z_FRONT * r


def pct(v, ps=(0, 1, 10, 50, 90, 99, 100)):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {f"p{p}": round(float(np.percentile(v, p)), 2) for p in ps} | {"n": int(v.size)} if v.size else {"n": 0}


def best_threshold(neg, pos):
    """Balanced accuracy of the best single cut 'depth >= c -> HCAL' between two samples."""
    neg, pos = np.sort(neg), np.sort(pos)
    if not neg.size or not pos.size:
        return None
    cands = np.unique(np.concatenate([neg, pos]))
    if cands.size > 2000:
        cands = np.quantile(cands, np.linspace(0, 1, 2000))
    tnr = np.searchsorted(neg, cands, side="left") / neg.size
    tpr = 1 - np.searchsorted(pos, cands, side="left") / pos.size
    bal = 0.5 * (tnr + tpr)
    i = int(np.argmax(bal))
    return {"balanced_accuracy": round(float(bal[i]), 4), "cut": round(float(cands[i]), 2)}


# ------------------------------------------------------------------------------------------------- part A: cells
def part_a(tree, n_events):
    a = tree.arrays(CELL, entry_stop=n_events, library="np")
    cat = {k: np.concatenate(a[k]) for k in CELL}
    r = np.hypot(cat["cell_x"], cat["cell_y"])
    az = np.abs(cat["cell_z"])
    out = {
        "events": n_events,
        "cells": int(r.size),
        "cell_rho_equals_r_xy": bool(np.allclose(cat["cell_rho"], r, rtol=1e-3, atol=1e-3)),
        "cell_rho_equals_r_3d": bool(np.allclose(cat["cell_rho"], np.hypot(r, az), rtol=1e-3, atol=1e-3)),
        "regions": {},
    }
    etot = cat["cell_e"].sum()
    env = depth_envelope(cat["cell_x"], cat["cell_y"], cat["cell_z"])
    proj = depth_projective(cat["cell_x"], cat["cell_y"], cat["cell_z"])
    bar = barrel_hit(cat["cell_x"], cat["cell_y"], cat["cell_z"])
    for g in np.unique(cat["cell_calo_region"]):
        m = cat["cell_calo_region"] == g
        out["regions"][str(int(g))] = {
            "cells": int(m.sum()),
            "energy_frac": round(float(cat["cell_e"][m].sum() / etot), 4),
            "r_mm": pct(r[m], (0, 1, 50, 99, 100)),
            "abs_z_mm": pct(az[m], (0, 1, 50, 99, 100)),
            "depth_envelope_mm": pct(env[m], (0, 1, 50, 99, 100)),
            "depth_projective_mm": pct(proj[m], (0, 1, 50, 99, 100)),
            "frac_barrel_ray": round(float(bar[m].mean()), 4),
        }
    # per-topo spread of cell depth (energy weighted), for the depth-bin scale
    spreads = {"envelope": [], "projective": []}
    for ev in range(n_events):
        ti, e = a["cell_topo_idx"][ev], a["cell_e"][ev]
        if ti.size == 0:
            continue
        d = {
            "envelope": depth_envelope(a["cell_x"][ev], a["cell_y"][ev], a["cell_z"][ev]),
            "projective": depth_projective(a["cell_x"][ev], a["cell_y"][ev], a["cell_z"][ev]),
        }
        for t in np.unique(ti[ti >= 0]):
            m = ti == t
            if m.sum() < 3 or e[m].sum() <= 0:
                continue
            w = e[m] / e[m].sum()
            for k in d:
                mu = (w * d[k][m]).sum()
                spreads[k].append(math.sqrt(max((w * (d[k][m] - mu) ** 2).sum(), 0.0)))
    out["topo_energy_weighted_depth_rms_mm"] = {k: pct(v, (10, 50, 90)) for k, v in spreads.items()}
    return out


# --------------------------------------------------------------------------------------------- parts B and C: nodes
def neighbours(u, v, w, ru, rv, rw, phi_cells):
    du = np.abs(u[None, :] - u[:, None])
    dv = np.abs(np.remainder(v[None, :] - v[:, None] + rv, phi_cells) - rv)
    dw = np.abs(w[None, :] - w[:, None])
    return (du <= ru) & (dv <= rv) & (dw <= rw)


def configs():
    """(name, depth mode or None for 2D, depth cell mm, eta cell, phi cells): the grids of the sweep proposal."""
    out = [
        ("2D production 0.2/32 k3", None, None, 0.2, 32),
        ("2D 0.1/64 k3", None, None, 0.1, 64),
        ("2D 0.4/16 k3", None, None, 0.4, 16),
    ]
    for mode in ("envelope", "projective"):
        for dz in (50.0, 100.0, 200.0, 400.0):
            out.append((f"3D {mode} dz{int(dz)} 0.2/32 k3x3x3", mode, dz, 0.2, 32))  # noqa: PERF401
    out += [
        ("3D envelope dz200 0.1/64 k3x3x3", "envelope", 200.0, 0.1, 64),
        ("3D compartment 0.2/32 k3x3x3", "compartment", 1.0, 0.2, 32),
    ]
    return out


def parts_bc(tree, n_events):
    a = tree.arrays(TOPO + TRACK, entry_stop=n_events, library="np")
    cfgs = configs()
    acc = {c[0]: {k: [] for k in ("occ", "deg", "same", "trk_own", "cross_frac", "split_keep", "split_n")} for c in cfgs}
    groups = ("ecal_like", "hcal_like", "ecal_like_barrel", "hcal_like_barrel", "ecal_like_endcap", "hcal_like_endcap", "track_own_cluster")
    depth = {m: {g: [] for g in groups} for m in ("envelope", "projective")}
    track_depth = {"envelope": [], "projective": []}
    kept = 0
    for ev in range(n_events):
        nt, nc = a["track_eta_int"][ev].size, a["topo_eta"][ev].size
        if nt + nc >= MAX_NODES:
            continue
        kept += 1
        # truth: leading particle of each topo; particle of each track
        tparts = np.full(nc, -1)
        best = np.zeros(nc)
        for ti, pi, en in zip(a["topo2particle_topo_idx"][ev], a["topo2particle_particle_idx"][ev], a["topo2particle_energy"][ev], strict=True):
            if 0 <= ti < nc and en > best[ti]:
                best[ti], tparts[ti] = en, pi
        kparts = np.asarray(a["track_particle_idx"][ev], dtype=np.int64)
        ee, eh, eo = a["topo_energy_ecal"][ev], a["topo_energy_hcal"][ev], a["topo_energy_other"][ev]
        tot = ee + eh + eo
        hf = np.where(tot > 0, eh / np.where(tot > 0, tot, 1), np.nan)
        tx, ty, tz = a["topo_x"][ev], a["topo_y"][ev], a["topo_z"][ev]
        d_env, d_proj, bar = depth_envelope(tx, ty, tz), depth_projective(tx, ty, tz), barrel_hit(tx, ty, tz)
        kx, ky, kz = a["track_x_int"][ev], a["track_y_int"][ev], a["track_z_int"][ev]
        track_depth["envelope"].append(depth_envelope(kx, ky, kz))
        track_depth["projective"].append(depth_projective(kx, ky, kz))
        for m, d in (("envelope", d_env), ("projective", d_proj)):
            depth[m]["ecal_like"].append(d[hf < 0.1])
            depth[m]["hcal_like"].append(d[hf > 0.9])
            depth[m]["ecal_like_barrel"].append(d[(hf < 0.1) & bar])
            depth[m]["hcal_like_barrel"].append(d[(hf > 0.9) & bar])
            depth[m]["ecal_like_endcap"].append(d[(hf < 0.1) & ~bar])
            depth[m]["hcal_like_endcap"].append(d[(hf > 0.9) & ~bar])
        # each track's own leading cluster: the highest-energy topo of its particle
        own = np.full(nt, -1)
        for k in range(nt):
            cand = np.flatnonzero(tparts == kparts[k]) if kparts[k] >= 0 else np.array([], dtype=int)
            if cand.size:
                own[k] = cand[np.argmax(best[cand])]
                for m, d in (("envelope", d_env), ("projective", d_proj)):
                    depth[m]["track_own_cluster"].append([d[own[k]]])
        # nodes: tracks first, then topos (the loader's order)
        eta = np.concatenate([a["track_eta_int"][ev], a["topo_eta"][ev]]).astype(np.float64)
        phi = np.concatenate([a["track_phi_int"][ev], a["topo_phi"][ev]]).astype(np.float64)
        part = np.concatenate([kparts, tparts])
        is_topo = np.r_[np.zeros(nt, bool), np.ones(nc, bool)]
        hcal = np.r_[np.zeros(nt, bool), hf >= 0.5]
        comp = np.r_[np.zeros(nt), np.where(hf >= 0.5, 2.0, 1.0)]
        dep = {"envelope": np.r_[np.zeros(nt), d_env], "projective": np.r_[np.zeros(nt), d_proj], "compartment": comp}
        ing = np.isfinite(eta) & np.isfinite(phi) & (np.abs(eta) <= ETA_MAX)
        idx = np.flatnonzero(ing)
        p, t_, hc = part[idx], is_topo[idx], hcal[idx]
        same = (p[None, :] == p[:, None]) & (p[:, None] >= 0)
        cross = t_[None, :] & t_[:, None] & (hc[None, :] != hc[:, None])
        eye = np.eye(idx.size, dtype=bool)
        pos = {n: i for i, n in enumerate(idx)}
        for name, mode, dz, eta_cell, phi_cells in cfgs:
            u = np.floor((eta[idx] + ETA_MAX) / eta_cell).astype(np.int64)
            v = np.remainder(np.floor(phi[idx] * (phi_cells / (2 * math.pi))), phi_cells).astype(np.int64)
            w = np.zeros_like(u) if mode is None else np.floor(np.clip(dep[mode][idx], 0, None) / dz).astype(np.int64)
            nb = neighbours(u, v, w, 1, 1, 0 if mode is None else 1, phi_cells) & ~eye
            s = acc[name]
            s["occ"].append(len(set(zip(u.tolist(), v.tolist(), w.tolist(), strict=True))))
            s["deg"].append(nb.sum(1).mean() if idx.size else 0.0)
            s["same"].append((nb & same).sum(1).mean() if idx.size else 0.0)
            npairs = nb.sum()
            s["cross_frac"].append((nb & cross).sum() / npairs if npairs else np.nan)
            split = cross & same & ~eye
            s["split_n"].append(int(split.sum()))
            s["split_keep"].append(int((nb & split).sum()))
            for k in range(nt):
                if own[k] >= 0 and ing[k] and ing[nt + own[k]]:
                    s["trk_own"].append(bool(nb[pos[k], pos[nt + own[k]]]))
    res = {
        "events_read": n_events,
        "events_kept": kept,
        "track_depth_mm": {m: pct(np.concatenate(v), (0, 50, 99, 100)) for m, v in track_depth.items()},
        "cluster_depth_mm": {},
    }
    for m, grp in depth.items():
        g = {k: np.concatenate(v) if v else np.array([]) for k, v in grp.items()}
        res["cluster_depth_mm"][m] = {k: pct(v) for k, v in g.items()} | {
            "separation_all": best_threshold(g["ecal_like"], g["hcal_like"]),
            "separation_barrel": best_threshold(g["ecal_like_barrel"], g["hcal_like_barrel"]),
            "separation_endcap": best_threshold(g["ecal_like_endcap"], g["hcal_like_endcap"]),
            "separation_endcap_with_barrel_cut": None,
        }
        # one cut for both regions: the barrel-optimal cut applied to the endcap
        cb = res["cluster_depth_mm"][m]["separation_barrel"]
        if cb:
            neg, posv = g["ecal_like_endcap"], g["hcal_like_endcap"]
            res["cluster_depth_mm"][m]["separation_endcap_with_barrel_cut"] = round(
                float(0.5 * ((neg < cb["cut"]).mean() + (posv >= cb["cut"]).mean())), 4
            )
    res["neighbourhoods"] = {}
    for name, *_ in cfgs:
        s = acc[name]
        res["neighbourhoods"][name] = {
            "occupied_cells_median": float(np.median(s["occ"])),
            "neighbours_per_node_mean": round(float(np.mean(s["deg"])), 3),
            "same_particle_neighbours_mean": round(float(np.mean(s["same"])), 3),
            "purity": round(float(np.sum(s["same"]) / max(np.sum(s["deg"]), 1e-9)), 4),
            "track_own_cluster_in_reach": round(float(np.mean(s["trk_own"])), 4) if s["trk_own"] else None,
            "ecal_hcal_pair_frac": round(float(np.nanmean(s["cross_frac"])), 4),
            "same_particle_ecal_hcal_pairs_kept": round(float(np.sum(s["split_keep"]) / max(np.sum(s["split_n"]), 1)), 4),
        }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=VAL)
    p.add_argument("--events", type=int, default=3000)
    p.add_argument("--cell-events", type=int, default=300)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    t0 = time.time()
    tree = uproot.open(args.file)["EventTree"]
    res = {"file": args.file, "R_front_mm": R_FRONT, "Z_front_mm": Z_FRONT}
    res["cells"] = part_a(tree, args.cell_events)
    print(f"[geometry] part A done {time.time() - t0:.0f}s", flush=True)
    res |= parts_bc(tree, args.events)
    res["wall_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
