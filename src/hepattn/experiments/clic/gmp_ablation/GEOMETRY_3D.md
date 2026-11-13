# 3D GMP: which depth coordinate, measured from the CLIC files

2026-11-13 · `gmp_ablation/geometry_study.py` on `val_clic_fix.root`: 3,000 events for clusters and tracks (2,972 after
the < 160-node cut), 300 events / 118,893 cells for the detector geometry. Polaris job 7841964,
`runs/gmp_ablation/validate3d/7841964/geometry_study.json`. numpy only. The definitions are in `calo_depth.py`.

## 1. What the detector data say

**Front envelope.** Track positions (`track_x/y/z_int`) lie on the barrel cylinder r = 1502.2 mm and the endcap planes
|z| = 2309.0 mm. Every one of the 62,723 tracks has depth exactly 0 under both positional definitions below.
`cell_rho` (and so `topo_rho`) is the **transverse** radius √(x² + y²), not the 3D distance.

**Calorimeter regions** (`cell_calo_region`; r = transverse radius; energy share over 300 events):

| region | r (mm) | \|z\| (mm) | energy | envelope depth (mm), p1–p99 | identification |
|---|---|---|---|---|---|
| 0 | 1502–1759 | 0–2208 | 25.8% | 16–231 (max 257) | ECAL barrel |
| 1 | 408–1754 | 2309–2506 | 25.5% | 11–203 | ECAL endcap |
| 2 | 260–393 | 2309–2506 | 4.6% | 0–197 | ECAL inner ring / plug |
| 3 | 1761–3422 | 15–2205 | 14.7% | 260–1641 | HCAL barrel |
| 4 | 234–3331 | 2560–4124 | 29.2% | 251–1762 | HCAL endcap |
| 5 | 1756–2830 | 2375–2534 | 0.4% | 324–1277 | HCAL ring (barrel/endcap overlap) |
| 6 | up to 6359 | up to 5519 | ~0% | > 2075 | beyond the calorimeters (yoke / muon) |

The calorimeter layers run parallel to the front faces: ECAL from depth 0, HCAL from ~251–260 mm, in both barrel and
endcap.

## 2. Candidate depth coordinates

All three are computable from fields the training loader reads (`topo_x/y/z`, the topo energy fractions). Tracks get
0 in all of them.

| mode | definition | ECAL-like vs HCAL-like clusters, one depth cut (balanced accuracy) |
|---|---|---|
| **envelope** | distance from the inner envelope {r ≤ R, \|z\| ≤ Z}: r − R in the barrel, \|z\| − Z in the endcap, continuous at the corners | barrel **1.0000** (cut 252 mm), endcap 0.9999 (cut 192), endcap with the barrel cut **0.9997** |
| projective | distance past the front face along the ray from the origin (= envelope / sin θ in the barrel) | barrel 0.9989 (cut 306), endcap 0.9999 (cut 214), endcap with the barrel cut 0.9967 |
| compartment | 0 track, 1 ECAL-dominated, 2 HCAL-dominated (HCAL energy fraction ≥ 0.5) | discrete by construction |

ECAL-like means HCAL energy fraction < 0.1 (97,729 clusters); HCAL-like means > 0.9 (10,543).

Cluster depth, envelope (mm): ECAL-like median 60 (p99 168); HCAL-like median 612 (p1 311, p90 1146, max 2504).
Energy-weighted depth RMS within one cluster, from its cells: median 90 mm, p90 401 mm.

**Recommendation: `envelope`.**
* It follows the detector's layer structure: the ECAL/HCAL boundary is one depth, ~250 mm, in barrel and endcap, so a
  depth bin means the same detector layer everywhere.
* Barrel and endcap need no special treatment: it is continuous through the overlap corners.
* Tracks are exactly 0.
* `projective` measures shower path length, but its compartment boundary moves with θ (ECAL reaches 398 mm in the
  barrel). `compartment` needs no positions, but it has only two cluster values.

The data justify this coordinate. The neighbourhood consequences below are a separate question.

## 3. What a 3D kernel does to the neighbourhoods

Production η/φ grid (0.2, 32 cells), kernel 3 in every axis. All nodes are on the grid. Truth: a cluster's leading
particle by energy, a track's particle.

| grid | occupied cells | neighbours / node | same-particle | purity | track's own leading cluster in reach | ECAL–HCAL pairs | same-particle ECAL–HCAL pairs kept |
|---|---|---|---|---|---|---|---|
| 2D 0.2/32 (production) | 42 | 5.80 | 0.567 | 9.8% | **90.8%** | 14.6% | 56% |
| 2D 0.1/64 | 51 | 2.60 | 0.466 | 18.0% | 79.8% | 12.9% | 34% |
| 2D 0.4/16 | 31 | 11.79 | 0.633 | 5.4% | 95.9% | 15.3% | 75% |
| 3D envelope dz100 | 51 | 3.90 | 0.310 | 8.0% | 47.7% | 1.2% | 10% |
| 3D envelope dz200 | 47 | 4.80 | 0.445 | 9.3% | 71.1% | 6.6% | 31% |
| 3D envelope dz400 | 45 | 5.61 | 0.545 | 9.7% | 87.9% | 13.2% | 52% |
| 3D projective dz200 | 48 | 4.63 | 0.423 | 9.1% | 67.2% | 5.3% | 26% |
| 3D compartment | 53 | 5.17 | 0.385 | 7.4% | 56.6% | 16.7% | 56% |
| 3D envelope dz200, 0.1/64 | 55 | 2.12 | 0.369 | 17.4% | 63.1% | 5.6% | 19% |

What this shows (measured on these events):
1. **Depth separates ECAL from HCAL, as intended.** ECAL–HCAL neighbour pairs fall from 14.6% (2D) to 1.2–6.6% at
   dz100–200.
2. **It does not make neighbourhoods purer.** The same-particle share of neighbours stays at or below 2D's 9.8% for
   every 3D grid: depth removes other-particle and same-particle neighbours at about the same rate. Purity goes up only
   with a finer η/φ grid (0.1/64: 18%), with or without depth.
3. **Tracks lose their own showers.** Tracks are at depth 0, but their particle's leading cluster is at median 178 mm
   (p90 564 mm), because hadrons shower in the HCAL. A depth kernel of ±1 bin cuts the track–shower link: reach falls
   from 91% to 71% at dz200 and 48% at dz100. A hadron shower split across ECAL and HCAL is likewise cut, and the
   compartment mode keeps those pairs only because ECAL and HCAL are adjacent bins.
4. dz400 is almost the 2D neighbourhood again, because ±1 bin spans up to 800 mm.

## 4. Track depth: the decision for the initial 3D baseline

**Decision (2026-11-13, by the user): tracks stay at depth 0 for the initial 3D baseline and the first grid-size
pilot.** A track is placed at its intersection with the calorimeter front face (`track_x/y/z_int`), so its depth is 0
under every positional mode, and the 3D kernel treats it like any other node: it reaches clusters within ±1 depth bin
(±`depth_kernel_size // 2` bins). This is implemented in `calo_depth.py` / `CLICDataset(gmp_depth=...)` and used by all
`g3d_*` sweep arms. Section 3 shows the cost: a track's own leading cluster is in reach for 71% of tracks at 200 mm bins,
against 91% in 2D. A controlled track-depth ablation (below and in the pilot report) decides whether to change it.

### Background: the options considered

The coordinate is data-backed. The open question is **how tracks should sit in depth**. With "track = depth 0" and a
symmetric kernel, 3D GMP mostly removes track→own-shower messages. Options:

* (a) Keep depth 0 for tracks and let the sweep decide. Cheap and already implemented: arms `g3d_env_dz100/200/400`,
  `g3d_proj_dz200`, `g3d_comp`.
* (b) Let tracks reach every depth bin (a track is a line through the calorimeter, not a point). This makes the
  neighbourhood asymmetric: a track→cluster offset no longer mirrors cluster→track. The production and loop backends
  handle that; the fused weights-first backend's symmetry trick does not, and would need a variant. This is an
  architectural change and needs your approval.
* (c) Give each track the depth of its expected shower: physics modelling, depends on particle type. Not recommended
  without more study.

Implemented and validated: (a), with envelope, projective and compartment depth. (b) and (c) are not implemented; (b) is
proposed as the third arm of the track-depth ablation (2D / 3D tracks at depth 0 / 3D tracks reaching every depth bin).
