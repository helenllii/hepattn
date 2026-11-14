"""CLICDataset(gmp_depth=...): the 3D-GMP depth coordinate, on real CLIC events (skipped where the file is absent).

Pins that the option is opt-in (without it nothing changes; with it the only change is the new node_depth_calo input)
and that the values are calo_depth's, recomputed here from the raw ROOT branches of the same original event.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import uproot

from hepattn.experiments.clic import calo_depth
from hepattn.experiments.clic.pflow_data import CLICDataset

VAL = Path("/eagle/hgcal-maskformer-fpga/clic_data/val_clic_fix.root")
SCALE = Path(__file__).resolve().parents[3] / "src/hepattn/experiments/clic/configs/clic_var_transform.yaml"
N_EVENTS = 200
pytestmark = pytest.mark.skipif(not VAL.exists(), reason="CLIC validation file not available here")


def _dataset(gmp_depth=None):
    return CLICDataset(
        filepath=str(VAL),
        inputs={"hit": ["x"]},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=str(SCALE),
        num_events=N_EVENTS,
        incidence_cutval=0.01,
        gmp_depth=gmp_depth,
    )


@pytest.fixture(scope="module")
def baseline():
    return _dataset()


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="gmp_depth must be one of"):
        _dataset("layers")


@pytest.mark.parametrize("mode", ["envelope", "projective", "compartment"])
def test_depth_is_opt_in_and_matches_raw_branches(baseline, mode):
    ds = _dataset(mode)
    branches = ["topo_x", "topo_y", "topo_z", "topo_energy_ecal", "topo_energy_hcal", "topo_energy_other"]  # topo_depth's order
    raw = uproot.open(VAL)["EventTree"].arrays(branches, entry_stop=N_EVENTS, library="np")
    assert len(ds) == len(baseline)
    for i in (0, 1, len(ds) // 2, len(ds) - 1):
        (inp, tgt), (inp0, tgt0) = ds[i], baseline[i]
        assert set(inp) - set(inp0) == {"node_depth_calo"} and set(inp0) <= set(inp)
        # Exact equality, NaN == NaN: the regression targets are NaN-padded beyond the real particles, and torch.equal
        # is False for any tensor holding a NaN (job 7841964 failed on that, not on a data difference).
        for k in inp0:  # every other input unchanged
            torch.testing.assert_close(inp[k], inp0[k], rtol=0, atol=0, equal_nan=True, msg=k)
        for k in tgt0:
            torch.testing.assert_close(tgt[k], tgt0[k], rtol=0, atol=0, equal_nan=True, msg=k)
        depth = inp["node_depth_calo"]
        nt, nc = int(ds.n_tracks[i]), int(ds.n_topos[i])
        assert depth.shape == inp0["node_valid"].shape and depth.dtype == torch.float32
        assert torch.all(depth[:nt] == 0)  # tracks sit on the front face
        assert torch.all(depth[nt + nc :] == 0)  # padding
        ev = int(ds.event_number[i])  # index in the ROOT file
        want = calo_depth.topo_depth(mode, *(raw[k][ev] for k in branches))
        np.testing.assert_allclose(depth[nt : nt + nc].numpy(), want, rtol=1e-5, atol=1e-3)
        assert torch.all(depth >= 0)


def test_track_depth_any_sets_tracks_to_the_sentinel():
    """gmp_track_depth "any": tracks at calo_depth.ANY_DEPTH, clusters and every other input exactly as with "front"."""
    front, anyd = (
        _dataset("envelope"),
        CLICDataset(
            filepath=str(VAL),
            inputs={"hit": ["x"]},
            targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
            scale_dict_path=str(SCALE),
            num_events=N_EVENTS,
            incidence_cutval=0.01,
            gmp_depth="envelope",
            gmp_track_depth="any",
        ),
    )
    for i in (0, len(front) // 2, len(front) - 1):
        (inp_f, tgt_f), (inp_a, tgt_a) = front[i], anyd[i]
        nt = int(front.n_tracks[i])
        assert torch.all(inp_a["node_depth_calo"][:nt] == calo_depth.ANY_DEPTH)
        assert torch.equal(inp_a["node_depth_calo"][nt:], inp_f["node_depth_calo"][nt:])
        for k in inp_f:
            if k != "node_depth_calo":
                torch.testing.assert_close(inp_a[k], inp_f[k], rtol=0, atol=0, equal_nan=True, msg=k)
        for k in tgt_f:
            torch.testing.assert_close(tgt_a[k], tgt_f[k], rtol=0, atol=0, equal_nan=True, msg=k)
