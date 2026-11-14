"""The GMP sweep is controlled: the committed arm configs are the generator's, and arms differ only in GMP settings."""

from __future__ import annotations

import copy

import yaml

from hepattn.experiments.clic.gmp_ablation import sweep

GMP_KEYS = {"gmp", "gmp_layers", "gmp_impl"}


def _load(name):
    return yaml.safe_load((sweep.SWEEP_DIR / f"{name}.yaml").read_text())


def test_committed_configs_are_generated():
    for name in sweep.ARMS:
        assert _load(name) == sweep.arm_config(name), name


def _strip(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop("name")
    cfg.pop("data", None)  # only gmp_depth / gmp_track_depth
    init = cfg["model"]["model"]["init_args"]
    init.pop("input_coord_fields", None)
    for k in GMP_KEYS:
        init["encoder"]["init_args"].pop(k)
    return cfg


def test_arms_differ_only_in_gmp():
    ref = _strip(_load("g2d_e020"))
    for name in sweep.ARMS:
        assert _strip(_load(name)) == ref, name
        data = _load(name).get("data", {})
        assert set(data) <= {"gmp_depth", "gmp_track_depth"}, name


def test_encoder_is_the_bench3_phat16_block():
    bench = yaml.safe_load((sweep.CLIC / "configs/bench3/phat16_gmp_all_production.yaml").read_text())
    enc = bench["model"]["model"]["init_args"]["encoder"]["init_args"]
    ours = _load("g2d_e020")["model"]["model"]["init_args"]["encoder"]["init_args"]
    for k in GMP_KEYS - {"gmp_impl"}:
        assert ours[k] == enc[k], k
    assert {k: v for k, v in ours.items() if k not in GMP_KEYS} == {k: v for k, v in enc.items() if k not in GMP_KEYS}


def test_depth_arms_are_consistent():
    for name, (gmp, depth, _) in sweep.ARMS.items():
        cfg = _load(name)
        fields = cfg["model"]["model"]["init_args"].get("input_coord_fields")
        three_d = gmp is not None and gmp.get("depth_cell") is not None
        assert (depth is not None) == three_d, name
        if gmp is None:
            assert fields is None
        else:
            assert fields == ["eta_calo", "phi_calo"] + (["depth_calo"] if three_d else []), name


def test_track_depth_arms_pair_with_depth_wildcard():
    """gmp_track_depth "any" (tracks at depth -1) is meaningful only with a 3D GMP that has depth_wildcard, and vice versa."""
    for name, (gmp, depth, _) in sweep.ARMS.items():
        data = _load(name).get("data", {})
        wildcard = gmp is not None and bool(gmp.get("depth_wildcard"))
        any_tracks = data.get("gmp_track_depth") == "any"
        assert wildcard == any_tracks, name
        if wildcard:
            assert depth is not None and gmp.get("depth_cell") is not None, name


def test_track_depth_ablation_arms_differ_only_in_track_depth():
    """The track-depth ablation's 3D arms are identical except the wildcard and the data's track depth."""
    base, any_ = _load("g3d_env_dz200"), _load("g3d_env_dz200_trkany")
    for cfg in (base, any_):
        cfg.pop("name")
    any_["model"]["model"]["init_args"]["encoder"]["init_args"]["gmp"].pop("depth_wildcard")
    any_["data"].pop("gmp_track_depth")
    assert base == any_
