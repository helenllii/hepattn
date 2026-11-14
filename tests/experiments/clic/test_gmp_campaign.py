"""The sequential 4-GPU DDP campaign machinery: plan, PBS afterok chain, manifest/duplicate protection, status, preflight.

The modules are loaded by file path (stdlib + PyYAML only), so this file runs without torch, on any node. PBS is never
called: qsub/qstat are replaced by fakes. The qstat texts below are copied from real Polaris records of the dependency
test (2026-11-14, jobs 7848490-7848495): a dependency-deleted job is job_state F with no stime and no Exit_status.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

CLIC = Path(__file__).resolve().parents[3] / "src/hepattn/experiments/clic"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_gmp_{name}", CLIC / "gmp_ablation" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


campaign = _load("campaign")
preflight = _load("preflight")
runcheck = _load("runcheck")
SPEC = CLIC / "configs/gmp_ablation/campaigns/ablation_v1.yaml"


# ------------------------------------------------------------------------------------------------------- the spec
def test_shipped_spec_is_complete_and_pending():
    spec = campaign.load_spec(SPEC)
    assert spec["queue"] is None and spec["walltime"] is None  # the user's decision, deliberately unset
    assert len(spec["runs"]) == 11 and spec["epochs"] == 10
    for r in spec["runs"]:
        assert (CLIC / "configs/gmp_ablation/sweep" / f"{r['arm']}.yaml").exists(), r
    arms = [r["arm"] for r in spec["runs"][:3]]
    assert arms == ["g2d_e020", "g3d_env_dz200", "g3d_env_dz200_trkany"]  # the track-depth ablation first


def test_spec_rejects_duplicates_and_bad_names(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "epochs": 2, "runs": [{"arm": "g2d_e020", "seed": 1}, {"arm": "g2d_e020", "seed": 1}]}))
    with pytest.raises(ValueError, match="twice"):
        campaign.load_spec(p)
    p.write_text(yaml.safe_dump({"name": "a b", "epochs": 2, "runs": []}))
    with pytest.raises(ValueError, match="letters"):
        campaign.load_spec(p)


def test_plan_one_job_per_run_or_segments():
    spec = campaign.load_spec(SPEC)
    jobs = campaign.plan_jobs(spec)
    assert [j["run_id"] for j in jobs][:2] == ["00_g2d_e020_s42", "01_g3d_env_dz200_s42"]
    assert all(j["stop_at_epoch"] is None and j["segments"] == 1 for j in jobs)
    seg = campaign.plan_jobs(spec | {"epochs_per_job": 6})
    assert len(seg) == 22
    assert [(j["segment"], j["stop_at_epoch"]) for j in seg[:2]] == [(0, 6), (1, 10)]
    assert all(len(j["job_name"]) <= 15 for j in seg)
    assert "53.6 GPU-h" in campaign.plan_text(spec)


# ------------------------------------------------------------------------------------------------ the qsub chain
def test_qsub_command_chains_with_afterok():
    spec = campaign.load_spec(SPEC) | {"queue": "QQ", "walltime": "02:00:00"}
    job = campaign.plan_jobs(spec)[1]
    first = campaign.qsub_command(job, spec, Path("/code"), "abc", Path("/r"), Path("/p"), None)
    chained = campaign.qsub_command(job, spec, Path("/code"), "abc", Path("/r"), Path("/p"), "123.pbs")
    assert "-W" not in first
    assert chained[chained.index("-W") + 1] == "depend=afterok:123.pbs"
    v = chained[chained.index("-v") + 1]
    assert "EXPECTED_COMMIT=abc" in v and "ARM=g3d_env_dz200" in v and "EPOCHS=10" in v and "STOP_AT_EPOCH" not in v
    assert chained[:5] == ["qsub", "-q", "QQ", "-l", "walltime=02:00:00"]
    assert chained[-1] == "/code/src/hepattn/experiments/clic/submit_gmp_run.sh"
    seg = campaign.plan_jobs(spec | {"epochs_per_job": 4})[0]
    seg_cmd = campaign.qsub_command(seg, spec, Path("/c"), "abc", Path("/r"), Path("/p"), None)
    assert "STOP_AT_EPOCH=4" in seg_cmd[seg_cmd.index("-v") + 1]
    with pytest.raises(ValueError, match="commas"):
        campaign.qsub_command(job, spec, Path("/a,b"), "abc", Path("/r"), Path("/p"), None)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A campaign directory under tmp_path, a pinned 'repo', and fake qsub/qstat."""
    monkeypatch.setattr(campaign, "CAMPAIGNS", tmp_path / "campaigns")
    submitted = []

    def fake_run(cmd, **_):
        if cmd[0] == "qsub":
            submitted.append(cmd)
            return f"{9000 + len(submitted)}.pbs"
        if cmd[:2] == ["git", "-C"] and "rev-parse" in cmd:
            return "c0ffee" * 6 + "c0ff"
        return ""

    monkeypatch.setattr(campaign, "run", fake_run)
    monkeypatch.setattr(campaign, "pin_code", lambda _repo, _name: (tmp_path / "snap", "c0ffee" * 6 + "c0ff"))
    states = {}
    monkeypatch.setattr(campaign, "qstat_state", lambda jid: states.get(jid, {}))
    spec = yaml.safe_load(SPEC.read_text()) | {"queue": "QQ", "walltime": "02:00:00", "runs": yaml.safe_load(SPEC.read_text())["runs"][:3]}
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p, submitted, states, tmp_path


def test_submit_refuses_unset_queue(sandbox, tmp_path):
    p, submitted, _, _ = sandbox
    spec = yaml.safe_load(p.read_text()) | {"queue": None}
    q = tmp_path / "noq.yaml"
    q.write_text(yaml.safe_dump(spec))
    with pytest.raises(SystemExit, match="unset"):
        campaign.submit(q, Path("/repo"), dry_run=False, resume=False)
    assert not submitted


def test_submit_chain_manifest_and_duplicate_protection(sandbox):
    p, submitted, states, tmp = sandbox
    campaign.submit(p, Path("/repo"), dry_run=False, resume=False)
    assert len(submitted) == 3
    assert "-W" not in submitted[0]
    assert submitted[1][submitted[1].index("-W") + 1] == "depend=afterok:9001.pbs"
    assert submitted[2][submitted[2].index("-W") + 1] == "depend=afterok:9002.pbs"
    man = json.loads((tmp / "campaigns/ablation_v1/manifest.json").read_text())
    assert [j["attempts"][0]["job_id"] for j in man["jobs"]] == ["9001.pbs", "9002.pbs", "9003.pbs"]
    assert len({j["run_dir"] for j in man["jobs"]}) == 3  # one directory per arm/seed
    with pytest.raises(SystemExit, match="already submitted"):
        campaign.submit(p, Path("/repo"), dry_run=False, resume=False)
    # resume refuses while any job of the campaign is still live
    states["9003.pbs"] = {"job_state": "H", "depend": "afterok:9002.pbs"}
    with pytest.raises(SystemExit, match="still queued"):
        campaign.submit(p, Path("/repo"), dry_run=False, resume=True)
    # after run 0 completed and run 1 failed (run 2 cancelled by PBS), resume re-chains runs 1 and 2 only
    states.clear()
    Path(man["jobs"][0]["run_dir"]).mkdir(parents=True)
    (Path(man["jobs"][0]["run_dir"]) / "DONE.json").write_text("{}")
    campaign.submit(p, Path("/repo"), dry_run=False, resume=True)
    assert len(submitted) == 5
    assert "-W" not in submitted[3]  # the first resubmitted job does not wait on the old failed one
    assert submitted[4][submitted[4].index("-W") + 1] == "depend=afterok:9004.pbs"
    man = json.loads((tmp / "campaigns/ablation_v1/manifest.json").read_text())
    assert [len(j["attempts"]) for j in man["jobs"]] == [1, 2, 2]


# ------------------------------------------------------------------------------------------------------- status
REAL_QSTAT = {
    "completed": "    job_state = F\n    stime = Sat Nov 14 18:42:57 2026\n    Exit_status = 0\n    substate = 92\n",
    "failed": "    job_state = F\n    stime = Sat Nov 14 18:42:58 2026\n    Exit_status = 1\n    substate = 92\n",
    "cancelled": "    job_state = F\n    depend = afterok:7848493.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov@polaris\n    substate = 92\n",
    "waiting": "    job_state = H\n    depend = afterok:7848490.polaris-pbs-01@polaris\n",
    "running": "    job_state = R\n    stime = Sat Nov 14 18:42:57 2026\n",
    "queued": "    job_state = Q\n",
}


@pytest.mark.parametrize("label", list(REAL_QSTAT))
def test_classify_real_qstat_records(label):
    got = campaign.classify(campaign.parse_qstat(REAL_QSTAT[label]), finished=label == "completed")
    assert got.lower().startswith(label if label != "waiting" else "waiting"), got


def test_exit_zero_without_done_marker_is_not_completed():
    """A job that exited 0 but left no DONE/segment marker is not counted as completed."""
    assert campaign.classify(campaign.parse_qstat(REAL_QSTAT["completed"]), finished=False).startswith("FAILED")


# ---------------------------------------------------------------------------------------------- config preflight
def _resolved(**over):
    cfg = {
        "seed_everything": 42,
        "trainer": {
            "devices": 4,
            "num_nodes": 1,
            "accumulate_grad_batches": 1,
            "precision": "bf16-mixed",
            "strategy": "auto",
            "max_epochs": 10,
            "callbacks": [{"class_path": c} for c in preflight.REQUIRED_CALLBACKS],
        },
        "data": {"batch_size": 256, "num_train": -1, "num_val": 5000},
        "model": {
            "model": {
                "init_args": {
                    "encoder": {"init_args": {"gmp": {"kernel_size": 3}, "gmp_layers": None, "gmp_impl": "weightsfirst"}},
                    "input_coord_fields": ["eta_calo", "phi_calo"],
                }
            }
        },
    }
    for path, value in over.items():
        cur = cfg
        keys = path.split("__")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    return cfg


ARM = {
    "model": {
        "model": {
            "init_args": {
                "encoder": {"init_args": {"gmp": {"kernel_size": 3}, "gmp_layers": None, "gmp_impl": "weightsfirst"}},
                "input_coord_fields": ["eta_calo", "phi_calo"],
            }
        }
    }
}


def test_preflight_accepts_the_required_setup():
    assert preflight.check(_resolved(), ARM, epochs=10, seed=42) == []


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"trainer__devices": 1}, "trainer.devices"),
        ({"trainer__accumulate_grad_batches": 4}, "accumulate_grad_batches"),
        ({"data__batch_size": 512}, "global batch"),
        ({"trainer__num_nodes": 2}, "num_nodes"),
        ({"trainer__precision": "32-true"}, "precision"),
        ({"trainer__strategy": "deepspeed"}, "strategy"),
        ({"trainer__max_epochs": 3}, "max_epochs"),
        ({"seed_everything": 7}, "seed_everything"),
        ({"data__num_train": 1000}, "num_train"),
    ],
)
def test_preflight_rejects_wrong_execution(over, match):
    bad = preflight.check(_resolved(**over), ARM, epochs=10, seed=42)
    assert any(match in b for b in bad), bad


def test_preflight_rejects_pilot_style_and_missing_guard():
    pilot = _resolved(trainer__devices=1, trainer__accumulate_grad_batches=4)  # the retired 1 GPU x accumulate 4 pilot
    assert len(preflight.check(pilot, ARM, epochs=10, seed=42)) >= 2
    cfg = _resolved()
    cfg["trainer"]["callbacks"] = [c for c in cfg["trainer"]["callbacks"] if "DDPGuard" not in c["class_path"]]
    assert any("DDPGuard" in b for b in preflight.check(cfg, ARM, epochs=10, seed=42))


def test_preflight_checks_the_arm():
    arm = copy.deepcopy(ARM)
    arm["model"]["model"]["init_args"]["encoder"]["init_args"]["gmp_layers"] = [0]
    assert any("gmp_layers" in b for b in preflight.check(_resolved(), arm, epochs=10, seed=42))
    arm3 = copy.deepcopy(ARM) | {"data": {"gmp_depth": "envelope"}}
    assert any("data.gmp_depth" in b for b in preflight.check(_resolved(), arm3, epochs=10, seed=42))


def test_shared_config_sets_ddp_execution():
    common = yaml.safe_load((CLIC / "configs/gmp_ablation/sweep/common.yaml").read_text())
    assert (common["trainer"]["devices"], common["trainer"]["num_nodes"], common["trainer"]["accumulate_grad_batches"]) == (4, 1, 1)
    assert common["data"]["batch_size"] == 256 and common["trainer"]["precision"] == "bf16-mixed"
    paths = [c["class_path"] for c in common["trainer"]["callbacks"]]
    assert all(c in paths for c in preflight.REQUIRED_CALLBACKS)
    assert "hepattn.experiments.clic.gmp_ablation.segment.SegmentStop" in paths
    ckpt = next(c for c in common["trainer"]["callbacks"] if c["class_path"] == "hepattn.callbacks.Checkpoint")
    assert ckpt["init_args"]["log_model"] is False


# ------------------------------------------------------------------------------------------------- run verdict
@pytest.mark.parametrize(
    ("diag", "stop", "code"),
    [
        ({"epochs": [{"epoch": e} for e in range(10)], "health": {"nonfinite_loss_batches": 0, "nonfinite_grad_steps": 0}}, None, 0),
        ({"epochs": [{"epoch": e} for e in range(6)], "health": {"nonfinite_loss_batches": 0, "nonfinite_grad_steps": 0}}, 6, 0),
        ({"epochs": [{"epoch": e} for e in range(6, 10)], "health": {"nonfinite_loss_batches": 0, "nonfinite_grad_steps": 0}}, None, 0),
        ({"epochs": [{"epoch": e} for e in range(4)], "health": {"nonfinite_loss_batches": 0, "nonfinite_grad_steps": 0}}, None, 4),
        ({"epochs": [{"epoch": e} for e in range(10)], "health": {"nonfinite_loss_batches": 0, "nonfinite_grad_steps": 2}}, None, 3),
        ({"epochs": [{"epoch": e} for e in range(10)], "health": {"nonfinite_loss_batches": 1, "nonfinite_grad_steps": 0}}, None, 3),
        ({"epochs": [{"epoch": e} for e in range(10)]}, None, 4),
        ({}, None, 4),
    ],
)
def test_runcheck_verdict(diag, stop, code):
    assert runcheck.verdict(diag, epochs=10, stop_at=stop)[0] == code
