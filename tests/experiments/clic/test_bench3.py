"""CPU tests for the bench3 tooling: import shim, timing callback (window, walltime guard, staged results, breakdown),
summary tables, and the config arms. Run inside the Round-0 PBS job, not on a login node.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml
from lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from hepattn.experiments.clic.bench3 import TrainingTimeBenchmark
from hepattn.experiments.clic.bench3 import summarize as summ

CLIC = Path(__file__).resolve().parents[3] / "src" / "hepattn" / "experiments" / "clic"
SHIM = CLIC / "bench3" / "shim"
PHAT_ARMS = ["phat16", "phat_gmp_einsumopt_early", "phat16_gmp_all_production", "phat16_gmp_all_einsumopt", "phat16_gmp_early_production"]
GMP_KEYS = {"gmp", "gmp_layers", "gmp_impl"}


@pytest.fixture(autouse=True)
def _no_job_env(monkeypatch):
    for k in ("BENCH3_DEADLINE", "BENCH3_RESULT", "BENCH3_PROFILE", "BENCH3_PROFILE_TRACE"):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------------------------------------- shim
def _fake_src(tmp_path: Path) -> Path:
    pkg = tmp_path / "src" / "hepattn"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("MARK = 'bench'\n")
    (pkg / "sub" / "__init__.py").write_text("")
    return tmp_path / "src"


def _run(code: str, src: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "HEPATTN_SRC": str(src), "PYTHONPATH": str(SHIM)}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)


def test_shim_redirects_hepattn_to_src(tmp_path):
    src = _fake_src(tmp_path)
    r = _run("import hepattn, hepattn.sub; print(hepattn.MARK, hepattn.__file__)", src)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "bench"
    assert r.stdout.split()[1].startswith(str(src))


def test_shim_is_strict_for_missing_modules(tmp_path):
    r = _run("import hepattn.models", _fake_src(tmp_path))
    assert r.returncode != 0
    assert "refusing to fall back" in r.stderr


def test_shim_leaves_other_packages_alone(tmp_path):
    r = _run("import json, torch; print('ok')", _fake_src(tmp_path))
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------------------------ callback
class TinyModule(LightningModule):
    def __init__(self, on_batch=None):
        super().__init__()
        self.name = "tiny"
        self.lin = torch.nn.Linear(4, 1)
        self.val_batches = 0
        self.on_batch = on_batch

    def training_step(self, batch, batch_idx):
        if self.on_batch is not None:
            self.on_batch(self.trainer.global_step)
        (x,) = batch
        return self.lin(x).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        self.val_batches += 1
        (x,) = batch
        return self.lin(x).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


class TinyData(LightningDataModule):
    def __init__(self, n_train=400, n_val=40, batch_size=8, num_train=-1):
        super().__init__()
        self.batch_size, self.num_train, self.num_val, self.train_path = batch_size, num_train, n_val, "train.root"
        self.tr = TensorDataset(torch.randn(n_train, 4))
        self.va = TensorDataset(torch.randn(n_val, 4))

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.va, batch_size=self.batch_size)


class Recorder(TrainingTimeBenchmark):
    """Records every status written, in order."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.statuses: list[str] = []

    def _write(self, trainer, res, status):
        self.statuses.append(status)
        super()._write(trainer, res, status)


def _fit(tmp_path, accum=1, num_train=-1, full_train_events=None, on_batch=None, cb_cls=Recorder, **cb_kwargs):
    cb_kwargs.setdefault("result_path", str(tmp_path / "res.json"))
    cb = cb_cls(warmup_batches=3, measure_batches=6, full_train_events=full_train_events, mem_poll_s=0.05, **cb_kwargs)
    model, dm = TinyModule(on_batch), TinyData(num_train=num_train)
    trainer = Trainer(
        accelerator="cpu", devices=1, max_epochs=5, accumulate_grad_batches=accum, callbacks=[cb], logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )  # fmt: skip
    trainer.fit(model, datamodule=dm)
    return cb, model, trainer


def _file(tmp_path) -> dict:
    return json.loads((tmp_path / "res.json").read_text())


def test_callback_window_validation_and_stop(tmp_path, capsys):
    cb, model, trainer = _fit(tmp_path)
    r = cb.result
    # stopped right after warmup + window (Lightning still bumps current_epoch on stop; the step count is the evidence)
    assert r["measured_batches"] == 6
    assert trainer.global_step == 9
    assert model.val_batches == 5 and r["val_batches_per_rank"] == 5 and r["val_status"] == "done"
    assert r["train_batches_per_epoch_per_rank"] == 50
    assert r["optimizer_steps_per_epoch"] == 50
    assert r["epoch_train_s"] == pytest.approx(50 * r["s_per_batch"])
    assert r["epoch_total_s"] == pytest.approx(r["epoch_train_s"] + r["epoch_val_s_steady"])
    assert r["optimizer_steps_per_s"] == pytest.approx(1 / r["s_per_batch"])
    # separately timed, never in the rate
    assert r["startup_s"] > 0 and r["first_batch_s_incl_compile"] > 0 and r["warmup_rest_s"] >= 0
    # staged writes: partial (after warmup), train_done BEFORE validation, complete after
    assert cb.statuses[0] == "partial" and cb.statuses[-2:] == ["train_done", "complete"]
    assert _file(tmp_path)["status"] == "complete"
    cpu = r["cpu"]
    assert cpu["torch_num_threads"] == torch.get_num_threads() and "env" in cpu and "oversubscribed_during_matching" in cpu
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("BENCH3_JSON "))
    payload = json.loads(line.removeprefix("BENCH3_JSON "))  # strict JSON: no NaN on CPU
    assert payload["gpu_peak_reserved_gib"] is None


def test_callback_accumulation_rounds_window_and_counts_steps(tmp_path):
    cb, _, _ = _fit(tmp_path, accum=4)
    r = cb.result
    assert r["measured_batches"] == 8  # 6 rounded up to whole optimizer steps
    assert r["optimizer_steps_per_epoch"] == 13  # ceil(50 / 4)
    assert r["optimizer_steps_per_s"] == pytest.approx(1 / (4 * r["s_per_batch"]))


def test_callback_extrapolates_truncated_epoch(tmp_path):
    cb, _, _ = _fit(tmp_path, num_train=400, full_train_events=1000)
    assert cb.result["train_batches_per_epoch_per_rank"] == 125  # ceil(1000 / 8)
    assert "extrapolated" in cb.result["batches_per_epoch_source"]


def test_callback_rejects_wrong_global_batch(tmp_path):
    with pytest.raises(ValueError, match="effective global batch"):
        _fit(tmp_path, expect_global_batch=1024)


def test_deadline_in_warmup_writes_partial_and_skips_validation(tmp_path):
    cb, model, trainer = _fit(tmp_path, deadline=time.time() - 1, guard_every=1)
    assert trainer.global_step == 1 and model.val_batches == 0
    assert _file(tmp_path)["status"] == "truncated_in_warmup"
    assert _file(tmp_path)["first_batch_s_incl_compile"] > 0


def test_deadline_mid_window_truncates_and_keeps_rates(tmp_path):
    holder = {}

    def expire(step):
        if step == 6:  # warmup 3 + 3 measured
            holder["cb"].deadline = time.time() - 1

    cb = Recorder(warmup_batches=3, measure_batches=6, guard_every=1, deadline=time.time() + 1e6, result_path=str(tmp_path / "res.json"))
    holder["cb"] = cb
    model, dm = TinyModule(expire), TinyData()
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=5, callbacks=[cb], logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=tmp_path)  # fmt: skip
    trainer.fit(model, datamodule=dm)
    r = _file(tmp_path)
    assert r["status"] == "truncated" and r["window_truncated"] and 1 <= r["measured_batches"] < 6
    assert r["s_per_batch"] > 0 and not r["window_reliable"]
    assert model.val_batches == 0 and r["val_status"].startswith("skipped")


def test_validation_skipped_when_time_is_short(tmp_path):
    cb, model, _ = _fit(tmp_path, deadline=time.time() + 3600, val_reserve_s=1e7)
    assert model.val_batches == 0
    assert cb.result["val_status"].startswith("skipped")
    assert _file(tmp_path)["status"] == "complete" and not cb.result["epoch_total_includes_val"]


def test_breakdown_and_profiler_after_window(tmp_path):
    cb, model, trainer = _fit(tmp_path, profile_batches=3, profiler_batches=2)
    r = cb.result
    assert trainer.global_step == 3 + 6 + 3 + 2
    assert r["measured_batches"] == 6  # the breakdown never enters the timed window
    bd = r["breakdown"]
    assert bd["instrumented_batches"] == 3
    assert bd["rank0"]["fwd_loss_host_ms"] > 0 and bd["rank0"]["bwd_host_ms"] >= 0
    assert bd["profiler"]["batches"] == 2 and Path(bd["profiler"]["table_path"]).is_file()
    assert r["profile_status"] == "done" and r["val_status"] == "done" and model.val_batches == 5


# ------------------------------------------------------------------------------------------------------ summary
def _result(tag, s_batch, val_s=10.0, **kw):
    r = {
        "tag": tag, "status": "complete", "s_per_batch": s_batch, "batch_p90_s": s_batch, "optimizer_steps_per_s": 1 / s_batch,
        "train_batches_per_epoch_per_rank": 972, "epoch_train_s": 972 * s_batch, "epoch_val_s_steady": val_s,
        "epoch_total_s": 972 * s_batch + val_s, "trainable_params": 1e6, "gpu_peak_reserved_gib": 10.0,
        "host_job_pss_gib_total": 50.0, "dataloader_wait_frac": 0.01, "world_size": 4, "micro_batch": 256, "accumulate": 1,
        "global_batch": 1024, "precision": "bf16-mixed", "num_workers": 7, "device": "A100", "batches_per_epoch_source": "exact",
        "cpu": {"torch_num_threads": 8, "matcher_n_jobs": 8, "affinity_cpus": 64},
    }  # fmt: skip
    r.update(kw)
    return r


def test_summary_speedup_relative_to_linformer(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_result("bench3_linformer_encoder", 0.4)))
    (tmp_path / "b.json").write_text(json.dumps(_result("bench3_phat_gmp_einsumopt_early", 0.2)))
    res = summ.load([str(tmp_path)])
    text = summ.table_epoch(res, summ.LINFORMER, markdown=False)
    phat = next(ln for ln in text.splitlines() if ln.startswith("PHAT16+GMP einsumopt/early"))
    assert "2.00x" in phat
    assert not summ.checks(res, summ.LINFORMER)


def test_summary_flags_mismatched_setup(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_result("bench3_linformer_encoder", 0.4)))
    (tmp_path / "b.json").write_text(json.dumps(_result("bench3_phat16", 0.3, global_batch=2048, cpu={"torch_num_threads": 4})))
    notes = summ.checks(summ.load([str(tmp_path)]), summ.LINFORMER)
    assert any("global_batch differs" in n for n in notes)
    assert any("torch_num_threads differs" in n for n in notes)


def test_summary_grid_compares_one_factor_at_a_time(tmp_path):
    for tag, s in [("bench3_phat16_gmp_all_production", 0.6), ("bench3_phat16_gmp_all_einsumopt", 0.5),
                   ("bench3_phat16_gmp_early_production", 0.45), ("bench3_phat_gmp_einsumopt_early", 0.44)]:  # fmt: skip
        (tmp_path / f"{tag}.json").write_text(json.dumps(_result(tag, s)))
    grid = summ.table_grid(summ.load([str(tmp_path)]), markdown=False)
    assert "1.200x" in grid  # impl effect at all-6 placement: 0.6 / 0.5


# ------------------------------------------------------------------------------------------------------ configs
def _load(rel: str) -> dict:
    return yaml.safe_load((CLIC / "configs" / rel).read_text())


def test_phat_arm_model_block_matches_validated_a4_arm():
    assert _load("bench3/phat_gmp_einsumopt_early.yaml")["model"] == _load("a4/bench_d256_einsumopt_early.yaml")["model"]


def test_arms_change_only_the_model():
    for f in (CLIC / "configs" / "bench3").glob("*.yaml"):
        if f.stem != "common":
            assert set(yaml.safe_load(f.read_text())) <= {"name", "model"}, f.name


@pytest.mark.parametrize("arm", PHAT_ARMS)
def test_phat_arms_differ_only_in_gmp_keys(arm):
    ref = _load("bench3/phat_gmp_einsumopt_early.yaml")["model"]["model"]["init_args"]
    cfg = _load(f"bench3/{arm}.yaml")["model"]["model"]["init_args"]
    assert {k: v for k, v in cfg.items() if k not in {"encoder", "input_coord_fields"}} == {
        k: v for k, v in ref.items() if k not in {"encoder", "input_coord_fields"}
    }
    assert cfg["encoder"]["class_path"] == ref["encoder"]["class_path"]
    e, er = cfg["encoder"]["init_args"], ref["encoder"]["init_args"]
    assert {k: v for k, v in e.items() if k not in GMP_KEYS} == {k: v for k, v in er.items() if k not in GMP_KEYS}
    assert ("input_coord_fields" in cfg) == (e.get("gmp") is not None)


def test_round3_grid_arms():
    want = {"phat16_gmp_all_production": (None, "production"), "phat16_gmp_all_einsumopt": (None, "einsumopt"),
            "phat16_gmp_early_production": ([0], "production"), "phat_gmp_einsumopt_early": ([0], "einsumopt")}  # fmt: skip
    for arm, (layers, impl) in want.items():
        e = _load(f"bench3/{arm}.yaml")["model"]["model"]["init_args"]["encoder"]["init_args"]
        assert (e["gmp_layers"], e["gmp_impl"]) == (layers, impl) and e["gmp"] is not None, arm
    assert _load("bench3/phat16.yaml")["model"]["model"]["init_args"]["encoder"]["init_args"]["gmp"] is None


def test_linformer_arm_is_encoder_only():
    cfg = _load("bench3/linformer_encoder.yaml")["model"]["model"]["init_args"]
    assert cfg["encoder"]["init_args"]["attn_type"] == "linformer"
    assert "decoder" not in cfg
