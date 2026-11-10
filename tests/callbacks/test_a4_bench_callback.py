"""Gates for the throughput benchmark callback itself.

The callback is the instrument: if its arithmetic or its stopping condition is wrong, the Polaris
numbers are wrong and nothing else in the run would notice. These tests drive it through the real
Lightning training loop on CPU with a trivial model, so the hook wiring, the warmup/measure
accounting, the global-batch guard and the early stop are all exercised for real.
"""

# The tests deliberately read the callback's measurement state: that state IS the deliverable,
# and asserting on the printed report alone would not catch a mis-sized window.
# ruff: noqa: SLF001
from __future__ import annotations

import json

import pytest
import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from hepattn.experiments.clic.a4.bench_callback import ThroughputBenchmark

BATCH = 4
STEPS_AVAILABLE = 400


class _Tiny(LightningModule):
    """The smallest thing that still produces a real optimizer step."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(3, 1)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        return self.net(x).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


def _loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(STEPS_AVAILABLE * BATCH, 3)), batch_size=BATCH)


def _run(accumulate: int = 1, **kwargs) -> tuple[ThroughputBenchmark, Trainer]:
    cb = ThroughputBenchmark(util_poll_s=0, **kwargs)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=accumulate,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(_Tiny(), train_dataloaders=_loader())
    return cb, trainer


def test_stops_the_run_after_its_window():
    """It must end the job itself: a benchmark that trains on is a benchmark nobody waits for."""
    cb, trainer = _run(warmup_steps=5, measure_steps=10)
    assert cb._reported
    assert trainer.should_stop
    assert len(cb._step_s) == 10
    # Stopping is checked at the epoch/batch boundary, so one extra batch may be consumed.
    assert 15 <= cb._seen <= 16, cb._seen


def test_warmup_steps_are_excluded_from_the_window():
    cb, _ = _run(warmup_steps=20, measure_steps=10)
    assert len(cb._step_s) == 10
    # The first measured step has no predecessor inside the window, so there is one fewer wait.
    assert len(cb._wait_s) == 9


def test_zero_warmup_is_allowed():
    cb, _ = _run(warmup_steps=0, measure_steps=5)
    assert len(cb._step_s) == 5


def test_rejects_a_changed_global_batch():
    """The 1-GPU vs 4-GPU comparison is void unless the global batch is held fixed."""
    with pytest.raises(ValueError, match="effective global batch is 8 "):
        _run(accumulate=2, warmup_steps=0, measure_steps=2, expect_global_batch=2048)


def test_accepts_the_matching_global_batch():
    cb, _unused = _run(accumulate=4, warmup_steps=0, measure_steps=3, expect_global_batch=BATCH * 4)
    assert cb._global_batch == BATCH * 4


def test_reports_the_parseable_json_line(capsys):
    _run(warmup_steps=3, measure_steps=8, full_train_events=1000, tag="unit")
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("A4BENCH_JSON ")]
    assert len(lines) == 1, out[-2000:]
    payload = json.loads(lines[0][len("A4BENCH_JSON ") :])

    assert payload["tag"] == "unit"
    assert payload["steps"] == 8
    assert payload["world_size"] == 1
    assert payload["micro_batch"] == BATCH
    assert payload["global_batch"] == BATCH
    assert payload["full_train_events"] == 1000
    # events/s is the per-rank rate times the world size, and the epoch estimate inverts it.
    assert payload["events_s_global"] == pytest.approx(payload["events_s_per_rank"], rel=1e-6)
    # abs= covers the JSON line's own rounding quantum, which dominates at this toy scale.
    assert payload["epoch_min"] == pytest.approx(1000 / payload["events_s_global"] / 60, rel=1e-2, abs=1e-4)
    assert payload["it_s_per_rank"] > 0
    assert 0.0 <= payload["dataloader_wait_frac"] <= 1.0
    assert payload["step_ms_p10"] <= payload["step_ms_p50"] <= payload["step_ms_p90"]


def test_report_names_every_requested_quantity(capsys):
    """Each of the five things the benchmark was asked for must appear in the printed report."""
    _run(warmup_steps=2, measure_steps=5, full_train_events=1000)
    out = capsys.readouterr().out
    for expected in (
        "1. STEADY-STATE RATE",
        "2. FULL-EPOCH ESTIMATE",
        "3. PEAK GPU MEMORY",
        "4. BOTTLENECK",
        "events/s",
        "epoch time",
        "reserved",
        "dataloader wait",
        "gpu util",
        "global batch",
    ):
        assert expected in out, expected


@pytest.mark.parametrize(("warmup", "measure"), [(-1, 10), (0, 0)])
def test_rejects_a_nonsense_window(warmup, measure):
    with pytest.raises(ValueError, match="need warmup_steps"):
        ThroughputBenchmark(warmup_steps=warmup, measure_steps=measure)
