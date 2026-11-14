"""DDPGuard refuses anything but 4-GPU single-node DDP at accumulate 1; SegmentStop ends a job at an epoch boundary.

CPU-only checks with stand-in trainer objects. The positive path of DDPGuard (a live 4-rank process group, 4 distinct
GPUs) is exercised by the DDP smoke test (submit_gmp_smoke.sh), not here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hepattn.experiments.clic.gmp_ablation.ddp_guard import DDPGuard
from hepattn.experiments.clic.gmp_ablation.segment import SegmentStop


class DDPStrategy:  # stand-in: DDPGuard only looks at the strategy's class name
    pass


class SingleDeviceStrategy:
    pass


def _trainer(**over):
    t = SimpleNamespace(
        world_size=4,
        num_devices=4,
        num_nodes=1,
        accumulate_grad_batches=1,
        precision="bf16-mixed",
        strategy=DDPStrategy(),
        datamodule=SimpleNamespace(batch_size=256),
        is_global_zero=True,
    )
    for k, v in over.items():
        setattr(t, k, v)
    return t


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"world_size": 1, "num_devices": 1}, "world_size"),
        ({"accumulate_grad_batches": 4}, "accumulate_grad_batches"),
        ({"datamodule": SimpleNamespace(batch_size=512)}, "batch_size"),
        ({"num_nodes": 2}, "num_nodes"),
        ({"precision": "32-true"}, "precision"),
        ({"strategy": SingleDeviceStrategy()}, "DDP strategy"),
    ],
)
def test_guard_rejects_wrong_setups(over, match):
    with pytest.raises(RuntimeError, match=match):
        DDPGuard().setup(_trainer(**over), None, "fit")


def test_guard_rejects_without_a_process_group():
    """Even a correct-looking config fails when torch.distributed is not initialised: it is not a DDP run."""
    with pytest.raises(RuntimeError, match="not initialised"):
        DDPGuard().setup(_trainer(), None, "fit")


def test_guard_ignores_other_stages():
    DDPGuard().setup(_trainer(world_size=1), None, "validate")


def test_segment_stop(monkeypatch):
    monkeypatch.setenv("GMP_STOP_AT_EPOCH", "3")
    cb = SegmentStop()
    for epoch, stop in ((0, False), (1, False), (2, True)):
        t = SimpleNamespace(current_epoch=epoch, should_stop=False)
        cb.on_train_epoch_end(t, None)
        assert t.should_stop is stop
    monkeypatch.setenv("GMP_STOP_AT_EPOCH", "")
    t = SimpleNamespace(current_epoch=99, should_stop=False)
    SegmentStop().on_train_epoch_end(t, None)
    assert t.should_stop is False  # unset: a no-op
    monkeypatch.setenv("GMP_STOP_AT_EPOCH", "0")
    with pytest.raises(ValueError, match=">= 1"):
        SegmentStop()
