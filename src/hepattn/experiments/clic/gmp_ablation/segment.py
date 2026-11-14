"""SegmentStop: end a training JOB cleanly at an epoch boundary without changing the RUN's schedule.

A run's budget is trainer.max_epochs (the OneCycle schedule spans it, step for step). When a run is split into several
shorter PBS jobs (segments), each job resumes from the previous segment's last epoch checkpoint and stops after epoch
$GMP_STOP_AT_EPOCH - 1 (0-based), i.e. once `current_epoch + 1 >= GMP_STOP_AT_EPOCH`. The learning rate, optimizer
state, epoch counter and data order then continue exactly where they would have in one uninterrupted job.

Unset (the default): a no-op, the job runs to max_epochs. Rank-consistent: every rank reads the same environment.
"""

from __future__ import annotations

import os

from lightning import Callback


class SegmentStop(Callback):
    def __init__(self) -> None:
        super().__init__()
        raw = os.environ.get("GMP_STOP_AT_EPOCH", "")
        self.stop_at = int(raw) if raw.strip() else None
        if self.stop_at is not None and self.stop_at < 1:
            raise ValueError(f"GMP_STOP_AT_EPOCH must be >= 1, got {raw!r}")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.stop_at is not None and trainer.current_epoch + 1 >= self.stop_at:
            trainer.should_stop = True
