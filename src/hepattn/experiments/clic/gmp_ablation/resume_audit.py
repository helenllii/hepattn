"""ResumeAudit: record the training state a job STARTS from, so a resume can be checked against its checkpoint.

At on_train_start, after Lightning has restored a checkpoint (if any) and before the first step, rank 0 writes
$GMP_DIAG_OUT.resume_audit.json:
  * global_step, current_epoch;
  * float64 sums of every model state_dict tensor and of every optimizer-state tensor (weights / Lion moments: a restored state
    must reproduce the checkpoint's sums exactly);
  * the LR scheduler's state (last_epoch, _step_count, total_steps, _last_lr) and the optimizer's current learning rate.
Then, at every training-epoch end, rank 0 appends (epoch, global_step, lr) to the same record. Cheap: one pass over the
parameters at train start; nothing per step.

gmp_ablation/resume_check.py compares this record with the previous segment's last checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from lightning import Callback


def _tensor_sum(obj) -> float:
    if isinstance(obj, torch.Tensor):
        return float(obj.detach().double().sum().item()) if obj.is_floating_point() else 0.0
    if isinstance(obj, dict):
        return sum(_tensor_sum(v) for v in obj.values())
    if isinstance(obj, list | tuple):
        return sum(_tensor_sum(v) for v in obj)
    return 0.0


def scheduler_state(sched) -> dict:
    keep = ("last_epoch", "_step_count", "total_steps", "_last_lr", "base_lrs", "_schedule_phases", "_anneal_func_type")
    return {k: v for k, v in sched.state_dict().items() if k in keep}


class ResumeAudit(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._rec: dict | None = None

    def _path(self, trainer) -> Path | None:
        base = os.environ.get("GMP_DIAG_OUT")
        return Path(f"{base}.resume_audit.json") if base and trainer.is_global_zero else None

    def on_train_start(self, trainer, pl_module) -> None:
        opt = trainer.optimizers[0]
        scheds = [c.scheduler for c in trainer.lr_scheduler_configs]
        self._rec = {
            "global_step": trainer.global_step,
            "current_epoch": trainer.current_epoch,
            "ckpt_path": str(trainer.ckpt_path) if trainer.ckpt_path else None,
            "param_sum": _tensor_sum(pl_module.state_dict()),  # the same tensors as a checkpoint's state_dict
            "optimizer_state_sum": _tensor_sum(opt.state_dict()["state"]),
            "optimizer_lr": [g["lr"] for g in opt.param_groups],
            "schedulers": [scheduler_state(s) for s in scheds],
            "epochs": [],
        }
        self._write(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._rec is None:
            return
        self._rec["epochs"].append({
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "lr": [g["lr"] for g in trainer.optimizers[0].param_groups],
        })
        self._write(trainer)

    def _write(self, trainer) -> None:
        path = self._path(trainer)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._rec, indent=1, default=str))
