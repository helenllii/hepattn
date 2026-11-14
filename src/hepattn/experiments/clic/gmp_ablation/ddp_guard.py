"""DDPGuard: fail a GMP ablation run before its first step unless it is ONE model on 4 GPUs of one node via DDP.

Runs in `setup("fit")`, after Lightning has initialised the process group and before any training step:
  * trainer.world_size == 4, num_devices == 4, num_nodes == 1, accumulate_grad_batches == 1, a DDP strategy,
    bf16-mixed precision, datamodule batch_size == 256, so the global batch is 256 x 4 x 1 = 1024;
  * inside the live process group: an all_reduce of 1 per rank must give 4, and the ranks' CUDA device UUIDs must
    be 4 distinct GPUs (all four take part in the same distributed run, not four copies on one device).
Rank 0 writes what it verified to $GMP_RUN_DIR/ddp_preflight.json (world size, ranks, local ranks, hosts, GPU
names and UUIDs, strategy, precision, batch). Any mismatch raises RuntimeError on every rank.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from lightning import Callback


class DDPGuard(Callback):
    def __init__(self, world_size: int = 4, micro_batch: int = 256, global_batch: int = 1024, accumulate: int = 1) -> None:
        super().__init__()
        self.world_size = world_size
        self.micro_batch = micro_batch
        self.global_batch = global_batch
        self.accumulate = accumulate

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "fit":
            return
        bad = []
        strategy = type(trainer.strategy).__name__
        batch = getattr(trainer.datamodule, "batch_size", None) if trainer.datamodule is not None else None
        checks = {
            "trainer.world_size": (trainer.world_size, self.world_size),
            "trainer.num_devices": (trainer.num_devices, self.world_size),
            "trainer.num_nodes": (trainer.num_nodes, 1),
            "trainer.accumulate_grad_batches": (trainer.accumulate_grad_batches, self.accumulate),
            "datamodule.batch_size": (batch, self.micro_batch),
            "trainer.precision": (str(trainer.precision), "bf16-mixed"),
        }
        bad += [f"{k} = {got!r}, required {want!r}" for k, (got, want) in checks.items() if got != want]
        if "DDP" not in strategy:
            bad.append(f"strategy {strategy}, required a DDP strategy")
        if (batch or 0) * trainer.world_size * trainer.accumulate_grad_batches != self.global_batch:
            bad.append(f"global batch {(batch or 0) * trainer.world_size * trainer.accumulate_grad_batches}, required {self.global_batch}")
        record = {
            "strategy": strategy,
            "precision": str(trainer.precision),
            "micro_batch": batch,
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
            "world_size": trainer.world_size,
        }
        if not (dist.is_available() and dist.is_initialized()):
            bad.append("torch.distributed is not initialised: not a DDP run")
        else:
            dev = torch.device("cuda", torch.cuda.current_device())
            one = torch.ones(1, device=dev)
            dist.all_reduce(one)
            props = torch.cuda.get_device_properties(dev)
            mine = {
                "rank": dist.get_rank(),
                "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
                "host": socket.gethostname(),
                "cuda_device": dev.index,
                "gpu": props.name,
                "uuid": str(getattr(props, "uuid", "")),
            }
            ranks: list = [None] * dist.get_world_size()
            dist.all_gather_object(ranks, mine)
            record |= {"all_reduce_sum": int(one.item()), "process_group_size": dist.get_world_size(), "ranks": ranks}
            if int(one.item()) != self.world_size:
                bad.append(f"all_reduce over the process group gives {int(one.item())}, required {self.world_size}")
            uuids = {r["uuid"] for r in ranks}
            if len(uuids) != self.world_size:
                bad.append(f"{len(uuids)} distinct GPUs in the process group, required {self.world_size}: {ranks}")
            if len({r["host"] for r in ranks}) != 1:
                bad.append(f"ranks on several hosts: {sorted({r['host'] for r in ranks})}")
        record["violations"] = bad
        if trainer.is_global_zero and os.environ.get("GMP_RUN_DIR"):
            Path(os.environ["GMP_RUN_DIR"], "ddp_preflight.json").write_text(json.dumps(record, indent=1))
        if bad:
            raise RuntimeError("DDPGuard: this run is not 4-GPU single-node DDP at accumulate 1:\n  " + "\n  ".join(bad))
        if trainer.is_global_zero:
            print(f"[ddp-guard] ok: {json.dumps({k: v for k, v in record.items() if k != 'ranks'})}", flush=True)
