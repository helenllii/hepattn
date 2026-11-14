"""GMPDiagnostics: what a full-model GMP run actually executes, without perturbing the compiled step being timed.

Appended to a bench3 run with
    --trainer.callbacks+=hepattn.experiments.clic.gmp_ablation.diagnostics.GMPDiagnostics
it records, per rank, into $GMP_DIAG_OUT (or <default_root_dir>/gmp_diagnostics.json) with suffix .rank<r>.json:

* parameter counts: model total, encoder only, decoder only, and the GMP module of every patch block (each block owns
  its own GMP; nothing is shared), plus the GMP class each block really runs (production / weights-first / ...);
* the dtypes that reach the encoder and every GMP block, and the neighbourhood statistics of the first GMP block, from
  ONE extra eager forward of training batch 0 under `torch.compiler.set_stance("force_eager")`. The hooks exist only for
  that call and are removed before the first real step, so they can never be traced into, guard, or recompile the
  compiled graph; the extra forward falls inside the benchmark's warmup;
* numerical health of every training step: non-finite loss count, non-finite pre-clip gradient count, and the largest
  finite pre-clip global gradient norm. Accumulated on the GPU without a host sync per step (one foreach-norm per
  step, read back once at the end), and paid identically by every arm;
* per training epoch (sweeps): training wall time with the validation loop subtracted (Lightning runs it inside the
  training epoch), validation time, optimizer steps, steps/s and peak GPU memory (allocated and reserved).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from lightning import Callback


def _numel(module: torch.nn.Module | None) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad) if module is not None else 0


class GMPDiagnostics(Callback):
    def __init__(self, out_path: str | None = None) -> None:
        super().__init__()
        self.out_path = out_path
        self._probed = False
        self._res: dict = {}
        self._health: torch.Tensor | None = None  # [steps, nonfinite_loss, nonfinite_grad, max_finite_gnorm]
        self._epoch_t0: float | None = None
        self._epoch_step0 = 0
        self._val_s = 0.0
        self._val_t0: float | None = None
        self._res["epochs"] = []

    # ------------------------------------------------------------------------------------------- parameters
    def on_train_start(self, trainer, pl_module) -> None:
        model = pl_module.model
        enc = getattr(model, "encoder", None)
        layers = list(getattr(enc, "layers", []))
        per_block = []
        for i, layer in enumerate(layers):
            gmp = getattr(layer, "gmp", None)
            per_block.append({
                "block": i,
                "block_params": _numel(layer),
                "gmp_class": f"{type(gmp).__module__}.{type(gmp).__qualname__}" if gmp is not None else None,
                "gmp_params": _numel(gmp),
                "gmp_geometry": getattr(gmp, "geometry", None) if gmp is not None else None,
            })
        gmp_ids = [id(lyr.gmp) for lyr in layers if getattr(lyr, "gmp", None) is not None]
        self._res["params"] = {
            "model_total": _numel(pl_module),
            "encoder": _numel(enc),
            "decoder": _numel(getattr(model, "decoder", None)),
            "gmp_total": sum(b["gmp_params"] for b in per_block),
            "gmp_blocks": len(gmp_ids),
            "gmp_modules_distinct": len(set(gmp_ids)) == len(gmp_ids),
            "per_block": per_block,
        }
        self._res["encoder_class"] = f"{type(enc).__module__}.{type(enc).__qualname__}"
        self._res["encoder_gmp_config"] = getattr(enc, "gmp", None)
        self._health = torch.zeros(4, device=pl_module.device, dtype=torch.float32)
        if trainer.is_global_zero:
            print(f"[gmp-diag] params {json.dumps(self._res['params'])}", flush=True)

    # ------------------------------------------------------------------------------------- eager dtype probe
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        if self._probed:
            return
        self._probed = True
        enc = pl_module.model.encoder
        seen: list[dict] = []
        handles = []

        def enc_hook(_m, args, kwargs):
            x = args[0] if args else kwargs.get("x")
            coords = kwargs.get("x_coords")
            mask = kwargs.get("kv_mask")
            seen.append({
                "where": "encoder",
                "x_dtype": str(x.dtype),
                "x_shape": list(x.shape),
                "x_coords_dtype": str(coords.dtype) if coords is not None else None,
                "x_coords_shape": list(coords.shape) if coords is not None else None,
                "valid_tokens_mean": float(mask.sum(-1).float().mean()) if mask is not None else None,
            })

        def gmp_hook(i):
            def hook(_m, args):
                x, neighbours, in_grid = args[:3]
                rec = {
                    "where": f"gmp[{i}]",
                    "x_dtype": str(x.dtype),
                    "neighbours_dtype": str(neighbours.dtype),
                    "neighbours_shape": list(neighbours.shape),
                }
                if not any(r["where"].startswith("gmp") for r in seen):  # the geometry is shared: stats once
                    on = in_grid.sum().item()
                    pairs = neighbours.float().sum().item()
                    rec |= {
                        "in_grid_tokens_mean": on / in_grid.shape[0],
                        "pairs_per_in_grid_token": pairs / max(on, 1),
                        "dense_entries": neighbours.numel(),
                        "pair_density": pairs / neighbours.numel(),
                    }
                seen.append(rec)

            return hook

        handles.append(enc.register_forward_pre_hook(enc_hook, with_kwargs=True))
        for i, layer in enumerate(enc.layers):
            if getattr(layer, "gmp", None) is not None:
                handles.append(layer.gmp.register_forward_pre_hook(gmp_hook(i)))
        try:
            with torch.no_grad(), torch.compiler.set_stance("force_eager"), trainer.precision_plugin.forward_context():
                out = pl_module.model(dict(batch[0]))  # shallow copy: the real step sees an untouched dict
            finite = all(bool(v.isfinite().all()) for v in _tensors(out))
        finally:
            for h in handles:
                h.remove()
        self._res["probe_batch0"] = {
            "records": seen,
            "outputs_finite": finite,
            "autocast": torch.is_autocast_enabled("cuda") if torch.cuda.is_available() else None,
        }
        if trainer.is_global_zero:
            print(f"[gmp-diag] probe {json.dumps(self._res['probe_batch0'])}", flush=True)

    # ------------------------------------------------------------------------------------- numerical health
    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        if not grads or self._health is None:
            return
        total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)).float())  # noqa: SLF001
        finite = torch.isfinite(total)
        self._health[2] += (~finite).float()
        self._health[3] = torch.maximum(self._health[3], torch.where(finite, total, 0.0))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if self._health is None:
            return
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self._health[0] += 1
        if isinstance(loss, torch.Tensor):
            self._health[1] += (~torch.isfinite(loss.detach().float())).float()

    # ------------------------------------------------------------------------------------------ per-epoch cost
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._epoch_t0 = time.perf_counter()
        self._epoch_step0 = trainer.global_step
        self._val_s = 0.0

    # Lightning runs the validation loop INSIDE the training epoch (before on_train_epoch_end), so it is timed here and
    # subtracted: train_wall_s / steps_per_s are training-only, val_s is the validation loop.
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or self._epoch_t0 is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._val_t0 = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or self._val_t0 is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._val_s += time.perf_counter() - self._val_t0
        self._val_t0 = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._epoch_t0 is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_wall = time.perf_counter() - self._epoch_t0
        val_s = self._val_s
        wall = epoch_wall - val_s
        steps = trainer.global_step - self._epoch_step0
        gib = 2.0**30
        rec = {
            "epoch": trainer.current_epoch,
            "train_wall_s": wall,
            "val_s": val_s,
            "epoch_wall_s": epoch_wall,
            "optimizer_steps": steps,
            "steps_per_s": steps / wall if wall > 0 else None,
        }
        if torch.cuda.is_available():
            rec |= {"peak_alloc_gib": torch.cuda.max_memory_allocated() / gib, "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib}
        self._res["epochs"].append(rec)
        self._snapshot_health()
        self._write(trainer)
        if trainer.is_global_zero:
            print(f"[gmp-diag] epoch {json.dumps(rec)}", flush=True)

    def _snapshot_health(self) -> None:
        """Copy the GPU health counters into the record (one host sync). Called at every epoch end and at train end,
        so a run that dies after an epoch (e.g. in a later callback) keeps its health record (job 7843955 lost it)."""
        if self._health is not None:
            steps, bad_loss, bad_grad, gmax = self._health.tolist()
            self._res["health"] = {
                "train_batches_seen": int(steps),
                "nonfinite_loss_batches": int(bad_loss),
                "nonfinite_grad_steps": int(bad_grad),
                "max_finite_preclip_grad_norm": gmax,
            }

    def on_train_end(self, trainer, pl_module) -> None:
        self._snapshot_health()
        path = self._write(trainer)
        if trainer.is_global_zero:
            print(f"[gmp-diag] health {json.dumps(self._res.get('health'))} -> {path}", flush=True)

    def _write(self, trainer) -> Path:
        """Write the record so far (also after every epoch, so a run cut short keeps its finished epochs)."""
        self._res["rank"] = trainer.global_rank
        self._res["pbs_jobid"] = os.environ.get("PBS_JOBID")
        base = self.out_path or os.environ.get("GMP_DIAG_OUT") or str(Path(trainer.default_root_dir) / "gmp_diagnostics.json")
        path = Path(f"{base}.rank{trainer.global_rank}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._res, indent=1, default=str))
        return path


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _tensors(v)
