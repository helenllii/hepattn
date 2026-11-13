"""Training-time-per-epoch benchmark callback for bench3 (Linformer / PHAT16 / PHAT16+GMP / MaskFormer on Polaris).

It runs inside the REAL training loop (real data, matcher, optimizer, DDP) and keeps apart costs that a tqdm rate mixes:

  startup         process start -> first training batch (imports, model build, DDP init, dataset construction).
  dataset init    wall time of each CLICDataset construction (train, val), per rank. Part of startup.
  compile         the first training batch (torch.compile happens on the first forward), and the rest of the warmup.
  steady state    `measure_batches` batches after `warmup_batches`. EVERY reported rate comes from this window only.
  breakdown       optional (`profile_batches`): after the window, an INSTRUMENTED window splits each batch into
                  forward+loss / backward / clip+optimizer (host clock and CUDA events at Lightning hook points), times
                  the CPU Hungarian matcher (GPU wait vs CPU work) and counts implicit CPU-GPU synchronizations; then a
                  short torch.profiler window ranks kernels and gives the GPU-busy fraction. Never inside the window.
  validation      one full validation pass after everything above (Lightning validates when `should_stop` is set),
                  timed cold (as measured) and as a steady per-epoch estimate.

WALLTIME GUARD. `deadline` (unix time; default $BENCH3_DEADLINE) is checked every `guard_every` batches with one
collective decision, so DDP ranks always stop at the same batch. Priority: steady state > validation > profiling.
A result JSON is written (rank 0, atomic replace) to `result_path` (default $BENCH3_RESULT):
  status "partial"             after warmup and every `partial_every` measured batches (rank-0 numbers only);
         "truncated_in_warmup" the deadline hit before any measurement;
         "train_done"          window complete, all ranks aggregated (written BEFORE validation/profiling);
         "truncated"           window cut short by the deadline (still valid if measured_batches >= 50);
         "complete"            plus validation (and profile, if requested) or their recorded skip reasons.

UNITS. A "batch" is one micro-batch on one rank; ranks step concurrently, so s/batch is the slowest rank's wall time
per batch and an epoch takes batches_per_epoch_per_rank x s_per_batch. An optimizer step is accumulate batches.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import json
import math
import os
import platform
import statistics
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is in every hepattn env
    psutil = None

GIB = 2**30
NAN = float("nan")


def _node_used_gib() -> float:
    if psutil is None:
        return NAN
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / GIB


def _pss_gib(proc) -> float:
    try:
        info = proc.memory_full_info()
        return getattr(info, "pss", info.rss) / GIB
    except Exception:  # noqa: BLE001 - a worker may exit between listing and reading
        return 0.0


def _med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else NAN


def _clean(x):
    """Strict-JSON-safe copy: non-finite floats -> None."""
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    return x


@contextlib.contextmanager
def _no_sync_debug():
    """Our own synchronizations must not be counted as the model's."""
    if not torch.cuda.is_available():
        yield
        return
    mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(0)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(mode)


class TrainingTimeBenchmark(Callback):
    """Time startup, compile, steady-state training, an optional breakdown, and one validation pass; then stop.

    Args:
        warmup_batches: Batches discarded before measuring (compile, recompiles, worker spin-up).
        measure_batches: Batches in the measurement window; rounded up to whole optimizer steps.
        full_train_events: Events in one real epoch; used only when `data.num_train` truncates the training set.
        expect_global_batch: If set, abort unless micro x world x accumulate equals it.
        synchronize: `torch.cuda.synchronize()` at batch boundaries in the window. Leave on.
        mem_poll_s: Host memory / CPU sampling period during the window, seconds.
        profile_batches: Instrumented breakdown batches after the window (0 = off; default $BENCH3_PROFILE or 0).
        profiler_batches: torch.profiler batches after the instrumented ones (0 = off; needs profile_batches > 0).
        profile_dir: Where the profiler table/trace go (default: next to result_path).
        profile_trace: Also export a Chrome trace (large).
        deadline: Unix time by which training must have stopped (default $BENCH3_DEADLINE; None = no guard).
        guard_every: Batches between collective deadline checks.
        min_measure_batches: Below this many measured batches a truncated window is flagged as unreliable.
        val_reserve_s: Fixed time reserved for a validation pass on top of its rate-based estimate.
        profile_reserve_s: Time reserved for the profile windows (incl. profiler export).
        result_path: JSON result file (default $BENCH3_RESULT; None = stdout only).
        partial_every: Measured batches between partial result writes.
        tag: Label; defaults to the run `name`.
    """

    def __init__(
        self,
        warmup_batches: int = 50,
        measure_batches: int = 150,
        full_train_events: int | None = None,
        expect_global_batch: int | None = None,
        synchronize: bool = True,
        mem_poll_s: float = 0.5,
        profile_batches: int | None = None,
        profiler_batches: int = 10,
        profile_dir: str | None = None,
        profile_trace: bool = False,
        deadline: float | None = None,
        guard_every: int = 10,
        min_measure_batches: int = 50,
        val_reserve_s: float = 120.0,
        profile_reserve_s: float = 180.0,
        result_path: str | None = None,
        partial_every: int = 25,
        tag: str | None = None,
    ) -> None:
        super().__init__()
        if warmup_batches < 0 or measure_batches < 1:
            raise ValueError(f"need warmup_batches >= 0 and measure_batches >= 1, got {warmup_batches}, {measure_batches}")
        env = os.environ
        self.warmup_batches = warmup_batches
        self.measure_batches = measure_batches
        self.full_train_events = full_train_events
        self.expect_global_batch = expect_global_batch
        self.synchronize = synchronize
        # CUDA in use by this trainer (set in setup), not merely present on the node
        self._cuda = torch.cuda.is_available()
        self.mem_poll_s = mem_poll_s
        self.profile_batches = int(env.get("BENCH3_PROFILE", "0")) if profile_batches is None else profile_batches
        self.profiler_batches = profiler_batches if self.profile_batches > 0 else 0
        self.profile_trace = profile_trace or env.get("BENCH3_PROFILE_TRACE") == "1"
        self.deadline = deadline if deadline is not None else (float(env["BENCH3_DEADLINE"]) if env.get("BENCH3_DEADLINE") else None)
        self.guard_every = max(1, guard_every)
        self.min_measure_batches = min_measure_batches
        self.val_reserve_s = val_reserve_s
        self.profile_reserve_s = profile_reserve_s
        self.result_path = result_path or env.get("BENCH3_RESULT") or None
        self.profile_dir = profile_dir or (str(Path(self.result_path).with_suffix("")) + "_profile" if self.result_path else None)
        self.partial_every = max(1, partial_every)
        self.tag = tag

        self._proc_start = psutil.Process().create_time() if psutil is not None else time.time()
        self._node_used_before_gib = _node_used_gib()
        self._dataset_init: list[dict[str, Any]] = []
        self._unpatch = self._patch_dataset_init()

        # phase: warmup -> measure -> instrument -> profile -> stopping -> done
        self._phase = "warmup"
        self._seen = 0
        self._warm_s: list[float] = []
        self._step_s: list[float] = []
        self._wait_s: list[float] = []
        self._t_start: float | None = None
        self._t_prev_end: float | None = None
        self._window_t0: float | None = None
        self._window_t1: float | None = None
        self._res: dict[str, Any] = {}
        self._truncated = False

        self._mem_stop = threading.Event()
        self._mem_thread: threading.Thread | None = None
        self._node_used_peak = NAN
        self._rss_peak = NAN
        self._cpu_samples: list[float] = []

        self._instr: list[dict[str, float]] = []
        self._instr_cur: dict[str, Any] = {}
        self._matcher_calls: list[tuple[float, float]] = []
        self._matcher_restore = None
        self._sync_cm = None
        self._sync_records: list = []
        self._profiler = None
        self._profiled = 0

        self._val_t0: float | None = None
        self._val_batch_t0: float | None = None
        self._val_batch_s: list[float] = []
        self._val_status = "not run"
        self._profile_status = "disabled" if self.profile_batches <= 0 else "pending"
        self._final = False

    # ======================================================================================== dataset init timing
    def _patch_dataset_init(self):
        """Time each CLICDataset construction. Lightning calls datamodule.setup before any callback hook, so the
        constructor is wrapped for the duration of setup (restored in `setup`).
        """
        try:
            from hepattn.experiments.clic.pflow_data import CLICDataset  # noqa: PLC0415
        except ImportError:
            return lambda: None
        original = CLICDataset.__init__
        records = self._dataset_init

        @functools.wraps(original)
        def timed_init(ds, *args, **kwargs):
            t0 = time.perf_counter()
            original(ds, *args, **kwargs)
            records.append({
                "filepath": str(kwargs.get("filepath", args[0] if args else "?")),
                "events": len(ds),
                "seconds": time.perf_counter() - t0,
            })

        CLICDataset.__init__ = timed_init

        def unpatch() -> None:
            CLICDataset.__init__ = original

        return unpatch

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._cuda = trainer.strategy.root_device.type == "cuda"
        self._unpatch()
        self._unpatch = lambda: None

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._unpatch()
        self._mem_stop.set()
        self._end_instrumentation(pl_module)
        if self._profiler is not None:
            with contextlib.suppress(Exception):
                self._profiler.stop()
            self._profiler = None

    # ================================================================================================ helpers
    def _remaining(self) -> float:
        return self.deadline - time.time() if self.deadline else math.inf

    def _any_rank(self, trainer: Trainer, flag: bool) -> bool:
        """One collective boolean OR, so every rank takes the same branch at the same batch."""
        return bool(trainer.strategy.reduce_boolean_decision(bool(flag), all=False))

    def _all_ranks(self, trainer: Trainer, flag: bool) -> bool:
        return bool(trainer.strategy.reduce_boolean_decision(bool(flag), all=True))

    def _gather(self, trainer: Trainer, values: list[float]) -> torch.Tensor:
        """(world, k) float64 tensor of every rank's values (a collective: call on all ranks)."""
        t = torch.tensor(values, dtype=torch.float64)
        if trainer.world_size == 1:
            return t.view(1, -1)
        out = trainer.strategy.all_gather(t.to(trainer.strategy.root_device))
        return out.view(trainer.world_size, -1).cpu()

    @staticmethod
    def _col(g: torch.Tensor, i: int, how: str = "max") -> float:
        col = g[:, i]
        col = col[~torch.isnan(col)]
        if not len(col):
            return NAN
        return float(col.max() if how == "max" else col.sum() if how == "sum" else col.min())

    def _write(self, trainer: Trainer, res: dict[str, Any], status: str) -> None:
        """Rank 0: atomically replace the result file, and echo a BENCH3_JSON line for the log."""
        if not trainer.is_global_zero:
            return
        res = {**res, "status": status, "written_at": time.time(), "deadline_remaining_s": self._remaining() if self.deadline else None}
        line = json.dumps(_clean(res), sort_keys=True)
        if self.result_path:
            path = Path(self.result_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(line + "\n")
            Path(tmp).replace(path)
        if status != "partial":
            print("BENCH3_JSON " + line, flush=True)

    def _sync(self) -> None:
        if self.synchronize and self._cuda:
            with _no_sync_debug():
                torch.cuda.synchronize()

    # ============================================================================================== train start
    @staticmethod
    def _micro_batch(trainer: Trainer) -> int:
        for obj in (getattr(trainer, "train_dataloader", None), getattr(trainer, "datamodule", None)):
            size = getattr(obj, "batch_size", None)
            if size:
                return int(size)
        raise ValueError("cannot determine the micro-batch size from the dataloader or the datamodule")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._t_train_start = time.perf_counter()
        self._startup_s = time.time() - self._proc_start
        self.tag = self.tag or getattr(pl_module, "name", None) or "bench3"
        self._micro = self._micro_batch(trainer)
        self._world = int(trainer.world_size)
        self._accum = int(trainer.accumulate_grad_batches)
        self._global_batch = self._micro * self._world * self._accum
        if self.measure_batches % self._accum:
            self.measure_batches += self._accum - self.measure_batches % self._accum
        if self.expect_global_batch is not None and self._global_batch != self.expect_global_batch:
            raise ValueError(
                f"effective global batch is {self._global_batch} (batch_size {self._micro} x world {self._world} x "
                f"accumulate {self._accum}), expected {self.expect_global_batch}"
            )
        self._n_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        if trainer.is_global_zero:
            print("-" * 100, flush=True)
            print(
                f"[bench3:{self.tag}] global batch {self._global_batch} = micro {self._micro} x world {self._world} x accum {self._accum}", flush=True
            )
            print(
                f"[bench3:{self.tag}] warmup {self.warmup_batches} + measure {self.measure_batches} + breakdown {self.profile_batches}"
                f"+{self.profiler_batches} batches, then validation; deadline in {self._remaining():.0f} s",
                flush=True,
            )
            print(f"[bench3:{self.tag}] trainable parameters {self._n_params:,}; startup {self._startup_s:.1f} s", flush=True)
            print("-" * 100, flush=True)

    # ============================================================================================ batch hooks
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        now = time.perf_counter()
        if self._phase == "warmup" and self._seen == self.warmup_batches:
            self._begin_window(trainer, pl_module)
            now = time.perf_counter()
        if self._phase == "measure" and self._t_prev_end is not None:
            self._wait_s.append(now - self._t_prev_end)
        if self._phase == "instrument":
            self._instr_begin()
        if self._phase == "profile" and self._profiler is None:
            self._start_profiler()
        self._t_start = now

    def on_before_backward(self, trainer: Trainer, pl_module: LightningModule, loss) -> None:
        if self._phase == "instrument":
            self._instr_mark("fwd_end")

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._phase == "instrument":
            self._instr_mark("bwd_end")

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer) -> None:
        if self._phase == "instrument":
            self._instr_mark("opt_start")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        if self._phase in {"stopping", "done"}:
            return
        self._sync()
        now = time.perf_counter()
        dur = now - self._t_start if self._t_start is not None else NAN
        self._seen += 1
        self._t_prev_end = now

        if self._phase == "warmup":
            self._warm_s.append(dur)
        elif self._phase == "measure":
            self._step_s.append(dur)
            n = len(self._step_s)
            if n % self.partial_every == 0 and n < self.measure_batches:
                self._write(trainer, self._partial_record(), "partial")
        elif self._phase == "instrument":
            self._instr_end(dur)
        elif self._phase == "profile":
            self._profiled += 1

        # Collective deadline check at fixed batch counts, identical on every rank.
        if self.deadline and self._seen % self.guard_every == 0:
            late = self._any_rank(trainer, self._remaining() < 0)
            if late:
                self._on_deadline(trainer, pl_module)
                return

        if self._phase == "measure" and len(self._step_s) >= self.measure_batches:
            self._close_window(trainer, pl_module)
        elif self._phase == "instrument" and len(self._instr) >= self.profile_batches:
            self._end_instrumentation(pl_module)
            if self.profiler_batches > 0:
                self._phase = "profile"
            else:
                self._finish_breakdown(trainer)
        elif self._phase == "profile" and self._profiled >= self.profiler_batches:
            self._stop_profiler(trainer)
            self._finish_breakdown(trainer)

    def _on_deadline(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.is_global_zero:
            print(f"[bench3:{self.tag}] DEADLINE reached in phase {self._phase} after {self._seen} batches", flush=True)
        if self._phase == "warmup":
            self._res = self._partial_record()
            self._val_status = self._profile_status = "skipped: deadline during warmup"
            self._stop(trainer, run_val=False)
            self._write(trainer, self._res, "truncated_in_warmup")
            self._final = True
        elif self._phase == "measure":
            self._truncated = True
            self._close_window(trainer, pl_module, deadline_hit=True)
        else:  # in the breakdown: keep what was measured, skip the rest
            self._end_instrumentation(pl_module)
            if self._profiler is not None:
                self._stop_profiler(trainer)
            self._profile_status = f"truncated by deadline ({len(self._instr)} instrumented, {self._profiled} profiled batches)"
            self._finish_breakdown(trainer, deadline_hit=True)

    def _stop(self, trainer: Trainer, run_val: bool) -> None:
        """Stop training; Lightning then validates iff limit_val_batches > 0 (it is read live)."""
        if not run_val:
            trainer.limit_val_batches = 0
        trainer.should_stop = True
        self._phase = "stopping"
        self._val_t0 = time.perf_counter() if run_val else None

    # =========================================================================================== measured window
    def _begin_window(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._sync()
        if self._cuda:
            self._peak_resv_warmup = torch.cuda.max_memory_reserved() / GIB
            torch.cuda.reset_peak_memory_stats()
        self._window_t0 = time.perf_counter()
        self._t_prev_end = None
        self._phase = "measure"
        self._cpu_settings_start = self._cpu_settings(pl_module)
        self._start_sampler()
        self._write(trainer, self._partial_record(), "partial")

    def _start_sampler(self) -> None:
        if psutil is None or self.mem_poll_s <= 0 or self._mem_thread is not None:
            return
        me = psutil.Process()
        self._node_used_peak = _node_used_gib()
        self._rss_peak = me.memory_info().rss / GIB
        psutil.cpu_percent(interval=None)  # prime: the next call reports usage since now

        def poll() -> None:
            while not self._mem_stop.wait(self.mem_poll_s):
                try:
                    self._node_used_peak = max(self._node_used_peak, _node_used_gib())
                    self._rss_peak = max(self._rss_peak, me.memory_info().rss / GIB)
                    self._cpu_samples.append(psutil.cpu_percent(interval=None))
                except Exception:  # noqa: BLE001 - reporting only
                    return

        self._mem_thread = threading.Thread(target=poll, daemon=True, name="bench3-mem")
        self._mem_thread.start()

    @staticmethod
    def _cpu_settings(pl_module: LightningModule) -> dict[str, Any]:
        matcher = getattr(getattr(pl_module, "model", None), "matcher", None)
        out: dict[str, Any] = {
            "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "logical_cpus": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False) if psutil is not None else None,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "matcher_n_jobs": getattr(matcher, "n_jobs", None),
            "matcher_parallel": getattr(matcher, "parallel_solver", None),
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        }
        if psutil is not None:
            me = psutil.Process()
            kids = me.children(recursive=True)
            out["rank_threads"] = me.num_threads()
            out["child_procs"] = len(kids)
            out["child_threads"] = sum(k.num_threads() for k in kids if k.is_running())
        return out

    def _partial_record(self) -> dict[str, Any]:
        """Rank-0-local snapshot (no collectives): enough to recover timings if the job dies."""
        n = len(self._step_s)
        wall = (self._t_prev_end - self._window_t0) if (n and self._window_t0 is not None and self._t_prev_end is not None) else NAN
        return {
            "tag": self.tag,
            "aggregation": "rank0-local",
            "world_size": getattr(self, "_world", None),
            "micro_batch": getattr(self, "_micro", None),
            "global_batch": getattr(self, "_global_batch", None),
            "startup_s": getattr(self, "_startup_s", NAN),
            "first_batch_s_incl_compile": self._warm_s[0] if self._warm_s else NAN,
            "warmup_rest_s": sum(self._warm_s[1:]) if len(self._warm_s) > 1 else NAN,
            "warmup_batches_done": len(self._warm_s),
            "measured_batches": n,
            "s_per_batch": wall / n if n else NAN,
            "dataset_init": self._dataset_init,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
        }

    def _close_window(self, trainer: Trainer, pl_module: LightningModule, deadline_hit: bool = False) -> None:
        """End of the measured window: aggregate over ranks (collective), write `train_done`, decide what follows."""
        self._window_t1 = time.perf_counter()
        self._mem_stop.set()
        cpu_end = self._cpu_settings(pl_module)
        n = len(self._step_s)
        wall = self._window_t1 - self._window_t0
        main_pss = workers_pss = NAN
        if psutil is not None:
            me = psutil.Process()
            main_pss = _pss_gib(me)
            workers_pss = sum(_pss_gib(k) for k in me.children(recursive=True))
        cuda = self._cuda
        local = {
            "s_per_batch": wall / n if n else NAN,
            "batch_p50_s": _med(self._step_s),
            "batch_p90_s": sorted(self._step_s)[int(0.9 * (n - 1))] if n else NAN,
            "wait_frac": sum(self._wait_s) / wall if wall > 0 else NAN,
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / GIB if cuda else NAN,
            "peak_resv_gib": torch.cuda.max_memory_reserved() / GIB if cuda else NAN,
            "peak_resv_warmup_gib": getattr(self, "_peak_resv_warmup", NAN),
            "rss_peak_gib": self._rss_peak,
            "main_pss_gib": main_pss,
            "workers_pss_gib": workers_pss,
            "node_used_peak_gib": self._node_used_peak,
            "startup_s": self._startup_s,
            "first_batch_s": self._warm_s[0] if self._warm_s else NAN,
            "warmup_rest_s": sum(self._warm_s[1:]) if len(self._warm_s) > 1 else NAN,
            "rank_threads": float(cpu_end.get("rank_threads", NAN) or NAN),
            "child_threads": float(cpu_end.get("child_threads", NAN) or NAN),
            "child_procs": float(cpu_end.get("child_procs", NAN) or NAN),
        }
        cols = list(local)
        init = self._dataset_init_seconds(trainer)
        g = self._gather(trainer, [local[c] for c in cols] + [init["train"], init["val"]])
        k = len(cols)

        def w(name: str, how: str = "max") -> float:
            return self._col(g, cols.index(name), how)

        s_batch = w("s_per_batch")
        bpe, bpe_src = self._batches_per_epoch(trainer)
        dl = trainer.train_dataloader
        cpu_util = _med(self._cpu_samples)
        res: dict[str, Any] = {
            "tag": self.tag,
            "aggregation": "all ranks (worst rank for times/memory, sum for host memory and threads)",
            "host": platform.node(),
            "device": torch.cuda.get_device_name() if cuda else "cpu",
            "world_size": self._world,
            "micro_batch": self._micro,
            "accumulate": self._accum,
            "global_batch": self._global_batch,
            "precision": str(trainer.precision),
            "trainable_params": self._n_params,
            "num_workers": getattr(dl, "num_workers", None),
            "persistent_workers": getattr(dl, "persistent_workers", None),
            # separately timed, never in throughput
            "startup_s": w("startup_s"),
            "dataset_init_train_s": self._col(g, k + 0),
            "dataset_init_val_s": self._col(g, k + 1),
            "dataset_events": {r["filepath"]: r["events"] for r in self._dataset_init},
            "first_batch_s_incl_compile": w("first_batch_s"),
            "warmup_rest_s": w("warmup_rest_s"),
            "warmup_batches": self.warmup_batches,
            "warmup_s_incl_compile": w("first_batch_s") + w("warmup_rest_s"),
            # steady state
            "measured_batches": n,
            "window_truncated": self._truncated,
            "window_reliable": n >= self.min_measure_batches,
            "s_per_batch": s_batch,
            "batch_p50_s": w("batch_p50_s"),
            "batch_p90_s": w("batch_p90_s"),
            "batches_per_s_per_rank": 1.0 / s_batch,
            "optimizer_steps_per_s": 1.0 / (s_batch * self._accum),
            "events_per_s_global": self._micro * self._world / s_batch,
            "dataloader_wait_frac": w("wait_frac"),
            # epoch
            "train_batches_per_epoch_per_rank": bpe,
            "optimizer_steps_per_epoch": math.ceil(bpe / self._accum),
            "batches_per_epoch_source": bpe_src,
            "epoch_train_s": bpe * s_batch,
            # memory
            "gpu_peak_alloc_gib": w("peak_alloc_gib"),
            "gpu_peak_reserved_gib": w("peak_resv_gib"),
            "gpu_peak_reserved_incl_warmup_gib": w("peak_resv_warmup_gib"),
            "host_node_used_before_data_gib": self._node_used_before_gib,
            "host_node_used_peak_gib": w("node_used_peak_gib"),
            "host_rank_rss_peak_gib_max": w("rss_peak_gib"),
            "host_ranks_pss_gib_sum": w("main_pss_gib", "sum"),
            "host_workers_pss_gib_sum": w("workers_pss_gib", "sum"),
            # CPU (refinement: effective settings and load)
            "cpu": {
                **self._cpu_settings_start,
                "threads_all_ranks": w("rank_threads", "sum"),
                "child_threads_all_ranks": w("child_threads", "sum"),
                "child_procs_all_ranks": w("child_procs", "sum"),
                "node_cpu_util_pct_median": cpu_util,
                "loadavg_1m_end": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
            },
            "pbs_jobid": os.environ.get("PBS_JOBID"),
        }
        res["host_job_pss_gib_total"] = res["host_ranks_pss_gib_sum"] + res["host_workers_pss_gib_sum"]
        cores = res["cpu"].get("physical_cores") or res["cpu"].get("logical_cpus")
        nj = res["cpu"].get("matcher_n_jobs") or 0
        nw = res["num_workers"] or 0
        res["cpu"]["planned_busy_threads_during_matching"] = self._world * (int(nj) + int(nw) + 1)
        res["cpu"]["oversubscribed_during_matching"] = bool(cores) and res["cpu"]["planned_busy_threads_during_matching"] > cores
        try:
            res["provenance"] = self._provenance(trainer)
        except Exception as exc:  # noqa: BLE001 - never lose the timing numbers over provenance
            res["provenance"] = {"error": f"{type(exc).__name__}: {exc}"}
        self._res = res
        self._write(trainer, res, "truncated" if self._truncated else "train_done")
        if trainer.is_global_zero:
            self._print_train(res)

        # What next, by priority: validation before profiling, both only if time allows (one collective decision).
        val_need = self.val_reserve_s + self._val_batches_estimate(trainer) * s_batch * 2
        prof_need = self.profile_reserve_s + (self.profile_batches + self.profiler_batches) * s_batch * 3
        rem = self._remaining()
        do_val = bool(trainer.limit_val_batches) and not deadline_hit and self._all_ranks(trainer, rem > val_need)
        do_prof = self.profile_batches > 0 and do_val and self._all_ranks(trainer, rem > val_need + prof_need)
        if not trainer.limit_val_batches:
            self._val_status = "disabled (limit_val_batches 0)"
        elif not do_val:
            self._val_status = f"skipped: {rem:.0f} s left, need ~{val_need:.0f} s" if not deadline_hit else "skipped: deadline"
        if self.profile_batches > 0 and not do_prof:
            self._profile_status = f"skipped: {rem:.0f} s left, need ~{val_need + prof_need:.0f} s (validation has priority)"
        if do_prof:
            self._phase = "instrument"
            self._begin_instrumentation(pl_module)
        else:
            self._stop(trainer, run_val=do_val)
            if not do_val:
                self._finalize(trainer)

    # ======================================================================================= breakdown window
    def _begin_instrumentation(self, pl_module: LightningModule) -> None:
        """Wrap the (eager) matcher and turn on sync debugging. Compiled modules are NOT hooked: a hook added after
        compile would change their guards and force recompiles, perturbing what is measured.
        """
        matcher = getattr(getattr(pl_module, "model", None), "matcher", None)
        if matcher is not None and "forward" not in vars(matcher):
            orig = matcher.forward
            calls = self._matcher_calls

            def timed_forward(*args, **kwargs):
                t0 = time.perf_counter()
                if self._cuda:
                    with _no_sync_debug():
                        torch.cuda.synchronize()  # isolate "waiting for the costs" from the CPU solve
                t1 = time.perf_counter()
                out = orig(*args, **kwargs)
                calls.append((t1 - t0, time.perf_counter() - t1))
                return out

            matcher.forward = timed_forward

            def restore() -> None:
                vars(matcher).pop("forward", None)

            self._matcher_restore = restore
        if self._cuda:
            self._sync_cm = warnings.catch_warnings(record=True)
            self._sync_records = self._sync_cm.__enter__()
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")

    def _end_instrumentation(self, pl_module: LightningModule | None) -> None:
        if self._matcher_restore is not None:
            self._matcher_restore()
            self._matcher_restore = None
        if self._sync_cm is not None:
            torch.cuda.set_sync_debug_mode(0)
            self._sync_cm.__exit__(None, None, None)
            self._sync_cm = None

    def _event(self):
        if not self._cuda:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _instr_begin(self) -> None:
        self._instr_cur = {
            "t": {"start": time.perf_counter()},
            "e": {"start": self._event()},
            "m0": len(self._matcher_calls),
            "s0": len(self._sync_records),
        }

    def _instr_mark(self, name: str) -> None:
        if self._instr_cur:
            self._instr_cur["t"][name] = time.perf_counter()
            self._instr_cur["e"][name] = self._event()

    def _instr_end(self, dur: float) -> None:
        cur = self._instr_cur
        if not cur:
            return
        self._instr_mark("end")
        t, e = cur["t"], cur["e"]
        if e["end"] is not None:
            with _no_sync_debug():
                e["end"].synchronize()

        def host(a: str, b: str) -> float:
            return (t[b] - t[a]) * 1e3 if a in t and b in t else NAN

        def gpu(a: str, b: str) -> float:
            return e[a].elapsed_time(e[b]) if e.get(a) is not None and e.get(b) is not None else NAN

        calls = self._matcher_calls[cur["m0"] :]
        rec = {
            "batch_ms": dur * 1e3,
            "fwd_loss_host_ms": host("start", "fwd_end"),
            "fwd_loss_gpu_ms": gpu("start", "fwd_end"),
            "bwd_host_ms": host("fwd_end", "bwd_end"),
            "bwd_gpu_ms": gpu("fwd_end", "bwd_end"),
            "clip_opt_host_ms": host("opt_start", "end"),
            "clip_opt_gpu_ms": gpu("opt_start", "end"),
            "matcher_calls": float(len(calls)),
            "matcher_gpu_wait_ms": sum(c[0] for c in calls) * 1e3,
            "matcher_cpu_ms": sum(c[1] for c in calls) * 1e3,
            "syncs": float(sum(1 for w in self._sync_records[cur["s0"] :] if "synchroniz" in str(w.message).lower())),
        }
        self._instr.append(rec)
        self._instr_cur = {}

    def _start_profiler(self) -> None:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self._cuda:
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(activities=acts, record_shapes=False, with_stack=False)
        self._profiler.start()
        self._prof_t0 = time.perf_counter()

    def _stop_profiler(self, trainer: Trainer) -> None:
        prof, self._profiler = self._profiler, None
        if prof is None:
            return
        self._sync()
        prof.stop()
        wall_ms = (time.perf_counter() - self._prof_t0) * 1e3
        out: dict[str, Any] = {"batches": self._profiled, "wall_ms_per_batch": wall_ms / max(self._profiled, 1)}
        try:
            ka = prof.key_averages()

            def dev(evt) -> float:
                return float(getattr(evt, "self_device_time_total", getattr(evt, "self_cuda_time_total", 0.0)))

            kernels_us = sum(dev(evt) for evt in ka)
            out["gpu_kernel_ms_per_batch"] = kernels_us / 1e3 / max(self._profiled, 1)
            out["gpu_busy_frac"] = (kernels_us / 1e3) / wall_ms if wall_ms > 0 else NAN
            top = sorted(ka, key=dev, reverse=True)[:25]
            out["top_gpu_ops"] = [[evt.key[:120], dev(evt) / 1e3 / max(self._profiled, 1)] for evt in top if dev(evt) > 0]
            top_cpu = sorted(ka, key=lambda evt: evt.self_cpu_time_total, reverse=True)[:15]
            out["top_cpu_ops"] = [[evt.key[:120], evt.self_cpu_time_total / 1e3 / max(self._profiled, 1)] for evt in top_cpu]
            if self.profile_dir and trainer.is_global_zero:
                d = Path(self.profile_dir)
                d.mkdir(parents=True, exist_ok=True)
                sort_key = "self_cuda_time_total" if self._cuda else "self_cpu_time_total"
                (d / "key_averages.txt").write_text(ka.table(sort_by=sort_key, row_limit=60))
                out["table_path"] = str(d / "key_averages.txt")
                if self.profile_trace:
                    prof.export_chrome_trace(str(d / "trace.json"))
                    out["trace_path"] = str(d / "trace.json")
        except Exception as exc:  # noqa: BLE001 - profiling is optional
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["note"] = "profiled on this rank; host times are inflated by the profiler, use GPU times and fractions only"
        self._profile_out = out

    def _finish_breakdown(self, trainer: Trainer, deadline_hit: bool = False) -> None:
        """Aggregate the instrumented batches (collective), then validate if time still allows."""
        keys = ["batch_ms", "fwd_loss_host_ms", "fwd_loss_gpu_ms", "bwd_host_ms", "bwd_gpu_ms", "clip_opt_host_ms", "clip_opt_gpu_ms",
                "matcher_calls", "matcher_gpu_wait_ms", "matcher_cpu_ms", "syncs"]  # fmt: skip
        local = [_med([r[k] for r in self._instr]) for k in keys]
        g = self._gather(trainer, local)
        bd: dict[str, Any] = {"instrumented_batches": len(self._instr), "rank0": dict(zip(keys, g[0].tolist(), strict=True))}
        bd["worst_rank"] = {k: self._col(g, i) for i, k in enumerate(keys)}
        r0 = bd["rank0"]
        if r0["batch_ms"] and math.isfinite(r0["batch_ms"]):
            bd["matcher_share_of_batch"] = (r0["matcher_cpu_ms"] + r0["matcher_gpu_wait_ms"]) / r0["batch_ms"]
            bd["matcher_cpu_share_of_batch"] = r0["matcher_cpu_ms"] / r0["batch_ms"]
        bd["sync_sites_top"] = self._sync_sites() if trainer.is_global_zero else None
        bd["profiler"] = getattr(self, "_profile_out", None)
        bd["note"] = (
            "medians over instrumented batches; fwd_loss includes the matcher; the matcher wrapper adds one explicit sync "
            "per call (the solve already waits for the costs); clip_opt = clip + optimizer step + scheduler"
        )
        self._res["breakdown"] = bd
        if self._profile_status == "pending":
            self._profile_status = "done"
        run_val = not deadline_hit and self._all_ranks(trainer, self._remaining() > self.val_reserve_s)
        if not run_val:
            self._val_status = "skipped: deadline after breakdown"
        self._stop(trainer, run_val=run_val)
        if not run_val:
            self._finalize(trainer)

    def _sync_sites(self) -> list:
        sites = collections.Counter(
            f"{os.path.basename(str(w.filename))}:{w.lineno}" for w in self._sync_records if "synchroniz" in str(w.message).lower()
        )
        n = max(len(self._instr), 1)
        return [[site, cnt / n] for site, cnt in sites.most_common(10)]

    # ============================================================================================== validation
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self._val_t0 is not None and not self._final:
            self._sync()
            self._val_batch_t0 = time.perf_counter()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self._val_batch_t0 is not None and not self._final:
            self._sync()
            self._val_batch_s.append(time.perf_counter() - self._val_batch_t0)
            self._val_batch_t0 = None

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._val_t0 is None or self._final:
            return
        self._sync()
        val_wall = time.perf_counter() - self._val_t0
        n = len(self._val_batch_s)
        first = self._val_batch_s[0] if n else NAN
        warm = _med(self._val_batch_s[1:])
        # Steady validation: the compile-dominated first batch replaced by a warm one. Keeps the worker spawn and the
        # epoch-end metric reduction, which production pays every epoch.
        steady = val_wall - first + warm if n > 1 else val_wall
        g = self._gather(trainer, [val_wall, first, warm, steady, float(n)])
        self._res.update({
            "val_wall_s_first": self._col(g, 0),
            "val_first_batch_s": self._col(g, 1),
            "val_warm_batch_s": self._col(g, 2),
            "epoch_val_s_steady": self._col(g, 3),
            "val_batches_per_rank": int(self._col(g, 4)) if math.isfinite(self._col(g, 4)) else 0,
        })
        self._val_status = "done"
        self._finalize(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._res and not self._final:
            self._finalize(trainer)

    # =================================================================================================== final
    def _finalize(self, trainer: Trainer) -> None:
        if self._final:
            return
        self._final = True
        self._phase = "done"
        res = self._res
        val = res.get("epoch_val_s_steady", NAN)
        res["epoch_total_s"] = res["epoch_train_s"] + (val if isinstance(val, float) and math.isfinite(val) else 0.0)
        res["epoch_total_includes_val"] = isinstance(val, float) and math.isfinite(val)
        res["val_status"] = self._val_status
        res["profile_status"] = self._profile_status
        self.result = res
        self._write(trainer, res, "truncated" if self._truncated else "complete")
        if trainer.is_global_zero:
            self._print_final(res)

    # ================================================================================================ bookkeeping
    def _dataset_init_seconds(self, trainer: Trainer) -> dict[str, float]:
        init = {"train": NAN, "val": NAN}
        train_path = getattr(trainer.datamodule, "train_path", None)
        for rec in self._dataset_init:
            key = "train" if rec["filepath"] == str(train_path) else "val"
            init[key] = rec["seconds"] if math.isnan(init[key]) else init[key] + rec["seconds"]
        return init

    def _batches_per_epoch(self, trainer: Trainer) -> tuple[int, str]:
        num_train = getattr(trainer.datamodule, "num_train", -1) if trainer.datamodule is not None else -1
        nb = trainer.num_training_batches
        if (num_train is None or num_train <= 0) and nb not in {None, float("inf")}:
            return int(nb), "exact: trainer.num_training_batches (full training set loaded)"
        if not self.full_train_events:
            return int(nb), "WARNING: truncated training set and no full_train_events; this is NOT the real epoch"
        per_rank_events = math.ceil(self.full_train_events / self._world)  # DistributedSampler pads to equal shards
        return math.ceil(per_rank_events / self._micro), f"extrapolated from full_train_events={self.full_train_events}"

    def _val_batches_estimate(self, trainer: Trainer) -> int:
        num_val = getattr(trainer.datamodule, "num_val", None)
        if not num_val or num_val <= 0:
            return 50
        return math.ceil(math.ceil(num_val / self._world) / self._micro)

    @staticmethod
    def _provenance(trainer: Trainer) -> dict[str, Any]:
        """Exactly which code ran: module files, the git commit of their checkout, hashes of uncommitted bench files."""
        import hashlib  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        import hepattn  # noqa: PLC0415

        prov: dict[str, Any] = {"hepattn_file": os.path.realpath(hepattn.__file__), "torch": torch.__version__}
        root = Path(prov["hepattn_file"]).parents[1]  # <checkout>/src

        def git(*args: str) -> str | None:
            try:
                return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
            except Exception:  # noqa: BLE001
                return None

        prov["git_toplevel"] = git("rev-parse", "--show-toplevel")
        prov["git_commit"] = git("rev-parse", "HEAD")
        prov["git_branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
        status = git("status", "--porcelain")
        prov["git_dirty"] = bool(status) if status is not None else None
        prov["git_status"] = status.splitlines() if status else []
        diff = git("diff", "HEAD")
        prov["git_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest() if diff else None

        model = getattr(trainer.lightning_module, "model", trainer.lightning_module)
        enc = getattr(model, "encoder", None)
        prov["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
        prov["model_file"] = os.path.realpath(inspect.getfile(type(model)))
        if enc is not None:
            prov["encoder_class"] = f"{type(enc).__module__}.{type(enc).__qualname__}"
            prov["encoder_file"] = os.path.realpath(inspect.getfile(type(enc)))
            prov["encoder_attn_type"] = getattr(enc, "attn_type", None)
            prov["encoder_gmp_layers"] = [i for i, layer in enumerate(getattr(enc, "layers", [])) if getattr(layer, "gmp", None) is not None]
            gmps = [type(layer.gmp) for layer in getattr(enc, "layers", []) if getattr(layer, "gmp", None) is not None]
            prov["encoder_gmp_impl"] = f"{gmps[0].__module__}.{gmps[0].__qualname__}" if gmps else None
            attn = [type(m) for m in enc.modules() if type(m).__name__ == "Attention"]
            if attn:
                prov["attention_file"] = os.path.realpath(inspect.getfile(attn[0]))
        for mod in ("masked_linformer", "flash_attn", "lap1015"):
            try:
                m = __import__(mod)
                prov[mod] = {"version": getattr(m, "__version__", None), "file": os.path.realpath(m.__file__)}
            except Exception:  # noqa: BLE001
                prov[mod] = None
        clic = root / "hepattn" / "experiments" / "clic"
        files = sorted([*(clic / "configs" / "bench3").glob("*.yaml"), *(clic / "bench3").glob("*.py"), clic / "configs" / "base.yaml"])
        prov["file_sha256"] = {str(f.relative_to(root.parent)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
        return prov

    # ================================================================================================== printing
    @staticmethod
    def _f(x, spec: str = ".3f") -> str:
        return format(x, spec) if isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) else "--"

    def _print_train(self, r: dict[str, Any]) -> None:
        f = self._f
        pv = r.get("provenance", {})
        cpu = r["cpu"]
        lines = [
            "=" * 100,
            f"BENCH3 [{r['tag']}] TRAINING WINDOW  {r['device']} x {r['world_size']}  {r['precision']}  host {r['host']}",
            "=" * 100,
            f"  global batch   : {r['global_batch']} (micro {r['micro_batch']} x world {r['world_size']} x accum {r['accumulate']})   params {r['trainable_params']:,}",
            f"  NOT in rates   : startup {f(r['startup_s'], '.1f')} s (dataset init train {f(r['dataset_init_train_s'], '.1f')} / val {f(r['dataset_init_val_s'], '.1f')} s);"
            f" first batch (compile) {f(r['first_batch_s_incl_compile'], '.1f')} s; rest of warmup {f(r['warmup_rest_s'], '.1f')} s",
            f"  STEADY STATE   : {f(r['s_per_batch'], '.4f')} s/batch (p50 {f(r['batch_p50_s'], '.4f')}, p90 {f(r['batch_p90_s'], '.4f')}) over {r['measured_batches']} batches"
            + ("  [TRUNCATED]" if r["window_truncated"] else ""),
            f"                   {f(r['optimizer_steps_per_s'])} optimizer steps/s, {f(r['events_per_s_global'], '.1f')} events/s, dataloader wait {f(100 * r['dataloader_wait_frac'], '.1f')}%",
            f"  EPOCH (train)  : {r['train_batches_per_epoch_per_rank']} batches/rank -> {f(r['epoch_train_s'] / 60, '.2f')} min   [{r['batches_per_epoch_source']}]",
            f"  GPU memory     : peak reserved {f(r['gpu_peak_reserved_gib'], '.2f')} GiB (incl. warmup {f(r['gpu_peak_reserved_incl_warmup_gib'], '.2f')})",
            f"  host RAM       : job PSS {f(r['host_job_pss_gib_total'], '.1f')} GiB; node used peak {f(r['host_node_used_peak_gib'], '.1f')} GiB",
            f"  CPU            : {cpu.get('physical_cores')} cores/{cpu.get('logical_cpus')} threads; per rank torch {cpu.get('torch_num_threads')} thr, "
            f"matcher n_jobs {cpu.get('matcher_n_jobs')}, workers {r['num_workers']}; util {f(cpu.get('node_cpu_util_pct_median'), '.0f')}%; "
            f"oversubscribed during matching: {cpu.get('oversubscribed_during_matching')}",
            f"  code           : {pv.get('hepattn_file')}  git {pv.get('git_branch')}@{str(pv.get('git_commit'))[:12]} dirty={pv.get('git_dirty')}",
            f"                   encoder {pv.get('encoder_class')} attn={pv.get('encoder_attn_type')} gmp_layers={pv.get('encoder_gmp_layers')} impl={pv.get('encoder_gmp_impl')}",
            "=" * 100,
        ]
        for line in lines:
            print(line, flush=True)

    def _print_final(self, r: dict[str, Any]) -> None:
        f = self._f
        print(
            f"BENCH3 [{r['tag']}] validation: {r['val_status']}; first pass {f(r.get('val_wall_s_first'), '.1f')} s, steady {f(r.get('epoch_val_s_steady'), '.1f')} s/epoch",
            flush=True,
        )
        print(
            f"BENCH3 [{r['tag']}] EPOCH TOTAL {f(r['epoch_total_s'] / 60, '.2f')} min (train {f(r['epoch_train_s'] / 60, '.2f')} + val, included: {r['epoch_total_includes_val']})",
            flush=True,
        )
        bd = r.get("breakdown")
        if bd:
            b = bd["rank0"]
            print(
                f"BENCH3 [{r['tag']}] breakdown (rank 0, median of {bd['instrumented_batches']}): batch {f(b['batch_ms'], '.0f')} ms = "
                f"fwd+loss {f(b['fwd_loss_host_ms'], '.0f')} (matcher cpu {f(b['matcher_cpu_ms'], '.0f')} + gpu-wait {f(b['matcher_gpu_wait_ms'], '.0f')}) + "
                f"bwd {f(b['bwd_host_ms'], '.0f')} + clip/opt {f(b['clip_opt_host_ms'], '.0f')} ms; syncs/batch {f(b['syncs'], '.0f')}",
                flush=True,
            )
        print(f"BENCH3 [{r['tag']}] profile: {r['profile_status']}", flush=True)
