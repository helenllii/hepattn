"""A short, self-terminating training-throughput benchmark, as a Lightning callback.

It runs the REAL training loop -- real data, real matcher, real optimizer -- discards a warmup
window, measures a fixed number of steps, prints a report and then sets `trainer.should_stop`.
Nothing about the model or the optimization is changed; the only perturbation is one
`torch.cuda.synchronize()` per step, without which a wall-clock step time measures when the work
was *enqueued* rather than when it finished (set `synchronize: false` to drop it).

WHAT IT REPORTS, AND WHY EACH NUMBER IS THERE
  * events/s, globally and per rank. This is the only throughput unit that stays comparable when
    the device count or the micro-batch size changes, so the 1-GPU vs 4-GPU comparison is read
    off it. `it/s` is also reported because that is what tqdm shows, but it counts micro-batches
    and therefore moves when `batch_size` moves.
  * estimated full-epoch time, from `full_train_events / events_per_s`. Taken from the event
    count rather than from `trainer.num_training_batches` so that a benchmark run over a
    truncated dataset (`--data.num_train`) still extrapolates to the real epoch.
  * peak GPU memory, `max_memory_allocated` (tensors) and `max_memory_reserved` (what the caching
    allocator holds, i.e. what has to fit in the card). Reset after warmup so compile and the
    sanity check are excluded. On a shared/MPS device this is the only honest per-process figure:
    `nvidia-smi` reports the whole device.
  * the dataloader-starvation fraction: the share of wall time between one step finishing and the
    next starting, which is where waiting on `__getitem__`/collate shows up. A high value means
    more GPUs will not help until the host side is fixed.
  * sampled GPU utilization, for corroboration only. It is a duty-cycle average, so a host-bound
    run that keeps the GPU busy in short bursts can still read high.

The effective global batch is printed prominently because the 1-vs-4 comparison is only
meaningful at fixed global batch, and it is set by three separate knobs
(`batch_size * devices * num_nodes * accumulate_grad_batches`). `expect_global_batch` turns a
silent mismatch into a loud one.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import threading
import time

import torch
from lightning import Callback, LightningModule, Trainer


class ThroughputBenchmark(Callback):
    """Measure steady-state training throughput over a fixed window, then stop the run.

    Args:
        warmup_steps: Steps discarded before measuring. Must cover `torch.compile` and the
            dataloader worker spin-up, both of which cost tens of seconds on the first steps.
        measure_steps: Steps in the measurement window.
        full_train_events: Events in one real training epoch, used for the full-epoch estimate.
            None falls back to `num_training_batches * global micro-batch`, which is only the real
            epoch when the run is not truncated.
        expect_global_batch: If set, fail unless the effective global batch equals it.
        synchronize: `torch.cuda.synchronize()` at each step boundary. Leave on.
        util_poll_s: GPU-utilization sampling period, seconds. 0 disables sampling.
        tag: Free-form label echoed into the report and the JSON line (e.g. "1gpu", "4gpu").

    Raises:
        ValueError: If the measurement window is empty or negative.
    """

    def __init__(
        self,
        warmup_steps: int = 50,
        measure_steps: int = 150,
        full_train_events: int | None = None,
        expect_global_batch: int | None = None,
        synchronize: bool = True,
        util_poll_s: float = 0.1,
        tag: str = "bench",
    ) -> None:
        super().__init__()
        if warmup_steps < 0 or measure_steps < 1:
            raise ValueError(f"need warmup_steps >= 0 and measure_steps >= 1, got {warmup_steps}, {measure_steps}")
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.full_train_events = full_train_events
        self.expect_global_batch = expect_global_batch
        self.synchronize = synchronize
        self.util_poll_s = util_poll_s
        self.tag = tag

        self._seen = 0
        self._step_s: list[float] = []
        self._wait_s: list[float] = []
        self._t_start: float | None = None
        self._t_prev_end: float | None = None
        self._window_t0: float | None = None
        self._util: list[float] = []
        self._util_stop = threading.Event()
        self._util_thread: threading.Thread | None = None
        self._reported = False

    # ---------------------------------------------------------------- setup and shape bookkeeping
    @staticmethod
    def _micro_batch(trainer: Trainer) -> int:
        """Events per optimizer micro-step on this rank.

        Read from the dataloader first and the datamodule only as a fallback: the dataloader is
        what actually feeds the loop, and a run driven by bare dataloaders has no datamodule.

        Raises:
            ValueError: If neither exposes a batch size.
        """
        for obj in (getattr(trainer, "train_dataloader", None), getattr(trainer, "datamodule", None)):
            size = getattr(obj, "batch_size", None)
            if size:
                return int(size)
        raise ValueError("cannot determine the micro-batch size from the dataloader or the datamodule")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._micro = self._micro_batch(trainer)
        self._world = int(trainer.world_size)
        self._accum = int(trainer.accumulate_grad_batches)
        self._global_batch = self._micro * self._world * self._accum
        if self.expect_global_batch is not None and self._global_batch != self.expect_global_batch:
            raise ValueError(
                f"effective global batch is {self._global_batch} "
                f"(batch_size {self._micro} x world_size {self._world} x accumulate {self._accum}) "
                f"but expect_global_batch is {self.expect_global_batch}. A 1-GPU vs N-GPU "
                f"comparison at different global batch is not a throughput comparison -- divide "
                f"accumulate_grad_batches by the device count instead."
            )
        if trainer.is_global_zero:
            need = self.warmup_steps + self.measure_steps
            print("-" * 80, flush=True)
            print(
                f"[a4bench:{self.tag}] global batch {self._global_batch} = micro {self._micro} x world {self._world} x accum {self._accum}",
                flush=True,
            )
            print(f"[a4bench:{self.tag}] warmup {self.warmup_steps} + measure {self.measure_steps} = {need} steps, then stop", flush=True)
            print("-" * 80, flush=True)

    # ------------------------------------------------------------------------ per-step timing
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        now = time.perf_counter()
        if self._seen == self.warmup_steps:
            # Start of the measurement window: forget everything the warmup allocated or timed.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self._window_t0 = now
            self._t_prev_end = None
            self._start_util_sampler()
        if self._t_prev_end is not None and self._seen >= self.warmup_steps:
            self._wait_s.append(now - self._t_prev_end)
        self._t_start = now

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self._seen >= self.warmup_steps and self._t_start is not None:
            self._step_s.append(now - self._t_start)
        self._t_prev_end = now
        self._seen += 1
        if len(self._step_s) >= self.measure_steps and not self._reported:
            self._window_t1 = now
            self._report(trainer)
            self._reported = True
            trainer.should_stop = True

    # ------------------------------------------------------------------- GPU utilization sampler
    def _start_util_sampler(self) -> None:
        if self.util_poll_s <= 0 or not torch.cuda.is_available() or self._util_thread is not None:
            return
        device = torch.cuda.current_device()

        def poll() -> None:
            while not self._util_stop.wait(self.util_poll_s):
                try:
                    self._util.append(float(torch.cuda.utilization(device)))
                except Exception:  # noqa: BLE001 - pynvml absent or no permission; util is corroboration only
                    return

        self._util_thread = threading.Thread(target=poll, daemon=True, name="a4bench-util")
        self._util_thread.start()

    # ------------------------------------------------------------------------------- the report
    def _report(self, trainer: Trainer) -> None:
        self._util_stop.set()
        wall = self._window_t1 - self._window_t0
        n = len(self._step_s)
        step = sorted(self._step_s)
        p50 = statistics.median(step)
        p10, p90 = step[int(0.10 * (n - 1))], step[int(0.90 * (n - 1))]

        wait_total = sum(self._wait_s)
        wait_frac = wait_total / wall if wall > 0 else float("nan")
        wait_p50 = statistics.median(self._wait_s) if self._wait_s else float("nan")

        # Per-rank rates from the wall clock of the window, so every gap is accounted for.
        local_it_s = n / wall
        local_ev_s = local_it_s * self._micro

        peak_alloc = peak_resv = float("nan")
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / 2**30
            peak_resv = torch.cuda.max_memory_reserved() / 2**30

        # Aggregate across ranks: throughput adds, memory and starvation take the worst rank.
        agg = self._all_reduce(trainer, local_ev_s, peak_alloc, peak_resv, wait_frac)
        global_ev_s, max_alloc, max_resv, max_wait_frac = agg

        if not trainer.is_global_zero:
            return

        full_events = self.full_train_events
        source = "config full_train_events"
        if full_events is None:
            nb = trainer.num_training_batches
            full_events = int(nb) * self._micro * self._world if nb not in {float("inf"), None} else None
            source = "trainer.num_training_batches (ONLY the real epoch if the dataset is not truncated)"
        epoch_s = full_events / global_ev_s if full_events else float("nan")

        util_txt = "unavailable (pynvml not usable)"
        util_mean = None
        if self._util:
            util_mean = statistics.mean(self._util)
            util_txt = f"{util_mean:.0f}% mean, {statistics.median(self._util):.0f}% median over {len(self._util)} samples"

        dl = trainer.train_dataloader
        pw = getattr(dl, "persistent_workers", None)
        pf = getattr(dl, "prefetch_factor", None)
        nw = getattr(dl, "num_workers", None)

        print(flush=True)
        print("=" * 80, flush=True)
        print(f"A4 THROUGHPUT BENCHMARK [{self.tag}]", flush=True)
        print("=" * 80, flush=True)
        print(f"  host              : {platform.node()}", flush=True)
        print(f"  device            : {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'} x {self._world} rank(s)", flush=True)
        print(f"  precision         : {trainer.precision}", flush=True)
        print(f"  global batch      : {self._global_batch}  (micro {self._micro} x world {self._world} x accum {self._accum})", flush=True)
        print(f"  window            : {n} steps in {wall:.1f} s, after {self.warmup_steps} warmup steps", flush=True)
        print(flush=True)
        print("  1. STEADY-STATE RATE", flush=True)
        print(f"     per rank       : {local_it_s:.3f} it/s   ({local_ev_s:.1f} events/s)", flush=True)
        print(f"     global         : {global_ev_s:.1f} events/s   ({global_ev_s / self._global_batch:.4f} optimizer steps/s)", flush=True)
        print(f"     step time      : p50 {p50 * 1e3:.1f} ms   p10 {p10 * 1e3:.1f}   p90 {p90 * 1e3:.1f}", flush=True)
        print(flush=True)
        print("  2. FULL-EPOCH ESTIMATE", flush=True)
        print(f"     events/epoch   : {full_events}  [{source}]", flush=True)
        print(f"     epoch time     : {epoch_s / 60:.1f} min   ({epoch_s / 3600:.2f} h)", flush=True)
        print(f"     400 epochs     : {epoch_s * 400 / 86400:.1f} days", flush=True)
        print(flush=True)
        print("  3. PEAK GPU MEMORY (this process, worst rank)", flush=True)
        print(f"     allocated      : {max_alloc:.2f} GiB", flush=True)
        print(f"     reserved       : {max_resv:.2f} GiB   <- must fit the card (A100-40GB = 39.5 GiB usable)", flush=True)
        print(flush=True)
        print("  4. BOTTLENECK", flush=True)
        print(f"     dataloader wait: {max_wait_frac * 100:.1f}% of wall time (worst rank), p50 {wait_p50 * 1e3:.1f} ms/step", flush=True)
        print(f"     dataloader     : num_workers={nw} persistent_workers={pw} prefetch_factor={pf}", flush=True)
        print(f"     gpu util       : {util_txt}", flush=True)
        verdict = (
            "HOST-BOUND: the dataloader is the limit, more GPUs will not help until it is fixed"
            if max_wait_frac > 0.25
            else ("mixed: host cost is visible but not dominant" if max_wait_frac > 0.10 else "GPU-bound or well overlapped")
        )
        print(f"     verdict        : {verdict}", flush=True)
        print("=" * 80, flush=True)

        # json.dumps writes a bare NaN for a non-finite float, which strict JSON parsers reject.
        # Memory is NaN whenever there is no CUDA device, so it goes out as null instead.
        def jnum(value: float, places: int) -> float | None:
            return round(value, places) if math.isfinite(value) else None

        payload = {
            "tag": self.tag,
            "world_size": self._world,
            "micro_batch": self._micro,
            "accumulate": self._accum,
            "global_batch": self._global_batch,
            "steps": n,
            "window_s": round(wall, 3),
            "it_s_per_rank": round(local_it_s, 4),
            "events_s_per_rank": round(local_ev_s, 2),
            "events_s_global": round(global_ev_s, 2),
            "step_ms_p50": round(p50 * 1e3, 2),
            "step_ms_p10": round(p10 * 1e3, 2),
            "step_ms_p90": round(p90 * 1e3, 2),
            "epoch_min": jnum(epoch_s / 60, 4),
            "full_train_events": full_events,
            "peak_alloc_gib": jnum(max_alloc, 3),
            "peak_reserved_gib": jnum(max_resv, 3),
            "dataloader_wait_frac": jnum(max_wait_frac, 4),
            "num_workers": nw,
            "persistent_workers": pw,
            "prefetch_factor": pf,
            "gpu_util_mean": round(util_mean, 1) if util_mean is not None else None,
            "precision": str(trainer.precision),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "pbs_jobid": os.environ.get("PBS_JOBID"),
        }
        print("A4BENCH_JSON " + json.dumps(payload, sort_keys=True), flush=True)

    def _all_reduce(self, trainer: Trainer, ev_s: float, alloc: float, resv: float, wait: float) -> tuple[float, float, float, float]:
        """Sum the throughput over ranks and take the worst rank for memory and starvation."""
        if trainer.world_size == 1:
            return ev_s, alloc, resv, wait
        try:
            dev = trainer.strategy.root_device
            summed = trainer.strategy.reduce(torch.tensor([ev_s], device=dev), reduce_op="sum")
            worst = trainer.strategy.reduce(torch.tensor([alloc, resv, wait], device=dev), reduce_op="max")
            return float(summed.item()), float(worst[0]), float(worst[1]), float(worst[2])
        except Exception as exc:  # noqa: BLE001 - a failed reduction must not lose the local numbers
            if trainer.is_global_zero:
                print(f"[a4bench] cross-rank reduction failed ({type(exc).__name__}: {exc}); reporting RANK 0 ONLY", flush=True)
            return ev_s * trainer.world_size, alloc, resv, wait
