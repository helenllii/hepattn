"""Verdict on one finished training JOB of a GMP ablation run; its exit code decides whether the PBS chain advances.

    python runcheck.py <run dir> --attempt-dir <run dir>/attempts/<job> --epochs N [--stop-at-epoch E]

Reads the job's GMPDiagnostics record (rank 0) and exits
  0  the job did what it was meant to: trained through epoch E-1 (a segment) or N-1 (the whole run), with zero
     non-finite losses and gradients. A finished RUN gets <run dir>/DONE.json, a finished segment
     <run dir>/segments/upto_<E>.json;
  3  a non-finite loss or gradient was seen (the run is numerically broken: the chain must stop);
  4  the job ended before its target epoch (walltime, preemption, crash) or left no diagnostics.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def verdict(diag: dict, epochs: int, stop_at: int | None) -> tuple[int, str]:
    target = min(stop_at, epochs) if stop_at else epochs
    health = diag.get("health") or {}
    if health.get("nonfinite_loss_batches", 0) or health.get("nonfinite_grad_steps", 0):
        return 3, f"non-finite values: {health}"
    done = [e["epoch"] for e in diag.get("epochs", [])]
    last = max(done) if done else -1
    if last + 1 < target:
        return 4, f"stopped after epoch {last} (0-based), target was epoch {target - 1}"
    if not health:
        return 4, "no health record: the job did not reach the end of an epoch cleanly"
    return 0, f"trained through epoch {last} of {epochs}; health {health}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--attempt-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--stop-at-epoch", type=int, default=None)
    args = p.parse_args()
    files = sorted(args.attempt_dir.glob("diag.json.rank0.json"))
    if not files:
        print(f"RUNCHECK FAIL (4): no diagnostics in {args.attempt_dir}")
        sys.exit(4)
    diag = json.loads(files[0].read_text())
    code, why = verdict(diag, args.epochs, args.stop_at_epoch)
    stamp = {
        "job": os.environ.get("PBS_JOBID"),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verdict": why,
        "epochs_trained": [e["epoch"] for e in diag.get("epochs", [])],
        "health": diag.get("health"),
    }
    if code == 0:
        final = not args.stop_at_epoch or args.stop_at_epoch >= args.epochs
        if final:
            (args.run_dir / "DONE.json").write_text(json.dumps(stamp | {"epochs": args.epochs}, indent=1))
        else:
            seg = args.run_dir / "segments"
            seg.mkdir(exist_ok=True)
            (seg / f"upto_{args.stop_at_epoch}.json").write_text(json.dumps(stamp, indent=1))
        print(f"RUNCHECK OK: {why}{' -> DONE' if final else ''}")
    else:
        print(f"RUNCHECK FAIL ({code}): {why}")
    sys.exit(code)


if __name__ == "__main__":
    main()
