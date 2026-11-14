"""Check that a segmented run resumed exactly: the restored state equals the checkpoint, and the LR schedule is continuous.

    python resume_check.py <run dir> --epochs N

For the LAST attempt that resumed from a checkpoint (attempts/*/diag.json.resume_audit.json with a ckpt_path), loads
that checkpoint and compares it with the state the job started from (ResumeAudit, recorded after Lightning's restore):
  * global_step and epoch: the job starts at the checkpoint's global_step, and at the epoch after the checkpoint's;
  * model weights: float64 sum of the checkpoint's state_dict parameters == the restored parameters' sum;
  * optimizer: the sum of every optimizer-state tensor (Lion's moments) is equal;
  * scheduler: last_epoch, _step_count, total_steps and _last_lr are equal;
  * learning-rate schedule: every LR logged by LearningRateMonitor in ANY attempt's metrics.csv equals a freshly built
    OneCycleLR (the checkpoint's total_steps / max_lr / initial_lr / final LR / pct_start) stepped to that global step
    -- one uninterrupted schedule across the segments;
  * coverage: the epochs trained over all attempts are exactly 0..N-1, each once (nothing repeated or skipped).
Exits 1 with every failed check listed. Needs torch (run it in the job, not on a login node).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


def _sum(obj) -> float:
    if isinstance(obj, torch.Tensor):
        return float(obj.double().sum().item()) if obj.is_floating_point() else 0.0
    if isinstance(obj, dict):
        return sum(_sum(v) for v in obj.values())
    if isinstance(obj, list | tuple):
        return sum(_sum(v) for v in obj)
    return 0.0


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def onecycle_lrs(sched_state: dict, group: dict, steps: list[int]) -> dict[int, float]:
    """LR of a fresh, uninterrupted OneCycleLR with the checkpoint's hyperparameters, at each global step.

    torch's OneCycleLR state keeps total_steps and the phase table, and the optimizer's param group keeps max_lr,
    initial_lr (= max_lr / div_factor) and min_lr (= initial_lr / final_div_factor); pct_start is recovered from the
    first phase's end_step (= pct_start * total_steps - 1).
    """
    total = sched_state["total_steps"]
    pct = (float(sched_state["_schedule_phases"][0]["end_step"]) + 1.0) / total
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=group["max_lr"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=group["max_lr"],
        total_steps=total,
        div_factor=group["max_lr"] / group["initial_lr"],
        final_div_factor=group["initial_lr"] / group["min_lr"],
        pct_start=pct,
        anneal_strategy=sched_state.get("_anneal_func_type", "cos"),
        cycle_momentum=False,
        three_phase=len(sched_state["_schedule_phases"]) == 3,
    )
    out = {}
    wanted = set(steps)
    for k in range(max(steps) + 1):
        if k in wanted:
            out[k] = opt.param_groups[0]["lr"]
        opt.step()
        sch.step()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--epochs", type=int, required=True)
    args = p.parse_args()
    bad, ok = [], []
    audits = sorted(args.run_dir.glob("attempts/*/diag.json.resume_audit.json"))
    resumed = [json.loads(a.read_text()) for a in audits]
    resumed = [a for a in resumed if a.get("ckpt_path")]
    if not resumed:
        print("RESUME CHECK FAILED: no attempt resumed from a checkpoint")
        sys.exit(1)
    audit = resumed[-1]
    ckpt = torch.load(audit["ckpt_path"], map_location="cpu", weights_only=False)

    # step / epoch
    if audit["global_step"] == ckpt["global_step"]:
        ok.append(f"global_step restored: {audit['global_step']}")
    else:
        bad.append(f"global_step {audit['global_step']} != checkpoint {ckpt['global_step']}")
    if audit["current_epoch"] == ckpt["epoch"] + 1:
        ok.append(f"resumed at epoch {audit['current_epoch']} after checkpoint epoch {ckpt['epoch']}")
    else:
        bad.append(f"resumed at epoch {audit['current_epoch']}, checkpoint epoch {ckpt['epoch']} (expected +1)")
    # weights, optimizer, scheduler
    w = _sum(dict(ckpt["state_dict"]))
    (ok if _close(w, audit["param_sum"]) else bad).append(f"model weight sum: checkpoint {w!r} vs restored {audit['param_sum']!r}")
    o = _sum(ckpt["optimizer_states"][0]["state"])
    (ok if _close(o, audit["optimizer_state_sum"]) else bad).append(
        f"optimizer state sum: checkpoint {o!r} vs restored {audit['optimizer_state_sum']!r}"
    )
    cs = ckpt["lr_schedulers"][0]
    rs = audit["schedulers"][0]
    for key in ("last_epoch", "_step_count", "total_steps", "_last_lr"):
        (ok if cs.get(key) == rs.get(key) else bad).append(f"scheduler {key}: checkpoint {cs.get(key)!r} vs restored {rs.get(key)!r}")
    # LR schedule across every attempt's CSV log
    logged = {}
    for f in args.run_dir.rglob("metrics.csv"):
        for r in csv.DictReader(f.open()):
            lr_cols = [k for k in r if k.startswith("lr-") and r[k] not in {"", None}]
            if lr_cols and r.get("step") not in {"", None}:
                logged[int(float(r["step"]))] = float(r[lr_cols[0]])
    if logged:
        want = onecycle_lrs(cs, ckpt["optimizer_states"][0]["param_groups"][0], sorted(logged))
        worst = max(abs(logged[s] - want[s]) / max(want[s], 1e-30) for s in logged)
        steps = sorted(logged)
        msg = f"LR schedule: {len(logged)} logged LRs (steps {steps[0]}..{steps[-1]}) vs uninterrupted OneCycleLR, max rel diff {worst:.2e}"
        (ok if worst <= 1e-6 else bad).append(msg)
    else:
        bad.append("no LR logged in any metrics.csv (LearningRateMonitor)")
    # epoch coverage over all attempts
    trained = []
    for d in sorted(args.run_dir.glob("attempts/*/diag.json.rank0.json")):
        trained += [e["epoch"] for e in json.loads(d.read_text()).get("epochs", [])]
    if sorted(trained) == list(range(args.epochs)):
        ok.append(f"epochs trained over all attempts: {sorted(trained)} (each once)")
    else:
        bad.append(f"epochs trained over all attempts {sorted(trained)}, expected 0..{args.epochs - 1} each once")
    print("\n".join(f"  ok   {m}" for m in ok))
    if bad:
        print("RESUME CHECK FAILED:\n" + "\n".join(f"  FAIL {m}" for m in bad))
        sys.exit(1)
    print("RESUME CHECK OK")


if __name__ == "__main__":
    main()
