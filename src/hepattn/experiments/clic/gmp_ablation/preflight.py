"""Preflight of a GMP ablation run's RESOLVED config: the execution setup is enforced before any GPU work.

Every ablation run must be ONE model trained by 4-GPU DDP on one node, micro-batch 256 per GPU, accumulation 1
(global batch 1024), bf16-mixed, torch.compile, full training set, the shared validation selection -- and the run's
arm, seed and epoch budget. `submit_gmp_run.sh` resolves the config with `main.py fit --print_config` and runs

    python preflight.py <resolved.yaml> --arm-config <sweep/arm.yaml> --epochs N --seed S

which exits 1 with every violation listed, so a wrong GPU / DDP / accumulation setup fails before training starts.
(The DDPGuard callback then checks the same at run time, inside the live process group.)

Stdlib + PyYAML only: it never imports torch or hepattn, so it is safe to run anywhere, including a login node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REQUIRED = {
    "devices": 4,
    "num_nodes": 1,
    "accumulate_grad_batches": 1,
    "micro_batch": 256,
    "global_batch": 1024,
    "precision": "bf16-mixed",
}
REQUIRED_CALLBACKS = (
    "hepattn.callbacks.Compile",
    "hepattn.callbacks.DynamoOptimizeDDP",
    "hepattn.experiments.clic.gmp_ablation.ddp_guard.DDPGuard",
    "hepattn.experiments.clic.gmp_ablation.diagnostics.GMPDiagnostics",
)


def _get(cfg: dict, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def check(resolved: dict, arm: dict, epochs: int, seed: int) -> list[str]:
    """All violations of the required setup in a resolved `main.py fit` config (empty list = pass)."""
    bad = []

    def want(path, value):
        got = _get(resolved, path)
        if got != value:
            bad.append(f"{path} = {got!r}, required {value!r}")

    want("trainer.devices", REQUIRED["devices"])
    want("trainer.num_nodes", REQUIRED["num_nodes"])
    want("trainer.accumulate_grad_batches", REQUIRED["accumulate_grad_batches"])
    want("trainer.precision", REQUIRED["precision"])
    want("data.batch_size", REQUIRED["micro_batch"])
    want("data.num_train", -1)
    want("data.num_val", 5000)
    want("trainer.max_epochs", epochs)
    want("seed_everything", seed)
    strategy = _get(resolved, "trainer.strategy")
    if strategy not in {"auto", "ddp"}:
        bad.append(f"trainer.strategy = {strategy!r}, required 'auto' or 'ddp' (Lightning's DDP for 4 devices)")
    gb = (_get(resolved, "data.batch_size") or 0) * (_get(resolved, "trainer.devices") or 0) * (_get(resolved, "trainer.num_nodes") or 0)
    gb *= _get(resolved, "trainer.accumulate_grad_batches") or 0
    if gb != REQUIRED["global_batch"]:
        bad.append(f"global batch = batch_size x devices x num_nodes x accumulate = {gb}, required {REQUIRED['global_batch']}")
    paths = [c.get("class_path") if isinstance(c, dict) else str(c) for c in (_get(resolved, "trainer.callbacks") or [])]
    bad += [f"callback {cb} missing from trainer.callbacks" for cb in REQUIRED_CALLBACKS if cb not in paths]
    if len(paths) != len(set(paths)):
        bad.append(f"duplicate callbacks: {paths}")
    # the arm's model settings must be what the resolved config will build
    enc = _get(resolved, "model.model.init_args.encoder.init_args") or {}
    arm_enc = _get(arm, "model.model.init_args.encoder.init_args") or {}
    for key in ("gmp", "gmp_layers", "gmp_impl"):
        if enc.get(key) != arm_enc.get(key):
            bad.append(f"encoder {key} = {enc.get(key)!r}, the arm config says {arm_enc.get(key)!r}")
    if arm_enc.get("gmp") is not None and enc.get("gmp_impl") != "weightsfirst":
        bad.append(f"encoder gmp_impl = {enc.get('gmp_impl')!r}, required 'weightsfirst' for GMP arms")
    for key, value in (arm.get("data") or {}).items():
        if _get(resolved, f"data.{key}") != value:
            bad.append(f"data.{key} = {_get(resolved, f'data.{key}')!r}, the arm config says {value!r}")
    want("model.model.init_args.input_coord_fields", _get(arm, "model.model.init_args.input_coord_fields"))
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("resolved", type=Path)
    p.add_argument("--arm-config", type=Path, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    bad = check(yaml.safe_load(args.resolved.read_text()), yaml.safe_load(args.arm_config.read_text()), args.epochs, args.seed)
    if bad:
        print("PREFLIGHT FAILED:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"preflight ok: 4-GPU DDP, micro-batch 256 x 4 x accumulate 1 = 1024, bf16-mixed, compile, epochs {args.epochs}, seed {args.seed}")


if __name__ == "__main__":
    main()
