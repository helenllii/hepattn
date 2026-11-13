"""Forward/backward smoke test for one bench3 arm, built from EXACTLY the config stack the benchmark uses.

Exit codes: 0 PASS, 1 FAIL, 3 OOM (the common micro-batch must then change for ALL arms).

Checks, on one real CLIC batch:
  * the model builds from base.yaml + configs/bench3/common.yaml + configs/bench3/<arm>.yaml;
  * forward + matching + loss run, and the total loss is finite;
  * backward gives a finite gradient to every trainable parameter. A parameter with NO gradient is an error by
    default: under DDP (strategy auto, find_unused_parameters False) it makes the multi-GPU benchmark hang or crash,
    and that is far cheaper to learn here;
  * one optimizer step (the configured optimizer: Lion/AdamW at lrs_config.initial) leaves a finite loss on the
    next forward.

Correctness runs eager; then (CUDA) the encoder and decoder are compiled exactly as hepattn.callbacks.Compile does
and `--compiled-steps` steps give the compiled memory peak. bf16 autocast on CUDA, fp32 on CPU. Not a timing measurement.
The peak is single-process (no DDP buckets: ~2 x 50 MB for 12M params, negligible against the limit).

  python -m hepattn.experiments.clic.bench3.smoke --arm linformer_encoder --data-dir /eagle/hgcal-maskformer-fpga/clic_data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import torch

CLIC_DIR = Path(__file__).resolve().parents[1]
ARMS = tuple(sorted(p.stem for p in (CLIC_DIR / "configs" / "bench3").glob("*.yaml") if p.stem != "common"))


def build_cli(arm: str, data_dir: str, batch_size: int, accelerator: str, workdir: str, extra: list[str]):
    """Instantiate model, datamodule and trainer through the same CLI and config files as main.py, without fitting."""
    from hepattn.experiments.clic.lightning_module import MPflow  # noqa: PLC0415
    from hepattn.experiments.clic.pflow_data import PflowDataModule  # noqa: PLC0415
    from hepattn.utils.cli import CLI  # noqa: PLC0415

    class SmokeCLI(CLI):
        # The parent hooks assume a subcommand (fit/test); with run=False there is none. Only the matmul precision
        # setting matters for a forward/backward check.
        def before_instantiate_classes(self) -> None:
            if self.config.get("matmul_precision"):
                torch.set_float32_matmul_precision(self.config["matmul_precision"])

        def after_instantiate_classes(self) -> None:
            pass

    cfg = CLIC_DIR / "configs"
    args = [
        "-c", str(cfg / "bench3" / "common.yaml"),
        "-c", str(cfg / "bench3" / f"{arm}.yaml"),
        "--data.train_path", f"{data_dir}/train_clic_fix.root",
        "--data.valid_path", f"{data_dir}/val_clic_fix.root",
        "--data.num_train", str(4 * batch_size),
        "--data.num_val", "64",
        "--data.num_workers", "0",
        "--data.batch_size", str(batch_size),
        "--trainer.accelerator", accelerator,
        "--trainer.devices", "1",
        "--trainer.default_root_dir", workdir,
        "--trainer.logger.init_args.save_dir", workdir,
        *extra,
    ]  # fmt: skip
    return SmokeCLI(
        model_class=MPflow,
        datamodule_class=PflowDataModule,
        args=args,
        run=False,
        parser_kwargs={"default_env": True, "default_config_files": [str(cfg / "base.yaml")]},
    )


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def flatten_losses(losses) -> list[tuple[str, torch.Tensor]]:
    out = []

    def walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}/{k}" if prefix else str(k), v)
        elif isinstance(node, torch.Tensor):
            out.append((prefix, node))

    walk("", losses)
    return out


def load_one_batch(arm: str, data_dir: str, batch_size: int, cuda: bool, workdir: str, extra: list[str]):
    """Build the arm through the benchmark's CLI config stack and load one real training batch."""
    cli = build_cli(arm, data_dir, batch_size, "gpu" if cuda else "cpu", workdir, extra)
    dm = cli.datamodule
    dm.trainer = cli.trainer  # PflowDataModule.setup reads trainer.is_global_zero
    dm.setup("fit")
    inputs, targets = next(iter(dm.train_dataloader()))
    return cli, inputs, targets


def describe(model) -> dict:
    import inspect  # noqa: PLC0415

    import hepattn  # noqa: PLC0415

    enc = model.encoder
    gmps = [type(layer.gmp).__qualname__ for layer in getattr(enc, "layers", []) if getattr(layer, "gmp", None) is not None]
    return {
        "hepattn_file": hepattn.__file__,
        "encoder_class": f"{type(enc).__module__}.{type(enc).__qualname__}",
        "encoder_file": inspect.getfile(type(enc)),
        "attn_type": getattr(enc, "attn_type", None),
        "gmp_layers": [i for i, layer in enumerate(getattr(enc, "layers", [])) if getattr(layer, "gmp", None) is not None],
        "gmp_impl": gmps[0] if gmps else None,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def run(
    arm: str, data_dir: str, batch_size: int, device: str, allow_unused: bool, extra: list[str], compiled_steps: int, mem_limit_gib: float
) -> dict:
    """Returns a result dict; status PASS / FAIL / OOM."""
    cuda = device == "cuda"
    res: dict = {"arm": arm, "batch_size": batch_size, "device": device, "status": "FAIL", "fail": []}
    with tempfile.TemporaryDirectory(prefix="bench3_smoke_") as workdir:
        t0 = time.perf_counter()
        cli, inputs, targets = load_one_batch(arm, data_dir, batch_size, cuda, workdir, extra)
        wrapper = cli.model
        res["load_s"] = time.perf_counter() - t0
        print(f"[smoke:{arm}] built + loaded one batch of {batch_size} in {res['load_s']:.1f} s", flush=True)

        model = wrapper.model.to(device).train()
        res.update(describe(model))
        print(f"[smoke:{arm}] hepattn {res['hepattn_file']}", flush=True)
        print(
            f"[smoke:{arm}] encoder {res['encoder_class']} ({res['encoder_file']}) attn_type={res['attn_type']} gmp_layers={res['gmp_layers']} gmp_impl={res['gmp_impl']}",
            flush=True,
        )
        print(f"[smoke:{arm}] {type(model).__name__} trainable params {res['trainable_params']:,}", flush=True)

        from hepattn.models.wrapper import AdamW, Lion  # noqa: PLC0415 - the classes configure_optimizers uses

        opt_cls = {"lion": Lion, "adamw": AdamW}[wrapper.optimizer.lower()]
        opt = opt_cls(model.parameters(), lr=float(wrapper.lrs_config["initial"]), weight_decay=float(wrapper.lrs_config["weight_decay"]))

        def step(tag: str) -> float:
            # Fresh copies each time: the loss path is allowed to modify its inputs in place.
            x, y = to_device(inputs, device), to_device(targets, device)
            t = time.perf_counter()
            with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
                out = model(x)
                out, y, losses = model.loss(out, y)
            parts = flatten_losses(losses)
            total = sum(v.float() for _, v in parts)
            bad = [k for k, v in parts if not torch.isfinite(v).all()]
            if bad:
                raise FloatingPointError(f"non-finite loss terms: {bad}")
            opt.zero_grad(set_to_none=True)
            total.backward()
            if cuda:
                torch.cuda.synchronize()
            print(f"[smoke:{arm}] {tag}: loss {total.item():.5f} ({len(parts)} terms), fwd+bwd {time.perf_counter() - t:.2f} s", flush=True)
            return total.item()

        def peak() -> float:
            return torch.cuda.max_memory_reserved() / 2**30 if cuda else float("nan")

        try:
            # ---- correctness, eager
            loss0 = step("eager step 0")
            no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            nonfinite = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            grad_norm = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))
            print(f"[smoke:{arm}] grad norm {grad_norm:.4g}; params without grad: {len(no_grad)}; non-finite grads: {len(nonfinite)}", flush=True)
            res.update({"loss0": loss0, "grad_norm": grad_norm, "params_without_grad": no_grad[:20], "n_params_without_grad": len(no_grad)})
            if nonfinite:
                res["fail"].append(f"non-finite gradients in {nonfinite[:10]}")
            if no_grad:
                msg = f"{len(no_grad)} trainable params got no gradient (DDP without find_unused_parameters will fail): {no_grad[:10]}"
                (print(f"[smoke:{arm}] WARNING {msg}", flush=True) if allow_unused else res["fail"].append(msg))
            if not math.isfinite(grad_norm) or grad_norm == 0.0:
                res["fail"].append(f"gradient norm is {grad_norm}")
            opt.step()
            loss1 = step("eager step 1 (after one optimizer step)")
            res["loss1"] = loss1
            if not math.isfinite(loss1):
                res["fail"].append("loss non-finite after one optimizer step")
            res["eager_peak_reserved_gib"] = peak()

            # ---- memory, compiled exactly as hepattn.callbacks.Compile does (encoder and decoder, dynamic=True)
            if compiled_steps > 0 and cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                for name in ("encoder", "decoder"):
                    getattr(model, name).compile(dynamic=True)
                t = time.perf_counter()
                for i in range(compiled_steps):
                    step(f"compiled step {i}")
                    opt.step()
                    if i == 0:
                        res["compile_first_step_s"] = time.perf_counter() - t
                res["compiled_peak_reserved_gib"] = peak()
                res["compiled_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        except torch.OutOfMemoryError as exc:
            res["status"] = "OOM"
            res["oom"] = str(exc).splitlines()[0]
            res["peak_reserved_at_oom_gib"] = peak()
            print(f"[smoke:{arm}] OOM at batch {batch_size}: {res['oom']}", flush=True)
            return res

        worst = (
            max(v for v in (res.get("eager_peak_reserved_gib"), res.get("compiled_peak_reserved_gib")) if v is not None and math.isfinite(v))
            if cuda
            else float("nan")
        )
        res["peak_reserved_gib"] = worst
        if cuda:
            print(
                f"[smoke:{arm}] peak GPU memory at batch {batch_size}: eager {res['eager_peak_reserved_gib']:.2f} GiB, "
                f"compiled {res.get('compiled_peak_reserved_gib', float('nan')):.2f} GiB (limit {mem_limit_gib} GiB)",
                flush=True,
            )
            if worst > mem_limit_gib:
                res["fail"].append(f"peak reserved {worst:.2f} GiB exceeds {mem_limit_gib} GiB")
        if res["fail"]:
            for f in res["fail"]:
                print(f"[smoke:{arm}] FAIL: {f}", flush=True)
            return res
        res["status"] = "PASS"
        print(f"[smoke:{arm}] PASS (loss {loss0:.4f} -> {loss1:.4f})", flush=True)
        return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--compiled-steps", type=int, default=3, help="steps after compiling encoder+decoder, for the compiled memory peak (CUDA only)")
    p.add_argument("--mem-limit-gib", type=float, default=38.0, help="fail if peak reserved exceeds this (A100-40GB: 39.5 usable)")
    p.add_argument("--out", help="write the result dict as JSON here")
    p.add_argument("--data-dir", default="/eagle/hgcal-maskformer-fpga/clic_data")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--allow-unused", action="store_true", help="warn instead of fail on params without gradient")
    p.add_argument("extra", nargs="*", help="extra main.py-style overrides, after --")
    a = p.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    res = run(a.arm, a.data_dir, a.batch_size, device, a.allow_unused, a.extra, a.compiled_steps, a.mem_limit_gib)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, default=str, sort_keys=True) + "\n")
    return {"PASS": 0, "OOM": 3}.get(res["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
