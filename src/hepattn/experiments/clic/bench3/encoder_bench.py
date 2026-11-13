"""Encoder-only cost of one bench3 arm: the encoder's own forward/backward, without matcher, decoder, DDP or optimizer.

The encoder's exact inputs (tokens after the input nets, sort value, mask, GMP coordinates) are captured from one real
training batch with a forward pre-hook during an eager full-model forward, then the encoder alone is timed:

  * eager forward, eager forward+backward;
  * compiled (`encoder.compile(dynamic=True)`, as hepattn.callbacks.Compile does) forward+backward, plus the compile time;
  * for arms with GMP: the geometry construction (`gmp_neighbourhood`, once per encoder forward) and ONE GMP module's
    forward+backward, eager and compiled, on the captured geometry. Multiply the module time by the number of GMP layers
    for the per-encoder GMP cost.

bf16 autocast, 1 GPU, CUDA-event timing, medians after warmup. Compare `compiled_fwdbwd_ms` with the end-to-end
s/batch of the same arm (bench3 result) to get the encoder's share of a training step.

  python -m hepattn.experiments.clic.bench3.encoder_bench --arm phat16_gmp_einsumopt_early --batch-size 256 --out r.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

from hepattn.experiments.clic.bench3.smoke import ARMS, describe, load_one_batch, to_device


def cuda_time(fn, warmup: int, iters: int) -> float:
    """Median milliseconds per call, CUDA events around each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def clone_inputs(obj, requires_grad_float: bool):
    if isinstance(obj, torch.Tensor):
        t = obj.detach().clone()
        if requires_grad_float and t.is_floating_point():
            t.requires_grad_(True)
        return t
    if isinstance(obj, dict):
        return {k: clone_inputs(v, False) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(clone_inputs(v, False) for v in obj)
    return obj


def run(arm: str, data_dir: str, batch_size: int, warmup: int, iters: int) -> dict:
    import hepattn.models.encoder as encmod  # noqa: PLC0415

    res: dict = {"arm": arm, "batch_size": batch_size, "warmup": warmup, "iters": iters}
    with tempfile.TemporaryDirectory(prefix="bench3_enc_") as workdir:
        cli, inputs, targets = load_one_batch(arm, data_dir, batch_size, True, workdir, [])
        model = cli.model.model.to("cuda").train()
        res.update(describe(model))
        enc = model.encoder

        # ---- capture the encoder's inputs, the GMP geometry inputs and one GMP module's inputs, from a real forward
        cap: dict = {}

        def keep(key):
            def hook(module, args, kwargs=None):
                cap.setdefault(key, (args, kwargs) if kwargs is not None else args)
                # None: a pre-hook's return value would replace the module's inputs

            return hook

        h = enc.register_forward_pre_hook(keep("enc"), with_kwargs=True)
        gmp_layers = [i for i, layer in enumerate(getattr(enc, "layers", [])) if getattr(layer, "gmp", None) is not None]
        gh = None
        if gmp_layers:
            gmp_mod = enc.layers[gmp_layers[0]].gmp
            gh = gmp_mod.register_forward_pre_hook(keep("gmp"))
            orig_geom = encmod.gmp_neighbourhood

            def capture_geom(*a, **k):
                cap.setdefault("geom", (a, k))
                return orig_geom(*a, **k)

            encmod.gmp_neighbourhood = capture_geom
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(to_device(inputs, "cuda"))
        finally:
            h.remove()
            if gh is not None:
                gh.remove()
                encmod.gmp_neighbourhood = orig_geom
        enc_args, enc_kwargs = cap["enc"]
        x0 = enc_args[0] if enc_args else enc_kwargs["x"]
        res["encoder_input_shape"] = list(x0.shape)
        res["real_tokens_frac"] = float(enc_kwargs["kv_mask"].float().mean()) if enc_kwargs.get("kv_mask") is not None else None

        def enc_call(compiled_mod=None):
            a, k = clone_inputs(enc_args, True), clone_inputs(enc_kwargs, False)
            if not a:
                k["x"].requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return (compiled_mod or enc)(*a, **k)

        grad_seed = torch.randn(x0.shape, device="cuda", dtype=torch.float32)

        def fwd():
            with torch.no_grad():
                enc_call()

        def fwdbwd():
            enc.zero_grad(set_to_none=True)
            out = enc_call()
            (out.float() * grad_seed).sum().backward()

        torch.cuda.reset_peak_memory_stats()
        res["encoder_eager_fwd_ms"] = cuda_time(fwd, warmup, iters)
        res["encoder_eager_fwdbwd_ms"] = cuda_time(fwdbwd, warmup, iters)
        res["encoder_eager_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30

        # ---- GMP pieces (eager), before the encoder is compiled
        if gmp_layers:
            res["gmp_layers"] = gmp_layers
            ga, gk = cap["geom"]
            res["gmp_geometry_eager_ms"] = cuda_time(lambda: orig_geom(*ga, **gk), warmup, iters)
            gx, neigh, in_grid = cap["gmp"]
            res["gmp_neighbour_matrix_shape"] = list(neigh.shape)
            res["gmp_neighbour_density"] = float((neigh != 0).float().mean())
            g_seed = torch.randn(gx.shape, device="cuda")

            def gmp_fwdbwd(mod):
                gmp_mod.zero_grad(set_to_none=True)
                xx = gx.detach().clone().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = mod(xx, neigh, in_grid)
                (out.float() * g_seed).sum().backward()

            res["gmp_module_eager_fwdbwd_ms"] = cuda_time(lambda: gmp_fwdbwd(gmp_mod), warmup, iters)
            cg = torch.compile(gmp_mod, dynamic=True)
            t = time.perf_counter()
            gmp_fwdbwd(cg)
            torch.cuda.synchronize()
            res["gmp_module_compile_s"] = time.perf_counter() - t
            res["gmp_module_compiled_fwdbwd_ms"] = cuda_time(lambda: gmp_fwdbwd(cg), warmup, iters)
            cgeom = torch.compile(orig_geom, dynamic=True)
            res["gmp_geometry_compiled_ms"] = cuda_time(lambda: cgeom(*ga, **gk), warmup, iters)

        # ---- compiled encoder, exactly as the Compile callback compiles it
        enc.compile(dynamic=True)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        fwdbwd()
        torch.cuda.synchronize()
        res["encoder_compile_s"] = time.perf_counter() - t
        res["encoder_compiled_fwd_ms"] = cuda_time(fwd, warmup, iters)
        res["encoder_compiled_fwdbwd_ms"] = cuda_time(fwdbwd, warmup, iters)
        res["encoder_compiled_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--data-dir", default="/eagle/hgcal-maskformer-fpga/clic_data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out")
    a = p.parse_args(argv)
    if not torch.cuda.is_available():
        print("encoder_bench needs CUDA", file=sys.stderr)
        return 2
    res = run(a.arm, a.data_dir, a.batch_size, a.warmup, a.iters)
    for k in sorted(res):
        if k.endswith(("_ms", "_s", "_gib", "density", "frac")):
            print(
                f"[encoder_bench:{a.arm}] {k} = {res[k]:.4g}" if isinstance(res[k], float) else f"[encoder_bench:{a.arm}] {k} = {res[k]}", flush=True
            )
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, default=str, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
