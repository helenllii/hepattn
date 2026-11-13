"""Micro-benchmark of the GMP backends: production (A), EinsumOpt (A'), weights-first (B: plain-autograd, fused, loop).

One GMP layer -- masking, depthwise conv over the occupied cells, bias, pointwise Linear, LayerNorm, residual, exactly
the production module's structure -- timed in isolation for every (variant, K, N, precision, mode). The layer is
`GMPLayer` below, generic in the number of kernel offsets K so that K = 27 (a 3x3x3 kernel, the scaling proxy for 3D
GMP) can be measured without modifying the production class. For K = 9 and 25, `GMPLayer` with the production core
is bit-identical to `GeometricMessagePassing` (asserted in tests/models/test_gmp_weights_first.py), so those rows are
production's numbers.

Per configuration (all on one GPU, CUDA-synchronized, after warmup):
  * correctness vs eager production at the same precision (forward, input grad, depthwise-weight grad);
  * first-call wall time of forward and of backward (= compile time under torch.compile, measured cold);
  * steady-state forward / backward / forward+backward latency from CUDA events (median, p90);
  * memory retained between forward and backward (allocated after forward - before, minus the output);
  * peak allocated during forward+backward, above the inputs;
  * eager only: what autograd saved, via saved_tensors_hooks, split into tensors aliasing the inputs/parameters and
    new activations, with the largest new tensors listed.

Results are one JSON line per configuration, after a provenance line (git commit + dirty state, file sha256s, torch/
triton/CUDA versions, GPU, PBS job). Nothing here is a training run. GPU only; run it through the PBS job
`submit_gmp_backends_bench.sh`, not on a login node.

    python -m hepattn.experiments.clic.a4.bench_gmp_backends --modes eager --precisions fp32 --out r.jsonl
    python -m hepattn.experiments.clic.a4.bench_gmp_backends --probe-dtypes --out probe.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch import Tensor, nn

from hepattn.experiments.clic.a4.gmp_weights_first import (
    einsumopt_conv,
    production_conv,
    weights_first_conv,
    weights_first_conv_autograd,
)
from hepattn.models.encoder import GeometricMessagePassing, gmp_neighbourhood

VARIANTS = {
    "A_production": production_conv,
    "A1_einsumopt": einsumopt_conv,
    "B0_wf_autograd": weights_first_conv_autograd,
    "B_wf_fused": partial(weights_first_conv, strategy="fused"),
    "B_wf_loop": partial(weights_first_conv, strategy="loop"),
}
# K -> geometry: 9 and 25 are production's 2D grid with kernel 3 and 5; 27 is the 3x3x3 proxy below
GEOMETRY = {
    9: {"dims": 2, "kernel_size": 3},
    25: {"dims": 2, "kernel_size": 5},
    27: {"dims": 3, "kernel_size": 3},
}
GRID = {"eta_cell": 0.2, "phi_cells": 32, "eta_max": 5.2}
DEPTH = {"depth_cell": 250.0, "depth_max": 2000.0}  # mm; a sizing proxy, NOT the reviewed 3D coordinate
PRECISIONS = {
    # name: (x dtype, neighbours dtype, autocast). Encoder._patch_forward builds neighbours ONCE in the encoder-input
    # dtype; x turns fp32 after the first GMP (its nn.LayerNorm returns fp32 under autocast). Dtype probe, job 7841399:
    # bf16 encoder input -> layer 0 (bf16, bf16), layers >= 1 (fp32, bf16); fp32 input -> (fp32, fp32) everywhere.
    "fp32": (torch.float32, torch.float32, False),
    "amp_xbf16": (torch.bfloat16, torch.bfloat16, True),  # layer 0
    "amp_xfp32_nbbf16": (torch.float32, torch.bfloat16, True),  # layers >= 1 (all-layer GMP)
    "amp_xfp32": (torch.float32, torch.float32, True),  # fp32 encoder input
}
SOURCES = [
    "src/hepattn/experiments/clic/a4/gmp_weights_first.py",
    "src/hepattn/experiments/clic/a4/gmp_impl.py",
    "src/hepattn/experiments/clic/a4/bench_gmp_backends.py",
    "src/hepattn/models/encoder.py",
]


class GMPLayer(nn.Module):
    """Production GMP's structure with a pluggable depthwise core and K offsets: conv(xm, A, W[D, K]) + bias."""

    def __init__(self, dim: int, num_offsets: int, conv) -> None:
        super().__init__()
        self.dw_weight = nn.Parameter(torch.empty(dim, num_offsets))
        self.dw_bias = nn.Parameter(torch.empty(dim))
        bound = 1.0 / math.sqrt(num_offsets)  # nn.Conv2d's default init for a depthwise kernel, fan_in = K
        nn.init.uniform_(self.dw_weight, -bound, bound)
        nn.init.uniform_(self.dw_bias, -bound, bound)
        self.pointwise = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.conv = conv

    @classmethod
    def from_module(cls, gmp: GeometricMessagePassing, conv) -> GMPLayer:
        dim, num_offsets = gmp.dw.weight.shape[0], gmp.num_offsets
        layer = cls(dim, num_offsets, conv).to(device=gmp.dw.weight.device, dtype=gmp.dw.weight.dtype)
        with torch.no_grad():
            layer.dw_weight.copy_(gmp.dw.weight.view(dim, num_offsets))
            layer.dw_bias.copy_(gmp.dw.bias)
            layer.pointwise.load_state_dict(gmp.pointwise.state_dict())
            layer.norm.load_state_dict(gmp.norm.state_dict())
        return layer

    def forward(self, x: Tensor, neighbours: Tensor, in_grid: Tensor) -> Tensor:
        xm = torch.where(in_grid.unsqueeze(-1), x, 0.0)
        z = self.conv(xm, neighbours, self.dw_weight) + self.dw_bias
        return torch.where(in_grid.unsqueeze(-1), x + self.norm(self.pointwise(z)), x)


def proxy_neighbourhood_3d(
    eta: Tensor,
    phi: Tensor,
    depth: Tensor,
    valid: Tensor,
    kernel_size: int,
    eta_cell: float,
    phi_cells: int,
    eta_max: float,
    depth_cell: float,
    depth_max: float,
) -> tuple[Tensor, Tensor]:
    """gmp_neighbourhood with a third, non-periodic depth axis. Offsets in lexicographic (du, dv, dw) order, so
    offset o and K-1-o are opposite, as in 2D. A benchmark/test proxy only: the 3D coordinate is still under review.
    """
    r = kernel_size // 2
    in_grid = valid & eta.isfinite() & phi.isfinite() & depth.isfinite() & (eta.abs() <= eta_max)
    in_grid = in_grid & (depth >= 0) & (depth < depth_max)
    eta = torch.where(in_grid, eta, 0.0)
    phi = torch.where(in_grid, phi, 0.0)
    depth = torch.where(in_grid, depth, 0.0)
    u = torch.floor((eta + eta_max) / eta_cell).long()
    v = torch.remainder(torch.floor(phi * (phi_cells / (2 * math.pi))), phi_cells).long()
    w = torch.floor(depth / depth_cell).long()
    du = u.unsqueeze(-2) - u.unsqueeze(-1)
    dv = torch.remainder(v.unsqueeze(-2) - v.unsqueeze(-1) + r, phi_cells) - r
    dw = w.unsqueeze(-2) - w.unsqueeze(-1)
    pairs = in_grid.unsqueeze(-1) & in_grid.unsqueeze(-2)
    rng = range(-r, r + 1)
    offsets = [pairs & (du == a) & (dv == b) & (dw == c) for a in rng for b in rng for c in rng]
    return torch.cat(offsets, dim=-2), in_grid


def make_geometry(batch: int, slots: int, num_offsets: int, device, seed: int = 0, min_valid: int | None = None):
    """Random events padded to `slots`: clustered coordinates (several tokens per cell), some off-grid, NaN padding.

    Returns (neighbours as bool, in_grid). The dense backends' cost depends only on the shapes, not on the content;
    the content matters for correctness and is made deliberately awkward (co-located tokens, phi wrap, padding).
    """
    gen = torch.Generator().manual_seed(seed)
    lo = min_valid if min_valid is not None else max(1, slots // 8)
    counts = torch.randint(lo, slots, (batch,), generator=gen)
    valid = torch.arange(slots).unsqueeze(0) < counts.unsqueeze(1)
    centres_eta = torch.randn(batch, 12, generator=gen) * 1.8
    centres_phi = torch.rand(batch, 12, generator=gen) * 2 * math.pi - math.pi
    pick = torch.randint(0, 12, (batch, slots), generator=gen)
    eta = centres_eta.gather(1, pick) + torch.randn(batch, slots, generator=gen) * 0.15
    phi = centres_phi.gather(1, pick) + torch.randn(batch, slots, generator=gen) * 0.15
    eta[:, ::17] = 6.0  # a few real tokens beyond eta_max: off grid
    eta = torch.where(valid, eta, float("nan"))
    phi = torch.where(valid, phi, float("nan"))
    geom = GEOMETRY[num_offsets]
    # Random numbers on the CPU (reproducible across devices), the O(K B N^2) masks built on the target device
    eta, phi, valid = eta.to(device), phi.to(device), valid.to(device)
    if geom["dims"] == 2:
        return gmp_neighbourhood(eta, phi, valid, kernel_size=geom["kernel_size"], **GRID)
    depth = torch.where(torch.rand(batch, slots, generator=gen) < 0.3, 0.0, torch.rand(batch, slots, generator=gen).pow(2) * 1500.0)
    return proxy_neighbourhood_3d(eta, phi, depth.to(device), valid, kernel_size=geom["kernel_size"], **GRID, **DEPTH)


# ----------------------------------------------------------------------------------------------------------- helpers
def _sync():
    torch.cuda.synchronize()


def _err(got: Tensor, ref: Tensor) -> dict:
    d = (got.double() - ref.double()).abs()
    scale = ref.double().abs().max().item()
    return {"max_abs": d.max().item(), "max_rel_to_max": d.max().item() / max(scale, 1e-30), "finite": bool(got.isfinite().all())}


def _layer(variant: str, num_offsets: int, dim: int, device, seed: int) -> GMPLayer:
    torch.manual_seed(seed)  # identical parameters for every variant
    return GMPLayer(dim, num_offsets, VARIANTS[variant]).to(device)


def _saved_tensor_report(fn, inputs: list[Tensor], params: list[Tensor]) -> dict:
    """What autograd saves for backward, by storage. Tensors sharing storage with an input/parameter are counted apart."""
    known = {t.untyped_storage().data_ptr() for t in inputs + params}
    seen: dict[int, tuple] = {}

    def pack(t):
        s = t.untyped_storage()
        seen.setdefault(s.data_ptr(), (s.nbytes(), tuple(t.shape), str(t.dtype).replace("torch.", "")))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = fn()
    new = {p: v for p, v in seen.items() if p not in known}
    aliased = {p: v for p, v in seen.items() if p in known}
    largest = sorted(new.values(), reverse=True)[:4]
    return y, {
        "new_bytes": sum(v[0] for v in new.values()),
        "input_aliased_bytes": sum(v[0] for v in aliased.values()),
        "num_new": len(new),
        "largest_new": [{"bytes": b, "shape": list(s), "dtype": d} for b, s, d in largest],
    }


def run_config(variant, num_offsets, slots, precision, mode, args, ref_cache) -> dict:
    device = torch.device("cuda")
    x_dtype, nb_dtype, amp = PRECISIONS[precision]
    rec = {
        "kind": "result",
        "variant": variant,
        "K": num_offsets,
        "N": slots,
        "precision": precision,
        "mode": mode,
        "batch": args.batch,
        "dim": args.dim,
    }
    torch._dynamo.reset()  # noqa: SLF001
    gc.collect()
    torch.cuda.empty_cache()

    nb_bool, in_grid = make_geometry(args.batch, slots, num_offsets, device, seed=args.seed)
    neighbours = nb_bool.to(nb_dtype)  # Encoder._patch_forward: neighbours.to(encoder-input dtype), shared by all layers
    del nb_bool
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    x = torch.randn(args.batch, slots, args.dim, generator=gen).to(device=device, dtype=x_dtype).requires_grad_(True)
    layer = _layer(variant, num_offsets, args.dim, device, args.seed)
    fn_mod = layer if mode == "eager" else torch.compile(layer, dynamic=False)
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else contextlib.nullcontext

    def fwd():
        with amp_ctx():
            return fn_mod(x, neighbours, in_grid)

    def zero():
        x.grad = None
        layer.zero_grad(set_to_none=True)

    # Upstream gradient: fixed, in the output dtype (from one eager no-grad call of the same module)
    with torch.no_grad(), amp_ctx():
        out_dtype = layer(x, neighbours, in_grid).dtype
    g = torch.randn(args.batch, slots, args.dim, generator=gen).to(device=device, dtype=out_dtype)
    rec["out_dtype"] = str(out_dtype).replace("torch.", "")

    # 1. first call: compile (or eager first-call) cost, forward and backward separately
    zero()
    _sync()
    t0 = time.perf_counter()
    y = fwd()
    _sync()
    rec["first_fwd_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    y.backward(g)
    _sync()
    rec["first_bwd_s"] = time.perf_counter() - t0

    # 2. correctness vs eager production at the same precision (cached per K, N, precision)
    key = (num_offsets, slots, precision)
    if key not in ref_cache:
        ref_layer = _layer("A_production", num_offsets, args.dim, device, args.seed)
        xr = x.detach().clone().requires_grad_(True)
        with amp_ctx():
            yr = ref_layer(xr, neighbours, in_grid)
        yr.backward(g)
        ref_cache.clear()  # keep one reference on the GPU at a time
        ref_cache[key] = (yr.detach(), xr.grad.detach(), ref_layer.dw_weight.grad.detach(), ref_layer.pointwise.weight.grad.detach())
        del ref_layer, xr, yr
    yr, gxr, gwr, gpr = ref_cache[key]
    rec["err_vs_production"] = {
        "out": _err(y.detach(), yr),
        "grad_x": _err(x.grad, gxr),
        "grad_dw_weight": _err(layer.dw_weight.grad, gwr),
        "grad_pointwise_weight": _err(layer.pointwise.weight.grad, gpr),
    }
    del y
    zero()

    # 3. warmup
    for _ in range(args.warmup):
        y = fwd()
        y.backward(g)
        zero()
    del y
    _sync()

    # 4. memory: retained between forward and backward, and peak over a forward+backward
    gc.collect()
    _sync()
    m0 = torch.cuda.memory_allocated()
    y = fwd()
    _sync()
    rec["retained_bytes"] = torch.cuda.memory_allocated() - m0 - y.untyped_storage().nbytes()
    y.backward(g)
    del y
    zero()
    gc.collect()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    y = fwd()
    y.backward(g)
    _sync()
    rec["peak_above_inputs_bytes"] = torch.cuda.max_memory_allocated() - m0
    del y
    zero()
    rec["inputs_bytes"] = {"neighbours": neighbours.untyped_storage().nbytes(), "x": x.untyped_storage().nbytes()}

    # 5. eager only: what autograd saved
    if mode == "eager":
        y, rep = _saved_tensor_report(fwd, [x, neighbours, in_grid], list(layer.parameters()))
        rec["saved"] = rep
        y.backward(g)
        del y
        zero()

    # 6. steady-state timing
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(args.iters)]
    for e0, e1, e2 in ev:
        e0.record()
        y = fwd()
        e1.record()
        y.backward(g)
        e2.record()
        del y
        zero()
    _sync()
    f = [a.elapsed_time(b) for a, b, _ in ev]
    b = [bb.elapsed_time(c) for _, bb, c in ev]
    t = [a.elapsed_time(c) for a, _, c in ev]

    def stats(v):
        v = sorted(v)
        return {"median_ms": statistics.median(v), "p90_ms": v[int(0.9 * (len(v) - 1))], "min_ms": v[0]}

    rec["fwd"], rec["bwd"], rec["fwd_bwd"] = stats(f), stats(b), stats(t)
    rec["compile_overhead_s"] = rec["first_fwd_s"] + rec["first_bwd_s"] - rec["fwd_bwd"]["median_ms"] / 1000.0
    return rec


def provenance(args) -> dict:
    here = Path(__file__).resolve()
    repo = Path(subprocess.check_output(["git", "-C", str(here.parent), "rev-parse", "--show-toplevel"], text=True).strip())  # noqa: S607

    def git(*a):
        return subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()  # noqa: S607

    import triton  # noqa: PLC0415

    return {
        "kind": "meta",
        "argv": sys.argv,
        "args": vars(args),
        "git_toplevel": str(repo),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain", "--untracked-files=no").splitlines(),
        "file_sha256": {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in SOURCES},
        "module_file": str(here),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("TORCHINDUCTOR", "TORCH_", "OMP_", "HEPATTN"))},
    }


def probe_dtypes(out: Path) -> None:
    """What dtype reach each GMP layer of the production patch encoder under bf16 autocast (no training)."""
    from hepattn.models.encoder import Encoder  # noqa: PLC0415

    rows = []
    for entry_dtype in (torch.bfloat16, torch.float32):
        torch.manual_seed(0)
        enc = Encoder(
            num_layers=3,
            dim=64,
            attn_type="patch",
            patch_size=16,
            patch_global=True,
            num_register_tokens=8,
            value_residual=True,
            hybrid_norm=True,
            dense_norm_placement="legacy",
            norm="FastLayerNorm",
            attn_kwargs={"num_heads": 4},
            gmp={"kernel_size": 3, **GRID},
        ).cuda()
        seen = []
        for i, layer in enumerate(enc.layers):
            layer.gmp.register_forward_pre_hook(
                lambda _m, a, i=i, seen=seen: seen.append({"layer": i, "x": str(a[0].dtype), "neighbours": str(a[1].dtype)})
            )
        b, s = 4, 160
        x = torch.randn(b, s, 64, device="cuda", dtype=entry_dtype)
        coords = torch.stack([torch.randn(b, s) * 1.5, torch.rand(b, s) * math.tau - math.pi], -1).cuda()
        valid = torch.arange(s).unsqueeze(0).expand(b, s).cuda() < 100
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc(x, kv_mask=valid, x_coords=coords)
        rows.append({"encoder_input_dtype": str(entry_dtype), "gmp_inputs": seen})
    out.write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))


def summarize(files: list[Path]) -> str:
    """Markdown tables from result files: one table per (precision, mode, N), rows K x variant."""
    rows, other = [], []
    for f in files:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            (rows if r.get("kind") == "result" else other).append(r)
    out = []
    mb = 2.0**20
    order = list(VARIANTS)
    for precision in PRECISIONS:
        for mode in ("eager", "compile"):
            for n in sorted({r["N"] for r in rows}):
                sel = [r for r in rows if (r["precision"], r["mode"], r["N"]) == (precision, mode, n)]
                if not sel:
                    continue
                out.extend([
                    f"\n### {precision} / {mode} / N={n} (B={sel[0]['batch']}, D={sel[0]['dim']})\n",
                    "| K | variant | fwd ms | bwd ms | fwd+bwd ms (p90) | retained MB | saved new MB | peak MB "
                    "| first fwd+bwd s | max rel err out / gx / gw |",
                    "|---|---|---|---|---|---|---|---|---|---|",
                ])
                for r in sorted(sel, key=lambda r: (r["K"], order.index(r["variant"]))):
                    e = r["err_vs_production"]
                    saved = f"{r['saved']['new_bytes'] / mb:.0f}" if "saved" in r else "-"
                    out.append(
                        f"| {r['K']} | {r['variant']} | {r['fwd']['median_ms']:.2f} | {r['bwd']['median_ms']:.2f} | "
                        f"{r['fwd_bwd']['median_ms']:.2f} ({r['fwd_bwd']['p90_ms']:.2f}) | {r['retained_bytes'] / mb:.0f} | {saved} | "
                        f"{r['peak_above_inputs_bytes'] / mb:.0f} | {r['first_fwd_s'] + r['first_bwd_s']:.1f} | "
                        f"{e['out']['max_rel_to_max']:.1e} / {e['grad_x']['max_rel_to_max']:.1e} / {e['grad_dw_weight']['max_rel_to_max']:.1e} |"
                    )
    meta = [r for r in other if r.get("kind") == "meta"]
    bad = [r for r in other if r.get("kind") in {"error", "skipped_deadline"}]
    head = [f"# GMP backend micro-benchmark: {len(rows)} results, {len(bad)} errors/skips"]
    if meta:
        m = meta[0]
        head.append(
            f"commit {m['git_commit']} ({m['git_branch']}), dirty={bool(m['git_status'])}, job {m['pbs_jobid']}, "
            f"{m['gpu']}, torch {m['torch']} cuda {m['cuda']} triton {m['triton']}"
        )
    head += [f"- {r['kind']}: {r.get('config') or (r['variant'], r['K'], r['N'], r['precision'], r['mode'])} {r.get('error', '')}" for r in bad]
    return "\n".join(head + out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--ks", default="9,25,27")
    p.add_argument("--ns", default="160,320")
    p.add_argument("--precisions", default=",".join(PRECISIONS))
    p.add_argument("--modes", default="eager,compile")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deadline", type=float, default=float("inf"), help="unix time after which no new config starts")
    p.add_argument("--out", type=Path)
    p.add_argument("--probe-dtypes", action="store_true")
    p.add_argument("--summarize", type=Path, nargs="+", help="print markdown tables from result files and exit")
    args = p.parse_args()
    if args.summarize:
        print(summarize(args.summarize))
        return
    if args.out is None:
        p.error("--out is required")
    if not torch.cuda.is_available():
        sys.exit("bench_gmp_backends needs a GPU: submit it with submit_gmp_backends_bench.sh")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.probe_dtypes:
        probe_dtypes(args.out)
        return

    done = set()
    if args.out.exists():  # resume: skip configurations already recorded
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "result":
                done.add((r["variant"], r["K"], r["N"], r["precision"], r["mode"]))
    ref_cache: dict = {}
    with args.out.open("a") as fh:
        fh.write(json.dumps(provenance(args), default=str) + "\n")
        fh.flush()
        for mode in args.modes.split(","):
            for precision in args.precisions.split(","):
                for slots in map(int, args.ns.split(",")):
                    for num_offsets in map(int, args.ks.split(",")):
                        for variant in args.variants.split(","):
                            cfg = (variant, num_offsets, slots, precision, mode)
                            if cfg in done:
                                continue
                            if time.time() > args.deadline:
                                fh.write(json.dumps({"kind": "skipped_deadline", "config": cfg}) + "\n")
                                continue
                            t0 = time.time()
                            try:
                                rec = run_config(variant, num_offsets, slots, precision, mode, args, ref_cache)
                            except Exception as e:  # noqa: BLE001 -- recorded in the results, never silently dropped
                                rec = {
                                    "kind": "error",
                                    "variant": variant,
                                    "K": num_offsets,
                                    "N": slots,
                                    "precision": precision,
                                    "mode": mode,
                                    "error": f"{type(e).__name__}: {e}"[:2000],
                                }
                                ref_cache.clear()
                                torch.cuda.empty_cache()
                            rec["wall_s"] = time.time() - t0
                            fh.write(json.dumps(rec) + "\n")
                            fh.flush()
                            brief = {k: rec.get(k) for k in ("variant", "K", "N", "precision", "mode", "kind")}
                            if rec["kind"] == "result":
                                brief |= {
                                    "fwd_bwd_ms": round(rec["fwd_bwd"]["median_ms"], 3),
                                    "retained_MB": round(rec["retained_bytes"] / 2**20, 1),
                                    "peak_MB": round(rec["peak_above_inputs_bytes"] / 2**20, 1),
                                }
                            print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
