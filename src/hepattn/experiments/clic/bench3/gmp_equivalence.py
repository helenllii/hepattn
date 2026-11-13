"""Round-3 precondition: the production and EinsumOpt GMP forwards compute the same encoder, at a FIXED placement.

One encoder is built from the placement's production arm (`phat16_gmp_all_production` or
`phat16_gmp_early_production`). A deep copy, with its GMP modules retyped to `GMPEinsumOpt` (no new state, so the weights
and the state_dict are identical), is the EinsumOpt encoder. Both get the SAME captured encoder inputs (one real batch,
tokens after the input nets) and therefore the same GMP geometry, in each precision:

  fp32 (TF32 off)  impl vs impl: forward rel. diff <= --tol-fp32 (1e-5), every gradient (input + params) <= 10x that.
  bf16 autocast    each impl vs an fp64 reference: neither is presumed exact. Pass if EinsumOpt's error is within
                   --bf16-factor (2x) of production's error (+1e-6), for the forward and for every gradient.

fp64 reference: the production encoder in float64 (falls back to fp32-highest, recorded, if an op lacks fp64).

  python -m hepattn.experiments.clic.bench3.gmp_equivalence --placement early --out eq_early.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

from hepattn.experiments.clic.bench3.encoder_bench import clone_inputs
from hepattn.experiments.clic.bench3.smoke import load_one_batch, to_device

PLACEMENT_ARM = {"all": "phat16_gmp_all_production", "early": "phat16_gmp_early_production"}


def capture_encoder_inputs(model, inputs) -> tuple[tuple, dict]:
    cap: dict = {}

    def hook(module, args, kwargs):
        cap.setdefault("enc", (args, kwargs))

    h = model.encoder.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(to_device(inputs, "cuda"))
    finally:
        h.remove()
    return cap["enc"]


def run_once(enc, args, kwargs, dtype, autocast: bool, grad_seed):
    """Forward + backward; returns (output fp64, input grad fp64, {param: grad fp64})."""
    # Only the tokens change precision. The sort value and the GMP coordinates keep their captured fp32 values, so
    # every precision sees the SAME token order and the SAME (eta, phi) cells: casting coordinates could move a token
    # across a cell boundary in floor((eta + eta_max) / eta_cell) and silently change the geometry.
    a, k = clone_inputs(args, True), clone_inputs(kwargs, False)
    if a:
        a = (a[0].detach().to(dtype), *a[1:])
    else:
        k["x"] = k["x"].detach().to(dtype)
    x = a[0] if a else k["x"]
    x.requires_grad_(True)
    enc.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = enc(*a, **k)
    (out.to(grad_seed.dtype) * grad_seed).sum().backward()
    grads = {n: p.grad.detach().double() for n, p in enc.named_parameters() if p.grad is not None}
    return out.detach().double(), x.grad.detach().double(), grads


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b||."""
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def compare(r1, r2) -> dict:
    o1, x1, g1 = r1
    o2, x2, g2 = r2
    param_rel = {n: rel(g1[n], g2[n]) for n in g2 if n in g1}
    worst = max(param_rel, key=param_rel.get) if param_rel else None
    return {
        "fwd_max_abs": float((o1 - o2).abs().max()),
        "fwd_rel": rel(o1, o2),
        "input_grad_rel": rel(x1, x2),
        "param_grad_rel_max": param_rel[worst] if worst else None,
        "param_grad_rel_worst": worst,
        "params_missing_grad": sorted(set(g2) ^ set(g1)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--placement", required=True, choices=sorted(PLACEMENT_ARM))
    p.add_argument("--data-dir", default="/eagle/hgcal-maskformer-fpga/clic_data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol-fp32", type=float, default=1e-5)
    p.add_argument("--bf16-factor", type=float, default=2.0)
    p.add_argument("--out")
    a = p.parse_args(argv)
    if not torch.cuda.is_available():
        print("gmp_equivalence needs CUDA", file=sys.stderr)
        return 2
    from hepattn.experiments.clic.a4.gmp_impl import GMP_IMPLS  # noqa: PLC0415

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    res: dict = {"placement": a.placement, "arm": PLACEMENT_ARM[a.placement], "batch_size": a.batch_size, "checks": {}}
    with tempfile.TemporaryDirectory(prefix="bench3_eq_") as workdir:
        cli, inputs, _ = load_one_batch(PLACEMENT_ARM[a.placement], a.data_dir, a.batch_size, True, workdir, [])
        model = cli.model.model.to("cuda").float().eval()  # eval: no dropout/drop-path randomness between runs
        args, kwargs = capture_encoder_inputs(model, inputs)
        enc_prod = model.encoder
        enc_opt = copy.deepcopy(enc_prod)
        for layer in enc_opt.layers:
            if getattr(layer, "gmp", None) is not None:
                layer.gmp.__class__ = GMP_IMPLS["einsumopt"]
        assert enc_prod.state_dict().keys() == enc_opt.state_dict().keys()
        res["gmp_layers"] = [i for i, layer in enumerate(enc_opt.layers) if getattr(layer, "gmp", None) is not None]
        res["impls"] = [type(enc_prod.layers[res["gmp_layers"][0]].gmp).__qualname__, type(enc_opt.layers[res["gmp_layers"][0]].gmp).__qualname__]
        x0 = args[0] if args else kwargs["x"]
        seed = torch.randn(x0.shape, device="cuda", dtype=torch.float64)

        # fp32, TF32 off: implementation vs implementation
        p32 = run_once(enc_prod, args, kwargs, torch.float32, False, seed.float())
        o32 = run_once(enc_opt, args, kwargs, torch.float32, False, seed.float())
        c32 = compare(o32, p32)
        c32["pass"] = (
            c32["fwd_rel"] <= a.tol_fp32
            and c32["input_grad_rel"] <= 10 * a.tol_fp32
            and (c32["param_grad_rel_max"] or 0) <= 10 * a.tol_fp32
            and not c32["params_missing_grad"]
        )
        res["checks"]["fp32_opt_vs_prod"] = c32

        # fp64 reference
        try:
            enc_ref = copy.deepcopy(enc_prod).double()
            ref = run_once(enc_ref, args, kwargs, torch.float64, False, seed)
            res["reference"] = "fp64 production"
        except Exception as exc:  # noqa: BLE001
            torch.set_float32_matmul_precision("highest")
            ref = p32
            res["reference"] = f"fp32 production, TF32 off (fp64 failed: {type(exc).__name__}: {str(exc)[:200]})"

        # bf16 autocast: each implementation against the reference
        pb = run_once(enc_prod, args, kwargs, torch.float32, True, seed.float())
        ob = run_once(enc_opt, args, kwargs, torch.float32, True, seed.float())
        ep, eo = compare(pb, ref), compare(ob, ref)
        f = a.bf16_factor

        def within(k: str) -> bool:
            return eo[k] is None or ep[k] is None or eo[k] <= f * ep[k] + 1e-6

        res["checks"]["bf16_prod_vs_ref"] = ep
        res["checks"]["bf16_opt_vs_ref"] = eo
        res["checks"]["bf16_pass"] = all(within(k) for k in ("fwd_rel", "input_grad_rel", "param_grad_rel_max"))
        res["checks"]["fp32_prod_vs_ref"] = compare(p32, ref)
    res["pass"] = bool(res["checks"]["fp32_opt_vs_prod"]["pass"] and res["checks"]["bf16_pass"])
    print(json.dumps(res, indent=1, default=str), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, default=str, sort_keys=True) + "\n")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
