"""Combine bench3 results (one JSON per arm, written by independent jobs) into comparison tables.

  python -m hepattn.experiments.clic.bench3.summarize <results_dir_or_json> [...] [--baseline TAG] [--markdown]
  python -m hepattn.experiments.clic.bench3.summarize --round0 <round0_dir>

Training results: the most recent file per tag (by mtime) is used. Speedups are ratios of time per epoch (> 1 = faster
than the baseline). The primary baseline is the Linformer arm; MaskFormer is shown as a second reference when present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

LINFORMER = "bench3_linformer_encoder"
MASKFORMER = "bench3_maskformer"
ORDER = [
    "bench3_linformer_encoder",
    "bench3_phat16",
    "bench3_phat_gmp_einsumopt_early",
    "bench3_maskformer",
    "bench3_phat16_gmp_all_production",
    "bench3_phat16_gmp_all_einsumopt",
    "bench3_phat16_gmp_early_production",
]
SHORT = {
    "bench3_maskformer": "MaskFormer (full attn)",
    "bench3_linformer_encoder": "Linformer (encoder)",
    "bench3_phat16": "PHAT16",
    "bench3_phat_gmp_einsumopt_early": "PHAT16+GMP einsumopt/early",
    "bench3_phat16_gmp_all_production": "PHAT16+GMP production/all6",
    "bench3_phat16_gmp_all_einsumopt": "PHAT16+GMP einsumopt/all6",
    "bench3_phat16_gmp_early_production": "PHAT16+GMP production/early",
}
# Round 3: (placement, impl) -> tag
GRID = {
    ("all 6 layers", "production"): "bench3_phat16_gmp_all_production",
    ("all 6 layers", "einsumopt"): "bench3_phat16_gmp_all_einsumopt",
    ("layer 0 only", "production"): "bench3_phat16_gmp_early_production",
    ("layer 0 only", "einsumopt"): "bench3_phat_gmp_einsumopt_early",
}
MUST_MATCH = ("world_size", "micro_batch", "accumulate", "global_batch", "precision", "num_workers", "device")
CPU_MUST_MATCH = ("torch_num_threads", "matcher_n_jobs", "affinity_cpus")


def load(paths: list[str]) -> dict[str, dict]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    latest: dict[str, tuple[float, dict]] = {}
    for f in files:
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {f}: {exc}", file=sys.stderr)
            continue
        r["_file"] = str(f)
        tag, mtime = r.get("tag", f.stem), f.stat().st_mtime
        if tag not in latest or mtime > latest[tag][0]:
            latest[tag] = (mtime, r)
    return {k: v[1] for k, v in latest.items()}


def num(x) -> float:
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) and x is not None else math.nan


def fmt(x, spec: str) -> str:
    v = num(x)
    return format(v, spec) if math.isfinite(v) else "--"


def ratio(base: dict | None, r: dict, key: str) -> str:
    if not base:
        return "--"
    v = num(base.get(key)) / num(r.get(key))
    return fmt(v, ".2f") + "x" if math.isfinite(v) else "--"


def render(rows: list[list[str]], markdown: bool) -> str:
    if markdown:
        out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join("---" for _ in rows[0]) + "|"]
        return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True))) for r in rows]
    out.insert(1, "-" * len(out[0]))
    return "\n".join(out)


def ordered(results: dict[str, dict]) -> list[str]:
    return [t for t in ORDER if t in results] + sorted(t for t in results if t not in ORDER)


def table_epoch(results: dict[str, dict], baseline: str, markdown: bool) -> str:
    base, mf = results.get(baseline), results.get(MASKFORMER)
    cols = [
        ("model", lambda r: SHORT.get(r["tag"], r["tag"])),
        ("status", lambda r: r.get("status", "?")),
        ("params", lambda r: fmt(num(r.get("trainable_params")) / 1e6, ".2f") + "M"),
        ("s/batch", lambda r: fmt(r.get("s_per_batch"), ".4f")),
        ("p90", lambda r: fmt(r.get("batch_p90_s"), ".4f")),
        ("opt steps/s", lambda r: fmt(r.get("optimizer_steps_per_s"), ".3f")),
        ("batches/epoch", lambda r: str(r.get("train_batches_per_epoch_per_rank", "--"))),
        ("train min/ep", lambda r: fmt(num(r.get("epoch_train_s")) / 60, ".2f")),
        ("val s/ep", lambda r: fmt(r.get("epoch_val_s_steady"), ".1f")),
        ("total min/ep", lambda r: fmt(num(r.get("epoch_total_s")) / 60, ".2f")),
        (f"speedup vs {SHORT.get(baseline, baseline).split()[0]}", lambda r: ratio(base, r, "epoch_train_s")),
        ("vs MaskFormer", lambda r: ratio(mf, r, "epoch_train_s")),
        ("GPU peak GiB", lambda r: fmt(r.get("gpu_peak_reserved_gib"), ".2f")),
        ("host PSS GiB", lambda r: fmt(r.get("host_job_pss_gib_total"), ".1f")),
        ("DL wait %", lambda r: fmt(100 * num(r.get("dataloader_wait_frac")), ".1f")),
    ]
    rows = [[c for c, _ in cols]] + [[fn(results[t]) for _, fn in cols] for t in ordered(results)]
    return render(rows, markdown)


def table_overheads(results: dict[str, dict], markdown: bool) -> str:
    cols = [
        ("model", lambda r: SHORT.get(r["tag"], r["tag"])),
        ("startup s", lambda r: fmt(r.get("startup_s"), ".1f")),
        ("init train s", lambda r: fmt(r.get("dataset_init_train_s"), ".1f")),
        ("init val s", lambda r: fmt(r.get("dataset_init_val_s"), ".1f")),
        ("1st batch (compile) s", lambda r: fmt(r.get("first_batch_s_incl_compile"), ".1f")),
        ("rest of warmup s", lambda r: fmt(r.get("warmup_rest_s"), ".1f")),
        ("1st val pass s", lambda r: fmt(r.get("val_wall_s_first"), ".1f")),
        ("val", lambda r: str(r.get("val_status", "--"))[:28]),
        ("CPU util %", lambda r: fmt((r.get("cpu") or {}).get("node_cpu_util_pct_median"), ".0f")),
        ("oversub.", lambda r: str((r.get("cpu") or {}).get("oversubscribed_during_matching", "--"))),
    ]
    rows = [[c for c, _ in cols]] + [[fn(results[t]) for _, fn in cols] for t in ordered(results)]
    return render(rows, markdown)


def table_breakdown(results: dict[str, dict], markdown: bool) -> str | None:
    tags = [t for t in ordered(results) if results[t].get("breakdown")]
    if not tags:
        return None

    def b(r, k):
        return (r["breakdown"].get("rank0") or {}).get(k)

    cols = [
        ("model", lambda r: SHORT.get(r["tag"], r["tag"])),
        ("batch ms", lambda r: fmt(b(r, "batch_ms"), ".0f")),
        ("fwd+loss host", lambda r: fmt(b(r, "fwd_loss_host_ms"), ".0f")),
        ("fwd+loss GPU", lambda r: fmt(b(r, "fwd_loss_gpu_ms"), ".0f")),
        ("matcher CPU", lambda r: fmt(b(r, "matcher_cpu_ms"), ".0f")),
        ("matcher GPU-wait", lambda r: fmt(b(r, "matcher_gpu_wait_ms"), ".0f")),
        ("matcher share", lambda r: fmt(100 * num(r["breakdown"].get("matcher_share_of_batch")), ".0f") + "%"),
        ("bwd host", lambda r: fmt(b(r, "bwd_host_ms"), ".0f")),
        ("bwd GPU", lambda r: fmt(b(r, "bwd_gpu_ms"), ".0f")),
        ("clip+opt host", lambda r: fmt(b(r, "clip_opt_host_ms"), ".0f")),
        ("syncs/batch", lambda r: fmt(b(r, "syncs"), ".0f")),
        ("GPU busy", lambda r: fmt(100 * num((r["breakdown"].get("profiler") or {}).get("gpu_busy_frac")), ".0f") + "%"),
    ]
    rows = [[c for c, _ in cols]] + [[fn(results[t]) for _, fn in cols] for t in tags]
    return render(rows, markdown)


def table_grid(results: dict[str, dict], markdown: bool) -> str | None:
    if not any(t in results for t in GRID.values()):
        return None
    rows = [["placement \\ impl", "production s/batch", "einsumopt s/batch", "impl effect (prod/opt)"]]
    for place in ("all 6 layers", "layer 0 only"):
        p, o = results.get(GRID[place, "production"]), results.get(GRID[place, "einsumopt"])
        eff = num(p.get("s_per_batch")) / num(o.get("s_per_batch")) if p and o else math.nan
        rows.append([place, fmt(p.get("s_per_batch") if p else None, ".4f"), fmt(o.get("s_per_batch") if o else None, ".4f"), fmt(eff, ".3f") + "x"])
    col = ["placement effect (all/early)"]
    for impl in ("production", "einsumopt"):
        a, e = results.get(GRID["all 6 layers", impl]), results.get(GRID["layer 0 only", impl])
        col.append(fmt(num(a.get("s_per_batch")) / num(e.get("s_per_batch")), ".3f") + "x" if a and e else "--")
    rows.append([*col, ""])
    return render(rows, markdown) + "\n(compare within a row or a column only; the diagonal changes two factors)"


def checks(results: dict[str, dict], baseline: str) -> list[str]:
    notes = []
    for key in MUST_MATCH:
        vals = {str(r.get(key)) for r in results.values()}
        if len(vals) > 1:
            notes.append(f"!! {key} differs across arms ({sorted(vals)}): NOT a like-for-like comparison")
    for key in CPU_MUST_MATCH:
        vals = {str((r.get("cpu") or {}).get(key)) for r in results.values()}
        if len(vals) > 1:
            notes.append(f"!! CPU setting {key} differs across arms ({sorted(vals)})")
    if baseline not in results:
        notes.append(f"!! no {baseline} result yet: baseline speedups unavailable")
    commits = {(r.get("provenance") or {}).get("git_commit") for r in results.values()}
    if len(commits) > 1:
        notes.append(f"!! arms ran from different git commits: {sorted(map(str, commits))}")
    hashes = [(r.get("provenance") or {}).get("file_sha256") or {} for r in results.values()]
    shared = set.intersection(*(set(h) for h in hashes)) if hashes else set()
    drift = sorted(f for f in shared if len({h[f] for h in hashes}) > 1)
    if drift:
        notes.append(f"!! bench files changed between arms: {drift}")
    for t, r in results.items():
        src = str(r.get("batches_per_epoch_source", ""))
        if "extrapolated" in src or "WARNING" in src:
            notes.append(f"note {t}: batches/epoch {src}; dataset init and host RAM are NOT full-scale")
        if r.get("status") not in {"complete", "train_done"}:
            notes.append(f"note {t}: status {r.get('status')} (measured {r.get('measured_batches')} batches, reliable={r.get('window_reliable')})")
        if num(r.get("dataloader_wait_frac")) > 0.10:
            notes.append(f"note {t}: dataloader wait {100 * num(r['dataloader_wait_frac']):.0f}% -> partly host-bound")
        if (r.get("cpu") or {}).get("oversubscribed_during_matching"):
            c = r["cpu"]
            notes.append(f"note {t}: CPU oversubscribed during matching: {c.get('planned_busy_threads_during_matching')} busy threads planned "
                         f"on {c.get('physical_cores')} cores; median node util {fmt(c.get('node_cpu_util_pct_median'), '.0f')}%")  # fmt: skip
    return notes


def provenance_lines(results: dict[str, dict]) -> list[str]:
    out = []
    for t in ordered(results):
        r, pv = results[t], results[t].get("provenance") or {}
        out.append(f"{t}: job {r.get('pbs_jobid')} {r.get('device')} x{r.get('world_size')} ({r.get('_file')})")
        out.append(f"    hepattn  {pv.get('hepattn_file')}")
        out.append(
            f"    git      {pv.get('git_branch')} @ {pv.get('git_commit')} dirty={pv.get('git_dirty')} diff_sha256={pv.get('git_diff_sha256')}"
        )
        out.append(
            f"    encoder  {pv.get('encoder_class')} attn={pv.get('encoder_attn_type')} gmp_layers={pv.get('encoder_gmp_layers')} impl={pv.get('encoder_gmp_impl')}"
        )
    return out


def round0(dirs: list[str], markdown: bool) -> str:
    smoke, enc = {}, {}
    for d in map(Path, dirs):
        for f in sorted(d.glob("smoke_*.json")):
            r = json.loads(f.read_text())
            smoke[r["arm"]] = r
        for f in sorted(d.glob("encoder_*.json")):
            r = json.loads(f.read_text())
            enc[r["arm"]] = r
    arms = sorted(set(smoke) | set(enc), key=lambda a: ORDER.index(f"bench3_{a}") if f"bench3_{a}" in ORDER else 99)
    rows = [["arm", "smoke", "batch", "eager peak GiB", "compiled peak GiB", "enc eager fwd+bwd ms", "enc compiled fwd+bwd ms",
             "GMP geom ms", "GMP module ms (compiled)", "real tokens"]]  # fmt: skip
    for a in arms:
        s, e = smoke.get(a, {}), enc.get(a, {})
        rows.append([
            a, s.get("status", "--"), str(s.get("batch_size", e.get("batch_size", "--"))),
            fmt(s.get("eager_peak_reserved_gib"), ".2f"), fmt(s.get("compiled_peak_reserved_gib"), ".2f"),
            fmt(e.get("encoder_eager_fwdbwd_ms"), ".1f"), fmt(e.get("encoder_compiled_fwdbwd_ms"), ".1f"),
            fmt(e.get("gmp_geometry_compiled_ms"), ".2f"), fmt(e.get("gmp_module_compiled_fwdbwd_ms"), ".2f"),
            fmt(100 * num(e.get("real_tokens_frac")), ".0f") + "%",
        ])  # fmt: skip
    oom = [a for a, s in smoke.items() if s.get("status") == "OOM"]
    tail = f"\n!! OOM in {oom}: change the COMMON micro-batch for ALL arms (keep global batch 1024 via accumulation)" if oom else ""
    return render(rows, markdown) + tail


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("--baseline", default=LINFORMER)
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--round0", action="store_true", help="summarize Round-0 smoke/memory/encoder-only results")
    a = p.parse_args(argv)
    if a.round0:
        print(round0(a.paths, a.markdown))
        return 0
    results = load(a.paths)
    if not results:
        print("no results found")
        return 1
    print("END-TO-END TRAINING (steady state; epoch = exact batches/epoch x s/batch)")
    print(table_epoch(results, a.baseline, a.markdown))
    print("\nNOT IN THROUGHPUT (timed separately)")
    print(table_overheads(results, a.markdown))
    bd = table_breakdown(results, a.markdown)
    if bd:
        print("\nSTEP BREAKDOWN (instrumented batches after the window; rank 0 medians, ms)")
        print(bd)
    grid = table_grid(results, a.markdown)
    if grid:
        print("\nROUND 3: GMP implementation x placement (s/batch)")
        print(grid)
    print()
    for line in provenance_lines(results):
        print(line)
    for n in checks(results, a.baseline):
        print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
