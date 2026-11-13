# Weights-first GMP (Option B): correctness and micro-benchmark report

2026-11-13 · branch `exp/gmp-ablation` · implementation commit `66ba4a6` · Polaris debug jobs 7841399 (first pass, found bugs)
and **7841435 (all numbers below)** · 1× A100-SXM4-40GB per process · torch 2.9.1+cu128, triton 3.5.1.
Results: `/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env/runs/gmp_ablation/microbench_weights_first/7841435/`
(`bench_*.jsonl`, `pytest.log`, `dtype_probe.json`, `summary.md`). Every result file records commit `66ba4a6` and the sha256 of
the four source files, all equal to the committed versions; the only uncommitted change in the tree was the unrelated
`submit_bench3_polaris.sh` defaults. All 240 benchmark rows are in `REPORT_gmp_weights_first_tables.md`.

## Verdict

**Use weights-first as the GMP backend for 3D and all-layer GMP** (`gmp_impl: weightsfirst`, the fused strategy). Under
`torch.compile` in the production precision regimes, one GMP layer is **1.4–2.0× faster** than production, keeps **3.6–14.5× less
memory for backward** (more at larger K), and has **1.5–2.6× lower peak memory**. All of this was measured. The advantage holds, and in memory grows,
from K = 9 to K = 27. Its accuracy is production's: equal to fp64 round-off in fp64, ≤ 8e-7 in fp32, and no worse than
production in bf16.

Do **not** use EinsumOpt (A′) for K = 27: compiled, its backward is 3–4× *slower* than production (measured, all compiled
regimes). The custom autograd Function earns its keep only in eager mode. Compiled, plain autograd ("B0") does the same
thing. The per-offset "loop" strategy is not worth it under compile.

Scale (estimated): for 6-layer GMP at B = 256 per GPU, the saving is ~11 ms (K = 9) to ~19 ms (K = 27) per training step,
against a measured 0.34 s step (Round 1). That is 3–6% of the step, plus 1–6 GiB less activation memory. GMP is not the
training bottleneck (the matcher is ~41% of the step), so this is a worthwhile clean-up rather than a large speed-up.

## What was implemented (`a4/gmp_weights_first.py`)

Production computes `y_i = Σ_o w_o ⊙ (Σ_j A[o,i,j] x_j)` as `z = bmm(A, x)` → `(B, K·N, D)`, then contracts z with the weights.
Option B computes `y_i = Σ_o Σ_j A[o,i,j] (w_o ⊙ x_j)`:

* `_WeightsFirstConv` is a custom `autograd.Function`. It saves only the masked input `(B,N,D)`, the `(D,K)` weights, and the
  neighbours tensor as passed in (the encoder already holds that one for all layers). Backward computes
  `g_xw = bmm(A, g)` once and takes both the input gradient and the weight gradient from it, so no forward recomputation is
  needed. Production's order cannot do this: its weight gradient needs `z` itself.
* **fused** (default): one bmm with contraction K·N. It uses the neighbourhood symmetry `A[o,i,j] = A[K−1−o,j,i]`, so `A^T` is
  a free stride view (the weights are taken in reversed offset order). This symmetry is a property of production's grid,
  including the periodic φ axis because `phi_cells ≥ kernel_size`. It is tested for k = 1/3/5, for `phi_cells = k`, and for
  the 3×3×3 proxy.
* **loop**: one `(B,N,N)` bmm per offset, accumulated in fp32. It makes no K-expanded tensor at all and needs no symmetry.
* Compute dtype is the autocast dtype when autocast is on (as production's bmm), else the input dtype. Accumulation widens
  16-bit compute to fp32 and never narrows fp64.
* Backends `GMPWeightsFirst` / `GMPWeightsFirstLoop` are stateless subclasses of `GeometricMessagePassing` that override only
  `forward`. They are registered in `GMP_IMPLS` as `weightsfirst` / `weightsfirst_loop`, so `PlacementEncoder` retypes to them.
  Parameters, `state_dict` keys, initialization, neighbourhood construction, masking and the per-patch-block GMP architecture
  are production's. Each of the 6 blocks keeps its own GMP; nothing is shared. `models/encoder.py` is unchanged.

## Correctness (job 7841435: 224 passed, CPU + CUDA)

The reference is production GMP in float64 with identical parameters, inputs, neighbourhoods and upstream gradients. The table
gives the max relative error (max |Δ| / max |ref|) over K ∈ {9, 25}, N ∈ {160, 320}, CPU and CUDA, for the output, the input
gradient, and all 6 GMP parameter gradients.

| backend | float64 (tol 1e-12) | float32 (tol 2e-5) | bf16 autocast: worst err / production's worst |
|---|---|---|---|
| production | 0 | 1.3e-7 / 1.4e-7 / 8.3e-7 | 3.1e-2 – 3.2e-2 (itself) |
| einsumopt (A′) | ≤ 7.7e-16 | ≤ 5.8e-7 | 2.5e-2 / 3.2e-2 |
| weightsfirst (fused) | ≤ 7.7e-16 | ≤ 7.4e-7 | 2.8e-2 / 3.2e-2 |
| weightsfirst_loop | ≤ 8.5e-16 | ≤ 7.9e-7 | 3.0e-2 / 3.2e-2 |

In bf16 the criterion is: error ≤ 2 × production's error + 2e-3 per tensor, and ≤ max(3e-2, production's error). Every
tensor passes; the worst per-tensor ratio to production is 1.82. Three bf16 regimes were tested: x/neighbours bf16/bf16,
fp32/bf16 and fp32/fp32. Also covered and passing:
* NaN/inf in padded and off-grid slots pass through unchanged, and real tokens are bit-identical to a clean run;
* empty events, single tokens, isolated tokens, and 8 co-located tokens (checked against a hand-computed sum);
* the K = 27 core in fp64;
* the full 6-layer encoder with GMP in every block, fp64 forward and all gradients;
* `torch.compile(fullgraph=True)` with no graph break, compiled equal to eager.

## Saved tensors: what autograd actually keeps (measured with `saved_tensors_hooks`)

| | production | B0: plain-autograd weights-first | B: custom Function |
|---|---|---|---|
| K-expanded `(B,K·N,D)` | **yes** (z) | **no** | no |
| bf16 copy of the neighbours, per layer (autocast + fp32 neighbours) | **yes** | **yes** | no |
| largest new saved tensor | K·N·D or K·N² | N·D, or K·N² under autocast | N·D |

The first guess was wrong: reordering alone already stops the K-expanded tensor from being saved. bmm saves an operand only
when its partner needs a gradient, and A needs none. The custom Function additionally avoids autocast's per-layer copy of an
fp32 neighbours tensor, but only in eager mode. Under compile, the partitioner chooses what to save, and with fp32 neighbours
it keeps the bf16 cast for the custom Function too: 1,471 MB retained at K = 27, N = 320.

## Which precision regime production runs in (dtype probe, job 7841435)

`Encoder._patch_forward` builds the neighbours once, in the encoder-input dtype. The GMP `nn.LayerNorm` returns fp32 under
autocast, so the residual stream is fp32 after the first GMP. With a bf16 encoder input (expected, since InputNet is Linear
layers under autocast; not verified end-to-end): layer 0 runs with **x bf16 / neighbours bf16** (`amp_xbf16`), and layers ≥ 1
run with **x fp32 / neighbours bf16** (`amp_xfp32_nbbf16`). `amp_xfp32` (fp32 neighbours) corresponds only to an fp32 encoder
input.

## Benchmark results (one GMP layer, B = 256, D = 256; median of 40 iterations after 10 warmup; CUDA events)

The layer is `GMPLayer`: masking, depthwise conv, bias, pointwise Linear, LayerNorm and residual. For K = 9/25 it is
bit-identical to production's module (tested). K = 27 uses a 3×3×3 proxy neighbourhood; the 3D coordinate is still under
review. Geometry is synthetic. The dense backends' cost depends on shapes only, not on content. "Retained" is memory alive
between forward and backward minus the output. "Peak" is above the inputs over one forward+backward. The neighbours tensor is
an input and is excluded from both.

**Compiled, production regime for layers ≥ 1 (`amp_xfp32_nbbf16`)**, fwd+bwd ms (speed-up vs A) · retained MB · peak MB:

| K, N | A production | A′ einsumopt | B fused | B loop |
|---|---|---|---|---|
| 9, 160 | 3.74 · 220 · 544 | 1.86 (2.01×) · 220 · 261 | **1.89 (1.98×) · 60 · 260** | 2.36 (1.59×) · 60 · 261 |
| 9, 320 | 6.98 · 441 · 800 | 4.01 (1.74×) · 441 · 522 | **4.03 (1.73×) · 121 · 520** | 4.79 (1.46×) · 121 · 525 |
| 25, 160 | 6.88 · 540 · 1504 | 3.92 (1.76×) · 540 · 581 | **3.86 (1.78×) · 60 · 580** | 5.66 (1.21×) · 60 · 582 |
| 25, 320 | 12.09 · 1081 · 2080 | 8.34 (1.45×) · 1081 · 1201 | **8.16 (1.48×) · 121 · 1160** | 11.76 (1.03×) · 121 · 1161 |
| 27, 160 | 7.18 · 580 · 1624 | 23.80 (**0.30×**) · 580 · 621 | **4.06 (1.77×) · 60 · 620** | 6.22 (1.15×) · 60 · 622 |
| 27, 320 | 12.71 · 1161 · 2240 | 48.29 (**0.26×**) · 1161 · 1242 | **8.74 (1.45×) · 121 · 1240** | 12.87 (0.99×) · 121 · 1241 |

Layer 0's regime (`amp_xbf16`, compiled) gives the same picture. B fused vs A: 1.40–1.96× faster, retained 40–81 vs 220–1161
MB, peak 241–1200 vs 544–2240 MB. Compiled fp32: B fused is 1.19–1.31× faster, retained 80–161 vs 440–2321 MB, peak 441–2320
vs 760–4400 MB.

**Eager** (for reference; training is compiled):
* B fused is 1.9–3.5× faster than A, with 3.7–16× less retained memory.
* Its peak is **at or above** A's, because the backward builds transient K-expanded tensors. Example, fp32 K = 27 N = 320:
  peak 4,855 vs 4,560 MB.
* B loop has the lowest eager peak (360–668 MB in fp32 vs 840–4,560 MB) at 10–35% more time than fused.
* Compile removes these transients for all variants.

**Compile overhead** (cold, fresh Inductor cache, FX/AOT caches off): 1.3–4.7 s per configuration (the upper end is the loop strategy at K ≥ 25,
which unrolls K matmuls). The first configuration in each process also pays ~27 s of one-off process/Triton start-up.

## Answers

| question | answer | basis |
|---|---|---|
| Does B reduce saved activation memory? | Yes. Retained per layer drops from K·N·D to N·D scale: 3.6–14.5× compiled (more at larger K), e.g. 1,161 → 121 MB at K = 27, N = 320. | measured |
| Does it reduce peak GPU memory? | Compiled: yes, ~1.5–2.6× (2,240 → 1,240 MB at K = 27, N = 320). Eager fused: no (at or above A); eager loop: yes, strongly. | measured |
| Runtime, extra traffic or kernel overhead? | Compiled fused: 1.4–2.0× faster (production regimes), 1.2–1.3× in fp32. The loop strategy's K matmuls cost enough that compiled loop ≈ production at K ≥ 25. | measured |
| Advantage as K goes 9 → 27? | Retained-memory saving grows with K (B is K-independent, A linear in K). Speed-up stays 1.4–1.8× compiled. | measured |
| Does `torch.compile` keep the benefit? | Yes for fused B. Compile also gives plain-autograd B0 the same result, and erases loop's advantage. With fp32 neighbours under autocast, compile keeps a bf16 neighbours copy even for B. | measured |
| Suitable for 3D and all-layer GMP? | Yes. 6-layer estimate, regimes as above, N = 160: fwd+bwd 43.1 → 24.3 ms (K = 27) and 22.4 → 11.4 ms (K = 9); retained 3.4 → 0.33 GiB (K = 27). | **estimate**: sum of measured single layers; not measured in the full model |

## Issues found and fixed (job 7841399)

1. **Bug in the backend:** accumulators hard-coded to fp32 downcast fp64 computations (errors ~1e-7 in the fp64 tests).
   Accumulation now widens only 16-bit compute. Benchmarked precisions (fp32, bf16) were unaffected.
2. The benchmark's provenance record contained `Path` objects and could not be written as JSON. Fixed.
3. Wrong test expectation: plain autograd does not save the K-expanded tensor in fp32. The test now asserts the measured facts.
4. A bf16 sanity cap of 3e-2 was stricter than production's own error (3.15e-2 on the LayerNorm-weight gradient). The cap is
   now max(3e-2, production's error); the primary "≤ 2 × production" criterion is unchanged.

## Not verified

* GMP inside the full compiled model. The all-layer numbers above sum isolated layers, and the partitioner's choices in a
  larger graph may differ. A short compiled PHAT16 + all-layer GMP training benchmark (bench3 harness, `gmp_impl:
  weightsfirst`) would measure this.
* The encoder-input dtype in the real model (bf16 expected, not measured end-to-end).
* The reason for EinsumOpt's compiled K = 27 backward slowdown (not investigated).
* K = 27 uses a proxy 3D neighbourhood; the 3D coordinate definition is pending review.
* Training-level equivalence (loss curves) was not run; only operator-level equality was checked.

## Reproduce

```bash
cd src/hepattn/experiments/clic && qsub submit_gmp_backends_bench.sh            # tests + probe + all benchmarks, ~10 min
python -m hepattn.experiments.clic.a4.bench_gmp_backends --summarize <out>/bench_*.jsonl
```
