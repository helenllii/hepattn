# bench3: controlled Polaris runtime comparison (Linformer / PHAT16 / PHAT16+GMP / MaskFormer), 1 node = 4x A100-40GB

Plan: `~/.claude/plans/humble-swimming-nebula.md`. The "PHAT+GMP is several times slower than Linformer" observation
(Falcon A4 vs others' Linformer runs, different hardware/recipes) is treated as a hypothesis; these runs measure it
like-for-like.

## Arms (`configs/bench3/<ARM>.yaml`, model-only overlays; training setup is `common.yaml` + launcher, identical)

| ARM | encoder | round |
|---|---|---|
| `linformer_encoder` | masked-linformer k=64 (byte-identical copy of hepattn-phat's untracked `configs/linformer_encoder.yaml`) | 1 (baseline) |
| `phat16` | PHAT16, no GMP | 1 |
| `phat_gmp_einsumopt_early` | PHAT16 + GMP layer 0, EinsumOpt (= validated a4 arm) | 1 |
| `maskformer` | base.yaml flash-varlen full attention | 2 |
| `phat16_gmp_all_production` | PHAT16 + GMP all 6 layers, production impl (= production patch_gmp) | 2 / 3 |
| `phat16_gmp_all_einsumopt` | all 6 layers, EinsumOpt | 3 |
| `phat16_gmp_early_production` | layer 0, production impl | 3 |

All PHAT arms share one PlacementEncoder block and differ only in `gmp`, `gmp_layers`, `gmp_impl`,
`input_coord_fields` (pinned by `tests/experiments/clic/test_bench3.py`). Round 3 is a 2x2 placement x impl grid,
compared one factor at a time, after `gmp_equivalence.py` passes.

Common: dim 256, bf16-mixed, Lion+OneCycle, clip 0.1, full training set, global batch 1024 = 256 x 4 GPUs x 1
(972 batches/epoch), compile + DynamoOptimizeDDP, 7 workers/rank, matcher n_jobs 8, OMP 8, BLAS threads 1.

## Steps (all PBS debug jobs; nothing GPU-side on the login node)

```bash
cd /home/siyingli/hepattn-polaris-bench/src/hepattn/experiments/clic
qsub -v ROUND0=1,ARMS=linformer_encoder:phat16:phat_gmp_einsumopt_early submit_bench3_polaris.sh  # smoke+memory @256, encoder-only, unit tests
qsub -v ARM=linformer_encoder,PROFILE=20         submit_bench3_polaris.sh                        # Round 1
qsub -v ARM=phat16,PROFILE=20                    submit_bench3_polaris.sh
qsub -v ARM=phat_gmp_einsumopt_early,PROFILE=20  submit_bench3_polaris.sh
python -m hepattn.experiments.clic.bench3.summarize $OUT/results       # $OUT = .../patch-attention-env/runs/bench3
python -m hepattn.experiments.clic.bench3.summarize --round0 $OUT/round0_<job>
```

Per job, in priority order under a walltime guard (deadline = start + walltime - 5 min): startup/dataset init/compile
/warmup timed separately -> steady-state window -> `train_done` JSON written -> optional breakdown (instrumented phases,
matcher CPU vs GPU-wait, sync count) + torch.profiler -> one validation pass -> `complete` JSON. Results go straight
to `$OUT/results/<arm>_<job>.json` with provenance (hepattn file, git commit/dirty/diff hash, bench-file sha256s) and
the effective CPU settings.

## Import shim

The clic env's hepattn is a scikit-build-core editable install; its finder at `sys.meta_path[0]` maps `hepattn.*` to
hepattn-phat and beats PYTHONPATH. `shim/sitecustomize.py` (PYTHONPATH + `HEPATTN_SRC`) puts a finder in front that
resolves only `hepattn.*` under this worktree and never falls back; `lap1015` still comes from the env build.
