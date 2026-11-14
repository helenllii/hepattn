# GMP ablation campaign: execution, runtime budget and scheduling options

Status 2026-11-14: the launcher, the guards, the campaign tool and the spec are implemented. **The queue and scheduling
strategy are pending the user's decision.** `configs/gmp_ablation/campaigns/ablation_v1.yaml` ships with `queue: null`
and `walltime: null`, and `campaign.py submit` refuses to submit while either is unset. No campaign job has been
submitted.

## 1. Execution setup: one model, 4 GPUs, one run at a time

| requirement | where it is set | where it is enforced (before the first training step) |
|---|---|---|
| 1 node, 4 × A100, DDP with 4 ranks | `sweep/common.yaml`: devices 4, num_nodes 1, strategy auto (Lightning DDP) | `preflight.py` on the resolved config; `DDPGuard`: world_size 4, a DDP strategy, `all_reduce` of 1 per rank = 4, 4 distinct GPU UUIDs, one host |
| micro-batch 256, accumulate 1, global 1024 | `common.yaml`: data.batch_size 256, accumulate_grad_batches 1 | both checks; the pilot's 1 GPU × accumulate 4 is rejected (tested) |
| bf16-mixed, torch.compile | `common.yaml`; Compile + DynamoOptimizeDDP callbacks | both checks |
| full training set, fixed validation selection | num_train -1, num_val 5000 | `preflight.py` |
| GMP in all 6 blocks, weights-first | the arm files (`sweep.py`); placement only in `g2d_e020_early` | `preflight.py` compares the resolved encoder with the arm |
| recipe unchanged | base.yaml's Lion / OneCycle over max_epochs (epochs from the campaign spec) | `preflight.py` checks max_epochs and the seed |
| one run at a time | one PBS job per run, chained `afterok` (`campaign.py`) | PBS: job k waits in H until k-1 exits 0 |
| own directory per arm/seed | `campaigns/<name>/runs/<NN>_<arm>_s<seed>/` | a DONE run is never retrained; a run directory owned by a live job is refused |
| code pinned | `campaign.py submit` makes a detached worktree of the current commit | each job checks the commit and a clean tracked tree (exit 10) |

Each job records in its run directory: `attempts/<job>/` (resolved config, preflight verdict, `ddp_preflight.json` with
world size / ranks / GPUs / UUIDs, GMP diagnostics, train log), plus `attempts.jsonl` (commit, code path, host, resume
point) and `DONE.json`. The `hepattn` import path is checked against the pinned code before training.

## 2. PBS dependencies on Polaris (verified, not assumed)

* Syntax: `qsub -W depend=afterok:<jobid>`, from Polaris's own `man qsub` (PBS Pro).
* Behaviour, tested 2026-11-14 with trivial no-GPU jobs (7848490–7848495, ~10 s each, in `preemptable`):
  * success chain A → B → C: the dependents waited in state H and ran strictly one after another;
  * failure chain: F exited 1; its dependents G and H **never started**. PBS deleted them: `job_state = F`, no start
    time, no exit status, no output. `campaign.py status` reports this as "cancelled (never ran: upstream failed or
    deleted)".
* A job's exit status is `runcheck.py`'s verdict: 0 only if the run (or segment) reached its target epoch with zero
  non-finite losses and gradients. A crash, a walltime kill, a preemption or a NaN therefore stops the chain, and
  nothing is skipped. `campaign.py submit --resume` re-chains from the first unfinished run; it refuses while any job
  of the campaign is still alive and never retrains a DONE run.

## 3. Runtime per epoch, 4-GPU DDP (estimated from existing measurements)

| component | 2D weights-first, all 6 blocks | 3D (3×3×3) weights-first, all 6 blocks | source |
|---|---|---|---|
| training step (mean) | 0.3484 s | 0.3784 s | bench3 harness, jobs 7841693 / 7841964 (150-step window, 4-GPU DDP, global 1024) |
| **training per epoch** (972 steps) | **5.64 min** | **6.13 min** | 972 × step |
| validation per epoch, steady | ~0.5 min (estimate) | ~0.5 min (estimate) | 5 batches per rank ≈ 1.4 s of compute plus loader start-up; the bench's "steady" 73–95 s included a one-off recompile for the partial last batch. The smoke test will measure it |
| checkpoint per epoch | seconds (116 MB) | seconds | pilot checkpoint size |
| per job, once: start-up + dataset init | ~2.0 min | ~2.0 min | bench startup 119–124 s |
| per job, once: first-batch compile (cold) | 3.5 min | 3.6 min | 208 / 213 s |
| per job, once: first validation compile | ~1.5 min | ~1.5 min | bench first validation pass |

The one-epoch pilot (job 7843955: 4 arms on one node, each 1 GPU × accumulate 4, CPUs shared) took 31.1–31.5 min of
training per epoch. 4-GPU DDP is about 5.5× faster in wall time, and it uses 22.6 GPU-minutes per epoch against 31.1,
because the pilot's arms competed for the node's CPUs (the matcher).

**Per run of 10 epochs, one job:** 2D ≈ 9 + 10 × (5.64 + 0.5) ≈ 70 min; 3D ≈ 75 min.

**Campaign `ablation_v1`, 11 runs:** ≈ 13.4 node-hours = 53.6 GPU-hours of compute, plus the queue wait before each job.
Split into segments, each extra job adds roughly 2 min of start-up plus a recompile (cold ~5 min; probably less with
the per-run Inductor cache, not measured).

## 4. Scheduling options (the user's decision)

Queues that accept a 1-node job (`qstat -Qf`, 2026-11-14):

| queue | nodes | max walltime | relevant limits |
|---|---|---|---|
| `debug` | 1–2 | 1 h | 1 running and **1 queued (Q)** job per user |
| `debug-scaling` | 1–10 | 1 h | 1 queued per user |
| `preemptable` | 1–10 | 72 h | 20 queued per user; 10 running per project; jobs can be preempted |
| `prod` → `small` / `backfill-small` | **≥ 10** | 3 h | not usable for a 1-node job (or charges 10 nodes) |
| `medium` / `large` | ≥ 25 | 6 h+ | not usable |
| `demand`, `run_next` | — | — | access lists (specific groups/users); not available to us |

| option | preemption risk | wall-clock overhead | reproducibility | operational complexity |
|---|---|---|---|---|
| **A. `preemptable`, one job per run** (walltime ~1.5 h 2D / ~1.6 h 3D, ~25% margin), whole chain pre-submitted with `afterok` | yes: a preempted job ends the chain; `--resume` restarts the run from its last epoch checkpoint (the interrupted epoch is lost) | lowest: one start-up and one compile per run (~9 min of ~70) | full; a resume restarts at an epoch boundary (optimizer, LR schedule, epoch and sampler seed restored by Lightning; bit-identity across a resume not guaranteed: dataloader workers, non-deterministic kernels) | low: submit once; after a preemption, run `submit --resume` once |
| **B. `preemptable`, segmented** (e.g. ≤ 4 epochs per job) | same queue, but a preemption costs at most one segment | +2 jobs per run: ~+10–20 min per run | same as A at every segment boundary (by design) | low (same tooling); about 33 jobs instead of 11 |
| **C. `debug`, segmented** (≤ 6 epochs per job, ~48 min for 3D) | none | +1 job per run (~+9 min), **plus a debug-queue wait before every job** (observed from minutes to about an hour) | as B | **high**: the 1-queued-job limit makes a pre-submitted chain impossible (verified: "would exceed queue generic's per-user limit of jobs in 'Q' state"). Automatic sequencing would need each job to `qsub` its successor from the compute node at the end (not verified to be allowed), and it would occupy the user's only debug slot for days. The debug queue is meant for debugging; a multi-day campaign there may be against ALCF's usage policy (to be checked with ALCF) |
| **D. ALCF reservation** (support request for 1 node over a time window) | none | none beyond start-up; no queue waits | best | needs ALCF approval; the chain then runs in the reservation queue with the same tooling |
| E. `prod` with 10 nodes, using 1 | none | — | — | wastes 9 nodes per job; not recommended |

Splitting a run into shorter checkpointed jobs is supported (`epochs_per_job` in the spec): `SegmentStop` ends a job
at an epoch boundary, and the next job resumes from the last epoch checkpoint with the run's full OneCycle schedule. It
reduces what a preemption can cost (B) and is the only way to fit 10 epochs into 1-hour queues (C). It does not remove
C's one-queued-job limit.

## 5. Proposed sequential order (`ablation_v1`, 10 epochs per run)

| # | run | ablation |
|---|---|---|
| 0 | g2d_e020, seed 42 | A: 2D baseline (also C's all-blocks arm) |
| 1 | g3d_env_dz200, seed 42 | A: 3D, tracks at depth 0 |
| 2 | g3d_env_dz200_trkany, seed 42 | A: 3D, track–cluster pairs ignore depth |
| 3–5 | the same three, seed 43 | A: run-to-run spread |
| 6 | g2d_e010, seed 42 | B: 2D η 0.1 / 64 φ cells |
| 7 | g2d_e040, seed 42 | B: 2D η 0.4 / 16 φ cells |
| 8 | g3d_env_dz100, seed 42 | B: 3D, 100 mm depth bins |
| 9 | g3d_env_dz400, seed 42 | B: 3D, 400 mm depth bins |
| 10 | g2d_e020_early, seed 42 | C: GMP in block 0 only |

## 6. Before the campaign

1. One short DDP smoke test (`submit_gmp_smoke.sh`, one debug job, 2 epochs × 100 batches, the new trkany arm). It runs
   segment 1, then resumes segment 2 in a second invocation, exercising the preflight, DDPGuard, checkpoint/resume,
   the verdict and DONE. It also gives a steady validation time. Pending approval.
2. The torch tests not yet executed (`tests/models/test_gmp3d_wildcard.py`, the data test for `gmp_track_depth`,
   `tests/experiments/clic/test_gmp_ddp_guard.py`) run in the same debug job.
3. The user picks the scheduling option; it is set in the spec's `queue`, `walltime` and `epochs_per_job`, committed,
   and submitted once:
   `python src/hepattn/experiments/clic/gmp_ablation/campaign.py submit src/hepattn/experiments/clic/configs/gmp_ablation/campaigns/ablation_v1.yaml`;
   then `... campaign.py status <spec>` at any time.
