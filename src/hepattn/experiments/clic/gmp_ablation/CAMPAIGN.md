# GMP ablation campaign: execution, runtime budget and scheduling

Status 2026-11-14. Implemented, and the 4-GPU DDP smoke test passed (job 7849183, section 6). The scheduling plan
recommended below (`capacity`, rolling `afterok` chain, 2 h walltime) awaits the user's approval.
`configs/gmp_ablation/campaigns/ablation_v1.yaml` keeps `queue`, `walltime` and `chain` null, and `campaign.py submit`
refuses while any of them is unset. No campaign job has been submitted.

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

## 4. Queues for one-node jobs (live `qstat -Qf`, 2026-11-14, and docs.alcf.anl.gov)

| queue | nodes | walltime | job limits (live config) | preemption |
|---|---|---|---|---|
| **`capacity`** | 1–4 (max 32 nodes in the queue) | up to **168 h** | `max_queued [p:2]`, `max_run [p:1]`: **2 jobs queued-or-running and 1 running PER PROJECT**. The docs say "per user": the live config is stricter, and other members of hgcal-maskformer-fpga share it | **none documented**. Its nodes carry the `prod` tag; the docs list preemption only for `preemptable` ("can be killed at any time if jobs are submitted to the demand queue") |
| `preemptable` | 1–10 | 72 h | 20 queued per user/project, 10 running per project | yes, by `demand` jobs ("killed without any warning"; `#PBS -r y` makes a job rerunnable) |
| `debug` | 1–2 | 1 h | 1 running and 1 queued (Q) per user | no |
| `debug-scaling` | 1–10 | 1 h | 1 queued per user | no |
| `prod` (`small`, `backfill-*`, ...) | ≥ 10 | — | — | — (1-node jobs impossible) |
| `demand`, `run_next` | — | — | access lists only | — |

* **Eligibility:** `capacity` has no access list. Our project's jobs are accepted: 7849194 ran and 7849258 was queued.
  Allocation `hgcal-maskformer-fpga`: 4,256.8 node-hours available until 2027-01-10 (`sbank`).
* **Not verified:** whether `capacity` jobs can ever be preempted or killed for other reasons; the docs are silent. An
  ALCF ticket would settle it. Checkpoint/resume (section 6) limits the damage either way.
* **Queue wait (measured 2026-11-14):** 7849194 waited 17 min (queued 19:47:32, started 20:04:06); its successor
  7849258, released when 7849194 exited at ~20:04:30, waited **58 min** (started 21:02:20; "Insufficient amount of
  resource: queue_tags"). The successor does not start until its predecessor has ended, so with a rolling chain such a
  wait falls between every two runs.

## 5. Sequential scheduling on `capacity`: the rolling `afterok` chain

`capacity` holds at most 2 queued-or-running jobs per project, so the full 11-job chain cannot be queued at once
(`chain: full` stays available for queues that allow it). With **`chain: rolling`**:

1. `campaign.py submit` pins the code (detached worktree of the current clean commit), writes the manifest and queues
   **only the first unfinished run**.
2. Each run's job, as soon as its config preflight has passed, calls `campaign.py advance`. That submits the **next
   unfinished run from the compute node with `-W depend=afterok:<this job>`**. The successor waits in state H while this
   run trains: 2 campaign jobs in the queue, within the limit.
3. The run succeeds (verdict 0): PBS releases the successor, which repeats step 2. The run fails (crash, NaN, walltime,
   kill): PBS deletes the successor unrun and the campaign stops. Nothing is skipped.
4. `advance` is idempotent (a live successor means no-op; it is called again after a successful verdict as a retry),
   never resubmits a finished run, and writes the manifest atomically. If its `qsub` is refused (e.g. another member of
   the project holds the second slot), it writes `STALLED.json` instead of failing the training job. `status` reports
   the stall, and `campaign.py submit --resume <spec>` continues once the running job has ended.

Verified on Polaris (trivial jobs, no GPU):
* `qsub` works from a compute node, end to end in `capacity`: job 7849194 ran `/opt/pbs/bin/qsub` on x3004c0s1b0n0
  and submitted 7849258 with `afterok:7849194` (PBS recorded `beforeok:7849258` on the parent). 7849194 exited 0;
  7849258 was released and ran at 21:02:20, exit 0.
* `afterok` after a non-zero exit deletes the dependents unrun: pre-submitted (7848493 → 7848494, 7848495) and
  in-job submitted (debug: 7849431 ran `qsub` on its compute node for 7849523 with `afterok:7849431`, then exited 1;
  7849523 ended in state F with no start time and no exit status, never run).
* Not documented by ALCF either way: whether in-job `qsub` is intended use. It is the standard PBS mechanism and costs
  nothing while the chain is idle. No login-node daemon or cron is used: the docs say nothing about their policy, and
  the rolling chain does not need one.

Alternative without in-job submission: one `capacity` job running all 11 runs one after another (≈ 15 h of the 168 h
limit), still each run in its own process and directory, stopping at the first failure. It is simpler, but it is not one
PBS job per run, and a 15 h job may wait longer in the queue.

## 6. Smoke test (debug job 7849183, commit 0278d1a): passed

* 352 tests passed (3D GMP, depth wildcard, gmp_track_depth data path, DDPGuard / SegmentStop, campaign, configs,
  weights-first, a4 placement, data).
* Two DDP segments of a 2-epoch run (arm g3d_env_dz200_trkany, 100 batches per epoch), on one run directory:
  * preflight ok; DDPGuard: `DDPStrategy`, world size 4, `all_reduce` = 4, 4 distinct GPU UUIDs, ranks 0-3 on CUDA
    0-3 of one host; global batch 1024;
  * segment 1 trained epoch 0, wrote its checkpoint and segments/upto_1.json; segment 2 resumed from that checkpoint,
    trained epoch 1 and wrote DONE.json; both with 0 non-finite losses and gradients; peak 10.96 GiB;
  * `resume_check.py`: global step 100 restored; resumed at epoch 1; model state_dict sum, Lion optimizer-state sum and
    scheduler (last_epoch 100, _step_count 101, total_steps 200, _last_lr) identical to the checkpoint; every logged LR
    equals one uninterrupted OneCycleLR (max rel diff 0); epochs 0 and 1 each trained once;
  * metric logging: each attempt wrote its own CSV (99 validation metrics); the summary merges them.
* Costs measured: a resumed job with a warm per-run Inductor cache spent ~0.4 min on start-up, ~1.5 min of extra
  first-batch compile and 70 s on its first validation pass.

**Verified by the smoke test:** the DDP configuration and guards, checkpoint save, resume of weights, optimizer,
scheduler, global step and epoch, LR continuity across a resume, no repeated or skipped epoch, the job verdict and the
DONE / segment markers, CSV logging across attempts.
**Not yet verified:** steady validation time per epoch (each smoke job ran a single, compile-including validation pass);
a full 972-step epoch under this launcher (the step time comes from the bench3 harness); a mid-epoch kill followed by
`--resume` (by construction it resumes from the last completed epoch, losing the partial one); the rolling chain end to
end with real runs (its parts are unit-tested and the PBS behaviours verified with trivial jobs).

## 7. Budget: ablation_v1 on `capacity`, rolling chain, 10 epochs per run

| | per run | campaign (11 runs) |
|---|---|---|
| compute: one-off (start-up + compile + first validation) | ~7–9 min | |
| compute: 10 epochs x (training + validation) | 2D 10 x (5.64 + ~0.5) min; 3D 10 x (6.13 + ~0.5) min | |
| **compute total** | **2D ≈ 70 min, 3D ≈ 75 min** | **≈ 13.4 node-hours = 53.6 GPU-hours** (0.3 % of the allocation's 4,256.8 node-hours) |
| recommended walltime | **02:00:00** (~1.6x margin; docs: shorter requests gain priority faster) | |
| queue wait before each run | 17 and 58 min observed (2 samples) | 11 waits ≈ 3–11 h |
| **wall clock** | | **≈ 13.4 h + 11 waits ≈ 16.5–24 h** if waits stay in the observed range |

Run order (unchanged): track depth (2D, 3D tracks at 0, 3D tracks at any depth) x seeds 42 and 43; then 2D η 0.1 and
0.4, 3D 100 and 400 mm bins; then GMP in block 0 only. Ten epochs are a screening budget, not convergence.

## 8. To submit (after approval)

Set in `configs/gmp_ablation/campaigns/ablation_v1.yaml`: `queue: capacity`, `walltime: "02:00:00"`, `chain: rolling`
(epochs_per_job stays null). Commit, then:

    python src/hepattn/experiments/clic/gmp_ablation/campaign.py submit src/hepattn/experiments/clic/configs/gmp_ablation/campaigns/ablation_v1.yaml
    python src/hepattn/experiments/clic/gmp_ablation/campaign.py status src/hepattn/experiments/clic/configs/gmp_ablation/campaigns/ablation_v1.yaml
