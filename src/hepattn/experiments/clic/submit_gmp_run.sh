#!/bin/bash -l
# ONE GMP ablation run (one arm, one seed), or one segment of it, as ONE PBS job: one model trained by 4-GPU DDP on one
# Polaris node (4 ranks, micro-batch 256 per GPU, accumulate 1 = global batch 1024, bf16-mixed, torch.compile).
# Normally submitted by gmp_ablation/campaign.py, which pins the code, sets every variable below and chains the jobs.
# NO queue is set here: the queue and walltime are given at submission (qsub -q ... -l walltime=...).
#
# Required (-v): CODE (pinned code checkout), EXPECTED_COMMIT, RUN_DIR (this arm/seed's own directory), ARM, SEED,
#   EPOCHS (the run's total budget = trainer.max_epochs, which the LR schedule spans).
# Optional (-v): STOP_AT_EPOCH (end this JOB after epoch STOP_AT_EPOCH-1: a segment of a split run),
#   LIMIT_TRAIN_BATCHES (smoke tests only), CAMPAIGN, ATTEMPT_TAG and JOB_START (several invocations in one job),
#   CAMPAIGN_MANIFEST (rolling chain: after the preflight passes, submit the next run with afterok on this job).
#
# Guarantees, all checked before any training step (exit codes: 10 code/commit, 11 GPUs, 12 import path, 13 config
# preflight, 14 a concurrent job owns RUN_DIR):
#   * the code is CODE at EXPECTED_COMMIT with a clean tracked tree, and hepattn is imported from it;
#   * 4 GPUs are visible; the resolved config passes gmp_ablation/preflight.py (4 devices, 1 node, accumulate 1,
#     batch 256, global 1024, bf16, compile/DDP/guard callbacks, epochs, seed, the arm's GMP settings);
#   * in the live process group the DDPGuard callback re-checks world size 4, 4 distinct GPUs, DDP, global batch 1024.
# Results never overwrite: a RUN_DIR with DONE.json exits 0 at once (a re-submitted chain passes through); otherwise the
# job resumes from the newest epoch checkpoint in RUN_DIR. Each job writes RUN_DIR/attempts/<job>/ (resolved config,
# preflight, DDP preflight, diagnostics, log) and appends RUN_DIR/attempts.jsonl.
# Exit status = gmp_ablation/runcheck.py's verdict (0 only if the target epoch was reached with finite loss/grads),
# so a PBS afterok chain advances only past a successful job.
# NEVER put a trailing comment on a #PBS line: qsub parses the rest of the line as arguments.
#PBS -A hgcal-maskformer-fpga
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -j oe

set -uo pipefail
ulimit -c 0
JOB_START=${JOB_START:-$(date +%s)}  # an outer job running several invocations passes its own start
JOB=${PBS_JOBID:-local}; JOB=${JOB%%.*}
: "${CODE:?}" "${EXPECTED_COMMIT:?}" "${RUN_DIR:?}" "${ARM:?}" "${SEED:?}" "${EPOCHS:?}"
STOP_AT_EPOCH=${STOP_AT_EPOCH:-}
LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-}
ENV_REPO=${ENV_REPO:-/home/siyingli/hepattn-phat}
ENVROOT=/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env
DATA=${DATA:-/eagle/hgcal-maskformer-fpga/clic_data}
PIXI=${PIXI:-${HOME}/.pixi/bin/pixi}
MARGIN_S=${MARGIN_S:-300}
CLIC=${CODE}/src/hepattn/experiments/clic
TOOLS=${CLIC}/gmp_ablation

mkdir -p "${RUN_DIR}"
if [ -f "${RUN_DIR}/DONE.json" ]; then echo "==> ${RUN_DIR} is already complete (DONE.json): nothing to do"; exit 0; fi
if [ -f "${RUN_DIR}/ACTIVE" ] && qstat "$(cat "${RUN_DIR}/ACTIVE")" >/dev/null 2>&1 && [ "$(cat "${RUN_DIR}/ACTIVE")" != "${PBS_JOBID:-}" ]; then
  echo "FATAL: job $(cat "${RUN_DIR}/ACTIVE") is active on ${RUN_DIR}"; exit 14
fi
echo "${PBS_JOBID:-local}" > "${RUN_DIR}/ACTIVE"
trap 'rm -f "${RUN_DIR}/ACTIVE"' EXIT
A=${RUN_DIR}/attempts/${JOB}${ATTEMPT_TAG:+_${ATTEMPT_TAG}}  # ATTEMPT_TAG: several invocations in one job
mkdir -p "${A}"
LOG=${A}/job.log
exec > >(tee -a "${LOG}") 2>&1

echo "gmp-run job ${PBS_JOBID:-none} host $(hostname) start $(date -Is)"
echo "  campaign ${CAMPAIGN:-?} arm ${ARM} seed ${SEED} epochs ${EPOCHS} stop_at ${STOP_AT_EPOCH:-end} limit_train ${LIMIT_TRAIN_BATCHES:-none}"
echo "  run dir ${RUN_DIR}"

# ---- code pin
head=$(git -C "${CODE}" rev-parse HEAD 2>/dev/null)
if [ "${head}" != "${EXPECTED_COMMIT}" ]; then echo "FATAL: ${CODE} is at ${head}, expected ${EXPECTED_COMMIT}"; exit 10; fi
if [ -n "$(git -C "${CODE}" status --porcelain --untracked-files=no)" ]; then echo "FATAL: ${CODE} has modified tracked files"; git -C "${CODE}" status --short; exit 10; fi
echo "  code ${CODE} @ ${head} (clean)"

# ---- environment
export PIXI_CACHE_DIR=${ENVROOT}/pixi-cache
set +u
eval "$("${PIXI}" shell-hook --manifest-path "${ENV_REPO}/pyproject.toml" -e clic --as-is)" || { echo "FATAL: pixi shell-hook"; exit 12; }
set -u
export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++
export PYTHONNOUSERSITE=1 HEPATTN_SRC=${CODE}/src PYTHONPATH=${CLIC}/bench3/shim
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_LOGS=${TORCH_LOGS:-recompiles,graph_breaks}
export TORCHINDUCTOR_CACHE_DIR=${RUN_DIR}/inductor_cache TRITON_CACHE_DIR=${RUN_DIR}/inductor_cache/triton
export GMP_RUN_DIR=${RUN_DIR} GMP_DIAG_OUT=${A}/diag.json GMP_STOP_AT_EPOCH=${STOP_AT_EPOCH}
unset CUDA_VISIBLE_DEVICES

# ---- GPUs and import path
ngpu=$(nvidia-smi -L | grep -c '^GPU')
nvidia-smi -L
if [ "${ngpu}" -ne 4 ]; then echo "FATAL: ${ngpu} GPUs visible, 4 required"; exit 11; fi
python - "${CODE}" <<'PY' || exit 12
import os, sys, torch, hepattn
f = os.path.realpath(hepattn.__file__)
print(f"  hepattn {f}; torch {torch.__version__} cuda {torch.version.cuda}; {torch.cuda.device_count()} CUDA devices")
sys.exit(0 if f.startswith(os.path.realpath(sys.argv[1]) + os.sep) and torch.cuda.device_count() == 4 else "FATAL: wrong hepattn or not 4 CUDA devices")
PY

# ---- config: resolve and preflight
cd "${CLIC}" || exit 13
ARGS=(-c configs/gmp_ablation/sweep/common.yaml -c "configs/gmp_ablation/sweep/${ARM}.yaml" --name "gmp_${ARM}_s${SEED}"
      --seed_everything "${SEED}" --trainer.max_epochs "${EPOCHS}"
      --data.train_path "${DATA}/train_clic_fix.root" --data.valid_path "${DATA}/val_clic_fix.root"
      --data.test_path "${DATA}/test_clic_common_infer.root"
      --trainer.default_root_dir "${RUN_DIR}" --trainer.logger.init_args.save_dir "${RUN_DIR}")
[ -n "${LIMIT_TRAIN_BATCHES}" ] && ARGS+=(--trainer.limit_train_batches "${LIMIT_TRAIN_BATCHES}")
ckpt=$(find "${RUN_DIR}" -path '*/ckpts/epoch=*.ckpt' 2>/dev/null | sed -E 's#.*/epoch=([0-9]+)-.*#\1 &#' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "${ckpt}" ] && { ARGS+=(--ckpt_path "${ckpt}"); echo "  resume from ${ckpt}"; }
python main.py fit "${ARGS[@]}" --print_config > "${A}/resolved.yaml" 2> "${A}/resolve.err" || { echo "FATAL: config does not resolve"; cat "${A}/resolve.err"; exit 13; }
python "${TOOLS}/preflight.py" "${A}/resolved.yaml" --arm-config "configs/gmp_ablation/sweep/${ARM}.yaml" --epochs "${EPOCHS}" --seed "${SEED}" \
  | tee "${A}/preflight.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 13

# ---- rolling chain (capacity queue: at most 2 campaign jobs queued+running): submit the successor now, with afterok on
# THIS job, so it waits in H while this one trains and is deleted by PBS if this one fails. Idempotent; retried below.
advance() {
  [ -n "${CAMPAIGN_MANIFEST:-}" ] || return 0
  python "${TOOLS}/campaign.py" advance "${CAMPAIGN_MANIFEST}" --job "${PBS_JOBID:-local}" 2>&1 | tee -a "${A}/advance.log"
}
advance

# ---- train (Lightning launches the 4 DDP ranks; DDPGuard checks the process group before the first step)
WALL=$(qstat -f "${PBS_JOBID:-none}" 2>/dev/null | awk -F' = ' '/Resource_List.walltime/ {print $2}')
if [[ "${WALL}" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then WALL_S=$(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} )); else WALL_S=3600; fi
LEFT=$(( JOB_START + WALL_S - MARGIN_S - $(date +%s) ))
python - "${RUN_DIR}/attempts.jsonl" <<PY
import json, os, sys, time
rec = {"job": os.environ.get("PBS_JOBID"), "host": os.uname().nodename, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "commit": "${EXPECTED_COMMIT}", "code": "${CODE}", "arm": "${ARM}", "seed": ${SEED}, "epochs": ${EPOCHS},
       "stop_at_epoch": "${STOP_AT_EPOCH}" or None, "resume_from": "${ckpt}" or None, "walltime_s": ${WALL_S}}
open(sys.argv[1], "a").write(json.dumps(rec) + "\n")
PY
echo "==> train $(date -Is), ${LEFT} s before the walltime margin"
timeout --signal=TERM "${LEFT}" python main.py fit "${ARGS[@]}" > "${A}/train.log" 2>&1
rc=$?
echo "==> main.py exit ${rc} $(date -Is)"
grep -h "\[ddp-guard\]\|DDPGuard" "${A}/train.log" | head -3
[ -f "${RUN_DIR}/ddp_preflight.json" ] && cp "${RUN_DIR}/ddp_preflight.json" "${A}/"

# ---- verdict: decides whether a PBS afterok chain advances
python "${TOOLS}/runcheck.py" "${RUN_DIR}" --attempt-dir "${A}" --epochs "${EPOCHS}" ${STOP_AT_EPOCH:+--stop-at-epoch "${STOP_AT_EPOCH}"}
verdict=$?
if [ "${rc}" -ne 0 ] && [ "${verdict}" -eq 0 ]; then echo "main.py exited ${rc} after reaching its target: failing the job anyway"; verdict=5; fi
[ "${verdict}" -eq 0 ] && advance  # no-op if the successor was submitted at start; retries a failed early attempt
echo "==> job verdict ${verdict} $(date -Is)"
exit "${verdict}"
