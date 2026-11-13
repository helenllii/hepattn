#!/bin/bash -l
# Full compiled PHAT16 model benchmark of several GMP arms in ONE Polaris debug job (same node, run one after another),
# through the unchanged bench3 harness (submit_bench3_polaris.sh): full training set, 4 GPUs DDP, global batch 1024,
# bf16-mixed, torch.compile, 50 warmup + 150 measured batches, PROFILE=20 step breakdown, one validation pass.
# Each arm also gets the GMPDiagnostics callback (parameter counts, real dtypes, NaN/inf checks) and its OWN empty
# Inductor/Triton cache, so every arm's first-batch time is a cold compile (no cache hits from the arm before it).
#
#   qsub submit_gmp_fullmodel_bench.sh                                     # production vs weights-first, all 6 layers
#   qsub -v ARMS="configs/gmp_ablation:phat16_gmp3d_all_weightsfirst" submit_gmp_fullmodel_bench.sh
#
# ARMS: whitespace-separated <arm dir>:<arm name> (dir relative to src/hepattn/experiments/clic).
# Results: $OUT/results/<arm>_<job>.json (bench3 format), $OUT/logs/, $OUT/diag/<arm>_<job>.json.rank<r>.json
# NEVER put a trailing comment on a #PBS line: qsub parses the rest of the line as arguments.
#PBS -N gmp-fullmodel
#PBS -A hgcal-maskformer-fpga
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe

set -uo pipefail
ulimit -c 0
export JOB_START=$(date +%s)
BENCH_REPO=${BENCH_REPO:-/home/siyingli/hepattn-gmp-ablation}
ENVROOT=/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env
export OUT=${OUT:-${ENVROOT}/runs/gmp_ablation/fullmodel}
ARMS=${ARMS:-"configs/bench3:phat16_gmp_all_production configs/gmp_ablation:phat16_gmp_all_weightsfirst"}
export PROFILE=${PROFILE:-20} BENCH_REPO
JOB=${PBS_JOBID:-local}; JOB=${JOB%%.*}
CLIC=${BENCH_REPO}/src/hepattn/experiments/clic
mkdir -p "${OUT}/diag" "${OUT}/inductor"

echo "gmp-fullmodel job ${PBS_JOBID:-none} host $(hostname) start $(date -Is) arms: ${ARMS}"
echo "code ${BENCH_REPO} $(git -C "${BENCH_REPO}" rev-parse --abbrev-ref HEAD) @ $(git -C "${BENCH_REPO}" rev-parse HEAD)"
git -C "${BENCH_REPO}" status --short
cd "${CLIC}" || exit 1
for spec in ${ARMS}; do
  dir=${spec%%:*}; arm=${spec##*:}
  cache=${OUT}/inductor/${arm}_${JOB}
  if [ -e "${cache}" ]; then echo "FATAL: ${cache} exists"; exit 1; fi
  mkdir -p "${cache}"
  echo "==> arm ${arm} (${dir}) start $(date -Is)"
  ARM=${arm} ARM_DIR=${dir} GMP_DIAG_OUT=${OUT}/diag/${arm}_${JOB}.json \
  TORCHINDUCTOR_CACHE_DIR=${cache} TRITON_CACHE_DIR=${cache}/triton \
  EXTRA_ARGS="--trainer.callbacks+=hepattn.experiments.clic.gmp_ablation.diagnostics.GMPDiagnostics" \
    bash "${CLIC}/submit_bench3_polaris.sh"
  echo "==> arm ${arm} exit $? $(date -Is)"
done
echo "==> done $(date -Is)"
