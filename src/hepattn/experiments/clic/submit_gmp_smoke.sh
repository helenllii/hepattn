#!/bin/bash -l
# Short DDP smoke test of the GMP ablation launcher, ONE debug job (no campaign). NOT submitted until the user approves.
#   1. pytest (GPU 0): the torch tests not yet executed -- the depth-wildcard GMP, the gmp_track_depth data path, DDPGuard
#      and SegmentStop -- plus the existing 3D / weights-first / placement / data suites (regression).
#   2. submit_gmp_run.sh twice on the SAME run directory, with all 4 GPUs (4-GPU DDP, micro-batch 256, accumulate 1),
#      arm g3d_env_dz200_trkany, EPOCHS=2, LIMIT_TRAIN_BATCHES=100:
#        segment 1: STOP_AT_EPOCH=1 -> trains epoch 0, checkpoint, verdict 0, segments/upto_1.json;
#        segment 2: resumes from epoch 0's checkpoint, trains epoch 1, verdict 0, DONE.json.
#      This exercises the config preflight, DDPGuard in the live process group, checkpoint/resume, SegmentStop, the
#      verdict and the DONE marker, and measures the validation time per epoch at 4 GPUs.
#
#   qsub submit_gmp_smoke.sh          # from src/hepattn/experiments/clic of a CLEAN checkout (the commit is pinned)
# NEVER put a trailing comment on a #PBS line: qsub parses the rest of the line as arguments.
#PBS -N gmp-smoke
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
CODE=${CODE:-/home/siyingli/hepattn-gmp-ablation}
ENV_REPO=${ENV_REPO:-/home/siyingli/hepattn-phat}
ENVROOT=/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env
JOB=${PBS_JOBID:-local}; JOB=${JOB%%.*}
OUT=${ENVROOT}/runs/gmp_ablation/smoke/${JOB}
if [ -e "${OUT}" ]; then echo "FATAL: ${OUT} exists"; exit 1; fi
mkdir -p "${OUT}"
COMMIT=$(git -C "${CODE}" rev-parse HEAD)
echo "gmp-smoke job ${PBS_JOBID:-none} host $(hostname) start $(date -Is) code ${CODE} @ ${COMMIT}" | tee "${OUT}/job.log"

# 1. tests
(
  export PIXI_CACHE_DIR=${ENVROOT}/pixi-cache
  set +u; eval "$(~/.pixi/bin/pixi shell-hook --manifest-path "${ENV_REPO}/pyproject.toml" -e clic --as-is)"; set -u
  export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++
  export PYTHONNOUSERSITE=1 HEPATTN_SRC=${CODE}/src PYTHONPATH=${CODE}/src/hepattn/experiments/clic/bench3/shim
  cd "${CODE}" || exit 1
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 timeout -k 30 1200 python -m pytest -v -s -p no:cacheprovider -rfEs --junitxml="${OUT}/pytest.xml" \
    tests/models/test_gmp3d_wildcard.py tests/experiments/clic/test_gmp_ddp_guard.py tests/experiments/clic/test_gmp_depth_data.py \
    tests/experiments/clic/test_gmp_campaign.py tests/experiments/clic/test_gmp_sweep_configs.py tests/models/test_gmp3d.py \
    tests/models/test_gmp3d_extra.py tests/models/test_gmp_weights_first.py tests/models/test_a4_placement.py > "${OUT}/pytest.log" 2>&1
  echo "==> pytest exit $? $(date -Is)" | tee -a "${OUT}/job.log"
)

# 2. two DDP segments on one run directory
RUN=${OUT}/run_g3d_env_dz200_trkany_s42
for seg in 1 2; do
  stop=""; [ "${seg}" = 1 ] && stop=1
  CODE=${CODE} EXPECTED_COMMIT=${COMMIT} RUN_DIR=${RUN} ARM=g3d_env_dz200_trkany SEED=42 EPOCHS=2 STOP_AT_EPOCH=${stop} \
  LIMIT_TRAIN_BATCHES=100 CAMPAIGN=smoke ATTEMPT_TAG=seg${seg} JOB_START=${JOB_START} \
    bash "${CODE}/src/hepattn/experiments/clic/submit_gmp_run.sh" > "${OUT}/segment${seg}.log" 2>&1
  echo "==> segment ${seg} exit $? $(date -Is)" | tee -a "${OUT}/job.log"
done
ls "${RUN}" "${RUN}/segments" 2>/dev/null | tee -a "${OUT}/job.log"
[ -f "${RUN}/DONE.json" ] && echo "==> smoke DONE" | tee -a "${OUT}/job.log"
echo "==> done $(date -Is)" | tee -a "${OUT}/job.log"
