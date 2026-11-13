#!/bin/bash -l
# 3D GMP validation on ONE Polaris debug node, no training run:
#   1. (parallel) pytest -- 3D GMP, weights-first, a4 placement, production GMP, patch attention, bench3 config pins,
#      data-loader options and the gmp_depth data test (CPU + CUDA on GPU0, cores 0-15) -- and the depth-coordinate
#      data study gmp_ablation/geometry_study.py (CPU, cores 16-31).
#   2. the full compiled PHAT16 model with 3D GMP in all 6 blocks through submit_gmp_fullmodel_bench.sh (bench3
#      harness, 4 GPUs): weights-first first, then the production backend; each arm cold-compiled, with diagnostics.
# Step 2 starts after step 1 so the benchmark has the node to itself.
#
#   qsub submit_gmp3d_validate.sh                      (from src/hepattn/experiments/clic of the exp/gmp-ablation tree)
#   qsub -v SKIP_ARMS=1 submit_gmp3d_validate.sh       tests + study only
# NEVER put a trailing comment on a #PBS line: qsub parses the rest of the line as arguments.
#PBS -N gmp3d-validate
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
ENV_REPO=${ENV_REPO:-/home/siyingli/hepattn-phat}
ENVROOT=/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env
JOB=${PBS_JOBID:-local}; JOB=${JOB%%.*}
VOUT=${VOUT:-${ENVROOT}/runs/gmp_ablation/validate3d/${JOB}}
ARMS=${ARMS:-"configs/gmp_ablation:phat16_gmp3d_all_weightsfirst configs/gmp_ablation:phat16_gmp3d_all_production"}
SKIP_ARMS=${SKIP_ARMS:-0}
if [ -e "${VOUT}" ]; then echo "FATAL: ${VOUT} exists"; exit 1; fi
mkdir -p "${VOUT}"
CLIC=${BENCH_REPO}/src/hepattn/experiments/clic

(
  export PIXI_CACHE_DIR=${ENVROOT}/pixi-cache
  set +u
  eval "$(~/.pixi/bin/pixi shell-hook --manifest-path "${ENV_REPO}/pyproject.toml" -e clic --as-is)" || exit 1
  set -u
  export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++
  export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8
  export HEPATTN_SRC=${BENCH_REPO}/src
  export PYTHONPATH=${CLIC}/bench3/shim${PYTHONPATH:+:${PYTHONPATH}}
  {
    echo "gmp3d-validate job ${PBS_JOBID:-none} host $(hostname) start $(date -Is)"
    echo "code ${BENCH_REPO} $(git -C "${BENCH_REPO}" rev-parse --abbrev-ref HEAD) @ $(git -C "${BENCH_REPO}" rev-parse HEAD)"
    git -C "${BENCH_REPO}" status --short
    python -c "import os, hepattn; print('hepattn', os.path.realpath(hepattn.__file__))"
  } 2>&1 | tee "${VOUT}/job.log"
  git -C "${BENCH_REPO}" diff HEAD > "${VOUT}/worktree.diff"
  cd "${BENCH_REPO}" || exit 1
  (CUDA_VISIBLE_DEVICES=0 timeout -k 30 1500 taskset -c 0-15 python -X faulthandler -m pytest -v -s -p no:cacheprovider -rfEs \
     -o faulthandler_timeout=600 --durations=20 --junitxml="${VOUT}/pytest.xml" \
     tests/models/test_gmp3d.py tests/models/test_gmp_weights_first.py tests/models/test_a4_placement.py tests/models/test_gmp.py \
     tests/models/test_patch_attention.py tests/experiments/clic/test_bench3.py tests/experiments/clic/test_pflow_dataloader_options.py \
     tests/experiments/clic/test_gmp_depth_data.py tests/experiments/clic/test_gmp_sweep_configs.py > "${VOUT}/pytest.log" 2>&1
   echo "==> pytest exit $? $(date -Is)" | tee -a "${VOUT}/job.log") &
  (CUDA_VISIBLE_DEVICES= timeout -k 30 1500 taskset -c 16-31 python -u -m hepattn.experiments.clic.gmp_ablation.geometry_study \
     --events 3000 --cell-events 300 --out "${VOUT}/geometry_study.json" > "${VOUT}/geometry_study.log" 2>&1
   echo "==> geometry study exit $? $(date -Is)" | tee -a "${VOUT}/job.log") &
  wait
)

if [ "${SKIP_ARMS}" != 1 ]; then
  echo "==> 3D full-model arms $(date -Is): ${ARMS}" | tee -a "${VOUT}/job.log"
  ARMS="${ARMS}" BENCH_REPO="${BENCH_REPO}" bash "${CLIC}/submit_gmp_fullmodel_bench.sh" 2>&1 | tee "${VOUT}/fullmodel.log"
  echo "==> arms done $(date -Is)" | tee -a "${VOUT}/job.log"
fi
echo "==> done $(date -Is); ${VOUT}" | tee -a "${VOUT}/job.log"
