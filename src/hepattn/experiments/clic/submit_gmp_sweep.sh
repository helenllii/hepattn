#!/bin/bash -l
# GMP grid-size / placement sweep: up to 4 arms side by side on ONE Polaris node, one GPU each (8 physical cores + their
# SMT siblings each), identical recipe/data/batch/seed (configs/gmp_ablation/sweep/common.yaml), EPOCHS epochs each.
# Arms are named in gmp_ablation/sweep.py (configs/gmp_ablation/sweep/<arm>.yaml).
#
#   qsub -v CAMPAIGN=pilot,EPOCHS=1,ARMS="g2d_e020 g2d_e010 g3d_env_dz200 g3d_env_dz100" submit_gmp_sweep.sh
#
# Output: $OUT/<CAMPAIGN>/<job>/<arm>/ (CSV metrics, checkpoints, config, diagnostics, log); a summary.md at the end.
# Throughput of side-by-side arms is measured under the same contention for every arm; it is not a dedicated-node
# number (use submit_gmp_fullmodel_bench.sh for those).
# NEVER put a trailing comment on a #PBS line: qsub parses the rest of the line as arguments.
#PBS -N gmp-sweep
#PBS -A hgcal-maskformer-fpga
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe

set -uo pipefail
ulimit -c 0
BENCH_REPO=${BENCH_REPO:-/home/siyingli/hepattn-gmp-ablation}
ENV_REPO=${ENV_REPO:-/home/siyingli/hepattn-phat}
ENVROOT=/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env
DATA=${DATA:-/eagle/hgcal-maskformer-fpga/clic_data}
CAMPAIGN=${CAMPAIGN:-pilot}
EPOCHS=${EPOCHS:-1}
SEED=${SEED:-42}
ARMS=${ARMS:?qsub -v ARMS="<arm> ..." (gmp_ablation/sweep.py)}
GEOMETRY=${GEOMETRY:-}
JOB=${PBS_JOBID:-local}; JOB=${JOB%%.*}
OUT=${OUT:-${ENVROOT}/runs/gmp_ablation/sweep}/${CAMPAIGN}/${JOB}
CLIC=${BENCH_REPO}/src/hepattn/experiments/clic
if [ -e "${OUT}" ]; then echo "FATAL: ${OUT} exists"; exit 1; fi
read -r -a arms <<< "${ARMS}"
[ "${#arms[@]}" -le 4 ] || { echo "FATAL: at most 4 arms per node"; exit 1; }
for a in "${arms[@]}"; do [ -f "${CLIC}/configs/gmp_ablation/sweep/${a}.yaml" ] || { echo "FATAL: no arm ${a}"; exit 1; }; done
mkdir -p "${OUT}"

export PIXI_CACHE_DIR=${ENVROOT}/pixi-cache
set +u
eval "$(~/.pixi/bin/pixi shell-hook --manifest-path "${ENV_REPO}/pyproject.toml" -e clic --as-is)" || exit 1
set -u
export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export HEPATTN_SRC=${BENCH_REPO}/src
export PYTHONPATH=${CLIC}/bench3/shim${PYTHONPATH:+:${PYTHONPATH}}
{
  echo "gmp-sweep ${CAMPAIGN} job ${PBS_JOBID:-none} host $(hostname) start $(date -Is) epochs ${EPOCHS} seed ${SEED} arms: ${ARMS}"
  echo "code ${BENCH_REPO} $(git -C "${BENCH_REPO}" rev-parse --abbrev-ref HEAD) @ $(git -C "${BENCH_REPO}" rev-parse HEAD)"
  git -C "${BENCH_REPO}" status --short
  python -c "import os, hepattn; print('hepattn', os.path.realpath(hepattn.__file__))"
  nvidia-smi -L
} 2>&1 | tee "${OUT}/job.log"
git -C "${BENCH_REPO}" diff HEAD > "${OUT}/worktree.diff"
cd "${CLIC}" || exit 1

i=0
for a in "${arms[@]}"; do
  d=${OUT}/${a}
  mkdir -p "${d}/inductor"
  echo "${a}" > "${d}/arm.txt"
  cores="$((8 * i))-$((8 * i + 7)),$((32 + 8 * i))-$((32 + 8 * i + 7))"
  ARGS=(-c configs/gmp_ablation/sweep/common.yaml -c "configs/gmp_ablation/sweep/${a}.yaml" --name "gmp_sweep_${a}"
        --seed_everything "${SEED}" --trainer.max_epochs "${EPOCHS}" --trainer.devices 1 --trainer.accumulate_grad_batches 4
        --data.train_path "${DATA}/train_clic_fix.root" --data.valid_path "${DATA}/val_clic_fix.root"
        --data.test_path "${DATA}/test_clic_common_infer.root"
        --model.model.init_args.matcher.init_args.n_jobs 4
        --trainer.default_root_dir "${d}" --trainer.logger.init_args.save_dir "${d}")
  python main.py fit "${ARGS[@]}" --print_config > "${d}/resolved.yaml" 2> "${d}/print_config.err" \
    || { echo "FATAL: ${a} config does not resolve" | tee -a "${OUT}/job.log"; continue; }
  echo "==> ${a} on GPU ${i} cores ${cores} start $(date -Is)" | tee -a "${OUT}/job.log"
  ( CUDA_VISIBLE_DEVICES=${i} GMP_DIAG_OUT="${d}/diag.json" TORCHINDUCTOR_CACHE_DIR="${d}/inductor" TRITON_CACHE_DIR="${d}/inductor/triton" \
      timeout --signal=TERM 3300 taskset -c "${cores}" python main.py fit "${ARGS[@]}" > "${d}/train.log" 2>&1
    echo "==> ${a} exit $? $(date -Is)" | tee -a "${OUT}/job.log" ) &
  i=$((i + 1))
done
wait
python -m hepattn.experiments.clic.gmp_ablation.sweep summarize "${OUT}" ${GEOMETRY:+--geometry "${GEOMETRY}"} > "${OUT}/summary.md" 2>&1
echo "==> done $(date -Is); ${OUT}" | tee -a "${OUT}/job.log"
