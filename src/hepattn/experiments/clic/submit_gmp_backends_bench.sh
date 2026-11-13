#!/bin/bash -l
# GMP backend correctness tests + micro-benchmark (production A, EinsumOpt A', weights-first B). ONE Polaris debug node,
# no training. Everything lands in a new per-job directory; nothing earlier is overwritten.
#
#   qsub submit_gmp_backends_bench.sh                    # from src/hepattn/experiments/clic of the exp/gmp-ablation tree
#   qsub -v BENCH_REPO=/path/to/tree,OUT=/path submit_gmp_backends_bench.sh
#
# Layout (each GPU process pinned to its own 8 physical cores, so the four streams do not share CPUs):
#   GPU0: pytest (new weights-first tests + the a4 placement tests, CPU and CUDA parametrizations), the autocast dtype
#         probe, the EAGER benchmark for all precisions, then the torch.compile benchmark for amp_xfp32.
#   GPU1/2/3: the torch.compile benchmark for fp32 / amp_xbf16 / amp_xfp32_nbbf16.
#   Every compile process has a fresh Inductor cache with the FX-graph/AOT-autograd caches off, so the recorded
#   first-call time is a cold compile.
# NEVER put a trailing comment on a #PBS line: qsub parses the rest of the line as arguments.
#PBS -N gmp-backends
#PBS -A hgcal-maskformer-fpga
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe

set -uo pipefail
ulimit -c 0
JOB_START=$(date +%s)
BENCH_REPO=${BENCH_REPO:-/home/siyingli/hepattn-gmp-ablation}
ENV_REPO=${ENV_REPO:-/home/siyingli/hepattn-phat}
ENVROOT=/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env
JOB=${PBS_JOBID:-local}; JOB=${JOB%%.*}
OUT=${OUT:-${ENVROOT}/runs/gmp_ablation/microbench_weights_first/${JOB}}
PIXI=${PIXI:-${HOME}/.pixi/bin/pixi}
MARGIN_S=${MARGIN_S:-240}
if [ -e "${OUT}" ]; then echo "FATAL: ${OUT} exists; refusing to overwrite"; exit 1; fi
mkdir -p "${OUT}"
DEADLINE=$(( JOB_START + 3600 - MARGIN_S ))

export PIXI_CACHE_DIR=${ENVROOT}/pixi-cache
set +u
eval "$("${PIXI}" shell-hook --manifest-path "${ENV_REPO}/pyproject.toml" -e clic --as-is)" || { echo "FATAL: pixi shell-hook"; exit 1; }
set -u
export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8
export HEPATTN_SRC=${BENCH_REPO}/src
export PYTHONPATH=${BENCH_REPO}/src/hepattn/experiments/clic/bench3/shim${PYTHONPATH:+:${PYTHONPATH}}

{
  echo "job ${PBS_JOBID:-none} host $(hostname) start $(date -Is)"
  echo "code ${BENCH_REPO} $(git -C "${BENCH_REPO}" rev-parse --abbrev-ref HEAD) @ $(git -C "${BENCH_REPO}" rev-parse HEAD)"
  git -C "${BENCH_REPO}" status --short
  echo "env ${CONDA_PREFIX}"
  nvidia-smi -L
} 2>&1 | tee "${OUT}/job.log"
git -C "${BENCH_REPO}" diff HEAD > "${OUT}/worktree.diff"

# Preflight: this tree's hepattn, not the env's editable install
python - "${BENCH_REPO}" <<'PY' 2>&1 | tee -a "${OUT}/job.log"
import os, sys, torch, hepattn
from hepattn.experiments.clic.a4 import GMP_IMPLS
root = os.path.realpath(sys.argv[1])
f = os.path.realpath(hepattn.__file__)
print(f"preflight: hepattn {f}; torch {torch.__version__} cuda {torch.version.cuda}; {torch.cuda.device_count()} GPUs; impls {sorted(GMP_IMPLS)}")
sys.exit(0 if f.startswith(root + os.sep) else f"FATAL: hepattn from {f}, not {root}")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL: preflight" | tee -a "${OUT}/job.log"; exit 1; }

left() { echo $(( DEADLINE - $(date +%s) )); }
bench() {  # gpu cores tag args...
  local gpu=$1 cores=$2 tag=$3; shift 3
  mkdir -p "${OUT}/inductor_${tag}"
  CUDA_VISIBLE_DEVICES=${gpu} TORCHINDUCTOR_CACHE_DIR="${OUT}/inductor_${tag}" TRITON_CACHE_DIR="${OUT}/inductor_${tag}/triton" \
  TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
    timeout -k 30 "$(left)" taskset -c "${cores}" python -m hepattn.experiments.clic.a4.bench_gmp_backends \
      --deadline "$(( DEADLINE - 120 ))" --out "${OUT}/bench_${tag}.jsonl" "$@" > "${OUT}/bench_${tag}.log" 2>&1
  echo "==> bench ${tag} exit $? $(date -Is)" | tee -a "${OUT}/job.log"
}

# GPU1-3: compile benchmarks, started first (the longest)
bench 1 8-15 compile_fp32 --modes compile --precisions fp32 &
P1=$!
bench 2 16-23 compile_amp_xbf16 --modes compile --precisions amp_xbf16 &
P2=$!
bench 3 24-31 compile_amp_xfp32_nbbf16 --modes compile --precisions amp_xfp32_nbbf16 &
P3=$!

# GPU0: tests, probe, eager benchmark
(
  cd "${BENCH_REPO}" || exit 1
  CUDA_VISIBLE_DEVICES=0 timeout -k 30 1200 taskset -c 0-7 python -X faulthandler -m pytest -v -s -p no:cacheprovider -rfEs \
    -o faulthandler_timeout=600 --junitxml="${OUT}/pytest.xml" \
    tests/models/test_gmp_weights_first.py tests/models/test_a4_placement.py > "${OUT}/pytest.log" 2>&1
  echo "==> pytest exit $? $(date -Is)" | tee -a "${OUT}/job.log"
  CUDA_VISIBLE_DEVICES=0 timeout -k 10 300 taskset -c 0-7 python -m hepattn.experiments.clic.a4.bench_gmp_backends \
    --probe-dtypes --out "${OUT}/dtype_probe.json" > "${OUT}/dtype_probe.log" 2>&1
  echo "==> dtype probe exit $? $(date -Is)" | tee -a "${OUT}/job.log"
  bench 0 0-7 eager --modes eager
  bench 0 0-7 compile_amp_xfp32 --modes compile --precisions amp_xfp32
)
wait ${P1} ${P2} ${P3}

python -m hepattn.experiments.clic.a4.bench_gmp_backends --summarize "${OUT}"/bench_*.jsonl > "${OUT}/summary.md" 2>&1
echo "==> done $(date -Is); results in ${OUT}" | tee -a "${OUT}/job.log"
