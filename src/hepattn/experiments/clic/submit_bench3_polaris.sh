#!/bin/bash -l
# bench3: controlled training-time comparison on ONE Polaris node (4x A100-40GB). One job per step, all debug queue.
#
#   Round 0 (GPU preflight; decides the common micro-batch; no throughput timing):
#     qsub -v ROUND0=1,ARMS=linformer_encoder:phat16:phat_gmp_einsumopt_early submit_bench3_polaris.sh
#   Round 1/2 benchmark, one arm per job (any configs/bench3/<ARM>.yaml):
#     qsub -v ARM=linformer_encoder submit_bench3_polaris.sh
#     qsub -v ARM=phat16,PROFILE=1 submit_bench3_polaris.sh        # + step breakdown and profiler after the window
#   Round 3 precondition (GMP impl equivalence at fixed placement):
#     qsub -v ROUND3_EQ=1 submit_bench3_polaris.sh
#   Table of everything finished so far:
#     python -m hepattn.experiments.clic.bench3.summarize $OUT/results            (and --round0 $OUT/round0_<job>)
#
# Optional -v knobs (identical defaults for every arm -- change them for all arms or for none):
#   GLOBAL_BATCH=1024 MICRO_BATCH=256   accumulate = GLOBAL_BATCH / (MICRO_BATCH * 4); 1024 = production effective batch.
#                                       If Round 0 reports OOM for ANY arm, set e.g. MICRO_BATCH=128 for ALL arms.
#   NUM_TRAIN=-1       full training set (exact batches/epoch, real dataset init and host RAM). >0 = faster debug.
#   COMPILE=1          torch.compile (production). 0 = eager, for all arms.
#   PROFILE=0          N>0: N instrumented breakdown batches + 10 torch.profiler batches AFTER the measured window.
#   SMOKE=0            1 = also run the GPU smoke test in a benchmark job (Round 0 normally covers it).
#   DRY_RUN=1          preflight + resolved config only; works on a login node.
#   NUM_WORKERS=7 MATCHER_JOBS=8 OMP=8   per-rank loader workers / matcher threads / torch intra-op threads
#                                         (production values; BLAS threads are pinned to 1 for every arm).
#   MARGIN_S=300       stop training this long before the walltime ends, to save results and print the summary.
# NEVER put a trailing comment on a #PBS line: qsub parses the rest of the line as arguments.
#PBS -N bench3
#PBS -A hgcal-maskformer-fpga
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe

set -uo pipefail
ulimit -c 0
JOB_START=${JOB_START:-$(date +%s)}  # a wrapper running several arms in one PBS job passes the real start

# ---------------------------------------------------------------------------------------------------------- knobs
# exp/gmp-ablation defaults: this worktree's code, and results kept apart from the bench3 Round 1 results in
# runs/bench3 (which were produced from polaris/bench-d256 @ ea2ab7b + the bench3 snapshot f1a55e6). Both remain
# overridable: qsub -v BENCH_REPO=...,OUT=... submit_bench3_polaris.sh
BENCH_REPO=${BENCH_REPO:-/home/siyingli/hepattn-gmp-ablation}
ENV_REPO=${ENV_REPO:-/home/siyingli/hepattn-phat}
ENVROOT=/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env
DATA=${DATA:-/eagle/hgcal-maskformer-fpga/clic_data}
OUT=${OUT:-${ENVROOT}/runs/gmp_ablation}
ARM_DIR=${ARM_DIR:-configs/bench3}   # where ARM.yaml lives, relative to src/hepattn/experiments/clic
EXTRA_ARGS=${EXTRA_ARGS:-}
GPUS=4
GLOBAL_BATCH=${GLOBAL_BATCH:-1024}
MICRO_BATCH=${MICRO_BATCH:-256}
NUM_TRAIN=${NUM_TRAIN:--1}
COMPILE=${COMPILE:-1}
PROFILE=${PROFILE:-0}
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-0}
ROUND0=${ROUND0:-0}
ROUND3_EQ=${ROUND3_EQ:-0}
NUM_WORKERS=${NUM_WORKERS:-7}
MATCHER_JOBS=${MATCHER_JOBS:-8}
OMP=${OMP:-8}
MARGIN_S=${MARGIN_S:-300}
PIXI=${PIXI:-${HOME}/.pixi/bin/pixi}
CLIC=${BENCH_REPO}/src/hepattn/experiments/clic

if (( GLOBAL_BATCH % (MICRO_BATCH * GPUS) )); then
  echo "FATAL: GLOBAL_BATCH=${GLOBAL_BATCH} is not a multiple of MICRO_BATCH*GPUS=$((MICRO_BATCH * GPUS))"; exit 1
fi
ACCUM=$(( GLOBAL_BATCH / (MICRO_BATCH * GPUS) ))
JOB=${PBS_JOBID:-local}; JOB=${JOB%%.*}

if [ "${ROUND0}" = 1 ]; then MODE=round0; elif [ "${ROUND3_EQ}" = 1 ]; then MODE=round3_eq; else MODE=bench; fi
if [ "${MODE}" = bench ]; then
  ARM=${ARM:?qsub -v ARM=<name of a ${ARM_DIR}/*.yaml>}
  [ -f "${CLIC}/${ARM_DIR}/${ARM}.yaml" ] && [ "${ARM}" != common ] || { echo "FATAL: no ${ARM_DIR}/${ARM}.yaml"; exit 1; }
  TAGNAME=${ARM}
else
  TAGNAME=${MODE}
fi
mkdir -p "${OUT}/logs" "${OUT}/results"
LOG=${OUT}/logs/${TAGNAME}_${JOB}.log

# --------------------------------------------------------------------------------------------- walltime guard
# Deadline = job start + walltime - MARGIN_S, exported to every rank (the benchmark callback stops training by then).
WALL=$(qstat -f "${PBS_JOBID:-none}" 2>/dev/null | awk -F' = ' '/Resource_List.walltime/ {print $2}')
if [[ "${WALL}" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
  WALL_S=$(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
else
  WALL_S=3600
fi
export BENCH3_DEADLINE=$(( JOB_START + WALL_S - MARGIN_S ))
left () { echo $(( BENCH3_DEADLINE - $(date +%s) )); }

# ---------------------------------------------------------------------------------------------------- environment
export PIXI_CACHE_DIR=${ENVROOT}/pixi-cache
set +u   # conda activation scripts read unset variables
eval "$("${PIXI}" shell-hook --manifest-path "${ENV_REPO}/pyproject.toml" -e clic --as-is)" || { echo "FATAL: pixi shell-hook failed"; exit 1; }
set -u
# Inductor compiles with $CXX: use the env's compilers, not the login profile's nvc/nvc++ (as in production).
export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++
export PYTHONNOUSERSITE=1
export HEPATTN_SRC=${BENCH_REPO}/src
export PYTHONPATH=${CLIC}/bench3/shim${PYTHONPATH:+:${PYTHONPATH}}
# CPU: torch intra-op threads per rank = OMP; BLAS pools pinned to 1 for every arm (loader workers gain nothing from
# them and they add contention on 32 cores shared by 4 ranks x (workers + matcher pool)).
export OMP_NUM_THREADS=${OMP} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_LOGS=${TORCH_LOGS:-recompiles,graph_breaks}
export TORCHINDUCTOR_CACHE_DIR=${OUT}/inductor_cache/${TAGNAME}

{
echo "================================================================================================"
echo "bench3 mode=${MODE} ${ARM:+arm=${ARM}}  job=${PBS_JOBID:-<none>}  host=$(hostname)  $(date -Is)"
echo "  code    : ${BENCH_REPO}  $(git -C "${BENCH_REPO}" rev-parse --abbrev-ref HEAD) @ $(git -C "${BENCH_REPO}" rev-parse HEAD)"
echo "  env     : ${CONDA_PREFIX}  python $(command -v python)"
echo "  batch   : global ${GLOBAL_BATCH} = micro ${MICRO_BATCH} x ${GPUS} GPUs x accum ${ACCUM}"
echo "  data    : ${DATA}  num_train=${NUM_TRAIN}"
echo "  cpu     : $(nproc) visible cpus; workers/rank=${NUM_WORKERS} matcher n_jobs=${MATCHER_JOBS} OMP=${OMP} OPENBLAS/MKL=1"
echo "  compile : ${COMPILE}  profile: ${PROFILE}  walltime ${WALL_S} s, deadline in $(left) s (margin ${MARGIN_S} s)"
echo "================================================================================================"
git -C "${BENCH_REPO}" status --short
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi)"
lscpu 2>/dev/null | grep -E "^(Socket|Core|Thread|CPU\(s\)|Model name)" || true
} 2>&1 | tee "${LOG}"
git -C "${BENCH_REPO}" diff HEAD > "${OUT}/logs/${TAGNAME}_${JOB}.diff"

# Preflight: prove THIS worktree's hepattn is imported and every backend the arms need is present.
python - "${BENCH_REPO}" <<'PY' 2>&1 | tee -a "${LOG}"
import os, sys, torch
bench = os.path.realpath(sys.argv[1])
import hepattn, hepattn.experiments.clic.a4, hepattn.experiments.clic.bench3, lap1015, masked_linformer, flash_attn
import hepattn.models.attention as att, hepattn.models.encoder as enc
print(f"preflight: hepattn {os.path.realpath(hepattn.__file__)}")
print(f"preflight: encoder {enc.__file__}  attention {att.__file__}  a4 {hepattn.experiments.clic.a4.__file__}")
print(f"preflight: torch {torch.__version__} cuda {torch.version.cuda}, {torch.cuda.device_count()} GPU(s); flash_attn {flash_attn.__version__}; masked_linformer {masked_linformer.__file__}; lap1015 {lap1015.__file__}")
for f in (hepattn.__file__, enc.__file__, att.__file__, hepattn.experiments.clic.a4.__file__):
    if not os.path.realpath(f).startswith(bench + os.sep):
        sys.exit(f"FATAL: {f} is not under {bench}: the import shim is not active")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL: preflight failed" | tee -a "${LOG}"; exit 1; }

cd "${CLIC}" || exit 1

# ============================================================================================ Round 0: preflight
if [ "${MODE}" = round0 ]; then
  R0=${OUT}/round0_${JOB}; mkdir -p "${R0}"
  IFS=: read -r -a R0ARMS <<< "${ARMS:-linformer_encoder:phat16:phat_gmp_einsumopt_early}"
  rc=0
  for a in "${R0ARMS[@]}"; do
    if (( $(left) < 240 )); then echo "!! deadline: skipping remaining arms from ${a}" | tee -a "${LOG}"; rc=1; break; fi
    echo "==> [round0] smoke+memory ${a} at micro ${MICRO_BATCH} ($(left) s left)" | tee -a "${LOG}"
    CUDA_VISIBLE_DEVICES=0 timeout $(( $(left) + MARGIN_S / 2 )) python -m hepattn.experiments.clic.bench3.smoke --arm "${a}" \
      --data-dir "${DATA}" --batch-size "${MICRO_BATCH}" --device cuda --compiled-steps 3 --out "${R0}/smoke_${a}.json" 2>&1 | tee -a "${LOG}"
    s=${PIPESTATUS[0]}; [ "${s}" -eq 0 ] || rc=1
    [ "${s}" -eq 3 ] && echo "!! ${a} OOM at micro ${MICRO_BATCH}: rerun Round 0 with a smaller MICRO_BATCH for ALL arms" | tee -a "${LOG}"
    if (( $(left) < 180 )); then echo "!! deadline: skipping encoder bench for ${a}" | tee -a "${LOG}"; continue; fi
    echo "==> [round0] encoder-only ${a} ($(left) s left)" | tee -a "${LOG}"
    CUDA_VISIBLE_DEVICES=0 timeout $(( $(left) + MARGIN_S / 2 )) python -m hepattn.experiments.clic.bench3.encoder_bench --arm "${a}" \
      --data-dir "${DATA}" --batch-size "${MICRO_BATCH}" --out "${R0}/encoder_${a}.json" 2>&1 | tee -a "${LOG}"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
  if (( $(left) > 120 )); then
    echo "==> [round0] unit tests ($(left) s left)" | tee -a "${LOG}"
    (cd "${BENCH_REPO}" && timeout $(( $(left) + MARGIN_S / 2 )) python -m pytest -q -p no:cacheprovider tests/experiments/clic/test_bench3.py) 2>&1 | tee -a "${LOG}" "${R0}/tests.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  fi
  python -m hepattn.experiments.clic.bench3.summarize --round0 "${R0}" 2>&1 | tee -a "${LOG}"
  echo "==> round0 done rc=${rc} $(date -Is)" | tee -a "${LOG}"
  exit "${rc}"
fi

# ================================================================================= Round 3 precondition (GMP eq.)
if [ "${MODE}" = round3_eq ]; then
  rc=0
  for pl in early all; do
    CUDA_VISIBLE_DEVICES=0 timeout $(( $(left) + MARGIN_S / 2 )) python -m hepattn.experiments.clic.bench3.gmp_equivalence \
      --placement "${pl}" --data-dir "${DATA}" --out "${OUT}/results_eq/gmp_equivalence_${pl}_${JOB}.json" 2>&1 | tee -a "${LOG}"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
  exit "${rc}"
fi

# ======================================================================================= benchmark (one arm)
RES=${OUT}/results/${ARM}_${JOB}.json
export BENCH3_RESULT=${RES} BENCH3_PROFILE=${PROFILE}
ARGS=(-c configs/bench3/common.yaml -c "${ARM_DIR}/${ARM}.yaml" --name "bench3_${ARM}"
      --data.train_path "${DATA}/train_clic_fix.root" --data.valid_path "${DATA}/val_clic_fix.root"
      --data.test_path "${DATA}/test_clic_common_infer.root"
      --data.num_train "${NUM_TRAIN}" --data.num_workers "${NUM_WORKERS}" --data.batch_size "${MICRO_BATCH}"
      --trainer.devices "${GPUS}" --trainer.accumulate_grad_batches "${ACCUM}"
      --model.model.init_args.matcher.init_args.n_jobs "${MATCHER_JOBS}"
      --trainer.default_root_dir "${OUT}/runs"
      --trainer.logger.init_args.save_dir "${OUT}/runs"
      "--trainer.callbacks+=hepattn.callbacks.DynamoOptimizeDDP")
# DynamoOptimizeDDP: the production Polaris setting for every arm (torch 2.9's DDP graph split fails on the patch encoder).
[ "${COMPILE}" = 1 ] && ARGS+=("--trainer.callbacks+=hepattn.callbacks.Compile")
# Optional extra CLI arguments, whitespace-separated (e.g. a diagnostics callback); none by default.
[ -n "${EXTRA_ARGS}" ] && read -r -a _extra <<< "${EXTRA_ARGS}" && ARGS+=("${_extra[@]}")

python main.py fit "${ARGS[@]}" --print_config > "${OUT}/logs/${ARM}_${JOB}_resolved.yaml" 2>>"${LOG}" \
  || { echo "FATAL: config does not resolve; see ${LOG}"; exit 1; }
echo "resolved config: ${OUT}/logs/${ARM}_${JOB}_resolved.yaml" | tee -a "${LOG}"
echo "command: python main.py fit ${ARGS[*]}" | tee -a "${LOG}"
[ "${DRY_RUN}" = 1 ] && { echo "DRY_RUN=1: stopping before any GPU work"; exit 0; }

if [ "${SMOKE}" = 1 ]; then
  echo "==> smoke test $(date -Is)" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES=0 python -m hepattn.experiments.clic.bench3.smoke --arm "${ARM}" --data-dir "${DATA}" \
    --batch-size "${MICRO_BATCH}" --device cuda --out "${OUT}/logs/${ARM}_${JOB}_smoke.json" 2>&1 | tee -a "${LOG}"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL: smoke test failed; not benchmarking" | tee -a "${LOG}"; exit 1; }
fi

echo "==> benchmark $(date -Is), $(left) s to the training deadline" | tee -a "${LOG}"
# Hard stop shortly after the deadline (the callback normally stops well before), so the summary below still runs.
timeout --signal=TERM $(( $(left) + MARGIN_S / 2 )) python main.py fit "${ARGS[@]}" 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
echo "==> main.py exit ${rc} $(date -Is)" | tee -a "${LOG}"
if [ -s "${RES}" ]; then
  echo "result: ${RES} (status $(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "${RES}"))" | tee -a "${LOG}"
else
  echo "!! no result file ${RES}" | tee -a "${LOG}"; exit 1
fi
python -m hepattn.experiments.clic.bench3.summarize "${OUT}/results" 2>&1 | tee -a "${LOG}"
exit "${rc}"
