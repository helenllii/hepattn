#!/bin/bash
# Polaris (ALCF, PBS Pro): short training-throughput benchmark of the full-size dim-256
# PHAT16 + GMP (EinsumOpt) model. Measures speed only -- it trains nothing to completion and
# writes no checkpoints.
#
#   qsub -A <project> submit_bench_polaris.sh
#   qsub -A <project> -v PHASES="A C" submit_bench_polaris.sh      # a subset
#   qsub -A <project> -v CLIC_DATA=/eagle/<project>/clic submit_bench_polaris.sh
#
# WHAT IT ANSWERS. Falcon measurements say this model is HOST-bound, not GPU-bound, and Polaris
# gives 8 cores per GPU where Falcon gave a single-GPU job all 32. So "is Polaris faster" splits
# into two questions that one run cannot separate, and the phases below separate them:
#
#   A  1 GPU,  8 workers   per-GPU rate at a node's per-GPU core share
#   B  1 GPU, 32 workers   per-GPU rate with the whole socket     -> B vs A = cost of 8 cores
#   C  4 GPU,  8 workers   whole node at FIXED global batch       -> C vs A = real scale-out
#   D  1 GPU, 32 workers, persistent_workers=false                -> B vs D = cost of re-forking
#   E  1 GPU, 32 workers, matcher device_solver=jv                -> B vs E = GPU matcher payoff
#
# EFFECTIVE GLOBAL BATCH IS HELD AT 2048 IN EVERY PHASE. accumulate_grad_batches is divided by
# the device count (1 GPU -> 16, 4 GPU -> 4); the benchmark callback's expect_global_batch makes
# a mismatch abort rather than quietly benchmark a different optimization problem.
#
#PBS -N a4bench-d256
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -j oe
# -A (project) is NOT set here on purpose: pass it at submit time so this file carries no
# site-specific allocation. qsub rejects the job immediately if it is missing.

set -uo pipefail

cd "${PBS_O_WORKDIR:-$(dirname "$0")}" || exit 1

# ---------------------------------------------------------------- knobs, all overridable by -v
PHASES="${PHASES:-A B C D}"
CLIC_DATA="${CLIC_DATA:-/projects/u5ar/data/clic}"
PY="${PY:-python}"
ENV_SETUP="${ENV_SETUP:-}"          # optional file to source (module load / conda activate)
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CORES="${CORES:-32}"
# The warmup/measure window lives in CFG_MODEL's ThroughputBenchmark init_args: it cannot be
# overridden from the command line (jsonargparse addresses callbacks by class, not by index), so
# edit it there rather than expecting a flag here.
OUT="${OUT:-bench_polaris_${PBS_JOBID:-local}}"

CFG_MODEL=configs/a4/bench_d256_einsumopt_early.yaml
CFG_HOST=configs/a4/polaris.yaml

mkdir -p "$OUT"
: > "$OUT/results.jsonl"   # one line per phase, appended by run_phase

# Lightning's DDP subprocess launcher re-executes this python process, so anything that must be
# true in every rank belongs in the environment, not in the shell below.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# The matcher is a THREAD pool (base.yaml: parallel_solver true, n_jobs 16) and torch opens its
# own. With 4 ranks on 32 cores the defaults oversubscribe the socket badly, which would be
# charged to "4 GPUs are slow" rather than to thread contention. One thread per rank for BLAS;
# the matcher's n_jobs is set per phase below.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[ -n "$ENV_SETUP" ] && { echo "sourcing $ENV_SETUP"; # shellcheck disable=SC1090
  source "$ENV_SETUP"; }

echo "================================================================================"
echo "A4 d256 PHAT+GMP throughput benchmark"
echo "  job        : ${PBS_JOBID:-<interactive>}  on $(hostname)  $(date -Is)"
echo "  phases     : $PHASES"
echo "  data       : $CLIC_DATA"
echo "  python     : $PY  ($($PY -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1 | tail -1))"
echo "  out        : $(pwd)/$OUT"
echo "  git        : $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null)"
echo "================================================================================"
nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv || true
echo "  cores seen : $($PY -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null)"
echo

for f in "$CFG_MODEL" "$CFG_HOST" main.py; do
  [ -f "$f" ] || { echo "FATAL: $f not found -- run this from src/hepattn/experiments/clic"; exit 1; }
done
for f in train_clic_fix.root val_clic_fix.root; do
  [ -f "$CLIC_DATA/$f" ] || { echo "FATAL: $CLIC_DATA/$f not found -- pass -v CLIC_DATA=<dir>"; exit 1; }
done

# ------------------------------------------------------------------------------ one phase
run_phase () {
  local tag="$1" devices="$2" workers="$3" accum="$4"; shift 4
  local log="$OUT/phase_${tag}.log"
  echo "--------------------------------------------------------------------------------"
  echo ">> PHASE $tag : ${devices} GPU, ${workers} workers, accumulate ${accum} (global batch $((128 * devices * accum)))"
  echo "   extra: $*"
  echo "   log  : $log"
  local vis=""
  if [ "$devices" -lt "$GPUS_PER_NODE" ]; then
    vis=$(seq -s, 0 $((devices - 1)))
  fi
  # The matcher's thread pool is per rank, so its n_jobs must be the per-rank core share or the
  # ranks fight over the socket.
  local njobs=$(( CORES / devices ))
  (
    # Empty $vis means "use every GPU on the node": leave CUDA_VISIBLE_DEVICES alone.
    if [ -n "$vis" ]; then export CUDA_VISIBLE_DEVICES="$vis"; fi
    "$PY" main.py fit \
      -c "$CFG_MODEL" -c "$CFG_HOST" \
      --data.train_path "$CLIC_DATA/train_clic_fix.root" \
      --data.valid_path "$CLIC_DATA/val_clic_fix.root" \
      --trainer.devices "$devices" \
      --trainer.accumulate_grad_batches "$accum" \
      --data.num_workers "$workers" \
      --model.model.init_args.matcher.init_args.n_jobs "$njobs" \
      --trainer.callbacks+=hepattn.callbacks.DynamoOptimizeDDP \
      "$@" 2>&1 ) | tee "$log"
  local rc=${PIPESTATUS[0]}
  if ! grep -q "^A4BENCH_JSON " "$log"; then
    echo "   !! PHASE $tag produced no A4BENCH_JSON line (exit $rc) -- see $log"
    echo
    return
  fi
  # The callback's own `tag` comes from the config and is the same in every phase, so the phase
  # letter is injected here, from the only place that knows it.
  grep -h "^A4BENCH_JSON " "$log" | sed 's/^A4BENCH_JSON //' \
    | "$PY" -c 'import json,sys
for line in sys.stdin:
    d = json.loads(line); d["phase"] = sys.argv[1]; print(json.dumps(d, sort_keys=True))' "$tag" >> "$OUT/results.jsonl"
  echo
}

# DynamoOptimizeDDP is appended in every phase, not only the multi-GPU one: with torch 2.9,
# dynamo's default DDP graph split fails to compile the patch encoder ("'int' object has no
# attribute 'meta'"). optimize_ddp=False compiles whole graphs and all-reduces after the
# backward -- same numerics, no overlap. It is a no-op on one GPU, so every phase runs the
# identical callback list and the comparison stays clean.

case " $PHASES " in *" A "*)
  run_phase A 1 8  16 ;; esac
case " $PHASES " in *" B "*)
  run_phase B 1 "$CORES" 16 ;; esac
case " $PHASES " in *" C "*)
  run_phase C "$GPUS_PER_NODE" $(( CORES / GPUS_PER_NODE )) $(( 16 / GPUS_PER_NODE )) ;; esac
case " $PHASES " in *" D "*)
  run_phase D 1 "$CORES" 16 --data.persistent_workers false ;; esac
case " $PHASES " in *" E "*)
  # Needs the torch-linear-assignment build: setup/build_torch_linear_assignment.sh
  run_phase E 1 "$CORES" 16 --model.model.init_args.matcher.init_args.device_solver jv ;; esac

# --------------------------------------------------------------------------------- summary
echo "================================================================================"
echo "SUMMARY  ($(pwd)/$OUT)"
echo "================================================================================"
"$PY" - "$OUT/results.jsonl" <<'PYEOF'
import json, sys, pathlib

rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
if not rows:
    print("no results -- every phase failed; read the phase logs")
    raise SystemExit(1)

hdr = f"{'phase':>6} {'gpus':>5} {'wrk':>4} {'glob.batch':>11} {'ev/s':>9} {'it/s/rank':>10} {'epoch min':>10} {'peak GiB':>9} {'dl wait':>8} {'util':>6}"
print(hdr)
print("-" * len(hdr))
def num(value, fmt):
    """A number, or a right-aligned '--' when the callback could not measure it (null in JSON)."""
    return format(value, fmt) if value is not None else format("--", ">" + fmt.split(".")[0].lstrip(">"))


for r in rows:
    name = r.get("phase") or r["tag"]
    util = r.get("gpu_util_mean")
    wait = r["dataloader_wait_frac"]
    print(f"{name[:6]:>6} {r['world_size']:>5} {str(r['num_workers']):>4} {r['global_batch']:>11} "
          f"{num(r['events_s_global'], '>9.1f')} {num(r['it_s_per_rank'], '>10.3f')} {num(r['epoch_min'], '>10.1f')} "
          f"{num(r['peak_reserved_gib'], '>9.2f')} {num(None if wait is None else wait * 100, '>7.1f')}% "
          f"{('%.0f%%' % util) if util is not None else '--':>6}")

print()
gb = {r["global_batch"] for r in rows}
if len(gb) > 1:
    print(f"!! global batch is not constant across phases ({sorted(gb)}): the throughput numbers "
          f"are NOT comparable.")

one = [r for r in rows if r["world_size"] == 1]
many = [r for r in rows if r["world_size"] > 1]
if one and many:
    # Compare against the single-GPU phase at the SAME per-rank worker count, so the ratio is
    # scale-out and not a change in cores per rank.
    for m in many:
        same = [r for r in one if r["num_workers"] == m["num_workers"]]
        base = (same or one)[0]
        eff = m["events_s_global"] / (base["events_s_global"] * m["world_size"])
        note = "" if same else "  (no 1-GPU phase at the same num_workers: conflates cores/rank)"
        print(f"scale-out {base['world_size']}->{m['world_size']} GPU at {m['num_workers']} workers/rank: "
              f"{m['events_s_global'] / base['events_s_global']:.2f}x throughput, "
              f"{eff * 100:.0f}% of linear{note}")
waits = [r["dataloader_wait_frac"] for r in rows if r["dataloader_wait_frac"] is not None]
if waits:
    worst = max(waits)
    print(f"worst dataloader starvation across phases: {worst * 100:.1f}%"
          + ("  -> HOST-BOUND: fix the input pipeline before buying more GPUs" if worst > 0.25 else ""))
mems = [r["peak_reserved_gib"] for r in rows if r["peak_reserved_gib"] is not None]
if mems:
    print(f"peak reserved GPU memory across phases: {max(mems):.2f} GiB (A100-40GB usable ~39.5 GiB)")
PYEOF
echo
echo "done $(date -Is)"
