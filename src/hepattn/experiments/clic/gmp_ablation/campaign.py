"""Sequential GMP ablation campaigns: one 4-GPU DDP job at a time, chained with PBS `afterok`, submitted once.

A campaign spec (configs/gmp_ablation/campaigns/<name>.yaml) lists runs (arm x seed) in their execution order:

    name: ablation_v1
    epochs: 10              # every run's budget = trainer.max_epochs (the LR schedule spans it)
    queue: null             # REQUIRED to submit; deliberately unset until the user chooses the queue
    walltime: null          # per job, "HH:MM:SS"; REQUIRED to submit
    epochs_per_job: null    # null: one job per run; N: split each run into checkpointed jobs of <= N epochs
    runs:
      - {arm: g2d_e020, seed: 42}
      - ...

    python campaign.py plan   <spec>                 # the job list, order, dirs, time estimate; submits nothing
    python campaign.py submit <spec> [--dry-run]     # pin the code, submit the whole chain once, write the manifest
    python campaign.py submit <spec> --resume        # after a failure: re-chain from the first unfinished job
    python campaign.py status <spec>                 # completed / running / queued / waiting / FAILED / cancelled

Chain semantics (verified on Polaris with trivial jobs, 2026-11-14, jobs 7848490-7848495): job k is submitted with
`-W depend=afterok:<job k-1>`; it stays in state H until k-1 exits 0, then runs. If k-1 exits non-zero, PBS deletes k
without running it (state F, no start time, no exit status), and in turn every later job. So a failed run stops the
campaign; nothing is skipped. Each job's exit status is gmp_ablation/runcheck.py's verdict (submit_gmp_run.sh).

Reproducibility and safety:
  * the code is pinned: `submit` creates a detached git worktree of the current commit (clean tree required) under
    patch-attention-env/code_snapshots/, and every job checks it is at that commit before training;
  * every run and seed has its own directory; a run with DONE.json is never retrained, and a job never starts on a
    directory another live job owns;
  * duplicate protection: `submit` refuses if the campaign already has a manifest (use --resume, which itself refuses
    while any job of the campaign is still queued, held or running).
Stdlib + PyYAML only (no torch / hepattn imports): run it by path on a login node.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import yaml

ENVROOT = Path("/lus/eagle/projects/hgcal-maskformer-fpga/siyingli/patch-attention-env")
CAMPAIGNS = ENVROOT / "runs" / "gmp_ablation" / "campaigns"
SNAPSHOTS = ENVROOT / "code_snapshots"
CLIC_REL = Path("src/hepattn/experiments/clic")

# Time model for `plan` (minutes), from measurements; see gmp_ablation/CAMPAIGN.md for the derivation.
TRAIN_MIN_PER_EPOCH = {"2d": 5.64, "3d": 6.13, "nogmp": 5.74}  # bench3 4-GPU DDP, jobs 7841693 / 7841964 / 7835439
VAL_MIN_PER_EPOCH = 0.5  # steady validation of 4959 events at 4 GPUs (estimate; the smoke test measures it)
JOB_OVERHEAD_MIN = 9.0  # environment + dataset init (~2) + first-batch compile (~3.5-4.4) + first validation compile (~1.5)


# ------------------------------------------------------------------------------------------------------------- plan
def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    for key in ("name", "epochs", "runs"):
        if key not in spec:
            raise ValueError(f"{path}: missing '{key}'")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec["name"]):
        raise ValueError(f"campaign name {spec['name']!r}: letters, digits, _ . - only")
    seen = set()
    for r in spec["runs"]:
        key = (r["arm"], int(r["seed"]))
        if key in seen:
            raise ValueError(f"run {key} listed twice")
        seen.add(key)
    return spec


def arm_kind(arm: str) -> str:
    return "nogmp" if arm == "nogmp" else ("3d" if arm.startswith("g3d") else "2d")


def plan_jobs(spec: dict) -> list[dict]:
    """The ordered job list: each run is one job, or ceil(epochs / epochs_per_job) segments."""
    epochs = int(spec["epochs"])
    per = spec.get("epochs_per_job") or epochs
    jobs = []
    for idx, run in enumerate(spec["runs"]):
        arm, seed = run["arm"], int(run["seed"])
        run_id = f"{idx:02d}_{arm}_s{seed}"
        bounds = [*range(per, epochs, per), epochs]
        for k, stop in enumerate(bounds):
            jobs.append({
                "run_id": run_id,
                "arm": arm,
                "seed": seed,
                "epochs": epochs,
                "segment": k,
                "segments": len(bounds),
                "stop_at_epoch": stop if len(bounds) > 1 else None,
                "job_name": f"gmp{idx:02d}{'' if len(bounds) == 1 else f's{k}'}_{arm}"[:15],
            })
    return jobs


def estimate_minutes(job: dict, prev_stop: int) -> float:
    n = (job["stop_at_epoch"] or job["epochs"]) - prev_stop
    return JOB_OVERHEAD_MIN + n * (TRAIN_MIN_PER_EPOCH[arm_kind(job["arm"])] + VAL_MIN_PER_EPOCH)


def plan_text(spec: dict) -> str:
    jobs = plan_jobs(spec)
    lines = [
        f"campaign {spec['name']}: {len(spec['runs'])} runs, {len(jobs)} jobs, {spec['epochs']} epochs per run, "
        f"queue {spec.get('queue') or 'UNSET'}, walltime {spec.get('walltime') or 'UNSET'}",
        "  #   job name         run                          epochs     est. min",
    ]
    total, prev = 0.0, {}
    for i, j in enumerate(jobs):
        start = prev.get(j["run_id"], 0)
        est = estimate_minutes(j, start)
        total += est
        stop = j["stop_at_epoch"] or j["epochs"]
        prev[j["run_id"]] = stop
        lines.append(f"  {i:<3d} {j['job_name']:<16s} {j['run_id']:<28s} {start:>3d}-{stop - 1:<3d}   {est:7.1f}")
    lines.append(f"  total {total / 60:.1f} h of node time (= {4 * total / 60:.1f} GPU-h), excluding queue waits between jobs")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- submission
def run(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout.strip()


def qsub_command(job: dict, spec: dict, code: Path, commit: str, run_dir: Path, pbs_dir: Path, after: str | None) -> list[str]:
    env = {
        "CODE": str(code),
        "EXPECTED_COMMIT": commit,
        "RUN_DIR": str(run_dir),
        "ARM": job["arm"],
        "SEED": str(job["seed"]),
        "EPOCHS": str(job["epochs"]),
        "CAMPAIGN": spec["name"],
    }
    if job["stop_at_epoch"]:
        env["STOP_AT_EPOCH"] = str(job["stop_at_epoch"])
    for k, v in env.items():
        if "," in v:
            raise ValueError(f"{k}={v!r}: qsub -v cannot carry commas")
    cmd = [
        "qsub",
        "-q",
        spec["queue"],
        "-l",
        f"walltime={spec['walltime']}",
        "-N",
        job["job_name"],
        "-o",
        f"{pbs_dir}/",
        "-v",
        ",".join(f"{k}={v}" for k, v in env.items()),
    ]
    if after:
        cmd += ["-W", f"depend=afterok:{after}"]
    return [*cmd, str(code / CLIC_REL / "submit_gmp_run.sh")]


def pin_code(repo: Path, name: str) -> tuple[Path, str]:
    commit = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"]):
        raise SystemExit(f"refusing: {repo} has uncommitted changes to tracked files; commit them first")
    snap = SNAPSHOTS / f"{name}-{commit[:12]}"
    if snap.exists():
        if run(["git", "-C", str(snap), "rev-parse", "HEAD"]) != commit:
            raise SystemExit(f"refusing: {snap} exists at another commit")
    else:
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        run(["git", "-C", str(repo), "worktree", "add", "--detach", str(snap), commit])
    return snap, commit


def live_jobs(job_ids: list[str]) -> list[str]:
    live = []
    for jid in job_ids:
        state = qstat_state(jid).get("job_state")
        if state in {"Q", "H", "R", "E", "W", "T", "B"}:
            live.append(f"{jid} ({state})")
    return live


def submit(spec_path: Path, repo: Path, dry_run: bool, resume: bool) -> None:
    spec = load_spec(spec_path)
    if not dry_run and (not spec.get("queue") or not spec.get("walltime")):
        raise SystemExit("refusing: the spec's queue and walltime are unset; choose them first (see CAMPAIGN.md)")
    spec.setdefault("queue", "<QUEUE>")
    spec.setdefault("walltime", "<HH:MM:SS>")
    spec["queue"] = spec["queue"] or "<QUEUE>"
    spec["walltime"] = spec["walltime"] or "<HH:MM:SS>"
    cdir = CAMPAIGNS / spec["name"]
    manifest_path = cdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and not resume:
        raise SystemExit(f"refusing: {manifest_path} exists (already submitted). Use `status`, or `submit --resume` after a failure")
    if resume and not manifest:
        raise SystemExit("refusing --resume: no manifest; submit first")
    if manifest:
        live = live_jobs([a["job_id"] for j in manifest["jobs"] for a in j["attempts"]])
        if live:
            raise SystemExit(f"refusing --resume: jobs still queued/held/running: {live}")
    if dry_run:
        code, commit = (repo, run(["git", "-C", str(repo), "rev-parse", "HEAD"]))
    else:
        code, commit = pin_code(repo, spec["name"])
        if manifest and manifest["commit"] != commit:
            raise SystemExit(f"refusing --resume: the manifest is at {manifest['commit']}, the repo at {commit}")
    jobs = plan_jobs(spec)
    if manifest is None:
        manifest = {
            "campaign": spec["name"],
            "spec": spec,
            "commit": commit,
            "code": str(code),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dependency": "afterok (PBS)",
            "jobs": [j | {"run_dir": str(cdir / "runs" / j["run_id"]), "attempts": []} for j in jobs],
        }
    pbs_dir = cdir / "pbs"
    if not dry_run:
        pbs_dir.mkdir(parents=True, exist_ok=True)
        (cdir / "runs").mkdir(exist_ok=True)
        (cdir / "spec.yaml").write_text(spec_path.read_text())
    prev = None
    for entry in manifest["jobs"]:
        run_dir = Path(entry["run_dir"])
        if is_finished(entry, run_dir):
            print(f"  {entry['job_name']:<16s} {entry['run_id']}: already finished, not resubmitted")
            continue
        cmd = qsub_command(entry, spec, code, commit, run_dir, pbs_dir, prev)
        if dry_run:
            print("  " + " ".join(cmd))
            prev = f"<job of {entry['job_name']}>"
            continue
        jid = run(cmd)
        entry["attempts"].append({"job_id": jid, "after": prev, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": cmd})
        manifest_path.write_text(json.dumps(manifest, indent=1))  # after every qsub: never lose a job id
        print(f"  {entry['job_name']:<16s} {entry['run_id']:<28s} {jid}{f'  afterok:{prev}' if prev else ''}")
        prev = jid
    if not dry_run:
        print(f"manifest: {manifest_path}")


def is_finished(entry: dict, run_dir: Path) -> bool:
    if (run_dir / "DONE.json").exists():
        return True
    return bool(entry.get("stop_at_epoch")) and (run_dir / "segments" / f"upto_{entry['stop_at_epoch']}.json").exists()


# ----------------------------------------------------------------------------------------------------------- status
def qstat_state(job_id: str) -> dict:
    try:
        out = subprocess.run(["qstat", "-x", "-f", job_id], capture_output=True, text=True, timeout=60).stdout  # noqa: S607
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return parse_qstat(out)


def parse_qstat(text: str) -> dict:
    keep = {"job_state", "Exit_status", "stime", "substate", "depend", "exec_host", "comment"}
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s+([\w.]+) = (.*)", line)
        if m and m.group(1) in keep:
            out[m.group(1)] = m.group(2).strip()
    return out


def classify(state: dict, finished: bool) -> str:
    js = state.get("job_state")
    if js in {"R", "E"}:
        return "running"
    if js == "Q":
        return "queued"
    if js in {"H", "W"}:
        return "waiting (held on dependency)" if state.get("depend") else "held"
    if js == "F":
        if "Exit_status" in state:
            return "completed" if state["Exit_status"] == "0" and finished else f"FAILED (exit {state['Exit_status']})"
        return "cancelled (never ran: upstream failed or deleted)" if not state.get("stime") else "ended (no exit status)"
    if finished:
        return "completed"
    return js or "unknown (no PBS record)"


def status(spec_path: Path) -> None:
    spec = load_spec(spec_path)
    manifest_path = CAMPAIGNS / spec["name"] / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"not submitted: no {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    print(f"campaign {manifest['campaign']} @ {manifest['commit'][:12]} ({manifest['code']})")
    current = None
    for entry in manifest["jobs"]:
        run_dir = Path(entry["run_dir"])
        att = entry["attempts"][-1] if entry["attempts"] else None
        state = qstat_state(att["job_id"]) if att else {}
        label = classify(state, is_finished(entry, run_dir)) if att else ("completed" if is_finished(entry, run_dir) else "not submitted")
        if label == "running":
            current = entry
        jid = att["job_id"].split(".")[0] if att else "-"
        print(f"  {entry['job_name']:<16s} {entry['run_id']:<28s} {jid:<9s} {label}  {run_dir}")
    print(f"currently running: {current['run_id'] if current else 'none'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "submit", "status"):
        s = sub.add_parser(name)
        s.add_argument("spec", type=Path)
        if name == "submit":
            s.add_argument("--dry-run", action="store_true", help="print the qsub commands; submit nothing")
            s.add_argument("--resume", action="store_true", help="re-chain from the first unfinished job after a failure")
            s.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[5])
    args = p.parse_args()
    if args.cmd == "plan":
        print(plan_text(load_spec(args.spec)))
    elif args.cmd == "submit":
        submit(args.spec, args.repo, args.dry_run, args.resume)
    else:
        status(args.spec)


if __name__ == "__main__":
    main()
