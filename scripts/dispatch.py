"""Run a queue of training jobs on a set of GPUs, packing them by memory.

    python scripts/dispatch.py --queue logs/queue.yaml --gpus 0 1 2 3

Queue file: a YAML list of jobs, highest priority first:
    - {name: <run name>, config: config/..yaml, set: ["train.seed=0", ...], save_features: [oof, val, test], mem_gb: 22}

Scheduling rules:
- A job starts on an allowed GPU only if free memory >= mem_gb + margin and that GPU runs fewer than --max-per-gpu of
  our jobs; among such GPUs the one with the most free memory wins. Pending jobs are scanned in priority order and the
  first one that fits starts (smaller jobs fill gaps).
- Every running job keeps (mem_gb - memory its process holds now) reserved, so a job's later peak (validation,
  checkpointing) is never handed to another job. One job starts at a time: the next start waits until the new process
  holds 60% of its mem_gb (or --settle seconds pass).
- Training processes of queued jobs that are already running (e.g. after a dispatcher restart) are adopted, not restarted.
- A run whose directory already has metrics.json is skipped; one with last.pt but no metrics.json is resumed.
- Jobs run in their own session (they survive this process) with stdout/stderr in logs/<name>.log.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def gpu_free_mb(gpus):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    free = {int(i): int(f) for i, f in (line.split(",") for line in out.strip().splitlines())}
    return {g: free[g] for g in gpus}


def proc_mem_mb():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    mem = {}
    for line in out.strip().splitlines():
        try:
            pid, m = (int(x) for x in line.split(","))
            mem[pid] = mem.get(pid, 0) + m
        except ValueError:
            pass
    return mem


class Adopted:
    """A training process started before this dispatcher (found with ps); poll() like subprocess.Popen."""

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            return 0


def running_training(names):
    """{run name: (pid, gpu)} for live train_baseline processes of queued jobs."""
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        tok = cmd.split()
        if "src.model.train_baseline" in tok and "--run-name" in tok and "--gpu" in tok:
            name = tok[tok.index("--run-name") + 1]
            if name in names:
                found[name] = (int(pid), int(tok[tok.index("--gpu") + 1]))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--margin-gb", type=float, default=2.0)
    ap.add_argument("--max-per-gpu", type=int, default=4)
    ap.add_argument("--settle", type=int, default=240, help="max seconds a new job's memory stays reserved")
    ap.add_argument("--poll", type=int, default=20)
    args = ap.parse_args()

    with open(args.queue) as f:
        jobs = yaml.safe_load(f)
    with open(os.path.join(PROJECT, "config", "base.yaml")) as f:
        runs = os.path.join(PROJECT, yaml.safe_load(f)["paths"]["runs"])
    env = dict(os.environ, PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1")

    def run_dir(j):  # honours a paths.runs=... override in the job's --set list
        root = next((x.split("=", 1)[1] for x in j.get("set", []) if x.startswith("paths.runs=")), runs)
        return os.path.join(PROJECT, root, j["name"])

    pending = []
    for j in jobs:
        if os.path.exists(os.path.join(run_dir(j), "metrics.json")):
            log(f"skip {j['name']} (finished)")
        else:
            pending.append(j)
    running = {}  # name -> dict(proc, gpu, mem_mb, started, settled, dir)
    failed = []
    for name, (pid, gpu) in running_training({j["name"] for j in pending}).items():
        j = next(x for x in pending if x["name"] == name)
        running[name] = {"proc": Adopted(pid), "gpu": gpu, "mem_mb": j["mem_gb"] * 1024, "started": time.time(),
                         "settled": True, "dir": run_dir(j)}
        pending.remove(j)
        log(f"adopt {name} (pid {pid}, GPU {gpu}, already running)")
    log(f"{len(pending)} jobs to run on GPUs {args.gpus}")

    while pending or running:
        for name, r in list(running.items()):
            code = r["proc"].poll()
            if code is not None:
                done = os.path.exists(os.path.join(r["dir"], "metrics.json"))
                log(f"{'done' if done and code == 0 else 'FAILED'} {name} on GPU {r['gpu']} (exit {code}, "
                    f"{(time.time() - r['started']) / 60:.1f} min)")
                if not (done and code == 0):
                    failed.append(name)
                del running[name]
        held = proc_mem_mb()
        for r in running.values():
            if not r["settled"] and (held.get(r["proc"].pid, 0) >= 0.6 * r["mem_mb"] or time.time() - r["started"] > args.settle):
                r["settled"] = True
        free = gpu_free_mb(args.gpus)
        for r in running.values():  # keep each job's declared peak reserved
            free[r["gpu"]] -= max(0, r["mem_mb"] - held.get(r["proc"].pid, 0))
        count = {g: sum(r["gpu"] == g for r in running.values()) for g in args.gpus}

        launched = False
        if not any(not r["settled"] for r in running.values()):  # one start at a time
            for j in pending:
                need = j["mem_gb"] * 1024 + args.margin_gb * 1024
                fits = [g for g in args.gpus if free[g] >= need and count[g] < args.max_per_gpu]
                if not fits:
                    continue
                g = max(fits, key=lambda x: free[x])
                cmd = [sys.executable, "-u", "-m", "src.model.train_baseline", "--config", j["config"], "--gpu", str(g),
                       "--run-name", j["name"], "--set", *j.get("set", [])]
                if j.get("save_features"):
                    cmd += ["--save-features", *j["save_features"]]
                if os.path.exists(os.path.join(run_dir(j), "last.pt")):
                    cmd.append("--resume")
                out = open(os.path.join(PROJECT, "logs", f"{j['name']}.log"), "a")
                proc = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=out, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                running[j["name"]] = {"proc": proc, "gpu": g, "mem_mb": j["mem_gb"] * 1024, "started": time.time(),
                                      "settled": False, "dir": run_dir(j)}
                pending.remove(j)
                log(f"start {j['name']} on GPU {g} (free {free[g] / 1024:.1f} GB, needs {j['mem_gb']} GB; "
                    f"{len(pending)} pending, {len(running)} running){' [resume]' if '--resume' in cmd else ''}")
                launched = True
                break
        time.sleep(5 if launched else args.poll)

    log(f"all jobs finished; failed: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
