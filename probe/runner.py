"""Run the label probe with the benchmark's own reference trainer.

The benchmark (KSHS x AIM Lab, branch codex/maskedkd-reference-runs) is cloned
at a pinned commit into external/ and imported unchanged. Each run happens in
its own worker process that swaps in only what the probe needs:

- where the run is stored: outputs/<dataset>/<arm>/seed_<s> (teacher stays in outputs/<dataset>/teacher)
- keep_patches: 98 (MaskedKD) or 196 (full teacher; top-196 is every patch)
- CE off: the student's cross-entropy term is replaced by zero inside the
  reference engine, so the loss becomes kd_alpha * KD. The KD weight is NOT
  changed (setting kd_alpha=1 instead would double it).

Everything else (data pipeline, optimizer, schedule, validation-selected
checkpoint, test evaluation, mask diagnostics, result.json) is the reference code.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
BENCH = HOME / "external" / "kshs-aimlab-benchmarks"
BENCH_COMMIT = "e1c39e7e1cc48e57305a2c12b0302e1df3349396"

# arm -> teacher tokens kept, whether ground-truth CE is in the student loss
ARMS = {
    "mask98_ce1": {"keep_patches": 98, "ce": True},   # identical to the reference MaskedKD run
    "mask98_ce0": {"keep_patches": 98, "ce": False},
    "full_ce1": {"keep_patches": 196, "ce": True},
    "full_ce0": {"keep_patches": 196, "ce": False},
    "mask59_ce1": {"keep_patches": 59, "ce": True},
    "mask59_ce0": {"keep_patches": 59, "ce": False},
}
DEFAULT_ARMS = ["full_ce1", "mask98_ce1", "full_ce0", "mask98_ce0"]
STUDENT_PEAK_GIB = 1.5  # reference runs measured 1.24 GiB reserved (batch 32); rounded up
TEACHER_PEAK_GIB = 2.5  # reference teacher measured 2.11 GiB


def use_benchmark():
    if not (BENCH / "reference" / "engine.py").is_file():
        sys.exit("benchmark missing: run ./run.sh setup first")
    head = subprocess.run(["git", "-C", str(BENCH), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head and head != BENCH_COMMIT:
        sys.exit(f"benchmark is at {head[:7]}, expected {BENCH_COMMIT[:7]}: run ./run.sh setup")
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))


def load_config(override=None):
    cfg = json.loads((BENCH / "configs" / "reference.json").read_text())
    if override:
        cfg.update(json.loads(Path(override).read_text()))
    return cfg


def default_data_root(dataset):
    return BENCH / "data" / ("coco_single" if dataset == "coco" else "waterbird_complete95_forest2water2")


def run_dir(output_root, dataset, role, seed, arm=None):
    base = Path(output_root) / dataset
    return base / "teacher" if role == "teacher" else base / arm / f"seed_{seed}"


# ----------------------------------------------------------------- worker
def worker(a):
    use_benchmark()
    import torch
    import reference.engine as engine

    arm = a.arm
    engine.run_directory = lambda output_root, dataset, role, seed: run_dir(output_root, dataset, role, seed, arm)
    cfg = load_config(a.config)
    if a.role == "student":
        spec = ARMS[arm]
        cfg = {**cfg, "keep_patches": spec["keep_patches"], "probe_arm": arm, "ce_in_loss": spec["ce"]}
        if not spec["ce"]:
            class NoCE:
                """torch.nn.functional with cross_entropy returning 0 (CE removed from the loss)."""
                def __getattr__(self, name):
                    return getattr(torch.nn.functional, name)

                @staticmethod
                def cross_entropy(input, *args, **kwargs):
                    return torch.zeros((), device=input.device, dtype=torch.float32)
            engine.F = NoCE()
    result = engine.train_run(a.dataset, Path(a.data_root), Path(a.output_root), cfg, a.role, a.seed,
                              a.device, a.debug, Path(a.seg_root))
    if a.role == "student" and result.get("method") != arm:
        result["method"] = arm
        engine.write_json(run_dir(a.output_root, a.dataset, "student", a.seed, arm) / "result.json", result)
    print(json.dumps({"role": a.role, "arm": arm, "seed": a.seed, "best_epoch": result["best_epoch"],
                      "best_validation": result["best_validation"]}), flush=True)


# ----------------------------------------------------------------- GPU etiquette
def free_gib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "--id=0"],
                         capture_output=True, text=True)
    return float(out.stdout.strip().splitlines()[0]) / 1024 if out.returncode == 0 else None


def co_user_running(pattern):
    return bool(pattern) and subprocess.run(["pgrep", "-f", "--", pattern], capture_output=True).returncode == 0


def wait_for_gpu(a, peak_gib):
    if a.device != "cuda":
        return
    from scripts.gpu_capacity import GLOBAL_RESERVE_GIB, job_budget_gib
    need = job_budget_gib(peak_gib) + GLOBAL_RESERVE_GIB
    while True:
        if co_user_running(a.wait_pattern):
            print(f"{time.strftime('%F %T')} waiting: a process matching '{a.wait_pattern}' is running", flush=True)
        else:
            free = free_gib()
            if free is None or free >= need:
                return
            print(f"{time.strftime('%F %T')} waiting: {free:.1f} GiB free < {need:.1f} GiB needed", flush=True)
        time.sleep(300)


# ----------------------------------------------------------------- orchestration
def worker_cmd(a, role, seed, arm="teacher"):
    cmd = [sys.executable, "-m", "probe.runner", "worker", "--dataset", a.dataset, "--role", role,
           "--seed", str(seed), "--arm", arm, "--data-root", str(a.data_root), "--output-root", str(a.output_root),
           "--seg-root", str(a.seg_root), "--device", a.device]
    return cmd + (["--config", str(a.config)] if a.config else []) + (["--debug"] if a.debug else [])


def prepare(a):
    use_benchmark()
    from reference.cli import prepare_data, prepare_segmentation, prefetch
    prepare_data(a.dataset, Path(a.data_root), a.data_root_given)
    prepare_segmentation(a.dataset, Path(a.data_root), Path(a.seg_root), a.seg_root_given)
    prefetch()


def teacher(a):
    use_benchmark()
    directory = run_dir(a.output_root, a.dataset, "teacher", 0)
    best = directory / "best.pt"
    if a.teacher_ckpt:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.teacher_ckpt, best)
        print(f"using provided teacher {a.teacher_ckpt} -> {best}", flush=True)
        return
    if (directory / "result.json").is_file():
        print(f"teacher done: {directory}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    wait_for_gpu(a, TEACHER_PEAK_GIB)
    with (directory / "train.log").open("a") as log:
        code = subprocess.run(worker_cmd(a, "teacher", 0), cwd=HOME, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f"teacher failed ({code}); see {directory / 'train.log'}")
    print(f"teacher done: {directory}", flush=True)


def probe(a):
    use_benchmark()
    if not (run_dir(a.output_root, a.dataset, "teacher", 0) / "best.pt").is_file():
        teacher(a)
    pending = [(arm, seed) for seed in a.seeds for arm in a.arms]  # seed-major: paired arms finish together
    active = {}
    failed = []
    while pending or active:
        while pending and len(active) < a.jobs:
            arm, seed = pending[0]
            directory = run_dir(a.output_root, a.dataset, "student", seed, arm)
            if (directory / "result.json").is_file():
                pending.pop(0)
                print(f"done already: {arm} seed={seed}", flush=True)
                continue
            wait_for_gpu(a, STUDENT_PEAK_GIB)
            pending.pop(0)
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "train.log").open("a")
            proc = subprocess.Popen(worker_cmd(a, "student", seed, arm), cwd=HOME, stdout=log, stderr=subprocess.STDOUT)
            active[(arm, seed)] = (proc, log)
            print(f"{time.strftime('%F %T')} start {a.dataset} {arm} seed={seed} pid={proc.pid}", flush=True)
            time.sleep(5)
        for key, (proc, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[key]
            print(f"{time.strftime('%F %T')} end {a.dataset} {key[0]} seed={key[1]} exit={code}", flush=True)
            if code:
                failed.append(key)
        if active:
            time.sleep(10)
    if failed:
        sys.exit(f"failed runs: {failed} (rerun the same command to resume them)")


def status(a):
    cfg = load_config(a.config)
    rows = [("teacher", 0, "teacher")] + [(arm, s, "student") for s in a.seeds for arm in a.arms]
    for arm, seed, role in rows:
        d = run_dir(a.output_root, a.dataset, role, seed, arm)
        hist = d / "history.json"
        done = len(json.loads(hist.read_text())) if hist.is_file() else 0
        total = cfg["teacher_epochs"] if role == "teacher" else cfg["student_epochs"]
        state = "complete" if (d / "result.json").is_file() else ("running/paused" if done else "pending")
        print(f"{a.dataset:10s} {arm:11s} seed={seed}  {done:3d}/{total}  {state}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "teacher", "probe", "all", "status", "worker"])
    p.add_argument("--dataset", choices=["coco", "waterbirds"], required=True)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=sorted(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--jobs", type=int, default=1, help="concurrent student runs (memory-checked before each start)")
    p.add_argument("--wait-pattern", default="", help="do not start a run while a process matching this is alive")
    p.add_argument("--teacher-ckpt", help="use an existing teacher best.pt (e.g. the lab's reference teacher)")
    p.add_argument("--data-root")
    p.add_argument("--seg-root")
    p.add_argument("--output-root", default=str(HOME / "outputs"))
    p.add_argument("--config", help="JSON overriding configs/reference.json (smoke tests only)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--debug", action="store_true", help="tiny random models (smoke tests only)")
    p.add_argument("--no-prepare", action="store_true")
    # worker-only
    p.add_argument("--role", choices=["teacher", "student"], default="student")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arm", default="teacher")
    a = p.parse_args()
    a.data_root_given, a.seg_root_given = a.data_root is not None, a.seg_root is not None
    a.data_root = str(Path(a.data_root).resolve()) if a.data_root else str(default_data_root(a.dataset))
    a.seg_root = str(Path(a.seg_root).resolve()) if a.seg_root else str(BENCH / "data" / "CUB_200_2011" / "segmentations")
    a.output_root = str(Path(a.output_root).resolve())
    if a.config:
        a.config = str(Path(a.config).resolve())
    if a.command == "worker":
        return worker(a)
    if a.command == "status":
        return status(a)
    if a.command == "prepare":
        return prepare(a)
    if a.command == "teacher":
        return teacher(a)
    if a.command == "all" and not a.no_prepare:
        prepare(a)
    probe(a)
    if a.command == "all":
        subprocess.run([sys.executable, "-m", "probe.report", "--output-root", a.output_root], cwd=HOME, check=False)


if __name__ == "__main__":
    main()
