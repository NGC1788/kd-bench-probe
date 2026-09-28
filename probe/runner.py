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
Runs execute one after another in the foreground; nothing is started in the
background and no other process on the machine is inspected.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
BENCH = HOME / "external" / "kshs-aimlab-benchmarks"
BENCH_URL = "https://github.com/jeehoo0507/kshs-aimlab-benchmarks"
BENCH_BRANCH = "codex/maskedkd-reference-runs"
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


def runner_sha256():
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(BENCH), *args], capture_output=True, text=True)


def use_benchmark():
    """Clone the benchmark at the pinned commit (first use only) and make it importable."""
    if not (BENCH / ".git").is_dir():
        BENCH.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--branch", BENCH_BRANCH, BENCH_URL, str(BENCH)], check=True)
    if git("rev-parse", "HEAD").stdout.strip() != BENCH_COMMIT:
        git("fetch", "--quiet", "origin", BENCH_BRANCH)
        if git("checkout", "--quiet", BENCH_COMMIT).returncode:
            sys.exit(f"cannot check out benchmark commit {BENCH_COMMIT[:7]}")
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
        # The runner's hash joins the benchmark's run signature, so editing this file
        # can never silently reuse student results produced by an older version of it.
        cfg = {**cfg, "keep_patches": spec["keep_patches"], "probe_arm": arm, "ce_in_loss": spec["ce"],
               "probe_runner_sha256": runner_sha256()}
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


# ----------------------------------------------------------------- steps
def worker_cmd(a, role, seed, arm="teacher"):
    cmd = [sys.executable, "-m", "probe.runner", "worker", "--dataset", a.dataset, "--role", role,
           "--seed", str(seed), "--arm", arm, "--data-root", str(a.data_root), "--output-root", str(a.output_root),
           "--seg-root", str(a.seg_root), "--device", a.device]
    return cmd + (["--config", str(a.config)] if a.config else []) + (["--debug"] if a.debug else [])


def run_logged(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        return subprocess.run(cmd, cwd=HOME, stdout=log, stderr=subprocess.STDOUT).returncode


def prepare(a):
    """Download and validate data with the benchmark's own scripts, run by this project's Python."""
    use_benchmark()
    data, seg = Path(a.data_root), Path(a.seg_root)
    check = [sys.executable, str(BENCH / "scripts" / "check_assets.py"), a.dataset, str(data)]
    if a.dataset == "coco":
        if not (data / "manifest.json").is_file():
            subprocess.run([sys.executable, str(BENCH / "datasets" / "coco_single" / "prepare.py"),
                            "--output", str(data)], cwd=BENCH, check=True)
    else:
        if not (data / "metadata.csv").is_file():
            subprocess.run([sys.executable, str(BENCH / "datasets" / "waterbirds" / "download.py"),
                            "--output", str(data)], cwd=BENCH, check=True)
        if not seg.is_dir():
            subprocess.run([sys.executable, str(BENCH / "datasets" / "waterbirds" / "download_masks.py"),
                            "--dataset", str(data), "--output", str(seg)], cwd=BENCH, check=True)
        check += ["--seg-root", str(seg)]
    subprocess.run(check, cwd=BENCH, check=True)
    from reference.cli import prefetch  # official DeiT-S / DeiT-Tiny ImageNet weights into .cache/torch
    prefetch()


def teacher(a):
    use_benchmark()
    directory = run_dir(a.output_root, a.dataset, "teacher", 0)
    if a.teacher_ckpt:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.teacher_ckpt, directory / "best.pt")
        print(f"using provided teacher {a.teacher_ckpt}", flush=True)
        return
    if (directory / "result.json").is_file():
        print("teacher: done", flush=True)
        return
    print(f"teacher: training (log: {directory / 'train.log'})", flush=True)
    if run_logged(worker_cmd(a, "teacher", 0), directory / "train.log"):
        sys.exit(f"teacher failed; see {directory / 'train.log'}")
    print("teacher: done", flush=True)


def probe(a):
    use_benchmark()
    if not (run_dir(a.output_root, a.dataset, "teacher", 0) / "best.pt").is_file():
        teacher(a)
    failed = []
    for seed in a.seeds:          # seed-major: the four paired arms of a seed finish together
        for arm in a.arms:
            directory = run_dir(a.output_root, a.dataset, "student", seed, arm)
            if (directory / "result.json").is_file():
                made_by = json.loads((directory / "result.json").read_text())["config"].get("probe_runner_sha256")
                if made_by != runner_sha256():
                    sys.exit(f"{directory} was made by a different probe/runner.py; move it aside to rerun")
                print(f"{arm} seed={seed}: done", flush=True)
                continue
            print(f"{arm} seed={seed}: training (log: {directory / 'train.log'})", flush=True)
            if run_logged(worker_cmd(a, "student", seed, arm), directory / "train.log"):
                failed.append(f"{arm}/seed_{seed}")
                print(f"{arm} seed={seed}: FAILED, see {directory / 'train.log'}", flush=True)
    if failed:
        sys.exit(f"failed runs: {failed}. Rerun the same command to resume them.")


def status(a):
    use_benchmark()
    cfg = load_config(a.config)
    rows = [("teacher", 0, "teacher")] + [(arm, s, "student") for s in a.seeds for arm in a.arms]
    for arm, seed, role in rows:
        d = run_dir(a.output_root, a.dataset, role, seed, arm)
        hist = d / "history.json"
        done = len(json.loads(hist.read_text())) if hist.is_file() else 0
        total = cfg["teacher_epochs"] if role == "teacher" else cfg["student_epochs"]
        state = "complete" if (d / "result.json").is_file() else ("in progress" if done else "pending")
        print(f"{a.dataset:10s} {arm:11s} seed={seed}  {done:3d}/{total}  {state}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "teacher", "probe", "all", "status", "worker"])
    p.add_argument("--dataset", choices=["coco", "waterbirds"], required=True)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=sorted(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
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
    a.data_root = str(Path(a.data_root).resolve()) if a.data_root else str(default_data_root(a.dataset))
    a.seg_root = str(Path(a.seg_root).resolve()) if a.seg_root else str(BENCH / "data" / "CUB_200_2011" / "segmentations")
    a.output_root = str(Path(a.output_root).resolve())
    if a.config:
        a.config = str(Path(a.config).resolve())
    if a.command == "worker":
        return worker(a)
    if a.command == "status":
        return status(a)
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            sys.exit("CUDA is not available to this process (use the GPU node / job the admin assigned)")
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
