"""Entry point. Run from this folder:

    uv run run.py all --dataset coco         # data -> teacher -> 4 arms x 3 seeds -> report
    uv run run.py all --dataset waterbirds
    uv run run.py status --dataset coco
    uv run run.py report
    uv run run.py saturation                 # did the teacher collapse onto the smoothed label?
    uv run run.py smoke                      # CPU, fake data, tiny models (pipeline check)

Everything the project writes stays in this folder (.venv, .cache, external, outputs).
"""
import os
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
CACHE = HOME / ".cache"
# Force every cache this project may touch into the folder (also inherited by child processes).
for key, sub in (("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "xdg"), ("CUDA_CACHE_PATH", "nv"),
                 ("MPLCONFIGDIR", "mpl"), ("HF_HOME", "hf"), ("PIP_CACHE_DIR", "pip"), ("UV_CACHE_DIR", "uv")):
    os.environ[key] = str(CACHE / sub)
os.chdir(HOME)
sys.path.insert(0, str(HOME))

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    if command == "report":
        from probe.report import main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    elif command == "saturation":
        from probe.saturation import main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    elif command == "smoke":
        subprocess.run([sys.executable, "tests/make_fake_coco.py", "outputs/smoke/data/coco_single"], check=True)
        sys.argv = [sys.argv[0], "all", "--dataset", "coco", "--data-root", "outputs/smoke/data/coco_single",
                    "--output-root", "outputs/smoke", "--config", "configs/smoke.json", "--device", "cpu",
                    "--debug", "--no-prepare"] + sys.argv[2:]
        from probe.runner import main
        main()
    elif command in ("all", "prepare", "teacher", "probe", "status", "worker"):
        from probe.runner import main
        main()
    else:
        print(__doc__)
