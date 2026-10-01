"""Copy every finished result (not the checkpoints) into results/<name>/ so it can be committed.

    ./kdb archive --name kebap-20261001

Refreshes report (runs.csv, summary.csv, p1.json) and writes saturation.txt and curves.txt first.
Copies all files under outputs/ except *.pt (model weights; the runs are bit-for-bit reproducible
from this repository, so the weights are not needed for the record).
"""
import argparse
import contextlib
import io
import shutil
import socket
import sys
from datetime import date
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]


def capture(module, output_root, target):
    """Run a summary command into a text file; a failure is recorded there and does not stop the copy."""
    buf = io.StringIO()
    sys.argv = [module, "--output-root", str(output_root)]
    try:
        with contextlib.redirect_stdout(buf):
            __import__(f"probe.{module}", fromlist=["main"]).main()
    except Exception as e:  # noqa: BLE001 - the raw files are what matter; keep going
        buf.write(f"\n{module} failed: {type(e).__name__}: {e}\n")
        print(f"warning: {module} failed ({e}); copying files anyway")
    target.write_text(buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(HOME / "outputs"))
    ap.add_argument("--name", default=f"{socket.gethostname()}-{date.today():%Y%m%d}")
    a = ap.parse_args()
    src = Path(a.output_root)
    if not src.is_dir():
        sys.exit(f"no outputs at {src}")
    for module in ("report", "saturation", "curves"):
        capture(module, src, src / f"{module}.txt")
    dst = HOME / "results" / a.name
    files = [p for p in src.rglob("*") if p.is_file() and p.suffix != ".pt"]
    for p in files:
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
    size = sum(p.stat().st_size for p in files)
    skipped = sum(1 for p in src.rglob("*.pt"))
    print(f"copied {len(files)} files ({size / 1e6:.1f} MB) to {dst.relative_to(HOME)}; skipped {skipped} .pt checkpoints")


if __name__ == "__main__":
    main()
