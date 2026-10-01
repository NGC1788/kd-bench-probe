"""Compare arms over the whole validation curve, not only at the selected checkpoint.

    ./kdb curves

Exploratory, not part of the pre-registered decision. The validation-selected checkpoint is the
maximum of a noisy per-epoch curve (Waterbirds validation worst group: 133 images), so a gap at that
single point can come from peak picking. Here each run is summarized by window means of its
validation primary metric, and full - mask98 is paired by seed on those means.
Reads only finished history.json files under outputs/ (no GPU, no training).
"""
import argparse
import json
import statistics as st
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
METRIC = {"coco": "macro_accuracy", "waterbirds": "worst_group_accuracy"}
WINDOWS = ((1, 20), (21, 50), (51, 100))
ARMS = ("full_ce1", "mask98_ce1", "full_ce0", "mask98_ce0")


def curve(path, metric):
    return {row["epoch"]: 100 * row["validation"][metric] for row in json.loads(path.read_text())}


def window_mean(c, lo, hi):
    xs = [v for e, v in c.items() if lo <= e <= hi]
    return st.mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(HOME / "outputs"))
    a = ap.parse_args()
    root = Path(a.output_root)
    for dataset, metric in METRIC.items():
        runs = {}
        for arm in ARMS:
            for hist in sorted((root / dataset / arm).glob("seed_*/history.json")):
                runs[(arm, int(hist.parent.name.split("_")[1]))] = curve(hist, metric)
        if not runs:
            continue
        print(f"{dataset}: validation {metric} (%), exploratory")
        print(f"  {'arm':11s} seed  best(epoch)   mean 1-20  21-50  51-100   last")
        for (arm, seed), c in sorted(runs.items()):
            best = max(c, key=lambda e: (c[e], -e))
            means = "  ".join(f"{window_mean(c, lo, hi):6.2f}" for lo, hi in WINDOWS)
            print(f"  {arm:11s} {seed:4d}  {c[best]:6.2f}({best:3d})   {means}   {c[max(c)]:6.2f}")
        for ce in ("ce1", "ce0"):
            seeds = sorted(s for (arm, s) in runs if arm == f"full_{ce}" and (f"mask98_{ce}", s) in runs)
            if not seeds:
                continue
            diffs = [window_mean(runs[(f"full_{ce}", s)], 21, 100) - window_mean(runs[(f"mask98_{ce}", s)], 21, 100)
                     for s in seeds]
            print(f"  full - mask98 ({ce}), mean over epochs 21-100: "
                  f"{', '.join(f'{d:+.2f}' for d in diffs)}  mean {st.mean(diffs):+.2f} pt")
        print()


if __name__ == "__main__":
    main()
