"""Check whether the teacher's training-set outputs collapsed onto the label-smoothed label.

    ./kdb saturation

Reads only finished history.json files under outputs/ (no GPU, no training).

If the teacher outputs exactly the label-smoothed label q on training images, then for a student s
    KD = KL(q || s) = CE_ls(s) - H(q),
so the student's last-epoch (train CE - train KD) equals the entropy floor H(q). A value near the floor
means KD carried no information beyond the label on the training set, and every KD variant (full,
masked, CE on/off) optimizes nearly the same objective.
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
NUM_CLASSES = {"coco": 10, "waterbirds": 2}


def ls_entropy(k, eps):
    """Entropy of the label-smoothed target (1 - eps + eps/k on the label, eps/k elsewhere)."""
    hit, miss = 1 - eps + eps / k, eps / k
    return -(hit * math.log(hit) + (k - 1) * miss * math.log(miss))


def last_train(path):
    return json.loads(path.read_text())[-1]["train"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(HOME / "outputs"))
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    a = ap.parse_args()
    root = Path(a.output_root)
    for dataset, k in NUM_CLASSES.items():
        teacher = root / dataset / "teacher" / "history.json"
        if not teacher.is_file():
            continue
        floor = ls_entropy(k, a.label_smoothing)
        t = last_train(teacher)
        print(f"{dataset}: label-smoothing floor H(q) = {floor:.4f}")
        print(f"  teacher last epoch: train loss {t['loss']:.4f}, train accuracy {t['accuracy']:.4f}")
        # CE-off arms report ce = 0, so only arms with CE in the loss can be checked.
        for arm_dir in sorted(d for d in (root / dataset).iterdir() if d.name.endswith("_ce1")):
            gaps = []
            for hist in sorted(arm_dir.glob("seed_*/history.json")):
                r = last_train(hist)
                gaps.append(r["ce"] - r["kd"])
                print(f"  {arm_dir.name:11s} {hist.parent.name}: train ce {r['ce']:.4f}  kd {r['kd']:.4f}"
                      f"  ce-kd {r['ce'] - r['kd']:.4f}")
            if gaps:
                print(f"  {arm_dir.name:11s} mean ce-kd {st.mean(gaps):.4f}  (floor {floor:.4f})")
        print()


if __name__ == "__main__":
    main()
