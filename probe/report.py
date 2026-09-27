"""Collect finished runs into benchmark-style CSVs and apply the probe's decision rule.

    ./run.sh report            # -> outputs/runs.csv, outputs/summary.csv, outputs/p1.json

runs.csv uses the benchmark's column set (method = arm name) plus `ce_in_loss`,
so rows can be placed next to the benchmark's MaskedKD reference rows.
"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

from probe.runner import ARMS, BENCH, HOME

FIELDS = ("dataset", "data_sha256", "foreground_sha256", "method", "ce_in_loss", "teacher_sha256",
          "pretrained_sha256", "student_arch", "student_init", "keep_patches", "seed", "epochs",
          "selection_metric", "best_epoch", "val_score", "test_primary", "test_overall", "last_test_primary",
          "last_test_overall", "train_hours", "wall_hours", "peak_vram_gib", "samples_per_second", "gpu",
          "parallel_jobs", "checkpoint_sha256", "code_sha256", "code_commit", "run_ref")
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def rows(output_root):
    for path in sorted(Path(output_root).glob("*/*/seed_*/result.json")) + sorted(Path(output_root).glob("*/teacher/result.json")):
        r = json.loads(path.read_text())
        m = r["selection_metric"]
        student = r["role"] == "student"
        yield {"dataset": r["dataset"], "data_sha256": r["dataset_sha256"],
               "foreground_sha256": r.get("foreground_sha256", ""), "method": r["method"],
               "ce_in_loss": r["config"].get("ce_in_loss", True) if student else "",
               "teacher_sha256": r["teacher_sha256"], "pretrained_sha256": r["pretrained_sha256"],
               "student_arch": r["student_arch"], "student_init": r["student_init"],
               "keep_patches": r["config"]["keep_patches"] if student else 196, "seed": r["seed"],
               "epochs": r["epochs"], "selection_metric": m, "best_epoch": r["best_epoch"],
               "val_score": r["best_validation"], "test_primary": r["best_test"][m],
               "test_overall": r["best_test"]["overall_accuracy"], "last_test_primary": r["last_test"][m],
               "last_test_overall": r["last_test"]["overall_accuracy"],
               "train_hours": r["train_seconds"] / 3600, "wall_hours": r["wall_seconds"] / 3600,
               "peak_vram_gib": r["peak_vram_gib"],
               "samples_per_second": r["epochs"] * r["train_samples"] / max(r["train_seconds"], 1e-9),
               "gpu": r["environment"]["gpu"], "parallel_jobs": r["parallel_jobs"],
               "checkpoint_sha256": r["checkpoint_sha256"], "code_sha256": r["code_sha256"],
               "code_commit": r["code_commit"], "run_ref": str(path.parent.relative_to(HOME))
               if path.parent.is_relative_to(HOME) else str(path.parent)}


def describe(xs):
    n, m = len(xs), st.mean(xs)
    out = {"n": n, "mean_pt": round(m, 3), "per_seed_pt": [round(x, 3) for x in xs]}
    if n >= 2:
        sd = st.stdev(xs)
        h = T975.get(n, 1.96) * sd / math.sqrt(n)
        out.update(sd_pt=round(sd, 3), ci95_pt=[round(m - h, 3), round(m + h, 3)],
                   same_sign=all(x > 0 for x in xs) or all(x < 0 for x in xs))
    return out


def verdict(gap0, dce):
    c1 = gap0["mean_pt"] >= 1.0 and gap0.get("same_sign") and gap0["mean_pt"] > 0
    c2 = dce["mean_pt"] >= 0.5 and dce.get("same_sign") and dce["mean_pt"] > 0
    if c1 and c2:
        return "A: CE hides the masking loss"
    if c1:
        return "B: masking loss with or without labels"
    if c2:
        return "C: interaction only (masking helps with labels), not P1"
    return "D: no effect"


def p1(table, dataset, masked, metric):
    """gap_c = full - masked (percentage points), paired by seed; ΔCE = gap_0 - gap_1."""
    def acc(arm, seed):
        return table.get((dataset, arm, seed), {}).get(metric)
    seeds = sorted({s for (d, arm, s) in table if d == dataset})
    arms = ("full_ce1", f"{masked}_ce1", "full_ce0", f"{masked}_ce0")
    done = [s for s in seeds if all(acc(arm, s) is not None for arm in arms)]
    if len(done) < 2:
        return {"complete_seeds": done, "verdict": "incomplete"}
    gap1 = [100 * (acc("full_ce1", s) - acc(f"{masked}_ce1", s)) for s in done]
    gap0 = [100 * (acc("full_ce0", s) - acc(f"{masked}_ce0", s)) for s in done]
    dce = [g0 - g1 for g0, g1 in zip(gap0, gap1)]
    out = {"complete_seeds": done, "gap_1": describe(gap1), "gap_0": describe(gap0), "delta_ce": describe(dce)}
    out["verdict"] = verdict(out["gap_0"], out["delta_ce"]) + ("" if len(done) >= 3 else " (provisional)")
    return out


def reference_check(all_rows, dataset):
    path = BENCH / "results" / dataset / "summary.csv"
    if not path.is_file():
        return None
    ref = next((r for r in csv.DictReader(path.open()) if r["method"] == "maskedkd"), None)
    ours = [r for r in all_rows if r["dataset"] == dataset and r["method"] == "mask98_ce1"]
    if not ref or not ours:
        return None
    mean = st.mean(float(r["test_primary"]) for r in ours)
    return {"reference_mean": float(ref["test_primary_mean"]), "reference_sd": float(ref["test_primary_sd"]),
            "ours_mask98_ce1_mean": mean, "difference_pt": round(100 * (mean - float(ref["test_primary_mean"])), 3),
            "same_teacher": {r["teacher_sha256"] for r in ours} == {ref["teacher_sha256"]},
            "same_data": {r["data_sha256"] for r in ours} == {ref["data_sha256"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(HOME / "outputs"))
    a = ap.parse_args()
    out = Path(a.output_root)
    all_rows = list(rows(out))
    if not all_rows:
        print(f"no finished runs under {out}")
        return
    with (out / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)
    groups = {}
    for r in all_rows:
        groups.setdefault((r["dataset"], r["method"]), []).append(r)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["dataset", "method", "seeds", "test_primary_mean", "test_primary_sd",
                    "last_test_primary_mean", "last_test_primary_sd", "mean_train_hours"])
        for (d, m), rs in sorted(groups.items()):
            tp = [float(r["test_primary"]) for r in rs]
            lp = [float(r["last_test_primary"]) for r in rs]
            w.writerow([d, m, "|".join(str(r["seed"]) for r in rs), st.mean(tp), st.stdev(tp) if len(tp) > 1 else "",
                        st.mean(lp), st.stdev(lp) if len(lp) > 1 else "", st.mean(float(r["train_hours"]) for r in rs)])
    table = {(r["dataset"], r["method"], r["seed"]): {"test_primary": float(r["test_primary"]),
                                                      "last_test_primary": float(r["last_test_primary"])}
             for r in all_rows if r["method"] in ARMS}
    result = {}
    for d in sorted({r["dataset"] for r in all_rows}):
        result[d] = {"decision_metric": "test_primary (validation-selected checkpoint)",
                     "mask98": p1(table, d, "mask98", "test_primary"),
                     "mask98_last_epoch": p1(table, d, "mask98", "last_test_primary"),
                     "reference_reproduction": reference_check(all_rows, d)}
        if any(k[1].startswith("mask59") for k in table if k[0] == d):
            result[d]["mask59"] = p1(table, d, "mask59", "test_primary")
    (out / "p1.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    print(f"wrote {out / 'runs.csv'}, {out / 'summary.csv'}, {out / 'p1.json'}")


if __name__ == "__main__":
    main()
