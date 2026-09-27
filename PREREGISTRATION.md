# Pre-registration: label probe on the KSHS × AIM Lab benchmark

**Status: DRAFT (2026-09-27).** To be confirmed, and dated here, before the first student run. After the first student result exists, changes go only to the deviation log.

This probe is separate from the ImageNet-100 probe ([NGC1788/kd-probe](https://github.com/NGC1788/kd-probe)). Its results are reported separately and do not replace it.

## Question
In the benchmark's reference regime, does removing the ground-truth CE term increase the **extra** loss caused by teacher-input masking? The regime is: a DeiT-S teacher fine-tuned from ImageNet and then frozen; an ImageNet-initialized DeiT-Tiny student; COCO single and Waterbirds.

## Fixed settings
| Item | Value |
|---|---|
| Code | Benchmark commit `e1c39e7` (branch `codex/maskedkd-reference-runs`), `reference/` trainer unchanged; this repository's `probe/runner.py` sets only the run directory, `keep_patches`, and CE on/off |
| Config | Benchmark `configs/reference.json`: teacher 30 epochs, student 100 epochs, batch 32 × 4 accumulation, lr 5e-5, warmup 5, cosine to 1e-6, wd 0.05, label smoothing 0.1, drop-path 0.1, clip 1.0, kd_alpha 0.5, T = 1 |
| Arms | `full_ce1`, `mask98_ce1`, `full_ce0`, `mask98_ce0`. `ce0` removes the CE term and keeps the KD weight at 0.5 |
| Seeds | 0, 1, 2, paired across arms |
| Teacher | One teacher per dataset, shared by all arms. It is either the lab's reference teacher (hash recorded) or retrained with the same protocol |
| Environment | Python 3.11, torch 2.5.1 + cu124, one RTX A5000 |

## Decision metric and rule
- **Metric:** the benchmark primary metric (COCO macro accuracy, Waterbirds worst-group accuracy) on test, at the **validation-selected** checkpoint (benchmark rule: never select with test). The last-epoch test value is reported but not used for decisions.
- **Notation:** `gap_c = A_full,c − A_mask98,c`, paired by seed; `ΔCE = gap_0 − gap_1`, in percentage points.
- **Rule** (same as the ImageNet-100 probe). Both conditions must hold, each with the same sign in all 3 seeds:
  1. `gap_0 ≥ 1.0 pt`
  2. `ΔCE ≥ 0.5 pt`
- **Outcomes:** A (both hold) = CE hides the masking loss. B (only 1) = loss with or without labels. C (only 2) = masking helps with labels only; this is a different finding. D = no effect.
- **Primary dataset: COCO single. Waterbirds** is evaluated with the same rule and reported as a separate robustness result.
- **Known limits:**
  - COCO accuracy is near the ceiling (reference MaskedKD 0.979 ± 0.006), so a 1 pt gap is hard to reach.
  - Waterbirds worst-group accuracy has more room but more spread (0.728 ± 0.024).
- **Sanity check, not a decision:** `mask98_ce1` should match the benchmark's MaskedKD reference within its seed spread. A large mismatch is investigated before anything is interpreted.

## Deviation log
(none)
