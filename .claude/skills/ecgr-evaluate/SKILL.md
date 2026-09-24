---
name: ecgr-evaluate
description: Scoring ecgr checkpoints and reading the numbers - EC57/bxb on mitdb, nstdb, escdb, ahadb, afdb, the held-out portal beat-eval set (reviewed window only), the portal-train/eval split samples, the lead-quality (most reliable channel) output, the no-regression diff against the 10 s baselines (ecgr regress / evaluate.py), the Se/+P/FP-per-1k vocabulary, and the calibration rule (portal-eval only). Use this for any request to evaluate, benchmark, compare, check for regressions, explain a metric, or produce an EC57 table for this project, including scoring the legacy 10 s checkpoints.
---

# Evaluating ecgr checkpoints

## Two entry points, same machinery (`ecgr/evaluation/ec57.py`)

```bash
PY=/home/ai-server/miniconda3/envs/beat/bin/python
# inside a run: writes <RUN_DIR>/ec57/<tag>/, defaults to the run's BEST_F1 checkpoint
$PY -m ecgr ec57 --model resumamba_1m [--checkpoint FILE ...] [--tag NAME] [--dbs mitdb nstdb]
$PY -m ecgr ec57 --model resumamba_1m --bxb-only          # re-score stored predictions, no GPU
$PY -m ecgr regress --model resumamba_1m                  # diff against the 10 s baseline, exit 1 on a drop

# a specific file, anywhere: edit the CONFIG block at the top, then run - no CLI arguments
$PY evaluate.py                                           # writes ./eval_results/<name>/, exits 1 on a regression
```

`evaluate.py` is the "one checkpoint, one record of what was measured" tool: it checks the
prerequisites first (interpreter, bxb on PATH, databases present, checkpoint exists), marks
acceptance targets (`*`/`!`), and with `BASELINE = 'auto'` diffs against the 10 s baseline the
checkpoint's size is held to. It does not score the portal splits (those are a training tool).

The window geometry is taken **from the checkpoint** (`load_checkpoints` →
`config.apply_geometry`): a 10 s legacy model in `checkpoints/` is swept in 10 s windows, a
60 s model in 60 s windows, by the same code. That is what makes baseline comparisons fair.

## Sources and how each is scored

| source | records | scored on | reads |
|---|---|---|---|
| mitdb (44, paced excluded), nstdb (12), escdb (90), ahadb (79), afdb (23, beats in `.qrs`) | whole records, `bxb` with the EC57 5-min learning period | both real leads (`native`) + zero fill | annotated lead 0 |
| `dataset-v4-beat` | 5,227 held-out 60 s strips | **reviewed window only** (`# startMarkSample/stopMarkSample` in the `.hea`, `scripts/bxb-script-mark-window.sh`) | reviewer's channel → model channel 0 |
| `portal-eval`, `portal-train` | deterministic md5-ordered sample of 5,000 reviewed events | reviewed window only | same |

`portal-train` is data the model saw: read it only as the overfitting gap to `portal-eval`.
Sweeps use 60 s windows with 10 s overlap; `labels.core_bounds` splits each overlap at the
midpoint so every beat is decoded exactly once, from the window where it sits furthest from
an edge.

Outputs per source under `<out>/<db>/`: `<db>_QRS_report_line.out` (bxb), and for a
two-output model `lead_quality_summary.json`; predictions in `<out>/_ann/<db>/*.ain` plus
`lead_quality.csv` (record, best lead in the record's own channel numbering, per-channel
mean quality). `ec57_summary.csv` aggregates the Gross lines.

## Reading the numbers

- `Q_Se / Q_+P`: any beat detected / any detection is a beat. `V_*`, `S_*`: per class, EC57 matching window. `-` = the database has no reference beats of that class (S on ahadb, V/S on afdb) - never a 0.
- `python -m ecgr compare tag1 tag2` adds **FP/1k** (false positives per 1000 reference beats of the class): +P is not comparable across databases because S prevalence runs from 0.14% (escdb) to 10.8% (portal).
- Acceptance targets (`evaluate.TARGETS`): mitdb Q Se/+P ≥ 99.95, S Se > 43, S +P > 80; v4-beat S Se > 88, S +P > 92.
- Known traps: mitdb S is dominated by record 232 (non-premature APCs, only visible as a P wave on V5); ~half the mitdb S false positives sit in AF records 222/219; nstdb +P is what noise costs. A change that "wins" mitdb S_Se alone is suspect - check portal-eval and v4-beat.
- Output 2 diagnostics: `best_lead_histogram` per source; on the portal sets `agrees_with_reviewer_channel`. The reviewer's channel is a default 83% of the time, so agreement is a sanity check, not a score. The head's real accuracy is measured where the answer is known: `stepeval` prints best-lead accuracy on a fixed-seed corrupted probe.

## The non-regression contract

Baselines: `assets/baselines/10s_3lead/<size>.csv` (the 10 s family's summaries, README 8b).
Mapping `config.BASELINE_FOR`: 5m and 3m → 2m, 1m → 1m, 100k → 100k. Tolerance
`REGRESSION_TOLERANCE_PP = 0.1` pp (bxb noise on 5,000 strips). `ecgr regress` / the sweep's
`regress` stage / `evaluate.py` list every cell that fell, skip cells the baseline cannot
score, and exit non-zero on any drop. Report a drop as a drop; do not tune it away on the
benchmark that showed it.

## Calibration rule

`--s-boost`, `--min-run`, `ECGR_DECODE_MIN_PEAK_PROB`, checkpoint selection (`ecgr select`),
refinement-epoch selection and any lead-mode/fill decision are chosen on **portal-eval**.
mitdb and dataset-v4-beat are reported on. A number tuned on the benchmark it is reported
on is not a benchmark number - if you did it anyway, say so in the write-up (README 7e lists
the two past cases).

## Lead policy knobs

`--lead-mode native|single|auto` (how many REAL leads: default `native` = both EC57 leads),
`--lead-fill zero|duplicate` (what fills the missing channel: default `zero`). Portal strips
are natively 3-lead and unaffected. `native` lifted 30/32 Physionet cells over one lead
repeated; keep it unless you are running the single-lead control.
