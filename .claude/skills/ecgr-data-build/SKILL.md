---
name: ecgr-data-build
description: How ecgr turns portal ECG records into 60 s training windows (npy -> tfrecord), including the reviewed-span label mask (IGNORE outside the span), the study-level hash split, the held-out beat-eval studies, the dataset-2 duplicate fix, short/long record handling, and the post-build leakage audit. Use this for any request to build, rebuild, inspect or debug training data, add a dataset, change the window/label geometry, or verify that no eval/EC57 study leaked into training. Also use when a training run complains about the tfrecord manifest.
---

# Building the ecgr training data

## What one record becomes

A portal record is a 60 s strip (15000 samples, 3 leads, 250 Hz) whose `.atr` carries
auto-annotations for the whole minute, but only the **reviewed span**
`[start_sample, stop_sample)` from `dataset_info_full.csv` was certified by a human
(usually 10 s, sometimes 20-30 s). The builder (`ecgr/data/build_npy.py`) therefore writes:

- **signal**: the whole strip, band-passed 0.5-30 Hz, annotated lead rolled to channel 0, per-lead z-score over the 60 s → `float32[15000, 3]`
- **labels**: `uint8[3000]` on the 20 ms grid. Inside the span: `0/1/2/3 = None/N/V/S` blocks of 8 steps before + 2 after each R peak (`labels.labels_from_annotations`). Outside the span, and in the padding of a short record: `config.IGNORE_LABEL = 255` (`labels.ignore_outside`).

Why IGNORE and not background: the unreviewed 50 s *do* contain beats; calling them
"no beat" would teach the model to miss them. IGNORE says "do not ask" - the loss and the
step metric skip zero-mass rows (`training/losses.py`, `evaluation/step_metrics.py`), while the
signal still gives the model (and the label-free stages) the rhythm context.

Geometry rules (`build_npy.window_starts`):
- span shorter than `MIN_REVIEWED_SAMPLES` (2475 = 10 s minus the slack that admits the 2499-sample spans) → record skipped (`skipped_short_span`). This keeps the same event population as the 10 s pipeline, so numbers compare event for event.
- span ≥ one window → slide by `SEGMENT_STRIDE_SECONDS` inside the span.
- otherwise one window: the whole strip when the strip is one window long (the normal case), centred on the span for longer records, edge-padded for shorter ones (`padded` counter; padding = IGNORE).
- flatness (`is_flat`) is judged on lead 0 **inside the span only**.

## Commands

```bash
PY=/home/ai-server/miniconda3/envs/beat/bin/python
$PY -m ecgr data --step npy --db dataset-1 --limit 300         # quick trial
$PY -m ecgr data --step all --workers 32                        # full build, all five datasets
$PY -m ecgr data --step all --db dataset-2                      # rebuild one dataset (manifest is merged)
$PY -m ecgr data --step tfrecord                                # npy -> tfrecord only
```

Outputs: `config.NPY_DIR` (`portal_data/npy_3000_4_250_60_3lead/<db>/{train,eval}/*_batch_N.npy`
+ `split_verification.json`, `studyid_info.json`) and `config.TFRECORD_DIR`
(`train/tfrecord_60s_3lead/<db>/{train,eval}/*.tfrecord` + `dataset_manifest.json`).
Budget: ~183 kB per example, ~90 GB for the train split; the npy build takes minutes with
32 workers, the tfrecord write somewhat longer.

To build into a scratch location without touching the real trees, set `ECGR_NPY_DIR` and
`ECGR_TFRECORD_DIR` (both are read by config). A smoke build with `--limit 400` on two
datasets takes about a minute.

## Datasets and the holdout

- Training datasets: exactly `config.TRAIN_DATASETS = dataset-1 .. dataset-5`. The re-curated sets (`dataset-3-filter-*`, `dataset 2_3_4 - AFib - v2`, `dataset_ivcd`) are duplicates of events already in 2/3/4 and are left out.
- Held out before the split: every study in `assets/list_studies_eval_v4.json` and every study of `config.TEST_DATASETS` (`dataset-eval`). The beat-eval set (5,227 records, 2,242 studies) is entirely inside that list.
- Split: `splits.study_split_side` = md5 of the study id, 80/20, **global per study** so a study is on the same side in every dataset.
- Three enforcement points: `held_out_studies` (before), `verify_split_integrity` (before any write; also cross-dataset), `audit_written_data` (after, from study ids stored in the npy). `--no-audit` skips the last one only.
- Dataset layout quirks handled by `_record_files`: some datasets nest one folder deeper; dataset-2 stores every strip twice byte-identically (deduped by content hash).

## When the manifest refuses to load

`pipeline.check_manifest` compares `segment_samples / in_channels / output_steps /
num_classes / dtypes` between the tfrecords and the live config. A mismatch means the run
points at the wrong tree - typically a 10 s tree (`tfrecord_3lead`, 2500 samples) under a
60 s config. Fix by rebuilding or by pointing `ECGR_TFRECORD_DIR` at the matching tree; never
by editing the manifest.

## Checking a build

```bash
$PY -c "import json; from ecgr import config; m=json.load(open(config.TFRECORD_DIR+'/dataset_manifest.json')); print(m['totals'])"
$PY -m pytest tests/test_leakage.py -q          # re-proves the separation from disk
```

`totals` lists segments, `class_steps` (labelled steps only) and `ignored_steps` per split.
Expect roughly 75-85% of steps ignored (10 s reviewed of 60 s) and class shares of the
labelled steps around None 64%, N 31%, V 2%, S 2-3%.

## Changing the geometry

`SEGMENT_SECONDS`, `STEP_SAMPLES` and `IN_CHANNELS` live in config; `NPY_DIR`/`TFRECORD_DIR`
names include the window length, so a new geometry lands in a new tree. After any change:
rebuild, then run `tests/test_signal_and_labels.py` (window geometry, ignore mask, one real
record through `process_record`) and `tests/test_pipeline_and_training.py` (parse round trip).
