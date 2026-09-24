---
name: ecgr-overview
description: Orientation for the ecg_resumamba (ecgr) project - the 60 s / 3-lead / 250 Hz ResUMamba beat detector with a label-free lead-quality output. Load this first for ANY task in this repo (reading, changing or running code; answering "how does X work"), then follow its pointers to the specialised ecgr-* skills. Covers the contract, the stage pipeline, where outputs land, the environment (conda env `beat`, two RTX 3090s, WFDB tools) and the non-negotiable rules (no EC57 data in training, labels only inside the reviewed span, calibrate on portal-eval only).
---

# ecgr - orientation

`ecgr` is a TensorFlow/Keras project that reads a **60 s, 3-lead, 250 Hz** ECG strip and
emits two things per 20 ms step:

| output | tensor | meaning |
|---|---|---|
| `beat_cls` | `(3000, 4)` softmax | `None / N / V / S` - detection **and** AAMI classification, no R peaks given |
| `lead_quality` | `(3000, 3)` sigmoid | how readable each lead is; `argmax(mean over time)` = the most reliable channel |

Input is `(15000, 3)` with **the annotated lead on channel 0** (`signal_ops.build_leads`).
Four model sizes: `resumamba_5m`, `resumamba_3m`, `resumamba_1m`, `resumamba_100k`
(under 5M / 3M / 1M / 100k parameters). Everything is configured in
[`ecgr/config.py`](../../../ecgr/config.py) and driven through `python -m ecgr <stage>`.

## The pipeline, in order

```
portal record ─► npy ─► tfrecord ─► ssl ─► cpc ─► train ─► select ─► stepeval ─► ec57 ─► regress
                                    └─ label-free ─┘     └ bxb on portal-eval picks the checkpoint
```

| stage | command | reads labels? | what it produces |
|---|---|---|---|
| data | `python -m ecgr data --step all` | yes (only inside the reviewed span) | `NPY_DIR`, `TFRECORD_DIR` + manifest |
| ssl | `python -m ecgr ssl --model X` | **no** | `checkpoints/<model>/ssl_backbone.weights.h5` |
| cpc | `python -m ecgr cpc --model X` | **no** | `checkpoints/<model>/cpc_context.weights.h5` |
| train | `python -m ecgr train --model X` | yes + label-free quality target | `BEST_F1/*.keras`, `epochs/epoch_NN.keras` |
| select | `python -m ecgr select --model X` | reference beats (portal-eval) | `epochs/selected_by_bxb.keras` |
| stepeval | `python -m ecgr stepeval --model X` | yes | step-level confusion + quality probe |
| ec57 | `python -m ecgr ec57 --model X` | reference beats | `<RUN_DIR>/ec57/<tag>/ec57_summary.csv`, `lead_quality.csv` |
| regress | `python -m ecgr regress --model X` | - | diff against `assets/baselines/10s_3lead/`, exit 1 on a drop |

`./run_pipeline.sh sweep [sizes...]` runs all of it per size; `./run_pipeline.sh test` runs
the tests. Long jobs must be started detached: `setsid nohup ./run_pipeline.sh sweep ... </dev/null >logs/x.log 2>&1 &`.

## Where things are

- Config and every path/hyper-parameter: `ecgr/config.py` (`python -m ecgr config` prints the resolved run).
- Run outputs: `<ECGR_WORK_DIR>/<ECGR_RUN_TAG>/{checkpoints,eval,ec57,logs}/`, default work dir `/mnt/md0/Dong_data/portal_data/train/`, run tag `yymmdd_60s`. **Pin `ECGR_RUN_TAG`** - the default changes at midnight.
- Data: portal datasets under `/mnt/md0/Dong_data/portal_data/dataset-{1..5}/`, EC57 databases under `/mnt/md0/Dong_data/physionet/`, the held-out beat-eval set at `portal_data/dataset-eval/v4/beat-eval-dataset/`.
- 10 s baselines to beat: `assets/baselines/10s_3lead/<size>.csv` (mapping in `config.BASELINE_FOR`).
- Legacy 10 s checkpoints (for comparison runs): `checkpoints/resumamba_{2m,1m,100k,30k}.keras`.
- The long-form design record (Vietnamese): `README.md`; paper and references: `docs/references/`.

## Environment

- Python: `/home/ai-server/miniconda3/envs/beat/bin/python` (TF 2.20, Keras 3.10, wfdb 4.3). The system `python3` has no TensorFlow - use the env, or `PY=... ./run_pipeline.sh`.
- GPUs: two RTX 3090 (24 GB). Set `CUDA_VISIBLE_DEVICES` (the launcher uses `ECGR_GPU`). Memory at batch 32: 5m 13.5 GB, 3m 9.2 GB, 1m 4.8 GB, 100k 1.9 GB.
- WFDB apps `bxb`, `sumstats`, `rdann`, `wrann` are on PATH (`/usr/local/bin`); shell drivers in `scripts/`.
- XLA needs `libdevice.10.bc`; `ecgr/xla.py` finds it. If a stage dies with `libdevice not found`, read that module.

## Rules that are never relaxed

1. **No EC57 database in training.** `pipeline.assert_no_benchmark_data` refuses paths under `PHYSIONET_DIR` or named mitdb/nstdb/escdb/ahadb/afdb. Never route around it.
2. **Labels only inside the reviewed span.** Outside it the label stream is `IGNORE_LABEL` (255); losses and metrics skip zero-mass rows. Never fill those steps with background.
3. **Held-out studies stay out.** `assets/list_studies_eval_v4.json` + `dataset-eval` are removed before the hash split; three checks enforce it (`splits.py`), `tests/test_leakage.py` re-proves it from disk.
4. **Calibrate on portal-eval only.** `s_boost`, `min-run`, `min-peak-prob`, checkpoint and head selection are chosen on the portal-eval split, never on mitdb or the beat-eval set - those are reported on.
5. **Nothing may decrease.** New checkpoints are diffed against the 10 s baselines with `ecgr regress` (tolerance `REGRESSION_TOLERANCE_PP` = 0.1 pp); a drop is a failed stage, not a footnote.

## Which skill next

- Build or rebuild data, fix a record-format problem, audit leakage → `ecgr-data-build`
- Launch / resume / debug training, choose batch and LR, pick a checkpoint → `ecgr-train-sweep`
- Score a checkpoint, read Se/+P, compare to baseline, output 2 diagnostics → `ecgr-evaluate`
- Change the architecture, add a size, keep the parameter budget → `ecgr-model-family`
- Anything about SSL / CPC / the quality target being label-free → `ecgr-label-free`
- Run or extend the tests, understand what a failing test guards → `ecgr-tests-guards`
