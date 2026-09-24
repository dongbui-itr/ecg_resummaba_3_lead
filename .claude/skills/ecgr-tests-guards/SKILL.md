---
name: ecgr-tests-guards
description: Running and extending the ecgr test suite (pytest under tests/, ~130 tests, GPU or CPU), what each test file guards and why, which tests need data on disk and skip otherwise, how to smoke-test the pipeline end to end on a scratch tree without the real 90 GB dataset, and the rule that every contract change ships with the test that would have caught the old behaviour. Use this when asked to run tests, when a test fails, before committing a change to any contract (window geometry, label mask, model outputs, lead policy, decoding), or when adding a new guard.
---

# Tests and guards in ecgr

## Running

```bash
PY=/home/ai-server/miniconda3/envs/beat/bin/python
CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true $PY -m pytest tests/ -q      # ~3-5 min on a 3090
CUDA_VISIBLE_DEVICES=""  $PY -m pytest tests/test_signal_and_labels.py tests/test_regress.py -q   # numpy-only files, seconds
./run_pipeline.sh test                                                                # same, through the launcher
```

Use the `beat` env; the system python has no TensorFlow. The model tests build all four
sizes at (15000, 3) - on CPU that is slow but works. Tests that need data on disk skip
cleanly when it is absent: `test_ec57.py` (mitdb, the portal sets), `test_leakage.py`
(the npy/tfrecord trees).

## What each file pins

| file | guards |
|---|---|
| `test_signal_and_labels.py` | annotated lead → channel 0 and the `lead_order` inverse; zero/duplicate fill; inference window tiling (`segment_starts`, `core_bounds`); training window rules (`window_starts`: whole strip, short/long records, `MIN_REVIEWED_SAMPLES`); `ignore_outside` semantics; **one real wfdb record through `process_record`** (labels only inside the span, padding = IGNORE); label ↔ decode round trip on 30 s and 180 s records; `best_lead`; dataset-2 duplicate collapse; call-time geometry (`apply_geometry`) |
| `test_models.py` | the family is exactly 5m/3m/1m/100k; each under budget and above 70% of it; two outputs with the right shapes and ranges; sub-models exist (59 CPC windows); legacy single-output layout; checkpoint round trip; `DiagSSM1D` export and length-free parameters; kernels 7-21 s; refinement head identity / p_None / pass-through |
| `test_pipeline_and_training.py` | byte-exact parse; IGNORE → zero rows; targets dict; exact lead duplication and matching quality; manifest refusal (incl. a 10 s tree under a 60 s run); masked losses ignore zero-mass rows; poly2 non-monotonicity; `StepConfusion` skips ignored steps; quality target follows the injected corruption / flat / drop / dup; SSL and CPC learn; one real `fit` step publishes `val_beat_cls_weighted_f1`; ensembles average both outputs; decoder filters; bxb selection rule |
| `test_label_free.py` | structural, lexical and causal proofs that ssl/cpc/quality never read a label |
| `test_ec57.py` | mitdb 2 leads @360 Hz → 3 @250 Hz; lead modes; header channel; `.ain` round trip; split sampling determinism; rewritten `.hea` parses; output 2 on a real record; **legacy 10 s checkpoint sets the geometry**; bare `read_leads()` follows config |
| `test_leakage.py` | no EC57 tfrecord in training; the guard fires; v4 studies in neither split; splits disjoint; selection set disjoint from the reporting sets - all read from disk |
| `test_evaluate.py` | `evaluate.py` CONFIG is runnable; preflight refuses missing prerequisites; `auto` baseline resolution; target marks; dashes survive |
| `test_regress.py` | drop vs tolerance; unscorable cells skipped; missing databases ignored; verdict JSON; every size has a shipped baseline; `ecgr regress` exit codes |

## Smoke-testing the whole pipeline without the real data

```bash
SCR=/tmp/ecgr_smoke; mkdir -p $SCR
ECGR_NPY_DIR=$SCR/npy ECGR_TFRECORD_DIR=$SCR/tfrecord ECGR_WORKERS=8 \
  $PY -m ecgr data --step all --db dataset-1 dataset-2 --limit 400        # ~1 min, 800 strips
ECGR_TFRECORD_DIR=$SCR/tfrecord ECGR_RUN_TAG=smoke $PY -m ecgr ssl   --model resumamba_100k --epochs 1 --steps-per-epoch 3 --val-steps 1
ECGR_TFRECORD_DIR=$SCR/tfrecord ECGR_RUN_TAG=smoke $PY -m ecgr cpc   --model resumamba_100k --epochs 1 --steps-per-epoch 3 --val-steps 1
ECGR_TFRECORD_DIR=$SCR/tfrecord ECGR_RUN_TAG=smoke $PY -m ecgr train --model resumamba_100k --epochs 1 --steps-per-epoch 5 --validation-steps 2 --ckpt-start-epoch 1
ECGR_RUN_TAG=smoke $PY -m ecgr ec57 --model resumamba_100k --dbs mitdb --max-records 1 --split-records 3
```

Every stage must complete; the numbers are meaningless. Delete the `smoke` run dir afterwards.

## The rule for contract changes

A contract is anything another module or a stored artefact relies on: window length and
step grid, the IGNORE value, channel-0-is-the-annotated-lead, the output names and shapes,
the `.keras` package string, the lead mode/fill defaults, the decoding rules, the baseline
mapping. When you change one:

1. Find the test that pins the old behaviour (grep the table above) and update it to pin the new one - do not delete it.
2. Add the test that would have failed on the bug you are fixing, phrased as the property, not the implementation.
3. Run the whole suite, not the one file: the geometry tests live in three files.
4. If the change affects what a trained checkpoint means (geometry, outputs, package string), say so in the README section 11 changelog table and in `checkpoints/manifest.json` if you ship weights.

## Reading a failure

- A manifest or shape error inside a test usually means config was left modified by an earlier test - the geometry tests restore `config.apply_geometry(15000)` in `finally`; copy that pattern.
- `DID NOT RAISE` in `test_ec57` geometry test: the ensemble geometry check in `load_checkpoints` was weakened.
- Quality-target tests use `tf.random.set_seed`; a change to the order of random draws in `corrupt` shifts them and can flip a threshold - re-derive the expected numbers, don't loosen the assertion blindly.
