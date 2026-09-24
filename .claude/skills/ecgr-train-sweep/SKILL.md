---
name: ecgr-train-sweep
description: Running and supervising ecgr training - the label-free ssl and cpc stages, the supervised two-output train stage, bxb-based checkpoint selection, the run_pipeline.sh sweep over the four sizes on two GPUs, batch/LR/memory choices at 60 s, monitoring (val_beat_cls_weighted_f1, quality probe), resuming a failed queue, and the common failures (libdevice, OOM, manifest mismatch, wrong run tag). Use this whenever asked to train, pretrain, fine-tune, resume, launch a sweep, pick a checkpoint, or read a training log in this repo.
---

# Training ecgr models

## The three stages and what each learns

| stage | objective | labels | trains |
|---|---|---|---|
| `ssl` | masked-span (35%, 12-step spans) + masked-lead (p 0.5) reconstruction, loss on the hidden part only | none | `backbone` (stem + ResU path + SSM path) |
| `cpc` | InfoNCE: context of windows 1..i predicts latent i+1, i+2 against every other window in the batch (59 windows of 2 s per strip) | none | `context_encoder`, then **frozen** |
| `train` | `poly2` on `beat_cls` over labelled steps only + `QUALITY_LOSS_WEIGHT` × BCE on `lead_quality` against the pipeline's label-free target | beats inside the reviewed span; quality target from the augmentation | everything except the context encoder |

`ssl`/`cpc` weights depend only on the architecture, so they are skipped when present and can
be shared across runs (`ECGR_SSL_RUN`, `ECGR_CPC_RUN`). `train` loads them
(`load_pretrained`) and fine-tunes the backbone.

## Launching

```bash
export ECGR_RUN_TAG=260923_60s                  # pin it: the default rolls over at midnight
ECGR_GPU=0 setsid nohup ./run_pipeline.sh sweep resumamba_5m resumamba_100k </dev/null >logs/q0.log 2>&1 &
ECGR_GPU=1 setsid nohup ./run_pipeline.sh sweep resumamba_3m resumamba_1m   </dev/null >logs/q1.log 2>&1 &
```

Each size runs `ssl -> cpc -> train -> select -> stepeval -> ec57 -> regress`; finished
stages are skipped on a rerun, so **the recovery from any failure is the same command
again**. Single stages: `python -m ecgr {ssl,cpc,train,select,stepeval,ec57} --model X`.
Smoke test of the whole fit loop on a small tree:

```bash
ECGR_TFRECORD_DIR=/path/to/small/tfrecord ECGR_RUN_TAG=smoke \
  $PY -m ecgr train --model resumamba_100k --epochs 1 --steps-per-epoch 5 --validation-steps 2 --ckpt-start-epoch 1
```

## Sizing at 60 s (measured, batch 32, RTX 3090)

| size | s/step | peak VRAM | ~min/epoch on 490k strips |
|---|---|---|---|
| 5m | 0.28 | 13.5 GB | ~75 |
| 3m | 0.19 | 9.2 GB | ~50 |
| 1m | 0.10 | 4.8 GB | ~25 |
| 100k | 0.24 (separable convs are slow on GPU) | 1.9 GB | ~60 |

Defaults: `BATCH_SIZE 32`, `LEARNING_RATE 7e-4`, `EPOCHS 30`, `PATIENCE 8`,
`CKPT_START_EPOCH 8`, `CACHE_DATASET off` (the train split is ~90 GB serialized; set
`ECGR_CACHE_DATASET=1` only if that fits in RAM once per concurrent job). Raising the batch
to 64 fits the 1m/100k sizes but not 5m. One card per queue; `wait_gpu` in the launcher
waits for `ECGR_GPU_FREE_MIB` (18000 for training) before each stage.

## What to watch

- `val_beat_cls_weighted_f1` (config.MONITOR): the step-level F1 over N/V/S on labelled eval steps. It drives EarlyStopping / ReduceLROnPlateau / the BEST_F1 checkpoint. **Never monitor val_loss** with poly2 - it bottoms out at epoch 1 and rises while the F1 improves.
- `val_quality_lead_acc`, `val_quality_sep`, `val_quality_mae`: the lead-quality head on a fixed-seed corrupted probe of eval batches (`step_metrics.lead_quality_report`). lead_acc = how often the head's best lead is the target's; should climb well above 0.33 within a few epochs.
- `confusion_log.txt` under `<RUN_DIR>/eval/<model>/` has the per-epoch matrix and the probe numbers.
- Step F1 does **not** pick the best beat-level checkpoint (README section 8). Every epoch from `CKPT_START_EPOCH` is saved under `epochs/`; `ecgr select` scores them with bxb on the portal-eval sample and copies the no-regression winner to `epochs/selected_by_bxb.keras`, which the launcher's `ec57` stage then scores.

## Failure modes, in the order they usually appear

1. `libdevice not found at ./libdevice.10.bc` - XLA cannot find CUDA's libdevice. `ecgr/xla.py` searches the toolkit, the `nvidia-cuda-nvcc` wheel and Triton; `pip install nvidia-cuda-nvcc-cu12` if it still fails. Never "fix" by setting jit_compile - it is already off.
2. `the tfrecords ... do not match this run` - wrong tree for the geometry; see the `ecgr-data-build` skill.
3. OOM at the first step - batch too large for the card (see the table) or a second job on the same GPU. Memory growth is on, so two jobs *can* share a card only if their peaks add up.
4. A stage "succeeds" but the next scores a stale checkpoint - `run_stage` takes the exit status from the python process, not from `grep`; if you write a new stage, keep `PIPESTATUS[0]`.
5. Eval that "finds nothing" the morning after training - unpinned `ECGR_RUN_TAG` rolled over at midnight.
6. `WeightedF1Checkpoint` raises that the monitor is missing - the model was compiled without `StepConfusion` on `beat_cls` or without validation data; with a single-output (legacy) model the key is `val_weighted_f1` and `train.monitor_key` resolves it.

## Fine-tuning and variants

- Start from a trained checkpoint: `--init-from <file>.keras` (all weights; the context encoder stays frozen unless `--ctx-trainable`).
- Warm-up with a frozen SSL backbone: `--freeze-backbone-epochs N` (recompiles at the unfreeze, resetting Adam moments - off by default for that reason).
- Refinement head on a frozen base: `python -m ecgr refine --model X` (see README 6f); the head passes `lead_quality` through untouched.
- SWA / ensembles: `python -m ecgr swa ...`, `python -m ecgr ec57 --checkpoint a.keras b.keras` (averages both outputs).
