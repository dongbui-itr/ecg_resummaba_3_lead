---
name: ecgr-label-free
description: The label-free (self-supervised) parts of ecgr and how to prove they stay label-free - masked-reconstruction SSL of the backbone, CPC/InfoNCE of the context encoder, and the lead-quality target derived from injected corruption rather than human labels; how to pretrain on an unlabelled corpus, what tests/test_label_free.py proves (structural, lexical, causal), and what would silently break it. Use this whenever a task mentions self-supervision, pretraining, unlabelled data, "free-label"/"label-free", the quality/reliable-channel target, or touches ssl.py, cpc.py, or pipeline.corrupt.
---

# Label-free learning in ecgr

Three things are learned without a human label. Keeping them that way is a property that
has to be **checked**, because the tfrecords carry the labels in the same record as the
signal and nothing downstream complains if a stage starts using them.

## 1. SSL - masked reconstruction of the backbone (`training/ssl.py`)

Corrupt the 3-lead strip → `backbone` → tiny 2-conv decoder → reconstruct **what was hidden**.
Two corruptions: 35% of steps zeroed in contiguous 12-step (240 ms) spans on every lead, and
with p 0.5 one whole lead zeroed. The target is the raw 5 samples behind each step on every
lead (`_target` is a reshape of the input). Loss is MSE on hidden elements only; `nmse` (loss
over the variance of the hidden part) is the number to watch - 1.0 = no better than the mean.
Saves the **best** epoch (`restore_best_weights`), to `ssl_backbone.weights.h5`.

## 2. CPC - InfoNCE for the context encoder (`training/cpc.py`)

60 s → 59 windows of 2 s at 50% overlap → shared embedding → causal SSM context `c_i`.
`W_j c_i` must pick `v_{i+j}` (j ∈ {1, 2}) out of every latent in the batch (other windows AND
other strips). The strip becomes the unit that has to be told apart - exactly what AdaIN
consumes. `lead_jitter=True` so degenerate montages are described as readily as clean ones.
The encoder is then **frozen** in `train`.

## 3. The lead-quality target (`data/pipeline.corrupt` → `quality_from_corruption`)

The pipeline knows what it did to each lead. For every step and lead:

```
a = local RMS (1 s) of the injected noise, in per-lead z units
q = sigmoid((QUALITY_NOISE_HALF - a) / QUALITY_NOISE_SCALE)     # 1.0 -> 0.5, 2.5 -> ~0
q = 0 where the ORIGINAL lead is flat (2 s local std < QUALITY_FLAT_STD)      # lead-off
q = q[lead 0] on every channel of a duplicated sample; q = 0 on a dropped lead
```

Injected corruption = `_noise_field` (wander, broadband, motion bump) + `_lead_noise_field`
(one lead wrecked, primary capped at 0.8, secondary up to 2.5). The eval split, which is not
corrupted, gets `clean_quality_target` (1 minus flatness). Because the eval target is nearly
trivial, the head is *measured* on a fixed-seed corrupted probe
(`step_metrics.lead_quality_report`: best-lead accuracy, MAE, readable-vs-unreadable separation).

## How label-freeness is proved (`tests/test_label_free.py`)

- **structural**: `make_dataset(..., signal_only=True)` yields ONE tensor - `parse_signal` does not even declare the `labels` feature, so a corpus with no labels at all parses.
- **lexical**: after stripping docstrings/comments, `ssl.py` and `cpc.py` contain none of `y_true`, `CLASS_WEIGHTS`, `SYMBOL_TO_LABEL`, `CLASS_NAMES`, `NUM_CLASSES`, `LOSSES`, `labels_from_annotations`; `quality_from_corruption`/`corrupt`/`flat_mask` mention no label machinery.
- **causal**: identical signals with two completely different label sets give **bit-identical** SSL and CPC losses and bit-identical quality targets. This is the check that cannot be fooled by an indirect path.

When you change anything in those modules, run `pytest tests/test_label_free.py`. If the
lexical test fires on an innocent word (it once caught `one_hot` used for a *channel* index),
fix the test's word list with a comment explaining why, not the code.

## Pretraining on unlabelled recordings

Write tfrecords with only a `signal` feature (`build_tfrecord._example` writes both; a
signal-only writer is three lines using `tf.train.Example` with the `signal` bytes) into a
separate tree, then:

```bash
ECGR_TFRECORD_DIR=/path/to/unlabelled/tree $PY -m ecgr ssl --model resumamba_3m
ECGR_TFRECORD_DIR=/path/to/unlabelled/tree $PY -m ecgr cpc --model resumamba_3m
$PY -m ecgr train --model resumamba_3m      # back on the labelled tree; picks up both weight files
```

The signal must follow the same contract: 15000 × 3 float32, band-passed, annotated (or
primary) lead first, per-lead z-scored. `check_manifest` prints a "reading on trust" line
when the tree has no manifest.

## What would silently break it

- Parsing labels in the SSL/CPC path "and dropping them" (makes the field mandatory again).
- Feeding `augment()` output (which carries `beat_cls`) to a trainer whose `train_step` takes a tuple.
- Deriving the quality target from the reviewer's `channel` column - that is a human label of *which lead was viewed*, not of quality, and it is 83% channel 1 by default. Use it only as the agreement diagnostic in `ec57.write_lead_quality`.
- Using the beat labels to mask the quality loss - quality is defined on every step, labelled or not.
