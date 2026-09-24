---
name: ecgr-model-family
description: The ecgr model architecture and its four sizes (resumamba_5m/3m/1m/100k under 5M/3M/1M/100k parameters) - dual-path ResU + diagonal-SSM backbone, CPC context encoder with AdaIN, rhythm cross-attention, the beat head and the lead-quality head, the two nested sub-models, custom Keras layers and the checkpoint format. Use this when asked to change the architecture, add or resize a model, count or rebalance parameters, change kernel lengths or receptive fields, add an output, export a model, or explain why the model is built the way it is.
---

# The ecgr model family (`ecgr/models/`)

## Layout of one model (`resumamba.build_resumamba_seq2seq`)

```
input (15000, 3)
 ├─ backbone  ──────────────────────────────► features (3000, width)
 │    stem: 2×conv9 → maxpool 5 → conv5        (down to the 20 ms grid)
 │    path 1: ResU blocks depths (3, 2)         (multi-scale morphology, Hwang 2023)
 │    path 2: ssm_blocks × [in_proj → dw3 → LayerNorm → SiLU → DiagSSM1D (bidirectional FIR) × SiLU gate → out_proj → LayerNorm → +residual]
 │    concat → conv1 fuse
 ├─ context_encoder (CPC): 59 windows of 2 s → shared conv net → causal SSM → mean f_p
 ├─ AdaIN ×2 (kernels 3, 7): normalise features over time, re-scale from f_p
 ├─ RhythmDescriptor (lead 0 envelope autocorrelation, 30-220 bpm) → tokens → MHA (query = steps)
 ├─ dropout → Conv1D(4, softmax)              → beat_cls     (3000, 4)
 └─ from backbone features: conv7 → Conv1D(3, sigmoid) → lead_quality (3000, 3)
```

Outputs are a **list** `[beat_cls, lead_quality]`; layer names `beat_cls` / `lead_quality`
are the keys the pipeline's target dict and the compile dicts use. `models.split_outputs`
and `models.has_quality_output` make single-output legacy checkpoints and two-output models
interchangeable for every consumer; `build(name, use_quality=False)` gives the legacy layout.

Two nested `keras.Model`s exist because they are pretrained without labels and loaded by
name: `models.sub_model(m, 'backbone')` (SSL) and `models.sub_model(m, 'context_encoder')`
(CPC, frozen in `train`).

## The sizes (`resumamba.SIZES`, budgets in `resumamba.BUDGETS`)

| name | params | width | ssm blocks × kernel (steps) | separable | notes |
|---|---|---|---|---|---|
| `resumamba_5m` | 4,901,171 | 304 | 5 × 1024 (20.5 s) | no | widest, longest memory |
| `resumamba_3m` | 2,835,368 | 236 | 4 × 768 (15.4 s) | no | |
| `resumamba_1m` | 961,839 | 128 | 3 × 512 (10.2 s) | no | the paper's scale |
| `resumamba_100k` | 97,252 | 48 | 2 × 384 (7.7 s) | yes | embedded target |

`python -m ecgr models` prints the live counts; `tests/test_models.py` fails a size that is
over its budget or under 70% of it. Parameter counts are independent of the window length
(everything is convolutional or per-step), so the same SIZES serve 10 s and 60 s.

Design rules when rebalancing:
- Spend the budget on `width`, `resu_mid`, `adain_channels`, `ctx_dim`; never remove SSM blocks or ResU depth - the receptive field is what separates S from N.
- `kernel_len` is **free** (a DiagSSM kernel costs 4 scalars per (channel, state) regardless of length; it is evaluated by FFT sized to the window). At 60 s use 8-20 s kernels.
- Below ~100k use `separable=True`; it is slower on GPU (depthwise convs) but cheap in parameters.
- Keep `attn_heads × attn_key_dim` modest; the MHA over 3000 queries and 4 tokens is memory, not parameters.
- After any change: `python -m ecgr models`, then `pytest tests/test_models.py`, then a GPU probe (see the `ecgr-train-sweep` skill for the measured table) - the 5m size is at 13.5 GB per batch of 32.

## Custom layers (`models/layers.py`) and the storage format

`DiagSSM1D`, `AdaIN`, `RhythmDescriptor`, `Frame1D`, `MergeWindows`, `SplitWindows` (plus
`refine.BeatClassRefine`, `refine.LogProb`, `step_metrics.StepConfusion`) are registered
under `PKG = "resumamba_seq2seq"`. **That string is a storage format**: Keras writes
`resumamba_seq2seq>DiagSSM1D` into every `.keras` file. Renaming the package orphans every
checkpoint. Every registered layer needs `get_config` and, when its build creates sublayers,
a `build` that constructs them (see `AdaIN`).

`DiagSSM1D` details worth knowing before touching it: the kernel is computed from
`(log_lam, omega, phase, mix)` banks, run as a causal convolution **via FFT** (cuDNN rejects
the filter gradient at widths ≥128), bidirectional = second bank on the reversed sequence,
L1-normalised. `kernel_numpy()` materialises the FIR for export to a `DepthwiseConv1D`.

`ssm_block` normalises with **LayerNormalization, never BatchNorm**. The block multiplies
its normalised branch by a SiLU gate and adds it to the residual stream, so any difference
between a normaliser's training-mode and inference-mode behaviour is multiplied block after
block: with BN the 5m size trained normally while its inference-mode SSL reconstruction
diverged (val_nmse 5.5 -> 187 -> 3e9 after lowering BN momentum). LayerNorm is the same
computation in both modes; `test_ssm_blocks_normalise_identically_in_both_modes` pins it.
The remaining BNs (stem, ResU, context encoder, quality head) use `layers.BN_MOMENTUM = 0.9`.
If you add a gated or multiplicative block, do not put BatchNorm in it.

`RhythmDescriptor` reads channel 0 only - the invariant that keeps it identical between a
3-lead strip and an EC57 record with a filled channel (`test_rhythm_descriptor_reads_lead_zero_only`).

## The lead-quality head (output 2)

Two convolutions on the backbone features, sigmoid per lead. It has no label of its own:
its target is built in `data/pipeline.quality_from_corruption` from the corruption the
augmentation injected and from a flatness test (see the `ecgr-label-free` skill). At
inference `labels.best_lead` averages it over the record and `ec57.record_lead_choice`
maps the argmax back to the record's channel numbering through `signal_ops.lead_order`
(filled channels can never win). To change its capacity: `quality_width` in
`build_resumamba_seq2seq` (default `width // 4`).

## Refinement head (`models/refine.py`)

`attach_refinement(base)` wraps a trained base: a small bidirectional SSM reads the
pre-softmax features + `p1` + `log p1` and emits a correction over the beat classes only;
`p_None` is preserved exactly, the head is zero-initialised (epoch 0 IS the base), and
`lead_quality` passes through untouched. Modes `beats` (N/V/S) and `s_only` (N↔S, p_V fixed).

## Adding a size or an output - checklist

1. Add the entry to `SIZES` and `BUDGETS`; keep the `resumamba_<tag>` naming (`keras_name` derives the folder name from it).
2. `python -m ecgr models` - under budget and above 70% of it.
3. Update `run_pipeline.sh` `SIZES`, `config.BASELINE_FOR` (which 10 s baseline it is held to) and the family assertion in `tests/test_models.py`.
4. For a new output: name the layer, add it to the pipeline target dict, `train.compile_targets`, `step_metrics`/`ec57` consumers, and `refine.attach_refinement` passthrough. Write the test that pins its shape and range.
