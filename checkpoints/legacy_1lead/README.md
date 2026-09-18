# Superseded: the 1-lead checkpoints of run `260917_resumamba`

These three `.keras` files take **(2500, 1)** - one band-passed lead - and the current model
family takes **(2500, 3)** leads. They cannot be loaded by this version of the package, and
`ecgr ec57` refuses them by design rather than reshaping the input behind your back:

```
ValueError: ...resumamba_30k.keras takes (2500, 1) but this run is configured for (2500, 3).
            Set ECGR_IN_CHANNELS to match the checkpoint.
```

They are kept because they are the reference the 3-lead family has to beat, and because the
numbers in `manifest.json` are the only measured EC57 results for the 1-lead contract. To
score them, pin the channel count and point at a 1-lead tfrecord tree:

```bash
ECGR_IN_CHANNELS=1 ECGR_TFRECORD_DIR=<a 1-lead tree> \
python -m ecgr ec57 --model resumamba_30k --checkpoint checkpoints/legacy_1lead/resumamba_30k.keras
```

Note that `--model` only selects the output folder here; the architecture comes from the file.

| file | params | val wF1 | mitdb S_Se/S_+P | portal S_Se/S_+P |
|---|---|---|---|---|
| `resumamba_1m.keras` | 933,660 | 0.8523 | 30.49 / 63.27 | 84.03 / 91.10 |
| `resumamba_100k.keras` | 93,235 | 0.8329 | 30.75 / 61.61 | 80.62 / 90.07 |
| `resumamba_30k.keras` | 29,575 | 0.8318 | 43.35 / 62.08 | 81.55 / 89.06 |

Two differences besides the lead count make these numbers **not** directly comparable with a
3-lead run's, and both are fixes rather than variations - see the README's "What changed"
section:

* the training data they were built from was missing 19,763 reviewed records (4% of the
  corpus, ~14% of all SVE/VE event strips) and had ~95k windows shifted up to 249 samples
  outside the span the reviewer certified;
* beat decoding deduplicated across overlapping windows by comparing against the previously
  accepted position, which keeps the copy from the window where the beat sits closest to an
  edge, instead of splitting each overlap at its midpoint.
