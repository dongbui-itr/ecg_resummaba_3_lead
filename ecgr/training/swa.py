"""Checkpoint averaging (SWA-style) over saved epochs of one model.

Averaging the weights of the last few good epochs of a run is the cheapest reliable way to
gain a little of both sensitivity and positive predictivity at once: it moves the solution
to a flatter part of the loss surface, which is what a single epoch's checkpoint - picked at
a noisy peak of a step-level metric - lacks. BatchNorm moving statistics are averaged along
with everything else; that is the usual approximation to re-estimating them, and the models
averaged here are consecutive epochs of one run, close enough for it to hold.

Every checkpoint must come from the same architecture and the same run - weights are matched
by position, which is only meaningful when the build order was identical.
"""
import numpy as np
import keras

from .. import models  # noqa: F401  - registers the custom layers


def average_checkpoints(paths, out_path):
    """Average the weights of `paths` (>= 2 .keras files) into one model saved at out_path."""
    if len(paths) < 2:
        raise ValueError("checkpoint averaging needs at least two checkpoints")
    loaded = [keras.models.load_model(p, compile=False) for p in paths]
    first = loaded[0]
    for k, weight in enumerate(first.weights):
        values = [m.weights[k].numpy() for m in loaded]
        if any(v.shape != values[0].shape for v in values):
            raise ValueError(f"weight {weight.path} differs in shape across checkpoints")
        if not np.issubdtype(values[0].dtype, np.floating):
            continue                              # counters etc. stay as in the first
        weight.assign(np.mean(values, axis=0))
    first.save(out_path)
    return out_path
