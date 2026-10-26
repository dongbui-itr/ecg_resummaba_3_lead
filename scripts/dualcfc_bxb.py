#!/usr/bin/env python
"""Beat-level AAMI/EC57 (bxb) for a rhythm-family checkpoint that has a 'beat' output.

The beat pipeline's `ecgr ec57` expects a single-output model with config.OUTPUT_STEPS steps
per window and the class order config.CLASS_NAMES = ['None', 'N', 'V', 'S']. A dualcfc /
dualunet rhythm checkpoint is multi-output and its beat head is (1250, 4) in the rhythm
order rc.BEAT_CLASSES = ['none', 'N', 'S', 'V'].

Two adaptations, and nothing else: the beat head is selected and its last two columns are
swapped into the beat pipeline's order, and labels.STEP_SAMPLES is retuned from 5 (500 steps)
to 2 (1250 steps) so decode_beats maps a step back to the right sample. Everything downstream
- the record inventories, the reviewed-window marking of the portal set, the paced-record
exclusions, bxb and the report - is the beat pipeline's own, unchanged, so the numbers are
directly comparable with every other `ecgr ec57` run.

Not done here: the rhythm side's test-time augmentation (rc.PREDICT_TTA) and window taper.
This is the plain forward pass, which is what `ecgr ec57` does for the beat models too.

    python scripts/dualcfc_bxb.py --checkpoint <best_model.keras> --tag <tag> [ec57 flags]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from ecgr import config, labels  # noqa: E402
from ecgr.evaluation import ec57  # noqa: E402
from ecgr.rhythm import config as rc  # noqa: E402
from ecgr.rhythm import model as rhythm_models  # noqa: E402,F401  registers the custom layers

# rhythm ['none', 'N', 'S', 'V'] -> beat ['None', 'N', 'V', 'S']
BEAT_COLUMN_ORDER = [0, 1, 3, 2]


class BeatHead:
    """A rhythm checkpoint seen by the beat pipeline as its single beat output.

    noise_gate: with a threshold, every step of a 2 s segment whose 'noise' output says
    p(NOISE) >= threshold is set to 'None' before decoding - the model declares the segment
    unreadable and calls no beat there. Missed beats in such a segment then count against
    QRS Se, never against +P: the AAMI way of scoring a monitor that blanks noise."""

    def __init__(self, path, noise_gate=None, gate_floor=None):
        self.model = tf.keras.models.load_model(path, compile=False)
        self.noise_gate = noise_gate
        # with a floor, a flagged segment loses only the steps the beat head is UNSURE of
        # (p(beat) = 1 - p(None) below the floor): a confident beat inside a segment the noise
        # head got wrong survives, a noise-triggered weak detection does not
        self.gate_floor = gate_floor
        if noise_gate is not None and 'noise' not in self.model.output_names:
            raise ValueError(f"--noise-gate needs a 'noise' output; {path} has "
                             f"{self.model.output_names}")
        out = self.model.output
        names = list(out) if isinstance(out, dict) else [t.name.split('/')[0] for t in out] \
            if isinstance(out, (list, tuple)) else [self.model.output_names[0]]
        if 'beat' not in self.model.output_names:
            raise ValueError(f"{path} has no 'beat' output (outputs: {names})")
        self.index = self.model.output_names.index('beat')
        self.name = f"{self.model.name}:beat"
        self.input_shape = self.model.input_shape
        self._fn = None

    def count_params(self):
        return self.model.count_params()

    def summary(self, *a, **kw):
        return self.model.summary(*a, **kw)

    def _compiled(self):
        if self._fn is None:
            @tf.function(reduce_retracing=True)
            def fn(x):
                y = self.model(x, training=False)
                beat = y['beat'] if isinstance(y, dict) else (
                    y[self.index] if isinstance(y, (list, tuple)) else y)
                if self.noise_gate is not None:
                    p_noise = y['noise'][..., 1]                              # (B, segments)
                    steps = beat.shape[1] // p_noise.shape[1]
                    blank = tf.repeat(p_noise >= self.noise_gate, steps, axis=1)  # (B, T)
                    if self.gate_floor is not None:
                        blank = blank & ((1.0 - beat[..., 0]) < self.gate_floor)
                    none = tf.one_hot(tf.zeros_like(blank, tf.int32), beat.shape[-1])
                    beat = tf.where(blank[..., None], none, beat)
                return tf.gather(beat, BEAT_COLUMN_ORDER, axis=-1)
            self._fn = fn
        return self._fn

    def predict_probs(self, segments, batch_size=None):
        fn = self._compiled()
        bs = int(batch_size or 64)
        x = np.asarray(segments, dtype=np.float32)
        return np.concatenate(
            [np.asarray(fn(tf.constant(x[i:i + bs]))) for i in range(0, len(x), bs)], axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--dbs', nargs='*', default=None)
    p.add_argument('--max-records', type=int, default=None)
    p.add_argument('--s-boost', type=float, default=1.0)
    p.add_argument('--bxb-only', action='store_true')
    p.add_argument('--skip-physionet', action='store_true')
    p.add_argument('--skip-portal', action='store_true')
    p.add_argument('--splits', nargs='*', default=[],
                   help="portal splits to sample as well (default: none)")
    p.add_argument('--split-records', type=int, default=None)
    p.add_argument('--lead-mode', default=None)
    p.add_argument('--fill-mode', default=None)
    p.add_argument('--noise-gate', type=float, default=None,
                   help="call no beat in a 2 s segment with p(NOISE) >= this (e.g. 0.5); "
                        "default off")
    p.add_argument('--noise-gate-floor', type=float, default=None,
                   help="with --noise-gate: in a flagged segment drop only steps whose "
                        "p(beat) is below this (e.g. 0.9); default: drop every step")
    args = p.parse_args()

    steps = rc.BEAT_STEPS
    labels.STEP_SAMPLES = config.SEGMENT_SAMPLES // steps
    print(f"beat head: {steps} steps/window, {labels.STEP_SAMPLES} samples/step, "
          f"columns {BEAT_COLUMN_ORDER} -> {config.CLASS_NAMES}")

    if args.noise_gate is not None:
        print(f"noise gate   : no beat where the 2 s segment has p(NOISE) >= {args.noise_gate}")
    ec57.load_checkpoints = lambda paths: BeatHead(
        paths[0] if isinstance(paths, (list, tuple)) else paths, noise_gate=args.noise_gate,
        gate_floor=args.noise_gate_floor)

    ec57.run(args.checkpoint, tag=args.tag, dbs=args.dbs, max_records=args.max_records,
             s_boost=args.s_boost, bxb_only=args.bxb_only,
             skip_physionet=args.skip_physionet, skip_portal=args.skip_portal,
             splits=tuple(args.splits), split_records=args.split_records,
             lead_mode=args.lead_mode, fill_mode=args.fill_mode)


if __name__ == '__main__':
    main()
