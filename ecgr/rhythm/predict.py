"""Run a rhythm checkpoint over a whole WFDB record: per-second classes, per-window best lead.

The record goes through exactly the preprocessing the training windows went through
(build.read_leads: resample to 250 Hz, band-pass, native lead order, missing leads zero), is
cut into 10 s windows - the last one pulled back to end at the record's end - and each window
is z-scored per lead.

    rhythm  per second (step_hz 1), or for a per-sample model at rc.SAMPLE_PROBS_HZ steps per
            second - each window's (2500, 6) averaged over 10 samples; a step covered by two
            windows gets the mean of both
    lead    per window: NOISE / CH1 / CH2 / CH3, and each second inherits p(NOISE) of the
            windows covering it, which is what marks NOISE episodes
    channel (dual U-Net) the same four classes per 2 s segment: p(NOISE) per segment, and
            the cleanest lead of each segment in the window records
"""
import csv
import json
import os

import numpy as np
import tensorflow as tf

from ..signal_ops import normalize_window
from . import config as rc
from .build import read_leads
from .labels import decode_episodes, to_current_classes
from .model import rhythm_steps


def window_starts(length, segment=rc.SEGMENT_SAMPLES, hop=None):
    """Starts every `hop` samples (default rc.PREDICT_HOP_SECONDS whole seconds; 10 s = no
    overlap), second-aligned, the last one ending at the record's last second."""
    if hop is None:
        hop = max(1, int(round(rc.PREDICT_HOP_SECONDS))) * rc.SECOND_SAMPLES
    if length <= segment:
        return [0]
    starts = list(range(0, length - segment + 1, hop))
    last = (length // rc.SECOND_SAMPLES) * rc.SECOND_SAMPLES - segment
    if last > starts[-1]:
        starts.append(last)
    return starts


def probs_step_hz(model):
    """Rows per second of predict_signal's rhythm / p_noise for this model: 1 for the
    per-second models, rc.SAMPLE_PROBS_HZ for the finer ones (20 ms, per sample)."""
    per_second = rhythm_steps(model) // rc.OUTPUT_SECONDS
    return 1 if per_second == 1 else min(per_second, rc.SAMPLE_PROBS_HZ)


def predict_tta(model, x, batch_size, variants=None):
    """model.predict averaged over test-time variants (rc.PREDICT_TTA): 'id', 'flip' (every
    lead's sign - a rhythm reads the same upside down, and training flips leads), 'swap' (the
    first two leads exchanged - training permutes the leads). Every output is invariant to
    both, so the mean is taken as is."""
    variants = rc.PREDICT_TTA if variants is None else variants
    outs = []
    for v in variants:
        xv = x
        if v == 'flip':
            xv = -x
        elif v == 'swap':
            xv = x[..., [1, 0, 2]]
        elif v != 'id':
            raise ValueError(f"unknown test-time variant {v!r}")
        outs.append(model.predict(xv, batch_size=batch_size, verbose=0))
    return {k: np.mean([np.asarray(o[k]) for o in outs], axis=0) for k in outs[0]}


def position_weights(rows, floor=None):
    """Weight of each output row of one window when overlapping windows are averaged:
    floor + (1 - floor) * sin(pi * (i + 0.5) / rows), or all ones when the taper is off
    (rc.PREDICT_TAPER_FLOOR = 0). A row covered by a single window gets that window's value
    either way (weighted mean of one)."""
    floor = rc.PREDICT_TAPER_FLOOR if floor is None else floor
    if not floor:
        return np.ones(rows)
    i = np.arange(rows)
    return floor + (1.0 - floor) * np.sin(np.pi * (i + 0.5) / rows)


def beat_step_hz():
    """Rows per second of the 'beat' output (BEAT_STEPS per 10 s window)."""
    return rc.BEAT_STEPS // rc.OUTPUT_SECONDS


def predict_signal(model, leads, batch_size=64, with_beats=False):
    """(n, 3) preprocessed leads -> (rhythm (steps, NUM_CLASSES), p_noise (steps,),
    windows [{start, stop, lead, probs} or {start, stop, noise}]), probs_step_hz(model) steps
    per second. Window start/stop are in seconds. p_noise is the 'lead' output's p(NOISE) of
    the window, or the 'noise' output's p(NOISE) of the 2 s segment the step falls in.
    with_beats adds a 4th value: the beats picked (beats.pick_beats) from the 'beat' output
    averaged over the overlapping windows, or None for a model without that output."""
    n = len(leads)
    if n < rc.SEGMENT_SAMPLES:
        leads = np.concatenate([leads, np.zeros((rc.SEGMENT_SAMPLES - n, leads.shape[1]),
                                                leads.dtype)])
    starts = window_starts(len(leads))
    x = np.stack([normalize_window(leads[s:s + rc.SEGMENT_SAMPLES]) for s in starts])
    y = predict_tta(model, x.astype(np.float32), batch_size)

    step_hz = probs_step_hz(model)
    rhythm_windows = to_current_classes(y['rhythm'])        # a 6-class checkpoint -> 5
    if rhythm_windows.shape[1] != rc.OUTPUT_SECONDS * step_hz:   # (w, 500|2500, K) -> 25 Hz
        rhythm_windows = rhythm_windows.reshape(
            len(starts), rc.OUTPUT_SECONDS * step_hz, -1, rc.NUM_CLASSES).mean(axis=2)
    per_window = rc.OUTPUT_SECONDS * step_hz
    if 'noise' in y:                            # (w, 5, 2) -> p(NOISE) per step
        seg = np.asarray(y['noise'])[..., 1]
        noise_windows = np.repeat(seg, per_window // rc.NOISE_SEGMENTS, axis=1)
    elif 'channel' in y:                        # (w, 5, 4) -> p(NOISE) per step
        seg = np.asarray(y['channel'])[..., rc.LEAD_NOISE]
        noise_windows = np.repeat(seg, per_window // rc.NOISE_SEGMENTS, axis=1)
    else:
        noise_windows = np.repeat(np.asarray(y['lead'])[:, rc.LEAD_NOISE:rc.LEAD_NOISE + 1],
                                  per_window, axis=1)
    step = rc.SECOND_SAMPLES // step_hz          # samples per step

    seconds = max(1, n // rc.SECOND_SAMPLES)
    row_w = position_weights(per_window)                 # (per_window,), 1s when the taper is off
    total = np.zeros((len(leads) // step, rc.NUM_CLASSES))
    noise = np.zeros(len(total))
    count = np.zeros(len(total))
    windows = []
    for i, (s, r, pn) in enumerate(zip(starts, rhythm_windows, noise_windows)):
        a = s // step
        total[a:a + per_window] += r * row_w[:, None]
        noise[a:a + per_window] += pn * row_w
        count[a:a + per_window] += row_w
        sec = s // rc.SECOND_SAMPLES
        win = {'start': int(sec), 'stop': int(min(sec + rc.OUTPUT_SECONDS, seconds))}
        if 'noise' in y:
            win['noise'] = [round(float(v), 4) for v in np.asarray(y['noise'])[i, :, 1]]
        elif 'channel' in y:
            q = np.asarray(y['channel'])[i]                                  # (5, 4)
            win['noise'] = [round(float(v), 4) for v in q[:, rc.LEAD_NOISE]]
            win['channel'] = [rc.LEAD_CLASSES[int(j)] for j in np.argmax(q, axis=-1)]
        else:
            q = y['lead'][i]
            win.update(lead=rc.LEAD_CLASSES[int(np.argmax(q))],
                       probs={k: float(v) for k, v in zip(rc.LEAD_CLASSES, q)})
        windows.append(win)
    count = np.maximum(count, 1e-6)
    keep = seconds * step_hz
    out = ((total / count[:, None])[:keep], (noise / count)[:keep], windows)
    if not with_beats:
        return out
    picked = None
    if 'beat' in y:
        from .beats import pick_beats
        bhz = beat_step_hz()
        bw = np.asarray(y['beat'])                       # (w, BEAT_STEPS, 4)
        bstep = rc.SECOND_SAMPLES // bhz
        btotal = np.zeros((len(leads) // bstep, bw.shape[-1]))
        bcount = np.zeros(len(btotal))
        bw_rows = position_weights(rc.BEAT_STEPS)
        for s, b in zip(starts, bw):
            a = s // bstep
            btotal[a:a + rc.BEAT_STEPS] += b * bw_rows[:, None]
            bcount[a:a + rc.BEAT_STEPS] += bw_rows
        bprobs = (btotal / np.maximum(bcount, 1e-6)[:, None])[:seconds * bhz]
        picked = pick_beats(bprobs, bhz)
    return out + (picked,)


def predict_record(model, path, out_dir=None):
    """Predict one record (path without extension). Writes <name>_rhythm.csv / .json."""
    if isinstance(model, str):
        model = tf.keras.models.load_model(model, compile=False)
    leads, _ = read_leads(path)
    rhythm, p_noise, windows = predict_signal(model, leads)
    step_hz = probs_step_hz(model)
    episodes = decode_episodes(rhythm, p_noise, step_hz=step_hz)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.basename(path)
        with open(os.path.join(out_dir, f"{name}_rhythm.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['second', 'rhythm', *rc.CLASS_NAMES, 'p_noise'])
            for i, (p, pn) in enumerate(zip(rhythm, p_noise)):
                w.writerow([f"{i / step_hz:g}", rc.CLASS_NAMES[int(np.argmax(p))],
                            *[f"{v:.4f}" for v in p], f"{pn:.4f}"])
        with open(os.path.join(out_dir, f"{name}_rhythm.json"), 'w') as f:
            json.dump({'record': path, 'seconds': len(rhythm) // step_hz, 'step_hz': step_hz,
                       'episodes': episodes,
                       'windows': windows}, f, indent=2)
    return rhythm, p_noise, windows, episodes
