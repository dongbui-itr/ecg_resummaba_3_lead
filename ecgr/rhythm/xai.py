"""Explainability for the dual U-Net rhythm model: what each layer does, what each decision rests
on, and which inference-time knobs move the result. Read-only: nothing here trains or changes a
checkpoint; every number is measured on the held-out 'eval' split of the training windows
(labelled per second, never EC57).

Four parts, each a function and a section of the report (python3 -m ecgr.rhythm xai ...):

  probe       counterfactual interventions on named layers, the beats left exactly as they are:
                atrial_off      the QRST-cancelled residual zeroed - the A branch is blind
                atrial_swap     the residual taken from a window of another class (donor)
                rr_swap         the RR autocorrelation descriptor taken from the donor
                rr_mean         the RR descriptor replaced by its mean over the sample
                beat_cond_off   the beat probabilities fed to the rhythm decoder set to 'none'
                lead_uniform    the lead weights of both stems forced uniform
                ssm_identity    every DiagSSM1D replaced by the identity (no long-range context)
                f_inject        synthetic 4-8 Hz fibrillatory waves added to SINUS windows
              and, per class, how the per-second recall and the mean p(class) move. The
              question it answers: does AFIB rest on atrial activity or on R-R irregularity
              alone (md_atrial_activity.md section 4), does AVB rest on P waves, VT on the beats.
  layers      what the layers learned: lead-weight entropy and agreement with the clean-lead
              output, lead weights under a noise-corrupted lead, the QRST cancellation's
              residual energy and 4-9 Hz share per class, the time constants of every SSM
              kernel, dead ReLU channels, and gradient x activation shares at the concatenations
              where the branches meet (V / A / SSM / beat condition).
  experiment  inference-time sweeps that need no retraining: the QRST cancellation's window
              (pre / post) and peak threshold, the lead-weight temperature, the SSM kernel
              truncation - per-second macro F1 per point. A point that wins here is a
              candidate for the validation records + EC57 run (eval_rhythm.py), never adopted
              from this split alone.
  explain     integrated gradients of one record second (time x lead), with the beats, the
              rhythm probabilities, the atrial residual and the lead weights, as a PNG.

`suggest` turns the measurements into a ranked list of experiments (report section 5).
"""
import contextlib
import json
import os
import time

import numpy as np
import tensorflow as tf

from . import config as rc
from .labels import to_current_classes

CLASS_NAMES = rc.CLASS_NAMES
K = rc.NUM_CLASSES
STEPS_PER_SECOND = rc.BEAT_STEPS // rc.OUTPUT_SECONDS          # rhythm rows per second (125)
GRID_SECONDS = rc.OUTPUT_SECONDS / rc.BACKBONE_STEPS             # SSM step (0.04 s)
SSM_LAYERS = ('vs_ssm', 'neck_ssm0_ssm', 'neck_ssm1_ssm', 'rhythm_ssm_ssm')


def ssm_layers(model):
    """The DiagSSM1D layers of this model (the 100k / 30k sizes have one neck block)."""
    names = {l.name for l in model.layers}
    return [n for n in SSM_LAYERS if n in names]
BRANCH_CONCATS = {                                               # concat -> names of its inputs
    'neck_bottom': ['V bottom', 'A bottom'],
    'neck2_skip': ['neck (up)', 'V skip 250', 'A skip 250', 'V SSM'],
    'rhythm_beat_cond': ['neck z', 'beat probs'],
    'rhythm_skip': ['rhythm (up)', 'V skip 1250', 'A skip 1250'],
}


# ---------------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------------

def load_model(path):
    import keras
    from . import dualunet  # noqa: F401  (registers the custom layers)
    from ..models import layers  # noqa: F401
    return keras.models.load_model(path, compile=False)


def load_windows(split='eval', per_class=300, seed=0, npy_dir=None):
    """A class-balanced sample of the split: (x (n, 2500, 3) float32 z-scored per lead as the
    model saw them, labels (n, 10) per second, category (n,) = the window's arrhythmia).
    At most `per_class` windows per category."""
    from ..signal_ops import normalize_window
    from .pipeline import load_arrays, window_categories
    segs, labs, _ = load_arrays(split, npy_dir=npy_dir)
    cat = window_categories(labs)
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(K):
        pool = np.flatnonzero(cat == c)
        idx += list(rng.choice(pool, min(per_class, len(pool)), replace=False)) if len(pool) else []
    idx = np.array(sorted(idx))
    x = np.stack([normalize_window(segs[i].astype(np.float32)) for i in idx]).astype(np.float32)
    return x, labs[idx].astype(np.int64), cat[idx]


def per_second(rhythm):
    """(n, 1250, K) -> (n, 10, K): mean probability per second."""
    rhythm = to_current_classes(np.asarray(rhythm))
    n, t, k = rhythm.shape
    return rhythm.reshape(n, rc.OUTPUT_SECONDS, t // rc.OUTPUT_SECONDS, k).mean(axis=2)


@contextlib.contextmanager
def patched(model, name, make_call):
    """Replace layer `name`'s call by make_call(original_call) for the duration of the block.
    Every forward pass built inside the block (eager or a new tf.function) sees the patch."""
    layer = model.get_layer(name)
    orig = layer.call
    object.__setattr__(layer, 'call', make_call(orig))
    try:
        yield layer
    finally:
        object.__delattr__(layer, 'call')


@contextlib.contextmanager
def patches(model, items):
    with contextlib.ExitStack() as stack:
        for name, make_call in items:
            stack.enter_context(patched(model, name, make_call))
        yield


def run(model, x, batch=64, outputs=('rhythm',)):
    """Forward pass in batches -> {output: array}. A new tf.function per call, so patches
    made before it are traced in."""
    f = tf.function(lambda z: model(z, training=False), reduce_retracing=True)
    acc = {k: [] for k in outputs}
    for i in range(0, len(x), batch):
        y = f(tf.constant(x[i:i + batch]))
        for k in outputs:
            acc[k].append(np.asarray(y[k]))
    return {k: np.concatenate(v) for k, v in acc.items()}


def sub_model(model, names):
    """keras.Model from the input to the outputs of the named layers (lists flattened)."""
    import keras
    outs = []
    for n in names:
        o = model.get_layer(n).output
        outs += list(o) if isinstance(o, (list, tuple)) else [o]
    return keras.Model(model.input, outs)


def scores(probs_s, labels):
    """Per-second recall per class, macro F1, mean p(true class) per class on labelled
    seconds (labels < K)."""
    pred = probs_s.argmax(-1)
    ok = labels < K
    t, p = labels[ok], pred[ok]
    cm = np.bincount(t * K + p, minlength=K * K).reshape(K, K)
    out = {}
    f1s = []
    for c, name in enumerate(CLASS_NAMES):
        tp, fn, fp = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
        se = tp / max(tp + fn, 1)
        pp = tp / max(tp + fp, 1)
        f1 = 2 * se * pp / max(se + pp, 1e-9)
        if tp + fn:
            f1s.append(f1)
        sel = labels == c
        out[name] = dict(se=float(se), ppv=float(pp), f1=float(f1), n=int(cm[c].sum()),
                         p_true=float(probs_s[..., c][sel].mean()) if sel.any() else float('nan'))
    out['macro_f1'] = float(np.mean(f1s)) if f1s else float('nan')
    return out


# ---------------------------------------------------------------------------
# 1. Probes
# ---------------------------------------------------------------------------

def _donor_patch(layer_name, index, donor_var):
    """Output `index` of the layer (or the single output) replaced by donor_var."""
    def make(orig):
        def call(inputs, *a, **k):
            out = orig(inputs, *a, **k)
            if isinstance(out, (list, tuple)):
                out = list(out)
                out[index] = tf.cast(donor_var, out[index].dtype)
                return out
            return tf.cast(donor_var, out.dtype)
        return call
    return (layer_name, make)


def _map_patch(layer_name, fn, index=None):
    def make(orig):
        def call(inputs, *a, **k):
            out = orig(inputs, *a, **k)
            if isinstance(out, (list, tuple)):
                out = list(out)
                out[index] = fn(out[index])
                return out
            return fn(out)
        return call
    return (layer_name, make)


def _run_with_donor(model, x, donor_values, layer_name, index, batch=64):
    """Forward pass where the layer's output is the donor's (per window, same order)."""
    shape = (batch,) + donor_values.shape[1:]
    var = tf.Variable(tf.zeros(shape, tf.float32), trainable=False)
    acc = []
    with patches(model, [_donor_patch(layer_name, index, var)]):
        f = tf.function(lambda z: model(z, training=False)['rhythm'])
        for i in range(0, len(x), batch):
            xb, db = x[i:i + batch], donor_values[i:i + batch]
            pad = batch - len(xb)
            if pad:
                xb = np.concatenate([xb, np.zeros((pad,) + xb.shape[1:], xb.dtype)])
                db = np.concatenate([db, np.zeros((pad,) + db.shape[1:], db.dtype)])
            var.assign(db.astype(np.float32))
            acc.append(np.asarray(f(tf.constant(xb)))[:batch - pad])
    return np.concatenate(acc)


def f_waves(n, amplitude, rng, fs=rc.SAMPLING_RATE, seconds=rc.OUTPUT_SECONDS):
    """(n, 2500, 3) synthetic fibrillatory waves: per lead a 4-8 Hz carrier whose frequency
    and amplitude wander slowly (as f waves do), random phase per lead. `amplitude` is a
    scalar or an (n, 3) array (per window and lead)."""
    amplitude = np.broadcast_to(np.asarray(amplitude, np.float32), (n, 3))
    t = np.arange(fs * seconds) / fs
    out = np.zeros((n, len(t), 3), np.float32)
    for i in range(n):
        for c in range(3):
            f0 = rng.uniform(4.0, 8.0)
            drift = np.cumsum(rng.normal(0, 0.02, len(t)))
            am = 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6.28))
            out[i, :, c] = amplitude[i, c] * am * np.sin(2 * np.pi * f0 * t + drift + rng.uniform(0, 6.28))
    return out


def probe(model, x, labels, cat, seed=0, batch=64, log=print):
    """All the interventions; returns {name: {'scores': scores(...), 'target': ...}}."""
    rng = np.random.default_rng(seed)
    res = {}
    base = per_second(run(model, x, batch)['rhythm'])
    res['baseline'] = dict(scores=scores(base, labels))
    log(f"  baseline macro F1 {100 * res['baseline']['scores']['macro_f1']:.1f}")

    # knock-outs on every window -----------------------------------------------------------
    knock = {
        'atrial_off': [_map_patch('qrst_cancel', tf.zeros_like, index=0)],
        'beat_cond_off': [_map_patch('beat_to_grid', lambda b: tf.concat(
            [tf.ones_like(b[..., :1]), tf.zeros_like(b[..., 1:])], -1))],
        'lead_uniform': [_map_patch('lead_w_logit', tf.zeros_like),
                         _map_patch('alead_w_logit', tf.zeros_like)],
        'ssm_identity': [(n, (lambda orig: (lambda inputs, *a, **k: inputs)))
                         for n in ssm_layers(model)],
    }
    rr_model = sub_model(model, ['rr_ac'])
    rr = np.concatenate([np.asarray(rr_model(x[i:i + batch], training=False))
                         for i in range(0, len(x), batch)])
    rr_mean = np.repeat(rr.mean(0, keepdims=True), len(x), 0)
    for name, items in knock.items():
        with patches(model, items):
            p = per_second(run(model, x, batch)['rhythm'])
        res[name] = dict(scores=scores(p, labels))
        log(f"  {name:14s} macro F1 {100 * res[name]['scores']['macro_f1']:.1f}")
    p = per_second(_run_with_donor(model, x, rr_mean, 'rr_ac', None, batch))
    res['rr_mean'] = dict(scores=scores(p, labels))
    log(f"  {'rr_mean':14s} macro F1 {100 * res['rr_mean']['scores']['macro_f1']:.1f}")

    # donor swaps: class C windows <- SINUS donors, SINUS windows <- class C donors ------------
    qrst = sub_model(model, ['qrst_cancel'])
    sinus = np.flatnonzero(cat == rc.SINUS)
    swaps = {}
    for c in range(K):
        if c == rc.SINUS:
            continue
        mine = np.flatnonzero(cat == c)
        if len(mine) < 10 or len(sinus) < 10:
            continue
        name = CLASS_NAMES[c]
        for direction, tgt, donors in (('from_sinus', mine, rng.choice(sinus, len(mine))),
                                       ('into_sinus', sinus, rng.choice(mine, len(sinus)))):
            xt, xd = x[tgt], x[donors]
            lt = labels[tgt]
            b = per_second(run(model, xt, batch)['rhythm'])
            res_d = np.concatenate([np.asarray(qrst(xd[i:i + batch], training=False)[0])
                                    for i in range(0, len(xd), batch)])
            pa = per_second(_run_with_donor(model, xt, res_d, 'qrst_cancel', 0, batch))
            pr = per_second(_run_with_donor(model, xt, rr[donors], 'rr_ac', None, batch))
            row = {}
            for kind, p in (('atrial', pa), ('rr', pr)):
                row[kind] = dict(p_class_before=float(b[..., c].mean()),
                                 p_class_after=float(p[..., c].mean()),
                                 argmax_class_before=float((b.argmax(-1) == c).mean()),
                                 argmax_class_after=float((p.argmax(-1) == c).mean()))
            swaps[f"{name}_{direction}"] = dict(target_class=name, n=int(len(tgt)), **row)
            log(f"  swap {name:5s} {direction:10s} n={len(tgt):4d}  p({name}) "
                f"{row['atrial']['p_class_before']:.2f} -> atrial {row['atrial']['p_class_after']:.2f}"
                f" | rr {row['rr']['p_class_after']:.2f}")
            del lt
    res['swaps'] = swaps

    # f-wave dose-response on SINUS windows (signal domain) ----------------------------------
    from ..signal_ops import normalize_window
    af = rc.CLASS_NAMES.index('AFIB')
    xs = x[sinus]
    peak = np.percentile(np.abs(xs), 99.5, axis=1)               # (n, 3) QRS-scale per lead
    dose = []
    # f waves are 0.05-0.1 mV against a 1-2 mV QRS: 3-10 % of the QRS peak is the real range
    for amp in (0.0, 0.03, 0.05, 0.1, 0.2):
        if amp:
            xi = xs + f_waves(len(xs), amp * peak, rng)
            xi = np.stack([normalize_window(w) for w in xi]).astype(np.float32)
        else:
            xi = xs
        p = per_second(run(model, xi, batch)['rhythm'])
        # the same windows with the atrial branch blind: whatever AF is left came through
        # the V branch, which reads the raw signal (f waves / noise on the baseline)
        with patches(model, [_map_patch('qrst_cancel', tf.zeros_like, index=0)]):
            pb = per_second(run(model, xi, batch)['rhythm'])
        dose.append(dict(amplitude_of_qrs=amp, p_afib=float(p[..., af].mean()),
                         afib_seconds=float((p.argmax(-1) == af).mean()),
                         p_afib_atrial_off=float(pb[..., af].mean()),
                         afib_seconds_atrial_off=float((pb.argmax(-1) == af).mean())))
        log(f"  f_inject {100 * amp:4.0f} % of QRS: p(AFIB) {dose[-1]['p_afib']:.3f}, "
            f"AFIB argmax {100 * dose[-1]['afib_seconds']:.1f} % of SINUS seconds; atrial "
            f"branch blind: {dose[-1]['p_afib_atrial_off']:.3f} / "
            f"{100 * dose[-1]['afib_seconds_atrial_off']:.1f} %")
    res['f_inject'] = dose
    return res


# ---------------------------------------------------------------------------
# 2. Layers
# ---------------------------------------------------------------------------

def _entropy(w, axis):
    return -(w * np.log(np.clip(w, 1e-9, 1))).sum(axis) / np.log(w.shape[axis])


def lead_weights(model, x, batch=64):
    """Lead-weight statistics of both stems, agreement with 'channel', and the shift when one
    lead is drowned in noise (the weights should move away from it)."""
    sm = sub_model(model, ['lead_w_logit', 'alead_w_logit', 'channel'])

    def collect(xx):
        outs = [sm(xx[i:i + batch], training=False) for i in range(0, len(xx), batch)]
        return [np.concatenate([np.asarray(o[j]) for o in outs]) for j in range(3)]
    lw, aw, ch = collect(x)
    sw = lambda z: np.exp(z[..., 0]) / np.exp(z[..., 0]).sum(1, keepdims=True)  # noqa: E731
    w_v, w_a = sw(lw), sw(aw)                                     # (n, 3, 50)
    out = {}
    for name, w in (('v_stem', w_v), ('a_stem', w_a)):
        out[name] = dict(entropy=float(_entropy(w, 1).mean()), max_weight=float(w.max(1).mean()),
                         mean_per_lead=[float(v) for v in w.mean((0, 2))])
    # agreement: per 2 s segment, the lead with most weight vs the clean lead the 'channel'
    # output names (segments it calls NOISE left out)
    seg = w_v.reshape(len(w_v), 3, rc.NOISE_SEGMENTS, -1).mean(-1).argmax(1)   # (n, 5)
    chs = ch.argmax(-1)                                            # (n, 5), 0 = NOISE
    ok = chs > 0
    out['v_stem']['agree_with_channel'] = float((seg[ok] == chs[ok] - 1).mean()) if ok.any() else None
    segA = w_a.reshape(len(w_a), 3, rc.NOISE_SEGMENTS, -1).mean(-1).argmax(1)
    out['a_stem']['agree_with_channel'] = float((segA[ok] == chs[ok] - 1).mean()) if ok.any() else None
    # one lead (0) corrupted with white noise at 0 dB of its own power, then re-z-scored
    from ..signal_ops import normalize_window
    rng = np.random.default_rng(1)
    xn = x.copy()
    xn[..., 0] = xn[..., 0] + rng.normal(0, 1.0, xn[..., 0].shape)
    xn = np.stack([normalize_window(w) for w in xn]).astype(np.float32)
    lw2, aw2, ch2 = collect(xn)
    out['noisy_lead0'] = dict(v_weight_lead0_before=float(w_v[:, 0].mean()),
                              v_weight_lead0_after=float(sw(lw2)[:, 0].mean()),
                              a_weight_lead0_before=float(w_a[:, 0].mean()),
                              a_weight_lead0_after=float(sw(aw2)[:, 0].mean()),
                              channel_picks_lead0_before=float((chs == 1).mean()),
                              channel_picks_lead0_after=float((ch2.argmax(-1) == 1).mean()))
    return out


def qrst_stats(model, x, cat, batch=64):
    """Per class: residual RMS outside the QRS mask relative to the (125 Hz) input RMS, the
    residual peak relative to the input peak, and the share of the residual's power (outside
    the QRS mask) in 4-9 Hz (f waves) vs 0.5-3 Hz (P / T scale)."""
    sm = sub_model(model, ['qrst_cancel'])
    res, msk = [], []
    for i in range(0, len(x), batch):
        r, q = sm(x[i:i + batch], training=False)
        res.append(np.asarray(r))
        msk.append(np.asarray(q)[..., 0])
    res, msk = np.concatenate(res), np.concatenate(msk)
    xd = x.reshape(len(x), res.shape[1], -1, 3).mean(2)
    fs = res.shape[1] / rc.OUTPUT_SECONDS
    freqs = np.fft.rfftfreq(res.shape[1], 1 / fs)
    band_f = (freqs >= 4) & (freqs <= 9)
    band_p = (freqs >= 0.5) & (freqs <= 3)
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        sel = cat == c
        if not sel.any():
            continue
        r, m, xx = res[sel], msk[sel], xd[sel]
        keep = (m < 0.5)[..., None]
        rms_r = np.sqrt(((r * keep) ** 2).sum((1, 2)) / np.maximum(keep.sum((1, 2)) * 3, 1))
        rms_x = np.sqrt((xx ** 2).mean((1, 2)))
        spec = np.abs(np.fft.rfft(r * keep, axis=1)) ** 2          # (n, F, 3)
        tot = spec[:, (freqs >= 0.5) & (freqs <= 40)].sum((1, 2)) + 1e-9
        out[name] = dict(n=int(sel.sum()),
                         residual_rms_ratio=float(np.median(rms_r / (rms_x + 1e-9))),
                         residual_peak_ratio=float(np.median(np.abs(r).max((1, 2)) /
                                                             (np.abs(xx).max((1, 2)) + 1e-9))),
                         share_4_9hz=float(np.median(spec[:, band_f].sum((1, 2)) / tot)),
                         share_0p5_3hz=float(np.median(spec[:, band_p].sum((1, 2)) / tot)))
    return out


def ssm_kernels(model):
    """Per DiagSSM1D layer: the 90 %-L1 memory of each channel's kernel (forward and backward)
    in seconds, and the share of channels whose memory reaches the kernel's end (truncated:
    a longer kernel_len could be used)."""
    out = {}
    for name in ssm_layers(model):
        layer = model.get_layer(name)
        k = np.abs(layer.kernel_numpy())                           # (2, C, L)
        cum = np.cumsum(k, -1) / (k.sum(-1, keepdims=True) + 1e-12)
        mem = (cum < 0.9).sum(-1) + 1                              # steps to 90 %
        L = k.shape[-1]
        tail = k[..., -L // 8:].sum(-1) / (k.sum(-1) + 1e-12)      # mass in the last 1/8
        out[name] = dict(kernel_len=int(L), kernel_seconds=float(L * GRID_SECONDS),
                         memory_median_s=float(np.median(mem) * GRID_SECONDS),
                         memory_p90_s=float(np.percentile(mem, 90) * GRID_SECONDS),
                         share_memory_ge_1s=float((mem * GRID_SECONDS >= 1.0).mean()),
                         share_memory_ge_2p5s=float((mem * GRID_SECONDS >= 2.5).mean()),
                         share_truncated=float((tail > 0.1).mean()))
    return out


def dead_units(model, x, batch=16, active_floor=0.01):
    """ReLU layers: share of channels positive at fewer than `active_floor` of positions over
    the sample (dead or nearly) - capacity that a smaller width would keep."""
    names = [l.name for l in model.layers if l.name.endswith('_relu')]
    sm = sub_model(model, names)
    pos = None
    count = 0
    for i in range(0, len(x), batch):
        outs = sm(x[i:i + batch], training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        cur = [np.asarray(o > 0).reshape(-1, o.shape[-1]).sum(0) for o in outs]
        sizes = [int(np.prod(o.shape[:-1])) for o in outs]
        pos = cur if pos is None else [a + b for a, b in zip(pos, cur)]
        count = [s + (count[j] if count else 0) for j, s in enumerate(sizes)] if count else sizes
    out = {}
    for n, p, c in zip(names, pos, count):
        frac = p / max(c, 1)
        out[n] = dict(channels=int(len(frac)), dead_share=float((frac < active_floor).mean()))
    return out


def branch_shares(model, x, labels, cat, batch=16):
    """Gradient x activation at the concatenations where the branches meet, for log p(class)
    summed over the window's seconds of its category class: the |share| each input group
    contributes, per class. A branch with ~0 share does not take part in that decision."""
    concat_names = [n for n in BRANCH_CONCATS if n in {l.name for l in model.layers}]
    widths = {n: [int(t.shape[-1]) for t in model.get_layer(n).input] for n in concat_names}
    store = {}
    deltas = {}

    def make(name):
        def mk(orig):
            def call(inputs, *a, **k):
                out = orig(inputs, *a, **k)
                store[name] = out
                return out + deltas[name]
            return call
        return mk

    out = {CLASS_NAMES[c]: {n: np.zeros(len(widths[n])) for n in concat_names} for c in range(K)}
    with patches(model, [(n, make(n)) for n in concat_names]):
        for c in range(K):
            idx = np.flatnonzero(cat == c)[:128]
            for i in range(0, len(idx), batch):
                b = idx[i:i + batch]
                xb = tf.constant(x[b])
                lab = labels[b]
                for n in concat_names:
                    shape = (len(b),) + tuple(model.get_layer(n).output.shape[1:])
                    deltas[n] = tf.Variable(tf.zeros(shape))
                with tf.GradientTape() as tape:
                    y = model(xb, training=False)['rhythm']           # (B, 1250, K)
                    ps = tf.reshape(y, (len(b), rc.OUTPUT_SECONDS, -1, y.shape[-1]))
                    ps = tf.reduce_mean(ps, 2)[..., c]                # (B, 10)
                    mask = tf.constant((lab == c).astype(np.float32))
                    obj = tf.reduce_sum(tf.math.log(ps + 1e-6) * mask)
                grads = tape.gradient(obj, [deltas[n] for n in concat_names])
                for n, g in zip(concat_names, grads):
                    ga = np.abs(np.asarray(g) * np.asarray(store[n]))  # (B, T, C)
                    edges = np.cumsum([0] + widths[n])
                    out[CLASS_NAMES[c]][n] += [ga[..., a:e].sum() for a, e in zip(edges, edges[1:])]
    shares = {}
    for cname, per in out.items():
        shares[cname] = {}
        for n, v in per.items():
            tot = v.sum()
            if tot > 0:
                shares[cname][n] = {lab: float(s / tot) for lab, s in zip(BRANCH_CONCATS[n], v)}
    return shares


def position_profile(model, x, labels, batch=64):
    """Per-second scores by position inside the 10 s window (0 = first second): how much
    a second loses when it sits at a window edge (the input for rc.PREDICT_TAPER_FLOOR / hop)."""
    p = per_second(run(model, x, batch)['rhythm'])
    rows = []
    for k in range(rc.OUTPUT_SECONDS):
        sc = scores(p[:, k:k + 1], labels[:, k:k + 1])
        rows.append(dict(position=k, macro_f1=sc['macro_f1'],
                         **{f"F1_{n}": sc[n]['f1'] for n in CLASS_NAMES}))
    return rows


def lead_order_invariance(model, x, batch=64):
    """Max |difference| of the rhythm output between the windows and their first two leads
    swapped, and with every lead's sign flipped - which test-time variants can change it."""
    a = run(model, x[:128], batch)['rhythm']
    b = run(model, x[:128][..., [1, 0, 2]], batch)['rhythm']
    c = run(model, -x[:128], batch)['rhythm']
    return dict(swap_max_diff=float(np.abs(a - b).max()), flip_max_diff=float(np.abs(a - c).max()))


def layer_report(model, x, labels, cat, log=print):
    out = {}
    log("  position in the window ...")
    out['position'] = position_profile(model, x, labels)
    out['tta'] = lead_order_invariance(model, x)
    log("  lead weights ...")
    out['lead_weights'] = lead_weights(model, x)
    log("  QRST cancellation ...")
    out['qrst'] = qrst_stats(model, x, cat)
    log("  SSM kernels ...")
    out['ssm'] = ssm_kernels(model)
    log("  dead ReLU channels ...")
    out['dead'] = dead_units(model, x[:256])
    log("  branch shares (gradient x activation) ...")
    out['branches'] = branch_shares(model, x, labels, cat)
    return out


# ---------------------------------------------------------------------------
# 3. Inference-time experiments
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def layer_attrs(model, name, **attrs):
    layer = model.get_layer(name)
    old = {k: getattr(layer, k) for k in attrs}
    for k, v in attrs.items():
        object.__setattr__(layer, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            object.__setattr__(layer, k, v)


def experiment(model, x, labels, log=print, batch=64):
    """One-factor sweeps around the trained values; rows {knob, value, macro_f1, per class F1}."""
    rows = []

    def score(knob, value):
        p = per_second(run(model, x, batch)['rhythm'])
        s = scores(p, labels)
        rows.append(dict(knob=knob, value=value, macro_f1=s['macro_f1'],
                         **{f"F1_{n}": s[n]['f1'] for n in CLASS_NAMES}))
        log(f"  {knob:16s} {str(value):>8s}  macro F1 {100 * s['macro_f1']:.2f}  " +
            ' '.join(f"{n} {100 * s[n]['f1']:.1f}" for n in CLASS_NAMES))

    q = model.get_layer('qrst_cancel')
    score('trained', '-')
    for v in (0.06, 0.08, 0.12, 0.14):
        with layer_attrs(model, 'qrst_cancel', pre=v):
            score('qrst_pre_s', v)
    for v in (0.35, 0.40, 0.50, 0.55):
        with layer_attrs(model, 'qrst_cancel', post=v):
            score('qrst_post_s', v)
    for v in (0.2, 0.4, 0.5):
        with layer_attrs(model, 'qrst_cancel', min_prob=v):
            score('qrst_min_prob', v)
    del q
    for temp in (0.5, 2.0):
        scale = 1.0 / temp
        with patches(model, [_map_patch('lead_w_logit', lambda z, s=scale: z * s),
                             _map_patch('alead_w_logit', lambda z, s=scale: z * s)]):
            score('lead_temperature', temp)
    for frac in (0.25, 0.5):
        def make(orig, frac=frac):
            def call(inputs, *a, **k):
                layer = call.layer
                n = max(2, int(layer.kernel_len * frac))
                kf = layer._kernel(layer.fwd)[:, :n]
                y = layer._causal_conv(inputs, kf)
                if layer.bidirectional:
                    kb = layer._kernel(layer.bwd)[:, :n]
                    y += tf.reverse(layer._causal_conv(tf.reverse(inputs, [1]), kb), [1])
                return y
            return call
        items = []
        for n in ssm_layers(model):
            mk = (lambda orig, n=n, make=make: _bind(make(orig), model.get_layer(n)))
            items.append((n, mk))
        with patches(model, items):
            score('ssm_kernel_frac', frac)
    return rows


def _bind(fn, layer):
    fn.layer = layer
    return fn


# ---------------------------------------------------------------------------
# 4. Explain one record second
# ---------------------------------------------------------------------------

def integrated_gradients(model, x, cls, seconds, steps=32):
    """IG of sum over `seconds` of log p(cls) w.r.t. the (2500, 3) input, zero baseline."""
    x = tf.constant(x[None], tf.float32)
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1)[1:], (-1, 1, 1))
    total = tf.zeros_like(x)
    sel = np.zeros(rc.OUTPUT_SECONDS, np.float32)
    sel[list(seconds)] = 1
    sel = tf.constant(sel)
    for a in tf.split(alphas, max(1, steps // 8)):
        xi = a * x
        with tf.GradientTape() as tape:
            tape.watch(xi)
            y = model(xi, training=False)['rhythm']
            ps = tf.reduce_mean(tf.reshape(y, (y.shape[0], rc.OUTPUT_SECONDS, -1, y.shape[-1])), 2)
            obj = tf.reduce_sum(tf.math.log(ps[..., cls] + 1e-6) * sel)
        total += tf.reduce_sum(tape.gradient(obj, xi), 0, keepdims=True)
    return (x * total / steps).numpy()[0]


def explain(model, record_path, second, out_png, cls=None):
    """One 10 s window around `second` of a WFDB record: IG for `cls` (default the predicted
    class at that second) and a figure with the signal coloured by |IG|, the beats, the
    rhythm probabilities, the atrial residual and the lead weights."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..signal_ops import normalize_window
    from .build import read_leads
    leads, _ = read_leads(record_path)
    total_s = len(leads) // rc.SAMPLING_RATE
    start = int(min(max(0, second - rc.OUTPUT_SECONDS // 2), max(0, total_s - rc.OUTPUT_SECONDS)))
    w = leads[start * rc.SAMPLING_RATE:(start + rc.OUTPUT_SECONDS) * rc.SAMPLING_RATE]
    x = normalize_window(w.astype(np.float32)).astype(np.float32)
    sm = sub_model(model, ['rhythm', 'beat', 'qrst_cancel', 'lead_w_logit', 'alead_w_logit'])
    rhythm, beat, resid, qrs, lw, aw = [np.asarray(o)[0] for o in sm(x[None], training=False)]
    rhythm = to_current_classes(rhythm[None])[0]
    ps = rhythm.reshape(rc.OUTPUT_SECONDS, -1, K).mean(1)
    rel = int(second - start)
    cls = int(ps[rel].argmax()) if cls is None else int(cls)
    ig = integrated_gradients(model, x, cls, [rel])
    from .beats import pick_beats
    bt = pick_beats(beat, rc.BEAT_STEPS // rc.OUTPUT_SECONDS)

    t = np.arange(len(x)) / rc.SAMPLING_RATE
    fig, ax = plt.subplots(6, 1, figsize=(16, 13), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2, 2, 1.4, 1.4, 1]))
    imp = np.abs(ig)
    imp = np.convolve(imp.sum(1), np.ones(13) / 13, mode='same')
    imp = imp / (imp.max() + 1e-9)
    for c in range(3):
        a = ax[c]
        a.plot(t, x[:, c], color='0.2', lw=0.7)
        a.scatter(t, x[:, c], c=np.abs(ig[:, c]), cmap='Reds', s=2,
                  vmin=0, vmax=np.abs(ig).max() + 1e-9)
        a.set_ylabel(f"lead {c + 1}")
        for tb, cb in zip(bt['t'], bt['cls']):
            a.axvline(tb, color={1: 'tab:green', 2: 'tab:orange', 3: 'tab:red'}[int(cb)],
                      lw=0.5, alpha=0.5)
    ax[0].set_title(f"{os.path.basename(record_path)}  {start}-{start + rc.OUTPUT_SECONDS} s  "
                    f"IG for {CLASS_NAMES[cls]} at second {second} (p={ps[rel, cls]:.2f})  "
                    f"beats: green N, orange S, red V")
    tr = np.arange(len(rhythm)) / (len(rhythm) / rc.OUTPUT_SECONDS)
    for k in range(K):
        ax[3].plot(tr, rhythm[:, k], label=CLASS_NAMES[k])
    ax[3].axvspan(rel, rel + 1, color='yellow', alpha=0.3)
    ax[3].legend(loc='upper right', ncol=K, fontsize=8)
    ax[3].set_ylabel('p(rhythm)')
    tq = np.arange(len(resid)) / (len(resid) / rc.OUTPUT_SECONDS)
    for c in range(3):
        ax[4].plot(tq, resid[:, c] + 3 * c, lw=0.6)
    ax[4].fill_between(tq, -1.5, 7.5, where=qrs[:, 0] > 0.5, color='0.85')
    ax[4].set_ylabel('atrial residual')
    sw = lambda z: np.exp(z[..., 0]) / np.exp(z[..., 0]).sum(0, keepdims=True)  # noqa: E731
    tl = np.arange(lw.shape[1]) * rc.OUTPUT_SECONDS / lw.shape[1]
    for c in range(3):
        ax[5].plot(tl, sw(lw)[c], label=f"V w{c + 1}")
        ax[5].plot(tl, sw(aw)[c], '--', label=f"A w{c + 1}")
    ax[5].plot(t, imp, color='k', lw=0.8, label='|IG| (all leads)')
    ax[5].legend(loc='upper right', ncol=7, fontsize=7)
    ax[5].set_ylabel('lead w / IG')
    ax[5].set_xlabel('s (window)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    share_atrial_qrs = None
    return dict(record=record_path, window_start=start, second=second, cls=CLASS_NAMES[cls],
                p=float(ps[rel, cls]), probs_at_second={n: float(v) for n, v in zip(CLASS_NAMES, ps[rel])},
                png=out_png, beats=int(len(bt['t'])), atrial_qrs=share_atrial_qrs)


# ---------------------------------------------------------------------------
# 5. Suggestions
# ---------------------------------------------------------------------------

def suggest(report):
    """Rule-based reading of the measurements -> [(priority, finding, experiment)], priority
    1 = most likely to move EC57."""
    s = []
    pr = report.get('probe', {})
    base = pr.get('baseline', {}).get('scores', {})

    def se_drop(name, cls):
        b = base.get(cls, {}).get('se')
        a = pr.get(name, {}).get('scores', {}).get(cls, {}).get('se')
        return None if b is None or a is None else b - a

    def mf1_drop(name):
        a = pr.get(name, {}).get('scores', {}).get('macro_f1')
        return None if a is None or not base else base['macro_f1'] - a

    sw = pr.get('swaps', {})
    moved = {}
    for key, v in sw.items():
        moved[key] = (v['atrial']['p_class_after'] - v['atrial']['p_class_before'],
                      v['rr']['p_class_after'] - v['rr']['p_class_before'])
    if moved:
        biggest = max(abs(d) for pair in moved.values() for d in pair)
        if biggest < 0.10:
            off = {n: se_drop('atrial_off', n) or 0.0 for n in CLASS_NAMES}
            hit = ', '.join(f"{n} {-100 * d:+.0f}" for n, d in off.items() if abs(d) >= 0.05)
            s.append((1, f"no class-specific evidence in the stage inputs: another class's atrial "
                         f"residual or R-R descriptor moves p(class) by at most {biggest:.2f}"
                         + (f", although blinding the atrial branch changes recall ({hit} pts)"
                            if hit else ""),
                      "the A branch carries non-specific information (it needs A residual, not "
                      "THIS window's P / f waves) and the class is formed from V-branch features "
                      "+ SSM context. Give the A branch its own target (P/QRS/T/f segmentation "
                      "head, md_atrial_activity.md 3.3) or train with V-branch dropout so AF / "
                      "AVB must be read from atrial activity; re-run this probe to verify"))
        for key, (da, dr) in sorted(moved.items()):
            if max(abs(da), abs(dr)) >= 0.10:
                src = 'atrial residual' if abs(da) >= abs(dr) else 'R-R descriptor'
                s.append((2, f"{key}: p moves {da:+.2f} with the donor's atrial residual, "
                             f"{dr:+.2f} with its R-R", f"{key.split('_')[0]} rests on the {src}"))
    d_atr = mf1_drop('atrial_off')
    d_rr = mf1_drop('rr_mean')
    if d_atr is not None and d_rr is not None:
        s.append((3, f"knock-outs: atrial branch blind -{100 * d_atr:.1f} macro-F1 pts, R-R "
                     f"descriptor at its mean {-100 * d_rr:+.1f} pts, beat condition off "
                     f"-{100 * (mf1_drop('beat_cond_off') or 0):.1f}",
                  "a knock-out that costs < 1 pt is a module the model does not use; an R-R "
                  "descriptor whose removal HELPS is noise to the decoder - drop rr_film or "
                  "regularise it on retraining" if d_rr < 0 else
                  "a knock-out that costs < 1 pt is a module the model does not use"))
    fi = pr.get('f_inject', [])
    real = [d for d in fi if 0 < d['amplitude_of_qrs'] <= 0.1]
    if real:
        top = max(real, key=lambda d: d['amplitude_of_qrs'])
        s.append((2 if top['afib_seconds'] < 0.2 else 4,
                  f"f waves at {100 * top['amplitude_of_qrs']:.0f} % of the QRS on regular sinus "
                  f"windows: p(AFIB) {fi[0]['p_afib']:.3f} -> {top['p_afib']:.3f}, AFIB on "
                  f"{100 * top['afib_seconds']:.0f} % of seconds",
                  "f waves of realistic size do not make AF on their own (expected while R-R "
                  "is regular); f-wave augmentation on AF windows with R-R made regular would "
                  "teach the model to use them" if top['afib_seconds'] < 0.2 else
                  ("the f-wave evidence enters through the V branch (raw signal), not the "
                   "atrial branch: AF stays at "
                   f"{100 * top.get('afib_seconds_atrial_off', 0):.0f} % with the A branch blind. "
                   "Baseline noise of f-wave frequency will read as AF - the source of noisy-record "
                   "AF false positives. Train with 4-9 Hz baseline noise on SINUS windows (noise "
                   "augmentation band) and/or band-stop the V-branch input above 3 Hz outside QRS"
                   if top.get('afib_seconds_atrial_off', 0) > 0.5 * top['afib_seconds'] else
                   "the atrial branch reads f waves")))
    ss = mf1_drop('ssm_identity')
    lay = report.get('layers', {})
    if ss is not None:
        trunc = max((v['share_truncated'] for v in lay.get('ssm', {}).values()), default=0)
        s.append((3, f"long-range context (every SSM -> identity) is worth {100 * ss:.0f} macro-F1 "
                     f"pts; at most {100 * trunc:.0f} % of SSM channels reach the kernel end",
                  "raise kernel_len (128 -> 192/250 steps = 7.7/10 s) on retraining" if trunc > 0.2
                  else "kernel_len (5.1 s) is not binding; keep it"))
    lw = lay.get('lead_weights', {})
    if lw:
        nl = lw['noisy_lead0']
        dv = nl['v_weight_lead0_before'] - nl['v_weight_lead0_after']
        da = nl['a_weight_lead0_before'] - nl['a_weight_lead0_after']
        if da < 0.1 <= dv:
            s.append((2, f"a lead drowned in noise loses {100 * dv:.0f} pts of V-stem weight but "
                         f"only {100 * da:.0f} pts of A-stem weight",
                      "the atrial branch keeps reading a noisy lead - noise there looks like f "
                      "waves (AF false positives on noisy records, afdb 04043 / nstdb). Share the "
                      "V-stem lead logits with the A stem, or supervise alead_w with the "
                      "'channel' target"))
        elif dv < 0.1:
            s.append((2, f"lead weights ignore a noisy lead (V -{100 * dv:.0f} pts)",
                      "supervise the lead weights with the 'channel' target"))
    q = lay.get('qrst', {})
    if 'AFIB' in q and 'SINUS' in q:
        r = q['AFIB']['share_4_9hz'] / max(q['SINUS']['share_4_9hz'], 1e-9)
        s.append((4, f"atrial residual: 4-9 Hz share AFIB / SINUS = {r:.2f}; VT residual peak "
                     f"{q.get('VT', {}).get('residual_peak_ratio', float('nan')):.2f} of the input",
                  "f waves survive the cancellation" if r > 1.3 else
                  "f waves do not stand out in the residual: per-beat template / shorter post"))
    dead = lay.get('dead', {})
    if dead:
        heavy = sorted(((v['dead_share'], n) for n, v in dead.items()), reverse=True)[:3]
        if heavy and heavy[0][0] > 0.2:
            s.append((4, "dead ReLU channels: " + ', '.join(f"{n} {100 * d:.0f} %" for d, n in heavy),
                      "narrow those layers or use SiLU on retraining"))
        else:
            s.append((5, f"no dead layers (worst {heavy[0][1]} {100 * heavy[0][0]:.0f} %)",
                      "capacity is used; no pruning lever"))
    ex = report.get('experiment', [])
    if ex:
        t0 = next(r for r in ex if r['knob'] == 'trained')
        best = max(ex, key=lambda r: r['macro_f1'])
        if best['knob'] != 'trained' and best['macro_f1'] - t0['macro_f1'] > 0.005:
            gains = ', '.join(f"{n} {100 * (best[f'F1_{n}'] - t0[f'F1_{n}']):+.1f}"
                              for n in CLASS_NAMES)
            s.append((1, f"inference knob {best['knob']}={best['value']}: macro F1 "
                         f"{100 * best['macro_f1']:.2f} vs {100 * t0['macro_f1']:.2f} ({gains})",
                      "confirm on the validation records + EC57 with eval_rhythm.py "
                      "(XAI_QRST_* overrides) before adopting; if it holds, set the trained "
                      "value in dualunet.py for the next training run"))
        else:
            s.append((5, "no inference-time knob beats the trained setting by > 0.5 pt",
                      "the trained QRST / lead / SSM settings are at their optimum"))
    pos = lay.get('position', [])
    if pos:
        mid = max(r['macro_f1'] for r in pos)
        edge = min(pos[0]['macro_f1'], pos[-1]['macro_f1'])
        if mid - edge > 0.02:
            s.append((1, f"edge effect: per-second macro F1 {100 * mid:.1f} at the best position "
                         f"vs {100 * edge:.1f} at a window edge",
                      "inference only: hop 2 s + position-weighted averaging "
                      "(rc.PREDICT_TAPER_FLOOR, eval_rhythm.py PREDICT_TAPER) so every second is "
                      "decided by windows where it is central; tune and score on validation first"))
    tta = lay.get('tta', {})
    if tta and tta['swap_max_diff'] < 1e-4:
        s.append((3, f"the lead-swap test-time variant changes nothing (max |dp| "
                     f"{tta['swap_max_diff']:.1e}): the lead pooling is order-invariant",
                  "drop 'swap' from PREDICT_TTA for the dual U-Net - a third less inference, "
                  "spend it on the smaller hop"))
    es = report.get('error_sources') or {}
    for c, v in es.items():
        pp, mo = v['fp_pp'] + v['fn_pp'], v['fp_model'] + v['fn_model']
        if pp + mo == 0:
            continue
        if pp > mo:
            worst = sorted(v['per_record'].items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:3]
            s.append((1, f"{c}: {pp} of {pp + mo} EC57 error seconds are made by the "
                         f"post-processing (FP {v['fp_pp']} s added, FN {v['fn_pp']} s removed "
                         f"against the raw model), worst " +
                         ', '.join(f"{k} {a}/{b}" for k, (a, b, _, _) in worst),
                      "change the decode / beat-PP rule for this class first (tune.py grid on "
                      "validation); the model is already right on these seconds"))
        else:
            s.append((2, f"{c}: {mo} of {pp + mo} EC57 error seconds come from the model "
                         f"(FP {v['fp_model']} s, FN {v['fn_model']} s)",
                      "post-processing cannot fix these - data / training lever for this class"))
    return sorted(s, key=lambda r: r[0])


# ---------------------------------------------------------------------------
# EC57 error seconds -> explain list
# ---------------------------------------------------------------------------

def ec57_error_seconds(ec57_out, db='mitdb', classes=('AFIB', 'VT', 'SVT', 'AVB'), top=2,
                       src_dir=None):
    """The longest false-positive and false-negative stretch of each class per database, from
    the per-class reference / hypothesis files an EC57 run left in <ec57_out>/_ann/<db>/.
    Returns [(record path, middle second, class index, 'FP'|'FN', length s)]."""
    import wfdb
    from .. import config as base_config
    ann = os.path.join(ec57_out, '_ann', db)
    src_dir = src_dir or os.path.join(base_config.PHYSIONET_DIR, db)
    found = []
    for cname in classes:
        ref_ext, hyp_ext = rc.class_extensions(cname)
        stretches = []
        for f in sorted(os.listdir(ann)):
            if not f.endswith('.' + hyp_ext):
                continue
            name = f[:-len(hyp_ext) - 1]
            if not os.path.exists(os.path.join(ann, f"{name}.{ref_ext}")):
                continue
            n = wfdb.rdheader(os.path.join(src_dir, name)).sig_len // \
                int(wfdb.rdheader(os.path.join(src_dir, name)).fs)

            def mask(ext, codes=('(AFIB',)):
                a = wfdb.rdann(os.path.join(ann, name), ext)
                m = np.zeros(n, bool)
                marks = list(zip(a.sample / a.fs, [(x or '').split('\x00')[0] for x in a.aux_note]))
                for (s0, c), (s1, _) in zip(marks, marks[1:] + [(n, None)]):
                    if c in codes:
                        m[int(s0):int(np.ceil(s1))] = True
                return m
            # flutter counts as AF (epicmp -x): an AFIB call during reference (AFL is no error
            # either way, so (AFL is left out of both the FP and the FN search
            afl = mask(ref_ext, ('(AFL',)) if cname == 'AFIB' else np.zeros(n, bool)
            r, h = mask(ref_ext), mask(hyp_ext)
            for kind, m in (('FP', h & ~r & ~afl), ('FN', r & ~h)):
                edges = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
                for a0, b0 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    stretches.append((b0 - a0, kind, name, (a0 + b0) // 2))
        for kind in ('FP', 'FN'):
            for length, _, name, mid in sorted([x for x in stretches if x[1] == kind],
                                               reverse=True)[:top]:
                found.append((os.path.join(src_dir, name), int(mid), CLASS_NAMES.index(cname),
                              kind, int(length)))
    return found


def error_sources(ec57_out, db='mitdb', classes=('AFIB', 'SVT', 'VT', 'AVB'), src_dir=None):
    """Split every EC57 error second into MODEL vs POST-PROCESS, from the record-level
    probabilities the decoder saw (<ec57_out>/_ann/<db>/<rec>.npz) and the per-class reference
    / hypothesis files of the same run:

        FP second (hyp & ~ref): 'model'  if the raw per-second argmax (prior-corrected with the
                                          run's class_scale) is the class, else 'post-process'
        FN second (ref & ~hyp): 'model'  if the raw argmax is NOT the class, else 'post-process'

    Reference (AFL seconds are left out of the AFIB rows (flutter counts as AF). Returns
    {class: {fp_model, fp_pp, fn_model, fn_pp, ref, hyp, per_record: {rec: (fp_pp, fn_pp)}}}."""
    import wfdb
    from .. import config as base_config
    from .ec57 import load_probs
    ann = os.path.join(ec57_out, '_ann', db)
    src_dir = src_dir or os.path.join(base_config.PHYSIONET_DIR, db)
    scale = {}
    used = os.path.join(ec57_out, 'decode_used.json')
    if os.path.exists(used):
        with open(used) as f:
            scale = json.load(f).get('class_scale') or {}
    sv = np.array([scale.get(n, 1.0) for n in CLASS_NAMES], np.float32)
    out = {c: dict(fp_model=0, fp_pp=0, fn_model=0, fn_pp=0, ref=0, hyp=0, per_record={})
           for c in classes}
    for f in sorted(os.listdir(ann)):
        if not f.endswith('.npz'):
            continue
        name = f[:-4]
        r, _p, fs, sl, hz = load_probs(ann, name)
        n = len(r) // hz
        ps = r[:n * hz].reshape(n, hz, K).mean(1) * sv
        raw = ps.argmax(-1)
        for cname in classes:
            ref_ext, hyp_ext = rc.class_extensions(cname)
            if not os.path.exists(os.path.join(ann, f"{name}.{hyp_ext}")):
                continue

            def mask(ext, codes=('(AFIB',)):
                a = wfdb.rdann(os.path.join(ann, name), ext)
                m = np.zeros(n, bool)
                marks = list(zip(a.sample / a.fs, [(x or '').split('\x00')[0] for x in a.aux_note]))
                for (s0, c), (s1, _) in zip(marks, marks[1:] + [(n, None)]):
                    if c in codes:
                        m[int(s0):int(np.ceil(s1))] = True
                return m
            ref, hyp = mask(ref_ext), mask(hyp_ext)
            afl = mask(ref_ext, ('(AFL',)) if cname == 'AFIB' else np.zeros(n, bool)
            ci = CLASS_NAMES.index(cname)
            said = raw == ci
            fp, fn = hyp & ~ref & ~afl, ref & ~hyp
            o = out[cname]
            o['ref'] += int(ref.sum())
            o['hyp'] += int((hyp & ~afl).sum())
            o['fp_model'] += int((fp & said).sum())
            o['fp_pp'] += int((fp & ~said).sum())
            o['fn_model'] += int((fn & ~said).sum())
            o['fn_pp'] += int((fn & said).sum())
            pr = (int((fp & ~said).sum()), int((fn & said).sum()), int((fp & said).sum()),
                  int((fn & ~said).sum()))
            if any(pr):
                o['per_record'][name] = pr
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(v):
    return '-' if v is None or v != v else f"{100 * v:.1f}"


def write_report(report, path):
    L = [f"# XAI report - {report['checkpoint']}", '',
         f"{report['windows']} windows of the '{report['split']}' split, up to {report['per_class']} "
         f"per class (category = rarest arrhythmia with >= {rc.STRAT_MIN_SECONDS} s); per-second "
         f"scores on labelled seconds.", '']
    pr = report.get('probe')
    if pr:
        L += ['## 1. Probes (counterfactual interventions)', '',
              '| intervention | macro F1 | ' + ' | '.join(f"{n} Se" for n in CLASS_NAMES) + ' |',
              '|---|---|' + '---|' * K]
        for name in ('baseline', 'atrial_off', 'rr_mean', 'beat_cond_off', 'lead_uniform',
                     'ssm_identity'):
            if name in pr:
                sc = pr[name]['scores']
                L.append(f"| {name} | {_pct(sc['macro_f1'])} | " +
                         ' | '.join(_pct(sc[n]['se']) for n in CLASS_NAMES) + ' |')
        L += ['', '**Donor swaps** (only one stage input replaced, beats untouched):', '',
              '| windows | n | p(class) before | atrial from donor | R-R from donor |',
              '|---|---|---|---|---|']
        for k, v in pr.get('swaps', {}).items():
            L.append(f"| {k} | {v['n']} | {v['atrial']['p_class_before']:.2f} | "
                     f"{v['atrial']['p_class_after']:.2f} | {v['rr']['p_class_after']:.2f} |")
        L += ['', '**f-wave injection into SINUS windows** (amplitude = fraction of each lead\'s '
              'QRS peak; real f waves are ~3-10 %):', '',
              '| amplitude | p(AFIB) | AFIB seconds | p(AFIB), A branch blind | AFIB seconds, A blind |',
              '|---|---|---|---|---|']
        for d in pr.get('f_inject', []):
            L.append(f"| {100 * d['amplitude_of_qrs']:.0f} % | {d['p_afib']:.3f} | "
                     f"{_pct(d['afib_seconds'])} % | {d.get('p_afib_atrial_off', float('nan')):.3f} | "
                     f"{_pct(d.get('afib_seconds_atrial_off'))} % |")
        L.append('')
    ly = report.get('layers')
    if ly:
        L += ['## 2. Layers', '', '**Lead weights**', '']
        for k in ('v_stem', 'a_stem'):
            v = ly['lead_weights'][k]
            L.append(f"- {k}: entropy {v['entropy']:.2f} (1 = uniform), mean max weight "
                     f"{v['max_weight']:.2f}, mean per lead {[round(a, 2) for a in v['mean_per_lead']]}, "
                     f"agrees with 'channel' {_pct(v['agree_with_channel'])} %")
        nl = ly['lead_weights']['noisy_lead0']
        L.append(f"- lead 1 drowned at 0 dB: V weight {nl['v_weight_lead0_before']:.2f} -> "
                 f"{nl['v_weight_lead0_after']:.2f}, A weight {nl['a_weight_lead0_before']:.2f} -> "
                 f"{nl['a_weight_lead0_after']:.2f}, 'channel' picks lead 1 "
                 f"{_pct(nl['channel_picks_lead0_before'])} -> {_pct(nl['channel_picks_lead0_after'])} %")
        L += ['', '**Per-second F1 by position in the 10 s window** (0 = first second)', '',
              '| position | macro F1 | ' + ' | '.join(CLASS_NAMES) + ' |', '|---|---|' + '---|' * K]
        for r in ly.get('position', []):
            L.append(f"| {r['position']} | {_pct(r['macro_f1'])} | " +
                     ' | '.join(_pct(r[f'F1_{n}']) for n in CLASS_NAMES) + ' |')
        tta = ly.get('tta', {})
        if tta:
            L.append(f"\nTest-time variants: max |delta p| lead swap {tta['swap_max_diff']:.2e}, "
                     f"sign flip {tta['flip_max_diff']:.2e} (a ~0 variant only costs inference).")
        L += ['', '**QRST cancellation (atrial residual, outside the QRS mask)**', '',
              '| class | residual RMS / input | residual peak / input | 4-9 Hz share | 0.5-3 Hz share |',
              '|---|---|---|---|---|']
        for n, v in ly['qrst'].items():
            L.append(f"| {n} | {v['residual_rms_ratio']:.2f} | {v['residual_peak_ratio']:.2f} | "
                     f"{v['share_4_9hz']:.2f} | {v['share_0p5_3hz']:.2f} |")
        L += ['', '**SSM kernels** (90 % of the L1 mass)', '',
              '| layer | kernel s | memory median s | p90 s | >= 1 s | >= 2.5 s | truncated |',
              '|---|---|---|---|---|---|---|']
        for n, v in ly['ssm'].items():
            L.append(f"| {n} | {v['kernel_seconds']:.1f} | {v['memory_median_s']:.2f} | "
                     f"{v['memory_p90_s']:.2f} | {_pct(v['share_memory_ge_1s'])} % | "
                     f"{_pct(v['share_memory_ge_2p5s'])} % | {_pct(v['share_truncated'])} % |")
        L += ['', '**Branch shares** (|gradient x activation| at the concatenation, per class)', '']
        for cname, per in ly['branches'].items():
            parts = [f"{n}: " + ', '.join(f"{k} {100 * s:.0f} %" for k, s in v.items())
                     for n, v in per.items()]
            L.append(f"- **{cname}** - " + ' | '.join(parts))
        dead = sorted(ly['dead'].items(), key=lambda kv: -kv[1]['dead_share'])[:10]
        L += ['', '**Dead ReLU channels** (positive at < 1 % of positions; top 10)', '']
        L += [f"- {n}: {_pct(v['dead_share'])} % of {v['channels']}" for n, v in dead]
        L.append('')
    ex = report.get('experiment')
    if ex:
        L += ['## 3. Inference-time experiments (one factor at a time)', '',
              '| knob | value | macro F1 | ' + ' | '.join(f"F1 {n}" for n in CLASS_NAMES) + ' |',
              '|---|---|---|' + '---|' * K]
        for r in ex:
            L.append(f"| {r['knob']} | {r['value']} | {_pct(r['macro_f1'])} | " +
                     ' | '.join(_pct(r[f'F1_{n}']) for n in CLASS_NAMES) + ' |')
        L.append('')
    exps = report.get('explain')
    if exps:
        L += ['## 4. Explained seconds', '']
        for e in exps:
            L.append(f"- {os.path.basename(e['record'])} s {e['second']}"
                     f"{' [' + e['ec57_error'] + ']' if 'ec57_error' in e else ''}: IG for {e['cls']} p={e['p']:.2f} "
                     f"({', '.join(f'{k} {v:.2f}' for k, v in e['probs_at_second'].items())}) - "
                     f"![]({os.path.basename(e['png'])})")
        L.append('')
    es = report.get('error_sources')
    if es:
        L += ['## 4b. EC57 error seconds: model or post-processing?', '',
              f"From {report.get('ec57_out')}: every false-positive / false-negative second of "
              "mitdb, split by what the raw record-level argmax said at that second.", '',
              '| class | ref s | FP s model | FP s post-proc | FN s model | FN s post-proc | '
              'worst post-proc records (FP s / FN s) |', '|---|---|---|---|---|---|---|']
        for c, v in es.items():
            worst = sorted(v['per_record'].items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:4]
            w = ', '.join(f"{k} ({a}/{b})" for k, (a, b, _, _) in worst if a + b)
            L.append(f"| {c} | {v['ref']} | {v['fp_model']} | {v['fp_pp']} | {v['fn_model']} | "
                     f"{v['fn_pp']} | {w or '-'} |")
        L.append('')
    L += ['## 5. Suggested experiments (ranked)', '']
    for prio, finding, action in report.get('suggestions', []):
        L.append(f"{prio}. **{finding}** -> {action}")
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return path


def run_all(checkpoint, out_dir, parts=('probe', 'layers', 'experiment', 'explain'),
            split='eval', per_class=300, explain_items=(), ec57_out=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    model = load_model(checkpoint)
    names = {l.name for l in model.layers}
    if 'qrst_cancel' not in names:
        raise ValueError("xai needs the dual U-Net (rhythm_dual_*): no 'qrst_cancel' layer")
    report = dict(checkpoint=checkpoint, split=split, per_class=per_class)
    log(f"loading {split} windows ...")
    x, labels, cat = load_windows(split, per_class)
    report['windows'] = int(len(x))
    report['category_counts'] = {n: int((cat == c).sum()) for c, n in enumerate(CLASS_NAMES)}
    log(f"  {len(x)} windows: {report['category_counts']}")
    if 'probe' in parts:
        log("== probe ==")
        report['probe'] = probe(model, x, labels, cat, log=log)
    if 'layers' in parts:
        log("== layers ==")
        report['layers'] = layer_report(model, x, labels, cat, log=log)
    if 'experiment' in parts:
        log("== experiment ==")
        report['experiment'] = experiment(model, x, labels, log=log)
    if 'explain' in parts and explain_items:
        log("== explain ==")
        report['explain'] = []
        for item in explain_items:
            rec, sec, cls = item[:3]
            png = os.path.join(out_dir, f"explain_{os.path.basename(rec)}_{sec}.png")
            e = explain(model, rec, sec, png, cls)
            if len(item) > 3:
                e['ec57_error'] = f"{item[3]} {CLASS_NAMES[cls]} stretch of {item[4]} s"
            report['explain'].append(e)
            log(f"  {os.path.basename(rec)} s{sec}: IG for {e['cls']} p={e['p']:.2f}"
                f"{'  (' + e['ec57_error'] + ')' if 'ec57_error' in e else ''}")
    if ec57_out:
        log("== EC57 error sources ==")
        report['ec57_out'] = ec57_out
        report['error_sources'] = error_sources(ec57_out)
        for c, v in report['error_sources'].items():
            log(f"  {c:5s} ref {v['ref']:5d} s  FP model {v['fp_model']:4d} / post-proc "
                f"{v['fp_pp']:4d}   FN model {v['fn_model']:4d} / post-proc {v['fn_pp']:4d}")
    report['suggestions'] = suggest(report)
    with open(os.path.join(out_dir, 'xai_report.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)
    path = write_report(report, os.path.join(out_dir, 'xai_report.md'))
    log(f"report -> {path}  ({time.time() - t0:.0f} s)")
    return report
