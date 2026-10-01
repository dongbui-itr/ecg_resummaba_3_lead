"""Score a rhythm checkpoint - by default on the rhythm test set, which training never saw.

Three views of the same predictions:

  * per second - the confusion over the six classes on labelled, clean seconds, with Se, +P
    and F1 per class. What the model is trained to do.
  * per strip  - does a 10 s window that holds an AF / SVT / VT / AVB2 / AVB3 episode get at
    least one second of that class (raw argmax, NOISE-gated), and does a called class have
    one in the reference? What a report reviewer sees. No DECODE_* post-processing here: that
    runs once over a WHOLE record (ec57.py, predict.py), never on a 10 s window.
  * per window - the 'lead' output: NOISE / CH1 / CH2 / CH3 confusion, accuracy, and how well
    NOISE (no readable lead) is recognised.

and at several noise levels: the stored windows as they are, then the same windows with noise
on every strip at a fixed SNR (config.CLEAN_SNR_DB decides which seconds of which lead count
as readable, and so which lead is the target). The noise is seeded, so two checkpoints are scored on identical corrupted strips.
"""
import json
import os

import numpy as np
import tensorflow as tf

from ..training.train import setup_gpus
from . import config as rc
from . import pipeline
from .model import output_names, rhythm_steps
from .objectives import per_class_table
from .train import format_confusion

DEFAULT_SNRS = (None, 18.0, 12.0, 6.0, 0.0, -6.0)


def predict_arrays(model, dataset):
    """({'rhythm', 'lead'} targets, {'rhythm', 'lead'} predictions) over a dataset."""
    predict = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    ys, ps = {}, {}
    for x, y in dataset:
        p = predict(x)
        for k in y:
            ys.setdefault(k, []).append(y[k].numpy())
            ps.setdefault(k, []).append(p[k].numpy())
    return ({k: np.concatenate(v) for k, v in ys.items()},
            {k: np.concatenate(v) for k, v in ps.items()})


def summarize(y_true, y_pred):
    K = rc.NUM_CLASSES
    rt, rp = y_true['rhythm'], y_pred['rhythm']

    # per second, labelled clean seconds (weight 1)
    sel = rt[..., K] > 0.5
    truth, pred = np.argmax(rt[..., :K], -1), np.argmax(rp, -1)
    cm = np.bincount(truth[sel] * K + pred[sel], minlength=K * K).reshape(K, K)
    rows = per_class_table(cm, rc.CLASS_NAMES)
    # a class that is present but never predicted scores 0, as in objectives.RhythmF1
    present = [0.0 if np.isnan(r['f1']) else r['f1'] for r in rows if r['support'] > 0]

    if 'noise' in y_true:
        lead, window_noise = _noise_summary(y_true['noise'], y_pred['noise'])
    else:
        lead, window_noise = _lead_summary(y_true['lead'], y_pred['lead'])

    # per strip: one reference/predicted set of arrhythmias per window
    strip = {n: dict(tp=0, fp=0, fn=0) for n in rc.CLASS_NAMES[1:]}
    for yt, yp, pn in zip(rt, rp, window_noise):
        ok = yt[:, :K].sum(-1) > 0.5
        ref = {rc.CLASS_NAMES[c] for c in np.argmax(yt[ok, :K], -1)} - {'SINUS'}
        # Raw per-step calls only: the DECODE_* post-processing (smoothing, bridging, minimum
        # durations) is a whole-record step (ec57.py / predict.py) and has no business on an
        # isolated 10 s window, where a 7 s AFIB minimum would erase most true episodes.
        hyp = set() if pn > rc.DECODE_NOISE_THRESHOLD else \
            {rc.CLASS_NAMES[c] for c in np.argmax(yp, -1)} - {'SINUS'}
        for n in strip:
            if n in ref and n in hyp:
                strip[n]['tp'] += 1
            elif n in hyp:
                strip[n]['fp'] += 1
            elif n in ref:
                strip[n]['fn'] += 1
    for n, v in strip.items():
        v['se'] = v['tp'] / (v['tp'] + v['fn']) if v['tp'] + v['fn'] else float('nan')
        v['ppv'] = v['tp'] / (v['tp'] + v['fp']) if v['tp'] + v['fp'] else float('nan')
    return dict(confusion=cm.tolist(), per_class=rows,
                macro_f1=float(np.mean(present)) if present else float('nan'),
                lead=lead, strip=strip, windows=int(len(rt)))


def _noise_summary(nt, np_):
    """'noise' output: CLEAN / NOISE confusion over the 2 s segments. A window counts as
    unreadable for the strip view when its mean p(NOISE) is above the decode threshold."""
    t, p = np.argmax(nt, -1).ravel(), np.argmax(np_, -1).ravel()
    cm = np.bincount(t * 2 + p, minlength=4).reshape(2, 2)
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    out = dict(kind='noise', confusion=cm.tolist(), segments=int(len(t)),
               accuracy=float(np.mean(t == p)), noise_segments=int((t == 1).sum()),
               noise_se=tp / (tp + fn) if tp + fn else float('nan'),
               noise_ppv=tp / (tp + fp) if tp + fp else float('nan'))
    return out, np_[..., 1].mean(-1)


def _lead_summary(lt, lp):
    # per window, the lead output
    L = rc.NUM_LEAD_CLASSES
    lead_t, lead_p = np.argmax(lt, -1), np.argmax(lp, -1)
    lead_cm = np.bincount(lead_t * L + lead_p, minlength=L * L).reshape(L, L)
    noisy_t, noisy_p = lead_t == rc.LEAD_NOISE, lead_p == rc.LEAD_NOISE
    tp, fp = int(np.sum(noisy_t & noisy_p)), int(np.sum(~noisy_t & noisy_p))
    fn = int(np.sum(noisy_t & ~noisy_p))
    both_readable = ~noisy_t & ~noisy_p
    lead = dict(confusion=lead_cm.tolist(), windows=int(len(lead_t)),
                accuracy=float(np.mean(lead_t == lead_p)),
                noise_windows=int(noisy_t.sum()),
                noise_se=tp / (tp + fn) if tp + fn else float('nan'),
                noise_ppv=tp / (tp + fp) if tp + fp else float('nan'),
                # of the windows both call readable, how often the SAME lead is picked
                lead_agreement=float(np.mean(lead_t[both_readable] == lead_p[both_readable]))
                if both_readable.any() else float('nan'))
    return lead, lp[:, rc.LEAD_NOISE]


def format_summary(s, title):
    lines = [format_confusion(np.asarray(s['confusion']), title),
             f"macro F1 (per second, clean labelled seconds): {100 * s['macro_f1']:.2f}", '',
             'per strip (10 s window, raw argmax - no post-processing; a NOISE window finds '
             'nothing):',
             f"{'class':<8s}{'TP':>8s}{'FN':>8s}{'FP':>8s}{'Se':>9s}{'+P':>9s}"]
    for n, v in s['strip'].items():
        lines.append(f"{n:<8s}{v['tp']:>8d}{v['fn']:>8d}{v['fp']:>8d}"
                     f"{100 * v['se']:>9.2f}{100 * v['ppv']:>9.2f}")
    z = s['lead']
    if z.get('kind') == 'noise':
        lines += ['', format_confusion(np.asarray(z['confusion']), 'noise output (per 2 s)',
                                       names=rc.NOISE_CLASSES, unit='segments'),
                  f"noise accuracy {100 * z['accuracy']:.2f} | NOISE: {z['noise_segments']:,} "
                  f"of {z['segments']:,} segments, Se {100 * z['noise_se']:.2f} +P "
                  f"{100 * z['noise_ppv']:.2f}"]
        return '\n'.join(lines)
    lines += ['', format_confusion(np.asarray(z['confusion']), 'lead output (per window)',
                                   names=rc.LEAD_CLASSES, unit='windows'),
              f"lead accuracy {100 * z['accuracy']:.2f} | NOISE: {z['noise_windows']:,} of "
              f"{z['windows']:,} windows, Se {100 * z['noise_se']:.2f} +P "
              f"{100 * z['noise_ppv']:.2f} | same lead when both readable "
              f"{100 * z['lead_agreement']:.2f}"]
    return '\n'.join(lines)


def evaluate(checkpoint, split='test', snrs=DEFAULT_SNRS, batch_size=None, max_windows=None,
             out_dir=None):
    setup_gpus()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    segs, labs, _ = pipeline.load_arrays(split, max_windows=max_windows,
                                         label_steps=rhythm_steps(model))
    outputs = output_names(model)
    print(f"{split}: {len(labs):,} windows, checkpoint {checkpoint}")
    out_dir = out_dir or os.path.join(rc.REPORT_DIR, model.name)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(checkpoint))[0]

    results, texts = {}, []
    for snr in snrs:
        if snr is None:
            ds = pipeline.make_dataset(segs, labs, batch_size or rc.BATCH_SIZE, mode='clean',
                                       outputs=outputs)
            tag = 'as recorded'
        else:
            ds = pipeline.make_dataset(segs, labs, batch_size or rc.BATCH_SIZE, mode='noisy',
                                       augment_kw=dict(noise_prob=1.0, permute_prob=0.0,
                                                       snr_range=(snr, snr), wreck_prob=0.0),
                                       outputs=outputs)
            tag = f"noise at {snr:g} dB SNR"
        y_true, y_pred = predict_arrays(model, ds)
        s = summarize(y_true, y_pred)
        results['clean' if snr is None else f"snr_{snr:g}"] = s
        text = format_summary(s, f"{split.upper()} - {tag} - {stem}")
        print('\n' + text)
        texts.append(text)

    with open(os.path.join(out_dir, f"{stem}_{split}_eval.txt"), 'w') as f:
        f.write(f"Checkpoint: {checkpoint}\n\n" + '\n\n'.join(texts) + '\n')
    with open(os.path.join(out_dir, f"{stem}_{split}_eval.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nreport -> {out_dir}/{stem}_{split}_eval.txt")
    return results
