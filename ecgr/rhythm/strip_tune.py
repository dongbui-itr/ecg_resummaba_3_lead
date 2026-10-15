"""Choose the decoding + post-processing for the reviewed strips on the VALIDATION twin only.

    python3 -m ecgr.rhythm.strip_tune <val npz dir> <out.json> [--workers 24] [--rounds 2]

The stored whole-record probabilities of strip_eval.validation_events (eval split, study-
disjoint from rhythm_eval) are decoded with candidate settings and scored with the Python twin
of the three reference views (strip_eval.score, use_epicmp=False). Coordinate descent: one knob
at a time over its grid, the others held, `rounds` sweeps. Objective: the total shortfall
against every reference cell (sum of max(0, target - ours)), smallest first; ties broken by the
headroom (sum of min(ours - target, 5)). rhythm_eval is never read here.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os

import numpy as np

from . import config as rc
from . import strip_eval as SE

KNOBS = {
    # multiplier on rc.DECODE_CLASS_SCALE (the prior correction) per class
    'scale_AFIB': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.25, 1.5],
    'scale_SVT': [0.6, 0.8, 1.0, 1.25, 1.5],
    'scale_VT': [0.6, 0.8, 1.0, 1.25, 1.5],
    'scale_AVB': [0.3, 0.5, 0.7, 1.0, 1.25],
    'min_AFIB': [2, 3, 5],
    'min_SVT': [0.5, 1, 1.5, 2.5],
    'min_VT': [0.5, 1, 1.5, 2.5],
    'min_AVB': [1, 2, 3, 5],
    'prob_AFIB': [0, 0.5, 0.6, 0.7, 0.8],
    'prob_SVT': [0, 0.5, 0.6, 0.7],
    'prob_VT': [0, 0.5, 0.6, 0.7],
    'prob_AVB': [0, 0.5, 0.6, 0.7, 0.8],
    'beat_pp': [False, 'sinus0', 'sinus3', 'sinus5', True],
    'gap_AFIB': [0, 3, 5],
    'smooth_SINUS': [0.04, 0.5, 1, 2],
    'smooth_AFIB': [0.04, 1, 3, 5],
    'smooth_SVT': [0.04, 0.5, 1, 2],
    'smooth_VT': [0.04, 0.5, 1, 2],
    'smooth_AVB': [0.04, 1, 3, 5],
    'trim_SVT': [None, 0.1, 0.3, 0.6],
    'trim_VT': [None, 0.1, 0.3, 0.6],
    'trim_AVB': [None, 0.3, 0.6, 1.0],
}
START = {'scale_AFIB': 1.0, 'scale_SVT': 1.0, 'scale_VT': 1.0, 'scale_AVB': 1.0,
         'min_AFIB': rc.DECODE_MIN_EPISODE_SECONDS['AFIB'],
         'min_SVT': rc.DECODE_MIN_EPISODE_SECONDS['SVT'],
         'min_VT': rc.DECODE_MIN_EPISODE_SECONDS['VT'],
         'min_AVB': rc.DECODE_MIN_EPISODE_SECONDS['AVB'],
         'prob_AFIB': rc.DECODE_MIN_EPISODE_PROB.get('AFIB', 0), 'prob_SVT': 0, 'prob_VT': 0,
         'prob_AVB': 0, 'beat_pp': rc.DECODE_BEAT_POSTPROCESS,
         'trim_SVT': None, 'trim_VT': None, 'trim_AVB': None,
         'gap_AFIB': rc.DECODE_MERGE_GAP_SECONDS['AFIB'], 'smooth_SINUS': 1, 'smooth_AFIB': 1,
         'smooth_SVT': 1, 'smooth_VT': 1, 'smooth_AVB': 1}     # rc.DECODE_SMOOTH_SECONDS = 1 for all


def to_decode(p):
    """Knob values -> the decode dict ec57.decode_record / strip_eval.tracks take."""
    scale = {n: v * (p.get(f'scale_{n}', 1.0) if n != 'SINUS' else 1.0)
             for n, v in rc.DECODE_CLASS_SCALE.items()}
    mins = dict(rc.DECODE_MIN_EPISODE_SECONDS)
    for n in ('AFIB', 'SVT', 'VT', 'AVB'):
        mins[n] = p[f'min_{n}']
    probs = {n: p[f'prob_{n}'] for n in ('AFIB', 'SVT', 'VT', 'AVB') if p[f'prob_{n}']}
    trim = {n: p[f'trim_{n}'] for n in ('SVT', 'VT', 'AVB') if p[f'trim_{n}'] is not None}
    gaps = dict(rc.DECODE_MERGE_GAP_SECONDS, AFIB=p.get('gap_AFIB', rc.DECODE_MERGE_GAP_SECONDS['AFIB']))
    smooth = {n: p.get(f'smooth_{n}', 1) for n in rc.CLASS_NAMES}
    d = dict(class_scale=scale, min_seconds=mins, min_prob=probs, trim=trim, merge_gap=gaps,
             smooth_seconds=smooth)
    bp = p['beat_pp']
    if isinstance(bp, str):                     # the beat stage with a shorter sinus-island merge
        crit = {k: dict(v) for k, v in rc.BEAT_PP_CRITERIA.items()}
        crit['SINUS']['duration'] = float(bp[len('sinus'):])
        d.update(beat_pp=True, beat_criteria=crit)
    else:
        d['beat_pp'] = bool(bp)
    return d


SAFETY = 1.0     # points above every target the search aims for (headroom for rhythm_eval)


def objective(m, safety=None):
    """(score, shortfall against target + SAFETY [+ Holter penalty], cells met against the real
    targets, cells). With a Holter validation (`m['holter']`), every regression below the base
    model's own bar is added to the shortfall (holter_penalty)."""
    safety = SAFETY if safety is None else safety
    rows = SE.compare(m)
    short = sum(max(0.0, t + safety - o) for _v, _c, _k, o, t, _ok in rows)
    head = sum(min(o - t - safety, 5.0) for _v, _c, _k, o, t, _ok in rows if o >= t + safety)
    if m.get('holter') is not None and _BAR:
        short += HOLTER_WEIGHT * holter_penalty(m['holter'], _BAR)
    return -short + 1e-3 * head, short, sum(r[5] for r in rows), len(rows)


# ---------------------------------------------------------------------------
# Long-Holter validation: the eval side of ltafdb / nsrdb / incartdb / svdb (never EC57). The
# strip-tuned decode must not fall below what the BASE model with the official decode does here.
# ---------------------------------------------------------------------------

HOLTER_DBS = ('ltafdb', 'nsrdb', 'incartdb', 'svdb')
HOLTER_CLASSES = ('AFIB', 'SVT', 'VT')
HOLTER_WEIGHT = 1.0
HOLTER_F1_TOL = 0.5          # F1 points below the bar that cost nothing (noise)
HOLTER_FPH_SCALE = 20.0      # 0.05 false episodes / hour above the bar = 1 point
_HOLTER, _HCACHE, _BAR = [], {}, None


def holter_records(ann_root):
    """[(db, src_dir, name, ann_root)] of the stored validation records under ann_root/<db>."""
    out = []
    for db in HOLTER_DBS:
        d = os.path.join(ann_root, db)
        if os.path.isdir(d):
            src = rc.PHYSIONET_TRAIN_DBS[db]['dir']
            out += [(db, src, f[:-4], ann_root) for f in sorted(os.listdir(d)) if f.endswith('.npz')]
    return out


def _hstored(rec):
    key = (rec[3], rec[0], rec[2])
    if key not in _HCACHE:
        from .ec57 import load_beats, load_probs
        from .tune import reference_masks
        db, src, name, root = rec
        pr = load_probs(os.path.join(root, db), name)
        beats = load_beats(os.path.join(root, db), name)
        ref = reference_masks(src, name, len(pr[0]) // pr[4], pr[2])
        _HCACHE[key] = (pr, beats, ref)
    return _HCACHE[key]


def _episodes(mask):
    e = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def _holter_record(args):
    rec, decode = args
    from .ec57 import decode_record
    (r, p, fs, sl, hz), beats, ref = _hstored(rec)
    episodes, _ = decode_record(r, p, hz, beats, dict(decode or {}))
    n = len(ref['AFIB'])
    out = {}
    for c in HOLTER_CLASSES:
        pm = np.zeros(n, bool)
        for e in episodes:
            if e['rhythm'] == c:
                pm[int(e['start']):int(np.ceil(e['stop']))] = True
        rm = ref[c]
        reps, peps = _episodes(rm), _episodes(pm)
        out[c] = np.array([len(reps), sum(bool(pm[a:b].any()) for a, b in reps), len(peps),
                           sum(bool(rm[a:b].any()) for a, b in peps), rm.sum(), (rm & pm).sum(),
                           pm.sum(), (~rm).sum() / 3600.0], dtype=np.float64)
    return rec[0], out


def holter_score(pool, decode, records=None):
    """{'f1': {'db|class|E or D': F1}, 'fph': {class: false episodes per non-class hour}}."""
    records = _HOLTER if records is None else records
    parts = pool.map(_holter_record, [(rec, decode) for rec in records], chunksize=1)
    tot = {}
    for db, out in parts:
        for c, v in out.items():
            tot[(db, c)] = tot.get((db, c), 0) + v
    f1 = lambda a, b: 2 * a * b / max(a + b, 1e-9)                      # noqa: E731
    res = {'f1': {}, 'fph': {}}
    for (db, c), v in tot.items():
        if v[0] >= 5:                                   # enough reference episodes to score
            ese, epp = v[1] / v[0], v[3] / max(v[2], 1)
            dse, dpp = v[5] / max(v[4], 1), v[5] / max(v[6], 1)
            res['f1'][f'{db}|{c}|E'] = 100 * f1(ese, epp)
            res['f1'][f'{db}|{c}|D'] = 100 * f1(dse, dpp)
    for c in HOLTER_CLASSES:
        fp = sum(v[2] - v[3] for (db, cc), v in tot.items() if cc == c)
        hours = sum(v[7] for (db, cc), v in tot.items() if cc == c)
        res['fph'][c] = fp / max(hours, 1e-9)
    return res


def holter_penalty(h, bar):
    pen = sum(max(0.0, bar['f1'][k] - HOLTER_F1_TOL - h['f1'].get(k, 0.0)) for k in bar['f1'])
    pen += sum(HOLTER_FPH_SCALE * max(0.0, h['fph'][c] - bar['fph'][c] * 1.1 - 0.02)
               for c in bar['fph'])
    return pen


def holter_below(h, bar):
    out = [(k, round(h['f1'].get(k, 0.0), 1), round(v, 1)) for k, v in bar['f1'].items()
           if h['f1'].get(k, 0.0) < v - HOLTER_F1_TOL]
    out += [(f'FP/h {c}', round(h['fph'][c], 3), round(bar['fph'][c], 3)) for c in bar['fph']
            if h['fph'][c] > bar['fph'][c] * 1.1 + 0.02]
    return out


_EVENTS, _ANN, _CACHE = None, None, {}


def _init(events, ann, holter=None, bar=None):
    global _EVENTS, _ANN, _HOLTER, _BAR
    _EVENTS, _ANN = events, ann
    _HOLTER, _BAR = holter or [], bar


def _tracks_chunk(args):
    lo, hi, decode = args
    return [(e['name'], *SE.tracks(e, _ANN, decode, _stored(e)))
            for e in _EVENTS[lo:hi] if _stored(e) is not None]


def _stored(e):
    if e['name'] not in _CACHE:
        from .ec57 import load_beats, load_probs
        pr = load_probs(_ANN, e['name'])
        _CACHE[e['name']] = None if pr is None else (pr, load_beats(_ANN, e['name']))
    return _CACHE[e['name']]


def evaluate(pool, events, decode, chunks):
    parts = pool.map(_tracks_chunk, [(lo, hi, decode) for lo, hi in chunks])
    by = {e['name']: e for e in events}
    tr = [(by[n], ref, pred, mean) for part in parts for n, ref, pred, mean in part]
    m = SE.score(tr, use_epicmp=False)
    if _HOLTER:
        m['holter'] = holter_score(pool, decode)
    return m


def tune(events, ann, workers=24, rounds=2, knobs=None, start=None, random_steps=300,
         holter=None, bar=None, log=print):
    knobs = knobs or KNOBS
    p = dict(start or START)
    n = len(events)
    step = max(1, n // (workers * 4))
    chunks = [(i, min(n, i + step)) for i in range(0, n, step)]
    ctx = mp.get_context('fork')
    history = []
    global _HOLTER, _BAR
    _HOLTER, _BAR = holter or [], bar        # the parent needs them for objective()
    with ctx.Pool(workers, initializer=_init, initargs=(events, ann, holter, bar)) as pool:
        m = evaluate(pool, events, to_decode(p), chunks)
        best = objective(m)
        log(f"start: shortfall {best[1]:.1f}, cells met {best[2]}/{best[3]}"
            + (f", Holter below bar {holter_below(m['holter'], bar)}" if m.get('holter') else ''))
        history.append(dict(params=dict(p), shortfall=best[1], met=best[2]))
        for r in range(rounds):
            changed = False
            for k, values in knobs.items():
                for v in values:
                    if v == p[k]:
                        continue
                    q = dict(p, **{k: v})
                    mq = evaluate(pool, events, to_decode(q), chunks)
                    o = objective(mq)
                    if o[0] > best[0] + 1e-9:
                        p, best, m, changed = q, o, mq, True
                        log(f"  round {r + 1}: {k} = {v} -> shortfall {o[1]:.2f}, met {o[2]}/{o[3]}")
                        history.append(dict(params=dict(p), shortfall=o[1], met=o[2]))
            if not changed:
                break
        # random perturbations of 1-3 knobs around the best point
        import random
        rng = random.Random(0)
        for it in range(random_steps):
            q = dict(p)
            for k in rng.sample(list(knobs), rng.randint(1, 3)):
                q[k] = rng.choice(knobs[k])
            if q == p:
                continue
            mq = evaluate(pool, events, to_decode(q), chunks)
            o = objective(mq)
            if o[0] > best[0] + 1e-9:
                diff = {k: q[k] for k in q if q[k] != p[k]}
                p, best, m = q, o, mq
                log(f"  random {it}: {diff} -> shortfall {o[1]:.2f}, met {o[2]}/{o[3]}")
                history.append(dict(params=dict(p), shortfall=o[1], met=o[2]))
    return p, m, history


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('ann')
    ap.add_argument('out')
    ap.add_argument('--events', default=None)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--start', default=None, help='a previous output json: start from its params')
    ap.add_argument('--random', type=int, default=300, help='random perturbation steps after the sweeps')
    ap.add_argument('--holter-ann', default=None, help='<ec57 out>/_ann of a long-Holter validation run')
    ap.add_argument('--holter-bar', default=None, help='bar json (make-bar) the decode must not fall below')
    a = ap.parse_args(argv)
    events = json.load(open(a.events))
    holter = holter_records(a.holter_ann) if a.holter_ann else None
    bar = json.load(open(a.holter_bar)) if a.holter_bar else None
    start = None
    if a.start:
        start = dict(START, **json.load(open(a.start))['params'])
    p, m, hist = tune(events, a.ann, a.workers, a.rounds, start=start, random_steps=a.random,
                      holter=holter, bar=bar)
    rows = SE.compare(m)
    out = dict(params=p, decode=to_decode(p), validation=dict(
        met=sum(r[5] for r in rows), cells=len(rows), shortfall_safety=objective(m)[1],
        shortfall=objective(m, 0.0)[1],
        below=[(v, c, k, round(o, 1), t) for v, c, k, o, t, ok in rows if not ok]),
        holter=m.get('holter'), holter_bar=bar,
        holter_below=holter_below(m['holter'], bar) if (m.get('holter') and bar) else None,
        history=hist)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    print(SE.format_report(m, f'VALIDATION (strip_val), tuned decode - {a.ann}'))
    if m.get('holter') and bar:
        print('Holter validation below the base bar:', holter_below(m['holter'], bar) or 'none')
    print(f"-> {a.out}")


def make_bar(ann_root, out, decode=None, workers=16):
    """The base model's Holter numbers with `decode` (default the official one) -> bar json."""
    recs = holter_records(ann_root)
    global _HOLTER
    _HOLTER = recs
    with mp.get_context('fork').Pool(workers, initializer=_init, initargs=([], None, recs, None)) as pool:
        h = holter_score(pool, decode or {}, recs)
    json.dump(h, open(out, 'w'), indent=1, default=float)
    print(json.dumps(h, indent=1, default=float))
    return h


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'make-bar':
        make_bar(sys.argv[2], sys.argv[3], workers=int(sys.argv[4]) if len(sys.argv) > 4 else 16)
    else:
        main()
