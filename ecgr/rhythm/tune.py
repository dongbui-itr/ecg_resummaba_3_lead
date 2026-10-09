"""Choose the decoding and the beat post-processing on the VALIDATION records only.

The validation records are the eval side of the training PhysioNet databases
(physionet_train.eval_records: ltafdb / nsrdb / incartdb / svdb records training never saw);
the EC57 databases are scored once per model and never tuned on. Everything here reads the
per-record .npz an `ec57 --dbs ltafdb nsrdb incartdb svdb --skip-rhythm-eval` run stored
(probabilities + beats) and scores it with fast approximations of the two WFDB tools:

    rhythm  episode Se / +P by overlap (a predicted episode is a hit when it overlaps a
            reference one) and duration Se / +P on the 1 s grid - epicmp's quantities
    beats   150 ms matching of the hypothesis beats to the reference .atr beats (bxb's window),
            QRS / V / S Se and +P, the first 5 minutes of every record skipped as bxb does

Three grids, in order, each on top of the previous choice:

    1. AF decoding        gap bridging, minimum episode, AF confidence floor, prior alpha -
                          best AF episode + duration Se under a false-AF-episodes-per-hour
                          budget (0.4 / h ~ the mitdb +P target)
    2. beat post-process  the beats.py criteria and options (VT / SVT rate, SVT with N beats,
                          sinus island length, long-invalid length, V / S runs promoted to
                          VT / SVT) - best sum of episode F1 + duration F1 over the ltafdb
                          AFIB / SVT / VT cells, AF budget kept
    3. beat symbols       the S prematurity gate - best S F1 on the beat-labelled records

`tune(ann_root)` returns the chosen decode dict in the shape ec57.run(decode=...) takes, and
writes it next to the annotations as tuned_decode.json.
"""
import itertools
import json
import multiprocessing
import os

import numpy as np
import wfdb

from . import beats as B
from . import config as rc
from .ec57 import load_beats, load_probs, prior_scale
from .labels import decode_track, episodes_from_track

RHYTHM_DBS = ('ltafdb', 'nsrdb', 'incartdb')       # rhythm cells (nsrdb/incartdb: AF FP only)
BEAT_DBS = ('svdb', 'incartdb', 'ltafdb', 'nsrdb')  # beat-labelled .atr
OBJECTIVE_CELLS = tuple((db, c) for db, c in rc.VALIDATION_TARGET_CELLS)
CLASSES = ['AFIB', 'SVT', 'VT']
REF = {'AFIB': ('(AFIB', '(AFL'), 'SVT': ('(SVTA',), 'VT': ('(VT',)}
# MIT-BIH beat symbols -> N / S / V as bxb's AAMI classes read them; F (fusion), Q, paced and
# non-beat symbols are left out of the beat scoring.
BEAT_SYMBOL_CLASS = {'N': 1, 'L': 1, 'R': 1, 'e': 1, 'j': 1,
                     'A': 2, 'a': 2, 'J': 2, 'S': 2, 'V': 3, 'E': 3}
BXB_WINDOW_S = 0.150
BXB_SKIP_S = 300.0

AF_GRID = dict(gap=[0, 3, 5], amin=[3, 5, 10], aprob=[0, 0.6, 0.7, 0.8, 0.9], alpha=[0, 0.5])
PP_GRID = dict(vt_hr=[100, 120], svt_frac=[0.3, 0.5], onset=[1.25, 1.5], sinus=[5, 8],
               long_inv=[10, 20], runs_vt=[False, True], runs_svt=[False, True])
SYMBOL_GRID = [(prem, af_n) for af_n in (True, False) for prem in (0.0, 0.8, 0.85, 0.9)]
# (S prematurity gate, S inside AFIB -> N)

_CACHE = {}
_CTX = {}


# ---------------------------------------------------------------------------
# Records and references
# ---------------------------------------------------------------------------

def validation_records(ann_root, dbs):
    """[(db, src_dir, name)] of the records with a stored .npz under <ann_root>/<db>/."""
    out = []
    for db in dbs:
        d = os.path.join(ann_root, db)
        if db not in rc.PHYSIONET_TRAIN_DBS or not os.path.isdir(d):
            continue
        src = rc.PHYSIONET_TRAIN_DBS[db]['dir']
        out += [(db, src, f[:-4]) for f in sorted(os.listdir(d)) if f.endswith('.npz')]
    return out


def reference_masks(src, name, n_seconds, fs):
    """{class: (n_seconds,) bool} from the record's .atr rhythm marks."""
    a = wfdb.rdann(os.path.join(src, name), 'atr')
    code = np.array(['(N'] * n_seconds, dtype=object)
    marks = [(s, (x or '').split('\x00')[0].strip()) for s, sym, x in
             zip(a.sample, a.symbol, a.aux_note) if sym == '+' and x]
    for (s0, c), (s1, _) in zip(marks, marks[1:] + [(n_seconds * fs, None)]):
        code[int(s0 // fs):int(np.ceil(s1 / fs))] = c
    return {c: np.isin(code, REF[c]) for c in CLASSES}


def reference_beats(src, name, fs):
    """(times in s, class 1/2/3) of the record's reference beats."""
    a = wfdb.rdann(os.path.join(src, name), 'atr')
    keep = [(s / fs, BEAT_SYMBOL_CLASS[y]) for s, y in zip(a.sample, a.symbol)
            if y in BEAT_SYMBOL_CLASS]
    if not keep:
        return np.zeros(0), np.zeros(0, int)
    t, c = zip(*keep)
    return np.asarray(t, np.float64), np.asarray(c, int)


def match_beats(ref_t, ref_c, hyp_t, hyp_c, window=BXB_WINDOW_S, skip=BXB_SKIP_S):
    """bxb-like pairing: each reference beat takes the nearest unpaired hypothesis beat within
    the window. Returns the 6 AAMI counts (QRS, V, S) x (tp Se-side, ref n, tp +P-side,
    hyp n): per class, tp = pairs where both sides carry the class."""
    rm, hm = ref_t >= skip, hyp_t >= skip
    ref_t, ref_c, hyp_t, hyp_c = ref_t[rm], ref_c[rm], hyp_t[hm], hyp_c[hm]
    paired_ref = np.full(len(ref_t), -1)
    j = 0
    used = np.zeros(len(hyp_t), bool)
    for i, t in enumerate(ref_t):
        while j < len(hyp_t) and hyp_t[j] < t - window:
            j += 1
        best, bd = -1, window
        k = j
        while k < len(hyp_t) and hyp_t[k] <= t + window:
            d = abs(hyp_t[k] - t)
            if not used[k] and d <= bd:
                best, bd = k, d
            k += 1
        if best >= 0:
            used[best] = True
            paired_ref[i] = best
    hit = paired_ref >= 0
    out = {}
    out['QRS'] = (int(hit.sum()), len(ref_t), int(hit.sum()), len(hyp_t))
    for name, c in (('V', 3), ('S', 2)):
        both = hit & (ref_c == c) & (hyp_c[np.where(hit, paired_ref, 0)] == c)
        out[name] = (int(both.sum()), int((ref_c == c).sum()), int(both.sum()),
                     int((hyp_c == c).sum()))
    return out


# ---------------------------------------------------------------------------
# Per-record loading (cached per worker process)
# ---------------------------------------------------------------------------

def _load(rec, ann_root):
    key = (rec, ann_root)
    if key not in _CACHE:
        db, src, name = rec
        r, p, fs, sl, hz = load_probs(os.path.join(ann_root, db), name)
        beats = load_beats(os.path.join(ann_root, db), name)
        n = len(r) // hz
        ref = reference_masks(src, name, n, fs)
        rb = reference_beats(src, name, fs)
        _CACHE[key] = (db, r, p, hz, beats, ref, rb)
    return _CACHE[key]


def _track(rec, ann_root, af):
    """The step track for one AF decode point (cached: the PP grid re-uses it)."""
    key = ('track', rec, ann_root, tuple(sorted(af.items())))
    if key not in _CACHE:
        db, r, p, hz, beats, ref, rb = _load(rec, ann_root)
        cls, _ = decode_track(r, p, step_hz=hz, **_af_decode(af))
        _CACHE[key] = cls
    return _CACHE[key]


def _af_decode(af):
    mg = dict(rc.DECODE_MERGE_GAP_SECONDS, AFIB=af['gap'])
    ms = dict(rc.DECODE_MIN_EPISODE_SECONDS, AFIB=af['amin'])
    mp = {'AFIB': af['aprob']} if af['aprob'] else {}
    return dict(merge_gap=mg, min_seconds=ms, min_prob=mp,
                class_scale=prior_scale(af['alpha'], _CTX.get('train_config')))


def _episode_masks(eps, n):
    out = {}
    for c in CLASSES:
        mask = np.zeros(n, bool)
        lst = []
        for e in eps:
            if e['rhythm'] == c:
                lo, hi = int(e['start']), int(np.ceil(e['stop']))
                mask[lo:hi] = True
                lst.append((lo, hi))
        out[c] = (mask, lst)
    return out


def _accumulate(tot, pred_s, db, pred, ref):
    fp_af, nonaf_h = 0, 0.0
    for c in CLASSES:
        mask, lst = pred[c]
        refm = ref[c]
        t = tot[(db, c)]
        edges = np.diff(np.concatenate([[0], refm.astype(np.int8), [0]]))
        for a0, b0 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            t[0] += 1
            t[1] += bool(mask[a0:b0].any())
        t[2] += len(lst)
        t[3] += sum(1 for lo, hi in lst if refm[lo:hi].any())
        t[4] += refm.sum()
        t[5] += (refm & mask).sum()
        pred_s[(db, c)] += mask.sum()
        if c == 'AFIB':
            fp_af += sum(1 for lo, hi in lst if not refm[lo:hi].any())
            nonaf_h += (~refm).sum() / 3600
    return fp_af, nonaf_h


def _rows(tot, pred_s):
    rows = {}
    for key, t in tot.items():
        ese, epp = t[1] / max(t[0], 1), t[3] / max(t[2], 1)
        dse, dpp = t[5] / max(t[4], 1), t[5] / max(pred_s[key], 1)
        f = lambda a, b: 2 * a * b / max(a + b, 1e-9)       # noqa: E731
        rows[key] = dict(E_Se=100 * ese, E_PP=100 * epp, E_F1=100 * f(ese, epp),
                         D_Se=100 * dse, D_PP=100 * dpp, D_F1=100 * f(dse, dpp), n=int(t[0]))
    return rows


def _score_point(args):
    """One grid point over every record: (point, rows, objective, AF FP/h)."""
    af, pp, recs, ann_root = args
    dbs = sorted({r[0] for r in recs})
    tot = {(db, c): np.zeros(6) for db in dbs for c in CLASSES}
    pred_s = {(db, c): 0 for db in dbs for c in CLASSES}
    fp_af, nonaf_h = 0, 0.0
    for rec in recs:
        db, r, p, hz, beats, ref, _rb = _load(rec, ann_root)
        track = _track(rec, ann_root, af)
        if pp is not None and beats is not None:
            crit, opt = pp_point(pp)
            track, _ = B.postprocess(track, beats, hz, crit, opt)
        eps = episodes_from_track(track, r, p, hz)
        f, h = _accumulate(tot, pred_s, db, _episode_masks(eps, len(ref['AFIB'])), ref)
        fp_af += f
        nonaf_h += h
    rows = _rows(tot, pred_s)
    obj = sum(rows[cell]['E_F1'] + rows[cell]['D_F1'] for cell in OBJECTIVE_CELLS
              if cell in rows and rows[cell]['n'])
    return (af, pp), rows, obj, fp_af / max(nonaf_h, 1e-9)


def _score_symbols(args):
    """Beat scoring of one (AF point, PP point, prematurity) over the beat-labelled records."""
    af, pp, (prem, af_n), recs, ann_root = args
    tot = {k: np.zeros(4, int) for k in ('QRS', 'V', 'S')}
    per_db = {}
    for rec in recs:
        db, r, p, hz, beats, ref, (rt, rcl) = _load(rec, ann_root)
        if beats is None or len(rt) == 0:
            continue
        track = _track(rec, ann_root, af)
        crit, opt = pp_point(pp)
        opt = dict(opt, s_prematurity=prem, s_in_afib_to_n=af_n)
        _, sym = B.postprocess(track, beats, hz, crit, opt)
        m = match_beats(rt, rcl, beats['t'], sym)
        for k in tot:
            tot[k] += np.array(m[k])
            per_db.setdefault(db, {k2: np.zeros(4, int) for k2 in tot})[k] += np.array(m[k])
    return (prem, af_n), tot, per_db


def pp_point(pp):
    """A PP grid point -> (criteria, options) for beats.postprocess."""
    c = {k: dict(v) for k, v in rc.BEAT_PP_CRITERIA.items()}
    c['VT']['min_hr'] = pp['vt_hr']
    c['SVT']['min_hr'] = pp['vt_hr']
    c['SVT']['min_frac'] = pp['svt_frac']
    c['SVT']['onset_ratio'] = pp['onset']
    c['SINUS']['duration'] = pp['sinus']
    opt = dict(long_invalid=pp['long_inv'],
               runs_to_rhythm={'VT': bool(pp['runs_vt']), 'SVT': bool(pp['runs_svt'])})
    return c, opt


def _init(train_config):
    _CTX['train_config'] = train_config


def _metrics(t):
    se = 100 * t[0] / max(t[1], 1)
    pp = 100 * t[2] / max(t[3], 1)
    return se, pp, 2 * se * pp / max(se + pp, 1e-9)


def _show_rows(label, rows):
    print(label)
    for (db, c), v in sorted(rows.items()):
        if v['n']:
            print(f"   {db:9s}{c:5s} n={v['n']:4d}  E Se/+P/F1 {v['E_Se']:5.1f}/{v['E_PP']:5.1f}/"
                  f"{v['E_F1']:5.1f}   D Se/+P/F1 {v['D_Se']:5.1f}/{v['D_PP']:5.1f}/{v['D_F1']:5.1f}")


# ---------------------------------------------------------------------------
# The three grids
# ---------------------------------------------------------------------------

def tune(ann_root, budget=0.4, workers=12, train_config=None, af_grid=None, pp_grid=None,
         symbol_grid=None, max_records=None, out_path=None):
    """Run the three grids on the stored validation records under `ann_root`
    (<ec57_out>/_ann). Returns the chosen decode dict for ec57.run(decode=...)."""
    recs = validation_records(ann_root, RHYTHM_DBS)
    missing = [db for db, _ in OBJECTIVE_CELLS if not any(r[0] == db for r in recs)]
    if missing:
        raise FileNotFoundError(f"no stored .npz for {missing} under {ann_root} - the validation "
                                f"inference failed for them (see the 'error on' lines above); "
                                f"the objective cells {list(OBJECTIVE_CELLS)} cannot be scored")
    beat_recs = validation_records(ann_root, BEAT_DBS)
    if max_records:
        recs, beat_recs = recs[:max_records], beat_recs[:max_records]
    if not recs:
        raise FileNotFoundError(f"no validation .npz under {ann_root}/{{{','.join(RHYTHM_DBS)}}}")
    af_grid = af_grid or AF_GRID
    pp_grid = pp_grid or PP_GRID
    symbol_grid = SYMBOL_GRID if symbol_grid is None else symbol_grid
    print(f"===== validation tuning: {len(recs)} rhythm records, {len(beat_recs)} beat records, "
          f"AF budget {budget} false episodes / non-AF hour =====")
    # The workers get train_config through _init; the main process builds the FINAL decode
    # (_af_decode below) and must read the same effective class weights, or the written
    # class_scale is the manifest's (pre-sampler) one - not what the grids scored.
    _init(train_config)
    # spawn, not fork: the caller usually holds a CUDA context (the model it just ran), and
    # a forked child of such a process must not touch it. The workers only need numpy, so
    # they are started without a visible GPU.
    saved = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    pool = multiprocessing.get_context('spawn').Pool(workers, initializer=_init,
                                                     initargs=(train_config,))
    if saved is None:
        del os.environ['CUDA_VISIBLE_DEVICES']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = saved
    try:
        # 1. AF decoding -----------------------------------------------------------------
        af_points = [dict(zip(af_grid, v)) for v in itertools.product(*af_grid.values())]
        res = pool.map(_score_point, [(af, None, recs, ann_root) for af in af_points],
                       chunksize=1)
        print(f"{'gap':>4s}{'min':>5s}{'prob':>6s}{'alpha':>6s}{'E_Se':>7s}{'D_Se':>7s}"
              f"{'FP/h':>7s}{'VT_F1':>7s}{'SVT_F1':>7s}")
        scored = []
        for (af, _), rows, obj, fph in res:
            r = rows[('ltafdb', 'AFIB')]
            vt, svt = rows[('ltafdb', 'VT')], rows[('ltafdb', 'SVT')]
            scored.append((af, r['E_Se'] + r['D_Se'], fph, rows))
            print(f"{af['gap']:4d}{af['amin']:5d}{af['aprob']:6.2f}{af['alpha']:6.2f}"
                  f"{r['E_Se']:7.1f}{r['D_Se']:7.1f}{fph:7.2f}{vt['E_F1']:7.1f}{svt['E_F1']:7.1f}")
        ok = [s for s in scored if s[2] <= budget] or scored
        af_best = max(ok, key=lambda s: s[1])
        print(f"CHOSEN AF decode {af_best[0]}  E_Se+D_Se {af_best[1]:.1f}  FP/h {af_best[2]:.2f}")
        af = af_best[0]

        # 2. beat post-processing ---------------------------------------------------------
        pp_points = [dict(zip(pp_grid, v)) for v in itertools.product(*pp_grid.values())]
        res = pool.map(_score_point, [(af, None, recs, ann_root)] +
                       [(af, pp, recs, ann_root) for pp in pp_points], chunksize=1)
        base = res[0]
        _show_rows(f"BASELINE (step decoding only)  obj {base[2]:.1f}  AF FP/h {base[3]:.2f}",
                   base[1])
        print(f"{'vt_hr':>6s}{'sfrac':>6s}{'onset':>6s}{'sinus':>6s}{'long':>5s}{'rVT':>5s}"
              f"{'rSVT':>5s}{'obj':>8s}{'AF FP/h':>8s}")
        for (_, pp), rows, obj, fph in sorted(res[1:], key=lambda x: -x[2]):
            print(f"{pp['vt_hr']:6d}{pp['svt_frac']:6.2f}{pp['onset']:6.2f}{pp['sinus']:6d}"
                  f"{pp['long_inv']:5d}{int(pp['runs_vt']):5d}{int(pp['runs_svt']):5d}"
                  f"{obj:8.1f}{fph:8.2f}")
        limit = max(budget, base[3] + 0.02)
        ok = [x for x in res[1:] if x[3] <= limit] or res[1:]
        pp_best = max(ok, key=lambda x: x[2])
        if pp_best[2] < base[2]:
            print("no PP point beats the baseline on validation - beat stage OFF")
            pp = None
        else:
            pp = pp_best[0][1]
        _show_rows(f"CHOSEN PP {pp}  obj {pp_best[2]:.1f}  AF FP/h {pp_best[3]:.2f}",
                   pp_best[1])

        # 3. beat symbols ------------------------------------------------------------------
        prem, af_n = 0.0, True
        if pp is not None and beat_recs and symbol_grid:
            res = pool.map(_score_symbols,
                           [(af, pp, s, beat_recs, ann_root) for s in symbol_grid], chunksize=1)
            print(f"{'prem':>6s}{'AF>N':>5s} | {'QRS Se':>7s}{'+P':>7s} | {'V Se':>7s}{'+P':>7s}"
                  f"{'F1':>7s} | {'S Se':>7s}{'+P':>7s}{'F1':>7s}")
            best = None
            for s, tot, per_db in res:
                q, v, sv = _metrics(tot['QRS']), _metrics(tot['V']), _metrics(tot['S'])
                print(f"{s[0]:6.2f}{int(s[1]):5d} | {q[0]:7.2f}{q[1]:7.2f} | {v[0]:7.2f}{v[1]:7.2f}"
                      f"{v[2]:7.2f} | {sv[0]:7.2f}{sv[1]:7.2f}{sv[2]:7.2f}")
                for db, t in sorted(per_db.items()):
                    ds = _metrics(t['S'])
                    print(f"       {db:9s} S Se/+P/F1 {ds[0]:6.2f}/{ds[1]:6.2f}/{ds[2]:6.2f}  "
                          f"(n_ref {t['S'][1]}, n_hyp {t['S'][3]})")
                if best is None or sv[2] > best[1]:
                    best = (s, sv[2])
            prem, af_n = best[0]
            print(f"CHOSEN S prematurity {prem}, S in AFIB -> N {af_n}  S F1 {best[1]:.2f}")
    finally:
        pool.close()
        pool.join()

    decode = dict(_af_decode(af))
    decode['beat_pp'] = pp is not None
    if pp is not None:
        crit, opt = pp_point(pp)
        decode['beat_criteria'] = crit
        decode['beat_options'] = dict(opt, s_prematurity=prem, s_in_afib_to_n=af_n)
    decode['_chosen'] = dict(af=af, pp=pp, s_prematurity=prem, s_in_afib_to_n=af_n)
    path = out_path or os.path.join(ann_root, 'tuned_decode.json')
    with open(path, 'w') as f:
        json.dump(decode, f, indent=2)
    print(f"tuned decode -> {path}")
    return {k: v for k, v in decode.items() if k != '_chosen'}   # the file keeps '_chosen'


def load_decode(path):
    """A tuned_decode.json -> the decode dict ec57.run(decode=...) takes."""
    with open(path) as f:
        d = json.load(f)
    d.pop('_chosen', None)
    return d


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='validation tuning of the rhythm decode + beat '
                                             'post-processing from stored npz')
    ap.add_argument('ann_root', help='<ec57_out>/_ann of a validation run')
    ap.add_argument('--budget', type=float, default=0.4)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--train-config', default=os.environ.get('ECGR_RHYTHM_TRAIN_CONFIG'))
    ap.add_argument('--max-records', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tune(a.ann_root, budget=a.budget, workers=a.workers, train_config=a.train_config,
         max_records=a.max_records, out_path=a.out)
