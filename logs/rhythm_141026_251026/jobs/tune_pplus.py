"""tune_pplus.py <key> [--sweeps N] [--random N] [--workers N]

P+-oriented decode tuning on VALIDATION ONLY (never EC57 / dataset-eval v4):
  * Holter validation = eval side of ltafdb / nsrdb / incartdb / svdb, stored npz under
    summary_111026/ec57/holterval_<key>/_ann (whole-record probabilities + picked beats)
  * strip_val2 = 4,719 portal events (eval split), npz under strip_eval/val_npz/<key>

Targets (the user's EC57 goals mapped onto validation proxies) and GUARDS (every other metric
may not drop below the official decode's value on the same validation data, within a tolerance):

  target                           proxy on validation
  VEB +P mitdb > 95                V +P, all beat-labelled Holter records (bxb-like, 150 ms)
  SVEB +P mitdb > 72               S +P pooled; S +P svdb (S-rich)
  SVEB +P escdb > 40               S +P ltafdb + incartdb pooled
  AFIB mitdb D+P / E Se / E+P > 90 ltafdb AFIB D +P, E Se, E +P
  SVT, VT mitdb all > 70           ltafdb SVT / VT E Se, E +P, D Se, D +P
  AVB mitdb E +P > 60              false AVB episodes per hour on the Holter records (no AVB
                                   reference there -> every AVB episode is false): <= half of
                                   the official decode's rate
  guards                           QRS / V / S Se, AFIB D Se, AF-SVT-VT false episodes per hour,
                                   every strip_val2 cell (E Se, E +P, D Se, D +P of AFIB / SVT /
                                   VT / AVB - the rhythm_eval twin)

Search = coordinate descent over KNOBS from the official decode, then random 2-knob moves.
Writes <out>/tuned_decode.json (ec57.run(decode=...) shape), <out>/trace.jsonl, <out>/report.md.
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

REPO = '/media/Project/ECG/Source_Dong/project_ai/ecg_resummaba_3_lead'
sys.path.insert(0, REPO)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from ecgr.rhythm import config as rc, strip_eval as SE, tune as T, beats as B   # noqa: E402
from ecgr.rhythm.ec57 import load_beats, load_probs                             # noqa: E402
from ecgr.rhythm.labels import decode_track, episodes_from_track               # noqa: E402

W = '/media/Project/ECG/Model_Dong/ecgr_rhythm/summary_111026'
HOLTER_DBS = ('ltafdb', 'nsrdb', 'incartdb', 'svdb')
RCLS = ['AFIB', 'SVT', 'VT', 'AVB']
REF = dict(T.REF, AVB=())          # no AVB reference on these databases

# ---------------------------------------------------------------- knobs
KNOBS = {
    'scale_AFIB': [0.7, 0.85, 1.0, 1.25],
    'scale_SVT': [0.6, 0.8, 1.0, 1.25],
    'scale_VT': [0.6, 0.8, 1.0, 1.25],
    'scale_AVB': [0.3, 0.5, 0.7, 1.0],
    'min_AFIB': [3, 5, 8, 12],
    'min_SVT': [1, 1.5, 2, 3],
    'min_VT': [1, 1.5, 2, 3],
    'min_AVB': [2, 3, 5, 8],
    'prob_AFIB': [0.6, 0.7, 0.8, 0.85],
    'prob_SVT': [0, 0.5, 0.6, 0.7],
    'prob_VT': [0, 0.5, 0.6, 0.7],
    'prob_AVB': [0, 0.5, 0.6, 0.7, 0.8],
    'gap_AFIB': [3, 5, 10, 20],
    'gap_SVT': [0, 1, 2],
    'gap_VT': [0, 1, 2],
    'smooth_AFIB': [1, 3, 5],
    'smooth_SVT': [0.5, 1, 2],
    'smooth_VT': [0.5, 1, 2],
    'smooth_AVB': [1, 3, 5],
    'ctx_AFIB': [None, (10, 10, 0.3), (15, 15, 0.3), (10, 10, 0.5)],
    'ctx_AVB': [None, (10, 10, 0.3), (15, 15, 0.4)],
    'trim_SVT': [None, 0.1, 0.3],
    'trim_VT': [None, 0.1, 0.3],
    'beat_pp': [True, False],
    'vt_hr': [100, 110, 120],
    'svt_hr': [100, 110, 120],
    'svt_frac': [0.3, 0.5],
    'onset': [1.25, 1.5],
    'sinus': [5, 8, 12],
    'long_inv': [10, 20],
    'runs_vt': [False, True],
    'runs_svt': [False, True],
    'avb_dur': [2.5, 4, 6],
    'avb_maxhr': [55, 60, 65],
    's_prem': [0.0, 0.8, 0.85, 0.9],
    's_afib_n': [True, False],
    'v_margin': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    's_margin': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
}
START = {'scale_AFIB': 1.0, 'scale_SVT': 1.0, 'scale_VT': 1.0, 'scale_AVB': 1.0,
         'min_AFIB': rc.DECODE_MIN_EPISODE_SECONDS['AFIB'], 'min_SVT': rc.DECODE_MIN_EPISODE_SECONDS['SVT'],
         'min_VT': rc.DECODE_MIN_EPISODE_SECONDS['VT'], 'min_AVB': rc.DECODE_MIN_EPISODE_SECONDS['AVB'],
         'prob_AFIB': rc.DECODE_MIN_EPISODE_PROB.get('AFIB', 0), 'prob_SVT': 0, 'prob_VT': 0, 'prob_AVB': 0,
         'gap_AFIB': rc.DECODE_MERGE_GAP_SECONDS['AFIB'], 'gap_SVT': rc.DECODE_MERGE_GAP_SECONDS['SVT'],
         'gap_VT': rc.DECODE_MERGE_GAP_SECONDS['VT'],
         'smooth_AFIB': 1, 'smooth_SVT': 1, 'smooth_VT': 1, 'smooth_AVB': 1,
         'ctx_AFIB': None, 'ctx_AVB': None, 'trim_SVT': None, 'trim_VT': None,
         'beat_pp': rc.DECODE_BEAT_POSTPROCESS,
         'vt_hr': rc.BEAT_PP_CRITERIA['VT']['min_hr'], 'svt_hr': rc.BEAT_PP_CRITERIA['SVT']['min_hr'],
         'svt_frac': rc.BEAT_PP_CRITERIA['SVT']['min_frac'], 'onset': rc.BEAT_PP_CRITERIA['SVT']['onset_ratio'],
         'sinus': rc.BEAT_PP_CRITERIA['SINUS']['duration'], 'long_inv': rc.BEAT_PP_LONG_INVALID_SECONDS,
         'runs_vt': rc.BEAT_PP_RUNS_TO_RHYTHM['VT'], 'runs_svt': rc.BEAT_PP_RUNS_TO_RHYTHM['SVT'],
         'avb_dur': rc.BEAT_PP_CRITERIA['AVB']['duration'], 'avb_maxhr': rc.BEAT_PP_CRITERIA['AVB']['max_hr'],
         's_prem': rc.BEAT_PP_S_PREMATURITY, 's_afib_n': rc.BEAT_PP_S_IN_AFIB_TO_N,
         'v_margin': 0, 's_margin': 0}
RHYTHM_KEYS = [k for k in KNOBS if k.split('_')[0] in ('scale', 'min', 'prob', 'gap', 'smooth', 'ctx')]


def to_decode(p):
    """Knob values -> the decode dict ec57.decode_record takes (= what EC57 will run)."""
    scale = {n: v * (p.get(f'scale_{n}', 1.0) if n != 'SINUS' else 1.0) for n, v in rc.DECODE_CLASS_SCALE.items()}
    mins = dict(rc.DECODE_MIN_EPISODE_SECONDS, **{n: p[f'min_{n}'] for n in RCLS})
    probs = {n: p[f'prob_{n}'] for n in RCLS if p[f'prob_{n}']}
    gaps = dict(rc.DECODE_MERGE_GAP_SECONDS, AFIB=p['gap_AFIB'], SVT=p['gap_SVT'], VT=p['gap_VT'])
    smooth = {'SINUS': 1, 'AFIB': p['smooth_AFIB'], 'SVT': p['smooth_SVT'], 'VT': p['smooth_VT'], 'AVB': p['smooth_AVB']}
    ctx = {n: list(p[f'ctx_{n}']) for n in ('AFIB', 'AVB') if p[f'ctx_{n}']}
    trim = {n: p[f'trim_{n}'] for n in ('SVT', 'VT') if p[f'trim_{n}'] is not None}
    crit = {k: dict(v) for k, v in rc.BEAT_PP_CRITERIA.items()}
    crit['VT']['min_hr'] = p['vt_hr']
    crit['SVT'].update(min_hr=p['svt_hr'], min_frac=p['svt_frac'], onset_ratio=p['onset'])
    crit['SINUS']['duration'] = p['sinus']
    crit['AVB'].update(duration=p['avb_dur'], max_hr=p['avb_maxhr'])
    opt = dict(long_invalid=p['long_inv'], runs_to_rhythm={'VT': bool(p['runs_vt']), 'SVT': bool(p['runs_svt'])},
               s_prematurity=p['s_prem'], s_in_afib_to_n=bool(p['s_afib_n']))
    return dict(class_scale=scale, min_seconds=mins, min_prob=probs, merge_gap=gaps, smooth_seconds=smooth,
                context_rule=ctx, trim=trim, beat_pp=bool(p['beat_pp']), beat_criteria=crit, beat_options=opt,
                v_margin=p['v_margin'], s_margin=p['s_margin'])


# ---------------------------------------------------------------- data (loaded in the parent, forked)
HREC = []      # [(db, r, p, hz, beats, ref masks {cls: bool}, (ref beat t, cls))]
SREC = []      # [(event, stored)]
_TRACK = {}    # worker cache: (rec index, rhythm key) -> int8 track


def load_holter(ann_root):
    for db, src, name in T.validation_records(ann_root, HOLTER_DBS):
        r, p, fs, sl, hz = load_probs(os.path.join(ann_root, db), name)
        beats = load_beats(os.path.join(ann_root, db), name)
        n = len(r) // hz
        a = __import__('wfdb').rdann(os.path.join(src, name), 'atr')
        code = np.array(['(N'] * n, dtype=object)
        marks = [(s, (x or '').split('\x00')[0].strip()) for s, sym, x in zip(a.sample, a.symbol, a.aux_note) if sym == '+' and x]
        for (s0, c), (s1, _) in zip(marks, marks[1:] + [(n * fs, None)]):
            code[int(s0 // fs):int(np.ceil(s1 / fs))] = c
        ref = {c: np.isin(code, REF[c]) if REF[c] else np.zeros(n, bool) for c in RCLS}
        HREC.append((db, name, r, p, hz, beats, ref, T.reference_beats(src, name, fs)))
    print(f'holter: {len(HREC)} records', {db: sum(1 for h in HREC if h[0] == db) for db in HOLTER_DBS}, flush=True)


def load_strip(events_json, npz_dir):
    ev = json.load(open(events_json))
    for e in ev:
        st = load_probs(npz_dir, e['name'])
        if st is None:
            continue
        SREC.append((e, (st, load_beats(npz_dir, e['name']))))
    print(f'strip_val2: {len(SREC)} of {len(ev)} events have npz', flush=True)


# ---------------------------------------------------------------- scoring (workers)
def _rhythm_part(d):
    return {k: d[k] for k in ('class_scale', 'min_seconds', 'min_prob', 'merge_gap', 'smooth_seconds', 'context_rule')}


def _decode_cached(i, d):
    """ec57.decode_record, with the step track cached per rhythm-decode point (the beat knobs
    re-use it). Returns (episodes, beat symbols)."""
    db, name, r, p, hz, beats, ref, rb = HREC[i]
    rk = (i, json.dumps(_rhythm_part(d), sort_keys=True))
    if rk not in _TRACK:
        cls, _ = decode_track(r, p, step_hz=hz, **_rhythm_part(d))
        if len(_TRACK) >= 6:
            _TRACK.pop(next(iter(_TRACK)))
        _TRACK[rk] = cls.astype(np.int8)
    cls = _TRACK[rk].astype(int)
    bt = B.relabel_s(B.relabel_v(beats, d['v_margin']), d['s_margin'])
    sym = None
    if bt is not None and d['beat_pp']:
        cls, sym = B.postprocess(cls, bt, hz, d['beat_criteria'], d['beat_options'])
    elif bt is not None:
        sym = np.asarray(bt['cls'], int)
    if bt is not None and d['trim']:
        cls = B.trim_to_beats(cls, bt, hz, d['trim'])
    return episodes_from_track(cls, r, p, hz), sym, bt


def score_holter(args):
    i, d = args
    db, name, r, p, hz, beats, ref, (rt, rcl) = HREC[i]
    eps, sym, bt = _decode_cached(i, d)
    n = len(ref['AFIB'])
    out = {}
    for c in RCLS:
        pm = np.zeros(n, bool)
        for e in eps:
            if e['rhythm'] == c:
                pm[int(e['start']):int(np.ceil(e['stop']))] = True
        rm = ref[c]
        reps, peps = SE._episodes(rm), SE._episodes(pm)
        out[c] = np.array([len(reps), sum(bool(pm[a:b].any()) for a, b in reps), len(peps),
                           sum(bool(rm[a:b].any()) for a, b in peps), rm.sum(), (rm & pm).sum(), pm.sum(),
                           (~rm).sum() / 3600.0], np.float64)
    bm = None
    if bt is not None and sym is not None and len(rt):
        bm = T.match_beats(rt, rcl, np.asarray(bt['t']), np.asarray(sym))
    return db, out, bm


def score_strip(args):
    lo, hi, d = args
    tot = {c: np.zeros(7) for c in SE.EVENT_CLASSES}
    for e, st in SREC[lo:hi]:
        ref, pred, _m = SE.tracks(e, None, dict(d), st)
        for c in SE.EVENT_CLASSES:
            ci = SE.PRED_NAMES.index(c)
            rset = [SE.REF_TYPES.index(t) for t in (('AVB2', 'AVB3') if c == 'AVB' else (c,))]
            rm, pm = np.isin(ref, rset), pred == ci
            t = tot[c]
            for a, b in SE._episodes(rm):
                t[0] += 1; t[1] += bool(pm[a:b].any())
            for a, b in SE._episodes(pm):
                t[2] += 1; t[3] += bool(rm[a:b].any())
            t[4] += rm.sum(); t[5] += pm.sum(); t[6] += (rm & pm).sum()
    return tot


def evaluate(pool, d, nchunk=16):
    tasks = [(i, d) for i in range(len(HREC))]
    step = max(1, len(SREC) // nchunk + 1)
    schunks = [(lo, min(lo + step, len(SREC)), d) for lo in range(0, len(SREC), step)]
    hres = pool.map_async(score_holter, tasks, chunksize=1)
    sres = pool.map(score_strip, schunks, chunksize=1)
    hres = hres.get()
    m = {}
    f1 = lambda a, b: 2 * a * b / max(a + b, 1e-9)    # noqa: E731
    # holter rhythm per db / pooled
    tot = {}
    for db, out, _ in hres:
        for c, v in out.items():
            tot[(db, c)] = tot.get((db, c), 0) + v
    for (db, c), v in tot.items():
        if v[0] >= 5:
            m[f'H|{db}|{c}|E_Se'] = 100 * v[1] / v[0]; m[f'H|{db}|{c}|E_PP'] = 100 * v[3] / max(v[2], 1)
            m[f'H|{db}|{c}|D_Se'] = 100 * v[5] / max(v[4], 1); m[f'H|{db}|{c}|D_PP'] = 100 * v[5] / max(v[6], 1)
    for c in RCLS:
        fp = sum(v[2] - v[3] for (db, cc), v in tot.items() if cc == c)
        hours = sum(v[7] for (db, cc), v in tot.items() if cc == c)
        m[f'FPH|{c}'] = fp / max(hours, 1e-9)
    # beats per db / pooled
    bt = {}
    for db, _, bm in hres:
        if bm is None:
            continue
        for k, v in bm.items():
            bt[(db, k)] = bt.get((db, k), 0) + np.array(v)
            bt[('all', k)] = bt.get(('all', k), 0) + np.array(v)
    for (db, k), v in bt.items():
        if v[1] >= 50:
            m[f'B|{db}|{k}|Se'] = 100 * v[0] / max(v[1], 1); m[f'B|{db}|{k}|PP'] = 100 * v[2] / max(v[3], 1)
    v = bt.get(('ltafdb', 'S'), 0) + bt.get(('incartdb', 'S'), 0)
    if np.ndim(v):
        m['B|ltaf+inc|S|Se'] = 100 * v[0] / max(v[1], 1); m['B|ltaf+inc|S|PP'] = 100 * v[2] / max(v[3], 1)
    # strip_val2
    st = {c: sum(t[c] for t in sres) for c in SE.EVENT_CLASSES}
    for c, t in st.items():
        m[f'S|{c}|E_Se'] = 100 * t[1] / max(t[0], 1); m[f'S|{c}|E_PP'] = 100 * t[3] / max(t[2], 1)
        m[f'S|{c}|D_Se'] = 100 * t[6] / max(t[4], 1); m[f'S|{c}|D_PP'] = 100 * t[6] / max(t[5], 1)
    return m


# ---------------------------------------------------------------- objective
# 5.4.1 EC57 table (image of 2026-10-16) mapped onto validation proxies, + the user's floors
TARGETS = [('B|all|QRS|PP', 99.8, 1.0),                                  # nstdb QRS +P 99.77
           ('B|all|V|PP', 95, 1.0), ('B|all|V|Se', 94, 0.2),               # VEB mitdb +P 94.7 / Se 93.9
           ('B|all|S|PP', 72, 1.0), ('B|svdb|S|PP', 72, 0.5), ('B|ltaf+inc|S|PP', 40, 0.5),
           ('B|all|S|Se', 60, 0.5),                                         # SVEB mitdb Se 58.5
           ('H|ltafdb|AFIB|D_Se', 98, 0.5), ('H|ltafdb|AFIB|D_PP', 90, 1.0),
           ('H|ltafdb|AFIB|E_Se', 90, 1.0), ('H|ltafdb|AFIB|E_PP', 92, 1.0),
           ('S|AFIB|E_PP', 92, 0.5), ('S|AFIB|D_PP', 90, 0.5),
           ('S|VT|E_Se', 82, 0.5), ('S|VT|D_PP', 70, 0.5), ('S|SVT|E_Se', 70, 0.5), ('S|SVT|D_PP', 70, 0.5)] + \
          [(f'H|ltafdb|{c}|{k}', 70, 0.5 if c == 'VT' else 0.2) for c in ('SVT', 'VT') for k in ('E_Se', 'E_PP', 'D_Se', 'D_PP')]
AVB_FPH_FRACTION = 0.2        # AVB E +P 100 on mitdb -> (almost) no false AVB episodes on validation
TOL = {'B': 0.3, 'H': 1.0, 'S': 1.0}
TIGHT = {'S|AVB|E_Se': 0.2, 'S|AVB|D_Se': 0.3, 'S|AFIB|D_Se': 0.3, 'S|AFIB|E_Se': 0.3, 'B|all|S|Se': 0.2, 'B|all|V|Se': 0.2}
GUARD_W = 10.0
# SE_TOL=<points>: the user's 2026-10-19 rule - the beat V / S sensitivities may drop by up to this much
# (they leave TARGETS and become guarded keys with this tolerance); everything else keeps its tolerance.
SE_TOL = float(os.environ['SE_TOL']) if os.environ.get('SE_TOL') else None
if SE_TOL is not None:
    TARGETS = [t for t in TARGETS if t[0] not in ('B|all|V|Se', 'B|all|S|Se')]
FPH_W = {'AFIB': 25.0, 'SVT': 10.0, 'VT': 10.0}     # false episodes / h -> points


def objective(m, base):
    short, guard, head = 0.0, 0.0, 0.0
    tk = {t[0] for t in TARGETS}
    for k, tgt, w in TARGETS:
        v = m.get(k)
        if v is None:
            continue
        short += w * max(0.0, tgt - v)
        head += w * min(max(0.0, v - tgt), 5.0)
    # AVB false episodes: <= half the official rate
    avb_t = AVB_FPH_FRACTION * base['FPH|AVB']
    short += 60.0 * max(0.0, m['FPH|AVB'] - avb_t)
    head += min(5.0, 60.0 * max(0.0, avb_t - m['FPH|AVB']))
    for k, bv in base.items():
        if k in tk or k not in m:
            continue
        if k.startswith('FPH|'):
            c = k.split('|')[1]
            if c in FPH_W:
                guard += FPH_W[c] * max(0.0, m[k] - bv * 1.1 - 0.02)
            continue
        tol = TIGHT.get(k, TOL[k[0]])
        if SE_TOL is not None and k.startswith('B|') and (k.endswith('|V|Se') or k.endswith('|S|Se')):
            tol = SE_TOL
        guard += max(0.0, bv - tol - m[k])
    score = -(short + GUARD_W * guard) + 1e-3 * head
    return score, short, guard


# ---------------------------------------------------------------- search
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--sweeps', type=int, default=2); ap.add_argument('--random', type=int, default=40)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--out', default=None)
    ap.add_argument('--start', default=None, help='json of knob values to start from')
    ap.add_argument('--ref', default=None, help='guard against tune_pplus_<ref>/base_metrics.json instead of this checkpoint\'s own official decode')
    a = ap.parse_args()
    out = a.out or f'{W}/strip_eval/tune_pplus_{a.key}'
    os.makedirs(out, exist_ok=True)
    load_holter(f'{W}/ec57/holterval_{a.key}/_ann')
    load_strip(f'{W}/strip_eval/strip_val2_events.json', f'{W}/strip_eval/val_npz/{a.key}')
    pool = get_context('fork').Pool(a.workers)
    trace = open(f'{out}/trace.jsonl', 'a')
    cache = {}

    def ev(p):
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in cache:
            t0 = time.time()
            cache[key] = evaluate(pool, to_decode(p))
            trace.write(json.dumps(dict(knobs=p, metrics=cache[key], sec=round(time.time() - t0, 1)), default=str) + '\n'); trace.flush()
        return cache[key]

    cur = dict(START)
    if a.start:
        cur.update(json.load(open(a.start)))
    base = ev(dict(START))
    json.dump(base, open(f'{out}/base_metrics.json', 'w'), indent=1)
    if a.ref:
        base = json.load(open(f'{W}/strip_eval/tune_pplus_{a.ref}/base_metrics.json'))
        print(f'guards / AVB floor relative to reference {a.ref}', flush=True)
    cur_m = ev(cur)
    cur_s, cur_short, cur_guard = objective(cur_m, base)
    print(f'START score {cur_s:.2f} shortfall {cur_short:.1f} guard {cur_guard:.1f}', flush=True)
    rng = random.Random(0)
    n_eval = 0
    for sweep in range(a.sweeps):
        keys = list(KNOBS); rng.shuffle(keys)
        improved = False
        for k in keys:
            best = None
            for v in KNOBS[k]:
                if v == cur[k]:
                    continue
                cand = dict(cur, **{k: v}); m = ev(cand); s = objective(m, base); n_eval += 1
                if best is None or s[0] > best[0][0]:
                    best = (s, cand, m)
            if best and best[0][0] > cur_s + 1e-6:
                cur, cur_m, (cur_s, cur_short, cur_guard) = best[1], best[2], best[0]
                improved = True
                print(f'sweep {sweep} {k}={cur[k]!r:>14} -> score {cur_s:.2f} short {cur_short:.1f} guard {cur_guard:.1f}', flush=True)
        json.dump(dict(knobs=cur, score=cur_s, short=cur_short, guard=cur_guard, metrics=cur_m), open(f'{out}/best.json', 'w'), indent=1, default=str)
        if not improved:
            break
    for it in range(a.random):
        cand = dict(cur)
        for k in rng.sample(list(KNOBS), 2):
            cand[k] = rng.choice(KNOBS[k])
        m = ev(cand); s = objective(m, base)
        if s[0] > cur_s + 1e-6:
            cur, cur_m, (cur_s, cur_short, cur_guard) = cand, m, s
            print(f'random {it} -> score {cur_s:.2f} short {cur_short:.1f} guard {cur_guard:.1f}', flush=True)
    pool.close(); pool.join()
    dec = to_decode(cur)
    json.dump(dict(decode=dec, knobs=cur, score=cur_s, shortfall=cur_short, guard=cur_guard), open(f'{out}/tuned_decode.json', 'w'), indent=1, default=str)
    # report
    lines = [f'# tune_pplus {a.key}: validation only (Holter eval side + strip_val2), overlap as stored', '',
             f'score {cur_s:.2f}  target shortfall {cur_short:.1f}  guard violation {cur_guard:.1f}', '', '## knobs changed from official', '']
    lines += [f'- {k}: {START[k]!r} -> {cur[k]!r}' for k in KNOBS if cur[k] != START[k]]
    lines += ['', '## validation metrics (official -> tuned)', '', '| metric | official | tuned | target |', '|---|---|---|---|']
    tg = {t[0]: t[1] for t in TARGETS}
    for k in sorted(base):
        lines.append(f'| {k} | {base[k]:.2f} | {cur_m.get(k, float("nan")):.2f} | {tg.get(k, "")} |')
    open(f'{out}/report.md', 'w').write('\n'.join(lines))
    print('\n'.join(lines)); print('TUNE_PPLUS_DONE', flush=True)


if __name__ == '__main__':
    main()
