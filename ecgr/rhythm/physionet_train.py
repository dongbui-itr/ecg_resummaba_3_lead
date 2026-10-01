"""Long PhysioNet recordings outside EC57 as rhythm training sources (config.PHYSIONET_TRAIN_*).

ltafdb (84 x 24 h, AF with rhythm marks), nsrdb (18 x 24 h, sinus), svdb (78 x 30 min, SVT
runs in the beat labels), incartdb (75 x 30 min, 12-lead, BBB and PVCs). None of them is an
EC57 database; the EC57 ones and rhythm_eval are refused by name.

A record becomes many events, each ONE 10 s window: `chunk` is the part of the file to read
(the window plus FILTER_MARGIN seconds either side, so the band-pass sees context and the
window itself is free of edge effects), `strip` the window inside the chunk, `spans` the
arrhythmia episodes and `known` the stretches whose unmarked seconds are SINUS - both relative
to the chunk start, at the record's own rate (build._scale_event resamples them). build.py
then treats the event like a portal strip.

Which windows: every 10 s grid window of the record, plus one window centred on each VT / SVT
episode (they are seconds long and would straddle grid windows), is given the first category of
config.PHYSIONET_TRAIN_CAPS it fits, and each category keeps at most its cap per record, chosen
by a hash so a rebuild keeps the same windows.
"""
import hashlib
import os
import re
from collections import Counter, defaultdict

import numpy as np
import wfdb

from . import config as rc
from .labels import beat_runs, class_index, merge_intervals

FILTER_MARGIN_SECONDS = 5
# `types` of an event are portal event types (build/inventory look them up in
# config.EVENT_TYPE_TO_CLASS); the window category itself travels as ev['category'].
CATEGORY_TYPE = {'VT': 'VT', 'SVT': 'SVT', 'AF_edge': 'AFIB', 'AF_ectopy': 'AFIB', 'AF': 'AFIB',
                 'sinus_pac': 'SINUS', 'sinus_tachy': 'SINUS', 'sinus_brady': 'SINUS',
                 'sinus_ectopy': 'SINUS', 'sinus': 'SINUS'}
BEAT_SYMBOLS = set('NLRBAaJSVEFejn/fQ')
VENTRICULAR = set('VE')
SUPRAVENTRICULAR = set('SAaJ')


def _hash01(text):
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16) / float(16 ** 12)


def _rng(text):
    return np.random.default_rng(int(hashlib.sha1(text.encode()).hexdigest()[:12], 16))


def records(db_dir):
    return sorted(f[:-4] for f in os.listdir(db_dir) if f.endswith('.hea'))


def patient_key(db, name, header):
    """Split unit: the patient where the header names one (incartdb), else the record."""
    for c in header.comments:
        m = re.match(r'\s*patient\s+(\d+)', c)
        if m:
            return f"{db}/patient{m.group(1)}"
    return f"{db}/{name}"


def read_annotations(path, n):
    """(rhythm intervals [(code, a, b)], beat samples, beat symbols) from <path>.atr."""
    ann = wfdb.rdann(path, 'atr')
    marks = [(int(s), (x or '').split('\x00')[0].strip())
             for s, sym, x in zip(ann.sample, ann.symbol, ann.aux_note) if sym == '+' and x]
    intervals = [(c, a, b) for (a, c), (b, _) in zip(marks, marks[1:] + [(n, None)]) if b > a]
    beat = np.array([s in BEAT_SYMBOLS for s in ann.symbol], dtype=bool)
    return intervals, np.asarray(ann.sample)[beat].astype(np.int64), \
        np.asarray(ann.symbol)[beat]


def label_record(kind, intervals, samples, symbols, n, fs):
    """(spans [(class, a, b)], known [(a, b)], ignore [(a, b)]) of a whole record, samples.

    kind 'rhythm': the rhythm marks decide; 'sinus': sinus throughout plus beat runs (and the
    few rhythm marks there are); 'runs': beat runs only, nothing else known."""
    spans, known, ignore = [], [], []
    if kind in ('rhythm', 'sinus'):
        if kind == 'sinus':
            known = [(0, n)]
        for code, a, b in intervals:
            if code in rc.PHYSIONET_TRAIN_CODE_TO_CLASS:
                spans.append((class_index(rc.PHYSIONET_TRAIN_CODE_TO_CLASS[code]), a, b))
            elif code in rc.PHYSIONET_TRAIN_SINUS_CODES:
                if kind == 'rhythm':
                    known.append((a, b))
            else:
                ignore.append((a, b))
    if kind in ('sinus', 'runs'):
        # runs are only derived where no rhythm mark already says what the stretch is
        marked = merge_intervals([(a, b) for _, a, b in spans] + ignore)
        for c, a, b in beat_runs(samples, symbols, fs=fs):
            if not any(a < mb and b > ma for ma, mb in marked):
                spans.append((c, a, b))
    known = [(a, b) for a, b in merge_intervals(known)]
    if ignore:                                      # carve the ignored codes out of `known`
        known = _subtract(known, merge_intervals(ignore))
    return sorted(spans, key=lambda s: s[1]), known, merge_intervals(ignore)


def _subtract(intervals, holes):
    out = []
    for a, b in intervals:
        cur = a
        for ha, hb in holes:
            if hb <= cur or ha >= b:
                continue
            if ha > cur:
                out.append((cur, ha))
            cur = max(cur, hb)
        if cur < b:
            out.append((cur, b))
    return out


def _overlap(intervals, a, b):
    return sum(max(0, min(b, y) - max(a, x)) for x, y in intervals)


def window_category(a, b, spans, known, ignore, samples, symbols, fs):
    """The first config.PHYSIONET_TRAIN_CAPS category window [a, b) fits, or None (skip)."""
    w = b - a
    if _overlap(ignore, a, b) > 0.2 * w:
        return None
    inside = [(c, x, y) for c, x, y in spans if x < b and y > a]
    classes = {c for c, _, _ in inside}
    # a run of a few beats is kept whatever else the window holds (svdb labels nothing else)
    if class_index('VT') in classes:
        return 'VT'
    if class_index('SVT') in classes:
        return 'SVT'
    labelled = _overlap([(x, y) for _, x, y in inside], a, b) + _overlap(known, a, b)
    if labelled < 0.5 * w:
        return None
    lo, hi = np.searchsorted(samples, [a, b])
    sym = symbols[lo:hi]
    n_v = int(np.isin(sym, list(VENTRICULAR)).sum())
    n_s = int(np.isin(sym, list(SUPRAVENTRICULAR)).sum())
    rate = 60.0 * (hi - lo) / (w / fs)
    af = class_index('AFIB')
    if af in classes:
        af_cover = _overlap([(x, y) for c, x, y in inside if c == af], a, b)
        if af_cover < 0.9 * w:
            return 'AF_edge'
        return 'AF_ectopy' if n_v >= 1 else 'AF'
    if classes - {rc.SINUS}:
        return None                                  # AV block: none in these databases
    if n_s >= 3:
        return 'sinus_pac'                          # frequent PACs: the 232 look-alike of AF
    if rate > 100:
        return 'sinus_tachy'
    if rate < 55:
        return 'sinus_brady'
    if n_v + n_s >= 2:
        return 'sinus_ectopy'
    return 'sinus'


def candidate_starts(n, fs, spans):
    """Grid windows every 10 s, plus one centred on each VT / SVT episode (1 s aligned)."""
    w = rc.SEGMENT_SECONDS * fs
    margin = FILTER_MARGIN_SECONDS * fs
    lo, hi = margin, n - w - margin
    if hi <= lo:
        return []
    starts = set(range(lo, hi, w))
    runs = {class_index('VT'), class_index('SVT')}
    for c, a, b in spans:
        if c in runs:
            s = (a + b) // 2 - w // 2
            starts.add(int(np.clip((s // fs) * fs, lo, hi - 1)))
    return sorted(starts)


def _relative(intervals, c0, c1):
    return [(max(a, c0) - c0, min(b, c1) - c0) for a, b in intervals if a < c1 and b > c0]


def record_events(db, name, cfg):
    """The chosen events of one record, and a Counter of their categories."""
    path = os.path.join(cfg['dir'], name)
    header = wfdb.rdheader(path)
    fs, n = int(round(header.fs)), header.sig_len
    intervals, samples, symbols = read_annotations(path, n)
    spans, known, ignore = label_record(cfg['labels'], intervals, samples, symbols, n, fs)

    w = rc.SEGMENT_SECONDS * fs
    margin = FILTER_MARGIN_SECONDS * fs
    by_cat = defaultdict(list)
    for s in candidate_starts(n, fs, spans):
        cat = window_category(s, s + w, spans, known, ignore, samples, symbols, fs)
        if cat:
            by_cat[cat].append(s)

    key = patient_key(db, name, header)
    events, counts = [], Counter()
    lead_names = list(header.sig_name)
    for cat, starts in by_cat.items():
        cap = rc.PHYSIONET_TRAIN_CAPS[cat]
        if len(starts) > cap:
            starts = sorted(_rng(f"{db}/{name}/{cat}").choice(starts, cap, replace=False))
        for s in starts:
            c0, c1 = int(s - margin), int(s + w + margin)
            ev_spans = [(c, max(a, c0) - c0, min(b, c1) - c0) for c, a, b in spans
                        if a < c1 and b > c0]
            ev = dict(source=db, study_id=None, patient=key, event_id=f"{db}_{name}_{s}",
                      types=[CATEGORY_TYPE[cat]], spans=ev_spans, strip=(margin, margin + w),
                      known=_relative(known, c0, c1), whole=False, needs_runs=False,
                      record=path, chunk=(c0, c1), category=cat)
            if cfg.get('twelve_lead'):
                from .ptbxl import LEAD_NAMES, lead_subsets
                leads = lead_subsets(int(_hash01(f"{db}/{name}/{s}") * 1e12), n=1)[0]
                order = [lead_names.index(LEAD_NAMES[i]) if LEAD_NAMES[i] in lead_names else i
                         for i in leads]
                ev.update(leads=order, lead_names=[lead_names[i] for i in order])
            events.append(ev)
            counts[cat] += 1
    return events, counts, key


def collect(dbs=None, eval_fraction=None):
    """({'train': [events], 'eval': [events]}, report) over config.PHYSIONET_TRAIN_DBS."""
    dbs = dbs or rc.PHYSIONET_TRAIN_DBS
    frac = rc.PHYSIONET_TRAIN_EVAL_FRACTION if eval_fraction is None else eval_fraction
    bad = set(dbs) & set(rc.EC57_DATABASES)
    if bad or any(os.path.basename(os.path.normpath(c['dir'])) in rc.EC57_DATABASES
                  for c in dbs.values()):
        raise ValueError(f"EC57 databases are never a training source: {sorted(bad)}")
    out, report = {'train': [], 'eval': []}, {}
    patients = {}
    for db, cfg in dbs.items():
        if not os.path.isdir(cfg['dir']):
            print(f"{db:10s}: not found at {cfg['dir']} - skipped")
            continue
        per = {'train': Counter(), 'eval': Counter()}
        n_rec = {'train': set(), 'eval': set()}
        for name in records(cfg['dir']):
            events, counts, key = record_events(db, name, cfg)
            split = 'eval' if _hash01(key) < frac else 'train'
            sid = patients.setdefault(key, str(rc.PHYSIONET_TRAIN_STUDY_OFFSET + len(patients)))
            for ev in events:
                ev['study_id'] = sid
            out[split] += events
            per[split].update(counts)
            n_rec[split].add(name)
        report[db] = {s: dict(records=len(n_rec[s]), windows=sum(per[s].values()),
                              categories=dict(per[s])) for s in per}
        for s in ('train', 'eval'):
            print(f"{db:10s}: {s:5s} {len(n_rec[s]):3d} records -> "
                  f"{sum(per[s].values()):>6,} windows {dict(per[s])}")
    return out, report


def eval_records(dbs=None, eval_fraction=None):
    """{db: [record names]} of the eval side - the held-out long records the decoding can be
    tuned on without touching EC57."""
    dbs = dbs or rc.PHYSIONET_TRAIN_DBS
    frac = rc.PHYSIONET_TRAIN_EVAL_FRACTION if eval_fraction is None else eval_fraction
    out = {}
    for db, cfg in dbs.items():
        if not os.path.isdir(cfg['dir']):
            continue
        names = []
        for name in records(cfg['dir']):
            key = patient_key(db, name, wfdb.rdheader(os.path.join(cfg['dir'], name)))
            if _hash01(key) < frac:
                names.append(name)
        out[db] = names
    return out
