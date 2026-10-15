"""Strip-level scoring of the reviewed rhythm strips (rhythm_eval) and of its validation twin.

rhythm_eval (rc.EVAL_DIR, the test split) holds 4,909 portal events: one 30-60 s record each,
a reviewer mark (the 10-50 s strip that was looked at) and one labelled span inside it. Only the
mark is known - outside it nobody vouched for anything. The reference product's rhythm_eval
tables ("Version 5.4.1": epicmp segment to segment + a sample-to-sample confusion; an event
confusion with Se / PPV / F1 per class) are therefore reproduced INSIDE THE MARK:

  sample  every 1/GRID_HZ s of the mark: reference = the span's type (AVB2 / AVB3 kept apart),
          SINUS elsewhere in the mark; prediction = the decoded class (SINUS / AFIB / SVT / VT /
          AVB / NOISE). Row-normalised confusion; the diagonal is the per-class recall (AVB2 and
          AVB3 rows are correct when the prediction is AVB, the model's merged class).
  event   one label per event: the arrhythmia (AFIB / SVT / VT / AVB) with the most predicted
          time inside the mark; when the mark holds none, the arrhythmia with the highest mean
          (prior-corrected) probability there - the reference matrix has no 'none' column. The
          'none' count is reported next to it. Se / PPV / F1 per class.
  epicmp  every event's mark becomes its own scoring record (header only, no signal), with one
          reference / hypothesis pair per class; `epicmp -A` per class, Gross episode and
          duration Se / +P - the segment-to-segment table.

The validation twin (`validation_events`) is built from the EVAL split of the five training
sources rhythm_eval was exported from (afib-2, dataset-5, rhythm-2, dataset-4, rhythm-3) -
study-disjoint from both train and rhythm_eval (build.audit_written / split_verification.json)
- with rhythm_eval's class composition and a 10 s mark around the labelled span where the export
has no reviewed strip. Every tuning choice is made on it; rhythm_eval is scored, never tuned on.
"""
import csv
import glob
import json
import os
import random
import shutil

import numpy as np
import wfdb

from . import config as rc

REF_TYPES = ['SINUS', 'AFIB', 'SVT', 'VT', 'AVB2', 'AVB3']
PRED_NAMES = list(rc.CLASS_NAMES) + ['NOISE']            # SINUS AFIB SVT VT AVB NOISE
EVENT_CLASSES = ['AFIB', 'SVT', 'VT', 'AVB']
TYPE_MAP = {'AFIB': 'AFIB', 'SVT': 'SVT', 'SVE_RUN': 'SVT', 'VT': 'VT', 'VE_RUN': 'VT',
            'AVB2': 'AVB2', 'AVB3': 'AVB3'}
# validation: only the event types rhythm_eval itself carries - the portal's SVE_RUN / VE_RUN
# exports are 3-5 beat runs (median span 1-1.6 s against rhythm_eval's 2.4-4.7 s for SVT / VT)
VAL_TYPES = ('AFIB', 'SVT', 'VT', 'AVB2', 'AVB3')
MAX_MARK_SECONDS = 50.0
VAL_SOURCES = ['afib-2', 'dataset-5', 'rhythm-2', 'dataset-4', 'rhythm-3']
TEST_COUNTS = {'AFIB': 1673, 'SVT': 1803, 'VT': 439, 'AVB2': 686, 'AVB3': 308}
GRID_HZ = 25
MARK_SECONDS = 10.0

# The reference product's rhythm_eval numbers (the images of 2026-10-12). AVB is the model's
# merged class: it must reach the better of the AVB2 / AVB3 rows on every metric.
TARGET_EPICMP = {'AFIB': (90, 94, 88, 96), 'SVT': (89, 88, 60, 78), 'VT': (81, 90, 62, 86),
                 'AVB': (25, 77, 31, 65)}                   # E Se, E +P, D Se, D +P ("5.4.1")
TARGET_SAMPLE_RECALL = {'SINUS': 92.89, 'AFIB': 88.07, 'SVT': 72.60, 'VT': 69.37,
                        'AVB2': 26.58, 'AVB3': 30.69}      # sample-to-sample diagonal ("5.4.1")
# event confusion (second image): Se / PPV / F1; AVB = AVB2 + AVB3 merged from its matrix
TARGET_EVENT = {'AFIB': (86.6, 80.3, 83.3), 'SVT': (86.6, 81.1, 83.8), 'VT': (70.1, 53.3, 60.6),
                'AVB': (40.2, 62.5, 48.9)}
TARGET_EVENT_AVB_ROWS = {'AVB2': 22.6, 'AVB3': 19.9}       # per-row Se


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _event(name, record, mark, spans, etype):
    h = wfdb.rdheader(record)
    a, b = max(0, int(mark[0])), min(int(h.sig_len), int(mark[1]))
    return dict(name=name, record=record, fs=float(h.fs), n=int(h.sig_len), mark=(a, b),
                spans=[(t, int(x), int(y)) for t, x, y in spans], type=etype)


def test_events(eval_dir=None):
    """rhythm_eval as strip events (name = <study>_<event>, as ec57.eval_record_name)."""
    out = []
    for p in sorted(glob.glob(os.path.join(eval_dir or rc.EVAL_DIR, '*', '*', '*.json'))):
        info = json.load(open(p))
        spans = [(TYPE_MAP.get(str(r['type']).upper()), r['start'], r['stop'])
                 for r in (info.get('ref') or {}).get('rhythms') or []]
        spans = [s for s in spans if s[0]]
        etype = TYPE_MAP.get(str(info['type']).upper())
        if not etype:
            continue
        out.append(_event(f"{info['studyId']}_{info['eventId']}", p[:-5],
                          (info['startMarkSample'], info['stopMarkSample']), spans, etype))
    return out


def validation_events(seed=0, scale=1.0, cache=None, fill_with_runs=False):
    """The eval-split twin of rhythm_eval (see the module docstring). Cached as JSON."""
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    from . import inventory
    from .build import header_span, record_path
    by_split, _report = inventory.collect(VAL_SOURCES)
    pool = {t: [] for t in TEST_COUNTS}
    runs_pool = {t: [] for t in TEST_COUNTS}
    names = {i: n for i, n in enumerate(rc.CLASS_NAMES)}
    for ev in by_split['eval']:
        raw = str(ev['types'][0]).upper() if ev.get('types') else ''
        etype = raw if raw in VAL_TYPES else None
        run = False
        if etype is None and fill_with_runs and raw in ('SVE_RUN', 'VE_RUN'):
            etype, run = TYPE_MAP[raw], True
        if not etype:
            continue
        path = record_path(ev)
        if not path:
            continue
        cls = 'AVB' if etype in ('AVB2', 'AVB3') else etype
        spans = [(c, a, b) for c, a, b in ev['spans'] if names.get(c) == cls]
        if not spans and ev.get('header_span'):
            hs = header_span(path)
            spans = [(None, *hs)] if hs else []
        if not spans:
            continue                                    # no span to score against
        spans = [(etype, a, b) for _, a, b in spans]
        (runs_pool if run else pool)[etype].append((ev, path, spans))
    rng = random.Random(seed)
    events = []
    for etype, items in pool.items():
        rng.shuffle(items)
        want = int(round(TEST_COUNTS[etype] * scale))
        items = items[:want]
        if fill_with_runs and len(items) < want:
            # the portal SVE_RUN / VE_RUN exports fill a class up to its count: short runs are
            # part of SVT / VT on the reviewed strips too (validation v1 had none and its decode
            # dropped them on rhythm_eval, 2026-10-12)
            extra = runs_pool[etype][:]
            rng.shuffle(extra)
            items = items + extra[:want - len(items)]
        for ev, path, spans in items:
            h = wfdb.rdheader(path)
            if ev.get('strip'):
                mark = ev['strip']
            elif spans[0][2] - spans[0][1] > MARK_SECONDS * h.fs:
                # a long span: rhythm_eval's mark covers it (20-30 s SVT strips), up to 50 s
                a = spans[0][1]
                mark = (a, min(spans[0][2], a + int(MAX_MARK_SECONDS * h.fs)))
            else:
                a, b = spans[0][1], spans[0][2]
                half = MARK_SECONDS * h.fs / 2
                c = (a + b) / 2
                mark = (int(c - half), int(c + half))
                if mark[0] < 0:
                    mark = (0, int(2 * half))
                if mark[1] > h.sig_len:
                    mark = (int(h.sig_len - 2 * half), int(h.sig_len))
            e = _event(f"{ev['study_id']}_{ev['event_id']}", path, mark, spans, etype)
            if any(min(y, e['mark'][1]) > max(x, e['mark'][0]) for _, x, y in e['spans']):
                events.append(e)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(events, open(cache, 'w'))
    return events


# ---------------------------------------------------------------------------
# Inference and decoding
# ---------------------------------------------------------------------------

def predict(model_path, events, ann_dir, log=print):
    """Whole-record probabilities of every event's record -> <ann_dir>/<name>.npz (skipped when
    present). Same inference as ec57 (predict.predict_signal: hop / TTA from rc)."""
    import tensorflow as tf
    from .ec57 import predict_record_probs, probs_path, save_probs
    todo = [e for e in events if not os.path.exists(probs_path(ann_dir, e['name']))]
    if not todo:
        return
    model = tf.keras.models.load_model(model_path, compile=False)
    for i, e in enumerate(todo, 1):
        *probs, fs, sig_len, step_hz, beats = predict_record_probs(model, e['record'])
        save_probs(ann_dir, e['name'], *probs, fs, sig_len, step_hz, beats)
        if i % 500 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} records predicted")


def tracks(e, ann_dir, decode=None, stored=None):
    """(reference (m,), prediction (m,), mean prob-corrected probabilities (K,)) on the
    GRID_HZ grid of the event's mark. Reference indexes REF_TYPES, prediction PRED_NAMES."""
    from .ec57 import decode_record, load_beats, load_probs
    r, p, fs, sig_len, hz = stored[0] if stored else load_probs(ann_dir, e['name'])
    beats = stored[1] if stored else load_beats(ann_dir, e['name'])
    episodes, _ = decode_record(r, p, hz, beats, dict(decode or {}))
    a, b = e['mark']
    g0, g1 = int(np.floor(a / e['fs'] * GRID_HZ)), int(np.ceil(b / e['fs'] * GRID_HZ))
    m = max(1, g1 - g0)
    ref = np.zeros(m, dtype=int)
    for t, x, y in e['spans']:
        x0 = int(np.floor(x / e['fs'] * GRID_HZ)) - g0
        y0 = int(np.ceil(y / e['fs'] * GRID_HZ)) - g0
        ref[max(0, x0):max(0, min(m, y0))] = REF_TYPES.index(t)
    pred = np.zeros(m, dtype=int)
    for ep in episodes:
        x0 = int(round(ep['start'] * GRID_HZ)) - g0
        y0 = int(round(ep['stop'] * GRID_HZ)) - g0
        if y0 > 0 and x0 < m:
            pred[max(0, x0):min(m, y0)] = PRED_NAMES.index(ep['rhythm'])
    # mean probability over the mark (rows at hz, the classes' prior correction applied)
    scale = (decode or {}).get('class_scale', rc.DECODE_CLASS_SCALE) or {}
    sv = np.array([scale.get(n, 1.0) for n in rc.CLASS_NAMES], np.float32)
    rr = r[int(a / fs * hz):max(int(a / fs * hz) + 1, int(b / fs * hz))] * sv
    mean = (rr / np.maximum(rr.sum(1, keepdims=True), 1e-9)).mean(0)
    return ref, pred, mean


def all_tracks(events, ann_dir, decode=None, cache=None):
    out = []
    for e in events:
        st = cache.get(e['name']) if cache is not None else None
        if cache is not None and st is None:
            from .ec57 import load_beats, load_probs
            st = (load_probs(ann_dir, e['name']), load_beats(ann_dir, e['name']))
            if st[0] is None:
                continue
            cache[e['name']] = st
        out.append((e, *tracks(e, ann_dir, decode, st)))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sample_confusion(tr):
    cm = np.zeros((len(REF_TYPES), len(PRED_NAMES)), dtype=np.int64)
    for _e, ref, pred, _m in tr:
        np.add.at(cm, (ref, pred), 1)
    rows = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    recall = {}
    for i, t in enumerate(REF_TYPES):
        j = PRED_NAMES.index('AVB' if t in ('AVB2', 'AVB3') else t)
        recall[t] = float(rows[i, j])
    return cm, rows, recall


def event_confusion(tr):
    """{row type: {AFIB, SVT, VT, AVB: count}}, the 'none' counts, and Se / PPV / F1."""
    k = [PRED_NAMES.index(c) for c in EVENT_CLASSES]
    cm = {t: {c: 0 for c in EVENT_CLASSES} for t in ['AFIB', 'SVT', 'VT', 'AVB2', 'AVB3']}
    none = {t: 0 for t in cm}
    for e, _ref, pred, mean in tr:
        counts = np.array([(pred == j).sum() for j in k])
        if counts.sum() == 0:
            none[e['type']] += 1
            c = EVENT_CLASSES[int(np.argmax(mean[k]))]
        else:
            c = EVENT_CLASSES[int(np.argmax(counts))]
        cm[e['type']][c] += 1
    truth = lambda t: 'AVB' if t in ('AVB2', 'AVB3') else t           # noqa: E731
    stats = {}
    for c in EVENT_CLASSES:
        tp = sum(v[c] for t, v in cm.items() if truth(t) == c)
        ref_n = sum(sum(v.values()) for t, v in cm.items() if truth(t) == c)
        pred_n = sum(v[c] for v in cm.values())
        se, ppv = 100 * tp / max(ref_n, 1), 100 * tp / max(pred_n, 1)
        stats[c] = (se, ppv, 2 * se * ppv / max(se + ppv, 1e-9))
    rows = {t: 100 * cm[t]['AVB'] / max(sum(cm[t].values()), 1) for t in ('AVB2', 'AVB3')}
    return cm, none, stats, rows


def _episodes(mask):
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def epicmp_like(tr):
    """Python twin of the per-class epicmp Gross line (episode: any overlap counts; duration:
    overlap time), for the tuning grids - checked against epicmp in `score`."""
    out = {}
    for c in EVENT_CLASSES:
        ci = PRED_NAMES.index(c)
        rset = [REF_TYPES.index(t) for t in (('AVB2', 'AVB3') if c == 'AVB' else (c,))]
        te = tde = pe = tpe = rs = ps = ov = 0
        for _e, ref, pred, _m in tr:
            rm, pm = np.isin(ref, rset), pred == ci
            for a, b in _episodes(rm):
                te += 1
                tde += bool(pm[a:b].any())
            for a, b in _episodes(pm):
                pe += 1
                tpe += bool(rm[a:b].any())
            rs += rm.sum()
            ps += pm.sum()
            ov += (rm & pm).sum()
        out[c] = (100 * tde / max(te, 1), 100 * tpe / max(pe, 1), 100 * ov / max(rs, 1),
                  100 * ov / max(ps, 1))
    return out


def run_epicmp(tr, work, log=print):
    """Real epicmp, one pass per class over the per-event mark records; Gross E/D Se/+P."""
    from ..evaluation import epicmp
    from ..evaluation.report import parse_gross
    out = {}
    for c in EVENT_CLASSES:
        ci = PRED_NAMES.index(c)
        rset = [REF_TYPES.index(t) for t in (('AVB2', 'AVB3') if c == 'AVB' else (c,))]
        d = os.path.join(work, c)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for e, ref, pred, _m in tr:
            fs = e['fs']
            n = int(round(len(ref) * fs / GRID_HZ))
            with open(os.path.join(d, e['name'] + '.hea'), 'w') as f:
                f.write(f"{e['name']} 0 {fs:g} {n}\n")
            for ext, mask in (('ref', np.isin(ref, rset)), ('hyp', pred == ci)):
                codes = np.where(mask, '(AFIB', '(N')
                ch = np.concatenate([[0], np.flatnonzero(codes[1:] != codes[:-1]) + 1])
                wfdb.Annotation(record_name=e['name'], extension=ext,
                                sample=np.round(ch * fs / GRID_HZ).astype(np.int64),
                                symbol=['+'] * len(ch), aux_note=[codes[i] + '\x00' for i in ch],
                                fs=fs).wrann(write_fs=True, write_dir=d)
        report = epicmp.run_epicmp('strip', d, os.path.join(work, 'reports'), 'ref', 'hyp',
                                   script=epicmp.SCRIPT_SHORT, label=c, quiet=True)
        g = parse_gross(report) if report else {}
        out[c] = tuple(float(g.get(k, 'nan')) if g.get(k, '-') != '-' else float('nan')
                       for k in ('E_Se', 'E_+P', 'D_Se', 'D_+P'))
    return out


# ---------------------------------------------------------------------------
# Comparison with the reference tables
# ---------------------------------------------------------------------------

def compare(m):
    """[(view, item, metric, ours, target, ok)] for every target cell."""
    rows = []
    for c, t in TARGET_EPICMP.items():
        for k, name in enumerate(('E Se', 'E +P', 'D Se', 'D +P')):
            v = m['epicmp'][c][k]
            rows.append(('epicmp', c, name, v, t[k], bool(v >= t[k])))
    for c, t in TARGET_SAMPLE_RECALL.items():
        v = m['sample_recall'][c]
        rows.append(('sample', c, 'recall', v, t, bool(v >= t)))
    for c, t in TARGET_EVENT.items():
        for k, name in enumerate(('Se', 'PPV', 'F1')):
            v = m['event'][c][k]
            rows.append(('event', c, name, v, t[k], bool(v >= t[k])))
    for c, t in TARGET_EVENT_AVB_ROWS.items():
        v = m['event_avb_rows'][c]
        rows.append(('event', c, 'Se (row)', v, t, bool(v >= t)))
    return rows


def margin(m):
    """Smallest (ours - target) over every cell: > 0 = every reference number beaten."""
    return min(r[3] - r[4] for r in compare(m))


def score(tr, work=None, use_epicmp=True, log=print):
    cm, rows, recall = sample_confusion(tr)
    ecm, none, stats, avb_rows = event_confusion(tr)
    m = dict(sample_cm=cm.tolist(), sample_rows=rows.tolist(), sample_recall=recall,
             event_cm=ecm, event_none=none, event=stats, event_avb_rows=avb_rows,
             epicmp_like=epicmp_like(tr), events=len(tr))
    m['epicmp'] = run_epicmp(tr, work, log) if (use_epicmp and work) else m['epicmp_like']
    return m


def format_report(m, title):
    L = [f"## {title} ({m['events']} events)", '',
         '**epicmp, segment to segment (inside the mark)** - ours vs target (E Se / E +P / D Se / D +P)', '',
         '| class | E Se | E +P | D Se | D +P | target |', '|---|---|---|---|---|---|']
    for c in EVENT_CLASSES:
        v, t = m['epicmp'][c], TARGET_EPICMP[c]
        L.append(f"| {c} | " + ' | '.join(f"{x:.0f}{'' if x >= y else ' !'}" for x, y in zip(v, t))
                 + f" | {'/'.join(str(x) for x in t)} |")
    L += ['', '**sample to sample (row %, inside the mark)**', '',
          '| true \\ pred | ' + ' | '.join(PRED_NAMES) + ' | target diag |',
          '|---|' + '---|' * (len(PRED_NAMES) + 1)]
    for i, t in enumerate(REF_TYPES):
        L.append(f"| {t} | " + ' | '.join(f"{x:.2f}" for x in m['sample_rows'][i])
                 + f" | {TARGET_SAMPLE_RECALL[t]} |")
    L += ['', '**event level (one label per event)**', '',
          '| true \\ pred | ' + ' | '.join(EVENT_CLASSES) + ' | (none inside mark) |',
          '|---|---|---|---|---|---|']
    for t, v in m['event_cm'].items():
        L.append(f"| {t} | " + ' | '.join(str(v[c]) for c in EVENT_CLASSES)
                 + f" | {m['event_none'][t]} |")
    L += ['', '| class | Se | PPV | F1 | target Se / PPV / F1 |', '|---|---|---|---|---|']
    for c in EVENT_CLASSES:
        v, t = m['event'][c], TARGET_EVENT[c]
        L.append(f"| {c} | " + ' | '.join(f"{x:.1f}{'' if x >= y else ' !'}" for x, y in zip(v, t))
                 + f" | {'/'.join(str(x) for x in t)} |")
    for c, t in TARGET_EVENT_AVB_ROWS.items():
        v = m['event_avb_rows'][c]
        L.append(f"| {c} row Se | {v:.1f}{'' if v >= t else ' !'} | | | {t} |")
    fails = [r for r in compare(m) if not r[5]]
    L += ['', f"**{len(compare(m)) - len(fails)}/{len(compare(m))} target cells met; smallest "
              f"margin {margin(m):+.1f}**"]
    if fails:
        L.append('below target: ' + ', '.join(f"{v} {c} {k} {o:.1f}<{t}" for v, c, k, o, t, _ in fails))
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------
# Raw model output (no post-processing at all)
# ---------------------------------------------------------------------------

TARGET_RAW_RECALL = {'SINUS': 96.04, 'AFIB': 96.83, 'SVT': 93.56, 'VT': 85.11,
                     'AVB2': 89.70, 'AVB3': 75.47}      # reference confusion matrix (2026-10-13)


def raw_windows(events, cache=None):
    """Model input for the raw protocol: the record cut into consecutive 10 s windows that
    start at the mark (the last one pulled back to end at the mark, or at the record end) -
    no overlap, no test-time variants. Returns (x (w, 2500, 3) float32, index [(event i,
    window start sample at 250 Hz, mark a, mark b at 250 Hz)]). Cached as npz."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z['x'], list(z['index'])
    from ..signal_ops import normalize_window
    from .build import read_leads
    seg = rc.SEGMENT_SAMPLES
    xs, index = [], []
    for i, e in enumerate(events):
        leads, ratio = read_leads(e['record'])
        a = int(round(e['mark'][0] * rc.SAMPLING_RATE / e['fs']))
        b = int(round(e['mark'][1] * rc.SAMPLING_RATE / e['fs']))
        starts = list(range(a, max(a + 1, b - seg + 1), seg)) or [a]
        if starts[-1] + seg < b:
            starts.append(b - seg)
        for s in starts:
            s = max(0, min(s, len(leads) - seg))
            w = leads[s:s + seg]
            if len(w) < seg:
                w = np.concatenate([w, np.zeros((seg - len(w), w.shape[1]), w.dtype)])
            xs.append(normalize_window(w.astype(np.float32)).astype(np.float16))
            index.append((i, s, a, b))
    x = np.stack(xs)
    if cache:
        np.savez(cache, x=x, index=np.array(index, dtype=object))
    return x.astype(np.float32), index


def raw_confusion(model, events, x, index, batch=128, bias=None, out=None):
    """Row-% confusion (REF_TYPES x CLASS_NAMES) of the raw per-step argmax inside the mark,
    and the recall per reference row (AVB2 / AVB3 correct when the prediction is AVB).
    `bias` (K,) is added to the log-probabilities - the output-bias calibration of the
    rhythm head (equivalent to changing the final layer's bias). `out` reuses a prediction."""
    if out is None:                                                # (N, steps, K) rhythm probabilities
        out = model.predict(x.astype(np.float32), batch_size=batch, verbose=0)['rhythm']
    steps = out.shape[1]
    hz = steps / rc.OUTPUT_SECONDS
    logp = np.log(np.clip(out, 1e-7, 1))
    if bias is not None:
        logp = logp + np.asarray(bias)[None, None, :]
    pred = logp.argmax(-1)
    K = rc.NUM_CLASSES
    cm = np.zeros((len(REF_TYPES), K), dtype=np.int64)
    for (i, s, a, b), pw in zip(index, pred):
        e = events[i]
        t = s + np.arange(steps) / hz * rc.SAMPLING_RATE           # step times, 250 Hz samples
        inside = (t >= a) & (t < b)
        ref = np.zeros(steps, int)
        for typ, x0, y0 in e['spans']:
            x0 = x0 * rc.SAMPLING_RATE / e['fs']
            y0 = y0 * rc.SAMPLING_RATE / e['fs']
            ref[(t >= x0) & (t < y0)] = REF_TYPES.index(typ)
        np.add.at(cm, (ref[inside], pw[inside]), 1)
    rows = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    recall = {t: float(rows[i, rc.CLASS_NAMES.index('AVB' if t in ('AVB2', 'AVB3') else t)])
              for i, t in enumerate(REF_TYPES)}
    return cm, rows, recall


def raw_margin(recall):
    return min(recall[t] - v for t, v in TARGET_RAW_RECALL.items())
