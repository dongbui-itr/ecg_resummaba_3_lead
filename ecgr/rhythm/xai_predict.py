"""'Predict vs label' tab of the XAI web service (xai_web.py): one whole record through a
checkpoint - or the probabilities an EC57 run already stored for it - decoded with a chosen
post-processing, next to the record's annotation.

    GET  /api/pred/dbs                    databases (PhysioNet, PhysioNet training, rhythm_eval)
    GET  /api/pred/sources?path=<record>  checkpoints + EC57 runs that stored this record's npz
    GET  /api/pred/presets                decode presets: config defaults + tuned *.json
    POST /api/pred/run                    {path, ann, leads, source, predict, pp} -> tracks, beats,
                                          episodes, per-second and beat scores
    POST /api/pred/signal                 {path, start, span, leads} -> mV leads to draw

Stage 1 (predict.predict_signal: overlapping windows, TTA, taper) is cached per (checkpoint,
record, leads, overlap, TTA, taper), so changing the post-processing only re-runs stage 2
(ec57.decode_record: decode_track -> beat post-processing -> trim -> AF gate) - the exact
path the EC57 scoring takes.
"""
import collections
import contextlib
import glob
import os
import time

import numpy as np

from . import config as rc

CLASS_NAMES = rc.CLASS_NAMES
K = len(CLASS_NAMES)
NOISE = K                                   # decoded track's extra class (labels.DECODE_NAMES)
DECODE_NAMES = list(CLASS_NAMES) + ['NOISE']
BEAT_LETTER = {1: 'N', 2: 'S', 3: 'V'}      # beats.N_BEAT / S_BEAT / V_BEAT
BEAT_TOL = 0.15                             # s, beat match window (EC57 bxb uses 150 ms)
SIGNAL_MAX_SPAN = 120                       # s per /api/pred/signal call
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_MODELS = 2                              # loaded checkpoints kept besides the XAI one
MAX_PREDICTIONS = 8


def _rel(p, base):
    try:
        return os.path.relpath(p, base)
    except ValueError:
        return p


class PredictLab:
    def __init__(self, engine):
        self.e = engine
        self.models = collections.OrderedDict()
        self.preds = collections.OrderedDict()
        self._ckpts = None
        self._presets = None

    # -- catalogues ----------------------------------------------------------------------------
    def dbs(self):
        out = []
        for root, tag in ((self.e.physionet_dir, ''), (rc.PHYSIONET_TRAIN_DIR, 'train/')):
            if not os.path.isdir(root):
                continue
            for d in sorted(os.listdir(root)):
                p = os.path.join(root, d)
                if os.path.isdir(p) and glob.glob(os.path.join(p, '*.hea')):
                    out.append(dict(name=tag + d, dir=p))
        if os.path.isdir(rc.EVAL_DIR):
            out.append(dict(name='rhythm_eval (strip, theo thư mục)', dir=rc.EVAL_DIR))
        return out

    def checkpoints(self, refresh=False):
        if self._ckpts is None or refresh:
            paths = glob.glob(os.path.join(rc.WORK_DIR, '*', 'checkpoints', '*', 'best_model.keras'))
            paths += glob.glob(os.path.join(REPO, 'checkpoints', '*', 'best_model.keras'))
            if self.e.checkpoint and os.path.exists(self.e.checkpoint):
                paths.append(self.e.checkpoint)
            paths = sorted(set(os.path.realpath(p) for p in paths), key=os.path.getmtime, reverse=True)
            self._ckpts = [dict(path=p, label=_rel(os.path.dirname(p), rc.WORK_DIR)
                                if p.startswith(os.path.realpath(rc.WORK_DIR))
                                else 'repo/' + _rel(os.path.dirname(p), os.path.join(REPO, 'checkpoints')),
                                loaded=p == os.path.realpath(self.e.checkpoint or ''))
                           for p in paths]
        return self._ckpts

    def stored_runs(self, path):
        """EC57 runs (<WORK_DIR>/<run>/ec57/<tag>) holding this record's probabilities."""
        db, name = os.path.basename(os.path.dirname(path)), os.path.basename(path)
        hits = glob.glob(os.path.join(rc.WORK_DIR, '*', 'ec57', '*', '_ann', db,
                                      f'{name}.{rc.RHYTHM_PROBS_EXTENSION}'))
        runs = [os.path.dirname(os.path.dirname(os.path.dirname(h))) for h in hits]
        return [dict(ec57_out=r, label=_rel(r, rc.WORK_DIR))
                for r in sorted(runs, key=os.path.getmtime, reverse=True)]

    def presets(self, refresh=False):
        if self._presets is None or refresh:
            files = glob.glob(os.path.join(REPO, 'checkpoints', '*', 'decode', '*.json'))
            files += glob.glob(os.path.join(rc.WORK_DIR, '*', 'ec57', '*', 'tuned_decode.json'))
            files += glob.glob(os.path.join(rc.WORK_DIR, 'summary_*', '**', 'tuned*.json'), recursive=True)
            files = sorted(set(files), key=os.path.getmtime, reverse=True)[:300]
            self._presets = [dict(id='config', label='config mặc định (rc.DECODE_*, beat PP, AF gate)')] + [
                dict(id=f, label=('repo/' + _rel(f, os.path.join(REPO, 'checkpoints'))) if f.startswith(REPO)
                     else _rel(f, rc.WORK_DIR)) for f in files]
        return self._presets

    def sources(self, path):
        path = self.e.resolve_path(path)
        return dict(checkpoints=self.checkpoints(), stored=self.stored_runs(path),
                    defaults=dict(overlap=rc.PREDICT_OVERLAP_SECONDS, tta=list(rc.PREDICT_TTA),
                                  taper=rc.PREDICT_TAPER_FLOOR))

    # -- decode settings -----------------------------------------------------------------------
    def decode_dict(self, pp):
        """Preset (+ overrides from the form) -> ec57.decode_record's `decode`."""
        from .tune import load_decode
        pp = pp or {}
        pid = pp.get('preset') or 'config'
        if pid == 'config':
            d = {}
        else:
            if pid not in {p['id'] for p in self.presets()}:
                raise PermissionError(f'unknown decode preset {pid}')
            d = load_decode(pid)
        if pp.get('beat_pp') is not None:
            d['beat_pp'] = bool(pp['beat_pp'])
        if pp.get('afib_gate') is not None:
            d['afib_gate'] = (d.get('afib_gate') or rc.AFIB_GATE_DEFAULT) if pp['afib_gate'] else False
        for key in ('noise_threshold', 'v_margin', 's_margin'):
            if pp.get(key) not in (None, ''):
                d[key] = float(pp[key])
        for key, default in (('min_prob', rc.DECODE_MIN_EPISODE_PROB),
                             ('min_seconds', rc.DECODE_MIN_EPISODE_SECONDS)):
            over = {c: float(v) for c, v in (pp.get(key) or {}).items() if v not in (None, '')}
            if over:
                d[key] = {**(d.get(key) or default), **over}
        return d

    # -- stage 1 -------------------------------------------------------------------------------
    def _model(self, ckpt):
        ckpt = os.path.realpath(ckpt)
        if self.e.checkpoint and ckpt == os.path.realpath(self.e.checkpoint):
            return self.e.model
        if ckpt not in {c['path'] for c in self.checkpoints()}:
            raise PermissionError(f'unknown checkpoint {ckpt}')
        if ckpt not in self.models:
            from .xai import load_model
            self.models[ckpt] = load_model(ckpt)
            while len(self.models) > MAX_MODELS:
                self.models.popitem(last=False)
        self.models.move_to_end(ckpt)
        return self.models[ckpt]

    @staticmethod
    @contextlib.contextmanager
    def _knobs(overlap, tta, taper):
        old = (rc.PREDICT_OVERLAP_SECONDS, rc.PREDICT_TTA, rc.PREDICT_TAPER_FLOOR)
        try:
            rc.PREDICT_OVERLAP_SECONDS, rc.PREDICT_TTA, rc.PREDICT_TAPER_FLOOR = overlap, tta, taper
            yield
        finally:
            rc.PREDICT_OVERLAP_SECONDS, rc.PREDICT_TTA, rc.PREDICT_TAPER_FLOOR = old

    def _predict(self, path, src, knobs, leads):
        """-> dict(rhythm, p_noise, step_hz, beats, how, elapsed)."""
        from . import ec57
        name = os.path.basename(path)
        if src.get('kind') == 'stored':
            out = src.get('ec57_out')
            if out not in {r['ec57_out'] for r in self.stored_runs(path)}:
                raise PermissionError(f'no stored probabilities of {name} in {out}')
            ann_dir = os.path.join(out, '_ann', os.path.basename(os.path.dirname(path)))
            rhythm, p_noise, _fs, _sl, step_hz = ec57.load_probs(ann_dir, name)
            return dict(rhythm=rhythm, p_noise=p_noise, step_hz=step_hz,
                        beats=ec57.load_beats(ann_dir, name), how=f'npz đã lưu: {_rel(out, rc.WORK_DIR)}',
                        elapsed=0.0, cached=True)
        ckpt = src.get('checkpoint') or self.e.checkpoint
        overlap = float(knobs.get('overlap', rc.PREDICT_OVERLAP_SECONDS))
        tta = tuple(knobs.get('tta') or rc.PREDICT_TTA)
        taper = float(knobs.get('taper', rc.PREDICT_TAPER_FLOOR) or 0)
        key = (os.path.realpath(ckpt), path, tuple(leads) if leads else None, overlap, tta, taper)
        if key in self.preds:
            self.preds.move_to_end(key)
            return dict(self.preds[key], cached=True)
        from .predict import predict_signal, probs_step_hz
        from .xai_web import read_record_leads
        t0 = time.time()
        model = self._model(ckpt)
        h = self.e._header(path)
        x, _ = read_record_leads(path, h.n_sig, leads)
        with self._knobs(overlap, tta, taper):
            rhythm, p_noise, _w, beats = predict_signal(model, x, batch_size=64, with_beats=True)
        res = dict(rhythm=rhythm, p_noise=p_noise, step_hz=probs_step_hz(model), beats=beats,
                   how=f'{_rel(os.path.dirname(key[0]), rc.WORK_DIR)} · overlap {overlap:g} s · '
                       f'TTA {",".join(tta)} · taper {taper:g}', elapsed=round(time.time() - t0, 1))
        self.preds[key] = res
        while len(self.preds) > MAX_PREDICTIONS:
            self.preds.popitem(last=False)
        return dict(res, cached=False)

    # -- reference -----------------------------------------------------------------------------
    def reference(self, path, ann):
        """Annotation (or .hea) reference: episodes [{code, cls, start, stop}], per second class
        (-1 = unknown), beats (t, symbol)."""
        from .xai_web import BEAT_SYMBOLS, header_events
        h = self.e._header(path)
        fs, n = float(h.fs), int(h.sig_len // h.fs)
        marks, (bs, bsym), _other = self.e._annotations(path, ann)
        eps = []
        if marks:
            for (s0, c), (s1, _) in zip(marks, marks[1:] + [(h.sig_len, None)]):
                if s1 > s0:
                    eps.append(dict(code=c, cls=self.e._mark_class(c), start=s0 / fs, stop=s1 / fs))
            source = f'.{ann} rhythm'
        else:
            eps = [dict(code=e['text'], cls=e['cls'], start=e['a'] / fs, stop=e['b'] / fs)
                   for e in header_events(h.comments, h.sig_len)]
            source = '.hea' if eps else 'không có'
        ref = np.full(n, -1, int)
        for e in eps:
            ref[int(e['start']):int(np.ceil(e['stop']))] = e['cls']
        t = np.asarray(bs, float) / fs
        sym = list(bsym)
        return dict(episodes=eps, per_second=ref, source=source, beat_t=t, beat_sym=sym,
                    beat_cls=[BEAT_SYMBOLS.get(s, 'Q') for s in sym], seconds=n, fs=fs,
                    sig_len=int(h.sig_len))

    # -- one run -------------------------------------------------------------------------------
    def run(self, req):
        from .ec57 import decode_record
        t0 = time.time()
        path = self.e.resolve_path(req.get('path'))
        ann = req.get('ann', 'atr')
        leads = req.get('leads')
        leads = None if not leads or list(leads) == [0, 1, 2] else leads
        with self.e.lock:
            P = self._predict(path, req.get('source') or {}, req.get('predict') or {}, leads)
            ref = self.reference(path, ann)
            decode = self.decode_dict(req.get('pp'))
            episodes, symbols = decode_record(P['rhythm'], P['p_noise'], P['step_hz'], P['beats'],
                                              dict(decode))
        n, hz = ref['seconds'], P['step_hz']
        pred = np.full(n, -1, int)
        for e in episodes:
            pred[int(e['start']):int(np.ceil(e['stop']))] = DECODE_NAMES.index(e['rhythm'])
        m = min(n, len(P['rhythm']) // hz)
        ps = P['rhythm'][:m * hz].reshape(m, hz, K).mean(1)
        raw = np.full(n, -1, int)
        raw[:m] = ps.argmax(-1)
        pmax = np.zeros(n)
        pmax[:m] = ps.max(-1)
        beats = P['beats']
        if beats is not None:
            bt = np.asarray(beats['t'], float)
            bcls = np.asarray(symbols if symbols is not None else beats['cls'], int)
            braw = np.asarray(beats['cls'], int)
            bconf = np.asarray(beats['conf'], float)
        else:
            bt, bcls, braw, bconf = np.zeros(0), np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        pcls = ''.join(BEAT_LETTER.get(int(c), 'Q') for c in bcls)
        match, ref_hit = match_beats(ref['beat_t'], bt)
        return dict(
            path=path, name=os.path.basename(path), seconds=n, fs=ref['fs'], how=P['how'],
            elapsed=round(time.time() - t0, 1), predict_elapsed=P['elapsed'], cached=P['cached'],
            decode={k: v for k, v in decode.items() if k != 'class_scale'},
            ref=dict(source=ref['source'], per_second=ref['per_second'].tolist(),
                     episodes=[dict(e, start=round(e['start'], 3), stop=round(e['stop'], 3))
                               for e in ref['episodes']],
                     beat_t=np.round(ref['beat_t'], 3).tolist(), beat_sym=''.join(
                         s if len(s) == 1 else '?' for s in ref['beat_sym']),
                     beat_cls=''.join(ref['beat_cls']), beat_hit=ref_hit.astype(int).tolist()),
            pred=dict(per_second=pred.tolist(), raw=raw.tolist(), pmax=np.round(pmax, 2).tolist(),
                      episodes=[dict(rhythm=e['rhythm'], cls=DECODE_NAMES.index(e['rhythm']),
                                     start=round(float(e['start']), 3), stop=round(float(e['stop']), 3),
                                     prob=round(float(e['prob']), 3)) for e in episodes],
                      beat_t=np.round(bt, 3).tolist(), beat_cls=pcls,
                      beat_raw=''.join(BEAT_LETTER.get(int(c), 'Q') for c in braw),
                      beat_conf=np.round(bconf, 2).tolist(), beat_match=match.tolist()),
            scores=dict(seconds=second_scores(ref['per_second'], pred),
                        raw_seconds=second_scores(ref['per_second'], raw),
                        beats=beat_scores(ref['beat_cls'], pcls, match)
                        if len(ref['beat_t']) and len(bt) else None))

    def signal(self, req):
        from .xai_web import read_record_leads
        path = self.e.resolve_path(req.get('path'))
        h = self.e._header(path)
        span = float(min(max(1.0, float(req.get('span', 10))), SIGNAL_MAX_SPAN))
        total = h.sig_len / h.fs
        start = float(min(max(0.0, float(req.get('start', 0))), max(0.0, total - span)))
        a = int(round(start * h.fs))
        b = min(h.sig_len, a + int(round(span * h.fs)))
        leads = req.get('leads')
        leads = None if not leads or list(leads) == [0, 1, 2] else leads
        w, chans = read_record_leads(path, h.n_sig, leads, chunk=(a, min(h.sig_len, b + int(h.fs))))
        w = w[:int(round(span * rc.SAMPLING_RATE))]
        names, units = list(h.sig_name or []), list(h.units or [])
        return dict(start=round(start, 3), span=span, fs=rc.SAMPLING_RATE,
                    signal=np.round(w.T, 4).tolist(),
                    lead_names=[names[c] if c is not None and c < len(names) else '—' for c in chans],
                    units=[units[c] if c is not None and c < len(units) else '' for c in chans])


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def match_beats(ref_t, pred_t, tol=BEAT_TOL):
    """Nearest-first one-to-one matching within `tol` s -> (pred -> ref index or -1, ref hit)."""
    ref_t, pred_t = np.asarray(ref_t, float), np.asarray(pred_t, float)
    match = np.full(len(pred_t), -1, int)
    hit = np.zeros(len(ref_t), bool)
    if not len(ref_t) or not len(pred_t):
        return match, hit
    pairs = []
    j = np.searchsorted(ref_t, pred_t)
    for i, k in enumerate(j):
        for r in (k - 1, k):
            if 0 <= r < len(ref_t) and abs(ref_t[r] - pred_t[i]) <= tol:
                pairs.append((abs(ref_t[r] - pred_t[i]), i, r))
    for _d, i, r in sorted(pairs):
        if match[i] < 0 and not hit[r]:
            match[i], hit[r] = r, True
    return match, hit


def _ratio(a, b):
    return round(100.0 * a / b, 2) if b else None


def second_scores(ref, pred):
    """Per class over the seconds with a known reference: Se, +P (pred NOISE = no class)."""
    ref, pred = np.asarray(ref), np.asarray(pred)
    m = ref >= 0
    rows = []
    for k, name in enumerate(CLASS_NAMES):
        tp = int(np.sum(m & (ref == k) & (pred == k)))
        fn = int(np.sum(m & (ref == k) & (pred != k)))
        fp = int(np.sum(m & (ref != k) & (pred == k)))
        rows.append(dict(cls=name, ref_s=tp + fn, pred_s=tp + fp, se=_ratio(tp, tp + fn),
                         ppv=_ratio(tp, tp + fp)))
    agree = int(np.sum(m & (ref == pred)))
    return dict(rows=rows, known=int(m.sum()), agree=_ratio(agree, int(m.sum())),
                noise_s=int(np.sum(pred == NOISE)))


def beat_scores(ref_cls, pred_cls, match):
    """QRS detection and V / S classification over the matched beats (bxb-like, simplified:
    reference Q beats count for QRS only)."""
    ref_cls, pred_cls = list(ref_cls), list(pred_cls)
    tp = int(np.sum(np.asarray(match) >= 0))
    out = dict(ref=len(ref_cls), pred=len(pred_cls), tp=tp, qrs_se=_ratio(tp, len(ref_cls)),
               qrs_ppv=_ratio(tp, len(pred_cls)))
    for c in 'VS':
        ref_c = sum(1 for x in ref_cls if x == c)
        pred_c = sum(1 for x in pred_cls if x == c)
        both = sum(1 for i, r in enumerate(match) if r >= 0 and pred_cls[i] == c and ref_cls[r] == c)
        out[c] = dict(ref=ref_c, pred=pred_c, tp=both, se=_ratio(both, ref_c), ppv=_ratio(both, pred_c))
    return out
