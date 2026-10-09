"""Interactive XAI web service for the dual U-Net rhythm model.

    python3 -m ecgr.rhythm xai-web --checkpoint best_model.keras [--port 8777] [--ec57-out DIR]

One page (xai_web.html) and a JSON API over the standard library's http.server - no web
framework is installed on the training machines. Runs on CPU by default (the GPU is shared
with training); one TensorFlow call at a time behind a lock.

What a request can do (POST /api/analyze):
  source        a PhysioNet record + window start, a window of the held-out 'eval' split, or an
                uploaded (n, <=3) array
  edits         signal-domain changes to the 10 s window before z-scoring: f waves, white /
                band (4-9 Hz) noise at an SNR, baseline wander, mains hum, per-lead gain (0 =
                missing lead, < 0 = inverted), lead order, time flip, a flat stretch
  interventions layer-level changes (xai.py probes, made interactive): A branch blind, any
                DiagSSM1D -> identity, uniform / masked lead weights, beat condition off, R-R
                descriptor at its sample mean, the A residual / R-R descriptor of a donor
                window, QRST cancellation window / threshold, lead temperature, SSM kernel cut
  target        class + seconds: the objective sum_s log p(class | s) whose gradient makes
                the heatmaps

and it returns, for the ORIGINAL window and the MODIFIED one (edits + interventions):
p(rhythm) per step and second, beats, noise/channel head, lead weights of both stems, the
atrial residual + QRS mask, gradient x activation of ~35 layers along time (the layer x time
heatmap), integrated gradients on the input. Per-layer channel x time heatmaps come from
GET /api/layer on the cached result.

POST /api/batch runs the same edits / interventions on n eval windows of a category (per
second recall / F1 / mean p before vs after); POST /api/scan slides the window over one
record second (edge effect); POST /api/finding re-measures one finding of md_xai.md and says
whether it holds.
"""
import base64
import collections
import contextlib
import functools
import glob
import json
import os
import threading
import time
import uuid

import numpy as np
import tensorflow as tf

from . import config as rc
from . import xai
from .labels import to_current_classes

CLASS_NAMES = rc.CLASS_NAMES
K = rc.NUM_CLASSES
FS = rc.SAMPLING_RATE
N_SAMPLES = rc.SEGMENT_SAMPLES                     # 2500
OVERVIEW_BINS = 250                                # layer x time heatmap columns (40 ms)
DETAIL_MAX_T = 500                                 # channel x time heatmap columns
HTML = os.path.join(os.path.dirname(__file__), 'xai_web.html')

# (layer, group, output index for list outputs). Missing layers (smaller models) are skipped.
KEY_LAYERS = [
    ('stem1_relu', 'V stem', None), ('stem_pool', 'V stem', None),
    ('v1_b_relu', 'V branch', None), ('v2_b_relu', 'V branch', None),
    ('v3_b_relu', 'V branch', None), ('vb_b_relu', 'V branch', None),
    ('vs_ssm', 'V branch', None), ('vs_out_relu', 'V branch', None),
    ('pos2_b_relu', 'beat head', None), ('pos1_b_relu', 'beat head', None),
    ('beat_pos_prob', 'beat head', None),
    ('qrst_cancel', 'A branch', 0), ('astem1_relu', 'A stem', None), ('astem_pool', 'A stem', None),
    ('a1_d8_res', 'A branch', None), ('a2_d4_res', 'A branch', None),
    ('a3_b_relu', 'A branch', None), ('ab_b_relu', 'A branch', None),
    ('neck3_b_relu', 'neck', None), ('neck2_b_relu', 'neck', None),
    ('neck_ssm0_ssm', 'neck', None), ('neck_ssm0_out_relu', 'neck', None),
    ('neck_ssm1_ssm', 'neck', None), ('neck_ssm1_out_relu', 'neck', None),
    ('type_b_relu', 'beat head', None), ('beat', 'beat head', None),
    ('beat_to_grid', 'rhythm head', None), ('rhythm_fuse_relu', 'rhythm head', None),
    ('rhythm_film', 'rhythm head', None), ('rhythm_ssm_ssm', 'rhythm head', None),
    ('rhythm_ssm_out_relu', 'rhythm head', None), ('rhythm_dec_b_relu', 'rhythm head', None),
    ('rhythm', 'output', None),
]
# layers without a time axis: shown as vectors (value, grad x value)
VECTOR_LAYERS = [('rr_ac', 'rhythm head'), ('rr_film', 'rhythm head')]

BEAT_SYMBOLS = {**{s: 'N' for s in 'NLRej'}, **{s: 'S' for s in 'AaJS'}, **{s: 'V' for s in 'VEF'}}


# ---------------------------------------------------------------------------
# Patching (composable: several wrappers on one layer)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def layer_patches(model, items):
    """items: [(layer name, make_call(orig) -> call)]. Wrappers of one layer are composed in
    order; every patch is removed on exit (xai.patched cannot nest on one layer)."""
    by = collections.OrderedDict()
    for name, make in items:
        by.setdefault(name, []).append(make)
    done = []
    try:
        for name, makes in by.items():
            layer = model.get_layer(name)
            call = layer.call
            for make in makes:
                call = make(call)
            object.__setattr__(layer, 'call', call)
            done.append(layer)
        yield
    finally:
        for layer in done:
            object.__delattr__(layer, 'call')


def _map(name, fn, index=None):
    return xai._map_patch(name, fn, index)


def _replace(name, var, index=None):
    return xai._donor_patch(name, index, var)


def _lead_mask(name, leads):
    """Lead logits of `leads` pushed to -1e4: the softmax over leads gives them no weight."""
    m = np.zeros((1, 3, 1, 1), np.float32)
    m[0, list(leads)] = -1e4
    return _map(name, lambda z: z + tf.constant(m, z.dtype))


def _ssm_cut(model, name, frac):
    layer = model.get_layer(name)

    def make(orig):
        def call(inputs, *a, **k):
            n = max(2, int(layer.kernel_len * frac))
            y = layer._causal_conv(inputs, layer._kernel(layer.fwd)[:, :n])
            if layer.bidirectional:
                kb = layer._kernel(layer.bwd)[:, :n]
                y += tf.reverse(layer._causal_conv(tf.reverse(inputs, [1]), kb), [1])
            return y
        return call
    return (name, make)


@contextlib.contextmanager
def layer_attrs(model, attrs):
    """{layer: {attr: value}} set for the duration of the block (QRSTCancel's pre/post/...)."""
    old = []
    try:
        for name, kv in attrs.items():
            layer = model.get_layer(name)
            for k, v in kv.items():
                old.append((layer, k, getattr(layer, k)))
                object.__setattr__(layer, k, v)
        yield
    finally:
        for layer, k, v in reversed(old):
            object.__setattr__(layer, k, v)


# ---------------------------------------------------------------------------
# Signal edits
# ---------------------------------------------------------------------------

def qrs_scale(raw):
    """Per lead: 99.5th percentile of |x - median| - the QRS amplitude scale (as xai.probe)."""
    return np.percentile(np.abs(raw - np.median(raw, 0)), 99.5, axis=0) + 1e-6


def _leads(spec, default=(0, 1, 2)):
    leads = spec.get('leads', default)
    return [int(c) for c in leads if 0 <= int(c) < 3]


def _band_noise(n, lo, hi, rng):
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = np.fft.rfft(rng.normal(size=n))
    spec[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(spec, n)
    return y / (y.std() + 1e-9)


def apply_edits(raw, edits, rng):
    """(2500, 3) raw window -> edited raw window. Amplitudes are fractions of each lead's QRS
    scale; noise levels are SNRs in dB against each lead's own power."""
    x = np.array(raw, np.float32, copy=True)
    e = edits or {}
    scale = qrs_scale(x)
    power = (x - x.mean(0)).var(0) + 1e-12
    t = np.arange(len(x)) / FS
    if e.get('f_wave', {}).get('on'):
        s = e['f_wave']
        fw = xai.f_waves(1, float(s.get('amp', 0.1)) * scale, rng)[0]
        for c in _leads(s):
            x[:, c] += fw[:, c]
    if e.get('white', {}).get('on'):
        s = e['white']
        for c in _leads(s, (0,)):
            sd = np.sqrt(power[c] / 10 ** (float(s.get('snr_db', 0)) / 10))
            x[:, c] += rng.normal(0, sd, len(x)).astype(np.float32)
    if e.get('band', {}).get('on'):
        s = e['band']
        for c in _leads(s, (0,)):
            sd = np.sqrt(power[c] / 10 ** (float(s.get('snr_db', 6)) / 10))
            x[:, c] += sd * _band_noise(len(x), float(s.get('lo', 4)), float(s.get('hi', 9)), rng)
    if e.get('wander', {}).get('on'):
        s = e['wander']
        for c in _leads(s):
            f0 = float(s.get('hz', 0.25))
            x[:, c] += float(s.get('amp', 0.5)) * scale[c] * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
    if e.get('mains', {}).get('on'):
        s = e['mains']
        for c in _leads(s):
            x[:, c] += float(s.get('amp', 0.1)) * scale[c] * np.sin(2 * np.pi * float(s.get('hz', 50)) * t)
    if e.get('flat', {}).get('on'):
        s = e['flat']
        a = int(max(0, float(s.get('from', 0))) * FS)
        b = int(min(rc.OUTPUT_SECONDS, float(s.get('to', 1))) * FS)
        for c in _leads(s):
            x[a:b, c] = x[a:b, c].mean() if b > a else x[a:b, c]
    gain = e.get('gain')
    if gain:
        x = x * np.asarray(gain, np.float32)[None, :3]
    order = e.get('order')
    if order and sorted(int(o) for o in order) == [0, 1, 2]:
        x = x[:, [int(o) for o in order]]
    if e.get('flip'):
        x = x[::-1].copy()
    return x


def normalize(raw):
    from ..signal_ops import normalize_window
    return normalize_window(np.asarray(raw, np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------

def _resample_time(a, bins):
    """(T, ...) -> (bins, ...): block mean when T > bins, repeat when T < bins."""
    a = np.asarray(a, np.float32)
    t = a.shape[0]
    if t == bins:
        return a
    if t > bins:
        if t % bins == 0:
            return a.reshape((bins, t // bins) + a.shape[1:]).mean(1)
        idx = (np.arange(bins + 1) * t / bins).astype(int)
        return np.stack([a[idx[i]:max(idx[i + 1], idx[i] + 1)].mean(0) for i in range(bins)])
    return a[np.minimum((np.arange(bins) * t / bins).astype(int), t - 1)]


def _as_time_channels(v):
    """Layer output of one window -> (T, C): (c, T, d) per-lead maps become T x (c * d)."""
    v = np.asarray(v, np.float32)
    if v.ndim == 3:                                           # (leads, T, d)
        return v.transpose(1, 0, 2).reshape(v.shape[1], -1)
    if v.ndim == 1:
        return v[:, None]
    return v


def _b64_u8(img):
    return base64.b64encode(np.ascontiguousarray(img, np.uint8).tobytes()).decode('ascii')


def _round(a, nd=4):
    return np.round(np.asarray(a, np.float64), nd).tolist()


def _softmax_leads(logit):
    z = np.asarray(logit, np.float64)[..., 0]                  # (3, 50)
    z = z - z.max(0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(0, keepdims=True)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, checkpoint, ec57_out=None, npy_dir=None, physionet_dir=None, model=None):
        from .. import config as base_config
        self.checkpoint = checkpoint
        self.model = model if model is not None else xai.load_model(checkpoint)
        names = {l.name for l in self.model.layers}
        self.key_layers = [(n, g, i) for n, g, i in KEY_LAYERS if n in names]
        self.vector_layers = [(n, g) for n, g in VECTOR_LAYERS if n in names]
        self.ssm = xai.ssm_layers(self.model)
        self.physionet_dir = physionet_dir or base_config.PHYSIONET_DIR
        self.npy_dir = npy_dir or rc.NPY_DIR
        self.ec57_out = ec57_out
        self.lock = threading.RLock()
        self.cache = collections.OrderedDict()                 # analysis id -> captured arrays
        self._eval = None
        self._rr_mean = None
        self._sub = {}

    # -- meta / sources ----------------------------------------------------------------------
    def meta(self):
        dbs = sorted(d for d in os.listdir(self.physionet_dir)
                     if os.path.isdir(os.path.join(self.physionet_dir, d))) \
            if os.path.isdir(self.physionet_dir) else []
        return dict(checkpoint=self.checkpoint, classes=CLASS_NAMES, fs=FS,
                    seconds=rc.OUTPUT_SECONDS, dbs=dbs,
                    eval_available=os.path.isdir(os.path.join(self.npy_dir, 'eval')),
                    ec57_out=self.ec57_out, ssm_layers=self.ssm,
                    layers=[dict(name=n, group=g) for n, g, _ in self.key_layers],
                    vector_layers=[dict(name=n, group=g) for n, g in self.vector_layers],
                    bookmarks=BOOKMARKS)

    def records(self, db):
        d = os.path.join(self.physionet_dir, os.path.basename(db))
        if not os.path.isdir(d):
            return []
        out = []
        for hea in sorted(glob.glob(os.path.join(d, '*.hea'))):
            name = os.path.basename(hea)[:-4]
            if not os.path.exists(os.path.join(d, name + '.dat')):
                continue
            try:
                h = self._header(os.path.join(d, name))
                out.append(dict(name=name, seconds=int(h.sig_len / h.fs), fs=h.fs,
                                atr=os.path.exists(os.path.join(d, name + '.atr'))))
            except Exception:                                    # noqa: BLE001
                continue
        return out

    @functools.lru_cache(maxsize=512)
    def _header(self, path):
        import wfdb
        return wfdb.rdheader(path)

    @functools.lru_cache(maxsize=64)
    def _annotations(self, path):
        """(rhythm marks [(sample, code)], beats (samples, symbols)) at the record's own rate."""
        import wfdb
        if not os.path.exists(path + '.atr'):
            return [], (np.zeros(0, int), [])
        a = wfdb.rdann(path, 'atr')
        aux = [(x or '').split('\x00')[0].strip() for x in a.aux_note]
        marks = [(int(s), c) for s, c, sym in zip(a.sample, aux, a.symbol)
                 if sym == '+' and c.startswith('(')]
        keep = [i for i, s in enumerate(a.symbol) if s in BEAT_SYMBOLS or s in 'Q/f']
        return marks, (np.asarray(a.sample)[keep], [a.symbol[i] for i in keep])

    def _record_window(self, db, name, start):
        from .build import read_leads
        path = os.path.join(self.physionet_dir, os.path.basename(db), os.path.basename(name))
        h = self._header(path)
        total = int(h.sig_len / h.fs)
        start = int(min(max(0, start), max(0, total - rc.OUTPUT_SECONDS)))
        a, b = int(start * h.fs), int((start + rc.OUTPUT_SECONDS) * h.fs)
        leads, _ = read_leads(path, chunk=(a, min(b + int(h.fs), h.sig_len)))
        w = leads[:N_SAMPLES]
        if len(w) < N_SAMPLES:
            w = np.concatenate([w, np.zeros((N_SAMPLES - len(w), w.shape[1]), w.dtype)])
        marks, (bs, bsym) = self._annotations(path)
        ref, codes = [], []
        for k in range(rc.OUTPUT_SECONDS):
            mid = (start + k + 0.5) * h.fs
            code = '(N'
            for s, c in marks:
                if s <= mid:
                    code = c
                else:
                    break
            codes.append(code if marks else '')
            cls = rc.PHYSIONET_AUX_TO_CLASS.get(code)
            if code == '(AFL':                                   # project convention: AFL = AF
                cls = 'AFIB'
            ref.append(CLASS_NAMES.index(cls) if cls else (rc.SINUS if marks else -1))
        sel = (bs >= a) & (bs < b)
        beats = [dict(t=float((s - a) / h.fs), sym=sym, cls=BEAT_SYMBOLS.get(sym, 'Q'))
                 for s, sym in zip(bs[sel], np.asarray(bsym, object)[sel])]
        info = dict(kind='record', title=f"{db}/{name}  {start}-{start + rc.OUTPUT_SECONDS} s",
                    db=db, record=name, start=start, total_seconds=total, ref=ref,
                    ref_codes=codes, ref_beats=beats)
        return np.asarray(w, np.float32), info

    def _eval_data(self):
        """mmap'd eval shards: (segments list, labels (n, 10), category (n,), offsets)."""
        if self._eval is None:
            from .pipeline import is_legacy, read_manifest, window_categories
            from .labels import to_current_labels
            root = os.path.join(self.npy_dir, 'eval')
            shards = sorted(glob.glob(os.path.join(root, 'segments_*.npy')),
                            key=lambda p: int(p.rsplit('_', 1)[1][:-4]))
            if not shards:
                raise FileNotFoundError(f"no eval shards under {root}")
            segs = [np.load(p, mmap_mode='r') for p in shards]
            labs = np.concatenate([np.load(p.replace('segments_', 'labels_')) for p in shards])
            manifest = os.path.join(self.npy_dir, 'manifest.json')
            if os.path.exists(manifest) and is_legacy(read_manifest(self.npy_dir)):
                labs = to_current_labels(labs)
            offsets = np.cumsum([0] + [len(s) for s in segs])
            self._eval = (segs, labs.astype(np.int64), window_categories(labs), offsets)
        return self._eval

    def eval_window(self, index):
        segs, labs, cat, off = self._eval_data()
        index = int(index) % int(off[-1])
        k = int(np.searchsorted(off, index, side='right') - 1)
        w = np.asarray(segs[k][index - off[k]], np.float32)
        info = dict(kind='eval', title=f"eval #{index}  ({CLASS_NAMES[cat[index]]} window)",
                    index=index, ref=[int(v) if v < K else -1 for v in labs[index]],
                    ref_codes=[], ref_beats=[], category=CLASS_NAMES[cat[index]])
        return w, info

    def eval_sample(self, category, n, seed=0):
        """Indices of n eval windows of `category` (a name, or 'ALL' = class-balanced)."""
        _, labs, cat, _ = self._eval_data()
        rng = np.random.default_rng(seed)
        if category == 'ALL':
            per = max(1, n // K)
            idx = []
            for c in range(K):
                pool = np.flatnonzero(cat == c)
                idx += list(rng.choice(pool, min(per, len(pool)), replace=False)) if len(pool) else []
            return np.array(sorted(idx))
        pool = np.flatnonzero(cat == CLASS_NAMES.index(category))
        return np.sort(rng.choice(pool, min(n, len(pool)), replace=False)) if len(pool) else np.zeros(0, int)

    def eval_batch(self, idx):
        segs, labs, _, off = self._eval_data()
        raw = []
        for i in idx:
            k = int(np.searchsorted(off, i, side='right') - 1)
            raw.append(np.asarray(segs[k][i - off[k]], np.float32))
        return np.stack(raw) if raw else np.zeros((0, N_SAMPLES, 3), np.float32), labs[idx]

    def load_source(self, src):
        kind = src.get('kind', 'record')
        if kind == 'record':
            return self._record_window(src['db'], src['record'], int(src.get('start', 0)))
        if kind == 'eval':
            return self.eval_window(src.get('index', 0))
        if kind == 'array':
            from ..signal_ops import resample_leads
            a = np.asarray(src['data'], np.float32)
            if a.ndim == 1:
                a = a[:, None]
            if a.shape[0] <= 3 and a.shape[1] > 3:
                a = a.T
            fs = float(src.get('fs', FS))
            if fs != FS:
                a = resample_leads(a, fs, FS).astype(np.float32)
            a = a[:, :3]
            if a.shape[1] < 3:
                a = np.concatenate([a, np.zeros((len(a), 3 - a.shape[1]), np.float32)], 1)
            a = a[:N_SAMPLES]
            if len(a) < N_SAMPLES:
                a = np.concatenate([a, np.zeros((N_SAMPLES - len(a), 3), np.float32)])
            return a, dict(kind='array', title=src.get('name', 'uploaded window'),
                           ref=[-1] * rc.OUTPUT_SECONDS, ref_codes=[], ref_beats=[])
        raise ValueError(f"unknown source kind {kind!r}")

    def record_overview(self, db, name):
        """Whole-record per-second track: stored EC57 probabilities (argmax, p) of --ec57-out
        if present, and the reference class per second - to find error stretches."""
        path = os.path.join(self.physionet_dir, os.path.basename(db), os.path.basename(name))
        h = self._header(path)
        n = int(h.sig_len / h.fs)
        marks, _ = self._annotations(path)
        ref = np.full(n, -1 if not marks else rc.SINUS, int)
        for (s0, c), (s1, _) in zip(marks, marks[1:] + [(h.sig_len, None)]):
            cls = 'AFIB' if c == '(AFL' else rc.PHYSIONET_AUX_TO_CLASS.get(c)
            if cls:
                ref[int(s0 / h.fs):int(np.ceil(s1 / h.fs))] = CLASS_NAMES.index(cls)
        out = dict(seconds=n, ref=ref.tolist(), pred=None, p=None)
        if self.ec57_out:
            from .ec57 import load_probs
            got = load_probs(os.path.join(self.ec57_out, '_ann', os.path.basename(db)),
                             os.path.basename(name))
            if got is not None:
                r, _pn, _fs, _sl, hz = got
                m = min(n, len(r) // hz)
                ps = r[:m * hz].reshape(m, hz, K).mean(1)
                out['pred'] = ps.argmax(-1).tolist()
                out['p'] = _round(ps.max(-1), 3)
        return out

    # -- interventions -----------------------------------------------------------------------
    def _rr_mean_value(self):
        if self._rr_mean is None:
            idx = self.eval_sample('ALL', 200, seed=7)
            raw, _ = self.eval_batch(idx)
            x = np.stack([normalize(w) for w in raw])
            sm = self._submodel(('rr_ac',))
            self._rr_mean = np.concatenate([np.asarray(sm(x[i:i + 64], training=False))
                                            for i in range(0, len(x), 64)]).mean(0)
        return self._rr_mean

    def _submodel(self, names):
        if names not in self._sub:
            self._sub[names] = xai.sub_model(self.model, list(names))
        return self._sub[names]

    def _donor_values(self, iv, n):
        """Donor A residual / R-R descriptor from a donor source, computed BEFORE any patch."""
        d = iv.get('donor')
        if not d or not (iv.get('atrial_donor') or iv.get('rr_donor')):
            return {}
        raw, _ = self.load_source(d)
        x = normalize(raw)[None]
        res, rr = self._donor_pair(x)
        out = {}
        if iv.get('atrial_donor'):
            out['atrial'] = np.repeat(res, n, 0)
        if iv.get('rr_donor'):
            out['rr'] = np.repeat(rr, n, 0)
        return out

    def _donor_pair(self, x):
        o = self._submodel(('qrst_cancel', 'rr_ac'))(x, training=False)
        return np.asarray(o[0]), np.asarray(o[2])

    def intervention_items(self, iv, donor_vars=None):
        """-> (patch items, layer attrs) for an interventions dict."""
        iv = iv or {}
        items, attrs = [], {}
        donor_vars = donor_vars or {}
        if iv.get('atrial_off'):
            items.append(_map('qrst_cancel', tf.zeros_like, index=0))
        elif 'atrial' in donor_vars:
            items.append(_replace('qrst_cancel', donor_vars['atrial'], index=0))
        if iv.get('beat_cond_off'):
            items.append(_map('beat_to_grid', lambda b: tf.concat(
                [tf.ones_like(b[..., :1]), tf.zeros_like(b[..., 1:])], -1)))
        if iv.get('rr_mean'):
            items.append(_map('rr_ac', lambda z, m=self._rr_mean_value():
                              tf.zeros_like(z) + tf.constant(m[None], z.dtype)))
        elif 'rr' in donor_vars:
            items.append(_replace('rr_ac', donor_vars['rr']))
        temp = float(iv.get('lead_temperature', 1.0) or 1.0)
        for logit, uniform, off in (('lead_w_logit', 'v_lead_uniform', 'v_lead_off'),
                                    ('alead_w_logit', 'a_lead_uniform', 'a_lead_off')):
            if iv.get(uniform):
                items.append(_map(logit, tf.zeros_like))
            elif temp != 1.0:
                items.append(_map(logit, lambda z, s=1.0 / temp: z * s))
            if iv.get(off):
                items.append(_lead_mask(logit, iv[off]))
        for name in iv.get('ssm_identity') or []:
            if name in self.ssm:
                items.append((name, lambda orig: (lambda inputs, *a, **k: inputs)))
        frac = float(iv.get('ssm_kernel_frac', 1.0) or 1.0)
        if frac < 1.0:
            items += [_ssm_cut(self.model, n, frac) for n in self.ssm
                      if n not in (iv.get('ssm_identity') or [])]
        q = {k: float(iv[f'qrst_{k}']) for k in ('pre', 'post', 'min_prob')
             if iv.get(f'qrst_{k}') not in (None, '')}
        if q:
            attrs['qrst_cancel'] = q
        return items, attrs

    @staticmethod
    def active(iv):
        iv = iv or {}
        return bool(iv.get('atrial_off') or iv.get('atrial_donor') or iv.get('rr_donor')
                    or iv.get('beat_cond_off') or iv.get('rr_mean') or iv.get('v_lead_uniform')
                    or iv.get('a_lead_uniform') or iv.get('v_lead_off') or iv.get('a_lead_off')
                    or iv.get('ssm_identity') or float(iv.get('lead_temperature', 1) or 1) != 1
                    or float(iv.get('ssm_kernel_frac', 1) or 1) < 1
                    or any(iv.get(f'qrst_{k}') not in (None, '') for k in ('pre', 'post', 'min_prob')))

    # -- forward passes ----------------------------------------------------------------------
    def forward_batch(self, x, iv=None, batch=64):
        """Graph-mode forward of many windows with the interventions -> rhythm (n, 1250, K)."""
        donors = self._donor_values(iv or {}, batch)
        dvars = {k: tf.Variable(v.astype(np.float32), trainable=False) for k, v in donors.items()}
        items, attrs = self.intervention_items(iv, dvars)
        acc = []
        with layer_patches(self.model, items), layer_attrs(self.model, attrs):
            f = tf.function(lambda z: self.model(z, training=False)['rhythm'])
            for i in range(0, len(x), batch):
                xb = x[i:i + batch]
                pad = batch - len(xb)
                if pad:
                    xb = np.concatenate([xb, np.zeros((pad,) + xb.shape[1:], xb.dtype)])
                acc.append(np.asarray(f(tf.constant(xb)))[:batch - pad])
        return np.concatenate(acc) if acc else np.zeros((0, rc.BEAT_STEPS, K), np.float32)

    def forward_explain(self, x, iv, cls, seconds):
        """One window, eager, with every key layer captured and the gradient of
        sum_{s in seconds} log p(cls | s) w.r.t. each captured output."""
        donors = self._donor_values(iv or {}, 1)
        dvars = {k: tf.constant(v.astype(np.float32)) for k, v in donors.items()}
        items, attrs = self.intervention_items(iv, dvars)
        store = {}
        tape = tf.GradientTape(persistent=False)

        def capture(name, index):
            def make(orig):
                def call(inputs, *a, **k):
                    out = orig(inputs, *a, **k)
                    o = out[index] if isinstance(out, (list, tuple)) else out
                    tape.watch(o)
                    store[name] = o
                    return out
                return call
            return (name, make)
        cap = [capture(n, i) for n, _, i in self.key_layers] + \
              [capture(n, None) for n, _ in self.vector_layers] + \
              [capture(n, None) for n in ('lead_w_logit', 'alead_w_logit')]
        cap.append(('qrst_cancel', self._mask_grab(store)))
        sel = np.zeros(rc.OUTPUT_SECONDS, np.float32)
        sel[[s for s in seconds if 0 <= s < rc.OUTPUT_SECONDS]] = 1
        with layer_patches(self.model, items + cap), layer_attrs(self.model, attrs):
            with tape:
                out = self.model(tf.constant(x[None]), training=False)
                y = out['rhythm']
                ps = tf.reduce_mean(tf.reshape(y, (1, rc.OUTPUT_SECONDS, -1, y.shape[-1])), 2)
                obj = tf.reduce_sum(tf.math.log(ps[0, :, cls] + 1e-6) * tf.constant(sel))
            names = list(store)
            grads = tape.gradient(obj, [store[n] for n in names],
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        acts = {n: np.asarray(store[n])[0] for n in names}
        grads = {n: np.asarray(g)[0] for n, g in zip(names, grads)}
        outs = {k: np.asarray(v)[0] for k, v in out.items()}
        return outs, acts, grads, float(obj)

    @staticmethod
    def _mask_grab(store):
        def make(orig):
            def call(inputs, *a, **k):
                out = orig(inputs, *a, **k)
                store['__qrs_mask'] = out[1]
                return out
            return call
        return make

    def integrated_gradients(self, x, iv, cls, seconds, steps=16):
        donors = self._donor_values(iv or {}, 1)
        if donors:                                              # donor shapes are per window
            donors = {k: np.repeat(v[:1], steps, 0) for k, v in donors.items()}
        dvars = {k: tf.constant(v.astype(np.float32)) for k, v in donors.items()}
        items, attrs = self.intervention_items(iv, dvars)
        sel = np.zeros(rc.OUTPUT_SECONDS, np.float32)
        sel[[s for s in seconds if 0 <= s < rc.OUTPUT_SECONDS]] = 1
        xt = tf.constant(x[None], tf.float32)
        alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1)[1:], (-1, 1, 1))
        with layer_patches(self.model, items), layer_attrs(self.model, attrs):
            xi = alphas * xt
            with tf.GradientTape() as tape:
                tape.watch(xi)
                y = self.model(xi, training=False)['rhythm']
                ps = tf.reduce_mean(tf.reshape(y, (steps, rc.OUTPUT_SECONDS, -1, y.shape[-1])), 2)
                obj = tf.reduce_sum(tf.math.log(ps[..., cls] + 1e-6) * tf.constant(sel))
            g = tape.gradient(obj, xi)
        return (xt * tf.reduce_mean(g, 0, keepdims=True)).numpy()[0]

    # -- analyze -----------------------------------------------------------------------------
    def analyze(self, req):
        t0 = time.time()
        with self.lock:
            raw, info = self.load_source(req.get('source', {}))
            edits = req.get('edits') or {}
            iv = req.get('interventions') or {}
            seed = int(req.get('seed', 0))
            raw_mod = apply_edits(raw, edits, np.random.default_rng(seed))
            flip = bool(edits.get('flip'))
            x0, x1 = normalize(raw), normalize(raw_mod)
            # target: class + seconds (default: argmax of the modified window at its centre)
            tgt = req.get('target') or {}
            seconds = [int(s) for s in tgt.get('seconds') or range(rc.OUTPUT_SECONDS)]
            seconds_in = [rc.OUTPUT_SECONDS - 1 - s for s in seconds] if flip else seconds
            cls = tgt.get('cls')
            if cls in (None, '', 'auto'):
                p = xai.per_second(self.forward_batch(x1[None], iv, batch=1))[0]
                cls = int(p[seconds_in].mean(0).argmax())
            cls = CLASS_NAMES.index(cls) if isinstance(cls, str) else int(cls)
            runs, cached = {}, {}
            for key, x, ivk, fl, sec in (('orig', x0, {}, False, seconds),
                                         ('mod', x1, iv, flip, seconds_in)):
                outs, acts, grads, obj = self.forward_explain(x, ivk, cls, sec)
                ig = self.integrated_gradients(x, ivk, cls, sec, int(req.get('ig_steps', 16))) \
                    if req.get('ig', True) else None
                runs[key] = self._summarize(outs, acts, grads, obj, ig, fl)
                cached[key] = dict(acts=acts, grads=grads, flip=fl)
            aid = uuid.uuid4().hex[:12]
            self.cache[aid] = cached
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
            disp = lambda a: a[::-1] if flip else a                 # noqa: E731
        return dict(id=aid, info=info, target=dict(cls=cls, name=CLASS_NAMES[cls], seconds=seconds),
                    signal=dict(orig=_round(x0.T, 3), mod=_round(disp(x1).T, 3)),
                    flip=flip, runs=runs, edits_active=_edits_active(edits),
                    interventions_active=self.active(iv), elapsed=round(time.time() - t0, 2))

    def _summarize(self, outs, acts, grads, obj, ig, flip):
        fl = (lambda a: a[::-1]) if flip else (lambda a: a)   # noqa: E731  back to real time
        rhythm = fl(to_current_classes(outs['rhythm'][None])[0])
        ps = rhythm.reshape(rc.OUTPUT_SECONDS, -1, K).mean(1)
        from .beats import pick_beats
        beat = fl(outs['beat']) if 'beat' in outs else None
        bt = pick_beats(beat, rc.BEAT_STEPS // rc.OUTPUT_SECONDS) if beat is not None else None
        layers = []
        for name, group, _ in self.key_layers + [(n, g, None) for n, g in self.vector_layers]:
            if name not in acts:
                continue
            a, g = acts[name], grads[name]
            if name in dict(self.vector_layers):
                rel = a * g
                layers.append(dict(name=name, group=group, vector=True, n=int(a.size),
                                   value=_round(a.ravel(), 3), rel=_round(rel.ravel(), 5),
                                   rel_total=float(np.abs(rel).sum()),
                                   grad_norm=float(np.abs(g).sum())))
                continue
            a2, g2 = fl(_as_time_channels(a)), fl(_as_time_channels(g))
            rel = a2 * g2                                         # (T, C) gradient x activation
            prof = _resample_time(np.abs(rel).sum(1), OVERVIEW_BINS)
            sprof = _resample_time(rel.sum(1), OVERVIEW_BINS)
            aprof = _resample_time(np.sqrt((a2 ** 2).mean(1)), OVERVIEW_BINS)
            layers.append(dict(name=name, group=group, vector=False, T=int(a2.shape[0]),
                               C=int(a2.shape[1]), shape=list(np.shape(a)),
                               rel_profile=_round(prof, 6), rel_signed=_round(sprof, 6),
                               act_profile=_round(aprof, 4),
                               rel_total=float(np.abs(rel).sum()),
                               grad_zero=bool(not np.any(g2))))
        out = dict(per_second=_round(ps, 4), argmax=ps.argmax(-1).tolist(),
                   rhythm=_round(_resample_time(rhythm, OVERVIEW_BINS), 4),
                   objective=obj, layers=layers)
        if bt is not None:
            out['beats'] = [dict(t=float(t), cls='NSV'[int(c) - 1], conf=float(p))
                            for t, c, p in zip(bt['t'], bt['cls'], bt['conf'])]
        if 'channel' in outs:
            out['channel'] = _round(outs['channel'], 3)
        for k, src in (('lead_w_v', 'lead_w_logit'), ('lead_w_a', 'alead_w_logit')):
            if src in acts:
                w = _softmax_leads(acts[src])
                out[k] = _round(w[:, ::-1] if flip else w, 3)
        if 'qrst_cancel' in acts:
            out['residual'] = _round(fl(acts['qrst_cancel']).T, 3)
        if '__qrs_mask' in acts:
            out['qrs_mask'] = _round(fl(acts['__qrs_mask'])[:, 0], 2)
        if 'rr_ac' in acts:
            out['rr_ac'] = _round(acts['rr_ac'].ravel(), 3)
        if ig is not None:
            out['ig'] = _round(fl(ig).T, 6)
        return out

    def layer_detail(self, aid, name, run='mod'):
        """Channel x time heatmaps of one layer of a cached analysis: activation (uint8, min/
        max given) and gradient x activation (uint8 around 128 = 0, symmetric scale)."""
        with self.lock:
            c = self.cache.get(aid)
            if c is None or name not in c[run]['acts']:
                raise KeyError('analysis expired or layer not captured - run analyze again')
            r = c[run]
        fl = (lambda a: a[::-1]) if r['flip'] else (lambda a: a)  # noqa: E731
        a = fl(_as_time_channels(r['acts'][name]))
        g = fl(_as_time_channels(r['grads'][name]))
        rel = a * g
        T = min(DETAIL_MAX_T, a.shape[0])
        a_d, rel_d = _resample_time(a, T), _resample_time(rel, T)
        lo, hi = float(np.percentile(a_d, 1)), float(np.percentile(a_d, 99))
        img_a = np.clip((a_d - lo) / (hi - lo + 1e-9) * 255, 0, 255).T
        m = float(np.percentile(np.abs(rel_d), 99.5)) + 1e-12
        img_r = np.clip(rel_d / m * 127.5 + 127.5, 0, 255).T
        ch_rel = np.abs(rel).sum(0)
        top = np.argsort(-ch_rel)[:12]
        return dict(name=name, run=run, T=int(T), C=int(a.shape[1]), act=_b64_u8(img_a),
                    act_range=[lo, hi], rel=_b64_u8(img_r), rel_scale=m,
                    channel_rel=_round(ch_rel, 6),
                    top_channels=[dict(ch=int(i), rel=float(ch_rel[i]),
                                       signed=float(rel[:, i].sum())) for i in top])

    # -- batch / scan --------------------------------------------------------------------------
    def batch(self, req):
        t0 = time.time()
        with self.lock:
            n = int(min(max(int(req.get('n', 60)), 5), 600))
            idx = self.eval_sample(req.get('category', 'ALL'), n, int(req.get('seed', 0)))
            raw, labels = self.eval_batch(idx)
            rng = np.random.default_rng(int(req.get('seed', 0)))
            edits, iv = req.get('edits') or {}, req.get('interventions') or {}
            x0 = np.stack([normalize(w) for w in raw])
            x1 = np.stack([normalize(apply_edits(w, edits, rng)) for w in raw])
            p0 = xai.per_second(self.forward_batch(x0, {}))
            p1 = xai.per_second(self.forward_batch(x1, iv))
            if edits.get('flip'):
                p1 = p1[:, ::-1]
        return dict(n=int(len(idx)), category=req.get('category', 'ALL'),
                    orig=_batch_summary(p0, labels), mod=_batch_summary(p1, labels),
                    max_abs_dp=float(np.abs(p1 - p0).max()) if len(idx) else 0.0,
                    elapsed=round(time.time() - t0, 2))

    def scan(self, req):
        """Slide the 10 s window over one record second s: the window starting at s-9 .. s
        puts s at position 9 .. 0. p(class | s) per position, for the ORIGINAL signal with the
        interventions applied (edits are window-relative and left out)."""
        with self.lock:
            src = req['source']
            second = int(req['second'])
            rows, xs = [], []
            for pos in range(rc.OUTPUT_SECONDS):
                start = second - pos
                if start < 0:
                    continue
                raw, info = self._record_window(src['db'], src['record'], start)
                if info['start'] != start:
                    continue
                xs.append(normalize(raw))
                rows.append(dict(position=pos, start=start, ref=info['ref'][pos]))
            ps = xai.per_second(self.forward_batch(np.stack(xs), req.get('interventions') or {}))
            for r, p in zip(rows, ps):
                r['p'] = _round(p[r['position']], 4)
                r['argmax'] = int(p[r['position']].argmax())
        return dict(second=second, rows=rows)


def _edits_active(e):
    e = e or {}
    return bool(any((e.get(k) or {}).get('on') for k in ('f_wave', 'white', 'band', 'wander',
                                                         'mains', 'flat'))
                or e.get('flip') or (e.get('order') and list(e['order']) != [0, 1, 2])
                or (e.get('gain') and list(e['gain']) != [1, 1, 1]))


def _batch_summary(p, labels):
    s = xai.scores(p, labels)
    share = np.bincount(p.argmax(-1).ravel(), minlength=K) / max(p.shape[0] * p.shape[1], 1)
    return dict(scores=s, argmax_share=dict(zip(CLASS_NAMES, _round(share, 4))),
                mean_p=dict(zip(CLASS_NAMES, _round(p.mean((0, 1)), 4))))


# ---------------------------------------------------------------------------
# Findings of md_xai.md, re-measured
# ---------------------------------------------------------------------------

BOOKMARKS = [
    dict(label='mitdb 221 s1041 - AFIB FN at the window edge (finding 1)', db='mitdb', record='221', second=1041),
    dict(label='mitdb 200 s149 - AFIB FP on a noisy lead (finding 4/5)', db='mitdb', record='200', second=149),
    dict(label='mitdb 223 s605 - VT lost to beat-PP HR rule (finding 6)', db='mitdb', record='223', second=605),
    dict(label='mitdb 207 s1617 - VFL called VT (finding 6)', db='mitdb', record='207', second=1617),
    dict(label='mitdb 231 s146 - 2nd-degree AV block', db='mitdb', record='231', second=146),
    dict(label='mitdb 202 s1534 - SVT / AFL', db='mitdb', record='202', second=1534),
    dict(label='mitdb 219 s1749 - AFIB FN', db='mitdb', record='219', second=1749),
    dict(label='mitdb 234 s858 - SVT', db='mitdb', record='234', second=858),
]

FINDINGS = {
    'edge': "Hiệu ứng mép window: F1 theo giây ở giữa window cao hơn rõ rệt so với ở hai mép",
    'swap': "TTA 'swap' (đổi thứ tự lead) không đổi output của dual U-Net; 'flip' (đảo dấu) thì có",
    'v_ssm': "Quyết định dựa vào nhánh V + ngữ cảnh SSM: SSM->identity sập macro F1; làm mù nhánh A "
             "giảm vừa phải (AVB, VT); RR descriptor / điều kiện beat gần như không dùng",
    'donor': "Bằng chứng nhĩ không đặc hiệu: thay residual nhĩ / RR bằng của window lớp khác đổi "
             "p(lớp) <= 0.05",
    'f_wave': "Sóng f đi vào qua nhánh V: chèn sóng f 10 % QRS vào SINUS tạo AFIB, vẫn còn khi nhánh A mù",
    'lead_noise': "Lead weight nhánh V tránh lead nhiễu 0 dB, nhánh A thì không",
    'ec57_errors': "Lỗi EC57 trên mitdb chủ yếu do model, ngoại lệ VT FN trên 223 do post-process",
}


def finding(engine, fid, n=60, seed=0, xai_report=None):
    """Re-measure one finding on n eval windows per class (CPU-sized) -> claim, numbers,
    verdict ('xác nhận' / 'một phần' / 'bác bỏ') and a reason."""
    if fid not in FINDINGS:
        raise KeyError(fid)
    t0 = time.time()
    out = dict(id=fid, claim=FINDINGS[fid])
    if fid == 'ec57_errors':
        src = None
        if xai_report and os.path.exists(xai_report):
            with open(xai_report) as f:
                src = json.load(f).get('error_sources')
        elif engine.ec57_out and os.path.isdir(os.path.join(engine.ec57_out, '_ann', 'mitdb')):
            src = xai.error_sources(engine.ec57_out)
        if not src:
            return dict(out, verdict='không đủ dữ liệu', reason='không có xai_report.json / --ec57-out')
        rows = []
        for c, v in src.items():
            fp, fn = v['fp_model'] + v['fp_pp'], v['fn_model'] + v['fn_pp']
            rows.append(dict(cls=c, fp_model=v['fp_model'], fp_pp=v['fp_pp'], fn_model=v['fn_model'],
                             fn_pp=v['fn_pp'], model_share=round((v['fp_model'] + v['fn_model']) /
                                                                 max(fp + fn, 1), 3),
                             top=sorted(v['per_record'].items(), key=lambda kv: -sum(kv[1]))[:3]))
        share = sum(r['fp_model'] + r['fn_model'] for r in rows) / max(
            sum(r['fp_model'] + r['fn_model'] + r['fp_pp'] + r['fn_pp'] for r in rows), 1)
        vt = src.get('VT', {}).get('per_record', {}).get('223')
        ok = share > 0.6 and vt is not None and vt[1] > vt[3]
        return dict(out, rows=rows, model_share=round(share, 3),
                    verdict='xác nhận' if ok else 'một phần',
                    reason=f"{100 * share:.0f} % giây lỗi là của model; mitdb 223 VT: FN post-process "
                           f"{vt[1] if vt else '-'} vs FN model {vt[3] if vt else '-'}",
                    elapsed=round(time.time() - t0, 2))

    with engine.lock:
        idx = engine.eval_sample('ALL', n * K, seed)
        raw, labels = engine.eval_batch(idx)
        x = np.stack([normalize(w) for w in raw])
        _, _, cat_all, _ = engine._eval_data()
        cat = cat_all[idx]
        model = engine.model
        if fid == 'edge':
            rows = xai.position_profile(model, x, labels)
            mid = np.mean([r['macro_f1'] for r in rows[3:7]])
            edge = np.mean([rows[0]['macro_f1'], rows[-1]['macro_f1']])
            d = mid - edge
            return dict(out, rows=rows, centre_f1=mid, edge_f1=edge, gap=d,
                        verdict='xác nhận' if d >= 0.03 else ('một phần' if d >= 0.01 else 'bác bỏ'),
                        reason=f"macro F1 giữa (giây 3-6) {100 * mid:.1f} vs mép (0, 9) {100 * edge:.1f}: "
                               f"chênh {100 * d:+.1f} điểm", elapsed=round(time.time() - t0, 2))
        if fid == 'swap':
            r = xai.lead_order_invariance(model, x)
            ok = r['swap_max_diff'] < 1e-3 and r['flip_max_diff'] > 0.05
            return dict(out, **r, verdict='xác nhận' if ok else 'bác bỏ',
                        reason=f"max |Δp| swap {r['swap_max_diff']:.2e}, đảo dấu {r['flip_max_diff']:.3f}",
                        elapsed=round(time.time() - t0, 2))
        if fid == 'v_ssm':
            base = xai.scores(xai.per_second(engine.forward_batch(x, {})), labels)
            res = dict(baseline=base)
            for name, iv in (('ssm_identity', dict(ssm_identity=engine.ssm)),
                             ('atrial_off', dict(atrial_off=True)),
                             ('rr_mean', dict(rr_mean=True)),
                             ('beat_cond_off', dict(beat_cond_off=True))):
                res[name] = xai.scores(xai.per_second(engine.forward_batch(x, iv)), labels)
            drop = {k: base['macro_f1'] - v['macro_f1'] for k, v in res.items() if k != 'baseline'}
            ok = drop['ssm_identity'] > 0.3 and drop['ssm_identity'] > drop['atrial_off'] > \
                max(drop['rr_mean'], drop['beat_cond_off'])
            se = {k: {c: res[k][c]['se'] for c in CLASS_NAMES} for k in res}
            return dict(out, macro_f1={k: v['macro_f1'] for k, v in res.items()}, se=se,
                        drop=drop, verdict='xác nhận' if ok else 'một phần',
                        reason='; '.join(f"{k} {-100 * v:+.1f}" for k, v in drop.items()) +
                               ' điểm macro F1', elapsed=round(time.time() - t0, 2))
        if fid == 'donor':
            sinus = np.flatnonzero(cat == rc.SINUS)
            rng = np.random.default_rng(seed)
            sm = engine._submodel(('qrst_cancel', 'rr_ac'))
            res_all, rr_all = [], []
            for i in range(0, len(x), 64):
                o = sm(x[i:i + 64], training=False)
                res_all.append(np.asarray(o[0]))
                rr_all.append(np.asarray(o[2]))
            res_all, rr_all = np.concatenate(res_all), np.concatenate(rr_all)
            rows = []
            for c in range(K):
                if c == rc.SINUS:
                    continue
                mine = np.flatnonzero(cat == c)
                if len(mine) < 5:
                    continue
                donors = rng.choice(sinus, len(mine))
                b = xai.per_second(engine.forward_batch(x[mine], {}))
                pa = xai.per_second(xai._run_with_donor(model, x[mine], res_all[donors], 'qrst_cancel', 0))
                pr = xai.per_second(xai._run_with_donor(model, x[mine], rr_all[donors], 'rr_ac', None))
                rows.append(dict(cls=CLASS_NAMES[c], n=int(len(mine)), p_before=float(b[..., c].mean()),
                                 p_atrial_from_sinus=float(pa[..., c].mean()),
                                 p_rr_from_sinus=float(pr[..., c].mean())))
            moved = max(max(abs(r['p_atrial_from_sinus'] - r['p_before']),
                            abs(r['p_rr_from_sinus'] - r['p_before'])) for r in rows)
            return dict(out, rows=rows, max_moved=moved,
                        verdict='xác nhận' if moved <= 0.05 else ('một phần' if moved <= 0.1 else 'bác bỏ'),
                        reason=f"thay residual nhĩ / RR bằng của window SINUS đổi p(lớp) tối đa {moved:.3f}",
                        elapsed=round(time.time() - t0, 2))
        if fid == 'f_wave':
            xs = x[cat == rc.SINUS]
            rs = raw[cat == rc.SINUS]
            rng = np.random.default_rng(seed)
            af = CLASS_NAMES.index('AFIB')
            rows = []
            for amp in (0.0, 0.05, 0.1, 0.2):
                xi = xs if not amp else np.stack([normalize(apply_edits(
                    w, dict(f_wave=dict(on=True, amp=amp)), rng)) for w in rs])
                p = xai.per_second(engine.forward_batch(xi, {}))
                pb = xai.per_second(engine.forward_batch(xi, dict(atrial_off=True)))
                rows.append(dict(amp=amp, afib_seconds=float((p.argmax(-1) == af).mean()),
                                 afib_seconds_atrial_off=float((pb.argmax(-1) == af).mean()),
                                 p_afib=float(p[..., af].mean()),
                                 p_afib_atrial_off=float(pb[..., af].mean())))
            r10 = rows[2]
            ok = r10['afib_seconds'] >= 0.15 and r10['afib_seconds_atrial_off'] >= 0.5 * r10['afib_seconds']
            return dict(out, rows=rows, verdict='xác nhận' if ok else
                        ('một phần' if r10['afib_seconds'] >= 0.05 else 'bác bỏ'),
                        reason=f"sóng f 10 % QRS: AFIB {100 * r10['afib_seconds']:.1f} % giây SINUS, "
                               f"nhánh A mù {100 * r10['afib_seconds_atrial_off']:.1f} %",
                        elapsed=round(time.time() - t0, 2))
        if fid == 'lead_noise':
            r = xai.lead_weights(model, x)['noisy_lead0']
            dv = r['v_weight_lead0_before'] - r['v_weight_lead0_after']
            da = r['a_weight_lead0_before'] - r['a_weight_lead0_after']
            ok = dv > 0.2 and da < 0.1
            return dict(out, **r, verdict='xác nhận' if ok else ('một phần' if dv > 0.2 else 'bác bỏ'),
                        reason=f"lead 1 nhiễu 0 dB: trọng số V {r['v_weight_lead0_before']:.2f} -> "
                               f"{r['v_weight_lead0_after']:.2f}, A {r['a_weight_lead0_before']:.2f} -> "
                               f"{r['a_weight_lead0_after']:.2f}", elapsed=round(time.time() - t0, 2))
    raise KeyError(fid)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_handler(engine, xai_report=None):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):                      # quieter: one line per API call
            if '/api/' in (args[0] if args else ''):
                print(f"[xai-web] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body, allow_nan=False,
                                                                   default=_json_default).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _guard(self, fn):
            try:
                self._send(200, _clean(fn()))
            except KeyError as e:
                self._send(404, dict(error=str(e)))
            except Exception as e:                              # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send(500, dict(error=f"{type(e).__name__}: {e}"))

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ('/', '/index.html'):
                with open(HTML, 'rb') as f:
                    return self._send(200, f.read(), 'text/html; charset=utf-8')
            if u.path == '/api/meta':
                return self._guard(engine.meta)
            if u.path == '/api/records':
                return self._guard(lambda: engine.records(q.get('db', 'mitdb')))
            if u.path == '/api/overview':
                return self._guard(lambda: engine.record_overview(q['db'], q['record']))
            if u.path == '/api/eval_list':
                def ev():
                    idx = engine.eval_sample(q.get('category', 'ALL'), int(q.get('n', 40)),
                                             int(q.get('seed', 0)))
                    return dict(indices=[int(i) for i in idx])
                return self._guard(ev)
            if u.path == '/api/layer':
                return self._guard(lambda: engine.layer_detail(q['id'], q['name'], q.get('run', 'mod')))
            if u.path == '/api/findings':
                return self._guard(lambda: FINDINGS)
            self._send(404, dict(error='not found'))

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get('Content-Length') or 0)
            req = json.loads(self.rfile.read(n) or b'{}')
            if u.path == '/api/analyze':
                return self._guard(lambda: engine.analyze(req))
            if u.path == '/api/batch':
                return self._guard(lambda: engine.batch(req))
            if u.path == '/api/scan':
                return self._guard(lambda: engine.scan(req))
            if u.path == '/api/finding':
                return self._guard(lambda: finding(engine, req['id'], int(req.get('n', 60)),
                                                   int(req.get('seed', 0)), xai_report))
            self._send(404, dict(error='not found'))

    return Handler


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o).__name__)


def _clean(o):
    """NaN / inf -> None so the JSON is strict."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def serve(checkpoint, host='0.0.0.0', port=8777, ec57_out=None, xai_report=None, npy_dir=None):
    from http.server import ThreadingHTTPServer
    print(f"[xai-web] loading {checkpoint}", flush=True)
    engine = Engine(checkpoint, ec57_out=ec57_out, npy_dir=npy_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, xai_report))
    print(f"[xai-web] http://{host}:{port}/  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
