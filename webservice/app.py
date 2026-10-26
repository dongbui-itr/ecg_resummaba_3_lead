#!/usr/bin/env python3
"""ECG rhythm viewer web service for the evaluate.py results.

    python3 webservice/app.py          # edit the CONFIG block first, then open http://HOST:PORT

No command-line arguments (same rule as evaluate.py): the CONFIG block below says which result
folders, signals and checkpoint the viewer serves.

Pick an evaluation run (EVAL_ROOT -> <tag>), a database inside it and a record. The page draws
three stacked layers: ECG paper grid, the record's original channels, and the rhythm regions as
arrows - the raw model decision (argmax of the stored probabilities, BEFORE post-processing) and
the post-processed episodes (.rhi, what EC57 scored). SINUS is not drawn.

Reads <eval>/_ann/<db>/<rec>.npz (raw probabilities) and <rec>.rhi (post-processed), signal from
<signal-root>/<db>/<rec>.{hea,dat}. The Run tab shows the run's evaluate_config.json (checkpoint,
decode, overlap, TTA) and its EC57 tables (epicmp + bxb report lines, read as summarize_ec57.py
does). The XAI tab computes integrated gradients of the checkpoint on CPU.

The STM32 kit tab streams a record from the signal folder to the STM32F746G-Discovery
firmware (firmware/stm32f746g_disco) over its ST-LINK serial port - first 10 s, then 9 s per
chunk, with the reference labels - and draws what the kit returns the way the kit's LCD
does. Port 'virtual' runs the same firmware code built for this PC (webservice/stm32_link.py).
"""
import asyncio
import json
import os
import sys

import numpy as np
import wfdb
from aiohttp import web

# ===========================================================================
# CONFIG - edit this block
# ===========================================================================

HOST = '127.0.0.1'               # '0.0.0.0' to reach it from another machine on the network
PORT = 8765
# Where evaluate.py writes: <WORK_DIR>/<RUN_TAG>/ec57/<TAG>/ (relative = the project root).
EVAL_ROOT = 'eval_results/dualunet_rhythm_1m_e26/ec57'
SIGNAL_ROOT = os.environ.get('ECGR_PHYSIONET_DIR', '/media/Project/ECG/PhysionetData')
CHECKPOINT = 'checkpoints/dualunet_rhythm_1m_e26/best_model.keras'     # for the XAI tab
KIT_BAUD = 921600                # STM32 kit tab: the firmware's APP_BAUD
KIT_ANN = 'atr'                  # annotation the kit tab reads reference labels from

# ===========================================================================
# End of CONFIG
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # the project root
for _p in (ROOT, HERE):                            # ecgr, stm32_link
    if _p not in sys.path:
        sys.path.insert(0, _p)
CLASS_NAMES = ['SINUS', 'AFIB', 'SVT', 'VT', 'AVB']
AUX_TO_CLASS = {'(N': 'SINUS', '(AFIB': 'AFIB', '(AFL': 'AFIB', '(SVTA': 'SVT', '(VT': 'VT',
                '(AVB': 'AVB'}
MAX_WINDOW_S = 120.0


def safe_name(s):
    s = str(s or '')
    if not s or s != os.path.basename(s) or s in ('.', '..'):
        raise web.HTTPBadRequest(text=f'bad name: {s!r}')
    return s


def project_path(path):
    """CONFIG paths are read relative to the project root."""
    path = os.path.expanduser(path or '')
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def resolve_dir(path):
    p = os.path.realpath(project_path(path))
    if not os.path.isdir(p):
        raise web.HTTPNotFound(text=f'not a directory: {path}')
    return p


def ann_dir(run, db):
    d = os.path.join(resolve_dir(run), '_ann', safe_name(db))
    if not os.path.isdir(d):
        raise web.HTTPNotFound(text=f'no _ann/{db} in {run}')
    return d


def find_runs(root, max_depth=3):
    """Folders under `root` (or root itself) that hold an _ann/ subfolder."""
    root = resolve_dir(root)
    runs = []
    for dirpath, dirs, _files in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if '_ann' in dirs:
            runs.append(dirpath)
            dirs[:] = []
        elif depth >= max_depth:
            dirs[:] = []
        else:
            dirs.sort()
    return sorted(runs)


def runs_of(arr):
    """(start, stop, value) of the constant runs of a 1-D array."""
    if len(arr) == 0:
        return []
    edge = np.flatnonzero(np.diff(arr)) + 1
    starts = np.r_[0, edge]
    stops = np.r_[edge, len(arr)]
    return [(int(a), int(b), int(arr[a])) for a, b in zip(starts, stops)]


def pre_episodes(npz_path):
    """Raw model decision: argmax of the stored probabilities, no smoothing / gap bridging /
    min duration / beat stage. Returns (episodes, step_hz, seconds)."""
    with np.load(npz_path) as z:
        rhythm = z['rhythm'].astype(np.float32)
        step_hz = int(z['step_hz']) if 'step_hz' in z.files else 1
    if rhythm.shape[1] == 6:                     # legacy AVB2 / AVB3 columns
        rhythm = np.concatenate([rhythm[:, :4], rhythm[:, 4:].sum(1, keepdims=True)], axis=1)
    cls = rhythm.argmax(1)
    eps = [dict(rhythm=CLASS_NAMES[c], start=a / step_hz, stop=b / step_hz,
                prob=float(rhythm[a:b, c].mean()))
       for a, b, c in runs_of(cls) if c != 0]
    return eps, len(rhythm) / step_hz


def post_episodes(ann_base, length_s):
    """Post-processed episodes from the .rhi file (one '+' per episode start; an episode ends
    where the next starts)."""
    path = ann_base + '.rhi'
    if not os.path.exists(path):
        return None
    ann = wfdb.rdann(ann_base, 'rhi')
    fs = ann.fs or 360
    rows = sorted((s / fs, AUX_TO_CLASS.get(n.strip('\x00').strip(), n.strip('\x00')))
                  for s, n in zip(ann.sample, ann.aux_note))
    eps = []
    for i, (t, name) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else length_s
        if name != 'SINUS' and end > t:
            eps.append(dict(rhythm=name, start=t, stop=end))
    return eps


def rhythm_diffs(pre, post, length, grid=0.04, merge=0.3):
    """Segments where the raw and the post-processed rhythm differ (SINUS = no episode)."""
    n = int(length / grid) + 1
    def track(eps):
        t = np.zeros(n, np.int8)
        for e in eps:
            t[int(e['start'] / grid):int(np.ceil(e['stop'] / grid))] = CLASS_NAMES.index(e['rhythm'])
        return t
    a, b = track(pre), track(post)
    idx = np.flatnonzero(a != b)
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) * grid > merge) + 1
    out = []
    for seg in np.split(idx, cut):
        lo, hi = seg[0], seg[-1] + 1
        pa, pb = np.bincount(a[lo:hi], minlength=5), np.bincount(b[lo:hi], minlength=5)
        # name the dominant label of each side inside the differing samples only
        m = a[lo:hi] != b[lo:hi]
        pa, pb = np.bincount(a[lo:hi][m], minlength=5), np.bincount(b[lo:hi][m], minlength=5)
        out.append(dict(kind='rhythm', start=lo * grid, stop=min(hi * grid, length),
                        pre=CLASS_NAMES[int(pa.argmax())], post=CLASS_NAMES[int(pb.argmax())]))
    return out


BEAT_SYMBOL = {1: 'N', 2: 'S', 3: 'V'}


def beat_diffs(ann_base, npz_path):
    """Beats whose raw class (npz) differs from the post-processed symbol (.bti)."""
    if not os.path.exists(ann_base + '.bti'):
        return []
    with np.load(npz_path) as z:
        if 'beat_t' not in z.files:
            return []
        t, cls = z['beat_t'].astype(float), z['beat_cls'].astype(int)
    ann = wfdb.rdann(ann_base, 'bti')
    bt = ann.sample / (ann.fs or 360)
    if len(bt) != len(t):
        return []
    out = []
    for ti, c, sym in zip(t, cls, ann.symbol):
        raw = BEAT_SYMBOL.get(int(c), '?')
        if raw != sym:
            out.append(dict(kind='beat', start=float(ti), stop=float(ti), pre=raw, post=sym))
    return out


async def api_runs(request):
    root = request.query.get('root') or request.app['eval_root']
    runs = find_runs(root)
    return web.json_response(dict(root=resolve_dir(root), runs=runs))


async def api_dbs(request):
    a = os.path.join(resolve_dir(request.query.get('run')), '_ann')
    dbs = sorted(d for d in os.listdir(a) if os.path.isdir(os.path.join(a, d)))
    return web.json_response(dict(dbs=dbs))


async def api_records(request):
    d = ann_dir(request.query.get('run'), request.query.get('db'))
    recs = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npz'))
    return web.json_response(dict(records=recs))


def signal_path(request, db, rec):
    root = resolve_dir(request.query.get('signal_root') or request.app['signal_root'])
    base = os.path.join(root, safe_name(db), safe_name(rec))
    if not os.path.exists(base + '.hea'):
        raise web.HTTPNotFound(text=f'signal not found: {base}.hea (set the signal folder)')
    return base


async def api_episodes(request):
    q = request.query
    d = ann_dir(q.get('run'), q.get('db'))
    rec = safe_name(q.get('rec'))
    npz = os.path.join(d, rec + '.npz')
    if not os.path.exists(npz):
        raise web.HTTPNotFound(text=f'no {rec}.npz')
    pre, length = pre_episodes(npz)
    try:
        hdr = wfdb.rdheader(signal_path(request, q.get('db'), rec))
        fs, names, length = hdr.fs, hdr.sig_name, hdr.sig_len / hdr.fs
    except web.HTTPNotFound:
        fs, names = None, []
    base = os.path.join(d, rec)
    post = post_episodes(base, length)
    diffs = (rhythm_diffs(pre, post, length) if post is not None else []) + beat_diffs(base, npz)
    diffs.sort(key=lambda e: e['start'])
    return web.json_response(dict(length=length, fs=fs, channels=names, pre=pre, post=post,
                                  diffs=diffs))


async def api_signal(request):
    q = request.query
    base = signal_path(request, q.get('db'), safe_name(q.get('rec')))
    hdr = wfdb.rdheader(base)
    fs = hdr.fs
    t0 = max(0.0, float(q.get('t0', 0)))
    t1 = min(hdr.sig_len / fs, t0 + min(float(q.get('win', 10)), MAX_WINDOW_S))
    a, b = int(t0 * fs), int(t1 * fs)
    rec = wfdb.rdrecord(base, sampfrom=a, sampto=b)
    sig = np.nan_to_num(rec.p_signal)
    return web.json_response(dict(fs=fs, t0=a / fs, names=rec.sig_name, units=rec.units,
                                  data=[np.round(sig[:, i], 4).tolist()
                                        for i in range(sig.shape[1])]))


_MODEL = {}
_XAI_LOCK = asyncio.Lock()
IG_STEPS = 24
XAI_HZ = 25


def get_model(path):
    """Load the checkpoint once, on CPU (the GPU is usually busy with training / eval)."""
    if path not in _MODEL:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import tensorflow as tf
        from ecgr.rhythm import dualunet, model as _model  # noqa: F401 (registers custom layers)
        _MODEL[path] = tf.keras.models.load_model(path, compile=False)
    return _MODEL[path]


def xai_compute(checkpoint, base, seg0, seg1):
    """Integrated gradients (zero baseline = the per-lead mean after z-scoring) of the mean
    p(class) inside [seg0, seg1] s, for every rhythm class, over the 10 s window around it."""
    import tensorflow as tf
    from ecgr.rhythm import config as rc
    from ecgr.rhythm.build import read_leads
    from ecgr.signal_ops import normalize_window
    model = get_model(checkpoint)
    leads, _ = read_leads(base)
    fs, seg_n = rc.SAMPLING_RATE, rc.SEGMENT_SAMPLES
    length = len(leads)
    mid = (seg0 + seg1) / 2
    s = int(round(mid * fs)) - seg_n // 2
    s = max(0, min(s, max(length - seg_n, 0)))
    win = leads[s:s + seg_n]
    if len(win) < seg_n:
        win = np.concatenate([win, np.zeros((seg_n - len(win), win.shape[1]), win.dtype)])
    x = normalize_window(win).astype(np.float32)[None]
    ws = s / fs
    out = model(tf.constant(x), training=False)['rhythm'].numpy()[0]        # (steps, 5)
    steps = out.shape[0]
    sps = steps / rc.SEGMENT_SECONDS
    a = int(np.clip((seg0 - ws) * sps, 0, steps - 1))
    b = int(np.clip(np.ceil((seg1 - ws) * sps), a + 1, steps))
    alphas = tf.reshape(tf.linspace(0.0, 1.0, IG_STEPS + 1)[1:], (-1, 1, 1))
    xs = tf.constant(x) * alphas                                          # (K, 2500, 3)
    attrib = []
    for c in range(rc.NUM_CLASSES):
        with tf.GradientTape() as tape:
            tape.watch(xs)
            y = model(xs, training=False)['rhythm'][:, a:b, c]
            target = tf.reduce_mean(y, axis=1)
        g = tape.gradient(target, xs).numpy().mean(axis=0)                  # (2500, 3)
        attrib.append(g * x[0])
    attrib = np.stack(attrib)                                               # (5, 2500, 3)
    per = int(round(steps / (rc.SEGMENT_SECONDS * XAI_HZ)))
    prob = out[:per * (steps // per)].reshape(-1, per, out.shape[1]).mean(1)   # (250, 5) @25Hz
    k = 5                                                                   # 250 -> 50 Hz
    att = attrib[:, :seg_n // k * k].reshape(rc.NUM_CLASSES, -1, k, 3).sum(2)   # signed, (5,500,3)
    sig = x[0][:seg_n // k * k].reshape(-1, k, 3).mean(1)                    # (500, 3)
    return dict(win_start=ws, win_len=rc.SEGMENT_SECONDS, seg=[seg0, seg1], classes=CLASS_NAMES,
                prob=np.round(prob, 4).tolist(), prob_hz=XAI_HZ,
                seg_prob=np.round(out[a:b].mean(0), 4).tolist(),
                att=np.round(att, 5).tolist(), att_hz=fs / k, signal=np.round(sig, 3).tolist(),
                signal_hz=fs / k, leads=['CH1', 'CH2', 'CH3'])


async def api_xai(request):
    q = request.query
    base = signal_path(request, q.get('db'), safe_name(q.get('rec')))
    ckpt = q.get('checkpoint') or request.app['checkpoint']
    if not os.path.exists(ckpt):
        raise web.HTTPNotFound(text=f'checkpoint not found: {ckpt}')
    seg0, seg1 = float(q.get('t0', 0)), float(q.get('t1', 0))
    seg1 = max(seg1, seg0 + 0.5)
    async with _XAI_LOCK:
        res = await asyncio.get_running_loop().run_in_executor(
            None, xai_compute, ckpt, base, seg0, seg1)
    return web.json_response(res)


async def api_run_info(request):
    """The run's evaluate_config.json and its EC57 tables (rhythm episodes / duration from the
    epicmp report lines, beats from the bxb ones)."""
    import summarize_ec57
    run = resolve_dir(request.query.get('run'))
    cfg = None
    path = os.path.join(run, 'evaluate_config.json')
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
    rhythm, _ = summarize_ec57.rhythm_rows(run)
    beats, _ = summarize_ec57.beat_rows(run)
    return web.json_response(dict(run=run, config=cfg, rhythm=rhythm, beats=beats),
                             dumps=lambda o: json.dumps(o, default=str))


async def api_config(request):
    return web.json_response(dict(signal_root=request.app['signal_root'],
                                  eval_root=request.app['eval_root'],
                                  checkpoint=request.app['checkpoint']))


# ---------- STM32F746G-Discovery kit ----------------------------------------------------------

def kit_link():
    import stm32_link
    return stm32_link


def kit_dbs(root):
    """Folders of the signal root that hold WFDB headers."""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and any(f.endswith('.hea') for f in os.listdir(p)):
            out.append(d)
    return out


async def api_kit_config(request):
    sl = kit_link()
    root = resolve_dir(request.query.get('signal_root') or request.app['signal_root'])
    return web.json_response(dict(ports=sl.list_ports(), virtual=sl.VirtualKit.available(),
                                  baud=KIT_BAUD, dbs=kit_dbs(root)))


async def api_kit_records(request):
    root = resolve_dir(request.query.get('signal_root') or request.app['signal_root'])
    d = os.path.join(root, safe_name(request.query.get('db')))
    recs = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.hea'))
    return web.json_response(dict(records=recs))


def kit_stop(app):
    sess = app.get('kit')
    if sess is not None:
        sess.stop()
    app['kit'] = None


async def api_kit_start(request):
    sl = kit_link()
    q = await request.json()
    root = resolve_dir(q.get('signal_root') or request.app['signal_root'])
    db, rec = safe_name(q.get('db')), safe_name(q.get('rec'))
    base = os.path.join(root, db, rec)
    if not os.path.exists(base + '.hea'):
        raise web.HTTPNotFound(text=f'no record {db}/{rec}')
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, kit_stop, request.app)
    start_s = max(0, int(float(q.get('start') or 0)))
    src = await loop.run_in_executor(None, lambda: sl.RecordSource(base, start_s, q.get('ann') or KIT_ANN))
    if src.n_chunks() == 0:
        raise web.HTTPBadRequest(text=f'{rec}: less than 10 s after {start_s} s')
    port, virtual = q.get('port') or 'virtual', None
    if os.path.exists(request.app['kit_snap']):
        os.remove(request.app['kit_snap'])               # no stale frame from the last session
    if port == 'virtual':
        virtual = sl.VirtualKit(request.app['kit_snap'])
        try:
            port = await loop.run_in_executor(None, virtual.start)
        except RuntimeError as e:
            raise web.HTTPInternalServerError(text=str(e))
    meta = dict(db=db, rec=rec, start=start_s, port=q.get('port') or 'virtual',
                seconds=src.seconds, has_rhythm=src.has_rhythm, has_beats=src.has_beats)
    request.app['kit'] = sl.KitSession(port, int(q.get('baud') or KIT_BAUD), src,
                                       max_windows=int(q.get('max') or 0) or None,
                                       pace=q.get('pace') or 'fast', virtual=virtual,
                                       meta=meta).start()
    return web.json_response(dict(ok=True, meta=meta, chunks=src.n_chunks()))


async def api_kit_stop(request):
    await asyncio.get_running_loop().run_in_executor(None, kit_stop, request.app)
    return web.json_response(dict(ok=True))


async def api_kit_state(request):
    sess = request.app.get('kit')
    if sess is None:
        return web.json_response(dict(status='idle', n=0, results=[], log=[]))
    return web.json_response(sess.snapshot(int(request.query.get('since', 0))))


async def api_kit_window(request):
    """The 10 s the kit analysed for window k (what it drew), with the labels it received."""
    sess = request.app.get('kit')
    if sess is None:
        raise web.HTTPNotFound(text='no kit session')
    k = int(request.query.get('index', 0))
    res = sess.result(k)
    if res is None:
        raise web.HTTPNotFound(text=f'no result for window {k} yet')
    w = sess.src.window(k)
    return web.json_response(dict(index=k, start=w['start'], uv=w['uv'].T.tolist(),
                                  labels=[int(v) for v in w['labels']], ref_beats=w['beats'],
                                  result=res, info=sess.info, meta=sess.meta))


async def api_kit_lcd(request):
    """The virtual kit's display (its three layers blended as on the board), as PNG."""
    path = request.app['kit_snap']
    if not os.path.exists(path):
        raise web.HTTPNotFound(text='no virtual kit frame yet')
    import io
    from PIL import Image
    buf = io.BytesIO()
    with Image.open(path) as im:
        im.save(buf, 'PNG')
    return web.Response(body=buf.getvalue(), content_type='image/png',
                        headers={'Cache-Control': 'no-store'})


async def index(request):
    return web.FileResponse(os.path.join(HERE, 'static', 'index.html'))


def make_app(eval_root, signal_root, checkpoint):
    import tempfile
    app = web.Application()
    app['eval_root'], app['signal_root'], app['checkpoint'] = eval_root, signal_root, checkpoint
    app['kit'] = None
    app['kit_snap'] = os.path.join(tempfile.gettempdir(), f'ecgkit_lcd_{os.getpid()}.ppm')
    app.on_shutdown.append(lambda a: asyncio.get_running_loop().run_in_executor(None, kit_stop, a))
    app.add_routes([web.get('/', index), web.get('/api/config', api_config), web.get('/api/runs', api_runs),
                    web.get('/api/dbs', api_dbs), web.get('/api/records', api_records),
                    web.get('/api/episodes', api_episodes), web.get('/api/xai', api_xai), web.get('/api/signal', api_signal),
                    web.get('/api/run_info', api_run_info),
                    web.get('/api/kit/config', api_kit_config), web.get('/api/kit/records', api_kit_records),
                    web.post('/api/kit/start', api_kit_start), web.post('/api/kit/stop', api_kit_stop),
                    web.get('/api/kit/state', api_kit_state), web.get('/api/kit/window', api_kit_window),
                    web.get('/api/kit/lcd.png', api_kit_lcd)])
    return app


def check_classes():
    """The page and the stored npz columns follow ecgr.rhythm.config's class order."""
    from ecgr.rhythm import config as rc
    if list(rc.CLASS_NAMES) != CLASS_NAMES:
        raise SystemExit(f"class order changed: config {rc.CLASS_NAMES} vs viewer {CLASS_NAMES}")


def main():
    """Serve with the CONFIG block (a caller such as run.py may set the constants first)."""
    check_classes()
    checkpoint = project_path(CHECKPOINT)
    print(f"eval root  : {project_path(EVAL_ROOT)}")
    print(f"signals    : {SIGNAL_ROOT}")
    print(f"checkpoint : {checkpoint}{'' if os.path.exists(checkpoint) else '  (missing: XAI off)'}")
    print(f"open       : http://{HOST}:{PORT}/")
    web.run_app(make_app(EVAL_ROOT, SIGNAL_ROOT, checkpoint), host=HOST, port=PORT)


if __name__ == '__main__':
    main()
