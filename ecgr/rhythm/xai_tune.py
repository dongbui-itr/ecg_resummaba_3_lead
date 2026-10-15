"""Automatic per-layer tuning for the XAI web service: which layer settings make the dual U-Net
focus on a class (POST /api/tune, see xai_web.py).

The search space is the layer interventions xai_web already applies by hand: A branch blind,
lead temperature, lead masks (V / A), SSM kernel cut, SSM -> identity (one toggle per DiagSSM1D
layer), beat condition off, R-R descriptor at its mean. Search is greedy coordinate ascent on

    score = mean over selection windows and target seconds of log p(class | second)

Round 0 also gives a one-at-a-time sensitivity table (which layer setting moves the score most
from the starting point). The result is a suggestion only: it is measured on a second set of
windows the search never saw (check set), with the side effects on every other class printed,
and must still go through the validation tune + EC57 run before it is used anywhere (see
memory rule: XAI never selects on EC57).
"""
import numpy as np

from . import config as rc
from . import xai

CLASS_NAMES = rc.CLASS_NAMES
EPS = 1e-6
DEFAULTS = dict(atrial_off=False, beat_cond_off=False, rr_mean=False, lead_temperature=1.0,
                ssm_kernel_frac=1.0, v_lead_off=[], a_lead_off=[], ssm_identity=[])
LEAD_MASKS = [[], [0], [1], [2]]


def moves(engine, iv):
    """All single-setting changes from `iv` -> [(knob, label, new_iv)]."""
    out = []

    def add(knob, label, **kv):
        out.append((knob, label, {**iv, **kv}))

    for key, cands in (('atrial_off', [False, True]), ('beat_cond_off', [False, True]),
                       ('rr_mean', [False, True]), ('lead_temperature', [0.5, 1.0, 2.0]),
                       ('ssm_kernel_frac', [1.0, 0.5, 0.25]),
                       ('v_lead_off', LEAD_MASKS), ('a_lead_off', LEAD_MASKS)):
        for v in cands:
            if (iv.get(key, DEFAULTS[key]) or DEFAULTS[key]) != (v or DEFAULTS[key]):
                add(key, f'{key}={v}', **{key: v})
    cur = list(iv.get('ssm_identity') or [])
    for name in engine.ssm:                                  # one toggle per SSM layer
        new = [s for s in cur if s != name] if name in cur else cur + [name]
        add('ssm_identity', f'{name} {"identity" if name not in cur else "normal"}',
            ssm_identity=new)
    return out


def _stats(p, cls):
    """p: (n, 10, K) per-second probabilities of a set of windows."""
    share = np.bincount(p.argmax(-1).ravel(), minlength=len(CLASS_NAMES)) / max(p[..., 0].size, 1)
    return dict(p_target=float(p[..., cls].mean()),
                target_seconds=float((p.argmax(-1) == cls).mean()),
                mean_p={c: float(v) for c, v in zip(CLASS_NAMES, p.mean((0, 1)))},
                argmax_share={c: float(v) for c, v in zip(CLASS_NAMES, share)})


def _windows(engine, req, cls_name, n):
    """Selection and check windows (normalised): either raw arrays in req['windows'] (split in
    half) or disjoint samples of the eval split of the target class."""
    from .xai_web import normalize
    if req.get('windows'):
        raw = np.asarray(req['windows'], np.float32)
        if len(raw) < 2:
            raise ValueError('need at least 2 windows (one to select, one to check)')
        half = len(raw) // 2
        sel, chk = raw[:half], raw[half:]
    else:
        idx = engine.eval_sample(cls_name, 2 * n, int(req.get('seed', 0)))
        if len(idx) < 2:
            raise ValueError(f'eval split has too few {cls_name} windows')
        half = len(idx) // 2
        sel = engine.eval_batch(idx[:half])[0]
        chk = engine.eval_batch(idx[half:])[0]
    return np.stack([normalize(w) for w in sel]), np.stack([normalize(w) for w in chk])


def tune(engine, req):
    cls_name = req.get('cls')
    if cls_name not in CLASS_NAMES:
        raise ValueError(f'cls must be one of {CLASS_NAMES}')
    cls = CLASS_NAMES.index(cls_name)
    seconds = [int(s) for s in req.get('seconds') or range(rc.OUTPUT_SECONDS)]
    max_rounds = int(min(max(int(req.get('max_rounds', 3)), 1), 6))
    min_gain = float(req.get('min_gain', 0.01))
    side_weight = float(req.get('side_weight', 0.0))
    max_evals = int(req.get('max_evals', 400))
    start = dict(req.get('interventions') or {})

    with engine.lock:
        xs, xc = _windows(engine, req, cls_name, int(req.get('n', 8)))
        evals = [0]

        def run(iv, x):
            evals[0] += 1
            p = xai.per_second(engine.forward_batch(x, iv, batch=len(x)))
            return p

        # side_weight > 0: also penalise moving the mean probability of the OTHER classes away
        # from the unmodified model (focus on the class without redistributing the rest)
        orig_mean = run({}, xs).mean((0, 1))
        others = np.arange(len(CLASS_NAMES)) != cls

        def score(iv):
            p = run(iv, xs)
            pc = np.clip(p[:, seconds, cls], EPS, 1.0)
            s = float(np.log(pc).mean())
            if side_weight:
                s -= side_weight * float(np.abs(p.mean((0, 1)) - orig_mean)[others].sum())
            return s

        orig_s, start_s = score({}), score(start)
        cur, cur_s = start, start_s
        sensitivity, history = [], []
        for rnd in range(max_rounds):
            best = None
            for knob, label, cand in moves(engine, cur):
                if evals[0] >= max_evals:
                    break
                s = score(cand)
                if rnd == 0:
                    sensitivity.append(dict(knob=knob, setting=label, dscore=s - cur_s))
                if best is None or s > best[0]:
                    best = (s, knob, label, cand)
            if best is None or best[0] - cur_s < min_gain or evals[0] >= max_evals:
                break
            cur_s, knob, label, cur = best
            history.append(dict(round=rnd + 1, knob=knob, setting=label, score=cur_s))

        final = cur
        report = {}
        for name, iv in (('orig', {}), ('start', start), ('tuned', final)):
            report[name] = dict(select=_stats(run(iv, xs), cls), check=_stats(run(iv, xc), cls))

    sensitivity.sort(key=lambda d: -d['dscore'])
    return dict(cls=cls_name, seconds=seconds, side_weight=side_weight,
                n_select=len(xs), n_check=len(xc),
                score=dict(orig=orig_s, start=start_s, tuned=cur_s),
                interventions=final, history=history,
                sensitivity=sensitivity[:30], report=report, evals=evals[0],
                budget_hit=evals[0] >= max_evals)
