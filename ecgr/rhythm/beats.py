"""Beats from the beat decoder, and rhythm post-processing on the beat train.

The production Holter pipeline runs its second-stage rhythm logic per BEAT: every beat carries
one rhythm class, contiguous same-class regions are validated against criteria (duration,
beat count, rate, runs of V or S beats), invalid regions are merged into their neighbours, VT
and SVT are stretched along V / S runs and cut where the beats do not match. This module is
that stage for the models with a 'beat' output, with the defects the analysis of the
production code lists fixed (config.BEAT_PP_* and the comments below).

    pick_beats       (T, 4) none/N/S/V probabilities at step_hz -> {t, cls, conf, probs}
    postprocess      step class track + beats -> corrected track, beat symbols (S in AF -> N)

The step track comes from labels.decode_track (smoothing, prior correction, minimums, the
AF confidence floor); the beat stage runs after it and only changes the stretch the beats
cover. A record with fewer than 3 beats is returned unchanged.

Order of the stage (production order, with the fixes):
    1. rhythm per beat (majority of the track in the beat's cell)
    2. runs of V / S beats that meet the VT / SVT criteria become VT / SVT on their own -
       the production VES_RUN / SVES_RUN events that the EC57 export counts as VT (K);
       off unless options['runs_to_rhythm'][class] (chosen on validation)
    3. a V (S) run with any VT (SVT) beat is VT (SVT) throughout
    4. criteria per region (A: beat counts enforced, J: VT / SVT need a fast rate)
    5. a maximal invalid stretch - the gaps between invalid regions tested on the beats, not
       on the valid mask (C) - of >= long_invalid seconds becomes SINUS (H: absolute seconds)
    6. the remaining invalid regions take their neighbours by an explicit rule (E)
    7. VT / SVT cut at non-matching beats, which take the surrounding rhythm (D)
    8. optional AFib / SVT window arbitration (B, G fixed)
    9. beat symbols: S inside AFIB -> N; optional prematurity gate on S outside SVT
"""
import numpy as np

from . import config as rc
from .labels import NOISE, class_index, runs

N_BEAT, S_BEAT, V_BEAT = 1, 2, 3


def pick_beats(beat_probs, step_hz, threshold=None, refractory_s=None):
    """Local maxima of p(beat) = 1 - p(none) above `threshold`, at least `refractory_s`
    apart (the higher peak wins). cls = argmax of N/S/V around the peak (+-2 steps)."""
    threshold = rc.BEAT_PICK_THRESHOLD if threshold is None else threshold
    refractory_s = rc.BEAT_REFRACTORY_SECONDS if refractory_s is None else refractory_s
    probs = np.asarray(beat_probs, dtype=np.float32)
    empty = dict(t=np.zeros(0), cls=np.zeros(0, int), conf=np.zeros(0, np.float32),
                 probs=np.zeros((0, 3), np.float32))
    if len(probs) < 3:
        return empty
    p = 1.0 - probs[:, 0]
    ps = np.convolve(p, np.ones(3, np.float32) / 3.0, mode='same')
    inner = ps[1:-1]
    cand = np.flatnonzero((inner >= ps[:-2]) & (inner > ps[2:]) & (inner > threshold)) + 1
    if len(cand) == 0:
        return empty
    r = max(1, int(round(refractory_s * step_hz)))
    taken = np.zeros(len(ps), dtype=bool)
    keep = []
    for i in cand[np.argsort(-ps[cand], kind='stable')]:
        if not taken[i]:
            keep.append(i)
            taken[max(0, i - r):i + r + 1] = True
    keep = np.array(sorted(keep), dtype=np.int64)
    win = np.stack([probs[max(0, i - 2):i + 3, 1:4].mean(axis=0) for i in keep])
    return dict(t=keep / float(step_hz), cls=1 + np.argmax(win, axis=1),
                conf=ps[keep].astype(np.float32), probs=win.astype(np.float32))


def beat_cells(t, n_steps, step_hz):
    """Step range [a, b) each beat owns: midpoints to its neighbours (the production export
    puts rhythm transitions at midpoints between beats); the first and last cell run to the
    record's ends so the beat rhythm covers the whole track."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    mids = np.round((t[1:] + t[:-1]) / 2.0 * step_hz).astype(int)
    mids = np.clip(mids, 0, n_steps)
    a = np.concatenate([[0], mids])
    b = np.concatenate([mids, [n_steps]])
    return a, b


def rhythm_per_beat(track, a, b):
    """Majority class of the track inside each beat's cell (SINUS for an empty cell)."""
    out = np.full(len(a), rc.SINUS, dtype=int)
    for i, (lo, hi) in enumerate(zip(a, b)):
        if hi > lo:
            out[i] = int(np.bincount(track[lo:hi], minlength=NOISE + 1).argmax())
    return out


# --- criteria ---------------------------------------------------------------------------------

def _duration(t, lo, hi):
    return float(t[hi - 1] - t[lo]) if hi - lo >= 2 else 0.0


def _hr(t, lo, hi):
    if hi - lo < 2:
        return float('nan')
    rr = np.diff(t[lo:hi])
    return 60.0 / float(rr.mean()) if rr.mean() > 0 else float('nan')


def _max_run(sym, lo, hi, value):
    best = cur = 0
    for v in sym[lo:hi]:
        cur = cur + 1 if v == value else 0
        best = max(best, cur)
    return best


def region_valid(name, t, sym, lo, hi, prev_hr=None, criteria=None):
    """Does the region of beats [lo, hi) called `name` meet its criteria? Fixes A (beat
    counts enforced) and J (VT / SVT need a fast rate). SVT accepts N beats: in sustained
    SVT the beats lose their prematurity and are labelled N, so a run of S, a fraction of S
    or an abrupt rate jump against the preceding beats is enough (fix D's companion)."""
    criteria = rc.BEAT_PP_CRITERIA if criteria is None else criteria
    c = criteria.get(name)
    if c is None:
        return True
    n = hi - lo
    dur, hr = _duration(t, lo, hi), _hr(t, lo, hi)
    if 'duration' in c and dur < c['duration']:
        return False
    if 'num_beat' in c and n < c['num_beat']:
        return False
    if 'min_hr' in c and not hr >= c['min_hr']:
        return False
    if 'max_hr' in c and not hr <= c['max_hr']:
        return False
    if name == 'VT':
        return _max_run(sym, lo, hi, V_BEAT) >= c['run']
    if name == 'SVT':
        if _max_run(sym, lo, hi, S_BEAT) >= c['run']:
            return True
        if np.mean(sym[lo:hi] == S_BEAT) >= c['min_frac']:
            return True
        return prev_hr is not None and prev_hr > 0 and hr >= c['onset_ratio'] * prev_hr
    return True


# --- options ---------------------------------------------------------------------------------

def default_options():
    """The tunable switches of the stage, from config. `postprocess(options=...)` overrides
    any of them (the validation grid in tune.py passes them explicitly)."""
    return dict(
        long_invalid=rc.BEAT_PP_LONG_INVALID_SECONDS,
        fast_rr=rc.BEAT_PP_FAST_RR_SECONDS,
        runs_to_rhythm=dict(rc.BEAT_PP_RUNS_TO_RHYTHM),
        afib_svt_merge=rc.BEAT_PP_AFIB_SVT_MERGE,
        svt_ratio=rc.BEAT_PP_SVT_RATIO,
        s_prematurity=rc.BEAT_PP_S_PREMATURITY,
        s_in_afib_to_n=rc.BEAT_PP_S_IN_AFIB_TO_N,
    )


def _options(options):
    out = default_options()
    for k, v in (options or {}).items():
        if k == 'runs_to_rhythm':
            out[k] = {**out[k], **(v or {})}
        else:
            out[k] = v
    return out


# --- the stage -----------------------------------------------------------------------------

def _spec():
    return {class_index(n) for n in ('VT', 'SVT', 'AVB')}


def _rank():
    order = {}
    for i, name in enumerate(reversed(rc.BEAT_PP_PRIORITY)):
        order[NOISE if name == 'NOISE' else class_index(name)] = i
    return order


def _runs_to_rhythm(r, t, sym, criteria, which):
    """Step 2: a run of V (S) beats that on its own meets the VT (SVT) criteria - length,
    rate, beat count - is VT (SVT) even where the step track never said so. This is the
    production VES_RUN / SVES_RUN beat event, which the EC57 export folds into VT (K); the
    reference VT episodes of mitdb have a median length of 1.8 s, which a 10 s rhythm head
    smooths over more often than the beat decoder misses the V beats. Beats in NOISE cells
    are left alone."""
    for name, value in (('VT', V_BEAT), ('SVT', S_BEAT)):
        if not which.get(name):
            continue
        c = class_index(name)
        for lo, hi in runs((sym == value).astype(int)):
            if sym[lo] != value or np.any(r[lo:hi] == NOISE):
                continue
            if region_valid(name, t, sym, lo, hi, None, criteria):
                r[lo:hi] = c
    return r


def _extend_along_runs(r, sym):
    """Step 3: a V (S) run with any VT (SVT) beat is VT (SVT) throughout."""
    for value, name in ((V_BEAT, 'VT'), (S_BEAT, 'SVT')):
        c = class_index(name)
        for lo, hi in runs((sym == value).astype(int)):
            if sym[lo] == value and np.any(r[lo:hi] == c):
                r[lo:hi] = c
    return r


def _validate(r, t, sym, criteria):
    valid = np.ones(len(r), dtype=bool)
    for lo, hi in runs(r):
        c = int(r[lo])
        name = 'NOISE' if c == NOISE else rc.CLASS_NAMES[c]
        prev_hr = _hr(t, max(0, lo - 4), lo) if lo >= 2 else None
        valid[lo:hi] = region_valid(name, t, sym, lo, hi, prev_hr, criteria)
    return valid


def _long_invalid_to_sinus(r, t, valid, long_invalid):
    """Step 5 (fixes C and H): the production step joins invalid VT / SVT / AVB regions
    across gaps and turns the joined stretch into SINUS when it is long enough - but tests
    the gap on the valid mask instead of the rhythm, so it joins across any invalid beats.
    Done properly, "invalid region, invalid gap, invalid region" is simply one maximal
    invalid stretch, whatever the classes: that is what is measured here, in seconds."""
    for lo, hi in runs(valid.astype(int)):
        if not valid[lo] and _duration(t, lo, hi) >= long_invalid:
            r[lo:hi] = rc.SINUS
            valid[lo:hi] = True
    return r, valid


def _merge_invalid(r, t, valid, long_invalid):
    """Step 6 (fixes E and H): invalid regions take their neighbours' rhythm by an explicit
    rule - both neighbours alike -> that; both 'spec' (VT/SVT/AVB) -> SINUS; one spec -> the
    other side; otherwise the higher of BEAT_PP_PRIORITY; a long invalid stretch -> SINUS."""
    spec, rank = _spec(), _rank()
    K = len(r)
    for lo, hi in runs(r):
        if valid[lo]:
            continue
        if _duration(t, lo, hi) >= long_invalid:
            r[lo:hi] = rc.SINUS
            continue
        pres = int(r[lo - 1]) if lo > 0 else None
        post = int(r[hi]) if hi < K else None
        if pres is None and post is None:
            new = rc.SINUS
        elif pres is None:
            new = post if post not in spec else rc.SINUS
        elif post is None:
            new = pres if pres not in spec else rc.SINUS
        elif pres == post:
            new = pres
        elif pres in spec and post in spec:
            new = rc.SINUS
        elif pres in spec:
            new = post
        elif post in spec:
            new = pres
        else:
            new = max((pres, post), key=lambda c: rank.get(c, -1))
        r[lo:hi] = new
    return r


def _background(r, lo, hi):
    """The non-spec rhythm around a VT/SVT region (left first), SINUS by default."""
    spec = _spec()
    for j in (lo - 1, hi):
        if 0 <= j < len(r) and r[j] not in spec and r[j] != NOISE:
            return int(r[j])
    return rc.SINUS


def _split_by_beats(r, t, sym, criteria, fast_rr):
    """Step 7 (fix D): inside VT, beats that are not V take the surrounding rhythm (not
    SINUS); inside SVT, beats that are neither S nor fast do. The fragments are validated
    again."""
    for name, value in (('VT', V_BEAT), ('SVT', S_BEAT)):
        c = class_index(name)
        for lo, hi in runs(r):
            if r[lo] != c:
                continue
            bg = _background(r, lo, hi)
            for i in range(lo, hi):
                if sym[i] == value:
                    continue
                if name == 'SVT' and i > 0 and t[i] - t[i - 1] <= fast_rr:
                    continue
                r[i] = bg
            for flo, fhi in runs(r[lo:hi]):
                flo, fhi = flo + lo, fhi + lo
                if r[flo] == c and not region_valid(name, t, sym, flo, fhi, None, criteria):
                    r[flo:fhi] = bg
    return r


def _afib_svt_windows(r, t, sym, criteria, svt_ratio):
    """Step 8: the production AFib/SVT arbitration with fixes B and G: a window of
    consecutive AFIB / SVT / short-SINUS regions (nothing else) becomes SVT when SVT time >=
    ratio x AF time, else AFIB."""
    af, svt = class_index('AFIB'), class_index('SVT')
    sinus_min = criteria.get('SINUS', {}).get('duration', 0.0)
    regs = runs(r)

    def member(i):
        lo, hi = regs[i]
        c = r[lo]
        return c in (af, svt) or (c == rc.SINUS and _duration(t, lo, hi) < sinus_min)
    i = 0
    while i < len(regs):
        if not member(i):
            i += 1
            continue
        j = i
        while j + 1 < len(regs) and member(j + 1):
            j += 1
        classes = {int(r[regs[k][0]]) for k in range(i, j + 1)}
        if af in classes and svt in classes:
            dur = {af: 0.0, svt: 0.0}
            for k in range(i, j + 1):
                lo, hi = regs[k]
                if r[lo] in dur:
                    dur[int(r[lo])] += _duration(t, lo, hi)
            new = svt if dur[svt] >= svt_ratio * dur[af] else af
            r[regs[i][0]:regs[j][1]] = new
        i = j + 1
    return r


def beat_symbols(r, t, sym, s_prematurity=0.0, s_in_afib_to_n=True):
    """Step 9: the beat labels written for bxb. S inside AFIB -> N (the reference databases
    do not label supraventricular beats inside AF; the production `S in AFIB -> N`). With
    `s_prematurity` > 0, an S beat outside SVT whose R-R is not shorter than that fraction
    of the median of the preceding 8 intervals is not premature and becomes N - inside SVT
    the beats are expected to lose their prematurity, so they are left alone."""
    out = np.asarray(sym, dtype=int).copy()
    af, svt = class_index('AFIB'), class_index('SVT')
    if s_in_afib_to_n:
        out = np.where((r == af) & (out == S_BEAT), N_BEAT, out)
    if s_prematurity and s_prematurity > 0 and len(t) > 2:
        rr = np.diff(t)
        for i in np.flatnonzero(out == S_BEAT):
            if i < 2 or r[i] == svt:
                continue
            ref = np.median(rr[max(0, i - 9):i - 1]) if i >= 3 else rr[i - 2]
            if ref > 0 and rr[i - 1] >= s_prematurity * ref:
                out[i] = N_BEAT
    return out


def postprocess(track, beats, step_hz, criteria=None, options=None):
    """Step class track (T,) + beats -> (track, beat symbols). See the module docstring."""
    criteria = rc.BEAT_PP_CRITERIA if criteria is None else criteria
    opt = _options(options)
    track = np.asarray(track).copy()
    t = np.asarray(beats['t'], dtype=np.float64)
    sym = np.asarray(beats['cls'], dtype=int)
    if len(t) < 3:
        return track, sym.copy()
    a, b = beat_cells(t, len(track), step_hz)
    r = rhythm_per_beat(track, a, b)

    r = _runs_to_rhythm(r, t, sym, criteria, opt['runs_to_rhythm'])
    r = _extend_along_runs(r, sym)
    valid = _validate(r, t, sym, criteria)
    r, valid = _long_invalid_to_sinus(r, t, valid, opt['long_invalid'])
    r = _merge_invalid(r, t, valid, opt['long_invalid'])
    r = _split_by_beats(r, t, sym, criteria, opt['fast_rr'])
    if opt['afib_svt_merge']:
        r = _afib_svt_windows(r, t, sym, criteria, opt['svt_ratio'])

    out_sym = beat_symbols(r, t, sym, opt['s_prematurity'], opt['s_in_afib_to_n'])
    for i, (lo, hi) in enumerate(zip(a, b)):
        if hi > lo and track[lo:hi].max() != NOISE:       # noise stays noise
            track[lo:hi] = r[i]
    return track, out_sym
